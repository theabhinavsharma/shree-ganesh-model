"""POINT-IN-TIME MARKET CAP — data/derived/mcap_pit.parquet (+ .manifest.json).

mcap_cr(t) = raw-basis price(t) x shares known at t / 1e7
  raw-basis price = adjusted close / price_adjustment_factor_to_present (undoes later splits/bonus
                    so price and share count are on the same, then-current basis)
  shares (primary, 'pnl_implied'): PAT / basic EPS from each quarterly filing, units normalised
                    per pnl_quarterly manifest (detail_api Rs lakh, xbrl Rs), median of the last 4
                    filed quarters (|EPS| >= 0.05, PAT != 0) to damp rounding; known from filing date
                    (merge_asof backward, <= 400 days old).
  fallback ('screener_backcast', LIVE names only): shares_now = screener market_cap_cr / current_price
                    on the latest fetch, applied to the adjusted close (present-share basis). Ignores
                    past dilution (QIP/preferential/merger) — flagged in mcap_source.
Rows: every panel session for every ISIN-master equity (no fund units). NULL = unknown, never guessed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402

OUT = ROOT / "data/derived/mcap_pit.parquet"


def main() -> None:
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
    fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "close", "price_adjustment_factor_to_present"])
    px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])].copy()
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    f = px["price_adjustment_factor_to_present"].replace(0, np.nan).fillna(1.0)
    px["raw_px"] = px["close"] / f

    q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["filing_dt", "pat", "eps_basic"])
    q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
    to_rs = np.where(q["source"] == "xbrl", 1.0, 1e5)                         # manifest: detail_api = Rs lakh
    ok = (q["eps_basic"].abs() >= 0.05) & (q["pat"] != 0)
    q["sh"] = np.where(ok, q["pat"] * to_rs / q["eps_basic"], np.nan)
    q.loc[q["sh"] <= 0, "sh"] = np.nan
    q["shares"] = q.groupby("symbol")["sh"].transform(lambda s: s.rolling(4, min_periods=1).median())
    q["known"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
    qq = q.dropna(subset=["shares"])[["symbol", "known", "shares"]].sort_values("known").rename(columns={"known": "trade_date"})
    px = pd.merge_asof(px.sort_values("trade_date"), qq, on="trade_date", by="symbol",
                       direction="backward", tolerance=pd.Timedelta(days=400))
    px["mcap_cr"] = px["raw_px"] * px["shares"] / 1e7
    px["mcap_source"] = np.where(px["mcap_cr"].notna(), "pnl_implied", None)

    # fallback: live names without P&L history — screener shares_now on the present-share basis
    scr = pd.read_parquet(ROOT / "data/derived/screener_fundamentals.parquet", columns=["symbol", "fetch_date", "market_cap_cr", "current_price"])
    scr = scr.dropna().sort_values("fetch_date").groupby("symbol").tail(1)
    scr = scr[scr["current_price"] > 0]
    sh_now = (scr.set_index("symbol")["market_cap_cr"] * 1e7 / scr.set_index("symbol")["current_price"])
    miss = px["mcap_cr"].isna() & px["symbol"].isin(sh_now.index)
    px.loc[miss, "mcap_cr"] = px.loc[miss, "close"] * px.loc[miss, "symbol"].map(sh_now) / 1e7
    px.loc[miss, "mcap_source"] = "screener_backcast"

    # second fallback: screener_mcap_backfill (symbols the first two cannot size, incl. delisted)
    bf_p = ROOT / "data/derived/screener_mcap_backfill.parquet"
    if bf_p.exists():
        bf = pd.read_parquet(bf_p)
        bf = bf[(bf["status"] == "OK") & (bf["current_price"] > 0)]
        sh_bf = bf.set_index("symbol")["market_cap_cr"] * 1e7 / bf.set_index("symbol")["current_price"]
        miss = px["mcap_cr"].isna() & px["symbol"].isin(sh_bf.index)
        px.loc[miss, "mcap_cr"] = px.loc[miss, "close"] * px.loc[miss, "symbol"].map(sh_bf) / 1e7
        px.loc[miss, "mcap_source"] = "screener_backfill"
    out = px[["symbol", "trade_date", "mcap_cr", "mcap_source"]]
    out.to_parquet(OUT, index=False)

    # validation vs screener's own current market cap
    last = px.sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
    v = last[last["mcap_source"] == "pnl_implied"].join(scr.set_index("symbol")[["market_cap_cr", "fetch_date"]], how="inner")
    v = v[(pd.to_datetime(v["fetch_date"]) - v["trade_date"]).abs() < pd.Timedelta(days=45)]
    r = (v["mcap_cr"] / v["market_cap_cr"])
    val = dict(n=int(len(r)), median_ratio=round(float(r.median()), 3), within_20pct=round(float(((r > .8) & (r < 1.2)).mean()), 3),
               within_50pct=round(float(((r > .5) & (r < 1.5)).mean()), 3))
    cov = out.dropna(subset=["mcap_cr"])
    stats = dict(rows=len(out), rows_with_mcap=len(cov), symbols=int(out["symbol"].nunique()), symbols_with_mcap=int(cov["symbol"].nunique()),
                 by_source=cov["mcap_source"].value_counts().to_dict(), validation_vs_screener_latest=val,
                 date_range=[str(out["trade_date"].min().date()), str(out["trade_date"].max().date())])
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="mcap_pit", path=str(OUT.relative_to(ROOT)), key=["symbol", "trade_date"], producer="src/agentic/build_mcap_pit.py",
        columns=dict(mcap_cr="market capitalisation, Rs CRORE, point-in-time (NULL = unknown)",
                     mcap_source="pnl_implied (PAT/EPS shares known at t) | screener_backcast / screener_backfill (present or last-trade shares x adjusted close; ignores past dilution)"),
        stats=stats, updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
