"""Domestic gold price (INR) from NSE-listed gold ETFs — data/derived/gold_inr_etf.parquet.

Why this source: the only gold series in the macro layer was Yahoo GC=F (USD), which has
been throttled and never landed ("No data fetched (Yahoo throttled or down)" in every
daily_data_layer global_macro log); gold_ppi is a monthly FRED producer-price index, not a
price. Gold ETFs trade on NSE, so their closes already arrive through the official NSE
bhavcopy path (stock_daily_facts_adjusted_2015plus.parquet) — no new or unofficial source.
Each ETF's NAV tracks domestic spot gold (LBMA x USDINR + duty), which is what Indian
jewellers' inventory and pricing actually follow.

Method: per day, median across the ETFs that traded of their close-to-close return
(adjusted close; |r| > 15% dropped as a data artifact — gold does not move 15% in a day),
requiring >= 3 ETFs; chained into an index (first valid day = 100). Median across funds
damps any single ETF's premium/discount-to-NAV swing (the MASPTOP50 failure mode).
UNITS: gold_inr_idx is an index level, NOT rupees per gram. ret columns are fractions.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/gold_inr_etf.parquet"
# Physical-gold ETFs only (renames chained implicitly: each is just another member).
# Excludes gold-named operating companies (SKYGOLD, SHANTIGOLD, GOLDIAM, GOLDTECH, ...).
GOLD_ETFS = ["GOLDBEES", "SETFGOLD", "BSLGOLDETF", "QGOLDHALF", "IVZINGOLD", "AXISGOLD", "GOLDAXIS",
             "KOTAKGOLD", "GOLD1", "GOLDSHARE", "IDBIGOLD", "RELGOLD", "ICICIGOLD", "GOLDIETF",
             "HDFCGOLD", "TATAGOLD", "GOLDCASE", "LICMFGOLD", "EGOLD", "GROWWGOLD", "BBNPPGOLD",
             "UNIONGOLD", "MOGOLD", "HSBCGOLD", "GOLDETF"]
MIN_ETFS, MAX_ABS_RET = 3, 0.15


def main() -> None:
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close"], filters=[("symbol", "in", GOLD_ETFS)])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    px["r"] = px.groupby("symbol")["close"].pct_change()
    px = px[px["r"].abs() <= MAX_ABS_RET]
    d = px.groupby("trade_date")["r"].agg(gold_inr_ret1d="median", n_etfs="size",
                                          disp_iqr=lambda s: s.quantile(.75) - s.quantile(.25))
    d.loc[d["n_etfs"] < MIN_ETFS, "gold_inr_ret1d"] = np.nan
    d["gold_inr_idx"] = 100 * (1 + d["gold_inr_ret1d"].fillna(0)).cumprod()
    for k in (5, 20, 60):
        d[f"gold_inr_{k}d_pct"] = d["gold_inr_idx"].pct_change(k)
    d = d.reset_index()
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="gold_inr_etf", path=str(OUT.relative_to(ROOT)), rows=len(d),
        date_range=[str(d["trade_date"].min().date()), str(d["trade_date"].max().date())],
        producer="src/agentic/build_gold_feed.py", source="NSE bhavcopy closes of physical-gold ETFs via the adjusted price panel",
        members=GOLD_ETFS,
        columns=dict(trade_date="NSE session", gold_inr_ret1d="median close-to-close return across traded gold ETFs (fraction; NaN if < 3 ETFs)",
                     n_etfs="ETFs contributing that day", disp_iqr="IQR of member returns (fraction) — high = premium/discount noise",
                     gold_inr_idx="chained index, first day = 100; NOT INR/gram",
                     gold_inr_5d_pct="5-session change of the index (fraction)", gold_inr_20d_pct="20-session change (fraction)",
                     gold_inr_60d_pct="60-session change (fraction)"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    last = d.dropna(subset=["gold_inr_ret1d"]).iloc[-1]
    print(f"gold_inr_etf: {len(d):,} sessions {d['trade_date'].min().date()}..{d['trade_date'].max().date()} · "
          f"last {last['trade_date'].date()} n={int(last['n_etfs'])} · 20d {last['gold_inr_20d_pct']*100:+.1f}% · "
          f"60d {last['gold_inr_60d_pct']*100:+.1f}%")


if __name__ == "__main__":
    main()
