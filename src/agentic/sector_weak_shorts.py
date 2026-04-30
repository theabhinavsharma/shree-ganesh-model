"""Sector-weak large-cap short overlay.

The short ML model trains on stock-level features (-5%/7d label) which rarely
fires on Nifty IT/Bank large-caps. So when sectors like IT crash -7%/5d, the
ML short model misses the obvious large-cap shorts (INFY, TCS, HDFCBANK).

This overlay ranks shorts by macro→sector→stock chain:
  1. Find sectors with 5d return <= -1% (or worst-N)
  2. Within each weak sector, pick top-N names by ADV (most liquid → futures-tradable)
  3. Filter to names that are above their 200-DMA (mean reversion candidate)
  4. Optional: stack with short ML score if available

Output: tmp/from_scratch_7d_run/sector_weak_shorts.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
SECT_MEMBERS = ROOT / "tmp/from_scratch_7d_run/alt2/sector_index_members.parquet"
SHORT_LIVE = ROOT / "tmp/from_scratch_7d_run/short_live_top100.csv"
OUT = ROOT / "tmp/from_scratch_7d_run/sector_weak_shorts.csv"

# Major sector universes (priority order so we use the most-specific membership)
SECT_PRIORITY = ["NIFTY IT", "NIFTY BANK", "NIFTY AUTO", "NIFTY METAL", "NIFTY PHARMA",
                 "NIFTY FMCG", "NIFTY REALTY", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSE",
                 "NIFTY PVT BANK", "NIFTY FINANCIAL SERVICES", "NIFTY CONSUMER DURABLES",
                 "NIFTY OIL & GAS", "NIFTY INFRA"]

WEAK_SECTOR_THRESHOLD = -0.01  # sectors with 5d <= -1% qualify
TOP_N_PER_SECTOR = 5
MIN_ADV_CR = 5.0  # require ≥ ₹5cr/day so futures liquidity is real


def main() -> None:
    sm = pd.read_parquet(SECT_MEMBERS)
    sm["pri"] = sm["index_name"].map({n: i for i, n in enumerate(SECT_PRIORITY)}).fillna(99)
    sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol", "index_name"]].rename(
        columns={"index_name": "sector"})

    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "sma_200",
                                           "return_1d", "return_20d", "rsi_14_daily",
                                           "avg_traded_value_20d", "series", "sma_20", "sma_50"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    snap = px[(px["trade_date"] == latest) & (px["series"] == "EQ")].copy()
    snap["adv_cr"] = snap["avg_traded_value_20d"] / 1e7
    snap = snap.merge(sec_map, on="symbol", how="left")
    snap = snap[snap["sector"].isin(SECT_PRIORITY)]
    snap = snap[snap["adv_cr"] >= MIN_ADV_CR]

    # 5-day sector return = sum of median 1-day returns per sector
    px_with_sec = px.merge(sec_map, on="symbol", how="left")
    px_with_sec = px_with_sec[px_with_sec["sector"].isin(SECT_PRIORITY)]
    sec_d = px_with_sec.groupby(["trade_date", "sector"])["return_1d"].median().reset_index()
    sec_d = sec_d.sort_values(["sector", "trade_date"])
    sec_d["s_5d"] = sec_d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(5).sum())
    sec_5d_latest = sec_d[sec_d["trade_date"] == latest][["sector", "s_5d"]]

    weak = sec_5d_latest[sec_5d_latest["s_5d"] <= WEAK_SECTOR_THRESHOLD].sort_values("s_5d")
    print(f"Weak sectors (5d <= {WEAK_SECTOR_THRESHOLD*100:.0f}%):")
    print(weak.to_string(index=False))
    print()

    if weak.empty:
        print("No weak sectors today — no overlay shorts.")
        OUT.write_text("symbol,sector,close,sector_5d_ret,return_20d,rsi_14_daily,adv_cr,short_score_cal,reason\n")
        return

    # short ML scores (optional stack)
    short_scores: dict[str, float] = {}
    if SHORT_LIVE.exists():
        ss = pd.read_csv(SHORT_LIVE)
        for _, r in ss.iterrows():
            short_scores[str(r["symbol"])] = float(r.get("score_calibrated", 0))

    rows = []
    for _, srow in weak.iterrows():
        sec = srow["sector"]
        s_5d = srow["s_5d"]
        # within sector, pick names that are over-extended OR matching ML short signal
        in_sec = snap[snap["sector"] == sec].copy()
        # sector-weak shorts: still above 200-DMA (so there's room to fall) AND high RSI (extended)
        in_sec["above_200"] = in_sec["close"] > in_sec["sma_200"]
        in_sec["mean_revert_dist"] = in_sec["close"] / in_sec["sma_50"] - 1
        # rank: highest mean-revert distance × sector weakness
        in_sec["score"] = in_sec["mean_revert_dist"].fillna(0) * abs(s_5d)
        in_sec = in_sec.sort_values("score", ascending=False).head(TOP_N_PER_SECTOR)

        for _, r in in_sec.iterrows():
            sym = str(r["symbol"])
            ml_short = short_scores.get(sym)
            reason_bits = [f"sector {sec} 5d={s_5d*100:+.1f}%",
                           f"close {r['close']/r['sma_50']-1:+.1%} above sma50"]
            if r.get("rsi_14_daily", 50) > 70:
                reason_bits.append(f"RSI {r['rsi_14_daily']:.0f} extended")
            if ml_short is not None and ml_short > 0.5:
                reason_bits.append(f"ML short_cal {ml_short:.2f}")
            rows.append({
                "symbol": sym,
                "sector": sec,
                "close": round(float(r["close"]), 2),
                "sector_5d_ret_pct": round(float(s_5d) * 100, 2),
                "return_20d_pct": round(float(r["return_20d"]) * 100, 2) if pd.notna(r["return_20d"]) else None,
                "rsi_14_daily": round(float(r["rsi_14_daily"]), 1) if pd.notna(r["rsi_14_daily"]) else None,
                "adv_cr": round(float(r["adv_cr"]), 1),
                "short_score_cal": round(ml_short, 3) if ml_short is not None else None,
                "reason": " | ".join(reason_bits),
            })

    out_df = pd.DataFrame(rows).sort_values(["sector_5d_ret_pct", "return_20d_pct"],
                                             ascending=[True, False])
    out_df.to_csv(OUT, index=False)
    print(f"\nSector-weak short candidates ({len(out_df)}):")
    print(out_df.head(20).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
