"""6TH ENGINE — Quiet Accumulation Scanner.

Built to capture the 52 winners that current 5 engines missed since April.
Pattern learned from `logs/miss_learnings.jsonl` (historical coverage backtest):

  Miss signature (avg of 52 missed winners):
    RSI 57.1        - moderate, NOT chase or oversold
    20d return +6.9% - mild positive momentum
    5d return  +2.1% - slight bullish drift
    Volume ratio 0.72x - BELOW normal (quiet)
    Above 200-DMA 62% - mostly in uptrend
    Distance from SMA200 +12% - moderately above

Scoring:
  quiet_score = product of soft "in-band" indicators, then rank across universe.
  Higher score = closer match to the historical winner profile.

This is a WORKING RL PROTOTYPE - not calibrated with a proper OOS backtest yet.
Ships alongside the 5 existing engines; consumed by the cross-engine consensus.

Per CONSTITUTION §1.4: any feature promoted to model requires A/B lift proof.
This engine is INITIAL SCREEN only until backtested.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/quiet_accumulation_predictions.parquet"


def _in_band_gauss(v: pd.Series, target: float, sigma: float) -> pd.Series:
    """Soft gaussian membership near target. Peaks at target=1.0."""
    return np.exp(-((v - target) ** 2) / (2 * sigma ** 2))


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Given a snapshot df, compute quiet_accumulation_score per symbol.

    Snapshot must include: rsi_14_daily, return_20d, ret_5d, volume_vs_20d,
    close, sma_50, sma_200, delivery_pct.
    """
    d = df.copy()
    d["dist_sma200"] = d["close"] / d["sma_200"] - 1
    d["above_50dma"] = (d["close"] > d["sma_50"]).astype(int)
    d["above_200dma"] = (d["close"] > d["sma_200"]).astype(int)

    # Soft gaussian components centred on the historical miss profile
    d["s_rsi"]    = _in_band_gauss(d["rsi_14_daily"].fillna(50), target=57.0, sigma=7.0)
    d["s_20d"]    = _in_band_gauss(d["return_20d"].fillna(0)*100, target=7.0, sigma=6.0)
    d["s_5d"]     = _in_band_gauss(d["ret_5d"].fillna(0)*100, target=2.0, sigma=3.0)
    d["s_vol"]    = _in_band_gauss(d["volume_vs_20d"].fillna(1), target=0.72, sigma=0.25)
    d["s_ma200"]  = _in_band_gauss(d["dist_sma200"].fillna(0)*100, target=12.0, sigma=10.0)

    # Trend bonus: prefer above 50-DMA (short trend intact)
    d["s_trend"] = np.where(d["above_50dma"]==1, 1.0, 0.4)

    # Aggregate score — product-like, then normalize to [0,1]
    d["quiet_score"] = (d["s_rsi"] * d["s_20d"] * d["s_5d"] * d["s_vol"] * d["s_ma200"] * d["s_trend"])
    # Rescale to have max ≈ 1 within this run
    max_s = d["quiet_score"].max()
    if max_s > 0:
        d["quiet_score"] = d["quiet_score"] / max_s
    return d


def main():
    print("== find_quiet_accumulation ==")
    prices = pd.read_parquet(PRICES, columns=[
        "symbol","trade_date","close","return_1d","rsi_14_daily",
        "return_20d","total_traded_value","volume_vs_20d","delivery_pct",
        "sma_50","sma_200"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["symbol","trade_date"])
    prices["ret_5d"] = prices.groupby("symbol")["close"].pct_change(5)
    prices["adv_cr"] = prices.groupby("symbol")["total_traded_value"].transform(
        lambda s: s.rolling(20).mean()) / 1e7
    latest = prices["trade_date"].max()
    snap = prices[prices["trade_date"] == latest].copy()
    print(f"  data as of {latest.date()}, universe {len(snap):,}")

    # Apply liquidity + basic sanity gate
    snap = snap[(snap["adv_cr"] >= 5) & (snap["close"] > 50)]
    print(f"  after ADV>=5cr + price>50: {len(snap):,}")

    scored = compute_scores(snap)
    scored = scored.sort_values("quiet_score", ascending=False)

    top = scored.head(30)
    print(f"\n=== Top 30 by Quiet Accumulation score ===")
    show = ["symbol","close","quiet_score","rsi_14_daily","return_20d","ret_5d","volume_vs_20d","adv_cr","dist_sma200"]
    print(top[show].to_string(index=False))

    # Save
    OUT.parent.mkdir(parents=True, exist_ok=True)
    scored[["symbol","close","quiet_score","s_rsi","s_20d","s_5d","s_vol","s_ma200","s_trend",
             "rsi_14_daily","return_20d","ret_5d","volume_vs_20d","adv_cr","dist_sma200"]].to_parquet(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}  (shape={scored.shape})")


if __name__ == "__main__":
    main()
