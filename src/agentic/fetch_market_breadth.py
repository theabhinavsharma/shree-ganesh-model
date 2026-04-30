"""Compute MARKET BREADTH features from existing prices parquet.

Pure derivation — no external fetch. Computes daily aggregate signals:
  • breadth_50:           % of liquid stocks above 50-DMA
  • breadth_200:          % above 200-DMA
  • new_52w_highs:        count of stocks at 52w high today
  • new_52w_lows:         count of stocks at 52w low today
  • adv_decl_ratio:       advancing/declining ratio
  • mcap_breadth_lcap:    % large-caps (top 100 by ADV) above 50-DMA
  • mcap_breadth_smid:    % small/mid (rank 100-500) above 50-DMA
  • smid_lcap_breadth_diff: small/mid breadth - largecap breadth (rotation signal)
  • cross_section_dispersion: std of return_20d across liquid universe (regime)
  • median_realized_vol:  median 20d realized vol (volatility regime)
  • upside_skew_count:    # stocks with return_20d > 25%
  • downside_skew_count:  # stocks with return_20d < -15%

Why: these capture the WHOLE-MARKET state (not per-stock), which is what
drives multibagger basket success per the regime gate analysis.

Output: data/derived/market_breadth_panel.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/market_breadth_panel.parquet"


def main() -> None:
    print("== fetch_market_breadth (derived from prices) ==")
    cols_wanted = ["symbol", "trade_date", "close", "high", "low",
            "sma_50", "sma_200", "return_1d", "return_20d",
            "realized_vol_20d", "avg_traded_value_20d", "series"]
    # only request cols that actually exist
    import pyarrow.parquet as pq
    schema_cols = {f.name for f in pq.read_schema(PRICES)}
    cols = [c for c in cols_wanted if c in schema_cols]
    df = pd.read_parquet(PRICES, columns=cols)
    if "realized_vol_20d" not in df.columns:
        # derive from return_1d if missing
        df = df.sort_values(["symbol", "trade_date"])
        df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["series"] == "EQ"]
    # liquid universe per day: ADV ≥ ₹1cr/day
    df = df[df["avg_traded_value_20d"] / 1e7 >= 1.0]

    # mcap proxy: ADV ranking on each day
    df["adv_rank"] = df.groupby("trade_date")["avg_traded_value_20d"].rank(
        ascending=False, method="min")
    df["is_lcap"] = (df["adv_rank"] <= 100).astype(int)
    df["is_smid"] = ((df["adv_rank"] > 100) & (df["adv_rank"] <= 500)).astype(int)

    # 52w high / low flags computed per symbol
    df = df.sort_values(["symbol", "trade_date"])
    df["52w_high"] = df.groupby("symbol")["high"].transform(
        lambda x: x.rolling(252, min_periods=126).max())
    df["52w_low"] = df.groupby("symbol")["low"].transform(
        lambda x: x.rolling(252, min_periods=126).min())
    df["at_52w_high"] = (df["close"] >= df["52w_high"] * 0.99).astype(int)
    df["at_52w_low"]  = (df["close"] <= df["52w_low"] * 1.01).astype(int)
    df["above_50dma"]  = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["adv"] = (df["return_1d"] > 0).astype(int)
    df["dec"] = (df["return_1d"] < 0).astype(int)

    # daily aggregates
    daily = df.groupby("trade_date").agg(
        n_universe          = ("symbol", "count"),
        breadth_50          = ("above_50dma", "mean"),
        breadth_200         = ("above_200dma", "mean"),
        new_52w_highs       = ("at_52w_high", "sum"),
        new_52w_lows        = ("at_52w_low", "sum"),
        advancing           = ("adv", "sum"),
        declining           = ("dec", "sum"),
        median_return_1d    = ("return_1d", "median"),
        median_return_20d   = ("return_20d", "median"),
        cross_section_dispersion_20d = ("return_20d", "std"),
        median_realized_vol_20d = ("realized_vol_20d", "median"),
    ).reset_index()
    daily["adv_decl_ratio"] = daily["advancing"] / daily["declining"].replace(0, np.nan)

    # large-cap vs small/mid breadth
    lcap = df[df["is_lcap"] == 1].groupby("trade_date").agg(
        breadth_50_lcap=("above_50dma", "mean"),
    ).reset_index()
    smid = df[df["is_smid"] == 1].groupby("trade_date").agg(
        breadth_50_smid=("above_50dma", "mean"),
    ).reset_index()
    daily = daily.merge(lcap, on="trade_date", how="left").merge(smid, on="trade_date", how="left")
    daily["smid_lcap_breadth_diff"] = daily["breadth_50_smid"] - daily["breadth_50_lcap"]

    # extreme moves: # of stocks with 20d return > 25% or < -15%
    upside = df.groupby("trade_date").apply(
        lambda g: int((g["return_20d"] > 0.25).sum())).rename("upside_skew_count").reset_index()
    downside = df.groupby("trade_date").apply(
        lambda g: int((g["return_20d"] < -0.15).sum())).rename("downside_skew_count").reset_index()
    daily = daily.merge(upside, on="trade_date", how="left").merge(downside, on="trade_date", how="left")

    # rolling-20d sum proxies (matches regime-gate v1 inputs)
    daily["market_20d_sum"] = daily["median_return_1d"].rolling(20).sum()
    daily["breadth_50_5d_chg"] = daily["breadth_50"].diff(5)
    daily["breadth_50_20d_chg"] = daily["breadth_50"].diff(20)
    daily["new_high_low_diff"] = daily["new_52w_highs"] - daily["new_52w_lows"]
    daily["dispersion_z_60d"] = (
        (daily["cross_section_dispersion_20d"] - daily["cross_section_dispersion_20d"].rolling(60).mean())
        / daily["cross_section_dispersion_20d"].rolling(60).std()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(daily)} days × {len(daily.columns)-1} breadth metrics")

    last = daily.iloc[-1]
    print(f"\n  Latest ({last['trade_date']:%Y-%m-%d}):")
    print(f"    breadth_50          = {last['breadth_50']*100:.1f}%")
    print(f"    breadth_200         = {last['breadth_200']*100:.1f}%")
    print(f"    new_52w_highs       = {int(last['new_52w_highs'])}")
    print(f"    new_52w_lows        = {int(last['new_52w_lows'])}")
    print(f"    adv_decl_ratio      = {last['adv_decl_ratio']:.2f}")
    print(f"    smid_lcap_diff      = {(last['smid_lcap_breadth_diff'] or 0)*100:+.1f}%")
    print(f"    market_20d_sum      = {(last['market_20d_sum'] or 0)*100:+.2f}%")
    print(f"    dispersion_z_60d    = {(last['dispersion_z_60d'] or 0):+.2f}")


if __name__ == "__main__":
    main()
