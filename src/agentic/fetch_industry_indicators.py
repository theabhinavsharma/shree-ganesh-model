"""Compute INDUSTRY/SECTOR aggregate indicators from existing prices parquet.

Pure derivation. For each sector:
  • sector_breadth_50:    % of names in sector above 50-DMA
  • sector_5d_ret:        equal-weighted 5d return (sector momentum)
  • sector_20d_ret:       20d return
  • sector_60d_ret:       60d return (medium-term rotation)
  • sector_dispersion:    std of 20d returns within sector
  • sector_leader_lag:    top-quartile - bottom-quartile 20d return spread
  • sector_volume_z:      sector aggregate volume z-score (60d)
  • sector_relative_strength: sector vs NIFTY (median across all)

Why: institutional rotation is the dominant driver of multi-month moves.
Sectors that are setting up (rising breadth + low dispersion) tend to deliver
sustained leadership. We feed these as MACRO-LEVEL features (no per-stock micros).

Output: data/derived/industry_panel.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/industry_panel.parquet"


def main() -> None:
    print("== fetch_industry_indicators (derived) ==")
    cols_needed = ["symbol", "trade_date", "close", "sma_50",
                   "return_1d", "return_5d", "return_20d", "return_60d",
                   "total_traded_qty", "avg_vol_20d",
                   "avg_traded_value_20d", "series"]
    import pyarrow.parquet as pq
    schema_cols = {f.name for f in pq.read_schema(PRICES)}
    use = [c for c in cols_needed if c in schema_cols]
    df = pd.read_parquet(PRICES, columns=use)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if "series" in df.columns:
        df = df[df["series"] == "EQ"]
    if "avg_traded_value_20d" in df.columns:
        df = df[df["avg_traded_value_20d"] / 1e7 >= 1.0]

    # join sector mapping from paper_trading_ledger or confluence_picks
    sec_map = None
    for src in [ROOT/"data/derived/paper_trading_ledger.parquet",
                ROOT/"data/derived/confluence_picks.parquet"]:
        if src.exists():
            try:
                m = pd.read_parquet(src, columns=["symbol", "sector"]).drop_duplicates("symbol")
                m = m[m["sector"].notna() & (m["sector"] != "")]
                sec_map = m if sec_map is None else pd.concat([sec_map, m]).drop_duplicates("symbol")
            except Exception:
                pass
    if sec_map is None or sec_map.empty:
        print("  no sector mapping available — skipping")
        return
    print(f"  sector mapping: {len(sec_map)} symbols")
    df = df.merge(sec_map, on="symbol", how="left")
    df = df[df["sector"].notna()]
    if df.empty:
        print("  no rows with sector after join")
        return

    # Compute return_5d / return_60d if missing
    if "return_5d" not in df.columns:
        df = df.sort_values(["symbol", "trade_date"])
        df["return_5d"] = df.groupby("symbol")["close"].pct_change(5)
    if "return_60d" not in df.columns:
        df = df.sort_values(["symbol", "trade_date"])
        df["return_60d"] = df.groupby("symbol")["close"].pct_change(60)

    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int) if "sma_50" in df.columns else np.nan

    # NIFTY proxy = median return across liquid universe
    nifty = df.groupby("trade_date").agg(
        nifty_5d=("return_5d", "median"),
        nifty_20d=("return_20d", "median"),
        nifty_60d=("return_60d", "median"),
    ).reset_index()

    # per-sector daily aggregates
    sec = df.groupby(["trade_date", "sector"]).agg(
        n_members            = ("symbol", "count"),
        sector_breadth_50    = ("above_50dma", "mean"),
        sector_5d_ret        = ("return_5d", "mean"),
        sector_20d_ret       = ("return_20d", "mean"),
        sector_60d_ret       = ("return_60d", "mean"),
        sector_dispersion_20d= ("return_20d", "std"),
        sector_volume_sum    = ("total_traded_qty", "sum"),
    ).reset_index()
    sec["sector_volume_sum"] = sec["sector_volume_sum"].fillna(0)

    # leader-lagger spread per sector per day (q75 - q25 of 20d ret)
    def _spread(g):
        if len(g) < 4:
            return np.nan
        return g["return_20d"].quantile(0.75) - g["return_20d"].quantile(0.25)
    spread = df.groupby(["trade_date", "sector"]).apply(_spread).rename("sector_leader_lag_spread").reset_index()
    sec = sec.merge(spread, on=["trade_date", "sector"], how="left")

    # relative strength vs NIFTY
    sec = sec.merge(nifty, on="trade_date", how="left")
    sec["rs_5d"]  = sec["sector_5d_ret"]  - sec["nifty_5d"]
    sec["rs_20d"] = sec["sector_20d_ret"] - sec["nifty_20d"]
    sec["rs_60d"] = sec["sector_60d_ret"] - sec["nifty_60d"]

    # sector volume z-score (60d) per sector
    sec = sec.sort_values(["sector", "trade_date"])
    sec["sector_volume_z_60d"] = sec.groupby("sector")["sector_volume_sum"].transform(
        lambda x: (x - x.rolling(60).mean()) / x.rolling(60).std()
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sec.to_parquet(OUT, index=False)
    n_sectors = sec["sector"].nunique()
    n_days = sec["trade_date"].nunique()
    print(f"wrote {OUT}: {len(sec):,} rows  {n_sectors} sectors × {n_days} days")

    # quick today snapshot
    latest_day = sec["trade_date"].max()
    today = sec[sec["trade_date"] == latest_day].sort_values("rs_60d", ascending=False)
    print(f"\n  Top 5 sectors by 60d RS ({latest_day:%Y-%m-%d}):")
    for _, r in today.head(5).iterrows():
        print(f"    {r['sector']:<28}  rs_60d={r['rs_60d']*100:+.2f}%  "
              f"breadth_50={(r['sector_breadth_50'] or 0)*100:.0f}%  "
              f"dispersion={(r['sector_dispersion_20d'] or 0)*100:.1f}%")
    print(f"\n  Bottom 5 sectors by 60d RS:")
    for _, r in today.tail(5).iterrows():
        print(f"    {r['sector']:<28}  rs_60d={r['rs_60d']*100:+.2f}%  "
              f"breadth_50={(r['sector_breadth_50'] or 0)*100:.0f}%  "
              f"dispersion={(r['sector_dispersion_20d'] or 0)*100:.1f}%")


if __name__ == "__main__":
    main()
