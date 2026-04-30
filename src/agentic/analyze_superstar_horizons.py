"""Multi-horizon superstar-confluence analysis.

Question: at what holding horizon (7 / 15 / 30 / 60 / 90 / 180 / 252 trading days)
do superstar-held stocks deliver disproportionate returns vs the baseline?

Method:
  1. For each (symbol, trade_date) in 2024-2025 OOS where the stock is liquid (ADV >= 1cr)
  2. Compute forward close-to-close return at each horizon
  3. Bucket by current superstar confluence (0 / 1 / 2-10 / 3+)
  4. Per bucket × horizon: mean, median, % >= 20%, % >= 30%, % >= 50%

Output:
  data/derived/superstar_horizon_analysis.parquet
  reports/superstar_horizons.md (rendered table)

Caveat: uses TODAY's confluence list against historical OOS prices. To do
this perfectly we'd need quarterly historical holdings — filed as TODO.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
HOLDINGS = ROOT / "data/derived/superstar_holdings.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_PARQUET = ROOT / "data/derived/superstar_horizon_analysis.parquet"
OUT_REPORT = ROOT / "reports/superstar_horizons.md"

HORIZONS = [7, 15, 30, 60, 90, 180, 252]


def main() -> None:
    print("== analyze_superstar_horizons ==")
    if not HOLDINGS.exists():
        print(f"missing {HOLDINGS}")
        return
    h = pd.read_parquet(HOLDINGS)
    h["fetch_date"] = pd.to_datetime(h["fetch_date"]).dt.date
    h = h[h["fetch_date"] == h["fetch_date"].max()]
    confluence = (h.groupby("symbol")
                    .agg(n_superstars=("investor_tag", "nunique"))
                    .reset_index())

    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close",
                                           "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    px["adv_cr"] = px["avg_traded_value_20d"] / 1e7
    print(f"  loaded {len(px):,} (symbol, date) rows")

    # forward returns at each horizon
    for h_days in HORIZONS:
        px[f"close_fwd_{h_days}"] = px.groupby("symbol")["close"].shift(-h_days)
        px[f"ret_{h_days}"] = px[f"close_fwd_{h_days}"] / px["close"] - 1

    # OOS window 2024-2025 + liquidity
    # use 2024 only for 252-day horizon to ensure we have 1y of forward data
    ret_cols = [f"ret_{h}" for h in HORIZONS]
    oos = px[(px["trade_date"] >= "2024-01-01") &
             (px["trade_date"] <= "2025-04-01") &  # leave room for 252-day forward
             (px["adv_cr"] >= 1.0)]
    oos = oos.merge(confluence, on="symbol", how="left")
    oos["n_superstars"] = oos["n_superstars"].fillna(0).astype(int)
    oos["bucket"] = pd.cut(oos["n_superstars"],
                            bins=[-1, 0, 1, 10, 999],
                            labels=["0_none", "1_solo", "2_to_10", "11plus"])

    print(f"  OOS rows: {len(oos):,}")
    print(f"  bucket sizes:")
    print(oos["bucket"].value_counts().to_string())

    rows = []
    for h_days in HORIZONS:
        col = f"ret_{h_days}"
        sub = oos.dropna(subset=[col])
        for bucket, grp in sub.groupby("bucket", observed=False):
            n = len(grp)
            if n < 100:
                continue
            r = {
                "horizon_days": h_days,
                "bucket": str(bucket),
                "n_stockdays": n,
                "mean": float(grp[col].mean()),
                "median": float(grp[col].median()),
                "pct_pos": float((grp[col] > 0).mean()),
                "pct_5pct": float((grp[col] >= 0.05).mean()),
                "pct_10pct": float((grp[col] >= 0.10).mean()),
                "pct_20pct": float((grp[col] >= 0.20).mean()),
                "pct_30pct": float((grp[col] >= 0.30).mean()),
                "pct_50pct": float((grp[col] >= 0.50).mean()),
            }
            rows.append(r)

    res = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    # render report
    md = ["# Superstar-confluence × multiple horizons", "",
          "Question: at what holding horizon (7 / 15 / 30 / 60 / 90 / 180 / 252 days) do",
          "stocks held by 2+ celebrity Indian investors deliver disproportionate returns?", "",
          "## Per-bucket forward returns by horizon", ""]

    for h_days in HORIZONS:
        sub = res[res["horizon_days"] == h_days]
        if sub.empty:
            continue
        md.append(f"### Horizon: {h_days} trading days (~{h_days/21:.1f} months)")
        md.append("")
        md.append("| Bucket | n | Mean | Median | % positive | % >= +10% | % >= +20% | % >= +30% | % >= +50% |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            md.append(f"| {r['bucket']} | {int(r['n_stockdays']):,} | "
                      f"{r['mean']*100:+.2f}% | {r['median']*100:+.2f}% | "
                      f"{r['pct_pos']*100:.1f}% | {r['pct_10pct']*100:.1f}% | "
                      f"{r['pct_20pct']*100:.1f}% | {r['pct_30pct']*100:.1f}% | "
                      f"{r['pct_50pct']*100:.1f}% |")
        md.append("")
        # delta row
        zero = sub[sub["bucket"] == "0_none"].iloc[0] if len(sub[sub["bucket"] == "0_none"]) else None
        two = sub[sub["bucket"] == "2_to_10"].iloc[0] if len(sub[sub["bucket"] == "2_to_10"]) else None
        if zero is not None and two is not None:
            md.append(f"_Delta (2_to_10 vs none):_  mean **{(two['mean']-zero['mean'])*100:+.2f}pp**, "
                      f"median {(two['median']-zero['median'])*100:+.2f}pp, "
                      f"% >= +20% {(two['pct_20pct']-zero['pct_20pct'])*100:+.1f}pp, "
                      f"% >= +30% {(two['pct_30pct']-zero['pct_30pct'])*100:+.1f}pp")
            md.append("")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PARQUET}")

    # console summary — pivot to compact table: bucket "2_to_10" across horizons
    print("\n=== HEADLINE: 2_to_10 superstar bucket across horizons ===")
    pivot_cols = ["horizon_days", "bucket", "n_stockdays", "mean", "median",
                  "pct_pos", "pct_20pct", "pct_30pct", "pct_50pct"]
    print(res[res["bucket"] == "2_to_10"][pivot_cols].round(4).to_string(index=False))
    print("\n=== BASELINE: 0_none across horizons ===")
    print(res[res["bucket"] == "0_none"][pivot_cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
