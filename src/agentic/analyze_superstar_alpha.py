"""Analyze whether superstar-held stocks deliver alpha vs non-held.

Method:
  1. Load current superstar holdings (data/derived/superstar_holdings.parquet)
  2. Bucket every NSE-liquid stock by confluence count (0, 1, 2, 3+)
  3. For each bucket, compute avg 7d forward return over 2024-2025 OOS
  4. Cross-reference today's confluence stocks with today's model picks
     → identifies intersection where BOTH agree (high conviction)

Caveats acknowledged:
  • Superstar holdings here are TODAY's snapshot, not historical.
    A stock held today by a superstar may not have been held in 2024.
    So this is a *proxy* test — assumes their picks were stable.
  • To do this properly we'd need quarterly historical holdings.
    Filed as future TODO.

Output:
  data/derived/superstar_alpha_analysis.parquet — per-stock summary
  data/derived/superstar_today_intersection.csv — today's confluence × model picks
  reports/superstar_alpha.md — human-readable report
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
HOLDINGS = ROOT / "data/derived/superstar_holdings.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
MH = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"
OUT_PARQUET = ROOT / "data/derived/superstar_alpha_analysis.parquet"
OUT_CSV = ROOT / "data/derived/superstar_today_intersection.csv"
OUT_REPORT = ROOT / "reports/superstar_alpha.md"


def main() -> None:
    print("== analyze_superstar_alpha ==")
    if not HOLDINGS.exists():
        print(f"missing {HOLDINGS} — run fetch_superstar_holdings.py first")
        return
    h = pd.read_parquet(HOLDINGS)
    # most recent fetch only
    h["fetch_date"] = pd.to_datetime(h["fetch_date"]).dt.date
    h = h[h["fetch_date"] == h["fetch_date"].max()]
    # confluence count (HDFCBANK/RELIANCE/TCS-style outliers stay in here; we'll handle in display)
    confluence = (h.groupby("symbol")
                    .agg(n_superstars=("investor_tag", "nunique"),
                         investors=("investor_tag", lambda s: ", ".join(sorted(set(s)))))
                    .reset_index())
    print(f"  superstar universe: {len(confluence):,} unique stocks across {h['investor_tag'].nunique()} investors")

    # Filter likely-noise (>10 superstars probably means page-header artifact)
    real_confluence = confluence[(confluence["n_superstars"] >= 2) &
                                  (confluence["n_superstars"] <= 10)]
    print(f"  real confluence (2-10 superstars): {len(real_confluence)} stocks")

    # OOS forward returns
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close",
                                           "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    px["close_fwd_7"] = px.groupby("symbol")["close"].shift(-7)
    px["fwd_c2c_7"] = px["close_fwd_7"] / px["close"] - 1
    px["adv_cr"] = px["avg_traded_value_20d"] / 1e7
    # OOS window: 2024-2025
    oos = px[(px["trade_date"] >= "2024-01-01") & (px["trade_date"] <= "2025-12-31") &
             (px["adv_cr"] >= 1.0)]
    oos = oos.dropna(subset=["fwd_c2c_7"])
    oos = oos.merge(confluence[["symbol", "n_superstars"]], on="symbol", how="left")
    oos["n_superstars"] = oos["n_superstars"].fillna(0).astype(int)
    # bucket
    oos["bucket"] = pd.cut(oos["n_superstars"],
                            bins=[-1, 0, 1, 2, 999],
                            labels=["0 - none", "1 - solo", "2 - pair", "3+ - cluster"])

    summary = (oos.groupby("bucket", observed=False)
                .agg(n_stockdays=("fwd_c2c_7", "size"),
                     mean_7d=("fwd_c2c_7", "mean"),
                     median_7d=("fwd_c2c_7", "median"),
                     pct_pos=("fwd_c2c_7", lambda s: (s > 0).mean()),
                     pct_5pct=("fwd_c2c_7", lambda s: (s >= 0.05).mean()))
                .round(4))
    print(f"\nOOS forward 7d return by superstar-confluence bucket (2024-2025):")
    print(summary.to_string())

    # exclude the noise tickers (HDFCBANK/RELIANCE/TCS-type with all-17)
    high_conf_excl = confluence[(confluence["n_superstars"] >= 2) &
                                  (confluence["n_superstars"] <= 10)]["symbol"].tolist()
    sub = oos[oos["symbol"].isin(high_conf_excl)]
    real_summary = {
        "n": len(sub),
        "mean_7d": sub["fwd_c2c_7"].mean(),
        "median_7d": sub["fwd_c2c_7"].median(),
        "pct_5pct": (sub["fwd_c2c_7"] >= 0.05).mean(),
        "pct_pos": (sub["fwd_c2c_7"] > 0).mean(),
    }
    sub_other = oos[~oos["symbol"].isin(high_conf_excl) & (oos["n_superstars"] == 0)]
    other_summary = {
        "n": len(sub_other),
        "mean_7d": sub_other["fwd_c2c_7"].mean(),
        "median_7d": sub_other["fwd_c2c_7"].median(),
        "pct_5pct": (sub_other["fwd_c2c_7"] >= 0.05).mean(),
        "pct_pos": (sub_other["fwd_c2c_7"] > 0).mean(),
    }

    # today's intersection: stocks with 2+ superstars × model top-30 picks
    long_df = pd.read_csv(LIVE_LONG) if LIVE_LONG.exists() else pd.DataFrame()
    mh_df = pd.read_csv(MH) if MH.exists() else pd.DataFrame()
    today_picks = pd.DataFrame()
    if len(long_df):
        top30_long = long_df.sort_values("score_calibrated", ascending=False).head(30)
        top30_long = top30_long.merge(confluence, on="symbol", how="left")
        top30_long["n_superstars"] = top30_long["n_superstars"].fillna(0).astype(int)
        today_picks = top30_long[["symbol", "sector", "close", "score_calibrated",
                                   "n_superstars", "investors"]].copy()
        today_picks.to_csv(OUT_CSV, index=False)

    # WRITE REPORT
    md = ["# Superstar-confluence alpha analysis", "",
          "_Question: do stocks held by 2+ celebrity investors deliver better forward returns_",
          "_than the broader liquid universe? Tested on 2024-2025 OOS data._", "",
          "## Caveat",
          "",
          "This analysis uses TODAY's superstar holdings against historical OOS prices.",
          "We don't have quarterly history of holdings yet, so this is a proxy — assumes",
          "their picks were broadly stable. To do it properly we need historical holdings;",
          "filed as next-cycle TODO.", "",
          "## Forward 7d return by superstar-confluence bucket", "",
          "| Confluence | n stock-days | Mean 7d | Median 7d | % positive | % >= +5% |",
          "|---|---:|---:|---:|---:|---:|"]
    for idx, r in summary.iterrows():
        md.append(f"| {idx} | {int(r['n_stockdays']):,} | {r['mean_7d']*100:+.2f}% | "
                  f"{r['median_7d']*100:+.2f}% | {r['pct_pos']*100:.1f}% | {r['pct_5pct']*100:.1f}% |")
    md.append("")
    md.append("## Filtered: real confluence (2-10 superstars, exclude noise) vs no-superstar")
    md.append("")
    md.append("| Group | n stock-days | Mean 7d | Median 7d | % positive | % >= +5% |")
    md.append("|---|---:|---:|---:|---:|---:|")
    md.append(f"| **Confluence 2-10** | {real_summary['n']:,} | "
              f"**{real_summary['mean_7d']*100:+.2f}%** | "
              f"{real_summary['median_7d']*100:+.2f}% | "
              f"{real_summary['pct_pos']*100:.1f}% | {real_summary['pct_5pct']*100:.1f}% |")
    md.append(f"| No superstar | {other_summary['n']:,} | "
              f"{other_summary['mean_7d']*100:+.2f}% | "
              f"{other_summary['median_7d']*100:+.2f}% | "
              f"{other_summary['pct_pos']*100:.1f}% | {other_summary['pct_5pct']*100:.1f}% |")
    md.append("")
    delta_mean = (real_summary["mean_7d"] - other_summary["mean_7d"]) * 100
    delta_5pct = (real_summary["pct_5pct"] - other_summary["pct_5pct"]) * 100
    md.append(f"**Mean 7d delta: {delta_mean:+.2f} pp** · **% >= 5% delta: {delta_5pct:+.1f} pp**")
    md.append("")
    if delta_mean > 0.3:
        md.append("✅ **Real signal**: superstar-confluence stocks meaningfully outperform.")
    elif delta_mean > 0:
        md.append("◯ **Marginal signal**: small positive lift, may not survive significance test.")
    else:
        md.append("❌ **No lift**: superstar-confluence stocks do NOT outperform.")
    md.append("")
    md.append("## Today's intersection — model top-30 LONG × superstar-confluence")
    md.append("")
    if len(today_picks):
        md.append("| Symbol | Sector | Close | Score | Superstars | Investors |")
        md.append("|---|---|---:|---:|---:|---|")
        for _, r in today_picks.head(30).iterrows():
            investors_val = r.get("investors")
            inv = (str(investors_val) if pd.notna(investors_val) else "")[:60]
            n_ss = int(r["n_superstars"]) if pd.notna(r["n_superstars"]) else 0
            stars = "⭐" * min(n_ss, 5) if n_ss > 0 else "—"
            md.append(f"| **{r['symbol']}** | {r['sector']} | ₹{r['close']:.2f} | "
                      f"{r['score_calibrated']:.2f} | {stars} ({n_ss}) | {inv} |")
    else:
        md.append("_No live picks file._")
    md.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_CSV}")

    # save the parquet
    summary_df = summary.reset_index()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(OUT_PARQUET, index=False)

    # console summary
    print(f"\n=== Filtered: real confluence vs no-superstar ===")
    print(f"  Confluence 2-10: n={real_summary['n']:,} mean_7d={real_summary['mean_7d']*100:+.2f}% "
          f"median={real_summary['median_7d']*100:+.2f}% pct>=+5%={real_summary['pct_5pct']*100:.1f}%")
    print(f"  No superstar:    n={other_summary['n']:,} mean_7d={other_summary['mean_7d']*100:+.2f}% "
          f"median={other_summary['median_7d']*100:+.2f}% pct>=+5%={other_summary['pct_5pct']*100:.1f}%")
    print(f"  DELTA mean_7d: {delta_mean:+.2f}pp · DELTA pct>=5%: {delta_5pct:+.1f}pp")


if __name__ == "__main__":
    main()
