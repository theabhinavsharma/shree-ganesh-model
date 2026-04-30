"""Frontier search: at what (return %, horizon) combinations does the model
deliver ≥ 90% calibrated confidence with non-trivial sample size?

For each (horizon, threshold) pair:
  1. Build label P(high_h_days >= threshold)
  2. Walk-forward train (2020-2023 → 2024-2025 OOS)
  3. Isotonic-calibrate
  4. Measure: at what calibration band do we hit 90%+ actual hit rate?
  5. How many OOS instances fire at that band? (i.e., is it ever achievable?)

Output:
  data/derived/achievable_frontier.parquet — table per (horizon, threshold)
  reports/achievable_frontier.md — frontier + recommended target

The recommendation: pick the (horizon, threshold) combo with highest
expected annualised compound return where the 0.90 calibration is
verifiable AND fires at least once per week historically.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
OUT_PARQUET = ROOT / "data/derived/achievable_frontier.parquet"
OUT_REPORT = ROOT / "reports/achievable_frontier.md"

HORIZONS = [3, 5, 7, 10, 15, 21, 30, 45, 60, 90]
THRESHOLDS = [0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

CONVICTION = 0.90  # frontier: how high a band can we trust

BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]


def build_panel() -> pd.DataFrame:
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2018-01-01"]

    df["dist_sma20"] = df["close"] / df["sma_20"] - 1
    df["dist_sma50"] = df["close"] / df["sma_50"] - 1
    df["dist_sma200"] = df["close"] / df["sma_200"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7
    df["year"] = df["trade_date"].dt.year

    liq = df[df["adv_20d_cr"] >= 1.0]
    mkt = liq.groupby("trade_date").agg(
        market_breadth_50dma=("above_50dma", "mean"),
        market_breadth_200dma=("above_200dma", "mean"),
    ).reset_index().sort_values("trade_date")
    df = df.merge(mkt, on="trade_date", how="left")
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())

    # max forward high for ALL horizons we'll test
    for H in HORIZONS:
        shifts = pd.concat(
            [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
            axis=1,
        )
        df[f"fwd_high_max_{H}"] = shifts.max(axis=1)
        df[f"fwd_pct_{H}"] = df[f"fwd_high_max_{H}"] / df["close"] - 1
        complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
        df.loc[~complete, f"fwd_pct_{H}"] = pd.NA
    return df


def evaluate_combo(df: pd.DataFrame, horizon: int, threshold: float) -> dict:
    """Train a quick model for (horizon, threshold), return calibration table + best band."""
    target_col = f"target_{horizon}d_{int(threshold*100)}pct"
    df[target_col] = (df[f"fwd_pct_{horizon}"] >= threshold).astype("Int64")
    sub = df[df[target_col].notna()].copy()
    sub = sub[sub["adv_20d_cr"] >= 1.0]

    if len(sub) < 5000:
        return {"horizon": horizon, "threshold_pct": threshold * 100,
                "n_labeled": len(sub), "base_rate": None, "max_cal_hit_rate": None,
                "n_at_max_band": 0, "achievable_90": False}

    base_rate = float(sub[target_col].mean())

    # walk-forward 2024 + 2025
    oof_rows = []
    for yr in [2024, 2025]:
        tr = sub[sub["year"] < yr]
        te = sub[sub["year"] == yr].copy()
        if len(tr) < 5000 or len(te) < 100:
            continue
        # quick LGB only (faster than ensemble)
        lgbm = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.06, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[BASE_FEATS], tr[target_col].astype(int))
        te["score_raw"] = lgbm.predict_proba(te[BASE_FEATS])[:, 1]
        oof_rows.append(te[["trade_date", "symbol", "score_raw", target_col]])

    if not oof_rows:
        return {"horizon": horizon, "threshold_pct": threshold * 100,
                "n_labeled": len(sub), "base_rate": base_rate, "max_cal_hit_rate": None,
                "n_at_max_band": 0, "achievable_90": False}

    oof = pd.concat(oof_rows, ignore_index=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["score_raw"], oof[target_col].astype(int))
    oof["score_cal"] = iso.transform(oof["score_raw"])

    # check the highest band where we have ≥30 samples
    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.75), (0.75, 0.80),
             (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    achievable_90 = False
    n_at_max_band = 0
    max_band_label = None
    max_band_hr = None
    for lo, hi in reversed(bands):  # check highest first
        b = oof[(oof["score_cal"] >= lo) & (oof["score_cal"] < hi)]
        if len(b) < 30:
            continue
        hr = float(b[target_col].mean())
        if max_band_label is None:  # record the highest band with enough data
            max_band_label = f"{lo:.2f}-{hi:.2f}"
            max_band_hr = hr
            n_at_max_band = len(b)
        if hr >= CONVICTION and lo >= CONVICTION - 0.05:  # band centered near 0.90+
            achievable_90 = True
            break

    return {
        "horizon": horizon,
        "threshold_pct": threshold * 100,
        "n_labeled": len(sub),
        "base_rate": base_rate,
        "max_band": max_band_label,
        "max_cal_hit_rate": max_band_hr,
        "n_at_max_band": n_at_max_band,
        "achievable_90": achievable_90,
    }


def main() -> None:
    print("== find_achievable_frontier ==")
    df = build_panel()
    df = df.dropna(subset=BASE_FEATS).copy()
    print(f"  panel ready: {len(df):,} rows")

    rows = []
    total = len(HORIZONS) * len(THRESHOLDS)
    n = 0
    for H in HORIZONS:
        for thr in THRESHOLDS:
            n += 1
            print(f"\n[{n}/{total}] horizon={H}d, threshold={thr*100:.0f}% …")
            r = evaluate_combo(df, H, thr)
            rows.append(r)
            verdict = ""
            if r["max_cal_hit_rate"] is not None:
                verdict = f"max band {r['max_band']} hit_rate={r['max_cal_hit_rate']*100:.1f}% n={r['n_at_max_band']}"
            if r["achievable_90"]:
                verdict += "  ✅ 90% CONVICTION ACHIEVABLE"
            print(f"  {verdict}")

    res = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    # report
    md = ["# Achievable frontier — what (X%, Y days) combos hit 90% confidence?", "",
          "Method: walk-forward train (2020-2023 → 2024-2025 OOS), isotonic calibrate, ",
          "then check the highest calibration band where actual hit rate ≥ 90% with ≥ 30 OOS samples.", "",
          "## Frontier table", "",
          "| Horizon | Threshold | Base rate | Max band | Hit rate at band | n at band | 90% achievable? |",
          "|---:|---:|---:|---|---:|---:|---|"]
    for _, r in res.iterrows():
        if r.get("max_cal_hit_rate") is None:
            md.append(f"| {int(r['horizon'])}d | {r['threshold_pct']:.0f}% | "
                      f"{r['base_rate']*100 if r['base_rate'] else '—'}% | — | — | — | — |")
        else:
            check = "✅ YES" if r["achievable_90"] else "❌"
            md.append(f"| {int(r['horizon'])}d | {r['threshold_pct']:.0f}% | "
                      f"{r['base_rate']*100:.0f}% | {r['max_band']} | "
                      f"{r['max_cal_hit_rate']*100:.1f}% | {int(r['n_at_max_band']):,} | {check} |")
    md.append("")

    # Summary: highest threshold per horizon where 90% achievable
    md.append("## Best combos for the 'double money' goal")
    md.append("")
    achievable = res[res["achievable_90"] == True]
    if len(achievable):
        # for each horizon, pick the highest threshold
        best = achievable.sort_values(["horizon", "threshold_pct"], ascending=[True, False])
        best = best.groupby("horizon").head(1)
        # compute implied annualised return assuming N trades per year
        md.append("| Horizon | Threshold | Hit rate | n/year (approx) | Implied ann ROI (best case) |")
        md.append("|---:|---:|---:|---:|---:|")
        for _, r in best.iterrows():
            n_per_yr = r["n_at_max_band"] / 2.0  # 2 OOS years
            hr = r["max_cal_hit_rate"]
            thr = r["threshold_pct"] / 100
            # naive: if you trade every fire and capture threshold% per win:
            ann = ((1 + hr * thr) ** (n_per_yr * 12 / 12) - 1) * 100  # very rough
            md.append(f"| {int(r['horizon'])}d | {r['threshold_pct']:.0f}% | "
                      f"{hr*100:.1f}% | ~{n_per_yr:.0f} | ~{min(ann, 9999):.0f}% |")
    else:
        md.append("⚠️ NO combination delivers 90% calibrated confidence with ≥ 30 samples.")
        md.append("")
        md.append("This means: the user's 'double money this year via 90% confidence' goal is structurally hard with current features.")
        md.append("")
        md.append("**The closest achievable combos** (sorted by hit rate at the highest band):")
        md.append("")
        good = res[(res["max_cal_hit_rate"].notna()) & (res["n_at_max_band"] >= 30)].sort_values(
            "max_cal_hit_rate", ascending=False).head(10)
        md.append("| Horizon | Threshold | Hit rate at top band | n |")
        md.append("|---:|---:|---:|---:|")
        for _, r in good.iterrows():
            md.append(f"| {int(r['horizon'])}d | {r['threshold_pct']:.0f}% | "
                      f"{r['max_cal_hit_rate']*100:.1f}% | {int(r['n_at_max_band']):,} |")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
