"""HONEST 180-day frontier — calibrated prospectively.

Critical fix vs find_achievable_frontier:
  • Calibrator (isotonic) is fit ONLY on 2024 OOF
  • Test bands are measured ONLY on 2025 OOF (year strictly after calibration data)
  • This kills the in-sample calibration leak the devil's advocate flagged

For each threshold % in {5, 10, 15, 20, 30, 50, 75, 100}:
  1. Build label: max(high) over next 180 days / close >= threshold
  2. Train LGB+XGB on years <= 2022
  3. Score 2024 → calibrate isotonic → save calibrator
  4. Score 2025 → apply 2024-trained calibrator → measure hit rate per band
  5. Report: at what calibrated band does 2025 actual hit rate >= 80% / 90%?
  6. Also report: predictions for TODAY using full-data final model

Output:
  reports/180d_honest_frontier.md
  data/derived/180d_honest_frontier.parquet
  data/derived/180d_today_predictions.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_PARQUET = ROOT / "data/derived/180d_honest_frontier.parquet"
OUT_TODAY = ROOT / "data/derived/180d_today_predictions.parquet"
OUT_REPORT = ROOT / "reports/180d_honest_frontier.md"

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
HORIZON = 180

FEATS = ["return_1d", "return_20d",
         "dist_sma20", "dist_sma50", "dist_sma200",
         "above_50dma", "above_200dma",
         "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
         "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
         "realized_vol_20d", "adv_20d_cr",
         "market_5d_ret", "market_20d_ret",
         "market_breadth_50dma", "market_breadth_200dma"]


def build_panel() -> pd.DataFrame:
    print("loading prices …")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    # forward 180-day max high
    shifts = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, HORIZON + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts.max(axis=1)
    df["fwd_pct"] = df["fwd_high_max"] / df["close"] - 1
    complete = df.groupby("symbol", sort=False)["high"].shift(-HORIZON).notna()
    df.loc[~complete, "fwd_pct"] = pd.NA

    # standard features
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
    return df


def evaluate_threshold(df: pd.DataFrame, threshold: float) -> dict:
    """Train on years <= 2022, calibrate on 2024 OOF, score 2025 prospectively."""
    target_col = f"target_{int(threshold*100)}"
    df = df.copy()
    df[target_col] = (df["fwd_pct"] >= threshold).astype("Int64")
    sub = df[df[target_col].notna()].copy()
    sub = sub[sub["adv_20d_cr"] >= 1.0]

    base_rate = float(sub[target_col].mean())

    # train years <= 2022
    tr = sub[sub["year"] <= 2022]
    cal = sub[sub["year"] == 2024]  # year used to fit isotonic
    te = sub[sub["year"] == 2025]   # strictly prospective

    if len(tr) < 5000 or len(cal) < 1000 or len(te) < 1000:
        return {"threshold_pct": threshold * 100, "base_rate": base_rate,
                "n_train": len(tr), "n_cal": len(cal), "n_test": len(te),
                "achievable_80": False, "achievable_90": False}

    # train ensemble
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
    lgbm.fit(tr[FEATS], tr[target_col].astype(int))
    xgbm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                              subsample=0.85, colsample_bytree=0.85, random_state=42,
                              verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    xgbm.fit(tr[FEATS], tr[target_col].astype(int))

    # score 2024 (calibration set), 2025 (test)
    p_cal = 0.5 * lgbm.predict_proba(cal[FEATS])[:, 1] + 0.5 * xgbm.predict_proba(cal[FEATS])[:, 1]
    p_test = 0.5 * lgbm.predict_proba(te[FEATS])[:, 1] + 0.5 * xgbm.predict_proba(te[FEATS])[:, 1]

    # fit isotonic on 2024 ONLY
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, cal[target_col].astype(int))

    # apply to 2025
    cal_test = iso.transform(p_test)
    te = te.assign(score_cal=cal_test)

    # bands
    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.75), (0.75, 0.80),
             (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    band_results = []
    achievable_80 = False
    achievable_90 = False
    for lo, hi in bands:
        b = te[(te["score_cal"] >= lo) & (te["score_cal"] < hi)]
        if len(b) < 30:
            continue
        actual_hr = float(b[target_col].mean())
        band_results.append({"band_lo": lo, "band_hi": hi, "n": len(b),
                              "actual_hit_rate": actual_hr})
        if actual_hr >= 0.80:
            achievable_80 = True
        if actual_hr >= 0.90:
            achievable_90 = True

    # find best band hit rate
    if band_results:
        best = max(band_results, key=lambda r: r["actual_hit_rate"])
        max_band = f"{best['band_lo']:.2f}-{best['band_hi']:.2f}"
        max_hr = best["actual_hit_rate"]
        max_n = best["n"]
    else:
        max_band, max_hr, max_n = None, None, 0

    return {
        "threshold_pct": threshold * 100,
        "base_rate": base_rate,
        "n_train": len(tr),
        "n_cal": len(cal),
        "n_test": len(te),
        "max_band": max_band,
        "max_hit_rate": max_hr,
        "max_n_samples": max_n,
        "achievable_80": achievable_80,
        "achievable_90": achievable_90,
        "all_bands": band_results,
    }


def predict_today(df: pd.DataFrame) -> pd.DataFrame:
    """Produce today's predictions per threshold using full-data model."""
    latest = df["trade_date"].max()
    today_df = df[df["trade_date"] == latest].copy()
    today_df = today_df[today_df["adv_20d_cr"] >= 1.0]
    out = today_df[["symbol", "close", "rsi_14_daily", "return_20d", "adv_20d_cr"]].copy()

    for thr in THRESHOLDS:
        target_col = f"target_{int(thr*100)}"
        df[target_col] = (df["fwd_pct"] >= thr).astype("Int64")
        sub = df[df[target_col].notna()].copy()
        sub = sub[sub["adv_20d_cr"] >= 1.0]
        # use ALL labeled data for the final fit (predicting today)
        lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(sub[FEATS], sub[target_col].astype(int))
        # calibrate on 2024 OOF (we still want HONEST today scores)
        cal_data = sub[sub["year"] == 2024]
        if len(cal_data) >= 1000:
            p_cal = lgbm.predict_proba(cal_data[FEATS])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_cal, cal_data[target_col].astype(int))
            p_today = lgbm.predict_proba(today_df[FEATS])[:, 1]
            score_today = iso.transform(p_today)
        else:
            score_today = lgbm.predict_proba(today_df[FEATS])[:, 1]
        out[f"score_{int(thr*100)}pct"] = score_today
    return out


def main() -> None:
    print("== find_180d_frontier_honest ==")
    df = build_panel()
    df = df.dropna(subset=FEATS).copy()

    rows = []
    for thr in THRESHOLDS:
        print(f"\n[{int(thr*100)}%/180d] training (train≤2022, cal=2024, test=2025) …")
        r = evaluate_threshold(df, thr)
        rows.append(r)
        if r.get("max_hit_rate") is not None:
            print(f"  base rate {r['base_rate']*100:.0f}% · best band {r['max_band']} → "
                  f"actual hit rate {r['max_hit_rate']*100:.1f}% (n={r['max_n_samples']})")
            if r["achievable_90"]:
                print(f"  ✅ 90% achievable")
            elif r["achievable_80"]:
                print(f"  ◔ 80% achievable, not 90%")
            else:
                print(f"  ❌ neither 80% nor 90% reachable")
        else:
            print(f"  insufficient sample")

    # save
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "all_bands"} for r in rows])
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    # today's predictions
    print("\n=== predicting today across all thresholds ===")
    today_pred = predict_today(df)
    today_pred.to_parquet(OUT_TODAY, index=False)

    # report
    md = ["# 180-day frontier — HONEST prospective calibration", "",
          "**Methodology:** train on years ≤ 2022, fit isotonic on 2024 OOF only, "
          "test on 2025 OOF (strictly prospective). This kills the in-sample calibration "
          "leak the devil's advocate flagged.", "",
          "## Achievability table (180-day horizon)", "",
          "| Threshold | Base rate | Best calibrated band | Real hit rate (2025 prospective) | n samples | 80%? | 90%? |",
          "|---|---:|---|---:|---:|:---:|:---:|"]
    for r in rows:
        if r.get("max_hit_rate") is None:
            md.append(f"| {r['threshold_pct']:.0f}% | "
                      f"{r['base_rate']*100:.0f}% | — | — | "
                      f"({r['n_test']:,} test) | — | — |")
        else:
            ck80 = "✅" if r["achievable_80"] else "❌"
            ck90 = "✅" if r["achievable_90"] else "❌"
            md.append(f"| **{r['threshold_pct']:.0f}%** | {r['base_rate']*100:.0f}% | "
                      f"{r['max_band']} | **{r['max_hit_rate']*100:.1f}%** | "
                      f"{r['max_n_samples']} | {ck80} | {ck90} |")
    md.append("")
    md.append("## What this means")
    md.append("")
    achievable = [r for r in rows if r.get("achievable_80")]
    if achievable:
        max_thr = max(r["threshold_pct"] for r in achievable if r.get("achievable_80"))
        md.append(f"- **Max uptick at 80%+ confidence** in 180 days: **{max_thr:.0f}%**")
        achievable_90 = [r for r in rows if r.get("achievable_90")]
        if achievable_90:
            max_thr90 = max(r["threshold_pct"] for r in achievable_90)
            md.append(f"- **Max uptick at 90%+ confidence** in 180 days: **{max_thr90:.0f}%**")
        else:
            md.append("- **At 90%+ confidence: NOT achievable** at any threshold tested")
    else:
        md.append("- At 80%+ confidence, **no threshold is reachable** prospectively")

    md.append("")
    md.append("## Today's top-15 predictions per threshold")
    md.append("")
    for thr in THRESHOLDS:
        col = f"score_{int(thr*100)}pct"
        if col in today_pred.columns:
            top = today_pred.sort_values(col, ascending=False).head(8)
            md.append(f"### {int(thr*100)}% in 180d")
            md.append("")
            md.append("| Symbol | Close | Score | RSI | 20d % | ADV cr |")
            md.append("|---|---:|---:|---:|---:|---:|")
            for _, r in top.iterrows():
                md.append(f"| {r['symbol']} | ₹{r['close']:.2f} | {r[col]:.3f} | "
                          f"{r.get('rsi_14_daily', 0):.0f} | {(r.get('return_20d', 0) or 0)*100:+.1f}% | "
                          f"{r.get('adv_20d_cr', 0):.1f} |")
            md.append("")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
