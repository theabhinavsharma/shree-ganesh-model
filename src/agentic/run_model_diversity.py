"""Train 5 diverse model families on the same feature set + label.

Why: LightGBM and XGBoost are the same algorithm class (gradient-boosted trees).
True ensemble diversity needs models that make different *kinds* of errors:

  1. LightGBM         (boosted trees, leaf-wise growth)
  2. XGBoost          (boosted trees, level-wise growth)
  3. RandomForest     (bagged trees, variance reduction)
  4. ExtraTrees       (extreme randomization, less overfit)
  5. LogisticRegression L2 (linear baseline — captures linear effects only)

Output:
  data/derived/model_diversity_scores.parquet — per-symbol predicted prob from each
  data/derived/model_diversity_metrics.parquet — fold-level top-5 hit-rate per model
  reports/model_diversity_summary.md — disagreement matrix + per-model top-5 picks

The hypothesis we test:
  "A 5-model average ensemble outperforms LGB+XGB alone on top-5 daily basket
   over the 2024-2025 OOS, because the 3 added models make uncorrelated errors."

Verdict in: docs/agent_decisions/model_diversity_<date>.md
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_SCORES = ROOT / "data/derived/model_diversity_scores.parquet"
OUT_METRICS = ROOT / "data/derived/model_diversity_metrics.parquet"
OUT_REPORT = ROOT / "reports/model_diversity_summary.md"

H = 7

FEATS = ["return_1d", "return_20d",
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
    shifts_high = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts_high.max(axis=1)
    df["forward_high_pct_7td"] = df["fwd_high_max"] / df["close"] - 1
    df["winner_5pct_7td"] = (df["forward_high_pct_7td"] >= 0.05).astype(int)
    df["close_fwd_7"] = df.groupby("symbol")["close"].shift(-H)
    df["fwd_c2c_7"] = df["close_fwd_7"] / df["close"] - 1
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, ["forward_high_pct_7td", "winner_5pct_7td"]] = pd.NA

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


def fit_model(name: str, X_train, y_train, X_test):
    """Returns (predicted probabilities for class 1, fit_time_seconds)."""
    t0 = time.time()
    if name == "LightGBM":
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
    elif name == "XGBoost":
        m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                              subsample=0.85, colsample_bytree=0.85, random_state=42,
                              verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
    elif name == "RandomForest":
        m = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200,
                                    n_jobs=-1, random_state=42)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
    elif name == "ExtraTrees":
        m = ExtraTreesClassifier(n_estimators=200, max_depth=12, min_samples_leaf=200,
                                  n_jobs=-1, random_state=42)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
    elif name == "LogisticL2":
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)
        m = LogisticRegression(penalty="l2", C=1.0, max_iter=200, n_jobs=-1, random_state=42)
        m.fit(X_train_s, y_train)
        p = m.predict_proba(X_test_s)[:, 1]
    else:
        raise ValueError(name)
    return p, time.time() - t0


MODELS = ["LightGBM", "XGBoost", "RandomForest", "ExtraTrees", "LogisticL2"]


def main() -> None:
    print("== run_model_diversity ==")
    print(f"  building panel …")
    df = build_panel()
    df = df.dropna(subset=FEATS).copy()
    labeled = df[df["winner_5pct_7td"].notna() & df["fwd_c2c_7"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"  labeled rows: {len(labeled):,}")

    test_years = [2024, 2025]
    score_rows = []
    metrics_rows = []
    for yr in test_years:
        tr = labeled[labeled["year"] < yr]
        te = labeled[labeled["year"] == yr].copy()
        if len(tr) < 5000 or len(te) < 100:
            continue
        print(f"\n=== {yr} fold (train n={len(tr):,}, test n={len(te):,}) ===")
        preds: dict[str, np.ndarray] = {}
        fit_times: dict[str, float] = {}
        for model in MODELS:
            try:
                p, ft = fit_model(model, tr[FEATS], tr["winner_5pct_7td"].astype(int), te[FEATS])
                preds[model] = p
                fit_times[model] = ft
                # daily top-5 hit-rate using this model
                te_with = te.copy()
                te_with["pred"] = p
                top5 = te_with.sort_values(["trade_date", "pred"], ascending=[True, False]).groupby("trade_date").head(5)
                basket = top5.groupby("trade_date")["fwd_c2c_7"].mean()
                hit5 = top5.groupby("trade_date")["winner_5pct_7td"].mean().mean()
                auc = roc_auc_score(te["winner_5pct_7td"].astype(int), p)
                metrics_rows.append({
                    "year": yr, "model": model, "fit_time_s": round(ft, 1),
                    "auc": round(float(auc), 4),
                    "top5_winner_rate": round(float(hit5), 4),
                    "top5_basket_mean_7d": round(float(basket.mean()), 4),
                    "top5_basket_median_7d": round(float(basket.median()), 4),
                    "top5_days_5pct": int((basket >= 0.05).sum()),
                    "top5_n_days": int(len(basket)),
                })
                print(f"  {model:<14} fit={ft:6.1f}s  auc={auc:.3f}  top5_hit={hit5*100:5.1f}%  basket_mean_7d={basket.mean()*100:+.2f}%")
            except Exception as exc:
                print(f"  {model:<14} FAIL: {type(exc).__name__}: {str(exc)[:140]}")

        # 5-model average ensemble
        if len(preds) == len(MODELS):
            avg_pred = np.mean([preds[m] for m in MODELS], axis=0)
            te_avg = te.copy()
            te_avg["pred"] = avg_pred
            top5 = te_avg.sort_values(["trade_date", "pred"], ascending=[True, False]).groupby("trade_date").head(5)
            basket = top5.groupby("trade_date")["fwd_c2c_7"].mean()
            hit5 = top5.groupby("trade_date")["winner_5pct_7td"].mean().mean()
            auc = roc_auc_score(te["winner_5pct_7td"].astype(int), avg_pred)
            metrics_rows.append({
                "year": yr, "model": "ENSEMBLE_5_AVG", "fit_time_s": sum(fit_times.values()),
                "auc": round(float(auc), 4),
                "top5_winner_rate": round(float(hit5), 4),
                "top5_basket_mean_7d": round(float(basket.mean()), 4),
                "top5_basket_median_7d": round(float(basket.median()), 4),
                "top5_days_5pct": int((basket >= 0.05).sum()),
                "top5_n_days": int(len(basket)),
            })
            print(f"  {'ENSEMBLE_5_AVG':<14} sum={sum(fit_times.values()):6.1f}s  auc={auc:.3f}  top5_hit={hit5*100:5.1f}%  basket_mean_7d={basket.mean()*100:+.2f}%")

            # save per-symbol scores from each model + ensemble
            te_save = te[["trade_date", "symbol"]].copy()
            for m in MODELS:
                te_save[f"score_{m}"] = preds[m]
            te_save["score_ensemble"] = avg_pred
            score_rows.append(te_save)

    metrics = pd.DataFrame(metrics_rows)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(OUT_METRICS, index=False)

    if score_rows:
        scores = pd.concat(score_rows, ignore_index=True)
        scores.to_parquet(OUT_SCORES, index=False)

    # per-pair score correlation across the 5 models (read from saved scores)
    if score_rows:
        all_scores = pd.concat(score_rows, ignore_index=True)
        cols = [f"score_{m}" for m in MODELS]
        corr = all_scores[cols].corr()
        print("\nScore correlation matrix (5 models, OOS):")
        print(corr.round(3).to_string())
    else:
        corr = pd.DataFrame()

    # write report
    md = ["# Model diversity — daily ensemble", "",
          "## Per-model OOS performance (2024-2025)", "",
          "| Year | Model | Fit time | AUC | Top-5 hit | Basket mean 7d | Basket median 7d | Days >=+5% |",
          "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in metrics.iterrows():
        md.append(f"| {int(r['year'])} | {r['model']} | {r['fit_time_s']:.1f}s | {r['auc']:.3f} | "
                  f"{r['top5_winner_rate']*100:.1f}% | {r['top5_basket_mean_7d']*100:+.2f}% | "
                  f"{r['top5_basket_median_7d']*100:+.2f}% | {r['top5_days_5pct']}/{r['top5_n_days']} |")
    if len(corr):
        md.append("\n## Score correlation matrix (5 models)\n")
        md.append("| | " + " | ".join(MODELS) + " |")
        md.append("|---|" + "|".join(["---:"] * len(MODELS)) + "|")
        for m in MODELS:
            md.append(f"| {m} | " + " | ".join(f"{corr.at[f'score_{m}', f'score_{m2}']:.3f}" for m2 in MODELS) + " |")
        md.append("")
        md.append("**Reading:** correlations near 1.0 = redundant models. Lower = genuine diversity.")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_METRICS}")
    print(f"     {OUT_SCORES}")


if __name__ == "__main__":
    main()
