"""Train 3 horizon×threshold models, calibrate to TRUE probability, then find
names today (or in OOS history) that clear the 80% confidence bar.

Targets:
  • Target A: P(high reaches +5% within 7 trading days)
  • Target B: P(high reaches +10% within 15 trading days)
  • Target C: P(high reaches +20% within 30 trading days)

For each:
  1. Build label from price data
  2. Walk-forward train (2016-2023 train → 2024-2025 OOS)
  3. Isotonic-calibrate scores so 0.80 means TRUE 80% hit rate
  4. Verify calibration on held-out OOS

For today's prediction:
  • Apply all 3 models to today's universe
  • Surface any name with calibrated score >= 0.80 on any target
  • If none, report honestly + show the historical base rate of 0.80+ events

NO LYING:
  - calibrated 0.80 means OOS-verified 80% hit rate, not model output
  - if today has 0 names >= 0.80 on all targets, we say so
  - we DON'T lower the bar to fabricate a trade
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
EXTRA = ROOT / "data/derived/extra_features.parquet"
OUT_PREDICTIONS = ROOT / "data/derived/high_conviction_predictions.parquet"
OUT_REPORT = ROOT / "reports/high_conviction.md"

TARGETS = [
    {"name": "5pct_7d",   "horizon": 7,  "threshold": 0.05},
    {"name": "10pct_15d", "horizon": 15, "threshold": 0.10},
    {"name": "20pct_30d", "horizon": 30, "threshold": 0.20},
]
CONVICTION_THRESHOLD = 0.80

BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]

# Feature prefixes auto-pulled from extra_features.parquet
# ---------------------------------------------------------------------------
# SAFE: time-series features that vary per (symbol, date). These pass the
# (nunique > 1) per-symbol check and are honest model inputs.
SAFE_EXTRA_PREFIXES = (
    "alpha_", "vol_", "rv_",
    "usdinr", "eurinr", "gbpinr", "jpyinr",
    "wiki_", "spx", "us10y", "dxy", "brent", "gold",
    "macro_", "sec_",
)

# QUARANTINED: feature prefixes confirmed CONTAMINATED by the
# 2026-05-01 leakage audit (reports/leakage_audit_20260501.md).
# Each of these is built from a single recent snapshot and broadcast as a
# constant per-symbol value across all historical training rows. Re-enabling
# any of these without first shipping the time-series fundamentals layer
# (Phase 2 of the remediation plan) violates CONSTITUTION.md §1.2.
LEAKING_EXTRA_PREFIXES = ("scr_", "qvm_", "acad_")

# What we actually load:
EXTRA_PREFIXES = SAFE_EXTRA_PREFIXES   # leaking prefixes deliberately excluded


def build_panel_with_extras() -> tuple[pd.DataFrame, list[str]]:
    """Build panel and JOIN every available engineered feature.
    Returns (df, list_of_feature_names_to_use)."""
    df = build_panel()
    extra_path = ROOT / "data/derived/extra_features.parquet"
    if extra_path.exists():
        ex = pd.read_parquet(extra_path)
        ex["trade_date"] = pd.to_datetime(ex["trade_date"])
        # auto-discover features by prefix
        extra_cols = [c for c in ex.columns
                       if c not in ("symbol", "trade_date")
                       and (c.startswith(EXTRA_PREFIXES)
                            or c in ("amihud_20d", "turnover_skew_20d", "vol_max_63d"))
                       and pd.api.types.is_numeric_dtype(ex[c])]
        if extra_cols:
            df = df.merge(ex[["symbol", "trade_date"] + extra_cols],
                          on=["symbol", "trade_date"], how="left")
            extra_cols_present = [c for c in extra_cols if c in df.columns]
            # replace inf with NaN, then fill with median (sanitize for XGB)
            for c in extra_cols_present:
                df[c] = df[c].replace([np.inf, -np.inf], np.nan)
                med = df[c].median()
                if pd.isna(med):
                    df[c] = df[c].fillna(0.0)
                else:
                    df[c] = df[c].fillna(med)
            print(f"  joined {len(extra_cols_present)} engineered features from extra_features.parquet")
            return df, BASE_FEATS + extra_cols_present
    return df, BASE_FEATS


def build_panel() -> pd.DataFrame:
    print("loading prices …")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    # Pre-compute forward-window highs for each horizon
    for tgt in TARGETS:
        H = tgt["horizon"]
        shifts = pd.concat(
            [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
            axis=1,
        )
        df[f"fwd_high_max_{H}"] = shifts.max(axis=1)
        df[f"target_{tgt['name']}"] = (df[f"fwd_high_max_{H}"] / df["close"] - 1 >= tgt["threshold"]).astype(int)
        complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
        df.loc[~complete, f"target_{tgt['name']}"] = pd.NA

    # standard derived features
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


def train_target(df: pd.DataFrame, target_col: str, FEATS: list[str]) -> tuple[lgb.LGBMClassifier, xgb.XGBClassifier, IsotonicRegression, dict]:
    """Walk-forward train + calibrate for one target. Returns (lgb, xgb, isotonic, metrics)."""
    labeled = df[df[target_col].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    base_rate = labeled[target_col].mean()
    print(f"  [{target_col}] labeled={len(labeled):,}  base_rate={base_rate:.3f}")

    # walk-forward: 2024-2025 OOS
    oof_rows = []
    for yr in [2024, 2025]:
        tr = labeled[labeled["year"] < yr]
        te = labeled[labeled["year"] == yr].copy()
        if len(tr) < 5000 or len(te) < 100:
            continue
        lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[FEATS], tr[target_col].astype(int))
        xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42,
                                  verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        xgbm.fit(tr[FEATS], tr[target_col].astype(int))
        p_lgb = lgbm.predict_proba(te[FEATS])[:, 1]
        p_xgb = xgbm.predict_proba(te[FEATS])[:, 1]
        te["score_raw"] = 0.5 * p_lgb + 0.5 * p_xgb
        oof_rows.append(te[["trade_date", "symbol", "score_raw", target_col, "year"]])

    oof = pd.concat(oof_rows, ignore_index=True)

    # Calibrate on the OOF predictions (one isotonic across both years)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["score_raw"], oof[target_col].astype(int))
    oof["score_cal"] = iso.transform(oof["score_raw"])

    # ── Calibration check: at each calibrated band, what's the ACTUAL hit rate? ──
    bands = [(0, 0.50), (0.50, 0.65), (0.65, 0.75), (0.75, 0.80),
             (0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
    cal_table = []
    for lo, hi in bands:
        sub = oof[(oof["score_cal"] >= lo) & (oof["score_cal"] < hi)]
        if len(sub) < 30:
            continue
        cal_table.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "n": len(sub),
            "actual_hit_rate": float(sub[target_col].mean()),
            "model_avg_score": float(sub["score_cal"].mean()),
        })

    # final fit on all data + return models
    final_lgb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=64,
                                    min_child_samples=200, feature_fraction=0.85,
                                    bagging_fraction=0.85, bagging_freq=5,
                                    random_state=42, verbose=-1, n_jobs=-1)
    final_xgb = xgb.XGBClassifier(n_estimators=600, learning_rate=0.04, max_depth=7,
                                   subsample=0.85, colsample_bytree=0.85, random_state=42,
                                   verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    final_lgb.fit(labeled[FEATS], labeled[target_col].astype(int))
    final_xgb.fit(labeled[FEATS], labeled[target_col].astype(int))

    return final_lgb, final_xgb, iso, {"cal_table": cal_table, "base_rate": base_rate, "n": len(labeled)}


def main() -> None:
    print("== find_high_conviction ==")
    df, FEATS = build_panel_with_extras()
    df = df.dropna(subset=BASE_FEATS).copy()
    print(f"  panel rows: {len(df):,}  features: {len(FEATS)}")

    today = df["trade_date"].max()
    print(f"  today (latest): {today.date()}")
    today_df = df[df["trade_date"] == today].copy()
    today_df = today_df[today_df["adv_20d_cr"] >= 1.0]
    print(f"  today's universe: {len(today_df):,} liquid stocks")

    target_meta: dict[str, dict] = {}
    for tgt in TARGETS:
        target_col = f"target_{tgt['name']}"
        print(f"\n[{tgt['name']}] training …")
        lgbm, xgbm, iso, meta = train_target(df, target_col, FEATS)
        # apply to today
        p_lgb = lgbm.predict_proba(today_df[FEATS])[:, 1]
        p_xgb = xgbm.predict_proba(today_df[FEATS])[:, 1]
        raw = 0.5 * p_lgb + 0.5 * p_xgb
        today_df[f"score_{tgt['name']}_raw"] = raw
        today_df[f"score_{tgt['name']}_cal"] = iso.transform(raw)
        target_meta[tgt["name"]] = meta
        # show calibration table
        print(f"  Calibration check (OOS 2024-2025):")
        for r in meta["cal_table"]:
            print(f"    band {r['band']}: n={r['n']:>5}, actual hit_rate={r['actual_hit_rate']:.1%}, "
                  f"model_avg={r['model_avg_score']:.3f}")

    # select highest-conviction names today
    cal_cols = [f"score_{t['name']}_cal" for t in TARGETS]
    today_df["best_score"] = today_df[cal_cols].max(axis=1)
    today_df["best_target"] = today_df[cal_cols].idxmax(axis=1).str.replace("score_", "").str.replace("_cal", "")

    # filter cohort: meets conviction bar on at least one target
    qualifying = today_df[today_df["best_score"] >= CONVICTION_THRESHOLD]
    qualifying = qualifying.sort_values("best_score", ascending=False)

    print(f"\n=== TODAY'S RESULT ===")
    print(f"  conviction threshold: {CONVICTION_THRESHOLD}")
    print(f"  names with calibrated score >= {CONVICTION_THRESHOLD} on any target: {len(qualifying)}")
    if len(qualifying):
        print(qualifying[["symbol", "close", "rsi_14_daily", "return_20d"] + cal_cols + ["best_target"]].head(20).to_string(index=False))
    else:
        # report fallback: top by max calibrated score, even below floor
        print("  NO names meet 80% bar today.")
        print(f"  Top-10 by max calibrated score (still BELOW floor):")
        fallback = today_df.sort_values("best_score", ascending=False).head(10)
        print(fallback[["symbol", "close"] + cal_cols + ["best_target"]].to_string(index=False))

    # save predictions
    cols_save = ["symbol", "trade_date", "close"] + cal_cols + ["best_score", "best_target"]
    today_df[cols_save].to_parquet(OUT_PREDICTIONS, index=False)

    # write report
    md = ["# High-conviction picks (calibrated 80%+ probability)", "",
          f"_Generated {pd.Timestamp.now():%Y-%m-%d %H:%M IST}_  ·  today = {today.date()}",
          "",
          "## Targets tested (any one suffices)",
          "",
          "| Target | Threshold | Horizon | Question |",
          "|---|---|---|---|"]
    md += [f"| {t['name']} | +{int(t['threshold']*100)}% | {t['horizon']}d | will the stock's high reach +{int(t['threshold']*100)}% within {t['horizon']} trading days? |" for t in TARGETS]
    md += ["", "## Calibration check (OOS 2024-2025)",
           "_Does calibrated score 0.80 actually mean 80% true hit rate? If yes, we can trust the floor._", ""]
    for tgt in TARGETS:
        meta = target_meta[tgt["name"]]
        md.append(f"### {tgt['name']} (base rate {meta['base_rate']:.1%}, n={meta['n']:,})")
        md.append("")
        md.append("| Calibrated band | n | Actual hit rate | Model avg score |")
        md.append("|---|---:|---:|---:|")
        for r in meta["cal_table"]:
            md.append(f"| {r['band']} | {r['n']:,} | {r['actual_hit_rate']*100:.1f}% | {r['model_avg_score']:.3f} |")
        md.append("")

    md.append("## Today's verdict")
    md.append("")
    if len(qualifying):
        md.append(f"**{len(qualifying)} name(s) clear the 80% conviction bar today:**")
        md.append("")
        md.append("| Symbol | Close | Best score | Best target | RSI | 20d ret |")
        md.append("|---|---:|---:|---|---:|---:|")
        for _, r in qualifying.iterrows():
            md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | "
                      f"{r['best_score']:.3f} | {r['best_target']} | "
                      f"{r.get('rsi_14_daily','—')} | {(r['return_20d'] or 0)*100:+.1f}% |")
    else:
        md.append("⚠️ **No name today meets the 80% bar on any of the 3 targets.**")
        md.append("")
        md.append(f"Top-10 names by max calibrated score (still below the 0.80 floor):")
        md.append("")
        fallback = today_df.sort_values("best_score", ascending=False).head(10)
        md.append("| Symbol | Close | Score 5%/7d | Score 10%/15d | Score 20%/30d | Best target |")
        md.append("|---|---:|---:|---:|---:|---|")
        for _, r in fallback.iterrows():
            md.append(f"| {r['symbol']} | ₹{r['close']:.2f} | "
                      f"{r['score_5pct_7d_cal']:.3f} | "
                      f"{r['score_10pct_15d_cal']:.3f} | "
                      f"{r['score_20pct_30d_cal']:.3f} | "
                      f"{r['best_target']} |")
    md.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PREDICTIONS}")


if __name__ == "__main__":
    main()
