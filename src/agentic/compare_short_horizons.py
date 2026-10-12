"""Head-to-head: 10%/30d vs 5%/15d short-horizon trade.

Trains BOTH targets walk-forward 2024-2025, calibrates with isotonic, ranks
today's universe by per-trade EV and ANNUALIZED EV (since faster horizons
turn capital more times per year).

Decision metrics:
  • EV per trade = P × target − (1−P) × |SL|
  • Annualized EV = (1 + EV)^(turns/yr) − 1
    where turns/yr = 252/horizon (assumes no overlap)
  • Compare under common SL = -3% (15d) vs -5% (30d) — proportional to target

Output:
  reports/compare_short_horizons.md
  data/derived/compare_short_horizons.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Code/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
OUT_REPORT = ROOT / "reports/compare_short_horizons.md"
OUT_PARQUET = ROOT / "data/derived/compare_short_horizons.parquet"

# Two contestants
TARGETS = [
    {"name": "5pct_15d",  "horizon": 15, "threshold": 0.05, "sl": -0.03},
    {"name": "10pct_30d", "horizon": 30, "threshold": 0.10, "sl": -0.05},
]

# Same feature set as find_high_conviction (post-leakage-quarantine)
BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]

SAFE_EXTRA_PREFIXES = (
    "alpha_", "vol_", "rv_",
    "usdinr", "eurinr", "gbpinr", "jpyinr",
    "wiki_", "spx", "us10y", "dxy", "brent", "gold",
    "macro_", "sec_",
)


def build_panel() -> pd.DataFrame:
    print("loading prices …")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    # Build forward-window highs for each horizon (CORRECT direction)
    for tgt in TARGETS:
        H = tgt["horizon"]
        shifts = pd.concat(
            [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
            axis=1,
        )
        df[f"fwd_high_max_{H}"] = shifts.max(axis=1)
        df[f"target_{tgt['name']}"] = (
            df[f"fwd_high_max_{H}"] / df["close"] - 1 >= tgt["threshold"]
        ).astype(int)
        complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
        df.loc[~complete, f"target_{tgt['name']}"] = pd.NA

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
    ).reset_index()
    df = df.merge(mkt, on="trade_date", how="left")
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())
    return df


def join_extras(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not EXTRA.exists():
        return df, BASE_FEATS
    ex = pd.read_parquet(EXTRA)
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])
    extra_cols = [c for c in ex.columns
                  if c not in ("symbol", "trade_date")
                  and c.startswith(SAFE_EXTRA_PREFIXES)
                  and pd.api.types.is_numeric_dtype(ex[c])]
    if not extra_cols:
        return df, BASE_FEATS
    df = df.merge(ex[["symbol", "trade_date"] + extra_cols],
                  on=["symbol", "trade_date"], how="left")
    # only sanitize columns that actually came through the merge
    extra_present = [c for c in extra_cols if c in df.columns]
    for c in extra_present:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)
    print(f"  joined {len(extra_present)} engineered features")
    return df, BASE_FEATS + extra_present


def train_target(df: pd.DataFrame, target_col: str, FEATS: list[str]) -> tuple[float, np.ndarray, np.ndarray]:
    """Walk-forward 2024-2025; calibrate with isotonic. Return (today_max_score, today_scores, today_idx)."""
    labeled = df[df[target_col].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    base_rate = labeled[target_col].mean()
    print(f"  base_rate={base_rate:.3f}  labeled n={len(labeled):,}")

    # walk-forward
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
        p = 0.5 * lgbm.predict_proba(te[FEATS])[:, 1] + 0.5 * xgbm.predict_proba(te[FEATS])[:, 1]
        te["score_raw"] = p
        oof_rows.append(te[["symbol", "trade_date", target_col, "score_raw"]])

    if not oof_rows:
        return None
    oof = pd.concat(oof_rows, ignore_index=True)

    # calibrate
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["score_raw"], oof[target_col].astype(int))
    oof["score_cal"] = iso.transform(oof["score_raw"])

    # train final model on all data including 2024 (use 2024 for cal-on-2025 already happened)
    full_tr = labeled[labeled["year"] < 2026]  # everything labeled
    lgbm_full = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                                    min_child_samples=200, feature_fraction=0.85,
                                    bagging_fraction=0.85, bagging_freq=5,
                                    random_state=42, verbose=-1, n_jobs=-1)
    lgbm_full.fit(full_tr[FEATS], full_tr[target_col].astype(int))
    xgbm_full = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                   subsample=0.85, colsample_bytree=0.85, random_state=42,
                                   verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    xgbm_full.fit(full_tr[FEATS], full_tr[target_col].astype(int))

    # predict on TODAY
    today_dt = df["trade_date"].max()
    today = df[(df["trade_date"] == today_dt) & (df["adv_20d_cr"] >= 1.0)].copy()
    p_today = 0.5 * lgbm_full.predict_proba(today[FEATS])[:, 1] + 0.5 * xgbm_full.predict_proba(today[FEATS])[:, 1]
    today["score_cal"] = iso.transform(p_today)
    return today[["symbol", "close", "adv_20d_cr", "score_cal"]].copy()


def main() -> None:
    print("== compare_short_horizons (10%/30d vs 5%/15d) ==\n")
    panel = build_panel()
    panel, FEATS = join_extras(panel)
    panel = panel.dropna(subset=FEATS).copy()
    print(f"  panel: {len(panel):,} rows, {len(FEATS)} features\n")

    results = {}
    for tgt in TARGETS:
        print(f"[{tgt['name']}] training (target +{tgt['threshold']*100:.0f}% in {tgt['horizon']}d)")
        col = f"target_{tgt['name']}"
        results[tgt["name"]] = train_target(panel, col, FEATS)

    # merge predictions
    a = results["5pct_15d"].rename(columns={"score_cal": "score_5pct_15d"})
    b = results["10pct_30d"].rename(columns={"score_cal": "score_10pct_30d"})
    merged = a.merge(b[["symbol", "score_10pct_30d"]], on="symbol", how="outer")

    # Compute EV per trade and annualized EV
    SL_15d = -0.03   # -3% SL on 15d trade
    SL_30d = -0.05   # -5% SL on 30d trade
    TARGET_15d = 0.05
    TARGET_30d = 0.10

    merged["EV_5pct_15d"]  = merged["score_5pct_15d"]  * TARGET_15d + (1 - merged["score_5pct_15d"])  * SL_15d
    merged["EV_10pct_30d"] = merged["score_10pct_30d"] * TARGET_30d + (1 - merged["score_10pct_30d"]) * SL_30d

    # Annualized: 252/15 ≈ 16.8 turns; 252/30 ≈ 8.4 turns (assumes serial non-overlap)
    TURNS_15d = 252 / 15
    TURNS_30d = 252 / 30
    merged["EV_ann_5pct_15d"]  = (1 + merged["EV_5pct_15d"]).clip(lower=0.01)  ** TURNS_15d - 1
    merged["EV_ann_10pct_30d"] = (1 + merged["EV_10pct_30d"]).clip(lower=0.01) ** TURNS_30d - 1

    # Best of the two per stock (which trade is better here?)
    merged["best_strategy"] = merged.apply(
        lambda r: "5pct_15d" if (r["EV_ann_5pct_15d"] > r["EV_ann_10pct_30d"]) else "10pct_30d",
        axis=1
    )
    merged["best_EV_ann"] = merged[["EV_ann_5pct_15d", "EV_ann_10pct_30d"]].max(axis=1)

    # Print summary
    print("\n=== HEAD-TO-HEAD on today's universe ===\n")
    print(f"  5%/15d:    max P = {merged['score_5pct_15d'].max():.3f}  "
          f"max per-trade EV = {merged['EV_5pct_15d'].max():.4f}  "
          f"max ann EV = {merged['EV_ann_5pct_15d'].max():+.1%}")
    print(f"  10%/30d:   max P = {merged['score_10pct_30d'].max():.3f}  "
          f"max per-trade EV = {merged['EV_10pct_30d'].max():.4f}  "
          f"max ann EV = {merged['EV_ann_10pct_30d'].max():+.1%}")
    print()

    # P required for various EV bars
    print("=== Required P(up) for given EV per-trade ===")
    print(f"  5%/15d, SL=-3%:")
    for ev_target in [0.02, 0.03, 0.04, 0.05]:
        p_req = (ev_target - SL_15d) / (TARGET_15d - SL_15d)
        print(f"    per-trade EV >= {ev_target:+.2%} requires P >= {p_req:.3f}")
    print(f"  10%/30d, SL=-5%:")
    for ev_target in [0.04, 0.05, 0.06, 0.08]:
        p_req = (ev_target - SL_30d) / (TARGET_30d - SL_30d)
        print(f"    per-trade EV >= {ev_target:+.2%} requires P >= {p_req:.3f}")
    print()

    # Top 15 by best annualized EV
    merged_sorted = merged.sort_values("best_EV_ann", ascending=False)
    print("=== TOP 15 by best annualized EV (whichever strategy is better per stock) ===\n")
    cols = ["symbol", "close", "adv_20d_cr",
            "score_5pct_15d", "EV_5pct_15d", "EV_ann_5pct_15d",
            "score_10pct_30d", "EV_10pct_30d", "EV_ann_10pct_30d",
            "best_strategy", "best_EV_ann"]
    print(merged_sorted[cols].head(15).to_string(index=False, float_format='%.3f'))
    print()

    # which strategy wins more often in the top tail
    top100 = merged_sorted.head(100)
    win5 = (top100["best_strategy"] == "5pct_15d").sum()
    win10 = (top100["best_strategy"] == "10pct_30d").sum()
    print(f"Among top 100: 5%/15d wins on {win5} names, 10%/30d wins on {win10} names")
    print()

    # Average EVs across deployable tier (top 50 by P for each strategy)
    top50_5 = merged.nlargest(50, "score_5pct_15d")
    top50_10 = merged.nlargest(50, "score_10pct_30d")
    print(f"Top-50 by P score:")
    print(f"  5%/15d:  avg P={top50_5['score_5pct_15d'].mean():.3f}  "
          f"avg per-trade EV={top50_5['EV_5pct_15d'].mean():+.3%}  "
          f"avg ann EV={top50_5['EV_ann_5pct_15d'].mean():+.1%}")
    print(f"  10%/30d: avg P={top50_10['score_10pct_30d'].mean():.3f}  "
          f"avg per-trade EV={top50_10['EV_10pct_30d'].mean():+.3%}  "
          f"avg ann EV={top50_10['EV_ann_10pct_30d'].mean():+.1%}")

    # Save parquet
    merged.to_parquet(OUT_PARQUET, index=False)

    # Markdown report
    md = ["# 10%/30d vs 5%/15d head-to-head", "",
          f"**Comparison framework:** per-trade E[return] = P × target − (1−P) × |SL|, annualized at 252/horizon turns.", "",
          "| | 5%/15d | 10%/30d |",
          "|---|---:|---:|",
          f"| Target | +5% | +10% |",
          f"| Stop-loss | -3% | -5% |",
          f"| Hold (trading days) | 15 | 30 |",
          f"| Turns/year (no overlap) | 16.8 | 8.4 |",
          f"| Max P today | {merged['score_5pct_15d'].max():.3f} | {merged['score_10pct_30d'].max():.3f} |",
          f"| Max per-trade EV | {merged['EV_5pct_15d'].max():+.3%} | {merged['EV_10pct_30d'].max():+.3%} |",
          f"| Max annualized EV | {merged['EV_ann_5pct_15d'].max():+.1%} | {merged['EV_ann_10pct_30d'].max():+.1%} |",
          ""]
    md.append("## Top 20 names by best annualized EV")
    md.append("")
    md.append("| Symbol | Close | ADV cr | P(5%/15d) | EV/trade 15d | Ann 15d | P(10%/30d) | EV/trade 30d | Ann 30d | Winner |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in merged_sorted.head(20).iterrows():
        md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | {r['adv_20d_cr']:.1f} | "
                  f"{r['score_5pct_15d']:.3f} | {r['EV_5pct_15d']:+.3%} | {r['EV_ann_5pct_15d']:+.1%} | "
                  f"{r['score_10pct_30d']:.3f} | {r['EV_10pct_30d']:+.3%} | {r['EV_ann_10pct_30d']:+.1%} | "
                  f"`{r['best_strategy']}` |")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"wrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()
