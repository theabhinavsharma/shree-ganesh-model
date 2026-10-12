"""A/B test: does adding 11.3y of catalyst/event features lift the 5%/15d model?

Two arms, identical except for the feature set:
  CONTROL:    BASE_FEATS + macro_panel + sector_panel  (~140 features)
  TREATMENT:  CONTROL + catalyst_features.parquet (16 announcement-type + 1 insider)

Same target: P(high reaches +5% within 15 trading days).
Same calibrator: isotonic on walk-forward 2024-2025 OOS predictions.

Compare:
  - Max calibrated P
  - Top-50 average P
  - Top-50 average per-trade EV
  - Top-50 average annualized EV (16.8 turns/yr)
  - Symbol-level Δ score (treatment - control) for any symbol in either's top 50

Output:
  reports/ab_test_event_features.md
  data/derived/ab_test_event_features.parquet
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
CAT = ROOT / "data/derived/catalyst_features.parquet"
OUT_REPORT = ROOT / "reports/ab_test_event_features.md"
OUT_PARQUET = ROOT / "data/derived/ab_test_event_features.parquet"

H = 15
THRESHOLD = 0.05
TARGET = 0.05  # +5%
SL = -0.03
TURNS = 252 / H

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

# Catalyst features that are built and HAVE data (excluding all-zero columns)
CAT_FEATS = [
    "ann_5d_count", "ann_30d_count",
    "catalyst_score_5d", "catalyst_score_30d",
    "ann_order_5d", "ann_order_30d",
    "ann_result_5d", "ann_capex_30d", "ann_fundraise_30d",
    "ann_buyback_30d", "ann_ma_30d", "ann_regulatory_30d",
    "ann_dividend_30d", "ann_bonussplit_30d", "ann_rating_30d",
    "ann_guidance_30d",
    "insider_stake_delta_60d",
]


def build_panel() -> pd.DataFrame:
    print("loading prices ...")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    # forward 15d high
    shifts = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts.max(axis=1)
    df["target"] = (df["fwd_high_max"] / df["close"] - 1 >= THRESHOLD).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, "target"] = pd.NA

    df["dist_sma20"]  = df["close"] / df["sma_20"]  - 1
    df["dist_sma50"]  = df["close"] / df["sma_50"]  - 1
    df["dist_sma200"] = df["close"] / df["sma_200"] - 1
    df["above_50dma"]  = (df["close"] > df["sma_50"]).astype(int)
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
        return df, []
    ex = pd.read_parquet(EXTRA)
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])
    extra_cols = [c for c in ex.columns
                  if c not in ("symbol", "trade_date")
                  and c.startswith(SAFE_EXTRA_PREFIXES)
                  and pd.api.types.is_numeric_dtype(ex[c])]
    if not extra_cols:
        return df, []
    df = df.merge(ex[["symbol", "trade_date"] + extra_cols],
                  on=["symbol", "trade_date"], how="left")
    extra_present = [c for c in extra_cols if c in df.columns]
    for c in extra_present:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)
    print(f"  joined {len(extra_present)} non-event engineered features")
    return df, extra_present


def join_catalysts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the 17 catalyst/event features (11.3y history)."""
    if not CAT.exists():
        return df, []
    cat = pd.read_parquet(CAT)
    cat["trade_date"] = pd.to_datetime(cat["trade_date"])
    keep = [c for c in CAT_FEATS if c in cat.columns]
    df = df.merge(cat[["symbol", "trade_date"] + keep], on=["symbol", "trade_date"], how="left")
    for c in keep:
        df[c] = df[c].fillna(0)  # absence of an event = 0
    print(f"  joined {len(keep)} catalyst features")
    return df, keep


def train_eval(df: pd.DataFrame, feats: list[str], label: str) -> dict:
    """Walk-forward train 2024 + 2025; calibrate; predict today; return analysis dict."""
    print(f"\n  [{label}] feats={len(feats)}")
    labeled = df[df["target"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    base_rate = labeled["target"].mean()
    print(f"    labeled n={len(labeled):,}  base_rate={base_rate:.3f}")

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
        lgbm.fit(tr[feats], tr["target"].astype(int))
        xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42,
                                  verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        xgbm.fit(tr[feats], tr["target"].astype(int))
        p = 0.5 * lgbm.predict_proba(te[feats])[:, 1] + 0.5 * xgbm.predict_proba(te[feats])[:, 1]
        te["score_raw"] = p
        oof_rows.append(te[["symbol", "trade_date", "target", "score_raw"]])

    oof = pd.concat(oof_rows, ignore_index=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["score_raw"], oof["target"].astype(int))

    # OOS calibration check (band 0.50-0.65 → actual hit rate)
    oof["score_cal"] = iso.transform(oof["score_raw"])
    band = oof[(oof["score_cal"] >= 0.50) & (oof["score_cal"] < 0.70)]
    cal_check = {
        "band_0.50-0.70_n": len(band),
        "band_0.50-0.70_hit_rate": float(band["target"].mean()) if len(band) else 0,
        "band_0.50-0.70_model_avg": float(band["score_cal"].mean()) if len(band) else 0,
    }

    # Train final model on all years <= 2025; predict on TODAY
    full_tr = labeled[labeled["year"] < 2026]
    lgbm_full = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                                    min_child_samples=200, feature_fraction=0.85,
                                    bagging_fraction=0.85, bagging_freq=5,
                                    random_state=42, verbose=-1, n_jobs=-1)
    lgbm_full.fit(full_tr[feats], full_tr["target"].astype(int))
    xgbm_full = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                                   subsample=0.85, colsample_bytree=0.85, random_state=42,
                                   verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    xgbm_full.fit(full_tr[feats], full_tr["target"].astype(int))

    today_dt = df["trade_date"].max()
    today = df[(df["trade_date"] == today_dt) & (df["adv_20d_cr"] >= 1.0)].copy()
    p_today = 0.5 * lgbm_full.predict_proba(today[feats])[:, 1] + 0.5 * xgbm_full.predict_proba(today[feats])[:, 1]
    today["score_cal"] = iso.transform(p_today)

    # feature importance (LGB) for inspection
    lgb_imp = pd.DataFrame({"feature": feats, "gain": lgbm_full.feature_importances_})
    lgb_imp = lgb_imp.sort_values("gain", ascending=False).head(15)

    return {
        "label": label,
        "today_predictions": today[["symbol", "close", "adv_20d_cr", "score_cal"]].copy(),
        "n_features": len(feats),
        "base_rate": float(base_rate),
        "calibration_check": cal_check,
        "top_feat_importance": lgb_imp,
    }


def main() -> None:
    print("== ab_test_event_features (5%/15d, control vs +catalysts) ==\n")
    panel = build_panel()
    panel, extra_cols = join_extras(panel)
    panel, cat_cols = join_catalysts(panel)
    print(f"\npanel: {len(panel):,} rows")
    print(f"  control feats: {len(BASE_FEATS) + len(extra_cols)}")
    print(f"  treatment feats: {len(BASE_FEATS) + len(extra_cols) + len(cat_cols)} (+{len(cat_cols)} catalyst)")

    panel_full = panel.dropna(subset=BASE_FEATS + extra_cols).copy()
    # for treatment, also need cat cols filled (already 0-filled in join)
    panel_full2 = panel_full.copy()

    control_res = train_eval(panel_full,  BASE_FEATS + extra_cols,             label="CONTROL")
    treat_res   = train_eval(panel_full2, BASE_FEATS + extra_cols + cat_cols,  label="TREATMENT (+catalysts)")

    # Merge for symbol-level comparison
    a = control_res["today_predictions"].rename(columns={"score_cal": "score_control"})
    b = treat_res["today_predictions"].rename(columns={"score_cal": "score_treatment"})
    both = a.merge(b[["symbol", "score_treatment"]], on="symbol", how="inner")
    both["delta"] = both["score_treatment"] - both["score_control"]

    # EVs
    both["EV_control"]   = both["score_control"]   * TARGET + (1 - both["score_control"])   * SL
    both["EV_treatment"] = both["score_treatment"] * TARGET + (1 - both["score_treatment"]) * SL
    both["ann_EV_control"]   = (1 + both["EV_control"]).clip(lower=0.01)   ** TURNS - 1
    both["ann_EV_treatment"] = (1 + both["EV_treatment"]).clip(lower=0.01) ** TURNS - 1

    # Headline
    print("\n=== HEADLINE A/B (today's universe, ADV>=1cr) ===")
    print(f"  Max P:        control = {both['score_control'].max():.3f}  treatment = {both['score_treatment'].max():.3f}  Δ = {both['score_treatment'].max() - both['score_control'].max():+.3f}")
    print(f"  Mean P:       control = {both['score_control'].mean():.3f}  treatment = {both['score_treatment'].mean():.3f}")
    top50_c = both.nlargest(50, 'score_control')
    top50_t = both.nlargest(50, 'score_treatment')
    print(f"  Top-50 avg P: control = {top50_c['score_control'].mean():.3f}  treatment = {top50_t['score_treatment'].mean():.3f}  Δ = {top50_t['score_treatment'].mean() - top50_c['score_control'].mean():+.3f}")
    print(f"  Top-50 avg ann EV: control = {top50_c['ann_EV_control'].mean()*100:+.1f}%  treatment = {top50_t['ann_EV_treatment'].mean()*100:+.1f}%")

    # OOS calibration check
    print(f"\n=== OOS calibration check (score_cal in [0.50, 0.70]) ===")
    cc = control_res['calibration_check']; tc = treat_res['calibration_check']
    print(f"  CONTROL  : n={cc['band_0.50-0.70_n']:>6}  actual hit = {cc['band_0.50-0.70_hit_rate']:.3f}  model avg = {cc['band_0.50-0.70_model_avg']:.3f}")
    print(f"  TREATMENT: n={tc['band_0.50-0.70_n']:>6}  actual hit = {tc['band_0.50-0.70_hit_rate']:.3f}  model avg = {tc['band_0.50-0.70_model_avg']:.3f}")

    # Top-15 by treatment score
    print("\n=== TOP 15 by TREATMENT (with-events) score ===")
    top = both.nlargest(15, 'score_treatment')
    print(top[['symbol','close','score_control','score_treatment','delta','ann_EV_treatment']].to_string(index=False, float_format='%.3f'))

    # Names where catalysts changed verdict (treatment crosses 0.55 but control didn't, OR reverse)
    print("\n=== NAMES WHERE CATALYSTS FLIP VERDICT (gain crossing 0.55 in either direction) ===")
    flips_up = both[(both['score_treatment'] >= 0.55) & (both['score_control'] < 0.55)].sort_values('delta', ascending=False)
    flips_dn = both[(both['score_control'] >= 0.55) & (both['score_treatment'] < 0.55)].sort_values('delta')
    print(f"  CATALYSTS LIFT INTO 0.55+ ({len(flips_up)} names):")
    print(flips_up[['symbol','close','score_control','score_treatment','delta','ann_EV_treatment']].head(10).to_string(index=False, float_format='%.3f'))
    print(f"\n  CATALYSTS DROP FROM 0.55+ ({len(flips_dn)} names):")
    print(flips_dn[['symbol','close','score_control','score_treatment','delta','ann_EV_control']].head(10).to_string(index=False, float_format='%.3f'))

    # Feature importance
    print("\n=== TOP FEATURE IMPORTANCE (TREATMENT model) ===")
    print(treat_res['top_feat_importance'].to_string(index=False))

    # Verdict
    print("\n=== VERDICT ===")
    delta_top50 = top50_t['score_treatment'].mean() - top50_c['score_control'].mean()
    delta_max   = both['score_treatment'].max() - both['score_control'].max()
    if delta_top50 >= 0.01:
        verdict = "KEEP -- catalyst features add real lift to top-50 average P"
    elif abs(delta_top50) < 0.005 and abs(delta_max) < 0.005:
        verdict = "DROP -- catalyst features add no measurable lift; keep the simpler model"
    else:
        verdict = "WATCHLIST -- mixed (top-50 vs max diverge); rerun with longer holdout"
    print(f"  {verdict}")

    # save parquet
    both.to_parquet(OUT_PARQUET, index=False)

    # markdown
    md = [f"# A/B test: catalyst event features on 5%/15d model", "",
          f"**Question:** does adding 16 announcement-type features + 1 insider-stake feature (11.3y history) lift the 5%/15d uptick model?", "",
          f"**Method:** identical walk-forward 2024 + 2025 OOS; identical isotonic calibration on the same OOF; only difference is the feature set.", "",
          f"## Headline", "",
          f"| Metric | CONTROL | TREATMENT (+catalysts) | Δ |",
          f"|---|---:|---:|---:|",
          f"| n features | {control_res['n_features']} | {treat_res['n_features']} | +{len(cat_cols)} |",
          f"| Max P | {both['score_control'].max():.3f} | {both['score_treatment'].max():.3f} | {both['score_treatment'].max()-both['score_control'].max():+.3f} |",
          f"| Top-50 avg P | {top50_c['score_control'].mean():.3f} | {top50_t['score_treatment'].mean():.3f} | {delta_top50:+.3f} |",
          f"| Top-50 avg ann EV | {top50_c['ann_EV_control'].mean()*100:+.1f}% | {top50_t['ann_EV_treatment'].mean()*100:+.1f}% | {(top50_t['ann_EV_treatment'].mean() - top50_c['ann_EV_control'].mean())*100:+.1f}pp |",
          f"| OOS calibration band 0.50-0.70 hit rate | {cc['band_0.50-0.70_hit_rate']:.3f} | {tc['band_0.50-0.70_hit_rate']:.3f} | — |",
          "",
          f"## Verdict",
          "",
          f"**{verdict}**",
          "",
          f"## Top 15 by treatment score (with catalysts)",
          "",
          f"| Symbol | Close | Score control | Score treatment | Δ | Ann EV |",
          f"|---|---:|---:|---:|---:|---:|"]
    for _, r in top.iterrows():
        md.append(f"| {r['symbol']} | ₹{r['close']:.2f} | {r['score_control']:.3f} | {r['score_treatment']:.3f} | {r['delta']:+.3f} | {r['ann_EV_treatment']*100:+.1f}% |")

    md.append(f"")
    md.append(f"## Names where catalysts FLIP the verdict")
    md.append(f"")
    md.append(f"### Catalysts LIFT into 0.55+ ({len(flips_up)} names) — events that the price-only model misses")
    md.append(f"")
    md.append(f"| Symbol | Close | Score control | Score treatment | Δ | Ann EV |")
    md.append(f"|---|---:|---:|---:|---:|---:|")
    for _, r in flips_up.head(15).iterrows():
        md.append(f"| {r['symbol']} | ₹{r['close']:.2f} | {r['score_control']:.3f} | {r['score_treatment']:.3f} | {r['delta']:+.3f} | {r['ann_EV_treatment']*100:+.1f}% |")
    md.append(f"")
    md.append(f"### Catalysts PENALIZE (drop below 0.55) — events that overrule a price-only buy signal")
    md.append(f"")
    md.append(f"| Symbol | Close | Score control | Score treatment | Δ | Control Ann EV |")
    md.append(f"|---|---:|---:|---:|---:|---:|")
    for _, r in flips_dn.head(15).iterrows():
        md.append(f"| {r['symbol']} | ₹{r['close']:.2f} | {r['score_control']:.3f} | {r['score_treatment']:.3f} | {r['delta']:+.3f} | {r['ann_EV_control']*100:+.1f}% |")

    md.append(f"")
    md.append(f"## Top-15 LightGBM feature importance (TREATMENT model)")
    md.append(f"")
    md.append(f"| feature | gain |")
    md.append(f"|---|---:|")
    for _, r in treat_res['top_feat_importance'].iterrows():
        md.append(f"| `{r['feature']}` | {r['gain']:.0f} |")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"wrote {OUT_PARQUET}")


if __name__ == "__main__":
    main()
