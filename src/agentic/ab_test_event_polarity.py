"""4-arm A/B test on the 5%/15d model:

  CONTROL:  baseline 140 features (no events)
  +PATH1:   CONTROL + 6 price-reaction polarity features (evp_px_pol_*)
  +PATH3:   CONTROL + 6 refined-regex-text polarity features (evp_text_pol_*)
  +BOTH:    CONTROL + all 12 polarity features

Same target (P(high reaches +5% in 15d)), same walk-forward 2024+2025 OOS,
same isotonic calibration. Only feature set varies.

Compares: max P, top-50 avg P, top-50 avg ann EV, OOS calibration band hit
rate, feature importance (does the model actually USE the new features?).
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
EVP = ROOT / "data/derived/event_polarity_features.parquet"
OUT_REPORT = ROOT / "reports/ab_test_event_polarity.md"

H = 15
THRESHOLD = 0.05
TARGET_PCT = 0.05
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

PATH1_FEATS = ["evp_px_pol_sum_5d", "evp_px_pol_pos_n_5d", "evp_px_pol_neg_n_5d",
               "evp_px_pol_sum_30d", "evp_px_pol_pos_n_30d", "evp_px_pol_neg_n_30d"]
PATH3_FEATS = ["evp_text_pol_sum_5d", "evp_text_pol_pos_n_5d", "evp_text_pol_neg_n_5d",
               "evp_text_pol_sum_30d", "evp_text_pol_pos_n_30d", "evp_text_pol_neg_n_30d"]


def build_panel() -> pd.DataFrame:
    print("loading prices ...")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

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
    df = df.merge(ex[["symbol", "trade_date"] + extra_cols],
                  on=["symbol", "trade_date"], how="left")
    extra_present = [c for c in extra_cols if c in df.columns]
    for c in extra_present:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)
    print(f"  joined {len(extra_present)} non-event engineered features")
    return df, extra_present


def join_polarity(df: pd.DataFrame) -> pd.DataFrame:
    if not EVP.exists():
        print("  WARN: event_polarity_features.parquet missing")
        return df
    ev = pd.read_parquet(EVP)
    ev["trade_date"] = pd.to_datetime(ev["trade_date"])
    df = df.merge(ev, on=["symbol", "trade_date"], how="left")
    for c in PATH1_FEATS + PATH3_FEATS:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    print(f"  joined {len(PATH1_FEATS)+len(PATH3_FEATS)} polarity features")
    return df


def train_eval(df: pd.DataFrame, feats: list[str], label: str) -> dict:
    print(f"\n  [{label}] feats={len(feats)}")
    labeled = df[df["target"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]

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
    oof["score_cal"] = iso.transform(oof["score_raw"])

    band = oof[(oof["score_cal"] >= 0.50) & (oof["score_cal"] < 0.70)]
    cal_check = {
        "n_band": len(band),
        "actual_hit": float(band["target"].mean()) if len(band) else 0,
        "model_avg": float(band["score_cal"].mean()) if len(band) else 0,
    }

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

    lgb_imp = pd.DataFrame({"feature": feats, "gain": lgbm_full.feature_importances_})

    return {
        "label": label,
        "today": today[["symbol", "close", "adv_20d_cr", "score_cal"]].copy(),
        "n_features": len(feats),
        "calibration": cal_check,
        "feat_importance": lgb_imp,
    }


def main() -> None:
    print("== ab_test_event_polarity (4-arm: control / +path1 / +path3 / +both) ==\n")
    panel = build_panel()
    panel, extra_cols = join_extras(panel)
    panel = join_polarity(panel)
    panel_full = panel.dropna(subset=BASE_FEATS + extra_cols).copy()

    feats_control = BASE_FEATS + extra_cols
    feats_path1   = feats_control + PATH1_FEATS
    feats_path3   = feats_control + PATH3_FEATS
    feats_both    = feats_control + PATH1_FEATS + PATH3_FEATS

    arms = []
    arms.append(train_eval(panel_full, feats_control, "CONTROL"))
    arms.append(train_eval(panel_full, feats_path1,   "+PATH1 (price-reaction polarity)"))
    arms.append(train_eval(panel_full, feats_path3,   "+PATH3 (refined-regex polarity)"))
    arms.append(train_eval(panel_full, feats_both,    "+BOTH"))

    # Headline
    print("\n=== HEADLINE ===")
    rows = []
    for a in arms:
        t = a["today"]
        top50 = t.nlargest(50, "score_cal")
        ev_top50 = top50["score_cal"] * TARGET_PCT + (1 - top50["score_cal"]) * SL
        ann_ev_top50 = ((1 + ev_top50).clip(lower=0.01) ** TURNS - 1).mean()
        rows.append({
            "arm": a["label"],
            "n_feat": a["n_features"],
            "max_P": float(t["score_cal"].max()),
            "top50_avg_P": float(top50["score_cal"].mean()),
            "top50_avg_annEV": float(ann_ev_top50),
            "cal_n": a["calibration"]["n_band"],
            "cal_actual_hit": a["calibration"]["actual_hit"],
            "cal_model_avg": a["calibration"]["model_avg"],
        })
    df_arms = pd.DataFrame(rows)
    print(df_arms.to_string(index=False, float_format='%.4f'))
    print()

    # Δ vs control
    ctrl = df_arms.iloc[0]
    print("=== Δ vs CONTROL ===")
    for _, r in df_arms.iloc[1:].iterrows():
        print(f"  {r['arm']:<40}  ΔmaxP={r['max_P']-ctrl['max_P']:+.4f}  "
              f"Δtop50_P={r['top50_avg_P']-ctrl['top50_avg_P']:+.4f}  "
              f"Δtop50_annEV={(r['top50_avg_annEV']-ctrl['top50_avg_annEV'])*100:+.2f}pp")

    # Feature importance for the BOTH model — does it actually USE polarity features?
    print("\n=== TOP 20 FEATURE IMPORTANCE in +BOTH model ===")
    fi = arms[3]["feat_importance"].sort_values("gain", ascending=False).head(20)
    print(fi.to_string(index=False))
    print()
    # specifically polarity-feature ranks
    fi_polarity = arms[3]["feat_importance"][arms[3]["feat_importance"]["feature"].str.startswith("evp_")]
    print("=== POLARITY FEATURES specifically (gain ranking out of all features) ===")
    fi_all_sorted = arms[3]["feat_importance"].sort_values("gain", ascending=False).reset_index(drop=True)
    fi_all_sorted["rank"] = range(1, len(fi_all_sorted)+1)
    print(fi_all_sorted[fi_all_sorted["feature"].str.startswith("evp_")].to_string(index=False))

    # Verdict
    print("\n=== VERDICT ===")
    delta_path1 = df_arms.iloc[1]["top50_avg_P"] - ctrl["top50_avg_P"]
    delta_path3 = df_arms.iloc[2]["top50_avg_P"] - ctrl["top50_avg_P"]
    delta_both  = df_arms.iloc[3]["top50_avg_P"] - ctrl["top50_avg_P"]
    print(f"  Δtop50_P PATH1: {delta_path1:+.4f}  PATH3: {delta_path3:+.4f}  BOTH: {delta_both:+.4f}")
    if max(delta_path1, delta_path3, delta_both) >= 0.005:
        print(f"  → KEEP — at least one arm shows top-50 P lift >= 0.005")
    elif min(delta_path1, delta_path3, delta_both) <= -0.005:
        print(f"  → DROP — all arms regress meaningfully")
    else:
        print(f"  → INCONCLUSIVE — Δ within ±0.005, no clean signal")

    # Markdown report
    md = ["# A/B test: polarity-aware event features on 5%/15d", "",
          "| Arm | n features | max P | top-50 avg P | top-50 avg ann EV | OOS cal band hit rate |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in df_arms.iterrows():
        md.append(f"| {r['arm']} | {int(r['n_feat'])} | {r['max_P']:.4f} | {r['top50_avg_P']:.4f} | {r['top50_avg_annEV']*100:+.2f}% | {r['cal_actual_hit']:.3f} (n={int(r['cal_n']):,}) |")
    md.append("")
    md.append("## Δ vs CONTROL")
    md.append("")
    md.append("| Arm | Δ max P | Δ top-50 avg P | Δ top-50 avg ann EV |")
    md.append("|---|---:|---:|---:|")
    for _, r in df_arms.iloc[1:].iterrows():
        md.append(f"| {r['arm']} | {r['max_P']-ctrl['max_P']:+.4f} | {r['top50_avg_P']-ctrl['top50_avg_P']:+.4f} | {(r['top50_avg_annEV']-ctrl['top50_avg_annEV'])*100:+.2f}pp |")
    md.append("")
    md.append("## Polarity features in +BOTH model — gain rank out of 152")
    md.append("")
    md.append("| feature | gain | rank |")
    md.append("|---|---:|---:|")
    for _, r in fi_all_sorted[fi_all_sorted["feature"].str.startswith("evp_")].iterrows():
        md.append(f"| `{r['feature']}` | {int(r['gain'])} | {int(r['rank'])} |")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
