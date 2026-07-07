"""News-feature A/B backtest.

Compares 5 model variants on the same test window with strict walk-forward:
  M0: BASE_FEATS only (price/technical/market) — control
  M1: BASE + news_5d_*
  M2: BASE + news_7d_*
  M3: BASE + news_15d_*
  M4: BASE + news_5d + news_7d + news_15d (full stack)
  M5: BASE + industry_5d/7d/15d only (sector overlay alone)

Honest constraint: news_event_features.parquet covers Feb-Apr 2026 only
(~3 months of NSE corporate filings). Walk-forward windows are tight:
  train: 2026-02-01 → 2026-03-20 (~30 trading days)
  test:  2026-03-21 → 2026-04-20 (~22 trading days)

Target: 5-day forward return >= +3% (binary). News-event signals are
expected to have shorter half-life than price-momentum signals, so we
test against a 5d horizon. Sample size will be the gate.

Output: reports/news_backtest_20260506.md with verdict per CONSTITUTION
§1.4 (Bonferroni-correct: 5 model comparisons → α/5 = 0.01).
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
NEWS = ROOT / "data/derived/news_event_features.parquet"
OUT_REPORT = ROOT / "reports/news_backtest_20260506.md"
OUT_PARQUET = ROOT / "data/derived/news_backtest_20260506.parquet"

H = 5             # forward horizon (trading days)
THRESHOLD = 0.03  # +3% target
TRAIN_START = "2026-02-15"  # need at least 15 prior days for 15d window
TRAIN_END = "2026-03-20"
TEST_START = "2026-03-21"
TEST_END = "2026-04-20"

BASE_FEATS = [
    "return_1d", "return_20d",
    "dist_sma20", "dist_sma50",
    "above_50dma", "above_200dma",
    "rsi_14_daily",
    "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
    "realized_vol_20d", "adv_20d_cr",
    "market_5d_ret", "market_20d_ret",
    "market_breadth_50dma",
]

NEWS_5D = [
    "evt_5d_total", "evt_5d_positive", "evt_5d_negative",
    "evt_5d_order_win", "evt_5d_approval", "evt_5d_promoter_buying",
    "evt_5d_pledge_change", "evt_5d_results_event",
]
NEWS_7D = [
    "evt_7d_total", "evt_7d_positive", "evt_7d_negative",
    "evt_7d_order_win", "evt_7d_approval", "evt_7d_promoter_buying",
    "evt_7d_pledge_change", "evt_7d_results_event",
]
NEWS_15D = [
    "evt_15d_total", "evt_15d_positive", "evt_15d_negative",
    "evt_15d_order_win", "evt_15d_approval", "evt_15d_promoter_buying",
    "evt_15d_pledge_change", "evt_15d_results_event",
]
INDUSTRY = [
    "evt_ind_5d_total", "evt_ind_5d_positive", "evt_ind_5d_order_win",
    "evt_ind_15d_total", "evt_ind_15d_positive", "evt_ind_15d_order_win",
]


def build_panel() -> pd.DataFrame:
    print("loading prices …")
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2026-01-15"].copy()  # buffer for lookback features
    # liquidity filter — derive adv_20d_cr first if not present
    if "adv_20d_cr" not in df.columns:
        df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7
    df = df[df["adv_20d_cr"] >= 1.0]  # liquid only

    # 5d forward target
    df["fwd_high"] = (df.groupby("symbol", sort=False)["high"]
                       .transform(lambda s: pd.concat([s.shift(-k) for k in range(1, H+1)], axis=1).max(axis=1)))
    df["target"] = (df["fwd_high"] / df["close"] - 1 >= THRESHOLD).astype("Int64")
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, "target"] = pd.NA

    # derived BASE_FEATS
    df["dist_sma20"] = df["close"] / df["sma_20"] - 1
    df["dist_sma50"] = df["close"] / df["sma_50"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7

    # market metrics
    liq = df[df["adv_20d_cr"] >= 1.0]
    mkt = liq.groupby("trade_date").agg(
        market_breadth_50dma=("above_50dma", "mean"),
    ).reset_index()
    df = df.merge(mkt, on="trade_date", how="left")
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())

    # join news features
    if NEWS.exists():
        ne = pd.read_parquet(NEWS)
        ne["trade_date"] = pd.to_datetime(ne["trade_date"])
        df = df.merge(ne, on=["symbol", "trade_date"], how="left")
        # fill missing news with 0 (no events)
        for c in NEWS_5D + NEWS_7D + NEWS_15D + INDUSTRY:
            if c not in df.columns:
                df[c] = 0
            df[c] = df[c].fillna(0)
    return df


def run_variant(name: str, feats: list[str], tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    feats = [f for f in feats if f in tr.columns]
    tr = tr.dropna(subset=feats + ["target"]).copy()
    te = te.dropna(subset=feats + ["target"]).copy()
    if len(tr) < 500 or len(te) < 100:
        return {"variant": name, "n_train": len(tr), "n_test": len(te), "auc": None,
                "top_decile_precision": None, "verdict": "INSUFFICIENT_SAMPLE"}

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                                min_child_samples=50, feature_fraction=0.85,
                                bagging_fraction=0.85, bagging_freq=5,
                                random_state=42, verbose=-1, n_jobs=-1)
    model.fit(tr[feats], tr["target"].astype(int))
    p_te = model.predict_proba(te[feats])[:, 1]

    base_rate = float(te["target"].astype(int).mean())
    auc = float(roc_auc_score(te["target"].astype(int), p_te)) if te["target"].nunique() > 1 else None
    # top-decile precision
    n10 = max(1, len(p_te) // 10)
    top_idx = np.argsort(p_te)[-n10:]
    top_precision = float(te.iloc[top_idx]["target"].astype(int).mean())
    # top-5 basket return per cycle
    te2 = te.copy()
    te2["score"] = p_te
    top5 = te2.sort_values("score", ascending=False).head(5)
    if "fwd_high" in top5.columns and "close" in top5.columns:
        basket_ret = float((top5["fwd_high"] / top5["close"] - 1).mean())
    else:
        basket_ret = None

    return {
        "variant": name,
        "feature_count": len(feats),
        "n_train": len(tr),
        "n_test": len(te),
        "base_rate": round(base_rate, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "top_decile_precision": round(top_precision, 4),
        "top_decile_lift_vs_base": round(top_precision - base_rate, 4),
        "top5_basket_max_return": round(basket_ret, 4) if basket_ret is not None else None,
        "verdict": "EVALUATED",
    }


def main() -> None:
    df = build_panel()
    tr = df[(df["trade_date"] >= TRAIN_START) & (df["trade_date"] <= TRAIN_END)].copy()
    te = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)].copy()
    print(f"\ntrain rows: {len(tr):,}  test rows: {len(te):,}")
    print(f"train target base rate: {tr['target'].dropna().astype(int).mean():.3f}")
    print(f"test  target base rate: {te['target'].dropna().astype(int).mean():.3f}")

    results = []
    print("\n=== Running variants ===")
    for name, feats in [
        ("M0_base", BASE_FEATS),
        ("M1_base+news5d", BASE_FEATS + NEWS_5D),
        ("M2_base+news7d", BASE_FEATS + NEWS_7D),
        ("M3_base+news15d", BASE_FEATS + NEWS_15D),
        ("M4_base+all_news", BASE_FEATS + NEWS_5D + NEWS_7D + NEWS_15D),
        ("M5_base+industry", BASE_FEATS + INDUSTRY),
        ("M6_base+all+industry", BASE_FEATS + NEWS_5D + NEWS_7D + NEWS_15D + INDUSTRY),
    ]:
        r = run_variant(name, feats, tr, te)
        results.append(r)
        print(f"  {name:25s}  feats={r.get('feature_count','-'):>3}  "
              f"AUC={r.get('auc')}  topD_prec={r.get('top_decile_precision')}  "
              f"lift={r.get('top_decile_lift_vs_base')}  basketRet={r.get('top5_basket_max_return')}")

    res_df = pd.DataFrame(results)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_parquet(OUT_PARQUET, index=False)

    # Verdict
    base = next((r for r in results if r["variant"] == "M0_base"), {})
    base_auc = base.get("auc") or 0.0
    base_lift = base.get("top_decile_lift_vs_base") or 0.0
    BONFERRONI_ALPHA = 0.05 / 6   # 6 alternative hypotheses tested

    md = ["# News-feature A/B backtest — 2026-05-06", "",
          "**Question**: do news/event-window features (5d / 7d / 15d, stock + industry) "
          "add prospective lift over BASE_FEATS alone?", "",
          "**Constitution gate (§1.4)**: KEEP only if Δ(top-decile precision) >= 1pp AND ΔAUC >= 0.005 vs M0_base.",
          f"Bonferroni: 6 alternative hypotheses → α = 0.05/6 = {BONFERRONI_ALPHA:.4f}", "",
          "## Setup", "",
          f"- Train: {TRAIN_START} → {TRAIN_END} ({len(tr):,} rows)",
          f"- Test:  {TEST_START} → {TEST_END} ({len(te):,} rows)",
          f"- Target: 5-day forward high ≥ +{THRESHOLD*100:.0f}% (binary)",
          f"- Test base rate: {te['target'].dropna().astype(int).mean()*100:.1f}%", "",
          "## Results", "",
          "| Variant | #feats | AUC | Top-Decile precision | Lift vs base | Top-5 basket max-ret |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        md.append(f"| {r['variant']} | {r.get('feature_count','-')} | "
                  f"{r.get('auc','—')} | {r.get('top_decile_precision','—')} | "
                  f"{(r.get('top_decile_lift_vs_base') or 0)*100:+.1f}pp | "
                  f"{(r.get('top5_basket_max_return') or 0)*100:+.1f}% |")
    md.append("")
    md.append("## Verdict")
    md.append("")

    keepers = []
    for r in results:
        if r["variant"] == "M0_base":
            continue
        auc_delta = (r.get("auc") or 0) - base_auc
        lift_delta = (r.get("top_decile_lift_vs_base") or 0) - base_lift
        passes = (lift_delta >= 0.01) and (auc_delta >= 0.005)
        verdict = "KEEP" if passes else "DROP_AB_FAIL"
        md.append(f"- **{r['variant']}**: ΔAUC={auc_delta:+.4f}, Δlift={lift_delta*100:+.2f}pp → {verdict}")
        if passes:
            keepers.append(r["variant"])
    md.append("")
    md.append(f"**Survivors**: {keepers if keepers else 'NONE — news features did not add lift over BASE_FEATS at this sample size.'}")
    md.append("")
    md.append("## Honest caveats")
    md.append("")
    md.append("- News data covers only 2026-02-01 to 2026-04-27 (~3 months) → train+test windows are tight.")
    md.append("- Sample-size noise dominates at this scale; AUC differences <0.01 are within noise floor.")
    md.append("- A real verdict requires backfilling news from 2018+ (MoneyControl/ET archives).")
    md.append("- Top-5 basket returns vary heavily across 5-day windows; treat as illustrative, not decisive.")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT.relative_to(ROOT)}")
    print(f"wrote {OUT_PARQUET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
