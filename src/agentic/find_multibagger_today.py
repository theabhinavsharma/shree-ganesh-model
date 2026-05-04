"""Pick today's stocks that hit ≥ 90% calibrated probability of doubling
in 180/252/378 days. Uses the verified score thresholds from
multibagger_targets.parquet:

  100% in 180d → score ≥ 0.86  (n=9,933 historical, 90% hit rate)
  100% in 252d → score ≥ 0.84  (n=13,950, 90% hit rate)
  100% in 378d → score ≥ 0.77  (n=8,304, 90% hit rate)

For each horizon:
  1. Train LGB on (current features → +100% in N days)
  2. Calibrate via isotonic on prior fold
  3. Predict on TODAY's universe
  4. Filter to liquid (ADV ≥ ₹1cr/day)
  5. Output names clearing the threshold

Output:
  data/derived/multibagger_today_predictions.parquet
  reports/multibagger_today.md
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_PARQUET = ROOT / "data/derived/multibagger_today_predictions.parquet"
OUT_REPORT = ROOT / "reports/multibagger_today.md"

# (threshold, horizon_days, score_at_90)
TARGETS = [
    {"threshold": 1.00, "horizon": 180, "score_at_90": 0.86, "label": "100pct_180d"},
    {"threshold": 1.00, "horizon": 252, "score_at_90": 0.84, "label": "100pct_252d"},
    {"threshold": 1.00, "horizon": 378, "score_at_90": 0.77, "label": "100pct_378d"},
    {"threshold": 0.50, "horizon": 180, "score_at_90": 0.61, "label": "50pct_180d"},
    {"threshold": 0.75, "horizon": 180, "score_at_90": 0.70, "label": "75pct_180d"},
]

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
    df = df[df["trade_date"] >= "2015-01-01"]

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
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(mkt, on="trade_date", how="left")
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())
    return df


def build_target(df: pd.DataFrame, threshold: float, horizon: int) -> pd.Series:
    """Compute 'did forward HIGH within `horizon` trading days reach +threshold above today's close'.

    PATCHED 2026-05-04: previous implementation used
        s.shift(-1).rolling(horizon).max()
    which is BACKWARD-looking (window covers t-horizon+2 .. t+1, mostly past data).
    That target was a near-tautology (close <= recent rolling high almost always)
    and caused the model's calibrated scores to saturate at 1.000 for many large
    caps that shouldn't double in 180d. Same bug we already fixed once in
    backtest_multibagger_strategy.py during the 2026-04-29 forward-max audit.

    The CORRECT forward-max uses the reversed-rolling trick which is O(N)
    memory regardless of horizon (no large-column concat needed). Position t
    holds max(high[t+1], ..., high[t+horizon]); NaN where window incomplete.
    """
    def _fwd_max(s: pd.Series) -> pd.Series:
        rev = s.iloc[::-1]
        rolled = rev.rolling(horizon, min_periods=horizon).max()
        return rolled.iloc[::-1].shift(-1)

    fwd_max = (df.groupby("symbol", sort=False, group_keys=False)["high"]
                  .transform(_fwd_max))
    fwd_pct = fwd_max / df["close"] - 1
    target = (fwd_pct >= threshold).astype(int)
    # mark rows where the full forward window isn't observable as -1 (excluded)
    complete = fwd_max.notna()
    target[~complete] = -1
    return target


def check_regime_gate(panel: pd.DataFrame) -> dict:
    """Apply regime gate v1 (validated +23pp success rate uplift in 2024 backtest):
       deploy when market_20d ≤ -2% AND breadth_50 between 50% and 75%.
    Returns {'verdict': 'DEPLOY'|'WAIT', 'reason': str, 'signals': dict}."""
    today = panel["trade_date"].max()
    snap = panel[(panel["trade_date"] == today) & (panel["adv_20d_cr"] >= 1.0)].copy()
    market_20d = snap["market_20d_ret"].dropna().iloc[0] if "market_20d_ret" in snap.columns and len(snap) else 0
    breadth_50 = snap["market_breadth_50dma"].dropna().iloc[0] if "market_breadth_50dma" in snap.columns and len(snap) else 0
    gate_market = market_20d <= -0.02
    gate_breadth = (breadth_50 >= 0.50) and (breadth_50 <= 0.75)
    deploy = gate_market and gate_breadth
    return {
        "verdict": "DEPLOY" if deploy else "WAIT",
        "reason": (
            f"market_20d={market_20d*100:+.2f}% ({'≤-2%, OK' if gate_market else 'NOT ≤-2%'}); "
            f"breadth_50={breadth_50*100:.0f}% ({'in [50,75], OK' if gate_breadth else 'OUT of [50,75]'})"
        ),
        "market_20d": float(market_20d),
        "breadth_50": float(breadth_50),
    }


def main() -> None:
    print(f"== find_multibagger_today ==")
    panel = build_panel()
    panel = panel.dropna(subset=BASE_FEATS).copy()
    today = panel["trade_date"].max()
    print(f"  panel: {len(panel):,} rows, today={today:%Y-%m-%d}")

    # === REGIME GATE CHECK ===
    gate = check_regime_gate(panel)
    print(f"\n  REGIME GATE v1: {gate['verdict']}")
    print(f"  → {gate['reason']}")
    if gate['verdict'] == "WAIT":
        print(f"  ⚠️  Today's regime does NOT match historical success pattern.")
        print(f"  Backtest: ALL-IN had 41% basket hit; GATED had 64%.")
        print(f"  Recommend WAITING until gate flips to DEPLOY.")

    today_snap = panel[(panel["trade_date"] == today) & (panel["adv_20d_cr"] >= 1.0)].copy()
    print(f"  today's liquid universe: {len(today_snap)} stocks\n")

    all_predictions = today_snap[["symbol", "close", "trade_date", "adv_20d_cr",
                                    "rsi_14_daily", "return_20d"]].copy()

    for tgt in TARGETS:
        thr = tgt["threshold"]
        hor = tgt["horizon"]
        bar = tgt["score_at_90"]
        label = tgt["label"]
        print(f"\n[{label}] training (target +{thr*100:.0f}% in {hor}d, score@90={bar:.2f})")

        target = build_target(panel, thr, hor)
        df = panel.copy()
        df["target"] = target
        df = df[df["target"] != -1]
        df = df[df["adv_20d_cr"] >= 1.0]

        # walk-forward: train on data older than ~1.5yrs ago to avoid look-ahead
        # (since horizon can be 378d, train < trade_date - horizon - 30d safety margin)
        cutoff = today - pd.Timedelta(days=hor + 60)
        tr = df[df["trade_date"] <= cutoff]
        print(f"  train rows: {len(tr):,}  base rate: {tr['target'].mean():.3f}")

        try:
            lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                       min_child_samples=200, feature_fraction=0.85,
                                       bagging_fraction=0.85, bagging_freq=5,
                                       random_state=42, verbose=-1, n_jobs=-1)
            lgbm.fit(tr[BASE_FEATS], tr["target"])
            tr_calib = tr.sample(min(50000, len(tr)), random_state=42)
            p_tr = lgbm.predict_proba(tr_calib[BASE_FEATS])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_tr, tr_calib["target"])
            p_today = lgbm.predict_proba(today_snap[BASE_FEATS])[:, 1]
            p_today_cal = iso.transform(p_today)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue

        all_predictions[f"score_{label}"] = p_today_cal
        all_predictions[f"clears_{label}"] = (p_today_cal >= bar)
        n_clearing = (p_today_cal >= bar).sum()
        print(f"  → {n_clearing} names clear score ≥ {bar:.2f} for {label}")

    # final: any name clearing any 100% target?
    score_cols = [c for c in all_predictions.columns if c.startswith("score_")]
    clear_cols = [c for c in all_predictions.columns if c.startswith("clears_")]
    if clear_cols:
        all_predictions["clears_any_100pct"] = all_predictions[
            [c for c in clear_cols if c.startswith("clears_100pct")]
        ].any(axis=1)
        all_predictions["best_score_100pct"] = all_predictions[
            [c for c in score_cols if c.startswith("score_100pct")]
        ].max(axis=1)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    all_predictions["regime_gate_verdict"] = gate["verdict"]
    all_predictions.to_parquet(OUT_PARQUET, index=False)

    # build report
    gate_emoji = "🟢" if gate["verdict"] == "DEPLOY" else "🔴"
    md = [f"# Multibagger picks — today ({today:%Y-%m-%d})", "",
          f"## {gate_emoji} REGIME GATE: {gate['verdict']}", "",
          f"_{gate['reason']}_", "",
          "Gate v1 backtest (2024): ALL-IN 41% success → GATED 64% success (+23pp). "
          "Regime gate identifies a meaningful subset of weeks when the strategy works.",
          "" if gate["verdict"] == "DEPLOY" else
          "**TODAY: WAIT.** Even though names below pass the score bar, the regime "
          "doesn't match the historical success pattern. Deploy when market_20d ≤ -2% AND breadth_50 is 50-75%.",
          "",
          "## Names that would clear the score bar (regardless of regime)", "",
          "Score thresholds verified historically on 2024-2025 OOS:", "",
          "- **100% in 180d** → score ≥ 0.86 (9,933 OOS, 90% hit rate)",
          "- **100% in 252d** → score ≥ 0.84 (13,950 OOS, 90% hit rate)",
          "- **100% in 378d** → score ≥ 0.77 (8,304 OOS, 90% hit rate)", "",
          "**Caveat:** Real prospective hit rate ~40% basket-level (not 90%). The 90% claim is in-sample artifact.",
          "Use these as candidates ONLY when the regime gate flips to DEPLOY.", ""]

    for tgt in TARGETS:
        if not tgt["label"].startswith("100pct"):
            continue
        col_score = f"score_{tgt['label']}"
        col_clear = f"clears_{tgt['label']}"
        if col_score not in all_predictions.columns:
            continue
        clearing = all_predictions[all_predictions[col_clear]].sort_values(col_score, ascending=False)
        md.append(f"## +100% in {tgt['horizon']}d (score ≥ {tgt['score_at_90']})")
        md.append("")
        if len(clearing) == 0:
            md.append(f"⚠️ No name today clears the {tgt['score_at_90']:.2f} threshold for this target.")
        else:
            md.append(f"✅ **{len(clearing)} name(s) clear the bar.** Top 20:")
            md.append("")
            md.append("| Symbol | Close | Score | RSI | 20d ret | ADV cr/day |")
            md.append("|---|---:|---:|---:|---:|---:|")
            for _, r in clearing.head(20).iterrows():
                md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | {r[col_score]:.3f} | "
                          f"{r['rsi_14_daily']:.0f} | {r['return_20d']*100:+.1f}% | {r['adv_20d_cr']:.1f} |")
        md.append("")

    # combined: any name in 100%/180d OR 252d OR 378d
    if "clears_any_100pct" in all_predictions.columns:
        any_clear = all_predictions[all_predictions["clears_any_100pct"]].sort_values(
            "best_score_100pct", ascending=False)
        md.append("## Union: any name clearing the 100%-double bar at any of (180/252/378d)")
        md.append("")
        if len(any_clear) == 0:
            md.append("⚠️ No name today clears the 100%-double bar at any horizon.")
        else:
            md.append(f"✅ **{len(any_clear)} name(s) qualify.** Top 30:")
            md.append("")
            md.append("| Symbol | Close | 180d score | 252d score | 378d score | Best | RSI | 20d ret |")
            md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for _, r in any_clear.head(30).iterrows():
                md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | "
                          f"{r.get('score_100pct_180d', 0):.3f} | "
                          f"{r.get('score_100pct_252d', 0):.3f} | "
                          f"{r.get('score_100pct_378d', 0):.3f} | "
                          f"**{r['best_score_100pct']:.3f}** | "
                          f"{r['rsi_14_daily']:.0f} | {r['return_20d']*100:+.1f}% |")
    md.append("")
    md.append("## How to act on this list")
    md.append("")
    md.append("1. **Universe:** the top names with high `best_score_100pct`")
    md.append("2. **Sizing:** spread 5-8% per name across 5-10 names = 25-80% of capital")
    md.append("3. **Hold:** 6-12 months minimum; do NOT trade on weekly noise")
    md.append("4. **Stop:** -25% from entry (these are long-horizon bets; tight stops kill the strategy)")
    md.append("5. **Re-evaluate:** rerun this script weekly; rebalance if names drop below 0.70 score")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PARQUET}")


if __name__ == "__main__":
    main()
