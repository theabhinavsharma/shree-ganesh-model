"""Multibagger achievability — for each (large-threshold × long-horizon) combo,
find the minimum calibrated score that delivers ≥90% real hit rate.

Tests:
  thresholds: 25%, 35%, 50%, 75%, 100%, 150%, 200%
  horizons:   90d, 126d, 180d, 252d, 378d (1.5y), 504d (2y)

This is the long-horizon counterpart to find_achievable_targets.py.

Goal: identify if "+100% in 180d at 90% confidence" is real or fantasy.
Output: data/derived/multibagger_targets.parquet, reports/multibagger_targets.md
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_PARQUET = ROOT / "data/derived/multibagger_targets.parquet"
OUT_REPORT = ROOT / "reports/multibagger_targets.md"

THRESHOLDS = [0.25, 0.35, 0.50, 0.75, 1.00, 1.50, 2.00]
HORIZONS = [90, 126, 180, 252, 378, 504]
CONVICTION_BAR = 0.90

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
    df = df[df["trade_date"] >= "2015-01-01"]  # need long history for 504d horizon

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
    """Forward max-high over horizon days; 1 if (max_high / close - 1) >= threshold."""
    # for very long horizons (>250d) avoid the big concat; use rolling-max trick
    if horizon <= 60:
        shifts = pd.concat(
            [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, horizon + 1)],
            axis=1,
        )
        fwd_max = shifts.max(axis=1)
    else:
        # rolling max: shift -1 then take rolling(window).max() on a reversed-time view
        # cheap implementation: compute per-group max via cummax-from-future
        df_g = df.copy()
        df_g["_idx"] = df_g.groupby("symbol").cumcount()
        df_g = df_g.sort_values(["symbol", "trade_date"])
        # for each row, max of high over rows idx+1 .. idx+horizon
        fwd_max = (df_g.groupby("symbol")["high"]
                       .transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max()))
    fwd_pct = fwd_max / df["close"] - 1
    target = (fwd_pct >= threshold).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-horizon).notna()
    target[~complete] = -1
    return target


def find_score_for_conviction(scores: np.ndarray, hits: np.ndarray, target_p: float = 0.90) -> dict:
    df = pd.DataFrame({"score": scores, "hit": hits}).sort_values("score", ascending=False)
    df["cum_hits"] = df["hit"].cumsum()
    df["cum_n"] = np.arange(1, len(df) + 1)
    df["cum_hit_rate"] = df["cum_hits"] / df["cum_n"]
    valid = df[(df["cum_hit_rate"] >= target_p) & (df["cum_n"] >= 30)]  # smaller min sample for rarer events
    if valid.empty:
        return {"score_threshold": None, "n_total": 0, "hit_rate": None}
    last = valid.iloc[-1]
    return {
        "score_threshold": float(last["score"]),
        "n_total": int(last["cum_n"]),
        "hit_rate": float(last["cum_hit_rate"]),
    }


def evaluate_cell(panel: pd.DataFrame, threshold: float, horizon: int) -> dict:
    target = build_target(panel, threshold, horizon)
    df = panel.copy()
    df["target"] = target
    df = df[df["target"] != -1]
    df = df[df["adv_20d_cr"] >= 1.0]
    if len(df) < 30000:
        return {"threshold": threshold, "horizon": horizon, "status": "INSUFFICIENT_DATA",
                "base_rate": None, "score_at_90": None, "n_at_90": 0}

    base_rate = df["target"].mean()
    if base_rate < 0.005:  # extremely rare event — model won't separate well
        return {"threshold": threshold, "horizon": horizon, "status": "TOO_RARE",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0}

    # for long horizons, leave more train data
    test_year_start = 2023 if horizon >= 252 else 2024
    tr = df[df["year"] < test_year_start]
    te = df[df["year"] >= test_year_start]
    if len(tr) < 5000 or len(te) < 1000:
        return {"threshold": threshold, "horizon": horizon, "status": "INSUFFICIENT_FOLD",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0}

    try:
        lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[BASE_FEATS], tr["target"])
        p = lgbm.predict_proba(te[BASE_FEATS])[:, 1]
        tr_calib = tr.sample(min(50000, len(tr)), random_state=42)
        p_tr = lgbm.predict_proba(tr_calib[BASE_FEATS])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, tr_calib["target"])
        p_cal = iso.transform(p)
    except Exception as e:
        return {"threshold": threshold, "horizon": horizon, "status": f"FIT_ERR: {type(e).__name__}",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0}

    res = find_score_for_conviction(p_cal, te["target"].values, target_p=CONVICTION_BAR)
    return {
        "threshold": threshold,
        "horizon": horizon,
        "status": "OK" if res["score_threshold"] is not None else "BAR_NOT_REACHED",
        "base_rate": float(base_rate),
        "score_at_90": res["score_threshold"],
        "n_at_90": res["n_total"],
        "hit_rate_at_90": res["hit_rate"],
    }


def main() -> None:
    print(f"== find_multibagger_targets ==")
    print(f"  thresholds: {[f'{t*100:.0f}%' for t in THRESHOLDS]}")
    print(f"  horizons:   {HORIZONS} (days)")
    print(f"  conviction bar: {CONVICTION_BAR*100:.0f}%")

    panel = build_panel()
    panel = panel.dropna(subset=BASE_FEATS).copy()
    print(f"  panel: {len(panel):,} rows\n")

    results = []
    started = time.time()
    cell_num = 0
    total = len(THRESHOLDS) * len(HORIZONS)
    for thr in THRESHOLDS:
        for hor in HORIZONS:
            cell_num += 1
            t0 = time.time()
            res = evaluate_cell(panel, thr, hor)
            results.append(res)
            score_str = f"{res['score_at_90']:.2f}" if res.get("score_at_90") is not None else "—"
            base_str = f"{res.get('base_rate', 0)*100:>5.1f}%" if res.get("base_rate") else "—"
            n_str = f"{res.get('n_at_90', 0):,}" if res.get("n_at_90") else "0"
            print(f"  [{cell_num:>2}/{total}] thr={thr*100:>3.0f}%  hor={hor:>3}d  "
                  f"status={res['status']:<22}  base={base_str}  score@90={score_str}  n={n_str}  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(results)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    md = ["# Multibagger achievability map — 90% conviction", "",
          "_For long-horizon big-move targets (25-200% gains over 90-504 days)._", "",
          "## Heatmap: score @ 90% conviction (`—` = bar not reached)", ""]
    header = "| threshold \\ horizon | " + " | ".join(f"{h}d" for h in HORIZONS) + " |"
    md.append(header)
    md.append("|---|" + "|".join(["---:"] * len(HORIZONS)) + "|")
    for thr in THRESHOLDS:
        row = [f"**{thr*100:.0f}%**"]
        for hor in HORIZONS:
            cell = df[(df["threshold"] == thr) & (df["horizon"] == hor)]
            if cell.empty or cell.iloc[0]["status"] != "OK":
                stat = cell.iloc[0]["status"] if not cell.empty else "—"
                row.append(stat[:10])
            else:
                c = cell.iloc[0]
                row.append(f"**{c['score_at_90']:.2f}** (n={int(c['n_at_90'])})")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("## OK cells, ranked by sample size at 90%")
    md.append("")
    ok = df[df["status"] == "OK"].sort_values("n_at_90", ascending=False).head(20)
    if len(ok):
        md.append("| Threshold | Horizon | Base rate | Score @ 90% | OOS hit rate | n total |")
        md.append("|---:|---:|---:|---:|---:|---:|")
        for _, r in ok.iterrows():
            md.append(f"| {r['threshold']*100:.0f}% | {int(r['horizon'])}d | "
                      f"{r['base_rate']*100:.1f}% | {r['score_at_90']:.3f} | "
                      f"{r['hit_rate_at_90']*100:.1f}% | {int(r['n_at_90']):,} |")
    else:
        md.append("_No cells reached the 90% bar with sufficient sample size._")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PARQUET}")


if __name__ == "__main__":
    main()
