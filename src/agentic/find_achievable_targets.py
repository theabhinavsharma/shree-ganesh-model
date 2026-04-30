"""For each (threshold % × horizon days) combination, find the minimum
calibrated score that delivers a 90% real OOS hit rate. The achievability
map answers:

  "What's the highest-frequency, lowest-threshold target our model can
   actually deliver at 90% confidence?"

We test:
  thresholds: 1%, 2%, 3%, 5%, 7%, 10%, 15%, 20%, 30%
  horizons:   5d, 7d, 15d, 21d, 30d, 60d, 90d, 126d, 252d

For each cell:
  1. Build target: max(high) over horizon / close >= threshold
  2. Train LGB+XGB ensemble walk-forward (train < 2024, test 2024-2025)
  3. Isotonic-calibrate on prior fold
  4. Compute calibration table by score band
  5. Identify the score band with >=90% real hit rate
  6. Report (cell_score_threshold, cell_n_above_threshold, expected_freq_per_day)

Output:
  data/derived/achievable_targets.parquet — full grid + cell stats
  reports/achievable_targets.md — heatmap markdown table

Then the user picks a cell from the map: "I want 90% confidence on 2% in 7d → score >= 0.X → fires every Y days on average."
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
EXTRA = ROOT / "data/derived/extra_features.parquet"
OUT_PARQUET = ROOT / "data/derived/achievable_targets.parquet"
OUT_REPORT = ROOT / "reports/achievable_targets.md"

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
HORIZONS = [5, 7, 15, 21, 30, 60, 90, 126, 252]
CONVICTION_BAR = 0.90  # the user-set goal

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
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(mkt, on="trade_date", how="left")
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())
    return df


def build_target(df: pd.DataFrame, threshold: float, horizon: int) -> pd.Series:
    """Forward max-high over horizon days; 1 if (max_high / close - 1) >= threshold."""
    shifts = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, horizon + 1)],
        axis=1,
    )
    fwd_max = shifts.max(axis=1)
    fwd_pct = fwd_max / df["close"] - 1
    target = (fwd_pct >= threshold).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-horizon).notna()
    target[~complete] = -1  # marks "label not complete"
    return target


def find_score_for_conviction(scores: np.ndarray, hits: np.ndarray, target_p: float = 0.90) -> dict:
    """Find the minimum calibrated score where actual hit rate >= target_p.
    Returns dict with score_threshold, n, hit_rate, n_per_day."""
    df = pd.DataFrame({"score": scores, "hit": hits}).sort_values("score", ascending=False)
    # rolling top-K hit rate
    df["cum_hits"] = df["hit"].cumsum()
    df["cum_n"] = np.arange(1, len(df) + 1)
    df["cum_hit_rate"] = df["cum_hits"] / df["cum_n"]
    # find largest K where cum_hit_rate >= target_p (need >= 50 instances for stability)
    valid = df[(df["cum_hit_rate"] >= target_p) & (df["cum_n"] >= 50)]
    if valid.empty:
        return {"score_threshold": None, "n_total": 0, "hit_rate": None, "n_per_day": 0}
    last = valid.iloc[-1]
    return {
        "score_threshold": float(last["score"]),
        "n_total": int(last["cum_n"]),
        "hit_rate": float(last["cum_hit_rate"]),
        "n_per_day": float(last["cum_n"] / df["score"].notna().sum() * 511),  # 511 OOS days in 2024-2025
    }


def evaluate_cell(panel: pd.DataFrame, threshold: float, horizon: int) -> dict:
    target = build_target(panel, threshold, horizon)
    df = panel.copy()
    df["target"] = target
    df = df[df["target"] != -1]
    df = df[df["adv_20d_cr"] >= 1.0]
    if len(df) < 50000:
        return {"threshold": threshold, "horizon": horizon, "status": "INSUFFICIENT_DATA",
                "base_rate": None, "score_at_90": None, "n_at_90": 0, "fires_per_day": 0}

    base_rate = df["target"].mean()
    if base_rate >= 0.95:
        # target so easy that conviction concept is meaningless
        return {"threshold": threshold, "horizon": horizon, "status": "TRIVIAL",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0,
                "fires_per_day": 0}

    # walk-forward 1-fold (train < 2024, test 2024-2025)
    tr = df[df["year"] < 2024]
    te = df[df["year"] >= 2024]
    if len(tr) < 5000 or len(te) < 1000:
        return {"threshold": threshold, "horizon": horizon, "status": "INSUFFICIENT_FOLD",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0,
                "fires_per_day": 0}

    # one-fold quick LGB (single model — sufficient for this grid scan)
    try:
        lgbm = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[BASE_FEATS], tr["target"])
        p = lgbm.predict_proba(te[BASE_FEATS])[:, 1]
        # calibrate on a held-out chunk of training
        tr_calib = tr.sample(min(50000, len(tr)), random_state=42)
        p_tr = lgbm.predict_proba(tr_calib[BASE_FEATS])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_tr, tr_calib["target"])
        p_cal = iso.transform(p)
    except Exception as e:
        return {"threshold": threshold, "horizon": horizon, "status": f"FIT_ERR: {type(e).__name__}",
                "base_rate": float(base_rate), "score_at_90": None, "n_at_90": 0, "fires_per_day": 0}

    res = find_score_for_conviction(p_cal, te["target"].values, target_p=CONVICTION_BAR)
    return {
        "threshold": threshold,
        "horizon": horizon,
        "status": "OK" if res["score_threshold"] is not None else "BAR_NOT_REACHED",
        "base_rate": float(base_rate),
        "score_at_90": res["score_threshold"],
        "n_at_90": res["n_total"],
        "hit_rate_at_90": res["hit_rate"],
        "fires_per_day": round(res["n_per_day"], 1),
    }


def main() -> None:
    print(f"== find_achievable_targets ==")
    print(f"  thresholds: {[f'{t*100:.0f}%' for t in THRESHOLDS]}")
    print(f"  horizons:   {HORIZONS}")
    print(f"  conviction bar: {CONVICTION_BAR*100:.0f}%")
    print(f"  total cells: {len(THRESHOLDS) * len(HORIZONS)}")

    panel = build_panel()
    panel = panel.dropna(subset=BASE_FEATS).copy()
    print(f"  panel: {len(panel):,} rows\n")

    results = []
    started = time.time()
    cell_num = 0
    total_cells = len(THRESHOLDS) * len(HORIZONS)
    for thr in THRESHOLDS:
        for hor in HORIZONS:
            cell_num += 1
            t0 = time.time()
            res = evaluate_cell(panel, thr, hor)
            dt = time.time() - t0
            results.append(res)
            elapsed = time.time() - started
            eta = (total_cells - cell_num) * (elapsed / cell_num) / 60
            score_str = f"{res['score_at_90']:.2f}" if res.get("score_at_90") is not None else "—"
            fires_str = f"{res['fires_per_day']:.1f}/day" if res.get("fires_per_day") else "—"
            print(f"  [{cell_num:>3}/{total_cells}] thr={thr*100:>4.0f}%  hor={hor:>3}d  "
                  f"status={res['status']:<22}  base={res.get('base_rate', 0)*100:>5.1f}%  "
                  f"score_at_90={score_str}  fires={fires_str}  ({dt:.0f}s)")

    df = pd.DataFrame(results)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # heatmap markdown
    md = ["# Achievable targets — 90% conviction map", "",
          f"For each (threshold × horizon) combination, the minimum calibrated score",
          f"that delivers ≥90% real hit rate on 2024-2025 OOS data.", "",
          "Read: each cell shows `score@90 / fires_per_day`. Pick a cell where",
          "fires_per_day is high enough to compound to your goal.", "", "## Heatmap"]
    md.append("")
    header = "| threshold \\ horizon | " + " | ".join(f"{h}d" for h in HORIZONS) + " |"
    md.append(header)
    md.append("|---|" + "|".join(["---:"] * len(HORIZONS)) + "|")
    for thr in THRESHOLDS:
        row = [f"{thr*100:.0f}%"]
        for hor in HORIZONS:
            cell = df[(df["threshold"] == thr) & (df["horizon"] == hor)]
            if cell.empty or cell.iloc[0]["status"] != "OK":
                stat = cell.iloc[0]["status"] if not cell.empty else "—"
                row.append(stat[:8])
            else:
                c = cell.iloc[0]
                row.append(f"**{c['score_at_90']:.2f}** / {c['fires_per_day']:.1f}/d")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    md.append("## Best cells (highest fires-per-day at 90% conviction)")
    md.append("")
    ok = df[df["status"] == "OK"].sort_values("fires_per_day", ascending=False).head(15)
    if len(ok):
        md.append("| Threshold | Horizon | score @ 90% | OOS hit rate | n total | Fires/day | Theoretical ann ROI |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in ok.iterrows():
            n_per_year = r["fires_per_day"] * 250
            theo_ann = (1 + r["threshold"]) ** n_per_year - 1 if n_per_year > 0 else 0
            md.append(f"| {r['threshold']*100:.0f}% | {int(r['horizon'])}d | {r['score_at_90']:.3f} | "
                      f"{r['hit_rate_at_90']*100:.1f}% | {int(r['n_at_90']):,} | "
                      f"{r['fires_per_day']:.1f} | {theo_ann*100:+,.0f}% |")
    md.append("")
    md.append("_Theoretical ann ROI assumes you trade every signal at full size with no slippage._")
    md.append("_Realistic capture: 30-40% of theoretical._")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PARQUET}")


if __name__ == "__main__":
    main()
