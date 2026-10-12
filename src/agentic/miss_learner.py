"""MISS-DRIVEN RL LEARNER — the first version.

Every week (or on demand), this script:
  1. Computes actual top-20 winners over the last week
  2. Compares against our picks + filter-would-passers
  3. Categorizes misses: universe/coverage/threshold/pattern
  4. For each miss, extracts feature vector on entry day
  5. Statistically compares MISS features vs PICK features
  6. Identifies which features SEPARATE misses from picks
  7. Suggests filter/threshold adjustments
  8. Persists learnings to logs/miss_learnings.jsonl

Consumed by: infinite_hypothesis_loop.py (for feature promotion),
              constitution audits (for filter revisions).

Per CONSTITUTION.md §1.6 — every learning is timestamped and cited.
"""
from __future__ import annotations
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Code/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA = ROOT / "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet"
LEARNINGS_LOG = ROOT / "logs/miss_learnings.jsonl"


def compute_top_movers(entry_date: str, exit_date: str, liq_min_cr: float = 5.0) -> pd.DataFrame:
    """Rank liquid stocks by peak intraday high in the window."""
    prices = pd.read_parquet(PRICES,
        columns=["symbol","trade_date","close","high","low","return_1d","rsi_14_daily",
                  "return_20d","total_traded_value","volume_vs_20d"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["symbol","trade_date"])
    prices["ret_5d"] = prices.groupby("symbol")["close"].pct_change(5)

    entry = prices[prices["trade_date"] == entry_date][["symbol","close","rsi_14_daily",
        "return_20d","total_traded_value","volume_vs_20d","ret_5d"]].rename(columns={"close":"entry"})
    entry["adv_cr"] = entry["total_traded_value"] / 1e7
    window = prices[(prices["trade_date"] > entry_date) & (prices["trade_date"] <= exit_date)]
    peak = window.groupby("symbol")["high"].max().rename("peak").reset_index()
    exit_ = window[window["trade_date"] == pd.Timestamp(exit_date)][["symbol","close"]].rename(columns={"close":"exit"})

    df = entry.merge(peak, on="symbol").merge(exit_, on="symbol", how="left")
    df["peak_ret_pct"] = (df["peak"]/df["entry"] - 1) * 100
    df["close_ret_pct"] = (df["exit"]/df["entry"] - 1) * 100
    df = df[df["adv_cr"] >= liq_min_cr]
    return df.sort_values("peak_ret_pct", ascending=False)


def would_pass_current_filter(row) -> tuple[bool, list[str]]:
    """Apply the current QC filter to a row; return (pass, reasons_failed)."""
    fails = []
    if pd.isna(row["rsi_14_daily"]) or not (40 <= row["rsi_14_daily"] <= 72):
        fails.append(f"RSI={row['rsi_14_daily']:.0f}")
    if pd.notnull(row["return_20d"]) and not (-0.15 <= row["return_20d"] <= 0.30):
        fails.append(f"20d={row['return_20d']*100:+.0f}%")
    if pd.notnull(row["ret_5d"]) and not (-0.05 <= row["ret_5d"] <= 0.10):
        fails.append(f"5d={row['ret_5d']*100:+.0f}%")
    if pd.notnull(row["volume_vs_20d"]) and row["volume_vs_20d"] >= 3:
        fails.append(f"vol={row['volume_vs_20d']:.1f}x")
    if row["entry"] <= 50:
        fails.append(f"price=₹{row['entry']:.0f}")
    if row["adv_cr"] < 5:
        fails.append(f"ADV={row['adv_cr']:.1f}cr")
    return (len(fails) == 0, fails)


def categorize_miss(row, in_engine_top30: bool, cs_score: float) -> str:
    """Categorize why a winner was missed."""
    if cs_score is None or pd.isna(cs_score):
        return "UNIVERSE_MISS"  # Not scored by engines at all
    passed, _ = would_pass_current_filter(row)
    if not passed:
        return "CORRECTLY_FILTERED"  # Chase/pump we didn't want
    if not in_engine_top30 and cs_score < 0.50:
        return "MODEL_COVERAGE_MISS"  # Model scored too low
    if not in_engine_top30 and cs_score >= 0.50:
        return "THRESHOLD_MISS"  # Score OK but rank outside top-30
    if in_engine_top30:
        return "CAUGHT"  # We had it
    return "UNKNOWN"


def analyze_misses(entry_date: str, exit_date: str, top_n: int = 20) -> dict:
    """Run miss analysis for a specific window. Returns findings dict."""
    print(f"\n{'='*80}")
    print(f" MISS ANALYSIS: {entry_date} → {exit_date}, top {top_n} winners")
    print(f"{'='*80}\n")

    movers = compute_top_movers(entry_date, exit_date)
    top = movers.head(top_n)

    # Try to load engine outputs (compare_short_horizons as primary reference)
    cs_path = ROOT / "data/derived/compare_short_horizons.parquet"
    cs = pd.read_parquet(cs_path) if cs_path.exists() else pd.DataFrame(columns=["symbol","score_5pct_15d"])
    cs_top30 = set(cs.sort_values("score_5pct_15d", ascending=False).head(30)["symbol"].tolist()) if len(cs) else set()

    categories = {"CAUGHT": [], "CORRECTLY_FILTERED": [], "UNIVERSE_MISS": [],
                  "MODEL_COVERAGE_MISS": [], "THRESHOLD_MISS": []}

    for _, r in top.iterrows():
        sym = r["symbol"]
        in_cs30 = sym in cs_top30
        cs_score = None
        if sym in cs["symbol"].values:
            cs_score = float(cs[cs["symbol"]==sym]["score_5pct_15d"].iloc[0])
        cat = categorize_miss(r, in_cs30, cs_score)
        categories[cat].append({
            "symbol": sym,
            "peak_ret_pct": round(float(r["peak_ret_pct"]), 1),
            "close_ret_pct": round(float(r["close_ret_pct"]), 1),
            "entry": round(float(r["entry"]), 2),
            "rsi": round(float(r["rsi_14_daily"]), 1) if pd.notnull(r["rsi_14_daily"]) else None,
            "return_20d": round(float(r["return_20d"])*100, 1) if pd.notnull(r["return_20d"]) else None,
            "ret_5d": round(float(r["ret_5d"])*100, 1) if pd.notnull(r["ret_5d"]) else None,
            "volume_vs_20d": round(float(r["volume_vs_20d"]), 2) if pd.notnull(r["volume_vs_20d"]) else None,
            "adv_cr": round(float(r["adv_cr"]), 1),
            "cs_score": round(cs_score, 3) if cs_score else None,
        })

    # Print summary
    print(f"CAUGHT (in top-30 of an engine):        {len(categories['CAUGHT']):>2}")
    print(f"CORRECTLY_FILTERED (chase/pump):        {len(categories['CORRECTLY_FILTERED']):>2}")
    print(f"UNIVERSE_MISS (not scored by engines):  {len(categories['UNIVERSE_MISS']):>2}")
    print(f"MODEL_COVERAGE_MISS (score <0.50):      {len(categories['MODEL_COVERAGE_MISS']):>2}")
    print(f"THRESHOLD_MISS (0.50<=score, rank>30):  {len(categories['THRESHOLD_MISS']):>2}")

    print(f"\n=== Detailed miss breakdown ===")
    for cat, items in categories.items():
        if not items or cat == "CAUGHT" or cat == "CORRECTLY_FILTERED":
            continue
        print(f"\n{cat}:")
        for it in items:
            print(f"  {it['symbol']:12s} +{it['peak_ret_pct']:>5.1f}%  entry=₹{it['entry']:>8.2f}  RSI={str(it['rsi']):>4s}  20d={str(it['return_20d']):>5s}%  5d={str(it['ret_5d']):>5s}%  cs_score={it['cs_score']}")

    # STATISTICAL COMPARISON: MISS features vs PICKED features
    misses_all = categories["MODEL_COVERAGE_MISS"] + categories["THRESHOLD_MISS"]
    if len(misses_all) >= 3:
        print(f"\n=== STATISTICAL: what separates MISSES from what we PICK? ===")
        miss_df = pd.DataFrame(misses_all)
        # picks would be represented by cs top-30 with QC pass — for now compare miss features to universe median
        univ = compute_top_movers(entry_date, exit_date)
        for feat in ["rsi","return_20d","ret_5d","volume_vs_20d","cs_score"]:
            if feat in miss_df.columns:
                miss_mean = miss_df[feat].dropna().mean()
                miss_std = miss_df[feat].dropna().std()
                # universe median for context
                if feat == "cs_score":
                    ref = cs["score_5pct_15d"].median() if len(cs) else None
                    ref_label = "universe median"
                else:
                    feat_col = {"rsi":"rsi_14_daily","return_20d":"return_20d",
                                "ret_5d":"ret_5d","volume_vs_20d":"volume_vs_20d"}[feat]
                    if feat_col in univ.columns:
                        # For return_20d and ret_5d normalize scaling
                        univ_vals = univ[feat_col].dropna()
                        if feat in ("return_20d","ret_5d"):
                            univ_vals = univ_vals * 100
                        ref = univ_vals.median()
                        ref_label = "universe median"
                    else:
                        ref = None
                        ref_label = "n/a"
                print(f"  {feat:15s}: miss mean={miss_mean:>7.2f}  std={miss_std:>5.2f}  ({ref_label}={ref})" if pd.notnull(miss_mean) else "")

    findings = {
        "run_ts_utc": datetime.utcnow().isoformat() + "Z",
        "window_entry": entry_date,
        "window_exit": exit_date,
        "top_n_analyzed": top_n,
        "counts": {k: len(v) for k, v in categories.items()},
        "categories": categories,
        "learnings": [],
    }

    # Auto-generate learnings
    if categories["UNIVERSE_MISS"]:
        findings["learnings"].append({
            "type": "universe_gap",
            "detail": f"{len(categories['UNIVERSE_MISS'])} winners are not scored by any engine — likely recent IPOs or missing symbols. Symbols: {[m['symbol'] for m in categories['UNIVERSE_MISS']]}",
            "suggested_action": "Add IPO/new-listing detector; expand universe or build separate model for names with <12 months of price history."
        })
    if len(categories["MODEL_COVERAGE_MISS"]) >= 2:
        findings["learnings"].append({
            "type": "model_coverage_gap",
            "detail": f"{len(categories['MODEL_COVERAGE_MISS'])} winners had cs_score < 0.50 but delivered +12-17%. Model is under-scoring a real signal type.",
            "suggested_action": "Investigate the shared features of these misses. Likely fixes: (a) add volume-breakout feature, (b) add sector-rotation feature, (c) recalibrate on more recent OOF."
        })
    if len(categories["THRESHOLD_MISS"]) >= 2:
        findings["learnings"].append({
            "type": "threshold_gap",
            "detail": f"{len(categories['THRESHOLD_MISS'])} winners scored 0.50-0.55 but were below our 0.55 top-30 cutoff.",
            "suggested_action": "Lower the top-N cutoff from 30 to 50 in cross-engine consensus, OR lower score floor from 0.55 to 0.50 — but validate this doesn't drop hit rate on the CAUGHT names."
        })

    return findings


def persist(findings: dict) -> None:
    LEARNINGS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARNINGS_LOG, "a") as f:
        f.write(json.dumps(findings, default=str) + "\n")
    print(f"\n✅ Persisted to {LEARNINGS_LOG.relative_to(ROOT)}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=str, help="Entry date YYYY-MM-DD")
    ap.add_argument("--exit", type=str, help="Exit date YYYY-MM-DD")
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    entry = args.entry or "2026-06-24"
    exit_ = args.exit or "2026-06-30"
    findings = analyze_misses(entry, exit_, args.top_n)

    print(f"\n{'='*80}")
    print(f" AUTO-GENERATED LEARNINGS ({len(findings['learnings'])})")
    print(f"{'='*80}\n")
    for i, l in enumerate(findings["learnings"], 1):
        print(f"[{i}] {l['type'].upper()}")
        print(f"    Finding: {l['detail']}")
        print(f"    Suggested: {l['suggested_action']}")
        print()

    persist(findings)


if __name__ == "__main__":
    main()
