"""Write today's live_predictions/YYYY-MM-DD.json from current pipeline outputs.

Cross-validates Pipeline-1 (find_180d_frontier_honest) with Pipeline-2
(find_multibagger_targets) and assigns Tier-1/2/3 based on:
  - Tier-1: in BOTH top-5 → highest confidence
  - Tier-2: in only one top-5
  - Tier-3: clean per-name but not in top-5

Per-name contamination filter applied (drops names with any single-day
return < -30% in their history when no matching corporate-action record
exists). Documented in logs/calibration_corrections.jsonl 2026-05-04.

Caveats embedded in JSON:
- Calibrator bias from un-adjusted corporate actions (Phase 2 pending)
- 100%-threshold band empirically empty in 2025 OOS (treat as relative ranking)
"""
from __future__ import annotations
import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
P1 = ROOT / "data/derived/180d_today_predictions.parquet"
P2 = ROOT / "data/derived/multibagger_today_predictions.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA = ROOT / "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet"


def per_name_contamination(symbols: list[str]) -> dict[str, list[str]]:
    """Flag names with any -30%+ drop without a matching CA record."""
    if not PRICES.exists():
        return {s: [] for s in symbols}
    prices = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close"])
    prices = prices[prices["symbol"].isin(symbols)].copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices = prices.sort_values(["symbol", "trade_date"])
    prices["ret"] = prices.groupby("symbol")["close"].pct_change()

    ca = pd.DataFrame()
    if CA.exists():
        ca = pd.read_parquet(CA)
        ca["ex_date"] = pd.to_datetime(ca["ex_date"])

    flags = {}
    for s in symbols:
        sub = prices[prices["symbol"] == s]
        drops = sub[sub["ret"] < -0.30]
        issues = []
        for _, r in drops.iterrows():
            if not ca.empty:
                near = ca[(ca["symbol"] == s) &
                          (abs(ca["ex_date"] - r["trade_date"]) <= pd.Timedelta(days=5))]
            else:
                near = ca
            if len(near) == 0:
                issues.append(f"{r['trade_date'].date()}: {r['ret']*100:.1f}% (no CA record)")
        flags[s] = issues
    return flags


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD")
    args = ap.parse_args()

    # Load both pipelines
    if not P1.exists() or not P2.exists():
        raise SystemExit(f"missing prediction parquets: P1={P1.exists()} P2={P2.exists()}")
    p1 = pd.read_parquet(P1)
    p1 = p1[p1["adv_20d_cr"] >= 1.0]
    p1_top10 = p1.sort_values("score_100pct", ascending=False).head(10)
    p1_top5 = set(p1_top10.head(5)["symbol"])

    p2 = pd.read_parquet(P2)
    p2 = p2[p2["adv_20d_cr"] >= 1.0]
    p2_top10 = p2.sort_values("score_100pct_180d", ascending=False).head(10)
    p2_top5 = set(p2_top10.head(5)["symbol"])

    # Tier classification
    cross_validated = sorted(p1_top5 & p2_top5)
    p1_only = sorted(p1_top5 - p2_top5)
    p2_only = sorted(p2_top5 - p1_top5)

    # Contamination filter on all candidates
    all_candidates = cross_validated + p1_only + p2_only
    flags = per_name_contamination(all_candidates)
    contaminated = [s for s, issues in flags.items() if issues]

    # Build clean tiers
    tier1 = [s for s in cross_validated if not flags.get(s)]
    tier2 = [s for s in (p1_only + p2_only) if not flags.get(s)]
    dropped = contaminated

    def pick_row(sym, df, score_col):
        sub = df[df["symbol"] == sym]
        if sub.empty:
            return None
        r = sub.iloc[0]
        return {
            "symbol": sym,
            "close_at_call": float(r["close"]),
            "score": round(float(r[score_col]), 4),
            "rsi_14": round(float(r.get("rsi_14_daily", 0)), 1),
            "adv_20d_cr": round(float(r["adv_20d_cr"]), 2),
        }

    out = {
        "as_of_date": args.date,
        "horizon_days": 180,
        "resolution_date": (datetime.fromisoformat(args.date) + timedelta(days=180)).date().isoformat(),
        "method": "cross_validated_two_pipelines + contamination_filter",
        "auto_generated_by": "src/agentic/daily_refresh.sh -> write_live_prediction.py",
        "ts_committed_utc": datetime.utcnow().isoformat() + "Z",
        "tier1_cross_validated_clean": [pick_row(s, p1_top10, "score_100pct") for s in tier1],
        "tier2_single_pipeline_clean": [pick_row(s, p1_top10, "score_100pct") or pick_row(s, p2_top10, "score_100pct_180d") for s in tier2],
        "dropped_due_to_contamination": [
            {"symbol": s, "issues": flags[s]} for s in dropped
        ],
        "sizing_rule": {
            "tier1_per_name_pct": 3.0,
            "tier2_per_name_pct": 1.5,
            "max_basket_total_pct": 12.5,
            "hard_sl_per_name_pct": -15,
            "trail_sl_target_pct": 30,
        },
        "constitution_caveats": [
            "Per CONSTITUTION.md §1.1 — calibrator carries upward bias from unadjusted corporate actions (Phase 2 pending). Treat scores as relative rankings.",
            "Per CONSTITUTION.md §1.4 — 80%-confidence at 100% threshold remains empirically empty.",
            "Per CONSTITUTION.md §1.5 — basket cap 12.5%; rest in LIQUIDPLUS at ~7%.",
        ],
    }

    out_path = ROOT / f"live_predictions/{args.date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_path.relative_to(ROOT)}")
    print(f"  tier-1 (cross-validated, clean): {tier1}")
    print(f"  tier-2 (single-pipeline, clean): {tier2}")
    print(f"  dropped (contamination):         {dropped}")


if __name__ == "__main__":
    main()
