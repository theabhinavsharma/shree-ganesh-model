"""Lean shadow check — every weekly run, compute the engine-free basket and diff vs prod.

Proves (or disproves) over time that the 5-engine layer changes nothing:
the engines affect the basket ONLY when tier1 (2+ engine consensus + QC clean) > 0.
Logs one JSONL row per run to logs/lean_shadow.jsonl. Never fails the pipeline.

After 4-6 consecutive identical weeks → evidence to demote engines to monthly diagnostic.
The first divergent week → shows exactly what consensus added.

2026-07-06 baseline: 8/8 identical, identical order, lean runtime 6.8s vs ~45min engines.
"""
from __future__ import annotations
import json, time
from datetime import date
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_hybrid_basket import build_snapshot, find_contaminated, apply_qc_filter, band_fit_score, ROOT

LOG = ROOT / "logs/lean_shadow.jsonl"


def lean_basket() -> tuple[list[dict], str, float]:
    t0 = time.time()
    prices = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet")
    ca = pd.read_parquet(ROOT / "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet")
    ml = pd.read_parquet(ROOT / "data/derived/missed_winner_classifier.parquet")

    snap, latest = build_snapshot(prices)
    contam = find_contaminated(prices, ca)
    m = snap.merge(ml[["symbol", "ml_score"]], on="symbol", how="left")

    liq = snap[snap["adv_cr"] >= 1]
    breadth = (liq["close"] > liq["sma_200"]).mean()
    regime = "DEPLOY" if breadth > 0.60 else ("WAIT" if breadth > 0.30 else "DEFENSIVE")

    clean = apply_qc_filter(m, contam)
    clean["band_fit"] = clean.apply(band_fit_score, axis=1)
    tier2 = clean[
        (clean["ml_score"].fillna(0).between(0.50, 0.75)) &
        (clean["band_fit"] >= 1.5)
    ].sort_values(["band_fit", "ml_score"], ascending=[False, False]).head(8)

    picks = [{"symbol": r["symbol"], "band_fit": round(float(r["band_fit"]), 2),
              "ml_score": round(float(r["ml_score"]), 3)} for _, r in tier2.iterrows()]
    return picks, regime, time.time() - t0, str(latest.date())


def main() -> None:
    today = date.today().isoformat()
    prod_path = ROOT / f"live_predictions/{today}_15d5pct.json"
    if not prod_path.exists():
        # fall back to newest basket file
        candidates = sorted((ROOT / "live_predictions").glob("*_15d5pct.json"))
        if not candidates:
            print("no prod basket found — skipping shadow check")
            return
        prod_path = candidates[-1]

    prod = json.loads(prod_path.read_text())
    prod_syms = [p["symbol"] for p in prod["picks"]]
    tier1_count = sum(1 for p in prod["picks"] if p.get("tier") == 1)

    lean_picks, lean_regime, elapsed, data_through = lean_basket()
    lean_syms = [p["symbol"] for p in lean_picks]

    row = {
        "run_date": today,
        "prod_file": prod_path.name,
        "data_through": data_through,
        "tier1_count": tier1_count,
        "sets_identical": set(prod_syms) == set(lean_syms),
        "order_identical": prod_syms == lean_syms,
        "prod_regime": prod.get("regime_gate"),
        "lean_regime": lean_regime,
        "lean_runtime_sec": round(elapsed, 1),
        "prod_syms": prod_syms,
        "lean_syms": lean_syms,
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(row) + "\n")

    verdict = "IDENTICAL" if row["order_identical"] else ("SAME SET, ORDER DIFFERS" if row["sets_identical"] else "DIVERGED")
    print(f"lean shadow check: {verdict}  (tier1={tier1_count}, lean {elapsed:.1f}s)")
    if not row["sets_identical"]:
        print(f"  prod-only: {sorted(set(prod_syms) - set(lean_syms))}")
        print(f"  lean-only: {sorted(set(lean_syms) - set(prod_syms))}")
    # streak summary
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    streak = 0
    for r in reversed(rows):
        if r["sets_identical"]: streak += 1
        else: break
    print(f"  identical-week streak: {streak}/{len(rows)} runs logged")


if __name__ == "__main__":
    main()
