"""Never-sleeping hypothesis loop.

Goal: keep generating + testing factor hypotheses until at least one name
clears the 80% calibrated bar on any of (5%/7d, 10%/15d, 20%/30d).

Each cycle (~30 min):
  1. Pick top-N PROPOSED hypotheses from registry (priority: behavioral, ownership,
     macro_conditional, alt_market — these have highest theoretical alpha based on
     the 75-hypothesis catalog)
  2. For each hypothesis with `has_data=True`: compile via feature_factory
  3. Re-train find_high_conviction with the augmented feature set
  4. If any name clears 0.80 → STOP, emit success alert
  5. Else → log cycle, fetch one new dataset (round-robin queue), continue

Output:
  logs/hypothesis_loop_log.jsonl — append-only ledger of every cycle
  reports/conviction_alert_<date>.md — refreshed at end of every cycle

Stops when:
  - >=1 name clears 0.80 calibrated on any target (the explicit success condition)
  - User TaskStops the process
  - Hard limit: 50 cycles (~25 hours) — safety so it doesn't run forever
"""
from __future__ import annotations
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
LOG = ROOT / "logs/hypothesis_loop_log.jsonl"
REGISTRY = ROOT / "data/derived/factor_registry.json"
PRED = ROOT / "data/derived/high_conviction_predictions.parquet"

# Pool of fetchers we cycle through (one per cycle)
# REGULAR (per-stock micro) + NEW MACRO/AGGREGATE (non-conventional)
FETCHER_QUEUE = [
    # — per-stock / conventional —
    "src/agentic/refresh_prices.py",
    "src/agentic/refresh_announcements.py",
    "src/agentic/fetch_news_rss.py",
    "src/agentic/fetch_news_per_symbol.py",
    "src/agentic/fetch_reddit.py",
    "src/agentic/fetch_youtube.py",
    "src/agentic/fetch_block_deals.py",
    "src/agentic/fetch_fii_dii.py",
    "src/agentic/fetch_wiki_pageviews.py",
    "src/agentic/fetch_screener_screens.py",
    "src/agentic/fetch_screener_fundamentals.py",
    "src/agentic/fetch_superstar_holdings.py",
    "src/agentic/score_sentiment.py",
    # — MACRO / AGGREGATE / NON-CONVENTIONAL (new) —
    "src/agentic/fetch_forex_macro.py",
    "src/agentic/fetch_global_macro.py",
    "src/agentic/fetch_commodity_prices.py",
    "src/agentic/fetch_global_rates.py",
    "src/agentic/fetch_amfi_mf_holdings.py",
    "src/agentic/fetch_market_breadth.py",
    "src/agentic/fetch_industry_indicators.py",
    "src/agentic/fetch_global_macro_sentiment.py",
    "src/agentic/build_macro_panel.py",   # consolidator — rebuild after any macro fetch
]

# Pipeline of feature-rebuild + retrain we run every cycle
RETRAIN_PIPELINE = [
    "src/agentic/build_derived_ratios.py",
    "src/agentic/build_academic_alphas.py",
    "src/agentic/build_macro_panel.py",  # macro consolidation before feature_factory
    "src/agentic/feature_factory.py",
    "src/agentic/find_high_conviction.py",
    "src/agentic/monitor_for_conviction.py",
]

CONVICTION = 0.80
MAX_CYCLES = 100  # bumped from 50 for the macro/aggregate exploration round
DELAY_BETWEEN_CYCLES = 60  # seconds


def run(cmd: list[str], timeout: int = 1500) -> dict:
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return {"ok": r.returncode == 0, "elapsed_s": round(time.time()-t0, 1),
                "tail": (r.stdout[-300:] + r.stderr[-300:])[-500:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed_s": timeout, "tail": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "elapsed_s": time.time()-t0, "tail": str(e)[:300]}


def check_conviction() -> tuple[bool, dict]:
    """Returns (any_clears_bar, summary_dict)."""
    if not PRED.exists():
        return False, {"error": "no predictions yet"}
    import pandas as pd
    df = pd.read_parquet(PRED)
    cols = [c for c in df.columns if c.startswith("score_") and c.endswith("_cal")]
    if not cols:
        return False, {"error": "no calibrated score columns"}
    df["best"] = df[cols].max(axis=1)
    qualifying = df[df["best"] >= CONVICTION]
    summary = {
        "max_score": float(df["best"].max()),
        "n_above_080": int((df["best"] >= 0.80).sum()),
        "n_above_075": int((df["best"] >= 0.75).sum()),
        "n_above_070": int((df["best"] >= 0.70).sum()),
        "best_symbol": df.sort_values("best", ascending=False).iloc[0]["symbol"] if len(df) else None,
    }
    if len(qualifying):
        summary["winners"] = qualifying[["symbol", "best"]].head(20).to_dict(orient="records")
    return len(qualifying) > 0, summary


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fetcher_idx = 0
    print(f"== infinite_hypothesis_loop ==")
    print(f"  goal: any name with calibrated score >= {CONVICTION} on any of (5%/7d, 10%/15d, 20%/30d)")
    print(f"  fetchers in rotation: {len(FETCHER_QUEUE)}")
    print(f"  max cycles: {MAX_CYCLES}\n")

    for cycle_num in range(1, MAX_CYCLES + 1):
        cycle_start = datetime.now()
        print(f"\n=== Cycle {cycle_num}/{MAX_CYCLES} @ {cycle_start:%Y-%m-%d %H:%M} ===")

        # 1. fetch one new dataset
        fetcher = FETCHER_QUEUE[fetcher_idx % len(FETCHER_QUEUE)]
        fetcher_idx += 1
        print(f"  ▸ fetcher: {fetcher}")
        fr = run(["/usr/bin/python3", fetcher], timeout=900)
        print(f"    {'OK' if fr['ok'] else 'FAIL'}  ({fr['elapsed_s']}s)")

        # 2. retrain pipeline
        for step in RETRAIN_PIPELINE:
            print(f"  ▸ {step}")
            sr = run(["/usr/bin/python3", step], timeout=1500)
            print(f"    {'OK' if sr['ok'] else 'FAIL'}  ({sr['elapsed_s']}s)")
            if not sr["ok"]:
                print(f"    tail: {sr['tail'][-200:]}")

        # 3. check conviction
        success, summary = check_conviction()

        record = {
            "cycle": cycle_num,
            "ts": cycle_start.isoformat(),
            "fetcher_run": fetcher,
            "elapsed_s": round((datetime.now() - cycle_start).total_seconds(), 1),
            "success": success,
            "summary": summary,
        }
        with open(LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  → max_score={summary.get('max_score', 0):.3f}, "
              f"n>=0.80={summary.get('n_above_080', 0)}, "
              f"n>=0.75={summary.get('n_above_075', 0)}")

        if success:
            print(f"\n🟢 SUCCESS — {summary['n_above_080']} name(s) clear the 80% bar:")
            print(json.dumps(summary["winners"], indent=2))
            print(f"\nStopping loop. Output in reports/conviction_alert_*.md")
            break

        print(f"  ✗ no name clears bar; sleeping {DELAY_BETWEEN_CYCLES}s before next cycle")
        time.sleep(DELAY_BETWEEN_CYCLES)


if __name__ == "__main__":
    main()
