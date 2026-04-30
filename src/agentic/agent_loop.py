"""Agent loop — one cycle of hypothesis → fetch → compile → evaluate → verdict.

Reads factor_registry.json. For each PROPOSED hypothesis:
  • If has_data=True  → compile (via feature_factory) + run factor_evaluator
  • If has_data=False → identify fetcher (best match by data_needed name);
                        if a fetcher script exists, run it & re-check; else mark BLOCKED
  • Updates registry state inline + writes a cycle log.

For IC_PASSED hypotheses (cross-sectional gate cleared but portfolio A/B
not yet run), the loop SUGGESTS running backtest_10yr_with_factors.py
but does not auto-run (it's a 30-min job).

Run on-demand:
  PYTHONPATH=. /usr/bin/python3 src/agentic/agent_loop.py

Or schedule:
  weekly via LaunchAgent / cron / scheduled-tasks MCP
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
REGISTRY = ROOT / "data/derived/factor_registry.json"
CYCLE_LOG = ROOT / "logs/agent_loop_cycles.jsonl"

# Map data-needed names → fetcher script names
FETCHER_MAP = {
    "fii_net_inr":           "fetch_fii_dii.py",
    "dii_net_inr":           "fetch_fii_dii.py",
    "usdinr_close":          "fetch_forex_macro.py",
    "brent_close":           "fetch_forex_macro.py",   # not yet wired but flag
    "india_10y_yield":       None,  # blocked
    "india_vix":             None,  # blocked (NSE endpoint)
    "nifty_pe":              None,  # not implemented
    "google_trends_7d":      None,  # blocked (pytrends throttle)
    "wiki_pageviews_7d":     "fetch_wiki_pageviews.py",
    "iv_skew":               None,  # IP-blocked
    "max_pain":              None,  # IP-blocked
    "futures_oi":            None,  # not implemented
    "consensus_pat":         None,  # paid data
    "promoter_pledge_pct":   None,
    "promoter_holding_pct":  None,
    "credit_rating_action":  None,
}


def load_registry() -> list[dict]:
    return json.loads(REGISTRY.read_text())


def save_registry(reg: list[dict]) -> None:
    REGISTRY.write_text(json.dumps(reg, indent=2))


def log_cycle(record: dict) -> None:
    CYCLE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_step(cmd: list[str], description: str, timeout: int = 600) -> dict:
    print(f"  → {description}")
    started = time.time()
    try:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                                 text=True, timeout=timeout)
        ok = result.returncode == 0
        return {
            "ok": ok,
            "elapsed": round(time.time() - started, 1),
            "stdout_tail": result.stdout[-500:] if result.stdout else "",
            "stderr_tail": result.stderr[-500:] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "elapsed": timeout, "error": "TIMEOUT"}
    except Exception as exc:
        return {"ok": False, "elapsed": time.time() - started, "error": str(exc)}


def cycle(max_proposed: int = 5, run_evaluator: bool = True) -> None:
    print(f"== agent_loop cycle  {datetime.now():%Y-%m-%d %H:%M} ==")
    reg = load_registry()
    proposed = [h for h in reg if h["state"] == "PROPOSED"]
    print(f"PROPOSED hypotheses: {len(proposed)}  (will work on up to {max_proposed})")

    actions: list[dict] = []
    fetchers_to_run: set[str] = set()
    blocked_count = 0

    for h in proposed[:max_proposed]:
        action = {"id": h["id"], "name": h["name"], "category": h["category"]}
        if h["has_data"]:
            action["plan"] = "compile + evaluate (data available)"
            action["status"] = "ready"
        else:
            # find fetcher
            fetcher = None
            for need in h["data_needed"]:
                if need in FETCHER_MAP and FETCHER_MAP[need]:
                    fetcher = FETCHER_MAP[need]
                    break
            if fetcher and (ROOT / "src/agentic" / fetcher).exists():
                action["plan"] = f"queue fetcher {fetcher}, then compile + evaluate"
                action["status"] = "queued"
                fetchers_to_run.add(fetcher)
            else:
                action["plan"] = "no fetcher available — BLOCKED"
                action["status"] = "blocked"
                h["state"] = "BLOCKED"
                blocked_count += 1
        actions.append(action)
        print(f"  [{h['id']}] {h['name'][:60]:<60}  → {action['status']}")

    # collect IC_PASSED for human follow-up
    ic_passed = [h for h in reg if h["state"] == "IC_PASSED"]
    if ic_passed:
        print(f"\n{len(ic_passed)} hypothesis(es) at IC_PASSED — awaiting portfolio A/B test:")
        for h in ic_passed:
            print(f"  - {h['id']}: {h['name']}")
        print(f"  → suggest running backtest_10yr_with_factors.py with these factors enabled")

    # run feature_factory + evaluator if any READY hypothesis
    if run_evaluator and any(a["status"] == "ready" for a in actions):
        print("\nrunning feature_factory + factor_evaluator …")
        ff = run_step(["/usr/bin/python3", "src/agentic/feature_factory.py"],
                      "feature_factory.py", timeout=600)
        actions.append({"step": "feature_factory", **ff})
        if ff["ok"]:
            ev = run_step(["/usr/bin/python3", "src/agentic/factor_evaluator.py"],
                          "factor_evaluator.py", timeout=900)
            actions.append({"step": "factor_evaluator", **ev})

    save_registry(reg)

    record = {
        "ts": datetime.now().isoformat(),
        "n_proposed_at_start": len(proposed),
        "n_processed": min(len(proposed), max_proposed),
        "n_blocked": blocked_count,
        "n_ic_passed_pending_ab": len(ic_passed),
        "fetchers_queued": sorted(fetchers_to_run),
        "actions": actions,
    }
    log_cycle(record)
    print(f"\ncycle log appended → {CYCLE_LOG}")
    print(f"summary: processed={record['n_processed']} blocked={blocked_count}  "
          f"ic_passed_pending={len(ic_passed)}  fetchers_queued={len(fetchers_to_run)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=5,
                    help="max number of PROPOSED hypotheses to inspect this cycle")
    ap.add_argument("--no-eval", action="store_true",
                    help="skip running feature_factory + factor_evaluator")
    args = ap.parse_args()
    cycle(max_proposed=args.max, run_evaluator=not args.no_eval)


if __name__ == "__main__":
    main()
