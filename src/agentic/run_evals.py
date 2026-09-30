"""Run the eval registry (evals/registry.yaml) and write a report a person can read.

Every eval is printed with its plain-English statement next to the measured result, so the report reads as
"what must be true" -> "is it true today". Human evals (owner abhinav) show PENDING until a person has done them.
Exit code 1 when an eval with action=block FAILS (PENDING human sign-off does not fail the daily run; it blocks real
money, which is stated in the report).
Usage: /usr/bin/python3 src/agentic/run_evals.py [--cadence daily|weekly|monthly|per_experiment|all]
                                                  [--only PREFIX,PREFIX --gate NAME]
Output: reports/eval_report_<date>.md, logs/evals/eval_run_<date>.json
Gate mode (--gate, used by run_sgm.py between stages): only evals whose id starts with one of --only, written to
logs/evals/gate_<NAME>_<date>.json; the daily report files are not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import eval_checks as ec  # noqa: E402

ICON = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "PENDING": "⏳", "SKIP": "·"}
CADENCE_ORDER = {"daily": ["daily"], "weekly": ["daily", "weekly", "per_experiment"],
                 "monthly": ["daily", "weekly", "per_experiment", "monthly"], "all": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default="all", choices=list(CADENCE_ORDER) + ["per_experiment"])
    ap.add_argument("--only", default="", help="comma-separated eval id prefixes")
    ap.add_argument("--gate", default="", help="gate name: write logs/evals/gate_<name>_<date>.json only")
    a = ap.parse_args()
    only = tuple(x.strip() for x in a.only.split(",") if x.strip())
    reg = yaml.safe_load((ROOT / "evals/registry.yaml").read_text())["evals"]
    allow = CADENCE_ORDER.get(a.cadence, [a.cadence])
    ctx = dict(last_session=ec.last_session())
    rows = []
    for e in reg:
        if allow is not None and e["cadence"] not in allow:
            continue
        if only and not e["id"].startswith(only):
            continue
        t0 = time.time()
        fn = ec.REGISTRY.get(e["check"])
        try:
            r = fn(ctx) if fn else dict(status="SKIP", value=None, detail=f"no check named {e['check']}")
        except Exception as x:                                        # a broken check is itself a failure
            r = dict(status="FAIL", value=None, detail=f"check crashed: {type(x).__name__}: {x}"[:300])
        rows.append(dict(id=e["id"], statement=e["statement"], action=e["action"], owner=e["owner"],
                         threshold=e["threshold"], **r, secs=round(time.time() - t0, 1)))
    blocking = [r for r in rows if r["action"] == "block" and r["status"] == "FAIL"]
    human_pending = [r for r in rows if r["status"] == "PENDING"]
    today = datetime.now().strftime("%Y-%m-%d")
    verdict = "BLOCKED" if blocking else ("OK — waiting on human sign-off for real money" if human_pending else "OK")

    if a.gate:
        (ROOT / f"logs/evals/gate_{a.gate}_{today}.json").write_text(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"),
            gate=a.gate, only=list(only), verdict=verdict, results=rows), indent=1, default=str))
        for r in rows:
            print(f"{ICON.get(r['status'], ''):2s} {r['status']:7s} {r['id']:28s} {str(r['value'] or ''):38s} {r['detail'][:90]}")
        print(f"\nGATE {a.gate}: {verdict}")
        return 1 if blocking else 0

    md = [f"# Eval report — {today} (data through {ctx['last_session'].date()})", "",
          f"**Verdict: {verdict}.** {len(rows)} evals · {sum(r['status'] == 'PASS' for r in rows)} pass · "
          f"{len(blocking)} blocking failures · {sum(r['status'] == 'WARN' for r in rows)} warnings · {len(human_pending)} waiting on a person.", "",
          "| | Eval | What must be true | Result | Action if it fails |", "|---|---|---|---|---|"]
    for r in rows:
        res = f"**{r['status']}** {r['value'] if r['value'] is not None else ''}" + (f" — {r['detail']}" if r["detail"] else "")
        md.append(f"| {ICON.get(r['status'], '')} | `{r['id']}` | {r['statement']} | {res.replace('|', '/')} | {r['action']} ({r['owner']}) |")
    md += ["", "Registry: `evals/registry.yaml` (statement, why, source of truth, threshold, action, owner). Checks: `src/agentic/eval_checks.py`."]
    (ROOT / f"reports/eval_report_{today}.md").write_text("\n".join(md) + "\n")
    (ROOT / f"logs/evals/eval_run_{today}.json").write_text(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), cadence=a.cadence,
        verdict=verdict, last_session=str(ctx["last_session"].date()), results=rows), indent=1, default=str))
    for r in rows:
        print(f"{ICON.get(r['status'], ''):2s} {r['status']:7s} {r['id']:28s} {str(r['value'] or ''):38s} {r['detail'][:90]}")
    print(f"\nVERDICT: {verdict}  ->  reports/eval_report_{today}.md")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
