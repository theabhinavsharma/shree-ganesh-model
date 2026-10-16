"""Weekly re-run of the registered tests behind the live rule (2026-10-16, INC-2026-10-16-reproduce-blocks-on-market-drift).

Plain English: every Friday a week of prices is added and the model scores are rebuilt, so the registered tests' numbers
move (1.6-3.6 points between Oct 12 and Oct 16). Comparing them with the previous week's log blocked the buy list on Oct 9
and Oct 16 for market movement alone. Now the Friday run first re-runs every test in reproduce.TESTS on this week's data
(each passes its own data-readiness gate and logs <exp>-RERUN-weekly-<date>), in order (the V3.2 test checks itself
against the V3.1 test's fresh numbers). Then:
  trust.live_rule_verdicts (block)  the arms in reproduce.LIVE_ARMS still pass
  trust.verdict_changes (warn)      any verdict that differs from the previous week's is listed
  trust.results_reproduce (block)   reproduce.py re-runs each test again and must match these fresh numbers
Usage: rerun_registered.py [tag]  (default weekly-<date>). Exit 1 if any test fails to run (script error, wiring check, data gate). Log: logs/trust/reruns.jsonl
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce  # noqa: E402

ROOT = reproduce.ROOT
LOG = ROOT / "logs/trust/reruns.jsonl"
PER_TEST_SECS = 1800


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else f"weekly-{datetime.now():%Y%m%d}"; bad = 0   # optional tag for a by-hand re-run after a data fix
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for exp, (script, _) in reproduce.TESTS.items():
        env = dict(os.environ, SGM_RERUN_TAG=tag, **reproduce.ENV.get(exp, {})); env.pop("SGM_REPRO", None); t0 = time.time()
        try:
            r = subprocess.run(["nice", "-n", "10", "/usr/bin/python3", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=PER_TEST_SECS)
            ok, detail = r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else ""
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timed out after {PER_TEST_SECS}s"
        if ok:
            new = reproduce.logged(exp)
            ok = new["id"] == f"{exp}-RERUN-{tag}" and new["ts"] >= datetime.fromtimestamp(t0).isoformat(timespec="seconds")
            detail = f"verdict {new.get('verdict')}" if ok else "ran but logged no RERUN line"
        bad += not ok
        row = dict(ts=datetime.now().isoformat(timespec="seconds"), exp=exp, tag=tag, ok=ok, secs=round(time.time() - t0), detail=detail)
        with LOG.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"{'OK  ' if ok else 'FAIL'} {exp} ({row['secs']}s) {detail}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
