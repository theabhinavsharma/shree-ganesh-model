"""Reproduction check (2026-09-29): re-run a registered test from the committed code and compare with the logged result.

Plain English: a result only counts if the code in the repo prints the same numbers again. This re-runs the test with
SGM_REPRO=1 (the test writes to a repro/ folder and does NOT add a line to the experiment log), then compares every
arm's return per year, per period and worst fall with the RESULT line in logs/experiments.jsonl.
  REPRODUCED      all numbers within 0.1 point
  DRIFTED         within 1 point, and the price data has been refreshed since the result was logged
  NOT REPRODUCED  anything else
Usage: /usr/bin/python3 src/agentic/trust/reproduce.py [EXP-ID ...]   (default: every experiment listed in TESTS)
Log: logs/trust/reproductions.jsonl
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
LOG = ROOT / "logs/trust/reproductions.jsonl"
PANEL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
TESTS = {   # experiment id -> (script, results.csv written in repro mode)
    "EXP-2026-09-29-industry-fundamentals": ("src/agentic/test_industry_fundamentals.py", "logs/leader_sleeve/industry_fundamentals/repro/results.csv"),
    "EXP-2026-09-29-industry-policy": ("src/agentic/test_industry_policy.py", "logs/leader_sleeve/industry_policy/repro/results.csv"),
}
COLS = ("cagr", "cagr_disc", "cagr_conf", "maxdd")


def logged(exp: str) -> dict:
    last = None
    for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines():
        if l.strip().startswith("{"):
            e = json.loads(l)
            if e.get("id") == exp + "-RESULT":
                last = e
    if last is None:
        raise SystemExit(f"no RESULT logged for {exp}")
    return last


def check(exp: str) -> dict:
    script, out = TESTS[exp]
    res = logged(exp)
    env = dict(os.environ, SGM_REPRO="1")
    r = subprocess.run(["nice", "-n", "10", "/usr/bin/python3", script], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return dict(exp=exp, status="NOT REPRODUCED", detail=f"script failed: {r.stderr.strip().splitlines()[-1:]}")
    now = pd.read_csv(ROOT / out).set_index("arm")
    diffs = {}
    for arm, vals in res["arms"].items():
        if arm in now.index:
            diffs[arm] = max(abs(float(now.loc[arm, c]) - float(v)) for c, v in zip(COLS, vals[:4]))
    worst = max(diffs.values()) if diffs else float("inf")
    refreshed = datetime.fromtimestamp(PANEL.stat().st_mtime) > datetime.fromisoformat(res["ts"])
    status = "REPRODUCED" if worst <= 0.1 else ("DRIFTED" if worst <= 1.0 and refreshed else "NOT REPRODUCED")
    return dict(exp=exp, status=status, worst_diff_pts=round(worst, 2), data_refreshed_since=refreshed,
                per_arm={k: round(v, 2) for k, v in diffs.items()}, logged_at=res["ts"])


def main() -> None:
    ids = sys.argv[1:] or list(TESTS)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for exp in ids:
        row = dict(ts=datetime.now().isoformat(timespec="seconds"), **check(exp))
        with LOG.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f"{row['status']:15s} {exp}  worst diff {row.get('worst_diff_pts')} pts  {row.get('detail', '')}")


if __name__ == "__main__":
    main()
