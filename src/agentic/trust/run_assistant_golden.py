"""Ask a fresh headless Claude the golden questions (evals/assistant_golden.yaml) and grade the answers (2026-09-29).

Plain English: this measures how often Claude gets facts about this system wrong. Each question has a known answer in
the repo; a fresh Claude session (read-only tools: Read, Grep, Glob) answers it; a regex grades it. Trap questions ask
about things that don't exist — the right answer says so instead of inventing something.
Result: accuracy and the list of wrong answers, logged to logs/trust/golden_runs.jsonl (one line per run, answers kept).
Usage: /usr/bin/python3 src/agentic/trust/run_assistant_golden.py [--limit N] [--ids a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/Users/abhinavs./Documents/Zoom")
LOG = ROOT / "logs/trust/golden_runs.jsonl"
CLAUDE = shutil.which("claude") or "/opt/homebrew/bin/claude"


def ask(q: str, timeout: int = 300) -> str:
    env = dict(os.environ, SGM_CLAIMCHECK_QUIET="1")
    r = subprocess.run([CLAUDE, "-p", q, "--allowedTools", "Read,Grep,Glob", "--output-format", "text"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or r.stderr).strip()


def grade(item: dict, ans: str) -> bool:
    ok = re.search(item["must_match"], ans, re.I) is not None
    if item.get("must_not_match") and re.search(item["must_not_match"], ans, re.I):
        ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    qs = yaml.safe_load((ROOT / "evals/assistant_golden.yaml").read_text())["questions"]
    if a.ids:
        qs = [q for q in qs if q["id"] in a.ids.split(",")]
    if a.limit:
        qs = qs[:a.limit]
    rows = []
    for item in qs:
        try:
            ans = ask(item["q"])
        except subprocess.TimeoutExpired:
            ans = "TIMEOUT"
        if re.search(r"Failed to authenticate|authentication_error|Please run /login|Invalid API key", ans):
            raise SystemExit("claude CLI is not logged in — run `claude` once in Terminal and /login, then re-run. Nothing logged.")
        ok = grade(item, ans)
        rows.append(dict(id=item["id"], correct=ok, answer=ans[:600]))
        print(f"{'✅' if ok else '❌'} {item['id']:24s} {ans[:110].replace(chr(10), ' ')}", flush=True)
    acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), n=len(rows), accuracy=round(acc, 3),
                                 wrong=[r["id"] for r in rows if not r["correct"]], answers=rows)) + "\n")
    print(f"\naccuracy {acc:.0%} ({sum(r['correct'] for r in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
