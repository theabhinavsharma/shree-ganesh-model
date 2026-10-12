"""Compare each registered test's original RESULT with its RERUN under a tag (2026-10-12 cash-flow fix: dividends counted,
big payouts / demergers as cash, model labels rebuilt). Prints and writes logs/trust/rerun_<tag>.json + .md.
Usage: /usr/bin/python3 src/agentic/trust/compare_rerun.py cashfix-20261012
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "cashfix-20261012"
    L = [json.loads(l) for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    first, rerun, ready = {}, {}, {}
    for e in L:
        i = str(e.get("id", ""))
        if i.endswith("-RESULT") and i[:-7] not in first:
            first[i[:-7]] = e
        if i.endswith(f"-RERUN-{tag}"):
            rerun[i[: -len(f"-RERUN-{tag}")]] = e
        if i.endswith("-DATA-READY") and e.get("run") == tag:
            ready[i[: -len("-DATA-READY")]] = e.get("status")
    rows = []
    for exp, e0 in first.items():
        e1 = rerun.get(exp)
        v0 = e0.get("verdict"); v1 = e1.get("verdict") if e1 else None
        a0, a1 = e0.get("arms") or e0.get("table") or {}, (e1.get("arms") or e1.get("table") or {}) if e1 else {}
        nums = {}
        for k in a1:
            if k in a0 and isinstance(a0[k], list) and isinstance(a1[k], list):
                nums[k] = dict(before=a0[k][:4], after=a1[k][:4])
        status = ("re-run" if e1 else ("NOT READY on current data" if ready.get(exp) == "NOT READY" else "not re-run"))
        rows.append(dict(exp=exp, status=status, verdict_before=v0, verdict_after=v1,
                         verdict_changed=(e1 is not None and json.dumps(v0, sort_keys=True) != json.dumps(v1, sort_keys=True)), arms=nums))
    out = dict(tag=tag, made=datetime.now().isoformat(timespec="seconds"), tests=rows)
    (ROOT / f"logs/trust/rerun_{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    md = [f"# Re-run of every registered test: {tag}", "", "| Test | Status | Verdict before | Verdict after | Changed |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['exp']} | {r['status']} | {r['verdict_before']} | {r['verdict_after'] if r['verdict_after'] is not None else '—'} | {'**yes**' if r['verdict_changed'] else ''} |")
    (ROOT / f"logs/trust/rerun_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
