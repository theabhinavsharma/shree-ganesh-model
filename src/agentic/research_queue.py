"""New-data research queue runner (2026-09-30): one feed at a time, slowly, each with its registered A/B test.

Plain English: configs/research_queue.json lists new data to add, most useful and cheapest first. Every hour
(launchd com.sgm.research) this takes the first unfinished item and does the next step:
  fetch   run its fetcher for up to ~45 minutes (polite, resumable); when the fetcher prints COMPLETE, move on
  test    run its registered A/B test; read the RESULT line from logs/experiments.jsonl
  report  send the result to Telegram (templated from the RESULT line), mark the item done, start the next
An item whose code does not exist yet is marked "needs code" and Telegram says so once; the daily message lists the
queue every day as the reminder. Never runs while the daily/weekly orchestrator holds its lock.
Usage: research_queue.py run | status
State: logs/research_queue/state.json · log: logs/research_queue/<ts>_<item>_<step>.log
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/research_queue.json"
DIR = ROOT / "logs/research_queue"
STATE = DIR / "state.json"
LOCK = DIR / ".lock"
ORCH_LOCK = ROOT / "logs/runs/.lock"
sys.path.insert(0, str(ROOT / "src/agentic"))


def load() -> tuple[list, dict]:
    items = json.loads(CFG.read_text())["items"]
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    return items, st


def save(st: dict) -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def has_code(it: dict) -> bool:
    return all((ROOT / it[k][0]).exists() for k in ("fetch", "test"))


def status_lines() -> list[str]:
    """For the daily message: where the queue stands (the daily reminder)."""
    if not CFG.exists():
        return []
    items, st = load()
    lines, now_set = [], False
    for i, it in enumerate(items, 1):
        s = st.get(it["id"], {})
        state = s.get("state") or ("waiting" if has_code(it) else "needs code")
        if state == "done":
            lines.append(f"✅ {i}. {it['title'].split(' (')[0]}: {s.get('verdict', 'done')}")
        elif not now_set:
            prog = f" · {s['progress']}" if s.get("progress") else ""
            lines.append(f"▶ {i}. {it['title'].split(' (')[0]}: {state}{prog}" + (" (ask Claude to build it)" if state == "needs code" else ""))
            now_set = True
        else:
            lines.append(f"· {i}. {it['title'].split(' (')[0]}" + (" (needs code)" if not has_code(it) else ""))
    return ["", f"🧪 NEW DATA QUEUE ({sum(st.get(it['id'], {}).get('state') == 'done' for it in items)}/{len(items)} done)"] + lines


def _run(cmd: list[str], log: Path, env: dict | None = None, timeout: int = 3 * 3600) -> tuple[int, str]:
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin", **(env or {}))
    with log.open("w") as fh:
        rc = subprocess.run(["nice", "-n", "10", "/usr/bin/python3", *cmd], cwd=ROOT, env=e, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    return rc, log.read_text()[-4000:]


def _result(rid: str) -> dict | None:
    last = None
    for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines():
        if l.strip().startswith("{"):
            e = json.loads(l)
            if e.get("id") == rid:
                last = e
    return last


def report(i: int, n: int, it: dict, res: dict | None, fetch_tail: str) -> str:
    import notify
    lines = [f"🧪 New data {i}/{n} done: {it['title']}"]
    fl = [l for l in fetch_tail.splitlines() if l.startswith(("listed", "parsed", "rows", "fetched"))]
    if fl:
        lines.append("Fetch: " + fl[-1][:200])
    if res:
        v = res.get("verdict")
        lines.append("A/B verdict: " + ("PASS: " + ", ".join(v) if isinstance(v, list) and v else str(v)))
        arms = res.get("arms", {})
        ref = next((k for k in ("G1", "S1M") if k in arms), None)
        for k, a in arms.items():
            lines.append(f"• {k}: {a[0]:.1f}%/yr (2019-22 {a[1]:.1f}%, 2023+ {a[2]:.1f}%) · worst fall {a[3]:.1f}%" + (f" · beat ref in {a[4]} phases" if a[4] else ""))
        if ref:
            lines.append(f"(reference = {ref}; a rule counts only if it beats it in both periods without a deeper fall)")
    else:
        lines.append("A/B: no RESULT line found; see logs/research_queue/")
    nxt = [x for x in json.loads(CFG.read_text())["items"]][i:i + 1]
    if nxt:
        lines.append(f"Next: {nxt[0]['title']}" + ("" if has_code(nxt[0]) else " (needs code: ask Claude)"))
    text = "\n".join(lines)
    notify.send(text, "research")
    return text


def run() -> int:
    if ORCH_LOCK.exists():
        print("orchestrator running: skip this hour"); return 0
    DIR.mkdir(parents=True, exist_ok=True)
    try:
        LOCK.mkdir()
    except FileExistsError:
        print("queue already running"); return 0
    try:
        items, st = load()
        for i, it in enumerate(items, 1):
            s = st.setdefault(it["id"], {})
            if s.get("state") == "done":
                continue
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not has_code(it):
                if s.get("state") != "needs code":
                    import notify
                    notify.send(f"🧪 New data queue: item {i}/{len(items)} needs code before it can run: {it['title']}. "
                                "Ask Claude to build its fetcher and registered test.", "research")
                s.update(state="needs code", since=ts); save(st)
                return 0
            if s.get("state") in (None, "waiting", "needs code", "fetching"):
                s.update(state="fetching", started=s.get("started") or ts); save(st)
                rc, tail = _run(it["fetch"], DIR / f"{ts}_{it['id']}_fetch.log")
                prog = [l for l in tail.splitlines() if l.startswith(("listed", "parsed", "rows", "fetched"))]
                s.update(progress=prog[-1][:120] if prog else f"exit {rc}", last_fetch=ts, fetch_tail=tail[-1500:])
                if "COMPLETE" not in tail:
                    save(st); return 0
                s.update(state="testing"); save(st)
            if s["state"] == "testing":
                rc, tail = _run(it["test"], DIR / f"{ts}_{it['id']}_test.log", env=it.get("test_env"))
                res = _result(it["result_id"]) if it.get("result_id") else None
                if rc != 0 or (it.get("result_id") and not res):
                    s.update(state="test failed", detail=tail[-600:]); save(st)
                    import notify
                    notify.send(f"❌ New data queue: the A/B test for '{it['title']}' failed. Log: logs/research_queue/{ts}_{it['id']}_test.log", "fail")
                    return 1
                v = res.get("verdict") if res else None
                s.update(state="done", finished=ts, verdict=("PASS: " + ", ".join(v)) if isinstance(v, list) and v else str(v))
                save(st)
                report(i, len(items), it, res, s.get("fetch_tail", ""))
                continue                      # next item starts in the same run if its code exists
            if s["state"] == "test failed":
                return 1
        return 0
    finally:
        LOCK.rmdir()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        sys.exit(run())
    print("\n".join(status_lines()))
