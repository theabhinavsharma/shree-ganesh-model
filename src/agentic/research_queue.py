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


def _short(it: dict) -> str:
    return it["title"].split(" (")[0].split(":")[0]


def status_lines() -> list[str]:
    """One line for the daily message (the daily reminder)."""
    if not CFG.exists():
        return []
    items, st = load()
    done = sum(st.get(it["id"], {}).get("state") == "done" for it in items)
    cur = next((it for it in items if st.get(it["id"], {}).get("state") != "done"), None)
    if cur is None:
        return [f"🧪 Queue {done}/{len(items)} done"]
    state = st.get(cur["id"], {}).get("state") or ("waiting" if has_code(cur) else "needs code")
    return [f"🧪 Queue {done}/{len(items)} · now: {_short(cur)} ({state}{': ask Claude' if state == 'needs code' else ''})"]


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
    """Short Telegram result (2026-09-30: condensed): verdict, the passing rule(s) vs today's rule, what's next."""
    import notify
    lines = [f"🧪 {_short(it)} · A/B done ({i}/{n})", ""]
    if res:
        arms = res.get("arms", {})
        ref = next((k for k in ("G1", "S1M") if k in arms), None)
        base = res["id"].split("-RESULT")[0].split("-RERUN")[0]
        reg = next((json.loads(l) for l in (ROOT / "logs/experiments.jsonl").read_text().splitlines()
                    if l.strip().startswith("{") and json.loads(l).get("id") == base), {})
        v = res.get("verdict")
        passed = v if isinstance(v, list) else []
        r0 = arms.get(ref) if ref else None
        for k in passed:
            a = arms[k]
            what = str(reg.get("arms", {}).get(k, k)).split(";")[0]
            what = what if len(what) <= 70 else what[:70].rsplit(" ", 1)[0]
            lines.append(f"✅ PASS: {what} → {a[0]:.1f}% vs {r0[0]:.1f}%/yr" if r0 else f"✅ PASS: {what}")
        failed = [k for k in arms if k != ref and k not in passed]
        if failed:
            lines.append(f"❌ {len(failed)} {'rule' if len(failed) == 1 else 'rules'} failed" + ("" if passed else " · Sri Lakshmi unchanged"))
    else:
        lines.append("A/B: no result found (see logs/research_queue/)")
    nxt = json.loads(CFG.read_text())["items"][i:i + 1]
    if nxt:
        lines.append("")
        lines.append(f"Next: {_short(nxt[0])}" + ("" if has_code(nxt[0]) else " (needs code: ask Claude)"))
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
    caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])   # keep the Mac awake while a step runs
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
                    notify.send(f"🧪 Queue {i}/{len(items)}: {_short(it)} needs code · ask Claude", "research")
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
                    notify.send(f"❌ Queue: {_short(it)} A/B crashed · log logs/research_queue/{ts}_{it['id']}_test.log", "fail")
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
        caff.terminate()
        LOCK.rmdir()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "run":
        sys.exit(run())
    print("\n".join(status_lines()))
