"""SGM watchdog (2026-10-06, INC-2026-10-06-daily-run-hung-silent): an independent check every 30 minutes (launchd
com.sgm.watchdog), so a hung or missed run cannot go silent. The daily run's own checks cannot fire while it is stuck.

1. Stuck run: a run holding logs/runs/.lock for more than 4 hours (wall clock) is killed with all its child processes, the
   lock is cleared, a failure note is sent, and a fresh catch-up run is started (daily on a weekday, weekend note on Sat/Sun).
2. Missed daily: on an NSE session day after 10:30 PM ET, no daily message sent since 11 AM ET -> alert (once a day).
3. Missed weekend note: Saturday / Sunday after 9 PM ET, none sent today -> alert (once a day).
Heartbeat: logs/runs/watchdog_heartbeat.json (eval ops.watchdog_alive warns if it is older than 2 hours).
Usage: watchdog.py [--dry-run]
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
RUNS = ROOT / "logs/runs"
LOCK = RUNS / ".lock"
STATE = RUNS / "watchdog_state.json"
STUCK_H = 4.0


def descendants(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout.split("\n")
    kids: dict[int, list[int]] = {}
    for line in out:
        p = line.split()
        if len(p) == 2:
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    res, todo = [], [pid]
    while todo:
        x = todo.pop()
        for k in kids.get(x, []):
            res.append(k); todo.append(k)
    return res


def main() -> None:
    dry = "--dry-run" in sys.argv
    import notify
    import nse_calendar
    import run_sgm
    now = datetime.now()
    st = json.loads(STATE.read_text()) if STATE.exists() else {"alerted": {}}
    notes = []

    def alert(key: str, text: str) -> None:
        if key in st["alerted"]:
            return
        notes.append(text)
        if not dry:
            notify.send(text, "fail")
        st["alerted"][key] = now.isoformat(timespec="seconds")

    # 1. stuck run
    pidf = LOCK / "pid"
    if pidf.exists():
        age_h = (time.time() - pidf.stat().st_mtime) / 3600
        try:
            pid = int(pidf.read_text().strip())
            os.kill(pid, 0); alive = True
        except (ValueError, ProcessLookupError, PermissionError):
            alive = False
        if alive and age_h > STUCK_H:
            stage = "?"
            logs = sorted(RUNS.glob("*_*.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                stage = logs[-1].stem.split("_", 2)[-1]
            msg = (f"❌ SGM run stuck {age_h:.0f}h (last stage: {stage}) — killed by the watchdog at {now:%-I:%M %p} ET; "
                   f"a fresh run starts now. Log: {logs[-1].name if logs else '-'}")
            if not dry:
                for p in descendants(pid)[::-1] + [pid]:
                    try:
                        os.kill(p, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                time.sleep(10)
                for p in descendants(pid) + [pid]:
                    try:
                        os.kill(p, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                pidf.unlink(missing_ok=True)
                try:
                    LOCK.rmdir()
                except OSError:
                    pass
            alert(f"stuck:{pidf.stat().st_mtime if pidf.exists() else now.date()}", msg)
            mode = "weekend" if now.weekday() >= 5 else "daily"
            if not dry:
                subprocess.Popen(["/usr/bin/python3", str(ROOT / "src/agentic/run_sgm.py"), mode, "--if-missed"], cwd=ROOT,
                                 stdout=open(RUNS / f"launchd_{mode}.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    # 2. / 3. missed messages
    today = now.date()
    if now.weekday() < 5 and nse_calendar.is_session(today) and (now.hour, now.minute) >= (22, 30):
        if not run_sgm._sent_since("daily", now.replace(hour=11, minute=0, second=0, microsecond=0)):
            alert(f"daily:{today}", f"⚠️ SGM: no daily message today ({today:%a %b %d}) by {now:%-I:%M %p} ET — the run failed, is "
                  f"still running, or the Mac was asleep/offline. Checking logs/runs/{today:%Y%m%d}_daily.json.")
    if now.weekday() >= 5 and now.hour >= 21:
        if not run_sgm._sent_since("weekend", now.replace(hour=0, minute=0, second=0, microsecond=0)):
            alert(f"weekend:{today}", f"⚠️ SGM: no weekend note today ({today:%a %b %d}) by {now:%-I:%M %p} ET.")
    st["alerted"] = {k: v for k, v in st["alerted"].items() if v >= (now.replace(day=1).isoformat())} | {k: v for k, v in st["alerted"].items() if k.endswith(str(today))}
    if not dry:
        STATE.write_text(json.dumps(st, indent=1))
        (RUNS / "watchdog_heartbeat.json").write_text(json.dumps(dict(ts=now.isoformat(timespec="seconds"), notes=notes)))
    print(f"[{now:%H:%M}] watchdog {'(dry run) ' if dry else ''}ok · " + (" | ".join(notes) if notes else "nothing to report"))


if __name__ == "__main__":
    main()
