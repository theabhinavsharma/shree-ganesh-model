"""SGM / Sri Lakshmi orchestrator (2026-09-29): the one scheduled entry point. Each stage runs, then its checks run;
a blocking check failure stops the run and you get a failure message instead of picks.

Plain English: this is the conveyor belt. Fetch the data, check it, work it up, check that, write the message, check
the message, send it. If any check that is marked "block" in evals/registry.yaml fails, the belt stops there.

daily   launchd (com.sgm.daily): weekdays 6:45 PM ET, retries 9:45 PM and 7:15 AM; a time missed while the Mac slept runs
        when it wakes. All use --if-missed: skipped when a run already sent since 11 AM ET (one per NSE data day)
  FETCH            daily_data_layer.sh: every feed, corporate actions before prices, panel rebuilt, status file
  GATE data        data.*, pit.filings_after_period, prov.*, ops.daily_run_happened
  ANALYSE          sgm_daily.sh: score every paper batch, basket / leader reports, freshness dashboard
  GATE analysis    output.* (screens unchanged, one per week, scored)
  REPORT           the full daily eval report (reports/eval_report_<date>.md), read by the message
  MESSAGE          notify.daily_text() -> GATE output (trust/check_message.py) -> send, or hold it and send a failure note
  MIRROR           Google Drive day_to_day copy (never blocks)
weekly  launchd (com.sgm.weekly): Friday 7:30 PM ET, Saturday 9 AM ET retry, or on wake if both were slept through
        (--if-missed: skipped when this ISO week already sent)
  FETCH + GATE data   fetch only if today's daily run has not fetched; the data gate always runs
  POLICY DATA      fetch_iip_core.py --refresh: new IIP / core-sector releases (failure reported, does not block)
  MODEL SCORES     build_mcap_pit -> anatomy_1p5x -> model_bakeoff_1p5x: model scores rebuilt from this week's prices (~15 min;
                   failure is reported; the screen itself refuses scores more than 7 days old)
  THEMES           fetch_announcements_historical (resume) -> fetch_fred_drivers -> build_themes (V3's fading-theme filter)
  SRI LAKSHMI      build_industry_scores -> build_policy_scores -> screen_sri_lakshmi -> score_sri_lakshmi (blocks on failure)
  REPRODUCE        trust/reproduce.py: registered results re-run from the committed code
  GATE analysis    output.*, pit.model_scores_current, model.*, strategy.*, trust.results_reproduce + this week's batch exists
  MESSAGE          the Sri Lakshmi G1 order sheet, through the output gate (only G1 and the Sep 8 production sleeve are tracked, 2026-09-30)
  15D PIPELINE     run_weekly_pipeline.sh --skip-fetch (the separate 15D/5% basket thread, last so it cannot hold Sri Lakshmi up)
weekend launchd (com.sgm.weekend): Saturday and Sunday 11:30 AM ET, retry 7:30 PM (Abhinav 2026-10-04: "let's not skip
        telegram on weekends"); --if-missed: skipped when a weekend note was already sent today
  ORDERS           refresh_announcements -> daily_orders_industry (companies file on weekends; no prices: NSE is closed)
  MESSAGE          notify.weekend_text() -> GATE output (the daily number check) -> send
One run at a time (logs/runs/.lock; a second run waits up to 3 hours). Keeps the Mac awake while running (caffeinate).
Record: logs/runs/<date>_<mode>.json (every stage: status, seconds, log). Children get SGM_ORCHESTRATED=1 so the
shell scripts skip the steps this file runs itself.
Usage: /usr/bin/python3 src/agentic/run_sgm.py daily|weekly|weekend [--if-missed] [--smoke]
  --smoke   nothing fetched, computed or sent: gates are reported (not enforced), messages are built, checked and
            written to logs/outbox/smoke/. Used to prove the schedule can run.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/usr/bin/python3"
RUNS = ROOT / "logs/runs"
LOCK = RUNS / ".lock"
sys.path.insert(0, str(ROOT / "src/agentic"))
sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import notify  # noqa: E402
import check_message  # noqa: E402

ENV = dict(os.environ, SGM_ORCHESTRATED="1", PYTHONIOENCODING="utf-8", LANG="en_US.UTF-8",
           PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")
GATES = {
    "data": ["data.", "pit.filings_after_period", "prov.", "ops.daily_run_happened"],
    "analysis_daily": ["output.screens_immutable", "output.one_screen_per_week", "output.paper_scored"],
    "analysis_weekly": ["output.screens_immutable", "output.one_screen_per_week", "output.paper_scored",
                        "pit.model_scores_current", "model.", "strategy.", "trust.results_reproduce"],
}


class Run:
    def __init__(self, mode: str, smoke: bool):
        self.mode, self.smoke = mode, smoke
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.rec = dict(mode=mode, smoke=smoke, started=datetime.now().isoformat(timespec="seconds"), stages=[], outcome=None)

    def log(self, msg: str) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    def stage(self, name: str, cmd: list[str], timeout: int) -> bool:
        lf = RUNS / f"{self.ts}_{name.lower().replace(' ', '_')}.log"
        t0 = time.time()
        # 2026-10-06 (INC-2026-10-06-daily-run-hung-silent): the limit is WALL-CLOCK time (time.time() keeps counting while
        # the Mac sleeps; subprocess's own timeout did not, and only killed the shell, not its children), checked every
        # 30 s, and it kills the step's whole process group.
        with lf.open("w") as fh:
            p = subprocess.Popen(cmd, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
            deadline, rc = time.time() + timeout, None
            while rc is None:
                try:
                    rc = p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if time.time() > deadline:
                        for sig in (signal.SIGTERM, signal.SIGKILL):
                            try:
                                os.killpg(p.pid, sig)
                            except ProcessLookupError:
                                break
                            time.sleep(10)
                        rc = "timeout"
        status = "ok" if rc == 0 else (f"timeout after {timeout // 60} min (wall clock, step killed)" if rc == "timeout" else f"exit {rc}")
        self.rec["stages"].append(dict(stage=name, status=status, secs=round(time.time() - t0), log=str(lf.relative_to(ROOT))))
        self.log(f"{'✅' if status == 'ok' else '❌'} {name}: {status} ({time.time() - t0:.0f}s)")
        return status == "ok"

    def gate(self, name: str, cadence: str = "all") -> list[dict]:
        """Run the registry evals for this gate; return the blocking failures (none in smoke mode is enforced)."""
        t0 = time.time()
        subprocess.run([PY, "src/agentic/run_evals.py", "--cadence", cadence, "--only", ",".join(GATES[name]), "--gate", name],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=3600)
        f = ROOT / f"logs/evals/gate_{name}_{datetime.now():%Y-%m-%d}.json"
        rows = json.loads(f.read_text())["results"] if f.exists() and f.stat().st_mtime >= t0 else []
        block = [r for r in rows if r["action"] == "block" and r["status"] == "FAIL"]
        if not rows:
            block = [dict(id=f"gate.{name}", statement="the gate ran", detail="no gate result written")]
        self.rec["stages"].append(dict(stage=f"GATE {name}", status="BLOCKED" if block else "pass", secs=round(time.time() - t0),
                                       failed=[r["id"] for r in block], evals=len(rows)))
        self.log(f"{'🛑' if block else '✅'} GATE {name}: {len(rows)} evals, {len(block)} blocking failures"
                 + (f" ({', '.join(r['id'] for r in block)})" if block else "") + (" [smoke: not enforced]" if self.smoke and block else ""))
        return [] if self.smoke else block

    def send(self, text: str, kind: str) -> None:
        if self.smoke:
            d = ROOT / "logs/outbox/smoke"; d.mkdir(parents=True, exist_ok=True)
            (d / f"{self.ts}_{kind}_{len(list(d.glob(self.ts + '*')))}.txt").write_text(text + "\n")
        else:
            notify.send(text, kind)

    def message(self, kind: str, text: str, track: str | None = None, send_as: str | None = None) -> bool:
        res = check_message.check(kind, text, track, log=not self.smoke)
        label = f"GATE output ({kind}{' ' + track if track else ''})"
        self.rec["stages"].append(dict(stage=label, status="pass" if res["ok"] else "HELD", **res))
        self.log(f"{'✅' if res['ok'] else '🛑'} {label}: " + ("sent" if res["ok"] else f"held — {res}"))
        if res["ok"]:
            self.send(text, send_as or kind)
        else:
            self.send(f"❌ SGM · {kind} message held (it did not pass its number check)\n"
                      + (f"No source for: {', '.join(res['unsourced'][:8])}\n" if res["unsourced"] else "")
                      + ("; ".join(res["problems"])[:500] + "\n" if res["problems"] else "")
                      + f"Draft kept in logs/runs/{self.ts}_held_{kind}.txt", "fail")
            (RUNS / f"{self.ts}_held_{kind}.txt").write_text(text + "\n")
        return res["ok"]

    def stop(self, where: str, block: list[dict]) -> int:
        lines = [f"❌ SGM {self.mode} run stopped at {where} ({datetime.now():%-I:%M %p} ET)"]
        lines += [f"• {notify.QC_LABEL.get(r['id'], r['id'])}: {r.get('detail') or r.get('statement', '')}"[:200] for r in block[:6]]
        lines += ["", f"Picks on hold until this passes · record logs/runs/{self.ts[:8]}_{self.mode}.json"]
        self.send("\n".join(lines), "fail")
        return self.finish("blocked at " + where, 1)

    def finish(self, outcome: str, code: int) -> int:
        self.rec.update(outcome="smoke" if self.smoke else outcome, finished=datetime.now().isoformat(timespec="seconds"))
        name = RUNS / f"{datetime.now():%Y%m%d}_{self.mode}{'_smoke' if self.smoke else ''}.json"
        runs = json.loads(name.read_text()) if name.exists() else []
        name.write_text(json.dumps(runs + [self.rec], indent=1, default=str))
        self.log(f"═══ {self.mode}: {self.rec['outcome']} ═══")
        return code


def _sent_since(mode: str, since: datetime) -> bool:
    for f in sorted(RUNS.glob(f"*_{mode}.json"))[-10:]:
        for r in json.loads(f.read_text()):
            if r.get("outcome") == "sent" and datetime.fromisoformat(r["started"]) >= since:
                return True
    return False


def _fetched_today() -> bool:
    f = RUNS / f"{datetime.now():%Y%m%d}_daily.json"
    return f.exists() and any(s["stage"] == "FETCH" for r in json.loads(f.read_text()) for s in r["stages"])


def _lock(wait_s: int = 3 * 3600) -> bool:
    t0 = time.time()
    while True:
        try:
            LOCK.mkdir()
            (LOCK / "pid").write_text(str(os.getpid()))
            return True
        except FileExistsError:
            try:
                os.kill(int((LOCK / "pid").read_text()), 0)          # holder still alive -> wait
            except (ProcessLookupError, ValueError, FileNotFoundError):
                (LOCK / "pid").unlink(missing_ok=True); LOCK.rmdir()  # stale lock from a crashed run
                continue
            if time.time() - t0 > wait_s:
                return False
            time.sleep(60)


def daily(R: Run) -> int:
    if not R.smoke:
        R.stage("FETCH", ["/bin/bash", "src/agentic/daily_data_layer.sh"], 3 * 3600)   # the data gate decides, not the exit code
    if b := R.gate("data"):
        return R.stop("the data check", b)
    if not R.smoke:
        R.stage("ANALYSE", ["/bin/bash", "src/agentic/sgm_daily.sh"], 2 * 3600)
    if b := R.gate("analysis_daily"):
        return R.stop("the analysis check", b)
    R.stage("REPORT", [PY, "src/agentic/run_evals.py", "--cadence", "daily"], 3600)
    ok = R.message("daily", notify.daily_text(run=R.rec))
    if not R.smoke:
        R.stage("MIRROR", ["/bin/bash", "src/agentic/sync_drive_mirror.sh"], 3600)
    return R.finish("sent" if ok else "message held", 0 if ok else 1)


def weekend(R: Run) -> int:
    if not R.smoke:
        R.stage("ORDERS", ["/bin/bash", "-c", "set -e; /usr/bin/python3 src/agentic/refresh_announcements.py; "
                           "/usr/bin/python3 src/agentic/daily_orders_industry.py"], 3600)   # a failure shows as a stale digest
    ok = R.message("daily", notify.weekend_text(run=R.rec), send_as="weekend")
    return R.finish("sent" if ok else "message held", 0 if ok else 1)


def weekly(R: Run) -> int:
    if not R.smoke and not _fetched_today():
        R.stage("FETCH", ["/bin/bash", "src/agentic/daily_data_layer.sh"], 3 * 3600)
    if b := R.gate("data"):
        return R.stop("the data check", b)
    if not R.smoke:
        # new monthly IIP / core-sector releases (cached; --refresh re-reads the release listings). A failure leaves the
        # previous releases in place (point-in-time, just older), so it is reported and does not block.
        # TODO: a new Union Budget (each February) needs its budget_id added to fetch_budget_capex.py by hand.
        if not R.stage("POLICY DATA", ["nice", "-n", "10", PY, "src/agentic/fetch_iip_core.py", "--refresh"], 2 * 3600):
            R.send(f"⚠ SGM · IIP/core-sector refresh failed; Sri Lakshmi uses the previous releases. Log: {R.rec['stages'][-1]['log']}", "fail")
        # Model scores (2026-09-30): rebuilt every week from the current price panel (walk-forward, same recipe). Before this
        # the file was built by hand and ended 2026-09-21. If the rebuild fails, the screen itself refuses stale scores.
        if not R.stage("MODEL SCORES", ["/bin/bash", "-c", "set -e; nice -n 10 /usr/bin/python3 src/agentic/build_mcap_pit.py; "
                       "nice -n 10 /usr/bin/python3 src/agentic/anatomy_1p5x.py; "
                       "nice -n 10 /usr/bin/python3 src/agentic/model_bakeoff_1p5x.py"], 4 * 3600):
            R.send(f"⚠ SGM · model score rebuild failed; the screen runs only if last week's scores are within 7 days. Log: {R.rec['stages'][-1]['log']}", "fail")
        # Themes (2026-10-02, Sri Lakshmi V3): new company filings, IMF prices, then rising/fading themes. If this fails, the
        # screen refuses theme data older than this week (no batch) rather than silently falling back to G1.
        if not R.stage("THEMES", ["/bin/bash", "-c", "set -e; nice -n 10 /usr/bin/python3 src/agentic/fetch_announcements_historical.py; "
                       "nice -n 10 /usr/bin/python3 src/agentic/fetch_fred_drivers.py; nice -n 10 /usr/bin/python3 src/agentic/build_themes.py"], 2 * 3600):
            R.send(f"⚠ SGM · theme refresh failed; the V3 screen will refuse stale themes. Log: {R.rec['stages'][-1]['log']}", "fail")
        if not R.stage("SRI LAKSHMI", ["/bin/bash", "-c", "set -e; for s in build_industry_scores build_policy_scores "
                       "screen_sri_lakshmi score_sri_lakshmi; do nice -n 10 /usr/bin/python3 src/agentic/$s.py; done"], 2 * 3600):
            return R.stop("the Sri Lakshmi screen", [dict(id="stage.sri_lakshmi", detail=f"see {R.rec['stages'][-1]['log']}")])
        R.stage("REPRODUCE", ["nice", "-n", "10", PY, "src/agentic/trust/reproduce.py"], 2 * 3600)
    block = R.gate("analysis_weekly", "weekly")
    f = notify._latest("screen_*.json", notify.TRACKS["sri_lakshmi"])
    sc = json.loads(f.read_text()) if f else None
    last = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())["date"]
    this_week = sc and datetime.fromisoformat(sc["data_through"]).isocalendar()[:2] == datetime.fromisoformat(last).isocalendar()[:2]
    if not this_week and not R.smoke:
        block.append(dict(id="stage.batch_exists", detail=f"no Sri Lakshmi batch for the week of {last}"))
    if block:
        return R.stop("the weekly analysis check", block)
    ok = R.message("weekly", notify.weekly_text("sri_lakshmi"), "sri_lakshmi")
    if not R.smoke and not R.stage("15D PIPELINE", ["/bin/bash", "src/agentic/run_weekly_pipeline.sh", "--skip-fetch"], 6 * 3600):
        R.send(f"⚠ SGM · 15D weekly pipeline failed (Sri Lakshmi already sent). Log: {R.rec['stages'][-1]['log']}", "fail")
    return R.finish("sent" if ok else "message held", 0 if ok else 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["daily", "weekly", "weekend"])
    ap.add_argument("--if-missed", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)

    def done() -> bool:                       # catch-up runs: nothing to do if this mode already sent
        if not a.if_missed or a.smoke:
            return False
        now = datetime.now()
        if a.mode == "daily":         # NSE's end-of-day files for an IST session are out by ~11 AM ET: one sent run per 11-to-11 window
            since = now.replace(hour=11, minute=0, second=0, microsecond=0)
            if now < since:
                since -= timedelta(days=1)
        elif a.mode == "weekend":     # one weekend note per calendar day
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            since = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)   # this ISO week
        if _sent_since(a.mode, since):
            print(f"[{now:%H:%M:%S}] {a.mode}: already sent since {since:%a %H:%M} — nothing to do", flush=True)
            return True
        return False

    if done():
        return 0
    R = Run(a.mode, a.smoke)
    R.rec["trigger"] = "schedule" if a.if_missed else "manual"
    if not _lock():
        if done():                            # the run we waited for sent it; a catch-up stays quiet
            return 0
        R.send(f"❌ SGM {a.mode} run gave up: another run held the lock for 3 hours (logs/runs/.lock)", "fail")
        return R.finish("lock timeout", 1)
    caff = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    try:
        if done():                            # re-check after waiting on the lock: never run the same day twice
            return 0
        R.log(f"═══ SGM {a.mode}{' (smoke)' if a.smoke else ''} · {ROOT} ═══")
        return daily(R) if a.mode == "daily" else (weekend(R) if a.mode == "weekend" else weekly(R))
    except Exception as x:                                                # the belt itself broke: say so, loudly
        R.send(f"❌ SGM {a.mode} orchestrator crashed: {type(x).__name__}: {x}"[:600], "fail")
        R.rec["error"] = repr(x)
        return R.finish("crashed", 1)
    finally:
        caff.terminate()
        (LOCK / "pid").unlink(missing_ok=True)
        if LOCK.exists():
            LOCK.rmdir()


if __name__ == "__main__":
    sys.exit(main())
