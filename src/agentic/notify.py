"""Phone messages for SGM / Sri Lakshmi (2026-09-29) — filled in from data files, never written by a language model.

Plain English: after the daily run this sends one short message (did the data come in, how are the paper batches doing,
any big order in a stock we hold, which industries just heated up). On Saturdays it sends the order sheet. When a step
fails it sends a failure message. Every number in a message is read from a file the pipeline wrote.

Where messages go:
  1. always: logs/outbox/<timestamp>_<kind>.txt (the record of what was sent)
  2. always: a macOS notification banner on this Mac
  3. Telegram, once ~/.config/sgm/telegram.env exists with TELEGRAM_BOT_TOKEN=... and TELEGRAM_CHAT_ID=...
     (the person creates the bot with @BotFather and writes this file; the token never goes into chat or git)
Sleeve size for rupee amounts: ~/.config/sgm/sleeve.json {"sleeve_inr": 3000000} (optional).
Usage: notify.py daily | weekly [--track sri_lakshmi|model|sleeve] | fail --step NAME --detail TEXT | test
Dates are written month-first ("Sep 29") so the output gate (trust/check_message.py) can tell dates from numbers.
Scheduled runs send through run_sgm.py, which checks every daily/weekly message before it goes out.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
OUTBOX = ROOT / "logs/outbox"
CFG = Path.home() / ".config/sgm"
TRACKS = {"sri_lakshmi": ROOT / "logs/sri_lakshmi", "model": ROOT / "logs/model_screen", "sleeve": ROOT / "logs/leader_sleeve"}
LABEL = {"sri_lakshmi": "Sri Lakshmi", "model": "Model screen", "sleeve": "Production"}


def _latest(pattern: str, folder: Path) -> Path | None:
    fs = sorted(folder.glob(pattern))
    return fs[-1] if fs else None


def _hm(secs: float) -> str:
    return f"{int(secs // 3600)}:{int(secs % 3600 // 60):02d}"


def run_line(run: dict) -> str:
    """When this run started vs when it was due, why it was late, and how long it took. Times only (h:mm), so the
    output gate reads them as times, not as numbers that need a source."""
    st = datetime.fromisoformat(run["started"])
    if run.get("trigger") != "schedule":
        head = f"Ran {st:%-I:%M %p} (started by hand)"
    else:
        due = st.replace(hour=18, minute=45, second=0, microsecond=0)
        if st < due:
            due -= timedelta(days=1)
        while due.weekday() >= 5:
            due -= timedelta(days=1)
        rec = ROOT / f"logs/runs/{due:%Y%m%d}_daily.json"
        earlier = json.loads(rec.read_text()) if rec.exists() else []
        blocked = [r for r in earlier if r.get("outcome") not in (None, "sent", "smoke") and r["started"] < run["started"]]
        if st - due <= timedelta(minutes=10):
            head = f"Ran {st:%-I:%M %p} (on time)"
        elif blocked:
            head = f"Ran {st:%-I:%M %p} (retry: the earlier run was stopped by a check)"
        else:
            head = f"Ran {st:%-I:%M %p}, due {due:%a %-I:%M %p} (Mac was asleep or off)"
    parts = [head]
    fetch = next((x["secs"] for x in run.get("stages", []) if x["stage"] == "FETCH"), None)
    if fetch is not None:
        parts.append(f"fetch {_hm(fetch)}")
    parts.append(f"total {_hm((datetime.now() - st).total_seconds())} (h:mm)")
    return " · ".join(parts)


def missed_line(run: dict) -> str:
    """Weekdays since the last sent daily message with no message at all (the Mac was asleep, off or away)."""
    day = lambda t: (datetime.fromisoformat(t) - timedelta(hours=11)).date()   # NSE data day, 11 AM ET to 11 AM ET  # noqa: E731
    sent = {day(r["started"]) for f in (ROOT / "logs/runs").glob("*_daily.json") for r in json.loads(f.read_text())
            if r.get("outcome") == "sent"}
    today = day(run["started"])
    prev = max((d for d in sent if d < today), default=None)
    if prev is None:
        return ""
    gap = pd.bdate_range(prev + timedelta(days=1), today - timedelta(days=1))
    return ("⚠ No message on " + ", ".join(f"{d:%a %b %d}" for d in gap) + " (Mac asleep, off or away)") if len(gap) else ""


def daily_text(run: dict | None = None) -> str:
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    ok, bad = st.get("passed", []), st.get("failed", [])
    lines = [f"SGM daily · data for {st.get('date')}"] + ([x for x in (run_line(run), missed_line(run)) if x] if run else []) + [
             f"Data {'✅' if not bad else '⚠️'} {len(ok)}/{len(ok) + len(bad)} feeds" + (f" · failed: {', '.join(bad)}" if bad else "")]
    ev = _latest("eval_run_*.json", ROOT / "logs/evals")
    if ev:
        e = json.loads(ev.read_text())
        fails = [r["id"] for r in e["results"] if r["status"] == "FAIL"]
        pend = [r["id"] for r in e["results"] if r["status"] == "PENDING"]
        warns = [r["id"] for r in e["results"] if r["status"] == "WARN"]
        lines.append(f"Checks: {e['verdict'].split(' —')[0]}" + (f" · failing: {', '.join(fails)}" if fails else "")
                     + (f" · warnings: {', '.join(warns)}" if warns else "")
                     + (f" · waiting on you: {len(pend)}" if pend else ""))
    book = []
    for tag, folder in TRACKS.items():
        of = folder / "outcomes.jsonl"
        if not of.exists():
            continue
        last = {}
        for l in of.read_text().splitlines():
            if l.strip():
                x = json.loads(l); last[x["screen_id"]] = x
        for sid, x in sorted(last.items()):
            if x.get("status") != "CLOSED":
                book.append(f"{LABEL[tag]} {pd.Timestamp(sid):%b %d} {x['ew_net']:+.1f}% (day {x['days_held']})")
    if book:
        lines.append("Paper: " + " · ".join(book))
    od = ROOT / "data/derived/order_daily.parquet"
    if od.exists():
        o = pd.read_parquet(od, columns=["symbol", "ts", "amount_cr", "pct_of_rev"])
        o = o[pd.to_datetime(o["ts"]) >= pd.Timestamp(st.get("date")) - pd.Timedelta(days=1)]
        held = set()
        for folder in TRACKS.values():
            for f in folder.glob("screen_*.json"):
                held |= {n["symbol"] for n in json.loads(f.read_text())["names"]}
        big = o[(o["pct_of_rev"] >= 0.15)].drop_duplicates(["symbol", "amount_cr"])
        mine = big[big["symbol"].isin(held)]
        if len(mine):
            lines.append("Held-name orders: " + " · ".join(f"{r.symbol} ₹{r.amount_cr:,.0f} cr ({r.pct_of_rev * 100:.0f}% of revenue)" for r in mine.itertuples()))
        elif len(big):
            lines.append(f"Big orders today (not held): {len(big)} — e.g. " + ", ".join(big["symbol"].head(3)))
    rep = _latest("daily_orders_industry_*.md", ROOT / "reports")
    if rep:
        m = re.search(r"Entered hot/warming[^:]*: (.*)", rep.read_text())
        if m and m.group(1).strip() != "none":
            lines.append("Industries warming up: " + m.group(1).strip())
    return "\n".join(lines)


def weekly_text(track: str) -> str:
    f = _latest("screen_*.json", TRACKS[track])
    if f is None:
        return f"{LABEL[track]}: no screen saved yet"
    sc = json.loads(f.read_text())
    sleeve = None
    if (CFG / "sleeve.json").exists():
        sleeve = json.loads((CFG / "sleeve.json").read_text()).get("sleeve_inr")
    per = sleeve / 26 / len(sc["names"]) if sleeve else None
    entry = pd.bdate_range(pd.Timestamp(sc["data_through"]) + pd.Timedelta(days=1), periods=1)[0]
    sell = pd.bdate_range(entry, periods=126)[-1]
    lines = [f"{LABEL[track]} batch · data {sc['data_through']} · buy at the {entry:%a %b %d} open",
             f"{len(sc['names'])} buys, {'₹{:,.0f} each (1/9 of a 1/26 batch)'.format(per) if per else 'equal weight (set ~/.config/sgm/sleeve.json for ₹ amounts)'}"]
    for n in sc["names"]:
        c = n.get("close")
        lim = f"limit ₹{c * 1.05:,.2f}" if c else "limit: last close +5%"
        flag = " ⚑ takeover" if n.get("takeover") else ""
        lines.append(f"{n['rank']}. {n['symbol']} — {lim}{flag}")
    lines.append(f"Sell: close of session 126, about {sell:%b %d %Y} (+ NSE holidays). No stop-loss.")
    lines.append("Real money only after you sign evals/human_review for this batch.")
    return "\n".join(lines)


def send(text: str, kind: str) -> None:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}.txt").write_text(text + "\n")
    title = {"daily": "SGM daily", "weekly": "SGM batch", "fail": "SGM ❌"}.get(kind, "SGM")
    first = text.splitlines()[1] if len(text.splitlines()) > 1 else text
    subprocess.run(["osascript", "-e", f'display notification {json.dumps(first[:200])} with title {json.dumps(title)}'], capture_output=True)
    env = CFG / "telegram.env"
    if env.exists():
        kv = dict(l.split("=", 1) for l in env.read_text().splitlines() if "=" in l and not l.startswith("#"))
        tok, chat = kv.get("TELEGRAM_BOT_TOKEN", "").strip(), kv.get("TELEGRAM_CHAT_ID", "").strip()
        if tok and chat:
            r = subprocess.run(["curl", "-s", "-m", "30", f"https://api.telegram.org/bot{tok}/sendMessage",
                                "--data-urlencode", f"chat_id={chat}", "--data-urlencode", f"text={text}"], capture_output=True, text=True)
            if '"ok":true' not in (r.stdout or ""):
                (OUTBOX / f"{datetime.now():%Y%m%d_%H%M%S}_telegram_error.txt").write_text((r.stdout or r.stderr)[:500])
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["daily", "weekly", "fail", "test"])
    ap.add_argument("--track", default="sri_lakshmi", choices=list(TRACKS))
    ap.add_argument("--step", default="")
    ap.add_argument("--detail", default="")
    a = ap.parse_args()
    if a.kind == "daily":
        send(daily_text(), "daily")
    elif a.kind == "weekly":
        send(weekly_text(a.track), "weekly")
    elif a.kind == "fail":
        send(f"❌ SGM · {a.step} failed\n{a.detail}\nPicks on hold until it passes. Logs: logs/daily_data_layer/", "fail")
    else:
        send("SGM test message · if you can read this on your phone, alerts work.", "test")


if __name__ == "__main__":
    main()
