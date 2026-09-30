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


FEED_GROUPS = [   # daily_data_layer.sh step labels -> how the phone message groups them (unknown labels land in "Other")
    ("Prices & filings", ["corp_actions", "prices", "announcements", "pnl_calendar", "pnl_history"]),
    ("Macro & flows", ["forex", "usdinr_history", "commodity", "global_macro", "fii_dii", "india_vix", "breadth", "gold_inr"]),
    ("News & policy", ["news_events", "news_rss", "sentiment", "global_sentiment", "pib_releases"]),
    ("Deals & industry", ["block_deals", "industry", "security_master", "screener_industry"]),
    ("Panel rebuilt", ["macro_panel"]),
]
QC_LABEL = {"data.coverage": "Every stock has a price for the day", "data.freshness": "All inputs fresh",
            "data.no_duplicates": "No duplicate rows", "data.no_holiday_copies": "No copied / stale price days",
            "data.ca_before_prices": "Corporate actions loaded before prices", "data.manifests": "Every data file documented",
            "pit.filings_after_period": "No future-dated filings", "prov.external_rows_sourced": "Government data rows have sources",
            "prov.order_amount_grounded": "Order amounts match the filing text", "ops.daily_run_happened": "Run happened for today"}
ICON = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "PENDING": "⏳", "SKIP": "·"}


def _last_sent_ist() -> datetime | None:
    """When the previous daily message finished, in IST (the order file's clock)."""
    from zoneinfo import ZoneInfo
    ts = [r.get("finished") or r["started"] for f in (ROOT / "logs/runs").glob("*_daily.json")   # finished: its own fetches came before
          for r in json.loads(f.read_text()) if r.get("outcome") == "sent"]
    if not ts:
        return None
    t = datetime.fromisoformat(max(ts)).replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)


def daily_text(run: dict | None = None) -> str:
    """The daily checklist: when it ran, every feed, every data check, new orders, paper batches, industries."""
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    ok, bad = st.get("passed", []), st.get("failed", [])
    lines = [f"SGM daily · data for {st.get('date')}"] + ([x for x in (run_line(run), missed_line(run)) if x] if run else [])

    lines += ["", f"📥 DATA {'✅' if not bad else '⚠️'} {len(ok)}/{len(ok) + len(bad)} feeds"]
    seen = set()
    for name, labels in FEED_GROUPS + [("Other", [x for x in ok + bad if x not in {l for _, g in FEED_GROUPS for l in g}])]:
        got = [x for x in labels if x in ok or x in bad]
        seen |= set(got)
        if got:
            failed = [x for x in got if x in bad]
            lines.append(f"{'❌' if failed else '✅'} {name}: " + ", ".join(("❌ " if x in bad else "") + x for x in got))

    gate = _latest("gate_data_*.json", ROOT / "logs/evals")
    if gate:
        g = json.loads(gate.read_text())["results"]
        npass = sum(r["status"] == "PASS" for r in g)
        lines += ["", f"🧪 QC {'✅' if npass == len(g) else '⚠️'} {npass}/{len(g)} data checks pass"]
        for r in g:
            val = str(r["value"]) if r["value"] not in (None, "") else ""
            det = r["detail"] if r["status"] != "PASS" or val in ("", "0") else ""
            lines.append(f"{ICON.get(r['status'], '')} {QC_LABEL.get(r['id'], r['id'])}" + (f": {val}" if val else "") + (f" — {det[:120]}" if det else ""))
    ev = _latest("eval_run_*.json", ROOT / "logs/evals")
    if ev:
        e = json.loads(ev.read_text())
        fails = [r["id"] for r in e["results"] if r["status"] == "FAIL"]
        pend = [r["id"] for r in e["results"] if r["status"] == "PENDING"]
        warns = [r["id"] for r in e["results"] if r["status"] == "WARN"]
        lines.append(f"All checks: {e['verdict'].split(' —')[0]}" + (f" · failing: {', '.join(fails)}" if fails else "")
                     + (f" · warnings: {', '.join(warns)}" if warns else "")
                     + (f" · waiting on you: {len(pend)}" if pend else ""))

    oj = ROOT / "logs/daily_orders/latest.json"
    if oj.exists():
        O = json.loads(oj.read_text())
        cut = _last_sent_ist()
        new = [f for f in O["filings"] if cut is None or datetime.fromisoformat(f["fetched_ist"]) > cut]
        new.sort(key=lambda f: -(f["pct_of_rev"] or 0))
        big = [f for f in new if (f["pct_of_rev"] or 0) >= 0.15]
        lines += ["", f"📦 NEW ORDERS {len(new)} since the last message · {len(big)} big (15%+ of a year's revenue)"]
        for f in new[:8]:
            mark = "⭐" if f["held"] else ("🔥" if (f["pct_of_rev"] or 0) >= 0.15 else "•")
            amt = f"₹{f['amount_cr']:,.0f} cr" if f["amount_cr"] is not None else "amount not stated"
            pc = f" = {f['pct_of_rev'] * 100:.0f}% of revenue" if f["pct_of_rev"] is not None else ""
            hp = f["heat_pct"]
            heat = "" if hp is None else f" ({'hot' if hp >= 0.9 else 'warming' if hp >= 0.7 else 'not hot'}, {hp:.2f})"
            ind = f" · {f['industry']}{heat}" if f["industry"] else ""
            tr = " · trend ✅" if f["trend"] else ""
            lines.append(f"{mark} {f['symbol']} {amt}{pc}{ind}{tr}" + (f" · HELD in {', '.join(f['held'])}" if f["held"] else ""))
        if len(new) > 8:
            lines.append(f"… and {len(new) - 8} smaller (full list: reports/daily_orders_industry_*.md)")

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
                book.append(f"{LABEL[tag]} {pd.Timestamp(sid):%b %d}: {x['ew_net']:+.1f}% (day {x['days_held']})")
    if book:
        lines += ["", "📈 PAPER BATCHES (after costs)"] + book

    if oj.exists() and (O.get("warming") or O.get("cooling")):
        lines += ["", "🌡 INDUSTRIES"]
        if O.get("warming"):
            lines.append("Warming up: " + ", ".join(O["warming"]))
        if O.get("cooling"):
            lines.append("Cooling off: " + ", ".join(O["cooling"]))
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
