"""Phone messages for SGM / Sri Lakshmi (2026-09-29) — filled in from data files, never written by a language model.

Plain English: after the daily run this sends one short message (did the data come in, how are the paper batches doing,
any big order in a stock we hold, which industries just heated up). On Fridays it sends the order sheet; on Saturdays and
Sundays a weekend note (no new prices: filings, open batches, input health, the next buy). When a step
fails it sends a failure message. Every number in a message is read from a file the pipeline wrote.

Where messages go:
  1. always: logs/outbox/<timestamp>_<kind>.txt (the record of what was sent)
  2. always: a macOS notification banner on this Mac
  3. Telegram, once ~/.config/sgm/telegram.env exists with TELEGRAM_BOT_TOKEN=... and TELEGRAM_CHAT_ID=...
     (the person creates the bot with @BotFather and writes this file; the token never goes into chat or git)
Sleeve size for rupee amounts: ~/.config/sgm/sleeve.json {"sleeve_inr": 3000000} (optional).
Usage: notify.py daily | weekend | weekly [--track sri_lakshmi|model|sleeve] | fail --step NAME --detail TEXT | test
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

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse_calendar  # noqa: E402

ROOT = Path("/Users/abhinavs./Code/Zoom")
OUTBOX = ROOT / "logs/outbox"
CFG = Path.home() / ".config/sgm"
# Only two tracks (Abhinav, 2026-09-30): Sri Lakshmi (G1 batch Sep 28, V3 from Oct 2) and the Sep 8 production sleeve.
TRACKS = {"sri_lakshmi": ROOT / "logs/sri_lakshmi", "sleeve": ROOT / "logs/leader_sleeve"}
LABEL = {"sri_lakshmi": "Sri Lakshmi V3", "sleeve": "Production"}
SHORT = {"sri_lakshmi": "SL", "sleeve": "Prod"}


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
        head = f"ran {st:%-I:%M %p} by hand"
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
            head = f"ran {st:%-I:%M %p} ✅"
        elif blocked:
            head = f"ran {st:%-I:%M %p} (retry after a stop)"
        else:
            head = f"ran {st:%-I:%M %p}, late (Mac asleep at {due:%-I:%M %p})"
    return f"{head} · {_hm((datetime.now() - st).total_seconds())}"


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
    return ("⚠ no message on " + ", ".join(f"{d:%a %b %d}" for d in gap) + " (Mac asleep/off)") if len(gap) else ""


FEED_GROUPS = [   # daily_data_layer.sh step labels -> how the phone message groups them (unknown labels land in "Other")
    ("Prices & filings", ["corp_actions", "prices", "announcements", "pnl_calendar", "pnl_history", "announcements_archive",
                          "order_amounts", "event_ledger", "pnl_update"]),
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
NSE_FILINGS = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
LINK = re.compile(r"📄 (https?://\S+)")


def _telegram_html(text: str) -> str:
    """Plain message -> Telegram HTML: escape everything, bold the first line (the title), turn '📄 <url>' into a
    short tappable link."""
    import html
    first, _, rest = text.partition("\n")
    out = f"<b>{html.escape(first, quote=False)}</b>" + ("\n" + html.escape(rest, quote=False) if rest else "")
    return LINK.sub(lambda m: f'<a href="{html.escape(m.group(1))}">📄 {"all filings on NSE" if m.group(1) == NSE_FILINGS else "read"}</a>', out)


def _last_sent_ist() -> datetime | None:
    """When the previous daily or weekend message finished, in IST (the order file's clock)."""
    from zoneinfo import ZoneInfo
    ts = [r.get("finished") or r["started"] for m in ("daily", "weekend") for f in (ROOT / "logs/runs").glob(f"*_{m}.json")   # finished: its own
          for r in json.loads(f.read_text()) if r.get("outcome") == "sent"]                          # fetches came before (weekend: 2026-10-04)
    if not ts:
        return None
    t = datetime.fromisoformat(max(ts)).replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)


def _orders_lines(st: dict) -> list[str]:
    """New order filings since the last sent message (big ones first)."""
    lines: list[str] = []
    oj = ROOT / "logs/daily_orders/latest.json"
    if oj.exists():
        lines.append("")
        O = json.loads(oj.read_text())
        if not (O["data_through"] >= str(st.get("date")) or O["written"][:10] >= str(st.get("date"))):
            lines.append(f"❌ orders digest stale (written {O['written'][:10]})")
        cut = _last_sent_ist()
        new = [f for f in O["filings"] if cut is None or datetime.fromisoformat(f["fetched_ist"]) > cut]
        new.sort(key=lambda f: -(f["pct_of_rev"] or 0))
        big = [f for f in new if (f["pct_of_rev"] or 0) >= 0.15]
        lines.append(f"📦 Orders: {len(new)} new · {len(big)} big" if new else "📦 Orders: none new")
        for f in new[:5]:
            mark = "⭐" if f["held"] else ("🔥" if (f["pct_of_rev"] or 0) >= 0.15 else "•")
            amt = f"₹{f['amount_cr']:,.0f}cr" if f["amount_cr"] is not None else "₹?"
            pc = f" ({f['pct_of_rev'] * 100:.0f}% rev)" if f["pct_of_rev"] is not None else ""
            hp = f["heat_pct"]
            heat = "" if hp is None else (" hot" if hp >= 0.9 else " warm" if hp >= 0.7 else "")
            lines.append(f"{mark} {f['symbol']} {amt}{pc}{' ·' + heat if heat else ''}{' · trend ✅' if f['trend'] else ''}"
                         + (" · HELD" if f["held"] else "") + (f" 📄 {f['url']}" if f.get("url") else ""))

    return lines


def _book_lines() -> list[str]:
    """Open paper batches (tracked folders only) and the sell list."""
    lines: list[str] = []
    book = []
    for tag, folder in TRACKS.items():
        of = folder / "outcomes.jsonl"
        if not of.exists():
            continue
        last = {}
        for l in of.read_text().splitlines():
            if l.strip():
                x = json.loads(l); last[x["screen_id"]] = x
        last = {k: v for k, v in last.items() if (folder / f"screen_{k}.json").exists()}   # not_invested/ batches are not shown
        open_ = [f"{pd.Timestamp(sid):%b %d} {x['ew_net']:+.1f}%" for sid, x in sorted(last.items()) if x.get("status") != "CLOSED"]
        if open_:
            book.append(f"{SHORT[tag]} " + ", ".join(open_))
    lines.append("")
    if book:
        lines.append("📈 " + " · ".join(book))
    lines += [l for l in sell_lines("sri_lakshmi") if l]
    return lines


def _heat_lines() -> list[str]:
    """Industries that just warmed up or cooled down (from the orders digest)."""
    lines: list[str] = []
    oj = ROOT / "logs/daily_orders/latest.json"
    O = json.loads(oj.read_text()) if oj.exists() else {}
    if oj.exists() and (O.get("warming") or O.get("cooling")):
        cap = lambda xs: ", ".join(xs[:3]) + (f" +{len(xs) - 3}" if len(xs) > 3 else "")  # noqa: E731
        lines.append("")
        if O.get("warming"):
            lines.append("🌡 Warming: " + cap(O["warming"]))
        if O.get("cooling"):
            lines.append("❄️ Cooling: " + cap(O["cooling"]))
    return lines


# Inputs of the strategies (2026-10-04, "do we have systems to robustly fetch all inputs ... health"): latest date in each
# file and how many calendar days old it may be before the message flags it. Shown every day so a stale input is seen.
INPUTS = [("prices", "data/derived/stock_daily_facts_adjusted_2015plus.parquet", "trade_date", 4),
          ("filings", "data/events_full_history/normalized/stock_announcements.parquet", "event_date", 4),
          ("order amounts", "data/derived/order_amounts.parquet", "ts", 7),
          ("P&L", "data/derived/pnl_quarterly.parquet", "filing_dt", 10),
          ("bad-news check", "data/derived/event_ledger.parquet", "filed_at", 7),
          ("industry heat", "data/derived/industry_scores_policy.parquet", "date", 8)]


def inputs_line() -> str:
    out = []
    for label, path, colname, days in INPUTS:
        try:
            d = pd.to_datetime(pd.read_parquet(ROOT / path, columns=[colname])[colname], errors="coerce").max()
            ok = (pd.Timestamp.now().normalize() - d.normalize()).days <= days
            out.append(f"{label} {d:%b %d} {'✅' if ok else '⚠️'}")
        except Exception:
            out.append(f"{label} ❌ unreadable")
    return "📚 Inputs: " + " · ".join(out)


def daily_text(run: dict | None = None) -> str:
    """Short daily note (2026-09-30: "condensed, not long unreadable ones"): one line per topic, problems spelled out,
    anything fine collapses to a tick. Full detail stays in reports/ and logs/."""
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    ok, bad = st.get("passed", []), st.get("failed", [])
    head = f"SGM · {pd.Timestamp(st.get('date')):%b %d}"
    if run:
        head += " · " + run_line(run)
    lines = [head] + ([missed_line(run)] if run and missed_line(run) else []) + [""]

    gate = _latest("gate_data_*.json", ROOT / "logs/evals")
    g = json.loads(gate.read_text())["results"] if gate else []
    npass = sum(r["status"] == "PASS" for r in g)
    ev = _latest("eval_run_*.json", ROOT / "logs/evals")
    e = json.loads(ev.read_text()) if ev else {"results": [], "verdict": "n/a"}
    pend = sum(r["status"] == "PENDING" for r in e["results"])
    lines.append(f"📥 Data {'✅' if not bad else '⚠️'} {len(ok)}/{len(ok) + len(bad)} · QC {'✅' if g and npass == len(g) else '⚠️'} {npass}/{len(g)}"
                 f" · checks {e['verdict'].split(' —')[0]}" + (f" · {pend} waiting on you" if pend else ""))
    if bad:
        lines.append("❌ feeds failed: " + ", ".join(bad))
    for r in g:
        if r["status"] != "PASS":
            lines.append(f"{ICON.get(r['status'], '')} {QC_LABEL.get(r['id'], r['id'])}: {(r['detail'] or str(r['value']))[:90]}")
    other = [r["id"] for r in e["results"] if r["status"] in ("FAIL", "WARN") and not r["id"].startswith(("data.", "pit.filings", "prov.", "ops."))]
    if other:
        lines.append("⚠️ other checks: " + ", ".join(other))
    lines.append(inputs_line())
    ch = _latest("input_chain_*.json", ROOT / "logs/evals")
    if ch:
        sys.path.insert(0, str(ROOT / "src/agentic/trust"))
        import input_chain
        lines.append(input_chain.line(json.loads(ch.read_text())))

    lines += _orders_lines(st) + _book_lines() + _heat_lines()
    try:
        import research_queue
        lines += [""] + [l for l in research_queue.status_lines() if l]
    except Exception as x:                      # the queue must never break the daily message
        lines.append(f"🧪 queue status unavailable ({type(x).__name__})")
    return tidy(lines)


def weekend_text(run: dict | None = None) -> str:
    """Saturday / Sunday note (Abhinav 2026-10-04: "let's not skip telegram on weekends - mujhe still reports chahye").
    NSE is closed, so no new prices: last data run, input health, order filings since the last message (companies do
    file on weekends), the open batches at the last close, and the next buy if one is due."""
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    ok, bad = st.get("passed", []), st.get("failed", [])
    now = datetime.now()
    head = f"SGM · {now:%a %b %d} · weekend · data {pd.Timestamp(st.get('date')):%b %d}"
    if run:
        head += f" · ran {datetime.fromisoformat(run['started']):%-I:%M %p}"
    lines = [head, f"Market closed · next session {nse_calendar.next_session(now):%a %b %d}", "",
             f"📥 Last data run {pd.Timestamp(st.get('date')):%b %d}: feeds {'✅' if not bad else '⚠️'} {len(ok)}/{len(ok) + len(bad)}"
             + (" · failed: " + ", ".join(bad) if bad else ""), inputs_line()]
    lines += _orders_lines(st) + _book_lines()
    f = _latest("screen_*.json", TRACKS["sri_lakshmi"])
    if f is not None:
        sc = json.loads(f.read_text())
        entry = nse_calendar.next_session(sc["data_through"])
        if entry >= pd.Timestamp(now.date()) and sc["names"]:
            name = "Sri Lakshmi V3" if "V3" in sc.get("status", "") else "Sri Lakshmi G1"
            lines += ["", f"🛒 {entry:%a %b %d} at open: buy {', '.join(n['symbol'] for n in sc['names'])} ({name}, list of "
                          f"{pd.Timestamp(sc['data_through']):%b %d}; limits in the Friday sheet)"]
    lines += _heat_lines()
    return tidy(lines)


def tidy(lines: list[str]) -> str:
    """Texting hygiene: single spaces, no trailing spaces, one blank line between sections, none at the ends."""
    if isinstance(lines, str):   # a plain string was once passed and went out one letter per line (2026-09-30)
        lines = lines.splitlines()
    out = []
    for l in lines:
        l = re.sub(r"[ \t]+", " ", l).strip()
        if l == "" and (not out or out[-1] == ""):
            continue
        out.append(l)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _sleeve() -> float | None:
    f = CFG / "sleeve.json"
    return json.loads(f.read_text()).get("sleeve_inr") if f.exists() else None


def _shares(per: float | None, price: float | None) -> int | None:
    """Planned shares for one lot: the per-stock amount at the price the buy list used (last close before entry)."""
    return int(per // price) if per and price else None


def sell_lines(track: str = "sri_lakshmi", preview_days: int | None = None) -> list[str]:
    """Lots that reach session 126 at the next Indian close (sell), those within 5 sessions, and the next sell date.
    The plan's lots, not your broker fills. preview_days pretends every open batch has held that many sessions."""
    of = TRACKS[track] / "outcomes.jsonl"
    if not of.exists():
        return []
    last = {}
    for l in of.read_text().splitlines():
        if l.strip():
            x = json.loads(l); last[x["screen_id"]] = x
    last = {k: v for k, v in last.items() if (TRACKS[track] / f"screen_{k}.json").exists()}   # only batches still tracked
    sl, out, soon, nxt = _sleeve(), [], [], None
    for sid, x in sorted(last.items()):
        if x.get("status") == "CLOSED" or not x.get("names"):
            continue
        days = x["days_held"] if preview_days is None else preview_days
        entry = pd.Timestamp(x["names"][0]["entry_date"])
        sell_on = nse_calendar.add_sessions(entry, 126)
        if 126 - days <= 1:
            sc = json.loads((TRACKS[track] / f"screen_{sid}.json").read_text())
            close = {n["symbol"]: n.get("close") for n in sc["names"]}
            per = sl / 26 / len(sc["names"]) if sl else None
            head = "OVERDUE · sell at the next close" if days >= 126 else "sell at today's close (3:30 PM IST = 6:00 AM ET)"
            out.append(f"🔴 SELL {SHORT[track]} batch {pd.Timestamp(sid):%b %d} · {head}")
            for n in x["names"]:
                q = _shares(per, close.get(n["symbol"]))
                out.append(f"• {n['symbol']}" + (f" {q} sh" if q else "") + f" · ₹{n['entry']:,.2f} → ₹{n['last']:,.2f} ({n['ret_gross']:+.1f}%)")
            out.append(f"Batch after costs {x['ew_net']:+.1f}% · money rolls into next week's batch")
        elif 126 - days <= 5:
            soon.append(f"🟡 {SHORT[track]} batch {pd.Timestamp(sid):%b %d} sells in {126 - days} sessions (~{sell_on:%a %b %d})")
        else:
            nxt = min(nxt, sell_on) if nxt is not None else sell_on
    if out or soon:
        return [""] + out + soon
    return [f"💤 Sell: none due" + (f" · next ~{nxt:%b %d %Y}" if nxt is not None else "")]


def weekly_text(track: str) -> str:
    """Friday buy list (format reviewed 2026-09-30): header, one line per stock, when to sell, one reminder."""
    f = _latest("screen_*.json", TRACKS[track])
    if f is None:
        return f"{LABEL[track]}: no batch saved yet"
    sc = json.loads(f.read_text())
    sleeve = _sleeve()
    per = sleeve / 26 / len(sc["names"]) if sleeve and sc["names"] else None
    entry = nse_calendar.next_session(sc["data_through"])     # NSE holiday list (a Thu list must not say "buy Fri" on a holiday)
    sell = nse_calendar.add_sessions(entry, 126)
    name = LABEL[track] if track != "sri_lakshmi" else ("Sri Lakshmi V3" if "V3" in sc.get("status", "") else "Sri Lakshmi G1")
    lines = [f"🛒 {name} · buy {entry:%a %b %d} at open · data {pd.Timestamp(sc['data_through']):%b %d}",
             f"{len(sc['names'])} stocks" + (f" · ₹{per:,.0f} each" if per else " · equal weight"), ""]
    for i, n in enumerate(sc["names"], 1):
        c = n.get("close")
        q = _shares(per, c)
        lines.append(f"{i}. {n['symbol']}" + (f" · {q} sh" if q else "") + (f" · limit ₹{c * 1.05:,.2f}" if c else "")
                     + (" · ⚑ takeover" if n.get("takeover") else ""))
    skip = [f"{k} {', '.join(sc[c])}" for k, c in (("financials", "dropped_financials"), ("fading theme", "dropped_fading_theme")) if sc.get(c)]
    if skip:
        lines.append("Skipped (not refilled): " + " · ".join(skip))
    lines += ["", f"Sell all at close ~{sell:%b %d %Y} · no stop-loss", "Sign the Saturday review before real money"]
    return tidy(lines)


def send(text: str, kind: str) -> None:
    OUTBOX.mkdir(parents=True, exist_ok=True)
    (OUTBOX / f"{datetime.now():%Y%m%d_%H%M%S}_{kind}.txt").write_text(text + "\n")
    title = {"daily": "SGM daily", "weekend": "SGM weekend", "weekly": "SGM batch", "fail": "SGM ❌"}.get(kind, "SGM")
    first = text.splitlines()[1] if len(text.splitlines()) > 1 else text
    subprocess.run(["osascript", "-e", f'display notification {json.dumps(first[:200])} with title {json.dumps(title)}'], capture_output=True)
    env = CFG / "telegram.env"
    if env.exists():
        kv = dict(l.split("=", 1) for l in env.read_text().splitlines() if "=" in l and not l.startswith("#"))
        tok, chat = kv.get("TELEGRAM_BOT_TOKEN", "").strip(), kv.get("TELEGRAM_CHAT_ID", "").strip()
        if tok and chat:
            def post(body: str, html_mode: bool):
                args = ["curl", "-s", "-m", "30", f"https://api.telegram.org/bot{tok}/sendMessage",
                        "--data-urlencode", f"chat_id={chat}", "--data-urlencode", f"text={body}",
                        "--data-urlencode", "disable_web_page_preview=true"]
                if html_mode:
                    args += ["--data-urlencode", "parse_mode=HTML"]
                return subprocess.run(args, capture_output=True, text=True)
            r = post(_telegram_html(text), True)                             # always HTML: bold title, tappable links
            if '"ok":true' not in (r.stdout or ""):                          # HTML refused: send it plain rather than lose it
                (OUTBOX / f"{datetime.now():%Y%m%d_%H%M%S}_telegram_html_fallback.txt").write_text((r.stdout or r.stderr)[:500])
                r = post(text, False)
            if '"ok":true' not in (r.stdout or ""):
                (OUTBOX / f"{datetime.now():%Y%m%d_%H%M%S}_telegram_error.txt").write_text((r.stdout or r.stderr)[:500])
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["daily", "weekend", "weekly", "fail", "test"])
    ap.add_argument("--track", default="sri_lakshmi", choices=list(TRACKS))
    ap.add_argument("--step", default="")
    ap.add_argument("--detail", default="")
    a = ap.parse_args()
    if a.kind == "daily":
        send(daily_text(), "daily")
    elif a.kind == "weekend":
        send(weekend_text(), "weekend")
    elif a.kind == "weekly":
        send(weekly_text(a.track), "weekly")
    elif a.kind == "fail":
        send(f"❌ SGM · {a.step} failed\n{a.detail}\nPicks on hold until it passes. Logs: logs/daily_data_layer/", "fail")
    else:
        send("SGM test message · if you can read this on your phone, alerts work.", "test")


if __name__ == "__main__":
    main()
