"""Output gate for phone messages (2026-09-29): a message is sent only if every number in it can be traced to a file.

Plain English: before notify.py sends anything, this reads the same files the message was built from and checks
(1) every number in the message is a file value rounded exactly as shown (no loose rescaling), or is a stated formula of a file value (buy limit =
last close x 1.05, rupees per name = sleeve / 26 / names, feed counts), or is a fixed number of the template itself
(26 weekly slots, 126-session hold); and (2) the format is sane: not empty, under Telegram's 4,096-character limit,
no "nan" / "None" / tracebacks, the date on the first line matches the data, every pick in the file is listed.
A message that fails is held, not sent; the orchestrator sends a failure note naming what had no source instead.
Failure notes and test messages are never held (an alarm that can be blocked is not an alarm).
Log: logs/outbox/checks.jsonl (one line per check). Usage: check(kind, text, track=None) -> dict(ok, unsourced, problems)
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import claim_check as cc  # noqa: E402
import notify  # noqa: E402

LOG = ROOT / "logs/outbox/checks.jsonl"
TEMPLATE_NUMBERS = (26.0, 126.0)                      # 26 weekly slots in the ladder, 126-session hold
SYMBOLS = re.compile(r"\b(?=[A-Z0-9&]*[A-Z])[A-Z0-9&]{2,}\b")   # tickers like 20MICRONS are names, not numbers
BAD = re.compile(r"\bnan\b|\bNaN\b|\bNone\b|Traceback|Error:|[{}]")


RAW = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _nums(text: str) -> list[float]:
    """Every number in a source file as written (no unit scaling, sign dropped like the message parser does)."""
    return [float(m.replace(",", "")) for m in RAW.findall(text)]


def _grounded(shown: str, dec: int, pool: np.ndarray) -> bool:
    """Strict: the number as displayed must be a pool value rounded to the shown decimals (x100 only for a percent)."""
    d = float(RAW.search(shown).group(0).replace(",", ""))
    tol = 0.5 * 10 ** (-dec) * (1 + 1e-6) + 1e-9
    for k in ((1.0, 100.0) if "%" in shown else (1.0,)):
        lo, hi = (d - tol) / k, (d + tol) / k
        i = np.searchsorted(pool, lo, side="left")
        if i < len(pool) and pool[i] <= hi:
            return True
    return False


def _pool_daily() -> tuple[list[float], str]:
    st = json.loads((ROOT / "logs/daily_data_layer_status.json").read_text())
    ok, bad = st.get("passed", []), st.get("failed", [])
    texts = [json.dumps(st)]
    extra = [len(ok), len(bad), len(ok) + len(bad)]
    ev = notify._latest("eval_run_*.json", ROOT / "logs/evals")
    if ev:
        e = json.loads(ev.read_text()); texts.append(ev.read_text())
        extra += [sum(r["status"] == s for r in e["results"]) for s in ("FAIL", "PENDING", "WARN", "PASS")]
    for folder in notify.TRACKS.values():
        of = folder / "outcomes.jsonl"
        if of.exists():
            texts.append("\n".join(of.read_text().splitlines()[-400:]))
    od = ROOT / "data/derived/order_daily.parquet"
    if od.exists():
        o = pd.read_parquet(od, columns=["symbol", "ts", "amount_cr", "pct_of_rev"])
        o = o[pd.to_datetime(o["ts"]) >= pd.Timestamp(st.get("date")) - pd.Timedelta(days=1)]
        extra += list(o["amount_cr"].dropna()) + list(o["pct_of_rev"].dropna())
        extra.append(len(o[o["pct_of_rev"] >= 0.15].drop_duplicates(["symbol", "amount_cr"])))
    rep = notify._latest("daily_orders_industry_*.md", ROOT / "reports")
    if rep:
        texts.append(rep.read_text())
    return [v for t in texts for v in _nums(t)] + [float(x) for x in extra], str(st.get("date"))


def _pool_weekly(track: str) -> tuple[list[float], str, list[str]]:
    f = notify._latest("screen_*.json", notify.TRACKS[track])
    sc = json.loads(f.read_text())
    extra = [len(sc["names"])] + [n["close"] * 1.05 for n in sc["names"] if n.get("close")]
    sl = notify.CFG / "sleeve.json"
    if sl.exists():
        s = json.loads(sl.read_text()).get("sleeve_inr")
        if s:
            extra += [s, s / 26, s / 26 / len(sc["names"])]
    return _nums(f.read_text()) + [float(x) for x in extra], sc["data_through"], [n["symbol"] for n in sc["names"]]


def check(kind: str, text: str, track: str | None = None, log: bool = True) -> dict:
    problems, unsourced = [], []
    if kind in ("daily", "weekly"):
        if not text.strip():
            problems.append("empty message")
        if len(text) > 4000:
            problems.append(f"{len(text)} characters (Telegram limit 4,096)")
        if BAD.search(text):
            problems.append(f"bad token: {BAD.search(text).group(0)!r}")
        if kind == "daily":
            pool, date = _pool_daily()
        else:
            pool, date, syms = _pool_weekly(track)
            missing = [s for s in syms if s not in text]
            if missing:
                problems.append("picks in the file but not in the message: " + ", ".join(missing))
        if date not in text.splitlines()[0]:
            problems.append(f"first line does not carry the data date {date}")
        pool = np.sort(np.array(pool + (list(TEMPLATE_NUMBERS) if kind == "weekly" else []), dtype=float))
        for s, v, dec in cc.values(SYMBOLS.sub(" ", text)):
            if not cc.trivial(s, v) and not _grounded(s, dec, pool):
                unsourced.append(s.strip())
    res = dict(ok=not problems and not unsourced, unsourced=unsourced, problems=problems)
    if not log:
        return res
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), kind=kind, track=track, **res)) + "\n")
    return res


if __name__ == "__main__":
    k = sys.argv[1] if len(sys.argv) > 1 else "daily"
    tr = sys.argv[2] if len(sys.argv) > 2 else None
    t = notify.daily_text() if k == "daily" else notify.weekly_text(tr)
    print(t, "\n---\n", check(k, t, tr, log=False))
