"""Claim checker — a Claude Code Stop hook (2026-09-29). Runs every time Claude finishes a reply in this project.

Plain English: every number Claude writes should come from something that was actually computed or given. This reads
the session log, collects every number that appeared in real tool output or in the user's own messages (the
"receipts"), then checks each number in Claude's latest reply. Numbers with no receipt are listed back to the user as
"unsourced" and logged, so the unsourced-claim rate can be tracked over time.

It never blocks the reply and never makes Claude loop: it only adds a visible note (systemMessage) and a log line.
Skipped as noise: years, dates, times, list numbers, single digits, commit hashes, version strings.
Log: logs/trust/claims.jsonl (one line per checked reply).
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
LOG = ROOT / "logs/trust/claims.jsonl"
TAIL_BYTES = 25_000_000                      # recent part of the session log is enough and keeps the hook fast

NUM = re.compile(r"(?<![\w.])(?:₹\s?)?(\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)(\s?(?:%|x|k|K|M|B|GB|MB|cr|crore|lakh|L|million|mn|billion|bn)\b|%)?")
DATEISH = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b|\b\d{1,2}:\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\b")
SUFFIX = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "L": 1e5, "lakh": 1e5, "cr": 1e7, "crore": 1e7,
          "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}


def _text_of(content) -> list[str]:
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    out.append(c.get("text", ""))
                elif c.get("type") == "tool_result":
                    out += _text_of(c.get("content"))
    return out


def read_session(path: Path) -> tuple[list[str], str]:
    """(receipt texts, text of Claude's reply since the last real user message)."""
    size = path.stat().st_size
    with open(path, "rb") as fh:
        fh.seek(max(0, size - TAIL_BYTES))
        raw = fh.read().decode("utf-8", "replace").splitlines()[1:]
    receipts, reply = [], []
    for l in raw:
        try:
            d = json.loads(l)
        except Exception:
            continue
        t, m = d.get("type"), d.get("message") or {}
        cont = m.get("content")
        if t == "user":
            is_tool = isinstance(cont, list) and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in cont)
            receipts += _text_of(cont)
            if not is_tool:
                reply = []                                  # a real user message starts a new turn
        elif t == "queue-operation" and d.get("content"):
            receipts.append(str(d.get("content")))          # messages the user typed mid-turn
        elif t == "assistant":
            for c in (cont or []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "text":
                    reply.append(c.get("text", ""))
                elif c.get("type") == "tool_use":            # files Claude wrote define thresholds / settings: they count as receipts
                    inp = c.get("input") or {}
                    receipts += [str(inp.get(k)) for k in ("content", "new_string", "command") if inp.get(k)]
    return receipts, "\n".join(reply)


def values(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals) for each number in text, noise removed."""
    text = DATEISH.sub(" ", text)
    text = re.sub(r"`[^`]*`|\b[0-9a-f]{7,40}\b|v?\d+\.\d+\.\d+", " ", text)   # code spans, hashes, versions
    out = []
    for m in NUM.finditer(text):
        s, suf = m.group(1), (m.group(2) or "").strip()
        v = float(s.replace(",", ""))
        dec = len(s.split(".")[1]) if "." in s else 0
        if suf in SUFFIX:
            v *= SUFFIX[suf]
        out.append((m.group(0).strip(), v, dec))
    return out


def trivial(s: str, v: float) -> bool:
    return (v.is_integer() and v <= 10 and "%" not in s) or (1900 <= v <= 2100 and v.is_integer() and "%" not in s)


def supported(v: float, dec: int, pool) -> bool:
    """True if some receipt r, as-is or rescaled (fraction <-> percent, rupees <-> lakh / crore), rounds to v."""
    import numpy as np
    tol = max(0.5 * 10 ** (-dec) if dec else 0.5, 0.005 * abs(v))
    for k in (1.0, 100.0, 0.01, 1e5, 1e7, 1e-5, 1e-7):          # c = r * k  ->  r = v / k
        lo, hi = (v - tol) / k, (v + tol) / k
        i = np.searchsorted(pool, lo, side="left")
        if i < len(pool) and pool[i] <= hi:
            return True
    return False


def main() -> None:
    try:
        ev = json.load(sys.stdin)
    except Exception:
        return
    tp = ev.get("transcript_path")
    if not tp or not Path(tp).exists():
        return
    receipts, reply = read_session(Path(tp))
    if not reply.strip():
        return
    import numpy as np
    pool = np.unique(np.array([v for t in receipts for _, v, _ in values(t)], dtype=float))
    claims = [(s, v, d) for s, v, d in values(reply) if not trivial(s, v)]
    unsourced = [s for s, v, d in claims if not supported(v, d, pool)]
    seen, uniq = set(), []
    for s in unsourced:
        if s not in seen:
            seen.add(s); uniq.append(s)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), session=ev.get("session_id"),
                                 n_numbers=len(claims), n_unsourced=len(unsourced), unsourced=uniq[:30])) + "\n")
    if uniq and os.environ.get("SGM_CLAIMCHECK_QUIET") != "1":
        msg = (f"Receipts check: {len(claims) - len(unsourced)} of {len(claims)} numbers in this reply trace to tool output "
               f"or your messages. No receipt for: {', '.join(uniq[:8])}{' …' if len(uniq) > 8 else ''}. "
               "Ask Claude to show the source for any of these.")
        print(json.dumps({"systemMessage": msg}))


if __name__ == "__main__":
    main()
