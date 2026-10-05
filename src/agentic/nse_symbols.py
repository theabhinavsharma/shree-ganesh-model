"""Company identity across NSE ticker renames (2026-10-04; lesson INC-2026-09-30-renamed-symbol-cliffs: key corporate data
by company, never by ticker alone). Source: NSE's symbolchange.csv (data/raw/nse_symbol_change/symbolchange.csv), lines
"company name,OLD,NEW,DD-MON-YYYY" (the name may contain commas, so split from the right).
now_map()  -> {old ticker: today's ticker}, following chains (A -> B -> C gives A -> C and B -> C).
now(sym)   -> today's ticker for any ticker (itself if never renamed).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data/raw/nse_symbol_change/symbolchange.csv"


@lru_cache(maxsize=1)
def now_map() -> dict:
    step = {}
    for line in open(FILE, encoding="utf-8", errors="replace"):
        parts = line.strip().rsplit(",", 3)
        if len(parts) == 4 and pd.notna(pd.to_datetime(parts[3], format="%d-%b-%Y", errors="coerce")):
            old, new, d = parts[1].strip(), parts[2].strip(), pd.to_datetime(parts[3], format="%d-%b-%Y")
            if old and new and old != new and (old not in step or d > step[old][1]):
                step[old] = (new, d)
    out = {}
    for old in step:
        s, seen = old, set()
        while s in step and s not in seen:
            seen.add(s); s = step[s][0]
        out[old] = s
    return out


def now(sym: str) -> str:
    return now_map().get(sym, sym)
