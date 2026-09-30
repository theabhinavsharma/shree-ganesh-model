"""Trading-day arithmetic from NSE's own holiday list (data/derived/nse_holidays.parquet, fetch_nse_holidays.py).

next_session(d)      first weekday after d that is not an NSE holiday (e.g. Thu 2026-10-01 -> Mon 2026-10-05)
add_sessions(d, n)   the n-th session counting d as session 1 (sell-date estimate; years NSE has not published yet
                     count weekdays only, so dates there are approximate)
covers(d)            True when the list has NSE's holidays for d's year
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "data/derived/nse_holidays.parquet"


@lru_cache(maxsize=1)
def _holidays() -> tuple[frozenset, frozenset]:
    if not FILE.exists():
        return frozenset(), frozenset()
    d = pd.to_datetime(pd.read_parquet(FILE, columns=["date"])["date"]).dt.normalize()
    return frozenset(d), frozenset(d.dt.year)


def covers(d) -> bool:
    return pd.Timestamp(d).year in _holidays()[1]


def is_session(d) -> bool:
    d = pd.Timestamp(d).normalize()
    return d.weekday() < 5 and d not in _holidays()[0]


def next_session(d) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while not is_session(d):
        d += pd.Timedelta(days=1)
    return d


def add_sessions(d, n: int) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize()
    while not is_session(d):
        d += pd.Timedelta(days=1)
    for _ in range(n - 1):
        d = next_session(d)
    return d
