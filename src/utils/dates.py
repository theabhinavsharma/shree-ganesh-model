from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


def iter_calendar_dates(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def is_weekend(value: date) -> bool:
    return value.weekday() >= 5
