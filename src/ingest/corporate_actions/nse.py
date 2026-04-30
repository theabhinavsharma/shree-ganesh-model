from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

CORPORATE_ACTIONS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
CORPORATE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions?{query}"

BONUS_RATIO_RE = re.compile(r"bonus(?:\s+issue)?[^\d]{0,20}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
SPLIT_RATIO_RE = re.compile(
    r"(?:face\s*value\s*split|stock\s*split|sub-division|subdivision|split)[^\d]{0,40}"
    r"(?:from)?\s*rs\.?\s*(\d+(?:\.\d+)?)\s*/?-?[^\d]{0,25}"
    r"(?:to)?\s*rs\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NseCorporateActionsFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    symbols: set[str] | None = None
    delay_seconds: float = 0.05
    window_days: int = 366


def load_corporate_actions_from_nse(config: NseCorporateActionsFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=CORPORATE_ACTIONS_REFERER)
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _iter_windows(config.start_date, config.end_date, config.window_days):
        if config.symbols:
            rows: list[dict[str, object]] = []
            for symbol in sorted(config.symbols):
                raw_name = f"{window_start.isoformat()}_{window_end.isoformat()}_{symbol}.json"
                raw_path = config.output_dir / "raw" / raw_name
                if raw_path.exists():
                    symbol_rows = json.loads(raw_path.read_text(encoding="utf-8"))
                else:
                    symbol_rows = _fetch_window(session, window_start, window_end, {symbol})
                    write_json(symbol_rows, raw_path)
                rows.extend(symbol_rows)
                time.sleep(config.delay_seconds)
        else:
            raw_name = f"{window_start.isoformat()}_{window_end.isoformat()}.json"
            raw_path = config.output_dir / "raw" / raw_name
            if raw_path.exists():
                rows = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                rows = _fetch_window(session, window_start, window_end, None)
                write_json(rows, raw_path)
            time.sleep(config.delay_seconds)
        if rows:
            frames.append(_normalize_rows(rows))

    actions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if actions.empty:
        return actions

    actions = actions.sort_values(["symbol", "ex_date", "subject"]).drop_duplicates(
        subset=["symbol", "ex_date", "subject"], keep="last"
    )
    actions = actions.reset_index(drop=True)
    write_parquet(actions, config.output_dir / "normalized" / "stock_corporate_actions.parquet")
    return actions


def _fetch_window(
    session,
    start_date: date,
    end_date: date,
    symbols: set[str] | None,
) -> list[dict[str, object]]:
    query = {
        "index": "equities",
        "from_date": start_date.strftime("%d-%m-%Y"),
        "to_date": end_date.strftime("%d-%m-%Y"),
    }
    if symbols and len(symbols) == 1:
        query["symbol"] = next(iter(symbols))
    return get_json(session, CORPORATE_ACTIONS_URL.format(query=urlencode(query)), referer=CORPORATE_ACTIONS_REFERER)


def _normalize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["symbol"] = frame.get("symbol", pd.Series(dtype="object")).astype(str).str.strip().str.upper()
    frame["series"] = frame.get("series", pd.Series(dtype="object")).astype(str).str.strip().str.upper()
    frame["company_name"] = frame.get("comp", pd.Series(dtype="object")).astype(str).str.strip()
    frame["subject"] = frame.get("subject", pd.Series(dtype="object")).astype(str).str.strip()
    frame["face_value"] = pd.to_numeric(frame.get("faceVal"), errors="coerce")
    frame["ex_date"] = _parse_date_series(frame.get("exDate"))
    frame["record_date"] = _parse_date_series(frame.get("recDate"))
    frame["broadcast_date"] = _parse_date_series(frame.get("caBroadcastDate"))
    frame["bonus_factor"] = frame["subject"].map(_parse_bonus_factor)
    frame["split_factor"] = frame["subject"].map(_parse_split_factor)
    frame["is_bonus"] = frame["bonus_factor"].notna()
    frame["is_split"] = frame["split_factor"].notna()
    frame["adjustment_factor"] = frame["bonus_factor"].fillna(1.0) * frame["split_factor"].fillna(1.0)
    frame["adjustment_factor"] = frame["adjustment_factor"].where(frame["is_bonus"] | frame["is_split"])
    normalized = pd.DataFrame(
        {
            "symbol": frame["symbol"],
            "series": frame["series"],
            "company_name": frame["company_name"],
            "isin": frame.get("isin", pd.Series(dtype="object")).astype(str).str.strip(),
            "subject": frame["subject"],
            "ex_date": frame["ex_date"],
            "record_date": frame["record_date"],
            "broadcast_date": frame["broadcast_date"],
            "face_value": frame["face_value"],
            "bonus_factor": frame["bonus_factor"],
            "split_factor": frame["split_factor"],
            "adjustment_factor": frame["adjustment_factor"],
            "is_bonus": frame["is_bonus"].astype("boolean"),
            "is_split": frame["is_split"].astype("boolean"),
            "source_url": "https://www.nseindia.com/api/corporates-corporateActions",
            "source_note": "official_nse_corporate_actions_split_bonus_only",
        }
    )
    return normalized.dropna(subset=["ex_date"]).copy()


def _parse_bonus_factor(subject: str) -> float | None:
    if not isinstance(subject, str):
        return None
    match = BONUS_RATIO_RE.search(subject)
    if not match:
        return None
    numerator = float(match.group(1))
    denominator = float(match.group(2))
    if denominator <= 0:
        return None
    return (numerator + denominator) / denominator


def _parse_split_factor(subject: str) -> float | None:
    if not isinstance(subject, str):
        return None
    match = SPLIT_RATIO_RE.search(subject)
    if not match:
        return None
    from_face = float(match.group(1))
    to_face = float(match.group(2))
    if to_face <= 0:
        return None
    factor = from_face / to_face
    if factor <= 0:
        return None
    return factor


def _parse_date_series(series: pd.Series | object) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="datetime64[ns]")
    values = series.astype(str).str.strip()
    parsed = pd.to_datetime(values, format="%d-%b-%Y", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(values, format="%d-%b-%Y %H:%M:%S", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(values, errors="coerce", dayfirst=True))
    return parsed.dt.normalize()


def _iter_windows(start_date: date, end_date: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=window_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows
