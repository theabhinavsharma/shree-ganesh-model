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

ANNOUNCEMENTS_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
ANNOUNCEMENTS_URL = "https://www.nseindia.com/api/corporate-announcements?{query}"

ORDER_PATTERNS = (
    "order",
    "contract",
    "work order",
    "purchase order",
    "award",
    "letter of award",
    "loi",
)
APPROVAL_PATTERNS = (
    "approval",
    "approved",
    "clearance",
    "authorisation",
    "authorization",
    "consent",
    "licence",
    "license",
)
PLEDGE_PATTERNS = (
    "pledge",
    "invocation of pledge",
    "release of pledge",
    "encumbrance",
)
PROMOTER_BUY_PATTERNS = (
    "promoter",
    "acquisition",
    "inter-se",
    "inter se",
    "buy",
    "purchase",
)
RESULTS_PATTERNS = (
    "financial result",
    "financial results",
    "results",
    "quarter ended",
    "unaudited",
    "audited",
)


@dataclass(frozen=True)
class NseAnnouncementFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    symbols: set[str] | None = None
    delay_seconds: float = 0.05
    window_days: int = 31


def load_announcements_from_nse(config: NseAnnouncementFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=ANNOUNCEMENTS_REFERER)
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _iter_windows(config.start_date, config.end_date, config.window_days):
        raw_name = f"{window_start.isoformat()}_{window_end.isoformat()}.json"
        raw_path = config.output_dir / "raw" / raw_name
        if raw_path.exists():
            rows = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            rows = _fetch_window(session, window_start, window_end, config.symbols)
            write_json(rows, raw_path)
        if rows:
            frames.append(_normalize_rows(rows))
        time.sleep(config.delay_seconds)

    announcements = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if announcements.empty:
        return announcements

    announcements = announcements.sort_values(["event_date", "symbol", "sequence_id"]).drop_duplicates(
        subset=["sequence_id", "symbol"], keep="last"
    )
    announcements = announcements.reset_index(drop=True)
    write_parquet(announcements, config.output_dir / "normalized" / "stock_announcements.parquet")
    return announcements


def _fetch_window(
    session,
    start_date: date,
    end_date: date,
    symbols: set[str] | None,
) -> list[dict[str, object]]:
    if symbols:
        rows: list[dict[str, object]] = []
        for symbol in sorted(symbols):
            query = urlencode(
                {
                    "index": "equities",
                    "symbol": symbol,
                    "from_date": start_date.strftime("%d-%m-%Y"),
                    "to_date": end_date.strftime("%d-%m-%Y"),
                }
            )
            rows.extend(get_json(session, ANNOUNCEMENTS_URL.format(query=query), referer=ANNOUNCEMENTS_REFERER))
        return rows

    query = urlencode(
        {
            "index": "equities",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
        }
    )
    return get_json(session, ANNOUNCEMENTS_URL.format(query=query), referer=ANNOUNCEMENTS_REFERER)


def _normalize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    event_date = _parse_event_series(frame.get("an_dt"))
    fallback_date = _parse_event_series(frame.get("dt"))
    sort_date = _parse_event_series(frame.get("sort_date"))
    frame["event_date"] = event_date.fillna(sort_date).fillna(fallback_date).dt.normalize()
    frame["headline_text"] = frame.apply(_headline_text, axis=1)
    frame["event_category"] = frame["headline_text"].map(_classify_event_category)
    frame["is_results_event"] = frame["headline_text"].map(lambda text: _contains_any(text, RESULTS_PATTERNS))
    frame["is_order_win"] = frame["headline_text"].map(lambda text: _contains_any(text, ORDER_PATTERNS))
    frame["is_approval"] = frame["headline_text"].map(lambda text: _contains_any(text, APPROVAL_PATTERNS))
    frame["is_pledge_change"] = frame["headline_text"].map(lambda text: _contains_any(text, PLEDGE_PATTERNS))
    frame["is_promoter_buying"] = frame["headline_text"].map(_looks_like_promoter_buying)
    frame["has_attachment"] = frame.get("attchmntFile").fillna("").astype(str).str.len().gt(0)

    normalized = pd.DataFrame(
        {
            "symbol": frame.get("symbol", pd.Series(dtype="object")).astype(str).str.strip().str.upper(),
            "event_date": frame["event_date"],
            "sequence_id": frame.get("seq_id", pd.Series(dtype="object")).astype(str).str.strip(),
            "description": frame.get("desc", pd.Series(dtype="object")).astype(str).str.strip(),
            "attachment_text": frame.get("attchmntText", pd.Series(dtype="object")).astype(str).str.strip(),
            "event_category": frame["event_category"],
            "is_results_event": frame["is_results_event"].astype("boolean"),
            "is_order_win": frame["is_order_win"].astype("boolean"),
            "is_approval": frame["is_approval"].astype("boolean"),
            "is_pledge_change": frame["is_pledge_change"].astype("boolean"),
            "is_promoter_buying": frame["is_promoter_buying"].astype("boolean"),
            "has_attachment": frame["has_attachment"].astype("boolean"),
            "industry_hint": frame.get("smIndustry", pd.Series(dtype="object")).astype(str).str.strip(),
            "source_url": "https://www.nseindia.com/api/corporate-announcements",
            "source_note": "official_nse_corporate_announcements",
        }
    )
    return normalized.dropna(subset=["event_date"]).copy()


def _parse_event_series(series: pd.Series | object) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="datetime64[ns]")
    values = series.astype(str).str.strip()
    parsed = pd.to_datetime(values, format="%d-%b-%Y %H:%M:%S", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(values, format="%d-%b-%Y", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(values, format="%Y-%m-%d", errors="coerce"))
    return parsed


def _iter_windows(start_date: date, end_date: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=window_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _headline_text(row: pd.Series) -> str:
    parts = [
        str(row.get("desc", "")).strip(),
        str(row.get("attchmntText", "")).strip(),
    ]
    text = " | ".join(part for part in parts if part and part != "nan")
    return re.sub(r"\s+", " ", text).strip().lower()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _looks_like_promoter_buying(text: str) -> bool:
    if "promoter" not in text:
        return False
    return _contains_any(text, PROMOTER_BUY_PATTERNS)


def _classify_event_category(text: str) -> str:
    if _contains_any(text, RESULTS_PATTERNS):
        return "results"
    if _contains_any(text, ORDER_PATTERNS):
        return "order_win"
    if _contains_any(text, APPROVAL_PATTERNS):
        return "approval"
    if _contains_any(text, PLEDGE_PATTERNS):
        return "pledge_change"
    if _looks_like_promoter_buying(text):
        return "promoter_buying"
    if "board meeting" in text:
        return "board_meeting"
    if "investor" in text or "analyst" in text or "conference call" in text:
        return "investor_communication"
    if "acquisition" in text or "merger" in text or "scheme of arrangement" in text or "demerger" in text:
        return "mna"
    if "allotment" in text or "qip" in text or "preferential" in text or "rights issue" in text:
        return "fund_raise"
    return "other"
