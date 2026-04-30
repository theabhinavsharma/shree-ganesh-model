from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import local

import pandas as pd

from src.ingest.fundamentals.nse import FINANCIALS_DETAIL_URL
from src.ingest.fundamentals.nse import FINANCIALS_LISTING_URL
from src.ingest.fundamentals.nse import FINANCIALS_REFERER
from src.ingest.fundamentals.nse import _add_growth_fields
from src.ingest.fundamentals.nse import _build_financial_detail_url
from src.ingest.fundamentals.nse import _normalize_financial_row
from src.ingest.fundamentals.nse import select_preferred_statement_scope
from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

_THREAD_LOCAL = local()


@dataclass(frozen=True)
class BatchedFundamentalsConfig:
    output_dir: Path
    start_date: date
    end_date: date
    statement_scope: str | None = None
    scope_preference: tuple[str, ...] = ("Non-Consolidated", "Consolidated")
    delay_seconds: float = 0.05
    max_workers: int = 4
    listing_window_months: int = 3
    limit_records: int | None = None
    batch_timeout_seconds: float = 180.0


def load_fundamentals_history_from_nse(config: BatchedFundamentalsConfig) -> pd.DataFrame:
    listing_rows = _fetch_listing_rows(config)
    return build_fundamentals_history_from_raw(
        output_dir=config.output_dir,
        listing_rows=listing_rows,
        statement_scope=config.statement_scope,
        scope_preference=config.scope_preference,
        limit_records=config.limit_records,
        fetch_missing=True,
        max_workers=config.max_workers,
        delay_seconds=config.delay_seconds,
        batch_timeout_seconds=config.batch_timeout_seconds,
    )


def build_fundamentals_history_from_raw(
    *,
    output_dir: Path,
    listing_rows: list[dict[str, object]] | None = None,
    statement_scope: str | None = None,
    scope_preference: tuple[str, ...] = ("Non-Consolidated", "Consolidated"),
    limit_records: int | None = None,
    fetch_missing: bool = False,
    max_workers: int = 4,
    delay_seconds: float = 0.05,
    batch_timeout_seconds: float | None = None,
) -> pd.DataFrame:
    if listing_rows is None:
        listing_rows = _read_cached_listing_rows(output_dir)
    if not listing_rows:
        return pd.DataFrame()

    if statement_scope is not None:
        listing_rows = [
            row
            for row in listing_rows
            if str(row.get("consolidated", "")).strip() == statement_scope
        ]
    deduped_rows = _dedupe_listing_rows(listing_rows)
    if limit_records is not None:
        deduped_rows = deduped_rows[: limit_records]

    if fetch_missing:
        normalized_rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch_and_normalize_detail, row, output_dir): row
                for row in deduped_rows
            }
            completed = 0
            try:
                for future in as_completed(futures, timeout=batch_timeout_seconds):
                    normalized = future.result()
                    if normalized:
                        normalized_rows.append(normalized)
                    completed += 1
                    time.sleep(delay_seconds)
            except TimeoutError as exc:
                for future in futures:
                    future.cancel()
                raise TimeoutError(
                    f"fundamentals batched detail fetch timed out after {batch_timeout_seconds}s "
                    f"with {completed} of {len(futures)} records completed"
                ) from exc
    else:
        normalized_rows = [_normalize_from_cached_detail(row, output_dir) for row in deduped_rows]
        normalized_rows = [row for row in normalized_rows if row]

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    df = df.sort_values(["symbol", "fiscal_period_end", "effective_from_date"]).reset_index(drop=True)
    df = _add_growth_fields(df)
    write_parquet(df, output_dir / "normalized" / "stock_quarterly_fundamentals_all_scopes.parquet")
    preferred = select_preferred_statement_scope(df, scope_preference=scope_preference)
    write_parquet(preferred, output_dir / "normalized" / "stock_quarterly_fundamentals.parquet")
    return preferred


def _fetch_listing_rows(config: BatchedFundamentalsConfig) -> list[dict[str, object]]:
    session = build_session(warm=True, referer=FINANCIALS_REFERER)
    rows: list[dict[str, object]] = []
    for start_dt, end_dt in _iter_quarter_windows(config.start_date, config.end_date, config.listing_window_months):
        url = (
            f"{FINANCIALS_LISTING_URL}&from_date={start_dt.strftime('%d-%m-%Y')}&to_date={end_dt.strftime('%d-%m-%Y')}"
        )
        payload = get_json(session, url, referer=FINANCIALS_REFERER)
        write_json(
            payload,
            config.output_dir / "raw" / "listing" / f"{start_dt.isoformat()}_{end_dt.isoformat()}.json",
        )
        rows.extend(payload)
        time.sleep(config.delay_seconds)
    return rows


def _read_cached_listing_rows(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "raw" / "listing").glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _fetch_and_normalize_detail(listing_row: dict[str, object], output_dir: Path) -> dict[str, object] | None:
    symbol = str(listing_row.get("symbol", "")).strip().upper()
    seq_number = str(listing_row.get("seqNumber", "")).strip()
    if not symbol or not seq_number:
        return None

    try:
        raw_path = output_dir / "raw" / "detail" / f"{symbol}_{seq_number}.json"
        if raw_path.exists():
            detail = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            session = _thread_session()
            detail_url = _build_financial_detail_url(listing_row)
            detail = get_json(session, detail_url, referer=FINANCIALS_REFERER)
            write_json(detail, raw_path)
        return _normalize_financial_row(listing_row, detail)
    except Exception as exc:  # pragma: no cover - defensive network path
        error_path = output_dir / "raw" / "errors" / f"{symbol}_{seq_number}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(str(exc), encoding="utf-8")
        return None


def _normalize_from_cached_detail(listing_row: dict[str, object], output_dir: Path) -> dict[str, object] | None:
    symbol = str(listing_row.get("symbol", "")).strip().upper()
    seq_number = str(listing_row.get("seqNumber", "")).strip()
    if not symbol or not seq_number:
        return None
    raw_path = output_dir / "raw" / "detail" / f"{symbol}_{seq_number}.json"
    if not raw_path.exists():
        return None
    detail = json.loads(raw_path.read_text(encoding="utf-8"))
    return _normalize_financial_row(listing_row, detail)


def _dedupe_listing_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("symbol", "")).strip().upper(),
            str(row.get("seqNumber", "")).strip(),
        )
        if key == ("", ""):
            continue
        existing = deduped.get(key)
        if existing is None or str(row.get("broadCastDate", "")) > str(existing.get("broadCastDate", "")):
            deduped[key] = row
    return list(deduped.values())


def _iter_quarter_windows(start_date: date, end_date: date, months: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    while cursor <= end_ts:
        window_end = min(cursor + pd.DateOffset(months=months) - pd.Timedelta(days=1), end_ts)
        windows.append((cursor.date(), window_end.date()))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def _thread_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session(warm=True, referer=FINANCIALS_REFERER)
        _THREAD_LOCAL.session = session
    return session
