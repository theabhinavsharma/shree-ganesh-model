from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock
from threading import local

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.ingest.shareholding.nse import PLEDGE_URL
from src.ingest.shareholding.nse import SHAREHOLDING_DETAIL_URL
from src.ingest.shareholding.nse import SHAREHOLDING_MASTER_URL
from src.ingest.shareholding.nse import SHAREHOLDING_REFERER
from src.ingest.shareholding.nse import _is_quarter_end
from src.ingest.shareholding.nse import _normalize_shareholding_row
from src.utils.io import write_json
from src.utils.io import write_parquet

_THREAD_LOCAL = local()


@dataclass(frozen=True)
class BatchedShareholdingConfig:
    output_dir: Path
    start_date: date
    end_date: date
    delay_seconds: float = 0.05
    max_workers: int = 4
    listing_window_months: int = 3
    limit_records: int | None = None
    batch_timeout_seconds: float = 180.0


def load_shareholding_history_from_nse(config: BatchedShareholdingConfig) -> pd.DataFrame:
    master_rows = _fetch_master_rows(config)
    return build_shareholding_history_from_raw(
        output_dir=config.output_dir,
        master_rows=master_rows,
        limit_records=config.limit_records,
        fetch_missing=True,
        max_workers=config.max_workers,
        delay_seconds=config.delay_seconds,
        batch_timeout_seconds=config.batch_timeout_seconds,
    )


def build_shareholding_history_from_raw(
    *,
    output_dir: Path,
    master_rows: list[dict[str, object]] | None = None,
    limit_records: int | None = None,
    fetch_missing: bool = False,
    max_workers: int = 4,
    delay_seconds: float = 0.05,
    batch_timeout_seconds: float | None = None,
) -> pd.DataFrame:
    if master_rows is None:
        master_rows = _read_cached_master_rows(output_dir)
    deduped_rows = _dedupe_master_rows(master_rows)
    if limit_records is not None:
        deduped_rows = deduped_rows[: limit_records]

    pledge_cache: dict[str, list[dict[str, object]]] = {}
    pledge_lock = Lock()
    normalized_rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_and_normalize_record if fetch_missing else _normalize_from_cached_record,
                row,
                output_dir,
                pledge_cache,
                pledge_lock,
            ): row
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
                f"shareholding batched detail fetch timed out after {batch_timeout_seconds}s "
                f"with {completed} of {len(futures)} records completed"
            ) from exc

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    df = df[df["quarter_end"].apply(_is_quarter_end)].copy()
    df = df.sort_values(["symbol", "quarter_end", "effective_from_date"]).reset_index(drop=True)
    for column in ["promoter_pct", "fii_fpi_pct", "dii_pct", "mf_pct"]:
        df[f"{column}_qoq_change"] = df.groupby("symbol")[column].diff()
    write_parquet(df, output_dir / "normalized" / "stock_shareholding_quarterly.parquet")
    return df


def _fetch_master_rows(config: BatchedShareholdingConfig) -> list[dict[str, object]]:
    session = build_session(warm=True, referer=SHAREHOLDING_REFERER)
    rows: list[dict[str, object]] = []
    for start_dt, end_dt in _iter_quarter_windows(config.start_date, config.end_date, config.listing_window_months):
        url = (
            f"{SHAREHOLDING_MASTER_URL}&from_date={start_dt.strftime('%d-%m-%Y')}&to_date={end_dt.strftime('%d-%m-%Y')}"
        )
        payload = get_json(session, url, referer=SHAREHOLDING_REFERER)
        write_json(
            payload,
            config.output_dir / "raw" / "master" / f"{start_dt.isoformat()}_{end_dt.isoformat()}.json",
        )
        rows.extend(payload)
        time.sleep(config.delay_seconds)
    return rows


def _read_cached_master_rows(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "raw" / "master").glob("*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _fetch_and_normalize_record(
    master_row: dict[str, object],
    output_dir: Path,
    pledge_cache: dict[str, list[dict[str, object]]],
    pledge_lock: Lock,
) -> dict[str, object] | None:
    symbol = str(master_row.get("symbol", "")).strip().upper()
    record_id = str(master_row.get("recordId", "")).strip()
    if not symbol or not record_id:
        return None

    try:
        detail_path = output_dir / "raw" / "detail_public_shareholder" / f"{record_id}.json"
        if detail_path.exists():
            public_rows = json.loads(detail_path.read_text(encoding="utf-8"))
        else:
            session = _thread_session()
            public_rows = get_json(
                session,
                SHAREHOLDING_DETAIL_URL.format(record_id=record_id, detail_index="public-shareholder"),
                referer=SHAREHOLDING_REFERER,
            )
            write_json(public_rows, detail_path)

        with pledge_lock:
            pledge_rows = pledge_cache.get(symbol)
        if pledge_rows is None:
            pledge_path = output_dir / "raw" / "pledgedata" / f"{symbol}.json"
            if pledge_path.exists():
                pledge_rows = json.loads(pledge_path.read_text(encoding="utf-8"))
            else:
                session = _thread_session()
                pledge_rows = get_json(session, PLEDGE_URL.format(symbol=symbol), referer=SHAREHOLDING_REFERER).get(
                    "data", []
                )
                write_json(pledge_rows, pledge_path)
            with pledge_lock:
                pledge_cache[symbol] = pledge_rows

        normalized = _normalize_shareholding_row(master_row, public_rows, pledge_rows)
        normalized["public_breakdown_available_flag"] = bool(public_rows)
        return normalized
    except Exception as exc:  # pragma: no cover - defensive network path
        error_path = output_dir / "raw" / "errors" / f"{symbol}_{record_id}.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(str(exc), encoding="utf-8")
        return None


def _normalize_from_cached_record(
    master_row: dict[str, object],
    output_dir: Path,
    pledge_cache: dict[str, list[dict[str, object]]],
    pledge_lock: Lock,
) -> dict[str, object] | None:
    symbol = str(master_row.get("symbol", "")).strip().upper()
    record_id = str(master_row.get("recordId", "")).strip()
    if not symbol or not record_id:
        return None
    detail_path = output_dir / "raw" / "detail_public_shareholder" / f"{record_id}.json"
    if not detail_path.exists():
        return None
    public_rows = json.loads(detail_path.read_text(encoding="utf-8"))
    with pledge_lock:
        pledge_rows = pledge_cache.get(symbol)
    if pledge_rows is None:
        pledge_path = output_dir / "raw" / "pledgedata" / f"{symbol}.json"
        pledge_rows = json.loads(pledge_path.read_text(encoding="utf-8")) if pledge_path.exists() else []
        with pledge_lock:
            pledge_cache[symbol] = pledge_rows
    normalized = _normalize_shareholding_row(master_row, public_rows, pledge_rows)
    normalized["public_breakdown_available_flag"] = bool(public_rows)
    return normalized


def _dedupe_master_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row.get("symbol", "")).strip().upper(),
            str(row.get("recordId", "")).strip(),
        )
        if key == ("", ""):
            continue
        existing = deduped.get(key)
        if existing is None or str(row.get("broadcastDate", "")) > str(existing.get("broadcastDate", "")):
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
        session = build_session(warm=True, referer=SHAREHOLDING_REFERER)
        _THREAD_LOCAL.session = session
    return session
