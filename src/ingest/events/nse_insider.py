from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
from requests import RequestException

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

INSIDER_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
INSIDER_PIT_URL = "https://www.nseindia.com/api/corporates-pit?{query}"


@dataclass(frozen=True)
class NseInsiderFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    symbols: set[str] | None = None
    delay_seconds: float = 0.05
    window_days: int = 31


def load_insider_trades_from_nse(config: NseInsiderFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=INSIDER_REFERER)
    frames: list[pd.DataFrame] = []
    for window_start, window_end in _iter_windows(config.start_date, config.end_date, config.window_days):
        raw_name = f"{window_start.isoformat()}_{window_end.isoformat()}.json"
        raw_path = config.output_dir / "raw" / raw_name
        if raw_path.exists():
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = _fetch_window(session, window_start, window_end, config.symbols)
                write_json(payload, raw_path)
            except RequestException as exc:
                write_json(
                    {
                        "window_start": window_start.isoformat(),
                        "window_end": window_end.isoformat(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    config.output_dir / "errors" / f"{window_start.isoformat()}_{window_end.isoformat()}.json",
                )
                continue
        rows = payload.get("data", []) if isinstance(payload, dict) else payload
        if rows:
            frames.append(_normalize_rows(rows))
        time.sleep(config.delay_seconds)

    insider = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if insider.empty:
        return insider
    insider = insider.sort_values(["symbol", "event_timestamp", "filing_id"]).drop_duplicates(
        subset=["filing_id", "symbol"], keep="last"
    )
    insider = insider.reset_index(drop=True)
    write_parquet(insider, config.output_dir / "normalized" / "stock_insider_trades.parquet")
    return insider


def _fetch_window(session, start_date: date, end_date: date, symbols: set[str] | None) -> dict[str, object] | list[dict[str, object]]:
    if symbols:
        rows: list[dict[str, object]] = []
        symbol_names: set[str] = set()
        for symbol in sorted({value.strip().upper() for value in symbols if value}):
            query = urlencode(
                {
                    "index": "equities",
                    "symbol": symbol,
                    "from_date": start_date.strftime("%d-%m-%Y"),
                    "to_date": end_date.strftime("%d-%m-%Y"),
                }
            )
            payload = get_json(session, INSIDER_PIT_URL.format(query=query), referer=INSIDER_REFERER)
            if isinstance(payload, dict):
                rows.extend(payload.get("data", []))
                symbol_names.update(payload.get("acqNameList", []))
            else:
                rows.extend(payload)
            time.sleep(0.02)
        return {"acqNameList": sorted(symbol_names), "data": rows}

    query = urlencode(
        {
            "index": "equities",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
        }
    )
    return get_json(session, INSIDER_PIT_URL.format(query=query), referer=INSIDER_REFERER)


def _normalize_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    event_timestamp = _parse_timestamp_series(frame.get("date"))
    intim_date = _parse_timestamp_series(frame.get("intimDt")).dt.normalize()
    from_date = _parse_timestamp_series(frame.get("acqfromDt")).dt.normalize()
    to_date = _parse_timestamp_series(frame.get("acqtoDt")).dt.normalize()
    transaction_date = to_date.fillna(from_date)
    event_date = event_timestamp.dt.normalize().fillna(intim_date).fillna(transaction_date)

    buy_value = _numeric_series(frame.get("buyValue"))
    sell_value = _numeric_series(frame.get("sellValue"))
    buy_qty = _numeric_series(frame.get("buyQuantity"))
    sell_qty = _numeric_series(frame.get("sellquantity"))
    sec_value = _numeric_series(frame.get("secVal"))
    before_pct = _numeric_series(frame.get("befAcqSharesPer"))
    after_pct = _numeric_series(frame.get("afterAcqSharesPer"))

    person_category = frame.get("personCategory", pd.Series(dtype="object")).astype(str).str.strip()
    transaction_type = frame.get("tdpTransactionType", pd.Series(dtype="object")).astype(str).str.strip().str.upper()
    normalized = pd.DataFrame(
        {
            "symbol": frame.get("symbol", pd.Series(dtype="object")).astype(str).str.strip().str.upper(),
            "company_name": frame.get("company", pd.Series(dtype="object")).astype(str).str.strip(),
            "event_timestamp": event_timestamp,
            "event_date": event_date,
            "transaction_from_date": from_date,
            "transaction_to_date": to_date,
            "intimation_date": intim_date,
            "filing_id": frame.get("pid", pd.Series(dtype="object")).astype(str).str.strip(),
            "acquirer_name": frame.get("acqName", pd.Series(dtype="object")).astype(str).str.strip(),
            "person_category": person_category,
            "transaction_type": transaction_type,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "net_value": buy_value.fillna(0.0) - sell_value.fillna(0.0),
            "buy_quantity": buy_qty,
            "sell_quantity": sell_qty,
            "net_quantity": buy_qty.fillna(0.0) - sell_qty.fillna(0.0),
            "security_value": sec_value,
            "before_holding_pct": before_pct,
            "after_holding_pct": after_pct,
            "holding_change_pct": after_pct - before_pct,
            "acquisition_mode": frame.get("acqMode", pd.Series(dtype="object")).astype(str).str.strip(),
            "exchange": frame.get("exchange", pd.Series(dtype="object")).astype(str).str.strip(),
            "remarks": frame.get("remarks", pd.Series(dtype="object")).astype(str).str.strip(),
            "is_promoter_group_or_promoter": person_category.str.contains("promoter", case=False, na=False),
            "is_director_or_kmp": person_category.str.contains("director|key managerial|kmp", case=False, na=False),
            "is_buy_transaction": transaction_type.eq("BUY") | buy_value.gt(0) | buy_qty.gt(0),
            "is_sell_transaction": transaction_type.eq("SELL") | sell_value.gt(0) | sell_qty.gt(0),
            "source_url": "https://www.nseindia.com/api/corporates-pit",
            "source_note": "official_nse_promoter_and_insider_transactions",
        }
    )
    return normalized.dropna(subset=["symbol", "event_date"]).copy()


def _numeric_series(series: pd.Series | object) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="float64")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_timestamp_series(series: pd.Series | object) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="datetime64[ns]")
    values = series.astype(str).str.strip()
    parsed = pd.to_datetime(values, format="%d-%b-%Y %H:%M", errors="coerce")
    parsed = parsed.fillna(pd.to_datetime(values, format="%d-%b-%Y", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(values, format="%Y-%m-%d %H:%M:%S", errors="coerce"))
    parsed = parsed.fillna(pd.to_datetime(values, errors="coerce", dayfirst=True))
    return parsed


def _iter_windows(start_date: date, end_date: date, window_days: int) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=window_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows
