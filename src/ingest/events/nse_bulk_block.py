from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path

import pandas as pd
from requests import RequestException

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

BULK_BLOCK_REFERER = "https://www.nseindia.com/report-detail/display-bulk-and-block-deals"
BULK_BLOCK_URL = (
    "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals?"
    "optionType={option_type}&from={from_date}&to={to_date}"
)


@dataclass(frozen=True)
class NseBulkBlockFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    option_types: tuple[str, ...] = ("bulk_deals", "block_deals")
    delay_seconds: float = 0.03


def load_bulk_block_deals_from_nse(config: NseBulkBlockFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=BULK_BLOCK_REFERER)
    frames: list[pd.DataFrame] = []
    for option_type in config.option_types:
        for trade_date in _iter_days(config.start_date, config.end_date):
            raw_path = config.output_dir / "raw" / option_type / f"{trade_date.isoformat()}.json"
            if raw_path.exists():
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                try:
                    payload = _fetch_trade_date(session, trade_date, option_type)
                    write_json(payload, raw_path)
                except RequestException as exc:
                    write_json(
                        {
                            "trade_date": trade_date.isoformat(),
                            "deal_type": option_type,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        config.output_dir / "errors" / option_type / f"{trade_date.isoformat()}.json",
                    )
                    continue
            rows = payload.get("data", []) if isinstance(payload, dict) else payload
            if rows:
                frames.append(_normalize_rows(rows, option_type=option_type))
            time.sleep(config.delay_seconds)

    deals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if deals.empty:
        return deals
    deals = deals.sort_values(["deal_type", "trade_date", "symbol", "client_name", "buy_sell"]).drop_duplicates(
        subset=["deal_type", "trade_date", "symbol", "client_name", "buy_sell", "quantity_traded", "price"],
        keep="last",
    )
    deals = deals.reset_index(drop=True)
    write_parquet(deals, config.output_dir / "normalized" / "stock_bulk_block_deals.parquet")
    return deals


def _fetch_trade_date(session, trade_date: date, option_type: str) -> dict[str, object] | list[dict[str, object]]:
    formatted = trade_date.strftime("%d-%m-%Y")
    url = BULK_BLOCK_URL.format(option_type=option_type, from_date=formatted, to_date=formatted)
    return get_json(session, url, referer=BULK_BLOCK_REFERER)


def _normalize_rows(rows: list[dict[str, object]], *, option_type: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    trade_date = pd.to_datetime(frame.get("BD_DT_DATE"), format="%d-%b-%Y", errors="coerce").dt.normalize()
    quantity = pd.to_numeric(frame.get("BD_QTY_TRD"), errors="coerce")
    price = pd.to_numeric(frame.get("BD_TP_WATP"), errors="coerce")
    buy_sell = frame.get("BD_BUY_SELL", pd.Series(dtype="object")).astype(str).str.strip().str.upper()
    normalized = pd.DataFrame(
        {
            "deal_type": option_type,
            "trade_date": trade_date,
            "symbol": frame.get("BD_SYMBOL", pd.Series(dtype="object")).astype(str).str.strip().str.upper(),
            "company_name": frame.get("BD_SCRIP_NAME", pd.Series(dtype="object")).astype(str).str.strip(),
            "client_name": frame.get("BD_CLIENT_NAME", pd.Series(dtype="object")).astype(str).str.strip(),
            "buy_sell": buy_sell,
            "quantity_traded": quantity,
            "price": price,
            "traded_value": quantity * price,
            "remarks": frame.get("BD_REMARKS", pd.Series(dtype="object")).astype(str).str.strip(),
            "is_buy": buy_sell.eq("BUY"),
            "is_sell": buy_sell.eq("SELL"),
            "source_url": "https://www.nseindia.com/api/historicalOR/bulk-block-short-deals",
            "source_note": f"official_nse_{option_type}_historical_deals",
        }
    )
    return normalized.dropna(subset=["trade_date", "symbol"]).copy()


def _iter_days(start_date: date, end_date: date) -> list[date]:
    days: list[date] = []
    cursor = start_date
    while cursor <= end_date:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days
