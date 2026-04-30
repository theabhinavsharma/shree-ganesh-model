from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote as url_quote

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

QUOTE_REFERER_TEMPLATE = "https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
QUOTE_URL_TEMPLATE = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"

SNAPSHOT_COLUMNS = [
    "symbol",
    "company_name",
    "sector",
    "industry",
    "basic_industry",
    "instrument_type",
    "issued_size",
    "quote_last_price",
    "quote_pe_ttm",
    "quote_last_update_time",
]


def build_quote_snapshot_from_symbols(
    symbols: list[str],
    *,
    output_dir: Path,
    delay_seconds: float = 0.05,
) -> pd.DataFrame:
    cleaned = sorted({symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()})
    if not cleaned:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    session = build_session(warm=True, referer=QUOTE_REFERER_TEMPLATE.format(symbol=cleaned[0]))
    as_of_date = pd.Timestamp.utcnow().date().isoformat()
    rows: list[dict[str, object]] = []
    for symbol in cleaned:
        encoded_symbol = url_quote(symbol, safe="")
        referer = QUOTE_REFERER_TEMPLATE.format(symbol=encoded_symbol)
        quote_payload = get_json(session, QUOTE_URL_TEMPLATE.format(symbol=encoded_symbol), referer=referer)
        write_json(quote_payload, output_dir / "raw" / f"as_of_date={as_of_date}" / "quote" / f"{symbol}.json")
        rows.append(_normalize_quote_snapshot_row(symbol, quote_payload))
        time.sleep(delay_seconds)
    df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    write_parquet(df, output_dir / "normalized" / "quote_snapshot.parquet")
    return df


def _normalize_quote_snapshot_row(symbol: str, quote: dict[str, object]) -> dict[str, object]:
    info = quote.get("info", {}) if isinstance(quote, dict) else {}
    industry_info = quote.get("industryInfo", {}) if isinstance(quote, dict) else {}
    security_info = quote.get("securityInfo", {}) if isinstance(quote, dict) else {}
    price_info = quote.get("priceInfo", {}) if isinstance(quote, dict) else {}
    metadata = quote.get("metadata", {}) if isinstance(quote, dict) else {}
    instrument_type = "Equity"
    if info.get("isETFSec"):
        instrument_type = "ETF"
    elif info.get("isDebtSec"):
        instrument_type = "Debt"
    return {
        "symbol": symbol,
        "company_name": info.get("companyName"),
        "sector": industry_info.get("sector"),
        "industry": industry_info.get("industry") or info.get("industry"),
        "basic_industry": industry_info.get("basicIndustry"),
        "instrument_type": instrument_type,
        "issued_size": _to_number(security_info.get("issuedSize")),
        "quote_last_price": _to_number(price_info.get("lastPrice")),
        "quote_pe_ttm": _to_number(metadata.get("pdSymbolPe")),
        "quote_last_update_time": metadata.get("lastUpdateTime"),
    }


def _to_number(value: object) -> float | None:
    if value in (None, "", "-"):
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
