from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json, write_parquet

STOCK_MASTER_REFERER = "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE"
META_URL = "https://www.nseindia.com/api/equity-meta-info?symbol={symbol}"
QUOTE_URL = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"


EXPECTED_COLUMNS = [
    "symbol",
    "isin",
    "company_name",
    "listing_status",
    "sector",
    "industry",
    "basic_industry",
    "instrument_type",
    "benchmark_index",
]


def load_stock_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"stock_master input missing columns: {missing}")
    return df[EXPECTED_COLUMNS].copy()


def build_stock_master_from_symbols(
    symbols: list[str],
    *,
    output_dir: Path,
    delay_seconds: float = 0.1,
    industry_mapping_path: Path | None = None,
) -> pd.DataFrame:
    session = build_session(warm=True, referer=STOCK_MASTER_REFERER)
    as_of_date = pd.Timestamp.utcnow().date().isoformat()
    industry_mapping = _load_industry_mapping(industry_mapping_path)

    rows: list[dict[str, object]] = []
    for raw_symbol in sorted({symbol.strip().upper() for symbol in symbols if symbol}):
        meta = get_json(session, META_URL.format(symbol=raw_symbol), referer=STOCK_MASTER_REFERER)
        quote = get_json(session, QUOTE_URL.format(symbol=raw_symbol), referer=STOCK_MASTER_REFERER)
        write_json(meta, output_dir / "raw" / f"as_of_date={as_of_date}" / "meta" / f"{raw_symbol}.json")
        write_json(quote, output_dir / "raw" / f"as_of_date={as_of_date}" / "quote" / f"{raw_symbol}.json")
        rows.append(_normalize_stock_master_row(raw_symbol, meta, quote, industry_mapping))
        time.sleep(delay_seconds)

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    write_parquet(df, output_dir / "normalized" / "stock_master.parquet")
    return df


def _normalize_stock_master_row(
    symbol: str,
    meta: dict[str, object],
    quote: dict[str, object],
    industry_mapping: pd.DataFrame,
) -> dict[str, object]:
    quote_metadata = quote.get("metadata", {}) if isinstance(quote, dict) else {}
    mapping_row = industry_mapping.loc[industry_mapping["symbol"].eq(symbol)] if not industry_mapping.empty else pd.DataFrame()
    sector = mapping_row.iloc[0]["sector"] if not mapping_row.empty and "sector" in mapping_row.columns else pd.NA
    basic_industry = mapping_row.iloc[0]["basic_industry"] if not mapping_row.empty and "basic_industry" in mapping_row.columns else pd.NA
    benchmark_index = mapping_row.iloc[0]["benchmark_index"] if not mapping_row.empty and "benchmark_index" in mapping_row.columns else pd.NA

    listing_status = quote_metadata.get("status")
    if not listing_status:
        if meta.get("isDelisted"):
            listing_status = "Delisted"
        elif meta.get("isSuspended"):
            listing_status = "Suspended"
        else:
            listing_status = "Listed"

    instrument_type = "Equity"
    if meta.get("isETFSec"):
        instrument_type = "ETF"
    elif meta.get("isDebtSec"):
        instrument_type = "Debt"

    return {
        "symbol": symbol,
        "isin": meta.get("isin"),
        "company_name": meta.get("companyName"),
        "listing_status": listing_status,
        "sector": sector,
        "industry": meta.get("industry") or quote_metadata.get("industry"),
        "basic_industry": basic_industry,
        "instrument_type": instrument_type,
        "benchmark_index": benchmark_index,
    }


def _load_industry_mapping(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
