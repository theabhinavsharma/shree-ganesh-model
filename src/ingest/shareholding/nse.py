from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json, write_parquet

SHAREHOLDING_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
SHAREHOLDING_MASTER_URL = "https://www.nseindia.com/api/corporate-share-holdings-master?index=equities"
SHAREHOLDING_DETAIL_URL = "https://www.nseindia.com/api/corporate-share-holdings-equities?ndsId={record_id}&index={detail_index}"
PLEDGE_URL = "https://www.nseindia.com/api/corporate-pledgedata?symbol={symbol}"

FPI_LABELS = {
    "Foreign Portfolio Investors Category I",
    "Foreign Portfolio Investors Category II",
    "Foreign Portfolio Investor (Category - III)",
    "Foreign Institutional Investors",
}

DII_LABELS = {
    "Mutual Funds",
    "Banks",
    "Insurance Companies",
    "Provident Funds/ Pension Funds",
    "Other Financial Institutions",
    "Venture Capital Funds",
    "Alternative Investment Funds",
    "NBFCs Registered with RBI",
    "Sovereign Wealth Funds",
}


@dataclass(frozen=True)
class NseShareholdingFetchConfig:
    output_dir: Path
    symbols: set[str] | None = None
    delay_seconds: float = 0.1
    include_pledge_data: bool = True
    from_date: date | None = None
    to_date: date | None = None


def load_shareholding_from_nse(config: NseShareholdingFetchConfig) -> pd.DataFrame:
    session = build_session(warm=True, referer=SHAREHOLDING_REFERER)
    as_of_date = pd.Timestamp.utcnow().date().isoformat()
    if config.symbols:
        master_rows = _fetch_symbol_history(session, config, as_of_date)
    else:
        master_rows = get_json(session, SHAREHOLDING_MASTER_URL, referer=SHAREHOLDING_REFERER)
        write_json(master_rows, config.output_dir / "raw" / f"as_of_date={as_of_date}" / "shareholding_master.json")

    pledge_cache: dict[str, list[dict[str, object]]] = {}
    normalized_rows: list[dict[str, object]] = []

    for row in master_rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        record_id = str(row.get("recordId", "")).strip()
        if not symbol or not record_id:
            continue

        public_rows = get_json(
            session,
            SHAREHOLDING_DETAIL_URL.format(record_id=record_id, detail_index="public-shareholder"),
            referer=SHAREHOLDING_REFERER,
        )
        write_json(
            public_rows,
            config.output_dir / "raw" / f"as_of_date={as_of_date}" / "public_shareholder" / f"{record_id}.json",
        )

        if config.include_pledge_data and symbol not in pledge_cache:
            pledge_rows = get_json(session, PLEDGE_URL.format(symbol=symbol), referer=SHAREHOLDING_REFERER).get("data", [])
            pledge_cache[symbol] = pledge_rows
            write_json(
                pledge_rows,
                config.output_dir / "raw" / f"as_of_date={as_of_date}" / "pledgedata" / f"{symbol}.json",
            )

        normalized_rows.append(_normalize_shareholding_row(row, public_rows, pledge_cache.get(symbol, [])))
        time.sleep(config.delay_seconds)

    if not normalized_rows:
        return pd.DataFrame()

    df = pd.DataFrame(normalized_rows)
    df = df[df["quarter_end"].apply(_is_quarter_end)].copy()
    df = df.sort_values(["symbol", "quarter_end", "effective_from_date"]).reset_index(drop=True)
    for column in ["promoter_pct", "fii_fpi_pct", "dii_pct", "mf_pct"]:
        df[f"{column}_qoq_change"] = df.groupby("symbol")[column].diff()
    write_parquet(df, config.output_dir / "normalized" / "stock_shareholding_quarterly.parquet")
    return df


def _fetch_symbol_history(session, config: NseShareholdingFetchConfig, as_of_date: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in sorted({value.strip().upper() for value in config.symbols or set() if value}):
        url = _build_shareholding_master_url(symbol, config.from_date, config.to_date)
        symbol_rows = get_json(session, url, referer=SHAREHOLDING_REFERER)
        write_json(
            symbol_rows,
            config.output_dir / "raw" / f"as_of_date={as_of_date}" / "shareholding_master_by_symbol" / f"{symbol}.json",
        )
        rows.extend(symbol_rows)
        time.sleep(config.delay_seconds)
    return rows


def _build_shareholding_master_url(symbol: str, from_date: date | None, to_date: date | None) -> str:
    query = {"index": "equities", "symbol": symbol}
    if from_date:
        query["from_date"] = from_date.strftime("%d-%m-%Y")
    if to_date:
        query["to_date"] = to_date.strftime("%d-%m-%Y")
    return f"https://www.nseindia.com/api/corporate-share-holdings-master?{urlencode(query)}"


def _normalize_shareholding_row(
    master_row: dict[str, object],
    public_rows: list[dict[str, object]],
    pledge_rows: list[dict[str, object]],
) -> dict[str, object]:
    symbol = str(master_row.get("symbol", "")).strip().upper()
    quarter_end = _normalize_timestamp(master_row.get("date"))
    filing_date = _normalize_timestamp(master_row.get("submissionDate"))
    published_date = _normalize_timestamp(master_row.get("broadcastDate"))
    system_date = _normalize_timestamp(master_row.get("systemDate"))
    effective_from = published_date
    if pd.isna(effective_from):
        effective_from = filing_date
    if pd.isna(effective_from):
        effective_from = system_date

    promoter_pct = _to_number(master_row.get("pr_and_prgrp"))
    mf_pct = _sum_matching_public_rows(public_rows, {"Mutual Funds"})
    fii_fpi_pct = _sum_matching_public_rows(public_rows, FPI_LABELS)
    dii_pct = _sum_matching_public_rows(public_rows, DII_LABELS)
    promoter_pledged_pct = _lookup_pledged_pct(pledge_rows, quarter_end)

    return {
        "symbol": symbol,
        "quarter_end": quarter_end,
        "filing_date": filing_date,
        "effective_from_date": effective_from,
        "promoter_pct": promoter_pct,
        "promoter_pledged_pct": promoter_pledged_pct,
        "fii_fpi_pct": fii_fpi_pct,
        "dii_pct": dii_pct,
        "mf_pct": mf_pct,
    }


def _lookup_pledged_pct(pledge_rows: list[dict[str, object]], quarter_end: pd.Timestamp) -> float | None:
    if pd.isna(quarter_end):
        return None
    for row in pledge_rows:
        shp = _parse_nse_timestamp(row.get("shp")).normalize()
        if shp == quarter_end:
            return _to_number(row.get("percSharesPledged"))
    return None


def _sum_matching_public_rows(rows: list[dict[str, object]], labels: set[str]) -> float | None:
    total = 0.0
    matched = False
    for row in rows:
        label = _normalize_public_label(row)
        if label in labels:
            value = _to_number(row.get("COL_VIII"))
            if value is not None:
                total += value
                matched = True
    return total if matched else None


def _normalize_public_label(row: dict[str, object]) -> str:
    primary = _clean_label(row.get("COL_I"))
    secondary = _clean_label(row.get("COL_II"))
    if primary and primary not in {"-", " "}:
        return primary
    return secondary


def _clean_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _to_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_numeric(str(value).replace(",", "").strip(), errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _parse_nse_timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    raw = str(value).strip()
    if not raw:
        return pd.NaT
    for pattern in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return pd.Timestamp(datetime.strptime(raw, pattern))
        except ValueError:
            continue
    return pd.to_datetime(raw, errors="coerce", dayfirst=True)


def _is_quarter_end(value: pd.Timestamp) -> bool:
    if pd.isna(value):
        return False
    value = pd.Timestamp(value)
    return bool(value.is_month_end and value.month in {3, 6, 9, 12})


def _normalize_timestamp(value: object) -> pd.Timestamp:
    parsed = _parse_nse_timestamp(value)
    if pd.isna(parsed):
        return pd.NaT
    return parsed.normalize()
