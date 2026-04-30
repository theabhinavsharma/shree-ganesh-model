from __future__ import annotations

from datetime import date
import io
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.ingest.nse.io import utc_now_iso

UDIFF_COLUMN_MAP = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "DATE1": "trade_date_source",
    "PREV_CLOSE": "prev_close",
    "OPEN_PRICE": "open",
    "HIGH_PRICE": "high",
    "LOW_PRICE": "low",
    "LAST_PRICE": "last_price",
    "CLOSE_PRICE": "close",
    "AVG_PRICE": "avg_price",
    "TTL_TRD_QNTY": "total_traded_qty",
    "TURNOVER_LACS": "turnover_lacs",
    "NO_OF_TRADES": "num_trades",
    "DELIV_QTY": "deliverable_qty",
    "DELIV_PER": "delivery_pct",
}

LEGACY_COLUMN_MAP = {
    "SYMBOL": "symbol",
    "SERIES": "series",
    "TIMESTAMP": "trade_date_source",
    "PREVCLOSE": "prev_close",
    "OPEN": "open",
    "HIGH": "high",
    "LOW": "low",
    "LAST": "last_price",
    "CLOSE": "close",
    "TOTTRDQTY": "total_traded_qty",
    "TOTTRDVAL": "total_traded_value",
    "TOTALTRADES": "num_trades",
    "ISIN": "isin",
}

MTO_COLUMNS = [
    "record_type",
    "record_number",
    "symbol",
    "series",
    "delivery_report_traded_qty",
    "deliverable_qty",
    "delivery_pct",
]


def read_bhavcopy_csv_text(raw_text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(raw_text))
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Unable to parse bhavcopy csv text") from exc


def normalize_bhavcopy_csv(
    path: Path,
    trade_date: date,
    source_url: str,
    delivery_path: Path | None = None,
    delivery_source_url: str | None = None,
) -> pd.DataFrame:
    raw_df = _read_nse_archive_file(path)
    raw_df.columns = [str(column).strip().upper() for column in raw_df.columns]
    normalized = raw_df.rename(columns=_column_map_for_raw_df(raw_df)).copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    if "symbol" in normalized.columns:
        normalized["symbol"] = normalized["symbol"].astype(str).str.strip()
    if "series" in normalized.columns:
        normalized["series"] = normalized["series"].astype(str).str.strip()
    normalized["trade_date"] = pd.to_datetime(trade_date)
    normalized["raw_file_name"] = path.name
    normalized["fetch_timestamp"] = utc_now_iso()
    normalized["source_url"] = source_url
    normalized["delivery_raw_file_name"] = pd.NA
    normalized["delivery_source_url"] = pd.NA
    for numeric_col in [
        "open",
        "high",
        "low",
        "last_price",
        "close",
        "avg_price",
        "prev_close",
        "total_traded_qty",
        "turnover_lacs",
        "num_trades",
        "deliverable_qty",
        "delivery_pct",
    ]:
        if numeric_col in normalized.columns:
            normalized[numeric_col] = pd.to_numeric(normalized[numeric_col], errors="coerce")
    if "total_traded_value" not in normalized.columns and "turnover_lacs" in normalized.columns:
        normalized["total_traded_value"] = normalized["turnover_lacs"] * 100000.0
    if "delivery_pct" in normalized.columns:
        normalized["delivery_pct"] = normalized["delivery_pct"] / 100.0
    if delivery_path is not None:
        normalized = _merge_delivery_data(normalized, normalize_delivery_file(delivery_path))
        normalized["delivery_raw_file_name"] = delivery_path.name
        normalized["delivery_source_url"] = delivery_source_url
    normalized["verified_price_flag"] = (
        normalized[["open", "high", "low", "close"]].notna().all(axis=1)
        if {"open", "high", "low", "close"}.issubset(normalized.columns)
        else False
    )
    return normalized


def normalize_trade_date_directory(
    trade_dir: Path,
    trade_date: date,
    market_source_url: str,
    delivery_source_url: str | None = None,
) -> pd.DataFrame:
    market_files = sorted(list(trade_dir.glob("sec_bhavdata_full_*.csv")) + list(trade_dir.glob("cm*bhav.csv.zip")))
    if not market_files:
        raise FileNotFoundError(f"No market artifact found in {trade_dir}")
    if len(market_files) > 1:
        raise ValueError(f"Multiple market artifacts found in {trade_dir}")
    delivery_files = sorted(trade_dir.glob("MTO_*.DAT"))
    if len(delivery_files) > 1:
        raise ValueError(f"Multiple delivery artifacts found in {trade_dir}")
    delivery_path = delivery_files[0] if delivery_files else None
    return normalize_bhavcopy_csv(
        market_files[0],
        trade_date,
        market_source_url,
        delivery_path=delivery_path,
        delivery_source_url=delivery_source_url,
    )


def normalize_delivery_file(path: Path) -> pd.DataFrame:
    delivery_df = pd.read_csv(
        path,
        skiprows=4,
        header=None,
        names=MTO_COLUMNS,
        usecols=range(len(MTO_COLUMNS)),
        sep=",",
    )
    delivery_df = delivery_df[delivery_df["record_type"].astype(str).str.strip().eq("20")].copy()
    delivery_df["symbol"] = delivery_df["symbol"].astype(str).str.strip()
    delivery_df["series"] = delivery_df["series"].astype(str).str.strip()
    for numeric_col in ["delivery_report_traded_qty", "deliverable_qty", "delivery_pct"]:
        delivery_df[numeric_col] = pd.to_numeric(delivery_df[numeric_col], errors="coerce")
    delivery_df["delivery_pct"] = delivery_df["delivery_pct"] / 100.0
    return delivery_df[["symbol", "series", "delivery_report_traded_qty", "deliverable_qty", "delivery_pct"]]


def _column_map_for_raw_df(raw_df: pd.DataFrame) -> dict[str, str]:
    if "DATE1" in raw_df.columns or "OPEN_PRICE" in raw_df.columns:
        return UDIFF_COLUMN_MAP
    return LEGACY_COLUMN_MAP


def _merge_delivery_data(market_df: pd.DataFrame, delivery_df: pd.DataFrame) -> pd.DataFrame:
    merged = market_df.merge(
        delivery_df,
        on=["symbol", "series"],
        how="left",
        suffixes=("", "_delivery"),
    )
    for column in ["deliverable_qty", "delivery_pct"]:
        delivery_column = f"{column}_delivery"
        if delivery_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(merged[delivery_column])
            else:
                merged[column] = merged[delivery_column]
            merged = merged.drop(columns=[delivery_column])
    return merged


def _read_nse_archive_file(path: Path) -> pd.DataFrame:
    signature = path.read_bytes()[:4]
    if signature == b"PK\x03\x04":
        return _read_zip_archive(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_excel(io.BytesIO(path.read_bytes()))


def _read_zip_archive(path: Path) -> pd.DataFrame:
    with ZipFile(path) as archive:
        member_names = [name for name in archive.namelist() if not name.endswith("/")]
        csv_members = [name for name in member_names if name.lower().endswith(".csv")]
        if csv_members:
            with archive.open(csv_members[0]) as csv_handle:
                return pd.read_csv(csv_handle)
    return pd.read_excel(io.BytesIO(path.read_bytes()))
