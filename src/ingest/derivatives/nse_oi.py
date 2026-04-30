from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

from src.utils.io import ensure_parent
from src.utils.io import write_parquet

NCL_OI_URL = "https://nsearchives.nseindia.com/archives/nsccl/mwpl/ncloi_{ddmmyyyy}.zip"


@dataclass(frozen=True)
class NseDerivativesOiFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    trade_dates: set[date] | None = None
    delay_seconds: float = 0.02


def load_derivatives_oi_from_nse(config: NseDerivativesOiFetchConfig) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    frames: list[pd.DataFrame] = []
    for trade_date in _iter_trade_dates(config):
        raw_path = config.output_dir / "raw" / f"{trade_date.isoformat()}.zip"
        if raw_path.exists():
            payload = raw_path.read_bytes()
        else:
            try:
                payload = _download_zip(session, trade_date)
                if payload is None:
                    continue
                ensure_parent(raw_path)
                raw_path.write_bytes(payload)
            except requests.RequestException as exc:
                error_path = config.output_dir / "errors" / f"{trade_date.isoformat()}.json"
                ensure_parent(error_path)
                error_path.write_text(
                    json.dumps(
                        {
                            "trade_date": trade_date.isoformat(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        indent=2,
                        ensure_ascii=True,
                    ),
                    encoding="utf-8",
                )
                continue
        daily = _parse_daily_zip(payload, trade_date)
        if not daily.empty:
            frames.append(daily)
        time.sleep(config.delay_seconds)

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if history.empty:
        return history
    history = history.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grouped = history.groupby("symbol", group_keys=False)
    history["oi_change_1d"] = grouped["ncl_open_interest"].diff()
    previous_oi = grouped["ncl_open_interest"].shift(1)
    history["oi_change_pct_1d"] = history["oi_change_1d"] / previous_oi.abs()
    history["futeq_oi_change_1d"] = grouped["ncl_futeq_oi"].diff()
    history["oi_share_of_mwpl_change_1d"] = grouped["oi_share_of_mwpl"].diff()
    write_parquet(history, config.output_dir / "normalized" / "stock_derivatives_oi.parquet")
    return history


def _download_zip(session: requests.Session, trade_date: date) -> bytes | None:
    url = NCL_OI_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))
    response = session.get(url, timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def _parse_daily_zip(payload: bytes, trade_date: date) -> pd.DataFrame:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile:
        return pd.DataFrame()
    members = archive.namelist()
    if not members:
        return pd.DataFrame()
    with archive.open(members[0]) as handle:
        frame = pd.read_csv(handle)
    symbol_col = "NSE Symbol" if "NSE Symbol" in frame.columns else "NSE SYMBOL"
    mwpl_col = "MWPL" if "MWPL" in frame.columns else None
    oi_col = "NCL Open Interest" if "NCL Open Interest" in frame.columns else "NCL OPEN INTEREST"
    futeq_col = "NCL FutEq OI" if "NCL FutEq OI" in frame.columns else "NCL FUTEQ OI"
    if symbol_col is None or oi_col is None or mwpl_col is None:
        return pd.DataFrame()
    normalized = pd.DataFrame(
        {
            "trade_date": pd.Timestamp(trade_date),
            "symbol": frame[symbol_col].astype(str).str.strip().str.upper(),
            "mwpl": pd.to_numeric(frame[mwpl_col], errors="coerce"),
            "ncl_open_interest": pd.to_numeric(frame[oi_col], errors="coerce"),
            "ncl_futeq_oi": pd.to_numeric(frame[futeq_col], errors="coerce") if futeq_col in frame.columns else pd.Series(dtype="float64"),
            "source_url": NCL_OI_URL.format(ddmmyyyy=trade_date.strftime("%d%m%Y")),
            "source_note": "official_nse_ncl_open_interest_archive",
        }
    )
    normalized["oi_share_of_mwpl"] = normalized["ncl_open_interest"] / normalized["mwpl"]
    return normalized.dropna(subset=["symbol"]).copy()


def _iter_trade_dates(config: NseDerivativesOiFetchConfig) -> list[date]:
    if config.trade_dates:
        return sorted(config.trade_dates)
    days: list[date] = []
    cursor = config.start_date
    while cursor <= config.end_date:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days
