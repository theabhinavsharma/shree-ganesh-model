from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
from src.utils.io import ensure_parent


def trade_date_dir(output_dir: Path, trade_date: date) -> Path:
    return output_dir / f"trade_date={trade_date.isoformat()}"


def market_file_name(trade_date: date, use_udiff: bool) -> str:
    if use_udiff:
        return f"sec_bhavdata_full_{trade_date.strftime('%d%m%Y')}.csv"
    return f"cm{trade_date.strftime('%d%b%Y').upper()}bhav.csv.zip"


def market_raw_file_path(output_dir: Path, trade_date: date, use_udiff: bool) -> Path:
    return trade_date_dir(output_dir, trade_date) / market_file_name(trade_date, use_udiff)


def delivery_file_name(trade_date: date) -> str:
    return f"MTO_{trade_date.strftime('%d%m%Y')}.DAT"


def delivery_raw_file_path(output_dir: Path, trade_date: date) -> Path:
    return trade_date_dir(output_dir, trade_date) / delivery_file_name(trade_date)


def manifest_file_path(output_dir: Path) -> Path:
    return output_dir / "_fetch_manifest.parquet"


def write_raw_bytes(path: Path, payload: bytes) -> None:
    ensure_parent(path)
    path.write_bytes(payload)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_fetch_manifest(output_dir: Path, records: Iterable[dict[str, object]]) -> None:
    manifest_path = manifest_file_path(output_dir)
    ensure_parent(manifest_path)
    new_df = pd.DataFrame(records)
    if manifest_path.exists():
        old_df = pd.read_parquet(manifest_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["trade_date", "artifact_type", "source_url", "status"], keep="last")
    else:
        combined = new_df
    combined.to_parquet(manifest_path, index=False)
