from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Iterable

from src.ingest.nse.io import (
    delivery_raw_file_path,
    market_raw_file_path,
    utc_now_iso,
    write_fetch_manifest,
    write_raw_bytes,
)
from src.ingest.nse.models import (
    BhavcopyFetchRequest,
    DELIVERY_URL_PATTERN,
    LEGACY_BHAVCOPY_URL_PATTERN,
    RawFetchResult,
    SourceArtifact,
    UDIFF_CUTOVER_DATE,
    UDIFF_URL_PATTERN,
)
from src.ingest.nse.session import build_session
from src.utils.dates import is_weekend, iter_calendar_dates
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_nse_bhavcopy_url(trade_date: date) -> str:
    if uses_udiff_bhavcopy(trade_date):
        return UDIFF_URL_PATTERN.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))
    month = trade_date.strftime("%b").upper()
    return LEGACY_BHAVCOPY_URL_PATTERN.format(
        dd=trade_date.strftime("%d"),
        mon=month,
        yyyy=trade_date.strftime("%Y"),
    )


def build_nse_delivery_url(trade_date: date) -> str:
    return DELIVERY_URL_PATTERN.format(ddmmyyyy=trade_date.strftime("%d%m%Y"))


def uses_udiff_bhavcopy(trade_date: date) -> bool:
    return trade_date >= UDIFF_CUTOVER_DATE


def build_source_artifacts(trade_date: date) -> list[SourceArtifact]:
    artifacts = [
        SourceArtifact(
            artifact_type="market",
            source_url=build_nse_bhavcopy_url(trade_date),
        )
    ]
    if not uses_udiff_bhavcopy(trade_date):
        artifacts.append(
            SourceArtifact(
                artifact_type="delivery",
                source_url=build_nse_delivery_url(trade_date),
            )
        )
    return artifacts


def iter_trading_fetch_dates(start_date: date, end_date: date) -> Iterable[date]:
    for current in iter_calendar_dates(start_date, end_date):
        if not is_weekend(current):
            yield current


def fetch_bhavcopy_range(request: BhavcopyFetchRequest) -> list[RawFetchResult]:
    session = build_session()
    results: list[RawFetchResult] = []
    manifest_rows: list[dict[str, object]] = []
    for trade_date in iter_trading_fetch_dates(request.start_date, request.end_date):
        for artifact in build_source_artifacts(trade_date):
            path = _artifact_output_path(request.output_dir, trade_date, artifact.artifact_type)
            if path.exists():
                result = RawFetchResult(trade_date, artifact.artifact_type, path, artifact.source_url, "skipped_existing")
                results.append(result)
                manifest_rows.append(_result_to_manifest_row(result))
                continue
            try:
                LOGGER.info("Fetching %s artifact for %s", artifact.artifact_type, trade_date)
                response = session.get(artifact.source_url, timeout=30)
                if response.status_code == 200 and response.content.strip():
                    write_raw_bytes(path, response.content)
                    result = RawFetchResult(
                        trade_date,
                        artifact.artifact_type,
                        path,
                        artifact.source_url,
                        "success",
                        http_status=response.status_code,
                    )
                else:
                    result = RawFetchResult(
                        trade_date,
                        artifact.artifact_type,
                        None,
                        artifact.source_url,
                        "error",
                        http_status=response.status_code,
                        error_message=f"status_{response.status_code}",
                    )
                results.append(result)
                manifest_rows.append(_result_to_manifest_row(result))
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Bhavcopy fetch failed for %s (%s)", trade_date, artifact.artifact_type)
                result = RawFetchResult(
                    trade_date,
                    artifact.artifact_type,
                    None,
                    artifact.source_url,
                    "error",
                    error_message=str(exc),
                )
                results.append(result)
                manifest_rows.append(_result_to_manifest_row(result))
        time.sleep(request.delay_seconds)
    if manifest_rows:
        write_fetch_manifest(request.output_dir, manifest_rows)
    return results


def _artifact_output_path(output_dir: Path, trade_date: date, artifact_type: str) -> Path:
    if artifact_type == "market":
        return market_raw_file_path(output_dir, trade_date, use_udiff=uses_udiff_bhavcopy(trade_date))
    if artifact_type == "delivery":
        return delivery_raw_file_path(output_dir, trade_date)
    raise ValueError(f"Unsupported artifact type: {artifact_type}")


def _result_to_manifest_row(result: RawFetchResult) -> dict[str, object]:
    return {
        "trade_date": result.trade_date,
        "artifact_type": result.artifact_type,
        "file_path": str(result.file_path) if result.file_path else None,
        "source_url": result.source_url,
        "status": result.status,
        "http_status": result.http_status,
        "error_message": result.error_message,
        "logged_at": utc_now_iso(),
    }
