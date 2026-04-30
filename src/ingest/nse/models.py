from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

NSE_HOME_URL = "https://www.nseindia.com"
UDIFF_URL_PATTERN = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
LEGACY_BHAVCOPY_URL_PATTERN = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mon}/cm{dd}{mon}{yyyy}bhav.csv.zip"
)
DELIVERY_URL_PATTERN = "https://nsearchives.nseindia.com/archives/equities/mto/MTO_{ddmmyyyy}.DAT"
# Conservative compatibility cutoff verified against sampled NSE archive dates.
UDIFF_CUTOVER_DATE = date(2020, 1, 1)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}
DOCUMENT_HEADERS = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


@dataclass(frozen=True)
class BhavcopyFetchRequest:
    start_date: date
    end_date: date
    output_dir: Path
    delay_seconds: float = 0.5
    symbol_filter: set[str] | None = None


@dataclass(frozen=True)
class SourceArtifact:
    artifact_type: str
    source_url: str


@dataclass(frozen=True)
class RawFetchResult:
    trade_date: date
    artifact_type: str
    file_path: Path | None
    source_url: str
    status: str
    http_status: int | None = None
    error_message: str | None = None
