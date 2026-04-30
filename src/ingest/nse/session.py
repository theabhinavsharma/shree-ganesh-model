from __future__ import annotations

from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.ingest.nse.models import DOCUMENT_HEADERS
from src.ingest.nse.models import HEADERS, NSE_HOME_URL


def build_session(*, warm: bool = False, referer: str | None = None) -> Session:
    session = Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    if referer:
        session.headers["Referer"] = referer
        session.headers["Origin"] = NSE_HOME_URL
    if warm:
        warmup_session(session, referer=referer)
    return session


def warmup_session(session: Session, referer: str | None = None) -> None:
    target = referer or NSE_HOME_URL
    doc_headers = dict(DOCUMENT_HEADERS)
    if referer:
        doc_headers["Referer"] = referer
        doc_headers["Sec-Fetch-Site"] = "same-origin"
    response = session.get(target, timeout=30, headers=doc_headers)
    # NSE often blocks the bare home page for scripted clients while still allowing
    # feature pages to mint the cookies needed for subsequent same-origin API calls.
    if response.status_code == 403 and target == NSE_HOME_URL:
        return
    response.raise_for_status()
