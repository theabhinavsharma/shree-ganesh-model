from __future__ import annotations

import time
from typing import Any

from requests import Session
from requests.exceptions import RequestException

from src.ingest.nse.models import HEADERS
from src.ingest.nse.session import warmup_session


def get_json(session: Session, url: str, *, referer: str | None = None, timeout: int = 30) -> Any:
    request_headers = _request_headers(session, referer=referer)
    response = _request_with_retries(session, url, request_headers=request_headers, referer=referer, timeout=timeout)
    return response.json()


def get_text(session: Session, url: str, *, referer: str | None = None, timeout: int = 30) -> str:
    request_headers = _request_headers(session, referer=referer)
    response = _request_with_retries(session, url, request_headers=request_headers, referer=referer, timeout=timeout)
    return response.text


def _request_headers(session: Session, *, referer: str | None = None) -> dict[str, str]:
    headers = {key: value for key, value in HEADERS.items()}
    if referer:
        headers["Referer"] = referer
    origin = session.headers.get("Origin")
    if origin:
        headers["Origin"] = origin
    return headers


def _request_with_retries(
    session: Session,
    url: str,
    *,
    request_headers: dict[str, str],
    referer: str | None,
    timeout: int,
    max_attempts: int = 4,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout, headers=request_headers)
            if response.status_code in {401, 403}:
                warmup_session(session, referer=referer)
                response = session.get(url, timeout=timeout, headers=request_headers)
            if response.status_code >= 500 and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            response.raise_for_status()
            return response
        except RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
            try:
                warmup_session(session, referer=referer)
            except RequestException:
                pass
    assert last_error is not None
    raise last_error
