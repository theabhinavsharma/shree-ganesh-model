from __future__ import annotations

from requests import Session

from src.ingest.nse.api import get_json
from src.ingest.nse.models import DOCUMENT_HEADERS
from src.ingest.nse.models import HEADERS
from src.ingest.nse.session import build_session
from src.ingest.nse.session import warmup_session


class _DummyResponse:
    def __init__(self, status_code: int, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")

    def json(self) -> object:
        return self._payload


def test_build_session_sets_browser_like_headers() -> None:
    session = build_session()
    for key, value in HEADERS.items():
        assert session.headers[key] == value


def test_warmup_session_uses_document_headers_for_referer(monkeypatch) -> None:
    session = Session()
    seen: dict[str, object] = {}

    def fake_get(url: str, timeout: int, headers: dict[str, str]) -> _DummyResponse:
        seen["url"] = url
        seen["timeout"] = timeout
        seen["headers"] = headers
        return _DummyResponse(200)

    monkeypatch.setattr(session, "get", fake_get)
    referer = "https://www.nseindia.com/companies-listing/corporate-filings-actions"

    warmup_session(session, referer=referer)

    assert seen["url"] == referer
    headers = seen["headers"]
    assert headers["Accept"] == DOCUMENT_HEADERS["Accept"]
    assert headers["Sec-Fetch-Dest"] == "document"
    assert headers["Sec-Fetch-Site"] == "same-origin"
    assert headers["Referer"] == referer


def test_get_json_rewarms_and_retries_after_403(monkeypatch) -> None:
    session = Session()
    calls: list[tuple[str, dict[str, str]]] = []
    responses = iter(
        [
            _DummyResponse(403),
            _DummyResponse(200),
            _DummyResponse(200, payload={"ok": True}),
        ]
    )

    def fake_get(url: str, timeout: int, headers: dict[str, str]) -> _DummyResponse:
        calls.append((url, headers))
        return next(responses)

    monkeypatch.setattr(session, "get", fake_get)
    referer = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
    payload = get_json(
        session,
        "https://www.nseindia.com/api/corporates-corporateActions?index=equities",
        referer=referer,
    )

    assert payload == {"ok": True}
    assert len(calls) == 3
    api_headers = calls[0][1]
    warm_headers = calls[1][1]
    retry_headers = calls[2][1]
    assert api_headers["Accept"] == HEADERS["Accept"]
    assert api_headers["Referer"] == referer
    assert warm_headers["Accept"] == DOCUMENT_HEADERS["Accept"]
    assert warm_headers["Sec-Fetch-Dest"] == "document"
    assert retry_headers["Accept"] == HEADERS["Accept"]
