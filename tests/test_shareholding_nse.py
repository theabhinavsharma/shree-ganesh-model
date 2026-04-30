from datetime import date

from src.ingest.shareholding.nse import _build_shareholding_master_url, _is_quarter_end


def test_shareholding_history_url_uses_symbol_and_date_range() -> None:
    url = _build_shareholding_master_url("RELIANCE", date(2020, 1, 1), date(2025, 12, 31))
    assert "symbol=RELIANCE" in url
    assert "from_date=01-01-2020" in url
    assert "to_date=31-12-2025" in url


def test_quarter_end_filter_rejects_non_quarter_dates() -> None:
    assert _is_quarter_end(date(2024, 9, 30))
    assert not _is_quarter_end(date(2024, 10, 29))
