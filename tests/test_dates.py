from datetime import date

from src.ingest.nse.fetch_bhavcopy import build_nse_bhavcopy_url, build_nse_delivery_url, iter_trading_fetch_dates


def test_weekends_skipped() -> None:
    dates = list(iter_trading_fetch_dates(date(2024, 1, 5), date(2024, 1, 8)))
    assert dates == [date(2024, 1, 5), date(2024, 1, 8)]


def test_nse_udiff_url_pattern() -> None:
    assert (
        build_nse_bhavcopy_url(date(2024, 11, 14))
        == "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_14112024.csv"
    )


def test_nse_legacy_url_pattern() -> None:
    assert (
        build_nse_bhavcopy_url(date(2015, 1, 2))
        == "https://nsearchives.nseindia.com/content/historical/EQUITIES/2015/JAN/cm02JAN2015bhav.csv.zip"
    )


def test_nse_delivery_url_pattern() -> None:
    assert build_nse_delivery_url(date(2015, 1, 2)) == "https://nsearchives.nseindia.com/archives/equities/mto/MTO_02012015.DAT"
