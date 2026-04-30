from __future__ import annotations

import pandas as pd

from src.report.production_weekly_run import _refresh_market_cache
from src.report.production_weekly_run import _refresh_fundamentals


def test_refresh_fundamentals_rebuilds_from_full_cached_history(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load(config):  # noqa: ANN001
        calls.append("incremental_fetch")
        return pd.DataFrame({"symbol": ["AAA"], "effective_from_date": [pd.Timestamp("2026-03-20")]})

    def fake_build(**kwargs):  # noqa: ANN003
        calls.append("full_rebuild")
        assert kwargs["listing_rows"] is None
        assert kwargs["fetch_missing"] is False
        return pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "effective_from_date": [pd.Timestamp("2026-03-20"), pd.Timestamp("2025-12-31")],
            }
        )

    monkeypatch.setattr("src.report.production_weekly_run.load_fundamentals_history_from_nse", fake_load)
    monkeypatch.setattr("src.report.production_weekly_run.build_fundamentals_history_from_raw", fake_build)

    step = _refresh_fundamentals(pd.Timestamp("2026-03-21 10:00:00+05:30"))

    assert calls == ["incremental_fetch", "full_rebuild"]
    assert step.success is True
    assert step.details["rows"] == 2
    assert step.details["latest_effective_from_date"] == "2026-03-20"


def test_refresh_market_cache_fails_closed_when_no_new_market_files_and_existing_facts_are_stale(monkeypatch, tmp_path) -> None:
    facts_path = tmp_path / "stock_daily_facts_adjusted_2015plus.parquet"
    facts_path.touch()
    monkeypatch.setattr("src.report.production_weekly_run.DEFAULT_DAILY_FACTS_PATH", facts_path)
    monkeypatch.setattr(
        "src.report.production_weekly_run.fetch_bhavcopy_range",
        lambda request: [
            type(
                "Result",
                (),
                {"artifact_type": "market", "status": "error"},
            )()
        ],
    )
    monkeypatch.setattr(
        "src.report.production_weekly_run._safe_read_parquet",
        lambda path, columns=None: pd.DataFrame({"trade_date": [pd.Timestamp("2026-03-25")]}),
    )

    step = _refresh_market_cache(pd.Timestamp("2026-04-02 10:00:00+05:30"))

    assert step.success is False
    assert step.details["max_trade_date"] == "2026-03-25"
    assert step.details["market_age_days"] == 8
    assert step.details["market_error_count"] == 1


def test_refresh_market_cache_keeps_skip_success_when_existing_facts_are_still_fresh(monkeypatch, tmp_path) -> None:
    facts_path = tmp_path / "stock_daily_facts_adjusted_2015plus.parquet"
    facts_path.touch()
    monkeypatch.setattr("src.report.production_weekly_run.DEFAULT_DAILY_FACTS_PATH", facts_path)
    monkeypatch.setattr(
        "src.report.production_weekly_run.fetch_bhavcopy_range",
        lambda request: [
            type(
                "Result",
                (),
                {"artifact_type": "market", "status": "error"},
            )()
        ],
    )
    monkeypatch.setattr(
        "src.report.production_weekly_run._safe_read_parquet",
        lambda path, columns=None: pd.DataFrame({"trade_date": [pd.Timestamp("2026-04-01")]}),
    )

    step = _refresh_market_cache(pd.Timestamp("2026-04-02 10:00:00+05:30"))

    assert step.success is True
    assert step.details["max_trade_date"] == "2026-04-01"
    assert step.details["market_age_days"] == 1
    assert step.details["market_error_count"] == 1
