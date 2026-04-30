from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.report import production_weekly_run as weekly_run
from src.report.production_weekly_run import GateResult
from src.report.production_weekly_run import SourceMetrics
from src.report.production_weekly_run import evaluate_preflight_gates
from src.report.production_weekly_run import validate_portfolio_report


def test_evaluate_preflight_gates_blocks_missing_smtp_and_stale_market() -> None:
    metrics = SourceMetrics(
        as_of_trade_date="2026-03-14",
        latest_market_manifest_date="2026-03-13",
        market_age_days=5,
        announcements_max_event_date="2026-03-10",
        announcements_age_days=9,
        macro_max_trade_date="2026-03-18",
        macro_age_days=1,
        fundamentals_max_effective_date="2026-02-20",
        fundamentals_age_days=28,
        shareholding_max_effective_date="2026-03-15",
        shareholding_age_days=5,
        fundamentals_symbol_coverage=0.72,
        shareholding_symbol_coverage=0.79,
        current_universe_symbol_count=2000,
    )
    gates = evaluate_preflight_gates(
        run_ts=pd.Timestamp("2026-03-19 19:00:00+05:30"),
        metrics=metrics,
        latest_cache_status={"ok": True, "run_timestamp": "2026-03-18T18:30:00+05:30"},
        smtp_ready=False,
        smtp_message="SMTP configuration missing",
    )
    failed = {gate.name for gate in gates if not gate.passed}
    assert "market_data_fresh" in failed
    assert "announcements_fresh" in failed
    assert "smtp_ready" in failed


def test_validate_portfolio_report_catches_bad_trade_levels() -> None:
    report = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "current_price": [100.0, 200.0],
            "buy_price_low": [99.0, 198.0],
            "buy_price_high": [103.0, 206.0],
            "sell_target": [150.0, 180.0],
            "stop_loss": [90.0, 205.0],
            "confidence_score": [90.0, 80.0],
            "allocation_pct": [45.0, 45.0],
        }
    )
    gates = validate_portfolio_report(report, expected_count=2, cash_buffer_pct=10.0)
    failed = {gate.name for gate in gates if not gate.passed}
    assert "portfolio_trade_levels" in failed
    assert "portfolio_allocation_total" not in failed


def test_refresh_fundamentals_merges_incremental_without_clobbering_history(monkeypatch, tmp_path: Path) -> None:
    base_dir = tmp_path / "fundamentals"
    normalized_dir = base_dir / "normalized"
    normalized_dir.mkdir(parents=True)

    existing_all_scopes = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "statement_scope": ["Non-Consolidated"],
            "seq_number": ["1"],
            "fiscal_period_end": ["2025-09-30"],
            "announced_date": ["2025-10-15"],
            "effective_from_date": ["2025-10-15"],
            "revenue": [10.0],
            "pat": [1.0],
        }
    )
    existing_all_scopes.to_parquet(normalized_dir / "stock_quarterly_fundamentals_all_scopes.parquet", index=False)
    existing_preferred = existing_all_scopes.drop(columns=["statement_scope", "seq_number", "announced_date"]).copy()
    existing_preferred.to_parquet(normalized_dir / "stock_quarterly_fundamentals.parquet", index=False)

    incremental = pd.DataFrame(
        {
            "symbol": ["BBB"],
            "statement_scope": ["Non-Consolidated"],
            "seq_number": ["2"],
            "fiscal_period_end": ["2025-12-31"],
            "announced_date": ["2026-01-20"],
            "effective_from_date": ["2026-01-20"],
            "revenue": [20.0],
            "pat": [2.0],
        }
    )

    monkeypatch.setattr(weekly_run, "DEFAULT_FUNDAMENTALS_DIR", base_dir)
    monkeypatch.setattr(weekly_run, "DEFAULT_FUNDAMENTALS_PATH", normalized_dir / "stock_quarterly_fundamentals.parquet")

    def fake_load(config):
        temp_normalized = config.output_dir / "normalized"
        temp_normalized.mkdir(parents=True, exist_ok=True)
        incremental.to_parquet(temp_normalized / "stock_quarterly_fundamentals_all_scopes.parquet", index=False)
        incremental.drop(columns=["statement_scope", "seq_number", "announced_date"]).to_parquet(
            temp_normalized / "stock_quarterly_fundamentals.parquet",
            index=False,
        )
        return incremental.drop(columns=["statement_scope", "seq_number", "announced_date"])

    monkeypatch.setattr(weekly_run, "load_fundamentals_history_from_nse", fake_load)

    step = weekly_run._refresh_fundamentals(pd.Timestamp("2026-04-02 18:00:00+05:30"))

    merged_all_scopes = pd.read_parquet(normalized_dir / "stock_quarterly_fundamentals_all_scopes.parquet")
    merged_preferred = pd.read_parquet(normalized_dir / "stock_quarterly_fundamentals.parquet")

    assert step.success is True
    assert sorted(merged_all_scopes["symbol"].tolist()) == ["AAA", "BBB"]
    assert sorted(merged_preferred["symbol"].tolist()) == ["AAA", "BBB"]


def test_refresh_shareholding_merges_incremental_without_clobbering_history(monkeypatch, tmp_path: Path) -> None:
    base_dir = tmp_path / "shareholding"
    normalized_dir = base_dir / "normalized"
    normalized_dir.mkdir(parents=True)

    existing = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "quarter_end": ["2025-09-30"],
            "effective_from_date": ["2025-10-15"],
            "promoter_pct": [55.0],
        }
    )
    existing.to_parquet(normalized_dir / "stock_shareholding_quarterly.parquet", index=False)

    incremental = pd.DataFrame(
        {
            "symbol": ["BBB"],
            "quarter_end": ["2025-12-31"],
            "effective_from_date": ["2026-01-20"],
            "promoter_pct": [60.0],
        }
    )

    monkeypatch.setattr(weekly_run, "DEFAULT_SHAREHOLDING_DIR", base_dir)
    monkeypatch.setattr(weekly_run, "DEFAULT_SHAREHOLDING_PATH", normalized_dir / "stock_shareholding_quarterly.parquet")

    def fake_load(config):
        temp_normalized = config.output_dir / "normalized"
        temp_normalized.mkdir(parents=True, exist_ok=True)
        incremental.to_parquet(temp_normalized / "stock_shareholding_quarterly.parquet", index=False)
        return incremental

    monkeypatch.setattr(weekly_run, "load_shareholding_history_from_nse", fake_load)

    step = weekly_run._refresh_shareholding(pd.Timestamp("2026-04-02 18:00:00+05:30"))

    merged = pd.read_parquet(normalized_dir / "stock_shareholding_quarterly.parquet")

    assert step.success is True
    assert sorted(merged["symbol"].tolist()) == ["AAA", "BBB"]


def test_refresh_market_cache_rebuilds_when_existing_daily_facts_is_corrupt(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    daily_facts_path = tmp_path / "stock_daily_facts.parquet"
    daily_facts_path.parent.mkdir(parents=True, exist_ok=True)
    daily_facts_path.write_text("not-a-parquet", encoding="utf-8")

    monkeypatch.setattr(weekly_run, "DEFAULT_MARKET_RAW_DIR", raw_dir)
    monkeypatch.setattr(weekly_run, "DEFAULT_DAILY_FACTS_PATH", daily_facts_path)
    monkeypatch.setattr(weekly_run, "DEFAULT_CORPORATE_ACTIONS_PATH", tmp_path / "actions.parquet")

    class FetchResult:
        artifact_type = "market"
        status = "skipped_existing"

    monkeypatch.setattr(weekly_run, "fetch_bhavcopy_range", lambda request: [FetchResult()])

    rebuilt = pd.DataFrame({"trade_date": ["2026-04-02"], "symbol": ["AAA"]})
    monkeypatch.setattr(weekly_run, "build_stock_daily_facts", lambda *args, **kwargs: rebuilt)

    step = weekly_run._refresh_market_cache(pd.Timestamp("2026-04-02 18:00:00+05:30"))

    repaired = pd.read_parquet(daily_facts_path)
    assert step.success is True
    assert len(repaired) == 1
    assert repaired["symbol"].tolist() == ["AAA"]
