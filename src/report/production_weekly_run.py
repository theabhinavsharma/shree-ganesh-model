from __future__ import annotations

import argparse
import json
import smtplib
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.analysis.current_year_end_universe import build_current_year_end_universe
from src.analysis.layer_edge_study import run_layer_edge_study
from src.ingest.corporate_actions.nse import NseCorporateActionsFetchConfig
from src.ingest.corporate_actions.nse import load_corporate_actions_from_nse
from src.ingest.events.nse import NseAnnouncementFetchConfig
from src.ingest.events.nse import load_announcements_from_nse
from src.ingest.fundamentals.nse import select_preferred_statement_scope
from src.ingest.fundamentals.nse_batched import BatchedFundamentalsConfig
from src.ingest.fundamentals.nse_batched import load_fundamentals_history_from_nse
from src.ingest.macro.nse_fred import MacroFetchConfig
from src.ingest.macro.nse_fred import load_macro_history
from src.ingest.nse.fetch_bhavcopy import fetch_bhavcopy_range
from src.ingest.nse.models import BhavcopyFetchRequest
from src.ingest.shareholding.nse_batched import BatchedShareholdingConfig
from src.ingest.shareholding.nse_batched import load_shareholding_history_from_nse
from src.report.weekly_portfolio_report import build_report_html
from src.report.weekly_portfolio_report import generate_weekly_portfolio_report
from src.report.weekly_portfolio_report import get_smtp_config
from src.report.weekly_portfolio_report import send_report_email
from src.transform.build_daily_facts import build_stock_daily_facts
from src.utils.io import write_json
from src.utils.io import write_parquet

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_PRODUCTION_ROOT = Path("reports/production_weekly")
DEFAULT_MARKET_RAW_DIR = Path("data/raw/nse_full_history_official")
DEFAULT_DAILY_FACTS_PATH = Path("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
DEFAULT_CORPORATE_ACTIONS_DIR = Path("data/corporate_actions_full_history")
DEFAULT_CORPORATE_ACTIONS_PATH = DEFAULT_CORPORATE_ACTIONS_DIR / "normalized" / "stock_corporate_actions.parquet"
DEFAULT_FUNDAMENTALS_DIR = Path("data/fundamentals_full_history")
DEFAULT_FUNDAMENTALS_PATH = DEFAULT_FUNDAMENTALS_DIR / "normalized" / "stock_quarterly_fundamentals.parquet"
DEFAULT_SHAREHOLDING_DIR = Path("data/shareholding_full_history")
DEFAULT_SHAREHOLDING_PATH = DEFAULT_SHAREHOLDING_DIR / "normalized" / "stock_shareholding_quarterly.parquet"
DEFAULT_EVENTS_DIR = Path("data/events_full_history")
DEFAULT_ANNOUNCEMENTS_PATH = DEFAULT_EVENTS_DIR / "normalized" / "stock_announcements.parquet"
DEFAULT_MACRO_DIR = Path("data/macro_full_history")
DEFAULT_MACRO_PATH = DEFAULT_MACRO_DIR / "normalized" / "macro_feature_daily.parquet"
DEFAULT_FEATURE_CONFIG_PATH = Path("configs/screening.yaml")


@dataclass(frozen=True)
class RefreshStep:
    name: str
    success: bool
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class SourceMetrics:
    as_of_trade_date: str | None
    latest_market_manifest_date: str | None
    market_age_days: int | None
    announcements_max_event_date: str | None
    announcements_age_days: int | None
    macro_max_trade_date: str | None
    macro_age_days: int | None
    fundamentals_max_effective_date: str | None
    fundamentals_age_days: int | None
    shareholding_max_effective_date: str | None
    shareholding_age_days: int | None
    fundamentals_symbol_coverage: float
    shareholding_symbol_coverage: float
    current_universe_symbol_count: int


def _now_ist(override: str | None = None) -> pd.Timestamp:
    if override:
        return pd.Timestamp(override, tz=IST)
    return pd.Timestamp(datetime.now(IST))


def _date_only(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize().date().isoformat()


def _safe_read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, columns=columns)


def _merge_unique_frames(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    *,
    key_cols: list[str],
    sort_cols: list[str],
) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    elif new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True, sort=False)
    if combined.empty:
        return combined
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined.sort_values(sort_cols).reset_index(drop=True)


def _refresh_market_cache(run_ts: pd.Timestamp, *, force_rebuild: bool = False) -> RefreshStep:
    try:
        start_date = max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=10)).date())
        fetch_results = fetch_bhavcopy_range(
            BhavcopyFetchRequest(
                start_date=start_date,
                end_date=run_ts.date(),
                output_dir=DEFAULT_MARKET_RAW_DIR,
                delay_seconds=0.2,
            )
        )
        new_market_successes = sum(1 for result in fetch_results if result.artifact_type == "market" and result.status == "success")
        market_errors = sum(1 for result in fetch_results if result.artifact_type == "market" and result.status == "error")
        if DEFAULT_DAILY_FACTS_PATH.exists() and not force_rebuild and new_market_successes == 0:
            try:
                existing = _safe_read_parquet(DEFAULT_DAILY_FACTS_PATH, columns=["trade_date"])
                max_trade_ts = pd.to_datetime(existing["trade_date"]).max() if not existing.empty else pd.NaT
                max_trade_date = _date_only(max_trade_ts) if not pd.isna(max_trade_ts) else None
                market_age_days = int((pd.Timestamp(run_ts.date()) - pd.Timestamp(max_trade_ts).normalize()).days) if not pd.isna(max_trade_ts) else None
                cache_fresh = market_age_days is not None and market_age_days <= 3
                return RefreshStep(
                    name="market_daily_facts",
                    success=cache_fresh,
                    message=(
                        "skipped daily facts rebuild because no new market file arrived"
                        if cache_fresh
                        else "market refresh failed closed because no new market file arrived and carried-forward daily facts are stale"
                    ),
                    details={
                        "rows": None,
                        "max_trade_date": max_trade_date,
                        "market_age_days": market_age_days,
                        "new_market_successes": new_market_successes,
                        "market_error_count": market_errors,
                    },
                )
            except Exception:
                # Preserve fail-closed semantics for freshness, but repair corrupted cache files by
                # rebuilding from the official raw archive instead of surfacing a stale local parquet.
                force_rebuild = True
        facts = build_stock_daily_facts(
            DEFAULT_MARKET_RAW_DIR,
            corporate_actions_path=DEFAULT_CORPORATE_ACTIONS_PATH,
            use_adjusted_prices=True,
        )
        write_parquet(facts, DEFAULT_DAILY_FACTS_PATH)
        max_trade_date = _date_only(pd.to_datetime(facts["trade_date"]).max()) if not facts.empty else None
        return RefreshStep(
            name="market_daily_facts",
            success=not facts.empty,
            message="rebuilt adjusted daily facts",
            details={
                "rows": int(len(facts)),
                "max_trade_date": max_trade_date,
                "new_market_successes": new_market_successes,
                "market_error_count": market_errors,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="market_daily_facts",
            success=False,
            message=f"market refresh failed: {exc}",
            details={},
        )


def _refresh_corporate_actions(run_ts: pd.Timestamp, *, full_backfill: bool = False) -> RefreshStep:
    try:
        start_date = date(2015, 1, 1) if full_backfill or not DEFAULT_CORPORATE_ACTIONS_PATH.exists() else max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=400)).date())
        temp_dir = DEFAULT_CORPORATE_ACTIONS_DIR / "_incremental"
        actions = load_corporate_actions_from_nse(
            NseCorporateActionsFetchConfig(
                output_dir=temp_dir,
                start_date=start_date,
                end_date=run_ts.date(),
            )
        )
        existing = _safe_read_parquet(DEFAULT_CORPORATE_ACTIONS_PATH)
        merged = _merge_unique_frames(
            existing,
            actions,
            key_cols=["symbol", "ex_date", "subject"],
            sort_cols=["symbol", "ex_date", "subject"],
        )
        if not merged.empty:
            write_parquet(merged, DEFAULT_CORPORATE_ACTIONS_PATH)
        latest_ex_date = _date_only(pd.to_datetime(merged["ex_date"]).max()) if not merged.empty else None
        existing_rows = int(len(existing))
        merged_rows = int(len(merged))
        return RefreshStep(
            name="corporate_actions",
            success=not merged.empty,
            message="merged corporate actions cache",
            details={"rows": merged_rows, "latest_ex_date": latest_ex_date, "new_rows_added": max(0, merged_rows - existing_rows)},
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="corporate_actions",
            success=False,
            message=f"corporate actions refresh failed: {exc}",
            details={},
        )


def _refresh_announcements(run_ts: pd.Timestamp, *, full_backfill: bool = False) -> RefreshStep:
    try:
        start_date = date(2015, 1, 1) if full_backfill or not DEFAULT_ANNOUNCEMENTS_PATH.exists() else max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=45)).date())
        temp_dir = DEFAULT_EVENTS_DIR / "_incremental"
        announcements = load_announcements_from_nse(
            NseAnnouncementFetchConfig(
                output_dir=temp_dir,
                start_date=start_date,
                end_date=run_ts.date(),
            )
        )
        existing = _safe_read_parquet(DEFAULT_ANNOUNCEMENTS_PATH)
        merged = _merge_unique_frames(
            existing,
            announcements,
            key_cols=["symbol", "event_date", "sequence_id"],
            sort_cols=["event_date", "symbol", "sequence_id"],
        )
        if not merged.empty:
            write_parquet(merged, DEFAULT_ANNOUNCEMENTS_PATH)
        latest_event_date = _date_only(pd.to_datetime(merged["event_date"]).max()) if not merged.empty else None
        return RefreshStep(
            name="announcements",
            success=not merged.empty,
            message="merged announcements cache",
            details={"rows": int(len(merged)), "latest_event_date": latest_event_date},
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="announcements",
            success=False,
            message=f"announcement refresh failed: {exc}",
            details={},
        )


def _refresh_macro(run_ts: pd.Timestamp, *, full_backfill: bool = False) -> RefreshStep:
    try:
        start_date = date(2015, 1, 1) if full_backfill or not DEFAULT_MACRO_PATH.exists() else max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=120)).date())
        temp_dir = DEFAULT_MACRO_DIR / "_incremental"
        _, macro_daily = load_macro_history(
            MacroFetchConfig(
                output_dir=temp_dir,
                start_date=start_date,
                end_date=run_ts.date(),
            )
        )
        existing = _safe_read_parquet(DEFAULT_MACRO_PATH)
        merged = _merge_unique_frames(existing, macro_daily, key_cols=["trade_date"], sort_cols=["trade_date"])
        if not merged.empty:
            write_parquet(merged, DEFAULT_MACRO_PATH)
        latest_trade_date = _date_only(pd.to_datetime(merged["trade_date"]).max()) if not merged.empty else None
        return RefreshStep(
            name="macro",
            success=not merged.empty,
            message="merged macro cache",
            details={"rows": int(len(merged)), "latest_trade_date": latest_trade_date},
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="macro",
            success=False,
            message=f"macro refresh failed: {exc}",
            details={},
        )


def _refresh_fundamentals(run_ts: pd.Timestamp) -> RefreshStep:
    try:
        temp_dir = DEFAULT_FUNDAMENTALS_DIR / "_incremental"
        load_fundamentals_history_from_nse(
            BatchedFundamentalsConfig(
                output_dir=temp_dir,
                start_date=max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=220)).date()),
                end_date=run_ts.date(),
                statement_scope=None,
                max_workers=4,
                delay_seconds=0.0,
                batch_timeout_seconds=900.0,
            )
        )
        incremental_all_scopes_path = temp_dir / "normalized" / "stock_quarterly_fundamentals_all_scopes.parquet"
        incremental_all_scopes = _safe_read_parquet(incremental_all_scopes_path)
        existing_all_scopes = _safe_read_parquet(
            DEFAULT_FUNDAMENTALS_DIR / "normalized" / "stock_quarterly_fundamentals_all_scopes.parquet"
        )
        merged_all_scopes = _merge_unique_frames(
            existing_all_scopes,
            incremental_all_scopes,
            key_cols=["symbol", "statement_scope", "fiscal_period_end", "effective_from_date"],
            sort_cols=["symbol", "statement_scope", "fiscal_period_end", "effective_from_date"],
        )
        if not merged_all_scopes.empty:
            write_parquet(
                merged_all_scopes,
                DEFAULT_FUNDAMENTALS_DIR / "normalized" / "stock_quarterly_fundamentals_all_scopes.parquet",
            )
        preferred = select_preferred_statement_scope(merged_all_scopes) if not merged_all_scopes.empty else pd.DataFrame()
        if not preferred.empty:
            write_parquet(preferred, DEFAULT_FUNDAMENTALS_PATH)
        latest_effective = _date_only(pd.to_datetime(preferred["effective_from_date"]).max()) if not preferred.empty else None
        return RefreshStep(
            name="fundamentals",
            success=not preferred.empty,
            message="merged recent fundamentals fetches into full cached history",
            details={"rows": int(len(preferred)), "latest_effective_from_date": latest_effective},
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="fundamentals",
            success=False,
            message=f"fundamentals refresh failed: {exc}",
            details={},
        )


def _refresh_shareholding(run_ts: pd.Timestamp) -> RefreshStep:
    try:
        temp_dir = DEFAULT_SHAREHOLDING_DIR / "_incremental"
        load_shareholding_history_from_nse(
            BatchedShareholdingConfig(
                output_dir=temp_dir,
                start_date=max(date(2015, 1, 1), (run_ts - pd.Timedelta(days=220)).date()),
                end_date=run_ts.date(),
                max_workers=4,
                listing_window_months=1,
                delay_seconds=0.0,
                batch_timeout_seconds=900.0,
            )
        )
        incremental = _safe_read_parquet(temp_dir / "normalized" / "stock_shareholding_quarterly.parquet")
        existing = _safe_read_parquet(DEFAULT_SHAREHOLDING_PATH)
        df = _merge_unique_frames(
            existing,
            incremental,
            key_cols=["symbol", "quarter_end", "effective_from_date"],
            sort_cols=["symbol", "quarter_end", "effective_from_date"],
        )
        if not df.empty:
            write_parquet(df, DEFAULT_SHAREHOLDING_PATH)
        latest_effective = _date_only(pd.to_datetime(df["effective_from_date"]).max()) if not df.empty else None
        return RefreshStep(
            name="shareholding",
            success=not df.empty,
            message="merged recent shareholding fetches into full cached history",
            details={"rows": int(len(df)), "latest_effective_from_date": latest_effective},
        )
    except Exception as exc:  # noqa: BLE001
        return RefreshStep(
            name="shareholding",
            success=False,
            message=f"shareholding refresh failed: {exc}",
            details={},
        )


def _build_source_metrics(run_ts: pd.Timestamp) -> SourceMetrics:
    daily_facts = _safe_read_parquet(DEFAULT_DAILY_FACTS_PATH, columns=["trade_date", "symbol"])
    market_manifest = _safe_read_parquet(DEFAULT_MARKET_RAW_DIR / "_fetch_manifest.parquet")
    announcements = _safe_read_parquet(DEFAULT_ANNOUNCEMENTS_PATH, columns=["symbol", "event_date"])
    macro = _safe_read_parquet(DEFAULT_MACRO_PATH, columns=["trade_date"])
    fundamentals = _safe_read_parquet(DEFAULT_FUNDAMENTALS_PATH, columns=["symbol", "effective_from_date"])
    shareholding = _safe_read_parquet(DEFAULT_SHAREHOLDING_PATH, columns=["symbol", "effective_from_date"])

    as_of_trade_date = pd.to_datetime(daily_facts["trade_date"], errors="coerce").max() if not daily_facts.empty else pd.NaT
    if not daily_facts.empty and not pd.isna(as_of_trade_date):
        latest_symbols = set(
            daily_facts.loc[pd.to_datetime(daily_facts["trade_date"]).dt.normalize().eq(as_of_trade_date.normalize()), "symbol"]
            .astype(str)
            .str.upper()
        )
    else:
        latest_symbols = set()

    manifest_market = market_manifest[market_manifest.get("artifact_type", pd.Series(dtype="object")).astype(str).eq("market")].copy()
    manifest_market = manifest_market[manifest_market.get("status", pd.Series(dtype="object")).astype(str).eq("success")].copy()
    latest_manifest_market = pd.to_datetime(manifest_market["trade_date"], errors="coerce").max() if not manifest_market.empty else pd.NaT

    announcements_max = pd.to_datetime(announcements["event_date"], errors="coerce").max() if not announcements.empty else pd.NaT
    macro_max = pd.to_datetime(macro["trade_date"], errors="coerce").max() if not macro.empty else pd.NaT
    fundamentals_effective = pd.to_datetime(fundamentals["effective_from_date"], errors="coerce") if not fundamentals.empty else pd.Series(dtype="datetime64[ns]")
    fundamentals_max = fundamentals_effective.max() if not fundamentals.empty else pd.NaT
    shareholding_effective = pd.to_datetime(shareholding["effective_from_date"], errors="coerce") if not shareholding.empty else pd.Series(dtype="datetime64[ns]")
    shareholding_max = shareholding_effective.max() if not shareholding.empty else pd.NaT

    run_date = pd.Timestamp(run_ts.date())

    fund_symbols = set(
        fundamentals.loc[fundamentals_effective.le(run_date), "symbol"].astype(str).str.upper()
    ) if not fundamentals.empty else set()
    share_symbols = set(
        shareholding.loc[shareholding_effective.le(run_date), "symbol"].astype(str).str.upper()
    ) if not shareholding.empty else set()

    current_count = len(latest_symbols)
    fundamentals_coverage = len(latest_symbols & fund_symbols) / current_count if current_count else 0.0
    shareholding_coverage = len(latest_symbols & share_symbols) / current_count if current_count else 0.0

    def age_days(ts: pd.Timestamp | Any) -> int | None:
        if ts is None or pd.isna(ts):
            return None
        return int((run_date - pd.Timestamp(ts).normalize()).days)

    return SourceMetrics(
        as_of_trade_date=_date_only(as_of_trade_date),
        latest_market_manifest_date=_date_only(latest_manifest_market),
        market_age_days=age_days(as_of_trade_date),
        announcements_max_event_date=_date_only(announcements_max),
        announcements_age_days=age_days(announcements_max),
        macro_max_trade_date=_date_only(macro_max),
        macro_age_days=age_days(macro_max),
        fundamentals_max_effective_date=_date_only(fundamentals_max),
        fundamentals_age_days=age_days(fundamentals_max),
        shareholding_max_effective_date=_date_only(shareholding_max),
        shareholding_age_days=age_days(shareholding_max),
        fundamentals_symbol_coverage=float(fundamentals_coverage),
        shareholding_symbol_coverage=float(shareholding_coverage),
        current_universe_symbol_count=current_count,
    )


def evaluate_preflight_gates(
    *,
    run_ts: pd.Timestamp,
    metrics: SourceMetrics,
    latest_cache_status: dict[str, Any] | None,
    smtp_ready: bool,
    smtp_message: str,
    min_fundamentals_coverage: float = 0.70,
    min_shareholding_coverage: float = 0.75,
) -> list[GateResult]:
    gates: list[GateResult] = []
    run_date = run_ts.normalize().date()

    cache_ok = bool(latest_cache_status and latest_cache_status.get("ok"))
    cache_age_hours = None
    if latest_cache_status and latest_cache_status.get("run_timestamp"):
        cache_age = run_ts - pd.Timestamp(latest_cache_status["run_timestamp"])
        cache_age_hours = round(cache_age.total_seconds() / 3600, 2)
        cache_ok = cache_ok and cache_age <= pd.Timedelta(hours=36)
    gates.append(
        GateResult(
            name="recent_cache_success",
            passed=cache_ok,
            message="recent cache refresh available" if cache_ok else "cache refresh missing or older than 36 hours",
            details={"cache_age_hours": cache_age_hours},
        )
    )

    market_consistent = metrics.as_of_trade_date is not None and metrics.as_of_trade_date == metrics.latest_market_manifest_date
    market_fresh = metrics.market_age_days is not None and metrics.market_age_days <= 3
    gates.append(
        GateResult(
            name="market_data_fresh",
            passed=market_consistent and market_fresh,
            message="market data fresh and manifest-consistent" if (market_consistent and market_fresh) else "market data stale or manifest mismatch",
            details=asdict(metrics),
        )
    )

    announcements_fresh = metrics.announcements_age_days is not None and metrics.announcements_age_days <= 3
    gates.append(
        GateResult(
            name="announcements_fresh",
            passed=announcements_fresh,
            message="announcements fresh" if announcements_fresh else "announcements older than 3 days",
            details={"announcements_max_event_date": metrics.announcements_max_event_date, "announcements_age_days": metrics.announcements_age_days},
        )
    )

    macro_fresh = metrics.macro_age_days is not None and metrics.macro_age_days <= 3
    gates.append(
        GateResult(
            name="macro_fresh",
            passed=macro_fresh,
            message="macro data fresh" if macro_fresh else "macro data older than 3 days",
            details={"macro_max_trade_date": metrics.macro_max_trade_date, "macro_age_days": metrics.macro_age_days},
        )
    )

    fundamentals_fresh = metrics.fundamentals_age_days is not None and metrics.fundamentals_age_days <= 180
    gates.append(
        GateResult(
            name="fundamentals_fresh",
            passed=fundamentals_fresh,
            message="fundamentals freshness acceptable" if fundamentals_fresh else "fundamentals older than 180 days",
            details={"fundamentals_max_effective_date": metrics.fundamentals_max_effective_date, "fundamentals_age_days": metrics.fundamentals_age_days},
        )
    )

    shareholding_fresh = metrics.shareholding_age_days is not None and metrics.shareholding_age_days <= 180
    gates.append(
        GateResult(
            name="shareholding_fresh",
            passed=shareholding_fresh,
            message="shareholding freshness acceptable" if shareholding_fresh else "shareholding older than 180 days",
            details={"shareholding_max_effective_date": metrics.shareholding_max_effective_date, "shareholding_age_days": metrics.shareholding_age_days},
        )
    )

    gates.append(
        GateResult(
            name="fundamentals_coverage",
            passed=metrics.fundamentals_symbol_coverage >= min_fundamentals_coverage,
            message="fundamentals coverage acceptable" if metrics.fundamentals_symbol_coverage >= min_fundamentals_coverage else "fundamentals symbol coverage below threshold",
            details={"coverage": metrics.fundamentals_symbol_coverage, "threshold": min_fundamentals_coverage, "current_universe_symbol_count": metrics.current_universe_symbol_count},
        )
    )
    gates.append(
        GateResult(
            name="shareholding_coverage",
            passed=metrics.shareholding_symbol_coverage >= min_shareholding_coverage,
            message="shareholding coverage acceptable" if metrics.shareholding_symbol_coverage >= min_shareholding_coverage else "shareholding symbol coverage below threshold",
            details={"coverage": metrics.shareholding_symbol_coverage, "threshold": min_shareholding_coverage, "current_universe_symbol_count": metrics.current_universe_symbol_count},
        )
    )
    gates.append(
        GateResult(
            name="smtp_ready",
            passed=smtp_ready,
            message=smtp_message,
            details={},
        )
    )

    if metrics.as_of_trade_date:
        as_of_trade = date.fromisoformat(metrics.as_of_trade_date)
        remaining_days = (date(as_of_trade.year, 12, 31) - as_of_trade).days
        positive_horizon = remaining_days > 0
    else:
        remaining_days = None
        positive_horizon = False
    gates.append(
        GateResult(
            name="positive_remaining_horizon",
            passed=positive_horizon,
            message="positive year-end horizon available" if positive_horizon else "no positive remaining horizon to year-end",
            details={"remaining_days": remaining_days, "run_date": run_date.isoformat()},
        )
    )
    return gates


def verify_smtp_health(*, require_smtp: bool = True) -> tuple[bool, str]:
    smtp = get_smtp_config()
    if smtp is None:
        return (False, "SMTP configuration missing") if require_smtp else (False, "SMTP configuration missing")
    try:
        if smtp["ssl"]:
            with smtplib.SMTP_SSL(smtp["host"], int(smtp["port"]), timeout=20) as client:
                client.ehlo()
                client.login(smtp["user"], smtp["password"])
        else:
            with smtplib.SMTP(smtp["host"], int(smtp["port"]), timeout=20) as client:
                client.ehlo()
                client.starttls()
                client.ehlo()
                client.login(smtp["user"], smtp["password"])
        return True, "SMTP login check passed"
    except Exception as exc:  # noqa: BLE001
        return False, f"SMTP health check failed: {exc}"


def validate_portfolio_report(
    report: pd.DataFrame,
    *,
    expected_count: int,
    cash_buffer_pct: float,
) -> list[GateResult]:
    gates: list[GateResult] = []
    gates.append(
        GateResult(
            name="portfolio_nonempty",
            passed=len(report) == expected_count,
            message="portfolio size matches target" if len(report) == expected_count else "portfolio size mismatch",
            details={"expected_count": expected_count, "actual_count": int(len(report))},
        )
    )
    duplicate_free = not report["symbol"].duplicated().any()
    gates.append(
        GateResult(
            name="portfolio_unique_symbols",
            passed=duplicate_free,
            message="portfolio symbols unique" if duplicate_free else "duplicate symbols present",
            details={},
        )
    )
    required_cols = ["current_price", "buy_price_low", "buy_price_high", "sell_target", "stop_loss", "confidence_score", "allocation_pct"]
    no_missing = report[required_cols].notna().all().all() if not report.empty else False
    gates.append(
        GateResult(
            name="portfolio_required_fields",
            passed=no_missing,
            message="portfolio required fields populated" if no_missing else "portfolio contains missing required fields",
            details={},
        )
    )
    investable_pct = round(100.0 - cash_buffer_pct, 2)
    allocation_total = round(float(pd.to_numeric(report["allocation_pct"], errors="coerce").sum()), 2) if not report.empty else 0.0
    gates.append(
        GateResult(
            name="portfolio_allocation_total",
            passed=abs(allocation_total - investable_pct) <= 0.1,
            message="allocation total matches investable capital" if abs(allocation_total - investable_pct) <= 0.1 else "allocation total mismatch",
            details={"allocation_total": allocation_total, "expected_investable_pct": investable_pct},
        )
    )
    trade_shape_ok = (
        pd.to_numeric(report["sell_target"], errors="coerce").gt(pd.to_numeric(report["buy_price_high"], errors="coerce")).all()
        and pd.to_numeric(report["stop_loss"], errors="coerce").lt(pd.to_numeric(report["buy_price_low"], errors="coerce")).all()
    ) if not report.empty else False
    gates.append(
        GateResult(
            name="portfolio_trade_levels",
            passed=trade_shape_ok,
            message="target and stop levels ordered correctly" if trade_shape_ok else "target or stop levels invalid",
            details={},
        )
    )
    return gates


def _load_latest_cache_status(state_dir: Path) -> dict[str, Any] | None:
    path = state_dir / "latest_cache_status.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_actions(current: pd.DataFrame, previous_success_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    report = current.copy()
    if not previous_success_path.exists():
        report["action"] = "New Buy"
        return report, pd.DataFrame(columns=["symbol", "stock_name", "action"])
    previous = pd.read_csv(previous_success_path)
    previous_symbols = set(previous.get("symbol", pd.Series(dtype="object")).astype(str))
    current_symbols = set(report.get("symbol", pd.Series(dtype="object")).astype(str))
    report["action"] = np.where(report["symbol"].astype(str).isin(previous_symbols), "Hold", "New Buy")
    exits = previous[~previous["symbol"].astype(str).isin(current_symbols)].copy()
    if exits.empty:
        exits = pd.DataFrame(columns=["symbol", "stock_name", "action"])
    else:
        exits["action"] = "Exit"
        keep = [column for column in ["symbol", "stock_name", "action"] if column in exits.columns]
        exits = exits[keep]
    return report, exits


def _blocked_html(run_ts: pd.Timestamp, gates: list[GateResult]) -> str:
    failed = [gate for gate in gates if not gate.passed]
    rows = "".join(
        f"<tr><td>{gate.name}</td><td>{gate.message}</td><td><pre>{json.dumps(gate.details, indent=2)}</pre></td></tr>"
        for gate in failed
    )
    return (
        "<html><body>"
        f"<p><strong>Weekly NSE portfolio run blocked</strong></p><p>Run timestamp: {run_ts.isoformat()}</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Gate</th><th>Message</th><th>Details</th></tr>"
        f"{rows}</table></body></html>"
    )


def _success_html(run_ts: pd.Timestamp, as_of_trade_date: str, report: pd.DataFrame, *, cash_buffer_pct: float) -> str:
    header = (
        f"<p><strong>Weekly NSE year-end portfolio</strong></p>"
        f"<p>Run timestamp: {run_ts.isoformat()}<br>"
        f"Data as of: {as_of_trade_date}<br>"
        f"Objective: 50%+ by {date.fromisoformat(as_of_trade_date).year}-12-31<br>"
        f"Cash buffer: {cash_buffer_pct:.1f}%</p>"
    )
    body = build_report_html(
        report,
        objective=f"50%+ by {date.fromisoformat(as_of_trade_date).year}-12-31",
        run_date=run_ts.date().isoformat(),
        cash_buffer_pct=cash_buffer_pct,
    )
    return body.replace("<body>", f"<body>{header}", 1)


def run_cache_mode(*, run_ts: pd.Timestamp, state_dir: Path) -> dict[str, Any]:
    corporate_actions_step = _refresh_corporate_actions(run_ts)
    refresh_steps = [
        corporate_actions_step,
        _refresh_market_cache(run_ts, force_rebuild=bool(corporate_actions_step.details.get("new_rows_added", 0))),
        _refresh_announcements(run_ts),
        _refresh_macro(run_ts),
        _refresh_fundamentals(run_ts),
        _refresh_shareholding(run_ts),
    ]
    ok = all(step.success for step in refresh_steps)
    payload = {
        "mode": "cache",
        "ok": ok,
        "run_timestamp": run_ts.isoformat(),
        "refresh_steps": [asdict(step) for step in refresh_steps],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    write_json(payload, state_dir / "latest_cache_status.json")
    return payload


def _train_end_date_for(as_of_trade_date: date, analysis_end_date: date) -> date:
    candidate = date(as_of_trade_date.year - 3, 12, 31)
    if candidate >= analysis_end_date:
        span_days = max((analysis_end_date - date(2015, 1, 1)).days, 1)
        return date(2015, 1, 1) + timedelta(days=int(span_days * 0.75))
    return candidate


def run_final_mode(
    *,
    run_ts: pd.Timestamp,
    output_root: Path,
    recipients: list[str],
    portfolio_size: int,
    cash_buffer_pct: float,
) -> dict[str, Any]:
    state_dir = output_root / "state"
    runs_dir = output_root / "runs"
    run_id = run_ts.strftime("%Y%m%dT%H%M%S")
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    corporate_actions_step = _refresh_corporate_actions(run_ts)
    light_refresh_steps = [
        corporate_actions_step,
        _refresh_market_cache(run_ts, force_rebuild=bool(corporate_actions_step.details.get("new_rows_added", 0))),
        _refresh_announcements(run_ts),
        _refresh_macro(run_ts),
    ]
    cache_status = _load_latest_cache_status(state_dir)
    smtp_ready, smtp_message = verify_smtp_health(require_smtp=True)
    metrics = _build_source_metrics(run_ts)
    preflight_gates = evaluate_preflight_gates(
        run_ts=run_ts,
        metrics=metrics,
        latest_cache_status=cache_status,
        smtp_ready=smtp_ready,
        smtp_message=smtp_message,
    )

    failed_preflight = [gate for gate in preflight_gates if not gate.passed]
    if failed_preflight:
        blocked_csv = run_dir / "blocked_status.csv"
        blocked_html = run_dir / "blocked_status.html"
        pd.DataFrame([asdict(gate) for gate in preflight_gates]).to_csv(blocked_csv, index=False)
        blocked_html.write_text(_blocked_html(run_ts, preflight_gates), encoding="utf-8")
        if smtp_ready:
            send_report_email(
                recipients=recipients,
                subject=f"BLOCKED - Weekly NSE portfolio run - {run_ts.date().isoformat()}",
                html_body=blocked_html.read_text(encoding="utf-8"),
                csv_path=blocked_csv,
                allow_sendmail_fallback=False,
            )
        payload = {
            "mode": "final",
            "ok": False,
            "blocked": True,
            "run_timestamp": run_ts.isoformat(),
            "refresh_steps": [asdict(step) for step in light_refresh_steps],
            "gates": [asdict(gate) for gate in preflight_gates],
            "artifacts": {"blocked_csv": str(blocked_csv), "blocked_html": str(blocked_html)},
        }
        write_json(payload, state_dir / "latest_final_status.json")
        return payload

    assert metrics.as_of_trade_date is not None
    as_of_trade_date = date.fromisoformat(metrics.as_of_trade_date)
    target_date = date(as_of_trade_date.year, 12, 31)
    remaining_days = (target_date - as_of_trade_date).days
    analysis_end_date = as_of_trade_date - timedelta(days=remaining_days)
    train_end_date = _train_end_date_for(as_of_trade_date, analysis_end_date)

    study_dir = run_dir / "exact_year_end_study"
    study = run_layer_edge_study(
        raw_dir=DEFAULT_MARKET_RAW_DIR,
        config_path=DEFAULT_FEATURE_CONFIG_PATH,
        analysis_start_date=date(2015, 1, 1),
        analysis_end_date=analysis_end_date,
        train_end_date=train_end_date,
        horizon_days=remaining_days,
        target_return=0.5,
        output_dir=study_dir,
        fundamentals_path=DEFAULT_FUNDAMENTALS_PATH if DEFAULT_FUNDAMENTALS_PATH.exists() else None,
        shareholding_path=DEFAULT_SHAREHOLDING_PATH if DEFAULT_SHAREHOLDING_PATH.exists() else None,
        macro_daily_path=DEFAULT_MACRO_PATH if DEFAULT_MACRO_PATH.exists() else None,
        announcements_path=DEFAULT_ANNOUNCEMENTS_PATH if DEFAULT_ANNOUNCEMENTS_PATH.exists() else None,
        event_daily_path=None,
        daily_facts_path=DEFAULT_DAILY_FACTS_PATH if DEFAULT_DAILY_FACTS_PATH.exists() else None,
        min_price=20.0,
    )

    study_summary = study.get("summary", {})
    study_gate = GateResult(
        name="exact_year_end_study",
        passed=int(study_summary.get("horizon_days", -1)) == remaining_days and float(study_summary.get("target_return", -1)) == 0.5,
        message="exact year-end study reran with correct horizon" if int(study_summary.get("horizon_days", -1)) == remaining_days and float(study_summary.get("target_return", -1)) == 0.5 else "exact year-end study output mismatch",
        details=study_summary,
    )

    if not study_gate.passed:
        blocked_csv = run_dir / "blocked_status.csv"
        blocked_html = run_dir / "blocked_status.html"
        all_gates = preflight_gates + [study_gate]
        pd.DataFrame([asdict(gate) for gate in all_gates]).to_csv(blocked_csv, index=False)
        blocked_html.write_text(_blocked_html(run_ts, all_gates), encoding="utf-8")
        if smtp_ready:
            send_report_email(
                recipients=recipients,
                subject=f"BLOCKED - Weekly NSE portfolio run - {run_ts.date().isoformat()}",
                html_body=blocked_html.read_text(encoding="utf-8"),
                csv_path=blocked_csv,
                allow_sendmail_fallback=False,
            )
        payload = {
            "mode": "final",
            "ok": False,
            "blocked": True,
            "run_timestamp": run_ts.isoformat(),
            "refresh_steps": [asdict(step) for step in light_refresh_steps],
            "gates": [asdict(gate) for gate in all_gates],
            "artifacts": {"blocked_csv": str(blocked_csv), "blocked_html": str(blocked_html)},
        }
        write_json(payload, state_dir / "latest_final_status.json")
        return payload

    current_universe_dir = run_dir / "current_universe"
    build_current_year_end_universe(
        daily_facts_path=DEFAULT_DAILY_FACTS_PATH,
        feature_results_path=study_dir / "layer_feature_results.csv",
        output_dir=current_universe_dir,
        config_path=DEFAULT_FEATURE_CONFIG_PATH,
        fundamentals_path=DEFAULT_FUNDAMENTALS_PATH if DEFAULT_FUNDAMENTALS_PATH.exists() else None,
        shareholding_path=DEFAULT_SHAREHOLDING_PATH if DEFAULT_SHAREHOLDING_PATH.exists() else None,
        macro_daily_path=DEFAULT_MACRO_PATH if DEFAULT_MACRO_PATH.exists() else None,
        announcements_path=DEFAULT_ANNOUNCEMENTS_PATH if DEFAULT_ANNOUNCEMENTS_PATH.exists() else None,
        event_daily_path=None,
        as_of_date=as_of_trade_date.isoformat(),
        top_n=max(portfolio_size * 3, 30),
    )

    report_dir = run_dir / "report"
    artifacts = generate_weekly_portfolio_report(
        output_dir=report_dir,
        current_universe_path=current_universe_dir / "current_scored_universe.parquet",
        feature_results_path=study_dir / "layer_feature_results.csv",
        daily_facts_path=DEFAULT_DAILY_FACTS_PATH,
        config_path=DEFAULT_FEATURE_CONFIG_PATH,
        fundamentals_path=DEFAULT_FUNDAMENTALS_PATH if DEFAULT_FUNDAMENTALS_PATH.exists() else None,
        shareholding_path=DEFAULT_SHAREHOLDING_PATH if DEFAULT_SHAREHOLDING_PATH.exists() else None,
        macro_daily_path=DEFAULT_MACRO_PATH if DEFAULT_MACRO_PATH.exists() else None,
        announcements_path=DEFAULT_ANNOUNCEMENTS_PATH if DEFAULT_ANNOUNCEMENTS_PATH.exists() else None,
        event_daily_path=None,
        as_of_date=as_of_trade_date.isoformat(),
        target_date=target_date.isoformat(),
        portfolio_size=portfolio_size,
        cash_buffer_pct=cash_buffer_pct,
    )

    report_frame, exits = _annotate_actions(artifacts.report_frame, state_dir / "last_successful_report.csv")
    report_frame.to_csv(artifacts.csv_path, index=False)
    exits_path = report_dir / f"exits_{run_ts.strftime('%Y%m%d')}.csv"
    exits.to_csv(exits_path, index=False)
    success_html = _success_html(run_ts, as_of_trade_date.isoformat(), report_frame, cash_buffer_pct=cash_buffer_pct)
    artifacts.html_path.write_text(success_html, encoding="utf-8")

    portfolio_gates = validate_portfolio_report(report_frame, expected_count=portfolio_size, cash_buffer_pct=cash_buffer_pct)
    all_gates = preflight_gates + [study_gate] + portfolio_gates
    failed_after_report = [gate for gate in all_gates if not gate.passed]
    if failed_after_report:
        blocked_csv = run_dir / "blocked_status.csv"
        blocked_html = run_dir / "blocked_status.html"
        pd.DataFrame([asdict(gate) for gate in all_gates]).to_csv(blocked_csv, index=False)
        blocked_html.write_text(_blocked_html(run_ts, all_gates), encoding="utf-8")
        send_report_email(
            recipients=recipients,
            subject=f"BLOCKED - Weekly NSE portfolio run - {run_ts.date().isoformat()}",
            html_body=blocked_html.read_text(encoding="utf-8"),
            csv_path=blocked_csv,
            allow_sendmail_fallback=False,
        )
        payload = {
            "mode": "final",
            "ok": False,
            "blocked": True,
            "run_timestamp": run_ts.isoformat(),
            "refresh_steps": [asdict(step) for step in light_refresh_steps],
            "gates": [asdict(gate) for gate in all_gates],
            "artifacts": {"blocked_csv": str(blocked_csv), "blocked_html": str(blocked_html)},
        }
        write_json(payload, state_dir / "latest_final_status.json")
        return payload

    delivery_method = send_report_email(
        recipients=recipients,
        subject=f"Weekly NSE year-end portfolio - {run_ts.date().isoformat()}",
        html_body=success_html,
        csv_path=artifacts.csv_path,
        allow_sendmail_fallback=False,
    )
    report_frame.to_csv(state_dir / "last_successful_report.csv", index=False)
    payload = {
        "mode": "final",
        "ok": True,
        "blocked": False,
        "run_timestamp": run_ts.isoformat(),
        "refresh_steps": [asdict(step) for step in light_refresh_steps],
        "gates": [asdict(gate) for gate in all_gates],
        "delivery_method": delivery_method,
        "artifacts": {
            "study_dir": str(study_dir),
            "universe_path": str(artifacts.universe_path),
            "report_csv": str(artifacts.csv_path),
            "report_html": str(artifacts.html_path),
            "exits_csv": str(exits_path),
        },
    }
    write_json(payload, state_dir / "latest_final_status.json")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cache", "final"], required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_PRODUCTION_ROOT))
    parser.add_argument("--run-timestamp", default="")
    parser.add_argument("--portfolio-size", type=int, default=10)
    parser.add_argument("--cash-buffer-pct", type=float, default=10.0)
    parser.add_argument("--recipient", action="append", default=[])
    args = parser.parse_args()

    run_ts = _now_ist(args.run_timestamp or None)
    output_root = Path(args.output_root)
    state_dir = output_root / "state"

    if args.mode == "cache":
        payload = run_cache_mode(run_ts=run_ts, state_dir=state_dir)
    else:
        if not args.recipient:
            raise SystemExit("At least one --recipient is required in final mode.")
        payload = run_final_mode(
            run_ts=run_ts,
            output_root=output_root,
            recipients=args.recipient,
            portfolio_size=args.portfolio_size,
            cash_buffer_pct=args.cash_buffer_pct,
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
