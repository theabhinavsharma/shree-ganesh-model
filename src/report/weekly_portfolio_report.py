from __future__ import annotations

import argparse
import os
import smtplib
import subprocess
from dataclasses import dataclass
from datetime import date
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.current_year_end_universe import build_current_year_end_universe
from src.report.checklist import enrich_scored_universe_for_checklist
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_report_directory_readme


DEFAULT_DAILY_FACTS_PATH = Path("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
DEFAULT_FEATURE_RESULTS_PATH = Path("tmp/layer_edge_2015plus_288d50pct_adjusted_fast/layer_feature_results.csv")
DEFAULT_CONFIG_PATH = Path("configs/screening.yaml")
DEFAULT_FUNDAMENTALS_PATH = Path("data/fundamentals_full_history/normalized/stock_quarterly_fundamentals.parquet")
DEFAULT_SHAREHOLDING_PATH = Path("data/shareholding_full_history/normalized/stock_shareholding_quarterly.parquet")
DEFAULT_MACRO_PATH = Path("data/macro_full_history/normalized/macro_feature_daily.parquet")
DEFAULT_ANNOUNCEMENTS_PATH = Path("data/events_full_history/normalized/stock_announcements.parquet")
DEFAULT_EVENTS_PATH = Path("data/events_full_history/normalized/event_feature_daily.parquet")
P0_SELECTION_FILTERS = [
    "filter_market_cap",
    "filter_debt",
    "filter_revenue_growth",
    "filter_profit_cagr",
    "filter_volume_expansion",
    "filter_volume_high_3m",
    "filter_delivery_above_5d_avg",
    "filter_rsi_daily",
    "filter_rsi_weekly",
    "filter_rsi_monthly",
    "filter_pe",
    "filter_promoter_holding",
    "filter_above_50_dma",
    "filter_above_200_dma",
]


@dataclass(frozen=True)
class ReportArtifacts:
    csv_path: Path
    html_path: Path
    universe_path: Path
    report_frame: pd.DataFrame


def _default_target_date(as_of_date: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(date(as_of_date.year, 12, 31))


def _normalize_company_name(frame: pd.DataFrame) -> pd.Series:
    if "company_name" in frame.columns:
        company = frame["company_name"].fillna("").astype(str).str.strip()
        return company.where(company.ne(""), frame["symbol"].astype(str))
    return frame["symbol"].astype(str)


def _liquidity_multiplier(avg_traded_value_20d: pd.Series) -> pd.Series:
    value = pd.to_numeric(avg_traded_value_20d, errors="coerce")
    conditions = [
        value.ge(100_000_000_0),
        value.ge(20_000_000_0) & value.lt(100_000_000_0),
        value.ge(5_000_000_0) & value.lt(20_000_000_0),
        value.ge(1_000_000_0) & value.lt(5_000_000_0),
    ]
    choices = [1.0, 0.95, 0.85, 0.75]
    return pd.Series(np.select(conditions, choices, default=0.60), index=avg_traded_value_20d.index, dtype=float)


def _confidence_score(model_score: pd.Series) -> pd.Series:
    score = pd.to_numeric(model_score, errors="coerce")
    min_score = score.min()
    max_score = score.max()
    if pd.isna(min_score) or pd.isna(max_score) or np.isclose(min_score, max_score):
        return pd.Series(75.0, index=score.index)
    normalized = (score - min_score) / (max_score - min_score)
    return (50.0 + normalized * 50.0).round(1)


def _suggested_allocation(frame: pd.DataFrame, *, cash_buffer_pct: float) -> pd.Series:
    base_score = pd.to_numeric(frame["model_score"], errors="coerce").fillna(0.0)
    base_score = base_score - base_score.min() + 0.25
    liquidity = _liquidity_multiplier(frame.get("avg_traded_value_20d", pd.Series(np.nan, index=frame.index)))
    weight = base_score * liquidity
    total_weight = float(weight.sum())
    investable_pct = max(0.0, 100.0 - cash_buffer_pct)
    if total_weight <= 0:
        return pd.Series(investable_pct / max(len(frame), 1), index=frame.index)
    return (weight / total_weight * investable_pct).round(2)


def _buy_range(close_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    close_value = pd.to_numeric(close_price, errors="coerce")
    return (close_value * 0.99).round(2), (close_value * 1.03).round(2)


def _stop_loss(close_price: pd.Series, sma_50: pd.Series) -> pd.Series:
    close_value = pd.to_numeric(close_price, errors="coerce")
    sma_value = pd.to_numeric(sma_50, errors="coerce")
    trailing_floor = (sma_value * 0.99).where((sma_value * 0.99).lt(close_value))
    hard_floor = close_value * 0.88
    return pd.concat([hard_floor, trailing_floor], axis=1).max(axis=1).round(2)


def _series_or_na(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _format_price(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{numeric:.2f}" if not pd.isna(numeric) else ""


def _reason_columns(row: pd.Series) -> str:
    reasons: list[str] = []
    if _truthy_flag(row.get("filter_sector_institutional_buying", False)):
        if _truthy_flag(row.get("sector_fii_dii_buying_exact_flag", False)):
            reasons.append("sector FII+DII 30d positive")
        elif _truthy_flag(row.get("sector_fii_dii_buying_proxy_flag", False)):
            reasons.append("sector FII+DII proxy positive")
    if _truthy_flag(row.get("recent_promoter_buy_flag", False)):
        reasons.append("recent promoter buy")
    if _truthy_flag(row.get("recent_order_win_flag", False)):
        reasons.append("recent order win")
    if _truthy_flag(row.get("recent_approval_flag", False)):
        reasons.append("recent approval")
    if _truthy_flag(row.get("filter_debt", False)):
        reasons.append("debt free")
    if _truthy_flag(row.get("filter_above_200_dma", False)):
        reasons.append("above 200 DMA")
    if _truthy_flag(row.get("filter_above_50_dma", False)):
        reasons.append("above 50 DMA")
    if _truthy_flag(row.get("filter_delivery_above_5d_avg", False)):
        reasons.append("delivery > 5d avg")
    if _truthy_flag(row.get("channel_valid_flag", False)) and str(row.get("trade_action", "")).strip():
        reasons.append(f"channel {row.get('trade_action')}".lower())
    pe_ttm = pd.to_numeric(row.get("pe_ttm"), errors="coerce")
    if not pd.isna(pe_ttm):
        reasons.append(f"PE {pe_ttm:.1f}")
    promoter_pct = pd.to_numeric(row.get("promoter_pct"), errors="coerce")
    if not pd.isna(promoter_pct) and promoter_pct >= 40:
        reasons.append(f"promoter {promoter_pct:.1f}%")
    market_cap_cr = pd.to_numeric(row.get("market_cap_cr"), errors="coerce")
    if not pd.isna(market_cap_cr) and market_cap_cr >= 5000:
        reasons.append(f"mcap {market_cap_cr:.0f} cr")
    revenue_cagr_5y = pd.to_numeric(row.get("revenue_cagr_5y"), errors="coerce")
    if not pd.isna(revenue_cagr_5y) and revenue_cagr_5y >= 0.10:
        reasons.append(f"revenue CAGR {revenue_cagr_5y * 100:.1f}%")
    pat_cagr_5y = pd.to_numeric(row.get("pat_cagr_5y"), errors="coerce")
    if not pd.isna(pat_cagr_5y) and pat_cagr_5y >= 0.20:
        reasons.append(f"PAT CAGR {pat_cagr_5y * 100:.1f}%")
    return ", ".join(reasons[:4])


def _truthy_flag(value: object) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _non_etf_mask(frame: pd.DataFrame) -> pd.Series:
    if "instrument_type" not in frame.columns:
        return pd.Series(True, index=frame.index, dtype=bool)
    return ~frame["instrument_type"].fillna("").astype(str).str.contains("ETF", case=False)


def _priority_p0_candidates(universe: pd.DataFrame) -> pd.DataFrame:
    stocks = universe.copy()
    stocks = stocks[_non_etf_mask(stocks)].copy()
    available_filters = [column for column in P0_SELECTION_FILTERS if column in stocks.columns]
    if not available_filters:
        return stocks.sort_values(["model_score", "model_pass_count", "symbol"], ascending=[False, False, True])
    filter_frame = stocks[available_filters].astype("boolean")
    stocks["screen_pass_count"] = filter_frame.fillna(False).sum(axis=1).astype(int)
    stocks["screen_available_count"] = filter_frame.notna().sum(axis=1).astype(int)
    available = stocks["screen_available_count"].replace(0, np.nan)
    stocks["screen_pass_ratio"] = (stocks["screen_pass_count"] / available).fillna(0.0).round(4)
    if "trade_action" in stocks.columns:
        stocks["timing_buy_bonus"] = stocks["trade_action"].fillna("").astype(str).str.upper().eq("BUY").astype(int)
    else:
        stocks["timing_buy_bonus"] = 0
    return stocks.sort_values(
        ["screen_pass_count", "screen_pass_ratio", "timing_buy_bonus", "model_score", "model_pass_count", "symbol"],
        ascending=[False, False, False, False, False, True],
    )


def build_weekly_portfolio_table(
    universe: pd.DataFrame,
    *,
    portfolio_size: int = 10,
    cash_buffer_pct: float = 10.0,
    target_date: str | pd.Timestamp | None = None,
    selection_mode: str = "strict",
) -> pd.DataFrame:
    stocks = universe.copy()
    normalized_selection_mode = selection_mode.strip().lower()
    if normalized_selection_mode == "strict":
        if "strategy_checklist_pass" in stocks.columns:
            stocks = stocks[stocks["strategy_checklist_pass"].astype("boolean").fillna(False)].copy()
        if "trade_action" in stocks.columns:
            stocks = stocks[stocks["trade_action"].fillna("").astype(str).str.upper().eq("BUY")].copy()
        if "instrument_type" in stocks.columns and "strategy_checklist_pass" not in universe.columns:
            stocks = stocks[_non_etf_mask(stocks)].copy()
        stocks = stocks.sort_values(["model_score", "model_pass_count", "symbol"], ascending=[False, False, True]).head(portfolio_size).copy()
    elif normalized_selection_mode == "priority_p0":
        stocks = _priority_p0_candidates(stocks).head(portfolio_size).copy()
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")
    stocks = stocks.reset_index(drop=True)

    stocks["rank"] = np.arange(1, len(stocks) + 1)
    stocks["stock_name"] = _normalize_company_name(stocks)
    stocks["current_price"] = pd.to_numeric(stocks["close"], errors="coerce").round(2)
    default_buy_low, default_buy_high = _buy_range(stocks["current_price"])
    stocks["buy_price_low"] = pd.to_numeric(_series_or_na(stocks, "channel_buy_price_low"), errors="coerce").fillna(default_buy_low).round(2)
    stocks["buy_price_high"] = pd.to_numeric(_series_or_na(stocks, "channel_buy_price_high"), errors="coerce").fillna(default_buy_high).round(2)
    stocks["sell_target"] = pd.to_numeric(_series_or_na(stocks, "channel_sell_target"), errors="coerce").fillna((stocks["current_price"] * 1.50).round(2)).round(2)
    stocks["stop_loss"] = pd.to_numeric(_series_or_na(stocks, "channel_stop_loss"), errors="coerce").fillna(
        _stop_loss(stocks["current_price"], stocks.get("sma_50", pd.Series(np.nan, index=stocks.index)))
    ).round(2)
    stocks["confidence_score"] = _confidence_score(stocks["model_score"])
    stocks["allocation_pct"] = _suggested_allocation(stocks, cash_buffer_pct=cash_buffer_pct)
    stocks["target_date"] = pd.Timestamp(target_date).date().isoformat() if target_date is not None else ""
    stocks["buy_price_range"] = stocks["buy_price_low"].map(_format_price).astype("string") + " - " + stocks["buy_price_high"].map(_format_price).astype("string")
    stocks["reasons"] = stocks.apply(_reason_columns, axis=1)
    stocks["liquidity_20d_cr"] = (pd.to_numeric(stocks.get("avg_traded_value_20d"), errors="coerce") / 10_000_000).round(2)
    stocks["selection_mode"] = normalized_selection_mode

    columns = [
        "rank",
        "symbol",
        "stock_name",
        "sector",
        "current_price",
        "buy_price_low",
        "buy_price_high",
        "buy_price_range",
        "sell_target",
        "stop_loss",
        "target_date",
        "trade_action",
        "confidence_score",
        "allocation_pct",
        "liquidity_20d_cr",
        "market_cap_cr",
        "model_score",
        "model_pass_count",
        "screen_pass_count",
        "screen_available_count",
        "screen_pass_ratio",
        "selection_mode",
        "reasons",
    ]
    for column in columns:
        if column not in stocks.columns:
            stocks[column] = pd.NA
    return stocks[columns].copy()


def build_report_html(report: pd.DataFrame, *, objective: str, run_date: str, cash_buffer_pct: float) -> str:
    display = report.copy()
    for column in [
        "current_price",
        "buy_price_low",
        "buy_price_high",
        "sell_target",
        "stop_loss",
        "confidence_score",
        "allocation_pct",
        "liquidity_20d_cr",
        "market_cap_cr",
        "model_score",
    ]:
        display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
    return f"""
    <html>
      <body>
        <p><strong>Weekly NSE portfolio report</strong></p>
        <p>Run date: {run_date}<br>
        Objective: {objective}<br>
        Cash buffer: {cash_buffer_pct:.1f}%</p>
        {display.to_html(index=False, border=0)}
      </body>
    </html>
    """.strip()


def get_smtp_config() -> dict[str, str] | None:
    host = os.getenv("REPORT_SMTP_HOST", "").strip()
    user = os.getenv("REPORT_SMTP_USER", "").strip()
    password = os.getenv("REPORT_SMTP_PASSWORD", "").strip()
    sender = os.getenv("REPORT_SMTP_FROM", "").strip() or user
    port = os.getenv("REPORT_SMTP_PORT", "").strip() or "587"
    if not host or not user or not password or not sender:
        return None
    return {
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "sender": sender,
        "ssl": os.getenv("REPORT_SMTP_SSL", "").strip().lower() in {"1", "true", "yes"},
    }


def send_report_email(
    *,
    recipients: list[str],
    subject: str,
    html_body: str,
    csv_path: Path,
    allow_sendmail_fallback: bool = True,
) -> str:
    message = EmailMessage()
    smtp = get_smtp_config()
    sender = smtp["sender"] if smtp else os.getenv("REPORT_FALLBACK_SENDER", "codex-report@localhost")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content("Weekly NSE portfolio report attached as CSV. Use an email client that supports HTML to view the table.")
    message.add_alternative(html_body, subtype="html")
    message.add_attachment(csv_path.read_bytes(), maintype="text", subtype="csv", filename=csv_path.name)

    if smtp:
        if smtp["ssl"]:
            with smtplib.SMTP_SSL(smtp["host"], int(smtp["port"])) as client:
                client.login(smtp["user"], smtp["password"])
                client.send_message(message)
        else:
            with smtplib.SMTP(smtp["host"], int(smtp["port"])) as client:
                client.starttls()
                client.login(smtp["user"], smtp["password"])
                client.send_message(message)
        return "smtp"

    sendmail_path = Path("/usr/sbin/sendmail")
    if allow_sendmail_fallback and sendmail_path.exists():
        process = subprocess.run(
            [str(sendmail_path), "-t", "-i"],
            input=message.as_bytes(),
            capture_output=True,
            check=False,
        )
        if process.returncode == 0:
            return "sendmail"
        stderr = process.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"sendmail failed: {stderr or 'unknown error'}")

    raise RuntimeError("No SMTP configuration found and sendmail fallback is unavailable.")


def generate_weekly_portfolio_report(
    *,
    output_dir: Path,
    current_universe_path: Path | None = None,
    daily_facts_path: Path = DEFAULT_DAILY_FACTS_PATH,
    feature_results_path: Path = DEFAULT_FEATURE_RESULTS_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    stock_master_path: Path | None = None,
    fundamentals_path: Path | None = DEFAULT_FUNDAMENTALS_PATH,
    shareholding_path: Path | None = DEFAULT_SHAREHOLDING_PATH,
    sector_state_daily_path: Path | None = None,
    macro_daily_path: Path | None = DEFAULT_MACRO_PATH,
    announcements_path: Path | None = DEFAULT_ANNOUNCEMENTS_PATH,
    event_daily_path: Path | None = DEFAULT_EVENTS_PATH,
    quote_snapshot_path: Path | None = None,
    as_of_date: str | None = None,
    portfolio_size: int = 10,
    cash_buffer_pct: float = 10.0,
    target_date: str | None = None,
    selection_mode: str = "strict",
) -> ReportArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_date = pd.Timestamp(as_of_date).normalize() if as_of_date else pd.Timestamp.today().normalize()

    if current_universe_path and current_universe_path.exists():
        universe_path = current_universe_path
    else:
        universe_output_dir = output_dir / "current_universe"
        build_current_year_end_universe(
            daily_facts_path=daily_facts_path,
            feature_results_path=feature_results_path,
            output_dir=universe_output_dir,
            config_path=config_path,
            stock_master_path=stock_master_path,
            fundamentals_path=fundamentals_path if fundamentals_path and fundamentals_path.exists() else None,
            shareholding_path=shareholding_path if shareholding_path and shareholding_path.exists() else None,
            sector_state_daily_path=sector_state_daily_path,
            macro_daily_path=macro_daily_path if macro_daily_path and macro_daily_path.exists() else None,
            announcements_path=announcements_path if announcements_path and announcements_path.exists() else None,
            event_daily_path=event_daily_path if event_daily_path and event_daily_path.exists() else None,
            as_of_date=run_date.date().isoformat(),
            top_n=max(portfolio_size * 3, 30),
        )
        universe_path = universe_output_dir / "current_scored_universe.parquet"

    universe = pd.read_parquet(universe_path)
    universe = enrich_scored_universe_for_checklist(
        universe,
        daily_facts_path=daily_facts_path,
        config_path=config_path,
        output_dir=output_dir / "checklist",
        quote_snapshot_path=quote_snapshot_path,
        candidate_count=max(portfolio_size * 20, 120),
    )
    target_ts = pd.Timestamp(target_date).normalize() if target_date else _default_target_date(run_date)
    report = build_weekly_portfolio_table(
        universe,
        portfolio_size=portfolio_size,
        cash_buffer_pct=cash_buffer_pct,
        target_date=target_ts,
        selection_mode=selection_mode,
    )

    date_tag = run_date.strftime("%Y%m%d")
    csv_path = output_dir / f"weekly_portfolio_{date_tag}.csv"
    html_path = output_dir / f"weekly_portfolio_{date_tag}.html"
    report.to_csv(csv_path, index=False)
    html_path.write_text(
        build_report_html(
            report,
            objective=f"50%+ by {target_ts.date().isoformat()}",
            run_date=run_date.date().isoformat(),
            cash_buffer_pct=cash_buffer_pct,
        ),
        encoding="utf-8",
    )
    write_dataframe_manifest(
        csv_path,
        report,
        generated_by="src.report.weekly_portfolio_report",
        as_of_date=run_date.date().isoformat(),
        extra_notes=[
            f"This weekly portfolio targets 50%+ by {target_ts.date().isoformat()}.",
        ],
    )
    write_report_directory_readme(
        output_dir,
        title=f"Weekly Portfolio Report For {run_date.date().isoformat()}",
        intro_lines=[
            "This folder contains a human-readable weekly portfolio output.",
            "Open the CSV manifest sidecar if you want column meanings and sample values without reading code.",
            f"Source universe file used for this report: `{universe_path}`.",
        ],
        files=[csv_path, html_path],
    )
    return ReportArtifacts(csv_path=csv_path, html_path=html_path, universe_path=universe_path, report_frame=report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--current-universe-path", default="")
    parser.add_argument("--daily-facts-path", default=str(DEFAULT_DAILY_FACTS_PATH))
    parser.add_argument("--feature-results-path", default=str(DEFAULT_FEATURE_RESULTS_PATH))
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--stock-master-path", default="")
    parser.add_argument("--fundamentals-path", default=str(DEFAULT_FUNDAMENTALS_PATH))
    parser.add_argument("--shareholding-path", default=str(DEFAULT_SHAREHOLDING_PATH))
    parser.add_argument("--sector-state-daily-path", default="")
    parser.add_argument("--macro-daily-path", default=str(DEFAULT_MACRO_PATH))
    parser.add_argument("--announcements-path", default=str(DEFAULT_ANNOUNCEMENTS_PATH))
    parser.add_argument("--event-daily-path", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--quote-snapshot-path", default="")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--target-date", default="")
    parser.add_argument("--portfolio-size", type=int, default=10)
    parser.add_argument("--cash-buffer-pct", type=float, default=10.0)
    parser.add_argument("--selection-mode", default="strict")
    parser.add_argument("--email", action="store_true")
    parser.add_argument("--recipient", action="append", default=[])
    args = parser.parse_args()

    artifacts = generate_weekly_portfolio_report(
        output_dir=Path(args.output_dir),
        current_universe_path=Path(args.current_universe_path) if args.current_universe_path else None,
        daily_facts_path=Path(args.daily_facts_path),
        feature_results_path=Path(args.feature_results_path),
        config_path=Path(args.config_path),
        stock_master_path=Path(args.stock_master_path) if args.stock_master_path else None,
        fundamentals_path=Path(args.fundamentals_path) if args.fundamentals_path else None,
        shareholding_path=Path(args.shareholding_path) if args.shareholding_path else None,
        sector_state_daily_path=Path(args.sector_state_daily_path) if args.sector_state_daily_path else None,
        macro_daily_path=Path(args.macro_daily_path) if args.macro_daily_path else None,
        announcements_path=Path(args.announcements_path) if args.announcements_path else None,
        event_daily_path=Path(args.event_daily_path) if args.event_daily_path else None,
        quote_snapshot_path=Path(args.quote_snapshot_path) if args.quote_snapshot_path else None,
        as_of_date=args.as_of_date or None,
        portfolio_size=args.portfolio_size,
        cash_buffer_pct=args.cash_buffer_pct,
        target_date=args.target_date or None,
        selection_mode=args.selection_mode,
    )

    if args.email:
        if not args.recipient:
            raise SystemExit("At least one --recipient is required when --email is set.")
        subject = f"Weekly NSE year-end portfolio report - {pd.Timestamp(args.as_of_date).date().isoformat() if args.as_of_date else pd.Timestamp.today().date().isoformat()}"
        delivery_method = send_report_email(
            recipients=args.recipient,
            subject=subject,
            html_body=artifacts.html_path.read_text(encoding="utf-8"),
            csv_path=artifacts.csv_path,
        )
        print(f"email_sent_via={delivery_method}")

    print(f"csv_path={artifacts.csv_path}")
    print(f"html_path={artifacts.html_path}")
    print(f"universe_path={artifacts.universe_path}")


if __name__ == "__main__":
    main()
