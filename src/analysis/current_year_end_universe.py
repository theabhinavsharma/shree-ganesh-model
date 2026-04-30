from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.forward_return_study import _read_config
from src.analysis.forward_return_study import _read_optional_table
from src.analysis.forward_return_study import _read_stock_master
from src.analysis.model_scoring import apply_feature_score
from src.screen.build_universe import build_daily_screen_universe
from src.transform.event_daily import build_event_feature_daily
from src.utils.io import write_parquet


def build_current_year_end_universe(
    *,
    daily_facts_path: Path,
    feature_results_path: Path,
    output_dir: Path,
    config_path: Path,
    stock_master_path: Path | None = None,
    fundamentals_path: Path | None = None,
    shareholding_path: Path | None = None,
    sector_state_daily_path: Path | None = None,
    macro_daily_path: Path | None = None,
    announcements_path: Path | None = None,
    event_daily_path: Path | None = None,
    as_of_date: str | None = None,
    top_n: int = 30,
) -> pd.DataFrame:
    daily_facts = pd.read_parquet(daily_facts_path)
    daily_facts["trade_date"] = pd.to_datetime(daily_facts["trade_date"]).dt.normalize()
    snapshot_date = pd.Timestamp(as_of_date).normalize() if as_of_date else daily_facts["trade_date"].max()
    daily_slice = daily_facts[daily_facts["trade_date"] == snapshot_date].copy()

    stock_master = _read_stock_master(stock_master_path)
    fundamentals = _read_optional_table(fundamentals_path)
    shareholding = _read_optional_table(shareholding_path)
    sector_state_daily = _read_optional_table(sector_state_daily_path)
    config = _read_config(config_path)
    universe = build_daily_screen_universe(
        daily_facts=daily_slice,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=sector_state_daily,
        config=config,
    )

    macro_daily = _read_optional_table(macro_daily_path)
    if not macro_daily.empty:
        macro_daily["trade_date"] = pd.to_datetime(macro_daily["trade_date"]).dt.normalize()
        universe = universe.merge(macro_daily, on="trade_date", how="left")

    events_daily = _read_optional_table(event_daily_path)
    if events_daily.empty and announcements_path and announcements_path.exists():
        announcements = _read_optional_table(announcements_path)
        events_daily = build_event_feature_daily(universe[["symbol", "trade_date"]], announcements)
    if not events_daily.empty:
        events_daily["trade_date"] = pd.to_datetime(events_daily["trade_date"]).dt.normalize()
        universe = universe.merge(events_daily, on=["symbol", "trade_date"], how="left")

    feature_results = pd.read_csv(feature_results_path)
    scoring_features = feature_results[feature_results["test_lift"].gt(1.0)].copy()
    universe = apply_feature_score(universe, scoring_features)
    universe = universe.sort_values(["model_score", "model_pass_count", "symbol"], ascending=[False, False, True]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(universe, output_dir / "current_scored_universe.parquet")
    stocks = universe[~universe.get("instrument_type", pd.Series("", index=universe.index)).astype(str).str.contains("ETF", case=False, na=False)].head(top_n)
    etfs = universe[universe.get("instrument_type", pd.Series("", index=universe.index)).astype(str).str.contains("ETF", case=False, na=False)].head(top_n)
    stocks.to_csv(output_dir / "top_stocks.csv", index=False)
    etfs.to_csv(output_dir / "top_etfs.csv", index=False)
    return universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-facts-path", required=True)
    parser.add_argument("--feature-results-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--stock-master-path", default="")
    parser.add_argument("--fundamentals-path", default="")
    parser.add_argument("--shareholding-path", default="")
    parser.add_argument("--sector-state-daily-path", default="")
    parser.add_argument("--macro-daily-path", default="")
    parser.add_argument("--announcements-path", default="")
    parser.add_argument("--event-daily-path", default="")
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--top-n", type=int, default=30)
    args = parser.parse_args()

    build_current_year_end_universe(
        daily_facts_path=Path(args.daily_facts_path),
        feature_results_path=Path(args.feature_results_path),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config_path),
        stock_master_path=Path(args.stock_master_path) if args.stock_master_path else None,
        fundamentals_path=Path(args.fundamentals_path) if args.fundamentals_path else None,
        shareholding_path=Path(args.shareholding_path) if args.shareholding_path else None,
        sector_state_daily_path=Path(args.sector_state_daily_path) if args.sector_state_daily_path else None,
        macro_daily_path=Path(args.macro_daily_path) if args.macro_daily_path else None,
        announcements_path=Path(args.announcements_path) if args.announcements_path else None,
        event_daily_path=Path(args.event_daily_path) if args.event_daily_path else None,
        as_of_date=args.as_of_date or None,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
