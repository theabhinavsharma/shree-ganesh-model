from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.analysis.forward_return_study import _read_config
from src.analysis.forward_return_study import _read_optional_table
from src.analysis.forward_return_study import _read_stock_master
from src.analysis.forward_return_study import _safe_divide
from src.analysis.forward_return_study import _safe_ratio
from src.analysis.forward_return_study import _select_numeric_threshold
from src.analysis.forward_return_study import add_bucket_columns
from src.analysis.forward_return_study import add_market_regime_features
from src.analysis.forward_return_study import build_forward_return_labels
from src.screen.build_universe import build_daily_screen_universe
from src.transform.build_daily_facts import build_stock_daily_facts
from src.transform.event_daily import build_event_feature_daily
from src.transform.lagged_join import latest_effective_join
from src.utils.io import write_parquet


@dataclass(frozen=True)
class LayerFeature:
    layer: str
    column: str
    direction: str
    feature_type: str


PRICE_NUMERIC = (
    LayerFeature("price", "return_20d", "ge", "numeric"),
    LayerFeature("price", "volume_vs_20d", "ge", "numeric"),
    LayerFeature("price", "traded_value_vs_20d", "ge", "numeric"),
    LayerFeature("price", "delivery_pct", "ge", "numeric"),
    LayerFeature("price", "delivery_pct_vs_20d", "ge", "numeric"),
    LayerFeature("price", "rsi_14_daily", "ge", "numeric"),
    LayerFeature("price", "rsi_14_weekly", "ge", "numeric"),
    LayerFeature("price", "rsi_14_monthly", "ge", "numeric"),
    LayerFeature("price", "avg_traded_value_20d_cr", "ge", "numeric"),
    LayerFeature("price", "breadth_above_50_dma", "ge", "numeric"),
    LayerFeature("price", "breadth_above_200_dma", "ge", "numeric"),
    LayerFeature("price", "breadth_volume_1_5x", "ge", "numeric"),
    LayerFeature("price", "market_median_return_20d", "ge", "numeric"),
)
PRICE_BOOLEAN = (
    LayerFeature("price", "filter_above_50_dma", "ge", "boolean"),
    LayerFeature("price", "filter_above_200_dma", "ge", "boolean"),
    LayerFeature("price", "volume_high_63d_flag", "ge", "boolean"),
    LayerFeature("price", "delivery_pct_high_63d_flag", "ge", "boolean"),
)

MACRO_NUMERIC = (
    LayerFeature("macro", "india_vix_level", "le", "numeric"),
    LayerFeature("macro", "india_vix_return_20d", "le", "numeric"),
    LayerFeature("macro", "nifty_50_return_20d", "ge", "numeric"),
    LayerFeature("macro", "nifty_500_return_20d", "ge", "numeric"),
    LayerFeature("macro", "nifty_bank_return_20d", "ge", "numeric"),
    LayerFeature("macro", "nifty_it_return_20d", "ge", "numeric"),
    LayerFeature("macro", "nifty_pharma_return_20d", "ge", "numeric"),
    LayerFeature("macro", "fred_usdinr_return_20d", "le", "numeric"),
    LayerFeature("macro", "fred_wti_crude_return_20d", "ge", "numeric"),
    LayerFeature("macro", "fred_sp500_return_20d", "ge", "numeric"),
    LayerFeature("macro", "fred_nasdaq_comp_return_20d", "ge", "numeric"),
    LayerFeature("macro", "nifty_50_pe", "le", "numeric"),
    LayerFeature("macro", "nifty_50_pb", "le", "numeric"),
    LayerFeature("macro", "nifty_50_dy", "ge", "numeric"),
    LayerFeature("macro", "macro_india_minus_spx_return_20d", "ge", "numeric"),
)
MACRO_BOOLEAN = (
    LayerFeature("macro", "macro_vix_below_20", "ge", "boolean"),
    LayerFeature("macro", "macro_vix_below_15", "ge", "boolean"),
    LayerFeature("macro", "macro_risk_on_flag", "ge", "boolean"),
    LayerFeature("macro", "nifty_50_above_50dma", "ge", "boolean"),
    LayerFeature("macro", "nifty_500_above_50dma", "ge", "boolean"),
)

EVENT_NUMERIC = (
    LayerFeature("event", "announcements_7d", "ge", "numeric"),
    LayerFeature("event", "announcements_30d", "ge", "numeric"),
    LayerFeature("event", "results_events_30d", "ge", "numeric"),
    LayerFeature("event", "order_wins_90d", "ge", "numeric"),
    LayerFeature("event", "approvals_90d", "ge", "numeric"),
    LayerFeature("event", "pledge_changes_90d", "ge", "numeric"),
    LayerFeature("event", "promoter_buys_180d", "ge", "numeric"),
    LayerFeature("event", "days_since_results_event", "le", "numeric"),
    LayerFeature("event", "days_since_order_win", "le", "numeric"),
    LayerFeature("event", "days_since_approval", "le", "numeric"),
)
EVENT_BOOLEAN = (
    LayerFeature("event", "recent_results_flag", "ge", "boolean"),
    LayerFeature("event", "recent_order_win_flag", "ge", "boolean"),
    LayerFeature("event", "recent_approval_flag", "ge", "boolean"),
    LayerFeature("event", "recent_pledge_change_flag", "ge", "boolean"),
    LayerFeature("event", "recent_promoter_buy_flag", "ge", "boolean"),
)

FILING_NUMERIC = (
    LayerFeature("filing", "promoter_pct", "ge", "numeric"),
    LayerFeature("filing", "fii_fpi_pct", "ge", "numeric"),
    LayerFeature("filing", "dii_pct", "ge", "numeric"),
    LayerFeature("filing", "mf_pct", "ge", "numeric"),
    LayerFeature("filing", "promoter_pct_qoq_change", "ge", "numeric"),
    LayerFeature("filing", "fii_fpi_pct_qoq_change", "ge", "numeric"),
    LayerFeature("filing", "dii_pct_qoq_change", "ge", "numeric"),
    LayerFeature("filing", "mf_pct_qoq_change", "ge", "numeric"),
    LayerFeature("filing", "revenue_cagr_5y", "ge", "numeric"),
    LayerFeature("filing", "pat_cagr_5y", "ge", "numeric"),
    LayerFeature("filing", "pe_ttm", "le", "numeric"),
    LayerFeature("filing", "interest_coverage", "ge", "numeric"),
    LayerFeature("filing", "debt_equity_ratio", "le", "numeric"),
    LayerFeature("filing", "debt_service_coverage_ratio", "ge", "numeric"),
)
FILING_BOOLEAN = (
    LayerFeature("filing", "ebitda_positive_last_5q_flag", "ge", "boolean"),
    LayerFeature("filing", "public_breakdown_available_flag", "ge", "boolean"),
)

ALL_FEATURES = PRICE_NUMERIC + PRICE_BOOLEAN + MACRO_NUMERIC + MACRO_BOOLEAN + EVENT_NUMERIC + EVENT_BOOLEAN + FILING_NUMERIC + FILING_BOOLEAN


@dataclass(frozen=True)
class LayerSummary:
    layer: str
    feature_count: int
    available_feature_count: int
    best_test_feature: str | None
    best_test_precision: float | None
    best_test_lift: float | None
    best_test_avg_return: float | None
    best_test_median_return: float | None


def run_layer_edge_study(
    *,
    raw_dir: Path,
    config_path: Path,
    analysis_start_date: date,
    analysis_end_date: date,
    train_end_date: date,
    horizon_days: int,
    target_return: float,
    output_dir: Path,
    stock_master_path: Path | None = None,
    fundamentals_path: Path | None = None,
    shareholding_path: Path | None = None,
    sector_state_daily_path: Path | None = None,
    macro_daily_path: Path | None = None,
    announcements_path: Path | None = None,
    event_daily_path: Path | None = None,
    daily_facts_path: Path | None = None,
    base_universe_path: Path | None = None,
    min_price: float | None = 20.0,
) -> dict[str, object]:
    if base_universe_path and base_universe_path.exists():
        universe = _read_parquet_with_fallback(base_universe_path)
        if horizon_days >= 300 and target_return >= 0.5 and "winner_1y_50_flag" in universe.columns:
            universe["winner_flag"] = universe["winner_1y_50_flag"]
        elif "winner_flag" not in universe.columns and "winner_1y_50_flag" in universe.columns:
            universe["winner_flag"] = universe["winner_1y_50_flag"]
        if horizon_days >= 300 and target_return >= 0.5 and "forward_1y_return" in universe.columns:
            universe["forward_return"] = universe["forward_1y_return"]
        elif "forward_return" not in universe.columns and "forward_1y_return" in universe.columns:
            universe["forward_return"] = universe["forward_1y_return"]
        if "breadth_above_50_dma" not in universe.columns:
            universe = add_market_regime_features(universe)
        if "avg_traded_value_20d_cr" not in universe.columns:
            universe = add_bucket_columns(universe)
    else:
        if daily_facts_path and daily_facts_path.exists():
            daily_facts = _read_parquet_with_fallback(daily_facts_path)
        else:
            daily_facts = build_stock_daily_facts(raw_dir)
        labels = build_forward_return_labels(
            daily_facts,
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
            horizon_days=horizon_days,
            target_return=target_return,
            min_price=min_price,
        )

        stock_master = _read_stock_master(stock_master_path)
        fundamentals = _read_optional_table(fundamentals_path)
        shareholding = _read_optional_table(shareholding_path)
        sector_state_daily = _read_optional_table(sector_state_daily_path)
        config = _read_config(config_path)

        universe = build_daily_screen_universe(
            daily_facts=labels,
            stock_master=stock_master,
            fundamentals=fundamentals,
            shareholding=shareholding,
            sector_state_daily=sector_state_daily,
            config=config,
            include_missing_inputs=False,
        )
        universe = add_market_regime_features(universe)
        universe = add_bucket_columns(universe)

    fundamentals = _read_optional_table_fallback(fundamentals_path)
    if not fundamentals.empty and "effective_from_date" in fundamentals.columns:
        universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.normalize()
        fundamentals["effective_from_date"] = pd.to_datetime(fundamentals["effective_from_date"]).dt.normalize()
        universe["symbol"] = universe["symbol"].astype(str)
        fundamentals["symbol"] = fundamentals["symbol"].astype(str)
        universe = latest_effective_join(
            universe,
            fundamentals,
            left_date_col="trade_date",
            right_date_col="effective_from_date",
            by="symbol",
        )
    shareholding = _read_optional_table_fallback(shareholding_path)
    if not shareholding.empty and "effective_from_date" in shareholding.columns:
        universe["trade_date"] = pd.to_datetime(universe["trade_date"]).dt.normalize()
        shareholding["effective_from_date"] = pd.to_datetime(shareholding["effective_from_date"]).dt.normalize()
        universe["symbol"] = universe["symbol"].astype(str)
        shareholding["symbol"] = shareholding["symbol"].astype(str)
        universe = latest_effective_join(
            universe,
            shareholding,
            left_date_col="trade_date",
            right_date_col="effective_from_date",
            by="symbol",
        )

    macro_daily = _read_optional_table_fallback(macro_daily_path)
    if not macro_daily.empty:
        macro_daily["trade_date"] = pd.to_datetime(macro_daily["trade_date"]).dt.normalize()
        universe = universe.merge(macro_daily, on="trade_date", how="left")

    events_daily = _read_optional_table_fallback(event_daily_path)
    if events_daily.empty and announcements_path and announcements_path.exists():
        announcements = _read_optional_table_fallback(announcements_path)
        events_daily = build_event_feature_daily(universe[["symbol", "trade_date"]], announcements)
        if not events_daily.empty:
            write_parquet(events_daily, output_dir / "event_feature_daily.parquet")
    if not events_daily.empty:
        events_daily["trade_date"] = pd.to_datetime(events_daily["trade_date"]).dt.normalize()
        universe = universe.merge(events_daily, on=["symbol", "trade_date"], how="left")

    universe = universe.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    train_mask = universe["trade_date"].le(pd.Timestamp(train_end_date))
    test_mask = ~train_mask
    baseline_test = float(universe.loc[test_mask, "winner_flag"].mean()) if test_mask.any() else None

    results = []
    layer_summaries: list[dict[str, object]] = []
    for layer in ["price", "macro", "event", "filing"]:
        layer_features = [feature for feature in ALL_FEATURES if feature.layer == layer]
        layer_results = [
            _evaluate_feature(universe, feature, train_mask=train_mask, test_mask=test_mask, baseline_test=baseline_test)
            for feature in layer_features
            if feature.column in universe.columns
        ]
        results.extend(layer_results)
        layer_summaries.append(_summarize_layer(layer, layer_features, layer_results))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(universe, output_dir / "anchor_universe.parquet")
    pd.DataFrame(results).to_csv(output_dir / "layer_feature_results.csv", index=False)
    pd.DataFrame(layer_summaries).to_csv(output_dir / "layer_summary.csv", index=False)

    payload = {
        "summary": {
            "analysis_start_date": analysis_start_date.isoformat(),
            "analysis_end_date": analysis_end_date.isoformat(),
            "train_end_date": train_end_date.isoformat(),
            "horizon_days": horizon_days,
            "target_return": target_return,
            "anchor_count": int(len(universe)),
            "unique_symbols": int(universe["symbol"].nunique()),
            "winner_rate_train": _safe_ratio(int(universe.loc[train_mask, "winner_flag"].sum()), int(train_mask.sum())),
            "winner_rate_test": _safe_ratio(int(universe.loc[test_mask, "winner_flag"].sum()), int(test_mask.sum())),
        },
        "layer_summary": layer_summaries,
        "feature_results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _evaluate_feature(
    frame: pd.DataFrame,
    feature: LayerFeature,
    *,
    train_mask: pd.Series,
    test_mask: pd.Series,
    baseline_test: float | None,
) -> dict[str, object]:
    if feature.feature_type == "boolean":
        return _evaluate_boolean_feature(frame, feature, train_mask=train_mask, test_mask=test_mask, baseline_test=baseline_test)
    return _evaluate_numeric_feature(frame, feature, train_mask=train_mask, test_mask=test_mask, baseline_test=baseline_test)


def _evaluate_boolean_feature(
    frame: pd.DataFrame,
    feature: LayerFeature,
    *,
    train_mask: pd.Series,
    test_mask: pd.Series,
    baseline_test: float | None,
) -> dict[str, object]:
    valid_train = train_mask & frame[feature.column].notna()
    valid_test = test_mask & frame[feature.column].notna()
    passed_train = valid_train & frame[feature.column].astype("object").eq(True)
    passed_test = valid_test & frame[feature.column].astype("object").eq(True)
    test_returns = pd.to_numeric(frame.loc[passed_test, "forward_return"], errors="coerce")
    return {
        "layer": feature.layer,
        "column": feature.column,
        "feature_type": feature.feature_type,
        "direction": feature.direction,
        "selected_threshold": True,
        "train_pass_rows": int(passed_train.sum()),
        "test_pass_rows": int(passed_test.sum()),
        "train_precision": _to_float(frame.loc[passed_train, "winner_flag"].mean()),
        "test_precision": _to_float(frame.loc[passed_test, "winner_flag"].mean()),
        "test_lift": _safe_divide(_to_float(frame.loc[passed_test, "winner_flag"].mean()), baseline_test),
        "test_coverage": _safe_ratio(int(passed_test.sum()), int(valid_test.sum())),
        "test_avg_return": _to_float(test_returns.mean()),
        "test_median_return": _to_float(test_returns.median()),
    }


def _evaluate_numeric_feature(
    frame: pd.DataFrame,
    feature: LayerFeature,
    *,
    train_mask: pd.Series,
    test_mask: pd.Series,
    baseline_test: float | None,
) -> dict[str, object]:
    values = pd.to_numeric(frame[feature.column], errors="coerce")
    valid_train = frame.loc[train_mask & values.notna(), [feature.column, "winner_flag"]].copy()
    valid_train[feature.column] = values.loc[train_mask & values.notna()]
    train_baseline = float(valid_train["winner_flag"].mean()) if len(valid_train) else 0.0
    selected = _select_numeric_threshold(
        valid_train,
        column=feature.column,
        direction=feature.direction,
        winner_col="winner_flag",
        baseline=train_baseline,
    )
    threshold = selected.get("threshold")
    valid_test = test_mask & values.notna()
    if threshold is None:
        passed_test = pd.Series(False, index=frame.index)
    elif feature.direction == "ge":
        passed_test = valid_test & values.ge(float(threshold))
    else:
        passed_test = valid_test & values.le(float(threshold))
    test_returns = pd.to_numeric(frame.loc[passed_test, "forward_return"], errors="coerce")
    return {
        "layer": feature.layer,
        "column": feature.column,
        "feature_type": feature.feature_type,
        "direction": feature.direction,
        "selected_threshold": threshold,
        "train_pass_rows": None if threshold is None else int(
            ((values.ge(float(threshold)) if feature.direction == "ge" else values.le(float(threshold))) & train_mask).sum()
        ),
        "test_pass_rows": int(passed_test.sum()),
        "train_precision": selected.get("precision"),
        "test_precision": _to_float(frame.loc[passed_test, "winner_flag"].mean()),
        "test_lift": _safe_divide(_to_float(frame.loc[passed_test, "winner_flag"].mean()), baseline_test),
        "test_coverage": _safe_ratio(int(passed_test.sum()), int(valid_test.sum())),
        "test_avg_return": _to_float(test_returns.mean()),
        "test_median_return": _to_float(test_returns.median()),
    }


def _summarize_layer(
    layer: str,
    features: list[LayerFeature],
    results: list[dict[str, object]],
) -> dict[str, object]:
    usable = [row for row in results if row.get("test_precision") is not None]
    best = max(usable, key=lambda row: (row.get("test_lift") or 0.0, row.get("test_precision") or 0.0), default=None)
    summary = LayerSummary(
        layer=layer,
        feature_count=len(features),
        available_feature_count=len(results),
        best_test_feature=None if best is None else str(best["column"]),
        best_test_precision=None if best is None else _to_float(best["test_precision"]),
        best_test_lift=None if best is None else _to_float(best["test_lift"]),
        best_test_avg_return=None if best is None else _to_float(best["test_avg_return"]),
        best_test_median_return=None if best is None else _to_float(best["test_median_return"]),
    )
    return asdict(summary)


def _to_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _read_optional_table_fallback(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return _read_parquet_with_fallback(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _read_parquet_with_fallback(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--config-path", default="configs/screening.yaml")
    parser.add_argument("--analysis-start-date", required=True)
    parser.add_argument("--analysis-end-date", required=True)
    parser.add_argument("--train-end-date", required=True)
    parser.add_argument("--horizon-days", required=True, type=int)
    parser.add_argument("--target-return", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stock-master-path", default="")
    parser.add_argument("--fundamentals-path", default="")
    parser.add_argument("--shareholding-path", default="")
    parser.add_argument("--sector-state-daily-path", default="")
    parser.add_argument("--macro-daily-path", default="")
    parser.add_argument("--announcements-path", default="")
    parser.add_argument("--event-daily-path", default="")
    parser.add_argument("--daily-facts-path", default="")
    parser.add_argument("--base-universe-path", default="")
    parser.add_argument("--min-price", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_layer_edge_study(
        raw_dir=Path(args.raw_dir),
        config_path=Path(args.config_path),
        analysis_start_date=date.fromisoformat(args.analysis_start_date),
        analysis_end_date=date.fromisoformat(args.analysis_end_date),
        train_end_date=date.fromisoformat(args.train_end_date),
        horizon_days=args.horizon_days,
        target_return=args.target_return,
        output_dir=Path(args.output_dir),
        stock_master_path=Path(args.stock_master_path) if args.stock_master_path else None,
        fundamentals_path=Path(args.fundamentals_path) if args.fundamentals_path else None,
        shareholding_path=Path(args.shareholding_path) if args.shareholding_path else None,
        sector_state_daily_path=Path(args.sector_state_daily_path) if args.sector_state_daily_path else None,
        macro_daily_path=Path(args.macro_daily_path) if args.macro_daily_path else None,
        announcements_path=Path(args.announcements_path) if args.announcements_path else None,
        event_daily_path=Path(args.event_daily_path) if args.event_daily_path else None,
        daily_facts_path=Path(args.daily_facts_path) if args.daily_facts_path else None,
        base_universe_path=Path(args.base_universe_path) if args.base_universe_path else None,
        min_price=args.min_price,
    )


if __name__ == "__main__":
    main()
