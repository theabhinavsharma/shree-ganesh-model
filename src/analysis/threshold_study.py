from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from src.master.stock_master import load_stock_master
from src.screen.build_universe import build_daily_screen_universe
from src.transform.build_daily_facts import build_stock_daily_facts
from src.utils.io import write_parquet

Direction = Literal["ge", "le"]


NUMERIC_RULES: tuple[tuple[str, Direction], ...] = (
    ("volume_vs_20d", "ge"),
    ("rsi_14_daily", "ge"),
    ("rsi_14_weekly", "ge"),
    ("rsi_14_monthly", "ge"),
    ("promoter_pct", "ge"),
    ("revenue_cagr_5y", "ge"),
    ("pat_cagr_5y", "ge"),
    ("pe_ttm", "le"),
)

BOOLEAN_RULES: tuple[str, ...] = (
    "filter_above_50_dma",
    "filter_above_200_dma",
    "volume_high_63d_flag",
    "delivery_pct_high_63d_flag",
    "ebitda_positive_last_5q_flag",
)


@dataclass(frozen=True)
class ThresholdStudySummary:
    analysis_start_date: str
    analysis_end_date: str
    anchor_count: int
    unique_symbols: int
    winner_count: int
    winner_rate: float | None
    winner_unique_symbols: int
    breakout_dimension: str


def build_forward_1y_labels(
    daily_facts: pd.DataFrame,
    *,
    analysis_start_date: date,
    analysis_end_date: date,
    horizon_days: int = 365,
    min_price: float | None = None,
) -> pd.DataFrame:
    ordered = daily_facts.sort_values(["symbol", "trade_date"]).copy()
    ordered["trade_date"] = pd.to_datetime(ordered["trade_date"]).dt.normalize()
    ordered["horizon_target_date"] = ordered["trade_date"] + pd.to_timedelta(horizon_days, unit="D")

    pieces: list[pd.DataFrame] = []
    for symbol, symbol_df in ordered.groupby("symbol", sort=False):
        left = symbol_df.sort_values("horizon_target_date").copy()
        right = symbol_df[["trade_date", "close"]].rename(
            columns={"trade_date": "forward_trade_date", "close": "forward_close"}
        )
        merged = pd.merge_asof(
            left,
            right.sort_values("forward_trade_date"),
            left_on="horizon_target_date",
            right_on="forward_trade_date",
            direction="forward",
            allow_exact_matches=True,
        )
        merged["symbol"] = symbol
        pieces.append(merged)
    labeled = pd.concat(pieces, ignore_index=True) if pieces else ordered.iloc[0:0].copy()
    labeled["forward_1y_return"] = labeled["forward_close"] / labeled["close"] - 1
    labeled["winner_1y_50_flag"] = labeled["forward_1y_return"] >= 0.50

    mask = labeled["trade_date"].between(pd.Timestamp(analysis_start_date), pd.Timestamp(analysis_end_date))
    if min_price is not None:
        mask &= labeled["close"].ge(min_price)
    return labeled.loc[mask].reset_index(drop=True)


def run_threshold_study(
    *,
    raw_dir: Path,
    config_path: Path,
    stock_master_path: Path | None,
    fundamentals_path: Path | None,
    shareholding_path: Path | None,
    sector_state_daily_path: Path | None,
    analysis_start_date: date,
    analysis_end_date: date,
    output_dir: Path,
    breakout_dimension: str = "industry",
    min_price: float | None = 20.0,
) -> dict[str, object]:
    daily_facts = build_stock_daily_facts(raw_dir)
    labels = build_forward_1y_labels(
        daily_facts,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        min_price=min_price,
    )

    stock_master = _read_optional_table(stock_master_path)
    if stock_master.empty:
        stock_master = pd.DataFrame(columns=["symbol", "sector", "industry"])
    else:
        stock_master = _coerce_stock_master(stock_master_path)

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
    )
    universe = universe.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    winners = universe[universe["winner_1y_50_flag"].fillna(False)].copy()
    numeric_stats = _summarize_numeric_rules(universe)
    boolean_stats = _summarize_boolean_rules(universe)
    breakout = _build_breakout(winners, breakout_dimension)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(universe, output_dir / "anchor_universe.parquet")
    winners.to_csv(output_dir / "winner_start_rows.csv", index=False)
    pd.DataFrame(numeric_stats).to_csv(output_dir / "numeric_rule_study.csv", index=False)
    pd.DataFrame(boolean_stats).to_csv(output_dir / "boolean_rule_study.csv", index=False)
    breakout.to_csv(output_dir / f"winner_breakout_by_{breakout_dimension}.csv", index=False)

    summary = ThresholdStudySummary(
        analysis_start_date=analysis_start_date.isoformat(),
        analysis_end_date=analysis_end_date.isoformat(),
        anchor_count=int(len(universe)),
        unique_symbols=int(universe["symbol"].nunique()),
        winner_count=int(winners["winner_1y_50_flag"].sum()),
        winner_rate=_safe_ratio(int(winners["winner_1y_50_flag"].sum()), int(len(universe))),
        winner_unique_symbols=int(winners["symbol"].nunique()),
        breakout_dimension=breakout_dimension,
    )
    payload = {
        "summary": asdict(summary),
        "numeric_rules": numeric_stats,
        "boolean_rules": boolean_stats,
        "winner_breakout": breakout.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _read_optional_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _coerce_stock_master(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["symbol", "sector", "industry"])
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        expected = ["symbol", "sector", "industry"]
        for column in expected:
            if column not in df.columns:
                df[column] = pd.NA
        return df[expected]
    return load_stock_master(path)


def _read_config(path: Path) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _summarize_numeric_rules(universe: pd.DataFrame) -> list[dict[str, object]]:
    baseline = float(universe["winner_1y_50_flag"].mean()) if len(universe) else 0.0
    stats: list[dict[str, object]] = []
    winners = universe[universe["winner_1y_50_flag"].fillna(False)]
    for column, direction in NUMERIC_RULES:
        if column not in universe.columns:
            continue
        series = pd.to_numeric(universe[column], errors="coerce")
        winner_series = pd.to_numeric(winners[column], errors="coerce")
        valid = universe.loc[series.notna(), [column, "winner_1y_50_flag"]].copy()
        if valid.empty:
            stats.append(
                {
                    "rule": column,
                    "direction": direction,
                    "available_rows": 0,
                    "winner_rows": 0,
                    "winner_mean": None,
                    "winner_median": None,
                    "winner_p25": None,
                    "winner_p75": None,
                    "recommended_threshold": None,
                    "recommended_precision": None,
                    "recommended_lift": None,
                    "recommended_coverage": None,
                    "selection_method": "insufficient_data",
                }
            )
            continue

        threshold_result = _select_numeric_threshold(valid, column=column, direction=direction, baseline=baseline)
        stats.append(
            {
                "rule": column,
                "direction": direction,
                "available_rows": int(len(valid)),
                "winner_rows": int(winner_series.notna().sum()),
                "winner_mean": _maybe_float(winner_series.mean()),
                "winner_median": _maybe_float(winner_series.median()),
                "winner_p25": _maybe_float(winner_series.quantile(0.25)),
                "winner_p75": _maybe_float(winner_series.quantile(0.75)),
                "recommended_threshold": threshold_result["threshold"],
                "recommended_precision": threshold_result["precision"],
                "recommended_lift": threshold_result["lift"],
                "recommended_coverage": threshold_result["coverage"],
                "selection_method": threshold_result["selection_method"],
            }
        )
    return stats


def _summarize_boolean_rules(universe: pd.DataFrame) -> list[dict[str, object]]:
    baseline = float(universe["winner_1y_50_flag"].mean()) if len(universe) else 0.0
    stats: list[dict[str, object]] = []
    for column in BOOLEAN_RULES:
        if column not in universe.columns:
            continue
        series = universe[column]
        valid = series.notna()
        if not valid.any():
            stats.append(
                {
                    "rule": column,
                    "available_rows": 0,
                    "pass_rows": 0,
                    "pass_rate": None,
                    "winner_rate_when_true": None,
                    "winner_lift_when_true": None,
                }
            )
            continue

        valid_df = universe.loc[valid, ["winner_1y_50_flag"]].copy()
        valid_df["rule_value"] = series.loc[valid].astype(bool)
        passed = valid_df[valid_df["rule_value"]]
        precision = _maybe_float(passed["winner_1y_50_flag"].mean())
        stats.append(
            {
                "rule": column,
                "available_rows": int(len(valid_df)),
                "pass_rows": int(len(passed)),
                "pass_rate": _safe_ratio(int(len(passed)), int(len(valid_df))),
                "winner_rate_when_true": precision,
                "winner_lift_when_true": _safe_divide(precision, baseline),
            }
        )
    return stats


def _select_numeric_threshold(
    valid: pd.DataFrame,
    *,
    column: str,
    direction: Direction,
    baseline: float,
    min_coverage: float = 0.05,
) -> dict[str, object]:
    values = pd.to_numeric(valid[column], errors="coerce")
    candidates = sorted({float(v) for v in values.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).dropna()})
    if not candidates:
        return {
            "threshold": None,
            "precision": None,
            "lift": None,
            "coverage": None,
            "selection_method": "insufficient_data",
        }

    total = len(valid)
    total_winners = int(valid["winner_1y_50_flag"].sum())
    best: dict[str, object] | None = None
    for threshold in candidates:
        if direction == "ge":
            passed = valid[values >= threshold]
        else:
            passed = valid[values <= threshold]
        if passed.empty:
            continue
        coverage = len(passed) / total
        if coverage < min_coverage:
            continue
        precision = float(passed["winner_1y_50_flag"].mean())
        recall = _safe_ratio(int(passed["winner_1y_50_flag"].sum()), total_winners)
        score = _f1(precision, recall)
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "lift": _safe_divide(precision, baseline),
            "coverage": coverage,
            "score": score,
            "selection_method": "max_f1_with_min_coverage_5pct",
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    if best is None:
        return {
            "threshold": None,
            "precision": None,
            "lift": None,
            "coverage": None,
            "selection_method": "no_candidate_after_coverage_filter",
        }
    best.pop("score", None)
    return best


def _build_breakout(winners: pd.DataFrame, breakout_dimension: str) -> pd.DataFrame:
    if winners.empty:
        return pd.DataFrame(
            columns=[
                breakout_dimension,
                "winner_rows",
                "winner_symbols",
                "median_forward_1y_return",
                "mean_forward_1y_return",
            ]
        )

    dimension = breakout_dimension if breakout_dimension in winners.columns else None
    if dimension is None:
        fallback = "sector" if "sector" in winners.columns else "industry" if "industry" in winners.columns else None
        dimension = fallback
    if dimension is None:
        breakout = winners.copy()
        breakout["unclassified"] = "Unclassified"
        dimension = "unclassified"

    grouped = winners.groupby(dimension, dropna=False)
    breakout = (
        grouped.agg(
            winner_rows=("symbol", "size"),
            winner_symbols=("symbol", "nunique"),
            median_forward_1y_return=("forward_1y_return", "median"),
            mean_forward_1y_return=("forward_1y_return", "mean"),
        )
        .reset_index()
        .sort_values(["winner_rows", "winner_symbols"], ascending=[False, False])
    )
    breakout[dimension] = breakout[dimension].fillna("Unclassified")
    return breakout


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    return numerator / denominator


def _f1(precision: float | None, recall: float | None) -> float:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw/nse_live_shortlist")
    parser.add_argument("--config-path", default="configs/screening.yaml")
    parser.add_argument("--stock-master-path", default="")
    parser.add_argument("--fundamentals-path", default="")
    parser.add_argument("--shareholding-path", default="")
    parser.add_argument("--sector-state-daily-path", default="")
    parser.add_argument("--analysis-start-date", required=True)
    parser.add_argument("--analysis-end-date", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--breakout-dimension", default="industry")
    parser.add_argument("--min-price", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_threshold_study(
        raw_dir=Path(args.raw_dir),
        config_path=Path(args.config_path),
        stock_master_path=Path(args.stock_master_path) if args.stock_master_path else None,
        fundamentals_path=Path(args.fundamentals_path) if args.fundamentals_path else None,
        shareholding_path=Path(args.shareholding_path) if args.shareholding_path else None,
        sector_state_daily_path=Path(args.sector_state_daily_path) if args.sector_state_daily_path else None,
        analysis_start_date=date.fromisoformat(args.analysis_start_date),
        analysis_end_date=date.fromisoformat(args.analysis_end_date),
        output_dir=Path(args.output_dir),
        breakout_dimension=args.breakout_dimension,
        min_price=args.min_price,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
