from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from src.master.stock_master import load_stock_master
from src.screen.build_universe import build_daily_screen_universe
from src.transform.build_daily_facts import build_stock_daily_facts
from src.utils.io import write_parquet

Direction = Literal["ge", "le"]


NUMERIC_COLUMNS: tuple[tuple[str, Direction], ...] = (
    ("close", "ge"),
    ("return_20d", "ge"),
    ("volume_vs_20d", "ge"),
    ("traded_value_vs_20d", "ge"),
    ("delivery_pct", "ge"),
    ("delivery_pct_vs_20d", "ge"),
    ("rsi_14_daily", "ge"),
    ("rsi_14_weekly", "ge"),
    ("rsi_14_monthly", "ge"),
    ("avg_traded_value_20d_cr", "ge"),
    ("breadth_above_50_dma", "ge"),
    ("breadth_above_200_dma", "ge"),
    ("breadth_rsi_60", "ge"),
    ("breadth_volume_1_5x", "ge"),
    ("market_median_return_20d", "ge"),
    ("promoter_pct", "ge"),
    ("revenue_cagr_5y", "ge"),
    ("pat_cagr_5y", "ge"),
    ("pe_ttm", "le"),
)

BOOLEAN_COLUMNS: tuple[str, ...] = (
    "filter_above_50_dma",
    "filter_above_200_dma",
    "volume_high_63d_flag",
    "delivery_pct_high_63d_flag",
    "filter_rsi_daily",
    "filter_rsi_weekly",
    "filter_rsi_monthly",
    "filter_promoter_holding",
    "filter_revenue_growth",
    "filter_profit_cagr",
    "filter_ebitda_positive",
    "filter_pe",
)


@dataclass(frozen=True)
class ForwardStudySummary:
    analysis_start_date: str
    analysis_end_date: str
    horizon_days: int
    target_return: float
    anchor_count: int
    unique_symbols: int
    winner_count: int
    winner_rate: float | None
    winner_unique_symbols: int


def build_forward_return_labels(
    daily_facts: pd.DataFrame,
    *,
    analysis_start_date: date,
    analysis_end_date: date,
    horizon_days: int,
    target_return: float,
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
    labeled["forward_return"] = labeled["forward_close"] / labeled["close"] - 1
    labeled["winner_flag"] = labeled["forward_return"] >= target_return

    mask = labeled["trade_date"].between(pd.Timestamp(analysis_start_date), pd.Timestamp(analysis_end_date))
    if min_price is not None:
        mask &= labeled["close"].ge(min_price)
    return labeled.loc[mask].reset_index(drop=True)


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["above_50_dma_raw"] = working["close"].gt(working["sma_50"]).where(
        working["close"].notna() & working["sma_50"].notna()
    )
    working["above_200_dma_raw"] = working["close"].gt(working["sma_200"]).where(
        working["close"].notna() & working["sma_200"].notna()
    )
    working["rsi_60_raw"] = pd.to_numeric(working["rsi_14_daily"], errors="coerce").ge(60).where(
        pd.to_numeric(working["rsi_14_daily"], errors="coerce").notna()
    )
    working["volume_1_5x_raw"] = pd.to_numeric(working["volume_vs_20d"], errors="coerce").ge(1.5).where(
        pd.to_numeric(working["volume_vs_20d"], errors="coerce").notna()
    )

    regime_rows: list[dict[str, object]] = []
    for trade_date, group in working.groupby("trade_date", sort=True):
        regime_rows.append(
            {
                "trade_date": trade_date,
                "breadth_above_50_dma": _mean_bool(group["above_50_dma_raw"]),
                "breadth_above_200_dma": _mean_bool(group["above_200_dma_raw"]),
                "breadth_rsi_60": _mean_bool(group["rsi_60_raw"]),
                "breadth_volume_1_5x": _mean_bool(group["volume_1_5x_raw"]),
                "market_median_return_20d": _maybe_float(pd.to_numeric(group["return_20d"], errors="coerce").median()),
                "market_median_volume_vs_20d": _maybe_float(pd.to_numeric(group["volume_vs_20d"], errors="coerce").median()),
                "market_median_delivery_pct": _maybe_float(pd.to_numeric(group.get("delivery_pct"), errors="coerce").median()),
                "universe_size": int(group["symbol"].nunique()),
            }
        )

    regime = pd.DataFrame(regime_rows)
    return working.drop(
        columns=["above_50_dma_raw", "above_200_dma_raw", "rsi_60_raw", "volume_1_5x_raw"]
    ).merge(regime, on="trade_date", how="left")


def add_bucket_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["avg_traded_value_20d_cr"] = pd.to_numeric(enriched.get("avg_traded_value_20d"), errors="coerce") / 1e7
    enriched["price_bucket"] = pd.cut(
        pd.to_numeric(enriched["close"], errors="coerce"),
        bins=[0, 50, 200, 500, 2000, float("inf")],
        labels=["<50", "50-200", "200-500", "500-2000", ">2000"],
    )
    enriched["liquidity_bucket_20d_cr"] = pd.cut(
        enriched["avg_traded_value_20d_cr"],
        bins=[0, 1, 5, 20, 100, float("inf")],
        labels=["<1", "1-5", "5-20", "20-100", ">100"],
    )
    enriched["breadth_50_bucket"] = pd.cut(
        pd.to_numeric(enriched["breadth_above_50_dma"], errors="coerce"),
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["<=30%", "30-50%", "50-70%", ">70%"],
        include_lowest=True,
    )
    return enriched


def run_forward_return_study(
    *,
    raw_dir: Path,
    config_path: Path,
    stock_master_path: Path | None,
    fundamentals_path: Path | None,
    shareholding_path: Path | None,
    sector_state_daily_path: Path | None,
    analysis_start_date: date,
    analysis_end_date: date,
    horizon_days: int,
    target_return: float,
    output_dir: Path,
    min_price: float | None = 20.0,
) -> dict[str, object]:
    daily_facts = build_stock_daily_facts(raw_dir)
    labeled = build_forward_return_labels(
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
        daily_facts=labeled,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=sector_state_daily,
        config=config,
        include_missing_inputs=False,
    )
    universe = add_market_regime_features(universe)
    universe = add_bucket_columns(universe)
    winners = universe[universe["winner_flag"].fillna(False)].copy()

    summary = ForwardStudySummary(
        analysis_start_date=analysis_start_date.isoformat(),
        analysis_end_date=analysis_end_date.isoformat(),
        horizon_days=horizon_days,
        target_return=target_return,
        anchor_count=int(len(universe)),
        unique_symbols=int(universe["symbol"].nunique()),
        winner_count=int(winners["winner_flag"].sum()),
        winner_rate=_safe_ratio(int(winners["winner_flag"].sum()), int(len(universe))),
        winner_unique_symbols=int(winners["symbol"].nunique()),
    )

    numeric_stats = _summarize_numeric_columns(universe, "winner_flag")
    boolean_stats = _summarize_boolean_columns(universe, "winner_flag")
    price_bucket_breakout = _breakout(universe, "price_bucket", "winner_flag")
    liquidity_breakout = _breakout(universe, "liquidity_bucket_20d_cr", "winner_flag")
    breadth_breakout = _breakout(universe, "breadth_50_bucket", "winner_flag")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(universe, output_dir / "anchor_universe.parquet")
    winners.to_csv(output_dir / "winner_start_rows.csv", index=False)
    pd.DataFrame(numeric_stats).to_csv(output_dir / "numeric_columns.csv", index=False)
    pd.DataFrame(boolean_stats).to_csv(output_dir / "boolean_columns.csv", index=False)
    price_bucket_breakout.to_csv(output_dir / "price_bucket_breakout.csv", index=False)
    liquidity_breakout.to_csv(output_dir / "liquidity_bucket_breakout.csv", index=False)
    breadth_breakout.to_csv(output_dir / "breadth_50_bucket_breakout.csv", index=False)

    payload = {
        "summary": asdict(summary),
        "numeric_columns": numeric_stats,
        "boolean_columns": boolean_stats,
        "price_bucket_breakout": price_bucket_breakout.to_dict(orient="records"),
        "liquidity_bucket_breakout": liquidity_breakout.to_dict(orient="records"),
        "breadth_breakout": breadth_breakout.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(summary, numeric_stats, boolean_stats), encoding="utf-8")
    return payload


def _read_optional_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table type: {path}")


def _read_stock_master(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(
            columns=[
                "symbol",
                "sector",
                "industry",
                "basic_industry",
                "instrument_type",
                "company_name",
                "issued_size",
            ]
        )
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        preferred_columns = [
            "symbol",
            "sector",
            "industry",
            "basic_industry",
            "instrument_type",
            "company_name",
            "issued_size",
        ]
        for column in preferred_columns:
            if column not in df.columns:
                df[column] = pd.NA
        return df[preferred_columns].copy()
    return load_stock_master(path)


def _read_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _summarize_numeric_columns(df: pd.DataFrame, winner_col: str) -> list[dict[str, object]]:
    baseline = _maybe_float(df[winner_col].mean()) or 0.0
    winners = df[df[winner_col].fillna(False)]
    rows: list[dict[str, object]] = []
    for column, direction in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        winner_series = pd.to_numeric(winners[column], errors="coerce")
        valid = df.loc[series.notna(), [winner_col]].copy()
        valid[column] = series.loc[series.notna()]
        if valid.empty:
            rows.append(
                {
                    "column": column,
                    "direction": direction,
                    "available_rows": 0,
                    "winner_mean": None,
                    "winner_median": None,
                    "winner_p25": None,
                    "winner_p75": None,
                    "recommended_threshold": None,
                    "recommended_precision": None,
                    "recommended_lift": None,
                    "recommended_coverage": None,
                }
            )
            continue
        selected = _select_numeric_threshold(valid, column=column, direction=direction, winner_col=winner_col, baseline=baseline)
        rows.append(
            {
                "column": column,
                "direction": direction,
                "available_rows": int(len(valid)),
                "winner_mean": _maybe_float(winner_series.mean()),
                "winner_median": _maybe_float(winner_series.median()),
                "winner_p25": _maybe_float(winner_series.quantile(0.25)),
                "winner_p75": _maybe_float(winner_series.quantile(0.75)),
                "recommended_threshold": selected["threshold"],
                "recommended_precision": selected["precision"],
                "recommended_lift": selected["lift"],
                "recommended_coverage": selected["coverage"],
            }
        )
    return rows


def _summarize_boolean_columns(df: pd.DataFrame, winner_col: str) -> list[dict[str, object]]:
    baseline = _maybe_float(df[winner_col].mean()) or 0.0
    rows: list[dict[str, object]] = []
    for column in BOOLEAN_COLUMNS:
        if column not in df.columns:
            continue
        valid = df[column].notna() & df[winner_col].notna()
        if not valid.any():
            continue
        passed = valid & df[column].astype("object").eq(True)
        precision = _maybe_float(df.loc[passed, winner_col].mean()) if passed.any() else None
        rows.append(
            {
                "column": column,
                "available_rows": int(valid.sum()),
                "pass_rows": int(passed.sum()),
                "pass_rate": _safe_ratio(int(passed.sum()), int(valid.sum())),
                "winner_rate_when_true": precision,
                "winner_lift_when_true": _safe_divide(precision, baseline),
            }
        )
    return rows


def _select_numeric_threshold(
    valid: pd.DataFrame,
    *,
    column: str,
    direction: Direction,
    winner_col: str,
    baseline: float,
    min_coverage: float = 0.03,
) -> dict[str, object]:
    values = pd.to_numeric(valid[column], errors="coerce")
    candidates = sorted({float(v) for v in values.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).dropna()})
    if not candidates:
        return {"threshold": None, "precision": None, "lift": None, "coverage": None}

    total = len(valid)
    total_winners = int(valid[winner_col].sum())
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
        precision = float(passed[winner_col].mean())
        recall = _safe_ratio(int(passed[winner_col].sum()), total_winners)
        score = _f1(precision, recall)
        candidate = {
            "threshold": float(threshold),
            "precision": precision,
            "lift": _safe_divide(precision, baseline),
            "coverage": coverage,
            "score": score,
        }
        if best is None or float(candidate["score"]) > float(best["score"]):
            best = candidate
    if best is None:
        return {"threshold": None, "precision": None, "lift": None, "coverage": None}
    best.pop("score", None)
    return best


def _breakout(df: pd.DataFrame, bucket_col: str, winner_col: str) -> pd.DataFrame:
    if bucket_col not in df.columns:
        return pd.DataFrame(columns=[bucket_col, "rows", "winners", "winner_rate", "median_forward_return"])
    working = df[df[bucket_col].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=[bucket_col, "rows", "winners", "winner_rate", "median_forward_return"])
    grouped = (
        working.groupby(bucket_col, dropna=False)
        .agg(
            rows=("symbol", "size"),
            winners=(winner_col, "sum"),
            winner_rate=(winner_col, "mean"),
            median_forward_return=("forward_return", "median"),
        )
        .reset_index()
    )
    grouped[bucket_col] = grouped[bucket_col].astype(str)
    return grouped.sort_values("rows", ascending=False)


def _render_report(
    summary: ForwardStudySummary,
    numeric_stats: list[dict[str, object]],
    boolean_stats: list[dict[str, object]],
) -> str:
    numeric_df = pd.DataFrame(numeric_stats)
    boolean_df = pd.DataFrame(boolean_stats)
    lines = [
        "# Forward Return Study",
        "",
        f"- Analysis window: {summary.analysis_start_date} to {summary.analysis_end_date}",
        f"- Horizon: {summary.horizon_days} calendar days",
        f"- Winner definition: forward return >= {summary.target_return:.0%}",
        f"- Anchor rows: {summary.anchor_count:,}",
        f"- Winner rows: {summary.winner_count:,}",
        f"- Winner rate: {summary.winner_rate:.2%}" if summary.winner_rate is not None else "- Winner rate: n/a",
        "",
        "## Numeric Columns",
        "",
        numeric_df.to_csv(index=False) if not numeric_df.empty else "No numeric columns available.",
        "",
        "## Boolean Columns",
        "",
        boolean_df.to_csv(index=False) if not boolean_df.empty else "No boolean columns available.",
        "",
    ]
    return "\n".join(lines)


def _maybe_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _mean_bool(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.astype(bool).mean())


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
    parser.add_argument("--horizon-days", type=int, required=True)
    parser.add_argument("--target-return", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-price", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_forward_return_study(
        raw_dir=Path(args.raw_dir),
        config_path=Path(args.config_path),
        stock_master_path=Path(args.stock_master_path) if args.stock_master_path else None,
        fundamentals_path=Path(args.fundamentals_path) if args.fundamentals_path else None,
        shareholding_path=Path(args.shareholding_path) if args.shareholding_path else None,
        sector_state_daily_path=Path(args.sector_state_daily_path) if args.sector_state_daily_path else None,
        analysis_start_date=date.fromisoformat(args.analysis_start_date),
        analysis_end_date=date.fromisoformat(args.analysis_end_date),
        horizon_days=args.horizon_days,
        target_return=args.target_return,
        output_dir=Path(args.output_dir),
        min_price=args.min_price,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
