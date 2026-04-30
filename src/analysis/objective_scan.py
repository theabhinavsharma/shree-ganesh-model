from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from src.analysis.forward_return_study import add_bucket_columns
from src.analysis.forward_return_study import add_market_regime_features
from src.analysis.forward_return_study import build_forward_return_labels
from src.screen.build_universe import build_daily_screen_universe
from src.transform.build_daily_facts import build_stock_daily_facts


@dataclass(frozen=True)
class ComboResult:
    conditions: str
    train_n: int
    train_precision: float
    test_n: int
    test_precision: float
    stable_precision: float
    avg_precision: float
    test_avg_return: float | None
    test_median_return: float | None
    test_p75_return: float | None


NUMERIC_CORR_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_20d",
    "volume_vs_20d",
    "traded_value_vs_20d",
    "delivery_pct",
    "delivery_pct_vs_20d",
    "rsi_14_daily",
    "rsi_14_weekly",
    "rsi_14_monthly",
    "avg_traded_value_20d_cr",
    "breadth_above_50_dma",
    "breadth_above_200_dma",
    "breadth_volume_1_5x",
    "market_median_return_20d",
)


def build_base_universe(
    *,
    raw_dir: Path,
    config_path: Path,
    analysis_start_date: date,
    analysis_end_date: date,
    min_price: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_facts = build_stock_daily_facts(raw_dir)
    base = daily_facts.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"]).dt.normalize()
    base = base.loc[
        base["trade_date"].between(pd.Timestamp(analysis_start_date), pd.Timestamp(analysis_end_date))
    ].copy()
    base = base.loc[pd.to_numeric(base["close"], errors="coerce").ge(min_price)].copy()

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    universe = build_daily_screen_universe(
        daily_facts=base,
        stock_master=pd.DataFrame(columns=["symbol", "sector", "industry"]),
        fundamentals=pd.DataFrame(),
        shareholding=pd.DataFrame(),
        sector_state_daily=pd.DataFrame(),
        config=config,
        include_missing_inputs=False,
    )
    universe = add_market_regime_features(universe)
    universe = add_bucket_columns(universe)
    universe = universe.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    return daily_facts, universe


def build_objective_frame(
    *,
    daily_facts: pd.DataFrame,
    base_universe: pd.DataFrame,
    analysis_start_date: date,
    analysis_end_date: date,
    horizon_days: int,
    target_return: float,
    min_price: float,
) -> pd.DataFrame:
    labels = build_forward_return_labels(
        daily_facts,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        horizon_days=horizon_days,
        target_return=target_return,
        min_price=min_price,
    )[
        ["symbol", "trade_date", "forward_trade_date", "forward_close", "forward_return", "winner_flag"]
    ].copy()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.normalize()

    merged = base_universe.drop(
        columns=["forward_trade_date", "forward_close", "forward_return", "winner_flag"],
        errors="ignore",
    ).merge(labels, on=["symbol", "trade_date"], how="inner")
    return merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def build_condition_arrays(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "above50": frame["filter_above_50_dma"].eq(True),
        "above200": frame["filter_above_200_dma"].eq(True),
        "ret20_ge_10": pd.to_numeric(frame["return_20d"], errors="coerce").ge(0.10),
        "ret20_ge_15": pd.to_numeric(frame["return_20d"], errors="coerce").ge(0.15),
        "ret20_ge_20": pd.to_numeric(frame["return_20d"], errors="coerce").ge(0.20),
        "ret20_ge_30": pd.to_numeric(frame["return_20d"], errors="coerce").ge(0.30),
        "vol_ge_1_5": pd.to_numeric(frame["volume_vs_20d"], errors="coerce").ge(1.5),
        "vol_ge_2": pd.to_numeric(frame["volume_vs_20d"], errors="coerce").ge(2.0),
        "vol_ge_3": pd.to_numeric(frame["volume_vs_20d"], errors="coerce").ge(3.0),
        "val_ge_1_5": pd.to_numeric(frame["traded_value_vs_20d"], errors="coerce").ge(1.5),
        "val_ge_2": pd.to_numeric(frame["traded_value_vs_20d"], errors="coerce").ge(2.0),
        "val_ge_3": pd.to_numeric(frame["traded_value_vs_20d"], errors="coerce").ge(3.0),
        "rsi_d_ge_60": pd.to_numeric(frame["rsi_14_daily"], errors="coerce").ge(60),
        "rsi_d_ge_70": pd.to_numeric(frame["rsi_14_daily"], errors="coerce").ge(70),
        "rsi_d_ge_75": pd.to_numeric(frame["rsi_14_daily"], errors="coerce").ge(75),
        "rsi_w_ge_55": pd.to_numeric(frame["rsi_14_weekly"], errors="coerce").ge(55),
        "rsi_w_ge_60": pd.to_numeric(frame["rsi_14_weekly"], errors="coerce").ge(60),
        "rsi_m_ge_40": pd.to_numeric(frame["rsi_14_monthly"], errors="coerce").ge(40),
        "rsi_m_ge_50": pd.to_numeric(frame["rsi_14_monthly"], errors="coerce").ge(50),
        "vol_high": frame["volume_high_63d_flag"].eq(True),
        "del_high": frame["delivery_pct_high_63d_flag"].eq(True),
        "breadth50_gt_70": pd.to_numeric(frame["breadth_above_50_dma"], errors="coerce").gt(0.70),
        "breadth50_gt_80": pd.to_numeric(frame["breadth_above_50_dma"], errors="coerce").gt(0.80),
        "breadthvol_gt_22": pd.to_numeric(frame["breadth_volume_1_5x"], errors="coerce").gt(0.22),
        "mktret20_gt_3": pd.to_numeric(frame["market_median_return_20d"], errors="coerce").gt(0.03),
    }


def build_universe_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    price = pd.to_numeric(frame["close"], errors="coerce")
    liq = pd.to_numeric(frame["avg_traded_value_20d_cr"], errors="coerce")
    return {
        "all_names": pd.Series(True, index=frame.index),
        "cheap_micro": price.lt(50) & liq.lt(1),
        "mid_small": price.ge(50) & price.lt(200) & liq.ge(1) & liq.lt(5),
        "liquid_5cr_plus": liq.ge(5),
        "liquid_20cr_plus": liq.ge(20),
    }


def _combo_mask(combo: tuple[tuple[str, pd.Series], ...]) -> pd.Series:
    mask = combo[0][1].fillna(False).astype(bool).copy()
    for _, series in combo[1:]:
        mask &= series.fillna(False).astype(bool)
    return mask


def find_best_combos(
    *,
    frame: pd.DataFrame,
    universe_mask: pd.Series,
    min_train_n: int,
    min_test_n: int,
    train_end_date: date,
) -> tuple[ComboResult | None, ComboResult | None]:
    frame = frame.copy()
    y = frame["winner_flag"].fillna(False).astype(bool)
    forward_return = pd.to_numeric(frame["forward_return"], errors="coerce")
    train_mask = frame["trade_date"].le(pd.Timestamp(train_end_date))
    test_mask = ~train_mask

    conditions = list(build_condition_arrays(frame).items())
    best_stable: ComboResult | None = None
    best_test: ComboResult | None = None

    for width in range(1, 5):
        for combo in itertools.combinations(conditions, width):
            mask = _combo_mask(combo) & universe_mask.fillna(False).astype(bool)
            tr = mask & train_mask
            te = mask & test_mask
            train_n = int(tr.sum())
            test_n = int(te.sum())
            if train_n < min_train_n or test_n < min_test_n:
                continue
            train_precision = float(y[tr].mean())
            test_precision = float(y[te].mean())
            stable_precision = min(train_precision, test_precision)
            avg_precision = (train_precision + test_precision) / 2.0
            test_returns = forward_return[te]
            result = ComboResult(
                conditions=" & ".join(name for name, _ in combo),
                train_n=train_n,
                train_precision=train_precision,
                test_n=test_n,
                test_precision=test_precision,
                stable_precision=stable_precision,
                avg_precision=avg_precision,
                test_avg_return=_maybe_float(test_returns.mean()),
                test_median_return=_maybe_float(test_returns.median()),
                test_p75_return=_maybe_float(test_returns.quantile(0.75)),
            )

            if best_stable is None or _stable_sort_key(result) > _stable_sort_key(best_stable):
                best_stable = result
            if best_test is None or _test_sort_key(result) > _test_sort_key(best_test):
                best_test = result

    return best_stable, best_test


def top_correlations(frame: pd.DataFrame, universe_mask: pd.Series, limit: int = 5) -> list[dict[str, float | str | None]]:
    scoped = frame.loc[universe_mask.fillna(False).astype(bool)].copy()
    rows: list[dict[str, float | str | None]] = []
    target = pd.to_numeric(scoped["forward_return"], errors="coerce")
    for column in NUMERIC_CORR_COLUMNS:
        if column not in scoped.columns:
            continue
        series = pd.to_numeric(scoped[column], errors="coerce")
        valid = pd.DataFrame({"x": series, "y": target}).dropna()
        if len(valid) < 100:
            continue
        corr = valid["x"].corr(valid["y"], method="spearman")
        rows.append(
            {
                "feature": column,
                "spearman_corr": _maybe_float(corr),
                "abs_spearman_corr": _maybe_float(abs(corr) if corr is not None else None),
            }
        )
    rows.sort(key=lambda row: (row["abs_spearman_corr"] or 0.0), reverse=True)
    return rows[:limit]


def run_objective_scan(
    *,
    raw_dir: Path,
    config_path: Path,
    analysis_start_date: date,
    analysis_end_date: date,
    train_end_date: date,
    horizons: list[int],
    targets: list[float],
    min_price: float,
    min_train_n: int,
    min_test_n: int,
    output_dir: Path,
) -> dict[str, object]:
    daily_facts, base_universe = build_base_universe(
        raw_dir=raw_dir,
        config_path=config_path,
        analysis_start_date=analysis_start_date,
        analysis_end_date=analysis_end_date,
        min_price=min_price,
    )

    summaries: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    objective_cache: dict[tuple[int, float], pd.DataFrame] = {}

    for horizon_days in horizons:
        for target_return in targets:
            objective_frame = build_objective_frame(
                daily_facts=daily_facts,
                base_universe=base_universe,
                analysis_start_date=analysis_start_date,
                analysis_end_date=analysis_end_date,
                horizon_days=horizon_days,
                target_return=target_return,
                min_price=min_price,
            )
            objective_cache[(horizon_days, target_return)] = objective_frame
            base_rate = _maybe_float(objective_frame["winner_flag"].fillna(False).mean())

            for universe_name, universe_mask in build_universe_masks(objective_frame).items():
                scoped = objective_frame.loc[universe_mask.fillna(False).astype(bool)]
                if scoped.empty:
                    continue
                best_stable, best_test = find_best_combos(
                    frame=objective_frame,
                    universe_mask=universe_mask,
                    min_train_n=min_train_n,
                    min_test_n=min_test_n,
                    train_end_date=train_end_date,
                )

                summaries.append(
                    {
                        "universe_name": universe_name,
                        "horizon_days": horizon_days,
                        "target_return": target_return,
                        "anchor_count": int(len(scoped)),
                        "unique_symbols": int(scoped["symbol"].nunique()),
                        "winner_rate": _maybe_float(scoped["winner_flag"].fillna(False).mean()),
                        "best_stable_conditions": best_stable.conditions if best_stable else None,
                        "best_stable_train_n": best_stable.train_n if best_stable else None,
                        "best_stable_train_precision": best_stable.train_precision if best_stable else None,
                        "best_stable_test_n": best_stable.test_n if best_stable else None,
                        "best_stable_test_precision": best_stable.test_precision if best_stable else None,
                        "best_stable_precision": best_stable.stable_precision if best_stable else None,
                        "best_stable_test_avg_return": best_stable.test_avg_return if best_stable else None,
                        "best_stable_test_median_return": best_stable.test_median_return if best_stable else None,
                        "best_test_conditions": best_test.conditions if best_test else None,
                        "best_test_train_n": best_test.train_n if best_test else None,
                        "best_test_train_precision": best_test.train_precision if best_test else None,
                        "best_test_test_n": best_test.test_n if best_test else None,
                        "best_test_test_precision": best_test.test_precision if best_test else None,
                        "base_rate": base_rate,
                    }
                )

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df.sort_values(
        ["universe_name", "best_stable_precision", "best_stable_test_n", "target_return", "horizon_days"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    for universe_name, group in summary_df.groupby("universe_name", sort=False):
        top = group.dropna(subset=["best_stable_precision"]).head(1)
        if top.empty:
            continue
        best_row = top.iloc[0].to_dict()
        objective_frame = objective_cache[(int(best_row["horizon_days"]), float(best_row["target_return"]))]
        universe_mask = build_universe_masks(objective_frame)[universe_name]
        best_row["top_correlations"] = top_correlations(objective_frame, universe_mask)
        best_rows.append(best_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "objective_scan_summary.csv", index=False)
    pd.DataFrame(best_rows).to_json(output_dir / "objective_scan_best.json", orient="records", indent=2)

    payload = {
        "analysis_start_date": analysis_start_date.isoformat(),
        "analysis_end_date": analysis_end_date.isoformat(),
        "train_end_date": train_end_date.isoformat(),
        "horizons": horizons,
        "targets": targets,
        "min_price": min_price,
        "min_train_n": min_train_n,
        "min_test_n": min_test_n,
        "top_by_universe": best_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _stable_sort_key(result: ComboResult) -> tuple[float, float, int, int]:
    return (
        result.stable_precision,
        result.avg_precision,
        result.test_n,
        result.train_n,
    )


def _test_sort_key(result: ComboResult) -> tuple[float, float, int, int]:
    return (
        result.test_precision,
        result.stable_precision,
        result.test_n,
        result.train_n,
    )


def _maybe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan short-horizon objective windows for automatable hit rates.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nse_live_shortlist"))
    parser.add_argument("--config", type=Path, default=Path("configs/screening.yaml"))
    parser.add_argument("--analysis-start-date", type=date.fromisoformat, default=date(2023, 3, 20))
    parser.add_argument("--analysis-end-date", type=date.fromisoformat, default=date(2025, 3, 19))
    parser.add_argument("--train-end-date", type=date.fromisoformat, default=date(2024, 6, 30))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 7, 10, 15, 20, 25, 29])
    parser.add_argument("--targets", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--min-train-n", type=int, default=100)
    parser.add_argument("--min-test-n", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/objective_scan"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_objective_scan(
        raw_dir=args.raw_dir,
        config_path=args.config,
        analysis_start_date=args.analysis_start_date,
        analysis_end_date=args.analysis_end_date,
        train_end_date=args.train_end_date,
        horizons=args.horizons,
        targets=args.targets,
        min_price=args.min_price,
        min_train_n=args.min_train_n,
        min_test_n=args.min_test_n,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
