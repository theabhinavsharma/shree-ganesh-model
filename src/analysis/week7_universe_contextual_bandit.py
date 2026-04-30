from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.config import ObjectiveSpec
from src.ml.config import load_research_config
from src.ml.expert_pipeline import ExpertConfig
from src.ml.expert_pipeline import ExpertHorizonSpec
from src.ml.expert_pipeline import FOCUS_OOF_CONTEXT_COLUMNS
from src.ml.expert_pipeline import _build_calibration_table
from src.ml.expert_pipeline import _fit_focus_models
from src.ml.expert_pipeline import _score_focus_current
from src.ml.expert_pipeline import load_or_evaluate_focus_horizon
from src.ml.feature_registry import available_feature_columns
from src.ml.panel import prepare_feature_panel
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme
from src.utils.io import write_json


MACRO_FEATURES = [
    "breadth_above_50_dma",
    "breadth_above_200_dma",
    "breadth_volume_1_5x",
    "market_median_return_20d",
    "nifty_50_return_20d",
    "nifty_500_return_20d",
    "macro_risk_on_flag",
    "macro_vix_below_20",
]

MICRO_FEATURES = [
    "top5_focus_score_mean",
    "top5_prob_5pct_mean",
    "top5_prob_10pct_mean",
    "top5_pred_return_mean",
    "top5_volume_vs_20d_mean",
    "top5_rsi_14_daily_mean",
    "top5_liquidity_cr_mean",
    "top1_focus_score",
    "top1_prob_5pct",
    "top1_pred_return",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    features: tuple[str, ...]
    alpha: float = 0.75
    ridge: float = 1.0


def _bool_to_float(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(float)


def _weekly_arm_dataset(predictions: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    working = predictions.copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"]).dt.normalize()
    iso = working["trade_date"].dt.isocalendar()
    working["year_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    first_days = working.groupby("year_week", sort=False)["trade_date"].min().reset_index()
    first_map = dict(zip(first_days["year_week"], pd.to_datetime(first_days["trade_date"])))
    working = working.loc[working.apply(lambda row: row["trade_date"] == first_map[row["year_week"]], axis=1)].copy()

    rows: list[dict[str, object]] = []
    for (year_week, universe_name), group in working.groupby(["year_week", "universe_name"], sort=False):
        ordered = group.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(top_n).copy()
        if ordered.empty:
            continue
        top1 = ordered.iloc[0]
        row: dict[str, object] = {
            "year_week": year_week,
            "trade_date": str(pd.to_datetime(top1["trade_date"]).date()),
            "universe_name": universe_name,
            "top5_precision_5pct": float(pd.to_numeric(ordered["winner_5pct"], errors="coerce").fillna(0).mean()),
            "top5_mean_return": float(pd.to_numeric(ordered["forward_return"], errors="coerce").mean()),
            "top5_median_return": float(pd.to_numeric(ordered["forward_return"], errors="coerce").median()),
            "top5_any_winner": bool(pd.to_numeric(ordered["winner_5pct"], errors="coerce").fillna(0).sum() >= 1),
            "top5_winner_count": int(pd.to_numeric(ordered["winner_5pct"], errors="coerce").fillna(0).sum()),
            "top5_focus_score_mean": float(pd.to_numeric(ordered["focus_score"], errors="coerce").mean()),
            "top5_prob_5pct_mean": float(pd.to_numeric(ordered["prob_5pct"], errors="coerce").mean()),
            "top5_prob_10pct_mean": float(pd.to_numeric(ordered["prob_10pct"], errors="coerce").mean()),
            "top5_pred_return_mean": float(pd.to_numeric(ordered["pred_return"], errors="coerce").mean()),
            "top5_volume_vs_20d_mean": float(pd.to_numeric(ordered.get("volume_vs_20d"), errors="coerce").mean()),
            "top5_rsi_14_daily_mean": float(pd.to_numeric(ordered.get("rsi_14_daily"), errors="coerce").mean()),
            "top5_liquidity_cr_mean": float(pd.to_numeric(ordered.get("avg_traded_value_20d_cr"), errors="coerce").mean()),
            "top1_focus_score": float(pd.to_numeric(top1.get("focus_score"), errors="coerce")),
            "top1_prob_5pct": float(pd.to_numeric(top1.get("prob_5pct"), errors="coerce")),
            "top1_pred_return": float(pd.to_numeric(top1.get("pred_return"), errors="coerce")),
        }
        for col in MACRO_FEATURES:
            if col in ordered.columns:
                if ordered[col].dtype == "boolean" or str(ordered[col].dtype) == "bool":
                    row[col] = bool(_bool_to_float(ordered[col]).iloc[0])
                else:
                    value = ordered[col].iloc[0]
                    row[col] = bool(value) if isinstance(value, (np.bool_, bool)) else value
            else:
                row[col] = np.nan
        rows.append(row)
    dataset = pd.DataFrame(rows).sort_values(["trade_date", "universe_name"]).reset_index(drop=True)
    for col in ("macro_risk_on_flag", "macro_vix_below_20", "top5_any_winner"):
        if col in dataset.columns:
            dataset[col] = dataset[col].astype("boolean").fillna(False).astype(int)
    return dataset


def _prepare_feature_matrix(frame: pd.DataFrame, feature_names: tuple[str, ...], *, scaler: dict[str, tuple[float, float]] | None = None) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    work = frame[list(feature_names)].copy()
    for col in work.columns:
        if str(work[col].dtype) in {"boolean", "bool"}:
            work[col] = _bool_to_float(work[col])
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if scaler is None:
        scaler = {}
        for col in work.columns:
            mean = float(work[col].mean()) if pd.notna(work[col].mean()) else 0.0
            std = float(work[col].std(ddof=0)) if pd.notna(work[col].std(ddof=0)) and float(work[col].std(ddof=0)) > 1e-8 else 1.0
            scaler[col] = (mean, std)
    for col in work.columns:
        mean, std = scaler[col]
        work[col] = work[col].fillna(mean)
        work[col] = (work[col] - mean) / std
    x = np.column_stack([np.ones(len(work), dtype=np.float64), work.to_numpy(dtype=np.float64)])
    return x, scaler


def _linucb_replay(dataset: pd.DataFrame, *, variant: VariantSpec) -> tuple[pd.DataFrame, dict[str, object]]:
    weeks = sorted(dataset["year_week"].unique())
    universes = sorted(dataset["universe_name"].unique())
    historical_x, scaler = _prepare_feature_matrix(dataset, variant.features)
    dataset = dataset.copy()
    dataset["_x"] = list(historical_x)

    dim = historical_x.shape[1]
    A = {u: np.eye(dim, dtype=np.float64) * variant.ridge for u in universes}
    b = {u: np.zeros(dim, dtype=np.float64) for u in universes}
    counts = {u: 0 for u in universes}
    rows: list[dict[str, object]] = []

    for week in weeks:
        week_rows = dataset.loc[dataset["year_week"] == week].copy()
        if week_rows.empty:
            continue
        candidate_universes = sorted(week_rows["universe_name"].unique())
        unexplored = [u for u in candidate_universes if counts[u] == 0]
        if unexplored:
            selected = unexplored[0]
            scored_rows = []
            for _, row in week_rows.iterrows():
                scored_rows.append({"universe_name": row["universe_name"], "bandit_score": np.nan})
        else:
            scored_rows = []
            best_score = -np.inf
            selected = candidate_universes[0]
            for _, row in week_rows.iterrows():
                universe = str(row["universe_name"])
                x = np.asarray(row["_x"], dtype=np.float64)
                A_inv = np.linalg.inv(A[universe])
                theta = A_inv @ b[universe]
                score = float(theta @ x + variant.alpha * np.sqrt(x @ A_inv @ x))
                scored_rows.append({"universe_name": universe, "bandit_score": score})
                if score > best_score:
                    best_score = score
                    selected = universe
        chosen = week_rows.loc[week_rows["universe_name"] == selected].iloc[0]
        x = np.asarray(chosen["_x"], dtype=np.float64)
        reward = float(chosen["top5_precision_5pct"])
        A[selected] += np.outer(x, x)
        b[selected] += reward * x
        counts[selected] += 1
        score_map = {item["universe_name"]: item["bandit_score"] for item in scored_rows}
        rows.append(
            {
                "year_week": week,
                "trade_date": chosen["trade_date"],
                "selected_universe": selected,
                "reward_precision_5pct": reward,
                "reward_mean_return": float(chosen["top5_mean_return"]),
                "reward_median_return": float(chosen["top5_median_return"]),
                "reward_any_winner": int(chosen["top5_any_winner"]),
                "reward_winner_count": int(chosen["top5_winner_count"]),
                **{f"score_{u}": score_map.get(u, np.nan) for u in universes},
            }
        )

    replay = pd.DataFrame(rows)
    summary = {
        "variant_name": variant.name,
        "alpha": variant.alpha,
        "ridge": variant.ridge,
        "weeks": int(len(replay)),
        "mean_precision_5pct": float(replay["reward_precision_5pct"].mean()) if len(replay) else np.nan,
        "mean_return": float(replay["reward_mean_return"].mean()) if len(replay) else np.nan,
        "median_return": float(replay["reward_median_return"].median()) if len(replay) else np.nan,
        "weeks_with_any_winner_rate": float(replay["reward_any_winner"].mean()) if len(replay) else np.nan,
        "avg_winners_per_week": float(replay["reward_winner_count"].mean()) if len(replay) else np.nan,
        "selection_counts": {u: int((replay["selected_universe"] == u).sum()) for u in universes},
        "scaler": {k: {"mean": v[0], "std": v[1]} for k, v in scaler.items()},
    }
    return replay, summary


def _fixed_universe_baseline(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for universe, group in dataset.groupby("universe_name", sort=False):
        rows.append(
            {
                "universe_name": universe,
                "weeks": int(group["year_week"].nunique()),
                "mean_precision_5pct": float(group["top5_precision_5pct"].mean()),
                "mean_return": float(group["top5_mean_return"].mean()),
                "median_return": float(group["top5_median_return"].median()),
                "weeks_with_any_winner_rate": float(group["top5_any_winner"].mean()),
                "avg_winners_per_week": float(group["top5_winner_count"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_precision_5pct", "mean_return", "universe_name"], ascending=[False, False, True]).reset_index(drop=True)


def _build_current_arm_dataset(
    *,
    config_path: Path,
    predictions: pd.DataFrame,
    analysis_start_date: str,
    live_end_date: str,
    min_price: float,
) -> pd.DataFrame:
    base_config = load_research_config(config_path)
    live_objective = ObjectiveSpec(
        name="week_7_5pct_bandit_live",
        horizon_days=7,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=live_end_date,
        min_price=min_price,
    )
    live_panel, _ = prepare_feature_panel(base_config, live_objective, force=False)
    feature_columns = available_feature_columns(list(live_panel.columns), base_config.feature_columns)
    live_panel["trade_date"] = pd.to_datetime(live_panel["trade_date"]).dt.normalize()
    as_of_trade_date = pd.to_datetime(live_panel["trade_date"]).max().normalize()
    current = live_panel.loc[live_panel["trade_date"].eq(as_of_trade_date)].copy()

    rows: list[dict[str, object]] = []
    from src.ml.universes import build_universe_masks

    for universe_name in base_config.universes:
        scoped_train = live_panel.loc[build_universe_masks(live_panel)[universe_name].fillna(False).astype(bool)].copy()
        if scoped_train.empty:
            continue
        bundle = _fit_focus_models(scoped_train, feature_columns=feature_columns)
        calibration = _build_calibration_table(
            predictions.loc[predictions["universe_name"] == universe_name].copy(),
            score_col="focus_score",
            target_col="winner_5pct",
            return_col="forward_return",
            bins=10,
        )
        scored = _score_focus_current(
            current,
            feature_columns=feature_columns,
            universe_name=universe_name,
            bundle=bundle,
            calibration=calibration,
        )
        if scored.empty:
            continue
        ordered = scored.sort_values(["focus_score", "symbol"], ascending=[False, True]).head(5).copy()
        top1 = ordered.iloc[0]
        row: dict[str, object] = {
            "trade_date": str(as_of_trade_date.date()),
            "universe_name": universe_name,
            "top5_focus_score_mean": float(pd.to_numeric(ordered["focus_score"], errors="coerce").mean()),
            "top5_prob_5pct_mean": float(pd.to_numeric(ordered["prob_5pct_7d"], errors="coerce").mean()),
            "top5_prob_10pct_mean": float(pd.to_numeric(ordered["prob_10pct_7d"], errors="coerce").mean()),
            "top5_pred_return_mean": float(pd.to_numeric(ordered["pred_return_7d"], errors="coerce").mean()),
            "top5_volume_vs_20d_mean": float(pd.to_numeric(ordered.get("volume_vs_20d"), errors="coerce").mean()),
            "top5_rsi_14_daily_mean": float(pd.to_numeric(ordered.get("rsi_14_daily"), errors="coerce").mean()),
            "top5_liquidity_cr_mean": float(pd.to_numeric(ordered.get("avg_traded_value_20d_cr"), errors="coerce").mean()),
            "top1_focus_score": float(pd.to_numeric(top1.get("focus_score"), errors="coerce")),
            "top1_prob_5pct": float(pd.to_numeric(top1.get("prob_5pct_7d"), errors="coerce")),
            "top1_pred_return": float(pd.to_numeric(top1.get("pred_return_7d"), errors="coerce")),
            "top_symbols": ",".join(ordered["symbol"].astype(str).tolist()),
        }
        for col in MACRO_FEATURES:
            if col in ordered.columns:
                value = ordered[col].iloc[0]
                row[col] = bool(value) if isinstance(value, (np.bool_, bool)) else value
            else:
                row[col] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("universe_name").reset_index(drop=True)


def _score_current_variants(current_arms: pd.DataFrame, replay_summaries: dict[str, dict[str, object]], variants: list[VariantSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in variants:
        scaler = {k: (float(v["mean"]), float(v["std"])) for k, v in replay_summaries[variant.name]["scaler"].items()}
        x_matrix, _ = _prepare_feature_matrix(current_arms, variant.features, scaler=scaler)
        counts = replay_summaries[variant.name]["selection_counts"]
        # reconstruct per-arm parameters from replay path is intentionally skipped; use empirical variant + micro/macro score proxy
        # for current ranking, use standardized context sum plus exploration bonus from inverse selection count
        for idx, row in current_arms.iterrows():
            universe = str(row["universe_name"])
            x = x_matrix[idx]
            exploitation = float(np.nanmean(x[1:])) if len(x) > 1 else 0.0
            exploration = float(variant.alpha / np.sqrt(max(int(counts.get(universe, 0)), 1)))
            rows.append(
                {
                    "variant_name": variant.name,
                    "universe_name": universe,
                    "current_bandit_score": exploitation + exploration,
                    "exploit_component": exploitation,
                    "explore_component": exploration,
                    "top_symbols": row["top_symbols"],
                }
            )
    return pd.DataFrame(rows).sort_values(["variant_name", "current_bandit_score", "universe_name"], ascending=[True, False, True]).reset_index(drop=True)


def run_week7_universe_contextual_bandit(
    *,
    config_path: Path,
    output_dir: Path,
    analysis_start_date: str,
    evaluation_end_date: str,
    live_end_date: str,
    min_price: float,
    top_n: int,
    calibration_bins: int,
    force_panel: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_research_config(config_path)
    focus_spec = ExpertHorizonSpec(
        name="week_7_bandit",
        horizon_days=7,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    expert_config = ExpertConfig(
        base_config_path=config_path,
        base_config=base_config,
        horizons=[focus_spec],
        focus_horizon=focus_spec.name,
        shortlist_size=max(top_n, 10),
        calibration_bins=calibration_bins,
        run_output_dir=output_dir / "cache",
    )
    eval_objective = ObjectiveSpec(
        name="week_7_5pct_bandit_eval",
        horizon_days=7,
        target_return=0.0,
        analysis_start_date=analysis_start_date,
        analysis_end_date=evaluation_end_date,
        min_price=min_price,
    )
    eval_panel, eval_panel_path = prepare_feature_panel(base_config, eval_objective, force=force_panel)
    feature_columns = available_feature_columns(list(eval_panel.columns), base_config.feature_columns)
    predictions, focus_summaries = load_or_evaluate_focus_horizon(
        eval_panel,
        feature_columns=feature_columns,
        config=expert_config,
        horizon_spec=focus_spec,
        panel_path=eval_panel_path,
        force=force_panel,
    )

    arm_dataset = _weekly_arm_dataset(predictions, top_n=top_n)
    fixed_baseline = _fixed_universe_baseline(arm_dataset)
    variants = [
        VariantSpec(name="macro_only", features=tuple(MACRO_FEATURES), alpha=0.75),
        VariantSpec(name="micro_only", features=tuple(MICRO_FEATURES), alpha=0.75),
        VariantSpec(name="macro_micro", features=tuple(MACRO_FEATURES + MICRO_FEATURES), alpha=0.75),
    ]
    replay_paths: dict[str, pd.DataFrame] = {}
    replay_summaries: dict[str, dict[str, object]] = {}
    variant_rows: list[dict[str, object]] = []
    for variant in variants:
        replay, replay_summary = _linucb_replay(arm_dataset, variant=variant)
        replay_paths[variant.name] = replay
        replay_summaries[variant.name] = replay_summary
        variant_rows.append(
            {
                "variant_name": variant.name,
                "weeks": replay_summary["weeks"],
                "mean_precision_5pct": replay_summary["mean_precision_5pct"],
                "mean_return": replay_summary["mean_return"],
                "median_return": replay_summary["median_return"],
                "weeks_with_any_winner_rate": replay_summary["weeks_with_any_winner_rate"],
                "avg_winners_per_week": replay_summary["avg_winners_per_week"],
                "selection_counts": replay_summary["selection_counts"],
            }
        )
    variant_summary = pd.DataFrame(variant_rows).sort_values(
        ["mean_precision_5pct", "mean_return", "variant_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    current_arms = _build_current_arm_dataset(
        config_path=config_path,
        predictions=predictions,
        analysis_start_date=analysis_start_date,
        live_end_date=live_end_date,
        min_price=min_price,
    )
    valid_universes = sorted(arm_dataset["universe_name"].dropna().unique().tolist())
    current_arms = current_arms.loc[current_arms["universe_name"].isin(valid_universes)].copy().reset_index(drop=True)
    current_scores = _score_current_variants(current_arms, replay_summaries, variants)
    best_variant = str(variant_summary.iloc[0]["variant_name"])
    best_current = current_scores.loc[current_scores["variant_name"] == best_variant].copy().head(1)

    summary = {
        "status": "ok",
        "objective": {
            "horizon_days": 7,
            "target_return": 0.05,
            "analysis_start_date": analysis_start_date,
            "evaluation_end_date": evaluation_end_date,
            "live_end_date": live_end_date,
            "min_price": min_price,
            "weekly_top_n": top_n,
        },
        "as_of_trade_date": str(pd.to_datetime(current_arms["trade_date"]).max().date()) if not current_arms.empty else live_end_date,
        "panel_path": str(eval_panel_path),
        "focus_prediction_rows": int(len(predictions)),
        "weekly_arm_rows": int(len(arm_dataset)),
        "universes": sorted(predictions["universe_name"].dropna().unique().tolist()),
        "best_fixed_universe": fixed_baseline.iloc[0].to_dict() if not fixed_baseline.empty else None,
        "best_variant": variant_summary.iloc[0].to_dict() if not variant_summary.empty else None,
        "current_best_variant": best_variant,
        "current_best_universe": best_current.iloc[0].to_dict() if not best_current.empty else None,
        "notes": [
            "This is a weekly contextual bandit replay over universe arms, not a stock-level RL policy.",
            "Macro-only, micro-only, and macro+micro variants are replayed on the same weekly decision dates.",
            "The reward is top-5 basket precision for +5 percent in the next 7 days, with mean return and any-winner rate reported alongside it.",
        ],
    }

    arm_dataset_path = output_dir / "weekly_universe_arm_dataset.csv"
    fixed_baseline_path = output_dir / "fixed_universe_baseline.csv"
    variant_summary_path = output_dir / "bandit_variant_summary.csv"
    current_arms_path = output_dir / "current_universe_arm_scores.csv"
    current_scores_path = output_dir / "current_bandit_scores.csv"
    summary_path = output_dir / "summary.json"
    arm_dataset.to_csv(arm_dataset_path, index=False)
    fixed_baseline.to_csv(fixed_baseline_path, index=False)
    variant_summary.to_csv(variant_summary_path, index=False)
    current_arms.to_csv(current_arms_path, index=False)
    current_scores.to_csv(current_scores_path, index=False)
    for variant_name, replay in replay_paths.items():
        replay.to_csv(output_dir / f"replay_path_{variant_name}.csv", index=False)
    write_json(summary, summary_path)

    for path, df, note in [
        (arm_dataset_path, arm_dataset, "Weekly arm dataset built from official-overlay OOF predictions across all configured universes."),
        (fixed_baseline_path, fixed_baseline, "Hindsight fixed-universe weekly baseline for comparison only."),
        (variant_summary_path, variant_summary, "Contextual bandit replay results for macro-only, micro-only, and macro+micro variants."),
        (current_arms_path, current_arms, "Current date arm-level macro and micro features per universe."),
        (current_scores_path, current_scores, "Current bandit-style universe scores per variant."),
    ]:
        write_dataframe_manifest(
            path,
            df,
            generated_by="src/analysis/week7_universe_contextual_bandit.py",
            as_of_date=summary["as_of_trade_date"],
            extra_notes=[note],
        )
    for variant_name, replay in replay_paths.items():
        write_dataframe_manifest(
            output_dir / f"replay_path_{variant_name}.csv",
            replay,
            generated_by="src/analysis/week7_universe_contextual_bandit.py",
            as_of_date=summary["as_of_trade_date"],
            extra_notes=[f"Weekly replay path for the {variant_name} contextual bandit variant."],
        )
    write_json_manifest(
        summary_path,
        summary,
        generated_by="src/analysis/week7_universe_contextual_bandit.py",
        as_of_date=summary["as_of_trade_date"],
        extra_notes=["Official-overlay market data only. This study explores all configured universes as bandit arms."],
    )
    write_report_directory_readme(
        output_dir,
        title="Week 7 Universe Contextual Bandit",
        intro_lines=[
            "This folder contains a weekly contextual bandit replay over universe arms for the 7-day 5 percent objective.",
            "Three variants are compared: macro-only, micro-only, and macro+micro.",
            "Open `summary.json` first, then `bandit_variant_summary.csv`, then the replay paths and current universe scores.",
        ],
        files=[
            summary_path,
            fixed_baseline_path,
            variant_summary_path,
            arm_dataset_path,
            current_arms_path,
            current_scores_path,
        ] + [output_dir / f"replay_path_{variant.name}.csv" for variant in variants],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a weekly contextual bandit replay across all configured universes for the 7-day 5 percent target.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--analysis-start-date", default="2015-01-01")
    parser.add_argument("--evaluation-end-date", default="2025-12-31")
    parser.add_argument("--live-end-date", default="2026-04-22")
    parser.add_argument("--min-price", type=float, default=20.0)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--output-dir", default="reports/week7_universe_contextual_bandit")
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    summary = run_week7_universe_contextual_bandit(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        analysis_start_date=args.analysis_start_date,
        evaluation_end_date=args.evaluation_end_date,
        live_end_date=args.live_end_date,
        min_price=args.min_price,
        top_n=args.top_n,
        calibration_bins=args.calibration_bins,
        force_panel=args.force_panel,
    )
    print(summary)


if __name__ == "__main__":
    main()
