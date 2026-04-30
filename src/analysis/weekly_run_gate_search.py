from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools

import pandas as pd


@dataclass(frozen=True)
class GateSearchConfig:
    input_dir: Path
    output_csv: Path
    universes: tuple[str, ...] = ("liquid_5cr_plus", "liquid_20cr_plus", "mid_small")
    top_n_values: tuple[int, ...] = (5, 8, 10, 12)
    min_winner_values: tuple[int, ...] = (2, 3)
    search_years: tuple[int, ...] = (2023, 2024)
    test_years: tuple[int, ...] = (2025,)
    min_search_weeks: int = 20
    min_test_weeks: int = 10


GATE_CANDIDATES: tuple[tuple[str, str, tuple[object, ...]], ...] = (
    ("macro_risk_on_flag", "bool", (True,)),
    ("macro_vix_below_20", "bool", (True,)),
    ("breadth_above_50_dma", "num", (0.55, 0.60, 0.65, 0.70)),
    ("breadth_above_200_dma", "num", (0.45, 0.50, 0.55, 0.60)),
    ("nifty_50_return_20d", "num", (0.00, 0.03, 0.05)),
    ("nifty_500_return_20d", "num", (0.00, 0.03, 0.05)),
    ("market_median_return_20d", "num", (0.00, 0.03, 0.05)),
    ("breadth_volume_1_5x", "num", (0.10, 0.15, 0.20)),
)


def search_weekly_run_gates(config: GateSearchConfig) -> pd.DataFrame:
    weekly_panel = _load_weekly_panel(config)
    rows: list[dict[str, object]] = []
    gate_defs = _gate_definitions()

    for universe in config.universes:
        universe_df = weekly_panel.loc[weekly_panel["universe_name"] == universe].copy()
        if universe_df.empty:
            continue
        for top_n in config.top_n_values:
            for min_winners in config.min_winner_values:
                for gate_tuple in gate_defs:
                    if not _gate_columns_available(universe_df, gate_tuple):
                        continue
                    gated = universe_df.loc[_gate_mask(universe_df, gate_tuple)].copy()
                    if gated.empty:
                        continue
                    weekly = _weekly_success(gated, top_n=top_n, min_winners=min_winners)
                    if weekly.empty:
                        continue
                    search = weekly.loc[weekly["year"].isin(config.search_years)].copy()
                    test = weekly.loc[weekly["year"].isin(config.test_years)].copy()
                    if len(search) < config.min_search_weeks or len(test) < config.min_test_weeks:
                        continue
                    row = {
                        "universe_name": universe,
                        "top_n": top_n,
                        "min_winners": min_winners,
                        "gate": _gate_label(gate_tuple),
                        "search_weeks": int(len(search)),
                        "test_weeks": int(len(test)),
                        "search_success_rate": float(search["success"].mean()),
                        "test_success_rate": float(test["success"].mean()),
                        "search_avg_winners": float(search["winner_count"].mean()),
                        "test_avg_winners": float(test["winner_count"].mean()),
                        "all_success_rate": float(weekly["success"].mean()),
                        "all_avg_winners": float(weekly["winner_count"].mean()),
                    }
                    row["stability_score"] = _stability_score(row)
                    rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["test_success_rate", "test_avg_winners", "search_success_rate", "search_weeks"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(config.output_csv, index=False)
    return result


def _load_weekly_panel(config: GateSearchConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for universe in config.universes:
        path = config.input_dir / f"{universe}_oof.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
        iso = frame["trade_date"].dt.isocalendar()
        frame["run_week"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
        week_end = frame.groupby("run_week", sort=False)["trade_date"].max().rename("week_trade_date")
        frame = frame.merge(week_end, on="run_week", how="left")
        frame = frame.loc[frame["trade_date"] == frame["week_trade_date"]].copy()
        frame["year"] = frame["run_week"].str.slice(0, 4).astype(int)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _gate_definitions() -> list[tuple[tuple[str, str, object], ...]]:
    singles: list[tuple[tuple[str, str, object], ...]] = [tuple()]
    for column, kind, values in GATE_CANDIDATES:
        for value in values:
            singles.append(((column, kind, value),))
    pair_candidates = [single[0] for single in singles if single]
    pairs = [pair for pair in itertools.combinations(pair_candidates, 2)]
    return singles + pairs


def _gate_mask(frame: pd.DataFrame, gate_tuple: tuple[tuple[str, str, object], ...]) -> pd.Series:
    if not gate_tuple:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for column, kind, value in gate_tuple:
        if kind == "bool":
            mask &= frame[column].fillna(False).eq(value)
        else:
            mask &= pd.to_numeric(frame[column], errors="coerce").ge(value)
    return mask


def _gate_columns_available(frame: pd.DataFrame, gate_tuple: tuple[tuple[str, str, object], ...]) -> bool:
    return all(column in frame.columns for column, _, _ in gate_tuple)


def _gate_label(gate_tuple: tuple[tuple[str, str, object], ...]) -> str:
    if not gate_tuple:
        return "no_gate"
    parts: list[str] = []
    for column, kind, value in gate_tuple:
        if kind == "bool":
            parts.append(f"{column}=True")
        else:
            parts.append(f"{column}>={value}")
    return " & ".join(parts)


def _weekly_success(frame: pd.DataFrame, *, top_n: int, min_winners: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pred_return_col = "pred_return_7d" if "pred_return_7d" in frame.columns else "pred_return"
    for run_week, group in frame.groupby("run_week", sort=False):
        ordered = group.sort_values(["focus_score", "prob_10pct", pred_return_col], ascending=[False, False, False]).head(top_n)
        winner_count = int(pd.to_numeric(ordered["winner_5pct"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "run_week": run_week,
                "year": int(str(run_week)[:4]),
                "winner_count": winner_count,
                "success": winner_count >= min_winners,
            }
        )
    return pd.DataFrame(rows)


def _stability_score(row: dict[str, object]) -> float:
    return (
        float(row["test_success_rate"]) * 0.55
        + float(row["search_success_rate"]) * 0.25
        + float(row["test_avg_winners"]) * 0.15
        + min(float(row["test_weeks"]), 52.0) / 52.0 * 0.05
    )
