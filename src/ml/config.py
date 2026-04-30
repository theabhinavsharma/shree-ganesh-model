from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    horizon_days: int
    target_return: float
    analysis_start_date: str
    analysis_end_date: str
    min_price: float = 20.0


@dataclass(frozen=True)
class ModelSpec:
    learning_rate: float
    epochs: int
    l2: float
    batch_size: int
    seed: int
    positive_class_weight: str = "balanced"


@dataclass(frozen=True)
class ResearchPaths:
    daily_facts: Path
    stock_master: Path | None
    fundamentals: Path | None
    shareholding: Path | None
    sector_state_daily: Path | None
    macro_daily: Path | None
    announcements: Path | None
    event_daily: Path | None
    panel_cache_dir: Path
    run_output_dir: Path


@dataclass(frozen=True)
class ResearchConfig:
    objectives: list[ObjectiveSpec]
    universes: list[str]
    feature_columns: list[str]
    train_end_date: str
    min_train_rows: int
    min_test_rows: int
    top_quantile: float
    top_n_daily: int
    model: ModelSpec
    paths: ResearchPaths


def load_research_config(path: Path) -> ResearchConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    objectives = [ObjectiveSpec(**item) for item in raw.get("objectives", [])]
    model = ModelSpec(**raw.get("model", {}))
    path_cfg = raw.get("paths", {})
    paths = ResearchPaths(
        daily_facts=Path(path_cfg["daily_facts"]),
        stock_master=_maybe_path(path_cfg.get("stock_master")),
        fundamentals=_maybe_path(path_cfg.get("fundamentals")),
        shareholding=_maybe_path(path_cfg.get("shareholding")),
        sector_state_daily=_maybe_path(path_cfg.get("sector_state_daily")),
        macro_daily=_maybe_path(path_cfg.get("macro_daily")),
        announcements=_maybe_path(path_cfg.get("announcements")),
        event_daily=_maybe_path(path_cfg.get("event_daily")),
        panel_cache_dir=Path(path_cfg["panel_cache_dir"]),
        run_output_dir=Path(path_cfg["run_output_dir"]),
    )
    return ResearchConfig(
        objectives=objectives,
        universes=list(raw.get("universes", [])),
        feature_columns=list(raw.get("features", {}).get("include", [])),
        train_end_date=str(raw["research"]["train_end_date"]),
        min_train_rows=int(raw["research"]["min_train_rows"]),
        min_test_rows=int(raw["research"]["min_test_rows"]),
        top_quantile=float(raw["research"]["top_quantile"]),
        top_n_daily=int(raw["research"]["top_n_daily"]),
        model=model,
        paths=paths,
    )


def research_config_fingerprint(config: ResearchConfig, *, objective: ObjectiveSpec) -> dict[str, Any]:
    return {
        "objective": {
            "name": objective.name,
            "horizon_days": objective.horizon_days,
            "target_return": objective.target_return,
            "analysis_start_date": objective.analysis_start_date,
            "analysis_end_date": objective.analysis_end_date,
            "min_price": objective.min_price,
        },
        "features": config.feature_columns,
        "universes": config.universes,
        "train_end_date": config.train_end_date,
        "min_train_rows": config.min_train_rows,
        "min_test_rows": config.min_test_rows,
        "top_quantile": config.top_quantile,
        "top_n_daily": config.top_n_daily,
        "model": {
            "learning_rate": config.model.learning_rate,
            "epochs": config.model.epochs,
            "l2": config.model.l2,
            "batch_size": config.model.batch_size,
            "seed": config.model.seed,
            "positive_class_weight": config.model.positive_class_weight,
        },
    }


def _maybe_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)

