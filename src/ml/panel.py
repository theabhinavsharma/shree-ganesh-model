from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.forward_return_study import add_bucket_columns
from src.analysis.forward_return_study import add_market_regime_features
from src.analysis.forward_return_study import build_forward_return_labels
from src.analysis.forward_return_study import _read_optional_table
from src.analysis.forward_return_study import _read_stock_master
from src.screen.build_universe import build_daily_screen_universe
from src.transform.event_daily import build_event_feature_daily
from src.utils.io import write_json
from src.utils.io import write_parquet
from src.ml.config import ObjectiveSpec
from src.ml.config import ResearchConfig
from src.ml.config import research_config_fingerprint
from src.ml.feature_registry import available_feature_columns


def prepare_feature_panel(config: ResearchConfig, objective: ObjectiveSpec, *, force: bool = False) -> tuple[pd.DataFrame, Path]:
    fingerprint = research_config_fingerprint(config, objective=objective)
    source_paths = [
        config.paths.daily_facts,
        config.paths.stock_master,
        config.paths.fundamentals,
        config.paths.shareholding,
        config.paths.sector_state_daily,
        config.paths.macro_daily,
        config.paths.announcements,
        config.paths.event_daily,
    ]
    fingerprint["source_files"] = [
        {
            "path": str(path),
            "exists": bool(path and path.exists()),
            "mtime_ns": path.stat().st_mtime_ns if path and path.exists() else None,
            "size": path.stat().st_size if path and path.exists() else None,
        }
        for path in source_paths
        if path is not None
    ]
    key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    panel_dir = config.paths.panel_cache_dir / f"{objective.name}_{key}"
    panel_path = panel_dir / "feature_panel.parquet"
    meta_path = panel_dir / "metadata.json"
    if panel_path.exists() and meta_path.exists() and not force:
        return pd.read_parquet(panel_path), panel_path

    daily_facts = pd.read_parquet(config.paths.daily_facts)
    daily_facts["trade_date"] = pd.to_datetime(daily_facts["trade_date"]).dt.normalize()
    labels = build_forward_return_labels(
        daily_facts,
        analysis_start_date=pd.Timestamp(objective.analysis_start_date).date(),
        analysis_end_date=pd.Timestamp(objective.analysis_end_date).date(),
        horizon_days=objective.horizon_days,
        target_return=objective.target_return,
        min_price=objective.min_price,
    )
    base_panel, _ = _prepare_base_feature_panel(
        config,
        objective=objective,
        daily_facts=daily_facts,
        force=force,
    )
    label_columns = ["symbol", "trade_date", "forward_trade_date", "forward_close", "forward_return", "winner_flag"]
    panel = base_panel.merge(labels[label_columns], on=["symbol", "trade_date"], how="inner")
    keep_columns = [
        "symbol",
        "trade_date",
        "forward_trade_date",
        "forward_close",
        "forward_return",
        "winner_flag",
        "close",
        "market_cap_cr",
        "instrument_type",
    ]
    keep_columns += available_feature_columns(list(panel.columns), config.feature_columns)
    keep_columns = [column for column in keep_columns if column in panel.columns]
    panel = panel[keep_columns].copy()
    panel = panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    write_parquet(panel, panel_path)
    write_json(
        {
            "objective": objective.__dict__,
            "feature_columns": config.feature_columns,
            "available_feature_columns": [column for column in keep_columns if column not in {"symbol", "trade_date", "forward_trade_date", "forward_close", "forward_return", "winner_flag", "close", "market_cap_cr", "instrument_type"}],
            "fingerprint": fingerprint,
            "row_count": int(len(panel)),
            "date_min": str(panel["trade_date"].min()),
            "date_max": str(panel["trade_date"].max()),
        },
        meta_path,
    )
    return panel, panel_path


def build_current_feature_slice(config: ResearchConfig) -> pd.DataFrame:
    daily_facts = pd.read_parquet(config.paths.daily_facts)
    daily_facts["trade_date"] = pd.to_datetime(daily_facts["trade_date"]).dt.normalize()
    latest_date = daily_facts["trade_date"].max()
    current = daily_facts.loc[daily_facts["trade_date"] == latest_date].copy()
    stock_master = _read_stock_master(config.paths.stock_master)
    fundamentals = _read_optional_table(config.paths.fundamentals)
    shareholding = _read_optional_table(config.paths.shareholding)
    sector_state_daily = _read_optional_table(config.paths.sector_state_daily)
    panel = build_daily_screen_universe(
        daily_facts=current,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=sector_state_daily,
        config=_load_screen_config(),
        include_missing_inputs=False,
    )
    panel = add_market_regime_features(panel)
    panel = add_bucket_columns(panel)
    panel = _merge_macro(panel, config.paths.macro_daily)
    panel = _merge_events(panel, config.paths.event_daily, config.paths.announcements)
    panel = _attach_latest_quote_snapshot_enrichment(panel)
    panel["avg_traded_value_20d_cr"] = pd.to_numeric(panel.get("avg_traded_value_20d"), errors="coerce") / 1e7
    return panel


def _prepare_base_feature_panel(
    config: ResearchConfig,
    *,
    objective: ObjectiveSpec,
    daily_facts: pd.DataFrame | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    fingerprint = _base_feature_fingerprint(config, objective=objective)
    source_paths = [
        config.paths.daily_facts,
        config.paths.stock_master,
        config.paths.fundamentals,
        config.paths.shareholding,
        config.paths.sector_state_daily,
        config.paths.macro_daily,
        config.paths.announcements,
        config.paths.event_daily,
    ]
    fingerprint["source_files"] = [
        {
            "path": str(path),
            "exists": bool(path and path.exists()),
            "mtime_ns": path.stat().st_mtime_ns if path and path.exists() else None,
            "size": path.stat().st_size if path and path.exists() else None,
        }
        for path in source_paths
        if path is not None
    ]
    key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    panel_dir = config.paths.panel_cache_dir / f"base_{key}"
    panel_path = panel_dir / "base_feature_panel.parquet"
    meta_path = panel_dir / "metadata.json"
    if panel_path.exists() and meta_path.exists() and not force:
        return pd.read_parquet(panel_path), panel_path

    working_daily = daily_facts.copy() if daily_facts is not None else pd.read_parquet(config.paths.daily_facts)
    working_daily["trade_date"] = pd.to_datetime(working_daily["trade_date"]).dt.normalize()
    analysis_mask = working_daily["trade_date"].between(
        pd.Timestamp(objective.analysis_start_date),
        pd.Timestamp(objective.analysis_end_date),
    )
    if objective.min_price is not None:
        analysis_mask &= pd.to_numeric(working_daily["close"], errors="coerce").ge(objective.min_price)
    working_daily = working_daily.loc[analysis_mask].copy()

    stock_master = _read_stock_master(config.paths.stock_master)
    fundamentals = _read_optional_table(config.paths.fundamentals)
    shareholding = _read_optional_table(config.paths.shareholding)
    sector_state_daily = _read_optional_table(config.paths.sector_state_daily)
    screen_config = _load_screen_config()

    panel = build_daily_screen_universe(
        daily_facts=working_daily,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=sector_state_daily,
        config=screen_config,
        include_missing_inputs=False,
    )
    panel = add_market_regime_features(panel)
    panel = add_bucket_columns(panel)
    panel = _merge_macro(panel, config.paths.macro_daily)
    panel = _merge_events(panel, config.paths.event_daily, config.paths.announcements)
    panel["avg_traded_value_20d_cr"] = pd.to_numeric(panel.get("avg_traded_value_20d"), errors="coerce") / 1e7
    panel = panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    write_parquet(panel, panel_path)
    write_json(
        {
            "base_window": {
                "analysis_start_date": objective.analysis_start_date,
                "analysis_end_date": objective.analysis_end_date,
                "min_price": objective.min_price,
            },
            "feature_columns": config.feature_columns,
            "available_feature_columns": available_feature_columns(list(panel.columns), config.feature_columns),
            "fingerprint": fingerprint,
            "row_count": int(len(panel)),
            "date_min": str(panel["trade_date"].min()),
            "date_max": str(panel["trade_date"].max()),
        },
        meta_path,
    )
    return panel, panel_path


def _base_feature_fingerprint(config: ResearchConfig, *, objective: ObjectiveSpec) -> dict[str, Any]:
    return {
        "base_window": {
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


def _merge_macro(panel: pd.DataFrame, macro_path: Path | None) -> pd.DataFrame:
    if macro_path is None or not macro_path.exists():
        return panel
    macro = pd.read_parquet(macro_path)
    macro["trade_date"] = pd.to_datetime(macro["trade_date"]).dt.normalize()
    macro = macro.sort_values("trade_date")
    left = panel.sort_values("trade_date")
    merged = pd.merge_asof(left, macro, on="trade_date", direction="backward")
    return merged.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _merge_events(panel: pd.DataFrame, event_daily_path: Path | None, announcements_path: Path | None) -> pd.DataFrame:
    if event_daily_path is not None and event_daily_path.exists():
        event_daily = pd.read_parquet(event_daily_path)
        event_daily["trade_date"] = pd.to_datetime(event_daily["trade_date"]).dt.normalize()
        if panel["trade_date"].max() <= event_daily["trade_date"].max():
            return panel.merge(event_daily, on=["symbol", "trade_date"], how="left")
    if announcements_path is None or not announcements_path.exists():
        return panel
    announcements = pd.read_parquet(announcements_path)
    trade_calendar = panel[["symbol", "trade_date"]].drop_duplicates().copy()
    event_daily = build_event_feature_daily(trade_calendar, announcements)
    return panel.merge(event_daily, on=["symbol", "trade_date"], how="left")


def _load_screen_config() -> dict[str, object]:
    import yaml

    with Path("configs/screening.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _attach_latest_quote_snapshot_enrichment(panel: pd.DataFrame) -> pd.DataFrame:
    working = panel.copy()
    live_snapshot_path = _find_latest_quote_snapshot_path()
    reference_snapshot_path = _find_latest_full_universe_snapshot_path()
    if live_snapshot_path is not None:
        working = _merge_snapshot_reference(
            working,
            live_snapshot_path,
            suffix="quote",
            columns=[
                "symbol",
                "company_name",
                "instrument_type",
                "issued_size",
                "quote_pe_ttm",
                "quote_last_price",
                "quote_last_update_time",
            ],
        )
    if reference_snapshot_path is not None:
        working = _merge_snapshot_reference(
            working,
            reference_snapshot_path,
            suffix="ref",
            columns=[
                "symbol",
                "company_name",
                "instrument_type",
                "issued_size",
            ],
        )

    if "instrument_type" in working.columns:
        instrument_type = working["instrument_type"]
    else:
        instrument_type = pd.Series(pd.NA, index=working.index, dtype="object")
    if "instrument_type_quote" in working.columns:
        instrument_type = instrument_type.where(instrument_type.notna(), working["instrument_type_quote"])
    if "instrument_type_ref" in working.columns:
        instrument_type = instrument_type.where(instrument_type.notna(), working["instrument_type_ref"])
    working["instrument_type"] = instrument_type

    if "company_name" in working.columns and "company_name_quote" in working.columns:
        working["company_name"] = working["company_name"].where(working["company_name"].notna(), working["company_name_quote"])
    elif "company_name" not in working.columns and "company_name_quote" in working.columns:
        working["company_name"] = working["company_name_quote"]
    if "company_name" in working.columns and "company_name_ref" in working.columns:
        working["company_name"] = working["company_name"].where(working["company_name"].notna(), working["company_name_ref"])
    elif "company_name" not in working.columns and "company_name_ref" in working.columns:
        working["company_name"] = working["company_name_ref"]

    issued_size = (
        pd.to_numeric(working["issued_size"], errors="coerce")
        if "issued_size" in working.columns
        else pd.Series(np.nan, index=working.index, dtype="float64")
    )
    if "issued_size_quote" in working.columns:
        issued_size = issued_size.where(issued_size.notna(), pd.to_numeric(working["issued_size_quote"], errors="coerce"))
    if "issued_size_ref" in working.columns:
        issued_size = issued_size.where(issued_size.notna(), pd.to_numeric(working["issued_size_ref"], errors="coerce"))
    working["issued_size"] = issued_size

    close = pd.to_numeric(working.get("close"), errors="coerce")
    market_cap = (
        pd.to_numeric(working["market_cap_cr"], errors="coerce")
        if "market_cap_cr" in working.columns
        else pd.Series(np.nan, index=working.index, dtype="float64")
    )
    market_cap = market_cap.where(market_cap.notna(), (close * issued_size / 10_000_000).round(2))
    working["market_cap_cr"] = market_cap

    if "pe_ttm" not in working.columns:
        working["pe_ttm"] = pd.Series(np.nan, index=working.index, dtype="float64")
    if "quote_pe_ttm" in working.columns:
        working["pe_ttm"] = pd.to_numeric(working["pe_ttm"], errors="coerce").where(
            pd.to_numeric(working["pe_ttm"], errors="coerce").notna(),
            pd.to_numeric(working["quote_pe_ttm"], errors="coerce"),
        )

    drop_columns = [
        column
        for column in [
            "instrument_type_quote",
            "company_name_quote",
            "issued_size_quote",
            "instrument_type_ref",
            "company_name_ref",
            "issued_size_ref",
        ]
        if column in working.columns
    ]
    if drop_columns:
        working = working.drop(columns=drop_columns)
    return working


def _find_latest_quote_snapshot_path() -> Path | None:
    roots = [Path("reports"), Path("tmp")]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob("quote_snapshot/normalized/quote_snapshot.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _find_latest_full_universe_snapshot_path() -> Path | None:
    root = Path("tmp")
    if not root.exists():
        return None
    candidates = list(root.rglob("current_universe_enriched.parquet"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _merge_snapshot_reference(
    panel: pd.DataFrame,
    snapshot_path: Path,
    *,
    suffix: str,
    columns: list[str],
) -> pd.DataFrame:
    try:
        snapshot = pd.read_parquet(snapshot_path)
    except Exception:
        return panel
    if snapshot.empty or "symbol" not in snapshot.columns:
        return panel
    available_columns = [column for column in columns if column in snapshot.columns]
    if "symbol" not in available_columns:
        return panel
    reference = snapshot[available_columns].drop_duplicates(subset=["symbol"]).copy()
    return panel.merge(reference, on="symbol", how="left", suffixes=("", f"_{suffix}"))
