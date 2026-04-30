from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest.nse.quote_snapshot import SNAPSHOT_COLUMNS
from src.ingest.nse.quote_snapshot import build_quote_snapshot_from_symbols
from src.screen.build_universe import apply_screen_filters
from src.screen.build_universe import load_screen_config
from src.utils.io import write_parquet


def enrich_scored_universe_for_checklist(
    universe: pd.DataFrame,
    *,
    daily_facts_path: Path,
    config_path: Path,
    output_dir: Path,
    quote_snapshot_path: Path | None = None,
    quote_snapshot: pd.DataFrame | None = None,
    candidate_count: int = 120,
    metadata_delay_seconds: float = 0.05,
) -> pd.DataFrame:
    working = universe.copy()
    ordered = working.sort_values(["model_score", "model_pass_count", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    candidate_symbols = ordered["symbol"].dropna().astype(str).head(max(candidate_count, 1)).tolist()
    snapshot = quote_snapshot.copy() if quote_snapshot is not None else _load_or_fetch_quote_snapshot(
        candidate_symbols,
        output_dir=output_dir,
        quote_snapshot_path=quote_snapshot_path,
        metadata_delay_seconds=metadata_delay_seconds,
    )
    working = _merge_quote_snapshot(working, snapshot)
    recent_price_features = _build_recent_price_feature_snapshot(daily_facts_path=daily_facts_path, symbols=set(candidate_symbols))
    if not recent_price_features.empty:
        working = working.merge(recent_price_features, on=["symbol", "trade_date"], how="left", suffixes=("", "_fresh"))
        for column in ["avg_delivery_pct_5d", "delivery_pct_vs_5d", "delivery_above_5d_avg_flag"]:
            fresh_column = f"{column}_fresh"
            if fresh_column not in working.columns:
                continue
            base = working[column] if column in working.columns else pd.Series(pd.NA, index=working.index, dtype="object")
            working[column] = base.where(base.notna(), working[fresh_column])
            working = working.drop(columns=[fresh_column])
    config = load_screen_config(config_path)
    working = apply_screen_filters(working, config, include_missing_inputs=True)
    channel = build_trade_channel_snapshot(
        daily_facts_path=daily_facts_path,
        symbols=set(candidate_symbols),
    )
    if not channel.empty:
        working = working.merge(channel, on="symbol", how="left")
    write_parquet(working, output_dir / "checklist_enriched_universe.parquet")
    return working


def build_trade_channel_snapshot(
    *,
    daily_facts_path: Path,
    symbols: set[str],
    lookback_bars: int = 90,
    pivot_span: int = 3,
    pivot_count: int = 4,
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=_channel_columns())
    history = pd.read_parquet(daily_facts_path, columns=["symbol", "trade_date", "high", "low", "close"])
    history = history[history["symbol"].astype(str).isin({symbol.upper() for symbol in symbols})].copy()
    if history.empty:
        return pd.DataFrame(columns=_channel_columns())
    history["trade_date"] = pd.to_datetime(history["trade_date"]).dt.normalize()
    rows = [_build_channel_row(symbol_df, lookback_bars=lookback_bars, pivot_span=pivot_span, pivot_count=pivot_count) for _, symbol_df in history.groupby("symbol", sort=False)]
    return pd.DataFrame(rows, columns=_channel_columns())


def _build_recent_price_feature_snapshot(*, daily_facts_path: Path, symbols: set[str]) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "trade_date", "avg_delivery_pct_5d", "delivery_pct_vs_5d", "delivery_above_5d_avg_flag"])
    history = pd.read_parquet(daily_facts_path, columns=["symbol", "trade_date", "delivery_pct"])
    history = history[history["symbol"].astype(str).isin({symbol.upper() for symbol in symbols})].copy()
    if history.empty:
        return pd.DataFrame(columns=["symbol", "trade_date", "avg_delivery_pct_5d", "delivery_pct_vs_5d", "delivery_above_5d_avg_flag"])
    history["trade_date"] = pd.to_datetime(history["trade_date"]).dt.normalize()
    history = history.sort_values(["symbol", "trade_date"]).copy()
    grouped = history.groupby("symbol")["delivery_pct"]
    history["avg_delivery_pct_5d"] = grouped.transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
    history["delivery_pct_vs_5d"] = pd.to_numeric(history["delivery_pct"], errors="coerce") / pd.to_numeric(history["avg_delivery_pct_5d"], errors="coerce")
    history["delivery_above_5d_avg_flag"] = (
        pd.to_numeric(history["delivery_pct"], errors="coerce").gt(pd.to_numeric(history["avg_delivery_pct_5d"], errors="coerce"))
    ).where(history["avg_delivery_pct_5d"].notna())
    latest_trade_date = history["trade_date"].max()
    return history.loc[history["trade_date"].eq(latest_trade_date), ["symbol", "trade_date", "avg_delivery_pct_5d", "delivery_pct_vs_5d", "delivery_above_5d_avg_flag"]].copy()


def _load_or_fetch_quote_snapshot(
    symbols: list[str],
    *,
    output_dir: Path,
    quote_snapshot_path: Path | None,
    metadata_delay_seconds: float,
) -> pd.DataFrame:
    if quote_snapshot_path is not None and quote_snapshot_path.exists():
        return pd.read_parquet(quote_snapshot_path)
    return build_quote_snapshot_from_symbols(
        symbols,
        output_dir=output_dir / "quote_snapshot",
        delay_seconds=metadata_delay_seconds,
    )


def _merge_quote_snapshot(universe: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        for column in SNAPSHOT_COLUMNS:
            if column != "symbol" and column not in universe.columns:
                universe[column] = pd.NA
        return universe
    working = universe.merge(snapshot, on="symbol", how="left", suffixes=("", "_quote"))
    for column in ["company_name", "sector", "industry", "basic_industry", "instrument_type"]:
        quote_column = f"{column}_quote"
        if quote_column not in working.columns:
            continue
        base = working[column] if column in working.columns else pd.Series(pd.NA, index=working.index, dtype="object")
        quote = working[quote_column]
        working[column] = _coalesce_text(base, quote)
        working = working.drop(columns=[quote_column])
    close = pd.to_numeric(working.get("close"), errors="coerce")
    issued = pd.to_numeric(working.get("issued_size"), errors="coerce")
    working["market_cap_cr"] = (close * issued / 10_000_000).round(2)
    if "quote_pe_ttm" in working.columns:
        existing_pe = pd.to_numeric(working.get("pe_ttm"), errors="coerce")
        quote_pe = pd.to_numeric(working.get("quote_pe_ttm"), errors="coerce")
        working["pe_ttm"] = existing_pe.where(existing_pe.notna(), quote_pe)
    return working


def _coalesce_text(left: pd.Series, right: pd.Series) -> pd.Series:
    left_text = left.fillna("").astype(str).str.strip()
    right_text = right.fillna("").astype(str).str.strip()
    return left_text.where(left_text.ne(""), right_text).replace("", pd.NA)


def _build_channel_row(
    symbol_df: pd.DataFrame,
    *,
    lookback_bars: int,
    pivot_span: int,
    pivot_count: int,
) -> dict[str, object]:
    recent = symbol_df.sort_values("trade_date").tail(lookback_bars).reset_index(drop=True)
    symbol = str(recent.iloc[-1]["symbol"]) if not recent.empty else ""
    default_row = {
        "symbol": symbol,
        "channel_valid_flag": False,
        "channel_lower": np.nan,
        "channel_upper": np.nan,
        "channel_position_pct": np.nan,
        "channel_slope_low": np.nan,
        "channel_slope_high": np.nan,
        "trade_action": pd.NA,
        "channel_buy_price_low": np.nan,
        "channel_buy_price_high": np.nan,
        "channel_sell_target": np.nan,
        "channel_stop_loss": np.nan,
    }
    if len(recent) < max(20, pivot_span * 6):
        return default_row
    pivot_highs = _find_pivots(pd.to_numeric(recent["high"], errors="coerce"), kind="high", span=pivot_span)
    pivot_lows = _find_pivots(pd.to_numeric(recent["low"], errors="coerce"), kind="low", span=pivot_span)
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return default_row
    high_idx, high_values = zip(*pivot_highs[-pivot_count:])
    low_idx, low_values = zip(*pivot_lows[-pivot_count:])
    if not _strictly_rising(high_values) or not _strictly_rising(low_values):
        return default_row
    high_coeff = np.polyfit(np.asarray(high_idx, dtype=float), np.asarray(high_values, dtype=float), 1)
    low_coeff = np.polyfit(np.asarray(low_idx, dtype=float), np.asarray(low_values, dtype=float), 1)
    current_x = float(len(recent) - 1)
    channel_upper = float(np.polyval(high_coeff, current_x))
    channel_lower = float(np.polyval(low_coeff, current_x))
    if high_coeff[0] <= 0 or low_coeff[0] <= 0 or not np.isfinite(channel_upper) or not np.isfinite(channel_lower) or channel_upper <= channel_lower:
        return default_row
    current_price = float(pd.to_numeric(recent.iloc[-1]["close"], errors="coerce"))
    if not np.isfinite(current_price):
        return default_row
    channel_width = channel_upper - channel_lower
    position = (current_price - channel_lower) / channel_width if channel_width > 0 else np.nan
    trade_action = "Buy" if position <= 0.35 else "Sell" if position >= 0.80 else "Hold"
    buy_low = round(max(channel_lower * 0.995, 0.0), 2)
    buy_high = round(max(channel_lower * 1.02, buy_low), 2)
    sell_target = round(max(channel_upper, buy_high * 1.05), 2)
    stop_loss = round(max(channel_lower * 0.97, 0.0), 2)
    return {
        "symbol": symbol,
        "channel_valid_flag": True,
        "channel_lower": round(channel_lower, 2),
        "channel_upper": round(channel_upper, 2),
        "channel_position_pct": round(position * 100, 1) if np.isfinite(position) else np.nan,
        "channel_slope_low": round(float(low_coeff[0]), 6),
        "channel_slope_high": round(float(high_coeff[0]), 6),
        "trade_action": trade_action,
        "channel_buy_price_low": buy_low,
        "channel_buy_price_high": buy_high,
        "channel_sell_target": sell_target,
        "channel_stop_loss": stop_loss,
    }


def _find_pivots(series: pd.Series, *, kind: str, span: int) -> list[tuple[int, float]]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    rows: list[tuple[int, float]] = []
    for idx in range(span, len(values) - span):
        center = values[idx]
        if not np.isfinite(center):
            continue
        left = values[idx - span : idx]
        right = values[idx + 1 : idx + span + 1]
        if kind == "high" and np.all(center > left) and np.all(center > right):
            rows.append((idx, float(center)))
        if kind == "low" and np.all(center < left) and np.all(center < right):
            rows.append((idx, float(center)))
    return rows


def _strictly_rising(values: tuple[float, ...]) -> bool:
    return all(float(current) > float(previous) for previous, current in zip(values, values[1:]))


def _channel_columns() -> list[str]:
    return [
        "symbol",
        "channel_valid_flag",
        "channel_lower",
        "channel_upper",
        "channel_position_pct",
        "channel_slope_low",
        "channel_slope_high",
        "trade_action",
        "channel_buy_price_low",
        "channel_buy_price_high",
        "channel_sell_target",
        "channel_stop_loss",
    ]
