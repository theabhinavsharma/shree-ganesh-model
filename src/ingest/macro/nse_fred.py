from __future__ import annotations

import json
from io import StringIO
import time
from dataclasses import dataclass
from datetime import date
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from src.ingest.nse.api import get_json
from src.ingest.nse.session import build_session
from src.utils.io import write_json
from src.utils.io import write_parquet

NSE_INDEX_REFERER = "https://www.nseindia.com/reports-indices-historical-index-data"
NSE_VIX_REFERER = "https://www.nseindia.com/reports-indices-historical-vix"
NSE_INDEX_MASTER_URL = "https://www.nseindia.com/api/equity-masterOR"
NSE_INDEX_HISTORY_URL = (
    "https://www.nseindia.com/api/historicalOR/indicesHistory?indexType={index_type}&from={from_date}&to={to_date}"
)
NSE_VIX_HISTORY_URL = "https://www.nseindia.com/api/historicalOR/vixhistory?from={from_date}&to={to_date}"
NSE_INDEX_YIELD_URL = (
    "https://www.nseindia.com/api/historicalOR/indicesYield?indexType={index_type}&from={from_date}&to={to_date}"
)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
DEFAULT_FRED_SERIES = {
    "DEXINUS": "fred_usdinr",
    "DCOILWTICO": "fred_wti_crude",
    "SP500": "fred_sp500",
    "NASDAQCOM": "fred_nasdaq_comp",
    "DJIA": "fred_djia",
}
DEFAULT_NSE_INDEXES = (
    "NIFTY 50",
    "NIFTY 500",
    "NIFTY BANK",
    "NIFTY IT",
    "NIFTY PHARMA",
    "NIFTY AUTO",
    "NIFTY METAL",
    "NIFTY OIL & GAS",
    "NIFTY PRIVATE BANK",
    "NIFTY PSU BANK",
    "NIFTY 10 YR BENCHMARK G-SEC",
)


@dataclass(frozen=True)
class MacroFetchConfig:
    output_dir: Path
    start_date: date
    end_date: date
    nse_indices: tuple[str, ...] = DEFAULT_NSE_INDEXES
    fred_series: dict[str, str] | None = None
    delay_seconds: float = 0.05
    include_sector_indices: bool = True
    include_fixed_income_indices: bool = False


def load_macro_history(config: MacroFetchConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = build_session(warm=True, referer=NSE_INDEX_REFERER)
    index_master_path = config.output_dir / "raw" / "nse_index_master.json"
    index_master = _load_json_with_cache_fallback(
        session,
        NSE_INDEX_MASTER_URL,
        raw_path=index_master_path,
        referer=NSE_INDEX_REFERER,
    )

    index_names = _build_index_list(index_master, config)

    level_frames: list[pd.DataFrame] = []
    valuation_frames: list[pd.DataFrame] = []
    for index_name in index_names:
        for window_start, window_end in iter_api_windows(config.start_date, config.end_date):
            history_url = NSE_INDEX_HISTORY_URL.format(
                index_type=quote(index_name, safe=""),
                from_date=window_start.strftime("%d-%m-%Y"),
                to_date=window_end.strftime("%d-%m-%Y"),
            )
            history_path = (
                config.output_dir
                / "raw"
                / "nse_index_history"
                / _safe_key(index_name)
                / f"{window_start.isoformat()}_{window_end.isoformat()}.json"
            )
            history = _load_json_with_cache_fallback(
                session,
                history_url,
                raw_path=history_path,
                referer=NSE_INDEX_REFERER,
            )
            history_rows = history.get("data", [])
            if history_rows:
                level_frames.append(_normalize_nse_index_history(history_rows, index_name=index_name, source_url=history_url))

            valuation_url = NSE_INDEX_YIELD_URL.format(
                index_type=quote(index_name, safe=""),
                from_date=window_start.strftime("%d-%m-%Y"),
                to_date=window_end.strftime("%d-%m-%Y"),
            )
            valuation_path = (
                config.output_dir
                / "raw"
                / "nse_index_yield"
                / _safe_key(index_name)
                / f"{window_start.isoformat()}_{window_end.isoformat()}.json"
            )
            valuation = _load_json_with_cache_fallback(
                session,
                valuation_url,
                raw_path=valuation_path,
                referer=NSE_INDEX_REFERER,
            )
            valuation_rows = valuation.get("data", [])
            if valuation_rows:
                valuation_frames.append(
                    _normalize_nse_index_valuation(valuation_rows, index_name=index_name, source_url=valuation_url)
                )
            time.sleep(config.delay_seconds)

    for window_start, window_end in iter_api_windows(config.start_date, config.end_date):
        vix_url = NSE_VIX_HISTORY_URL.format(
            from_date=window_start.strftime("%d-%m-%Y"),
            to_date=window_end.strftime("%d-%m-%Y"),
        )
        vix_path = config.output_dir / "raw" / "nse_vix_history" / f"{window_start.isoformat()}_{window_end.isoformat()}.json"
        vix = _load_json_with_cache_fallback(
            session,
            vix_url,
            raw_path=vix_path,
            referer=NSE_VIX_REFERER,
        )
        vix_rows = vix.get("data", [])
        if vix_rows:
            level_frames.append(_normalize_nse_vix_history(vix_rows, source_url=vix_url))
        time.sleep(config.delay_seconds)

    fred_frames: list[pd.DataFrame] = []
    for series_id, series_key in (config.fred_series or DEFAULT_FRED_SERIES).items():
        fred_path = config.output_dir / "raw" / "fred" / f"{series_id}.parquet"
        fred_df = _load_fred_series(series_id, raw_path=fred_path)
        fred_frames.append(_normalize_fred_series(fred_df, series_id=series_id, series_key=series_key))

    macro_levels = pd.concat(level_frames + fred_frames, ignore_index=True) if (level_frames or fred_frames) else pd.DataFrame()
    if not macro_levels.empty:
        macro_levels = macro_levels.sort_values(["series_key", "trade_date"]).reset_index(drop=True)
        write_parquet(macro_levels, config.output_dir / "normalized" / "macro_series_long.parquet")

    macro_valuations = pd.concat(valuation_frames, ignore_index=True) if valuation_frames else pd.DataFrame()
    if not macro_valuations.empty:
        macro_valuations = macro_valuations.sort_values(["series_key", "trade_date"]).reset_index(drop=True)
        write_parquet(macro_valuations, config.output_dir / "normalized" / "macro_index_valuation_long.parquet")

    macro_daily = build_macro_feature_daily(macro_levels, macro_valuations)
    if not macro_daily.empty:
        write_parquet(macro_daily, config.output_dir / "normalized" / "macro_feature_daily.parquet")
    return macro_levels, macro_daily


def iter_api_windows(start_date: date, end_date: date, max_window_days: int = 365) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        window_end = min(cursor + timedelta(days=max_window_days - 1), end_date)
        windows.append((cursor, window_end))
        cursor = window_end + timedelta(days=1)
    return windows


def _load_fred_series(series_id: str, *, raw_path: Path, max_attempts: int = 4) -> pd.DataFrame:
    if raw_path.exists():
        return pd.read_parquet(raw_path)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(FRED_CSV_URL.format(series_id=series_id), timeout=30)
            response.raise_for_status()
            fred_df = pd.read_csv(StringIO(response.text))
            write_parquet(fred_df, raw_path)
            return fred_df
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= max_attempts:
                raise
            time.sleep(min(2 ** (attempt - 1), 8))
    assert last_error is not None
    raise last_error


def _load_json_with_cache_fallback(
    session,
    url: str,
    *,
    raw_path: Path,
    referer: str,
):
    try:
        payload = get_json(session, url, referer=referer)
        write_json(payload, raw_path)
        return payload
    except Exception:
        if raw_path.exists():
            return json.loads(raw_path.read_text(encoding="utf-8"))
        raise


def build_macro_feature_daily(levels: pd.DataFrame, valuations: pd.DataFrame) -> pd.DataFrame:
    if levels.empty and valuations.empty:
        return pd.DataFrame()

    wide = pd.DataFrame()
    if not levels.empty:
        level_pivot = levels.pivot_table(index="trade_date", columns="series_key", values="close", aggfunc="last")
        level_pivot = level_pivot.sort_index().ffill()
        level_pivot.columns = [f"{column}_level" for column in level_pivot.columns]
        wide = level_pivot
        for column in list(level_pivot.columns):
            source = column.removesuffix("_level")
            wide[f"{source}_return_1d"] = level_pivot[column].pct_change(1)
            wide[f"{source}_return_5d"] = level_pivot[column].pct_change(5)
            wide[f"{source}_return_20d"] = level_pivot[column].pct_change(20)
            wide[f"{source}_sma_20"] = level_pivot[column].rolling(20, min_periods=20).mean()
            wide[f"{source}_sma_50"] = level_pivot[column].rolling(50, min_periods=50).mean()
            wide[f"{source}_above_50dma"] = level_pivot[column].gt(wide[f"{source}_sma_50"])

    if not valuations.empty:
        valuation_pivot = valuations.pivot_table(
            index="trade_date",
            columns="series_key",
            values=["pe", "pb", "dy"],
            aggfunc="last",
        )
        valuation_pivot = valuation_pivot.sort_index().ffill()
        valuation_pivot.columns = [f"{series_key}_{metric}" for metric, series_key in valuation_pivot.columns]
        wide = valuation_pivot if wide.empty else wide.join(valuation_pivot, how="outer")

    if wide.empty:
        return pd.DataFrame()

    wide = wide.sort_index().ffill()
    if "india_vix_level" in wide.columns:
        wide["macro_vix_below_20"] = wide["india_vix_level"].lt(20)
        wide["macro_vix_below_15"] = wide["india_vix_level"].lt(15)
    if "nifty_50_return_20d" in wide.columns and "fred_sp500_return_20d" in wide.columns:
        wide["macro_india_minus_spx_return_20d"] = wide["nifty_50_return_20d"] - wide["fred_sp500_return_20d"]
    if "nifty_500_return_20d" in wide.columns and "india_vix_level" in wide.columns:
        wide["macro_risk_on_flag"] = wide["nifty_500_return_20d"].gt(0).astype("boolean") & wide["india_vix_level"].lt(20)
    return wide.reset_index()


def _normalize_nse_index_history(rows: list[dict[str, object]], *, index_name: str, source_url: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "series_name": index_name,
            "series_key": _safe_key(index_name),
            "source_family": "nse_index_history",
            "close": pd.to_numeric(df["EOD_CLOSE_INDEX_VAL"], errors="coerce"),
            "open": pd.to_numeric(df["EOD_OPEN_INDEX_VAL"], errors="coerce"),
            "high": pd.to_numeric(df["EOD_HIGH_INDEX_VAL"], errors="coerce"),
            "low": pd.to_numeric(df["EOD_LOW_INDEX_VAL"], errors="coerce"),
            "source_url": source_url,
            "source_note": "official_nse_index_history",
        }
    ).dropna(subset=["trade_date"])


def _normalize_nse_vix_history(rows: list[dict[str, object]], *, source_url: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["EOD_TIMESTAMP"], format="%d-%b-%Y", errors="coerce")
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "series_name": "INDIA VIX",
            "series_key": "india_vix",
            "source_family": "nse_vix_history",
            "close": pd.to_numeric(df["EOD_CLOSE_INDEX_VAL"], errors="coerce"),
            "open": pd.to_numeric(df["EOD_OPEN_INDEX_VAL"], errors="coerce"),
            "high": pd.to_numeric(df["EOD_HIGH_INDEX_VAL"], errors="coerce"),
            "low": pd.to_numeric(df["EOD_LOW_INDEX_VAL"], errors="coerce"),
            "source_url": source_url,
            "source_note": "official_nse_vix_history",
        }
    ).dropna(subset=["trade_date"])


def _normalize_nse_index_valuation(rows: list[dict[str, object]], *, index_name: str, source_url: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["IY_DT"], format="%d-%b-%Y", errors="coerce")
    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "series_name": index_name,
            "series_key": _safe_key(index_name),
            "pe": pd.to_numeric(df["IY_PE"], errors="coerce"),
            "pb": pd.to_numeric(df["IY_PB"], errors="coerce"),
            "dy": pd.to_numeric(df["IY_DY"], errors="coerce"),
            "source_url": source_url,
            "source_note": "official_nse_index_pe_pb_dy",
        }
    ).dropna(subset=["trade_date"])


def _normalize_fred_series(df: pd.DataFrame, *, series_id: str, series_key: str) -> pd.DataFrame:
    value_column = next(column for column in df.columns if column != "observation_date")
    normalized = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(df["observation_date"], errors="coerce"),
            "series_name": series_id,
            "series_key": series_key,
            "source_family": "fred_public",
            "close": pd.to_numeric(df[value_column], errors="coerce"),
            "open": pd.NA,
            "high": pd.NA,
            "low": pd.NA,
            "source_url": FRED_CSV_URL.format(series_id=series_id),
            "source_note": "fred_public_macro_series",
        }
    )
    return normalized.dropna(subset=["trade_date"])


def _build_index_list(index_master: dict[str, object], config: MacroFetchConfig) -> list[str]:
    names = list(config.nse_indices)
    if config.include_sector_indices:
        names.extend(_clean_index_list(index_master.get("Sectoral Market Indices", [])))
    if config.include_fixed_income_indices:
        names.extend(_clean_index_list(index_master.get("Fixed Income Indices", [])))
    return list(dict.fromkeys(name for name in names if name))


def _clean_index_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _safe_key(value: str) -> str:
    cleaned = (
        value.strip()
        .lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("-", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(":", " ")
        .replace(",", " ")
        .replace(".", " ")
    )
    return "_".join(cleaned.split())
