from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


def add_daily_price_features(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["symbol", "trade_date"]).copy()
    grouped_close = ordered.groupby("symbol")["close"]
    grouped_vol = ordered.groupby("symbol")["total_traded_qty"]
    grouped_value = ordered.groupby("symbol")["total_traded_value"] if "total_traded_value" in ordered.columns else None

    for window in (20, 50, 100, 200):
        ordered[f"sma_{window}"] = grouped_close.transform(lambda s: s.rolling(window, min_periods=window).mean())
    for span in (20, 50, 200):
        ordered[f"ema_{span}"] = grouped_close.transform(lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean())

    ordered["rsi_14_daily"] = grouped_close.transform(compute_rsi)
    ordered["return_1d"] = grouped_close.transform(lambda s: s.pct_change(1))
    ordered["return_20d"] = grouped_close.transform(lambda s: s.pct_change(20))
    ordered["avg_vol_20d"] = grouped_vol.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ordered["avg_vol_60d"] = grouped_vol.transform(lambda s: s.rolling(60, min_periods=60).mean())
    ordered["vol_max_63d"] = grouped_vol.transform(lambda s: s.rolling(63, min_periods=63).max())
    ordered["volume_vs_20d"] = ordered["total_traded_qty"] / ordered["avg_vol_20d"]
    ordered["volume_vs_60d"] = ordered["total_traded_qty"] / ordered["avg_vol_60d"]
    ordered["volume_high_63d_flag"] = (ordered["total_traded_qty"] >= ordered["vol_max_63d"]).where(ordered["vol_max_63d"].notna())
    if grouped_value is not None:
        ordered["avg_traded_value_20d"] = grouped_value.transform(lambda s: s.rolling(20, min_periods=20).mean())
        ordered["avg_traded_value_60d"] = grouped_value.transform(lambda s: s.rolling(60, min_periods=60).mean())
        ordered["traded_value_vs_20d"] = ordered["total_traded_value"] / ordered["avg_traded_value_20d"]
        ordered["traded_value_vs_60d"] = ordered["total_traded_value"] / ordered["avg_traded_value_60d"]
    if "deliverable_qty" in ordered.columns:
        grouped_delivery_qty = ordered.groupby("symbol")["deliverable_qty"]
        ordered["avg_delivery_qty_20d"] = grouped_delivery_qty.transform(lambda s: s.rolling(20, min_periods=20).mean())
        ordered["delivery_qty_vs_20d"] = ordered["deliverable_qty"] / ordered["avg_delivery_qty_20d"]
    if "delivery_pct" not in ordered.columns and "deliverable_qty" in ordered.columns:
        ordered["delivery_pct"] = ordered["deliverable_qty"] / ordered["total_traded_qty"]
    if "delivery_pct" in ordered.columns:
        grouped_delivery_pct = ordered.groupby("symbol")["delivery_pct"]
        ordered["avg_delivery_pct_5d"] = grouped_delivery_pct.transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean())
        ordered["avg_delivery_pct_20d"] = grouped_delivery_pct.transform(lambda s: s.rolling(20, min_periods=20).mean())
        ordered["delivery_pct_max_63d"] = grouped_delivery_pct.transform(lambda s: s.rolling(63, min_periods=63).max())
        ordered["delivery_pct_vs_5d"] = ordered["delivery_pct"] / ordered["avg_delivery_pct_5d"]
        ordered["delivery_pct_vs_20d"] = ordered["delivery_pct"] / ordered["avg_delivery_pct_20d"]
        ordered["delivery_above_5d_avg_flag"] = (ordered["delivery_pct"] > ordered["avg_delivery_pct_5d"]).where(
            ordered["avg_delivery_pct_5d"].notna()
        )
        ordered["delivery_pct_high_63d_flag"] = (ordered["delivery_pct"] >= ordered["delivery_pct_max_63d"]).where(
            ordered["delivery_pct_max_63d"].notna()
        )
    ordered = _add_higher_timeframe_rsi(ordered, freq="W-FRI", target_column="rsi_14_weekly")
    ordered = _add_higher_timeframe_rsi(ordered, freq="ME", target_column="rsi_14_monthly")
    return ordered


def _add_higher_timeframe_rsi(df: pd.DataFrame, *, freq: str, target_column: str) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for symbol, symbol_df in df.groupby("symbol", sort=False):
        period_close = (
            symbol_df.sort_values("trade_date")
            .set_index("trade_date")["close"]
            .resample(freq)
            .last()
            .dropna()
            .reset_index()
        )
        period_close[target_column] = compute_rsi(period_close["close"])
        merged = pd.merge_asof(
            symbol_df.sort_values("trade_date")[["trade_date"]],
            period_close[["trade_date", target_column]].sort_values("trade_date"),
            on="trade_date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["symbol"] = symbol
        pieces.append(merged[["symbol", "trade_date", target_column]])
    if not pieces:
        df[target_column] = pd.NA
        return df
    merged_values = pd.concat(pieces, ignore_index=True)
    return df.merge(merged_values, on=["symbol", "trade_date"], how="left")
