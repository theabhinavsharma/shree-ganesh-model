from __future__ import annotations

import numpy as np
import pandas as pd


def build_event_feature_daily(
    trade_calendar: pd.DataFrame,
    announcements: pd.DataFrame,
    *,
    insider_trades: pd.DataFrame | None = None,
    bulk_block_deals: pd.DataFrame | None = None,
    derivatives_oi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if trade_calendar.empty:
        return pd.DataFrame(columns=_output_columns())

    calendar = trade_calendar[["symbol", "trade_date"]].copy()
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    calendar = calendar.drop_duplicates().sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    announcement_daily = _prepare_announcement_daily(announcements)
    insider_daily = _prepare_insider_daily(insider_trades)
    bulk_daily = _prepare_bulk_block_daily(bulk_block_deals, "bulk_deals")
    block_daily = _prepare_bulk_block_daily(bulk_block_deals, "block_deals")
    oi_daily = _prepare_oi_daily(derivatives_oi)

    pieces: list[pd.DataFrame] = []
    for symbol, trade_dates in calendar.groupby("symbol", sort=False):
        piece = trade_dates.copy()
        trade_ns = piece["trade_date"].to_numpy(dtype="datetime64[ns]")

        _apply_announcement_features(piece, trade_ns, announcement_daily.get(symbol))
        _apply_insider_features(piece, trade_ns, insider_daily.get(symbol))
        _apply_bulk_block_features(piece, trade_ns, bulk_daily.get(symbol), prefix="bulk")
        _apply_bulk_block_features(piece, trade_ns, block_daily.get(symbol), prefix="block")
        _apply_oi_features(piece, trade_ns, oi_daily.get(symbol))

        pieces.append(piece)

    result = pd.concat(pieces, ignore_index=True)
    return result[_output_columns()]


def _prepare_announcement_daily(announcements: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if announcements is None or announcements.empty:
        return {}
    events = announcements.copy()
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    events = events.dropna(subset=["symbol", "event_date"]).copy()
    daily = (
        events.groupby(["symbol", "event_date"], as_index=False)
        .agg(
            any_event_count=("sequence_id", "size"),
            results_event_count=("is_results_event", lambda s: int(pd.Series(s).fillna(False).sum())),
            order_win_count=("is_order_win", lambda s: int(pd.Series(s).fillna(False).sum())),
            approval_count=("is_approval", lambda s: int(pd.Series(s).fillna(False).sum())),
            pledge_change_count=("is_pledge_change", lambda s: int(pd.Series(s).fillna(False).sum())),
            promoter_buy_count=("is_promoter_buying", lambda s: int(pd.Series(s).fillna(False).sum())),
        )
        .sort_values(["symbol", "event_date"])
    )
    return {symbol: frame.copy() for symbol, frame in daily.groupby("symbol", sort=False)}


def _prepare_insider_daily(insider_trades: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if insider_trades is None or insider_trades.empty:
        return {}
    frame = insider_trades.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.normalize()
    frame = frame.dropna(subset=["symbol", "event_date"]).copy()
    for column in [
        "buy_value",
        "sell_value",
        "net_value",
        "buy_quantity",
        "sell_quantity",
        "net_quantity",
        "holding_change_pct",
    ]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    promoter_or_director = _bool_series(frame, "is_promoter_group_or_promoter") | _bool_series(frame, "is_director_or_kmp")
    daily = (
        frame.groupby(["symbol", "event_date"], as_index=False)
        .agg(
            insider_buy_value=("buy_value", "sum"),
            insider_sell_value=("sell_value", "sum"),
            insider_net_value=("net_value", "sum"),
            insider_buy_count=("is_buy_transaction", lambda s: int(pd.Series(s).fillna(False).sum())),
            insider_sell_count=("is_sell_transaction", lambda s: int(pd.Series(s).fillna(False).sum())),
            promoter_director_buy_value=("buy_value", lambda s: float(s[promoter_or_director.loc[s.index]].fillna(0.0).sum())),
            promoter_director_sell_value=("sell_value", lambda s: float(s[promoter_or_director.loc[s.index]].fillna(0.0).sum())),
            promoter_director_net_value=("net_value", lambda s: float(s[promoter_or_director.loc[s.index]].fillna(0.0).sum())),
            promoter_director_buy_count=(
                "is_buy_transaction",
                lambda s: int((pd.Series(s).fillna(False).astype(bool) & promoter_or_director.loc[s.index]).sum()),
            ),
        )
        .sort_values(["symbol", "event_date"])
    )
    return {symbol: symbol_frame.copy() for symbol, symbol_frame in daily.groupby("symbol", sort=False)}


def _prepare_bulk_block_daily(bulk_block_deals: pd.DataFrame | None, deal_type: str) -> dict[str, pd.DataFrame]:
    if bulk_block_deals is None or bulk_block_deals.empty:
        return {}
    if "deal_type" not in bulk_block_deals.columns:
        return {}
    frame = bulk_block_deals.loc[bulk_block_deals["deal_type"].astype(str) == deal_type].copy()
    if frame.empty:
        return {}
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.dropna(subset=["symbol", "trade_date"]).copy()
    frame["traded_value"] = pd.to_numeric(frame.get("traded_value"), errors="coerce")
    frame["quantity_traded"] = pd.to_numeric(frame.get("quantity_traded"), errors="coerce")
    daily = (
        frame.groupby(["symbol", "trade_date"], as_index=False)
        .agg(
            buy_value=("traded_value", lambda s: float(s[frame.loc[s.index, "is_buy"].fillna(False)].fillna(0.0).sum())),
            sell_value=("traded_value", lambda s: float(s[frame.loc[s.index, "is_sell"].fillna(False)].fillna(0.0).sum())),
            net_value=("traded_value", lambda s: float((s.where(frame.loc[s.index, "is_buy"].fillna(False), 0.0) - s.where(frame.loc[s.index, "is_sell"].fillna(False), 0.0)).fillna(0.0).sum())),
            buy_count=("is_buy", lambda s: int(pd.Series(s).fillna(False).sum())),
            sell_count=("is_sell", lambda s: int(pd.Series(s).fillna(False).sum())),
            buy_quantity=("quantity_traded", lambda s: float(s[frame.loc[s.index, "is_buy"].fillna(False)].fillna(0.0).sum())),
            sell_quantity=("quantity_traded", lambda s: float(s[frame.loc[s.index, "is_sell"].fillna(False)].fillna(0.0).sum())),
        )
        .sort_values(["symbol", "trade_date"])
    )
    return {symbol: symbol_frame.copy() for symbol, symbol_frame in daily.groupby("symbol", sort=False)}


def _prepare_oi_daily(derivatives_oi: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    if derivatives_oi is None or derivatives_oi.empty:
        return {}
    frame = derivatives_oi.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    keep_columns = [
        "symbol",
        "trade_date",
        "mwpl",
        "ncl_open_interest",
        "ncl_futeq_oi",
        "oi_share_of_mwpl",
        "oi_change_1d",
        "oi_change_pct_1d",
        "futeq_oi_change_1d",
        "oi_share_of_mwpl_change_1d",
    ]
    keep_columns = [column for column in keep_columns if column in frame.columns]
    frame = frame[keep_columns].dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"])
    return {symbol: symbol_frame.copy() for symbol, symbol_frame in frame.groupby("symbol", sort=False)}


def _apply_announcement_features(piece: pd.DataFrame, trade_ns: np.ndarray, symbol_events: pd.DataFrame | None) -> None:
    if symbol_events is None or symbol_events.empty:
        _fill_missing(piece, [
            "announcements_7d",
            "announcements_30d",
            "results_events_30d",
            "order_wins_90d",
            "approvals_90d",
            "pledge_changes_90d",
            "promoter_buys_180d",
            "days_since_any_announcement",
            "days_since_results_event",
            "days_since_order_win",
            "days_since_approval",
            "recent_results_flag",
            "recent_order_win_flag",
            "recent_approval_flag",
            "recent_pledge_change_flag",
            "recent_promoter_buy_flag",
        ])
        return
    shifted = _shift_daily_table(symbol_events, trade_ns, "event_date", {
        "any_event_count": "sum",
        "results_event_count": "sum",
        "order_win_count": "sum",
        "approval_count": "sum",
        "pledge_change_count": "sum",
        "promoter_buy_count": "sum",
    })
    event_ns = shifted["effective_trade_date"].to_numpy(dtype="datetime64[ns]")
    piece["announcements_7d"] = _window_sum(trade_ns, event_ns, shifted["any_event_count"].to_numpy(dtype=float), 7)
    piece["announcements_30d"] = _window_sum(trade_ns, event_ns, shifted["any_event_count"].to_numpy(dtype=float), 30)
    piece["results_events_30d"] = _window_sum(trade_ns, event_ns, shifted["results_event_count"].to_numpy(dtype=float), 30)
    piece["order_wins_90d"] = _window_sum(trade_ns, event_ns, shifted["order_win_count"].to_numpy(dtype=float), 90)
    piece["approvals_90d"] = _window_sum(trade_ns, event_ns, shifted["approval_count"].to_numpy(dtype=float), 90)
    piece["pledge_changes_90d"] = _window_sum(trade_ns, event_ns, shifted["pledge_change_count"].to_numpy(dtype=float), 90)
    piece["promoter_buys_180d"] = _window_sum(trade_ns, event_ns, shifted["promoter_buy_count"].to_numpy(dtype=float), 180)
    piece["days_since_any_announcement"] = _days_since_last_event(trade_ns, event_ns)
    piece["days_since_results_event"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["results_event_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece["days_since_order_win"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["order_win_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece["days_since_approval"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["approval_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece["recent_results_flag"] = piece["results_events_30d"].gt(0)
    piece["recent_order_win_flag"] = piece["order_wins_90d"].gt(0)
    piece["recent_approval_flag"] = piece["approvals_90d"].gt(0)
    piece["recent_pledge_change_flag"] = piece["pledge_changes_90d"].gt(0)
    piece["recent_promoter_buy_flag"] = piece["promoter_buys_180d"].gt(0)


def _apply_insider_features(piece: pd.DataFrame, trade_ns: np.ndarray, symbol_events: pd.DataFrame | None) -> None:
    if symbol_events is None or symbol_events.empty:
        _fill_missing(piece, [
            "insider_buy_value_30d",
            "insider_sell_value_30d",
            "insider_net_value_30d",
            "insider_buy_count_30d",
            "promoter_director_buy_value_90d",
            "promoter_director_net_value_90d",
            "promoter_director_buy_count_90d",
            "days_since_insider_buy",
            "days_since_promoter_or_director_buy",
            "recent_insider_buy_flag",
            "recent_promoter_or_director_buy_flag",
        ])
        return
    shifted = _shift_daily_table(symbol_events, trade_ns, "event_date", {
        "insider_buy_value": "sum",
        "insider_sell_value": "sum",
        "insider_net_value": "sum",
        "insider_buy_count": "sum",
        "promoter_director_buy_value": "sum",
        "promoter_director_net_value": "sum",
        "promoter_director_buy_count": "sum",
    })
    event_ns = shifted["effective_trade_date"].to_numpy(dtype="datetime64[ns]")
    piece["insider_buy_value_30d"] = _window_sum(trade_ns, event_ns, shifted["insider_buy_value"].to_numpy(dtype=float), 30)
    piece["insider_sell_value_30d"] = _window_sum(trade_ns, event_ns, shifted["insider_sell_value"].to_numpy(dtype=float), 30)
    piece["insider_net_value_30d"] = _window_sum(trade_ns, event_ns, shifted["insider_net_value"].to_numpy(dtype=float), 30)
    piece["insider_buy_count_30d"] = _window_sum(trade_ns, event_ns, shifted["insider_buy_count"].to_numpy(dtype=float), 30)
    piece["promoter_director_buy_value_90d"] = _window_sum(trade_ns, event_ns, shifted["promoter_director_buy_value"].to_numpy(dtype=float), 90)
    piece["promoter_director_net_value_90d"] = _window_sum(trade_ns, event_ns, shifted["promoter_director_net_value"].to_numpy(dtype=float), 90)
    piece["promoter_director_buy_count_90d"] = _window_sum(trade_ns, event_ns, shifted["promoter_director_buy_count"].to_numpy(dtype=float), 90)
    piece["days_since_insider_buy"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["insider_buy_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece["days_since_promoter_or_director_buy"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["promoter_director_buy_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece["recent_insider_buy_flag"] = piece["insider_buy_count_30d"].gt(0)
    piece["recent_promoter_or_director_buy_flag"] = piece["promoter_director_buy_count_90d"].gt(0)


def _apply_bulk_block_features(
    piece: pd.DataFrame,
    trade_ns: np.ndarray,
    symbol_events: pd.DataFrame | None,
    *,
    prefix: str,
) -> None:
    target_columns = [
        f"{prefix}_buy_value_30d",
        f"{prefix}_sell_value_30d",
        f"{prefix}_net_value_30d",
        f"{prefix}_buy_count_30d",
        f"days_since_{prefix}_buy",
        f"recent_{prefix}_buy_flag",
    ]
    if symbol_events is None or symbol_events.empty:
        _fill_missing(piece, target_columns)
        return
    shifted = _shift_daily_table(symbol_events, trade_ns, "trade_date", {
        "buy_value": "sum",
        "sell_value": "sum",
        "net_value": "sum",
        "buy_count": "sum",
    })
    event_ns = shifted["effective_trade_date"].to_numpy(dtype="datetime64[ns]")
    piece[f"{prefix}_buy_value_30d"] = _window_sum(trade_ns, event_ns, shifted["buy_value"].to_numpy(dtype=float), 30)
    piece[f"{prefix}_sell_value_30d"] = _window_sum(trade_ns, event_ns, shifted["sell_value"].to_numpy(dtype=float), 30)
    piece[f"{prefix}_net_value_30d"] = _window_sum(trade_ns, event_ns, shifted["net_value"].to_numpy(dtype=float), 30)
    piece[f"{prefix}_buy_count_30d"] = _window_sum(trade_ns, event_ns, shifted["buy_count"].to_numpy(dtype=float), 30)
    piece[f"days_since_{prefix}_buy"] = _days_since_last_event(
        trade_ns,
        shifted.loc[shifted["buy_count"] > 0, "effective_trade_date"].to_numpy(dtype="datetime64[ns]"),
    )
    piece[f"recent_{prefix}_buy_flag"] = piece[f"{prefix}_buy_count_30d"].gt(0)


def _apply_oi_features(piece: pd.DataFrame, trade_ns: np.ndarray, symbol_oi: pd.DataFrame | None) -> None:
    target_columns = [
        "mwpl",
        "ncl_open_interest",
        "ncl_futeq_oi",
        "oi_share_of_mwpl",
        "oi_change_1d",
        "oi_change_pct_1d",
        "futeq_oi_change_1d",
        "oi_share_of_mwpl_change_1d",
    ]
    if symbol_oi is None or symbol_oi.empty:
        _fill_missing(piece, target_columns)
        return
    shifted = _shift_daily_table(symbol_oi, trade_ns, "trade_date", {column: "last" for column in target_columns})
    merged = piece.merge(
        shifted.rename(columns={"effective_trade_date": "trade_date"}),
        on="trade_date",
        how="left",
    )
    for column in target_columns:
        piece[column] = merged[column]


def _shift_daily_table(frame: pd.DataFrame, trade_ns: np.ndarray, date_col: str, agg_map: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["effective_trade_date", *agg_map.keys()])
    dates = pd.to_datetime(frame[date_col]).to_numpy(dtype="datetime64[ns]")
    effective_dates = _map_to_next_trade_date(trade_ns, dates)
    working = frame.copy()
    working["effective_trade_date"] = effective_dates
    working = working.dropna(subset=["effective_trade_date"]).copy()
    if working.empty:
        return pd.DataFrame(columns=["effective_trade_date", *agg_map.keys()])
    shifted = (
        working.groupby("effective_trade_date", as_index=False)
        .agg(agg_map)
        .sort_values("effective_trade_date")
        .reset_index(drop=True)
    )
    return shifted


def _map_to_next_trade_date(trade_dates: np.ndarray, event_dates: np.ndarray) -> np.ndarray:
    if len(event_dates) == 0:
        return np.array([], dtype="datetime64[ns]")
    idx = np.searchsorted(trade_dates, event_dates, side="right")
    result = np.full(len(event_dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = idx < len(trade_dates)
    if valid.any():
        result[valid] = trade_dates[idx[valid]]
    return result


def _window_sum(
    trade_dates: np.ndarray,
    event_dates: np.ndarray,
    event_values: np.ndarray,
    lookback_days: int,
) -> np.ndarray:
    if len(event_dates) == 0:
        return np.full(len(trade_dates), np.nan)
    values = event_values.astype(float)
    cumulative = values.cumsum()
    left_bounds = trade_dates - np.timedelta64(lookback_days - 1, "D")
    end_idx = np.searchsorted(event_dates, trade_dates, side="right") - 1
    start_idx = np.searchsorted(event_dates, left_bounds, side="left")
    result = np.zeros(len(trade_dates), dtype=float)
    valid = end_idx >= 0
    result[~valid] = np.nan
    valid_idx = np.where(valid)[0]
    if len(valid_idx):
        ends = end_idx[valid]
        starts = start_idx[valid]
        totals = cumulative[ends]
        subtract = np.where(starts > 0, cumulative[starts - 1], 0.0)
        result[valid_idx] = totals - subtract
    return result


def _days_since_last_event(trade_dates: np.ndarray, event_dates: np.ndarray) -> np.ndarray:
    if len(event_dates) == 0:
        return np.full(len(trade_dates), np.nan)
    idx = np.searchsorted(event_dates, trade_dates, side="right") - 1
    valid = idx >= 0
    result = np.full(len(trade_dates), np.nan)
    if valid.any():
        last_dates = event_dates[idx[valid]]
        result[valid] = (trade_dates[valid] - last_dates).astype("timedelta64[D]").astype(float)
    return result


def _fill_missing(piece: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        piece[column] = pd.NA


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def _output_columns() -> list[str]:
    return [
        "symbol",
        "trade_date",
        "announcements_7d",
        "announcements_30d",
        "results_events_30d",
        "order_wins_90d",
        "approvals_90d",
        "pledge_changes_90d",
        "promoter_buys_180d",
        "days_since_any_announcement",
        "days_since_results_event",
        "days_since_order_win",
        "days_since_approval",
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_pledge_change_flag",
        "recent_promoter_buy_flag",
        "insider_buy_value_30d",
        "insider_sell_value_30d",
        "insider_net_value_30d",
        "insider_buy_count_30d",
        "promoter_director_buy_value_90d",
        "promoter_director_net_value_90d",
        "promoter_director_buy_count_90d",
        "days_since_insider_buy",
        "days_since_promoter_or_director_buy",
        "recent_insider_buy_flag",
        "recent_promoter_or_director_buy_flag",
        "bulk_buy_value_30d",
        "bulk_sell_value_30d",
        "bulk_net_value_30d",
        "bulk_buy_count_30d",
        "days_since_bulk_buy",
        "recent_bulk_buy_flag",
        "block_buy_value_30d",
        "block_sell_value_30d",
        "block_net_value_30d",
        "block_buy_count_30d",
        "days_since_block_buy",
        "recent_block_buy_flag",
        "mwpl",
        "ncl_open_interest",
        "ncl_futeq_oi",
        "oi_share_of_mwpl",
        "oi_change_1d",
        "oi_change_pct_1d",
        "futeq_oi_change_1d",
        "oi_share_of_mwpl_change_1d",
    ]
