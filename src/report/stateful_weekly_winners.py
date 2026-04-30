from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio.state import DEFAULT_PORTFOLIO_STATE_DIR
from src.portfolio.state import load_current_positions
from src.portfolio.state import prepare_portfolio_state
from src.utils.data_catalog import write_dataframe_manifest


ACTION_PRIORITY = {
    "Buy New": 0,
    "Buy More": 1,
    "Hold": 2,
    "Sell Partly": 3,
    "Sell Wholly": 4,
}


@dataclass(frozen=True)
class DecisionSheetArtifacts:
    csv_path: Path
    report_frame: pd.DataFrame
    state_dir: Path


def _stock_name(frame: pd.DataFrame) -> pd.Series:
    if "stock_name" in frame.columns:
        name = frame["stock_name"].fillna("").astype(str).str.strip()
        return name.where(name.ne(""), frame["symbol"].astype(str))
    if "company_name" in frame.columns:
        name = frame["company_name"].fillna("").astype(str).str.strip()
        return name.where(name.ne(""), frame["symbol"].astype(str))
    return frame["symbol"].astype(str)


def _confidence_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    min_score = numeric.min()
    max_score = numeric.max()
    if pd.isna(min_score) or pd.isna(max_score) or np.isclose(min_score, max_score):
        return pd.Series(75.0, index=series.index)
    normalized = (numeric - min_score) / (max_score - min_score)
    return (55.0 + normalized * 45.0).round(1)


def _liquidity_multiplier(avg_traded_value_20d_cr: pd.Series) -> pd.Series:
    value = pd.to_numeric(avg_traded_value_20d_cr, errors="coerce")
    conditions = [
        value.ge(100.0),
        value.ge(20.0) & value.lt(100.0),
        value.ge(5.0) & value.lt(20.0),
        value.ge(1.0) & value.lt(5.0),
    ]
    choices = [1.0, 0.95, 0.85, 0.75]
    return pd.Series(np.select(conditions, choices, default=0.60), index=avg_traded_value_20d_cr.index, dtype=float)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[column]
    return series.astype("boolean").fillna(False).astype(bool)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _apply_tradeability_overlay(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    base_score = pd.to_numeric(working.get("ranking_score"), errors="coerce").fillna(0.0)
    macro_risk_on = _bool_series(working, "macro_risk_on_flag")
    breadth_50 = _numeric_series(working, "breadth_above_50_dma")
    weak_macro = (~macro_risk_on) & breadth_50.lt(0.30)

    monthly_rsi = _numeric_series(working, "rsi_14_monthly")
    delivery_pct = _numeric_series(working, "delivery_pct")
    market_delivery = _numeric_series(working, "market_median_delivery_pct")
    above_50 = _bool_series(working, "filter_above_50_dma")
    above_200 = _bool_series(working, "filter_above_200_dma")
    volume_high = _bool_series(working, "volume_high_63d_flag")

    supportive_structure = (monthly_rsi.ge(45.0)) | above_50 | above_200 | volume_high
    weak_participation = monthly_rsi.lt(45.0) & delivery_pct.lt(market_delivery.fillna(0.45)) & (~volume_high)

    event_cols = [
        "recent_results_flag",
        "recent_order_win_flag",
        "recent_approval_flag",
        "recent_promoter_buy_flag",
        "recent_promoter_or_director_buy_flag",
        "recent_bulk_buy_flag",
    ]
    event_count = pd.concat([_bool_series(working, col).astype(int) for col in event_cols], axis=1).sum(axis=1)
    thin_event_case = event_count.le(1)

    penalty = pd.Series(1.0, index=working.index, dtype="float64")
    penalty = penalty.where(~(weak_macro & (~supportive_structure)), penalty * 0.92)
    penalty = penalty.where(~(weak_macro & weak_participation), penalty * 0.95)
    penalty = penalty.where(~(weak_macro & thin_event_case & (~supportive_structure)), penalty * 0.93)

    notes = []
    for idx in working.index:
        row_notes: list[str] = []
        if bool(weak_macro.loc[idx]) and not bool(supportive_structure.loc[idx]):
            row_notes.append("weak-macro structure penalty")
        if bool(weak_macro.loc[idx]) and bool(weak_participation.loc[idx]):
            row_notes.append("low delivery/monthly support")
        if bool(weak_macro.loc[idx]) and bool(thin_event_case.loc[idx]) and not bool(supportive_structure.loc[idx]):
            row_notes.append("thin event stack in risk-off tape")
        notes.append(", ".join(row_notes))

    working["base_ranking_score"] = base_score
    working["trade_guardrail_penalty"] = penalty.round(4)
    working["trade_guardrail_pass"] = (penalty >= 0.9999)
    working["trade_guardrail_note"] = pd.Series(notes, index=working.index).replace("", pd.NA)
    working["ranking_score"] = (base_score * penalty).astype("float64")
    return working


def _suggested_allocation(frame: pd.DataFrame, *, cash_buffer_pct: float) -> pd.Series:
    base_score = pd.to_numeric(frame["ranking_score"], errors="coerce").fillna(0.0)
    base_score = base_score - base_score.min() + 0.25
    liquidity = _liquidity_multiplier(frame.get("avg_traded_value_20d_cr", pd.Series(np.nan, index=frame.index)))
    weight = base_score * liquidity
    total_weight = float(weight.sum())
    investable_pct = max(0.0, 100.0 - cash_buffer_pct)
    if total_weight <= 0:
        return pd.Series(investable_pct / max(len(frame), 1), index=frame.index)
    return (weight / total_weight * investable_pct).round(2)


def _buy_range(close_price: pd.Series) -> tuple[pd.Series, pd.Series]:
    close_value = pd.to_numeric(close_price, errors="coerce")
    return (close_value * 0.99).round(2), (close_value * 1.02).round(2)


def _stop_loss(close_price: pd.Series) -> pd.Series:
    close_value = pd.to_numeric(close_price, errors="coerce")
    return (close_value * 0.94).round(2)


def _target_return(frame: pd.DataFrame) -> pd.Series:
    pred = pd.to_numeric(frame.get("pred_return_7d"), errors="coerce")
    if "calibrated_avg_return_7d" in frame.columns:
        calibrated = pd.to_numeric(frame["calibrated_avg_return_7d"], errors="coerce")
    else:
        calibrated = pd.Series(np.nan, index=frame.index, dtype="float64")
    stacked = pd.concat([pred, calibrated], axis=1).max(axis=1).fillna(0.0)
    return stacked.clip(lower=0.05, upper=0.10)


def _format_price(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    return f"{numeric:.2f}" if not pd.isna(numeric) else ""


def _model_rationale(row: pd.Series) -> str:
    reasons: list[str] = []
    calibrated = pd.to_numeric(row.get("calibrated_confidence_5pct_7d"), errors="coerce")
    if not pd.isna(calibrated):
        reasons.append(f"7d bucket hit rate {calibrated * 100:.1f}%")
    volume = pd.to_numeric(row.get("volume_vs_20d"), errors="coerce")
    if not pd.isna(volume) and volume >= 1.5:
        reasons.append(f"volume {volume:.2f}x 20d")
    daily_rsi = pd.to_numeric(row.get("rsi_14_daily"), errors="coerce")
    if not pd.isna(daily_rsi) and daily_rsi >= 60:
        reasons.append(f"daily RSI {daily_rsi:.1f}")
    if _row_bool(row, "recent_results_flag"):
        reasons.append("fresh results")
    if _row_bool(row, "recent_order_win_flag"):
        reasons.append("recent order win")
    if _row_bool(row, "recent_approval_flag"):
        reasons.append("recent approval")
    if _row_bool(row, "recent_promoter_buy_flag") or _row_bool(row, "recent_promoter_or_director_buy_flag"):
        reasons.append("promoter/director buy")
    if _row_bool(row, "recent_bulk_buy_flag"):
        reasons.append("recent bulk buy")
    promoter = pd.to_numeric(row.get("promoter_pct"), errors="coerce")
    if not pd.isna(promoter) and promoter >= 50:
        reasons.append(f"promoter {promoter:.1f}%")
    raw_guardrail_note = row.get("trade_guardrail_note", "")
    guardrail_note = "" if pd.isna(raw_guardrail_note) else str(raw_guardrail_note).strip()
    if guardrail_note:
        reasons.append(guardrail_note)
    return ", ".join(reasons[:4])


def _row_bool(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if pd.isna(value):
        return False
    return bool(value)


def _action_rationale(row: pd.Series, *, threshold: float) -> str:
    action = str(row["recommended_action"])
    current_alloc = float(pd.to_numeric(row.get("current_allocation_pct"), errors="coerce") or 0.0)
    target_alloc = float(pd.to_numeric(row.get("recommended_allocation_pct"), errors="coerce") or 0.0)
    if action == "Buy New":
        return f"New entrant in this week's shortlist; target allocation {target_alloc:.2f}%."
    if action == "Buy More":
        return f"Still shortlisted and target allocation increased from {current_alloc:.2f}% to {target_alloc:.2f}%."
    if action == "Sell Partly":
        if bool(row.get("hit_sell_target_flag", False)):
            return f"Price has reached or exceeded the tactical sell target; trim down toward {target_alloc:.2f}%."
        return f"Still shortlisted, but target allocation has dropped from {current_alloc:.2f}% to {target_alloc:.2f}%."
    if action == "Sell Wholly":
        if bool(row.get("hit_stop_loss_flag", False)):
            return "Price is at or below the stop-loss level, so the position should be exited fully."
        return "This stock is no longer in the active shortlist, so capital should rotate into stronger names."
    return f"Still shortlisted and the current allocation is within {threshold:.2f} percentage points of the new target."


def _choose_action(row: pd.Series, *, threshold: float) -> str:
    held = bool(row.get("currently_held_flag", False))
    current_alloc = float(pd.to_numeric(row.get("current_allocation_pct"), errors="coerce") or 0.0)
    target_alloc = float(pd.to_numeric(row.get("recommended_allocation_pct"), errors="coerce") or 0.0)
    hit_stop = bool(row.get("hit_stop_loss_flag", False))
    hit_target = bool(row.get("hit_sell_target_flag", False))
    if held and hit_stop:
        return "Sell Wholly"
    if not held and target_alloc > 0:
        return "Buy New"
    if held and target_alloc <= 0:
        return "Sell Wholly"
    delta = target_alloc - current_alloc
    if held and hit_target and delta < -threshold:
        return "Sell Partly"
    if delta >= threshold:
        return "Buy More"
    if delta <= -threshold:
        return "Sell Partly"
    return "Hold"


def generate_stateful_weekly_decision_sheet(
    *,
    shortlist: pd.DataFrame,
    live_market_frame: pd.DataFrame,
    output_dir: Path,
    state_dir: Path = DEFAULT_PORTFOLIO_STATE_DIR,
    as_of_trade_date: str,
    objective_name: str = "weekly_7d_5pct",
    cadence_day_local: str = "MONDAY",
    cadence_time_local: str = "20:30",
    timezone_name: str = "Asia/Kolkata",
    cash_buffer_pct: float = 10.0,
    allocation_shift_threshold_pct: float = 2.0,
) -> DecisionSheetArtifacts:
    prepare_portfolio_state(
        state_dir,
        objective_name=objective_name,
        cadence_day_local=cadence_day_local,
        cadence_time_local=cadence_time_local,
        timezone_name=timezone_name,
        note="Stateful weekly investing workflow for the 7-day winners model.",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = load_current_positions(state_dir)

    shortlist_working = shortlist.copy()
    shortlist_working = _apply_tradeability_overlay(shortlist_working)
    shortlist_working = shortlist_working.sort_values(
        ["ranking_score", "focus_score", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    shortlist_working["shortlist_rank"] = np.arange(1, len(shortlist_working) + 1)
    shortlist_working["stock_name"] = _stock_name(shortlist_working)
    shortlist_working["current_price"] = pd.to_numeric(shortlist_working["close"], errors="coerce").round(2)
    buy_low, buy_high = _buy_range(shortlist_working["current_price"])
    shortlist_working["buy_price_low"] = buy_low
    shortlist_working["buy_price_high"] = buy_high
    shortlist_working["buy_price_range"] = buy_low.map(_format_price).astype("string") + " - " + buy_high.map(_format_price).astype("string")
    shortlist_working["sell_target"] = (shortlist_working["current_price"] * (1.0 + _target_return(shortlist_working))).round(2)
    shortlist_working["stop_loss"] = _stop_loss(shortlist_working["current_price"])
    shortlist_working["confidence_score"] = _confidence_score(shortlist_working["ranking_score"])
    shortlist_working["recommended_allocation_pct"] = _suggested_allocation(shortlist_working, cash_buffer_pct=cash_buffer_pct)
    shortlist_working["model_rationale"] = shortlist_working.apply(_model_rationale, axis=1)
    shortlist_working["objective_name"] = objective_name
    shortlist_working["cadence_day_local"] = cadence_day_local
    shortlist_working["cadence_time_local"] = cadence_time_local
    shortlist_working["timezone_name"] = timezone_name
    shortlist_working["selection_status"] = "shortlisted"

    live_lookup = live_market_frame.copy()
    live_lookup["stock_name"] = _stock_name(live_lookup)
    if "current_price" not in live_lookup.columns:
        live_lookup["current_price"] = pd.to_numeric(live_lookup.get("close"), errors="coerce").round(2)
    live_lookup = live_lookup.sort_values(["trade_date", "symbol"]).drop_duplicates("symbol", keep="last")

    union_symbols = pd.Index(
        pd.Series(shortlist_working["symbol"].astype(str).tolist() + positions.get("symbol", pd.Series(dtype="object")).astype(str).tolist())
        .dropna()
        .drop_duplicates()
    )
    combined = pd.DataFrame({"symbol": union_symbols})
    combined = combined.merge(live_lookup, on="symbol", how="left", suffixes=("", "_live"))
    combined = combined.merge(
        shortlist_working,
        on="symbol",
        how="left",
        suffixes=("_live", ""),
    )
    combined = combined.merge(
        positions.add_prefix("held_"),
        left_on="symbol",
        right_on="held_symbol",
        how="left",
    )

    for column in ["trade_date", "stock_name", "sector", "industry", "basic_industry", "close", "current_price"]:
        live_column = f"{column}_live"
        if column in combined.columns and live_column in combined.columns:
            combined[column] = combined[column].fillna(combined[live_column])
        elif column not in combined.columns and live_column in combined.columns:
            combined[column] = combined[live_column]
    combined["stock_name"] = combined["stock_name"].fillna(combined["symbol"])
    combined["currently_held_flag"] = combined["held_symbol"].notna()
    combined["current_allocation_pct"] = pd.to_numeric(combined.get("held_current_allocation_pct"), errors="coerce").fillna(0.0).round(2)
    combined["recommended_allocation_pct"] = pd.to_numeric(combined.get("recommended_allocation_pct"), errors="coerce").fillna(0.0).round(2)
    combined["allocation_delta_pct"] = (combined["recommended_allocation_pct"] - combined["current_allocation_pct"]).round(2)
    combined["entry_trade_date"] = combined.get("held_entry_trade_date")
    combined["entry_price"] = pd.to_numeric(combined.get("held_entry_price"), errors="coerce").round(4)
    combined["weeks_held"] = (
        (pd.Timestamp(as_of_trade_date).normalize() - pd.to_datetime(combined["entry_trade_date"], errors="coerce")).dt.days / 7.0
    ).round(1)
    combined["unrealized_return_pct"] = (
        (pd.to_numeric(combined["current_price"], errors="coerce") / pd.to_numeric(combined["entry_price"], errors="coerce") - 1.0) * 100.0
    ).replace([np.inf, -np.inf], np.nan).round(2)
    effective_stop = pd.to_numeric(combined.get("held_last_stop_loss"), errors="coerce").fillna(pd.to_numeric(combined.get("stop_loss"), errors="coerce"))
    effective_target = pd.to_numeric(combined.get("held_last_sell_target"), errors="coerce").fillna(pd.to_numeric(combined.get("sell_target"), errors="coerce"))
    combined["hit_stop_loss_flag"] = pd.to_numeric(combined["current_price"], errors="coerce").le(effective_stop).fillna(False)
    combined["hit_sell_target_flag"] = pd.to_numeric(combined["current_price"], errors="coerce").ge(effective_target).fillna(False)
    combined["selection_status"] = np.where(combined["recommended_allocation_pct"].gt(0.0), "shortlisted", "held_not_shortlisted")
    combined["recommended_action"] = combined.apply(_choose_action, axis=1, threshold=allocation_shift_threshold_pct)
    combined["action_rationale"] = combined.apply(_action_rationale, axis=1, threshold=allocation_shift_threshold_pct)
    combined["decision_rank"] = (
        combined["recommended_action"].map(ACTION_PRIORITY).fillna(99).astype(int) * 1000
        + combined.get("shortlist_rank", pd.Series(np.arange(len(combined)), index=combined.index)).fillna(999).astype(int)
    )
    combined = combined.sort_values(
        ["decision_rank", "recommended_allocation_pct", "current_allocation_pct", "symbol"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    combined["decision_rank"] = np.arange(1, len(combined) + 1)

    core_columns = [
        "decision_rank",
        "recommended_action",
        "action_rationale",
        "selection_status",
        "currently_held_flag",
        "symbol",
        "stock_name",
        "sector",
        "industry",
        "basic_industry",
        "current_price",
        "buy_price_low",
        "buy_price_high",
        "buy_price_range",
        "sell_target",
        "stop_loss",
        "entry_trade_date",
        "entry_price",
        "weeks_held",
        "unrealized_return_pct",
        "current_allocation_pct",
        "recommended_allocation_pct",
        "allocation_delta_pct",
        "confidence_score",
        "calibrated_confidence_5pct_7d",
        "ranking_score",
        "base_ranking_score",
        "trade_guardrail_penalty",
        "trade_guardrail_pass",
        "trade_guardrail_note",
        "focus_score",
        "prob_up_7d",
        "prob_5pct_7d",
        "prob_10pct_7d",
        "pred_return_7d",
        "pred_price_7d",
        "model_rationale",
        "objective_name",
        "cadence_day_local",
        "cadence_time_local",
        "timezone_name",
    ]
    for column in core_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    ordered_columns = core_columns + [column for column in combined.columns if column not in core_columns and not column.startswith("held_")]
    report = combined[ordered_columns].copy()

    date_tag = pd.Timestamp(as_of_trade_date).strftime("%Y%m%d")
    csv_path = output_dir / f"weekly_position_decision_sheet_{date_tag}.csv"
    report.to_csv(csv_path, index=False)
    write_dataframe_manifest(
        csv_path,
        report,
        generated_by="src.report.stateful_weekly_winners",
        as_of_date=pd.Timestamp(as_of_trade_date).date().isoformat(),
        extra_notes=[
            "This decision sheet is stateful: it compares the latest shortlist against confirmed open positions.",
            "Recommended actions are expressed as Buy New, Buy More, Hold, Sell Partly, or Sell Wholly.",
        ],
    )
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                f"# Weekly Position Decision Sheet For {pd.Timestamp(as_of_trade_date).date().isoformat()}",
                "",
                "This folder contains the stateful weekly decision sheet for the 7-day winners model.",
                "",
                "## How to read this folder",
                "",
                "- Open the decision sheet CSV first.",
                "- Open the matching `.manifest.json` sidecar for column meanings and sample values.",
                "- `recommended_action` is the action to take only after comparing this week's shortlist with confirmed open positions.",
                "- `recommended_allocation_pct` is the new target weight as a percent of total portfolio capital.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return DecisionSheetArtifacts(csv_path=csv_path, report_frame=report, state_dir=state_dir)
