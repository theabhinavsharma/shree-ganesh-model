from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.model_scoring import apply_feature_score


@dataclass(frozen=True)
class RebalanceSummary:
    portfolio_size: int
    frequency: str
    step_trading_days: int
    periods: int
    avg_period_return_gross: float
    median_period_return_gross: float
    avg_period_return_net: float
    median_period_return_net: float
    gross_win_rate: float
    net_win_rate: float
    annualized_return_gross: float
    annualized_return_net: float
    annualized_vol_net: float
    sharpe_like_net: float | None
    max_drawdown_net: float
    avg_turnover: float
    annual_turnover_multiple: float
    avg_cost_rate: float


FREQUENCY_STEPS = {
    "daily": 1,
    "weekly": 5,
    "fortnightly": 10,
    "monthly": 21,
    "six_weekly": 30,
    "quarterly": 63,
}


def run_rebalance_frequency_study(
    *,
    anchor_universe_path: Path,
    feature_results_path: Path,
    output_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    portfolio_sizes: tuple[int, ...] = (5, 10, 20),
) -> dict[str, object]:
    frame = pd.read_parquet(anchor_universe_path)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    if start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(start_date)].copy()
    if end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(end_date)].copy()

    feature_results = pd.read_csv(feature_results_path)
    scoring_features = feature_results[feature_results["column"].isin(_score_columns())].copy()
    scoring_features = scoring_features[scoring_features["test_lift"].gt(1.0)].copy()
    scored = apply_feature_score(frame, scoring_features)
    scored = scored.sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    price = scored.pivot(index="trade_date", columns="symbol", values="close").sort_index().ffill()
    costs = scored.pivot(index="trade_date", columns="symbol", values="_one_way_cost_rate").sort_index().ffill()
    daily_groups = {
        trade_date: group.sort_values(["model_score", "symbol"], ascending=[False, True]).reset_index(drop=True)
        for trade_date, group in scored.groupby("trade_date", sort=True)
    }
    trade_dates = price.index.to_list()

    summaries: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    for portfolio_size in portfolio_sizes:
        for frequency, step in FREQUENCY_STEPS.items():
            rebalance_dates = trade_dates[::step]
            gross_returns: list[float] = []
            net_returns: list[float] = []
            turnover_rates: list[float] = []
            cost_rates: list[float] = []
            previous_weights: dict[str, float] | None = None

            for idx in range(len(rebalance_dates) - 1):
                start = rebalance_dates[idx]
                end = rebalance_dates[idx + 1]
                ranking = daily_groups.get(start)
                if ranking is None or ranking.empty:
                    continue
                ranking = ranking[ranking["model_score"] > 0].head(portfolio_size).copy()
                if ranking.empty:
                    continue

                picks = ranking["symbol"].tolist()
                start_prices = price.loc[start, picks]
                end_prices = price.loc[end, picks]
                valid = start_prices.notna() & end_prices.notna() & start_prices.gt(0)
                if valid.sum() == 0:
                    continue
                picks = list(pd.Index(picks)[valid])
                equal_weight = 1.0 / len(picks)
                current_weights = {symbol: equal_weight for symbol in picks}

                gross_return = float(((end_prices[picks] / start_prices[picks]) - 1.0).mean())
                cost_rate, turnover = _estimate_turnover_cost(previous_weights, current_weights, costs.loc[start].to_dict())
                net_return = gross_return - cost_rate

                gross_returns.append(gross_return)
                net_returns.append(net_return)
                turnover_rates.append(turnover)
                cost_rates.append(cost_rate)
                period_rows.append(
                    {
                        "portfolio_size": portfolio_size,
                        "frequency": frequency,
                        "start_date": start.date().isoformat(),
                        "end_date": end.date().isoformat(),
                        "gross_return": gross_return,
                        "net_return": net_return,
                        "turnover": turnover,
                        "cost_rate": cost_rate,
                        "positions": len(picks),
                    }
                )
                previous_weights = current_weights

            if not gross_returns:
                continue

            gross = pd.Series(gross_returns)
            net = pd.Series(net_returns)
            periods_per_year = 252 / step
            net_equity = (1 + net).cumprod()
            net_peak = net_equity.cummax()
            net_drawdown = net_equity / net_peak - 1
            ann_return_gross = float((1 + gross.mean()) ** periods_per_year - 1)
            ann_return_net = float((1 + net.mean()) ** periods_per_year - 1)
            ann_vol_net = float(net.std(ddof=1) * math.sqrt(periods_per_year)) if len(net) > 1 else 0.0
            sharpe_net = ann_return_net / ann_vol_net if ann_vol_net > 0 else None

            summaries.append(
                asdict(
                    RebalanceSummary(
                        portfolio_size=portfolio_size,
                        frequency=frequency,
                        step_trading_days=step,
                        periods=len(net),
                        avg_period_return_gross=float(gross.mean()),
                        median_period_return_gross=float(gross.median()),
                        avg_period_return_net=float(net.mean()),
                        median_period_return_net=float(net.median()),
                        gross_win_rate=float(gross.gt(0).mean()),
                        net_win_rate=float(net.gt(0).mean()),
                        annualized_return_gross=ann_return_gross,
                        annualized_return_net=ann_return_net,
                        annualized_vol_net=ann_vol_net,
                        sharpe_like_net=sharpe_net,
                        max_drawdown_net=float(net_drawdown.min()),
                        avg_turnover=float(pd.Series(turnover_rates).mean()),
                        annual_turnover_multiple=float(pd.Series(turnover_rates).mean() * periods_per_year),
                        avg_cost_rate=float(pd.Series(cost_rates).mean()),
                    )
                )
            )

    summary_df = pd.DataFrame(summaries).sort_values(["portfolio_size", "annualized_return_net"], ascending=[True, False])
    periods_df = pd.DataFrame(period_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "rebalance_summary.csv", index=False)
    periods_df.to_csv(output_dir / "rebalance_period_returns.csv", index=False)
    payload = {"summary": summary_df.to_dict(orient="records")}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _score_columns() -> tuple[str, ...]:
    return (
        "filter_above_50_dma",
        "filter_above_200_dma",
        "volume_high_63d_flag",
        "rsi_14_monthly",
        "promoter_pct",
        "promoter_pct_qoq_change",
        "recent_promoter_buy_flag",
        "recent_approval_flag",
        "recent_order_win_flag",
        "revenue_cagr_5y",
        "pat_cagr_5y",
        "ebitda_positive_last_5q_flag",
        "pe_ttm",
        "interest_coverage",
        "debt_equity_ratio",
        "fii_fpi_pct",
        "dii_pct",
        "mf_pct",
    )


def _estimate_turnover_cost(
    previous_weights: dict[str, float] | None,
    current_weights: dict[str, float],
    one_way_costs: dict[str, float],
) -> tuple[float, float]:
    if previous_weights is None:
        total_trade = sum(current_weights.values())
    else:
        symbols = set(previous_weights) | set(current_weights)
        total_trade = sum(abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in symbols)
    # turnover here is one-way traded notional fraction.
    turnover = total_trade
    cost_rate = 0.0
    if previous_weights is None:
        for symbol, weight in current_weights.items():
            cost_rate += weight * one_way_costs.get(symbol, 0.0)
    else:
        symbols = set(previous_weights) | set(current_weights)
        for symbol in symbols:
            traded_weight = abs(current_weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
            cost_rate += traded_weight * one_way_costs.get(symbol, 0.0)
    return cost_rate, turnover


def attach_cost_rates(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    adv = pd.to_numeric(enriched.get("avg_traded_value_20d_cr"), errors="coerce")
    one_way = pd.Series(0.0075, index=enriched.index, dtype="float64")
    one_way = one_way.where(~adv.ge(100), 0.0005)
    one_way = one_way.where(~(adv.ge(20) & adv.lt(100)), 0.0010)
    one_way = one_way.where(~(adv.ge(5) & adv.lt(20)), 0.0020)
    one_way = one_way.where(~(adv.ge(1) & adv.lt(5)), 0.0035)
    one_way = one_way.where(~adv.lt(1), 0.0060)
    enriched["_one_way_cost_rate"] = one_way
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-universe-path", required=True)
    parser.add_argument("--feature-results-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    args = parser.parse_args()

    frame = pd.read_parquet(args.anchor_universe_path)
    frame = attach_cost_rates(frame)
    temp_anchor = Path(args.output_dir) / "_anchor_with_costs.parquet"
    temp_anchor.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(temp_anchor, index=False)
    run_rebalance_frequency_study(
        anchor_universe_path=temp_anchor,
        feature_results_path=Path(args.feature_results_path),
        output_dir=Path(args.output_dir),
        start_date=args.start_date or None,
        end_date=args.end_date or None,
    )


if __name__ == "__main__":
    main()
