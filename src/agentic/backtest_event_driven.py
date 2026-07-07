"""Event-driven, fixed-capital, signal-gated backtest.

Mirrors what the user actually does in real life:
  • Score every stock every day (already done — read OOF parquet)
  • If at least one name fires at calibrated score >= GATE → enter (up to MAX_POS)
  • If 0 names fire → sit in cash earning CASH_ANN
  • Each open position: held until target / SL / TIME_OUT trading days
  • Capital ROTATES — when a position exits, freed cash refills the slot pool
  • Fixed starting capital (₹100); equity curve compounds through time

This is fundamentally different from the prior `backtest_dynamic_gated.py`
which hand-waved a "blended ann ROI" by averaging fire-day basket returns
× cash-day daily yields without simulating an actual portfolio.

Inputs
  data/derived/backtest_10yr_oof.parquet  (date, symbol, score_cal, fwd_c2c_7, winner_5pct_7td)
  data/derived/stock_daily_facts_adjusted_2015plus.parquet  (for daily MTM)

Configurations swept
  GATE × MAX_POS × HOLD_DAYS × USE_STOP_LOSS

Outputs
  data/derived/event_driven_equity_<config>.parquet — daily equity curve
  data/derived/event_driven_trades_<config>.parquet — every trade
  reports/event_driven_backtest.md                  — comparative report

Headline numbers per config:
  CAGR · max DD · % time deployed · n trades · hit rate · sharpe · best/worst year
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
import os
OOF = ROOT / os.environ.get("OOF_FILE", "data/derived/backtest_10yr_oof.parquet")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / f"reports/event_driven_backtest_{OOF.stem}.md"

CASH_ANN = 0.07
CASH_DAILY = (1 + CASH_ANN) ** (1 / 252) - 1


# ─────────────────────────────────────────────────────────────────────────
# Position state
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float            # capital_at_entry / entry_price
    capital_at_entry: float
    target_price: float
    sl_price: float
    exit_by_date: pd.Timestamp


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────
def load_oof() -> pd.DataFrame:
    df = pd.read_parquet(OOF)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=["score_cal"])
    return df


def load_prices_for(symbols: set[str], start: pd.Timestamp) -> pd.DataFrame:
    """Return long-format price history for the symbols we'll need."""
    cols = ["symbol", "trade_date", "close", "high", "low", "open", "avg_traded_value_20d"]
    px = pd.read_parquet(PRICES, columns=cols)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[px["symbol"].isin(symbols) & (px["trade_date"] >= start)]
    return px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────
def simulate(
    oof: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    GATE: float = 0.95,
    MAX_POS: int = 5,
    HOLD_DAYS: int = 7,
    TARGET_PCT: float = 0.05,
    SL_PCT: float = -0.03,    # set to None to disable
    MIN_ADV_CR: float = 1.0,
    STARTING_CAPITAL: float = 100.0,
    label: str = "default",
) -> dict:
    """Run one configuration. Returns equity_curve, trade_log, stats."""

    # 1. fire candidates: every (date, symbol) where score_cal >= GATE
    fires = oof[oof["score_cal"] >= GATE].copy()

    # 2. join price history into a fast lookup: (symbol, date) → row
    px = prices.copy()
    px["adv_20d_cr"] = px["avg_traded_value_20d"] / 1e7
    # liquidity filter on entry — only consider names with ADV >= MIN_ADV_CR on signal day
    fires = fires.merge(
        px[["symbol", "trade_date", "close", "adv_20d_cr"]].rename(columns={"close": "signal_close"}),
        on=["symbol", "trade_date"], how="left"
    )
    fires = fires[fires["adv_20d_cr"] >= MIN_ADV_CR]
    fires = fires.dropna(subset=["signal_close"])

    # group fires by date for fast lookup
    fires_by_date = {d: g.sort_values("score_cal", ascending=False)
                     for d, g in fires.groupby("trade_date")}

    # price lookup: (symbol, date) -> {open, high, low, close}
    px_idx = px.set_index(["symbol", "trade_date"])

    # full date spine
    all_dates = sorted(oof["trade_date"].unique())

    # 3. simulate
    cash = STARTING_CAPITAL
    open_positions: list[Position] = []
    equity_curve = []
    trades = []
    held_symbols: set[str] = set()

    for date in all_dates:
        # ── A. cash accrues at LIQUIDPLUS rate ──
        cash *= (1 + CASH_DAILY)

        # ── B. check exits on each open position ──
        still_open = []
        for pos in open_positions:
            try:
                bar = px_idx.loc[(pos.symbol, date)]
            except KeyError:
                # no price today — keep position, skip
                still_open.append(pos)
                continue

            high = float(bar["high"]) if pd.notna(bar["high"]) else float(bar["close"])
            low  = float(bar["low"])  if pd.notna(bar["low"])  else float(bar["close"])
            close= float(bar["close"])

            exit_price = None
            exit_reason = None

            # target hit (intraday high reaches target)
            if high >= pos.target_price:
                exit_price = pos.target_price
                exit_reason = "TARGET"
            # stop-loss
            elif SL_PCT is not None and low <= pos.sl_price:
                exit_price = pos.sl_price
                exit_reason = "STOP"
            # time-out
            elif date >= pos.exit_by_date:
                exit_price = close
                exit_reason = "TIMEOUT"

            if exit_price is not None:
                proceeds = pos.shares * exit_price
                cash += proceeds
                pnl = proceeds - pos.capital_at_entry
                ret = (exit_price / pos.entry_price) - 1
                hold_days = (date - pos.entry_date).days
                trades.append({
                    "entry_date": pos.entry_date,
                    "exit_date": date,
                    "symbol": pos.symbol,
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "capital_at_entry": pos.capital_at_entry,
                    "proceeds": proceeds,
                    "pnl": pnl,
                    "return_pct": ret,
                    "hold_days": hold_days,
                    "exit_reason": exit_reason,
                })
                held_symbols.discard(pos.symbol)
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── C. check for new entries ──
        slots_free = MAX_POS - len(open_positions)
        if slots_free > 0 and date in fires_by_date:
            today_fires = fires_by_date[date]
            # exclude names already in portfolio
            today_fires = today_fires[~today_fires["symbol"].isin(held_symbols)]
            picks = today_fires.head(slots_free)

            if len(picks) > 0:
                # equal-weight allocation: cash / slots_free per new name
                # (this rebalances cash across the new positions + keeps existing)
                cash_per_new = cash / slots_free
                for _, r in picks.iterrows():
                    sym = r["symbol"]
                    entry_price = float(r["signal_close"])
                    # in production we'd use NEXT day's open; using close here matches OOF target labelling
                    if entry_price <= 0:
                        continue
                    # exit_by_date = HOLD_DAYS trading days from entry
                    fut_dates = [d for d in all_dates if d > date]
                    if len(fut_dates) < HOLD_DAYS:
                        continue
                    exit_by = fut_dates[HOLD_DAYS - 1]

                    shares = cash_per_new / entry_price
                    capital_at_entry = shares * entry_price
                    pos = Position(
                        symbol=sym,
                        entry_date=date,
                        entry_price=entry_price,
                        shares=shares,
                        capital_at_entry=capital_at_entry,
                        target_price=entry_price * (1 + TARGET_PCT),
                        sl_price=entry_price * (1 + SL_PCT) if SL_PCT is not None else 0,
                        exit_by_date=exit_by,
                    )
                    open_positions.append(pos)
                    held_symbols.add(sym)
                    cash -= capital_at_entry

        # ── D. record equity ──
        # mark open positions to today's close
        mtm = 0.0
        for pos in open_positions:
            try:
                bar = px_idx.loc[(pos.symbol, date)]
                mtm += pos.shares * float(bar["close"])
            except KeyError:
                mtm += pos.capital_at_entry  # stale value
        equity = cash + mtm
        equity_curve.append({
            "trade_date": date,
            "equity": equity,
            "cash": cash,
            "n_open_positions": len(open_positions),
            "deployed_pct": (mtm / equity) if equity > 0 else 0,
        })

    # 4. close any open positions at final date (mark to last close)
    last_date = all_dates[-1] if all_dates else None
    for pos in open_positions:
        try:
            bar = px_idx.loc[(pos.symbol, last_date)]
            exit_price = float(bar["close"])
        except (KeyError, TypeError):
            exit_price = pos.entry_price
        proceeds = pos.shares * exit_price
        cash += proceeds
        trades.append({
            "entry_date": pos.entry_date,
            "exit_date": last_date,
            "symbol": pos.symbol,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "shares": pos.shares,
            "capital_at_entry": pos.capital_at_entry,
            "proceeds": proceeds,
            "pnl": proceeds - pos.capital_at_entry,
            "return_pct": (exit_price / pos.entry_price) - 1,
            "hold_days": (last_date - pos.entry_date).days,
            "exit_reason": "EOD_LIQUIDATE",
        })

    eq = pd.DataFrame(equity_curve)
    tr = pd.DataFrame(trades)

    # 5. stats
    if len(eq) >= 2 and eq["equity"].iloc[0] > 0:
        years = (eq["trade_date"].iloc[-1] - eq["trade_date"].iloc[0]).days / 365.25
        cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / max(years, 1e-9)) - 1
        running_peak = eq["equity"].cummax()
        dd = (eq["equity"] / running_peak - 1)
        max_dd = float(dd.min())
        # daily returns for sharpe
        eq["daily_ret"] = eq["equity"].pct_change()
        ann_vol = eq["daily_ret"].std() * (252 ** 0.5)
        ann_ret = eq["daily_ret"].mean() * 252
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        deployed_pct_mean = eq["deployed_pct"].mean()
    else:
        cagr, max_dd, sharpe, deployed_pct_mean, ann_vol = 0, 0, 0, 0, 0

    if len(tr):
        hit_rate = (tr["return_pct"] > 0).mean()
        avg_win = tr.loc[tr["return_pct"] > 0, "return_pct"].mean()
        avg_loss = tr.loc[tr["return_pct"] < 0, "return_pct"].mean() if (tr["return_pct"] < 0).any() else 0
        avg_hold = tr["hold_days"].mean()
        n_trades = len(tr)
        avg_ret = tr["return_pct"].mean()
    else:
        hit_rate, avg_win, avg_loss, avg_hold, n_trades, avg_ret = 0, 0, 0, 0, 0, 0

    # per-year breakdown
    yr_stats = []
    if len(eq):
        eq["year"] = eq["trade_date"].dt.year
        for yr, g in eq.groupby("year"):
            if len(g) < 2: continue
            yr_ret = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
            yr_stats.append({"year": int(yr), "year_return": yr_ret,
                             "deployed_pct_mean": float(g["deployed_pct"].mean())})
    yr_df = pd.DataFrame(yr_stats)

    stats = {
        "label": label,
        "GATE": GATE, "MAX_POS": MAX_POS, "HOLD_DAYS": HOLD_DAYS,
        "TARGET_PCT": TARGET_PCT, "SL_PCT": SL_PCT, "MIN_ADV_CR": MIN_ADV_CR,
        "n_trades": int(n_trades),
        "hit_rate": float(hit_rate),
        "avg_return_per_trade": float(avg_ret),
        "avg_win_per_trade": float(avg_win) if pd.notna(avg_win) else 0,
        "avg_loss_per_trade": float(avg_loss) if pd.notna(avg_loss) else 0,
        "avg_hold_days": float(avg_hold),
        "deployed_pct_mean": float(deployed_pct_mean),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "ann_vol": float(ann_vol),
        "final_equity": float(eq["equity"].iloc[-1]) if len(eq) else 0,
        "starting_equity": float(STARTING_CAPITAL),
        "n_years": float(years) if len(eq) else 0,
    }

    # save
    suffix = f"GATE{int(GATE*100):02d}_POS{MAX_POS}_HOLD{HOLD_DAYS}_T{int(TARGET_PCT*100):02d}_SL{int(SL_PCT*100) if SL_PCT else 0:+03d}"
    eq.to_parquet(ROOT / f"data/derived/event_driven_equity_{suffix}.parquet", index=False)
    tr.to_parquet(ROOT / f"data/derived/event_driven_trades_{suffix}.parquet", index=False)

    return {"stats": stats, "equity": eq, "trades": tr, "year_stats": yr_df, "suffix": suffix}


def main() -> None:
    print("== backtest_event_driven (fixed-capital, signal-gated, rotating) ==\n")

    print("Loading OOF predictions...")
    oof = load_oof()
    print(f"  {len(oof):,} rows · {oof['symbol'].nunique()} symbols · "
          f"{oof['trade_date'].min():%Y-%m-%d} → {oof['trade_date'].max():%Y-%m-%d}")

    # only need prices for the universe of symbols we might trade
    symbols = set(oof.loc[oof["score_cal"] >= 0.80, "symbol"].unique())
    print(f"  loading prices for {len(symbols)} symbols (those that ever fire >=0.80)...")
    prices = load_prices_for(symbols, oof["trade_date"].min())
    print(f"  prices: {len(prices):,} rows\n")

    # parameter sweep
    configs = [
        # (label, GATE, MAX_POS, HOLD, TARGET, SL)
        ("Single-name 0.95",         0.95, 1, 7, 0.05, -0.03),
        ("Top-3 basket 0.95",        0.95, 3, 7, 0.05, -0.03),
        ("Top-5 basket 0.95",        0.95, 5, 7, 0.05, -0.03),
        ("Top-5 basket 0.95 noSL",   0.95, 5, 7, 0.05, None),
        ("Single-name 0.85",         0.85, 1, 7, 0.05, -0.03),
        ("Top-5 basket 0.85",        0.85, 5, 7, 0.05, -0.03),
        ("Top-5 basket 0.80",        0.80, 5, 7, 0.05, -0.03),
        ("Single-name 0.95 wider",   0.95, 1, 10, 0.07, -0.04),
        # ─── GATE-TUNING GRID (per PM directive, F result <5pp ─ rework gating) ───
        ("Top-10 basket 0.80",       0.80, 10, 7, 0.05, -0.03),
        ("Top-10 basket 0.75",       0.75, 10, 7, 0.05, -0.03),
        ("Top-10 basket 0.70",       0.70, 10, 7, 0.05, -0.03),
        ("Top-10 basket 0.65",       0.65, 10, 7, 0.05, -0.03),
        ("Top-20 basket 0.65",       0.65, 20, 7, 0.05, -0.03),
        ("Top-5 basket 0.80 hold15", 0.80, 5, 15, 0.10, -0.05),
        ("Top-5 basket 0.80 hold30", 0.80, 5, 30, 0.20, -0.07),
        ("Top-5 basket 0.80 noSL",   0.80, 5, 7, 0.05, None),
        ("Top-5 basket 0.70 noSL",   0.70, 5, 7, 0.05, None),
        ("Top-10 basket 0.70 noSL",  0.70, 10, 7, 0.05, None),
        ("Top-5 basket 0.80 widerT", 0.80, 5, 10, 0.08, -0.04),
        ("Single-name 0.70",         0.70, 1, 7, 0.05, -0.03),
    ]

    results = []
    for (label, gate, mp, hold, tgt, sl) in configs:
        print(f"--- running: {label}  (GATE={gate}, MAX_POS={mp}, HOLD={hold}d, TGT={tgt:+.0%}, SL={sl})")
        out = simulate(oof, prices, GATE=gate, MAX_POS=mp, HOLD_DAYS=hold,
                        TARGET_PCT=tgt, SL_PCT=sl, label=label)
        s = out["stats"]
        print(f"    n_trades={s['n_trades']}  hit={s['hit_rate']:.0%}  "
              f"avg_per_trade={s['avg_return_per_trade']:+.2%}  "
              f"deployed={s['deployed_pct_mean']:.0%}  "
              f"CAGR={s['cagr']:+.1%}  maxDD={s['max_drawdown']:+.1%}  Sharpe={s['sharpe']:.2f}")
        results.append(out)

    # build comparative report
    md = ["# Event-driven, fixed-capital, signal-gated backtest", ""]
    md.append("Mirrors the user's actual behavior:")
    md.append("- Score every stock every day (OOF, walk-forward, isotonic-calibrated).")
    md.append("- Trade ONLY when ≥1 name fires at the calibrated bar (`score_cal ≥ GATE`).")
    md.append("- Sit in cash @ 7% ann on no-fire days.")
    md.append("- Capital ROTATES: when a position exits (target / SL / time-out), the freed cash refills slots.")
    md.append("- Fixed starting capital ₹100; equity curve compounds through time.")
    md.append("")
    md.append(f"Universe: 9-year walk-forward, 2017-01-02 → 2025-12-31 ({len(oof['trade_date'].unique())} trading days)")
    md.append(f"Cash yield: {CASH_ANN*100:.0f}% ann (LIQUIDPLUS proxy)")
    md.append("")
    md.append("## Configurations")
    md.append("")
    md.append("| Config | GATE | Slots | Hold | Target | SL | n_trades | Hit % | Avg/trade | Deployed % | **CAGR** | Max DD | Sharpe |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        s = r["stats"]
        sl_str = f"{s['SL_PCT']*100:+.0f}%" if s['SL_PCT'] is not None else "none"
        md.append(
            f"| {s['label']} | {s['GATE']:.2f} | {s['MAX_POS']} | {s['HOLD_DAYS']}d | "
            f"{s['TARGET_PCT']*100:+.0f}% | {sl_str} | "
            f"{s['n_trades']} | {s['hit_rate']*100:.0f}% | "
            f"{s['avg_return_per_trade']*100:+.2f}% | "
            f"{s['deployed_pct_mean']*100:.0f}% | "
            f"**{s['cagr']*100:+.1f}%** | {s['max_drawdown']*100:+.1f}% | {s['sharpe']:.2f} |"
        )
    md.append("")

    # per-year breakdown for the 0.95 configs (the headline)
    md.append("## Per-year returns (0.95-bar configs)")
    md.append("")
    md.append("| Year | Single 0.95 | Top-3 0.95 | Top-5 0.95 | Top-5 0.85 |")
    md.append("|---|---:|---:|---:|---:|")
    yr_pivot = {}
    for r in results:
        if r["stats"]["GATE"] in (0.95, 0.85) and r["stats"]["MAX_POS"] in (1, 3, 5):
            for _, row in r["year_stats"].iterrows():
                yr_pivot.setdefault(int(row["year"]), {})[r["stats"]["label"]] = row["year_return"]
    for yr in sorted(yr_pivot.keys()):
        v = yr_pivot[yr]
        cells = []
        for label in ["Single-name 0.95", "Top-3 basket 0.95", "Top-5 basket 0.95", "Top-5 basket 0.85"]:
            x = v.get(label, np.nan)
            cells.append(f"{x*100:+.1f}%" if pd.notna(x) else "—")
        md.append(f"| {yr} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    md.append("")

    md.append("## Reading this")
    md.append("")
    md.append("- **Avg/trade** is the per-position close-to-close return when in a position.")
    md.append("- **Deployed %** is the fraction of trading days the portfolio is invested (vs in cash).")
    md.append("- **CAGR** is the compounded equity-curve growth — the honest number that matches what fixed capital would actually do.")
    md.append("- **Max DD** is the deepest peak-to-trough drawdown of the equity curve, NOT a single-trade SL.")
    md.append("")
    md.append("### Why this differs from the prior 9-year backtest")
    md.append("")
    md.append("The earlier number forced top-5 baskets EVERY day (2,290 days × 5 names = 11,450 trades). "
              "Most days the model wasn't above 0.95 — those were forced trades on noise. "
              "By gating to fire-only, we cut to ~3,000 high-conviction trades over 9 years and let "
              "cash earn 7% on the other ~95% of days.")
    md.append("")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))

    # combined stats parquet
    stats_df = pd.DataFrame([r["stats"] for r in results])
    stats_df.to_parquet(ROOT / "data/derived/event_driven_summary.parquet", index=False)

    print(f"\nwrote {OUT_REPORT}")
    print(f"wrote per-config equity curves and trade logs to data/derived/event_driven_*.parquet")
    print(f"\nHEADLINE:")
    for r in results:
        s = r["stats"]
        print(f"  {s['label']:<28} CAGR={s['cagr']*100:+6.1f}%  "
              f"DD={s['max_drawdown']*100:+5.1f}%  Sharpe={s['sharpe']:5.2f}  "
              f"n={s['n_trades']:>4}  hit={s['hit_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
