# Event-driven picks — 2026-05-04

**Strategy:** Top-10 basket @ score_cal ≥ 0.65 on any of (5%/7d, 10%/15d, 20%/30d).

**Backtest evidence (9-yr walk-forward OOS, BASELINE model — no macro):**
- CAGR +154.8% · Sharpe 6.18 · Max drawdown -20.1%
- Hit rate 56% per trade · ~575 trades/yr · ~2 trades/trading-day with MAX_POS=10
- Source: `reports/event_driven_backtest_backtest_10yr_oof.md`
- IMPORTANT: macro features tested in F round did NOT help on average. Baseline model is honest deployment.

**Caveats:** Sweep bias (top of 20 configs); Sharpe 6.17 likely overstated by 30-40% after slippage; performance concentrated in 2018 + 2024 folds; backtest is OOS-walk-forward but config was selected from sweep — forward paper-trade for genuine validation before scaling capital.

**Sizing:** ₹10% per name. SL: -3%. Target: +5%. Hold: ≤7 trading days.

## Today's qualified picks (0 / 10 slots)

🔴 **No names cross 0.65 bar today.** All capital → LIQUIDPLUS / CASHIETF (~7% ann).

Top-10 by score (still BELOW 0.65 bar):

| # | Symbol | Close | best score | 5%/7d | 10%/15d | 20%/30d |
|---|---|---:|---:|---:|---:|---:|
| 1 | **AUSOMENT** | ₹147.79 | 0.535 | 0.535 | 0.463 | 0.344 |
| 2 | **GOODYEAR** | ₹780.30 | 0.535 | 0.535 | 0.463 | 0.300 |
| 3 | **NOVARTIND** | ₹1030.50 | 0.535 | 0.535 | 0.463 | 0.331 |
| 4 | **MANAKALUCO** | ₹37.32 | 0.526 | 0.526 | 0.463 | 0.322 |
| 5 | **GRAUWEIL** | ₹69.55 | 0.526 | 0.526 | 0.463 | 0.322 |
| 6 | **BESTAGRO** | ₹17.96 | 0.521 | 0.521 | 0.443 | 0.322 |
| 7 | **HARDWYN** | ₹27.19 | 0.520 | 0.520 | 0.443 | 0.289 |
| 8 | **TIL** | ₹197.16 | 0.509 | 0.509 | 0.463 | 0.249 |
| 9 | **NGLFINE** | ₹2214.50 | 0.509 | 0.509 | 0.443 | 0.300 |
| 10 | **DECCANCE** | ₹654.70 | 0.509 | 0.509 | 0.463 | 0.272 |

## Comparison vs prior 0.95-gate plan

| | 0.95 single-name (prior) | 0.65 Top-10 (NEW) |
|---|---:|---:|
| 9-yr backtested CAGR | +12.2% | **+144.3%** |
| Trades/year | 8 | 534 |
| Capital deployment % | 5% | ~50% |
| Max drawdown | -17.7% | -21.6% |
| Sharpe ratio | 1.11 | 6.17 |