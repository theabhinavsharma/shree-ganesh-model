# Event-driven picks — 2026-05-01

**Strategy:** Top-10 basket @ score_cal ≥ 0.65 on any of (5%/7d, 10%/15d, 20%/30d).

**Backtest evidence (9-yr walk-forward OOS, macro-enriched):**
- CAGR +144.3% · Sharpe 6.17 · Max drawdown -21.6%
- Hit rate 57% per trade · ~534 trades/yr · ~2 trades/trading-day with MAX_POS=10
- Source: `reports/event_driven_backtest_backtest_10yr_macro_oof.md`

**Caveats:** Sweep bias (top of 20 configs); Sharpe 6.17 likely overstated by 30-40% after slippage; performance concentrated in 2018 + 2024 folds; backtest is OOS-walk-forward but config was selected from sweep — forward paper-trade for genuine validation before scaling capital.

**Sizing:** ₹10% per name. SL: -3%. Target: +5%. Hold: ≤7 trading days.

## Today's qualified picks (0 / 10 slots)

🔴 **No names cross 0.65 bar today.** All capital → LIQUIDPLUS / CASHIETF (~7% ann).

Top-10 by score (still BELOW 0.65 bar):

| # | Symbol | Close | best score | 5%/7d | 10%/15d | 20%/30d |
|---|---|---:|---:|---:|---:|---:|
| 1 | **DOLLAR** | ₹299.80 | 0.532 | 0.532 | 0.391 | 0.175 |
| 2 | **MTARTECH** | ₹6457.10 | 0.532 | 0.532 | 0.459 | 0.222 |
| 3 | **RTNINDIA** | ₹35.52 | 0.532 | 0.532 | 0.415 | 0.186 |
| 4 | **BHARATWIRE** | ₹222.03 | 0.532 | 0.532 | 0.415 | 0.201 |
| 5 | **SEPC** | ₹8.11 | 0.531 | 0.531 | 0.363 | 0.153 |
| 6 | **TDPOWERSYS** | ₹1152.70 | 0.528 | 0.528 | 0.334 | 0.190 |
| 7 | **OPTIEMUS** | ₹415.80 | 0.528 | 0.528 | 0.463 | 0.194 |
| 8 | **TANLA** | ₹511.95 | 0.528 | 0.528 | 0.391 | 0.145 |
| 9 | **VPRPL** | ₹43.90 | 0.527 | 0.527 | 0.452 | 0.197 |
| 10 | **MANINDS** | ₹526.65 | 0.527 | 0.527 | 0.452 | 0.236 |

## Comparison vs prior 0.95-gate plan

| | 0.95 single-name (prior) | 0.65 Top-10 (NEW) |
|---|---:|---:|
| 9-yr backtested CAGR | +12.2% | **+144.3%** |
| Trades/year | 8 | 534 |
| Capital deployment % | 5% | ~50% |
| Max drawdown | -17.7% | -21.6% |
| Sharpe ratio | 1.11 | 6.17 |