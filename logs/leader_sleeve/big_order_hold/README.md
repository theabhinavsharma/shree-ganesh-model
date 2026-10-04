# Big order, 18 / 24 month hold (EXP-2026-10-04-big-order-hold-18-24m)

What this tests: buy every company that files an order win worth at least 15% of its last 12 months' sales (any
industry, cleaned amounts, no bad filing in the prior 90 days, profitable, liquid) at the next open, and hold 18 months
(A18) or 24 months (A24). Registered in `logs/experiments.jsonl` before any outcome; script `src/agentic/test_big_order_hold.py`.

Files:
- `reference_trades.parquet` — the exploration's trade list; the test must reproduce it exactly (wiring check).
- `trades_A18.csv`, `trades_A24.csv` — every trade in the test window (entries 2019-01-01..2024-12-31 with a full hold),
  with its return and the typical stock's return over the same dates. Units in each `.manifest.json`.

Result (2026-10-04): no arm passes. Both beat the typical stock in both eras, in 5 of 5 entry lags, and lost money in
<= 2% of 12-month buying windows, but the portfolio's worst fall (-66%, Jun 2019 to Apr 2020, when only 4-6 trades were
open) was more than 10 points worse than the market's (-50%). Not out of sample: 2019+ was explored before registering.

## Variant: orders >= 50% of revenue (EXP-2026-10-04-big-order-hold-18-24m-50pct)
Same rules on the >= 50% subset (`trades_A18_50pct.csv`, `trades_A24_50pct.csv`). Result (2026-10-04): both arms PASS
the registered checks (eras, 5/5 lags, windows lost 7% / 9%, drawdown -52% / -49% vs market -43%). But as a portfolio
(equal money in each open trade) A18 made +17.8%/yr and A24 +27.1%/yr against +29.0% / +28.4% for an equal-weight basket of
every NSE stock on the same days: the per-trade edge is measured against the median stock, which is a low bar.
