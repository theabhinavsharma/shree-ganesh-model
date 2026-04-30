# 10-year Backtest WITH 5 KEEP factors

Added factors: alpha_volume_signed_revert, amihud_20d, rv_60d, vol_of_vol_60d, turnover_skew_20d, scr_stock_price_cagr_3_years, scr_stock_price_cagr_5_years, scr_stock_price_cagr_1_year, scr_compounded_profit_growth_5_years, scr_compounded_profit_growth_3_years, scr_roe, scr_roce, scr_price_to_book

**Note:** factor data starts ~2023-06, so pre-2024 folds get median-fill (no real signal).
Compare years 2024 + 2025 against baseline `backtest_10yr_summary.md` for the A/B verdict.

## Per-year summary

| Year | OOS days | mean 7d | median 7d | days ≥+5% (%) | days <0 (%) |
|---:|---:|---:|---:|---:|---:|
| 2017 | 248 | +1.73% | +1.29% | 32.7% | 44.4% |
| 2018 | 246 | -0.46% | -0.77% | 23.6% | 57.3% |
| 2019 | 244 | -0.80% | -1.47% | 23.8% | 56.6% |
| 2020 | 261 | +3.64% | +3.70% | 46.4% | 37.2% |
| 2021 | 261 | +5.75% | +4.80% | 49.0% | 26.4% |
| 2022 | 259 | +1.81% | +1.93% | 31.7% | 40.5% |
| 2023 | 260 | +7.50% | +3.67% | 45.8% | 30.0% |
| 2024 | 261 | +0.12% | +0.51% | 26.8% | 46.4% |
| 2025 | 250 | +1.04% | +0.70% | 24.8% | 46.0% |
| **all** | 2290 | **+2.31%** | +1.54% | 34.0% | 42.5% |