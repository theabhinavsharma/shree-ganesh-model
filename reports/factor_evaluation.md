# Factor Evaluation — 2026-04-29

Horizon: 7 trading days forward close-to-close.

Thresholds: IC mean ≥ 0.02, |IR| ≥ 0.5.

| feature | n | ic_mean | ic_t | decile_spread | ir_annualised | verdict |
|---|---:|---:|---:|---:|---:|---|
| `alpha_geom_mid_vs_vwap` | 1,475,771 | 0.01892 | 9.08 | 0.00066 | 0.209 | **DROP** |
| `scr_compounded_sales_growth_5_years` | 118,067 | 0.01687 | 4.16 | 0.00295 | 1.649 | **DROP** |
| `alpha_open_volume_corr_10` | 1,468,520 | 0.0168 | 10.01 | 0.00299 | 2.086 | **DROP** |
| `scr_return_on_equity_5_years` | 115,511 | 0.01404 | 3.25 | 0.00367 | 2.51 | **DROP** |
| `scr_compounded_sales_growth_3_years` | 121,497 | 0.01386 | 2.82 | 0.00456 | 2.287 | **DROP** |
| `scr_book_value` | 123,712 | 0.01193 | 2.42 | -0.00358 | -1.964 | **DROP** |
| `scr_market_cap_cr` | 124,639 | 0.01014 | 1.43 | -0.00696 | -2.747 | **DROP** |
| `scr_earnings_yield` | 121,322 | 0.0029 | 0.52 | -0.00171 | -0.751 | **DROP** |
| `scr_peg_3y` | 117,741 | -0.00077 | -0.18 | 0.00017 | 0.135 | **DROP** |
| `scr_pe` | 121,322 | -0.0029 | -0.52 | 0.00146 | 0.644 | **DROP** |
| `vol_max_63d` | 1,427,473 | -0.00531 | -1.71 | -0.00173 | -1.321 | **DROP** |
| `vol_term_20_60` | 1,427,872 | -0.00975 | -4.58 | 0.0014 | 0.475 | **DROP** |
| `vol_z_60d` | 1,413,040 | -0.01489 | -6.76 | -0.00092 | -0.635 | **DROP** |
| `alpha_intraday_norm_range` | 1,472,170 | -0.02797 | -12.11 | -0.00059 | -0.298 | **DROP** |
| `scr_stock_price_cagr_3_years` | 118,773 | 0.08139 | 13.06 | 0.02348 | 10.383 | **IC_PASSED** |
| `scr_stock_price_cagr_5_years` | 115,613 | 0.07139 | 11.89 | 0.02027 | 9.259 | **IC_PASSED** |
| `scr_stock_price_cagr_1_year` | 124,340 | 0.0539 | 9.32 | 0.01326 | 5.706 | **IC_PASSED** |
| `scr_compounded_profit_growth_5_years` | 116,949 | 0.04119 | 10.22 | 0.00815 | 5.848 | **IC_PASSED** |
| `scr_roe` | 123,712 | 0.03699 | 9.52 | 0.00634 | 4.189 | **IC_PASSED** |
| `scr_compounded_profit_growth_3_years` | 118,842 | 0.02952 | 7.26 | 0.00852 | 4.945 | **IC_PASSED** |
| `scr_roce` | 124,452 | 0.02906 | 6.38 | 0.00553 | 3.522 | **IC_PASSED** |
| `scr_return_on_equity_3_years` | 119,785 | 0.02316 | 5.6 | 0.00229 | 1.488 | **IC_PASSED** |
| `alpha_volume_signed_revert` | 1,474,966 | 0.02258 | 16.55 | 0.00203 | 3.546 | **IC_PASSED** |
| `turnover_skew_20d` | 1,445,692 | -0.0235 | -17.18 | -0.00231 | -1.653 | **IC_PASSED** |
| `scr_price_to_book` | 123,712 | -0.02505 | -4.24 | -0.00969 | -4.723 | **IC_PASSED** |
| `amihud_20d` | 1,459,758 | -0.03075 | -9.1 | 0.00926 | 2.269 | **IC_PASSED** |
| `vol_of_vol_60d` | 1,413,354 | -0.05937 | -11.8 | 0.00753 | 1.693 | **IC_PASSED** |
| `rv_60d` | 1,428,225 | -0.0663 | -10.24 | 0.00842 | 1.821 | **IC_PASSED** |
| `alpha_high_extension_revert` | 1,475,771 | 0.01698 | 9.27 | -0.00192 | nan | **INSUFFICIENT** |
| `eurinr` | 1,145,616 | 0.0128 | nan | nan | nan | **INSUFFICIENT** |
| `gbpinr` | 1,145,616 | 0.01279 | nan | nan | nan | **INSUFFICIENT** |
| `scr_dividend_yield` | 124,639 | 0.00777 | 1.38 | nan | nan | **INSUFFICIENT** |
| `jpyinr_5d_chg` | 1,132,225 | 0.00273 | 2.17 | nan | nan | **INSUFFICIENT** |
| `jpyinr_20d_chg` | 1,092,330 | 0.00215 | 1.38 | nan | nan | **INSUFFICIENT** |
| `gbpinr_20d_chg` | 1,092,330 | -0.00044 | -0.26 | nan | nan | **INSUFFICIENT** |
| `eurinr_20d_chg` | 1,092,330 | -0.00071 | -0.42 | nan | nan | **INSUFFICIENT** |
| `gbpinr_5d_chg` | 1,132,225 | -0.00138 | -1.06 | nan | nan | **INSUFFICIENT** |
| `usdinr_5d_chg` | 1,132,225 | -0.0038 | -3.13 | nan | nan | **INSUFFICIENT** |
| `eurinr_5d_chg` | 1,132,225 | -0.00386 | -3.09 | nan | nan | **INSUFFICIENT** |
| `jpyinr` | 1,145,616 | -0.00457 | nan | nan | nan | **INSUFFICIENT** |
| `usdinr_20d_chg` | 1,092,330 | -0.01072 | -6.95 | nan | nan | **INSUFFICIENT** |
| `usdinr` | 1,145,616 | -0.01163 | nan | nan | nan | **INSUFFICIENT** |
| `wiki_views` | 1,275 | -0.20465 | nan | -0.04703 | nan | **INSUFFICIENT** |
| `wiki_views_z` | 255 | -0.22617 | nan | -0.04325 | nan | **INSUFFICIENT** |
| `dow` | 1,475,771 | nan | nan | nan | nan | **INSUFFICIENT** |
| `dom` | 1,475,771 | nan | nan | nan | nan | **INSUFFICIENT** |
| `is_month_end_3d` | 1,475,771 | nan | nan | nan | nan | **INSUFFICIENT** |
| `is_expiry_week` | 1,475,771 | nan | nan | nan | nan | **INSUFFICIENT** |