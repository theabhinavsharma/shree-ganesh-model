# Macro Factor Evaluation — 2026-04-30

Time-series IC + regime-split test for date-level features (commodities, global rates, breadth, MF AUM, macro sentiment).

Method:
- Target: median 7d forward return across liquid universe (NIFTY-equivalent broad market).
- IC: Spearman correlation of (feature − 252d rolling mean) vs (market_fwd_ret_7 − 252d rolling mean).
- Regime split: top tertile vs bottom tertile of feature → mean market 7d return spread.

Thresholds: KEEP ≥ |IC|=0.05 AND |spread|=0.5%; WATCHLIST ≥ |IC|=0.025 AND spread.

## Summary: KEEP=31  WATCHLIST=2  DROP=60  INSUFFICIENT=16

## All features (sorted by verdict then |ts_ic|)

| feature | n_dates | ts_ic | ts_ic_t | regime spread | regime IR | verdict |
|---|---:|---:|---:|---:|---:|---|
| `macro_equity_aum_mom_pct` | 504 | 0.2064 | 4.44 | +1.45% | 30.85 | **KEEP** |
| `eurinr_20d_chg` | 535 | 0.1697 | 3.75 | +1.05% | 23.01 | **KEEP** |
| `macro_em_hy_oas` | 742 | 0.1482 | 3.91 | +1.70% | 46.69 | **KEEP** |
| `macro_gold_ppi` | 742 | 0.1408 | 3.71 | +1.45% | 37.29 | **KEEP** |
| `macro_median_realized_vol_20d` | 742 | 0.1396 | 3.68 | +0.97% | 23.32 | **KEEP** |
| `macro_hy_oas` | 742 | 0.1374 | 3.62 | +1.95% | 53.5 | **KEEP** |
| `gbpinr_20d_chg` | 535 | 0.1294 | 2.84 | +0.88% | 17.14 | **KEEP** |
| `macro_us_3m` | 742 | 0.1162 | 3.05 | +0.98% | 24.45 | **KEEP** |
| `macro_ig_oas` | 742 | 0.104 | 2.73 | +1.72% | 47.7 | **KEEP** |
| `jpyinr` | 555 | 0.0971 | 2.17 | +0.95% | 18.23 | **KEEP** |
| `macro_wheat` | 742 | 0.0866 | 2.27 | +1.10% | 31.58 | **KEEP** |
| `macro_copper_5d_pct` | 742 | 0.0828 | 2.17 | -0.65% | -17.58 | **KEEP** |
| `sec_sector_volume_z_60d` | 742 | 0.08 | 2.09 | +0.91% | 23.43 | **KEEP** |
| `jpyinr_20d_chg` | 535 | 0.0754 | 1.65 | +0.62% | 12.99 | **KEEP** |
| `macro_fed_funds` | 742 | 0.0737 | 1.93 | +1.00% | 23.42 | **KEEP** |
| `macro_usdjpy` | 742 | -0.0553 | -1.44 | -0.94% | -24.71 | **KEEP** |
| `macro_zinc` | 742 | -0.06 | -1.57 | -0.97% | -24.18 | **KEEP** |
| `macro_new_52w_lows` | 742 | -0.0664 | -1.74 | -0.74% | -19.81 | **KEEP** |
| `macro_wti` | 742 | -0.0675 | -1.77 | +0.52% | 13.15 | **KEEP** |
| `macro_brent` | 742 | -0.0701 | -1.83 | +0.60% | 15.5 | **KEEP** |
| `macro_declining` | 742 | -0.0791 | -2.07 | -0.72% | -17.74 | **KEEP** |
| `macro_aluminum` | 742 | -0.0857 | -2.24 | -1.08% | -30.89 | **KEEP** |
| `macro_cross_section_dispersion_20d` | 742 | -0.0957 | -2.51 | -1.17% | -28.58 | **KEEP** |
| `macro_breadth_50_5d_chg` | 742 | -0.1091 | -2.86 | -0.72% | -17.75 | **KEEP** |
| `macro_brent_5d_pct` | 742 | -0.1124 | -2.95 | -0.64% | -15.23 | **KEEP** |
| `macro_us_vix_z_60d` | 742 | -0.1218 | -3.2 | -0.76% | -20.66 | **KEEP** |
| `macro_spx` | 742 | -0.1281 | -3.37 | -1.19% | -33.49 | **KEEP** |
| `macro_brent_60d_pct` | 742 | -0.1409 | -3.71 | -0.78% | -19.15 | **KEEP** |
| `macro_smid_lcap_breadth_diff` | 742 | -0.1725 | -4.57 | -0.99% | -25.09 | **KEEP** |
| `macro_new_52w_highs` | 742 | -0.1907 | -5.07 | -1.05% | -26.68 | **KEEP** |
| `macro_sip_inflow_yoy_pct` | 250 | -0.2056 | -2.89 | -2.94% | -43.15 | **KEEP** |
| `macro_natgas` | 742 | 0.04 | 1.05 | -0.73% | -17.91 | **WATCHLIST** |
| `macro_nickel` | 742 | 0.0381 | 1.0 | +0.86% | 21.75 | **WATCHLIST** |
| `macro_equity_aum_cr` | 504 | 0.2003 | 4.3 | -0.31% | -6.37 | **DROP** |
| `macro_total_aum_cr` | 504 | 0.2003 | 4.3 | -0.31% | -6.37 | **DROP** |
| `macro_sip_inflow_cr` | 504 | 0.1939 | 4.16 | -0.31% | -6.37 | **DROP** |
| `macro_equity_aum_yoy_pct` | 250 | 0.1745 | 2.44 | +0.33% | 5.06 | **DROP** |
| `macro_total_aum_yoy_pct` | 250 | 0.1745 | 2.44 | +0.33% | 5.06 | **DROP** |
| `eurinr` | 555 | 0.1188 | 2.66 | +0.27% | 5.23 | **DROP** |
| `macro_eurusd` | 742 | 0.118 | 3.1 | +0.00% | 0.03 | **DROP** |
| `gbpinr` | 555 | 0.114 | 2.55 | -0.13% | -2.65 | **DROP** |
| `eurinr_5d_chg` | 550 | 0.105 | 2.33 | +0.09% | 1.75 | **DROP** |
| `macro_copper_60d_pct` | 742 | 0.0614 | 1.61 | +0.03% | 0.68 | **DROP** |
| `macro_median_return_1d` | 742 | 0.0555 | 1.45 | +0.29% | 7.26 | **DROP** |
| `macro_downside_skew_count` | 742 | 0.0515 | 1.35 | -0.12% | -3.01 | **DROP** |
| `gbpinr_5d_chg` | 550 | 0.0474 | 1.05 | -0.40% | -8.23 | **DROP** |
| `macro_copper` | 742 | 0.0432 | 1.13 | -0.20% | -5.0 | **DROP** |
| `macro_adv_decl_ratio` | 742 | 0.0392 | 1.02 | +0.38% | 9.27 | **DROP** |
| `macro_spx_5d_pct` | 742 | 0.0389 | 1.02 | +0.22% | 5.65 | **DROP** |
| `usdinr_20d_chg` | 535 | 0.0365 | 0.8 | -0.33% | -6.57 | **DROP** |
| `macro_inr_20d_pct` | 535 | 0.0365 | 0.8 | -0.33% | -6.57 | **DROP** |
| `macro_us_10y` | 742 | 0.0293 | 0.76 | +0.18% | 4.6 | **DROP** |
| `macro_cotton` | 742 | 0.0222 | 0.58 | +1.11% | 28.91 | **DROP** |
| `jpyinr_5d_chg` | 550 | 0.017 | 0.38 | -0.20% | -3.83 | **DROP** |
| `macro_us_2y` | 742 | 0.015 | 0.39 | +0.89% | 24.17 | **DROP** |
| `macro_advancing` | 742 | 0.0141 | 0.37 | -0.10% | -2.42 | **DROP** |
| `macro_corn` | 742 | 0.0019 | 0.05 | +0.53% | 14.45 | **DROP** |
| `sec_rs_60d` | 742 | -0.0004 | -0.01 | +0.25% | 6.09 | **DROP** |
| `macro_usdcny` | 742 | -0.0064 | -0.17 | +0.21% | 5.17 | **DROP** |
| `sec_rs_20d` | 742 | -0.0126 | -0.33 | +0.53% | 13.57 | **DROP** |
| `macro_us_vix` | 742 | -0.0174 | -0.45 | -0.38% | -9.55 | **DROP** |
| `macro_n_universe` | 742 | -0.0176 | -0.46 | -0.89% | -22.34 | **DROP** |
| `macro_yc_10y2y` | 742 | -0.0196 | -0.51 | -0.72% | -20.04 | **DROP** |
| `macro_market_20d_sum` | 742 | -0.0496 | -1.3 | +0.17% | 4.03 | **DROP** |
| `usdinr` | 555 | -0.053 | -1.18 | -0.45% | -9.26 | **DROP** |
| `macro_sugar` | 742 | -0.0532 | -1.39 | +0.49% | 13.68 | **DROP** |
| `macro_spx_60d_pct` | 742 | -0.0583 | -1.52 | -0.34% | -8.22 | **DROP** |
| `usdinr_5d_chg` | 550 | -0.0603 | -1.33 | -0.20% | -3.99 | **DROP** |
| `macro_inr_5d_pct` | 550 | -0.0603 | -1.33 | -0.20% | -3.99 | **DROP** |
| `macro_us_10y_60d_chg` | 742 | -0.061 | -1.59 | -0.03% | -0.84 | **DROP** |
| `macro_dxy_5d_pct` | 742 | -0.063 | -1.65 | -0.28% | -7.47 | **DROP** |
| `sec_sector_leader_lag_spread` | 742 | -0.0659 | -1.72 | -0.09% | -2.41 | **DROP** |
| `macro_new_high_low_diff` | 742 | -0.0666 | -1.74 | -0.25% | -6.65 | **DROP** |
| `sec_sector_20d_ret` | 742 | -0.0676 | -1.77 | +0.29% | 7.23 | **DROP** |
| `macro_breadth_50_20d_chg` | 742 | -0.0682 | -1.78 | -0.06% | -1.4 | **DROP** |
| `macro_breadth_50_lcap` | 742 | -0.0709 | -1.85 | -0.02% | -0.39 | **DROP** |
| `macro_us_10y_5d_chg` | 742 | -0.0712 | -1.86 | -0.03% | -0.66 | **DROP** |
| `macro_brent_20d_pct` | 742 | -0.0745 | -1.95 | -0.47% | -11.43 | **DROP** |
| `macro_brent_inr_5d_pct` | 550 | -0.0792 | -1.76 | -0.41% | -7.74 | **DROP** |
| `macro_upside_skew_count` | 742 | -0.0807 | -2.11 | -0.13% | -3.29 | **DROP** |
| `macro_median_return_20d` | 742 | -0.083 | -2.17 | +0.03% | 0.76 | **DROP** |
| `sec_sector_5d_ret` | 742 | -0.0862 | -2.26 | -0.39% | -9.11 | **DROP** |
| `sec_sector_60d_ret` | 742 | -0.0867 | -2.27 | +0.40% | 9.97 | **DROP** |
| `macro_dxy` | 742 | -0.0878 | -2.3 | -0.31% | -7.38 | **DROP** |
| `macro_dispersion_z_60d` | 742 | -0.0881 | -2.31 | -0.11% | -3.09 | **DROP** |
| `macro_dxy_20d_pct` | 742 | -0.0919 | -2.41 | -0.29% | -7.4 | **DROP** |
| `sec_rs_5d` | 742 | -0.0956 | -2.51 | -0.03% | -0.62 | **DROP** |
| `macro_brent_inr` | 555 | -0.0989 | -2.21 | +0.33% | 6.86 | **DROP** |
| `sec_sector_dispersion_20d` | 742 | -0.1036 | -2.72 | -0.16% | -4.14 | **DROP** |
| `sec_sector_breadth_50` | 742 | -0.1216 | -3.2 | -0.09% | -2.21 | **DROP** |
| `macro_breadth_50` | 742 | -0.1406 | -3.71 | -0.11% | -2.7 | **DROP** |
| `macro_breadth_50_smid` | 742 | -0.1501 | -3.96 | -0.36% | -8.95 | **DROP** |
| `macro_breadth_200` | 742 | -0.1549 | -4.09 | +0.18% | 4.53 | **DROP** |
| `macro_sent__china_econ` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__crude_oil` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__earnings_outlook` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__fed_policy` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__fii_flow` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__geopolitics` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__global_liquidity` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__gold` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__india_credit` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__india_inflation` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__india_monsoon` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__indian_economy` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__rbi_policy` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__recession_risk` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent__rupee` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |
| `macro_sent_avg` | 742 | nan | nan | -0.61% | -17.31 | **INSUFFICIENT** |