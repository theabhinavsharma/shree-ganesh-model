# Devil's Advocate audit — 2026-04-30 12:11 IST

**Job: falsify, not validate.** This audit assumes every modelling claim is wrong until evidence forces otherwise.

## Severity summary

| Severity | Count |
|---|---:|
| 🔴 CRITICAL | 2 |
| 🟠 HIGH | 3 |
| 🟡 MEDIUM | 4 |
| 🟢 LOW | 0 |
| ✅ PASS | 2 |

## 🔴 CRITICAL findings (block deployment)

### LEAKAGE · Screener / academic / qvm features used as time-series
- **Evidence:** 8 of 49 sampled features are constant per-symbol across all dates: scr_pe, scr_market_cap_cr, scr_dividend_yield, scr_book_value, scr_roce…
- **Impact:** Today's-snapshot features applied to historical labels. Backtest on these features cannot generalize forward — IC and lift numbers are partially tautological.
- **Fix required:** Fetch quarterly historical Screener fundamentals; rebuild features with proper trade_date stamping.

### REGIME_FIT · Regime gate v1 lifts hit rate 41% → 64%
- **Evidence:** The gate parameters (market_20d ≤ -2%, breadth ∈ [50,75]) are documented in risk_envelope.py without specifying which years they were derived from. If derived from same 2024 OOS we tested on, this is curve-fitting.
- **Impact:** Gate may not generalize. If true prospective lift is 41% → 50% (not 64%), per-name hit rate stays ~16%, expected ann ≈ -0.5%.
- **Fix required:** Train regime gate on 2018-2022 only; test on 2023-2025 prospective. Report year-by-year hit rate. If 2018 (bear year) shows 41% basket hit rate same as unconditional, the gate is weather-vane, not edge.

## 🟠 HIGH findings (must address before next ship)

### CAL_DRIFT · 0.80 band delivers 83.5% real hit rate (5%/7d target)
- Evidence: Calibrator was fit on the concat of 2024+2025 OOF. When applied back to those same predictions, the hit-rate match is partially mechanical (isotonic minimises this gap by construction).
- Impact: True prospective hit rate is likely 5-10pp lower than claimed. User-corrected risk_envelope.py shows in-sample 90% maps to 12.4% prospective per-name on multibagger — analogous gap may exist on the 5%/7d band.
- Fix: Rerun: fit isotonic on 2024 OOF only, score 2025 OOF, report band hit rates separately. Headline must use the 2025-only number.

### MULTI_TEST · Frontier reports 70 (X%, Y days) combos. Achievable count: 50.
- Evidence: Tested 70 hypotheses with α=0.05 nominal. Bonferroni-corrected α = 0.0007. Many of the 'IC_PASSED' factors won't survive this.
- Impact: Some subset of 'achievable' combos are noise. Apply 1/N correction.
- Fix: Re-evaluate each achievable combo at α=Bonferroni. Promote only those still significant.

### CHERRY_PICK · Frontier publishes 'best combo per horizon' (e.g. 5%/7d, 7%/15d)
- Evidence: 70 combos tested; reporting only the achievable ones inflates impressions. The 'achievable frontier' table contains both the winners AND the non-significant ones, but downstream summaries (find_high_conviction.py, risk_envelope.py) cite winners only.
- Impact: Over-confident headlines. Real expectation across all combos is average, not best.
- Fix: Headlines must say 'X of N tests passed Bonferroni-corrected significance'. Report median achievable as the strategy benchmark, not max.

## 🟡 MEDIUM findings (worth investigating)

- **LEAKAGE**: All features are point-in-time computable — Manual review needed: dist_sma200, market_breadth, sector_5d_ret all look backward, but verify no per-symbol shifts use future data
- **SURVIVORSHIP**: Backtest universe includes delisted / merged / suspended names — 2016 had 1,654 symbols; 2025 has 2,552; 545 symbols (33%) present in 2016 but absent in 2025.
- **SAMPLE_SIZE**: Frontier marks 50 combos as 'achievable @ 90%' — 5 of these have < 100 OOS samples at the reported band. Examples: 3d×5% (n=96), 3d×7% (n=87), 10d×7% (n=43), 30d×10% (n=81), 30d×15% (n=57)
- **HYPER_LEAK**: LGB/XGB hyperparameters are documented but appear hand-tuned — n_estimators=400, learning_rate=0.05, num_leaves=64, etc are repeated across run_v3, run_short_side, run_multi_horizon. No evidence they were derived via held-out validation; likely chosen by inspecting OOS performance.

## ✅ PASS (already validated / corrected)

- CAL_DRIFT: Multibagger 90% claim was overfit — User's risk_envelope.py corrected HIT_RATE from 0.90 → 0.124 based on prospective 2024 backtest (41% basket-level instead of expected 99%+).
- SIG_INDEPEND: Stacking 3+ signals → 80%+ confidence — joint_signal_analyzer.py empirically tested: 3-signal stocks have 40.4% hit rate vs 42.8% baseline. NOT independent. Falsified.

## What this means for today

**2 CRITICAL issue(s) found.** Do not ship strategy claims to user until these are resolved.
