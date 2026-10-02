# Weekly 7-day-winners logic, held 6 months — EXP-2026-10-02-weekly7d-6m-hold

_generated 2026-10-02 11:07 · picks: data/ml/expert_runs/wf2020_week7_oof.parquet (walk-forward, train ≤ year-1, test 2020-2025) · prices: data/derived/stock_daily_facts_adjusted_2015plus.parquet through 2026-10-01 · top 12/week, next open → close of day 126, 0.30% RT_

## Universe `mid_small` (the one the 8-Apr live run used)

| entry year | weeks | avg 6M % | median 6M % | worst week % | best week % | % picks up | universe median 6M % | % weeks beating universe |
|---|---|---|---|---|---|---|---|---|
| 2020 | 52 | +43.7 | +45.2 | -28.0 | +128.3 | 79 | +33.7 | 75 |
| 2021 | 52 | +31.5 | +21.8 | -29.4 | +114.7 | 64 | +6.5 | 92 |
| 2022 | 52 | +14.0 | +15.2 | -27.7 | +57.2 | 56 | +1.1 | 85 |
| 2023 | 52 | +40.3 | +41.1 | +1.8 | +112.3 | 77 | +24.9 | 94 |
| 2024 | 53 | +12.0 | +3.3 | -29.9 | +75.0 | 44 | -2.0 | 75 |
| 2025 | 52 | +2.7 | +1.7 | -25.9 | +53.0 | 39 | -3.0 | 67 |
| all | 313 | +24.0 | +20.6 | -29.9 | +128.3 | 60 | +6.2 | 81 |

## Universe `liquid_5cr_plus` (secondary)

| entry year | weeks | avg 6M % | median 6M % | worst week % | best week % | % picks up | universe median 6M % | % weeks beating universe |
|---|---|---|---|---|---|---|---|---|
| 2020 | 52 | +33.2 | +32.6 | -29.2 | +73.5 | 69 | +29.4 | 71 |
| 2021 | 52 | +15.8 | +17.3 | -25.9 | +69.9 | 54 | +3.0 | 69 |
| 2022 | 52 | +1.1 | +2.6 | -43.0 | +42.3 | 45 | -0.5 | 56 |
| 2023 | 52 | +33.4 | +29.0 | -11.2 | +81.7 | 75 | +20.3 | 81 |
| 2024 | 53 | -2.0 | -5.3 | -33.3 | +45.1 | 38 | -3.7 | 57 |
| 2025 | 52 | -5.4 | -8.1 | -32.4 | +25.7 | 33 | -4.4 | 31 |
| all | 313 | +12.6 | +12.0 | -43.0 | +81.7 | 52 | +6.0 | 61 |

Fidelity check: 2023-25 weekly top-12 overlap with the saved 8-Apr run (same folds, pre-repair panel): mean 4.0/12 over 149 weeks.

## Caveats

- The live run only produced a shortlist when its regime gate was active; this invests every week.
- Ranking drops the isotonic calibration (monotone) and the 0.10 weight on the 1-day/15-day models, which the walk-forward file doesn't carry.
- 12-name weekly cohorts overlap heavily week to week; weeks are not independent samples.
- Universe median is computed on a random 150 names per week when the universe is large.
