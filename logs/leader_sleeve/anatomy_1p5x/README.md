# 1.5x anatomy (EXP-2026-09-27-1p5x-anatomy)

What precedes a +50% move within 95 sessions, for every stock-week with PIT market cap >= Rs50cr since 2016.

- `rows.parquet` — one row per stock-week: ~60 features known at the entry close, targets, out-of-sample model score (see manifest).
- `univariate_lift.csv` — P(+50% within 95 sessions) by feature bucket, per era, with lift vs base.
- `results.json` — base rates, permutation importance, gain share, all-in portfolio summary.
- Full console output: `logs/leader_sleeve/anatomy_1p5x_20260927.log`.

Eras: disc = 2016-2022, conf = 2023+. A finding is only trusted if it holds in both.
