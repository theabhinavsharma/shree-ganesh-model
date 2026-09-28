# 1.5x anatomy (EXP-2026-09-27-1p5x-anatomy)

What precedes a +50% move within 95 sessions, for every stock-week with market cap >= Rs50cr since 2016.

- `rows.parquet` — one row per stock-week: features known at the entry close, targets, out-of-sample model score (see manifest; `model_features` lists what the model used).
- `rows_audit.parquet` — per-row provenance: mcap_source (PIT or not), industry label source, then-traded close, fundamentals basis (con/sa) behind each fundamentals value, observed sessions in the label window (forward-looking, audit only).
- `feature_coverage_by_year.csv` — % of labelled rows where each feature is known, by year; count features also get the non-zero rate.
- `univariate_lift.csv` / `univariate_lift_pnl_implied.csv` — P(+50% within 95 sessions) by feature bucket, per era, lift vs the base of rows where the feature is known; second file = PIT-mcap rows only.
- `results.json` — base rates, walk-forward by year, era metrics (all rows and PIT-mcap rows), permutation importance, gain share, all-in portfolio per entry mode with per-era CAGR.

Eras: disc = 2016-2022, conf = 2023+. A finding is only trusted if it holds in both.

Sessions are the panel's dates minus the NSE-holiday dates on which the panel copies every symbol's previous row (75 dates 2020-2026 on the 2026-09-27 backup; list in results.json), so 95 sessions means 95 real sessions in every era.

Limitations: mcap is PIT only for pnl_implied rows; industry is a 2026 label applied to all years; labels on windows with missing sessions are lower bounds; panel-computed rolling columns (SMAs, ADV, volume ratios, RSI, 20d delivery) still include the copied holiday rows; see the module docstring for the 2026-09-27 audit fixes.

## Model bake-off (src/agentic/model_bakeoff_1p5x.py)

- `bakeoff.json` — within-week AUC / top-1% and weekly top-10 tradable hit rate per model, per era and per year, close-entry and next-open entry; fold details and feature coverage by year.
- `bakeoff_preds.parquet` — every model's out-of-sample score for each test row (see manifest).
