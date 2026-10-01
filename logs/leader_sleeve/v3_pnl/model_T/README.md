# 1.5x anatomy (EXP-2026-09-27-1p5x-anatomy)

## Model bake-off (src/agentic/model_bakeoff_1p5x.py)

- `bakeoff.json` — within-week AUC / top-1% and weekly top-10 tradable hit rate per model, per era and per year, close-entry and next-open entry; fold details and feature coverage by year.
- `bakeoff_preds.parquet` — every model's out-of-sample score for each test row (see manifest).
