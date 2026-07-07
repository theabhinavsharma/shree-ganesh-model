# Data Freshness Status — 2026-07-07
_generated 2026-07-07T14:57:52_

## ✅ ALL FRESH — Pipeline may proceed

- All **10** inputs pass file-level contracts
- All **14** column-level checks pass

## Summary

| Input | File | Cols passing | Worst column | Fix |
|---|---|---|---|---|
| **PRICES** | ✅ 1/3bd | 4/4 | ✅ `close` 1/3bd |  |
| **CS_ENGINE** | ✅ 0/3bd | — | — |  |
| **HC_ENGINE** | ✅ 0/3bd | — | — |  |
| **MB_ENGINE** | ✅ 0/3bd | — | — |  |
| **F180_ENGINE** | ✅ 0/3bd | — | — |  |
| **ML_CLASSIFIER** | ✅ 0/3bd | — | — |  |
| **MACRO_PANEL** | ✅ 1/3bd | 7/7 | ✅ `usdinr` 1/3bd |  |
| **INDUSTRY** | ✅ 1/3bd | 3/3 | ✅ `sector_5d_ret` 1/3bd |  |
| **NEWS_EVENTS** | ✅ 0/5bd | — | — |  |
| **ANNOUNCEMENTS** | ✅ 0/10bd | — | — |  |

## Full detail — every check

### ✅ PRICES

- **Path**: `data/derived/stock_daily_facts_adjusted_2015plus.parquet`
- **File-level**: max=2026-07-06  ·  stale=1bd  ·  limit=3bd  ·  ✅ OK
- **Column-level** (4 checks):
  - ✅ `close` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `rsi_14_daily` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `return_20d` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `volume_vs_20d` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd

### ✅ CS_ENGINE

- **Path**: `data/derived/compare_short_horizons.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=3bd  ·  ✅ OK

### ✅ HC_ENGINE

- **Path**: `data/derived/high_conviction_predictions.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=3bd  ·  ✅ OK

### ✅ MB_ENGINE

- **Path**: `data/derived/multibagger_today_predictions.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=3bd  ·  ✅ OK

### ✅ F180_ENGINE

- **Path**: `data/derived/180d_today_predictions.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=3bd  ·  ✅ OK

### ✅ ML_CLASSIFIER

- **Path**: `data/derived/missed_winner_classifier.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=3bd  ·  ✅ OK

### ✅ MACRO_PANEL

- **Path**: `data/derived/macro_panel.parquet`
- **File-level**: max=2026-07-06  ·  stale=1bd  ·  limit=3bd  ·  ✅ OK
- **Column-level** (7 checks):
  - ✅ `usdinr` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `brent` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd
  - ✅ `wti` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd
  - ✅ `us_10y` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd
  - ✅ `dxy` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd
  - ✅ `us_vix` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd
  - ✅ `spx` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=5bd

### ✅ INDUSTRY

- **Path**: `data/derived/industry_panel.parquet`
- **File-level**: max=2026-07-06  ·  stale=1bd  ·  limit=3bd  ·  ✅ OK
- **Column-level** (3 checks):
  - ✅ `sector_5d_ret` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `sector_20d_ret` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd
  - ✅ `rs_20d` — last non-null=2026-07-06  ·  stale=1bd  ·  limit=3bd

### ✅ NEWS_EVENTS

- **Path**: `data/derived/news_event_features.parquet`
- **File-level**: max=2026-08-06  ·  stale=0bd  ·  limit=5bd  ·  ✅ OK

### ✅ ANNOUNCEMENTS

- **Path**: `data/events_full_history/normalized/stock_announcements.parquet`
- **File-level**: max=2026-07-07  ·  stale=0bd  ·  limit=10bd  ·  ✅ OK

---
_This report is regenerated on every pipeline run. If you see stale entries, run the fix command listed._