# Monthly 30TD Winner RCA

This folder explains why the system missed stocks that later finished up at least 15% over a trailing 30-trading-day window.

The audit is anchored on:
- as-of date: `2026-04-21`
- month-earlier anchor date: `2026-03-10`

The core finding is horizon mismatch:
- most eventual month-winners did not look like `+15% in the next 7 trading days` setups one month earlier
- many of the small number that did were rebound or distressed reversal names rather than healthy trend continuations

## Files

- `summary.json`
  High-level counts and the `PFOCUS` explanation.
- `monthly_30td_winners_by_first_week_category.csv`
  Every trailing-30TD winner, with its forward 7/15/30TD returns from the month-earlier anchor and a first-week behavior bucket.
- `first_week_structure_by_bucket.csv`
  Structural summary by bucket: share above `50 DMA` / `200 DMA`, median RSI, and median participation metrics.
- `would_fit_15pct_7d_target.csv`
  The small subset of month-winners that actually did `+15%` or more in the first 7 trading days from the anchor.
- `pfocus_monthly_rca.csv`
  Single-name detail for `PFOCUS`.

## How to read this folder

- Start with `summary.json`.
- Use `pfocus_monthly_rca.csv` to understand the concrete `PFOCUS` story.
- Use `monthly_30td_winners_by_first_week_category.csv` to inspect the whole month-winner population.
- Use `first_week_structure_by_bucket.csv` to see whether the catchable subset looked like healthy trends or reversals.
- Open each matching `.manifest.json` sidecar when you need row-grain, null-count, and column-level detail.
