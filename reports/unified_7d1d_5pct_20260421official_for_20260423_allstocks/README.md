# Unified 7D And 1D Five Percent Calls For 23 April 2026

This folder contains unified all-stocks 7D +5% and next-session 1D +5% scorebooks.
Scoring uses the latest completed official daily data through 21 April 2026.
Company descriptors and live quote metadata were enriched from official NSE quote-equity snapshot for the final shortlisted names only.
Open summary.json first, then unified_top10_7d1d_5pct_calls_20260423.csv.

## Files

- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.
- `top10_1d_5pct_calls_20260423.csv`: Generated artifact.
- `top10_7d_5pct_calls_20260423.csv`: Generated artifact.
- `unified_top10_7d1d_5pct_calls_20260423.csv`: Generated artifact.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
