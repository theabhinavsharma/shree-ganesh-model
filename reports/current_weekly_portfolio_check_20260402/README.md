# Stateful Portfolio Check For 2026-04-02

This folder contains the mid-week stateful check against confirmed holdings as of 2026-04-02.
It is not a fresh weekly universe rotation.
Technical values use official NSE 2026-04-01 bhavcopy and current prices use official NSE quote snapshot fetched on 2026-04-02.

## Files

- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.
- `weekly_position_decision_sheet_20260402.csv`: Stateful weekly investing sheet that compares this week's shortlist against confirmed open positions and outputs Buy New, Buy More, Hold, Sell Partly, or Sell Wholly.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
