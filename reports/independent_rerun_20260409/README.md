# Independent Weekly Winners Rerun 2026-04-09

This folder contains an independent rerun completed from freshly rebuilt week-7 OOF artifacts and a newly assembled current decision sheet.
The selected regime from the rerun was liquid_20cr_plus with no_gate.

## Files

- `summary.json`: Top-level summary of the live screening run, including final counts, missing-input diagnostics, and rule-by-rule survivor counts.
- `weekly_position_decision_sheet_20260407.csv`: Stateful weekly investing sheet that compares this week's shortlist against confirmed open positions and outputs Buy New, Buy More, Hold, Sell Partly, or Sell Wholly.
- `weekly_winners_shortlist.csv`: Final fail-closed shortlist for the 7-day 5 percent winners model after the active regime gate and production validations pass.

## How to read this folder

- Open `summary.json` first when it exists.
- For each data file, open the matching `.manifest.json` sidecar to see row grain, column meanings, null counts, and sample values.
- `individual_counts` means a rule tested alone across the universe.
- `sequential_counts` means rules applied in checklist order, so each step shows how many names survived up to that point.
- `cutoff_before_*` and `cutoff_after_*` show the names just before and just after the first rule where survivor count drops below 30.
