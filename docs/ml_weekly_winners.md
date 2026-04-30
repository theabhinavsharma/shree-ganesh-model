# Weekly Winners Model

This is the fail-closed production path for the short-horizon winners model.

## Objective

- Horizon: 7 calendar days
- Target: at least 5 percent forward return
- Weekly basket objective: at least 2 winners inside the live basket

## What makes it production-style

- It uses walk-forward out-of-sample predictions, not one in-sample fit.
- It searches regime gates only on historical weekly run results.
- It will not emit a live shortlist unless an active gate is above the configured historical thresholds.
- It writes a human-readable run folder with:
  - `weekly_winners_summary.json`
  - `weekly_winners_shortlist.csv` when the run passes
  - `weekly_winners_gate_candidates.csv`
  - `weekly_winners_universe_summary.csv`
  - `.manifest.json` sidecars
  - folder `README.md`

## Current default thresholds

- Basket size: 12 names
- Minimum winners objective: 2 names up 5 percent or more
- Minimum search-period success rate: 55 percent
- Minimum holdout-period success rate: 60 percent
- Minimum all-period success rate: 55 percent

## Run command

```bash
python3 -m src.report.production_weekly_winners
```

## Important limitation

- This is still a research-production bridge, not a brokerage execution engine.
- The output is suitable for disciplined weekly review and research tracking.
- It does not yet model live slippage, order-book impact, or broker execution state.
