# Stateful Weekly Portfolio Workflow

This workflow is the durable investing layer on top of the weekly 7-day winners model.

## Operating cadence

- Default cadence day: `MONDAY`
- Default cadence time: `20:30`
- Time zone: `Asia/Kolkata`

This cadence is an operating choice, not a claim that Monday has magical alpha. It is chosen because it is easy to run consistently after a completed trading day and it fits a once-a-week rotation discipline.

## Folder structure

- `data/portfolio_state/current_positions.csv`
  Current open positions after confirmed trade execution.
- `data/portfolio_state/executed_trade_ledger.csv`
  Append-only audit trail of confirmed actions.
- `data/portfolio_state/workflow_settings.json`
  Objective, cadence, and time-zone settings.

## Decision-sheet logic

Every weekly run now creates a stateful decision sheet:

- `Buy New`
  Stock is in the latest shortlist and is not currently held.
- `Buy More`
  Stock is already held and the new target allocation is materially above the current confirmed allocation.
- `Hold`
  Stock is still shortlisted and the new target allocation is close to the current confirmed allocation.
- `Sell Partly`
  Stock is still shortlisted, but the new target allocation is materially lower than the current confirmed allocation.
- `Sell Wholly`
  Stock has dropped out of the active shortlist or has breached the carried stop-loss level.

## Confirmation step

Recommendations do not become positions automatically.

After the user says the trades were placed, confirm the latest decision sheet:

```bash
python3 -m src.portfolio.state confirm \
  --decision-sheet-path reports/ml_weekly_winners/<run_id>/decision_sheet/weekly_position_decision_sheet_YYYYMMDD.csv \
  --state-dir data/portfolio_state \
  --execution-date YYYY-MM-DD \
  --confirmed-by user
```

This updates:

- the open position book
- the execution ledger
- the latest confirmation JSON

## Readability rules

- Every important CSV or JSON has a `.manifest.json` sidecar.
- The portfolio state folder has a `README.md`.
- Allocation fields are percentages of total portfolio capital, not returns.
- The execution ledger is append-only.
