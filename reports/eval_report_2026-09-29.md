# Eval report — 2026-09-29 (data through 2026-09-28)

**Verdict: BLOCKED.** 13 evals · 12 pass · 1 blocking failures · 0 warnings · 0 waiting on a person.

| | Eval | What must be true | Result | Action if it fails |
|---|---|---|---|---|
| ✅ | `data.coverage` | Every stock in NSE's own daily file for the latest session is in our price panel, with the same closing price to the paisa. | **PASS** 2944/2944 (100.00%) | block (system) |
| ✅ | `data.freshness` | Every input the pipeline reads is younger than its declared maximum age, file by file and column by column. | **PASS** 14 inputs fresh | block (system) |
| ✅ | `data.no_duplicates` | Each stock has at most one row per trading day in the price panel. | **PASS** 0 — 5,418,543 rows checked | block (system) |
| ✅ | `data.no_holiday_copies` | None of the last 20 sessions is a copy of the session before it. | **PASS** max repeat share 0.2% | block (system) |
| ✅ | `data.ca_before_prices` | Corporate actions were refreshed before prices in the latest daily run. | **PASS** 2026-09-28 — corp_actions ran before prices | block (system) |
| ✅ | `data.manifests` | Every derived table the strategy reads has a manifest that states its units and its source. | **PASS** 15/15 | warn (system) |
| ✅ | `pit.filings_after_period` | No quarterly result is dated as filed before its quarter ended, and no order filing is dated after we fetched it. | **PASS** 0 — P&L filed before quarter end: 0; orders dated after fetch: 0 | block (system) |
| ✅ | `prov.external_rows_sourced` | Every row in the datasets built from government sources carries the URL it came from and the date it was published. | **PASS** 0 of 25,515 rows unsourced | block (system) |
| ✅ | `prov.order_amount_grounded` | Every order amount we report can be found, as written, in the filing text it was extracted from. | **PASS** 40/40 grounded (100%) | warn (system) |
| ✅ | `output.screens_immutable` | A weekly screen, once saved, is never rewritten. | **PASS** 3 screens tracked (0 new) | block (system) |
| ✅ | `output.one_screen_per_week` | Each track has at most one screen per calendar week. | **PASS** 0 | block (system) |
| ✅ | `output.paper_scored` | Every open paper batch was scored through the latest session. | **PASS** 0 | warn (system) |
| ❌ | `ops.daily_run_happened` | The daily data run finished for the latest weekday. | **FAIL** 2026-09-28 — expected a run for 2026-09-29 (NSE holidays not yet excluded) | block (system) |

Registry: `evals/registry.yaml` (statement, why, source of truth, threshold, action, owner). Checks: `src/agentic/eval_checks.py`.
