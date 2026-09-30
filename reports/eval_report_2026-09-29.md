# Eval report — 2026-09-29 (data through 2026-09-28)

**Verdict: BLOCKED.** 22 evals · 17 pass · 1 blocking failures · 0 warnings · 4 waiting on a person.

| | Eval | What must be true | Result | Action if it fails |
|---|---|---|---|---|
| ✅ | `data.coverage` | Every stock in NSE's own daily file for the latest session is in our price panel, with the same closing price to the paisa. | **PASS** 2944/2944 (100.00%) | block (system) |
| ✅ | `data.freshness` | Every input the pipeline reads is younger than its declared maximum age, file by file and column by column. | **PASS** 14 inputs fresh | block (system) |
| ✅ | `data.no_duplicates` | Each stock has at most one row per trading day in the price panel. | **PASS** 0 — 5,418,543 rows checked | block (system) |
| ✅ | `data.no_holiday_copies` | None of the last 20 sessions is a copy of the session before it. | **PASS** max repeat share 0.2% | block (system) |
| ✅ | `data.ca_before_prices` | Corporate actions were refreshed before prices in the latest daily run. | **PASS** 2026-09-28 — corp_actions ran before prices | block (system) |
| ✅ | `data.manifests` | Every derived table the strategy reads has a manifest that states its units and its source. | **PASS** 15/15 | warn (system) |
| ✅ | `pit.filings_after_period` | No quarterly result is dated as filed before its quarter ended, and no order filing is dated after we fetched it. | **PASS** 0 — P&L filed before quarter end: 0; orders dated after fetch: 0 | block (system) |
| ✅ | `pit.model_scores_current` | The model scores used by this week's screen are at most 7 days old. | **PASS** 7 days — latest score 2026-09-21 vs last session 2026-09-28 | warn (system) |
| ✅ | `prov.external_rows_sourced` | Every row in the datasets built from government sources carries the URL it came from and the date it was published. | **PASS** 0 of 25,515 rows unsourced | block (system) |
| ✅ | `prov.order_amount_grounded` | Every order amount we report can be found, as written, in the filing text it was extracted from. | **PASS** 40/40 grounded (100%) | warn (system) |
| ✅ | `model.ranking_lift` | On the most recent fully labelled year, the model's weekly top-10 liquid picks touch +50% at least 1.5 times as often as the average liquid stock. | **PASS** 2.06x in 2025 — top-10 hit 15.5% vs base 7.5% | warn (system) |
| ✅ | `strategy.preregistered` | Every experiment with a result in the log was registered earlier, with its pass rule, before the result was written. | **PASS** 9/9 results pre-registered | block (system) |
| ✅ | `output.screens_immutable` | A weekly screen, once saved, is never rewritten. | **PASS** 4 screens tracked (1 new) | block (system) |
| ✅ | `output.one_screen_per_week` | Each track has at most one screen per calendar week. | **PASS** 0 | block (system) |
| ✅ | `output.paper_scored` | Every open paper batch was scored through the latest session. | **PASS** 0 | warn (system) |
| ❌ | `ops.daily_run_happened` | The daily data run finished for the latest weekday. | **FAIL** 2026-09-28 — expected a run for 2026-09-29 (NSE holidays not yet excluded) | block (system) |
| ✅ | `backup.drive_checksums` | The latest Google Drive backup matches its manifest checksums, file by file. | **PASS** 14/14 volumes match (2026-09-28) | block (system) |
| ⏳ | `trust.claims_sourced` | When Claude states a number to Abhinav, that number appears in a file, a command output or his own message from the same session. | **PENDING**  — no replies logged by the Stop hook in the last 7 days (hook loads on a new session) | warn (system) |
| ✅ | `trust.results_reproduce` | Re-running each registered test from the committed code gives the logged numbers again. | **PASS** 0.05 — 2 tests, worst gap 0.05 pts | block (system) |
| ⏳ | `trust.claude_golden` | A fresh Claude session answers the golden questions about this system correctly, including saying 'that does not exist' to trap questions. | **PENDING**  — no golden run yet (needs the claude CLI logged in) | warn (system) |
| ⏳ | `human.order_book_golden` | On a sample of 50 investor-deck filings graded by a person, the extracted figure is the filing's total order book at least 80% of the time. | **PENDING** 0/50 graded — grade evals/golden/order_book_to_grade.csv | warn (abhinav) |
| ⏳ | `human.weekly_pick_review` | Before real money moves, a person has read each weekly batch and signed off that no pick is a takeover target, carries a fraud or governance red flag, or has an unexplained price spike. | **PENDING** 0/3 signed — sleeve_20260923, model_20260928, sri_lakshmi_20260928 | block (abhinav) |

Registry: `evals/registry.yaml` (statement, why, source of truth, threshold, action, owner). Checks: `src/agentic/eval_checks.py`.
