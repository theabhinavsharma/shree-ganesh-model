---
name: sgm-triage
description: Fix a failed SGM / Sri Lakshmi run. Use when an eval report says BLOCKED, a phone alert says a feed failed, or the user says something broke (stale data, missing picks, no daily message).
---

# SGM triage — when something breaks

Goal: find the real cause, fix it, prove it's fixed, and make sure it can't happen silently again.

1. Read today's `reports/eval_report_<date>.md`. List every FAIL and WARN with its plain-English statement.
2. For each FAIL, open the matching log in `logs/daily_data_layer/<ts>_<step>.log` (or `logs/weekly_pipeline/`). Read the last 40 lines, not the whole file.
3. Name the cause in one sentence (e.g. "NSE returned 403 for the bhavcopy", "cron never started — no status file for today").
4. Fix the cause, not the symptom. Never edit data by hand to make a check pass. Never fake a value.
5. Re-run only the failed step, then `/usr/bin/python3 src/agentic/run_evals.py --cadence daily`. It must show PASS for what failed.
6. If the failure was silent (no check caught it), add a new eval to `evals/registry.yaml` + `src/agentic/eval_checks.py` so it's caught next time.
7. Tell the user in plain English: what broke, why, what you changed, proof it passes now. Times in US Eastern.
