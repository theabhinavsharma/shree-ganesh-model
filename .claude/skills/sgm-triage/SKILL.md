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
7. Record it: follow `.claude/skills/sgm-incident/SKILL.md` (`src/agentic/trust/incident.py add ...`). Guard eval + one-line lesson in every affected skill. Append-only; `incident.py check` must say OK.
8. Tell the user in plain English: what broke, why, what you changed, proof it passes now. Times in US Eastern.

## Lessons from incidents
(appended by src/agentic/trust/incident.py; never edit or delete these lines)
- 2026-09-30 [INC-2026-09-24-cron-tcc] If the daily message did not arrive, check logs/runs/launchd_daily.log and logs/runs/<date>_daily.json first; never keep the project under ~/Documents, ~/Desktop or ~/Downloads (macOS blocks background jobs there).
- 2026-09-30 [INC-2026-09-30-insider-feed-silent-zero] A feed that returns 0 rows is broken until the source proves 0 is real (compare the same window a year earlier), and every feed gets a freshness eval.
- 2026-09-30 [INC-2026-09-30-renamed-symbol-cliffs] Key corporate data by company identity (ISIN issuer + NSE symbolchange.csv), never by ticker alone, and check the whole history for one-day moves of 45%+ without an NSE record, not only the new rows.
- 2026-09-30 [INC-2026-09-30-lessons-not-in-skills] Every failure ends with src/agentic/trust/incident.py add: a guard eval plus a one-line lesson in each affected skill; lessons only in commit messages are forgotten.
- 2026-09-30 [INC-2026-09-30-feeds-without-guards] Add every new feed to configs/feed_guards.json in the same change (file, date column, allowed weekdays); data.every_feed_guarded blocks the daily run for a feed without one. Rules that matter go into an eval, not only into a skill.
