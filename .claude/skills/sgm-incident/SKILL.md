---
name: sgm-incident
description: Record a failure so the system never repeats it. Use whenever something turned out wrong: a feed silently empty or stale, wrong numbers in the data, a bug found by looking at results, a check that missed something, or the user says "fuck-up", "galti", "why didn't the checks catch this".
---

# SGM incident: every failure becomes a check and a lesson

1. Fix the cause first (use sgm-triage). Prove it with numbers.
2. Decide the guard: which eval would have caught this BEFORE a person did? Write it (statement, why, source of truth,
   threshold, action, cadence, owner, check) as a YAML block file and add its check function to src/agentic/eval_checks.py.
   It must FAIL on the old broken data and PASS on the fixed data; show both.
3. Record everything in one command (append-only, backed up, rolled back if anything existing would be lost):
   `/usr/bin/python3 src/agentic/trust/incident.py add --id INC-<date>-<slug> --title ... --found-by ... --root-cause ...
    --blast-radius ... --fix ... --guard-eval <eval id> --eval-file <block.yaml> --skills <skill,skill> --lesson "<one line>" --commit <sha>`
4. The lesson is one plain sentence a future session must act on ("key corporate data by ISIN issuer, not ticker").
   Put it in every skill whose work could repeat the mistake.
5. Never edit or delete old lessons, incidents or evals. To change an eval, add a `history:` note under it saying what
   changed and why. `incident.py check` (daily eval trust.lessons_intact) blocks the run if anything recorded is missing.
6. If results were computed on the broken data, re-run the registered tests (src/agentic/trust/reproduce.py) and tell
   the user which verdicts changed.

## Lessons from incidents
(appended by src/agentic/trust/incident.py; never edit or delete these lines)
- 2026-09-30 [INC-2026-09-30-lessons-not-in-skills] Every failure ends with src/agentic/trust/incident.py add: a guard eval plus a one-line lesson in each affected skill; lessons only in commit messages are forgotten.
