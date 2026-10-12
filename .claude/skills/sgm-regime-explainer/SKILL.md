---
name: sgm-regime-explainer
description: Weekly (Friday 3 PM IST, before the 7:30 PM list) non-technical "why" note for the Sri Lakshmi model — which quarters' V3.1 lists did well or badly, the macro picture and events (elections, budgets, bills, RBI/Fed turns, SEBI rules, wars, tariffs) while they were held, whether event types repeat, today's closest past quarters, and a short phone note. Use on Fridays, or when the user asks "what's going on in the market / does this look like a past regime / why did that period fail".
---

# SGM regime explainer (weekly)

Exploration only. It explains and suggests; it never changes a rule. Anything that looks like a pattern becomes a proposed
registered test (sgm-new-idea) and needs the user's yes.

1. Run `/usr/bin/python3 src/agentic/regime_explainer.py`. It writes `reports/regime/regime_<date>.md` and
   `logs/regime/regime_<date>.json` (numbers: V3.1 lists by quarter vs the market, macro readings known at the time, events
   while each quarter's lists were held, the repeat check by event type, today's three closest past quarters, upcoming events,
   tests logged in the last 7 days). If it fails, follow sgm-triage and send a one-line ❌ note.
2. Verify before citing. Any event the note will mention that is `verified: false` in `configs/event_calendar.yaml`: open its
   official source (ECI, indiabudget.gov.in, PIB, RBI, SEBI, Fed / FRED, WHO, USTR...) with WebFetch / WebSearch or the built-in
   browser, confirm the date, then set `verified: true` and put the exact page in `source`. Never cite an unverified date as fact.
3. New events. Scan the last 7-10 days of `data/derived/pib_releases.parquet` (government releases) and
   `data/derived/news_feed.parquet` (headlines) for big non-technical events: elections, bills / acts passed, tax changes,
   RBI / SEBI / exchange decisions, wars, tariffs, oil or currency shocks. Add the important ones to the calendar with their
   primary source (`verified: true` only after checking it).
4. Fill the report's "Context (not evidence)" section: 3-6 plain-English lines on what was going on in today's closest past
   quarters and in the most recent great / bad quarters, each with a date and a source. No AI-isms, no forced symmetry.
5. Hypotheses: at most 3, each written as something testable ("lists bought within 90 days after X do worse"), with how many
   past episodes it rests on (from the repeat table) and whether it meets the pattern bar (>= 5 episodes, >= 80% same side).
   Below the bar it is a story; say so. At or above the bar, propose a registered test to the user.
6. Tests logged in the last 7 days: say which calendar years each arm won or lost and any listed events in those years.
7. Phone note (sgm-telegram rules: <= 8 lines, first line `🧭 Regime note · <Mon DD>`, numbers only from the report / json,
   plain words): `telegram_lines` from the json plus 1-2 context lines. Send with
   `notify.send(text, "regime")` (src/agentic/notify.py; `tidy()` the lines first).
8. Commit the report, the json and any calendar changes in one commit (no link in the reply).

Honesty rules: about 15 independent half-years since 2019 and 2-3 market tops, so most "reasons" are stories; the repeat
table counts episodes (events within 120 days are one episode; policy rates count only cycle turns). Market-level macro and
FPI-flow rules have failed registered tests (EXP-2026-10-11-v31-breadth-pause, EXP-2026-10-12-v31-fpi-sizing,
EXP-2026-10-12-v31-fpi-sector-tilt); a new idea must be different from those, not a re-run.
