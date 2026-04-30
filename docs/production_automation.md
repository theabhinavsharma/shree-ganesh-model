# Production Weekly Automation

This repo now has a fail-closed weekly portfolio runner:

- Cache mode:
  - `python3 -m src.report.production_weekly_run --mode cache`
- Final mode:
  - `python3 -m src.report.production_weekly_run --mode final --recipient <email> --recipient <email>`

Behavior:

- Cache mode refreshes and rebuilds:
  - corporate actions
  - market raw files and adjusted daily facts
  - announcements
  - macro history
  - fundamentals cache
  - shareholding cache

- Final mode:
  - reruns a light refresh for market, corporate actions, announcements, and macro
  - blocks unless the recent cache run succeeded
  - blocks unless freshness and coverage gates pass
  - reruns the exact current year-end study using the remaining days to December 31
  - blocks unless the generated portfolio passes sanity checks
  - sends the portfolio email only when every gate passes

Blocked run behavior:

- No portfolio recommendation is sent
- A blocked status HTML and CSV are written to the run directory
- If SMTP is healthy, a blocked-status email is sent instead of a portfolio

Required SMTP environment variables for production final mode:

- `REPORT_SMTP_HOST`
- `REPORT_SMTP_PORT`
- `REPORT_SMTP_USER`
- `REPORT_SMTP_PASSWORD`
- `REPORT_SMTP_FROM`
- optional: `REPORT_SMTP_SSL=true`

Important:

- Production mode does not trust local `sendmail` by default
- The final mail run is fail-closed on missing SMTP credentials
