#!/usr/bin/env bash
# Install the SGM cron schedule (idempotent — replaces only lines tagged # SGM).
#   Mon-Fri 18:45 IST  sgm_daily.sh           (NSE bhavcopy is published ~18:30)
#   Fri     19:30 IST  run_weekly_pipeline.sh  (after the daily layer has landed)
#   Sat     09:00 IST  run_weekly_pipeline.sh --skip-fetch   (retry if Friday failed the gate)
# Run: bash src/agentic/sgm_schedule.sh        Inspect: crontab -l
# Sleep: cron does not wake a sleeping Mac. Plugged in + lid open, or once:
#   sudo pmset repeat wakeorpoweron MTWRF 18:40:00   (and see `pmset -g sched`)
set -eu
ROOT=/Users/abhinavs./Documents/Zoom
TMP=$(mktemp)
( crontab -l 2>/dev/null | grep -v '# SGM' || true ) > "$TMP"
cat >> "$TMP" <<CRON
45 18 * * 1-5 /bin/bash $ROOT/src/agentic/sgm_daily.sh >> $ROOT/logs/sgm_daily/cron.log 2>&1   # SGM daily
30 19 * * 5   /bin/bash $ROOT/src/agentic/run_weekly_pipeline.sh >> $ROOT/logs/weekly_pipeline/cron.log 2>&1   # SGM weekly
0  9  * * 6   /bin/bash $ROOT/src/agentic/run_weekly_pipeline.sh --skip-fetch >> $ROOT/logs/weekly_pipeline/cron.log 2>&1   # SGM weekly-retry
CRON
crontab "$TMP"; rm -f "$TMP"
echo "installed:"; crontab -l | grep '# SGM'
