#!/usr/bin/env bash
# One-off (2026-09-24): walk-forward replay of the 5 engines' top-30 sets, serial
# (memory law: never parallel), 55 GB python-RSS watchdog. EXP-2026-09-24-engines-count-sizing.
cd /Users/abhinavs./Documents/Zoom
LOG=logs/engine_replay_20260924.log
( while true; do r=$(ps -axo rss,comm | awk '/[Pp]ython/{s+=$1} END{print int(s/1048576)}')
    [ "$r" -ge 55 ] && { echo "WATCHDOG ${r}GB — killing engine_replay" >> $LOG; pkill -f engine_replay.py; }
    sleep 15; done ) & WD=$!
for e in mb 180d cs mh hc; do
  echo "═══ $e $(date +%T)" >> $LOG
  nice -n 10 /usr/bin/python3 src/agentic/engine_replay.py --engine $e >> $LOG 2>&1 && echo "✅ $e $(date +%T)" >> $LOG || echo "❌ $e $(date +%T)" >> $LOG
done
kill $WD; echo DONE >> $LOG
