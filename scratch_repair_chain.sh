#!/usr/bin/env bash
# 2026-09-27 panel repair chain (audit root causes 1 + 5). Backup -> BE/BZ backfill 2015+ -> price-only CA factors -> coverage eval.
set -u
cd /Users/abhinavs./Code/Zoom
LOG=logs/panel_repair_20260927.log
P=data/derived/stock_daily_facts_adjusted_2015plus.parquet
echo "[$(date +%T)] backup" >> $LOG
cp -p $P $P.bak-2026-09-27 || { echo "backup failed" >> $LOG; exit 1; }
( while true; do kb=$(ps -axo rss,comm | awk '/[Pp]ython/{s+=$1} END{print s+0}'); if [ "$kb" -gt 57671680 ]; then echo "[$(date +%T)] WATCHDOG: python RSS ${kb}KB > 55GB — killing" >> $LOG; pkill -f repair_be_series_gaps; pkill -f build_price_only_ca_factors; fi; sleep 20; done ) &
WD=$!
echo "[$(date +%T)] step 1: BE/BZ backfill from 2015-01-01" >> $LOG
nice -n 10 /usr/bin/python3 src/agentic/repair_be_series_gaps.py 2015-01-01 >> $LOG 2>&1 || { echo "STEP1 FAILED" >> $LOG; kill $WD; exit 1; }
echo "[$(date +%T)] step 2: price-only CA factors (rebuild on repaired panel) + apply" >> $LOG
nice -n 10 /usr/bin/python3 src/agentic/build_price_only_ca_factors.py --apply >> $LOG 2>&1 || { echo "STEP2 FAILED" >> $LOG; kill $WD; exit 1; }
echo "[$(date +%T)] step 3: full-history coverage eval" >> $LOG
nice -n 10 /usr/bin/python3 src/agentic/eval_panel_coverage.py --sessions 3000 >> $LOG 2>&1 || echo "coverage eval reported FAIL (see above)" >> $LOG
kill $WD
echo "[$(date +%T)] REPAIR CHAIN DONE" >> $LOG
