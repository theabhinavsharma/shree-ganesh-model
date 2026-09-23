#!/usr/bin/env bash
# One-off (2026-09-24): re-insert the BE/BZ rows dropped 2026-09-08..23, then re-run the
# coverage eval + gate. Uses the idempotent 08-28 repair script on raw partitions on disk.
cd /Users/abhinavs./Documents/Zoom
mkdir -p logs/sgm_daily
nohup bash -c '
  cp data/derived/stock_daily_facts_adjusted_2015plus.parquet data/derived/stock_daily_facts_adjusted_2015plus.parquet.bak-2026-09-24
  /usr/bin/python3 src/agentic/repair_be_series_gaps.py
  /usr/bin/python3 src/agentic/eval_panel_coverage.py --sessions 12
  /usr/bin/python3 src/agentic/verify_freshness.py
  /usr/bin/python3 src/agentic/render_basket_report.py
  echo DONE' > logs/repair_coverage_20260924.log 2>&1 &
echo "launched pid $! — tail -f logs/repair_coverage_20260924.log"
