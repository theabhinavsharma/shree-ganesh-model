#!/usr/bin/env bash
# Parallel P&L crawl driver (user-approved 2026-09-05: "run parallel fetchers").
# 4 detail workers + 3 XBRL workers — network I/O only, a few MB RAM each; the
# memory law (no parallel PANEL loads) does not apply here. Each worker has its
# own NSE session + own checkpoint shard; resume-safe.
set -u
cd /Users/abhinavs./Documents/Zoom
PY=/usr/bin/python3
for i in 0 1 2 3; do
  nice -n 10 $PY src/agentic/fetch_pnl_history.py details $i 4 > logs/pnl_det_w$i.log 2>&1 &
done
wait
echo "=== details done $(date) ==="
for i in 0 1 2; do
  nice -n 10 $PY src/agentic/fetch_pnl_history.py integrated $i 3 > logs/pnl_int_w$i.log 2>&1 &
done
wait
echo "=== integrated done $(date) ==="
nice -n 10 $PY src/agentic/fetch_pnl_history.py normalize > logs/pnl_norm.log 2>&1
echo "=== PNL PARALLEL CRAWL COMPLETE $(date) ==="
