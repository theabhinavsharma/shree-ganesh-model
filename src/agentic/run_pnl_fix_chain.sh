#!/usr/bin/env bash
# Stage-C v2 refetch (context-date XBRL parse, financial files only) -> normalize
# -> QC gate + 3-horizon valuation A/B. Serial after the fetch shards; memory-safe.
set -u
cd /Users/abhinavs./Documents/Zoom
PY=/usr/bin/python3
for i in 0 1 2; do
  nice -n 10 $PY src/agentic/fetch_pnl_history.py integrated $i 3 > logs/pnl_int2_w$i.log 2>&1 &
done
wait
echo "=== integrated2 done $(date) ==="
nice -n 10 $PY src/agentic/fetch_pnl_history.py normalize > logs/pnl_norm.log 2>&1
echo "=== normalize done $(date) ==="
nice -n 10 $PY src/agentic/ab_valuation_3h.py > logs/ab_valuation_3h.log 2>&1
echo "=== AB EXIT CODE: $? $(date) ==="
