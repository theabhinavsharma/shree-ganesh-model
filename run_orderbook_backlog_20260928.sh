#!/usr/bin/env bash
# 2026-09-28 ~01:15 ET, replaces stream B of run_backfill_20260928.sh:
#   NSE's archive host caps each connection at ~130 KB/s (4 parallel curls each got the same speed), so order-book
#   throughput scales with workers: 40 workers, up to 5 documents per company-quarter, each quarter's best document
#   first (worklist cached in data/derived/order_book/worklist.parquet). Then a retry pass for HTTP_/ERR_, consolidate,
#   wait for the P&L stream (STREAM A DONE: pnl_quarterly normalized + mcap_pit rebuilt), and run the pre-registered
#   backlog ÷ PIT TTM revenue test (EXP-2026-09-28-order-backlog). Resumable: re-run after any stop.
set -u
cd /Users/abhinavs./Documents/Zoom
L=logs/backfill_20260928; P=/usr/bin/python3
log() { echo "[$(date +%T) ET] $*" | tee -a $L/main.log; }
log "orderbook v3 start: 40 workers"
for n in $(seq 0 39); do nice -n 10 $P src/agentic/fetch_order_book.py --shard $n --of 40 >> $L/ob40_$n.log 2>&1 & sleep 2; done
wait
log "orderbook v3: main pass done; retrying HTTP_/ERR_"
for n in 0 1 2 3 4 5 6 7; do nice -n 10 $P src/agentic/fetch_order_book.py --shard $n --of 8 >> $L/ob_retry_$n.log 2>&1 & done
wait
nice -n 10 $P src/agentic/fetch_order_book.py --consolidate >> $L/orderbook_consolidate.log 2>&1
log "orderbook v3 consolidated: $(tail -1 $L/orderbook_consolidate.log)"
until sed -n '/00:39:19\] start/,$p' $L/main.log | grep -q "STREAM A DONE"; do sleep 60; done
log "P&L stream done; running backlog test"
nice -n 10 $P src/agentic/test_order_backlog.py > logs/leader_sleeve/order_backlog_test.log 2>&1
log "BACKLOG TEST DONE (exit $?) -> logs/leader_sleeve/order_backlog_test.log"
