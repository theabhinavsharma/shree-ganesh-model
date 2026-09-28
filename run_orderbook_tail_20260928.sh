#!/usr/bin/env bash
# 2026-09-28 07:20 ET: tail of run_orderbook_backlog_20260928.sh. 45 of its 80 workers had finished while 35 held long
# per-company queues (~3,000 documents left), so the rest is re-split across 80 workers BY DOCUMENT. Then the retry pass,
# consolidate, and the pre-registered backlog test (the P&L stream finished at 03:19 ET). Resumable.
set -u
cd /Users/abhinavs./Documents/Zoom
L=logs/backfill_20260928; P=/usr/bin/python3
log() { echo "[$(date +%T) ET] $*" | tee -a $L/main.log; }
log "orderbook tail: 80 workers by document"
for n in $(seq 0 79); do nice -n 10 $P src/agentic/fetch_order_book.py --shard $n --of 80 --by-doc >> $L/ob80_$n.log 2>&1 & sleep 0.5; done
wait
log "orderbook tail done; retrying HTTP_/ERR_"
for n in $(seq 0 15); do nice -n 10 $P src/agentic/fetch_order_book.py --shard $n --of 16 --by-doc >> $L/ob_retry_$n.log 2>&1 & done
wait
nice -n 10 $P src/agentic/fetch_order_book.py --consolidate >> $L/orderbook_consolidate.log 2>&1
log "orderbook consolidated: $(tail -1 $L/orderbook_consolidate.log)"
nice -n 10 $P src/agentic/test_order_backlog.py > logs/leader_sleeve/order_backlog_test.log 2>&1
log "BACKLOG TEST DONE (exit $?) -> logs/leader_sleeve/order_backlog_test.log"
