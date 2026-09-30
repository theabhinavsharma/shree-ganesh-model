#!/usr/bin/env bash
# 2026-09-28 overnight backfill (v2 00:45: 6 P&L workers, 8 order-book workers, one document per company-quarter) (runs entirely on this Mac; resumable — re-run the same command after any stop).
#   A. P&L for ALL equities from NSE (fetch_pnl_history: calendar -> details x2 shards -> integrated -> normalize) -> rebuild mcap_pit
#   B. Order BOOK totals from decks / transcripts / results press releases (fetch_order_book x2 shards) -> consolidate
# Logs: logs/backfill_20260928/*.log   Progress: tail -f logs/backfill_20260928/*.log
set -u
cd /Users/abhinavs./Code/Zoom
L=logs/backfill_20260928; mkdir -p $L
P=/usr/bin/python3
echo "[$(date +%T)] start" | tee -a $L/main.log
( while true; do kb=$(ps -axo rss,comm | awk '/[Pp]ython/{s+=$1} END{print s+0}'); if [ "$kb" -gt 57671680 ]; then echo "[$(date +%T)] WATCHDOG: python ${kb}KB > 55GB, killing backfill" | tee -a $L/main.log; pkill -f "fetch_pnl_history|fetch_order_book"; fi; sleep 30; done ) & WD=$!

streamA() {
  nice -n 10 $P src/agentic/fetch_pnl_history.py calendar >> $L/pnl_calendar.log 2>&1      # resumable: instant when done
  echo "[$(date +%T)] A: calendar done" | tee -a $L/main.log
  for n in 0 1 2 3 4 5; do nice -n 10 $P src/agentic/fetch_pnl_history.py details $n 6 >> $L/pnl_details_$n.log 2>&1 & done
  wait
  echo "[$(date +%T)] A: details done" | tee -a $L/main.log
  nice -n 10 $P src/agentic/fetch_pnl_history.py integrated 0 1 >> $L/pnl_integrated.log 2>&1
  nice -n 10 $P src/agentic/fetch_pnl_history.py normalize >> $L/pnl_normalize.log 2>&1
  echo "[$(date +%T)] A: pnl_quarterly normalized" | tee -a $L/main.log
  nice -n 10 $P src/agentic/build_mcap_pit.py >> $L/mcap_pit.log 2>&1
  echo "[$(date +%T)] A: mcap_pit rebuilt — STREAM A DONE" | tee -a $L/main.log
}
streamB() {
  for n in 0 1 2 3 4 5 6 7; do nice -n 10 $P src/agentic/fetch_order_book.py --shard $n --of 8 >> $L/orderbook_$n.log 2>&1 & done
  wait
  nice -n 10 $P src/agentic/fetch_order_book.py --consolidate >> $L/orderbook_consolidate.log 2>&1
  echo "[$(date +%T)] B: order book consolidated — STREAM B DONE" | tee -a $L/main.log
}
streamA & A=$!
streamB & B=$!
wait $A $B
kill $WD 2>/dev/null
echo "[$(date +%T)] ALL BACKFILL DONE" | tee -a $L/main.log
tail -3 $L/pnl_normalize.log $L/orderbook_consolidate.log | tee -a $L/main.log
