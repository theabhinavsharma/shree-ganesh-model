#!/usr/bin/env bash
# Serial resume of the Sep-3 run — ONE process at a time (memory law 2026-09-03:
# parallel engines each load the 5M-row panel and together eat 90-100GB; serial
# costs ~30 min more and keeps the laptop alive). nice -n 10 everything.
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/weekly_pipeline; TS=$(date +%Y%m%d_%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local label="$1"; shift
  if nice -n 10 "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "✅ $label"; else
    log "❌ $label"; tail -6 "$LOG_DIR/${TS}_${label}.log"; return 1; fi }

log "☐ ENGINES (serial)"
run cs   /usr/bin/python3 src/agentic/compare_short_horizons.py    || exit 1
run hc   /usr/bin/python3 src/agentic/find_high_conviction.py      || exit 1
run mb   /usr/bin/python3 src/agentic/find_multibagger_today.py    || exit 1
run mh   /usr/bin/python3 src/agentic/run_multi_horizon.py         || exit 1
# 180d: another session may already be running it — wait for that process, use its output if fresh
while pgrep -f "find_180d_frontier_honest.py" > /dev/null; do sleep 30; done
if [ -n "$(find data/derived/180d_today_predictions.parquet -newermt '2026-09-03 20:00' 2>/dev/null)" ]; then
  log "✅ 180d (from other session)"
else
  run 180d /usr/bin/python3 src/agentic/find_180d_frontier_honest.py || exit 1
fi

log "☐ GATE → BASKET → FORENSICS → SHADOW → SLEEVE → ATTRIBUTION (serial)"
nice -n 10 /usr/bin/python3 src/agentic/emit_freshness_status.py
run gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED — NO BASKET"; exit 1; }
run basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
run forensics /usr/bin/python3 src/agentic/tape_forensics.py || log "⚠ forensics non-fatal"
grep -E "SEVERITY-3" "$LOG_DIR/${TS}_forensics.log" || log "(no severity-3)"
run shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "⚠ shadow non-fatal"
run sleeve /usr/bin/python3 src/agentic/paper_sleeve.py || log "⚠ sleeve non-fatal"
run attribution /usr/bin/python3 src/agentic/score_engine_attribution.py || log "⚠ attribution non-fatal"
log "═══ SERIAL RUN COMPLETE ═══"
