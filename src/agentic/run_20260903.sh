#!/usr/bin/env bash
# Sep-3 evening run — entry Sep-4 pre-open. Data layer already fresh (manual 20:00 run).
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/weekly_pipeline; TS=$(date +%Y%m%d_%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local label="$1"; shift
  if "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "✅ $label"; else
    log "❌ $label"; tail -6 "$LOG_DIR/${TS}_${label}.log"; return 1; fi }

log "☐ RL: miss learner + classifier"
/usr/bin/python3 src/agentic/miss_learner.py --entry 2026-08-24 --exit 2026-08-28 --top-n 20 > "$LOG_DIR/${TS}_miss.log" 2>&1 && log "✅ miss_learner" || log "⚠ miss_learner non-fatal"
run classifier /usr/bin/python3 src/agentic/train_missed_winner_classifier.py || exit 1
grep -Eo "AUC=[0-9.]+ AUC-PR=[0-9.]+" "$LOG_DIR/${TS}_classifier.log" | tail -1

log "☐ ENGINES x5"
/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_cs.log"   2>&1 & P1=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_hc.log"   2>&1 & P2=$!
/usr/bin/python3 src/agentic/find_multibagger_today.py    > "$LOG_DIR/${TS}_mb.log"   2>&1 & P3=$!
/usr/bin/python3 src/agentic/run_multi_horizon.py         > "$LOG_DIR/${TS}_mh.log"   2>&1 & P4=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_180d.log" 2>&1 & P5=$!
for np in "cs:$P1" "hc:$P2" "mb:$P3" "mh:$P4" "180d:$P5"; do
  n=${np%%:*}; p=${np##*:}; wait $p && log "✅ engine $n" || log "❌ engine $n"
done

log "☐ GATE → BASKET → FORENSICS → SHADOW"
/usr/bin/python3 src/agentic/emit_freshness_status.py
run gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED — NO BASKET"; exit 1; }
run basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
run forensics /usr/bin/python3 src/agentic/tape_forensics.py || log "⚠ forensics non-fatal"
grep -E "SEVERITY-3" "$LOG_DIR/${TS}_forensics.log" || log "(no severity-3)"
run shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "⚠ shadow non-fatal"
run sleeve /usr/bin/python3 src/agentic/paper_sleeve.py || log "⚠ sleeve non-fatal"
run attribution /usr/bin/python3 src/agentic/score_engine_attribution.py || log "⚠ attribution non-fatal"
log "═══ RUN COMPLETE ═══"
