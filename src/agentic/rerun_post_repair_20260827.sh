#!/usr/bin/env bash
# POST-REPAIR RERUN (2026-08-27 ~23:55): classifier + engines + gate + basket + forensics
# + shadow on the CA-repaired parquet. Provisional basket preserved as *_prerepair.json.
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/weekly_pipeline; TS=$(date +%Y%m%d_%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local label="$1"; shift
  log "▸ $label"
  if "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "  ✅ $label"; else
    log "  ❌ $label FAILED"; tail -8 "$LOG_DIR/${TS}_${label}.log"; return 1; fi }

log "☐ R1. CLASSIFIER RETRAIN (repaired data)"
run r1_classifier /usr/bin/python3 src/agentic/train_missed_winner_classifier.py || exit 1
grep -Eo "AUC=[0-9.]+ AUC-PR=[0-9.]+" "$LOG_DIR/${TS}_r1_classifier.log" | tail -1

log "☐ R2. ENGINES x5 (repaired data)"
/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_r2_cs.log"   2>&1 & P1=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_r2_hc.log"   2>&1 & P2=$!
/usr/bin/python3 src/agentic/find_multibagger_today.py    > "$LOG_DIR/${TS}_r2_mb.log"   2>&1 & P3=$!
/usr/bin/python3 src/agentic/run_multi_horizon.py         > "$LOG_DIR/${TS}_r2_mh.log"   2>&1 & P4=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_r2_180d.log" 2>&1 & P5=$!
for np in "cs:$P1" "hc:$P2" "mb:$P3" "mh:$P4" "180d:$P5"; do
  n=${np%%:*}; p=${np##*:}
  wait $p && log "  ✅ engine $n" || log "  ❌ engine $n failed"
done

log "☐ R3. GATE → BASKET → FORENSICS → SHADOW"
/usr/bin/python3 src/agentic/emit_freshness_status.py
run r3_gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED"; exit 1; }
run r3_basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
run r4_forensics /usr/bin/python3 src/agentic/tape_forensics.py || log "  ⚠ forensics non-fatal"
grep -E "SEVERITY-3" "$LOG_DIR/${TS}_r4_forensics.log" 2>/dev/null || log "  (no severity-3)"
run r5_shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "  ⚠ shadow non-fatal"
log "════════ POST-REPAIR RERUN COMPLETE ════════"
