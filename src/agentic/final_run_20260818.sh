#!/usr/bin/env bash
# FINAL RUN 2026-08-18 — on today's close, corrected CA adjustments, C2 contract.
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/weekly_pipeline; TS=$(date +%Y%m%d_%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local label="$1"; shift
  log "▸ $label"
  if "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "  ✅ $label"; else
    log "  ❌ $label FAILED"; tail -8 "$LOG_DIR/${TS}_${label}.log"; return 1; fi }
qc(){ local label="$1"
  if /usr/bin/python3 -c "$2"; then log "  🛡 QC PASS: $label"; else
    log "  🛑 QC FAIL: $label — ABORTING"; exit 1; fi }

log "════════ FINAL RUN on 2026-08-18 close (corrected CA) ════════"
run F_news     /usr/bin/python3 src/agentic/build_news_event_features.py || exit 1
run F_panel1   /usr/bin/python3 src/agentic/build_macro_panel.py || exit 1
run F_industry /usr/bin/python3 src/agentic/fetch_industry_indicators.py || exit 1
run F_breadth  /usr/bin/python3 src/agentic/fetch_market_breadth.py || log "  ⚠ breadth non-fatal"
run F_panel2   /usr/bin/python3 src/agentic/build_macro_panel.py || exit 1

log "═══ ENGINES (5 parallel, corrected data) ═══"
/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_F_e_cs.log"   2>&1 & P1=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_F_e_hc.log"   2>&1 & P2=$!
/usr/bin/python3 src/agentic/find_multibagger_today.py    > "$LOG_DIR/${TS}_F_e_mb.log"   2>&1 & P3=$!
/usr/bin/python3 src/agentic/run_multi_horizon.py         > "$LOG_DIR/${TS}_F_e_mh.log"   2>&1 & P4=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_F_e_180d.log" 2>&1 & P5=$!
for np in "cs:$P1" "hc:$P2" "mb:$P3" "mh:$P4" "180d:$P5"; do
  n=${np%%:*}; p=${np##*:}
  wait $p && log "  ✅ engine $n" || log "  ❌ engine $n failed"
done
qc "engines non-trivial" "
import pandas as pd
cs = pd.read_parquet('data/derived/compare_short_horizons.parquet')
assert len(cs) > 100; print('cs rows:', len(cs))"

run F_classifier /usr/bin/python3 src/agentic/train_missed_winner_classifier.py || exit 1
/usr/bin/python3 src/agentic/emit_freshness_status.py
run F_gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED"; grep FAIL "$LOG_DIR/${TS}_F_gate.log"; exit 1; }
run F_basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
qc "basket contract (C2 + ranks + Aug-18)" "
import json
b = json.load(open('live_predictions/2026-08-18_15d5pct.json'))
assert b['data_through'] >= '2026-08-18', b['data_through']
assert len(b['picks']) == 8 and abs(sum(p['weight_pct'] for p in b['picks']) - 100) < 0.01
for p in b['picks'] + b.get('reserves', []):
    assert 'rank' in p and 'confidence' in p and p.get('confidence_rationale') and 'sl_pct' in p
print('contract OK, data_through', b['data_through'])"
run F_shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "  ⚠ shadow non-fatal"
run F_score  /usr/bin/python3 src/agentic/score_basket_outcomes.py || true

log "═══ ENRICHMENT (no timeout cmd on macOS — plain runs) ═══"
for s in fetch_fii_dii fetch_block_deals fetch_news_rss score_sentiment fetch_global_macro_sentiment; do
  /usr/bin/python3 src/agentic/$s.py > "$LOG_DIR/${TS}_enr_$s.log" 2>&1 \
    && log "  ✅ $s" || log "  ⚠ $s failed (non-fatal)"
done
log "════════ FINAL RUN COMPLETE ════════"
