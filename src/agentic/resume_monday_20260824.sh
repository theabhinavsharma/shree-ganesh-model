#!/usr/bin/env bash
# Resume Monday run after macOS TCC blackout — items 9(partial)→12.
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/weekly_pipeline; TS=$(date +%Y%m%d_%H%M%S)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
run(){ local label="$1"; shift
  log "▸ $label"
  if "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "  ✅ $label"; else
    log "  ❌ $label FAILED"; tail -8 "$LOG_DIR/${TS}_${label}.log"; return 1; fi }
qc(){ local label="$1"
  if /usr/bin/python3 -c "$2"; then log "  🛡 QC: $label"; else
    log "  🛑 QC FAIL: $label — ABORT"; exit 1; fi }

log "☐ 9(resume). ENGINES cs/hc/180d (mb+mh already fresh today)"
/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_r9_cs.log"   2>&1 & P1=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_r9_hc.log"   2>&1 & P2=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_r9_180d.log" 2>&1 & P3=$!
for np in "cs:$P1" "hc:$P2" "180d:$P3"; do
  n=${np%%:*}; p=${np##*:}
  wait $p && log "  ✅ engine $n" || log "  ❌ engine $n failed"
done
qc "all 5 engine outputs fresh today" "
from pathlib import Path
from datetime import datetime, date
for f in ['data/derived/compare_short_horizons.parquet','data/derived/high_conviction_predictions.parquet',
          'data/derived/180d_today_predictions.parquet','data/derived/multibagger_today_predictions.parquet',
          'tmp/from_scratch_7d_run/multi_horizon_top.csv']:
    m = datetime.fromtimestamp(Path(f).stat().st_mtime).date()
    assert m >= date(2026,8,23), f'{f} stale mtime {m}'
    print(f.split('/')[-1], m)"

log "☐ 10. GATE (27 checks) → BASKET"
/usr/bin/python3 src/agentic/emit_freshness_status.py
run r10_gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED"; grep FAIL "$LOG_DIR/${TS}_r10_gate.log"; exit 1; }
run r10_basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
qc "basket contract" "
import json, datetime
b = json.load(open(f'live_predictions/{datetime.date.today()}_15d5pct.json'))
assert len(b['picks']) == 8
for p in b['picks'] + b.get('reserves', []):
    assert 'rank' in p and 'confidence' in p and p.get('confidence_rationale') and 'sl_pct' in p
print('contract OK, data_through', b['data_through'])"

log "☐ 11. TAPE FORENSICS (every name)"
run r11_forensics /usr/bin/python3 src/agentic/tape_forensics.py || log "  ⚠ forensics non-fatal"
grep -E "SEVERITY-3" "$LOG_DIR/${TS}_r11_forensics.log" || log "  (no severity-3 names)"

log "☐ 12. SHADOW"
run r12_shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "  ⚠ shadow non-fatal"
log "════════ MONDAY RUN COMPLETE ════════"
