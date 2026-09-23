#!/usr/bin/env bash
# RECONCILE RUN 2 — 2026-09-19 evening. After fixes: canonical announcements writer,
# weekly step 1 = daily_data_layer.sh, CA store full re-parse, macro_sent quarantine,
# report renderer. Same watchdog. Usage: bash reconcile_20260919b.sh ; tail -f logs/reconcile_20260919b.log
set -u
cd /Users/abhinavs./Documents/Zoom
LOG=logs/reconcile_20260919b.log; PY=/usr/bin/python3; MEM_CAP_GB=55
if [ "${1:-}" != "--inner" ]; then nohup bash "$0" --inner > "$LOG" 2>&1 & echo "launched pid $! — tail -f $LOG"; exit 0; fi
log() { echo "[$(date +%H:%M:%S)] $*"; }
( while true; do gb=$(( $(ps -axo rss,comm | awk '/[Pp]ython/ {s+=$1} END {print s+0}') / 1048576 ))
    [ "$gb" -gt "$MEM_CAP_GB" ] && { echo "[$(date +%H:%M:%S)] 🛑 WATCHDOG ${gb}GB — killing python" >> "$LOG"; pkill -f "src/agentic/"; sleep 5; pkill -9 -f "src/agentic/"; }
    sleep 15; done ) & WD=$!; trap 'kill $WD 2>/dev/null' EXIT

log "══════ 0. CA STORE RE-PARSE + PRICE RE-ADJUST (runs inside the data layer; shown here for the record) ══════"
log "══════ 1. WEEKLY PIPELINE — full fetch via daily_data_layer.sh, dry-run (no git) ══════"
bash src/agentic/run_weekly_pipeline.sh --dry-run; PIPE_RC=$?; log "pipeline exit: $PIPE_RC"

log "══════ 2. TESTS (after the data layer re-adjusted prices) ══════"
$PY -m pytest tests/test_no_per_symbol_constants.py tests/test_no_unadjusted_corporate_actions.py tests/test_corporate_actions.py -q 2>&1 | grep -vE "Warning|warn" | tail -15
log "tests exit: ${PIPESTATUS[0]}"

log "══════ CONVERGENCE CHECK 1 — verify_freshness.py ══════"
$PY src/agentic/verify_freshness.py | tail -3; log "freshness exit: ${PIPESTATUS[0]}"
DATE=$(date +%Y-%m-%d); B="live_predictions/${DATE}_15d5pct.json"
log "══════ CONVERGENCE CHECK 2 — basket contract on $B ══════"
if [ -f "$B" ]; then $PY - "$B" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); ok=True
def chk(c,m):
    global ok; print(("  ✅ " if c else "  ❌ ")+m); ok=ok and c
chk(len(d["picks"])==8,f"8 picks ({len(d['picks'])})"); chk(all(abs(p["weight_pct"]-12.5)<1e-9 for p in d["picks"]),"12.5% each")
chk(len(d.get("reserves",[]))==2,"2 reserves")
need=["rank","confidence","confidence_rationale","sl_pct","sl_3pct","buy_low","buy_high","target_5pct","tier","band_fit","ml_score","engines_count"]
for k in ("picks","reserves"):
    for p in d[k]:
        c=p["close"]; miss=[x for x in need if x not in p]
        chk(not miss and abs(p["buy_low"]-round(c*0.99,2))<0.011 and abs(p["buy_high"]-round(c*1.01,2))<0.011 and abs(p["target_5pct"]-round(c*1.05,2))<0.011 and -0.12-1e-9<=p["sl_pct"]/100<=-0.03+1e-9, f"{k}/{p['symbol']} fields+levels+SL ok" if not miss else f"{k}/{p['symbol']} missing {miss}")
chk(len({round(p["sl_pct"],2) for p in d["picks"]})>1,"per-pick vol-scaled SL")
chk(all(0.50<=p["ml_score"]<=0.75 for p in d["picks"] if p["tier"]==2),"tier-2 in honest zone")
print("CHECK 2:","PASS" if ok else "FAIL")
PYEOF
else log "❌ no basket at $B"; fi
log "══════ CONVERGENCE CHECK 3 — generator determinism ══════"
if [ -f "$B" ]; then cp "$B" tmp/_check3_first.json; $PY src/agentic/generate_hybrid_basket.py > logs/reconcile_check3_rerun.log 2>&1
$PY - "$B" <<'PYEOF'
import json,sys
a=json.load(open("tmp/_check3_first.json")); b=json.load(open(sys.argv[1]))
for d in (a,b): d.pop("as_of_date",None); d.pop("inputs_mtime",None)
print("  A:",[p["symbol"] for p in a["picks"]]); print("  B:",[p["symbol"] for p in b["picks"]])
print("CHECK 3:","PASS — identical picks/ranks/scores/SLs" if a==b else "FAIL — differs on identical inputs")
PYEOF
fi
log "══════ REPORTS ══════"; ls -la reports/basket_report_${DATE}.md reports/daily_actions_*.md 2>&1 | tail -3
log "DONE rc=$PIPE_RC"
