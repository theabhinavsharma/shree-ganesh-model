#!/usr/bin/env bash
# RECONCILE RUN 2026-09-19 — full convergence check per SHOWCASE.html §Recreation Kit.
# One command:   bash reconcile_20260919.sh
# Runs in the background under nohup with a 55 GB memory watchdog; tail the log:
#   tail -f logs/reconcile_20260919.log
set -u
cd /Users/abhinavs./Documents/Zoom
LOG=logs/reconcile_20260919.log
PY=/usr/bin/python3
MEM_CAP_GB=55

if [ "${1:-}" != "--inner" ]; then
  nohup bash "$0" --inner > "$LOG" 2>&1 &
  echo "launched pid $! — tail -f $LOG"
  exit 0
fi

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- memory watchdog: kill the whole pipeline if python RSS sum > cap ----
(
  while true; do
    rss_kb=$(ps -axo rss,comm | awk '/[Pp]ython/ {s+=$1} END {print s+0}')
    gb=$((rss_kb / 1024 / 1024))
    if [ "$gb" -gt "$MEM_CAP_GB" ]; then
      echo "[$(date +%H:%M:%S)] 🛑 WATCHDOG: python RSS ${gb} GB > ${MEM_CAP_GB} GB — killing python" >> "$LOG"
      pkill -f "src/agentic/" ; sleep 5; pkill -9 -f "src/agentic/"
    fi
    sleep 15
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

log "══════ 0. TESTS ══════"
$PY -m pytest tests/test_no_per_symbol_constants.py tests/test_no_unadjusted_corporate_actions.py -v 2>&1 | tail -25
log "tests exit: ${PIPESTATUS[0]}"

log "══════ 1. WEEKLY PIPELINE — with fetch (data is 12d stale, gate needs it), dry-run (no git) ══════"
bash src/agentic/run_weekly_pipeline.sh --dry-run
PIPE_RC=$?
log "pipeline exit: $PIPE_RC"

log "══════ CONVERGENCE CHECK 1 — verify_freshness.py ══════"
$PY src/agentic/verify_freshness.py; log "freshness exit: $?"

DATE=$(date +%Y-%m-%d)
B="live_predictions/${DATE}_15d5pct.json"
log "══════ CONVERGENCE CHECK 2 — basket contract on $B ══════"
if [ -f "$B" ]; then
$PY - "$B" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1])); ok=True
def chk(c,msg):
    global ok
    print(("  ✅ " if c else "  ❌ ")+msg); ok = ok and c
chk(len(d["picks"])==8, f"8 picks (got {len(d['picks'])})")
chk(all(abs(p["weight_pct"]-12.5)<1e-9 for p in d["picks"]), "all picks 12.5%")
chk(len(d.get("reserves",[]))==2, f"2 reserves (got {len(d.get('reserves',[]))})")
need=["rank","confidence","confidence_rationale","sl_pct","sl_3pct","buy_low","buy_high","target_5pct","tier","band_fit","ml_score","engines_count"]
for name in ("picks","reserves"):
    for p in d[name]:
        miss=[k for k in need if k not in p]; chk(not miss, f"{name}/{p['symbol']} fields {miss or 'ok'}")
        c=p["close"]; chk(abs(p["buy_low"]-round(c*0.99,2))<0.011 and abs(p["buy_high"]-round(c*1.01,2))<0.011, f"{p['symbol']} buy zone ×0.99/1.01")
        chk(abs(p["target_5pct"]-round(c*1.05,2))<0.011, f"{p['symbol']} target ×1.05")
        chk(-0.12-1e-9<=p["sl_pct"]<=-0.03+1e-9, f"{p['symbol']} sl_pct {p['sl_pct']:.4f} in [-12%,-3%]")
sls={round(p["sl_pct"],4) for p in d["picks"]}; chk(len(sls)>1, f"SL is per-pick vol-scaled ({len(sls)} distinct values)")
chk(all(0.50<=p["ml_score"]<=0.75 for p in d["picks"] if p["tier"]==2), "tier-2 picks in ML honest zone [0.50,0.75]")
chk("regime_gate" in d and "exit_contract" in d and "inputs_mtime" in d, "regime_gate/exit_contract/inputs_mtime present")
print("CHECK 2:", "PASS" if ok else "FAIL")
PYEOF
else log "❌ no basket emitted at $B"; fi

log "══════ CONVERGENCE CHECK 3 — generator determinism on identical inputs ══════"
# Same inputs + same filter must give the same 8 names/order/scores. Re-emit and diff
# everything except as_of_date/inputs_mtime. (Full engine-retrain reproduce of an OLD
# committed basket needs the price panel truncated to that data_through — engines take
# no date arg today; recorded in the debt ledger below.)
if [ -f "$B" ]; then
  cp "$B" tmp/_check3_first.json
  $PY src/agentic/generate_hybrid_basket.py > logs/reconcile_check3_rerun.log 2>&1
  $PY - <<'PYEOF'
import json
a=json.load(open("tmp/_check3_first.json")); b=json.load(open(__import__("glob").glob("live_predictions/*_15d5pct.json") and sorted(__import__("glob").glob("live_predictions/*_15d5pct.json"))[-1]))
for d in (a,b):
    d.pop("as_of_date",None); d.pop("inputs_mtime",None)
same = a==b
print("  names A:", [p["symbol"] for p in a["picks"]]); print("  names B:", [p["symbol"] for p in b["picks"]])
print("CHECK 3:", "PASS — identical picks, ranks, scores, SLs" if same else "FAIL — outputs differ on identical inputs")
PYEOF
fi

log "══════ SUMMARY ══════"
log "pipeline rc=$PIPE_RC · log=$LOG · basket=$B"
log "DONE"
