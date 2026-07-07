#!/usr/bin/env bash
# WEEKLY 15D/+5% PIPELINE ORCHESTRATOR
#
# Runs the complete workflow end-to-end:
#   1. Data refresh (prices, announcements, news, macro, industry)
#   2. All 5 ML engines in parallel
#   3. Miss learner (learn from prior week's misses)
#   4. Supervised ML classifier retrain
#   5. Hybrid basket generation (15D/+5%)
#   6. Git commit + push
#
# Usage:
#   bash src/agentic/run_weekly_pipeline.sh              # full pipeline
#   bash src/agentic/run_weekly_pipeline.sh --skip-fetch # skip data fetch
#   bash src/agentic/run_weekly_pipeline.sh --dry-run    # no git operations
#
# Per CONSTITUTION.md §1.7 — this script is the reproducibility contract.
# A future user clones the repo, runs this once, gets the current basket.

set -euo pipefail

ROOT="/Users/abhinavs./Documents/Zoom"
cd "$ROOT"

TS=$(date +%Y%m%d_%H%M%S)
DATE=$(date +%Y-%m-%d)
LOG_DIR="logs/weekly_pipeline"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/pipeline_${TS}.log"

# ---------------- helpers ----------------

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }
fail() { log "❌ FAIL: $*"; exit 1; }
step() { log ""; log "═══════════════ $* ═══════════════"; }

run_py() {
  local script="$1"; local label="$2"; local logfile="$LOG_DIR/${TS}_${label}.log"
  log "▸ $label — python3 $script"
  if /usr/bin/python3 "$script" > "$logfile" 2>&1; then
    log "  ✅ $label ($(wc -l < "$logfile") log lines)"
  else
    tail -20 "$logfile" | tee -a "$MASTER_LOG"
    fail "$label — see $logfile"
  fi
}

# ---------------- arg parsing ----------------

SKIP_FETCH=0
DRY_RUN=0
for a in "$@"; do
  case "$a" in
    --skip-fetch) SKIP_FETCH=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) sed -n '1,25p' "$0"; exit 0 ;;
  esac
done

log "═══════════════════════════════════════════════════════════════"
log " WEEKLY 15D/+5% PIPELINE  ·  run ${TS}"
log "═══════════════════════════════════════════════════════════════"
log " ROOT      : $ROOT"
log " Date      : $DATE"
log " Skip fetch: $SKIP_FETCH"
log " Dry run   : $DRY_RUN"
log " Master log: $MASTER_LOG"

# ---------------- 1. Data layer ----------------

if [ $SKIP_FETCH -eq 0 ]; then
  step "1. DATA LAYER"
  run_py src/agentic/refresh_prices.py               "01_prices"
  run_py src/agentic/refresh_announcements.py        "02_announcements"     || log "  ⚠ announcements failed — continuing"
  run_py src/agentic/build_news_event_features.py    "03_news_events"
  run_py src/agentic/build_macro_panel.py            "04_macro"
  run_py src/agentic/fetch_industry_indicators.py    "05_industry"
else
  log "▸ SKIPPING data refresh (--skip-fetch)"
fi

# ---------------- 2. ML engines (parallel) ----------------

step "2. ML ENGINES (5 in parallel)"

/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_e_cs.log"    2>&1 & PID_CS=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_e_hc.log"    2>&1 & PID_HC=$!
/usr/bin/python3 src/agentic/find_multibagger_today.py    > "$LOG_DIR/${TS}_e_mb.log"    2>&1 & PID_MB=$!
/usr/bin/python3 src/agentic/run_multi_horizon.py         > "$LOG_DIR/${TS}_e_mh.log"    2>&1 & PID_MH=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_e_180d.log"  2>&1 & PID_180=$!

log "  launched: cs=$PID_CS hc=$PID_HC mb=$PID_MB mh=$PID_MH 180d=$PID_180"
log "  waiting for all 5 to complete…"

FAILED=""
for name_pid in "cs:$PID_CS" "hc:$PID_HC" "mb:$PID_MB" "mh:$PID_MH" "180d:$PID_180"; do
  name=${name_pid%%:*}; pid=${name_pid##*:}
  if wait $pid; then
    log "  ✅ $name completed"
  else
    log "  ❌ $name failed"
    FAILED="$FAILED $name"
  fi
done

if [ -n "$FAILED" ]; then
  log "⚠️  Some engines failed:$FAILED — continuing with what completed"
fi

# ---------------- 3. Miss learner + ML classifier ----------------

step "3. RL LOOP — miss_learner + train classifier"

# Miss learner needs entry/exit dates for last week's window
LAST_MONDAY=$(python3 -c "
from datetime import date, timedelta
t = date.today()
last_mon = t - timedelta(days=t.weekday() + 7)
print(last_mon)
")
LAST_FRIDAY=$(python3 -c "
from datetime import date, timedelta
t = date.today()
# 4 = Friday, iso weekday-1
last_fri = t - timedelta(days=(t.weekday() + 3) % 7 + 7)
print(last_fri)
")

/usr/bin/python3 src/agentic/miss_learner.py --entry "$LAST_MONDAY" --exit "$LAST_FRIDAY" --top-n 20 > "$LOG_DIR/${TS}_miss_learner.log" 2>&1 \
  && log "  ✅ miss_learner ($LAST_MONDAY → $LAST_FRIDAY)" \
  || log "  ⚠ miss_learner failed — likely first run or insufficient history"

run_py src/agentic/train_missed_winner_classifier.py "06_ml_classifier"

# ---------------- 3.5. FRESHNESS GATE + STATUS ----------------
# HARD stop — refuse to proceed if any critical input is stale (file OR column level).
# This is here because on 2026-07-01 the pipeline emitted a basket on 55-day-stale macro data.
step "3.5. FRESHNESS GATE"

# Always emit the dashboard first — it never fails, so we get a report even when the gate blocks.
/usr/bin/python3 src/agentic/emit_freshness_status.py 2>&1 | tee -a "$MASTER_LOG"

if ! /usr/bin/python3 src/agentic/verify_freshness.py 2>&1 | tee "$LOG_DIR/${TS}_freshness.log"; then
  fail "FRESHNESS GATE FAILED — one or more inputs are stale. See $LOG_DIR/${TS}_freshness.log AND reports/freshness_status.md. Fix stale inputs and rerun. NO BASKET WAS EMITTED."
fi

# ---------------- 4. Hybrid basket generation ----------------

step "4. HYBRID BASKET (15D/+5%)"

run_py src/agentic/generate_hybrid_basket.py "07_hybrid_basket"

BASKET_FILE="live_predictions/${DATE}_15d5pct.json"
if [ ! -f "$BASKET_FILE" ]; then
  fail "Expected basket file not found: $BASKET_FILE"
fi

log ""
log "═══════════════ BASKET SUMMARY ═══════════════"
/usr/bin/python3 -c "
import json
d = json.load(open('$BASKET_FILE'))
print(f\"  Date        : {d['as_of_date']}\")
print(f\"  Data through: {d['data_through']}\")
print(f\"  Regime      : {d['regime_gate']}\")
print(f\"  Total exposure: {d['total_exposure_pct']}%  (rest {d['rest_in_liquidplus_pct']}% LIQUIDPLUS)\")
print(f\"  Names       : {len(d['picks'])}\")
for p in d['picks']:
    print(f\"    T{p['tier']} {p['symbol']:12s}  buy {p['buy_low']:.2f}-{p['buy_high']:.2f}  tgt {p['target_5pct']:.2f}  sl {p['sl_3pct']:.2f}  wt {p['weight_pct']}%\")
" | tee -a "$MASTER_LOG"

# ---------------- 4.5. Simplicity audit + recreation kit ----------------
# Keeps the public showcase inventory CURRENT (never historical) and tracks
# code size / findings drift per run. Neither step is fatal to the basket.

step "4.5. SIMPLICITY AUDIT + RECREATION KIT"

/usr/bin/python3 src/agentic/simplicity_auditor.py audit > "$LOG_DIR/${TS}_simplicity.log" 2>&1 \
  && log "  ✅ simplicity audit ($(grep -c '^-' reports/simplicity_audit.md 2>/dev/null || echo '?') findings, metrics appended)" \
  || log "  ⚠ simplicity audit failed — see $LOG_DIR/${TS}_simplicity.log"

/usr/bin/python3 src/agentic/build_recreation_kit.py > "$LOG_DIR/${TS}_kit.log" 2>&1 \
  && log "  ✅ recreation kit regenerated (SHOWCASE.html inventory current)" \
  || log "  ⚠ recreation kit failed — see $LOG_DIR/${TS}_kit.log"

# Sync showcase to deploy dir so the next `vercel --prod` ships the current inventory
cp SHOWCASE.html shreeganeshmodel-deploy/index.html 2>/dev/null \
  && log "  ✅ showcase synced to deploy dir" \
  || log "  ⚠ showcase sync failed"

# ---------------- 5. Git commit + push ----------------

step "5. GIT COMMIT + PUSH"

if [ $DRY_RUN -eq 1 ]; then
  log "▸ DRY RUN — skipping git operations"
else
  git add "$BASKET_FILE" logs/miss_learnings.jsonl logs/coverage_backtest_since_april.json data/derived/missed_winner_classifier.parquet \
          SHOWCASE.html shreeganeshmodel-deploy/index.html assets/recreation_manifest.json \
          reports/simplicity_audit.md reports/freshness_status.md \
          logs/simplicity_metrics.jsonl logs/debt_ledger.jsonl 2>/dev/null || true

  MSG="Weekly 15D/5pct pipeline — $DATE

Full 5-engine + ML classifier run.
Basket: $BASKET_FILE
Log: $MASTER_LOG

$(cat << EOF
$(python3 -c "
import json; d=json.load(open('$BASKET_FILE'))
print(f'Regime: {d[\"regime_gate\"]}')
print(f'Total exposure: {d[\"total_exposure_pct\"]}%')
print(f'Names: {len(d[\"picks\"])}')
for p in d['picks']:
    print(f'  T{p[\"tier\"]} {p[\"symbol\"]}  {p[\"weight_pct\"]}%  engines={p[\"engines_count\"]}  ml={p[\"ml_score\"]}')
")
EOF
)"

  git -c user.name="abhinavs" -c user.email="abhinavs@users.noreply.github.com" \
      commit -m "$MSG" 2>&1 | tail -3 | tee -a "$MASTER_LOG" \
    || log "  ⚠ nothing to commit"

  # Try to push (may fail if token expired — user handles)
  git -c credential.helper=osxkeychain push 2>&1 | tail -3 | tee -a "$MASTER_LOG" \
    || log "  ⚠ push failed — token may be expired"
fi

log ""
log "═══════════════ PIPELINE COMPLETE ═══════════════"
log " Total runtime: $((SECONDS/60))m $((SECONDS%60))s"
log " Basket: $BASKET_FILE"
log " Master log: $MASTER_LOG"
