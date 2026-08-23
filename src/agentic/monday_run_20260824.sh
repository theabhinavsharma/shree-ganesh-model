#!/usr/bin/env bash
# MONDAY-ENTRY RUN (entry 2026-08-24 pre-open) — checklist-by-checklist, data through Fri Aug-21.
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

log "☐ 1. PRICES (through Fri Aug-21) + continuity"
run c1_prices /usr/bin/python3 src/agentic/refresh_prices.py || exit 1
qc "prices Fri + continuous" "
import sys; sys.path.insert(0,'src/agentic')
from verify_freshness import check_continuity
import pandas as pd
r = check_continuity(); assert not r['is_stale'], r['msg']
px = pd.read_parquet('data/derived/stock_daily_facts_adjusted_2015plus.parquet', columns=['trade_date'])
mx = pd.to_datetime(px['trade_date']).max().date(); print('max', mx); assert str(mx) >= '2026-08-21'"

log "☐ 2. CORPORATE ACTIONS (both stores)"
/usr/bin/python3 - > "$LOG_DIR/${TS}_c2_ca.log" 2>&1 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from datetime import date
from pathlib import Path
import pandas as pd
from src.ingest.corporate_actions.nse import NseCorporateActionsFetchConfig, load_corporate_actions_from_nse
ROOT = Path('.')
new = load_corporate_actions_from_nse(NseCorporateActionsFetchConfig(
    output_dir=ROOT/'data/corporate_actions_full_history/_incremental',
    start_date=date(2026, 8, 10), end_date=date.today()))
new['ex_date'] = pd.to_datetime(new['ex_date'], errors='coerce')
FULL = ROOT/'data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet'
full = pd.read_parquet(FULL); full['ex_date'] = pd.to_datetime(full['ex_date'], errors='coerce')
key = ['symbol','ex_date','subject','series']
m = pd.concat([full, new], ignore_index=True).drop_duplicates(subset=key, keep='last').sort_values(['ex_date','symbol'])
m.to_parquet(FULL, index=False)
splits = new[new['subject'].str.contains('plit|onus|ub-divi', na=False)]
print(f"full store: {len(m):,} rows, max {m['ex_date'].max().date()}, new splits/bonuses this fetch: {len(splits)}")
if len(splits): print(splits[['symbol','ex_date','subject']].to_string(index=False))
PYEOF
if [ $? -eq 0 ]; then log "  ✅ c2_ca"; cat "$LOG_DIR/${TS}_c2_ca.log" | head -3; else log "  ❌ c2_ca FAILED"; tail -5 "$LOG_DIR/${TS}_c2_ca.log"; exit 1; fi

log "☐ 3. ANNOUNCEMENTS (real store)"
/usr/bin/python3 - > "$LOG_DIR/${TS}_c3_ann.log" 2>&1 << 'PYEOF'
import sys; sys.path.insert(0, '.')
from datetime import date
from pathlib import Path
import pandas as pd
from src.ingest.events.nse import NseAnnouncementFetchConfig, load_announcements_from_nse
from src.utils.io import write_parquet
ROOT = Path('.')
DEST = ROOT/'data/events_full_history/normalized/stock_announcements.parquet'
new = load_announcements_from_nse(NseAnnouncementFetchConfig(
    output_dir=ROOT/'data/events_full_history/_incremental',
    start_date=date(2026, 8, 12), end_date=date.today()))
old = pd.read_parquet(DEST); old['event_date'] = pd.to_datetime(old['event_date'], errors='coerce')
m = pd.concat([old, new], ignore_index=True).sort_values(['event_date','symbol','sequence_id'])
m = m.drop_duplicates(subset=['sequence_id','symbol'], keep='last').reset_index(drop=True)
write_parquet(m, DEST)
print(f"{len(m):,} rows, max {m['event_date'].max()}")
PYEOF
if [ $? -eq 0 ]; then log "  ✅ c3_ann"; else log "  ❌ c3_ann FAILED"; exit 1; fi
qc "announcements fresh" "
import pandas as pd
a = pd.read_parquet('data/events_full_history/normalized/stock_announcements.parquet', columns=['event_date'])
d = pd.to_datetime(a['event_date'], errors='coerce'); print('max', d.max().date())
assert str(d.max().date()) >= '2026-08-21'"

log "☐ 4. NEWS EVENTS"
run c4_news /usr/bin/python3 src/agentic/build_news_event_features.py || exit 1
log "☐ 5. MACRO (FX + commodities + global/FRED)"
run c5a_fx /usr/bin/python3 src/agentic/fetch_forex_macro.py || exit 1
run c5b_cmdty /usr/bin/python3 src/agentic/fetch_commodity_prices.py || exit 1
run c5c_global /usr/bin/python3 src/agentic/fetch_global_macro.py || log "  ⚠ yahoo leg (expected) — FRED fallback ran inside"
log "☐ 6. PANELS + INDUSTRY + BREADTH"
run c6a_panel /usr/bin/python3 src/agentic/build_macro_panel.py || exit 1
run c6b_industry /usr/bin/python3 src/agentic/fetch_industry_indicators.py || exit 1
run c6c_breadth /usr/bin/python3 src/agentic/fetch_market_breadth.py || log "  ⚠ breadth non-fatal"
run c6d_panel2 /usr/bin/python3 src/agentic/build_macro_panel.py || exit 1
qc "macro cols non-null fresh" "
import pandas as pd
m = pd.read_parquet('data/derived/macro_panel.parquet'); m['trade_date']=pd.to_datetime(m['trade_date'])
for c in ['usdinr','brent','us_10y','dxy','us_vix','spx']:
    s = m.dropna(subset=[c]); d = str(s['trade_date'].max().date()); print(c, d)
    assert d >= '2026-08-17', f'{c} stale {d}'"

log "☐ 7. ENRICHMENT (qual layer, non-fatal)"
for s in fetch_fii_dii fetch_block_deals fetch_news_rss score_sentiment fetch_global_macro_sentiment; do
  /usr/bin/python3 src/agentic/$s.py > "$LOG_DIR/${TS}_c7_$s.log" 2>&1 && log "  ✅ $s" || log "  ⚠ $s failed (non-fatal)"
done

log "☐ 8. RL: self-score + miss learner + classifier"
run c8a_score /usr/bin/python3 src/agentic/score_basket_outcomes.py || log "  ⚠ score non-fatal"
cat "$LOG_DIR/${TS}_c8a_score.log" 2>/dev/null | tail -4
/usr/bin/python3 src/agentic/miss_learner.py --entry 2026-08-19 --exit 2026-08-21 --top-n 20 > "$LOG_DIR/${TS}_c8b_miss.log" 2>&1 \
  && log "  ✅ miss_learner (Aug-19→21 partial window)" || log "  ⚠ miss_learner failed"
run c8c_classifier /usr/bin/python3 src/agentic/train_missed_winner_classifier.py || exit 1
qc "classifier sane" "
import re
t = open('$LOG_DIR/${TS}_c8c_classifier.log').read()
m = re.search(r'AUC=([0-9.]+) AUC-PR=([0-9.]+)', t); assert m
print('AUC', m.group(1), 'PR', m.group(2))
assert float(m.group(1)) > 0.55 and float(m.group(2)) > 0.05"

log "☐ 9. ENGINES x5"
/usr/bin/python3 src/agentic/compare_short_horizons.py    > "$LOG_DIR/${TS}_c9_cs.log"   2>&1 & P1=$!
/usr/bin/python3 src/agentic/find_high_conviction.py      > "$LOG_DIR/${TS}_c9_hc.log"   2>&1 & P2=$!
/usr/bin/python3 src/agentic/find_multibagger_today.py    > "$LOG_DIR/${TS}_c9_mb.log"   2>&1 & P3=$!
/usr/bin/python3 src/agentic/run_multi_horizon.py         > "$LOG_DIR/${TS}_c9_mh.log"   2>&1 & P4=$!
/usr/bin/python3 src/agentic/find_180d_frontier_honest.py > "$LOG_DIR/${TS}_c9_180d.log" 2>&1 & P5=$!
for np in "cs:$P1" "hc:$P2" "mb:$P3" "mh:$P4" "180d:$P5"; do
  n=${np%%:*}; p=${np##*:}
  wait $p && log "  ✅ engine $n" || log "  ❌ engine $n failed"
done

log "☐ 10. GATE (27 checks) → BASKET → FORENSICS → SHADOW"
/usr/bin/python3 src/agentic/emit_freshness_status.py
run c10_gate /usr/bin/python3 src/agentic/verify_freshness.py || { log "🛑 GATE FAILED"; grep FAIL "$LOG_DIR/${TS}_c10_gate.log"; exit 1; }
run c10_basket /usr/bin/python3 src/agentic/generate_hybrid_basket.py || exit 1
qc "basket contract" "
import json, datetime
b = json.load(open(f'live_predictions/{datetime.date.today()}_15d5pct.json'))
assert len(b['picks']) == 8
for p in b['picks'] + b.get('reserves', []):
    assert 'rank' in p and 'confidence' in p and p.get('confidence_rationale') and 'sl_pct' in p
print('contract OK, data_through', b['data_through'])"
run c11_forensics /usr/bin/python3 src/agentic/tape_forensics.py || log "  ⚠ forensics non-fatal"
cat "$LOG_DIR/${TS}_c11_forensics.log" 2>/dev/null | grep -E "SEVERITY-3|sev3" || true
run c12_shadow /usr/bin/python3 src/agentic/lean_shadow_check.py || log "  ⚠ shadow non-fatal"
log "════════ MONDAY RUN COMPLETE ════════"
