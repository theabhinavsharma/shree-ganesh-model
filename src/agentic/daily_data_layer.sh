#!/usr/bin/env bash
# DAILY DATA LAYER — every data feed, every trading day. Cron-scheduled (launchd is
# TCC-blocked on this Mac: the old com.shree-ganesh.daily-refresh died with
# "Operation not permitted" at 18:30 every weekday from 2026-05-07 to 2026-08-28).
#
# Cadence: daily post-close. Monday additionally runs the slow/weekly feeds.
# Data only — engines/baskets stay on the weekly pipeline per the RL protocol.
# Failure is LOUD: summary line + logs/daily_data_layer_status.json for the gate.
set -uo pipefail
cd /Users/abhinavs./Documents/Zoom
LOG_DIR=logs/daily_data_layer; mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S); DOW=$(date +%u)
log(){ echo "[$(date +%H:%M:%S)] $*"; }
PASS=(); FAIL=()
run(){ local label="$1"; shift
  if "$@" > "$LOG_DIR/${TS}_${label}.log" 2>&1; then log "✅ $label"; PASS+=("$label"); else
    log "❌ $label"; tail -3 "$LOG_DIR/${TS}_${label}.log"; FAIL+=("$label"); fi }

log "═══ DAILY DATA LAYER $TS (dow=$DOW) ═══"

# --- core price/CA/filings spine (order matters: CA before prices) ---
run corp_actions /usr/bin/python3 src/agentic/refresh_corporate_actions.py
run prices /usr/bin/python3 src/agentic/refresh_prices.py
run announcements /usr/bin/python3 src/agentic/refresh_announcements.py
run gold_inr /usr/bin/python3 src/agentic/build_gold_feed.py   # NSE gold ETFs -> INR gold (needs prices)
run news_events /usr/bin/python3 src/agentic/build_news_event_features.py

# --- daily macro + flows ---
run forex /usr/bin/python3 src/agentic/fetch_forex_macro.py
run usdinr_history /usr/bin/python3 src/agentic/fetch_usdinr_history.py   # FRED DEXINUS (pre-2024 FX for USD amounts)
run commodity /usr/bin/python3 src/agentic/fetch_commodity_prices.py
run global_macro /usr/bin/python3 src/agentic/fetch_global_macro.py
run fii_dii /usr/bin/python3 src/agentic/fetch_fii_dii.py
run block_deals /usr/bin/python3 src/agentic/fetch_block_deals.py
run breadth /usr/bin/python3 src/agentic/fetch_market_breadth.py
run india_vix /usr/bin/python3 src/agentic/fetch_india_vix.py
run industry /usr/bin/python3 src/agentic/fetch_industry_indicators.py

# --- daily news / narrative ---
run news_rss /usr/bin/python3 src/agentic/fetch_news_rss.py
run sentiment /usr/bin/python3 src/agentic/score_sentiment.py
run global_sentiment /usr/bin/python3 src/agentic/fetch_global_macro_sentiment.py
run pib_releases /usr/bin/python3 src/agentic/fetch_pib_releases.py --start "$(date -v-7d +%Y-%m-%d)" --match-symbols

# --- weekly (Mondays): fundamentals + holdings + recos ---
if [ "$DOW" = "1" ]; then
  # fetch_fundamentals.py (NSE quote API) is 403-blocked as of 2026-08-30 — 2,625/2,625
  # symbols rate-limited over 7h with ok=0. Screener.in path works; use it wide.
  run screener_fundamentals /usr/bin/python3 -c "import sys; sys.path.insert(0,'src/agentic'); import fetch_screener_fundamentals as m; m.main(1200)"
  run amfi_mf /usr/bin/python3 src/agentic/fetch_amfi_mf_holdings.py
  run superstar /usr/bin/python3 src/agentic/fetch_superstar_holdings.py
  run broker_recos /usr/bin/python3 src/agentic/fetch_broker_recos.py
  # security master (ISIN, fund units, renames, industry provenance) + screener industry for
  # names NSE no longer labels (checkpointed: only new unmapped symbols are fetched)
  run security_master /usr/bin/python3 src/agentic/build_security_master.py
  run screener_industry /usr/bin/python3 src/agentic/fetch_screener_industry.py
fi

# --- rebuild the panel last so everything above folds in ---
run macro_panel /usr/bin/python3 src/agentic/build_macro_panel.py

# --- daily orders + industry heat digest (2026-09-28): monitoring, not a feed and not a signal ---
if /usr/bin/python3 src/agentic/daily_orders_industry.py > "$LOG_DIR/${TS}_orders_industry.log" 2>&1; then log "✅ orders_industry digest (not a feed)"
else log "⚠ orders_industry digest failed (not a feed) — see $LOG_DIR/${TS}_orders_industry.log"; fi

# --- loud status ---
/usr/bin/python3 - << PYEOF
import json, datetime
json.dump({"ts": "$TS", "date": str(datetime.date.today()),
           "passed": "${PASS[*]:-}".split(), "failed": "${FAIL[*]:-}".split()},
          open("logs/daily_data_layer_status.json","w"), indent=1)
PYEOF
log "═══ DONE: ${#PASS[@]} ok, ${#FAIL[@]} failed (${FAIL[*]:-none}) ═══"

# evals read the status file written above, so they run after it; then the mirror copies the eval report too
# --- eval registry (evals/registry.yaml): plain-English statements + coded checks -> reports/eval_report_<date>.md ---
# Not a feed. Until the orchestrator lands it reports (exit 1 = a blocking eval failed); it does not yet stop later steps.
if /usr/bin/python3 src/agentic/run_evals.py --cadence daily > "$LOG_DIR/${TS}_evals.log" 2>&1; then log "✅ evals (not a feed)"
else log "❌ evals: blocking failure — see reports/eval_report_$(date +%Y-%m-%d).md"; fi

# --- mirror the live working set to Google Drive (My Drive/SGM backups/day_to_day; 2026-09-28) ---
# Not a data feed: logged here but never added to PASS/FAIL, so it cannot change the status file or the exit code.
if /bin/bash src/agentic/sync_drive_mirror.sh > "$LOG_DIR/${TS}_drive_mirror.log" 2>&1; then log "✅ drive_mirror (not a feed)"
else log "⚠ drive_mirror failed (not a feed) — see $LOG_DIR/${TS}_drive_mirror.log"; fi

# --- phone message (src/agentic/notify.py; 2026-09-29): templated from the status/eval files, never model-written ---
# Outbox + macOS banner always; Telegram once ~/.config/sgm/telegram.env exists. Not a feed.
[ ${#FAIL[@]} -eq 0 ] || /usr/bin/python3 src/agentic/notify.py fail --step "daily data layer" \
  --detail "failed feeds: ${FAIL[*]} (log $LOG_DIR/${TS}_*.log)" > /dev/null 2>&1
/usr/bin/python3 src/agentic/notify.py daily > "$LOG_DIR/${TS}_notify.log" 2>&1 && log "✅ notify (not a feed)" \
  || log "⚠ notify failed (not a feed) — see $LOG_DIR/${TS}_notify.log"

[ ${#FAIL[@]} -le 3 ] || exit 1
