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
run news_events /usr/bin/python3 src/agentic/build_news_event_features.py

# --- daily macro + flows ---
run forex /usr/bin/python3 src/agentic/fetch_forex_macro.py
run commodity /usr/bin/python3 src/agentic/fetch_commodity_prices.py
run global_macro /usr/bin/python3 src/agentic/fetch_global_macro.py
run fii_dii /usr/bin/python3 src/agentic/fetch_fii_dii.py
run block_deals /usr/bin/python3 src/agentic/fetch_block_deals.py
run breadth /usr/bin/python3 src/agentic/fetch_market_breadth.py
run industry /usr/bin/python3 src/agentic/fetch_industry_indicators.py

# --- daily news / narrative ---
run news_rss /usr/bin/python3 src/agentic/fetch_news_rss.py
run sentiment /usr/bin/python3 src/agentic/score_sentiment.py
run global_sentiment /usr/bin/python3 src/agentic/fetch_global_macro_sentiment.py
run pib_releases /usr/bin/python3 src/agentic/fetch_pib_releases.py --start "$(date -v-7d +%Y-%m-%d)" --match-symbols

# --- weekly (Mondays): fundamentals + holdings + recos ---
if [ "$DOW" = "1" ]; then
  run fundamentals /usr/bin/python3 src/agentic/fetch_fundamentals.py
  run screener_fundamentals /usr/bin/python3 src/agentic/fetch_screener_fundamentals.py
  run amfi_mf /usr/bin/python3 src/agentic/fetch_amfi_mf_holdings.py
  run superstar /usr/bin/python3 src/agentic/fetch_superstar_holdings.py
  run broker_recos /usr/bin/python3 src/agentic/fetch_broker_recos.py
fi

# --- rebuild the panel last so everything above folds in ---
run macro_panel /usr/bin/python3 src/agentic/build_macro_panel.py

# --- loud status ---
/usr/bin/python3 - << PYEOF
import json, datetime
json.dump({"ts": "$TS", "date": str(datetime.date.today()),
           "passed": "${PASS[*]:-}".split(), "failed": "${FAIL[*]:-}".split()},
          open("logs/daily_data_layer_status.json","w"), indent=1)
PYEOF
log "═══ DONE: ${#PASS[@]} ok, ${#FAIL[@]} failed (${FAIL[*]:-none}) ═══"
[ ${#FAIL[@]} -le 3 ] || exit 1
