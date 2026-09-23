#!/usr/bin/env bash
# One-off: refresh all data through today's close and re-render the daily actions.
cd /Users/abhinavs./Documents/Zoom
nohup bash src/agentic/sgm_daily.sh > logs/update_now.log 2>&1 &
echo "launched pid $! — tail -f logs/update_now.log"
