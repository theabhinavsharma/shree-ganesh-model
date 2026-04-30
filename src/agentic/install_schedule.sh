#!/bin/bash
# Install / reinstall the launchd schedule for the daily pipeline.
# Run me with: bash src/agentic/install_schedule.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

LABEL="com.zoom.daily-pipeline"
SRC_PLIST="$ROOT/configs/${LABEL}.plist"
DST_PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

if [[ ! -f "$SRC_PLIST" ]]; then
    echo "ERROR: missing $SRC_PLIST" >&2
    exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"

# unload existing if present
if launchctl list | grep -q "$LABEL"; then
    echo "unloading existing $LABEL"
    launchctl unload "$DST_PLIST" 2>/dev/null || true
fi

cp "$SRC_PLIST" "$DST_PLIST"
launchctl load "$DST_PLIST"

echo ""
echo "==> installed schedule: $LABEL"
echo "    plist: $DST_PLIST"
echo ""
echo "Schedule (LOCAL time):"
echo "  Mon-Fri 06:47 IST (pre-market)"
echo "  Mon-Fri 16:23 IST (post-close)"
echo ""
echo "Manage:"
echo "  list:    launchctl list | grep $LABEL"
echo "  run now: launchctl start $LABEL"
echo "  unload:  launchctl unload $DST_PLIST"
echo ""
echo "Logs: $ROOT/logs/"
