#!/usr/bin/env bash
# Mirror the live working set to Google Drive: My Drive/SGM backups/day_to_day (2026-09-28, user request).
# Goes through the Google Drive for desktop folder, which uploads in the background.
# Frozen, verified snapshots go to SGM backups/backup/<date> instead (src/agentic/make_drive_backup.py).
#
# Mirrored: data/ (minus *.bak* rollback copies of the price panel), src, configs, live_predictions, logs, research,
# assets, the tmp/ inputs the 15D pipeline reads, reports modified in the last 60 days, top-level scripts and docs.
# Not mirrored: .git (on GitHub), data_archive and backups (backup/ folder), old report dumps, other tmp scratch.
# Additive (no --delete): a file removed locally stays on Drive until removed there.
# Usage: bash src/agentic/sync_drive_mirror.sh [--exclude-pattern GLOB ...]   (extra excludes, e.g. files being written)
set -u
cd /Users/abhinavs./Code/Zoom
DST="$HOME/Library/CloudStorage/GoogleDrive-abhiengg.98@gmail.com/My Drive/SGM backups/day_to_day"
[ -d "$DST" ] || { echo "[$(date +%T)] Drive folder not mounted: $DST"; exit 1; }
EXTRA=()
while [ $# -gt 0 ]; do [ "$1" = "--exclude-pattern" ] && { EXTRA+=(--exclude "$2"); shift; }; shift; done
RS=(rsync -a --exclude '*.bak*' --exclude '__pycache__' --exclude '*.pyc' --exclude '.DS_Store' "${EXTRA[@]+"${EXTRA[@]}"}")
"${RS[@]}" data src configs live_predictions logs research assets "$DST/"
mkdir -p "$DST/tmp" "$DST/reports"
"${RS[@]}" tmp/from_scratch_7d_run tmp/event_flow_upgrade_7d tmp/layer_edge_2015plus_288d50pct_adjusted_fast \
  tmp/snapshot_manifest.json tmp/_check3_first.json "$DST/tmp/"
(cd reports && find . -type f -newermt "$(date -v-60d +%Y-%m-%d)" -print0 | "${RS[@]}" --from0 --files-from=- ./ "$DST/reports/")
"${RS[@]}" ./*.sh ./*.md "$DST/" 2>/dev/null
echo "[$(date +%T)] day_to_day mirror synced -> $DST ($(du -sh "$DST" 2>/dev/null | cut -f1))"
