#!/usr/bin/env bash
# Laptop handover backup (2026-10-08, user: "move all essential files related to this compressed format mein to my google
# drive? I need to submit my laptop tomorrow"). Writes compressed bundles to a local stage on the same disk, tests them,
# then MOVES them into the Google Drive for desktop folder (a rename on the same volume: no second copy on a full disk).
# Order = importance, so a slow upload still gets the essentials up first:
#   1 code: the repo with .git history, untracked files, configs, logs, research, assets, live_predictions, skills,
#     chatgpt_ads, the tmp/ inputs the 15D pipeline reads (no data/, data_archive/, backups/, other tmp/, reports/)
#   2 keys and settings: ~/.config/sgm (Telegram), ~/.config/gspread, .env.reachy, installed launchd plists, Claude memory
#   3 live data: make_drive_backup.py (parquets re-encoded zstd and verified column by column, sha256 manifest)
#   4 reports/ (16 GB of report dumps), compressed
#   5 data_archive/ (the 2026-08-23 full archive, already compressed volumes): moved as is
# Usage: bash src/agentic/handover_backup.sh [from_step]   (log: logs/handover_backup_<date>.log)
set -euo pipefail
cd /Users/abhinavs./Code/Zoom
TAG=$(date +%F)
DRV="$HOME/Library/CloudStorage/GoogleDrive-abhiengg.98@gmail.com/My Drive/SGM backups/handover_$TAG"
ST="$HOME/sgm_handover_stage_$TAG"
FROM=${1:-1}
mkdir -p "$DRV" "$ST"
log() { echo "[$(date +%T)] $*"; }
put() { mv "$1" "$DRV/"; log "moved to Drive: $(basename "$1") ($(du -sh "$DRV/$(basename "$1")" | cut -f1))"; }

if [ "$FROM" -le 1 ]; then
  log "1 code bundle"
  # note: bsdtar --exclude=./tmp also drops the tmp/ paths named below; 2026-10-08 they went up separately as 1b_tmp_pipeline_inputs.tar.zst
  tar --exclude=./data --exclude=./data_archive --exclude=./backups --exclude=./tmp --exclude=./reports \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='.DS_Store' -cf - . \
      tmp/from_scratch_7d_run tmp/event_flow_upgrade_7d tmp/layer_edge_2015plus_288d50pct_adjusted_fast tmp/snapshot_manifest.json \
      2>/dev/null | zstd -T0 -9 -q -o "$ST/1_code_repo_git_configs_logs.tar.zst"
  zstd -t -q "$ST/1_code_repo_git_configs_logs.tar.zst" && log "1 tested ok"
  put "$ST/1_code_repo_git_configs_logs.tar.zst"
fi
if [ "$FROM" -le 2 ]; then
  log "2 keys and settings"
  tar -czf "$ST/2_keys_and_settings.tar.gz" -C "$HOME" .config/sgm .config/gspread .claude/projects/-Users-abhinavs--Code-Zoom/memory \
      $(cd "$HOME" && ls Library/LaunchAgents/com.sgm.*.plist) -C /Users/abhinavs./Code/Zoom .env.reachy
  tar -tzf "$ST/2_keys_and_settings.tar.gz" >/dev/null && log "2 tested ok"
  put "$ST/2_keys_and_settings.tar.gz"
fi
if [ "$FROM" -le 3 ]; then
  log "3 live data snapshot (verified)"
  /usr/bin/python3 src/agentic/make_drive_backup.py
  mv "backups/$TAG" "$ST/3_data_snapshot"
  put "$ST/3_data_snapshot"
fi
if [ "$FROM" -le 4 ]; then
  log "4 reports"
  tar --exclude='.DS_Store' -cf - reports | zstd -T0 -9 -q -o "$ST/4_reports.tar.zst"
  zstd -t -q "$ST/4_reports.tar.zst" && log "4 tested ok"
  put "$ST/4_reports.tar.zst"
fi
if [ "$FROM" -le 5 ]; then
  log "5 data_archive (2026-08-23 full archive)"
  mv data_archive "$ST/5_full_archive_2026-08-23"
  put "$ST/5_full_archive_2026-08-23"
fi
rmdir "$ST" 2>/dev/null || true
log "ALL DONE -> $DRV"
