# Housekeeping 2026-09-28 (moved to the macOS Trash, not deleted)

Approved by the user in chat ("move the removable folders to the Trash rather than delete them").
Project size 191 GB -> 81 GB. Space is freed only when the Trash is emptied (user's call).
Restore: Finder -> Trash -> select -> right-click "Put Back" (moved with /usr/bin/trash, which records the origin).

| Moved | Size | Why it was safe |
|---|---|---|
| data_archive/stage | 47 GB | Unpacked copy of the 2026-08-23 backup. All 40 backup volumes matched their manifest sha256 and size; all 218 data files in stage are in the parquet_vol_*.tar volumes with identical sizes; the other 218 stage files were empty `.ok` progress markers. The volumes stay in data_archive/. |
| data/ml/panels | 52 GB | 90 hash-keyed ML feature-panel caches (2026-04-23..05-04), last read 2026-05-04. Used only by the ML research pipeline (src/ml/expert_pipeline.py, configs/ml_*.yaml), which no scheduled job runs and which rebuilds a missing cache. |
| tmp/* (84 entries) | ~11 GB | April runtime benchmarks and scratch runs: not named in any code under src/ or *.sh, no file modified after 2026-09-01. |

Kept in tmp/ (named in live code or recently written): from_scratch_7d_run, event_flow_upgrade_7d,
layer_edge_2015plus_288d50pct_adjusted_fast, data_snapshot_2026-08-23.tar.gz, snapshot_manifest.json,
_check3_first.json, _showcase_text.txt, vmrun.
Not touched: reports/ (16 GB, archive candidates), data_archive/*.tar (44 GB backup; better kept off this laptop).
