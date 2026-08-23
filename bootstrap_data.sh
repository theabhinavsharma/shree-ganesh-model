#!/usr/bin/env bash
# BOOTSTRAP — get a fresh clone from zero data to a runnable system in minutes.
#
# The repo is ~11 MB of code; the data tree (61 GB locally) is gitignored by design.
# This script downloads the latest published data snapshot (~1 GB — the keystone
# 15-year adjusted price parquet + macro/events/corp-action panels), verifies
# checksums, then tells you exactly how to catch up to today.
#
# Requirements: curl + tar (or `gh` for private forks). No auth needed on the
# public repo — release assets are plain HTTPS downloads.
#
# Usage:  bash bootstrap_data.sh          # quick start (~1 GB essential snapshot)
#         bash bootstrap_data.sh --full   # FULL lossless archive (~all fetched+derived data,
#                                         # tens of GB — every parquet panel, raw JSON, zip)
set -euo pipefail
cd "$(dirname "$0")"

REPO="theabhinavsharma/shree-ganesh-model"

if [ "${1:-}" = "--full" ]; then
  echo "═══ FULL archive bootstrap ═══"
  TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases" \
    | grep -o '"tag_name": *"data-full-[0-9-]*"' | head -1 | grep -o 'data-full-[0-9-]*')
  [ -n "$TAG" ] || { echo "❌ no data-full-* release found"; exit 1; }
  echo "latest full archive: $TAG — downloading all volumes…"
  mkdir -p data_archive && cd data_archive
  gh release download "$TAG" --repo "$REPO" --skip-existing 2>/dev/null \
    || { echo "gh not available — listing asset URLs for curl:"; \
         curl -fsSL "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
           | grep browser_download_url | cut -d'"' -f4; exit 1; }
  cd ..
  echo "verifying sha256 against full_manifest.json…"
  python3 - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path("data_archive/full_manifest.json").read_text())
bad = 0
for v in m["volumes"]:
    p = Path("data_archive") / v["name"]
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != v["sha256"]:
        print("  ❌", v["name"]); bad += 1
print(f"{len(m['volumes'])-bad}/{len(m['volumes'])} volumes verified")
raise SystemExit(1 if bad else 0)
PY
  echo "extracting…"
  for f in data_archive/raw_*.tar.gz; do [ -e "$f" ] && tar -xzf "$f"; done
  for stem in $(ls data_archive/raw_*.part_aa 2>/dev/null | sed 's/\.part_aa//'); do
    cat "$stem".part_* > "$stem" && tar -xzf "$stem" && rm "$stem"
  done
  for f in data_archive/parquet_vol_*.tar; do [ -e "$f" ] && tar -xf "$f"; done
  [ -e data_archive/parquet_tree_sidecars.tar.gz ] && tar -xzf data_archive/parquet_tree_sidecars.tar.gz
  echo "✅ full archive restored (zstd parquets are drop-in identical content)."
  exit 0
fi

echo "═══ Shree Ganesh Model — data bootstrap ═══"

# Find the latest data-* release tag via the public GitHub API (no auth).
TAG=$(curl -fsSL "https://api.github.com/repos/$REPO/releases" \
  | grep -o '"tag_name": *"data-[0-9-]*"' | head -1 | grep -o 'data-[0-9-]*')
[ -n "$TAG" ] || { echo "❌ no data-* release found on $REPO"; exit 1; }
echo "latest snapshot release: $TAG"

ASSET_URL="https://github.com/$REPO/releases/download/$TAG/data_snapshot_${TAG#data-}.tar.gz"
echo "downloading $ASSET_URL (~1 GB)…"
curl -fL --progress-bar "$ASSET_URL" -o /tmp/sgm_snapshot.tar.gz

echo "extracting into ./data/ …"
tar -xzf /tmp/sgm_snapshot.tar.gz -C .

# Verify checksums from the embedded manifest
python3 - <<'PY'
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path("data/snapshot_manifest.json").read_text())
bad = []
for f in m["files"]:
    p = Path(f["path"])
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
        bad.append(f["path"])
print(f"snapshot {m['created']}: {len(m['files'])} files, {len(bad)} checksum failures")
sys.exit(1 if bad else 0)
PY

echo ""
echo "✅ Data bootstrapped to snapshot date. Catch up to today (minutes, not days):"
echo "   python3 src/agentic/refresh_prices.py            # bhavcopy delta since snapshot"
echo "   python3 src/agentic/refresh_announcements.py"
echo "   python3 src/agentic/fetch_forex_macro.py && python3 src/agentic/fetch_commodity_prices.py"
echo "   python3 src/agentic/fetch_global_macro.py && python3 src/agentic/build_macro_panel.py"
echo "   python3 src/agentic/fetch_industry_indicators.py && python3 src/agentic/build_news_event_features.py"
echo ""
echo "then gate + run:"
echo "   python3 src/agentic/verify_freshness.py          # must exit 0"
echo "   bash src/agentic/run_weekly_pipeline.sh --skip-fetch --dry-run"
