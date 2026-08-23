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
# Usage:  bash bootstrap_data.sh
set -euo pipefail
cd "$(dirname "$0")"

REPO="theabhinavsharma/shree-ganesh-model"

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
