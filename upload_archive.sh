#!/usr/bin/env bash
# Upload every archive volume to the GitHub release, resumably.
# Skips assets already on the release; safe to re-run any number of times.
set -euo pipefail
cd "$(dirname "$0")"

TAG="${1:-data-full-$(date +%Y-%m-%d)}"
DIR="data_archive"

[ -f "$DIR/_ALL_DONE" ] || { echo "❌ $DIR/_ALL_DONE missing — run make_full_archive.py first"; exit 1; }

# Create release if absent
gh release view "$TAG" > /dev/null 2>&1 || gh release create "$TAG" \
  --title "Full lossless data archive ($(date +%Y-%m-%d))" \
  --notes "Complete fetched+derived dataset: 15-yr NSE prices, events, shareholding, fundamentals, macro, ML feature panels. Lossless (parquet→zstd re-encode, raw trees tar.gz). sha256 manifest: full_manifest.json. Restore: bash bootstrap_data.sh --full. NSE data-usage terms apply."

EXISTING=$(gh release view "$TAG" --json assets --jq '.assets[].name')

for f in "$DIR"/parquet_vol_*.tar "$DIR"/raw_*.tar.gz "$DIR"/raw_*.part_* "$DIR"/parquet_tree_sidecars.tar.gz "$DIR"/full_manifest.json; do
  [ -e "$f" ] || continue
  base=$(basename "$f")
  if echo "$EXISTING" | grep -qx "$base"; then
    echo "skip (already uploaded): $base"
    continue
  fi
  echo "uploading: $base ($(du -h "$f" | cut -f1)) …"
  gh release upload "$TAG" "$f" --clobber
done

echo "✅ all volumes uploaded to release $TAG"
gh release view "$TAG" --json assets --jq '.assets | length' | xargs echo "assets on release:"
