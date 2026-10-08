#!/usr/bin/env bash
# Split every file > 50 MB in a Drive folder into 40 MB pieces (2026-10-08: this network cuts Drive for desktop uploads of
# files >= ~99 MB; files <= 52 MB went up). Pieces are written to a local stage, joined back and checked against the
# original's sha256, then the original is removed from the Drive folder and the pieces moved in (same disk: a rename).
# Restore: cat name.part_* > name ; shasum -a 256 name must equal parts_manifest.json.
# Usage: bash src/agentic/split_for_drive.sh "<drive folder>"
set -euo pipefail
D="$1"; ST="$HOME/sgm_split_stage_$(date +%s)"; mkdir -p "$ST"
MAN="$D/parts_manifest.json"; echo "[" > "$ST/m.json"; first=1
while IFS= read -r -d '' f; do
  rel="${f#$D/}"; mkdir -p "$ST/$(dirname "$rel")"
  sum=$(shasum -a 256 "$f" | cut -d' ' -f1); bytes=$(stat -f %z "$f")
  split -b 40m -a 3 "$f" "$ST/$rel.part_"
  chk=$(cat "$ST/$rel".part_* | shasum -a 256 | cut -d' ' -f1)
  [ "$chk" = "$sum" ] || { echo "JOIN CHECK FAILED: $rel"; exit 1; }
  n=$(ls "$ST/$rel".part_* | wc -l | tr -d ' ')
  [ $first = 1 ] || echo "," >> "$ST/m.json"; first=0
  printf '{"file": "%s", "bytes": %s, "sha256": "%s", "parts": %s}' "$rel" "$bytes" "$sum" "$n" >> "$ST/m.json"
  echo "[$(date +%T)] $rel: $n pieces, join check ok"
  rm "$f"
  (cd "$ST" && for p in "$rel".part_*; do mkdir -p "$D/$(dirname "$p")"; mv "$p" "$D/$p"; done)
done < <(find "$D" -type f -size +50M -print0)
echo "]" >> "$ST/m.json"; mv "$ST/m.json" "$MAN"; rm -rf "$ST"
echo "[$(date +%T)] done: $(grep -c '"file"' "$MAN") files split -> $MAN"
