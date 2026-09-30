"""Google Drive backup of the live data (2026-09-28) — lossless, verified, volumed.

Reuses make_full_archive's phases with four changes:
  * OUT = backups/<date>/ (volumes + manifest to upload); STAGE = a temp dir outside the repo (zstd copies, packed then
    left for the OS to clean), so no second copy sits in the project.
  * PARQUET_TREES = data/derived, data/ml (data/ml/panels, the stale ML caches, went to the Trash on 2026-09-28).
    RAW_TREES = make_full_archive's list, those that exist.
  * The sidecar tar leaves out *.bak* files (rollback copies of the price panel kept during the September repairs:
    they are not the live data, and the verified repairs made them redundant).
  * Every re-encoded parquet is checked against its original column by column (same schema, same row count,
    ChunkedArray.equals on every column); a mismatch stops the run. make_full_archive never checked this.
Output: backups/<date>/{parquet_vol_*.tar, raw_*.tar.gz[.part_*], parquet_tree_sidecars.tar.gz, full_manifest.json,
README.md}. Restore steps are in the manifest and README.
"""
from __future__ import annotations

import json
import sys
import tarfile
import tempfile
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import make_full_archive as mfa  # noqa: E402

TAG = date.today().isoformat()
mfa.OUT = ROOT / "backups" / TAG
mfa.STAGE = Path(tempfile.gettempdir()) / f"sgm_backup_stage_{TAG}"
mfa.PARQUET_TREES = ["data/derived", "data/ml"]
mfa.RAW_TREES = [t for t in mfa.RAW_TREES if (ROOT / t).exists()]
_transcode = mfa.transcode_parquet
checked = {"files": 0, "rows": 0}


def verified_transcode(src: Path, dst: Path) -> None:
    _transcode(src, dst)
    a, b = pq.ParquetFile(src), pq.ParquetFile(dst)
    if a.schema_arrow != b.schema_arrow or a.metadata.num_rows != b.metadata.num_rows:
        raise SystemExit(f"VERIFY FAILED (schema/rows): {src}")
    for c in a.schema_arrow.names:
        if not a.read(columns=[c]).column(0).equals(b.read(columns=[c]).column(0)):
            raise SystemExit(f"VERIFY FAILED (column {c}): {src}")
    checked["files"] += 1; checked["rows"] += a.metadata.num_rows


def sidecars_no_bak() -> Path | None:
    out = mfa.OUT / "parquet_tree_sidecars.tar.gz"
    files = [p for t in mfa.PARQUET_TREES for p in sorted((ROOT / t).rglob("*"))
             if p.is_file() and p.suffix != ".parquet" and ".bak" not in p.name]
    if not files:
        return None
    with tarfile.open(out, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    mfa.log(f"sidecars: {len(files)} files → {out.name} ({out.stat().st_size / 1e6:.0f} MB)")
    return out


mfa.transcode_parquet = verified_transcode
mfa.phase_sidecars = sidecars_no_bak

if __name__ == "__main__":
    mfa.main()
    m = json.loads((mfa.OUT / "full_manifest.json").read_text())
    m.update(note="Google Drive backup of the live data: parquets re-encoded zstd and VERIFIED column-by-column against the "
                  "originals; raw trees tar.gz'd as-is; *.bak panel rollback copies excluded.",
             verified=checked, trees=dict(parquet=mfa.PARQUET_TREES, raw=mfa.RAW_TREES),
             restore={"parquet_vol_*.tar": "tar -xf vol.tar -C <repo-root>  (paths are repo-relative)",
                      "raw_*.tar.gz": "tar -xzf at repo root", "*.part_*": "cat name.part_* > name, then tar -xzf",
                      "check": "shasum -a 256 <file> must equal the manifest's sha256"})
    (mfa.OUT / "full_manifest.json").write_text(json.dumps(m, indent=1))
    (mfa.OUT / "README.md").write_text(
        f"# SGM data backup {TAG}\n\nLossless backup of data/ (live) for Google Drive. Verified: {checked['files']} parquet "
        f"files / {checked['rows']:,} rows identical after re-encoding. Restore: see full_manifest.json -> restore; check "
        "every file's sha256 against the manifest before restoring.\n")
    mfa.log(f"verified {checked['files']} parquets ({checked['rows']:,} rows)")
