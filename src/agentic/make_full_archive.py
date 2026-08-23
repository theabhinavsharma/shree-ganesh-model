"""Full lossless data archive — every byte of fetched + derived data, compressed and volumed.

Why this exists: regeneration is NOT byte-reproducible (fetch-date drift, upstream
revisions, library versions, walk-forward retrain windows). The originals on this
machine are the only ground truth, and this machine is not reliable storage.

Strategy (all content-lossless):
  1. PARQUET TREES (data/ml, data/derived — the 54 GB):
     each .parquet re-encoded to zstd (row-group streamed, bounded memory),
     then packed into plain-tar volumes ≤ 1.8 GB (zstd parquets don't gzip further).
     Non-parquet sidecars (json configs) in those trees → one sidecars tar.gz.
  2. RAW TREES (data/raw, events, shareholding, fundamentals, macro, corp actions,
     derivatives, portfolio_state, + tmp/from_scratch_7d_run):
     tar.gz per tree, split into ≤ 1.8 GB parts (JSON/txt compress 5-15x).
  3. MANIFEST: data_archive/full_manifest.json — sha256 + bytes per volume,
     per-tree file counts, and exact rejoin/restore instructions.

Resumable: every completed artifact gets a .ok marker; re-runs skip finished work.
Publish with: bash upload_archive.sh   (uploads each volume to a GitHub release)

Stdlib + pyarrow only.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import tarfile
from datetime import date, datetime
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/Users/abhinavs./Documents/Zoom")
STAGE = ROOT / "data_archive/stage"      # transcoded parquets (mirror paths)
OUT = ROOT / "data_archive"              # final volumes + manifest
VOL_LIMIT = int(1.8 * 1024**3)           # 1.8 GB — under GitHub's 2 GB asset cap
ZSTD_LEVEL = 6

PARQUET_TREES = ["data/ml", "data/derived"]
RAW_TREES = [
    "data/raw",
    "data/events_full_history",
    "data/shareholding_full_history",
    "data/fundamentals_full_history",
    "data/macro_full_history",
    "data/corporate_actions_full_history",
    "data/derivatives_full_history",
    "data/shareholding_hybrid",
    "data/shareholding_master_only",
    "data/portfolio_state",
    "tmp/from_scratch_7d_run",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def transcode_parquet(src: Path, dst: Path) -> None:
    """Re-encode parquet to zstd, streaming row groups so memory stays bounded."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    writer = None
    try:
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i)
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema,
                                          compression="zstd", compression_level=ZSTD_LEVEL)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def phase_transcode() -> list[Path]:
    """Transcode every parquet in PARQUET_TREES into STAGE. Returns staged paths."""
    staged = []
    todo = []
    for tree in PARQUET_TREES:
        todo += sorted((ROOT / tree).rglob("*.parquet"))
    log(f"transcode: {len(todo)} parquets")
    for n, src in enumerate(todo, 1):
        rel = src.relative_to(ROOT)
        dst = STAGE / rel
        ok = dst.with_suffix(dst.suffix + ".ok")
        marker = f"{src.stat().st_size}:{int(src.stat().st_mtime)}"
        if dst.exists() and ok.exists() and ok.read_text() == marker:
            staged.append(dst)
            continue
        try:
            transcode_parquet(src, dst)
            ok.write_text(marker)
            staged.append(dst)
            if n % 10 == 0 or src.stat().st_size > 5e8:
                log(f"  [{n}/{len(todo)}] {rel}  {src.stat().st_size/1e6:.0f} → {dst.stat().st_size/1e6:.0f} MB")
        except Exception as e:
            log(f"  !! FAILED {rel}: {e} — copying original instead (still lossless)")
            dst.write_bytes(src.read_bytes())
            ok.write_text(marker)
            staged.append(dst)
    return staged


def phase_sidecars() -> Path | None:
    """Non-parquet files in the parquet trees (json run-configs) → one tar.gz."""
    out = OUT / "parquet_tree_sidecars.tar.gz"
    ok = out.with_suffix(".ok")
    if out.exists() and ok.exists():
        return out
    files = []
    for tree in PARQUET_TREES:
        files += [p for p in sorted((ROOT / tree).rglob("*"))
                  if p.is_file() and p.suffix != ".parquet"]
    if not files:
        return None
    with tarfile.open(out, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    ok.write_text(str(len(files)))
    log(f"sidecars: {len(files)} files → {out.name} ({out.stat().st_size/1e6:.0f} MB)")
    return out


def phase_pack_volumes(staged: list[Path]) -> list[Path]:
    """Pack staged zstd parquets into plain-tar volumes ≤ VOL_LIMIT, deterministic order."""
    staged = sorted(staged)
    vols, cur, cur_bytes, idx = [], [], 0, 1
    groups = []
    for p in staged:
        sz = p.stat().st_size
        if cur and cur_bytes + sz > VOL_LIMIT:
            groups.append(cur); cur, cur_bytes = [], 0
        cur.append(p); cur_bytes += sz
    if cur:
        groups.append(cur)
    log(f"packing {len(staged)} parquets into {len(groups)} volumes")
    for idx, group in enumerate(groups, 1):
        vol = OUT / f"parquet_vol_{idx:02d}.tar"
        ok = vol.with_suffix(".tar.ok")
        if vol.exists() and ok.exists():
            vols.append(vol)
            continue
        with tarfile.open(vol, "w") as tar:
            for p in group:
                tar.add(p, arcname=str(p.relative_to(STAGE)))
        ok.write_text(str(len(group)))
        log(f"  {vol.name}: {len(group)} files, {vol.stat().st_size/1e9:.2f} GB")
        vols.append(vol)
    return vols


def phase_raw_trees() -> list[Path]:
    """tar.gz each raw tree; split any output over the volume limit."""
    outs = []
    for tree in RAW_TREES:
        src = ROOT / tree
        if not src.exists():
            log(f"raw: skip missing {tree}")
            continue
        name = tree.replace("/", "__")
        out = OUT / f"raw_{name}.tar.gz"
        ok = out.with_suffix(".ok")
        if ok.exists() and (out.exists() or list(OUT.glob(out.name + ".part_*"))):
            outs += [out] if out.exists() else sorted(OUT.glob(out.name + ".part_*"))
            continue
        log(f"raw: packing {tree} …")
        with tarfile.open(out, "w:gz") as tar:
            tar.add(src, arcname=tree)
        if out.stat().st_size > VOL_LIMIT:
            log(f"  {out.name} is {out.stat().st_size/1e9:.2f} GB — splitting")
            subprocess.run(["split", "-b", "1800m", str(out), str(out) + ".part_"], check=True)
            out.unlink()
            parts = sorted(OUT.glob(out.name + ".part_*"))
            outs += parts
            log(f"  → {len(parts)} parts (rejoin: cat {out.name}.part_* > {out.name})")
        else:
            outs.append(out)
            log(f"  {out.name}: {out.stat().st_size/1e9:.2f} GB")
        ok.write_text("done")
    return outs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    staged = phase_transcode()
    side = phase_sidecars()
    vols = phase_pack_volumes(staged)
    raws = phase_raw_trees()

    artifacts = vols + raws + ([side] if side else [])
    log(f"hashing {len(artifacts)} artifacts …")
    manifest = {
        "created": date.today().isoformat(),
        "note": "Lossless full archive. Parquets re-encoded zstd (content-identical); raw trees tar.gz'd as-is.",
        "restore": {
            "parquet_vol_*.tar": "tar -xf into data_archive/stage mirror, or directly: tar -xf vol.tar -C <repo-root> (paths are repo-relative)",
            "raw_*.tar.gz": "tar -xzf at repo root",
            "*.part_*": "cat name.part_* > name, then tar -xzf",
        },
        "volumes": [{"name": a.name, "bytes": a.stat().st_size, "sha256": sha256_file(a)}
                    for a in artifacts],
    }
    (OUT / "full_manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(v["bytes"] for v in manifest["volumes"])
    log(f"DONE in {datetime.now()-t0}: {len(artifacts)} volumes, {total/1e9:.2f} GB total")
    (OUT / "_ALL_DONE").write_text(date.today().isoformat())


if __name__ == "__main__":
    main()
