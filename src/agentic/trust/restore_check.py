"""Restore check for the Google Drive handover backup (2026-10-08; user: "test the fetch from gdrive once to see all is
working ... input data same <> analyses quality and rigor same <> output same").

  fingerprint <root> <out.json>   content fingerprint of <root>/data (every *.parquet: row count + an order-sensitive hash
                                  built column by column from pandas' row hashes; every other file: sha256; *.bak* skipped)
                                  plus sha256 of <root>/src and <root>/configs, and sample prices
  compare <a.json> <b.json>       file-by-file comparison; prints what differs (missing / rows / hash)
  prices <root>                   close / open for sample stocks and dates, read through research_panel (the same code path
                                  the tests use)
The parquets in the backup were re-encoded (zstd), so their bytes differ from the originals by design; the comparison is
on content (rows and values), which is what the analyses read.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SAMPLE = {"symbols": ["BIRLACABLE", "DEEDEV", "STLTECH", "ARROWGREEN", "JNKINDIA", "KABRAEXTRU", "GANDHAR", "SUNFLAG", "ALPHAGEO", "RELIANCE"],
          "dates": ["2019-04-02", "2021-10-06", "2023-10-05", "2025-10-07", "2026-10-07"]}
MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fp_parquet(p: Path) -> dict:
    pf = pq.ParquetFile(p)
    n = pf.metadata.num_rows
    acc = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for name in pf.schema_arrow.names:
            s = pf.read(columns=[name]).column(0).to_pandas()
            try:
                h = pd.util.hash_pandas_object(s, index=False).to_numpy()
            except TypeError:                                            # nested values (lists / dicts)
                h = pd.util.hash_pandas_object(s.astype(str), index=False).to_numpy()
            acc = acc * np.uint64(1099511628211) + h.astype(np.uint64)
        tot = int((acc * (np.arange(n, dtype=np.uint64) + np.uint64(1))).sum() & MASK) if n else 0
    return dict(rows=n, cols=len(pf.schema_arrow.names), hash=f"{tot:016x}")


def fingerprint(root: Path) -> dict:
    out = {"data": {}, "code": {}}
    for p in sorted((root / "data").rglob("*")):
        if not p.is_file() or ".bak" in p.name or p.name == ".DS_Store":
            continue
        rel = str(p.relative_to(root))
        out["data"][rel] = fp_parquet(p) if p.suffix == ".parquet" else {"sha256": sha(p)}
    for d in ("src", "configs"):
        for p in sorted((root / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != ".DS_Store":
                out["code"][str(p.relative_to(root))] = sha(p)
    out["prices"] = prices(root)
    return out


def prices(root: Path) -> dict:
    sys.path.insert(0, str(root / "src/agentic"))
    import research_panel as rp
    if Path(rp.__file__).resolve().parents[2] != root.resolve():
        raise SystemExit(f"research_panel imported from {rp.__file__}, not from {root}")
    P = rp.load_panel(["open", "close"], filters=[("symbol", "in", SAMPLE["symbols"])])
    P = P[P["symbol"].isin(SAMPLE["symbols"]) & pd.to_datetime(P["trade_date"]).isin(pd.to_datetime(SAMPLE["dates"]))]
    return {f"{r.symbol}|{pd.Timestamp(r.trade_date).date()}": [round(float(r.open), 6), round(float(r.close), 6)] for r in P.itertuples()}


def compare(a: dict, b: dict) -> int:
    bad = 0
    for part in ("data", "code"):
        ka, kb = set(a[part]), set(b[part])
        miss, extra = sorted(ka - kb), sorted(kb - ka)
        diff = sorted(k for k in ka & kb if a[part][k] != b[part][k])
        same = len(ka & kb) - len(diff)
        print(f"{part}: {same} of {len(ka)} identical · differ {len(diff)} · missing in second {len(miss)} · only in second {len(extra)}")
        for k in (diff + miss + extra)[:25]:
            print(f"   {'DIFF' if k in diff else 'MISSING' if k in miss else 'EXTRA'} {k}: {a[part].get(k)} vs {b[part].get(k)}")
        bad += len(diff) + len(miss) + len(extra)
    pa, pb = a.get("prices", {}), b.get("prices", {})
    pd_ = [k for k in pa if pa[k] != pb.get(k)]
    print(f"prices: {len(pa) - len(pd_)} of {len(pa)} identical" + ("".join(f"\n   {k}: {pa[k]} vs {pb.get(k)}" for k in pd_[:10])))
    return bad + len(pd_)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fingerprint":
        r = fingerprint(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(r, indent=0))
        print(f"fingerprinted {len(r['data'])} data files, {len(r['code'])} code files, {len(r['prices'])} sample prices -> {sys.argv[3]}")
    elif cmd == "compare":
        sys.exit(1 if compare(json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text())) else 0)
    elif cmd == "prices":
        print(json.dumps(prices(Path(sys.argv[2])), indent=1))
