"""Historical USD/INR — FRED series DEXINUS (Federal Reserve H.10, Indian rupees per US dollar, noon
buying rates in New York, daily since 1973). data/derived/usdinr_history.parquet (+ manifest).

Why (audit 2026-09-27): macro_panel.usdinr only starts 2024-02-19, so every USD order amount from
2016 to Feb-2024 was converted at a hard-coded 83.0 — a fabricated FX rate (AGENTS.md). This is the
official historical series; build_macro_panel fills usdinr gaps from it and records usdinr_source.
Note: New York noon rates, not RBI reference rates — differences are a few paise; fine for sizing orders.
"""
from __future__ import annotations

import io
import json
import time

import requests
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
OUT = ROOT / "data/derived/usdinr_history.parquet"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS"


def main() -> None:
    raw = None
    for attempt in range(2):
        try:
            r = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
            r.raise_for_status(); raw = r.text; break
        except requests.RequestException as e:
            print(f"  FRED attempt {attempt + 1} failed: {type(e).__name__}", flush=True); time.sleep(3)
    if raw is None:   # this Mac's python links LibreSSL 2.8.3, which fails FRED's TLS; the system curl does not
        import subprocess
        res = subprocess.run(["curl", "-s", "-m", "90", "--retry", "3", URL], capture_output=True, text=True)
        raw = res.stdout if res.returncode == 0 and res.stdout.startswith("observation_date") else None
        print("  fetched via system curl" if raw else f"  curl failed rc={res.returncode}", flush=True)
    if raw is None:
        raise SystemExit("FRED DEXINUS unreachable (requests + curl) — usdinr_history not updated")
    d = pd.read_csv(io.StringIO(raw))
    d.columns = ["trade_date", "usdinr"]
    d["trade_date"] = pd.to_datetime(d["trade_date"], errors="coerce")
    d["usdinr"] = pd.to_numeric(d["usdinr"], errors="coerce")          # FRED marks holidays with '.'
    d = d.dropna().sort_values("trade_date").reset_index(drop=True)
    if len(d) < 10000 or d["usdinr"].iloc[-1] < 50:
        raise SystemExit(f"FRED DEXINUS looks wrong: {len(d)} rows, last {d['usdinr'].iloc[-1]}")
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="usdinr_history", path=str(OUT.relative_to(ROOT)), rows=len(d), key=["trade_date"],
        date_range=[str(d["trade_date"].min().date()), str(d["trade_date"].max().date())],
        producer="src/agentic/fetch_usdinr_history.py", source=f"FRED DEXINUS ({URL}) — Federal Reserve H.10",
        columns=dict(trade_date="US business day", usdinr="Indian rupees per 1 US dollar (noon buying rate, New York)"),
        use="convert USD amounts to Rs with the rate on or before the event date (merge_asof backward)",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"usdinr_history: {len(d):,} rows {d['trade_date'].min().date()}..{d['trade_date'].max().date()} last {d['usdinr'].iloc[-1]}")


if __name__ == "__main__":
    main()
