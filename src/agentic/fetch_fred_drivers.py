"""Macro driver histories from FRED for the industry macro pillar (EXP-2026-09-29-industry-fundamentals).

macro_panel's commodity columns start 2020-12-29, too late for 36-month industry sensitivities before 2023. FRED holds the
official long histories: EIA daily Brent and Henry Hub gas, the Fed's USDINR / US 10y / broad dollar, and the IMF Primary
Commodity Prices (monthly averages) for metals and agricultural goods.

Point in time: a daily value is known after its date; an IMF monthly average (FRED dates it on the 1st of its month) is
treated as known at the END of the FOLLOWING month (registered: "known 1 month after their month").
Output: data/derived/macro_drivers_fred.parquet (series, date, value, freq, known_from) + manifest. Fetched with the system
curl (this Mac's python links LibreSSL, which fails FRED's TLS; see fetch_usdinr_history.py).
"""
from __future__ import annotations

import io
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "data/derived/macro_drivers_fred.parquet"
SERIES = {  # FRED id: (name, freq, meaning)
    "DCOILBRENTEU": ("brent", "D", "Brent crude, USD/bbl (EIA)"),
    "DHHNGSP": ("henry_hub_gas", "D", "Henry Hub natural gas spot, USD/MMBtu (EIA)"),
    "DEXINUS": ("usdinr", "D", "INR per USD (Fed H.10)"),
    "DGS10": ("us_10y", "D", "US 10-year Treasury yield, percent"),
    "DTWEXBGS": ("broad_dollar", "D", "Nominal broad US dollar index (Fed)"),
    "PCOPPUSDM": ("copper", "M", "IMF copper, USD/metric ton, monthly average"),
    "PALUMUSDM": ("aluminium", "M", "IMF aluminium, USD/metric ton, monthly average"),
    "PZINCUSDM": ("zinc", "M", "IMF zinc, USD/metric ton, monthly average"),
    "PNICKUSDM": ("nickel", "M", "IMF nickel, USD/metric ton, monthly average"),
    "PIORECRUSDM": ("iron_ore", "M", "IMF iron ore, USD/dry metric ton, monthly average"),
    "PSUGAISAUSDM": ("sugar", "M", "IMF sugar (ISA), US cents/lb, monthly average"),
    "PCOTTINDUSDM": ("cotton", "M", "IMF cotton (Cotlook A), US cents/lb, monthly average"),
    "PWHEAMTUSDM": ("wheat", "M", "IMF wheat, USD/metric ton, monthly average"),
}


def fetch(sid: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    r = subprocess.run(["curl", "-s", "-m", "120", "--retry", "3", url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip() or "<html" in r.stdout[:200].lower():
        raise SystemExit(f"FRED {sid} unreachable (curl rc={r.returncode})")
    df = pd.read_csv(io.StringIO(r.stdout))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")        # FRED marks missing days with "."
    return df.dropna()


def main() -> None:
    rows = []
    for sid, (name, freq, _) in SERIES.items():
        df = fetch(sid)
        df["series"], df["fred_id"], df["freq"] = name, sid, freq
        df["known_from"] = (df["date"] + pd.offsets.MonthEnd(2)) if freq == "M" else (df["date"] + pd.Timedelta(days=1))
        rows.append(df)
        print(f"{sid:14s} {name:14s} {freq} {df['date'].min().date()} .. {df['date'].max().date()} ({len(df):,})", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="macro_drivers_fred", path=str(OUT.relative_to(ROOT)), rows=len(out), producer="src/agentic/fetch_fred_drivers.py",
        source="FRED (St. Louis Fed): EIA, Federal Reserve, IMF Primary Commodity Prices",
        series={v[0]: dict(fred_id=k, freq=v[1], meaning=v[2]) for k, v in SERIES.items()},
        columns=dict(date="observation date (IMF monthly: first day of the month the average covers)", value="as published, units per series",
                     known_from="first date the value may be used: daily = date + 1 day; monthly = end of the following month"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(out):,} rows)")


if __name__ == "__main__":
    main()
