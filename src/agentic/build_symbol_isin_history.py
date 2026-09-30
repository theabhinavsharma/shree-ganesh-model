"""Which symbol traded under which ISIN, and when (2026-09-30), from NSE's own bhavcopies (cm*bhav.csv.zip, 2015 to
mid-2020; the later sec_bhavdata files carry no ISIN). Used to re-key NSE corporate actions (listed under a company's
CURRENT symbol and ISIN) to the symbol it traded under on the ex-date, when NSE's symbolchange.csv misses the rename
(e.g. PHILIPCARB -> PCBL, whose 2018-04-19 1:5 split sat unapplied). The first 9 ISIN characters identify the issuer
(INE602A01 = Phillips Carbon Black / PCBL), the rest changes with face value.
Output: data/derived/symbol_isin_history.parquet (symbol, isin, issuer, first, last) + manifest.
"""
import io, json, sys, zipfile
from datetime import datetime
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/nse_full_history_official"
OUT = ROOT / "data/derived/symbol_isin_history.parquet"
rows = []
for d in sorted(RAW.glob("trade_date=*")):
    for f in d.glob("cm*bhav.csv.zip"):
        try:
            with zipfile.ZipFile(f) as z:
                b = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), usecols=["SYMBOL", "SERIES", "ISIN"], dtype=str)
            b = b[b["SERIES"].str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]
            b["date"] = pd.Timestamp(d.name.split("=")[1]); rows.append(b[["SYMBOL", "ISIN", "date"]])
        except Exception as x:
            print("skip", f.name, type(x).__name__)
A = pd.concat(rows, ignore_index=True)
A["SYMBOL"], A["ISIN"] = A["SYMBOL"].str.strip(), A["ISIN"].str.strip()
H = A.groupby(["SYMBOL", "ISIN"])["date"].agg(["min", "max"]).reset_index().rename(columns={"SYMBOL": "symbol", "ISIN": "isin", "min": "first", "max": "last"})
H["issuer"] = H["isin"].str[:9]
H.to_parquet(OUT, index=False)
OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(dataset="symbol_isin_history", producer="src/agentic/build_symbol_isin_history.py",
    source="NSE cm bhavcopies in data/raw/nse_full_history_official (2015 to mid-2020)", rows=len(H),
    columns=dict(symbol="NSE symbol", isin="ISIN", issuer="first 9 ISIN characters (same company across face-value changes)", first="first trading day seen", last="last trading day seen"),
    known_gaps="no ISIN in NSE's sec_bhavdata files (mid-2020 onward): renames after that rely on NSE symbolchange.csv", updated=datetime.now().isoformat(timespec="seconds")), indent=1))
print(f"symbol-ISIN history: {len(H)} rows · {H['symbol'].nunique()} symbols · {A['date'].min().date()}..{A['date'].max().date()}")
print(H[H["issuer"] == "INE602A01"].to_string(index=False))
