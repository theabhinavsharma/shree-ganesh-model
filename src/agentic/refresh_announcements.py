"""
Incrementally extend the corp_announcements parquet by fetching the last 14 days
from NSE on every run. Dedupes on (symbol, dt, desc).
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.ingest.nse.session import build_session
from src.ingest.nse.api import get_json

OUT = Path("tmp/from_scratch_7d_run/alt/corp_announcements.parquet")
PIT_OUT = Path("tmp/from_scratch_7d_run/alt/insider_trading_pit.parquet")
ANN_REF = "https://www.nseindia.com/companies-listing/corporate-filings-announcements"
PIT_REF = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"


def fetch_window(api: str, ref: str, frm: date, to: date) -> list[dict]:
    s = build_session(warm=True, referer=ref)
    rows = []
    cur = frm
    while cur <= to:
        chunk_end = min(cur + timedelta(days=14), to)
        f = cur.strftime("%d-%m-%Y")
        t = chunk_end.strftime("%d-%m-%Y")
        url = f"https://www.nseindia.com/api/{api}?index=equities&from_date={f}&to_date={t}"
        try:
            j = get_json(s, url, referer=ref)
            data = j if isinstance(j, list) else j.get("data", [])
            print(f"  {api} {cur}..{chunk_end}: {len(data)}")
            rows.extend(data)
        except Exception as e:
            print(f"  {api} {cur}..{chunk_end} ERR {str(e)[:120]}")
        cur = chunk_end + timedelta(days=1)
    return rows


def main() -> None:
    today = date.today()
    start = today - timedelta(days=14)

    # announcements
    print("== refresh announcements ==")
    new_ann = fetch_window("corporate-announcements", ANN_REF, start, today)
    if new_ann:
        df_new = pd.DataFrame(new_ann)
        if "sort_date" in df_new.columns:
            df_new["ann_ts"] = pd.to_datetime(df_new["sort_date"], errors="coerce")
        if OUT.exists():
            old = pd.read_parquet(OUT)
            before = len(old)
            merged = pd.concat([old, df_new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["symbol", "dt", "desc"], keep="first") if "dt" in merged.columns else merged
            merged.to_parquet(OUT, index=False)
            print(f"  ann: {before} → {len(merged)} (delta {len(merged)-before})")
        else:
            df_new.to_parquet(OUT, index=False)
            print(f"  ann fresh: {len(df_new)}")

    # insider
    print("\n== refresh insider trading ==")
    new_pit = fetch_window("corporates-pit", PIT_REF, start, today)
    if new_pit:
        df_new = pd.DataFrame(new_pit)
        # add parsed columns to match prior schema
        for col in ["buyValue", "sellValue", "buyQuantity", "sellquantity",
                    "befAcqSharesPer", "afterAcqSharesPer", "secAcq"]:
            if col in df_new.columns:
                df_new[col] = pd.to_numeric(df_new[col].astype(str).str.replace(",", ""), errors="coerce")
        if {"afterAcqSharesPer", "befAcqSharesPer"} <= set(df_new.columns):
            df_new["delta_pct"] = df_new["afterAcqSharesPer"] - df_new["befAcqSharesPer"]
        if PIT_OUT.exists():
            old = pd.read_parquet(PIT_OUT)
            before = len(old)
            # align columns (old has more derived cols)
            for c in old.columns:
                if c not in df_new.columns:
                    df_new[c] = pd.NA
            df_new = df_new[old.columns]
            merged = pd.concat([old, df_new], ignore_index=True)
            key_cols = [c for c in ["symbol", "intimDt", "secAcq", "buyQuantity", "sellquantity"] if c in merged.columns]
            merged = merged.drop_duplicates(subset=key_cols, keep="first")
            merged.to_parquet(PIT_OUT, index=False)
            print(f"  pit: {before} → {len(merged)} (delta {len(merged)-before})")
        else:
            df_new.to_parquet(PIT_OUT, index=False)


if __name__ == "__main__":
    main()
