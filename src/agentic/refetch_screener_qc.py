"""One-off: fresh screener PE for the QC symbol set (reference for the valuation
QC gate — the Apr/Aug snapshot has ~5% internally inconsistent rows)."""
import sys, time
sys.path.insert(0, "/Users/abhinavs./Documents/Zoom/src/agentic")
import pandas as pd
import fetch_screener_fundamentals as m

syms = pd.read_csv("/Users/abhinavs./Documents/Zoom/logs/qc_symbols.csv")["symbol"].tolist()
opener = m._opener()
rows = []
for i, s in enumerate(syms):
    try:
        d = m.fetch_one(opener, s)
        if d:
            rows.append(d)
    except Exception as e:
        print(f"  {s} ERR {str(e)[:60]}", flush=True)
    if i % 50 == 0:
        print(f"[{i}/{len(syms)}] ok={len(rows)}", flush=True)
    time.sleep(m.DELAY)
    if i and i % m.LONG_BREAK_EVERY == 0:
        time.sleep(m.LONG_BREAK_SEC)
df = pd.DataFrame(rows)
df.to_parquet("/Users/abhinavs./Documents/Zoom/data/derived/screener_qc_fresh.parquet", index=False)
print(f"fresh screener: {len(df)} rows -> screener_qc_fresh.parquet", flush=True)
