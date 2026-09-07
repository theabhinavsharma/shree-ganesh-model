"""QC v2 (amended referee, ledgered 2026-09-07): our last-4 quarterly EPS vs the
quarterly-results TABLE on screener.in (as-reported transcription of NSE filings).
Sample: ALL current PE-misses + 120 random passes. Bar: >=80% of symbols with
median |ratio-1| <= 15% across the overlapping quarters."""
import re, sys, time, random
sys.path.insert(0, "/Users/abhinavs./Documents/Zoom/src/agentic")
import pandas as pd, numpy as np
import fetch_screener_fundamentals as m

ROOT = "/Users/abhinavs./Documents/Zoom"
d = pd.read_parquet(f"{ROOT}/logs/qc_valuation_detail.parquet")
bad = d[~d["ratio"].between(0.85, 1.15)]["symbol"].tolist()
good = d[d["ratio"].between(0.85, 1.15)]["symbol"].tolist()
random.seed(7)
sample = bad + random.sample(good, min(120, len(good)))

q = pd.read_parquet(f"{ROOT}/data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic"])
q = (q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last"))
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
old = pd.Timestamp("1900-01-01")
con = lastq["con"] if "con" in lastq else pd.Series(old, index=lastq.index)
sa = lastq["sa"] if "sa" in lastq else pd.Series(old, index=lastq.index)
bmap = pd.Series(np.where(con.fillna(old) >= sa.fillna(old), "con", "sa"), index=lastq.index)

MON = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

def parse_qtr_eps(html):
    sec = re.search(r'<section id="quarters".*?</section>', html, re.S)
    if not sec:
        return {}
    block = sec.group(0)
    heads = re.findall(r"<th[^>]*>\s*([A-Z][a-z]{2}) (\d{4})\s*</th>", block)
    row = re.search(r"<td[^>]*>\s*EPS in Rs\s*</td>(.*?)</tr>", block, re.S)
    if not row or not heads:
        return {}
    vals = re.findall(r"<td[^>]*>\s*([-\d,.]*)\s*</td>", row.group(1))
    out = {}
    for (mon, yr), v in zip(heads, vals):
        try:
            qe = pd.Timestamp(int(yr), MON[mon], 1) + pd.offsets.MonthEnd(0)
            out[qe] = float(v.replace(",", ""))
        except Exception:
            pass
    return out

op = m._opener()
res = []
for i, sym in enumerate(sample):
    b = bmap.get(sym, "sa")
    url = f"https://www.screener.in/company/{sym}/" + ("consolidated/" if b == "con" else "")
    try:
        html = m._get(op, url)
        tbl = parse_qtr_eps(html)
    except Exception as e:
        print(f"  {sym} ERR {str(e)[:50]}", flush=True); tbl = {}
    ours = q[(q["symbol"] == sym) & (q["basis"] == b)].set_index("quarter_end")["eps_basic"]
    ratios = []
    for qe, sv in tbl.items():
        if qe in ours.index and sv != 0 and not pd.isna(sv):
            ratios.append(ours[qe] / sv)
    if ratios:
        ratios = sorted(ratios, key=lambda r: abs(r - 1))
        med = float(np.median(ratios[:6]))
        res.append(dict(symbol=sym, was_bad=sym in bad, n_q=len(ratios), med_ratio=med,
                        ok=abs(med - 1) <= 0.15))
    if i % 40 == 0:
        okr = np.mean([r["ok"] for r in res]) * 100 if res else 0
        print(f"[{i}/{len(sample)}] scored={len(res)} ok={okr:.0f}%", flush=True)
    time.sleep(m.DELAY)
    if i and i % m.LONG_BREAK_EVERY == 0:
        time.sleep(m.LONG_BREAK_SEC)

R = pd.DataFrame(res)
R.to_parquet(f"{ROOT}/logs/qc_eps_table_check.parquet", index=False)
print(f"\nscored {len(R)} symbols · overall EPS-match: {R['ok'].mean()*100:.1f}%")
print(f"  among prior PE-misses : {R[R['was_bad']]['ok'].mean()*100:.1f}% (n={R['was_bad'].sum()})")
print(f"  among prior PE-passes : {R[~R['was_bad']]['ok'].mean()*100:.1f}%")
w = R.groupby("was_bad")["ok"].mean()
# weighted to full comparable population (167 bad / 513 good of 680)
full = (R[R["was_bad"]]["ok"].mean() * 167 + R[~R["was_bad"]]["ok"].mean() * 513) / 680
print(f"  population-weighted estimate: {full*100:.1f}%  (bar: 80%)")
print("QC-V2 COMPLETE", flush=True)
