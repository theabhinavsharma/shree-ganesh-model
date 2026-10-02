"""GDELT coverage check (2026-10-02): could GDELT give us media history (2017+) for the stocks V3 trades?

Media layer, step 2 of the news plan: our own RSS collection starts 2026-04-28, so any historical test of media signals
needs an archive. GDELT (api.gdeltproject.org, DOC 2.0, open) counts articles per day by query since 2017.
Sample: liquid (core-band) stocks on 2023-06-30 by market cap: 30 small (Rs 1,000-5,000 cr), 15 mid (5,000-20,000),
15 large (>= 20,000); fixed seed. Query = the company name in quotes (Limited/Ltd/India suffixes removed, first 3 words),
sourcecountry:IN. One request every 20 seconds, 60-second back-off when GDELT says slow down (shared proxy address).
Output: logs/news/gdelt_coverage/coverage.csv (+ manifest): articles per company per year. Feasibility only; no signal.
Limits: name matching misses articles that use another name and counts unrelated uses of generic names.
"""
import json
import re
import subprocess
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/news/gdelt_coverage"; OUT.mkdir(parents=True, exist_ok=True)
M = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"])
M = M[M["trade_date"] == pd.Timestamp("2023-06-30")]
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date", "close", "avg_traded_value_20d"])
px = px[px["trade_date"] == pd.Timestamp("2023-06-30")]
core = set(px.loc[(px["avg_traded_value_20d"] >= 5e7) & (px["close"] > 50), "symbol"])
M = M[M["symbol"].isin(core)]
C = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "companyName"]).dropna().drop_duplicates("symbol", keep="last").set_index("symbol")["companyName"]
band = lambda v: "small" if v < 5000 else ("mid" if v < 20000 else "large")  # noqa: E731
M = M[M["mcap_cr"] >= 1000].assign(band=lambda x: x["mcap_cr"].map(band))
S = pd.concat([M[M.band == "small"].sample(30, random_state=7), M[M.band == "mid"].sample(15, random_state=7), M[M.band == "large"].sample(15, random_state=7)])


def clean(n: str) -> str:
    n = re.sub(r"\b(limited|ltd\.?|\(india\)|india|pvt|private|corporation|corp\.?|company|co\.)\b", " ", str(n), flags=re.I)
    return " ".join(n.replace("&", "and").split()[:3])


rows = []
T0, BUDGET_MIN = time.time(), 30          # hard stop: a feasibility check must not run for hours (2026-10-02, ran 2 h unnoticed)
for r in S.itertuples():
    done = len(rows); bad = sum(x["status"] != "OK" for x in rows)
    if (done >= 5 and bad / done > 0.5) or (time.time() - T0) / 60 > BUDGET_MIN:
        print(f"STOPPED EARLY: {bad}/{done} failed, {(time.time() - T0) / 60:.0f} min elapsed (budget {BUDGET_MIN} min)", flush=True)
        break
    name = clean(C.get(r.symbol, r.symbol))
    q = urllib.parse.quote(f'"{name}" sourcecountry:IN')
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=timelinevolraw&format=json&startdatetime=20170101000000&enddatetime=20251231235959"
    for attempt in range(4):                     # GDELT throttles shared addresses (this Mac sits behind a company proxy)
        txt = subprocess.run(["/usr/bin/curl", "-s", "--max-time", "60", url], capture_output=True, text=True).stdout
        if not txt.startswith("Please limit"):
            break
        time.sleep(60)
    try:
        data = json.loads(txt)["timeline"][0]["data"]
        s = pd.Series({pd.Timestamp(x["date"][:8]): x["value"] for x in data})
        by = s.groupby(s.index.year).sum().to_dict(); ok = "OK"
    except Exception:
        by, ok = {}, f"ERR {txt[:60]!r}"
    rows.append(dict(symbol=r.symbol, name=name, band=r.band, mcap_cr=round(r.mcap_cr), status=ok, **{str(y): int(by.get(y, 0)) for y in range(2017, 2026)}))
    print(f"{r.symbol:14s} {r.band:5s} {name:32s} {ok[:3]} " + " ".join(f"{y % 100}:{by.get(y, 0)}" for y in range(2017, 2026)), flush=True)
    time.sleep(20)
D = pd.DataFrame(rows); D.to_csv(OUT / "coverage.csv", index=False)
(OUT / "coverage.csv.manifest.json").write_text(json.dumps(dict(dataset="coverage.csv", producer="src/agentic/explore_gdelt_coverage.py",
    definitions=__doc__, units={"2017..2025": "GDELT article count for the year, Indian sources, exact company-name phrase"},
    updated=datetime.now().isoformat(timespec="seconds")), indent=1))
yrs = [str(y) for y in range(2017, 2026)]
print("\nshare of companies with >= 12 articles in the year (about one a month), by size band:")
print(D[D.status == "OK"].groupby("band")[yrs].agg(lambda v: (v >= 12).mean()).round(2).to_string())
print("\nmedian articles per company-year:"); print(D[D.status == "OK"].groupby("band")[yrs].median().to_string())
