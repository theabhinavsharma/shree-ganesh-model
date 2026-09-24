"""Industry classification for symbols NSE no longer labels — screener.in company pages.

NSE's smIndustry exists for only ~1,213 of 3,378 equities in the panel (announcements
history); the quote API is 403-blocked (2026-08-30), results-calendar 'industry' is '-'
on every row, BSE's API refuses automated access. screener.in (already the repo's
fundamentals fallback) shows NSE's 4-level classification on each company page:
Broad Sector > Sector > Broad Industry > Industry.
Input : security_master equities with industry_source == 'unmapped'
        + 250 random NSE-mapped equities (crosswalk validation sample).
Output: data/derived/screener_industry.jsonl (checkpoint, resume-safe) ->
        data/derived/screener_industry.parquet (+ manifest) on completion.
Status per symbol: OK | NO_CLASSIFICATION (page without breadcrumb; typical for delisted)
| HTTP_<code>. Nothing is inferred when a page lacks the fields.
Rate: ~1.6 s/request, 60 s back-off on 429.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/Users/abhinavs./Documents/Zoom")
MASTER = ROOT / "data/derived/security_master.parquet"
CKPT = ROOT / "data/derived/screener_industry.jsonl"
OUT = ROOT / "data/derived/screener_industry.parquet"
H = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
FIELDS = ("Broad Sector", "Sector", "Broad Industry", "Industry")
DELAY = 1.6


def parse(html: str) -> dict:
    got = dict(re.findall(r'title="(Broad Sector|Sector|Broad Industry|Industry)"[^>]*>\s*([^<]+?)\s*<', html))
    return {k.lower().replace(" ", "_"): got.get(k) for k in FIELDS}


def main() -> None:
    m = pd.read_parquet(MASTER)
    eq = m[~m["is_fund_unit"]]
    todo = list(eq.loc[eq["industry_source"] == "unmapped", "symbol"])
    todo += list(eq[eq["industry_source"] == "nse_smIndustry"].sample(250, random_state=7)["symbol"])
    done = set()
    if CKPT.exists():
        done = {json.loads(l)["symbol"] for l in CKPT.read_text().splitlines() if l.strip()}
    todo = [s for s in todo if s not in done]
    print(f"{len(todo)} to fetch ({len(done)} already checkpointed)", flush=True)
    s = requests.Session()
    with CKPT.open("a") as fh:
        for i, sym in enumerate(todo):
            url = f"https://www.screener.in/company/{requests.utils.quote(sym, safe='')}/"
            for attempt in range(3):
                try:
                    r = s.get(url, headers=H, timeout=30)
                except requests.RequestException as e:
                    r = None; err = type(e).__name__
                if r is not None and r.status_code == 429:
                    time.sleep(60); continue
                break
            rec = dict(symbol=sym, fetched_at=datetime.now().isoformat(timespec="seconds"), url=url)
            if r is None:
                rec["status"] = f"ERR_{err}"
            elif r.status_code != 200:
                rec["status"] = f"HTTP_{r.status_code}"
            else:
                p = parse(r.text); rec.update(p)
                rec["status"] = "OK" if p.get("industry") else "NO_CLASSIFICATION"
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {sym} {rec['status']}", flush=True)
            time.sleep(DELAY)
    d = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates("symbol", keep="last")
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="screener_industry", path=str(OUT.relative_to(ROOT)), rows=len(d), key=["symbol"],
        producer="src/agentic/fetch_screener_industry.py", source="screener.in company page breadcrumb (NSE 4-level industry classification); NOT an NSE feed",
        columns=dict(broad_sector="level 1", sector="level 2", broad_industry="level 3", industry="level 4 (comparable to NSE smIndustry after crosswalk)",
                     status="OK | NO_CLASSIFICATION | HTTP_<code> | ERR_<type>", url="page fetched", fetched_at="local time"),
        status_counts=d["status"].value_counts().to_dict(), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("SCREENER INDUSTRY COMPLETE", d["status"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    main()
