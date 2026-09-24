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
--retry: second pass over HTTP_404 / NO_CLASSIFICATION: (1) NSE rename chain (security_master
renamed_to, and symbols renamed INTO this one), (2) screener search API by registered company
name (announcements_historical.sm_name, CA store, NSE symbolchange). Status OK_VIA_RENAME:<sym>
| OK_VIA_SEARCH:<url>; the matched page is recorded, never assumed.
"""
from __future__ import annotations

import json
import re
import sys
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


def _get(s, url, **kw):
    for _ in range(3):
        try:
            r = s.get(url, headers=H, timeout=30, **kw)
        except requests.RequestException:
            time.sleep(5); continue
        if r.status_code == 429:
            time.sleep(60); continue
        return r
    return None


def retry() -> None:
    d = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates("symbol", keep="last")
    bad = d[~d["status"].str.startswith("OK")]["symbol"].tolist()
    m = pd.read_parquet(MASTER).set_index("symbol")
    sc = pd.read_csv(ROOT / "data/raw/nse_reference/symbolchange.csv", header=None, names=["name", "old", "new", "date"], skipinitialspace=True)
    sc = sc.apply(lambda c: c.str.strip() if c.dtype == object else c)
    names = {}
    for src in (sc.set_index("old")["name"],
                pd.read_parquet(ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet",
                                columns=["symbol", "company_name"]).dropna().drop_duplicates("symbol").set_index("symbol")["company_name"],
                pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "sm_name"])
                  .dropna().drop_duplicates("symbol").set_index("symbol")["sm_name"]):
        for k, v in src.items():
            names[k] = v
    ok = d[d["status"] == "OK"].set_index("symbol")
    s = requests.Session()
    print(f"retry pass: {len(bad)} symbols", flush=True)
    with CKPT.open("a") as fh:
        for i, sym in enumerate(bad):
            rec = dict(symbol=sym, fetched_at=datetime.now().isoformat(timespec="seconds"))
            chain = set(sc.loc[sc["old"] == sym, "new"]) | set(sc.loc[sc["new"] == sym, "old"])
            hit = None
            for c in chain:
                if c in ok.index:
                    hit = ok.loc[c]; rec.update({k: hit[k] for k in ("broad_sector", "sector", "broad_industry", "industry")})
                    rec.update(status=f"OK_VIA_RENAME:{c}", url=hit["url"]); break
            if hit is None:
                for c in chain:
                    r = _get(s, f"https://www.screener.in/company/{requests.utils.quote(c, safe='')}/"); time.sleep(DELAY)
                    if r is not None and r.status_code == 200 and parse(r.text).get("industry"):
                        rec.update(parse(r.text)); rec.update(status=f"OK_VIA_RENAME:{c}", url=r.url); hit = True; break
            if hit is None and names.get(sym):
                q = str(names[sym]).replace("Limited", "").replace("Ltd.", "").replace("Ltd", "").strip()
                r = _get(s, "https://www.screener.in/api/company/search/", params={"q": q}); time.sleep(DELAY)
                res = r.json() if r is not None and r.status_code == 200 else []
                if res:
                    u = "https://www.screener.in" + res[0]["url"]
                    r2 = _get(s, u); time.sleep(DELAY)
                    if r2 is not None and r2.status_code == 200 and parse(r2.text).get("industry"):
                        rec.update(parse(r2.text)); rec.update(status=f"OK_VIA_SEARCH:{res[0]['name']}", url=u); hit = True
            if hit is None:
                rec.update(status="UNRESOLVED", company_name=names.get(sym))
            fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
            if i % 50 == 0:
                print(f"  retry {i}/{len(bad)} {sym} {rec['status']}", flush=True)
    finalize()


def finalize() -> None:
    d = pd.DataFrame([json.loads(l) for l in CKPT.read_text().splitlines() if l.strip()]).drop_duplicates("symbol", keep="last")
    d["ok"] = d["status"].str.startswith("OK")
    d.to_parquet(OUT, index=False)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="screener_industry", path=str(OUT.relative_to(ROOT)), rows=len(d), key=["symbol"],
        producer="src/agentic/fetch_screener_industry.py", source="screener.in company page breadcrumb (NSE 4-level industry classification); NOT an NSE feed",
        columns=dict(broad_sector="level 1", sector="level 2", broad_industry="level 3", industry="level 4 (comparable to NSE smIndustry after crosswalk)",
                     status="OK | OK_VIA_RENAME:<sym> | OK_VIA_SEARCH:<name> | NO_CLASSIFICATION | HTTP_<code> | UNRESOLVED",
                     ok="status starts with OK", url="page the labels came from", company_name="registered name when unresolved"),
        status_counts=d["status"].str.replace(r":.*", "", regex=True).value_counts().to_dict(),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print("SCREENER INDUSTRY COMPLETE", d["status"].str.replace(r":.*", "", regex=True).value_counts().to_dict(), flush=True)


def main() -> None:
    m = pd.read_parquet(MASTER)
    eq = m[~m["is_fund_unit"]]
    todo = list(eq.loc[eq["industry_source"] == "unmapped", "symbol"])
    mapped = eq[eq["industry_source"] == "nse_smIndustry"]
    # --all-mapped: every NSE-labelled equity too -> full crosswalk + a single-taxonomy (NSE 4-level) map
    todo += list(mapped["symbol"] if "--all-mapped" in sys.argv else mapped.sample(250, random_state=7)["symbol"])
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
    retry()


if __name__ == "__main__":
    retry() if "--retry" in sys.argv else main()
