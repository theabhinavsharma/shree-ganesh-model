"""Insider / promoter trades from NSE's new filing system (corporates-pit-gg), 2026-05-01 onward (2026-09-30).

Plain English: on 1 May 2026 NSE moved insider-trading disclosures to a new system. The old address
(api/corporates-pit, used by refresh_announcements.py and fetch_event_history_fleet.py) has returned 0 rows every
fortnight since, without an error, so promoter-buying data went silently stale. The new address lists one iXBRL file
per filing; this fetcher lists them, downloads each file slowly (resumable), reads the tagged fields and merges them
into data/derived/pit_history.parquet with the same column names the rest of the code reads.

Source (official): https://www.nseindia.com/api/corporates-pit-gg?index=equities&from_date=DD-MM-YYYY&to_date=DD-MM-YYYY
                    -> ixbrl links on nsearchives.nseindia.com (tags in-bse-co:*). Nothing estimated; a tag that is
                    missing stays blank.
Files: data/raw/pit_gg/listing.parquet (every filing listed), data/raw/pit_gg/html/<file> (downloaded filings),
       data/derived/pit_gg.parquet (+ manifest; one row per disclosure line), data/derived/pit_history.parquet
       (old rows before 2026-05-01 + these rows, column `source`; the pre-merge file is kept once as
       pit_history_pre_gg.parquet).
Usage: fetch_pit_gg.py --backfill [--max-minutes N]   (list 2026-04-15 -> today, then parse until done or out of time)
       fetch_pit_gg.py --daily                         (last 10 days: list + parse new filings; for the daily layer)
Prints "COMPLETE" when every listed filing has been parsed (the research queue uses this).
"""
from __future__ import annotations

import argparse
import html as H
import json
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.ingest.nse.session import build_session  # noqa: E402
from src.ingest.nse.api import get_json  # noqa: E402

REF = "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading"
API = "https://www.nseindia.com/api/corporates-pit-gg?index=equities&from_date={f}&to_date={t}"
START = date(2026, 4, 15)            # overlap before the 1 May 2026 switch
RAW = ROOT / "data/raw/pit_gg"
LIST = RAW / "listing.parquet"
OUT = ROOT / "data/derived/pit_gg.parquet"
HIST = ROOT / "data/derived/pit_history.parquet"
SLEEP = 1.5
TAG = re.compile(r"<ix:(?:nonNumeric|nonFraction)\b([^>]*)>(.*?)</ix:(?:nonNumeric|nonFraction)>", re.S | re.I)
ATTR = lambda a, k: (re.search(rf"\b{k}=['\"]([^'\"]+)['\"]", a) or [None, None])[1]  # noqa: E731
FIELDS = {  # in-bse-co tag -> pit_history column
    "Symbol": "symbol", "NameOfTheCompany": "company", "CategoryOfPerson": "personCategory", "NameOfThePerson": "acqName",
    "TypeOfInstrument": "secType", "SecuritiesAcquiredOrDisposedNumberOfSecurity": "secAcq",
    "SecuritiesAcquiredOrDisposedValueOfSecurity": "secVal", "SecuritiesAcquiredOrDisposedTransactionType": "tdpTransactionType",
    "ModeOfAcquisitionOrDisposal": "acqMode", "DateOfIntimationToCompany": "intimDt",
    "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyFromDate": "acqfromDt",
    "DateOfAllotmentAdviceOrAcquisitionOfSharesOrSaleOfSharesSpecifyToDate": "acqtoDt",
    "SecuritiesHeldPriorToAcquisitionOrDisposalNumberOfSecurity": "befAcqSharesNo",
    "SecuritiesHeldPriorToAcquisitionOrDisposalPercentageOfShareholding": "befAcqSharesPer",
    "SecuritiesHeldPostAcquistionOrDisposalNumberOfSecurity": "afterAcqSharesNo",
    "SecuritiesHeldPostAcquistionOrDisposalPercentageOfShareholding": "afterAcqSharesPer",
    "ExchangeOnWhichTheTradeWasExecuted": "exchange", "DisclosureUnderRegulation": "regulation"}


def _dmy(s: str | None) -> str | None:
    """'29-09-2026' -> '29-Sep-2026' (the old file's format)."""
    try:
        return datetime.strptime(s.strip(), "%d-%m-%Y").strftime("%d-%b-%Y") if s else None
    except ValueError:
        return s


def list_filings(s, frm: date, to: date) -> pd.DataFrame:
    rows, cur = [], frm
    while cur <= to:
        end = min(cur + timedelta(days=13), to)
        j = get_json(s, API.format(f=cur.strftime("%d-%m-%Y"), t=end.strftime("%d-%m-%Y")), referer=REF)
        d = j.get("data", []) if isinstance(j, dict) else (j or [])
        print(f"  list {cur}..{end}: {len(d)}", flush=True)
        rows += d
        cur = end + timedelta(days=1)
        time.sleep(SLEEP)
    L = pd.DataFrame(rows)
    if len(L):
        L = L[L["ixbrl"].notna()].drop_duplicates("ixbrl")
    return L


def parse(text: str) -> list[dict]:
    """One dict per disclosure line (grouped by the XBRL context), plus the company-level fields."""
    by_ctx, common = {}, {}
    for attrs, val in TAG.findall(text):
        name, ctx = ATTR(attrs, "name"), ATTR(attrs, "contextRef") or ATTR(attrs, "contextref")
        if not name or not name.startswith("in-bse-co:"):
            continue
        key = FIELDS.get(name.split(":", 1)[1])
        if not key:
            continue
        v = H.unescape(re.sub(r"<[^>]+>", "", val)).strip()
        by_ctx.setdefault(ctx, {})[key] = v
    for ctx, d in list(by_ctx.items()):
        if "personCategory" not in d and "secAcq" not in d:
            common.update(d); del by_ctx[ctx]
    out = []
    for d in by_ctx.values():
        r = {**common, **d}
        for k in ("intimDt", "acqfromDt", "acqtoDt"):
            r[k] = _dmy(r.get(k))
        out.append(r)
    return out


def merge_history(G: pd.DataFrame) -> None:
    pre = HIST.with_name("pit_history_pre_gg.parquet")
    if not pre.exists():
        pd.read_parquet(HIST).to_parquet(pre, index=False)
    old = pd.read_parquet(pre)
    old["source"] = "corporates-pit"
    od = pd.to_datetime(old["date"], format="%d-%b-%Y %H:%M", errors="coerce")
    old = old[od.isna() | (od < pd.Timestamp("2026-05-01"))]
    new = G.drop(columns=["ixbrl_url", "regulation", "fetched"], errors="ignore").copy()
    for c in old.columns:
        if c not in new.columns:
            new[c] = None
    both = pd.concat([old, new[old.columns]], ignore_index=True)
    for c in both.columns:
        if both[c].dtype == object:
            both[c] = both[c].astype("string")
    both.to_parquet(HIST, index=False)
    print(f"  pit_history: {len(old)} old rows (< 2026-05-01) + {len(new)} new = {len(both)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backfill", action="store_true")
    g.add_argument("--daily", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    a = ap.parse_args()
    t0 = time.time()
    (RAW / "html").mkdir(parents=True, exist_ok=True)
    lock = RAW / ".lock"                      # the daily feed and the queue backfill never write at the same time
    while True:
        try:
            lock.mkdir(); break
        except FileExistsError:
            if time.time() - lock.stat().st_mtime > 3 * 3600:
                lock.rmdir(); continue        # stale lock from a killed run
            if time.time() - t0 > 55 * 60:
                sys.exit("another fetch_pit_gg run held the lock for 55 minutes")
            time.sleep(30)
    try:
        _main(a, t0)
    finally:
        lock.rmdir()


def _main(a, t0: float) -> None:
    s = build_session(warm=True, referer=REF)
    today = date.today()
    frm = START if a.backfill else today - timedelta(days=10)
    L = list_filings(s, frm, today)
    if LIST.exists():
        L = pd.concat([pd.read_parquet(LIST), L], ignore_index=True).drop_duplicates("ixbrl", keep="last")
    L.to_parquet(LIST, index=False)
    G = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["ixbrl_url"])
    gone_f = RAW / "not_on_nse.json"            # filings NSE lists but its archive answers 404 for: recorded, not retried forever
    gone = set(json.loads(gone_f.read_text())) if gone_f.exists() else set()
    done = set(G["ixbrl_url"]) | gone
    todo = L[~L["ixbrl"].isin(done)]
    print(f"listed {len(L)} filings · parsed before {len(done)} · to parse {len(todo)}", flush=True)
    new_rows, errors = [], 0
    for i, r in enumerate(todo.itertuples(), 1):
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print(f"  time box reached after {i - 1} filings", flush=True)
            break
        f = RAW / "html" / r.ixbrl.rsplit("/", 1)[-1]
        try:
            if not f.exists():
                resp = s.get(r.ixbrl, timeout=60)
                resp.raise_for_status()
                f.write_bytes(resp.content)
                time.sleep(SLEEP)
            for d in parse(f.read_text(encoding="utf-8", errors="ignore")):
                d.update(ixbrl_url=r.ixbrl, date=pd.to_datetime(r.broadcastDateTime, format="%d-%b-%Y %H:%M:%S").strftime("%d-%b-%Y %H:%M"),
                         symbol=d.get("symbol") or r.symbol, source="corporates-pit-gg", fetched=datetime.now().isoformat(timespec="seconds"))
                new_rows.append(d)
        except Exception as x:
            if "404" in str(x):
                gone.add(r.ixbrl); gone_f.write_text(json.dumps(sorted(gone), indent=1))
                print(f"  404 (not on NSE's archive, recorded in {gone_f.name}): {r.symbol} {r.ixbrl.rsplit('/', 1)[-1]}", flush=True)
                continue
            errors += 1
            print(f"  ERR {r.symbol} {r.ixbrl.rsplit('/', 1)[-1]}: {type(x).__name__} {str(x)[:80]}", flush=True)
        if i % 200 == 0:
            print(f"  parsed {i}/{len(todo)}", flush=True)
    if new_rows:
        G = pd.concat([G, pd.DataFrame(new_rows)], ignore_index=True)
        for c in G.columns:
            if G[c].dtype == object:
                G[c] = G[c].astype("string")
        G.to_parquet(OUT, index=False)
    remaining = int((~L["ixbrl"].isin(set(G["ixbrl_url"]) | gone)).sum())
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="pit_gg", path=str(OUT.relative_to(ROOT)), rows=len(G), filings_listed=len(L), filings_unparsed=remaining,
        source=API.replace("{f}", "DD-MM-YYYY").replace("{t}", "DD-MM-YYYY") + " -> ixbrl (in-bse-co tags)",
        columns={v: f"in-bse-co:{k}" for k, v in FIELDS.items()} | dict(date="NSE broadcast time (IST), '%d-%b-%Y %H:%M' like pit_history",
                 source="corporates-pit-gg", ixbrl_url="the filing"),
        units=dict(secAcq="number of securities", secVal="rupees as filed (text, commas kept)", befAcqSharesPer="percent as filed"),
        known_gaps="filings NSE lists without an ixbrl link are skipped; tags missing in a filing stay blank; "
                   f"{len(gone)} listed filings return 404 on NSE's archive (data/raw/pit_gg/not_on_nse.json)",
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    if len(G):
        merge_history(G.rename(columns={"ixbrl_url": "ixbrl_url"}))
    print(f"parsed {len(new_rows)} disclosure lines this run · errors {errors} · unparsed filings left {remaining} · {time.time() - t0:.0f}s")
    if remaining == 0 and errors == 0:
        print("COMPLETE")


if __name__ == "__main__":
    main()
