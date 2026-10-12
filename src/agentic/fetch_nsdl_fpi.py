"""Fetch the daily FPI (foreign portfolio investor) investment history from NSDL's FPI Monitor (official depository data).

Source: https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx ("Daily Trends in FPI Investments up to <date>"). One request
returns every reporting date of that month up to the chosen date, so one request per month. NSDL compiles these from
custodians' reports of CONFIRMED trades "on and up to the previous trading day(s)"; a reporting date's row is public on
that date, and this table stores known_from = the next calendar day (conservative, no look-ahead).
(NSE's own provisional FII/DII figures, data/derived/fii_dii_flows.parquet, only cover the last ~30 days NSE serves.)

Output data/derived/nsdl_fpi_daily.parquet, one row per reporting date x asset class x route:
  reporting_date, known_from, asset (Equity, Debt-General Limit, Debt-VRR, Debt-FAR, Hybrid, Mutual Funds, AIFs, Total),
  route (Stock Exchange, Primary market & others, Sub-total, ...), gross_purchases_cr, gross_sales_cr, net_cr (Rs crore,
  negative = net selling), net_usd_mn, usdinr (NSDL's conversion rate), source_url, archive_month, fetched_at.
Raw pages are kept in data/raw/nsdl_fpi/fpi_daily_<YYYY-MM>.html (completed months are fetched once; the current and the
previous month are fetched again on every run because NSDL revises them).
Network: system curl (macOS keychain trust; certificate verification on), browser headers (the site resets bare clients),
2 s between requests, time budget 20 min, stops when more than half of the first 5+ requests fail.
Usage: /usr/bin/python3 src/agentic/fetch_nsdl_fpi.py [--start 2015-01] [--refetch]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
RAW = ROOT / "data/raw/nsdl_fpi"
OUT = ROOT / "data/derived/nsdl_fpi_daily.parquet"
URL = "https://www.fpi.nsdl.co.in/web/Reports/Archive.aspx"
HDR = ["-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
       "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "-H", "Accept-Language: en-US,en;q=0.9"]
BUDGET_S, PAUSE_S = 20 * 60, 2.0
COLS = ["reporting_date", "asset", "route", "gross_purchases_cr", "gross_sales_cr", "net_cr", "net_usd_mn", "usdinr"]


def _curl(args: list[str], out: Path) -> str:
    r = subprocess.run(["curl", "-s", "-m", "60", "--http1.1", *HDR, *args, "-o", str(out), "-w", "%{http_code}"], capture_output=True, text=True)
    return r.stdout.strip()


def _form(cookies: Path) -> dict | None:
    tmp = RAW / "_form.html"
    if _curl(["-c", str(cookies), URL], tmp) != "200":
        return None
    h = tmp.read_text(encoding="utf-8", errors="ignore")
    hid = dict(re.findall(r'<input type="hidden" name="([^"]+)" id="[^"]+" value="([^"]*)"', h))
    return {k: hid.get(k, "") for k in ("__VIEWSTATE", "__VIEWSTATEGENERATOR", "__EVENTVALIDATION")} if "__VIEWSTATE" in hid else None


def fetch_month(month: pd.Period, form: dict, cookies: Path) -> Path | None:
    day = min(month.end_time.normalize(), pd.Timestamp.now().normalize())
    f = dict(form, __EVENTTARGET="btnSubmit1", __EVENTARGUMENT="", hdnDate=day.strftime("%d-%b-%Y"), txtDate=day.strftime("%d-%b-%Y"), HdnValexceldata="", hdnFlag="")
    body = RAW / "_post.txt"; body.write_text(urllib.parse.urlencode(f))
    out = RAW / f"fpi_daily_{month}.html"; tmp = out.with_suffix(".part")
    code = _curl(["-b", str(cookies), "-c", str(cookies), "-H", "Content-Type: application/x-www-form-urlencoded", "-H", f"Referer: {URL}",
                  "--data-binary", f"@{body}", URL], tmp)
    if code != "200" or "Daily Trends in FPI Investments" not in tmp.read_text(encoding="utf-8", errors="ignore"):
        tmp.unlink(missing_ok=True); return None
    tmp.replace(out)
    return out


def _num(x) -> float | None:
    s = str(x).replace(",", "").replace("Rs.", "").strip()
    if s in ("", "nan", "-", "NaN"):
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse(page: Path) -> pd.DataFrame:
    h = page.read_text(encoding="utf-8", errors="ignore")
    seg = h[h.find('id="dvArchiveData"'):]
    seg = seg[:seg.find("Daily Trends in FPI Derivative")] if "Daily Trends in FPI Derivative" in seg else seg
    t = pd.read_html(io.StringIO(seg), flavor="lxml", header=None)[0]
    t = t.iloc[:, :8]; t.columns = COLS
    t = t[t["reporting_date"].astype(str).str.match(r"^\d{2}-[A-Za-z]{3}-\d{4}$")].copy()
    t["reporting_date"] = pd.to_datetime(t["reporting_date"], format="%d-%b-%Y")
    for c in COLS[3:]:
        t[c] = t[c].map(_num)
    t["asset"] = t["asset"].astype(str).str.strip(); t["route"] = t["route"].astype(str).str.strip()
    tot = (t["route"] == "Total") | (t["asset"] == "Total") | (t["asset"] == t["route"])   # the 'Total' row spans both columns
    t.loc[tot, ["asset", "route"]] = ["Total", "Total"]
    t["usdinr"] = t.groupby("reporting_date")["usdinr"].transform("first")
    return t.drop_duplicates(subset=["reporting_date", "asset", "route"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01")
    ap.add_argument("--refetch", action="store_true", help="fetch every month again (default: only months not on disk + the last two)")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    cookies = RAW / "_cookies.txt"
    now = pd.Timestamp.now().to_period("M")
    months = list(pd.period_range(a.start, now, freq="M"))
    todo = [m for m in months if a.refetch or m >= now - 1 or not (RAW / f"fpi_daily_{m}.html").exists()]
    t0, ok, bad, form = time.time(), 0, [], None
    for m in todo:
        if time.time() - t0 > BUDGET_S:
            print(f"time budget reached ({BUDGET_S // 60} min); {len(todo) - ok - len(bad)} months left for the next run"); break
        form = form or _form(cookies)
        p = fetch_month(m, form, cookies) if form else None
        if p is None:
            form = None; time.sleep(PAUSE_S); form = _form(cookies)       # fresh page state, one retry
            p = fetch_month(m, form, cookies) if form else None
        (bad.append(str(m)) if p is None else None); ok += p is not None
        tried = ok + len(bad)
        if tried >= 5 and len(bad) > tried / 2:
            raise SystemExit(f"stopping: {len(bad)} of {tried} requests failed ({bad[:5]}); NSDL unreachable or the page changed")
        time.sleep(PAUSE_S)
    frames = []
    for p in sorted(RAW.glob("fpi_daily_*.html")):
        try:
            d = parse(p)
        except Exception as e:                       # noqa: BLE001 — a page that does not parse is listed, never guessed
            print(f"could not parse {p.name}: {e}"); continue
        d["archive_month"] = p.stem.replace("fpi_daily_", ""); d["source_url"] = URL
        d["fetched_at"] = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        frames.append(d)
    D = pd.concat(frames, ignore_index=True).sort_values(["reporting_date", "asset", "route"])
    D = D.drop_duplicates(subset=["reporting_date", "asset", "route"], keep="last")
    D["known_from"] = D["reporting_date"] + pd.Timedelta(days=1)
    D.to_parquet(OUT, index=False)
    eq = D[(D["asset"] == "Equity") & (D["route"] == "Sub-total")]
    months_have = sorted(D["archive_month"].unique())
    gaps = [str(m) for m in months if str(m) not in months_have]
    per_month = eq.groupby(eq["reporting_date"].dt.to_period("M")).size()
    thin = {str(k): int(v) for k, v in per_month.items() if v < 12 and k < now}
    OUT.with_name(OUT.name + ".manifest.json").write_text(json.dumps(dict(
        dataset=OUT.name, producer="src/agentic/fetch_nsdl_fpi.py", source=URL, source_owner="NSDL (National Securities Depository Ltd), FPI Monitor",
        definitions=__doc__, rows=len(D), reporting_dates=int(D["reporting_date"].nunique()),
        coverage=[str(D["reporting_date"].min().date()), str(D["reporting_date"].max().date())],
        units=dict(gross_purchases_cr="Rs crore", gross_sales_cr="Rs crore", net_cr="Rs crore, negative = net selling", net_usd_mn="US$ million", usdinr="rupees per US$ (NSDL's rate)"),
        point_in_time="known_from = reporting_date + 1 day; rows reflect confirmed trades on and up to the previous trading day(s)",
        reconciled=("2026-10-12: monthly sums of the daily Equity Sub-total rows equal NSDL's own 'FPI Net Investment Details (Calendar Year)' "
                    "page to the rupee for 2019 (12/12 months), 2024-01 and 2026-01..09 (9/9)"),
        known_gaps=dict(months_missing=gaps, months_with_under_12_reporting_dates=thin, failed_this_run=bad),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    print(f"fetched {ok} month pages this run ({len(bad)} failed: {bad[:6]}); table: {len(D)} rows, {D['reporting_date'].nunique()} reporting dates, "
          f"{D['reporting_date'].min().date()}..{D['reporting_date'].max().date()}; months missing {gaps[:6]}; thin months {list(thin)[:6]}")


if __name__ == "__main__":
    main()
