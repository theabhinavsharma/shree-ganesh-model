"""Fetch Press Information Bureau (PIB) press releases (English, PIB Delhi).

PIB (pib.gov.in) is the official government press-release outlet — covers
every ministry's announcements, regulatory approvals, policy changes,
budget items, infrastructure approvals, defence orders, etc. These are
exactly the third-party-news events the user flagged: "even govt
announcements that the company might not be making but govt would (or
dependent parties would) will affect the price and sentiment."

Site layout (verified 2026-09-29; the May-2026 layout this script was first
written for is gone — see "2026-09-29 fix" below):
  https://www.pib.gov.in/AllRelease.aspx?reg=3&lang=1
    → ASP.NET page listing EVERY release of the CURRENT month for the chosen
      region/language (reg=3 PIB Delhi = HQ, lang=1 English). Items are grouped
      under <h3>ministry</h3>; each <li> has
      <a href='/PressReleseDetail.aspx?PRID=<id>'> and "Posted on: DD Mon YYYY".
    → other months: postback with ddlMonth/ddlYear (+ __VIEWSTATE); the year
      dropdown goes back to 2017. The old ?ydate=DD/MM/YYYY parameter is ignored.
  https://www.pib.gov.in/PressReleasePage.aspx?PRID=<id>&reg=3&lang=1
    → release page: #MinistryName, h2 title, #PrDateTime
      ("Posted On: 25 SEP 2026 2:32PM by PIB Delhi"), body, "(Release ID: N)".

2026-09-29 fix (root cause of "total releases: 0" since at least 2026-08-25):
  pib.gov.in 302-redirects the old ?ydate URL to the Hindi/National month page;
  links are now PressReleseDetail.aspx (the old regex wanted PressReleasePage.aspx)
  so every day parsed to 0 rows, and the 0-row shard was then treated as done.
  Now: one listing request per month, English PIB Delhi, per-day shards keyed on
  the listing's "Posted on" date; 0-row / old-schema shards are not "done".

This script:
  1) For each month touching [start, end]: fetch the month listing (1 request).
  2) For each day without a final shard: fetch release pages (title + ministry
     come from the listing). Body policy (--bodies):
       econ (default) — only releases whose title hits ECON_GATE or whose
                        ministry is in ALWAYS_FETCH_MINISTRIES; the rest get
                        body_fetched=False. Same policy for backfill and daily,
                        so history and live rows are built the same way.
       all / none.
  3) Writes shards data/raw/pib_releases/YYYY/<date>.parquet (atomic).
  4) Consolidates to data/derived/pib_releases.parquet (+ .manifest.json):
     symbols_matched  — company names / uppercase NSE symbols from
                        data/derived/security_master.parquet found in title+body
     industries_tagged — OUR industry names from the transparent keyword map
                        data/derived/pib_industry_keywords.csv

Point in time:
  pub_date     = the listing's "Posted on" date (IST calendar date).
  pub_datetime = release page "Posted On" date+time (IST, +05:30); NULL when the
                 body was not fetched (time is only on the release page).
  A shard is final only when its listing was fetched after its day ended (IST),
  so late-evening releases are not lost; the daily run refreshes provisional
  shards and reuses bodies already fetched.

Politeness: >= 1 s sleep after every request (enforced), retries with backoff,
aborts after repeated consecutive failures.

Usage:
  python3 src/agentic/fetch_pib_releases.py --start 2019-01-01 --newest-first
  python3 src/agentic/fetch_pib_releases.py --start 2026-09-22 --match-symbols   # daily
  python3 src/agentic/fetch_pib_releases.py --consolidate-only
"""
from __future__ import annotations
import argparse
import calendar
import csv
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT_DIR = ROOT / "data/raw/pib_releases"
CONSOLIDATED = ROOT / "data/derived/pib_releases.parquet"
MANIFEST = CONSOLIDATED.with_suffix(".parquet.manifest.json")
CHECKPOINT = OUT_DIR / "_checkpoint.jsonl"
SECURITY_MASTER = ROOT / "data/derived/security_master.parquet"
SCREENER_INDUSTRY = ROOT / "data/derived/screener_industry.parquet"
INDUSTRY_KEYWORDS = ROOT / "data/derived/pib_industry_keywords.csv"

BASE = "https://www.pib.gov.in"
REGION_ID, REGION_NAME, LANG_ID = "3", "PIB Delhi", "1"  # PIB HQ, English
LISTING_URL = f"{BASE}/AllRelease.aspx?reg={REGION_ID}&lang={LANG_ID}"
RELEASE_URL = BASE + "/PressReleasePage.aspx?PRID={prid}&reg=" + REGION_ID + "&lang=" + LANG_ID

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept-Language": "en-US,en;q=0.5",
}

SCHEMA_VERSION = 2
MIN_SLEEP_S = 1.0
SLEEP_S = 1.0
BODY_MAX_CHARS = 8000          # unchanged from v1; body_chars_full records the untruncated length
LEAD_CHARS = 600               # body lead used for industry tagging
MAX_CONSECUTIVE_ERRORS = 25    # abort the run (site blocking / outage) instead of hammering
IST = timezone(timedelta(hours=5, minutes=30))

# Body-fetch gate (--bodies econ). Matched case-insensitively against the TITLE.
ALWAYS_FETCH_MINISTRIES = re.compile(
    r"^(Cabinet|Cabinet Committee on Economic Affairs|Ministry of Finance|Ministry of Commerce & Industry)\b", re.I)
ECON_GATE_TERMS = [
    r"capex", r"capital expenditure", r"allocat\w*", r"approv\w*", r"sanction\w*", r"scheme", r"\bPLI\b",
    r"production[- ]linked", r"incentive", r"subsid\w*", r"tenders?", r"\bbids?\b", r"auction\w*", r"procure\w*",
    r"contracts?", r"orders?\b", r"invest\w*", r"infrastructure", r"projects?", r"budget\w*", r"polic(y|ies)",
    r"dut(y|ies)", r"customs", r"excise", r"\bGST\b", r"tariffs?", r"import\w*", r"export\w*", r"\btrade\b",
    r"\bMSP\b", r"minimum support price", r"ethanol", r"sugar\w*", r"crore", r"billion", r"capacity", r"production",
    r"output", r"manufactur\w*", r"industr\w*", r"\bsectors?\b", r"pric(e|es|ing)\b", r"disinvest\w*", r"privati[sz]\w*",
    r"\bstake\b", r"\bIPO\b", r"\bPSUs?\b", r"\bCPSEs?\b", r"\bFDI\b", r"\bloans?\b", r"\bcredit\b", r"\bbank\w*",
    r"insurance", r"\btax\w*", r"revenue", r"\bGDP\b", r"inflation", r"\b(IIP|WPI|CPI)\b", r"rail\w*", r"highways?",
    r"\broads?\b", r"expressway", r"\bports?\b", r"airports?", r"\bpower\b", r"electricity", r"\benergy\b", r"solar",
    r"\bwind\b", r"renewable", r"\bcoal\b", r"lignite", r"\bsteel\b", r"\bmines?\b", r"\bmining\b", r"mineral\w*",
    r"\boil\b", r"\bgas\b", r"petrol\w*", r"fertili[sz]\w*", r"\burea\b", r"pharma\w*", r"\bdrugs?\b",
    r"medical devices?", r"textiles?", r"telecom\w*", r"\b5G\b", r"spectrum", r"semiconductor\w*", r"electronics",
    r"\bDAC\b", r"defence acquisition", r"missiles?", r"aircraft", r"helicopters?", r"shipbuilding|shipyard|warships?",
    r"\bvessels?\b", r"automobile\w*|automotive", r"electric vehicles?|\bEVs?\b", r"vehicles?", r"batter(y|ies)",
    r"cement", r"housing", r"real estate", r"water supply", r"irrigation", r"kharif|rabi", r"foodgrains?", r"wheat",
    r"\brice\b", r"pulses", r"edible oil", r"oilseeds?", r"cotton", r"\bjute\b", r"\btea\b", r"coffee", r"rubber",
    r"fisher(y|ies)", r"dairy", r"tourism", r"aviation", r"\bUDAN\b", r"logistics", r"freight", r"cargo", r"\bUPI\b",
    r"digital payments?", r"fintech", r"start-?ups?", r"\bMSMEs?\b", r"\bSEBI\b", r"\bRBI\b", r"\bgold\b",
    r"economy|economic", r"\bMoUs?\b", r"agreement", r"growth", r"\bjobs?\b|employment", r"\bfunds?\b|funding",
    r"\bRs\.?\s?[\d,]+", r"₹",
]
ECON_GATE = re.compile("|".join(f"(?:{t})" for t in ECON_GATE_TERMS), re.I)

SHARD_COLUMNS = {
    "pib_id": "string", "pub_date": "string", "pub_datetime": "string", "ministry": "string", "title": "string",
    "body_text": "string", "body_fetched": "bool", "body_status": "string", "body_chars_full": "Int64",
    "econ_gate_hit": "string", "listing_fetched_at": "string", "source_region": "string", "schema_version": "Int64",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def now_ist() -> datetime:
    return datetime.now(IST)


def _norm(s: str | None) -> str:
    return " ".join(html.unescape(s or "").split())


def polite_sleep(sleep_s: float) -> None:
    time.sleep(max(MIN_SLEEP_S, sleep_s))


# ----------------------------------------------------------------------------------------------
# listing
# ----------------------------------------------------------------------------------------------
class MonthListing:
    """Drives the AllRelease.aspx month view (GET = current month, POST back = other months)."""

    def __init__(self, session, sleep_s: float):
        self.session, self.sleep_s = session, sleep_s
        self.form: dict | None = None
        self.url = LISTING_URL

    def _absorb_form(self, soup) -> None:
        form = {i["name"]: i.get("value", "") for i in soup.select("input[type=hidden]") if i.get("name")}
        for sel in soup.find_all("select"):
            if not sel.get("name"):
                continue
            opt = sel.find("option", selected=True) or sel.find("option")
            form[sel["name"]] = opt["value"] if opt else ""
        self.form = form

    def _request(self, year: int, month: int):
        from bs4 import BeautifulSoup
        if self.form is None:
            r = self.session.get(self.url, headers=HEADERS, timeout=90)
            r.raise_for_status()
            polite_sleep(self.sleep_s)
            self.url = r.url
            soup = BeautifulSoup(r.text, "html.parser")
            self._absorb_form(soup)
            if self._selected(soup) == (year, month):
                return soup
        form = dict(self.form)
        form.update({
            "ctl00$Bar1$ddlregion": REGION_ID, "ctl00$Bar1$ddlLang": LANG_ID,
            "ctl00$ContentPlaceHolder1$ddlMinistry": "0", "ctl00$ContentPlaceHolder1$ddlday": "0",
            "ctl00$ContentPlaceHolder1$ddlMonth": str(month), "ctl00$ContentPlaceHolder1$ddlYear": str(year),
            "__EVENTTARGET": "ctl00$ContentPlaceHolder1$ddlYear", "__EVENTARGUMENT": "",
        })
        r = self.session.post(self.url, data=form, headers=HEADERS, timeout=180)
        r.raise_for_status()
        polite_sleep(self.sleep_s)
        soup = BeautifulSoup(r.text, "html.parser")
        self._absorb_form(soup)
        return soup

    @staticmethod
    def _selected(soup) -> tuple[int, int] | None:
        def sel(name):
            s = soup.find("select", attrs={"name": name})
            o = s.find("option", selected=True) if s else None
            return o["value"] if o else None
        y, m = sel("ctl00$ContentPlaceHolder1$ddlYear"), sel("ctl00$ContentPlaceHolder1$ddlMonth")
        lang, reg = sel("ctl00$Bar1$ddlLang"), sel("ctl00$Bar1$ddlregion")
        if not (y and m) or lang != LANG_ID or reg != REGION_ID:
            return None
        return int(y), int(m)

    def fetch(self, year: int, month: int) -> tuple[list[dict], dict]:
        """Return (items, meta) for one month. Raises when the page is not the month asked for."""
        last_err = None
        for attempt in range(3):
            try:
                soup = self._request(year, month)
                got = self._selected(soup)
                if got != (year, month):
                    raise RuntimeError(f"listing returned {got}, wanted {(year, month)} (lang/region/postback drift)")
                return self._parse(soup, year, month)
            except Exception as e:  # noqa: BLE001
                last_err = e
                self.form = None  # restart from a fresh GET
                _log(f"  listing {year}-{month:02d} attempt {attempt + 1} failed: {str(e)[:160]}")
                time.sleep(10 * (attempt + 1) ** 2)
        raise RuntimeError(f"listing {year}-{month:02d} failed: {last_err}")

    @staticmethod
    def _parse(soup, year: int, month: int) -> tuple[list[dict], dict]:
        area = soup.select_one("div.content-area")
        if area is None:
            raise RuntimeError("no div.content-area on listing page (layout change?)")
        items, seen = [], set()
        for li in area.find_all("li"):
            a = li.find("a", href=re.compile(r"PRID=\d+", re.I))
            if not a:
                continue
            prid = re.search(r"PRID=(\d+)", a["href"], re.I).group(1)
            if prid in seen:
                continue
            seen.add(prid)
            h3 = li.find_previous("h3")
            span = li.find("span", class_="publishdatesmall")
            m = re.search(r"(\d{1,2}\s+\w{3}\s+\d{4})", span.get_text(" ", strip=True) if span else "")
            pub = datetime.strptime(m.group(1), "%d %b %Y").date() if m else None
            items.append({"prid": prid, "title": _norm(a.get("title") or a.get_text(" ", strip=True)),
                          "ministry": _norm(h3.get_text(" ", strip=True)) if h3 else "", "pub_date": pub})
        shown = None
        box = soup.select_one("div.search_box_result")
        if box:
            mm = re.search(r"Displaying\s+([\d,]+)", box.get_text(" ", strip=True))
            shown = int(mm.group(1).replace(",", "")) if mm else None
        n_nodate = sum(1 for i in items if i["pub_date"] is None)
        n_other = sum(1 for i in items if i["pub_date"] and (i["pub_date"].year, i["pub_date"].month) != (year, month))
        meta = {"month": f"{year}-{month:02d}", "parsed": len(items), "site_count": shown,
                "no_date": n_nodate, "outside_month": n_other}
        if shown is not None and shown != len(items):
            _log(f"  WARNING {year}-{month:02d}: site says {shown} releases, parsed {len(items)}")
        return items, meta


# ----------------------------------------------------------------------------------------------
# release page
# ----------------------------------------------------------------------------------------------
def fetch_body(session, prid: str) -> tuple[str, str | None, int]:
    """Return (body_text[:BODY_MAX_CHARS], pub_datetime_iso_ist | None, full_body_len)."""
    from bs4 import BeautifulSoup
    r = session.get(RELEASE_URL.format(prid=prid), headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    pub_dt = None
    dt_el = soup.select_one("#PrDateTime")
    if dt_el:
        m = re.search(r"(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\s+(\d{1,2}:\d{2}\s*[AP]M)", dt_el.get_text(" ", strip=True), re.I)
        if m:
            dt = datetime.strptime(f"{m.group(1)} {m.group(2).replace(' ', '').upper()}", "%d %b %Y %I:%M%p")
            pub_dt = dt.replace(tzinfo=IST).isoformat()
    box = soup.select_one("div.innner-page-main-about-us-content-right-part")
    if box is None:  # layout fallback: whole page text, flagged by the caller via body_status
        text = soup.get_text("\n", strip=True)
    else:
        for sel in ("#MinistryName", "#PrDateTime", "h2", "#lg_g"):
            for el in box.select(sel):
                el.decompose()
        text = box.get_text("\n", strip=True)
        cut = re.search(r"\(Release ID:\s*\d+\)", text)
        if cut:
            text = text[:cut.start()]
        text = re.sub(r"\n\*+\s*\n?[A-Z/ .&\-]*$", "", text.strip()).strip()  # trailing "****\nAB/CD" sign-off
    text = re.sub(r"[ \t\xa0]+", " ", text)
    return text[:BODY_MAX_CHARS], pub_dt, len(text)


# ----------------------------------------------------------------------------------------------
# shards
# ----------------------------------------------------------------------------------------------
def shard_path(day: date) -> Path:
    return OUT_DIR / str(day.year) / f"{day.isoformat()}.parquet"


def read_shard(day: date) -> pd.DataFrame | None:
    p = shard_path(day)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:  # noqa: BLE001
        return None


def shard_is_final(df: pd.DataFrame | None, day: date) -> bool:
    """Final = v2 shard + meta, listing fetched after the day ended (IST), no body errors pending.
    v1 shards (the broken parser's 0-row files) and provisional days are re-fetched."""
    if df is None or "schema_version" not in df.columns:
        return False
    meta = _shard_meta(day)
    if not meta or meta.get("schema_version") != SCHEMA_VERSION:
        return False
    if datetime.fromisoformat(meta["listing_fetched_at"]).astimezone(IST).date() <= day:
        return False
    return not bool((df["body_status"] == "error").any())


def _meta_path(day: date) -> Path:
    return shard_path(day).with_suffix(".meta.json")


def _shard_meta(day: date) -> dict | None:
    p = _meta_path(day)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:  # noqa: BLE001
        return None


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_shard(day: date, rows: list[dict], listing_fetched_at: str, listing_meta: dict) -> None:
    df = pd.DataFrame(rows, columns=list(SHARD_COLUMNS))
    for c, t in SHARD_COLUMNS.items():
        df[c] = df[c].astype(t)
    _atomic_parquet(df, shard_path(day))
    meta = {"day": str(day), "schema_version": SCHEMA_VERSION, "n": len(df), "listing_fetched_at": listing_fetched_at,
            "listing_month": listing_meta, "source": LISTING_URL, "region": REGION_NAME}
    mp = _meta_path(day)
    tmp = mp.with_name(f".{mp.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, indent=1, default=str))
    os.replace(tmp, mp)


def append_checkpoint(entry: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def econ_gate(title: str, ministry: str) -> str:
    if ALWAYS_FETCH_MINISTRIES.search(ministry or ""):
        return f"ministry:{ministry}"
    m = ECON_GATE.search(title or "")
    return f"title:{m.group(0)}" if m else ""


# ----------------------------------------------------------------------------------------------
# backfill
# ----------------------------------------------------------------------------------------------
def _months(start: date, end: date, newest_first: bool) -> list[tuple[int, int]]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out[::-1] if newest_first else out


def backfill(start_date: date, end_date: date, sleep_s: float = SLEEP_S, bodies: str = "econ",
             newest_first: bool = False) -> dict:
    import requests
    _log(f"== PIB backfill {start_date} → {end_date} (bodies={bodies}, sleep={max(MIN_SLEEP_S, sleep_s)}s, "
         f"{'newest' if newest_first else 'oldest'} first, region={REGION_NAME}, lang=English) ==")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    lister = MonthListing(session, sleep_s)
    stats = {"days_written": 0, "days_final_skipped": 0, "releases": 0, "bodies_fetched": 0, "bodies_reused": 0,
             "body_errors": 0, "listing_failures": [], "aborted": False}
    consecutive_errors = 0
    t_start = time.time()

    for (y, m) in _months(start_date, end_date, newest_first):
        first = max(start_date, date(y, m, 1))
        last = min(end_date, date(y, m, calendar.monthrange(y, m)[1]))
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        if newest_first:
            days.reverse()
        todo = [d for d in days if not shard_is_final(read_shard(d), d)]
        stats["days_final_skipped"] += len(days) - len(todo)
        if not todo:
            continue
        try:
            items, lmeta = lister.fetch(y, m)
            fetched_at = now_ist().isoformat(timespec="seconds")
            consecutive_errors = 0
        except Exception as e:  # noqa: BLE001
            _log(f"  {y}-{m:02d}: LISTING ERROR {str(e)[:200]}")
            append_checkpoint({"month": f"{y}-{m:02d}", "ok": False, "err": str(e)[:300], "schema_version": SCHEMA_VERSION})
            stats["listing_failures"].append(f"{y}-{m:02d}")
            consecutive_errors += 5
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                stats["aborted"] = True
                break
            continue
        _log(f"  {y}-{m:02d}: listing {lmeta['parsed']} releases (site count {lmeta['site_count']})")
        by_day: dict[date, list[dict]] = {}
        for it in items:
            if it["pub_date"] is not None:
                by_day.setdefault(it["pub_date"], []).append(it)

        for d in todo:
            t0 = time.time()
            old = read_shard(d)
            reuse = {}
            if old is not None and "body_status" in old.columns:
                ok = old[old["body_status"] == "ok"]
                reuse = {r.pib_id: r for r in ok.itertuples(index=False)}
            rows, n_fetch, n_reuse, n_err = [], 0, 0, 0
            for it in sorted(by_day.get(d, []), key=lambda x: int(x["prid"])):
                gate = econ_gate(it["title"], it["ministry"])
                want = bodies == "all" or (bodies == "econ" and bool(gate))
                row = {"pib_id": it["prid"], "pub_date": str(d), "pub_datetime": None, "ministry": it["ministry"],
                       "title": it["title"], "body_text": "", "body_fetched": False,
                       "body_status": "skipped_not_econ" if bodies == "econ" else "skipped",
                       "body_chars_full": None, "econ_gate_hit": gate, "listing_fetched_at": fetched_at,
                       "source_region": REGION_NAME, "schema_version": SCHEMA_VERSION}
                if want and it["prid"] in reuse:
                    r = reuse[it["prid"]]
                    row.update(body_text=r.body_text, pub_datetime=r.pub_datetime, body_fetched=True, body_status="ok",
                               body_chars_full=r.body_chars_full)
                    n_reuse += 1
                elif want:
                    for attempt in range(3):
                        try:
                            body, pub_dt, full_len = fetch_body(session, it["prid"])
                            row.update(body_text=body, pub_datetime=pub_dt, body_fetched=True,
                                       body_status="ok" if body else "empty", body_chars_full=full_len)
                            consecutive_errors = 0
                            break
                        except Exception as e:  # noqa: BLE001
                            row.update(body_status="error", body_text="")
                            _log(f"    PRID {it['prid']} attempt {attempt + 1}: {str(e)[:120]}")
                            time.sleep(5 * (attempt + 1) ** 2)
                        finally:
                            polite_sleep(sleep_s)
                    if row["body_status"] == "error":
                        n_err += 1
                        consecutive_errors += 1
                    else:
                        n_fetch += 1
                rows.append(row)
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    break
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                _log(f"  ABORT: {consecutive_errors} consecutive failures (blocked or outage) — day {d} not written")
                stats["aborted"] = True
                break
            write_shard(d, rows, fetched_at, lmeta)
            elapsed = round(time.time() - t0, 1)
            final = datetime.fromisoformat(fetched_at).date() > d
            stats["days_written"] += 1
            stats["releases"] += len(rows)
            stats["bodies_fetched"] += n_fetch
            stats["bodies_reused"] += n_reuse
            stats["body_errors"] += n_err
            _log(f"  {d}: {len(rows)} releases, bodies fetched {n_fetch} reused {n_reuse} errors {n_err}"
                 f"{'' if final else ' (provisional: day not over in IST)'} ({elapsed}s, run {round((time.time() - t_start) / 60, 1)} min)")
            append_checkpoint({"date": str(d), "n": len(rows), "bodies_fetched": n_fetch, "bodies_reused": n_reuse,
                               "body_errors": n_err, "final": final, "ok": True, "elapsed_s": elapsed,
                               "schema_version": SCHEMA_VERSION})
        if stats["aborted"]:
            break

    _log("\n== PIB backfill done ==")
    _log(f"  days written: {stats['days_written']}  already final: {stats['days_final_skipped']}")
    _log(f"  releases: {stats['releases']:,}  bodies fetched: {stats['bodies_fetched']:,}  reused: "
         f"{stats['bodies_reused']:,}  body errors: {stats['body_errors']}")
    if stats["listing_failures"]:
        _log(f"  LISTING FAILURES: {stats['listing_failures']}")
    if stats["aborted"]:
        _log("  ABORTED early — rerun resumes from the shards already written")
    return stats


# ----------------------------------------------------------------------------------------------
# company matching (security_master)
# ----------------------------------------------------------------------------------------------
_CORP_SUFFIX = {"limited", "ltd", "pvt", "private"}
_CONNECTORS = {"of", "and", "the", "for"}
_INDIA_WORDS = {"india", "indian", "bharat", "hindustan"}  # dictionary words that are distinctive in company names
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&.'\-]*")
# Non-listed entities whose names contain a listed company's name ("Reserve Bank of India" ⊃ "Bank of India");
# matched first so the shorter company name inside them is not counted.
NAME_BLOCKERS = [
    "Reserve Bank of India", "Small Industries Development Bank of India", "Export-Import Bank of India",
    "Exim Bank of India", "National Bank for Agriculture and Rural Development", "India Post Payments Bank",
    "Food Corporation of India", "Cotton Corporation of India", "Jute Corporation of India",
    "Airports Authority of India", "National Highways Authority of India", "Inland Waterways Authority of India",
    "Securities and Exchange Board of India", "Insurance Regulatory and Development Authority of India",
    "Solar Energy Corporation of India", "Central Electricity Authority", "Oil and Natural Gas Corporation Videsh",
]
# Uppercase acronyms PIB uses for listed companies -> NSE symbol. Curated because generic "token == symbol"
# matching is wrong far more often than right in PIB text (MCL/CCL = Mahanadi/Central Coalfields, ITC = input tax
# credit, SHANTI = a bill, APCL = Assam Petro-Chemicals). Entries whose symbol is absent from security_master are
# dropped at load time.
ACRONYM_TO_SYMBOL = {
    "BHEL": "BHEL", "NTPC": "NTPC", "ONGC": "ONGC", "GAIL": "GAIL", "SAIL": "SAIL", "NHPC": "NHPC", "SJVN": "SJVN",
    "NMDC": "NMDC", "MOIL": "MOIL", "KIOCL": "KIOCL", "HUDCO": "HUDCO", "IRFC": "IRFC", "IREDA": "IREDA",
    "RVNL": "RVNL", "IRCON": "IRCON", "RITES": "RITES", "NBCC": "NBCC", "BEML": "BEML", "BEL": "BEL", "HAL": "HAL",
    "BDL": "BDL", "GRSE": "GRSE", "MIDHANI": "MIDHANI", "MDL": "MAZDOCK", "CSL": "COCHINSHIP", "NALCO": "NATIONALUM",
    "CIL": "COALINDIA", "IOCL": "IOC", "BPCL": "BPCL", "HPCL": "HINDPETRO", "MRPL": "MRPL", "CPCL": "CHENNPETRO",
    "PFC": "PFC", "PGCIL": "POWERGRID", "POWERGRID": "POWERGRID", "NLCIL": "NLCINDIA", "SCI": "SCI",
    "CONCOR": "CONCOR", "IRCTC": "IRCTC", "RCF": "RCF", "NFL": "NFL", "GSFC": "GSFC", "GNFC": "GNFC",
    "MSTC": "MSTCLTD", "ITDC": "ITDC", "MMTC": "MMTC", "HMT": "HMT", "MTNL": "MTNL", "EIL": "ENGINERSIN",
    "LIC": "LICI", "GIC": "GICRE", "NIACL": "NIACL", "SBI": "SBIN", "PNB": "PNB", "IOB": "IOB", "UCO": "UCOBANK",
    "IDBI": "IDBI", "CDSL": "CDSL", "NSDL": "NSDL", "BSE": "BSE", "MCX": "MCX", "IEX": "IEX", "TCS": "TCS",
    "HUL": "HINDUNILVR", "RIL": "RELIANCE", "JSPL": "JINDALSTEL", "RAILTEL": "RAILTEL", "HFCL": "HFCL",
    "GMDC": "GMDCLTD", "IFCI": "IFCI", "HCC": "HCC", "IRB": "IRB", "BHARTI": "BHARTIARTL",
}
# Place names behave like dictionary words in company names ("Gujarat Energy Development Agency" is not
# GUJARAT ENERGY LIMITED), and these single-word cores are person names / common Hindi words in PIB text.
_PLACE_WORDS = {"gujarat", "maharashtra", "punjab", "rajasthan", "andhra", "madhya", "uttar", "karnataka", "kerala",
                "tamil", "bengal", "assam", "odisha", "orissa", "bihar", "haryana", "delhi", "mumbai", "bombay",
                "kolkata", "calcutta", "chennai", "madras", "hyderabad", "deccan", "goa", "telangana", "jharkhand",
                "uttarakhand", "himachal", "kashmir", "sikkim", "nagaland", "manipur", "mizoram", "tripura",
                "meghalaya", "arunachal", "chhattisgarh", "pune", "ahmedabad", "surat", "bangalore", "bengaluru",
                "kanpur", "lucknow", "jaipur", "indore", "nagpur", "cochin", "kochi", "mysore", "vizag", "baroda"}
SINGLE_WORD_CORE_STOPLIST = {"arvind", "sonam", "suraj", "vipul", "shrenik", "sandesh", "kamdhenu", "nakoda",
                             "pakka", "ashima", "kalpataru", "kusumgar", "sonam", "shanti", "sobha"}


def _dict_words() -> set[str]:
    p = Path("/usr/share/dict/words")
    try:
        return {w.strip().lower() for w in p.read_text().splitlines()} if p.exists() else set()
    except Exception:  # noqa: BLE001
        return set()


def _toks(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.replace("&", " & ")):
        t = re.sub(r"(['’]s|[.,'])+$", "", t)
        if t:
            out.append(t)
    return out


def load_symbol_universe() -> dict:
    """Company-name phrases and uppercase-symbol tokens from data/derived/security_master.parquet."""
    if not SECURITY_MASTER.exists():
        _log(f"  WARNING {SECURITY_MASTER.relative_to(ROOT)} missing — symbols_matched left empty")
        return {"phrases": {}, "acronyms": {}, "n_companies": 0}
    sm = pd.read_parquet(SECURITY_MASTER, columns=["symbol", "company_name", "is_fund_unit", "is_non_equity",
                                                   "is_rights_entitlement", "first_trade", "last_trade"])
    sm = sm[~sm["is_fund_unit"].fillna(False).astype(bool) & ~sm["is_non_equity"].fillna(False).astype(bool)
            & ~sm["is_rights_entitlement"].fillna(False).astype(bool) & sm["symbol"].notna()]
    words = _dict_words()
    phrases: dict[str, list[tuple[tuple[str, ...], str | None]]] = {}

    def add(v: tuple[str, ...], sym: str | None) -> None:
        if v and len(" ".join(v)) >= 5:
            phrases.setdefault(v[0], []).append((v, sym))

    for sym, name in zip(sm["symbol"], sm["company_name"]):
        if not (isinstance(name, str) and name.strip()):
            continue
        toks = tuple(t.lower() for t in _toks(name))
        add(toks, sym)                                   # full registered name, e.g. "bharat electronics limited"
        core = list(toks)
        while core and core[-1] in _CORP_SUFFIX:
            core.pop()
        if core and core[0] == "the":
            core = core[1:]
        content = [t for t in core if t not in _CONNECTORS]
        distinctive = [t for t in content if t.isalpha() and t not in _PLACE_WORDS
                       and (t not in words or t in _INDIA_WORDS)]
        # name without "Limited" only when it is unlikely to be ordinary prose:
        #   >= 3 content words, or 2 with a distinctive one ("Coal India", "Bharat Electronics"),
        #   or 1 distinctive word of >= 5 letters ("Infosys"). "Global Education" needs "Limited".
        if len(content) >= 3 or (len(content) == 2 and distinctive) or \
                (len(content) == 1 and distinctive and len(content[0]) >= 5 and content[0] not in _INDIA_WORDS
                 and content[0] not in SINGLE_WORD_CORE_STOPLIST):
            add(tuple(core), sym)
    for b in NAME_BLOCKERS:
        add(tuple(t.lower() for t in _toks(b)), None)
    have = set(map(str, sm["symbol"]))
    acronyms = {a: sym for a, sym in ACRONYM_TO_SYMBOL.items() if sym in have}
    # trading window per symbol, so a release matches the symbol that traded on its date (renames such as
    # NEYVELILIG -> NLCINDIA, and no matches to companies that listed after the release). The latest panel
    # session counts as open-ended.
    last_session = sm["last_trade"].max()
    window = {str(r.symbol): (r.first_trade.date() if pd.notna(r.first_trade) else None,
                              None if (pd.isna(r.last_trade) or r.last_trade >= last_session) else r.last_trade.date())
              for r in sm.itertuples(index=False)}
    return {"phrases": phrases, "acronyms": acronyms, "window": window, "n_companies": int(len(sm))}


def _active(sym: str, on: date | None, universe: dict) -> bool:
    if on is None:
        return True
    first, last = universe.get("window", {}).get(sym, (None, None))
    return (first is None or first <= on) and (last is None or on <= last + timedelta(days=7))


def match_symbols(text: str, universe: dict, on: date | None = None) -> list[str]:
    """Company-name phrases (every non-connector word Capitalised or ALL-CAPS in the text; leftmost-longest match
    wins, so "Reserve Bank of India" / "RBL Bank of India" do not yield "Bank of India") plus the curated
    ACRONYM_TO_SYMBOL uppercase tokens in mixed-case text (e.g. 'BHEL', 'NTPC', 'RVNL').
    on = release date: only symbols trading on that date (security_master first/last_trade) are returned."""
    if not text or not universe.get("phrases"):
        return []
    toks = _toks(text)
    low = [t.lower() for t in toks]
    phrases = universe["phrases"]
    spans: dict[tuple[int, int], set] = {}
    for i, w in enumerate(low):
        for v, sym in phrases.get(w, ()):
            n = len(v)
            if tuple(low[i:i + n]) == v and all(t[0].isupper() or t[0].isdigit() or lw in _CONNECTORS
                                                 for t, lw in zip(toks[i:i + n], v)):
                if sym is None or _active(sym, on, universe):
                    spans.setdefault((i, n), set()).add(sym)
    taken = [False] * len(toks)
    found: set[str] = set()
    for (i, n) in sorted(spans, key=lambda c: (c[0], -c[1])):  # leftmost-longest
        if any(taken[i:i + n]):
            continue
        for k in range(i, i + n):
            taken[k] = True
        found.update(x for x in spans[(i, n)] if x is not None)
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < 0.5:  # skip ALL-CAPS texts
        for k, t in enumerate(toks):
            if not taken[k] and t in universe["acronyms"] and _active(universe["acronyms"][t], on, universe):
                found.add(universe["acronyms"][t])
    return sorted(found)


# ----------------------------------------------------------------------------------------------
# industry tagging (data/derived/pib_industry_keywords.csv)
# ----------------------------------------------------------------------------------------------
def load_industry_keywords() -> list[dict]:
    if not INDUSTRY_KEYWORDS.exists():
        _log(f"  WARNING {INDUSTRY_KEYWORDS.relative_to(ROOT)} missing — industries_tagged left empty")
        return []
    rows = list(csv.DictReader(open(INDUSTRY_KEYWORDS, newline="")))
    valid = None
    if SCREENER_INDUSTRY.exists():
        valid = {html.unescape(x) for x in pd.read_parquet(SCREENER_INDUSTRY, columns=["industry"])["industry"].dropna()}
    bad = [r["industry"] for r in rows if valid is not None and r["industry"] not in valid]
    if bad:
        raise ValueError(f"pib_industry_keywords.csv has industries not in our taxonomy: {sorted(set(bad))}")
    for r in rows:
        r["rx"] = re.compile(rf"\b(?:{r['pattern']})\b", re.I)
        r["mrx"] = re.compile(r["ministry_regex"], re.I) if r.get("ministry_regex") else None
    return rows


def _ministry_stripper(ministries: pd.Series) -> re.Pattern | None:
    """Regex removing ministry/minister/department NAME phrases ("Minister of Coal", "Coal Ministry") so that a
    ministry's own name does not tag its industry; the release content still can."""
    tails = set()
    for mn in ministries.dropna().unique():
        for part in re.split(r"\s+-\s+", str(mn)):
            m = re.match(r"(?:Ministry|Department)\s+of\s+(.+)", part.strip(), re.I)
            if m:
                tails.add(m.group(1).strip())
    if not tails:
        return None
    alts = "|".join(sorted((re.escape(t).replace(r"\ ", r"\s+").replace(r"\&", "(?:&|and)")
                            .replace(r"\s+and\s+", r"\s+(?:&|and)\s+").replace(",", ",?") for t in tails), key=len, reverse=True))
    return re.compile(
        rf"\b(?:(?:Union\s+)?(?:Ministry|Ministers?|Minister\s+of\s+State|MoS|Secretary|Department|Dept\.?)"
        rf"(?:\s+of\s+State)?\s+(?:of|for)\s+(?:the\s+)?(?:{alts})|(?:{alts})\s+(?:Ministry|Minister|Secretary))\b", re.I)


def tag_industries(title: str, lead: str, ministry: str, kw: list[dict], stripper) -> tuple[list[str], list[str]]:
    title_c = stripper.sub(" ", title or "") if stripper else (title or "")
    lead_c = stripper.sub(" ", lead or "") if stripper else (lead or "")
    inds, ev = set(), []
    for r in kw:
        if r["mrx"] is not None and not r["mrx"].search(ministry or ""):
            continue
        for field, txt in (("title", title_c), ("lead", lead_c)):
            m = r["rx"].search(txt)
            if m:
                if r["industry"] not in inds:
                    ev.append(f"{r['industry']}<={m.group(0)}@{field}")
                inds.add(r["industry"])
                break
    return sorted(inds), ev


# ----------------------------------------------------------------------------------------------
# consolidation
# ----------------------------------------------------------------------------------------------
def consolidate(symbol_match: bool = True) -> None:
    files = sorted(p for p in OUT_DIR.glob("[0-9][0-9][0-9][0-9]/*.parquet"))
    frames, stale, provisional = [], 0, 0
    for f in files:
        try:
            d = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            _log(f"  skip {f}: {e}")
            continue
        if "schema_version" not in d.columns:
            stale += 1  # v1 shards written by the broken parser (all 0 rows) — not data
            continue
        day = date.fromisoformat(f.stem)
        if not shard_is_final(d, day):
            provisional += 1
        if len(d):
            frames.append(d)
    if not frames:
        _log(f"no usable shards (v1/stale shards ignored: {stale})")
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["pub_date", "pib_id"]).drop_duplicates(subset=["pib_id"], keep="last").reset_index(drop=True)
    _log(f"consolidated: {len(df):,} rows from {len(frames)} non-empty shards "
         f"(provisional shards: {provisional}, v1 shards ignored: {stale})")

    universe = load_symbol_universe()
    body = df["body_text"].fillna("")
    title = df["title"].fillna("")
    df["symbols_matched"] = [match_symbols(f"{t}\n{b}", universe, date.fromisoformat(pdt))
                             for t, b, pdt in zip(title, body, df["pub_date"])]
    df["n_symbols"] = df["symbols_matched"].apply(len)
    kw = load_industry_keywords()
    stripper = _ministry_stripper(df["ministry"])
    tagged = [tag_industries(t, b[:LEAD_CHARS], mn, kw, stripper)
              for t, b, mn in zip(title, body, df["ministry"].fillna(""))]
    df["industries_tagged"] = [x[0] for x in tagged]
    df["industry_tag_evidence"] = [x[1] for x in tagged]

    cols = ["pib_id", "pub_date", "pub_datetime", "ministry", "title", "body_text", "symbols_matched",
            "industries_tagged", "industry_tag_evidence", "n_symbols", "body_fetched", "body_status",
            "body_chars_full", "econ_gate_hit", "listing_fetched_at", "source_region", "schema_version"]
    df = df[cols]
    _atomic_parquet(df, CONSOLIDATED)
    _log(f"wrote {CONSOLIDATED.relative_to(ROOT)}")
    write_manifest(df, universe, kw, stale, provisional)


def _coverage_days() -> dict:
    final, prov = [], []
    for p in OUT_DIR.glob("[0-9][0-9][0-9][0-9]/*.meta.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if m.get("schema_version") != SCHEMA_VERSION:
            continue
        d = date.fromisoformat(m["day"])
        (final if datetime.fromisoformat(m["listing_fetched_at"]).astimezone(IST).date() > d else prov).append(d)
    return {"final": sorted(final), "provisional": sorted(prov)}


def write_manifest(df: pd.DataFrame, universe: dict, kw: list[dict], stale: int, provisional: int) -> None:
    cov = _coverage_days()
    yr = df["pub_date"].str[:4]
    by_year = {}
    for y, g in df.groupby(yr):
        by_year[y] = {"releases": int(len(g)), "body_fetched": int(g["body_fetched"].sum()),
                      "body_errors": int((g["body_status"] == "error").sum()),
                      "with_symbols": int((g["n_symbols"] > 0).sum()),
                      "with_industries": int(g["industries_tagged"].apply(len).gt(0).sum())}
    final_days = cov["final"]
    gaps = []
    if final_days:
        s = set(final_days)
        d, end = final_days[0], final_days[-1]
        while d <= end:
            if d not in s:
                gaps.append(str(d))
            d += timedelta(days=1)
    dt_mismatch = int(((df["pub_datetime"].notna()) & (df["pub_datetime"].str[:10] != df["pub_date"])).sum())
    kw_sha = hashlib.sha256(INDUSTRY_KEYWORDS.read_bytes()).hexdigest()[:16] if INDUSTRY_KEYWORDS.exists() else None
    man = {
        "dataset": "pib_releases",
        "path": str(CONSOLIDATED.relative_to(ROOT)),
        "key": ["pib_id"],
        "producer": "src/agentic/fetch_pib_releases.py",
        "source": {
            "listing": LISTING_URL + " (month view; other months via ASP.NET postback ddlMonth/ddlYear)",
            "release_page": RELEASE_URL.format(prid="<pib_id>"),
            "region": f"{REGION_NAME} (reg={REGION_ID}); English (lang={LANG_ID})",
            "official": "Press Information Bureau, Government of India",
        },
        "point_in_time": {
            "pub_date": "listing 'Posted on' date, IST calendar date",
            "pub_datetime": "release page 'Posted On' timestamp, IST (+05:30); NULL when body_fetched is False",
            "pub_datetime_date_differs_from_pub_date": dt_mismatch,
            "provisional_shards": "a day is final only when its listing was fetched after the day ended in IST; "
                                  "provisional days are refreshed by the next run",
        },
        "body_policy": {
            "mode": "econ gate (same for backfill and daily run)",
            "always_fetch_ministries": ALWAYS_FETCH_MINISTRIES.pattern,
            "title_gate_terms": ECON_GATE_TERMS,
            "body_max_chars": BODY_MAX_CHARS,
            "note": "releases not passing the gate have body_fetched=False, body_text='' and pub_datetime NULL; "
                    "their symbols/industries come from the title only",
        },
        "columns": {
            "pib_id": "PIB PRID (string); English PIB Delhi release id",
            "pub_date": "YYYY-MM-DD (IST) from the month listing",
            "pub_datetime": "ISO timestamp +05:30 from the release page, NULL if not fetched",
            "ministry": "listing <h3> group (whitespace-normalised)",
            "title": "release title from the listing",
            "body_text": f"release text (header and '(Release ID)' footer removed), first {BODY_MAX_CHARS} chars; '' if not fetched",
            "symbols_matched": "NSE symbols whose security_master company name (with or without 'Limited') appears "
                               "Capitalised in title+body (leftmost-longest; person/place-name guards in code), or a curated "
                               "uppercase acronym (ACRONYM_TO_SYMBOL, e.g. BHEL, NALCO->NATIONALUM) in mixed-case text; only symbols "
                               "trading on pub_date (security_master first_trade..last_trade) are kept. "
                               "Best effort: false positives/negatives expected",
            "industries_tagged": "OUR industry names (screener basic-industry taxonomy) from "
                                 "data/derived/pib_industry_keywords.csv matched in title + first "
                                 f"{LEAD_CHARS} chars of body, after removing ministry-name phrases",
            "industry_tag_evidence": "'<industry> <= <matched text> @ title|lead' per tag, for review",
            "n_symbols": "len(symbols_matched)",
            "body_fetched": "True if the release page was fetched",
            "body_status": "ok | empty | error | skipped_not_econ | skipped",
            "body_chars_full": "untruncated body length (truncated when > body_max_chars)",
            "econ_gate_hit": "why the body was fetched ('ministry:..' or 'title:<term>'), '' if not",
            "listing_fetched_at": "IST timestamp of the listing request that produced the row",
            "source_region": "PIB region of the listing",
            "schema_version": SCHEMA_VERSION,
        },
        "coverage": {
            "rows": int(len(df)),
            "pub_date_min": df["pub_date"].min(), "pub_date_max": df["pub_date"].max(),
            "final_days": len(final_days), "provisional_days": [str(d) for d in cov["provisional"]],
            "missing_days_inside_final_range": {"count": len(gaps), "first_50": gaps[:50]},
            "by_year": by_year,
            "v1_shards_ignored": stale,
        },
        "matching_inputs": {
            "security_master": str(SECURITY_MASTER.relative_to(ROOT)),
            "companies_used": universe.get("n_companies", 0),
            "industry_keywords": str(INDUSTRY_KEYWORDS.relative_to(ROOT)),
            "industry_keywords_sha256_16": kw_sha,
            "industry_keyword_rows": len(kw),
        },
        "limitations": [
            "Only the English PIB Delhi (HQ) stream; regional-office and Hindi/regional-language releases are separate "
            "PRIDs and are not collected.",
            "Bodies are fetched only for econ-gated releases (~40% of all); the rest have title + ministry only.",
            "The listing year dropdown starts in 2017; data before 2019-01-01 was not backfilled. Shards under "
            "data/raw/pib_releases/2016-2018 are v1 artefacts of the broken parser (0 rows) and are ignored.",
            "symbols_matched and industries_tagged are keyword heuristics, not labels; review before high-stakes use. "
            "The keyword map is static config applied to point-in-time text, re-applied on every consolidation.",
        ],
        "updated": now_ist().isoformat(timespec="seconds"),
    }
    tmp = MANIFEST.with_name(f".{MANIFEST.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(man, indent=1, default=str))
    os.replace(tmp, MANIFEST)
    _log(f"wrote {MANIFEST.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01", help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today in IST)")
    ap.add_argument("--sleep", type=float, default=SLEEP_S, help=f"seconds after each request (min {MIN_SLEEP_S})")
    ap.add_argument("--bodies", choices=["econ", "all", "none"], default="econ")
    ap.add_argument("--no-bodies", action="store_true", help="alias for --bodies none")
    ap.add_argument("--newest-first", action="store_true", help="walk months/days newest → oldest")
    ap.add_argument("--consolidate-only", action="store_true")
    ap.add_argument("--match-symbols", action="store_true",
                    help="kept for CLI compatibility; symbol + industry matching always runs at consolidation")
    args = ap.parse_args()

    if args.consolidate_only:
        consolidate()
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else now_ist().date()
    stats = backfill(start, end, args.sleep, bodies="none" if args.no_bodies else args.bodies,
                     newest_first=args.newest_first)
    consolidate()
    if stats["aborted"] or stats["listing_failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
