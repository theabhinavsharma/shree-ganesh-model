"""Point-in-time monthly IIP (MoSPI) and Index of Eight Core Industries (OEA/DPIIT), 2016 -> today.

Phase 2 of the industry-first stock-selection study. Every value carries the date of the official
press release that first published *that exact number* and its vintage, so a backtest can use only
figures known on a given date.

Sources (all official; no third-party mirrors):
  IIP  latest vintage : MoSPI eSankhyiki REST API  https://api.mospi.gov.in/api/iip/getIIPMonthly
                        (base_year 2011-12 -> Apr-2012..Mar-2026; base_year 2022-23 -> Apr-2023..today)
  IIP  vintages       : MoSPI monthly press releases (PDF, plus an index workbook from Oct-2025),
                        listed by the MoSPI site API  www.mospi.gov.in/api/latest-release/... and
                        www.mospi.gov.in/api/archive/archival-data (model_type=latest_release)
  ICI  vintages       : OEA press releases  https://eaindustry.nic.in/archive_data/ici_press_release/IPR_YYYY_MM.pdf
                        (+ the current release linked from ici_download_data.asp)
  ICI  latest vintage : OEA workbooks linked from https://eaindustry.nic.in/ici_download_data.asp
  Gaps in those archives are filled from the Press Information Bureau copy of the same release
  (PIB_IIP_PRIDS / PIB_ICI below; located by searching pib.gov.in, PIB 'Posted On' = release date).

Bases: IIP 2011-12 (releases 12-May-2017 .. 28-Apr-2026) and 2022-23 (from 01-Jun-2026); ICI 2004-05
(real-time up to Feb-2017 data), 2011-12 (31-May-2017 .. May-2026 data) and 2022-23 (from 20-Jul-2026,
adds Iron Ore). Levels of different bases are never spliced; base_year is a column.

Raw files are cached under data/raw/iip/ and data/raw/core_sector/ (re-runs only download what is
missing; --refresh re-pulls the listings, the API and the "current" files).

Vintage logic (read off each release, not assumed):
  IIP : release for month M carries the Quick Estimate (QE) of M, the 1st revision of an earlier
        month and the final revision of another; the months are parsed from the release text
        ("the indices for May 2024 have undergone the first revision and those for March 2024 have
        undergone final revision"). Statement IV / the monthly tables re-print the last 12 months, so
        every release is a full 12-month vintage snapshot for NIC-2, sectors and the general index,
        and Statement III (use-based) re-prints the financial year to date.
  ICI : release for month M prints 13 months of indices and y-o-y growth (Annex II); months starred
        '*' are provisional, the note names the month whose growth is final.
  Each (series, month) keeps the first print and every later print whose value changed, plus the
  official final print; see the manifest for the vintage vocabulary.
Outputs: data/derived/iip_monthly.parquet, core_sector_monthly.parquet, activity_industry_map.csv
(each with a .manifest.json). Industry names come from --industries (column 'industry').

Usage:
  python3 src/agentic/fetch_iip_core.py            # fetch (cached) + parse + write outputs
  python3 src/agentic/fetch_iip_core.py --refresh  # also re-pull listings / API / current files
  python3 src/agentic/fetch_iip_core.py --no-fetch # parse the cache only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import urllib.parse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path("/Users/abhinavs./Documents/Zoom")
RAW_IIP = ROOT / "data/raw/iip"
RAW_CORE = ROOT / "data/raw/core_sector"
DERIVED = ROOT / "data/derived"
UA = {"User-Agent": "Mozilla/5.0 (research; SGM industry study)"}
SLEEP = 1.3

MOSPI_API = "https://api.mospi.gov.in/api/iip/getIIPMonthly"
MOSPI_SITE = "https://www.mospi.gov.in/"
MOSPI_LIST_WEB = "https://www.mospi.gov.in/api/latest-release/get-web-latest-release-list"
MOSPI_LIST_ARCH = "https://www.mospi.gov.in/api/archive/archival-data"
EA = "https://eaindustry.nic.in/"

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"], start=1)}
MON3 = {k[:3]: v for k, v in MONTHS.items()}
MON3["sept"] = 9


# ----------------------------------------------------------------------------------------------
# HTTP (python requests with normal TLS verification; system curl as fallback for the proxy)
# ----------------------------------------------------------------------------------------------
def _curl(args: list[str]) -> bytes:
    res = subprocess.run(["curl", "-sS", "-f", "-m", "180", "--retry", "2", "-A", UA["User-Agent"], *args],
                         capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"curl rc={res.returncode}: {res.stderr.decode(errors='ignore')[:200]}")
    return res.stdout


def http_get(url: str) -> bytes:
    try:
        r = requests.get(url, headers=UA, timeout=180)
        r.raise_for_status()
        return r.content
    except requests.exceptions.SSLError:
        return _curl([url])


def http_post_json(url: str, body: dict) -> dict:
    try:
        r = requests.post(url, json=body, headers=UA, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.SSLError:
        return json.loads(_curl(["-H", "Content-Type: application/json", "-X", "POST", url,
                                 "-d", json.dumps(body)]))


def cached_download(url: str, dest: Path, refresh: bool = False) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        blob = http_get(url)
    except Exception as e:  # noqa: BLE001 - log and continue; the gap is reported downstream
        print(f"  ! download failed {url}: {type(e).__name__} {str(e)[:120]}", flush=True)
        return None
    finally:
        time.sleep(SLEEP)
    if blob[:5] == b"<!DOC" or blob[:5] == b"<html":
        print(f"  ! got HTML instead of a file for {url}", flush=True)
        return None
    dest.write_bytes(blob)
    return dest


# ----------------------------------------------------------------------------------------------
# FETCH: IIP
# ----------------------------------------------------------------------------------------------
def fetch_iip_api(refresh: bool) -> None:
    """Latest-vintage monthly indices + growth for every series, per base year, 200 rows/page."""
    out = RAW_IIP / "api"
    out.mkdir(parents=True, exist_ok=True)
    this_year = date.today().year
    for base, years in [("2011-12", range(2012, this_year + 1)), ("2022-23", range(2022, this_year + 1))]:
        for y in years:
            page = 1
            while True:
                f = out / f"iip_monthly_base{base}_{y}_p{page}.json"
                current = (y >= this_year - 1)
                if f.exists() and not (refresh and current):
                    d = json.loads(f.read_text())
                else:
                    q = urllib.parse.urlencode(dict(base_year=base, year=y, type="All", Format="JSON",
                                                    limit=200, page=page))
                    d = json.loads(http_get(f"{MOSPI_API}?{q}"))
                    f.write_text(json.dumps(d))
                    time.sleep(SLEEP)
                meta = d.get("meta_data") or {}
                if page >= int(meta.get("totalPages") or 1):
                    break
                page += 1


def fetch_mospi_release_lists(refresh: bool) -> list[dict]:
    """Every IIP press release MoSPI lists (current web list + archive); returns normalised rows."""
    out = RAW_IIP / "release_index"
    out.mkdir(parents=True, exist_ok=True)
    f_web, f_arch = out / "mospi_latest_release_web.json", out / "mospi_latest_release_archival.json"
    if refresh or not f_web.exists():
        rows, p = [], 1
        while True:
            d = http_post_json(MOSPI_LIST_WEB, {"page_no": p, "page_size": 100, "search_term": "", "lang": "en"})
            rows += d.get("data") or []
            time.sleep(SLEEP)
            if p >= int((d.get("pagination") or {}).get("totalPages") or 1):
                break
            p += 1
        f_web.write_text(json.dumps(rows, indent=1))
    if refresh or not f_arch.exists():
        rows, p = [], 1
        while True:
            d = http_post_json(MOSPI_LIST_ARCH, {"model_type": "latest_release", "lang": "en", "page_size": 100,
                                                 "page_no": p, "search_term": "", "sort_by": "published_year",
                                                 "sort_order": "DESC", "start_date": "", "end_date": ""})
            rows += d.get("data") or []
            time.sleep(SLEEP)
            if p >= int((d.get("pagination") or {}).get("totalPages") or 1):
                break
            p += 1
        f_arch.write_text(json.dumps(rows, indent=1))

    return load_release_index()


def load_release_index() -> list[dict]:
    """Normalised IIP release rows from the cached MoSPI listings (no network)."""
    out = RAW_IIP / "release_index"
    f_web, f_arch = out / "mospi_latest_release_web.json", out / "mospi_latest_release_archival.json"
    norm = []
    for src, f in [("web", f_web), ("archival", f_arch)]:
        for r in json.loads(f.read_text()):
            title = (r.get("title") or r.get("title_en") or "").strip().replace("\r\n", " ")
            if not re.search(r"quick estimates? of (iip|index of industrial production)|index of industrial production"
                             r".*(new series|base year 2022-23)|new series of iip", title, re.I):
                continue
            if re.search(r"FAQ|Frequently|Technical Advisory|Discussion|Workshop|Corrigendum", title, re.I):
                continue
            # pre-2016 releases are out of scope (archive 'published_year' is an upload date for old items,
            # so also drop files whose name carries a 2014/2015 release date)
            if (r.get("published_year") or "9999") < "2016-01-01":
                continue
            names = " ".join([r.get("pdf_file") or ""])
            if re.search(r"(1[45])\.pdf$|_(\d{1,2})[a-z]+1[45]\.pdf$", names, re.I):
                continue
            files = []
            if r.get("pdf_file"):
                files.append(r["pdf_file"])
            for k in ("file_one", "file_two", "file_three"):
                v = r.get(k)
                if isinstance(v, dict) and v.get("path"):
                    files.append(v["path"])
            norm.append(dict(list_source=src, list_id=str(r.get("id")), title=title,
                             listed_date=r.get("published_year"), files=files))
    return norm


def fetch_iip_press_releases(refresh: bool) -> None:
    rel = fetch_mospi_release_lists(refresh)
    out = RAW_IIP / "press_releases"
    seen = set()
    for r in rel:
        for path in r["files"]:
            if path in seen:
                continue
            seen.add(path)
            name = Path(path).name
            cached_download(MOSPI_SITE + urllib.parse.quote(path), out / name)
    (RAW_IIP / "release_index" / "iip_release_files.json").write_text(json.dumps(rel, indent=1))
    print(f"  IIP press-release listing: {len(rel)} releases, {len(seen)} files cached under {out}", flush=True)


# Releases missing from the MoSPI archive, recovered from the Press Information Bureau (official
# government channel that carries the same MoSPI release; PIB "Posted On" = release date). PRIDs were
# located by searching pib.gov.in for each missing reference month (2026-09-29). Sep-2017 was not found.
PIB_IIP_PRIDS = {
    "2020-04": "1631154", "2021-06": "1745171", "2021-07": "1753865", "2021-08": "1763250",
    "2021-09": "1771219", "2021-10": "1780176", "2024-10": "2083705", "2024-11": "2091785",
    "2024-12": "2102261", "2025-01": "2110777", "2025-02": "2120934", "2025-03": "2124850",
    "2025-04": "2132055", "2025-05": "2140774", "2025-06": "2149266", "2025-07": "2161516",
    "2025-08": "2172702",
}
PIB_PAGE = "https://www.pib.gov.in/PressReleasePage.aspx?PRID={}"


def fetch_pib_iip(refresh: bool) -> None:
    out = RAW_IIP / "press_releases" / "pib"
    out.mkdir(parents=True, exist_ok=True)
    for ref, prid in PIB_IIP_PRIDS.items():
        f = out / f"PIB_PRID{prid}.html"
        if refresh or not f.exists():
            f.write_bytes(http_get(PIB_PAGE.format(prid)))
            time.sleep(SLEEP)
        html = f.read_text(errors="ignore")
        for u in sorted(set(re.findall(r"https?://static\.pib\.gov\.in/WriteReadData/specificdocs/documents/[^\"' ]+\.pdf", html))):
            cached_download(u, out / f"PIB_PRID{prid}__{Path(u).name}")


def pib_html_text(path: Path) -> tuple[str, date | None]:
    """Release body as text with table rows as whitespace-separated lines (parser-compatible), + 'Posted On' date."""
    from bs4 import BeautifulSoup
    html = path.read_text(errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    posted = None
    m = re.search(r"Posted On:\s*(\d{1,2}\s+[A-Z]{3}\s+\d{4})", html)
    if m:
        posted = datetime.strptime(m.group(1).title(), "%d %b %Y").date()
    body = soup.select_one("div.innner-page-main-about-us-content-right-part") or soup.select_one("#PdfDiv") or soup
    lines = []
    for el in body.find_all(["p", "h2", "h3", "li", "table"], recursive=True):
        if el.name == "table":
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                lines.append("   ".join(c for c in cells if c))
            el.decompose()
        elif el.name in ("p", "h2", "h3", "li"):
            t = el.get_text(" ", strip=True)
            if t:
                lines.append(t)
    txt = "\n".join(lines)
    return re.sub("[‐‑‒–—−]", "-", txt), posted


# ----------------------------------------------------------------------------------------------
# FETCH: Core sector (ICI)
# ----------------------------------------------------------------------------------------------
def fetch_core(refresh: bool) -> None:
    out = RAW_CORE
    (out / "press_releases").mkdir(parents=True, exist_ok=True)
    (out / "data_files").mkdir(parents=True, exist_ok=True)
    pages = {}
    for page in ("ici_press_release_archive.asp", "ici_download_data.asp"):
        f = out / "listing" / page.replace(".asp", ".html")
        f.parent.mkdir(parents=True, exist_ok=True)
        if refresh or not f.exists():
            f.write_bytes(http_get(EA + page))
            time.sleep(SLEEP)
        pages[page] = f.read_text(encoding="latin-1")
    links = set()
    for html in pages.values():
        links |= set(re.findall(r'href="([^"]+)"', html))
    # archived monthly releases (reference month in the file name), 2016 onwards
    for l in sorted(links):
        m = re.search(r"archive_data/ici_press_release/IPR_(\d{4})_(\d{2})\.pdf$", l)
        if m and (int(m.group(1)), int(m.group(2))) >= (2015, 12):   # Dec-2015 announces the Jan-2016 date
            cached_download(EA + l, out / "press_releases" / Path(l).name)
    # current release + workbooks (their names carry the release date)
    for l in sorted(links):
        if re.search(r"eight_core_infra/(Press_Release_ICI_\d{8}\.pdf|Core_Industries_\d{4}_\d{2}_\d{8}\.xlsx)$", l):
            sub = "press_releases" if l.endswith(".pdf") else "data_files"
            cached_download(EA + urllib.parse.quote(l), out / sub / Path(l).name)
    print(f"  ICI: {len(list((out / 'press_releases').glob('*.pdf')))} release PDFs, "
          f"{len(list((out / 'data_files').glob('*.xlsx')))} workbooks cached", flush=True)



# ----------------------------------------------------------------------------------------------
# PARSE helpers
# ----------------------------------------------------------------------------------------------
MON_ALT = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
NUMTOK = re.compile(r"^[-−–]?\d+(?:\.\d+)?$")


def mnum(tok: str) -> int:
    t = tok.strip().lower().rstrip(".,")
    return MONTHS.get(t) or MON3.get(t[:4] if t.startswith("sept") else t[:3])


def ym(y: int, m: int) -> pd.Period:
    return pd.Period(year=int(y), month=int(m), freq="M")


def fnum(tok: str) -> float:
    return float(tok.replace("−", "-").replace("–", "-"))


def pdf_text(path: Path) -> str:
    res = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True)
    t = res.stdout.decode("utf-8", errors="ignore")
    return re.sub("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", t)   # unicode dashes/minus -> '-' 


def norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def find_dates(text: str) -> list[date]:
    """All explicit calendar dates in a text snippet ('12th March, 2019', 'August 12th, 2024', '12 of May, 2017')."""
    t = norm_ws(text.replace("’", "'"))
    t = re.sub(r"(\d)\s+(st|nd|rd|th)\b", r"\1\2", t)
    out = []
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:of\s+|day of\s+)?" + MON_ALT + r"[a-z]*[,.]?\s+(\d{4})\b", t, re.I):
        try:
            out.append(date(int(m.group(3)), mnum(m.group(2)), int(m.group(1))))
        except (TypeError, ValueError):
            pass
    for m in re.finditer(MON_ALT + r"[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", t, re.I):
        try:
            out.append(date(int(m.group(3)), mnum(m.group(1)), int(m.group(2))))
        except (TypeError, ValueError):
            pass
    return out


def release_date_from_text(text: str, ref: pd.Period | None) -> date | None:
    """The press-release date printed on page 1 ('Dated: ...', 'Date: ...', embargo line)."""
    head = "\n".join(text.splitlines()[:40])
    cands = []
    for line_block in re.split(r"\n(?=\s*\S)", head):
        if re.search(r"dated|date\s*:|embargo|till|new delhi", line_block, re.I):
            cands += find_dates(line_block)
    # 'Dated the 12 of May, 2017' has the ordinal on the line above; also scan the whole head
    cands += find_dates(head)
    good = [d for d in cands if ref is None or (ref.end_time.date() < d <= (ref + 3).end_time.date())]
    if not good:
        return None
    return max(set(good), key=good.count)


def month_year_in(text: str) -> list[pd.Period]:
    out = []
    for m in re.finditer(MON_ALT + r"[a-z]*[,']?\s*(\d{4})\b", text, re.I):
        k = mnum(m.group(1))
        if k:
            out.append(ym(m.group(2), k))
    return out


def revision_months(text: str) -> dict[str, list[pd.Period]]:
    """Months the release says it revised: {'first': [...], 'final': [...]} (read, not assumed).
    Handles 'the indices for May 2024 have undergone the first revision and those for March 2024 have
    undergone final revision' and lists ('indices for December 2024, January 2025 and February 2025 have
    undergone final revision'). Falls back to the table note 'Indices for the months of Jan'17 and Mar'17
    incorporate updated production data' (older month = final, newer = first) when no sentence is found."""
    # drop page furniture that can split the sentence: page numbers, embargo banners
    t = "\n".join(l for l in text.splitlines() if not re.fullmatch(r"\s*(\d{1,2}|Page\s*\|?\s*\d+.*)\s*", l))
    t = norm_ws(t)
    t = re.sub(r"EMBARGO ADVISORY.*?embargoed against.*?(?:19|20)\d\d\.?", " ", t, flags=re.I)
    t = re.sub(r"(?<=\d{4}) \d (?=have|has)", " ", t)
    out: dict[str, list[pd.Period]] = {"first": [], "second": [], "final": []}
    one = MON_ALT + r"[a-z]*,?\s*\d{4}"
    pat = (r"(?:indices|index|those|figures)\s+for\s+(?:the\s+months?\s+of\s+)?((?:" + one +
           r"(?:\s*,\s*|\s+and\s+)?)+)\s+(?:have|has)\s+(?:also\s+)?undergone\s+(?:the\s+)?(first|second|final)\s+revision")
    for m in re.finditer(pat, t, re.I):
        out[m.group(m.lastindex).lower()] += month_year_in(m.group(1))
    if not out["first"] or not out["final"]:
        m = re.search(r"Indices for the months? of\s+([A-Za-z]{3,9})['’](\d{2})(?:\s+and\s+([A-Za-z]{3,9})['’](\d{2}))?\s+"
                      r"incorporate updated", t, re.I)
        if m:
            ms = [ym(2000 + int(m.group(2)), mnum(m.group(1)))]
            if m.group(3):
                ms.append(ym(2000 + int(m.group(4)), mnum(m.group(3))))
            ms = sorted(ms)
            if len(ms) == 2:
                out["final"] = out["final"] or [ms[0]]
                out["first"] = out["first"] or [ms[1]]
            out["note_fallback"] = [str(x) for x in ms]
    return out


def fy_start(p: pd.Period) -> int:
    return p.year if p.month >= 4 else p.year - 1


def fy_month(fy0: int, mon: int) -> pd.Period:
    return ym(fy0 if mon >= 4 else fy0 + 1, mon)


def numeric_tokens(line: str) -> list[str]:
    return [t for t in line.replace("−", "-").split() if NUMTOK.match(t) or t in ("--", "-")]


def strip_code(line: str) -> str:
    """Drop a leading industry code token ('10', '05', '10-32', '3512', '36-38') when text or a weight follows."""
    toks = line.split()
    if len(toks) >= 2 and re.fullmatch(r"\d{1,4}(?:-\d{1,4})?", toks[0]):
        return " ".join(toks[1:])
    return line


# ----------------------------------------------------------------------------------------------
# IIP series dictionaries (weights identify rows/columns: robust to wrapped descriptions)
# ----------------------------------------------------------------------------------------------
NIC2_NAMES = {
    "10": "Manufacture of food products", "11": "Manufacture of beverages",
    "12": "Manufacture of tobacco products", "13": "Manufacture of textiles",
    "14": "Manufacture of wearing apparel", "15": "Manufacture of leather and related products",
    "16": "Manufacture of wood and products of wood and cork, except furniture",
    "17": "Manufacture of paper and paper products", "18": "Printing and reproduction of recorded media",
    "19": "Manufacture of coke and refined petroleum products",
    "20": "Manufacture of chemicals and chemical products",
    "21": "Manufacture of pharmaceuticals, medicinal chemical and botanical products",
    "22": "Manufacture of rubber and plastics products",
    "23": "Manufacture of other non-metallic mineral products", "24": "Manufacture of basic metals",
    "25": "Manufacture of fabricated metal products, except machinery and equipment",
    "26": "Manufacture of computer, electronic and optical products",
    "27": "Manufacture of electrical equipment", "28": "Manufacture of machinery and equipment n.e.c.",
    "29": "Manufacture of motor vehicles, trailers and semi-trailers",
    "30": "Manufacture of other transport equipment", "31": "Manufacture of furniture",
    "32": "Other manufacturing",
}
USE_NAMES = ["Primary goods", "Capital goods", "Intermediate goods", "Infrastructure/construction goods",
             "Consumer durables", "Consumer non-durables"]

# (series_type, series_code, series) keyed by published weight
W1112_NIC = {5.3025: "10", 1.0354: "11", 0.7985: "12", 3.2913: "13", 1.3225: "14", 0.5021: "15",
             0.1930: "16", 0.8724: "17", 0.6798: "18", 11.7749: "19", 7.8730: "20", 4.9810: "21",
             2.4222: "22", 4.0853: "23", 12.8043: "24", 2.6549: "25", 1.5704: "26", 2.9983: "27",
             4.7653: "28", 4.8573: "29", 1.7763: "30", 0.1311: "31", 0.9415: "32"}
W1112_SECT = {14.3725: ("sector", "05", "Mining"), 77.6332: ("sector", "10-32", "Manufacturing"),
              7.9943: ("sector", "35", "Electricity"), 100.0: ("general", "GEN", "General")}
W1112_SECT_HDR = {14.372472: ("sector", "05", "Mining"), 77.63321: ("sector", "10-32", "Manufacturing"),
                  7.994318: ("sector", "35", "Electricity"), 100.0: ("general", "GEN", "General")}
W1112_USE = {34.048612: 0, 8.223043: 1, 17.221487: 2, 12.338363: 3, 12.839296: 4, 15.329199: 5}

W2223_NIC = {5.679: "10", 1.113: "11", 0.774: "12", 3.275: "13", 1.970: "14", 0.607: "15", 0.239: "16",
             1.375: "17", 0.506: "18", 7.721: "19", 7.813: "20", 5.833: "21", 3.384: "22", 3.519: "23",
             9.198: "24", 2.481: "25", 2.085: "26", 3.175: "27", 5.020: "28", 6.417: "29", 2.089: "30",
             0.283: "31", 1.509: "32"}
W2223_SECT = {11.053: ("sector", "05-08", "Mining & Quarrying"), 76.062: ("sector", "10-32", "Manufacturing"),
              10.865: ("sector", "35", "Electricity & Gas Supply"),
              2.020: ("sector", "36-38", "Water Supply, Sewerage & Waste Management"),
              100.0: ("general", "GEN", "General")}
W2223_SUB = {5.647: ("sub_sector", "05-06", "Fuel Minerals"), 1.995: ("sub_sector", "07", "Metallic Minerals"),
             3.411: ("sub_sector", "08", "Non-Metallic Minerals"),
             2.332: ("sub_sector", "3511R", "Renewable Electricity"),
             7.839: ("sub_sector", "3511N", "Non-Renewable Electricity"),
             10.171: ("sub_sector", "351", "Electricity"), 0.694: ("sub_sector", "352", "Gas Supply"),
             1.059: ("sub_sector", "36", "Water Supply"), 0.961: ("sub_sector", "37-38", "Sewerage & Waste Management")}
W2223_USE = {31.136: 0, 8.082: 1, 22.416: 2, 10.908: 3, 11.311: 4, 16.147: 5}


def series_by_weight(w: float, base: str, line: str = "", dec: int | None = None):
    if base == "2011-12":
        tables = [({k: ("nic2", v, NIC2_NAMES[v]) for k, v in W1112_NIC.items()}, 0.0006),
                  (W1112_SECT, 0.0006), (W1112_SECT_HDR, 0.00002)]
    else:
        tables = [({k: ("nic2", v, NIC2_NAMES[v]) for k, v in W2223_NIC.items()}, 0.0006),
                  (W2223_SECT, 0.0006), (W2223_SUB, 0.0006)]
    for tab, tol in tables:
        if dec is not None:
            tol = max(tol, 0.5 * 10 ** (-dec) + 1e-9)
        for k, v in tab.items():
            if abs(w - k) <= tol:
                return v
    return None


def use_by_weight(w: float, base: str):
    tab, tol = (W1112_USE, 0.0006) if base == "2011-12" else (W2223_USE, 0.0015)
    for k, v in tab.items():
        if abs(w - k) <= tol:
            return ("use_based", f"U{v + 1}", USE_NAMES[v])
    return None


def sect_hdr_by_weight(w: float, base: str):
    if base == "2011-12":
        for k, v in W1112_SECT_HDR.items():
            if abs(w - k) <= 0.0006:
                return v
        return None
    for k, v in W2223_SECT.items():
        if abs(w - k) <= 0.0015:
            return v
    return None


# ----------------------------------------------------------------------------------------------
# PARSE: IIP press-release PDF
# ----------------------------------------------------------------------------------------------
def split_statements(text: str) -> list[tuple[str, list[str]]]:
    """[(statement title line, lines until next statement heading)]."""
    lines = text.splitlines()
    idx = [i for i, l in enumerate(lines) if re.match(r"\s*STATEMENT\s+[IVX]+\b", l)]
    out = []
    for j, i in enumerate(idx):
        end = idx[j + 1] if j + 1 < len(idx) else len(lines)
        out.append((lines[i].strip(), lines[i + 1:end]))
    return out


ROW_LABEL = re.compile(r"^\s*(?:Growth\s+in\s+)?(Apr|May|Jun|Jul|Aug|Sept?|Oct|Nov|Dec|Jan|Feb|Mar)[a-z]*\s*[*#]*\s*[*#]*\s*(?!(?:19|20)\d\d\b)(?=[-−\d]|--|$)", re.I)
RANGE_LABEL = re.compile(r"^\s*(?:Growth\s+in\s+)?[A-Za-z]{3,9}\s*\*?\s*-\s*[A-Za-z]")


def _is_numeric_line(l: str) -> bool:
    toks = l.split()
    return bool(toks) and all(NUMTOK.match(t.replace("−", "-")) or t in ("--", "-", "#", "*") for t in toks)


def parse_fy_table(lines: list[str], qe: pd.Period, base: str, kind: str) -> tuple[list[dict], list[str]]:
    """Statement I (sectoral) / III (use-based): rows = months Apr..Mar, columns = series x financial years
    (series-major; the current FY column is blank after the QE month). Series come from the header weights;
    the number of FY columns from the longest month row (header FY labels are sometimes split or mistyped).
    Returns observations [{series_type, series_code, series, month, index_value | yoy_pub}] and warnings."""
    warn: list[str] = []
    first_row = next((i for i, l in enumerate(lines) if ROW_LABEL.match(l) and numeric_tokens(l)), None)
    if first_row is None:
        return [], [f"{kind}: no month rows"]
    # series order = left-to-right position of the header weights (pdftotext -layout keeps columns aligned;
    # a header can put 'General (100)' on an earlier line than the other weights)
    found = []
    for li, l in enumerate(lines[:first_row]):
        l = re.sub(r"Base[^)]*\)?|20\d\d\s*-\s*\d\d", lambda m: " " * len(m.group(0)), l, flags=re.I)
        for mt in re.finditer(r"(?<![\d.])(\d+\.\d+|100)(?![\d.])", l):
            w = float(mt.group(1))
            sr = use_by_weight(w, base) if kind == "use" else sect_hdr_by_weight(w, base)
            if sr and sr not in [x[2] for x in found]:
                found.append((li, mt.start(), sr))
    lines_used = {x[0] for x in found}
    ser = [x[2] for x in (found if len(lines_used) == 1 else sorted(found, key=lambda x: (x[1], x[0])))]
    if not ser:
        return [], [f"{kind}: header weights not understood"]
    nser = len(ser)
    # pass 1: logical rows (label, section, values) with numeric-only continuation lines merged
    rows, section, i = [], "index", first_row
    while i < len(lines):
        l = lines[i]
        if re.match(r"\s*(Average|Annual\s+Index|Annual\s*$|Cumulative)", l, re.I):
            section = "skip"
        if re.search(r"Growth\s+over|^\s*Growth\s*$|Growth\s+in\s+(Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Jan|Feb|Mar)", l, re.I):
            section = "growth"
        m = ROW_LABEL.match(l)
        if m and section in ("index", "growth") and not RANGE_LABEL.match(l):
            vals = [t for t in numeric_tokens(l[m.end():])]
            j = i + 1
            while j < len(lines) and not ROW_LABEL.match(lines[j]) and _is_numeric_line(lines[j]) and \
                    len(vals) % nser != 0:
                vals += numeric_tokens(lines[j])
                j += 1
            rows.append((mnum(m.group(1)), section, vals, l.strip()[:40]))
            i = j
            continue
        i += 1
    idx_lens = [len(v) for _, sec, v, _ in rows if sec == "index"]
    if not idx_lens:
        return [], warn + [f"{kind}: no index rows"]
    nfy = max(idx_lens) // nser
    if nfy < 1 or max(idx_lens) % nser:
        return [], warn + [f"{kind}: longest row has {max(idx_lens)} values for {nser} series"]
    fys = list(range(fy_start(qe) - nfy + 1, fy_start(qe) + 1))
    obs = []
    for mon, sec, vals, lab in rows:
        if len(vals) == nser * nfy:
            use_fys = fys
        elif len(vals) == nser * (nfy - 1) and sec == "index":
            use_fys = fys[:-1]
        else:
            warn.append(f"{kind}: row '{lab}' has {len(vals)} values (nser={nser}, nfy={nfy})")
            continue
        k = 0
        for sr in ser:
            for fy in use_fys:
                tok = vals[k]
                k += 1
                if tok in ("--", "-"):
                    continue
                rec = dict(series_type=sr[0], series_code=sr[1], series=sr[2], month=fy_month(fy, mon))
                rec["index_value" if sec == "index" else "yoy_pub"] = fnum(tok)
                obs.append(rec)
    # sanity: every month of the QE's FY up to the QE month must be present for each series
    got = {(o["series_code"], o["month"]) for o in obs if "index_value" in o}
    miss = [str(qe)] if any((sr[1], qe) not in got for sr in ser) else []
    if miss:
        warn.append(f"{kind}: QE month missing from table")
    return obs, warn


def _assemble_weight_rows(lines: list[str], base: str, nvals: int, nvals_ok: tuple = ()) -> tuple[list[tuple[tuple, list[float]]], list[str]]:
    """Rows of the form [code] description weight v1..vN, tolerant of wrapped descriptions and of
    value blocks printed on the line above/below the code+weight line. Identifies rows by weight."""
    ok = tuple(nvals_ok) or (nvals,)
    parsed = []
    for l in lines:
        body = strip_code(l.strip())
        toks = numeric_tokens(body)
        wtok, wi = None, None
        for k, t in enumerate(toks):
            if t in ("100", "100.00", "100.000") and re.search(r"general", body, re.I):
                wtok, wi = t, k
                break
            mdec = re.fullmatch(r"\d+\.(\d{2,})", t)
            if mdec and series_by_weight(float(t), base, dec=len(mdec.group(1))) is not None and \
                    (len(mdec.group(1)) >= 3 or len(toks) - k - 1 in ok):
                wtok, wi = t, k
                break
        if wtok is not None:
            after = toks[wi + 1:]
            dec = len(wtok.split(".")[1]) if "." in wtok else None
            parsed.append(dict(kind="w", w=float(wtok), dec=dec, vals=after, line=l, used=False))
        elif toks and len(_trailing_nums(body)) in ok:
            parsed.append(dict(kind="orphan", vals=_trailing_nums(body), line=l, used=False))
        else:
            parsed.append(dict(kind="text", line=l, used=False))
    rows, warn = [], []
    for i, p in enumerate(parsed):
        if p["kind"] != "w":
            continue
        s = series_by_weight(p["w"], base, p["line"], dec=p.get("dec"))
        if s is None:
            continue
        vals = p["vals"]
        if len(vals) not in ok:
            cand = None
            for d in (-1, 1, -2, 2):
                k = i + d
                if 0 <= k < len(parsed) and parsed[k]["kind"] == "orphan" and not parsed[k]["used"]:
                    cand = k
                    break
            if cand is None or vals:
                warn.append(f"row w={p['w']} ({s[2][:30]}) has {len(vals)} values, no orphan block")
                continue
            parsed[cand]["used"] = True
            vals = parsed[cand]["vals"]
        rows.append((s, [None if v in ("--", "-") else fnum(v) for v in vals]))
    return rows, warn


def _trailing_nums(body: str) -> list[str]:
    out = []
    for t in reversed(body.replace("\u2212", "-").split()):
        if NUMTOK.match(t) or t in ("--", "-"):
            out.append(t)
        else:
            break
    return list(reversed(out))


def parse_nic_qe(lines: list[str], qe: pd.Period, base: str) -> tuple[list[dict], list[str]]:
    """Statement II (2-digit): weight, index M-12, index M (QE), cumulative x2, growth M, growth cumulative."""
    # 6 values normally; COVID-era releases (2020-21) print '#' instead of some growth rates (4-5 values)
    rows, warn = _assemble_weight_rows(lines, base, 6, nvals_ok=(6, 5, 4))
    obs = []
    for s, v in rows:
        base_rec = dict(series_type=s[0], series_code=s[1], series=s[2])
        if v[0] is not None:
            obs.append({**base_rec, "month": qe - 12, "index_value": v[0]})
        if v[1] is not None:
            rec = {**base_rec, "month": qe, "index_value": v[1]}
            if len(v) == 6 and v[4] is not None:
                rec["yoy_pub"] = v[4]
            obs.append(rec)
    return obs, warn


def parse_nic_monthly(lines: list[str], base: str) -> tuple[list[dict], list[str]]:
    """Statement IV: weight + the last 12 (or 13) months by NIC-2 and sector."""
    mre = r"\b([A-Z][a-z]{2,3})[-'\u2019](\d{2})\b"
    cands = [l for l in lines[:12] if len(re.findall(mre, l)) >= 2]
    hdr = max(cands, key=lambda l: len(re.findall(mre, l))) if cands else None
    if hdr is None:
        return [], ["stmt IV: month header not found"]
    months = [ym(2000 + int(y), mnum(m)) for m, y in re.findall(mre, hdr)]
    rows, warn = _assemble_weight_rows(lines, base, len(months))
    obs = []
    for s, v in rows:
        for mth, x in zip(months, v):
            if x is not None:
                obs.append(dict(series_type=s[0], series_code=s[1], series=s[2], month=mth, index_value=x))
    return obs, warn


def parse_launch_backseries(text: str) -> list[dict]:
    """12-May-2017 base-revision release, Statement I: Mon-YY rows x (Mining, Manufacturing, Electricity, General)."""
    obs = []
    st = [b for t, b in split_statements(text) if re.search(r"Monthly indices at Sectoral", t, re.I)]
    for l in (st[0] if st else []):
        m = re.match(r"\s*([A-Z][a-z]{2})-(\d{2})\s+(.*)$", l)
        if not m:
            continue
        vals = numeric_tokens(m.group(3))
        if len(vals) != 4:
            continue
        mth = ym(2000 + int(m.group(2)), mnum(m.group(1)))
        for (st_, code, name), v in zip([("sector", "05", "Mining"), ("sector", "10-32", "Manufacturing"),
                                         ("sector", "35", "Electricity"), ("general", "GEN", "General")], vals):
            obs.append(dict(series_type=st_, series_code=code, series=name, month=mth, index_value=fnum(v)))
    return obs


def parse_iip_pdf(path: Path) -> dict:
    return parse_iip_text(pdf_text(path), path.name)


def parse_iip_text(text: str, name: str) -> dict:
    info = dict(file=name, ok=False, warnings=[])
    if len(text.strip()) < 500:
        info["warnings"].append("no text layer (image PDF)")
        return info
    t1 = norm_ws(text[:6000])
    base = "2022-23" if re.search(r"base(?:\s*year)?\s*:?\s*2022-23|2022-23\s*=\s*100", t1, re.I) else "2011-12"
    if re.search(r"2004-05", t1) and not re.search(r"2011-12", t1):
        info["warnings"].append("base 2004-05 release (out of scope)")
        return info
    launch = bool(re.search(r"REVISION OF BASE YEAR OF ALL-INDIA INDEX OF INDUSTRIAL PRODUCTION FROM 2004-05", t1, re.I))
    qe = None
    m = re.search(r"figures\s+for\s+" + MON_ALT + r"[a-z]*,?\s+(\d{4})\s+are\s+quick\s+estimates?", norm_ws(text), re.I) or \
        re.search(r"indices\s+for\s+" + MON_ALT + r"[a-z]*,?\s+(\d{4})\s+are\s+quick\s+estimates?", norm_ws(text), re.I) or \
        re.search(r"for\s+the\s+month\s+of\s+" + MON_ALT + r"[a-z]*,?\s+(\d{4})", t1, re.I)
    if m:
        qe = ym(m.group(2), mnum(m.group(1)))
    if launch:
        qe = ym(2017, 3)
    if qe is None:
        info["warnings"].append("QE month not found")
        return info
    rd = release_date_from_text(text, qe)
    info.update(base=base, qe_month=str(qe), release_date=rd, launch=launch,
                revisions={k: [str(p) for p in v] for k, v in revision_months(text).items() if k != "note_fallback"})
    obs: list[dict] = []
    if launch:
        for o in parse_launch_backseries(text):
            obs.append({**o, "table": "launch_stmt_I"})
    for title, body in split_statements(text):
        tu = title.upper()
        try:
            if re.search(r"SECTORAL", tu) and "MONTHLY INDICES" not in tu and "GROWTH" not in tu:
                o, w = parse_fy_table(body, qe, base, "sector")
                obs += [{**x, "table": "stmt_I"} for x in o]
                info["warnings"] += w
            elif re.search(r"USE-?\s*BASED", tu) and "ANNUAL" not in tu:
                o, w = parse_fy_table(body, qe, base, "use")
                obs += [{**x, "table": "stmt_III"} for x in o]
                info["warnings"] += w
            elif re.search(r"MONTHLY INDEX OF INDUSTRIAL PRODUCTION|LAST 1[23] MONTHS", tu):
                o, w = parse_nic_monthly(body, base)
                obs += [{**x, "table": "stmt_IV"} for x in o]
                info["warnings"] += w
            elif re.search(r"2-DIGIT|NIC 2 DIGIT|MINING & QUARRYING|ELECTRICITY & GAS|WATER SUPPLY", tu):
                o, w = parse_nic_qe(body, qe, base)
                obs += [{**x, "table": "stmt_II"} for x in o]
                info["warnings"] += w
        except Exception as e:  # noqa: BLE001
            info["warnings"].append(f"{title[:40]}: {type(e).__name__} {e}")
    info["obs"] = obs
    info["ok"] = bool(obs) and rd is not None
    if rd is None:
        info["warnings"].append("release date not found")
    return info


# ----------------------------------------------------------------------------------------------
# PARSE: IIP workbook attached to releases (full-history vintage snapshot)
# ----------------------------------------------------------------------------------------------
def _xlsx_series(label, desc, base: str):
    lab = str(label).strip() if label is not None else ""
    d = (str(desc).strip().lower() if desc else "")
    if lab == "General":
        return ("general", "GEN", "General")
    if base == "2011-12":
        if lab in ("Mining", "Manufacturing", "Electricity"):
            return {"Mining": ("sector", "05", "Mining"), "Manufacturing": ("sector", "10-32", "Manufacturing"),
                    "Electricity": ("sector", "35", "Electricity")}[lab]
    else:
        secs = {"Mining & Quarrying": ("sector", "05-08", "Mining & Quarrying"),
                "Manufacturing": ("sector", "10-32", "Manufacturing"),
                "Electricity & Gas Supply": ("sector", "35", "Electricity & Gas Supply"),
                "Water Supply, Sewerage & Waste Management": ("sector", "36-38", "Water Supply, Sewerage & Waste Management")}
        if lab in secs:
            return secs[lab]
        for key, v in [("fuel minerals", W2223_SUB[5.647]), ("metallic minerals incl", W2223_SUB[1.995]),
                       ("non-metallic minerals", W2223_SUB[3.411]), ("non-renewable electricity", W2223_SUB[7.839]),
                       ("renewable electricity", W2223_SUB[2.332]), ("gas supply", W2223_SUB[0.694]),
                       ("sewerage", W2223_SUB[0.961]), ("water supply", W2223_SUB[1.059])]:
            if d.startswith(key):
                return v
        if d == "electricity":
            return W2223_SUB[10.171]
    if re.fullmatch(r"\d{2}", lab) or (isinstance(label, (int, float)) and 10 <= int(label) <= 32):
        code = f"{int(float(lab)):02d}"
        if code in NIC2_NAMES:
            return ("nic2", code, NIC2_NAMES[code])
    for i, n in enumerate(USE_NAMES):
        if lab.lower().replace(" ", "") == n.lower().replace(" ", ""):
            return ("use_based", f"U{i + 1}", n)
    return None


def parse_iip_xlsx(path: Path) -> tuple[str, list[dict]]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    obs = []
    base = "2022-23" if "2022-23" in path.name else "2011-12"
    for sn in ("NIC 2d, sectoral monthly", "UBC monthly"):
        if sn not in wb.sheetnames:
            continue
        rows = list(wb[sn].iter_rows(values_only=True))
        section, months, first_col = None, [], 0
        for r in rows:
            if r is None:
                continue
            dcols = [(j, c) for j, c in enumerate(r) if isinstance(c, datetime)]
            if len(dcols) >= 6:
                months = [(j, ym(c.year, c.month)) for j, c in dcols]
                section = "index" if section is None else "growth"
                continue
            if section is None or not months:
                continue
            if sn.startswith("UBC"):
                s = _xlsx_series(r[0], None, base)
            else:
                s = _xlsx_series(r[0], r[1], base)
            if s is None:
                continue
            for j, mth in months:
                v = r[j] if j < len(r) else None
                if isinstance(v, (int, float)):
                    rec = dict(series_type=s[0], series_code=s[1], series=s[2], month=mth, table=sn)
                    if section == "index":
                        rec["index_value"] = round(float(v), 1)
                    else:
                        rec["yoy_pub"] = round(float(v), 1)
                    obs.append(rec)
    return base, obs



# ----------------------------------------------------------------------------------------------
# PARSE: Core sector (ICI) press releases
# ----------------------------------------------------------------------------------------------
ICI_W = {
    "2011-12": {10.33: "Coal", 8.98: "Crude Oil", 6.88: "Natural Gas", 28.04: "Refinery Products",
                2.63: "Fertilizers", 17.92: "Steel", 5.37: "Cement", 19.85: "Electricity", 100.0: "Overall Index"},
    "2022-23": {5.596: "Coal", 3.841: "Natural Gas", 7.43: "Crude Oil", 22.572: "Refinery Products",
                2.731: "Fertilizers", 17.584: "Steel", 4.41: "Cement", 30.932: "Electricity", 4.905: "Iron Ore",
                100.0: "Overall Index"},
    # 2004-05 base: weights are the items' IIP weights (sum 37.90), used by releases up to Apr-2017
    "2004-05": {4.379: "Coal", 5.216: "Crude Oil", 1.708: "Natural Gas", 5.939: "Refinery Products",
                1.254: "Fertilizers", 6.684: "Steel", 2.406: "Cement", 10.316: "Electricity", 37.903: "Overall Index"},
}
ICI_ORDER_1112 = ["Coal", "Crude Oil", "Natural Gas", "Refinery Products", "Fertilizers", "Steel", "Cement",
                  "Electricity", "Overall Index"]


def ocr_pdf(path: Path) -> str:
    """Text of an image-only PDF via tesseract (layout-preserving psm 6). Cached next to the PDF."""
    cache = path.with_suffix(".ocr.txt")
    if cache.exists():
        return cache.read_text()
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", "300", "-gray", str(path), f"{td}/p"], check=True)
        pages = sorted(Path(td).glob("p*.pgm")) or sorted(Path(td).glob("p*"))
        txt = []
        for pg in pages:
            r = subprocess.run(["tesseract", str(pg), "-", "--psm", "6"], capture_output=True)
            txt.append(r.stdout.decode("utf-8", errors="ignore"))
    out = "\n\f\n".join(txt)
    cache.write_text(out)
    return out


def ici_base_of(text: str) -> str:
    t = norm_ws(text[:5000])
    if re.search(r"2022-23", t):
        return "2022-23"
    if re.search(r"2011-12", t):
        return "2011-12"
    if re.search(r"2004-05", t):
        return "2004-05"
    return "2011-12"


def _ici_cols(weight_line: str, base: str) -> list[str] | None:
    vals = [fnum(t) for t in numeric_tokens(weight_line)]
    tab = ICI_W[base]
    cols = []
    for v in vals:
        hit = [n for w, n in tab.items() if abs(v - w) <= 0.006 or (w == 100.0 and abs(v - 100) < 0.01)]
        if not hit:
            return None
        cols.append(hit[0])
    return cols if len(set(cols)) == len(cols) else None


def parse_ici_text(text: str, ref: pd.Period | None) -> dict:
    lines = text.splitlines()
    base = ici_base_of(text)
    # the weights row identifies the base unambiguously (FY labels like '2022-23' also occur in 2011-12 releases)
    for l in lines:
        if re.match(r"\s*Weights?\b", l) and len(numeric_tokens(l)) >= 8:
            for b_ in ("2011-12", "2022-23", "2004-05"):
                if _ici_cols(l, b_):
                    base = b_
                    break
            break
    info = dict(base=base, warnings=[])
    # reference month
    t = norm_ws(text[:6000])
    m = re.search(r"(?:FOR|for)\s+(?:the\s+month\s+of\s+)?" + MON_ALT + r"[a-z]*,?\s+(\d{4})", t) or \
        re.search(r"Base(?:\s+Year)?:?\s*20\d\d-\d\d\s*=\s*100\)?\s+" + MON_ALT + r"[a-z]*,?\s+(\d{4})", t, re.I)
    qm = ym(m.group(2), mnum(m.group(1))) if m else ref
    if ref is not None:
        qm = ref            # OEA archive file names carry the reference month
    info["ref_month"] = qm
    info["release_date"] = release_date_from_text(text, qm)
    nxt = re.search(r"Release of the index for (?:the month of )?" + MON_ALT + r"[a-z]*,?\s+(\d{4})\s+will be on\s+(.{0,60})",
                    norm_ws(text), re.I)
    if nxt:
        d = find_dates(nxt.group(3))
        if d:
            info["next_release"] = (str(ym(nxt.group(2), mnum(nxt.group(1)))), d[0])
    fm = re.search(r"final (?:growth rate|index)[^.]{0,80}?(?:for|of)\s+(?:the\s+month\s+of\s+)?" + MON_ALT +
                   r"[a-z]*,?\s+(\d{4})", norm_ws(text), re.I)
    info["final_month"] = ym(fm.group(2), mnum(fm.group(1))) if fm else None
    prov_note = set()
    for pm in re.finditer(r"Data for ([^.]{0,140}?)\s+(?:is|are)\s+provisional", norm_ws(text), re.I):
        prov_note |= set(month_year_in(pm.group(1)))
    info["provisional_note"] = sorted(str(x) for x in prov_note)
    obs = []
    cols, table = None, None
    for i, l in enumerate(lines):
        if re.match(r"\s*Weights?\b", l) and len(numeric_tokens(l)) >= 8:
            cols = _ici_cols(l, base)
            # nearest table heading above the weight row: a bare 'Index' line, 'Growth Rates ...', or 'Table n: ...'
            # search above the column-header block (which itself ends in '... Overall / Index')
            top = next((k for k in range(i - 1, max(-1, i - 8), -1) if re.match(r"\s*Sector\b", lines[k])), i)
            heads = [lines[k].strip() for k in range(max(0, top - 8), top)
                     if re.match(r"^(Index|INDEX)$|^(Growth Rates?|GROWTH RATES?)\b|^Table\s+\d+\s*:", lines[k].strip())]
            table = "growth" if heads and re.search(r"growth", heads[-1], re.I) else "index"
            if cols is None:
                info["warnings"].append(f"weights row not understood: {l.strip()[:80]}")
            continue
        mm = re.match(r"\s*([A-Z][a-z]{2})[-'’ ]\s?(\d{2})\s*(\*|\(P\))?\s+(.*)$", l)
        if not mm or cols is None:
            continue
        vals = numeric_tokens(mm.group(4))
        if len(vals) != len(cols):
            continue
        mth = ym(2000 + int(mm.group(2)), mnum(mm.group(1)))
        prov = bool(mm.group(3)) or mth in prov_note
        for c, v in zip(cols, vals):
            if v in ("--", "-"):
                continue
            rec = dict(series=c, month=mth, provisional=prov)
            rec["index_value" if table == "index" else "yoy_pub"] = fnum(v)
            obs.append(rec)
    info["obs"] = obs
    return info


def parse_ici_pdf(path: Path) -> dict:
    text = pdf_text(path)
    method = "pdftotext"
    if len(text.strip()) < 500:
        text = ocr_pdf(path)
        method = "ocr"
    m = re.search(r"IPR_(\d{4})_(\d{2})", path.name)
    ref = ym(m.group(1), m.group(2)) if m else None
    info = parse_ici_text(text, ref)
    info.update(file=path.name, parse_method=method)
    return info


# ICI releases whose OEA PDF has no text layer (or lacks the table), recovered from PIB. Keys: reference
# month -> PIB page. Located by searching pib.gov.in (2026-09-29). Not found: May/Sep/Dec-2016,
# Jan/Mar-2017 (base 2004-05) and Jul-2026 (base 2022-23, image-only OEA PDF; OCR of its tables is unreliable).
PIB_ICI = {
    "2016-06": "https://pib.gov.in/newsite/PrintRelease.aspx?relid=148122",
    "2017-02": "https://pib.gov.in/newsite/PrintRelease.aspx?relid=160316",
    "2017-04": "https://pib.gov.in/newsite/PrintRelease.aspx?relid=163282",
    "2024-12": "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2098037",
}


def _pib_file(url: str) -> str:
    m = re.search(r"(relid|PRID)=(\d+)", url, re.I)
    return f"PIB_{m.group(1)}{m.group(2)}.html"


def fetch_pib_ici(refresh: bool) -> None:
    out = RAW_CORE / "press_releases" / "pib"
    out.mkdir(parents=True, exist_ok=True)
    for ref, url in PIB_ICI.items():
        f = out / _pib_file(url)
        if refresh or not f.exists():
            f.write_bytes(http_get(url))
            time.sleep(SLEEP)


def pib_posted_date(html: str) -> date | None:
    m = re.search(r"Posted On:\s*(\d{1,2}\s+[A-Z]{3}\s+\d{4})", html)
    if m:
        return datetime.strptime(m.group(1).title(), "%d %b %Y").date()
    m = re.search(r"\b(\d{1,2})-([A-Za-z]{3,9})-(\d{4})\b", html)      # old 'newsite' pages: '31-May-2017'
    if m and mnum(m.group(2)):
        return date(int(m.group(3)), mnum(m.group(2)), int(m.group(1)))
    return None


def parse_ici_xlsx(path: Path) -> tuple[str, list[dict]]:
    """OEA workbook: monthly index (and growth) for every industry, latest vintage as of the file date."""
    import openpyxl
    base = "2022-23" if "2022_23" in path.name else "2011-12"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    obs = []
    names = {"coal": "Coal", "crude": "Crude Oil", "natural": "Natural Gas", "refinery": "Refinery Products",
             "fertili": "Fertilizers", "steel": "Steel", "cement": "Cement", "electricity": "Electricity",
             "iron ore": "Iron Ore", "overall": "Overall Index", "core": "Overall Index", "ici": "Overall Index"}
    for ws in wb.worksheets:
        rows = [r for r in ws.iter_rows(values_only=True)]
        title = " ".join(str(c) for r in rows[:6] for c in r if c)
        kind = "growth" if re.search(r"growth", ws.title + " " + title, re.I) else "index"
        # layout A: a header row of industry names, then one row per month
        hdr_i, colmap = None, {}
        for i, r in enumerate(rows[:15]):
            cm = {}
            for j, c in enumerate(r):
                if isinstance(c, str):
                    cl = re.sub(r"^(index|growth)\s+of\s+|^petroleum\s+", "", re.sub(r"\s+", " ", c.strip().lower()))
                    for k, v in names.items():
                        if cl.startswith(k) and v not in cm.values():
                            cm[j] = v
                            break
            if len(cm) >= 8:
                hdr_i, colmap = i, cm
                break
        if hdr_i is None:
            continue
        for r in rows[hdr_i + 1:]:
            lab = r[0] if r else None
            mth = None
            if isinstance(lab, datetime):
                mth = ym(lab.year, lab.month)
            elif isinstance(lab, str):
                mm = re.match(r"\s*([A-Za-z]{3,9})[-' ,]*\s*(\d{2,4})\s*\*?\s*$", lab)
                if mm and mnum(mm.group(1)):
                    y = int(mm.group(2))
                    mth = ym(y + 2000 if y < 100 else y, mnum(mm.group(1)))
            if mth is None:
                continue
            for j, name in colmap.items():
                v = r[j] if j < len(r) else None
                if isinstance(v, (int, float)):
                    rec = dict(series=name, month=mth, sheet=ws.title)
                    rec["index_value" if kind == "index" else "yoy_pub"] = round(float(v), 1)
                    obs.append(rec)
    return base, obs



def yoy_consistency(ob: pd.DataFrame, key: list[str]) -> dict:
    """Within each release: published y-o-y vs index(M)/index(M-12)-1 from the same release (rounding tolerance 0.15)."""
    iv = ob.dropna(subset=["index_value"]).drop_duplicates(key)
    gp = ob.dropna(subset=["yoy_pub"]).drop_duplicates(key)
    prev = iv.copy()
    prev["month"] = prev["month"] + 12
    m = gp[key + ["yoy_pub"]].merge(iv[key + ["index_value"]], on=key).merge(
        prev[key + ["index_value"]].rename(columns={"index_value": "idx_prev"}), on=key)
    m = m[m["idx_prev"] > 0]
    d = ((m["index_value"] / m["idx_prev"] - 1) * 100 - m["yoy_pub"]).abs()
    bad = m[d > 0.15]
    return dict(n_checked=int(len(m)), n_within_0p15=int((d <= 0.15).sum()),
                worst=[{k: str(v) for k, v in r.items()} for r in bad.assign(diff=d[d > 0.15]).sort_values("diff", ascending=False)
                       .head(8).to_dict("records")])


# ----------------------------------------------------------------------------------------------
# BUILD: IIP vintage table
# ----------------------------------------------------------------------------------------------
def _api_series(r: dict, base: str):
    typ, cat, sub = r.get("type"), (r.get("category") or "").strip(), (r.get("sub_category") or "").strip()
    if typ == "General":
        return ("general", "GEN", "General")
    if typ == "Use-based category":
        k = re.sub(r"[^a-z]", "", cat.lower())
        for i, n in enumerate(USE_NAMES):
            if re.sub(r"[^a-z]", "", n.lower()) == k:
                return ("use_based", f"U{i + 1}", n)
        return None
    if not sub:
        return _xlsx_series(cat, None, base) if cat != "Electricity" or base == "2011-12" else None
    if cat == "Manufacturing":
        sl = sub.lower()
        if "pharmaceutical" in sl:
            return ("nic2", "21", NIC2_NAMES["21"])
        for code, name in NIC2_NAMES.items():
            if sl.startswith(name.lower()[:28]):
                return ("nic2", code, name)
        return None
    return _xlsx_series(None, sub, base)


def load_iip_api() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW_IIP / "api").glob("iip_monthly_base*_p*.json")):
        base = re.search(r"base(\d{4}-\d{2})", f.name).group(1)
        for r in json.loads(f.read_text()).get("data") or []:
            s = _api_series(r, base)
            if s is None:
                continue
            try:
                idx = float(r["index"])
            except (TypeError, ValueError):
                continue
            try:
                g = float(r["growth_rate"])
            except (TypeError, ValueError):
                g = None
            rows.append(dict(base_year=base, series_type=s[0], series_code=s[1], series=s[2],
                             month=ym(r["year"], MONTHS[r["month"].lower()]), index_value=idx, yoy_pub=g))
    d = pd.DataFrame(rows).drop_duplicates(["base_year", "series_code", "month"])
    return d


def _file_url(path: str) -> str:
    return MOSPI_SITE + urllib.parse.quote(path)


def collect_iip_releases() -> tuple[list[dict], list[dict]]:
    """Parse every cached IIP release. Returns (release records with obs, problems)."""
    rel_rows = load_release_index()
    seen_pdf, releases, problems = set(), [], []
    for r in rel_rows:
        pdfs = [p for p in r["files"] if p.lower().endswith(".pdf")]
        xls = [p for p in r["files"] if p.lower().endswith(".xlsx") and "indices" in Path(p).name.lower()]
        if not pdfs:
            continue
        pdf = pdfs[0]
        if Path(pdf).name in seen_pdf:
            continue
        seen_pdf.add(Path(pdf).name)
        path = RAW_IIP / "press_releases" / Path(pdf).name
        if not path.exists():
            problems.append(dict(file=Path(pdf).name, problem="not downloaded", title=r["title"]))
            continue
        info = parse_iip_pdf(path)
        if not info.get("qe_month"):
            problems.append(dict(file=path.name, problem="; ".join(info["warnings"]) or "unparsed", title=r["title"]))
            continue
        qe = pd.Period(info["qe_month"], freq="M")
        rd, basis = info.get("release_date"), "printed_on_release"
        if rd is None and r.get("listed_date"):
            ld = pd.Timestamp(r["listed_date"]).date()
            if qe.end_time.date() < ld <= (qe + 3).end_time.date():
                rd, basis = ld, "mospi_listing_date"
        rec = dict(base=info["base"], qe_month=qe, release_date=rd, release_date_basis=basis,
                   launch=info.get("launch", False), revisions=info.get("revisions", {}),
                   pdf=path.name, pdf_url=_file_url(pdf), title=r["title"], warnings=info["warnings"],
                   obs=info.get("obs", []), obs_source="pdf")
        m = re.search(r"Release of the Index for\s+" + MON_ALT + r"[a-z]*,?\s+(\d{4})\s+will be on\s+(.{0,50})",
                      norm_ws(pdf_text(path)), re.I)
        if m and find_dates(m.group(3)):
            rec["announces"] = (ym(m.group(2), mnum(m.group(1))), find_dates(m.group(3))[0])
        if xls:
            xp = RAW_IIP / "press_releases" / Path(xls[0]).name
            if xp.exists():
                b2, xobs = parse_iip_xlsx(xp)
                if xobs:
                    rec.update(obs=xobs, obs_source="xlsx", xlsx=xp.name, xlsx_url=_file_url(xls[0]), base=b2)
        releases.append(rec)
    # releases missing from the MoSPI archive: PIB copies (PDF attachment if present, else the HTML body)
    have = {(r["base"], r["qe_month"]) for r in releases}
    pib_dir = RAW_IIP / "press_releases" / "pib"
    for ref, prid in PIB_IIP_PRIDS.items():
        h = pib_dir / f"PIB_PRID{prid}.html"
        if not h.exists():
            problems.append(dict(file=h.name, problem="PIB page not downloaded", title=ref))
            continue
        txt, posted = pib_html_text(h)
        pdfs = sorted(pib_dir.glob(f"PIB_PRID{prid}__*.pdf"))
        info = parse_iip_pdf(pdfs[0]) if pdfs else parse_iip_text(txt, h.name)
        if not info.get("qe_month") or pd.Period(info["qe_month"], freq="M") != pd.Period(ref, freq="M"):
            problems.append(dict(file=h.name, problem=f"PIB copy parsed as {info.get('qe_month')} not {ref}", title=ref))
            continue
        qe = pd.Period(ref, freq="M")
        if (info["base"], qe) in have:
            continue
        pdf_url = (re.findall(r"https?://static\.pib\.gov\.in/WriteReadData/specificdocs/documents/[^\"' ]+\.pdf",
                              h.read_text(errors="ignore")) or [None])[0] if pdfs else None
        revs = info.get("revisions") or {k: [str(p) for p in v] for k, v in revision_months(txt).items() if k != "note_fallback"}
        if not (revs.get("first") or revs.get("final")):
            revs = {k: [str(p) for p in v] for k, v in revision_months(txt).items() if k != "note_fallback"}
        releases.append(dict(base=info["base"], qe_month=qe, release_date=posted, release_date_basis="pib_posted_on",
                             launch=False, revisions=revs, pdf=(pdfs[0].name if pdfs else h.name),
                             pdf_url=pdf_url or PIB_PAGE.format(prid), title=f"PIB PRID {prid}",
                             warnings=info["warnings"], obs=info.get("obs", []), obs_source="pib_pdf" if pdfs else "pib_html"))
    # release dates missing on the document: use the date the previous release announced
    ann = {}
    for r in releases:
        if r.get("announces"):
            ann[(r["base"], r["announces"][0])] = r["announces"][1]
            ann[("any", r["announces"][0])] = r["announces"][1]
    for r in releases:
        if r["release_date"] is None:
            d = ann.get((r["base"], r["qe_month"])) or ann.get(("any", r["qe_month"]))
            if d:
                r["release_date"], r["release_date_basis"] = d, "announced_in_previous_release"
            else:
                problems.append(dict(file=r["pdf"], problem="release date unknown", title=r["title"]))
    releases = [r for r in releases if r["release_date"] is not None]
    # one release per (base, QE month): prefer the richer parse
    best = {}
    for r in releases:
        k = (r["base"], r["qe_month"])
        if k not in best or len(r["obs"]) > len(best[k]["obs"]):
            best[k] = r
    return sorted(best.values(), key=lambda r: (r["release_date"], r["base"])), problems


def build_iip(fetch_day: date) -> tuple[pd.DataFrame, dict]:
    releases, problems = collect_iip_releases()
    first_of_base = {}
    for r in releases:
        first_of_base.setdefault(r["base"], r)
    for r in first_of_base.values():
        r["launch"] = True      # 2011-12: 12-May-2017 base revision; 2022-23: 01-Jun-2026 first release
    api = load_iip_api()
    obs_rows = []
    for r in releases:
        src_url = r.get("xlsx_url") or r["pdf_url"]
        for o in r["obs"]:
            obs_rows.append(dict(base_year=r["base"], series_type=o["series_type"], series_code=o["series_code"],
                                 series=o["series"], month=o["month"], index_value=o.get("index_value"),
                                 yoy_pub=o.get("yoy_pub"), release_date=pd.Timestamp(r["release_date"]),
                                 release_ref_month=r["qe_month"], table=o.get("table"),
                                 source_url=src_url, source_file=r.get("xlsx") or r["pdf"],
                                 release_date_basis=r["release_date_basis"], launch=r["launch"],
                                 first_rev=[pd.Period(x, freq="M") for x in r["revisions"].get("first", [])],
                                 final_rev=[pd.Period(x, freq="M") for x in r["revisions"].get("final", [])]))
    ob = pd.DataFrame(obs_rows)
    key = ["base_year", "series_code", "month", "release_date"]
    yoy_chk = yoy_consistency(ob, ["base_year", "series_code", "release_date", "month"])
    # growth published in the same release (any table)
    g = ob.dropna(subset=["yoy_pub"]).drop_duplicates(key)[key + ["yoy_pub"]]
    iv = ob.dropna(subset=["index_value"]).copy()
    # within one release the same value can be printed by several statements; report conflicts, keep the
    # monthly-table print (stmt I/III/IV or workbook) over the QE-only statement II
    iv["pri"] = iv["table"].map({"stmt_II": 1}).fillna(0)
    conflicts = iv.groupby(key)["index_value"].nunique()
    n_conflict = int((conflicts > 1).sum())
    conflict_examples = [list(map(str, k)) for k in conflicts[conflicts > 1].index[:10]]
    iv = iv.sort_values(key + ["pri"]).drop_duplicates(key)
    iv = iv.drop(columns=["yoy_pub"]).merge(g, on=key, how="left")

    out = []
    for (base, code, mth), grp in iv.groupby(["base_year", "series_code", "month"], sort=False):
        grp = grp.sort_values("release_date")
        last_val, seen = None, set()
        for _, o in grp.iterrows():
            lag = (o["release_ref_month"] - mth).n
            if o["launch"] and lag > 0:
                vint = "base_launch"
            elif lag == 0:
                vint = "quick_estimate"
            elif mth in o["first_rev"]:
                vint = "first_revision"
            elif mth in o["final_rev"]:
                vint = "final_revision"
            else:
                vint = "restated"
            changed = last_val is None or abs(o["index_value"] - last_val) > 1e-9
            keep = (vint != "restated" and vint not in seen) or (vint == "restated" and changed)
            if keep:
                seen.add(vint)
                out.append(dict(o, vintage=vint, vintage_lag_months=lag, value_changed=bool(changed) if last_val is not None else None,
                                value_source="press_release_" + ("xlsx" if str(o["source_file"]).endswith(".xlsx") else "pdf")))
                last_val = o["index_value"]
    vt = pd.DataFrame(out)

    # --- API latest vintage and schedule-inferred final revisions --------------------------------
    # Releases declare which month they finalise; before Statement IV (Jun-2020) the NIC-2 revisions were
    # not printed, so for series/months without a final print we date the final value (current API value,
    # checked against every later print) to the release that finalised it -> vintage 'final_revision_inferred'.
    finals = {}
    for r in releases:
        for x in r["revisions"].get("final", []):
            finals.setdefault((r["base"], pd.Period(x, freq="M")), pd.Timestamp(r["release_date"]))
    # a release missing from every archive (Sep-2017 QE): its date was announced by the previous release and
    # the policy of the day finalised month QE-3 in it
    parsed = {(r["base"], r["qe_month"]) for r in releases}
    for r in releases:
        ann = r.get("announces")
        if ann and (r["base"], ann[0]) not in parsed and pd.Timestamp(ann[1]) <= pd.Timestamp(fetch_day):
            fin = [pd.Period(x, freq="M") for x in r["revisions"].get("final", [])]
            if fin:
                lag = (r["qe_month"] - max(fin)).n
                finals.setdefault((r["base"], ann[0] - lag), pd.Timestamp(ann[1]))
    kcols = ["base_year", "series_code", "month"]
    has_final = set(map(tuple, vt.loc[vt["vintage"] == "final_revision", kcols].itertuples(index=False, name=None)))
    prints = {k: g.sort_values("release_date") for k, g in vt.groupby(kcols)}
    api_rows = []
    cmp = dict(n_final_prints_compared=0, n_final_prints_equal_api=0, api_disagrees_with_final_print=[],
               n_last_print_compared=0, n_last_print_equal_api=0)
    meta_series = {k: (g["series_type"].iloc[0], g["series"].iloc[0]) for k, g in vt.groupby(kcols)}
    api_idx = {(a["base_year"], a["series_code"], a["month"]): a for _, a in api.iterrows()}
    for a in api.itertuples(index=False):
        meta_series.setdefault((a.base_year, a.series_code, a.month), (a.series_type, a.series))
    all_keys = set(prints) | set(api_idx)
    for k in all_keys:
        a = api_idx.get(k)
        g = prints.get(k)
        st, sname = meta_series[k]
        api_url = MOSPI_API + f"?base_year={k[0]}&year={k[2].year}&type=All&Format=JSON"
        if g is not None and a is not None:
            cmp["n_last_print_compared"] += 1
            cmp["n_last_print_equal_api"] += int(abs(g["index_value"].iloc[-1] - a["index_value"]) < 1e-9)
        if k in has_final:
            if a is not None:
                fv = g.loc[g["vintage"] == "final_revision", "index_value"].iloc[0]
                cmp["n_final_prints_compared"] += 1
                if abs(fv - a["index_value"]) < 1e-9:
                    cmp["n_final_prints_equal_api"] += 1
                elif len(cmp["api_disagrees_with_final_print"]) < 40:
                    cmp["api_disagrees_with_final_print"].append(
                        dict(base=k[0], series_code=k[1], month=str(k[2]), final_print=float(fv), api=a["index_value"]))
            continue   # a printed final wins over the API (the API was seen lagging a final revision)
        fr = finals.get((k[0], k[2]))
        cand = a["index_value"] if a is not None else None
        if fr is not None and fr <= pd.Timestamp(fetch_day):
            later = g[g["release_date"] > fr]["index_value"] if g is not None else pd.Series(dtype=float)
            if cand is None and len(later):
                cand = later.iloc[0]
            if cand is not None and all(abs(v - cand) < 1e-9 for v in later) and \
                    (g is None or not ((g["release_date"] == fr).any())):
                api_rows.append(dict(base_year=k[0], series_type=st, series_code=k[1], series=sname, month=k[2],
                                     index_value=cand, yoy_pub=(a["yoy_pub"] if a is not None else None),
                                     release_date=fr, release_ref_month=pd.NaT, table="api" if a is not None else "later_print",
                                     vintage="final_revision_inferred", vintage_lag_months=None, value_changed=None,
                                     source_url=api_url, source_file="data/raw/iip/api",
                                     release_date_basis="revision_schedule: the release of this date announced the final "
                                     "revision of this month but did not print this series; value = current API value, "
                                     "equal to every later print held",
                                     value_source="api_latest" if a is not None else "later_print"))
                continue
        if a is None:
            continue
        if g is not None and abs(g["index_value"].iloc[-1] - a["index_value"]) < 1e-9:
            continue
        if g is None and k[0] == "2011-12" and k[2] < pd.Period("2017-03", freq="M"):
            rd, basis = pd.Timestamp("2017-05-12"), "base_launch_date (2011-12 series first published 2017-05-12; value = current API)"
        else:
            rd, basis = pd.Timestamp(fetch_day), "fetch_date (value differs from every print held; only known to be public by the fetch date)"
        api_rows.append(dict(base_year=k[0], series_type=st, series_code=k[1], series=sname, month=k[2],
                             index_value=a["index_value"], yoy_pub=a["yoy_pub"], release_date=rd,
                             release_ref_month=pd.NaT, table="api", vintage="latest", vintage_lag_months=None,
                             value_changed=None if g is None else True, source_url=api_url,
                             source_file="data/raw/iip/api", release_date_basis=basis, value_source="api_latest"))
    vt = pd.concat([vt, pd.DataFrame(api_rows)], ignore_index=True)
    # drop 'restated' prints that merely repeat the value already known (e.g. after an inferred final)
    vt = vt.sort_values(kcols + ["release_date"]).reset_index(drop=True)
    prev_val = vt.groupby(kcols)["index_value"].shift(1)
    vt = vt[~((vt["vintage"] == "restated") & (prev_val - vt["index_value"]).abs().lt(1e-9))].reset_index(drop=True)

    # y-o-y: published in the same release if available; else computed from values known at that release date
    vt["yoy_source"] = vt["yoy_pub"].notna().map({True: "published", False: None})
    vt = vt.sort_values("release_date").reset_index(drop=True)
    look = vt[["base_year", "series_code", "month", "release_date", "index_value"]]
    comp = []
    by = {k: g.sort_values("release_date") for k, g in look.groupby(["base_year", "series_code", "month"])}
    for i, r in vt[vt["yoy_pub"].isna()].iterrows():
        prev = by.get((r["base_year"], r["series_code"], r["month"] - 12))
        if prev is None:
            continue
        known = prev[prev["release_date"] <= r["release_date"]]
        if known.empty:
            continue
        comp.append((i, round((r["index_value"] / known["index_value"].iloc[-1] - 1) * 100, 1)))
    for i, v in comp:
        vt.at[i, "yoy_pub"] = v
        vt.at[i, "yoy_source"] = "computed_pit"
    vt = vt.rename(columns={"yoy_pub": "yoy_pct"})
    vt["month"] = vt["month"].dt.to_timestamp()
    vt["release_ref_month"] = pd.to_datetime(vt["release_ref_month"].astype(str).where(vt["release_ref_month"].notna()), errors="coerce")
    vt = vt[vt["month"] >= "2016-01-01"]
    cols = ["month", "base_year", "series_type", "series_code", "series", "index_value", "yoy_pct", "yoy_source",
            "release_date", "vintage", "vintage_lag_months", "release_ref_month", "value_changed", "value_source",
            "release_date_basis", "source_url", "source_file", "table"]
    vt = vt[cols].sort_values(["base_year", "series_type", "series_code", "month", "release_date"]).reset_index(drop=True)
    vt["vintage_lag_months"] = vt["vintage_lag_months"].astype("Int64")
    vt = vt.rename(columns={"table": "parse_table"})
    meta = dict(releases=[dict(base=r["base"], qe_month=str(r["qe_month"]), release_date=str(r["release_date"]),
                               release_date_basis=r["release_date_basis"], first_revision=r["revisions"].get("first"),
                               final_revision=r["revisions"].get("final"), source=r.get("xlsx") or r["pdf"],
                               n_obs=len(r["obs"]), warnings=r["warnings"][:5]) for r in releases],
                problems=problems, api_vs_last_print=cmp, within_release_conflicts=n_conflict, yoy_consistency=yoy_chk,
                conflict_examples=conflict_examples)
    return vt, meta



# ----------------------------------------------------------------------------------------------
# BUILD: Core-sector (ICI) vintage table
# ----------------------------------------------------------------------------------------------
def collect_ici_releases() -> tuple[list[dict], list[dict]]:
    rels, problems = [], []
    for f in sorted((RAW_CORE / "press_releases").glob("*.pdf")):
        info = parse_ici_pdf(f)
        m = re.search(r"Press_Release_ICI_(\d{8})", f.name)
        if m:
            fd = datetime.strptime(m.group(1), "%Y%m%d").date()
            url = EA + "eight_core_infra/" + f.name
            if info.get("release_date") is None:
                info["release_date"] = fd
        else:
            url = EA + "archive_data/ici_press_release/" + f.name
        info["url"] = url
        info["release_date_basis"] = "printed_on_release" if info.get("release_date") else None
        if m and info.get("release_date_basis") is None:
            info["release_date_basis"] = "file_name_date"
        if info.get("ref_month") is None:
            problems.append(dict(file=f.name, problem="reference month not found"))
            continue
        rels.append(info)
    have = {r["ref_month"] for r in rels if r["obs"]}
    for ref, url in PIB_ICI.items():
        f = RAW_CORE / "press_releases" / "pib" / _pib_file(url)
        if not f.exists():
            continue
        html = f.read_text(errors="ignore")
        if pd.Period(ref, freq="M") in have:
            for r in rels:
                if r["ref_month"] == pd.Period(ref, freq="M") and r.get("release_date") is None:
                    r["release_date"], r["release_date_basis"] = pib_posted_date(html), "pib_posted_on"
            continue
        txt, _ = pib_html_text(f)
        info = parse_ici_text(txt, pd.Period(ref, freq="M"))
        info.update(file=f.name, parse_method="pib_html", url=url, release_date=pib_posted_date(html),
                    release_date_basis="pib_posted_on")
        if info["obs"]:
            rels = [r for r in rels if r["ref_month"] != info["ref_month"]] + [info]
    ann = {}
    for r in rels:
        if r.get("next_release"):
            ann[pd.Period(r["next_release"][0], freq="M")] = r["next_release"][1]
    for r in rels:
        if r.get("release_date") is None and r["ref_month"] in ann:
            r["release_date"], r["release_date_basis"] = ann[r["ref_month"]], "announced_in_previous_release"
    for r in rels:
        if r.get("release_date") is None:
            problems.append(dict(file=r["file"], problem="release date unknown"))
        if not r["obs"]:
            problems.append(dict(file=r["file"], problem="no table parsed (" + r["parse_method"] + ")",
                                 release_date=str(r.get("release_date"))))
    rels = [r for r in rels if r.get("release_date") is not None]
    return sorted(rels, key=lambda r: r["release_date"]), problems


def build_core(fetch_day: date) -> tuple[pd.DataFrame, dict]:
    rels, problems = collect_ici_releases()
    rows = []
    first_by_base = {}
    for r in rels:
        if r["obs"]:
            first_by_base.setdefault(r["base"], r["release_date"])
    for r in rels:
        for o in r["obs"]:
            rows.append(dict(base_year=r["base"], series=o["series"], month=o["month"], provisional=o["provisional"],
                             index_value=o.get("index_value"), yoy_pub=o.get("yoy_pub"),
                             release_date=pd.Timestamp(r["release_date"]), release_ref_month=r["ref_month"],
                             final_month=r.get("final_month"), source_url=r["url"], source_file=r["file"],
                             release_date_basis=r["release_date_basis"], parse_method=r["parse_method"]))
    ob = pd.DataFrame(rows)
    key = ["base_year", "series", "month", "release_date"]
    yoy_chk = yoy_consistency(ob, ["base_year", "series", "release_date", "month"])
    g = ob.dropna(subset=["yoy_pub"]).drop_duplicates(key)[key + ["yoy_pub"]]
    iv = ob.dropna(subset=["index_value"]).drop(columns=["yoy_pub"]).drop_duplicates(key).merge(g, on=key, how="left")
    out = []
    for (base, ser, mth), grp in iv.groupby(["base_year", "series", "month"], sort=False):
        grp = grp.sort_values("release_date")
        last_val, seen, was_prov = None, set(), False
        for _, o in grp.iterrows():
            lag = (o["release_ref_month"] - mth).n
            changed = last_val is None or abs(o["index_value"] - last_val) > 1e-9
            if lag == 0:
                vint = "provisional"
            elif o["release_date"] == pd.Timestamp(first_by_base[base]) and lag > 0 and last_val is None:
                vint = "base_launch" if base != "2004-05" else "restated"
            elif (o["final_month"] is not None and not pd.isna(o["final_month"]) and o["final_month"] == mth) or \
                    (not o["provisional"] and was_prov and "final" not in seen):
                vint = "final"
            elif o["provisional"]:
                vint = "provisional_revised"
            else:
                vint = "restated"
            keep = (vint in ("provisional", "final", "base_launch") and vint not in seen) or \
                   (vint in ("provisional_revised", "restated") and changed)
            was_prov = was_prov or bool(o["provisional"])
            if keep:
                seen.add(vint)
                out.append(dict(o, vintage=vint, vintage_lag_months=lag,
                                value_changed=None if last_val is None else bool(changed), value_source="press_release_pdf"))
                last_val = o["index_value"]
    vt = pd.DataFrame(out)
    # workbooks (latest vintage as of their file date): add values we do not already hold
    last = vt.sort_values("release_date").groupby(["base_year", "series", "month"]).tail(1).set_index(["base_year", "series", "month"])
    wb_rows, cmp = [], {}
    for f in sorted((RAW_CORE / "data_files").glob("Core_Industries_*.xlsx")):
        base, obs = parse_ici_xlsx(f)
        fd = pd.Timestamp(datetime.strptime(re.search(r"_(\d{8})\.xlsx$", f.name).group(1), "%Y%m%d"))
        o = pd.DataFrame(obs)
        if o.empty:
            problems.append(dict(file=f.name, problem="workbook layout not understood"))
            continue
        gi = o.dropna(subset=["index_value"]).drop_duplicates(["series", "month"])
        gg = o.dropna(subset=["yoy_pub"]).drop_duplicates(["series", "month"])[["series", "month", "yoy_pub"]] \
            if "yoy_pub" in o else pd.DataFrame(columns=["series", "month", "yoy_pub"])
        gi = gi.drop(columns=[c for c in ["yoy_pub"] if c in gi]).merge(gg, on=["series", "month"], how="left")
        c = dict(n_compared=0, n_equal=0, diffs=[])
        for _, a in gi.iterrows():
            k = (base, a["series"], a["month"])
            prior = last.loc[k] if k in last.index else None
            if prior is not None:
                c["n_compared"] += 1
                eq = abs(prior["index_value"] - a["index_value"]) < 1e-9
                c["n_equal"] += int(eq)
                if eq:
                    continue
                if len(c["diffs"]) < 12:
                    c["diffs"].append(dict(series=k[1], month=str(k[2]), last_print=prior["index_value"],
                                           last_print_date=str(prior["release_date"].date()), workbook=a["index_value"]))
            if prior is None and base == "2011-12" and a["month"] < pd.Period("2016-04", freq="M"):
                rd, basis = pd.Timestamp(first_by_base.get("2011-12", fd)), \
                    "base_launch_date (first 2011-12 release; value from the current OEA workbook)"
            else:
                rd, basis = fd, "workbook_file_date (OEA workbook posted with the release of that date)"
            wb_rows.append(dict(base_year=base, series=a["series"], month=a["month"], provisional=None,
                                index_value=a["index_value"], yoy_pub=a.get("yoy_pub"), release_date=rd,
                                release_ref_month=pd.NaT, final_month=None, vintage="latest", vintage_lag_months=None,
                                value_changed=None if prior is None else True, source_url=EA + "eight_core_infra/" + f.name,
                                source_file=f.name, release_date_basis=basis, parse_method="xlsx",
                                value_source="oea_workbook"))
        cmp[f.name] = c
    vt = pd.concat([vt, pd.DataFrame(wb_rows)], ignore_index=True)
    vt["yoy_source"] = vt["yoy_pub"].notna().map({True: "published", False: None})
    vt = vt.sort_values("release_date").reset_index(drop=True)
    by = {k: g2.sort_values("release_date") for k, g2 in vt.groupby(["base_year", "series", "month"])}
    for i, r in vt[vt["yoy_pub"].isna()].iterrows():
        prev = by.get((r["base_year"], r["series"], r["month"] - 12))
        if prev is None:
            continue
        known = prev[prev["release_date"] <= r["release_date"]]
        if not known.empty:
            vt.at[i, "yoy_pub"] = round((r["index_value"] / known["index_value"].iloc[-1] - 1) * 100, 1)
            vt.at[i, "yoy_source"] = "computed_pit"
    vt = vt.rename(columns={"yoy_pub": "yoy_pct"})
    vt["month"] = vt["month"].dt.to_timestamp()
    vt["release_ref_month"] = pd.to_datetime(vt["release_ref_month"].astype(str).where(vt["release_ref_month"].notna()), errors="coerce")
    vt = vt[vt["month"] >= "2016-01-01"]
    cols = ["month", "base_year", "series", "index_value", "yoy_pct", "yoy_source", "release_date", "vintage",
            "vintage_lag_months", "provisional", "release_ref_month", "value_changed", "value_source",
            "release_date_basis", "source_url", "source_file", "parse_method"]
    vt = vt[cols].sort_values(["base_year", "series", "month", "release_date"]).reset_index(drop=True)
    vt["vintage_lag_months"] = vt["vintage_lag_months"].astype("Int64")
    meta = dict(releases=[dict(file=r["file"], base=r["base"], ref_month=str(r["ref_month"]), release_date=str(r["release_date"]),
                               release_date_basis=r["release_date_basis"], final_month=str(r.get("final_month")),
                               parse_method=r["parse_method"], n_obs=len(r["obs"])) for r in rels],
                problems=problems, workbook_vs_last_print=cmp, yoy_consistency=yoy_chk)
    return vt, meta


# ----------------------------------------------------------------------------------------------
# Industry map: activity series -> our industry names (exact strings of the industry universe)
# link: direct = the division/category is the industry's own output; direct_partial = the industry's core
# products sit in this series but are a subset of it, or the industry spans two series.
# confidence: high = NIC-2008 division content / same product; medium = judgement on use-based (end-use)
# classification or on a mixed NSE industry.
# ----------------------------------------------------------------------------------------------
_N = NIC2_NAMES
ACTIVITY_MAP = [
    # --- IIP NIC-2 (manufacturing divisions; same codes in base 2011-12 NIC-2008 and base 2022-23 NIC-2025)
    ("iip", "nic2", "10", _N["10"], "Packaged Foods", "direct_partial", "high", "Processed/packaged foods are NIC 10 (food products) output"),
    ("iip", "nic2", "10", _N["10"], "Sugar", "direct", "high", "Sugar manufacture is NIC 1072, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Edible Oil", "direct", "high", "Vegetable and animal oils and fats are NIC 1040, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Dairy Products", "direct", "high", "Dairy products are NIC 1050, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Tea & Coffee", "direct_partial", "high", "Tea/coffee processing is NIC 1079 (food products); plantation output is agriculture, not IIP"),
    ("iip", "nic2", "10", _N["10"], "Other Food Products", "direct", "high", "Other food products are NIC 107x, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Seafood", "direct_partial", "high", "Processing and preserving of fish is NIC 1020, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Meat Products including Poultry", "direct_partial", "high", "Processing and preserving of meat is NIC 1010, inside division 10"),
    ("iip", "nic2", "10", _N["10"], "Animal Feed", "direct", "high", "Prepared animal feeds are NIC 1080, inside division 10"),
    ("iip", "nic2", "11", _N["11"], "Breweries & Distilleries", "direct", "high", "Spirits, wine and malt liquor are NIC 1101-1103 (beverages)"),
    ("iip", "nic2", "11", _N["11"], "Other Beverages", "direct", "high", "Soft drinks and mineral water are NIC 1104 (beverages)"),
    ("iip", "nic2", "12", _N["12"], "Cigarettes & Tobacco Products", "direct", "high", "Division 12 is tobacco products"),
    ("iip", "nic2", "13", _N["13"], "Other Textile Products", "direct", "high", "Spinning, weaving and finishing of textiles are division 13"),
    ("iip", "nic2", "13", _N["13"], "Jute & Jute Products", "direct", "high", "Jute yarn and fabric are textile manufacture (division 13)"),
    ("iip", "nic2", "14", _N["14"], "Garments & Apparels", "direct", "high", "Division 14 is wearing apparel"),
    ("iip", "nic2", "15", _N["15"], "Leather And Leather Products", "direct", "high", "Division 15 is leather and related products"),
    ("iip", "nic2", "15", _N["15"], "Footwear", "direct", "high", "Footwear is NIC 1520, inside division 15"),
    ("iip", "nic2", "16", _N["16"], "Plywood Boards/ Laminates", "direct", "high", "Veneer sheets, plywood and boards are NIC 1621 (wood products)"),
    ("iip", "nic2", "17", _N["17"], "Paper & Paper Products", "direct", "high", "Division 17 is paper and paper products"),
    ("iip", "nic2", "17", _N["17"], "Packaging", "direct_partial", "medium", "Paper/paperboard packaging is NIC 1702; plastic packaging sits in division 22"),
    ("iip", "nic2", "18", _N["18"], "Printing & Publication", "direct_partial", "medium", "Printing is NIC 1811 (division 18); publishing itself is a service outside IIP"),
    ("iip", "nic2", "19", _N["19"], "Refineries & Marketing", "direct", "high", "Refined petroleum products are NIC 1920 (division 19)"),
    ("iip", "nic2", "19", _N["19"], "Lubricants", "direct", "high", "Lubricating oils and greases are refined petroleum products (NIC 1920)"),
    ("iip", "nic2", "20", _N["20"], "Commodity Chemicals", "direct", "high", "Basic chemicals are NIC 2011 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Specialty Chemicals", "direct", "high", "Specialty/other chemical products are NIC 2029 and related classes (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Petrochemicals", "direct", "high", "Petrochemicals and plastics in primary forms are NIC 2011/2013 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Fertilizers", "direct", "high", "Fertilizers are NIC 2012 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Pesticides & Agrochemicals", "direct", "high", "Pesticides and agrochemicals are NIC 2021 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Dyes And Pigments", "direct", "high", "Dyes and pigments are NIC 2011 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Paints", "direct", "high", "Paints and varnishes are NIC 2022 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Printing Inks", "direct", "high", "Printing ink is NIC 2022 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Industrial Gases", "direct", "high", "Industrial gases are NIC 2011 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Carbon Black", "direct", "high", "Carbon black is a basic chemical, NIC 2011 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Explosives", "direct", "high", "Explosives are NIC 2029 (division 20)"),
    ("iip", "nic2", "20", _N["20"], "Personal Care", "direct_partial", "medium", "Soaps, detergents, cosmetics are NIC 2023 (division 20); a small share of the division"),
    ("iip", "nic2", "20", _N["20"], "Household Products", "direct_partial", "medium", "Detergents and cleaning preparations are NIC 2023 (division 20); a small share of the division"),
    ("iip", "nic2", "21", _N["21"], "Pharmaceuticals", "direct", "high", "Division 21 is pharmaceuticals"),
    ("iip", "nic2", "21", _N["21"], "Biotechnology", "direct_partial", "medium", "Biologics/biopharma manufacture is NIC 2100; biotech research services are outside IIP"),
    ("iip", "nic2", "22", _N["22"], "Tyres & Rubber Products", "direct", "high", "Rubber tyres and tubes are NIC 2211 (division 22)"),
    ("iip", "nic2", "22", _N["22"], "Rubber", "direct_partial", "medium", "Rubber products are NIC 221x; natural-rubber plantation output is agriculture"),
    ("iip", "nic2", "22", _N["22"], "Plastic Products - Industrial", "direct", "high", "Plastic products are NIC 2220 (division 22)"),
    ("iip", "nic2", "22", _N["22"], "Plastic Products - Consumer", "direct", "high", "Plastic products are NIC 2220 (division 22)"),
    ("iip", "nic2", "22", _N["22"], "Packaging", "direct_partial", "medium", "Plastic packaging is NIC 2220; paper packaging sits in division 17"),
    ("iip", "nic2", "23", _N["23"], "Cement & Cement Products", "direct", "high", "Cement and concrete products are NIC 2394/2395 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Ceramics", "direct", "high", "Ceramic tiles and products are NIC 2392/2393 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Sanitary Ware", "direct", "high", "Ceramic sanitary fixtures are NIC 2393 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Glass - Industrial", "direct", "high", "Glass and glass products are NIC 2310 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Glass - Consumer", "direct", "high", "Glass and glass products are NIC 2310 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Granites & Marbles", "direct", "high", "Cutting, shaping and finishing of stone is NIC 2396 (division 23)"),
    ("iip", "nic2", "23", _N["23"], "Electrodes & Refractories", "direct", "high", "Refractory products (NIC 2391) and graphite electrodes (NIC 2399) are division 23"),
    ("iip", "nic2", "23", _N["23"], "Other Construction Materials", "direct", "medium", "Clay/concrete/other building materials are division 23 classes"),
    ("iip", "nic2", "23", _N["23"], "Abrasives & Bearings", "direct_partial", "medium", "Abrasive products are NIC 2399 (division 23); bearings are NIC 2814 (division 28)"),
    ("iip", "nic2", "24", _N["24"], "Iron & Steel", "direct", "high", "Basic iron and steel is NIC 2410 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Iron & Steel Products", "direct_partial", "medium", "Steel tubes, wires and other semis are NIC 2410; fabricated steel structures are division 25"),
    ("iip", "nic2", "24", _N["24"], "Ferro & Silica Manganese", "direct", "high", "Ferro-alloys are NIC 2410 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Sponge Iron", "direct", "high", "Direct-reduced (sponge) iron is NIC 2410 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Pig Iron", "direct", "high", "Pig iron is NIC 2410 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Aluminium", "direct", "high", "Aluminium production is NIC 2420 (basic precious and non-ferrous metals)"),
    ("iip", "nic2", "24", _N["24"], "Copper", "direct", "high", "Copper production is NIC 2420 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Zinc", "direct", "high", "Zinc production is NIC 2420 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Aluminium, Copper & Zinc Products", "direct", "high", "Non-ferrous metal semis are NIC 2420 (division 24)"),
    ("iip", "nic2", "24", _N["24"], "Precious Metals", "direct_partial", "medium", "Precious-metal refining is NIC 2420; jewellery is division 32"),
    ("iip", "nic2", "24", _N["24"], "Diversified Metals", "direct_partial", "medium", "Metal smelting is division 24; the mining arm sits in IIP Mining"),
    ("iip", "nic2", "24", _N["24"], "Castings & Forgings", "direct_partial", "high", "Casting of metals is NIC 2431/2432 (division 24); forging is NIC 2591 (division 25)"),
    ("iip", "nic2", "25", _N["25"], "Castings & Forgings", "direct_partial", "high", "Forging, pressing and stamping of metal is NIC 2591 (division 25)"),
    ("iip", "nic2", "25", _N["25"], "Aerospace & Defense", "direct_partial", "medium", "Weapons and ammunition are NIC 2520 (division 25); aircraft/military vehicles are division 30"),
    ("iip", "nic2", "26", _N["26"], "Computers Hardware & Equipments", "direct", "high", "Computers and peripherals are NIC 2620 (division 26)"),
    ("iip", "nic2", "26", _N["26"], "Consumer Electronics", "direct", "high", "Consumer electronics are NIC 2640 (division 26)"),
    ("iip", "nic2", "26", _N["26"], "Telecom -  Equipment & Accessories", "direct", "high", "Communication equipment is NIC 2630 (division 26)"),
    ("iip", "nic2", "26", _N["26"], "Medical Equipment & Supplies", "direct_partial", "medium", "Electromedical equipment is NIC 2660 (division 26); instruments and supplies are NIC 3250 (division 32)"),
    ("iip", "nic2", "27", _N["27"], "Heavy Electrical Equipment", "direct", "high", "Motors, generators, transformers and switchgear are NIC 2710 (division 27)"),
    ("iip", "nic2", "27", _N["27"], "Other Electrical Equipment", "direct", "high", "Division 27 is electrical equipment"),
    ("iip", "nic2", "27", _N["27"], "Cables - Electricals", "direct", "high", "Wires and cables are NIC 2732 (division 27)"),
    ("iip", "nic2", "27", _N["27"], "Household Appliances", "direct", "high", "Domestic appliances are NIC 2750 (division 27)"),
    ("iip", "nic2", "28", _N["28"], "Industrial Machinery", "direct", "high", "Division 28 is machinery and equipment n.e.c."),
    ("iip", "nic2", "28", _N["28"], "Compressors, Pumps & Diesel Engines", "direct", "high", "Engines, pumps and compressors are NIC 2811/2813 (division 28)"),
    ("iip", "nic2", "28", _N["28"], "Abrasives & Bearings", "direct_partial", "medium", "Bearings are NIC 2814 (division 28); abrasives are division 23"),
    ("iip", "nic2", "28", _N["28"], "Tractors", "direct", "high", "Agricultural tractors are NIC 2821 (division 28), not motor vehicles"),
    ("iip", "nic2", "28", _N["28"], "Construction Vehicles", "direct", "high", "Mining and construction machinery is NIC 2824 (division 28)"),
    ("iip", "nic2", "29", _N["29"], "Passenger Cars & Utility Vehicles", "direct", "high", "Motor vehicles are NIC 2910 (division 29)"),
    ("iip", "nic2", "29", _N["29"], "Commercial Vehicles", "direct", "high", "Motor vehicles are NIC 2910 (division 29)"),
    ("iip", "nic2", "29", _N["29"], "Auto Components & Equipments", "direct", "high", "Parts and accessories for motor vehicles are NIC 2930 (division 29)"),
    ("iip", "nic2", "30", _N["30"], "2/3 Wheelers", "direct", "high", "Motorcycles are NIC 3091 (division 30, other transport), not division 29"),
    ("iip", "nic2", "30", _N["30"], "Cycles", "direct", "high", "Bicycles are NIC 3092 (division 30)"),
    ("iip", "nic2", "30", _N["30"], "Ship Building & Allied Services", "direct", "high", "Building of ships and boats is NIC 3011/3012 (division 30)"),
    ("iip", "nic2", "30", _N["30"], "Railway Wagons", "direct", "high", "Railway locomotives and rolling stock are NIC 3020 (division 30)"),
    ("iip", "nic2", "30", _N["30"], "Aerospace & Defense", "direct_partial", "medium", "Aircraft (NIC 3030) and military fighting vehicles (NIC 3040) are division 30"),
    ("iip", "nic2", "31", _N["31"], "Furniture, Home Furnishing", "direct_partial", "medium", "Furniture is division 31; home textiles sit in division 13"),
    ("iip", "nic2", "32", _N["32"], "Gems, Jewellery And Watches", "direct", "high", "Jewellery is NIC 3211 (division 32, other manufacturing)"),
    ("iip", "nic2", "32", _N["32"], "Medical Equipment & Supplies", "direct_partial", "medium", "Medical and dental instruments and supplies are NIC 3250 (division 32)"),
    ("iip", "nic2", "32", _N["32"], "Leisure Products", "direct", "medium", "Sports goods and toys are NIC 3230/3240 (division 32)"),
    # --- IIP sectors / 2022-23 sub-sectors
    ("iip", "sector", "05", "Mining", "Coal", "direct_partial", "high", "Coal output is part of the IIP Mining sector (base 2011-12)"),
    ("iip", "sector", "05", "Mining", "Oil Exploration & Production", "direct_partial", "high", "Crude oil and natural gas output is part of the IIP Mining sector (base 2011-12)"),
    ("iip", "sector", "05", "Mining", "Industrial Minerals", "direct_partial", "medium", "Mineral extraction is part of the IIP Mining sector (base 2011-12)"),
    ("iip", "sector", "05-08", "Mining & Quarrying", "Coal", "direct_partial", "high", "Coal is part of Mining & Quarrying (base 2022-23); see sub-sector Fuel Minerals"),
    ("iip", "sector", "05-08", "Mining & Quarrying", "Oil Exploration & Production", "direct_partial", "high", "Crude oil/gas are part of Mining & Quarrying (base 2022-23); see sub-sector Fuel Minerals"),
    ("iip", "sub_sector", "05-06", "Fuel Minerals", "Coal", "direct", "high", "Fuel minerals (NIC 05-06) = coal, lignite, crude petroleum, natural gas (base 2022-23 only)"),
    ("iip", "sub_sector", "05-06", "Fuel Minerals", "Oil Exploration & Production", "direct", "high", "Fuel minerals (NIC 05-06) include crude petroleum and natural gas (base 2022-23 only)"),
    ("iip", "sub_sector", "07", "Metallic Minerals", "Industrial Minerals", "direct_partial", "medium", "Metal ore mining (NIC 07, e.g. iron ore, manganese) (base 2022-23 only)"),
    ("iip", "sub_sector", "08", "Non-Metallic Minerals", "Industrial Minerals", "direct_partial", "medium", "Other mining and quarrying (NIC 08) (base 2022-23 only)"),
    ("iip", "sector", "35", "Electricity", "Power Generation", "direct", "high", "IIP Electricity (base 2011-12) is electricity generation"),
    ("iip", "sector", "35", "Electricity", "Integrated Power Utilities", "direct_partial", "high", "Generation is the output measured; utilities also transmit/distribute"),
    ("iip", "sub_sector", "351", "Electricity", "Power Generation", "direct", "high", "Electricity generation, NIC 351 (base 2022-23 only)"),
    ("iip", "sub_sector", "351", "Electricity", "Integrated Power Utilities", "direct_partial", "high", "Electricity generation, NIC 351 (base 2022-23 only)"),
    ("iip", "sub_sector", "3511R", "Renewable Electricity", "Power Generation", "direct_partial", "high", "Generation from renewable sources (base 2022-23 only)"),
    ("iip", "sub_sector", "3511N", "Non-Renewable Electricity", "Power Generation", "direct_partial", "high", "Generation from non-renewable sources (base 2022-23 only)"),
    ("iip", "sub_sector", "352", "Gas Supply", "LPG/CNG/PNG/LNG Supplier", "direct", "high", "NIC 352 = distribution of gaseous fuels through mains (city gas) (base 2022-23 only)"),
    ("iip", "sub_sector", "352", "Gas Supply", "Gas Transmission/Marketing", "direct_partial", "medium", "NIC 352 covers gas distribution through mains (base 2022-23 only)"),
    ("iip", "sub_sector", "36", "Water Supply", "Water Supply & Management", "direct", "high", "NIC 36 = water collection, treatment and supply (base 2022-23 only)"),
    ("iip", "sub_sector", "37-38", "Sewerage & Waste Management", "Waste Management", "direct", "high", "NIC 37-38 = sewerage, waste collection and treatment (base 2022-23 only)"),
    # --- IIP use-based (end-use aggregates across divisions; only clear-cut product links)
    ("iip", "use_based", "U5", "Consumer durables", "Consumer Electronics", "direct_partial", "medium", "TVs, phones and similar are consumer durables in the use-based classification"),
    ("iip", "use_based", "U5", "Consumer durables", "Household Appliances", "direct_partial", "medium", "Refrigerators, ACs, washing machines are consumer durables"),
    ("iip", "use_based", "U5", "Consumer durables", "2/3 Wheelers", "direct_partial", "medium", "Motorcycles and scooters are consumer durables"),
    ("iip", "use_based", "U5", "Consumer durables", "Gems, Jewellery And Watches", "direct_partial", "medium", "Jewellery is a consumer durable in the use-based classification"),
    ("iip", "use_based", "U6", "Consumer non-durables", "Packaged Foods", "direct_partial", "medium", "Processed foods are consumer non-durables"),
    ("iip", "use_based", "U6", "Consumer non-durables", "Personal Care", "direct_partial", "medium", "Soaps and cosmetics are consumer non-durables"),
    ("iip", "use_based", "U6", "Consumer non-durables", "Cigarettes & Tobacco Products", "direct_partial", "medium", "Cigarettes are consumer non-durables"),
    ("iip", "use_based", "U6", "Consumer non-durables", "Diversified FMCG", "direct_partial", "medium", "FMCG output (foods, toiletries, tobacco) is consumer non-durables"),
    ("iip", "use_based", "U2", "Capital goods", "Heavy Electrical Equipment", "direct_partial", "medium", "Generators, transformers, boilers are capital goods"),
    ("iip", "use_based", "U2", "Capital goods", "Industrial Machinery", "direct_partial", "medium", "Industrial machinery is capital goods"),
    ("iip", "use_based", "U2", "Capital goods", "Construction Vehicles", "direct_partial", "medium", "Earth-moving/construction machinery is capital goods"),
    ("iip", "use_based", "U2", "Capital goods", "Railway Wagons", "direct_partial", "medium", "Rolling stock is capital goods"),
    ("iip", "use_based", "U2", "Capital goods", "Commercial Vehicles", "direct_partial", "medium", "Commercial vehicles are capital goods"),
    ("iip", "use_based", "U4", "Infrastructure/construction goods", "Cement & Cement Products", "direct_partial", "medium", "Cement is an infrastructure/construction good"),
    ("iip", "use_based", "U4", "Infrastructure/construction goods", "Other Construction Materials", "direct_partial", "medium", "Building materials are infrastructure/construction goods"),
    ("iip", "use_based", "U1", "Primary goods", "Coal", "direct_partial", "medium", "Mining output (coal) is primary goods"),
    ("iip", "use_based", "U1", "Primary goods", "Oil Exploration & Production", "direct_partial", "medium", "Crude oil and natural gas are primary goods"),
    ("iip", "use_based", "U1", "Primary goods", "Power Generation", "direct_partial", "medium", "Electricity is primary goods"),
    # --- Core sector (ICI)
    ("core_sector", "core", "COAL", "Coal", "Coal", "direct", "high", "ICI Coal = coal production (excl. lignite in 2011-12 base)"),
    ("core_sector", "core", "CRUDE", "Crude Oil", "Oil Exploration & Production", "direct", "high", "ICI Crude Oil = domestic crude production"),
    ("core_sector", "core", "NATGAS", "Natural Gas", "Oil Exploration & Production", "direct", "high", "ICI Natural Gas = domestic gas production"),
    ("core_sector", "core", "REFINERY", "Refinery Products", "Refineries & Marketing", "direct", "high", "ICI Refinery Products = refinery throughput/output"),
    ("core_sector", "core", "FERT", "Fertilizers", "Fertilizers", "direct", "high", "ICI Fertilizers = urea, DAP and other fertilizer production"),
    ("core_sector", "core", "STEEL", "Steel", "Iron & Steel", "direct", "high", "ICI Steel = finished and alloy steel production"),
    ("core_sector", "core", "STEEL", "Steel", "Iron & Steel Products", "direct_partial", "medium", "Finished steel volumes drive steel-products makers (pipes, wires) that are part of the ICI steel basket"),
    ("core_sector", "core", "CEMENT", "Cement", "Cement & Cement Products", "direct", "high", "ICI Cement = cement production"),
    ("core_sector", "core", "ELEC", "Electricity", "Power Generation", "direct", "high", "ICI Electricity = electricity generation"),
    ("core_sector", "core", "ELEC", "Electricity", "Integrated Power Utilities", "direct_partial", "high", "Generation is the measured output"),
    ("core_sector", "core", "IRONORE", "Iron Ore", "Industrial Minerals", "direct_partial", "medium", "Iron ore production (added in base 2022-23 only)"),
]
CORE_CODES = {"Coal": "COAL", "Crude Oil": "CRUDE", "Natural Gas": "NATGAS", "Refinery Products": "REFINERY",
              "Fertilizers": "FERT", "Steel": "STEEL", "Cement": "CEMENT", "Electricity": "ELEC", "Iron Ore": "IRONORE",
              "Overall Index": "OVERALL"}


def build_map(industries_csv: Path, iip: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    inds = set(pd.read_csv(industries_csv)["industry"])
    m = pd.DataFrame(ACTIVITY_MAP, columns=["source", "series_type", "series_code", "series", "industry", "link",
                                             "confidence", "rationale"])
    bad = sorted(set(m["industry"]) - inds)
    if bad:
        raise SystemExit(f"industry names not in the universe file: {bad}")
    # series must exist in the outputs; record the bases each series exists in
    have = iip.groupby(["series_type", "series_code"])["base_year"].agg(lambda x: ";".join(sorted(set(x))))
    cb = core.assign(series_code=core["series"].map(CORE_CODES)).groupby("series_code")["base_year"].agg(
        lambda x: ";".join(sorted(set(x))))
    bases = []
    for r in m.itertuples():
        if r.source == "iip":
            k = (r.series_type, r.series_code)
            if k not in have.index:
                raise SystemExit(f"mapped IIP series not in output: {k}")
            bases.append(have.loc[k])
        else:
            bases.append(cb.get(r.series_code, ""))
    m["base_years"] = bases
    return m[["source", "series_type", "series_code", "series", "base_years", "industry", "link", "confidence", "rationale"]]


# ----------------------------------------------------------------------------------------------
# WRITE
# ----------------------------------------------------------------------------------------------
INDUSTRIES_CSV = ROOT / "configs/policy_industries.csv"   # was a session scratchpad file; moved into the repo 2026-09-29


def _lag_table(rel_dates: list[tuple]) -> list[dict]:
    """[(ref_month Period, release_date)] -> per era: release day-of-month and days after month end."""
    d = pd.DataFrame(rel_dates, columns=["ref", "rd"])
    d["rd"] = pd.to_datetime(d["rd"])
    d["days_after_month_end"] = (d["rd"] - d["ref"].map(lambda p: p.end_time.normalize())).dt.days
    d["months_ahead"] = d.apply(lambda r: (pd.Period(r["rd"], freq="M") - r["ref"]).n, axis=1)
    d["year"] = d["ref"].map(lambda p: p.year)
    out = []
    for y, g in d.groupby("year"):
        out.append(dict(ref_year=int(y), n=int(len(g)), median_days_after_month_end=float(g["days_after_month_end"].median()),
                        min_days=int(g["days_after_month_end"].min()), max_days=int(g["days_after_month_end"].max()),
                        release_month_offset=sorted(set(int(x) for x in g["months_ahead"])),
                        release_days_of_month=sorted(set(int(x.day) for x in g["rd"]))))
    return out


def _coverage(df: pd.DataFrame, key: str) -> dict:
    out = {}
    for b, g in df.groupby("base_year"):
        out[b] = dict(months=[str(g["month"].min().date())[:7], str(g["month"].max().date())[:7]],
                      n_months=int(g["month"].nunique()), n_series=int(g[key].nunique()),
                      rows_by_vintage={k: int(v) for k, v in g["vintage"].value_counts().items()})
    return out


def write_manifest(path: Path, body: dict) -> None:
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(body, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--industries", default=str(INDUSTRIES_CSV))
    a = ap.parse_args()
    fetch_day = date.today()
    if not a.no_fetch:
        print("== fetch IIP API (latest vintage)", flush=True)
        fetch_iip_api(a.refresh)
        print("== fetch IIP press releases (MoSPI + PIB copies of missing ones)", flush=True)
        fetch_iip_press_releases(a.refresh)
        fetch_pib_iip(a.refresh)
        print("== fetch ICI press releases + workbooks (OEA + PIB copies)", flush=True)
        fetch_core(a.refresh)
        fetch_pib_ici(a.refresh)
    if a.fetch_only:
        return
    api_files = sorted((RAW_IIP / "api").glob("*.json"))
    if api_files:
        fetch_day = max(datetime.fromtimestamp(f.stat().st_mtime).date() for f in api_files)
    print("== build IIP", flush=True)
    iip, im = build_iip(fetch_day)
    print("== build core sector", flush=True)
    core, cm = build_core(fetch_day)
    core.insert(3, "series_code", core["series"].map(CORE_CODES))
    core["provisional"] = core["provisional"].astype("boolean")
    core.loc[core["base_year"] == "2004-05", "provisional"] = pd.NA   # 2004-05 releases only say 'Data are provisional
    core["value_changed"] = core["value_changed"].astype("boolean")
    iip["value_changed"] = iip["value_changed"].astype("boolean")
    DERIVED.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")

    # ---------------- IIP
    p_iip = DERIVED / "iip_monthly.parquet"
    iip.to_parquet(p_iip, index=False)
    rel = [r for r in im["releases"]]
    lag_iip = {b: _lag_table([(pd.Period(r["qe_month"], freq="M"), r["release_date"]) for r in rel if r["base"] == b])
               for b in sorted({r["base"] for r in rel})}
    write_manifest(p_iip, dict(
        dataset="iip_monthly", path=str(p_iip.relative_to(ROOT)), producer="src/agentic/fetch_iip_core.py",
        rows=len(iip), updated=now, fetch_date=str(fetch_day),
        key=["base_year", "series_code", "month", "release_date", "vintage"],
        grain="one row per (base_year, series, reference month, vintage): every distinct published value of a month with the date it became public",
        point_in_time_recipe=("as of date D: rows with release_date < D (IIP is released at 16:00-17:30 IST, so a same-day "
                              "trade must use release_date < D); per (base_year, series_code, month) keep the row with the "
                              "latest release_date. Strict mode: also drop value_source=='api_latest' "
                              "(vintage 'final_revision_inferred' / 'latest')."),
        sources=dict(
            press_releases="MoSPI monthly IIP press releases, listed by https://www.mospi.gov.in/api/latest-release/get-web-latest-release-list and https://www.mospi.gov.in/api/archive/archival-data (model_type=latest_release); files https://www.mospi.gov.in/uploads/...",
            press_release_workbooks="'Indices IIP <base> Monthly_annual <Mon YY>.xlsx' attached to releases from 28-Oct-2025 (full-history snapshot of that release)",
            pib_copies="https://www.pib.gov.in/PressReleasePage.aspx?PRID=<id> for 17 releases missing from the MoSPI archive (PIB 'Posted On' = release date; PDF attachment parsed when present, else the HTML tables)",
            api_latest="https://api.mospi.gov.in/api/iip/getIIPMonthly (eSankhyiki; base_year 2011-12 and 2022-23, type=All) - current vintage only",
            raw_cache=["data/raw/iip/press_releases/", "data/raw/iip/press_releases/pib/", "data/raw/iip/api/", "data/raw/iip/release_index/"]),
        columns=dict(
            month="reference month (first day of month)",
            base_year="'2011-12' (NIC-2008 series, Apr-2012..Mar-2026) or '2022-23' (new series, Apr-2023..; first released 01-Jun-2026). Index levels of different bases are NOT comparable; do not splice levels, use yoy_pct or rebase",
            series_type="general | sector | nic2 | use_based | sub_sector (sub_sector exists only in base 2022-23)",
            series_code="GEN; sectors 05 / 10-32 / 35 (2011-12), 05-08 / 10-32 / 35 / 36-38 (2022-23); NIC-2 division '10'..'32'; use-based U1..U6; 2022-23 sub-sectors 05-06, 07, 08, 351, 3511R, 3511N, 352, 36, 37-38",
            series="canonical name. NIC-2 names are the NIC-2008 division titles for both bases (2022-23 uses NIC-2025 whose manufacturing divisions keep the same codes; division 21 is titled 'basic pharmaceutical products and pharmaceutical preparations' there)",
            index_value="index number, base year = 100 (dimensionless), as printed (1 decimal)",
            yoy_pct="year-on-year change in PERCENT (4.2 = +4.2%), same month previous year",
            yoy_source="published = printed in the same release as index_value (or the API growth_rate for API rows); computed_pit = index_value / (value of month-12 known on release_date) - 1, rounded to 0.1; null = month-12 unknown at that date (e.g. first 12 months of base 2022-23)",
            release_date="date the value became public (see release_date_basis)",
            vintage="quick_estimate (first print, month M in the release for M) | first_revision (release that states it revised M for the first time; until Feb-2025 policy) | final_revision (release that states it finalised M) | base_launch (back-series value first printed in a base's first release: 2011-12 on 12-May-2017, 2022-23 on 01-Jun-2026) | restated (any other print, kept only if the value changed or it is the first print we hold of that month) | final_revision_inferred (NIC-2/pre-Statement-IV months: the finalising release did not print the series; value = current API value, equal to every later print held) | latest (API value differing from every print held; dated by release_date_basis)",
            vintage_lag_months="months between the reference month and the QE month of the release that printed it",
            release_ref_month="QE month of the release that printed the value",
            value_changed="vs the previous row of the same series-month (null for the first)",
            value_source="press_release_pdf | press_release_xlsx | api_latest | later_print",
            release_date_basis="printed_on_release | pib_posted_on | mospi_listing_date | announced_in_previous_release | revision_schedule... | base_launch_date... | fetch_date...",
            source_url="document the value was read from", source_file="cached file name", parse_table="statement/table of origin"),
        units_note="index_value is an index level; yoy_pct is in percent (not a fraction).",
        revision_policy_observed={
            "2011-12, releases 12-Jun-2017..11-Apr-2025 (QE months Apr-2017..Feb-2025)": "QE of M; first revision of M-1; final revision of M-3 (read from each release)",
            "2011-12, release 28-Apr-2025 (QE Mar-2025)": "transition: Dec-2024, Jan-2025 and Feb-2025 all finalised; no first revisions after this",
            "2011-12, releases May-2025..Apr-2026": "QE of M; final revision of M-1",
            "2022-23 (Jun-2026..)": "QE of M; final revision of M-1. The 29-Jun-2026 release re-issued the whole 2022-23 series (Output-PPI deflator, electricity weights), superseding the 01-Jun-2026 figures: those rows appear as vintage 'restated' with value_changed=True",
        },
        release_lag_observed=lag_iip,
        release_calendar_note="QE was released on the 12th of M+2 (or the previous working day) through Feb-2025 data (~40-43 days after month end); from Mar-2025 data (28-Apr-2025) on the 28th of M+1 (or next working day), ~28-32 days after month end. A backtest without a row for a month must not assume it was known.",
        coverage=_coverage(iip, "series_code"),
        releases_parsed=len(rel), releases=rel,
        gaps=[
            "QE release for Sep-2017 (released ~10-Nov-2017) is in neither the MoSPI archive nor the PIB search: Sep-2017 has no quick_estimate row; its first print is the Oct-2017 release (first_revision).",
            "Base 2011-12 IIP did not exist before 12-May-2017. Months Jan-2016..Feb-2017 first appear as base_launch (general/sectors) or as later prints (use-based from 12-Jun-2017; NIC-2 Apr-2016..Mar-2017 via the previous-year column a year later; NIC-2 Jan..Mar-2016 only in the 28-Oct-2025 workbook). The real-time 2016 series was base 2004-05, which is NOT collected here (TODO).",
            "NIC-2 first revisions before Statement IV (Jun-2020 release) were never printed: no first_revision rows for NIC-2 before then; final values are dated by the revision schedule (final_revision_inferred).",
            "2011-12 and 2022-23 levels are separate series; there is no official linking factor used here.",
            "API getIIPMonthly lags some final revisions (e.g. Aug-2025, base 2011-12: 27 series differ from the printed final of 28-Oct-2025 and every later workbook). Printed finals are kept; API disagreements are listed in validation.",
        ],
        validation=dict(api_vs_prints=im["api_vs_last_print"], within_release_statement_conflicts=im["within_release_conflicts"],
                        yoy_consistency_same_release=im["yoy_consistency"],
                        problems=[p for p in im["problems"] if "out of scope" not in p["problem"]],
                        base_2004_05_releases_skipped=[p["file"] for p in im["problems"] if "out of scope" in p["problem"]]),
    ))

    # ---------------- Core sector
    p_core = DERIVED / "core_sector_monthly.parquet"
    core.to_parquet(p_core, index=False)
    lag_core = {b: _lag_table([(pd.Period(r["ref_month"], freq="M"), r["release_date"]) for r in cm["releases"] if r["base"] == b and r["n_obs"]])
                for b in sorted({r["base"] for r in cm["releases"]})}
    write_manifest(p_core, dict(
        dataset="core_sector_monthly", path=str(p_core.relative_to(ROOT)), producer="src/agentic/fetch_iip_core.py",
        rows=len(core), updated=now, fetch_date=str(fetch_day),
        key=["base_year", "series", "month", "release_date", "vintage"],
        grain="one row per (base_year, industry, reference month, vintage)",
        point_in_time_recipe=("as of date D: rows with release_date < D (ICI is released at 17:00 IST); per (base_year, series, month) "
                              "keep the row with the latest release_date. Strict mode: drop value_source=='oea_workbook' rows whose "
                              "release_date_basis starts with 'base_launch_date'."),
        sources=dict(
            press_releases="OEA/DPIIT monthly ICI releases https://eaindustry.nic.in/archive_data/ici_press_release/IPR_YYYY_MM.pdf (file name = reference month) and the current https://eaindustry.nic.in/eight_core_infra/Press_Release_ICI_YYYYMMDD.pdf",
            pib_copies="PIB copies for releases whose OEA PDF has no text layer: " + ", ".join(f"{k}: {v}" for k, v in PIB_ICI.items()),
            workbooks="https://eaindustry.nic.in/eight_core_infra/Core_Industries_2011_12_20260720.xlsx and Core_Industries_2022_23_20260921.xlsx (latest vintage as of the date in the file name)",
            raw_cache=["data/raw/core_sector/press_releases/", "data/raw/core_sector/press_releases/pib/", "data/raw/core_sector/data_files/", "data/raw/core_sector/listing/"]),
        columns=dict(
            month="reference month (first day)", base_year="'2004-05' (real-time series up to Feb-2017 data), '2011-12' (from the Apr-2017 release of 31-May-2017 to May-2026 data), '2022-23' (nine industries incl. Iron Ore, from 20-Jul-2026). Levels of different bases are not comparable",
            series="Coal | Crude Oil | Natural Gas | Refinery Products | Fertilizers | Steel | Cement | Electricity | Iron Ore (2022-23 only) | Overall Index",
            series_code="COAL CRUDE NATGAS REFINERY FERT STEEL CEMENT ELEC IRONORE OVERALL",
            index_value="index, base year = 100", yoy_pct="year-on-year change in PERCENT",
            yoy_source="published | computed_pit (from values known at release_date) | null",
            release_date="date the value became public (release_date_basis)",
            vintage="provisional (first print, month M in the release for M) | provisional_revised (a later provisional print with a changed value) | final (first print after the release states the month is final / drops its provisional mark) | base_launch (back-series value in a base's first release) | restated (other changed print, or first print we hold when the month's own release is missing) | latest (workbook value not printed in any release held)",
            provisional="True if the release marks the month provisional ('*' or its 'Data for <months> are provisional' note); NA for base 2004-05 rows (those releases only say 'Data are provisional') and workbook rows",
            vintage_lag_months="months between the reference month and the release's reference month",
            value_source="press_release_pdf (incl. PIB copies) | oea_workbook", parse_method="pdftotext | pib_html | xlsx"),
        revision_policy_observed={
            "2011-12 until Feb-2025 data": "provisional for M, revised provisional for M-1 and M-2, final for M-3",
            "transition Mar..May-2025 data": "final month still M-3 while provisional window narrows",
            "2011-12 from Jun-2025 data (release 21-Jul-2025)": "provisional for M, final for M-1",
            "2022-23": "provisional for M, final for M-1 ('PROVISIONAL ESTIMATES ... FOR <M>, AND FINAL INDEX FOR <M-1>')",
        },
        release_lag_observed=lag_core,
        release_calendar_note="Last working day of M+1 (~28-33 days after month end) through Feb-2025 data; from Mar-2025 data around the 20th of M+1 (20-22 days). Releases without a printed date (2015-12..2020-03) are dated by the date announced in the previous release.",
        coverage=_coverage(core, "series"),
        releases=cm["releases"],
        gaps=[
            "Image-only OEA PDFs with no PIB copy found: May-2016, Sep-2016, Dec-2016, Jan-2017, Mar-2017 (base 2004-05) and Jul-2026 (base 2022-23, released 20-Aug-2026). Their months' first rows are later prints (vintage restated/final). OCR of these tables was tried and rejected as unreliable.",
            "Base 2011-12 ICI starts with the Apr-2017 release (31-May-2017); 2016 months in base 2011-12 are base_launch rows (Apr-2016..) or workbook rows dated to that launch (Jan..Mar-2016, strict mode excludes them). Real-time 2016 values are the base 2004-05 rows.",
            "Base 2022-23 back series (Apr-2023..) was published 20-Jul-2026 but only the 21-Sep-2026 workbook is still online: months not printed in a release carry release_date 2026-09-21 (workbook_file_date).",
            "Releases up to the Mar-2020 data carry no printed date; they are dated by the date announced in the previous release (release_date_basis='announced_in_previous_release'; Jan-2016 from the Dec-2015 release, whose own date is unknown and out of scope). A rescheduled release would be off by that change.",
        ],
        validation=dict(workbook_vs_prints=cm["workbook_vs_last_print"], yoy_consistency_same_release=cm["yoy_consistency"],
                        problems=cm["problems"]),
    ))

    # ---------------- Map
    mp = build_map(Path(a.industries), iip, core)
    p_map = DERIVED / "activity_industry_map.csv"
    mp.to_csv(p_map, index=False)
    write_manifest(p_map, dict(
        dataset="activity_industry_map", path=str(p_map.relative_to(ROOT)), producer="src/agentic/fetch_iip_core.py (ACTIVITY_MAP)",
        rows=len(mp), updated=now, key=["source", "series_type", "series_code", "industry"],
        industry_universe=str(a.industries) + " (column 'industry'; every mapped name is asserted to exist there)",
        columns=dict(source="iip -> data/derived/iip_monthly.parquet; core_sector -> data/derived/core_sector_monthly.parquet",
                     series_type="as in the source parquet (core rows: 'core')", series_code="join key into the source parquet",
                     series="series name as in the source parquet", base_years="bases in which the series exists in the parquet",
                     industry="our industry name (exact string)",
                     link="direct = the series measures the industry's own output; direct_partial = the industry's core products are a subset of the series, or the industry spans two series (it then appears twice)",
                     confidence="high = NIC-2008 division/class content or same product; medium = judgement (use-based end-use aggregates, mixed industries)",
                     rationale="one-line reason"),
        scope="Only direct producer links. No demand-side or input-cost links (e.g. cement -> construction, steel -> autos). Industries without a direct IIP/ICI counterpart (services, finance, retail, construction contractors, trading) are intentionally absent.",
        stats=dict(n_links=len(mp), n_industries=int(mp["industry"].nunique()),
                   by_source={k: int(v) for k, v in mp["source"].value_counts().items()},
                   by_link={k: int(v) for k, v in mp["link"].value_counts().items()}),
    ))
    print(f"iip_monthly: {len(iip):,} rows | core_sector_monthly: {len(core):,} rows | map: {len(mp)} links, "
          f"{mp['industry'].nunique()} industries", flush=True)


if __name__ == "__main__":
    main()
