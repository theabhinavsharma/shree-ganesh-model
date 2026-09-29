"""Union Budget capital expenditure by ministry/department, point-in-time (budgets 2016-17 .. 2026-27).

Phase 2 of the industry-first study: "which industries is the nation investing in", using only what
was published by each date.

SOURCES (official only; indiabudget.gov.in, every file cached under data/raw/budget/<budget_id>/)
  * Expenditure Profile, "Expenditure of Ministries and Departments" (Statement 3 / 3A; in 2016-17
    Expenditure Budget Vol. I Statement 2 "Total Expenditure of Ministries/Departments"). One
    statement per budget carries four column blocks: Actuals t-2, BE t-1, RE t-1, BE t, each split
    Revenue / Capital / Total (2016-17: Plan / Non-Plan / Total inside Revenue and Capital rows).
    We keep the Capital figure only  -> definition "capital_expenditure".
  * Expenditure Profile, "Allocation under the object head Grants for creation of Capital Assets"
    (Statement 6), demand level -> definition "grants_for_creation_of_capital_assets" (a distinct
    concept: grants to States/bodies that create assets they own; NOT in the capital column).
  * Budget at a Glance (bag1): "Effective Capital Expenditure" (= capital expenditure + grants for
    creation of capital assets, per the document's own note) and, for the July-2024 budget only, the
    "Provisional Actuals 2023-24" column. Totals only, only where the document prints the row.
  * Budget Speech first pages: pub_date of each budget is READ from the speech, compared with the
    catalogue below, and the run aborts on any disagreement.

POINT-IN-TIME. Every number carries pub_date = the day the budget containing it was presented. A figure
re-printed in a later budget (e.g. BE t-1 restated in budget t, or RE/Actuals repeated by the July
full budgets of 2019 and 2024) is kept as a separate observation with that later pub_date. As-of
lookups: filter pub_date <= d, then take the latest pub_date per (fiscal_year, ministry_key,
estimate_type, definition). `is_first_print` marks the first publication of each such key.

PARSING. pdftotext -layout; a data row is a line ending in 12 numeric tokens (4 for Statement 6);
"..." (the budget's Nil marker) is stored as 0.0 with raw_value "...". Guards that abort the run:
statement column headers must be exactly [Actuals t-2, BE t-1, RE t-1, BE t]; the demand rows must sum
to the printed Grand Total in every column (tolerance 1 crore); Grand Total capital must equal the
Budget-at-a-Glance "On Capital Account" row where that row is machine-readable; effective capex must
equal capex + grants for capital assets. Where an .xlsx twin of Statement 3 exists it is compared
cell-by-cell with the PDF parse (report only).

Downloads use the system curl (trusts the macOS keychain, which holds the TLS-inspecting proxy's root
CA); certificate verification is never disabled. 1.5 s between requests; cached files are reused.

Outputs
  data/derived/budget_capex.parquet (+ .manifest.json)
  data/derived/budget_industry_map.csv (+ .manifest.json)

Run:  python3 src/agentic/fetch_budget_capex.py            (download if missing, parse, write)
      python3 src/agentic/fetch_budget_capex.py --offline  (parse cached files only)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
RAW = ROOT / "data/raw/budget"
OUT = ROOT / "data/derived/budget_capex.parquet"
MAP_OUT = ROOT / "data/derived/budget_industry_map.csv"
INDUSTRY_SRC = ROOT / "data/derived/screener_industry.parquet"   # column `industry` = our industry names
SITE = "https://www.indiabudget.gov.in/"
SLEEP_S = 1.5
TOL_CR = 1.0

# ----------------------------------------------------------------------------------------------------
# Catalogue. `base` = directory the budget's own index page links its documents from (checked by hand
# on indiabudget.gov.in: budget<yy>/budget.asp|index.php -> vol1.asp|expenditure_profile.php).
# pub_date is re-read from the Budget Speech at run time (verify_pub_dates) and must agree.
# ----------------------------------------------------------------------------------------------------
BUDGETS = [
    dict(budget_id="2016-17", fy="2016-17", kind="Full", pub_date="2016-02-29",
         base="budget2016-2017/ub2016-17/", stat3="eb/stat02.pdf", stat3_xls=None,
         gcca=None, bag=None, speech="bs/bs.pdf",
         doc_name="Expenditure Budget Vol. I 2016-2017"),
    dict(budget_id="2017-18", fy="2017-18", kind="Full", pub_date="2017-02-01",
         base="budget2017-2018/ub2017-18/", stat3="eb/stat3a.pdf", stat3_xls="eb/stat3a.xlsx",
         gcca="eb/stat6.pdf", bag="bag/bag1.pdf", speech="bs/bs.pdf",
         doc_name="Expenditure Profile 2017-2018"),
    dict(budget_id="2018-19", fy="2018-19", kind="Full", pub_date="2018-02-01",
         base="budget2018-2019/ub2018-19/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="bag/bag1.pdf", speech="bs/bs.pdf",
         doc_name="Expenditure Profile 2018-2019"),
    dict(budget_id="2019-20_interim", fy="2019-20", kind="Interim", pub_date="2019-02-01",
         base="budget2019-20(I)/ub2019-20/", stat3="eb/stat3a.pdf", stat3_xls="eb/stat3a.xlsx",
         gcca="eb/stat6.pdf", bag="bag/bag1.pdf", speech="bs/bs.pdf",
         doc_name="Expenditure Profile 2019-2020 (Interim Budget)"),
    dict(budget_id="2019-20", fy="2019-20", kind="Full", pub_date="2019-07-05",
         base="budget2019-20/doc/", stat3="eb/stat3a.pdf", stat3_xls="eb/stat3a.xlsx",
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2019-2020 (July 2019 full Budget)"),
    dict(budget_id="2020-21", fy="2020-21", kind="Full", pub_date="2020-02-01",
         base="budget2020-21/doc/", stat3="eb/stat3a.pdf", stat3_xls="eb/stat3a.xlsx",
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2020-2021"),
    dict(budget_id="2021-22", fy="2021-22", kind="Full", pub_date="2021-02-01",
         base="budget2021-22/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2021-2022"),
    dict(budget_id="2022-23", fy="2022-23", kind="Full", pub_date="2022-02-01",
         base="budget2022-23/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2022-2023"),
    dict(budget_id="2023-24", fy="2023-24", kind="Full", pub_date="2023-02-01",
         base="budget2023-24/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2023-2024"),
    dict(budget_id="2024-25_interim", fy="2024-25", kind="Interim", pub_date="2024-02-01",
         base="budget2024-25(I)/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2024-2025 (Interim Budget)"),
    dict(budget_id="2024-25", fy="2024-25", kind="Full", pub_date="2024-07-23",
         base="budget2024-25/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2024-2025 (July 2024 full Budget)"),
    dict(budget_id="2025-26", fy="2025-26", kind="Full", pub_date="2025-02-01",
         base="budget2025-26/doc/", stat3="eb/stat3a.pdf", stat3_xls=None,
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2025-2026"),
    dict(budget_id="2026-27", fy="2026-27", kind="Full", pub_date="2026-02-01",
         base="doc/", stat3="eb/stat3a.pdf", stat3_xls="eb/stat3a.xlsx",
         gcca="eb/stat6.pdf", bag="Budget_at_Glance/bag1.pdf", speech="Budget_Speech.pdf",
         doc_name="Expenditure Profile 2026-2027"),
]
# Note: the .xls (not .xlsx) twins of 2021-22/2022-23 need xlrd (not installed) and the 2023-24..2025-26
# twins 404 on the site, so the xlsx cross-check covers 2017-18, 2019-20 (both), 2020-21 and 2026-27.


def url_of(b: dict, rel: str) -> str:
    return SITE + b["base"] + rel


def local_of(b: dict, rel: str) -> Path:
    return RAW / b["budget_id"] / rel.replace("/", "__")


def fetch(url: str, dest: Path, offline: bool) -> Path | None:
    """Download once via system curl (keychain trust, verification ON). Returns path or None."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    res = subprocess.run(["curl", "-sS", "-L", "-m", "180", "--retry", "2", "-A", "Mozilla/5.0",
                          "-o", str(tmp), "-w", "%{http_code} %{content_type}", url],
                         capture_output=True, text=True)
    time.sleep(SLEEP_S)
    code, _, ctype = res.stdout.partition(" ")
    head = tmp.read_bytes()[:8] if tmp.exists() else b""
    is_html = head.lstrip().lower().startswith((b"<!doc", b"<html"))
    if res.returncode != 0 or code != "200" or is_html:
        print(f"  MISS {url} rc={res.returncode} http={code} ctype={ctype} html={is_html}", flush=True)
        tmp.unlink(missing_ok=True)
        return None
    tmp.rename(dest)
    print(f"  got  {url} -> {dest.relative_to(ROOT)} ({dest.stat().st_size:,} B)", flush=True)
    return dest


def download_all(offline: bool) -> None:
    for b in BUDGETS:
        for key in ("stat3", "stat3_xls", "gcca", "bag", "speech"):
            rel = b.get(key)
            if rel:
                fetch(url_of(b, rel), local_of(b, rel), offline)


def pdftext(path: Path, first: int | None = None, last: int | None = None, layout: bool = True) -> str:
    cmd = ["pdftotext"] + (["-layout"] if layout else [])
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    res = subprocess.run(cmd + [str(path), "-"], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"pdftotext failed on {path}: {res.stderr[:300]}")
    return res.stdout


# ----------------------------------------------------------------------------------------------------
# pub_date verification from the Budget Speech
# ----------------------------------------------------------------------------------------------------
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July",
                                      "August", "September", "October", "November", "December"], 1)}


def verify_pub_dates() -> dict:
    out = {}
    for b in BUDGETS:
        p = local_of(b, b["speech"])
        if not p.exists():
            raise SystemExit(f"speech missing for {b['budget_id']}: cannot verify pub_date")
        t = pdftext(p, 1, 6, layout=False)
        m = re.search(r"(January|February|March|July)\s+(\d{1,2}),\s*(20\d\d)", t)
        if not m:
            raise SystemExit(f"no date found in speech of {b['budget_id']}")
        d = f"{m.group(3)}-{MONTHS[m.group(1)]:02d}-{int(m.group(2)):02d}"
        if d != b["pub_date"]:
            raise SystemExit(f"pub_date mismatch {b['budget_id']}: catalogue {b['pub_date']} speech {d}")
        out[b["budget_id"]] = dict(date=d, speech_url=url_of(b, b["speech"]), speech_text=m.group(0))
    return out


# ----------------------------------------------------------------------------------------------------
# Statement 3 (ministry-wise Revenue / Capital / Total)
# ----------------------------------------------------------------------------------------------------
NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$|^\.\.\.$|^…$")
HEADER_RE = re.compile(r"MINISTRY/DEPARTMENT|Expenditure Profile|Expenditure Budget|STATEMENT|EXPENDITURE OF "
                       r"MINISTRIES|TOTAL EXPENDITURE OF|[Cc]rores|Revenue\s+Capital\s+Total|Plan\s+Non-Plan"
                       r"|Actuals|Estimates", re.I)
BLOCK_RE = re.compile(r"(Actuals|Budget|Revised)(?:\s*Estimates)?\s*(\d{4})\s*-\s*(\d{4})")
GCCA_HDR_RE = re.compile(r"ALLOCATION UNDER|GRANTS FOR CREATION|MàQNQDR|In\s*\S?\s*Crores", re.I)
CATEGORY_RE = re.compile(r"^(\d+\.\s*)?(Central Sector Schemes|Centrally Sponsored Schemes|Establishment"
                         r"|Other Central (Sector )?Expenditure|Other Grants/Loans/Transfers|Other Transfers"
                         r"|Finance Commission Grants)", re.I)


def fy_str(y1: int) -> str:
    return f"{y1}-{str(y1 + 1)[2:]}"


def fy_start(fy: str) -> int:
    return int(fy[:4])


def trailing_nums(toks: list[str]) -> int:
    k = 0
    for t in reversed(toks):
        if NUM_RE.match(t):
            k += 1
        else:
            break
    return k


def to_num(tok: str) -> float:
    return 0.0 if tok in ("...", "…") else float(tok)


def expected_blocks(b: dict) -> list[tuple[str, str]]:
    y = fy_start(b["fy"])
    last = "Interim-BE" if b["kind"] == "Interim" else "BE"
    return [("Actual", fy_str(y - 2)), ("BE", fy_str(y - 1)), ("RE", fy_str(y - 1)), (last, fy_str(y))]


def page_meta(page: str) -> tuple[str | None, str | None]:
    """(statement label e.g. 'Statement 3', printed page number) from the page header."""
    st = re.search(r"STATEMENT\s+(\d+[A-Z]?)", page)
    pp = None
    for line in page.split("\n")[:4]:
        m = re.search(r"(Expenditure Profile|Expenditure Budget Vol\. I),?\s*\d{4}\s*-\s*\d{4}\s+(\d+)\s*$",
                      line.strip())
        if m:
            pp = m.group(2)
            break
    return (f"Statement {st.group(1)}" if st else None), pp


def parse_stat3(b: dict) -> tuple[list[dict], dict]:
    pdf = local_of(b, b["stat3"])
    pages = pdftext(pdf).split("\f")
    old = b["budget_id"] == "2016-17"
    exp = expected_blocks(b)
    rows: list[dict] = []
    cur: dict | None = None
    section = "I. CENTRAL SECTOR" if old else None
    stmt_label = None
    stop = False
    for pi, page in enumerate(pages, start=1):
        if stop or not page.strip():
            continue
        found = BLOCK_RE.findall(page[:4000])
        if found:
            kinds = [{"Actuals": "Actual", "Budget": "BE", "Revised": "RE"}[k] for k, _, _ in found[:4]]
            got = [(k, fy_str(int(y1))) for k, (_, y1, _) in zip(kinds, found[:4])]
            want = [("BE" if k == "Interim-BE" else k, f) for k, f in exp]
            if got != want:
                raise SystemExit(f"{b['budget_id']} p{pi}: column header {got} != expected {want}")
        st, pp = page_meta(page)
        stmt_label = st or stmt_label
        for line in page.split("\n"):
            toks = line.split()
            if not toks:
                continue
            k = trailing_nums(toks)
            ind = len(line) - len(line.lstrip())
            if k >= 12:
                vals, label = toks[-12:], " ".join(toks[:-12]).strip()
                if old:
                    # an entity line is followed by its own 'Revenue' and 'Capital' rows; a 'Revenue' line
                    # after a completed triple is the Department of Revenue's entity line.
                    stage = cur["stage"] if cur is not None else 2
                    if (label == "Revenue" and stage == 0) or (label == "Capital" and stage == 1):
                        cur["awaiting_name"] = False
                        cur["stage"] += 1
                        if label == "Capital":
                            cur["vals"] = [vals[2], vals[5], vals[8], vals[11]]   # Plan+Non-Plan totals
                        continue
                    if label == "Capital" or (label == "Revenue" and stage != 2):
                        raise SystemExit(f"{b['budget_id']} p{pi}: unexpected {label} row (stage {stage})")
                    rt = ("grand_total" if label.startswith("Grand Total") else
                          "section_total" if re.match(r"^(I|II|III|IV)\.\s", label) else
                          "subsection_total" if re.match(r"^[AB]\.\s", label) else
                          "state_plan_item" if re.match(r"^[a-z]{1,2}\.\s", label) else
                          "ut_item" if re.match(r"^\(\d+\)\s", label) else "demand")
                    if rt == "section_total":
                        section = label
                    cur = dict(name_parts=[label], demand_no=None, row_type=rt, section=section,
                               pdf_page=pi, printed_page=pp, statement=stmt_label, awaiting_name=True,
                               vals=None, stage=0)
                    rows.append(cur)
                    if rt == "grand_total":
                        stop = True       # its Revenue/Capital rows follow on this page; later pages skipped
                    continue
                # 2017-18 onward
                if label.startswith("Grand Total"):
                    cur = dict(name_parts=["Grand Total"], demand_no=None, row_type="grand_total",
                               section=None, pdf_page=pi, printed_page=pp, statement=stmt_label,
                               awaiting_name=False, vals=[vals[1], vals[4], vals[7], vals[10]])
                    rows.append(cur)
                    stop = True
                    break
                m1 = re.match(r"^Demand\s*No\.?\s*(\d+)$", label)
                m2 = re.match(r"^(\d+)\.\s*(.+)$", label)
                if ind > 3 or CATEGORY_RE.match(label):
                    if cur is not None:
                        cur["awaiting_name"] = False
                    continue                                   # scheme-category sub-row
                if m1 or m2:
                    cur = dict(name_parts=[] if m1 else [m2.group(2)],
                               demand_no=int((m1 or m2).group(1)), row_type="demand", section=None,
                               pdf_page=pi, printed_page=pp, statement=stmt_label, awaiting_name=True,
                               vals=[vals[1], vals[4], vals[7], vals[10]])
                    rows.append(cur)
                    continue
                raise SystemExit(f"{b['budget_id']} p{pi}: unclassified top-level row {label!r}")
            # text-only line (or page header with a trailing page number)
            if HEADER_RE.search(line):
                continue
            if 3 < k < 12:
                raise SystemExit(f"{b['budget_id']} p{pi}: row with {k} numbers (expected 12): {line.strip()[:120]}")
            if cur is not None and cur["awaiting_name"] and ind <= 3 and k == 0:
                cur["name_parts"].append(line.strip())
    for r in rows:
        r["name"] = re.sub(r"\s+", " ", " ".join(r.pop("name_parts"))).strip()
        r.pop("awaiting_name", None)
        r.pop("stage", None)
    missing = [r["name"] for r in rows if r["vals"] is None]
    if missing:
        raise SystemExit(f"{b['budget_id']}: rows without Capital values: {missing[:5]}")
    # ---- reconciliation guards ----
    gt = [r for r in rows if r["row_type"] == "grand_total"]
    if len(gt) != 1:
        raise SystemExit(f"{b['budget_id']}: expected one Grand Total row, got {len(gt)}")
    check = {}
    for j in range(4):
        g = to_num(gt[0]["vals"][j])
        if old:
            s1 = sum(to_num(r["vals"][j]) for r in rows if r["row_type"] == "demand")
            cs = [r for r in rows if r["name"].startswith("I. CENTRAL SECTOR")][0]
            iv = [r for r in rows if r["name"].startswith("IV. Total Central Assistance")][0]
            d1 = s1 - to_num(cs["vals"][j])
            d2 = to_num(cs["vals"][j]) + to_num(iv["vals"][j]) - g
            if abs(d1) > TOL_CR or abs(d2) > TOL_CR:
                raise SystemExit(f"2016-17 block {j}: ministries-vs-central-sector {d1:.2f}, I+IV-GT {d2:.2f}")
            check[exp[j][0] + " " + exp[j][1]] = dict(grand_total=g, sum_central_sector_ministries=round(s1, 2),
                                                        diff_vs_section_I=round(d1, 2), diff_I_plus_IV_vs_GT=round(d2, 2))
        else:
            s = sum(to_num(r["vals"][j]) for r in rows if r["row_type"] == "demand")
            if abs(s - g) > TOL_CR:
                raise SystemExit(f"{b['budget_id']} block {j}: sum of demands {s:.2f} != Grand Total {g:.2f}")
            check[exp[j][0] + " " + exp[j][1]] = dict(grand_total=g, sum_demands=round(s, 2), diff=round(s - g, 2))
    return rows, check


# ----------------------------------------------------------------------------------------------------
# Statement 6: grants for creation of capital assets (4 numbers per row)
# ----------------------------------------------------------------------------------------------------
def parse_gcca(b: dict) -> tuple[list[dict], dict] | tuple[None, str]:
    pdf = local_of(b, b["gcca"]) if b.get("gcca") else None
    if pdf is None or not pdf.exists():
        return None, "statement not available"
    pages = pdftext(pdf).split("\f")
    y = fy_start(b["fy"])
    rows, cur, stop, pending_bare = [], None, False, None
    for pi, page in enumerate(pages, start=1):
        if stop or not page.strip():
            continue
        head = page[:2500]
        if "GRANTS FOR CREATION OF CAPITAL ASSETS" not in head.upper():
            return None, f"page {pi} is not the grants-for-capital-assets statement"
        yrs = sorted(set(int(a) for a, _ in re.findall(r"(\d{4})\s*-\s*(\d{4})", head)))
        if yrs != [y - 2, y - 1, y] or not all(w in head for w in ("Actuals", "Budget", "Revised")):
            return None, f"page {pi}: header years {yrs} not [{y-2},{y-1},{y}]"
        st, pp = page_meta(page)
        for line in page.split("\n"):
            toks = line.split()
            if not toks:
                continue
            k = trailing_nums(toks)
            ind = len(line) - len(line.lstrip())
            if k >= 4:
                vals, label = toks[-4:], " ".join(toks[:-4]).strip()
                if label:
                    pending_bare = None
                if label.startswith("Grand Total"):
                    rows.append(dict(name="Grand Total", demand_no=None, row_type="grand_total", vals=vals,
                                     pdf_page=pi, printed_page=pp, statement=st))
                    stop = True
                    break
                m = re.match(r"^Demand\s*No\.?\s*(\d+)$", label)
                if m:
                    cur = dict(name_parts=[], demand_no=int(m.group(1)), row_type="demand", vals=vals,
                               pdf_page=pi, printed_page=pp, statement=st, awaiting=True)
                    rows.append(cur)
                    continue
                if not label:
                    pending_bare = vals     # numbers whose label pdftotext put on the next line (2025-26)
                    continue
                if re.match(r"^\d+\.?\s", label) or not label[0].isalpha():
                    if cur is not None:
                        cur["awaiting"] = False
                    continue                                    # scheme row
                if ind <= 8 and re.match(r"^(Ministry|Department|Atomic|Space|President|Cabinet|Police|"
                                         r"Interest|Pensions|Transfers|Indirect|Direct|Election|Law)", label):
                    rows.append(dict(name=label, demand_no=None, row_type="ministry_total", vals=vals,
                                     pdf_page=pi, printed_page=pp, statement=st))
                    cur = None
                    continue
                if cur is not None:
                    cur["awaiting"] = False
                continue
            md = re.match(r"^Demand\s*No\.?\s*(\d+)$", line.strip())
            if md and k <= 1 and pending_bare is not None:
                cur = dict(name_parts=[], demand_no=int(md.group(1)), row_type="demand", vals=pending_bare,
                           pdf_page=pi, printed_page=pp, statement=st, awaiting=True, label_split=True)
                rows.append(cur)
                pending_bare = None
                continue
            if HEADER_RE.search(line) or k > 0 or GCCA_HDR_RE.search(line) or line.strip().isupper():
                continue
            pending_bare = None
            if cur is not None and cur.get("awaiting") and ind <= 8:
                nm = re.sub(r"^\d+\s+", "", line.strip())
                cur["name_parts"].append(nm)
                if not line.rstrip().endswith((",", "and", "of", "&")):
                    cur["awaiting"] = False
    for r in rows:
        if "name_parts" in r:
            r["name"] = re.sub(r"\s+", " ", " ".join(r.pop("name_parts"))).strip()
            r.pop("awaiting", None)
    gt = [r for r in rows if r["row_type"] == "grand_total"]
    if len(gt) != 1:
        return None, f"Grand Total rows found: {len(gt)}"
    check = {}
    for j in range(4):
        g = to_num(gt[0]["vals"][j])
        s = sum(to_num(r["vals"][j]) for r in rows if r["row_type"] == "demand")
        if abs(s - g) > TOL_CR:
            return None, f"block {j}: sum of demands {s:.2f} != Grand Total {g:.2f}"
        check[j] = dict(grand_total=g, sum_demands=round(s, 2))
    check["demands_without_printed_name"] = [r["demand_no"] for r in rows if r["row_type"] == "demand" and not r["name"]]
    # ministry-level rows are informational only; dropped from output (demand rows carry the data)
    rows = [r for r in rows if r["row_type"] != "ministry_total"]
    return rows, check


# ----------------------------------------------------------------------------------------------------
# Budget at a Glance: On Capital Account / Grants in Aid for creation of Capital Assets / Effective capex
# ----------------------------------------------------------------------------------------------------
INT_RE = re.compile(r"^-?\d{4,}$")


def bag_row(lines: list[str], label_re: str, n: int) -> list[tuple[list[int], int]]:
    """All occurrences of an English row label; numbers from the same line or the next 3 lines."""
    out = []
    for i, line in enumerate(lines):
        if not re.search(label_re, line):
            continue
        for j in range(i, min(i + 4, len(lines))):
            toks = lines[j].split()
            k = 0
            for t in reversed(toks):
                if INT_RE.match(t):
                    k += 1
                else:
                    break
            if k >= n:
                out.append(([int(t) for t in toks[-n:]], j + 1))
                break
    return out


def parse_bag(b: dict, stat3_gt: list[float]) -> tuple[list[dict], dict]:
    if not b.get("bag"):
        return [], {"status": "no Budget at a Glance PDF in the catalogue (2016-17: only .xls published)"}
    p = local_of(b, b["bag"])
    if not p.exists():
        return [], {"status": "file missing"}
    lines = pdftext(p).split("\n")
    body = "\n".join(lines)
    y = fy_start(b["fy"])
    has_pa = bool(re.search(r"Provisional\s*\n?.*Actuals", body)) and b["budget_id"] == "2024-25"
    n = 5 if has_pa else 4
    cols = [("Actual", fy_str(y - 2)), ("BE", fy_str(y - 1)), ("RE", fy_str(y - 1))]
    if has_pa:
        cols.append(("Provisional-Actual", fy_str(y - 1)))
    cols.append(("Interim-BE" if b["kind"] == "Interim" else "BE", fy_str(y)))
    cap = bag_row(lines, r"\bOn Capital(\s+Account)?\b", n)
    gia = bag_row(lines, r"\d+\.\s*Grants[ -]in[ -][Aa]id for\b|\s{2,}Capital Assets\s+\d{4,}", n)
    eff = bag_row(lines, r"Effective Capital", n)
    info = {"columns": [f"{a} {f}" for a, f in cols]}
    # the capital-account row that reconciles with Statement 3's Grand Total
    gt_round = [round(v) for v in stat3_gt]
    cap_ok = None
    for vals, ln in cap:
        std = [v for (c, _), v in zip(cols, vals) if c != "Provisional-Actual"]
        if all(abs(a - g) <= 1 for a, g in zip(std, gt_round)):
            cap_ok = (vals, ln)
            break
    if cap and not cap_ok:
        raise SystemExit(f"{b['budget_id']}: Budget at a Glance 'On Capital Account' {[c[0] for c in cap]} does not "
                         f"reconcile with Statement 3 Grand Total {gt_round}")
    info["capital_account_row"] = ("matches Statement 3 Grand Total (line %d)" % cap_ok[1]) if cap_ok else \
        "NOT machine-readable"
    out = []
    if cap_ok and has_pa:
        pa_idx = [c for c, _ in cols].index("Provisional-Actual")
        out.append(dict(definition="capital_expenditure", estimate_type="Provisional-Actual",
                        fiscal_year=cols[pa_idx][1], value=float(cap_ok[0][pa_idx]), line=cap_ok[1],
                        label="On Capital Account"))
    eff_ok = None
    if eff:
        vals, ln = eff[0]
        if cap_ok and gia:
            g = gia[0][0]
            if all(abs(e - (c + a)) <= 2 for e, c, a in zip(vals, cap_ok[0], g)):
                eff_ok = (vals, ln)
                info["effective_capex"] = "row found; equals capital account + grants for capital assets"
            else:
                raise SystemExit(f"{b['budget_id']}: effective capex {vals} != capex {cap_ok[0]} + GIA {g}")
        elif cap_ok:
            eff_ok = (vals, ln)
            info["effective_capex"] = "row found; GIA row not machine-readable so identity unchecked"
        else:
            info["effective_capex"] = "row found but capital-account row not reconciled; not used"
    else:
        info["effective_capex"] = "row not printed / not machine-readable"
    if eff_ok:
        for (et, fy), v in zip(cols, eff_ok[0]):
            out.append(dict(definition="effective_capital_expenditure", estimate_type=et, fiscal_year=fy,
                            value=float(v), line=eff_ok[1], label="Effective Capital Expenditure"))
    info["gia_row_values"] = gia[0][0] if gia else None
    return out, info


# ----------------------------------------------------------------------------------------------------
# xlsx cross-check (report only)
# ----------------------------------------------------------------------------------------------------
def xlsx_crosscheck(b: dict, rows: list[dict]) -> dict:
    rel = b.get("stat3_xls")
    if not rel or not rel.endswith(".xlsx"):
        return {"status": "no xlsx twin"}
    p = local_of(b, rel)
    if not p.exists():
        return {"status": "xlsx not downloaded"}
    x = pd.read_excel(p, sheet_name=0, header=None)
    cap_cols = None
    for _, r in x.iterrows():
        idx = [i for i, v in enumerate(r.tolist()) if isinstance(v, str) and v.strip() == "Capital"]
        if len(idx) == 4:
            cap_cols = idx
            break
    if cap_cols is None:
        return {"status": "no Capital header row in xlsx"}
    pdf_by = {r["demand_no"]: r for r in rows if r["row_type"] == "demand"}
    gt = [r for r in rows if r["row_type"] == "grand_total"][0]
    n_cmp = n_bad = 0
    bad = []
    for _, r in x.iterrows():
        cells = r.tolist()
        lab = next((str(c).strip() for c in cells[:cap_cols[0] - 1] if isinstance(c, str) and c.strip()), "")
        m = re.match(r"^(?:Demand No\.?\s*(\d+)|(\d+)\.\s)", lab)
        target = None
        if lab.startswith("Grand Total"):
            target = gt
        elif m:
            target = pdf_by.get(int(m.group(1) or m.group(2)))
        if target is None:
            continue
        for j, c in enumerate(cap_cols):
            v = cells[c]
            xv = 0.0 if (isinstance(v, str) and v.strip() in ("...", "…")) or pd.isna(v) else float(v)
            n_cmp += 1
            if abs(xv - to_num(target["vals"][j])) > 0.011:
                n_bad += 1
                bad.append((lab[:40], j, xv, target["vals"][j]))
    return {"status": "compared", "cells": n_cmp, "mismatches": n_bad, "examples": bad[:5]}


# ----------------------------------------------------------------------------------------------------
# Ministry keys (stable names across renames) and parent-ministry grouping labels
# ----------------------------------------------------------------------------------------------------
ALIASES = {
    "Defence Services": "Capital Outlay on Defence Services",   # 2016-17 Stmt 2: capital part of Defence Services
    "Defence (Misc.)": "Defence (Civil)",   # 'Ministry of Defence (Misc.)' renamed '(Civil)' in July 2019 (same Actual 2017-18)
    "Investment and Public Asset Management (DIPAM)": "Investment and Public Asset Management",
    "Skill Development and Entreprenuership": "Skill Development and Entrepreneurship",   # 2017-18 misprint
    "Shipping": "Ports, Shipping and Waterways",
    "Heavy Industry": "Heavy Industries",
    "Agriculture, Cooperation and Farmers Welfare": "Agriculture and Farmers Welfare",
    "Agriculture Research and Education": "Agricultural Research and Education",
    "Industrial Policy and Promotion": "Promotion of Industry and Internal Trade",
    "Disinvestment": "Investment and Public Asset Management",
    "Personnel, Public Grievances and Pensions": "Personnel, Public Grievances and Pensions",
    "Skill Development and Entrepreneurship": "Skill Development and Entrepreneurship",
}
GROUP = {
    "Capital Outlay on Defence Services": "Defence", "Defence (Civil)": "Defence",
    "Defence Services (Revenue)": "Defence", "Defence Pensions": "Defence",
    "Drinking Water and Sanitation": "Jal Shakti",
    "Water Resources, River Development and Ganga Rejuvenation": "Jal Shakti",
    "Telecommunications": "Communications", "Posts": "Communications",
    "Health and Family Welfare": "Health and Family Welfare", "Health Research": "Health and Family Welfare",
    "Urban Development": "Housing and Urban Affairs",
    "Housing and Urban Poverty Alleviation": "Housing and Urban Affairs",
    "Grand Total": "Total (Union Government)",
}


def ministry_key(name: str) -> str:
    n = re.sub(r"\s+", " ", name).strip().replace("&", "and").replace("’", "'")
    n = re.sub(r"'s\b", "s", n).replace("'", "")
    n = re.sub(r"^(Ministry of|Department of|Department for|Ministry for)\s+", "", n, flags=re.I)
    n = re.sub(r"\s*\(\s*", " (", n).replace(" )", ")")
    if re.match(r"^Ayurveda|^Ayush", n, re.I):
        return "Ayush"
    return ALIASES.get(n, n)


# ----------------------------------------------------------------------------------------------------
# Industry map (ministry_key -> OUR industry names). Direct links only: the ministry's own capital or
# capital-asset grants pay for the industry's output, or the ministry is the capital provider of the
# listed PSUs in that industry. Validated against data/derived/screener_industry.parquet at run time.
# ----------------------------------------------------------------------------------------------------
INDUSTRY_MAP = [
    ("Capital Outlay on Defence Services", "Aerospace & Defense",
     "Capital outlay buys aircraft, aero-engines, missiles, radars, electronics and armoured vehicles (HAL/BEL/BDL-type orders)."),
    ("Capital Outlay on Defence Services", "Ship Building & Allied Services",
     "Naval fleet and dockyard projects are a capital-outlay head; defence shipyards (MDL, Cochin Shipyard) are in this industry."),
    ("Railways", "Railway Wagons",
     "Rolling stock (wagons, coaches, metro/Vande Bharat sets) is procured from railway capital expenditure."),
    ("Railways", "Civil Construction",
     "New lines, doubling, gauge conversion, electrification and station works are EPC contracts (RVNL, IRCON, RITES are in this industry)."),
    ("Railways", "Heavy Electrical Equipment",
     "Electric locomotive propulsion and traction equipment contracts (e.g. Siemens, BHEL, ABB) are charged to railway capital."),
    ("Road Transport and Highways", "Civil Construction",
     "National-highway EPC and HAM construction contracts are paid from MoRTH capital outlay."),
    ("Road Transport and Highways", "Road AssetsToll, Annuity, Hybrid-Annuity",
     "HAM/BOT road-asset owners receive MoRTH construction-period and annuity payments."),
    ("Housing and Urban Affairs", "Civil Construction",
     "Capital goes as equity/sub-debt to metro-rail projects whose civil works are EPC contracts (NBCC, L&T-type)."),
    ("Housing and Urban Affairs", "Railway Wagons",
     "Metro rolling stock is bought by metro SPVs funded through MoHUA capital (Titagarh-type coach makers)."),
    ("Urban Development", "Civil Construction",
     "Pre-merger (to 2017-18) home of the metro-rail capital outlay; same channel as Housing and Urban Affairs."),
    ("Power", "Power Generation",
     "Ministry of Power capital is equity/loans to central generation PSUs (NTPC/NHPC/SJVN-type hydro and thermal projects)."),
    ("Power", "Power - Transmission",
     "Central transmission-strengthening schemes (NE, J&K, Ladakh) are capital works executed by transmission utilities."),
    ("Power", "Heavy Electrical Equipment",
     "Boilers, turbines and generators for central power projects (BHEL-type orders)."),
    ("New and Renewable Energy", "Power Generation",
     "MNRE capital supports renewable generation projects of central PSUs (e.g. SECI)."),
    ("New and Renewable Energy", "Other Electrical Equipment",
     "Solar modules/cells makers (Waaree, Premier Energies are in this industry) supply MNRE-funded capacity."),
    ("New and Renewable Energy", "Financial Institution",
     "MNRE is the capital provider of IREDA (equity infusions are MNRE capital)."),
    ("Drinking Water and Sanitation", "Water Supply & Management",
     "Jal Jeevan Mission / Swachh Bharat water-supply works (mostly as grants for creation of capital assets)."),
    ("Drinking Water and Sanitation", "Plastic Products - Industrial",
     "Jal Jeevan Mission household tap connections are built with PVC/HDPE pipes."),
    ("Drinking Water and Sanitation", "Iron & Steel Products",
     "Jal Jeevan Mission trunk mains use ductile-iron / steel pipes (Jindal Saw, Electrosteel are in this industry)."),
    ("Water Resources, River Development and Ganga Rejuvenation", "Civil Construction",
     "Dams, irrigation and river-linking works (e.g. Polavaram, Ken-Betwa) are EPC contracts."),
    ("Water Resources, River Development and Ganga Rejuvenation", "Water Supply & Management",
     "Namami Gange sewage-treatment and water-treatment plants (Wabag-type contracts)."),
    ("Telecommunications", "Telecom -  Equipment & Accessories",
     "BSNL 4G/5G and BharatNet capital is spent on RAN, optical and transmission equipment (ITI, Tejas, STL are in this industry)."),
    ("Telecommunications", "Telecom - Infrastructure",
     "BharatNet optical-fibre roll-out contracts (HFCL-type)."),
    ("Telecommunications", "Telecom - Cellular & Fixed line services",
     "DoT capital infusions and revival packages for BSNL/MTNL (MTNL is in this industry)."),
    ("Petroleum and Natural Gas", "Refineries & Marketing",
     "Capital support to oil marketing companies (e.g. BE 2023-24 capital support for energy transition) is MoPNG capital."),
    ("Petroleum and Natural Gas", "Oil Storage & Transportation",
     "Strategic Petroleum Reserve caverns (ISPRL) are MoPNG capital projects."),
    ("Petroleum and Natural Gas", "Gas Transmission/Marketing",
     "Capital grant / viability-gap support to national gas-grid pipelines (e.g. Urja Ganga, NE gas grid)."),
    ("Civil Aviation", "Airport & Airport services",
     "MoCA capital funds airport works (AAI, regional-connectivity airports); NB the 2021-22 spike (~Rs 66,900 cr) was Air India legacy debt, not airports."),
    ("Ports, Shipping and Waterways", "Port & Port services",
     "Port and Sagarmala port-infrastructure capital."),
    ("Ports, Shipping and Waterways", "Ship Building & Allied Services",
     "Shipbuilding financial assistance and shipyard (Cochin Shipyard) capital are administered by this ministry."),
    ("Ports, Shipping and Waterways", "Dredging",
     "Inland-waterway (IWAI) and port channel dredging capital (Dredging Corporation is in this industry)."),
    ("Health and Family Welfare", "Medical Equipment & Supplies",
     "New AIIMS / PMSSY / health-infrastructure capital buys medical equipment."),
    ("Steel", "Iron & Steel",
     "Ministry of Steel capital is equity/loans to steel PSUs (SAIL, RINL, NMDC Steel-type)."),
    ("Heavy Industries", "Heavy Electrical Equipment",
     "Administrative ministry of BHEL and other heavy-engineering CPSEs; its capital goes to these CPSEs and capital-goods schemes."),
    ("Heavy Industries", "Industrial Machinery",
     "Scheme for enhancement of competitiveness in the capital-goods (machine-tool) sector."),
    ("Electronics and Information Technology", "Consumer Electronics",
     "Electronics manufacturing (mobile/IT-hardware PLI, EMC) is MeitY's industrial programme (Dixon-type EMS)."),
    ("Electronics and Information Technology", "Computers Hardware & Equipments",
     "IT-hardware PLI and semiconductor/display programmes target this industry."),
    ("Atomic Energy", "Power Generation",
     "DAE capital funds nuclear power projects (NPCIL equity)."),
    ("Atomic Energy", "Heavy Electrical Equipment",
     "Turbine-generator and heavy-equipment packages for nuclear power projects."),
    ("Space", "Aerospace & Defense",
     "Launch vehicles and satellites are Department of Space capital projects."),
    ("Financial Services", "Public Sector Bank",
     "Recapitalisation of public sector banks is Department of Financial Services capital expenditure."),
    ("Financial Services", "General Insurance",
     "Capital infusion in public sector general insurers (New India, United, Oriental, National)."),
    ("Financial Services", "Financial Institution",
     "Equity in development financial institutions (EXIM, NaBFID, IIFCL, SIDBI-type)."),
    ("Fertilisers", "Fertilizers",
     "Department of Fertilisers' capital goes to fertiliser PSUs/plants, i.e. to the industry itself (small and sporadic)."),
]


def build_industry_map(df: pd.DataFrame) -> pd.DataFrame:
    import csv
    ind = pd.read_parquet(INDUSTRY_SRC, columns=["industry"])["industry"].dropna().map(html.unescape)
    ours = set(ind)
    bad = [i for _, i, _ in INDUSTRY_MAP if i not in ours]
    if bad:
        raise SystemExit(f"industry names not in {INDUSTRY_SRC.name}: {bad}")
    keys = set(df["ministry_key"])
    missing = sorted({k for k, _, _ in INDUSTRY_MAP} - keys)
    if missing:
        raise SystemExit(f"industry map uses ministry keys not present in budget_capex: {missing}")
    seen = (df[df.definition == "capital_expenditure"].groupby("ministry_key")["ministry"]
            .agg(lambda s: " | ".join(sorted(set(s)))))
    m = pd.DataFrame(INDUSTRY_MAP, columns=["ministry_key", "industry", "rationale"])
    m["ministry_group"] = m["ministry_key"].map(lambda k: GROUP.get(k, k))
    m["ministry_names_as_printed"] = m["ministry_key"].map(seen)
    m = m[["ministry_key", "ministry_group", "ministry_names_as_printed", "industry", "rationale"]]
    m.to_csv(MAP_OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    return m


# ----------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    download_all(a.offline)
    if a.download_only:
        return
    speech = verify_pub_dates()
    recs, audit = [], {}
    for b in BUDGETS:
        bid = b["budget_id"]
        exp = expected_blocks(b)
        rows, check = parse_stat3(b)
        st_label = next((r["statement"] for r in rows if r.get("statement")), "Statement 3")
        title = ("Total Expenditure of Ministries/Departments" if bid == "2016-17"
                 else "Expenditure of Ministries and Departments")
        src = url_of(b, b["stat3"])
        for r in rows:
            key = ministry_key(r["name"]) if r["row_type"] in ("demand", "grand_total") else None
            if r["row_type"] == "state_plan_item":
                key = ministry_key(re.sub(r"^[a-z]{1,2}\.\s*", "", r["name"])) + " [Central assistance for State Plans]"
            elif r["row_type"] in ("section_total", "subsection_total", "ut_item"):
                key = "[2016-17 section] " + r["name"]
            for j, (et, fy) in enumerate(exp):
                recs.append(dict(
                    fiscal_year=fy, ministry=r["name"], estimate_type=et, capex_cr=to_num(r["vals"][j]),
                    pub_date=b["pub_date"], source_url=src,
                    source_doc=f"{b['doc_name']} (Union Budget {b['fy']}{' Interim' if b['kind'] == 'Interim' else ''})",
                    table_ref=(f"{r['statement'] or st_label} '{title}', column "
                               f"'{ {'Actual': 'Actuals', 'BE': 'Budget Estimates', 'Interim-BE': 'Budget Estimates', 'RE': 'Revised Estimates'}[et] } {fy[:4]}-20{fy[5:]}' / Capital"
                               f"{' (Total of Plan+Non-Plan)' if bid == '2016-17' else ''}; pdf page {r['pdf_page']}"
                               f"{', printed page ' + r['printed_page'] if r.get('printed_page') else ''}"),
                    definition="capital_expenditure", budget_id=bid, budget_kind=b["kind"],
                    demand_no=r["demand_no"], row_type=r["row_type"], ministry_key=key,
                    raw_value=r["vals"][j], section=r.get("section")))
        gt_vals = [to_num(v) for v in [r for r in rows if r["row_type"] == "grand_total"][0]["vals"]]
        # Statement 6 (grants for creation of capital assets)
        g_rows, g_check = parse_gcca(b) if b.get("gcca") else (None, "not in catalogue (2016-17 plan-era volume)")
        if g_rows:
            gsrc = url_of(b, b["gcca"])
            # Demand numbers are common to all statements of one Expenditure Profile: take the name from
            # Statement 3 (consistent keys across definitions); report where Statement 6 prints another name.
            s3_names = {r["demand_no"]: r["name"] for r in rows if r["row_type"] == "demand"}
            name_diff = []
            for r in g_rows:
                if r["row_type"] != "demand":
                    continue
                if r["demand_no"] not in s3_names:
                    raise SystemExit(f"{bid}: Statement 6 demand {r['demand_no']} not in Statement 3")
                if r["name"] and ministry_key(r["name"]) != ministry_key(s3_names[r["demand_no"]]):
                    name_diff.append((r["demand_no"], r["name"], s3_names[r["demand_no"]]))
                r["name"] = s3_names[r["demand_no"]]
            g_check["stat6_vs_stat3_name_differences"] = name_diff
            for r in g_rows:
                for j, (et, fy) in enumerate(exp):
                    recs.append(dict(
                        fiscal_year=fy, ministry=r["name"], estimate_type=et, capex_cr=to_num(r["vals"][j]),
                        pub_date=b["pub_date"], source_url=gsrc,
                        source_doc=f"{b['doc_name']} (Union Budget {b['fy']}{' Interim' if b['kind'] == 'Interim' else ''})",
                        table_ref=(f"{r['statement'] or 'Statement 6'} 'Allocation under the object head Grants "
                                   f"for creation of Capital Assets', column {j + 1} ({et} {fy}); pdf page {r['pdf_page']}"
                                   f"{', printed page ' + r['printed_page'] if r.get('printed_page') else ''}"),
                        definition="grants_for_creation_of_capital_assets", budget_id=bid, budget_kind=b["kind"],
                        demand_no=r["demand_no"], row_type=r["row_type"], ministry_key=ministry_key(r["name"]),
                        raw_value=r["vals"][j], section=None))
        # Budget at a Glance
        bag_out, bag_info = parse_bag(b, gt_vals)
        if g_rows and bag_info.get("gia_row_values"):
            gg = [round(to_num(v)) for v in [r for r in g_rows if r["row_type"] == "grand_total"][0]["vals"]]
            gv = bag_info["gia_row_values"]
            gv_std = gv if len(gv) == 4 else gv[:3] + gv[4:]
            bag_info["gia_vs_statement6"] = "match" if all(abs(x - y) <= 1 for x, y in zip(gv_std, gg)) \
                else f"differ bag {gv_std} vs stmt6 {gg}"
        for o in bag_out:
            recs.append(dict(
                fiscal_year=o["fiscal_year"], ministry="Grand Total", estimate_type=o["estimate_type"],
                capex_cr=o["value"], pub_date=b["pub_date"], source_url=url_of(b, b["bag"]),
                source_doc=f"Budget at a Glance {b['fy'][:4]}-20{b['fy'][5:]} (Union Budget {b['fy']}{' Interim' if b['kind'] == 'Interim' else ''})",
                table_ref=f"Budget at a Glance table, row '{o['label']}' ({o['estimate_type']} {o['fiscal_year']}); text line {o['line']} of pdftotext -layout",
                definition=o["definition"], budget_id=bid, budget_kind=b["kind"], demand_no=None,
                row_type="grand_total", ministry_key="Grand Total", raw_value=str(int(o["value"])), section=None))
        audit[bid] = dict(pub_date=b["pub_date"], speech=speech[bid], stat3_rows=len(rows),
                          stat3_reconciliation=check, xlsx_crosscheck=xlsx_crosscheck(b, rows),
                          stat6=(dict(rows=len(g_rows), reconciliation=g_check) if g_rows else f"not used: {g_check}"),
                          budget_at_a_glance=bag_info)
        print(f"{bid:16s} stat3 rows {len(rows):3d}  GT capex {gt_vals}  stat6 "
              f"{len(g_rows) if g_rows else g_check}  bag {bag_info.get('capital_account_row')}", flush=True)

    df = pd.DataFrame(recs)
    df["ministry_group"] = df["ministry_key"].map(lambda k: GROUP.get(k, k) if isinstance(k, str) else None)
    df["pub_date"] = pd.to_datetime(df["pub_date"])
    df["demand_no"] = df["demand_no"].astype("Int64")
    key = ["fiscal_year", "ministry_key", "estimate_type", "definition"]
    df = df.sort_values(["pub_date", "definition", "row_type", "demand_no", "ministry", "fiscal_year"]).reset_index(drop=True)
    df["is_first_print"] = ~df.duplicated(subset=key, keep="first")
    dup = df[df.duplicated(subset=key + ["pub_date"], keep=False) & df["ministry_key"].notna()]
    if len(dup):
        print("WARNING duplicate keys within one document:\n", dup[key + ["ministry", "budget_id"]].head(20))
    cols = ["fiscal_year", "ministry", "estimate_type", "capex_cr", "pub_date", "source_url", "source_doc",
            "table_ref", "definition", "ministry_key", "ministry_group", "demand_no", "row_type", "section",
            "budget_id", "budget_kind", "raw_value", "is_first_print"]
    df = df[cols]
    df.to_parquet(OUT, index=False)
    m = build_industry_map(df)
    write_manifests(df, m, audit)
    print(f"budget_capex: {len(df):,} rows; industry map {len(m)} rows")


def coverage_table(df: pd.DataFrame) -> dict:
    cap = df[df.definition == "capital_expenditure"]
    out = {}
    for fy in sorted(cap.fiscal_year.unique()):
        s = cap[cap.fiscal_year == fy]
        out[fy] = {et: sorted(s[s.estimate_type == et].budget_id.unique().tolist())
                   for et in ["Interim-BE", "BE", "RE", "Provisional-Actual", "Actual"] if (s.estimate_type == et).any()}
    return out


def write_manifests(df: pd.DataFrame, m: pd.DataFrame, audit: dict) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    man = dict(
        dataset="budget_capex", path=str(OUT.relative_to(ROOT)), rows=len(df),
        key=["fiscal_year", "ministry_key", "estimate_type", "definition", "pub_date"],
        producer="src/agentic/fetch_budget_capex.py",
        source=("Government of India, Union Budget documents on https://www.indiabudget.gov.in/ (official): "
                "Expenditure Profile Statement 3/3A 'Expenditure of Ministries and Departments' (2016-17: "
                "Expenditure Budget Vol. I Statement 2), Expenditure Profile Statement 6 'Allocation under the "
                "object head Grants for creation of Capital Assets', Budget at a Glance (bag1). Raw files cached "
                "in data/raw/budget/<budget_id>/; full parse audit under `parse_audit` below."),
        units="capex_cr: Indian rupees crore (Rs 10 million), nominal, as printed (2 decimals in Statements; "
              "whole crore in Budget at a Glance).",
        budgets_covered={b["budget_id"]: dict(pub_date=b["pub_date"], kind=b["kind"],
                                               stat3_url=url_of(b, b["stat3"]),
                                               stat6_url=url_of(b, b["gcca"]) if b.get("gcca") else None,
                                               bag_url=url_of(b, b["bag"]) if b.get("bag") else None)
                         for b in BUDGETS},
        columns=dict(
            fiscal_year="Indian fiscal year Apr-Mar the figure refers to, 'YYYY-YY'",
            ministry="ministry/department/demand name exactly as printed (whitespace-normalised); 'Grand Total' = whole Union Government",
            estimate_type=("BE = Budget Estimate; Interim-BE = Budget Estimate printed in an Interim Budget (Feb 2019, "
                           "Feb 2024); RE = Revised Estimate; Actual = column headed 'Actuals' (some Budget at a Glance "
                           "editions footnote them as provisional, e.g. 2019-20 Interim and 2021-22); Provisional-Actual = "
                           "July-2024 Budget at a Glance column 'Provisional Actuals 2023-24' (Grand Total only)"),
            capex_cr="value in Rs crore; '...' (Nil) stored as 0.0 (see raw_value)",
            pub_date="date the budget containing this printing was presented (read from the Budget Speech; aborts on mismatch)",
            source_url="URL of the official PDF the number was read from",
            source_doc="document name and which Union Budget it belongs to",
            table_ref="statement / table, column and pdf page (and printed page) of the number",
            definition=("capital_expenditure = Capital column of the ministry-wise statement (capital outlay + loans and "
                        "advances; Grand Total reconciles with Budget at a Glance 'On Capital Account', i.e. net of "
                        "receipts and recoveries as that document states); grants_for_creation_of_capital_assets = "
                        "Statement 6 object-head allocation (a REVENUE-account item, never add it into capital_expenditure "
                        "except to form effective capex); effective_capital_expenditure = Budget at a Glance row, "
                        "defined there as capital expenditure + grants for creation of capital assets (Grand Total only)"),
            ministry_key="stable name across renames (prefix 'Ministry of'/'Department of' dropped; aliases in ALIASES in the producer); 2016-17 section rows are prefixed '[2016-17 section]'",
            ministry_group="parent-ministry label for grouping departments (Defence, Jal Shakti, Communications, Health, Housing and Urban Affairs); a label only, no sums are stored",
            demand_no="Demand for Grants number as printed (numbering changes every year; do not join on it across budgets)",
            row_type="demand (a ministry/department row) | grand_total | 2016-17 only: section_total, subsection_total, state_plan_item, ut_item",
            section="2016-17 only: statement section (I. Central Sector / II. State Plans / III. Union Territories)",
            budget_id="which budget printed the number (e.g. 2024-25_interim, 2024-25)",
            budget_kind="Full | Interim",
            raw_value="token exactly as printed ('...' = Nil)",
            is_first_print=("True for the first publication of (fiscal_year, ministry_key, estimate_type, definition) WITHIN "
                            "this catalogue (budgets 2016-17..2026-27). For BE/RE of FY2014-15/2015-16 the true first print "
                            "predates the catalogue, so those rows are restatements even when flagged True.")),
        point_in_time_use=("as of date d: rows with pub_date <= d; per (fiscal_year, ministry_key, estimate_type, definition) "
                           "take the latest pub_date. BE t first appears on budget day t; RE t with budget t+1; Actuals t with "
                           "budget t+2. The July 2019 and July 2024 full budgets re-print RE/Actuals of the interim documents "
                           "and replace the Interim-BE with a BE."),
        coverage=coverage_table(df),
        limitations=[
            "Railways: the ministry row carries only the Gross Budgetary Support (capital from the Consolidated Fund). "
            "Railway capex funded by internal resources and extra-budgetary resources (IRFC borrowings) is NOT included; "
            "it is in the separate Railway statements (railstat*) which this producer does not parse. TODO if needed.",
            "Defence: 'Capital Outlay on Defence Services' is the capital-outlay demand, net of receipts/recoveries as printed. "
            "2016-17 has no such demand line: Statement 2 prints 'Defence Services' (Revenue+Capital); its Capital row is keyed "
            "to 'Capital Outlay on Defence Services' (its BE 2016-17, 78586.68, is identical to the 2017-18 restatement under the "
            "new demand name). 'Ministry of Defence (Misc.)' (to the Feb-2019 interim budget; 'Defence(Misc.)' in 2016-17, which "
            "also carried defence pensions on the revenue side) is keyed to 'Defence (Civil)', its July-2019 successor name "
            "(Actual 2017-18 = 4992.71 under both names).",
            "2016-17 statement is plan-era (Plan/Non-Plan); capital_expenditure = Plan+Non-Plan Capital total. Its ministry rows "
            "cover Section I (Central Sector) only; Central Assistance for State Plans (Section II) and Union Territories (Section III) "
            "are kept as separate rows (row_type state_plan_item / subsection_total / ut_item). From 2017-18 those transfers sit "
            "inside ministry rows, so 2016-17 ministry rows are not strictly comparable with later years.",
            "Ministry reorganisations: Urban Development + Housing and Urban Poverty Alleviation merged into Housing and Urban Affairs "
            "(restated from BE 2017-18); Drinking Water and Water Resources became departments of Jal Shakti (July 2019); "
            "Animal Husbandry, Dairying and Fisheries split (2019); Department of Heavy Industry printed as Ministry of Heavy Industries "
            "from the 2022-23 budget; Ministry of Shipping printed as Ministry of Ports, Shipping and Waterways from 2021-22. "
            "Keys follow names; restated prior-year columns in later budgets use the new structure.",
            "Grants for creation of capital assets (Statement 6) not available for 2016-17 (plan-era volume has no such statement in "
            "this catalogue). Budget at a Glance rows are used only where machine-readable: 2019-20 Interim BAG table is an image "
            "(no text) and the 2016-17 BAG is published only as .xls; effective capex is printed as a row only from 2023-24.",
            "Actuals of FY2025-26 onward are not yet published (next: Union Budget 2027-28). RE 2026-27 will appear in Feb 2027.",
            "Same key, different scope across the 2016-17 -> 2017-18 restructuring of demands: 2016-17 'Home Affairs' covers the "
            "whole ministry (incl. Police), 'Revenue' covers the whole department (incl. Direct/Indirect Taxes); from 2017-18 these "
            "are split into separate demands, so BE 2016-17 as printed in 2016-17 vs restated in 2017-18 differs (e.g. Home Affairs "
            "9271.41 vs 300.36). Use the latest print for like-for-like YoY comparisons.",
            "Scope: FY2014-15 and FY2015-16 rows exist only as carried in the 2016-17/2017-18 documents (Actual/BE/RE columns); "
            "full vintage coverage starts with BE 2016-17.",
            "Values are nominal Rs crore; no inflation adjustment, no interpolation, no estimates.",
        ],
        audit_summary={bid: dict(stat3_reconciliation=(
                                     "Section I ministry rows sum to the Section I total and I + IV = Grand Total in all 4 columns"
                                     if bid == "2016-17" else "demand rows sum to Grand Total in all 4 columns"),
                                 xlsx=a["xlsx_crosscheck"].get("status") + (
                                     f" ({a['xlsx_crosscheck']['cells']} cells, {a['xlsx_crosscheck']['mismatches']} mismatches)"
                                     if a["xlsx_crosscheck"].get("status") == "compared" else ""),
                                 bag=(a["budget_at_a_glance"].get("capital_account_row")
                                      or a["budget_at_a_glance"].get("status")),
                                 stat6=a["stat6"] if isinstance(a["stat6"], str) else "demand rows sum to Grand Total")
                       for bid, a in audit.items()},
        parse_audit=audit,
        updated=now)
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(man, indent=1, default=str))
    MAP_OUT.with_suffix(".csv.manifest.json").write_text(json.dumps(dict(
        dataset="budget_industry_map", path=str(MAP_OUT.relative_to(ROOT)), rows=len(m),
        key=["ministry_key", "industry"], producer="src/agentic/fetch_budget_capex.py",
        source=("hand-curated in the producer (INDUSTRY_MAP); industry names validated against "
                "data/derived/screener_industry.parquet column `industry` (HTML-unescaped); ministry keys validated "
                "against data/derived/budget_capex.parquet"),
        columns=dict(
            ministry_key="joins budget_capex.ministry_key",
            ministry_group="parent-ministry label (same as budget_capex.ministry_group)",
            ministry_names_as_printed="all printed names of this key in Statement 3 across budgets (pipe-separated)",
            industry="one of OUR industry names (exact string, e.g. 'Telecom -  Equipment & Accessories' has two spaces)",
            rationale="one line: why the ministry's capital spending pays this industry directly"),
        rule=("direct links only: the ministry's capital (or capital-asset grants) buys the industry's output, or the ministry "
              "is the capital provider of listed PSUs in that industry. Ministries whose capital is not industry-specific "
              "(Transfers to States, Home Affairs/Police, Economic Affairs, Rural Development, etc.) are deliberately unmapped."),
        caveats=["Mapping is analyst judgement, not an official classification; it says nothing about how much of a "
                 "ministry's capex reaches listed companies.",
                 "Drinking Water: nearly all Jal Jeevan Mission money is grants_for_creation_of_capital_assets, not the "
                 "capital column; use both definitions for this ministry.",
                 "Electronics and IT / New and Renewable Energy: industrial incentives (PLI, ISM, PM Surya Ghar) are mostly "
                 "revenue-side; the capital column is small."],
        updated=now), indent=1))


if __name__ == "__main__":
    main()
