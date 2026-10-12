"""Fetch NSDL's fortnightly sector-wise FPI investment reports (official, FPI Monitor) and build a sector x fortnight table.

Source: https://www.fpi.nsdl.co.in/web/Reports/FPI_Fortnightly_Selection.aspx lists every report (Jan 2012 on); each is a
static page StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestSector_<date>.html with four blocks per sector:
AUC at the middle date (end of the earlier fortnight = start of the later one), net investment in the earlier fortnight, net
investment in the later fortnight, AUC at the end, each in Rs crore and US$ million, split by asset class (equity first).
(Example, the Sep 30 2026 page: AUC Sep 15 | net Sep 1-15 | net Sep 16-30 | AUC Sep 30.)
The report date is read from the page itself ('AUC as on <date>' of the last block); the selection list's labels are not used
(one is wrong: FIIInvestSector_Feb282023.html is labelled 'Feb 28, 2022').
NSDL changed its sector list around 2022 (35 sectors such as 'Software & Services' -> 24 NSE sectors such as 'Information
Technology'); `scheme` records which list a report uses. The pages carry no publication date, so known_from = the next
report's date (the end of the following fortnight; conservative).
Output data/derived/nsdl_fpi_sector.parquet: report_date, known_from, scheme, sector, eq_auc_mid_cr, eq_net_prev_cr (earlier
fortnight), eq_net_cr (later fortnight), eq_auc_end_cr, source_file (all equity, Rs crore).
Check: for every report the numbered sector rows add up to NSDL's Grand Total (equity, each block) within Rs 5 crore.
Network: system curl (keychain trust, verification on), browser headers, 2 s between requests, 30 min budget, stop rule.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
RAW = ROOT / "data/raw/nsdl_fpi_sector"
OUT = ROOT / "data/derived/nsdl_fpi_sector.parquet"
BASE = "https://www.fpi.nsdl.co.in/web/"
SEL = BASE + "Reports/FPI_Fortnightly_Selection.aspx"
HDR = ["-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
       "-H", "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "-H", "Accept-Language: en-US,en;q=0.9"]
BUDGET_S, PAUSE_S = 30 * 60, 2.0
NEW_MARK = "Information Technology"


def _get(url: str, out: Path) -> bool:
    tmp = out.with_suffix(".part")
    r = subprocess.run(["curl", "-s", "-m", "60", "--http1.1", *HDR, "-o", str(tmp), "-w", "%{http_code}", url], capture_output=True, text=True)
    if r.stdout.strip() != "200" or tmp.stat().st_size < 5000:
        tmp.unlink(missing_ok=True); return False
    tmp.replace(out); return True


def _cells(row: str) -> list[str]:
    return [re.sub(r"<[^>]+>|&nbsp;|\s+", " ", x).replace("&amp;", "&").strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S | re.I)]


def _num(s: str) -> float | None:
    s = s.replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")"); s = s.strip("()")
    if s in ("", "-"):
        return 0.0
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _read(page: Path) -> str:
    """Some 2013-2019 pages are saved as UTF-16 (byte-order mark or NUL bytes between letters)."""
    raw = page.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw[:2000].count(b"\x00") > 200:
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def parse(page: Path) -> tuple[pd.DataFrame, dict]:
    h = _read(page)
    dates = re.findall(r"AUC as on\s*([A-Za-z]+\s*\d{1,2},?\s*\d{4})", re.sub(r"<[^>]+>|&nbsp;", " ", h))
    if len(dates) < 2:
        raise ValueError("no 'AUC as on' dates")
    start_d, end_d = (pd.Timestamp(re.sub(r"\s+", " ", x)) for x in dates[:2])
    rows, total, eq_at = [], None, None
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", h, re.S | re.I):
        c = _cells(r)
        if len(c) > 2 and c[0].replace(" ", "").lower() == "sr.no." and eq_at is None:
            labels, starts, st = c[2:], [], 0                        # blocks end at their 'Total' column; equity is each block's first
            for i, lab in enumerate(labels):
                if lab.lower().startswith("total"):
                    starts.append(st); st = i + 1
            if len(starts) == 8 and all(labels[x].lower().startswith("equity") for x in starts):
                eq_at = [starts[0], starts[2], starts[4], starts[6]]   # INR blocks: AUC mid, net earlier, net later, AUC end
            continue
        if len(c) < 10:
            continue
        nums = [_num(x) for x in c[2:]]
        if c[0].isdigit() and all(v is not None for v in nums):
            rows.append((re.sub(r"\s+", " ", re.sub(r"\s+\d+$", "", c[1])).strip(), nums))   # 'Total  Financial Services' -> one space
        elif "Grand Total" in " ".join(c[:2]):
            total = [_num(x) for x in c[2:]]
            break
    if not rows or total is None:
        raise ValueError("no sector rows or no Grand Total")
    n = len(rows[0][1])
    if any(len(v) != n for _, v in rows) or len(total) != n:
        raise ValueError(f"rows of different lengths ({n})")
    if eq_at is None:                                    # no usable header: equal-width blocks only
        if n % 8:
            raise ValueError(f"unexpected column count {n} and no block header")
        k = n // 8; eq_at = [0, 2 * k, 4 * k, 6 * k]
    k = eq_at[1] // 2 if eq_at[1] else 0
    eq = lambda v, j: v[eq_at[j]]  # noqa: E731     # equity, INR, block j (AUC mid, net earlier, net later, AUC end)
    df = pd.DataFrame([dict(sector=s, eq_auc_mid_cr=eq(v, 0), eq_net_prev_cr=eq(v, 1), eq_net_cr=eq(v, 2), eq_auc_end_cr=eq(v, 3)) for s, v in rows])
    chk = {f"block{j}": round(abs(sum(eq(v, j) for _, v in rows) - eq(total, j)), 1) for j in range(4)}
    df["report_date"] = end_d; df["scheme"] = "new24" if (df["sector"] == NEW_MARK).any() else "old35"
    df["source_file"] = page.name
    return df, dict(file=page.name, report_date=str(end_d.date()), start=str(start_d.date()), k=k, sector_rows=len(rows), gap_vs_total=chk)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sel = RAW / "_selection.html"
    if not _get(SEL, sel):
        raise SystemExit("could not load NSDL's report list")
    files = [v.replace("~/", "") for v in re.findall(r'<option[^>]*value="([^"]+\.html)"', sel.read_text(encoding="utf-8", errors="ignore"))]
    t0, ok, bad = time.time(), 0, []
    newest = sorted(files)[-4:]
    for f in files:
        out = RAW / Path(f).name
        if out.exists() and out.name not in {Path(x).name for x in newest}:
            continue
        if time.time() - t0 > BUDGET_S:
            print("time budget reached; the rest next run"); break
        if _get(BASE + f, out):
            ok += 1
        else:
            bad.append(Path(f).name)
        tried = ok + len(bad)
        if tried >= 5 and len(bad) > tried / 2:
            raise SystemExit(f"stopping: {len(bad)} of {tried} downloads failed ({bad[:5]})")
        time.sleep(PAUSE_S)
    frames, checks, unparsed = [], [], []
    for p in sorted(RAW.glob("FIIInvest*.html")):
        try:
            d, c = parse(p); frames.append(d); checks.append(c)
        except Exception as e:                           # noqa: BLE001 — a page that does not parse is listed, never guessed
            unparsed.append(f"{p.name}: {e}")
    D = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["report_date", "sector"], keep="last").sort_values(["report_date", "sector"])
    dates = sorted(D["report_date"].unique())
    nxt = dict(zip(dates[:-1], dates[1:]))
    D["known_from"] = D["report_date"].map(nxt)                         # the latest report has no known_from until the next one exists
    D.to_parquet(OUT, index=False)
    C = pd.DataFrame(checks)
    worst = C["gap_vs_total"].map(lambda g: max(g.values())) if len(C) else pd.Series(dtype=float)
    bad_sum = C.loc[worst > 5, "file"].tolist() if len(C) else []
    switch = D.groupby("report_date")["scheme"].first(); first_new = switch[switch == "new24"].index.min()
    OUT.with_name(OUT.name + ".manifest.json").write_text(json.dumps(dict(
        dataset=OUT.name, producer="src/agentic/fetch_nsdl_fpi_sector.py", source=SEL, source_owner="NSDL FPI Monitor", definitions=__doc__,
        rows=len(D), reports=len(dates), coverage=[str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())],
        new_sector_list_from=str(pd.Timestamp(first_new).date()) if pd.notna(first_new) else None,
        units=dict(eq_net_cr="Rs crore, FPI equity net investment in the later fortnight (negative = net selling)", eq_net_prev_cr="Rs crore, the earlier fortnight",
                   eq_auc_mid_cr="Rs crore, FPI equity holdings at the middle date", eq_auc_end_cr="Rs crore, FPI equity holdings at the report date"),
        point_in_time="known_from = the next report's date (end of the following fortnight); NSDL pages carry no publication date",
        reconciliation=dict(reports_checked=len(C), reports_off_by_more_than_5cr=bad_sum[:20]), known_gaps=dict(downloads_failed=bad, unparsed=unparsed[:20]),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"downloaded {ok} ({len(bad)} failed); parsed {len(C)} reports, {len(unparsed)} unparsed; {len(dates)} report dates "
          f"{pd.Timestamp(dates[0]).date()}..{pd.Timestamp(dates[-1]).date()}; new sector list from {pd.Timestamp(first_new).date() if pd.notna(first_new) else None}; "
          f"reports whose sectors miss the Grand Total by > Rs 5 cr: {len(bad_sum)} {bad_sum[:5]}; unparsed {unparsed[:3]}")


if __name__ == "__main__":
    main()
