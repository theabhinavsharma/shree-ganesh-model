"""P&L HISTORY CRAWL — quarterly results figures for the fair-value program (2026-09-05).

Why: valuation-deviation A/B at 15d/63td/126td needs point-in-time historical PE;
we hold every filing DATE since 2007 but no figures. Endpoint verified 2026-09-05:
  corporates-financial-results-data?index=equities&params=<composite>&seq_id=<seq>
    &industry=<industry>&ind=<reInd>&format=<format>
serves the full re_* quarterly P&L for format=New filings (~mid-2016 onward);
Old-format filings return "no data found" (pre-2016 unreachable via JSON).
2025+ figures come from integrated-filing XBRL links (INTEGRATED_FILING_INDAS_*.xml).

Stages (run sequentially; each checkpointed, resume-safe, single worker):
  A calendar   : re-fetch per-symbol results calendar keeping ALL fields (params/
                 industry/reInd/format/xbrl were dropped by the old fleet parse)
                 -> data/derived/results_calendar_full.parquet
  B details    : dedup New-format filings (symbol, periodEnd; prefer Consolidated,
                 latest filing) -> detail JSON per filing
                 -> data/derived/pnl_history/details.jsonl (checkpoint)
  C integrated : dedup integrated filings (2025+) -> XBRL XML fetch+parse
                 -> data/derived/pnl_history/integrated.jsonl (checkpoint)
  D normalize  : one parquet, one row per (symbol, quarter_end, basis):
                 data/derived/pnl_quarterly.parquet
                 [symbol, quarter_end, filing_dt, basis, bank, format, eps_basic,
                  eps_diluted, net_sales, total_income, pbt, pat, face_value, source]

Usage: python3 fetch_pnl_history.py [calendar|details|integrated|normalize|all]
Rate: 1.2-1.6s/request, session rebuilt on error. Memory: trivial (I/O only).
"""
from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.nse.session import build_session  # noqa: E402

OUTDIR = ROOT / "data/derived/pnl_history"
OUTDIR.mkdir(parents=True, exist_ok=True)
CAL_OUT = ROOT / "data/derived/results_calendar_full.parquet"
DET_OUT = OUTDIR / "details.jsonl"
INT_OUT = OUTDIR / "integrated.jsonl"
NORM_OUT = ROOT / "data/derived/pnl_quarterly.parquet"
SLEEP = 1.3


def symbols() -> list[str]:
    a = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")["symbol"]
    b = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")["symbol"]
    return sorted(set(a) | set(b))


def jget(s, url, retries=3):
    for i in range(retries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200:
                t = r.text.strip()
                if t.startswith("{") or t.startswith("["):
                    return r.json()
                return None  # served but empty/non-json => treat as no data
        except Exception:
            pass
        time.sleep(3 + 4 * i)
        s = build_session()
    return "ERR"


# ---------------- stage A: calendar with all fields ----------------
def stage_calendar():
    syms = symbols()
    done = set()
    part = OUTDIR / "calendar_rows.jsonl"
    if part.exists():
        with open(part) as f:
            done = {json.loads(l)["_sym"] for l in f if l.strip()}
    print(f"calendar: {len(syms)} symbols, {len(done)} already done", flush=True)
    s = build_session()
    with open(part, "a") as f:
        for i, sym in enumerate(syms):
            if sym in done:
                continue
            j = jget(s, "https://www.nseindia.com/api/corporates-financial-results"
                        f"?index=equities&symbol={sym}&period=Quarterly")
            if j == "ERR":
                print(f"  {sym}: ERR (skipped)", flush=True)
                s = build_session()
                continue
            rows = j if isinstance(j, list) else (j or {}).get("data", []) or []
            f.write(json.dumps({"_sym": sym, "rows": rows}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  [{i}/{len(syms)}] {sym}: {len(rows)} filings", flush=True)
            time.sleep(SLEEP)
    # fold to parquet
    recs = []
    with open(part) as f:
        for l in f:
            d = json.loads(l)
            for r in d["rows"]:
                r["symbol"] = d["_sym"]
                recs.append(r)
    df = pd.DataFrame(recs)
    df.to_parquet(CAL_OUT, index=False)
    print(f"calendar_full: {len(df):,} rows, {df['symbol'].nunique()} symbols -> {CAL_OUT.name}", flush=True)


# ---------------- stage B: New-format detail fetch ----------------
def details_worklist() -> pd.DataFrame:
    cal = pd.read_parquet(CAL_OUT)
    cal = cal[cal["format"] == "New"].copy()
    cal["qe"] = pd.to_datetime(cal["toDate"], format="%d-%b-%Y", errors="coerce")
    cal["fd"] = pd.to_datetime(cal["filingDate"], format="%d-%b-%Y %H:%M", errors="coerce")
    cal = cal.dropna(subset=["qe", "params", "seqNumber"])
    cal["is_con"] = (cal["consolidated"] == "Consolidated").astype(int)
    # keep BOTH bases (consolidated + standalone) but only latest filing per basis
    cal = (cal.sort_values("fd")
              .drop_duplicates(["symbol", "qe", "is_con"], keep="last"))
    return cal


def load_done(prefix: str) -> set:
    done = set()
    for p in OUTDIR.glob(f"{prefix}*.jsonl"):
        with open(p) as f:
            for l in f:
                if l.strip():
                    done.add(json.loads(l)["_key"])
    return done


def stage_details(n: int = 0, k: int = 1):
    wl = details_worklist()
    done = load_done("details")
    out = DET_OUT if k == 1 else OUTDIR / f"details_w{n}.jsonl"
    todo = [r for i, (_, r) in enumerate(wl.iterrows())
            if i % k == n and f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}" not in done]
    print(f"details w{n}/{k}: worklist {len(wl):,}, done {len(done):,}, todo {len(todo):,} "
          f"(~{len(todo)*(SLEEP+0.5)/3600:.1f}h)", flush=True)
    s = build_session()
    nofetch = 0
    with open(out, "a") as f:
        for i, r in enumerate(todo):
            key = f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}"
            u = ("https://www.nseindia.com/api/corporates-financial-results-data?index=equities"
                 f"&params={r['params']}&seq_id={r['seqNumber']}&industry={r['industry']}"
                 f"&ind={r['reInd']}&format={r['format']}")
            j = jget(s, u)
            if j == "ERR":
                s = build_session()
                nofetch += 1
                continue
            data = (j or {}).get("resultsData2") or (j or {}).get("resultsData") or {}
            f.write(json.dumps({"_key": key, "_sym": r["symbol"], "_qe": str(r["qe"].date()),
                                "_con": int(r["is_con"]), "_fd": str(r["fd"]),
                                "_bank": r.get("bank"), "d": data}) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"  [{i}/{len(todo)}] {key} ok={bool(data)} errs={nofetch}", flush=True)
            time.sleep(SLEEP)
    print(f"details done; transient errors skipped: {nofetch} (re-run to retry)", flush=True)


# ---------------- stage C: integrated XBRL (2025+) ----------------
# v2 2026-09-07: context-DATE-verified parsing (quarter ctx = ends at quarter_end,
# duration 75-100d — ctx NAMES like "OneI" lie); governance/insurance files excluded
# (they carry no P&L and were poisoning dedup); bank EPS tags added.
import re as _re

XTAGS = {
    "eps_basic": ["BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                  "BasicEarningsLossPerShareFromContinuingOperations",
                  "BasicEarningsLossPerShare",
                  "BasicEarningsPerShareAfterExtraordinaryItems",
                  "BasicEarningsPerShareBeforeExtraordinaryItems"],
    "eps_diluted": ["DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                    "DilutedEarningsLossPerShareFromContinuingOperations",
                    "DilutedEarningsPerShareAfterExtraordinaryItems"],
    "net_sales": ["RevenueFromOperations", "InterestEarned", "Income"],
    "total_income": ["Income", "TotalIncome"],
    "pbt": ["ProfitBeforeTax", "ProfitLossBeforeTax"],
    "pat": ["ProfitLossForPeriod", "NetProfitLoss", "ProfitLossForThePeriod"],
    "face_value": ["FaceValueOfEquityShareCapital", "FaceValuePerShare"],
}
FIN_FILE = _re.compile(r"INTEGRATED_FILING_(INDAS|NBFC_INDAS|BANKING|NONINDAS)_|INTEGRATED_FILING_\d")


def parse_xbrl(text: str, qe) -> dict:
    """Extract quarter-period facts: context must END at qe with 75-100d duration."""
    ctxs = {}
    for cid, sd, ed in _re.findall(
            r'<xbrli:context id="([^"]+)">.*?<xbrli:startDate>([^<]+)</xbrli:startDate>'
            r'\s*<xbrli:endDate>([^<]+)</xbrli:endDate>.*?</xbrli:context>', text, _re.S):
        ctxs[cid] = (sd.strip(), ed.strip())
    qs = str(qe.date()) if hasattr(qe, "date") else str(qe)[:10]
    valid = set()
    for cid, (sd, ed) in ctxs.items():
        if ed != qs:
            continue
        try:
            dur = (pd.Timestamp(ed) - pd.Timestamp(sd)).days
        except Exception:
            continue
        if 75 <= dur <= 100:
            valid.add(cid)
    if not valid:
        return {}
    facts = {}
    for pre_tag, ctx, val in _re.findall(
            r'<([a-z-]+:[A-Za-z0-9]+)\s+contextRef="([^"]+)"[^>]*>([^<]+)<', text):
        tag = pre_tag.split(":")[1]
        if ctx in valid and tag not in facts:
            facts[tag] = val.strip()
        elif tag not in facts and XTAGS["face_value"][0] == tag:
            facts[tag] = val.strip()  # face value: any context acceptable
    out = {}
    for k, cands in XTAGS.items():
        for c in cands:
            if c in facts:
                try:
                    out[k] = float(facts[c])
                except ValueError:
                    pass
                break
    return out


def stage_integrated(n: int = 0, k: int = 1):
    ic = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
    ic["qe"] = pd.to_datetime(ic["period_to"], format="%d-%b-%Y", errors="coerce")
    ic["bd"] = pd.to_datetime(ic["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    ic = ic.dropna(subset=["qe", "detail_link"])
    ic = ic[ic["detail_link"].str.contains(FIN_FILE, na=False)]  # financial files only
    ic["is_con"] = (ic["consolidated"] == "Consolidated").astype(int)
    ic = ic.sort_values("bd").drop_duplicates(["symbol", "qe", "is_con"], keep="last")
    done = load_done("integrated2")
    out = OUTDIR / (f"integrated2_w{n}.jsonl" if k > 1 else "integrated2.jsonl")
    todo = [r for i, (_, r) in enumerate(ic.iterrows())
            if i % k == n and f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}" not in done]
    print(f"integrated2 w{n}/{k}: worklist {len(ic):,}, done {len(done):,}, todo {len(todo):,}", flush=True)
    s = build_session()
    okc = 0
    with open(out, "a") as f:
        for i, r in enumerate(todo):
            key = f"{r['symbol']}|{r['qe'].date()}|{r['is_con']}"
            try:
                resp = s.get(r["detail_link"], timeout=30)
                d = parse_xbrl(resp.text, r["qe"]) if resp.status_code == 200 else {}
            except Exception:
                s = build_session()
                continue
            okc += bool(d.get("eps_basic") is not None)
            f.write(json.dumps({"_key": key, "_sym": r["symbol"], "_qe": str(r["qe"].date()),
                                "_con": int(r["is_con"]), "_fd": str(r["bd"]), "d": d}) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"  [{i}/{len(todo)}] {key} eps_ok={okc}", flush=True)
            time.sleep(SLEEP)
    print("integrated2 done", flush=True)


# ---------------- stage D: normalize ----------------
NUM = lambda d, *ks: next((float(d[k]) for k in ks if d.get(k) not in (None, "", "-")), None)  # noqa: E731


def stage_normalize():
    rows = []
    det_lines = []
    for p in sorted(OUTDIR.glob("details*.jsonl")):
        det_lines += [l for l in open(p) if l.strip()]
    if True:
        if True:
            for l in det_lines:
                r = json.loads(l)
                d = r.get("d") or {}
                if not d:
                    continue
                rows.append(dict(
                    symbol=r["_sym"], quarter_end=r["_qe"], filing_dt=r["_fd"],
                    basis="con" if r["_con"] else "sa", bank=r.get("_bank"), source="detail_api",
                    eps_basic=NUM(d, "re_basic_eps_for_cont_dic_opr", "re_basic_eps", "re_bsc_eps_bfr_exi"),
                    eps_diluted=NUM(d, "re_dilut_eps_for_cont_dic_opr", "re_diluted_eps"),
                    net_sales=NUM(d, "re_net_sale", "re_int_ernd"),
                    total_income=NUM(d, "re_total_inc", "re_tot_inc"),
                    pbt=NUM(d, "re_pro_loss_bef_tax"),
                    pat=NUM(d, "re_con_pro_loss", "re_proloss_ord_act", "re_net_prft"),
                    face_value=NUM(d, "re_face_val"),
                ))
    int_lines = []
    for p in sorted(OUTDIR.glob("integrated2*.jsonl")):
        int_lines += [l for l in open(p) if l.strip()]
    if True:
        if True:
            for l in int_lines:
                r = json.loads(l)
                d = r.get("d") or {}
                if not d:
                    continue
                rows.append(dict(symbol=r["_sym"], quarter_end=r["_qe"], filing_dt=r["_fd"],
                                 basis="con" if r["_con"] else "sa", bank=None, source="xbrl",
                                 **{k: d.get(k) for k in ["eps_basic", "eps_diluted", "net_sales",
                                                          "total_income", "pbt", "pat", "face_value"]}))
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["filing_dt"] = pd.to_datetime(df["filing_dt"], errors="coerce")
    df = (df.sort_values(["symbol", "quarter_end", "filing_dt"])
            .drop_duplicates(["symbol", "quarter_end", "basis", "source"], keep="last"))
    df.to_parquet(NORM_OUT, index=False)
    print(f"pnl_quarterly: {len(df):,} rows, {df['symbol'].nunique()} symbols, "
          f"{df['quarter_end'].min().date()} -> {df['quarter_end'].max().date()}", flush=True)
    print(df.groupby(df["quarter_end"].dt.year)["eps_basic"]
            .agg(rows="size", eps_nonnull=lambda s: f"{s.notna().mean()*100:.0f}%"), flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if which in ("calendar", "all"):
        stage_calendar()
    if which in ("details", "all"):
        stage_details(n, k)
    if which in ("integrated", "all"):
        stage_integrated(n, k)
    if which in ("normalize", "all"):
        stage_normalize()
    print("PNL HISTORY CRAWL COMPLETE", flush=True)
