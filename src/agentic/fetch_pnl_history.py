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

ROOT = Path("/Users/abhinavs./Code/Zoom")
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
    # 2026-09-28: the two calendars above cover ~1,550 companies, so pnl_quarterly held only 58-75% of
    # results filers (17-19% for 2016-17). Add every equity in the security master that traded since 2016.
    sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
    ne_col = "is_non_equity" if "is_non_equity" in sm.columns else "is_fund_unit"
    eq = sm[~sm[ne_col].astype(bool) & (pd.to_datetime(sm["last_trade"]) >= "2016-01-01")]["symbol"]
    return sorted(set(a) | set(b) | set(eq))


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
    "finance_cost": ["FinanceCosts"],                                   # 2026-10-04: interest, for the leverage test
}
FIN_FILE = _re.compile(r"INTEGRATED_FILING_(INDAS|NBFC_INDAS|BANKING|NONINDAS)_|INTEGRATED_FILING_\d")


def parse_xbrl(text: str, qe) -> dict:
    """Extract quarter-period facts: context must END at qe with 75-100d duration."""
    ctxs = {}
    # 2026-10-04: one context block at a time. The old single pattern ran past an instant context (no startDate) into the
    # next context, so that next context (often the quarter or the year) was swallowed and the filing parsed empty.
    for cid, body in _re.findall(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', text, _re.S):
        m = _re.search(r'<xbrli:startDate>([^<]+)</xbrli:startDate>\s*<xbrli:endDate>([^<]+)</xbrli:endDate>', body)
        if m:
            ctxs[cid] = (m.group(1).strip(), m.group(2).strip())
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
        return _parse_cumulative(text, qs, ctxs)
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


def _parse_cumulative(text: str, qs: str, ctxs: dict) -> dict:
    """2026-10-04: some filings carry no 3-month context for the quarter, only cumulative ones ending at the quarter end
    (year-end filings: 6 months + full year; Q3: 9 months). Return the year-to-date period's P&L facts as cum_<field>
    plus cum_start; stage_normalize derives the quarter = cumulative - the earlier quarters inside it. Contexts with a
    dimension (segments, members) are skipped so only the entity total is read."""
    blocks = dict(_re.findall(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', text, _re.S))
    best = None
    for cid, (sd, ed) in ctxs.items():
        if ed != qs or "explicitMember" in blocks.get(cid, "") or "typedMember" in blocks.get(cid, ""):
            continue
        try:
            dur = (pd.Timestamp(ed) - pd.Timestamp(sd)).days
        except Exception:
            continue
        # year-to-date only (starts on April 1, the fiscal-year start): a "6-month" context starting Oct 1 in a year-end
        # file was found to hold the Jan-Mar quarter mislabelled (AMRUTANJAN Mar-2025), so it is not trusted
        if sd.strip()[5:10] == "04-01" and any(lo <= dur <= hi for lo, hi in ((170, 195), (260, 285), (330, 375))) and (best is None or dur > best[1]):
            best = (cid, dur, sd)
    if best is None:
        return {}
    facts = {}
    for pre_tag, ctx, val in _re.findall(r'<([a-z-]+:[A-Za-z0-9]+)\s+contextRef="([^"]+)"[^>]*>([^<]+)<', text):
        tag = pre_tag.split(":")[1]
        if ctx == best[0] and tag not in facts:
            facts[tag] = val.strip()
    out = {}
    for k in ("net_sales", "total_income", "pbt", "pat", "finance_cost"):
        for c in XTAGS[k]:
            if c in facts:
                try:
                    out[f"cum_{k}"] = float(facts[c])
                except ValueError:
                    pass
                break
    if out:
        out["cum_start"] = best[2]; out["cum_days"] = best[1]
    return out


def _derive_from_cumulative(df: pd.DataFrame, cum: pd.DataFrame) -> pd.DataFrame:
    """Quarter = cumulative period - the quarters before it inside the same period (same symbol and basis), in Rs LAKH.
    Kept only when every earlier quarter is present and the derived sales are > 0 and within 1/3x..3x of the previous
    quarter. source 'xbrl_derived' (Rs LAKH, like detail_api); EPS left blank."""
    if cum.empty:
        return pd.DataFrame()
    lakh = lambda r: 1e-5 if r == "xbrl" else 1.0  # noqa: E731 (detail_api already in lakh)
    base = df.assign(f=df["source"].map(lakh))
    idx = {k: g.set_index("quarter_end") for k, g in base.groupby(["symbol", "basis"])}
    out = []
    for r in cum.itertuples():
        g = idx.get((r.symbol, r.basis))
        qe, st = pd.Timestamp(r.quarter_end), pd.Timestamp(r.cum_start)
        prev = [q for q in pd.date_range(st, qe, freq="QE") if q < qe]
        if g is None or not prev or not all(q in g.index for q in prev):
            continue
        rows = g.loc[prev]
        rec = dict(symbol=r.symbol, quarter_end=qe, filing_dt=r.filing_dt, basis=r.basis, bank=None, source="xbrl_derived",
                   eps_basic=None, eps_diluted=None, face_value=None)
        for k in ("net_sales", "total_income", "pbt", "pat", "finance_cost"):
            c = getattr(r, f"cum_{k}", None)
            rec[k] = (c * 1e-5 - float((rows[k] * rows["f"]).sum())) if c == c and c is not None and k in rows and rows[k].notna().all() else None
        last = float(rows.iloc[-1]["net_sales"] * rows.iloc[-1]["f"]) if rows.iloc[-1]["net_sales"] == rows.iloc[-1]["net_sales"] else None
        ns = rec["net_sales"]
        if ns is None or ns <= 0 or not last or not (1 / 3 <= ns / last <= 3):
            continue
        out.append(rec)
    return pd.DataFrame(out)


def _xbrl_fallback_fd(ic: pd.DataFrame) -> pd.Series:
    """Filing time for integrated-filing rows with no broadcast time (2026-09-28: 4,710 of 32,273 calendar rows; they left
    1,679 pnl_quarterly xbrl rows with NaT filing_dt and broke PIT TTMs in 2025-26). Fallback = the upload time in the XBRL
    filename (_DDMMYYYYhhmmss_WEB.xml). Its clock is 12-hour with no AM/PM (hours 1..12; broadcast - filename is 0 or
    720 min), so read hours 1-11 as PM and add 20 minutes. On the 27,563 rows that have both, that reading is never on an
    earlier day than the broadcast and at most 19.8 min earlier on the same day, so +20 min cannot leak a result early."""
    ft = pd.to_datetime(ic["detail_link"].str.extract(r"_(\d{14})_WEB", expand=False), format="%d%m%Y%H%M%S", errors="coerce")
    late = ft + pd.to_timedelta(ft.dt.hour.lt(12).astype(int) * 12, unit="h") + pd.Timedelta(minutes=20)
    return ic["bd"].fillna(late)


def integrated_filing_times() -> pd.DataFrame:
    """(symbol, qe, is_con) -> filing time of the XBRL that stage_integrated fetched (same selection and dedupe)."""
    ic = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
    ic["qe"] = pd.to_datetime(ic["period_to"], format="%d-%b-%Y", errors="coerce")
    ic["bd"] = pd.to_datetime(ic["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    ic = ic.dropna(subset=["qe", "detail_link"])
    ic = ic[ic["detail_link"].str.contains(FIN_FILE, na=False)]
    ic["is_con"] = (ic["consolidated"] == "Consolidated").astype(int)
    ic = ic.sort_values("bd").drop_duplicates(["symbol", "qe", "is_con"], keep="last")
    return ic.assign(fd=_xbrl_fallback_fd(ic))[["symbol", "qe", "is_con", "fd"]]


def stage_integrated(n: int = 0, k: int = 1):
    ic = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
    ic["qe"] = pd.to_datetime(ic["period_to"], format="%d-%b-%Y", errors="coerce")
    ic["bd"] = pd.to_datetime(ic["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
    ic = ic.dropna(subset=["qe", "detail_link"])
    ic = ic[ic["detail_link"].str.contains(FIN_FILE, na=False)]  # financial files only
    ic["is_con"] = (ic["consolidated"] == "Consolidated").astype(int)
    ic = ic.sort_values("bd").drop_duplicates(["symbol", "qe", "is_con"], keep="last")
    ic["bd"] = _xbrl_fallback_fd(ic)          # after the dedupe, so the same file is fetched as before
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
                    face_value=NUM(d, "re_face_val"), finance_cost=NUM(d, "re_int_new"),
                ))
    int_lines, cum_rows = [], []
    best = {}   # 2026-10-04: one record per filing key. Prefer a record WITH figures (quarter > year-to-date > empty), then
    #             the newest re-read (zfc = finance-cost re-read > inc = incremental / empty re-read > original crawl), then
    #             the later line. File-name order alone let an old empty record override a good re-read (caught same day).
    for p in sorted(OUTDIR.glob("integrated2*.jsonl")):
        tier = 2 if "_zfc_" in p.name else (1 if "_inc_" in p.name else 0)
        for i, l in enumerate(open(p)):
            if not l.strip():
                continue
            x = json.loads(l); d = x.get("d") or {}
            score = (2 if d.get("net_sales") is not None else (1 if "cum_start" in d else 0), tier, i)
            k = x["_key"]
            if k not in best or score > best[k][0]:
                best[k] = (score, l)
    int_lines = [v[1] for v in best.values()]
    if True:
        if True:
            for l in int_lines:
                r = json.loads(l)
                d = r.get("d") or {}
                if not d:
                    continue
                if "cum_start" in d and d.get("net_sales") is None:     # cumulative-only filing: derived below
                    cum_rows.append(dict(symbol=r["_sym"], quarter_end=r["_qe"], filing_dt=r["_fd"], basis="con" if r["_con"] else "sa", **d))
                    continue
                rows.append(dict(symbol=r["_sym"], quarter_end=r["_qe"], filing_dt=r["_fd"],
                                 basis="con" if r["_con"] else "sa", bank=None, source="xbrl",
                                 **{k: d.get(k) for k in ["eps_basic", "eps_diluted", "net_sales",
                                                          "total_income", "pbt", "pat", "face_value", "finance_cost"]}))
    df = pd.DataFrame(rows)
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["filing_dt"] = pd.to_datetime(df["filing_dt"], errors="coerce")
    x = df["source"].eq("xbrl") & df["filing_dt"].isna()
    if x.any():                                # records fetched before the fallback existed carry "NaT"
        it = integrated_filing_times()
        fk = it["symbol"] + "|" + it["qe"].dt.strftime("%Y-%m-%d") + "|" + it["is_con"].astype(str)
        key = (df.loc[x, "symbol"] + "|" + df.loc[x, "quarter_end"].dt.strftime("%Y-%m-%d") + "|"
               + df.loc[x, "basis"].eq("con").astype(int).astype(str))
        df.loc[x, "filing_dt"] = key.map(pd.Series(it["fd"].values, index=fk.values))
        print(f"xbrl filing_dt from the XBRL filename time: {int(df.loc[x, 'filing_dt'].notna().sum())} of {int(x.sum())} filled", flush=True)
    df = (df.sort_values(["symbol", "quarter_end", "filing_dt"])
            .drop_duplicates(["symbol", "quarter_end", "basis", "source"], keep="last"))
    # 2026-10-04: quarters derived from cumulative-only XBRL, then quarters read from results PDFs (pnl_quarterly_pdf.parquet),
    # each only where no structured row exists for the same (symbol, quarter_end, basis)
    have = set(zip(df["symbol"], df["quarter_end"], df["basis"]))
    C = pd.DataFrame(cum_rows)
    if len(C):
        C["quarter_end"] = pd.to_datetime(C["quarter_end"]); C["filing_dt"] = pd.to_datetime(C["filing_dt"], errors="coerce")
        C = C[[k not in have for k in zip(C["symbol"], C["quarter_end"], C["basis"])]].sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="first")
    Dv = _derive_from_cumulative(df, C)
    if len(Dv):
        df = pd.concat([df, Dv], ignore_index=True); have |= set(zip(Dv["symbol"], Dv["quarter_end"], Dv["basis"]))
    pdfp = ROOT / "data/derived/pnl_quarterly_pdf.parquet"
    if pdfp.exists():
        Pp = pd.read_parquet(pdfp); Pp["quarter_end"] = pd.to_datetime(Pp["quarter_end"]); Pp["filing_dt"] = pd.to_datetime(Pp["filing_dt"])
        Pp = Pp[Pp["qc_ok"] & [k not in have for k in zip(Pp["symbol"], Pp["quarter_end"], Pp["basis"])]]
        df = pd.concat([df, Pp[[c for c in df.columns if c in Pp.columns]]], ignore_index=True)
    print(f"derived from cumulative XBRL: {len(Dv)} quarters (of {len(C)} cumulative-only filings) · from results PDFs: "
          f"{len(Pp) if pdfp.exists() else 0}", flush=True)
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
