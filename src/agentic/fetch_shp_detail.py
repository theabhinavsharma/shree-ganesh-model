"""Quarterly shareholding detail: mutual fund %, foreign portfolio investor %, domestic institutions %, retail
shareholder count (2026-09-30, research queue item 3).

Plain English: who owns each company, quarter by quarter. Two sources, never mixed inside one change:
  archive  the 2015-2026 table SGM built earlier (src/ingest/shareholding/nse.py), packed into
           data_archive/raw_data__shareholding_full_history.tar.gz. Restored as is; its institution split is used
           only from 2022-07 (before that it matched only a few category labels and is wrong: checked against the
           filings 2026-09-30); promoter % is used for every quarter.
  xbrl     each filing's official XBRL (links in data/derived/stock_shareholding.parquet, NSE archives), read here
           for 2021 onward: in-bse-shp:ShareholdingAsAPercentageOfTotalNumberOfShares and NumberOfShareholders
           per CategoryOfShareholdersAxis member. Adds 2021 to mid-2022 and the newest quarters, plus the retail
           shareholder count the archive never had.
Order: filings of stocks that were ever in the Sri Lakshmi (G1) pool since 2021 first (what the A/B needs), the rest
of the market after (--universe). Polite (1.5 s apart), cached (data/raw/shp_xbrl/), resumable, time-boxed.
Definitions (xbrl): mf_pct = MutualFundsOrUti; fii_fpi_pct = max(InstitutionsForeignPortfolioInvestor,
ForeignPortfolioInvestor) (the same line under two filing formats); dii_pct = mutual funds + banks / financial
institutions + insurance + provident & pension funds + AIF + venture capital funds + NBFCs (the archive's list);
retail_holders = NumberOfShareholders of IndividualShareholdersHoldingNominalShareCapitalUpToRsTwoLakh;
promoter_pct = ShareholdingOfPromoterAndPromoterGroup. A category a filing does not report stays blank.
Output: data/derived/shareholding_detail_history.parquet (+ manifest incl. archive-vs-xbrl agreement on overlap).
Usage: fetch_shp_detail.py [--max-minutes N] [--universe]     prints COMPLETE when the pool's filings are all read
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
from src.ingest.nse.session import build_session  # noqa: E402

ARCH = ROOT / "data_archive/raw_data__shareholding_full_history.tar.gz"
MEMBER = "data/shareholding_full_history/normalized/stock_shareholding_quarterly.parquet"
RAW = ROOT / "data/raw/shp_xbrl"
ARCHIVE_OUT = ROOT / "data/derived/shareholding_archive_restored.parquet"
XOUT = RAW / "parsed.parquet"
OUT = ROOT / "data/derived/shareholding_detail_history.parquet"
POOL = ROOT / "data/derived/g1_pool_symbols_2021plus.json"
REF = "https://www.nseindia.com/companies-listing/corporate-filings-shareholding-pattern"
DII = ("MutualFundsOrUtiMember", "FinancialInstitutionOrBanksMember", "InsuranceCompaniesMember",
       "ProvidentFundsOrPensionFundsMember", "AlternativeInvestmentFundsMember", "VentureCapitalFundsMember",
       "NBFCsRegisteredWithRbiMember")          # public-shareholder (Table III) institutions only
DII_EXTRA = re.compile(r"^(OtherFinancialInstitutions|SovereignWealthFunds)\w*Member$")
# NOT IndianFinancialInstitutionsOrBanksMember: that member is a PROMOTER category (Table II), e.g. PNB for PNBHOUSING.


def restore_archive() -> pd.DataFrame:
    if not ARCHIVE_OUT.exists():
        with tarfile.open(ARCH, "r:gz") as tf:
            A = pd.read_parquet(tf.extractfile(MEMBER))
        A.to_parquet(ARCHIVE_OUT, index=False)
        print(f"restored {len(A)} archive rows -> {ARCHIVE_OUT.relative_to(ROOT)}")
    return pd.read_parquet(ARCHIVE_OUT)


def pool_symbols() -> list[str]:
    if POOL.exists():
        return json.loads(POOL.read_text())
    import sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, test_industry_fundamentals as tif  # noqa: E401
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    wk = [d for d in sp.weekly_grid(D["cal"], 0) if d >= pd.Timestamp("2020-10-01")]
    sre.TOPN = 100000
    pool = tif.select_elig(X["F"], wk, imap, P, S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)])
    syms = sorted({s for v in pool.values() for s in v})
    POOL.write_text(json.dumps(syms)); print(f"G1 pool symbols since 2020-10: {len(syms)}")
    return syms


def parse(text: str) -> dict:
    ctx = {}
    for cid, body in re.findall(r'<xbrli:context id="([^"]+)">(.*?)</xbrli:context>', text, re.S):
        mem = re.findall(r'<xbrldi:explicitMember dimension="[^"]+">([^<]+)<', body)
        if len(mem) == 1 and "typedMember" not in body:
            ctx[cid] = mem[0].split(":")[1]
    pct, cnt = {}, {}
    for name, cid, v in re.findall(r'<in-bse-shp:(ShareholdingAsAPercentageOfTotalNumberOfShares|NumberOfShareholders)\s[^>]*contextRef="([^"]+)"[^>]*>([^<]*)<', text):
        m = ctx.get(cid)
        if m is None:
            continue
        try:
            (pct if name.startswith("Share") else cnt)[m] = float(v)
        except ValueError:
            pass
    fpi = [pct[k] for k in ("InstitutionsForeignPortfolioInvestorMember", "ForeignPortfolioInvestorMember") if k in pct]
    return dict(mf_pct=pct.get("MutualFundsOrUtiMember"), fii_fpi_pct=max(fpi) if fpi else None,
                dii_pct=(sum(pct[k] for k in DII if k in pct) + sum(v for k, v in pct.items() if DII_EXTRA.match(k)))
                        if any(k in pct for k in DII) else None,
                promoter_pct=pct.get("ShareholdingOfPromoterAndPromoterGroupMember"),
                retail_holders=cnt.get("IndividualShareholdersHoldingNominalShareCapitalUpToRsTwoLakhMember"),
                total_holders=cnt.get("ShareholdingPatternMember"))


def combine(A: pd.DataFrame, Xp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    a = A.assign(source="archive", quarter_end=pd.to_datetime(A["quarter_end"]),
                 available=pd.to_datetime(A["filing_date"]).fillna(pd.to_datetime(A["quarter_end"]) + pd.Timedelta(days=21)))
    # the archive's institution split is only complete from 2022-07 (older filings: a few labels matched, e.g.
    # CAMLINFINE 2021-03 dii 0.00 vs 18.65 mutual funds in the filing) -> older archive rows keep promoter % only
    old = a["quarter_end"] < "2022-07-01"
    a.loc[old, ["mf_pct", "fii_fpi_pct", "dii_pct"]] = None
    a = a[["symbol", "quarter_end", "available", "source", "mf_pct", "fii_fpi_pct", "dii_pct", "promoter_pct"]]
    x = Xp.assign(source="xbrl")[["symbol", "quarter_end", "available", "source", "mf_pct", "fii_fpi_pct", "dii_pct", "promoter_pct",
                                   "retail_holders", "total_holders"]] if len(Xp) else pd.DataFrame(columns=list(a.columns) + ["retail_holders", "total_holders"])
    both = a.merge(x, on=["symbol", "quarter_end"], suffixes=("_a", "_x"))
    agree = {c: dict(n=int(both[[f"{c}_a", f"{c}_x"]].dropna().shape[0]),
                     median_abs_gap_pts=round(float((both[f"{c}_a"] - both[f"{c}_x"]).abs().median()), 3) if len(both) else None,
                     within_0_1_pts=round(float(((both[f"{c}_a"] - both[f"{c}_x"]).abs() <= 0.1).mean()), 3) if len(both) else None)
             for c in ("mf_pct", "fii_fpi_pct", "dii_pct", "promoter_pct")}
    keyx = set(zip(x["symbol"], x["quarter_end"]))
    a_only = a[[k not in keyx for k in zip(a["symbol"], a["quarter_end"])]]
    H = pd.concat([x, a_only], ignore_index=True).sort_values(["symbol", "quarter_end"]).reset_index(drop=True)
    return H, agree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--universe", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    RAW.mkdir(parents=True, exist_ok=True)
    A = restore_archive()
    L = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet", columns=["symbol", "quarter_end", "submission", "xbrl"])
    L["quarter_end"] = pd.to_datetime(L["quarter_end"], format="mixed", errors="coerce")
    L["submission"] = pd.to_datetime(L["submission"], format="mixed", errors="coerce")
    L = L[L["xbrl"].notna() & (L["quarter_end"] >= "2021-01-01") & L["quarter_end"].dt.is_month_end & L["quarter_end"].dt.month.isin([3, 6, 9, 12])]
    L = L.sort_values("submission").drop_duplicates(["symbol", "quarter_end"], keep="last")   # latest revision of each quarter
    pool = set(pool_symbols())
    need = L if a.universe else L[L["symbol"].isin(pool)]
    Xp = pd.read_parquet(XOUT) if XOUT.exists() else pd.DataFrame(columns=["xbrl"])
    gone_f = RAW / "not_on_nse.json"
    gone = set(json.loads(gone_f.read_text())) if gone_f.exists() else set()
    todo = need[~need["xbrl"].isin(set(Xp["xbrl"]) | gone)]
    print(f"filings 2021+: {len(L)} · needed now ({'universe' if a.universe else 'G1 pool'}): {len(need)} · parsed before {len(set(Xp['xbrl']) & set(need['xbrl']))} · to read {len(todo)}", flush=True)
    s = build_session(warm=True, referer=REF)
    rows, errors = [], 0
    for i, r in enumerate(todo.itertuples(), 1):
        if a.max_minutes and time.time() - t0 > a.max_minutes * 60:
            print(f"time box reached after {i - 1} filings", flush=True); break
        f = RAW / r.xbrl.rsplit("/", 1)[-1]
        try:
            if not f.exists():
                resp = s.get(r.xbrl, timeout=60)
                if resp.status_code == 404:
                    gone.add(r.xbrl); gone_f.write_text(json.dumps(sorted(gone))); continue
                resp.raise_for_status(); f.write_bytes(resp.content); time.sleep(1.5)
            rows.append(dict(symbol=r.symbol, quarter_end=r.quarter_end, available=r.submission, xbrl=r.xbrl,
                             **parse(f.read_text(encoding="utf-8", errors="ignore"))))
        except Exception as x:
            errors += 1; print(f"  ERR {r.symbol} {r.quarter_end.date()}: {type(x).__name__} {str(x)[:80]}", flush=True)
        if i % 250 == 0:
            print(f"  read {i}/{len(todo)}", flush=True)
            Xp = pd.concat([Xp, pd.DataFrame(rows)], ignore_index=True); Xp.to_parquet(XOUT, index=False); rows = []
    if rows:
        Xp = pd.concat([Xp, pd.DataFrame(rows)], ignore_index=True); Xp.to_parquet(XOUT, index=False)
    H, agree = combine(A, Xp)
    H.to_parquet(OUT, index=False)
    left = int((~need["xbrl"].isin(set(Xp["xbrl"]) | gone)).sum())
    cov = H.assign(y=H["quarter_end"].dt.year).groupby("y")[["mf_pct", "fii_fpi_pct", "dii_pct", "retail_holders"]].apply(lambda g: g.notna().mean().round(2))
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="shareholding_detail_history", path=str(OUT.relative_to(ROOT)), rows=len(H), symbols=int(H["symbol"].nunique()),
        quarters=[str(H["quarter_end"].min().date()), str(H["quarter_end"].max().date())], definitions=__doc__.split("Definitions (xbrl): ")[1].split("Output:")[0],
        columns=dict(available="date the numbers were public: filing / submission date; archive rows without one use quarter end + 21 days (SEBI deadline)",
                     source="archive | xbrl (changes are only computed between two rows of the same source)",
                     mf_pct="percent of total shares", fii_fpi_pct="percent of total shares", dii_pct="percent of total shares",
                     promoter_pct="percent of total shares", retail_holders="number of shareholders", total_holders="number of shareholders"),
        coverage_by_year={str(k): v for k, v in cov.to_dict("index").items()}, archive_vs_xbrl_on_overlap=agree,
        xbrl_filings_not_on_nse=len(gone), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"parsed {len(Xp)} xbrl filings total · errors {errors} · pool filings left {left} · archive-vs-xbrl agreement {json.dumps(agree)} · {time.time() - t0:.0f}s")
    if left == 0 and errors == 0:
        print("COMPLETE")


if __name__ == "__main__":
    main()
