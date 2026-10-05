"""EXP-2026-10-04-v3-leverage (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: does dropping V3 picks that are stretched on debt avoid losers without costing return?
Arms (no refill in any): V3 (live rule, v3_rule.picks) · V3_IC minus weak interest cover · V3_CR minus recent credit
events · V3_LEV minus both.
  interest cover = sum over the latest 4 consecutive quarters filed before the decision date (one basis, consolidated
  first) of (profit before tax + finance cost) / sum of finance cost. Weak = < 1.5x, or a loss before interest while
  paying interest. Finance cost 0 = no interest = kept; not computable = kept (the data gate counts it).
  credit event = event_ledger bucket rating_down or default_insolvency filed in the 180 days before the decision date.
  Tickers: a pick's old ticker is mapped to today's (nse_symbols, NSE symbolchange.csv) before the P&L / ledger lookup.
Engine: sim_screen_rank_exit.run_exit(D, X, picks, weekly grid, "E0"), weekly entries 2019+ (tif.START), phases 0-4.
PASS (per arm vs V3) = sp.beats (CAGR higher in 2019-22 and 2023+, max drawdown not worse by > 2 pts) at phase 0 and
in >= 4 of 5 phases. Diagnostic: loser catch rate on phase-0 picks (dropped vs kept: share ending <= -30% at session
126, share touching +50%).
Before any outcome: trust/data_ready.gate(). Wiring check: V3 from the full G1 pool equals v3_rule.picks.
SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402
import nse_symbols  # noqa: E402
import v3_rule  # noqa: E402

EXP_ID = "EXP-2026-10-04-v3-leverage"
OUTDIR = ROOT / "logs/leader_sleeve/v3_leverage"
IC_MIN, CR_DAYS = 1.5, 180
ARMS = ("V3", "V3_IC", "V3_CR", "V3_LEV")


def interest_cover(symbols: set | None = None) -> dict:
    """{symbol: (filing times, cover, status)} after each filing: status ok / weak / none (no interest) / na."""
    Q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet",
                        columns=["symbol", "quarter_end", "filing_dt", "basis", "source", "pbt", "finance_cost"])
    if symbols is not None:
        Q = Q[Q["symbol"].isin(symbols)]
    Q["quarter_end"] = pd.to_datetime(Q["quarter_end"]); Q["filing_dt"] = pd.to_datetime(Q["filing_dt"], errors="coerce")
    f = np.where(Q["source"] == "xbrl", 1e-5, 1.0)                    # -> Rs lakh (manifest: xbrl rupees, others lakh)
    Q["pbt"], Q["fc"] = Q["pbt"] * f, Q["finance_cost"] * f
    Q = Q.dropna(subset=["filing_dt"]).sort_values("filing_dt")
    out = {}
    for s, g in Q.groupby("symbol"):
        ts, val = [], []
        for t in g["filing_dt"].unique():
            h = g[g["filing_dt"] <= t]
            res = ("na", np.nan)
            for b in ("con", "sa"):
                x = h[h["basis"] == b].drop_duplicates("quarter_end", keep="first").sort_values("quarter_end").tail(4)
                if len(x) == 4 and (x["quarter_end"].iloc[-1] - x["quarter_end"].iloc[0]).days in range(260, 285) \
                        and x["pbt"].notna().all() and x["fc"].notna().all():
                    fc, ebit = float(x["fc"].sum()), float((x["pbt"] + x["fc"]).sum())
                    res = ("none", np.inf) if fc <= 0 else (("weak" if (ebit <= 0 or ebit / fc < IC_MIN) else "ok"), ebit / fc)
                    break
            ts.append(t); val.append(res)
        out[s] = (np.array(ts, dtype="datetime64[ns]"), val)
    return out


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    START = pd.Timestamp(tif.START)
    V3ALL = {o: v3_rule.picks(X["F"], [d for d in sp.weekly_grid(cal, o) if d >= START], imap, P, ctx) for o in range(5)}
    IC = interest_cover({nse_symbols.now(s) for v in V3ALL.values() for names in v.values() for s in names})   # P&L is keyed by today's ticker

    def ic_at(s, d):
        a = IC.get(nse_symbols.now(s))                                   # a 2019 pick may carry its old ticker (NIITTECH -> COFORGE)
        if a is None:
            return ("na", np.nan)
        i = np.searchsorted(a[0], np.datetime64(pd.Timestamp(d)), side="left") - 1    # filed strictly before the decision date
        return a[1][i] if i >= 0 else ("na", np.nan)
    L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket"])
    L = L[L["bucket"].isin(["rating_down", "default_insolvency"])]
    CR = {s: np.sort(pd.to_datetime(g["filed_at"]).values) for s, g in L.groupby("symbol")}

    def cr_at(s, d):
        a = [x for x in (CR.get(s), CR.get(nse_symbols.now(s))) if x is not None]   # ledger: ticker as filed or today's
        if not a:
            return False
        a = np.concatenate(a)
        d = np.datetime64(pd.Timestamp(d)); return bool(((a <= d) & (a > d - np.timedelta64(CR_DAYS, "D"))).any())

    def arms(o):
        v3 = V3ALL[o]
        out = {"V3": v3, "V3_IC": {}, "V3_CR": {}, "V3_LEV": {}}
        for d, names in v3.items():
            ic = {s: ic_at(s, d)[0] == "weak" for s in names}; cr = {s: cr_at(s, d) for s in names}
            out["V3_IC"][d] = [s for s in names if not ic[s]]
            out["V3_CR"][d] = [s for s in names if not cr[s]]
            out["V3_LEV"][d] = [s for s in names if not ic[s] and not cr[s]]
        return out

    # ---- data-readiness gate (before any outcome)
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    v30 = V3ALL[0]
    rows = pd.DataFrame([dict(d=d, s=s, ic=ic_at(s, d)[0]) for d, v in v30.items() for s in v])
    rows["known"] = np.where(rows["ic"] != "na", 1.0, np.nan)
    Pq = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "net_sales"]); Pq["quarter_end"] = pd.to_datetime(Pq["quarter_end"])
    have = Pq.dropna(subset=["net_sales"]).drop_duplicates(["symbol", "quarter_end"]).groupby(Pq["quarter_end"].dt.year).size()
    Cf = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "toDate", "period"])
    Cf = Cf[Cf["period"].astype(str).str.lower() == "quarterly"]; Cf["qe"] = pd.to_datetime(Cf["toDate"], format="%d-%b-%Y", errors="coerce")
    Ci = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet", columns=["symbol", "period_to"]); Ci["qe"] = pd.to_datetime(Ci["period_to"], format="%d-%b-%Y", errors="coerce")
    off = pd.concat([Cf[["symbol", "qe"]], Ci[["symbol", "qe"]]]).dropna().drop_duplicates()
    A = pd.read_csv(ROOT / "logs/news/announcements_completeness.csv").dropna(subset=["nse"]); A = A[A["nse"] > 0]
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.completeness("live P&L company-quarters vs NSE's results calendar + integrated filings", have, off.groupby(off["qe"].dt.year).size(), range(START.year - 1, cal[-1].year + 1)),
              dr.coverage("interest cover computable for V3 picks (phase 0)", rows, "d", ["known"], yrs),
              dr.completeness("NSE announcements we hold vs NSE's own day feed (event-ledger source)", A.groupby("year")["ours"].sum(), A.groupby("year")["nse"].sum(), yrs),
              dr.span("event ledger", L["filed_at"], START - pd.Timedelta(days=CR_DAYS), cal[-1] - pd.Timedelta(days=10))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check: V3 from the full G1 pool, cut to the top 9, then the keep() filter == v3_rule.picks
    top = sre.TOPN; sre.TOPN = 100000
    full = tif.select_elig(X["F"], wk0, imap, P, ctx["G1"]); sre.TOPN = top
    alt = {d: [s for s in full.get(d, [])[:9] if ctx["keep"](s, d)] for d in wk0}
    same = sum(alt[d] == v30[d] for d in wk0)
    print(f"wiring check: V3 from the full pool equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")

    # ---- outcomes
    res, ph = {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        sel = arms(o)
        R = {}
        for a in ARMS:
            nav, info = sre.run_exit(D, X, sel[a], wk, "E0")
            m = sp.metrics(nav); R[a] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"])
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3"])))
        if o == 0:
            res = R; sel0 = sel; wk_0 = wk
            for a in ARMS[1:]:
                n_drop = sum(len(sel["V3"][d]) - len(sel[a][d]) for d in wk)
                print(f"{a}: dropped {n_drop} of {sum(len(v) for v in sel['V3'].values())} picks · avg names {R[a]['avg_names']:.1f}", flush=True)
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]

    # ---- loser catch rate (phase 0): each pick's session-126 return and whether it touched +50%
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
    pr = []
    for d in wk_0:
        i0 = cal.get_loc(d) + 1
        if i0 + 126 > len(cal):
            continue
        for s in sel0["V3"][d]:
            e = O[s].iloc[i0] if s in O.columns else np.nan
            if not np.isfinite(e):
                continue
            pr.append(dict(d=d, s=s, ret=C[s].iloc[i0 + 125] / e - 1, hit50=bool((Hh[s].iloc[i0:i0 + 126] >= 1.5 * e).any()),
                           ic=ic_at(s, d)[0] == "weak", cr=cr_at(s, d)))
    Pk = pd.DataFrame(pr); diag = {}
    print(f"\n=== {EXP_ID} · weekly entries from {START.date()} · loser catch rate (phase 0, {len(Pk)} picks with 126 sessions) ===")
    for a, m in (("V3_IC", Pk["ic"]), ("V3_CR", Pk["cr"]), ("V3_LEV", Pk["ic"] | Pk["cr"])):
        dp, kp = Pk[m], Pk[~m]
        diag[a] = dict(dropped=len(dp), dropped_lost30=round(float((dp.ret <= -0.3).mean()), 3) if len(dp) else None,
                       kept_lost30=round(float((kp.ret <= -0.3).mean()), 3), dropped_hit50=round(float(dp.hit50.mean()), 3) if len(dp) else None,
                       kept_hit50=round(float(kp.hit50.mean()), 3), dropped_avg=round(float(dp.ret.mean()), 3) if len(dp) else None, kept_avg=round(float(kp.ret.mean()), 3))
        print(f"{a}: dropped {len(dp)} · ended <= -30%: dropped {diag[a]['dropped_lost30']} vs kept {diag[a]['kept_lost30']} · "
              f"touched +50%: dropped {diag[a]['dropped_hit50']} vs kept {diag[a]['kept_hit50']} · avg 6m: dropped {diag[a]['dropped_avg']} vs kept {diag[a]['kept_avg']}")
    print()
    for a in ARMS:
        r = res[a]
        print(f"{a:6s} CAGR {r['cagr']:5.1f} · {START.year}-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in ph else None) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_leverage.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), loser_catch=diag, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nDoes dropping V3 picks with weak interest cover or recent credit events avoid losers without costing return? "
                                      f"Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. Numbers in results.csv; definitions and the loser catch rate in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=verdict or "no arm passes", arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 loser_catch=diag, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
