"""EXP-2026-10-11-v3-badnews-refill (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: replace a V3 pick whose company filed bad news in the 90 days before the list date with the next-ranked G1
name (instead of leaving cash, the flaw of EXP-2026-10-07-v3-badnews-veto). Arms (full G1 pool by model rank -> remove flagged
names -> top 9 -> V3's financial / fading-only filters, no refill for those, exactly as V3):
  V3         live rule (v3_rule.picks)
  V3_BADF    refill after removing bad news: default_insolvency, mgmt_exit, independent_director_exit, auditor_exit, suspension,
             order_cancel, delayed_results, strike_disruption (event_ledger, filed in the 90 calendar days up to the list
             date's 15:30 close)  — primary
  V3_BADREGF V3_BADF plus regulatory_action — diagnostic only, no verdict (its category settles only in late 2024)
Window: weekly entries from 2023-01-01; eras 2023-24 and 2025+ (CAGR of the NAV inside each era); phases 0-4.
Engine: sim_screen_rank_exit.run_exit(..., "E0"), 126-session hold. PASS (verdict arms vs V3) = CAGR higher in both eras,
full maxDD not worse by more than 2 pts, at phase 0 and in >= 4 of 5. One era of regime -> a pass is provisional.
Before any outcome: trust/data_ready.gate(). SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
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
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402
import v3_rule  # noqa: E402

EXP_ID = "EXP-2026-10-11-v3-badnews-refill"
OUTDIR = ROOT / "logs/leader_sleeve/v3_badnews_refill"
START, SPLIT, LOOKBACK = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01"), pd.Timedelta(days=90)
CLOSE = pd.Timedelta(hours=15, minutes=30)
BAD = ["default_insolvency", "mgmt_exit", "independent_director_exit", "auditor_exit", "suspension", "order_cancel",
       "delayed_results", "strike_disruption"]
SETS = {"V3_BADF": BAD, "V3_BADREGF": BAD + ["regulatory_action"]}
ARMS = ("V3", "V3_BADF", "V3_BADREGF")
VERDICT_ARMS = ("V3_BADF",)
STABLE_MIN = 0.4


def era_metrics(nav: pd.Series) -> dict:
    a, b = sp.seg_metrics(nav, START, SPLIT), sp.seg_metrics(nav, SPLIT, None)
    full = rp.nav_metrics(nav)
    return dict(cagr=full["cagr"], cagr_e1=a["cagr"], cagr_e2=b["cagr"], maxdd=full["maxdd"], final=nav.iloc[-1] / nav.iloc[0])


def beats(r: dict, base: dict) -> bool:
    return bool(r["cagr_e1"] > base["cagr_e1"] and r["cagr_e2"] > base["cagr_e2"] and r["maxdd"] >= base["maxdd"] - 2)


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket", "direction"])
    L["filed_at"] = pd.to_datetime(L["filed_at"]); L["sym"] = L["symbol"].map(nse_symbols.now)
    EV = {a: {s: np.sort(g["filed_at"].values) for s, g in L[L["bucket"].isin(b)].groupby("sym")} for a, b in SETS.items()}

    def flagged(a, s, d):
        x = EV[a].get(nse_symbols.now(s))
        if x is None:
            return False
        hi, lo = np.datetime64(d + CLOSE), np.datetime64(d + CLOSE - LOOKBACK)
        return bool(((x <= hi) & (x > lo)).any())

    def pool(wk):
        top = sre.TOPN; sre.TOPN = 100000
        try:
            return tif.select_elig(X["F"], wk, imap, P, ctx["G1"])
        finally:
            sre.TOPN = top

    def arms(wk, flag_off=False):
        v3 = v3_rule.picks(X["F"], wk, imap, P, ctx)
        full = pool(wk)
        out = {"V3": v3}
        for a in ARMS[1:]:
            out[a] = {d: [s for s in [x for x in full.get(d, []) if flag_off or not flagged(a, x, d)][:9] if ctx["keep"](s, d)] for d in wk}
        return out

    # ---- data-readiness gate (before any outcome)
    A = pd.read_csv(ROOT / "logs/news/announcements_completeness.csv").dropna(subset=["nse"]); A = A[A["nse"] > 0]
    yrs = range(2022, cal[-1].year + 1)
    Px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date"])
    Px["trade_date"] = pd.to_datetime(Px["trade_date"]); Px = Px[Px["trade_date"] >= START]
    listed = Px.groupby(Px["trade_date"].dt.year)["symbol"].nunique()
    days = Px.groupby(Px["trade_date"].dt.year)["trade_date"].agg(lambda x: (x.max() - pd.Timestamp(f"{x.min().year}-01-01")).days + 1)
    stab, checks = {}, []
    checks.append(dr.completeness("NSE announcements we hold vs NSE's own day feed (sample days)", A.groupby("year")["ours"].sum(), A.groupby("year")["nse"].sum(), yrs))
    checks.append(dr.span("event ledger", L["filed_at"], START - LOOKBACK, cal[-1] - pd.Timedelta(days=10)))
    for a in ("V3_BADF", "V3_BADREGF"):
        e = L[L["bucket"].isin(SETS[a]) & (L["filed_at"] >= START)]
        rate = e.groupby(e["filed_at"].dt.year).size() / listed * 365 / days * 1000
        stab[a] = {int(y): round(float(v), 1) for y, v in rate.items()}
        ratio = float(rate.min() / rate.max())
        print(f"{a} events per 1,000 listed companies a year (annualised): {stab[a]} · lowest/highest {ratio:.2f}")
        if a in VERDICT_ARMS:
            checks.append(dr.qc(f"{a} event types steady 2023+ (lowest / highest annualised yearly rate)", ratio, STABLE_MIN))
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring check: the refill path with nothing flagged reproduces the live rule exactly
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    sel0 = arms(wk0)
    off = arms(wk0, flag_off=True)
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    same = sum(off["V3_BADF"][d] == live.get(d, []) for d in wk0)
    print(f"wiring check: refill path with nothing flagged equals v3_rule.picks on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")
    for a in ARMS[1:]:
        ch = sum(len(set(sel0[a][d]) - set(sel0["V3"][d])) for d in wk0)
        print(f"{a}: names swapped in vs V3 {ch} (phase 0)", flush=True)
    for a in ARMS[1:]:
        n = sum(len(set(sel0["V3"][d]) - set(sel0[a][d])) for d in wk0)
        print(f"{a}: V3 picks removed {n} of {sum(len(v) for v in sel0['V3'].values())} · avg names {np.mean([len(sel0[a][d]) for d in wk0]):.1f} vs V3 {np.mean([len(sel0['V3'][d]) for d in wk0]):.1f} (phase 0)", flush=True)

    # ---- outcomes
    res, ph = {}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        sel = sel0 if o == 0 else arms(wk)
        R = {}
        for a in ARMS:
            nav, info = sre.run_exit(D, X, sel[a], wk, "E0")
            R[a] = dict(**era_metrics(nav), avg_names=info["avg_names"])
        for a in ARMS[1:]:
            ph[a].append(beats(R[a], R["V3"]))
        if o == 0:
            res = R
        print(f"phase {o}: " + " · ".join(f"{a} {R[a]['cagr_e1']:.1f}/{R[a]['cagr_e2']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    verdict = [a for a in VERDICT_ARMS if ph[a][0] and sum(ph[a]) >= 4]

    # ---- dropped vs kept (phase 0)
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX

    def outcome(s, d):
        i0 = cal.get_loc(d) + 1
        if i0 + 126 > len(cal) or s not in O.columns or not np.isfinite(O[s].iloc[i0]) or not np.isfinite(C[s].iloc[i0 + 125]):
            return None
        e = O[s].iloc[i0]; return C[s].iloc[i0 + 125] / e - 1, bool((Hh[s].iloc[i0:i0 + 126] >= 1.5 * e).any())
    f = lambda v: dict(n=len(v), avg=round(float(np.mean([r for r, _ in v])), 3) if v else None, median=round(float(np.median([r for r, _ in v])), 3) if v else None,  # noqa: E731
                       hit50=round(float(np.mean([h for _, h in v])), 3) if v else None, lost30=round(float(np.mean([r <= -0.3 for r, _ in v])), 3) if v else None)
    diag = {}
    for a in ARMS[1:]:
        drop, keep = [], []
        for d in wk0:
            for s in sel0["V3"][d]:
                x = outcome(s, d)
                if x:
                    (keep if s in sel0[a][d] else drop).append(x)
        diag[a] = dict(dropped=f(drop), kept=f(keep))
        print(f"{a}: dropped {diag[a]['dropped']} · kept {diag[a]['kept']}")
    print(f"\n=== {EXP_ID} · weekly entries from {START.date()} to {cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:9s} CAGR {r['cagr']:5.1f} · 2023-24 {r['cagr_e1']:5.1f} · 2025+ {r['cagr_e2']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
              + (f" · beats V3 in {sum(ph[a])}/5 phases (phase 0 {'yes' if ph[a][0] else 'no'})" if a != "V3" else "")
              + (" · diagnostic, no verdict" if a == "V3_BADREGF" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS (provisional: one era)' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **res[a], phases_beaten=(f"{sum(ph[a])}/5" if a in ph else None), PASS=(a in verdict) if a in VERDICT_ARMS else None)
                  for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(
        dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_badnews_refill.py", definitions=__doc__,
        columns=dict(cagr="percent a year, full window", cagr_e1="percent a year inside 2023-01-01..2024-12-31", cagr_e2="percent a year from 2025-01-01",
                     maxdd="percent, full window", final="NAV multiple", avg_names="average names held", phases_beaten="phases where the arm beats V3",
                     PASS="verdict arm only; V3_BADREGF is diagnostic"),
        event_rates_per_1000_listed=stab, dropped_vs_kept=diag, phase0_only_for_metrics=True, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nDoes replacing V3 picks that had bad news in the prior 90 days with the next-ranked name beat V3 (2023+)? "
                                      f"Verdict: {', '.join(verdict) + ' PASS (provisional)' if verdict else 'no arm passes'}. Numbers in results.csv (phase 0); "
                                      "definitions, event rates and dropped-vs-kept stats in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_e1", "cagr_e2", "maxdd")] + [f"{sum(ph[a])}/5" if a in ph else None] for a in ARMS},
                                 dropped_vs_kept=diag, cols="CAGR, 2023-24, 2025+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
