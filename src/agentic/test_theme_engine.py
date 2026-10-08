"""EXP-2026-09-30-theme-engine (registered, then amended, in logs/experiments.jsonl before any outcome was computed).

Plain English: does it help Sri Lakshmi v2 to put first the stocks whose own filings tie them to a theme that is
rising (government attention or global prices), and to skip stocks tied only to fading themes?
Reference V2 = G1 top 9 minus Financial Services, no refill (production from 2026-10-02).
  TUP    pool names (rank <= 25) exposed to a rising theme first, then model order; top 9; financials dropped
  TFADE  V2 minus names exposed only to fading themes (none rising); no refill
  TBOTH  TUP ordering, then the TFADE filter
Data: data/derived/theme_intensity.parquet, theme_exposure.parquet (src/agentic/build_themes.py, configs/themes.json).
PASS = sp.beats vs V2 at phase 0 and in >= 4 of 5 phases. Output: logs/leader_sleeve/theme_engine/ + RESULT line.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-30-theme-engine"
OUTDIR = ROOT / "logs/leader_sleeve/theme_engine"
ARMS = ("TUP", "TFADE", "TBOTH")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"]); sc["industry"] = sc["industry"].map(html.unescape)
    sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
    fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
    I = pd.read_parquet(ROOT / "data/derived/theme_intensity.parquet"); I["date"] = pd.to_datetime(I["date"])
    state = {(d, t): st for d, t, st in zip(I["date"], I["theme"], I["state"])}
    E = pd.read_parquet(ROOT / "data/derived/theme_exposure.parquet"); E["an_dt"] = pd.to_datetime(E["an_dt"])
    ex = {(s, t): np.sort(g["an_dt"].to_numpy()) for (s, t), g in E.groupby(["symbol", "theme"])}
    themes_of = {}
    for (s, t) in ex:
        themes_of.setdefault(s, []).append(t)

    def exposure(s, d):
        d64 = np.datetime64(d + pd.Timedelta(hours=23, minutes=59))
        out = []
        for t in themes_of.get(s, []):
            a = ex[(s, t)]
            if np.searchsorted(a, d64, side="right") - np.searchsorted(a, d64 - np.timedelta64(365, "D"), side="right") > 0:
                out.append(state.get((d, t), "n/a"))
        return out

    rising = lambda s, d: "rising" in exposure(s, d)  # noqa: E731
    fading_only = lambda s, d: (lambda st: "fading" in st and "rising" not in st)(exposure(s, d))  # noqa: E731
    grid = lambda o: [d for d in sp.weekly_grid(D["cal"], o) if d >= pd.Timestamp(tif.START)]  # noqa: E731

    def run_phase(o: int) -> dict:
        wk = grid(o)
        ref9 = tif.select_elig(X["F"], wk, imap, P, G1)
        top = sre.TOPN; sre.TOPN = 100000
        full = tif.select_elig(X["F"], wk, imap, P, G1)
        sre.TOPN = top
        v2 = {d: [s for s in full.get(d, [])[:9] if not fin(s)] for d in wk}
        if o == 0:
            same = sum(v2[d] == [s for s in ref9.get(d, []) if not fin(s)] for d in wk)
            print(f"wiring check: V2 from the full pool equals V2 from the G1 top 9 on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — no result reported")
            cov = pd.Series({d: np.mean([bool(exposure(s, d)) for s in v2[d]]) if v2[d] else np.nan for d in wk})
            print("share of V2 picks tied to any theme, by year: " + str((cov.groupby(pd.DatetimeIndex(cov.index).year).mean() * 100).round(0).to_dict()), flush=True)

        def tup(d):
            first = [s for s in full.get(d, [])[:25] if rising(s, d)]
            return [s for s in (first + [s for s in full.get(d, []) if s not in first])[:9] if not fin(s)]
        sels = {"G1": ref9, "V2": v2, "TUP": {d: tup(d) for d in wk},
                "TFADE": {d: [s for s in v2[d] if not fading_only(s, d)] for d in wk},
                "G1T": {d: [s for s in ref9.get(d, []) if not fading_only(s, d)] for d in wk}}   # EXP-2026-09-30-g1-fade
        sels["TBOTH"] = {d: [s for s in sels["TUP"][d] if not fading_only(s, d)] for d in wk}
        if o == 0:
            print("weeks where the arm changed the picks: " + " · ".join(f"{a} {sum(sels[a][d] != v2[d] for d in wk)}" for a in ARMS), flush=True)
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")}, avg_names=info["avg_names"])
        return res

    if os.environ.get("SGM_V3_VS_G1") == "1":   # data-readiness gate (2026-10-08) before any outcome of the V3 adoption test
        sys.path.insert(0, str(ROOT / "src/agentic/trust"))
        import data_ready as dr
        cal, wk0 = D["cal"], grid(0)
        yrs = range(pd.Timestamp(tif.START).year, cal[-1].year + 1)
        Px = pd.to_datetime(pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])["trade_date"])
        bh = dr.bhav_rows()
        top = sre.TOPN; sre.TOPN = 100000
        pool0 = tif.select_elig(X["F"], wk0, imap, P, G1)
        sre.TOPN = top
        PS = {s: np.sort(g["trade_date"].to_numpy()) for s, g in P.groupby("symbol")}

        def scored(s, d):
            a = PS.get(s)
            if a is None:
                return np.nan
            i = np.searchsorted(a, np.datetime64(d), side="right") - 1
            return 1.0 if i >= 0 and np.datetime64(d) - a[i] <= np.timedelta64(7, "D") else np.nan
        pool_rows = pd.DataFrame([dict(d=d, known=scored(s, d)) for d in wk0 for s in pool0.get(d, [])])
        Fw = X["F"][X["F"]["trade_date"].isin(set(wk0)) & X["F"]["core"]][["symbol", "trade_date"]].copy()
        Fw["ind"] = Fw["symbol"].map(imap).notna().astype(float).where(lambda x: x > 0)
        A = pd.read_csv(ROOT / "logs/news/announcements_completeness.csv").dropna(subset=["nse"]); A = A[A["nse"] > 0]
        end = cal[-1] - pd.Timedelta(days=10)
        checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(),
                                  bh.groupby(bh.index.year).sum(), range(yrs.start - 1, yrs.stop)),
                  dr.coverage("model score known for G1-pool names on list dates (phase 0)", pool_rows, "d", ["known"], yrs),
                  dr.coverage("industry known for core stocks on list dates (phase 0)", Fw, "trade_date", ["ind"], yrs),
                  dr.completeness("NSE announcements we hold vs NSE's own day feed (theme-exposure source)", A.groupby("year")["ours"].sum(),
                                  A.groupby("year")["nse"].sum(), range(yrs.start - 1, yrs.stop)),
                  dr.span("model scores", P["trade_date"], pd.Timestamp(tif.START), end),
                  dr.span("industry heat scores (G1 eligibility)", S["date"], pd.Timestamp(tif.START), end),
                  dr.span("theme states", I["date"], pd.Timestamp(tif.START), end),
                  dr.span("theme exposure filings", E["an_dt"], pd.Timestamp(tif.START) - pd.Timedelta(days=365), end)]
        if os.environ.get("SGM_REPRO") != "1":
            dr.gate("EXP-2026-09-30-v3-vs-g1", checks, run=os.environ.get("SGM_RERUN_TAG") or "RESULT")
    R0 = run_phase(0)
    if os.environ.get("SGM_V3_VS_G1") == "1":   # EXP-2026-09-30-v3-vs-g1 (V3 = TFADE) and EXP-2026-09-30-g1-fade (G1T), both vs G1
        vs = ("V3", "G1T")
        res = {"G1": R0["G1"], "V3": R0["TFADE"], "G1T": R0["G1T"]}
        ph = {k: [bool(sp.beats(res[k], res["G1"]))] for k in vs}
        for o in (1, 2, 3, 4):
            Ro = run_phase(o); Ro["V3"] = Ro["TFADE"]
            for k in vs:
                ph[k].append(bool(sp.beats(Ro[k], Ro["G1"])))
        for k in ("G1", *vs):
            r = res[k]; print(f"{k:3s}: CAGR {r['cagr']:.1f} · 2019-22 {r['cagr_disc']:.1f} · 2023+ {r['cagr_conf']:.1f} · maxDD {r['maxdd']:.1f} · names {r['avg_names']:.1f}"
                              + (f" · beats G1 in {sum(ph[k])}/5 phases · PASS {bool(ph[k][0] and sum(ph[k]) >= 4)}" if k in vs else ""))
        if os.environ.get("SGM_REPRO") == "1":
            print("REPRO RUN (not logged)"); return
        tag, now = os.environ.get("SGM_RERUN_TAG", ""), datetime.now().isoformat(timespec="seconds")
        arms = lambda k: {g: [round(res[g][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph[k])}/5"] if g == k else [None]) for g in ("G1", k)}  # noqa: E731
        ok = lambda k: bool(ph[k][0] and sum(ph[k]) >= 4)  # noqa: E731
        with (ROOT / "logs/experiments.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(ts=now, id="EXP-2026-09-30-v3-vs-g1" + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3"] if ok("V3") else "no arm passes",
                                     note=("same registered test re-run: " + tag) if tag else None, arms=arms("V3"), cols="CAGR, disc, conf, maxDD, phases")) + "\n")
            if os.environ.get("SGM_G1T") == "1":
                fh.write(json.dumps(dict(ts=now, id="EXP-2026-09-30-g1-fade-RESULT", verdict=["G1T"] if ok("G1T") else "no arm passes",
                                         arms=arms("G1T"), cols="CAGR, disc, conf, maxDD, phases")) + "\n")
        return
    ph = {k: [bool(sp.beats(R0[k], R0["V2"]))] for k in ARMS}
    for o in (1, 2, 3, 4):
        Ro = run_phase(o)
        for k in ARMS:
            ph[k].append(bool(sp.beats(Ro[k], Ro["V2"])))
    for k in ARMS:
        R0[k]["phases_beaten"] = f"{sum(ph[k])}/5"; R0[k]["PASS"] = bool(ph[k][0] and sum(ph[k]) >= 4)
    print(f"\n=== {EXP_ID} · reference V2 ===")
    for k in ["V2", *ARMS]:
        r = R0[k]
        print(f"{k:5s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f} · "
              f"phases {r.get('phases_beaten', '')} · PASS {r.get('PASS', '')}")
    repro = os.environ.get("SGM_REPRO") == "1"
    out = OUTDIR / "repro" if repro else OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **r) for k, r in R0.items()]).to_csv(out / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_theme_engine.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), updated=now), indent=1))
    passed = [k for k in ARMS if R0[k]["PASS"]]
    (out / "README.md").write_text(f"# {EXP_ID}\n\nTheme engine vs Sri Lakshmi v2. Verdict: {'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}.\n")
    if repro:
        print("REPRO RUN (not logged)"); return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=passed or "no arm passes",
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats V2 by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
