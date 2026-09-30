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
                "TFADE": {d: [s for s in v2[d] if not fading_only(s, d)] for d in wk}}
        sels["TBOTH"] = {d: [s for s in sels["TUP"][d] if not fading_only(s, d)] for d in wk}
        if o == 0:
            print("weeks where the arm changed the picks: " + " · ".join(f"{a} {sum(sels[a][d] != v2[d] for d in wk)}" for a in ARMS), flush=True)
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")}, avg_names=info["avg_names"])
        return res

    R0 = run_phase(0)
    if os.environ.get("SGM_V3_VS_G1") == "1":                       # EXP-2026-09-30-v3-vs-g1: V3 (= TFADE) against G1
        res = {"G1": R0["G1"], "V3": R0["TFADE"]}
        ph = [bool(sp.beats(res["V3"], res["G1"]))]
        for o in (1, 2, 3, 4):
            Ro = run_phase(o); ph.append(bool(sp.beats(Ro["TFADE"], Ro["G1"])))
        passed = bool(ph[0] and sum(ph) >= 4)
        for k in ("G1", "V3"):
            r = res[k]; print(f"{k}: CAGR {r['cagr']:.1f} · 2019-22 {r['cagr_disc']:.1f} · 2023+ {r['cagr_conf']:.1f} · maxDD {r['maxdd']:.1f}")
        print(f"V3 beats G1 in {sum(ph)}/5 phases · PASS {passed}")
        with (ROOT / "logs/experiments.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-09-30-v3-vs-g1-RESULT", verdict=["V3"] if passed else "no arm passes",
                                     arms={k: [round(res[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if k == "V3" else [None]) for k in res},
                                     cols="CAGR, disc, conf, maxDD, phases")) + "\n")
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
