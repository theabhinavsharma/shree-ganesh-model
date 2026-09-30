"""EXP-2026-09-29-industry-policy (Phase 2; registered in logs/experiments.jsonl before any outcome).

Same engine, selector, wiring check and pass rule as test_industry_fundamentals.py, with the Phase 2 scores
(data/derived/industry_scores_policy.parquet, build_policy_scores.py):
  G1  trend & heat_pct >= 0.70 & NOT (P_pct < 0.30)   (veto hot industries with bottom-30% budget/activity; no data passes)
  G2  trend & F5_pct >= 0.70                           (A, B, C, E, G; no price heat)
  G3  trend & heat_pct >= 0.70 & F5_pct >= 0.50
PASS = beats S1M (both eras CAGR, maxDD <= 2pp worse) at phase 0 and in >= 4 of 5 phases. Q1 IC for E, G, P, F5.
Output: logs/leader_sleeve/industry_policy/{results.csv, ic.csv, README.md} + manifests; RESULT line in experiments.jsonl.
SGM_POLICY_H=1 (2026-09-30): the registered H_policy follow-up: the same arms with the PIB pillar H added to P and F5
(P_H, F5_H from build_policy_scores.py): G1H, G2H, G3H. Registered verdict vs S1M; also reported vs G1 (production),
because that is the question that matters now. Output industry_policy/H_policy/; RESULT id ...-H_policy-RESULT.
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

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-29-industry-policy"
SCORES = ROOT / "data/derived/industry_scores_policy.parquet"
OUTDIR = ROOT / "logs/leader_sleeve/industry_policy"
H_MODE = os.environ.get("SGM_POLICY_H") == "1"


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(SCORES); S["date"] = pd.to_datetime(S["date"])
    cal = D["cal"]
    grid = lambda o: [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp(tif.START)]  # noqa: E731
    ELIG = {"G1": S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)],
            "G2": S[S["F5_pct"] >= 0.70],
            "G3": S[(S["heat_pct"] >= 0.70) & (S["F5_pct"] >= 0.50)]}
    PROD = ELIG["G1"]
    if H_MODE:
        ELIG = {"G1H": S[(S["heat_pct"] >= 0.70) & ~(S["P_H_pct"] < 0.30)],
                "G2H": S[S["F5_H_pct"] >= 0.70],
                "G3H": S[(S["heat_pct"] >= 0.70) & (S["F5_H_pct"] >= 0.50)]}

    def run_phase(o: int) -> dict:
        wk = grid(o)
        ref = sre.select_s1(X["F"], wk, imap, "model", P)
        if o == 0:
            mine = tif.select_elig(X["F"], wk, imap, P, S[S["heat_pct"] >= 0.70])
            same = sum(ref.get(d, []) == mine.get(d, []) for d in wk)
            print(f"wiring check: custom selector reproduces S1M on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — no result reported")
        sels = {"S1M": ref, "S0": sp.select(D["px"], wk, "core", 3, imap)}
        if H_MODE:
            sels["G1"] = tif.select_elig(X["F"], wk, imap, P, PROD)
        sels.update({k: tif.select_elig(X["F"], wk, imap, P, e) for k, e in ELIG.items()})
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")},
                          avg_names=info["avg_names"], pct_touched50=info["pct_touched50"], years=m["years"])
        return res

    R0 = run_phase(0)
    ph = {k: [bool(sp.beats(R0[k], R0["S1M"]))] for k in ELIG}
    pg = {k: [bool(sp.beats(R0[k], R0["G1"]))] for k in ELIG} if H_MODE else {}
    for o in (1, 2, 3, 4):
        Ro = run_phase(o)
        for k in ELIG:
            ph[k].append(bool(sp.beats(Ro[k], Ro["S1M"])))
            if H_MODE:
                pg[k].append(bool(sp.beats(Ro[k], Ro["G1"])))
            R0[k].setdefault("phase_cagr", []).append(round(Ro[k]["cagr"], 1))
    for k in ELIG:
        R0[k]["phases_beaten"] = f"{sum(ph[k])}/5"; R0[k]["PASS"] = bool(ph[k][0] and sum(ph[k]) >= 4)
        if H_MODE:
            R0[k]["vs_G1_phases"] = f"{sum(pg[k])}/5"; R0[k]["beats_G1"] = bool(pg[k][0] and sum(pg[k]) >= 4)
    print(f"\n=== {EXP_ID} · weekly cohorts {tif.START}.. ===")
    print(f"{'arm':5s} {'CAGR':>6s} {'disc':>6s} {'conf':>6s} {'maxDD':>7s} {'Sharpe':>6s} {'names':>5s} {'P+50':>5s}  phases  PASS")
    for k in ["S0", "S1M", *(["G1"] if H_MODE else []), *ELIG]:
        r = R0[k]
        print(f"{k:5s} {r['cagr']:6.1f} {r['cagr_disc']:6.1f} {r['cagr_conf']:6.1f} {r['maxdd']:7.1f} {r['sharpe']:6.2f} "
              f"{r['avg_names']:5.1f} {r['pct_touched50']:5.1f}  {r.get('phases_beaten', ''):6s}  {r.get('PASS', '')}")

    I = tif.industry_daily(D, imap)
    cs = np.log1p(I.fillna(0)).cumsum(); valid = I.notna().astype(int).cumsum()
    fwd = np.expm1(cs.shift(-tif.FWD) - cs).where((valid.shift(-tif.FWD) - valid) >= int(0.75 * tif.FWD))
    fl = fwd.stack().rename("fwd").reset_index(); fl.columns = ["date", "industry", "fwd"]
    Q = S.merge(fl, on=["date", "industry"], how="inner")
    ics = []
    for d, g in Q.groupby("date"):
        row = dict(date=d)
        for c in (("E", "G", "H", "P_H", "F5_H") if H_MODE else ("E", "G", "P", "F5")):
            row[f"ic_{c}"] = tif.spearman(g[c], g["fwd"]); row[f"pic_{c}"] = tif.partial(g[c], g["fwd"], g["heat_pct"])
        ics.append(row)
    IC = pd.DataFrame(ics).set_index("date").sort_index(); samp = IC.iloc[::tif.FWD]
    summ = {}
    print("\nQ1 industry IC with the next 65-session industry return (non-overlapping; pic = partial on heat_pct):")
    for c in IC.columns:
        line = []
        for era, seg in (("disc", samp[samp.index < tif.ERA]), ("conf", samp[samp.index >= tif.ERA])):
            x = seg[c].dropna()
            t = float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan
            summ[f"{era}|{c}"] = dict(mean=float(x.mean()) if len(x) else np.nan, t=t, n=len(x))
            line.append(f"{era} {summ[f'{era}|{c}']['mean']:+.3f} (t {t:+.1f}, n {len(x)})")
        print(f"  {c:8s} " + " | ".join(line))

    repro = os.environ.get("SGM_REPRO") == "1"          # reproduction run (src/agentic/trust/reproduce.py): no ledger write
    out = OUTDIR / "repro" if repro else (OUTDIR / "H_policy" if H_MODE else OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **{c: v for c, v in r.items() if c != "years"}) for k, r in R0.items()]).to_csv(out / "results.csv", index=False)
    IC.to_csv(out / "ic.csv")
    now = datetime.now().isoformat(timespec="seconds")
    for f in ("results.csv", "ic.csv"):
        (out / (f + ".manifest.json")).write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_industry_policy.py",
            definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent", ic="Spearman correlation"), updated=now), indent=1))
    (out / "README.md").write_text(f"# {EXP_ID}\n\nPhase 2 industry-filter test (budget capex + IIP/core activity). See the manifests and "
                                      "src/agentic/test_industry_policy.py. The PIB policy pillar is run after its backfill.\n")
    passed = [k for k in ELIG if R0[k]["PASS"]]
    if repro:
        print(f"\nREPRO RUN (not logged) · VERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}")
        return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + ("-H_policy-RESULT" if H_MODE else "-RESULT"), verdict=passed or "no arm passes",
                                 vs_G1=({k: [R0[k]["vs_G1_phases"], R0[k]["beats_G1"]] for k in ELIG} if H_MODE else None),
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 ic={k: v for k, v in summ.items()}, cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats S1M by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
