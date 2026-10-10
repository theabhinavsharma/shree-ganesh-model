"""EXP-2026-10-10-model-minus-boom (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: the same V3 rule on two sets of model scores: the current model (trained on every year since 2016) and a model
that never learns from the COVID boom (training rows whose outcome window touches Mar 2020 - Dec 2021 removed). Which one
picks better?
  V3_OLD  v3_rule.picks on logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet (the live scores)
  V3_NB   v3_rule.picks on logs/leader_sleeve/anatomy_1p5x_noboom/bakeoff_preds.parquet
Window: weekly entries from 2019-01-01; eras 2019-22 and 2023+; phases 0-4. Engine sim_screen_rank_exit.run_exit("E0").
PASS = sp.beats(V3_NB, V3_OLD) at phase 0 and in >= 4 of 5. Wiring: V3_OLD == the live rule; 2019-2020 scores identical.
Diagnostics: per test year model skill from both bakeoff.json files; how many names the two lists share each year.
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
import v3_rule  # noqa: E402

EXP_ID = "EXP-2026-10-10-model-minus-boom"
OLD, NEW = ROOT / "logs/leader_sleeve/anatomy_1p5x", ROOT / "logs/leader_sleeve/anatomy_1p5x_noboom"
OUTDIR = ROOT / "logs/leader_sleeve/model_minus_boom"
START = pd.Timestamp("2019-01-01")
ARMS = ("V3_OLD", "V3_NB")


def scores(path: Path) -> pd.DataFrame:
    P = pd.read_parquet(path, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
    ctx = v3_rule.context(imap)
    PO, PN = scores(OLD / "bakeoff_preds.parquet"), scores(NEW / "bakeoff_preds.parquet")

    # ---- data-readiness gate
    R = pd.read_parquet(OLD / "rows.parquet", columns=["trade_date"]); R["trade_date"] = pd.to_datetime(R["trade_date"])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    end = cal[-1] - pd.Timedelta(days=10)
    checks = [dr.span("anatomy rows (model training data)", R["trade_date"], pd.Timestamp("2016-01-06"), end),
              dr.span("current model scores", PO["trade_date"], START, end),
              dr.span("no-boom model scores", PN["trade_date"], START, end),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2016, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring checks
    m = PO.merge(PN, on=["symbol", "trade_date"], suffixes=("_o", "_n"))
    early = m[m["trade_date"] < "2021-01-01"]
    diff = float((early["ensemble_o"] - early["ensemble_n"]).abs().max())
    print(f"wiring check 1: 2019-2020 scores identical in both models (max difference {diff:.2e} over {len(early):,} rows)", flush=True)
    if diff > 1e-9:
        raise SystemExit("wiring check FAILED: the exclusion changed pre-boom models — no result reported")
    later = m[m["trade_date"] >= "2021-01-01"]
    print("score correlation old vs new by year: " + " · ".join(f"{y} {g['ensemble_o'].corr(g['ensemble_n']):.3f}" for y, g in later.groupby(later["trade_date"].dt.year)))
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    sel_o = v3_rule.picks(X["F"], wk0, imap, PO, ctx)
    live = v3_rule.picks(X["F"], wk0, imap, sre.model_scores(), ctx)
    same = sum(sel_o[d] == live.get(d, []) for d in wk0)
    print(f"wiring check 2: V3_OLD equals the live rule on {same}/{len(wk0)} weeks", flush=True)
    if same < len(wk0):
        raise SystemExit("wiring check FAILED — no result reported")
    sel_n = v3_rule.picks(X["F"], wk0, imap, PN, ctx)
    ov = pd.Series({d: len(set(sel_o.get(d, [])) & set(sel_n.get(d, []))) / max(len(sel_o.get(d, [])), 1) for d in wk0})
    print("share of V3_OLD names also in V3_NB, by year: " + " · ".join(f"{y} {v*100:.0f}%" for y, v in ov.groupby(pd.DatetimeIndex(ov.index).year).mean().items()))

    # ---- model skill per test year (from each run's own bakeoff.json)
    skill = {}
    for lab, d in (("old", OLD), ("no-boom", NEW)):
        b = json.loads((d / "bakeoff.json").read_text())["per_year"]
        skill[lab] = {k.split("|")[1]: dict(auc=v["wk_auc"], top1=v["wk_top1"], base=v["wk_base"], lift=v["wk_top1_lift"]) for k, v in b.items() if k.startswith("ensemble|")}
    print("\nmodel skill by test year (ensemble): AUC old / no-boom · top-1% hit old / no-boom (base) · lift old / no-boom")
    for y in sorted(skill["old"]):
        o, n = skill["old"][y], skill["no-boom"].get(y, {})
        print(f"  {y}: AUC {o['auc']:.3f} / {n.get('auc', np.nan):.3f} · top-1% {o['top1']:.1f}% / {n.get('top1', np.nan):.1f}% (base {o['base']:.1f}%) · lift {o['lift']:.2f} / {n.get('lift', np.nan):.2f}")

    # ---- outcomes
    res, ph = {}, []
    for o in range(5):
        wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        so = sel_o if o == 0 else v3_rule.picks(X["F"], wk, imap, PO, ctx)
        sn = sel_n if o == 0 else v3_rule.picks(X["F"], wk, imap, PN, ctx)
        Rr = {}
        for a, sel in (("V3_OLD", so), ("V3_NB", sn)):
            nav, info = sre.run_exit(D, X, sel, wk, "E0")
            mm = sp.metrics(nav); Rr[a] = dict(**{c: mm[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"],
                                               years={int(k): round(float(v), 1) for k, v in (mm.get("years") or {}).items()} if isinstance(mm.get("years"), dict) else None)
        ph.append(bool(sp.beats(Rr["V3_NB"], Rr["V3_OLD"])))
        if o == 0:
            res = Rr
        print(f"phase {o}: V3_OLD {Rr['V3_OLD']['cagr_disc']:.1f}/{Rr['V3_OLD']['cagr_conf']:.1f}/{Rr['V3_OLD']['maxdd']:.1f} · V3_NB {Rr['V3_NB']['cagr_disc']:.1f}/{Rr['V3_NB']['cagr_conf']:.1f}/{Rr['V3_NB']['maxdd']:.1f} · beats {ph[-1]}", flush=True)
    passed = ph[0] and sum(ph) >= 4
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · phase 0 ===")
    for a in ARMS:
        r = res[a]
        print(f"{a:7s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}" + (f" · calendar years {r['years']}" if r["years"] else ""))
    print(f"V3_NB beats V3_OLD in {sum(ph)}/5 phases (phase 0 {'yes' if ph[0] else 'no'}) -> VERDICT: {'V3_NB PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: v for k, v in res[a].items() if k != "years"}, phases_beaten=(f"{sum(ph)}/5" if a == "V3_NB" else None), PASS=(passed if a == "V3_NB" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_noboom_model.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent"), skill=skill, list_overlap_by_year={int(y): round(float(v), 3) for y, v in ov.groupby(pd.DatetimeIndex(ov.index).year).mean().items()},
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nV3 on the current model's scores vs on a model trained without the COVID boom (Mar 2020 - Dec 2021). Verdict: "
                                      f"{'V3_NB PASS' if passed else 'no pass'}. Numbers in results.csv; model skill by year and list overlap in its manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3_NB"] if passed else "no arm passes",
                                 arms={a: [round(res[a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if a == "V3_NB" else [None]) for a in ARMS},
                                 skill=skill, cols="CAGR, 2019-22, 2023+, maxDD, phases"), default=str) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
