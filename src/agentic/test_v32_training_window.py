"""EXP-2026-10-12-v32-training-window (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: does the model pick better for V3.2 if it learns only from recent years, or from every year except the COVID boom?
Arms (each = V3.2 on that model's scores: v3_rule.picks_v31 minus trade-for-trade on the list date, no refill):
  V3.2       live model (50 features, all rows from 2016)
  V3.2_R3    model retrained per test year Y on rows dated Y-3-01-01 onward only (rolling 3 years)
  V3.2_NC    model retrained on 2016+ minus every row whose 95-session outcome window touches 2020-2021
  V3.2_SEED  noise floor: live recipe with every random seed shifted by 1 (diagnostic, not a candidate)
Window: weekly entries from 2019-04-01, schedules 0-4; eras 2019-22 / 2023+. Engine: test_v3_l1_newlist_cost.run_cost
(dividends counted, big payouts / demergers as cash, liquidity-scaled costs), entry at the signal-day close; next open reported.
PASS per arm (R3, NC): beats live V3.2 (CAGR higher in both eras, max drawdown not worse by > 2 points) at schedule 0 and in >= 4
of 5. If the schedule-0 lead in either era is not bigger than SEED's gap from live, it reads 'PASS within retrain noise'.
Models: logs/leader_sleeve/v32_training_window/model_R3, model_NC, model_SEED. SGM_REPRO=1: log nothing. SGM_RERUN_TAG=<tag>.
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
from test_v3_l1_newlist_cost import run_cost  # noqa: E402

EXP_ID = "EXP-2026-10-12-v32-training-window"
OUTDIR = ROOT / "logs/leader_sleeve/v32_training_window"
START, YEARS = pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026)
ARMS = ("V3.2", "V3.2_R3", "V3.2_NC", "V3.2_SEED")
MODELS = {"V3.2_R3": "model_R3", "V3.2_NC": "model_NC", "V3.2_SEED": "model_SEED"}


def scores(path: Path) -> pd.DataFrame:
    P = pd.read_parquet(path, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    PS = {"V3.2": sre.model_scores(), **{a: scores(OUTDIR / m / "bakeoff_preds.parquet") for a, m in MODELS.items()}}
    for a, P in PS.items():
        print(f"{a}: scores {P['trade_date'].min().date()}..{P['trade_date'].max().date()} · {len(P):,} rows", flush=True)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L

    # ---- trade-for-trade on each list date (latest series on or before it), frozen copy of test_v32_promoter_buyers
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731

    def v32(P, wk):
        return {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}

    # ---- data gate (no outcomes yet)
    R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["trade_date"])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.span("anatomy rows (model training data)", R["trade_date"], pd.Timestamp("2016-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2016, cal[-1].year + 1))]
    checks += [dr.span(f"{a} model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)) for a, P in PS.items()]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")
    del R, Px

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_cost_open(D, X, picks, weekly, rule, costs_of):")
    fn = fn.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fn = fn.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_open = ns["run_cost_open"]

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    res, res_open = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    ph = {a: [] for a in ARMS[1:]}
    changed = {}
    # wiring (exact, same data): V3.2 from the separately written pool code of test_v32_promoter_buyers.py, run now in repro mode.
    # (Comparing with its logged line drifts whenever the panel gains a session: Oct 12 was added after it ran.)
    import subprocess
    rr = subprocess.run(["/usr/bin/python3", "src/agentic/test_v32_promoter_buyers.py"], cwd=ROOT, env=dict(os.environ, SGM_REPRO="1"), capture_output=True, text=True)
    if rr.returncode != 0:
        raise SystemExit("wiring reference run failed — no result reported")
    rv = pd.read_csv(ROOT / "logs/leader_sleeve/v32_promoter_buyers/repro/results.csv").set_index("arm").loc["V3.2"]
    ref = [round(float(rv[k]), 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    logged = __import__("reproduce").logged("EXP-2026-10-12-v32-promoter-buyers")["arms"]["V3.2"][:4]
    for o in range(5):
        sel = {a: v32(P, weeks[o]) for a, P in PS.items()}
        Ro = {"V3.2": met(run_cost(D, X, sel["V3.2"], weeks[o], "E0", cost)[0])}
        if o == 0:
            got = [round(Ro["V3.2"][k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
            print(f"wiring check: live V3.2 {got} vs promoter-buyers code on the same data {ref} (its logged line, older data: {logged})", flush=True)
            if got != ref:
                raise SystemExit("wiring check FAILED — no result reported")
            changed = {a: sum(sel[a][d] != sel["V3.2"][d] for d in weeks[o]) for a in ARMS[1:]}
            print(f"lists that differ from live (of {len(weeks[o])}): {changed} · avg names " + " ".join(f"{a} {np.mean([len(v) for v in sel[a].values()]):.1f}" for a in ARMS), flush=True)
        Ro.update({a: met(run_cost(D, X, sel[a], weeks[o], "E0", cost)[0]) for a in ARMS[1:]})
        for a in ARMS:
            res[a].append(Ro[a]); res_open[a].append(met(run_open(D, X, sel[a], weeks[o], "E0", cost)[0]))
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(Ro[a], Ro["V3.2"])))
        print(f"schedule {o}: " + " · ".join(f"{a} {Ro[a]['cagr_disc']:.1f}/{Ro[a]['cagr_conf']:.1f}/{Ro[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    r0, s0 = res["V3.2"][0], res["V3.2_SEED"][0]
    noise = {e: abs(s0[e] - r0[e]) for e in ("cagr_disc", "cagr_conf")}
    verdict = {}
    for a in ("V3.2_R3", "V3.2_NC"):
        p = ph[a][0] and sum(ph[a]) >= 4
        within = p and any(res[a][0][e] - r0[e] <= noise[e] for e in noise)
        verdict[a] = "PASS within retrain noise" if within else ("PASS" if p else "no pass")
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}
    auc = {}
    for a, m in MODELS.items():
        b = json.loads((OUTDIR / m / "bakeoff.json").read_text())
        auc[a] = {k.split("|")[1]: round(float(v.get("wk_auc", np.nan)), 3) for k, v in b.get("per_year", {}).items()
                  if k.startswith("ensemble|") and isinstance(v, dict)}
        print(f"{a} model weekly AUC by test year: {auc[a]}", flush=True)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r, ro = res[a][0], res_open[a][0]
        print(f"{a:7s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + f" · next open {ro['cagr']:.1f} ({ro['cagr_disc']:.1f}/{ro['cagr_conf']:.1f}/{ro['maxdd']:.1f})")
    for k, v in ph.items():
        print(f"{k} vs live V3.2: beats in {sum(v)}/5 (schedule 0 {'yes' if v[0] else 'no'})")
    print(f"retrain noise (SEED vs live, schedule 0): 2019-22 {noise['cagr_disc']:.1f} pts · 2023+ {noise['cagr_conf']:.1f} pts")
    print("VERDICT: " + " · ".join(f"{a} {v}" for a, v in verdict.items()))
    if os.environ.get("SGM_REPRO") == "1":
        (OUTDIR / "repro").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}) for a in ARMS]).to_csv(OUTDIR / "repro/results.csv", index=False)
        print("REPRO RUN (not logged)"); return
    out = OUTDIR / (f"rerun_{tag}" if tag else "results"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       next_open_cagr=round(res_open[a][0]["cagr"], 1)) for a in ARMS]).to_csv(out / "results.csv", index=False)
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_promoter_in_model.py",
        definitions=__doc__, phases={k: f"{sum(v)}/5" for k, v in ph.items()}, verdict=verdict, retrain_noise=noise, lists_changed=changed, model_weekly_auc=auc,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (out / "README.md").write_text(f"# {EXP_ID}\n\nV3.2 on models trained on a rolling 3-year window (R3) or on 2016+ minus everything touching "
                                   f"2020-2021 (NC), vs the live model; SEED = live recipe with shifted seeds (retrain noise). Verdict: {verdict}. "
                                   f"Arms in results.csv; models in ../model_R3, ../model_NC, ../model_SEED.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=verdict, retrain_noise={k: round(v, 1) for k, v in noise.items()},
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 phases={k: f"{sum(v)}/5" for k, v in ph.items()},
                                 next_open={a: [round(res_open[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 calendar_years=yrs_avg, model_weekly_auc=auc, lists_changed=changed, cols="CAGR, 2019-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
