"""EXP-2026-10-12-promoter-in-model (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: instead of a hand rule that moves promoter buyers up within the model's top 25, let the model itself see promoter
market buying and selling and decide how much they matter. Two models trained identically on rows from April 2019 on (when the
promoter data starts): A = today's 50 inputs, B = the same + prom_buys90 + prom_sells90. Run V3.2 on each model's scores.
Arms: V3.2 (live model, trained from 2016; reference), V3.2_A, V3.2_B. Selection: v3_rule.picks_v31 minus trade-for-trade on the
list date, no refill. Window: weekly entries from 2021-01-01, schedules 0-4; eras 2021-22 / 2023+ (AMENDMENT before any outcome: a
model trained from 2019-04 cannot score 2020, the MLP has no fit block). Engine:
test_v3_l1_newlist_cost.run_cost (dividends counted, big payouts / demergers as cash, liquidity-scaled costs), entry at the
signal-day close; the next-open variant is reported too.
PROMOTER EFFECT (the verdict): V3.2_B beats V3.2_A (CAGR higher in both eras, max drawdown not worse by > 2 points) at schedule 0
and in >= 4 of 5. ADOPTION AS-IS: V3.2_B also beats live V3.2 by the same rule.
Models: logs/leader_sleeve/promoter_in_model/model_A, model_B (model_bakeoff_1p5x.py --entry close --train-from 2019-04-01).
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
import eval_checks  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402

EXP_ID = "EXP-2026-10-12-promoter-in-model"
OUTDIR = ROOT / "logs/leader_sleeve/promoter_in_model"
START, TRAIN_FROM, YEARS = pd.Timestamp("2021-01-01"), pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026)
PROM = ["prom_buys90", "prom_sells90"]
ARMS = ("V3.2", "V3.2_A", "V3.2_B")


def scores(path: Path) -> pd.DataFrame:
    P = pd.read_parquet(path, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    PS = {"V3.2": sre.model_scores(), "V3.2_A": scores(OUTDIR / "model_A/bakeoff_preds.parquet"), "V3.2_B": scores(OUTDIR / "model_B/bakeoff_preds.parquet")}
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
    R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["trade_date", "core"] + PROM)
    R["trade_date"] = pd.to_datetime(R["trade_date"]); R = R[R["core"].astype(bool) & (R["trade_date"] >= TRAIN_FROM)]
    em = eval_checks.insider_no_empty_month({}); nm = len(pd.period_range("2019-01", pd.Timestamp.now().to_period("M") - 1, freq="M"))
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.coverage("promoter buying / selling filled in the model rows (liquid stocks) from 2019-04", R, "trade_date", PROM, range(2019, cal[-1].year + 1)),
              dr.qc("insider history: months 2019-01..last month with disclosures (share)", 1.0 - float(em["value"]) / nm, 1.0),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    checks += [dr.span(f"{a} model scores", P["trade_date"], START, cal[-1] - pd.Timedelta(days=10)) for a, P in PS.items()]
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

    # ---- wiring: live V3.2 from 2019-04-01 (schedule 0) reproduces the logged V3.2 of EXP-2026-10-12-v32-promoter-buyers
    w0 = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp("2019-04-01")]
    got = [round(met(run_cost(D, X, v32(PS["V3.2"], w0), w0, "E0", cost)[0])[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    ref = __import__("reproduce").logged("EXP-2026-10-12-v32-promoter-buyers")["arms"]["V3.2"][:4]
    print(f"wiring check: live V3.2 from 2019-04-01 {got} vs logged {ref}", flush=True)
    if got != ref:
        raise SystemExit("wiring check FAILED — no result reported")

    res, res_open = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    ph = {"B_vs_A": [], "B_vs_live": [], "A_vs_live": []}
    changed = {}
    for o in range(5):
        sel = {a: v32(P, weeks[o]) for a, P in PS.items()}
        if o == 0:
            changed = {a: sum(sel[a][d] != sel["V3.2"][d] for d in weeks[o]) for a in ("V3.2_A", "V3.2_B")}
            changed["B_vs_A"] = sum(sel["V3.2_B"][d] != sel["V3.2_A"][d] for d in weeks[o])
            nB = sum(len(set(sel["V3.2_B"][d]) - set(sel["V3.2_A"][d])) for d in weeks[o])
            print(f"lists that differ (of {len(weeks[o])}): A vs live {changed['V3.2_A']} · B vs live {changed['V3.2_B']} · B vs A {changed['B_vs_A']} ({nB} names) · "
                  f"avg names " + " ".join(f"{a} {np.mean([len(v) for v in sel[a].values()]):.1f}" for a in ARMS), flush=True)
        Ro = {a: met(run_cost(D, X, sel[a], weeks[o], "E0", cost)[0]) for a in ARMS}
        for a in ARMS:
            res[a].append(Ro[a]); res_open[a].append(met(run_open(D, X, sel[a], weeks[o], "E0", cost)[0]))
        ph["B_vs_A"].append(bool(sp.beats(Ro["V3.2_B"], Ro["V3.2_A"]))); ph["B_vs_live"].append(bool(sp.beats(Ro["V3.2_B"], Ro["V3.2"])))
        ph["A_vs_live"].append(bool(sp.beats(Ro["V3.2_A"], Ro["V3.2"])))
        print(f"schedule {o}: " + " · ".join(f"{a} {Ro[a]['cagr_disc']:.1f}/{Ro[a]['cagr_conf']:.1f}/{Ro[a]['maxdd']:.1f}" for a in ARMS)
              + f" · B beats A {ph['B_vs_A'][-1]}", flush=True)
    passed = ph["B_vs_A"][0] and sum(ph["B_vs_A"]) >= 4
    adopt = passed and ph["B_vs_live"][0] and sum(ph["B_vs_live"]) >= 4
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}
    auc = {}
    for a, m in (("V3.2_A", "model_A"), ("V3.2_B", "model_B")):
        b = json.loads((OUTDIR / m / "bakeoff.json").read_text())
        auc[a] = {k.split("|")[1]: round(float(v.get("wk_auc", np.nan)), 3) for k, v in b.get("per_year", {}).items()
                  if k.startswith("ensemble|") and isinstance(v, dict)}
        print(f"{a} model weekly AUC by test year: {auc[a]}", flush=True)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r, ro = res[a][0], res_open[a][0]
        print(f"{a:7s} CAGR {r['cagr']:5.1f} · 2021-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + f" · next open {ro['cagr']:.1f} ({ro['cagr_disc']:.1f}/{ro['cagr_conf']:.1f}/{ro['maxdd']:.1f})")
    for k, v in ph.items():
        print(f"{k}: beats in {sum(v)}/5 (schedule 0 {'yes' if v[0] else 'no'})")
    print(f"PROMOTER EFFECT (B vs A): {'PASS' if passed else 'no pass'} · ADOPTION AS-IS (B also beats live V3.2): {'PASS' if adopt else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        (OUTDIR / "repro").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}) for a in ARMS]).to_csv(OUTDIR / "repro/results.csv", index=False)
        print("REPRO RUN (not logged)"); return
    out = OUTDIR / (f"rerun_{tag}" if tag else "results"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       next_open_cagr=round(res_open[a][0]["cagr"], 1)) for a in ARMS]).to_csv(out / "results.csv", index=False)
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_promoter_in_model.py",
        definitions=__doc__, phases={k: f"{sum(v)}/5" for k, v in ph.items()}, lists_changed=changed, model_weekly_auc=auc,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (out / "README.md").write_text(f"# {EXP_ID}\n\nPromoter buying / selling as model inputs (model B) vs the same model without them (model A), both "
                                   f"trained from April 2019, each run through V3.2. Promoter effect: {'PASS' if passed else 'no pass'}; adoption as-is: "
                                   f"{'PASS' if adopt else 'no pass'}. Arms in results.csv; models in ../model_A, ../model_B.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=(["V3.2_B"] if passed else "no pass"), adoption_as_is=("PASS" if adopt else "no pass"),
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 phases={k: f"{sum(v)}/5" for k, v in ph.items()},
                                 next_open={a: [round(res_open[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 calendar_years=yrs_avg, model_weekly_auc=auc, lists_changed=changed, cols="CAGR, 2021-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
