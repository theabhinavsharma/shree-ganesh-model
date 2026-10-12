"""EXP-2026-10-13-real-goal-model (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: can the rules live inside the model? Model NG learns what we actually earn (6-month return from the list-day close,
dividends and payouts counted, after costs, >= +50%) and sees V3.2's rule ingredients as inputs (trade-for-trade, financial sector,
fading theme, G1 industry, trend test, new listing). B6 keeps only the liquidity bar and buys NG's top 6.
Arms: V3.2 (live), B6 (PRIMARY: Rs 1 cr ADV + price > 50, top 6 by NG), B9 (top 9), V3.2_NG (V3.2's rules ranked by NG),
V3.2_C (V3.2's rules ranked by control model C: old target, same 2019 training start).
Models: logs/leader_sleeve/model_real_goal/model_NG, model_C (model_bakeoff_1p5x.py --train-from 2019-01-01 --years 2021..2026).
Window: weekly entries from 2021-01-01, schedules 0-4; eras 2021-22 / 2023+. Engine: test_v3_l1_newlist_cost.run_cost (dividends,
payouts as cash, liquidity-scaled costs), entry at the signal-day close; next open reported.
ADOPTION: B6 beats live V3.2 (CAGR higher in both eras, max drawdown not worse by > 2 points) at schedule 0 and >= 4 of 5.
SGM_REPRO=1: log nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
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

EXP_ID = "EXP-2026-10-13-real-goal-model"
OUTDIR = ROOT / "logs/leader_sleeve/model_real_goal"
START, YEARS = pd.Timestamp("2021-01-01"), (2023, 2024, 2025, 2026)
ARMS = ("V3.2", "B6", "B9", "V3.2_NG", "V3.2_C")
NEW = ["t2t", "fin_sector", "fading_only", "g1_elig", "trend_ok", "young_listing"]
PHASES = {"B6_vs_V3.2": ("B6", "V3.2"), "B9_vs_V3.2": ("B9", "V3.2"), "B6_vs_V3.2_C": ("B6", "V3.2_C"), "V3.2_NG_vs_V3.2_C": ("V3.2_NG", "V3.2_C")}


def scores(path: Path) -> pd.DataFrame:
    P = pd.read_parquet(path, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def wk_auc(y, s, wk) -> float:
    """Mean within-week AUC (Mann-Whitney, average ranks) over weeks with both classes."""
    D = pd.DataFrame(dict(y=y, s=s, w=wk)).dropna()
    out = []
    for _, g in D.groupby("w"):
        n1 = int(g["y"].sum()); n0 = len(g) - n1
        if n1 and n0:
            r = g["s"].rank()
            out.append((r[g["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else np.nan


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    PS = {"live": sre.model_scores(), "NG": scores(OUTDIR / "model_NG/bakeoff_preds.parquet"), "C": scores(OUTDIR / "model_C/bakeoff_preds.parquet")}
    for k, P in PS.items():
        print(f"{k}: scores {P['trade_date'].min().date()}..{P['trade_date'].max().date()} · {len(P):,} rows", flush=True)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731

    def v32(P, wk):
        return {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}

    liquid = F1[F1["core"].fillna(False).astype(bool)][["symbol", "trade_date"]]

    def top_liquid(P, wk, n):
        """Liquidity bar only (V3.1's Rs 1 cr ADV + price > 50), top n by the model's score that week."""
        W = liquid[liquid["trade_date"].isin(set(wk))].sort_values("trade_date").copy(); W["trade_date"] = W["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(W, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7)).dropna(subset=["ensemble"])
        m = m.sort_values(["trade_date", "ensemble", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = m.groupby("trade_date").head(n).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in wk}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}

    # ---- data gate (no outcomes yet)
    R = pd.read_parquet(OUTDIR / "rows.parquet", columns=["symbol", "trade_date", "core", "y126c50"] + NEW)
    R["trade_date"] = pd.to_datetime(R["trade_date"])
    done = R[R["trade_date"] <= cal[-127]]
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    bo = json.loads((OUTDIR / "model_NG/bakeoff.json").read_text())
    gaps = {Y: int(cal.get_loc(pd.Timestamp(f["first_test_session"])) - cal.get_loc(pd.Timestamp(f["train_last_date"]))) for Y, f in bo["folds"].items()}
    print(f"sessions from NG's last training row to each test year (target spans 126): {gaps}", flush=True)
    leak = sum(g < 127 for g in gaps.values())   # AMENDMENT 2026-10-13: session-counted target -> session gap (was the row-counted audit)
    checks = [dr.coverage("real target known where its 126 sessions have passed", done, "trade_date", ["y126c50"], range(2019, cal[-127].year + 1)),
              dr.coverage("rule inputs filled in the model rows from 2019", R[R["trade_date"] >= "2019-01-01"], "trade_date", NEW, range(2019, cal[-1].year + 1)),
              dr.qc("model NG: every training row's 126-session window ends before its test year", 1.0 if leak == 0 else 0.0, 1.0),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    checks += [dr.span(f"{k} model scores", P["trade_date"], START, cal[-1] - pd.Timedelta(days=10)) for k, P in PS.items()]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_cost_open(D, X, picks, weekly, rule, costs_of):")
    fn = fn.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fn = fn.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_open = ns["run_cost_open"]

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    # ---- wiring (exact, same data): live V3.2 from 2019-04-01 == the promoter-buyers pool code run now in repro mode
    w19 = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp("2019-04-01")]
    got = [round(met(run_cost(D, X, v32(PS["live"], w19), w19, "E0", cost)[0])[k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    rr = subprocess.run(["/usr/bin/python3", "src/agentic/test_v32_promoter_buyers.py"], cwd=ROOT, env=dict(os.environ, SGM_REPRO="1"), capture_output=True, text=True)
    if rr.returncode != 0:
        raise SystemExit("wiring reference run failed — no result reported")
    rv = pd.read_csv(ROOT / "logs/leader_sleeve/v32_promoter_buyers/repro/results.csv").set_index("arm").loc["V3.2"]
    want = [round(float(rv[k]), 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
    print(f"wiring check: live V3.2 from 2019-04-01 {got} vs promoter-buyers code on the same data {want}", flush=True)
    if got != want:
        raise SystemExit("wiring check FAILED — no result reported")

    res, res_open, ph, changed = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {k: [] for k in PHASES}, {}
    for o in range(5):
        wk = weeks[o]
        sel = {"V3.2": v32(PS["live"], wk), "B6": top_liquid(PS["NG"], wk, 6), "B9": top_liquid(PS["NG"], wk, 9),
               "V3.2_NG": v32(PS["NG"], wk), "V3.2_C": v32(PS["C"], wk)}
        if o == 0:
            changed = {a: int(sum(set(sel[a][d]) != set(sel["V3.2"][d]) for d in wk)) for a in ARMS[1:]}
            ov = np.mean([len(set(sel["B6"][d]) & set(sel["V3.2"][d])) / max(1, len(sel["B6"][d])) for d in wk])
            print(f"lists that differ from V3.2 (of {len(wk)}): {changed} · B6 names also in V3.2 that week: {ov:.0%} · avg names "
                  + " ".join(f"{a} {np.mean([len(v) for v in sel[a].values()]):.1f}" for a in ARMS), flush=True)
        Ro = {a: met(run_cost(D, X, sel[a], wk, "E0", cost)[0]) for a in ARMS}
        for a in ARMS:
            res[a].append(Ro[a]); res_open[a].append(met(run_open(D, X, sel[a], wk, "E0", cost)[0]))
        for k, (a, b) in PHASES.items():
            ph[k].append(bool(sp.beats(Ro[a], Ro[b])))
        print(f"schedule {o}: " + " · ".join(f"{a} {Ro[a]['cagr_disc']:.1f}/{Ro[a]['cagr_conf']:.1f}/{Ro[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    passed = ph["B6_vs_V3.2"][0] and sum(ph["B6_vs_V3.2"]) >= 4
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- how well each model ranks the REAL target, by test year (rows whose 126 sessions have passed)
    Rl = done[done["trade_date"] >= START][["symbol", "trade_date", "y126c50", "core"]]
    auc = {}
    for k, P in PS.items():
        m = Rl.merge(P, on=["symbol", "trade_date"], how="inner")
        auc[k] = {int(y): round(wk_auc(g["y126c50"], g["ensemble"], g["trade_date"]), 3) for y, g in m.groupby(m["trade_date"].dt.year)}
    print("within-week AUC on the real target (6-month >= +50% after costs): " + " · ".join(f"{k} {v}" for k, v in auc.items()), flush=True)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r, ro = res[a][0], res_open[a][0]
        print(f"{a:8s} CAGR {r['cagr']:5.1f} · 2021-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + f" · next open {ro['cagr']:.1f}")
    for k, v in ph.items():
        print(f"{k}: beats in {sum(v)}/5 (schedule 0 {'yes' if v[0] else 'no'})")
    print(f"ADOPTION (B6 vs live V3.2): {'PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    out = OUTDIR / "results"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       next_open_cagr=round(res_open[a][0]["cagr"], 1)) for a in ARMS]).to_csv(out / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_real_goal_model.py",
        definitions=__doc__, phases={k: f"{sum(v)}/5" for k, v in ph.items()}, lists_changed=changed, auc_real_target=auc,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"), updated=now), indent=1))
    (out / "README.md").write_text(f"# {EXP_ID}\n\nRules inside the model: model NG trained on the real 6-month goal with the rule ingredients as inputs; B6 = "
                                   f"liquidity bar + NG's top 6. Adoption (B6 vs live V3.2): {'PASS' if passed else 'no pass'}. Arms in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=(["B6"] if passed else "no pass"),
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 phases={k: f"{sum(v)}/5" for k, v in ph.items()},
                                 next_open={a: round(res_open[a][0]["cagr"], 1) for a in ARMS}, calendar_years=yrs_avg,
                                 auc_real_target=auc, lists_changed=changed, cols="CAGR, 2021-22, 2023+, maxDD")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
