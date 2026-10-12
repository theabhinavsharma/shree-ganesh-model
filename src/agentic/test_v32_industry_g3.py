"""EXP-2026-10-12-v32-industry-g3 (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3.2 picks from industries that are hot on price (heat >= 0.70) unless their budget / activity data rank in the
bottom 30% (G1). Would it pick better from industries that are hot AND in the top half on industry fundamentals (G3: F5_pct >=
0.50), or the same with the government-press-release pillar (G3H: F5_H_pct >= 0.50)? Everything else is V3.2: Rs 1 cr bar,
new listings, model top 9, financial / fading filters, trade-for-trade dropped, no refill. Weekly entries from 2019, schedules
0-4, test_v3_l1_newlist_cost.run_cost (cash-flow corrected, liquidity-scaled costs), signal-day close (next open reported).
PASS per arm: CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and in >= 4 of 5.
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
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402

G3H_ONLY = os.environ.get("SGM_G3H_ONLY") == "1"      # EXP-2026-10-12-v32-industry-g3h: G3H alone (G3's input failed the data gate)
EXP_ID = "EXP-2026-10-12-v32-industry-g3h" if G3H_ONLY else "EXP-2026-10-12-v32-industry-g3"
OUTDIR = ROOT / ("logs/leader_sleeve/v32_industry_g3h" if G3H_ONLY else "logs/leader_sleeve/v32_industry_g3")
START, YEARS, HOLD = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026), 126
ARMS = ("V3.2", "V3.2_G3H") if G3H_ONLY else ("V3.2", "V3.2_G3", "V3.2_G3H")


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy()
    F = X["F"].merge(L[["symbol", "trade_date", "raw", "adv"]], on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    del L
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F["age"] = F.groupby("symbol").cumcount()
    F["young"] = F["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren) & F["age"].between(63, 251)
    hot = set(zip(pd.to_datetime(ctx["G1"]["date"]), ctx["G1"]["industry"]))

    def v31_ranked(weeks: list, topn: int = 20) -> dict:
        """{date: V3.1 pool top `topn` in model order, before the financial / fading filters} (frozen copy)."""
        W = F[F["trade_date"].isin(set(weeks)) & (F["adv"] >= 1.0) & (F["raw"] > 50)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W[[(d, i) in hot for d, i in zip(W["trade_date"], W["ind"])]]
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in weeks}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    Sx = ctx["scores"]; Sx = Sx.assign(date=pd.to_datetime(Sx["date"]))
    HOT = {"V3.2": hot, "V3.2_G3": set(zip(Sx.loc[(Sx["heat_pct"] >= 0.70) & (Sx["F5_pct"] >= 0.50), "date"], Sx.loc[(Sx["heat_pct"] >= 0.70) & (Sx["F5_pct"] >= 0.50), "industry"])),
           "V3.2_G3H": set(zip(Sx.loc[(Sx["heat_pct"] >= 0.70) & (Sx["F5_H_pct"] >= 0.50), "date"], Sx.loc[(Sx["heat_pct"] >= 0.70) & (Sx["F5_H_pct"] >= 0.50), "industry"]))}
    ranked = {}
    for a, hs in HOT.items():                              # v31_ranked reads `hot` when called: rebind it per arm
        if a not in ARMS:
            continue
        hot = hs
        ranked[a] = {o: v31_ranked(weeks[o], 9) for o in range(5)}
    hot = HOT["V3.2"]



    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731

    def arms(o: int) -> dict:
        return {a: {d: [s for s in ranked[a][o][d][:9] if ctx["keep"](s, d) and not t2t(s, d)] for d in weeks[o]} for a in ARMS}

    # ---- data gate (no outcomes yet): fundamentals scores known for hot industries
    hotrows = Sx[(Sx["heat_pct"] >= 0.70) & (Sx["date"] >= START)].assign(F5=lambda x: x["F5_pct"].where(x["F5_pct"].notna()), F5H=lambda x: x["F5_H_pct"])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("industry fundamentals score known for hot industries (G3: F5_pct, G3H: F5_H_pct)", hotrows, "date", ["F5_H_pct"] if G3H_ONLY else ["F5_pct", "F5_H_pct"], yrs),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    A0 = arms(0)
    F1 = v3_rule.frame_v31(X["F"], D); ref = v3_rule.picks_v31(F1, weeks[0], imap, P, ctx)
    same = sum(A0["V3.2"][d] == [s for s in ref[d] if not t2t(s, d)] for d in weeks[0])
    print(f"wiring check: V3.2 arm equals v3_rule V3.1 minus trade-for-trade on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")
    for a in ARMS[1:]:
        print(f"{a} differs from V3.2 on {sum(A0[a][d] != A0['V3.2'][d] for d in weeks[0])}/{len(weeks[0])} weeks · avg names {np.mean([len(v) for v in A0[a].values()]):.1f} vs {np.mean([len(v) for v in A0['V3.2'].values()]):.1f}", flush=True)

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_cost_open(D, X, picks, weekly, rule, costs_of):")
    fn = fn.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fn = fn.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_open = ns["run_cost_open"]

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    res, res_open, ph = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        Ao = A0 if o == 0 else arms(o)
        R = {a: met(run_cost(D, X, Ao[a], weeks[o], "E0", cost)[0]) for a in ARMS}
        for a in ARMS:
            res[a].append(R[a]); res_open[a].append(met(run_open(D, X, Ao[a], weeks[o], "E0", cost)[0]))
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3.2"])))
        print(f"schedule {o}: " + " · ".join(f"{a} {R[a]['cagr_disc']:.1f}/{R[a]['cagr_conf']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r, ro = res[a][0], res_open[a][0]
        print(f"{a:9s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS)
              + f" · next open {ro['cagr']:.1f}" + (f" · beats V3.2 in {sum(ph[a])}/5 (schedule 0 {'yes' if ph[a][0] else 'no'})" if a != "V3.2" else ""))
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       next_open_cagr=round(res_open[a][0]["cagr"], 1), phases_beaten=(f"{sum(ph[a])}/5" if a != "V3.2" else None), PASS=((a in verdict) if a != "V3.2" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_industry_g3.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nG3 / G3H industry eligibility on top of V3.2. Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no pass'}. Arms in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no pass",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph[a])}/5"] if a != "V3.2" else [None]) for a in ARMS},
                                 calendar_years=yrs_avg, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
