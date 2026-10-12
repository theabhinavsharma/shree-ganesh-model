"""EXP-2026-10-12-v32-promoter-buyers (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: on top of V3.2, move names whose promoters bought shares in the market in the last 90 days to the front of the
candidate pool (only within the model's top 25), then pick as V3.2 does. Does that beat V3.2?
Arms: V3.2 (pool by model rank -> top 9 -> financial / fading filters -> trade-for-trade dropped, no refill) and V3.2_PB (names
with prom_buys90 > 0 and pool rank <= 25 first, in model order; then the same steps). prom_buys90: anatomy rows as of the list
date, at most 7 days old. Window: weekly entries from 2019-04-01, schedules 0-4. Engine: test_v3_l1_newlist_cost.run_cost
(dividends counted, big payouts / demergers as cash, liquidity-scaled costs), entry at the signal-day close; the next-open
variant is reported too. PASS: CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and >= 4 of 5.
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

EXP_ID = "EXP-2026-10-12-v32-promoter-buyers"
OUTDIR = ROOT / "logs/leader_sleeve/v32_promoter_buyers"
START, YEARS, HOLD, BOOST_RANK = pd.Timestamp("2019-04-01"), (2023, 2024, 2025, 2026), 126, 25


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
    ranked = {o: v31_ranked(weeks[o], 25) for o in range(5)}
    lists = {o: {d: [s for s in ranked[o][d][:9] if ctx["keep"](s, d)] for d in weeks[o]} for o in range(5)}   # = V3.1


    # ---- trade-for-trade on each list date (latest series on or before it) and promoter buying
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"])
    A["trade_date"] = pd.to_datetime(A["trade_date"])
    PBd = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}

    def pb(s, d):
        a = PBd.get(s)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
        return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan

    def arms(o: int) -> dict:
        out = {"V3.2": {}, "V3.2_PB": {}}
        for d in weeks[o]:
            pool = ranked[o][d]
            first = [s for s in pool[:BOOST_RANK] if pb(s, d) > 0]
            for a, pl in (("V3.2", pool), ("V3.2_PB", first + [s for s in pool if s not in set(first)])):
                out[a][d] = [s for s in pl[:9] if ctx["keep"](s, d) and not t2t(s, d)]
        return out

    # ---- data gate (no outcomes yet)
    K = pd.DataFrame([dict(d=d, known=(1.0 if pd.notna(pb(s, d)) else np.nan)) for d in weeks[0] for s in ranked[0][d][:BOOST_RANK]])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("promoter-buying signal known for the pool's top-25 names on list dates", K, "d", ["known"], yrs),
              dr.span("anatomy rows (promoter buying)", A["trade_date"], START, cal[-1] - pd.Timedelta(days=10)),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring: the V3.2 arm equals v3_rule.picks_v31 minus trade-for-trade
    A0 = arms(0)
    F1 = v3_rule.frame_v31(X["F"], D); ref = v3_rule.picks_v31(F1, weeks[0], imap, P, ctx)
    same = sum(A0["V3.2"][d] == [s for s in ref[d] if not t2t(s, d)] for d in weeks[0])
    print(f"wiring check: V3.2 arm equals v3_rule V3.1 minus trade-for-trade on {same}/{len(weeks[0])} weeks", flush=True)
    if same < len(weeks[0]):
        raise SystemExit("wiring check FAILED — no result reported")
    moved = sum(len(set(A0["V3.2_PB"][d]) - set(A0["V3.2"][d])) for d in weeks[0]); wk_changed = sum(A0["V3.2_PB"][d] != A0["V3.2"][d] for d in weeks[0])
    print(f"V3.2_PB changes the list on {wk_changed}/{len(weeks[0])} weeks ({moved} names moved in)", flush=True)

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("def run_cost(D, X, picks, weekly, rule, costs_of):", "def run_cost_open(D, X, picks, weekly, rule, costs_of):")
    fn = fn.replace("                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])", "                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])\n                seg[0] = np.nan_to_num(D[\"first_fac\"][i0, cols] - 1.0)")
    fn = fn.replace("            names = [n for n in picks.get(d, []) if n in cidx]", "            names = [n for n in picks.get(d, []) if n in cidx and not D[\"locked\"][i0, cidx[n]] and np.isfinite(D[\"first_fac\"][i0, cidx[n]])]")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_open = ns["run_cost_open"]

    def met(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}
    ARMS = ("V3.2", "V3.2_PB")
    res, ph, res_open = {a: [] for a in ARMS}, [], {a: [] for a in ARMS}
    for o in range(5):
        Ao = A0 if o == 0 else arms(o)
        R = {a: met(run_cost(D, X, Ao[a], weeks[o], "E0", cost)[0]) for a in ARMS}
        for a in ARMS:
            res[a].append(R[a]); res_open[a].append(met(run_open(D, X, Ao[a], weeks[o], "E0", cost)[0]))
        ph.append(bool(sp.beats(R["V3.2_PB"], R["V3.2"])))
        print(f"schedule {o}: V3.2 {R['V3.2']['cagr_disc']:.1f}/{R['V3.2']['cagr_conf']:.1f}/{R['V3.2']['maxdd']:.1f} · V3.2_PB {R['V3.2_PB']['cagr_disc']:.1f}/{R['V3.2_PB']['cagr_conf']:.1f}/{R['V3.2_PB']['maxdd']:.1f} · beats {ph[-1]}", flush=True)
    passed = ph[0] and sum(ph) >= 4
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r, ro = res[a][0], res_open[a][0]
        print(f"{a:8s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + f" · bought at the next open: {ro['cagr']:.1f} ({ro['cagr_disc']:.1f}/{ro['cagr_conf']:.1f}/{ro['maxdd']:.1f})"
              + (f" · beats V3.2 in {sum(ph)}/5 (schedule 0 {'yes' if ph[0] else 'no'})" if a != "V3.2" else ""))
    print(f"VERDICT: {'V3.2_PB PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        (OUTDIR / "repro").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}) for a in ARMS]).to_csv(OUTDIR / "repro/results.csv", index=False)
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       next_open_cagr=round(res_open[a][0]["cagr"], 1), phases_beaten=(f"{sum(ph)}/5" if a != "V3.2" else None), PASS=(passed if a != "V3.2" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v32_promoter_buyers.py", definitions=__doc__,
        weeks_changed=int(wk_changed), names_moved_in=int(moved), units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nPromoter buyers first on top of V3.2. Verdict: {'V3.2_PB PASS' if passed else 'no pass'}. Arms in results.csv.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3.2_PB"] if passed else "no pass",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if a != "V3.2" else [None]) for a in ARMS},
                                 next_open={a: [round(res_open[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] for a in ARMS},
                                 calendar_years=yrs_avg, weeks_changed=int(wk_changed), names_moved_in=int(moved), cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
