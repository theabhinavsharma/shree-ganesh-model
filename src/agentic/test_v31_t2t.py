"""EXP-2026-10-12-v31-t2t (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: more and more V3.1 picks sit in the trade-for-trade segment (BE/BZ series: delivery only, usually under
exchange surveillance). Do they do worse, and does leaving them out help?
Arms: V3.1 (frozen copy of the selection) vs V3.1_NOT2T (BE/BZ picks dropped, not refilled) and V3.1_NOT2T_REFILL (dropped and
refilled with the next model-ranked names that pass the same filters and are not BE/BZ, up to 20 deep, same count).
Series = the stock's series on the list date (latest earlier one if it did not trade that day), from the price panel.
Engine: test_v3_l1_newlist_cost.run_cost (liquidity-scaled costs), weekly entries 2019+, schedules 0-4.
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

EXP_ID = "EXP-2026-10-12-v31-t2t"
OUTDIR = ROOT / "logs/leader_sleeve/v31_t2t"
START, YEARS, HOLD = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026), 126

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
    ranked = {o: v31_ranked(weeks[o]) for o in range(5)}
    lists = {o: {d: [s for s in ranked[o][d][:9] if ctx["keep"](s, d)] for d in weeks[o]} for o in range(5)}   # = V3.1

    # ---- series on each list date (latest known on or before it)
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    t2t_w = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill()
    tcol = {s: i for i, s in enumerate(t2t_w.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TW = t2t_w.to_numpy(); del t2t_w

    def is_t2t(s, d) -> float:
        c = tcol.get(s)
        return np.nan if c is None else TW[tpos[d], c]

    # ---- data gate (no outcomes yet)
    rows = [dict(d=d, s=s, known=(1.0 if np.isfinite(is_t2t(s, d)) else np.nan)) for o in range(5) for d in weeks[o] for s in ranked[o][d]]
    K = pd.DataFrame(rows)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("series known on the list date for every ranked candidate (all 5 schedules)", K, "d", ["known"], yrs, min_share=0.99),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    def arms(o: int) -> dict:
        out = {"V3.1": lists[o], "V3.1_NOT2T": {}, "V3.1_NOT2T_REFILL": {}}
        for d in weeks[o]:
            v = lists[o][d]
            out["V3.1_NOT2T"][d] = [s for s in v if is_t2t(s, d) != 1.0]
            out["V3.1_NOT2T_REFILL"][d] = [s for s in ranked[o][d] if ctx["keep"](s, d) and is_t2t(s, d) != 1.0][:len(v)]
        return out

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731

    def run(sel: dict, o: int) -> dict:
        nav, info = run_cost(D, X, sel, weeks[o], "E0", cost)
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "names": info["avg_names"], **{y: float(r.get(y, np.nan)) for y in YEARS}}

    ARMS = ("V3.1", "V3.1_NOT2T", "V3.1_NOT2T_REFILL")
    res, ph = {a: [] for a in ARMS}, {a: [] for a in ARMS[1:]}
    for o in range(5):
        A = arms(o)
        R = {a: run(A[a], o) for a in ARMS}
        if o == 0:
            want = tuple(round(x, 1) for x in __import__("reproduce").logged("EXP-2026-10-11-v3-l1-newlist-cost")["arms"]["V3_L1Nc"][:4]); got = tuple(round(R["V3.1"][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
            print(f"wiring check: V3.1 schedule 0 = {got}, registered V3_L1Nc = {want}", flush=True)
            if got != want:
                raise SystemExit("wiring check FAILED — no result reported")
            A0 = A
        for a in ARMS:
            res[a].append(R[a])
        for a in ARMS[1:]:
            ph[a].append(bool(sp.beats(R[a], R["V3.1"])))
        print(f"schedule {o}: " + " · ".join(f"{a} {R[a]['cagr_disc']:.1f}/{R[a]['cagr_conf']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
    verdict = [a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4]
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- pick level (schedule 0, full windows)
    PXo = rp.load_panel(["open", "high", "low", "close"]); O, Hh, Lw, C = (rp.wide(PXo, c, cal) for c in ("open", "high", "low", "close")); del PXo
    On, Ln, Cn = O.to_numpy(), Lw.to_numpy(), C.ffill(limit=300).to_numpy(); Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    Q = pd.DataFrame([dict(trade_date=d, symbol=s, t2t=is_t2t(s, d)) for d in weeks[0] for s in lists[0][d]])
    e = Q["trade_date"].map(pos).to_numpy() + 1; c = Q["symbol"].map(col).fillna(-1).astype(int).to_numpy()
    ok = (c >= 0) & (e + HOLD - 1 <= T - 1); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
    entry = np.where(ok, On[e2, c2], np.nan); good = ok & (entry > 0)
    Q["ret"] = np.where(good, Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1, np.nan)
    Q["hit"] = np.where(good, (Hf[e2, c2] / entry - 1 >= 0.5).astype(float), np.nan)
    e1 = np.minimum(e, T - 1); e1 = np.where(c >= 0, e1, 0)
    prevc = Cn[np.maximum(e1 - 1, 0), np.maximum(c, 0)]
    Q["locked_open"] = np.where((c >= 0) & (e < T), ((On[e1, np.maximum(c, 0)] >= 1.049 * prevc) & (Ln[e1, np.maximum(c, 0)] >= On[e1, np.maximum(c, 0)] - 1e-9)).astype(float), np.nan)
    Q["era"] = np.where(Q["trade_date"] < pd.Timestamp("2023-01-01"), "2019-22", "2023+")
    Q["seg"] = np.where(Q["t2t"] == 1.0, "trade-for-trade", "normal")
    agg = dict(picks=("ret", "size"), stocks=("symbol", "nunique"), avg6m=("ret", "mean"), med6m=("ret", "median"), hit=("hit", "mean"),
               lost30=("ret", lambda x: (x <= -0.3).mean()), locked_open=("locked_open", "mean"))
    by_era = Q[Q["ret"].notna()].groupby(["era", "seg"]).agg(**agg).round(3)
    by_year = Q[Q["ret"].notna() & (Q["trade_date"] >= "2023-01-01")].groupby([Q["trade_date"].dt.year, "seg"]).agg(**agg).round(3)
    share = Q.groupby(Q["trade_date"].dt.year)["t2t"].mean().round(2).to_dict()

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r = res[a][0]
        print(f"{a:18s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['names']:.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + (f" · beats V3.1 in {sum(ph[a])}/5 (schedule 0 {'yes' if ph[a][0] else 'no'})" if a != "V3.1" else ""))
    print(f"trade-for-trade share of V3.1 picks by year: {share}")
    print("picks by segment and era (full 6-month windows):\n" + by_era.to_string())
    print("2023+ by year:\n" + by_year.to_string())
    print(f"VERDICT: {', '.join(verdict) + ' PASS' if verdict else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":   # trust/reproduce.py reads this file
        (OUTDIR / "repro").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}) for a in ARMS]).to_csv(OUTDIR / "repro/results.csv", index=False)
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       phases_beaten=(f"{sum(ph[a])}/5" if a != "V3.1" else None), PASS=((a in verdict) if a != "V3.1" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_t2t.py", definitions=__doc__,
        t2t_share=share, by_era=by_era.reset_index().to_dict("records"), by_year=by_year.reset_index().to_dict("records"),
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules", avg6m="fraction", hit="share reaching +50% within 126 sessions",
                   locked_open="share of picks whose entry day opened >= 4.9% above the previous close and never traded below the open"), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nV3.1 with and without trade-for-trade (BE/BZ) picks. Verdict: {', '.join(verdict) + ' PASS' if verdict else 'no pass'}. Arms in results.csv; pick-level split in the manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no pass",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph[a])}/5"] if a != "V3.1" else [None]) for a in ARMS},
                                 calendar_years=yrs_avg, t2t_share=share, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
