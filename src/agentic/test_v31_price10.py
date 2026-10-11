"""EXP-2026-10-11-v31-price10 (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: V3.1 only buys stocks whose actual traded price is above Rs 50 (a floor carried over from the older
models' "core band", never tested on its own). Does a Rs 10 floor do better?
Arms: V3.1 (Rs 50, frozen copy of the live selection) vs V3.1_P10 (Rs 10); V3.1_P10_stress = Rs 10 with the round-trip
cost doubled on positions priced Rs 10-50 at entry (reported, not the pass arm).
Engine: test_v3_l1_newlist_cost.run_cost (liquidity-scaled costs), weekly entries 2019+, schedules 0-4.
PASS: CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and in >= 4 of 5.
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

EXP_ID = "EXP-2026-10-11-v31-price10"
OUTDIR = ROOT / "logs/leader_sleeve/v31_price10"
START, YEARS, HOLD = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026), 126

def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy()
    F = X["F"].merge(L[["symbol", "trade_date", "raw", "adv"]], on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    RAWW = rp.wide(L, "raw", cal).reindex(columns=D["syms"]).to_numpy()
    del L
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F["age"] = F.groupby("symbol").cumcount()
    F["young"] = F["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren) & F["age"].between(63, 251)
    hot = set(zip(pd.to_datetime(ctx["G1"]["date"]), ctx["G1"]["industry"]))

    def v31_lists(weeks: list, floor: float = 50.0) -> dict:
        """{date: V3.1 picks in model order} (frozen copy; price floor as a parameter)."""
        W = F[F["trade_date"].isin(set(weeks)) & (F["adv"] >= 1.0) & (F["raw"] > floor)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W[[(d, i) in hot for d, i in zip(W["trade_date"], W["ind"])]]
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: [s for s in g.get(d, []) if ctx["keep"](s, d)] for d in weeks}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    lists = {o: v31_lists(weeks[o]) for o in range(5)}
    lists10 = {o: v31_lists(weeks[o], 10.0) for o in range(5)}

    # ---- data gate (no outcomes yet): model score known for the newly eligible Rs 10-50 stocks
    wk_all = sorted(set(d for o in range(5) for d in weeks[o]))
    new = F[F["trade_date"].isin(set(wk_all)) & (F["adv"] >= 1.0) & (F["raw"] > 10) & (F["raw"] <= 50)][["symbol", "trade_date"]].sort_values("trade_date")
    mm = pd.merge_asof(new.astype({"trade_date": "datetime64[ns]"}), P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    new["scored"] = np.where(mm["ensemble"].notna().to_numpy(), 1.0, np.nan)
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("model score known for Rs 10-50 / Rs 1 cr stocks on list dates", new, "trade_date", ["scored"], yrs),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    base = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    stress = lambda i0, cols: base(i0, cols) * np.where(RAWW[i0, cols] <= 50, 2.0, 1.0)  # noqa: E731

    def run(sel: dict, o: int, cost) -> dict:
        nav, info = run_cost(D, X, sel, weeks[o], "E0", cost)
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "names": info["avg_names"], **{y: float(r.get(y, np.nan)) for y in YEARS}}

    ARMS = ("V3.1", "V3.1_P10", "V3.1_P10_stress")
    res, ph = {a: [] for a in ARMS}, []
    for o in range(5):
        a = run(lists[o], o, base)
        if o == 0:
            want = (51.8, 48.9, 54.9, -40.0); got = tuple(round(a[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
            print(f"wiring check: V3.1 schedule 0 = {got}, registered V3_L1Nc = {want}", flush=True)
            if got != want:
                raise SystemExit("wiring check FAILED — no result reported")
        b = run(lists10[o], o, base); c = run(lists10[o], o, stress)
        res["V3.1"].append(a); res["V3.1_P10"].append(b); res["V3.1_P10_stress"].append(c)
        ph.append(bool(sp.beats(b, a)))
        print(f"schedule {o}: V3.1 {a['cagr_disc']:.1f}/{a['cagr_conf']:.1f}/{a['maxdd']:.1f} · P10 {b['cagr_disc']:.1f}/{b['cagr_conf']:.1f}/{b['maxdd']:.1f} · beats {ph[-1]}", flush=True)
    passed = ph[0] and sum(ph) >= 4
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in ARMS}

    # ---- the Rs 10-50 picks (schedule 0): how many, how they did, trade-for-trade share
    PXo = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PXo, c, cal) for c in ("open", "high", "close")); del PXo
    On, Cn = O.to_numpy(), C.ffill(limit=300).to_numpy(); Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    rawmap = F.set_index(["symbol", "trade_date"])["raw"]
    rows = []
    for d in weeks[0]:
        old = set(lists[0][d])
        for s in lists10[0][d]:
            rows.append(dict(trade_date=d, symbol=s, low=(s not in old)))
    Q = pd.DataFrame(rows)
    Q["raw"] = [rawmap.get((s, d), np.nan) for s, d in zip(Q["symbol"], Q["trade_date"])]
    Q = Q.merge(S, on=["symbol", "trade_date"], how="left")
    e = Q["trade_date"].map(pos).to_numpy() + 1; c = Q["symbol"].map(col).fillna(-1).astype(int).to_numpy()
    ok = (c >= 0) & (e + HOLD - 1 <= T - 1); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
    entry = np.where(ok, On[e2, c2], np.nan); good = ok & (entry > 0)
    Q["ret"] = np.where(good, Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1, np.nan)
    Q["hit"] = np.where(good, (Hf[e2, c2] / entry - 1 >= 0.5).astype(float), np.nan)
    Q["era"] = np.where(Q["trade_date"] < pd.Timestamp("2023-01-01"), "2019-22", "2023+")
    Q["band"] = np.where(Q["raw"] <= 50, "Rs 10-50", "above Rs 50")
    share = Q.groupby(Q["trade_date"].dt.year)["band"].apply(lambda x: round(float((x == "Rs 10-50").mean()), 2)).to_dict()
    diag = Q[Q["ret"].notna()].groupby(["era", "band"]).agg(picks=("ret", "size"), stocks=("symbol", "nunique"), avg6m=("ret", "mean"), med6m=("ret", "median"),
                                                          hit=("hit", "mean"), lost30=("ret", lambda x: (x <= -0.3).mean()), t2t=("series", lambda x: (x != "EQ").mean())).round(3)
    pushed = Q.groupby("trade_date")["low"].sum().mean()

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in ARMS:
        r = res[a][0]
        print(f"{a:16s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['names']:.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + (f" · beats V3.1 in {sum(ph)}/5 (schedule 0 {'yes' if ph[0] else 'no'})" if a == "V3.1_P10" else ""))
    print(f"share of P10 picks priced Rs 10-50, by year: {share} · names swapped in per weekly list: {pushed:.1f}")
    print("P10 picks by price band (schedule 0, full 6-month windows):\n" + diag.to_string())
    print(f"VERDICT: {'V3.1_P10 PASS' if passed else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       phases_beaten=(f"{sum(ph)}/5" if a == "V3.1_P10" else None), PASS=(passed if a == "V3.1_P10" else None)) for a in ARMS]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_price10.py", definitions=__doc__,
        low_price_share=share, by_band=diag.reset_index().to_dict("records"), units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules",
        avg6m="fraction", hit="share reaching +50% within 126 sessions", t2t="share of picks in the BE/BZ series on the list date"), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nV3.1 with a Rs 10 price floor instead of Rs 50. Verdict: {'V3.1_P10 PASS' if passed else 'no pass'}. Arms in results.csv; the Rs 10-50 picks in the manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=["V3.1_P10"] if passed else "no pass",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph)}/5"] if a == "V3.1_P10" else [None]) for a in ARMS},
                                 calendar_years=yrs_avg, low_price_share=share, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
