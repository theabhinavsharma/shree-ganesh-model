"""EXP-2026-10-12-v31-fpi-sector-tilt (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: inside each weekly V3.1 list, put more money into names whose sector foreign investors (FPIs) are still
buying and less, or nothing, into names whose sector they are selling. The week's total stays the same.
Data: data/derived/nsdl_fpi_sector.parquet (NSDL fortnightly sector-wise FPI reports, official), a report used only from the
next report's date. Sector flow = FPI equity net investment over the report's two fortnights / FPI equity holdings at its
middle date, ranked across the report's sectors (Sovereign, Others, Diversified and sectors under Rs 1,000 crore left out).
Industry -> sector: configs/industry_to_nsdl_sector.json (new24 / old35, written before any outcome).
Arms: TILT_050_150 (0.5 / 1 / 1.5 by sector third), TILT_075_125, DROP_LOW (bottom-third names dropped, money to the rest;
kept as is if every name would go). Weights scaled so the list's total is unchanged.
Engine: run_wcost = test_v3_l1_newlist_cost.run_cost with per-name weights; wiring: equal weights reproduce run_cost exactly.
PASS per arm: CAGR higher in both eras and max drawdown not worse by > 2 points, at schedule 0 and in >= 4 of 5; a rule
change needs >= 2 of the 3 arms to pass.
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

EXP_ID = "EXP-2026-10-12-v31-fpi-sector-tilt"
OUTDIR = ROOT / "logs/leader_sleeve/v31_fpi_sector_tilt"
START, SPLIT, YEARS, HOLD = pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), (2023, 2024, 2025, 2026), 126
LEAVE_OUT, MIN_AUC = {"Sovereign", "Others", "Diversified"}, 1000.0
ARMS = {"TILT_050_150": (0.5, 1.0, 1.5), "TILT_075_125": (0.75, 1.0, 1.25), "DROP_LOW": (0.0, 1.0, 1.0)}


def run_wcost(D, X, picks, weekly, rule, costs_of, weights_of):
    """Copy of test_v3_l1_newlist_cost.run_cost with per-name weights (weights_of(d, names) -> array summing to anything > 0)."""
    Rv, dpos, cidx, cal, nextrow = D["Rv"], D["dpos"], D["cidx"], D["cal"], D["nextrow"]
    T, slots = len(cal), sp.slots_for(sre.HOLD)
    start_i = dpos[weekly[0]]
    V = np.full((slots, T), np.nan); names_n = []
    for s in range(slots):
        cap = 1.0; v = V[s]; v[start_i:] = cap; late = np.zeros(T); pend = []
        for k in range(s, len(weekly), slots):
            d = weekly[k]; i0 = dpos[d]
            end = min(i0 + sre.HOLD, T - 1)
            nxt = dpos[weekly[k + slots]] if k + slots < len(weekly) else T - 1
            keep = []
            for j, pj in pend:
                if j <= i0:
                    cap += pj; late[j + 1:i0 + 1] += pj
                else:
                    keep.append((j, pj))
            pend = keep
            names = [n for n in picks.get(d, []) if n in cidx]
            if names and end > i0:
                cols = np.array([cidx[n] for n in names])
                seg = np.nan_to_num(Rv[i0 + 1:end + 1, cols])
                growth = np.cumprod(1 + seg, axis=0)
                w = np.asarray(weights_of(d, names), dtype=float); w = w / w.sum()
                wts = (1 - np.asarray(costs_of(i0, cols), dtype=float)) * w
                vals, held = np.empty_like(growth), np.ones(len(cols))
                for q, c in enumerate(cols):
                    vp, hu, lk, kind = sre.exit_path(growth[:, q], X["orl"][i0 + 1:end + 1, c], X["hrl"][i0 + 1:end + 1, c], X["below"][i0 + 1:end + 1, c], rule)
                    vals[:, q], held[q] = vp, hu
                path = (vals * wts).sum(axis=1)
                v[i0 + 1:end + 1] = cap * path
                heldval = cap * wts * held * growth[-1]
                cap = cap * float(path[-1]); names_n.append(len(cols))
                if i0 + sre.HOLD <= T - 1:
                    for q, c in enumerate(cols):
                        if held[q] <= 0:
                            continue
                        j = int(nextrow[end, c])
                        if end < j < T:
                            gj = np.cumprod(1 + np.nan_to_num(Rv[end + 1:j + 1, c]))
                            late[end + 1:j + 1] += heldval[q] * gj; cap -= heldval[q]; pend.append((j, heldval[q] * float(gj[-1])))
            else:
                v[i0 + 1:end + 1] = cap
            v[end + 1:nxt + 1] = cap
        for j, pj in pend:
            late[j + 1:] += pj
        v[start_i:] += late[start_i:]
    nav = V[:, start_i:].sum(axis=0) / slots
    return pd.Series(nav, index=cal[start_i:]), dict(avg_names=float(np.mean(names_n)) if names_n else 0.0)


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

    def v31_lists(weeks: list) -> dict:
        """{date: V3.1 picks in model order} (frozen copy)."""
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
        g = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: [s for s in g.get(d, []) if ctx["keep"](s, d)] for d in weeks}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    lists = {o: v31_lists(weeks[o]) for o in range(5)}

    # ---- sector signal (point in time)
    SEC = pd.read_parquet(ROOT / "data/derived/nsdl_fpi_sector.parquet")
    SEC["report_date"] = pd.to_datetime(SEC["report_date"]); SEC["known_from"] = pd.to_datetime(SEC["known_from"])
    SEC = SEC[~SEC["sector"].isin(LEAVE_OUT) & (SEC["eq_auc_mid_cr"] >= MIN_AUC)].copy()
    SEC["flow"] = (SEC["eq_net_prev_cr"] + SEC["eq_net_cr"]) / SEC["eq_auc_mid_cr"]
    SEC["pct"] = SEC.groupby("report_date")["flow"].rank(pct=True)
    MAPS = json.loads((ROOT / "configs/industry_to_nsdl_sector.json").read_text())
    reps = SEC.dropna(subset=["known_from"]).groupby("report_date").agg(kf=("known_from", "first"), scheme=("scheme", "first")).sort_values("kf")
    by_rep = {r: g.set_index("sector")["pct"].to_dict() for r, g in SEC.groupby("report_date")}
    kf_arr, rep_arr = reps["kf"].to_numpy(), reps.index.to_numpy()

    def report_on(d):
        k = np.searchsorted(kf_arr, np.datetime64(d), side="right")
        return None if k == 0 else pd.Timestamp(rep_arr[k - 1])

    def third(s, d) -> float:
        """1 = top-third sector, -1 = bottom third, 0 = middle / not ranked / unmapped."""
        r = report_on(d)
        if r is None:
            return np.nan
        sec = MAPS[reps.loc[r, "scheme"]].get(imap.get(s))
        p = by_rep[r].get(sec)
        if p is None or not np.isfinite(p):
            return 0.0
        return 1.0 if p > 2 / 3 else (-1.0 if p <= 1 / 3 else 0.0)

    # ---- data gate (no outcomes yet)
    all_weeks = sorted(set(d for o in range(5) for d in weeks[o]))
    G1 = pd.DataFrame({"d": all_weeks, "known": [1.0 if report_on(d) is not None else np.nan for d in all_weeks]})
    pk_rows = []
    for o in range(5):
        for d in weeks[o]:
            r = report_on(d)
            for s in lists[o][d]:
                sec = MAPS[reps.loc[r, "scheme"]].get(imap.get(s)) if r is not None else None
                pk_rows.append(dict(d=d, ranked=1.0 if (r is not None and sec in by_rep[r]) else np.nan))
    G2 = pd.DataFrame(pk_rows)
    raw = json.loads((ROOT / "data/derived/nsdl_fpi_sector.parquet.manifest.json").read_text())
    n_rep, n_off = raw["reports"], len(raw["reconciliation"]["reports_off_by_more_than_5cr"])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("an NSDL sector report already known on every list date", G1, "d", ["known"], yrs, min_share=0.99),
              dr.coverage("V3.1 picks whose industry maps to a ranked NSDL sector", G2, "d", ["ranked"], yrs, min_share=0.90),
              dict(check="reconcile", name="NSDL sector rows add up to the report's Grand Total (within Rs 5 crore)", ok=(n_rep - n_off) / n_rep >= 0.99,
                   fails=[] if (n_rep - n_off) / n_rep >= 0.99 else [f"{n_off} of {n_rep} reports off"]),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    equal = lambda d, names: np.ones(len(names))  # noqa: E731

    def tilt(arm):
        lo, mid, hi = ARMS[arm]

        def w(d, names):
            t = np.array([third(s, d) for s in names]); t = np.nan_to_num(t)
            out = np.where(t > 0, hi, np.where(t < 0, lo, mid))
            return out if out.sum() > 0 else np.ones(len(names))
        return w

    def metrics(nav):
        m = sp.metrics(nav); n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
        return {**{k: float(m[k]) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{y: float(r.get(y, np.nan)) for y in YEARS}}

    # ---- wiring
    nav_a, _ = run_cost(D, X, lists[0], weeks[0], "E0", cost)
    nav_b, _ = run_wcost(D, X, lists[0], weeks[0], "E0", cost, equal)
    diff = float(np.nanmax(np.abs(nav_a.to_numpy() - nav_b.to_numpy())))
    base0 = metrics(nav_b); got = tuple(round(base0[k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd"))
    print(f"wiring check: weighted engine with equal weights vs run_cost max NAV difference {diff:.1e}; V3.1 = {got} (registered 51.8 / 48.9 / 54.9 / -40.0)", flush=True)
    if diff > 1e-9 or got != (51.8, 48.9, 54.9, -40.0):
        raise SystemExit("wiring check FAILED — no result reported")

    res, ph = {"V3.1": []}, {a: [] for a in ARMS}
    for a in ARMS:
        res[a] = []
    for o in range(5):
        b = base0 if o == 0 else metrics(run_wcost(D, X, lists[o], weeks[o], "E0", cost, equal)[0])
        res["V3.1"].append(b)
        for a in ARMS:
            r = metrics(run_wcost(D, X, lists[o], weeks[o], "E0", cost, tilt(a))[0])
            res[a].append(r); ph[a].append(bool(sp.beats(r, b)))
        print(f"schedule {o}: " + " · ".join(f"{a} {res[a][-1]['cagr_disc']:.1f}/{res[a][-1]['cagr_conf']:.1f}/{res[a][-1]['maxdd']:.1f}" for a in res), flush=True)
    passed = [a for a in ARMS if ph[a][0] and sum(ph[a]) >= 4]
    real = len(passed) >= 2
    yrs_avg = {a: {y: round(float(np.mean([x[y] for x in res[a]])), 1) for y in YEARS} for a in res}

    # ---- pick level by sector third (schedule 0, full 6-month windows)
    PXo = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PXo, c, cal) for c in ("open", "high", "close")); del PXo
    On, Cn = O.to_numpy(), C.ffill(limit=300).to_numpy(); Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)
    Q = pd.DataFrame([dict(trade_date=d, symbol=s, third=third(s, d)) for d in weeks[0] for s in lists[0][d]])
    e = Q["trade_date"].map(pos).to_numpy() + 1; c = Q["symbol"].map(col).fillna(-1).astype(int).to_numpy()
    ok = (c >= 0) & (e + HOLD - 1 <= T - 1); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
    entry = np.where(ok, On[e2, c2], np.nan); good = ok & (entry > 0)
    Q["ret"] = np.where(good, Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1, np.nan)
    Q["hit"] = np.where(good, (Hf[e2, c2] / entry - 1 >= 0.5).astype(float), np.nan)
    Q["era"] = np.where(Q["trade_date"] <= SPLIT, "2019-22", "2023+")
    Q["sector_third"] = Q["third"].map({1.0: "FPIs buying (top third)", 0.0: "middle / not ranked", -1.0: "FPIs selling (bottom third)"})
    pl = Q[Q["ret"].notna()].groupby(["era", "sector_third"]).agg(picks=("ret", "size"), avg6m=("ret", "mean"), med6m=("ret", "median"), hit=("hit", "mean"),
                                                                  lost30=("ret", lambda x: (x <= -0.3).mean())).round(3)
    share = Q.groupby([Q["trade_date"].dt.year, "sector_third"]).size().unstack(fill_value=0)

    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · schedule 0 (years = avg of 5) ===")
    for a in res:
        r = res[a][0]
        print(f"{a:13s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · range {min(x['cagr'] for x in res[a]):.1f}..{max(x['cagr'] for x in res[a]):.1f}"
              + " · years " + " ".join(f"{y}:{yrs_avg[a][y]:+.0f}" for y in YEARS) + (f" · beats V3.1 in {sum(ph[a])}/5 (schedule 0 {'yes' if ph[a][0] else 'no'})" if a != "V3.1" else ""))
    print("picks by sector third (full 6-month windows):\n" + pl.to_string())
    print("picks per year by sector third:\n" + share.to_string())
    print(f"arms passing: {passed} · VERDICT: {'sector tilt is real (>= 2 of 3 arms pass)' if real else 'no pass'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=a, **{k: round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, **{str(y): yrs_avg[a][y] for y in YEARS},
                       phases_beaten=(f"{sum(ph[a])}/5" if a != "V3.1" else None), PASS=((a in passed) if a != "V3.1" else None)) for a in res]).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v31_fpi_sector_tilt.py", definitions=__doc__,
        arms_passing=passed, rule_change_candidate=real, by_sector_third=pl.reset_index().to_dict("records"),
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules", avg6m="fraction", hit="share reaching +50% within 126 sessions"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nWeights inside each V3.1 list tilted by NSDL sector-wise FPI flows. Arms passing: {passed or 'none'}; "
                                      f"{'rule-change candidate' if real else 'no pass'}. Arms in results.csv; picks by sector third in the manifest.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=(["sector tilt real"] + passed) if real else f"no pass ({passed or 'no arm'})",
                                 arms={a: [round(res[a][0][k], 1) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + ([f"{sum(ph[a])}/5"] if a != "V3.1" else [None]) for a in res},
                                 calendar_years=yrs_avg, cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
