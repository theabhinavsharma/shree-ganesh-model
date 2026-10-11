"""EXP-2026-10-11-v3-l1-newlist-cost (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: the Rs 1 cr liquidity bar (alone, and with new listings after 3 months) beat V3 at a flat 0.5% trading cost.
Does it still win when smaller stocks cost more to trade?
Costs (every arm, every position, charged at entry as the engine does): research_panel.cost_rt on the stock's 20-day average
traded value on the list date: 0.5% round trip at >= Rs 5 cr, 1.0% at Rs 1-5 cr, 2.0% below Rs 1 cr. Stress: all doubled.
  V3c      live rule (Rs 5 cr bar)
  V3_L1c   Rs 1 cr bar (price > Rs 50 kept), everything else as V3
  V3_L1Nc  Rs 1 cr bar + new listings aged 63-251 sessions skip the trend test (they must clear the Rs 1 cr bar)
Window: weekly entries from 2019-01-01; eras 2019-22 / 2023+; phases 0-4; 126-session hold (copy of run_exit with a
per-position cost; wiring: at a flat 0.5% it reproduces run_exit exactly). PASS per arm = sp.beats vs V3c at phase 0 and in
>= 4 of 5. Also reported: calendar years 2023-2026 and the doubled-cost result. SGM_REPRO=1 / SGM_RERUN_TAG as usual.
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

EXP_ID = "EXP-2026-10-11-v3-l1-newlist-cost"
OUTDIR = ROOT / "logs/leader_sleeve/v3_l1_newlist_cost"
START, YEARS = pd.Timestamp("2019-01-01"), (2023, 2024, 2025, 2026)
ARMS = ("V3c", "V3_L1c", "V3_L1Nc")


def run_cost(D, X, picks, weekly, rule, costs_of):
    """sim_screen_rank_exit.run_exit with a per-position round-trip cost (costs_of(i0, cols) -> array)."""
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
                wts = (1 - np.asarray(costs_of(i0, cols), dtype=float)) / len(cols)
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
    L = L[["symbol", "trade_date", "raw", "adv"]]
    F1 = X["F"].merge(L, on=["symbol", "trade_date"], how="left")
    F1["core"] = (F1["adv"] >= 1.0) & (F1["raw"] > 50)
    F1 = F1.drop(columns=["raw", "adv"]).sort_values(["symbol", "trade_date"])
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F1["age"] = F1.groupby("symbol").cumcount(); F1["first_close"] = F1.groupby("symbol")["close"].transform("first")
    F1["young"] = F1["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F1["symbol"].isin(ren) & F1["age"].between(63, 251)

    def nl_select(weekly):
        W = F1[F1["trade_date"].isin(set(weekly)) & F1["core"]].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(ctx["G1"][["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g1 = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: [s for s in g1.get(d, []) if ctx["keep"](s, d)] for d in weekly}

    def arms(wk):
        return {"V3c": v3_rule.picks(X["F"], wk, imap, P, ctx), "V3_L1c": v3_rule.picks(F1.drop(columns=["age", "first_close", "young"]), wk, imap, P, ctx), "V3_L1Nc": nl_select(wk)}

    def cost_std(mult=1.0):
        return lambda i0, cols: mult * np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))

    # ---- data-readiness gate
    wk0 = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    sel0 = arms(wk0)
    picks = pd.DataFrame([dict(d=d, s=s) for a in ARMS for d in wk0 for s in sel0[a][d]])
    picks["adv_known"] = [1.0 if np.isfinite(ADV[D["dpos"][d], D["cidx"][s]]) else np.nan for d, s in zip(picks.d, picks.s)]
    cand = F1[(F1["core"]) & F1["trade_date"].isin(set(wk0))][["symbol", "trade_date"]].sort_values("trade_date")
    mm = pd.merge_asof(cand.astype({"trade_date": "datetime64[ns]"}), P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    cand["scored"] = mm["ensemble"].notna().astype(float).where(lambda x: x > 0).to_numpy()
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    yrs = range(START.year, cal[-1].year + 1)
    checks = [dr.coverage("model score known for Rs 1 cr-eligible stocks (incl. new listings) on list dates", cand, "trade_date", ["scored"], yrs),
              dr.coverage("average traded value known for every pick (sets its cost)", picks, "d", ["adv_known"], yrs),
              dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1))]
    tag = os.environ.get("SGM_RERUN_TAG")
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run=tag or "RESULT")

    # ---- wiring checks
    live = v3_rule.picks(X["F"], wk0, imap, P, ctx)
    same = sum(sel0["V3c"][d] == live.get(d, []) for d in wk0)
    nav_a, _ = sre.run_exit(D, X, live, wk0, "E0")
    nav_b, _ = run_cost(D, X, live, wk0, "E0", lambda i0, cols: np.full(len(cols), sp.COST))
    diff = float(np.nanmax(np.abs(nav_a.to_numpy() - nav_b.to_numpy())))
    print(f"wiring checks: V3 arm equals the live rule on {same}/{len(wk0)} weeks · cost engine at a flat 0.5% reproduces run_exit (max NAV difference {diff:.1e})", flush=True)
    if same < len(wk0) or diff > 1e-9:
        raise SystemExit("wiring check FAILED — no result reported")
    for a in ARMS:
        c = cost_std()
        avg_cost = np.mean([c(D["dpos"][d], np.array([D["cidx"][s] for s in sel0[a][d] if s in D["cidx"]])).mean() for d in wk0 if sel0[a][d]])
        print(f"{a}: average round-trip cost charged {avg_cost*100:.2f}%", flush=True)

    # ---- outcomes (standard costs) + stress (doubled)
    out = {}
    for lab, mult in (("std", 1.0), ("stress", 2.0)):
        res, ph, yr = {}, {a: [] for a in ARMS[1:]}, {a: [] for a in ARMS}
        for o in range(5):
            wk = wk0 if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
            sel = sel0 if o == 0 else arms(wk)
            R = {}
            for a in ARMS:
                nav, info = run_cost(D, X, sel[a], wk, "E0", cost_std(mult))
                m = sp.metrics(nav); R[a] = dict(**{k: m[k] for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, avg_names=info["avg_names"])
                n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
                yr[a].append({y: float(r.get(y, np.nan)) for y in YEARS})
            for a in ARMS[1:]:
                ph[a].append(bool(sp.beats(R[a], R["V3c"])))
            if o == 0:
                res = R
            print(f"[{lab}] phase {o}: " + " · ".join(f"{a} {R[a]['cagr_disc']:.1f}/{R[a]['cagr_conf']:.1f}/{R[a]['maxdd']:.1f}" for a in ARMS), flush=True)
        out[lab] = dict(res=res, ph=ph, years={a: {y: round(float(np.mean([x[y] for x in yr[a]])), 1) for y in YEARS} for a in ARMS},
                        verdict=[a for a in ARMS[1:] if ph[a][0] and sum(ph[a]) >= 4])
    print(f"\n=== {EXP_ID} · weekly entries {START.date()}..{cal[-1].date()} · phase 0 ===")
    for lab in ("std", "stress"):
        o = out[lab]
        print(f"-- {'liquidity-scaled costs (0.5 / 1 / 2%)' if lab == 'std' else 'stress: costs doubled (1 / 2 / 4%)'}")
        for a in ARMS:
            r = o["res"][a]
            print(f"   {a:8s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f}"
                  + (f" · beats V3c in {sum(o['ph'][a])}/5 (phase 0 {'yes' if o['ph'][a][0] else 'no'})" if a != "V3c" else "")
                  + " · years " + " ".join(f"{y}:{o['years'][a][y]:+.0f}" for y in YEARS))
        print(f"   verdict: {', '.join(o['verdict']) + ' PASS' if o['verdict'] else 'no arm passes'}")
    verdict = out["std"]["verdict"]
    print(f"VERDICT (registered, liquidity-scaled costs): {', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = [dict(costs=lab, arm=a, **out[lab]["res"][a], **{str(y): out[lab]["years"][a][y] for y in YEARS},
                 phases_beaten=(f"{sum(out[lab]['ph'][a])}/5" if a != "V3c" else None), PASS=(a in out[lab]["verdict"]) if a != "V3c" else None) for lab in out for a in ARMS]
    pd.DataFrame(rows).to_csv(OUTDIR / "results.csv", index=False)
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_v3_l1_newlist_cost.py", definitions=__doc__,
        units=dict(cagr="percent a year", maxdd="percent", years="calendar-year return percent, average of 5 schedules"), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nRs 1 cr liquidity bar (+ new listings) vs V3 with realistic, liquidity-scaled trading costs. Verdict: "
                                      f"{', '.join(verdict) + ' PASS' if verdict else 'no arm passes'}. results.csv has both cost levels and calendar years.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                                 arms={a: [round(out["std"]["res"][a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(out['std']['ph'][a])}/5" if a != "V3c" else None] for a in ARMS},
                                 stress={a: [round(out["stress"]["res"][a][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [f"{sum(out['stress']['ph'][a])}/5" if a != "V3c" else None] for a in ARMS},
                                 calendar_years=out["std"]["years"], cols="CAGR, 2019-22, 2023+, maxDD, phases")) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
