"""EXP-2026-10-13-walkforward-rule-choice (registered in logs/experiments.jsonl before any outcome was computed). MEASUREMENT.

Plain English: V3.2's rules were picked after seeing 2019-2026 results. How much of its backtest is that hindsight? Replay the
choice: every Jan 1 from 2021, an investor who knows only the past applies our own adoption rule to 96 rule sets (every
combination of the rules we considered, including ones we rejected) and invests the next year with whatever it picks. The gap
between V3.2 and that investor's return over 2021-2026 is the hindsight premium.
Rule sets: G1 base (model top 9 of hot-industry stocks passing the trend test) x F drop financials x T drop fading-theme-only
x L liquidity bar (old V3 bar / Rs 1 cr ADV + price > 50 / Rs 1 cr + price > 10) x N new listings skip the trend test x X drop
trade-for-trade x PB promoter buyers first within the pool's top 25. V3.2 = FT|1cr50|NX-. BASE = --|old|---.
Engine: test_v3_l1_newlist_cost.run_cost (dividends, payouts as cash, liquidity-scaled costs), weekly entries from 2019-04-01,
schedules 0-4, entry at the signal-day close.
PRIMARY procedure: start Jan 2021 with BASE; each Jan 1 a candidate replaces the current set if, on the history so far (split
at its midpoint), it has higher CAGR in both halves and max drawdown not worse by > 2 pts, at schedule 0 and >= 4 of 5; if
several pass, the highest history CAGR (schedule 0). Year return = average of 5 schedules (2026 = year to date).
SECONDARY: each Jan 1 the highest history CAGR (schedule 0). Bands: premium <= 5 small, 5-15 moderate, > 15 large.
SGM_REPRO=1: log nothing.
"""
from __future__ import annotations

import itertools
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

EXP_ID = "EXP-2026-10-13-walkforward-rule-choice"
OUTDIR = ROOT / "logs/leader_sleeve/walkforward_rule_choice"
START, DECIDE, BOOST = pd.Timestamp("2019-04-01"), range(2021, 2027), 25
LBARS = {"old": "core_old", "1cr50": "core", "1cr10": "core_p10"}
V32, BASE = "FT|1cr50|NX-", "--|old|---"


def name(f, t, l, n, x, pb):
    return f"{'F' if f else '-'}{'T' if t else '-'}|{l}|{'N' if n else '-'}{'X' if x else '-'}{'P' if pb else '-'}"


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); G1 = ctx["G1"]
    F0 = X["F"].copy(); F0["core_old"] = F0["core"].astype(bool)
    F1 = v3_rule.frame_v31(F0, D); F1["core_p10"] = (F1["adv"] >= 1.0) & (F1["raw"] > 10)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"])
    A["trade_date"] = pd.to_datetime(A["trade_date"])
    PBd = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}

    def pb(s, d):   # frozen copy of test_v32_promoter_buyers.pb
        a = PBd.get(s)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
        return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan

    def ranked(weeks, core_col, young_on, topn=BOOST):
        """v3_rule.ranked_v31 / test_industry_fundamentals.select_elig with the liquidity column and the new-listing switch."""
        W = F1[F1["trade_date"].isin(set(weeks)) & F1[core_col].fillna(False).astype(bool)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(G1[["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = (W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])
        if young_on:
            tr = tr | W["young"].fillna(False).astype(bool)
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in weeks}

    weeks = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
    pools = {(lb, n, o): ranked(weeks[o], col, n) for lb, col in LBARS.items() for n in (False, True) for o in range(5)}
    fin, fade = ctx["fin"], {}

    def fading(s, d):
        k = (s, d)
        if k not in fade:
            fade[k] = bool(ctx["fading_only"](s, d))
        return fade[k]

    def lists(f, t, lb, n, x, p, o):
        out = {}
        for d in weeks[o]:
            pool = pools[(lb, n, o)][d]
            if p:
                first = [s for s in pool[:BOOST] if pb(s, d) > 0]
                pool = first + [s for s in pool if s not in set(first)]
            out[d] = [s for s in pool[:9] if not (f and fin(s)) and not (t and fading(s, d)) and not (x and t2t(s, d))]
        return out

    CONF = list(itertools.product((0, 1), (0, 1), LBARS, (0, 1), (0, 1), (0, 1)))
    NAMES = [name(*c) for c in CONF]

    # ---- data gate (no outcomes yet)
    K = pd.DataFrame([dict(d=d, known=(1.0 if pd.notna(pb(s, d)) else np.nan)) for d in weeks[0] for s in pools[("1cr50", True, 0)][d][:BOOST]])
    Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
    checks = [dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), range(2018, cal[-1].year + 1)),
              dr.span("model scores", P["trade_date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.span("industry scores (G1 eligibility)", ctx["scores"]["date"], pd.Timestamp("2019-01-01"), cal[-1] - pd.Timedelta(days=10)),
              dr.coverage("promoter-buying signal known for the pool's top-25 names on list dates", K, "d", ["known"], range(START.year, cal[-1].year + 1))]
    if os.environ.get("SGM_REPRO") != "1":
        dr.gate(EXP_ID, checks, run="RESULT")
    del Px

    # ---- wiring (exact list equality, schedule 0)
    w0 = weeks[0]
    ref32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, w0, imap, P, ctx).items()}
    my32 = lists(1, 1, "1cr50", 1, 1, 0, 0)
    ref3 = v3_rule.picks(X["F"], w0, imap, P, ctx); my3 = lists(1, 1, "old", 0, 0, 0, 0)
    a, b = sum(my32[d] == ref32[d] for d in w0), sum(my3[d] == ref3.get(d, []) for d in w0)
    print(f"wiring: V3.2 set == v3_rule V3.1 minus trade-for-trade on {a}/{len(w0)} weeks · V3 set == v3_rule.picks on {b}/{len(w0)} weeks", flush=True)
    if a < len(w0) or b < len(w0):
        raise SystemExit("wiring check FAILED — no result reported")

    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    nav, cache = {}, {}
    for k, (c, nm) in enumerate(zip(CONF, NAMES)):
        for o in range(5):
            l = lists(*c, o)
            key = (o, tuple((d, tuple(v)) for d, v in l.items()))
            if key not in cache:
                cache[key] = run_cost(D, X, l, weeks[o], "E0", cost)[0]
            nav[(nm, o)] = cache[key]
        if k % 12 == 11:
            print(f"  {k + 1}/{len(CONF)} rule sets · {len(cache)} distinct runs · {time.time() - t0:.0f}s", flush=True)

    rr = subprocess.run(["/usr/bin/python3", "src/agentic/test_v32_promoter_buyers.py"], cwd=ROOT, env=dict(os.environ, SGM_REPRO="1"), capture_output=True, text=True)
    if rr.returncode != 0:
        raise SystemExit("wiring reference run failed — no result reported")
    ref = pd.read_csv(ROOT / "logs/leader_sleeve/v32_promoter_buyers/repro/results.csv").set_index("arm")
    for arm, nm in (("V3.2", V32), ("V3.2_PB", "FT|1cr50|NXP")):
        m = sp.metrics(nav[(nm, 0)]); got = [round(float(m[k]), 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
        want = [round(float(ref.loc[arm, k]), 2) for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")]
        print(f"wiring: {arm} {got} vs promoter-buyers code on the same data {want}", flush=True)
        if got != want:
            raise SystemExit("wiring check FAILED — no result reported")

    # ---- the replay
    def last_on(s, day):
        s = s[s.index <= day]
        return s.iloc[-1] if len(s) else np.nan

    def hist(s, end):
        s = s[s.index <= end]
        mid = s.index[0] + (s.index[-1] - s.index[0]) / 2
        def cg(x):
            yrs = (x.index[-1] - x.index[0]).days / 365.25
            return (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
        return dict(c1=cg(s[s.index <= mid]) * 100, c2=cg(s[s.index >= s.index[s.index <= mid][-1]]) * 100,
                    call=cg(s) * 100, dd=float((s / s.cummax() - 1).min() * 100))

    def yr(nm, Y):
        r = [last_on(nav[(nm, o)], pd.Timestamp(f"{Y}-12-31")) / last_on(nav[(nm, o)], pd.Timestamp(f"{Y - 1}-12-31")) - 1 for o in range(5)]
        return float(np.mean(r)) * 100

    H = {}
    path, cur, naive = [], BASE, []
    for Y in DECIDE:
        end = pd.Timestamp(f"{Y - 1}-12-31")
        H = {(nm, o): hist(nav[(nm, o)], end) for nm in NAMES for o in range(5)}
        def passes(c, inc):
            ok = [H[(c, o)]["c1"] > H[(inc, o)]["c1"] and H[(c, o)]["c2"] > H[(inc, o)]["c2"] and H[(c, o)]["dd"] >= H[(inc, o)]["dd"] - 2 for o in range(5)]
            return ok[0] and sum(ok) >= 4
        win = [c for c in NAMES if c != cur and passes(c, cur)]
        new = max(win, key=lambda c: H[(c, 0)]["call"]) if win else cur
        best = max(NAMES, key=lambda c: H[(c, 0)]["call"])
        path.append(dict(year=Y, held_before=cur, candidates_passing=len(win), chosen=new, chosen_return=round(yr(new, Y), 1),
                         v32_return=round(yr(V32, Y), 1), base_return=round(yr(BASE, Y), 1), naive_best_past=best, naive_return=round(yr(best, Y), 1)))
        cur = new
        print(f"{Y}: held {path[-1]['held_before']} · {len(win)} pass · choose {new} -> {path[-1]['chosen_return']:+.1f}% · V3.2 {path[-1]['v32_return']:+.1f}% · "
              f"BASE {path[-1]['base_return']:+.1f}% · naive {best} {path[-1]['naive_return']:+.1f}%", flush=True)
    W = pd.DataFrame(path)
    yrs = (cal[-1] - pd.Timestamp("2020-12-31")).days / 365.25
    chain = lambda col: ((1 + W[col] / 100).prod() ** (1 / yrs) - 1) * 100  # noqa: E731
    cagr = {k: round(chain(c), 1) for k, c in (("walk_forward", "chosen_return"), ("V3.2", "v32_return"), ("BASE", "base_return"), ("naive_best_past", "naive_return"))}
    prem = round(cagr["V3.2"] - cagr["walk_forward"], 1)
    band = "small" if prem <= 5 else ("moderate" if prem <= 15 else "large")
    full = pd.DataFrame([dict(rule_set=nm, cagr_s0=round(float(sp.metrics(nav[(nm, 0)])["cagr"]), 1),
                              cagr_avg5=round(float(np.mean([sp.metrics(nav[(nm, o)])["cagr"] for o in range(5)])), 1),
                              **{str(Y): round(yr(nm, Y), 1) for Y in range(2020, 2027)}) for nm in NAMES]).sort_values("cagr_avg5", ascending=False)
    rank = int((full["rule_set"].tolist()).index(V32)) + 1

    print(f"\n=== {EXP_ID} · 2021..{cal[-1].date()} (year returns = average of 5 schedules) ===")
    print(W.to_string(index=False))
    print(f"CAGR 2021-now: V3.2 (hindsight) {cagr['V3.2']} · walk-forward choice {cagr['walk_forward']} · naive best-past {cagr['naive_best_past']} · BASE (no rules) {cagr['BASE']}")
    print(f"HINDSIGHT PREMIUM: {prem} pts/yr -> {band} · V3.2 ranks {rank} of {len(NAMES)} rule sets on 2019-04..now CAGR (avg of 5 schedules)")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    W.to_csv(OUTDIR / "replay.csv", index=False); full.to_csv(OUTDIR / "rule_sets.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    for f, what in (("replay.csv", "one row per Jan 1 decision: rule set held, how many candidates passed, the choice, and year returns (percent, average of 5 schedules)"),
                    ("rule_sets.csv", "one row per rule set: CAGR 2019-04..now (schedule 0 and average of 5) and calendar-year returns (percent, average of 5 schedules)")):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_walkforward_rule_choice.py",
            definitions=__doc__, contents=what, rule_set_code="F/T/N/X/P letters = rule on, '-' = off; middle = liquidity bar", updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nHow much of V3.2's backtest is hindsight in choosing the rules. CAGR 2021-now: V3.2 {cagr['V3.2']}, "
                                      f"walk-forward choice {cagr['walk_forward']} -> premium {prem} pts/yr ({band}). replay.csv = the year-by-year choices; "
                                      f"rule_sets.csv = all {len(NAMES)} rule sets.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=f"hindsight premium {prem} pts/yr ({band})", cagr_2021_now=cagr,
                                 replay=path, v32_rank=f"{rank}/{len(NAMES)}", distinct_runs=len(cache))) + "\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
