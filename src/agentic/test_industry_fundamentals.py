"""EXP-2026-09-29-industry-fundamentals (registered in logs/experiments.jsonl before any outcome was computed).

PRIMARY (Q2): the model-ranked screen with a different INDUSTRY filter, same stocks rule and ranking:
  S1M (reference)  trend rule & mean-heat pct >= 0.70, top 9 by bake-off ensemble (sim_screen_rank_exit.select_s1)
  F1               trend rule & F_pct >= 0.70 (fundamentals only, no price heat)
  F2               trend rule & heat_pct >= 0.70 & F_pct >= 0.50
  H2               trend rule & heatfix_pct >= 0.70 & breadth >= 0.60 (median heat + breadth)
  Engine sim_screen_rank_exit.run_exit, exit E0 (126 sessions), close entry, 0.5% round trip, weekly cohorts 2019+.
  PASS = beats S1M by sim_leader_portfolio_7x.beats (CAGR higher in both eras, full-period maxDD not worse by > 2pp) at
  grid phase 0 AND in >= 4 of 5 phases. Production S0 reported for reference.
  Wiring check before any result: the custom selector with the S1M filter must reproduce select_s1's picks exactly.
SECONDARY (Q1, descriptive): weekly cross-industry Spearman IC of F and each pillar with the next 65-session
  equal-weight industry return (core members at the prior close), raw and partial on heat_pct (rank residuals); mean and
  t on non-overlapping 65-session samples, per era. D_flows 2022+ only.
Industry scores: data/derived/industry_scores.parquet (build_industry_scores.py).
Output: logs/leader_sleeve/industry_fundamentals/{results.csv, ic.csv, README.md} + manifests; RESULT line in experiments.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402

EXP_ID = "EXP-2026-09-29-industry-fundamentals"
SCORES = ROOT / "data/derived/industry_scores.parquet"
OUTDIR = ROOT / "logs/leader_sleeve/industry_fundamentals"
START, FWD, MIN_N = "2019-01-01", 65, 5
ERA = pd.Timestamp("2023-01-01")


def select_elig(F: pd.DataFrame, weekly: list, imap: pd.Series, P: pd.DataFrame, elig: pd.DataFrame) -> dict:
    """select_s1's stock rule and model ranking with an arbitrary (date, industry) eligibility table."""
    W = F[F["trade_date"].isin(set(weekly)) & F["core"]].copy()
    W["ind"] = W["symbol"].map(imap)
    W = W[W["ind"].notna() & W["ret60"].notna()]
    W = W.merge(elig[["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"])
                 & (W["sma_50"] > W["sma_200"]))
    Pool = W[trend].copy()
    L = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date")
    L["trade_date"] = L["trade_date"].astype("datetime64[ns]")
    m = pd.merge_asof(L, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
    Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
    return Pool.groupby("trade_date").head(sre.TOPN).groupby("trade_date")["symbol"].apply(list).to_dict()


def industry_daily(D: dict, imap: pd.Series) -> pd.DataFrame:
    cal, syms = D["cal"], D["syms"]
    R = pd.DataFrame(D["Rv"], index=cal, columns=syms)
    core = D["px"].pivot(index="trade_date", columns="symbol", values="core").reindex(index=cal, columns=syms)
    core = core.astype(float).shift(1).fillna(0).astype(bool)
    ind_of = imap.reindex(syms)
    out = {}
    for ind, cols in ind_of.groupby(ind_of).groups.items():
        v = np.where(core[list(cols)].to_numpy(), R[list(cols)].to_numpy(), np.nan)
        cnt = np.isfinite(v).sum(axis=1)
        with np.errstate(invalid="ignore"):
            s = np.nansum(v, axis=1) / np.where(cnt > 0, cnt, np.nan)
        out[ind] = pd.Series(np.where(cnt >= MIN_N, s, np.nan), index=cal)
    return pd.DataFrame(out)


def spearman(a: pd.Series, b: pd.Series) -> float:
    ok = a.notna() & b.notna()
    return float(a[ok].rank().corr(b[ok].rank())) if ok.sum() >= 10 else np.nan


def partial(a: pd.Series, b: pd.Series, z: pd.Series) -> float:
    ok = a.notna() & b.notna() & z.notna()
    if ok.sum() < 10:
        return np.nan
    ra, rb, rz = a[ok].rank(), b[ok].rank(), z[ok].rank()
    res = lambda y: y - np.polyval(np.polyfit(rz, y, 1), rz)  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(SCORES); S["date"] = pd.to_datetime(S["date"])
    cal = D["cal"]
    grid = lambda o: [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp(START)]  # noqa: E731
    ELIG = {"F1": S[S["F_pct"] >= 0.70], "F2": S[(S["heat_pct"] >= 0.70) & (S["F_pct"] >= 0.50)],
            "H2": S[(S["heatfix_pct"] >= 0.70) & (S["breadth"] >= 0.60)]}
    check = S[S["heat_pct"] >= 0.70]

    def run_phase(o: int) -> dict:
        wk = grid(o)
        ref = sre.select_s1(X["F"], wk, imap, "model", P)
        if o == 0:
            mine = select_elig(X["F"], wk, imap, P, check)
            same = sum(ref.get(d, []) == mine.get(d, []) for d in wk)
            print(f"wiring check: custom selector reproduces S1M on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — custom selector differs from select_s1; no result reported")
        sels = {"S1M": ref, "S0": sp.select(D["px"], wk, "core", 3, imap)}
        sels.update({k: select_elig(X["F"], wk, imap, P, e) for k, e in ELIG.items()})
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")},
                          avg_names=info["avg_names"], pct_touched50=info["pct_touched50"], years=m["years"])
        return res

    R0 = run_phase(0)
    phases = {k: [bool(sp.beats(R0[k], R0["S1M"]))] for k in ELIG}
    for o in (1, 2, 3, 4):
        Ro = run_phase(o)
        for k in ELIG:
            phases[k].append(bool(sp.beats(Ro[k], Ro["S1M"])))
            R0[k].setdefault("phase_cagr", []).append(round(Ro[k]["cagr"], 1))
        R0["S1M"].setdefault("phase_cagr", []).append(round(Ro["S1M"]["cagr"], 1))
    for k in ELIG:
        R0[k]["beats_phase0"], R0[k]["phases_beaten"] = phases[k][0], f"{sum(phases[k])}/5"
        R0[k]["PASS"] = bool(phases[k][0] and sum(phases[k]) >= 4)
    print(f"\n=== {EXP_ID} · weekly cohorts {START}.. · 126-session ladder, close entry ===")
    print(f"{'arm':5s} {'CAGR':>6s} {'disc':>6s} {'conf':>6s} {'maxDD':>7s} {'Sharpe':>6s} {'names':>5s} {'P+50':>5s}  phases  PASS")
    for k in ["S0", "S1M", *ELIG]:
        r = R0[k]
        print(f"{k:5s} {r['cagr']:6.1f} {r['cagr_disc']:6.1f} {r['cagr_conf']:6.1f} {r['maxdd']:7.1f} {r['sharpe']:6.2f} "
              f"{r['avg_names']:5.1f} {r['pct_touched50']:5.1f}  {r.get('phases_beaten', ''):6s}  {r.get('PASS', '')}")

    # Q1: industry IC
    I = industry_daily(D, imap)
    lg = np.log1p(I.fillna(0)); cs = lg.cumsum()
    valid = I.notna().astype(int).cumsum()
    fwd = np.expm1(cs.shift(-FWD) - cs).where((valid.shift(-FWD) - valid) >= int(0.75 * FWD))
    fl = fwd.stack().rename("fwd").reset_index(); fl.columns = ["date", "industry", "fwd"]
    Q = S.merge(fl, on=["date", "industry"], how="inner")
    ics = []
    for d, g in Q.groupby("date"):
        row = dict(date=d)
        for c in ("F", "A", "B", "C", "D_flows", "heat_mean"):
            row[f"ic_{c}"] = spearman(g[c], g["fwd"])
            row[f"pic_{c}"] = partial(g[c], g["fwd"], g["heat_pct"]) if c != "heat_mean" else np.nan
        ics.append(row)
    IC = pd.DataFrame(ics).set_index("date").sort_index()
    samp = IC.iloc[::FWD]
    summ = {}
    for era, seg in (("disc", samp[samp.index < ERA]), ("conf", samp[samp.index >= ERA])):
        for c in IC.columns:
            x = seg[c].dropna()
            summ[f"{era}|{c}"] = dict(mean=float(x.mean()) if len(x) else np.nan, n=len(x),
                                      t=float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan)
    print("\nQ1 industry IC with the next 65-session industry return (non-overlapping samples; pic = partial on heat_pct):")
    for c in IC.columns:
        a, b = summ[f"disc|{c}"], summ[f"conf|{c}"]
        print(f"  {c:12s} disc {a['mean']:+.3f} (t {a['t']:+.1f}, n {a['n']}) | conf {b['mean']:+.3f} (t {b['t']:+.1f}, n {b['n']})")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **{c: v for c, v in r.items() if c != "years"}) for k, r in R0.items()]).to_csv(OUTDIR / "results.csv", index=False)
    IC.to_csv(OUTDIR / "ic.csv")
    now = datetime.now().isoformat(timespec="seconds")
    for f, desc in (("results.csv", "one row per arm (phase 0 metrics; phase_cagr = phases 1-4)"), ("ic.csv", "per-date cross-industry Spearman IC (ic_) and partial IC on heat_pct (pic_)")):
        (OUTDIR / (f + ".manifest.json")).write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_industry_fundamentals.py",
            description=desc, definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent", ic="Spearman correlation"), updated=now), indent=1))
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nRegistered industry-filter test (Phase 1). `results.csv`: portfolio arms; `ic.csv`: industry-level "
                                      "IC series. Definitions and the pass rule are in the manifests and src/agentic/test_industry_fundamentals.py.\n")
    passed = [k for k in ELIG if R0[k]["PASS"]]
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=passed or "no arm passes",
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 ic={k: {kk: round(vv, 3) if isinstance(vv, float) and np.isfinite(vv) else vv for kk, vv in v.items()} for k, v in summ.items() if "|ic_F" in k or "|pic_F" in k},
                                 cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats S1M by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
