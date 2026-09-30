"""EXP-2026-09-30-trait-rules (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: does Sri Lakshmi do better if, inside its weekly pool, it prefers stocks where buyers take delivery,
where the promoter has been buying, and smaller companies? Same engine, same pass rule as every other test.
The 2019-2022 years were already looked at for these traits (exploration), so the 2023+ result is the honest one.
Arms (reference G1 = production Sri Lakshmi):
  D1   drop the bottom half of the week's pool by 20-day average delivery % (unknown passes)
  D1S  same, judged within the pool's size tercile (small / mid / large): delivery relative to similar-size peers
  PB   promoter market purchase in the prior 90 days and pool rank <= 25 go first, then the rest by model
  SZ   drop the largest fifth of the week's pool by market cap
  ALL  SZ, then D1, then PB ordering
PASS = sp.beats vs G1 at phase 0 and in >= 4 of 5 phases. Also: conf-era gap, feature coverage by year, and the
+50% hit spread for delivery / promoter buying within each size tercile and the 6 largest industries (per era).
Output: logs/leader_sleeve/trait_rules/{results.csv, granularity.csv, coverage.csv, README.md} + manifests; RESULT line.
SGM_REPRO=1: writes to trait_rules/repro/ and does not touch the ledger (src/agentic/trust/reproduce.py).
SGM_PBUY_SOURCE=pit + SGM_RERUN_TAG=<tag> (2026-09-30, research queue): same registered test re-run with promoter
buying counted straight from data/derived/pit_history.parquet (repaired insider feed: promoter / promoter-group market
purchases disclosed in the 90 days up to the screen date, future-dated typos dropped); writes trait_rules/rerun_<tag>/
and logs RESULT as EXP-2026-09-30-trait-rules-RERUN-<tag>. A re-run with repaired data, not a new hypothesis.
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
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-30-trait-rules"
SCORES = ROOT / "data/derived/industry_scores_policy.parquet"
ROWS = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"
OUTDIR = ROOT / "logs/leader_sleeve/trait_rules"
ARMS = ("D1", "D1S", "PB", "SZ", "ALL")
PIT_SOURCE = os.environ.get("SGM_PBUY_SOURCE") == "pit"
RERUN = os.environ.get("SGM_RERUN_TAG", "")


def pool_frame(pool: dict, R: pd.DataFrame) -> pd.DataFrame:
    """Every week's full G1 pool in model order, with the traits as of the screen date (<= 7 days old)."""
    L = pd.DataFrame([dict(week=d, symbol=s, prank=i + 1) for d, names in pool.items() for i, s in enumerate(names)])
    if L.empty:
        return L
    L = L.sort_values("week")
    L = pd.merge_asof(L, R, left_on="week", right_on="fdate", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    return L.sort_values(["week", "prank"]).reset_index(drop=True)


def pit_buys(L: pd.DataFrame) -> list:
    """Promoter / promoter-group 'Market Purchase' disclosures in (screen date - 90 days, end of screen date]."""
    pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet", columns=["symbol", "personCategory", "acqMode", "date"])
    pit = pit[pit["personCategory"].fillna("").str.contains("Promoter", case=False) & (pit["acqMode"].fillna("").str.strip() == "Market Purchase")]
    d = pd.to_datetime(pit["date"], format="%d-%b-%Y %H:%M", errors="coerce")
    pit = pit.assign(d=d)[d.notna() & (d <= pd.Timestamp.now())]
    by = {s: np.sort(g["d"].to_numpy()) for s, g in pit.groupby("symbol")}
    out = []
    for w, s in zip(L["week"], L["symbol"]):
        a = by.get(s)
        if a is None:
            out.append(0); continue
        hi = np.searchsorted(a, np.datetime64(w + pd.Timedelta(hours=23, minutes=59)), side="right")
        lo = np.searchsorted(a, np.datetime64(w - pd.Timedelta(days=90)), side="right")
        out.append(int(hi - lo))
    return out


def choose(g: pd.DataFrame, arm: str, top: int = 9) -> list:
    """One week's picks under an arm. g = that week's pool rows in model order."""
    keep = pd.Series(True, index=g.index)
    if arm in ("SZ", "ALL"):
        cap = g["mcap_cr"].quantile(0.8)
        keep &= g["mcap_cr"].isna() | (g["mcap_cr"] <= cap)
    if arm in ("D1", "ALL"):
        med = g.loc[keep, "deliv"].median()
        keep &= g["deliv"].isna() | (g["deliv"] >= med)
    if arm == "D1S":
        terc = pd.qcut(g["mcap_cr"].rank(method="first"), 3, labels=False) if g["mcap_cr"].notna().sum() >= 3 else pd.Series(0, index=g.index)
        med = g.groupby(terc)["deliv"].transform("median")
        keep &= g["deliv"].isna() | terc.isna() | (g["deliv"] >= med)
    h = g[keep]
    if arm in ("PB", "ALL"):
        boost = (h["pbuy"] > 0) & (h["prank"] <= 25)
        h = pd.concat([h[boost], h[~boost]])
    return h["symbol"].head(top).tolist()


def main() -> None:
    t0 = time.time()
    repro = os.environ.get("SGM_REPRO") == "1"
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(SCORES); S["date"] = pd.to_datetime(S["date"])
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    R = pd.read_parquet(ROWS, columns=["symbol", "trade_date", "avg_delivery_pct_20d", "prom_buys90", "mcap_cr"])
    R = R.rename(columns={"trade_date": "fdate", "avg_delivery_pct_20d": "deliv", "prom_buys90": "pbuy"}).sort_values("fdate")
    cal = D["cal"]
    grid = lambda o: [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp(tif.START)]  # noqa: E731

    def run_phase(o: int) -> tuple[dict, pd.DataFrame]:
        wk = grid(o)
        ref = tif.select_elig(X["F"], wk, imap, P, G1)
        top = sre.TOPN; sre.TOPN = 100000
        full = tif.select_elig(X["F"], wk, imap, P, G1)
        sre.TOPN = top
        L = pool_frame(full, R)
        if PIT_SOURCE and len(L):
            L["pbuy"] = pit_buys(L)
        byw = {d: g for d, g in L.groupby("week")} if len(L) else {}
        if o == 0:
            same = sum(ref.get(d, []) == (byw[d]["symbol"].head(9).tolist() if d in byw else []) for d in wk)
            print(f"wiring check: rule-free selector reproduces G1 on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — no result reported")
        sels = {"G1": ref}
        sels.update({a: {d: choose(byw[d], a) for d in wk if d in byw} for a in ARMS})
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")},
                          avg_names=info["avg_names"], pct_touched50=info["pct_touched50"])
        return res, L

    R0, L0 = run_phase(0)
    ph = {k: [bool(sp.beats(R0[k], R0["G1"]))] for k in ARMS}
    for o in (1, 2, 3, 4):
        Ro, _ = run_phase(o)
        for k in ARMS:
            ph[k].append(bool(sp.beats(Ro[k], Ro["G1"])))
            R0[k].setdefault("phase_cagr", []).append(round(Ro[k]["cagr"], 1))
    for k in ARMS:
        R0[k]["phases_beaten"] = f"{sum(ph[k])}/5"; R0[k]["PASS"] = bool(ph[k][0] and sum(ph[k]) >= 4)
        R0[k]["conf_gap_vs_G1"] = round(R0[k]["cagr_conf"] - R0["G1"]["cagr_conf"], 1)
    print(f"\n=== {EXP_ID} · weekly cohorts {tif.START}.. · reference G1 ===")
    print(f"{'arm':5s} {'CAGR':>6s} {'disc':>6s} {'conf':>6s} {'maxDD':>7s} {'Sharpe':>6s} {'names':>5s} {'P+50':>5s} {'conf gap':>8s}  phases  PASS")
    for k in ["G1", *ARMS]:
        r = R0[k]
        print(f"{k:5s} {r['cagr']:6.1f} {r['cagr_disc']:6.1f} {r['cagr_conf']:6.1f} {r['maxdd']:7.1f} {r['sharpe']:6.2f} {r['avg_names']:5.1f} "
              f"{r['pct_touched50']:5.1f} {r.get('conf_gap_vs_G1', 0):+8.1f}  {r.get('phases_beaten', ''):6s}  {r.get('PASS', '')}")

    # coverage by year (the "is the data there" check) and the granularity read (per size tercile / industry, per era)
    L0["year"] = L0["week"].dt.year
    cov = L0.groupby("year").agg(pool_rows=("symbol", "size"), deliv=("deliv", lambda x: round(100 * x.notna().mean(), 1)),
                                 promoter_buys=("pbuy", lambda x: round(100 * x.notna().mean(), 1)),
                                 mcap=("mcap_cr", lambda x: round(100 * x.notna().mean(), 1)))
    print("\nfeature coverage in the G1 pool by year (% of pool rows known):\n" + cov.to_string())
    PX = rp.load_panel(["open", "high"])
    O, H = rp.wide(PX, "open", cal), rp.wide(PX, "high", cal); del PX
    hit = []
    for d, s in zip(L0["week"], L0["symbol"]):
        i0 = cal.get_loc(d) + 1
        if s not in O.columns or i0 + 126 > len(cal) or not np.isfinite(O[s].iloc[i0]):
            hit.append(np.nan); continue
        hit.append(float(np.nanmax(H[s].iloc[i0:i0 + 126].to_numpy()) >= 1.5 * O[s].iloc[i0]))
    L0["hit"] = hit
    L0["era"] = np.where(L0["week"] < tif.ERA, "disc", "conf")
    L0["size"] = L0.groupby("week")["mcap_cr"].transform(lambda x: pd.qcut(x.rank(method="first"), 3, labels=["small", "mid", "large"]) if x.notna().sum() >= 3 else np.nan)
    L0["industry"] = L0["symbol"].map(imap)
    L0["deliv_hi"] = L0["deliv"] >= L0.groupby("week")["deliv"].transform("median")
    L0["pb"] = L0["pbuy"] > 0
    top6 = L0["industry"].value_counts().head(6).index
    gran = []
    for era in ("disc", "conf"):
        for kind, groups in (("size", ["small", "mid", "large"]), ("industry", list(top6))):
            for grp in groups:
                x = L0[(L0["era"] == era) & (L0[kind] == grp) & L0["hit"].notna() & (L0["prank"] <= 25)]
                for trait in ("deliv_hi", "pb"):
                    a, b = x[x[trait] & x["deliv"].notna()] if trait == "deliv_hi" else x[x[trait]], x[~x[trait] & x["deliv"].notna()] if trait == "deliv_hi" else x[~x[trait]]
                    gran.append(dict(era=era, by=kind, group=grp, trait=trait, n_yes=len(a), n_no=len(b),
                                     hit_yes=round(100 * a["hit"].mean(), 1) if len(a) >= 20 else np.nan,
                                     hit_no=round(100 * b["hit"].mean(), 1) if len(b) >= 20 else np.nan))
    Gr = pd.DataFrame(gran); Gr["gap"] = (Gr["hit_yes"] - Gr["hit_no"]).round(1)
    print("\n+50% hit rate, pool ranks 1-25, trait yes vs no, within each size tercile / large industry (n >= 20 per side):")
    print(Gr.pivot_table(index=["by", "group", "trait"], columns="era", values="gap").round(1).to_string())

    out = OUTDIR / "repro" if repro else (OUTDIR / f"rerun_{RERUN}" if RERUN else OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **r) for k, r in R0.items()]).to_csv(out / "results.csv", index=False)
    Gr.to_csv(out / "granularity.csv", index=False); cov.to_csv(out / "coverage.csv")
    now = datetime.now().isoformat(timespec="seconds")
    units = dict(cagr="percent a year", maxdd="percent", conf_gap_vs_G1="percentage points of CAGR, 2023+",
                 hit_yes="percent of pool rows (rank <= 25) whose high reached 1.5x the next-open entry within 126 sessions",
                 gap="percentage points", coverage="percent of pool rows where the feature is known")
    for f in ("results.csv", "granularity.csv", "coverage.csv"):
        (out / (f + ".manifest.json")).write_text(json.dumps(dict(dataset=f, experiment=EXP_ID, producer="src/agentic/test_trait_rules.py",
                                                                  definitions=__doc__, units=units, updated=now), indent=1))
    passed = [k for k in ARMS if R0[k]["PASS"]]
    (out / "README.md").write_text(f"# {EXP_ID}\n\nDoes Sri Lakshmi improve by preferring high-delivery, promoter-bought and smaller "
                                   f"names inside its pool? Verdict: {'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}. "
                                   "results.csv = arms vs G1; granularity.csv = the trait effect inside each size tercile and the "
                                   "6 largest industries per era; coverage.csv = how much of the pool has each feature. See the manifests "
                                   "and src/agentic/test_trait_rules.py.\n")
    if repro:
        print(f"\nREPRO RUN (not logged) · VERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}")
        return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + (f"-RERUN-{RERUN}" if RERUN else "-RESULT"), verdict=passed or "no arm passes",
                                 note=("re-run of the registered test with repaired data (promoter buying from pit_history)" if RERUN else None),
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 conf_gap_vs_G1={k: R0[k]["conf_gap_vs_G1"] for k in ARMS}, cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats G1 by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
