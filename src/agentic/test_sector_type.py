"""EXP-2026-09-30-sector-type (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: does Sri Lakshmi do better if it avoids industry types that rarely give +50% moves? Arms vs G1:
  NOFIN   skip Financial Services stocks            NOFIN3  skip Financial Services, FMCG and Energy
  NOBIG   skip the week's big-company industries (top fifth by median core-band market cap)
  AGE2    the industry must also have been hot on the previous weekly screen
  H80     industry heat pct >= 0.80 instead of 0.70
2019-2022 was explored for these traits, so 2023+ is the honest read. PASS = sp.beats vs G1 at phase 0 and >= 4/5.
Output: logs/leader_sleeve/sector_type/ + RESULT line.
"""
from __future__ import annotations

import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-30-sector-type"
OUTDIR = ROOT / "logs/leader_sleeve/sector_type"
import os
SHRINK = os.environ.get("SGM_SHRINK") == "1"     # EXP-2026-09-30-sector-shrink: drop from the top 9, no refill
ARMS = ("NOFIN_S", "NOFIN3_S", "NOBIG_S", "AGE2_S") if SHRINK else ("NOFIN", "NOFIN3", "NOBIG", "AGE2", "H80")
if SHRINK:
    EXP_ID, OUTDIR = "EXP-2026-09-30-sector-shrink", OUTDIR / "shrink"


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"]); sc["industry"] = sc["industry"].map(html.unescape)
    sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
    R = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "mcap_cr"]).sort_values("trade_date")
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    H80 = S[(S["heat_pct"] >= 0.80) & ~(S["P_pct"] < 0.30)]
    grid = lambda o: [d for d in sp.weekly_grid(D["cal"], o) if d >= pd.Timestamp(tif.START)]  # noqa: E731
    bad1, bad3 = {"Financial Services"}, {"Financial Services", "Fast Moving Consumer Goods", "Energy"}

    def big_industries(d) -> set:
        F = X["F"][(X["F"]["trade_date"] == d) & X["F"]["core"]][["symbol"]]
        m = R[R["trade_date"] <= d].groupby("symbol")["mcap_cr"].last()
        ind = pd.DataFrame(dict(ind=F["symbol"].map(imap).to_numpy(), mcap=m.reindex(F["symbol"]).to_numpy())).dropna()
        med = ind.groupby("ind")["mcap"].median()
        return set(med[med >= med.quantile(0.8)].index) if len(med) >= 5 else set()

    def run_phase(o: int) -> dict:
        wk = grid(o)
        ref = tif.select_elig(X["F"], wk, imap, P, G1)
        top = sre.TOPN; sre.TOPN = 100000
        full = tif.select_elig(X["F"], wk, imap, P, G1)
        sre.TOPN = top
        if o == 0:
            same = sum(ref.get(d, []) == full.get(d, [])[:9] for d in wk)
            print(f"wiring check: rule-free selector reproduces G1 on {same}/{len(wk)} weeks", flush=True)
            if same < len(wk):
                raise SystemExit("wiring check FAILED — no result reported")
        prev = {d: wk[i - 1] if i else d - pd.Timedelta(days=7) for i, d in enumerate(wk)}
        hot_prev = {}
        for d in wk:
            p = S[S["date"] <= prev[d]]
            p = p[p["date"] == p["date"].max()] if len(p) else p
            hot_prev[d] = set(p.loc[p["heat_pct"] >= 0.70, "industry"])
        if SHRINK:
            base = lambda d: ref.get(d, [])  # noqa: E731
            sels = {"G1": ref,
                    "NOFIN_S": {d: [s for s in base(d) if sector.get(imap.get(s)) not in bad1] for d in wk},
                    "NOFIN3_S": {d: [s for s in base(d) if sector.get(imap.get(s)) not in bad3] for d in wk},
                    "AGE2_S": {d: [s for s in base(d) if imap.get(s) in hot_prev[d]] for d in wk}}
            sels["NOBIG_S"] = {}
            for d in wk:
                b = big_industries(d)
                sels["NOBIG_S"][d] = [s for s in base(d) if imap.get(s) not in b]
            if o == 0:
                print("weeks where the arm changed the picks: " + " · ".join(f"{a} {sum(sels[a].get(d, []) != ref.get(d, []) for d in wk)}" for a in ARMS), flush=True)
            res = {}
            for k, pk in sels.items():
                nav, info = sre.run_exit(D, X, pk, wk, "E0")
                m = sp.metrics(nav)
                res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")}, avg_names=info["avg_names"])
            return res
        sels = {"G1": ref,
                "NOFIN": {d: [s for s in full.get(d, []) if sector.get(imap.get(s)) not in bad1][:9] for d in wk},
                "NOFIN3": {d: [s for s in full.get(d, []) if sector.get(imap.get(s)) not in bad3][:9] for d in wk},
                "AGE2": {d: [s for s in full.get(d, []) if imap.get(s) in hot_prev[d]][:9] for d in wk},
                "H80": tif.select_elig(X["F"], wk, imap, P, H80)}
        sels["NOBIG"] = {}
        for d in wk:
            b = big_industries(d)
            sels["NOBIG"][d] = [s for s in full.get(d, []) if imap.get(s) not in b][:9]
        if o == 0:
            print("weeks where the arm changed the picks: " + " · ".join(f"{a} {sum(sels[a].get(d, []) != ref.get(d, []) for d in wk)}" for a in ARMS), flush=True)
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")}, avg_names=info["avg_names"])
        return res

    R0 = run_phase(0)
    ph = {k: [bool(sp.beats(R0[k], R0["G1"]))] for k in ARMS}
    for o in (1, 2, 3, 4):
        Ro = run_phase(o)
        for k in ARMS:
            ph[k].append(bool(sp.beats(Ro[k], Ro["G1"])))
    for k in ARMS:
        R0[k]["phases_beaten"] = f"{sum(ph[k])}/5"; R0[k]["PASS"] = bool(ph[k][0] and sum(ph[k]) >= 4)
    print(f"\n=== {EXP_ID} · reference G1 ===")
    for k in ["G1", *ARMS]:
        r = R0[k]
        print(f"{k:6s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · names {r['avg_names']:.1f} · "
              f"phases {r.get('phases_beaten', '')} · PASS {r.get('PASS', '')}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **r) for k, r in R0.items()]).to_csv(OUTDIR / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_sector_type.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), updated=now), indent=1))
    passed = [k for k in ARMS if R0[k]["PASS"]]
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nIndustry-type rules vs Sri Lakshmi. Verdict: {'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=passed or "no arm passes",
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats G1 by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
