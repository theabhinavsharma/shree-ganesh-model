"""EXP-2026-09-30-bulk-buyers (registered in logs/experiments.jsonl before the bulk-deal history was even fetched).

Plain English: NSE names the client in every bulk deal (one client trading > 0.5% of a company in a day). Most are
high-frequency desks that buy and sell the same stock the same day. Take those out; does it help Sri Lakshmi to put
first the stocks that a fund (or a well-known investor) bought outright in the last 60 days, or to skip stocks a fund
sold? Same engine, same pass rule as every test. Neither era has been looked at for this data.
Events (per client, stock, day; bulk deals dated on or before the screen date, published the same evening):
  round trip       the client bought AND sold that stock that day -> ignored
  churner          a client whose deals in the previous calendar year were >= 50% round trips -> ignored
  institution      client name matches MUTUAL FUND | MF | ASSET MANAGEMENT | AMC | INSURANCE | LIFE | PENSION |
                   PROVIDENT | FUND(S) | TRUSTEE
  known investor   client name contains a name from data/derived/superstar_holdings.parquet (list compiled in 2026:
                   hindsight in who counts as "known", reported as its own arm)
Arms (reference G1 = production Sri Lakshmi):
  BI  institution net buy in the prior 60 days and pool rank <= 25 go first, then the rest by model; top 9
  BK  same for a known investor
  BA  same for any client that is not a churner
  XI  drop pool names with an institution net sell in the prior 60 days
PASS = sp.beats vs G1 at phase 0 and in >= 4 of 5 phases. Output: logs/leader_sleeve/bulk_buyers/ + RESULT line.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-30-bulk-buyers"
OUTDIR = ROOT / "logs/leader_sleeve/bulk_buyers"
ARMS = ("BI", "BK", "BA", "XI")
INST = re.compile(r"MUTUAL FUND|\bMF\b|ASSET MANAGEMENT|\bAMC\b|INSURANCE|\bLIFE\b|PENSION|PROVIDENT|\bFUNDS?\b|TRUSTEE")


def events() -> dict[str, dict[str, np.ndarray]]:
    B = pd.read_parquet(ROOT / "data/derived/bulk_deals_history.parquet")
    B["client_name"] = B["client_name"].str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    E = B.pivot_table(index=["trade_date", "symbol", "client_name"], columns="buy_sell", values="qty", aggfunc="sum", fill_value=0).reset_index()
    for c in ("BUY", "SELL"):
        if c not in E:
            E[c] = 0
    E["rt"] = (E["BUY"] > 0) & (E["SELL"] > 0)
    E["year"] = E["trade_date"].dt.year
    churn = E.groupby(["client_name", "year"])["rt"].mean()
    prev = pd.MultiIndex.from_arrays([E["client_name"], E["year"] - 1])
    E["churner"] = churn.reindex(prev).to_numpy() >= 0.5
    names = pd.read_parquet(ROOT / "data/derived/superstar_holdings.parquet", columns=["investor_name"])["investor_name"].dropna()
    known = sorted({n.upper().strip() for n in names if len(n.strip()) >= 6})
    E["inst"] = E["client_name"].str.contains(INST)
    E["known"] = E["client_name"].map(lambda c: any(k in c for k in known))
    clean = ~E["rt"] & ~E["churner"]
    kinds = {"inst_buy": clean & E["inst"] & (E["BUY"] > 0), "known_buy": clean & E["known"] & (E["BUY"] > 0),
             "any_buy": clean & (E["BUY"] > 0), "inst_sell": clean & E["inst"] & (E["SELL"] > 0)}
    print("events: " + " · ".join(f"{k} {int(v.sum())}" for k, v in kinds.items()) + f" · known-investor names {len(known)}", flush=True)
    return {k: {s: np.sort(g["trade_date"].to_numpy()) for s, g in E[m].groupby("symbol")} for k, m in kinds.items()}


def count(ev: dict, w: pd.Timestamp, s: str, days: int = 60) -> int:
    a = ev.get(s)
    if a is None:
        return 0
    return int(np.searchsorted(a, np.datetime64(w), side="right") - np.searchsorted(a, np.datetime64(w - pd.Timedelta(days=days)), side="right"))


def choose(names: list, w: pd.Timestamp, arm: str, EV: dict, top: int = 9) -> list:
    if arm == "XI":
        return [s for s in names if count(EV["inst_sell"], w, s) == 0][:top]
    key = {"BI": "inst_buy", "BK": "known_buy", "BA": "any_buy"}[arm]
    first = [s for s in names[:25] if count(EV[key], w, s) > 0]
    return (first + [s for s in names if s not in first])[:top]


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    EV = events()
    grid = lambda o: [d for d in sp.weekly_grid(D["cal"], o) if d >= pd.Timestamp(tif.START)]  # noqa: E731

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
            hits = {a: sum(choose(full.get(d, []), d, a, EV) != full.get(d, [])[:9] for d in wk) for a in ARMS}
            print("weeks where the arm changed the picks: " + " · ".join(f"{a} {n}" for a, n in hits.items()), flush=True)
        sels = {"G1": ref, **{a: {d: choose(full.get(d, []), d, a, EV) for d in wk} for a in ARMS}}
        res = {}
        for k, pk in sels.items():
            nav, info = sre.run_exit(D, X, pk, wk, "E0")
            m = sp.metrics(nav)
            res[k] = dict(**{c: m[c] for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "final")},
                          avg_names=info["avg_names"], pct_touched50=info["pct_touched50"])
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
        print(f"{k:3s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · "
              f"names {r['avg_names']:.1f} · phases {r.get('phases_beaten', '')} · PASS {r.get('PASS', '')}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **r) for k, r in R0.items()]).to_csv(OUTDIR / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_bulk_buyers.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), updated=now), indent=1))
    passed = [k for k in ARMS if R0[k]["PASS"]]
    (OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nDo named bulk-deal buyers help Sri Lakshmi? Verdict: "
                                      f"{'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}. See results.csv + manifest.\n")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=passed or "no arm passes",
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats G1 by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
