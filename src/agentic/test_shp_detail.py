"""EXP-2026-09-30-shp-detail (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: every quarter a company files who owns it. If mutual funds, foreign funds or Indian institutions
raised their stake last quarter, or small retail holders left (strong hands collecting), does it help Sri Lakshmi to
put those stocks first? And to skip stocks institutions are cutting? Same engine and pass rule as every test.
Data: data/derived/shareholding_detail_history.parquet, source == "xbrl" only (one consistent reading of the official
filings; 2021 onward). A change needs two consecutive quarters, so signals start around mid-2021: before that every
arm equals G1, and the 2019-2022 comparison rests on 2021-H2..2022 only (stated limit).
Signal at a screen date: the newest quarter filed on or before it (filing / submission date) and filed within the last
120 days, versus the quarter before it (both xbrl). Changes in percentage points of total shares.
Arms (reference G1 = production Sri Lakshmi):
  MFUP    mutual funds +0.5 pts or more        -> first if pool rank <= 25, then the rest by model; top 9
  FIIUP   foreign portfolio investors +0.5 pts -> same
  DIIUP   Indian institutions +0.5 pts         -> same
  RETDN   shareholders holding up to Rs 2 lakh fell 5% or more in number -> same
  INSTDN  Indian + foreign institutions together -1.0 pts or more -> drop the stock
PASS = sp.beats vs G1 at phase 0 and in >= 4 of 5 phases. Output: logs/leader_sleeve/shp_detail/ + RESULT line.
SGM_RERUN_TAG=<tag>: the same registered test re-run on extended data (e.g. filings back to 2016 read with
fetch_shp_detail.py --history); writes shp_detail/rerun_<tag>/ and logs EXP-2026-09-30-shp-detail-RERUN-<tag>.
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
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_industry_fundamentals as tif  # noqa: E402

EXP_ID = "EXP-2026-09-30-shp-detail"
OUTDIR = ROOT / "logs/leader_sleeve/shp_detail"
ARMS = ("MFUP", "FIIUP", "DIIUP", "RETDN", "INSTDN")


def signals() -> dict:
    H = pd.read_parquet(ROOT / "data/derived/shareholding_detail_history.parquet")
    H = H[H["source"] == "xbrl"].copy()
    H["quarter_end"] = pd.to_datetime(H["quarter_end"]); H["available"] = pd.to_datetime(H["available"])
    H = H.sort_values(["symbol", "quarter_end"])
    g = H.groupby("symbol")
    prev_q = g["quarter_end"].shift(1)
    consecutive = (H["quarter_end"] - prev_q).dt.days.between(80, 100)
    for c in ("mf_pct", "fii_fpi_pct", "dii_pct"):
        H[f"d_{c}"] = (H[c] - g[c].shift(1)).where(consecutive)
    H["d_retail"] = (H["retail_holders"] / g["retail_holders"].shift(1) - 1).where(consecutive)
    H = H.dropna(subset=["available"])
    print(f"xbrl quarters {len(H)} · with a consecutive previous quarter {int(consecutive.sum())} · symbols {H['symbol'].nunique()}", flush=True)
    return {s: d.sort_values("available").reset_index(drop=True) for s, d in H.groupby("symbol")}


def latest(sig: dict, s: str, w: pd.Timestamp):
    d = sig.get(s)
    if d is None:
        return None
    i = np.searchsorted(d["available"].to_numpy(), np.datetime64(w), side="right") - 1
    if i < 0 or (w - d["available"].iloc[i]).days > 120:
        return None
    return d.iloc[i]


def flag(r, arm: str) -> bool:
    if r is None:
        return False
    v = lambda c: r[c] if pd.notna(r[c]) else None  # noqa: E731
    if arm == "MFUP":
        return (v("d_mf_pct") or 0) >= 0.5
    if arm == "FIIUP":
        return (v("d_fii_fpi_pct") or 0) >= 0.5
    if arm == "DIIUP":
        return (v("d_dii_pct") or 0) >= 0.5
    if arm == "RETDN":
        return (v("d_retail") or 0) <= -0.05
    if arm == "INSTDN":
        a, b = v("d_dii_pct"), v("d_fii_fpi_pct")
        return a is not None and b is not None and a + b <= -1.0
    return False


def choose(names: list, w: pd.Timestamp, arm: str, sig: dict, top: int = 9) -> list:
    if arm == "INSTDN":
        return [s for s in names if not flag(latest(sig, s, w), arm)][:top]
    first = [s for s in names[:25] if flag(latest(sig, s, w), arm)]
    return (first + [s for s in names if s not in first])[:top]


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    G1 = S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)]
    sig = signals()
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
            ch = {a: sum(choose(full.get(d, []), d, a, sig) != full.get(d, [])[:9] for d in wk) for a in ARMS}
            print("weeks where the arm changed the picks: " + " · ".join(f"{a} {n}" for a, n in ch.items()), flush=True)
        sels = {"G1": ref, **{a: {d: choose(full.get(d, []), d, a, sig) for d in wk} for a in ARMS}}
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
        print(f"{k:6s} CAGR {r['cagr']:5.1f} · 2019-22 {r['cagr_disc']:5.1f} · 2023+ {r['cagr_conf']:5.1f} · maxDD {r['maxdd']:6.1f} · "
              f"phases {r.get('phases_beaten', '')} · PASS {r.get('PASS', '')}")
    tag = os.environ.get("SGM_RERUN_TAG", "")
    out = OUTDIR / f"rerun_{tag}" if tag else OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(arm=k, **r) for k, r in R0.items()]).to_csv(out / "results.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    (out / "results.csv.manifest.json").write_text(json.dumps(dict(dataset="results.csv", experiment=EXP_ID, producer="src/agentic/test_shp_detail.py",
        definitions=__doc__, units=dict(cagr="percent a year", maxdd="percent"), updated=now), indent=1))
    passed = [k for k in ARMS if R0[k]["PASS"]]
    (out / "README.md").write_text(f"# {EXP_ID}\n\nDo quarterly shareholding changes help Sri Lakshmi? Verdict: "
                                      f"{'PASS: ' + ', '.join(passed) if passed else 'no arm passes'}. See results.csv + manifest.\n")
    if os.environ.get("SGM_REPRO") == "1":
        print("REPRO RUN (not logged)"); return
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=passed or "no arm passes",
                                 note=("same registered rules, data extended back to 2016 (no re-tuning)" if tag else None),
                                 arms={k: [round(R0[k][c], 1) for c in ("cagr", "cagr_disc", "cagr_conf", "maxdd")] + [R0[k].get("phases_beaten")] for k in R0},
                                 cols="CAGR, disc, conf, maxDD, phases"), default=str) + "\n")
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats G1 by the registered rule'} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
