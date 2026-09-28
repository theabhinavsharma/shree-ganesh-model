"""SCREEN x RANK x EXIT for the leader sleeve (EXP-2026-09-28-screen-rank-exit; registered in logs/experiments.jsonl
before any run).

User 2026-09-28: "1. Screen for trend plus a hot or warming industry. 2. Let the model rank within that pool.
3. Test the exit rules on top of it."

Portfolio engine: sim_leader_portfolio_7x.run's registered ladder (26 overlapping weekly slots, 126-session hold, entry
at the signal-day close, 0.5% round trip at entry, delisted names frozen at the last close, a name with no row on its
exit session held to its next row). The one change: a per-name exit rule may end a position, or half of it, before
session t+126. Exited money is cash in its slot until the slot's next cohort. The 0.5% round trip already covers the
sale, so an early or split exit costs nothing extra.

Selections (core band in every arm: 20-day ADV >= Rs5cr and then-traded close > Rs50):
  S0   PRODUCTION     sp.select: industries in the top decile of heat (mean own ret60 of their core names, >= 5 names),
                      top 3 by own ret60 in each, ret252 > 50% (screen_theme_leaders / BASELINE of the 7x factorial).
  S1R  SCREEN + MOM   trend screen: close >= 1.5x its 252-row low, ret252 >= 30%, close > SMA200, SMA50 > SMA200;
                      industry heat percentile >= 0.70 (hot OR warming; same heat as S0). Top 9 by own ret60.
  S1M  SCREEN + MODEL same pool, top 9 by the bake-off ensemble score (model_bakeoff_1p5x walk-forward out-of-sample
                      scores, bakeoff_preds.parquet): the symbol's latest weekly score dated on or before the cohort date
                      and at most 7 days old; names without one rank last.
  9 = S0's average names per cohort (8.8).
Exits (every selection):
  E0 TIME          hold 126 sessions (production).
  E1 HALF50        sell half when the day's high reaches 1.5x entry, filled at 1.5x (at the open if it opens above);
                   the rest is held to session 126.
  E2 TRAIL25       sell everything at the next open after a close more than 25% below the highest close since entry
                   (entry close included).
  E3 DMA200        sell everything at the next open after a close below the 200-day average (panel sma_200).
  E4 HALF50+TRAIL  E2's rule on the whole position until the +50% touch; then half is sold as in E1 and the other half
                   stays on E2's rule.
  A trigger on the last held session is the time exit at that close. Intraday triggers use the symbol's own open/high
  relative to its prior close; sessions without a row cannot trigger.
Window: weekly cohorts from 2019-01-01 to the panel end for every arm (model scores start in 2019); eras disc
2019-2022, conf 2023+ (sp.metrics).
Registered rule: an arm beats S0-E0 if (sp.beats) CAGR is higher in BOTH eras and the full-period max drawdown is no
worse by more than 2 points, AND it also does so in >= 4 of the 5 weekly-grid phases (offset 0..4). 14 arms are
compared with S0-E0, so a pass without the phase check is weak evidence.
Secondary (not part of the rule): S0 x E0..E4 from 2016-06-01, the production cell's full history.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402

EXP_ID = "EXP-2026-09-28-screen-rank-exit"
PREDS = ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet"
OUTDIR = ROOT / "logs/leader_sleeve/screen_rank_exit"
START2, HOLD, TOPN, HOTWARM, TOUCH, TRAIL = "2019-01-01", 126, 9, 0.70, 1.5, 0.25
EXITS = ("E0", "E1", "E2", "E3", "E4")
EXIT_NAMES = dict(E0="TIME 126", E1="HALF@+50%", E2="TRAIL 25%", E3="BELOW 200DMA", E4="HALF@+50% + TRAIL 25%")


# ---------------- data ----------------
def features(D: dict) -> dict:
    """Open/high relative to the prior close, the below-SMA200 flag (wide, aligned to D) and the weekly screen frame."""
    cal, syms = D["cal"], D["syms"]
    px = rp.load_panel(["open", "high", "low", "close", "sma_50", "sma_200"])
    px = px[px["trade_date"].isin(set(cal))].reset_index(drop=True)
    px["lo252"] = px.groupby("symbol")["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
    C = rp.wide(px, "close", cal).reindex(columns=syms)
    prev = C.ffill().shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        orl = (rp.wide(px, "open", cal).reindex(columns=syms) / prev).to_numpy()
        hrl = (rp.wide(px, "high", cal).reindex(columns=syms) / prev).to_numpy()
        below = (C < rp.wide(px, "sma_200", cal).reindex(columns=syms)).to_numpy()
    F = px[["symbol", "trade_date", "close", "lo252", "sma_50", "sma_200"]].merge(
        D["px"][["symbol", "trade_date", "ret60", "ret252", "core"]], on=["symbol", "trade_date"], how="inner")
    return dict(orl=orl, hrl=hrl, below=below, F=F)


def model_scores() -> pd.DataFrame:
    P = pd.read_parquet(PREDS, columns=["symbol", "trade_date", "ensemble"])
    P["trade_date"] = pd.to_datetime(P["trade_date"]).astype("datetime64[ns]")
    return P.dropna(subset=["ensemble"]).sort_values("trade_date")


def select_s1(F: pd.DataFrame, weekly: list, imap: pd.Series, rank: str, P: pd.DataFrame | None) -> dict:
    W = F[F["trade_date"].isin(set(weekly)) & F["core"]].copy()
    W["ind"] = W["symbol"].map(imap)
    W = W[W["ind"].notna() & W["ret60"].notna()]
    W["nm"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size")
    W = W[W["nm"] >= 5]                                              # heat exactly as sp.select
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("heat").reset_index()
    h["hp"] = h.groupby("trade_date")["heat"].rank(pct=True)
    W = W.merge(h[["trade_date", "ind", "hp"]], on=["trade_date", "ind"])
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"])
                 & (W["sma_50"] > W["sma_200"]))
    Pool = W[(W["hp"] >= HOTWARM) & trend].copy()
    if rank == "model":
        L = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date")
        L["trade_date"] = L["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(L, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
    else:
        Pool["score"] = Pool["ret60"]
    Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
    return Pool.groupby("trade_date").head(TOPN).groupby("trade_date")["symbol"].apply(list).to_dict()


# ---------------- exits ----------------
def exit_path(g: np.ndarray, orl: np.ndarray, hrl: np.ndarray, below: np.ndarray, rule: str
              ) -> tuple[np.ndarray, float, float, int]:
    """g = close path relative to the entry close over sessions t+1..t+L. Returns (value path, units still held at the
    end, cash already realised, exit kind: 0 held / 1 half sold / 2 fully sold early)."""
    L = len(g)
    if rule == "E0" or L == 0:
        return g, 1.0, 0.0, 0
    prev = np.concatenate([[1.0], g[:-1]])
    op, hi = prev * orl, prev * hrl

    def next_open(r: int) -> tuple[int, float]:
        if r + 1 >= L:
            return L - 1, float(g[L - 1])                        # trigger on the last session = the time exit
        x = op[r + 1]
        return r + 1, float(x if np.isfinite(x) and x > 0 else g[r + 1])

    def first(mask: np.ndarray) -> int | None:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else None

    touch = first(np.nan_to_num(hi, nan=-np.inf) >= TOUCH) if rule in ("E1", "E4") else None
    fill = float(max(op[touch], TOUCH)) if touch is not None and np.isfinite(op[touch]) else TOUCH
    if rule in ("E2", "E4"):
        peak = np.maximum.accumulate(np.concatenate([[1.0], g]))[1:]
        trig = first(g < (1 - TRAIL) * peak)
    elif rule == "E3":
        trig = first(below)
    else:
        trig = None
    v = g.copy()
    if rule in ("E2", "E3") or (rule == "E4" and trig is not None and (touch is None or trig < touch)):
        if trig is None:
            return v, 1.0, 0.0, 0
        e, x = next_open(trig)
        v[e:] = x
        return v, 0.0, x, 2
    if touch is None:                                             # E1 / E4 without a touch (and E4 without a trigger)
        return v, 1.0, 0.0, 0
    v[touch:] = 0.5 * fill + 0.5 * g[touch:]
    if rule == "E4" and trig is not None:                         # remaining half on the trailing rule
        e, x = next_open(trig)
        v[e:] = 0.5 * fill + 0.5 * x
        return v, 0.0, 0.5 * fill + 0.5 * x, 2
    return v, 0.5, 0.5 * fill, 1


def run_exit(D: dict, X: dict, picks: dict, weekly: list, rule: str) -> tuple[pd.Series, dict]:
    """sp.run (entry close, no gate, no rebalance, gap_exit next_row) with per-name exits."""
    Rv, dpos, cidx, cal, nextrow = D["Rv"], D["dpos"], D["cidx"], D["cal"], D["nextrow"]
    T, slots = len(cal), sp.slots_for(HOLD)
    start_i = dpos[weekly[0]]
    V = np.full((slots, T), np.nan)
    st = dict(positions=0, half_sold=0, sold_early=0, touched50=0, names=[], held_sessions=[])
    for s in range(slots):
        cap = 1.0
        v = V[s]
        v[start_i:] = cap
        late = np.zeros(T)
        pend = []
        for k in range(s, len(weekly), slots):
            d = weekly[k]; i0 = dpos[d]
            end = min(i0 + HOLD, T - 1)
            nxt = dpos[weekly[k + slots]] if k + slots < len(weekly) else T - 1
            assert k + slots >= len(weekly) or nxt > end
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
                wts = np.full(len(cols), (1 - sp.COST) / len(cols))
                vals, held, locked = np.empty_like(growth), np.ones(len(cols)), np.zeros(len(cols))
                for q, c in enumerate(cols):
                    vp, hu, lk, kind = exit_path(growth[:, q], X["orl"][i0 + 1:end + 1, c], X["hrl"][i0 + 1:end + 1, c],
                                                 X["below"][i0 + 1:end + 1, c], rule)
                    vals[:, q], held[q], locked[q] = vp, hu, lk
                    st["positions"] += 1; st["half_sold"] += kind == 1; st["sold_early"] += kind == 2
                    st["touched50"] += bool((np.concatenate([[1.0], growth[:-1, q]]) * X["hrl"][i0 + 1:end + 1, c] >= TOUCH).any())
                path = (vals * wts).sum(axis=1)
                v[i0 + 1:end + 1] = cap * path
                heldval = cap * wts * held * growth[-1]                  # still-held value at the scheduled exit
                cap = cap * float(path[-1])
                st["names"].append(len(cols))
                if i0 + HOLD <= T - 1:
                    for q, c in enumerate(cols):
                        if held[q] <= 0:
                            continue
                        j = int(nextrow[end, c])
                        if end < j < T:                                  # no row at exit: held to its next row
                            gj = np.cumprod(1 + np.nan_to_num(Rv[end + 1:j + 1, c]))
                            late[end + 1:j + 1] += heldval[q] * gj
                            cap -= heldval[q]
                            pend.append((j, heldval[q] * float(gj[-1])))
            else:
                v[i0 + 1:end + 1] = cap
            v[end + 1:nxt + 1] = cap
        for j, pj in pend:
            late[j + 1:] += pj
        v[start_i:] += late[start_i:]
    nav = V[:, start_i:].sum(axis=0) / slots
    info = dict(avg_names=float(np.mean(st["names"])) if st["names"] else 0.0, positions=st["positions"],
                pct_touched50=st["touched50"] / max(st["positions"], 1) * 100,
                pct_half_sold=st["half_sold"] / max(st["positions"], 1) * 100,
                pct_sold_early=st["sold_early"] / max(st["positions"], 1) * 100)
    return pd.Series(nav, index=cal[start_i:]), info


# ---------------- experiment ----------------
def arms(D: dict, X: dict, weekly: list, imap: pd.Series, P: pd.DataFrame, only_s0: bool = False) -> dict:
    sels = {"S0": sp.select(D["px"], weekly, "core", 3, imap)}
    if not only_s0:
        sels["S1R"] = select_s1(X["F"], weekly, imap, "mom", None)
        sels["S1M"] = select_s1(X["F"], weekly, imap, "model", P)
    out = {}
    for sname, pk in sels.items():
        for e in EXITS:
            nav, info = run_exit(D, X, pk, weekly, e)
            m = sp.metrics(nav)
            out[f"{sname}-{e}"] = dict(sel=sname, exit=e, exit_name=EXIT_NAMES[e],
                                       **{k: m[k] for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd", "maxdd_disc",
                                                            "maxdd_conf", "sharpe", "final")}, **info, years=m["years"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-phases", action="store_true", help="skip the phase robustness check (smoke runs)")
    ap.add_argument("--symbols", default=None, help="comma list: restrict the panel (smoke test; nothing is written)")
    a = ap.parse_args()
    t0 = time.time()
    syms = [s.strip() for s in a.symbols.split(",")] if a.symbols else None
    D = sp.load(syms)
    X = features(D)
    imap = sp.industry_maps()["analogs"]
    P = model_scores()
    cal = D["cal"]
    print(f"loaded {len(cal)} sessions, {len(D['syms'])} symbols · {time.time() - t0:.0f}s", flush=True)

    grid = lambda o, start: [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp(start)]  # noqa: E731
    R = arms(D, X, grid(0, START2), imap, P)
    base = R["S0-E0"]
    for k, r in R.items():
        r["beats_phase0"] = sp.beats(r, base) if k != "S0-E0" else None
    cand = [k for k, r in R.items() if r["beats_phase0"]]
    phases: dict = {}
    if cand and not a.no_phases:
        for o in (1, 2, 3, 4):
            Ro = arms(D, X, grid(o, START2), imap, P)
            for k in cand:
                phases.setdefault(k, []).append(bool(sp.beats(Ro[k], Ro["S0-E0"])))
                R[k].setdefault("phase_cagr", []).append(round(Ro[k]["cagr"], 1))
    for k, r in R.items():
        if r["beats_phase0"]:
            n_ok = 1 + sum(phases.get(k, []))
            r["phases_beaten"] = f"{n_ok}/5" if phases else "phase check skipped"
            r["PASS"] = bool(phases) and n_ok >= 4
        else:
            r["PASS"] = False if k != "S0-E0" else None
    S0full = arms(D, X, grid(0, sp.START), imap, P, only_s0=True)

    cols = ["cagr", "cagr_disc", "cagr_conf", "maxdd", "sharpe", "avg_names", "pct_touched50", "pct_half_sold", "pct_sold_early"]
    print(f"\n=== {EXP_ID}: weekly cohorts {START2}.. (disc 2019-2022, conf 2023+), 126-session ladder, close entry ===")
    print(f"{'arm':8s} {'exit':24s} " + " ".join(f"{c[:10]:>10s}" for c in cols) + "   beats  phases  PASS")
    for k, r in R.items():
        print(f"{k:8s} {r['exit_name']:24s} " + " ".join(f"{r[c]:10.1f}" for c in cols)
              + f"   {str(r['beats_phase0'])[:5]:5s}  {r.get('phases_beaten', ''):6s}  {r['PASS']}")
    print(f"\n--- secondary: production selection S0 from {sp.START} (full history) ---")
    for k, r in S0full.items():
        print(f"{k:8s} {r['exit_name']:24s} " + " ".join(f"{r[c]:10.1f}" for c in cols))
    passed = [k for k, r in R.items() if r["PASS"]]
    print(f"\nVERDICT: {'PASS: ' + ', '.join(passed) if passed else 'no arm beats S0-E0 by the registered rule'}  ({time.time() - t0:.0f}s)")

    if syms:
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = [dict(window=START2, arm=k, **{kk: vv for kk, vv in r.items() if kk not in ("years",)}) for k, r in R.items()]
    rows += [dict(window=sp.START, arm=k, **{kk: vv for kk, vv in r.items() if kk != "years"}) for k, r in S0full.items()]
    pd.DataFrame(rows).to_csv(OUTDIR / "results.csv", index=False)
    years = {f"{w}|{k}": r["years"] for w, RR in ((START2, R), (sp.START, S0full)) for k, r in RR.items()}
    (OUTDIR / "years.json").write_text(json.dumps(years, indent=1))
    now = datetime.now().isoformat(timespec="seconds")
    (OUTDIR / "results.csv.manifest.json").write_text(json.dumps(dict(
        dataset="screen_rank_exit results", path=str((OUTDIR / "results.csv").relative_to(ROOT)), experiment=EXP_ID,
        producer="src/agentic/sim_screen_rank_exit.py", definitions=__doc__,
        units=dict(cagr="percent a year, sleeve NAV", maxdd="percent, worst peak-to-trough of the daily NAV",
                   pct_touched50="percent of positions whose high reached 1.5x entry within the hold",
                   pct_half_sold="percent of positions where half was sold at +50% and half held to the end",
                   pct_sold_early="percent of positions fully sold before session 126", avg_names="names per cohort"),
        inputs=[str(rp.PANEL.relative_to(ROOT)), str(PREDS.relative_to(ROOT)), "data/derived/mcap_pit.parquet",
                "data/derived/screener_industry.parquet", "data/derived/industry_analyst_labels.csv"],
        verdict=passed or "no arm passes", updated=now), indent=1))
    (OUTDIR / "README.md").write_text(
        f"# Screen x rank x exit (leader sleeve)\n\n{EXP_ID}, registered in logs/experiments.jsonl before the run. "
        "`results.csv` has one row per arm (selection x exit) for the registered 2019+ window and the secondary 2016+ "
        "production runs; `years.json` has calendar-year returns. Definitions and the pass rule are in the manifest and "
        "in src/agentic/sim_screen_rank_exit.py.\n")
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=now, id=EXP_ID + "-RESULT", verdict=passed or "no arm passes",
                                 table={k: [round(r["cagr"], 1), round(r["cagr_disc"], 1), round(r["cagr_conf"], 1),
                                            round(r["maxdd"], 1), r.get("phases_beaten")] for k, r in R.items()},
                                 secondary_2016={k: [round(r["cagr"], 1), round(r["cagr_disc"], 1), round(r["cagr_conf"], 1),
                                                     round(r["maxdd"], 1)] for k, r in S0full.items()},
                                 cols="CAGR, disc, conf, maxDD, phases", out=str(OUTDIR.relative_to(ROOT))), default=str) + "\n")


if __name__ == "__main__":
    main()
