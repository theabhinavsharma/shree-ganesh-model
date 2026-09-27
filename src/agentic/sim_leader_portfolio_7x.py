"""LEADER SLEEVE — 10-year PORTFOLIO backtest, full factorial of 3 levers (EXP-2026-09-27-leader-7x-factorial).

BASELINE = production cell: nse4 industry map, heat + rank on the core band (ADV>=5cr & close>50),
MEAN heat top decile (>=5 names), top-3 by own ret60, EXTENDED (ret252 > 50%).
Levers (every non-empty combination = 7 experiments):
  G  market gate  — enter a weekly cohort only if prior-session breadth (share of core-band stocks
                    above their SMA50, from the panel) >= 0.50, else that slot holds cash
  B  basket       — top-10 instead of top-3 per hot industry
  H  broad heat   — heat + rank on the PIT mcap >= Rs50cr universe; BUY only core-band names
Portfolio: 26 overlapping weekly slots (1/26 each), equal-weight buy-and-hold per cohort, entry and
exit at close, 126-session hold, 0.5% round trip charged at entry, delisted names frozen at last
close, empty cohort = cash. Daily mark-to-market NAV. Benchmark: equal-weight core band, daily.
Sensitivity (not the registered result): same NAV with single-day stock returns clipped to +-40%.
Output: logs/leader_sleeve/portfolio_7x_<date>.log (stdout) + portfolio_7x_nav.parquet (+manifest).
"""
from __future__ import annotations

import html
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402

import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--hold", type=int, default=126)
HOLD = _ap.parse_args().hold
SLOTS = max(1, round(HOLD / 5))                  # weekly ladder: one slot per week of the hold
COST, START = 0.005, "2016-06-01"
ERA_SPLIT = pd.Timestamp("2023-01-01")

# ---------------- data ----------------
sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
imap = sc.set_index("symbol")["industry"].map(html.unescape)
al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
imap = pd.concat([imap, al[~al["symbol"].isin(imap.index)].set_index("symbol")["analog_symbol"].map(imap).dropna()])

px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                     columns=["symbol", "trade_date", "close", "sma_50", "avg_traded_value_20d"])
px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])].copy()
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60, fill_method=None)
px["ret252"] = g["close"].pct_change(252, fill_method=None)
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["ind"] = px["symbol"].map(imap)
mc = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"])
mc["trade_date"] = pd.to_datetime(mc["trade_date"])
px = px.merge(mc, on=["symbol", "trade_date"], how="left")
px["core"] = (px["adv"] >= 5) & (px["close"] > 50)
px["broad"] = px["mcap_cr"] >= 50

# daily returns matrix (close-to-close); NaN after delisting -> 0 (value frozen)
R = px.pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
days = R.index
dpos = {d: i for i, d in enumerate(days)}

# breadth from the panel, prior session
core_rows = px[px["core"] & px["sma_50"].notna()]
breadth = core_rows.assign(up=core_rows["close"] > core_rows["sma_50"]).groupby("trade_date")["up"].mean()
breadth_prev = breadth.shift(1)

weekly = [d for d in days[::5] if d >= pd.Timestamp(START)]


def select(heat_col: str, top_n: int) -> dict:
    """cohort date -> list of symbols to buy (core band only)."""
    W = px[px["trade_date"].isin(set(weekly)) & px[heat_col] & px["ind"].notna() & px["ret60"].notna()].copy()
    W["nm"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size")
    W = W[W["nm"] >= 5]
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("heat").reset_index()
    h["hot"] = h.groupby("trade_date")["heat"].rank(pct=True) >= 0.9
    W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    W["rk"] = W.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    P = W[W["hot"] & (W["rk"] <= top_n) & (W["ret252"] > 0.50) & W["core"]]
    return P.groupby("trade_date")["symbol"].apply(list).to_dict()


def run(picks: dict, gate: bool, Rm: pd.DataFrame) -> tuple[pd.Series, dict]:
    nav = pd.Series(0.0, index=days)
    start_i = dpos[weekly[0]]
    names_per, invested = [], 0
    for s in range(SLOTS):
        cap = 1.0 / SLOTS
        val = pd.Series(np.nan, index=days)
        val.iloc[start_i:] = cap                               # idle cash until its first cohort
        for k in range(s, len(weekly), SLOTS):
            d = weekly[k]; i0 = dpos[d]
            names = picks.get(d, [])
            if gate and not (breadth_prev.get(d, np.nan) >= 0.50):
                names = []
            end = min(i0 + HOLD, len(days) - 1)
            nxt = dpos[weekly[k + SLOTS]] if k + SLOTS < len(weekly) else len(days) - 1
            if names:
                invested += 1; names_per.append(len(names))
                path = (1 + Rm.iloc[i0 + 1:end + 1][names].fillna(0)).cumprod().mean(axis=1) * cap * (1 - COST)
                val.iloc[i0 + 1:end + 1] = path.values
                cap = float(path.iloc[-1]) if len(path) else cap * (1 - COST)
            else:
                val.iloc[i0 + 1:end + 1] = cap                  # empty / gated cohort: slot holds CASH
            val.iloc[end + 1:nxt + 1] = cap                     # cash between exit and next cohort
        nav = nav.add(val.fillna(0), fill_value=0)
    nav = nav.iloc[start_i:]
    return nav, dict(avg_names=float(np.mean(names_per)) if names_per else 0.0, pct_invested=invested / len(weekly) * 100)


def metrics(nav: pd.Series) -> dict:
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan

    def era(a, b):
        x = nav[(nav.index >= a) & (nav.index < b)]
        y = (x.index[-1] - x.index[0]).days / 365.25
        return (x.iloc[-1] / x.iloc[0]) ** (1 / y) - 1

    cal = nav.groupby(nav.index.year).agg(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    return dict(cagr=cagr * 100, maxdd=dd * 100, sharpe=sharpe, cagr_disc=era(nav.index[0], ERA_SPLIT) * 100,
                cagr_conf=era(ERA_SPLIT, nav.index[-1] + pd.Timedelta(days=1)) * 100, final=nav.iloc[-1] / nav.iloc[0],
                years={int(k): round(v * 100, 1) for k, v in cal.items()})


EXPS = {"BASELINE": ""}
for k in range(1, 4):
    for combo in itertools.combinations("GBH", k):
        EXPS["+".join(combo)] = "".join(combo)

pick_cache = {}
out_rows, navs = [], {}
Rclip = R.clip(-0.4, 0.4)
print(f"HOLD {HOLD} sessions · {SLOTS} weekly slots · panel {days[0].date()}..{days[-1].date()} · weekly cohorts {len(weekly)} from {weekly[0].date()} · "
      f"breadth>=0.50 on {(breadth_prev.reindex(weekly) >= 0.5).mean()*100:.0f}% of cohort dates\n", flush=True)
for name, lev in EXPS.items():
    key = ("broad" if "H" in lev else "core", 10 if "B" in lev else 3)
    if key not in pick_cache:
        pick_cache[key] = select(*key)
    nav, info = run(pick_cache[key], "G" in lev, R)
    navc, _ = run(pick_cache[key], "G" in lev, Rclip)
    m = metrics(nav); mc_ = metrics(navc)
    navs[name] = nav
    out_rows.append(dict(exp=name, **{k: v for k, v in m.items() if k != "years"}, clip_cagr=mc_["cagr"], clip_dd=mc_["maxdd"], **info, years=m["years"]))
    print(f"{name:<9} CAGR {m['cagr']:>+6.1f}% (disc {m['cagr_disc']:>+6.1f} / conf {m['cagr_conf']:>+6.1f}) · maxDD {m['maxdd']:>6.1f}% · "
          f"Sharpe {m['sharpe']:.2f} · x{m['final']:.1f} · names/cohort {info['avg_names']:.1f} · invested {info['pct_invested']:.0f}% "
          f"| clip±40%: CAGR {mc_['cagr']:+.1f} DD {mc_['maxdd']:.1f}", flush=True)

bench = px[px["core"]].pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
bench = bench[bench.index >= weekly[0]].mean(axis=1).fillna(0)
bnav = (1 + bench).cumprod()
bm = metrics(bnav)
print(f"{'BENCH EW':<9} CAGR {bm['cagr']:>+6.1f}% (disc {bm['cagr_disc']:>+6.1f} / conf {bm['cagr_conf']:>+6.1f}) · maxDD {bm['maxdd']:>6.1f}% · Sharpe {bm['sharpe']:.2f}")

print("\nCALENDAR-YEAR RETURNS (%)")
yrs = sorted(out_rows[0]["years"])
print(f"{'':<9}" + "".join(f"{y:>7}" for y in yrs))
for r in out_rows:
    print(f"{r['exp']:<9}" + "".join(f"{r['years'].get(y, np.nan):>7.1f}" for y in yrs))
print(f"{'BENCH EW':<9}" + "".join(f"{bm['years'].get(y, np.nan):>7.1f}" for y in yrs))

base = out_rows[0]
print("\nVERDICT vs BASELINE (registered: CAGR higher in BOTH eras AND maxDD not worse by >2pp)")
for r in out_rows[1:]:
    ok = r["cagr_disc"] > base["cagr_disc"] and r["cagr_conf"] > base["cagr_conf"] and r["maxdd"] >= base["maxdd"] - 2
    print(f"  {r['exp']:<7} {'BEATS' if ok else 'no   '} · disc {r['cagr_disc']-base['cagr_disc']:+.1f}pp · conf {r['cagr_conf']-base['cagr_conf']:+.1f}pp · "
          f"maxDD {r['maxdd']-base['maxdd']:+.1f}pp", flush=True)

out = ROOT / f"logs/leader_sleeve/portfolio_7x_nav{'' if HOLD == 126 else f'_h{HOLD}'}.parquet"
pd.DataFrame(navs).assign(BENCH_EW=bnav.reindex(next(iter(navs.values())).index)).to_parquet(out)
out.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
    dataset="leader sleeve 7x factorial NAVs", experiment="EXP-2026-09-27-leader-7x-factorial", producer="src/agentic/sim_leader_portfolio_7x.py",
    columns={"<exp>": "daily NAV, starts at 1.0 on the first cohort date (G=market gate, B=top-10 basket, H=broad heat)", "BENCH_EW": "equal-weight core band, daily rebalanced"},
    summary=[{k: v for k, v in r.items() if k != "years"} for r in out_rows], updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=float))
print("\nPORTFOLIO 7X COMPLETE")
