"""Point-in-time industry scores for EXP-2026-09-29-industry-fundamentals (registered in logs/experiments.jsonl first).

For every session from 2019-01-01 and every industry with >= 5 core-band names (20d ADV >= Rs5cr, then-traded close
> Rs50; nse4 map + analyst analogs, as the leader screen):
  price heat   mean / median own 60-session return of the core names, breadth = share with ret60 > 0; percentile ranks
               heat_pct (mean, the production definition) and heatfix_pct (median)
  A growth     medians over the core names of: latest-quarter revenue YoY, its acceleration (latest YoY minus the prior
               quarter's YoY), latest-quarter PAT change vs the same quarter last year / max(|PAT a year ago|, 1% of
               revenue) clipped to [-2, 2]; point in time by pnl_quarterly filing_dt (as filed first; one basis per
               symbol as build_ttm; a stale late filing never replaces a newer quarter); >= 3 names with data;
               A = mean of the available cross-industry percentile ranks
  B demand     order filings (order_fulltext + order_daily; order / tender_L1 with an amount) by the core names with
               effective session in (d-90d, d], summed / summed PIT TTM revenue of the core names; change = that level
               minus the level for (d-180d, d-90d]; NaN when no filing in (d-180d, d]; B = mean percentile rank
  C macro      36-month rolling univariate regressions of the industry's monthly equal-weight return (core members at
               the prior close) on each driver's monthly change (log; US 10y in points), drivers from
               macro_drivers_fred.parquet; |t| > 2 kept; tailwind = sum(beta x driver change over the last 3 known
               months). At a date in month m: industry and daily drivers use months <= m-1; IMF monthly drivers
               (known a month late) use months <= m-2. C = percentile rank of the tailwind
  D flows      median change in FII+DII holding (percentage points) at the latest quarter filed by d (filing_date, else
               quarter_end + 21 days); REPORTED ONLY (sparse before 2022)
  F            mean of the available A, B, C percentile ranks (>= 2 present); F_pct = its percentile rank that day
Output: data/derived/industry_scores.parquet (+ manifest). Units: returns, growth and ratios are fractions (0.1 = 10%).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
from test_hot_order_combo import CUTOFF_MIN, PNL, PNL_UNIT_TO_CR, asof_ttm, build_ttm, effective_session  # noqa: E402

OUT = ROOT / "data/derived/industry_scores.parquet"
FRED = ROOT / "data/derived/macro_drivers_fred.parquet"
SHP = ROOT / "data/shareholding_full_history/normalized/stock_shareholding_quarterly.parquet"
SHP_INCR = ROOT / "data/shareholding_full_history/_incremental/normalized/stock_shareholding_quarterly.parquet"
START, MIN_N, MIN_DATA, WIN_M, T_MIN = "2019-01-01", 5, 3, 36, 2.0
GROWTH_TOL = pd.Timedelta(days=200)


def pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def growth_rows() -> pd.DataFrame:
    q = pd.read_parquet(PNL, columns=["symbol", "quarter_end", "filing_dt", "basis", "source", "net_sales", "pat"])
    q = q.dropna(subset=["filing_dt", "net_sales"])
    unit = q["source"].map(PNL_UNIT_TO_CR)
    q["sales"], q["pat_cr"] = q["net_sales"] * unit, q["pat"] * unit
    q = q.sort_values("filing_dt", kind="mergesort").drop_duplicates(["symbol", "quarter_end", "basis"], keep="first")
    n = q.groupby(["symbol", "basis"]).size().unstack(fill_value=0).reindex(columns=["con", "sa"], fill_value=0)
    keep = pd.DataFrame({"symbol": n.index, "basis": np.where(n["con"] >= n["sa"], "con", "sa")})
    q = q.merge(keep, on=["symbol", "basis"])
    q["qn"] = q["quarter_end"].dt.year * 4 + q["quarter_end"].dt.quarter
    k = q[["symbol", "qn", "sales", "pat_cr"]]
    q = q.merge(k.assign(qn=k["qn"] + 4).rename(columns={"sales": "sales_4", "pat_cr": "pat_4"}), on=["symbol", "qn"], how="left")
    with np.errstate(invalid="ignore", divide="ignore"):
        q["rev_yoy"] = np.where((q["sales"] > 0) & (q["sales_4"] > 0), q["sales"] / q["sales_4"] - 1, np.nan)
        den = np.maximum(q["pat_4"].abs(), 0.01 * q["sales"].abs())
        q["pat_chg"] = ((q["pat_cr"] - q["pat_4"]) / den).clip(-2, 2)
    q = q.merge(q[["symbol", "qn", "rev_yoy"]].assign(qn=q["qn"] + 1).rename(columns={"rev_yoy": "rev_yoy_prev"}),
                on=["symbol", "qn"], how="left")
    q["accel"] = q["rev_yoy"] - q["rev_yoy_prev"]
    q = q.sort_values(["symbol", "filing_dt", "quarter_end"], kind="mergesort")
    q = q[q["quarter_end"] >= q.groupby("symbol")["quarter_end"].cummax()]         # stale late filings never win
    q["avail"] = pd.to_datetime(q["filing_dt"]).astype("datetime64[ns]")
    return q[["symbol", "avail", "rev_yoy", "accel", "pat_chg"]].sort_values("avail")


def order_events(cal: pd.DatetimeIndex) -> pd.DataFrame:
    a = pd.read_parquet(ROOT / "data/derived/order_fulltext.parquet", columns=["symbol", "cat", "ts", "amount_cr"])
    a = a[a["cat"].isin(["order", "tender_L1"])]
    parts = [a[["symbol", "ts", "amount_cr"]]]
    daily = ROOT / "data/derived/order_daily.parquet"
    if daily.exists():
        b = pd.read_parquet(daily, columns=["symbol", "ts", "amount_cr"])
        parts.append(b[pd.to_datetime(b["ts"]) > pd.to_datetime(a["ts"]).max()])
    o = pd.concat(parts, ignore_index=True).dropna(subset=["amount_cr"])
    o["ts"] = pd.to_datetime(o["ts"])
    o["eff"] = effective_session(o["ts"], cal)
    return o.dropna(subset=["eff"])


def window_sums(frame: pd.DataFrame, o: pd.DataFrame) -> pd.DataFrame:
    """orders90 / orders_prev for every (date, symbol) in frame, from per-symbol cumulative sums."""
    og = {s: g for s, g in o.sort_values(["symbol", "eff"]).groupby("symbol")}
    out90 = np.zeros(len(frame)); outp = np.zeros(len(frame))
    d = frame["date"].values.astype("datetime64[ns]")
    for sym, idx in frame.groupby("symbol").indices.items():
        g = og.get(sym)
        if g is None:
            continue
        e = g["eff"].values.astype("datetime64[ns]"); cs = np.concatenate([[0.0], np.cumsum(g["amount_cr"].values)])
        di = d[idx]
        at = lambda t: cs[np.searchsorted(e, t, side="right")]  # noqa: E731  sum of amounts with eff <= t
        out90[idx] = at(di) - at(di - np.timedelta64(90, "D"))
        outp[idx] = at(di - np.timedelta64(90, "D")) - at(di - np.timedelta64(180, "D"))
    return frame.assign(orders90=out90, orders_prev=outp)


def macro_tailwind(D: dict, imap: pd.Series) -> pd.DataFrame:
    """(month, industry) -> tailwind known during the NEXT month (industry/daily drivers <= month, IMF <= month-1)."""
    cal, syms = D["cal"], D["syms"]
    R = pd.DataFrame(D["Rv"], index=cal, columns=syms)
    px = D["px"]
    core = px.pivot(index="trade_date", columns="symbol", values="core").reindex(index=cal, columns=syms)
    core = core.astype(float).shift(1).fillna(0).astype(bool)                          # members at the prior close
    ind_of = imap.reindex(syms)
    ret = {}
    for ind, cols in ind_of.groupby(ind_of).groups.items():
        m = core[list(cols)].to_numpy(); r = R[list(cols)].to_numpy()
        with np.errstate(invalid="ignore"):
            v = np.where(m, r, np.nan)
            cnt = np.isfinite(v).sum(axis=1)
            ret[ind] = pd.Series(np.where(cnt >= MIN_N, np.nanmean(np.where(np.isfinite(v), v, np.nan), axis=1), np.nan), index=cal)
    I = pd.DataFrame(ret)
    Im = (1 + I).groupby(I.index.to_period("M")).prod(min_count=10) - 1
    Im[I.groupby(I.index.to_period("M")).count() < 10] = np.nan
    F = pd.read_parquet(FRED); F["date"] = pd.to_datetime(F["date"])
    chg, lag = {}, {}
    for name, g in F.groupby("series"):
        s = g.set_index("date")["value"].sort_index()
        if g["freq"].iloc[0] == "D":
            s = s.groupby(s.index.to_period("M")).last()
            chg[name] = s.diff() if name == "us_10y" else np.log(s).diff()
            lag[name] = 0
        else:
            s.index = s.index.to_period("M")
            chg[name] = np.log(s).diff(); lag[name] = 1
    months = Im.index
    tail = pd.DataFrame(0.0, index=months, columns=Im.columns)
    used = pd.DataFrame(0, index=months, columns=Im.columns)
    for name, x in chg.items():
        x = x.reindex(months)
        L = lag[name]
        y = Im.shift(L)                 # IMF drivers: pair industry month m-1 with driver month m-1, known at month m
        xs = x.shift(L)
        cov = y.rolling(WIN_M, min_periods=24).cov(xs)
        var = xs.rolling(WIN_M, min_periods=24).var()
        corr = y.rolling(WIN_M, min_periods=24).corr(xs)
        n = y.notna().astype(float).mul(xs.notna().astype(float), axis=0).rolling(WIN_M, min_periods=1).sum()
        beta = cov.div(var, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = corr * np.sqrt((n - 2) / (1 - corr ** 2))
        c3 = xs.rolling(3).sum()
        contrib = beta.mul(c3, axis=0).where(t.abs() > T_MIN)
        tail = tail.add(contrib.fillna(0)); used = used.add(contrib.notna().astype(int))
    tail = tail.where(Im.notna().rolling(24, min_periods=1).sum() >= 24)               # need 24 months of industry history
    for X_ in (tail, used):
        X_.index.name, X_.columns.name = "month", "industry"
    out = tail.stack().rename("C_tailwind").reset_index()
    u = used.stack().rename("C_drivers").reset_index()
    out = out.merge(u, on=["month", "industry"], how="left")
    out["apply_month"] = out["month"] + 1                                                 # known during the next month
    return out


def flows_rows() -> pd.DataFrame:
    parts = [pd.read_parquet(p) for p in (SHP, SHP_INCR) if p.exists()]
    s = pd.concat(parts, ignore_index=True)
    s["quarter_end"] = pd.to_datetime(s["quarter_end"])
    s = s.drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
    s["inst"] = s["fii_fpi_pct"] + s["dii_pct"]
    s["inst_chg"] = s.groupby("symbol")["inst"].diff()
    s["avail"] = pd.to_datetime(s["filing_date"], errors="coerce").fillna(s["quarter_end"] + pd.Timedelta(days=21))
    return s.dropna(subset=["inst_chg"])[["symbol", "avail", "inst_chg"]].assign(avail=lambda x: x["avail"].astype("datetime64[ns]")).sort_values("avail")


def asof(frame: pd.DataFrame, rows: pd.DataFrame, cols: list[str], tol: pd.Timedelta | None) -> pd.DataFrame:
    L = frame[["symbol", "asof"]].reset_index().sort_values("asof")
    m = pd.merge_asof(L, rows, left_on="asof", right_on="avail", by="symbol", direction="backward", tolerance=tol)
    return m.set_index("index")[cols].reindex(frame.index)


def main() -> None:
    t0 = time.time()
    D = sp.load(None)
    imap = sp.industry_maps()["analogs"]
    cal = D["cal"]
    px = D["px"]
    fr = px[(px["trade_date"] >= pd.Timestamp(START)) & px["core"]][["symbol", "trade_date", "ret60"]].copy()
    fr = fr.rename(columns={"trade_date": "date"})
    fr["industry"] = fr["symbol"].map(imap)
    fr = fr[fr["industry"].notna() & fr["ret60"].notna()].reset_index(drop=True)
    fr["pos"] = (fr["ret60"] > 0).astype(float)
    fr["asof"] = (fr["date"] + pd.Timedelta(minutes=CUTOFF_MIN - 1)).astype("datetime64[ns]")
    print(f"frame {len(fr):,} (date, core symbol) rows · {time.time() - t0:.0f}s", flush=True)

    G = growth_rows()
    fr[["rev_yoy", "accel", "pat_chg"]] = asof(fr, G, ["rev_yoy", "accel", "pat_chg"], GROWTH_TOL)
    T, _ = build_ttm(None)
    fr["rev_ttm_cr"] = asof_ttm(fr, "asof", T)
    O = order_events(cal)
    fr = window_sums(fr, O)
    Fl = flows_rows()
    fr["inst_chg"] = asof(fr, Fl, ["inst_chg"], pd.Timedelta(days=200))
    print(f"symbol pillars done · {time.time() - t0:.0f}s", flush=True)

    g = fr.groupby(["date", "industry"])
    agg = g.agg(n_core=("symbol", "size"), heat_mean=("ret60", "mean"), heat_median=("ret60", "median"),
                breadth=("pos", "mean"),
                n_growth=("rev_yoy", "count"), A_rev_yoy=("rev_yoy", "median"), A_accel=("accel", "median"),
                A_pat=("pat_chg", "median"), orders90=("orders90", "sum"), orders_prev=("orders_prev", "sum"),
                rev_ttm_sum=("rev_ttm_cr", "sum"), n_flows=("inst_chg", "count"), D_flows=("inst_chg", "median")).reset_index()
    agg = agg[agg["n_core"] >= MIN_N].copy()
    for c in ("A_rev_yoy", "A_accel", "A_pat"):
        agg.loc[agg["n_growth"] < MIN_DATA, c] = np.nan
    agg.loc[agg["n_flows"] < MIN_DATA, "D_flows"] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        agg["B_level"] = agg["orders90"] / agg["rev_ttm_sum"]
        agg["B_change"] = agg["B_level"] - agg["orders_prev"] / agg["rev_ttm_sum"]
    none = (agg["orders90"] + agg["orders_prev"]) <= 0
    agg.loc[none | (agg["rev_ttm_sum"] <= 0), ["B_level", "B_change"]] = np.nan

    C = macro_tailwind(D, imap)
    agg["apply_month"] = agg["date"].dt.to_period("M")
    agg = agg.merge(C[["apply_month", "industry", "C_tailwind", "C_drivers"]], on=["apply_month", "industry"], how="left")

    agg["heat_pct"] = pct(agg, "heat_mean"); agg["heatfix_pct"] = pct(agg, "heat_median")
    comp = {"A": ["A_rev_yoy", "A_accel", "A_pat"], "B": ["B_level", "B_change"], "C": ["C_tailwind"]}
    for p, cols in comp.items():
        agg[p] = pd.concat([pct(agg, c) for c in cols], axis=1).mean(axis=1)
    agg["D_pct"] = pct(agg, "D_flows")
    k = agg[["A", "B", "C"]].notna().sum(axis=1)
    agg["F"] = agg[["A", "B", "C"]].mean(axis=1).where(k >= 2)
    agg["F_pct"] = pct(agg, "F")
    agg = agg.drop(columns=["apply_month"])
    agg.to_parquet(OUT, index=False)
    cov = agg.groupby(agg["date"].dt.year)[["A", "B", "C", "D_flows", "F"]].apply(lambda x: (x.notna().mean() * 100).round(0))
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="industry_scores", path=str(OUT.relative_to(ROOT)), rows=len(agg), key=["date", "industry"],
        producer="src/agentic/build_industry_scores.py", experiment="EXP-2026-09-29-industry-fundamentals",
        definitions=__doc__, coverage_pct_by_year=cov.to_dict(orient="index"),
        units=dict(heat_mean="fraction (mean own 60-session return)", A_rev_yoy="fraction", A_accel="fraction (difference of two YoY fractions)",
                   A_pat="ratio clipped to [-2, 2]", B_level="order amount / TTM revenue over 90 days, fraction", C_tailwind="sum of beta x 3-month driver change (return units)",
                   D_flows="percentage points of FII+DII holding", pct="percentile rank across industries on that date (0-1]"),
        inputs=["data/derived/stock_daily_facts_adjusted_2015plus.parquet", str(PNL.relative_to(ROOT)), "data/derived/order_fulltext.parquet",
                "data/derived/order_daily.parquet", str(FRED.relative_to(ROOT)), str(SHP.relative_to(ROOT))],
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(agg):,} rows, {agg['industry'].nunique()} industries · {time.time() - t0:.0f}s")
    print("pillar coverage by year (% of industry-days):\n" + cov.to_string())


if __name__ == "__main__":
    main()
