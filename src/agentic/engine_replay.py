"""Walk-forward replay of the five engines' top-30 sets (EXP-2026-09-24-engines-count-sizing).

The live engines take no as-of date: each fits on the full panel and scores only the
latest day. To know which names were in each engine's top-30 on a past entry day
without peeking, this script refits every engine ONCE PER SCORING YEAR Y (2020..2026)
on rows whose label window closed before Y's first entry day (embargo: td_idx + H <
idx(d0)), then scores every weekly entry day inside Y.

Fidelity to the live engines (same panel code, features, targets, hyper-parameters —
the panel builders are imported, not copied):
  cs   compare_short_horizons  LGBM400+XGB400 on target 5pct_15d, extras joined; rank raw
  hc   find_high_conviction    3 targets; isotonic per target on OOF years Y-2,Y-1 (live:
                               2024-25); final LGBM600+XGB600; rank max calibrated score
  mb   find_multibagger_today  LGBM300 on 50pct_180d, train <= d0 - 240 calendar days; rank raw
  mh   run_multi_horizon       h1/h7/h21 isotonic on OOF years, train years before them
                               (live: train 2020-23, OOF 2024-25); rank geo-mean consensus
  180d find_180d_frontier_honest LGBM300 on fwd +15%/180d; rank raw
Where live ranks by an isotonic transform of ONE score (cs, mb, 180d) the raw score is
used: isotonic is monotone, so the order is the same except inside isotonic plateaus,
where live order is arbitrary anyway.

Known differences from live (documented, not hidden):
  - refit yearly, live refits every run: a December score uses a model up to 1 year old
  - labels embargoed at the scoring day; live trains on every complete label
  - extras (cs, hc) exist only from 2023-06-01 and are median-filled elsewhere, exactly as
    live does; the median is taken over the whole panel (live does the same)
  - mh's sector map is today's index membership (not point-in-time) and catalyst_features
    ends 2026-06-01 (0-filled after), both as live

Usage:  /usr/bin/python3 src/agentic/engine_replay.py --engine cs|hc|mb|mh|180d
Output: data/derived/engine_replay/top30_<engine>.parquet  (+ .manifest.json)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
OUT_DIR = ROOT / "data/derived/engine_replay"
YEARS = list(range(2020, 2027))
TOP_N = 30


def lgbm(n, lr=0.05):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=64, min_child_samples=200,
                              feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
                              random_state=42, verbose=-1, n_jobs=-1)


def xgbm(n, lr=0.05):
    return xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=7, subsample=0.85,
                             colsample_bytree=0.85, random_state=42, verbosity=0, n_jobs=-1,
                             tree_method="hist", eval_metric="logloss")


def fit_pair(tr, feats, y, n=400, lr=0.05):
    a, b = lgbm(n, lr), xgbm(n, lr)
    a.fit(tr[feats], tr[y].astype(int)); b.fit(tr[feats], tr[y].astype(int))
    return lambda X: 0.5 * a.predict_proba(X[feats])[:, 1] + 0.5 * b.predict_proba(X[feats])[:, 1]


def fit_lgb(tr, feats, y, n=300):
    a = lgbm(n); a.fit(tr[feats], tr[y].astype(int))
    return lambda X: a.predict_proba(X[feats])[:, 1]


def entry_days(dates: pd.Series) -> list[pd.Timestamp]:
    """First trading day on/after each Monday — the same snapping as backtest_10yr_15d5pct."""
    td = np.array(sorted(pd.to_datetime(dates.unique())))
    out = []
    for w in pd.date_range("2016-01-04", pd.Timestamp(td[-1]), freq="W-MON"):
        i = np.searchsorted(td, np.datetime64(w))
        if i < len(td):
            out.append(pd.Timestamp(td[i]))
    return sorted(set(out))


def add_td_idx(df):
    cal = {d: i for i, d in enumerate(sorted(df["trade_date"].unique()))}
    df["td_idx"] = df["trade_date"].map(cal).astype(int)
    return cal


def year_plan(days):
    """{Y: (d0, [entry days in Y])}"""
    plan = {}
    for Y in YEARS:
        ds = [d for d in days if d.year == Y]
        if ds:
            plan[Y] = (ds[0], ds)
    return plan


def top_n(scored: pd.DataFrame, col: str, engine: str, tiebreak: str | None = None) -> pd.DataFrame:
    keys = [col] + ([tiebreak] if tiebreak else []) + ["symbol"]
    s = scored.sort_values(["trade_date"] + keys, ascending=[True] + [False] * (len(keys) - 1) + [True])
    s = s.groupby("trade_date").head(TOP_N).copy()
    s["rank"] = s.groupby("trade_date").cumcount() + 1
    s["engine"] = engine
    return s[["engine", "trade_date", "symbol", "rank", col]].rename(columns={col: "score"})


def replay_cs():
    import compare_short_horizons as E
    df = E.build_panel(); df, feats = E.join_extras(df); df = df.dropna(subset=feats).copy()
    cal = add_td_idx(df); y, H = "target_5pct_15d", 15
    lab = df[df[y].notna() & (df["adv_20d_cr"] >= 1.0)]
    out = []
    for Y, (d0, ds) in year_plan(entry_days(df["trade_date"])).items():
        pred = fit_pair(lab[lab["td_idx"] + H < cal[d0]], feats, y, 400)
        sc = df[df["trade_date"].isin(ds) & (df["adv_20d_cr"] >= 1.0)].copy()
        sc["raw"] = pred(sc); out.append(top_n(sc, "raw", "cs")); print(f"  cs {Y}: {len(ds)} days", flush=True)
    return pd.concat(out)


def replay_hc():
    import find_high_conviction as E
    df, feats = E.build_panel_with_extras(); df = df.dropna(subset=E.BASE_FEATS).copy()
    cal = add_td_idx(df)
    plan = year_plan(entry_days(df["trade_date"]))
    first_day = {y: cal[df.loc[df["year"] == y, "trade_date"].min()] for y in range(2016, 2027)}
    scored = {Y: df[df["trade_date"].isin(ds) & (df["adv_20d_cr"] >= 1.0)].copy() for Y, (d0, ds) in plan.items()}
    cal_cols = []
    for t in E.TARGETS:
        y, H = f"target_{t['name']}", t["horizon"]
        lab = df[df[y].notna() & (df["adv_20d_cr"] >= 1.0)]
        oof = {}
        for yr in range(min(YEARS) - 2, max(YEARS)):          # OOF model per year, cached across Y
            tr = lab[(lab["year"] < yr) & (lab["td_idx"] + H < first_day[yr])]
            te = lab[lab["year"] == yr].copy()
            te["raw"] = fit_pair(tr, feats, y, 400)(te); oof[yr] = te[["td_idx", "year", y, "raw"]]
        for Y, (d0, ds) in plan.items():
            o = pd.concat([oof[Y - 2], oof[Y - 1]]); o = o[o["td_idx"] + H < cal[d0]]
            iso = IsotonicRegression(out_of_bounds="clip").fit(o["raw"], o[y].astype(int))
            pred = fit_pair(lab[lab["td_idx"] + H < cal[d0]], feats, y, 600, 0.04)
            raw = pred(scored[Y]); scored[Y][f"raw_{t['name']}"] = raw; scored[Y][f"cal_{t['name']}"] = iso.transform(raw)
            print(f"  hc {t['name']} {Y}", flush=True)
        cal_cols.append(f"cal_{t['name']}")
    out = []
    for Y, sc in scored.items():
        sc["best"] = sc[cal_cols].max(axis=1)
        sc["best_raw"] = sc[[c.replace("cal_", "raw_") for c in cal_cols]].max(axis=1)
        out.append(top_n(sc, "best", "hc", tiebreak="best_raw"))
    return pd.concat(out)


def replay_mb():
    import find_multibagger_today as E
    df = E.build_panel().dropna(subset=E.BASE_FEATS).copy(); cal = add_td_idx(df)
    df["target"] = E.build_target(df, 0.50, 180); H = 180
    lab = df[(df["target"] != -1) & (df["adv_20d_cr"] >= 1.0)]
    out = []
    for Y, (d0, ds) in year_plan(entry_days(df["trade_date"])).items():
        tr = lab[(lab["td_idx"] + H < cal[d0]) & (lab["trade_date"] <= d0 - pd.Timedelta(days=H + 60))]
        pred = fit_lgb(tr, E.BASE_FEATS, "target", 300)
        sc = df[df["trade_date"].isin(ds) & (df["adv_20d_cr"] >= 1.0)].copy()
        sc["raw"] = pred(sc); out.append(top_n(sc, "raw", "mb")); print(f"  mb {Y}", flush=True)
    return pd.concat(out)


def replay_180d():
    import find_180d_frontier_honest as E
    df = E.build_panel().dropna(subset=E.FEATS).copy(); cal = add_td_idx(df); H = E.HORIZON
    df["target_15"] = (df["fwd_pct"] >= 0.15).astype("Int64")
    lab = df[df["target_15"].notna() & (df["adv_20d_cr"] >= 1.0)]
    out = []
    for Y, (d0, ds) in year_plan(entry_days(df["trade_date"])).items():
        pred = fit_lgb(lab[lab["td_idx"] + H < cal[d0]], E.FEATS, "target_15", 300)
        sc = df[df["trade_date"].isin(ds) & (df["adv_20d_cr"] >= 1.0)].copy()
        sc["raw"] = pred(sc); out.append(top_n(sc, "raw", "180d")); print(f"  180d {Y}", flush=True)
    return pd.concat(out)


def replay_mh():
    os.chdir(ROOT)                                   # run_multi_horizon reads relative paths
    import run_multi_horizon as E
    df, feats = E.build_panel(); cal = add_td_idx(df)
    plan = year_plan(entry_days(df["trade_date"]))
    first_day = {y: cal[df.loc[df["year"] == y, "trade_date"].min()] for y in range(2018, 2027)}
    scored = {Y: df[df["trade_date"].isin(ds) & (df["adv_20d_cr"] >= 1.0)].copy() for Y, (d0, ds) in plan.items()}
    for H, _ in E.HORIZONS:
        y = f"label_h{H}"; lab = df[df[y].notna()]
        for Y, (d0, ds) in plan.items():
            oof_yrs = [v for v in (Y - 2, Y - 1) if v >= 2019]
            train_yrs = list(range(max(2018, Y - 6), min(oof_yrs)))
            tr = lab[lab["year"].isin(train_yrs) & (lab["td_idx"] + H < first_day[min(oof_yrs)])]
            o = lab[lab["year"].isin(oof_yrs) & (lab["td_idx"] + H < cal[d0])].copy()
            o["raw"] = fit_pair(tr, feats, y, 400)(o)
            iso = IsotonicRegression(out_of_bounds="clip").fit(o["raw"], o[y].astype(int))
            raw = fit_pair(lab[lab["td_idx"] + H < cal[d0]], feats, y, 400)(scored[Y])
            scored[Y][f"cal_h{H}"] = iso.transform(raw)
            print(f"  mh h{H} {Y} (train {train_yrs}, oof {oof_yrs})", flush=True)
    out = []
    for Y, sc in scored.items():
        sc["consensus"] = (sc["cal_h1"] * sc["cal_h7"] * sc["cal_h21"]) ** (1 / 3)
        out.append(top_n(sc, "consensus", "mh"))
    return pd.concat(out)


ENGINES = {"cs": replay_cs, "hc": replay_hc, "mb": replay_mb, "mh": replay_mh, "180d": replay_180d}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True, choices=ENGINES)
    eng = ap.parse_args().engine
    t0 = time.time(); print(f"== engine_replay {eng} ==", flush=True)
    res = ENGINES[eng]()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"top30_{eng}.parquet"; res.to_parquet(out, index=False)
    (OUT_DIR / f"top30_{eng}.manifest.json").write_text(json.dumps({
        "file": out.name, "producer": "src/agentic/engine_replay.py", "engine": eng,
        "experiment": "EXP-2026-09-24-engines-count-sizing",
        "grain": "one row per (entry trade_date, symbol) in the engine's top-30 that day",
        "columns": {"rank": "1..30 within the day", "score": "the engine's ranking score (raw prob for cs/mb/180d, max isotonic prob for hc, geo-mean consensus for mh)"},
        "entry_days": "first trading day on/after each Monday", "years": [min(YEARS), max(YEARS)],
        "refit": "once per scoring year; training rows need td_idx + horizon < idx(first entry day of the year)",
        "date_range": [str(res["trade_date"].min().date()), str(res["trade_date"].max().date())],
        "n_rows": int(len(res)), "n_days": int(res["trade_date"].nunique()),
        "runtime_s": round(time.time() - t0), "generated": pd.Timestamp.now().isoformat(timespec="seconds")}, indent=2))
    print(f"wrote {out.relative_to(ROOT)}  {len(res):,} rows  {res['trade_date'].nunique()} days  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
