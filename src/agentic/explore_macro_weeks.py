"""Exploration (not a registered test; changes no rule): do macro readings on the list date go with better or worse V3.1 lists?

List outcome = average 6-month return of a phase-0 weekly V3.1 list after liquidity-scaled costs (finished windows only,
logs/leader_sleeve/v31_hits_by_year/v31_picks.csv). Every macro reading is what was known on the list date:
  india_vix, vix_1m      India VIX close and its change over ~1 month (data/derived/india_vix.parquet)
  rupee_3m               USD/INR change over 3 months, + = rupee weaker (FRED DEXINUS, by known_from)
  us10y_3m               US 10-year yield change over 3 months, percentage points (FRED DGS10, by known_from)
  dollar_3m              broad dollar index change over 3 months (FRED DTWEXBGS, by known_from)
  brent_3m               Brent crude change over 3 months (FRED DCOILBRENTEU, by known_from)
  us_vix, spx_vs_200     US VIX and the S&P 500 vs its 200-day average (us_macro_fred.parquet, as of the day before)
  core_yoy, iip_yoy      latest released year-on-year growth of India's core sector index and IIP general index (by release date)
Reported per era (2019-22, 2023+): rank correlation with the list outcome and the average outcome in the low / high third.
Neighbouring weekly lists overlap (each 6-month window shares most of its days), so ~15 independent half-years in all.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/leader_sleeve/macro_weeks"


def asof(dates: pd.DatetimeIndex, s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    return pd.Series(s.reindex(s.index.union(dates)).ffill().reindex(dates).to_numpy(), index=dates)


def main() -> None:
    pk = pd.read_csv(ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv", parse_dates=["trade_date"])
    pk["cost"] = np.where(pk["adv"] >= 5, 0.005, np.where(pk["adv"] >= 1, 0.010, 0.020))
    f = pk[pk["full"]]
    L = f.groupby("trade_date").agg(ret=("ret", "mean"), cost=("cost", "mean"))
    L["list_ret"] = L["ret"] - L["cost"]
    d = L.index
    M = pd.DataFrame(index=d)
    v = pd.read_parquet(ROOT / "data/derived/india_vix.parquet"); v["date"] = pd.to_datetime(v["date"])
    vs = v.set_index("date")["vix_close"]
    M["india_vix"] = asof(d, vs); M["vix_1m"] = M["india_vix"] / asof(d - pd.Timedelta(days=30), vs).to_numpy() - 1
    fr = pd.read_parquet(ROOT / "data/derived/macro_drivers_fred.parquet"); fr["known_from"] = pd.to_datetime(fr["known_from"])
    for name, ser, kind in (("rupee_3m", "usdinr", "pct"), ("us10y_3m", "us_10y", "diff"), ("dollar_3m", "broad_dollar", "pct"), ("brent_3m", "brent", "pct")):
        s = fr[fr["series"] == ser].groupby("known_from")["value"].last()
        now, then = asof(d, s), asof(d - pd.DateOffset(months=3), s).to_numpy()
        M[name] = (now / then - 1) if kind == "pct" else (now - then)
    um = pd.read_parquet(ROOT / "data/derived/us_macro_fred.parquet"); um["trade_date"] = pd.to_datetime(um["trade_date"])
    um = um.set_index("trade_date").sort_index()
    M["us_vix"] = asof(d - pd.Timedelta(days=1), um["us_vix"]).to_numpy()
    spx = um["spx"].dropna(); spx200 = spx / spx.rolling(200).mean() - 1
    M["spx_vs_200"] = asof(d - pd.Timedelta(days=1), spx200).to_numpy()

    def released(df: pd.DataFrame) -> pd.Series:
        df = df.dropna(subset=["yoy_pct", "release_date"]).copy(); df["release_date"] = pd.to_datetime(df["release_date"]); df["month"] = pd.to_datetime(df["month"])
        out = []
        for day in d:
            k = df[df["release_date"] <= day]
            if k.empty:
                out.append(np.nan); continue
            k = k[k["month"] == k["month"].max()].sort_values("release_date")
            out.append(float(k["yoy_pct"].iloc[-1]))
        return pd.Series(out, index=d)
    cs = pd.read_parquet(ROOT / "data/derived/core_sector_monthly.parquet")
    M["core_yoy"] = released(cs[cs["series"] == "Overall Index"])
    ip = pd.read_parquet(ROOT / "data/derived/iip_monthly.parquet")
    gen = ip[ip["series"].str.contains("General", case=False, na=False)]
    M["iip_yoy"] = released(gen)

    X = L.join(M); X["era"] = np.where(X.index.year <= 2022, "2019-22", "2023+")
    labels = {"india_vix": "India VIX (fear) level", "vix_1m": "India VIX change, 1 month", "rupee_3m": "rupee weakening, 3 months",
              "us10y_3m": "US 10-year yield change, 3 months", "dollar_3m": "dollar index change, 3 months", "brent_3m": "crude oil change, 3 months",
              "us_vix": "US VIX level", "spx_vs_200": "S&P 500 vs its 200-day", "core_yoy": "India core sector growth (latest)", "iip_yoy": "India IIP growth (latest)"}
    rows = []
    for c, lab in labels.items():
        r = {"factor": lab, "known_share": round(float(X[c].notna().mean()), 2)}
        for era, g in X.groupby("era"):
            g = g.dropna(subset=[c, "list_ret"])
            if len(g) < 30:
                r[f"{era} corr"] = None; r[f"{era} low / high third"] = "n/a"; continue
            r[f"{era} corr"] = round(float(g[c].rank().corr(g["list_ret"].rank())), 2)
            q = pd.qcut(g[c].rank(method="first"), 3, labels=["low", "mid", "high"])
            m = g.groupby(q, observed=True)["list_ret"].mean()
            r[f"{era} low / high third"] = f"{100 * m['low']:+.0f}% / {100 * m['high']:+.0f}%"
        a, b = r.get("2019-22 corr"), r.get("2023+ corr")
        r["same direction, both >= 0.15"] = bool(a is not None and b is not None and np.sign(a) == np.sign(b) and min(abs(a), abs(b)) >= 0.15)
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f"weekly lists with a finished 6 months: {len(X)} ({X.index.min().date()}..{X.index.max().date()})")
    print(T.to_string(index=False))
    wk = pd.Timestamp("2024-09-11")
    pct = {labels[c]: round(float((X[c].dropna() <= X.loc[wk, c]).mean()), 2) for c in labels if pd.notna(X.loc[wk, c])}
    print(f"\nthe worst list (2024-09-11, {100 * X.loc[wk, 'list_ret']:+.0f}%): where each reading sat among all list weeks (0 = lowest, 1 = highest): {pct}")
    OUT.mkdir(parents=True, exist_ok=True)
    X.to_csv(OUT / "weekly_macro.csv")
    (OUT / "weekly_macro.csv.manifest.json").write_text(json.dumps(dict(dataset="weekly_macro.csv", producer="src/agentic/explore_macro_weeks.py", kind="exploration",
        definitions=__doc__, table=T.to_dict("records"), units=dict(list_ret="fraction, 6-month list return after costs", rupee_3m="fraction", us10y_3m="percentage points",
        dollar_3m="fraction", brent_3m="fraction", core_yoy="percent", iip_yoy="percent", india_vix="index points", vix_1m="fraction"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUT / "README.md").write_text("# Macro readings vs V3.1 weekly lists\n\nExploration. weekly_macro.csv = each weekly list's 6-month outcome with the macro readings known on its date; "
                                   "the summary table is in the manifest.\n")


if __name__ == "__main__":
    main()
