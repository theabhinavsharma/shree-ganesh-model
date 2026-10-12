"""Exploration (not a registered test; changes no rule): do foreign investors' (FPI) recent flows go with better or worse
V3.1 weekly lists?

Flows: data/derived/nsdl_fpi_daily.parquet (NSDL FPI Monitor, official; a reporting date counts from its known_from date).
Equity, stock-exchange route (secondary market buying minus selling), Rs crore. On each list date, using only rows known by then:
  fpi_20d / fpi_60d      net flow over the last 20 / 60 reporting dates (Rs crore; - = net selling)
  fpi_20d_pct / 60d_pct  the same net flow as a share of FPIs' gross buying + selling over those days (scale-free, so 2019
                         and 2024 compare fairly)
  sell_days_20           share of the last 20 reporting dates with net selling
List outcome = average 6-month return of the phase-0 V3.1 list after liquidity-scaled costs (finished windows only).
Reported per era (2019-22, 2023+): rank correlation and the average outcome in the low / high third; where the worst list
(2024-09-11) sat; NSDL's monthly equity totals around 2024-25 and 2026, and today's readings.
Neighbouring weekly lists overlap, so ~15 independent half-years in all.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/leader_sleeve/fpi_weeks"


def main() -> None:
    F = pd.read_parquet(ROOT / "data/derived/nsdl_fpi_daily.parquet")
    F["reporting_date"] = pd.to_datetime(F["reporting_date"]); F["known_from"] = pd.to_datetime(F["known_from"])
    se = F[(F["asset"] == "Equity") & (F["route"] == "Stock Exchange")].sort_values("reporting_date").reset_index(drop=True)
    se["gross"] = se["gross_purchases_cr"] + se["gross_sales_cr"]
    for w in (20, 60):
        se[f"fpi_{w}d"] = se["net_cr"].rolling(w, min_periods=int(0.8 * w)).sum()
        se[f"fpi_{w}d_pct"] = se[f"fpi_{w}d"] / se["gross"].rolling(w, min_periods=int(0.8 * w)).sum()
    se["sell_days_20"] = (se["net_cr"] < 0).astype(float).rolling(20, min_periods=16).mean()
    sig = ["fpi_20d", "fpi_60d", "fpi_20d_pct", "fpi_60d_pct", "sell_days_20"]

    pk = pd.read_csv(ROOT / "logs/leader_sleeve/v31_hits_by_year/v31_picks.csv", parse_dates=["trade_date"])
    pk["cost"] = np.where(pk["adv"] >= 5, 0.005, np.where(pk["adv"] >= 1, 0.010, 0.020))
    L = pk[pk["full"]].groupby("trade_date").agg(ret=("ret", "mean"), cost=("cost", "mean"))
    L["list_ret"] = L["ret"] - L["cost"]
    X = pd.merge_asof(L.reset_index().sort_values("trade_date"), se[["known_from"] + sig].sort_values("known_from"),
                      left_on="trade_date", right_on="known_from", direction="backward").set_index("trade_date")
    X["era"] = np.where(X.index.year <= 2022, "2019-22", "2023+")
    labels = {"fpi_20d": "FPI net flow, last 20 days (Rs cr)", "fpi_60d": "FPI net flow, last 60 days (Rs cr)",
              "fpi_20d_pct": "FPI net flow, 20 days, % of their turnover", "fpi_60d_pct": "FPI net flow, 60 days, % of their turnover",
              "sell_days_20": "share of the last 20 days FPIs sold"}
    rows = []
    for c, lab in labels.items():
        r = {"signal": lab, "known_share": round(float(X[c].notna().mean()), 2)}
        for era, g in X.groupby("era"):
            g = g.dropna(subset=[c, "list_ret"])
            r[f"{era} corr"] = round(float(g[c].rank().corr(g["list_ret"].rank())), 2)
            q = pd.qcut(g[c].rank(method="first"), 3, labels=["low", "mid", "high"])
            m = g.groupby(q, observed=True)["list_ret"].mean()
            r[f"{era} low / high third"] = f"{100 * m['low']:+.0f}% / {100 * m['high']:+.0f}%"
        a, b = r["2019-22 corr"], r["2023+ corr"]
        r["same direction, both >= 0.15"] = bool(np.sign(a) == np.sign(b) and min(abs(a), abs(b)) >= 0.15)
        rows.append(r)
    T = pd.DataFrame(rows)
    print(f"weekly lists with a finished 6 months: {len(X)} · flows known for {100 * X['fpi_20d'].notna().mean():.0f}% of them")
    print(T.to_string(index=False))
    wk = pd.Timestamp("2024-09-11")
    print(f"\nworst list 2024-09-11 ({100 * X.loc[wk, 'list_ret']:+.0f}%): readings " + ", ".join(f"{c} {X.loc[wk, c]:,.3g}" for c in sig)
          + " · percentile among list weeks: " + str({c: round(float((X[c].dropna() <= X.loc[wk, c]).mean()), 2) for c in sig}))
    eq = F[(F["asset"] == "Equity") & (F["route"] == "Sub-total")]
    mo = eq.groupby(eq["reporting_date"].dt.to_period("M"))["net_cr"].sum().round(0)
    print("\nNSDL FPI equity net by month (Rs crore, from the daily rows):")
    print({str(k): int(v) for k, v in mo.loc["2024-06":"2025-04"].items()})
    print({str(k): int(v) for k, v in mo.loc["2025-10":].items()})
    last = se.iloc[-1]
    print(f"latest reporting date {last['reporting_date'].date()}: 20-day net {last['fpi_20d']:,.0f} cr ({100 * last['fpi_20d_pct']:.1f}% of turnover), "
          f"60-day net {last['fpi_60d']:,.0f} cr, sold on {100 * last['sell_days_20']:.0f}% of the last 20 days")
    OUT.mkdir(parents=True, exist_ok=True)
    X.to_csv(OUT / "weekly_fpi.csv")
    (OUT / "weekly_fpi.csv.manifest.json").write_text(json.dumps(dict(dataset="weekly_fpi.csv", producer="src/agentic/explore_fpi_weeks.py", kind="exploration",
        definitions=__doc__, table=T.to_dict("records"), monthly_equity_net_cr={str(k): float(v) for k, v in mo.items()},
        units=dict(list_ret="fraction", fpi_20d="Rs crore", fpi_60d="Rs crore", fpi_20d_pct="fraction of FPI gross turnover", sell_days_20="share 0-1"),
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    (OUT / "README.md").write_text("# Foreign investor flows vs V3.1 weekly lists\n\nExploration. weekly_fpi.csv = each weekly list's outcome with the FPI flow readings known on its date; "
                                   "summary table and NSDL monthly totals in the manifest.\n")


if __name__ == "__main__":
    main()
