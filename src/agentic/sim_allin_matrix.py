"""ALL-IN x HOLD MATRIX (EXP-2026-09-27-allin-hold-matrix).

Every approach x every hold, with 100% of capital entering at once, equal weight, held h sessions,
sold, and rotated into that date's picks. Phase offsets = h/5 weekly starts; median and worst phase
reported. 0.5% round trip per rotation. Empty pick set = cash. Buys restricted to tradable names
(ADV >= 5cr & close > 50).
Approaches: 8 rule variants of the leader cell (BASELINE + G gate / B top-10 / H broad heat),
MODEL top-10 / top-20 (walk-forward LightGBM out-of-sample score from anatomy_1p5x, 2019+),
equal-weight tradable market.
Inputs: logs/leader_sleeve/anatomy_1p5x/rows.parquet (weekly mcap>=50cr rows with features + pred),
        adjusted price panel for daily returns.
Output: logs/leader_sleeve/allin_matrix.csv (+ manifest) + stdout.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
HOLDS = [30, 60, 90, 100, 180, 250]
S = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet",
                    columns=["symbol", "trade_date", "core", "ind", "ret60", "ret252", "mkt_breadth", "pred"])
wk = sorted(S["trade_date"].unique())
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date", "close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
R = px.pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
del px
dpos = {d: i for i, d in enumerate(R.index)}


def rule(broad: bool, top: int, gate: bool) -> dict:
    W = (S if broad else S[S["core"]]).dropna(subset=["ind", "ret60"]).copy()
    W["n"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size"); W = W[W["n"] >= 5]
    h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
    h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
    W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    W["rk"] = W.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    P = W[W["hot"] & (W["rk"] <= top) & (W["ret252"] > 0.5) & W["core"]]
    if gate:
        P = P[P["mkt_breadth"] >= 0.5]
    return P.groupby("trade_date")["symbol"].apply(list).to_dict()


T = S[S["core"]]
APPROACHES = {f"{'+'.join(c) or 'BASELINE'}": rule("H" in c, 10 if "B" in c else 3, "G" in c)
              for c in [(), ("G",), ("B",), ("H",), ("G", "B"), ("G", "H"), ("B", "H"), ("G", "B", "H")]}
M = T.dropna(subset=["pred"]).sort_values("pred", ascending=False)
APPROACHES["MODEL top-10"] = M.groupby("trade_date").head(10).groupby("trade_date")["symbol"].apply(list).to_dict()
APPROACHES["MODEL top-20"] = M.groupby("trade_date").head(20).groupby("trade_date")["symbol"].apply(list).to_dict()
APPROACHES["EW market"] = T.groupby("trade_date")["symbol"].apply(list).to_dict()


def allin(picks: dict, h: int, start: str) -> pd.DataFrame:
    dates = [d for d in wk if d >= pd.Timestamp(start)]
    out = []
    for ph in range(h // 5):
        seq = dates[ph::h // 5]
        vals, idx = [], []
        v = 1.0
        for d in seq:
            i0 = dpos.get(d)
            if i0 is None or i0 + 1 >= len(R):
                continue
            end = min(i0 + h, len(R) - 1)
            names = [n for n in picks.get(d, []) if n in R.columns]
            if names:
                path = (1 + R.iloc[i0 + 1:end + 1][names].fillna(0)).cumprod().mean(axis=1).values * (1 - 0.005)
            else:
                path = np.ones(end - i0)
            vals.extend(v * path); idx.extend(R.index[i0 + 1:end + 1]); v = vals[-1]
        s = pd.Series(vals, index=idx); s = s[~s.index.duplicated()]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out.append(dict(cagr=(s.iloc[-1] ** (1 / yrs) - 1) * 100, dd=(s / s.cummax() - 1).min() * 100))
    return pd.DataFrame(out)


rows = []
for window, start in (("2019+", "2019-01-01"), ("2016+", "2016-06-01")):
    for name, pk in APPROACHES.items():
        if window == "2016+" and name.startswith("MODEL"):
            continue
        for h in HOLDS:
            r = allin(pk, h, start)
            rows.append(dict(window=window, approach=name, hold=h, cagr_med=r["cagr"].median(), cagr_worst=r["cagr"].min(),
                             dd_med=r["dd"].median(), dd_worst=r["dd"].min(), phases=len(r)))
        print(f"done {window} {name}", flush=True)
X = pd.DataFrame(rows)
out = ROOT / "logs/leader_sleeve/allin_matrix.csv"
X.to_csv(out, index=False)
Path(str(out) + ".manifest.json").write_text(json.dumps(dict(
    dataset="all-in hold matrix", experiment="EXP-2026-09-27-allin-hold-matrix", producer="src/agentic/sim_allin_matrix.py",
    columns=dict(window="evaluation window start", approach="pick rule", hold="sessions held per rotation", cagr_med="median CAGR % across phase offsets",
                 cagr_worst="worst-phase CAGR %", dd_med="median max drawdown %", dd_worst="worst-phase max drawdown %", phases="number of start offsets"),
    updated=datetime.now().isoformat(timespec="seconds")), indent=1))
for window in ("2019+", "2016+"):
    W = X[X["window"] == window]
    for metric, lab in (("cagr_med", "MEDIAN CAGR %"), ("cagr_worst", "WORST-PHASE CAGR %"), ("dd_med", "MEDIAN MAX DRAWDOWN %")):
        print(f"\n=== {window} · {lab} (rows = approach, cols = hold sessions) ===")
        print(W.pivot(index="approach", columns="hold", values=metric).reindex([a for a in APPROACHES if a in set(W["approach"])]).round(1).to_string())
print("\nALLIN MATRIX COMPLETE")
