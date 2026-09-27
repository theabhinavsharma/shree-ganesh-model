"""HOT INDUSTRY x MATERIAL ORDER x ABOVE 200DMA (EXP-2026-09-27-hot-x-material-order).

Part A (registered): condition C = industry broad-heat pct >= 0.90 AND an order/L1 filing in the prior 60 days
with full-text amount >= 15% of PIT TTM revenue AND close > SMA200; control = hot & > SMA200 & no such order.
Outcomes on the anatomy stock-weeks (mcap >= 50cr): P(touch +50% within 95), P(sustained), mean 95-session
return, by era. Pass: C >= 1.5x base AND >= 1.2x control in BOTH eras, n >= 50 per era.
Part B (drawdown question): B+H rule (broad heat, top-10, EXTENDED), all-in, 90-session hold, 18 phases:
(1) as is, (2) HARD filter = only material-order names, (3) TIE-BREAK = material-order names first within each
hot industry, then own ret60. Windows 2016+ and 2019+.
Inputs: data/derived/order_fulltext.parquet, logs/leader_sleeve/anatomy_1p5x/rows.parquet, pnl_quarterly (units per manifest).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
MAT, WIN, HOLD = 0.15, 60, 90

O = pd.read_parquet(ROOT / "data/derived/order_fulltext.parquet")
O["d"] = pd.to_datetime(O["d"]); O = O.dropna(subset=["d", "amount_cr"])
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
q["sales_cr"] = q["net_sales"] * np.where(q["source"] == "xbrl", 1e-7, 1e-2)
q["rev_ttm_cr"] = q.groupby("symbol")["sales_cr"].transform(lambda s: s.rolling(4).sum())
q["d"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
O = pd.merge_asof(O.sort_values("d"), q.dropna(subset=["rev_ttm_cr"])[["symbol", "d", "rev_ttm_cr"]].sort_values("d"),
                  on="d", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
O["ratio"] = O["amount_cr"] / O["rev_ttm_cr"]
print("orders:", len(O), "· with ratio:", int(O["ratio"].notna().sum()), "· material (>=15% rev):", int((O["ratio"] >= MAT).sum()),
      "· status:", O["status"].value_counts().to_dict(), flush=True)
MO = O[O["ratio"] >= MAT][["symbol", "d"]].sort_values(["symbol", "d"])

S = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet",
                    columns=["symbol", "trade_date", "era", "core", "ind", "ret60", "ret252", "ind_heat_pct", "px_sma200", "y95", "s95", "fc95"])
S = S.reset_index(drop=True)
arr = {s: v["d"].values.astype("datetime64[ns]") for s, v in MO.groupby("symbol")}
t = S["trade_date"].values.astype("datetime64[ns]"); lo = t - np.timedelta64(WIN, "D"); flag = np.zeros(len(S), bool)
for s, idx in S.groupby("symbol").indices.items():
    a = arr.get(s)
    if a is not None:
        flag[idx] = (np.searchsorted(a, t[idx], side="right") - np.searchsorted(a, lo[idx], side="right")) > 0
S["mat_order"] = flag

# ---------------- Part A ----------------
L = S.dropna(subset=["y95"])
base = L.groupby("era")["y95"].mean()
hot, up = L["ind_heat_pct"] >= 0.9, L["px_sma200"] > 0
cells = {"C = hot & material order & >200DMA": hot & L["mat_order"] & up, "control = hot & >200DMA & no order": hot & up & ~L["mat_order"],
         "material order alone": L["mat_order"], "material order & >200DMA": L["mat_order"] & up, "hot & material order": hot & L["mat_order"]}
res = {}
print("\n=== PART A ===")
for k, m in cells.items():
    out = []
    for e in ("disc", "conf"):
        E = L[m & (L["era"] == e)]
        r = dict(n=len(E), p=E["y95"].mean() * 100, sus=E["s95"].mean() * 100, ret=(E["fc95"].mean() - 1) * 100, lift=E["y95"].mean() / base[e])
        res[f"{k}|{e}"] = r
        out.append(f"{e} n={r['n']:>5} P(+50%) {r['p']:5.1f}% ({r['lift']:.2f}x) sustained {r['sus']:4.1f}% mean95 {r['ret']:+5.1f}%")
    print(f"{k:<38}| " + " | ".join(out))
C, K = "C = hot & material order & >200DMA", "control = hot & >200DMA & no order"
ok = all(res[f"{C}|{e}"]["n"] >= 50 and res[f"{C}|{e}"]["lift"] >= 1.5 and res[f"{C}|{e}"]["p"] >= 1.2 * res[f"{K}|{e}"]["p"] for e in ("disc", "conf"))
print(f"\nREGISTERED VERDICT: {'PASS' if ok else 'FAIL'} (C >= 1.5x base and >= 1.2x control, both eras, n >= 50)")

# ---------------- Part B ----------------
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date", "close"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
R = px.pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None); del px
dpos = {d: i for i, d in enumerate(R.index)}
wk = sorted(S["trade_date"].unique())
W = S.dropna(subset=["ind", "ret60"]).copy()
W["n"] = W.groupby(["trade_date", "ind"])["ret60"].transform("size"); W = W[W["n"] >= 5]
h = W.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
W = W.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
W = W[W["hot"] & (W["ret252"] > 0.5) & W["core"]]


def picks(mode: str) -> dict:
    X = W.copy()
    if mode == "hard":
        X = X[X["mat_order"]]
    X = X.sort_values(["trade_date", "ind", "mat_order", "ret60"] if mode == "tie" else ["trade_date", "ind", "ret60"],
                      ascending=[True, True, False, False] if mode == "tie" else [True, True, False])
    if mode != "hard":
        X = X.groupby(["trade_date", "ind"]).head(10)
    return X.groupby("trade_date")["symbol"].apply(list).to_dict()


def allin(pk: dict, start: str) -> pd.DataFrame:
    dates = [d for d in wk if d >= pd.Timestamp(start)]; out = []
    for ph in range(HOLD // 5):
        v, vals, idx = 1.0, [], []
        for d in dates[ph::HOLD // 5]:
            i0 = dpos.get(d)
            if i0 is None or i0 + 1 >= len(R):
                continue
            end = min(i0 + HOLD, len(R) - 1)
            names = [n for n in pk.get(d, []) if n in R.columns]
            path = ((1 + R.iloc[i0 + 1:end + 1][names].fillna(0)).cumprod().mean(axis=1).values * 0.995) if names else np.ones(end - i0)
            vals.extend(v * path); idx.extend(R.index[i0 + 1:end + 1]); v = vals[-1]
        s = pd.Series(vals, index=idx); s = s[~s.index.duplicated()]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        out.append(dict(cagr=(s.iloc[-1] ** (1 / yrs) - 1) * 100, dd=(s / s.cummax() - 1).min() * 100))
    return pd.DataFrame(out)


print("\n=== PART B: B+H all-in, 90-session hold ===")
for mode in ("asis", "hard", "tie"):
    pk = picks(mode); n = np.mean([len(v) for v in pk.values()]) if pk else 0
    inv = len(pk) / len(wk) * 100
    for start in ("2016-06-01", "2019-01-01"):
        r = allin(pk, start)
        res[f"B|{mode}|{start[:4]}"] = dict(cagr_med=r["cagr"].median(), cagr_worst=r["cagr"].min(), dd_med=r["dd"].median(), dd_worst=r["dd"].min(), names=n, weeks_with_picks=inv)
        print(f"  {mode:<5} {start[:4]}+  CAGR median {r['cagr'].median():+6.1f}% (worst {r['cagr'].min():+6.1f}) · maxDD median {r['dd'].median():6.1f}% "
              f"(worst {r['dd'].min():6.1f}) · names/rotation {n:.1f} · weeks with picks {inv:.0f}%", flush=True)
json.dump(res, open(ROOT / "logs/leader_sleeve/hot_order_combo.json", "w"), indent=1, default=float)
print("HOT ORDER COMBO COMPLETE")
