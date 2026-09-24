"""POCKET SEARCH inside the hot-industry leader cell (EXP-2026-09-24-leader-pockets).

Question: is there a sub-universe where net ROI per trade is >= +20% (126td hold)?
Base rows: nse4 industry map, fund units out, survivorship-clean paths (same machinery as
sim_leader_cell_v2.py), weekly 2016-06+, core band, MEAN heat percentile >= 0.8, rank <= 5.
Protocol (registered before running): 1-D buckets over every dimension, 2-D pairs of the
top 1-D disc survivors. DISCOVER on disc<=2022 (n>=150, net >= +20%) -> CONFIRM on conf>=2023
(n>=100): PASS = conf net >= +20% AND conf median > 0 AND conf worst weekly cohort > -40%.
Conf-only winners are printed as 'conf-only, unconfirmed'. Cells tested are counted.
Also answers: is the old smIndustry-labelled universe better, and if so why.
Output: logs/leader_sleeve/pockets_20260924.log (stdout) + pockets_20260924.parquet (+manifest).
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

COST, HOLD, BAR = 0.5, 126, 20.0
MIN_DISC, MIN_CONF = 150, 100

# ---------- maps ----------
sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
old_labelled = set(sm.loc[sm["industry_source"].str.startswith("nse_smIndustry"), "symbol"])
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
imap = sc.set_index("symbol")["industry"].map(html.unescape)
al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
imap = pd.concat([imap, al[~al["symbol"].isin(imap.index)].set_index("symbol")["analog_symbol"].map(imap).dropna()])

# ---------- panel ----------
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                     columns=["symbol", "trade_date", "high", "low", "close", "return_1d", "avg_traded_value_20d"])
px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])]
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60, fill_method=None)
px["ret252"] = g["close"].pct_change(252, fill_method=None)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
px["hi252"] = g["high"].transform(lambda s: s.rolling(252, min_periods=120).max())
pos = g.cumcount().to_numpy(); n = g["close"].transform("size").to_numpy()
exit_i = (np.arange(len(px)) - pos) + np.minimum(pos + HOLD, n - 1)
panel_end = px["trade_date"].max()
delisted = (g["trade_date"].transform("max") < panel_end - pd.Timedelta(days=10)).to_numpy()
px["valid"] = (pos + HOLD <= n - 1) | delisted
px["exit_close"] = px["close"].to_numpy()[exit_i]
px["hi_f"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(HOLD, min_periods=1).max()[::-1])
px["lo_f"] = g["low"].transform(lambda s: s.shift(-1)[::-1].rolling(HOLD, min_periods=1).min()[::-1])
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["ind"] = px["symbol"].map(imap)
first_dt = g["trade_date"].transform("min")
px["age_yrs"] = (px["trade_date"] - first_dt).dt.days / 365.25          # listing age within panel (2015+ floor)

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna() & px["ret60"].notna()].copy()
wk["nm"] = wk.groupby(["trade_date", "ind"])["ret60"].transform("size")
wk = wk[wk["nm"] >= 5].copy()
h = wk.groupby(["trade_date", "ind"])["ret60"].mean().rename("heat").reset_index()
h["heat_pct"] = h.groupby("trade_date")["heat"].rank(pct=True)
wk = wk.merge(h, on=["trade_date", "ind"])
wk["rk"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
L = wk[(wk["heat_pct"] >= 0.8) & (wk["rk"] <= 5) & wk["valid"]].copy()

# ---------- PIT PE vs industry (TTM EPS known by filing date) ----------
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
q["ttm"] = q.groupby("symbol")["eps_basic"].transform(lambda s: s.rolling(4).sum())
q["known"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
tt = q.dropna(subset=["ttm"])[["symbol", "known", "ttm"]].rename(columns={"known": "trade_date"}).sort_values("trade_date")
L = pd.merge_asof(L.sort_values("trade_date"), tt, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
W = pd.merge_asof(wk.sort_values("trade_date"), tt, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
W["pe"] = np.where(W["ttm"] > 0, W["close"] / W["ttm"], np.nan)
ind_pe = W.dropna(subset=["pe"]).groupby(["trade_date", "ind"])["pe"].median().rename("ind_pe")
L = L.merge(ind_pe.reset_index(), on=["trade_date", "ind"], how="left")
L["pe_ind"] = np.where(L["ttm"] > 0, (L["close"] / L["ttm"]) / L["ind_pe"], np.nan)

mac = pd.read_parquet(ROOT / "data/derived/macro_panel.parquet", columns=["trade_date", "breadth_50"])
mac["trade_date"] = pd.to_datetime(mac["trade_date"])
L = pd.merge_asof(L.sort_values("trade_date"), mac.sort_values("trade_date"), on="trade_date", direction="backward")

# ---------- outcomes ----------
L["net"] = (L["exit_close"] / L["close"] - 1) * 100 - COST
L["x2"] = L["hi_f"] / L["close"] - 1 >= 1.0
L["era"] = np.where(L["trade_date"].dt.year >= 2023, "conf", "disc")
L["old_map"] = L["symbol"].isin(old_labelled)

# ---------- dimensions ----------
def cut(col, edges, labels):
    return pd.cut(L[col], edges, labels=labels).astype(str)

D = {
    "heat_pct": cut("heat_pct", [0.8, 0.9, 0.95, 1.01], ["h80-90", "h90-95", "h95+"]),
    "rank": cut("rk", [0, 1, 3, 5], ["r1", "r2-3", "r4-5"]),
    "own60": cut("ret60", [-9, 0.3, 0.6, 1.0, 99], ["60<30%", "60 30-60", "60 60-100", "60>100%"]),
    "own252": cut("ret252", [-9, 0.5, 1.0, 2.0, 999], ["252<50%", "252 50-100", "252 100-200", "252>200%"]),
    "adv": cut("adv", [0, 10, 30, 100, 1e9], ["adv5-10", "adv10-30", "adv30-100", "adv100+"]),
    "price": cut("close", [0, 100, 300, 1000, 1e9], ["px<100", "px100-300", "px300-1k", "px1k+"]),
    "dvol": cut("dvol", [0, 0.02, 0.03, 0.045, 9], ["vol<2%", "vol2-3%", "vol3-4.5%", "vol4.5+%"]),
    "off_high": pd.cut(1 - L["close"] / L["hi252"], [-1, 0.03, 0.10, 0.25, 2],
                       labels=["at-high(<3%)", "3-10% off", "10-25% off", ">25% off"]).astype(str),
    "grp_size": cut("nm", [0, 7, 15, 9999], ["grp5-7", "grp8-15", "grp16+"]),
    "pe_ind": pd.Series(np.select([L["ttm"] <= 0, L["pe_ind"] < 0.7, L["pe_ind"] < 1.0, L["pe_ind"] < 1.5, L["pe_ind"] >= 1.5],
                                  ["LOSS", "pe<0.7x", "pe0.7-1x", "pe1-1.5x", "pe1.5x+"], "pe n/a"), index=L.index),
    "breadth": cut("breadth_50", [-1, 0.35, 0.55, 2], ["mkt weak", "mkt mid", "mkt strong"]),
    "age": cut("age_yrs", [-1, 1, 3, 99], ["listed<1y", "1-3y", "3y+"]),
    "old_map": L["old_map"].map({True: "old-map labelled", False: "new-only"}),
}


def stats(S: pd.DataFrame) -> dict:
    if len(S) == 0:
        return dict(n=0)
    coh = S.groupby("trade_date")["net"].mean()
    return dict(n=len(S), net=S["net"].mean(), med=S["net"].median(), win=(S["net"] > 0).mean() * 100,
                p2x=S["x2"].mean() * 100, worst=coh.min(), weeks=len(coh))


rows, tested = [], 0
def evaluate(label: str, mask: pd.Series):
    global tested
    tested += 1
    dd, cc = stats(L[mask & (L["era"] == "disc")]), stats(L[mask & (L["era"] == "conf")])
    rows.append(dict(pocket=label, **{f"d_{k}": v for k, v in dd.items()}, **{f"c_{k}": v for k, v in cc.items()}))


evaluate("ALL rows (heat>=0.8, rk<=5)", pd.Series(True, index=L.index))
evaluate("PRODUCTION (heat>=0.9, rk<=3, 252>50%)", (L["heat_pct"] >= 0.9) & (L["rk"] <= 3) & (L["ret252"] > 0.5))
for dim, s in D.items():
    for b in sorted(s.unique()):
        if b in ("nan", "None"):
            continue
        evaluate(f"{dim}={b}", s == b)
one = pd.DataFrame(rows)
top1 = one[(one["d_n"] >= MIN_DISC) & ~one["pocket"].str.startswith(("ALL", "PRODUCTION"))].sort_values("d_net", ascending=False).head(12)
for (a, b) in itertools.combinations(top1["pocket"], 2):
    da, va = a.split("=", 1); db, vb = b.split("=", 1)
    if da == db:
        continue
    evaluate(f"{a} & {b}", (D[da] == va) & (D[db] == vb))
res = pd.DataFrame(rows)

pd.set_option("display.width", 250)
fmt = lambda r: (f"{r['pocket'][:58]:<58} | disc n={r['d_n']:>5.0f} {r.get('d_net', np.nan):>+6.1f}% med {r.get('d_med', np.nan):>+6.1f}% "
                 f"| conf n={r['c_n']:>5.0f} {r.get('c_net', np.nan):>+6.1f}% med {r.get('c_med', np.nan):>+6.1f}% "
                 f"worst {r.get('c_worst', np.nan):>+6.1f}% P2x {r.get('c_p2x', np.nan):>4.1f}")
print(f"panel through {panel_end.date()} · leader rows {len(L):,} (disc {int((L['era']=='disc').sum()):,}, conf {int((L['era']=='conf').sum()):,}) · cells tested {tested}\n")
print("=== REFERENCE ===")
for _, r in res[res["pocket"].str.startswith(("ALL", "PRODUCTION"))].iterrows():
    print(fmt(r))
print("\n=== 1-D BUCKETS (sorted by disc net) ===")
for _, r in res.iloc[2:2 + sum(len(set(s)) for s in D.values())].sort_values("d_net", ascending=False).iterrows():
    if r["d_n"] >= 50:
        print(fmt(r))
disc_short = res[(res["d_n"] >= MIN_DISC) & (res["d_net"] >= BAR)]
passed = disc_short[(disc_short["c_n"] >= MIN_CONF) & (disc_short["c_net"] >= BAR) & (disc_short["c_med"] > 0) & (disc_short["c_worst"] > -40)]
print(f"\n=== DISCOVERED on disc (n>={MIN_DISC}, net>={BAR:.0f}%): {len(disc_short)} ===")
for _, r in disc_short.sort_values("d_net", ascending=False).iterrows():
    print(("PASS  " if r["pocket"] in set(passed["pocket"]) else "fail  ") + fmt(r))
conf_only = res[(res["c_n"] >= MIN_CONF) & (res["c_net"] >= BAR) & ~res["pocket"].isin(disc_short["pocket"])]
print(f"\n=== CONF-ONLY >= {BAR:.0f}% (n>={MIN_CONF}) — unconfirmed, discovery era did not show it: {len(conf_only)} ===")
for _, r in conf_only.sort_values("c_net", ascending=False).head(15).iterrows():
    print("conf-only " + fmt(r))
print(f"\nPASSED (disc-discovered, conf-confirmed): {len(passed)} of {tested} cells tested")

out = ROOT / "logs/leader_sleeve/pockets_20260924.parquet"
res.to_parquet(out, index=False)
out.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
    dataset="leader pocket search", experiment="EXP-2026-09-24-leader-pockets", rows=len(res), cells_tested=tested,
    producer="src/agentic/pocket_search_leader.py",
    columns={"pocket": "bucket definition", "d_*/c_*": "disc<=2022 / conf>=2023 stats", "n": "trades", "net": "mean net return per trade, % (126td, 0.5% rt)",
             "med": "median net %, ", "win": "% trades > 0", "p2x": "% touching 2x within 126td", "worst": "worst weekly-cohort mean net %", "weeks": "weekly cohorts"},
    updated=datetime.now().isoformat(timespec="seconds")), indent=1))
print("POCKET SEARCH COMPLETE")
