"""HINDSIGHT autopsy of the hot-industry leader cell (2026-09-23). DESCRIPTIVE — no ship bar.

Companion to autopsy_leader_winners.py (ex-ante features at entry). This one looks at
what happened AFTER entry, split by outcome within 126td:
  2x   = high touched >= +100%     50x = touched +50..100%     rest = never +50%
Path features: close return at day 5/10/20/40, trough by day 40, the industry's OWN
forward 126td return (peers, self excluded), and the first earnings print after entry
(EPS yoy change sign, loss->profit turn). Also P(2x) by day-20 return bucket, both
eras — the one path feature that is observable while holding.
Anything here that looks tradeable is a candidate for a NEW registered tournament,
never a rule: trailing stops were killed on this cell (they sell the trough).
Cell/universe identical to sim_leader_sleeve.py (core band, weekly, 2016-06+).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "low", "close", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["ret252"] = g["close"].pct_change(252)
fwd = lambda col, fn, w: g[col].transform(lambda s: getattr(s.shift(-1)[::-1].rolling(w, min_periods=int(w * .8)), fn)()[::-1])
px["hi126"] = fwd("high", "max", 126)
px["lo40"] = fwd("low", "min", 40)
for k in (5, 10, 20, 40, 126):
    px[f"r{k}"] = g["close"].shift(-k) / px["close"] - 1
px["adv"] = px["avg_traded_value_20d"] / 1e7

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
px["ind"] = px["symbol"].map(imap)

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna() & px["ret60"].notna()
        & px["hi126"].notna() & px["r126"].notna()].copy()
gs = wk.groupby(["trade_date", "ind"])
wk["nm"] = gs["ret60"].transform("size")
wk = wk[wk["nm"] >= 5]
gr = wk.groupby(["trade_date", "ind"])["ret60"].mean().rename("gr").reset_index()
gr["gd"] = gr.groupby("trade_date")["gr"].rank(pct=True)
wk = wk.merge(gr, on=["trade_date", "ind"])
# industry's own forward 126td return, peers only (self excluded)
s_, n_ = wk.groupby(["trade_date", "ind"])["r126"].transform("sum"), wk["nm"]
wk["peer_fwd126"] = (s_ - wk["r126"]) / (n_ - 1)
wk["rk"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False)
L = wk[(wk["gd"] >= 0.9) & (wk["rk"] <= 3)].copy()

# first earnings print after entry (point-in-time filing date), yoy EPS change
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
q["eps_yoy"] = q.groupby("symbol")["eps_basic"].diff(4)
q["turn"] = (q["eps_basic"] > 0) & (q.groupby("symbol")["eps_basic"].shift(4) <= 0)
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
L = pd.merge_asof(L.sort_values("trade_date"),
                  q.dropna(subset=["eps_yoy"])[["symbol", "filing_dt", "eps_yoy", "turn"]].rename(columns={"filing_dt": "trade_date"}).sort_values("trade_date"),
                  on="trade_date", by="symbol", direction="forward", tolerance=pd.Timedelta(days=120))

up = L["hi126"] / L["close"] - 1
L["outcome"] = np.where(up >= 1.0, "2x", np.where(up >= 0.5, "50-100", "rest"))
L["era"] = np.where(L["trade_date"].dt.year >= 2023, "conf>=23", "disc<=22")
L["trough40"] = L["lo40"] / L["close"] - 1
L["print_up"] = np.where(L["eps_yoy"].notna(), L["eps_yoy"] > 0, np.nan)

print(f"leader-cell rows {len(L):,}  ·  panel through {px['trade_date'].max().date()}")
print("\n=== MEDIAN PATH BY OUTCOME (126td) ===")
print(f"{'era':<9}{'outcome':<8}{'n':>6} {'d5':>7}{'d10':>7}{'d20':>7}{'d40':>7} {'trough40':>9} {'peers fwd126':>13} {'print EPS up':>13} {'loss→profit':>12}")
for era in ("disc<=22", "conf>=23"):
    for oc in ("2x", "50-100", "rest"):
        S = L[(L["era"] == era) & (L["outcome"] == oc)]
        m = lambda c: S[c].median() * 100
        pu = S["print_up"].dropna()
        print(f"{era:<9}{oc:<8}{len(S):>6} {m('r5'):>+6.1f}%{m('r10'):>+6.1f}%{m('r20'):>+6.1f}%{m('r40'):>+6.1f}% "
              f"{m('trough40'):>+8.1f}% {m('peer_fwd126'):>+12.1f}% {pu.mean()*100 if len(pu) else np.nan:>12.0f}% "
              f"{S['turn'].fillna(False).astype(bool).mean()*100:>11.1f}%")

def table(col, edges, labels, title):
    print(f"\n=== P(outcome) BY {title} ===")
    L["_b"] = pd.cut(L[col], edges, labels=labels)
    print(f"{'bucket':<14}{'era':<10}{'n':>6} {'P2x':>6} {'P50':>6} {'mean 126td':>11}")
    for era in ("disc<=22", "conf>=23"):
        E = L[L["era"] == era]
        for b in labels:
            S = E[E["_b"] == b]
            if len(S) < 30:
                continue
            print(f"{b:<14}{era:<10}{len(S):>6} {(S['outcome']=='2x').mean()*100:>5.1f}% "
                  f"{(S['outcome']!='rest').mean()*100:>5.1f}% {S['r126'].mean()*100:>+10.1f}%")

table("r20", [-9, -0.10, 0, 0.10, 0.25, 9], ["<-10%", "-10..0", "0..+10", "+10..+25", ">+25%"], "DAY-20 RETURN (observable in-trade)")
table("peer_fwd126", [-9, 0, 0.15, 0.40, 99], ["peers<0", "0..15%", "15..40%", ">40%"], "INDUSTRY PEERS' FWD 126td (hindsight only)")
print("\nAUTOPSY PATHS COMPLETE")
