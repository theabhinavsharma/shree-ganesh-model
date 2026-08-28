"""DOUBLER IGNITION MINER — walk-forward test of pre-registered 2x-ignition cells.

Registered 2026-08-28 from the 78-doubler dossier audit (research/doubler_audit/).
Cells testable over 10 years (price/volume/delivery only — the announcements store
starts 2026-02, so announcement-anchored cells are forward-tracked, not backtested):

  C1 IGNITION-BURST : ret1d>=+8% & vol>=4x & (dlv<=25% or dlv<=0.5x own 20d mean)
                      & off52<=-25%
  C5 GRIND-ACCUM    : trailing 15 sessions: >=5 up-days of >=+3.5% on vol 1-3x with
                      delivery>=35%, cumulative +15..+35%
  C6 IPO-BASE-BREAK : first parquet date <=252td ago, then first day ret>=+8% & vol>=3x
  B0 BASELINE-THRUST: ret1d>=+5% & vol>=1.5x (the generic thrust — base-rate anchor)
  ADV overlay       : each cell also reported restricted to ADV 2-25cr

Outcomes from event CLOSE: touch 2x within 100td · touch +25% within 30td ·
median max-drawdown within 100td · weekly-cohort (portfolio) mean of hold-100td
return. Discovery <=2022 · confirm >=2023 · yearly positivity. One event per
symbol per 60td (first qualifying day wins).
"""
import pandas as pd, numpy as np

print("loading…", flush=True)
px = pd.read_parquet('data/derived/stock_daily_facts_adjusted_2015plus.parquet',
    columns=['symbol','trade_date','close','high','low','return_1d','volume_vs_20d',
             'delivery_pct','avg_delivery_pct_20d','avg_traded_value_20d'])
px['trade_date'] = pd.to_datetime(px['trade_date'])
px = px.sort_values(['symbol','trade_date']).reset_index(drop=True)
g = px.groupby('symbol')

px['adv'] = px['avg_traded_value_20d']/1e7
px['hi252'] = g['close'].transform(lambda s: s.rolling(252, min_periods=60).max())
px['off52'] = px['close']/px['hi252'] - 1
px['dlv_mu20'] = g['delivery_pct'].transform(lambda s: s.rolling(20, min_periods=8).mean())
px['age_td'] = g.cumcount()

up = (px['return_1d'] >= 0.035) & px['volume_vs_20d'].between(1, 3) & (px['delivery_pct'] >= 35)
px['grind_days15'] = up.groupby(px['symbol']).transform(lambda s: s.rolling(15).sum())
px['ret15'] = g['close'].pct_change(15)

def fwd(col, n, fn):
    return g[col].transform(lambda s: fn(s.shift(-1)[::-1].rolling(n, min_periods=min(20, n))) [::-1])
print("forward outcomes…", flush=True)
px['f_hi100'] = fwd('high', 100, lambda r: r.max())
px['f_lo100'] = fwd('low', 100, lambda r: r.min())
px['f_hi30']  = fwd('high', 30, lambda r: r.max())
px['f_cl100'] = g['close'].transform(lambda s: s.shift(-100))
px['t2x']  = px['f_hi100']/px['close'] - 1 >= 1.00
px['t25']  = px['f_hi30']/px['close'] - 1 >= 0.25
px['dd']   = (px['f_lo100']/px['close'] - 1)*100
px['end100'] = (px['f_cl100']/px['close'] - 1)*100

base_univ = (px['close'] > 50) & (px['adv'] >= 2) & px['f_hi100'].notna()

CELLS = {
 'B0 baseline thrust': (px['return_1d']>=0.05) & (px['volume_vs_20d']>=1.5),
 'C1 ignition-burst':  (px['return_1d']>=0.08) & (px['volume_vs_20d']>=4)
                        & ((px['delivery_pct']<=25) | (px['delivery_pct']<=0.5*px['dlv_mu20']))
                        & (px['off52']<=-0.25),
 'C5 grind-accum':     (px['grind_days15']>=5) & px['ret15'].between(0.15, 0.35),
 'C6 ipo-base-break':  (px['age_td']<=252) & (px['return_1d']>=0.08) & (px['volume_vs_20d']>=3),
}

def one_per_60td(df):
    keep = []
    for sym, gg in df.groupby('symbol'):
        last = None
        for i, r in gg.sort_values('trade_date').iterrows():
            if last is None or (r['trade_date'] - last).days > 90:  # ~60td
                keep.append(i); last = r['trade_date']
    return df.loc[keep]

print(f"\n{'CELL':<28s}| era      |     n |  P(2x/100td) | P(+25%/30td) | med DD | wk-mean end100 | yrs+")
for name, m in CELLS.items():
    for adv_tag, adv_m in [('', px['adv']>=2), (' [ADV 2-25cr]', px['adv'].between(2,25))]:
        ev = one_per_60td(px[base_univ & m & adv_m])
        if len(ev) < 40: continue
        ev = ev.copy(); ev['year'] = ev['trade_date'].dt.year
        ev['week'] = ev['trade_date'].dt.to_period('W')
        for era, em in [('disc<=22', ev['year']<=2022), ('conf>=23', ev['year']>=2023)]:
            s = ev[em]
            if len(s) < 25:
                print(f"{name+adv_tag:<28s}| {era} | {len(s):>5,} |  (too few)"); continue
            wk = s.groupby('week')['end100'].mean()
            yr = s.groupby('year')['end100'].mean()
            print(f"{name+adv_tag:<28s}| {era} | {len(s):>5,} |       {s['t2x'].mean()*100:5.1f}% |       {s['t25'].mean()*100:5.1f}% | {s['dd'].median():+6.1f}% |        {wk.mean():+6.2f}% | {(yr>0).mean()*100:3.0f}%")
print("\nbase rate (any liquid day):", flush=True)
allday = px[base_univ]
print(f"  P(2x/100td)={allday['t2x'].mean()*100:.2f}% · P(+25%/30td)={allday['t25'].mean()*100:.2f}%")
print("MINER COMPLETE", flush=True)
