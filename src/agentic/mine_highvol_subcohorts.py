"""HIGH-VOL SUB-COHORT MINER — find the conditional structure inside the 'scary' cohort.

User directive 2026-08-25: 'Everyone sees high-vol as a negative cohort — your job is to
find the sub-cohort that gives the highest positive returns.'

Universe: TOP-TERCILE dvol names (per weekly cross-section), ADV>=5cr, close>50, 10 years.
Outcomes: end15 (hold-15 return, ROI metric), touch5 (+5% touch), trough15 (risk).
Discipline: discover 2016-2022 → confirm 2023-2026.08 → survivors reported with both.
Survival bar: discovery t>=3 & n>=300; confirm mean>0 & t>=2; positive in >=60% of years.
"""
import pandas as pd, numpy as np

print("loading…", flush=True)
px = pd.read_parquet('data/derived/stock_daily_facts_adjusted_2015plus.parquet',
    columns=['symbol','trade_date','open','high','low','close','return_1d','rsi_14_daily',
             'return_20d','volume_vs_20d','delivery_pct','avg_delivery_pct_20d',
             'avg_traded_value_20d','sma_200'])
px['trade_date'] = pd.to_datetime(px['trade_date'])
px = px.sort_values(['symbol','trade_date'])
g = px.groupby('symbol')

px['dvol'] = g['return_1d'].transform(lambda s: s.rolling(20).std())
px['dvol5'] = g['return_1d'].transform(lambda s: s.rolling(5).std())
px['r60'] = g['close'].pct_change(60)
px['adv'] = px['avg_traded_value_20d']/1e7
px['dlv'] = px['delivery_pct']/px['avg_delivery_pct_20d']
px['dlv5'] = g['dlv'].transform(lambda s: s.rolling(5).mean())
px['dlv20'] = g['dlv'].transform(lambda s: s.rolling(20).mean())
px['hi252'] = g['close'].transform(lambda s: s.rolling(252, min_periods=60).max())
px['off52'] = px['close']/px['hi252'] - 1
px['rsi_mu'] = g['rsi_14_daily'].transform(lambda s: s.rolling(252, min_periods=120).mean())
px['rsi_sd'] = g['rsi_14_daily'].transform(lambda s: s.rolling(252, min_periods=120).std())
px['rsi_z'] = (px['rsi_14_daily'] - px['rsi_mu'])/px['rsi_sd']

# volume asymmetry: up-day volume vs down-day volume (15d)
px['up_vol'] = np.where(px['return_1d'] > 0, px['volume_vs_20d'], np.nan)
px['dn_vol'] = np.where(px['return_1d'] < 0, px['volume_vs_20d'], np.nan)
px['up_vol15'] = g['up_vol'].transform(lambda s: s.rolling(15, min_periods=3).mean())
px['dn_vol15'] = g['dn_vol'].transform(lambda s: s.rolling(15, min_periods=3).mean())
px['vol_asym'] = px['up_vol15']/px['dn_vol15']

# institutional prints (15d counts)
px['hv_up'] = ((px['return_1d'] >= 0.03) & (px['volume_vs_20d'] > 1.5)).astype(float)
px['hv_dn'] = ((px['return_1d'] <= -0.03) & (px['volume_vs_20d'] > 1.5)).astype(float)
px['hv_up15'] = g['hv_up'].transform(lambda s: s.rolling(15).sum())
px['hv_dn15'] = g['hv_dn'].transform(lambda s: s.rolling(15).sum())

# forward outcomes
def fmax(s): return s.shift(-1)[::-1].rolling(15, min_periods=10).max()[::-1]
def fmin(s): return s.shift(-1)[::-1].rolling(15, min_periods=10).min()[::-1]
def fend(s): return s.shift(-15)
px['f_hi'] = g['high'].transform(fmax)
px['f_lo'] = g['low'].transform(fmin)
px['f_cl'] = g['close'].transform(fend)
px['end15'] = (px['f_cl']/px['close'] - 1)*100
px['touch5'] = (px['f_hi']/px['close'] - 1 >= 0.05)
px['trough'] = (px['f_lo']/px['close'] - 1)*100

days = sorted(px['trade_date'].unique())
weekly = set(days[::5])
u = px[(px['adv']>=5) & (px['close']>50) & px['trade_date'].isin(weekly) & px['end15'].notna() & px['dvol'].notna()].copy()
u['dvol_rank'] = u.groupby('trade_date')['dvol'].rank(pct=True)
hv = u[u['dvol_rank'] >= 2/3].copy()   # THE high-vol cohort
hv['year'] = hv['trade_date'].dt.year
print(f"high-vol universe rows: {len(hv):,} · baseline end15 {hv['end15'].mean():+.2f}% · touch {hv['touch5'].mean()*100:.1f}%", flush=True)

CONDS = {
 'vol_asym>1.3 (up-day vol dominates)': hv['vol_asym'] > 1.3,
 'vol_asym>1.6': hv['vol_asym'] > 1.6,
 'hv_up>=2 & hv_dn==0 (only institutional BUY prints)': (hv['hv_up15']>=2) & (hv['hv_dn15']==0),
 'hv_up>=1 & hv_dn==0': (hv['hv_up15']>=1) & (hv['hv_dn15']==0),
 'dlv5/dlv20>1.2 (delivery accelerating)': hv['dlv5']/hv['dlv20'] > 1.2,
 'dlv>1.2 level': hv['dlv'] > 1.2,
 'coil: dvol5/dvol20<0.6 (quiet after loud)': hv['dvol5']/hv['dvol'] < 0.6,
 'near 52wH (>-10%)': hv['off52'] > -0.10,
 'above 200DMA': hv['close'] > hv['sma_200'],
 'r60>+20%': hv['r60'] > 0.20,
 'r60<-20% (crashed)': hv['r60'] < -0.20,
 'rsi_z<-1 (own-scale washed)': hv['rsi_z'] < -1.0,
 'rsi_z in [-1,0.25]': hv['rsi_z'].between(-1.0,0.25),
 '20d flat own-scale (|r20z|<0.5)': (hv['return_20d']/(hv['dvol']*np.sqrt(20))).abs() < 0.5,
}

def cell_stats(mask, era_mask):
    s = hv[mask & era_mask]
    if len(s) < 30: return None
    m = s['end15'].mean(); sd = s['end15'].std()
    t = m/(sd/np.sqrt(len(s))) if sd > 0 else 0
    yr = s.groupby('year')['end15'].mean()
    return dict(n=len(s), mean=m, t=t, touch=s['touch5'].mean()*100,
                trough=s['trough'].mean(), yrs_pos=(yr>0).mean()*100)

DISC = hv['year'] <= 2022
CONF = hv['year'] >= 2023
print(f"\n{'CONDITION':<52s} | {'disc n':>7s} {'mean':>7s} {'t':>5s} | {'conf n':>7s} {'mean':>7s} {'t':>5s} | {'yrs+':>5s} verdict")
survivors = []
for name, m in CONDS.items():
    d = cell_stats(m, DISC); c = cell_stats(m, CONF)
    if d is None or c is None: continue
    allc = cell_stats(m, hv['year']>0)
    ok = d['n']>=300 and d['t']>=3 and c['mean']>0 and c['t']>=2 and allc['yrs_pos']>=60
    v = '✅ SURVIVES' if ok else ''
    print(f"{name:<52s} | {d['n']:>7,} {d['mean']:>+6.2f}% {d['t']:>5.1f} | {c['n']:>7,} {c['mean']:>+6.2f}% {c['t']:>5.1f} | {allc['yrs_pos']:>4.0f}% {v}", flush=True)
    if ok: survivors.append(name)

# pairs among promising singles
print("\nPAIR SCAN (top singles × top singles):", flush=True)
promising = [k for k in CONDS if (cell_stats(CONDS[k], DISC) or {'t':0})['t'] >= 2][:6]
pair_survivors = []
for i, a in enumerate(promising):
    for b in promising[i+1:]:
        m = CONDS[a] & CONDS[b]
        d = cell_stats(m, DISC); c = cell_stats(m, CONF)
        if d is None or c is None or d['n'] < 300: continue
        allc = cell_stats(m, hv['year']>0)
        ok = d['t']>=3 and c['mean']>0 and c['t']>=2 and allc['yrs_pos']>=60
        if d['t'] >= 2.5 or ok:
            v = '✅ SURVIVES' if ok else ''
            print(f"  [{a}] AND [{b}]\n    disc n={d['n']:,} {d['mean']:+.2f}% t={d['t']:.1f} | conf n={c['n']:,} {c['mean']:+.2f}% t={c['t']:.1f} | yrs+ {allc['yrs_pos']:.0f}% {v}", flush=True)
            if ok: pair_survivors.append((a,b))

print(f"\nfamilies tested: {len(CONDS)} singles + pairs — Bonferroni context for survivor count")
print(f"SURVIVORS: {survivors + pair_survivors}")
print("MINER COMPLETE", flush=True)
