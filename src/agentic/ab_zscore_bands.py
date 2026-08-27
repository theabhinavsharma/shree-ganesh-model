"""A/B: ABSOLUTE entry bands (current prod) vs Z-SCORE bands (each stock's own scale).

Arms differ ONLY in the three band tests (all trailing-data, no fitting → walk-forward valid):
  ABS: RSI in [42,60] · ret20d in [-10%,+15%] · ret5d in [-5%,+5%]
  Z  : RSI_z in [-1.0,+0.25] (vs own 252d mean/std) · ret20d/(dvol*sqrt20) in [-1.0,+0.75]
       · ret5d/(dvol*sqrt5) in [-1.0,+1.0]
Common base: ADV>=5cr, close>50, vol_vs_20d<2.

Metrics per arm, weekly grid over 10 years:
  - pool size · pool-level P(+5% touch in 15d)   ← the ship/kill criterion
  - top-8-by-delivery C2 realized net (day-by-day, next-open, 0.30% cost)
  - since-April-2026 subwindow reported separately (the standing bar)
"""
import pandas as pd, numpy as np

print("loading…", flush=True)
px = pd.read_parquet('data/derived/stock_daily_facts_adjusted_2015plus.parquet',
    columns=['symbol','trade_date','open','high','low','close','return_1d','rsi_14_daily',
             'return_20d','volume_vs_20d','delivery_pct','avg_delivery_pct_20d','avg_traded_value_20d'])
px['trade_date'] = pd.to_datetime(px['trade_date'])
px = px.sort_values(['symbol','trade_date'])
g = px.groupby('symbol')
px['ret_5d'] = g['close'].pct_change(5)
px['dvol'] = g['return_1d'].transform(lambda s: s.rolling(20).std())
px['rsi_mu'] = g['rsi_14_daily'].transform(lambda s: s.rolling(252, min_periods=120).mean())
px['rsi_sd'] = g['rsi_14_daily'].transform(lambda s: s.rolling(252, min_periods=120).std())
px['adv'] = px['avg_traded_value_20d']/1e7
px['dlv'] = px['delivery_pct']/px['avg_delivery_pct_20d']
def fwd_max(s):
    return s.shift(-1)[::-1].rolling(15, min_periods=8).max()[::-1]
px['fwd_hi'] = g['high'].transform(fwd_max)
px['touch'] = (px['fwd_hi']/px['close'] - 1 >= 0.05)

px['rsi_z'] = (px['rsi_14_daily'] - px['rsi_mu'])/px['rsi_sd']
px['r20_z'] = px['return_20d']/(px['dvol']*np.sqrt(20))
px['r5_z']  = px['ret_5d']/(px['dvol']*np.sqrt(5))

days = sorted(px['trade_date'].unique())
weekly = set(days[::5])
base = px[(px['adv']>=5) & (px['close']>50) & (px['volume_vs_20d']<2) &
          (px['trade_date'].isin(weekly)) & px['fwd_hi'].notna()].copy()

ABS = (base['rsi_14_daily'].between(42,60) & base['return_20d'].between(-0.10,0.15)
       & base['ret_5d'].between(-0.05,0.05))
Z   = (base['rsi_z'].between(-1.0,0.25) & base['r20_z'].between(-1.0,0.75)
       & base['r5_z'].between(-1.0,1.0))

print("\n=== POOL-LEVEL TOUCH RATE (the ship/kill metric) ===", flush=True)
def pool_stats(mask, label, since=None):
    s = base[mask]
    if since is not None: s = s[s['trade_date'] >= since]
    wk = s.groupby('trade_date').agg(n=('touch','size'), t=('touch','mean'))
    return f"{label:<22s} pool/wk {wk['n'].mean():6.1f} · touch {s['touch'].mean()*100:5.1f}% · weekly-median touch {wk['t'].median()*100:5.1f}%"
for since, tag in [(None, '10-YEAR'), (pd.Timestamp('2026-04-01'), 'SINCE-APRIL')]:
    print(f"-- {tag} --")
    print(pool_stats(ABS, 'ABS bands (prod)', since))
    print(pool_stats(Z,   'Z-SCORE bands', since))
    both = base[ABS & Z]; onlyz = base[Z & ~ABS]; onlya = base[ABS & ~Z]
    if since is not None:
        both, onlyz, onlya = [d[d['trade_date']>=since] for d in (both,onlyz,onlya)]
    print(f"{'overlap':<22s} both {both['touch'].mean()*100:.1f}% (n={len(both):,}) · Z-only {onlyz['touch'].mean()*100:.1f}% (n={len(onlyz):,}) · ABS-only {onlya['touch'].mean()*100:.1f}% (n={len(onlya):,})")

print("\n=== TOP-8/WK C2 REALIZED (day-by-day, next-open, 0.30% cost) ===", flush=True)
sf = {s: gg.reset_index(drop=True) for s, gg in px.groupby('symbol')}
def c2(sym, d):
    gg = sf[sym]
    sig = gg[gg['trade_date']==d]; fut = gg[gg['trade_date']>d].head(15)
    if len(sig)==0 or len(fut)<8 or pd.isna(fut.iloc[0]['open']): return None
    ep = float(fut.iloc[0]['open']); dv = float(sig.iloc[0]['dvol']) if pd.notna(sig.iloc[0]['dvol']) else 0.02
    tgt = ep*1.05; sl = ep*(1-np.clip(3*dv,0.03,0.12)); half=False; trail=None
    for i in range(len(fut)):
        r=fut.iloc[i]; lo,hi=r['low'],r['high']
        if pd.isna(lo): continue
        if not half:
            if lo<=sl: return (sl/ep-1)*100
            if pd.notna(hi) and hi>=tgt: half=True; trail=ep*1.025; continue
        else:
            if lo<=trail: return ((0.05+(trail/ep-1))/2)*100
    last=float(fut.iloc[-1]['close'])
    return ((0.05+(last/ep-1))/2)*100 if half else (last/ep-1)*100

for mask, label in [(ABS,'ABS'), (Z,'Z')]:
    picks = base[mask].sort_values(['trade_date','dlv'], ascending=[True,False]).groupby('trade_date').head(8)
    rets=[]
    for _, r in picks.iterrows():
        v = c2(r['symbol'], r['trade_date'])
        if v is not None: rets.append((r['trade_date'], v))
    rr = pd.DataFrame(rets, columns=['d','ret'])
    for since, tag in [(None,'10yr'), (pd.Timestamp('2026-04-01'),'sinceApr')]:
        s = rr if since is None else rr[rr['d']>=since]
        net = s['ret'].mean()-0.30
        print(f"  {label:<4s} [{tag:>8s}] n={len(s):,}  net/trade {net:+.3f}%  CAGR~17cyc {((1+net/100)**17-1)*100:+.1f}%", flush=True)
print("A/B COMPLETE", flush=True)
