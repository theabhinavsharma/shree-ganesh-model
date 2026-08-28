"""Assemble reports/doubler_audit_20260828.md from the audit artifacts."""
import json
import pandas as pd
from pathlib import Path

R = Path('research/doubler_audit')
d = pd.read_csv(R/'doublers_2x_anytime.csv', parse_dates=['trough_date','hit2x_date'])
inv = d[d['investable']].sort_values('peak_mult_from_apr1', ascending=False)
flags = json.load(open(R/'point_in_time_flags.json'))
doss = json.load(open(R/'dossiers_full.json'))
dmap = {x['symbol']: x for x in doss['dossiers']}
synth = doss['synthesis']
miner = Path('logs/mine_doubler_ignition.log').read_text()

flagged_by = {}
for tag, syms in flags['flagged'].items():
    for s in syms: flagged_by.setdefault(s, []).append(tag)
picked_by = {}
for date, syms in flags['picked'].items():
    for s in syms: picked_by.setdefault(s, []).append(date)

L = []
L.append("# Doubler Audit — every investable 2x since April 2026\n")
L.append(f"_Generated 2026-08-28 on the CA-repaired panel. Artifacts: `research/doubler_audit/`._\n")
L.append("## 1. Funnel\n")
L.append(f"- NSE names with ≥20 sessions since Apr-1: **2,884**")
L.append(f"- Touched 2x from their April-1 close at any point: **202** (8.2%)")
L.append(f"- Touched 2x from ANY post-April trough: **{len(d)}**")
L.append(f"- …of which investable at the trough (ADV≥5cr, >₹50): **{len(inv)}**")
L.append(f"- Median trading days trough→2x: **{int(inv['days_to_2x'].median())}** · 44% of ALL NSE touched 1.5x\n")
L.append("## 2. Attribution — what our models said at the time\n")
L.append("| where | flagged | became doublers |\n|---|---|---|")
L.append("| hc @2026-04-30 (git) | 10 | **5** — APOLLO CUPID DEEDEV HIRECT IDEAFORGE |")
L.append("| hc @2026-05-04 (git) | 10 | 3 — MANINDS MTARTECH OPTIEMUS |")
L.append("| mb @Apr-30/May-04 (git) | 1 | 0 |")
L.append("| 180d @Apr-30 (git) | 13 | 0 (its July hits were post-facto) |")
L.append("| 15d baskets since April | 62 | 4 — BALAMINES JNKINDIA NELCO RAIN (traded as +5% slices) |")
L.append("| **never flagged anywhere** | — | **65 of 78** |\n")
L.append("## 3. Catalysts (from 78 dossiers)\n")
cf = {k: v for k, v in synth['catalyst_freq'].items() if not k.startswith('_')}
L.append("| catalyst | n |\n|---|---|")
for k, v in sorted(cf.items(), key=lambda x: -x[1]): L.append(f"| {k} | {v} |")
L.append("\n**Themes**: " + " · ".join(f"{t['theme']} ({t['count']})" for t in synth['themes'][:8]))
L.append("\n**Knowability**: " + synth['knowability'] + "\n")
L.append("## 4. Ignition signatures\n")
for i in synth['ignition_signatures']:
    L.append(f"- **[{i['count']}]** {i['signature']}")
L.append("\nTape stats: 78% printed a ≥+5% day on ≥1.5× vol within 15 sessions of trough; "
         "median first thrust session 5, price +13.9% off low, **median +76% still left to 2x after the thrust close**.\n")
L.append("## 5. Walk-forward miner (pre-registered cells, 10yr disc≤2022 / conf≥2023)\n")
L.append("```\n" + miner.split('loading…')[-1].strip() + "\n```\n")
L.append("Announcement-anchored cells (post-results thrust, order-win cadence, results-proximity drift) "
         "are FORWARD-TRACKED hypotheses only — the announcements store starts 2026-02, so any backtest "
         "on them would be circular.\n")
L.append("## 6. Per-stock catalog (all 78, by peak multiple)\n")
L.append("| # | symbol | trough | 2x date | td | peak | catalyst | theme | flagged/picked | knowable pre-run |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
for i, (_, r) in enumerate(inv.iterrows(), 1):
    s = r['symbol']; x = dmap.get(s, {})
    fl = ' '.join(flagged_by.get(s, [])) or ('basket:' + picked_by[s][0] if s in picked_by else '—')
    if s in picked_by and s in flagged_by: fl += ' +basket'
    kn = (x.get('knowable_pre_run','') or '')[:140].replace('|','/')
    L.append(f"| {i} | {s} | {r['trough_date'].date()} | {r['hit2x_date'].date()} | {r['days_to_2x']} | "
             f"{r['peak_mult_from_apr1']:.2f}x | {x.get('catalyst_type','?')} | {x.get('theme','?')} | {fl} | {kn} |")
L.append("\n## 7. Full dossiers\n")
for _, r in inv.iterrows():
    s = r['symbol']; x = dmap.get(s)
    if not x: continue
    L.append(f"### {s} — {x.get('catalyst_type')} · {x.get('theme')}")
    L.append(f"- **Drove the 2x**: {x.get('catalyst_summary')}")
    L.append(f"- **Knowable pre-run**: {x.get('knowable_pre_run')}")
    L.append(f"- **Ignition**: {x.get('ignition_signature')}\n")

out = Path('reports/doubler_audit_20260828.md')
out.write_text('\n'.join(L))
print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB)")
