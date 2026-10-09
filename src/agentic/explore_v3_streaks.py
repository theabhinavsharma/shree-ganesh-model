import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src/agentic")
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier  # noqa: E402

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
wk = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp("2019-01-01")]
V = v3_rule.picks(X["F"], wk, imap, P, ctx)
F = X["F"].set_index(["symbol", "trade_date"])[["close", "lo252", "sma_200", "ret60", "ret252"]]
S = ctx["scores"][["date", "industry", "heat_pct"]].copy(); S["date"] = pd.to_datetime(S["date"])
heat = {(d, i): h for d, i, h in zip(S["date"], S["industry"], S["heat_pct"])}
Ps = P.set_index(["symbol", "trade_date"])["ensemble"]
PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
On, Hn, Cn = O.to_numpy(), Hh.to_numpy(), C.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(O.columns)}

# per (week, stock): streak, first-week entry, rank history, features known at the list date
streak, first, rank1, prevc, rows = {}, {}, {}, {}, []
for j, d in enumerate(wk):
    cur = V.get(d, [])
    st = {s: streak.get(s, 0) + 1 for s in cur}
    first = {s: (first[s] if s in first and streak.get(s, 0) > 0 else d) for s in cur}
    rank1 = {s: (rank1[s] if s in rank1 and streak.get(s, 0) > 0 else r + 1) for r, s in enumerate(cur)}
    streak = st
    di = cal.get_loc(d); i0 = di + 1
    inds = [imap.get(s) for s in cur]
    for r, s in enumerate(cur):
        c = col.get(s)
        if c is None or i0 >= len(cal):
            continue
        f0 = cal.get_loc(first[s]) + 1
        e1 = On[f0, c]
        f = F.loc[(s, d)] if (s, d) in F.index else None
        sc = Ps.get((s, d), np.nan)
        sc1 = Ps.get((s, first[s]), np.nan)
        fwd = hit = np.nan
        if i0 + 126 <= len(cal) and np.isfinite(On[i0, c]) and On[i0, c] > 0 and np.isfinite(Cn[i0 + 125, c]):
            fwd = Cn[i0 + 125, c] / On[i0, c] - 1; hit = float(np.nanmax(Hn[i0:i0 + 126, c]) >= 1.5 * On[i0, c])
        rows.append(dict(j=j, d=d, s=s, k=streak[s], rank=r + 1, rank_change=(r + 1) - rank1[s],
                         runup=(Cn[di, c] / e1 - 1) if np.isfinite(e1) and e1 > 0 else np.nan,
                         wk_chg=(Cn[di, c] / prevc[s] - 1) if s in prevc and prevc[s] > 0 else np.nan,
                         score=sc, score_chg=sc - sc1 if np.isfinite(sc) and np.isfinite(sc1) else np.nan,
                         heat=heat.get((d, imap.get(s)), np.nan), same_ind=sum(x == imap.get(s) for x in inds) - 1,
                         ret60=f["ret60"] if f is not None else np.nan, ret252=f["ret252"] if f is not None else np.nan,
                         above_sma200=(f["close"] / f["sma_200"] - 1) if f is not None and f["sma_200"] else np.nan,
                         above_lo252=(f["close"] / f["lo252"] - 1) if f is not None and f["lo252"] else np.nan,
                         fwd=fwd, hit=hit, entry_runup=(On[i0, c] / e1 - 1) if np.isfinite(e1) and e1 > 0 and np.isfinite(On[i0, c]) else np.nan))
    prevc = {s: Cn[di, col[s]] for s in cur if s in col}
R = pd.DataFrame(rows); R["era"] = np.where(R.d < "2023-01-01", "2019-22", "2023+")
# final streak length of the run each row belongs to
R = R.sort_values(["s", "j"]); R["run_id"] = (R.groupby("s").j.diff() != 1).cumsum(); R["run_len"] = R.groupby("run_id").k.transform("max")
R["wkb"] = R.k.clip(upper=16).astype(int)

print("=== 2) weeks 8..15 and 16+ on the list · already gained since the 1st-week entry (entry open) · next 6 months from this week · all years")
G = R[R.k >= 8].dropna(subset=["fwd"])
t = G.groupby("wkb").apply(lambda g: pd.Series(dict(picks=len(g), stocks=g.s.nunique(), gained_so_far_med=g.entry_runup.median() * 100, next6m_avg=g.fwd.mean() * 100,
                                                    next6m_med=g.fwd.median() * 100, hit50=g.hit.mean() * 100, down30=(g.fwd <= -0.3).mean() * 100,
                                                    next6m_avg_2023=g[g.era == "2023+"].fwd.mean() * 100, picks_2023=(g.era == "2023+").sum())))
print(t.round(1).to_string())

print("\n=== 3) week-8+ picks: what they had ALREADY gained when bought, split by whether they then hit +50%")
for w in (8, 10, 12):
    g = G[G.k == w] if w == 8 else G[(G.k >= w) & (G.k < w + 2)]
    lab = f"week {w}" if w == 8 else f"weeks {w}-{w + 1}"
    for h, gg in g.groupby("hit"):
        print(f"  {lab} {'HITTERS' if h else 'others '}: n {len(gg):3d} · gained so far median {gg.entry_runup.median()*100:+.1f}% · next 6m avg {gg.fwd.mean()*100:+.1f}% median {gg.fwd.median()*100:+.1f}%")
H = G[G.hit == 1]; N_ = G[G.hit == 0]
print(f"  all week 8+: HITTERS n {len(H)} gained so far median {H.entry_runup.median()*100:+.1f}% avg {H.entry_runup.mean()*100:+.1f}% · next 6m avg {H.fwd.mean()*100:+.1f}%"
      f" | others n {len(N_)} gained so far median {N_.entry_runup.median()*100:+.1f}% · next 6m avg {N_.fwd.mean()*100:+.1f}%")

print("\n=== 1) why 2023+ long streaks kept running: share of runs that reach week 8, and weeks 1-7 vs 8+ by era")
for era, g in R.groupby("era"):
    runs = g.drop_duplicates("run_id")
    print(f"  {era}: runs {len(runs)} · reach week 8: {(runs.run_len >= 8).mean()*100:.1f}% · reach 12: {(runs.run_len >= 12).mean()*100:.1f}% · "
          f"next 6m avg weeks 1-7 {g[g.k < 8].fwd.mean()*100:+.1f}% vs 8+ {g[g.k >= 8].fwd.mean()*100:+.1f}% · down30 8+ {(g[g.k >= 8].fwd <= -0.3).mean()*100:.1f}%")

print("\n=== 4) can weeks 2/3/4 predict reaching week 8?")
feats = ["runup", "wk_chg", "rank", "rank_change", "score", "score_chg", "heat", "same_ind", "ret60", "ret252", "above_sma200", "above_lo252"]
last_j = R.j.max()
for kk in (2, 3, 4):
    A = R[(R.k == kk) & (R.j + (8 - kk) <= last_j)].copy()
    A["y"] = (A.run_len >= 8).astype(int)
    print(f"\n -- at week {kk}: {len(A)} picks · reach week 8: {A.y.mean()*100:.1f}% (2019-22 {A[A.era=='2019-22'].y.mean()*100:.1f}%, 2023+ {A[A.era=='2023+'].y.mean()*100:.1f}%)")
    for fe in feats:
        a = A.dropna(subset=[fe])
        if a[fe].nunique() < 3:
            continue
        q = pd.qcut(a[fe].rank(method="first"), 3, labels=["low", "mid", "high"])
        rates = a.groupby(q, observed=True).y.mean() * 100
        print(f"    {fe:13s} reach-8 rate by tercile low/mid/high: {rates.get('low', np.nan):5.1f} / {rates.get('mid', np.nan):5.1f} / {rates.get('high', np.nan):5.1f}"
              f"   (cut points {a[fe].quantile(1/3):.3g}, {a[fe].quantile(2/3):.3g})")
    tr, te = A[A.era == "2019-22"].copy(), A[A.era == "2023+"].copy()
    med = tr[feats].median()
    Xtr, Xte = tr[feats].fillna(med), te[feats].fillna(med)
    lr = LogisticRegression(max_iter=2000).fit((Xtr - Xtr.mean()) / Xtr.std(), tr.y)
    p_lr = lr.predict_proba((Xte - Xtr.mean()) / Xtr.std())[:, 1]
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xtr, tr.y)
    p_gb = gb.predict_proba(Xte)[:, 1]
    print(f"    trained on 2019-22, tested on 2023+: AUC logistic {roc_auc_score(te.y, p_lr):.3f} · boosted trees {roc_auc_score(te.y, p_gb):.3f} (0.5 = coin flip)")
    imp = pd.Series(gb.feature_importances_, index=feats).sort_values(ascending=False).head(5)
    print("    most used signals: " + ", ".join(f"{k} {v:.2f}" for k, v in imp.items()))
    te["p"] = p_gb
    te["tier"] = pd.qcut(te.p.rank(method="first"), 3, labels=["least likely", "middle", "most likely"])
    o = te.dropna(subset=["fwd"]).groupby("tier", observed=True).agg(picks=("y", "size"), reached8=("y", lambda x: x.mean() * 100), next6m_avg=("fwd", lambda x: x.mean() * 100),
                                                                       next6m_med=("fwd", lambda x: x.median() * 100), hit50=("hit", lambda x: x.mean() * 100), down30=("fwd", lambda x: (x <= -0.3).mean() * 100))
    print("    2023+ picks at this week, by predicted chance of reaching week 8 (out of sample):")
    print("    " + o.round(1).to_string().replace("\n", "\n    "))
