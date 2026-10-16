"""Exploration (2026-10-16, not a registered test): does a different model / rule do better in a certain kind of market?

Plain English: for every weekly list since Apr 2019 (schedule 0), each arm's 9-name list is held 6 months (after
liquidity-scaled costs, equal weight) and compared with the equal-weight liquid market over the same 6 months. Each list
date is tagged with the market regime known that day:
  trend     equal-weight liquid index above / below its 200-day average
  breadth   share of liquid stocks above their own 200-day average >= / < 50%
  vix       India VIX (day before) above / below its trailing 1-year median
  fpi       foreign portfolio investors' 60-session net flow on exchanges (share of their turnover) buying / selling
Arms: V3.2 (live), V3.1, V3 (Rs 5 cr bar), NOGATE (V3.2 without the hot-industry gate), MIX720 (7 hot + 2 mid-heat
industries), and V3.2's pool ranked by each single model instead of the ensemble: LOGIT, LGBM, XGB, MLP.
Per regime: mean 6-month return, gap vs V3.2 with a 26-week block bootstrap 95% range, and whether the gap has the same
sign in 2019-22 and 2023+ (a real regime edge should). Hindsight-free only in the sense that regimes and lists use data
known at the list date; the comparison itself is in-sample. Anything worth acting on needs a registered walk-forward test.
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
OUT = ROOT / "reports/models_by_regime"; START, HOLD = pd.Timestamp("2019-04-01"), 126


def boot_gap(x, block=26, n=4000, seed=7):
    x = np.asarray(x, float); T = len(x)
    if T < 2 * block:
        return np.nan, np.nan
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / block))
    out = [x[((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]).ravel()[:T]) % T].mean() for _ in range(n)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]; T = len(cal)
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); tpos = {d: i for i, d in enumerate(cal)}; syms = D["syms"]
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    L["liq"] = ((L["adv"] >= 1) & (rp.raw_price(L, "close") > 50)).astype(float)
    ADV = rp.wide(L, "adv", cal).reindex(columns=syms).to_numpy(); LIQ = rp.wide(L, "liq", cal).reindex(columns=syms).fillna(0).to_numpy() > 0; del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    keep = lambda s, d: ctx["keep"](s, d) and not t2t(s, d)  # noqa: E731
    Rv = np.nan_to_num(D["Rv"]); lg = np.vstack([np.zeros((1, Rv.shape[1])), np.cumsum(np.log1p(Rv), axis=0)]); vr = D.get("vr", {})
    weeks = [d for d in sp.weekly_grid(cal, 0) if d >= START and tpos[d] + HOLD <= T - 1]

    # ---- arms
    Sc = ctx["scores"]; ok = ~(Sc["P_pct"] < 0.30); G1 = ctx["G1"]
    allind = pd.DataFrame([(d, i) for d in weeks for i in imap.dropna().unique()], columns=["date", "industry"])
    MID = Sc[ok & (Sc["heat_pct"] >= 0.40) & (Sc["heat_pct"] < 0.70)]
    fin = lambda R: {d: [s for s in R[d] if keep(s, d)] for d in weeks}  # noqa: E731
    A = {"V3.2": fin(v3_rule.ranked_v31(F1, weeks, imap, P, G1, 9)),
         "V3.1": {d: [s for s in l if ctx["keep"](s, d)] for d, l in v3_rule.ranked_v31(F1, weeks, imap, P, G1, 9).items()},
         "V3": v3_rule.picks(X["F"], weeks, imap, P, ctx),
         "NOGATE": fin(v3_rule.ranked_v31(F1, weeks, imap, P, allind, 9))}
    hot, mid = v3_rule.ranked_v31(F1, weeks, imap, P, G1, 9), v3_rule.ranked_v31(F1, weeks, imap, P, MID, 2)
    A["MIX720"] = {d: [s for s in hot[d][:7 + 2 - len(mid[d])] + mid[d] if keep(s, d)] for d in weeks}
    Pm = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/bakeoff_preds.parquet", columns=["symbol", "trade_date", "logit", "lgbm", "xgb", "mlp"])
    Pm["trade_date"] = pd.to_datetime(Pm["trade_date"]).astype("datetime64[ns]")
    for m in ("logit", "lgbm", "xgb", "mlp"):
        Pk = Pm[["symbol", "trade_date", m]].rename(columns={m: "ensemble"}).dropna().sort_values("trade_date")
        A[m.upper()] = fin(v3_rule.ranked_v31(F1, weeks, imap, Pk, G1, 9))
    ref = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, weeks, imap, P, ctx).items()}
    same = sum(A["V3.2"][d] == ref[d] for d in weeks); print(f"wiring: V3.2 arm == v3_rule V3.2 on {same}/{len(weeks)} weeks", flush=True)
    if same < len(weeks):
        raise SystemExit("wiring check FAILED")

    def r6(s, d):
        q = D["cidx"].get(s); i0 = tpos[d]
        if q is None:
            return np.nan
        g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
        for k, f in vr.get(q, ()):
            if i0 + 1 <= k <= i0 + HOLD - 1:
                g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
        return ((1 - float(cost(i0, np.array([q]))[0])) * g - 1) * 100

    # ---- regimes known at d
    dm = np.array([Rv[i][LIQ[i - 1]].mean() if i > 0 and LIQ[i - 1].any() else 0.0 for i in range(T)])
    idx = pd.Series(np.cumprod(1 + dm), index=cal); trend = idx / idx.rolling(200, min_periods=200).mean() - 1
    Fw = F1[F1["trade_date"].isin(set(weeks)) & F1["core"] & F1["sma_200"].notna()]
    breadth = (Fw["close"] > Fw["sma_200"]).groupby(Fw["trade_date"]).mean()
    v = pd.read_parquet(ROOT / "data/derived/india_vix.parquet"); v["date"] = pd.to_datetime(v["date"]); v = v.set_index("date")["vix_close"].sort_index()
    vrel = v / v.rolling(252, min_periods=200).median()
    Fp = pd.read_parquet(ROOT / "data/derived/nsdl_fpi_daily.parquet"); se = Fp[(Fp["asset"] == "Equity") & (Fp["route"] == "Stock Exchange")].copy()
    se["known_from"] = pd.to_datetime(se["known_from"]); se = se.sort_values("known_from"); g = se["gross_purchases_cr"] + se["gross_sales_cr"]
    fpi = pd.Series((se["net_cr"].rolling(60, min_periods=48).sum() / g.rolling(60, min_periods=48).sum()).to_numpy(), index=se["known_from"]).groupby(level=0).last()
    asof = lambda s, d: s.loc[:d].iloc[-1] if len(s.loc[:d]) else np.nan  # noqa: E731
    rows = []
    for d in weeks:
        i = tpos[d]; mk = LIQ[i]
        mret = float(np.mean(np.exp(lg[i + HOLD + 1, mk] - lg[i + 1, mk]) - 1) * 100)
        reg = dict(trend="up" if trend.loc[d] > 0 else "down", breadth="wide" if breadth.get(d, np.nan) >= 0.5 else "narrow",
                   vix="high" if asof(vrel, d - pd.Timedelta(days=1)) > 1 else "low", fpi="buying" if asof(fpi, d) > 0 else "selling")
        row = dict(d=d, era="2019-22" if d.year <= 2022 else "2023+", market=mret, **reg)
        for a, pk in A.items():
            rs = [r6(s, d) for s in pk[d]]; rs = [x for x in rs if np.isfinite(x)]
            row[a] = float(np.mean(rs)) if rs else 0.0                   # empty list = cash (0%)
        rows.append(row)
    W = pd.DataFrame(rows); arms = list(A)

    # ---- per regime
    out = []
    for var in ("trend", "breadth", "vix", "fpi"):
        for state, sub in W.groupby(var):
            for a in arms:
                gap = (sub[a] - sub["V3.2"]).to_numpy(); lo, hi = boot_gap(gap) if a != "V3.2" else (np.nan, np.nan)
                e = {k: float((sub[sub["era"] == k][a] - sub[sub["era"] == k]["V3.2"]).mean()) if (sub["era"] == k).any() else np.nan for k in ("2019-22", "2023+")}
                out.append(dict(regime=var, state=state, weeks=len(sub), arm=a, avg6m=round(sub[a].mean(), 1), vs_market=round((sub[a] - sub["market"]).mean(), 1),
                                gap_vs_v32=round(gap.mean(), 1), lo=round(lo, 1), hi=round(hi, 1), gap_2019_22=round(e["2019-22"], 1), gap_2023=round(e["2023+"], 1),
                                both_eras_same_sign=bool(np.sign(e["2019-22"]) == np.sign(e["2023+"]) and e["2019-22"] != 0)))
    R = pd.DataFrame(out)
    print(f"\n=== models by regime · {len(W)} weekly lists {weeks[0].date()}..{weeks[-1].date()} (6-month outcome known) ===")
    print("all weeks, avg 6M %: " + " · ".join(f"{a} {W[a].mean():.1f}" for a in arms) + f" · market {W['market'].mean():.1f}")
    for (var, state), sub in R.groupby(["regime", "state"], sort=False):
        n = int(sub["weeks"].iloc[0]); best = sub.sort_values("avg6m", ascending=False).iloc[0]
        print(f"\n{var}={state} ({n} weeks, ~{n / 26:.0f} independent 6M periods; market {W[W[var] == state]['market'].mean():.1f}%) · best {best['arm']} {best['avg6m']}%")
        for r in sub.itertuples():
            sig = "" if r.arm == "V3.2" else (f" gap {r.gap_vs_v32:+.1f} (95% {r.lo:+.1f}..{r.hi:+.1f})" + (" *" if (r.lo > 0 or r.hi < 0) else "") + f" · 2019-22 {r.gap_2019_22:+.1f} / 2023+ {r.gap_2023:+.1f}" + (" same sign" if r.both_eras_same_sign else ""))
            print(f"  {r.arm:7s} {r.avg6m:6.1f}%{sig}")
    OUT.mkdir(parents=True, exist_ok=True); now = datetime.now().isoformat(timespec="seconds")
    W.to_csv(OUT / "weekly_lists.csv", index=False); R.to_csv(OUT / "by_regime.csv", index=False)
    for f, what in (("weekly_lists.csv", "one row per weekly list (schedule 0): regime tags and each arm's equal-weight 6-month return after costs"),
                    ("by_regime.csv", "per regime state and arm: average 6-month return, gap vs V3.2 with a 26-week block bootstrap range, gap by era")):
        (OUT / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, producer="src/agentic/explore_models_by_regime.py", kind="exploration (not a registered test)",
            definitions=__doc__, contents=what, units=dict(avg6m="percent, 6-month return after costs", market="percent, equal-weight liquid stocks, no costs"), updated=now), indent=1))
    (OUT / "README.md").write_text("# Models by market regime (exploration)\n\nDoes a different model or rule do better in a certain kind of market? "
                                   "See the run log for the plain-English summary. weekly_lists.csv, by_regime.csv + manifests. Exploration only: any rule built on this needs a registered walk-forward test.\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
