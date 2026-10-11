"""Exploration (not a registered test; changes no rule): V3.1 picks year by year, and why 2025 went wrong.

V3.1 = the rule invested from 2026-10-12: V3 with the Rs 1 cr liquidity bar (price > Rs 50 kept) and new listings aged
63-251 sessions skipping the trend test (same steps as test_v3_l1_newlist_cost.nl_select). V3 (Rs 5 cr bar, no new
listings) is run through the same code as the reference; wiring check: it must equal v3_rule.picks on every week.
Per pick (phase-0 weekly lists from 2023-01-01): bought at the next session's open. 'hit' = the day's high reached +50%
within 126 sessions; '6m return' = close 126 sessions after entry vs entry; 'market' = plain average of the same 6m return
over every stock that cleared the Rs 1 cr / Rs 50 bar on that list date (equal-weight market).
Funnel: every stock-week that cleared the Rs 1 cr bar, tagged with the first V3.1 step that dropped it.
Hits and returns count only entries with a full 126-session window; later 2026 entries are reported to date.
Top-N sweep: V3.1 baskets cut to the first N names by model rank (N = 1..9), same engine and liquidity-scaled costs as
test_v3_l1_newlist_cost (run_cost imported from it), weekly entries from 2019, 5 schedules. Score gradient: pick outcomes by
rank position and by model-score fifth. Both are look-backs on data already seen: any N or cutoff they suggest must be
registered and tested before it changes the rule.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402

OUTDIR = ROOT / "logs/leader_sleeve/v31_hits_by_year"
START, HOLD = pd.Timestamp("2019-01-01"), 126
STAGES = ["no industry / short history", "industry not hot", "failed trend test", "ranked below top 9", "financial / fading filter", "picked"]


def main() -> None:
    t0 = time.time()
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
    ctx = v3_rule.context(imap)
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    F = X["F"].merge(L[["symbol", "trade_date", "raw", "adv"]], on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F["age"] = F.groupby("symbol").cumcount()
    F["young_nl"] = F["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren) & F["age"].between(63, 251)
    wk = [d for d in sp.weekly_grid(cal, 0) if d >= START]
    hot = set(zip(pd.to_datetime(ctx["G1"]["date"]), ctx["G1"]["industry"]))

    def funnel(bar: float, newlist: bool, weeks: list) -> pd.DataFrame:
        W = F[F["trade_date"].isin(set(weeks)) & (F["adv"] >= bar) & (F["raw"] > 50)].copy()
        W["ind"] = W["symbol"].map(imap)
        W["eligible"] = W["ind"].notna() & W["ret60"].notna()
        W["hot"] = [(d, i) in hot for d, i in zip(W["trade_date"], W["ind"])]
        with np.errstate(invalid="ignore", divide="ignore"):
            W["c_lo"] = W["close"] / W["lo252"] - 1 >= 0.5; W["c_r252"] = W["ret252"] >= 0.30
            W["c_200"] = W["close"] > W["sma_200"]; W["c_50"] = W["sma_50"] > W["sma_200"]
        W["young"] = W["young_nl"] & newlist
        W["trend"] = (W["c_lo"] & W["c_r252"] & W["c_200"] & W["c_50"]) | W["young"]
        Lx = W[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        W["score"] = m.set_index("index")["ensemble"].reindex(W.index).fillna(-np.inf)
        pool = W[W["eligible"] & W["hot"] & W["trend"]].sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        W["rank"] = pool.groupby("trade_date").cumcount().add(1).reindex(W.index)
        W["pool_n"] = W["trade_date"].map(pool.groupby("trade_date").size())
        W["top9"] = W["rank"].le(9)
        W["keep"] = [bool(t) and ctx["keep"](s, d) for s, d, t in zip(W["symbol"], W["trade_date"], W["top9"])]
        W["stage"] = np.select([~W["eligible"], ~W["hot"], ~W["trend"], ~W["top9"], ~W["keep"]], STAGES[:5], STAGES[5])
        return W

    V31, V3 = funnel(1.0, True, wk), funnel(5.0, False, wk)
    live = v3_rule.picks(X["F"], wk, imap, P, ctx)
    mine = V3[V3["stage"] == "picked"].sort_values(["trade_date", "rank"]).groupby("trade_date")["symbol"].apply(list).to_dict()
    same = sum(mine.get(d, []) == live.get(d, []) for d in wk)
    print(f"wiring check: V3 through this code equals v3_rule.picks on {same}/{len(wk)} weeks", flush=True)
    if same < len(wk):
        raise SystemExit("wiring check FAILED")
    rec = ROOT / "logs/sri_lakshmi/screen_20261009.json"
    if rec.exists() and pd.Timestamp("2026-10-09") in set(wk):
        r = json.loads(rec.read_text()); want = [p["symbol"] for p in r.get("picks", [])]
        got = V31[(V31["trade_date"] == "2026-10-09") & (V31["stage"] == "picked")].sort_values("rank")["symbol"].tolist()
        print(f"wiring check: V3.1 list for 2026-10-09 {'matches' if got == want else 'DIFFERS from'} the batch record ({len(got)} names)", flush=True)

    # ---- outcomes
    PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
    On, Cn = O.to_numpy(), C.ffill(limit=300).to_numpy()
    Hf = Hh.iloc[::-1].rolling(HOLD, min_periods=1).max().iloc[::-1].to_numpy()
    col = {s: i for i, s in enumerate(O.columns)}; pos = {d: i for i, d in enumerate(cal)}; T = len(cal)

    def outcomes(W: pd.DataFrame) -> pd.DataFrame:
        e = W["trade_date"].map(pos).to_numpy() + 1; c = W["symbol"].map(col).fillna(-1).astype(int).to_numpy()
        ok = (c >= 0) & (e < T); e2, c2 = np.where(ok, e, 0), np.where(ok, c, 0)
        entry = np.where(ok, On[e2, c2], np.nan)
        W["valid"] = np.isfinite(entry) & (entry > 0)
        W["full"] = W["valid"] & (e + HOLD - 1 <= T - 1)
        W["peak"] = Hf[e2, c2] / entry - 1
        W["ret"] = Cn[np.minimum(e2 + HOLD - 1, T - 1), c2] / entry - 1
        W["hit"] = W["peak"] >= 0.5
        return W
    V31, V3 = outcomes(V31), outcomes(V3)
    mk = V31[V31["full"]].groupby("trade_date").agg(mkt_ret=("ret", "mean"), mkt_med=("ret", "median"), mkt_hit=("hit", "mean"), n=("ret", "size"))
    for W in (V31, V3):
        W["mkt_ret"] = W["trade_date"].map(mk["mkt_ret"])

    def year_rows(W: pd.DataFrame) -> list:
        out = []
        Pk = W[(W["stage"] == "picked") & W["valid"]]
        for y, g in Pk.groupby(Pk["trade_date"].dt.year):
            f = g[g["full"]]; o = g[~g["full"]]
            hb = f.groupby("trade_date")["hit"].sum(); mky = mk[mk.index.year == y]
            out.append(dict(year=int(y), weeks=int(g["trade_date"].nunique()), picks=len(g), per_basket=round(len(g) / g["trade_date"].nunique(), 1), stocks=int(g["symbol"].nunique()),
                            full_weeks=int(f["trade_date"].nunique()), hit_pct=round(100 * f["hit"].mean(), 1) if len(f) else None,
                            hitters_per_basket=round(hb.mean(), 2) if len(hb) else None, baskets_zero_hit=round(100 * (hb == 0).mean(), 0) if len(hb) else None,
                            baskets_3plus=round(100 * (hb >= 3).mean(), 0) if len(hb) else None,
                            avg6m=round(100 * f["ret"].mean(), 1) if len(f) else None, med6m=round(100 * f["ret"].median(), 1) if len(f) else None,
                            beat_mkt=round(100 * (f["ret"] > f["mkt_ret"]).mean(), 0) if len(f) else None, lost30=round(100 * (f["ret"] <= -0.3).mean(), 0) if len(f) else None,
                            mkt6m=round(100 * mky["mkt_ret"].mean(), 1) if len(mky) else None, mkt_med6m=round(100 * mky["mkt_med"].mean(), 1) if len(mky) else None,
                            mkt_hit=round(100 * mky["mkt_hit"].mean(), 1) if len(mky) else None,
                            open_picks=len(o), open_hit_so_far=round(100 * o["hit"].mean(), 1) if len(o) else None, open_avg_to_date=round(100 * o["ret"].mean(), 1) if len(o) else None))
        return out
    Y31, Y3 = year_rows(V31), year_rows(V3)
    print("\n=== V3.1 picks by entry year (full 126-session windows; market = equal-weight Rs 1 cr universe) ===")
    for r in Y31:
        print(json.dumps(r))
    print("--- V3 (reference) ---")
    for r in Y3:
        print(json.dumps({k: r[k] for k in ("year", "picks", "per_basket", "hit_pct", "hitters_per_basket", "baskets_zero_hit", "avg6m", "med6m", "beat_mkt", "lost30")}))

    # ---- runners
    Pk = V31[(V31["stage"] == "picked") & V31["valid"]]
    runners = {}
    print("\n=== top runners per entry year (best peak within 126 sessions of any week it was listed) ===")
    for y, g in Pk[Pk["trade_date"] >= "2023-01-01"].groupby(Pk["trade_date"].dt.year):
        b = g.sort_values("peak", ascending=False).drop_duplicates("symbol").head(8)
        wks = g.groupby("symbol").size()
        runners[int(y)] = [dict(symbol=s, industry=i, peak=round(100 * p, 0), ret6m=round(100 * r, 0), entry=str(d.date()), weeks_listed=int(wks[s]), new_listing=bool(yg))
                           for s, i, p, r, d, yg in zip(b["symbol"], b["ind"], b["peak"], b["ret"], b["trade_date"], b["young"])]
        print(y, " · ".join(f"{x['symbol']} +{x['peak']:.0f}% (6m {x['ret6m']:+.0f}%, {x['weeks_listed']}w)" for x in runners[int(y)]))

    # ---- entry-month view around 2025
    mo = []
    print("\n=== V3.1 by entry month 2024-07..2026-04 (full windows) ===")
    for mth, g in Pk[(Pk["trade_date"] >= "2024-07-01") & Pk["full"]].groupby(Pk["trade_date"].dt.to_period("M")):
        mm = mk[mk.index.to_period("M") == mth]
        top_ind = g["ind"].value_counts().head(3)
        mo.append(dict(month=str(mth), picks=len(g), hit_pct=round(100 * g["hit"].mean(), 0), avg6m=round(100 * g["ret"].mean(), 1), med6m=round(100 * g["ret"].median(), 1),
                       mkt6m=round(100 * mm["mkt_ret"].mean(), 1), mkt_hit=round(100 * mm["mkt_hit"].mean(), 0), pool=round(float(g.drop_duplicates("trade_date")["pool_n"].mean()), 0),
                       top_industries=", ".join(f"{i} {n}" for i, n in top_ind.items())))
        print(json.dumps(mo[-1]))

    # ---- funnel: hit rate at each step, and where the winners were dropped
    fun = []
    print("\n=== funnel by entry year (Rs 1 cr universe, full windows): stock-weeks surviving and their hit rate ===")
    Fu = V31[V31["full"]].copy(); Fu["k"] = Fu["stage"].map({s: i for i, s in enumerate(STAGES)})
    for y, g in Fu.groupby(Fu["trade_date"].dt.year):
        row = dict(year=int(y))
        for k, lab in enumerate(["all liquid", "has industry + history", "in hot industry", "passed trend / new listing", "model top 9", "picked"]):
            s = g[g["k"] >= k]; row[lab] = f"{len(s)} rows, hit {100 * s['hit'].mean():.1f}%, avg6m {100 * s['ret'].mean():+.1f}%"
        w = g[g["hit"]]; row["winners dropped at"] = {lab: f"{100 * (w['stage'] == lab).mean():.0f}%" for lab in STAGES}
        fun.append(row); print(json.dumps(row))

    # ---- 2025 detail: trend-test failures among winners, model rank, industries
    W25 = Fu[Fu["trade_date"].dt.year == 2025]; w25 = W25[W25["hit"]]
    tf = w25[w25["stage"] == "failed trend test"]
    trend_fail = {lab: round(100 * (~tf[c]).mean(), 0) for c, lab in (("c_lo", "not 50% above 52w low"), ("c_r252", "1y return under 30%"), ("c_200", "below 200-day avg"), ("c_50", "50-day avg below 200-day"))}
    rk = w25[w25["stage"] == "ranked below top 9"]["rank"]
    print(f"\n2025 winners that failed the trend test ({len(tf)} stock-weeks): share failing each condition {trend_fail}")
    print(f"2025 winners in the pool but below top 9: {len(rk)} stock-weeks, median rank {rk.median():.0f} (pool median {W25.drop_duplicates('trade_date')['pool_n'].median():.0f})")
    hot_win = w25.groupby("ind").agg(winner_weeks=("hit", "size"), hot_share=("hot", "mean"), picked=("stage", lambda s: int((s == "picked").sum()))).sort_values("winner_weeks", ascending=False).head(12)
    print("2025 winner stock-weeks by industry (share of those weeks the industry was 'hot'):")
    print(hot_win.assign(hot_share=lambda x: (100 * x["hot_share"]).round(0)).to_string())
    pk25 = W25[W25["stage"] == "picked"].groupby("ind").agg(picks=("hit", "size"), hit_pct=("hit", "mean"), avg6m=("ret", "mean")).sort_values("picks", ascending=False).head(10)
    print("2025 V3.1 picks by industry:")
    print(pk25.assign(hit_pct=lambda x: (100 * x["hit_pct"]).round(0), avg6m=lambda x: (100 * x["avg6m"]).round(1)).to_string())

    # ---- the big 2025 movers and why V3.1 missed them
    yrs = cal.year; i24, i25 = np.where(yrs == 2024)[0][-1], np.where(yrs == 2025)[0][-1]
    liq25 = V31[V31["trade_date"].dt.year == 2025]
    movers = []
    for s, g in liq25.groupby("symbol"):
        c = col.get(s)
        if c is None:
            continue
        base = Cn[i24, c]
        if not np.isfinite(base):
            fin = np.where(np.isfinite(Cn[i24 + 1:i25 + 1, c]))[0]
            if not len(fin):
                continue
            base = Cn[i24 + 1 + fin[0], c]
        r = Cn[i25, c] / base - 1
        win = g[g["full"] & g["hit"]]; vc = g["stage"].value_counts()
        movers.append(dict(symbol=s, industry=g["ind"].iat[0], ret2025=round(100 * r, 0), weeks_liquid=len(g), weeks_hot=int(g["hot"].sum()),
                           weeks_trend=int(g["trend"].sum()), weeks_picked=int(vc.get("picked", 0)),
                           winning_weeks=len(win), main_block=win["stage"].mode().iat[0] if len(win) else (vc.index[0] if len(vc) else None)))
    Mv = pd.DataFrame(movers).sort_values("ret2025", ascending=False)
    show = pd.concat([Mv.head(20), Mv[Mv["symbol"].isin(["ATHERENERG"]) & ~Mv["symbol"].isin(Mv.head(20)["symbol"])]])
    print("\n=== biggest 2025 movers (calendar 2025, or from listing) cleared the Rs 1 cr bar in 2025 ===")
    print(show.to_string(index=False))
    print("main block across top 50 movers:", Mv.head(50)["main_block"].value_counts().to_dict())


    # ---- score gradient: outcome by position in the final list and by model score (phase 0, full windows)
    Pk = V31[(V31["stage"] == "picked") & V31["full"]].sort_values(["trade_date", "rank"]).copy()
    Pk["pos"] = Pk.groupby("trade_date").cumcount() + 1
    Pk["era"] = np.where(Pk["trade_date"] < "2023-01-01", "2019-22", "2023+")
    grad = []
    print("\n=== V3.1 picks by position in the weekly list (1 = model's top) ===")
    for (era, p), g in Pk.groupby(["era", "pos"]):
        grad.append(dict(era=era, pos=int(p), n=len(g), hit_pct=round(100 * g["hit"].mean(), 1), avg6m=round(100 * g["ret"].mean(), 1), med6m=round(100 * g["ret"].median(), 1),
                         beat_mkt=round(100 * (g["ret"] > g["mkt_ret"]).mean(), 0)))
        print(json.dumps(grad[-1]))
    Pk["q"] = pd.qcut(Pk["score"], 5, labels=False) + 1; edges = pd.qcut(Pk["score"], 5, retbins=True)[1]
    sq = []
    print(f"=== V3.1 picks by model-score fifth (1 = lowest; edges {np.round(edges, 3).tolist()}) ===")
    for (era, q), g in Pk.groupby(["era", "q"]):
        sq.append(dict(era=era, fifth=int(q), n=len(g), hit_pct=round(100 * g["hit"].mean(), 1), avg6m=round(100 * g["ret"].mean(), 1), med6m=round(100 * g["ret"].median(), 1)))
        print(json.dumps(sq[-1]))
    for y in (2023, 2024, 2025):
        g = Pk[Pk["trade_date"].dt.year == y]
        print(f"{y}: by fifth hit% " + " ".join(f"{q}:{100 * h:.0f}" for q, h in g.groupby("q")["hit"].mean().items()) + " · avg6m% " + " ".join(f"{q}:{100 * h:+.0f}" for q, h in g.groupby("q")["ret"].mean().items()))

    # ---- top-N sweep (portfolio engine, liquidity-scaled costs, 5 schedules)
    from test_v3_l1_newlist_cost import run_cost
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy()
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    NS, YRS = range(1, 10), (2023, 2024, 2025, 2026)
    sw = {n: [] for n in NS}
    for o in range(5):
        wko = wk if o == 0 else [d for d in sp.weekly_grid(cal, o) if d >= START]
        Wo = V31 if o == 0 else funnel(1.0, True, wko)
        lists = Wo[Wo["stage"] == "picked"].sort_values(["trade_date", "rank"]).groupby("trade_date")["symbol"].apply(list).to_dict()
        for n in NS:
            nav, info = run_cost(D, X, {d: lists.get(d, [])[:n] for d in wko}, wko, "E0", cost)
            m = sp.metrics(nav); nn = nav / nav.iloc[0]; ye = nn.groupby(nn.index.year).last(); r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
            sw[n].append({**{k: m[k] for k in ("cagr", "cagr_disc", "cagr_conf", "maxdd")}, "names": info["avg_names"], **{y: float(r.get(y, np.nan)) for y in YRS}})
        print(f"top-N sweep: schedule {o} done", flush=True)
    top = []
    print("\n=== top-N sweep: V3.1 cut to the first N names (phase 0 CAGR/eras/maxDD; years and range = 5 schedules) ===")
    for n in NS:
        a = sw[n][0]; cg = [x["cagr"] for x in sw[n]]
        beats9 = sum(bool(sp.beats(sw[n][o], sw[9][o])) for o in range(5)) if n != 9 else None
        top.append(dict(N=n, cagr=round(a["cagr"], 1), era_2019_22=round(a["cagr_disc"], 1), era_2023p=round(a["cagr_conf"], 1), maxdd=round(a["maxdd"], 1),
                        cagr_range=f"{min(cg):.1f}..{max(cg):.1f}", worst_maxdd=round(min(x["maxdd"] for x in sw[n]), 1), names=round(a["names"], 1),
                        **{str(y): round(float(np.mean([x[y] for x in sw[n]])), 1) for y in YRS}, beats_N9=beats9))
        print(json.dumps(top[-1]))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cols = ["trade_date", "symbol", "ind", "adv", "young", "score", "rank", "pool_n", "stage", "valid", "full", "peak", "ret", "hit", "mkt_ret"]
    V31[V31["stage"] == "picked"][cols].to_csv(OUTDIR / "v31_picks.csv", index=False)
    Mv.to_csv(OUTDIR / "movers_2025.csv", index=False)
    res = dict(v31_years=Y31, v3_years=Y3, runners=runners, months=mo, funnel=fun, trend_fail_2025=trend_fail, by_position=grad, by_score_fifth=sq, top_n_sweep=top)
    (OUTDIR / "results.json").write_text(json.dumps(res, indent=1, default=str))
    man = dict(producer="src/agentic/explore_v31_hits_by_year.py", kind="exploration, not a registered test", definitions=__doc__,
               units=dict(peak="fraction (0.5 = +50%) from entry open to the highest high within 126 sessions", ret="fraction, close 126 sessions after entry vs entry open",
                          mkt_ret="fraction, equal-weight average ret of the Rs 1 cr universe on that list date", adv="Rs crore, 20-day average traded value",
                          ret2025="percent, close on the last 2025 session vs last 2024 session (or first close if listed in 2025)"),
               updated=datetime.now().isoformat(timespec="seconds"))
    for f in ("v31_picks.csv", "movers_2025.csv", "results.json"):
        (OUTDIR / f"{f}.manifest.json").write_text(json.dumps(dict(dataset=f, **man), indent=1))
    (OUTDIR / "README.md").write_text("# V3.1 hits by year and the 2025 post-mortem\n\nExploration. v31_picks.csv = every V3.1 weekly pick since 2023 with its 6-month outcome; "
                                      "movers_2025.csv = the 2025 movers and the step of V3.1 that dropped them; results.json = the year, month and funnel tables.\n")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
