"""V3.2 year by year: backtest headline vs after tax + Monday-open buying vs the hindsight-free rule choice (2026-10-13; Abhinav:
'y-o-y batao bro for each year from 2019'). EXPLORATION, no rule changed.

Continuous 26-slice ladder from 2019-04-01, calendar-year returns averaged over the 5 weekly schedules.
  headline   V3.2, before tax, bought at the list-day close (run_cost)
  realistic  V3.2, after tax (20.8% short-term, 13.0% long-term incl. cess; test_v32_hold_after_tax.engines), bought at the next open
  no_hindsight  the rule sets EXP-2026-10-13-walkforward-rule-choice's past-only investor held: FT|1cr10|--P in 2021-2022,
             FT|1cr10|-XP from 2023 (financial + fading drops, Rs 1 cr ADV with price > 10, promoter buyers first; trade-for-trade
             drop from 2023), after tax, next open. Not defined for 2019-2020 (no history to choose from).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
from test_v32_hold_after_tax import engines, ST, LT  # noqa: E402
START, YEARS = pd.Timestamp("2019-04-01"), range(2019, 2027)


def main():
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); G1 = ctx["G1"]; F1 = v3_rule.frame_v31(X["F"], D); F1["core_p10"] = (F1["adv"] >= 1.0) & (F1["raw"] > 10)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"]); A["trade_date"] = pd.to_datetime(A["trade_date"])
    PBd = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}

    def pb(s, d):
        a = PBd.get(s)
        if a is None:
            return np.nan
        i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
        return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan

    def ranked(weeks, core_col, topn=25):   # frozen copy of test_walkforward_rule_choice.ranked (no new-listing exemption)
        W = F1[F1["trade_date"].isin(set(weeks)) & F1[core_col].fillna(False).astype(bool)].copy()
        W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
        W = W.merge(G1[["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = (W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])
        Pool = W[tr].copy()
        Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
        m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
        Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
        Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
        g = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict()
        return {d: g.get(d, []) for d in weeks}

    def wf_lists(weeks, x):
        out = {}
        R = ranked(weeks, "core_p10")
        for d in weeks:
            pool = R[d]; first = [s for s in pool[:25] if pb(s, d) > 0]; pool = first + [s for s in pool if s not in set(first)]
            out[d] = [s for s in pool[:9] if not ctx["fin"](s) and not ctx["fading_only"](s, d) and not (x and t2t(s, d))]
        return out

    run_tax, run_tax_open = engines()
    def yr(nav, Y):
        a = nav[nav.index <= pd.Timestamp(f"{Y}-12-31")]; b = nav[nav.index <= pd.Timestamp(f"{Y - 1}-12-31")]
        return (a.iloc[-1] / (b.iloc[-1] if len(b) else nav.iloc[0]) - 1) * 100
    rows = []
    for o in range(5):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        v32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}
        navs = {"headline": run_cost(D, X, v32, wk, "E0", cost)[0], "realistic": run_tax_open(D, X, v32, wk, "E0", cost, ST, LT)[0],
                "wf_noX": run_tax_open(D, X, wf_lists(wk, False), wk, "E0", cost, ST, LT)[0], "wf_X": run_tax_open(D, X, wf_lists(wk, True), wk, "E0", cost, ST, LT)[0]}
        for Y in YEARS:
            r = {k: yr(n, Y) for k, n in navs.items()}
            rows.append(dict(o=o, year=Y, headline=r["headline"], realistic=r["realistic"],
                             no_hindsight=(r["wf_noX"] if Y in (2021, 2022) else r["wf_X"] if Y >= 2023 else np.nan)))
        for k, n in navs.items():
            rows.append(dict(o=o, year=0, **{k + "_cagr19": (n.iloc[-1] / n.iloc[0]) ** (365.25 / (n.index[-1] - n.index[0]).days) * 100 - 100}))
        print(f"schedule {o} done", flush=True)
    R = pd.DataFrame(rows)
    Y = R[R.year > 0].groupby("year")[["headline", "realistic", "no_hindsight"]].mean().round(1)
    print(Y.to_string())
    yrs_ = (cal[-1] - pd.Timestamp("2020-12-31")).days / 365.25
    ch = lambda c: (((1 + Y.loc[2021:, c] / 100).prod()) ** (1 / yrs_) - 1) * 100  # noqa: E731
    print(f"\nper year 2021-now (chained, avg of 5 schedules): headline {ch('headline'):.1f} · realistic {ch('realistic'):.1f} · no hindsight {ch('no_hindsight'):.1f}")
    C = R[R.year == 0].mean(numeric_only=True)
    print(f"per year Apr 2019-now: headline {C['headline_cagr19']:.1f} · realistic {C['realistic_cagr19']:.1f}")
    Y.to_csv(ROOT / "reports/v32_yearly_growth/yearly_realistic.csv")


if __name__ == "__main__":
    main()
