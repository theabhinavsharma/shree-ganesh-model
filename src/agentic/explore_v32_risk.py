"""V3.2 risk: standard deviation at four levels, against the equal-weight liquid market on the same 6-month ladder (2026-10-13;
Abhinav: 'what's the standard deviation for us?'). EXPLORATION, no rule changed.

Portfolio = the full 26-slice ladder (test_v3_l1_newlist_cost.run_cost: dividends, payouts as cash, liquidity-scaled costs,
list-day close entry), weekly lists from 2019-04-01. Market = every stock passing V3.1's liquidity bar (Rs 1 cr ADV, price > 50)
bought on the same lists, same engine and costs. Levels: (1) daily NAV swings, annualised; (2) calendar-year returns 2020-2025;
(3) one weekly lot's 6-month return (equal weight, after costs); (4) one stock's 6-month return. Sharpe uses the average India
call money rate (FRED IRSTCI01INM156N, macro_drivers_fred.parquet) over the window.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
from test_v3_l1_newlist_cost import run_cost  # noqa: E402
START, HOLD = pd.Timestamp("2019-04-01"), 126


def main():
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy()
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))  # noqa: E731
    liquid = F1[F1["core"].fillna(False).astype(bool) & F1["symbol"].isin(D["cidx"])].groupby("trade_date")["symbol"].apply(list).to_dict()
    fr = pd.read_parquet(ROOT / "data/derived/macro_drivers_fred.parquet"); fr = fr[fr["fred_id"] == "IRSTCI01INM156N"]
    rf = float(fr[pd.to_datetime(fr["date"]) >= START]["value"].mean())
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)])
    vr = D.get("vr", {})

    def lot_returns(picks):
        """{date: [net 6-month return per name]} for lots whose 126 sessions have passed (payout / demerger as cash)."""
        out = {}
        for d, names in picks.items():
            i0 = tpos[d]
            if i0 + HOLD > len(cal) - 1 or not names:
                continue
            rs = []
            for s in names:
                q = D["cidx"].get(s)
                if q is None:
                    continue
                g = np.exp(lg[i0 + HOLD + 1, q] - lg[i0 + 1, q])
                for k, f in vr.get(q, ()):
                    if i0 + 1 <= k <= i0 + HOLD - 1:
                        g = f * g + (1 - f) * np.exp(lg[k + 1, q] - lg[i0 + 1, q]); break
                rs.append((1 - float(cost(i0, np.array([q]))[0])) * g - 1)
            out[d] = rs
        return out

    rows = []
    for o in range(5):
        wk = [d for d in sp.weekly_grid(cal, o) if d >= START]
        v32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}
        mkt = {d: liquid.get(d, []) for d in wk}
        for arm, pk in (("V3.2", v32), ("Market", mkt)):
            nav = run_cost(D, X, pk, wk, "E0", cost)[0]
            r = np.log(nav).diff().dropna(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25; spy = len(r) / yrs
            cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
            vol = r.std() * np.sqrt(spy); down = r[r < 0].std() * np.sqrt(spy)
            ye = nav.groupby(nav.index.year).last(); cy = (ye / ye.shift(1) - 1).loc[2020:2025] * 100
            lots = lot_returns(pk); lr = np.array([np.mean(v) for v in lots.values() if v]) * 100; sr = np.concatenate([v for v in lots.values() if v]) * 100
            rows.append(dict(o=o, arm=arm, cagr=cagr * 100, vol=vol * 100, down=down * 100, sharpe=(cagr * 100 - rf) / (vol * 100),
                             maxdd=float((nav / nav.cummax() - 1).min() * 100), cy_mean=cy.mean(), cy_sd=cy.std(), cy_min=cy.min(), cy_max=cy.max(),
                             lot_n=len(lr), lot_mean=lr.mean(), lot_med=np.median(lr), lot_sd=lr.std(), lot_p5=np.percentile(lr, 5), lot_p95=np.percentile(lr, 95), lot_neg=(lr < 0).mean() * 100,
                             stk_n=len(sr), stk_mean=sr.mean(), stk_sd=sr.std(), stk_neg=(sr < 0).mean() * 100, stk_bad=(sr <= -30).mean() * 100, stk_good=(sr >= 50).mean() * 100))
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"India call rate average since Apr 2019: {rf:.2f}%")
    for arm, g in R.groupby("arm", sort=False):
        f = g[g.o == 3].iloc[0]
        print(f"\n{arm} (Friday-ish schedule 3; range over 5 schedules in brackets)")
        print(f"  1 portfolio: CAGR {f.cagr:.1f}% · volatility (SD of daily moves, annualised) {f.vol:.1f}% [{g.vol.min():.1f}-{g.vol.max():.1f}] · downside {f.down:.1f}% · Sharpe {f.sharpe:.2f} [{g.sharpe.min():.2f}-{g.sharpe.max():.2f}] · worst fall {f.maxdd:.1f}%")
        print(f"  2 calendar years 2020-2025: mean {f.cy_mean:.1f}% · SD {f.cy_sd:.1f} pts [{g.cy_sd.min():.1f}-{g.cy_sd.max():.1f}] · range {f.cy_min:.1f}..{f.cy_max:.1f}")
        print(f"  3 one weekly lot, 6 months ({f.lot_n:.0f} lots): mean {f.lot_mean:.1f}% · median {f.lot_med:.1f}% · SD {f.lot_sd:.1f} pts [{g.lot_sd.min():.1f}-{g.lot_sd.max():.1f}] · 5th-95th pct {f.lot_p5:.1f}..{f.lot_p95:.1f} · lots that lost money {f.lot_neg:.0f}%")
        print(f"  4 one stock, 6 months ({f.stk_n:.0f} picks): mean {f.stk_mean:.1f}% · SD {f.stk_sd:.1f} pts · lost money {f.stk_neg:.0f}% · lost 30%+ {f.stk_bad:.0f}% · made 50%+ {f.stk_good:.0f}%")


if __name__ == "__main__":
    main()
