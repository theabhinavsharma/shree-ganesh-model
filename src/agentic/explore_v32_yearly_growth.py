"""V3.2 money path by calendar year (2026-10-13; Abhinav: chart per year of input money vs portfolio value week by week, and
ROI % month by month). EXPLORATION, no rule changed.

Each year is a fresh start on its first Friday list (Apr 2019 for 2019): Rs 1 lakh goes into that week's V3.2 list every Friday
for 26 weeks (the plan's 26 slices); each lot is sold 126 sessions later and the money rolls into that slice's next list, so no
new money after week 26. Engine = test_v3_l1_newlist_cost.run_cost (dividends counted, big payouts / demergers as cash,
liquidity-scaled costs 0.5/1/2%), entry at the list-day close, Friday lists (schedule 3). Value = invested slices only (cash not
yet put in is not counted). Output: reports/v32_yearly_growth/growth.json (+ manifest) and README.md.
"""
import json, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
OUT = ROOT / "reports/v32_yearly_growth"; LAKH = 1.0


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
    src_ = open(ROOT / "src/agentic/test_v3_l1_newlist_cost.py").read()
    fn = src_[src_.index("def run_cost("):src_.index("def main()")].replace("return pd.Series(nav, index=cal[start_i:]),", "return V, start_i, slots,")
    ns = dict(np=np, pd=pd, sp=sp, sre=sre, tr=__import__("total_return")); exec(fn, ns); run_slots = ns["run_cost"]
    grid = sp.weekly_grid(cal, 3)
    years = {}
    for Y in range(2019, cal[-1].year + 1):
        wk = [d for d in grid if pd.Timestamp(f"{Y}-01-01") <= d <= pd.Timestamp(f"{Y}-12-31") and d >= pd.Timestamp("2019-04-01")]
        picks = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}
        V, si, slots, _ = run_slots(D, X, picks, wk, "E0", cost)
        first = {s: tpos[wk[s]] for s in range(min(slots, len(wk)))}            # slot s first buys on list s
        last_day = min(tpos[cal[cal <= pd.Timestamp(f"{Y}-12-31")][-1]], len(cal) - 1)
        def at(i):
            inv = [s for s, f in first.items() if f <= i]
            return len(inv) * LAKH, float(sum(V[s, i] for s in inv)) * LAKH
        weekly = []
        for k, d in enumerate(wk):
            i = tpos[d]; inp, val = at(i)
            weekly.append(dict(week=k + 1, date=str(d.date()), input_lakh=inp, value_lakh=round(val, 3), names=len(picks[d])))
        monthly = []
        for m in range(1, 13):
            me = cal[(cal.year == Y) & (cal.month == m)]
            if not len(me) or tpos[me[-1]] < first[0]:
                continue
            i = tpos[me[-1]]; inp, val = at(min(i, last_day))
            monthly.append(dict(month=f"{Y}-{m:02d}", date=str(me[-1].date()), input_lakh=inp, value_lakh=round(val, 3), roi_pct=round((val / inp - 1) * 100, 1)))
        years[Y] = dict(weekly=weekly, monthly=monthly, lists=len(wk), empty_lists=sum(not picks[d] for d in wk))
        e = weekly[-1]; print(f"{Y}: {len(wk)} lists · end {e['date']} input {e['input_lakh']:.0f} L value {e['value_lakh']:.1f} L "
                              f"ROI {(e['value_lakh'] / e['input_lakh'] - 1) * 100:+.1f}% · monthly ROI " + " ".join(f"{x['roi_pct']:+.0f}" for x in monthly), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "growth.json").write_text(json.dumps(years, indent=1))
    (OUT / "growth.json.manifest.json").write_text(json.dumps(dict(dataset="growth.json", producer="src/agentic/explore_v32_yearly_growth.py",
        definitions=__doc__, units=dict(input_lakh="Rs lakh put in so far (1 per Friday list, weeks 1-26)", value_lakh="Rs lakh, market value of the invested slices at that close (costs and dividends counted)",
        roi_pct="value / input - 1, percent, at the month's last session"), data_through=str(cal[-1].date()), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
    (OUT / "README.md").write_text("# V3.2 money path by year\n\nFresh start each year: Rs 1 lakh into each Friday V3.2 list for 26 weeks, lots sold after 6 months and "
                                   "rolled into the next list. growth.json: weekly input vs value, monthly ROI. Backtest (signal-day close, costs and dividends counted).\n")


if __name__ == "__main__":
    main()
