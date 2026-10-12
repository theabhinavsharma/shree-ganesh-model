"""V3.2 money path when every weekly list of a year gets new money (2026-10-13; Abhinav: 'mujhe all investments done that year ke
liye batao - roi then agle saal ke first 6 months tak jayega'). EXPLORATION, no rule changed.
--roll (Abhinav: 'iss week jo becha usko agle 6M wale cycle mein dalunga'): every list in Y also gets the proceeds of the lot sold
just before it (the lot bought 26 lists earlier, same year); after Dec 31 nothing new is bought and sales are kept as cash, so
the year's books close when its last lot is sold, around July of Y+1. Outputs weekly_roll.csv / monthly_roll.csv.

Year Y: Rs 1 lakh of NEW money into every weekly V3.2 list dated in Y (schedule 3, from Apr 2019); each lot is sold at the close
126 sessions after its list date and its money is kept as cash (not reinvested), so the year is complete when its last lot is
sold, around June-July of Y+1. Lot = equal weight of the list's names after liquidity-scaled costs (0.5/1/2%), total return with
dividends, a big payout / demerger held through turned into cash (total_return rules); an empty list keeps its lakh as cash.
Bought at the list-day close. Outputs reports/v32_yearly_growth/weekly_all.csv, monthly_all.csv (gitignored, regenerable).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule  # noqa: E402
HOLD = 126; OUT = ROOT / "reports/v32_yearly_growth"; ROLL = "--roll" in sys.argv


def main():
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); T = len(cal)
    L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
    ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
    TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
    tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy()
    t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0  # noqa: E731
    lg = np.vstack([np.zeros((1, D["Rv"].shape[1])), np.cumsum(np.log1p(np.nan_to_num(D["Rv"])), axis=0)])
    vr = D.get("vr", {})
    grid = [d for d in sp.weekly_grid(cal, 3) if d >= pd.Timestamp("2019-04-01")]
    picks = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, grid, imap, P, ctx).items()}

    def lot_path(d):
        """Value of Rs 1 at sessions i0..min(i0+HOLD, T-1); after the sale it stays flat (cash)."""
        i0 = tpos[d]; end = min(i0 + HOLD, T - 1)
        qs = [D["cidx"][s] for s in picks[d] if s in D["cidx"]]
        if not qs:
            return i0, end, np.ones(end - i0 + 1)
        qs = np.array(qs); G = np.exp(lg[i0 + 1:end + 2][:, qs] - lg[i0 + 1, qs])           # row j = session i0 + j
        for j, q in enumerate(qs):
            for k, f in vr.get(int(q), ()):
                if i0 + 1 <= k <= i0 + HOLD - 1 and k <= end:
                    r = k - i0; G[r + 1:, j] = f * G[r + 1:, j] + (1 - f) * G[r, j]; break
        c = np.where(ADV[i0, qs] >= 5, 0.005, np.where(ADV[i0, qs] >= 1, 0.010, 0.020))
        return i0, end, (G * ((1 - c) / len(qs))).sum(axis=1)

    W, M = [], []
    for Y in range(2019, cal[-1].year + 1):
        lots = [d for d in grid if d.year == Y]
        paths = [lot_path(d) for d in lots]
        last = max(e for _, e, _ in paths)
        done = all(i0 + HOLD <= T - 1 for i0, _, _ in paths)
        evals = [d for d in grid if lots[0] <= d <= cal[last]]
        if cal[last] not in evals:
            evals.append(cal[last])

        cap = [1.0] * len(paths)                                   # money going into each lot: Rs 1 lakh new (+ proceeds if rolled)
        if ROLL:
            for k in range(26, len(paths)):
                i0p, ep, vp = paths[k - 26]
                assert ep < paths[k][0], "lot k-26 must be sold before list k"
                cap[k] = 1.0 + cap[k - 26] * vp[-1]

        def at(i):
            inp = val = 0.0
            for k, (i0, e, v) in enumerate(paths):
                if i0 <= i:
                    inp += 1
                    if ROLL and k + 26 < len(paths) and paths[k + 26][0] <= i:
                        continue                                       # its money now sits inside lot k + 26
                    val += cap[k] * v[min(i, e) - i0]
            return inp, val
        for n, d in enumerate(evals, 1):
            inp, val = at(tpos[d])
            W.append(dict(year=Y, step=n, date=str(d.date()), input_lakh=inp, value_lakh=round(val, 3), complete=done))
        months = pd.period_range(lots[0].to_period("M"), cal[last].to_period("M"), freq="M")
        for m, p in enumerate(months, 1):
            me = cal[(cal.to_period("M") == p) & (cal <= cal[last])]
            if not len(me):
                continue
            inp, val = at(tpos[me[-1]])
            M.append(dict(year=Y, idx=m, month=str(p), date=str(me[-1].date()), input_lakh=inp, value_lakh=round(val, 3), roi_pct=round((val / inp - 1) * 100, 1)))
        inp, val = at(last)
        print(f"{Y}: {len(lots)} lots, Rs {inp:.0f} L in · last sale {cal[last].date()}{'' if done else ' (lots still open: value to date)'} · "
              f"value {val:.1f} L · ROI {(val / inp - 1) * 100:+.1f}% · monthly ROI " + " ".join(f"{x['month'][2:]}:{x['roi_pct']:+.0f}" for x in M if x["year"] == Y), flush=True)
    tag = "roll" if ROLL else "all"
    pd.DataFrame(W).to_csv(OUT / f"weekly_{tag}.csv", index=False); pd.DataFrame(M).to_csv(OUT / f"monthly_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
