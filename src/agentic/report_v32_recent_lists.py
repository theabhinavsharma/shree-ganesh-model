"""What V3.2 would have picked on past Friday lists, at what price, and what each lot fetched at its 6-month sell (2026-10-12;
Abhinav: '6M backwards from this week what would v3.2 have suggested at what price? I want to verify', then 'the picks we'd
have made in april 2026 overall as a month ... so I can check manually how it'd have done in oct').

List dates = the last NSE session of each Mon-Fri week (what the Friday run uses; a Thursday when Friday is a holiday).
Plain English: rebuilds the V3.2 list for each past Friday exactly as the backtest does (v3_rule.ranked_v31 top 9 -> financial /
fading-theme drops -> trade-for-trade names dropped, no refill) and prints the prices as they traded that day, so they can be
checked against NSE's own bhavcopy. These are RECONSTRUCTED lists (today's data, walk-forward model scores); the lists actually
sent on those Fridays used older rules (V2 / V3). Check: the newest list must equal the live Oct 9 list in v32_overrides.jsonl.
Prices are NSE as-traded rupees (adjusted close / price_adjustment_factor_to_present): Friday close, the next session's open,
and the latest close; buy range = Friday close to +5% (the live rule). 'since list' = adjusted close change from the Friday
close to the latest close (splits / bonuses handled, dividends not added). Sell = the close 126 sessions after the list date
(the backtest's exit; NSE holiday list for dates not yet traded); 'lot return' = adjusted close at the sell (or latest, if not
yet due) / list-date close - 1.
Output: reports/v32_recent_lists/<label>/lists.csv (+ manifest) and README.md (one table per week).
Usage: /usr/bin/python3 src/agentic/report_v32_recent_lists.py last26 | 2026-04
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import v3_rule  # noqa: E402

OUT = ROOT / "reports/v32_recent_lists"
HOLD = 126


def main(arg: str = "last26") -> None:
    import nse_calendar
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
    ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D)
    s_ = pd.Series(cal, index=cal); ends = list(s_.groupby(cal.to_period("W-FRI")).max())
    ends = [d for d in ends if d < cal[-1] or cal[-1].dayofweek == 4]                 # a week still in progress has no list yet
    weeks = ends[-int(arg[4:]):] if arg.startswith("last") else [d for d in ends if f"{d:%Y-%m}" == arg]
    out = OUT / arg; n = len(weeks)
    top9 = v3_rule.ranked_v31(F1, weeks, imap, P, ctx["G1"], 9)
    syms = sorted({s for d in weeks for s in top9[d]})
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series", "open", "close", "price_adjustment_factor_to_present"],
                        filters=[("symbol", "in", syms), ("trade_date", ">=", weeks[0] - pd.Timedelta(days=10))])
    S["trade_date"] = pd.to_datetime(S["trade_date"]); S["raw_close"] = rp.raw_price(S, "close"); S["raw_open"] = rp.raw_price(S, "open")
    S = S.set_index(["symbol", "trade_date"]).sort_index()
    last = cal[-1]; pos = {d: i for i, d in enumerate(cal)}
    sc = pd.merge_asof(pd.DataFrame([(s, d) for d in weeks for s in top9[d]], columns=["symbol", "trade_date"]).sort_values("trade_date"),
                       P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    score = {(r.symbol, r.trade_date): r.ensemble for r in sc.itertuples()}

    def series_on(s, d):
        h = S.loc[s] if s in S.index.get_level_values(0) else None
        h = h[h.index <= d] if h is not None else None
        return None if h is None or h.empty else str(h["series"].iloc[-1])

    rows = []
    for d in weeks:
        nxt = cal[pos[d] + 1] if pos[d] + 1 < len(cal) else None
        k = pos[d] + HOLD; due = k < len(cal)
        sell = cal[k] if due else nse_calendar.add_sessions(nse_calendar.next_session(last), k - (len(cal) - 1))
        for rank, s in enumerate(top9[d], 1):
            ser = series_on(s, d)
            why = None
            if not ctx["keep"](s, d):
                why = "dropped: financial or fading-theme only"
            elif ser in ("BE", "BZ"):
                why = f"dropped: trade-for-trade ({ser} series)"
            h = S.loc[s]
            c0 = h["raw_close"].get(d, np.nan); a0 = h["close"].get(d, np.nan)
            o1 = h["raw_open"].get(nxt, np.nan) if nxt is not None else np.nan
            cl = h["raw_close"].get(last, np.nan); al = h["close"].get(last, np.nan)
            sc_ = h["raw_close"].get(sell, np.nan) if due else np.nan; sa = h["close"].get(sell, np.nan) if due else al
            rows.append(dict(list_date=d.date(), rank=rank, symbol=s, picked=why is None, note=why or "", series=ser,
                             model_score=round(float(score.get((s, d), np.nan)), 3), fri_close=round(c0, 2), buy_upto=round(c0 * 1.05, 2),
                             next_session=nxt.date() if nxt is not None else None, next_open=round(o1, 2),
                             open_in_range=bool(o1 <= c0 * 1.05) if np.isfinite(o1) else None,
                             sessions_held=len(cal) - 1 - pos[d], latest_date=last.date(), latest_close=round(cl, 2),
                             since_list_pct=round((al / a0 - 1) * 100, 1) if a0 and np.isfinite(al) else np.nan,
                             sell_date=pd.Timestamp(sell).date(), sold=bool(due), sell_close=round(sc_, 2),
                             lot_return_pct=round((sa / a0 - 1) * 100, 1) if a0 and np.isfinite(sa) else np.nan))
    T = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "lists.csv", index=False)
    (out / "lists.csv.manifest.json").write_text(json.dumps(dict(
        dataset="lists.csv", producer="src/agentic/report_v32_recent_lists.py", definitions=__doc__,
        columns=dict(list_date="Friday list date (data through that close)", rank="model rank inside the V3.1 pool's top 9 (1 = best)",
                     picked="True = in the V3.2 list; False = dropped (see note)", series="NSE series on the list date (EQ normal; BE/BZ trade-for-trade)",
                     model_score="ensemble = mean within-week percentile rank of LightGBM / XGBoost / MLP (0-1, NOT a probability)",
                     fri_close="NSE close on the list date, rupees as traded", buy_upto="fri_close x 1.05 (live buy limit)",
                     next_open="open of the next NSE session, rupees as traded", open_in_range="next_open <= buy_upto",
                     sessions_held="NSE sessions from the list date to latest_date (a lot is sold at 126)",
                     latest_close="NSE close on latest_date, rupees as traded (differs from fri_close basis after a split / bonus)",
                     since_list_pct="percent, adjusted close latest / list date - 1 (splits and bonuses handled, dividends not added)",
                     sell_date="the close 126 NSE sessions after the list date (future dates from the NSE holiday list)",
                     sold="True if sell_date <= latest_date", sell_close="NSE close on sell_date, rupees as traded (blank if not yet due)",
                     lot_return_pct="percent, adjusted close at sell_date (or latest_date if not yet due) / list-date close - 1, dividends not added"),
        weeks=[str(d.date()) for d in weeks], updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    md = [f"# V3.2 lists, {arg} ({n} weekly lists, reconstructed)", "",
          "What V3.2 would have picked each Friday and the prices as traded, to check against NSE's bhavcopy. Rebuilt with today's",
          "data and the walk-forward model; the lists actually sent on these Fridays used older rules. Buy at the next open up to",
          "Friday close + 5%; each lot is sold at the close 126 sessions later. 'Since list' = price change to the latest close",
          f"({last.date()}), dividends not added. Columns are defined in lists.csv.manifest.json.", ""]
    for d in reversed(weeks):
        W = T[T["list_date"] == d.date()]
        k = W[W["picked"]]
        sd = W["sell_date"].iat[0]; sold = bool(W["sold"].iat[0])
        md += [f"## {d:%a %d %b %Y} · {len(k)} stocks · sell {'on' if sold else 'due'} {sd:%a %d %b %Y}" + (f" · lot return {k['lot_return_pct'].mean():+.1f}%" if len(k) else ""), "",
               f"| # | Stock | List-day close ₹ | Buy up to ₹ | Next open ₹ | {'Sell close' if sold else 'Latest'} ₹ | Lot return |", "|---|---|---|---|---|---|---|"]
        md += [f"| {r.rank} | {r.symbol} | {r.fri_close:,.2f} | {r.buy_upto:,.2f} | {r.next_open:,.2f}{'' if r.open_in_range in (True, None) else ' (above limit)'} | "
               f"{(r.sell_close if sold else r.latest_close):,.2f} | {r.lot_return_pct:+.1f}% |" for r in k.itertuples()]
        dr = W[~W["picked"]]
        if len(dr):
            md += ["", "Dropped: " + ", ".join(f"{r.symbol} ({r.note.replace('dropped: ', '')})" for r in dr.itertuples())]
        md += [""]
    (out / "README.md").write_text("\n".join(md))
    print(f"{len(weeks)} lists {weeks[0].date()}..{weeks[-1].date()} · {int(T['picked'].sum())} picks · -> {out / 'README.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "last26")
