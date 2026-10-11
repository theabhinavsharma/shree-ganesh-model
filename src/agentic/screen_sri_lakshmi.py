"""SRI LAKSHMI weekly screen — PAPER track (2026-09-29). Rule: V3.1 from the 2026-10-16 run (Abhinav, 2026-10-11).

V3.1 = V3 with the stock liquidity bar at Rs 1 cr a day (20-day average traded value; then-traded price > Rs 50 kept) instead
of Rs 5 cr, plus liquid new listings aged 63-251 sessions without the 1-year trend test (v3_rule.frame_v31 / ranked_v31).
EXP-2026-10-11-v3-l1-newlist-cost: 51.8 vs 41.8 %/yr with liquidity-scaled costs, 2023+ 54.9 vs 41.5, worst fall -40.0 vs
-40.7, 5/5 phases, ahead in 2023-2026 each; still ahead with costs doubled. The first V3.1 batch (2026-10-09 data) was
built by hand on 2026-10-11 and recorded in logs/evals/screen_replacements.jsonl. Reserves = the next 3 names after the top
9 that pass the same filters (looking up to 20 deep). Each name carries its traded value and a max order (~1% of it).

V3 (2026-10-02 to the 2026-10-09 list):

V3 = the G1 top 9 minus Financial Services stocks minus stocks whose own filings tie them only to fading themes
(build_themes.fading_checker, the TFADE rule), NOT refilled. EXP-2026-09-30-v3-vs-g1-RERUN-rescored: 43.1 vs 41.1 %/yr,
2023+ 43.8 vs 40.8, worst fall -41.4 vs -40.0, 5/5 phases. Needs this week's theme data: theme_intensity dated the
screen session and company filings (announcements_historical) at most 10 days old, else no batch is written.

History: v2 (G1 minus Financial Services, no refill) ran for one day, 2026-09-30. It passed 5/5 on the pre-fix price
panel but only 3/5 after the split/dividend fix, so Abhinav chose G1 for all money and asked to track only G1 and the
Sep 8 production sleeve. The v2-era shadow folder logs/sri_lakshmi_g1/ was removed on 2026-10-02 (its one batch duplicated the Sep 28 batch; it is in git history).

Plain English: the model-ranked screen (trend stock, hot or warming industry, top 9 by the model score) with one extra
rule — skip a hot industry when the government's budget money and activity data for it sit in the bottom 30%.
G1 passed its registered test 5/5 phases: 36.5%/yr vs 33.5% for the screen without the veto (2019-01-04 onward),
worst fall -39.1% vs -38.8%. It is still a backtest, so batches are PAPER until Abhinav signs the weekly review.

Rule (identical to test_industry_policy.py G1 via test_industry_fundamentals.select_elig): eligibility =
industry_scores_policy rows on the screen date with heat_pct >= 0.70 and NOT P_pct < 0.30 (an industry with no
budget/activity data passes). Needs data/derived/industry_scores_policy.parquet dated the same session as the prices;
if it is older, no batch is written (stale scores are not silently used).
Output: logs/sri_lakshmi/screen_<date>.json (one per ISO week, immutable) + reports/sri_lakshmi_<date>.md.
Scored daily by score_sri_lakshmi.py (same contract as the leader sleeve: next open, 126 sessions, 0.5% round trip).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
from render_leader_report import FULLTEXT, _takeover  # noqa: E402
from screen_model_ranked import _opt  # noqa: E402
from render_basket_report import ANN  # noqa: E402

CDIR = ROOT / "logs/sri_lakshmi"
EXTRA = 3


def main() -> None:
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    d = D["cal"][-1]
    import v3_rule                                   # the V3 rule lives in one place (v3_rule.py, 2026-10-02)
    ctx = v3_rule.context(imap)
    F1 = v3_rule.frame_v31(X["F"], D)                # V3.1: Rs 1 cr bar + new listings (2026-10-16 run on)
    S = ctx["scores"]
    if S["date"].max() != d:
        raise SystemExit(f"industry_scores_policy is dated {S['date'].max().date()}, prices {d.date()} — rebuild "
                         "build_industry_scores.py + build_policy_scores.py first; no batch written")
    newest = P["trade_date"].max()
    if (d - newest).days > 7:   # select_s1 matches scores within 7 days; beyond that every stock ranks with no score
        raise SystemExit(f"model scores end {newest.date()}, prices {d.date()}: more than 7 days apart, so no stock has a "
                         "score. Rebuild anatomy_1p5x.py + model_bakeoff_1p5x.py first; no batch written")
    fading_only, fin, filings_to, themes_to = ctx["fading_only"], ctx["fin"], ctx["filings_to"], ctx["themes_to"]
    if themes_to < d or (d - filings_to).days > 10:
        raise SystemExit(f"theme data stale (intensity {themes_to.date()}, filings {filings_to.date()}, prices {d.date()}): run "
                         "fetch_announcements_historical.py + fetch_fred_drivers.py + build_themes.py first; no batch written")
    Sd = S[S["date"] == d]
    g1 = Sd[(Sd["heat_pct"] >= 0.70) & ~(Sd["P_pct"] < 0.30)]
    vetoed = Sd[(Sd["heat_pct"] >= 0.70) & (Sd["P_pct"] < 0.30)]
    top = sre.TOPN
    ranked = v3_rule.ranked_v31(F1, [d], imap, P, g1, top + 20)[d]
    s1m = v3_rule.ranked_v31(F1, [d], imap, P, Sd[Sd["heat_pct"] >= 0.70], top)[d]
    F = F1[F1["trade_date"] == d].set_index("symbol")
    Ps = P[P["trade_date"] <= d].sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")
    Sx = Sd.set_index("industry")
    ann = _opt(ANN, filters=[("symbol", "in", ranked)]) if ranked else pd.DataFrame()
    if not ann.empty:
        ann["event_date"] = pd.to_datetime(ann["event_date"], errors="coerce")
    ft = _opt(FULLTEXT, columns=["symbol", "text"])

    def row(i: int, sym: str) -> dict:
        ind = imap.get(sym)
        return dict(rank=i + 1, symbol=sym, rtw=10 - (i + 1), industry=str(ind), close=round(float(F.loc[sym, "close"]), 2),
                    model_score=None if pd.isna(Ps["ensemble"].get(sym)) else round(float(Ps["ensemble"].get(sym)), 4),
                    heat_pct=round(float(Sx.loc[ind, "heat_pct"]), 3),
                    P_pct=None if pd.isna(Sx.loc[ind, "P_pct"]) else round(float(Sx.loc[ind, "P_pct"]), 3),
                    takeover=bool(_takeover(sym, ann[ann["symbol"] == sym] if not ann.empty else None, ft, d)),
                    adv_cr=round(float(F.loc[sym, "adv"]), 2), max_order_inr=int(round(round(float(F.loc[sym, "adv"]), 2) * 1e5)),
                    new_listing=bool(F.loc[sym, "young"]))

    keep = lambda s: ctx["keep"](s, d)  # noqa: E731
    g1_top = ranked[:top]
    dropped_fin = [s for s in g1_top if fin(s)]
    dropped_fading = [s for s in g1_top if not fin(s) and fading_only(s, d)]
    names = [dict(row(i, s), g1_rank=g1_top.index(s) + 1) for i, s in enumerate([s for s in g1_top if keep(s)])]   # no refill
    reserves = [s for s in ranked[top:] if keep(s)][:EXTRA]
    import nse_calendar
    entry = nse_calendar.next_session(d)
    CDIR.mkdir(parents=True, exist_ok=True)
    wk = tuple(d.isocalendar()[:2])
    have = [p for p in CDIR.glob("screen_*.json") if tuple(pd.Timestamp(json.loads(p.read_text())["data_through"]).isocalendar()[:2]) == wk]
    if have:
        print(f"weekly cadence: {have[0].name} already covers ISO week {wk[1]} — no second batch")
    else:
        late = pd.Timestamp.now(tz="Asia/Kolkata") >= pd.Timestamp(f"{entry.date()} 09:15", tz="Asia/Kolkata")
        (CDIR / f"screen_{d.strftime('%Y%m%d')}.json").write_text(json.dumps(dict(
            screen_id=d.strftime("%Y%m%d"), data_through=str(d.date()), created_at=datetime.now().isoformat(timespec="seconds"),
            created_note=("saved after the entry session had started; selection uses data through data_through only"
                          if late else "saved before the entry session"),
            entry="next session open after data_through", hold_td=126, cost_rt_pct="0.5 / 1.0 / 2.0 by liquidity (research_panel.cost_rt)",
            exit="TIME (126 sessions, close); no stops",
            sizing="PAPER", status="SLM V3.1 (EXP-2026-10-11-v3-l1-newlist-cost PASS 5/5 with liquidity-scaled costs); live rule "
                                   "from the 2026-10-16 run by Abhinav's decision; paper until the weekly review is signed",
            rule="V3 with the stock liquidity bar at Rs 1 cr (20-day average traded value; price > Rs 50) instead of Rs 5 cr, plus "
                 "liquid new listings aged 3-12 months without the 1-year trend test; G1 hot-industry eligibility, model rank, "
                 "top 9, minus Financial Services and fading-theme-only stocks, no refill",
            max_order_rule="about 1% of the stock's 20-day average daily traded value",
            names=names, reserves=[dict(rank=i + 1, symbol=s) for i, s in enumerate(reserves)],
            dropped_financials=dropped_fin, dropped_fading_theme=dropped_fading, g1_top9=g1_top,
            vetoed_industries=sorted(vetoed["industry"]), vs_model_screen=dict(dropped=sorted(set(s1m) - set(ranked[:top])),
                                                                            added=sorted(set(ranked[:top]) - set(s1m)))), indent=1))
        print(f"saved paper batch logs/sri_lakshmi/screen_{d.strftime('%Y%m%d')}.json ({len(names)} names)")
    out = [f"# Sri Lakshmi batch — data through {d.date()}", "",
           "PAPER. SLM V3.1: model-ranked screen of stocks trading Rs 1 cr+ a day (price above Rs 50; new listings after "
           "3 months), minus hot industries whose budget money + activity data rank in the bottom 30% (G1), then financial "
           "stocks and stocks tied only to fading themes dropped, no refill. Buy at the next open with a limit, sell at the "
           "close of session 126, no stop. Keep each order under the max order (about 1% of a day's trading).", "",
           f"Dropped: financials {', '.join(dropped_fin) or 'none'}; fading theme {', '.join(dropped_fading) or 'none'}",
           f"Hot industries vetoed today: {', '.join(sorted(vetoed['industry'])) or 'none'}",
           f"Different from the plain model screen: dropped {', '.join(sorted(set(s1m) - set(ranked[:top]))) or 'none'}; "
           f"added {', '.join(sorted(set(ranked[:top]) - set(s1m))) or 'none'}.", "",
           "| Rank | Stock | Industry | Heat pct | Budget/activity pct | Last close | Buy limit (+5%) | Traded/day (Rs cr) | Max order (Rs) | Flag |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        pp = "no data" if n["P_pct"] is None else f"{n['P_pct']:.2f}"
        out.append(f"| {n['rank']} | {n['symbol']} | {n['industry']} | {n['heat_pct']:.2f} | {pp} | "
                   f"{n['close']:,.2f} | {n['close'] * 1.05:,.2f} | {n['adv_cr']:,.2f} | {n['max_order_inr']:,} | "
                   f"{', '.join(x for x, on in (('takeover', n['takeover']), ('new listing', n['new_listing'])) if on)} |")
    out.append(f"\nReserves: {', '.join(reserves) or 'none'}")
    rep = ROOT / f"reports/sri_lakshmi_{d.strftime('%Y%m%d')}.md"
    rep.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
