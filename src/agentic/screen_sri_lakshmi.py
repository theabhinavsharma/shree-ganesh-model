"""SRI LAKSHMI weekly screen — PAPER track (2026-09-29; v2 from 2026-09-30).

v2 (2026-09-30, EXP-2026-09-30-sector-shrink PASS 5/5: 38.5 vs 36.9%/yr, 2023+ 41.3 vs 38.9): the G1 top 9 minus any
Financial Services stock (screener.in broad sector), NOT refilled: the batch is split among the remaining names.
The old G1 top 9 is still saved every week to logs/sri_lakshmi_g1/ (paper shadow) so the two can be compared live.

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
import test_industry_fundamentals as tif  # noqa: E402
from render_leader_report import FULLTEXT, _takeover  # noqa: E402
from screen_model_ranked import _opt  # noqa: E402
from render_basket_report import ANN  # noqa: E402

SCORES = ROOT / "data/derived/industry_scores_policy.parquet"
CDIR = ROOT / "logs/sri_lakshmi"
SHADOW = ROOT / "logs/sri_lakshmi_g1"
EXTRA = 3


def main() -> None:
    D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores()
    d = D["cal"][-1]
    S = pd.read_parquet(SCORES); S["date"] = pd.to_datetime(S["date"])
    if S["date"].max() != d:
        raise SystemExit(f"industry_scores_policy is dated {S['date'].max().date()}, prices {d.date()} — rebuild "
                         "build_industry_scores.py + build_policy_scores.py first; no batch written")
    Sd = S[S["date"] == d]
    g1 = Sd[(Sd["heat_pct"] >= 0.70) & ~(Sd["P_pct"] < 0.30)]
    vetoed = Sd[(Sd["heat_pct"] >= 0.70) & (Sd["P_pct"] < 0.30)]
    top = sre.TOPN
    sre.TOPN = top + EXTRA
    ranked = tif.select_elig(X["F"], [d], imap, P, g1).get(d, [])
    s1m = tif.select_elig(X["F"], [d], imap, P, Sd[Sd["heat_pct"] >= 0.70]).get(d, [])[:top]
    sre.TOPN = top
    F = X["F"][X["F"]["trade_date"] == d].set_index("symbol")
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
                    takeover=bool(_takeover(sym, ann[ann["symbol"] == sym] if not ann.empty else None, ft, d)))

    import html as _html
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
    sc["industry"] = sc["industry"].map(_html.unescape)
    sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
    fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
    g1_top = ranked[:top]
    dropped = [s for s in g1_top if fin(s)]
    v2 = [s for s in g1_top if not fin(s)]                      # no refill
    names = [dict(row(i, s), g1_rank=g1_top.index(s) + 1) for i, s in enumerate(v2)]
    reserves_v2 = [s for s in ranked[top:] if not fin(s)]
    entry = pd.bdate_range(d + pd.Timedelta(days=1), periods=1)[0]
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
            entry="next session open after data_through", hold_td=126, cost_rt_pct=0.5, exit="TIME (126 sessions, close); no stops",
            sizing="PAPER", status="SLM v2: G1 (EXP-2026-09-29-industry-policy) minus Financial Services, no refill "
                                   "(EXP-2026-09-30-sector-shrink PASS 5/5); paper until the weekly review is signed",
            rule="model-ranked screen (trend, heat_pct >= 0.70, top 9 by ensemble) minus hot industries with P_pct < 0.30; "
                 "then drop Financial Services stocks from the 9 without refilling",
            names=names, reserves=[dict(rank=i + 1, symbol=s) for i, s in enumerate(reserves_v2)],
            dropped_financials=dropped, g1_top9=g1_top,
            vetoed_industries=sorted(vetoed["industry"]), vs_model_screen=dict(dropped=sorted(set(s1m) - set(ranked[:top])),
                                                                            added=sorted(set(ranked[:top]) - set(s1m)))), indent=1))
        print(f"saved paper batch logs/sri_lakshmi/screen_{d.strftime('%Y%m%d')}.json ({len(names)} names; dropped financials: {dropped or 'none'})")
        SHADOW.mkdir(parents=True, exist_ok=True)       # old rule (G1 top 9) as a paper shadow, same cadence
        if not any(tuple(pd.Timestamp(json.loads(p.read_text())["data_through"]).isocalendar()[:2]) == wk for p in SHADOW.glob("screen_*.json")):
            (SHADOW / f"screen_{d.strftime('%Y%m%d')}.json").write_text(json.dumps(dict(
                screen_id=d.strftime("%Y%m%d"), data_through=str(d.date()), created_at=datetime.now().isoformat(timespec="seconds"),
                entry="next session open after data_through", hold_td=126, cost_rt_pct=0.5, exit="TIME (126 sessions, close); no stops",
                sizing="PAPER SHADOW", status="old Sri Lakshmi rule (G1 top 9, financials kept) for comparison with v2",
                names=[row(i, s) for i, s in enumerate(g1_top)]), indent=1))
    out = [f"# Sri Lakshmi batch — data through {d.date()}", "",
           "PAPER. SLM v2: model-ranked screen minus hot industries whose budget money + activity data rank in the bottom 30% "
           "(G1), then Financial Services stocks dropped without refill (EXP-2026-09-30-sector-shrink). Buy at the next open, "
           "sell at the close of session 126, no stop.", "",
           f"Financials dropped this week: {', '.join(dropped) or 'none'}",
           f"Hot industries vetoed today: {', '.join(sorted(vetoed['industry'])) or 'none'}",
           f"Different from the plain model screen: dropped {', '.join(sorted(set(s1m) - set(ranked[:top]))) or 'none'}; "
           f"added {', '.join(sorted(set(ranked[:top]) - set(s1m))) or 'none'}. Full QUANT/QUAL notes per name: "
           f"reports/model_screen_{d.strftime('%Y%m%d')}.md", "",
           "| Rank | Stock | Industry | Heat pct | Budget/activity pct | Last close | Buy limit (+5%) | Flag |", "|---|---|---|---|---|---|---|---|"]
    for n in names:
        pp = "no data" if n["P_pct"] is None else f"{n['P_pct']:.2f}"
        out.append(f"| {n['rank']} | {n['symbol']} | {n['industry']} | {n['heat_pct']:.2f} | {pp} | "
                   f"{n['close']:,.2f} | {n['close'] * 1.05:,.2f} | {'takeover' if n['takeover'] else ''} |")
    out.append(f"\nReserves: {', '.join(ranked[top:]) or 'none'}")
    rep = ROOT / f"reports/sri_lakshmi_{d.strftime('%Y%m%d')}.md"
    rep.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
