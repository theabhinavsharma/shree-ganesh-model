"""MODEL-RANKED SCREEN (S1M of EXP-2026-09-28-screen-rank-exit) on the latest session, with QUANT / QUAL / MACRO rationale.

Status: PAPER / NOT VALIDATED. In the registered backtest this arm earned 33.8%/yr vs production's 23.2% (2019+, both
eras higher) but its max drawdown was 5.1 points deeper, so it FAILED the registered rule. Listed for information and
forward paper tracking only; production remains screen_theme_leaders.py.

Rule (identical to sim_screen_rank_exit.select_s1): core band (20d ADV >= Rs5cr, then-traded close > Rs50); trend =
close >= 1.5x its 252-row low, 12-month return >= 30%, close > SMA200, SMA50 > SMA200; industry heat percentile >= 0.70
(mean own 60d return of the industry's core names, >= 5 names; nse4 map + analyst analogs); top 9 by the bake-off
ensemble score (walk-forward, latest weekly score <= 7 days old). Hold 126 sessions, no stops (exits lowered returns).
Qual/macro text comes from render_basket_report / render_leader_report helpers (same sources as the production report).
Output: stdout + reports/model_screen_<date>.md. Nothing else is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
from render_basket_report import ANN, BLOCK, CA, INDUSTRY, MACRO, PIT, SUPERSTAR, _macro, _qual, _sector_map  # noqa: E402
from render_leader_report import FULLTEXT, NEWS_FEED, _news, _takeover  # noqa: E402

ROWS = ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet"   # delivery_pct is a 0-1 fraction in the data (median 0.53) although its manifest says "%"
EXTRA = 3                                                          # reserves shown after the top 9


def _opt(path: Path, **kw) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, **kw) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def pct(v, nd=0) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v * 100:+.{nd}f}%"


def main() -> None:
    D = sp.load(None)
    X = sre.features(D)
    imap = sp.industry_maps()["analogs"]
    P = sre.model_scores()
    cal = D["cal"]; d = cal[-1]
    # the full ranked pool at d (select_s1 keeps the top 9; rebuild the pool to show reserves and pool size)
    sre_top = sre.TOPN
    sre.TOPN = sre_top + EXTRA
    ranked = sre.select_s1(X["F"], [d], imap, "model", P).get(d, [])
    sre.TOPN = sre_top
    prod = sp.select(D["px"], [d], "core", 3, imap).get(d, [])

    F = X["F"][X["F"]["trade_date"] == d].set_index("symbol")
    F["ind"] = F.index.map(imap)
    core = F[F["core"] & F["ind"].notna() & F["ret60"].notna()]
    grp = core.groupby("ind")["ret60"].agg(["mean", "size"])
    grp = grp[grp["size"] >= 5]
    grp["hp"] = grp["mean"].rank(pct=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        trend = ((core["close"] / core["lo252"] - 1 >= 0.5) & (core["ret252"] >= 0.30) & (core["close"] > core["sma_200"])
                 & (core["sma_50"] > core["sma_200"]))
    pool_n = int((trend & core["ind"].map(grp["hp"]).ge(sre.HOTWARM)).sum())
    Ps = P[P["trade_date"] <= d].sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")

    R = pd.read_parquet(ROWS, columns=["symbol", "trade_date", "pe", "pe_ind", "eps_yoy", "sales_yoy", "profitable",
                                       "promoter_pct", "prom_delta", "delivery_pct", "rsi_14_daily", "days_since_results",
                                       "mcap_cr", "adv", "ev60_order", "ev60_capex", "ev60_acquisition", "ev60_fundraise_qip",
                                       "block_buys60", "block_sells60", "prom_buys90", "prom_sells90"])
    R["trade_date"] = pd.to_datetime(R["trade_date"])
    R = R[R["trade_date"] <= d].sort_values("trade_date").groupby("symbol").tail(1).set_index("symbol")

    names = ranked
    HOLD_TD = 126
    sub_ann = lambda sym: ann[ann["symbol"] == sym] if not ann.empty else None  # noqa: E731
    ann = _opt(ANN, filters=[("symbol", "in", names)])
    if not ann.empty:
        ann["event_date"] = pd.to_datetime(ann["event_date"], errors="coerce")
    pit = _opt(PIT, columns=["symbol", "intimDt", "personCategory", "tdpTransactionType"])
    if not pit.empty:
        pit = pit[pit["symbol"].isin(names)].copy(); pit["d"] = pd.to_datetime(pit["intimDt"], errors="coerce")
    blk = _opt(BLOCK)
    if not blk.empty:
        blk = blk[blk["BD_SYMBOL"].isin(names)].copy()
        blk["dt"] = pd.to_datetime(blk["BD_DT_DATE"], format="mixed", errors="coerce")
    st = _opt(SUPERSTAR, columns=["symbol", "investor_name"])
    stars = {s: sorted(set(g["investor_name"])) for s, g in st.groupby("symbol")} if not st.empty else {}
    ca = _opt(CA, columns=["symbol", "company_name"])
    company = dict(ca.drop_duplicates("symbol").values) if not ca.empty else {}
    sec_map, ind = _sector_map(), _opt(INDUSTRY)
    macp = _opt(MACRO)
    mac = macp.sort_values("trade_date").iloc[-1] if not macp.empty else None
    news = _opt(NEWS_FEED, columns=["title", "desc", "pub_ts"])
    if not news.empty:
        news["d"] = pd.to_datetime(news["pub_ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        news = news.drop_duplicates("title"); news["txt"] = news["title"].fillna("") + " " + news["desc"].fillna("")
    ft = _opt(FULLTEXT, columns=["symbol", "text"])
    src = dict(prices=str(d.date()), model_score=str(Ps["trade_date"].max().date()) if len(Ps) else "n/a",
               fundamentals_rows=str(R["trade_date"].max().date()),
               announcements=str(ann["event_date"].max().date()) if not ann.empty else "n/a")

    out = [f"# Model-ranked screen — data through {d.date()}",
           "", "**PAPER / NOT VALIDATED** — S1M of EXP-2026-09-28-screen-rank-exit: 33.8%/yr vs production 23.2% (2019+) "
           "but max drawdown -38.8% vs -33.7%, so it failed the registered rule. Hold 126 sessions, no stops.", "",
           f"Pool: {pool_n} core-band stocks pass the trend screen in hot or warming industries (heat percentile >= 0.70); "
           f"top 9 by model score, then {EXTRA} reserves. Sources through: " + " · ".join(f"{k} {v}" for k, v in src.items()), "",
           f"Production screen (theme leaders) today: {', '.join(prod) if prod else 'none'}; overlap: "
           f"{', '.join(sorted(set(prod) & set(ranked[:9]))) or 'none'}", ""]
    for i, sym in enumerate(ranked):
        f, r = F.loc[sym], (R.loc[sym] if sym in R.index else pd.Series(dtype=float))
        hp = grp["hp"].get(f["ind"]); gm = grp["mean"].get(f["ind"])
        sc = Ps["ensemble"].get(sym); scd = Ps["trade_date"].get(sym)
        quant = (f"industry **{f['ind']}** heat pct {hp:.2f} ({'HOT' if hp >= 0.9 else 'warming'}), industry 60d {pct(gm)} · "
                 f"own 60d {pct(f['ret60'])}, 12m {pct(f['ret252'])} · {f['close'] / f['lo252'] - 1:+.0%} off 52w low · "
                 f"{f['close'] / f['sma_200'] - 1:+.0%} vs 200DMA, 50DMA {f['sma_50'] / f['sma_200'] - 1:+.0%} vs 200DMA · "
                 f"model score {sc:.2f} ({scd.date() if pd.notna(scd) else 'n/a'}) · "
                 f"PE {r.get('pe', np.nan):.1f} = {r.get('pe_ind', np.nan):.2f}x industry · EPS YoY {pct(r.get('eps_yoy'))}, "
                 f"sales YoY {pct(r.get('sales_yoy'))} · promoter {r.get('promoter_pct', np.nan):.1f}% "
                 f"(Δ {r.get('prom_delta', np.nan):+.2f}) · mcap Rs {r.get('mcap_cr', np.nan):,.0f} cr · "
                 f"delivery {r.get('delivery_pct', np.nan) * 100:.0f}% · RSI {r.get('rsi_14_daily', np.nan):.0f}")
        sub = lambda df, col="symbol": df[df[col] == sym] if not df.empty else None  # noqa: E731
        qual = _qual(sym, company.get(sym), f["ind"], sub(ann), sub(pit), sub(blk, "BD_SYMBOL"), stars.get(sym, []), d)
        extra = [x for x in (_takeover(sym, sub(ann), ft, d), _news(sym, company.get(sym), news, d)) if x]
        if extra:
            qual = " · ".join(extra) + " · " + qual
        macro = _macro(sym, "n/a (sleeve is not regime-gated)", sec_map, ind, mac)
        tag = f"#{i + 1}" if i < 9 else f"R{i - 8}"
        out += [f"## {tag} {sym}" + (" (also in production)" if sym in prod else ""), "",
                f"- **QUANT:** {quant}", f"- **QUAL:** {qual}", f"- **MACRO:** {macro}", ""]
    # deployment table under the TESTED rule: equal weight inside the batch, entry at the next session's open (a name
    # upper-circuit locked at that open is skipped), 126-session time exit, no stop and no target (exits lowered returns).
    # The AMO limit (last close +5%) only approximates "buy at the open": an open at/above the 4.9% UC-lock threshold
    # does not fill. The exit date counts weekdays only; NSE holidays push it a few sessions later.
    entry = pd.bdate_range(d + pd.Timedelta(days=1), periods=1)[0]
    exit_est = pd.bdate_range(entry, periods=HOLD_TD)[-1]
    top = [s_ for s_ in ranked][:9]
    plan = ["## Deployment plan (paper, tested rule)", "",
            f"Entry: open of {entry.date()} (AMO before the 09:00-09:07 IST pre-open). Each name = 1/{len(top)} of the batch; "
            f"in the tested ladder a weekly batch is 1/26 of the sleeve. Exit: close of session 126, about {exit_est.date()} "
            "(plus NSE holidays). No stop-loss, no profit target.", "",
            "| Rank | Stock | Weight | Last close | AMO buy limit (+5%) | +50% level | Flags |", "|---|---|---|---|---|---|---|"]
    for i, sym in enumerate(ranked):
        c = float(F.loc[sym, "close"])
        fl = "⚠ takeover target" if _takeover(sym, sub_ann(sym), ft, d) else ""
        w = f"{100 / len(top):.1f}%" if i < 9 else "reserve"
        plan.append(f"| {'#' + str(i + 1) if i < 9 else 'R' + str(i - 8)} | {sym} | {w} | {c:,.2f} | {c * 1.05:,.2f} | {c * 1.5:,.2f} | {fl} |")
    out += plan
    # 2026-09-28 (user: "you gave a new batch — what about it?"): the top 9 become an immutable PAPER cohort in
    # logs/model_screen/, one per ISO week like the production sleeve, scored daily by score_model_screen.py with the
    # sleeve's contract (next open, 126 sessions, 0.5% round trip). Kept apart from logs/leader_sleeve so the registered
    # sleeve's record stays clean. rtw is display-only in the scorer; set to 10 - rank.
    cdir = ROOT / "logs/model_screen"
    cdir.mkdir(parents=True, exist_ok=True)
    wk = d.isocalendar()[:2]
    have = [p for p in cdir.glob("screen_*.json")
            if tuple(pd.Timestamp(json.loads(p.read_text())["data_through"]).isocalendar()[:2]) == tuple(wk)]
    if have:
        print(f"weekly cadence: {have[0].name} already covers ISO week {wk[1]} — no second cohort")
    else:
        names_out = [dict(rank=i + 1, symbol=sym, rtw=10 - (i + 1), industry=str(F.loc[sym, "ind"]),
                          close=round(float(F.loc[sym, "close"]), 2),
                          model_score=None if pd.isna(Ps["ensemble"].get(sym)) else round(float(Ps["ensemble"].get(sym)), 4),
                          heat_pct=round(float(grp["hp"].get(F.loc[sym, "ind"])), 3),
                          own60=round(float(F.loc[sym, "ret60"]), 4), own252=round(float(F.loc[sym, "ret252"]), 4),
                          takeover=bool(_takeover(sym, sub_ann(sym), ft, d)))
                     for i, sym in enumerate(ranked[:9])]
        (cdir / f"screen_{d.strftime('%Y%m%d')}.json").write_text(json.dumps(dict(
            screen_id=d.strftime("%Y%m%d"), data_through=str(d.date()), entry="next session open after data_through",
            hold_td=HOLD_TD, cost_rt_pct=0.5, exit="TIME (126 sessions, close); no stops", sizing="PAPER",
            status="NOT VALIDATED: S1M of EXP-2026-09-28-screen-rank-exit failed the registered drawdown rule",
            rule="core band; close >= 1.5x 252-row low, ret252 >= 30%, close > SMA200, SMA50 > SMA200; industry heat pct "
                 ">= 0.70; top 9 by bake-off ensemble score", names=names_out,
            reserves=[dict(rank=i + 1, symbol=sym) for i, sym in enumerate(ranked[9:])]), indent=1))
        print(f"saved paper cohort logs/model_screen/screen_{d.strftime('%Y%m%d')}.json ({len(names_out)} names)")
    rep = ROOT / f"reports/model_screen_{d.strftime('%Y%m%d')}.md"
    rep.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
