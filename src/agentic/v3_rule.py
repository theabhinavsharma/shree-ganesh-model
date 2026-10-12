"""Sri Lakshmi V3 rule, in one place (2026-10-02 cleanup; before this it was copy-pasted into 9 scripts).

V3 = G1 top 9 (model-ranked trend screen in hot/warming industries minus the budget/activity veto) minus Financial
Services stocks minus stocks whose own filings tie them only to fading themes; NOT refilled.
Used by the live screen (screen_sri_lakshmi.py) and the V3 reports/explorations. The registered tests
(test_theme_engine.py, test_v3_pnl.py) keep their own frozen copies on purpose: they are the evidence behind the rule
and must not change when this file does (reproduce.py re-runs them).
"""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def context(imap: pd.Series) -> dict:
    """Everything the rule needs besides prices and model scores: G1 eligibility, the finance test, the fading test."""
    import build_themes
    S = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet"); S["date"] = pd.to_datetime(S["date"])
    sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
    sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry", "broad_sector"])
    sc["industry"] = sc["industry"].map(html.unescape)
    sector = sc.groupby("industry")["broad_sector"].agg(lambda x: x.mode().iat[0])
    fading_only, filings_to, themes_to = build_themes.fading_checker()
    fin = lambda s: sector.get(imap.get(s)) == "Financial Services"  # noqa: E731
    return dict(scores=S, G1=S[(S["heat_pct"] >= 0.70) & ~(S["P_pct"] < 0.30)], vetoed=S[(S["heat_pct"] >= 0.70) & (S["P_pct"] < 0.30)],
                fin=fin, fading_only=fading_only, keep=lambda s, d: not fin(s) and not fading_only(s, d),
                filings_to=filings_to, themes_to=themes_to)


def picks(F: pd.DataFrame, dates: list, imap: pd.Series, P: pd.DataFrame, ctx: dict, top: int = 9) -> dict:
    """{date: V3 picks}: the G1 top `top` by model score, then financials and fading-only names dropped, no refill."""
    import test_industry_fundamentals as tif
    import sim_screen_rank_exit as sre
    keep_top = sre.TOPN; sre.TOPN = top
    try:
        g1 = tif.select_elig(F, dates, imap, P, ctx["G1"])
    finally:
        sre.TOPN = keep_top
    return {d: [s for s in g1.get(d, []) if ctx["keep"](s, d)] for d in dates}


# ---------------------------------------------------------------- V3.1 (live from the 2026-10-12 investment, pipeline from 2026-10-16)
# V3 with the stock liquidity bar at Rs 1 cr (20-day average traded value; then-traded price > Rs 50 kept) and liquid new
# listings aged 63-251 sessions skipping the 1-year trend test. Evidence: EXP-2026-10-11-v3-l1-newlist-cost (V3_L1Nc PASS
# 5/5 with liquidity-scaled costs). Same steps as that test's nl_select; `picks` above stays the V3 reference for old tests.
V31_ADV_CR, V31_PRICE = 1.0, 50.0


def frame_v31(F: pd.DataFrame, D: dict) -> pd.DataFrame:
    """X['F'] plus raw price, ADV (Rs cr) and the new-listing flag; `core` = V3.1's liquidity bar."""
    import nse_symbols
    import research_panel as rp
    L = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    L["raw"] = rp.raw_price(L, "close"); L["adv"] = L["avg_traded_value_20d"] / 1e7
    F1 = F.merge(L[["symbol", "trade_date", "raw", "adv"]], on=["symbol", "trade_date"], how="left").sort_values(["symbol", "trade_date"])
    F1["core"] = (F1["adv"] >= V31_ADV_CR) & (F1["raw"] > V31_PRICE)
    firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
    F1["age"] = F1.groupby("symbol").cumcount()
    F1["young"] = F1["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F1["symbol"].isin(ren) & F1["age"].between(63, 251)
    return F1


def ranked_v31(F1: pd.DataFrame, dates: list, imap: pd.Series, P: pd.DataFrame, elig: pd.DataFrame, topn: int = 9) -> dict:
    """{date: top `topn` of the V3.1 pool by model score} before the financial / fading filters (elig = industry rows)."""
    import numpy as np
    W = F1[F1["trade_date"].isin(set(dates)) & F1["core"]].copy()
    W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
    W = W.merge(elig[["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = ((W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])) | W["young"]
    Pool = W[tr].copy()
    Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
    m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
    Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
    g = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict()
    return {d: g.get(d, []) for d in dates}


def picks_v31(F1: pd.DataFrame, dates: list, imap: pd.Series, P: pd.DataFrame, ctx: dict, top: int = 9) -> dict:
    """{date: V3.1 picks}: model top `top` of the V3.1 pool, then financials and fading-only names dropped, no refill."""
    g = ranked_v31(F1, dates, imap, P, ctx["G1"], top)
    return {d: [s for s in g[d] if ctx["keep"](s, d)] for d in dates}



# ---------------------------------------------------------------- V3.2 (live from the 2026-10-12 investment, pipeline from 2026-10-16)
# V3.1 minus names in the trade-for-trade segment (BE / BZ series) on the list date, NOT refilled. Evidence:
# EXP-2026-10-12-v31-t2t (V3.1_NOT2T PASS 5/5: 56.7 vs 51.8 %/yr, 2023+ 64.2 vs 54.9, worst fall -38.7 vs -40.0).
def t2t_flags(symbols: list, d) -> dict:
    """{symbol: True if its latest series on or before d is BE / BZ} from the price panel (NSE bhavcopy series)."""
    import research_panel as rp
    S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"], filters=[("symbol", "in", list(symbols))])
    S["trade_date"] = pd.to_datetime(S["trade_date"]); S = S[S["trade_date"] <= pd.Timestamp(d)].sort_values("trade_date")
    last = S.groupby("symbol")["series"].last()
    return {s: bool(last.get(s) in ("BE", "BZ")) for s in symbols}
