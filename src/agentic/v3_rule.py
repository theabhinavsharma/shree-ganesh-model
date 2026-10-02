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
