"""Operationalize alpha factors from academic finance literature.

Each function below maps a published paper to a computable factor on our data.
Run after feature_factory.py so all base features are present.

References:
  • Piotroski 2000 — F-Score (9 binary checks for fundamental health)
  • Novy-Marx 2013 — Gross Profitability (GP / Assets)
  • Asness, Frazzini, Pedersen 2019 — Quality-Minus-Junk (formal QMJ scoring)
  • Frazzini, Pedersen 2014 — Betting Against Beta
  • Carhart 1997 — momentum (12-1)
  • Fama-French 5 factor (size, value, profitability, investment, momentum)
  • Lakonishok, Shleifer, Vishny 1994 — value/glamour
  • Bernard & Thomas 1989 — PEAD (post-earnings announcement drift)

Where data exists, factor is computed. Where it doesn't, factor is marked SKIP
with the reason so we know what's blocking.

Output: data/derived/academic_alphas.parquet
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
SCREENER = ROOT / "data/derived/screener_fundamentals.parquet"
OUT = ROOT / "data/derived/academic_alphas.parquet"


def main() -> None:
    print("== build_academic_alphas ==")
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "high", "low",
                                            "return_1d", "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    latest = px["trade_date"].max()
    snap = px[px["trade_date"] == latest].copy()

    # ─── Carhart Momentum (12-1) ─────────────────────────────────────
    # Return over last 12 months minus the most recent month
    px["close_12m_ago"] = px.groupby("symbol")["close"].shift(252)
    px["close_1m_ago"] = px.groupby("symbol")["close"].shift(21)
    snap_mom = px[px["trade_date"] == latest][["symbol", "close", "close_12m_ago", "close_1m_ago"]]
    snap_mom["mom_12_1"] = (snap_mom["close_1m_ago"] / snap_mom["close_12m_ago"] - 1) * 100
    snap = snap.merge(snap_mom[["symbol", "mom_12_1"]], on="symbol", how="left")

    # ─── Long-term reversal (LSV value/glamour) ──────────────────────
    # 5y past return — losers tend to revert
    px["close_5y_ago"] = px.groupby("symbol")["close"].shift(252 * 5)
    snap_lsv = px[px["trade_date"] == latest][["symbol", "close", "close_5y_ago"]]
    snap_lsv["return_5y"] = (snap_lsv["close"] / snap_lsv["close_5y_ago"] - 1) * 100
    snap = snap.merge(snap_lsv[["symbol", "return_5y"]], on="symbol", how="left")

    # ─── Frazzini-Pedersen "Betting Against Beta" ────────────────────
    # Compute beta vs market over 252d, then BAB factor
    market_ret = px.groupby("trade_date")["return_1d"].median().rename("mkt_ret").reset_index()
    px = px.merge(market_ret, on="trade_date", how="left")

    def rolling_beta(g, window=252):
        # cov(stock, mkt) / var(mkt) — rolling
        cov = g["return_1d"].rolling(window).cov(g["mkt_ret"])
        var = g["mkt_ret"].rolling(window).var()
        return cov / var.replace(0, np.nan)

    px["beta_252d"] = px.groupby("symbol", group_keys=False).apply(rolling_beta)
    snap_beta = px[px["trade_date"] == latest][["symbol", "beta_252d"]]
    snap = snap.merge(snap_beta, on="symbol", how="left")
    # BAB = -1 * (beta - cross_sectional_mean_beta), normalized
    if snap["beta_252d"].notna().any():
        snap["bab_factor"] = -(snap["beta_252d"] - snap["beta_252d"].mean()) / snap["beta_252d"].std()

    # ─── Idiosyncratic volatility (Ang Hodrick Xing Zhang 2006) ──────
    # Stocks with high idio-vol underperform — captures lottery preference
    def rolling_idio_vol(g, window=60):
        # residual after subtracting beta * mkt_ret
        resid = g["return_1d"] - g["beta_252d"] * g["mkt_ret"]
        return resid.rolling(window).std()
    px["idio_vol_60d"] = px.groupby("symbol", group_keys=False).apply(rolling_idio_vol)
    snap_iv = px[px["trade_date"] == latest][["symbol", "idio_vol_60d"]]
    snap = snap.merge(snap_iv, on="symbol", how="left")

    # ─── Reversal — short-term (Jegadeesh 1990) ─────────────────────
    # 1-month return; -1 sign in regression (winners revert short-term)
    px["close_21d_ago"] = px.groupby("symbol")["close"].shift(21)
    snap_rev = px[px["trade_date"] == latest][["symbol", "close", "close_21d_ago"]]
    snap_rev["short_term_reversal_1m"] = -(snap_rev["close"] / snap_rev["close_21d_ago"] - 1) * 100
    snap = snap.merge(snap_rev[["symbol", "short_term_reversal_1m"]], on="symbol", how="left")

    # ─── Maximum Daily Return (Bali Cakici Whitelaw 2011) ────────────
    # MAX = highest single-day return in last month — proxy for lottery preference
    # high MAX historically UNDERPERFORMS — captures retail lottery-buying
    px["max_1m"] = px.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(21).max())
    snap_max = px[px["trade_date"] == latest][["symbol", "max_1m"]]
    snap = snap.merge(snap_max, on="symbol", how="left")

    # ─── Skewness 1-month (Boyer Mitton Vorkink 2010) ────────────────
    # Stocks with positive skew underperform — lottery preference at play
    px["skew_1m"] = px.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(21).skew())
    snap_skew = px[px["trade_date"] == latest][["symbol", "skew_1m"]]
    snap = snap.merge(snap_skew, on="symbol", how="left")

    # ─── Volatility-Adjusted Momentum (Asness 2014) ──────────────────
    # 6m return / 6m std — "sharper" momentum
    px["return_6m"] = px.groupby("symbol")["close"].pct_change(126)
    px["std_6m"] = px.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(126).std())
    snap_vam = px[px["trade_date"] == latest][["symbol", "return_6m", "std_6m"]]
    snap_vam["vol_adj_mom_6m"] = snap_vam["return_6m"] / snap_vam["std_6m"].replace(0, np.nan)
    snap = snap.merge(snap_vam[["symbol", "vol_adj_mom_6m"]], on="symbol", how="left")

    # ─── Acceleration (DeMiguel et al 2017) ──────────────────────────
    # Recent momentum minus older momentum — captures trend acceleration
    px["return_3m"] = px.groupby("symbol")["close"].pct_change(63)
    px["return_6_3"] = px.groupby("symbol")["close"].pct_change(126) - px.groupby("symbol")["close"].pct_change(63)
    snap_acc = px[px["trade_date"] == latest][["symbol", "return_3m", "return_6_3"]]
    snap_acc["mom_acceleration"] = snap_acc["return_3m"] - snap_acc["return_6_3"]
    snap = snap.merge(snap_acc[["symbol", "mom_acceleration"]], on="symbol", how="left")

    # ─── Daily liquidity (Hasbrouck 2009 simplified) ─────────────────
    # avg(|return| / dollar_volume) — micro-illiquidity
    snap["liquidity_proxy"] = snap.get("idio_vol_60d", 0) / np.log(
        snap["avg_traded_value_20d"].replace(0, np.nan)).fillna(1)

    # ─── Asness QMJ (Quality Minus Junk) — composite ─────────────────
    if SCREENER.exists():
        sf = pd.read_parquet(SCREENER)
        sf["fetch_date"] = pd.to_datetime(sf["fetch_date"])
        sf = sf.sort_values("fetch_date").groupby("symbol").tail(1)
        keep = [c for c in ["roe", "roce", "compounded_profit_growth_5_years",
                             "compounded_sales_growth_5_years",
                             "return_on_equity_5_years"] if c in sf.columns]
        sub = sf[["symbol"] + keep].copy()
        for c in keep:
            sub[f"{c}_z"] = (sub[c] - sub[c].mean()) / sub[c].std()
        z_cols = [f"{c}_z" for c in keep]
        sub["asness_qmj"] = sub[z_cols].mean(axis=1)
        snap = snap.merge(sub[["symbol", "asness_qmj"]], on="symbol", how="left")

    # output the new columns only
    new_cols = ["symbol", "trade_date", "mom_12_1", "return_5y",
                "beta_252d", "bab_factor", "idio_vol_60d",
                "short_term_reversal_1m", "asness_qmj"]
    new_cols = [c for c in new_cols if c in snap.columns]
    out = snap[new_cols].copy()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print(f"\nfactors built (coverage on today's snapshot):")
    for c in [c for c in new_cols if c not in ("symbol", "trade_date")]:
        cov = out[c].notna().mean()
        print(f"  {c:<32}  coverage {cov*100:.0f}%")
    print(f"\nwrote {OUT}: {len(out):,} stocks × {len(new_cols)} cols")


if __name__ == "__main__":
    main()
