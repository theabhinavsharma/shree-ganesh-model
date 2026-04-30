"""Portfolio sizer: given today's long/short candidates, produce a sized,
regime-aware, sector-diversified, Kelly-fractioned portfolio recommendation.

Logic:
  1. Detect market regime (bull / chop / bear) from breadth + 20d return + VIX-proxy
  2. Set base aggression by regime: bull=1.0x, chop=0.5x, bear=0.2x (long-only weight)
  3. Pull v3 long candidates (top 30) and short candidates (top 20)
  4. Apply Kelly sizing per name: f* = (p*b - q) / b   where b = mean_win/mean_loss
     We use FRACTIONAL Kelly (0.25× full Kelly) to dampen drawdown
  5. Cap any single name at 8% of capital and any sector at 25%
  6. Enforce that sum(longs) + sum(shorts) <= leverage_cap (default 1.0 unlevered)
  7. Output a CSV the user can hand to a broker
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom/tmp/from_scratch_7d_run")


def detect_regime() -> dict:
    """Return regime label + breadth/return/vol stats from latest day."""
    df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "close", "sma_50", "sma_200",
                                  "return_1d", "return_20d", "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["adv_cr"] = df["avg_traded_value_20d"] / 1e7
    df["above_50"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200"] = (df["close"] > df["sma_200"]).astype(int)

    latest = df["trade_date"].max()
    snap = df[(df["trade_date"] == latest) & (df["adv_cr"] >= 1.0)]
    breadth_50 = snap["above_50"].mean()
    breadth_200 = snap["above_200"].mean()

    # market 20d trend (median of liquid universe)
    liq = df[df["adv_cr"] >= 1.0]
    last20_dates = sorted(liq["trade_date"].unique())[-20:]
    mkt_20d_ret = liq[liq["trade_date"].isin(last20_dates)].groupby("trade_date")["return_1d"].median().sum()

    # vol proxy: cross-sectional dispersion
    snap_vol = liq[liq["trade_date"] == latest]["return_1d"].std()

    if breadth_50 >= 0.65 and mkt_20d_ret >= 0.02:
        regime = "BULL"
        aggression = 1.0
    elif breadth_50 <= 0.40 or mkt_20d_ret <= -0.04:
        regime = "BEAR"
        aggression = 0.2
    else:
        regime = "CHOP"
        aggression = 0.5

    return {
        "as_of": latest,
        "regime": regime,
        "aggression": aggression,
        "breadth_50": breadth_50,
        "breadth_200": breadth_200,
        "market_20d_ret": mkt_20d_ret,
        "cross_sectional_vol_1d": snap_vol,
    }


def kelly_fraction(p: float, win: float, loss: float) -> float:
    """Fractional Kelly: f* = (p*b - q) / b, b = win/|loss|, q = 1-p."""
    if loss == 0 or p <= 0 or win <= 0:
        return 0.0
    b = win / abs(loss)
    q = 1 - p
    f_full = (p * b - q) / b
    return max(0.0, f_full)


def build_portfolio(capital_inr: float = 1_000_000, leverage_cap: float = 1.0,
                    fractional_kelly: float = 0.25, max_per_name: float = 0.08,
                    max_per_sector: float = 0.25, n_long: int = 30,
                    n_short: int = 0) -> pd.DataFrame:
    regime = detect_regime()
    print(f"\n=== regime: {regime['regime']} (aggression={regime['aggression']}) ===")
    print(f"   breadth_50dma={regime['breadth_50']:.1%}  "
          f"breadth_200dma={regime['breadth_200']:.1%}  "
          f"mkt_20d_ret={regime['market_20d_ret']:.2%}  "
          f"x-sec vol={regime['cross_sectional_vol_1d']:.4f}")

    longs_csv = ROOT / "v3_live_top100.csv"
    if not longs_csv.exists():
        raise SystemExit(f"missing {longs_csv} — run v3 ensemble first")
    longs = pd.read_csv(longs_csv).sort_values("score_ens", ascending=False).head(n_long * 2)

    shorts_csv = ROOT / "short_live_top100.csv"
    shorts = pd.read_csv(shorts_csv).sort_values("score_ens", ascending=False).head(n_short * 2) \
        if shorts_csv.exists() and n_short > 0 else pd.DataFrame()

    # historical win/loss expectancy from OOF (close-to-close 7TD return)
    oof_long = pd.read_parquet(ROOT / "v3_oof.parquet")
    px = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "close"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    px["close_7td"] = px.groupby("symbol")["close"].shift(-7)
    px["ret_c7"] = px["close_7td"] / px["close"] - 1

    oof_long = oof_long.merge(px[["symbol", "trade_date", "ret_c7"]], on=["symbol", "trade_date"])

    # bucket by score and compute expectancy
    long_bins = pd.cut(oof_long["score"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    long_exp = oof_long.dropna(subset=["ret_c7"]).groupby(long_bins, observed=False).agg(
        p=("ret_c7", lambda s: (s >= 0).mean()),
        win=("ret_c7", lambda s: s[s >= 0].mean() if (s >= 0).any() else 0),
        loss=("ret_c7", lambda s: s[s < 0].mean() if (s < 0).any() else -0.05),
    )
    print("\nlong expectancy by score band:")
    print(long_exp.round(3).to_string())

    def expectancy_for(score):
        for idx, row in long_exp.iterrows():
            if score in idx:
                return float(row["p"]), float(row["win"]), float(row["loss"])
        return 0.55, 0.07, -0.07

    rows = []
    sector_alloc = {}
    total_alloc = 0.0
    for _, r in longs.iterrows():
        p, w, l = expectancy_for(r["score_ens"])
        f = kelly_fraction(p, w, l) * fractional_kelly * regime["aggression"]
        f = min(f, max_per_name)
        sec = r.get("sector", "OTHER")
        if sector_alloc.get(sec, 0) + f > max_per_sector:
            f = max(0.0, max_per_sector - sector_alloc.get(sec, 0))
        if f <= 0.005:
            continue
        if total_alloc + f > leverage_cap:
            f = max(0.0, leverage_cap - total_alloc)
            if f <= 0.005:
                break
        rows.append({
            "side": "LONG",
            "symbol": r["symbol"],
            "sector": sec,
            "pwin_ens": r["score_ens"],
            "pwin_cal": r["score_calibrated"],
            "expected_p": p,
            "expected_win_pct": w,
            "expected_loss_pct": l,
            "alloc_pct": f,
            "alloc_inr": int(round(f * capital_inr)),
            "shares": int(f * capital_inr / r["close"]) if r["close"] > 0 else 0,
            "entry_close": r["close"],
        })
        sector_alloc[sec] = sector_alloc.get(sec, 0) + f
        total_alloc += f
        if len(rows) >= n_long:
            break

    df = pd.DataFrame(rows)
    print(f"\n=== portfolio ({len(df)} positions) ===")
    print(f"capital   ₹{capital_inr:,.0f}")
    print(f"deployed  ₹{int(df['alloc_inr'].sum()):,}  ({df['alloc_pct'].sum():.1%})")
    print(f"cash      ₹{int(capital_inr - df['alloc_inr'].sum()):,}")
    print()
    print(df.to_string(index=False))
    out = ROOT / "portfolio_today.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--kelly-fraction", type=float, default=0.25)
    p.add_argument("--n-long", type=int, default=20)
    p.add_argument("--n-short", type=int, default=0)
    args = p.parse_args()
    build_portfolio(args.capital, args.leverage, args.kelly_fraction,
                    n_long=args.n_long, n_short=args.n_short)
