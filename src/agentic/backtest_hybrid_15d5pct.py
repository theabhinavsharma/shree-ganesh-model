"""HONEST BACKTEST — hybrid 15D/+5% pipeline.

For each historical entry date (weekly since April):
  1. Train ML classifier on all PRIOR windows (walk-forward)
  2. Score current universe, pick top-8 by ml_score after QC
  3. Simulate trades: TARGET+5% / SL-3% / TIMEOUT 15d
  4. Report aggregate hit rate, avg return, drawdown, per-basket P&L
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/abhinavs./Code/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA = ROOT / "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet"

FEATURES = [
    "rsi_14_daily","rsi_14_weekly","rsi_14_monthly",
    "return_1d","return_20d","ret_5d","ret_10d",
    "dist_sma_50","dist_sma_200","above_50dma","above_200dma",
    "volume_vs_20d","traded_value_vs_20d","delivery_pct",
    "realized_vol_20d","adv_20d_cr",
    "market_5d_ret","market_20d_ret","market_breadth_50dma","market_breadth_200dma",
    "price_log","delivery_pct_vs_20d",
]

TARGET_PCT = 0.05
SL_PCT = -0.03
HOLD_DAYS = 15
BASKET_SIZE = 8
WIN_THRESHOLD = 0.15

WINDOWS = [
    "2026-04-24","2026-05-01","2026-05-08","2026-05-15","2026-05-22",
    "2026-05-27","2026-06-02","2026-06-09","2026-06-16",
]


def load_prices():
    df = pd.read_parquet(PRICES, columns=[
        "symbol","trade_date","close","high","low","open",
        "return_1d","return_20d","rsi_14_daily","rsi_14_weekly","rsi_14_monthly",
        "total_traded_value","volume_vs_20d","traded_value_vs_20d","delivery_pct",
        "delivery_pct_vs_20d","sma_20","sma_50","sma_200"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol","trade_date"])
    df["ret_5d"] = df.groupby("symbol")["close"].pct_change(5)
    df["ret_10d"] = df.groupby("symbol")["close"].pct_change(10)
    df["dist_sma_50"] = df["close"]/df["sma_50"] - 1
    df["dist_sma_200"] = df["close"]/df["sma_200"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["adv_20d_cr"] = df.groupby("symbol")["total_traded_value"].transform(
        lambda s: s.rolling(20).mean()) / 1e7
    df["price_log"] = np.log(df["close"].clip(lower=1))
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(20).std() * np.sqrt(252))
    liq = df[df["adv_20d_cr"]>=1.0]
    mkt = liq.groupby("trade_date").agg(
        market_1d_ret=("return_1d","median"),
        market_breadth_50dma=("above_50dma","mean"),
        market_breadth_200dma=("above_200dma","mean")).reset_index()
    mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
    mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
    df = df.merge(mkt[["trade_date","market_5d_ret","market_20d_ret",
                        "market_breadth_50dma","market_breadth_200dma"]],
                  on="trade_date", how="left")
    return df


def contamination_set(df):
    ca = pd.read_parquet(CA); ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    big = df[df["return_1d"].fillna(0) < -0.30]
    contam = set()
    for sym, g in big.groupby("symbol"):
        for _, r in g.iterrows():
            if len(ca[(ca["symbol"]==sym) & ((ca["ex_date"]-r["trade_date"]).abs() <= pd.Timedelta(days=5))]) == 0:
                contam.add(sym); break
    return contam


def snap_features(df, entry_date):
    snap = df[df["trade_date"]==pd.Timestamp(entry_date)].copy()
    snap = snap[(snap["adv_20d_cr"]>=5) & (snap["close"]>50)]
    return snap.dropna(subset=FEATURES)


def peak_forward(df, entry_date, sym, hold):
    trading_days = sorted(df["trade_date"].unique())
    if pd.Timestamp(entry_date) not in trading_days:
        return None, None, None
    idx = trading_days.index(pd.Timestamp(entry_date))
    end_idx = min(idx + hold, len(trading_days) - 1)
    win = df[(df["symbol"]==sym) & (df["trade_date"] > pd.Timestamp(entry_date)) &
             (df["trade_date"] <= trading_days[end_idx])]
    if len(win) == 0:
        return None, None, None
    return float(win["high"].max()), float(win["low"].min()), float(win["close"].iloc[-1])


def simulate(entry_p, peak_h, min_l, close_end, tgt_pct, sl_pct):
    tgt = entry_p * (1 + tgt_pct)
    sl = entry_p * (1 + sl_pct)
    hit_tgt = peak_h >= tgt
    hit_sl = min_l <= sl
    if hit_tgt and hit_sl:
        return 0.01, "WHIPSAW"
    if hit_tgt:
        remainder_ret = max((close_end/entry_p - 1), 0.025)
        return (tgt_pct + remainder_ret) / 2, "TARGET"
    if hit_sl:
        return sl_pct, "SL"
    return (close_end/entry_p - 1), "TIMEOUT"


def qc(snap, contam):
    return snap[
        (snap["rsi_14_daily"].between(40, 72)) &
        (snap["return_20d"].between(-0.15, 0.30)) &
        (snap["ret_5d"].between(-0.05, 0.10)) &
        (snap["volume_vs_20d"].fillna(1) < 3) &
        (snap["close"] > 50) &
        (snap["adv_20d_cr"] >= 5) &
        (~snap["symbol"].isin(contam))
    ].copy()


def labels(df, entry_date, hold):
    snap = df[df["trade_date"]==pd.Timestamp(entry_date)][["symbol","close"]].rename(columns={"close":"entry"})
    trading_days = sorted(df["trade_date"].unique())
    idx = trading_days.index(pd.Timestamp(entry_date))
    end = trading_days[min(idx+hold, len(trading_days)-1)]
    win = df[(df["trade_date"] > pd.Timestamp(entry_date)) & (df["trade_date"] <= end)]
    peak = win.groupby("symbol")["high"].max().rename("peak").reset_index()
    m = snap.merge(peak, on="symbol")
    m["peak_ret"] = m["peak"]/m["entry"] - 1
    m["label"] = (m["peak_ret"] >= WIN_THRESHOLD).astype(int)
    return m[["symbol","label"]]


def main():
    print("== BACKTEST hybrid 15D/+5% (walk-forward) ==")
    df = load_prices()
    contam = contamination_set(df)
    trading_days = sorted(df["trade_date"].unique())

    # snap all windows to actual trading days
    windows = []
    for w in WINDOWS:
        wts = pd.Timestamp(w)
        if wts in trading_days:
            windows.append(str(wts.date())); continue
        for d in trading_days:
            if d >= wts:
                windows.append(str(d.date())); break
    print(f"windows (snapped to trading days): {windows}\n")

    results = []
    for entry in windows:
        idx = trading_days.index(pd.Timestamp(entry))
        if idx + HOLD_DAYS >= len(trading_days):
            print(f"  {entry}: window not complete — skipping")
            continue
        # Walk-forward training: use windows STRICTLY before this one's exit
        train_wds = [w for w in windows if pd.Timestamp(w) < pd.Timestamp(entry) - pd.Timedelta(days=HOLD_DAYS+5)]
        if len(train_wds) < 2:
            print(f"  {entry}: insufficient train history ({len(train_wds)}) — skipping")
            continue

        tr = []
        for td in train_wds:
            s = snap_features(df, td); l = labels(df, td, HOLD_DAYS)
            tr.append(s.merge(l, on="symbol"))
        train_df = pd.concat(tr, ignore_index=True)

        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=32,
            min_child_samples=50, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)
        model.fit(train_df[FEATURES], train_df["label"])

        test = snap_features(df, entry)
        test["ml_score"] = model.predict_proba(test[FEATURES])[:,1]
        clean = qc(test, contam)
        picks = clean.sort_values("ml_score", ascending=False).head(BASKET_SIZE)

        for _, r in picks.iterrows():
            sym = r["symbol"]; e_p = float(r["close"])
            peak_h, min_l, ce = peak_forward(df, entry, sym, HOLD_DAYS)
            if peak_h is None: continue
            realized, reason = simulate(e_p, peak_h, min_l, ce, TARGET_PCT, SL_PCT)
            results.append({
                "entry_date": entry, "symbol": sym, "entry_price": e_p,
                "peak_ret": (peak_h/e_p-1)*100, "min_ret": (min_l/e_p-1)*100,
                "close_ret": (ce/e_p-1)*100, "realized_pct": realized*100,
                "exit_reason": reason, "ml_score": float(r["ml_score"]),
                "rsi": float(r["rsi_14_daily"]), "ret_20d": float(r["return_20d"])*100,
                "adv_cr": float(r["adv_20d_cr"]),
            })
        print(f"  {entry}: {len(picks)} picks scored & simulated (train windows: {len(train_wds)})")

    if not results:
        print("no results")
        return

    res = pd.DataFrame(results)
    print(f"\n{'='*80}\n BACKTEST RESULTS — {len(res)} trades across {res['entry_date'].nunique()} windows\n{'='*80}")

    n = len(res)
    for reason in ["TARGET","SL","WHIPSAW","TIMEOUT"]:
        c = (res["exit_reason"]==reason).sum()
        print(f"  {reason:<8s}: {c:>3} ({c/n*100:5.1f}%)  avg_realized={res[res['exit_reason']==reason]['realized_pct'].mean():+.2f}%")

    print(f"\nOverall return stats:")
    print(f"  Mean per trade   : {res['realized_pct'].mean():+.3f}%")
    print(f"  Median per trade : {res['realized_pct'].median():+.3f}%")
    print(f"  Std dev          : {res['realized_pct'].std():+.3f}%")
    print(f"  Sharpe per-trade : {res['realized_pct'].mean()/max(res['realized_pct'].std(),0.001):+.3f}")
    print(f"  Best trade       : {res['realized_pct'].max():+.2f}%")
    print(f"  Worst trade      : {res['realized_pct'].min():+.2f}%")

    print(f"\nPeak-high (max intraday potential if you had a perfect exit):")
    print(f"  % hit +5%        : {(res['peak_ret']>=5).mean()*100:.1f}%")
    print(f"  % hit +10%       : {(res['peak_ret']>=10).mean()*100:.1f}%")
    print(f"  % hit +15%       : {(res['peak_ret']>=15).mean()*100:.1f}%")

    print(f"\nDrawdown:")
    print(f"  % dropped ≤ -3%  : {(res['min_ret']<=-3).mean()*100:.1f}%")
    print(f"  % dropped ≤ -5%  : {(res['min_ret']<=-5).mean()*100:.1f}%")

    print(f"\nPer-basket P&L (avg 8 names, per week):")
    pb = res.groupby("entry_date").agg(n=("realized_pct","size"),
                                        avg=("realized_pct","mean"),
                                        sum=("realized_pct","sum"))
    pb["basket_impact_at_1pct_each"] = pb["sum"] * 0.01  # if each name is 1% of portfolio
    print(pb.round(3).to_string())

    print(f"\nAggregate basket impact (avg across weeks, at 1%/name = 8% deploy):")
    print(f"  Mean basket return : {pb['sum'].mean():+.2f}%")
    print(f"  Mean portfolio impact (8% deploy): {pb['basket_impact_at_1pct_each'].mean():+.3f}%")
    print(f"  Best week          : {pb['sum'].max():+.2f}%")
    print(f"  Worst week         : {pb['sum'].min():+.2f}%")
    print(f"  Weeks positive     : {(pb['sum']>0).sum()}/{len(pb)}")

    out = ROOT / "data/derived/backtest_hybrid_15d5pct.parquet"
    res.to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
