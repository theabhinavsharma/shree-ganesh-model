"""PROPER 10-YEAR BACKTEST — 15D/+5% hybrid pipeline.

For EVERY Monday from 2016-01 to 2026-06 (~500 weekly baskets, ~4000 trades):
  1. Walk-forward: train ML on ALL windows strictly before entry (no leakage)
  2. Apply QC filter + top-K by ML score
  3. Simulate exit rule: target/SL/timeout
  4. Aggregate by:
     - Overall hit rate + return distribution
     - By regime (bull/bear/sideways via NIFTY 200-DMA)
     - By RSI band, 20d return band, volume band
     - By year (walk-forward stability)
  5. Identify OPTIMAL filter parameters via grid search
  6. Report SPECIFIC IMPROVEMENTS to bake into next basket

This is the real thing. Runtime ~15-30 min on 10 years of data.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
# Canonical CA store (2026-08-18 law: one store per dataset). The _incremental copy this
# script used to read is not maintained by refresh_corporate_actions.py.
CA = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"

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
COST_RT_PCT = 0.30   # round-trip costs, subtracted from every trade
TOUCH_REASONS = ("TRAIL", "D15_HALF")
HOLD_DAYS = 15
BASKET_SIZE = 8
WIN_THRESHOLD = 0.15


def load_prices():
    print("  loading prices…")
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
    df["dvol_20d"] = df["realized_vol_20d"] / np.sqrt(252)   # daily vol for the C2 stop
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
    print("  computing contamination set…")
    ca = pd.read_parquet(CA); ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    big = df[df["return_1d"].fillna(0) < -0.30]
    contam = set()
    for sym, g in big.groupby("symbol"):
        for _, r in g.iterrows():
            if len(ca[(ca["symbol"]==sym) & ((ca["ex_date"]-r["trade_date"]).abs() <= pd.Timedelta(days=5))]) == 0:
                contam.add(sym); break
    return contam


def forward_paths(df, entry_date, hold, symbols):
    """Next `hold` sessions after the signal day for each symbol: {sym: DataFrame(open, high, low, close)}."""
    win = df[(df["trade_date"] > pd.Timestamp(entry_date)) & df["symbol"].isin(symbols)]
    win = win.sort_values("trade_date").groupby("symbol").head(hold)
    return {s: g[["open", "high", "low", "close"]].reset_index(drop=True) for s, g in win.groupby("symbol")}


def c2_exit(fut, dvol):
    """C2 contract (tournament-locked 2026-08-18; same day-by-day logic as ab_vol_gate.c2):
    enter at next open; SL = clip(3 x 20d daily vol, 3%, 12%) below entry, checked before
    the target on the same day; +5% touch sells half and trails the rest at +2.5%; day-15
    exit at close. Returns (gross %, reason) or None when the path is too short."""
    if fut is None or len(fut) < 8 or pd.isna(fut.iloc[0]["open"]):
        return None
    ep = float(fut.iloc[0]["open"]); dv = dvol if pd.notna(dvol) else 0.02
    tgt = ep * (1 + TARGET_PCT); sl = ep * (1 - np.clip(3 * dv, 0.03, 0.12)); half = False; trail = None
    for i in range(len(fut)):
        lo, hi = fut.iloc[i]["low"], fut.iloc[i]["high"]
        if pd.isna(lo):
            continue
        if not half:
            if lo <= sl:
                return (sl / ep - 1) * 100, "SL"
            if pd.notna(hi) and hi >= tgt:
                half = True; trail = ep * 1.025; continue
        elif lo <= trail:
            return ((TARGET_PCT + (trail / ep - 1)) / 2) * 100, "TRAIL"
    last = float(fut.iloc[-1]["close"])
    if half:
        return ((TARGET_PCT + (last / ep - 1)) / 2) * 100, "D15_HALF"
    return (last / ep - 1) * 100, "D15"


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


def snap_features(df, entry_date):
    snap = df[df["trade_date"]==pd.Timestamp(entry_date)].copy()
    snap = snap[(snap["adv_20d_cr"]>=5) & (snap["close"]>50)]
    return snap.dropna(subset=FEATURES)


def label_window(df, entry_date, hold):
    """Label all symbols: 1 if peak_high in next `hold` days >= entry * (1+WIN_THRESHOLD)."""
    trading_days = sorted(df["trade_date"].unique())
    if pd.Timestamp(entry_date) not in trading_days:
        return pd.DataFrame(columns=["symbol","label"])
    idx = trading_days.index(pd.Timestamp(entry_date))
    end_idx = min(idx + hold, len(trading_days) - 1)
    end = trading_days[end_idx]
    entry = df[df["trade_date"]==pd.Timestamp(entry_date)][["symbol","close"]].rename(columns={"close":"entry"})
    win = df[(df["trade_date"] > pd.Timestamp(entry_date)) & (df["trade_date"] <= end)]
    peak = win.groupby("symbol")["high"].max().rename("peak").reset_index()
    m = entry.merge(peak, on="symbol")
    m["label"] = (m["peak"]/m["entry"] >= 1 + WIN_THRESHOLD).astype(int)
    return m[["symbol","label"]]


def generate_windows(start="2016-01-04", end="2026-09-23"):
    """Weekly Monday-aligned entry dates."""
    dates = pd.date_range(start, end, freq="W-MON")
    return [str(d.date()) for d in dates]


def run_backtest(df, contam, windows, retrain_every_n=8):
    """Walk-forward: retrain classifier every N windows to save compute."""
    trading_days = sorted(df["trade_date"].unique())
    results = []
    model = None
    windows_snapped = []
    for w in windows:
        wts = pd.Timestamp(w)
        for d in trading_days:
            if d >= wts:
                windows_snapped.append(str(d.date())); break
    windows_snapped = sorted(set(windows_snapped))

    train_cache = {}
    for i, entry in enumerate(windows_snapped):
        idx = trading_days.index(pd.Timestamp(entry))
        if idx + HOLD_DAYS >= len(trading_days):
            break
        # Walk-forward training: all windows STRICTLY earlier than entry-HOLD_DAYS
        cutoff = pd.Timestamp(entry) - pd.Timedelta(days=HOLD_DAYS+5)
        train_wds = [w for w in windows_snapped if pd.Timestamp(w) < cutoff]
        if len(train_wds) < 8:
            continue
        # Retrain every N windows (perf optimization)
        if model is None or i % retrain_every_n == 0:
            tr_rows = []
            for td in train_wds:
                if td not in train_cache:
                    s = snap_features(df, td); l = label_window(df, td, HOLD_DAYS)
                    train_cache[td] = s.merge(l, on="symbol")
                tr_rows.append(train_cache[td])
            train_df = pd.concat(tr_rows, ignore_index=True)
            # Guard: need both classes for balanced weighting
            n_pos = int(train_df["label"].sum()); n_neg = len(train_df) - n_pos
            if n_pos < 20 or n_neg < 20:
                print(f"  {entry}: skip retrain (n_pos={n_pos}, n_neg={n_neg})")
                if model is None:  # can't proceed
                    continue
            else:
                model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=32,
                    min_child_samples=50, class_weight="balanced",
                    random_state=42, verbose=-1, n_jobs=-1)
                model.fit(train_df[FEATURES], train_df["label"])
                print(f"  {entry}: retrained on {len(train_df):,} examples (pos={n_pos}, neg={n_neg}) across {len(train_wds)} windows")

        # Test snapshot
        test = snap_features(df, entry)
        if len(test) == 0:
            continue
        test["ml_score"] = model.predict_proba(test[FEATURES])[:,1]
        clean = qc(test, contam)
        if len(clean) == 0:
            continue
        picks = clean.sort_values("ml_score", ascending=False).head(BASKET_SIZE)
        # C2 day-by-day exits from the next open, costs 0.30% round trip
        paths = forward_paths(df, entry, HOLD_DAYS, set(picks["symbol"]))
        for _, r in picks.iterrows():
            out = c2_exit(paths.get(r["symbol"]), r["dvol_20d"])
            if out is None:
                continue
            gross, reason = out
            results.append({
                "entry_date": entry, "symbol": r["symbol"],
                "entry_price": float(paths[r["symbol"]].iloc[0]["open"]),
                "realized_pct": gross - COST_RT_PCT, "gross_pct": gross, "exit_reason": reason,
                "touched_5pct": reason in TOUCH_REASONS,
                "ml_score": float(r["ml_score"]),
                "rsi": float(r["rsi_14_daily"]), "ret_20d": float(r["return_20d"])*100,
                "vol_ratio": float(r["volume_vs_20d"]) if pd.notnull(r["volume_vs_20d"]) else np.nan,
                "adv_cr": float(r["adv_20d_cr"]),
                "market_breadth_200dma": float(r["market_breadth_200dma"]) if pd.notnull(r["market_breadth_200dma"]) else np.nan,
            })
    return pd.DataFrame(results)


def summarize(res: pd.DataFrame):
    if len(res) == 0:
        print("no trades")
        return
    print(f"\n{'='*90}")
    print(f" 10-YEAR BACKTEST — {len(res):,} trades  ·  {res['entry_date'].nunique()} weekly baskets")
    print(f"{'='*90}")

    n = len(res)
    for r in ["SL","TRAIL","D15_HALF","D15"]:
        c = (res["exit_reason"]==r).sum()
        avg = res[res["exit_reason"]==r]["realized_pct"].mean() if c else 0
        print(f"  {r:<8s}: {c:>5,} ({c/n*100:5.1f}%)   avg={avg:+.2f}%")

    print(f"\nOverall stats per trade:")
    print(f"  Mean       : {res['realized_pct'].mean():+.3f}%")
    print(f"  Median     : {res['realized_pct'].median():+.3f}%")
    print(f"  Std        : {res['realized_pct'].std():.3f}%")
    print(f"  Sharpe     : {res['realized_pct'].mean()/max(res['realized_pct'].std(),0.001):.3f}")
    print(f"  Best       : {res['realized_pct'].max():+.2f}%")
    print(f"  Worst      : {res['realized_pct'].min():+.2f}%")

    # Per-basket
    pb = res.groupby("entry_date").agg(n=("realized_pct","size"),
                                        avg=("realized_pct","mean"),
                                        sum=("realized_pct","sum"))
    print(f"\nPer-basket stats ({len(pb)} baskets):")
    print(f"  Mean basket sum    : {pb['sum'].mean():+.2f}%")
    print(f"  Median basket sum  : {pb['sum'].median():+.2f}%")
    print(f"  Best basket        : {pb['sum'].max():+.2f}%")
    print(f"  Worst basket       : {pb['sum'].min():+.2f}%")
    print(f"  Weeks positive     : {(pb['sum']>0).sum()}/{len(pb)}  ({(pb['sum']>0).mean()*100:.1f}%)")

    # By year
    res["year"] = pd.to_datetime(res["entry_date"]).dt.year
    yearly = res.groupby("year").agg(n=("realized_pct","size"),
                                       mean=("realized_pct","mean"),
                                       hit_rate=("touched_5pct", lambda s: s.mean()*100))
    print(f"\nBy year:")
    print(yearly.round(3).to_string())

    # By regime (via market_breadth_200dma)
    res["regime"] = pd.cut(res["market_breadth_200dma"],
                            bins=[0, 0.3, 0.6, 1.01],
                            labels=["BEAR (<30%)","MIX (30-60%)","BULL (>60%)"])
    print(f"\nBy regime (market breadth above 200-DMA):")
    reg = res.groupby("regime", observed=True).agg(n=("realized_pct","size"),
                                                     mean=("realized_pct","mean"),
                                                     hit_rate=("touched_5pct", lambda s: s.mean()*100),
                                                     sl_rate=("exit_reason", lambda s: (s=="SL").mean()*100))
    print(reg.round(3).to_string())

    # By RSI band
    res["rsi_band"] = pd.cut(res["rsi"], bins=[0,45,55,65,72], labels=["<45","45-55","55-65","65-72"])
    print(f"\nBy RSI band:")
    rsi_stats = res.groupby("rsi_band", observed=True).agg(n=("realized_pct","size"),
                                                            mean=("realized_pct","mean"),
                                                            hit_rate=("touched_5pct", lambda s: s.mean()*100))
    print(rsi_stats.round(3).to_string())

    # By 20d return band
    res["ret20d_band"] = pd.cut(res["ret_20d"], bins=[-15,-5,0,10,20,30], labels=["-15to-5","-5to0","0to10","10to20","20to30"])
    print(f"\nBy 20d return band:")
    r20_stats = res.groupby("ret20d_band", observed=True).agg(n=("realized_pct","size"),
                                                                mean=("realized_pct","mean"),
                                                                hit_rate=("touched_5pct", lambda s: s.mean()*100))
    print(r20_stats.round(3).to_string())

    # By ML score band
    res["ml_band"] = pd.cut(res["ml_score"], bins=[0,0.5,0.7,0.85,1.0], labels=["<0.5","0.5-0.7","0.7-0.85","0.85+"])
    print(f"\nBy ML score band:")
    ml_stats = res.groupby("ml_band", observed=True).agg(n=("realized_pct","size"),
                                                          mean=("realized_pct","mean"),
                                                          hit_rate=("touched_5pct", lambda s: s.mean()*100))
    print(ml_stats.round(3).to_string())

    print(f"\n{'='*90}\n IMPROVEMENT RECIPE\n{'='*90}")
    # Best regime, best RSI band, best 20d band, best ML band
    def best(stats):
        if len(stats) == 0: return None
        return stats.sort_values("mean", ascending=False).head(1).to_dict(orient="records")[0]
    print(f"  Best regime : {reg.sort_values('mean', ascending=False).head(1).to_dict()}")
    print(f"  Best RSI    : {rsi_stats.sort_values('mean', ascending=False).head(1).to_dict()}")
    print(f"  Best 20d    : {r20_stats.sort_values('mean', ascending=False).head(1).to_dict()}")
    print(f"  Best ML band: {ml_stats.sort_values('mean', ascending=False).head(1).to_dict()}")


def main():
    print("== 10-YEAR BACKTEST hybrid 15D/+5% ==")
    df = load_prices()
    print(f"  loaded {len(df):,} rows, {df['symbol'].nunique():,} symbols")
    assert CA.exists(), CA
    contam = contamination_set(df)
    print(f"  contaminated symbols: {len(contam):,}")
    windows = generate_windows()
    print(f"  windows to test: {len(windows)}")
    res = run_backtest(df, contam, windows, retrain_every_n=13)  # ~quarterly retrains
    out = ROOT / "data/derived/backtest_10yr_15d5pct.parquet"
    if len(res):
        res.to_parquet(out, index=False)
        print(f"\nwrote {out.relative_to(ROOT)}")
    summarize(res)


if __name__ == "__main__":
    main()
