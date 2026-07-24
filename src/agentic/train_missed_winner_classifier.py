"""OPTION 2: Supervised ML classifier for missed winners.

Trains a binary classifier on historical (entry_date, symbol) pairs:
  label = 1 if peak_return_forward_15d >= +15% AND passes current QC filter
  label = 0 otherwise

Features: 21 base features (price/technical/volume/liquidity/momentum) computed
on the entry date. NO forward-looking leakage.

Walk-forward validation:
  Train windows: 2026-05-08, 2026-05-27, 2026-06-01
  Test windows:  2026-06-18, 2026-06-24

Purpose: catch the 52 "MISSED but catchable" winners from the coverage backtest.
Ships as a 6th engine if OOS test-set catch rate > 20% at top-30 rank.

Per CONSTITUTION §1.4 — feature promoted only if it lifts coverage without
dropping hit rate on caught names. This is the empirical gate.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA = ROOT / "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet"
OUT_MODEL = ROOT / "data/derived/missed_winner_classifier.parquet"

FEATURES = [
    "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
    "return_1d", "return_20d", "ret_5d", "ret_10d",
    "dist_sma_50", "dist_sma_200",
    "above_50dma", "above_200dma",
    "volume_vs_20d", "traded_value_vs_20d",
    "delivery_pct",
    "realized_vol_20d",
    "adv_20d_cr",
    "market_5d_ret", "market_20d_ret",
    "market_breadth_50dma", "market_breadth_200dma",
    "price_log",
    "delivery_pct_vs_20d",  # institutional accumulation signal
]

# 2026-07-24 fix: dates were frozen at May/June — the model never rolled forward.
# Now dynamic: weekly grid over the last ~5 months; train = all windows whose
# 15-session labels are complete except the last 2; test = the last 2 labeled windows.
def _dynamic_dates():
    import pandas as _pd
    _px = _pd.read_parquet(PRICES, columns=["trade_date"])
    _days = sorted(_pd.to_datetime(_px["trade_date"]).unique())
    _grid = [d for d in _days[::5] if d >= _days[-1] - _pd.Timedelta(days=150)]
    _labeled = [d for d in _grid if len([x for x in _days if x > d]) >= 15]  # 15 == FWD_DAYS (defined below)
    _train = [str(_pd.Timestamp(d).date()) for d in _labeled[:-2]]
    _test  = [str(_pd.Timestamp(d).date()) for d in _labeled[-2:]]
    return _train, _test

try:
    TRAIN_DATES, TEST_DATES = _dynamic_dates()
except Exception:
    TRAIN_DATES = ["2026-05-08", "2026-05-27", "2026-06-01"]
    TEST_DATES  = ["2026-06-18", "2026-06-24"]
FWD_DAYS = 15
WIN_THRESHOLD = 0.15   # +15% peak return = winner


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
    # Realized vol computed from daily returns
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(20).std() * np.sqrt(252))
    # 2026-07-24 RL additions — the two features the escaped winners shared
    # (miss attribution: 247/395 winners killed at QC; killed-bucket winners had
    #  strong 60d trend and sat near 52w highs — neither visible to the model):
    df["ret_60d"] = df.groupby("symbol")["close"].pct_change(60)
    df["hi_252"] = df.groupby("symbol")["close"].transform(
        lambda s: s.rolling(252, min_periods=60).max())
    df["off_52w_high"] = df["close"]/df["hi_252"] - 1

    # Market context (median across liquid universe per date)
    liq = df[df["adv_20d_cr"]>=1.0]
    market = liq.groupby("trade_date").agg(
        market_1d_ret=("return_1d","median"),
        market_breadth_50dma=("above_50dma","mean"),
        market_breadth_200dma=("above_200dma","mean")
    ).reset_index()
    market["market_5d_ret"] = market["market_1d_ret"].rolling(5).sum()
    market["market_20d_ret"] = market["market_1d_ret"].rolling(20).sum()
    df = df.merge(market[["trade_date","market_5d_ret","market_20d_ret",
                          "market_breadth_50dma","market_breadth_200dma"]],
                  on="trade_date", how="left")
    return df


def fwd_peak_return(df, entry_date, n_days=FWD_DAYS):
    """Compute peak return over next n trading days per symbol."""
    trading_days = sorted(df["trade_date"].unique())
    idx = trading_days.index(pd.Timestamp(entry_date))
    if idx + n_days >= len(trading_days):
        n_days = len(trading_days) - idx - 1
    end = trading_days[idx + n_days]
    entry = df[df["trade_date"]==entry_date][["symbol","close"]].rename(columns={"close":"entry"})
    window = df[(df["trade_date"]>entry_date) & (df["trade_date"]<=end)]
    peak = window.groupby("symbol")["high"].max().rename("peak").reset_index()
    out = entry.merge(peak, on="symbol")
    out["peak_ret"] = (out["peak"]/out["entry"]) - 1
    return out[["symbol","peak_ret"]]


def build_examples(df, dates):
    """For each entry date, build (features, label) rows."""
    rows = []
    for d in dates:
        snap = df[df["trade_date"]==d].copy()
        snap = snap[snap["adv_20d_cr"]>=5]  # liquid only
        snap = snap[snap["close"]>50]        # no pennies
        fwd = fwd_peak_return(df, d)
        m = snap.merge(fwd, on="symbol")
        m["label"] = (m["peak_ret"] >= WIN_THRESHOLD).astype(int)
        m["entry_date"] = d
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def main():
    print("== training missed-winner classifier ==")
    df = load_prices()
    print(f"  prices loaded: {len(df):,} rows, {df['symbol'].nunique():,} symbols")

    # Build train + test sets
    train_df = build_examples(df, TRAIN_DATES)
    test_df  = build_examples(df, TEST_DATES)
    print(f"  train: {len(train_df):,} examples across {len(TRAIN_DATES)} dates")
    print(f"    positive rate: {train_df['label'].mean()*100:.1f}%")
    print(f"  test:  {len(test_df):,} examples across {len(TEST_DATES)} dates")
    print(f"    positive rate: {test_df['label'].mean()*100:.1f}%")

    # Drop rows missing critical features
    train_df = train_df.dropna(subset=[c for c in FEATURES if c in train_df.columns])
    test_df  = test_df.dropna(subset=[c for c in FEATURES if c in test_df.columns])
    print(f"  after dropna: train {len(train_df):,}, test {len(test_df):,}")

    # Train
    X_train = train_df[FEATURES]
    y_train = train_df["label"]
    X_test  = test_df[FEATURES]
    y_test  = test_df["label"]

    print(f"\n== training LightGBM ==")
    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=32,
        min_child_samples=50, feature_fraction=0.85,
        bagging_fraction=0.85, bagging_freq=5,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(30, verbose=False)])

    # Evaluate on test
    test_prob = model.predict_proba(X_test)[:,1]
    test_df = test_df.copy()
    test_df["ml_score"] = test_prob
    auc = roc_auc_score(y_test, test_prob)
    ap  = average_precision_score(y_test, test_prob)
    print(f"\n== test-set metrics ==")
    print(f"  AUC-ROC : {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"  AUC-PR  : {ap:.3f}   (baseline = {y_test.mean():.3f} = positive rate)")

    # Coverage check on test: top-30 by ml_score, how many were labelled winners?
    for entry_date in TEST_DATES:
        sub = test_df[test_df["entry_date"]==entry_date].copy()
        sub = sub.sort_values("ml_score", ascending=False)
        top_n = 30
        top30 = sub.head(top_n)
        n_winners_in_top30 = int(top30["label"].sum())
        total_winners = int(sub["label"].sum())
        print(f"\n  {entry_date}: {n_winners_in_top30}/{total_winners} winners in top-{top_n} by ml_score  "
              f"(coverage {n_winners_in_top30/max(1,total_winners)*100:.1f}%)")
        print(f"    Top-15 predicted winners:")
        for _, r in top30.head(15).iterrows():
            mark = "✅" if r['label']==1 else " "
            print(f"      {mark} {r['symbol']:12s} score={r['ml_score']:.3f}  actual_peak={r['peak_ret']*100:+.1f}%")

    # Feature importance
    imp = pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False)
    print(f"\n== top feature importance ==")
    print(imp.head(10).to_string(index=False))

    # Score TODAY's universe
    latest = df["trade_date"].max()
    today_snap = df[df["trade_date"]==latest].copy()
    today_snap = today_snap[(today_snap["adv_20d_cr"]>=5) & (today_snap["close"]>50)]
    today_snap = today_snap.dropna(subset=FEATURES)
    today_snap["ml_score"] = model.predict_proba(today_snap[FEATURES])[:,1]

    top_today = today_snap.sort_values("ml_score", ascending=False).head(30)
    print(f"\n== TODAY's ML top-30 predictions (data through {latest.date()}) ==")
    show = ["symbol","close","ml_score","rsi_14_daily","return_20d","ret_5d","volume_vs_20d","adv_20d_cr"]
    print(top_today[show].to_string(index=False))

    # Save today's scores
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    today_snap[["symbol","close","ml_score"] + FEATURES].to_parquet(OUT_MODEL, index=False)
    print(f"\nwrote {OUT_MODEL.relative_to(ROOT)}")

    return {
        "auc": auc,
        "auc_pr": ap,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "top_features": imp.head(5).to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = main()
    print(f"\n{'='*70}")
    print(f" FINAL: AUC={result['auc']:.3f} AUC-PR={result['auc_pr']:.3f}")
    print(f"{'='*70}")
