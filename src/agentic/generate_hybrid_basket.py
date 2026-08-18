"""Generate the WEEKLY 15D/+5% hybrid basket.

Combines:
  - 5 ML engines (cross-engine consensus for validated signal)
  - Supervised ML classifier trained on missed-winners
  - QC filter (contamination + operator pump + chase + illiquid)
  - PANAMAPET-clone pattern preference

Emits:
  live_predictions/YYYY-MM-DD_15d5pct.json  (immutable forward-record)

Per CONSTITUTION.md §1.7 — reproducibility is the publishability test.
Every input parquet's mtime is recorded so a future reader can verify.
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")

INPUTS = {
    "prices": "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    "compare_short_horizons": "data/derived/compare_short_horizons.parquet",
    "high_conviction": "data/derived/high_conviction_predictions.parquet",
    "multibagger": "data/derived/multibagger_today_predictions.parquet",
    "180d_frontier": "data/derived/180d_today_predictions.parquet",
    "multi_horizon": "tmp/from_scratch_7d_run/multi_horizon_top.csv",
    "ml_classifier": "data/derived/missed_winner_classifier.parquet",
    "corp_actions": "data/corporate_actions_full_history/_incremental/normalized/stock_corporate_actions.parquet",
}


def load_inputs() -> dict:
    """Load all inputs; record mtimes for reproducibility."""
    d = {}
    for k, p in INPUTS.items():
        fp = ROOT / p
        if not fp.exists():
            print(f"  ⚠️  MISSING: {k} ({p})")
            d[k] = None; d[k+"_mtime"] = None
            continue
        if p.endswith(".csv"):
            d[k] = pd.read_csv(fp)
        else:
            d[k] = pd.read_parquet(fp)
        d[k+"_mtime"] = datetime.fromtimestamp(fp.stat().st_mtime).isoformat()
    return d


def build_snapshot(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-symbol snapshot on latest trade_date + derived features."""
    prices = prices.sort_values(["symbol","trade_date"])
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["ret_5d"] = prices.groupby("symbol")["close"].pct_change(5)
    prices["adv_cr"] = prices.groupby("symbol")["total_traded_value"].transform(
        lambda s: s.rolling(20).mean()) / 1e7
    # C2 exit contract (locked 2026-08-18): per-stock daily vol for vol-scaled stops
    prices["dvol_20d"] = prices.groupby("symbol")["return_1d"].transform(
        lambda s: s.rolling(20).std())
    latest = prices["trade_date"].max()
    snap = prices[prices["trade_date"] == latest].copy()

    # UC streak
    recent = prices[prices["trade_date"] >= latest - pd.Timedelta(days=45)].copy()
    recent["is_uc"] = (recent["return_1d"].fillna(0) >= 0.0498) & (recent["return_1d"].fillna(0) <= 0.1005)
    def _streak(s):
        cur = mx = 0
        for v in s:
            if v: cur += 1; mx = max(mx, cur)
            else: cur = 0
        return mx
    uc = recent.groupby("symbol")["is_uc"].apply(_streak).rename("max_uc").reset_index()
    snap = snap.merge(uc, on="symbol", how="left")
    return snap, latest


def find_contaminated(prices: pd.DataFrame, ca: pd.DataFrame) -> set:
    """Symbols with any -30%+ drop that has no corporate-action match within ±5d."""
    prices = prices.sort_values(["symbol","trade_date"])
    big = prices[prices["return_1d"].fillna(0) < -0.30]
    ca = ca.copy(); ca["ex_date"] = pd.to_datetime(ca["ex_date"])
    contam = set()
    for sym, g in big.groupby("symbol"):
        for _, r in g.iterrows():
            if len(ca[(ca["symbol"]==sym) & ((ca["ex_date"]-r["trade_date"]).abs() <= pd.Timedelta(days=5))]) == 0:
                contam.add(sym); break
    return contam


def apply_qc_filter(df: pd.DataFrame, contam: set) -> pd.DataFrame:
    """The 15D/+5% QC filter — data-earned from 10-yr backtest (4,224 trades).
    Findings (target hit rate baseline 23.3%):
      • return_20d [-5%, 0%]  → 27.5% hit  (BEST — consolidation)
      • return_20d [20%, 30%] → 19.5% hit  (WORST — chase)
      • RSI [45, 55]           → 25.5% hit  (BEST)
      • RSI [65, 72]           → 21.8% hit  (WORST — extended)
      • ML score [0.5, 0.7]    → 23.9% hit  (BEST)
      • ML score [0.85+]        → 18.9% hit  (WORST — overconfident)
    """
    return df[
        (df["rsi_14_daily"].between(42, 60)) &        # tightened from 40-72
        (df["return_20d"].between(-0.10, 0.15)) &      # tightened from -15 to +30
        (df["ret_5d"].between(-0.05, 0.05)) &          # tightened from -5 to +10
        (df["volume_vs_20d"] < 2) &                    # tightened from <3
        (df["close"] > 50) &
        (df["adv_cr"] >= 5) &
        (df["max_uc"].fillna(0) < 2) &
        (~df["symbol"].isin(contam))
    ].copy()


def band_fit_score(row) -> float:
    """0-3 score: how many of the 3 optimal bands does this name sit in?
    Higher score = higher expected 15D/+5% hit rate per backtest.
    """
    s = 0.0
    if -0.05 <= row["return_20d"] <= 0.00: s += 1.0
    elif -0.10 <= row["return_20d"] <= 0.02: s += 0.5
    if 45 <= row["rsi_14_daily"] <= 55: s += 1.0
    elif 42 <= row["rsi_14_daily"] <= 58: s += 0.5
    ml = float(row.get("ml_score", 0) or 0)
    if 0.50 <= ml <= 0.70: s += 1.0
    elif 0.45 <= ml <= 0.75: s += 0.5
    elif ml >= 0.85: s -= 0.5  # PENALTY — overconfident band underperforms
    return s


def build_basket(inputs: dict) -> dict:
    prices = inputs["prices"]
    snap, latest = build_snapshot(prices)
    print(f"Data as of: {latest.date()}")
    print(f"Liquid active universe (ADV>=1cr): {(snap['adv_cr']>=1).sum():,}")

    # Contamination
    contam = find_contaminated(prices, inputs["corp_actions"])
    print(f"Contaminated symbols dropped: {len(contam):,}")

    # 5 engine top-30 sets (focused on 15D/+5% — CS is primary)
    cs = inputs["compare_short_horizons"]
    hc = inputs["high_conviction"]
    mb = inputs["multibagger"]
    mh = inputs["multi_horizon"]
    f180 = inputs["180d_frontier"]
    ml = inputs["ml_classifier"]

    cs_top30 = set(cs.sort_values("score_5pct_15d", ascending=False).head(30)["symbol"].tolist())
    hc_cols = [c for c in hc.columns if c.startswith("score_") and c.endswith("_cal")]
    hc = hc.copy(); hc["best"] = hc[hc_cols].max(axis=1)
    hc_top30 = set(hc.sort_values("best", ascending=False).head(30)["symbol"].tolist())
    mb_top30 = set(mb.sort_values("score_50pct_180d", ascending=False).head(30)["symbol"].tolist())
    mh_top30 = set(mh.head(30)["symbol"].tolist())
    f180_top30 = set(f180.sort_values("score_15pct", ascending=False).head(30)["symbol"].tolist())

    # ML classifier top-100 (looser threshold since it's exploratory)
    ml_top100 = set(ml.sort_values("ml_score", ascending=False).head(100)["symbol"].tolist())

    # Merge everything
    m = snap.merge(cs[["symbol","score_5pct_15d","EV_ann_5pct_15d"]], on="symbol", how="left")
    m = m.merge(ml[["symbol","ml_score"]], on="symbol", how="left")
    m["engines"] = m["symbol"].apply(lambda s: sum([s in cs_top30, s in hc_top30, s in mb_top30, s in mh_top30, s in f180_top30]))
    m["in_ml_top100"] = m["symbol"].isin(ml_top100)

    # Apply QC
    clean = apply_qc_filter(m, contam)

    # Score every clean name by backtest-earned band-fit
    clean["band_fit"] = clean.apply(band_fit_score, axis=1)

    # Regime gate (from multibagger)
    regime = mb["regime_gate_verdict"].iloc[0] if "regime_gate_verdict" in mb.columns else "UNKNOWN"

    # Tier 1: 2+ engines AND clean (rank by band_fit then cs_score)
    tier1 = clean[clean["engines"] >= 2].copy().sort_values(
        ["band_fit","score_5pct_15d"], ascending=[False, False])

    # Tier 2: strong ML band-fit (NOT overconfident) AND some validated signal
    tier2_pool = clean[
        (clean["ml_score"].fillna(0).between(0.50, 0.75)) &  # backtest sweet spot
        (clean["band_fit"] >= 1.5) &                          # at least 1.5/3 bands
        (~clean["symbol"].isin(tier1["symbol"]))
    ].sort_values(["band_fit","ml_score"], ascending=[False, False])
    tier2 = tier2_pool.head(10)  # 8 for the basket + up to 2 reserves (gap-away substitution)

    print(f"\nRegime: {regime}")
    print(f"Tier 1 (2+ engines + clean): {len(tier1)} names")
    print(f"Tier 2 (ML-discovered):      {len(tier2)} names")

    # Assemble basket
    def _pick(row, weight, tier):
        return {
            "symbol": row["symbol"],
            "close": round(float(row["close"]), 2),
            "buy_low": round(float(row["close"])*0.99, 2),
            "buy_high": round(float(row["close"])*1.01, 2),
            "target_5pct": round(float(row["close"])*1.05, 2),
            # C2 contract: vol-scaled SL = 3x the stock's own 20d daily vol,
            # floored at -3%, capped at -12% (10-yr tournament: +11.6% CAGR,
            # -18.6% maxDD, best risk-adjusted of A/B/C1/C2)
            "sl_pct": round(-min(max(3*float(row.get("dvol_20d") or 0.01), 0.03), 0.12)*100, 2),
            "sl_3pct": round(float(row["close"]) * (1 - min(max(3*float(row.get("dvol_20d") or 0.01), 0.03), 0.12)), 2),
            "weight_pct": weight,
            "tier": tier,
            "engines_count": int(row["engines"]),
            "cs_score": round(float(row.get("score_5pct_15d", 0) or 0), 3),
            "ml_score": round(float(row.get("ml_score", 0) or 0), 3),
            "band_fit": round(float(row["band_fit"]), 2),
            "rsi": round(float(row["rsi_14_daily"]), 1),
            "return_20d_pct": round(float(row["return_20d"])*100, 1),
            "ret_5d_pct": round(float(row["ret_5d"])*100, 1),
            "adv_cr": round(float(row["adv_cr"]), 2),
        }

    # 100% DEPLOY: 8 names equal-weight × 12.5% each
    # Priority order: Tier-1 (2+ engines) first, backfill with Tier-2 (ML band-fit)
    N_TARGET = 8
    WT_EACH = 100.0 / N_TARGET  # 12.5% each

    picks = []
    for _, r in tier1.head(N_TARGET).iterrows():
        picks.append(_pick(r, WT_EACH, 1))
    remaining = N_TARGET - len(picks)
    if remaining > 0:
        for _, r in tier2.head(remaining).iterrows():
            picks.append(_pick(r, WT_EACH, 2))

    # RESERVES (2026-07-27): when a pick gaps beyond its buy zone at open,
    # its 12.5% deploys into the highest-ranked reserve that IS inside its zone.
    # Never chase a gapped name — its +5% is already spent.
    chosen = {p["symbol"] for p in picks}
    reserves = []
    for _, r in tier2.iterrows():
        if r["symbol"] not in chosen and len(reserves) < 2:
            rp = _pick(r, 0.0, 2)
            rp["role"] = "RESERVE"
            reserves.append(rp)

    # ── CONFIDENCE RANKING (standing rule, 2026-07-24) ─────────────────────
    # Every emitted pick is ranked by confidence in the GOAL (+5% touch ≤15d)
    # and carries a plain-language rationale. Basket refuses to emit otherwise.
    # Evidence weights: engine consensus (only cross-validated signal),
    # band_fit (backtest-earned hit-rate bands), ML honest-zone proximity
    # (walk-forward +16pp p@30 lift), CS engine score.
    def _confidence(p):
        score = (min(p["engines_count"], 3) * 1.5     # consensus: strongest evidence
                 + p["band_fit"] * 1.0                 # earned bands (27.5% vs 19.5% hit)
                 + max(0.0, 0.7 - abs(p["ml_score"] - 0.60)) * 2.0  # peak at honest-zone center
                 + p["cs_score"] * 0.5)
        return round(score, 2)

    def _confidence_rationale(p):
        parts = []
        if p["engines_count"] >= 2:
            parts.append(f"{p['engines_count']}-engine consensus (Tier-1, only cross-validated signal)")
        elif p["engines_count"] == 1:
            parts.append("1 engine confirms")
        else:
            parts.append("ML-only (Tier-2, no consensus)")
        parts.append(f"band_fit {p['band_fit']}/3 (RSI {p['rsi']}, 20d {p['return_20d_pct']:+.1f}% vs earned bands)")
        ml = p["ml_score"]
        zone = "honest zone 0.5-0.7" if 0.5 <= ml <= 0.7 else ("above honest zone" if ml > 0.7 else "below honest zone")
        parts.append(f"ML {ml:.2f} ({zone})")
        parts.append(f"cs {p['cs_score']:.2f}")
        return " · ".join(parts)

    for p in picks:
        p["confidence"] = _confidence(p)
        p["confidence_rationale"] = _confidence_rationale(p)
    picks.sort(key=lambda p: p["confidence"], reverse=True)
    for i, p in enumerate(picks, 1):
        p["rank"] = i

    # Reserves carry the same contract (ranked R1, R2 by confidence)
    for p in reserves:
        p["confidence"] = _confidence(p)
        p["confidence_rationale"] = _confidence_rationale(p)
    reserves.sort(key=lambda p: p["confidence"], reverse=True)
    for i, p in enumerate(reserves, 1):
        p["rank"] = f"R{i}"

    # HARD contract: no basket ships without rank + confidence + rationale per pick
    for p in picks + reserves:
        assert "rank" in p and "confidence" in p and p.get("confidence_rationale"), \
            f"CONTRACT VIOLATION: pick {p['symbol']} missing confidence ranking/rationale"
    # ────────────────────────────────────────────────────────────────────────
    total_wt = sum(p["weight_pct"] for p in picks)

    basket = {
        "as_of_date": str(date.today()),
        "data_through": str(latest.date()),
        "version": "v_hybrid_15d5pct",
        "horizon_days": 15,
        "target_pct": 5,
        "stop_loss_pct": -3,
        "regime_gate": regime,
        "inputs_mtime": {k: v for k, v in inputs.items() if k.endswith("_mtime")},
        "counts": {
            "universe_liquid": int((snap["adv_cr"]>=1).sum()),
            "contaminated_dropped": len(contam),
            "qc_clean": len(clean),
            "engines_2plus": len(tier1),
            "ml_top100_new": len(tier2),
        },
        "picks": picks,
        "reserves": reserves,
        "total_exposure_pct": total_wt,
        "rest_in_liquidplus_pct": round(100 - total_wt, 1),
        "entry_rules": [
            "Place AMO limit orders over the weekend at buy_high — they enter Monday's 9:00-9:07 pre-open auction; you get the auction price if it opens inside your limit",
            "If a pick opens ABOVE buy_high: DO NOT CHASE — its +5% is already spent in the gap",
            "Gapped pick's 12.5% deploys into the highest-ranked RESERVE whose open is inside its own buy zone",
            "If no reserve qualifies either, that 12.5% stays in cash for the week",
        ],
        "exit_contract": "C2 (locked 2026-08-18 after 10-yr A/B/C tournament: +11.6% CAGR, -18.6% maxDD, best risk-adjusted)",
        "exit_rules": [
            "+5% target touch: sell HALF, trail remainder at +2.5%",
            "VOL-SCALED SL per pick (sl_pct field = 3x stock's own 20d daily vol, floor -3%, cap -12%): sell 100% at sl_3pct price, no exceptions",
            "Day 15: exit whatever remains at market",
            "Weekly refresh: rerun pipeline next Monday",
        ],
    }
    return basket


def main():
    print("== generate_hybrid_basket (15D/+5%) ==")

    # HARD FRESHNESS GATE — refuse to emit basket if any critical input is stale.
    # This is here because on 2026-07-01 we emitted a basket on 55-day-stale macro data.
    # Never again.
    from verify_freshness import verify_or_die
    verify_or_die()

    inputs = load_inputs()
    basket = build_basket(inputs)

    out = ROOT / f"live_predictions/{basket['as_of_date']}_15d5pct.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(basket, indent=2, default=str))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"\n=== BASKET SUMMARY ===")
    print(f"Total exposure: {basket['total_exposure_pct']:.1f}%  (rest {basket['rest_in_liquidplus_pct']:.1f}% LIQUIDPLUS)")
    print(f"Regime: {basket['regime_gate']}\n")
    print(f"{'Rank':>4s} {'Conf':>5s} {'Tier':>4s} {'Symbol':12s} {'LTP':>9s} {'Buy':>18s} {'Target':>9s} {'SL':>9s} {'Wt%':>5s} {'Eng':>3s} {'ML':>5s}")
    print("-"*110)
    for p in basket["picks"]:
        print(f"{p['rank']:>4d} {p['confidence']:>5.2f} {p['tier']:>4d} {p['symbol']:12s} {p['close']:>9.2f} {p['buy_low']:>8.2f}-{p['buy_high']:>7.2f} {p['target_5pct']:>9.2f} {p['sl_3pct']:>9.2f} {p['weight_pct']:>4.1f}% {p['engines_count']:>3d} {p['ml_score']:>5.2f}")
        print(f"{'':>10s} └ {p['confidence_rationale']}")


if __name__ == "__main__":
    main()
