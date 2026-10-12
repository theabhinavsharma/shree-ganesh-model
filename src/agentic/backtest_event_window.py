"""Event-window backtest — different architecture from event-feature stacking.

Question: instead of asking "do event features as model inputs lift CAGR" (the
question we already answered NO to in 3 separate A/B tests), this asks:

  "If we run our existing winning strategy (Top-10 @ 0.65 calibrated, 7d hold)
   but FILTER signal candidates to ONLY those in an event-window, does CAGR
   change?"

This is a different question. The model's score doesn't change. The signal
threshold doesn't change. We just LIMIT which stocks the strategy can pick —
and see if event-conditional selection has a different conditional EV than
the unfiltered baseline.

Eight arms tested:
  ALL                — baseline, no event filter (= original gate-tuning win)
  EVENT_1D           — trade only if any event in last 1 trading day
  EVENT_5D           — trade only if any event in last 5 trading days
  EVENT_10D          — trade only if any event in last 10 trading days
  EVENT_30D          — trade only if any event in last 30 trading days
  POSITIVE_EVENT_5D  — trade only if a POSITIVE event in last 5d
                       (px_polarity > 0 OR text_polarity > 0)
  NEGATIVE_EVENT_5D  — trade only if a NEGATIVE event in last 5d
                       (could be fade-bad-news pattern)
  NO_EVENT_5D        — trade only if NO event in last 5d (the residual /
                       "boring" trades — control for survivorship)

Each arm uses the same simulator: fixed-capital ₹100, MAX_POS=10, GATE=0.65,
HOLD=7 trading days, TARGET=+5%, SL=-3%, cash @ 7% ann on no-fire days.

Output:
  reports/event_window_backtest.md
  data/derived/event_window_summary.parquet
  data/derived/event_window_equity_<arm>.parquet  per-arm equity curves
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Reuse the simulate() machinery from the event-driven backtester
sys.path.insert(0, "/Users/abhinavs./Code/Zoom")
from src.agentic.backtest_event_driven import simulate, load_prices_for

ROOT = Path("/Users/abhinavs./Code/Zoom")
OOF = ROOT / "data/derived/backtest_10yr_oof.parquet"
EVP = ROOT / "data/derived/event_polarity_features.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / "reports/event_window_backtest.md"
OUT_SUMMARY = ROOT / "data/derived/event_window_summary.parquet"


def load_oof() -> pd.DataFrame:
    df = pd.read_parquet(OOF)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.dropna(subset=["score_cal"])
    return df


def load_event_panel() -> pd.DataFrame:
    """Returns (symbol, trade_date, has_event_*, has_pos_event_*, has_neg_event_*) flags."""
    ev = pd.read_parquet(EVP)
    ev["trade_date"] = pd.to_datetime(ev["trade_date"])

    # Aggregate any-polarity flag (price OR text polarity)
    # 5d-window
    ev["any_event_5d"] = ((ev["evp_px_pol_pos_n_5d"] + ev["evp_px_pol_neg_n_5d"] +
                            ev["evp_text_pol_pos_n_5d"] + ev["evp_text_pol_neg_n_5d"]) > 0).astype(int)
    # 30d-window  (acts as 10d-30d composite signal)
    ev["any_event_30d"] = ((ev["evp_px_pol_pos_n_30d"] + ev["evp_px_pol_neg_n_30d"] +
                             ev["evp_text_pol_pos_n_30d"] + ev["evp_text_pol_neg_n_30d"]) > 0).astype(int)
    # Positive: net polarity > 0 in last 5d
    ev["pos_event_5d"] = (((ev["evp_px_pol_sum_5d"] + ev["evp_text_pol_sum_5d"]) > 0) &
                            (ev["any_event_5d"] == 1)).astype(int)
    # Negative: net polarity < 0 in last 5d
    ev["neg_event_5d"] = (((ev["evp_px_pol_sum_5d"] + ev["evp_text_pol_sum_5d"]) < 0) &
                            (ev["any_event_5d"] == 1)).astype(int)
    return ev[["symbol", "trade_date", "any_event_5d", "any_event_30d",
                "pos_event_5d", "neg_event_5d"]]


def filter_oof(oof: pd.DataFrame, ev_panel: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter OOF predictions to a specific event-window mode."""
    if mode == "ALL":
        return oof
    j = oof.merge(ev_panel, on=["symbol", "trade_date"], how="left")
    for c in ["any_event_5d", "any_event_30d", "pos_event_5d", "neg_event_5d"]:
        j[c] = j[c].fillna(0).astype(int)

    if mode == "EVENT_5D":
        return j[j["any_event_5d"] == 1].drop(columns=[c for c in j.columns if c.startswith(("any_","pos_","neg_"))])
    if mode == "EVENT_30D":
        return j[j["any_event_30d"] == 1].drop(columns=[c for c in j.columns if c.startswith(("any_","pos_","neg_"))])
    if mode == "POSITIVE_EVENT_5D":
        return j[j["pos_event_5d"] == 1].drop(columns=[c for c in j.columns if c.startswith(("any_","pos_","neg_"))])
    if mode == "NEGATIVE_EVENT_5D":
        return j[j["neg_event_5d"] == 1].drop(columns=[c for c in j.columns if c.startswith(("any_","pos_","neg_"))])
    if mode == "NO_EVENT_5D":
        return j[j["any_event_5d"] == 0].drop(columns=[c for c in j.columns if c.startswith(("any_","pos_","neg_"))])
    raise ValueError(mode)


def main() -> None:
    print("== backtest_event_window (8-arm event-window study) ==\n")
    print("Loading OOF + event panel...")
    oof = load_oof()
    print(f"  OOF: {len(oof):,} rows, dates {oof['trade_date'].min():%Y-%m-%d} → {oof['trade_date'].max():%Y-%m-%d}")
    ev = load_event_panel()
    print(f"  event panel: {len(ev):,} rows")
    # quick stats: how often does an event coincide with score_cal >= 0.65?
    j = oof.merge(ev, on=["symbol", "trade_date"], how="left").fillna(0)
    sig = j[j["score_cal"] >= 0.65]
    print(f"  signal rows (score_cal>=0.65): {len(sig):,}")
    print(f"    of those, with event in last 5d: {(sig['any_event_5d']==1).sum():,} ({(sig['any_event_5d']==1).mean()*100:.1f}%)")
    print(f"    of those, with event in last 30d: {(sig['any_event_30d']==1).sum():,} ({(sig['any_event_30d']==1).mean()*100:.1f}%)")
    print(f"    of those, with POSITIVE event in last 5d: {(sig['pos_event_5d']==1).sum():,} ({(sig['pos_event_5d']==1).mean()*100:.1f}%)")
    print(f"    of those, with NEGATIVE event in last 5d: {(sig['neg_event_5d']==1).sum():,} ({(sig['neg_event_5d']==1).mean()*100:.1f}%)")
    print()

    # Hit rate by event-window membership (diagnostic before full backtest)
    print("Diagnostic: hit rate (winner_5pct_7td) by event-window membership at score_cal>=0.65...")
    h_all  = sig["winner_5pct_7td"].mean()
    h_e5   = sig.loc[sig["any_event_5d"]==1, "winner_5pct_7td"].mean()
    h_e30  = sig.loc[sig["any_event_30d"]==1, "winner_5pct_7td"].mean()
    h_pos  = sig.loc[sig["pos_event_5d"]==1, "winner_5pct_7td"].mean()
    h_neg  = sig.loc[sig["neg_event_5d"]==1, "winner_5pct_7td"].mean()
    h_none = sig.loc[sig["any_event_5d"]==0, "winner_5pct_7td"].mean()
    print(f"    ALL signals               hit={h_all:.3f}  n={len(sig):,}")
    print(f"    +event in last 5d         hit={h_e5:.3f}   n={(sig['any_event_5d']==1).sum():,}")
    print(f"    +event in last 30d        hit={h_e30:.3f}  n={(sig['any_event_30d']==1).sum():,}")
    print(f"    +POSITIVE event last 5d   hit={h_pos:.3f}  n={(sig['pos_event_5d']==1).sum():,}")
    print(f"    +NEGATIVE event last 5d   hit={h_neg:.3f}  n={(sig['neg_event_5d']==1).sum():,}")
    print(f"    NO event in last 5d       hit={h_none:.3f}  n={(sig['any_event_5d']==0).sum():,}")
    print()

    # Same diagnostic for fwd_c2c_7
    c_all  = sig["fwd_c2c_7"].mean()
    c_e5   = sig.loc[sig["any_event_5d"]==1, "fwd_c2c_7"].mean()
    c_pos  = sig.loc[sig["pos_event_5d"]==1, "fwd_c2c_7"].mean()
    c_neg  = sig.loc[sig["neg_event_5d"]==1, "fwd_c2c_7"].mean()
    c_none = sig.loc[sig["any_event_5d"]==0, "fwd_c2c_7"].mean()
    print(f"Avg fwd 7d c2c return at score>=0.65 by event filter:")
    print(f"    ALL                       avg c2c={c_all*100:+.3f}%")
    print(f"    +event in last 5d         avg c2c={c_e5*100:+.3f}%")
    print(f"    +POSITIVE event last 5d   avg c2c={c_pos*100:+.3f}%")
    print(f"    +NEGATIVE event last 5d   avg c2c={c_neg*100:+.3f}%")
    print(f"    NO event in last 5d       avg c2c={c_none*100:+.3f}%")
    print()

    # Now run full event-driven backtest for each arm
    print("Loading prices for backtest...")
    symbols = set(oof.loc[oof["score_cal"] >= 0.60, "symbol"].unique())
    prices = load_prices_for(symbols, oof["trade_date"].min())
    print(f"  prices: {len(prices):,} rows for {len(symbols)} symbols\n")

    # Use the winning gate config (Top-10 @ 0.65, 7d hold) as the baseline
    arms_to_run = [
        ("ALL",                "ALL"),
        ("EVENT_5D",           "EVENT_5D"),
        ("EVENT_30D",          "EVENT_30D"),
        ("POSITIVE_EVENT_5D",  "POSITIVE_EVENT_5D"),
        ("NEGATIVE_EVENT_5D",  "NEGATIVE_EVENT_5D"),
        ("NO_EVENT_5D",        "NO_EVENT_5D"),
    ]

    results = []
    for label, mode in arms_to_run:
        print(f"--- arm: {label} ---")
        oof_filtered = filter_oof(oof, ev, mode)
        n_sig = (oof_filtered["score_cal"] >= 0.65).sum()
        print(f"    filtered OOF rows: {len(oof_filtered):,}  (signals >=0.65: {n_sig:,})")
        if n_sig == 0:
            print(f"    skip — no signals after filter")
            continue
        out = simulate(oof_filtered, prices,
                        GATE=0.65, MAX_POS=10, HOLD_DAYS=7,
                        TARGET_PCT=0.05, SL_PCT=-0.03, label=label)
        s = out["stats"]
        print(f"    n_trades={s['n_trades']}  hit={s['hit_rate']*100:.0f}%  "
              f"avg_per_trade={s['avg_return_per_trade']*100:+.3f}%  "
              f"deployed={s['deployed_pct_mean']*100:.0f}%  "
              f"CAGR={s['cagr']*100:+.1f}%  maxDD={s['max_drawdown']*100:+.1f}%  "
              f"Sharpe={s['sharpe']:.2f}\n")
        s["arm"] = label
        results.append(s)

    # comparative table
    res = pd.DataFrame(results)
    cols = ["arm","n_trades","hit_rate","avg_return_per_trade","deployed_pct_mean","cagr","max_drawdown","sharpe"]
    print("=" * 80)
    print("COMPARATIVE TABLE")
    print("=" * 80)
    print(res[cols].to_string(index=False, float_format='%.4f'))

    # vs ALL baseline
    if (res["arm"]=="ALL").any():
        baseline = res[res["arm"]=="ALL"].iloc[0]
        print("\n=== Δ vs ALL baseline ===")
        for _, r in res[res["arm"]!="ALL"].iterrows():
            print(f"  {r['arm']:<20}  ΔCAGR={r['cagr']-baseline['cagr']:+.4f}  "
                  f"Δhit={r['hit_rate']-baseline['hit_rate']:+.3f}  "
                  f"Δavg/trade={(r['avg_return_per_trade']-baseline['avg_return_per_trade'])*100:+.3f}pp")

    # save
    res.to_parquet(OUT_SUMMARY, index=False)
    md = ["# Event-window backtest", "",
          "**Architecture difference from prior A/B tests:**",
          "Prior 3 A/B tests stacked event features into the cross-sectional model and showed no lift.",
          "This test keeps the SAME model and SAME 0.65 gate, but FILTERS candidates to event-window-only.",
          "Different question: 'is event-conditional EV different from baseline?'", "",
          "**Setup:** Top-10 basket @ score_cal >= 0.65, 7d hold, +5%/-3% target/SL, cash @ 7% on no-fire days.", "",
          "## Diagnostic: hit rate at score_cal>=0.65 by event-window membership", "",
          "| Filter | n signals | hit rate (5%/7d) | avg fwd c2c 7d |",
          "|---|---:|---:|---:|",
          f"| ALL signals | {len(sig):,} | {h_all:.3f} | {c_all*100:+.3f}% |",
          f"| +event in last 5d | {(sig['any_event_5d']==1).sum():,} | {h_e5:.3f} | {c_e5*100:+.3f}% |",
          f"| +event in last 30d | {(sig['any_event_30d']==1).sum():,} | {h_e30:.3f} | — |",
          f"| +POSITIVE event last 5d | {(sig['pos_event_5d']==1).sum():,} | {h_pos:.3f} | {c_pos*100:+.3f}% |",
          f"| +NEGATIVE event last 5d | {(sig['neg_event_5d']==1).sum():,} | {h_neg:.3f} | {c_neg*100:+.3f}% |",
          f"| NO event in last 5d | {(sig['any_event_5d']==0).sum():,} | {h_none:.3f} | {c_none*100:+.3f}% |",
          "",
          "## Full backtest by arm",
          "",
          "| Arm | n_trades | Hit % | Avg/trade | Deployed % | **CAGR** | MaxDD | Sharpe |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        md.append(f"| {r['arm']} | {int(r['n_trades'])} | {r['hit_rate']*100:.0f}% | "
                  f"{r['avg_return_per_trade']*100:+.3f}% | {r['deployed_pct_mean']*100:.0f}% | "
                  f"**{r['cagr']*100:+.1f}%** | {r['max_drawdown']*100:+.1f}% | {r['sharpe']:.2f} |")
    md.append("")

    if (res["arm"]=="ALL").any():
        baseline = res[res["arm"]=="ALL"].iloc[0]
        md.append("## Δ vs ALL baseline")
        md.append("")
        md.append("| Arm | ΔCAGR | Δhit_rate | Δavg/trade |")
        md.append("|---|---:|---:|---:|")
        for _, r in res[res["arm"]!="ALL"].iterrows():
            md.append(f"| {r['arm']} | {(r['cagr']-baseline['cagr'])*100:+.1f}pp | "
                      f"{(r['hit_rate']-baseline['hit_rate'])*100:+.1f}pp | "
                      f"{(r['avg_return_per_trade']-baseline['avg_return_per_trade'])*100:+.3f}pp |")
    md.append("")
    md.append("## Verdict reading")
    md.append("")
    md.append("- If POSITIVE_EVENT_5D shows higher CAGR/hit_rate than ALL → events confirm signal, deploy event-conditional sub-strategy")
    md.append("- If NEGATIVE_EVENT_5D shows lower CAGR/hit_rate than ALL → events warn (avoid signal during bad news)")
    md.append("- If NO_EVENT_5D shows similar to ALL → events are uninformative for this strategy")
    md.append("- If everything is within noise → events don't matter for 5%/7d gating")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
