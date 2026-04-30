"""Devil's Advocate — automated audit battery.

Runs across every claim in the system. Emits red flags, not affirmations.

Vectors tested:
  1. LEAKAGE        — features computed using future data per-symbol/per-day
  2. CAL_DRIFT      — calibrator fit on the same data being scored
  3. MULTI_TEST     — N variants tried; nominal p-value needs Bonferroni
  4. SURVIVORSHIP   — universe filter consistency over time
  5. REGIME_FIT     — regime gate trained and tested on same period
  6. SIG_INDEPEND   — confluence/stack signals correlate, don't multiply
  7. SAMPLE_SIZE    — < 100 OOS at the headline band = unreliable
  8. DIST_SHIFT     — train and test feature distributions differ materially
  9. HYPER_LEAK     — hyperparameter selected using test set
 10. CHERRY_PICK    — best-of-N reporting without correction

Outputs:
  reports/devils_advocate_audit.md — red flags + required validation
  data/derived/devils_advocate_audit.parquet — machine-readable findings
"""
from __future__ import annotations
import json
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
EXTRA = ROOT / "data/derived/extra_features.parquet"
HC_PRED = ROOT / "data/derived/high_conviction_predictions.parquet"
MULTI_PRED = ROOT / "data/derived/multibagger_today_predictions.parquet"
FRONTIER = ROOT / "data/derived/achievable_frontier.parquet"
RISK_ENV = ROOT / "data/derived/risk_envelope.parquet"
SUPERSTAR_HOLDINGS = ROOT / "data/derived/superstar_holdings.parquet"

OUT_REPORT = ROOT / "reports/devils_advocate_audit.md"
OUT_PARQUET = ROOT / "data/derived/devils_advocate_audit.parquet"


def severity_emoji(severity: str) -> str:
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "PASS": "✅"}[severity]


def check_leakage_in_features() -> list[dict]:
    """Vector 1: are any features in extra_features.parquet computed using
    future data per (symbol, trade_date)?"""
    findings = []
    if not EXTRA.exists():
        return [{"vector": "LEAKAGE", "severity": "MEDIUM",
                 "claim": "extra_features parquet exists",
                 "evidence": "missing file",
                 "impact": "cannot audit"}]

    ex = pd.read_parquet(EXTRA)
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])

    # Specifically check static-snapshot features that were broadcast across all dates.
    # These are KNOWN leakage candidates: superstar holdings, screener fundamentals,
    # multibagger_predictions all use TODAY's snapshot applied historically.
    suspect_prefixes = ["scr_", "qvm_", "acad_"]
    suspects = []
    for c in ex.columns:
        if any(c.startswith(p) for p in suspect_prefixes):
            suspects.append(c)

    if suspects:
        # check: do these columns have constant value per symbol across all dates? (telltale leakage sign)
        constant_per_sym = []
        for c in suspects[:8]:  # sample 8
            ex_c = ex[["symbol", "trade_date", c]].dropna()
            if len(ex_c) == 0:
                continue
            uniq_per_sym = ex_c.groupby("symbol")[c].nunique().mean()
            if uniq_per_sym < 1.5:  # essentially same value per symbol across all dates
                constant_per_sym.append(c)
        if constant_per_sym:
            findings.append({
                "vector": "LEAKAGE",
                "severity": "CRITICAL",
                "claim": "Screener / academic / qvm features used as time-series",
                "evidence": f"{len(constant_per_sym)} of {len(suspects)} sampled features "
                           f"are constant per-symbol across all dates: {', '.join(constant_per_sym[:5])}…",
                "impact": "Today's-snapshot features applied to historical labels. "
                          "Backtest on these features cannot generalize forward — IC and "
                          "lift numbers are partially tautological.",
                "fix": "Fetch quarterly historical Screener fundamentals; rebuild features "
                       "with proper trade_date stamping.",
            })
        else:
            findings.append({
                "vector": "LEAKAGE", "severity": "LOW",
                "claim": "Screener features vary per-date per-symbol",
                "evidence": f"{len(suspects)} suspect features show variation",
                "impact": "Likely safe; spot-check needed",
            })

    # check label construction: forward_high_max requires future bars — that's CORRECT
    # but features that LOOK forward (e.g., dist_from_52w_high computed using future highs) are a leak
    # this would be subtle; flag for manual review
    findings.append({
        "vector": "LEAKAGE", "severity": "MEDIUM",
        "claim": "All features are point-in-time computable",
        "evidence": "Manual review needed: dist_sma200, market_breadth, sector_5d_ret all "
                    "look backward, but verify no per-symbol shifts use future data",
        "impact": "Spot-check by selecting one historical row and recomputing the feature "
                  "using only data up to that date.",
    })
    return findings


def check_calibration_drift() -> list[dict]:
    """Vector 2: was the isotonic calibrator fit on the same data being scored?"""
    findings = []
    if not HC_PRED.exists():
        return [{"vector": "CAL_DRIFT", "severity": "MEDIUM",
                 "claim": "high-conviction predictions exist",
                 "evidence": "missing", "impact": "cannot audit"}]

    # The find_high_conviction.py code fits isotonic on the OOF concat of 2024+2025.
    # The published 0.80 band hit rate of 83.5% was measured on that same OOF.
    # Honest prospective test: fit isotonic ONLY on 2024 OOF, score 2025 OOF.
    findings.append({
        "vector": "CAL_DRIFT", "severity": "HIGH",
        "claim": "0.80 band delivers 83.5% real hit rate (5%/7d target)",
        "evidence": "Calibrator was fit on the concat of 2024+2025 OOF. "
                    "When applied back to those same predictions, the hit-rate "
                    "match is partially mechanical (isotonic minimises this gap by "
                    "construction).",
        "impact": "True prospective hit rate is likely 5-10pp lower than claimed. "
                  "User-corrected risk_envelope.py shows in-sample 90% maps to "
                  "12.4% prospective per-name on multibagger — analogous gap "
                  "may exist on the 5%/7d band.",
        "fix": "Rerun: fit isotonic on 2024 OOF only, score 2025 OOF, report band "
               "hit rates separately. Headline must use the 2025-only number.",
    })

    # multibagger correction: user already exposed this
    findings.append({
        "vector": "CAL_DRIFT", "severity": "PASS",
        "claim": "Multibagger 90% claim was overfit",
        "evidence": "User's risk_envelope.py corrected HIT_RATE from 0.90 → 0.124 "
                    "based on prospective 2024 backtest (41% basket-level instead of "
                    "expected 99%+).",
        "impact": "Correction acknowledged. Devil's advocate validated.",
    })
    return findings


def check_multiple_testing() -> list[dict]:
    """Vector 3: how many (X, Y) combos did we test in the frontier?
    What's the Bonferroni threshold for the achieved IC?"""
    findings = []
    if not FRONTIER.exists():
        return [{"vector": "MULTI_TEST", "severity": "LOW",
                 "claim": "frontier ran",
                 "evidence": "missing parquet", "impact": "cannot audit"}]
    fr = pd.read_parquet(FRONTIER)
    n_tests = len(fr)
    nominal_alpha = 0.05
    bonf_alpha = nominal_alpha / n_tests
    findings.append({
        "vector": "MULTI_TEST", "severity": "HIGH",
        "claim": f"Frontier reports {n_tests} (X%, Y days) combos. Achievable count: "
                 f"{int(fr['achievable_90'].sum())}.",
        "evidence": f"Tested {n_tests} hypotheses with α=0.05 nominal. "
                    f"Bonferroni-corrected α = {bonf_alpha:.4f}. "
                    f"Many of the 'IC_PASSED' factors won't survive this.",
        "impact": "Some subset of 'achievable' combos are noise. Apply 1/N correction.",
        "fix": "Re-evaluate each achievable combo at α=Bonferroni. Promote only those "
               "still significant.",
    })
    return findings


def check_survivorship() -> list[dict]:
    """Vector 4: does the universe today include only stocks that survived?"""
    findings = []
    if not PRICES.exists():
        return [{"vector": "SURVIVORSHIP", "severity": "LOW",
                 "claim": "price parquet exists", "evidence": "missing", "impact": "n/a"}]
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    n_2016 = df[(df["trade_date"] >= "2016-01-01") & (df["trade_date"] < "2017-01-01")]["symbol"].nunique()
    n_2025 = df[(df["trade_date"] >= "2025-01-01") & (df["trade_date"] < "2026-01-01")]["symbol"].nunique()
    n_both = (df[df["trade_date"].dt.year == 2016]["symbol"].drop_duplicates().tolist())
    n_both = set(n_both) & set(df[df["trade_date"].dt.year == 2025]["symbol"].drop_duplicates().tolist())
    n_dropped = n_2016 - len(n_both)
    findings.append({
        "vector": "SURVIVORSHIP", "severity": "MEDIUM" if n_dropped > 50 else "LOW",
        "claim": "Backtest universe includes delisted / merged / suspended names",
        "evidence": f"2016 had {n_2016:,} symbols; 2025 has {n_2025:,}; "
                    f"{n_dropped:,} symbols ({n_dropped/n_2016*100:.0f}%) present in 2016 but absent in 2025.",
        "impact": "If we filter to today's universe + apply to 2016 returns, we miss "
                  "the failed names — survivorship bias inflates returns.",
        "fix": "Confirm backtest uses universe-as-of-each-date, not today's universe.",
    })
    return findings


def check_regime_fit() -> list[dict]:
    """Vector 5: was the regime gate (market_20d ≤ -2% AND breadth 50-75%) parameters
    tuned on the same period it was tested on?"""
    findings = []
    findings.append({
        "vector": "REGIME_FIT", "severity": "CRITICAL",
        "claim": "Regime gate v1 lifts hit rate 41% → 64%",
        "evidence": "The gate parameters (market_20d ≤ -2%, breadth ∈ [50,75]) are "
                    "documented in risk_envelope.py without specifying which years they "
                    "were derived from. If derived from same 2024 OOS we tested on, this "
                    "is curve-fitting.",
        "impact": "Gate may not generalize. If true prospective lift is 41% → 50% "
                  "(not 64%), per-name hit rate stays ~16%, expected ann ≈ -0.5%.",
        "fix": "Train regime gate on 2018-2022 only; test on 2023-2025 prospective. "
               "Report year-by-year hit rate. If 2018 (bear year) shows 41% basket "
               "hit rate same as unconditional, the gate is weather-vane, not edge.",
    })
    return findings


def check_signal_independence() -> list[dict]:
    """Vector 6: are the stacked signals independent? We already proved no via
    joint_signal_analyzer.py — flag for the record."""
    return [{
        "vector": "SIG_INDEPEND", "severity": "PASS",
        "claim": "Stacking 3+ signals → 80%+ confidence",
        "evidence": "joint_signal_analyzer.py empirically tested: 3-signal stocks have "
                    "40.4% hit rate vs 42.8% baseline. NOT independent. Falsified.",
        "impact": "Stacking thesis is dead. Don't use stack count as a confidence proxy.",
    }]


def check_sample_size() -> list[dict]:
    """Vector 7: are headline claims backed by ≥100 OOS samples per band per year?"""
    findings = []
    if not FRONTIER.exists():
        return findings
    fr = pd.read_parquet(FRONTIER)
    achievable = fr[fr["achievable_90"] == True]
    small_samples = achievable[achievable["n_at_max_band"] < 100]
    if len(small_samples):
        findings.append({
            "vector": "SAMPLE_SIZE", "severity": "MEDIUM",
            "claim": f"Frontier marks {len(achievable)} combos as 'achievable @ 90%'",
            "evidence": f"{len(small_samples)} of these have < 100 OOS samples at the "
                        f"reported band. Examples: " +
                        ", ".join([f"{int(r['horizon'])}d×{r['threshold_pct']:.0f}% "
                                    f"(n={r['n_at_max_band']})"
                                    for _, r in small_samples.head(5).iterrows()]),
            "impact": "95% confidence intervals around the hit rate are very wide. "
                      "True hit rate could be 75-100%; we can't tell.",
            "fix": "Either accumulate more OOS data, or downgrade the 90% claim.",
        })
    return findings


def check_distribution_shift() -> list[dict]:
    """Vector 8: do feature distributions differ materially between train and test years?"""
    findings = []
    if not PRICES.exists():
        return findings
    df = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "return_1d", "rsi_14_daily",
                                            "avg_traded_value_20d"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["year"] = df["trade_date"].dt.year
    # mean RSI by year
    by_year = df.groupby("year").agg(
        rsi_mean=("rsi_14_daily", "mean"),
        ret_std=("return_1d", "std"),
        adv_median=("avg_traded_value_20d", "median"),
    ).round(4)
    train_rsi = by_year.loc[2016:2023, "rsi_mean"].mean() if 2023 in by_year.index else None
    test_rsi = by_year.loc[2024:2025, "rsi_mean"].mean() if 2025 in by_year.index else None
    if train_rsi and test_rsi and abs(train_rsi - test_rsi) / train_rsi > 0.05:
        findings.append({
            "vector": "DIST_SHIFT", "severity": "MEDIUM",
            "claim": "Train (2016-23) and test (2024-25) feature distributions are stationary",
            "evidence": f"Mean RSI shifted from {train_rsi:.1f} (train) to {test_rsi:.1f} (test).",
            "impact": "Model trained on different RSI distribution may behave differently in test.",
            "fix": "Add distribution-shift monitoring in production; alert when test stats "
                   "drift > 1σ from train.",
        })
    return findings


def check_hyperparameter_leak() -> list[dict]:
    """Vector 9: were LGB/XGB hyperparameters chosen by looking at test results?"""
    return [{
        "vector": "HYPER_LEAK", "severity": "MEDIUM",
        "claim": "LGB/XGB hyperparameters are documented but appear hand-tuned",
        "evidence": "n_estimators=400, learning_rate=0.05, num_leaves=64, etc are repeated "
                    "across run_v3, run_short_side, run_multi_horizon. No evidence they "
                    "were derived via held-out validation; likely chosen by inspecting "
                    "OOS performance.",
        "impact": "Mild. Tree GBM is fairly robust to hyperparameter choice in this range. "
                  "But the 'optimal' hyperparameters may be 2024-specific.",
        "fix": "Run hyperparameter search on 2018-2022 only; lock parameters; test on 2023-25.",
    }]


def check_cherry_pick() -> list[dict]:
    """Vector 10: was the headline number (5%/7d at 0.95 = 97.6%) the best of many?"""
    return [{
        "vector": "CHERRY_PICK", "severity": "HIGH",
        "claim": "Frontier publishes 'best combo per horizon' (e.g. 5%/7d, 7%/15d)",
        "evidence": "70 combos tested; reporting only the achievable ones inflates impressions. "
                    "The 'achievable frontier' table contains both the winners AND the "
                    "non-significant ones, but downstream summaries (find_high_conviction.py, "
                    "risk_envelope.py) cite winners only.",
        "impact": "Over-confident headlines. Real expectation across all combos is "
                  "average, not best.",
        "fix": "Headlines must say 'X of N tests passed Bonferroni-corrected significance'. "
               "Report median achievable as the strategy benchmark, not max.",
    }]


def main() -> None:
    print("== devils_advocate audit ==")
    all_findings = []
    all_findings += check_leakage_in_features()
    all_findings += check_calibration_drift()
    all_findings += check_multiple_testing()
    all_findings += check_survivorship()
    all_findings += check_regime_fit()
    all_findings += check_signal_independence()
    all_findings += check_sample_size()
    all_findings += check_distribution_shift()
    all_findings += check_hyperparameter_leak()
    all_findings += check_cherry_pick()

    res = pd.DataFrame(all_findings)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    # severity buckets
    sev_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "PASS"]
    counts = {s: int((res["severity"] == s).sum()) for s in sev_order}

    md = ["# Devil's Advocate audit — " + datetime.now().strftime("%Y-%m-%d %H:%M IST"), ""]
    md.append("**Job: falsify, not validate.** This audit assumes every modelling claim is "
              "wrong until evidence forces otherwise.")
    md.append("")
    md.append("## Severity summary")
    md.append("")
    md.append("| Severity | Count |")
    md.append("|---|---:|")
    for s in sev_order:
        md.append(f"| {severity_emoji(s)} {s} | {counts[s]} |")
    md.append("")

    crit = res[res["severity"] == "CRITICAL"]
    if len(crit):
        md.append("## 🔴 CRITICAL findings (block deployment)")
        md.append("")
        for _, r in crit.iterrows():
            md.append(f"### {r['vector']} · {r['claim']}")
            md.append(f"- **Evidence:** {r['evidence']}")
            md.append(f"- **Impact:** {r['impact']}")
            if r.get("fix"):
                md.append(f"- **Fix required:** {r['fix']}")
            md.append("")

    high = res[res["severity"] == "HIGH"]
    if len(high):
        md.append("## 🟠 HIGH findings (must address before next ship)")
        md.append("")
        for _, r in high.iterrows():
            md.append(f"### {r['vector']} · {r['claim']}")
            md.append(f"- Evidence: {r['evidence']}")
            md.append(f"- Impact: {r['impact']}")
            if r.get("fix"):
                md.append(f"- Fix: {r['fix']}")
            md.append("")

    med = res[res["severity"] == "MEDIUM"]
    if len(med):
        md.append("## 🟡 MEDIUM findings (worth investigating)")
        md.append("")
        for _, r in med.iterrows():
            md.append(f"- **{r['vector']}**: {r['claim']} — {r['evidence']}")
        md.append("")

    pass_findings = res[res["severity"] == "PASS"]
    if len(pass_findings):
        md.append("## ✅ PASS (already validated / corrected)")
        md.append("")
        for _, r in pass_findings.iterrows():
            md.append(f"- {r['vector']}: {r['claim']} — {r['evidence']}")
        md.append("")

    md.append("## What this means for today")
    md.append("")
    n_critical = counts["CRITICAL"]
    n_high = counts["HIGH"]
    if n_critical > 0:
        md.append(f"**{n_critical} CRITICAL issue(s) found.** Do not ship strategy claims to user "
                  f"until these are resolved.")
    elif n_high > 0:
        md.append(f"**{n_high} HIGH issue(s) found.** Headlines should be downgraded; reported "
                  f"hit rates need prospective re-validation.")
    else:
        md.append("No critical issues. Strategy claims are defensible.")
    md.append("")

    OUT_REPORT.write_text("\n".join(md))
    print(f"\n=== summary ===")
    for s in sev_order:
        print(f"  {severity_emoji(s)} {s}: {counts[s]}")
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
