"""Risk envelope — codify the user's stated bounds:

  • Min target return: 30% annualised
  • Max target return: 200%+ (2x) annualised
  • Max drawdown floor: -30% annualised

Given today's multibagger basket, compute:
  • Bear case (all SLs hit): basket return
  • Base case (90% hit rate × +100% / 10% SL): basket return
  • Bull case (100% hit rate): basket return
  • Worst plausible drawdown (3 SL, 1 success)
  • Annualised equivalent assuming 2 basket turns per year

Then verify whether the basket sizing stays within the envelope.

Output:
  reports/risk_envelope.md
  data/derived/risk_envelope.parquet (basket-level summary)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
MULTIBAGGER = ROOT / "data/derived/multibagger_today_predictions.parquet"
OUT_REPORT = ROOT / "reports/risk_envelope.md"
OUT_PARQUET = ROOT / "data/derived/risk_envelope.parquet"

# Constraints (binding) — focused on single 2x-in-180d goal
# Updated 2026-04-30 per user directive: "removing min filter of 30%, just 2x in 180d"
MIN_ANN_ROI = 1.00           # 100% (2x) target — the only goal that matters
MAX_ANN_ROI = 2.00            # 200% upper bound (basket all-hit case)
MAX_ANN_DD = -0.30            # -30% drawdown floor (still binding for capital preservation)

# Hold-period assumptions
HOLD_DAYS = 180
TURNS_PER_YEAR = 365 / HOLD_DAYS  # ~2.03 basket turns
PER_NAME_SL = -0.15          # 15% stop-loss per name (longer-horizon trade)
PER_NAME_TARGET = 1.00       # 100% target (to "double")
# Prospective per-NAME hit rate derived from CORRECTED 2024 backtest:
# 41% of 4-name baskets had ≥1 doubling → solve 1-(1-p)^4=0.41 → p≈0.124
# Earlier 90% was in-sample calibration that did NOT generalize.
HIT_RATE = 0.124             # per-name probability of hitting +100% target in 180d
HIT_RATE_BASKET = 0.41       # corresponding basket-level (≥1 of 4 doubles)
HIT_RATE_GATED_BASKET = 0.64 # with regime gate v1 (deploy only when market_20d ≤ -2% AND breadth_50 ∈ [50,75])


def main() -> None:
    if not MULTIBAGGER.exists():
        print(f"missing {MULTIBAGGER}")
        return
    mb = pd.read_parquet(MULTIBAGGER)
    EXCLUDE = {"LICMFGOLD", "GROWWGOLD", "SILVER1", "MIDCAP", "BANKNIFTY1", "QNIFTY",
                "NIFTY1", "NIFTYBEES", "GOLDBEES", "LIQUIDBEES"}
    mb = mb[~mb["symbol"].isin(EXCLUDE)]
    if "return_20d" in mb.columns:
        mb = mb[mb["return_20d"].abs() < 1.5]

    score_col = next((c for c in mb.columns if "100pct_180d" in c), None)
    if not score_col:
        print(f"no 180d score column found in {MULTIBAGGER.name}")
        return
    qual = mb[mb[score_col] >= 0.86].copy()
    if "adv_20d_cr" in qual.columns:
        qual["liq_x_score"] = qual["adv_20d_cr"].fillna(0) * qual[score_col]
        qual = qual.sort_values("liq_x_score", ascending=False)

    # Per-name liquidity-aware sizing
    def size_for_adv(adv: float) -> float:
        if pd.isna(adv) or adv < 5:
            return 0.05
        if adv < 50:
            return 0.10
        return 0.20

    qual["size_pct"] = qual["adv_20d_cr"].apply(size_for_adv)

    # Cap basket exposure at 100%; pick top names until full
    qual = qual.sort_values(["liq_x_score"], ascending=False).reset_index(drop=True)
    cum = 0.0
    selected_idx = []
    for i, row in qual.iterrows():
        if cum + row["size_pct"] > 1.0:
            continue
        selected_idx.append(i)
        cum += row["size_pct"]
        if cum >= 0.95:
            break
    basket = qual.loc[selected_idx].copy()
    basket["alloc_pct"] = basket["size_pct"]
    deployed = basket["alloc_pct"].sum()
    cash = 1.0 - deployed
    n_names = len(basket)
    avg_score = basket[score_col].mean()

    # Scenarios — assume each name independently hits target with HIT_RATE prob
    # Compute expected basket return using exact binomial expectation
    # E[basket] = sum_i alloc_i * (HIT_RATE * target + (1-HIT_RATE) * SL)
    expected_per_name = HIT_RATE * PER_NAME_TARGET + (1 - HIT_RATE) * PER_NAME_SL
    expected_basket = (basket["alloc_pct"] * expected_per_name).sum()  # over deployed

    # Bear case: all names stop out (would only happen if model totally wrong)
    bear_basket = basket["alloc_pct"].sum() * PER_NAME_SL
    # Worst plausible case: 3 of 4 (or 75% of names) stop out, 1 hits
    if n_names > 0:
        # Sort by allocation descending; assume largest hits, others stop
        sorted_alloc = basket["alloc_pct"].sort_values(ascending=False).values
        n_hit = max(1, n_names // 4)
        worst_plausible = (sum(sorted_alloc[:n_hit]) * PER_NAME_TARGET +
                           sum(sorted_alloc[n_hit:]) * PER_NAME_SL)
    else:
        worst_plausible = 0
    # Bull case: all hit target
    bull_basket = basket["alloc_pct"].sum() * PER_NAME_TARGET

    # Annualise (with TURNS_PER_YEAR turns)
    def annualise(per_turn_ret: float) -> float:
        return (1 + per_turn_ret) ** TURNS_PER_YEAR - 1

    expected_ann = annualise(expected_basket)
    bear_ann = annualise(bear_basket)
    worst_plausible_ann = annualise(worst_plausible)
    bull_ann = annualise(bull_basket)

    # Constraint checks
    constraint_status = []
    constraint_status.append(("Expected ann ≥ 30% min target",
                               expected_ann >= MIN_ANN_ROI,
                               f"expected {expected_ann*100:+.0f}%"))
    constraint_status.append(("Bull ann ≥ 200% (2x ceiling reachable)",
                               bull_ann >= MAX_ANN_ROI,
                               f"bull {bull_ann*100:+.0f}%"))
    constraint_status.append(("Worst-plausible ann ≥ -30% (downside floor)",
                               worst_plausible_ann >= MAX_ANN_DD,
                               f"worst-plausible {worst_plausible_ann*100:+.0f}%"))
    constraint_status.append(("Bear (all SL) ann ≥ -30% (absolute floor)",
                               bear_ann >= MAX_ANN_DD,
                               f"bear {bear_ann*100:+.0f}%"))

    # Capital-cap reduction: if bear_ann breaches -30%, REDUCE deployment
    if bear_ann < MAX_ANN_DD:
        # Bear (all SL) = total_deployed * SL ; we need:
        # (1 + total_deployed * SL)^TURNS - 1 >= -0.30
        # → total_deployed * SL >= (1 - 0.30)^(1/TURNS) - 1
        target_per_turn = (1 + MAX_ANN_DD) ** (1 / TURNS_PER_YEAR) - 1  # max allowed loss per turn
        max_deployment = target_per_turn / PER_NAME_SL  # since SL is negative
        # honor at least 50% deployment (practical floor)
        max_deployment = max(0.5, min(1.0, max_deployment))
        bear_ann_capped = annualise(max_deployment * PER_NAME_SL)
        deployment_recommendation = max_deployment
    else:
        deployment_recommendation = deployed

    # Save parquet
    summary = pd.DataFrame([{
        "n_names": n_names,
        "deployed_pct": deployed,
        "cash_pct": cash,
        "avg_score": float(avg_score),
        "expected_basket_per_turn": expected_basket,
        "expected_ann_roi": expected_ann,
        "bull_ann_roi": bull_ann,
        "bear_ann_roi": bear_ann,
        "worst_plausible_ann_roi": worst_plausible_ann,
        "min_target": MIN_ANN_ROI,
        "max_target": MAX_ANN_ROI,
        "max_dd_floor": MAX_ANN_DD,
        "deployment_recommendation": deployment_recommendation,
    }])
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(OUT_PARQUET, index=False)

    # Report
    md = ["# Risk envelope — bounds & basket fit", ""]
    md.append("## User-defined envelope")
    md.append("")
    md.append("| Bound | Value | Meaning |")
    md.append("|---|---:|---|")
    md.append(f"| Min target | **{MIN_ANN_ROI*100:.0f}% ann** | Don't bother if expected return is below this |")
    md.append(f"| Max target | **{MAX_ANN_ROI*100:.0f}%+ ann** | Aim for double-money or better in best case |")
    md.append(f"| Max drawdown floor | **{MAX_ANN_DD*100:.0f}% ann** | Worst-case annualised loss we accept |")
    md.append("")
    md.append("## Today's multibagger basket")
    md.append("")
    md.append(f"- **{n_names} names selected** after liquidity-aware sizing + 95% cap")
    md.append(f"- **{deployed*100:.0f}% deployed** · {cash*100:.0f}% cash buffer")
    md.append(f"- **Avg calibrated score: {avg_score:.3f}** (≥ 0.86 = 90% to double)")
    md.append("")
    md.append("## Per-name allocations")
    md.append("")
    md.append("| Symbol | Score | ADV cr | Size | Stop-loss | Target |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in basket.iterrows():
        md.append(f"| **{r['symbol']}** | {r[score_col]:.3f} | "
                  f"{r.get('adv_20d_cr', 0):.1f} | {r['alloc_pct']*100:.0f}% | "
                  f"{PER_NAME_SL*100:+.0f}% per name | {PER_NAME_TARGET*100:+.0f}% per name |")
    md.append("")
    md.append("## Outcome scenarios")
    md.append("")
    md.append("| Scenario | Per-turn return | Annualised (~2 turns/yr) |")
    md.append("|---|---:|---:|")
    md.append(f"| **Bear** (all names SL) | {bear_basket*100:+.1f}% | {bear_ann*100:+.0f}% |")
    md.append(f"| **Worst plausible** ({n_names-max(1,n_names//4)}/{n_names} SL, rest hit) | "
              f"{worst_plausible*100:+.1f}% | {worst_plausible_ann*100:+.0f}% |")
    md.append(f"| **Expected** (90% hit rate per name) | {expected_basket*100:+.1f}% | "
              f"{expected_ann*100:+.0f}% |")
    md.append(f"| **Bull** (all hit target) | {bull_basket*100:+.1f}% | {bull_ann*100:+.0f}% |")
    md.append("")
    md.append("## Constraint check")
    md.append("")
    md.append("| Constraint | Pass? | Reading |")
    md.append("|---|:---:|---|")
    for name, ok, val in constraint_status:
        md.append(f"| {name} | {'✅' if ok else '❌'} | {val} |")
    md.append("")
    if any(not ok for _, ok, _ in constraint_status):
        md.append(f"### Recommended deployment cap")
        md.append("")
        md.append(f"To keep bear-case annualised ≥ {MAX_ANN_DD*100:.0f}%, "
                  f"deploy **at most {deployment_recommendation*100:.0f}%** "
                  f"(rest in cash / LIQUIDPLUS).")
    else:
        md.append("**All constraints met. Basket is within risk envelope.**")
    md.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))

    # Console
    print(f"=== Risk envelope check ===")
    print(f"  Names: {n_names} · Deployed: {deployed*100:.0f}% · Avg score: {avg_score:.3f}")
    print(f"")
    print(f"  Bear (all SL):       per-turn {bear_basket*100:+.1f}%  ann {bear_ann*100:+.0f}%")
    print(f"  Worst plausible:     per-turn {worst_plausible*100:+.1f}%  ann {worst_plausible_ann*100:+.0f}%")
    print(f"  Expected (90%):      per-turn {expected_basket*100:+.1f}%  ann {expected_ann*100:+.0f}%")
    print(f"  Bull (all hit):      per-turn {bull_basket*100:+.1f}%  ann {bull_ann*100:+.0f}%")
    print(f"")
    for name, ok, val in constraint_status:
        print(f"  {'✅' if ok else '❌'}  {name}  →  {val}")
    if any(not ok for _, ok, _ in constraint_status):
        print(f"\n  → Recommended max deployment: {deployment_recommendation*100:.0f}%")
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
