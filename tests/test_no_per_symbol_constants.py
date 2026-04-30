"""Anti-leak CI gate.

This test enforces CONSTITUTION.md §1.2 (pre-commit to falsification) by
asserting that every feature column the model trains on varies per
(symbol, date). A column whose values are constant per symbol across
all dates is either a snapshot leak (the 2026-05-01 incident) or a
useless constant — either way, it must not be auto-loaded.

Run with: pytest tests/test_no_per_symbol_constants.py -v

If this test ever fails, do NOT relax it. Either:
  1. Fix the offending feature builder to produce real per-(symbol, date)
     values (build a time-series, not a snapshot broadcast), OR
  2. Move the prefix to LEAKING_EXTRA_PREFIXES in find_high_conviction.py
     and re-run.

History of why this test exists: see logs/calibration_corrections.jsonl
entry dated 2026-05-01 and reports/leakage_audit_20260501.md.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path("/Users/abhinavs./Documents/Zoom")
EXTRA = ROOT / "data/derived/extra_features.parquet"

# Prefixes that are KNOWN-LEAKING and must never be loaded into the model
# without first shipping per-date inputs. Mirrors find_high_conviction.py.
LEAKING_PREFIXES = ("scr_", "qvm_", "acad_")

# Threshold: every active feature must have median-per-symbol-nunique > 1.
# We use median (not min) because a brand-new symbol with only 1 row is
# legitimately constant for that symbol — but the typical symbol must vary.
MIN_MEDIAN_NUNIQUE = 2


@pytest.mark.skipif(not EXTRA.exists(), reason="extra_features parquet not built yet")
def test_no_leaking_prefixes_in_safe_set():
    """find_high_conviction.SAFE_EXTRA_PREFIXES must not include any
    prefix from the LEAKING set."""
    from src.agentic.find_high_conviction import SAFE_EXTRA_PREFIXES

    overlap = set(SAFE_EXTRA_PREFIXES) & set(LEAKING_PREFIXES)
    assert not overlap, (
        f"Leaking prefixes found in SAFE_EXTRA_PREFIXES: {overlap}. "
        f"See reports/leakage_audit_20260501.md."
    )


@pytest.mark.skipif(not EXTRA.exists(), reason="extra_features parquet not built yet")
def test_extra_features_safe_columns_vary_per_symbol():
    """Every column in extra_features that is currently safelisted must
    vary per (symbol, date) for the typical symbol."""
    from src.agentic.find_high_conviction import SAFE_EXTRA_PREFIXES

    df = pd.read_parquet(EXTRA)
    assert "symbol" in df.columns, "extra_features must have a 'symbol' column"

    # Drop non-feature cols
    feat_cols = [c for c in df.columns
                 if c not in ("symbol", "trade_date")
                 and c.startswith(SAFE_EXTRA_PREFIXES)]

    if not feat_cols:
        pytest.skip("no safe-prefix columns present in extra_features yet")

    failed = []
    for c in feat_cols:
        # nunique per symbol; require median > 1 (most symbols vary)
        nuniq = df.groupby("symbol")[c].nunique()
        if nuniq.median() < MIN_MEDIAN_NUNIQUE:
            failed.append((c, float(nuniq.median()), int(nuniq.min())))

    assert not failed, (
        f"\n{len(failed)} columns in SAFE_EXTRA_PREFIXES are constant per symbol "
        f"(snapshot-broadcast leak):\n"
        + "\n".join(f"  {c}: median nunique = {m:.1f}, min = {mn}"
                    for c, m, mn in failed[:20])
        + f"\n\nThis is the 2026-05-01 leak pattern. See "
        f"reports/leakage_audit_20260501.md for remediation."
    )


@pytest.mark.skipif(not EXTRA.exists(), reason="extra_features parquet not built yet")
def test_leaking_prefixes_documented_if_present_in_parquet():
    """If leaked features are still in the parquet (we haven't dropped
    them from the dataset), they must NOT be safelisted. This is a
    belt-and-braces check."""
    from src.agentic.find_high_conviction import EXTRA_PREFIXES

    df = pd.read_parquet(EXTRA)
    leaked_in_data = [c for c in df.columns if c.startswith(LEAKING_PREFIXES)]
    if not leaked_in_data:
        pytest.skip("No leaked-prefix columns in extra_features (clean dataset)")

    safelisted_leaked = [c for c in leaked_in_data
                          if c.startswith(EXTRA_PREFIXES)]
    assert not safelisted_leaked, (
        f"Leaked-prefix columns present in extra_features AND safelisted "
        f"for model loading: {safelisted_leaked[:10]}. "
        f"Update find_high_conviction.EXTRA_PREFIXES to exclude these."
    )
