"""REPAIR CA ADJUSTMENTS — fix unadjusted split/bonus cliffs in the prices parquet.

Discovered 2026-08-27: the NSE CA subject parser required "Rs" on both sides of a
face-value split ("From Rs 10 ... To Rs 1") but NSE writes the target as "Re 1/-",
so 153 (symbol, ex_date) split/bonus groups carried NO adjustment_factor. 127 of
them left REAL unadjusted cliffs in stock_daily_facts_adjusted_2015plus.parquet
(TATASTEEL -89.5% 2022, NESTLEIND -90.2% 2024, KOTAKBANK -80.3% 2026-01, ...).
Same bug class as the 2026-08-18 "113-day CA rot".

Also: the bonus regex wrongly parses preference-share bonuses ("Bonus NCRPS 3:1"
→ factor 4.0) which have NO equity price effect — those factors must be nulled or
the next rebuild corrupts the symbol (SIYSIL-class).

Policy — EMPIRICAL VALIDATION FIRST:
  a factor is only trusted where the RAW price series (raw_close, falling back to
  close where raw is null) shows the matching cliff at the ex-date (ratio within
  ±20% of the factor). Parsed-but-cliffless factors are reported, not applied.
  Factor-bearing groups whose raw series shows NO cliff (ratio < 0.5×factor) are
  BOGUS and get nulled.

Usage:
  python3 src/agentic/repair_ca_adjustments.py            # dry-run: full report, no writes
  python3 src/agentic/repair_ca_adjustments.py --apply    # backup, fix store, rebuild, verify
"""
from __future__ import annotations

import re
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.transform.corporate_actions import (  # noqa: E402
    PRICE_COLUMNS,
    QTY_COLUMNS,
    apply_split_bonus_adjustments,
)
from src.features.indicators import add_daily_price_features  # noqa: E402

STORE = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
PARQUET = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"

# Fixed parsers (mirror the fix going into src/ingest/corporate_actions/nse.py):
# "Re 1" accepted alongside "Rs 1"; NCRPS/preference bonuses excluded.
SPLIT_RE = re.compile(
    r"(?:face\s*value\s*split|stock\s*split|sub-division|subdivision|split)[^\d]{0,40}"
    r"(?:from)?\s*r(?:s|e)\.?\s*(\d+(?:\.\d+)?)\s*/?-?[^\d]{0,25}"
    r"(?:to)?\s*r(?:s|e)\.?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
BONUS_RE = re.compile(
    r"bonus(?:\s+issue)?[^\d]{0,20}(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)
NON_EQUITY_RE = re.compile(r"ncrps|preference|debenture|warrant", re.IGNORECASE)

# Human-verified factors the tape cannot confirm (suspensions, no post-ex prints).
# These are exempt from null/normalize verdicts. Source: NSE records checked by
# Abhinav 2026-08-28 — KOTYARK 10:1 bonus ex 2026-06-24 (no NSE prints since
# 2026-06-23); SUMEETINDS 1:5 split ex 2025-10-03 (3-month halt; tape ratio 3.06
# = factor 5 × real +63% relist move).
CONFIRMED = {
    ("KOTYARK", "2026-06-24"): 11.0,
    ("SUMEETINDS", "2025-10-03"): 5.0,
}


def parse_factor(subject: str) -> float | None:
    if not isinstance(subject, str):
        return None
    m = SPLIT_RE.search(subject)
    if m:
        frm, to = float(m.group(1)), float(m.group(2))
        if to > 0 and frm / to > 1:
            return frm / to
    if NON_EQUITY_RE.search(subject):
        return None
    m = BONUS_RE.search(subject)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if b > 0:
            return (a + b) / b
    return None


def raw_close_series(px: pd.DataFrame) -> pd.Series:
    """True traded close: raw_close where recorded, else close (identity rows)."""
    if "raw_close" in px.columns:
        return px["raw_close"].fillna(px["close"])
    return px["close"]


def empirical_ratios(g: pd.DataFrame, ex: pd.Timestamp) -> list[tuple[float, pd.Timestamp]]:
    """Candidate raw-close ratios at the ex-date.

    Two views, both returned: (a) last pre-ex trade vs first post-ex trade — survives
    trading-halt gaps around record dates (VERTOZ 2024 lesson: last trade 8 days before
    ex, so a ±7d consecutive-day scan saw only post-split rows and called a 19.2x split
    "no effect"); (b) biggest consecutive-day drop within ±7d — catches effects landing
    a session early/late (GOODLUCK cliff a day before its recorded ex).
    """
    out: list[tuple[float, pd.Timestamp]] = []
    # last trade before ex with NO date cap: long suspensions around record dates
    # (AARTECH 2024: 3.5 months) mean the nearest pre-ex information can be far back,
    # and by construction there are no intervening trades to prefer.
    pre = g[g["trade_date"] < ex].tail(1)
    post = g[(g["trade_date"] >= ex) & (g["trade_date"] <= ex + pd.Timedelta(days=10))].head(1)
    if len(pre) and len(post) and post.iloc[0]["_raw_close"] > 0:
        out.append((float(pre.iloc[0]["_raw_close"] / post.iloc[0]["_raw_close"]), post.iloc[0]["trade_date"]))
    win = g[(g["trade_date"] >= ex - pd.Timedelta(days=7)) & (g["trade_date"] <= ex + pd.Timedelta(days=7))]
    if len(win) >= 2:
        rc = win["_raw_close"].to_numpy()
        ratios = rc[:-1] / rc[1:]
        if len(ratios) and np.isfinite(ratios).any():
            i = int(np.nanargmax(ratios))
            out.append((float(ratios[i]), win.iloc[i + 1]["trade_date"]))
    return out


def main() -> None:
    apply_mode = "--apply" in sys.argv
    tag = date.today().isoformat()

    print("loading store + parquet …", flush=True)
    ca = pd.read_parquet(STORE)
    ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")
    px = pd.read_parquet(PARQUET)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    px["_raw_close"] = raw_close_series(px)
    pmax = px["trade_date"].max()
    by_sym = dict(tuple(px[["symbol", "trade_date", "_raw_close"]].groupby("symbol")))

    splitish = ca[ca["subject"].str.contains("plit|onus", na=False, case=False) & ca["ex_date"].notna()]

    # ---- Phase A: per-group verdicts. Production semantics: the adjuster takes the
    # PRODUCT of all non-NaN factor rows in a (symbol, ex_date) group, so validation
    # and repair operate on the group's effective product, not any single row.
    to_fill: list[tuple[str, pd.Timestamp, float]] = []      # group gets effective factor f
    to_null: list[tuple[str, pd.Timestamp, float]] = []      # group factors provably bogus → all NaN
    to_norm: list[tuple[str, pd.Timestamp, float, float]] = []  # (sym, ex, old_prod, new_f): fix double-counted groups
    unconfirmed: list[tuple[str, pd.Timestamp, float | None, str]] = []

    for (sym, ex), subj_rows in splitish.groupby(["symbol", "ex_date"]):
        facs = subj_rows["adjustment_factor"].dropna().astype(float)
        stored_prod = float(facs.prod()) if len(facs) else np.nan
        parsed_each = [parse_factor(s) for s in subj_rows["subject"]]
        parsed_prod = float(np.prod([p for p in parsed_each if p])) if any(parsed_each) else None
        g = by_sym.get(sym)
        if ex > pmax or g is None or g[g["trade_date"] <= ex].empty:
            if pd.isna(stored_prod) and parsed_prod:
                unconfirmed.append((sym, ex, parsed_prod, "no traded history at ex-date (future/unlisted)"))
            continue
        conf = CONFIRMED.get((sym, str(ex.date())))
        if conf is not None:
            if pd.isna(stored_prod):
                to_fill.append((sym, ex, conf))
            elif abs(stored_prod / conf - 1) > 0.01:
                to_norm.append((sym, ex, stored_prod, conf))
            continue  # human-verified: exempt from tape verdicts

        emp = empirical_ratios(g, ex)
        max_ratio = max((r for r, _ in emp), default=None)

        def matches(f: float | None) -> bool:
            return (f is not None and f > 1
                    and any(abs(r / f - 1) <= 0.20 for r, _ in emp))

        if pd.isna(stored_prod):
            if parsed_prod is None:
                unconfirmed.append((sym, ex, None, "subject unparseable"))
            elif matches(parsed_prod):
                to_fill.append((sym, ex, parsed_prod))
            else:
                seen = f"{max_ratio:.2f}" if max_ratio is not None else "none"
                unconfirmed.append((sym, ex, parsed_prod, f"cliff mismatch (parsed {parsed_prod:.2f}, seen {seen})"))
        else:
            if not emp:
                unconfirmed.append((sym, ex, stored_prod, "factor kept — no traded rows near ex-date to validate"))
            elif matches(stored_prod):
                pass  # group's effective product confirmed by the tape — leave alone
            else:
                # try alternates: a single row's factor (double-count case) or parsed product
                alt = next((f for f in sorted(set(facs), reverse=True) if matches(f)), None)
                alt = alt if alt is not None else (parsed_prod if matches(parsed_prod) else None)
                if alt is not None:
                    to_norm.append((sym, ex, stored_prod, alt))
                elif stored_prod >= 1.5 and max_ratio < 0.5 * stored_prod:
                    to_null.append((sym, ex, stored_prod))
                else:
                    unconfirmed.append((sym, ex, stored_prod, f"kept — tape ratio {max_ratio:.2f} inconclusive vs prod {stored_prod:.2f}"))

    print(f"\nPhase A verdicts: fill {len(to_fill)} · null-bogus {len(to_null)} · "
          f"normalize {len(to_norm)} · unconfirmed {len(unconfirmed)}")
    for sym, ex, f in sorted(to_fill):
        print(f"  FILL  {sym:12s} {ex.date()}  factor {f:.2f}")
    for sym, ex, f in sorted(to_null):
        print(f"  NULL  {sym:12s} {ex.date()}  bogus factor {f:.2f} (no price effect in raw series)")
    for sym, ex, old, new in sorted(to_norm):
        print(f"  NORM  {sym:12s} {ex.date()}  group prod {old:.2f} → tape-confirmed {new:.2f}")
    for sym, ex, f, why in sorted(unconfirmed):
        print(f"  SKIP  {sym:12s} {ex.date()}  {why}")

    # ---- DEFINITIVE rebuild set: rows whose stored price factor disagrees with
    # the FIXED store's expected step function. Catches missing adjustments, bogus
    # ones, AND partial-application batch seams (SIYSIL-class) in one scan.
    # effective per-group factor AFTER the store fix, production semantics (prod of rows)
    fill_map_scan = {(s, e): f for s, e, f in to_fill}
    null_set_scan = {(s, e) for s, e, _ in to_null}
    norm_map_scan = {(s, e): new for s, e, _, new in to_norm}
    eff = (
        splitish.groupby(["symbol", "ex_date"])["adjustment_factor"]
        .agg(lambda s: float(s.dropna().prod()) if s.notna().any() else np.nan)
        .reset_index().rename(columns={"adjustment_factor": "factor"})
    )
    eff["factor"] = [
        np.nan if (s, e) in null_set_scan
        else norm_map_scan.get((s, e), fill_map_scan.get((s, e), f))
        for s, e, f in zip(eff["symbol"], eff["ex_date"], eff["factor"])
    ]
    eff = eff.dropna(subset=["factor"])
    eff = eff[eff["factor"] > 0]  # mirror production adjuster's gt(0) filter exactly

    print("\nscanning all symbols for factor mismatches vs fixed store …", flush=True)
    mismatched: dict[str, int] = {}
    stored_col = "price_adjustment_factor_to_present"
    for sym, g in px.groupby("symbol"):
        sa = eff[eff["symbol"] == sym]
        td = g["trade_date"].to_numpy(dtype="datetime64[ns]")
        if sa.empty:
            expected = np.ones(len(g))
        else:
            exs = sa.sort_values("ex_date")["ex_date"].to_numpy(dtype="datetime64[ns]")
            fs = sa.sort_values("ex_date")["factor"].astype(float).to_numpy()
            suffix = np.cumprod(fs[::-1])[::-1]
            idx = np.searchsorted(exs, td, side="right")
            share = np.ones(len(g))
            share[idx < len(exs)] = suffix[idx[idx < len(exs)]]
            expected = 1.0 / share
        stored = g[stored_col].fillna(1.0).to_numpy(dtype=float) if stored_col in g.columns else np.ones(len(g))
        n_bad = int((~np.isclose(stored, expected, rtol=1e-6)).sum())
        if n_bad:
            mismatched[sym] = n_bad
    affected = sorted(mismatched)
    print(f"parquet rebuild set: {len(affected)} symbols "
          f"({sum(mismatched.values()):,} mismatched rows)")
    for s in affected:
        print(f"  REBUILD {s:12s} {mismatched[s]:,} rows")

    if not apply_mode:
        print("\nDRY RUN — nothing written. Re-run with --apply to execute.")
        return

    # ---- Phase B: write fixed store ------------------------------------------
    store_bak = STORE.with_suffix(f".parquet.bak-{tag}")
    if not store_bak.exists():
        shutil.copy2(STORE, store_bak)
    # Row-level rewrite: for FILL/NORM groups the FIRST row carries the validated
    # effective factor and siblings go NaN, so the production adjuster's per-group
    # PRODUCT equals exactly the tape-confirmed value. NULL groups go all-NaN.
    target_map = {**{(s, e): f for s, e, f in to_fill}, **norm_map_scan}
    null_set = {(s, e) for s, e, _ in to_null}
    new_factors = ca["adjustment_factor"].copy()
    for (sym, ex), grp in ca.groupby(["symbol", "ex_date"]):
        k = (sym, ex)
        if k in null_set:
            new_factors.loc[grp.index] = np.nan
        elif k in target_map:
            new_factors.loc[grp.index] = np.nan
            new_factors.loc[grp.index[0]] = target_map[k]
    ca["adjustment_factor"] = new_factors
    store_tmp = STORE.with_suffix(".parquet.tmp")
    ca.to_parquet(store_tmp, index=False)
    store_tmp.replace(STORE)
    print(f"store written atomically ({store_bak.name} kept as backup)")

    # ---- Phase C: rebuild affected symbols from raw --------------------------
    pq_bak = PARQUET.with_suffix(f".parquet.bak-{tag}")
    if not pq_bak.exists():
        shutil.copy2(PARQUET, pq_bak)

    px = px.drop(columns=["_raw_close"])
    mask = px["symbol"].isin(affected)
    fixed_part = px[mask].copy()
    # PRECONDITION (review F5): the fillna(col) fallback treats the current value as
    # raw, which is only valid where no adjustment was applied. Any adjusted row
    # (factor != 1) with a null raw_ cell would silently double-adjust — abort instead.
    fcol = "price_adjustment_factor_to_present"
    if fcol in fixed_part.columns:
        adj_rows = fixed_part[fcol].fillna(1.0) != 1.0
        for col in PRICE_COLUMNS:
            raw = f"raw_{col}"
            if col in fixed_part.columns and raw in fixed_part.columns:
                n_bad_raw = int((adj_rows & fixed_part[raw].isna() & fixed_part[col].notna()).sum())
                if n_bad_raw:
                    raise RuntimeError(
                        f"{n_bad_raw} adjusted rows have null {raw} — restoration would "
                        "double-adjust. Inspect manually; NOT proceeding.")
    # restore true raw values, then let the production adjuster redo everything
    for col in list(PRICE_COLUMNS) + list(QTY_COLUMNS):
        raw = f"raw_{col}"
        if col in fixed_part.columns and raw in fixed_part.columns:
            fixed_part[col] = fixed_part[raw].fillna(fixed_part[col])
    drop_cols = [c for c in fixed_part.columns if c.startswith("raw_")] + [
        "share_adjustment_factor_to_present",
        "price_adjustment_factor_to_present",
        "future_split_bonus_action_count",
    ]
    fixed_part = fixed_part.drop(columns=[c for c in drop_cols if c in fixed_part.columns])
    ca_aff = ca[ca["symbol"].isin(affected)]
    fixed_part = apply_split_bonus_adjustments(fixed_part, ca_aff)

    combined = pd.concat([px[~mask], fixed_part], ignore_index=True)
    combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    roll_cols = [c for c in combined.columns if c.startswith(
        ("sma_", "rsi_", "return_", "volume_vs_", "traded_value_vs_", "avg_traded_value_"))]
    print(f"recomputing {len(roll_cols)} rolling feature cols over {len(combined):,} rows …", flush=True)
    featured = add_daily_price_features(combined.drop(columns=roll_cols))

    tmp = PARQUET.with_suffix(".parquet.tmp")
    featured.to_parquet(tmp, index=False)
    tmp.replace(PARQUET)
    print(f"parquet written ({pq_bak.name} kept as backup)")

    # ---- Phase D: verification ------------------------------------------------
    print("\nVERIFY:", flush=True)
    vcols = ["symbol", "trade_date", "close", "return_1d", "price_adjustment_factor_to_present"]
    v = pd.read_parquet(PARQUET, columns=vcols)
    v["trade_date"] = pd.to_datetime(v["trade_date"])
    bad = 0

    # D1: every filled cliff must be gone; every nulled group must have no fake jump
    for kind, triples in (("FILL", to_fill), ("NULL", to_null)):
        for sym, ex, f in triples:
            if (sym, str(ex.date())) in CONFIRMED:
                print(f"  ⏭  {kind} {sym:12s} {ex.date()}  human-verified — cliff check skipped (real move may exceed 30%)")
                continue
            g = v[(v["symbol"] == sym) & (v["trade_date"].between(ex - pd.Timedelta(days=15), ex + pd.Timedelta(days=10)))]
            worst = g["return_1d"].abs().max()
            ok = pd.isna(worst) or worst < 0.30
            bad += 0 if ok else 1
            print(f"  {'✅' if ok else '❌'} {kind} {sym:12s} {ex.date()}  max |ret| ±5d now {worst*100:+.1f}%")

    # D2: definitive re-scan — stored factor must match the fixed store everywhere
    print("  re-running factor-mismatch scan on repaired parquet …", flush=True)
    n_mismatch = 0
    for sym, g in v.groupby("symbol"):
        sa = eff[eff["symbol"] == sym]
        td = g["trade_date"].to_numpy(dtype="datetime64[ns]")
        if sa.empty:
            expected = np.ones(len(g))
        else:
            exs = sa.sort_values("ex_date")["ex_date"].to_numpy(dtype="datetime64[ns]")
            fs = sa.sort_values("ex_date")["factor"].astype(float).to_numpy()
            suffix = np.cumprod(fs[::-1])[::-1]
            idx = np.searchsorted(exs, td, side="right")
            share = np.ones(len(g))
            share[idx < len(exs)] = suffix[idx[idx < len(exs)]]
            expected = 1.0 / share
        stored = g["price_adjustment_factor_to_present"].fillna(1.0).to_numpy(dtype=float)
        n_mismatch += int((~np.isclose(stored, expected, rtol=1e-6)).sum())
    print(f"  factor mismatches remaining: {n_mismatch}")
    bad += 1 if n_mismatch else 0

    # D3 (review F2): the factor column alone can be right while close is doubly
    # adjusted — assert close == raw_close × stored_factor wherever raw is recorded.
    v3 = pd.read_parquet(PARQUET, columns=["symbol", "close", "raw_close", "price_adjustment_factor_to_present"])
    v3 = v3[v3["symbol"].isin(affected) & v3["raw_close"].notna() & v3["close"].notna()]
    resid = (v3["close"] - v3["raw_close"] * v3["price_adjustment_factor_to_present"].fillna(1.0)).abs()
    rel = (resid / v3["close"].abs().clip(lower=1e-9)).max() if len(v3) else 0.0
    print(f"  close vs raw×factor max relative residual (affected symbols): {rel:.2e}")
    if rel > 1e-6:
        bad += 1
        print("  ❌ close values inconsistent with raw×factor — possible double adjustment")

    if bad:
        print(f"\n❌ {bad} verification failures — INVESTIGATE (backups kept: {pq_bak.name}, {store_bak.name}).")
        sys.exit(1)
    print(f"\n✅ REPAIR COMPLETE: {len(to_fill)} cliffs filled, {len(to_null)} bogus factors nulled, "
          f"{len(affected)} symbols rebuilt, zero factor mismatches remain.")


if __name__ == "__main__":
    main()
