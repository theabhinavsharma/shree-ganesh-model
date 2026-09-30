"""PRICE-ONLY corporate-action factors — demergers, schemes, rights, special dividends, capital
reductions / consolidations (2026-09-27, audit root cause 5).

The production adjuster only handled splits/bonuses (share-count identity). Every other corporate action
left an unadjusted cliff in the 'adjusted' close (CROMPGREAV -71.7% on 2016-03-15 demerger, IIFL -60.6%
2019-06-14): 101 demergers, 286 rights, 47 schemes, 5 capital reductions, 2 consolidations and 344 special
dividends had no factor. This builds one row per such CA with the factor, the method and its inputs:

  demerger (subject 'demerger')      empirical: raw open of the first session on/after ex_date /
                                     raw close of the last session before it (NSE's special pre-open
                                     prices the residual). Applied when 0.02 <= f <= 0.95 and the gap
                                     between the two sessions <= 30 days.
  scheme of arrangement (no demerger) same empirical factor, applied only when f <= 0.85.
  capital reduction / consolidation  same empirical factor, applied when f <= 0.80 or f >= 1.25
                                     (price-only: quantities are not rescaled — documented limitation).
  rights 'a:b @ Premium Rs P'         TERP: S = face value + premium, Pcum = raw close before ex;
                                     f = (b*Pcum + a*S) / ((a+b)*Pcum). Applied when f < 0.995.
  special dividend 'Special Dividend - Rs D'  f = (Pcum - D) / Pcum when 0 < D < Pcum.
Regular dividends stay unadjusted (the panel is a price-return series).

Output: data/corporate_actions_full_history/normalized/price_only_ca_factors.parquet (+ manifest).
The production adjuster (src/transform/corporate_actions.py) loads the APPLIED rows automatically;
refresh_prices' LATE-CA SELF-HEAL expects them. --apply re-adjusts the panel for symbols whose stored
price factor differs from the expected one (backup first) and recomputes their rolling features.
Raw-basis prices come from the panel as adjusted / price_adjustment_factor_to_present (idempotent).
"""
from __future__ import annotations

import argparse
import re as _re
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT))
from src.transform.corporate_actions import (  # noqa: E402
    PRICE_COLUMNS, QTY_COLUMNS, PRICE_ONLY_PATH, apply_split_bonus_adjustments, expected_price_factor, load_price_only_factors,
)
from src.features.indicators import add_daily_price_features  # noqa: E402

PANEL = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CA_PATH = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
RIGHTS_RE = re.compile(r"rights\s*(\d+)\s*:\s*(\d+)\s*@\s*(?:(?:prem\w*|premium)\s*(?:of\s*)?(?:rs|re)\.?\s*([\d.]+)|(par))", re.I)
SPDIV_RE = re.compile(r"special\s+dividend\s*[-:]?\s*(?:of\s*)?(?:rs|re)\.?\s*([\d.]+)", re.I)


DIV_AMT_RE = _re.compile(r"(?:Rs\.?|Re\.?|₹)\s*([0-9]+(?:\.[0-9]+)?)", _re.I)


def kind_of(subject: str) -> str | None:
    s = subject.lower()
    if "demerger" in s:
        return "demerger"
    if "consolidat" in s:
        return "consolidation"
    if "capital reduction" in s or "reduction of capital" in s:
        return "capital_reduction"
    if "scheme of arrangement" in s:
        return "scheme"
    if re.search(r"\brights\b", s) and not re.search(r"rights?\s+entitlement", s):
        return "rights"
    if "special dividend" in s:
        return "special_dividend"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="re-adjust panel rows of affected symbols (back up the panel first)")
    args = ap.parse_args()

    ca = pd.read_parquet(CA_PATH)
    ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce").dt.normalize()
    ca["kind"] = ca["subject"].astype(str).map(kind_of)
    # 2026-09-30: a LARGE ordinary dividend (>= 10% of the price) moves the price like a special one (MAJESCO paid
    # Rs 974 as an "Interim Dividend" on 2020-12-23: -99% in the panel). Candidate here; applied below only if >= 10%.
    div = ca["kind"].isna() & ca["subject"].astype(str).str.contains("dividend", case=False) & \
        ~ca["subject"].astype(str).str.contains("split|sub-division|bonus", case=False)
    ca.loc[div, "kind"] = "large_dividend"
    C = ca[ca["kind"].notna() & ca["ex_date"].notna()].drop_duplicates(["symbol", "ex_date", "kind"]).copy()
    print(f"price-only CA candidates: {len(C)} · {C['kind'].value_counts().to_dict()}", flush=True)

    px = pd.read_parquet(PANEL, columns=["symbol", "trade_date", "open", "close", "price_adjustment_factor_to_present"],
                         filters=[("symbol", "in", sorted(C["symbol"].unique()))])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    f = px["price_adjustment_factor_to_present"].replace(0, np.nan).fillna(1.0)
    px["raw_open"] = px["open"] / f; px["raw_close"] = px["close"] / f
    px = px.sort_values(["symbol", "trade_date"])
    by = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}

    rows = []
    for r in C.itertuples():
        g = by.get(r.symbol)
        rec = dict(symbol=r.symbol, ex_date=r.ex_date, kind=r.kind, subject=r.subject, face_value=getattr(r, "face_value", np.nan),
                   price_factor=np.nan, applied=False, method=None, reason=None)
        if g is None or g.empty:
            rec["reason"] = "symbol not in panel"; rows.append(rec); continue
        i = int(np.searchsorted(g["trade_date"].values, np.datetime64(r.ex_date), side="left"))
        if i == 0 or i >= len(g):
            rec["reason"] = "no session before/after ex_date in panel"; rows.append(rec); continue
        prev, ex = g.iloc[i - 1], g.iloc[i]
        rec.update(prev_session=prev["trade_date"], ex_session=ex["trade_date"], raw_prev_close=prev["raw_close"], raw_ex_open=ex["raw_open"],
                   gap_days=int((ex["trade_date"] - prev["trade_date"]).days))
        pcum = prev["raw_close"]
        if r.kind in ("demerger", "scheme", "capital_reduction", "consolidation"):
            fac = ex["raw_open"] / pcum if pcum and pd.notna(ex["raw_open"]) and ex["raw_open"] > 0 else np.nan
            rec.update(method="empirical ex-open / prior close", price_factor=fac)
            if pd.isna(fac):
                rec["reason"] = "missing prices"
            elif rec["gap_days"] > 30:
                rec["reason"] = f"sessions {rec['gap_days']}d apart — value transfer not separable"
            elif r.kind == "demerger" and 0.02 <= fac <= 0.95:
                rec["applied"] = True
            elif r.kind == "scheme" and 0.02 <= fac <= 0.85:
                rec["applied"] = True
            elif r.kind in ("capital_reduction", "consolidation") and (fac <= 0.80 or fac >= 1.25):
                rec["applied"] = True
            else:
                rec["reason"] = f"ex-date move {fac - 1:+.1%} not a value transfer under the {r.kind} rule"
        elif r.kind == "rights":
            m = RIGHTS_RE.search(str(r.subject))
            fv = pd.to_numeric(rec["face_value"], errors="coerce")
            if not m:
                rec["reason"] = "rights ratio/price not parseable"
            elif pd.isna(fv) and not m.group(4):
                rec["reason"] = "face value missing"
            else:
                a, b = float(m.group(1)), float(m.group(2))
                s_px = (0.0 if pd.isna(fv) else float(fv)) + (0.0 if m.group(4) else float(m.group(3)))
                if m.group(4):
                    s_px = float(fv) if pd.notna(fv) else np.nan
                fac = (b * pcum + a * s_px) / ((a + b) * pcum) if pcum and pd.notna(s_px) else np.nan
                rec.update(method=f"TERP rights {int(a)}:{int(b)} @ {s_px:g}", price_factor=fac, issue_price=s_px)
                if pd.notna(fac) and fac < 0.995:
                    rec["applied"] = True
                else:
                    rec["reason"] = "issue price >= cum price (no value transfer)" if pd.notna(fac) else "missing prices"
        elif r.kind == "large_dividend":
            amts = [float(a) for a in DIV_AMT_RE.findall(str(r.subject))]
            d = sum(amts)
            share = d / pcum if pcum and d > 0 else np.nan
            rec.update(method=f"dividend Rs {d:g} = {share:.1%} of the prior close" if pd.notna(share) else "dividend amount not parseable",
                       price_factor=(pcum - d) / pcum if pd.notna(share) and share < 1 else np.nan, dividend=d)
            if pd.notna(share) and 0.10 <= share < 1:
                rec["applied"] = True
            else:
                rec["reason"] = "ordinary dividend under 10% of the price (price-return series keeps it)" if pd.notna(share) else "amount not parseable"
        elif r.kind == "special_dividend":
            m = SPDIV_RE.search(str(r.subject))
            if not m:
                rec["reason"] = "special dividend amount not parseable"
            else:
                d = float(m.group(1))
                fac = (pcum - d) / pcum if pcum and 0 < d < pcum else np.nan
                rec.update(method=f"special dividend Rs {d:g}", price_factor=fac, dividend=d)
                if pd.notna(fac):
                    rec["applied"] = True
                else:
                    rec["reason"] = "dividend >= price or missing prices"
        rows.append(rec)

    T = pd.DataFrame(rows)
    T["price_factor"] = pd.to_numeric(T["price_factor"], errors="coerce")
    T.to_parquet(PRICE_ONLY_PATH, index=False)
    summ = T.groupby("kind")["applied"].agg(["size", "sum"]).rename(columns={"size": "rows", "sum": "applied"})
    PRICE_ONLY_PATH.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="price_only_ca_factors", path=str(PRICE_ONLY_PATH.relative_to(ROOT)), key=["symbol", "ex_date", "kind"],
        producer="src/agentic/build_price_only_ca_factors.py", consumers=["src/transform/corporate_actions.py (auto-loaded)", "src/agentic/refresh_prices.py self-heal"],
        columns=dict(price_factor="multiplier applied to every PRICE before ex_date (quantities untouched); <1 = value left the stock",
                     applied="True rows are used by the adjuster", method="how the factor was derived", reason="why a row was not applied",
                     raw_prev_close="raw close of last session before ex_date (Rs)", raw_ex_open="raw open of first session on/after ex_date (Rs)",
                     gap_days="calendar days between those sessions"),
        summary=summ.to_dict("index"), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(summ.to_string(), flush=True)
    print(T[T["applied"]].sort_values("price_factor").head(8)[["symbol", "ex_date", "kind", "price_factor", "method"]].to_string(index=False), flush=True)

    if not args.apply:
        return
    # ---------------- re-adjust affected symbols in the panel ----------------
    print("loading panel for apply ...", flush=True)
    panel = pd.read_parquet(PANEL)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    eff = (ca[ca["adjustment_factor"].notna() & ca["adjustment_factor"].gt(0) & ca["ex_date"].notna()]
           .groupby(["symbol", "ex_date"])["adjustment_factor"].prod().reset_index())
    po = load_price_only_factors()
    stale = []
    for sym, g in panel[panel["symbol"].isin(set(po["symbol"]))].groupby("symbol"):
        exp = expected_price_factor(g["trade_date"].to_numpy(dtype="datetime64[ns]"), eff[eff["symbol"] == sym], po[po["symbol"] == sym])
        if (~np.isclose(g["price_adjustment_factor_to_present"].fillna(1.0).to_numpy(dtype=float), exp, rtol=1e-6)).any():
            stale.append(sym)
    print(f"re-adjusting {len(stale)} symbols", flush=True)
    if not stale:
        return
    mask = panel["symbol"].isin(stale)
    part = panel[mask].copy()
    for col in list(PRICE_COLUMNS) + list(QTY_COLUMNS):
        raw = f"raw_{col}"
        if col in part.columns and raw in part.columns:
            part[col] = part[raw].fillna(part[col] / part["price_adjustment_factor_to_present"].fillna(1.0)
                                         if col in PRICE_COLUMNS else part[col] / part["share_adjustment_factor_to_present"].fillna(1.0))
    drop = [c for c in part.columns if c.startswith("raw_")] + ["share_adjustment_factor_to_present", "price_adjustment_factor_to_present",
                                                                 "future_split_bonus_action_count"]
    keep_cols = panel.columns
    part = apply_split_bonus_adjustments(part.drop(columns=[c for c in drop if c in part.columns]), ca[ca["symbol"].isin(stale)], price_only=po[po["symbol"].isin(stale)])
    roll = [c for c in part.columns if c.startswith(("sma_", "ema_", "rsi_", "return_", "volume_vs_", "traded_value_vs_", "avg_traded_value_",
                                                       "avg_vol_", "vol_max_", "volume_high_", "avg_delivery", "delivery_qty_vs", "delivery_pct_max",
                                                       "delivery_pct_vs", "delivery_above", "delivery_pct_high"))]
    part = add_daily_price_features(part.drop(columns=roll))
    part = part.reindex(columns=keep_cols)
    out = pd.concat([panel[~mask], part], ignore_index=True).sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    assert not out.duplicated(["symbol", "trade_date"]).any()
    out.to_parquet(PANEL, index=False)
    chk = out[(out["symbol"] == "CROMPGREAV") & out["trade_date"].between("2016-03-11", "2016-03-16")][["trade_date", "close", "return_1d"]]
    print("CROMPGREAV check:\n" + chk.to_string(index=False), flush=True)
    print(f"wrote {PANEL} ({len(out):,} rows)", flush=True)


if __name__ == "__main__":
    main()
