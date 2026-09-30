"""Phase 2 pillars for EXP-2026-09-29-industry-policy (registered in logs/experiments.jsonl before any outcome).

Adds to data/derived/industry_scores.parquet (Phase 1, build_industry_scores.py), for every (date, industry):
  E_budget   latest Union Budget published strictly before the date: per ministry_key, capital expenditure + grants for
             creation of capital assets (demand rows): BE (or Interim-BE) of the new fiscal year / RE of the current
             fiscal year - 1, both printed in that same document; industry = mean over ministries mapped in
             budget_industry_map.csv; E = cross-industry percentile rank (mapped industries only)
  G_activity IIP + core-sector rows released strictly before the date (latest vintage per month; per series the newest
             base year with >= 6 known months): mean YoY% of the last 3 known months (level) and that minus the mean of
             the 3 months before (acceleration); industry = mean over HIGH-confidence links in activity_industry_map.csv;
             G = mean of the two percentile ranks
  P          mean of available E, G percentile ranks;  F5 = mean of available A, B, C, E, G percentile ranks (>= 2)
Output: data/derived/industry_scores_policy.parquet (+ manifest).
H_policy (added 2026-09-30 after the PIB backfill, as registered in EXP-2026-09-29-industry-policy): PIB releases
tagged to the industry (pib_releases.parquet industries_tagged) with pub_date in (d-90d, d] vs (d-180d, d-90d]:
H_raw = log((n1+1)/(n2+1)); H = percentile rank that day. Industries never tagged, and dates before 2019-06-30 (the
PIB history starts 2019-01-01, so the earlier window would be empty), have no H. New columns only: P_H = mean of
available E, G, H; F5_H = mean of available A, B, C, E, G, H (>= 2 present). P / F5 (production G1) are unchanged.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Code/Zoom")
DER = ROOT / "data/derived"
IN, OUT = DER / "industry_scores.parquet", DER / "industry_scores_policy.parquet"


def pct(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def budget_table() -> pd.DataFrame:
    b = pd.read_parquet(DER / "budget_capex.parquet")
    b = b[b["definition"].isin(["capital_expenditure", "grants_for_creation_of_capital_assets"]) & (b["row_type"] == "demand")]
    b["pub_date"] = pd.to_datetime(b["pub_date"])
    rows = []
    for bid, g in b.groupby("budget_id"):
        pub = g["pub_date"].max()
        be = g[g["estimate_type"].isin(["BE", "Interim-BE"])]
        new_fy = be["fiscal_year"].max()
        re_ = g[g["estimate_type"] == "RE"]
        cur_fy = re_["fiscal_year"].max()
        s_be = be[be["fiscal_year"] == new_fy].groupby("ministry_key")["capex_cr"].sum()
        s_re = re_[re_["fiscal_year"] == cur_fy].groupby("ministry_key")["capex_cr"].sum()
        both = pd.concat([s_be.rename("be"), s_re.rename("re")], axis=1).dropna()
        both = both[(both["be"] > 0) & (both["re"] > 0)]
        rows.append(pd.DataFrame(dict(budget_id=bid, pub_date=pub, new_fy=new_fy, ministry_key=both.index,
                                      growth=both["be"] / both["re"] - 1)))
    T = pd.concat(rows, ignore_index=True)
    m = pd.read_csv(DER / "budget_industry_map.csv")
    T = T.merge(m[["ministry_key", "industry"]], on="ministry_key")
    return T.groupby(["pub_date", "industry"], as_index=False)["growth"].mean().rename(columns={"growth": "E_raw"})


def activity_table() -> pd.DataFrame:
    iip = pd.read_parquet(DER / "iip_monthly.parquet").assign(source="iip")
    core = pd.read_parquet(DER / "core_sector_monthly.parquet").assign(source="core", series_type="core")
    cols = ["source", "series_type", "series_code", "base_year", "month", "yoy_pct", "release_date"]
    A = pd.concat([iip[cols], core[cols]], ignore_index=True)
    A["series_code"] = A["series_code"].astype(str)
    A["release_date"] = pd.to_datetime(A["release_date"]); A["month"] = pd.to_datetime(A["month"])
    mp = pd.read_csv(DER / "activity_industry_map.csv")
    mp = mp[mp["confidence"] == "high"].copy(); mp["series_code"] = mp["series_code"].astype(str)
    key = ["source", "series_type", "series_code"]
    A = A.merge(mp[key].drop_duplicates(), on=key)
    out = []
    for r in np.sort(A["release_date"].dropna().unique()):
        K = A[A["release_date"] <= r].sort_values("release_date").drop_duplicates(key + ["base_year", "month"], keep="last")
        for sk, g in K.groupby(key):
            g = g.dropna(subset=["yoy_pct"])
            bases = g.groupby("base_year")["month"].count()
            bases = bases[bases >= 6]
            if bases.empty:
                continue
            g = g[g["base_year"] == max(bases.index)].sort_values("month")
            y = g["yoy_pct"].to_numpy()
            if len(y) < 6:
                continue
            lvl, prev = np.nanmean(y[-3:]), np.nanmean(y[-6:-3])
            out.append(dict(known_after=pd.Timestamp(r), source=sk[0], series_type=sk[1], series_code=sk[2], level=lvl, accel=lvl - prev))
    S = pd.DataFrame(out).merge(mp[key + ["industry"]], on=key)
    return S.groupby(["known_after", "industry"], as_index=False)[["level", "accel"]].mean()


def asof_by_industry(D: pd.DataFrame, T: pd.DataFrame, tcol: str, cols: list[str]) -> pd.DataFrame:
    """For each (date, industry) take T's latest row with T[tcol] strictly before the date."""
    L = D[["date", "industry"]].reset_index().sort_values("date")
    R = T.sort_values(tcol).rename(columns={tcol: "_t"})
    m = pd.merge_asof(L, R, left_on="date", right_on="_t", by="industry", direction="backward", allow_exact_matches=False)
    return m.set_index("index")[cols].reindex(D.index)


def pib_attention(S: pd.DataFrame) -> pd.Series:
    """log((n1+1)/(n2+1)): PIB releases tagged to the industry in (d-90d, d] vs (d-180d, d-90d]."""
    P = pd.read_parquet(ROOT / "data/derived/pib_releases.parquet", columns=["pub_date", "industries_tagged"])
    P["pub_date"] = pd.to_datetime(P["pub_date"], errors="coerce")
    T = P.assign(t=P["industries_tagged"].map(lambda x: list(x) if x is not None else [])).explode("t").dropna(subset=["t", "pub_date"])
    by = {k: np.sort(g["pub_date"].to_numpy()) for k, g in T.groupby("t")}
    start = P["pub_date"].min() + pd.Timedelta(days=180)
    out = np.full(len(S), np.nan)
    d = S["date"].to_numpy()
    for i, (dt, ind) in enumerate(zip(d, S["industry"])):
        a = by.get(ind)
        if a is None or dt < np.datetime64(start):
            continue
        c = lambda lo, hi: np.searchsorted(a, hi, side="right") - np.searchsorted(a, lo, side="right")  # noqa: E731
        n1 = c(dt - np.timedelta64(90, "D"), dt); n2 = c(dt - np.timedelta64(180, "D"), dt - np.timedelta64(90, "D"))
        out[i] = np.log((n1 + 1) / (n2 + 1))
    return pd.Series(out, index=S.index)


def main() -> None:
    S = pd.read_parquet(IN); S["date"] = pd.to_datetime(S["date"]).astype("datetime64[ns]")
    E = budget_table(); E["pub_date"] = E["pub_date"].astype("datetime64[ns]")
    G = activity_table(); G["known_after"] = G["known_after"].astype("datetime64[ns]")
    S["E_raw"] = asof_by_industry(S, E, "pub_date", ["E_raw"])["E_raw"]
    S[["G_level", "G_accel"]] = asof_by_industry(S, G, "known_after", ["level", "accel"]).to_numpy()
    S["E"] = pct(S, "E_raw")
    S["G"] = pd.concat([pct(S, "G_level"), pct(S, "G_accel")], axis=1).mean(axis=1)
    S["P"] = S[["E", "G"]].mean(axis=1)
    S["P_pct"] = pct(S, "P")
    k = S[["A", "B", "C", "E", "G"]].notna().sum(axis=1)
    S["F5"] = S[["A", "B", "C", "E", "G"]].mean(axis=1).where(k >= 2)
    S["F5_pct"] = pct(S, "F5")
    S["H_raw"] = pib_attention(S)
    S["H"] = pct(S, "H_raw")
    S["P_H"] = S[["E", "G", "H"]].mean(axis=1)
    S["P_H_pct"] = pct(S, "P_H")
    k6 = S[["A", "B", "C", "E", "G", "H"]].notna().sum(axis=1)
    S["F5_H"] = S[["A", "B", "C", "E", "G", "H"]].mean(axis=1).where(k6 >= 2)
    S["F5_H_pct"] = pct(S, "F5_H")
    S.to_parquet(OUT, index=False)
    cov = S.groupby(S["date"].dt.year)[["E", "G", "H", "P", "F5"]].apply(lambda x: (x.notna().mean() * 100).round(0))
    OUT.with_suffix(".parquet.manifest.json").write_text(json.dumps(dict(
        dataset="industry_scores_policy", path=str(OUT.relative_to(ROOT)), rows=len(S), key=["date", "industry"],
        producer="src/agentic/build_policy_scores.py", experiment="EXP-2026-09-29-industry-policy", definitions=__doc__,
        coverage_pct_by_year=cov.to_dict(orient="index"),
        units=dict(E_raw="fraction (BE/RE - 1)", G_level="percent (mean YoY% of 3 months)", G_accel="percentage points", pct="percentile rank that day (0-1]"),
        inputs=["data/derived/pib_releases.parquet", "data/derived/industry_scores.parquet", "data/derived/budget_capex.parquet", "data/derived/budget_industry_map.csv",
                "data/derived/iip_monthly.parquet", "data/derived/core_sector_monthly.parquet", "data/derived/activity_industry_map.csv"],
        updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(S):,} rows\ncoverage % of industry-days:\n{cov.to_string()}")
    last = S[S["date"] == S["date"].max()].dropna(subset=["P"]).sort_values("P", ascending=False)
    print("\nlatest date, industries with policy/activity data (top 12 by P):")
    print(last[["industry", "E_raw", "G_level", "G_accel", "P_pct", "heat_pct"]].head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
