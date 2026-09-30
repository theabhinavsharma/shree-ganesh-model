"""Theme engine data (2026-09-30): which themes are rising or fading, and which companies are tied to each.

Plain English: configs/themes.json lists 22 themes (defence, railways, solar, EV, AI, chemicals, metals, ...).
  intensity  government themes: the theme's SHARE of PIB release titles in the last 4 quarters vs the 4 before
             (log(((n1+1)/(N1+1)) / ((n2+1)/(N2+1))); rising >= +0.2, fading <= -0.2). PIB's archive starts
             2017-01, so the signal starts 2019-01. Titles only, so every year is measured the same way.
             commodity themes: 6-month change of IMF monthly prices (FRED, point in time via known_from;
             rising >= +10%, fading <= -10%). Weekly, on every weekly screen date.
  exposure   a company is tied to a theme when its own NSE filing (desc + attchmntText, the company's own name
             removed) matches the theme's company keywords at a word boundary; kept as filing dates so any date can
             ask "matched in the last 365 days?" (point in time).
Output: data/derived/theme_intensity.parquet (date, theme, family, driver, score, state) and
        data/derived/theme_exposure.parquet (symbol, theme, an_dt) + manifests.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
CFG = json.loads((ROOT / "configs/themes.json").read_text())
TH = CFG["thresholds"]


def intensity(dates: pd.DatetimeIndex) -> pd.DataFrame:
    P = pd.read_parquet(ROOT / "data/derived/pib_releases.parquet", columns=["pub_date", "title"])
    P["pub_date"] = pd.to_datetime(P["pub_date"], errors="coerce")
    P = P.dropna(subset=["pub_date"])
    txt = P["title"].fillna("")
    first = P["pub_date"].min()
    allp = np.sort(P["pub_date"].to_numpy())
    F = pd.read_parquet(ROOT / "data/derived/macro_drivers_fred.parquet")
    F["date"] = pd.to_datetime(F["date"]); F["known_from"] = pd.to_datetime(F["known_from"])
    rows = []
    for t in CFG["themes"]:
        if t["driver"] == "govt":
            hits = np.sort(P.loc[txt.str.contains(t["pib"], flags=re.I, regex=True), "pub_date"].dropna().to_numpy())
            for d in dates:
                d64 = np.datetime64(d)
                if d - pd.Timedelta(days=730) < first:
                    score = np.nan
                else:
                    c = lambda a, lo, hi: np.searchsorted(a, hi, side="right") - np.searchsorted(a, lo, side="right")  # noqa: E731
                    y1, y2 = (d64 - np.timedelta64(365, "D"), d64), (d64 - np.timedelta64(730, "D"), d64 - np.timedelta64(365, "D"))
                    n1, n2, N1, N2 = c(hits, *y1), c(hits, *y2), c(allp, *y1), c(allp, *y2)
                    score = float(np.log(((n1 + 1) / (N1 + 1)) / ((n2 + 1) / (N2 + 1))))
                st = "n/a" if np.isnan(score) else ("rising" if score >= TH["govt_rising"] else "fading" if score <= TH["govt_fading"] else "steady")
                rows.append(dict(date=d, theme=t["id"], family=t["family"], driver="govt", score=score, state=st))
        else:
            G = F[F["series"].isin(t["commodities"])]
            for d in dates:
                k = G[G["known_from"] <= d]                                   # only months public by d
                chg = []
                for ser, g in k.groupby("series"):
                    g = g.sort_values("date")
                    if len(g) >= 7:
                        chg.append(g["value"].iloc[-1] / g["value"].iloc[-7] - 1)   # latest known month vs 6 months before
                score = float(np.mean(chg)) if chg else np.nan
                st = "n/a" if np.isnan(score) else ("rising" if score >= TH["commodity_rising"] else "fading" if score <= TH["commodity_fading"] else "steady")
                rows.append(dict(date=d, theme=t["id"], family=t["family"], driver="commodity", score=score, state=st))
    return pd.DataFrame(rows)


def exposure() -> pd.DataFrame:
    A = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "an_dt", "desc", "attchmntText"])
    A["an_dt"] = pd.to_datetime(A["an_dt"], errors="coerce", format="mixed")
    body = A["attchmntText"].fillna("").str.replace(r"^.{0,200}? has (informed|submitted|intimated|disclosed|announced)\b", "", regex=True)
    txt = A["desc"].fillna("") + " " + body                                    # the company's own name removed
    out = []
    for t in CFG["themes"]:
        m = txt.str.contains(r"\b(?:" + t["company"] + ")", flags=re.I, regex=True)
        out.append(A.loc[m, ["symbol", "an_dt"]].assign(theme=t["id"]))
        print(f"  {t['id']:12s} filings matched {int(m.sum()):7d} · companies {A.loc[m, 'symbol'].nunique()}", flush=True)
    return pd.concat(out, ignore_index=True).dropna(subset=["an_dt"])


def fading_checker():
    """fading_only(symbol, date) -> True when the company's filings in the last 365 days tie it to at least one fading
    theme and to no rising theme. Same rule as test_theme_engine.py TFADE (EXP-2026-09-30-theme-engine / v3-vs-g1);
    used live by screen_sri_lakshmi.py (V3). Also returns the newest filing date and the newest intensity date."""
    I = pd.read_parquet(ROOT / "data/derived/theme_intensity.parquet"); I["date"] = pd.to_datetime(I["date"])
    state = {(d, t): st for d, t, st in zip(I["date"], I["theme"], I["state"])}
    E = pd.read_parquet(ROOT / "data/derived/theme_exposure.parquet"); E["an_dt"] = pd.to_datetime(E["an_dt"])
    ex = {(s, t): np.sort(g["an_dt"].to_numpy()) for (s, t), g in E.groupby(["symbol", "theme"])}
    tof: dict = {}
    for (s, t) in ex:
        tof.setdefault(s, []).append(t)

    def fading_only(s, d) -> bool:
        d64 = np.datetime64(d + pd.Timedelta(hours=23, minutes=59)); st = []
        for t in tof.get(s, []):
            a = ex[(s, t)]
            if np.searchsorted(a, d64, side="right") - np.searchsorted(a, d64 - np.timedelta64(365, "D"), side="right") > 0:
                st.append(state.get((d, t), "n/a"))
        return "fading" in st and "rising" not in st
    src = pd.to_datetime(pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["an_dt"])["an_dt"]).max()
    return fading_only, src, I["date"].max()


def main() -> None:
    import sim_leader_portfolio_7x as sp
    cal = pd.to_datetime(pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["trade_date"])["trade_date"].unique())
    cal = pd.DatetimeIndex(sorted(cal))
    dates = pd.DatetimeIndex(sorted({d for o in range(5) for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp("2016-01-01")}))
    I = intensity(dates)
    I.to_parquet(ROOT / "data/derived/theme_intensity.parquet", index=False)
    E = exposure()
    E.to_parquet(ROOT / "data/derived/theme_exposure.parquet", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    st = I[I["state"] != "n/a"].groupby([I["date"].dt.year, "state"]).size().unstack(fill_value=0)
    (ROOT / "data/derived/theme_intensity.parquet.manifest.json").write_text(json.dumps(dict(
        dataset="theme_intensity", producer="src/agentic/build_themes.py", config="configs/themes.json", rows=len(I), definitions=__doc__,
        columns=dict(date="weekly screen date", theme="configs/themes.json id", score="govt: log((n1+1)/(n2+1)); commodity: 6-month price change as a fraction",
                     state="rising / steady / fading / n/a (not enough history)"),
        states_by_year={str(k): v for k, v in st.to_dict("index").items()}, updated=now), indent=1))
    (ROOT / "data/derived/theme_exposure.parquet.manifest.json").write_text(json.dumps(dict(
        dataset="theme_exposure", producer="src/agentic/build_themes.py", config="configs/themes.json", rows=len(E),
        columns=dict(symbol="NSE symbol", theme="theme id", an_dt="filing time of the matching NSE announcement"),
        source="data/derived/announcements_historical.parquet (desc + attchmntText)", updated=now), indent=1))
    latest = I[I["date"] == I["date"].max()].sort_values("score", ascending=False)
    print(f"\nwrote theme_intensity ({len(I)} rows) and theme_exposure ({len(E)} rows)\nstates by year:\n{st.to_string()}")
    print("\nlatest:", " · ".join(f"{r.theme} {r.state}" for r in latest.itertuples()))


if __name__ == "__main__":
    main()
