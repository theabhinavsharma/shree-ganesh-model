"""Order-book announcement event study (DESCRIPTIVE, not pre-registered; 2026-09-28, follow-up to EXP-2026-09-28-order-backlog).

Question from the user after the backlog test failed: does the announcement itself move the stock, and short of +50%, what
average / median return did these stocks give? The registered test used weekly rows; this looks at each announcement once.

Event   the first order-book statement of a company after >= 45 days without one (a results deck / press release / transcript
        cluster counts once), from data/derived/order_book_filings.parquet (fixed extraction) via test_order_backlog.statements
        (PIT consensus book, ratio = book / PIT TTM revenue with the registered build_ttm). t0 = effective session (filed
        before 15:30 IST on a session -> that session, else the next one).
Prices  adjusted panel (research_panel), session calendar. Closes are carried forward over non-trading sessions.
Windows (all relative to the session calendar)
  pre20      close(t0-1) / close(t0-21) - 1              run-up into the announcement
  ann1       close(t0) / close(t0-1) - 1                  announcement-session move (not tradeable: the filing lands during/after it)
  ann5       close(t0+4) / close(t0-1) - 1                first five sessions incl. t0 (not tradeable)
  peak95     max high over t0+1..t0+95 / close(t0) - 1    best point within 95 sessions after buying at the t0 close
  end95      close(t0+95) / close(t0) - 1                 (frozen at the final close if the stock stopped trading)
  hit50      peak95 >= +50%;  days_to_peak = sessions from t0 to that max high
  peak95_pre max high over t0..t0+95 / close(t0-1) - 1    same, but counting the announcement move (holder before the filing)
Base    every weekly anatomy row with a PIT TTM revenue in the era: peak95 / end95 from its close (the average stock).
Groups  ratio >= 2 (HIGH), 1-2, < 1 (LOW) (statements without a PIT TTM or with ratio > 20 are not events); eras disc 2019-02-22..2022-12-31, conf 2023-01-01..; also > SMA200 at t0-1.
Only events whose 95-session window lies inside the panel are used.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import test_order_backlog as tob  # noqa: E402
from test_hot_order_combo import ROWS, asof_ttm, build_ttm, CUTOFF_MIN  # noqa: E402

OUTDIR = ROOT / "logs/leader_sleeve/order_book_events"
H, GAP_D, DISC0, CONF0 = 95, 45, pd.Timestamp("2019-02-22"), pd.Timestamp("2023-01-01")


def summarize(D: pd.DataFrame, cols: list[str]) -> dict:
    out = dict(n=len(D), n_sym=int(D["symbol"].nunique()))
    for c in cols:
        x = D[c].dropna() * 100
        out[f"{c}_mean"], out[f"{c}_median"] = float(x.mean()), float(x.median())
    out["hit50_pct"] = float(D["hit50"].mean() * 100) if "hit50" in D else np.nan
    if "days_to_peak" in D:
        out["days_to_peak_median"] = float(D["days_to_peak"].median())
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    px = rp.load_panel(["high", "low", "close"])
    cal = rp.session_calendar(px)
    close_w, high_w, low_w = (rp.wide(px, c, cal) for c in ("close", "high", "low"))
    del px
    T, _ = build_ttm(None)
    tob.BOOK_IN = tob.BOOK                                             # fixed-extraction producer output
    E, _ = tob.statements(cal, lambda left, when: asof_ttm(left, when, T))
    # statements() keeps statements with a PIT TTM and ratio in (0, 20]; the event is the first of each >= 45-day cluster
    E = E.sort_values(["symbol", "eff"], kind="mergesort")
    first = E.groupby("symbol")["eff"].diff().dt.days.fillna(1e9) > GAP_D
    E = E[first].reset_index(drop=True)

    cf = close_w.ffill()
    sma200 = cf.rolling(200, min_periods=150).mean()
    C, HI, SM = cf.to_numpy(float), high_w.to_numpy(float), sma200.to_numpy(float)
    CR = close_w.to_numpy(float)
    ci = close_w.columns.get_indexer(E["symbol"])
    ti = cal.get_indexer(E["eff"])
    last_ok = len(cal) - 1 - H
    rows = []
    for k in range(len(E)):
        c, t = ci[k], ti[k]
        if c < 0 or t < 21 or t > last_ok or not np.isfinite(CR[t, c]):
            continue                                                   # no print at t0, or window outside the panel
        p0, pm1, pm21 = C[t, c], C[t - 1, c], C[t - 21, c]
        win = HI[t + 1: t + H + 1, c]
        if not np.isfinite(win).any() or not np.isfinite(pm1):
            continue
        j = int(np.nanargmax(win))
        peak = win[j]
        hi0 = np.nanmax(HI[t: t + H + 1, c])
        rows.append(dict(symbol=E.at[k, "symbol"], eff=E.at[k, "eff"], ratio=E.at[k, "ratio"], book_cr=E.at[k, "book_pit"],
                         rev_ttm_cr=E.at[k, "rev_ttm_cr"],
                         above200=bool(np.isfinite(SM[t - 1, c]) and pm1 > SM[t - 1, c]),
                         pre20=pm1 / pm21 - 1 if np.isfinite(pm21) else np.nan, ann1=p0 / pm1 - 1, ann5=C[t + 4, c] / pm1 - 1,
                         peak95=peak / p0 - 1, end95=C[t + H, c] / p0 - 1, days_to_peak=j + 1, peak95_pre=hi0 / pm1 - 1))
    V = pd.DataFrame(rows)
    V["hit50"] = V["peak95"] >= 0.5
    V["era"] = np.where(V["eff"] >= CONF0, "conf", np.where(V["eff"] >= DISC0, "disc", "pre"))
    V["grp"] = pd.cut(V["ratio"], [0, 1, 2, np.inf], right=False, labels=["LOW <1", "1-2", "HIGH >=2"]).astype(str)
    V = V[V["era"] != "pre"]

    # base: every covered weekly anatomy row (the average stock), same peak / end definitions from its close
    S = pd.read_parquet(ROWS, columns=["symbol", "trade_date"])
    S["trade_date"] = pd.to_datetime(S["trade_date"]).astype("datetime64[ns]")
    S["asof"] = S["trade_date"] + pd.Timedelta(minutes=CUTOFF_MIN - 1)
    S = S[asof_ttm(S, "asof", T).notna()]
    fw = rp.forward_window(close_w, high_w, low_w, H)
    r = cal.get_indexer(S["trade_date"]); c = close_w.columns.get_indexer(S["symbol"])
    ok = (r >= 0) & (c >= 0) & (r <= last_ok)
    r, c, S = r[ok], c[ok], S[ok].copy()
    c0 = CR[r, c]
    S["peak95"] = fw["hi"].to_numpy(float)[r, c] / c0 - 1
    S["end95"] = fw["exit"].to_numpy(float)[r, c] / c0 - 1
    S = S[np.isfinite(c0) & (c0 > 0)]
    S["hit50"] = S["peak95"] >= 0.5
    S["era"] = np.where(S["trade_date"] >= CONF0, "conf", np.where(S["trade_date"] >= DISC0, "disc", "pre"))
    S = S[S["era"] != "pre"]

    cols = ["pre20", "ann1", "ann5", "peak95", "end95", "peak95_pre"]
    out = []
    for e in ("disc", "conf"):
        out.append(dict(era=e, group="BASE: average stock-week", **summarize(S[S["era"] == e], ["peak95", "end95"])))
        Ve = V[V["era"] == e]
        out.append(dict(era=e, group="ALL announcements", **summarize(Ve, cols)))
        for g in ("HIGH >=2", "1-2", "LOW <1"):
            out.append(dict(era=e, group=g, **summarize(Ve[Ve["grp"] == g], cols)))
            out.append(dict(era=e, group=g + " & >200DMA", **summarize(Ve[(Ve["grp"] == g) & Ve["above200"]], cols)))
    R = pd.DataFrame(out)
    V.to_parquet(OUTDIR / "events.parquet", index=False)
    R.to_csv(OUTDIR / "summary.csv", index=False)
    now = datetime.now().isoformat(timespec="seconds")
    units = {c_: "fraction in events.parquet; percent in summary.csv (_mean / _median)" for c_ in cols + ["peak95", "end95"]}
    for f, desc in (("events.parquet", "one row per order-book announcement (first statement after >= 45 days without one)"),
                    ("summary.csv", "mean / median by era x ratio group; BASE = every covered weekly anatomy row")):
        (OUTDIR / (f + ".manifest.json")).write_text(json.dumps(dict(
            dataset=f, path=str((OUTDIR / f).relative_to(ROOT)), producer="src/agentic/study_order_book_events.py",
            status="DESCRIPTIVE - not pre-registered", description=desc, definitions=__doc__, units=units,
            inputs=["data/derived/order_book_filings.parquet", "data/derived/pnl_quarterly.parquet", str(rp.PANEL.relative_to(ROOT)),
                    str(ROWS.relative_to(ROOT))], updated=now), indent=1))
    (OUTDIR / "README.md").write_text(
        "# Order-book announcement event study (descriptive)\n\n"
        "One row per order-book announcement (`events.parquet`) and a mean/median summary by era and backlog group (`summary.csv`).\n"
        "Follow-up to the failed registered test EXP-2026-09-28-order-backlog; not pre-registered. Definitions are in the manifests "
        "and the docstring of src/agentic/study_order_book_events.py.\n")
    pd.set_option("display.width", 250)
    show = ["era", "group", "n", "n_sym", "pre20_median", "ann1_mean", "ann5_mean", "ann5_median", "peak95_mean", "peak95_median",
            "hit50_pct", "end95_mean", "end95_median", "days_to_peak_median"]
    print(R[[c_ for c_ in show if c_ in R]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
