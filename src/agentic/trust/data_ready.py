"""Data-readiness gate for registered A/B tests (2026-10-02). Garbage in, garbage out: no result on data below the bar.

Abhinav (2026-10-01): "A/B mein GIGO prevent karna hai — don't run the test if data is lacking or not up to the mark."
Every registered test calls gate() BEFORE computing any outcome. Each check compares the data a test uses with a bar
from configs/data_bar.json (a REGISTERED line may set a stricter or looser bar; that is visible in the log):
  completeness  rows we hold per year vs the official count (e.g. NSE's results calendar); an official count that is
                missing for a year is a FAIL (cannot verify = not ready)
  coverage      share of rows with the feature filled, per year, for every feature an arm depends on
  span          the input's dates reach the start and end of the test window
  qc            a quality number (e.g. share of EPS values that match profit / shares)
All checks pass -> "<EXP>-DATA-READY" (status READY) is logged and the test runs. Any check fails -> "<EXP>-DATA-READY"
(status NOT READY, with the failing checks) is logged and the test stops with exit code 3: no RESULT line, no verdict.
The eval trust.results_data_ready fails if a RESULT / RERUN line has no READY line before it on the same day.
SGM_REPRO=1 (reproduce.py) runs the gate but logs nothing.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BAR = json.loads((ROOT / "configs/data_bar.json").read_text())


def completeness(name: str, have: pd.Series, official: pd.Series, years, min_share: float | None = None) -> dict:
    m = min_share if min_share is not None else BAR["completeness_min"]
    rows, bad = {}, []
    for y in years:
        o, h = float(official.get(y, 0) or 0), float(have.get(y, 0) or 0)
        s = h / o if o > 0 else None
        rows[int(y)] = None if s is None else round(s, 3)
        if s is None or s < m:
            bad.append(f"{y}: " + ("official count missing" if s is None else f"{s:.0%}"))
    return dict(check="completeness", name=name, bar=m, by_year=rows, ok=not bad, fails=bad)


def coverage(name: str, df: pd.DataFrame, date_col: str, cols: list[str], years, min_share: float | None = None) -> dict:
    m = min_share if min_share is not None else BAR["coverage_min"]
    yr = pd.to_datetime(df[date_col]).dt.year
    rows, bad = {}, []
    for c in cols:
        s = df[c].notna().groupby(yr).mean()
        rows[c] = {int(y): round(float(s.get(y, 0)), 3) for y in years}
        bad += [f"{c} {y}: {v:.0%}" for y, v in rows[c].items() if v < m]
    return dict(check="coverage", name=name, bar=m, by_feature=rows, ok=not bad, fails=bad)


def span(name: str, dates: pd.Series, start, end) -> dict:
    d = pd.to_datetime(dates)
    lo, hi = d.min(), d.max()
    tol = pd.Timedelta(days=BAR["span_tolerance_days"])
    bad = ([f"starts {lo.date()} after {pd.Timestamp(start).date()}"] if lo > pd.Timestamp(start) + tol else []) + \
          ([f"ends {hi.date()} before {pd.Timestamp(end).date()}"] if hi < pd.Timestamp(end) - tol else [])
    return dict(check="span", name=name, first=str(lo.date()), last=str(hi.date()), ok=not bad, fails=bad)


def qc(name: str, value: float, min_value: float) -> dict:
    return dict(check="qc", name=name, value=round(float(value), 4), bar=min_value, ok=value >= min_value,
                fails=[] if value >= min_value else [f"{value:.1%} < {min_value:.0%}"])


def bhav_rows() -> pd.Series:
    """Official stock-day count per session (2026-10-08): EQ/BE/BZ rows in NSE's own daily bhavcopy files
    (data/raw/nse_full_history_official/trade_date=*/sec_bhavdata_full_*.csv; before 2020 cm*bhav.csv.zip), the
    series the price panel keeps. Each file counts under the trade date written inside it: NSE also publishes a file on
    trading holidays that repeats the previous session (folder 2020-03-10, Holi, holds 09-Mar-2020), so a folder date is
    not proof of a session. Counted once per folder, cached in logs/evals/bhav_rows_by_date.parquet."""
    cache = ROOT / "logs/evals/bhav_rows_by_date.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["folder", "session", "rows"])
    seen, new = set(pd.to_datetime(have["folder"])), []
    for day in sorted((ROOT / "data/raw/nse_full_history_official").glob("trade_date=*")):
        d = pd.Timestamp(day.name.split("=", 1)[1])
        f = next(iter(sorted(day.glob("sec_bhavdata_full_*.csv"))), None) or next(iter(sorted(day.glob("cm*bhav.csv.zip"))), None)
        if d in seen or f is None:
            continue
        want = lambda c: str(c).strip() in ("SERIES", "DATE1", "TIMESTAMP")  # noqa: E731
        if f.suffix == ".zip":   # before 2020: the classic cm bhavcopy
            r = pd.read_csv(f, usecols=want)
        elif f.open("rb").read(2) == b"PK":   # NSE served 2022-08-08 as an Excel workbook under the .csv name
            r = pd.read_excel(f); r = r[[c for c in r.columns if want(c)]]
        else:
            r = pd.read_csv(f, usecols=want)
        r.columns = [str(c).strip() for c in r.columns]
        dt = pd.to_datetime(r["DATE1" if "DATE1" in r else "TIMESTAMP"].astype(str).str.strip(), format="%d-%b-%Y", errors="coerce")
        sess = dt.mode().iat[0] if dt.notna().any() else pd.NaT
        new.append(dict(folder=d, session=sess, rows=int(r["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ"]).sum())))
    if new:
        have = pd.concat([x for x in (have, pd.DataFrame(new)) if len(x)], ignore_index=True)
        have.to_parquet(cache)
        cache.with_name(cache.name + ".manifest.json").write_text(json.dumps(dict(
            dataset=cache.name, producer="src/agentic/trust/data_ready.py bhav_rows()", source="NSE daily bhavcopy files in data/raw/nse_full_history_official",
            columns=dict(folder="raw folder date (trade_date=...)", session="trade date written inside the file (DATE1 / TIMESTAMP)",
                         rows="count of EQ, BE and BZ series rows in that file"),
            note="a holiday folder repeats the previous session; bhav_rows() keeps one file per session",
            folders=len(have), updated=datetime.now().isoformat(timespec="seconds")), indent=1, default=str))
    h = have.dropna(subset=["session"]).sort_values("folder").drop_duplicates("session")
    return pd.Series(h["rows"].astype(int).to_numpy(), index=pd.DatetimeIndex(h["session"]), name="rows").sort_index()


def gate(exp_id: str, checks: list[dict], run: str = "RESULT") -> None:
    ok = all(c["ok"] for c in checks)
    print(f"\n=== DATA READINESS · {exp_id} ({run}) · {'READY' if ok else 'NOT READY'} ===")
    for c in checks:
        print(f"  {'✅' if c['ok'] else '❌'} {c['check']:12s} {c['name']}" + ("" if c["ok"] else " · " + "; ".join(c["fails"][:8])))
    if os.environ.get("SGM_REPRO") != "1":
        with (ROOT / "logs/experiments.jsonl").open("a") as fh:
            fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=f"{exp_id}-DATA-READY", run=run,
                                     status="READY" if ok else "NOT READY", checks=checks), default=str) + "\n")
    if not ok:
        raise SystemExit(3)
