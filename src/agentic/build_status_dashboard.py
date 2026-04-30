"""Live status dashboard — what's the system doing RIGHT NOW.

Output: reports/status.md (regenerated every pipeline run + on demand)

Includes:
  • Last pipeline run timestamp + which steps succeeded / failed
  • Freshness per data source (latest trade_date vs today)
  • Universe + completeness coverage % per group
  • Today's macro state + actionable list count
  • Top picks + filter cascade verdict
  • Background jobs known to be running
  • What's blocking / what's auto-resolving
"""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import datetime as _dt
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "reports/status.md"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
LOGS_DIR = ROOT / "logs"


def _file_age(p: Path) -> str:
    if not p.exists():
        return "—"
    mt = _dt.datetime.fromtimestamp(p.stat().st_mtime)
    delta = _dt.datetime.now() - mt
    if delta.total_seconds() < 60:
        return f"{int(delta.total_seconds())}s ago"
    if delta.total_seconds() < 3600:
        return f"{int(delta.total_seconds()/60)}m ago"
    if delta.days == 0:
        return f"{int(delta.total_seconds()/3600)}h ago"
    return f"{delta.days}d ago"


def _data_freshness() -> list[dict]:
    sources = [
        ("Prices",         "data/derived/stock_daily_facts_adjusted_2015plus.parquet", "trade_date"),
        ("Catalysts",      "data/derived/catalyst_features.parquet",                   "trade_date"),
        ("Fundamentals",   "data/derived/fundamentals_snapshot.parquet",               "fetch_date"),
        ("News (raw)",     "data/derived/news_feed.parquet",                           "pub_ts"),
        ("Reddit",         "data/derived/reddit_feed.parquet",                         "created_utc"),
        ("YouTube",        "data/derived/youtube_videos.parquet",                      "published_ts"),
        ("News (per-sym)", "data/derived/news_features.parquet",                       "as_of"),
        ("Macro sent.",    "data/derived/macro_sentiment.parquet",                     "as_of"),
        ("FX (USDINR)",    "data/derived/macro_timeseries.parquet",                    "trade_date"),
        ("FII/DII",        "data/derived/fii_dii_flows.parquet",                       "trade_date"),
        ("Wiki views",     "data/derived/wiki_pageviews.parquet",                      "trade_date"),
        ("Block deals",    "data/derived/block_features.parquet",                      "trade_date"),
        ("Options",        "data/derived/options_chain_snapshot.parquet",              "trade_date"),
        ("Paper ledger",   "data/derived/paper_trading_ledger.parquet",                "snapshot_date"),
        ("Completeness",   "data/derived/completeness.parquet",                        "audit_date"),
    ]
    rows = []
    for name, rel, date_col in sources:
        p = ROOT / rel
        if not p.exists():
            rows.append({"source": name, "exists": "❌", "rows": "—", "latest": "—", "age": "—"})
            continue
        try:
            df = pd.read_parquet(p)
            n = len(df)
            if date_col in df.columns:
                col = df[date_col]
                if pd.api.types.is_numeric_dtype(col):
                    latest = pd.to_datetime(col.max(), unit="s", errors="coerce")
                else:
                    latest = pd.to_datetime(col, errors="coerce").max()
                latest_str = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "—"
            else:
                latest_str = "—"
            rows.append({"source": name, "exists": "✓", "rows": f"{n:,}",
                         "latest": latest_str, "age": _file_age(p)})
        except Exception as e:
            rows.append({"source": name, "exists": "⚠️", "rows": "?", "latest": str(e)[:30], "age": _file_age(p)})
    return rows


def _completeness_summary() -> dict:
    p = ROOT / "data/derived/completeness.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df["audit_date"] = pd.to_datetime(df["audit_date"]).dt.date
    latest = df["audit_date"].max()
    sub = df[df["audit_date"] == latest]
    by_group = sub.groupby("group").agg(avg_cov=("coverage", "mean"),
                                         n=("param", "size")).reset_index()
    return {"audit_date": str(latest), "by_group": by_group.to_dict(orient="records")}


def _last_pipeline_log() -> dict:
    if not LOGS_DIR.exists():
        return {}
    logs = sorted(LOGS_DIR.glob("daily_pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return {}
    last = logs[0]
    text = last.read_text(errors="replace")
    starts = text.count(">>> START:")
    oks = text.count("<<< OK:")
    fails = text.count("<<< FAIL")
    started_at = _dt.datetime.fromtimestamp(last.stat().st_mtime - 1)  # rough
    return {"file": str(last.relative_to(ROOT)), "started": starts, "ok": oks,
            "fail": fails, "age": _file_age(last)}


def _factor_registry_summary() -> dict:
    p = ROOT / "data/derived/factor_registry.json"
    if not p.exists():
        return {}
    reg = json.loads(p.read_text())
    by_state = {}
    for h in reg:
        by_state[h["state"]] = by_state.get(h["state"], 0) + 1
    keep = [h for h in reg if h["state"] == "KEEP"]
    return {"total": len(reg), "by_state": by_state, "keep_factors": [h["id"] for h in keep]}


def _running_jobs() -> list[str]:
    """Best-effort: scan tasks output dir for files modified in last 30 min."""
    tdir = Path("/private/tmp/claude-501")
    if not tdir.exists():
        return []
    out = []
    cutoff = _dt.datetime.now().timestamp() - 1800
    for f in tdir.glob("**/tasks/*.output"):
        try:
            if f.stat().st_mtime > cutoff:
                out.append(f.name.replace(".output", ""))
        except Exception:
            pass
    return out


def _macro_today() -> dict:
    p = ROOT / "data/derived/macro_sentiment.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df["as_of"] = pd.to_datetime(df["as_of"]).dt.date
    return df.sort_values("as_of").iloc[-1].to_dict()


def _actionable_today() -> dict:
    p = ROOT / "tmp/from_scratch_7d_run/actionable_today.csv"
    if not p.exists():
        return {"n": 0, "names": []}
    df = pd.read_csv(p)
    return {"n": len(df), "names": df["symbol"].astype(str).tolist() if "symbol" in df.columns else []}


def main() -> None:
    print("== build_status_dashboard ==")
    now = _dt.datetime.now()
    sections: list[str] = []
    sections.append(f"# System Status — {now:%Y-%m-%d %H:%M %Z}")
    sections.append("")
    sections.append(f"_Auto-generated. Re-run: `python src/agentic/build_status_dashboard.py`_")
    sections.append("")

    # 1. last pipeline run
    sections.append("## Last pipeline run")
    plog = _last_pipeline_log()
    if plog:
        sections.append(f"- File: `{plog['file']}`  ({plog['age']})")
        sections.append(f"- Started: **{plog['started']} steps**, OK: **{plog['ok']}**, FAIL: **{plog['fail']}**")
        if plog["fail"] > 0:
            sections.append(f"- ⚠️ Some steps failed — see log for details")
    else:
        sections.append("- No pipeline log found yet. Run `bash src/agentic/daily_pipeline.sh`.")
    sections.append("")

    # 2. background jobs
    sections.append("## Background jobs (recent activity)")
    rj = _running_jobs()
    if rj:
        for j in rj[:10]:
            sections.append(f"- `{j}` — output file modified within last 30 min")
    else:
        sections.append("- _none active_")
    sections.append("")

    # 3. data freshness
    sections.append("## Data sources")
    sections.append("")
    sections.append("| Source | Exists | Rows | Latest | File age |")
    sections.append("|---|:---:|---:|---|---|")
    for r in _data_freshness():
        sections.append(f"| {r['source']} | {r['exists']} | {r['rows']} | {r['latest']} | {r['age']} |")
    sections.append("")

    # 4. completeness
    sections.append("## Parameter completeness (today)")
    cs = _completeness_summary()
    if cs:
        sections.append(f"_Audit: {cs['audit_date']}_")
        sections.append("")
        sections.append("| Group | Avg coverage | # params |")
        sections.append("|---|---:|---:|")
        for r in cs["by_group"]:
            cov = r["avg_cov"]
            flag = " ⚠️" if cov < 0.5 else (" ✓" if cov >= 0.95 else "")
            sections.append(f"| {r['group']} | {cov*100:.1f}% | {r['n']}{flag} |")
    else:
        sections.append("_No completeness audit yet — run `python src/agentic/data_completeness.py`._")
    sections.append("")

    # 5. macro state
    sections.append("## Macro state (today)")
    mt = _macro_today()
    if mt:
        gms = mt.get("global_macro_sent", 0) or 0
        dms = mt.get("domestic_macro_sent", 0) or 0
        overall = (gms + dms) / 2
        regime = "🔴 RISK_OFF" if overall <= -0.3 else ("🟢 RISK_ON" if overall >= 0.3 else "◯ NEUTRAL")
        sections.append(f"- Global: {gms:+.2f}  •  Domestic: {dms:+.2f}  •  Overall: **{regime}**")
        sections.append(f"- USDINR sent: {mt.get('usdinr_sentiment', 0):+.2f}  •  Oil: {mt.get('oil_sentiment', 0):+.2f}")
        sections.append(f"- Hawkish/dovish rates: {mt.get('rate_hawkish_score', 0)} / {mt.get('rate_dovish_score', 0)}")
    sections.append("")

    # 6. actionable
    sections.append("## Actionable picks (filter cascade output)")
    at = _actionable_today()
    if at["n"] == 0:
        sections.append("- ⚠️ **0 names cleared all gates today** — park in cash.")
    else:
        sections.append(f"- ✓ **{at['n']} names** cleared all gates: {', '.join(at['names'])}")
    sections.append("")

    # 7. factor registry
    sections.append("## Factor / hypothesis registry")
    fr = _factor_registry_summary()
    if fr:
        sections.append(f"- Total hypotheses: **{fr['total']}**")
        for state, n in fr["by_state"].items():
            sections.append(f"  - {state}: {n}")
        if fr["keep_factors"]:
            sections.append(f"- KEEP factors: {', '.join(fr['keep_factors'])}")
    sections.append("")

    # 8. links
    sections.append("## Quick links")
    sections.append("")
    sections.append("- **Workflow diagram**: [`reports/WORKFLOW.md`](WORKFLOW.md)")
    sections.append("- **Today's brief**: latest `reports/daily_pro_brief_*.md`")
    sections.append("- **Filter cascade**: latest `reports/filter_cascade_*.md`")
    sections.append("- **Completeness audit**: latest `reports/data_completeness_*.md`")
    sections.append("- **Factor evaluation**: [`reports/factor_evaluation.md`](factor_evaluation.md)")
    sections.append("")

    OUT.write_text("\n".join(sections))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
