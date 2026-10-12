"""Emit reports/freshness_status.md — a standalone freshness dashboard.

Reads verify_freshness.snapshot() and writes a Markdown report with:
  - Overall pass/fail at the top (so it's the first thing you see)
  - Per-input file-level status
  - Per-input column-level breakdown (the check that caught the 2026-07-01 bug)
  - Fix commands next to any stale row

Usage:
  python3 src/agentic/emit_freshness_status.py
  → writes reports/freshness_status.md and exits 0 (this script never fails, unlike verify_freshness which fails hard)
"""
from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
from verify_freshness import snapshot

ROOT = Path("/Users/abhinavs./Code/Zoom")
OUT = ROOT / "reports/freshness_status.md"


FIX_COMMANDS = {
    "PRICES":         "`python3 src/agentic/refresh_prices.py`",
    "MACRO_PANEL":    "`python3 src/agentic/fetch_forex_macro.py && python3 src/agentic/fetch_commodity_prices.py && python3 src/agentic/fetch_global_macro.py && python3 src/agentic/build_macro_panel.py`",
    "INDUSTRY":       "`python3 src/agentic/fetch_industry_indicators.py`",
    "NEWS_EVENTS":    "`python3 src/agentic/build_news_event_features.py`",
    "ANNOUNCEMENTS":  "`python3 src/agentic/refresh_announcements.py`",
    "CS_ENGINE":      "`python3 src/agentic/compare_short_horizons.py`",
    "HC_ENGINE":      "`python3 src/agentic/find_high_conviction.py`",
    "MB_ENGINE":      "`python3 src/agentic/find_multibagger_today.py`",
    "F180_ENGINE":    "`python3 src/agentic/find_180d_frontier_honest.py`",
    "ML_CLASSIFIER":  "`python3 src/agentic/train_missed_winner_classifier.py`",
}


def main() -> None:
    rows = snapshot()
    any_stale = any(r["is_stale"] for r in rows)
    n_files_stale = sum(1 for r in rows if r["file_is_stale"])
    n_cols_stale  = sum(1 for r in rows for c in r["col_results"] if c["is_stale"])
    n_col_checks  = sum(len(r["col_results"]) for r in rows)

    lines = []
    lines.append(f"# Data Freshness Status — {date.today().isoformat()}")
    lines.append(f"_generated {datetime.now().isoformat(timespec='seconds')}_\n")

    if any_stale:
        lines.append("## ❌ STALE — Pipeline BLOCKED\n")
        lines.append(f"- **{n_files_stale}** file-level failure(s) out of {len(rows)} inputs")
        lines.append(f"- **{n_cols_stale}** column-level failure(s) out of {n_col_checks} column checks")
        lines.append(f"- Fix all rows marked ❌ below, then re-run `python3 src/agentic/verify_freshness.py`\n")
    else:
        lines.append("## ✅ ALL FRESH — Pipeline may proceed\n")
        lines.append(f"- All **{len(rows)}** inputs pass file-level contracts")
        lines.append(f"- All **{n_col_checks}** column-level checks pass\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Input | File | Cols passing | Worst column | Fix |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        fstat = "❌" if r["file_is_stale"] else "✅"
        fstr = f"{fstat} {r['file_stale_bd']}/{r['max_stale_bd']}bd" if r["file_stale_bd"] < 9999 else f"{fstat} MISSING"
        if r["col_results"]:
            n_pass = sum(1 for c in r["col_results"] if not c["is_stale"])
            n_total = len(r["col_results"])
            col_cell = f"{n_pass}/{n_total}"
            worst = max(r["col_results"], key=lambda c: (c["is_stale"], c["stale_bd"]))
            worst_stat = "❌" if worst["is_stale"] else "✅"
            worst_cell = f"{worst_stat} `{worst['col']}` {worst['stale_bd']}/{worst['limit']}bd"
        else:
            col_cell = "—"
            worst_cell = "—"
        fix = FIX_COMMANDS.get(r["tag"], "")
        fix_cell = fix if r["is_stale"] else ""
        lines.append(f"| **{r['tag']}** | {fstr} | {col_cell} | {worst_cell} | {fix_cell} |")

    # Full detail per input
    lines.append("\n## Full detail — every check\n")
    for r in rows:
        icon = "❌" if r["is_stale"] else "✅"
        lines.append(f"### {icon} {r['tag']}\n")
        lines.append(f"- **Path**: `{r['path']}`")
        lines.append(f"- **File-level**: max={r['file_max_date']}  ·  stale={r['file_stale_bd']}bd  ·  limit={r['max_stale_bd']}bd  ·  {'❌ FAIL' if r['file_is_stale'] else '✅ OK'}")
        if r["col_results"]:
            lines.append(f"- **Column-level** ({len(r['col_results'])} checks):")
            for cr in r["col_results"]:
                c_icon = "❌" if cr["is_stale"] else "✅"
                if cr.get("err"):
                    lines.append(f"  - {c_icon} `{cr['col']}` — {cr['err']}")
                else:
                    lines.append(f"  - {c_icon} `{cr['col']}` — last non-null={cr['last_date']}  ·  stale={cr['stale_bd']}bd  ·  limit={cr['limit']}bd")
        if r["is_stale"]:
            fix = FIX_COMMANDS.get(r["tag"])
            if fix:
                lines.append(f"- **Fix**: {fix}")
        lines.append("")

    lines.append("---")
    lines.append("_This report is regenerated on every pipeline run. If you see stale entries, run the fix command listed._")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
