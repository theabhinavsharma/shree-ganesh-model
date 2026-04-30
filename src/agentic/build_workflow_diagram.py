"""Auto-generate a Mermaid workflow diagram from daily_pipeline.sh.

Output:
  reports/WORKFLOW.md   — the diagram + per-step descriptions + file links

Run any time. Reads the pipeline script as source of truth, so the diagram
never drifts.
"""
from __future__ import annotations
from pathlib import Path
import re
import datetime as _dt

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PIPE = ROOT / "src/agentic/daily_pipeline.sh"
OUT = ROOT / "reports/WORKFLOW.md"

# Group/category map for visual coloring
GROUPS: dict[str, str] = {
    # data fetch
    "refresh_prices":           "DATA",
    "refresh_announcements":    "DATA",
    "catalyst_tagger":          "DATA",
    "fetch_block_deals":        "DATA",
    "build_catalyst_features":  "DATA",
    "fetch_news_rss":           "DATA",
    "fetch_news_per_symbol":    "DATA",
    "fetch_reddit":             "DATA",
    "fetch_youtube":            "DATA",
    "fetch_options_chain":      "DATA",
    "fetch_fundamentals":       "DATA",
    "fetch_forex_macro":        "DATA",
    "fetch_fii_dii":            "DATA",
    "fetch_wiki_pageviews":     "DATA",
    "score_sentiment":          "DATA",
    # models
    "run_v3_with_catalysts":    "MODEL",
    "run_short_side":           "MODEL",
    "sector_weak_shorts":       "MODEL",
    "run_multi_horizon":        "MODEL",
    "portfolio_sizer":          "MODEL",
    # discipline / output
    "data_completeness":        "GATE",
    "filter_cascade":           "GATE",
    "paper_trading_recorder":   "GATE",
    "generate_daily_brief":     "OUTPUT",
    "generate_pro_brief":       "OUTPUT",
}

GROUP_COLOR = {
    "DATA":   "#1e3a5f",
    "MODEL":  "#3b1e5f",
    "GATE":   "#5f4f1e",
    "OUTPUT": "#1e5f2c",
}


def parse_pipeline() -> list[tuple[str, str]]:
    """Return list of (label, script_basename) tuples in pipeline order."""
    text = PIPE.read_text()
    steps: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = re.search(r'run_step\s+"([^"]+)"\s+\$PY\s+(?:-m\s+)?([\w./]+)', line)
        if not m:
            continue
        label = m.group(1)
        script = m.group(2)
        # extract the basename of the script (handle module path and file path)
        bn = script.split("/")[-1].replace(".py", "").replace("src.agentic.", "")
        steps.append((label, bn))
    return steps


def render_mermaid(steps: list[tuple[str, str]]) -> str:
    """Bulletproof mermaid: simple ASCII labels, no <br/>, no emoji in nodes."""
    nodes = ["flowchart TD"]
    nodes.append(f"  start([18:00 IST kickoff])")
    last = "start"
    for i, (label, script) in enumerate(steps):
        node_id = f"s{i}"
        # truncate label; replace special chars
        short = label.split("(")[0].strip()
        # strip non-ascii (mermaid in some browsers chokes on emojis in node labels)
        short = "".join(ch for ch in short if ord(ch) < 128)
        if len(short) > 35:
            short = short[:33] + ".."
        nodes.append(f'  {node_id}["{i+1}. {short}"]')
        nodes.append(f"  {last} --> {node_id}")
        last = node_id
    nodes.append(f"  done([dashboard updated])")
    nodes.append(f"  {last} --> done")

    # styling per group
    nodes.append("")
    for i, (_, script) in enumerate(steps):
        group = GROUPS.get(script, "DATA")
        color = GROUP_COLOR[group]
        nodes.append(f"  style s{i} fill:{color},stroke:#fff,color:#fff")
    nodes.append("  style start fill:#000,color:#fff")
    nodes.append("  style done fill:#1e5f2c,color:#fff")
    return "\n".join(nodes)


def render_legend() -> str:
    rows = ["| Color | Stage | Purpose |", "|---|---|---|"]
    desc = {
        "DATA":   "Data ingest — pulls fresh inputs from external sources",
        "MODEL":  "ML — retrains and scores against universe",
        "GATE":   "Discipline — completeness audit, cascade, paper-ledger",
        "OUTPUT": "Brief — human-readable reports + actionable CSVs",
    }
    for g, color in GROUP_COLOR.items():
        rows.append(f"| 🟦{color[1:]} | **{g}** | {desc[g]} |")
    return "\n".join(rows)


def render_step_table(steps: list[tuple[str, str]]) -> str:
    rows = ["| # | Step | Script | Group | What it produces |", "|---|---|---|---|---|"]
    outputs = {
        "refresh_prices":          "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
        "refresh_announcements":   "tmp/from_scratch_7d_run/alt/corp_announcements.parquet",
        "catalyst_tagger":         "announcements_tagged.parquet",
        "fetch_block_deals":       "data/derived/block_deals.parquet, block_features.parquet",
        "build_catalyst_features": "data/derived/catalyst_features.parquet",
        "fetch_news_rss":          "data/derived/news_feed.parquet",
        "fetch_news_per_symbol":   "appends to news_feed.parquet (per-symbol Google News)",
        "fetch_reddit":            "data/derived/reddit_feed.parquet",
        "fetch_youtube":           "data/derived/youtube_videos.parquet",
        "score_sentiment":         "news_features.parquet, macro_sentiment.parquet",
        "fetch_options_chain":     "options_chain_snapshot.parquet (IP-blocked)",
        "fetch_fundamentals":      "data/derived/fundamentals_snapshot.parquet",
        "fetch_forex_macro":       "data/derived/macro_timeseries.parquet",
        "fetch_fii_dii":           "data/derived/fii_dii_flows.parquet",
        "fetch_wiki_pageviews":    "data/derived/wiki_pageviews.parquet",
        "run_v3_with_catalysts":   "v3_live_top100.csv, v3_live_full.csv, v3_oof.parquet",
        "run_short_side":          "short_live_top100.csv, short_live_full.csv",
        "sector_weak_shorts":      "sector_weak_shorts.csv (macro overlay)",
        "run_multi_horizon":       "multi_horizon_top.csv (1d/7d/21d triangulation)",
        "portfolio_sizer":         "portfolio_today.csv",
        "data_completeness":       "reports/data_completeness_*.md, completeness.parquet",
        "filter_cascade":          "actionable_today.csv + filter_cascade_*.md",
        "paper_trading_recorder":  "data/derived/paper_trading_ledger.parquet",
        "generate_daily_brief":    "reports/daily_brief_*.md",
        "generate_pro_brief":      "reports/daily_pro_brief_*.md",
    }
    for i, (label, script) in enumerate(steps, 1):
        group = GROUPS.get(script, "DATA")
        out = outputs.get(script, "—")
        rows.append(f"| {i} | {label} | `{script}` | {group} | `{out}` |")
    return "\n".join(rows)


def main() -> None:
    steps = parse_pipeline()
    doc = [
        f"# Workflow — daily pipeline ({len(steps)} steps)",
        "",
        f"_Auto-generated by `build_workflow_diagram.py` at {_dt.datetime.now():%Y-%m-%d %H:%M}_",
        "",
        "## Diagram",
        "",
        "```mermaid",
        render_mermaid(steps),
        "```",
        "",
        "## Legend",
        "",
        render_legend(),
        "",
        "## Per-step inventory",
        "",
        render_step_table(steps),
        "",
        "## Workflow control plane",
        "",
        "- Kickoff: macOS LaunchAgent `com.zoom.daily-pipeline` at 18:00 IST",
        "  → runs [`src/agentic/daily_pipeline.sh`](../src/agentic/daily_pipeline.sh)",
        "- Status: [`status.md`](status.md) regenerates after every run",
        "- Logs: `logs/daily_pipeline_<YYYYMMDD_HHMM>.log`",
        "- Brief: `reports/daily_pro_brief_<YYYYMMDD>.md`",
        "- This file: `reports/WORKFLOW.md` (re-render with `python src/agentic/build_workflow_diagram.py`)",
        "",
        "## Background processes (not part of the daily pipeline)",
        "",
        "These run on demand or as long-running jobs:",
        "",
        "- `factor_evaluator.py` — measures lift of new factors against OOS forward returns",
        "- `feature_factory.py` — compiles WorldQuant-style alphas + macro-conditional features",
        "- `factor_registry.py` — the hypothesis catalog with KEEP/DROP verdicts",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(doc))
    print(f"wrote {OUT}")
    print(f"  parsed {len(steps)} pipeline steps")


if __name__ == "__main__":
    main()
