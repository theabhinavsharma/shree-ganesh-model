"""Recreation-kit generator — keeps SHOWCASE.html's inventory CURRENT, never historical.

Scans the repo for every script, agent prompt, and orchestrator we actually use,
extracts each one's one-line purpose (module docstring / heading / header comment),
and injects the generated inventory INSIDE the magic-prompt block of SHOWCASE.html
between the markers:

    ═══ CURRENT INVENTORY (auto-generated) ═══
    ═══ END INVENTORY ═══

Because the inventory lives inside <pre id="magic-prompt-text">, the existing
one-click "Copy Prompt" button copies the recreation instructions AND the
complete, current file inventory in a single click.

Also writes assets/recreation_manifest.json (machine-readable) for tooling.

Run from run_weekly_pipeline.sh on every pipeline run — the showcase inventory
is regenerated weekly by construction, so it cannot drift into a historical doc.

Stdlib-only, per the simplicity policy (see simplicity_auditor.py).
"""
from __future__ import annotations
import ast
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
SHOWCASE = ROOT / "SHOWCASE.html"
MANIFEST = ROOT / "assets/recreation_manifest.json"

BEGIN = "═══ CURRENT INVENTORY (auto-generated) ═══"
END = "═══ END INVENTORY ═══"

CATEGORIES = [
    ("Data fetchers",      re.compile(r"(fetch_|refresh_|build_news|build_macro)")),
    ("ML engines",         re.compile(r"(find_180d|find_high_conviction|find_multibagger|run_multi_horizon|compare_short_horizons|find_achievable)")),
    ("Backtests",          re.compile(r"backtest_")),
    ("QC / gates",         re.compile(r"(verify_freshness|devils_advocate|filter_cascade|data_completeness|emit_freshness)")),
    ("RL loop",            re.compile(r"(miss_learner|train_missed_winner)")),
    ("Basket / output",    re.compile(r"(generate_hybrid_basket|generate_daily_brief|build_status_dashboard)")),
    ("Simplicity layer",   re.compile(r"(simplicity_auditor|build_recreation_kit)")),
    ("Orchestrators",      re.compile(r"(run_weekly_pipeline|daily_pipeline|daily_refresh)")),
    ("Agent runner",       re.compile(r"run_agent")),
]


def one_liner_py(p: Path) -> str:
    try:
        tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        doc = ast.get_docstring(tree)
        if doc:
            return doc.strip().splitlines()[0][:110]
    except SyntaxError:
        pass
    return ""


def one_liner_sh(p: Path) -> str:
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[1:6]:
        line = line.strip()
        if line.startswith("#") and len(line) > 3:
            return line.lstrip("# ").strip()[:110]
    return ""


def one_liner_md(p: Path) -> str:
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:10]:
        if line.startswith("#"):
            return line.lstrip("# ").strip()[:110]
    return ""


def categorize(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "Other tools"


def collect() -> dict:
    inv = {}
    # Python + shell in src/agentic
    for p in sorted((ROOT / "src/agentic").glob("*.py")) + sorted((ROOT / "src/agentic").glob("*.sh")):
        purpose = one_liner_py(p) if p.suffix == ".py" else one_liner_sh(p)
        loc = p.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        cat = categorize(p.name)
        inv.setdefault(cat, []).append({"path": f"src/agentic/{p.name}", "loc": loc, "purpose": purpose})
    # Agent prompts
    for p in sorted((ROOT / "prompts").glob("*.md")):
        inv.setdefault("Agent prompts", []).append(
            {"path": f"prompts/{p.name}", "loc": p.read_text().count("\n") + 1, "purpose": one_liner_md(p)})
    return inv


def render(inv: dict) -> str:
    n_files = sum(len(v) for v in inv.values())
    n_loc = sum(f["loc"] for v in inv.values() for f in v)
    lines = [BEGIN,
             f"As of {date.today().isoformat()} · {n_files} files · {n_loc:,} LOC · regenerated on every",
             "weekly pipeline run by src/agentic/build_recreation_kit.py — this list IS current.",
             ""]
    order = [c for c, _ in CATEGORIES] + ["Agent prompts", "Other tools"]
    for cat in order:
        files = inv.get(cat, [])
        if not files:
            continue
        lines.append(f"[{cat}]  ({len(files)})")
        for f in files:
            purpose = f" — {f['purpose']}" if f["purpose"] else ""
            lines.append(f"  {f['path']:<52s}{purpose}")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def inject(block: str) -> bool:
    html = SHOWCASE.read_text(encoding="utf-8")
    if BEGIN in html and END in html:
        pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
        html = pattern.sub(block.replace("\\", r"\\"), html, count=1)
        SHOWCASE.write_text(html, encoding="utf-8")
        return True
    # First run: insert just before the closing </code></pre> of the magic prompt
    anchor = "Reference URL: https://github.com/theabhinavsharma/shree-ganesh-model"
    if anchor in html:
        html = html.replace(anchor, block + "\n\n" + anchor, 1)
        SHOWCASE.write_text(html, encoding="utf-8")
        return True
    return False


def main() -> None:
    inv = collect()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"generated": date.today().isoformat(), "inventory": inv}, indent=1))
    block = render(inv)
    ok = inject(block)
    n = sum(len(v) for v in inv.values())
    print(f"manifest: {MANIFEST.relative_to(ROOT)} ({n} files)")
    print(f"showcase inventory {'INJECTED' if ok else 'FAILED — anchor not found'}")


if __name__ == "__main__":
    main()
