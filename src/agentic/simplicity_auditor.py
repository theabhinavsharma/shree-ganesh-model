"""Simplicity auditor — stdlib-first policy enforcement + technical-debt ledger.

Three jobs:
  1. AUDIT   — scan the codebase for over-engineering, dead code, unnecessary
               abstractions, and third-party imports where stdlib suffices.
  2. DEBT    — append/report deliberate shortcuts to logs/debt_ledger.jsonl,
               each with its measured impact on code size and speed.
  3. METRICS — snapshot repo size (files/LOC) per run into
               logs/simplicity_metrics.jsonl so size drift is measurable.

Deliberately stdlib-only (ast, argparse, json, pathlib, hashlib, time, re)
— this file practices the policy it enforces.

Usage:
  python3 src/agentic/simplicity_auditor.py audit
      → prints findings, writes reports/simplicity_audit.md
        and appends a metrics row to logs/simplicity_metrics.jsonl

  python3 src/agentic/simplicity_auditor.py debt-add \
      --where src/agentic/foo.py --shortcut "hardcoded 45d window" \
      --why "ship weekly basket today" --planned-fix "config param" \
      --loc-impact -12 --speed-impact "none"
      → appends one entry to logs/debt_ledger.jsonl

  python3 src/agentic/simplicity_auditor.py debt-report
      → prints the open debt ledger with totals

Audit rules (each maps to one finding type):
  DEAD_FUNC       top-level function defined but never referenced anywhere else
  DEAD_CLASS      class defined but never referenced anywhere else
  UNUSED_IMPORT   imported name never used in the file
  ONE_METHOD_CLS  class with a single public method and no state → function
  TRIVIAL_WRAPPER function whose body is a single call to another function
  DUP_FUNC        identical (normalised) function body in 2+ places
  DEP_OVER_STDLIB third-party import with a known stdlib equivalent
  GOD_FILE        file > 800 LOC (split candidate)
  DEEP_NESTING    function with nesting depth > 5 (simplify candidate)
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
SCAN_DIRS = ["src"]
EXCLUDE_PARTS = {"__pycache__", ".git", "node_modules", "tmp"}
AUDIT_MD = ROOT / "reports/simplicity_audit.md"
DEBT_LEDGER = ROOT / "logs/debt_ledger.jsonl"
METRICS_LOG = ROOT / "logs/simplicity_metrics.jsonl"

# Third-party packages with a stdlib equivalent good enough for our use.
# We only FLAG — the audit never auto-rewrites.
DEP_OVER_STDLIB = {
    "requests":        "urllib.request (already used by every fetcher in this repo)",
    "click":           "argparse",
    "typer":           "argparse",
    "python-dotenv":   "os.environ + a 5-line loader",
    "dotenv":          "os.environ + a 5-line loader",
    "pytz":            "zoneinfo (3.9+)",
    "dateutil":        "datetime + zoneinfo for our fixed-format dates",
    "six":             "nothing — py2 compat is dead",
    "retrying":        "a 6-line for-loop with time.sleep",
    "tenacity":        "a 6-line for-loop with time.sleep",
    "loguru":          "logging",
    "tqdm":            "print(f'{i}/{n}') every k iterations",
}
# Heavy deps that are LEGITIMATE here (data work) — never flagged:
ALLOWED_HEAVY = {"pandas", "numpy", "pyarrow", "lightgbm", "sklearn", "scipy", "shap"}


def iter_py_files() -> list[Path]:
    out = []
    for d in SCAN_DIRS:
        for p in (ROOT / d).rglob("*.py"):
            if not any(part in EXCLUDE_PARTS for part in p.parts):
                out.append(p)
    return sorted(out)


def parse(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None


def normalised_body_hash(fn: ast.FunctionDef) -> str:
    """Hash a function body with names/docstrings stripped → duplicate detection."""
    body = fn.body[1:] if (fn.body and isinstance(fn.body[0], ast.Expr)
                           and isinstance(fn.body[0].value, ast.Constant)) else fn.body
    dump = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
    dump = re.sub(r"'[^']*'", "'_'", dump)  # strip literals/names to shape-only
    return hashlib.md5(dump.encode()).hexdigest()


def nesting_depth(node, depth=0) -> int:
    worst = depth
    for child in ast.iter_child_nodes(node):
        d = depth + (1 if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)) else 0)
        worst = max(worst, nesting_depth(child, d))
    return worst


def audit() -> dict:
    files = iter_py_files()
    findings = defaultdict(list)   # type -> [(file, name, detail)]
    total_loc = 0

    # Pass 1: collect every defined top-level symbol and every referenced name, per file and globally
    defs = {}          # (file, name) -> node type
    file_texts = {}
    global_refs = defaultdict(int)   # name -> count of files whose TEXT mentions it
    body_hashes = defaultdict(list)  # hash -> [(file, func_name, loc)]

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        file_texts[f] = text
        total_loc += text.count("\n") + 1
        tree = parse(f)
        if tree is None:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[(f, node.name)] = "func"
                if len(node.body) >= 3:  # skip trivial defs in dup detection
                    loc = (node.end_lineno or node.lineno) - node.lineno + 1
                    body_hashes[normalised_body_hash(node)].append((f, node.name, loc))
            elif isinstance(node, ast.ClassDef):
                defs[(f, node.name)] = "class"

    for f, text in file_texts.items():
        names = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", text))
        for n in names:
            global_refs[n] += 1

    # Pass 2: per-file checks
    for f in files:
        tree = parse(f)
        if tree is None:
            continue
        text = file_texts[f]
        rel = str(f.relative_to(ROOT))
        loc = text.count("\n") + 1

        if loc > 800:
            findings["GOD_FILE"].append((rel, "", f"{loc} LOC"))

        # imports
        imported = {}  # local name -> module
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imported[(a.asname or a.name).split(".")[0]] = a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    if a.name != "*":
                        imported[a.asname or a.name] = node.module.split(".")[0]
        body_text = text
        for local, module in imported.items():
            if module == "__future__":
                continue  # compiler directive, not a runtime name
            # unused import: local name appears only on its import line
            uses = len(re.findall(rf"\b{re.escape(local)}\b", body_text))
            if uses <= 1:
                findings["UNUSED_IMPORT"].append((rel, local, f"from {module}"))
            if module in DEP_OVER_STDLIB and module not in ALLOWED_HEAVY:
                findings["DEP_OVER_STDLIB"].append((rel, module, DEP_OVER_STDLIB[module]))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # dead function: name appears in only 1 file (its own) and <2 times overall there
                if not node.name.startswith("_") and node.name != "main":
                    own_uses = len(re.findall(rf"\b{re.escape(node.name)}\b", text))
                    if global_refs[node.name] <= 1 and own_uses <= 1:
                        findings["DEAD_FUNC"].append((rel, node.name, f"line {node.lineno}"))
                # trivial wrapper
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Return)
                        and isinstance(node.body[0].value, ast.Call)):
                    findings["TRIVIAL_WRAPPER"].append((rel, node.name, f"line {node.lineno}"))
                if nesting_depth(node) > 5:
                    findings["DEEP_NESTING"].append((rel, node.name, f"depth>{5}"))
            elif isinstance(node, ast.ClassDef):
                # dead only if unreferenced in OTHER files AND used at most once in its own
                own_uses = len(re.findall(rf"\b{re.escape(node.name)}\b", text))
                if global_refs[node.name] <= 1 and own_uses <= 1:
                    findings["DEAD_CLASS"].append((rel, node.name, f"line {node.lineno}"))
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                public = [m for m in methods if not m.name.startswith("_")]
                has_state = any(isinstance(n, (ast.Assign, ast.AnnAssign)) for n in node.body) \
                    or any(m.name == "__init__" for m in methods)
                if len(public) == 1 and not has_state and len(methods) <= 2:
                    findings["ONE_METHOD_CLS"].append((rel, node.name, f"→ function {public[0].name}()"))

    # duplicates
    for h, sites in body_hashes.items():
        if len(sites) >= 2:
            locs = ", ".join(f"{s[0].relative_to(ROOT)}:{s[1]}" for s in sites[:4])
            findings["DUP_FUNC"].append((str(len(sites)) + " copies", sites[0][1], locs))

    return {"files": len(files), "loc": total_loc, "findings": dict(findings)}


def write_report(result: dict) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    n_findings = sum(len(v) for v in result["findings"].values())
    lines = [
        f"# Simplicity Audit — {ts}",
        "",
        f"**Scanned**: {result['files']} files · {result['loc']:,} LOC · **{n_findings} findings**",
        "",
        "Policy: stdlib-first, simplest correct solution, no speculative abstraction.",
        "Findings are candidates for deletion/simplification — audit never auto-rewrites.",
        "",
    ]
    order = ["DEAD_FUNC", "DEAD_CLASS", "UNUSED_IMPORT", "ONE_METHOD_CLS",
             "TRIVIAL_WRAPPER", "DUP_FUNC", "DEP_OVER_STDLIB", "GOD_FILE", "DEEP_NESTING"]
    desc = {
        "DEAD_FUNC": "Dead functions (defined, never referenced anywhere)",
        "DEAD_CLASS": "Dead classes",
        "UNUSED_IMPORT": "Unused imports",
        "ONE_METHOD_CLS": "Single-method stateless classes (should be functions)",
        "TRIVIAL_WRAPPER": "Trivial wrappers (single-call bodies)",
        "DUP_FUNC": "Duplicated function bodies (shape-identical)",
        "DEP_OVER_STDLIB": "Third-party deps with a stdlib equivalent",
        "GOD_FILE": "Files > 800 LOC (split candidates)",
        "DEEP_NESTING": "Functions nested > 5 deep",
    }
    for key in order:
        items = result["findings"].get(key, [])
        lines.append(f"## {desc[key]} — {len(items)}")
        lines.append("")
        for a, b, c in items[:60]:
            lines.append(f"- `{a}` **{b}** — {c}")
        if len(items) > 60:
            lines.append(f"- … and {len(items)-60} more")
        lines.append("")

    # Debt ledger summary at the bottom
    if DEBT_LEDGER.exists():
        entries = [json.loads(l) for l in DEBT_LEDGER.read_text().splitlines() if l.strip()]
        open_e = [e for e in entries if not e.get("repaid")]
        lines.append(f"## Debt ledger — {len(open_e)} open / {len(entries)} total")
        lines.append("")
        for e in open_e[-20:]:
            lines.append(f"- {e['ts'][:10]} `{e['where']}` — {e['shortcut']} "
                         f"(why: {e['why']}; loc {e.get('loc_impact','?')}; speed {e.get('speed_impact','?')})")
    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(lines))
    print(f"wrote {AUDIT_MD.relative_to(ROOT)}")


def append_metrics(result: dict, elapsed_s: float) -> None:
    METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "files": result["files"],
        "loc": result["loc"],
        "findings": {k: len(v) for k, v in result["findings"].items()},
        "audit_seconds": round(elapsed_s, 2),
    }
    # Delta vs previous run
    if METRICS_LOG.exists():
        prev_lines = METRICS_LOG.read_text().splitlines()
        if prev_lines:
            prev = json.loads(prev_lines[-1])
            row["loc_delta"] = result["loc"] - prev["loc"]
            row["files_delta"] = result["files"] - prev["files"]
    with METRICS_LOG.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    d = row.get("loc_delta")
    delta = f"  Δloc={d:+,}" if d is not None else ""
    print(f"metrics: {result['files']} files, {result['loc']:,} LOC{delta}  ({row['audit_seconds']}s)")


def debt_add(args) -> None:
    DEBT_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "where": args.where,
        "shortcut": args.shortcut,
        "why": args.why,
        "planned_fix": args.planned_fix,
        "loc_impact": args.loc_impact,     # negative = shortcut SAVED lines
        "speed_impact": args.speed_impact, # free text: "none", "+40s/run", ...
        "repaid": False,
    }
    with DEBT_LEDGER.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    print(f"debt recorded: {args.where} — {args.shortcut}")


def debt_report() -> None:
    if not DEBT_LEDGER.exists():
        print("debt ledger empty")
        return
    entries = [json.loads(l) for l in DEBT_LEDGER.read_text().splitlines() if l.strip()]
    open_e = [e for e in entries if not e.get("repaid")]
    print(f"DEBT LEDGER — {len(open_e)} open / {len(entries)} total")
    for e in open_e:
        print(f"  {e['ts'][:10]}  {e['where']}")
        print(f"    shortcut : {e['shortcut']}")
        print(f"    why      : {e['why']}")
        print(f"    fix      : {e['planned_fix']}")
        print(f"    impact   : loc {e.get('loc_impact','?')}, speed {e.get('speed_impact','?')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    d = sub.add_parser("debt-add")
    d.add_argument("--where", required=True)
    d.add_argument("--shortcut", required=True)
    d.add_argument("--why", required=True)
    d.add_argument("--planned-fix", dest="planned_fix", default="")
    d.add_argument("--loc-impact", dest="loc_impact", type=int, default=0)
    d.add_argument("--speed-impact", dest="speed_impact", default="none")
    sub.add_parser("debt-report")
    args = ap.parse_args()

    if args.cmd == "audit":
        t0 = time.time()
        result = audit()
        elapsed = time.time() - t0
        write_report(result)
        append_metrics(result, elapsed)
        n = sum(len(v) for v in result["findings"].values())
        print(f"{n} findings across {result['files']} files — see report for detail")
    elif args.cmd == "debt-add":
        debt_add(args)
    elif args.cmd == "debt-report":
        debt_report()


if __name__ == "__main__":
    main()
