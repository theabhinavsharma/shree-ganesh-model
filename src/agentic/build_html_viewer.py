"""Build a single self-contained HTML file that renders all the mermaid
diagrams + the live status dashboard.

Open with:
  open reports/visualize.html

No server needed, no plugins. Just double-click.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "reports/visualize.html"


def read(p: Path) -> str:
    return p.read_text() if p.exists() else "(file missing)"


def extract_mermaid_blocks(md_text: str) -> list[str]:
    return re.findall(r"```mermaid\s*\n(.*?)\n```", md_text, re.DOTALL)


def main() -> None:
    arch = read(ROOT / "ARCHITECTURE.md")
    workflow = read(ROOT / "reports/WORKFLOW.md")
    status = read(ROOT / "reports/status.md")
    cascade = ""
    cascade_files = sorted((ROOT / "reports").glob("filter_cascade_*.md"), reverse=True)
    if cascade_files:
        cascade = read(cascade_files[0])
    brief = ""
    brief_files = sorted((ROOT / "reports").glob("daily_pro_brief_*.md"), reverse=True)
    if brief_files:
        brief = read(brief_files[0])
    completeness = ""
    comp_files = sorted((ROOT / "reports").glob("data_completeness_*.md"), reverse=True)
    if comp_files:
        completeness = read(comp_files[0])

    # collect all agent prompts
    prompt_files = sorted((ROOT / "prompts").glob("*.md"))
    prompts_payload = []
    for pf in prompt_files:
        prompts_payload.append({"name": pf.name, "content": read(pf)})

    # extract mermaid blocks from architecture + workflow
    arch_diagrams = extract_mermaid_blocks(arch)
    workflow_diagrams = extract_mermaid_blocks(workflow)

    # registry stats
    reg_path = ROOT / "data/derived/factor_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        by_state: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for h in reg:
            by_state[h["state"]] = by_state.get(h["state"], 0) + 1
            by_category[h["category"]] = by_category.get(h["category"], 0) + 1
    else:
        reg, by_state, by_category = [], {}, {}

    # cycle log
    cycle_log = ROOT / "logs/agent_loop_cycles.jsonl"
    cycle_records = []
    if cycle_log.exists():
        for line in cycle_log.read_text().splitlines():
            try:
                cycle_records.append(json.loads(line))
            except Exception:
                pass

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NSE Agentic Pipeline — Visualizer</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; background: #0d1117; color: #e6edf3; }}
  header {{ background: #161b22; padding: 16px 24px; border-bottom: 1px solid #30363d; }}
  header h1 {{ margin: 0; font-size: 20px; }}
  header .subtitle {{ color: #8b949e; font-size: 13px; margin-top: 4px; }}
  nav {{ background: #161b22; padding: 0 24px; border-bottom: 1px solid #30363d;
         position: sticky; top: 0; z-index: 100; }}
  nav button {{ background: transparent; border: none; color: #8b949e; padding: 12px 16px;
                font-size: 14px; cursor: pointer; border-bottom: 2px solid transparent; }}
  nav button.active {{ color: #58a6ff; border-bottom-color: #58a6ff; }}
  nav button:hover {{ color: #c9d1d9; }}
  main {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 12px; margin-bottom: 24px; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; }}
  .stat .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; }}
  .stat .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; color: #58a6ff; }}
  .mermaid {{ background: #0d1117; padding: 16px; border-radius: 8px;
              border: 1px solid #30363d; margin: 16px 0; }}
  .markdown-body {{ background: #0d1117; padding: 24px; border-radius: 8px;
                    border: 1px solid #30363d; }}
  .markdown-body h1, .markdown-body h2, .markdown-body h3 {{ color: #f0f6fc; }}
  .markdown-body h1 {{ border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  .markdown-body table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  .markdown-body th, .markdown-body td {{ border: 1px solid #30363d; padding: 6px 12px; text-align: left; }}
  .markdown-body th {{ background: #161b22; color: #c9d1d9; }}
  .markdown-body code {{ background: #161b22; padding: 2px 6px; border-radius: 3px;
                          font-size: 13px; color: #ff7b72; }}
  .markdown-body pre {{ background: #161b22; padding: 12px; border-radius: 6px; overflow-x: auto; }}
  .markdown-body pre code {{ background: transparent; padding: 0; color: #e6edf3; }}
  .markdown-body blockquote {{ border-left: 3px solid #30363d; padding-left: 12px;
                                color: #8b949e; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 11px; font-weight: 600; margin-right: 6px; }}
  .badge.proposed {{ background: #1f3a5f; color: #58a6ff; }}
  .badge.evaluated {{ background: #5f4f1e; color: #f0c674; }}
  .badge.keep {{ background: #1e5f2c; color: #56d364; }}
  .badge.drop {{ background: #5f1e1e; color: #f85149; }}
  .badge.ic_passed {{ background: #5f4f1e; color: #f0c674; }}
  .badge.drop_ab_fail {{ background: #5f1e1e; color: #f85149; }}
  .badge.blocked {{ background: #3a3a3a; color: #8b949e; }}
  details {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
             padding: 12px; margin: 8px 0; }}
  details summary {{ cursor: pointer; font-weight: 600; color: #c9d1d9; }}
  .updated {{ color: #8b949e; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>🧠 NSE Agentic Pipeline — Visualizer</h1>
  <div class="subtitle">Generated {datetime.now():%Y-%m-%d %H:%M IST} · {len(reg)} hypotheses · {sum(by_state.values())} tracked</div>
</header>

<nav>
  <button class="tab-btn active" data-target="overview">📊 Overview</button>
  <button class="tab-btn" data-target="architecture">🏗 Architecture</button>
  <button class="tab-btn" data-target="pipeline">⚙ Pipeline (25 steps)</button>
  <button class="tab-btn" data-target="agents">🤖 Agents (75 hypotheses)</button>
  <button class="tab-btn" data-target="status">📈 Live Status</button>
  <button class="tab-btn" data-target="cascade">🛑 Today's Cascade</button>
  <button class="tab-btn" data-target="brief">📋 Today's Brief</button>
  <button class="tab-btn" data-target="prompts">🧬 Agent Prompts</button>
</nav>

<main>

<div id="overview" class="panel active">
  <h2>What this system does</h2>
  <div class="stats-grid">
    <div class="stat"><div class="label">Total hypotheses</div><div class="value">{len(reg)}</div></div>
    <div class="stat"><div class="label">PROPOSED</div><div class="value">{by_state.get("PROPOSED", 0)}</div></div>
    <div class="stat"><div class="label">EVALUATED</div><div class="value">{by_state.get("EVALUATED", 0) + by_state.get("DROP", 0) + by_state.get("KEEP", 0) + by_state.get("DROP_AB_FAIL", 0) + by_state.get("IC_PASSED", 0)}</div></div>
    <div class="stat"><div class="label">KEEP (production)</div><div class="value">{by_state.get("KEEP", 0)}</div></div>
    <div class="stat"><div class="label">DROP_AB_FAIL</div><div class="value">{by_state.get("DROP_AB_FAIL", 0)}</div></div>
    <div class="stat"><div class="label">Cycles run</div><div class="value">{len(cycle_records)}</div></div>
  </div>
  <h3>Latest cycle</h3>
"""
    if cycle_records:
        last = cycle_records[-1]
        html += f"""
  <div class="markdown-body">
    <p><strong>Run at:</strong> {last.get("ts", "—")}</p>
    <p>Processed {last.get("n_processed", 0)} hypotheses ·
       Blocked {last.get("n_blocked", 0)} ·
       IC_PASSED awaiting A/B {last.get("n_ic_passed_pending_ab", 0)}</p>
  </div>
"""
    html += """
  <h3>Hypothesis distribution</h3>
  <div class="markdown-body">
    <table><tr><th>Category</th><th>Count</th></tr>
"""
    for c, n in sorted(by_category.items(), key=lambda x: -x[1]):
        html += f"      <tr><td><code>{c}</code></td><td>{n}</td></tr>\n"
    html += """    </table>
  </div>
</div>

<div id="architecture" class="panel">
  <h2>5-Layer Architecture</h2>
"""
    for d in arch_diagrams:
        html += f'  <div class="mermaid">{d}</div>\n'
    html += """
</div>

<div id="pipeline" class="panel">
  <h2>Daily pipeline (25 sequential steps)</h2>
"""
    for d in workflow_diagrams:
        html += f'  <div class="mermaid">{d}</div>\n'
    html += """
</div>

<div id="agents" class="panel">
  <h2>All 75 Hypotheses (the agentic catalog)</h2>
  <div id="hypotheses-list"></div>
</div>

<div id="status" class="panel">
  <h2>Live Status</h2>
  <div class="markdown-body" id="status-md"></div>
</div>

<div id="cascade" class="panel">
  <h2>Today's Filter Cascade</h2>
  <div class="markdown-body" id="cascade-md"></div>
</div>

<div id="brief" class="panel">
  <h2>Today's Pro Brief</h2>
  <div class="markdown-body" id="brief-md"></div>
</div>

<div id="prompts" class="panel">
  <h2>Anthropic-grade Agent Prompts</h2>
  <p style="color:#8b949e; font-size:13px;">
    These are the structured system prompts for each agent in the pipeline.
    Each follows Anthropic engineering style: identity + mission + operating
    principles + I/O contract + quality gates + anti-patterns + reference reading.
  </p>
  <div id="prompts-list"></div>
</div>

</main>

<script>
mermaid.initialize({ startOnLoad: true, theme: 'dark' });

// tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.target).classList.add('active');
  });
});

// load registry into agents panel
const REGISTRY = """ + json.dumps(reg) + """;
function renderHypotheses() {
  const container = document.getElementById('hypotheses-list');
  const groups = {};
  for (const h of REGISTRY) {
    if (!groups[h.category]) groups[h.category] = [];
    groups[h.category].push(h);
  }
  const order = Object.keys(groups).sort((a,b) => groups[b].length - groups[a].length);
  for (const cat of order) {
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = `${cat} (${groups[cat].length})`;
    det.appendChild(sum);
    for (const h of groups[cat]) {
      const div = document.createElement('div');
      div.style.padding = '8px 0';
      div.style.borderBottom = '1px solid #30363d';
      const stateClass = h.state.toLowerCase().replace('_', '_');
      div.innerHTML = `
        <span class="badge ${stateClass}">${h.state}</span>
        <strong>${h.id}</strong> — ${h.name}<br>
        <span style="color: #8b949e; font-size: 13px;">${h.description}</span>
        ${h.formula ? `<br><code>${h.formula}</code>` : ''}
        ${h.notes ? `<br><span style="color:#f85149; font-size:12px;">⚠ ${h.notes}</span>` : ''}
      `;
      det.appendChild(div);
    }
    container.appendChild(det);
  }
}
renderHypotheses();

// markdown rendering
const STATUS_MD = """ + json.dumps(status) + """;
const CASCADE_MD = """ + json.dumps(cascade) + """;
const BRIEF_MD = """ + json.dumps(brief) + """;
document.getElementById('status-md').innerHTML = marked.parse(STATUS_MD);
document.getElementById('cascade-md').innerHTML = marked.parse(CASCADE_MD);
document.getElementById('brief-md').innerHTML = marked.parse(BRIEF_MD);

// agent prompts
const PROMPTS = """ + json.dumps(prompts_payload) + """;
function renderPrompts() {
  const container = document.getElementById('prompts-list');
  for (const p of PROMPTS) {
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = p.name;
    sum.style.fontFamily = 'monospace';
    det.appendChild(sum);
    const body = document.createElement('div');
    body.className = 'markdown-body';
    body.style.marginTop = '8px';
    body.innerHTML = marked.parse(p.content);
    det.appendChild(body);
    container.appendChild(det);
  }
}
renderPrompts();
</script>
</body>
</html>
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB)")
    print(f"  open with: open {OUT}")


if __name__ == "__main__":
    main()
