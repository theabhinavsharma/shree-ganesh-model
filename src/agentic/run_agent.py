"""Programmatic agent runner — actually invokes the prompt files.

Instead of treating prompts/*.md as documentation, this wraps the Claude API
with each prompt as a real system message. Use:

  python src/agentic/run_agent.py devils_advocate "Claim: KOTAKBANK 90% to double"
  python src/agentic/run_agent.py hypothesis_generator
  python src/agentic/run_agent.py trade_reviewer

Without an ANTHROPIC_API_KEY env var, it falls back to printing the system
prompt that WOULD have been used (so the user can manually use it).

This makes the prompts/*.md actually executable, not decorative.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PROMPTS = ROOT / "prompts"

AGENT_FILES = {
    "principles":         "00_SYSTEM_PRINCIPLES.md",
    "hypothesis":         "01_HYPOTHESIS_GENERATOR.md",
    "data_fetcher":       "02_DATA_FETCHER.md",
    "backtest":           "03_BACKTEST_VALIDATOR.md",
    "trade_reviewer":     "04_TRADE_REVIEWER.md",
    "devils_advocate":    "05_DEVILS_ADVOCATE.md",
    "meta":               "META_REPROMPT.md",
}


def load_system_prompt(agent: str) -> str:
    """Load principles + the named agent prompt as a single system message."""
    if agent not in AGENT_FILES:
        raise SystemExit(f"unknown agent '{agent}'. Available: {', '.join(AGENT_FILES)}")
    principles = (PROMPTS / "00_SYSTEM_PRINCIPLES.md").read_text()
    role = (PROMPTS / AGENT_FILES[agent]).read_text()
    return principles + "\n\n---\n\n" + role


def call_claude_api(system: str, user_msg: str, model: str = "claude-opus-4-5") -> str:
    """Invoke Claude API. Returns the response text. Falls back to manual mode if no key."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("[NO ANTHROPIC_API_KEY env var set]\n\n"
                "To actually invoke the agent, set ANTHROPIC_API_KEY and re-run.\n\n"
                "Manual fallback: paste the following system prompt + your message into Claude:\n\n"
                "=== SYSTEM PROMPT ===\n" + system + "\n\n"
                "=== USER MESSAGE ===\n" + user_msg)
    try:
        import anthropic
    except ImportError:
        return "[anthropic SDK not installed]\nInstall: pip install anthropic\n\n" + \
               "Or use manual mode by unsetting ANTHROPIC_API_KEY."
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=list(AGENT_FILES.keys()),
                    help="which agent prompt to wrap")
    ap.add_argument("message", nargs="?", default="",
                    help="user message; if blank, reads stdin")
    ap.add_argument("--model", default="claude-opus-4-5")
    ap.add_argument("--show-prompt", action="store_true",
                    help="just print the system prompt without invoking API")
    args = ap.parse_args()

    user_msg = args.message
    if not user_msg:
        user_msg = sys.stdin.read().strip()
    if not user_msg:
        raise SystemExit("no user message provided")

    system = load_system_prompt(args.agent)
    if args.show_prompt:
        print(f"=== {args.agent.upper()} system prompt ({len(system):,} chars) ===\n")
        print(system)
        return

    print(f"[invoking {args.agent} via {args.model}]\n")
    response = call_claude_api(system, user_msg, args.model)
    print(response)


if __name__ == "__main__":
    main()
