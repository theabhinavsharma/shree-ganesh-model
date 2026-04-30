# Agent Prompts — Index

The prompts in this directory are the **system messages** for each agent
in our pipeline. They are written in Anthropic-style: structured, calibrated,
with explicit output contracts, anti-patterns, and reference reading.

Read [`00_SYSTEM_PRINCIPLES.md`](00_SYSTEM_PRINCIPLES.md) first. Every
other prompt inherits its non-negotiables.

## The agents

| File | Agent | When it fires |
|---|---|---|
| [`00_SYSTEM_PRINCIPLES.md`](00_SYSTEM_PRINCIPLES.md) | Umbrella principles | Read before any agent run |
| [`01_HYPOTHESIS_GENERATOR.md`](01_HYPOTHESIS_GENERATOR.md) | Senior quant researcher | Weekly cycle (`agent_loop.py`); on user "what if" / "test factor X" |
| [`02_DATA_FETCHER.md`](02_DATA_FETCHER.md) | Senior data engineer | When new hypothesis needs new data; on user "fetch", "scrape" |
| [`03_BACKTEST_VALIDATOR.md`](03_BACKTEST_VALIDATOR.md) | Statistician + risk officer | When `state=IC_PASSED` factor needs portfolio A/B |
| [`04_TRADE_REVIEWER.md`](04_TRADE_REVIEWER.md) | PM + CRO | Daily 18:00 IST; on user "should I buy/sell X" |
| [`META_REPROMPT.md`](META_REPROMPT.md) | Self-prompt for Claude | Every user turn (apply before responding) |

## How to use these

### As an LLM operator (manual)

When you (the user) want a specific agent's behaviour, prefix your
request with the agent's role file:

```
[Apply prompts/04_TRADE_REVIEWER.md] What should I trade tomorrow?
```

Claude will then operate inside the role's contract.

### As an automation operator (programmatic)

In code, when calling Claude API:

```python
import anthropic
client = anthropic.Anthropic()
with open("prompts/00_SYSTEM_PRINCIPLES.md") as f:
    principles = f.read()
with open("prompts/04_TRADE_REVIEWER.md") as f:
    reviewer = f.read()

resp = client.messages.create(
    model="claude-opus-4-5",
    system=principles + "\n\n" + reviewer,
    messages=[{"role": "user", "content": today_context_dump}],
)
```

### As a skill (Claude Code)

In a future iteration, each prompt file can become a Claude Code skill:

```
.claude/skills/trade-reviewer/SKILL.md  ← prompts/04_TRADE_REVIEWER.md
```

Then `/trade-reviewer` activates it.

## Style guarantees (Anthropic-grade)

Every prompt in this directory:
- Has **identity + mission** in first 5 lines
- Has **operating principles** as numbered, binding rules
- Has **input contract** (XML schema for what the agent receives)
- Has **output contract** (exact XML structure of what it emits)
- Has **quality gates** (self-checks before emitting)
- Has **anti-patterns** (what will be rejected)
- Has **reference reading** (papers, books, Anthropic docs)
- Has **style examples** (one bad, one good)

If you write a new agent prompt, mirror this structure.

## Calibration log

When an agent's prompt evolves, log here:

| Date | File | Change | Reason |
|---|---|---|---|
| 2026-04-29 | All | Initial creation | Establishing Anthropic-grade structure |
| 2026-04-29 | 04_TRADE_REVIEWER | Added cascade-binding rule | 2026-04-28 Path B microcap failure |
| 2026-04-29 | 03_BACKTEST_VALIDATOR | Added 75% sign-consistency rule | 5-KEEP-factors A/B fail (high IC, no lift) |
