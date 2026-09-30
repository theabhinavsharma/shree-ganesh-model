---
name: sgm-receipts
description: How to state numbers to Abhinav without hallucinating. Use for every answer that contains figures about the model, returns, data or the portfolio.
---

# Receipts rule

1. Every number that could drive a decision must come from something computed in this session (a command's output) or from a committed file, and you should be able to name it.
2. If you haven't computed it, say "estimate" and how you'd check it — or check it first.
3. When two numbers look inconsistent (e.g. 2,944 NSE rows vs 2,594 equities), explain the difference before the user asks.
4. Never carry a number over from another thread or project (the 15D 5% basket is not the Sri Lakshmi model).
5. The Stop hook `src/agentic/trust/claim_check.py` lists numbers in your reply that have no receipt; if it flags one, fix it in the next reply.
