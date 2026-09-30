"""Human-graded sample for eval human.order_book_golden (2026-09-29).

Draws 50 investor-deck / transcript / results filings that state an order book (data/derived/order_book_filings.parquet,
seed 29 — a different seed from the automated audits, seeds 7 and 11) and writes evals/golden/order_book_to_grade.csv
for a person to grade. Each row shows what the extractor stored, every amount it saw in that filing, the text around
the first mention, and a link to the filing on NSE. The grader fills:
  human_verdict        correct | wrong_amount | segment_only | not_a_backlog  (correct = within 5% of the filing's
                       stated TOTAL order book)
  human_total_book_cr  the total order book the filing states, Rs crore (blank if it states none)
  notes                anything worth knowing
Never overwrites a file that already has grades in it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "evals/golden/order_book_to_grade.csv"
N, SEED = 50, 29


def main() -> None:
    if OUT.exists():
        x = pd.read_csv(OUT)
        if x["human_verdict"].notna().any():
            raise SystemExit(f"{OUT.relative_to(ROOT)} already has grades — not overwriting")
    O = pd.read_parquet(ROOT / "data/derived/order_book_filings.parquet")
    O = O[O["backlog_cr"].notna()].sample(N, random_state=SEED)
    urls = {}
    for f in (ROOT / "data/derived/order_book").glob("mentions_*of*.jsonl"):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l); urls[(r["symbol"], str(r["seq_id"]))] = r.get("url")
    out = pd.DataFrame(dict(
        id=range(1, N + 1), symbol=O["symbol"].values, filed=O["filed"].astype(str).str[:10].values, doc=O["desc"].values,
        filing_url=[urls.get((s, str(q))) for s, q in zip(O["symbol"], O["seq_id"])],
        extracted_book_cr=O["backlog_cr"].round(1).values, all_amounts_seen_cr=O["backlog_all_cr"].values,
        text_around_first_mention=O["snippet"].str.slice(0, 300).values,
        human_verdict="", human_total_book_cr="", notes=""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({N} filings to grade)")


if __name__ == "__main__":
    main()
