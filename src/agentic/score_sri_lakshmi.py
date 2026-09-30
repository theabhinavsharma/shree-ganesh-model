"""Daily forward scorer for the Sri Lakshmi PAPER batches (logs/sri_lakshmi/screen_*.json, written by
screen_sri_lakshmi.py). Same contract and code as score_leader_sleeve.py (next-open entry, 126 sessions, time exit,
0.5% round trip, equal weight); only the folder and the outcomes file differ. Outcomes: logs/sri_lakshmi/outcomes.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import score_leader_sleeve as sls  # noqa: E402

sls.DIR = ROOT / "logs/sri_lakshmi"
sls.OUT = sls.DIR / "outcomes.jsonl"

if __name__ == "__main__":
    sls.main()
