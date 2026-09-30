"""Daily forward scorer for the model-ranked PAPER screen (logs/model_screen/screen_*.json, written by
screen_model_ranked.py). Same contract and code as score_leader_sleeve.py (next-open entry, 126 sessions, time exit,
0.5% round trip, equal weight); only the folder and the outcomes file differ, so the registered sleeve's record in
logs/leader_sleeve/ is untouched. Outcomes: logs/model_screen/outcomes.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import score_leader_sleeve as sls  # noqa: E402

sls.DIR = ROOT / "logs/model_screen"
sls.OUT = sls.DIR / "outcomes.jsonl"

if __name__ == "__main__":
    sls.main()
