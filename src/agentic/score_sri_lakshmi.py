"""Daily forward scorer for the Sri Lakshmi PAPER batches (logs/sri_lakshmi/screen_*.json, written by
screen_sri_lakshmi.py). Same contract and code as score_leader_sleeve.py (next-open entry, 126 sessions, time exit,
0.5% round trip, equal weight); only the folder and the outcomes file differ. Outcomes: logs/sri_lakshmi/outcomes.jsonl.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/abhinavs./Code/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
import score_leader_sleeve as sls  # noqa: E402

sls.DIR = ROOT / ("logs/sri_lakshmi_g1" if "--shadow" in sys.argv else "logs/sri_lakshmi")   # --shadow: old-rule paper track
sls.OUT = sls.DIR / "outcomes.jsonl"
sys.argv = [a for a in sys.argv if a != "--shadow"]

if __name__ == "__main__":
    if not list(sls.DIR.glob("screen_*.json")):   # e.g. 2026-09-30..10-02: G1 batch moved to not_invested/, first V3 batch on Friday
        print(f"no Sri Lakshmi batches in {sls.DIR} yet — nothing to score"); sys.exit(0)
    sls.main()
