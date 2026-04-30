from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.config import load_research_config
from src.ml.pipeline import run_research_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cached walk-forward ML research across multiple universes.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_research.yaml"))
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    config = load_research_config(args.config)
    payload = run_research_pipeline(config, force_panel=args.force_panel)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

