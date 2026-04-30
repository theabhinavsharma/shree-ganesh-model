from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.expert_pipeline import load_expert_config
from src.ml.expert_pipeline import run_expert_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the expert walk-forward ML stack for short-horizon NSE ranking.")
    parser.add_argument("--config", type=Path, default=Path("configs/ml_expert.yaml"))
    parser.add_argument("--force-panel", action="store_true")
    args = parser.parse_args()

    config = load_expert_config(args.config)
    payload = run_expert_pipeline(config, force_panel=args.force_panel)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
