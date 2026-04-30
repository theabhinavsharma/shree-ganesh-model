from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.ingest.nse.fetch_bhavcopy import fetch_bhavcopy_range
from src.ingest.nse.models import BhavcopyFetchRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default="data/raw/nse")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--symbols", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = BhavcopyFetchRequest(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        output_dir=Path(args.output_dir),
        delay_seconds=args.delay,
        symbol_filter=set(args.symbols) if args.symbols else None,
    )
    fetch_bhavcopy_range(request)


if __name__ == "__main__":
    main()
