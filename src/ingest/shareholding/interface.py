from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from src.ingest.shareholding.nse import NseShareholdingFetchConfig, load_shareholding_from_nse


@dataclass(frozen=True)
class ShareholdingSourceConfig:
    input_path: Path | None = None
    source_name: str = "nse_api"
    output_dir: Path = Path("data/shareholding")
    symbols: set[str] | None = None
    delay_seconds: float = 0.1
    from_date: date | None = None
    to_date: date | None = None


def load_shareholding(config: ShareholdingSourceConfig) -> pd.DataFrame:
    if config.input_path and config.input_path.exists():
        return pd.read_parquet(config.input_path) if config.input_path.suffix == ".parquet" else pd.read_csv(config.input_path)
    if config.source_name != "nse_api":
        return pd.DataFrame()
    return load_shareholding_from_nse(
        NseShareholdingFetchConfig(
            output_dir=config.output_dir,
            symbols=config.symbols,
            delay_seconds=config.delay_seconds,
            from_date=config.from_date,
            to_date=config.to_date,
        )
    )
