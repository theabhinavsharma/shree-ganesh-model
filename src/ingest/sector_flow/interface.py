from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ingest.sector_flow.nsdl import NsdlSectorFlowFetchConfig, load_sector_flow_from_nsdl


@dataclass(frozen=True)
class SectorFlowSourceConfig:
    input_path: Path | None = None
    source_name: str = "nsdl_report"
    output_dir: Path = Path("data/sector_flow")
    limit: int | None = None
    delay_seconds: float = 0.1


def load_sector_flow(config: SectorFlowSourceConfig) -> pd.DataFrame:
    if config.input_path and config.input_path.exists():
        if config.input_path.suffix == ".parquet":
            return pd.read_parquet(config.input_path)
        return pd.read_csv(config.input_path, parse_dates=["fortnight_end_date", "published_date", "effective_from_date"])
    if config.source_name != "nsdl_report":
        return pd.DataFrame()
    return load_sector_flow_from_nsdl(
        NsdlSectorFlowFetchConfig(
            output_dir=config.output_dir,
            limit=config.limit,
            delay_seconds=config.delay_seconds,
        )
    )
