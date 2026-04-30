from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_contract(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
