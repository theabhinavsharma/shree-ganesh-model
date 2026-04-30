from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.data_catalog import resolve_artifact_spec
from src.utils.data_catalog import sidecar_manifest_path
from src.utils.data_catalog import write_dataframe_manifest
from src.utils.data_catalog import write_json_manifest
from src.utils.data_catalog import write_report_directory_readme


def test_write_dataframe_manifest_uses_contract_metadata(tmp_path: Path) -> None:
    path = tmp_path / "current_universe_enriched.parquet"
    frame = pd.DataFrame(
        {
            "trade_date": ["2026-03-25"],
            "symbol": ["AAA"],
            "close": [100.0],
            "missing_inputs": [""],
        }
    )
    frame.to_parquet(path, index=False)

    manifest_path = write_dataframe_manifest(
        path,
        frame,
        generated_by="tests",
        as_of_date="2026-03-25",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_key"] == "current_universe_enriched"
    columns = {column["name"]: column for column in payload["columns"]}
    assert columns["trade_date"]["definition"] == "Screening date"
    assert columns["symbol"]["definition"] == "NSE symbol"


def test_write_json_manifest_records_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"as_of_trade_date": "2026-03-25", "results": {}}), encoding="utf-8")

    manifest_path = write_json_manifest(
        path,
        {"as_of_trade_date": "2026-03-25", "results": {}},
        generated_by="tests",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["artifact_key"] == "screen_run_summary"
    assert payload["top_level_keys"] == ["as_of_trade_date", "results"]


def test_write_report_directory_readme_lists_file_purposes(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text("{}", encoding="utf-8")
    counts = tmp_path / "mcap_5000_individual_counts.csv"
    counts.write_text("rule_column,individual_pass_count\n", encoding="utf-8")

    readme = write_report_directory_readme(
        tmp_path,
        title="Sample Output",
        intro_lines=["This is a sample."],
        files=[summary, counts],
    )

    text = readme.read_text(encoding="utf-8")
    assert "summary.json" in text
    assert "Per-rule pass counts" in text


def test_resolve_artifact_spec_understands_weekly_portfolio_pattern() -> None:
    spec = resolve_artifact_spec(Path("weekly_portfolio_20260326.csv"))
    assert spec.artifact_key == "weekly_portfolio_csv"
    assert spec.friendly_name == "Weekly Portfolio Report"


def test_sidecar_manifest_path_appends_manifest_suffix() -> None:
    path = Path("tmp/example.csv")
    assert sidecar_manifest_path(path) == Path("tmp/example.csv.manifest.json")
