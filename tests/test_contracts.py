from pathlib import Path

from src.utils.schema import load_contract


def test_all_contracts_load() -> None:
    contract_dir = Path("data_contracts")
    for path in contract_dir.glob("*.yaml"):
        contract = load_contract(path)
        assert "table" in contract
        assert "columns" in contract
        assert len(contract["columns"]) > 0
