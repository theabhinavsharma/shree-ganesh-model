import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402


def test_etfs_excluded():
    s = pd.Series(["GOLDBEES", "SETFGOLD", "GOLDIETF", "MASPTOP50", "NIFTYBEES", "HDFCGOLD",
                   "LIQUID", "SILVER", "SETFNIF50", "EBBETF0433", "N100", "TOP10ADD"])
    assert non_equity(s).all()


def test_gold_named_companies_kept():
    s = pd.Series(["SKYGOLD", "SHANTIGOLD", "GOLDTECH", "GOLDIAM", "TBZ", "TITAN"])
    assert not non_equity(s).any()
