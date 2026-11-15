"""Security labelling and dating in the visual ledger export."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_visual_ledger import holder_column, issue_maturity, security_type, term_label  # noqa: E402


def test_issue_maturity_follows_the_engine_dating_rules() -> None:
    # The export derives maturity dates from settlement and term; they must match how the
    # engine dated the securities, or the book would show bills outliving their term.
    settled = pd.Timestamp("2026-07-07")
    four_week = 28 / 365.25
    assert security_type("Fixed", four_week) == "bill"
    assert term_label("bill", four_week) == "4w"
    assert issue_maturity(settled, "bill", four_week) == pd.Timestamp("2026-08-04")
    assert issue_maturity(pd.Timestamp("2026-08-31"), "note", 2.0) == pd.Timestamp("2028-08-31")
    assert security_type("Fixed", 7.0) == "note" and security_type("Fixed", 20.0) == "bond"


def test_holders_map_to_the_five_export_columns_and_unknowns_raise() -> None:
    assert holder_column("CB", None) == "fed"
    assert holder_column("Private", "mmf_cash_fund_route") == "mmf"
    assert holder_column("Private", float("nan")) == "other_domestic"
    with pytest.raises(ValueError):
        holder_column("TrustFunds", None)
