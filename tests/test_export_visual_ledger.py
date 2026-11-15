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
    assert holder_column("Private", "domestic_nonbank_deposit_funded") == "other_domestic"
    # A Private row without a route is malformed; guessing a column would hide it.
    for holder, route in (("Private", float("nan")), ("TrustFunds", None)):
        with pytest.raises(ValueError):
            holder_column(holder, route)


def test_fiscal_incidence_splits_the_primary_deficit_without_remainder(tmp_path) -> None:
    # The deck shows deposit and non-deposit fiscal flows side by side; they must add to the
    # deficit, and the non-deposit part must follow the run's own incidence policy.
    import export_visual_ledger as ledger

    run = tmp_path
    inputs = run / ledger.FISCAL_POLICY
    inputs.parent.mkdir(parents=True)
    pd.DataFrame([
        {"policy_id": "p", "incidence_basis": "signed_net_primary_proxy", "du_share": 0.9, "ru_share": 0.06,
         "foreign_share": 0.04, "other_share": 0.0, "primary_outlays_bil": None, "primary_receipts_bil": None},
    ]).to_csv(inputs, index=False)
    dates = pd.to_datetime(["2026-06-21", "2026-06-22"])
    (run / ledger.TRC).parent.mkdir(parents=True)
    pd.DataFrame({"Date": dates, "TRC_FiscalFlow": [0.0, 2.0]}).to_csv(run / ledger.TRC, index=False)
    results = pd.DataFrame({"PrimaryDeficit": [0.0, 2.0], "TDC_FiscalFlow": [0.0, 1.8]}, index=pd.Index(dates, name="Date"))

    fiscal, checks = ledger.build_fiscal(run, results, "p")
    assert fiscal["fiscal_to_rs"].tolist() == pytest.approx([0.0, 0.2])
    assert checks["gross_outlays_and_receipts_modeled"] is False
    with pytest.raises(SystemExit, match="not in the run"):
        ledger.build_fiscal(run, results, "missing")
    results.loc[dates[1], "TDC_FiscalFlow"] = 1.0
    with pytest.raises(SystemExit, match="does not add up"):
        ledger.build_fiscal(run, results, "p")
