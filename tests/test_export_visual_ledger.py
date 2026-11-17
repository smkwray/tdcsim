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
    # Deposit and non-deposit fiscal flows are read side by side; they must add to the
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


def test_daily_financing_cost_matches_the_fiscal_year_totals_the_family_reports(tmp_path) -> None:
    # The daily cumulative cost is read next to the scenario-family totals; the two must
    # be one basis, so the daily running sum has to land on every fiscal-year total.
    import export_visual_ledger as ledger

    dates = pd.to_datetime(["2026-06-21", "2026-09-30", "2026-10-01", "2027-09-30"])
    flows = {"InterestOutlay_Period": [0.0, 1.0, 2.0, 3.0], "IssueDiscountCost_Period": [0.0, 0.5, 0.0, 0.5],
             "NonMarketableInterestCapitalized_Period": [0.0] * 4, "TIPSInflationAccretion_Period": [0.0, 0.1, 0.1, 0.1]}
    results = pd.DataFrame(flows, index=pd.Index(dates, name="Date"))
    results["FinancingCost_Period"] = results.sum(axis=1)
    annual = pd.DataFrame({
        "period_end": ["2026-09-30", "2027-09-30"],
        "cumulative_interest_outlay_bil": [1.0, 6.0], "cumulative_issue_discount_cost_bil": [0.5, 1.0],
        "cumulative_nonmarketable_interest_capitalized_bil": [0.0, 0.0],
        "cumulative_tips_inflation_accretion_bil": [0.1, 0.3], "cumulative_modeled_financing_cost_bil": [1.6, 7.3],
    })
    (tmp_path / ledger.ANNUAL_SUMMARY).parent.mkdir(parents=True)
    annual.to_csv(tmp_path / ledger.ANNUAL_SUMMARY, index=False)

    cumulative, checks = ledger.build_financing_cost(tmp_path, results)
    assert cumulative.tolist() == pytest.approx([0.0, 1.6, 3.7, 7.3])
    assert checks["components_bil"] == pytest.approx({
        "interest_outlay": 6.0, "issue_discount_cost": 1.0, "nonmarketable_interest_capitalized": 0.0,
        "tips_inflation_accretion": 0.3, "modeled_financing_cost": 7.3})
    # A cost flow the annual summary does not carry (here on the opening day) breaks the basis.
    results.loc[dates[0], ["InterestOutlay_Period", "FinancingCost_Period"]] = 0.2
    with pytest.raises(SystemExit, match="differs from the run's annual summary"):
        ledger.build_financing_cost(tmp_path, results)


def test_opening_book_lists_pre_start_replacements_under_their_own_cusips(tmp_path) -> None:
    # Securities issued between the source book's record date and the opening must appear as
    # the separate securities Treasury issued, merged into a CUSIP the book already holds and
    # never pooled into one synthetic security.
    import export_visual_ledger as ledger

    def row(bond_id, issue, maturity, face, holder="Banks"):
        return {"BondID": bond_id, "SecurityType": "Fixed", "IssueDate": issue, "MaturityDate": maturity,
                "OriginalMaturityYears": 0.33, "CouponRate": 0.0, "FaceValue": face, "AdjustedPrincipal": 0.0,
                "HolderType": holder, "HolderSubBucket": None}

    for path in (ledger.OPENING_PORTFOLIO, ledger.OPENING_SOURCE, ledger.ROLLFORWARD):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        row(1, "2026-03-03", "2026-06-30", 10.0),
        row(9, "2026-03-03", "2026-06-30", 4.0, "Foreign"),  # reinvestment in the book's CUSIP
        row(10, "2026-06-04", "2026-12-03", 6.0),  # reinvestment in a new issue
        row(11, "2026-06-11", "2026-12-10", 5.0),  # and another
    ]).to_csv(tmp_path / ledger.OPENING_PORTFOLIO, index=False)
    pd.DataFrame([{"BondID": 1, "cusip_or_synthetic_id": "912797UC9", "MaturityDate": "2026-06-30"}]).to_csv(
        tmp_path / ledger.OPENING_SOURCE, index=False)
    pd.DataFrame([
        {"row_type": "summary", "replacement_bond_id": None, "replacement_cusip": None},
        {"row_type": "replacement", "replacement_bond_id": 9, "replacement_cusip": "912797UC9"},
        {"row_type": "replacement", "replacement_bond_id": 10, "replacement_cusip": "912797VA2"},
        {"row_type": "replacement", "replacement_bond_id": 11, "replacement_cusip": "912797VG9"},
    ]).to_csv(tmp_path / ledger.ROLLFORWARD, index=False)

    book = ledger.build_opening(tmp_path).set_index("security_id")
    assert book["face_total"].to_dict() == pytest.approx({"912797UC9": 14.0, "912797VA2": 6.0, "912797VG9": 5.0})
    assert book.loc["912797UC9", "face_foreign"] == pytest.approx(4.0)
    # A row that is neither in the source book nor a named replacement fails the export.
    pd.DataFrame([{"row_type": "summary", "replacement_bond_id": None, "replacement_cusip": None}]).to_csv(
        tmp_path / ledger.ROLLFORWARD, index=False)
    with pytest.raises(SystemExit, match="neither in the source book"):
        ledger.build_opening(tmp_path)
