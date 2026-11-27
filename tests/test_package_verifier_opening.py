"""Package-verifier checks on the opening book and the opening Fed identity.

A package built from any correctly sourced opening book must pass, so the verifier
rebuilds the book's counts from the packaged source rather than pinning one book's
counts; and the engine's opening Fed row (a nonsettling statement that the opening
stock meets the Fed target) must be accepted only when it really moves nothing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import scripts.run_cbo_forecast_smoke as smoke
from cbo_opening_actuals import (
    DTS_FISCAL_LINES, DTS_TGA_CLOSING_ACCOUNT, DTS_TGA_OPENING_ACCOUNT, SOMA_SOURCE,
    build_opening_fy_splice, opening_cash_reconciliation,
)
from mmf_eligibility import mmf_may_acquire
from tdc_shared import PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF
from scripts.verify_cbo_forecast_package import (
    _OPENING_FED_FLOW_COLUMNS,
    _verify_fed_stock_bridge,
    _verify_opening_counts_against_source_book,
)

START = "2026-06-21"


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _book_row(bond_id, face, maturity, cusip, holder, subbucket="", category="bills") -> dict:
    return {
        "BondID": bond_id, "SecurityType": "Fixed", "IssueDate": "2026-01-02", "MaturityDate": maturity,
        "FaceValue": face, "CouponRate": "0.03" if category == "notes" else "0.0",
        "HolderType": holder, "HolderSubBucket": subbucket, "Status": "Active",
        "MaturityCategory": category, "cusip_or_synthetic_id": cusip,
    }


def _write_auctions(path: Path) -> None:
    issues = [
        ("N1", "Note", "2026-06-01", "2028-06-01", 55, 5),
        ("B1", "Bill", "2026-06-02", "2026-09-02", 20, 0),
        ("B2", "Bill", "2026-06-18", "2026-09-17", 8, 4),
        ("B3", "Bill", "2026-06-22", "2026-09-22", 9, 0),
    ]
    _write_rows(
        path,
        [
            {
                "cusip": cusip, "security_type": kind, "issue_date": issue, "original_issue_date": issue,
                "maturity_date": maturity, "total_accepted": face * 10**9, "soma_accepted": soma * 10**9,
                "int_rate": "3.0", "price_per100": "99.0", "high_investment_rate": "4.0", "high_yield": "3.1",
                "floating_rate": "No", "inflation_index_security": "No",
            }
            for cusip, kind, issue, maturity, face, soma in issues
        ],
    )


def test_opening_builder_and_verifier_roll_net_issuance_soma_and_mmf(tmp_path: Path, monkeypatch) -> None:
    book = tmp_path / "sources/opening/tdcsim_initial_portfolio_cohorts.csv"
    governed = tmp_path / "sources/opening/tdcsim_ratewall_input_manifest.json"
    auctions = tmp_path / "sources/fiscaldata/auctions_query.csv"
    soma = tmp_path / "sources" / SOMA_SOURCE
    _write_rows(
        book,
        [
            _book_row(1, 10, "2026-05-31", "UNPAID", "Private", PRIVATE_SUBBUCKET_MMF),
            _book_row(2, 20, "2026-05-31", "UNPAID", "Banks"),
            _book_row(3, 5, "2026-05-31", "UNPAID", "CB"),
            _book_row(4, 6, "2026-06-18", "OLD-BILL", "Banks"),
            _book_row(5, 4, "2026-06-18", "OLD-BILL", "CB"),
            _book_row(6, 20, "2028-06-15", "OLD-NOTE", "Banks", category="notes"),
            _book_row(7, 5, "2028-06-15", "OLD-NOTE", "Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, category="notes"),
            _book_row(8, 5, "2028-06-15", "OLD-NOTE", "CB", category="notes"),
            _book_row(9, 15, "2027-06-10", "KEEP-BILL", "Foreign"),
        ],
    )
    shares = {
        "Banks": 0.3, "Foreign": 0.3, f"Private/{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}": 0.3,
        f"Private/{PRIVATE_SUBBUCKET_MMF}": 0.1,
    }
    governed.write_text(json.dumps({"artifacts": {"initial_portfolio": {
        "cb_source": smoke.OPENING_PORTFOLIO_CB_SOURCE, "mspd_record_date": "2026-05-31",
        "noncb_weights": {"bills": shares, "nonbills": shares},
    }}}))
    _write_auctions(auctions)
    _write_rows(soma, [
        {"As Of Date": "2026-06-17", "CUSIP": cusip, "Security Type": kind,
         "Maturity Date": maturity, "Par Value": face * 10**9, "Inflation Compensation": ""}
        for cusip, kind, maturity, face in [
            ("OLD-BILL", "Bills", "2026-06-18", 4), ("OLD-NOTE", "Notes", "2028-06-15", 8),
            ("N1", "Notes", "2028-06-01", 5),
        ]
    ])
    empty_index_fixture = tmp_path / "empty.csv"
    empty_index_fixture.touch()
    monkeypatch.setattr(smoke, "OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL", 90)
    arguments = {
        "source_manifest": governed, "tips_cpi_detail": empty_index_fixture,
        "frn_daily_indexes": empty_index_fixture, "auctions_query": auctions, "soma_holdings": soma,
    }
    portfolio, metadata = smoke.load_opening_portfolio(book, simulation_start_date=START, **arguments)
    assert smoke.SIMULATION_START_DATE == START
    assert metadata["opening_close_date"] == "2026-06-18"
    assert portfolio["FaceValue"].sum() == pytest.approx(90 + 55 + 20 + 8 - 45)
    assert portfolio.loc[portfolio["HolderType"].eq("CB"), "FaceValue"].sum() == pytest.approx(17)
    diagnostics = metadata["opening_rollforward_diagnostic_rows"]
    assert diagnostics[0]["mmf_reinvestment_deferred_bil"] == 10
    assert {"row_type": "mmf_acquisition", "acquisition_date": "2026-06-02",
            "security_type": "Fixed", "maturity_date": "2026-09-02"} in diagnostics
    mmf = portfolio.loc[portfolio["HolderSubBucket"].eq(PRIVATE_SUBBUCKET_MMF)]
    assert mmf["FaceValue"].sum() == pytest.approx(11)
    assert all(
        mmf_may_acquire(row.SecurityType, row.MaturityDate, row.IssueDate) for row in mmf.itertuples()
    )
    # No maturities from May 31 or June 18 remain, and no June 22 issue enters.
    assert set(portfolio["IssueDate"].dropna().dt.strftime("%Y-%m-%d")) == {
        "2026-01-02", "2026-06-01", "2026-06-02", "2026-06-18",
    }
    manifest = {"opening_portfolio_construction": {
        **metadata, "soma_snapshot_date": "2026-06-17", "opening_cb_holdings_bil": 17,
    }}
    _verify_opening_counts_against_source_book(tmp_path, manifest)
    manifest["opening_portfolio_construction"]["prestart_rollforward_rows"] -= 1
    with pytest.raises(ValueError, match="prestart_rollforward_rows"):
        _verify_opening_counts_against_source_book(tmp_path, manifest)
    with pytest.raises(ValueError, match="simulation_start_date=2026-06-21"):
        smoke.load_opening_portfolio(book, simulation_start_date="2026-06-18", **arguments)
    # The governed book must already be eligibility-clean: a money fund holding a note with
    # two years left is an upstream defect, not something the roll may quietly reassign.
    rows = list(csv.DictReader(book.open(newline="")))
    rows[6]["HolderSubBucket"] = PRIVATE_SUBBUCKET_MMF
    rows[6]["MaturityDate"] = "2027-07-15"
    _write_rows(book, rows)
    assert mmf_may_acquire("Fixed", "2027-07-15", "2026-06-18")
    with pytest.raises(ValueError, match="ineligible OLD-NOTE on 2026-05-31"):
        smoke.load_opening_portfolio(book, simulation_start_date=START, **arguments)


def test_opening_cash_and_splice_use_same_close(tmp_path: Path) -> None:
    cash = tmp_path / "table_i.csv"
    flows = tmp_path / "table_ii.csv"
    cash_rows = [
        {"record_date": "2026-06-01", "account_type": DTS_TGA_OPENING_ACCOUNT, "open_today_bal": 99_000},
        {"record_date": "2026-06-18", "account_type": DTS_TGA_CLOSING_ACCOUNT, "open_today_bal": 100_000},
    ]
    _write_rows(cash, cash_rows)
    values = [50_000, 49_000, 25_000, 21_000, 2_000]
    _write_rows(flows, [
        {"record_date": "2026-06-18", "transaction_type": kind, "account_type": account,
         "transaction_catg": category, "transaction_mtd_amt": value}
        for (kind, account, category), value in zip(DTS_FISCAL_LINES.values(), values)
    ])
    dts = opening_cash_reconciliation(cash, flows)
    assert dts["opening_close_date"] == "2026-06-18"
    assert dts["tga_close_bil"] == 100
    assert dts["tga_identity_residual_bil"] == 0
    assert dts["primary_deficit_bil"] == 1
    splice = build_opening_fy_splice(
        scenario_id="fixture", cbo_primary_deficit_bil=800, mts_deficit_bil=1200, mts_interest_bil=700, dts=dts,
    )
    assert splice["fiscal_actuals_through"] == dts["opening_close_date"]
    assert splice["observation_date"] == "2026-06-18"
    assert splice["actuals_available_as_of"] == "2026-06-22"
    assert splice["simulation_start_date"] == START
    assert splice["actual_primary_deficit_fytd_bil"] == 501
    assert splice["remaining_cbo_primary_deficit_bil"] == 299
    _write_rows(cash, [cash_rows[0], {**cash_rows[1], "record_date": "2026-06-16"}])
    with pytest.raises(ValueError, match="2026-06-18"):
        opening_cash_reconciliation(cash, flows)
    _write_rows(cash, [cash_rows[0], {**cash_rows[1], "open_today_bal": 100_003}])
    with pytest.raises(ValueError, match="3 million, exceeding 2 million"):
        opening_cash_reconciliation(cash, flows)


def _fed_row(**values: object) -> dict[str, str]:
    row = {column: "0.0" for column in _OPENING_FED_FLOW_COLUMNS}
    row.update(
        {
            "CBOFedHoldingsTargetApplicable": "1.0",
            "CBOFedHoldingsTarget": "4452.2",
            "CBOFedHoldingsTargetError": "0.0",
            "DebtHeld_CentralBank": "4452.2",
            "CBOFedSettlementScope": "opening_stock_target_match_nonsettling",
        }
    )
    row.update({key: str(value) for key, value in values.items()})
    return row


def test_opening_fed_identity_row_is_accepted_only_when_it_moves_nothing() -> None:
    """Since the stock-flow accounting change the engine states at the opening date that
    the Fed's opening stock already meets its target. That row is applicable but is not a
    trade: it must match the opening book's Fed holdings and carry no flow or cash."""

    _verify_fed_stock_bridge("baseline", [_fed_row()], opening_cb_holdings=4452.2)
    with pytest.raises(ValueError, match="opening book holdings"):
        _verify_fed_stock_bridge("baseline", [_fed_row()], opening_cb_holdings=4000.0)
    with pytest.raises(ValueError, match="moves CBOFedSecondaryPurchaseCash"):
        _verify_fed_stock_bridge(
            "baseline", [_fed_row(CBOFedSecondaryPurchaseCash=5.0)], opening_cb_holdings=4452.2
        )
    with pytest.raises(ValueError, match="nonsettling opening identity"):
        _verify_fed_stock_bridge(
            "baseline",
            [_fed_row(CBOFedSettlementScope="beneficial_holder_with_model_dirty_value_settlement")],
            opening_cb_holdings=4452.2,
        )


def test_worktree_builds_record_the_same_portable_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Governed inputs name files in sibling projects. A package built from a linked
    worktree must write them relative to the main checkout's folder, as a main-checkout
    build does, or the package carries machine-local paths and fails verification."""

    main = tmp_path / "workspace" / "project"
    worktree = main / ".worktrees" / "branch"
    worktree.mkdir(parents=True)
    monkeypatch.setattr(smoke, "PROJECT_ROOT", worktree)
    monkeypatch.setattr(smoke, "_git_output", lambda args: str(main / ".git"))
    sibling = tmp_path / "workspace" / "other" / "data" / "x.csv"
    assert smoke._portable_path(sibling) == "other/data/x.csv"
    assert smoke._portable_path(main / "data" / "inputs.json") == "data/inputs.json"
    assert smoke._portable_path(worktree / "scripts" / "run.py") == "scripts/run.py"


def test_debt_verifier_reconstructs_first_period_end_from_rolled_opening(tmp_path: Path) -> None:
    from forecast_bundle_builders import build_debt_stock_path_rows, opening_cbo_public_debt_bil
    from simulation_calendar import build_simulation_calendar
    from scripts.verify_cbo_forecast_package import _verify_debt_stock_path

    inputs = tmp_path / "forecast_inputs"
    _write_rows(inputs / "tdcsim_opening_portfolio.csv", [
        {"SecurityType": "Fixed", "FaceValue": 100, "AdjustedPrincipal": 0},
        {"SecurityType": "TIPS", "FaceValue": 10, "AdjustedPrincipal": 20},
    ])
    _write_rows(inputs / "tdcsim_cbo_fiscal_baseline.csv", [
        {"fiscal_year": 2026, "debt_held_public_end_bil": 221},
    ])
    nonmarketable, residual = 4, -3
    periods = build_simulation_calendar("2026-06-21", "2026-09-30", "daily")
    rows = build_debt_stock_path_rows(
        scenario_id="cbo_full_horizon_local_smoke_3m", periods=periods,
        opening_state_date="2026-06-21",
        opening_cbo_federal_debt_held_public_bil=opening_cbo_public_debt_bil(120, nonmarketable, residual),
        cbo_fy_end_public_debt_targets_bil={2026: 221},
        public_nonmarketable_treasury_bil=nonmarketable, non_treasury_and_definition_residual_bil=residual,
        observation_date="2026-06-18", available_date="2026-06-22",
    )
    assert rows[0]["period_end"] == "2026-06-22"
    assert rows[0]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(121 + 100 / 101)
    _write_rows(inputs / "tdcsim_debt_stock_path.csv", rows)
    _verify_debt_stock_path(tmp_path, {"date_range": {"simulation_start_date": "2026-06-21"}, "source_values": {
        "row_counts": {"period_rows": len(rows)},
        "debt_bridge": {"treasury_public_debt_bil": 104, "public_nonmarketable_treasury_bil": nonmarketable,
                        "non_treasury_and_definition_residual_bil": residual},
    }})

def test_run_verifier_checks_acquisitions_at_the_date_of_purchase(tmp_path: Path) -> None:
    from tdcsim_cbo.verifier import VerificationError, _verify_mmf_eligibility

    _write_rows(tmp_path / "tdcsim_opening_portfolio.csv", [
        {"BondID": 1, "Status": "Active", "HolderType": "Private", "HolderSubBucket": PRIVATE_SUBBUCKET_MMF,
         "SecurityType": "Fixed", "MaturityDate": "2027-07-15"},
    ])
    fields = dict.fromkeys(
        ("row_type", "source_record_date", "opening_close_date", "mmf_acquisition_rows",
         "acquisition_date", "security_type", "maturity_date"), "",
    )
    _write_rows(tmp_path / "tdcsim_opening_rollforward_diagnostics.csv", [
        {**fields, "row_type": "summary", "source_record_date": "2026-05-31", "opening_close_date": "2026-06-18",
         "mmf_acquisition_rows": 1},
        {**fields, "row_type": "mmf_acquisition", "acquisition_date": "2026-05-31",
         "security_type": "Fixed", "maturity_date": "2027-07-15"},
    ])
    with pytest.raises(VerificationError, match="ineligible pre-start money-fund acquisition on 2026-05-31"):
        _verify_mmf_eligibility(tmp_path, {"simulation": {"start_date": START, "end_date": "2026-07-01"}}, tmp_path)
