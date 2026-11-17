"""Package-verifier checks on the opening book and the opening Fed identity.

A package built from any correctly sourced opening book must pass, so the verifier
rebuilds the book's counts from the packaged source rather than pinning one book's
counts; and the engine's opening Fed row (a nonsettling statement that the opening
stock meets the Fed target) must be accepted only when it really moves nothing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.run_cbo_forecast_smoke as smoke
from scripts.verify_cbo_forecast_package import (
    _OPENING_FED_FLOW_COLUMNS,
    _verify_fed_stock_bridge,
    _verify_opening_counts_against_source_book,
)

START = "2026-06-21"


def _write_source_book(root: Path, rows: list[dict[str, str]]) -> None:
    path = root / "sources" / "opening" / "tdcsim_initial_portfolio_cohorts.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "BondID": str(index)} for index, row in enumerate(rows, start=1))


def _book_row(security_type: str, maturity: str, cusip: str) -> dict[str, str]:
    return {
        "BondID": "", "SecurityType": security_type, "IssueDate": "2026-01-02", "MaturityDate": maturity,
        "FaceValue": "1.0", "CouponRate": "0.0", "HolderType": "Banks", "HolderSubBucket": "", "Status": "Active",
        "cusip_or_synthetic_id": cusip,
    }


def _write_auctions(root: Path) -> None:
    """Two bills settle on each pre-start payment date, so each matured row becomes two."""

    path = root / "sources" / "fiscaldata" / "auctions_query.csv"
    path.parent.mkdir(parents=True)
    columns = ["cusip", "security_type", "issue_date", "original_issue_date", "maturity_date", "total_accepted",
               "int_rate", "price_per100", "high_investment_rate", "high_yield", "floating_rate",
               "inflation_index_security"]
    issues = [("A4", "2026-06-11", "2026-07-09"), ("A13", "2026-06-11", "2026-09-10"),
              ("B4", "2026-06-18", "2026-07-16"), ("B13", "2026-06-18", "2026-09-17")]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for cusip, issue, maturity in issues:
            writer.writerow([cusip, "Bill", issue, "null", maturity, "5e10", "null", "99.0", "4.0", "null", "No", "No"])


def _manifest(**counts: int) -> dict:
    return {
        "date_range": {"simulation_start_date": START},
        "opening_portfolio_construction": {"record_date": "2026-05-31", **counts},
    }


def test_opening_counts_follow_the_packaged_source_book(tmp_path: Path) -> None:
    """Two bills mature before the start and roll forward into two issues each; two TIPS
    rows share a CUSIP. The manifest's counts must be those of this book, whatever book it is."""

    _write_source_book(
        tmp_path,
        [
            _book_row("Fixed", "2026-06-10", "B1"),
            _book_row("Fixed", "2026-06-18", "B2"),
            _book_row("Fixed", "2027-01-01", "N1"),
            _book_row("TIPS", "2030-01-15", "T1"),
            _book_row("TIPS", "2030-01-15", "T1"),
            _book_row("FRN", "2027-10-31", "F1"),
        ],
    )
    _write_auctions(tmp_path)
    counts = {
        "source_rows_loaded": 6,
        "prestart_matured_rows": 2,
        "prestart_rollforward_rows": 4,
        "row_count": 8,
        "tips_rows_enriched": 2,
        "tips_unique_cusips_enriched": 1,
        "frn_rows_enriched": 1,
        "frn_unique_cusips_enriched": 1,
    }
    _verify_opening_counts_against_source_book(tmp_path, _manifest(**counts))
    # A book that dropped a TIPS row while enriching is caught.
    with pytest.raises(ValueError, match="tips_rows_enriched"):
        _verify_opening_counts_against_source_book(tmp_path, _manifest(**{**counts, "tips_rows_enriched": 1}))
    with pytest.raises(ValueError, match="prestart_rollforward_rows"):
        _verify_opening_counts_against_source_book(tmp_path, _manifest(**{**counts, "prestart_rollforward_rows": 2}))


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
