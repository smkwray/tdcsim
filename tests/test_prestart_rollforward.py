"""Opening auction terms and coverage must agree with the source book."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from issuance_calibration import IssuanceCalibrationError, prestart_rollforward, prestart_settlements
from tdc_shared import PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF, TRANSFER_SCALABLE_COLUMNS

RECORD = "2026-05-31"
CLOSE = "2026-06-18"
AUCTION_COLUMNS = (
    "cusip", "security_type", "security_term", "issue_date", "original_issue_date", "maturity_date",
    "total_accepted", "soma_accepted", "int_rate", "price_per100", "high_investment_rate", "high_yield",
    "floating_rate", "inflation_index_security",
)


def _auction(cusip, kind, issue, maturity, accepted_bil, *, original="null", coupon_pct="null", tips="No"):
    return {
        "cusip": cusip, "security_type": kind, "security_term": "", "issue_date": issue,
        "original_issue_date": original, "maturity_date": maturity, "total_accepted": str(accepted_bil * 1e9),
        "soma_accepted": "0", "int_rate": coupon_pct, "price_per100": "99.0", "high_investment_rate": "4.0", "high_yield": "4.1",
        "floating_rate": "No", "inflation_index_security": tips,
    }


def _write_auctions(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUCTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(bond_id, cusip, issue, maturity, face, holder, subbucket="", coupon=0.0, price=0.988):
    return {
        "BondID": bond_id, "_cusip": cusip, "SecurityType": "Fixed", "IssueDate": pd.Timestamp(issue),
        "MaturityDate": pd.Timestamp(maturity), "OriginalMaturityYears": 0.3, "FaceValue": face,
        "CouponRate": coupon, "HolderType": holder, "HolderSubBucket": subbucket or pd.NA, "Status": "Active",
        "MaturityCategory": "bills", "IssuePriceRatio": price, "IssueProceeds": face * price,
    }


BOOK = pd.DataFrame([
    _row(1, "RB", "2026-03-03", "2026-06-30", 10.0, "Banks"),  # reopened in the window
    _row(2, "M1", "2026-02-03", "2026-06-02", 30.0, "Banks"),
    _row(3, "M1", "2026-02-03", "2026-06-02", 50.0, "Private", PRIVATE_SUBBUCKET_MMF),
    _row(4, "M2", "2025-12-04", "2026-06-04", 60.0, "Foreign"),
    _row(5, "N1", "2023-05-31", "2026-05-31", 40.0, "CB", coupon=0.03),  # Sunday maturity, paid 6/1
    _row(6, "N1", "2023-05-31", "2026-05-31", 20.0, "Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, coupon=0.03),
])
WINDOW = [
    _auction("OLD", "Bill", "2026-05-28", "2026-08-27", 90),  # settles before the record date
    _auction("C2", "Note", "2026-06-01", "2028-05-31", 70, coupon_pct="4.0"),
    _auction("C5", "Note", "2026-06-01", "2031-05-31", 30, coupon_pct="4.125"),
    _auction("RB", "Bill", "2026-06-02", "2026-06-30", 50, original="2026-03-03"),
    _auction("B17", "Bill", "2026-06-02", "2026-09-29", 50),
    _auction("B13", "Bill", "2026-06-04", "2026-09-03", 60),
    _auction("B26", "Bill", "2026-06-04", "2026-12-03", 40),
    _auction("CLOSE", "Bill", "2026-06-18", "2026-07-16", 28),
    _auction("NEXT", "Bill", "2026-06-22", "2026-07-20", 28),
]


def _roll(tmp_path: Path, auctions: list[dict[str, str]], *, soma_rows=()):
    settlements = prestart_settlements(
        _write_auctions(tmp_path / "auctions.csv", auctions), record_date=RECORD, opening_close_date=CLOSE,
    )
    shares = {
        "Banks": 0.25, "Foreign": 0.25, f"Private/{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}": 0.25,
        f"Private/{PRIVATE_SUBBUCKET_MMF}": 0.25,
    }
    return prestart_rollforward(
        BOOK, settlements, opening_close_date=CLOSE, record_date=RECORD, cusip_column="_cusip",
        noncb_weights={"bills": shares, "nonbills": shares},
        soma=pd.DataFrame(soma_rows, columns=["cusip", "amount_mil", "maturity_date"]), soma_as_of_date="2026-06-17",
    )


def test_actual_issue_terms_and_face_are_preserved(tmp_path) -> None:
    opening, summary, diagnostics = _roll(tmp_path, WINDOW)
    assert opening.groupby("_cusip")["FaceValue"].sum().to_dict() == pytest.approx(
        {"RB": 60, "C2": 70, "C5": 30, "B17": 50, "B13": 60, "B26": 40, "CLOSE": 28}
    )
    assert summary["opening_face_bil"] == pytest.approx(338)
    assert summary["matured_face_bil"] == pytest.approx(200)
    added = opening[opening["BondID"] > 6]
    reopened = added[added["_cusip"] == "RB"].iloc[0]
    assert reopened["IssueDate"] == pd.Timestamp("2026-03-03")
    assert reopened["IssuePriceRatio"] == pytest.approx(0.99)
    new_note = added[added["_cusip"] == "C2"].iloc[0]
    assert new_note["CouponRate"] == pytest.approx(0.04) and new_note["MaturityCategory"] == "notes"
    six_month = added[added["_cusip"] == "B26"].iloc[0]
    assert six_month["OriginalMaturityYears"] == pytest.approx(182 / 365.25)
    # The June 2 bank lot receives half its maturity and half the bank's net issuance allotment.
    bank_reopening, = added.loc[added["_cusip"].eq("RB") & added["HolderType"].eq("Banks"), "FaceValue"]
    assert bank_reopening == pytest.approx(15 + 20 * 0.5 * 0.25)
    replacement_ids = {row["replacement_bond_id"] for row in diagnostics if row["row_type"] == "replacement"}
    assert replacement_ids == set(added["BondID"])


def test_unreconstructable_windows_raise(tmp_path) -> None:
    with pytest.raises(IssuanceCalibrationError, match="must cover and extend beyond"):
        _roll(tmp_path, WINDOW[:-1])
    with pytest.raises(IssuanceCalibrationError, match="indexed security"):
        _roll(tmp_path, [*WINDOW, _auction("T5", "Note", "2026-06-15", "2031-04-15", 20, coupon_pct="1.5", tips="Yes")])
    wrong = [*WINDOW[:3], _auction("RB", "Bill", "2026-06-02", "2026-07-07", 50, original="2026-03-03"), *WINDOW[4:]]
    with pytest.raises(IssuanceCalibrationError, match="disagree on the terms of RB"):
        _roll(tmp_path, wrong)

def test_soma_transfer_keeps_reopened_bill_tranche_proceeds(tmp_path) -> None:
    opening, _, _ = _roll(tmp_path, WINDOW, soma_rows=[("RB", 30_000, "2026-06-30")])
    bill = opening.loc[opening["_cusip"].eq("RB")]
    assert bill.groupby("IssuePriceRatio")["FaceValue"].sum().to_dict() == pytest.approx({0.988: 10, 0.99: 50})
    assert bill["IssueProceeds"].sum() == pytest.approx(10 * 0.988 + 50 * 0.99)
    assert bill.loc[bill["HolderType"].eq("CB"), "FaceValue"].sum() == pytest.approx(30)


def test_a_tiny_soma_gap_merges_into_the_fed_tranche_lots_instead_of_leaving_slivers(tmp_path) -> None:
    # A few thousand dollars between the SOMA snapshot and the book must not leave
    # sliver lots: a sliver indexes or accrues by less than the accounting journal
    # records each day, so the results and the journal would disagree over a year.
    gap = 2.0e-6
    opening, _, _ = _roll(tmp_path, WINDOW, soma_rows=[("RB", (30.0 + gap) * 1000.0, "2026-06-30")])
    bill = opening.loc[opening["_cusip"].eq("RB")]
    cb = bill.loc[bill["HolderType"].eq("CB")]
    assert cb["FaceValue"].sum() == pytest.approx(30.0 + gap, abs=1e-12)
    assert cb.groupby("IssuePriceRatio").size().max() == 1
    terms = [column for column in opening.columns if column not in TRANSFER_SCALABLE_COLUMNS and column != "BondID"]
    assert not opening.duplicated(terms).any()
    assert bill["IssueProceeds"].sum() == pytest.approx(10 * 0.988 + 50 * 0.99)
