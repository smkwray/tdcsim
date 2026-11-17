"""Rolling the opening book from its record date to the simulation start.

The source book is Treasury's month-end statement; securities that mature before the
simulation starts must be replaced the way Treasury actually refinanced them. If the
rollforward piled the whole window's maturities onto one security or one date, the opening
book and the first months of every run would show a payment spike no Treasury calendar has.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from issuance_calibration import IssuanceCalibrationError, prestart_rollforward, prestart_settlements

RECORD = "2026-05-31"
START = "2026-06-21"
AUCTION_COLUMNS = (
    "cusip", "security_type", "security_term", "issue_date", "original_issue_date", "maturity_date",
    "total_accepted", "int_rate", "price_per100", "high_investment_rate", "high_yield",
    "floating_rate", "inflation_index_security",
)


def _auction(cusip, kind, issue, maturity, accepted_bil, *, original="null", coupon_pct="null", tips="No"):
    return {
        "cusip": cusip, "security_type": kind, "security_term": "", "issue_date": issue,
        "original_issue_date": original, "maturity_date": maturity, "total_accepted": str(accepted_bil * 1e9),
        "int_rate": coupon_pct, "price_per100": "99.0", "high_investment_rate": "4.0", "high_yield": "4.1",
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
    _row(3, "M1", "2026-02-03", "2026-06-02", 50.0, "Private", "mmf_cash_fund_route"),
    _row(4, "M2", "2025-12-04", "2026-06-04", 60.0, "Foreign"),
    _row(5, "N1", "2023-05-31", "2026-05-31", 40.0, "CB", coupon=0.03),  # Sunday maturity, paid 6/1
    _row(6, "N1", "2023-05-31", "2026-05-31", 20.0, "Private", "domestic_nonbank_deposit_funded", coupon=0.03),
])
WINDOW = [
    _auction("OLD", "Bill", "2026-05-28", "2026-08-27", 90),  # settles before the record date
    _auction("C2", "Note", "2026-06-01", "2028-05-31", 70, coupon_pct="4.0"),
    _auction("C5", "Note", "2026-06-01", "2031-05-31", 30, coupon_pct="4.125"),
    _auction("RB", "Bill", "2026-06-02", "2026-06-30", 50, original="2026-03-03"),
    _auction("B17", "Bill", "2026-06-02", "2026-09-29", 50),
    _auction("B13", "Bill", "2026-06-04", "2026-09-03", 60),
    _auction("B26", "Bill", "2026-06-04", "2026-12-03", 40),
]


def _roll(tmp_path: Path, auctions: list[dict[str, str]], book: pd.DataFrame = BOOK):
    settlements = prestart_settlements(_write_auctions(tmp_path / "auctions.csv", auctions), record_date=RECORD,
                                       simulation_start_date=START)
    return prestart_rollforward(book, settlements, simulation_start_date=START, next_bond_id=100, cusip_column="_cusip")


def test_matured_securities_are_reinvested_in_what_treasury_issued_on_their_payment_date(tmp_path) -> None:
    kept, added = _roll(tmp_path, WINDOW)
    assert kept["BondID"].tolist() == [1]
    # Each matured row is paid on the first settlement on or after its maturity and spread
    # over that day's issues by accepted amount: the Sunday note maturity is paid Monday.
    paid = added.groupby("_SourceBondID")["_PaymentDate"].first().dt.strftime("%Y-%m-%d").to_dict()
    assert paid == {2: "2026-06-02", 3: "2026-06-02", 4: "2026-06-04", 5: "2026-06-01", 6: "2026-06-01"}
    banks_m1 = added[added["_SourceBondID"] == 2].set_index("_cusip")["FaceValue"]
    assert banks_m1.to_dict() == pytest.approx({"RB": 15.0, "B17": 15.0})
    # Face and each holder's face are unchanged, and holders stay with their own securities.
    assert added["FaceValue"].sum() == pytest.approx(BOOK.loc[BOOK["BondID"] > 1, "FaceValue"].sum())
    assert added.groupby("HolderType")["FaceValue"].sum().to_dict() == pytest.approx(
        {"Banks": 30.0, "Private": 70.0, "Foreign": 60.0, "CB": 40.0})
    # A reopened CUSIP takes the book's own terms; a new issue takes the auction's.
    reopened = added[added["_cusip"] == "RB"].iloc[0]
    assert reopened["IssueDate"] == pd.Timestamp("2026-03-03") and reopened["IssuePriceRatio"] == 0.988
    new_note = added[added["_cusip"] == "C2"].iloc[0]
    assert new_note["CouponRate"] == pytest.approx(0.04) and new_note["MaturityCategory"] == "notes"
    six_month = added[added["_cusip"] == "B26"].iloc[0]
    assert six_month["OriginalMaturityYears"] == pytest.approx(182 / 365.25)
    assert sorted(added["BondID"]) == list(range(100, 100 + len(added)))


def test_the_rollforward_never_concentrates_more_than_one_days_maturities_on_a_date(tmp_path) -> None:
    # Here every pre-start issue matures on one date, which would put the whole window's
    # maturities, 200 billion, where the largest single day that matured was 80 billion.
    piled = [
        _auction("X1", "Bill", "2026-06-01", "2026-09-21", 10),
        _auction("X2", "Bill", "2026-06-02", "2026-09-21", 10),
        _auction("X3", "Bill", "2026-06-04", "2026-09-21", 10),
    ]
    with pytest.raises(IssuanceCalibrationError, match="more than the 80.0 billion"):
        _roll(tmp_path, piled)
    # Treasury's own spread of issues stays within that bound.
    _, added = _roll(tmp_path, WINDOW)
    assert added.groupby("MaturityDate")["FaceValue"].sum().max() <= 80.0 + 1e-9


def test_unreconstructable_windows_raise(tmp_path) -> None:
    # A maturity with no settlement after it, an indexed issue, or an auction that
    # contradicts the book's terms for a CUSIP is a gap, not something to guess around.
    with pytest.raises(IssuanceCalibrationError, match="no Treasury issue settles between 2026-06-04"):
        _roll(tmp_path, WINDOW[:5])
    with pytest.raises(IssuanceCalibrationError, match="indexed security"):
        _roll(tmp_path, [*WINDOW, _auction("T5", "Note", "2026-06-15", "2031-04-15", 20, coupon_pct="1.5", tips="Yes")])
    wrong = [*WINDOW[:3], _auction("RB", "Bill", "2026-06-02", "2026-07-07", 50, original="2026-03-03"), *WINDOW[4:]]
    with pytest.raises(IssuanceCalibrationError, match="disagree on the terms of RB"):
        _roll(tmp_path, wrong)
