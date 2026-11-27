import pandas as pd
import pytest

from fed_purchase_policy import par_capacity
from sim_engine import _transfer_public_marketable_debt_to_cb
from tdc_shared import BOND_PORTFOLIO_COLS, PRIVATE_SUBBUCKET_MMF


DAY = pd.Timestamp("2027-01-01")


def _lot(bond_id, holder, face, days, *, tenor=0.5, security="Fixed", spread=0.002, cmb=False):
    issue_price = 0.98 if tenor <= 1 else 1.0
    return {
        "BondID": bond_id,
        "SecurityType": security,
        "IssueDate": DAY + pd.Timedelta(days=days) - pd.Timedelta(days=round(tenor * 365)),
        "MaturityDate": DAY + pd.Timedelta(days=days),
        "OriginalMaturityYears": float(tenor),
        "FaceValue": face,
        "CouponRate": 0.0 if tenor <= 1 or security == "FRN" else 0.04,
        "HolderType": holder,
        "HolderSubBucket": PRIVATE_SUBBUCKET_MMF if holder == "Private" else "",
        "Status": "Active",
        "IssuePriceRatio": issue_price,
        "IssueProceeds": face * issue_price,
        "FixedSpread": spread,
        "AccruedInterest_FRN": face * 0.001 if security == "FRN" else 0.0,
        "IsCashManagementBill": cmb,
    }


def _book(*lots):
    return pd.DataFrame(lots, columns=[*BOND_PORTFOLIO_COLS, "IsCashManagementBill"])


def _buy(book, amount, budget=None):
    summary = {}
    after, stock, sellers, private, cash = _transfer_public_marketable_debt_to_cb(
        book, amount, DAY, [0.25, 5.0], [0.04, 0.04], portfolio_row_budget=budget, operation_summary=summary
    )
    return after.set_index("BondID").sort_index(), stock, sellers, private, cash, summary


def test_purchases_preserve_pro_rata_sellers_book_stock_and_cash_in_either_row_order():
    book = _book(_lot(1, "Banks", 60.0, 80), _lot(2, "Private", 10.0, 80), _lot(3, "Foreign", 30.0, 80))
    after, stock, sellers, private, cash, operation = _buy(book, 12.0, budget=4)
    reversed_after, *other = _buy(book.iloc[::-1].reset_index(drop=True), 12.0, budget=4)
    pd.testing.assert_frame_equal(after, reversed_after)
    assert other == [stock, sellers, private, cash, operation]
    assert after.loc[[1, 2, 3], "FaceValue"].tolist() == pytest.approx([52.8, 8.8, 26.4])
    assert stock == pytest.approx(12.0)
    assert after.loc[4, "FaceValue"] == pytest.approx(stock)
    assert after["FaceValue"].sum() == pytest.approx(book["FaceValue"].sum())
    assert after["IssueProceeds"].sum() == pytest.approx(book["IssueProceeds"].sum())
    assert sellers["Banks"] == pytest.approx(0.60 * cash)
    assert sellers["Foreign"] == pytest.approx(0.30 * cash)
    assert private[PRIVATE_SUBBUCKET_MMF] == pytest.approx(0.10 * cash)
    assert sum(sellers.values()) == pytest.approx(cash)
    with pytest.raises(RuntimeError, match="Portfolio-row budget"):
        _buy(book, 12.0, budget=3)


def test_sector_weights_water_fill_at_issue_cap_then_use_other_bill_sector():
    book = _book(
        _lot(1, "Banks", 40.0, 60), _lot(2, "CB", 60.0, 60),
        _lot(3, "Banks", 100.0, 85), _lot(4, "Banks", 100.0, 155),
        _lot(5, "CB", 0.1, 60),
    )
    after, *_cash, operation = _buy(book, 25.0, budget=7)
    short = operation["sectors"]["bills_1_4m"]
    long = operation["sectors"]["bills_4_12m"]
    assert short["par"] == pytest.approx(17.5 + 0.005 * 100.1)
    assert short["capacity_binds"] == 2
    assert long["par"] == pytest.approx(25.0 - short["par"])
    assert after.loc[2, "FaceValue"] == pytest.approx(60.1 + 0.005 * 100.1)
    assert 5 not in after.index
    assert after["FaceValue"].sum() == pytest.approx(book["FaceValue"].sum())


@pytest.mark.parametrize(("bill", "cb", "cap"), [
    (True, 0, 17.5), (True, 15, 2.5), (True, 33, 2), (True, 34, 1),
    (True, 59.7, 0.5), (True, 69.9, 0.1), (True, 70, 0), (True, 72, 0),
    (False, 0, 35), (False, 30, 5), (False, 46, 4), (False, 48, 2.5),
    (False, 59, 1), (False, 69.9, 0.1), (False, 70, 0), (False, 72, 0),
])
def test_per_issue_operation_limits_use_pre_operation_ownership(bill, cb, cap):
    assert par_capacity(100.0, cb, bill=bill) == pytest.approx(cap)


def test_coupon_fallback_only_after_bills_exhausted_and_exclusions_apply():
    book = _book(
        _lot(1, "Banks", 100.0, 28), _lot(2, "Banks", 100.0, 50, cmb=True),
        _lot(3, "Banks", 100.0, 84), _lot(4, "Banks", 100.0, 500, tenor=2),
        _lot(5, "Banks", 100.0, 600, tenor=2),
        _lot(6, "Banks", 100.0, 700, tenor=2, security="FRN"),
    )
    after, *_cash, operation = _buy(book, 17.0)
    assert list(operation["sectors"]) == ["bills_1_4m"]
    assert after.loc[[1, 2, 4, 5, 6], "FaceValue"].eq(100).all()
    after, *_cash, operation = _buy(book, 30.0)
    assert operation["sectors"]["bills_1_4m"]["par"] == pytest.approx(17.5)
    assert operation["sectors"]["coupons_1_18m"]["par"] == pytest.approx(6.25)
    assert operation["sectors"]["coupons_18_36m"]["par"] == pytest.approx(6.25)
    # The newest nominal 2-year note is on the run, even though an older one is eligible.
    assert after.loc[[1, 2, 5], "FaceValue"].eq(100).all()
    assert after.loc[[4, 6], "FaceValue"].tolist() == pytest.approx([93.75, 93.75])
    before = book.copy(deep=True)
    with pytest.raises(RuntimeError, match=r"capacity exhausted.*shortfall=12.500000000000"):
        _buy(book, 100.0)
    pd.testing.assert_frame_equal(book, before)

def test_coupon_purchase_capacity_and_sellers_follow_issue_id_not_matching_terms():
    book = _book(
        {**_lot(1, "CB", 70.0, 500, tenor=5), "IssueID": "NOTE-A"},
        {**_lot(2, "Banks", 30.0, 500, tenor=5), "IssueID": "NOTE-A"},
        {**_lot(3, "Foreign", 100.0, 500, tenor=7), "IssueID": "NOTE-B"},
        _lot(4, "Banks", 100.0, 1400, tenor=5),
        _lot(5, "Banks", 100.0, 1500, tenor=7),
    )
    after, stock, sellers, _, cash, _ = _buy(book, 5.0)
    assert stock == pytest.approx(5)
    assert after.loc[1, "FaceValue"] == pytest.approx(70)
    assert after.loc[2, "FaceValue"] == pytest.approx(30)
    assert after.loc[3, "FaceValue"] == pytest.approx(95)
    assert sellers["Banks"] == 0
    assert sellers["Foreign"] == pytest.approx(cash)
    assert after.loc[6, "IssueID"] == "NOTE-B"
