import pandas as pd
import pytest

from sim_pricing import get_coupon_dates_in_period


def test_coupon_schedule_uses_first_interest_payment_date_not_issue_day():
    coupons = get_coupon_dates_in_period(
        pd.Timestamp("2026-01-31"),
        pd.Timestamp("2031-01-15"),
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-31"),
        frequency=2,
        first_interest_payment_date=pd.Timestamp("2026-07-15"),
        interest_payment_frequency=2,
    )

    assert coupons == [pd.Timestamp("2026-07-15")]


def test_coupon_schedule_falls_back_to_maturity_cycle_for_legacy_rows():
    coupons = get_coupon_dates_in_period(
        pd.Timestamp("2026-01-31"),
        pd.Timestamp("2031-01-15"),
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-31"),
        frequency=2,
    )

    assert coupons == [pd.Timestamp("2026-07-15")]


def _daily_coupon_dates(issue, maturity, first_interest, frequency):
    from sim_engine import _daily_coupon_due_indices

    tranche = pd.DataFrame(
        {
            "IssueDate": [pd.Timestamp(issue)],
            "MaturityDate": [pd.Timestamp(maturity)],
            "FirstInterestPaymentDate": [pd.Timestamp(first_interest)],
            "InterestPaymentFrequency": [frequency],
        }
    )
    return [
        day
        for day in pd.date_range(issue, maturity, freq="D")
        if _daily_coupon_due_indices(tranche, day, frequency=frequency)
    ]


@pytest.mark.parametrize(
    ("issue", "maturity", "first_interest", "frequency", "expected"),
    [
        # A month-end note whose first coupon falls on February 28 pays on August 31,
        # not August 28; otherwise it pays a regular coupon three days before the final
        # coupon at maturity and overstates interest by half a year's coupon.
        ("2026-08-31", "2028-08-31", "2027-02-28", 2, ["2027-02-28", "2027-08-31", "2028-02-29"]),
        ("2026-10-31", "2028-10-31", "2027-04-30", 2, ["2027-04-30", "2027-10-31", "2028-04-30"]),
        # Quarterly FRNs that mature on a month end follow the same rule.
        ("2026-04-30", "2027-04-30", "2026-07-31", 4, ["2026-07-31", "2026-10-31", "2027-01-31"]),
        # Mid-month schedules are unchanged.
        ("2026-05-15", "2028-05-15", "2026-11-15", 2, ["2026-11-15", "2027-05-15", "2027-11-15"]),
    ],
)
def test_regular_coupons_follow_the_maturity_date_so_the_final_coupon_is_paid_once(
    issue, maturity, first_interest, frequency, expected
):
    expected_dates = [pd.Timestamp(day) for day in expected]
    period_dates = get_coupon_dates_in_period(
        pd.Timestamp(issue),
        pd.Timestamp(maturity),
        pd.Timestamp(issue),
        pd.Timestamp(maturity),
        first_interest_payment_date=pd.Timestamp(first_interest),
        interest_payment_frequency=frequency,
    )

    # The maturity handler pays the final coupon, so regular coupons stop one full
    # period before maturity on both the daily and the multi-day step paths.
    assert period_dates == expected_dates
    assert _daily_coupon_dates(issue, maturity, first_interest, frequency) == expected_dates
