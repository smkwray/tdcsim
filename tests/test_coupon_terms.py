import pandas as pd

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
