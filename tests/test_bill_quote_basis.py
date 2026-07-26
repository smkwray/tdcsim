"""Treasury bill quote-basis conversion anchors.

CBO forecasts the 3-month bill on the secondary-market discount basis the Federal Reserve
publishes; TDCSim anchors a surface declared to be a nominal *par-yield* surface. Par yields are
bond-equivalent, so the raw quote must be converted before it can serve as a par anchor.
"""

import pytest

from bill_quote_basis import discount_price_ratio, discount_rate_to_investment_rate


def test_discount_price_matches_treasury_formula():
    """P = 1 - dn/360, per 31 CFR 356 Appendix B."""

    assert discount_price_ratio(0.045, 91) * 100 == pytest.approx(98.862500, abs=1e-6)
    assert discount_price_ratio(0.045, 182) * 100 == pytest.approx(97.725000, abs=1e-6)
    assert discount_price_ratio(0.02, 91) * 100 == pytest.approx(99.494444, abs=1e-6)


def test_short_bill_investment_rate_conversion():
    """For a bill of half a year or less: i = 365d / (360 - dn)."""

    assert discount_rate_to_investment_rate(0.045, 91) == pytest.approx(0.046150, abs=1e-6)
    assert discount_rate_to_investment_rate(0.045, 182) == pytest.approx(0.046687, abs=1e-6)
    assert discount_rate_to_investment_rate(0.03, 91) == pytest.approx(0.030649, abs=1e-6)


def test_long_bill_uses_the_coupon_period_form():
    """Beyond 182 days the bill spans a notional coupon period.

    Extending the short-bill formula mechanically overstates the rate exactly where the model's
    one-year bucket sits, so the long form must solve the semiannual-plus-stub identity instead.
    """

    long_form = discount_rate_to_investment_rate(0.045, 364)
    naive_short_form = (365.0 * 0.045) / (360.0 - 0.045 * 364)

    assert long_form == pytest.approx(0.047243, abs=1e-6)
    assert long_form < naive_short_form
    # The identity it solves: P = 1 / [(1 + i/2)(1 + (n/365 - 0.5) i)]
    stub = 364 / 365.0 - 0.5
    implied = 1.0 / ((1.0 + long_form / 2.0) * (1.0 + stub * long_form))
    assert implied == pytest.approx(discount_price_ratio(0.045, 364), abs=1e-9)


def test_conversion_is_monotone_across_the_bill_tenors_the_model_issues():
    """The model issues 0.25/0.5/1.0-year buckets; the conversion must stay ordered and sane."""

    rates = [discount_rate_to_investment_rate(0.045, days) for days in (91, 182, 364)]
    assert rates == sorted(rates)
    assert all(0.045 < rate < 0.05 for rate in rates)


def test_raw_discount_quote_is_not_used_as_a_par_yield():
    """The conversion must actually move the number - a no-op would be the original defect."""

    for rate in (0.03, 0.045, 0.0525):
        converted = discount_rate_to_investment_rate(rate, 91)
        assert converted > rate
        assert (converted - rate) * 10000 > 5.0  # at least 5bp at these levels


def test_zero_and_invalid_inputs_fail_closed():
    assert discount_rate_to_investment_rate(0.0, 91) == 0.0
    with pytest.raises(ValueError):
        discount_rate_to_investment_rate(0.045, 0)
    with pytest.raises(ValueError):
        discount_price_ratio(0.045, -1)
