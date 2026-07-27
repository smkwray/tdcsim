"""Treasury bill quote-basis conversions.

CBO forecasts the 3-month Treasury bill on the *secondary-market discount basis* the Federal
Reserve publishes, but TDCSim anchors a surface declared
``nominal_par_yield_coupon_setting_surface``. Par yields are bond-equivalent, so the raw quote
cannot be used as a par anchor without conversion: at a 4.5% discount rate the investment-basis
equivalent is 4.6150%, understating the 3-month par point by about 11.5 basis points and
propagating through the short end via the curve's log-tenor blend.

Formulas are Treasury's own (31 CFR 356 Appendix B).
"""

from __future__ import annotations

# CBO's 3-month bill forecast is a 13-week quote; 91 days is its actual term.
CBO_3M_BILL_DAYS = 91.0

# Treasury quotes bill discount rates on a 360-day year and converts to an investment
# (bond-equivalent) rate on a 365-day year.
_DISCOUNT_YEAR_DAYS = 360.0
_INVESTMENT_YEAR_DAYS = 365.0
_HALF_YEAR_DAYS = 182.0


def discount_rate_to_investment_rate(discount_rate: float, days_to_maturity: float) -> float:
    """Convert a bill discount rate to its bond-equivalent investment rate.

    For a bill of half a year or less Treasury uses ``i = 365d / (360 - dn)``. Beyond that the
    security spans a coupon period, so the conversion takes the quadratic form rather than the
    short-bill formula extended mechanically - which would drift exactly where the model's
    one-year bucket sits.
    """

    rate = float(discount_rate)
    days = float(days_to_maturity)
    if days <= 0.0:
        raise ValueError("bill days_to_maturity must be positive")
    if rate == 0.0:
        return 0.0

    if days <= _HALF_YEAR_DAYS:
        denominator = _DISCOUNT_YEAR_DAYS - rate * days
        if denominator <= 0.0:
            raise ValueError("bill discount rate is too large for its maturity")
        return (_INVESTMENT_YEAR_DAYS * rate) / denominator

    price_ratio = discount_price_ratio(rate, days)
    # Beyond half a year the bill spans a notional coupon period, so its bond-equivalent rate
    # is the i solving  P = 1 / [(1 + i/2)(1 + (n/365 - 0.5) i)]  - semiannual compounding to
    # the first notional coupon, then simple interest over the stub. Extending the short-bill
    # formula here instead would overstate the rate exactly where the model's one-year bucket
    # sits. Solved by bisection: the function is monotone in i, so this is exact to tolerance
    # and avoids a hand-transcribed closed form going quietly wrong.
    stub = days / _INVESTMENT_YEAR_DAYS - 0.5
    low, high = -0.99, 1.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        implied = 1.0 / ((1.0 + mid / 2.0) * (1.0 + stub * mid))
        if implied > price_ratio:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def discount_price_ratio(discount_rate: float, days_to_maturity: float) -> float:
    """Price per unit of face for a bill quoted on the discount basis: ``P = 1 - dn/360``."""

    rate = float(discount_rate)
    days = float(days_to_maturity)
    if days <= 0.0:
        raise ValueError("bill days_to_maturity must be positive")
    ratio = 1.0 - rate * days / _DISCOUNT_YEAR_DAYS
    if ratio <= 0.0:
        raise ValueError("bill discount rate implies a nonpositive price")
    return ratio


__all__ = [
    "CBO_3M_BILL_DAYS",
    "discount_rate_to_investment_rate",
    "discount_price_ratio",
]
