
"""Pricing and coupon-schedule utilities for Treasury securities."""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline, PchipInterpolator

from evaluated_nominal_curve import CurveContractError, EvaluatedNominalShock
from tdc_shared import DAYS_PER_YEAR_ACTUAL, TGA_FLOOR_TOLERANCE

# Appendix B counts coupon periods on a 365-day nominal year; actual/actual applies within.
_DAYS_PER_YEAR_NOMINAL = 365.0

def _is_bill_like_fixed(security_type, original_maturity_years, coupon_rate):
    """Return True for zero-coupon fixed-rate securities at or below the bill cutoff."""
    if security_type != 'Fixed':
        return False
    try:
        maturity_val = float(original_maturity_years)
    except Exception:
        maturity_val = np.nan
    try:
        coupon_val = float(coupon_rate)
    except Exception:
        coupon_val = np.nan
    return not pd.isna(maturity_val) and maturity_val <= 1.0 + TGA_FLOOR_TOLERANCE and (pd.isna(coupon_val) or coupon_val <= TGA_FLOOR_TOLERANCE)

def calculate_issue_price_ratio(
    security_type,
    maturity_years,
    coupon_rate,
    yield_at_issuance,
    *,
    apply_rate_floor=True,
):
    """
    Returns the cash-proceeds-to-face ratio at auction.

    Coupon-bearing nominal notes and bonds and TIPS are priced as the present value of their
    contractual cash flows at the auction yield. Because Treasury sets the coupon on a discrete
    1/8-point grid, the coupon generally differs from the stop-out yield, so the price is
    generally not par: a coupon below the yield issues at a discount, above it at a premium.

    Bills are issued at a discount using the simple zero-coupon convention used elsewhere in
    this module. That is a declared approximation to Treasury's discount/investment-rate
    quoting, not an implementation of it.

    FRNs and nonmarketables remain at par: an FRN's coupon resets to its index, so it prices at
    par at issue by construction, and nonmarketables are not auctioned.
    """
    if _is_bill_like_fixed(security_type, maturity_years, coupon_rate):
        eff_yield = 0.0 if pd.isna(yield_at_issuance) else float(yield_at_issuance)
        if apply_rate_floor:
            eff_yield = max(0.0, eff_yield)
        maturity_val = 0.0 if pd.isna(maturity_years) else max(0.0, float(maturity_years))
        try:
            return max(TGA_FLOOR_TOLERANCE, 1.0 / (1.0 + eff_yield) ** maturity_val)
        except Exception:
            return 1.0
    if security_type in ('TIPS', 'Fixed'):
        return calculate_coupon_security_issue_price_ratio(
            maturity_years,
            coupon_rate,
            yield_at_issuance,
        )
    return 1.0


def calculate_auction_coupon_rate(
    yield_at_issuance,
    *,
    minimum_coupon_rate=0.00125,
    coupon_increment=0.00125,
):
    """Return the Treasury auction coupon: the 1/8-point grid rate closest to but not above par.

    Treasury establishes the coupon that gives a price closest to, but not above, par at the
    accepted yield, subject to a 0.125% minimum (31 CFR 356 App. B). For ordinary positive
    yields that is exactly flooring the yield onto the 1/8-point grid: a coupon at or below the
    yield prices at or below par, so the largest grid rate not exceeding the yield is the one
    closest to par from below. Applies to nominal notes and bonds and to TIPS real coupons.
    """

    if pd.isna(yield_at_issuance):
        return max(0.0, float(minimum_coupon_rate))
    yld = float(yield_at_issuance)
    increment = max(TGA_FLOOR_TOLERANCE, float(coupon_increment))
    rounded = np.floor(max(yld, 0.0) / increment) * increment
    return max(float(minimum_coupon_rate), float(rounded))


def calculate_coupon_security_issue_price_ratio(
    maturity_years,
    coupon_rate,
    yield_at_issuance,
    *,
    frequency=2,
):
    """Price a new coupon security as the PV of its contractual cash flows per unit of face.

    Semiannual discounting per 31 CFR 356 Appendix B. Nominal notes and bonds discount nominal
    cash flows at the nominal auction yield; TIPS discount real cash flows at the real yield,
    per dollar of original principal. The mathematics is identical - only the units differ.
    """

    if pd.isna(maturity_years) or float(maturity_years) <= TGA_FLOOR_TOLERANCE:
        return 1.0
    if pd.isna(yield_at_issuance):
        return 1.0
    maturity_val = max(0.0, float(maturity_years))
    freq = max(1, int(round(float(frequency))))
    periods = max(1, int(round(maturity_val * freq)))
    yld_per_period = float(yield_at_issuance) / freq
    if yld_per_period <= -1.0 + 1e-9:
        yld_per_period = -1.0 + 1e-9
    coupon_per_period = max(0.0, float(coupon_rate)) / freq
    if abs(yld_per_period) <= 1e-12:
        price = 1.0 + coupon_per_period * periods
    else:
        discounts = np.array([(1.0 + yld_per_period) ** -period for period in range(1, periods + 1)])
        price = coupon_per_period * float(discounts.sum()) + float(discounts[-1])
    return round(max(TGA_FLOOR_TOLERANCE, float(price)), 8)


def value_treasury_security(
    *,
    settlement_date,
    maturity_date,
    coupon_rate,
    discount_yield,
    security_type='Fixed',
    face_value=1.0,
    adjusted_principal=None,
    original_principal=None,
    accrued_frn=None,
    issue_date=None,
    first_interest_payment_date=None,
    frequency=2,
    projected_adjusted_principal_at_maturity=None,
    nominal_discount_yield=None,
):
    """Value one Treasury security at a settlement date, for callers using this kernel.

    Not the only valuation path in the project: ordinary preference trading still prices
    through ``calculate_bond_market_price``, which discounts TIPS at a nominal yield. That
    path is disabled in the CBO runner. Do not describe this function as project-wide.

    Discounts the *dated* remaining cash flows at the semiannual bond-equivalent convention
    of 31 CFR 356 Appendix B, including its fractional stub exponent v**(r/s) for a
    settlement date inside a coupon period. That combination matters:

    * The convention must be semiannual because the model's yields come from a Treasury par
      curve (`nominal_par_yield_coupon_setting_surface`), and par yields are semiannual
      bond-equivalent by definition. Annual-effective discounting of the same numeric yield
      misprices by roughly duration * y**2 / 4 — about 24bp on a 7-year note at 4%, 141bp on
      a 12.5-year at 8.84%.
    * The periods must be dated rather than rounded to whole halves, because rounding
      structurally deletes accrued interest from the price — up to a full half-coupon, 1.125
      per 100 face on a 4.5% coupon.

    A TIPS is priced as two legs, because they are not discountable at the same rate. The
    indexed base carries *real* cash flows — the model holds adjusted principal at today's
    index ratio rather than projecting it forward — so `discount_yield` must be a **real**
    yield at the security's *remaining* maturity. The deflation floor is a *nominal* dollar
    payoff, max(original - adjusted-at-maturity, 0), and takes `nominal_discount_yield`.
    Discounting real flows at a nominal rate misprices by roughly the inflation wedge times
    duration: a 10-year TIPS with a 1.25% real coupon and adjusted principal 105 prices at
    77.76 against a correct 105.00.

    Coupons continue to accrue on unfloored adjusted principal even when the floor binds, and
    the engine still pays max(original, realized adjusted) at actual maturity.

    The floor is a **deterministic scenario approximation with no option time value**: it
    values the payoff implied by `projected_adjusted_principal_at_maturity` rather than an
    expectation over an inflation distribution. Reported as `deflation_floor_basis`. Pricing
    the embedded option properly would need an inflation volatility surface or a joint
    nominal-real term-structure model, neither of which this project holds.

    Negative yields are honoured. TIPS real yields were materially negative in 2020-22, and
    the issuance path already supports negative-yield premium pricing.

    Returns `clean`, `accrued`, `dirty`, plus the floor value and its basis. Settlement of an
    actual purchase or sale is the *dirty* value — that invoice amount becomes reserves and
    deposits.
    """

    if pd.isna(maturity_date) or pd.isna(settlement_date):
        raise ValueError('Treasury valuation requires settlement and maturity dates.')
    if settlement_date >= maturity_date:
        raise ValueError('Treasury valuation requires a settlement date before maturity.')
    if pd.isna(face_value) or float(face_value) <= TGA_FLOOR_TOLERANCE:
        raise ValueError('Treasury valuation requires a positive face value.')

    face = float(face_value)
    coupon = 0.0 if pd.isna(coupon_rate) else max(0.0, float(coupon_rate))

    # Nonmarketables are not traded and FRNs reset to their index, so both sit at par plus
    # any accrued. Neither is a discounting question.
    if security_type in ('NonMarketable', 'FRN'):
        accrued = 0.0
        if security_type == 'FRN' and accrued_frn is not None and not pd.isna(accrued_frn):
            accrued = max(0.0, float(accrued_frn))
        return {'clean': face, 'accrued': accrued, 'dirty': face + accrued}

    if pd.isna(discount_yield):
        raise ValueError('Treasury valuation requires a discount yield for a coupon security.')

    principal_at_maturity = face
    principal_for_coupons = face
    floor_excess = 0.0
    if security_type == 'TIPS':
        adj = float(adjusted_principal) if adjusted_principal not in (None,) and not pd.isna(adjusted_principal) and float(adjusted_principal) > 0 else face
        orig = float(original_principal) if original_principal not in (None,) and not pd.isna(original_principal) and float(original_principal) > 0 else face
        # A TIPS decomposes into a floorless indexed bond plus a nominal contingent claim on
        # terminal deflation. The two legs are NOT discountable at the same rate: the indexed
        # base carries real cash flows and must be discounted at a real yield, while the floor
        # top-up max(original - adjusted_at_maturity, 0) is a nominal dollar payoff and takes
        # the nominal maturity discount factor. Collapsing them into one
        # max(original, adjusted) term discounted at a single rate misprices whichever leg
        # gets the wrong curve.
        principal_at_maturity = adj
        principal_for_coupons = adj
        projected = adj if projected_adjusted_principal_at_maturity is None or pd.isna(projected_adjusted_principal_at_maturity) else float(projected_adjusted_principal_at_maturity)
        floor_excess = max(0.0, orig - projected)

    freq = max(1, int(round(float(frequency))))
    yld_per_period = float(discount_yield) / freq
    if yld_per_period <= -1.0 + 1e-9:
        yld_per_period = -1.0 + 1e-9
    discount_base = 1.0 + yld_per_period

    remaining = _remaining_coupon_dates(
        settlement_date=settlement_date,
        maturity_date=maturity_date,
        issue_date=issue_date,
        first_interest_payment_date=first_interest_payment_date,
        frequency=freq,
    )
    coupon_payment = principal_for_coupons * coupon / freq

    # Appendix B indexes cash flows by WHOLE coupon periods, with a single fractional stub
    # r/s for the part-period between settlement and the next coupon. Counting every flow in
    # days against a nominal period length instead would drift, because real semiannual
    # periods are 181-184 days: a par bond would not price at exactly 100.
    prior_coupon = remaining[0] - relativedelta(months=12 // freq) if remaining else settlement_date
    span_days = max(1, (remaining[0] - prior_coupon).days) if remaining else 1
    stub = (remaining[0] - settlement_date).days / span_days if remaining else 0.0

    clean_pv = 0.0
    for index, _pay_date in enumerate(remaining):
        try:
            clean_pv += coupon_payment / discount_base ** (stub + index)
        except (OverflowError, ValueError, ZeroDivisionError):
            raise ValueError('Treasury valuation overflowed while discounting a coupon.')
    terminal_exponent = stub + max(0, len(remaining) - 1)
    try:
        clean_pv += principal_at_maturity / discount_base ** terminal_exponent
    except (OverflowError, ValueError, ZeroDivisionError):
        raise ValueError('Treasury valuation overflowed while discounting principal.')

    # The deflation floor is a nominal payoff, so it takes the nominal maturity discount
    # factor rather than the real one used for the indexed base above. Deterministic
    # scenario approximation: it carries no option time value, which is a stated limitation.
    floor_value = 0.0
    if floor_excess > TGA_FLOOR_TOLERANCE:
        nominal_yield = float(discount_yield) if nominal_discount_yield is None or pd.isna(nominal_discount_yield) else float(nominal_discount_yield)
        nominal_per_period = max(-1.0 + 1e-9, nominal_yield / freq)
        try:
            floor_value = floor_excess / (1.0 + nominal_per_period) ** terminal_exponent
        except (OverflowError, ValueError, ZeroDivisionError):
            raise ValueError('Treasury valuation overflowed while discounting the deflation floor.')
        clean_pv += floor_value

    accrued = 0.0
    if coupon > TGA_FLOOR_TOLERANCE and remaining:
        next_coupon = remaining[0]
        prior_coupon = next_coupon - relativedelta(months=12 // freq)
        span = (next_coupon - prior_coupon).days
        if span > 0:
            elapsed = max(0, (settlement_date - prior_coupon).days)
            accrued = coupon_payment * min(1.0, elapsed / span)

    clean = clean_pv - accrued
    return {
        'clean': clean,
        'accrued': accrued,
        'dirty': clean_pv,
        'deflation_floor_value': floor_value,
        'deflation_floor_basis': (
            'deterministic_scenario_payoff_no_option_time_value' if floor_excess > TGA_FLOOR_TOLERANCE else 'not_binding'
        ),
    }


def _remaining_coupon_dates(*, settlement_date, maturity_date, issue_date, first_interest_payment_date, frequency):
    """Coupon dates strictly after settlement, through maturity, newest schedule info first."""

    months = 12 // max(1, int(frequency))
    anchor = first_interest_payment_date if first_interest_payment_date is not None and not pd.isna(first_interest_payment_date) else None
    if anchor is None:
        # Treasury coupons fall on the maturity day-of-month, so walk back from maturity.
        anchor = maturity_date
    dates = []
    cursor = pd.Timestamp(anchor)
    while cursor > settlement_date + pd.Timedelta(days=1):
        cursor = cursor - relativedelta(months=months)
    cursor = cursor + relativedelta(months=months)
    while cursor <= maturity_date + pd.Timedelta(days=1):
        if cursor > settlement_date:
            dates.append(min(pd.Timestamp(cursor), pd.Timestamp(maturity_date)))
        cursor = cursor + relativedelta(months=months)
    if not dates or dates[-1] < maturity_date:
        dates.append(pd.Timestamp(maturity_date))
    return sorted(set(dates))

def calculate_face_from_proceeds_target(security_type, maturity_years, coupon_rate, yield_at_issuance, proceeds_target):
    """Convert a proceeds target into face issued, accounting for bill discounts."""
    if proceeds_target <= TGA_FLOOR_TOLERANCE:
        return (0.0, 0.0, 1.0)
    issue_price_ratio = calculate_issue_price_ratio(security_type, maturity_years, coupon_rate, yield_at_issuance)
    if issue_price_ratio <= TGA_FLOOR_TOLERANCE:
        return (0.0, 0.0, issue_price_ratio)
    face_value = proceeds_target / issue_price_ratio
    actual_proceeds = face_value * issue_price_ratio
    return (face_value, actual_proceeds, issue_price_ratio)

def quote_issuance_from_face_target(
    security_type,
    maturity_years,
    coupon_rate,
    yield_at_issuance,
    face_target,
    *,
    apply_rate_floor=False,
):
    """Convert a face issuance target into auction proceeds and issue price."""
    if face_target <= TGA_FLOOR_TOLERANCE:
        return (0.0, 0.0, 1.0)
    issue_price_ratio = calculate_issue_price_ratio(
        security_type,
        maturity_years,
        coupon_rate,
        yield_at_issuance,
        apply_rate_floor=apply_rate_floor,
    )
    if issue_price_ratio <= TGA_FLOOR_TOLERANCE:
        return (0.0, 0.0, issue_price_ratio)
    face_value = float(face_target)
    auction_proceeds = face_value * issue_price_ratio
    return (face_value, auction_proceeds, issue_price_ratio)

def infer_issue_data_for_loaded_bill(face_value, original_maturity_years, issue_yield_at_issue=None, yield_curve_years=None, yield_curve_rates=None):
    """Infer bill issue proceeds for legacy portfolios missing issue data."""
    yld = issue_yield_at_issue
    if pd.isna(yld):
        yld = get_yield_for_maturity(original_maturity_years, yield_curve_years or [], yield_curve_rates or [])
    yld = 0.0 if pd.isna(yld) else max(0.0, float(yld))
    issue_price_ratio = calculate_issue_price_ratio('Fixed', original_maturity_years, 0.0, yld)
    issue_proceeds = float(face_value) * issue_price_ratio
    return (issue_price_ratio, issue_proceeds, yld)

def get_maturity_category(maturity_years, issuance_profile):
    """
    Determines if a *fixed-rate marketable* maturity falls into bills, notes, or bonds.
    Uses cutoffs defined in the issuance profile.
    """
    if not isinstance(issuance_profile, dict):
        return 'unknown'
    bills_cutoff = issuance_profile.get('bills', {}).get('category_cutoff_years', 1.0)
    notes_cutoff = issuance_profile.get('notes', {}).get('category_cutoff_years', 10.0)
    if maturity_years <= bills_cutoff + TGA_FLOOR_TOLERANCE:
        return 'bills'
    elif maturity_years <= notes_cutoff + TGA_FLOOR_TOLERANCE:
        return 'notes'
    else:
        return 'bonds'

def get_security_category_for_prefs(security_type, maturity_years, issuance_profile):
    """
    Determines the category key ('bills', 'notes', 'bonds', 'tips', 'frn', 'nonmarketable')
    used for looking up sector preference percentages.
    """
    if security_type == 'TIPS':
        return 'tips'
    if security_type == 'FRN':
        return 'frn'
    if security_type == 'NonMarketable':
        return 'nonmarketable'
    if security_type == 'Fixed':
        if pd.isna(maturity_years):
            return None
        return get_maturity_category(maturity_years, issuance_profile)
    return None

def get_yield_for_maturity(
    target_maturity_years,
    yield_curve_years,
    yield_curve_rates,
    method='linear',
    floor_zero=True,
):
    """
    Interpolates yield for a given maturity from the provided yield curve data.
    Supports 'linear', 'pchip' (Piecewise Cubic Hermite Interpolating Polynomial), and 'cubic' (Cubic Spline).
    PCHIP is recommended for yield curves to preserve monotonicity and avoid oscillations.
    """
    if pd.isna(target_maturity_years) or target_maturity_years < 0:
        return np.nan
    if not isinstance(yield_curve_years, (list, np.ndarray)) or not isinstance(yield_curve_rates, (list, np.ndarray)):
        return np.nan
    if len(yield_curve_years) == 0 or len(yield_curve_years) != len(yield_curve_rates):
        return np.nan
    yield_curve_years_np = np.array(yield_curve_years)
    yield_curve_rates_np = np.array(yield_curve_rates)
    if not np.all(yield_curve_years_np[:-1] <= yield_curve_years_np[1:]):
        sort_idx = np.argsort(yield_curve_years_np)
        yield_curve_years_np = yield_curve_years_np[sort_idx]
        yield_curve_rates_np = yield_curve_rates_np[sort_idx]
    if floor_zero:
        yield_curve_rates_np = np.maximum(yield_curve_rates_np, 0.0)
    if target_maturity_years <= yield_curve_years_np[0]:
        return yield_curve_rates_np[0]
    if target_maturity_years >= yield_curve_years_np[-1]:
        return yield_curve_rates_np[-1]
    try:
        if method == 'pchip':
            interpolator = PchipInterpolator(yield_curve_years_np, yield_curve_rates_np)
            interp_rate = float(interpolator(target_maturity_years))
        elif method == 'cubic':
            interpolator = CubicSpline(yield_curve_years_np, yield_curve_rates_np)
            interp_rate = float(interpolator(target_maturity_years))
        else:
            interp_rate = np.interp(target_maturity_years, yield_curve_years_np, yield_curve_rates_np)
        return max(0.0, interp_rate) if floor_zero else interp_rate
    except Exception as e:
        return np.nan


def evaluate_nominal_yield(
    maturity_years,
    curve_years,
    curve_rates,
    *,
    method,
    floor_zero,
    shock: EvaluatedNominalShock | None,
):
    """Evaluate the unchanged baseline nominal curve, then apply an optional shock."""

    base = get_yield_for_maturity(
        maturity_years,
        curve_years,
        curve_rates,
        method=method,
        floor_zero=floor_zero,
    )
    if shock is None:
        return base
    if not isinstance(shock, EvaluatedNominalShock):
        raise CurveContractError("nominal curve shock must be an EvaluatedNominalShock")
    if method != "pchip" or floor_zero is not False:
        raise CurveContractError(
            "evaluated additive shock requires baseline pchip with floor_zero=false"
        )

    return shock.apply_to_baseline(maturity_years, base)


def calculate_coupon_rate(security_type, maturity_years, yield_at_issuance, tips_real_coupon):
    """
    Determines the nominal coupon rate at issuance based on security type and market yield.
    """
    if pd.isna(yield_at_issuance) and security_type == 'Fixed':
        return 0.0
    if pd.isna(maturity_years):
        maturity_years = 0.0
    if security_type == 'Fixed':
        if maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE:
            return 0.0
        # Notes and bonds carry a coupon set on Treasury's 1/8-point grid, not the stop-out
        # yield itself. Setting coupon == yield would make every security price at exactly par
        # by construction and understate both issue discount and the coupon/principal split.
        return calculate_auction_coupon_rate(yield_at_issuance)
    elif security_type == 'TIPS':
        if not pd.isna(yield_at_issuance):
            return calculate_auction_coupon_rate(yield_at_issuance)
        return max(0.0, tips_real_coupon)
    elif security_type == 'FRN':
        return 0.0
    elif security_type == 'NonMarketable':
        return 0.0
    else:
        return 0.0

def get_payment_date(year, month, day):
    """
    Attempts to create a valid pd.Timestamp for a given year, month, day.
    Falls back to the last valid day of that month if day is invalid.
    """
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        try:
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        except ValueError:
            raise ValueError(f'Invalid year/month combination for payment date: {year}-{month}')

def get_coupon_dates_in_period(
    issue_date,
    maturity_date,
    prev_date,
    current_date,
    frequency=2,
    first_interest_payment_date=None,
    interest_payment_frequency=None,
):
    """
    Returns a list of coupon payment dates that fall in the half-open interval
    (prev_date, current_date] for a bond with the given issue/maturity dates.

    Correctly handles multi-year bonds by computing recurring coupon months from
    explicit Treasury-style interest terms when available. Legacy rows without
    those fields fall back to the maturity-date month/day schedule.

    Parameters:
        issue_date:    pd.Timestamp — bond issue date
        maturity_date: pd.Timestamp — bond maturity date
        prev_date:     pd.Timestamp — start of period (exclusive)
        current_date:  pd.Timestamp — end of period (inclusive)
        frequency:     int — fallback payments per year (2=semi-annual, 4=quarterly)
        first_interest_payment_date: optional first contractual coupon date
        interest_payment_frequency: optional contractual payments per year

    Returns:
        List[pd.Timestamp] — coupon dates in the period, sorted ascending.

    Notes:
        - Coupon dates ON the maturity date are excluded (the maturity handler
          pays the final coupon + principal separately).
        - Coupon dates ON or before the issue date are excluded.
        - The day-of-month follows FirstInterestPaymentDate when supplied.
          Otherwise it follows the maturity date's month/day, with fallback to
          month-end for invalid days (e.g., Feb 29 in non-leap years).
    """
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return []
    issue_date = pd.Timestamp(issue_date).normalize()
    maturity_date = pd.Timestamp(maturity_date).normalize()
    prev_date = pd.Timestamp(prev_date).normalize()
    current_date = pd.Timestamp(current_date).normalize()
    frequency = _normalize_payment_frequency(interest_payment_frequency, frequency)
    months_between = 12 // frequency
    first_interest = (
        pd.Timestamp(first_interest_payment_date).normalize()
        if first_interest_payment_date is not None and not pd.isna(first_interest_payment_date)
        else pd.NaT
    )
    anchor = first_interest if not pd.isna(first_interest) else maturity_date
    anchor_month = anchor.month
    anchor_day = anchor.day
    coupon_months = set()
    for i in range(frequency):
        m = (anchor_month + months_between * i - 1) % 12 + 1
        coupon_months.add(m)
    dates_in_period = []
    for year in range(prev_date.year, current_date.year + 1):
        for month in coupon_months:
            try:
                pmt_date = get_payment_date(year, month, anchor_day)
            except ValueError:
                continue
            if (
                pmt_date > issue_date
                and pmt_date < maturity_date
                and (pd.isna(first_interest) or pmt_date >= first_interest)
                and (prev_date < pmt_date <= current_date)
            ):
                dates_in_period.append(pmt_date)
    dates_in_period.sort()
    return dates_in_period


def _normalize_payment_frequency(value, fallback):
    if value is None or pd.isna(value):
        value = fallback
    try:
        frequency = int(round(float(value)))
    except (TypeError, ValueError):
        frequency = int(round(float(fallback)))
    if frequency <= 0 or 12 % frequency != 0:
        return int(round(float(fallback)))
    return frequency

def find_last_coupon_date(settlement_date, issue_date, frequency=2):
    """
    Finds the most recent coupon payment date that occurred on or before the settlement_date.
    """
    if pd.isna(issue_date) or pd.isna(settlement_date) or settlement_date < issue_date:
        return pd.NaT
    if frequency not in [2, 4]:
        frequency = 2
    months_between_payments = 12 // frequency
    issue_month = issue_date.month
    issue_day = issue_date.day
    potential_payment_months = sorted([(issue_month + i * months_between_payments - 1) % 12 + 1 for i in range(frequency)])
    last_found_coupon_date = pd.NaT
    for year_to_check in [settlement_date.year, settlement_date.year - 1]:
        for month in reversed(potential_payment_months):
            try:
                potential_date = get_payment_date(year_to_check, month, issue_day)
                if potential_date <= settlement_date and potential_date >= issue_date:
                    if pd.isna(last_found_coupon_date) or potential_date > last_found_coupon_date:
                        last_found_coupon_date = potential_date
            except ValueError:
                continue
        if not pd.isna(last_found_coupon_date):
            break
    return issue_date if pd.isna(last_found_coupon_date) else last_found_coupon_date

def calculate_accrued_interest(face_value, coupon_rate, settlement_date, issue_date, security_type='Fixed', adjusted_principal=None, accrued_frn=None, frequency=2):
    """
    Calculates accrued interest for a bond up to the settlement date.
    """
    if pd.isna(settlement_date) or pd.isna(issue_date) or settlement_date <= issue_date:
        return 0.0
    if pd.isna(face_value) or face_value < TGA_FLOOR_TOLERANCE:
        return 0.0
    if security_type == 'FRN':
        return accrued_frn if accrued_frn is not None and accrued_frn > 0 else 0.0
    if security_type == 'NonMarketable':
        return 0.0
    if security_type not in ['Fixed', 'TIPS']:
        return 0.0
    if coupon_rate <= TGA_FLOOR_TOLERANCE:
        return 0.0
    principal_base = face_value
    if security_type == 'TIPS':
        if adjusted_principal is not None and adjusted_principal > TGA_FLOOR_TOLERANCE:
            principal_base = adjusted_principal
        elif face_value > TGA_FLOOR_TOLERANCE:
            principal_base = face_value
        else:
            return 0.0
    if principal_base < TGA_FLOOR_TOLERANCE:
        return 0.0
    last_coupon_dt = find_last_coupon_date(settlement_date, issue_date, frequency)
    if pd.isna(last_coupon_dt):
        last_coupon_dt = issue_date
    months_between = 12 // frequency
    issue_month = issue_date.month
    issue_day = issue_date.day
    potential_payment_months = [(issue_month + i * months_between - 1) % 12 + 1 for i in range(frequency)]
    next_coupon_dt = pd.NaT
    year_cursor = last_coupon_dt.year
    search_limit = settlement_date.year + 2
    while pd.isna(next_coupon_dt) and year_cursor < search_limit:
        for month in potential_payment_months:
            try:
                potential_next_dt = get_payment_date(year_cursor, month, issue_day)
                if potential_next_dt > last_coupon_dt:
                    if pd.isna(next_coupon_dt) or potential_next_dt < next_coupon_dt:
                        next_coupon_dt = potential_next_dt
            except ValueError:
                continue
        if not pd.isna(next_coupon_dt):
            break
        year_cursor += 1
    if pd.isna(next_coupon_dt):
        return 0.0
    days_accrued = (settlement_date - last_coupon_dt).days
    days_in_period = (next_coupon_dt - last_coupon_dt).days
    if days_accrued < 0 or days_in_period <= 0:
        return 0.0
    periodic_coupon_payment = principal_base * coupon_rate / frequency
    accrued = periodic_coupon_payment * (days_accrued / days_in_period)
    return max(0.0, accrued)

def calculate_bond_market_price(face_value, coupon_rate, maturity_date, current_date, discount_yield, security_type='Fixed', adjusted_principal=None, original_principal=None, accrued_frn=None, frequency=2):
    """
    Calculates the 'clean' market price (present value of future cash flows).
    """
    if pd.isna(maturity_date) or pd.isna(current_date) or current_date >= maturity_date:
        return 0.0
    if pd.isna(face_value) or face_value < TGA_FLOOR_TOLERANCE:
        return 0.0
    time_to_maturity_years = (maturity_date - current_date).total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
    if time_to_maturity_years <= TGA_FLOOR_TOLERANCE:
        return 0.0
    if security_type == 'NonMarketable':
        return face_value
    if security_type == 'FRN':
        return face_value
    if pd.isna(discount_yield):
        return face_value
    eff_discount_yield = max(discount_yield, 1e-09)
    principal_at_maturity = face_value
    principal_for_coupons = face_value
    if security_type == 'TIPS':
        adj_p = adjusted_principal if adjusted_principal is not None and adjusted_principal > 0 else face_value
        orig_p = original_principal if original_principal is not None and original_principal > 0 else face_value
        principal_at_maturity = max(orig_p, adj_p)
        principal_for_coupons = adj_p
    market_price = 0.0
    try:
        if coupon_rate <= TGA_FLOOR_TOLERANCE:
            market_price = principal_at_maturity / (1 + eff_discount_yield) ** time_to_maturity_years
        else:
            periods_per_year = float(frequency)
            periodic_coupon_payment = principal_for_coupons * coupon_rate / periods_per_year
            payment_dates = []
            try:
                est_issue_dt_for_pattern = maturity_date - relativedelta(years=int(round(time_to_maturity_years)))
            except ValueError:
                est_issue_dt_for_pattern = current_date
            issue_month_pattern = est_issue_dt_for_pattern.month
            issue_day_pattern = est_issue_dt_for_pattern.day
            months_between = 12 // frequency
            potential_payment_months = [(issue_month_pattern + i * months_between - 1) % 12 + 1 for i in range(frequency)]
            search_year = current_date.year
            max_search_year = maturity_date.year + 1
            while search_year <= max_search_year:
                found_in_year = False
                for month in potential_payment_months:
                    try:
                        pmt_date = get_payment_date(search_year, month, issue_day_pattern)
                        if pmt_date > current_date and pmt_date <= maturity_date:
                            payment_dates.append(pmt_date)
                            found_in_year = True
                    except ValueError:
                        continue
                if payment_dates and payment_dates[-1] >= maturity_date:
                    break
                search_year += 1
            payment_dates = sorted(list(set(payment_dates)))
            pv_coupons = 0.0
            for pmt_date in payment_dates:
                time_to_pmt = (pmt_date - current_date).total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
                if time_to_pmt > TGA_FLOOR_TOLERANCE:
                    try:
                        pv_coupons += periodic_coupon_payment / (1 + eff_discount_yield) ** time_to_pmt
                    except (OverflowError, ValueError):
                        pass
            pv_face_value = 0.0
            try:
                pv_face_value = principal_at_maturity / (1 + eff_discount_yield) ** time_to_maturity_years
            except (OverflowError, ValueError):
                pv_face_value = 0.0
            market_price = pv_coupons + pv_face_value
    except (OverflowError, ValueError, ZeroDivisionError) as e:
        market_price = 0.0
    return max(0.0, market_price)


__all__ = [
    '_is_bill_like_fixed',
    'calculate_issue_price_ratio',
    'calculate_face_from_proceeds_target',
    'quote_issuance_from_face_target',
    'calculate_auction_coupon_rate',
    'value_treasury_security',
    'calculate_coupon_security_issue_price_ratio',
    'infer_issue_data_for_loaded_bill',
    'get_maturity_category',
    'get_security_category_for_prefs',
    'get_yield_for_maturity',
    'evaluate_nominal_yield',
    'calculate_coupon_rate',
    'get_payment_date',
    'get_coupon_dates_in_period',
    'find_last_coupon_date',
    'calculate_accrued_interest',
    'calculate_bond_market_price',
]
