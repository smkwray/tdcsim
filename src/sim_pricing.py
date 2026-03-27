
"""Pricing and coupon-schedule utilities for Treasury securities."""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.interpolate import CubicSpline, PchipInterpolator

from tdc_shared import DAYS_PER_YEAR_ACTUAL, TGA_FLOOR_TOLERANCE

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

def calculate_issue_price_ratio(security_type, maturity_years, coupon_rate, yield_at_issuance):
    """
    Returns the cash-proceeds-to-face ratio at auction.

    The simulator keeps coupon-bearing nominal/TIPS/FRN/nonmarketable issuance at par for now.
    Bills are issued at a discount using the same simple zero-coupon discount convention
    used elsewhere in the model's pricing logic.
    """
    if _is_bill_like_fixed(security_type, maturity_years, coupon_rate):
        eff_yield = 0.0 if pd.isna(yield_at_issuance) else max(0.0, float(yield_at_issuance))
        maturity_val = 0.0 if pd.isna(maturity_years) else max(0.0, float(maturity_years))
        try:
            return max(TGA_FLOOR_TOLERANCE, 1.0 / (1.0 + eff_yield) ** maturity_val)
        except Exception:
            return 1.0
    return 1.0

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

def get_yield_for_maturity(target_maturity_years, yield_curve_years, yield_curve_rates, method='linear'):
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
        return max(0.0, interp_rate)
    except Exception as e:
        return np.nan

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
        else:
            return max(0.0, yield_at_issuance)
    elif security_type == 'TIPS':
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

def get_coupon_dates_in_period(issue_date, maturity_date, prev_date, current_date, frequency=2):
    """
    Returns a list of coupon payment dates that fall in the half-open interval
    (prev_date, current_date] for a bond with the given issue/maturity dates.

    Correctly handles multi-year bonds by computing recurring coupon months from
    the issue date and checking all relevant years.

    Parameters:
        issue_date:    pd.Timestamp — bond issue date
        maturity_date: pd.Timestamp — bond maturity date
        prev_date:     pd.Timestamp — start of period (exclusive)
        current_date:  pd.Timestamp — end of period (inclusive)
        frequency:     int — payments per year (2=semi-annual, 4=quarterly)

    Returns:
        List[pd.Timestamp] — coupon dates in the period, sorted ascending.

    Notes:
        - Coupon dates ON the maturity date are excluded (the maturity handler
          pays the final coupon + principal separately).
        - Coupon dates ON or before the issue date are excluded.
        - The day-of-month follows the issue date's day, with fallback to
          month-end for invalid days (e.g., Feb 29 in non-leap years).
    """
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return []
    months_between = 12 // frequency
    issue_month = issue_date.month
    issue_day = issue_date.day
    coupon_months = set()
    for i in range(1, frequency + 1):
        m = (issue_month + months_between * i - 1) % 12 + 1
        coupon_months.add(m)
    dates_in_period = []
    for year in range(prev_date.year, current_date.year + 1):
        for month in coupon_months:
            try:
                pmt_date = get_payment_date(year, month, issue_day)
            except ValueError:
                continue
            if pmt_date > issue_date and pmt_date < maturity_date and (prev_date < pmt_date <= current_date):
                dates_in_period.append(pmt_date)
    dates_in_period.sort()
    return dates_in_period

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
    'infer_issue_data_for_loaded_bill',
    'get_maturity_category',
    'get_security_category_for_prefs',
    'get_yield_for_maturity',
    'calculate_coupon_rate',
    'get_payment_date',
    'get_coupon_dates_in_period',
    'find_last_coupon_date',
    'calculate_accrued_interest',
    'calculate_bond_market_price',
]
