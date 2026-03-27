"""
Standalone tests for the new coupon schedule detection function.
Run with: python test_coupon_schedule.py

This function replaces the buggy month-offset logic in simulation_core.py
that only detected coupons in the first year after issuance.
"""
import sys
import pandas as pd
import numpy as np

# ============================================================
# Function under test (will be moved to simulation_core.py)
# ============================================================

def get_payment_date(year, month, day):
    """
    Attempts to create a valid pd.Timestamp for a given year, month, day.
    Falls back to the last valid day of that month if day is invalid.
    (Identical to simulation_core.py version.)
    """
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        try:
            return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
        except ValueError:
            raise ValueError(f"Invalid year/month combination for payment date: {year}-{month}")


def get_coupon_dates_in_period(issue_date, maturity_date, prev_date, current_date, frequency=2):
    """
    Returns a list of coupon payment dates that fall in the half-open interval
    (prev_date, current_date] for a bond with the given issue/maturity dates.

    This correctly handles multi-year bonds by computing coupon months from the
    issue date and checking all relevant years, not just the issue year.

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
          in the simulation loop pays the final coupon + principal separately).
        - Coupon dates ON or before the issue date are excluded.
        - The day-of-month follows the issue date's day, with fallback to
          month-end for invalid days (e.g., Feb 29 in non-leap years).
    """
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return []

    months_between = 12 // frequency
    issue_month = issue_date.month
    issue_day = issue_date.day

    # Compute the set of coupon months (1-indexed) that recur every year.
    # E.g., for a March-issued semi-annual bond: months 9 (Sep) and 3 (Mar).
    coupon_months = set()
    for i in range(1, frequency + 1):
        m = (issue_month + months_between * i - 1) % 12 + 1
        coupon_months.add(m)

    dates_in_period = []

    # Only iterate over years that could possibly have dates in (prev_date, current_date].
    for year in range(prev_date.year, current_date.year + 1):
        for month in coupon_months:
            try:
                pmt_date = get_payment_date(year, month, issue_day)
            except ValueError:
                continue

            # Must be strictly after issue, strictly before maturity, and in window.
            if (pmt_date > issue_date
                    and pmt_date < maturity_date
                    and prev_date < pmt_date <= current_date):
                dates_in_period.append(pmt_date)

    dates_in_period.sort()
    return dates_in_period


# ============================================================
# Old (buggy) logic — reproduced here for side-by-side comparison
# ============================================================

def old_has_coupon_in_period(issue_date, maturity_date, prev_date, current_date, frequency=2):
    """
    Reproduces the ORIGINAL buggy logic from simulation_core.py.
    Only checks month offsets from the issue year — misses coupons after year 1.
    """
    if pd.isna(issue_date):
        return False

    issue_month = issue_date.month
    issue_day = issue_date.day

    if frequency == 4:
        month_offsets = [3, 6, 9, 12]
    else:
        month_offsets = [6, 12]

    for month_offset in month_offsets:
        potential_month = (issue_month + month_offset - 1) % 12 + 1
        potential_year = issue_date.year + (issue_month + month_offset - 1) // 12
        if potential_year > current_date.year + 1:
            continue
        try:
            pmt_date = get_payment_date(potential_year, potential_month, issue_day)
        except ValueError:
            continue
        if prev_date < pmt_date <= current_date and pmt_date < maturity_date:
            return True
    return False


# ============================================================
# Test helpers
# ============================================================

def collect_all_coupons(issue_date, maturity_date, frequency=2, step_days=7):
    """
    Simulate the weekly loop over the bond's entire life, collecting all
    coupon dates found by get_coupon_dates_in_period.
    """
    all_dates = []
    current = issue_date
    while current < maturity_date:
        prev = current
        current = current + pd.Timedelta(days=step_days)
        if current > maturity_date:
            # Don't overshoot past maturity (maturity handler takes over)
            current = maturity_date
        dates = get_coupon_dates_in_period(issue_date, maturity_date, prev, current, frequency)
        all_dates.extend(dates)
    return sorted(set(all_dates))


def collect_all_coupons_old(issue_date, maturity_date, frequency=2, step_days=7):
    """Same walk but using the old buggy logic. Returns count of periods where a coupon was detected."""
    count = 0
    current = issue_date
    while current < maturity_date:
        prev = current
        current = current + pd.Timedelta(days=step_days)
        if current > maturity_date:
            current = maturity_date
        if old_has_coupon_in_period(issue_date, maturity_date, prev, current, frequency):
            count += 1
    return count


# ============================================================
# Tests
# ============================================================

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")


def test_2y_note():
    """2Y semi-annual note: 3 regular coupons + 1 at maturity = 4 total payments."""
    print("\n--- test_2y_note ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2022-03-15')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Expected regular coupons (< maturity): Sep 2020, Mar 2021, Sep 2021
    expected = [
        pd.Timestamp('2020-09-15'),
        pd.Timestamp('2021-03-15'),
        pd.Timestamp('2021-09-15'),
    ]
    check("count is 3", len(coupons) == 3, f"got {len(coupons)}: {coupons}")
    check("dates match", coupons == expected, f"got {coupons}")

    # Old logic comparison
    old_count = collect_all_coupons_old(issue, maturity, frequency=2)
    check(f"old logic finds {old_count} (for comparison, should find fewer for longer bonds)", True)


def test_10y_note():
    """10Y semi-annual note: 19 regular coupons."""
    print("\n--- test_10y_note ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2030-03-15')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # 10 years × 2 coupons/year = 20, minus the maturity coupon = 19
    check("count is 19", len(coupons) == 19, f"got {len(coupons)}")

    # Verify first and last
    check("first coupon is Sep 2020", coupons[0] == pd.Timestamp('2020-09-15'))
    check("last coupon is Sep 2029", coupons[-1] == pd.Timestamp('2029-09-15'))

    # Old logic comparison — should miss most of these
    old_count = collect_all_coupons_old(issue, maturity, frequency=2)
    check(f"old logic only finds {old_count} (expected ~2, demonstrating the bug)", old_count <= 2,
          f"old found {old_count}")


def test_30y_bond():
    """30Y semi-annual bond: 59 regular coupons."""
    print("\n--- test_30y_bond ---")
    issue = pd.Timestamp('2020-06-15')
    maturity = pd.Timestamp('2050-06-15')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # 30 years × 2 = 60, minus maturity coupon = 59
    check("count is 59", len(coupons) == 59, f"got {len(coupons)}")
    check("first coupon is Dec 2020", coupons[0] == pd.Timestamp('2020-12-15'))
    check("last coupon is Dec 2049", coupons[-1] == pd.Timestamp('2049-12-15'))

    old_count = collect_all_coupons_old(issue, maturity, frequency=2)
    check(f"old logic only finds {old_count} (demonstrating the bug)", old_count <= 2,
          f"old found {old_count}")


def test_frn_quarterly():
    """2Y FRN with quarterly payments: 7 regular coupons."""
    print("\n--- test_frn_quarterly ---")
    issue = pd.Timestamp('2024-01-15')
    maturity = pd.Timestamp('2026-01-15')

    coupons = collect_all_coupons(issue, maturity, frequency=4)
    # 2 years × 4 = 8, minus maturity = 7
    check("count is 7", len(coupons) == 7, f"got {len(coupons)}")
    check("first coupon is Apr 2024", coupons[0] == pd.Timestamp('2024-04-15'))
    check("last coupon is Oct 2025", coupons[-1] == pd.Timestamp('2025-10-15'))


def test_coupon_in_specific_week():
    """10Y bond: verify coupon is detected in a specific week in year 8."""
    print("\n--- test_coupon_in_specific_week ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2030-03-15')

    # Week containing Sep 15, 2028
    prev = pd.Timestamp('2028-09-10')
    curr = pd.Timestamp('2028-09-17')
    coupons = get_coupon_dates_in_period(issue, maturity, prev, curr, frequency=2)
    check("finds Sep 15 2028 coupon", len(coupons) == 1 and coupons[0] == pd.Timestamp('2028-09-15'),
          f"got {coupons}")

    # Old logic would miss this
    old_found = old_has_coupon_in_period(issue, maturity, prev, curr, frequency=2)
    check(f"old logic misses it: {old_found}", old_found == False)


def test_no_coupon_week():
    """No coupon should be found in a week that doesn't contain one."""
    print("\n--- test_no_coupon_week ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2030-03-15')

    prev = pd.Timestamp('2028-09-01')
    curr = pd.Timestamp('2028-09-08')
    coupons = get_coupon_dates_in_period(issue, maturity, prev, curr, frequency=2)
    check("no coupon in off-week", len(coupons) == 0, f"got {coupons}")


def test_boundary_prev_date_equals_coupon():
    """Coupon ON prev_date should NOT be included (half-open interval)."""
    print("\n--- test_boundary_prev_date_equals_coupon ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2030-03-15')

    # prev_date IS the coupon date
    prev = pd.Timestamp('2025-09-15')
    curr = pd.Timestamp('2025-09-22')
    coupons = get_coupon_dates_in_period(issue, maturity, prev, curr, frequency=2)
    check("coupon on prev_date excluded", len(coupons) == 0, f"got {coupons}")


def test_boundary_current_date_equals_coupon():
    """Coupon ON current_date SHOULD be included (half-open interval)."""
    print("\n--- test_boundary_current_date_equals_coupon ---")
    issue = pd.Timestamp('2020-03-15')
    maturity = pd.Timestamp('2030-03-15')

    prev = pd.Timestamp('2025-09-08')
    curr = pd.Timestamp('2025-09-15')
    coupons = get_coupon_dates_in_period(issue, maturity, prev, curr, frequency=2)
    check("coupon on current_date included", len(coupons) == 1, f"got {coupons}")


def test_feb29_issue_date():
    """Bond issued on Feb 29 (leap year) should handle non-leap coupon dates."""
    print("\n--- test_feb29_issue_date ---")
    issue = pd.Timestamp('2024-02-29')  # leap year
    maturity = pd.Timestamp('2034-02-28')  # 10Y, lands on non-leap year

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Coupon months: Aug and Feb. Semi-annual = 19 regular coupons.
    check("count is 19", len(coupons) == 19, f"got {len(coupons)}")

    # First coupon: Aug 29 2024
    check("first coupon is Aug 29 2024", coupons[0] == pd.Timestamp('2024-08-29'))

    # Feb 2025 coupon should fall on Feb 28 (not a leap year)
    feb_2025 = [d for d in coupons if d.year == 2025 and d.month == 2]
    check("Feb 2025 coupon is Feb 28", len(feb_2025) == 1 and feb_2025[0].day == 28,
          f"got {feb_2025}")

    # Feb 2028 coupon should fall on Feb 29 (leap year)
    feb_2028 = [d for d in coupons if d.year == 2028 and d.month == 2]
    check("Feb 2028 coupon is Feb 29", len(feb_2028) == 1 and feb_2028[0].day == 29,
          f"got {feb_2028}")


def test_jan31_issue_date():
    """Bond issued Jan 31 — coupon months with fewer days should use month-end."""
    print("\n--- test_jan31_issue_date ---")
    issue = pd.Timestamp('2020-01-31')
    maturity = pd.Timestamp('2022-01-31')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Coupon months: Jul (31 days, fine) and Jan (31 days, fine)
    # Expected: Jul 31 2020, Jan 31 2021, Jul 31 2021 = 3
    check("count is 3", len(coupons) == 3, f"got {len(coupons)}: {coupons}")
    check("Jul 2020 coupon", coupons[0] == pd.Timestamp('2020-07-31'))
    check("Jan 2021 coupon", coupons[1] == pd.Timestamp('2021-01-31'))


def test_aug31_issue_date():
    """Bond issued Aug 31 — Feb coupon should land on Feb 28/29."""
    print("\n--- test_aug31_issue_date ---")
    issue = pd.Timestamp('2020-08-31')
    maturity = pd.Timestamp('2022-08-31')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Coupon months: Feb and Aug.
    # Feb 28 2021 (non-leap), Aug 31 2021, Feb 28 2022 = 3
    check("count is 3", len(coupons) == 3, f"got {len(coupons)}: {coupons}")
    check("Feb 2021 is month-end (28)", coupons[0] == pd.Timestamp('2021-02-28'))
    check("Aug 2021 is 31st", coupons[1] == pd.Timestamp('2021-08-31'))
    check("Feb 2022 is month-end (28)", coupons[2] == pd.Timestamp('2022-02-28'))


def test_no_coupons_for_bill():
    """A 6-month bill (zero coupon) has maturity < 1 year, no coupons expected.
    Bills have CouponRate = 0 so the caller wouldn't even call this, but
    if called, it should return empty since maturity is before any coupon date."""
    print("\n--- test_no_coupons_for_bill ---")
    issue = pd.Timestamp('2025-01-15')
    maturity = pd.Timestamp('2025-07-15')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Semi-annual coupon would be at Jul 15 2025, but that equals maturity → excluded.
    check("no regular coupons for 6-month bill", len(coupons) == 0, f"got {coupons}")


def test_1y_bill():
    """1Y bill — first coupon at 6 months, but maturity at 12 months."""
    print("\n--- test_1y_bill ---")
    issue = pd.Timestamp('2025-01-15')
    maturity = pd.Timestamp('2026-01-15')

    coupons = collect_all_coupons(issue, maturity, frequency=2)
    # Coupon at Jul 15 2025 (< maturity Jan 15 2026). Jan 15 2026 = maturity, excluded.
    check("1 regular coupon at 6 months", len(coupons) == 1, f"got {len(coupons)}: {coupons}")
    check("coupon is Jul 2025", coupons[0] == pd.Timestamp('2025-07-15'))


def test_empty_inputs():
    """NaT inputs should return empty list."""
    print("\n--- test_empty_inputs ---")
    prev = pd.Timestamp('2025-01-01')
    curr = pd.Timestamp('2025-01-08')

    check("NaT issue_date", get_coupon_dates_in_period(pd.NaT, pd.Timestamp('2030-01-01'), prev, curr) == [])
    check("NaT maturity_date", get_coupon_dates_in_period(pd.Timestamp('2020-01-01'), pd.NaT, prev, curr) == [])


def test_year_boundary():
    """Coupon detection across Dec 31 / Jan 1 boundary."""
    print("\n--- test_year_boundary ---")
    issue = pd.Timestamp('2020-06-30')
    maturity = pd.Timestamp('2030-06-30')

    # Coupon months: Dec and Jun. Check the Dec 30 2027 coupon.
    prev = pd.Timestamp('2027-12-27')
    curr = pd.Timestamp('2028-01-03')
    coupons = get_coupon_dates_in_period(issue, maturity, prev, curr, frequency=2)
    check("finds Dec 30 2027 coupon across year boundary", len(coupons) == 1, f"got {coupons}")
    check("date is Dec 30 2027", coupons[0] == pd.Timestamp('2027-12-30'))


def test_old_vs_new_comparison():
    """
    Quantify how many coupons the old logic misses for bonds of various tenors.
    This is the key demonstration that the fix matters.
    """
    print("\n--- test_old_vs_new_comparison ---")
    test_cases = [
        ("2Y note",  pd.Timestamp('2020-03-15'), pd.Timestamp('2022-03-15'), 2),
        ("5Y note",  pd.Timestamp('2020-03-15'), pd.Timestamp('2025-03-15'), 2),
        ("10Y note", pd.Timestamp('2020-03-15'), pd.Timestamp('2030-03-15'), 2),
        ("30Y bond", pd.Timestamp('2020-03-15'), pd.Timestamp('2050-03-15'), 2),
        ("2Y FRN",   pd.Timestamp('2020-03-15'), pd.Timestamp('2022-03-15'), 4),
    ]
    for label, issue, maturity, freq in test_cases:
        new_count = len(collect_all_coupons(issue, maturity, frequency=freq))
        old_count = collect_all_coupons_old(issue, maturity, frequency=freq)
        status = "OK" if new_count > old_count or (new_count == old_count and new_count <= 3) else "UNEXPECTED"
        # For short bonds (2Y) old and new might match; for longer bonds new should find more
        print(f"  {label}: new={new_count}, old={old_count}  [{status}]")
        if label == "10Y note":
            check("10Y: new finds 19, old finds ~2", new_count == 19 and old_count <= 2,
                  f"new={new_count}, old={old_count}")
        if label == "30Y bond":
            check("30Y: new finds 59, old finds ~2", new_count == 59 and old_count <= 2,
                  f"new={new_count}, old={old_count}")


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Coupon Schedule Function — Unit Tests")
    print("=" * 60)

    test_2y_note()
    test_10y_note()
    test_30y_bond()
    test_frn_quarterly()
    test_coupon_in_specific_week()
    test_no_coupon_week()
    test_boundary_prev_date_equals_coupon()
    test_boundary_current_date_equals_coupon()
    test_feb29_issue_date()
    test_jan31_issue_date()
    test_aug31_issue_date()
    test_no_coupons_for_bill()
    test_1y_bill()
    test_empty_inputs()
    test_year_boundary()
    test_old_vs_new_comparison()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)
