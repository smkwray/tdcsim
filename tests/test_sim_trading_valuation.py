"""Regression tests for secondary-trading clean and dirty valuation semantics."""

import pandas as pd
import pytest

from fed_income import accrued_interest
from sim_pricing import calculate_bond_market_price
from sim_trading import calculate_portfolio_value_and_composition
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES


CURRENT_DATE = pd.Timestamp('2025-02-15')
CURVE_YEARS = [0.25, 1.0, 5.0, 10.0, 30.0]
CURVE_RATES = [0.045] * len(CURVE_YEARS)


def _bond_row(**overrides):
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update({
        'BondID': 1,
        'SecurityType': 'Fixed',
        'IssueDate': pd.Timestamp('2024-11-15'),
        'MaturityDate': pd.Timestamp('2031-11-15'),
        'OriginalMaturityYears': 7.0,
        'FaceValue': 100.0,
        'CouponRate': 0.0425,
        'HolderType': 'Banks',
        'Status': 'Active',
        'MaturityCategory': 'notes',
        'OriginalPrincipal': 100.0,
        'AdjustedPrincipal': 100.0,
        'AccruedInterest_FRN': 0.0,
    })
    row.update(overrides)
    return row


def _price_row(**overrides):
    portfolio = pd.DataFrame(
        [_bond_row(**overrides)],
        columns=BOND_PORTFOLIO_COLS,
    ).astype(PORTFOLIO_DTYPES, errors='ignore')
    calculate_portfolio_value_and_composition(
        portfolio,
        CURRENT_DATE,
        CURVE_YEARS,
        CURVE_RATES,
    )
    return portfolio.iloc[0]


def _legacy_discounted_cash_flow(row, *, frequency=2):
    return calculate_bond_market_price(
        row['FaceValue'],
        row['CouponRate'],
        row['MaturityDate'],
        CURRENT_DATE,
        0.045,
        row['SecurityType'],
        row.get('AdjustedPrincipal'),
        row.get('OriginalPrincipal'),
        row.get('AccruedInterest_FRN'),
        frequency,
    )


def _expected_accrued(row):
    return float(accrued_interest(pd.DataFrame([row]), CURRENT_DATE).to_numpy().sum())


def test_preference_trades_accrue_from_treasurys_dated_date():
    """912810UT3 was issued on 2026-03-02 but dated 2026-02-15: on 2026-06-22 a buyer owes
    127 of the half-year's 181 days of its 4.625 percent coupon, 16.2258 per 1,000, not the
    14.0761 that accruing from the issue date gives."""

    portfolio = pd.DataFrame([_bond_row(
        IssueDate=pd.Timestamp('2026-03-02'), DatedDate=pd.Timestamp('2026-02-15'),
        MaturityDate=pd.Timestamp('2056-02-15'), OriginalMaturityYears=30.0, CouponRate=0.04625,
        FaceValue=1000.0, OriginalPrincipal=1000.0, AdjustedPrincipal=1000.0, MaturityCategory='bonds',
    )], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors='ignore')
    calculate_portfolio_value_and_composition(portfolio, pd.Timestamp('2026-06-22'), CURVE_YEARS, CURVE_RATES)
    assert portfolio.iloc[0]['AccruedInterest'] == pytest.approx(16.2258, abs=1e-4)


def test_mid_coupon_fixed_uses_discounted_cash_flows_as_dirty_value():
    row = _price_row()
    discounted_cash_flows = _legacy_discounted_cash_flow(row)
    accrued = _expected_accrued(row)

    assert accrued > 0.0
    assert row['AccruedInterest'] == pytest.approx(accrued)
    assert row['DirtyValue'] == pytest.approx(discounted_cash_flows)
    assert row['CleanPrice'] == pytest.approx(discounted_cash_flows - accrued)
    assert row['DirtyPriceRatio'] == pytest.approx(discounted_cash_flows / row['FaceValue'])


def test_mid_coupon_tips_uses_discounted_cash_flows_as_dirty_value():
    row = _price_row(
        SecurityType='TIPS',
        CouponRate=0.0125,
        OriginalPrincipal=100.0,
        AdjustedPrincipal=105.0,
        MaturityCategory='tips',
    )
    discounted_cash_flows = _legacy_discounted_cash_flow(row)
    accrued = _expected_accrued(row)

    assert accrued > 0.0
    assert row['AccruedInterest'] == pytest.approx(accrued)
    assert row['DirtyValue'] == pytest.approx(discounted_cash_flows)
    assert row['CleanPrice'] == pytest.approx(discounted_cash_flows - accrued)
    assert row['DirtyPriceRatio'] == pytest.approx(discounted_cash_flows / row['FaceValue'])


def test_frn_remains_par_plus_accrued():
    row = _price_row(
        SecurityType='FRN',
        CouponRate=0.0,
        OriginalMaturityYears=2.0,
        MaturityDate=pd.Timestamp('2027-11-15'),
        AccruedInterest_FRN=1.75,
        MaturityCategory=pd.NA,
    )

    assert row['CleanPrice'] == pytest.approx(100.0)
    assert row['AccruedInterest'] == pytest.approx(1.75)
    assert row['DirtyValue'] == pytest.approx(101.75)
    assert row['DirtyPriceRatio'] == pytest.approx(1.0175)


def test_zero_coupon_bill_valuation_is_unchanged():
    row = _price_row(
        CouponRate=0.0,
        IssueDate=pd.Timestamp('2025-01-15'),
        MaturityDate=pd.Timestamp('2025-07-15'),
        OriginalMaturityYears=0.5,
        MaturityCategory='bills',
    )
    discounted_value = _legacy_discounted_cash_flow(row)

    assert row['AccruedInterest'] == pytest.approx(0.0)
    assert row['CleanPrice'] == pytest.approx(discounted_value)
    assert row['DirtyValue'] == pytest.approx(discounted_value)
    assert row['DirtyPriceRatio'] == pytest.approx(discounted_value / row['FaceValue'])
