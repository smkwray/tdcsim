from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_pricing import apply_model_pricing, load_nominal_yield_curve, load_real_yield_curve


def test_fixed_coupon_pricing_uses_quarter_end_curve_and_is_rate_sensitive():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "Fixed",
                "IssueDate": "2020-03-31",
                "MaturityDate": "2035-03-31",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "CouponRate": 0.02,
                "IssueYieldAtIssue": 0.01,
            }
        ]
    )
    low_curve = pd.DataFrame(
        [
            {"observation_date": "2025-03-31", "tenor_years": 10.0, "yield_decimal": 0.02},
        ]
    )
    high_curve = pd.DataFrame(
        [
            {"observation_date": "2025-03-31", "tenor_years": 10.0, "yield_decimal": 0.05},
        ]
    )

    low = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=low_curve).iloc[0]
    high = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=high_curve).iloc[0]

    assert low["pricing_method"] == "coupon_pv_from_current_curve"
    assert low["DiscountYield"] == pytest.approx(0.02)
    assert high["DiscountYield"] == pytest.approx(0.05)
    assert high["DirtyPriceRatio"] < low["DirtyPriceRatio"] - 0.01


def test_bill_pricing_prefers_curve_over_issue_yield():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "Bill",
                "IssueDate": "2025-01-01",
                "MaturityDate": "2025-06-30",
                "FaceValue": 100.0,
                "CouponRate": 0.0,
                "IssueYieldAtIssue": 0.01,
            }
        ]
    )
    curve = pd.DataFrame(
        [
            {"observation_date": "2025-03-31", "tenor_years": 0.25, "yield_decimal": 0.05},
        ]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=curve).iloc[0]

    assert priced["pricing_method"] == "zero_coupon_current_curve_discount"
    assert priced["DiscountYield"] == pytest.approx(0.05)
    assert priced["DirtyPriceRatio"] < 1.0


def test_frn_pricing_uses_index_plus_spread_par_approximation_and_loaded_accrued_interest():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "FRN",
                "IssueDate": "2025-01-01",
                "MaturityDate": "2027-01-01",
                "FaceValue": 100.0,
                "CouponRate": 0.0,
                "IssueYieldAtIssue": 0.0,
                "AccruedInterest_FRN": 1.25,
                "BenchmarkRate_FRN": 0.0525,
                "FixedSpread": 0.0015,
            }
        ]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=pd.DataFrame()).iloc[0]

    assert priced["pricing_method"] == "frn_index_plus_spread_par_approximation"
    assert priced["pricing_curve_status"] == "frn_index_plus_spread_par_approximation"
    assert priced["DiscountYield"] == pytest.approx(0.054)
    assert priced["CleanPrice"] == pytest.approx(100.0)
    assert priced["AccruedInterest"] == pytest.approx(1.25)
    assert priced["DirtyPriceRatio"] == pytest.approx(1.0125)
    assert priced["DirtyValue"] == pytest.approx(101.25)


def test_tips_pricing_uses_adjusted_principal_and_discloses_nominal_proxy():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "TIPS",
                "IssueDate": "2020-01-15",
                "MaturityDate": "2030-01-15",
                "FaceValue": 100.0,
                "OriginalPrincipal": 90.0,
                "AdjustedPrincipal": 108.0,
                "IndexRatio": 1.2,
                "CouponRate": 0.01,
                "IssueYieldAtIssue": 0.01,
            }
        ]
    )
    curve = pd.DataFrame(
        [
            {"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": 0.03},
        ]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=curve).iloc[0]

    assert priced["pricing_method"] == "tips_coupon_pv_from_nominal_curve_proxy"
    assert priced["pricing_claim_boundary"] == "model_implied_not_observed_market_price"
    assert priced["DirtyValue"] == pytest.approx(108.0 * priced["DirtyPriceRatio"])


def test_tips_pricing_uses_real_curve_when_available():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "TIPS",
                "IssueDate": "2020-01-15",
                "MaturityDate": "2030-01-15",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 108.0,
                "CouponRate": 0.01,
                "IssueYieldAtIssue": 0.01,
            }
        ]
    )
    nominal_curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": 0.05}]
    )
    real_curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": 0.01}]
    )

    priced = apply_model_pricing(
        portfolio,
        quarter="2025Q1",
        yield_curve=nominal_curve,
        real_yield_curve=real_curve,
    ).iloc[0]

    assert priced["pricing_method"] == "tips_coupon_pv_from_real_curve"
    assert priced["DiscountYield"] == pytest.approx(0.01)
    assert priced["DirtyValue"] == pytest.approx(108.0 * priced["DirtyPriceRatio"])


def test_negative_real_yield_tips_prices_above_zero_real_yield_tips():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "TIPS",
                "IssueDate": "2020-03-31",
                "MaturityDate": "2030-03-31",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "CouponRate": 0.005,
                "IssueYieldAtIssue": 0.005,
                "interest_pay_date_1": "03/31",
                "interest_pay_date_2": "09/30",
            }
        ]
    )
    nominal_curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": 0.03}]
    )
    negative_real_curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": -0.01}]
    )
    zero_real_curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 5.0, "yield_decimal": 0.0}]
    )

    negative = apply_model_pricing(
        portfolio,
        quarter="2025Q1",
        yield_curve=nominal_curve,
        real_yield_curve=negative_real_curve,
    ).iloc[0]
    zero = apply_model_pricing(
        portfolio,
        quarter="2025Q1",
        yield_curve=nominal_curve,
        real_yield_curve=zero_real_curve,
    ).iloc[0]

    assert negative["pricing_method"] == "tips_coupon_pv_from_real_curve"
    assert negative["DiscountYield"] == pytest.approx(-0.01)
    assert negative["DirtyPriceRatio"] > zero["DirtyPriceRatio"]
    assert negative["DirtyPriceRatio"] > 1.0


def test_coupon_pricing_uses_payment_dates_for_accrual_reset_and_cashflows():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "Fixed",
                "IssueDate": "2024-03-31",
                "MaturityDate": "2026-03-31",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "CouponRate": 0.04,
                "IssueYieldAtIssue": 0.04,
                "interest_pay_date_1": "03/31",
                "interest_pay_date_2": "09/30",
            }
        ]
    )
    curve = pd.DataFrame(
        [{"observation_date": "2025-03-31", "tenor_years": 1.0, "yield_decimal": 0.04}]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=curve).iloc[0]

    assert priced["pricing_curve_status"] == "single_tenor_current_curve_scheduled_payment_dates"
    assert priced["AccruedInterest"] == pytest.approx(0.0)
    assert priced["DirtyPriceRatio"] == pytest.approx(1.0, abs=0.001)


def test_coupon_accrual_uses_previous_and_next_payment_dates():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q2",
                "SecurityType": "Fixed",
                "IssueDate": "2024-03-31",
                "MaturityDate": "2026-03-31",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "CouponRate": 0.04,
                "IssueYieldAtIssue": 0.04,
                "interest_pay_date_1": "03/31",
                "interest_pay_date_2": "09/30",
            }
        ]
    )
    curve = pd.DataFrame(
        [{"observation_date": "2025-06-30", "tenor_years": 1.0, "yield_decimal": 0.04}]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q2", yield_curve=curve).iloc[0]

    assert priced["AccruedInterest"] == pytest.approx(2.0 * 91 / 183)


def test_long_nominal_curve_interpolates_between_20y_and_30y():
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "SecurityType": "Fixed",
                "IssueDate": "2020-03-31",
                "MaturityDate": "2050-03-31",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "CouponRate": 0.02,
                "IssueYieldAtIssue": 0.01,
            }
        ]
    )
    curve = pd.DataFrame(
        [
            {"observation_date": "2025-03-31", "tenor_years": 10.0, "yield_decimal": 0.02},
            {"observation_date": "2025-03-31", "tenor_years": 20.0, "yield_decimal": 0.04},
            {"observation_date": "2025-03-31", "tenor_years": 30.0, "yield_decimal": 0.06},
        ]
    )

    priced = apply_model_pricing(portfolio, quarter="2025Q1", yield_curve=curve).iloc[0]

    assert priced["pricing_curve_status"] == "interpolated_current_curve"
    assert priced["DiscountYield"] == pytest.approx(0.049998631074606434)


def test_load_nominal_yield_curve_reads_local_source_spec(tmp_path):
    curve_path = tmp_path / "curve.csv"
    pd.DataFrame(
        [
            {"observation_date": "2025-03-30", "DGS3MO": 4.5},
            {"observation_date": "2025-03-31", "DGS3MO": "."},
        ]
    ).to_csv(curve_path, index=False)

    curve = load_nominal_yield_curve(
        curve_sources={
            "fixture_3m": {
                "path": str(curve_path),
                "value_col": "DGS3MO",
                "tenor_years": 0.25,
            }
        }
    )

    assert len(curve) == 1
    assert curve.iloc[0]["yield_decimal"] == pytest.approx(0.045)
    assert curve.iloc[0]["tenor_years"] == pytest.approx(0.25)


def test_load_real_yield_curve_keeps_negative_real_yields(tmp_path):
    curve_path = tmp_path / "real_curve.csv"
    pd.DataFrame(
        [
            {"observation_date": "2025-03-30", "DFII5": -0.5},
        ]
    ).to_csv(curve_path, index=False)

    curve = load_real_yield_curve(
        curve_sources={
            "fixture_5y_real": {
                "path": str(curve_path),
                "value_col": "DFII5",
                "tenor_years": 5.0,
            }
        }
    )

    assert len(curve) == 1
    assert curve.iloc[0]["yield_decimal"] == pytest.approx(-0.005)
    assert curve.iloc[0]["curve_family"] == "real"
