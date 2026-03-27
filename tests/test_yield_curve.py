"""Yield curve interpolation verification tests.

Replaces the standalone verify_yield_curve.py script with proper pytest functions.
"""
import numpy as np
import pytest

from simulation_core import get_yield_for_maturity

YIELD_CURVE_YEARS = [0.25, 2.0, 5.0, 10.0, 20.0, 30.0]
YIELD_CURVE_RATES = [0.05, 0.045, 0.048, 0.052, 0.055, 0.056]


def test_pchip_no_nans():
    """PCHIP interpolation should not produce NaN for in-range maturities."""
    maturities = np.linspace(0.25, 30, 100)
    results = [
        get_yield_for_maturity(m, YIELD_CURVE_YEARS, YIELD_CURVE_RATES, method='pchip')
        for m in maturities
    ]
    assert not any(np.isnan(results)), "PCHIP produced NaN values"


def test_linear_no_nans():
    """Linear interpolation should not produce NaN for in-range maturities."""
    maturities = np.linspace(0.25, 30, 100)
    results = [
        get_yield_for_maturity(m, YIELD_CURVE_YEARS, YIELD_CURVE_RATES, method='linear')
        for m in maturities
    ]
    assert not any(np.isnan(results)), "Linear produced NaN values"


def test_interpolation_at_knots():
    """At knot points, interpolated rate should match input rate."""
    for y, r in zip(YIELD_CURVE_YEARS, YIELD_CURVE_RATES):
        result = get_yield_for_maturity(y, YIELD_CURVE_YEARS, YIELD_CURVE_RATES, method='pchip')
        assert abs(result - r) < 1e-6, f"PCHIP at {y}y: expected {r}, got {result}"


def test_extrapolation_beyond_range():
    """Maturity beyond max knot should return a finite value (not NaN)."""
    result = get_yield_for_maturity(35.0, YIELD_CURVE_YEARS, YIELD_CURVE_RATES)
    assert not np.isnan(result), "Extrapolation beyond 30y returned NaN"


def test_zero_maturity():
    """Very short maturity should return a finite value."""
    result = get_yield_for_maturity(0.01, YIELD_CURVE_YEARS, YIELD_CURVE_RATES)
    assert not np.isnan(result), "Near-zero maturity returned NaN"
