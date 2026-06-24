import pytest

from tdcsim_cbo.transforms.rates import (
    TransformError,
    apply_cpi_override,
    apply_frn_override,
    apply_nominal_yield_curve_override,
    apply_tips_real_yield_override,
)


def test_nominal_parallel_shift_updates_decimal_and_percent_units() -> None:
    rows = [
        {
            "curve_date": "2027-01-01",
            "tenor_years": 0.25,
            "nominal_rate_decimal": 0.04,
            "nominal_rate": 4.0,
            "anchor_3m_pct": 4.0,
            "anchor_10y_pct": 4.5,
            "source_role": "hard_input",
        }
    ]

    transformed = apply_nominal_yield_curve_override(rows, {"mode": "parallel_bp", "shock_bp": 75})

    assert transformed[0]["nominal_rate_decimal"] == pytest.approx(0.0475)
    assert transformed[0]["nominal_rate"] == pytest.approx(4.75)
    assert transformed[0]["anchor_3m_pct"] == pytest.approx(4.75)
    assert transformed[0]["anchor_10y_pct"] == pytest.approx(5.25)
    assert transformed[0]["source_role"] == "scenario_assumption"


def test_nominal_key_rate_shift_interpolates_on_log_tenor() -> None:
    rows = [{"curve_date": "2027-01-01", "tenor_years": 1.0, "nominal_rate_decimal": 0.03, "nominal_rate": 3.0}]
    override = {
        "mode": "key_rate_bp",
        "interpolation": "log_tenor_linear",
        "shocks": [
            {"tenor_years": 0.25, "shock_bp": 100},
            {"tenor_years": 4.0, "shock_bp": 200},
        ],
    }

    transformed = apply_nominal_yield_curve_override(rows, override)

    assert transformed[0]["nominal_rate_decimal"] == pytest.approx(0.045)
    assert transformed[0]["nominal_rate"] == pytest.approx(4.5)


def test_nominal_full_surface_replacement_requires_columns() -> None:
    replacement = [{"curve_date": "2027-01-01", "tenor_years": 10.0, "nominal_rate_decimal": 0.05}]

    transformed = apply_nominal_yield_curve_override([], {"mode": "full_surface_file"}, replacement_rows=replacement)

    assert transformed == [{**replacement[0], "source_role": "scenario_assumption", "scenario_transform": "full_surface_file"}]


def test_frn_linked_to_nominal_curve_uses_three_month_rate_plus_spread() -> None:
    frn_rows = [{"period_start": "2027-01-01", "period_end": "2027-01-02", "benchmark_rate_decimal": 0.02, "auction_high_rate_decimal": 0.02, "money_market_yield_decimal": 0.02}]
    nominal_rows = [
        {"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.041},
        {"curve_date": "2027-01-01", "tenor_years": 10.0, "nominal_rate_decimal": 0.05},
    ]

    transformed = apply_frn_override(
        frn_rows,
        {"mode": "linked_to_nominal_curve", "spread_bp": 12},
        nominal_curve_rows=nominal_rows,
    )

    assert transformed[0]["benchmark_rate_decimal"] == pytest.approx(0.0422)
    assert transformed[0]["auction_high_rate_decimal"] == pytest.approx(0.0422)
    assert transformed[0]["rate_source_family"] == "scenario_nominal_curve_3m_linked"


def test_frn_absolute_path_file_uses_compiler_replacement_rows() -> None:
    replacement = [{"period_start": "2027-01-01", "period_end": "2027-01-02", "benchmark_rate_decimal": 0.07}]

    transformed = apply_frn_override([], {"mode": "absolute_path_file"}, replacement_rows=replacement)

    assert transformed[0]["benchmark_rate_decimal"] == 0.07
    assert transformed[0]["scenario_transform"] == "absolute_path_file"


def test_cpi_annualized_shift_preserves_opening_reference_and_changes_terminal_growth() -> None:
    rows = [
        {"month": "2027-01-01", "cbo_cpi_u_index": 100.0, "tips_cpi_u_index": 101.0, "anchor_reference_cpi": 99.0, "terminal_annualized_cpi_growth_decimal": 0.02},
        {"month": "2028-01-01", "cbo_cpi_u_index": 102.0, "tips_cpi_u_index": 103.0, "anchor_reference_cpi": 99.0, "terminal_annualized_cpi_growth_decimal": 0.02},
    ]

    transformed = apply_cpi_override(rows, {"mode": "annualized_inflation_shift_bp", "shock_bp": 100, "terminal_rule": "carry_last_scenario_growth"})

    assert transformed[0]["cbo_cpi_u_index"] == pytest.approx(100.0)
    assert transformed[0]["anchor_reference_cpi"] == 99.0
    assert transformed[1]["cbo_cpi_u_index"] == pytest.approx(103.02)
    assert transformed[1]["terminal_annualized_cpi_growth_decimal"] == pytest.approx(0.03)
    assert transformed[1]["scenario_terminal_rule"] == "carry_last_scenario_growth"


def test_cpi_monthly_path_file_replacement_is_supported() -> None:
    replacement = [{"month": "2027-01-01", "tips_cpi_u_index": 120.0}]

    transformed = apply_cpi_override([], {"mode": "monthly_path_file"}, replacement_rows=replacement)

    assert transformed[0]["tips_cpi_u_index"] == 120.0


def test_tips_linked_recompute_uses_nominal_curve_and_expected_inflation() -> None:
    rows = [{"curve_date": "2027-01-01", "tenor_years": 10.0, "nominal_rate_decimal": 0.04, "expected_inflation_decimal": 0.02, "real_yield_decimal": 0.02, "real_coupon_decimal": 0.02}]
    nominal = [{"curve_date": "2027-01-01", "tenor_years": 10.0, "nominal_rate_decimal": 0.055}]

    transformed = apply_tips_real_yield_override(
        rows,
        {"mode": "linked_recompute", "additional_parallel_bp": 25},
        nominal_curve_rows=nominal,
    )

    assert transformed[0]["nominal_rate_decimal"] == pytest.approx(0.055)
    assert transformed[0]["real_yield_decimal"] == pytest.approx(0.0375)
    assert transformed[0]["real_coupon_decimal"] == pytest.approx(0.0375)


def test_tips_absolute_path_file_requires_replacement_rows() -> None:
    with pytest.raises(TransformError, match="replacement_rows"):
        apply_tips_real_yield_override([], {"mode": "absolute_path_file"})
