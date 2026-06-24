from __future__ import annotations

import pytest

from budget_interest import (
    bill_discount_amortization,
    build_average_interest_rate_diagnostic,
    build_net_interest_diagnostic,
    component_total,
    day_count_fraction,
    fixed_coupon_accrual,
    frn_accrual,
    simulated_average_interest_rate_pct,
    tips_principal_indexation,
)


def test_fixed_coupon_accrual_uses_day_count_fraction():
    fraction = day_count_fraction("2026-01-01", "2026-04-01")

    assert fraction == pytest.approx(90.0 / 365.0)
    assert fixed_coupon_accrual(100.0, 0.04, fraction) == pytest.approx(100.0 * 0.04 * 90.0 / 365.0)


def test_bill_discount_amortization_recognizes_discount_over_term():
    assert bill_discount_amortization(100.0, 98.0, elapsed_days=91.0, term_days=182.0) == pytest.approx(1.0)


def test_frn_accrual_uses_benchmark_plus_spread():
    assert frn_accrual(100.0, benchmark_rate=0.05, fixed_spread=0.0025, day_fraction=0.5) == pytest.approx(
        2.625
    )


def test_frn_accrual_rejects_percentage_point_rates():
    with pytest.raises(ValueError, match="decimal annual rate"):
        frn_accrual(100.0, benchmark_rate=5.0, fixed_spread=0.25, day_fraction=0.5)


def test_signed_tips_principal_indexation_allows_deflation():
    assert tips_principal_indexation(100.0, beginning_index_ratio=1.00, ending_index_ratio=1.03) == pytest.approx(3.0)
    assert tips_principal_indexation(100.0, beginning_index_ratio=1.00, ending_index_ratio=0.98) == pytest.approx(
        -2.0
    )


def test_net_interest_diagnostic_does_not_create_cbo_plug():
    components = [
        {"component_key": "fixed_coupon_accrual", "amount_bil": 700.0},
        {"component_key": "bill_discount_amortization", "amount_bil": 125.0},
        {"component_key": "frn_accrual", "amount_bil": 50.0},
        {"component_key": "tips_principal_indexation", "amount_bil": -25.0},
    ]

    diagnostic = build_net_interest_diagnostic(cbo_net_interest_bil=1000.0, components=components)

    assert component_total(components) == pytest.approx(850.0)
    assert diagnostic["modeled_net_interest_bil"] == pytest.approx(850.0)
    assert diagnostic["residual_bil"] == pytest.approx(150.0)
    assert diagnostic["threshold_status"] == "red"
    assert "plug" not in diagnostic
    assert diagnostic["claim_status"] == "red_diagnostic_only_nonbinding"


def test_complete_scope_residual_blocks_reproduction_claim_without_binding_model():
    diagnostic = build_net_interest_diagnostic(
        cbo_net_interest_bil=1000.0,
        components=[
            {"component_key": "fixed_coupon_accrual", "amount_bil": 700.0},
            {"component_key": "bill_discount_amortization", "amount_bil": 180.0},
            {"component_key": "frn_accrual", "amount_bil": 75.0},
            {"component_key": "signed_tips_principal_indexation", "amount_bil": 25.0},
        ],
        scope_status="complete",
        calibration_mode="not_calibrated",
    )

    assert diagnostic["threshold_status"] == "warning"
    assert diagnostic["claim_status"] == "blocks_complete_net_interest_reproduction_claim"


def test_complete_scope_claim_rejected_when_components_are_missing():
    with pytest.raises(ValueError, match="missing components"):
        build_net_interest_diagnostic(
            cbo_net_interest_bil=1000.0,
            components=[
                {"component_key": "fixed_coupon_accrual", "amount_bil": 700.0},
                {"component_key": "bill_discount_amortization", "amount_bil": 180.0},
                {"component_key": "signed_tips_principal_indexation", "amount_bil": 25.0},
            ],
            scope_status="complete",
        )


def test_complete_scope_red_residual_is_hard_release_failure_for_complete_claim():
    diagnostic = build_net_interest_diagnostic(
        cbo_net_interest_bil=1000.0,
        components=[
            {"component_key": "fixed_coupon_accrual", "amount_bil": 600.0},
            {"component_key": "bill_discount_amortization", "amount_bil": 180.0},
            {"component_key": "frn_accrual", "amount_bil": 75.0},
            {"component_key": "signed_tips_principal_indexation", "amount_bil": 25.0},
        ],
        scope_status="complete",
    )

    assert diagnostic["threshold_status"] == "red"
    assert diagnostic["claim_status"] == "hard_release_failure_complete_net_interest_reconciliation_claim"


def test_simulated_average_interest_rate_diagnostic_is_reporting_only():
    simulated_rate = simulated_average_interest_rate_pct(
        modeled_net_interest_bil=980.0,
        average_debt_held_public_bil=32000.0,
    )
    diagnostic = build_average_interest_rate_diagnostic(
        fiscal_year=2026,
        cbo_average_interest_rate_pct=3.25,
        simulated_average_rate_pct=simulated_rate,
    )

    assert simulated_rate == pytest.approx(3.0625)
    assert diagnostic["simulated_average_interest_rate_pct"] == pytest.approx(3.0625)
    assert diagnostic["average_interest_rate_residual_pct"] == pytest.approx(-0.1875)
    assert diagnostic["threshold_status"] == "ok"
    assert diagnostic["runtime_role"] == "diagnostic_only"
