import pytest

from tdcsim_cbo.transforms.fiscal import (
    FiscalTransformError,
    apply_cash_residual_override,
    apply_debt_target_override,
    apply_fiscal_incidence_override,
    apply_operating_cash_override,
    apply_primary_deficit_override,
)


def _cash_rows() -> list[dict]:
    return [
        {
            "period_end": "2027-01-01",
            "operating_cash_target_bil": 100.0,
            "tga_target_bil": 90.0,
            "ttl_target_bil": 5.0,
            "other_operating_cash_target_bil": 5.0,
            "inflation_index_level": 200.0,
        },
        {
            "period_end": "2028-01-01",
            "operating_cash_target_bil": 120.0,
            "tga_target_bil": 108.0,
            "ttl_target_bil": 6.0,
            "other_operating_cash_target_bil": 6.0,
            "inflation_index_level": 220.0,
        },
    ]


def test_operating_cash_constant_nominal_preserves_component_identity() -> None:
    transformed = apply_operating_cash_override(_cash_rows(), {"mode": "constant_nominal"})

    assert transformed[1]["operating_cash_target_bil"] == pytest.approx(100.0)
    assert transformed[1]["tga_target_bil"] == pytest.approx(90.0)
    assert transformed[1]["construction_mode"] == "scenario_constant_nominal"
    assert transformed[1]["claim_boundary"] == "operating_cash_proxy_not_debt_target_or_issuance_supply"


def test_operating_cash_constant_real_uses_inflation_index() -> None:
    transformed = apply_operating_cash_override(_cash_rows(), {"mode": "constant_real"})

    assert transformed[1]["operating_cash_target_bil"] == pytest.approx(110.0)
    assert transformed[1]["tga_target_bil"] == pytest.approx(99.0)


def test_operating_cash_scale_baseline_scales_all_components() -> None:
    transformed = apply_operating_cash_override(_cash_rows(), {"mode": "scale_baseline", "scale": 1.25})

    assert transformed[0]["operating_cash_target_bil"] == pytest.approx(125.0)
    assert transformed[0]["tga_target_bil"] == pytest.approx(112.5)


def test_operating_cash_component_path_file_accepts_compiler_replacement_rows() -> None:
    replacement = [
        {
            "period_end": "2027-01-01",
            "operating_cash_target_bil": 70.0,
            "tga_target_bil": 60.0,
            "ttl_target_bil": 7.0,
            "other_operating_cash_target_bil": 3.0,
        }
    ]

    transformed = apply_operating_cash_override([], {"mode": "component_path_file"}, replacement_rows=replacement)

    assert transformed[0]["operating_cash_target_bil"] == pytest.approx(70.0)
    assert transformed[0]["scenario_transform"] == "component_path_file"


def test_cash_residual_zero_cannot_affect_funding_or_deficit() -> None:
    rows = [{"period_end": "2027-01-01", "cash_reconciliation_residual_bil": 15.0, "affects_issuance_size": True}]

    transformed = apply_cash_residual_override(rows, {"mode": "zero", "funding_effect": "none"})

    assert transformed[0]["cash_reconciliation_residual_bil"] == 0.0
    assert transformed[0]["affects_issuance_size"] is False
    assert transformed[0]["affects_debt_target"] is False
    assert transformed[0]["runtime_role"] == "reconciliation_only"


def test_cash_residual_tracking_operating_cash_is_reconciliation_only() -> None:
    rows = [{"period_end": "2027-01-01"}, {"period_end": "2028-01-01"}]

    transformed = apply_cash_residual_override(rows, {"mode": "track_operating_cash_target"}, operating_cash_rows=_cash_rows())

    assert transformed[0]["cash_reconciliation_residual_bil"] == pytest.approx(0.0)
    assert transformed[1]["cash_reconciliation_residual_bil"] == pytest.approx(20.0)
    assert transformed[1]["affects_total_deficit"] is False


def test_cash_residual_explicit_path_file_keeps_non_funding_flags() -> None:
    replacement = [{"period_end": "2027-01-01", "cash_reconciliation_residual_bil": 4.0, "affects_issuance_size": True}]

    transformed = apply_cash_residual_override([], {"mode": "explicit_path_file"}, replacement_rows=replacement)

    assert transformed[0]["cash_reconciliation_residual_bil"] == pytest.approx(4.0)
    assert transformed[0]["affects_issuance_size"] is False
    assert transformed[0]["runtime_role"] == "reconciliation_only"


def test_primary_deficit_scale_and_additive_transforms() -> None:
    rows = [{"source_fiscal_year": 2027, "primary_deficit_bil": 10.0, "annual_or_remaining_primary_deficit_bil": 1000.0}]

    scaled = apply_primary_deficit_override(rows, {"mode": "scale_path", "scale": 1.1})
    added = apply_primary_deficit_override(rows, {"mode": "additive_bil", "additive_bil": 5.0})

    assert scaled[0]["primary_deficit_bil"] == pytest.approx(11.0)
    assert scaled[0]["annual_or_remaining_primary_deficit_bil"] == pytest.approx(1100.0)
    assert added[0]["primary_deficit_bil"] == pytest.approx(15.0)


def test_debt_target_fy_endpoint_anchor_interpolates_and_scales_related_column() -> None:
    rows = [
        {
            "source_fiscal_year": 2028,
            "cbo_federal_debt_held_public_target_bil": 100.0,
            "marketable_treasury_public_target_bil": 90.0,
        }
    ]
    override = {
        "mode": "fy_endpoint_anchors",
        "anchors": [{"fiscal_year": 2027, "value_bil": 100.0}, {"fiscal_year": 2029, "value_bil": 200.0}],
    }

    transformed = apply_debt_target_override(rows, override)

    assert transformed[0]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(150.0)
    assert transformed[0]["marketable_treasury_public_target_bil"] == pytest.approx(135.0)


def test_debt_target_absolute_path_file_accepts_compiler_replacement_rows() -> None:
    replacement = [{"period_end": "2027-01-01", "cbo_federal_debt_held_public_target_bil": 100.0}]

    transformed = apply_debt_target_override([], {"mode": "absolute_path_file"}, replacement_rows=replacement)

    assert transformed[0]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(100.0)
    assert transformed[0]["claim_boundary"] == "debt_target_scenario_transform_no_plug"


def test_fiscal_incidence_requires_exact_unit_sum() -> None:
    rows = [{"policy_id": "central"}]
    override = {
        "mode": "static_shares",
        "domestic_ultimate_share": 0.97,
        "rest_of_world_share": 0.02,
        "foreign_official_share": 0.01,
        "other_share": 0.0,
    }

    transformed = apply_fiscal_incidence_override(rows, override)

    assert transformed[0]["du_share"] == pytest.approx(0.97)
    assert transformed[0]["foreign_share"] == pytest.approx(0.01)


def test_fiscal_incidence_rejects_non_unit_sum() -> None:
    with pytest.raises(FiscalTransformError, match="sum to 1.0"):
        apply_fiscal_incidence_override(
            [{}],
            {
                "mode": "static_shares",
                "domestic_ultimate_share": 0.9,
                "rest_of_world_share": 0.2,
                "foreign_official_share": 0.0,
                "other_share": 0.0,
            },
        )
