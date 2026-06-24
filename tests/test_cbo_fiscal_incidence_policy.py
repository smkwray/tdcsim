from __future__ import annotations

import pandas as pd
import pytest

from cbo_policy_bundle import (
    allocate_signed_primary_flow,
    build_fiscal_incidence_policy_rows,
    build_fiscal_incidence_sensitivity_results,
    build_net_interest_bridge_rows,
    validate_fiscal_incidence_policy_rows,
)
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_fiscal_incidence_policy, load_net_interest_bridge


def test_cbo_baseline_incidence_has_central_and_required_sensitivities(tmp_path):
    rows = build_fiscal_incidence_policy_rows(
        scenario_id="baseline",
        signed_net_primary_flow_bil=813.727,
    )
    csv_path = tmp_path / "tdcsim_fiscal_incidence_policy.csv"
    write_forecast_rows_csv(csv_path, rows)

    loaded = load_fiscal_incidence_policy(csv_path)

    assert set(zip(loaded["du_share"], loaded["ru_share"])) == {(0.99, 0.01), (1.00, 0.00), (0.95, 0.05)}
    central = loaded[loaded["policy_id"] == "baseline_central_99du_1ru"].iloc[0]
    assert central["policy_mode"] == "explicit_scenario_assumption"
    assert central["incidence_basis"] == "signed_net_primary_proxy"
    assert pd.isna(central["primary_outlays_bil"])
    assert pd.isna(central["primary_receipts_bil"])
    assert central["du_share"] + central["ru_share"] + central["foreign_share"] + central["other_share"] == pytest.approx(
        1.0
    )


def test_missing_or_gross_signed_net_policy_is_rejected():
    with pytest.raises(ValueError, match="required"):
        validate_fiscal_incidence_policy_rows([])

    rows = build_fiscal_incidence_policy_rows(
        scenario_id="baseline",
        signed_net_primary_flow_bil=813.727,
    )
    invalid = [{**rows[0], "primary_outlays_bil": 1000.0}]

    with pytest.raises(ValueError, match="gross fields"):
        validate_fiscal_incidence_policy_rows(invalid)


def test_incidence_split_does_not_change_budget_or_tga_totals_for_deficit_or_surplus():
    deficit_rows = build_fiscal_incidence_policy_rows(
        scenario_id="baseline",
        signed_net_primary_flow_bil=813.727,
    )
    surplus_rows = build_fiscal_incidence_policy_rows(
        scenario_id="surplus_case",
        signed_net_primary_flow_bil=-125.0,
    )

    for row in deficit_rows + surplus_rows:
        allocation = allocate_signed_primary_flow(row)
        assert allocation["aggregate_signed_primary_flow_bil"] == pytest.approx(row["signed_net_primary_flow_bil"])
        assert (
            allocation["du_flow_bil"]
            + allocation["ru_flow_bil"]
            + allocation["foreign_flow_bil"]
            + allocation["other_flow_bil"]
        ) == pytest.approx(row["signed_net_primary_flow_bil"])


def test_required_incidence_sensitivity_results_prove_aggregate_invariance():
    rows = build_fiscal_incidence_policy_rows(
        scenario_id="baseline",
        signed_net_primary_flow_bil=813.727,
    )

    results = build_fiscal_incidence_sensitivity_results(rows)

    assert set(zip((row["du_share"] for row in results), (row["ru_share"] for row in results))) == {
        (0.99, 0.01),
        (1.00, 0.00),
        (0.95, 0.05),
    }
    for result in results:
        assert result["result_status"] == "aggregate_invariant"
        assert result["aggregate_signed_primary_flow_invariant"] is True
        assert result["allocation_total_invariant"] is True
        assert result["budget_total_invariant"] is True
        assert result["tga_total_invariant"] is True
        assert result["allocated_total_bil"] == pytest.approx(813.727)
        assert result["aggregate_budget_flow_bil"] == pytest.approx(813.727)
        assert result["aggregate_tga_flow_bil"] == pytest.approx(813.727)


def test_incidence_sensitivity_results_reject_missing_required_split():
    rows = build_fiscal_incidence_policy_rows(
        scenario_id="baseline",
        signed_net_primary_flow_bil=813.727,
        splits=(("central_99du_1ru", 0.99, 0.01), ("sensitivity_100du_0ru", 1.00, 0.00)),
    )

    with pytest.raises(ValueError, match="required incidence sensitivities missing"):
        build_fiscal_incidence_sensitivity_results(rows)


def test_net_interest_bridge_uses_explicit_components_and_nonbinding_cbo_check(tmp_path):
    rows = build_net_interest_bridge_rows(
        scenario_id="baseline",
        fiscal_year=2026,
        source_vintage="cbo_2026_02_baseline",
        cbo_reported_net_interest_bil=1038.976,
        components=[
            {"component_key": "fixed_coupon_accrual", "amount_bil": 700.0},
            {"component_key": "bill_discount_amortization", "amount_bil": 125.0},
            {"component_key": "frn_accrual", "amount_bil": 50.0},
            {"component_key": "signed_tips_principal_indexation", "amount_bil": -25.0},
        ],
    )
    csv_path = tmp_path / "tdcsim_net_interest_bridge.csv"
    write_forecast_rows_csv(csv_path, rows)

    loaded = load_net_interest_bridge(csv_path)

    assert "opaque_plug" not in set(loaded["component_key"])
    assert "cbo_reported_net_interest_check" in set(loaded["component_key"])
    cbo_check = loaded[loaded["component_key"] == "cbo_reported_net_interest_check"].iloc[0]
    assert bool(cbo_check["include_in_budget_interest"]) is False
    assert cbo_check["runtime_role"] == "memo_only"
    assert cbo_check["amount_bil"] == pytest.approx(1038.976)


def test_net_interest_bridge_rejects_opaque_plug_component():
    with pytest.raises(ValueError, match="opaque plug"):
        build_net_interest_bridge_rows(
            scenario_id="baseline",
            fiscal_year=2026,
            source_vintage="cbo_2026_02_baseline",
            components=[
                {
                    "component_key": "opaque_plug",
                    "amount_bil": 188.976,
                }
            ],
        )
