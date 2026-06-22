from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_tdc import (
    CANONICAL_TIER2_ROW,
    COMPONENT_ANCHORED_CANONICAL_ROW,
    REGRESSION_BANK_ONLY_ROW,
    REGRESSION_DI_ROW,
    apply_auction_absorption_to_tdc_panel,
    build_mechanism_component_validation,
    build_historical_replay_tdc_panel,
    build_modern_canonical_formula_crosscheck,
    build_selected_tdc_panel,
)


def test_modern_canonical_formula_rebuilds_tdcest_row():
    quarterly = pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "fed_tsy_tx": 10.0,
                "us_chartered_tsy_tx": 20.0,
                "foreign_offices_tsy_tx": 3.0,
                "affiliated_areas_tsy_tx": 2.0,
                "np_credit_unions_tsy_tx": 5.0,
                "row_tsy_tx": 7.0,
                "treasury_operating_cash_tx": -11.0,
                "fed_remit_or_deferred": 13.0,
                "fed_tsy_coupon_interest_proxy": 1.0,
                "bank_tier2_component_interest_proxy": 2.0,
                "row_tier2_component_interest_proxy": 3.0,
                "credit_union_tier2_component_interest_proxy": 4.0,
                "mmf_rrp_adjustment_prop": 6.0,
            }
        ]
    )
    expected = 10.0 + (20.0 + 3.0 + 2.0 + 5.0) + 7.0 - (-11.0) + 13.0 - 1.0 - 2.0 - 3.0 - 4.0 + 6.0
    estimates = pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                CANONICAL_TIER2_ROW: expected,
                COMPONENT_ANCHORED_CANONICAL_ROW: expected,
            }
        ]
    )

    crosscheck = build_modern_canonical_formula_crosscheck(quarterly, estimates)

    assert crosscheck.iloc[0]["canonical_tier2_recomputed_mil"] == pytest.approx(expected)
    assert crosscheck.iloc[0]["canonical_formula_difference_mil"] == pytest.approx(0.0)
    assert crosscheck.iloc[0]["component_anchored_difference_mil"] == pytest.approx(0.0)
    assert crosscheck.iloc[0]["canonical_formula_status"] == "matched"


def test_selected_tdc_panel_uses_exact_ladder_and_transfer_memo():
    dates = pd.to_datetime(["2002-03-31", "2011-03-31", "2014-03-31", "2022-03-31"])
    estimates = pd.DataFrame(
        {
            "date": dates,
            CANONICAL_TIER2_ROW: [pd.NA, pd.NA, pd.NA, 400.0],
        }
    )
    regression = pd.DataFrame(
        {
            "date": dates,
            REGRESSION_DI_ROW: [100.0, 200.0, 300.0, 399.0],
            REGRESSION_BANK_ONLY_ROW: [90.0, 190.0, 280.0, 389.0],
            "tier2_regression_di_method_tier": [
                "pre_component_h15_scaled_backcast",
                "component_pool_wamest_bucket_backcast",
                "component_pool_wamest_bucket_backcast",
                "constrained_component",
            ],
        }
    )
    components = pd.DataFrame(
        {
            "date": dates,
            "minus_treasury_operating_cash_tx": [1.0, 1.0, 1.0, 1.0],
            "fed_remit_positive": [2.0, 2.0, 2.0, 2.0],
            "du_noninterest_outlay_proxy": [10.0, 10.0, 10.0, 10.0],
            "du_receipt_proxy": [3.0, 3.0, 3.0, 3.0],
            "du_coupon_proxy_selected_narrow": [4.0, 4.0, 4.0, 4.0],
        }
    )
    fiscal = pd.DataFrame(
        {
            "date": dates,
            "treasury_total_outlays_proxy": [1000.0, 1000.0, 1000.0, 1000.0],
            "treasury_total_receipts_proxy": [700.0, 700.0, 700.0, 700.0],
        }
    )

    panel = build_selected_tdc_panel(
        {
            "tdc_estimates": estimates,
            "tdc_tier2_regression_series": regression,
            "tdc_components": components,
            "tdc_du_fiscal_flow_research": fiscal,
        }
    )

    assert panel["selected_tdc_value_mil"].tolist() == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert panel["selected_tdc_series_key"].tolist() == [
        REGRESSION_DI_ROW,
        REGRESSION_DI_ROW,
        REGRESSION_DI_ROW,
        CANONICAL_TIER2_ROW,
    ]
    assert panel["replay_tdc_method_label"].tolist() == [
        "tier2_regression_di_pre_component_h15_scaled_backcast_mmf_rrp_structural_zero",
        "tier2_regression_di_component_pool_wamest_bucket_backcast_mmf_rrp_structural_zero",
        "tier2_regression_di_component_pool_wamest_bucket_backcast_mmf_rrp_prop",
        "canonical_tier2_component_anchored_di_mmf_rrp_prop",
    ]
    assert panel["assumed_treasury_to_ru_transfer_mil"].tolist() == pytest.approx([3.0, 3.0, 3.0, 3.0])
    assert panel["tdc_unobserved_ru_transfer_contribution_mil"].tolist() == pytest.approx([-3.0, -3.0, -3.0, -3.0])
    assert panel["treasury_to_ru_transfer_share_of_deficit"].tolist() == pytest.approx([0.01, 0.01, 0.01, 0.01])
    assert panel["tdc_secondary_trades_mil"].isna().all()


def test_selected_tdc_panel_can_use_zero_treasury_to_ru_transfer_share():
    dates = pd.to_datetime(["2022-03-31"])
    panel = build_selected_tdc_panel(
        {
            "tdc_estimates": pd.DataFrame({ "date": dates, CANONICAL_TIER2_ROW: [400.0] }),
            "tdc_tier2_regression_series": pd.DataFrame(
                {
                    "date": dates,
                    REGRESSION_DI_ROW: [399.0],
                    REGRESSION_BANK_ONLY_ROW: [389.0],
                    "tier2_regression_di_method_tier": ["constrained_component"],
                }
            ),
            "tdc_components": pd.DataFrame(
                {
                    "date": dates,
                    "minus_treasury_operating_cash_tx": [1.0],
                    "fed_remit_positive": [2.0],
                    "du_noninterest_outlay_proxy": [10.0],
                    "du_receipt_proxy": [3.0],
                    "du_coupon_proxy_selected_narrow": [4.0],
                }
            ),
            "tdc_du_fiscal_flow_research": pd.DataFrame(
                {
                    "date": dates,
                    "treasury_total_outlays_proxy": [1000.0],
                    "treasury_total_receipts_proxy": [700.0],
                }
            ),
        },
        treasury_to_ru_transfer_share_of_deficit=0.0,
    )

    assert panel.loc[0, "deficit_mil"] == pytest.approx(300.0)
    assert panel.loc[0, "treasury_to_ru_transfer_share_of_deficit"] == pytest.approx(0.0)
    assert panel.loc[0, "assumed_treasury_to_ru_transfer_mil"] == pytest.approx(0.0)
    assert panel.loc[0, "tdc_unobserved_ru_transfer_contribution_mil"] == pytest.approx(-0.0)


def test_apply_auction_absorption_to_tdc_panel_updates_signed_component_and_residual():
    panel = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_tdc_value_mil": 100.0,
                "tdc_fiscal_flow_mil": 20.0,
                "tdc_debt_service_mil": 5.0,
                "tdc_auction_absorption_mil": 0.0,
                "tdc_other_mil": 1.0,
                "tdc_secondary_trades_mil": pd.NA,
                "secondary_trades_measurement_status": "unobserved_not_zero_evidence",
            }
        ]
    )
    auction = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "broad_investor_class": "investment_funds",
                "tdcsim_holder": "Private",
                "tdc_absorption_role": "identified_du_primary_allotment_gross_signed_negative",
                "is_bridge_class": False,
                "included_in_identified_primary_allotment": True,
                "included_in_tdc_auction_absorption": True,
                "auction_count": 1,
                "allotment_amount": 30_000_000.0,
                "allotment_amount_mil": 30.0,
                "signed_tdc_auction_absorption_mil": -30.0,
                "unique_auction_accepted_amount_mil": 101.0,
                "unique_auction_offering_amount_mil": 100.0,
                "unique_auction_allotment_total_clean_mil": 100.0,
                "quarter_allotment_reconciliation_gap_mil": 1.0,
                "source_status": "observed",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "broad_investor_class": "dealers",
                "tdcsim_holder": "dealer_bridge",
                "tdc_absorption_role": "dealer_bridge_not_final_holder",
                "is_bridge_class": True,
                "included_in_identified_primary_allotment": False,
                "included_in_tdc_auction_absorption": False,
                "auction_count": 1,
                "allotment_amount": 60_000_000.0,
                "allotment_amount_mil": 60.0,
                "signed_tdc_auction_absorption_mil": 0.0,
                "unique_auction_accepted_amount_mil": 101.0,
                "unique_auction_offering_amount_mil": 100.0,
                "unique_auction_allotment_total_clean_mil": 100.0,
                "quarter_allotment_reconciliation_gap_mil": 1.0,
                "source_status": "observed",
                "evidence_label": "observed",
            },
        ]
    )

    out = apply_auction_absorption_to_tdc_panel(panel, auction)
    validation = build_mechanism_component_validation(out)

    assert out.loc[0, "tdc_auction_absorption_mil"] == pytest.approx(-30.0)
    assert out.loc[0, "auction_identified_du_absorption_gross_mil"] == pytest.approx(30.0)
    assert out.loc[0, "auction_bridge_or_unresolved_mil"] == pytest.approx(60.0)
    assert out.loc[0, "tdc_residual_mil"] == pytest.approx(104.0)
    assert "dealer_and_bridge_amounts_explicit" in out.loc[0, "auction_absorption_measurement_status"]
    assert validation.loc[validation["component"] == "auction_absorption", "value_mil"].iloc[0] == pytest.approx(-30.0)


def test_live_tdc_panel_has_expected_coverage_and_formula_match():
    panel, formula, manifest = build_historical_replay_tdc_panel(start_quarter="2002Q1", end_quarter="2025Q4")

    assert len(panel.index) == 96
    assert len(formula.index) == 16
    assert manifest["source_key"].isin(["quarterly_inputs", "tdc_estimates"]).any()
    assert formula["canonical_formula_difference_mil"].abs().max() <= 1e-6
    assert formula["component_anchored_difference_mil"].abs().max() <= 1e-6
    assert panel.groupby("replay_tdc_method_tier").size().to_dict() == {
        "component_pool_wamest_bucket_backcast": 47,
        "constrained_component": 16,
        "pre_component_h15_scaled_backcast": 33,
    }
