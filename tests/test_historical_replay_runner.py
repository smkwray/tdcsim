from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from historical_replay import (
    _build_z1_transaction_flow_diagnostics,
    _deterministic_large_artifact_sample,
    _large_portfolio_sample_strata,
)
from simulation_core import run_simulation


def _write_mmf_component_source(path: Path, *, total: float, bills: float, date: str = "2025-03-31") -> None:
    pd.DataFrame(
        [
            {
                "date": date,
                "mmf_tsy_level": total,
                "mmf_tsy_bills_level": bills,
            }
        ]
    ).to_csv(path, index=False)


def test_z1_transaction_flow_diagnostics_convert_fa_saar_to_quarterly_moment():
    observations = pd.DataFrame(
        [
            {
                "quarter": "2024Q4",
                "sector": "abs_issuers",
                "native_sector": "abs_issuers",
                "broad_holder_class": "abs_issuers",
                "measure": "level",
                "value": 100.0,
            },
            {
                "quarter": "2025Q1",
                "sector": "abs_issuers",
                "native_sector": "abs_issuers",
                "broad_holder_class": "abs_issuers",
                "measure": "level",
                "value": 125.0,
            },
            {
                "quarter": "2025Q1",
                "sector": "abs_issuers",
                "native_sector": "abs_issuers",
                "broad_holder_class": "abs_issuers",
                "measure": "transaction",
                "value": 100.0,
            },
        ]
    )
    levels = observations[observations["measure"].eq("level")].copy()

    diagnostics = _build_z1_transaction_flow_diagnostics(observations, levels)
    row = diagnostics[diagnostics["quarter"].eq("2025Q1")].iloc[0]

    assert row["z1_transaction_flow_saar_mil"] == pytest.approx(100.0)
    assert row["z1_transaction_flow_mil"] == pytest.approx(25.0)
    assert row["z1_transaction_flow_conversion"] == "FA_divided_by_4_to_quarterly_moment"
    assert row["implied_valuation_or_other_volume_mil"] == pytest.approx(0.0)


def test_large_artifact_sample_selects_each_declared_positive_stratum():
    frame = pd.DataFrame(
        [
            {"quarter": "2023Q4", "source_sector": "other", "native_sector": "other", "value": 1},
            {"quarter": "2024Q1", "source_sector": "other", "native_sector": "other", "value": 2},
            {"quarter": "2025Q4", "source_sector": "other", "native_sector": "other", "value": 3},
            {"quarter": "2020Q1", "source_sector": "money_market_funds", "native_sector": "x", "value": 4},
            {"quarter": "2020Q2", "source_sector": "holding_companies", "native_sector": "x", "value": 5},
            {"quarter": "2020Q3", "source_sector": "SourceBasisResidual", "native_sector": "x", "value": 6},
        ]
    )

    sample, counts = _deterministic_large_artifact_sample(frame, _large_portfolio_sample_strata)

    assert set(frame["value"]) <= set(sample["value"])
    for stratum, stratum_counts in counts.items():
        if stratum_counts["source_row_count"] > 0:
            assert stratum_counts["selected_row_count"] > 0, stratum


def test_run_simulation_dispatches_historical_replay(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    mmf_component_path = tmp_path / "quarterly_inputs.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 125.0,
                "selected_operating_cash_level_source": "dts_quarter_end_tga_plus_ttl",
                "selected_operating_cash_tx_mil": 12.5,
                "selected_operating_cash_tx_source": "z1_quarterly_treasury_operating_cash_transaction",
                "dts_tga_mil": 100.0,
                "dts_ttl_mil": 25.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "us_chartered_depository_institutions", "broad_holder_class": "banks", "measure": "level", "value": 700.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "money_market_funds", "broad_holder_class": "money_market_cash", "measure": "level", "value": 300.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "CUSIP1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cohorts_path, index=False)
    _write_mmf_component_source(mmf_component_path, total=30.0, bills=30.0)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "sector_value_unit_scale": 10.0,
            "mmf_component_constraints": str(mmf_component_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="replay_fixture")

    assert results.attrs["run_metadata"]["mode"] == "historical_replay"
    assert results.attrs["run_metadata"]["sector_value_unit_scale"] == pytest.approx(10.0)
    assert results.iloc[0]["TOC_Level"] == pytest.approx(125.0)
    assert results.iloc[0]["TOC_Change"] == pytest.approx(12.5)
    assert results.iloc[0]["TGAChange"] == pytest.approx(12.5)
    assert results.iloc[0]["TGA"] + results.iloc[0]["TTL"] == pytest.approx(results.iloc[0]["TOC_Level"])
    assert results.iloc[0]["TotalDebt_Agg"] == pytest.approx(100.0)
    assert results.iloc[0]["Z1_Holder_Total"] == pytest.approx(100.0)
    assert results.iloc[0]["MSPD_Cohort_Total"] == pytest.approx(100.0)
    assert results.iloc[0]["MSPD_Z1_SourceBasisDiff"] == pytest.approx(0.0)
    assert results.iloc[0]["HolderScaleFactor"] == pytest.approx(1.0)
    assert set(results.attrs["period_end_portfolios"]) == {"2025Q1"}
    assert results.attrs["replay_ledger"].iloc[0]["quarter"] == "2025Q1"
    assert portfolio["FaceValue"].sum() == pytest.approx(100.0)
    assert set(portfolio["native_sector"]) == {"us_chartered_depository_institutions", "money_market_funds"}
    assert set(portfolio["broad_holder_class"]) == {"banks", "money_market_cash"}


def test_historical_replay_ffiec_maturity_prior_affects_bank_allocation_and_exports(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    ffiec_path = tmp_path / "ffiec.csv"
    ncua_path = tmp_path / "ncua.csv"
    output_dir = tmp_path / "exports"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 0.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 80.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "us_chartered_depository_institutions", "broad_holder_class": "banks", "measure": "level", "value": 50.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "household_sector", "broad_holder_class": "individuals", "measure": "level", "value": 50.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "SHORT",
                "cusip": "912345SH1",
                "security_type": "bill",
                "issue_date": "2025-01-01",
                "maturity_date": "2025-04-15",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "LONG",
                "cusip": "912345LG1",
                "security_type": "bond",
                "issue_date": "2020-01-01",
                "maturity_date": "2045-01-01",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cohorts_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "reporter_id": 1,
                "bank_class": "all_commercial_banks",
                "treasury_bucket_3m_or_less": 100000.0,
                "treasury_bucket_3_12m": 0.0,
                "treasury_bucket_1_3y": 0.0,
                "treasury_bucket_3_5y": 0.0,
                "treasury_bucket_5_15y": 0.0,
                "treasury_bucket_over_15y": 0.0,
            }
        ]
    ).to_csv(ffiec_path, index=False)
    pd.DataFrame(columns=["date"]).to_csv(ncua_path, index=False)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "output_dir": str(output_dir),
            "ffiec_interest_constraints": str(ffiec_path),
            "ncua_interest_constraints": str(ncua_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, _ = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="ffiec_maturity_prior")

    snapshot = results.attrs["period_end_portfolios"]["2025Q1"]
    bank_short = snapshot[
        snapshot["native_sector"].astype(str).eq("us_chartered_depository_institutions")
        & snapshot["cohort_id"].astype(str).eq("SHORT")
    ]["FaceValue"].sum()
    assert bank_short > 49.0
    maturity_prior = results.attrs["maturity_prior_reconciliation"]
    assert not maturity_prior.empty
    assert "ffiec_bank_maturity_prior" in set(maturity_prior["source_scope"])
    assert (maturity_prior["prior_status"] == "solver_prior_applied").any()
    assert "maturity_prior_reconciliation" in results.attrs["run_metadata"]["historical_replay_output_paths"]


def test_historical_replay_ncua_all_investment_ladder_stays_soft_prior_not_eligibility(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    ffiec_path = tmp_path / "ffiec.csv"
    ncua_path = tmp_path / "ncua.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 0.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 80.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "credit_unions",
                "broad_holder_class": "banks",
                "measure": "level",
                "value": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "SHORT",
                "cusip": "912345SH1",
                "security_type": "bill",
                "issue_date": "2025-01-01",
                "maturity_date": "2025-06-30",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "LONG",
                "cusip": "912345LG1",
                "security_type": "bond",
                "issue_date": "2020-01-01",
                "maturity_date": "2045-01-01",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cohorts_path, index=False)
    pd.DataFrame(columns=["date"]).to_csv(ffiec_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "total_treasuries_amortized_cost": 100_000_000.0,
                "investment_bucket_le_1y": 100_000_000.0,
                "investment_bucket_1_3y": 0.0,
                "investment_bucket_3_5y": 0.0,
                "investment_bucket_5_10y": 0.0,
                "investment_bucket_over_10y": 0.0,
                "fallback_split_basis": "fixture_all_investments_not_treasury_ladder",
            }
        ]
    ).to_csv(ncua_path, index=False)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "ffiec_interest_constraints": str(ffiec_path),
            "ncua_interest_constraints": str(ncua_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="ncua_soft_prior")

    snapshot = results.attrs["period_end_portfolios"]["2025Q1"]
    assert set(snapshot["native_sector"].astype(str)) == {"credit_unions"}
    assert not snapshot["source_sector"].astype(str).str.startswith("credit_unions__").any()
    assert snapshot.groupby("cohort_id")["FaceValue"].sum().to_dict() == pytest.approx(
        {"SHORT": 50.0, "LONG": 50.0}
    )
    assert portfolio.groupby("cohort_id")["FaceValue"].sum().to_dict() == pytest.approx(
        {"SHORT": 50.0, "LONG": 50.0}
    )
    maturity_prior = results.attrs["maturity_prior_reconciliation"]
    ncua_rows = maturity_prior[maturity_prior["source_scope"].astype(str).eq("ncua_credit_union_maturity_proxy")]
    assert not ncua_rows.empty
    assert set(ncua_rows["constraint_role"]) == {"soft_all_investment_prior_only"}
    assert "solver_prior_applied" not in set(ncua_rows["prior_status"])
    assert "soft_solver_prior_applied" in set(ncua_rows["prior_status"])


def test_historical_replay_solves_multiple_quarters_with_stateful_prior(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": -10.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 90.0,
                "dts_ttl_mil": 10.0,
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q2",
                "selected_operating_cash_level_mil": 200.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 100.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 175.0,
                "dts_ttl_mil": 25.0,
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "measure": "level", "value": 100.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q2", "sector": "Banks", "measure": "level", "value": 200.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "CUSIP1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q2",
                "cohort_id": "CUSIP1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 200.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cohorts_path, index=False)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-06-30", scenario_name="two_quarter_replay")

    assert results["TotalDebt_Agg"].tolist() == pytest.approx([100.0, 200.0])
    assert results["DebtHeld_Banks"].tolist() == pytest.approx([100.0, 200.0])
    assert results["TOC_Change"].tolist() == pytest.approx([-10.0, 100.0])
    assert set(results.attrs["period_end_portfolios"]) == {"2025Q1", "2025Q2"}
    assert results.attrs["period_end_portfolios"]["2025Q1"]["FaceValue"].sum() == pytest.approx(100.0)
    assert results.attrs["period_end_portfolios"]["2025Q2"]["FaceValue"].sum() == pytest.approx(200.0)
    assert portfolio["FaceValue"].sum() == pytest.approx(200.0)


def test_historical_replay_exposes_source_basis_difference_and_exports(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    output_dir = tmp_path / "exports"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 125.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 12.5,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 100.0,
                "dts_ttl_mil": 25.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "measure": "level", "value": 80.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "CUSIP1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cohorts_path, index=False)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "output_dir": str(output_dir),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="mismatch_replay")

    row = results.iloc[0]
    assert row["Z1_Holder_Total"] == pytest.approx(80.0)
    assert row["MSPD_Cohort_Total"] == pytest.approx(100.0)
    assert row["MSPD_Z1_SourceBasisDiff"] == pytest.approx(20.0)
    assert row["Replay_CohortResidual"] == pytest.approx(20.0)
    assert pd.isna(row["HolderScaleFactor"])
    assert portfolio["FaceValue"].sum() == pytest.approx(100.0)
    residual_rows = portfolio[portfolio["source_sector"] == "MSPD_Z1_SourceBasisResidual"]
    assert residual_rows["FaceValue"].sum() == pytest.approx(20.0)
    assert set(residual_rows["HolderType"].astype(str)) == {"SourceBasisResidual"}
    assert portfolio.groupby("HolderType")["FaceValue"].sum().to_dict() == pytest.approx(
        {"Banks": 80.0, "SourceBasisResidual": 20.0}
    )

    ledger = results.attrs["replay_ledger"]
    assert ledger.iloc[0]["MSPD_Z1_SourceBasisDiff"] == pytest.approx(20.0)
    assert ledger.iloc[0]["debt_surface_role"] == "synthetic_portfolio_from_mspd_cohorts_and_z1_holders"
    export_paths = results.attrs["run_metadata"]["historical_replay_output_paths"]
    expected_export_subset = [
        "diagnostics",
        "final_portfolio",
        "ledger",
        "event_ledger",
        "event_rollforward",
        "exact_observation_coverage",
            "holder_basis_bridge",
            "maturity_prior_reconciliation",
            "mmf_component_reconciliation",
            "negative_sector_netting_bridge",
            "observation_registry",
            "portfolio_native_sector_similarity",
        "portfolio_constraint_diagnostics",
        "portfolio_snapshots",
        "pricing_scope_diagnostics",
        "replay_input_manifest",
        "results",
        "strips_scope_diagnostics",
        "tips_inflation_monthly_detail",
        "tips_inflation_reconciliation",
        "treasury_interest_expense_diagnostic",
        "unexplained_change_ledger",
        "valuation_scope_diagnostics",
    ]
    assert set(expected_export_subset).issubset(export_paths)
    for name, path in export_paths.items():
        if str(path).endswith(".md"):
            assert pd.notna(path)
            continue
        frame = pd.read_csv(path)
        if name in {
                "auction_absorption_reconciliation",
                "auction_allotment_proxy",
                "fixed_coupon_interest_reconciliation",
                "fixed_coupon_monthly_detail",
                "fixed_coupon_principal_adjustments",
                "frn_cusip_coverage",
                "frn_daily_index_validation",
                "frn_interest_flow_detail",
                "frn_interest_reconciliation",
                "frn_principal_reconciliation",
                    "holder_mix_differentiation",
                    "interest_component_detail",
                "interest_component_detail_sample",
                "interest_proxy_alignment",
                "large_artifact_sample_manifest",
                "mmf_component_reconciliation",
                "negative_sector_netting_bridge",
                "portfolio_snapshots_sample",
                "portfolio_transition_diagnostics",
                "soma_fixed_allocations",
                "soma_holdings",
                "soma_holdout_diagnostics",
                "tips_inflation_monthly_detail",
                "tips_inflation_reconciliation",
                "tips_coupon_detail",
                "tips_principal_identity",
                "valuation_basis_feasibility_certificate",
                "unexplained_change_ledger",
            "treasury_interest_expense_diagnostic",
        }:
            assert frame is not None
            continue
        assert frame.shape[0] >= 1, name
    strips_scope = pd.read_csv(export_paths["strips_scope_diagnostics"])
    assert set(strips_scope["status"]) == {"no_strips_detected"}
    assert strips_scope["strips_like_rows"].sum() == 0
    valuation_scope = pd.read_csv(export_paths["valuation_scope_diagnostics"])
    assert set(valuation_scope["valuation_scope_status"]) == {
        "pricing_values_present"
    }
    assert valuation_scope["rows_with_any_pricing_value"].sum() > 0
    pricing_scope = pd.read_csv(export_paths["pricing_scope_diagnostics"])
    assert set(pricing_scope["claim_boundary"]) == {"model_implied_not_observed_market_price"}
    observation_registry = pd.read_csv(export_paths["observation_registry"])
    assert set(observation_registry["scope"]) >= {"z1_holder_level"}
    event_rollforward = pd.read_csv(export_paths["event_rollforward"])
    assert event_rollforward.iloc[0]["status"] == "closed_by_source_events"
    assert event_rollforward.iloc[0]["unexplained_cohort_change_mil"] == pytest.approx(0.0)
    input_manifest = pd.read_csv(export_paths["replay_input_manifest"])
    assert set(input_manifest["source_key"]) >= {
        "quarterly_cash",
        "sector_positions",
        "mspd_cohorts",
        "ffiec_interest_constraints",
        "ncua_interest_constraints",
        "tier2_interest_constraints",
        "tdc_tdc_empirical_anchor",
        "tdc_treasury_interest_expense",
    }
    required_inputs = input_manifest[input_manifest["required"].astype(bool)]
    assert set(required_inputs["status"]) == {"present"}
    anchor = input_manifest[input_manifest["source_key"].eq("tdc_tdc_empirical_anchor")].iloc[0]
    assert str(anchor["consumed_in_run"]).lower() == "true"
    mmf_reference = input_manifest[input_manifest["source_key"].eq("tdc_tdc_mmf_rrp_quarterly_adjustments")].iloc[0]
    assert str(mmf_reference["required_for_claim"]).lower() == "true"
    assert str(mmf_reference["consumed_in_run"]).lower() == "false"
    method_meta = input_manifest[input_manifest["source_key"].eq("tdc_method_meta")].iloc[0]
    assert str(method_meta["required_for_claim"]).lower() == "true"
    assert str(method_meta["consumed_in_run"]).lower() == "false"


def test_historical_replay_excludes_z1_aggregate_controls_from_solver_inputs(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    mmf_component_path = tmp_path / "quarterly_inputs.csv"
    output_dir = tmp_path / "exports"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 0.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 80.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "banks", "broad_holder_class": "banks", "measure": "level", "value": 30.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "money_market_funds", "broad_holder_class": "money_market_cash", "measure": "level", "value": 70.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "total_treasury_bills", "broad_holder_class": "total", "measure": "level", "value": 100.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "all_sector_treasury_assets", "broad_holder_class": "total", "measure": "level", "value": 100.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "912345AA1",
                "cusip": "912345AA1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cohorts_path, index=False)
    _write_mmf_component_source(mmf_component_path, total=70.0, bills=70.0)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "output_dir": str(output_dir),
            "mmf_component_constraints": str(mmf_component_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="aggregate_scope")

    assert results.iloc[0]["Z1_Holder_Total"] == pytest.approx(100.0)
    assert results.iloc[0]["HolderScaleFactor"] == pytest.approx(1.0)
    diagnostics = results.attrs["portfolio_constraint_diagnostics"]
    assert not diagnostics["native_sector"].astype(str).str.startswith(("total_", "all_sector_")).any()
    scope = results.attrs["z1_scope_reconciliation"].iloc[0]
    assert scope["included_granular_z1_total"] == pytest.approx(100.0)
    assert scope["excluded_aggregate_z1_total"] == pytest.approx(200.0)
    assert scope["aggregate_rows_entered_solver"] == 0
    assert "z1_scope_reconciliation" in results.attrs["run_metadata"]["historical_replay_output_paths"]
    assert portfolio["FaceValue"].sum() == pytest.approx(100.0)


def test_historical_replay_enforces_mmf_bill_component_and_maturity_eligibility(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    mmf_component_path = tmp_path / "quarterly_inputs.csv"
    output_dir = tmp_path / "exports"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 0.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 80.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "banks", "broad_holder_class": "banks", "measure": "level", "value": 30.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "money_market_funds", "broad_holder_class": "money_market_cash", "measure": "level", "value": 70.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "BILL",
                "cusip": "912345BIL",
                "security_type": "bill",
                "issue_date": "2025-01-01",
                "maturity_date": "2025-06-30",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "FRN",
                "cusip": "912345FRN",
                "security_type": "frn",
                "issue_date": "2024-01-01",
                "maturity_date": "2027-01-31",
                "outstanding": 20.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "LONGNOTE",
                "cusip": "912345LNG",
                "security_type": "note",
                "issue_date": "2020-01-01",
                "maturity_date": "2035-01-31",
                "outstanding": 30.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cohorts_path, index=False)
    _write_mmf_component_source(mmf_component_path, total=70.0, bills=50.0)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "output_dir": str(output_dir),
            "mmf_component_constraints": str(mmf_component_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="mmf_components")

    reconciliation = results.attrs["mmf_component_reconciliation"]
    assert reconciliation.iloc[0]["post_solve_status"] == "matched"
    assert reconciliation.iloc[0]["modeled_bill_component_mil"] == pytest.approx(50.0)
    assert reconciliation.iloc[0]["modeled_nonbill_component_mil"] == pytest.approx(20.0)
    assert reconciliation.iloc[0]["fixed_rate_gt_397d_mil"] == pytest.approx(0.0)
    mmf = portfolio[portfolio["native_sector"].astype(str).eq("money_market_funds")]
    assert mmf.groupby("cohort_id")["FaceValue"].sum().to_dict() == pytest.approx({"BILL": 50.0, "FRN": 20.0})
    assert set(pd.read_csv(output_dir / "mmf_component_reconciliation.csv")["post_solve_status"]) == {"matched"}


def test_historical_replay_protects_mmf_direct_levels_from_negative_netting(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    mmf_component_path = tmp_path / "quarterly_inputs.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 100.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 0.0,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 80.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "banks", "broad_holder_class": "banks", "measure": "level", "value": 20.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "money_market_funds", "broad_holder_class": "money_market_cash", "measure": "level", "value": 70.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "otherfinancial", "measure": "level", "value": 50.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "individuals", "measure": "level", "value": -20.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "BILL",
                "cusip": "912345BIL",
                "security_type": "bill",
                "issue_date": "2025-01-01",
                "maturity_date": "2025-06-30",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "FRN",
                "cusip": "912345FRN",
                "security_type": "frn",
                "issue_date": "2024-01-01",
                "maturity_date": "2027-01-31",
                "outstanding": 20.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "NOTE",
                "cusip": "912345NOT",
                "security_type": "note",
                "issue_date": "2020-01-01",
                "maturity_date": "2035-01-31",
                "outstanding": 50.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(cohorts_path, index=False)
    _write_mmf_component_source(mmf_component_path, total=70.0, bills=50.0)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "mmf_component_constraints": str(mmf_component_path),
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="mmf_negative_netting")

    reconciliation = results.attrs["mmf_component_reconciliation"].iloc[0]
    assert reconciliation["modeled_total_mmf_treasury_mil"] == pytest.approx(70.0)
    assert reconciliation["modeled_bill_component_mil"] == pytest.approx(50.0)
    assert reconciliation["modeled_nonbill_component_mil"] == pytest.approx(20.0)
    assert reconciliation["direct_total_gap_mil"] == pytest.approx(0.0, abs=1.0e-8)
    assert reconciliation["direct_bill_component_gap_mil"] == pytest.approx(0.0, abs=1.0e-8)
    assert reconciliation["direct_nonbill_component_gap_mil"] == pytest.approx(0.0, abs=1.0e-8)
    mmf = portfolio[portfolio["native_sector"].astype(str).eq("money_market_funds")]
    assert mmf.groupby("cohort_id")["FaceValue"].sum().to_dict() == pytest.approx({"BILL": 50.0, "FRN": 20.0})
    assert portfolio.groupby("native_sector")["FaceValue"].sum().to_dict() == pytest.approx(
        {
            "banks": 20.0,
            "money_market_funds": 70.0,
            "otherfinancial": 30.0,
        }
    )
    bridge = results.attrs["negative_sector_netting_bridge"]
    assert "money_market_funds" not in set(bridge["native_sector"].astype(str))
    other = bridge[bridge["native_sector"].astype(str).eq("otherfinancial")].iloc[0]
    assert other["raw_sector_level_mil"] == pytest.approx(50.0)
    assert other["adjusted_sector_level_mil"] == pytest.approx(30.0)


def test_historical_replay_nets_negative_subcategory_inside_broad_holder_bucket(tmp_path):
    cash_path = tmp_path / "cash.csv"
    sectors_path = tmp_path / "sectors.csv"
    cohorts_path = tmp_path / "cohorts.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "selected_operating_cash_level_mil": 125.0,
                "selected_operating_cash_level_source": "fixture",
                "selected_operating_cash_tx_mil": 12.5,
                "selected_operating_cash_tx_source": "fixture",
                "dts_tga_mil": 100.0,
                "dts_ttl_mil": 25.0,
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cash_path, index=False)
    pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "measure": "level", "value": 20.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "otherfinancial", "measure": "level", "value": 100.0, "source_status": "fixture", "evidence_label": "observed"},
            {"quarter": "2025Q1", "sector": "individuals", "measure": "level", "value": -20.0, "source_status": "fixture", "evidence_label": "observed"},
        ]
    ).to_csv(sectors_path, index=False)
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "CUSIP1",
                "security_type": "bill",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
                "source_status": "fixture",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(cohorts_path, index=False)
    params = {
        "simulation_period": {"mode": "historical_replay"},
        "historical_replay": {
            "amount_unit_scale": 1.0,
            "paths": {
                "cash": str(cash_path),
                "sector_positions": str(sectors_path),
                "cohorts": str(cohorts_path),
            },
        },
    }

    results, portfolio = run_simulation(params, "2025-01-01", "2025-03-31", scenario_name="negative_subcategory_replay")

    assert results.iloc[0]["DebtHeld_Banks"] == pytest.approx(20.0)
    assert results.iloc[0]["DebtHeld_Private"] == pytest.approx(80.0)
    assert results.iloc[0]["TotalDebt_Agg"] == pytest.approx(100.0)
    assert results.iloc[0]["MSPD_Z1_SourceBasisDiff"] == pytest.approx(0.0)
    assert portfolio.groupby("HolderType")["FaceValue"].sum().to_dict() == pytest.approx(
        {"Banks": 20.0, "Private": 80.0}
    )
    assert set(portfolio["native_sector"]) == {"Banks", "otherfinancial"}
    diagnostics = results.attrs["portfolio_constraint_diagnostics"]
    netting = diagnostics[diagnostics["status"] == "negative_native_sector_netting"]
    assert not netting.empty


def test_historical_replay_validation_artifacts_pin_acceptance_invariants():
    validation_dir = Path(__file__).resolve().parents[1] / "data" / "historical_replay" / "validation"
    required = [
        "artifact_integrity.csv",
        "tdcest_selected_ladder_crosscheck_summary.csv",
        "tdcest_modern_formula_summary.csv",
        "historical_replay_event_ledger.csv",
        "historical_replay_unexplained_change_ledger.csv",
        "historical_replay_portfolio_snapshots.csv",
        "interest_proxy_alignment.csv",
        "maturity_prior_reconciliation.csv",
        "historical_replay_source_scope_audit.csv",
        "historical_replay_plausibility_audit.csv",
        "historical_replay_closeout_audit.md",
        "portfolio_native_sector_similarity.csv",
        "valuation_basis_feasibility_certificate.csv",
        "z1_transaction_flow_diagnostics.csv",
        "portfolio_transition_diagnostics.csv",
        "historical_replay_large_artifact_sample_manifest.csv",
        "historical_replay_portfolio_snapshots_sample.csv",
        "interest_component_detail_sample.csv",
        "historical_replay_runtime_manifest.csv",
        "historical_replay_runtime_lock.csv",
        "historical_replay_code_identity_manifest.csv",
    ]
    for filename in required:
        assert (validation_dir / filename).exists(), filename
    assert not (validation_dir / "interest_constraint_feasibility.csv").exists()

    artifact_integrity = pd.read_csv(validation_dir / "artifact_integrity.csv")
    assert int((artifact_integrity["status"] == "present").sum()) >= len(required)
    for column in ["sha256", "row_count", "column_count", "header_sha256", "run_id", "config_sha256", "code_sha256"]:
        assert column in artifact_integrity.columns
    assert "interest_constraint_feasibility" not in set(artifact_integrity["artifact"].astype(str))
    assert "valuation_basis_feasibility_certificate" in set(artifact_integrity["artifact"].astype(str))
    assert "runtime_lock" in set(artifact_integrity["artifact"].astype(str))
    assert "code_identity_manifest" in set(artifact_integrity["artifact"].astype(str))
    acceptance = (validation_dir / "historical_replay_acceptance.md").read_text(encoding="utf-8")
    assert "This artifact is generated from live validation CSVs" in acceptance
    assert "Solver methods: `exact_feasible_highs_projection: 1, exact_weighted_entropy_projection: 99`" in acceptance
    assert "Aggregate-only certificate quarters: `0` (``)" in acceptance
    assert "Portfolio transition diagnostics:" in acceptance
    assert "interval_quarters values `[1]`" in acceptance

    plausibility = pd.read_csv(validation_dir / "historical_replay_plausibility_audit.csv")
    assert set(plausibility["status"]) == {"pass"}
    assert set(plausibility["check"]) >= {
        "selected_tdc_target_wiring",
        "event_rollforward_identity",
        "interest_projection_feasibility",
        "mmf_bill_component_constraint",
        "native_sector_nonidentification_boundary",
        "pricing_scope_boundary",
        "weighted_sector_constraint_fit",
        "post_export_constraint_fit",
        "exact_observation_boundary",
        "z1_transaction_flow_diagnostic",
        "portfolio_transition_explanation",
        "large_artifact_sample_manifest",
    }

    source_scope = pd.read_csv(validation_dir / "historical_replay_source_scope_audit.csv")
    assert source_scope["closeout_status"].notna().all()
    assert "accepted" in set(source_scope["closeout_status"])

    selected = pd.read_csv(validation_dir / "tdcest_selected_ladder_crosscheck_summary.csv").iloc[0]
    assert int(selected["compared_rows"]) == 96
    assert int(selected["matched_rows"]) == 96
    assert int(selected["mismatch_rows"]) == 0

    modern = pd.read_csv(validation_dir / "tdcest_modern_formula_summary.csv").iloc[0]
    assert int(modern["compared_rows"]) == 16
    assert int(modern["canonical_formula_mismatch_rows"]) == 0

    z1_flow = pd.read_csv(validation_dir / "z1_transaction_flow_diagnostics.csv")
    assert len(z1_flow.index) == 2800
    assert int(z1_flow["z1_transaction_flow_mil"].notna().sum()) == 2400
    assert "z1_transaction_flow_saar_mil" in z1_flow.columns
    assert (
        (
            pd.to_numeric(z1_flow["z1_transaction_flow_mil"], errors="coerce") * 4.0
            - pd.to_numeric(z1_flow["z1_transaction_flow_saar_mil"], errors="coerce")
        )
        .dropna()
        .abs()
        .max()
        <= 1.0e-6
    )
    assert set(z1_flow["claim_boundary"]) == {
        "z1_transaction_flow_is_aggregate_transition_diagnostic_not_exact_transfer"
    }

    diagnostics = pd.read_csv(validation_dir / "historical_replay_diagnostics.csv")
    solver = diagnostics[diagnostics["diagnostic_type"].eq("solver")]
    aggregate_only_quarters = set(
        solver.loc[
            solver["solver_method"].eq("aggregate_only_minimum_weighted_slack_certificate"),
            "quarter",
        ].astype(str)
    )
    portfolio = pd.read_csv(validation_dir / "historical_replay_portfolio_snapshots.csv")
    assert int(portfolio["quarter"].astype(str).isin(aggregate_only_quarters).sum()) == 0
    results = pd.read_csv(validation_dir / "historical_replay_results.csv")
    result_quarters = pd.to_datetime(results["date"], errors="raise").dt.to_period("Q").astype(str)
    aggregate_results = results[result_quarters.isin(aggregate_only_quarters)]
    debt_held_columns = [col for col in results.columns if col.startswith("DebtHeld_")]
    if aggregate_only_quarters:
        assert not aggregate_results.empty
        assert aggregate_results[debt_held_columns].isna().all().all()
        assert set(aggregate_results["HolderSurfaceStatus"]) == {"aggregate_only_no_cell_portfolio"}
        assert set(aggregate_results["DebtSurfaceRole"]) == {"aggregate_certificate_no_cell_portfolio"}
        assert pd.to_numeric(aggregate_results["Replay_SectorResidual"], errors="coerce").isna().all()
    ledger = pd.read_csv(validation_dir / "historical_replay_ledger.csv")
    aggregate_ledger = ledger[ledger["quarter"].astype(str).isin(aggregate_only_quarters)]
    if aggregate_only_quarters:
        assert set(aggregate_ledger["holder_surface_status"]) == {"aggregate_only_no_cell_portfolio"}
        assert set(aggregate_ledger["debt_surface_role"]) == {"aggregate_certificate_no_cell_portfolio"}

    transitions = pd.read_csv(validation_dir / "portfolio_transition_diagnostics.csv")
    high_turnover = transitions[transitions["high_tv_transition"].fillna(False).astype(bool)]
    assert {
        "interval_quarters",
        "z1_transaction_flow_saar_mil",
        "z1_flow_context_status",
        "event_context_status",
        "event_ledger_gross_activity_mil",
        "modeled_minus_z1_flow_mil",
        "modeled_minus_z1_flow_share",
        "event_source_issue_mil",
        "event_source_redemption_mil",
        "event_source_indexation_mil",
        "event_source_reclassification_mil",
        "event_unexplained_cohort_change_mil",
        "event_unexplained_residual_change_mil",
        "event_component_reconciliation_gap_mil",
        "event_component_scope",
    }.issubset(transitions.columns)
    assert pd.to_numeric(transitions["interval_quarters"], errors="coerce").eq(1).all()
    if not high_turnover.empty:
        assert high_turnover["event_component_reconciliation_gap_mil"].abs().max() <= 1.0e-6
        assert set(high_turnover["event_component_scope"]) == {
            "treasury_wide_aggregate_context_not_sector_allocated"
        }
        assert not high_turnover["transition_explanation_status"].eq(
            "high_turnover_missing_transition_context"
        ).any()
        assert not high_turnover["transition_explanation_status"].isin(
            {
                "high_turnover_with_z1_flow_and_cohort_churn_context",
                "high_turnover_with_z1_flow_context",
                "high_turnover_with_cohort_churn_context",
            }
        ).any()

    sample_manifest = pd.read_csv(validation_dir / "historical_replay_large_artifact_sample_manifest.csv")
    assert set(sample_manifest["source_artifact"]) == {
        "historical_replay_portfolio_snapshots.csv",
        "interest_component_detail.csv",
    }
    assert sample_manifest["sample_row_count"].min() > 0
    assert set(sample_manifest["sample_status"]) == {"sample_nonempty_all_declared_strata_selected"}
    assert sample_manifest["strata_with_zero_selected_rows"].fillna("").eq("").all()

    event_ledger = pd.read_csv(
        validation_dir / "historical_replay_event_ledger.csv",
        usecols=[
            "event_type",
            "source_row_key",
            "quarter",
            "cusip",
            "cohort_id",
            "current_principal_delta_mil",
            "derivation",
        ],
    )
    terminal = event_ledger[
        event_ledger["source_row_key"].astype(str).str.endswith("|terminal_exit", na=False)
    ]
    terminal_counts = terminal["event_type"].value_counts().to_dict()
    assert terminal_counts == {
        "maturity_redemption": 5303,
        "called_redemption": 11,
        "source_discontinuity_exit": 9,
    }

    reclass = event_ledger[event_ledger["event_type"] == "mspd_source_reclassification"].copy()
    reclass["current_principal_delta_mil"] = pd.to_numeric(
        reclass["current_principal_delta_mil"],
        errors="coerce",
    ).fillna(0.0)
    reclass["maturity_date"] = reclass["cohort_id"].astype(str).str.split("|").str[2]
    reclass_groups = (
        reclass.groupby(["quarter", "cusip", "maturity_date"], sort=False)
        .agg(
            net_current_principal_delta_mil=("current_principal_delta_mil", "sum"),
            derivations=("derivation", lambda values: set(values.astype(str))),
        )
        .reset_index()
    )
    nonzero_reclass = reclass_groups[reclass_groups["net_current_principal_delta_mil"].abs() > 1.0e-6]
    assert len(nonzero_reclass.index) == 37
    assert set(nonzero_reclass["quarter"]) == {"2013Q4"}
    assert all(
        derivations == {"old_reopening_initial_split_residual_stock_shift"}
        for derivations in nonzero_reclass["derivations"]
    )

    unexplained = pd.read_csv(
        validation_dir / "historical_replay_unexplained_change_ledger.csv",
        usecols=["quarter", "evidence_status", "derivation"],
    )
    old_split_unexplained = unexplained[
        unexplained["evidence_status"].eq("unexplained_old_reopening_initial_split_residual")
    ]
    assert not old_split_unexplained.empty
    assert set(old_split_unexplained["quarter"]) == {"2013Q4"}
    assert set(old_split_unexplained["derivation"]) == {
        "old_reopening_initial_split_counterpart_left_unexplained"
    }

    interest = pd.read_csv(
        validation_dir / "interest_proxy_alignment.csv",
        usecols=["native_sector", "tdcest_point", "within_feasible_bounds"],
    )
    referenced = interest[pd.to_numeric(interest["tdcest_point"], errors="coerce").notna()].copy()
    infeasible = referenced[~referenced["within_feasible_bounds"].fillna(False).astype(bool)]
    assert len(infeasible.index) == 29
    assert set(infeasible["native_sector"]) == {"Banks", "Foreign"}

    portfolio = pd.read_csv(
        validation_dir / "historical_replay_portfolio_snapshots.csv",
        usecols=["source_sector", "HolderType"],
    )
    residual_rows = portfolio[portfolio["source_sector"].astype(str).eq("MSPD_Z1_SourceBasisResidual")]
    assert not residual_rows.empty
    assert not residual_rows["HolderType"].astype(str).eq("Private").any()
