"""RateWall contract export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ratewall_input_builder as rib
from ratewall_contract import export_ratewall_bundle
from ratewall_input_builder import (
    SCENARIO_CURVE_MAP,
    _cbo_budget_values,
    build_primary_flow_path,
    build_holder_absorption_path,
    build_yield_curve_surface,
)
from sim_groups import resolve_worker_count
from simulation_core import run_simulation
from yield_curve_path import curve_for_date, load_yield_curve_surface

from test_engine_and_validation import make_bond_row, minimal_params


def _run_contract_case():
    params = minimal_params()
    params["initial_values"]["tga"] = 200.0
    params["tga_params"]["target_balance"] = 200.0
    params["treasury_issuance_profile"]["bills"]["maturities"] = [0.25]
    params["treasury_issuance_profile"]["bills"]["maturity_distribution"] = [1.0]
    bill = make_bond_row(
        BondID=401,
        SecurityType="Fixed",
        HolderType="Private",
        FaceValue=100.0,
        CouponRate=0.0,
        MaturityDate=pd.Timestamp("2025-01-10"),
        OriginalMaturityYears=0.25,
        MaturityCategory="bills",
        IssueProceeds=98.0,
        IssueYieldAtIssue=0.08,
    )
    frn = make_bond_row(
        BondID=402,
        SecurityType="FRN",
        HolderType="Private",
        FaceValue=50.0,
        CouponRate=0.0,
        MaturityDate=pd.Timestamp("2027-01-01"),
        OriginalMaturityYears=2.0,
        MaturityCategory=None,
        FixedSpread=0.001,
        LastAccrualDate=pd.Timestamp("2025-01-01"),
    )
    params["initial_bonds_df"] = pd.DataFrame([bill, frn]).astype("object")
    return run_simulation(params, "2025-01-01", "2025-04-15", freq="W", scenario_name="ratewall_contract_case")


def test_ratewall_bundle_component_identity_and_overlap(tmp_path):
    results, _ = _run_contract_case()
    paths = export_ratewall_bundle({"contract_case": results}, tmp_path)

    summary = pd.read_csv(paths["summary"])
    components = pd.read_csv(paths["components"])
    assert not summary.empty
    assert not components.empty
    for _, row in summary.iterrows():
        expected = (
            row["tdc_fiscal_flow_bil"]
            + row["tdc_debt_service_principal_to_du_bil"]
            + row["tdc_debt_service_interest_to_du_bil"]
            + row["tdc_auction_absorption_du_bil"]
            + row["tdc_secondary_trades_bil"]
            + row["tdc_other_bil"]
        )
        assert abs(expected - row["tdc_change_bil"]) < 1e-7
        assert abs(
            row["tdc_change_bil"]
            - row["overlap_cashflow_bil"]
            - row["tdc_change_ex_overlap_bil"]
        ) < 1e-7
    assert not (
        components["enters_direct_interest_support"]
        & components["enters_tdc_deposit_support_default"]
    ).any()


def test_ratewall_contract_splits_bill_discount_from_principal(tmp_path):
    results, _ = _run_contract_case()
    paths = export_ratewall_bundle({"contract_case": results}, tmp_path)
    summary = pd.read_csv(paths["summary"])

    assert summary["bill_discount_interest_to_du_bil"].sum() > 0
    assert summary["tdc_debt_service_principal_to_du_bil"].sum() > 0
    first_redemption = summary[summary["bill_discount_interest_to_du_bil"] >= 2.0].iloc[0]
    assert abs(
        first_redemption["tdc_debt_service_principal_to_du_bil"]
        + first_redemption["bill_discount_interest_to_du_bil"]
        - 100.0
    ) < 1e-7


def test_ratewall_contract_tips_inflation_compensation_is_memo_default(tmp_path):
    params = minimal_params()
    params["initial_values"]["tga"] = 200.0
    params["tga_params"]["target_balance"] = 200.0
    params["tips_params"] = {
        "cpi_start_level": 115.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 3,
        "default_real_coupon_rate": 0.0,
    }
    params["initial_bonds_df"] = pd.DataFrame(
        [
            make_bond_row(
                BondID=701,
                SecurityType="TIPS",
                HolderType="Private",
                FaceValue=100.0,
                CouponRate=0.0,
                MaturityDate=pd.Timestamp("2025-01-10"),
                OriginalMaturityYears=5.0,
                MaturityCategory="tips",
                OriginalPrincipal=100.0,
                AdjustedPrincipal=115.0,
                ReferenceCPI_Issue=100.0,
                IndexRatio=1.15,
            )
        ]
    ).astype("object")
    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="tips_maturity_contract_case",
    )

    paths = export_ratewall_bundle({"tips_case": results}, tmp_path)
    summary = pd.read_csv(paths["summary"])
    components = pd.read_csv(paths["components"])
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    tips_component = components[
        components["component_key"] == "tips_inflation_compensation_to_du"
    ]
    assert not tips_component.empty
    assert tips_component["amount_bil"].astype(float).sum() > 0.0
    assert not tips_component["enters_tdc_deposit_support_default"].astype(str).str.lower().eq("true").any()
    default_components = components[
        components["enters_tdc_deposit_support_default"].astype(str).str.lower().eq("true")
    ]
    for _, row in summary.iterrows():
        component_sum = default_components[
            (default_components["scenario_id"] == row["scenario_id"])
            & (default_components["quarter"] == row["quarter"])
        ]["amount_bil"].astype(float).sum()
        assert abs(
            component_sum
            - (row["tdc_change_bil"] - row["overlap_cashflow_bil"])
        ) < 1e-7
    assert manifest["validation"]["validation_status"] == "pass"


def test_ratewall_contract_exports_remittance_as_non_tdc_component(tmp_path):
    params = minimal_params()
    params["other_flows"]["cb_net_expense"] = 0.0
    params["initial_bonds_df"] = pd.DataFrame(
        [
            make_bond_row(
                BondID=501,
                SecurityType="Fixed",
                HolderType="CB",
                FaceValue=100.0,
                CouponRate=0.08,
                MaturityDate=pd.Timestamp("2030-01-01"),
                OriginalMaturityYears=5.0,
                MaturityCategory="notes",
            )
        ]
    ).astype("object")
    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-07-15",
        freq="W",
        scenario_name="ratewall_remittance_contract_case",
    )

    paths = export_ratewall_bundle({"remittance_case": results}, tmp_path)
    summary = pd.read_csv(paths["summary"])
    components = pd.read_csv(paths["components"])

    assert summary["cb_remittance_to_tga_bil"].sum() > 0
    remittance = components[
        components["component_key"] == "central_bank_remittance_to_tga"
    ]
    assert not remittance.empty
    assert set(remittance["cash_component_key"]) == {"central_bank_remittance_to_tga"}
    assert not remittance["enters_direct_interest_support"].any()
    assert not remittance["enters_tdc_deposit_support_default"].any()


def test_dynamic_yield_curve_surface_applies_by_period(tmp_path):
    curve_path = tmp_path / "curve.csv"
    pd.DataFrame(
        [
            {"scenario_id": "baseline", "curve_date": "2025-01-01", "tenor_years": 1, "nominal_rate": 0.03},
            {"scenario_id": "baseline", "curve_date": "2025-01-01", "tenor_years": 10, "nominal_rate": 0.04},
            {"scenario_id": "baseline", "curve_date": "2025-04-01", "tenor_years": 1, "nominal_rate": 0.05},
            {"scenario_id": "baseline", "curve_date": "2025-04-01", "tenor_years": 10, "nominal_rate": 0.06},
        ]
    ).to_csv(curve_path, index=False)

    surface = load_yield_curve_surface(curve_path)
    years, rates, status = curve_for_date(surface, "2025-05-01", scenario_id="baseline")
    assert years == [1.0, 10.0]
    assert rates == [0.05, 0.06]
    assert status == "dynamic_curve_surface:2025-04-01"


def test_dynamic_yield_curve_surface_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="yield curve surface file is missing"):
        load_yield_curve_surface(tmp_path / "missing_curve.csv")


def test_ratewall_configured_missing_input_paths_raise(tmp_path):
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "primary_flow_to_du_file": str(tmp_path / "missing_primary.csv"),
    }

    with pytest.raises(FileNotFoundError, match="Primary flow path file is missing"):
        run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="missing_primary_case",
        )

    params = minimal_params()
    params["ratewall_input_paths"] = {
        "holder_absorption_path_file": str(tmp_path / "missing_holder.csv"),
    }
    with pytest.raises(FileNotFoundError, match="Holder absorption path file is missing"):
        run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="missing_holder_case",
        )


def test_ratewall_primary_flow_fallback_warns_once(tmp_path, capsys):
    flow_path = tmp_path / "primary_flow.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": "default",
                "quarter": "2025Q1",
                "primary_fiscal_flow_to_du_bil": 30.0,
            }
        ]
    ).to_csv(flow_path, index=False)
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "primary_flow_to_du_file": str(flow_path),
    }

    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="unlisted_scenario",
    )

    captured = capsys.readouterr()
    assert "primary_flow path used fallback scenario 'default'" in captured.out
    assert captured.out.count("primary_flow path used fallback scenario") == 1
    assert results.attrs["run_metadata"]["ratewall_primary_flow_used_periods"] > 0


def test_ratewall_contract_status_reflects_empty_loaded_primary_flow(tmp_path):
    flow_path = tmp_path / "empty_primary_flow.csv"
    pd.DataFrame(columns=["scenario_id", "quarter", "primary_fiscal_flow_to_du_bil"]).to_csv(
        flow_path,
        index=False,
    )
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "primary_flow_to_du_file": str(flow_path),
    }
    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="empty_flow_case",
    )

    paths = export_ratewall_bundle(
        {"empty_flow_case": results},
        tmp_path / "bundle",
        config={"ratewall_input_paths": {"primary_flow_to_du_file": str(flow_path)}},
    )
    summary = pd.read_csv(paths["summary"])

    assert results.attrs["run_metadata"]["ratewall_primary_flow_loaded_rows"] == 0
    assert set(summary["primary_flow_status"]) == {"simulation_fiscal_flow_to_du_proxy"}


def test_ratewall_source_registry_blocks_weak_wamest_rows(tmp_path):
    results, _ = run_simulation(
        minimal_params(),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="source_registry_case",
    )
    paths = export_ratewall_bundle({"source_registry_case": results}, tmp_path)
    registry = pd.read_csv(paths["source_registry"])

    weak_wamest = registry[registry["source_key"] == "weak_revaluation_wam_rows"].iloc[0]
    assert not weak_wamest["central_default_eligible"]
    assert weak_wamest["sensitivity_only"]
    mmf_disclosure = registry[
        registry["source_key"] == "mmf_collapsed_into_du_current_private_bucket"
    ].iloc[0]
    assert mmf_disclosure["central_default_eligible"]
    assert "full_private_mmf_route_split_owner_gated" in mmf_disclosure["binding_blocker"]


def test_ratewall_source_registry_exports_domestic_nonbank_route_contract(tmp_path):
    results, _ = run_simulation(
        minimal_params(),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="source_registry_case",
    )
    paths = export_ratewall_bundle({"source_registry_case": results}, tmp_path)
    registry = pd.read_csv(paths["source_registry"])

    route_rows = registry[registry["source_family"] == "holder_route_contract"]
    assert set(route_rows["source_key"]) >= {
        "Private",
        "domestic_nonbank_non_deposit_funded",
        "mmf_on_rrp_reserve_user_like",
    }
    private_route = route_rows[route_rows["source_key"] == "Private"].iloc[0]
    assert private_route["ratewall_role"] == (
        "domestic_nonbank_undifferentiated_current_contract"
    )
    assert private_route["central_default_eligible"]
    non_deposit_route = route_rows[
        route_rows["source_key"] == "domestic_nonbank_non_deposit_funded"
    ].iloc[0]
    assert not non_deposit_route["central_default_eligible"]
    assert "split_from_current_private_holder_bucket" in non_deposit_route[
        "binding_blocker"
    ]
    mmf_route = route_rows[
        route_rows["source_key"] == "mmf_on_rrp_reserve_user_like"
    ].iloc[0]
    assert not mmf_route["central_default_eligible"]
    assert "mmf_on_rrp_route_split" in mmf_route["binding_blocker"]


def test_ratewall_contract_exports_private_route_sensitivity_sidecar(tmp_path):
    results, _ = run_simulation(
        minimal_params(),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="source_registry_case",
    )
    sensitivity_path = tmp_path / "tdc_private_route_sensitivity.csv"
    pd.DataFrame(
        [
            {
                "contract_version": "tdc_tdcsim_private_route_allocation_sensitivity_v1",
                "ref_quarter": "2025Q4",
                "object_family": "stock_interest_quarter_end",
                "route_class": "deposit_funded_domestic_nonbank_possible",
                "share_lambda_0": 0.0,
                "share_lambda_0_5": 0.1,
                "share_lambda_1": 0.2,
                "evidence_tier": "bounded_proxy",
                "measurement_stage": "holder_stock",
                "mapping_burden": "requires_unobserved_actor_split",
                "assumption_status": "bounded_assumption",
                "current_demand_eligible": "false",
                "canonical_tdc_math_change": "false",
                "source_backed_private_bucket_split_status": (
                    "not_source_backed_private_bucket_split"
                ),
                "evidence_mode_enabled": "false",
                "holder_allocation_enabled": "false",
            }
        ]
    ).to_csv(sensitivity_path, index=False)

    paths = export_ratewall_bundle(
        {"source_registry_case": results},
        tmp_path / "bundle",
        config={"private_route_sensitivity_file": str(sensitivity_path)},
    )
    sidecar = pd.read_csv(paths["private_route_sensitivity"])
    registry = pd.read_csv(paths["source_registry"])
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    assert sidecar["tdcsim_contract_key"].iloc[0] == (
        "tdcsim_private_route_sensitivity_contract_v1"
    )
    assert not sidecar["central_default_eligible"].iloc[0]
    assert sidecar["sensitivity_only"].iloc[0]
    assert sidecar["does_not_modify_default_holder_perimeter"].iloc[0]
    assert "private_route_sensitivity" in manifest["files"]
    registry_row = registry[
        registry["source_key"] == "tdc_tdcsim_private_route_allocation_sensitivity_v1"
    ].iloc[0]
    assert registry_row["sensitivity_only"]
    assert not registry_row["central_default_eligible"]
    assert "requires_source_backed_split" in registry_row["binding_blocker"]


def test_ratewall_contract_rejects_promoted_private_route_sensitivity(tmp_path):
    results, _ = run_simulation(
        minimal_params(),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="source_registry_case",
    )
    sensitivity_path = tmp_path / "tdc_private_route_sensitivity.csv"
    pd.DataFrame(
        [
            {
                "contract_version": "tdc_tdcsim_private_route_allocation_sensitivity_v1",
                "ref_quarter": "2025Q4",
                "object_family": "stock_interest_quarter_end",
                "route_class": "deposit_funded_domestic_nonbank_possible",
                "share_lambda_0": 0.0,
                "share_lambda_0_5": 0.1,
                "share_lambda_1": 0.2,
                "evidence_tier": "source_backed_measurement",
                "mapping_burden": "none",
                "assumption_status": "none_source_observed",
                "current_demand_eligible": "false",
                "canonical_tdc_math_change": "false",
                "source_backed_private_bucket_split_status": (
                    "not_source_backed_private_bucket_split"
                ),
                "evidence_mode_enabled": "false",
                "holder_allocation_enabled": "false",
            }
        ]
    ).to_csv(sensitivity_path, index=False)

    with pytest.raises(ValueError, match="evidence_tier must be bounded_proxy"):
        export_ratewall_bundle(
            {"source_registry_case": results},
            tmp_path / "bundle",
            config={"private_route_sensitivity_file": str(sensitivity_path)},
        )


def test_ratewall_contract_primary_flow_status_uses_input_manifest(tmp_path):
    results, _ = _run_contract_case()
    manifest_path = tmp_path / "input_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": {"tdcsim_primary_flow_to_du_path.csv": {"sha256": "abc"}},
                "source_hierarchy": {
                    "primary_flow": "CBO total deficit less net interest as aggregate cash proxy",
                },
            }
        ),
        encoding="utf-8",
    )

    paths = export_ratewall_bundle(
        {"contract_case": results},
        tmp_path / "bundle",
        config={"input_manifest": str(manifest_path)},
    )
    summary = pd.read_csv(paths["summary"])
    registry = pd.read_csv(paths["source_registry"])
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))

    assert set(summary["primary_flow_status"]) == {"simulation_fiscal_flow_to_du_proxy"}
    assert "tdcsim_primary_flow_to_du_path.csv" in set(registry["source_key"])
    assert "tdcsim_primary_flow_to_du_path.csv" in manifest["input_artifacts"]


def test_ratewall_input_builder_parses_cbo_billions_section():
    ratewall_root = Path(__file__).resolve().parents[2] / "ratewall"
    cbo_path = ratewall_root / "data/raw/cbo/51118-2026-02-Budget-Projections.xlsx"
    if not cbo_path.exists():
        return

    values = _cbo_budget_values(cbo_path)
    assert values[2026]["net_interest_bil"] > 1000
    assert values[2026]["deficit_bil"] < -1000
    assert -values[2026]["deficit_bil"] - values[2026]["net_interest_bil"] > 800


def test_primary_flow_path_maps_cbo_fiscal_year_to_calendar_quarters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        rib,
        "_cbo_budget_values",
        lambda path: {2026: {"deficit_bil": -1200.0, "net_interest_bil": 200.0}},
    )

    frame, metadata = build_primary_flow_path(
        tmp_path / "primary_flow.csv",
        cbo_budget_path=tmp_path / "fake_cbo.xlsx",
        scenario_ids=["baseline"],
    )

    assert metadata["rows"] == 4
    assert frame["quarter"].tolist() == ["2025Q4", "2026Q1", "2026Q2", "2026Q3"]
    assert set(frame["primary_fiscal_flow_to_du_bil"].astype(float)) == {250.0}
    assert frame["source_status"].str.contains(
        "federal_fiscal_year_mapped_to_calendar_quarters"
    ).all()


def test_ratewall_source_backed_holder_path_is_instrument_specific(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    ratewall_root = repo_root.parent / "ratewall"
    tdcmix_prior = (
        repo_root.parent
        / "tdcmix"
        / "data"
        / "processed"
        / "holder_absorption_prior_contract.csv"
    )
    z1_absorption = (
        repo_root.parent
        / "tdcmix"
        / "data"
        / "processed"
        / "z1_exact_holder_absorption_panel.csv"
    )
    bills_primary = (
        repo_root.parent
        / "tsyparty"
        / "data"
        / "interim"
        / "bills_quarterly_composition.csv"
    )
    coupons_primary = (
        repo_root.parent
        / "tsyparty"
        / "data"
        / "interim"
        / "nominal_coupons_quarterly_composition.csv"
    )
    if not (ratewall_root.exists() and tdcmix_prior.exists() and z1_absorption.exists()):
        return

    frame, metadata = build_holder_absorption_path(
        tmp_path / "holder.csv",
        tdcmix_prior_path=tdcmix_prior,
        z1_absorption_path=z1_absorption,
        bills_primary_path=bills_primary,
        coupons_primary_path=coupons_primary,
    )

    assert set(frame["scenario_id"]) == set(SCENARIO_CURVE_MAP)
    assert "primary_market_instrument_buyer_composition" in metadata["baseline_sources"]
    assert metadata["z1_effective_quarters_n_eff"] != "0.000000000000"
    assert metadata["z1_shrinkage_weight"] != "0.000000000000"
    assert metadata["primary_market_overlay_coverage"] == (
        "low_coverage_only_dealers_fed_foreign_official_share_weighted"
    )
    assert metadata["scenario_shift_lambda_default"] == "0.50"
    assert metadata["tic_foreign_cross_check_status"].startswith("DEFERRED")
    current = frame[
        (frame["scenario_id"] == "current_mix_baseline")
        & (frame["quarter"] == "2026Q1")
    ]
    for column in ["bills_pct", "notes_pct", "bonds_pct", "tips_pct", "frn_pct"]:
        assert abs(current[column].astype(float).sum() - 1.0) < 1e-9
    banks = current[current["holder_type"] == "Banks"].iloc[0]
    assert banks["bills_pct"] != banks["notes_pct"]
    private = current[current["holder_type"] == "Private"].iloc[0]
    assert str(private["mmf_collapsed_into_du"]).lower() == "true"
    assert "mmf_collapsed_into_du_current_private_bucket" in private["source_status"]
    assert "dealer_bridge_redistributed_by_pre_overlay_baseline_not_mapped_to_banks" in private["source_status"]
    assert "primary_market_overlay_low_coverage_only_dealers_fed_foreign_official_share_weighted" in private["source_status"]


def test_ratewall_absorption_z1_shrinkage_drops_trailing_all_nan_quarter(tmp_path):
    rows = []
    for quarter in [
        "2024-03-31",
        "2024-06-30",
        "2024-09-30",
        "2024-12-31",
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
        "2025-12-31",
    ]:
        rows.append(
            {
                "quarter": quarter,
                "positive_absorption_total": 100.0,
                "pos_abs_share_fed": 0.2,
                "pos_abs_share_banks": 0.1,
                "pos_abs_share_dealer_bridge": 0.5,
                "pos_abs_share_row": 0.3,
                "pos_abs_share_mmf": 0.0,
                "pos_abs_share_domestic_nonbank": 0.4,
            }
        )
    rows.append(
        {
            "quarter": "2026-03-31",
            "positive_absorption_total": float("nan"),
            "pos_abs_share_fed": float("nan"),
            "pos_abs_share_banks": float("nan"),
            "pos_abs_share_dealer_bridge": float("nan"),
            "pos_abs_share_row": float("nan"),
            "pos_abs_share_mmf": float("nan"),
            "pos_abs_share_domestic_nonbank": float("nan"),
        }
    )
    path = tmp_path / "z1.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    shares, n_eff = rib._latest_z1_absorption_shares(path)

    assert n_eff == pytest.approx(8.0)
    assert shares == pytest.approx(
        {
            "CB": 0.2,
            "Banks": 0.1,
            "Foreign": 0.3,
            "Private": 0.4,
        }
    )


def test_ratewall_absorption_primary_dealers_are_redistributed_by_baseline(tmp_path):
    path = tmp_path / "primary.csv"
    pd.DataFrame(
        [
            {"date": "2026-03-31", "buyer_class": "dealers", "share": 0.8},
            {"date": "2026-03-31", "buyer_class": "fed", "share": 0.05},
            {"date": "2026-03-31", "buyer_class": "foreign_official", "share": 0.15},
        ]
    ).to_csv(path, index=False)

    shares = rib._latest_primary_market_instrument_shares(
        path,
        baseline_scalar={"Banks": 0.2, "CB": 0.1, "Foreign": 0.3, "Private": 0.4},
    )

    assert shares == pytest.approx(
        {
            "Banks": 0.16,
            "CB": 0.13,
            "Foreign": 0.39,
            "Private": 0.32,
        }
    )


def test_ratewall_absorption_scenario_lambda_comes_from_calibration(tmp_path):
    tdcmix_path = tmp_path / "tdcmix.csv"
    pd.DataFrame(
        [
            {"bucket": "fed", "default_prior_value": 0.25, "lower_prior_value": 0.8, "upper_prior_value": 0.1},
            {"bucket": "banks", "default_prior_value": 0.25, "lower_prior_value": 0.1, "upper_prior_value": 0.1},
            {"bucket": "row", "default_prior_value": 0.25, "lower_prior_value": 0.1, "upper_prior_value": 0.1},
            {
                "bucket": "domestic_nonbank",
                "default_prior_value": 0.25,
                "lower_prior_value": 0.0,
                "upper_prior_value": 0.7,
            },
            {"bucket": "mmf", "default_prior_value": 0.0, "lower_prior_value": 0.0, "upper_prior_value": 0.0},
            {"bucket": "dealer_bridge", "default_prior_value": 0.0, "lower_prior_value": 0.0, "upper_prior_value": 0.0},
        ]
    ).to_csv(tdcmix_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "holder_absorption_calibration:\n"
        "  scenario_shift_lambda_grid: [0.25, 0.50, 0.75, 1.00]\n"
        "  scenario_shift_lambda_default: 1.00\n",
        encoding="utf-8",
    )
    calibration = rib._load_holder_absorption_calibration(config_path)

    frame, metadata = build_holder_absorption_path(
        tmp_path / "holder.csv",
        tdcmix_prior_path=tdcmix_path,
        z1_absorption_path=tmp_path / "missing_z1.csv",
        bills_primary_path=tmp_path / "missing_bills.csv",
        coupons_primary_path=tmp_path / "missing_coupons.csv",
        holder_absorption_calibration=calibration,
    )

    current = frame[
        (frame["scenario_id"] == "domestic_nonbank_absorption_shift")
        & (frame["quarter"] == "2026Q1")
    ].set_index("holder_type")
    assert metadata["scenario_shift_lambda_default"] == "1.00"
    assert float(current.loc["CB", "bills_pct"]) == pytest.approx(0.1)
    assert float(current.loc["Banks", "bills_pct"]) == pytest.approx(0.1)
    assert float(current.loc["Foreign", "bills_pct"]) == pytest.approx(0.1)
    assert float(current.loc["Private", "bills_pct"]) == pytest.approx(0.7)


def test_ratewall_source_backed_combined_scenarios_reuse_holder_shift_targets(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    ratewall_root = repo_root.parent / "ratewall"
    tdcmix_prior = (
        ratewall_root
        / "data/raw/ratewall_sibling_calibration/tdcsim/tdcsim_ratewall_source_registry.csv"
    )
    z1_absorption = (
        ratewall_root.parent
        / "tdc_estimates"
        / "data"
        / "processed"
        / "z1_exact_holder_absorption_panel.csv"
    )
    bills_primary = (
        repo_root.parent
        / "tsyparty"
        / "data"
        / "interim"
        / "bills_quarterly_composition.csv"
    )
    coupons_primary = (
        repo_root.parent
        / "tsyparty"
        / "data"
        / "interim"
        / "nominal_coupons_quarterly_composition.csv"
    )
    if not (ratewall_root.exists() and tdcmix_prior.exists() and z1_absorption.exists()):
        return

    frame, _ = build_holder_absorption_path(
        tmp_path / "holder.csv",
        tdcmix_prior_path=tdcmix_prior,
        z1_absorption_path=z1_absorption,
        bills_primary_path=bills_primary,
        coupons_primary_path=coupons_primary,
    )

    assert (
        SCENARIO_CURVE_MAP["domestic_nonbank_absorption_shift_higher_for_longer"]
        == "higher_for_longer_sensitivity"
    )
    assert (
        SCENARIO_CURVE_MAP["domestic_nonbank_absorption_shift_rapid_easing"]
        == "rapid_easing_sensitivity"
    )
    assert (
        SCENARIO_CURVE_MAP["reserve_user_absorption_shift_higher_for_longer"]
        == "higher_for_longer_sensitivity"
    )
    assert (
        SCENARIO_CURVE_MAP["reserve_user_absorption_shift_rapid_easing"]
        == "rapid_easing_sensitivity"
    )

    share_columns = ["bills_pct", "notes_pct", "bonds_pct", "tips_pct", "frn_pct"]
    checks = [
        (
            "domestic_nonbank_absorption_shift_higher_for_longer",
            "domestic_nonbank_absorption_shift",
        ),
        (
            "domestic_nonbank_absorption_shift_rapid_easing",
            "domestic_nonbank_absorption_shift",
        ),
        (
            "reserve_user_absorption_shift_higher_for_longer",
            "reserve_user_absorption_shift",
        ),
        (
            "reserve_user_absorption_shift_rapid_easing",
            "reserve_user_absorption_shift",
        ),
    ]
    for combined_id, holder_shift_id in checks:
        combined = frame[frame["scenario_id"] == combined_id].sort_values(["quarter", "holder_type"])
        holder_shift = frame[frame["scenario_id"] == holder_shift_id].sort_values(["quarter", "holder_type"])
        assert combined[["quarter", "holder_type"] + share_columns].reset_index(drop=True).equals(
            holder_shift[["quarter", "holder_type"] + share_columns].reset_index(drop=True)
        )


def test_ratewall_initial_portfolio_converts_mspd_millions_to_billions(tmp_path, monkeypatch):
    def fake_fetch(endpoint, *, record_date, page_size=10000):
        if endpoint.endswith("mspd_table_1"):
            return [
                {
                    "security_type_desc": "Marketable",
                    "security_class_desc": "Bills",
                    "debt_held_public_mil_amt": "1000000",
                }
            ]
        return [
            {
                "security_class1_desc": "Bills Maturity Value",
                "security_class2_desc": "FAKEBILL",
                "outstanding_amt": "500000",
                "issue_date": "2026-01-01",
                "maturity_date": "2026-04-01",
                "interest_rate_pct": "0",
                "yield_pct": "4.0",
            }
        ]

    monkeypatch.setattr(rib, "_latest_record_date", lambda endpoint: "2026-04-30")
    monkeypatch.setattr(rib, "_fetch_fiscaldata_rows", fake_fetch)
    monkeypatch.setattr(rib, "_load_stock_holder_shares", lambda path: {"Private": 1.0})

    frame, _ = rib.build_initial_portfolio(
        tmp_path / "cohorts.csv",
        tsyparty_z1_path=tmp_path / "unused.csv",
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert abs(float(row["face_value_bil"]) - 1000.0) < 1e-9
    assert abs(float(row["FaceValue"]) - 1000.0) < 1e-9
    assert "converted_to_bil" in row["source_status"]


def test_worker_count_respects_config_env_and_upper_bound(monkeypatch):
    assert resolve_worker_count(configured=None, default=4, upper_bound=7) == 4
    assert resolve_worker_count(configured=12, default=4, upper_bound=7) == 7
    assert resolve_worker_count(configured="auto", default=4, upper_bound=7) == 4

    monkeypatch.setenv("TDCSIM_SCENARIO_WORKERS", "3")
    assert (
        resolve_worker_count(
            configured=6,
            env_var="TDCSIM_SCENARIO_WORKERS",
            default=4,
            upper_bound=7,
        )
        == 3
    )


def test_ratewall_source_backed_yield_surface_has_curve_scenarios(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    cbo_path = (
        repo_root.parent
        / "ratewall"
        / "data/raw/cbo/51135-2026-02-Economic-Projections.xlsx"
    )
    if not cbo_path.exists():
        return

    frame, _ = build_yield_curve_surface(tmp_path / "curve.csv", cbo_economic_path=cbo_path)

    assert set(frame["scenario_id"]) >= set(SCENARIO_CURVE_MAP.values())
    baseline = frame[frame["scenario_id"] == "cbo_shape_preserving_baseline"]
    higher = frame[frame["scenario_id"] == "higher_for_longer_sensitivity"]
    merged = baseline.merge(
        higher,
        on=["curve_date", "tenor_years"],
        suffixes=("_baseline", "_higher"),
    )
    assert (
        merged["nominal_rate_higher"].astype(float)
        > merged["nominal_rate_baseline"].astype(float)
    ).all()
