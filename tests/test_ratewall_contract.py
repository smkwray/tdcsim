"""RateWall contract export tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ratewall_input_builder as rib
from bill_quote_basis import CBO_3M_BILL_DAYS, discount_rate_to_investment_rate
from ratewall_contract import export_ratewall_bundle
from ratewall_input_builder import (
    RATEWALL_SCENARIO_REGISTRY,
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


def _write_holder_path(path: Path, *, scenario_id: str, quarter: str) -> None:
    pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "quarter": quarter,
                "holder_type": "Private",
                "bills_pct": 0.75,
                "notes_pct": 0.65,
                "bonds_pct": 0.55,
                "tips_pct": 0.45,
                "frn_pct": 0.35,
            },
            {
                "scenario_id": scenario_id,
                "quarter": quarter,
                "holder_type": "Banks",
                "bills_pct": 0.25,
                "notes_pct": 0.35,
                "bonds_pct": 0.45,
                "tips_pct": 0.55,
                "frn_pct": 0.65,
            },
        ]
    ).to_csv(path, index=False)


def test_ratewall_holder_path_missing_named_scenario_fails_before_simulation(tmp_path):
    holder_path = tmp_path / "holder.csv"
    _write_holder_path(holder_path, scenario_id="other_scenario", quarter="2025Q1")
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "holder_absorption_path_file": str(holder_path),
    }

    with pytest.raises(ValueError, match="no exact scenario 'missing_holder_scenario'"):
        run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="missing_holder_scenario",
        )


def test_ratewall_holder_path_missing_required_quarter_fails_before_simulation(tmp_path):
    holder_path = tmp_path / "holder.csv"
    _write_holder_path(holder_path, scenario_id="quarter_gap_scenario", quarter="2025Q2")
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "holder_absorption_path_file": str(holder_path),
    }

    with pytest.raises(ValueError, match="missing required quarters: \\['2025Q1'\\]"):
        run_simulation(
            params,
            "2025-01-01",
            "2025-01-19",
            freq="W",
            scenario_name="quarter_gap_scenario",
        )


def test_ratewall_holder_path_exact_scenario_preserves_preferences_and_records_source(tmp_path):
    holder_path = tmp_path / "holder.csv"
    _write_holder_path(holder_path, scenario_id="exact_holder_scenario", quarter="2025Q1")
    params = minimal_params()
    baseline_banks = dict(params["sector_preferences"]["Banks"])
    params["ratewall_input_paths"] = {
        "holder_absorption_path_file": str(holder_path),
    }

    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="exact_holder_scenario",
    )

    metadata = results.attrs["run_metadata"]
    assert metadata["ratewall_holder_absorption_resolution_status"] == "exact_scenario"
    assert metadata["ratewall_holder_absorption_source_scenario_id"] == "exact_holder_scenario"
    assert baseline_banks == params["sector_preferences"]["Banks"]
    final_public_debt = float(
        results[["DebtHeld_Banks", "DebtHeld_DomesticNonBanks"]].iloc[-1].sum()
    )
    assert float(results["DebtHeld_Banks"].iloc[-1]) / final_public_debt == pytest.approx(0.25)
    assert float(results["DebtHeld_DomesticNonBanks"].iloc[-1]) / final_public_debt == pytest.approx(0.75)
    paths = export_ratewall_bundle(
        {"exact_holder_scenario": results},
        tmp_path / "bundle",
    )
    summary = pd.read_csv(paths["summary"])
    assert set(summary["holder_path_resolution_status"]) == {"exact_scenario"}
    assert set(summary["holder_path_source_scenario_id"]) == {"exact_holder_scenario"}


def test_ratewall_holder_path_configured_fallback_is_explicit_and_recorded(tmp_path):
    holder_path = tmp_path / "holder.csv"
    _write_holder_path(holder_path, scenario_id="declared_source", quarter="2025Q1")
    params = minimal_params()
    params["ratewall_input_paths"] = {
        "holder_absorption_path_file": str(holder_path),
        "holder_absorption_scenario_fallbacks": {
            "requested_scenario": "declared_source",
        },
    }

    results, _ = run_simulation(
        params,
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="requested_scenario",
    )

    metadata = results.attrs["run_metadata"]
    assert metadata["ratewall_holder_absorption_resolution_status"] == "configured_fallback"
    assert metadata["ratewall_holder_absorption_source_scenario_id"] == "declared_source"
    paths = export_ratewall_bundle(
        {"requested_scenario": results},
        tmp_path / "bundle",
    )
    summary = pd.read_csv(paths["summary"])
    assert set(summary["holder_path_resolution_status"]) == {"configured_fallback"}
    assert set(summary["holder_path_source_scenario_id"]) == {"declared_source"}


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
    assert not mmf_disclosure["central_default_eligible"]
    assert "superseded_by_private_subbucket_split_contract_0_3_0" in mmf_disclosure["binding_blocker"]
    split_routes = registry[registry["source_family"] == "tdcsim_private_subbucket"]
    assert set(split_routes["source_key"]) == {
        "domestic_nonbank_deposit_funded",
        "mmf_cash_fund_route",
    }


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
        "mmf_cash_fund_route",
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
        route_rows["source_key"] == "mmf_cash_fund_route"
    ].iloc[0]
    assert mmf_route["central_default_eligible"]
    assert not mmf_route["sensitivity_only"]
    assert "non_deposit_settlement_requires_an_explicit_on_rrp_share" in mmf_route[
        "binding_blocker"
    ]
    assert "does_not_identify_final_deposit_recipient" in mmf_route["binding_blocker"]


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


def test_ratewall_scenario_registry_drives_catalog_and_primary_rows(tmp_path, monkeypatch):
    registry_map = {
        scenario.scenario_id: scenario.curve_id
        for scenario in RATEWALL_SCENARIO_REGISTRY
    }
    assert SCENARIO_CURVE_MAP == registry_map
    monkeypatch.setattr(
        rib,
        "_cbo_budget_values",
        lambda _path: {2026: {"deficit_bil": -100.0, "net_interest_bil": 20.0}},
    )

    frame, _ = build_primary_flow_path(
        tmp_path / "primary.csv",
        cbo_budget_path=tmp_path / "unused.xlsx",
    )

    assert set(frame["scenario_id"]) == set(registry_map)


@pytest.mark.integration
def test_ratewall_source_backed_holder_path_is_instrument_specific(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    project_parent = next(
        (
            parent
            for parent in repo_root.parents
            if (parent / "ratewall").exists()
            and (parent / "tdcmix").exists()
            and (parent / "tsyparty").exists()
        ),
        repo_root.parent,
    )
    ratewall_root = project_parent / "ratewall"
    tdcmix_prior = (
        project_parent
        / "tdcmix"
        / "data"
        / "processed"
        / "holder_absorption_prior_contract.csv"
    )
    z1_absorption = (
        project_parent
        / "tdcmix"
        / "data"
        / "processed"
        / "z1_exact_holder_absorption_panel.csv"
    )
    bills_primary = (
        project_parent
        / "tsyparty"
        / "data"
        / "interim"
        / "bills_quarterly_composition.csv"
    )
    coupons_primary = (
        project_parent
        / "tsyparty"
        / "data"
        / "interim"
        / "nominal_coupons_quarterly_composition.csv"
    )
    assert ratewall_root.exists()
    assert tdcmix_prior.exists()
    assert z1_absorption.exists()

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
    assert str(private["mmf_collapsed_into_du"]).lower() == "false"
    assert "mmf_split_from_du_private_subbucket_explicit" in private["source_status"]
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

    assert set(frame["scenario_id"]) == {
        scenario.scenario_id for scenario in RATEWALL_SCENARIO_REGISTRY
    }
    current = frame[
        (frame["scenario_id"] == "domestic_nonbank_absorption_shift")
        & (frame["quarter"] == "2026Q1")
    ]
    current_by_holder = (
        current.assign(bills_pct=current["bills_pct"].astype(float))
        .groupby("holder_type")["bills_pct"]
        .sum()
    )
    assert metadata["scenario_shift_lambda_default"] == "1.00"
    assert current_by_holder["CB"] == pytest.approx(0.1)
    assert current_by_holder["Banks"] == pytest.approx(0.1)
    assert current_by_holder["Foreign"] == pytest.approx(0.1)
    assert current_by_holder["Private"] == pytest.approx(0.7)
    private_rows = current[current["holder_type"] == "Private"]
    assert set(private_rows["holder_subbucket"]) == {
        "domestic_nonbank_deposit_funded",
        "mmf_cash_fund_route",
    }


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


# WAMEST holdings in USD millions: (level, bill share). Measured bills are banks 100,
# foreign 200, money funds 400 and households 150 bn; measured nonbills are 100, 800,
# 100 and 450 bn. Foreign holders and households report market values (Financial
# Accounts LM series); banks and money funds report book values (FL series).
_OPENING_MARKET_VALUE_SECTORS = {"foreigners_total", "households_nonprofits"}
_OPENING_WAMEST_LEVELS = {
    "bank_us_chartered": (200_000.0, 0.5),
    "foreigners_total": (1_000_000.0, 0.2),
    "money_market_funds": (500_000.0, 0.8),
    "households_nonprofits": (600_000.0, 0.25),
}
# Auctions as (issue date, CUSIP, SOMA accepted in dollars, TIPS flag, index ratio). Only
# settlements after the SOMA as-of date (2026-04-29) and by the record date (2026-04-30)
# are Fed holdings SOMA does not yet show.
_OPENING_AUCTIONS = [
    ("2026-04-28", "FAKEBILL1", "9000000000", "No", ""),
    ("2026-04-30", "FAKEBILL1", "5000000000", "No", ""),
    ("2026-04-30", "FAKETIPS1", "10000000000", "Yes", "1.2"),
    ("2026-05-05", "FAKEBILL1", "3000000000", "No", ""),
]


def _opening_book_inputs(
    tmp_path,
    *,
    soma_rows=None,
    drop_sector=None,
    extra_table3=None,
    soma_as_of="2026-04-29",
    wamest_levels=None,
    auction_rows=None,
    nmfp_rows=None,
    frns=(),
):
    """Synthetic MSPD, SOMA, WAMEST, curve, auction and N-MFP inputs shaped like the real files.

    One bill, one note and one TIPS security; 100,000 of notes matured on the record
    date and are unpaid; 1% of notes are held intragovernmentally, so Table 1 is below
    the Table 3 class total. The baseline curve is a flat 5% on and before the record
    date, so the bill and the 4% note trade below par. ``frns`` adds floating-rate notes as
    (CUSIP, outstanding in millions). The money funds' N-MFP filings at the record date sit
    within the mix tolerance of the book they produce.
    """

    table1 = [
        {"security_type_desc": "Marketable", "security_class_desc": "Bills", "debt_held_public_mil_amt": "1000000"},
        {"security_type_desc": "Marketable", "security_class_desc": "Notes", "debt_held_public_mil_amt": "2079000"},
        {
            "security_type_desc": "Marketable",
            "security_class_desc": "Treasury Inflation-Protected Securities",
            "debt_held_public_mil_amt": "300000",
        },
    ]

    def security(class_desc, cusip, outstanding, maturity="", rate="0", issue="2026-01-01"):
        return {
            "security_class1_desc": class_desc,
            "security_class2_desc": cusip,
            "outstanding_amt": outstanding,
            "issue_date": issue,
            "maturity_date": maturity,
            "interest_rate_pct": rate,
            "yield_pct": "4.0",
            "issued_amt": outstanding,
            "redeemed_amt": "0",
        }

    table3 = [
        security("Bills Maturity Value", "FAKEBILL1", "1000000", "2026-09-01"),
        security("Bills Maturity Value", "Total Unmatured Treasury Bills", "1000000", issue=""),
        security("Bills Maturity Value", "Total Matured Treasury Bills", "*", issue=""),
        security("Bills Maturity Value", "Total Treasury Bills", "1000000", issue=""),
        security("Notes", "FAKENOTE1", "2000000", "2031-01-01", "4.0"),
        security("Notes", "Total Unmatured Treasury Notes", "2000000", issue=""),
        security("Notes", "Total Matured Treasury Notes", "100000", issue=""),
        security("Notes", "Total Treasury Notes", "2100000", issue=""),
        security("Inflation-Protected Securities", "FAKETIPS1", "300000", "2036-01-15", "1.0"),
        # Reopening detail rows carry no outstanding amount and are counted, not dropped silently.
        security("Inflation-Protected Securities", "FAKETIPS1", "", "2036-01-15", "1.0"),
        security("Inflation-Protected Securities", "Total Treasury TIPS", "300000", issue=""),
        *(extra_table3 or []),
    ]
    if frns:
        frn_total = str(sum(float(outstanding) for _, outstanding in frns))
        table1.append(
            {"security_type_desc": "Marketable", "security_class_desc": "Floating Rate Notes", "debt_held_public_mil_amt": frn_total}
        )
        table3 += [security("Floating Rate Notes", cusip, outstanding, "2027-04-30", "4.0") for cusip, outstanding in frns]
        table3.append(security("Floating Rate Notes", "Total Treasury Floating Rate Notes", frn_total, issue=""))
    soma = soma_rows or [
        ("FAKEBILL1", "Bills", "2026-09-01", "100000000000", ""),
        ("FAKENOTE1", "NotesBonds", "2031-01-01", "500000000000", ""),
        ("FAKETIPS1", "TIPS", "2036-01-15", "50000000000", "10000000000"),
        ("FAKEMAT01", "NotesBonds", "2026-04-30", "20000000000", ""),
        ("FAKEOLD01", "Bills", "2026-04-28", "7000000000", ""),
    ]
    soma_path = tmp_path / "soma.csv"
    pd.DataFrame(
        [
            {
                "As Of Date": soma_as_of,
                "CUSIP": f"'{cusip}'",
                "Security Type": kind,
                "Maturity Date": maturity,
                "Par Value": par,
                "Inflation Compensation": inflation,
            }
            for cusip, kind, maturity, par, inflation in soma
        ]
    ).to_csv(soma_path, index=False)

    levels = wamest_levels or _OPENING_WAMEST_LEVELS
    wamest_rows = []
    for sector in rib.WAMEST_SECTOR_ROUTES:
        if sector == drop_sector:
            continue
        level, bill_share = levels.get(sector, (0.0, 0.0))
        wamest_rows.append(
            {
                "date": "2026-03-31",
                "sector_key": sector,
                "level": level,
                "bill_share": bill_share,
                "publication_status": "published_estimate",
                "revaluation_valuation_basis": (
                    "z1_market_value_level" if sector in _OPENING_MARKET_VALUE_SECTORS
                    else "z1_level_basis_not_confirmed_market_value"
                ),
            }
        )
    wamest_path = tmp_path / "wamest.csv"
    pd.DataFrame(wamest_rows).to_csv(wamest_path, index=False)

    # A later baseline curve and another scenario carry other rates, so pricing on any
    # curve but the baseline on or before the record date changes the factors.
    curve_path = tmp_path / "curve.csv"
    pd.DataFrame(
        [
            {"scenario_id": scenario, "curve_date": curve_date, "tenor_years": tenor, "nominal_rate": rate, "rate_unit": "decimal"}
            for scenario, curve_date, rate in (
                ("cbo_shape_preserving_baseline", "2026-04-01", 0.05),
                ("cbo_shape_preserving_baseline", "2026-07-01", 0.09),
                ("higher_for_longer_sensitivity", "2026-04-01", 0.09),
            )
            for tenor in (0.25, 1.0, 10.0, 30.0)
        ]
    ).to_csv(curve_path, index=False)

    auctions_path = tmp_path / "auctions.csv"
    auctions = [
        {
            "issue_date": issue,
            "cusip": cusip,
            "soma_accepted": amount,
            "inflation_index_security": tips,
            "index_ratio_on_issue_date": ratio,
            "floating_rate": "No",
            "total_accepted": "",
        }
        for issue, cusip, amount, tips, ratio in (_OPENING_AUCTIONS if auction_rows is None else auction_rows)
    ]
    if auction_rows is None:
        auctions += _frn_auction_rows(_CALIBRATION_FRN_AUCTIONS)
    pd.DataFrame(auctions).to_csv(auctions_path, index=False)
    return {
        "record_date": "2026-04-30",
        "table1_rows": table1,
        "table3_rows": table3,
        "soma_path": soma_path,
        "wamest_path": wamest_path,
        "yield_curve_path": curve_path,
        "auctions_path": auctions_path,
        "nmfp_holdings_path": _write_nmfp(
            tmp_path / rib.NMFP_HOLDINGS_FILENAME, [*(nmfp_rows or _OPENING_NMFP_ROWS), *_CALIBRATION_NMFP_ROWS]
        ),
    }


# Two FRNs settle in the calibration window, each a new issue and two reopenings of 86 bn
# to the public (the Fed adds 2 bn at the new issue). A month after its last tranche money
# funds hold 60.2 bn of the first and 51.6 bn of the second: shares 0.70 and 0.60, 0.65 in all.
_CALIBRATION_FRN_AUCTIONS = [
    # issue date, CUSIP, reopening, SOMA accepted, total accepted (dollars)
    ("2025-04-30", "FAKEFRNA1", "No", "2000000000", "32000000000"),
    ("2025-05-30", "FAKEFRNA1", "Yes", "0", "28000000000"),
    ("2025-06-27", "FAKEFRNA1", "Yes", "0", "28000000000"),
    ("2025-07-31", "FAKEFRNB1", "No", "0", "30000000000"),
    ("2025-08-29", "FAKEFRNB1", "Yes", "0", "28000000000"),
    ("2025-09-26", "FAKEFRNB1", "Yes", "0", "28000000000"),
]
_CALIBRATION_FRN_MATURITY = {"FAKEFRNA1": "2027-04-30", "FAKEFRNB1": "2027-07-31", "FAKEFRNZ1": "2026-04-30"}


def _frn_auction_rows(tranches) -> list[dict]:
    """Auction-file rows for FRN tranches: a reopening names its new issue's date."""

    first_issue = {}
    rows = []
    for issue, cusip, reopening, soma, total in tranches:
        first_issue.setdefault(cusip, issue)
        rows.append(
            {
                "issue_date": issue,
                "cusip": cusip,
                "soma_accepted": soma,
                "inflation_index_security": "No",
                "index_ratio_on_issue_date": "",
                "floating_rate": "Yes",
                "total_accepted": total,
                "reopening": reopening,
                "original_issue_date": first_issue[cusip] if reopening == "Yes" else "",
                "maturity_date": _CALIBRATION_FRN_MATURITY[cusip],
                "security_term": "2-Year" if reopening == "No" else "1-Year 11-Month",
            }
        )
    return rows
_CALIBRATION_NMFP_ROWS = [
    # Before the last tranche settles; not the measurement.
    ("2025-06-30", "FAKEFRNA1", "frn", "2027-04-30", 50.0),
    ("2025-07-31", "FAKEFRNA1", "frn", "2027-04-30", 60.2),
    ("2025-10-31", "FAKEFRNB1", "frn", "2027-07-31", 51.6),
]


# N-MFP money-fund holdings as (report date, CUSIP, class, maturity, value in bn).
# The book gives money funds 487.6 bn of bills and 19.1 bn of notes matured on the record
# date; the filings sit within 2% of each.
_OPENING_NMFP_ROWS = [
    ("2026-04-30", "FAKEBILL1", "bill", "2026-09-01", 480.0),
    ("2026-04-30", "FAKEMAT01", "fixed_coupon", "2026-04-30", 19.0),
    # Filed in another month, so not part of this book's check.
    ("2026-03-31", "FAKEBILL1", "bill", "2026-09-01", 1.0),
]


def _write_nmfp(path: Path, rows) -> Path:
    """Write an N-MFP extract and the provenance file its producer would write beside it."""

    pd.DataFrame(
        [
            {
                "report_date": report_date,
                "cusip": cusip,
                "security_class": security_class,
                "maturity_date": maturity,
                "value_bil": value,
                "class_source": "mspd_cusip",
                "series_count": 1,
            }
            for report_date, cusip, security_class, maturity, value in rows
        ]
    ).to_csv(path, index=False)
    provenance = {
        "producer": "scripts/extract_sec_nmfp_mmf_holdings.py",
        "zips": {"fake_nmfp.zip": {"sha256": "0" * 64, "bytes": 1}},
        "filed_through": "2026-06-22",
        "selected_filings": {"2026-04": [["S1", "0000000000-26-000001", "2026-05-06", "fake_nmfp.zip"]]},
        "output": {"file": path.name, "sha256": rib._sha256(path)},
    }
    path.with_suffix(".provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return path


def test_opening_book_pins_cb_to_soma_and_keeps_measured_holders_at_par(tmp_path):
    frame, metadata = rib.build_initial_portfolio(tmp_path / "cohorts.csv", **_opening_book_inputs(tmp_path))
    book = frame.assign(FaceValue=frame["FaceValue"].astype(float))
    face = book.set_index(["cusip_or_synthetic_id", "HolderType", "HolderSubBucket"])["FaceValue"]
    is_bill = book["cusip_or_synthetic_id"].isin(["FAKEBILL1"])
    route = book["HolderType"] + "/" + book["HolderSubBucket"].fillna("")
    route = route.str.rstrip("/")
    bills = book[is_bill].groupby(route[is_bill])["FaceValue"].sum()
    nonbills = book[~is_bill].groupby(route[~is_bill])["FaceValue"].sum()
    dn, mmf = "Private/domestic_nonbank_deposit_funded", "Private/mmf_cash_fund_route"

    # MSPD and SOMA are in millions; the book is in billions. CB holds exactly the
    # SOMA amount of each CUSIP, so Fed maturities land on the actual dates and sizes.
    # The 1% intragovernmental scaling applies to the public total, never to SOMA.
    # The Fed's 5 bn add-on settling after the SOMA as-of date is a Fed holding on the
    # record date; add-ons SOMA already shows, or settling later, are not.
    assert face[("FAKEBILL1", "CB", "")] == pytest.approx(105.0)
    assert face[("FAKENOTE1", "CB", "")] == pytest.approx(500.0)
    # TIPS outstanding in MSPD includes the inflation adjustment, so CB does too, and a
    # TIPS add-on carries its index ratio (10 bn par at 1.2).
    assert face[("FAKETIPS1", "CB", "")] == pytest.approx(60.0 + 12.0)
    add_ons = metadata["soma_auction_add_ons"]
    assert add_ons["auctions"] == 2 and add_ons["amount_bil"] == pytest.approx(17.0)
    assert add_ons["amount_bil_by_cusip"] == {"FAKEBILL1": pytest.approx(5.0), "FAKETIPS1": pytest.approx(12.0)}
    assert "auction_soma_add_ons" in book.loc[book["HolderType"].eq("CB") & is_bill, "source_status"].iloc[0]

    # The bill (124 days to maturity, flat 5% curve) trades below par, and a bill carries
    # that discount at market value or at book, so measured bill holders hold more par
    # than they report. Banks, foreign holders and money funds keep exactly that; domestic
    # nonbanks, the residual sector, take the rest of the 895 bn non-Fed bills.
    # Simple interest over the part of the half-year to maturity, as Appendix B discounts a
    # part-period (for a bill, Treasury's investment-rate convention).
    bill_factor = 1.0 + (124.0 / 184.0) * 0.025
    allocation = metadata["opening_allocation"]
    assert allocation["curve_date"] == "2026-04-01"
    assert allocation["groups"]["bills"]["par_to_market_factor"] == pytest.approx(bill_factor)
    assert bills["Banks"] == pytest.approx(100.0 * bill_factor)
    assert bills["Foreign"] == pytest.approx(200.0 * bill_factor)
    swaps = allocation["money_fund_eligibility_swap"]["face_bil_by_class"]
    assert sum(swaps["from_mmf"].values()) == pytest.approx(sum(swaps["to_mmf"].values()))
    assert swaps["from_mmf"]["Notes"] > 0.0
    assert swaps["from_mmf"]["Treasury Inflation-Protected Securities"] > 0.0
    assert swaps["to_mmf"]["Bills"] > 0.0
    bills_to_mmf = swaps["to_mmf"]["Bills"]
    assert bills[mmf] == pytest.approx(400.0 * bill_factor + bills_to_mmf)
    assert bills[dn] == pytest.approx(895.0 - 700.0 * bill_factor - bills_to_mmf)
    assert allocation["groups"]["bills"]["domestic_nonbank_residual_bil"] == pytest.approx(bills[dn] + bills_to_mmf)
    from mmf_eligibility import mmf_eligible_mask

    assert mmf_eligible_mask(book.loc[route.eq(mmf)], "2026-04-30").all()
    # Spreading the unmeasured remainder in proportion to measured holdings, the old
    # rule, would hand foreign holders par that no source says they hold.
    assert abs(bills["Foreign"] - 895.0 * 200.0 / 850.0) > 5.0

    # Nonbills: the 4% note is below par on the 5% curve, matured-unpaid notes are paid at
    # par, and TIPS, which need a real curve, stay out of the factor sample but take the
    # group's factor through their holders.
    group = allocation["groups"]["nonbills"]
    note_noncb = 2000.0 * 0.99 - 500.0
    matured_noncb = 100.0 * 0.99 - 20.0
    assert group["priced_par_bil"] == pytest.approx(note_noncb + matured_noncb)
    assert group["unpriced_tips_par_bil"] == pytest.approx(300.0 - 72.0)
    factor = group["par_to_market_factor"]
    assert factor == pytest.approx(group["priced_par_bil"] / group["priced_market_value_bil"])
    assert factor > 1.02
    # Foreign holders report market value, converted to par. Banks and money funds report
    # book value, which for coupon securities is read as par: converting it with the
    # discount-bond factor would hand them par no source says they hold.
    assert nonbills["Foreign"] == pytest.approx(800.0 * factor)
    assert nonbills["Banks"] == pytest.approx(100.0)
    assert bills[mmf] + nonbills[mmf] == pytest.approx(400.0 * bill_factor + 100.0)
    assert nonbills[dn] == pytest.approx(note_noncb + matured_noncb + 228.0 - 800.0 * factor - 200.0 + bills_to_mmf)
    assert group["measured_bil_by_basis"]["Banks"] == {"book_value": pytest.approx(100.0), "market_value": 0.0}
    # The split stays uniform within a group, so each TIPS holder takes the group weight.
    tips_noncb = 300.0 - 72.0
    assert face[("FAKETIPS1", "Foreign", "")] / tips_noncb == pytest.approx(group["route_weights"]["Foreign"])

    # New issuance keeps the measured WAMEST shares; the opening residual is a stock gap,
    # not evidence of who buys at auction.
    assert metadata["noncb_weights"]["bills"] == {
        "Banks": pytest.approx(100.0 / 850.0),
        "Foreign": pytest.approx(200.0 / 850.0),
        dn: pytest.approx(150.0 / 850.0),
        mmf: pytest.approx(400.0 / 850.0),
    }

    # Notes that matured on the record date and are paid the next business day stay in
    # the book as one cohort maturing on the record date, including the Fed's share, so
    # the pre-start rollover replaces them; live notes are not inflated to absorb them.
    matured = frame[frame["cusip_or_synthetic_id"].eq("matured_unpaid_notes")]
    assert set(matured["MaturityDate"]) == {"2026-04-30"}
    assert set(matured["MaturityCategory"]) == {"notes"}
    assert matured["FaceValue"].astype(float).sum() == pytest.approx(100.0 * 0.99)
    assert face[("matured_unpaid_notes", "CB", "")] == pytest.approx(20.0)

    # Splitting holders never changes the public totals in Table 1.
    assert book["FaceValue"].sum() == pytest.approx(1000.0 + 2079.0 + 300.0)

    assert metadata["soma_matured_unpaid_at_record_date"] == {"cusips": 1, "amount_bil": pytest.approx(20.0)}
    # A SOMA security paid before the record date is reported, not silently dropped.
    assert metadata["soma_paid_before_record_date"] == {"cusips": 1, "amount_bil": pytest.approx(7.0)}
    assert metadata["matured_unpaid_bil"] == {"Notes": pytest.approx(100.0)}
    assert metadata["mspd_rows_skipped"] == {"no_outstanding_amount": 1}
    assert metadata["wamest_date"] == "2026-03-31"
    assert set(frame.loc[frame["HolderType"].eq("CB") & ~frame["cusip_or_synthetic_id"].str.startswith("matured"), "observability_tier"]) == {
        "security_level_observed_holder"
    }


def test_opening_book_fails_when_measured_holdings_exceed_non_fed_par(tmp_path):
    # Foreign bills measured at 1,000 bn exceed the 895 bn of non-Fed bills; a negative
    # domestic nonbank holding must stop the build rather than be clipped to zero.
    levels = {**_OPENING_WAMEST_LEVELS, "foreigners_total": (5_000_000.0, 0.2)}
    inputs = _opening_book_inputs(tmp_path, wamest_levels=levels)
    with pytest.raises(ValueError, match="bills par holdings would be negative"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


def test_opening_book_fails_if_the_swap_cannot_replace_ineligible_money_fund_paper(tmp_path):
    levels = {**_OPENING_WAMEST_LEVELS, "money_market_funds": (675_000.0, 570.0 / 675.0)}
    inputs = _opening_book_inputs(tmp_path, wamest_levels=levels)
    with pytest.raises(ValueError, match="eligible domestic nonbank paper"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"soma_rows": [("FAKENEW01", "NotesBonds", "2030-01-01", "1000000000", "")]}, "absent from MSPD"),
        ({"soma_rows": [("FAKEBILL1", "Bills", "2026-09-01", "1500000000000", "")]}, "holds more of FAKEBILL1"),
        # An add-on can push CB past the public amount, or name a security MSPD lacks.
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKEBILL1", "950000000000", "No", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "holds more of FAKEBILL1"),
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKENEW02", "1000000000", "No", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "auctions settling in"),
        # A file that ends by the record date may miss add-ons, and a missing SOMA amount
        # in the window is not a zero.
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKEBILL1", "5000000000", "No", "")]}, "may miss settlements"),
        # So may one that starts after the SOMA date, and a CUSIP listed twice on one day.
        ({"auction_rows": [("2026-04-30", "FAKEBILL1", "5000000000", "No", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "may miss settlements after"),
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKEBILL1", "1", "No", ""), ("2026-04-30", "FAKEBILL1", "1", "No", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "twice on one settlement date"),
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKEBILL1", "", "No", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "unusable SOMA rows"),
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKETIPS1", "1000000000", "Yes", ""), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "unusable SOMA rows"),
        ({"auction_rows": [("2026-04-01", "FAKEOLD01", "0", "No", ""), ("2026-04-30", "FAKETIPS1", "1000000000", "", "1.2"), ("2026-05-05", "FAKEBILL1", "0", "No", "")]}, "unusable SOMA rows"),
        ({"soma_rows": [("FAKEBMAT1", "Bills", "2026-04-30", "1000000000", "")]}, "no matured amount"),
        ({"soma_as_of": "2026-05-06"}, "after MSPD record date"),
        # A dated security row without dates would otherwise vanish into the class scaling.
        ({"extra_table3": [{"security_class1_desc": "Notes", "security_class2_desc": "FAKENOTE2", "outstanding_amt": "5"}]}, "without valid dates"),
        (
            {
                "extra_table3": [
                    {
                        "security_class1_desc": "Bonds",
                        "security_class2_desc": "FAKEBOND1",
                        "outstanding_amt": "10",
                        "issue_date": "2020-01-01",
                        "maturity_date": "2050-01-01",
                    },
                    {"security_class1_desc": "Bonds", "security_class2_desc": "Total Treasury Bonds", "outstanding_amt": "10"},
                    {"security_class1_desc": "Bonds", "security_class2_desc": "Total Unmatured Treasury Bonds", "outstanding_amt": "12"},
                ]
            },
            "unmatured total",
        ),
    ],
)
def test_opening_book_fails_loudly_on_inconsistent_sources(tmp_path, overrides, message):
    inputs = _opening_book_inputs(tmp_path, **overrides)
    with pytest.raises(ValueError, match=message):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


def test_opening_book_fails_on_an_unknown_wamest_level_basis(tmp_path):
    # A level whose valuation basis is not declared cannot be converted to par safely.
    inputs = _opening_book_inputs(tmp_path)
    frame = pd.read_csv(inputs["wamest_path"])
    frame.loc[frame["sector_key"].eq("bank_us_chartered"), "revaluation_valuation_basis"] = "amortized_cost_v2"
    frame.to_csv(inputs["wamest_path"], index=False)
    with pytest.raises(ValueError, match="unknown level basis"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


@pytest.mark.parametrize("drop_sector", ["money_market_funds", None])
def test_opening_book_fails_on_unmapped_or_missing_wamest_sector(tmp_path, drop_sector):
    inputs = _opening_book_inputs(tmp_path, drop_sector=drop_sector)
    if drop_sector is None:
        frame = pd.read_csv(inputs["wamest_path"])
        extra = frame.iloc[[0]].assign(sector_key="new_sector")
        pd.concat([frame, extra]).to_csv(inputs["wamest_path"], index=False)
    with pytest.raises(ValueError, match="route registry"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


def test_opening_book_rebuild_keeps_manifest_bound_to_the_files(tmp_path):
    # The CBO package build trusts this manifest, so its hashes and opening-book entry
    # must describe the rebuilt bytes, and unrelated artifacts must be left alone.
    inputs = _opening_book_inputs(tmp_path)
    t1 = tmp_path / "mspd_table_1.csv"
    t3 = tmp_path / "mspd_table_3_market.csv"
    pd.DataFrame([{**row, "record_date": "2026-04-30"} for row in inputs["table1_rows"]]).to_csv(t1, index=False)
    pd.DataFrame([{**row, "record_date": "2026-04-30"} for row in inputs["table3_rows"]]).to_csv(t3, index=False)
    out = tmp_path / "inputs"
    out.mkdir()
    (out / "other.csv").write_text("x\n1\n", encoding="utf-8")
    other = {"sha256": "unchanged", "bytes": 4}
    curve = out / rib.YIELD_CURVE_SURFACE_FILENAME
    curve.write_bytes(inputs["yield_curve_path"].read_bytes())
    nmfp = inputs["nmfp_holdings_path"]
    for path in (nmfp, nmfp.with_suffix(".provenance.json")):
        (out / path.name).write_bytes(path.read_bytes())
    rebuild = {
        "record_date": "2026-04-30",
        "mspd_table_1": t1,
        "mspd_table_3_market": t3,
        "soma_holdings": inputs["soma_path"],
        "wamest_canonical": inputs["wamest_path"],
        "auctions_query": inputs["auctions_path"],
    }

    def write_manifest(curve_sha256):
        files = {"other.csv": other, curve.name: {"sha256": curve_sha256, "bytes": curve.stat().st_size}}
        (out / "tdcsim_ratewall_input_manifest.json").write_text(
            json.dumps({"artifacts": {"initial_portfolio": {}, "other": {"rows": 1}}, "files": files, "source_hierarchy": {}}),
            encoding="utf-8",
        )

    # The book is priced on the directory's curve, so a curve the manifest does not
    # describe would bind the book to an ungoverned surface.
    write_manifest("0" * 64)
    with pytest.raises(ValueError, match="does not match the manifest"):
        rib.rebuild_opening_book(out, **rebuild)
    write_manifest(rib._sha256(curve))
    manifest = rib.rebuild_opening_book(out, **rebuild)

    on_disk = json.loads((out / "tdcsim_ratewall_input_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    provenance = nmfp.with_suffix(".provenance.json").name
    for name in ("tdcsim_initial_portfolio_cohorts.csv", rib.SOMA_HOLDINGS_FILENAME, rib.NMFP_HOLDINGS_FILENAME, provenance):
        assert manifest["files"][name]["sha256"] == rib._sha256(out / name)
    assert manifest["files"]["other.csv"] == other
    assert manifest["artifacts"]["other"] == {"rows": 1}
    assert manifest["artifacts"]["initial_portfolio"]["cb_source"] == "nyfed_soma_treasury_holdings_by_cusip"
    assert manifest["artifacts"]["initial_portfolio"]["opening_allocation"]["curve_sha256"] == rib._sha256(curve)
    assert set(manifest["artifacts"]["initial_portfolio"]["mspd_source_sha256"]) == {t1.name, t3.name}


# Two FRNs, 100 and 50 bn, none held by the Fed. Money funds filed 60 bn of the first,
# more than they and domestic nonbanks hold of it, and 10 bn of the second, which
# domestic nonbanks can cover alone. Bills and matured notes sit within 1% of the book.
_FRN_FIXTURE = (("FAKEFRN01", "100000"), ("FAKEFRN02", "50000"))
_FRN_NMFP_ROWS = [
    ("2026-04-30", "FAKEBILL1", "bill", "2026-09-01", 420.0),
    ("2026-04-30", "FAKEMAT01", "fixed_coupon", "2026-04-30", 15.3),
    ("2026-04-30", "FAKEFRN01", "frn", "2027-04-30", 60.0),
    ("2026-04-30", "FAKEFRN02", "frn", "2027-04-30", 10.0),
]


def test_money_funds_hold_their_filed_frns_for_bills_drawing_domestic_nonbanks_first(tmp_path):
    # Observed filings outrank the uniform split for FRNs only: money funds hold what they
    # filed, the notes come from domestic nonbanks before foreign holders and banks, and
    # every giver takes money-fund bills at equal par, so no sector's par changes.
    inputs = _opening_book_inputs(tmp_path, frns=_FRN_FIXTURE, nmfp_rows=_FRN_NMFP_ROWS)
    frame, metadata = rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)
    book = frame.assign(FaceValue=frame["FaceValue"].astype(float))
    route = (book["HolderType"] + "/" + book["HolderSubBucket"].fillna("")).str.rstrip("/")
    face = book.groupby([book["cusip_or_synthetic_id"], route])["FaceValue"].sum()
    dn, mmf = "Private/domestic_nonbank_deposit_funded", "Private/mmf_cash_fund_route"
    allocation = metadata["opening_allocation"]
    weights = allocation["groups"]["nonbills"]["route_weights"]
    override = allocation["money_fund_frn_holdings_override"]
    moved = override["frn_to_mmf_bil_by_sector"]

    assert face[("FAKEFRN01", mmf)] == pytest.approx(60.0)
    assert face[("FAKEFRN02", mmf)] == pytest.approx(10.0)
    # Domestic nonbanks cover the second FRN alone: foreign holders and banks keep the split.
    for holder in ("Foreign", "Banks"):
        assert face[("FAKEFRN02", holder)] == pytest.approx(50.0 * weights[holder])
    assert face[("FAKEFRN02", dn)] > 1.0
    assert override["by_cusip"]["FAKEFRN02"]["from_bil"]["Foreign"] == 0.0
    # For the first, domestic nonbanks give all they hold, and foreign holders and banks the
    # shortfall in proportion to their holdings.
    assert ("FAKEFRN01", dn) not in face.index
    assert face[("FAKEFRN01", "Foreign")] + face[("FAKEFRN01", "Banks")] == pytest.approx(40.0)
    assert face[("FAKEFRN01", "Foreign")] / face[("FAKEFRN01", "Banks")] == pytest.approx(weights["Foreign"] / weights["Banks"])
    assert moved["Foreign"] + moved["Banks"] == pytest.approx(100.0 * (weights["Foreign"] + weights["Banks"]) - 40.0)
    assert moved["Foreign"] > 1.0 and moved["Banks"] > 0.1

    # Equal par: each sector keeps its opening par, and each giver's bills rise by the notes it gave.
    totals = book.groupby(route)["FaceValue"].sum()
    for key in (dn, "Foreign", "Banks", mmf):
        expected = allocation["groups"]["bills"]["par_bil"][key] + allocation["groups"]["nonbills"]["par_bil"][key]
        assert totals[key] == pytest.approx(expected)
    for key in ("Foreign", "Banks"):
        assert face[("FAKEBILL1", key)] == pytest.approx(allocation["groups"]["bills"]["par_bil"][key] + moved[key])
    assert override["bills_from_mmf_bil_by_sector"] == moved
    assert sum(moved.values()) == pytest.approx(
        sum(row["filed_bil"] - row["pre_override_bil"] for row in override["by_cusip"].values())
    )
    assert override["source"]["zips"] == {"fake_nmfp.zip": {"sha256": "0" * 64, "bytes": 1}}

    check = metadata["nmfp_mix_check"]
    assert check["status"] == "pass" and check["buckets"]["frn"]["relative_gap"] == pytest.approx(0.0)
    mmf_frns = book.loc[book["SecurityType"].eq("FRN") & route.eq(mmf)]
    assert set(mmf_frns["observability_tier"]) == {"security_level_observed_holder"}
    assert mmf_frns["source_family"].str.contains("sec_nmfp").all()


@pytest.mark.parametrize(
    ("frns", "filed_frns", "message"),
    [
        # Filings above the non-Fed outstanding cannot be met by any exchange.
        (_FRN_FIXTURE, {"FAKEFRN01": 101.0, "FAKEFRN02": 10.0}, "more than its non-Fed outstanding"),
        # Money funds must hold the bills they give.
        ((("FAKEFRN01", "1000000"),), {"FAKEFRN01": 600.0}, "bills to give"),
        # The override only moves notes to money funds; a filing below the split is not handled silently.
        (_FRN_FIXTURE, {"FAKEFRN01": 60.0, "FAKEFRN02": 1.0}, "less than"),
        # A book FRN the filings do not cover is missing input, not a zero holding.
        (_FRN_FIXTURE, {"FAKEFRN01": 60.0}, "do not cover FRN FAKEFRN02"),
        (_FRN_FIXTURE, {"FAKEFRN01": 60.0, "FAKEFRN02": 10.0, "FAKEFRN09": 1.0}, "not FRNs in the opening book"),
    ],
)
def test_frn_override_fails_loudly_when_the_filings_cannot_be_met(tmp_path, frns, filed_frns, message):
    rows = _FRN_NMFP_ROWS[:2] + [("2026-04-30", cusip, "frn", "2027-04-30", value) for cusip, value in filed_frns.items()]
    inputs = _opening_book_inputs(tmp_path, frns=frns, nmfp_rows=rows)
    with pytest.raises(ValueError, match=message):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


@pytest.mark.parametrize(
    ("filed_bills", "passes"),
    [(465.0, True), (463.0, False), (512.0, True), (514.0, False)],
)
def test_opening_book_rejects_a_money_fund_mix_outside_the_filings_tolerance(tmp_path, filed_bills, passes):
    # The book gives money funds 487.6 bn of bills: 465 and 512 filed are within 5%, 463 and
    # 514 are not, and a book that far from the filings must not be released.
    rows = [("2026-04-30", "FAKEBILL1", "bill", "2026-09-01", filed_bills), *_OPENING_NMFP_ROWS[1:]]
    inputs = _opening_book_inputs(tmp_path, nmfp_rows=rows)
    if passes:
        _, metadata = rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)
        assert abs(metadata["nmfp_mix_check"]["buckets"]["bills"]["relative_gap"]) <= rib.NMFP_MIX_TOLERANCE
    else:
        with pytest.raises(ValueError, match="more than 5% from the N-MFP filings"):
            rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)
        assert not (tmp_path / "cohorts.csv").exists()


def test_opening_book_rejects_n_mfp_holdings_without_matching_provenance(tmp_path):
    inputs = _opening_book_inputs(tmp_path)
    path = inputs["nmfp_holdings_path"]
    # A filing made after the declared information date was not known on it.
    sidecar = path.with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    provenance["selected_filings"]["2026-04"].append(["S2", "0000000000-26-000002", "2026-06-23", "fake_nmfp.zip"])
    sidecar.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="declared information date"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)
    inputs = _opening_book_inputs(tmp_path)
    path = inputs["nmfp_holdings_path"]
    path.write_text(path.read_text(encoding="utf-8").replace("480.0", "481.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="does not describe"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)
    # A month with no filings at the record date is missing input, not an empty override.
    inputs = _opening_book_inputs(tmp_path, nmfp_rows=[_OPENING_NMFP_ROWS[2]])
    with pytest.raises(ValueError, match="no money-fund holdings reported at 2026-04-30"):
        rib.build_initial_portfolio(tmp_path / "cohorts.csv", **inputs)


def test_opening_book_calibrates_the_money_fund_share_of_new_frns_from_n_mfp(tmp_path):
    # Money funds' share of new FRN supply is measured, not assumed: their holdings a month
    # after each FRN's last tranche over the public amount accepted across its tranches.
    from cbo_holder_preferences import holder_profile_from_opening_weights

    _, metadata = rib.build_initial_portfolio(tmp_path / "cohorts.csv", **_opening_book_inputs(tmp_path))
    calibration = metadata["frn_mmf_auction_share_calibration"]
    share = metadata["noncb_weights"]["frn_mmf_auction_share"]
    assert share == pytest.approx((60.2 + 51.6) / 172.0)
    assert calibration["range"] == pytest.approx([0.60, 0.70])
    assert calibration["auctions"] == 6
    observed = {row["cusip"]: row for row in calibration["observations"]}
    assert observed["FAKEFRNA1"]["measured_at"] == "2025-07-31"
    assert observed["FAKEFRNA1"]["public_accepted_bil"] == pytest.approx(86.0)
    assert calibration["auctions_source"]["sha256"] == rib._sha256(tmp_path / "auctions.csv")
    # The package's new-issuance profile reads these weights, so money funds take that share of new FRNs.
    preferences, routes = holder_profile_from_opening_weights(metadata["noncb_weights"])
    assert preferences["Private"]["frn_pct"] * routes["frn"]["mmf_cash_fund_route"] == pytest.approx(share)


_TRANCHES = _CALIBRATION_FRN_AUCTIONS
_NOT_THREE_TRANCHES = "not a 2-Year new issue and two reopenings"


def _calibrate(tmp_path, auctions, nmfp):
    auctions_path = tmp_path / "auctions.csv"
    pd.DataFrame(_frn_auction_rows(auctions)).to_csv(auctions_path, index=False)
    holdings, _ = rib.load_nmfp_mmf_holdings(_write_nmfp(tmp_path / rib.NMFP_HOLDINGS_FILENAME, nmfp))
    return rib.calibrate_mmf_frn_auction_share(holdings, auctions_path)


@pytest.mark.parametrize(
    ("auctions", "nmfp", "message"),
    [
        # An FRN with a tranche outside the window would be measured with unpaired holdings.
        (_TRANCHES + [("2026-04-30", "FAKEFRNB1", "Yes", "0", "28000000000")], _CALIBRATION_NMFP_ROWS, "outside the calibration window"),
        (_TRANCHES, _CALIBRATION_NMFP_ROWS[:2], "no report at 2025-10-31"),
        # A dropped reopening would cut the denominator and move the measurement month; a
        # duplicate would inflate it; a missing new issue leaves only reopenings.
        ([row for row in _TRANCHES if row[0] != "2025-08-29"], _CALIBRATION_NMFP_ROWS, _NOT_THREE_TRANCHES),
        ([row for row in _TRANCHES if row[0] != "2025-09-26"], _CALIBRATION_NMFP_ROWS, _NOT_THREE_TRANCHES),
        (_TRANCHES + [_TRANCHES[1]], _CALIBRATION_NMFP_ROWS, _NOT_THREE_TRANCHES),
        (_TRANCHES[1:], _CALIBRATION_NMFP_ROWS, _NOT_THREE_TRANCHES),
        ([*_TRANCHES[:5], ("2025-09-26", "FAKEFRNB1", "Yes", "0", "")], _CALIBRATION_NMFP_ROWS, "positive public amount"),
        # The month is reported but the FRN is absent: missing coverage, not zero holdings.
        (_TRANCHES, [*_CALIBRATION_NMFP_ROWS[:2], ("2025-10-31", "FAKEFRNA1", "frn", "2027-04-30", 60.0)], "do not cover FRN FAKEFRNB1"),
        ([("2024-04-30", "FAKEFRNZ1", "No", "0", "30000000000")], _CALIBRATION_NMFP_ROWS, "no FRN auction settling"),
    ],
)
def test_frn_auction_share_calibration_fails_loudly_on_incomplete_inputs(tmp_path, auctions, nmfp, message):
    with pytest.raises(ValueError, match=message):
        _calibrate(tmp_path, auctions, nmfp)


def test_frn_auction_share_counts_a_verified_zero_holding(tmp_path):
    # The extract writes a zero row for an outstanding FRN no fund holds; that is evidence
    # of no money-fund holdings, so it enters the share as zero.
    nmfp = [*_CALIBRATION_NMFP_ROWS[:2], ("2025-10-31", "FAKEFRNB1", "frn", "2027-07-31", 0.0)]
    calibration = _calibrate(tmp_path, _TRANCHES, nmfp)
    assert calibration["share"] == pytest.approx(60.2 / 172.0)
    assert calibration["range"] == pytest.approx([0.0, 0.70])


def _rebuilt_input_dir(tmp_path, name, **overrides):
    """Rebuild an opening book into its own input directory, as the governed build does."""

    sources = tmp_path / f"{name}_sources"
    sources.mkdir()
    inputs = _opening_book_inputs(sources, **overrides)
    out = tmp_path / name
    out.mkdir()
    tables = {}
    for key, rows in (("mspd_table_1", inputs["table1_rows"]), ("mspd_table_3_market", inputs["table3_rows"])):
        tables[key] = tmp_path / f"{name}_{key}.csv"
        pd.DataFrame([{**row, "record_date": "2026-04-30"} for row in rows]).to_csv(tables[key], index=False)
    curve = out / rib.YIELD_CURVE_SURFACE_FILENAME
    curve.write_bytes(inputs["yield_curve_path"].read_bytes())
    nmfp = inputs["nmfp_holdings_path"]
    for path in (nmfp, nmfp.with_suffix(".provenance.json")):
        (out / path.name).write_bytes(path.read_bytes())
    (out / "tdcsim_ratewall_input_manifest.json").write_text(
        json.dumps({"artifacts": {}, "files": {curve.name: {"sha256": rib._sha256(curve)}}, "source_hierarchy": {}}),
        encoding="utf-8",
    )
    manifest = rib.rebuild_opening_book(
        out,
        record_date="2026-04-30",
        soma_holdings=inputs["soma_path"],
        wamest_canonical=inputs["wamest_path"],
        auctions_query=inputs["auctions_path"],
        **tables,
    )
    return out, manifest, inputs["auctions_path"]


def test_package_rederives_the_n_mfp_book_and_share_and_rejects_an_older_book(tmp_path):
    # A package carries the book, its manifest and the N-MFP extract. Everything the
    # manifest claims from the filings must rebuild from those exact files, so a book from
    # before the filings were applied cannot ride along with the newer manifest.
    out, manifest, auctions = _rebuilt_input_dir(tmp_path, "new", frns=_FRN_FIXTURE, nmfp_rows=_FRN_NMFP_ROWS)
    book = out / rib.OPENING_BOOK_FILENAME
    check = {
        "book_path": book,
        "nmfp_holdings_path": out / rib.NMFP_HOLDINGS_FILENAME,
        "auctions_path": auctions,
        "information_date": "2026-06-22",
    }
    result = rib.verify_opening_book_against_nmfp(manifest, **check)
    assert result["frn_cusips"] == 2
    assert result["frn_mmf_auction_share"] == pytest.approx((60.2 + 51.6) / 172.0)

    # The older book: money funds filed 50 bn of the first FRN, not 60.
    older_rows = [_FRN_NMFP_ROWS[0], *_FRN_NMFP_ROWS[1:2], ("2026-04-30", "FAKEFRN01", "frn", "2027-04-30", 50.0), _FRN_NMFP_ROWS[3]]
    older, _, _ = _rebuilt_input_dir(tmp_path, "older", frns=_FRN_FIXTURE, nmfp_rows=older_rows)
    book.write_bytes((older / rib.OPENING_BOOK_FILENAME).read_bytes())
    with pytest.raises(ValueError, match="is not the file the opening-book manifest hashes"):
        rib.verify_opening_book_against_nmfp(manifest, **check)
    # Re-hashing the manifest to the older book still fails on the money funds' FRNs.
    rehashed = json.loads(json.dumps(manifest))
    rehashed["files"][rib.OPENING_BOOK_FILENAME]["sha256"] = rib._sha256(book)
    with pytest.raises(ValueError, match=r"differ from the N-MFP filings: \['FAKEFRN01'\]"):
        rib.verify_opening_book_against_nmfp(rehashed, **check)

    out, manifest, auctions = _rebuilt_input_dir(tmp_path, "again", frns=_FRN_FIXTURE, nmfp_rows=_FRN_NMFP_ROWS)
    check = {**check, "book_path": out / rib.OPENING_BOOK_FILENAME, "nmfp_holdings_path": out / rib.NMFP_HOLDINGS_FILENAME}
    # An edited auction share no longer matches the calibration it claims.
    edited = json.loads(json.dumps(manifest))
    edited["artifacts"]["initial_portfolio"]["noncb_weights"]["frn_mmf_auction_share"] = 0.054
    with pytest.raises(ValueError, match="auction share does not rebuild"):
        rib.verify_opening_book_against_nmfp(edited, **check)
    # Filings selected after the package's information date were not known on it.
    with pytest.raises(ValueError, match="after the information date 2026-06-21"):
        rib.verify_opening_book_against_nmfp(manifest, **{**check, "information_date": "2026-06-21"})


def test_soma_loader_rejects_files_without_required_columns(tmp_path):
    path = tmp_path / "soma.csv"
    pd.DataFrame([{"As Of Date": "2026-04-29", "CUSIP": "'FAKEBILL1'"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        rib.load_soma_cusip_holdings(path)


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


def test_ratewall_yield_surface_converts_cbo_discount_quote_to_investment_basis(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        rib,
        "_latest_treasury_curve",
        lambda: (
            "2026-02-20",
            {
                0.25: 0.04,
                0.5: 0.041,
                1.0: 0.042,
                2.0: 0.043,
                5.0: 0.044,
                10.0: 0.045,
                20.0: 0.046,
                30.0: 0.047,
            },
        ),
    )
    monkeypatch.setattr(
        rib,
        "_cbo_rate_values",
        lambda _path: {2026: {"cbo_3m_pct": 4.5, "cbo_10y_pct": 5.0}},
    )

    frame, _ = build_yield_curve_surface(
        tmp_path / "curve.csv",
        cbo_economic_path=tmp_path / "unused.xlsx",
    )

    baseline_3m = frame[
        (frame["scenario_id"] == "cbo_shape_preserving_baseline")
        & (frame["curve_date"] == "2026-01-01")
        & (frame["tenor_years"].astype(float) == 0.25)
    ]
    expected_3m = discount_rate_to_investment_rate(0.045, CBO_3M_BILL_DAYS)
    assert float(baseline_3m.iloc[0]["nominal_rate"]) == pytest.approx(expected_3m)
    assert (expected_3m - 0.045) * 10_000 == pytest.approx(11.4999, abs=1e-3)


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


def test_a_reopened_bill_is_priced_over_every_auction_tranche():
    # 912797UB1 (MSPD Table 3, 2026-05-31) was auctioned three times before it matured on
    # 2026-06-23. Pricing the whole issue at the first tranche's discount overstated its
    # historical discount, which the engine later reports as bill interest at redemption.
    maturity = pd.Timestamp("2026-06-23")
    tranches = [
        (69193.5912, rib._bill_discount_price(0.03595, pd.Timestamp("2026-02-24"), maturity)),
        (75998.3991, rib._bill_discount_price(0.03605, pd.Timestamp("2026-04-28"), maturity)),
        (103895.3776, rib._bill_discount_price(0.0361, pd.Timestamp("2026-05-26"), maturity)),
    ]
    # Each tranche reproduces Treasury's auction price per 100.
    assert [round(100 * price, 6) for _, price in tranches] == [98.811653, 99.439222, 99.719222]
    price = rib._bill_tranche_price("912797UB1", tranches, 249087.3679)
    public_face = 249.068759812
    assert public_face * price == pytest.approx(247.528717238, abs=1e-6)
    assert public_face * tranches[0][1] == pytest.approx(246.108958123, abs=1e-6)  # the old first-tranche basis


def test_bill_tranches_must_cover_the_outstanding_amount():
    with pytest.raises(ValueError, match="bill tranches of X1 sum to"):
        rib._bill_tranche_price("X1", [(100.0, 0.99)], 250.0)
    with pytest.raises(ValueError, match="bill tranches of X2 sum to"):
        rib._bill_tranche_price("X2", [], 250.0)
