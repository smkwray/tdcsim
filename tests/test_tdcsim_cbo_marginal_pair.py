import json
from pathlib import Path

import pandas as pd
import pytest

from ratewall_marginal_tdc_contract import validate_ratewall_marginal_tdc_summary
from tdcsim_cbo._json import sha256_file, write_json
from tdcsim_cbo.marginal_tdc import (
    MANIFEST_FILE,
    SUMMARY_FILE,
    DENOMINATOR_EQUIVALENCE_KEY,
    FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY,
    FISCAL_INJECTION_OBJECT_ID,
    FISCAL_INJECTION_SHOCK_PATH_ID,
    OBJECT_ID,
    SHOCK_PATH_ID,
    STATE_MANIFEST_FILE,
    MarginalTdcPairError,
    assemble_marginal_tdc_pair,
    verify_marginal_tdc_pair,
)


def test_marginal_tdc_pair_assembles_ratewall_summary_and_verifies(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"
    assert validate_ratewall_marginal_tdc_summary(summary)["status"] == "pass"
    assert summary.loc[0, "delta_tdc_change_bil"] == pytest.approx(3.0)
    assert summary.loc[0, "delta_overlap_bil"] == pytest.approx(1.0)
    assert summary.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(2.0)
    assert summary.loc[0, "legacy_chi_support_diagnostic_bil"] == pytest.approx(0.4)
    assert summary.loc[0, "state_manifest_status"] == "pass"
    assert summary.loc[0, "tdc_amount_basis"] == "pre_beta_ex_overlap_delta"
    assert (result.output_dir / STATE_MANIFEST_FILE).exists()
    assert summary.loc[0, "canonical_ratio_entry"] is False or str(summary.loc[0, "canonical_ratio_entry"]).lower() == "false"


def test_marginal_tdc_pair_deposit_creation_split_current_check_against(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    _write_current_split_check_against(baseline, shock)
    spec = _pair_spec(tmp_path, baseline, shock)
    spec["demand_conversion_cases"][0]["beta"] = 0.5307509589554447
    spec["demand_conversion_cases"][0]["chi"] = 0.07

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    components = pd.read_csv(result.components_path)
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"
    assert summary.loc[0, "delta_tdc_ex_overlap_interest_driven_excluded_bil"] == pytest.approx(
        3.1419512411257706
    )
    assert summary.loc[0, "delta_tdc_ex_overlap_non_interest_admissible_bil"] == pytest.approx(
        1.0937727536351083
    )
    assert summary.loc[0, "delta_tdc_ex_overlap_split_remainder_bil"] == pytest.approx(0.0)
    assert summary.loc[0, "delta_tdc_ex_overlap_reconciled_bil"] == pytest.approx(4.235723994760747)
    assert summary.loc[0, "tdc_materialized_deposit_stock_admissible_bil"] == pytest.approx(
        0.5805209378711711
    )
    assert summary.loc[0, "tdc_income_addendum_gross_interest_bil"] == pytest.approx(
        0.02031823282549099
    )
    principal = components[components["payment_type"] == "principal"]
    assert set(principal["deposit_creation_driver_bucket"]) == {"non_interest_principal_redemption_admissible"}
    assert principal["tdc_split_component_non_interest_admissible_bil"].sum() == pytest.approx(
        -6.939569771067226
    )
    admitted = components[components["tdc_split_component_admitted"].astype(str).str.lower() == "true"]
    assert not admitted["component_family"].astype(str).eq("debt_service_interest").any()
    assert not admitted["payment_type"].astype(str).isin(["bill_discount", "fixed_coupon", "frn_interest", "tips_coupon"]).any()
    plumbing = components[components["component_family"] == "route_plumbing_memo"].iloc[0]
    assert plumbing["deposit_creation_driver_bucket"] == "route_plumbing_memo_excluded"
    assert plumbing["tdc_split_component_non_interest_admissible_bil"] == pytest.approx(0.0)


def test_marginal_tdc_pair_verifier_fails_closed_on_missing_split(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)
    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")
    summary = pd.read_csv(result.summary_path).drop(columns=["delta_tdc_ex_overlap_non_interest_admissible_bil"])
    summary.to_csv(result.summary_path, index=False)
    _refresh_pair_manifest_file(result.output_dir, SUMMARY_FILE)

    with pytest.raises(MarginalTdcPairError, match="missing required columns|missing required column|missing required"):
        verify_marginal_tdc_pair(result.output_dir)


def test_marginal_tdc_pair_uses_ratewall_period_when_present(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)
    spec["ratewall_period"] = "2026"

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    components = pd.read_csv(result.components_path)
    assert set(summary["period"].astype(str)) == {"2026"}
    assert set(components["period"].astype(str)) == {"2026"}
    assert set(summary["period_end"].astype(str)) == {"2027-01-01"}
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"


def test_marginal_tdc_pair_collapses_raw_rows_to_ratewall_period(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    _split_tdc_rows(baseline, tdc_values=[4.0, 6.0], overlap_values=[1.0, 1.0])
    _split_tdc_rows(shock, tdc_values=[6.0, 7.0], overlap_values=[1.5, 1.5])
    spec = _pair_spec(tmp_path, baseline, shock)
    spec["ratewall_period"] = "2026"

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    assert len(summary) == 1
    assert summary.loc[0, "tdc_change_baseline_bil"] == pytest.approx(10.0)
    assert summary.loc[0, "tdc_change_shock_bil"] == pytest.approx(13.0)
    assert summary.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(2.0)
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"


def test_marginal_tdc_pair_excludes_overlap_components_from_support(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    _append_component(
        baseline,
        component_key="direct_interest_overlap",
        amount=0.0,
        enters_tdc_default=False,
    )
    _append_component(
        shock,
        component_key="direct_interest_overlap",
        amount=1.0,
        enters_tdc_default=False,
    )
    spec = _pair_spec(tmp_path, baseline, shock)

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    components = pd.read_csv(result.components_path)
    excluded = components[components["component_key"] == "direct_interest_overlap"].iloc[0]
    assert excluded["included_in_delta_tdc_ex_overlap"] is False or str(
        excluded["included_in_delta_tdc_ex_overlap"]
    ).lower() == "false"
    assert excluded["marginal_component_support_bil"] == pytest.approx(0.0)
    assert summary.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(2.0)
    assert summary.loc[0, "legacy_chi_support_diagnostic_bil"] == pytest.approx(0.4)
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"


def test_marginal_tdc_pair_rejects_state_fingerprint_mismatch(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)
    spec["shock_state_fingerprint_sha256"] = "b" * 64

    with pytest.raises(MarginalTdcPairError, match="state fingerprints"):
        assemble_marginal_tdc_pair(spec, tmp_path / "pair")


def test_two_state_pairs_may_differ_across_pairs_without_cross_state_delta(tmp_path: Path) -> None:
    baseline_a, shock_a = _write_pair_runs(tmp_path / "a", tdc_base=10.0, tdc_shock=13.0)
    baseline_b, shock_b = _write_pair_runs(tmp_path / "b", tdc_base=20.0, tdc_shock=25.0)
    spec_a = _pair_spec(
        tmp_path / "a",
        baseline_a,
        shock_a,
        state_id="fixture_state_a",
        state_fingerprint="a" * 64,
        opening_tdc_stock=10.0,
    )
    spec_b = _pair_spec(
        tmp_path / "b",
        baseline_b,
        shock_b,
        state_id="fixture_state_b",
        state_fingerprint="b" * 64,
        opening_tdc_stock=20.0,
    )

    result_a = assemble_marginal_tdc_pair(spec_a, tmp_path / "pair-a")
    result_b = assemble_marginal_tdc_pair(spec_b, tmp_path / "pair-b")

    summary_a = pd.read_csv(result_a.summary_path)
    summary_b = pd.read_csv(result_b.summary_path)
    assert summary_a.loc[0, "state_fingerprint_sha256"] != summary_b.loc[0, "state_fingerprint_sha256"]
    assert summary_a.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(2.0)
    assert summary_b.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(4.0)


def test_marginal_tdc_pair_fails_closed_on_non_rate_input_drift(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path, shock_primary_deficit="mutated primary path")
    spec = _pair_spec(tmp_path, baseline, shock)

    with pytest.raises(MarginalTdcPairError, match="non-rate compiled input drift"):
        assemble_marginal_tdc_pair(spec, tmp_path / "pair")


def test_marginal_tdc_pair_fails_closed_on_wrong_path_area(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path, shock_bp=50.0)
    spec = _pair_spec(tmp_path, baseline, shock)

    with pytest.raises(MarginalTdcPairError, match="\\+100bp|100 bp-years"):
        assemble_marginal_tdc_pair(spec, tmp_path / "pair")


def test_marginal_tdc_pair_fails_closed_when_overlap_removed(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    summary_path = shock / "outputs" / "tdcsim_period_tdc_summary.csv"
    summary = pd.read_csv(summary_path).drop(columns=["overlap_cashflow_bil"])
    summary.to_csv(summary_path, index=False)
    _refresh_output_artifact(shock, "tdcsim_period_tdc_summary.csv")
    spec = _pair_spec(tmp_path, baseline, shock)

    with pytest.raises(MarginalTdcPairError, match="missing required columns"):
        assemble_marginal_tdc_pair(spec, tmp_path / "pair")


def test_marginal_tdc_pair_verifier_rejects_gross_delta_substitution(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)
    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")
    summary = pd.read_csv(result.summary_path)
    summary.loc[0, "legacy_chi_support_diagnostic_bil"] = (
        summary.loc[0, "delta_tdc_change_bil"] * summary.loc[0, "beta"] * summary.loc[0, "chi"]
    )
    summary.to_csv(result.summary_path, index=False)
    _refresh_pair_manifest_file(result.output_dir, SUMMARY_FILE)

    with pytest.raises(MarginalTdcPairError, match="support identity"):
        verify_marginal_tdc_pair(result.output_dir)


def test_marginal_tdc_pair_verifier_rejects_beta_change_without_manifest_update(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path)
    spec = _pair_spec(tmp_path, baseline, shock)
    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")
    summary = pd.read_csv(result.summary_path)
    summary.loc[0, "beta"] = 0.9
    summary.to_csv(result.summary_path, index=False)

    with pytest.raises(MarginalTdcPairError, match="SHA mismatch"):
        verify_marginal_tdc_pair(result.output_dir)


def test_fiscal_injection_pair_allows_named_non_rate_drift_without_rate_shock(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path, shock_bp=0.0, tdc_shock=3013.0)
    _make_fiscal_injection_run(shock)
    spec = _pair_spec(tmp_path, baseline, shock)
    spec.update(
        {
            "object_id": FISCAL_INJECTION_OBJECT_ID,
            "shock_path_id": FISCAL_INJECTION_SHOCK_PATH_ID,
            "shock_bps_year": 0,
            "denominator_equivalence_key": FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY,
            "require_same_non_rate_compiled_inputs": False,
            "one_named_rate_shock_only": False,
            "compiled_non_rate_inputs_digest": "",
            "demand_conversion_cases": [
                {
                    "demand_conversion_case": "pre_beta_pair",
                    "beta": 1.0,
                    "beta_assumption_id": "ratewall_side_conversion_pending",
                    "beta_source_status": "not_applied_in_tdcsim_fiscal_injection_pair",
                    "chi": 1.0,
                    "chi_assumption_id": "ratewall_side_conversion_pending",
                    "chi_source_status": "not_applied_in_tdcsim_fiscal_injection_pair",
                }
            ],
        }
    )

    result = assemble_marginal_tdc_pair(spec, tmp_path / "pair")

    summary = pd.read_csv(result.summary_path)
    assert verify_marginal_tdc_pair(result.output_dir)["status"] == "pass"
    assert summary.loc[0, "shock_path_id"] == FISCAL_INJECTION_SHOCK_PATH_ID
    assert summary.loc[0, "shock_bps_year"] == 0
    assert summary.loc[0, "rate_shock_only_status"] == "not_applicable_fiscal_injection_no_rate_shock"
    assert summary.loc[0, "delta_tdc_ex_overlap_bil"] == pytest.approx(3002.0)
    assert summary.loc[0, "tdc_amount_basis"] == "pre_beta_ex_overlap_delta"


def test_fiscal_injection_pair_rejects_rate_input_drift(tmp_path: Path) -> None:
    baseline, shock = _write_pair_runs(tmp_path, shock_bp=25.0, tdc_shock=3013.0)
    _make_fiscal_injection_run(shock)
    spec = _pair_spec(tmp_path, baseline, shock)
    spec.update(
        {
            "object_id": FISCAL_INJECTION_OBJECT_ID,
            "shock_path_id": FISCAL_INJECTION_SHOCK_PATH_ID,
            "shock_bps_year": 0,
            "denominator_equivalence_key": FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY,
            "require_same_non_rate_compiled_inputs": False,
            "one_named_rate_shock_only": False,
            "compiled_non_rate_inputs_digest": "",
        }
    )

    with pytest.raises(MarginalTdcPairError, match="must not change rate input"):
        assemble_marginal_tdc_pair(spec, tmp_path / "pair")


def _write_pair_runs(
    tmp_path: Path,
    *,
    shock_bp: float = 100.0,
    shock_primary_deficit: str = "same primary path",
    tdc_base: float = 10.0,
    tdc_shock: float = 13.0,
) -> tuple[Path, Path]:
    baseline = _write_run(
        tmp_path / "baseline-run",
        scenario_id="baseline_v1",
        run_id="baseline_v1-abc",
        tdc_change=tdc_base,
        overlap=2.0,
        shock_bp=0.0,
        primary_deficit_payload="same primary path",
        shock=False,
    )
    shock = _write_run(
        tmp_path / "shock-run",
        scenario_id="shock_v1",
        run_id="shock_v1-def",
        tdc_change=tdc_shock,
        overlap=3.0,
        shock_bp=shock_bp,
        primary_deficit_payload=shock_primary_deficit,
        shock=True,
    )
    return baseline, shock


def _write_run(
    run_dir: Path,
    *,
    scenario_id: str,
    run_id: str,
    tdc_change: float,
    overlap: float,
    shock_bp: float,
    primary_deficit_payload: str,
    shock: bool,
) -> Path:
    outputs = run_dir / "outputs"
    inputs = run_dir / "compile" / "compiled" / "forecast_inputs"
    outputs.mkdir(parents=True)
    inputs.mkdir(parents=True)
    summary = pd.DataFrame(
        [
            {
                "period_start": "2026-01-01",
                "period_end": "2027-01-01",
                "tdc_change_bil": tdc_change,
                "tdc_fiscal_flow_bil": tdc_change,
                "tdc_debt_service_bil": 0.0,
                "tdc_auction_absorption_du_bil": 0.0,
                "tdc_secondary_trades_bil": 0.0,
                "tdc_other_bil": 0.0,
                "overlap_cashflow_bil": overlap,
                "tdc_change_ex_overlap_bil": tdc_change - overlap,
                "component_sum_bil": tdc_change,
                "component_sum_error_bil": 0.0,
            }
        ]
    )
    components = pd.DataFrame(
        [
            {
                "period_start": "2026-01-01",
                "period_end": "2027-01-01",
                "component_key": "fiscal_flow",
                "component_family": "fiscal",
                "holder_sector": "Private",
                "holder_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "all",
                "payment_type": "primary_deficit_or_surplus",
                "accounting_basis": "signed_net_primary_proxy",
                "amount_bil": tdc_change - overlap,
                "enters_direct_interest_support": False,
                "enters_tdc_deposit_support_default": True,
                "tdc_amount_basis": "post_mmf_route_pass_through_pre_ratewall_beta_chi",
                "overlap_policy": "domestic_nonbank_nominal_interest_components_enter_direct_support_not_default_tdc_support",
            }
        ]
    )
    route = pd.DataFrame(
        [
            {
                "period_start": "2026-01-01",
                "period_end": "2027-01-01",
                "route_holder_sector": "Private",
                "route_holder_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "Fixed",
                "maturity_bucket": "notes",
                "debt_scope": "controlled_public_marketable",
                "opening_route_stock_bil": 10.0,
                "route_face_issued_bil": 1.0,
                "route_face_redeemed_bil": 0.0,
                "route_stock_residual_or_indexation_bil": 0.0,
                "closing_route_stock_bil": 11.0,
                "closure_identity_error_bil": 0.0,
                "route_stock_basis": "tdc_principal_settlement_route",
            }
        ]
    )
    summary.to_csv(outputs / "tdcsim_period_tdc_summary.csv", index=False)
    components.to_csv(outputs / "tdcsim_period_tdc_components.csv", index=False)
    route.to_csv(outputs / "tdcsim_tdc_principal_route_stock_closure.csv", index=False)
    _write_surface(inputs / "tdcsim_yield_curve_surface.csv", shock_bp=shock_bp)
    (inputs / "tdcsim_primary_deficit_path.csv").write_text(primary_deficit_payload + "\n", encoding="utf-8")
    (inputs / "tdcsim_debt_stock_path.csv").write_text("same debt path\n", encoding="utf-8")
    (inputs / "tdcsim_fed_holdings_path.csv").write_text("same fed path\n", encoding="utf-8")
    write_json(inputs / "tdcsim_issuance_mix_assumptions.json", {"mode": "same"})
    scenario = {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": scenario_id,
        "overrides": (
            {
                "nominal_yield_curve": {
                    "mode": "full_surface_file",
                    "file": {"relative_path": "tdcsim_yield_curve_surface.csv", "sha256": "0" * 64},
                }
            }
            if shock
            else {}
        ),
    }
    write_json(run_dir / "scenario.json", scenario)
    manifest = {
        "run_id": run_id,
        "baseline": {
            "package_id": "pkg",
            "package_sha256": "1" * 64,
            "manifest_sha256": "2" * 64,
            "release_attestation_sha256": "3" * 64,
        },
        "scenario": {"scenario_id": scenario_id, "relative_path": "scenario.json"},
        "compiled_inputs_digest": f"digest-{run_id}",
        "compiled_inputs": [
            _manifest_artifact(run_dir, "tdcsim_yield_curve_surface.csv", inputs / "tdcsim_yield_curve_surface.csv"),
            _manifest_artifact(run_dir, "tdcsim_primary_deficit_path.csv", inputs / "tdcsim_primary_deficit_path.csv"),
            _manifest_artifact(run_dir, "tdcsim_debt_stock_path.csv", inputs / "tdcsim_debt_stock_path.csv"),
            _manifest_artifact(run_dir, "tdcsim_fed_holdings_path.csv", inputs / "tdcsim_fed_holdings_path.csv"),
            _manifest_artifact(run_dir, "tdcsim_issuance_mix_assumptions.json", inputs / "tdcsim_issuance_mix_assumptions.json"),
        ],
        "simulation": {"start_date": "2026-01-01", "end_date": "2027-01-01", "frequency": "daily"},
        "output_manifest": {
            "row_metadata": {
                "actuals_available_as_of": "2026-01-01",
                "source_vintage": "fixture_vintage",
            }
        },
        "outputs": [
            _manifest_artifact(run_dir, "tdcsim_period_tdc_summary.csv", outputs / "tdcsim_period_tdc_summary.csv"),
            _manifest_artifact(run_dir, "tdcsim_period_tdc_components.csv", outputs / "tdcsim_period_tdc_components.csv"),
            _manifest_artifact(
                run_dir,
                "tdcsim_tdc_principal_route_stock_closure.csv",
                outputs / "tdcsim_tdc_principal_route_stock_closure.csv",
            ),
        ],
    }
    write_json(run_dir / "tdcsim_cbo_run_manifest.json", manifest)
    return run_dir


def _make_fiscal_injection_run(run_dir: Path) -> None:
    inputs = run_dir / "compile" / "compiled" / "forecast_inputs"
    (inputs / "tdcsim_primary_deficit_path.csv").write_text("injection primary path\n", encoding="utf-8")
    (inputs / "tdcsim_debt_stock_path.csv").write_text("injection debt path\n", encoding="utf-8")
    (inputs / "tdcsim_fed_holdings_path.csv").write_text("injection fed path\n", encoding="utf-8")
    write_json(inputs / "tdcsim_issuance_mix_assumptions.json", {"mode": "injection"})
    scenario = {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": "shock_v1",
        "overrides": {
            "primary_deficit": {"mode": "absolute_path_file", "file": {"relative_path": "primary.csv", "sha256": "0" * 64}},
            "debt_target": {"mode": "absolute_path_file", "file": {"relative_path": "debt.csv", "sha256": "0" * 64}},
            "fed_holdings": {"mode": "absolute_path_file", "file": {"relative_path": "fed.csv", "sha256": "0" * 64}},
            "issuance_mix": {"mode": "replace_shares"},
        },
    }
    write_json(run_dir / "scenario.json", scenario)
    manifest_path = run_dir / "tdcsim_cbo_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["compiled_inputs"]:
        path = run_dir / item["relative_path"]
        item["sha256"] = sha256_file(path)
        item["bytes"] = path.stat().st_size
    write_json(manifest_path, manifest)


def _write_surface(path: Path, *, shock_bp: float) -> None:
    rows = []
    for date_value, delta_bp in (("2026-01-01", shock_bp), ("2027-01-01", 0.0)):
        for tenor in (0.25, 10.0):
            rate = 0.04 + delta_bp / 10000.0
            rows.append(
                {
                    "schema_version": "tdcsim_yield_curve_surface_v1",
                    "scenario_id": "fixture",
                    "curve_date": date_value,
                    "tenor_years": tenor,
                    "nominal_rate": rate * 100.0,
                    "nominal_rate_decimal": rate,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _manifest_artifact(run_dir: Path, logical_name: str, path: Path) -> dict:
    return {
        "logical_name": logical_name,
        "relative_path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": "text/csv",
    }


def _pair_spec(
    tmp_path: Path,
    baseline: Path,
    shock: Path,
    *,
    state_id: str = "fixture_state",
    state_fingerprint: str = "a" * 64,
    opening_tdc_stock: float = 10.0,
) -> dict:
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "pair_id": "fixture_pair",
        "scenario_state_set_id": "fixture_state_set",
        "state_id": state_id,
        "state_kind": "current_state",
        "state_period": "2026-01-01",
        "scenario_id": "fixture_scenario",
        "state_fingerprint_sha256": state_fingerprint,
        "state_component_inventory_sha256": "c" * 64,
        "baseline_state_fingerprint_sha256": state_fingerprint,
        "shock_state_fingerprint_sha256": state_fingerprint,
        "opening_state_date": "2026-01-01",
        "actuals_available_as_of": "2026-01-01",
        "source_vintage": "fixture_vintage",
        "horizon_start_date": "2026-01-01",
        "horizon_end_date": "2027-01-01",
        "horizon": "annual_h1_100bp_year",
        "horizon_index": "h1",
        "compiled_non_rate_inputs_digest": "fixture_non_rate_digest",
        "opening_tdc_stock_bil": opening_tdc_stock,
        "opening_route_stock_total_bil": opening_tdc_stock,
        "opening_route_stock_domestic_nonbank_bil": opening_tdc_stock,
        "baseline_run_dir": str(baseline),
        "shock_run_dir": str(shock),
        "baseline_scenario_id": "baseline_v1",
        "shock_scenario_id": "shock_v1",
        "object_id": OBJECT_ID,
        "shock_path_id": SHOCK_PATH_ID,
        "shock_bps_year": 100,
        "denominator_equivalence_key": DENOMINATOR_EQUIVALENCE_KEY,
        "require_same_baseline_hashes": True,
        "require_same_opening_state": True,
        "require_same_actuals_available_as_of": True,
        "require_same_simulation_dates": True,
        "require_same_period_index": True,
        "require_same_non_rate_compiled_inputs": True,
        "one_named_rate_shock_only": True,
        "demand_conversion_cases": [
            {
                "demand_conversion_case": "central",
                "beta": 0.5,
                "beta_assumption_id": "beta_fixture",
                "beta_source_status": "fixture",
                "chi": 0.4,
                "chi_assumption_id": "chi_fixture",
                "chi_source_status": "fixture",
            }
        ],
    }


def _write_current_split_check_against(baseline: Path, shock: Path) -> None:
    summary_columns = [
        "period_start",
        "period_end",
        "tdc_change_bil",
        "tdc_fiscal_flow_bil",
        "tdc_debt_service_bil",
        "tdc_auction_absorption_du_bil",
        "tdc_secondary_trades_bil",
        "tdc_other_bil",
        "overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil",
        "component_sum_bil",
        "component_sum_error_bil",
    ]
    summary_rows = [
        ["2026-01-01", "2027-01-01", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            "2026-01-01",
            "2027-01-01",
            4.235723994760747,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.235723994760747,
            4.235723994760747,
            0.0,
        ],
    ]
    for run_dir, row in ((baseline, summary_rows[0]), (shock, summary_rows[1])):
        pd.DataFrame([dict(zip(summary_columns, row))]).to_csv(
            run_dir / "outputs" / "tdcsim_period_tdc_summary.csv",
            index=False,
        )
        _refresh_output_artifact(run_dir, "tdcsim_period_tdc_summary.csv")

    rows = [
        ("auction_absorption_domestic_nonbank", "auction_absorption", "domestic_nonbank_deposit_funded", "issuance_proceeds", 6.070535912873311, True),
        ("auction_absorption_mmf", "auction_absorption", "mmf_cash_fund_route", "issuance_proceeds", 1.9628066118290235, True),
        ("auction_absorption_mmf_ru_plumbing_memo", "route_plumbing_memo", "mmf_cash_fund_route", "issuance_proceeds", 0.06070535912873311, False),
        ("bill_discount_interest_to_du_domestic_nonbank", "debt_service_interest", "domestic_nonbank_deposit_funded", "bill_discount", 5.224032322444145, False),
        ("bill_discount_interest_to_du_mmf", "debt_service_interest", "mmf_cash_fund_route", "bill_discount", 1.7155374486231132, True),
        ("fiscal_flow", "fiscal", "domestic_ultimate_net_primary_proxy", "primary_deficit_or_surplus", 0.0, True),
        ("fixed_coupon_interest_to_du_domestic_nonbank", "debt_service_interest", "domestic_nonbank_deposit_funded", "fixed_coupon", 15.868552, False),
        ("fixed_coupon_interest_to_du_mmf", "debt_service_interest", "mmf_cash_fund_route", "fixed_coupon", 1.4264137925026574, True),
        ("frn_interest_to_du_mmf", "debt_service_interest", "mmf_cash_fund_route", "frn_interest", 0.0, True),
        ("principal_to_du_domestic_nonbank", "debt_service_principal", "domestic_nonbank_deposit_funded", "principal", -5.224032322444145, True),
        ("principal_to_du_mmf", "debt_service_principal", "mmf_cash_fund_route", "principal", -1.7155374486230812, True),
        ("tips_coupon_interest_to_du_mmf", "debt_service_interest", "mmf_cash_fund_route", "tips_coupon", 0.0, True),
    ]
    baseline_components = [_component_row(key, family, subsector, payment_type, 0.0, included) for key, family, subsector, payment_type, _, included in rows]
    shock_components = [_component_row(key, family, subsector, payment_type, amount, included) for key, family, subsector, payment_type, amount, included in rows]
    for run_dir, component_rows in ((baseline, baseline_components), (shock, shock_components)):
        pd.DataFrame(component_rows).to_csv(run_dir / "outputs" / "tdcsim_period_tdc_components.csv", index=False)
        _refresh_output_artifact(run_dir, "tdcsim_period_tdc_components.csv")


def _component_row(
    component_key: str,
    component_family: str,
    holder_subsector: str,
    payment_type: str,
    amount: float,
    included: bool,
) -> dict:
    return {
        "period_start": "2026-01-01",
        "period_end": "2027-01-01",
        "component_key": component_key,
        "component_family": component_family,
        "holder_sector": "Private",
        "holder_subsector": holder_subsector,
        "instrument_type": "all",
        "payment_type": payment_type,
        "accounting_basis": "fixture_signed_basis",
        "amount_bil": amount,
        "enters_direct_interest_support": component_family == "debt_service_interest",
        "enters_tdc_deposit_support_default": included,
        "tdc_amount_basis": "post_mmf_route_pass_through_pre_ratewall_beta_chi",
        "overlap_policy": "fixture_overlap_policy",
    }


def _refresh_output_artifact(run_dir: Path, filename: str) -> None:
    manifest_path = run_dir / "tdcsim_cbo_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = run_dir / "outputs" / filename
    for item in manifest["outputs"]:
        if item["logical_name"] == filename:
            item["sha256"] = sha256_file(path)
            item["bytes"] = path.stat().st_size
    write_json(manifest_path, manifest)


def _split_tdc_rows(
    run_dir: Path,
    *,
    tdc_values: list[float],
    overlap_values: list[float],
) -> None:
    summary_path = run_dir / "outputs" / "tdcsim_period_tdc_summary.csv"
    template = pd.read_csv(summary_path).iloc[0].to_dict()
    rows = []
    for idx, (tdc_value, overlap_value) in enumerate(zip(tdc_values, overlap_values)):
        row = dict(template)
        row["period_start"] = f"2026-0{idx + 1}-01"
        row["period_end"] = f"2026-0{idx + 2}-01" if idx == 0 else "2027-01-01"
        row["tdc_change_bil"] = tdc_value
        row["overlap_cashflow_bil"] = overlap_value
        row["tdc_change_ex_overlap_bil"] = tdc_value - overlap_value
        row["component_sum_bil"] = tdc_value
        rows.append(row)
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    components_path = run_dir / "outputs" / "tdcsim_period_tdc_components.csv"
    component_template = pd.read_csv(components_path).iloc[0].to_dict()
    component_rows = []
    for idx, (tdc_value, overlap_value) in enumerate(zip(tdc_values, overlap_values)):
        row = dict(component_template)
        row["period_start"] = f"2026-0{idx + 1}-01"
        row["period_end"] = f"2026-0{idx + 2}-01" if idx == 0 else "2027-01-01"
        row["amount_bil"] = tdc_value - overlap_value
        component_rows.append(row)
    pd.DataFrame(component_rows).to_csv(components_path, index=False)
    _refresh_output_artifact(run_dir, "tdcsim_period_tdc_summary.csv")
    _refresh_output_artifact(run_dir, "tdcsim_period_tdc_components.csv")


def _append_component(
    run_dir: Path,
    *,
    component_key: str,
    amount: float,
    enters_tdc_default: bool,
) -> None:
    path = run_dir / "outputs" / "tdcsim_period_tdc_components.csv"
    frame = pd.read_csv(path)
    row = frame.iloc[0].to_dict()
    row["component_key"] = component_key
    row["amount_bil"] = amount
    row["enters_tdc_deposit_support_default"] = enters_tdc_default
    row["enters_direct_interest_support"] = not enters_tdc_default
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    frame.to_csv(path, index=False)
    _refresh_output_artifact(run_dir, "tdcsim_period_tdc_components.csv")


def _refresh_pair_manifest_file(pair_dir: Path, filename: str) -> None:
    manifest_path = pair_dir / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = pair_dir / filename
    manifest["files"][filename]["sha256"] = sha256_file(path)
    manifest["files"][filename]["bytes"] = path.stat().st_size
    write_json(manifest_path, manifest)
