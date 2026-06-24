import csv
import json
import zipfile
from pathlib import Path

import pytest

from tdcsim_cbo import CboBaselinePackage, CboScenarioCompiler, CboScenarioSpec
from tdcsim_cbo._json import sha256_file, write_json
from tdcsim_cbo.compiler import CompilerError, ISSUANCE_MIX_FILE, digest_input_tree
from test_tdcsim_cbo_baseline import RELEASE_SHA, VERIFIER_SHA


def test_noop_compile_preserves_input_digest_and_does_not_mark_rows(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    spec = CboScenarioSpec.from_mapping(_scenario_mapping(baseline))

    compiled = CboScenarioCompiler().compile(baseline, spec, tmp_path / "work")

    assert compiled.changed_inputs == ()
    assert compiled.compiled_inputs_digest == compiled.baseline_forecast_inputs_digest
    assert compiled.compiled_inputs_digest == digest_input_tree(compiled.baseline_dir / "forecast_inputs")
    header = _csv_header(compiled.forecast_inputs_dir / "tdcsim_yield_curve_surface.csv")
    assert "scenario_transform" not in header
    assert sha256_file(baseline.package_path) == baseline.package_sha256


def test_same_baseline_and_scenario_compile_digest_repeats_across_work_dirs(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["coupling"]["frn_benchmark"] = "derive_from_scenario_nominal_curve"
    scenario["overrides"] = {
        "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 100},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 10},
    }
    spec = CboScenarioSpec.from_mapping(scenario)

    first = CboScenarioCompiler().compile(baseline, spec, tmp_path / "work-a")
    second = CboScenarioCompiler().compile(baseline, spec, tmp_path / "work-b")

    assert first.compiled_inputs_digest == second.compiled_inputs_digest
    assert first.manifest["input_hashes"] == second.manifest["input_hashes"]


def test_compiler_dag_is_independent_of_override_key_order(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    first = _scenario_mapping(baseline)
    first["coupling"]["frn_benchmark"] = "derive_from_scenario_nominal_curve"
    first["overrides"] = {
        "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 100},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 10},
        "operating_cash": {"mode": "scale_baseline", "scale": 2.0},
        "cash_reconciliation": {"mode": "track_operating_cash_target"},
    }
    second = _scenario_mapping(baseline)
    second["coupling"]["frn_benchmark"] = "derive_from_scenario_nominal_curve"
    second["overrides"] = {
        "cash_reconciliation": {"mode": "track_operating_cash_target"},
        "operating_cash": {"mode": "scale_baseline", "scale": 2.0},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 10},
        "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 100},
    }

    compiled_first = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(first), tmp_path / "work-first")
    compiled_second = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(second), tmp_path / "work-second")

    assert compiled_first.compiled_inputs_digest == compiled_second.compiled_inputs_digest


def test_frn_linked_mode_uses_compiled_nominal_curve_not_baseline(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["coupling"]["frn_benchmark"] = "derive_from_scenario_nominal_curve"
    scenario["overrides"] = {
        "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 100},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 10},
    }

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_frn_rate_path.csv")
    assert float(rows[0]["benchmark_rate_decimal"]) == pytest.approx(0.041)


def test_scenario_cpi_recomputes_tips_real_yield_without_explicit_tips_override(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "inflation_cpi": {"mode": "annualized_inflation_shift_bp", "shock_bp": 100, "terminal_rule": "carry_last_scenario_growth"},
    }

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_tips_real_yield_path.csv")
    assert "tdcsim_tips_real_yield_path.csv" in compiled.changed_inputs
    assert float(rows[0]["expected_inflation_decimal"]) == pytest.approx(0.03)
    assert float(rows[0]["real_yield_decimal"]) == pytest.approx(0.0)


def test_operating_cash_constant_real_uses_compiled_scenario_cpi(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["coupling"]["operating_cash_inflation"] = "scenario_cpi"
    scenario["overrides"] = {
        "inflation_cpi": {"mode": "cpi_level_scale", "scale": 1.12},
        "operating_cash": {"mode": "constant_real"},
    }

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_operating_cash_path.csv")
    assert float(rows[1]["inflation_index_level"]) == pytest.approx(113.12)
    assert float(rows[1]["operating_cash_target_bil"]) == pytest.approx(100.0)
    assert rows[1]["construction_mode"] == "scenario_constant_real"


def test_cash_residual_tracking_uses_compiled_operating_cash(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "operating_cash": {"mode": "scale_baseline", "scale": 2.0},
        "cash_reconciliation": {"mode": "track_operating_cash_target"},
    }

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_cash_reconciliation_residual.csv")
    assert float(rows[0]["cash_reconciliation_residual_bil"]) == pytest.approx(0.0)
    assert float(rows[1]["cash_reconciliation_residual_bil"]) == pytest.approx(20.0)
    assert rows[1]["affects_issuance_size"] == "False"


def test_primary_deficit_fy_anchors_write_period_flows_not_annual_value_per_row(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "primary_deficit": {
            "mode": "fy_endpoint_anchors",
            "anchors": [{"fiscal_year": 2027, "value_bil": 1000.0}, {"fiscal_year": 2028, "value_bil": 2000.0}],
        }
    }

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_primary_deficit_path.csv")
    by_year: dict[int, float] = {}
    for row in rows:
        by_year[int(row["source_fiscal_year"])] = by_year.get(int(row["source_fiscal_year"]), 0.0) + float(row["primary_deficit_bil"])
    assert by_year[2027] == pytest.approx(1000.0)
    assert by_year[2028] == pytest.approx(2000.0)
    assert float(rows[0]["annual_or_remaining_primary_deficit_bil"]) == pytest.approx(1000.0)


def test_compiler_rejects_frn_file_mode_with_linked_coupling(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "frn.csv"
    _write_csv(
        replacement,
        [
            {"period_start": "2027-01-01", "period_end": "2027-01-02", "benchmark_rate_decimal": 0.05},
            {"period_start": "2027-01-02", "period_end": "2027-01-03", "benchmark_rate_decimal": 0.06},
        ],
    )
    scenario = _scenario_mapping(baseline)
    scenario["coupling"]["frn_benchmark"] = "derive_from_scenario_nominal_curve"
    scenario["overrides"] = {
        "frn_benchmark": {
            "mode": "absolute_path_file",
            "file": {"relative_path": "frn.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)
    spec = CboScenarioSpec.from_file(scenario_path)

    with pytest.raises(CompilerError, match="independent FRN"):
        CboScenarioCompiler().compile(baseline, spec, tmp_path / "work")


def test_compiler_rejects_tips_file_mode_with_recompute_coupling(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "tips.csv"
    _write_csv(
        replacement,
        [
            {"curve_date": "2027-01-01", "tenor_years": 0.25, "real_yield_decimal": 0.01},
            {"curve_date": "2027-01-02", "tenor_years": 0.25, "real_yield_decimal": 0.02},
        ],
    )
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "tips_real_yield": {
            "mode": "absolute_path_file",
            "file": {"relative_path": "tips.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)
    spec = CboScenarioSpec.from_file(scenario_path)

    with pytest.raises(CompilerError, match="independent TIPS"):
        CboScenarioCompiler().compile(baseline, spec, tmp_path / "work")


def test_compiler_rejects_file_reference_on_non_file_mode(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "nominal_yield_curve": {
            "mode": "parallel_bp",
            "shock_bp": 10,
            "file": {"relative_path": "curve.csv", "sha256": "0" * 64, "media_type": "text/csv"},
        }
    }
    spec = CboScenarioSpec.from_mapping(scenario)

    with pytest.raises(CompilerError, match="file reference"):
        CboScenarioCompiler().compile(baseline, spec, tmp_path / "work")


def test_file_backed_override_sha_mismatch_rejected(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "nominal_yield_curve": {
            "mode": "full_surface_file",
            "file": {"relative_path": "curve.csv", "sha256": "0" * 64, "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    _write_csv(tmp_path / "curve.csv", [{"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.05}])
    write_json(scenario_path, scenario)
    spec = CboScenarioSpec.from_file(scenario_path)

    with pytest.raises(CompilerError, match="SHA-256 mismatch"):
        CboScenarioCompiler().compile(baseline, spec, tmp_path / "work")


def test_file_backed_override_missing_horizon_coverage_rejected(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "curve.csv"
    _write_csv(replacement, [{"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.05}])
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "nominal_yield_curve": {
            "mode": "full_surface_file",
            "file": {"relative_path": "curve.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)

    with pytest.raises(CompilerError, match="row count must match baseline coverage"):
        CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_file(scenario_path), tmp_path / "work")


def test_file_backed_override_duplicate_keys_rejected_even_when_row_count_matches(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "curve.csv"
    _write_csv(
        replacement,
        [
            {"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.05},
            {"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.06},
        ],
    )
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "nominal_yield_curve": {
            "mode": "full_surface_file",
            "file": {"relative_path": "curve.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)

    with pytest.raises(CompilerError, match="duplicate keys"):
        CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_file(scenario_path), tmp_path / "work")


def test_file_backed_fiscal_replacements_overwrite_claim_labels(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "primary.csv"
    _write_csv(
        replacement,
        [
            {
                "source_fiscal_year": 2027,
                "primary_deficit_bil": 12.0,
                "source_role": "official_cbo_source",
                "runtime_role": "cash_plug",
                "claim_boundary": "bad",
            },
            {
                "source_fiscal_year": 2028,
                "primary_deficit_bil": 24.0,
                "source_role": "official_cbo_source",
                "runtime_role": "cash_plug",
                "claim_boundary": "bad",
            },
        ],
    )
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "primary_deficit": {
            "mode": "absolute_path_file",
            "file": {"relative_path": "primary.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_file(scenario_path), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_primary_deficit_path.csv")
    assert {row["source_role"] for row in rows} == {"scenario_assumption"}
    assert {row["runtime_role"] for row in rows} == {"hard_target"}
    assert {row["claim_boundary"] for row in rows} == {"primary_deficit_scenario_transform_no_plug"}


def test_file_backed_operating_cash_replacements_overwrite_runtime_role(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    replacement = tmp_path / "cash.csv"
    _write_csv(
        replacement,
        [
            {"period_end": "2027-01-01", "operating_cash_target_bil": 80.0, "source_role": "source", "runtime_role": "plug", "claim_boundary": "bad"},
            {"period_end": "2027-01-02", "operating_cash_target_bil": 90.0, "source_role": "source", "runtime_role": "plug", "claim_boundary": "bad"},
        ],
    )
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "operating_cash": {
            "mode": "aggregate_path_file",
            "file": {"relative_path": "cash.csv", "sha256": sha256_file(replacement), "media_type": "text/csv"},
        }
    }
    scenario_path = tmp_path / "scenario.json"
    write_json(scenario_path, scenario)

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_file(scenario_path), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_operating_cash_path.csv")
    assert {row["source_role"] for row in rows} == {"scenario_assumption"}
    assert {row["runtime_role"] for row in rows} == {"hard_target"}
    assert {row["claim_boundary"] for row in rows} == {"operating_cash_proxy_not_debt_target_or_issuance_supply"}


def test_holder_preferences_reject_cb_auction_share_when_baseline_fed_target_active(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {"holder_preferences": {"mode": "static_shares", "rows": _holder_preference_rows(cb_share=0.10)}}

    with pytest.raises(ValueError, match="CB auction share"):
        CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")


def test_holder_preferences_overwrite_baseline_claim_labels(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {"holder_preferences": {"mode": "static_shares", "rows": _holder_preference_rows(cb_share=0.0)}}

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    rows = _read_csv(compiled.forecast_inputs_dir / "tdcsim_holder_profile_assumptions.csv")
    assert {row["source_role"] for row in rows} == {"scenario_assumption"}
    assert {row["runtime_role"] for row in rows} == {"memo_only"}
    assert {row["claim_boundary"] for row in rows} == {"holder preference profile not exact holder ownership"}
    assert {row["scenario_transform"] for row in rows} == {"static_shares"}


def test_issuance_mix_override_materializes_hashed_compiled_artifact(tmp_path: Path) -> None:
    baseline = _compiler_baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {"issuance_mix": _issuance_mix_override()}

    compiled = CboScenarioCompiler().compile(baseline, CboScenarioSpec.from_mapping(scenario), tmp_path / "work")

    artifact = compiled.forecast_inputs_dir / ISSUANCE_MIX_FILE
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact.exists()
    assert ISSUANCE_MIX_FILE in compiled.changed_inputs
    assert any(item["path"] == ISSUANCE_MIX_FILE for item in compiled.manifest["input_hashes"])
    assert payload["mode"] == "replace_shares"
    assert payload["security_shares"]["tips"] == pytest.approx(0.08)


def test_compiler_rejects_directory_baseline_without_package_digest(tmp_path: Path) -> None:
    package, attestation = _write_compiler_package(tmp_path)
    baseline_zip = CboBaselinePackage.open(package, attestation_path=attestation)
    directory = baseline_zip.materialize(tmp_path / "directory-baseline")
    baseline_dir = CboBaselinePackage.open(directory, attestation_path=attestation, verify=False)
    scenario = _scenario_mapping(baseline_zip)
    spec = CboScenarioSpec.from_mapping(scenario)

    with pytest.raises(CompilerError, match="zip baseline"):
        CboScenarioCompiler().compile(baseline_dir, spec, tmp_path / "work")


def _compiler_baseline(tmp_path: Path) -> CboBaselinePackage:
    package, attestation = _write_compiler_package(tmp_path)
    return CboBaselinePackage.open(package, attestation_path=attestation)


def _scenario_mapping(baseline: CboBaselinePackage) -> dict:
    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": "compiler_noop_v1",
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "provenance": {"kind": "user_stress_assumption", "label": "Compiler test"},
        "coupling": {
            "frn_benchmark": "independent_explicit_path",
            "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
            "operating_cash_inflation": "baseline_cpi",
            "primary_deficit_to_debt_target": "independent_no_plug",
        },
        "overrides": {},
        "output": {"profile": "compact", "compression": "gzip"},
    }


def _holder_preference_rows(*, cb_share: float = 0.0) -> list[dict]:
    private_share = 1.0 - cb_share
    return [
        {
            "security_type": security_type,
            "shares": {
                "Banks": 0.0,
                "CB": cb_share,
                "Foreign": 0.0,
                "Private": private_share,
                "TrustFunds": 0.0,
                "FedInternal": 0.0,
            },
        }
        for security_type in ("bills", "notes", "bonds", "tips", "frn")
    ]


def _issuance_mix_override() -> dict:
    return {
        "mode": "replace_shares",
        "tips_share": 0.08,
        "frn_share": 0.04,
        "fixed_remainder_shares": {"bills": 0.30, "notes": 0.50, "bonds": 0.20},
        "maturity_distributions": {
            "bills": [{"maturity_years": 0.5, "share": 1.0}],
            "notes": [{"maturity_years": 5.0, "share": 1.0}],
            "bonds": [{"maturity_years": 20.0, "share": 1.0}],
            "tips": [{"maturity_years": 10.0, "share": 1.0}],
            "frn": [{"maturity_years": 2.0, "share": 1.0}],
        },
        "negative_issuance_action": "retire_shortest_public_marketable",
    }


def _write_compiler_package(tmp_path: Path) -> tuple[Path, Path]:
    package_dir = tmp_path / "compiler_pkg"
    inputs = package_dir / "forecast_inputs"
    inputs.mkdir(parents=True)
    (package_dir / "requirements.lock.txt").write_text("pandas==0\n", encoding="utf-8")
    lock_sha = sha256_file(package_dir / "requirements.lock.txt")

    _write_csv(
        inputs / "tdcsim_yield_curve_surface.csv",
        [
            {"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.03, "nominal_rate": 3.0},
            {"curve_date": "2027-01-02", "tenor_years": 0.25, "nominal_rate_decimal": 0.031, "nominal_rate": 3.1},
        ],
    )
    _write_csv(
        inputs / "tdcsim_frn_rate_path.csv",
        [
            {"period_start": "2027-01-01", "period_end": "2027-01-02", "benchmark_rate_decimal": 0.02, "auction_high_rate_decimal": 0.02, "money_market_yield_decimal": 0.02, "affects_issuance_size": True},
            {"period_start": "2027-01-02", "period_end": "2027-01-03", "benchmark_rate_decimal": 0.021, "auction_high_rate_decimal": 0.021, "money_market_yield_decimal": 0.021, "affects_issuance_size": True},
        ],
    )
    _write_csv(
        inputs / "tdcsim_tips_cpi_path.csv",
        [
            {"month": "2027-01-01", "cbo_cpi_u_index": 100.0, "tips_cpi_u_index": 101.0, "terminal_annualized_cpi_growth_decimal": 0.02},
            {"month": "2027-02-01", "cbo_cpi_u_index": 101.0, "tips_cpi_u_index": 102.0, "terminal_annualized_cpi_growth_decimal": 0.02},
        ],
    )
    _write_csv(
        inputs / "tdcsim_tips_real_yield_path.csv",
        [
            {"curve_date": "2027-01-01", "tenor_years": 0.25, "nominal_rate_decimal": 0.03, "expected_inflation_decimal": 0.01, "real_yield_decimal": 0.02},
            {"curve_date": "2027-01-02", "tenor_years": 0.25, "nominal_rate_decimal": 0.031, "expected_inflation_decimal": 0.01, "real_yield_decimal": 0.021},
        ],
    )
    _write_csv(
        inputs / "tdcsim_operating_cash_path.csv",
        [
            {"period_end": "2027-01-01", "operating_cash_target_bil": 100.0, "tga_target_bil": 90.0, "ttl_target_bil": 5.0, "other_operating_cash_target_bil": 5.0, "inflation_index_level": 100.0},
            {"period_end": "2027-01-02", "operating_cash_target_bil": 110.0, "tga_target_bil": 99.0, "ttl_target_bil": 6.0, "other_operating_cash_target_bil": 5.0, "inflation_index_level": 101.0},
        ],
    )
    _write_csv(
        inputs / "tdcsim_cash_reconciliation_residual.csv",
        [
            {"period_end": "2027-01-01", "cash_reconciliation_residual_bil": 0.0, "affects_issuance_size": True},
            {"period_end": "2027-01-02", "cash_reconciliation_residual_bil": 0.0, "affects_issuance_size": True},
        ],
    )
    _write_csv(
        inputs / "tdcsim_primary_deficit_path.csv",
        [
            {"source_fiscal_year": 2027, "primary_deficit_bil": 10.0, "annual_or_remaining_primary_deficit_bil": 100.0},
            {"source_fiscal_year": 2028, "primary_deficit_bil": 20.0, "annual_or_remaining_primary_deficit_bil": 200.0},
        ],
    )
    _write_csv(
        inputs / "tdcsim_debt_stock_path.csv",
        [
            {"period_end": "2027-01-01", "source_fiscal_year": 2027, "cbo_federal_debt_held_public_target_bil": 1000.0, "marketable_treasury_public_target_bil": 900.0},
            {"period_end": "2027-01-02", "source_fiscal_year": 2028, "cbo_federal_debt_held_public_target_bil": 1100.0, "marketable_treasury_public_target_bil": 1000.0},
        ],
    )
    _write_csv(
        inputs / "tdcsim_fed_holdings_path.csv",
        [
            {"period_end": "2027-01-01", "source_fiscal_year": 2027, "holder_type": "CB", "cbo_fed_holdings_target_bil": 100.0},
            {"period_end": "2027-01-02", "source_fiscal_year": 2028, "holder_type": "CB", "cbo_fed_holdings_target_bil": 100.0},
        ],
    )
    _write_csv(
        inputs / "tdcsim_fiscal_incidence_policy.csv",
        [{"policy_id": "central", "du_share": 1.0, "ru_share": 0.0, "foreign_share": 0.0, "other_share": 0.0}],
    )
    _write_csv(
        inputs / "tdcsim_holder_profile_assumptions.csv",
        [
            {"holder_type": "Banks", "bills_pct": 0.5, "notes_pct": 0.5, "bonds_pct": 0.5, "tips_pct": 0.5, "frn_pct": 0.5},
            {"holder_type": "Private", "bills_pct": 0.5, "notes_pct": 0.5, "bonds_pct": 0.5, "tips_pct": 0.5, "frn_pct": 0.5},
            {"holder_type": "CB", "bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0, "frn_pct": 0.0},
        ],
    )

    manifest = {
        "schema_version": "tdcsim_cbo_forecast_smoke_manifest_v1",
        "scenario_id": "compiler_fixture",
        "trust_anchor": {
            "schema_version": "tdcsim_cbo_package_trust_anchor_v1",
            "code_revision": RELEASE_SHA,
            "dirty_state": False,
            "requirements_lock_sha256": lock_sha,
            "verifier_sha256": VERIFIER_SHA,
        },
    }
    write_json(package_dir / "manifest.json", manifest)
    write_json(inputs / "source_contract_smoke.json", manifest)

    package = tmp_path / "compiler_pkg.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())
    attestation = tmp_path / "compiler_attestation.json"
    write_json(
        attestation,
        {
            "schema_version": "tdcsim_cbo_external_release_attestation_v1",
            "attestation_created_at_utc": "2026-06-24T00:00:00+00:00",
            "release_commit_sha": RELEASE_SHA,
            "release_commit_short": "639c033",
            "branch_state": "detached_release_validation_worktree",
            "worktree_path": "/tmp/tdcsim-cbo-release-639c033",
            "git_status_short": "",
            "dirty_state": False,
            "baseline_package_dir": "output/cbo_forecast_release_bound",
            "baseline_package_zip": "output/cbo_forecast_release_bound_package.zip",
            "baseline_package_zip_sha256": sha256_file(package),
            "baseline_manifest_sha256": sha256_file(package_dir / "manifest.json"),
            "source_contract_sha256": sha256_file(inputs / "source_contract_smoke.json"),
            "verifier_sha256": VERIFIER_SHA,
            "requirements_lock_sha256": lock_sha,
            "manifest_trust_anchor": manifest["trust_anchor"],
            "artifact_hashes_count": 0,
            "required_files_count": 14,
            "scenario_id": "compiler_fixture",
            "run_summaries": [],
            "source_package_used": "test-fixture",
            "validation_grade": "release_reproducible_local_clean_commit",
            "claim_boundary": {"package_role": "test"},
            "unsupported_components": {},
            "commands": [{"name": "unit_fixture", "command": "test", "cwd": str(tmp_path), "exit_code": 0, "stdout_log": "fixture.txt", "stdout_sha256": "0" * 64, "result": "pass"}],
            "attestation_log_hashes": {},
        },
    )
    return package, attestation


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
