from pathlib import Path

import pytest

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec
from tdcsim_cbo._json import write_json
from test_tdcsim_cbo_baseline import _write_release_package


def test_scenario_spec_canonical_hash_is_stable(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    first = _scenario_mapping(baseline)
    second = {
        "output": first["output"],
        "overrides": first["overrides"],
        "coupling": first["coupling"],
        "provenance": first["provenance"],
        "baseline": first["baseline"],
        "scenario_id": first["scenario_id"],
        "schema_version": first["schema_version"],
    }

    first_spec = CboScenarioSpec.from_mapping(first)
    second_spec = CboScenarioSpec.from_mapping(second)

    assert first_spec.canonical_json() == second_spec.canonical_json()
    assert first_spec.canonical_sha256() == second_spec.canonical_sha256()
    first_spec.assert_baseline_matches(baseline)


def test_scenario_spec_rejects_unknown_fields(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["surprise"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_advertised_but_unsupported_fields(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["nominal_yield_curve"] = {
        "mode": "parallel_bp",
        "shock_bp": 10,
        "rate_unit": "decimal",
    }

    with pytest.raises(ValueError, match="unknown fields"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_mode_missing_required_control(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["nominal_yield_curve"] = {"mode": "parallel_bp"}

    with pytest.raises(ValueError, match="missing required fields"):
        CboScenarioSpec.from_mapping(scenario)


@pytest.mark.parametrize(
    ("name", "override", "field"),
    [
        ("nominal_yield_curve", {"mode": "parallel_bp", "shock_bp": 10, "interpolation": "pchip_log_tenor"}, "interpolation"),
        ("nominal_yield_curve", {"mode": "parallel_bp", "shock_bp": 10, "shocks": [{"tenor_years": 10, "shock_bp": 1}]}, "shocks"),
        ("frn_benchmark", {"mode": "linked_to_nominal_curve", "spread_bp": 5, "shock_bp": 10}, "shock_bp"),
        ("inflation_cpi", {"mode": "cpi_level_scale", "scale": 1.02, "terminal_rule": "carry_last_scenario_growth"}, "terminal_rule"),
        ("tips_real_yield", {"mode": "parallel_bp", "shock_bp": 10, "additional_parallel_bp": 5}, "additional_parallel_bp"),
        ("operating_cash", {"mode": "scale_baseline", "scale": 1.1, "file": {"relative_path": "cash.csv", "sha256": "0" * 64}}, "file"),
        ("cash_reconciliation", {"mode": "zero", "file": {"relative_path": "cash.csv", "sha256": "0" * 64}}, "file"),
        ("fed_holdings", {"mode": "scale_path", "scale": 1.1, "additive_bil": 5}, "additive_bil"),
        ("debt_target", {"mode": "scale_path", "scale": 1.1, "anchors": [{"fiscal_year": 2027, "value_bil": 1}]}, "anchors"),
    ],
)
def test_scenario_spec_rejects_mode_inapplicable_controls(tmp_path: Path, name: str, override: dict, field: str) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"][name] = override

    with pytest.raises(ValueError, match=rf"mode-inapplicable fields.*{field}"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_primary_deficit_scale_path_ignored_controls(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["primary_deficit"] = {
        "mode": "scale_path",
        "scale": 1.0,
        "additive_bil": 999999,
        "freeze_pre_start_actuals": True,
        "anchors": [{"fiscal_year": 2027, "value_bil": 1}],
    }

    with pytest.raises(ValueError, match="mode-inapplicable fields.*additive_bil.*anchors.*freeze_pre_start_actuals"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_allows_mode_specific_optional_controls(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"] = {
        "nominal_yield_curve": {"mode": "key_rate_bp", "shocks": [{"tenor_years": 10, "shock_bp": 1}], "interpolation": "pchip_log_tenor"},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 5},
        "inflation_cpi": {"mode": "annualized_inflation_shift_bp", "shock_bp": 25, "terminal_rule": "carry_last_scenario_growth"},
        "tips_real_yield": {"mode": "linked_recompute", "additional_parallel_bp": 2},
        "primary_deficit": {
            "mode": "fy_endpoint_anchors",
            "anchors": [{"fiscal_year": 2027, "value_bil": 1}],
            "freeze_pre_start_actuals": True,
        },
    }

    CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_holder_temporal_or_file_surface(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["holder_preferences"] = {
        "mode": "static_shares",
        "rows": [
            {
                "effective_quarter": "2027Q1",
                "security_type": "bills",
                "shares": {"Banks": 0.0, "CB": 0.0, "Foreign": 0.0, "Private": 1.0, "TrustFunds": 0.0, "FedInternal": 0.0},
            }
        ],
    }

    with pytest.raises(ValueError, match="unknown fields"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_allows_dated_holder_preference_rows(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["holder_preferences"] = {
        "mode": "dated_static_shares",
        "rows": [
            {
                "effective_date": "2030-01-15",
                "security_type": "bills",
                "shares": {"Banks": 0.1, "CB": 0.0, "Foreign": 0.2, "Private": 0.7, "TrustFunds": 0.0, "FedInternal": 0.0},
            }
        ],
    }

    CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_effective_date_on_static_holder_rows(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["holder_preferences"] = {
        "mode": "static_shares",
        "rows": [
            {
                "effective_date": "2030-01-15",
                "security_type": "bills",
                "shares": {"Banks": 0.1, "CB": 0.0, "Foreign": 0.2, "Private": 0.7, "TrustFunds": 0.0, "FedInternal": 0.0},
            }
        ],
    }

    with pytest.raises(ValueError, match="mode-inapplicable fields.*effective_date"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_comparator_file_and_assertions_surface(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["assertions"] = [{"metric": "x", "operator": "eq", "value": 1}]
    scenario["overrides"]["net_interest_comparator"] = {
        "mode": "comparison_path_file",
        "role": "diagnostic_nonbinding",
        "file": {"relative_path": "comparison.csv", "sha256": "0" * 64, "media_type": "text/csv"},
    }

    with pytest.raises(ValueError, match="unknown fields|not in enum"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_compatible_baseline_mode(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["baseline"]["allow_compatible_baseline"] = True

    with pytest.raises(ValueError, match="exact baseline hashes"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_spec_rejects_unsafe_file_references(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["overrides"]["nominal_yield_curve"] = {
        "mode": "full_surface_file",
        "file": {
            "relative_path": "../escape.csv",
            "sha256": "0" * 64,
            "media_type": "text/csv",
        },
    }

    with pytest.raises(ValueError, match="pattern|required pattern|package-relative"):
        CboScenarioSpec.from_mapping(scenario)


def test_scenario_baseline_hash_pin_must_match(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    scenario["baseline"]["manifest_sha256"] = "0" * 64
    spec = CboScenarioSpec.from_mapping(scenario)

    with pytest.raises(ValueError, match="manifest_sha256"):
        spec.assert_baseline_matches(baseline)


def test_scenario_yaml_and_json_authoring_are_canonical_equivalent(tmp_path: Path) -> None:
    baseline = _baseline(tmp_path)
    scenario = _scenario_mapping(baseline)
    json_path = tmp_path / "scenario.json"
    yaml_path = tmp_path / "scenario.yaml"
    write_json(json_path, scenario)
    yaml_path.write_text(
        "\n".join(
            [
                "schema_version: tdcsim_cbo_scenario_v1",
                "scenario_id: noop_baseline_v1",
                "baseline:",
                f"  package_id: {baseline.package_id}",
                f"  package_sha256: {baseline.package_sha256}",
                f"  manifest_sha256: {baseline.manifest_sha256}",
                f"  release_attestation_sha256: {baseline.attestation.sha256}",
                "provenance:",
                "  kind: user_stress_assumption",
                "  label: No-op baseline scenario",
                "coupling:",
                "  frn_benchmark: derive_from_scenario_nominal_curve",
                "  tips_real_yield: recompute_from_nominal_and_scenario_inflation",
                "  operating_cash_inflation: baseline_cpi",
                "  primary_deficit_to_debt_target: independent_no_plug",
                "overrides: {}",
                "output:",
                "  profile: compact",
                "  compression: gzip",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    json_spec = CboScenarioSpec.from_file(json_path)
    yaml_spec = CboScenarioSpec.from_file(yaml_path)

    assert json_spec.canonical_sha256() == yaml_spec.canonical_sha256()


def _baseline(tmp_path: Path) -> CboBaselinePackage:
    package, attestation = _write_release_package(tmp_path)
    return CboBaselinePackage.open(package, attestation_path=attestation)


def _scenario_mapping(baseline: CboBaselinePackage) -> dict:
    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": "noop_baseline_v1",
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "provenance": {
            "kind": "user_stress_assumption",
            "label": "No-op baseline scenario",
        },
        "coupling": {
            "frn_benchmark": "derive_from_scenario_nominal_curve",
            "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
            "operating_cash_inflation": "baseline_cpi",
            "primary_deficit_to_debt_target": "independent_no_plug",
        },
        "overrides": {},
        "output": {
            "profile": "compact",
            "compression": "gzip",
        },
    }
