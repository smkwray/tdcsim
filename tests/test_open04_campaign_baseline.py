from __future__ import annotations

from copy import deepcopy
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluated_nominal_curve import (
    NOMINAL_EVALUATED_SHOCK_FILE,
    OPEN04_FIXED_COUPLING,
    OPEN04_OUTPUT_CONTRACT,
)
import tdcsim_cbo.open04_campaign as open04_campaign_module
from tdcsim_cbo import CboBaselinePackage, CboScenarioCompiler, CboScenarioSpec
from tdcsim_cbo._json import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)
from tdcsim_cbo.compiler import INPUT_FILES, ISSUANCE_MIX_FILE
from tdcsim_cbo.open04_campaign import (
    OPEN04_BASELINE_IDENTITY,
    OPEN04_CAMPAIGN_CONTRACT_ID,
    OPEN04_CAMPAIGN_ROLES,
    OPEN04_FUNDING_CLOSURE_MODE,
    OPEN04_ROLE_TO_SCENARIO_ID,
    Open04CampaignError,
    Open04CampaignMarker,
    parse_open04_campaign_marker,
    requires_open04_strict_execution,
    validate_open04_scenario_contract,
)
from tdcsim_cbo.verifier import verify_compiled_scenario
from test_tdcsim_cbo_compiler import (
    _open04_scenario_mapping,
    _read_csv,
    _write_compiler_package,
    _write_csv,
)


_START = "2026-06-21"
_END = "2036-09-30"
_BASELINE_PACKAGE_ID = "cbo_full_horizon_local_smoke_3m"
_BASELINE_PACKAGE_SHA256 = (
    "be49f5a5d256863649ccf1b139d259679c9a3ce669642751c22e8a7dad9a2d2c"
)
_BASELINE_MANIFEST_SHA256 = (
    "4cf0d3571abcd0d80c91a28e23f9c4e00ce6075564f5354e3b12e5364727e16c"
)
_BASELINE_ATTESTATION_SHA256 = (
    "d30a6f89263cdb80f8f9d81131dc0004b7b9751289e0594ee5005e115641124a"
)
_EXPECTED_ROLE_TO_SCENARIO_ID = {
    "baseline": "tdcsim_open04_paired_baseline_noop_v1",
    "candidate_a": (
        "tdcsim_open04_candidate_a_shorter_10y_down_"
        "bond_bank_substitution_1pp_v2"
    ),
    "candidate_b": "tdcsim_open04_candidate_b_longer_10y_up_v1",
}


def test_stable_marker_and_role_to_scenario_id_mapping_are_exact() -> None:
    scenario = _marked_baseline_mapping()
    spec = CboScenarioSpec.from_mapping(scenario)

    assert OPEN04_CAMPAIGN_CONTRACT_ID == "open04_paired_tradeoff_v1"
    assert OPEN04_CAMPAIGN_ROLES == (
        "baseline",
        "candidate_a",
        "candidate_b",
    )
    assert dict(OPEN04_ROLE_TO_SCENARIO_ID) == (
        _EXPECTED_ROLE_TO_SCENARIO_ID
    )
    assert dict(OPEN04_BASELINE_IDENTITY) == scenario["baseline"]
    expected_marker = Open04CampaignMarker(
        contract_id=OPEN04_CAMPAIGN_CONTRACT_ID,
        role="baseline",
        funding_closure_mode=OPEN04_FUNDING_CLOSURE_MODE,
    )
    assert parse_open04_campaign_marker(spec.data) == expected_marker
    assert validate_open04_scenario_contract(spec.data) == expected_marker


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong_contract",
        "invalid_role",
        "extra_marker_field",
        "missing_funding_closure_mode",
        "legacy_funding_closure_mode",
        "role_to_id_mismatch",
    ),
)
def test_marker_schema_and_role_to_id_mutations_fail_closed(
    mutation: str,
) -> None:
    scenario = _marked_baseline_mapping()
    marker = scenario["open04_campaign"]
    if mutation == "wrong_contract":
        marker["contract_id"] = "open04_paired_tradeoff_v0"
    elif mutation == "invalid_role":
        marker["role"] = "candidate_c"
    elif mutation == "extra_marker_field":
        marker["campaign_id"] = "host-runtime-label"
    elif mutation == "missing_funding_closure_mode":
        del marker["funding_closure_mode"]
    elif mutation == "legacy_funding_closure_mode":
        marker["funding_closure_mode"] = "cbo_public_debt_target"
    elif mutation == "role_to_id_mismatch":
        marker["role"] = "candidate_a"
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ValueError):
        CboScenarioSpec.from_mapping(scenario)


def test_open04_baseline_requires_all_exact_release_pins() -> None:
    for field in (
        "package_id",
        "package_sha256",
        "manifest_sha256",
        "release_attestation_sha256",
    ):
        scenario = _marked_baseline_mapping()
        del scenario["baseline"][field]
        with pytest.raises(ValueError):
            CboScenarioSpec.from_mapping(scenario)

    for field in (
        "package_id",
        "package_sha256",
        "manifest_sha256",
        "release_attestation_sha256",
    ):
        scenario = _marked_baseline_mapping()
        scenario["baseline"][field] = (
            "wrong-release-package"
            if field == "package_id"
            else "0" * 64
        )
        with pytest.raises(Open04CampaignError):
            CboScenarioSpec.from_mapping(scenario)

    compatible = _marked_baseline_mapping()
    compatible["baseline"]["allow_compatible_baseline"] = True
    with pytest.raises(ValueError, match="compatible_baseline"):
        CboScenarioSpec.from_mapping(compatible)


@pytest.mark.parametrize(
    "mutation",
    ("horizon", "coupling", "output", "override"),
)
def test_marked_baseline_perimeter_mutations_fail_closed(
    mutation: str,
) -> None:
    scenario = _marked_baseline_mapping()
    if mutation == "horizon":
        scenario["simulation"]["end_date"] = "2035-09-30"
    elif mutation == "coupling":
        scenario["coupling"]["tips_real_yield"] = (
            "recompute_from_nominal_and_scenario_inflation"
        )
    elif mutation == "output":
        scenario["output"]["catalog_sqlite"] = True
    elif mutation == "override":
        scenario["overrides"]["nominal_yield_curve"] = {
            "mode": "parallel_bp",
            "shock_bp": 0.0,
        }
    else:  # pragma: no cover - guarded by the parameter list
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(Open04CampaignError):
        CboScenarioSpec.from_mapping(scenario)


def test_runtime_environment_does_not_change_canonical_scenario_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = _marked_baseline_mapping()
    before = CboScenarioSpec.from_mapping(scenario).canonical_sha256()

    monkeypatch.setenv("TDCSIM_CBO_CAMPAIGN_ID", "runtime-campaign-a")
    monkeypatch.setenv(
        "TDCSIM_CBO_CAMPAIGN_ROOT",
        "runtime-only/open04-a",
    )
    monkeypatch.setenv(
        "TDCSIM_CBO_WATCHDOG_HANDOFF",
        "runtime-only/handoff.json",
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    after = CboScenarioSpec.from_mapping(scenario).canonical_sha256()

    assert after == before
    assert len(before) == 64


def test_marked_baseline_compile_is_strict_without_a_curve_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _full_horizon_compiler_baseline(tmp_path)
    monkeypatch.setattr(
        open04_campaign_module,
        "OPEN04_BASELINE_IDENTITY",
        {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
    )
    scenario = _marked_baseline_mapping(baseline)
    spec = CboScenarioSpec.from_mapping(scenario)

    compiled = CboScenarioCompiler().compile(
        baseline,
        spec,
        tmp_path / "compiled-open04-baseline",
    )

    surface_name = "tdcsim_yield_curve_surface.csv"
    baseline_surface = (
        compiled.baseline_dir / "forecast_inputs" / surface_name
    )
    compiled_surface = compiled.forecast_inputs_dir / surface_name
    assert sha256_file(compiled_surface) == sha256_file(baseline_surface)
    assert NOMINAL_EVALUATED_SHOCK_FILE not in compiled.changed_inputs
    assert not (
        compiled.forecast_inputs_dir / NOMINAL_EVALUATED_SHOCK_FILE
    ).exists()

    manifest = compiled.manifest
    assert manifest["open04_campaign"] == scenario["open04_campaign"]
    assert "evaluated_nominal_curve" not in manifest
    simulation_contract = dict(manifest["open04_simulation_contract"])
    selected_date_set_sha256 = simulation_contract.pop(
        "runtime_selected_curve_date_set_sha256"
    )
    assert simulation_contract == {
        "schema_version": "tdcsim_open04_simulation_contract_v1",
        "frequency": "daily",
        "start_date": _START,
        "end_date": _END,
        "runtime_selected_curve_date_count": 2,
        "output_profile": "compact",
        "compression": "gzip",
    }
    assert selected_date_set_sha256 == canonical_json_sha256(
        [_START, _END]
    )
    perimeter = manifest["open04_change_perimeter"]
    assert perimeter["economic_changed_paths"] == []
    assert perimeter["physical_changed_inputs"] == []
    assert perimeter["fixed_adapter_inputs"] == []
    assert perimeter["fixed_input_comparison_status"] == "pass"

    issuance = read_json(compiled.forecast_inputs_dir / ISSUANCE_MIX_FILE)
    assert issuance["selection_status"] == "configured_default"
    assert issuance["security_shares"] == {
        "bills": 0.225,
        "notes": 0.495,
        "bonds": 0.18,
        "tips": 0.06,
        "frn": 0.04,
    }
    assert issuance["maturity_distributions"] == {
        "bills": [{"maturity_years": 0.5, "share": 1.0}],
        "notes": [{"maturity_years": 5.0, "share": 1.0}],
        "bonds": [{"maturity_years": 20.0, "share": 1.0}],
        "tips": [{"maturity_years": 10.0, "share": 1.0}],
        "frn": [{"maturity_years": 2.0, "share": 1.0}],
    }
    assert issuance["weighted_average_maturity_years"] == pytest.approx(
        6.8675
    )


def test_marked_baseline_classifies_opening_fed_identity_as_fixed_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _full_horizon_compiler_baseline(
        tmp_path,
        include_opening_fed_target=False,
    )
    monkeypatch.setattr(
        open04_campaign_module,
        "OPEN04_BASELINE_IDENTITY",
        {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
    )
    compiled = CboScenarioCompiler().compile(
        baseline,
        CboScenarioSpec.from_mapping(_marked_baseline_mapping(baseline)),
        tmp_path / "compiled-open04-baseline",
    )
    candidate = CboScenarioCompiler().compile(
        baseline,
        CboScenarioSpec.from_mapping(
            _open04_scenario_mapping(baseline, shock_bp=-25.0)
        ),
        tmp_path / "compiled-open04-candidate",
    )

    perimeter = compiled.manifest["open04_change_perimeter"]
    assert perimeter["economic_changed_paths"] == []
    assert perimeter["physical_changed_inputs"] == []
    assert perimeter["fixed_adapter_inputs"] == [
        INPUT_FILES["fed_holdings"]
    ]
    assert perimeter["fixed_input_comparison_status"] == "pass"
    assert INPUT_FILES["fed_holdings"] in compiled.changed_inputs
    assert verify_compiled_scenario(compiled.compiled_dir)["status"] == "pass"
    assert verify_compiled_scenario(candidate.compiled_dir)["status"] == "pass"
    assert (
        sha256_file(
            compiled.forecast_inputs_dir / INPUT_FILES["fed_holdings"]
        )
        == sha256_file(
            candidate.forecast_inputs_dir / INPUT_FILES["fed_holdings"]
        )
    )


def test_shared_strict_predicate_covers_marked_baseline_and_sidecar_mode() -> None:
    marked_baseline = CboScenarioSpec.from_mapping(
        _marked_baseline_mapping()
    )

    fixture_baseline = SimpleNamespace(
        package_id="fixture",
        package_sha256="1" * 64,
        manifest_sha256="2" * 64,
        attestation=SimpleNamespace(sha256="3" * 64),
    )
    evaluated_sidecar = CboScenarioSpec.from_mapping(
        _open04_scenario_mapping(fixture_baseline, shock_bp=-25.0)
    )

    generic = deepcopy(marked_baseline.data)
    generic.pop("open04_campaign")
    generic["scenario_id"] = "generic_noop_baseline_v1"
    generic_spec = CboScenarioSpec.from_mapping(generic)

    assert requires_open04_strict_execution(marked_baseline.data) is True
    assert requires_open04_strict_execution(evaluated_sidecar.data) is True
    assert requires_open04_strict_execution(generic_spec.data) is False


def _marked_baseline_mapping(
    baseline: CboBaselinePackage | None = None,
) -> dict:
    baseline_block = {
        "package_id": _BASELINE_PACKAGE_ID,
        "package_sha256": _BASELINE_PACKAGE_SHA256,
        "manifest_sha256": _BASELINE_MANIFEST_SHA256,
        "release_attestation_sha256": _BASELINE_ATTESTATION_SHA256,
    }
    if baseline is not None:
        baseline_block = {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        }
    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": OPEN04_ROLE_TO_SCENARIO_ID["baseline"],
        "baseline": baseline_block,
        "provenance": {
            "kind": "user_stress_assumption",
            "label": "common_release_bound_baseline_noop",
        },
        "simulation": {
            "frequency": "daily",
            "start_date": _START,
            "end_date": _END,
        },
        "open04_campaign": {
            "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
            "role": "baseline",
            "funding_closure_mode": OPEN04_FUNDING_CLOSURE_MODE,
        },
        "coupling": dict(OPEN04_FIXED_COUPLING),
        "overrides": {},
        "output": dict(OPEN04_OUTPUT_CONTRACT),
    }


def _full_horizon_compiler_baseline(
    tmp_path: Path,
    *,
    include_opening_fed_target: bool = True,
) -> CboBaselinePackage:
    fixture_root = tmp_path / "fixture-release"
    package, attestation_path = _write_compiler_package(
        fixture_root,
        include_opening_fed_target=include_opening_fed_target,
    )
    package_dir = fixture_root / "compiler_pkg"
    inputs = package_dir / "forecast_inputs"

    debt_rows = _read_csv(inputs / "tdcsim_debt_stock_path.csv")
    debt_rows[0]["period_end"] = _START
    debt_rows[-1]["period_end"] = _END
    _write_csv(inputs / "tdcsim_debt_stock_path.csv", debt_rows)

    surface_rows = _read_csv(inputs / "tdcsim_yield_curve_surface.csv")
    original_dates = sorted({row["curve_date"] for row in surface_rows})
    assert len(original_dates) == 2
    for row in surface_rows:
        row["curve_date"] = (
            _START if row["curve_date"] == original_dates[0] else _END
        )
    _write_csv(inputs / "tdcsim_yield_curve_surface.csv", surface_rows)

    with zipfile.ZipFile(
        package,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    path.relative_to(package_dir).as_posix(),
                )

    attestation = read_json(attestation_path)
    attestation["baseline_package_zip_sha256"] = sha256_file(package)
    write_json(attestation_path, attestation)
    return CboBaselinePackage.open(
        package,
        attestation_path=attestation_path,
    )
