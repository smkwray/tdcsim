from __future__ import annotations

import json
import math
import os
import struct
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from evaluated_nominal_curve import EvaluatedNominalShock
import sim_engine
from sim_pricing import get_yield_for_maturity
import tdcsim_cbo.runner as runner_module
import tdcsim_cbo.curve_runtime as curve_runtime_module
import tdcsim_cbo.verifier as verifier_module
from tdcsim_cbo import (
    CboScenarioCompiler,
    CboScenarioSpec,
    run_cbo_scenario,
)
from tdcsim_cbo._json import read_json, sha256_file, write_json
from tdcsim_cbo.curve_runtime import (
    EvaluatedNominalRuntimeError,
    build_evaluated_nominal_runtime_binding,
    load_compiled_evaluated_nominal_contract,
)
from tdcsim_cbo.bounded_output import BoundedResourceLimits
from tdcsim_cbo.compiler import (
    ISSUANCE_MIX_FILE,
    digest_input_tree,
    input_tree_hashes,
)
from tdcsim_cbo.process_watchdog import (
    GIB,
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    WatchdogResult,
)
from tdcsim_cbo.verifier import (
    VerificationError,
    _verify_compiled_run_scenario_identity,
    _verify_run_evaluated_nominal_curve,
    _verify_wheel_artifact,
    verify_compiled_scenario,
    verify_scenario_run,
)
from tdcsim_cbo.runtime_identity import wheel_file_digest
from tdcsim_cbo.runner import (
    RunnerError,
    build_runtime_params,
    finalize_watchdog_handoff,
)
from test_tdcsim_cbo_closeout_interface import (
    _runner_baseline_and_scenarios,
)
from test_tdcsim_cbo_compiler import (
    _compiler_baseline,
    _open04_scenario_mapping,
)


def _compiled_open04(tmp_path: Path):
    baseline = _compiler_baseline(tmp_path)
    spec = CboScenarioSpec.from_mapping(
        _open04_scenario_mapping(baseline, shock_bp=-25.0)
    )
    return CboScenarioCompiler().compile(
        baseline,
        spec,
        tmp_path / "work",
    )


def _compiled_runner_open04(tmp_path: Path):
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_runner_fixture_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=-25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario_path = tmp_path / "open04-runner-fixture.json"
    write_json(scenario_path, scenario)
    return CboScenarioCompiler().compile(
        baseline,
        CboScenarioSpec.from_file(scenario_path),
        tmp_path / "open04-runner-compile",
    )


def test_compiled_verifier_recomputes_evaluated_nominal_contract(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)

    result = verify_compiled_scenario(compiled.compiled_dir)

    assert result["status"] == "pass"
    assert result["evaluated_nominal_curve"] == (
        compiled.manifest["evaluated_nominal_curve"]
    )


def test_runtime_binding_is_streamed_bounded_and_reproducible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled = _compiled_open04(tmp_path)
    original_evaluate = (
        curve_runtime_module.sim_pricing.evaluate_nominal_yield
    )
    calls = 0

    def tracked_evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        curve_runtime_module.sim_pricing,
        "evaluate_nominal_yield",
        tracked_evaluate,
    )

    first = build_evaluated_nominal_runtime_binding(
        compiled.forecast_inputs_dir,
        start_date="2027-01-01",
        end_date="2027-01-02",
    )
    second = build_evaluated_nominal_runtime_binding(
        compiled.forecast_inputs_dir,
        start_date="2027-01-01",
        end_date="2027-01-02",
    )

    assert first == second
    assert first is not None
    assert first["runtime_selected_curve_date_count"] == 2
    assert first["grid_record_count"] > 2 * 8193
    assert first["short_end_bitwise_mismatch_count"] == 0
    assert first["max_abs_analytic_delta_error_decimal"] <= 1e-12
    assert len(first["evaluated_delta_sha256"]) == 64
    assert calls == 2 * first["grid_record_count"]


def test_runtime_rejects_one_byte_sidecar_mutation_even_when_json_still_parses(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)
    sidecar_path = (
        compiled.forecast_inputs_dir
        / "tdcsim_nominal_curve_evaluated_shock.json"
    )
    sidecar_path.write_bytes(sidecar_path.read_bytes() + b" ")

    with pytest.raises(
        EvaluatedNominalRuntimeError,
        match="metadata does not match runtime bytes",
    ):
        load_compiled_evaluated_nominal_contract(
            compiled.forecast_inputs_dir
        )


def test_compiled_verifier_rejects_manifest_curve_metadata_mutation(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)
    manifest = json.loads(
        compiled.manifest_path.read_text(encoding="utf-8")
    )
    manifest["evaluated_nominal_curve"]["baseline_surface"][
        "date_set_sha256"
    ] = "0" * 64
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(
        VerificationError,
        match="metadata does not match runtime bytes",
    ):
        verify_compiled_scenario(compiled.compiled_dir)


def test_compiled_verifier_rejects_change_perimeter_mutation(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)
    manifest = json.loads(
        compiled.manifest_path.read_text(encoding="utf-8")
    )
    manifest["open04_change_perimeter"]["economic_changed_paths"] = [
        "nominal_yield_curve_assumption"
    ]
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(
        VerificationError,
        match="change perimeter does not match runtime bytes",
    ):
        verify_compiled_scenario(compiled.compiled_dir)


def test_compiled_verifier_rejects_false_fixed_adapter_declaration(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)
    manifest = read_json(compiled.manifest_path)
    manifest["open04_change_perimeter"]["fixed_adapter_inputs"] = [
        "tdcsim_fed_holdings_path.csv"
    ]
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(
        VerificationError,
        match="fixed_adapter_inputs",
    ):
        verify_compiled_scenario(compiled.compiled_dir)


def test_compiled_verifier_rejects_fixed_adapter_record_digest_mutation(
    tmp_path: Path,
) -> None:
    compiled = _compiled_open04(tmp_path)
    manifest = read_json(compiled.manifest_path)
    manifest["open04_change_perimeter"][
        "fixed_adapter_input_records_sha256"
    ] = "0" * 64
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(
        VerificationError,
        match="change perimeter does not match runtime bytes",
    ):
        verify_compiled_scenario(compiled.compiled_dir)


@pytest.mark.parametrize(
    ("field", "mutation", "message"),
    [
        (
            "baseline",
            lambda manifest: manifest["baseline"].update(
                {"package_sha256": "0" * 64}
            ),
            "scenario contract identity",
        ),
        (
            "coupling",
            lambda manifest: manifest["coupling"].update(
                {"frn_benchmark": "derive_from_scenario_nominal_curve"}
            ),
            "scenario contract identity",
        ),
        (
            "horizon",
            lambda manifest: manifest[
                "open04_simulation_contract"
            ].update({"end_date": "2027-01-01"}),
            "simulation contract does not match inputs",
        ),
    ],
)
def test_compiled_verifier_rejects_lineage_and_horizon_relabeling(
    tmp_path: Path,
    field: str,
    mutation,
    message: str,
) -> None:
    del field
    compiled = _compiled_open04(tmp_path)
    manifest = read_json(compiled.manifest_path)
    mutation(manifest)
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(VerificationError, match=message):
        verify_compiled_scenario(compiled.compiled_dir)


@pytest.mark.parametrize("mutation", ["remove_fixed", "add_unapproved"])
def test_compiled_verifier_rejects_coherently_rehashed_input_set_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    compiled = _compiled_open04(tmp_path)
    if mutation == "remove_fixed":
        (
            compiled.forecast_inputs_dir / "tdcsim_frn_rate_path.csv"
        ).unlink()
    elif mutation == "add_unapproved":
        (
            compiled.forecast_inputs_dir / "rogue_behavior_input.json"
        ).write_text("{}\n", encoding="utf-8")
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    manifest = read_json(compiled.manifest_path)
    manifest["compiled_inputs_digest"] = digest_input_tree(
        compiled.forecast_inputs_dir
    )
    manifest["input_hashes"] = input_tree_hashes(
        compiled.forecast_inputs_dir
    )
    write_json(compiled.manifest_path, manifest)

    with pytest.raises(
        VerificationError,
        match="change perimeter is invalid",
    ):
        verify_compiled_scenario(compiled.compiled_dir)


def test_production_runner_loads_hash_bound_sidecar_and_fixed_controls(
    tmp_path: Path,
) -> None:
    compiled = _compiled_runner_open04(tmp_path)

    params = build_runtime_params(
        compiled.forecast_inputs_dir,
        simulation_start_date="2026-09-20",
        simulation_end_date="2026-09-30",
    )

    shock = params["yield_curve_surface"]["evaluated_nominal_shock"]
    assert shock == EvaluatedNominalShock(-25.0)
    assert params["yield_curve_surface"]["interpolation_method"] == "pchip"
    assert params["yield_curve_surface"]["floor_zero"] is False
    assert params["rate_sensitive_demand"] == {"enabled": False}
    assert params["simulation_period"]["enable_preference_trading"] is False
    assert params["events"] == []


def test_baseline_a_b_compiles_preserve_fixed_inputs_and_opening_state(
    tmp_path: Path,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    baseline_spec = CboScenarioSpec.from_file(scenarios["noop"])
    compiled_baseline = CboScenarioCompiler().compile(
        baseline,
        baseline_spec,
        tmp_path / "compiled-baseline",
    )

    candidates = {}
    for label, shock_bp in (("a", -25.0), ("b", 25.0)):
        scenario = read_json(scenarios["noop"])
        scenario["scenario_id"] = f"open04_fixed_perimeter_{label}_v1"
        scenario["coupling"]["tips_real_yield"] = (
            "independent_explicit_path"
        )
        scenario["overrides"] = _open04_scenario_mapping(
            baseline,
            shock_bp=shock_bp,
        )["overrides"]
        scenario["output"] = {"profile": "compact", "compression": "gzip"}
        path = tmp_path / f"open04-fixed-perimeter-{label}.json"
        write_json(path, scenario)
        candidates[label] = CboScenarioCompiler().compile(
            baseline,
            CboScenarioSpec.from_file(path),
            tmp_path / f"compiled-{label}",
        )

    def hashes(compiled) -> dict[str, str]:
        return {
            item["path"]: item["sha256"]
            for item in compiled.manifest["input_hashes"]
        }

    baseline_hashes = hashes(compiled_baseline)
    a_hashes = hashes(candidates["a"])
    b_hashes = hashes(candidates["b"])
    all_paths = set(baseline_hashes) | set(a_hashes) | set(b_hashes)

    def differing(
        left: dict[str, str],
        right: dict[str, str],
    ) -> set[str]:
        return {
            path
            for path in all_paths
            if left.get(path) != right.get(path)
        }

    assert differing(baseline_hashes, a_hashes) == {
        ISSUANCE_MIX_FILE,
        "tdcsim_nominal_curve_evaluated_shock.json",
    }
    assert differing(baseline_hashes, b_hashes) == {
        ISSUANCE_MIX_FILE,
        "tdcsim_nominal_curve_evaluated_shock.json",
    }
    assert differing(a_hashes, b_hashes) == {
        "tdcsim_nominal_curve_evaluated_shock.json",
    }
    assert a_hashes["tdcsim_yield_curve_surface.csv"] == (
        baseline_hashes["tdcsim_yield_curve_surface.csv"]
    )
    assert b_hashes["tdcsim_yield_curve_surface.csv"] == (
        baseline_hashes["tdcsim_yield_curve_surface.csv"]
    )

    baseline_params = build_runtime_params(
        compiled_baseline.forecast_inputs_dir
    )
    candidate_params = {
        label: build_runtime_params(
            compiled.forecast_inputs_dir,
            simulation_start_date="2026-09-20",
            simulation_end_date="2026-09-30",
        )
        for label, compiled in candidates.items()
    }
    opening = baseline_params["initial_bonds_df"]
    for params in candidate_params.values():
        pd.testing.assert_frame_equal(
            opening,
            params["initial_bonds_df"],
            check_exact=True,
        )
    opening_sha = baseline_hashes["tdcsim_opening_portfolio.csv"]
    assert a_hashes["tdcsim_opening_portfolio.csv"] == opening_sha
    assert b_hashes["tdcsim_opening_portfolio.csv"] == opening_sha


def test_active_sidecar_requires_horizon_for_frn_preflight(
    tmp_path: Path,
) -> None:
    compiled = _compiled_runner_open04(tmp_path)

    with pytest.raises(
        RunnerError,
        match="explicit simulation horizon",
    ):
        build_runtime_params(compiled.forecast_inputs_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("remove", "valid explicit FRN path"),
        ("truncate", "covering 2026-09-30"),
        ("nonnumeric", "valid explicit FRN path"),
        ("duplicate_end", "duplicate runtime"),
        ("partial_exact", "covering 2026-09-26"),
    ],
)
def test_active_sidecar_frn_path_fails_closed_before_simulation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    compiled = _compiled_runner_open04(tmp_path)
    path = compiled.forecast_inputs_dir / "tdcsim_frn_rate_path.csv"
    if mutation == "remove":
        path.unlink()
    else:
        frame = pd.read_csv(path)
        if mutation == "truncate":
            frame.loc[frame.index[-1], "period_end"] = "2026-09-29"
            frame.loc[
                frame.index[-1], "rate_effective_end"
            ] = "2026-09-29"
        elif mutation == "nonnumeric":
            frame["benchmark_rate_decimal"] = frame[
                "benchmark_rate_decimal"
            ].astype(object)
            frame.loc[frame.index[-1], "benchmark_rate_decimal"] = "invalid"
        elif mutation == "duplicate_end":
            duplicate = frame.iloc[[-1]].copy()
            duplicate["period_start"] = "2026-09-28"
            frame = pd.concat([frame, duplicate], ignore_index=True)
        elif mutation == "partial_exact":
            fallback = frame.copy()
            fallback["scenario_id"] = "default"
            exact = frame.copy()
            exact["scenario_id"] = "baseline"
            exact["period_end"] = "2026-09-25"
            exact["rate_effective_end"] = "2026-09-25"
            frame = pd.concat([fallback, exact], ignore_index=True)
        else:
            raise AssertionError(f"unhandled mutation: {mutation}")
        frame.to_csv(path, index=False)

    with pytest.raises(RunnerError, match=message):
        build_runtime_params(
            compiled.forecast_inputs_dir,
            simulation_start_date="2026-09-20",
            simulation_end_date="2026-09-30",
        )


def test_active_open04_runner_rejects_unmonitored_direct_execution(
    tmp_path: Path,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_live_entry_fixture_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario_path = tmp_path / "open04-live-entry-fixture.json"
    write_json(scenario_path, scenario)

    output_dir = tmp_path / "run-open04-live-entry"
    with pytest.raises(
        RunnerError,
        match="require the parent RSS watchdog",
    ):
        run_cbo_scenario(
            baseline,
            CboScenarioSpec.from_file(scenario_path),
            output_dir,
        )

    assert not output_dir.exists()
    assert not list(tmp_path.glob("run-open04-live-entry.failure-*.json"))


def test_active_open04_runner_rejects_output_profile_override(
    tmp_path: Path,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_output_profile_fixture_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario_path = tmp_path / "open04-output-profile-fixture.json"
    write_json(scenario_path, scenario)

    output_dir = tmp_path / "run-open04-output-profile"
    handoff = tmp_path / f".{output_dir.name}.watchdog-handoff-test.json"
    with pytest.raises(
        RunnerError,
        match="forbid output-profile overrides",
    ):
        run_cbo_scenario(
            baseline,
            CboScenarioSpec.from_file(scenario_path),
            output_dir,
            output_profile="audit",
            watchdog_handoff=handoff,
        )

    assert not output_dir.exists()
    assert not handoff.exists()


def test_active_open04_runner_rejects_noncanonical_memory_limits(
    tmp_path: Path,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_memory_envelope_fixture_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=-25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario_path = tmp_path / "open04-memory-envelope-fixture.json"
    write_json(scenario_path, scenario)
    output_dir = tmp_path / "run-open04-memory-envelope"
    handoff = tmp_path / f".{output_dir.name}.watchdog-handoff-test.json"

    with pytest.raises(
        RunnerError,
        match="exact 4/6/8/10/12 GiB memory envelope",
    ):
        run_cbo_scenario(
            baseline,
            CboScenarioSpec.from_file(scenario_path),
            output_dir,
            watchdog_handoff=handoff,
            resource_limits=BoundedResourceLimits(
                minimum_available_bytes=0,
            ),
        )

    assert not output_dir.exists()
    assert not handoff.exists()


def test_active_open04_runner_requires_single_numerical_threads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_thread_envelope_fixture_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=-25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario_path = tmp_path / "open04-thread-envelope-fixture.json"
    write_json(scenario_path, scenario)
    output_dir = tmp_path / "run-open04-thread-envelope"
    handoff = tmp_path / f".{output_dir.name}.watchdog-handoff-test.json"
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")

    with pytest.raises(
        RunnerError,
        match="numerical thread limits pinned to one",
    ):
        run_cbo_scenario(
            baseline,
            CboScenarioSpec.from_file(scenario_path),
            output_dir,
            watchdog_handoff=handoff,
        )

    assert not output_dir.exists()
    assert not handoff.exists()


def test_active_open04_release_identity_requires_retained_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, _scenarios = _runner_baseline_and_scenarios(tmp_path)
    for name in (
        "TDCSIM_CBO_WHEEL_PATH",
        "TDCSIM_CBO_WHEEL_SHA256",
        "TDCSIM_CBO_CODE_COMMIT_SHA",
        "TDCSIM_CBO_DIRTY_STATE",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        RunnerError,
        match="retained release wheel",
    ):
        runner_module._assert_open04_release_identity(baseline)


def test_strict_wheel_verifier_accepts_blank_environment_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "runtime" / "tdcsim-0.1-py3-none-any.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sim_engine.py", b"release bytes\n")
    wheel_sha = sha256_file(wheel)
    monkeypatch.delenv("TDCSIM_CBO_WHEEL_SHA256", raising=False)

    _verify_wheel_artifact(
        tmp_path,
        {
            "wheel_sha256": wheel_sha,
            "wheel_artifact": {
                "relative_path": wheel.relative_to(tmp_path).as_posix(),
                "sha256": wheel_sha,
                "bytes": wheel.stat().st_size,
            },
            "code_commit_sha": "a" * 40,
            "dirty_state": False,
            "distribution_file_digest": wheel_file_digest(wheel),
        },
        required=True,
    )


def test_strict_wheel_verifier_rejects_zero_commit(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "runtime" / "tdcsim-0.1-py3-none-any.whl"
    wheel.parent.mkdir()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sim_engine.py", b"release bytes\n")
    wheel_sha = sha256_file(wheel)

    with pytest.raises(
        VerificationError,
        match="code_commit_sha is not release-bound",
    ):
        _verify_wheel_artifact(
            tmp_path,
            {
                "wheel_sha256": wheel_sha,
                "wheel_artifact": {
                    "relative_path": wheel.relative_to(tmp_path).as_posix(),
                    "sha256": wheel_sha,
                    "bytes": wheel.stat().st_size,
                },
                "code_commit_sha": "0" * 40,
                "dirty_state": False,
                "distribution_file_digest": wheel_file_digest(wheel),
            },
            required=True,
        )


def test_open04_campaign_root_requires_isolated_role_writer_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root = tmp_path / "campaign"
    monkeypatch.setenv("TDCSIM_CBO_CAMPAIGN_ROOT", str(campaign_root))
    monkeypatch.setenv(
        "TDCSIM_CBO_CAMPAIGN_ID",
        "open04-paired-contract-v1",
    )

    assert (
        runner_module._assert_open04_campaign_root(
            campaign_root / "candidate-a-role" / "candidate-a"
        )
        == "open04-paired-contract-v1"
    )
    with pytest.raises(
        RunnerError,
        match="inside the declared campaign root",
    ):
        runner_module._assert_open04_campaign_root(
            tmp_path / "other-parent" / "candidate-b"
        )


def test_monitored_production_entry_consumes_open04_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    scenario = read_json(scenarios["noop"])
    scenario["scenario_id"] = "open04_monitored_live_entry_v1"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    scenario["overrides"] = _open04_scenario_mapping(
        baseline,
        shock_bp=-25.0,
    )["overrides"]
    scenario["output"] = {"profile": "compact", "compression": "gzip"}
    scenario["overrides"]["issuance_mix"]["maturity_distributions"][
        "notes"
    ] = [
        {"maturity_years": 1.75, "share": 0.25},
        {"maturity_years": 5.0, "share": 0.375},
        {"maturity_years": 10.0, "share": 0.375},
    ]
    scenario_path = tmp_path / "open04-monitored-live-entry.json"
    write_json(scenario_path, scenario)

    production_sink = runner_module.BoundedScenarioEvidenceSink

    def deterministic_sink(output_dir, *, limits, progress_callback=None):
        return production_sink(
            output_dir,
            limits=limits,
            rss_reader=lambda: 128 * 1024**2,
            available_reader=lambda: 256 * GIB,
            cpu_reader=lambda: 1.0,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(
        runner_module,
        "host_available_memory_bytes",
        lambda: 256 * GIB,
    )
    monkeypatch.setattr(
        runner_module,
        "BoundedScenarioEvidenceSink",
        deterministic_sink,
    )
    original_code_environment = runner_module._code_environment
    monkeypatch.setattr(
        runner_module,
        "_assert_open04_release_identity",
        lambda _baseline: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_code_environment",
        lambda baseline_package, run_root, **_kwargs: original_code_environment(
            baseline_package,
            run_root,
            require_release_identity=False,
        ),
    )
    monkeypatch.setattr(
        verifier_module,
        "_verify_code_environment",
        lambda _root, _manifest, **_kwargs: None,
    )
    monkeypatch.setenv("TDCSIM_CBO_CAMPAIGN_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "TDCSIM_CBO_CAMPAIGN_ID",
        "open04-monitored-live-entry-v1",
    )
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "1")

    original_evaluate = sim_engine.evaluate_nominal_yield
    observed: dict[float, tuple[float, float]] = {}

    def tracked_evaluate(
        maturity_years,
        curve_years,
        curve_rates,
        *,
        method,
        floor_zero,
        shock,
    ):
        candidate = original_evaluate(
            maturity_years,
            curve_years,
            curve_rates,
            method=method,
            floor_zero=floor_zero,
            shock=shock,
        )
        maturity = float(maturity_years)
        if shock is not None and maturity in {0.5, 1.75, 5.0, 10.0}:
            baseline_value = get_yield_for_maturity(
                maturity,
                curve_years,
                curve_rates,
                method=method,
                floor_zero=floor_zero,
            )
            observed.setdefault(
                maturity,
                (float(baseline_value), float(candidate)),
            )
        return candidate

    monkeypatch.setattr(
        sim_engine,
        "evaluate_nominal_yield",
        tracked_evaluate,
    )

    output_dir = (
        tmp_path
        / "baseline-role"
        / "run-open04-monitored-live-entry"
    )
    handoff = output_dir.parent / f".{output_dir.name}.watchdog-handoff-test.json"
    prepared = run_cbo_scenario(
        baseline,
        CboScenarioSpec.from_file(scenario_path),
        output_dir,
        watchdog_handoff=handoff,
    )

    assert prepared.run_manifest["status"] == (
        "pending_parent_watchdog_acceptance"
    )
    finalized = finalize_watchdog_handoff(
        output_dir,
        handoff,
        WatchdogResult(
            child_pid=os.getpid(),
            returncode=0,
            action="completed",
            peak_rss_bytes=128 * 1024**2,
            last_rss_bytes=128 * 1024**2,
            acceptance_peak_rss_bytes=6 * GIB,
            terminate_rss_bytes=10 * GIB,
            kill_rss_bytes=12 * GIB,
            poll_interval_seconds=1.0,
        ),
    )

    assert finalized is not None
    assert set(observed) == {0.5, 1.75, 5.0, 10.0}
    for maturity in (0.5, 1.75):
        short_base, short_candidate = observed[maturity]
        assert struct.pack(">d", short_candidate) == struct.pack(">d", short_base)
    five_base, five_candidate = observed[5.0]
    ten_base, ten_candidate = observed[10.0]
    assert five_candidate - five_base == pytest.approx(
        -25.0 * math.log(2.5) / math.log(5.0) / 10_000.0,
        abs=1e-15,
    )
    assert ten_candidate - ten_base == pytest.approx(-0.0025, abs=1e-15)
    verification = verify_scenario_run(output_dir)
    assert verification["status"] == "pass"
    assert finalized["parent_watchdog"]["status"] == "accepted"


def test_run_verifier_rejects_compiled_scenario_relabeling() -> None:
    with pytest.raises(
        VerificationError,
        match="compiled scenario hash",
    ):
        _verify_compiled_run_scenario_identity(
            {
                "scenario_id": "candidate_a",
                "scenario_sha256": "a" * 64,
            },
            {
                "scenario_id": "candidate_b",
                "canonical_sha256": "b" * 64,
            },
        )


def test_run_verifier_rejects_scenario_sign_sidecar_relabeling(
    tmp_path: Path,
) -> None:
    baseline = _compiler_baseline(tmp_path)
    candidate_a = CboScenarioSpec.from_mapping(
        _open04_scenario_mapping(baseline, shock_bp=-25.0)
    )
    compiled_a = CboScenarioCompiler().compile(
        baseline,
        candidate_a,
        tmp_path / "compiled-a",
    )
    candidate_b = CboScenarioSpec.from_mapping(
        _open04_scenario_mapping(baseline, shock_bp=25.0)
    )
    run_root = tmp_path / "run-relabel"
    run_root.mkdir()
    scenario_path = run_root / "scenario.json"
    write_json(scenario_path, candidate_b.data)

    with pytest.raises(
        VerificationError,
        match="signed 10-year shock does not match compiled sidecar",
    ):
        _verify_run_evaluated_nominal_curve(
            compiled_a.compiled_dir,
            run_root,
            {
                "scenario": {"relative_path": "scenario.json"},
                "simulation": {
                    "start_date": "2027-01-01",
                    "end_date": "2027-01-02",
                },
                "evaluated_nominal_curve": {},
            },
            compiled_curve=compiled_a.manifest["evaluated_nominal_curve"],
        )
