import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, run_cbo_scenario
from tdcsim_cbo._json import read_json, sha256_file, write_json
from tdcsim_cbo.forecast_state import (
    ForecastStateExportError,
    ForecastStateWindow,
    export_forecast_state_package,
    run_no_shock_rollforward,
)
from test_tdcsim_cbo_closeout_interface import (
    _write_runner_package,
    _write_scenario,
)


def test_transformed_rollforward_export_carries_one_verified_world(tmp_path: Path) -> None:
    """The next opening must be the transformed run's controls and verified close."""

    baseline = _baseline_with_dates(tmp_path)
    scenario_path = _write_scenario(
        tmp_path / "transformed-rollforward.json",
        baseline,
        overrides={
            "primary_deficit": {"mode": "scale_path", "scale": 1.25},
            "debt_target": {"mode": "scale_path", "scale": 1.05},
            "fed_holdings": {"mode": "scale_path", "scale": 1.10},
        },
    )
    scenario = read_json(scenario_path)
    scenario["simulation"]["end_date"] = "2026-09-25"
    scenario["output"]["compression"] = "gzip"
    write_json(scenario_path, scenario)
    run = run_cbo_scenario(
        baseline,
        CboScenarioSpec.from_file(scenario_path),
        tmp_path / "transformed-run",
        output_profile="compact",
    )
    window = ForecastStateWindow(
        state_period=2029,
        state_id="sentinel_state::2028_to_2029_to_2031",
        opening_state_date="2026-09-25",
        horizon_end_date="2026-09-30",
    )
    exported = export_forecast_state_package(
        baseline,
        state_window=window,
        rollforward_run_dir=run.output_dir,
        output_zip=tmp_path / "state.zip",
        output_attestation=tmp_path / "state.attestation.json",
        output_manifest=tmp_path / "state.export.json",
    )

    package_dir = tmp_path / "exported"
    with zipfile.ZipFile(exported.package_zip) as archive:
        archive.extractall(package_dir)
    exported_inputs = package_dir / "forecast_inputs"

    for filename in (
        "tdcsim_primary_deficit_path.csv",
        "tdcsim_debt_stock_path.csv",
        "tdcsim_fed_holdings_path.csv",
    ):
        compiled = pd.read_csv(run.compiled.forecast_inputs_dir / filename)
        carried = pd.read_csv(exported_inputs / filename)
        assert "scenario_transform" in carried.columns, (
            f"{filename} reverted to parent controls"
        )
        assert set(carried["scenario_transform"].dropna()) == set(
            compiled["scenario_transform"].dropna()
        ), f"{filename} reverted to parent controls"

    closing = pd.read_csv(run.results_path).iloc[-1]
    opening_state = read_json(exported_inputs / "tdcsim_opening_runtime_state.json")
    assert opening_state["initial_values"] == pytest.approx(
        {
            "tga": float(closing["TGA"]),
            "reserves": float(closing["Reserves"]),
            "tdc_level": float(closing["TDC_Level"]),
        }
    )

    final_portfolio = pd.read_csv(
        run.output_dir / "outputs" / "final_portfolio_compact.csv.gz"
    )
    final_fed_stock = float(
        final_portfolio.loc[
            final_portfolio["Status"].eq("Active")
            & final_portfolio["HolderType"].eq("CB"),
            "FaceValue",
        ].sum()
    )
    assert opening_state["fed_state"]["holdings_bil"] == pytest.approx(final_fed_stock)
    assert opening_state["fed_state"]["target_bil"] == pytest.approx(
        float(
            pd.read_csv(exported_inputs / "tdcsim_fed_holdings_path.csv").iloc[0][
                "cbo_fed_holdings_target_bil"
            ]
        )
    )

    derived = read_json(package_dir / "manifest.json")["derived_forecast_state"]
    assert derived["construction_kind"] == "transformed_scenario"
    assert derived["compiled_controls_digest"] == run.compiled.compiled_inputs_digest

    derived_baseline = CboBaselinePackage.open(
        exported.package_zip,
        attestation_path=exported.attestation_path,
    )
    next_scenario_path = _write_scenario(
        tmp_path / "continued-rollforward.json",
        derived_baseline,
        overrides={},
    )
    next_scenario = read_json(next_scenario_path)
    next_scenario["simulation"] = {
        "frequency": "daily",
        "start_date": window.opening_state_date,
        "end_date": "2026-09-29",
    }
    next_scenario["output"]["compression"] = "gzip"
    write_json(next_scenario_path, next_scenario)
    next_run = run_cbo_scenario(
        derived_baseline,
        CboScenarioSpec.from_file(next_scenario_path),
        tmp_path / "continued-run",
        output_profile="compact",
    )
    continued_results = pd.read_csv(next_run.results_path)
    for output_column, state_key in (
        ("TGA", "tga"),
        ("Reserves", "reserves"),
        ("TDC_Level", "tdc_level"),
    ):
        assert float(continued_results.iloc[0][output_column]) == pytest.approx(
            float(opening_state["initial_values"][state_key])
        )

    final_window = ForecastStateWindow(
        state_period=2031,
        state_id="sentinel_state::2031_from_2029",
        opening_state_date="2026-09-29",
        horizon_end_date="2026-09-30",
    )
    continued_export = export_forecast_state_package(
        derived_baseline,
        state_window=final_window,
        rollforward_run_dir=next_run.output_dir,
        output_zip=tmp_path / "continued-state.zip",
        output_attestation=tmp_path / "continued-state.attestation.json",
        output_manifest=tmp_path / "continued-state.export.json",
    )
    continued_package = tmp_path / "continued-exported"
    with zipfile.ZipFile(continued_export.package_zip) as archive:
        archive.extractall(continued_package)
    continued_opening = read_json(
        continued_package
        / "forecast_inputs"
        / "tdcsim_opening_runtime_state.json"
    )
    continued_closing = continued_results.iloc[-1]
    assert continued_opening["initial_values"] == pytest.approx(
        {
            "tga": float(continued_closing["TGA"]),
            "reserves": float(continued_closing["Reserves"]),
            "tdc_level": float(continued_closing["TDC_Level"]),
        }
    )
    for state_key in ("tga", "reserves", "tdc_level"):
        assert continued_opening["initial_values"][state_key] != pytest.approx(
            opening_state["initial_values"][state_key]
        )
    for filename in (
        "tdcsim_primary_deficit_path.csv",
        "tdcsim_debt_stock_path.csv",
        "tdcsim_fed_holdings_path.csv",
    ):
        continued_control = pd.read_csv(
            continued_package / "forecast_inputs" / filename
        )
        assert set(continued_control["scenario_transform"].dropna()) == set(
            pd.read_csv(run.compiled.forecast_inputs_dir / filename)[
                "scenario_transform"
            ].dropna()
        )
    continued_derived = read_json(continued_package / "manifest.json")[
        "derived_forecast_state"
    ]
    assert continued_derived["construction_kind"] == "transformed_scenario"
    assert continued_derived["parent_transforms_digest"] == derived[
        "transforms_digest"
    ]

    transformed_control = (
        run.compiled.forecast_inputs_dir / "tdcsim_primary_deficit_path.csv"
    )
    original_control = transformed_control.read_bytes()
    baseline_replacement = (
        baseline.materialize(tmp_path / "parent-copy")
        / "forecast_inputs"
        / transformed_control.name
    ).read_bytes()
    for tamper_kind in ("deleted", "altered", "replaced"):
        if tamper_kind == "deleted":
            transformed_control.unlink()
        elif tamper_kind == "altered":
            transformed_control.write_bytes(original_control + b"\n")
        else:
            transformed_control.write_bytes(baseline_replacement)
        with pytest.raises(
            ForecastStateExportError,
            match="rollforward run verification failed",
        ):
            export_forecast_state_package(
                baseline,
                state_window=window,
                rollforward_run_dir=run.output_dir,
                output_zip=tmp_path / f"{tamper_kind}.zip",
                output_attestation=tmp_path / f"{tamper_kind}.attestation.json",
                output_manifest=tmp_path / f"{tamper_kind}.export.json",
            )
        transformed_control.write_bytes(original_control)


def test_no_shock_forecast_state_caller_keeps_source_grade_contract(
    tmp_path: Path,
) -> None:
    baseline = _baseline_with_dates(tmp_path)
    window = ForecastStateWindow(
        state_period=2029,
        state_id="cbo_baseline_state::2029",
        opening_state_date="2026-09-25",
        horizon_end_date="2026-09-30",
    )
    run_dir = run_no_shock_rollforward(
        baseline,
        state_window=window,
        output_dir=tmp_path / "rollforward",
        scenario_dir=tmp_path / "scenarios",
    )
    exported = export_forecast_state_package(
        baseline,
        state_window=window,
        rollforward_run_dir=run_dir,
        output_zip=tmp_path / "baseline-state.zip",
        output_attestation=tmp_path / "baseline-state.attestation.json",
        output_manifest=tmp_path / "baseline-state.export.json",
    )
    export_manifest = read_json(exported.export_manifest_path)
    assert exported.construction_kind == "baseline_noop"
    assert export_manifest["construction_kind"] == "baseline_noop"
    assert export_manifest["construction_method"] == "baseline_rollforward_export_v1"
    assert export_manifest["claim_boundary"].startswith(
        "source_grade_cbo_baseline_rollforward"
    )


def _baseline_with_dates(tmp_path: Path) -> CboBaselinePackage:
    package, attestation_path = _write_runner_package(tmp_path)
    package_dir = tmp_path / "runner_pkg"
    manifest_path = package_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["forecast_publication_date"] = "2026-09-20"
    manifest["date_range"] = {
        "actuals_available_as_of": "2026-09-20",
        "opening_state_date": "2026-09-20",
        "simulation_start_date": "2026-09-20",
    }
    write_json(manifest_path, manifest)
    write_json(package_dir / "forecast_inputs" / "source_contract_smoke.json", manifest)
    package.unlink()
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())
    attestation = read_json(attestation_path)
    attestation["baseline_package_zip_sha256"] = sha256_file(package)
    attestation["baseline_manifest_sha256"] = sha256_file(manifest_path)
    attestation["source_contract_sha256"] = sha256_file(
        package_dir / "forecast_inputs" / "source_contract_smoke.json"
    )
    write_json(attestation_path, attestation)
    return CboBaselinePackage.open(package, attestation_path=attestation_path)
