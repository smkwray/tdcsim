import zipfile
from pathlib import Path

import pandas as pd
import pytest

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, run_cbo_scenario
from tdcsim_cbo._json import read_json, sha256_file, write_json
from tdcsim_cbo.forecast_state import (
    ForecastStateWindow,
    export_forecast_state_package,
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
