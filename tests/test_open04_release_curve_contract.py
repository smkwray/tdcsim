import csv
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy

from evaluated_nominal_curve import (
    NOMINAL_EVALUATED_SHOCK_FILE,
    EvaluatedNominalShock,
    build_evaluated_nominal_sidecar,
    evaluated_nominal_contract_metadata,
)
from tdcsim_cbo._json import sha256_file, write_json
from tdcsim_cbo.curve_runtime import (
    COMPILED_MANIFEST_FILE,
    _baseline_pchip_vector,
    _evidence_grid,
    build_evaluated_nominal_runtime_binding,
)
from yield_curve_path import load_yield_curve_surface


_RELEASE_PACKAGE = "output/cbo_forecast_release_bound_package.zip"
_RELEASE_ATTESTATION = "output/cbo_forecast_release_bound_attestation.json"
_REQUIREMENTS_LOCK = "requirements.lock.txt"
_PACKAGE_SHA256 = (
    "be49f5a5d256863649ccf1b139d259679c9a3ce669642751c22e8a7dad9a2d2c"
)
_ATTESTATION_SHA256 = (
    "d30a6f89263cdb80f8f9d81131dc0004b7b9751289e0594ee5005e115641124a"
)
_LOCK_SHA256 = (
    "16e3fc32257a01e1fd2e5a53867cc9e73496b9d35c6a5d4db19c671617325a4e"
)
_SURFACE_MEMBER = "forecast_inputs/tdcsim_yield_curve_surface.csv"
_SURFACE_SHA256 = (
    "3e950b670653ce5a75779cac099ac6ed0a2e8454e041e9b4f1bac798d0906381"
)
_SELECTED_DATE_SET_SHA256 = (
    "0e0b8049b03260ab243442e34b970ad0e74076d26be7810639d36c566ef2d1fc"
)
_EVALUATED_DELTA_SHA256 = (
    "b1b445bbe3fcceb004ecfa23fafc9af3418da3e9dc822bdc2a32c7f15360e0b4"
)


def _extract_release_surface(tmp_path: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    package_path = project_root / _RELEASE_PACKAGE
    assert package_path.is_file(), (
        f"required release package is missing: {package_path}"
    )
    with zipfile.ZipFile(package_path) as archive:
        assert _SURFACE_MEMBER in archive.namelist(), (
            f"release package is missing exact member {_SURFACE_MEMBER!r}"
        )
        return Path(archive.extract(_SURFACE_MEMBER, tmp_path))


@pytest.mark.integration
def test_open04_all_release_dates_satisfy_static_baseline_a_b_contract(
    tmp_path: Path,
) -> None:
    surface_path = _extract_release_surface(tmp_path / "static")
    assert sha256_file(surface_path) == _SURFACE_SHA256
    surface = load_yield_curve_surface(surface_path)
    curve_dates = sorted(surface["curve_date"].drop_duplicates().tolist())
    assert len(curve_dates) == 56

    # The full runtime evidence grid is evaluated one date at a time. The
    # additional point is far enough above 2y for the correctly signed scenario
    # delta to survive binary64 rounding; nextafter(2y) remains in the grid as a
    # boundary-continuity check.
    grid = np.unique(
        np.concatenate(
            [
                _evidence_grid(surface),
                np.asarray([2.0 + 1e-8], dtype=float),
            ]
        )
    )
    short_mask = grid <= 2.0
    middle_mask = (grid > 2.0) & (grid < 10.0)
    upper_mask = grid >= 10.0
    at_2y = int(np.flatnonzero(grid == 2.0)[0])
    above_2y = int(np.flatnonzero(grid == 2.0 + 1e-8)[0])
    at_10y = int(np.flatnonzero(grid == 10.0)[0])
    shock_a = EvaluatedNominalShock(-25.0)
    shock_b = EvaluatedNominalShock(25.0)
    expected_a = np.fromiter(
        (shock_a.shock_bp(float(maturity)) / 10_000.0 for maturity in grid),
        dtype=float,
        count=len(grid),
    )
    expected_b = np.fromiter(
        (shock_b.shock_bp(float(maturity)) / 10_000.0 for maturity in grid),
        dtype=float,
        count=len(grid),
    )
    assert np.array_equal(expected_a, -expected_b)
    assert expected_a[at_2y] == expected_b[at_2y] == 0.0
    assert expected_a[at_10y] == -0.0025
    assert expected_b[at_10y] == 0.0025
    assert np.all(expected_a[upper_mask] == -0.0025)
    assert np.all(expected_b[upper_mask] == 0.0025)

    scenario_ids = set(surface["scenario_id"].dropna().astype(str).tolist())
    assert len(scenario_ids) == 1
    scenario_id = next(iter(scenario_ids))
    for curve_date in curve_dates:
        rows = surface[
            (surface["scenario_id"].astype(str) == scenario_id)
            & (surface["curve_date"] == curve_date)
        ].sort_values("tenor_years")
        assert len(rows) == 9
        baseline = _baseline_pchip_vector(
            grid,
            rows["tenor_years"].to_numpy(dtype=float),
            rows["nominal_rate_decimal"].to_numpy(dtype=float),
        )
        candidate_a = np.fromiter(
            (
                shock_a.apply_to_baseline(maturity, base)
                for maturity, base in zip(grid, baseline, strict=True)
            ),
            dtype=float,
            count=len(grid),
        )
        candidate_b = np.fromiter(
            (
                shock_b.apply_to_baseline(maturity, base)
                for maturity, base in zip(grid, baseline, strict=True)
            ),
            dtype=float,
            count=len(grid),
        )

        # The baseline object is returned without scenario arithmetic through 2y.
        assert np.array_equal(
            candidate_a[short_mask].view(np.uint64),
            baseline[short_mask].view(np.uint64),
        )
        assert np.array_equal(
            candidate_b[short_mask].view(np.uint64),
            baseline[short_mask].view(np.uint64),
        )
        observed_a = candidate_a - baseline
        observed_b = candidate_b - baseline
        assert float(np.max(np.abs(observed_a - expected_a))) <= 1e-12
        assert float(np.max(np.abs(observed_b - expected_b))) <= 1e-12
        assert observed_a[at_2y] == observed_b[at_2y] == 0.0
        assert observed_a[above_2y] < 0.0
        assert observed_b[above_2y] > 0.0
        assert observed_a[at_10y] == pytest.approx(-0.0025, abs=1e-15)
        assert observed_b[at_10y] == pytest.approx(0.0025, abs=1e-15)
        assert float(np.max(np.abs(observed_a[upper_mask] + 0.0025))) <= 1e-15
        assert float(np.max(np.abs(observed_b[upper_mask] - 0.0025))) <= 1e-15
        assert np.all(observed_a[middle_mask] <= 0.0)
        assert np.all(observed_b[middle_mask] >= 0.0)
        assert float(np.max(np.abs(observed_a + observed_b))) <= 1e-15


@pytest.mark.integration
def test_open04_release_surface_satisfies_evaluated_curve_contract(
    tmp_path: Path,
    record_property,
) -> None:
    compiled_dir = tmp_path / "compiled"
    surface_path = _extract_release_surface(compiled_dir)

    assert sha256_file(surface_path) == _SURFACE_SHA256
    with surface_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 504
    assert len({row["curve_date"] for row in rows}) == 56
    tenors = {float(row["tenor_years"]) for row in rows}
    assert tenors == {
        1.0 / 12.0,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        30.0,
    }

    inputs_dir = compiled_dir / "forecast_inputs"
    override = {
        "mode": "evaluated_additive_key_rate_bp",
        "application": "post_baseline_evaluation",
        "interpolation": "log_tenor_linear",
        "lower_endpoint": "zero_at_or_below_first_key",
        "upper_endpoint": "flat_at_or_above_last_key",
        "time_profile": "constant_across_curve_dates",
        "compounding": "none",
        "shocks": [
            {"tenor_years": 2.0, "shock_bp": 0.0},
            {"tenor_years": 10.0, "shock_bp": -25.0},
        ],
    }
    sidecar_path = inputs_dir / NOMINAL_EVALUATED_SHOCK_FILE
    write_json(
        sidecar_path,
        build_evaluated_nominal_sidecar(override, surface_path),
    )
    metadata = evaluated_nominal_contract_metadata(
        sidecar_path,
        surface_path,
    )
    write_json(
        compiled_dir / COMPILED_MANIFEST_FILE,
        {
            "schema_version": "tdcsim_cbo_compiled_scenario_manifest_v1",
            "evaluated_nominal_curve": metadata,
        },
    )

    binding = build_evaluated_nominal_runtime_binding(
        inputs_dir,
        start_date="2026-06-21",
        end_date="2036-09-30",
    )
    assert binding is not None
    assert binding["baseline_surface_sha256"] == _SURFACE_SHA256
    assert binding["runtime_selected_curve_date_count"] == 42
    assert (
        binding["runtime_selected_curve_date_set_sha256"]
        == _SELECTED_DATE_SET_SHA256
    )
    assert binding["short_end_bitwise_mismatch_count"] == 0
    assert binding["max_abs_analytic_delta_error_decimal"] <= 1e-12
    assert binding["grid_record_count"] == 431_088
    assert binding["grid_record_count"] < 500_000
    assert binding["evaluated_delta_sha256"] == _EVALUATED_DELTA_SHA256

    project_root = Path(__file__).resolve().parents[1]
    assert sha256_file(project_root / _RELEASE_PACKAGE) == _PACKAGE_SHA256
    assert (
        sha256_file(project_root / _RELEASE_ATTESTATION)
        == _ATTESTATION_SHA256
    )
    assert sha256_file(project_root / _REQUIREMENTS_LOCK) == _LOCK_SHA256
    for name, value in (
        ("open04_baseline_package_sha256", _PACKAGE_SHA256),
        ("open04_baseline_attestation_sha256", _ATTESTATION_SHA256),
        ("open04_requirements_lock_sha256", _LOCK_SHA256),
        ("open04_release_surface_sha256", _SURFACE_SHA256),
        ("open04_release_curve_date_count", 56),
        ("open04_runtime_selected_curve_date_count", 42),
        (
            "open04_runtime_selected_curve_date_set_sha256",
            _SELECTED_DATE_SET_SHA256,
        ),
        ("open04_grid_record_count", binding["grid_record_count"]),
        (
            "open04_short_end_bitwise_mismatch_count",
            binding["short_end_bitwise_mismatch_count"],
        ),
        (
            "open04_max_abs_analytic_delta_error_decimal",
            binding["max_abs_analytic_delta_error_decimal"],
        ),
        (
            "open04_evaluated_delta_sha256",
            binding["evaluated_delta_sha256"],
        ),
        ("open04_python_version", platform.python_version()),
        ("open04_scipy_version", scipy.__version__),
        ("open04_numpy_version", np.__version__),
        ("open04_pandas_version", pd.__version__),
    ):
        record_property(name, value)
