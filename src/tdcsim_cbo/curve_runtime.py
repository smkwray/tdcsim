"""Runtime and receipt binding for the OPEN-04 evaluated nominal curve."""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

import sim_pricing
from evaluated_nominal_curve import (
    NOMINAL_EVALUATED_SHOCK_FILE,
    CurveContractError,
    EvaluatedNominalShock,
    evaluated_nominal_contract_metadata,
    load_evaluated_nominal_shock,
)
from yield_curve_path import load_yield_curve_surface

from ._json import canonical_json_bytes, read_json, sha256_file


COMPILED_MANIFEST_FILE = "tdcsim_cbo_compiled_manifest.json"
NOMINAL_SURFACE_FILE = "tdcsim_yield_curve_surface.csv"
RUNTIME_BINDING_SCHEMA = "tdcsim_evaluated_nominal_curve_runtime_v1"
EVIDENCE_GRID_SPEC_ID = "open04_evaluated_curve_grid_v1"


class EvaluatedNominalRuntimeError(ValueError):
    """Raised when a compiled evaluated-curve contract is not runtime-safe."""


def load_compiled_evaluated_nominal_contract(
    inputs_dir: str | Path,
) -> tuple[EvaluatedNominalShock | None, dict[str, Any] | None]:
    """Load an optional sidecar and bind it to its compiled manifest.

    The manifest hash is intentionally checked here, before the engine starts.
    A sidecar that remains semantically parseable after even a whitespace-only
    byte mutation is therefore still rejected.
    """

    inputs = Path(inputs_dir)
    sidecar_path = inputs / NOMINAL_EVALUATED_SHOCK_FILE
    manifest_path = inputs.parent / COMPILED_MANIFEST_FILE
    if not manifest_path.exists():
        if sidecar_path.exists():
            raise EvaluatedNominalRuntimeError(
                "evaluated nominal sidecar requires its compiled manifest"
            )
        return None, None
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise EvaluatedNominalRuntimeError("compiled manifest must be an object")
    expected = manifest.get("evaluated_nominal_curve")
    if expected is None and not sidecar_path.exists():
        return None, None
    if not isinstance(expected, Mapping):
        raise EvaluatedNominalRuntimeError(
            "compiled manifest evaluated_nominal_curve block is missing"
        )
    if not sidecar_path.exists():
        raise EvaluatedNominalRuntimeError(
            "compiled evaluated nominal sidecar is missing"
        )
    surface_path = inputs / NOMINAL_SURFACE_FILE
    try:
        actual = evaluated_nominal_contract_metadata(
            sidecar_path, surface_path
        )
        shock = load_evaluated_nominal_shock(sidecar_path, surface_path)
    except (CurveContractError, FileNotFoundError, TypeError, ValueError) as exc:
        raise EvaluatedNominalRuntimeError(
            f"evaluated nominal contract is invalid: {exc}"
        ) from exc
    if dict(expected) != actual:
        raise EvaluatedNominalRuntimeError(
            "compiled evaluated nominal metadata does not match runtime bytes"
        )
    return shock, actual


def build_evaluated_nominal_runtime_binding(
    inputs_dir: str | Path,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, Any] | None:
    """Recompute the bounded evaluated-curve evidence used by a run receipt."""

    inputs = Path(inputs_dir)
    shock, _compiled_metadata = load_compiled_evaluated_nominal_contract(inputs)
    if shock is None:
        return None
    sidecar_path = inputs / NOMINAL_EVALUATED_SHOCK_FILE
    surface_path = inputs / NOMINAL_SURFACE_FILE
    sidecar = read_json(sidecar_path)
    if not isinstance(sidecar, Mapping):
        raise EvaluatedNominalRuntimeError("evaluated nominal sidecar must be an object")
    evidence = _stream_evaluated_curve_evidence(
        surface_path,
        shock,
        start_date=start_date,
        end_date=end_date,
    )
    evaluator = sidecar.get("baseline_evaluator")
    if not isinstance(evaluator, Mapping):
        raise EvaluatedNominalRuntimeError(
            "evaluated nominal sidecar baseline_evaluator must be an object"
        )
    return {
        "schema_version": RUNTIME_BINDING_SCHEMA,
        "contract_id": str(sidecar.get("contract_id") or ""),
        "sidecar_sha256": sha256_file(sidecar_path),
        "baseline_surface_sha256": sha256_file(surface_path),
        "interpolation_method": str(
            evaluator.get("interpolation_method") or ""
        ),
        "floor_zero": evaluator.get("floor_zero"),
        "endpoint_behavior": str(evaluator.get("endpoint_behavior") or ""),
        **evidence,
    }


def _stream_evaluated_curve_evidence(
    surface_path: Path,
    shock: EvaluatedNominalShock,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Hash a fixed grid one curve date at a time without retaining records."""

    surface = load_yield_curve_surface(surface_path)
    scenario_ids = sorted(
        set(surface["scenario_id"].dropna().astype(str).tolist())
    )
    if len(scenario_ids) != 1:
        raise EvaluatedNominalRuntimeError(
            "evaluated nominal surface must contain exactly one scenario_id"
        )
    selected_dates = _runtime_selected_curve_dates(
        surface, start_date=start_date, end_date=end_date
    )
    grid = _evidence_grid(surface)
    header = {
        "schema_version": "tdcsim_evaluated_nominal_curve_evidence_v1",
        "grid_spec_id": EVIDENCE_GRID_SPEC_ID,
        "record_count": len(selected_dates) * len(grid),
        "grid": {
            "short_linear_start_years": 0.0,
            "short_linear_end_years": 2.0,
            "short_linear_count": 8193,
            "long_log_start": "nextafter_2y_toward_positive_infinity",
            "long_log_end_years": 31.0,
            "long_log_count": 2049,
            "includes_surface_tenors_and_adjacent_binary64_points": True,
        },
        "record_order": ["curve_date", "maturity_binary64"],
        "record_fields": [
            "maturity_years",
            "baseline_yield_decimal",
            "scenario_yield_decimal",
            "scenario_minus_baseline_decimal",
        ],
        "signed_10y_shock_bp": float(shock.signed_10y_shock_bp),
    }
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes(header))
    digest.update(b"\n")
    short_mismatches = 0
    max_delta_error = 0.0
    scenario_id = scenario_ids[0]
    for curve_date in selected_dates:
        rows = surface[
            (surface["scenario_id"].astype(str) == scenario_id)
            & (surface["curve_date"] == curve_date)
        ].sort_values("tenor_years")
        if rows.empty:
            raise EvaluatedNominalRuntimeError(
                f"selected curve date has no rows: {curve_date.date()}"
            )
        years = rows["tenor_years"].to_numpy(dtype=float)
        rates = rows["nominal_rate_decimal"].to_numpy(dtype=float)
        baseline = _baseline_pchip_vector(grid, years, rates)
        expected_delta = _analytic_delta_vector(grid, shock)
        scenario = np.fromiter(
            (
                sim_pricing.evaluate_nominal_yield(
                    float(maturity),
                    years,
                    rates,
                    method="pchip",
                    floor_zero=False,
                    shock=shock,
                )
                for maturity in grid
            ),
            dtype=float,
            count=len(grid),
        )
        observed_delta = scenario - baseline
        long_mask = grid > 2.0
        short_mask = ~long_mask
        short_mismatches += int(
            np.count_nonzero(
                baseline[short_mask].view(np.uint64)
                != scenario[short_mask].view(np.uint64)
            )
        )
        max_delta_error = max(
            max_delta_error,
            float(np.max(np.abs(observed_delta - expected_delta))),
        )
        date_bytes = pd.Timestamp(curve_date).date().isoformat().encode("ascii")
        for maturity, base, candidate, delta in zip(
            grid, baseline, scenario, observed_delta, strict=True
        ):
            digest.update(date_bytes)
            digest.update(
                struct.pack(
                    ">dddd",
                    float(maturity),
                    float(base),
                    float(candidate),
                    float(delta),
                )
            )
    selected_iso = [
        pd.Timestamp(value).date().isoformat() for value in selected_dates
    ]
    return {
        "runtime_selected_curve_date_count": len(selected_iso),
        "runtime_selected_curve_date_set_sha256": hashlib.sha256(
            canonical_json_bytes(selected_iso)
        ).hexdigest(),
        "grid_spec_id": EVIDENCE_GRID_SPEC_ID,
        "grid_record_count": len(selected_iso) * len(grid),
        "evaluated_delta_sha256": digest.hexdigest(),
        "short_end_bitwise_mismatch_count": short_mismatches,
        "max_abs_analytic_delta_error_decimal": max_delta_error,
    }


def _runtime_selected_curve_dates(
    surface: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise EvaluatedNominalRuntimeError(
            "runtime curve evidence end_date precedes start_date"
        )
    stored = sorted(
        pd.Timestamp(value).normalize()
        for value in surface["curve_date"].drop_duplicates().tolist()
    )
    prior = [value for value in stored if value <= start]
    if not prior:
        raise EvaluatedNominalRuntimeError(
            "yield surface has no curve date at or before runtime start"
        )
    selected = [prior[-1]]
    selected.extend(value for value in stored if start < value <= end)
    return selected


def _evidence_grid(surface: pd.DataFrame) -> np.ndarray:
    short = np.linspace(0.0, 2.0, 8193, dtype=float)
    long = np.geomspace(
        np.nextafter(2.0, math.inf),
        31.0,
        2049,
        dtype=float,
    )
    tenors = sorted(
        set(float(value) for value in surface["tenor_years"].tolist())
    )
    boundary_points: set[float] = {2.0, 5.0, 10.0, 20.0, 30.0, 31.0}
    for value in {*tenors, 2.0, 10.0, 30.0}:
        boundary_points.add(float(value))
        boundary_points.add(float(np.nextafter(value, -math.inf)))
        boundary_points.add(float(np.nextafter(value, math.inf)))
    merged = np.concatenate(
        [
            short,
            long,
            np.asarray(sorted(boundary_points), dtype=float),
        ]
    )
    return np.unique(merged)


def _baseline_pchip_vector(
    maturities: np.ndarray,
    years: np.ndarray,
    rates: np.ndarray,
) -> np.ndarray:
    if (
        len(years) < 2
        or len(years) != len(rates)
        or not np.all(np.isfinite(years))
        or not np.all(np.isfinite(rates))
        or not np.all(years[1:] > years[:-1])
    ):
        raise EvaluatedNominalRuntimeError(
            "yield surface date has an invalid PCHIP input"
        )
    interpolator = PchipInterpolator(years, rates)
    values = np.asarray(interpolator(maturities), dtype=float)
    values[maturities <= years[0]] = rates[0]
    values[maturities >= years[-1]] = rates[-1]
    if not np.all(np.isfinite(values)):
        raise EvaluatedNominalRuntimeError(
            "yield surface produced a nonfinite baseline value"
        )
    return values


def _analytic_delta_vector(
    maturities: np.ndarray,
    shock: EvaluatedNominalShock,
) -> np.ndarray:
    values = np.zeros_like(maturities, dtype=float)
    middle = (maturities > 2.0) & (maturities < 10.0)
    values[middle] = (
        float(shock.signed_10y_shock_bp)
        * np.log(maturities[middle] / 2.0)
        / math.log(5.0)
        / 10_000.0
    )
    values[maturities >= 10.0] = (
        float(shock.signed_10y_shock_bp) / 10_000.0
    )
    return values


__all__ = [
    "EVIDENCE_GRID_SPEC_ID",
    "EvaluatedNominalRuntimeError",
    "build_evaluated_nominal_runtime_binding",
    "load_compiled_evaluated_nominal_contract",
]
