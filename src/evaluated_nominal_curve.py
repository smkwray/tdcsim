"""Canonical OPEN-04 evaluated nominal-curve shock contract.

The baseline nominal surface remains byte-identical.  This module represents the
approved scenario delta as a small analytic sidecar that is applied only after
baseline evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any


NOMINAL_EVALUATED_SHOCK_FILE = "tdcsim_nominal_curve_evaluated_shock.json"
OPEN04_FIXED_COUPLING = MappingProxyType(
    {
        "frn_benchmark": "independent_explicit_path",
        "tips_real_yield": "independent_explicit_path",
        "operating_cash_inflation": "baseline_cpi",
        "primary_deficit_to_debt_target": "independent_no_plug",
    }
)
OPEN04_OUTPUT_CONTRACT = MappingProxyType(
    {
        "profile": "compact",
        "compression": "gzip",
    }
)

_SIDECAR_SCHEMA_VERSION = "tdcsim_nominal_curve_evaluated_shock_v1"
_METADATA_SCHEMA_VERSION = "tdcsim_nominal_curve_evaluated_contract_metadata_v1"
_CONTRACT_ID = "open04_log_tenor_2y10y_25bp_v1"
_BASELINE_SURFACE_FILE = "tdcsim_yield_curve_surface.csv"

_OVERRIDE_FIELDS = frozenset(
    {
        "mode",
        "application",
        "interpolation",
        "lower_endpoint",
        "upper_endpoint",
        "time_profile",
        "compounding",
        "shocks",
    }
)
_SIDECAR_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "application",
        "rate_unit",
        "interpolation",
        "lower_endpoint",
        "upper_endpoint",
        "time_profile",
        "compounding",
        "key_rates",
        "baseline_surface",
        "baseline_evaluator",
    }
)
_BASELINE_SURFACE_FIELDS = frozenset(
    {
        "relative_path",
        "sha256",
        "row_count",
        "curve_date_count",
        "tenor_set",
    }
)
_BASELINE_EVALUATOR = {
    "interpolation_method": "pchip",
    "floor_zero": False,
    "endpoint_behavior": "clamp",
}
_OVERRIDE_FIXED_VALUES = {
    "mode": "evaluated_additive_key_rate_bp",
    "application": "post_baseline_evaluation",
    "interpolation": "log_tenor_linear",
    "lower_endpoint": "zero_at_or_below_first_key",
    "upper_endpoint": "flat_at_or_above_last_key",
    "time_profile": "constant_across_curve_dates",
    "compounding": "none",
}
_SIDECAR_FIXED_VALUES = {
    "schema_version": _SIDECAR_SCHEMA_VERSION,
    "contract_id": _CONTRACT_ID,
    "application": "post_baseline_evaluation",
    "rate_unit": "basis_points",
    "interpolation": "log_tenor_linear",
    "lower_endpoint": "zero_at_or_below_2y",
    "upper_endpoint": "flat_at_or_above_10y",
    "time_profile": "constant_across_curve_dates",
    "compounding": "none",
}


class CurveContractError(ValueError):
    """Raised when an evaluated nominal-curve contract fails closed."""


@dataclass(frozen=True)
class EvaluatedNominalShock:
    """The approved signed OPEN-04 nominal-yield shock in basis points."""

    signed_10y_shock_bp: float

    def __post_init__(self) -> None:
        value = _finite_number(
            self.signed_10y_shock_bp,
            label="signed_10y_shock_bp",
        )
        if value not in {-25.0, 25.0}:
            raise CurveContractError("signed_10y_shock_bp must be exactly -25.0 or 25.0")
        object.__setattr__(self, "signed_10y_shock_bp", value)

    def shock_bp(self, maturity_years: float) -> float:
        """Return the signed analytic shock at ``maturity_years``."""

        maturity = _finite_number(maturity_years, label="maturity_years")
        if maturity < 0.0:
            raise CurveContractError("maturity_years must be nonnegative")
        if maturity <= 2.0:
            return 0.0
        if maturity >= 10.0:
            return self.signed_10y_shock_bp
        return (
            self.signed_10y_shock_bp
            * math.log(maturity / 2.0)
            / math.log(5.0)
        )

    def apply_to_baseline(
        self,
        maturity_years: float,
        baseline_yield_decimal: float,
    ) -> float:
        """Apply the shock after baseline evaluation with exact short-end identity."""

        maturity = _finite_number(maturity_years, label="maturity_years")
        baseline = _finite_number(
            baseline_yield_decimal,
            label="baseline_yield_decimal",
        )
        if maturity < 0.0:
            raise CurveContractError("maturity_years must be nonnegative")
        if maturity <= 2.0:
            return baseline_yield_decimal
        return baseline + self.shock_bp(maturity) / 10_000.0


def normalize_open04_override(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize the exact OPEN-04 scenario override."""

    override = _require_mapping(mapping, label="nominal_yield_curve override")
    _require_exact_fields(
        override,
        _OVERRIDE_FIELDS,
        label="nominal_yield_curve override",
    )
    for field, expected in _OVERRIDE_FIXED_VALUES.items():
        if override[field] != expected:
            raise CurveContractError(
                f"nominal_yield_curve override {field} must be {expected!r}"
            )
    key_rates = _normalize_key_rates(override["shocks"], label="override shocks")
    return {
        **_OVERRIDE_FIXED_VALUES,
        "shocks": key_rates,
    }


def build_evaluated_nominal_sidecar(
    override: Mapping[str, Any],
    baseline_surface_path: str | Path,
) -> dict[str, Any]:
    """Build the canonical sidecar without copying any baseline curve rows."""

    normalized = normalize_open04_override(override)
    surface = _surface_profile(baseline_surface_path)
    return {
        **_SIDECAR_FIXED_VALUES,
        "key_rates": normalized["shocks"],
        "baseline_surface": {
            "relative_path": _BASELINE_SURFACE_FILE,
            "sha256": surface["sha256"],
            "row_count": surface["row_count"],
            "curve_date_count": surface["curve_date_count"],
            "tenor_set": surface["tenor_set"],
        },
        "baseline_evaluator": dict(_BASELINE_EVALUATOR),
    }


def load_evaluated_nominal_shock(
    sidecar_path: str | Path,
    baseline_surface_path: str | Path,
) -> EvaluatedNominalShock:
    """Load a sidecar only after recomputing and checking its baseline binding."""

    sidecar = _load_and_validate_sidecar(sidecar_path, baseline_surface_path)
    return EvaluatedNominalShock(
        signed_10y_shock_bp=float(sidecar["key_rates"][1]["shock_bp"])
    )


def evaluated_nominal_contract_metadata(
    sidecar_path: str | Path,
    baseline_surface_path: str | Path,
) -> dict[str, Any]:
    """Return bounded manifest metadata after fully validating the sidecar."""

    sidecar_file = Path(sidecar_path)
    sidecar = _load_and_validate_sidecar(sidecar_file, baseline_surface_path)
    surface = _surface_profile(baseline_surface_path)
    method_contract = {
        key: sidecar[key]
        for key in (
            "schema_version",
            "contract_id",
            "application",
            "rate_unit",
            "interpolation",
            "lower_endpoint",
            "upper_endpoint",
            "time_profile",
            "compounding",
            "key_rates",
            "baseline_evaluator",
        )
    }
    return {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "contract_id": _CONTRACT_ID,
        "signed_10y_shock_bp": float(sidecar["key_rates"][1]["shock_bp"]),
        "sidecar": {
            "relative_path": sidecar_file.name,
            "sha256": _sha256_file(sidecar_file),
            "bytes": sidecar_file.stat().st_size,
            "canonical_sha256": _canonical_json_sha256(sidecar),
        },
        "baseline_surface": {
            "relative_path": _BASELINE_SURFACE_FILE,
            "sha256": surface["sha256"],
            "row_count": surface["row_count"],
            "curve_date_count": surface["curve_date_count"],
            "date_set_sha256": surface["date_set_sha256"],
            "tenor_count": len(surface["tenor_set"]),
            "tenor_set": surface["tenor_set"],
            "tenor_set_sha256": surface["tenor_set_sha256"],
        },
        "baseline_evaluator": dict(_BASELINE_EVALUATOR),
        "canonical_curve_method_sha256": _canonical_json_sha256(method_contract),
    }


def _load_and_validate_sidecar(
    sidecar_path: str | Path,
    baseline_surface_path: str | Path,
) -> dict[str, Any]:
    sidecar_file = Path(sidecar_path)
    if not sidecar_file.is_file():
        raise CurveContractError(f"evaluated nominal sidecar is missing: {sidecar_file}")
    sidecar = _read_json_object(sidecar_file)
    _require_exact_fields(sidecar, _SIDECAR_FIELDS, label="evaluated nominal sidecar")
    for field, expected in _SIDECAR_FIXED_VALUES.items():
        if sidecar[field] != expected:
            raise CurveContractError(
                f"evaluated nominal sidecar {field} must be {expected!r}"
            )

    key_rates = _normalize_key_rates(sidecar["key_rates"], label="sidecar key_rates")
    if sidecar["key_rates"] != key_rates:
        raise CurveContractError("evaluated nominal sidecar key_rates are not canonical")

    evaluator = _require_mapping(
        sidecar["baseline_evaluator"],
        label="sidecar baseline_evaluator",
    )
    _require_exact_fields(
        evaluator,
        frozenset(_BASELINE_EVALUATOR),
        label="sidecar baseline_evaluator",
    )
    if dict(evaluator) != _BASELINE_EVALUATOR:
        raise CurveContractError(
            "evaluated nominal sidecar baseline_evaluator does not match "
            "pchip/floor_zero=false/clamp"
        )

    declared_surface = _require_mapping(
        sidecar["baseline_surface"],
        label="sidecar baseline_surface",
    )
    _require_exact_fields(
        declared_surface,
        _BASELINE_SURFACE_FIELDS,
        label="sidecar baseline_surface",
    )
    if isinstance(declared_surface["row_count"], bool) or not isinstance(
        declared_surface["row_count"], int
    ):
        raise CurveContractError("sidecar baseline_surface.row_count must be an integer")
    if isinstance(declared_surface["curve_date_count"], bool) or not isinstance(
        declared_surface["curve_date_count"], int
    ):
        raise CurveContractError(
            "sidecar baseline_surface.curve_date_count must be an integer"
        )

    surface = _surface_profile(baseline_surface_path)
    expected_surface = {
        "relative_path": _BASELINE_SURFACE_FILE,
        "sha256": surface["sha256"],
        "row_count": surface["row_count"],
        "curve_date_count": surface["curve_date_count"],
        "tenor_set": surface["tenor_set"],
    }
    if dict(declared_surface) != expected_surface:
        raise CurveContractError(
            "evaluated nominal sidecar baseline_surface binding does not match actual bytes"
        )
    return dict(sidecar)


def _surface_profile(path: str | Path) -> dict[str, Any]:
    surface_path = Path(path)
    if surface_path.name != _BASELINE_SURFACE_FILE:
        raise CurveContractError(
            f"baseline surface must be named {_BASELINE_SURFACE_FILE}"
        )
    if not surface_path.is_file():
        raise CurveContractError(f"baseline surface is missing: {surface_path}")

    try:
        with surface_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            if len(fieldnames) != len(set(fieldnames)):
                raise CurveContractError("baseline surface has duplicate CSV columns")
            required = {
                "scenario_id",
                "curve_date",
                "tenor_years",
                "nominal_rate_decimal",
            }
            missing = sorted(required - set(fieldnames))
            if missing:
                raise CurveContractError(
                    f"baseline surface is missing required columns: {missing}"
                )
            rows = list(reader)
    except CurveContractError:
        raise
    except (OSError, csv.Error, UnicodeError) as exc:
        raise CurveContractError(f"baseline surface is unreadable: {surface_path}") from exc

    if not rows:
        raise CurveContractError("baseline surface has no data rows")
    scenarios: set[str] = set()
    tenor_rows_by_date: dict[str, list[float]] = {}
    seen_keys: set[tuple[str, float]] = set()
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise CurveContractError(
                f"baseline surface row {row_number} has extra CSV values"
            )
        scenario_id = str(row["scenario_id"]).strip()
        if not scenario_id:
            raise CurveContractError(
                f"baseline surface row {row_number} has an empty scenario_id"
            )
        scenarios.add(scenario_id)
        raw_date = str(row["curve_date"]).strip()
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise CurveContractError(
                f"baseline surface row {row_number} has an invalid curve_date"
            ) from exc
        if raw_date != parsed_date.isoformat():
            raise CurveContractError(
                f"baseline surface row {row_number} curve_date must be ISO-8601"
            )
        tenor = _finite_number(
            row["tenor_years"],
            label=f"baseline surface row {row_number} tenor_years",
        )
        if tenor <= 0.0:
            raise CurveContractError(
                f"baseline surface row {row_number} tenor_years must be positive"
            )
        _finite_number(
            row["nominal_rate_decimal"],
            label=f"baseline surface row {row_number} nominal_rate_decimal",
        )
        key = (raw_date, tenor)
        if key in seen_keys:
            raise CurveContractError(
                f"baseline surface has duplicate date/tenor key {raw_date}/{tenor}"
            )
        seen_keys.add(key)
        tenor_rows_by_date.setdefault(raw_date, []).append(tenor)

    if len(scenarios) != 1:
        raise CurveContractError("baseline surface must contain exactly one scenario_id")

    dates = sorted(tenor_rows_by_date)
    reference_tenors: tuple[float, ...] | None = None
    for curve_date in dates:
        tenors = tenor_rows_by_date[curve_date]
        if any(left >= right for left, right in zip(tenors, tenors[1:])):
            raise CurveContractError(
                f"baseline surface tenors must be strictly increasing at {curve_date}"
            )
        current = tuple(tenors)
        if reference_tenors is None:
            reference_tenors = current
        elif current != reference_tenors:
            raise CurveContractError(
                "baseline surface must use an identical tenor set at every curve_date"
            )
    assert reference_tenors is not None
    if 2.0 not in reference_tenors or 10.0 not in reference_tenors:
        raise CurveContractError(
            "baseline surface must contain exact 2-year and 10-year tenors"
        )
    tenor_set = [float(value) for value in reference_tenors]
    return {
        "sha256": _sha256_file(surface_path),
        "row_count": len(rows),
        "curve_date_count": len(dates),
        "date_set_sha256": _canonical_json_sha256(dates),
        "tenor_set": tenor_set,
        "tenor_set_sha256": _canonical_json_sha256(tenor_set),
    }


def _normalize_key_rates(value: Any, *, label: str) -> list[dict[str, float]]:
    if not isinstance(value, list) or len(value) != 2:
        raise CurveContractError(f"{label} must contain exactly two key rates")
    parsed: dict[float, float] = {}
    for index, raw in enumerate(value):
        row = _require_mapping(raw, label=f"{label}[{index}]")
        _require_exact_fields(
            row,
            frozenset({"tenor_years", "shock_bp"}),
            label=f"{label}[{index}]",
        )
        tenor = _finite_number(
            row["tenor_years"],
            label=f"{label}[{index}].tenor_years",
        )
        shock = _finite_number(
            row["shock_bp"],
            label=f"{label}[{index}].shock_bp",
        )
        if tenor in parsed:
            raise CurveContractError(f"{label} contains a duplicate tenor")
        parsed[tenor] = shock
    if set(parsed) != {2.0, 10.0}:
        raise CurveContractError(f"{label} tenors must be exactly 2.0 and 10.0")
    if parsed[2.0] != 0.0:
        raise CurveContractError(f"{label} 2-year shock must be exactly 0.0 bp")
    if parsed[10.0] not in {-25.0, 25.0}:
        raise CurveContractError(
            f"{label} 10-year shock must be exactly -25.0 or 25.0 bp"
        )
    return [
        {"tenor_years": 2.0, "shock_bp": 0.0},
        {"tenor_years": 10.0, "shock_bp": float(parsed[10.0])},
    ]


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CurveContractError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CurveContractError(f"{label} field names must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise CurveContractError(
            f"{label} fields must be exact; missing={missing}, extra={extra}"
        )


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise CurveContractError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CurveContractError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise CurveContractError(f"{label} must be a finite number")
    return number


def _read_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CurveContractError(
                    f"evaluated nominal sidecar has duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_raise_nonfinite_json(token)),
        )
    except CurveContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurveContractError(f"evaluated nominal sidecar is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CurveContractError("evaluated nominal sidecar must be a JSON object")
    return value


def _raise_nonfinite_json(token: str) -> None:
    raise CurveContractError(
        f"evaluated nominal sidecar contains nonfinite JSON number {token}"
    )


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CurveContractError("curve contract cannot be canonicalized") from exc
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CurveContractError(f"curve contract artifact is unreadable: {path}") from exc
    return digest.hexdigest()


__all__ = [
    "OPEN04_FIXED_COUPLING",
    "OPEN04_OUTPUT_CONTRACT",
    "NOMINAL_EVALUATED_SHOCK_FILE",
    "CurveContractError",
    "EvaluatedNominalShock",
    "build_evaluated_nominal_sidecar",
    "evaluated_nominal_contract_metadata",
    "load_evaluated_nominal_shock",
    "normalize_open04_override",
]
