"""Scenario overlay contract for CBO baseline runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from ._json import canonical_json_sha256, canonical_json_text, read_json, sha256_file
from ._schema import SchemaValidationError, validate_schema
from .baseline import CboBaselinePackage


SCENARIO_SCHEMA_RESOURCE = "schemas/cbo-scenario-v1.schema.json"


@dataclass(frozen=True)
class CboScenarioSpec:
    """Canonical scenario overlay specification."""

    path: Path | None
    data: Mapping[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CboScenarioSpec":
        spec_path = Path(path).expanduser().resolve()
        data = _read_scenario_file(spec_path)
        return cls.from_mapping(data, path=spec_path)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, path: str | Path | None = None) -> "CboScenarioSpec":
        if not isinstance(data, Mapping):
            raise ValueError("CBO scenario spec must be a mapping")
        schema = _scenario_schema()
        validate_schema(data, schema, label="scenario")
        _validate_no_compatible_baseline(data)
        _validate_file_references(data)
        _validate_mode_specific_overrides(data)
        return cls(path=Path(path).resolve() if path is not None else None, data=dict(data))

    @property
    def scenario_id(self) -> str:
        return str(self.data["scenario_id"])

    def canonical_json(self) -> str:
        return canonical_json_text(self.data)

    def canonical_sha256(self) -> str:
        return canonical_json_sha256(self.data)

    def assert_baseline_matches(self, baseline: CboBaselinePackage) -> None:
        declared = self.data.get("baseline", {})
        if not isinstance(declared, Mapping):
            raise ValueError("scenario baseline block must be a mapping")
        expected_package = str(declared.get("package_sha256") or "")
        expected_manifest = str(declared.get("manifest_sha256") or "")
        expected_attestation = str(declared.get("release_attestation_sha256") or "")
        if expected_package != baseline.package_sha256:
            raise ValueError("scenario baseline.package_sha256 does not match opened baseline")
        if expected_manifest != baseline.manifest_sha256:
            raise ValueError("scenario baseline.manifest_sha256 does not match opened baseline")
        if expected_attestation != baseline.attestation.sha256:
            raise ValueError("scenario baseline.release_attestation_sha256 does not match opened attestation")


def _read_scenario_file(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return read_json(path)


def _scenario_schema() -> dict[str, Any]:
    schema_path = files("tdcsim_cbo").joinpath(SCENARIO_SCHEMA_RESOURCE)
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise SchemaValidationError("scenario schema must be a JSON object")
    return schema


def _validate_no_compatible_baseline(data: Mapping[str, Any]) -> None:
    baseline = data.get("baseline", {})
    if isinstance(baseline, Mapping) and bool(baseline.get("allow_compatible_baseline", False)):
        raise ValueError("CBO scenario v1 requires exact baseline hashes; allow_compatible_baseline is forbidden")


def _validate_file_references(value: Any, *, path: str = "scenario") -> None:
    if isinstance(value, Mapping):
        if "relative_path" in value:
            rel = str(value["relative_path"])
            _validate_relative_path(rel, path=f"{path}.relative_path")
        for key, child in value.items():
            if key in {"source_role", "runtime_role", "claim_boundary"}:
                raise ValueError(f"{path}.{key}: scenario authors may not set source/runtime/claim labels")
            _validate_file_references(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_file_references(child, path=f"{path}[{index}]")


def _validate_relative_path(value: str, *, path: str) -> None:
    candidate = Path(value)
    if value.startswith("/") or candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(f"{path}: referenced files must be package-relative safe paths")
    if not value or value.endswith("/"):
        raise ValueError(f"{path}: referenced file path must name a file")


def _validate_mode_specific_overrides(data: Mapping[str, Any]) -> None:
    overrides = data.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("scenario.overrides must be a mapping")
    for name, override in overrides.items():
        if not isinstance(override, Mapping):
            raise ValueError(f"scenario.overrides.{name}: override must be a mapping")
        mode = str(override.get("mode") or "")
        if name == "nominal_yield_curve":
            _require_by_mode(name, override, mode, {"parallel_bp": ("shock_bp",), "key_rate_bp": ("shocks",), "full_surface_file": ("file",)})
        elif name == "frn_benchmark":
            _require_by_mode(name, override, mode, {"parallel_bp": ("shock_bp",), "absolute_path_file": ("file",), "linked_to_nominal_curve": ()})
        elif name == "inflation_cpi":
            _require_by_mode(name, override, mode, {"annualized_inflation_shift_bp": ("shock_bp",), "cpi_level_scale": ("scale",), "monthly_path_file": ("file",)})
        elif name == "tips_real_yield":
            _require_by_mode(name, override, mode, {"parallel_bp": ("shock_bp",), "key_rate_bp": ("shocks",), "absolute_path_file": ("file",), "linked_recompute": ()})
        elif name == "operating_cash":
            _require_by_mode(name, override, mode, {"constant_real": (), "constant_nominal": (), "scale_baseline": ("scale",), "aggregate_path_file": ("file",), "component_path_file": ("file",)})
        elif name == "cash_reconciliation":
            _require_by_mode(name, override, mode, {"zero": (), "track_operating_cash_target": (), "explicit_path_file": ("file",)})
        elif name == "fed_holdings":
            _require_by_mode(name, override, mode, _stock_path_requirements())
        elif name in {"primary_deficit", "debt_target"}:
            _require_by_mode(name, override, mode, _stock_path_requirements())
        elif name == "holder_preferences":
            _require_by_mode(name, override, mode, {"static_shares": ("rows",)})
        elif name == "net_interest_comparator":
            _require_by_mode(name, override, mode, {"official_cbo_baseline": ("role",)})


def _stock_path_requirements() -> dict[str, tuple[str, ...]]:
    return {
        "scale_path": ("scale",),
        "additive_bil": ("additive_bil",),
        "fy_endpoint_anchors": ("anchors",),
        "absolute_path_file": ("file",),
    }


def _require_by_mode(
    name: str,
    override: Mapping[str, Any],
    mode: str,
    requirements: Mapping[str, tuple[str, ...]],
) -> None:
    if mode not in requirements:
        return
    missing = [key for key in requirements[mode] if key not in override]
    if missing:
        raise ValueError(f"scenario.overrides.{name}.{mode}: missing required fields {missing}")


__all__ = [
    "CboScenarioSpec",
    "SchemaValidationError",
]
