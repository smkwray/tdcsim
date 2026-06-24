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


__all__ = [
    "CboScenarioSpec",
    "SchemaValidationError",
]
