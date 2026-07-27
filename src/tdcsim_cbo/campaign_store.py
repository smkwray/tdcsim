"""Package-relative source-run storage for retained marginal-pair campaigns."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

from ._json import read_json, sha256_file, write_json
from ._schema import validate_schema


SOURCE_RUN_CATALOG_FILE = "tdcsim_cbo_source_run_catalog.json"
SOURCE_RUN_CATALOG_SCHEMA_VERSION = "tdcsim_cbo_source_run_catalog_v1"


class CampaignStoreError(ValueError):
    """Raised when a campaign source-run catalog is incomplete or inconsistent."""


@dataclass(frozen=True)
class CatalogSourceRun:
    run_id: str
    root: Path
    manifest_sha256: str
    baseline_package: Path | None
    attestation: Path | None


def register_source_run(
    campaign_root: str | Path,
    run_dir: str | Path,
    *,
    baseline_package: str | Path,
    attestation: str | Path,
) -> str:
    """Store one source run and its replay inputs once in a movable campaign."""

    campaign = Path(campaign_root).expanduser().resolve()
    source = Path(run_dir).expanduser().resolve()
    package = Path(baseline_package).expanduser().resolve()
    attestation_path = Path(attestation).expanduser().resolve()
    manifest_path = source / "tdcsim_cbo_run_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise CampaignStoreError("source run manifest must be an object")
    run_id = str(manifest.get("run_id") or "")
    _validate_run_id(run_id)
    baseline = manifest.get("baseline")
    if not isinstance(baseline, Mapping):
        raise CampaignStoreError("source run manifest has no baseline identity")
    package_sha = sha256_file(package)
    attestation_sha = sha256_file(attestation_path)
    if package_sha != baseline.get("package_sha256"):
        raise CampaignStoreError("baseline package hash does not match source run manifest")
    if attestation_sha != baseline.get("release_attestation_sha256"):
        raise CampaignStoreError("attestation hash does not match source run manifest")

    run_relative = Path("source_runs") / run_id
    stored_run = campaign / run_relative
    manifest_sha = sha256_file(manifest_path)
    if stored_run.exists():
        stored_manifest = stored_run / manifest_path.name
        if not stored_manifest.is_file() or sha256_file(stored_manifest) != manifest_sha:
            raise CampaignStoreError(f"campaign source run conflicts with run_id: {run_id}")
    else:
        stored_run.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, stored_run)

    baseline_root = Path("baseline_packages") / package_sha
    package_relative = baseline_root / "baseline_package.zip"
    attestation_relative = baseline_root / "attestation.json"
    _copy_exact(package, campaign / package_relative, expected_sha=package_sha)
    _copy_exact(attestation_path, campaign / attestation_relative, expected_sha=attestation_sha)

    entry = {
        "run_id": run_id,
        "run_relative_path": run_relative.as_posix(),
        "run_manifest_sha256": manifest_sha,
        "baseline_package_relative_path": package_relative.as_posix(),
        "baseline_package_sha256": package_sha,
        "attestation_relative_path": attestation_relative.as_posix(),
        "attestation_sha256": attestation_sha,
    }
    catalog_path = campaign / SOURCE_RUN_CATALOG_FILE
    catalog = _read_or_create_catalog(catalog_path)
    entries = {
        str(item["run_id"]): dict(item)
        for item in catalog.get("source_runs", [])
        if isinstance(item, Mapping) and item.get("run_id")
    }
    existing = entries.get(run_id)
    if existing is not None and existing != entry:
        raise CampaignStoreError(f"source-run catalog conflicts with run_id: {run_id}")
    entries[run_id] = entry
    catalog["source_runs"] = [entries[key] for key in sorted(entries)]
    _validate_catalog(catalog)
    write_json(catalog_path, catalog)
    return run_id


def write_fixture_source_run_catalog(
    campaign_root: str | Path,
    run_dirs: list[str | Path],
    *,
    copy_runs: bool = False,
) -> Path:
    """Write a package-relative catalog for synthetic fixture runs without replay inputs."""

    campaign = Path(campaign_root).expanduser().resolve()
    entries = []
    for run_dir in run_dirs:
        root = Path(run_dir).expanduser().resolve()
        manifest_path = root / "tdcsim_cbo_run_manifest.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, Mapping) or not manifest.get("run_id"):
            raise CampaignStoreError("fixture source run manifest has no run_id")
        run_id = str(manifest["run_id"])
        _validate_run_id(run_id)
        if copy_runs:
            relative = Path("source_runs") / run_id
            stored = campaign / relative
            if not stored.exists():
                stored.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(root, stored)
            root = stored
            manifest_path = root / manifest_path.name
        else:
            try:
                relative = root.relative_to(campaign)
            except ValueError as exc:
                raise CampaignStoreError("fixture source run must be inside its campaign") from exc
        entries.append(
            {
                "run_id": run_id,
                "run_relative_path": relative.as_posix(),
                "run_manifest_sha256": sha256_file(manifest_path),
            }
        )
    catalog = {
        "schema_version": SOURCE_RUN_CATALOG_SCHEMA_VERSION,
        "source_runs": sorted(entries, key=lambda item: item["run_id"]),
    }
    _validate_catalog(catalog)
    path = campaign / SOURCE_RUN_CATALOG_FILE
    write_json(path, catalog)
    return path


def locate_source_run_catalog(start: str | Path) -> Path:
    """Find the nearest enclosing campaign source-run catalog."""

    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate_root in (current, *current.parents):
        candidate = candidate_root / SOURCE_RUN_CATALOG_FILE
        if candidate.is_file():
            return candidate
    raise CampaignStoreError(f"no {SOURCE_RUN_CATALOG_FILE} found above {current}")


def resolve_source_run(catalog_path: str | Path, run_id: str) -> CatalogSourceRun:
    """Resolve a stable run ID to package-relative campaign artifacts."""

    _validate_run_id(run_id)
    path = Path(catalog_path).expanduser().resolve()
    catalog = _read_catalog(path)
    matches = [item for item in catalog["source_runs"] if item.get("run_id") == run_id]
    if len(matches) != 1:
        raise CampaignStoreError(f"source-run catalog must contain exactly one entry for {run_id}")
    entry = matches[0]
    root = path.parent
    run_root = _resolve_relative(root, entry["run_relative_path"], label="source run")
    manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != entry["run_manifest_sha256"]:
        raise CampaignStoreError(f"source run manifest hash mismatch for {run_id}")

    package = _optional_artifact(root, entry, "baseline_package_relative_path", "baseline_package_sha256")
    attestation = _optional_artifact(root, entry, "attestation_relative_path", "attestation_sha256")
    if (package is None) != (attestation is None):
        raise CampaignStoreError(f"source run replay inputs are incomplete for {run_id}")
    return CatalogSourceRun(
        run_id=run_id,
        root=run_root,
        manifest_sha256=str(entry["run_manifest_sha256"]),
        baseline_package=package,
        attestation=attestation,
    )


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id or Path(run_id).name != run_id:
        raise CampaignStoreError("source run_id must be a path-safe stable identifier")


def _copy_exact(source: Path, destination: Path, *, expected_sha: str) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha:
            raise CampaignStoreError(f"campaign artifact conflicts with expected hash: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected_sha:
        raise CampaignStoreError(f"copied campaign artifact hash mismatch: {destination.name}")


def _read_or_create_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": SOURCE_RUN_CATALOG_SCHEMA_VERSION, "source_runs": []}
    return _read_catalog(path)


def _read_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignStoreError(f"source-run catalog is missing: {path.name}")
    try:
        data = read_json(path)
    except Exception as exc:
        raise CampaignStoreError(f"source-run catalog cannot be read: {path.name}") from exc
    if not isinstance(data, dict):
        raise CampaignStoreError("source-run catalog must be an object")
    _validate_catalog(data)
    run_ids = [str(item.get("run_id") or "") for item in data.get("source_runs", [])]
    if len(run_ids) != len(set(run_ids)):
        raise CampaignStoreError("source-run catalog has duplicate run IDs")
    return data


def _validate_catalog(catalog: Mapping[str, Any]) -> None:
    with files("tdcsim_cbo").joinpath("schemas/cbo-source-run-catalog-v1.schema.json").open(
        "r", encoding="utf-8"
    ) as handle:
        schema = json.load(handle)
    try:
        validate_schema(dict(catalog), schema, label="source_run_catalog")
    except Exception as exc:
        raise CampaignStoreError(f"source-run catalog schema validation failed: {exc}") from exc


def _resolve_relative(root: Path, value: Any, *, label: str) -> Path:
    relative = Path(str(value or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignStoreError(f"{label} path must be campaign-relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CampaignStoreError(f"{label} path escapes campaign root") from exc
    return resolved


def _optional_artifact(root: Path, entry: Mapping[str, Any], path_key: str, sha_key: str) -> Path | None:
    if path_key not in entry and sha_key not in entry:
        return None
    if path_key not in entry or sha_key not in entry:
        raise CampaignStoreError(f"catalog artifact requires both {path_key} and {sha_key}")
    path = _resolve_relative(root, entry[path_key], label=path_key)
    if not path.is_file() or sha256_file(path) != entry[sha_key]:
        raise CampaignStoreError(f"catalog artifact hash mismatch: {path_key}")
    return path
