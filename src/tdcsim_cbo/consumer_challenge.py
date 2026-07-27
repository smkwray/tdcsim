"""Producer-side ingest challenge for the RateWall export boundary.

The full-file SHA-256 values are the decisive byte-identity evidence. Semantic digests are
additional receipts over the original CSV strings, never parsed floats, so last-decimal drift
cannot hide behind binary floating-point collisions.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._json import canonical_json_sha256, read_json, sha256_file, write_json
from .runtime_identity import canonical_file_digest

SELECTED_FIELD = "delta_tdc_ex_overlap_non_interest_admissible_bil"
SUPPORTING_FIELDS = (
    "delta_tdc_ex_overlap_interest_driven_excluded_bil",
    "tdc_materialized_deposit_stock_admissible_bil",
    "tdc_income_addendum_full_level_rate",
)
CUMULATIVE_GATE_FIELDS = (
    "delta_tdc_ex_overlap_split_remainder_bil",
    "aggregation_reconciliation_status",
    "demand_conversion_case",
)
CURRENT_SUMMARY_GATE_FIELDS = (
    "tdc_deposit_creation_split_schema_version",
    "tdc_income_addendum_admission_status",
    "tdc_income_addendum_collision_status",
    "delta_tdc_ex_overlap_split_remainder_bil",
)
EXPECTED_ABSENT_FIELDS = (
    "legacy_chi_support_diagnostic_bil",
    "marginal_tdc_support_bil",
)
AUDIT_ONLY_NON_SELECTED_FIELDS = ("delta_tdc_ex_overlap_bil",)
# Backward-compatible name for code reading v1 declarations. V2 separates the two semantics.
EXCLUDED_FIELDS = EXPECTED_ABSENT_FIELDS + AUDIT_ONLY_NON_SELECTED_FIELDS
CANONICAL_ROW_KEY = "period"
CUMULATIVE_INPUT_ROLE = "cumulative_schedule"
CURRENT_SUMMARY_INPUT_ROLE = "current_summary_fallback"
CHALLENGE_SCHEMA_VERSION_V1 = "tdcsim_ratewall_ingest_challenge_v1"
CHALLENGE_SCHEMA_VERSION = "tdcsim_ratewall_ingest_challenge_v2"
PACKAGE_MANIFEST_SCHEMA_VERSION = "tdcsim_ratewall_handoff_package_manifest_v1"
TRANSFER_MANIFEST_SCHEMA_VERSION = "tdcsim_ratewall_transfer_manifest_v1"
EXPECTED_CUMULATIVE_KEY_COUNT = 11
EXPECTED_CURRENT_SUMMARY_KEY_COUNT = 1

CUMULATIVE_PROJECTION_FIELDS = (
    CANONICAL_ROW_KEY,
    "demand_conversion_case",
    SELECTED_FIELD,
    *SUPPORTING_FIELDS,
    "delta_tdc_ex_overlap_split_remainder_bil",
    "aggregation_reconciliation_status",
)
CURRENT_SUMMARY_PROJECTION_FIELDS = (
    CANONICAL_ROW_KEY,
    "demand_conversion_case",
    SELECTED_FIELD,
    *SUPPORTING_FIELDS,
    "delta_tdc_ex_overlap_split_remainder_bil",
    "tdc_deposit_creation_split_schema_version",
    "tdc_income_addendum_admission_status",
    "tdc_income_addendum_collision_status",
)


def collect_release_identity(project_root: str | Path) -> dict[str, Any]:
    """Return a clean, fail-closed identity for the source tree executing the producer."""

    root = Path(project_root).resolve()
    release_sha = _git(root, "rev-parse", "HEAD")
    if not _is_hex_digest(release_sha, 40):
        raise ValueError("producer release identity requires a valid lowercase 40-hex commit")
    dirty_output = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty_output:
        raise ValueError("producer release identity requires dirty_state == false")

    tracked_raw = _git_bytes(root, "ls-files", "-z")
    tracked_paths = [item.decode("utf-8") for item in tracked_raw.split(b"\0") if item]
    if not tracked_paths:
        raise ValueError("producer source tree has no tracked files")
    records: list[dict[str, Any]] = []
    for relative_path in sorted(tracked_paths):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"tracked producer source file is missing: {relative_path}")
        records.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    lock_files = []
    for name in ("uv.lock", "requirements.lock.txt"):
        path = root / name
        if path.is_file():
            lock_files.append(_artifact(path, root))
    if not lock_files:
        raise ValueError("producer release identity requires a dependency lock file")

    return {
        "release_commit_sha": release_sha,
        "dirty_state": False,
        "runtime_identity_source": "clean_tracked_source_tree_files",
        "source_tree_sha256": canonical_file_digest(records),
        "source_tree_file_count": len(records),
        "dependency_lock_files": lock_files,
        "dependency_lock_set_sha256": canonical_json_sha256(lock_files),
        "python_version": platform.python_version(),
    }


def build_handoff_package_manifest(
    package_root: str | Path,
    *,
    campaign_id: str,
    pair_index_path: str | Path,
    pair_manifest_paths: Sequence[str | Path],
    cumulative_csv: str | Path,
    current_summary_csv: str | Path,
    producer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the complete consumed input set and its transitive pair/run lineage."""

    root = Path(package_root).resolve()
    if not campaign_id.strip():
        raise ValueError("handoff package requires a nonblank campaign_id")
    _validate_producer_identity(producer_identity)

    pair_index = Path(pair_index_path).resolve()
    cumulative = Path(cumulative_csv).resolve()
    current = Path(current_summary_csv).resolve()
    for path in (pair_index, cumulative, current):
        _require_inside(root, path)
        if not path.is_file():
            raise FileNotFoundError(f"handoff package input is missing: {path}")

    pair_records: list[dict[str, Any]] = []
    source_run_records: list[dict[str, str]] = []
    for raw_path in pair_manifest_paths:
        path = Path(raw_path).resolve()
        _require_inside(root, path)
        manifest = read_json(path)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"pair manifest must be an object: {path}")
        pair_id = str(manifest.get("pair_id") or "")
        if not pair_id:
            raise ValueError(f"pair manifest has no pair_id: {path}")
        sources = []
        for role in ("baseline_run", "shock_run"):
            block = manifest.get(role)
            if not isinstance(block, Mapping):
                raise ValueError(f"pair manifest {pair_id} is missing {role}")
            source = {
                "pair_id": pair_id,
                "role": role,
                "run_id": str(block.get("run_id") or ""),
                "manifest_sha256": str(block.get("manifest_sha256") or ""),
            }
            if not source["run_id"] or not _is_hex_digest(source["manifest_sha256"], 64):
                raise ValueError(f"pair manifest {pair_id} has invalid {role} lineage")
            sources.append(source)
            source_run_records.append(source)
        pair_records.append(
            {
                "pair_id": pair_id,
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "source_runs": sources,
            }
        )
    pair_records.sort(key=lambda item: item["pair_id"])
    source_run_records.sort(key=lambda item: (item["pair_id"], item["role"], item["run_id"]))
    pair_ids = [item["pair_id"] for item in pair_records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("handoff package has duplicate pair manifest identities")
    _validate_pair_index(pair_index, pair_ids)

    return {
        "schema_version": PACKAGE_MANIFEST_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "producer_identity": dict(producer_identity),
        "pair_index": _artifact(pair_index, root),
        "pair_manifests": pair_records,
        "pair_manifest_identities_sha256": canonical_json_sha256(pair_records),
        "source_run_identities": source_run_records,
        "source_run_identities_sha256": canonical_json_sha256(source_run_records),
        "consumed_inputs": [
            {"role": CUMULATIVE_INPUT_ROLE, **_artifact(cumulative, root)},
            {"role": CURRENT_SUMMARY_INPUT_ROLE, **_artifact(current, root)},
        ],
    }


def build_ingest_challenge(
    cumulative_csv: str | Path,
    current_summary_csv: str | Path,
    *,
    package_manifest_path: str | Path,
    validate_package_records: bool = True,
) -> dict[str, Any]:
    """Describe the exact bytes and projections used to construct one selected schedule."""

    package_path = Path(package_manifest_path).resolve()
    package = read_json(package_path)
    if not isinstance(package, Mapping) or package.get("schema_version") != PACKAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported or malformed handoff package manifest")
    _validate_producer_identity(package.get("producer_identity"))

    cumulative = Path(cumulative_csv).resolve()
    current = Path(current_summary_csv).resolve()
    projections = [
        _cumulative_projection(cumulative),
        _current_summary_projection(current),
    ]
    for projection, path in zip(projections, (cumulative, current), strict=True):
        _require_inside(package_path.parent, path)
        projection["source_relative_path"] = path.relative_to(package_path.parent).as_posix()
    if validate_package_records:
        _validate_consumed_input_records(package, projections)

    cumulative_keys = projections[0]["canonical_keys"]
    if "2026" not in cumulative_keys:
        raise ValueError(
            "cumulative input must include 2026: the consumer fallback cannot contribute to a passing 11-key schedule"
        )
    lineage = {
        "campaign_id": str(package.get("campaign_id") or ""),
        "package_manifest_sha256": sha256_file(package_path),
        "pair_index_sha256": str(package.get("pair_index", {}).get("sha256") or ""),
        "pair_manifest_identities": package.get("pair_manifests", []),
        "pair_manifest_identities_sha256": str(package.get("pair_manifest_identities_sha256") or ""),
        "source_run_identities": package.get("source_run_identities", []),
        "source_run_identities_sha256": str(package.get("source_run_identities_sha256") or ""),
    }
    if not lineage["campaign_id"]:
        raise ValueError("handoff package manifest has no campaign_id")

    return {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "selected_object": "ratewall_split_tdc_schedule",
        "producer_identity": dict(package["producer_identity"]),
        "lineage": lineage,
        "canonical_row_key": CANONICAL_ROW_KEY,
        "expected_selected_key_count": EXPECTED_CUMULATIVE_KEY_COUNT,
        "selected_key_set_digest_sha256": projections[0]["key_set_digest_sha256"],
        "selected_projection_digest_sha256": projections[0]["projection_digest_sha256"],
        "selected_projection_digest_basis": (
            "sha256_over_field_names_and_original_csv_strings_not_parsed_floats"
        ),
        "input_files": projections,
        "claim": (
            "producer published both hash-pinned input files and expects the consumer to construct "
            "one 11-key schedule from the cumulative central rows; the current-summary file is a "
            "required hash-pinned conditional fallback, but cannot contribute to a passing schedule "
            "under the captured consumer count logic"
        ),
    }


def read_ingest_challenge(path: str | Path) -> dict[str, Any]:
    """Read either the retained v1 challenge or the current v2 challenge."""

    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("ingest challenge must be a JSON object")
    if data.get("schema_version") not in {CHALLENGE_SCHEMA_VERSION_V1, CHALLENGE_SCHEMA_VERSION}:
        raise ValueError("unsupported ingest challenge schema_version")
    return data


def verify_ingest_handoff(
    package_root: str | Path,
    *,
    challenge_name: str = "tdcsim_ratewall_ingest_challenge.json",
    package_manifest_name: str = "tdcsim_ratewall_handoff_package_manifest.json",
    transfer_manifest_name: str = "tdcsim_ratewall_transfer_manifest.json",
) -> dict[str, Any]:
    """Recompute the retained challenge immediately before handoff and emit one transfer manifest."""

    root = Path(package_root).resolve()
    package_path = root / package_manifest_name
    package = read_json(package_path)
    if not isinstance(package, Mapping):
        raise ValueError("handoff package manifest must be an object")
    inputs = _consumed_input_map(package)
    cumulative = root / inputs[CUMULATIVE_INPUT_ROLE]["relative_path"]
    current = root / inputs[CURRENT_SUMMARY_INPUT_ROLE]["relative_path"]
    recomputed = build_ingest_challenge(
        cumulative,
        current,
        package_manifest_path=package_path,
        validate_package_records=False,
    )
    challenge_path = root / challenge_name
    expected_bytes = _json_file_bytes(recomputed)
    if not challenge_path.is_file() or challenge_path.read_bytes() != expected_bytes:
        raise ValueError("retained ingest challenge differs from recomputed packaged inputs")

    transfer = {
        "schema_version": TRANSFER_MANIFEST_SCHEMA_VERSION,
        "campaign_id": str(package.get("campaign_id") or ""),
        "package_manifest": _artifact(package_path, root),
        "consumed_inputs": [
            {"role": CUMULATIVE_INPUT_ROLE, **_artifact(cumulative, root)},
            {"role": CURRENT_SUMMARY_INPUT_ROLE, **_artifact(current, root)},
        ],
        "ingest_challenge": _artifact(challenge_path, root),
        "verification_status": "pass_recomputed_challenge_byte_identity",
    }
    transfer_path = root / transfer_manifest_name
    write_json(transfer_path, transfer)
    return {
        "status": "pass",
        "transfer_manifest_path": str(transfer_path),
        "transfer_manifest_sha256": sha256_file(transfer_path),
        "manifest": transfer,
    }


def _cumulative_projection(path: Path) -> dict[str, Any]:
    header, rows = _read_csv(path)
    required = (*CUMULATIVE_PROJECTION_FIELDS, *AUDIT_ONLY_NON_SELECTED_FIELDS)
    _validate_header(path, header, required)
    admitted = []
    for row_number, row in enumerate(rows, start=2):
        _validate_row_values(path, row_number, row, required)
        if row["aggregation_reconciliation_status"] != "pass":
            raise ValueError(f"cumulative reconciliation gate failed at CSV row {row_number}")
        if _decimal(row["delta_tdc_ex_overlap_split_remainder_bil"], path, row_number) != 0:
            raise ValueError(f"cumulative split remainder gate failed at CSV row {row_number}")
        if row["demand_conversion_case"] == "central":
            admitted.append(row)
    keys = _validate_keys(admitted, expected_count=EXPECTED_CUMULATIVE_KEY_COUNT, label="cumulative")
    return _projection_record(
        path,
        role=CUMULATIVE_INPUT_ROLE,
        header=header,
        pre_filter_count=len(rows),
        admitted=admitted,
        keys=keys,
        fields=CUMULATIVE_PROJECTION_FIELDS,
        contribution="primary_11_key_schedule",
    )


def _current_summary_projection(path: Path) -> dict[str, Any]:
    header, rows = _read_csv(path)
    required = (*CURRENT_SUMMARY_PROJECTION_FIELDS, *AUDIT_ONLY_NON_SELECTED_FIELDS)
    _validate_header(path, header, required)
    admitted = []
    for row_number, row in enumerate(rows, start=2):
        _validate_row_values(path, row_number, row, required)
        if row["tdc_deposit_creation_split_schema_version"] != "tdc_deposit_creation_split_v1":
            continue
        if row["tdc_income_addendum_admission_status"] != "admitted_split_non_interest_bucket":
            continue
        if row["tdc_income_addendum_collision_status"] != "pass_split_collision_excluded":
            continue
        if _decimal(row["delta_tdc_ex_overlap_split_remainder_bil"], path, row_number) != 0:
            continue
        admitted.append(row)
    keys = _validate_keys(admitted, expected_count=EXPECTED_CURRENT_SUMMARY_KEY_COUNT, label="current summary")
    if keys != ["2026"]:
        raise ValueError("current summary fallback candidate must carry period 2026")
    return _projection_record(
        path,
        role=CURRENT_SUMMARY_INPUT_ROLE,
        header=header,
        pre_filter_count=len(rows),
        admitted=admitted,
        keys=keys,
        fields=CURRENT_SUMMARY_PROJECTION_FIELDS,
        contribution="required_hash_pinned_fallback_candidate_not_selected_when_cumulative_contains_2026",
    )


def _projection_record(
    path: Path,
    *,
    role: str,
    header: list[str],
    pre_filter_count: int,
    admitted: list[dict[str, str]],
    keys: list[str],
    fields: Sequence[str],
    contribution: str,
) -> dict[str, Any]:
    ordered = sorted(admitted, key=lambda row: str(row[CANONICAL_ROW_KEY]))
    return {
        "role": role,
        "source_file_name": path.name,
        "source_file_sha256": sha256_file(path),
        "source_file_bytes": path.stat().st_size,
        "actual_header": header,
        "header_digest_sha256": canonical_json_sha256(header),
        "required_projection_fields": list(fields),
        "expected_absent_fields": list(EXPECTED_ABSENT_FIELDS),
        "audit_only_fields_prohibited_from_selection": list(AUDIT_ONLY_NON_SELECTED_FIELDS),
        "pre_filter_row_count": pre_filter_count,
        "post_filter_row_count": len(admitted),
        "expected_key_count": len(keys),
        "canonical_keys": keys,
        "key_set_digest_sha256": canonical_json_sha256(sorted(keys)),
        "projection_digest_sha256": _projection_digest(ordered, fields),
        "projection_digest_basis": "original_csv_strings_not_parsed_floats",
        "schedule_contribution": contribution,
    }


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"consumer CSV is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    if not header:
        raise ValueError(f"consumer CSV has no header: {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"consumer CSV has duplicate header fields: {path}")
    if not rows:
        raise ValueError(f"consumer CSV has no rows: {path}")
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"consumer CSV has extra cells at row {row_number}: {path}")
    return header, rows


def _validate_header(path: Path, header: Sequence[str], required: Sequence[str]) -> None:
    missing = [field for field in required if field not in header]
    if missing:
        raise ValueError(f"consumer CSV is missing required fields {missing}: {path}")
    present_forbidden = [field for field in EXPECTED_ABSENT_FIELDS if field in header]
    if present_forbidden:
        raise ValueError(f"consumer CSV contains fields required to be absent {present_forbidden}: {path}")


def _validate_row_values(path: Path, row_number: int, row: Mapping[str, Any], required: Sequence[str]) -> None:
    missing = [field for field in required if row.get(field) is None]
    if missing:
        raise ValueError(f"consumer CSV row {row_number} has missing values for {missing}: {path}")


def _validate_keys(rows: Sequence[Mapping[str, str]], *, expected_count: int, label: str) -> list[str]:
    keys = [str(row[CANONICAL_ROW_KEY]) for row in rows]
    if any(not key.strip() for key in keys):
        raise ValueError(f"{label} canonical period keys must be nonblank")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} canonical period keys must be unique after consumer string normalization")
    if len(keys) != expected_count:
        raise ValueError(f"{label} expected {expected_count} admitted keys, found {len(keys)}")
    return sorted(keys)


def _projection_digest(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        for field in fields:
            digest.update(field.encode("utf-8"))
            digest.update(b"\x1f")
            digest.update(str(row[field]).encode("utf-8"))
            digest.update(b"\x1d")
        digest.update(b"\x1e")
    return digest.hexdigest()


def _decimal(value: str, path: Path, row_number: int) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"consumer CSV has malformed decimal at row {row_number}: {path}") from exc


def _validate_pair_index(path: Path, pair_ids: Sequence[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "pair_id" not in (reader.fieldnames or []):
            raise ValueError("pair-index manifest is missing pair_id")
        indexed = [str(row.get("pair_id") or "") for row in reader]
    if any(not item for item in indexed) or len(indexed) != len(set(indexed)):
        raise ValueError("pair-index manifest has blank or duplicate pair_id values")
    if set(indexed) != set(pair_ids):
        raise ValueError("pair-index manifest identities do not match pair manifests")


def _validate_consumed_input_records(package: Mapping[str, Any], projections: Sequence[Mapping[str, Any]]) -> None:
    records = _consumed_input_map(package)
    for projection in projections:
        record = records.get(str(projection["role"]))
        if not isinstance(record, Mapping):
            raise ValueError(f"handoff package is missing consumed input role {projection['role']}")
        if record.get("relative_path") != projection.get("source_relative_path"):
            raise ValueError(f"handoff package consumed input path mismatch for {projection['role']}")
        if record.get("sha256") != projection.get("source_file_sha256"):
            raise ValueError(f"handoff package consumed input SHA mismatch for {projection['role']}")
        if int(record.get("bytes", -1)) != int(projection.get("source_file_bytes", -2)):
            raise ValueError(f"handoff package consumed input byte-count mismatch for {projection['role']}")


def _consumed_input_map(package: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = package.get("consumed_inputs")
    if not isinstance(raw, list):
        raise ValueError("handoff package consumed_inputs must be a list")
    records = {str(item.get("role") or ""): item for item in raw if isinstance(item, Mapping)}
    required = {CUMULATIVE_INPUT_ROLE, CURRENT_SUMMARY_INPUT_ROLE}
    if set(records) != required:
        raise ValueError("handoff package must bind exactly both consumed input roles")
    return records


def _validate_producer_identity(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("producer_identity must be an object")
    if not _is_hex_digest(str(value.get("release_commit_sha") or ""), 40):
        raise ValueError("producer_identity release_commit_sha must be lowercase 40-hex")
    if value.get("dirty_state") is not False:
        raise ValueError("producer_identity dirty_state must be false")
    if not _is_hex_digest(str(value.get("source_tree_sha256") or ""), 64):
        raise ValueError("producer_identity source_tree_sha256 must be 64-hex")
    if not _is_hex_digest(str(value.get("dependency_lock_set_sha256") or ""), 64):
        raise ValueError("producer_identity dependency_lock_set_sha256 must be 64-hex")
    locks = value.get("dependency_lock_files")
    if not isinstance(locks, list) or not locks:
        raise ValueError("producer_identity requires dependency_lock_files")
    if not str(value.get("runtime_identity_source") or "") or not str(value.get("python_version") or ""):
        raise ValueError("producer_identity requires runtime and Python identity")


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    _require_inside(root.resolve(), path)
    return {
        "relative_path": path.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _require_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"handoff path is outside package root: {path}") from exc


def _is_hex_digest(value: str, length: int) -> bool:
    return len(value) == length and value == value.lower() and all(char in "0123456789abcdef" for char in value)


def _git(root: Path, *args: str) -> str:
    return _git_bytes(root, *args).decode("utf-8").strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"producer release identity could not run git {' '.join(args)}") from exc


def _json_file_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


__all__ = [
    "AUDIT_ONLY_NON_SELECTED_FIELDS",
    "CANONICAL_ROW_KEY",
    "CHALLENGE_SCHEMA_VERSION",
    "CHALLENGE_SCHEMA_VERSION_V1",
    "CUMULATIVE_GATE_FIELDS",
    "CURRENT_SUMMARY_GATE_FIELDS",
    "EXCLUDED_FIELDS",
    "EXPECTED_ABSENT_FIELDS",
    "PACKAGE_MANIFEST_SCHEMA_VERSION",
    "SELECTED_FIELD",
    "SUPPORTING_FIELDS",
    "TRANSFER_MANIFEST_SCHEMA_VERSION",
    "build_handoff_package_manifest",
    "build_ingest_challenge",
    "collect_release_identity",
    "read_ingest_challenge",
    "verify_ingest_handoff",
]
