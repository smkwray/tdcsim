"""Release-bound CBO baseline package reader."""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ._json import read_json, sha256_file


EXPECTED_ATTESTATION_SCHEMAS = {
    "tdcsim_cbo_external_release_attestation_v1",
    "tdcsim_cbo_release_attestation_v1",
}

APPROVED_EXTERNAL_ATTESTATION_KEYS = {
    "artifact_hashes_count",
    "attestation_created_at_utc",
    "attestation_log_hashes",
    "baseline_manifest_sha256",
    "baseline_package_dir",
    "baseline_package_zip",
    "baseline_package_zip_sha256",
    "branch_state",
    "claim_boundary",
    "commands",
    "dirty_state",
    "git_status_short",
    "manifest_trust_anchor",
    "release_commit_sha",
    "release_commit_short",
    "required_files_count",
    "requirements_lock_sha256",
    "run_summaries",
    "scenario_id",
    "schema_version",
    "source_contract_sha256",
    "source_package_used",
    "unsupported_components",
    "validation_grade",
    "verifier_sha256",
    "worktree_path",
}

APPROVED_SCAFFOLD_ATTESTATION_KEYS = {
    "baseline_manifest_sha256",
    "commands",
    "created_at_utc",
    "dirty_state",
    "package_id",
    "package_zip_sha256",
    "python_version",
    "release_commit_sha",
    "requirements_lock_sha256",
    "schema_version",
    "signature",
    "status",
    "verifier_sha256",
}


@dataclass(frozen=True)
class ReleaseAttestation:
    """Validated external release attestation."""

    path: Path
    sha256: str
    data: Mapping[str, Any]

    @property
    def package_sha256(self) -> str:
        return str(
            self.data.get("baseline_package_zip_sha256")
            or self.data.get("package_zip_sha256")
            or ""
        )

    @property
    def manifest_sha256(self) -> str:
        return str(self.data.get("baseline_manifest_sha256") or "")

    @property
    def release_commit_sha(self) -> str:
        return str(self.data.get("release_commit_sha") or "")


@dataclass(frozen=True)
class CboBaselinePackage:
    """Verified CBO baseline package plus its release attestation."""

    package_path: Path
    package_sha256: str
    manifest_sha256: str
    source_contract_sha256: str
    manifest: Mapping[str, Any]
    attestation: ReleaseAttestation
    is_zip: bool

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        attestation_path: str | Path | None = None,
        verify: bool = True,
    ) -> "CboBaselinePackage":
        package_path = Path(path).expanduser().resolve()
        if not package_path.exists():
            raise FileNotFoundError(f"CBO baseline package is missing: {package_path}")
        is_zip = package_path.suffix.lower() == ".zip"
        if is_zip:
            _validate_zip_members(package_path)
            package_sha256 = sha256_file(package_path)
            manifest_bytes = _read_zip_bytes(package_path, "manifest.json")
            source_contract_bytes = _read_zip_bytes(package_path, "forecast_inputs/source_contract_smoke.json")
            manifest = _read_json_payload(manifest_bytes)
            source_contract = _read_json_payload(source_contract_bytes)
            lock_bytes = _read_zip_bytes(package_path, "requirements.lock.txt")
        else:
            package_sha256 = ""
            manifest_path = package_path / "manifest.json"
            source_contract_path = package_path / "forecast_inputs" / "source_contract_smoke.json"
            manifest_bytes = manifest_path.read_bytes()
            source_contract_bytes = source_contract_path.read_bytes()
            manifest = read_json(manifest_path)
            source_contract = read_json(source_contract_path)
            lock_bytes = (package_path / "requirements.lock.txt").read_bytes()
        if manifest != source_contract:
            raise ValueError("manifest.json and forecast_inputs/source_contract_smoke.json differ")

        manifest_sha256 = _payload_sha256(manifest_bytes)
        source_contract_sha256 = _payload_sha256(source_contract_bytes)
        if verify and attestation_path is None:
            raise ValueError("verified CBO baseline open requires an external release attestation")
        attestation = _load_attestation(attestation_path) if attestation_path is not None else _dummy_attestation()
        if verify:
            _verify_attestation(
                package_path=package_path,
                package_sha256=package_sha256,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                lock_bytes=lock_bytes,
                attestation=attestation,
            )
        return cls(
            package_path=package_path,
            package_sha256=package_sha256,
            manifest_sha256=manifest_sha256,
            source_contract_sha256=source_contract_sha256,
            manifest=manifest,
            attestation=attestation,
            is_zip=is_zip,
        )

    @property
    def package_id(self) -> str:
        return str(
            self.attestation.data.get("package_id")
            or self.attestation.data.get("scenario_id")
            or self.package_path.stem
        )

    def materialize(self, work_dir: str | Path) -> Path:
        target = Path(work_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"materialize target already exists: {target}")
        target.mkdir(parents=True)
        if self.is_zip:
            _extract_zip_safely(self.package_path, target)
        else:
            shutil.copytree(self.package_path, target, dirs_exist_ok=True)
        return target


def _load_attestation(path: str | Path) -> ReleaseAttestation:
    attestation_path = Path(path).expanduser().resolve()
    data = read_json(attestation_path)
    if not isinstance(data, Mapping):
        raise ValueError("release attestation must be a JSON object")
    _verify_attestation_schema(data)
    return ReleaseAttestation(path=attestation_path, sha256=sha256_file(attestation_path), data=data)


def _dummy_attestation() -> ReleaseAttestation:
    return ReleaseAttestation(path=Path(), sha256="", data={})


def _verify_attestation_schema(data: Mapping[str, Any]) -> None:
    schema = data.get("schema_version")
    if schema not in EXPECTED_ATTESTATION_SCHEMAS:
        raise ValueError(f"unsupported CBO release attestation schema: {schema!r}")
    approved = (
        APPROVED_EXTERNAL_ATTESTATION_KEYS
        if schema == "tdcsim_cbo_external_release_attestation_v1"
        else APPROVED_SCAFFOLD_ATTESTATION_KEYS
    )
    unknown = sorted(set(str(key) for key in data) - approved)
    if unknown:
        raise ValueError(f"release attestation contains unknown fields: {unknown}")
    if data.get("dirty_state") is not False:
        raise ValueError("release attestation dirty_state must be false")
    release_sha = str(data.get("release_commit_sha") or "")
    if len(release_sha) != 40 or release_sha.lower() != release_sha or any(c not in "0123456789abcdef" for c in release_sha):
        raise ValueError("release attestation release_commit_sha must be a 40-character lowercase git SHA")
    commands = data.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("release attestation commands must be a nonempty list")
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            raise ValueError(f"release attestation command {index} must be an object")
        if command.get("exit_code") != 0:
            raise ValueError(f"release attestation command {index} did not exit 0")


def _verify_attestation(
    *,
    package_path: Path,
    package_sha256: str,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    lock_bytes: bytes,
    attestation: ReleaseAttestation,
) -> None:
    if package_path.suffix.lower() == ".zip" and attestation.package_sha256 != package_sha256:
        raise ValueError("release attestation package SHA-256 does not match baseline zip")
    if attestation.manifest_sha256 != manifest_sha256:
        raise ValueError("release attestation manifest SHA-256 does not match package manifest")
    lock_sha = _payload_sha256(lock_bytes)
    if str(attestation.data.get("requirements_lock_sha256") or "") != lock_sha:
        raise ValueError("release attestation requirements lock SHA-256 does not match package")
    trust = manifest.get("trust_anchor", {})
    if not isinstance(trust, Mapping):
        raise ValueError("package manifest trust_anchor must be an object")
    if trust.get("dirty_state") is not False:
        raise ValueError("package manifest trust_anchor dirty_state must be false for release use")
    if str(trust.get("code_revision") or "") != attestation.release_commit_sha:
        raise ValueError("package manifest code_revision does not match release attestation")
    if str(trust.get("requirements_lock_sha256") or "") != lock_sha:
        raise ValueError("package manifest requirements lock SHA-256 does not match package")
    verifier_sha = str(attestation.data.get("verifier_sha256") or "")
    if verifier_sha and str(trust.get("verifier_sha256") or "") != verifier_sha:
        raise ValueError("package manifest verifier SHA-256 does not match release attestation")


def _validate_zip_members(path: Path) -> None:
    seen: set[str] = set()
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            name = info.filename
            if name in seen:
                raise ValueError(f"duplicate zip member: {name}")
            seen.add(name)
            _validate_relative_archive_name(name)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"zip member is a symlink: {name}")


def _validate_relative_archive_name(name: str) -> None:
    pure = Path(name)
    if name.startswith("/") or pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"unsafe zip member path: {name}")


def _extract_zip_safely(path: Path, target: Path) -> None:
    _validate_zip_members(path)
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            dest = (target / info.filename).resolve()
            if target not in dest.parents and dest != target:
                raise ValueError(f"zip member escapes target directory: {info.filename}")
            archive.extract(info, target)


def _read_zip_json(path: Path, member: str) -> Any:
    return _read_json_payload(_read_zip_bytes(path, member))


def _read_json_payload(payload: bytes) -> Any:
    import json

    return json.loads(payload.decode("utf-8"))


def _read_zip_bytes(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        try:
            return archive.read(member)
        except KeyError as exc:
            raise FileNotFoundError(f"CBO baseline package is missing {member}") from exc


def _payload_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
