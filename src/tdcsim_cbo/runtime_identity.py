"""Portable runtime identity helpers for release-bound CBO runs."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any


def distribution_identity(package_name: str = "tdcsim") -> dict[str, str]:
    """Return a canonical digest of installed package files, not installer metadata."""

    try:
        dist = metadata.distribution(package_name)
    except metadata.PackageNotFoundError:
        return {
            "name": package_name,
            "version": "source-tree",
            "file_digest": "0" * 64,
            "identity_source": "source_tree_no_installed_distribution",
        }
    records: list[dict[str, Any]] = []
    for item in sorted(dist.files or [], key=lambda value: str(value)):
        rel = str(item).replace("\\", "/")
        if not include_runtime_file(rel):
            continue
        path = Path(dist.locate_file(item))
        if not path.is_file():
            continue
        records.append({"path": rel, "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    return {
        "name": dist.metadata.get("Name", package_name),
        "version": dist.version,
        "file_digest": canonical_file_digest(records),
        "identity_source": "installed_distribution_files",
    }


def wheel_file_digest(path: str | Path) -> str:
    """Return the canonical digest for runtime files contained in a wheel."""

    wheel = Path(path)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(wheel) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            rel = info.filename.replace("\\", "/")
            if info.is_dir() or not include_runtime_file(rel):
                continue
            with archive.open(info) as handle:
                payload = handle.read()
            records.append({"path": rel, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
    return canonical_file_digest(records)


def distribution_file_path(
    relative_path: str,
    package_name: str = "tdcsim",
) -> Path:
    """Resolve one installed distribution file without accepting source shadowing."""

    rel = PurePosixPath(relative_path)
    if (
        rel.is_absolute()
        or not rel.parts
        or any(part in {"", ".", ".."} for part in rel.parts)
    ):
        raise ValueError("distribution relative_path is invalid")
    dist = metadata.distribution(package_name)
    matches = [
        member
        for member in (dist.files or [])
        if str(member).replace("\\", "/") == str(rel)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "installed distribution does not own exactly one file at "
            f"{relative_path}"
        )
    path = Path(dist.locate_file(matches[0])).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"installed distribution file is missing: {relative_path}"
        )
    return path


def installed_archive_sha256(package_name: str = "tdcsim") -> str:
    """Return pip's PEP 610 hash for the exact installed wheel archive."""

    dist = metadata.distribution(package_name)
    matches = [
        member
        for member in (dist.files or [])
        if str(member).replace("\\", "/").endswith(
            ".dist-info/direct_url.json"
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "installed distribution does not carry one direct_url.json"
        )
    payload = json.loads(
        Path(dist.locate_file(matches[0])).read_text(encoding="utf-8")
    )
    archive = payload.get("archive_info")
    if not isinstance(archive, dict):
        raise ValueError("installed distribution is not a wheel archive install")
    hashes = archive.get("hashes")
    sha = hashes.get("sha256") if isinstance(hashes, dict) else None
    if not isinstance(sha, str):
        legacy = archive.get("hash")
        if isinstance(legacy, str) and legacy.startswith("sha256="):
            sha = legacy.removeprefix("sha256=")
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(char not in "0123456789abcdef" for char in sha)
    ):
        raise ValueError(
            "installed distribution direct_url.json has no SHA-256 archive pin"
        )
    return sha


def assert_loaded_distribution_modules(
    required_modules: set[str],
    package_name: str = "tdcsim",
) -> list[dict[str, str]]:
    """Fail when any loaded TDCSim module is absent from or shadows the wheel."""

    dist = metadata.distribution(package_name)
    module_members: dict[str, Any] = {}
    for member in dist.files or []:
        rel = str(member).replace("\\", "/")
        if rel.startswith("../") or not rel.endswith(".py"):
            continue
        path = PurePosixPath(rel)
        if path.name == "__init__.py":
            module_name = ".".join(path.parts[:-1])
        else:
            module_name = ".".join((*path.parts[:-1], path.stem))
        if not module_name:
            continue
        if module_name in module_members:
            raise ValueError(
                f"installed distribution owns duplicate module {module_name}"
            )
        module_members[module_name] = member
    missing = sorted(required_modules - set(module_members))
    if missing:
        raise ValueError(
            f"installed distribution omits required modules: {missing}"
        )
    records: list[dict[str, str]] = []
    for module_name, member in sorted(module_members.items()):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        origin = getattr(module, "__file__", None)
        if not origin:
            raise ValueError(f"loaded module has no file origin: {module_name}")
        observed = Path(origin).resolve()
        expected = Path(dist.locate_file(member)).resolve()
        if observed != expected:
            raise ValueError(
                f"loaded module shadows installed distribution: {module_name}"
            )
        records.append(
            {
                "module": module_name,
                "relative_path": str(member).replace("\\", "/"),
            }
        )
    for module_name in required_modules:
        if module_name not in sys.modules:
            importlib.import_module(module_name)
            return assert_loaded_distribution_modules(
                required_modules,
                package_name=package_name,
            )
    return records


def verify_wheel_against_git_commit(
    wheel_path: str | Path,
    source_repository: str | Path,
    commit_sha: str,
) -> dict[str, Any]:
    """Bind every packaged runtime byte to the declared Git commit."""

    if (
        len(commit_sha) != 40
        or commit_sha == "0" * 40
        or any(char not in "0123456789abcdef" for char in commit_sha)
    ):
        raise ValueError("wheel source commit must be nonzero lowercase 40-hex")
    repository = Path(source_repository).expanduser().resolve()
    root = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode(
            "utf-8"
        ).strip()
    ).resolve()
    _git(root, "cat-file", "-e", f"{commit_sha}^{{commit}}")
    pyproject = tomllib.loads(
        _git(root, "show", f"{commit_sha}:pyproject.toml").decode("utf-8")
    )
    setuptools = pyproject.get("tool", {}).get("setuptools", {})
    modules = setuptools.get("py-modules", [])
    packages = setuptools.get("packages", [])
    package_data = setuptools.get("package-data", {})
    if (
        not isinstance(modules, list)
        or not isinstance(packages, list)
        or not isinstance(package_data, dict)
    ):
        raise ValueError("commit pyproject has an unsupported setuptools layout")

    expected: dict[str, str] = {}
    for module in modules:
        if not isinstance(module, str) or not module:
            raise ValueError("commit pyproject contains an invalid py-module")
        expected[f"{module}.py"] = f"src/{module}.py"
    tracked = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        commit_sha,
        "--",
        "src",
    ).decode("utf-8").splitlines()
    tracked_set = set(tracked)
    for package in packages:
        if not isinstance(package, str) or not package:
            raise ValueError("commit pyproject contains an invalid package")
        package_rel = package.replace(".", "/")
        prefix = f"src/{package_rel}/"
        for repo_path in tracked:
            if repo_path.startswith(prefix) and repo_path.endswith(".py"):
                expected[repo_path.removeprefix("src/")] = repo_path
    for package, patterns in package_data.items():
        if not isinstance(package, str) or not isinstance(patterns, list):
            raise ValueError("commit pyproject contains invalid package-data")
        package_rel = package.replace(".", "/")
        prefix = f"src/{package_rel}/"
        for pattern in patterns:
            if not isinstance(pattern, str):
                raise ValueError("commit pyproject package-data pattern is invalid")
            for repo_path in tracked:
                if not repo_path.startswith(prefix):
                    continue
                within = repo_path.removeprefix(prefix)
                if PurePosixPath(within).match(pattern):
                    expected[repo_path.removeprefix("src/")] = repo_path
    if not expected or any(path not in tracked_set for path in expected.values()):
        raise ValueError("commit packaging contract references missing source files")

    wheel = Path(wheel_path).expanduser().resolve()
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(wheel) as archive:
        runtime_members = {
            info.filename.replace("\\", "/"): info
            for info in archive.infolist()
            if not info.is_dir()
            and include_runtime_file(info.filename.replace("\\", "/"))
        }
        if set(runtime_members) != set(expected):
            raise ValueError(
                "wheel runtime members differ from the commit packaging contract: "
                f"missing={sorted(set(expected) - set(runtime_members))}, "
                f"extra={sorted(set(runtime_members) - set(expected))}"
            )
        for wheel_rel, repo_rel in sorted(expected.items()):
            wheel_bytes = archive.read(runtime_members[wheel_rel])
            source_bytes = _git(
                root,
                "show",
                f"{commit_sha}:{repo_rel}",
            )
            if wheel_bytes != source_bytes:
                raise ValueError(
                    f"wheel runtime bytes differ from commit: {wheel_rel}"
                )
            records.append(
                {
                    "path": wheel_rel,
                    "sha256": hashlib.sha256(wheel_bytes).hexdigest(),
                    "bytes": len(wheel_bytes),
                }
            )
    return {
        "schema_version": "tdcsim_wheel_git_binding_v1",
        "release_commit_sha": commit_sha,
        "runtime_file_count": len(records),
        "runtime_file_set_sha256": canonical_file_digest(records),
    }


def git_commit_source_identity(
    source_repository: str | Path,
    commit_sha: str,
) -> dict[str, Any]:
    """Recompute the tracked-tree and lock identity directly from Git objects."""

    repository = Path(source_repository).expanduser().resolve()
    root = Path(
        _git(repository, "rev-parse", "--show-toplevel").decode(
            "utf-8"
        ).strip()
    ).resolve()
    tracked = sorted(
        path
        for path in _git(
            root,
            "ls-tree",
            "-r",
            "--name-only",
            commit_sha,
        )
        .decode("utf-8")
        .splitlines()
        if path
    )
    if not tracked:
        raise ValueError("Git commit has no tracked source files")
    records: list[dict[str, Any]] = []
    for relative_path in tracked:
        payload = _git(root, "show", f"{commit_sha}:{relative_path}")
        record = {
            "path": relative_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        records.append(record)
    by_path = {record["path"]: record for record in records}
    lock_files = [
        {
            "relative_path": name,
            "sha256": by_path[name]["sha256"],
            "bytes": by_path[name]["bytes"],
        }
        for name in ("uv.lock", "requirements.lock.txt")
        if name in by_path
    ]
    if not lock_files:
        raise ValueError("Git commit has no dependency lock file")
    lock_payload = json.dumps(
        lock_files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "release_commit_sha": commit_sha,
        "source_tree_sha256": canonical_file_digest(records),
        "source_tree_file_count": len(records),
        "dependency_lock_files": lock_files,
        "dependency_lock_set_sha256": hashlib.sha256(lock_payload).hexdigest(),
    }


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout


def locked_environment_mismatches(
    lock_payload: bytes | str,
) -> dict[str, dict[str, str]]:
    """Compare every exact requirement pin with the installed environment."""

    text = (
        lock_payload.decode("utf-8")
        if isinstance(lock_payload, bytes)
        else lock_payload
    )
    pins: dict[str, str] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)(?:\s*;.*)?$"
    )
    for raw_line in text.splitlines():
        match = pattern.fullmatch(raw_line.strip())
        if match is None:
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        version = match.group(2)
        prior = pins.get(name)
        if prior is not None and prior != version:
            raise ValueError(f"requirements lock has conflicting pins for {name}")
        pins[name] = version
    if not pins:
        raise ValueError("requirements lock has no exact distribution pins")
    mismatches: dict[str, dict[str, str]] = {}
    for name, required in sorted(pins.items()):
        try:
            observed = metadata.version(name)
        except metadata.PackageNotFoundError:
            observed = "<missing>"
        if observed != required:
            mismatches[name] = {
                "required": required,
                "observed": observed,
            }
    return mismatches


def canonical_file_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def include_runtime_file(rel: str) -> bool:
    path = PurePosixPath(rel)
    parts = path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return False
    if any(part == "__pycache__" for part in parts):
        return False
    if rel.endswith((".pyc", ".pyo")):
        return False
    if any(part.endswith(".dist-info") or part.endswith(".egg-info") for part in parts):
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "distribution_file_path",
    "distribution_identity",
    "git_commit_source_identity",
    "include_runtime_file",
    "installed_archive_sha256",
    "locked_environment_mismatches",
    "assert_loaded_distribution_modules",
    "verify_wheel_against_git_commit",
    "wheel_file_digest",
]
