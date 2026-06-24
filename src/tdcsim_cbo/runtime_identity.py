"""Portable runtime identity helpers for release-bound CBO runs."""

from __future__ import annotations

import hashlib
import json
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


__all__ = ["distribution_identity", "include_runtime_file", "wheel_file_digest"]
