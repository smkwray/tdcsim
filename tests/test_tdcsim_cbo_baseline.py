import json
import zipfile
from pathlib import Path

import pytest

from tdcsim_cbo import CboBaselinePackage
from tdcsim_cbo._json import sha256_file, write_json


RELEASE_SHA = "639c033a25b540f7c0b747ff05644e523e82abad"
VERIFIER_SHA = "4f49871864261be66382db58a879c0058118171760beec2b1aca1c3bf2639a0a"


def test_baseline_open_requires_valid_attestation_and_exact_hashes(tmp_path: Path) -> None:
    package, attestation = _write_release_package(tmp_path)

    baseline = CboBaselinePackage.open(package, attestation_path=attestation)

    assert baseline.package_sha256 == sha256_file(package)
    assert baseline.attestation.release_commit_sha == RELEASE_SHA
    assert baseline.manifest["trust_anchor"]["dirty_state"] is False
    assert baseline.package_id == "cbo_full_horizon_local_smoke_3m"


def test_baseline_open_rejects_missing_attestation_when_verified(tmp_path: Path) -> None:
    package, _ = _write_release_package(tmp_path)

    with pytest.raises(ValueError, match="requires an external release attestation"):
        CboBaselinePackage.open(package)


def test_baseline_open_rejects_wrong_package_hash(tmp_path: Path) -> None:
    package, attestation = _write_release_package(tmp_path)
    data = json.loads(attestation.read_text(encoding="utf-8"))
    data["baseline_package_zip_sha256"] = "0" * 64
    write_json(attestation, data)

    with pytest.raises(ValueError, match="package SHA-256"):
        CboBaselinePackage.open(package, attestation_path=attestation)


def test_baseline_open_rejects_dirty_attestation(tmp_path: Path) -> None:
    package, attestation = _write_release_package(tmp_path)
    data = json.loads(attestation.read_text(encoding="utf-8"))
    data["dirty_state"] = True
    write_json(attestation, data)

    with pytest.raises(ValueError, match="dirty_state"):
        CboBaselinePackage.open(package, attestation_path=attestation)


def test_baseline_open_rejects_unsafe_zip_member(tmp_path: Path) -> None:
    unsafe_zip = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe_zip, "w") as archive:
        archive.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe zip member"):
        CboBaselinePackage.open(unsafe_zip, verify=False)


def test_baseline_materialize_extracts_verified_zip(tmp_path: Path) -> None:
    package, attestation = _write_release_package(tmp_path)
    baseline = CboBaselinePackage.open(package, attestation_path=attestation)

    materialized = baseline.materialize(tmp_path / "materialized")

    assert (materialized / "manifest.json").exists()
    assert (materialized / "forecast_inputs" / "source_contract_smoke.json").exists()


def test_real_release_bound_package_opens_when_local_artifact_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    package = root / "output" / "cbo_forecast_release_bound_package.zip"
    attestation = root / "output" / "cbo_forecast_release_bound_attestation.json"
    if not package.exists() or not attestation.exists():
        pytest.skip("local release-bound CBO artifacts are ignored and not present")

    baseline = CboBaselinePackage.open(package, attestation_path=attestation)

    assert baseline.package_sha256 == "be49f5a5d256863649ccf1b139d259679c9a3ce669642751c22e8a7dad9a2d2c"
    assert baseline.attestation.sha256 == "d30a6f89263cdb80f8f9d81131dc0004b7b9751289e0594ee5005e115641124a"
    assert baseline.manifest_sha256 == "4cf0d3571abcd0d80c91a28e23f9c4e00ce6075564f5354e3b12e5364727e16c"


def _write_release_package(tmp_path: Path) -> tuple[Path, Path]:
    package_dir = tmp_path / "pkg"
    (package_dir / "forecast_inputs").mkdir(parents=True)
    lock_text = "pandas==0\n"
    (package_dir / "requirements.lock.txt").write_text(lock_text, encoding="utf-8")
    lock_sha = sha256_file(package_dir / "requirements.lock.txt")
    manifest = {
        "schema_version": "tdcsim_cbo_forecast_smoke_manifest_v1",
        "scenario_id": "cbo_full_horizon_local_smoke_3m",
        "trust_anchor": {
            "schema_version": "tdcsim_cbo_package_trust_anchor_v1",
            "code_revision": RELEASE_SHA,
            "dirty_state": False,
            "requirements_lock_sha256": lock_sha,
            "verifier_sha256": VERIFIER_SHA,
        },
    }
    write_json(package_dir / "manifest.json", manifest)
    write_json(package_dir / "forecast_inputs" / "source_contract_smoke.json", manifest)
    package = tmp_path / "pkg.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())
    attestation = tmp_path / "attestation.json"
    write_json(
        attestation,
        {
            "schema_version": "tdcsim_cbo_external_release_attestation_v1",
            "attestation_created_at_utc": "2026-06-24T00:00:00+00:00",
            "release_commit_sha": RELEASE_SHA,
            "release_commit_short": "639c033",
            "branch_state": "detached_release_validation_worktree",
            "worktree_path": "/tmp/tdcsim-cbo-release-639c033",
            "git_status_short": "",
            "dirty_state": False,
            "baseline_package_dir": "output/cbo_forecast_release_bound",
            "baseline_package_zip": "output/cbo_forecast_release_bound_package.zip",
            "baseline_package_zip_sha256": sha256_file(package),
            "baseline_manifest_sha256": sha256_file(package_dir / "manifest.json"),
            "source_contract_sha256": sha256_file(package_dir / "forecast_inputs" / "source_contract_smoke.json"),
            "verifier_sha256": VERIFIER_SHA,
            "requirements_lock_sha256": sha256_file(package_dir / "requirements.lock.txt"),
            "manifest_trust_anchor": manifest["trust_anchor"],
            "artifact_hashes_count": 0,
            "required_files_count": 3,
            "scenario_id": "cbo_full_horizon_local_smoke_3m",
            "run_summaries": [],
            "source_package_used": "test-fixture",
            "validation_grade": "release_reproducible_local_clean_commit",
            "claim_boundary": {"package_role": "test"},
            "unsupported_components": {},
            "commands": [
                {
                    "name": "unit_fixture",
                    "command": "test",
                    "cwd": str(tmp_path),
                    "exit_code": 0,
                    "stdout_log": "fixture.txt",
                    "stdout_sha256": "0" * 64,
                    "result": "pass",
                }
            ],
            "attestation_log_hashes": {},
        },
    )
    return package, attestation
