from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest


_PACKAGE_SHA256 = (
    "be49f5a5d256863649ccf1b139d259679c9a3ce669642751c22e8a7dad9a2d2c"
)
_ATTESTATION_SHA256 = (
    "d30a6f89263cdb80f8f9d81131dc0004b7b9751289e0594ee5005e115641124a"
)
_LOCK_SHA256 = (
    "16e3fc32257a01e1fd2e5a53867cc9e73496b9d35c6a5d4db19c671617325a4e"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.integration
def test_open04_installed_wheel_is_bound_to_clean_commit_and_not_shadowed(
    tmp_path: Path,
    record_property,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    package = project_root / "output" / "cbo_forecast_release_bound_package.zip"
    attestation = (
        project_root / "output" / "cbo_forecast_release_bound_attestation.json"
    )
    lock = project_root / "requirements.lock.txt"
    assert _sha256(package) == _PACKAGE_SHA256
    assert _sha256(attestation) == _ATTESTATION_SHA256
    assert _sha256(lock) == _LOCK_SHA256

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert len(commit) == 40 and commit != "0" * 40
    worktrees = (project_root / ".worktrees").resolve()
    worktrees.mkdir(exist_ok=True)
    source_repository = (
        worktrees / f"open04-wheel-identity-{uuid4().hex[:12]}"
    ).resolve()
    assert source_repository.parent == worktrees
    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(source_repository),
            commit,
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        assert not subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=source_repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        build_source = tmp_path / "build-source"
        build_source.mkdir()
        for name in ("pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(source_repository / name, build_source / name)
        shutil.copytree(source_repository / "src", build_source / "src")
        wheelhouse = tmp_path / "wheelhouse"
        wheelhouse.mkdir()
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--wheel-dir",
                str(wheelhouse),
                str(build_source),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        wheels = list(wheelhouse.glob("tdcsim-*.whl"))
        assert len(wheels) == 1
        wheel = wheels[0]
        wheel_sha = _sha256(wheel)

        installed = tmp_path / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-compile",
                "--target",
                str(installed),
                str(wheel),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        outside = tmp_path / "outside"
        outside.mkdir()
        probe = (
            "import json,platform,sys;"
            "from pathlib import Path;"
            "from tdcsim_cbo import CboBaselinePackage;"
            "from tdcsim_cbo.runner import _assert_open04_release_identity;"
            "from tdcsim_cbo.verifier import _verify_producer_source_identity;"
            "baseline=CboBaselinePackage.open("
            "sys.argv[1],attestation_path=sys.argv[2]);"
            "identity=_assert_open04_release_identity(baseline);"
            "_verify_producer_source_identity({"
            "'code_commit_sha':sys.argv[4],"
            "'wheel_sha256':sys.argv[5],"
            "'python_version':platform.python_version(),"
            "'producer_source_identity':identity},Path(sys.argv[3]));"
            "print(json.dumps(identity,"
            "sort_keys=True))"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(installed),
                "TDCSIM_CBO_CODE_COMMIT_SHA": commit,
                "TDCSIM_CBO_DIRTY_STATE": "false",
                "TDCSIM_CBO_REQUIREMENTS_LOCK_SHA256": _LOCK_SHA256,
                "TDCSIM_CBO_SOURCE_REPOSITORY": str(source_repository),
                "TDCSIM_CBO_WHEEL_PATH": str(wheel),
                "TDCSIM_CBO_WHEEL_SHA256": wheel_sha,
            }
        )
        positive = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(package),
                str(attestation),
                str(wheel),
                commit,
                wheel_sha,
            ],
            cwd=outside,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        identity = json.loads(positive.stdout)
        assert identity["source_tree"]["release_commit_sha"] == commit
        assert identity["source_tree"]["dirty_state"] is False
        assert identity["wheel_git_binding"]["release_commit_sha"] == commit
        assert identity["installed_archive_sha256"] == wheel_sha

        shadowed_environment = dict(environment)
        shadowed_environment["PYTHONPATH"] = os.pathsep.join(
            [str(source_repository / "src"), str(installed)]
        )
        negative = subprocess.run(
            [
                sys.executable,
                "-c",
                probe,
                str(package),
                str(attestation),
                str(wheel),
                commit,
                wheel_sha,
            ],
            cwd=outside,
            env=shadowed_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert negative.returncode != 0
        assert "release wheel/source qualification failed" in negative.stderr
        assert "shadows installed distribution" in negative.stderr

        record_property("open04_release_commit_sha", commit)
        record_property("open04_release_wheel_sha256", wheel_sha)
        record_property(
            "open04_wheel_runtime_source_sha256",
            identity["wheel_git_binding"]["runtime_file_set_sha256"],
        )
        record_property(
            "open04_loaded_distribution_module_count",
            identity["loaded_distribution_module_count"],
        )
        record_property("open04_source_shadow_negative_gate", "pass")
        record_property("open04_requirements_lock_sha256", _LOCK_SHA256)
        record_property("open04_baseline_package_sha256", _PACKAGE_SHA256)
        record_property(
            "open04_baseline_attestation_sha256",
            _ATTESTATION_SHA256,
        )
    finally:
        if source_repository.exists():
            assert source_repository.parent == worktrees
            subprocess.run(
                [
                    "git",
                    "worktree",
                    "remove",
                    "--force",
                    str(source_repository),
                ],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
