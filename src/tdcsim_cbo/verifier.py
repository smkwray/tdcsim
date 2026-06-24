"""Independent verifier for compiled and run CBO scenario packages."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import subprocess
from importlib import metadata
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from sim_engine import run_simulation

from ._json import read_json, sha256_file
from .baseline import CboBaselinePackage
from .compiler import CboScenarioCompiler, digest_input_tree
from .contract import CboScenarioSpec
from .output import hash_output_tree
from . import runner as runner_module
from ._schema import validate_schema


class VerificationError(ValueError):
    """Raised when a compiled or run package fails verification."""


REQUIRED_RESULT_COLUMNS = (
    "CBOControlledDebtTargetError",
    "CBOFedAuctionShare",
    "CBOFedAuctionRolloverAddons",
    "CBORemittanceCashEffect",
    "NetInterestDiagnosticStatus",
)
REQUIRED_SUMMARY_KEYS = (
    "CBOControlledDebtTargetError_max_abs",
    "CBOFedAuctionShare_max_abs",
    "CBOFedAuctionRolloverAddons_max_abs",
)


def verify_compiled_scenario(compiled_dir: str | Path) -> dict[str, Any]:
    root = Path(compiled_dir).expanduser().resolve()
    manifest_path = root / "tdcsim_cbo_compiled_manifest.json"
    manifest = _manifest(manifest_path)
    inputs = root / "forecast_inputs"
    if not inputs.exists():
        raise VerificationError("compiled forecast_inputs directory is missing")
    actual_digest = digest_input_tree(inputs)
    expected_digest = str(manifest.get("compiled_inputs_digest") or "")
    if actual_digest != expected_digest:
        raise VerificationError("compiled input digest mismatch")
    expected_hashes = manifest.get("input_hashes")
    if expected_hashes != _strip_absent_order(hash_output_tree(inputs)):
        raise VerificationError("compiled input hash list mismatch")
    _verify_claim_boundary(manifest)
    return {"status": "pass", "compiled_inputs_digest": actual_digest, "input_count": len(expected_hashes or [])}


def verify_scenario_run(
    run_dir: str | Path,
    *,
    baseline_package: str | Path | None = None,
    attestation: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest = _manifest(root / "tdcsim_cbo_run_manifest.json")
    _validate_run_manifest_schema(manifest)
    _verify_release_claims(manifest)
    _verify_code_environment(manifest)
    _verify_validation_block(manifest)
    compiled_rel = Path(str(manifest.get("compiled_manifest") or ""))
    if compiled_rel.is_absolute() or ".." in compiled_rel.parts:
        raise VerificationError("run manifest compiled_manifest must be package-relative")
    compiled_manifest_path = root / compiled_rel
    compiled = verify_compiled_scenario(compiled_manifest_path.parent)
    if compiled["compiled_inputs_digest"] != manifest.get("compiled_inputs_digest"):
        raise VerificationError("run manifest compiled input digest does not match actual compiled tree")
    outputs = root / "outputs"
    if not outputs.exists():
        raise VerificationError("run outputs directory is missing")
    if manifest.get("output_hashes") != hash_output_tree(outputs):
        raise VerificationError("run output hash list mismatch")
    _verify_manifest_artifacts(root, manifest.get("outputs"), base=root)
    _verify_manifest_artifacts(root, manifest.get("compiled_inputs"), base=root)
    _verify_scenario_copy(root, manifest)
    if baseline_package is not None or attestation is not None:
        if baseline_package is None or attestation is None:
            raise VerificationError("baseline_package and attestation must be supplied together")
        _verify_recompile(root, manifest, baseline_package, attestation)
    boundaries = manifest.get("boundary_checks")
    if not isinstance(boundaries, dict):
        raise VerificationError("run manifest boundary_checks must be an object")
    if boundaries.get("net_interest_role") != "diagnostic_nonbinding":
        raise VerificationError("net interest role must remain diagnostic_nonbinding")
    if boundaries.get("remittance_deferred_asset_status") != "unsupported_in_cbo_scenario_lane":
        raise VerificationError("remittance/deferred-asset status must remain unsupported")
    if boundaries.get("fed_target_holder_allocation_only") is not True:
        raise VerificationError("Fed target holder-allocation boundary failed")
    recomputed = _verify_output_invariants(root, manifest)
    return {
        "status": "pass",
        "compiled": compiled,
        "output_count": len(manifest.get("output_hashes") or []),
        "recomputed": recomputed,
    }


def _manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise VerificationError(f"manifest is missing: {path.name}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise VerificationError("manifest must be a JSON object")
    _reject_absolute_paths(data)
    return data


def _verify_claim_boundary(manifest: dict[str, Any]) -> None:
    claim = manifest.get("claim_boundary")
    if not isinstance(claim, dict):
        raise VerificationError("compiled manifest claim_boundary must be an object")
    if claim.get("does_not_run_engine") is not True:
        raise VerificationError("compiled manifest must declare does_not_run_engine")
    if claim.get("net_interest_role") != "diagnostic_nonbinding":
        raise VerificationError("compiled manifest net interest role must be diagnostic_nonbinding")
    if claim.get("fed_holdings_role") != "holder_allocation_target_not_total_issuance":
        raise VerificationError("compiled manifest Fed holdings role is invalid")
    if claim.get("operating_cash_role") != "cash_path_not_issuance_plug":
        raise VerificationError("compiled manifest operating cash role is invalid")


def _validate_run_manifest_schema(manifest: dict[str, Any]) -> None:
    with files("tdcsim_cbo").joinpath("schemas/cbo-run-manifest-v1.schema.json").open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    try:
        validate_schema(manifest, schema, label="run_manifest")
    except Exception as exc:
        raise VerificationError(f"run manifest schema validation failed: {exc}") from exc


def _verify_release_claims(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise VerificationError("run manifest status must be complete")
    grade = manifest.get("verification_grade")
    if grade == "release_verified":
        raise VerificationError("release_verified grade requires external package identity evidence")
    if grade not in {"local", "release_reproducible"}:
        raise VerificationError(f"unsupported verification_grade: {grade!r}")


def _verify_code_environment(manifest: dict[str, Any]) -> None:
    env = manifest.get("code_environment")
    if not isinstance(env, dict):
        raise VerificationError("run manifest code_environment must be an object")
    expected = _actual_code_environment()
    for key, expected_value in expected.items():
        if env.get(key) != expected_value:
            raise VerificationError(f"run manifest code_environment {key} does not match verifier runtime")
    lock_sha = str(env.get("requirements_lock_sha256") or "")
    if not lock_sha or lock_sha == "0" * 64:
        raise VerificationError("run manifest code environment requirements lock hash is not release-bound")
    wheel_sha = str(env.get("wheel_sha256") or "")
    expected_wheel_sha = os.environ.get("TDCSIM_CBO_WHEEL_SHA256", "")
    if wheel_sha and not expected_wheel_sha:
        raise VerificationError("run manifest code environment wheel_sha256 cannot be verified")
    if wheel_sha and wheel_sha != expected_wheel_sha:
        raise VerificationError("run manifest code environment wheel_sha256 does not match verifier runtime")


def _actual_code_environment() -> dict[str, Any]:
    dist = _distribution_identity()
    return {
        "code_commit_sha": _module_git_value(["rev-parse", "HEAD"], default="0" * 40),
        "dirty_state": bool(_module_git_value(["status", "--short"], default="")),
        "runner_version": "tdcsim_cbo_runner_v1",
        "verifier_version": "tdcsim_cbo_verifier_v1",
        "runner_source_sha256": sha256_file(Path(runner_module.__file__)),
        "sim_engine_source_sha256": sha256_file(Path(run_simulation.__code__.co_filename)),
        "package_name": dist["name"],
        "package_version": dist["version"],
        "distribution_record_digest": dist["record_digest"],
        "runtime_identity_source": dist["identity_source"],
    }


def _distribution_identity() -> dict[str, str]:
    try:
        dist = metadata.distribution("tdcsim")
        files_list = sorted(str(item) for item in (dist.files or []))
        record_path = _distribution_record_path(dist)
        return {
            "name": dist.metadata.get("Name", "tdcsim"),
            "version": dist.version,
            "record_digest": sha256_file(record_path) if record_path is not None else _record_digest(files_list),
            "identity_source": "installed_distribution_record" if record_path is not None else "installed_distribution_metadata",
        }
    except metadata.PackageNotFoundError:
        return {
            "name": "tdcsim",
            "version": "source-tree",
            "record_digest": "0" * 64,
            "identity_source": "source_tree_no_installed_distribution",
        }


def _distribution_record_path(dist: metadata.Distribution) -> Path | None:
    for item in dist.files or []:
        if str(item).endswith(".dist-info/RECORD"):
            path = Path(dist.locate_file(item))
            if path.exists():
                return path
    return None


def _record_digest(records: list[str]) -> str:
    import hashlib

    payload = "\n".join(records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _module_git_value(args: list[str], *, default: str) -> str:
    git_root = _module_git_root()
    if git_root is None:
        return default
    try:
        return subprocess.run(
            ["git", *args],
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return default


def _module_git_root() -> Path | None:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None
    return Path(root) if root else None


def _verify_validation_block(manifest: dict[str, Any]) -> None:
    validation = manifest.get("validation")
    if not isinstance(validation, dict):
        raise VerificationError("run manifest validation must be an object")
    if validation.get("status") != "pass":
        raise VerificationError("run manifest validation.status must be pass")
    for section in ("gates", "invariants"):
        items = validation.get(section)
        if not isinstance(items, list) or not items:
            raise VerificationError(f"run manifest validation.{section} must be a nonempty array")
        for item in items:
            if not isinstance(item, dict):
                raise VerificationError(f"run manifest validation.{section} item must be an object")
            if item.get("status") != "pass":
                raise VerificationError(f"run manifest validation item failed: {item.get('id')}")
    boundaries = manifest.get("boundary_checks")
    if not isinstance(boundaries, dict):
        raise VerificationError("run manifest boundary_checks must be an object")
    cash_flags = _normalized_bool_set(boundaries.get("cash_residual_affects_issuance_size"))
    nested_flags = boundaries.get("cash_residual_nonfunding_flags")
    if isinstance(nested_flags, dict) and "affects_issuance_size" in nested_flags:
        nested_cash_flags = _normalized_bool_set(nested_flags.get("affects_issuance_size"))
        if nested_cash_flags != cash_flags:
            raise VerificationError("cash_residual_affects_issuance_size disagrees with nonfunding flags")
    if cash_flags != {False}:
        raise VerificationError("cash_residual_affects_issuance_size must be exactly false")
    invariant_ids = {str(item.get("id")) for item in validation.get("invariants", []) if isinstance(item, dict)}
    if "cash_residual_not_issuance_sizing" not in invariant_ids:
        raise VerificationError("cash_residual_not_issuance_sizing validation invariant is required")


def _normalized_bool_set(values: Any) -> set[bool]:
    if isinstance(values, (str, bool)) or values is None:
        iterable = [values]
    elif isinstance(values, list):
        iterable = values
    else:
        return set()
    out: set[bool] = set()
    for value in iterable:
        if isinstance(value, bool):
            out.add(value)
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "false":
                out.add(False)
            elif lowered == "true":
                out.add(True)
            else:
                return set()
        else:
            return set()
    return out


def _verify_manifest_artifacts(root: Path, artifacts: Any, *, base: Path) -> None:
    if not isinstance(artifacts, list):
        raise VerificationError("manifest artifact list must be an array")
    for item in artifacts:
        if not isinstance(item, dict):
            raise VerificationError("manifest artifact item must be an object")
        rel = Path(str(item.get("relative_path") or ""))
        if rel.is_absolute() or ".." in rel.parts:
            raise VerificationError("manifest artifact path must be package-relative")
        path = base / rel
        if not path.exists():
            raise VerificationError(f"manifest artifact is missing: {rel}")
        if sha256_file(path) != item.get("sha256"):
            raise VerificationError(f"manifest artifact SHA mismatch: {rel}")
        if path.stat().st_size != int(item.get("bytes", -1)):
            raise VerificationError(f"manifest artifact byte count mismatch: {rel}")


def _verify_scenario_copy(root: Path, manifest: dict[str, Any]) -> None:
    scenario = manifest.get("scenario")
    if not isinstance(scenario, dict):
        raise VerificationError("run manifest scenario must be an object")
    rel = Path(str(scenario.get("relative_path") or ""))
    if rel.is_absolute() or ".." in rel.parts:
        raise VerificationError("scenario relative_path must be package-relative")
    path = root / rel
    if not path.exists():
        raise VerificationError("run scenario copy is missing")
    if sha256_file(path) != scenario.get("source_file_sha256"):
        raise VerificationError("run scenario copy hash mismatch")
    spec = CboScenarioSpec.from_file(path)
    if spec.canonical_sha256() != scenario.get("canonical_sha256"):
        raise VerificationError("run scenario canonical hash mismatch")
    _verify_manifest_artifacts(root, scenario.get("referenced_files", []), base=root)
    declared = _scenario_file_refs(spec.data)
    artifacts = {
        str(item.get("relative_path")): str(item.get("sha256"))
        for item in scenario.get("referenced_files", [])
        if isinstance(item, dict)
    }
    if declared != artifacts:
        raise VerificationError("run scenario referenced-file artifacts do not match scenario declarations")


def _scenario_file_refs(value: Any) -> dict[str, str]:
    refs: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "relative_path" in node and "sha256" in node:
                rel = str(node["relative_path"])
                sha = str(node["sha256"])
                if rel in refs and refs[rel] != sha:
                    raise VerificationError(f"scenario referenced file has conflicting hashes: {rel}")
                refs[rel] = sha
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return refs


def _verify_recompile(root: Path, manifest: dict[str, Any], baseline_package: str | Path, attestation: str | Path) -> None:
    scenario_rel = Path(str(manifest["scenario"]["relative_path"]))
    baseline = CboBaselinePackage.open(baseline_package, attestation_path=attestation)
    baseline_manifest = manifest.get("baseline")
    if not isinstance(baseline_manifest, dict):
        raise VerificationError("run manifest baseline must be an object")
    if baseline.package_sha256 != baseline_manifest.get("package_sha256"):
        raise VerificationError("verification baseline package hash does not match run manifest")
    if baseline.manifest_sha256 != baseline_manifest.get("manifest_sha256"):
        raise VerificationError("verification baseline manifest hash does not match run manifest")
    if baseline.attestation.sha256 != baseline_manifest.get("release_attestation_sha256"):
        raise VerificationError("verification attestation hash does not match run manifest")
    code_env = manifest.get("code_environment")
    if not isinstance(code_env, dict):
        raise VerificationError("run manifest code_environment must be an object")
    if code_env.get("requirements_lock_sha256") != baseline.attestation.data.get("requirements_lock_sha256"):
        raise VerificationError("run manifest requirements lock hash does not match baseline attestation")
    spec = CboScenarioSpec.from_file(root / scenario_rel)
    with tempfile.TemporaryDirectory(prefix="tdcsim-cbo-verify-recompile-") as tmp:
        compiled = CboScenarioCompiler().compile(baseline, spec, Path(tmp) / "work")
    if compiled.compiled_inputs_digest != manifest.get("compiled_inputs_digest"):
        raise VerificationError("recompiled input digest mismatch")


def _verify_output_invariants(root: Path, manifest: dict[str, Any]) -> dict[str, float]:
    results_path = _result_artifact_path(root, manifest)
    results = _read_results(results_path)
    _require_columns(results, REQUIRED_RESULT_COLUMNS, label="results")
    target_error = _max_abs(results, "CBOControlledDebtTargetError")
    fed_share = _max_abs(results, "CBOFedAuctionShare")
    fed_face = _max_abs(results, "CBOFedAuctionRolloverAddons")
    remittance_cash = _max_abs(results, "CBORemittanceCashEffect")
    if target_error > 1e-5:
        raise VerificationError(f"CBO target error exceeds tolerance: {target_error}")
    if fed_share > 1e-12 or fed_face > 1e-12:
        raise VerificationError(f"Fed auction boundary failed: share={fed_share}, face={fed_face}")
    if remittance_cash > 1e-12:
        raise VerificationError(f"remittance cash effect must remain zero: {remittance_cash}")
    statuses = set(str(value) for value in results["NetInterestDiagnosticStatus"].dropna().unique())
    if not statuses:
        raise VerificationError("NetInterestDiagnosticStatus must contain evidence")
    if statuses - {"cbo_reported_check_only", "not_loaded_check_only"}:
        raise VerificationError(f"unexpected net-interest diagnostic statuses: {sorted(statuses)}")
    summary_path = root / "outputs" / "summary.json"
    if not summary_path.exists():
        raise VerificationError("summary.json is required")
    summary = read_json(summary_path)
    _require_summary_keys(summary, REQUIRED_SUMMARY_KEYS)
    _compare_summary(summary, "CBOControlledDebtTargetError_max_abs", target_error)
    _compare_summary(summary, "CBOFedAuctionShare_max_abs", fed_share)
    _compare_summary(summary, "CBOFedAuctionRolloverAddons_max_abs", fed_face)
    return {
        "max_abs_target_error": target_error,
        "max_abs_fed_auction_share": fed_share,
        "max_abs_fed_auction_face": fed_face,
        "max_abs_remittance_cash_effect": remittance_cash,
    }


def _result_artifact_path(root: Path, manifest: dict[str, Any]) -> Path:
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("logical_name", "")).startswith("results_"):
            return root / str(item["relative_path"])
    raise VerificationError("run manifest does not list a results output")


def _read_results(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def _max_abs(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise VerificationError(f"required result column is missing: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.notna().sum() == 0:
        raise VerificationError(f"required result column has no numeric evidence: {column}")
    return float(values.fillna(0.0).abs().max())


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise VerificationError(f"{label} missing required columns: {missing}")


def _require_summary_keys(summary: Any, keys: tuple[str, ...]) -> None:
    if not isinstance(summary, dict):
        raise VerificationError("summary.json must be an object")
    missing = [key for key in keys if key not in summary]
    if missing:
        raise VerificationError(f"summary missing required keys: {missing}")


def _compare_summary(summary: Any, key: str, expected: float) -> None:
    if abs(float(summary[key]) - expected) > 1e-9:
        raise VerificationError(f"summary value mismatch for {key}")


def _reject_absolute_paths(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_paths(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_paths(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            raise VerificationError(f"{path}: absolute path is not allowed")


def _strip_absent_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: item["path"])


__all__ = ["VerificationError", "verify_compiled_scenario", "verify_scenario_run"]
