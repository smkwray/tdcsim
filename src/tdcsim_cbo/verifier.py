"""Independent verifier for compiled and run CBO scenario packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._json import read_json, sha256_file
from .compiler import digest_input_tree
from .output import hash_output_tree


class VerificationError(ValueError):
    """Raised when a compiled or run package fails verification."""


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


def verify_scenario_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest = _manifest(root / "tdcsim_cbo_run_manifest.json")
    compiled_rel = Path(str(manifest.get("compiled_manifest") or ""))
    if compiled_rel.is_absolute() or ".." in compiled_rel.parts:
        raise VerificationError("run manifest compiled_manifest must be package-relative")
    compiled_manifest_path = root / compiled_rel
    compiled = verify_compiled_scenario(compiled_manifest_path.parent)
    outputs = root / "outputs"
    if not outputs.exists():
        raise VerificationError("run outputs directory is missing")
    if manifest.get("output_hashes") != hash_output_tree(outputs):
        raise VerificationError("run output hash list mismatch")
    boundaries = manifest.get("boundary_checks")
    if not isinstance(boundaries, dict):
        raise VerificationError("run manifest boundary_checks must be an object")
    if boundaries.get("net_interest_role") != "diagnostic_nonbinding":
        raise VerificationError("net interest role must remain diagnostic_nonbinding")
    if boundaries.get("remittance_deferred_asset_status") != "unsupported_in_cbo_scenario_lane":
        raise VerificationError("remittance/deferred-asset status must remain unsupported")
    if boundaries.get("fed_target_holder_allocation_only") is not True:
        raise VerificationError("Fed target holder-allocation boundary failed")
    return {"status": "pass", "compiled": compiled, "output_count": len(manifest.get("output_hashes") or [])}


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
