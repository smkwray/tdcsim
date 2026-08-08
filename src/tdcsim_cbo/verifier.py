"""Independent verifier for compiled and run CBO scenario packages."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from datetime import date
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

import evaluated_nominal_curve
import sim_pricing
from evaluated_nominal_curve import (
    OPEN04_FIXED_COUPLING,
    normalize_open04_override,
)
from sim_engine import run_simulation

from ._json import read_json, sha256_file
from .baseline import CboBaselinePackage
from .bounded_output import (
    AGGREGATION_CLOCK_ID,
    ANNUAL_COLUMNS,
    BoundedResourceLimits,
    BoundedScenarioEvidenceSink,
    CASH_TOLERANCE_BIL,
    COMMITMENT_COLUMNS,
    EVENT_SCHEMA_VERSION,
    ISSUANCE_COLUMNS,
    LEDGER_COLUMNS,
    PAYMENT_COLUMNS,
    PRINCIPAL_COLUMNS,
    RESOURCE_COLUMNS,
    STOCK_CLOSURE_COLUMNS,
    STOCK_TOLERANCE_BIL,
    VERIFICATION_GRADE as BOUNDED_VERIFICATION_GRADE,
    EVIDENCE_PROFILE as BOUNDED_EVIDENCE_PROFILE,
    _ALLOWED_EVENT_TYPES,
    _ALLOWED_HOLDERS,
    _ALLOWED_INSTRUMENTS,
    _ALLOWED_MATURITY_BUCKETS,
)
from .compiler import (
    ISSUANCE_MIX_FILE,
    OPEN04_ALLOWED_FIXED_ADAPTER_INPUTS,
    OPEN04_ALLOWED_MATERIALIZED_DEFAULTS,
    CboScenarioCompiler,
    _open04_change_perimeter,
    _open04_simulation_contract,
    digest_input_tree,
)
from .contract import CboScenarioSpec
from .curve_runtime import (
    EvaluatedNominalRuntimeError,
    build_evaluated_nominal_runtime_binding,
    load_compiled_evaluated_nominal_contract,
)
from .manifest import RUN_CLAIM_BOUNDARY, RUN_UNSUPPORTED_COMPONENTS, cash_closure_validation_invariants
from .open04_campaign import (
    OPEN04_FUNDING_CLOSURE_MODE,
    open04_expected_change_perimeter,
    parse_open04_campaign_marker,
    requires_open04_strict_execution,
)
from .output import (
    HANDOFF_TABLE_COLUMNS,
    TDC_AMOUNT_BASIS,
    TDC_COMPONENT_SPECS,
    TDC_HOLDER_SCOPE,
    TDC_IDENTITY_COLUMNS,
    TDC_OVERLAP_COLUMNS,
    TDC_OVERLAP_POLICY,
    _route_stock_closure_handoff_tables,
    hash_output_tree,
    write_bounded_scenario_outputs,
    write_scenario_outputs,
)
from .process_watchdog import THREAD_LIMIT_ENVIRONMENT_VARIABLES
from .runtime_identity import (
    assert_loaded_distribution_modules,
    distribution_identity,
    git_commit_source_identity,
    installed_archive_sha256,
    locked_environment_mismatches,
    verify_wheel_against_git_commit,
    wheel_file_digest,
)
from . import runner as runner_module
from ._schema import validate_schema
from .marginal_tdc import verify_marginal_tdc_pair


class VerificationError(ValueError):
    """Raised when a compiled or run package fails verification."""


REQUIRED_RESULT_COLUMNS = (
    "TGA",
    "CBOCashReconciliationResidual",
    "CBOCashResidualStatus",
    "CBOOperatingCashTarget",
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

RUN_MANIFEST_V1 = "tdcsim_cbo_scenario_run_manifest_v1"
RUN_MANIFEST_V2 = "tdcsim_cbo_scenario_run_manifest_v2"

_BOUNDED_ARTIFACT_SPECS: dict[str, tuple[str, list[str]]] = {
    "ledger": ("tdcsim_period_ledger_totals.csv.gz", LEDGER_COLUMNS),
    "issuance": ("tdcsim_period_issuance_aggregates.csv.gz", ISSUANCE_COLUMNS),
    "principal": ("tdcsim_period_principal_aggregates.csv.gz", PRINCIPAL_COLUMNS),
    "payment": ("tdcsim_period_payment_aggregates.csv.gz", PAYMENT_COLUMNS),
    "accounting": (
        "tdcsim_period_accounting_closure.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_accounting_closure"],
    ),
    "stock": ("tdcsim_period_stock_closure.csv.gz", STOCK_CLOSURE_COLUMNS),
    "route": (
        "tdcsim_period_route_stock_closure.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_tdc_principal_route_stock_closure"],
    ),
    "tdc_summary": (
        "tdcsim_period_tdc_summary.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_summary"],
    ),
    "tdc_components": (
        "tdcsim_period_tdc_components.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_components"],
    ),
    "debt_bridge": (
        "tdcsim_debt_target_bridge.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"],
    ),
    "scenario_metrics": (
        "tdcsim_scenario_metrics.csv.gz",
        HANDOFF_TABLE_COLUMNS["tdcsim_scenario_metrics"],
    ),
    "commitments": ("tdcsim_event_commitments.csv.gz", COMMITMENT_COLUMNS),
    "annual": ("tdcsim_annual_economic_summary.csv.gz", ANNUAL_COLUMNS),
    "resources": ("tdcsim_resource_samples.csv.gz", RESOURCE_COLUMNS),
}
_BOUNDED_DETERMINISTIC_ARTIFACTS = frozenset(_BOUNDED_ARTIFACT_SPECS) - {
    "resources"
}
_BOUNDED_EXECUTION_MILESTONES = (
    "compile_complete",
    "engine_started",
    "engine_complete",
    "source_evidence_finalized",
    "bounded_period_closure_passed",
    "run_manifest_finalized",
)
_BOUNDED_WATCHDOG_ACCEPTED_MILESTONES = (
    *_BOUNDED_EXECUTION_MILESTONES,
    "parent_watchdog_accepted",
)
_BOUNDED_TARGET_TOLERANCE_BIL = 1e-6

_BOUNDED_TDC_SUMMARY_RESULT_BRIDGES = {
    "tdc_change_bil": "TDC_Change",
    "tdc_fiscal_flow_bil": "TDC_FiscalFlow",
    "tdc_debt_service_bil": "TDC_DebtService",
    "tdc_debt_service_principal_to_du_bil": "TDC_PrincipalToDU",
    "tdc_debt_service_interest_to_du_bil": "TDC_InterestToDU",
    "gross_principal_cash_paid_to_du_bil": "TDC_PrincipalCashToDU",
    "principal_redeemed_to_du_domestic_nonbank_bil": (
        "TDC_PrincipalToDU_DomesticNonbank"
    ),
    "principal_redeemed_to_du_mmf_bil": "TDC_PrincipalToDU_MMF",
    "gross_principal_cash_paid_to_du_domestic_nonbank_bil": (
        "TDC_PrincipalCashToDU_DomesticNonbank"
    ),
    "gross_principal_cash_paid_to_du_mmf_bil": "TDC_PrincipalCashToDU_MMF",
    "gross_principal_cash_paid_to_du_mmf_plumbing_bil": (
        "TDC_PrincipalCashToDU_MMFPlumbing"
    ),
    "tdc_auction_absorption_du_bil": "TDC_AuctionAbsorption",
    "tdc_secondary_trades_bil": "TDC_SecondaryTrades",
    "tdc_other_bil": "TDC_Other",
    "gross_issuance_cash_proceeds_bil": "AuctionProceeds",
    "gross_issuance_proceeds_absorbed_by_du_bil": (
        "TDC_GrossIssuanceProceedsAbsorbedByDU"
    ),
    "net_du_principal_issuance_cashflow_bil": (
        "TDC_NetPrincipalIssuanceCashflowToDU"
    ),
}
_BOUNDED_FINANCE_RESULT_BRIDGES = {
    "interest_outlay_bil": "InterestOutlay_Period",
    "issue_discount_cost_bil": "IssueDiscountCost_Period",
    "nonmarketable_interest_capitalized_bil": (
        "NonMarketableInterestCapitalized_Period"
    ),
    "tips_inflation_accretion_bil": "TIPSInflationAccretion_Period",
    "modeled_financing_cost_bil": "FinancingCost_Period",
}


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
    _verify_compiled_lineage(root, manifest)
    _verify_claim_boundary(manifest)
    evaluated_nominal_curve = _verify_compiled_evaluated_nominal_curve(
        root, manifest
    )
    scenario_contract = manifest.get("scenario_contract")
    open04_marker = (
        parse_open04_campaign_marker(scenario_contract)
        if isinstance(scenario_contract, Mapping)
        else None
    )
    return {
        "status": "pass",
        "compiled_inputs_digest": actual_digest,
        "input_count": len(expected_hashes or []),
        "evaluated_nominal_curve": evaluated_nominal_curve,
        "open04_campaign": (
            {
                "contract_id": open04_marker.contract_id,
                "role": open04_marker.role,
            }
            if open04_marker is not None
            else None
        ),
    }


def verify_scenario_run(
    run_dir: str | Path,
    *,
    baseline_package: str | Path | None = None,
    attestation: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    manifest = _manifest(root / "tdcsim_cbo_run_manifest.json")
    _validate_run_manifest_schema(manifest)
    bounded = _is_bounded_manifest(manifest)
    _verify_release_claims(manifest)
    _verify_run_claim_boundary(manifest)
    _verify_validation_block(manifest)
    compiled_rel = Path(str(manifest.get("compiled_manifest") or ""))
    if compiled_rel.is_absolute() or ".." in compiled_rel.parts:
        raise VerificationError("run manifest compiled_manifest must be package-relative")
    compiled_manifest_path = root / compiled_rel
    compiled = verify_compiled_scenario(compiled_manifest_path.parent)
    compiled_manifest = _manifest(compiled_manifest_path)
    compiled_scenario_contract = compiled_manifest.get("scenario_contract")
    if not isinstance(compiled_scenario_contract, Mapping):
        raise VerificationError(
            "compiled scenario contract must be an object"
        )
    open04_active = requires_open04_strict_execution(
        compiled_scenario_contract
    )
    _verify_code_environment(
        root,
        manifest,
        require_release_identity=open04_active,
    )
    if compiled["compiled_inputs_digest"] != manifest.get("compiled_inputs_digest"):
        raise VerificationError("run manifest compiled input digest does not match actual compiled tree")
    run_scenario = manifest.get("scenario")
    if not isinstance(run_scenario, Mapping):
        raise VerificationError("run manifest scenario must be an object")
    _verify_compiled_run_scenario_identity(compiled_manifest, run_scenario)
    _verify_compiled_run_lineage(manifest, compiled_manifest)
    outputs = root / "outputs"
    if not outputs.exists():
        raise VerificationError("run outputs directory is missing")
    if manifest.get("output_hashes") != hash_output_tree(outputs):
        raise VerificationError("run output hash list mismatch")
    _verify_manifest_artifacts(root, manifest.get("outputs"), base=root)
    _verify_manifest_artifacts(root, manifest.get("compiled_inputs"), base=root)
    _verify_exact_compiled_input_artifacts(manifest, compiled_manifest)
    _verify_scenario_copy(root, manifest)
    _verify_run_evaluated_nominal_curve(
        compiled_manifest_path.parent,
        root,
        manifest,
        compiled_curve=compiled.get("evaluated_nominal_curve"),
    )
    verification_grade = (
        BOUNDED_VERIFICATION_GRADE if bounded else "local"
    )
    if baseline_package is not None or attestation is not None:
        if baseline_package is None or attestation is None:
            raise VerificationError("baseline_package and attestation must be supplied together")
        _verify_recompile(root, manifest, baseline_package, attestation)
        verification_grade = (
            BOUNDED_VERIFICATION_GRADE if bounded else "replay"
        )
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
        "verification_grade": verification_grade,
        "compiled": compiled,
        "output_count": len(manifest.get("output_hashes") or []),
        "recomputed": recomputed,
    }


def _verify_compiled_run_scenario_identity(
    compiled_manifest: Mapping[str, Any],
    run_scenario: Mapping[str, Any],
) -> None:
    if compiled_manifest.get("scenario_sha256") != run_scenario.get(
        "canonical_sha256"
    ):
        raise VerificationError(
            "compiled scenario hash does not match run scenario canonical hash"
        )
    if compiled_manifest.get("scenario_id") != run_scenario.get(
        "scenario_id"
    ):
        raise VerificationError(
            "compiled scenario ID does not match run scenario ID"
        )


def _verify_compiled_lineage(
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    scenario_contract = manifest.get("scenario_contract")
    if not isinstance(scenario_contract, Mapping):
        raise VerificationError(
            "compiled manifest scenario_contract must be an object"
        )
    try:
        spec = CboScenarioSpec.from_mapping(scenario_contract)
    except ValueError as exc:
        raise VerificationError(
            f"compiled scenario contract is invalid: {exc}"
        ) from exc
    if (
        spec.scenario_id != manifest.get("scenario_id")
        or spec.canonical_sha256() != manifest.get("scenario_sha256")
        or spec.data.get("baseline") != manifest.get("baseline")
        or spec.data.get("coupling") != manifest.get("coupling")
    ):
        raise VerificationError(
            "compiled scenario contract identity, baseline, or coupling mismatch"
        )
    baseline_root = root.parent / "baseline"
    baseline_inputs = baseline_root / "forecast_inputs"
    if not baseline_inputs.is_dir():
        raise VerificationError(
            "compiled verification requires the materialized baseline inputs"
        )
    actual_baseline_digest = digest_input_tree(baseline_inputs)
    if manifest.get("baseline_forecast_inputs_digest") != actual_baseline_digest:
        raise VerificationError(
            "compiled baseline forecast-input digest mismatch"
        )
    baseline = manifest.get("baseline")
    required_baseline_fields = {
        "package_id",
        "package_sha256",
        "manifest_sha256",
        "release_attestation_sha256",
    }
    if (
        not isinstance(baseline, Mapping)
        or set(baseline) != required_baseline_fields
        or not str(baseline.get("package_id") or "")
    ):
        raise VerificationError(
            "compiled baseline identity block is invalid"
        )
    for field in (
        "package_sha256",
        "manifest_sha256",
        "release_attestation_sha256",
    ):
        value = str(baseline.get(field) or "")
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise VerificationError(
                f"compiled baseline {field} is not a SHA-256"
            )
    materialized_manifest = baseline_root / "manifest.json"
    source_contract = (
        baseline_root / "forecast_inputs" / "source_contract_smoke.json"
    )
    if (
        not materialized_manifest.is_file()
        or not source_contract.is_file()
        or sha256_file(materialized_manifest)
        != baseline["manifest_sha256"]
        or sha256_file(source_contract) != baseline["manifest_sha256"]
    ):
        raise VerificationError(
            "compiled baseline manifest identity does not match materialized bytes"
        )
    defaults = manifest.get("materialized_defaults")
    if (
        not isinstance(defaults, list)
        or not all(isinstance(item, str) for item in defaults)
        or len(defaults) != len(set(defaults))
        or manifest.get("materialized_default_count") != len(defaults)
    ):
        raise VerificationError(
            "compiled materialized-default count or set is invalid"
        )


def _verify_compiled_run_lineage(
    run_manifest: Mapping[str, Any],
    compiled_manifest: Mapping[str, Any],
) -> None:
    if run_manifest.get("baseline") != compiled_manifest.get("baseline"):
        raise VerificationError(
            "run baseline identity does not match compiled baseline"
        )
    if run_manifest.get("coupling_decisions") != compiled_manifest.get(
        "coupling"
    ):
        raise VerificationError(
            "run coupling decisions do not match compiled coupling"
        )
    if run_manifest.get("open04_campaign") != compiled_manifest.get(
        "open04_campaign"
    ):
        raise VerificationError(
            "run OPEN-04 campaign marker does not match compiled lineage"
        )


def _verify_exact_compiled_input_artifacts(
    run_manifest: Mapping[str, Any],
    compiled_manifest: Mapping[str, Any],
) -> None:
    expected_hashes = compiled_manifest.get("input_hashes")
    artifacts = run_manifest.get("compiled_inputs")
    if not isinstance(expected_hashes, list) or not isinstance(artifacts, list):
        raise VerificationError(
            "compiled input hashes and run artifacts must be arrays"
        )
    expected = {
        str(item.get("path")): (
            str(item.get("sha256")),
            int(item.get("bytes", -1)),
        )
        for item in expected_hashes
        if isinstance(item, Mapping)
    }
    observed: dict[str, tuple[str, int]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise VerificationError(
                "run compiled-input artifact must be an object"
            )
        logical_name = str(item.get("logical_name") or "")
        rel = Path(str(item.get("relative_path") or ""))
        prefix = Path("compile") / "compiled" / "forecast_inputs"
        try:
            path = rel.relative_to(prefix).as_posix()
        except ValueError as exc:
            raise VerificationError(
                "run compiled-input artifact is outside the compiled input tree"
            ) from exc
        if logical_name != path or path in observed:
            raise VerificationError(
                "run compiled-input artifact names are duplicated or stale"
            )
        observed[path] = (
            str(item.get("sha256") or ""),
            int(item.get("bytes", -1)),
        )
    if observed != expected:
        raise VerificationError(
            "run compiled-input artifacts do not exactly match compiled hashes"
        )


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


def _verify_compiled_evaluated_nominal_curve(
    root: Path,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_contract = manifest.get("scenario_contract")
    if not isinstance(scenario_contract, Mapping):
        raise VerificationError(
            "compiled manifest scenario_contract must be an object"
        )
    open04_marker = parse_open04_campaign_marker(scenario_contract)
    open04_active = requires_open04_strict_execution(scenario_contract)
    expected_marker = (
        dict(scenario_contract["open04_campaign"])
        if open04_marker is not None
        else None
    )
    if (
        expected_marker is None
        and "open04_campaign" in manifest
    ) or (
        expected_marker is not None
        and manifest.get("open04_campaign") != expected_marker
    ):
        raise VerificationError(
            "compiled OPEN-04 campaign marker does not match the scenario"
        )
    try:
        shock, metadata = load_compiled_evaluated_nominal_contract(
            root / "forecast_inputs"
        )
    except EvaluatedNominalRuntimeError as exc:
        raise VerificationError(str(exc)) from exc
    if shock is None and not open04_active:
        if any(
            key in manifest
            for key in (
                "open04_simulation_contract",
                "open04_change_perimeter",
            )
        ):
            raise VerificationError(
                "non-OPEN-04 compiled manifest carries OPEN-04 contract blocks"
            )
        return None
    changed = manifest.get("changed_inputs")
    if not isinstance(changed, list):
        raise VerificationError(
            "compiled manifest changed_inputs must be an array"
        )
    if shock is None:
        if open04_marker is None or open04_marker.role != "baseline":
            raise VerificationError(
                "OPEN-04 campaign candidate is missing its evaluated curve sidecar"
            )
        if manifest.get("overrides_applied") != []:
            raise VerificationError(
                "compiled OPEN-04 baseline must not apply scenario overrides"
            )
        expected_physical: set[str] = set()
        economic_paths: tuple[str, ...] = ()
        scenario_changed: set[str] = set()
    else:
        if open04_marker is None:
            expected_physical = {
                ISSUANCE_MIX_FILE,
                "tdcsim_nominal_curve_evaluated_shock.json",
            }
            economic_paths = (
                "issuance_mix",
                "nominal_yield_curve_assumption",
            )
            expected_overrides = ["issuance_mix", "nominal_yield_curve"]
        else:
            economic_paths, physical_paths = (
                open04_expected_change_perimeter(open04_marker.role)
            )
            expected_physical = set(physical_paths)
            expected_overrides = ["issuance_mix", "nominal_yield_curve"]
            if open04_marker.role == "candidate_a":
                expected_overrides.insert(0, "holder_preferences")
        missing_changed = expected_physical - set(changed)
        if missing_changed:
            raise VerificationError(
                "compiled OPEN-04 physical inputs are absent from "
                f"changed_inputs: {sorted(missing_changed)}"
            )
        if manifest.get("overrides_applied") != expected_overrides:
            raise VerificationError(
                "evaluated nominal mode overrides differ from the approved "
                "role-specific perimeter"
            )
        scenario_changed = set(expected_physical)
    if "tdcsim_yield_curve_surface.csv" in changed:
        raise VerificationError(
            "evaluated nominal mode must not mark the baseline surface changed"
        )
    if manifest.get("coupling") != dict(OPEN04_FIXED_COUPLING):
        raise VerificationError(
            "compiled OPEN-04 coupling does not match the fixed contract"
        )
    try:
        simulation_contract = _open04_simulation_contract(
            root / "forecast_inputs",
            scenario=None,
        )
    except ValueError as exc:
        raise VerificationError(
            f"compiled OPEN-04 simulation contract is invalid: {exc}"
        ) from exc
    if manifest.get("open04_simulation_contract") != simulation_contract:
        raise VerificationError(
            "compiled OPEN-04 simulation contract does not match inputs"
        )
    baseline_surface = (
        root.parent
        / "baseline"
        / "forecast_inputs"
        / "tdcsim_yield_curve_surface.csv"
    )
    compiled_surface = (
        root / "forecast_inputs" / "tdcsim_yield_curve_surface.csv"
    )
    if not baseline_surface.exists():
        raise VerificationError(
            "evaluated nominal verification requires the materialized baseline surface"
        )
    if sha256_file(baseline_surface) != sha256_file(compiled_surface):
        raise VerificationError(
            "evaluated nominal compiled surface differs from the baseline bytes"
        )
    try:
        materialized_defaults = manifest.get("materialized_defaults")
        if not isinstance(materialized_defaults, list) or not all(
            isinstance(item, str) for item in materialized_defaults
        ):
            raise VerificationError(
                "compiled OPEN-04 materialized_defaults must be an array of paths"
            )
        materialized_set = set(materialized_defaults)
        if (
            len(materialized_set) != len(materialized_defaults)
            or not materialized_set <= OPEN04_ALLOWED_MATERIALIZED_DEFAULTS
        ):
            raise VerificationError(
                "compiled OPEN-04 materialized_defaults are outside the approved set"
            )
        declared_perimeter = manifest.get("open04_change_perimeter")
        if not isinstance(declared_perimeter, Mapping):
            raise VerificationError(
                "compiled OPEN-04 change perimeter must be an object"
            )
        fixed_adapter_inputs = declared_perimeter.get(
            "fixed_adapter_inputs"
        )
        if not isinstance(fixed_adapter_inputs, list) or not all(
            isinstance(item, str) for item in fixed_adapter_inputs
        ):
            raise VerificationError(
                "compiled OPEN-04 fixed_adapter_inputs must be an array "
                "of paths"
            )
        fixed_adapter_set = set(fixed_adapter_inputs)
        if (
            len(fixed_adapter_set) != len(fixed_adapter_inputs)
            or not fixed_adapter_set
            <= OPEN04_ALLOWED_FIXED_ADAPTER_INPUTS
        ):
            raise VerificationError(
                "compiled OPEN-04 fixed_adapter_inputs are outside the "
                "approved set"
            )
        actual_perimeter = _open04_change_perimeter(
            root.parent / "baseline" / "forecast_inputs",
            root / "forecast_inputs",
            scenario_changed=scenario_changed,
            materialized_defaults=materialized_set,
            fixed_adapter_inputs=fixed_adapter_set,
            changed_inputs=changed,
            expected_physical_inputs=expected_physical,
            economic_changed_paths=economic_paths,
        )
    except (ValueError, VerificationError) as exc:
        raise VerificationError(
            f"compiled OPEN-04 change perimeter is invalid: {exc}"
        ) from exc
    if manifest.get("open04_change_perimeter") != actual_perimeter:
        raise VerificationError(
            "compiled OPEN-04 change perimeter does not match runtime bytes"
        )
    return metadata if shock is not None else None


def _verify_run_evaluated_nominal_curve(
    compiled_dir: Path,
    run_root: Path,
    manifest: dict[str, Any],
    *,
    compiled_curve: Any,
) -> None:
    compiled_active = isinstance(compiled_curve, Mapping)
    runtime_block = manifest.get("evaluated_nominal_curve")
    if compiled_active != isinstance(runtime_block, Mapping):
        raise VerificationError(
            "run evaluated nominal curve block does not match compiled sidecar presence"
        )
    scenario = manifest.get("scenario")
    if not isinstance(scenario, Mapping):
        raise VerificationError("run scenario must be an object")
    scenario_path = run_root / Path(str(scenario.get("relative_path") or ""))
    spec = CboScenarioSpec.from_file(scenario_path)
    open04_marker = parse_open04_campaign_marker(spec.data)
    expected_marker = None
    if open04_marker is not None:
        expected_marker = {
            "contract_id": open04_marker.contract_id,
            "role": open04_marker.role,
            "funding_closure_mode": open04_marker.funding_closure_mode,
        }
    if manifest.get("open04_campaign") != expected_marker:
        raise VerificationError(
            "run OPEN-04 campaign marker does not match the scenario copy"
        )
    open04_active = requires_open04_strict_execution(spec.data)
    overrides = spec.data.get("overrides")
    nominal = (
        overrides.get("nominal_yield_curve")
        if isinstance(overrides, Mapping)
        else None
    )
    scenario_active = (
        isinstance(nominal, Mapping)
        and nominal.get("mode") == "evaluated_additive_key_rate_bp"
    )
    if scenario_active != compiled_active:
        raise VerificationError(
            "scenario evaluated nominal mode does not match compiled sidecar"
        )
    if compiled_active:
        normalized_nominal = normalize_open04_override(nominal)
        signed_scenario_shock = float(
            normalized_nominal["shocks"][1]["shock_bp"]
        )
        if signed_scenario_shock != float(
            compiled_curve.get("signed_10y_shock_bp")
        ):
            raise VerificationError(
                "run scenario signed 10-year shock does not match compiled sidecar"
            )
    simulation = manifest.get("simulation")
    if not isinstance(simulation, Mapping):
        raise VerificationError("run simulation block must be an object")
    compiled_manifest = _manifest(
        compiled_dir / "tdcsim_cbo_compiled_manifest.json"
    )
    simulation_contract = compiled_manifest.get(
        "open04_simulation_contract"
    )
    if not isinstance(simulation_contract, Mapping):
        if open04_active:
            raise VerificationError(
                "compiled OPEN-04 simulation contract must be an object"
            )
        return
    expected_simulation = {
        "start_date": simulation_contract.get("start_date"),
        "end_date": simulation_contract.get("end_date"),
        "frequency": simulation_contract.get("frequency"),
    }
    if dict(simulation) != expected_simulation:
        raise VerificationError(
            "run simulation dates do not match the compiled OPEN-04 horizon"
        )
    _verify_open04_output_contract(manifest, simulation_contract)
    if not compiled_active:
        if open04_marker is None or open04_marker.role != "baseline":
            raise VerificationError(
                "OPEN-04 run without a curve sidecar must be the marked baseline"
            )
        return
    try:
        actual = build_evaluated_nominal_runtime_binding(
            compiled_dir / "forecast_inputs",
            start_date=str(simulation.get("start_date") or ""),
            end_date=str(simulation.get("end_date") or ""),
        )
    except EvaluatedNominalRuntimeError as exc:
        raise VerificationError(str(exc)) from exc
    if actual != dict(runtime_block):
        raise VerificationError(
            "run evaluated nominal curve evidence does not match runtime bytes"
        )
    if (
        actual.get("runtime_selected_curve_date_count")
        != simulation_contract.get("runtime_selected_curve_date_count")
        or actual.get("runtime_selected_curve_date_set_sha256")
        != simulation_contract.get(
            "runtime_selected_curve_date_set_sha256"
        )
    ):
        raise VerificationError(
            "run evaluated nominal curve date set does not match compiled horizon"
        )


def _verify_open04_output_contract(
    manifest: Mapping[str, Any],
    simulation_contract: Mapping[str, Any],
) -> None:
    expected = {
        "profile": simulation_contract.get("output_profile"),
        "compression": simulation_contract.get("compression"),
    }
    if expected != {"profile": "compact", "compression": "gzip"}:
        raise VerificationError(
            "compiled OPEN-04 output contract is not compact/gzip"
        )
    output_manifest = manifest.get("output_manifest")
    if not isinstance(output_manifest, Mapping):
        raise VerificationError("run output_manifest must be an object")
    observed = {
        "profile": output_manifest.get("profile"),
        "compression": output_manifest.get("compression"),
    }
    if observed != expected:
        raise VerificationError(
            "run output manifest disagrees with the compiled OPEN-04 output contract"
        )


def _validate_run_manifest_schema(manifest: dict[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    schema_names = {
        RUN_MANIFEST_V1: "cbo-run-manifest-v1.schema.json",
        RUN_MANIFEST_V2: "cbo-run-manifest-v2.schema.json",
    }
    schema_name = schema_names.get(schema_version)
    if schema_name is None:
        raise VerificationError(
            f"unsupported run manifest schema_version: {schema_version!r}"
        )
    with files("tdcsim_cbo").joinpath(f"schemas/{schema_name}").open(
        "r", encoding="utf-8"
    ) as handle:
        schema = json.load(handle)
    try:
        validate_schema(manifest, schema, label="run_manifest")
    except Exception as exc:
        raise VerificationError(f"run manifest schema validation failed: {exc}") from exc
    if schema_version == RUN_MANIFEST_V2:
        milestones = manifest.get("execution_milestones")
        has_acceptance = isinstance(milestones, list) and (
            "parent_watchdog_accepted" in milestones
        )
        if has_acceptance != ("parent_watchdog" in manifest):
            raise VerificationError(
                "run manifest schema validation failed: parent_watchdog "
                "must be present exactly with parent_watchdog_accepted"
            )


def _verify_release_claims(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "complete":
        raise VerificationError("run manifest status must be complete")
    grade = manifest.get("verification_grade")
    if _is_bounded_manifest(manifest):
        if grade != BOUNDED_VERIFICATION_GRADE:
            raise VerificationError(
                f"unsupported bounded verification_grade: {grade!r}"
            )
        if manifest.get("evidence_profile") != BOUNDED_EVIDENCE_PROFILE:
            raise VerificationError("bounded evidence_profile is unsupported")
        return
    if grade == "release_verified":
        raise VerificationError("release_verified grade requires external package identity evidence")
    if grade not in {"local", "release_reproducible"}:
        raise VerificationError(f"unsupported verification_grade: {grade!r}")


def _is_bounded_manifest(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("schema_version") == RUN_MANIFEST_V2


def _manifest_funding_closure_mode(manifest: Mapping[str, Any]) -> str:
    """Return the declared OPEN-04 mode; generic runs retain the legacy mode."""

    open04 = manifest.get("open04_campaign")
    if open04 is None:
        return "cbo_public_debt_target"
    if not isinstance(open04, Mapping):
        raise VerificationError("run manifest open04_campaign must be an object")
    if open04.get("funding_closure_mode") != OPEN04_FUNDING_CLOSURE_MODE:
        raise VerificationError(
            "run manifest OPEN-04 funding closure mode is missing or unsupported"
        )
    return OPEN04_FUNDING_CLOSURE_MODE


def _verify_code_environment(
    root: Path,
    manifest: dict[str, Any],
    *,
    require_release_identity: bool = False,
) -> None:
    env = manifest.get("code_environment")
    if not isinstance(env, dict):
        raise VerificationError("run manifest code_environment must be an object")
    expected = _actual_code_environment()
    _verify_python_version(str(env.get("python_version") or ""))
    for key, expected_value in expected.items():
        if env.get(key) != expected_value:
            raise VerificationError(f"run manifest code_environment {key} does not match verifier runtime")
    lock_sha = str(env.get("requirements_lock_sha256") or "")
    if not lock_sha or lock_sha == "0" * 64:
        raise VerificationError("run manifest code environment requirements lock hash is not release-bound")
    expected_lock = os.environ.get("TDCSIM_CBO_REQUIREMENTS_LOCK_SHA256", "")
    if expected_lock and lock_sha != expected_lock:
        raise VerificationError("run manifest code environment requirements lock hash does not match verifier runtime")
    if require_release_identity:
        retained_lock = root / "compile" / "baseline" / "requirements.lock.txt"
        if not retained_lock.is_file():
            raise VerificationError(
                "OPEN-04 retained baseline requirements lock is missing"
            )
        if sha256_file(retained_lock) != lock_sha:
            raise VerificationError(
                "OPEN-04 code environment requirements lock hash does not "
                "match the retained baseline lock"
            )
        try:
            dependency_mismatches = locked_environment_mismatches(
                retained_lock.read_bytes()
            )
        except Exception as exc:
            raise VerificationError(
                "OPEN-04 retained requirements lock could not be checked "
                "against the verifier runtime"
            ) from exc
        if dependency_mismatches:
            raise VerificationError(
                "OPEN-04 verifier dependency versions do not match the "
                f"requirements lock: {dependency_mismatches}"
            )
        commit = str(env.get("code_commit_sha") or "")
        if (
            len(commit) != 40
            or commit == "0" * 40
            or any(char not in "0123456789abcdef" for char in commit)
        ):
            raise VerificationError(
                "OPEN-04 code environment commit is not release-bound"
            )
        if env.get("dirty_state") is not False:
            raise VerificationError(
                "OPEN-04 code environment must record a clean producer"
            )
        if env.get("runtime_import_mode") != "installed_distribution":
            raise VerificationError(
                "OPEN-04 runtime must execute the installed distribution"
            )
        if env.get("source_shadow_guard") is not True:
            raise VerificationError(
                "OPEN-04 runtime source-shadow guard is not recorded"
            )
        if str(env.get("python_version") or "") != platform.python_version():
            raise VerificationError(
                "OPEN-04 Python patch version does not match verifier runtime"
            )
    wheel_path = _verify_wheel_artifact(
        root,
        env,
        required=require_release_identity,
    )
    if require_release_identity:
        if wheel_path is None:
            raise VerificationError(
                "OPEN-04 retained release wheel unexpectedly resolved to no path"
            )
        _verify_producer_source_identity(env, wheel_path)


def _verify_wheel_artifact(
    root: Path,
    env: dict[str, Any],
    *,
    required: bool = False,
) -> Path | None:
    wheel_sha = str(env.get("wheel_sha256") or "")
    artifact = env.get("wheel_artifact")
    expected_wheel_sha = os.environ.get("TDCSIM_CBO_WHEEL_SHA256", "")
    if not wheel_sha:
        if artifact is not None:
            raise VerificationError("run manifest wheel_artifact is present without wheel_sha256")
        if required:
            raise VerificationError(
                "OPEN-04 run manifest requires retained release wheel bytes"
            )
        return None
    commit = str(env.get("code_commit_sha") or "")
    if len(commit) != 40 or commit == "0" * 40 or any(char not in "0123456789abcdef" for char in commit):
        raise VerificationError("run manifest code environment code_commit_sha is not release-bound")
    expected_commit = os.environ.get("TDCSIM_CBO_CODE_COMMIT_SHA", "")
    if expected_commit and commit != expected_commit:
        raise VerificationError("run manifest code environment code_commit_sha does not match verifier runtime")
    if env.get("dirty_state") is not False:
        raise VerificationError("run manifest code environment dirty_state must be false for release wheel runs")
    if not isinstance(artifact, dict):
        raise VerificationError("run manifest code environment wheel_artifact is required when wheel_sha256 is set")
    if expected_wheel_sha and wheel_sha != expected_wheel_sha:
        raise VerificationError("run manifest code environment wheel_sha256 does not match verifier runtime")
    rel = Path(str(artifact.get("relative_path") or ""))
    if rel.is_absolute() or ".." in rel.parts:
        raise VerificationError("run manifest wheel_artifact path must be package-relative")
    path = root / rel
    if not path.is_file():
        raise VerificationError("run manifest wheel_artifact is missing")
    if sha256_file(path) != artifact.get("sha256") or artifact.get("sha256") != wheel_sha:
        raise VerificationError("run manifest wheel_artifact SHA does not match wheel_sha256")
    if path.stat().st_size != int(artifact.get("bytes", -1)):
        raise VerificationError("run manifest wheel_artifact byte count mismatch")
    try:
        wheel_digest = wheel_file_digest(path)
    except Exception as exc:
        raise VerificationError("run manifest wheel artifact digest could not be computed") from exc
    if wheel_digest != env.get("distribution_file_digest"):
        raise VerificationError("run manifest wheel artifact digest does not match installed runtime files")
    return path


def _verify_producer_source_identity(
    env: Mapping[str, Any],
    wheel_path: Path,
) -> None:
    identity = env.get("producer_source_identity")
    if not isinstance(identity, Mapping):
        raise VerificationError(
            "OPEN-04 producer source qualification receipt is missing"
        )
    source_tree = identity.get("source_tree")
    wheel_binding = identity.get("wheel_git_binding")
    commit = str(env.get("code_commit_sha") or "")
    if (
        not isinstance(source_tree, Mapping)
        or set(source_tree)
        != {
            "release_commit_sha",
            "dirty_state",
            "runtime_identity_source",
            "source_tree_sha256",
            "source_tree_file_count",
            "dependency_lock_files",
            "dependency_lock_set_sha256",
            "python_version",
        }
        or source_tree.get("release_commit_sha") != commit
        or source_tree.get("dirty_state") is not False
        or source_tree.get("runtime_identity_source")
        != "clean_tracked_source_tree_files"
    ):
        raise VerificationError(
            "OPEN-04 producer source-tree identity is invalid"
        )
    if (
        identity.get("installed_archive_sha256")
        != env.get("wheel_sha256")
    ):
        raise VerificationError(
            "OPEN-04 producer installed-archive identity disagrees"
        )
    source_repository = os.environ.get(
        "TDCSIM_CBO_SOURCE_REPOSITORY", ""
    )
    if not source_repository:
        raise VerificationError(
            "OPEN-04 verification requires the declared source repository"
        )
    try:
        expected_source_tree = git_commit_source_identity(
            source_repository,
            commit,
        )
        for field, expected in expected_source_tree.items():
            if source_tree.get(field) != expected:
                raise ValueError(
                    f"producer source-tree receipt disagrees on {field}"
                )
        if source_tree.get("python_version") != env.get("python_version"):
            raise ValueError(
                "producer source-tree Python version disagrees"
            )
        actual_binding = verify_wheel_against_git_commit(
            wheel_path,
            source_repository,
            commit,
        )
        if actual_binding != wheel_binding:
            raise ValueError("wheel/Git binding receipt does not recompute")
        if installed_archive_sha256() != env.get("wheel_sha256"):
            raise ValueError("installed archive SHA-256 differs from retained wheel")
        loaded_modules = assert_loaded_distribution_modules(
            set(runner_module._OPEN04_REQUIRED_RUNTIME_MODULES)
            | {"tdcsim_cbo.verifier"}
        )
    except Exception as exc:
        raise VerificationError(
            "OPEN-04 wheel/source qualification could not be recomputed"
        ) from exc
    producer_count = identity.get("loaded_distribution_module_count")
    if (
        not isinstance(producer_count, int)
        or producer_count < len(runner_module._OPEN04_REQUIRED_RUNTIME_MODULES)
        or len(loaded_modules)
        < len(runner_module._OPEN04_REQUIRED_RUNTIME_MODULES) + 1
    ):
        raise VerificationError(
            "OPEN-04 loaded-distribution module qualification is incomplete"
        )


def _verify_python_version(version: str) -> None:
    parts = version.split(".")
    if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
        raise VerificationError("run manifest code_environment python_version is not a numeric X.Y.Z version")
    major, minor, _patch = (int(part) for part in parts[:3])
    current = platform.python_version_tuple()
    if (major, minor) != (int(current[0]), int(current[1])):
        raise VerificationError("run manifest code_environment python_version does not match verifier Python major.minor")
    if major < 3 or (major == 3 and minor < 11):
        raise VerificationError("run manifest code_environment python_version is below the supported floor")


def _actual_code_environment() -> dict[str, Any]:
    dist = distribution_identity()
    return {
        "runner_version": "tdcsim_cbo_runner_v1",
        "verifier_version": "tdcsim_cbo_verifier_v1",
        "runner_source_sha256": sha256_file(Path(runner_module.__file__)),
        "sim_engine_source_sha256": sha256_file(Path(run_simulation.__code__.co_filename)),
        **runner_module._open04_code_surface_hashes(),
        "package_name": dist["name"],
        "package_version": dist["version"],
        "distribution_file_digest": dist["file_digest"],
        "runtime_identity_source": dist["identity_source"],
    }


def _verify_run_claim_boundary(manifest: dict[str, Any]) -> None:
    if manifest.get("claim_boundary") != RUN_CLAIM_BOUNDARY:
        raise VerificationError("run manifest claim_boundary does not match the supported CBO lane claim")
    if manifest.get("unsupported_components") != RUN_UNSUPPORTED_COMPONENTS:
        raise VerificationError("run manifest unsupported_components does not match the supported CBO lane claim")


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
    required_cash_closure_ids = {"tga_nonnegative", "cash_residual_fully_booked"}
    if not required_cash_closure_ids <= invariant_ids:
        raise VerificationError("cash closure validation invariants are required")


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
    if scenario.get("referenced_file_hashes") != declared:
        raise VerificationError(
            "run scenario referenced_file_hashes do not match scenario declarations"
        )
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
        if _is_bounded_manifest(manifest):
            _verify_bounded_engine_replay(
                root, manifest, compiled.forecast_inputs_dir
            )
        else:
            _verify_engine_replay(root, manifest, compiled.forecast_inputs_dir)


def _build_replay_runtime_params(
    manifest: Mapping[str, Any],
    inputs_dir: Path,
    *,
    row_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the runtime parameters recorded by the production runner."""

    simulation = manifest.get("simulation")
    if not isinstance(simulation, Mapping):
        raise VerificationError("run manifest simulation must be an object")
    start = str(simulation.get("start_date") or "")
    end = str(simulation.get("end_date") or "")
    funding_closure_mode = _manifest_funding_closure_mode(manifest)
    engine_scenario_id = runner_module._compiled_scenario_id(inputs_dir)
    return runner_module.build_runtime_params(
        inputs_dir,
        actuals_available_as_of=str(
            row_metadata.get("actuals_available_as_of") or ""
        ),
        simulation_start_date=start,
        simulation_end_date=end,
        scenario_id=engine_scenario_id,
        funding_closure_mode=funding_closure_mode,
    )


def _verify_engine_replay(root: Path, manifest: dict[str, Any], inputs_dir: Path) -> None:
    simulation = manifest.get("simulation")
    if not isinstance(simulation, dict):
        raise VerificationError("run manifest simulation must be an object")
    start = str(simulation.get("start_date") or "")
    end = str(simulation.get("end_date") or "")
    output_manifest = manifest.get("output_manifest")
    if not isinstance(output_manifest, dict):
        raise VerificationError("run manifest output_manifest must be an object")
    row_metadata = output_manifest.get("row_metadata", {})
    if not isinstance(row_metadata, dict):
        row_metadata = {}
    params = _build_replay_runtime_params(
        manifest, inputs_dir, row_metadata=row_metadata
    )
    params = runner_module._engine_runtime_params(params)
    engine_scenario_id = runner_module._compiled_scenario_id(inputs_dir)
    results, final_portfolio = run_simulation(params, start, end, freq="D", scenario_name=engine_scenario_id)
    profile = str(output_manifest.get("profile") or "compact")
    compression = str(output_manifest.get("compression") or "gzip")
    with tempfile.TemporaryDirectory(prefix="tdcsim-cbo-verify-output-") as tmp:
        replay_outputs = Path(tmp) / "outputs"
        write_scenario_outputs(
            results,
            final_portfolio,
            replay_outputs,
            profile=profile,
            compression=compression,
            catalog_sqlite="catalog_sqlite" in output_manifest,
            metadata=row_metadata,
        )
        if hash_output_tree(replay_outputs) != manifest.get("output_hashes"):
            raise VerificationError("engine replay output hash mismatch")


def _verify_bounded_engine_replay(
    root: Path, manifest: dict[str, Any], inputs_dir: Path
) -> None:
    """Replay v2 through the bounded sink and compare deterministic evidence."""

    simulation = manifest.get("simulation")
    output_manifest = manifest.get("output_manifest")
    bounded = manifest.get("bounded_evidence")
    if not isinstance(simulation, dict):
        raise VerificationError("run manifest simulation must be an object")
    if not isinstance(output_manifest, dict):
        raise VerificationError("run manifest output_manifest must be an object")
    if not isinstance(bounded, dict):
        raise VerificationError("run manifest bounded_evidence must be an object")
    row_metadata = output_manifest.get("row_metadata", {})
    if not isinstance(row_metadata, dict):
        row_metadata = {}
    params = _build_replay_runtime_params(
        manifest, inputs_dir, row_metadata=row_metadata
    )
    params = runner_module._engine_runtime_params(params)
    engine_scenario_id = runner_module._compiled_scenario_id(inputs_dir)
    thresholds = bounded.get("memory_thresholds")
    if not isinstance(thresholds, dict):
        raise VerificationError("bounded replay memory thresholds are missing")
    limits = BoundedResourceLimits(
        minimum_available_bytes=_bounded_int(
            thresholds,
            "minimum_available_bytes",
            label="bounded replay thresholds",
            minimum=0,
        ),
        acceptance_peak_rss_bytes=_bounded_int(
            thresholds,
            "acceptance_peak_rss_bytes",
            label="bounded replay thresholds",
            minimum=0,
        ),
        application_abort_rss_bytes=_bounded_int(
            thresholds,
            "application_abort_rss_bytes",
            label="bounded replay thresholds",
            minimum=0,
        ),
        parent_graceful_stop_rss_bytes=_bounded_int(
            thresholds,
            "parent_graceful_stop_rss_bytes",
            label="bounded replay thresholds",
            minimum=0,
        ),
        parent_kill_rss_bytes=_bounded_int(
            thresholds,
            "parent_kill_rss_bytes",
            label="bounded replay thresholds",
            minimum=0,
        ),
        portfolio_row_budget=_bounded_int(
            bounded,
            "portfolio_row_budget",
            label="bounded replay",
            minimum=1,
        ),
        key_cardinality_budget=_bounded_int(
            bounded,
            "key_cardinality_budget",
            label="bounded replay",
            minimum=1,
        ),
    )
    with tempfile.TemporaryDirectory(
        prefix="tdcsim-cbo-verify-bounded-replay-"
    ) as tmp:
        replay_outputs = Path(tmp) / "outputs"
        sink = BoundedScenarioEvidenceSink(replay_outputs, limits=limits)
        try:
            results, final_portfolio = run_simulation(
                params,
                str(simulation.get("start_date") or ""),
                str(simulation.get("end_date") or ""),
                freq="D",
                scenario_name=engine_scenario_id,
                handoff_sink=sink,
                require_bounded_handoff=True,
            )
        except Exception as exc:
            sink.abort(exc)
            raise VerificationError(f"bounded engine replay failed: {exc}") from exc
        replay_summary = results.attrs.get("bounded_handoff_summary")
        if not isinstance(replay_summary, Mapping):
            raise VerificationError(
                "bounded engine replay did not return finalization evidence"
            )
        for field in (
            "event_schema_version",
            "event_count",
            "event_root_sha256",
            "final_state_sha256",
            "period_count",
            "portfolio_row_budget",
            "max_portfolio_rows",
            "max_active_portfolio_rows",
            "key_cardinality_budget",
            "max_key_cardinality",
        ):
            if replay_summary.get(field) != bounded.get(field):
                raise VerificationError(
                    f"bounded engine replay disagrees on {field}"
                )
        if (
            limits.acceptance_peak_rss_bytes
            and int(replay_summary.get("peak_rss_bytes", -1))
            > limits.acceptance_peak_rss_bytes
        ):
            raise VerificationError(
                "bounded engine replay exceeds the acceptance RSS limit"
            )
        if (
            replay_summary.get("deterministic_artifacts")
            != bounded.get("deterministic_artifacts")
        ):
            raise VerificationError(
                "bounded engine replay deterministic artifact hashes mismatch"
            )

        replay_manifest = write_bounded_scenario_outputs(
            results,
            final_portfolio,
            replay_outputs,
            bounded_summary=replay_summary,
            profile=str(output_manifest.get("profile") or "compact"),
            compression=str(output_manifest.get("compression") or "gzip"),
            metadata=row_metadata,
        )
        for logical_name in ("results", "final_portfolio"):
            expected = output_manifest.get(logical_name)
            actual = replay_manifest.get(logical_name)
            if expected is None and actual is None:
                continue
            if not isinstance(expected, dict) or not isinstance(actual, dict):
                raise VerificationError(
                    f"bounded engine replay output presence differs: {logical_name}"
                )
            for field in ("path", "sha256", "bytes"):
                if actual.get(field) != expected.get(field):
                    raise VerificationError(
                        f"bounded engine replay output mismatch: "
                        f"{logical_name}.{field}"
                    )


def _verify_output_invariants(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    results_path = _result_artifact_path(root, manifest)
    results = _read_results(results_path)
    _require_columns(results, REQUIRED_RESULT_COLUMNS, label="results")
    cash_closure = runner_module._cash_closure_checks(results)
    if cash_closure["tga_nonnegative"] is not True:
        raise VerificationError(
            "negative TGA breaches the modeled cash chain: "
            f"min_tga={cash_closure['min_tga']}, periods={cash_closure['negative_tga_periods']}"
        )
    if cash_closure["cash_residual_fully_booked"] is not True:
        raise VerificationError(
            "cash reconciliation residual is not fully booked: "
            f"sum_abs={cash_closure['sum_abs_unbooked_cash_residual']}"
        )
    compiled_rel = Path(str(manifest.get("compiled_manifest") or ""))
    inputs_dir = root / compiled_rel.parent / "forecast_inputs"
    try:
        omf_comparison = runner_module.omf_reconciliation(results, inputs_dir)
    except Exception as exc:
        raise VerificationError(f"operating-cash comparison could not be recomputed: {exc}") from exc
    recomputed_cash = {**cash_closure, **omf_comparison}
    _verify_recomputed_cash_claims(manifest, recomputed_cash)
    bounded_limits = (
        _verify_bounded_result_limits(results)
        if _is_bounded_manifest(manifest)
        else {}
    )
    target_error = (
        bounded_limits["max_abs_controlled_debt_target_error"]
        if bounded_limits
        else _max_abs(results, "CBOControlledDebtTargetError")
    )
    fed_share = _max_abs(results, "CBOFedAuctionShare")
    fed_face = _max_abs(results, "CBOFedAuctionRolloverAddons")
    remittance_cash = _max_abs(results, "CBORemittanceCashEffect")
    target_tolerance = (
        _BOUNDED_TARGET_TOLERANCE_BIL if bounded_limits else 1e-5
    )
    if target_error > target_tolerance:
        raise VerificationError(f"CBO target error exceeds tolerance: {target_error}")
    funding_closure_mode = _manifest_funding_closure_mode(manifest)
    if funding_closure_mode == "cbo_debt_reference_plus_tga_floor_financing_v1":
        _verify_cbo_reference_financing_results(
            results,
            require_target_applicable=not _is_bounded_manifest(manifest),
        )
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
    if _is_bounded_manifest(manifest):
        bounded = _verify_bounded_outputs(root, manifest, results)
    else:
        _verify_tdc_handoff_outputs(root, manifest)
        bounded = {}
    return {
        "max_abs_target_error": target_error,
        "max_abs_fed_auction_share": fed_share,
        "max_abs_fed_auction_face": fed_face,
        **bounded_limits,
        "max_abs_remittance_cash_effect": remittance_cash,
        **recomputed_cash,
        **bounded,
    }


def _verify_cbo_reference_financing_results(
    results: pd.DataFrame,
    *,
    require_target_applicable: bool = True,
) -> None:
    columns = (
        "CBOControlledDebtReference",
        "ScenarioControlledDebt",
        "DebtDriftFromReference",
        "CashFinancingFaceIssued",
        "CashFinancingProceeds",
        "IssuePriceCashGap",
        "CBOControlledDebtTarget",
        "CBOControlledDebtPostIssuance",
        "CBOControlledDebtTargetError",
        "NewDebtIssued",
        "AuctionProceeds",
    )
    _require_columns(results, columns, label="reference-financing results")
    if require_target_applicable and "CBOControlledDebtTargetApplicable" not in results:
        raise VerificationError(
            "reference-financing results missing required columns: "
            "['CBOControlledDebtTargetApplicable']"
        )
    numeric = {
        column: pd.to_numeric(results[column], errors="raise")
        for column in columns
    }
    tolerance = _BOUNDED_TARGET_TOLERANCE_BIL
    identities = (
        (
            numeric["CBOControlledDebtReference"]
            - numeric["CBOControlledDebtTarget"],
            "CBO debt reference/legacy reference alias",
        ),
        (
            numeric["ScenarioControlledDebt"]
            - numeric["CBOControlledDebtPostIssuance"],
            "scenario debt/post-issuance debt",
        ),
        (
            numeric["DebtDriftFromReference"]
            - (
                numeric["ScenarioControlledDebt"]
                - numeric["CBOControlledDebtReference"]
            ),
            "debt drift identity",
        ),
        (
            numeric["IssuePriceCashGap"]
            - (numeric["NewDebtIssued"] - numeric["AuctionProceeds"]),
            "signed issue-price cash-gap identity",
        ),
    )
    for error, label in identities:
        if float(error.abs().max()) > tolerance:
            raise VerificationError(f"reference-financing {label} failed")
    for column in ("CashFinancingFaceIssued", "CashFinancingProceeds"):
        if float(numeric[column].min()) < -tolerance:
            raise VerificationError(
                f"reference-financing {column} must be nonnegative"
            )
    if float(numeric["DebtDriftFromReference"].min()) < -tolerance:
        raise VerificationError(
            "reference-financing scenario debt may not fall below the CBO reference"
        )
    if (
        "CBOControlledDebtTargetApplicable" in results
        and float(
            pd.to_numeric(
                results["CBOControlledDebtTargetApplicable"], errors="raise"
            ).abs().max()
        )
        > tolerance
    ):
        raise VerificationError(
            "reference-financing CBO path must be reported as nonbinding"
        )
    if float(numeric["CBOControlledDebtTargetError"].abs().max()) > tolerance:
        raise VerificationError(
            "reference-financing legacy target-error field must remain inapplicable"
        )


def _verify_bounded_result_limits(results: pd.DataFrame) -> dict[str, float]:
    limits = {
        "CBOControlledDebtTargetError": (
            "max_abs_controlled_debt_target_error",
            "bounded controlled-debt target error",
        ),
        "CBOFedHoldingsTargetError": (
            "max_abs_fed_holdings_target_error",
            "bounded Fed holdings target error",
        ),
    }
    observed: dict[str, float] = {}
    for column, (result_key, label) in limits.items():
        values = _strict_numeric(results, column, label="bounded results")
        if values.empty:
            raise VerificationError(
                f"bounded results contain no evidence for {column}"
            )
        maximum = float(values.abs().max())
        if maximum > _BOUNDED_TARGET_TOLERANCE_BIL:
            raise VerificationError(f"{label} exceeds 1e-6 bil: {maximum}")
        observed[result_key] = maximum
    return observed


def _verify_recomputed_cash_claims(manifest: dict[str, Any], recomputed: dict[str, Any]) -> None:
    boundaries = manifest.get("boundary_checks")
    if not isinstance(boundaries, dict):
        raise VerificationError("run manifest boundary_checks must be an object")
    for key, expected in recomputed.items():
        if key not in boundaries:
            raise VerificationError(f"run manifest boundary_checks is missing recomputed value: {key}")
        _compare_recomputed_value(f"boundary_checks {key}", boundaries[key], expected)

    validation = manifest.get("validation")
    invariants = validation.get("invariants") if isinstance(validation, dict) else None
    if not isinstance(invariants, list):
        raise VerificationError("run manifest validation.invariants must be an array")
    by_id = {
        str(item.get("id")): item
        for item in invariants
        if isinstance(item, dict) and item.get("id") is not None
    }
    for expected in cash_closure_validation_invariants(recomputed):
        invariant_id = str(expected["id"])
        actual = by_id.get(invariant_id)
        if actual is None:
            raise VerificationError(f"cash closure validation invariant is missing: {invariant_id}")
        if actual.get("status") != expected.get("status"):
            raise VerificationError(f"validation invariant {invariant_id} status disagrees with recomputed results")
        _compare_observation(invariant_id, actual.get("observed"), expected.get("observed"))


def _compare_observation(invariant_id: str, actual: Any, expected: Any) -> None:
    """Compare a ``key=value;...`` observation, numerically where the value is a number.

    The observed strings embed floats, so an exact string comparison rejects a faithful run
    over a last-digit repr difference: reading results back from CSV can return
    ``85.13373083354395`` where the producer wrote ``85.13373083354418``. That is
    representation noise, not a changed claim. Keys, ordering, and every non-numeric value
    still have to match exactly, so a tampered status or a substituted magnitude is caught.
    """

    actual_fields = _observation_fields(actual)
    expected_fields = _observation_fields(expected)
    if actual_fields is None or expected_fields is None or list(actual_fields) != list(expected_fields):
        if actual != expected:
            raise VerificationError(f"validation invariant {invariant_id} disagrees with recomputed results")
        return
    for key, expected_value in expected_fields.items():
        actual_value = actual_fields[key]
        if actual_value == expected_value:
            continue
        try:
            if math.isclose(float(actual_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9):
                continue
        except (TypeError, ValueError):
            pass
        raise VerificationError(
            f"validation invariant {invariant_id} disagrees with recomputed results on {key}: "
            f"manifest={actual_value!r}, actual={expected_value!r}"
        )


def _observation_fields(value: Any) -> dict[str, str] | None:
    if not isinstance(value, str) or "=" not in value:
        return None
    fields: dict[str, str] = {}
    for part in value.split(";"):
        if not part:
            continue
        key, sep, item = part.partition("=")
        if not sep:
            return None
        if key in fields:
            return None
        fields[key] = item
    return fields or None


def _compare_recomputed_value(label: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, bool):
        matches = actual is expected
    elif isinstance(expected, int):
        matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    elif isinstance(expected, float):
        try:
            matches = math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9)
        except (TypeError, ValueError):
            matches = False
    else:
        matches = actual == expected
    if not matches:
        raise VerificationError(f"{label} disagrees with recomputed results: manifest={actual!r}, actual={expected!r}")


def _result_artifact_path(root: Path, manifest: dict[str, Any]) -> Path:
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("logical_name", "")).startswith("results_"):
            return root / str(item["relative_path"])
    raise VerificationError("run manifest does not list a results output")


def _output_artifact_path(root: Path, manifest: dict[str, Any], logical_name_stem: str) -> Path:
    for item in manifest.get("outputs", []):
        if isinstance(item, dict) and str(item.get("logical_name", "")).startswith(f"{logical_name_stem}.csv"):
            return root / str(item["relative_path"])
    raise VerificationError(f"run manifest does not list required output: {logical_name_stem}")


def _verify_bounded_outputs(
    root: Path,
    manifest: dict[str, Any],
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Verify the compact v2 evidence lane without reconstructing event history."""

    records = _verify_bounded_manifest_contract(root, manifest)
    periods, commitment_counts = _verify_bounded_commitments(
        root, records, manifest
    )
    result_evidence = _bounded_period_result_evidence(results, periods)
    ledger = _verify_bounded_ledger(root, records, periods)
    if ledger["event_count"] != commitment_counts["event_count"]:
        raise VerificationError(
            "bounded ledger event count does not match event commitments"
        )
    if ledger["event_type_counts"] != commitment_counts["event_type_counts"]:
        raise VerificationError(
            "bounded ledger event-type counts do not match event commitments"
        )

    accounting = _verify_bounded_accounting(
        root, records, periods, ledger["period_sums"], results
    )
    stock = _verify_bounded_stock_closure(root, records, periods)
    _cross_check_bounded_accounting_stock(accounting, stock, results)
    _verify_bounded_route_closure(root, records, periods)

    components = _verify_bounded_tdc_components(
        root, records, periods, result_evidence
    )
    summaries = _verify_bounded_tdc_summary(
        root, records, periods, components, result_evidence
    )
    principal = _verify_bounded_principal(
        root, records, periods, result_evidence
    )
    _cross_check_bounded_principal_summary(principal, summaries)
    issuance = _verify_bounded_issuance(
        root, records, periods, result_evidence
    )
    payment = _verify_bounded_payment(
        root, records, periods, result_evidence
    )
    financing = _cross_check_bounded_flow_bridges(
        periods,
        ledger=ledger["period_sums"],
        issuance=issuance["period"],
        principal=principal,
        payment=payment,
        results=result_evidence,
    )
    metrics = _verify_bounded_scenario_metrics(
        root, records, periods, result_evidence
    )
    funding_closure_mode = _manifest_funding_closure_mode(manifest)
    expected_bridge_funding_modes = (
        {funding_closure_mode}
        if funding_closure_mode
        == "cbo_debt_reference_plus_tga_floor_financing_v1"
        else {"cbo_public_debt_target", "cbo_target"}
    )
    _verify_bounded_debt_bridge(
        root,
        records,
        periods,
        issuance=issuance["period"],
        principal=principal,
        ledger=ledger["period_sums"],
        results=result_evidence,
        expected_funding_modes=expected_bridge_funding_modes,
    )
    _verify_bounded_annual(
        root,
        records,
        periods,
        summaries=summaries,
        issuance=issuance["annual"],
        metrics=metrics,
        financing=financing,
    )
    resource = _verify_bounded_resources(
        root,
        records,
        manifest,
        periods,
        commitment_counts["period_counts"],
    )
    return {
        "bounded_period_count": len(periods),
        "bounded_event_count": int(commitment_counts["event_count"]),
        "bounded_peak_rss_bytes": resource["peak_rss_bytes"],
    }


def _verify_bounded_manifest_contract(
    root: Path, manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    bounded = manifest.get("bounded_evidence")
    if not isinstance(bounded, dict):
        raise VerificationError("run manifest bounded_evidence must be an object")
    if bounded.get("event_schema_version") != EVENT_SCHEMA_VERSION:
        raise VerificationError("bounded event schema version is unsupported")
    if bounded.get("invariant_status") != "pass":
        raise VerificationError("bounded invariant status must be pass")
    if manifest.get("evidence_profile") != BOUNDED_EVIDENCE_PROFILE:
        raise VerificationError("bounded evidence profile is unsupported")
    clock = manifest.get("aggregation_clock")
    if not isinstance(clock, dict) or clock.get("clock_id") != AGGREGATION_CLOCK_ID:
        raise VerificationError("bounded aggregation clock is unsupported")
    milestones = tuple(manifest.get("execution_milestones") or ())
    if milestones not in {
        _BOUNDED_EXECUTION_MILESTONES,
        _BOUNDED_WATCHDOG_ACCEPTED_MILESTONES,
    }:
        raise VerificationError("bounded execution milestones are incomplete or out of order")
    open04_active = (
        isinstance(manifest.get("evaluated_nominal_curve"), Mapping)
        or isinstance(manifest.get("open04_campaign"), Mapping)
    )
    if (
        open04_active
        and milestones != _BOUNDED_WATCHDOG_ACCEPTED_MILESTONES
    ):
        raise VerificationError(
            "OPEN-04 run requires parent watchdog acceptance"
        )
    watchdog_accepted = milestones == _BOUNDED_WATCHDOG_ACCEPTED_MILESTONES
    _verify_execution_contract(
        manifest,
        parent_watchdog_accepted=watchdog_accepted,
        open04_active=open04_active,
    )
    if open04_active:
        _verify_open04_memory_thresholds(bounded)
    _verify_parent_watchdog_contract(
        manifest,
        bounded,
        accepted=watchdog_accepted,
    )

    deterministic = bounded.get("deterministic_artifacts")
    if not isinstance(deterministic, dict):
        raise VerificationError("bounded deterministic_artifacts must be an object")
    if set(deterministic) != _BOUNDED_DETERMINISTIC_ARTIFACTS:
        raise VerificationError(
            "bounded deterministic artifact set does not match the compact contract"
        )
    resource = bounded.get("resource_artifact")
    if not isinstance(resource, dict):
        raise VerificationError("bounded resource_artifact must be an object")
    records = {
        name: dict(record)
        for name, record in deterministic.items()
        if isinstance(record, dict)
    }
    if len(records) != len(deterministic):
        raise VerificationError("bounded artifact record must be an object")
    records["resources"] = dict(resource)

    output_manifest = manifest.get("output_manifest")
    if not isinstance(output_manifest, dict):
        raise VerificationError("bounded output_manifest must be an object")
    if output_manifest.get("evidence_profile") != BOUNDED_EVIDENCE_PROFILE:
        raise VerificationError("bounded output manifest evidence profile disagrees")
    if output_manifest.get("verification_grade") != BOUNDED_VERIFICATION_GRADE:
        raise VerificationError("bounded output manifest verification grade disagrees")
    if output_manifest.get("deterministic_evidence_artifacts") != deterministic:
        raise VerificationError(
            "bounded output manifest deterministic artifacts disagree with producer manifest"
        )

    declared_outputs = {
        str(item.get("relative_path")): item
        for item in manifest.get("outputs", [])
        if isinstance(item, dict)
    }
    for name, record in records.items():
        expected_filename, _columns = _BOUNDED_ARTIFACT_SPECS[name]
        rel = Path(str(record.get("path") or ""))
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or rel.as_posix() != expected_filename
        ):
            raise VerificationError(
                f"bounded artifact path is not the declared compact path: {name}"
            )
        path = root / "outputs" / rel
        if not path.is_file():
            raise VerificationError(f"bounded artifact is missing: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise VerificationError(f"bounded artifact SHA mismatch: {name}")
        if path.stat().st_size != _bounded_int(
            record, "bytes", label=f"bounded artifact {name}", minimum=0
        ):
            raise VerificationError(f"bounded artifact byte count mismatch: {name}")
        _bounded_int(
            record, "row_count", label=f"bounded artifact {name}", minimum=0
        )

        output_record = output_manifest.get(name)
        if not isinstance(output_record, dict):
            raise VerificationError(
                f"bounded output manifest is missing artifact record: {name}"
            )
        for key in ("path", "sha256", "bytes", "row_count"):
            if output_record.get(key) != record.get(key):
                raise VerificationError(
                    f"bounded output manifest artifact disagrees on {name}.{key}"
                )
        listed = declared_outputs.get(f"outputs/{rel.as_posix()}")
        if not isinstance(listed, dict):
            raise VerificationError(
                f"bounded artifact is absent from the public output list: {name}"
            )
        if (
            listed.get("sha256") != record.get("sha256")
            or listed.get("bytes") != record.get("bytes")
        ):
            raise VerificationError(
                f"bounded public output record disagrees: {name}"
            )

    limits = bounded.get("memory_thresholds")
    if not isinstance(limits, dict):
        raise VerificationError("bounded memory_thresholds must be an object")
    acceptance = _bounded_int(
        limits, "acceptance_peak_rss_bytes", label="memory thresholds", minimum=0
    )
    application = _bounded_int(
        limits, "application_abort_rss_bytes", label="memory thresholds", minimum=0
    )
    graceful = _bounded_int(
        limits, "parent_graceful_stop_rss_bytes", label="memory thresholds", minimum=0
    )
    kill = _bounded_int(
        limits, "parent_kill_rss_bytes", label="memory thresholds", minimum=0
    )
    if not acceptance <= application <= graceful <= kill:
        raise VerificationError("bounded RSS thresholds are not monotonic")
    peak = _bounded_int(
        bounded, "peak_rss_bytes", label="bounded evidence", minimum=1
    )
    if acceptance and peak > acceptance:
        raise VerificationError(
            f"bounded acceptance RSS exceeded: observed={peak}, limit={acceptance}"
        )
    max_portfolio = _bounded_int(
        bounded, "max_portfolio_rows", label="bounded evidence", minimum=0
    )
    max_active = _bounded_int(
        bounded, "max_active_portfolio_rows", label="bounded evidence", minimum=0
    )
    portfolio_budget = _bounded_int(
        bounded, "portfolio_row_budget", label="bounded evidence", minimum=1
    )
    if max_portfolio > portfolio_budget or max_active > max_portfolio:
        raise VerificationError("bounded portfolio-row evidence exceeds its declared cap")
    max_keys = _bounded_int(
        bounded, "max_key_cardinality", label="bounded evidence", minimum=0
    )
    key_budget = _bounded_int(
        bounded, "key_cardinality_budget", label="bounded evidence", minimum=1
    )
    if max_keys > key_budget:
        raise VerificationError("bounded key-cardinality evidence exceeds its declared cap")

    summary = read_json(root / "outputs" / "summary.json")
    expected_summary = {
        "evidence_profile": BOUNDED_EVIDENCE_PROFILE,
        "verification_grade": BOUNDED_VERIFICATION_GRADE,
        "event_count": _bounded_int(
            bounded, "event_count", label="bounded evidence", minimum=1
        ),
        "event_root_sha256": _bounded_sha256(
            bounded.get("event_root_sha256"), label="bounded event root"
        ),
        "final_state_sha256": _bounded_sha256(
            bounded.get("final_state_sha256"), label="bounded final state"
        ),
        "peak_rss_bytes": peak,
        "portfolio_row_budget": portfolio_budget,
        "max_portfolio_rows": max_portfolio,
        "max_key_cardinality": max_keys,
    }
    if not isinstance(summary, dict):
        raise VerificationError("bounded summary.json must be an object")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise VerificationError(f"bounded summary disagrees with manifest: {key}")
    _verify_bounded_validation_bindings(manifest, bounded)
    return records


def _verify_bounded_validation_bindings(
    manifest: Mapping[str, Any], bounded: Mapping[str, Any]
) -> None:
    validation = manifest.get("validation")
    invariants = validation.get("invariants") if isinstance(validation, dict) else None
    by_id = {
        str(item.get("id")): item
        for item in invariants or []
        if isinstance(item, dict)
    }
    expected = {
        "bounded_period_closure": {
            "observed": f"periods={_bounded_int(bounded, 'period_count', label='bounded evidence', minimum=1)}"
        },
        "portfolio_row_budget": {
            "observed": _bounded_int(
                bounded, "max_portfolio_rows", label="bounded evidence", minimum=0
            ),
            "limit": _bounded_int(
                bounded, "portfolio_row_budget", label="bounded evidence", minimum=1
            ),
        },
        "key_cardinality_budget": {
            "observed": _bounded_int(
                bounded, "max_key_cardinality", label="bounded evidence", minimum=0
            ),
            "limit": _bounded_int(
                bounded, "key_cardinality_budget", label="bounded evidence", minimum=1
            ),
        },
    }
    for invariant_id, fields in expected.items():
        item = by_id.get(invariant_id)
        if not isinstance(item, dict) or item.get("status") != "pass":
            raise VerificationError(
                f"bounded validation invariant is missing or failed: {invariant_id}"
            )
        for key, value in fields.items():
            if item.get(key) != value:
                raise VerificationError(
                    f"bounded validation invariant is stale: {invariant_id}.{key}"
                )


def _verify_open04_memory_thresholds(bounded: Mapping[str, Any]) -> None:
    thresholds = bounded.get("memory_thresholds")
    if not isinstance(thresholds, Mapping):
        raise VerificationError(
            "OPEN-04 run is missing memory thresholds"
        )
    expected = BoundedResourceLimits()
    for field in (
        "minimum_available_bytes",
        "acceptance_peak_rss_bytes",
        "application_abort_rss_bytes",
        "parent_graceful_stop_rss_bytes",
        "parent_kill_rss_bytes",
    ):
        observed = _bounded_int(
            thresholds,
            field,
            label="OPEN-04 memory thresholds",
            minimum=1,
        )
        required = int(getattr(expected, field))
        if observed != required:
            raise VerificationError(
                "OPEN-04 run requires the exact "
                f"4/6/8/10/12 GiB memory envelope: {field}="
                f"{observed}, required={required}"
            )


def _verify_execution_contract(
    manifest: Mapping[str, Any],
    *,
    parent_watchdog_accepted: bool,
    open04_active: bool,
) -> None:
    contract = manifest.get("execution_contract")
    if not isinstance(contract, Mapping):
        raise VerificationError(
            "bounded execution_contract must be an object"
        )
    required_fields = {
        "schema_version",
        "single_writer_claim",
        "writer_claim_file_name",
        "writer_claim_scope",
        "writer_claim_scope_id",
        "one_scenario_per_worker",
        "process_pool_enabled",
        "parent_watchdog_required",
        "scenario_process_mode",
        "numerical_thread_environment",
    }
    if set(contract) != required_fields:
        raise VerificationError(
            "bounded execution_contract fields do not match the producer contract"
        )
    exact_values = {
        "schema_version": "tdcsim_cbo_execution_contract_v1",
        "writer_claim_file_name": ".tdcsim-cbo-bounded-writer.claim",
        "writer_claim_scope": "output_parent",
        "scenario_process_mode": (
            "parent_watchdog_worker"
            if parent_watchdog_accepted
            else "library_call"
        ),
    }
    for field, expected in exact_values.items():
        if contract.get(field) != expected:
            raise VerificationError(
                f"bounded execution_contract disagrees on {field}"
            )
    scope_id = contract.get("writer_claim_scope_id")
    marker = manifest.get("open04_campaign")
    expected_scope_id = "library-output-parent"
    if isinstance(marker, Mapping):
        expected_scope_id = f"{marker.get('contract_id')}.{marker.get('role')}"
    if (
        not isinstance(scope_id, str)
        or len(scope_id) < 3
        or len(scope_id) > 160
        or not scope_id[0].isalnum()
        or any(
            char not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in scope_id
        )
        or scope_id != expected_scope_id
    ):
        raise VerificationError(
            "bounded execution_contract writer_claim_scope_id is invalid"
        )
    for field, expected in (
        ("single_writer_claim", True),
        ("one_scenario_per_worker", True),
        ("process_pool_enabled", False),
        ("parent_watchdog_required", parent_watchdog_accepted),
    ):
        if contract.get(field) is not expected:
            raise VerificationError(
                f"bounded execution_contract disagrees on {field}"
            )
    thread_environment = contract.get("numerical_thread_environment")
    if not isinstance(thread_environment, Mapping):
        raise VerificationError(
            "bounded execution_contract numerical thread environment must be an object"
        )
    expected_thread_names = set(THREAD_LIMIT_ENVIRONMENT_VARIABLES)
    if set(thread_environment) != expected_thread_names:
        raise VerificationError(
            "bounded execution_contract numerical thread variables are incomplete"
        )
    if parent_watchdog_accepted:
        mismatches = {
            name: thread_environment.get(name)
            for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
            if thread_environment.get(name) != "1"
        }
        if mismatches:
            raise VerificationError(
                "bounded watchdog execution requires every numerical thread "
                f"limit pinned to one: mismatches={mismatches}"
            )
    elif not all(
        isinstance(thread_environment.get(name), str)
        for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    ):
        raise VerificationError(
            "bounded execution_contract numerical thread values must be strings"
        )


def _verify_parent_watchdog_contract(
    manifest: Mapping[str, Any],
    bounded: Mapping[str, Any],
    *,
    accepted: bool,
) -> None:
    if not accepted:
        if "parent_watchdog" in manifest:
            raise VerificationError(
                "bounded parent_watchdog requires its accepted terminal milestone"
            )
        return
    parent = manifest.get("parent_watchdog")
    if not isinstance(parent, Mapping):
        raise VerificationError(
            "bounded watchdog-accepted run is missing parent_watchdog"
        )
    if (
        parent.get("status") != "accepted"
        or parent.get("sampler") != "parent_process_rss_poll_v1"
    ):
        raise VerificationError("bounded parent_watchdog identity is unsupported")
    _bounded_int(
        parent,
        "child_pid",
        label="bounded parent watchdog",
        minimum=1,
    )
    child_returncode = _bounded_int(
        parent,
        "child_returncode",
        label="bounded parent watchdog",
        minimum=0,
    )
    if child_returncode != 0:
        raise VerificationError(
            "bounded parent_watchdog child_returncode is not successful"
        )
    if parent.get("action") != "completed":
        raise VerificationError(
            "bounded parent_watchdog action is not completed"
        )
    thresholds = bounded.get("memory_thresholds")
    if not isinstance(thresholds, Mapping):
        raise VerificationError(
            "bounded parent watchdog memory thresholds are missing"
        )
    acceptance = _bounded_int(
        thresholds,
        "acceptance_peak_rss_bytes",
        label="bounded parent watchdog thresholds",
        minimum=1,
    )
    graceful = _bounded_int(
        thresholds,
        "parent_graceful_stop_rss_bytes",
        label="bounded parent watchdog thresholds",
        minimum=1,
    )
    kill = _bounded_int(
        thresholds,
        "parent_kill_rss_bytes",
        label="bounded parent watchdog thresholds",
        minimum=1,
    )
    for field, expected in (
        ("acceptance_peak_rss_bytes", acceptance),
        ("terminate_rss_bytes", graceful),
        ("kill_rss_bytes", kill),
    ):
        if _bounded_int(
            parent,
            field,
            label="bounded parent watchdog",
            minimum=1,
        ) != expected:
            raise VerificationError(
                f"bounded parent_watchdog disagrees with {field}"
            )
    parent_peak = _bounded_int(
        parent,
        "peak_rss_bytes",
        label="bounded parent watchdog",
        minimum=0,
    )
    worker_peak = _bounded_int(
        parent,
        "worker_peak_rss_bytes",
        label="bounded parent watchdog",
        minimum=0,
    )
    bounded_peak = _bounded_int(
        bounded,
        "peak_rss_bytes",
        label="bounded evidence",
        minimum=0,
    )
    if worker_peak != bounded_peak:
        raise VerificationError(
            "bounded parent_watchdog worker peak disagrees with bounded evidence"
        )
    effective_peak = _bounded_int(
        parent,
        "effective_peak_rss_bytes",
        label="bounded parent watchdog",
        minimum=0,
    )
    if effective_peak != max(parent_peak, worker_peak):
        raise VerificationError(
            "bounded parent_watchdog effective peak is stale"
        )
    if effective_peak > acceptance:
        raise VerificationError(
            "bounded parent_watchdog effective peak exceeds the acceptance ceiling"
        )
    poll_interval = _bounded_float(
        parent,
        "poll_interval_seconds",
        label="bounded parent watchdog",
    )
    if poll_interval != 1.0:
        raise VerificationError(
            "bounded parent_watchdog poll interval is unsupported"
        )
    validation = manifest.get("validation")
    invariants = (
        validation.get("invariants")
        if isinstance(validation, Mapping)
        else None
    )
    matches = [
        item
        for item in invariants or []
        if isinstance(item, Mapping)
        and item.get("id") == "parent_watchdog_peak_rss"
    ]
    if len(matches) != 1 or matches[0].get("status") != "pass":
        raise VerificationError(
            "bounded parent watchdog validation invariant is missing or duplicated"
        )
    invariant = matches[0]
    if (
        _bounded_int(
            invariant,
            "observed",
            label="bounded parent watchdog invariant",
            minimum=0,
        )
        != effective_peak
        or _bounded_int(
            invariant,
            "limit",
            label="bounded parent watchdog invariant",
            minimum=1,
        )
        != acceptance
    ):
        raise VerificationError(
            "bounded parent watchdog validation invariant is stale"
        )


def _bounded_rows(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    name: str,
) -> Iterator[dict[str, str]]:
    record = records[name]
    filename, columns = _BOUNDED_ARTIFACT_SPECS[name]
    path = root / "outputs" / filename
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise VerificationError(
                f"bounded artifact has unexpected columns or order: {name}"
            )
        for row in reader:
            count += 1
            if None in row:
                raise VerificationError(
                    f"bounded artifact has an over-wide row: {name}"
                )
            yield row
    expected = _bounded_int(
        record, "row_count", label=f"bounded artifact {name}", minimum=0
    )
    if count != expected:
        raise VerificationError(
            f"bounded artifact row count mismatch: {name}; "
            f"manifest={expected}, actual={count}"
        )


def _verify_bounded_commitments(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    periods: list[tuple[str, str]] = []
    period_counts: dict[tuple[str, str], int] = {}
    period_types: dict[tuple[str, str], Counter[str]] = {}
    next_sequence = 1
    final_root = ""
    empty_root = hashlib.sha256(b"").hexdigest()
    for row in _bounded_rows(root, records, "commitments"):
        period = _bounded_period(row, label="event commitments")
        if periods and period[0] != periods[-1][1]:
            raise VerificationError("bounded event commitment periods are not continuous")
        if period in period_counts:
            raise VerificationError("bounded event commitments contain a duplicate period")
        count = _bounded_int(
            row, "event_count", label="event commitments", minimum=0
        )
        start = _bounded_int(
            row, "event_seq_start", label="event commitments", minimum=0
        )
        end = _bounded_int(
            row, "event_seq_end", label="event commitments", minimum=0
        )
        if count:
            if start != next_sequence or end != start + count - 1:
                raise VerificationError(
                    "bounded event commitment sequence is discontinuous"
                )
            next_sequence = end + 1
        elif start != 0 or end != 0:
            raise VerificationError(
                "zero-event commitment must use zero sequence endpoints"
            )
        period_root = _bounded_sha256(
            row.get("period_event_root_sha256"), label="period event root"
        )
        if count == 0 and period_root != empty_root:
            raise VerificationError("zero-event commitment has a nonempty period root")
        final_root = _bounded_sha256(
            row.get("whole_run_root_through_period_sha256"),
            label="whole-run event root",
        )
        raw_counts = str(row.get("event_type_counts_json") or "")
        try:
            decoded = json.loads(raw_counts)
        except json.JSONDecodeError as exc:
            raise VerificationError(
                "event commitment type counts are malformed JSON"
            ) from exc
        if not isinstance(decoded, dict) or json.dumps(
            decoded, sort_keys=True, separators=(",", ":")
        ) != raw_counts:
            raise VerificationError(
                "event commitment type counts are not canonical JSON"
            )
        counts: Counter[str] = Counter()
        for event_type, value in decoded.items():
            if event_type not in _ALLOWED_EVENT_TYPES:
                raise VerificationError(
                    f"event commitment contains unknown event type: {event_type}"
                )
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 1
            ):
                raise VerificationError(
                    "event commitment type counts must be positive integers"
                )
            counts[event_type] = value
        if sum(counts.values()) != count:
            raise VerificationError(
                "event commitment type counts do not sum to event_count"
            )
        periods.append(period)
        period_counts[period] = count
        period_types[period] = counts
    if not periods:
        raise VerificationError("bounded event commitments must contain period rows")

    bounded = manifest["bounded_evidence"]
    declared_periods = _bounded_int(
        bounded, "period_count", label="bounded evidence", minimum=1
    )
    declared_events = _bounded_int(
        bounded, "event_count", label="bounded evidence", minimum=1
    )
    if len(periods) != declared_periods:
        raise VerificationError("bounded commitment period count disagrees with manifest")
    if sum(period_counts.values()) != declared_events:
        raise VerificationError("bounded commitment event count disagrees with manifest")
    if final_root != bounded.get("event_root_sha256"):
        raise VerificationError("bounded final event root disagrees with manifest")
    simulation = manifest.get("simulation")
    if not isinstance(simulation, dict):
        raise VerificationError("bounded simulation contract must be an object")
    if (
        periods[0][0] != simulation.get("start_date")
        or periods[-1][1] != simulation.get("end_date")
    ):
        raise VerificationError(
            "bounded commitment coverage disagrees with simulation dates"
        )
    return periods, {
        "event_count": declared_events,
        "period_counts": period_counts,
        "event_type_counts": period_types,
    }


def _verify_bounded_resources(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    periods: list[tuple[str, str]],
    period_event_counts: Mapping[tuple[str, str], int],
) -> dict[str, int]:
    bounded = manifest["bounded_evidence"]
    limits = bounded["memory_thresholds"]
    rows = 0
    peak = 0
    max_keys = 0
    max_portfolio = 0
    prior_bytes = -1
    prior_cpu = -1.0
    sample_period_ends: list[str] = []
    observed_event_counts: list[int] = []
    for row in _bounded_rows(root, records, "resources"):
        rows += 1
        kind = str(row.get("sample_kind") or "")
        if rows == 1:
            if kind != "admission":
                raise VerificationError("first bounded resource sample must be admission")
        elif kind != "period_close":
            raise VerificationError(
                "post-admission bounded resource samples must be period_close"
            )
        rss = _bounded_int(row, "rss_bytes", label="resource samples", minimum=1)
        row_peak = _bounded_int(
            row, "peak_rss_bytes", label="resource samples", minimum=1
        )
        if row_peak < rss or row_peak < peak:
            raise VerificationError("bounded resource peak RSS is not monotonic")
        peak = row_peak
        available = _bounded_int(
            row, "host_available_bytes", label="resource samples", minimum=0
        )
        if rows == 1 and available < _bounded_int(
            limits, "minimum_available_bytes", label="memory thresholds", minimum=0
        ):
            raise VerificationError("bounded admission sample fails available-memory gate")
        cpu_seconds = _bounded_float(
            row, "process_cpu_seconds", label="resource samples"
        )
        if cpu_seconds < 0.0 or cpu_seconds < prior_cpu:
            raise VerificationError(
                "bounded resource process CPU time is not monotonic"
            )
        prior_cpu = cpu_seconds
        portfolio = _bounded_int(
            row, "portfolio_rows", label="resource samples", minimum=0
        )
        active = _bounded_int(
            row, "active_portfolio_rows", label="resource samples", minimum=0
        )
        if active > portfolio:
            raise VerificationError(
                "bounded active portfolio rows exceed total portfolio rows"
            )
        max_portfolio = max(max_portfolio, portfolio)
        current_keys = _bounded_int(
            row, "current_key_cardinality", label="resource samples", minimum=0
        )
        row_max_keys = _bounded_int(
            row, "max_key_cardinality", label="resource samples", minimum=0
        )
        if current_keys > row_max_keys or row_max_keys < max_keys:
            raise VerificationError(
                "bounded resource key-cardinality maximum is inconsistent"
            )
        max_keys = row_max_keys
        observed_event_counts.append(
            _bounded_int(
                row, "event_count", label="resource samples", minimum=0
            )
        )
        bytes_written = _bounded_int(
            row, "bytes_written", label="resource samples", minimum=0
        )
        if bytes_written < prior_bytes:
            raise VerificationError("bounded resource bytes_written is not monotonic")
        prior_bytes = bytes_written
        sample_period_ends.append(
            _bounded_date(row.get("period_end"), label="resource samples").isoformat()
        )
    expected_ends = [periods[0][0], *[end for _start, end in periods]]
    if sample_period_ends != expected_ends:
        raise VerificationError(
            "bounded resource samples do not cover admission and every period close"
        )
    if rows != len(periods) + 1:
        raise VerificationError("bounded resource sample count is incomplete")
    cumulative_events = 0
    expected_event_counts = [0]
    for period in periods:
        cumulative_events += int(period_event_counts[period])
        expected_event_counts.append(cumulative_events)
    if observed_event_counts != expected_event_counts:
        raise VerificationError(
            "bounded resource event counts disagree with committed progress"
        )
    if peak != bounded.get("peak_rss_bytes"):
        raise VerificationError("bounded resource peak RSS disagrees with manifest")
    if max_keys != bounded.get("max_key_cardinality"):
        raise VerificationError(
            "bounded resource key-cardinality maximum disagrees with manifest"
        )
    if max_portfolio > bounded.get("max_portfolio_rows"):
        raise VerificationError(
            "bounded resource portfolio maximum exceeds manifest maximum"
        )
    return {"peak_rss_bytes": peak}


def _verify_bounded_ledger(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
) -> dict[str, Any]:
    allowed_periods = set(periods)
    period_sums: dict[tuple[str, str], dict[str, float]] = {
        period: {
            "face_stock_change_bil": 0.0,
            "adjusted_principal_change_bil": 0.0,
            "treasury_cash_change_bil": 0.0,
            "reserve_change_bil": 0.0,
            "deposit_change_bil": 0.0,
            "nonmarketable_interest_capitalized_bil": 0.0,
            "tips_inflation_accretion_bil": 0.0,
        }
        for period in periods
    }
    period_counts = {period: 0 for period in periods}
    period_types = {period: Counter() for period in periods}
    seen: set[tuple[str, ...]] = set()
    for row in _bounded_rows(root, records, "ledger"):
        period = _bounded_period(row, label="bounded ledger")
        if period not in allowed_periods:
            raise VerificationError("bounded ledger contains an undeclared period")
        key = tuple(str(row.get(column) or "") for column in LEDGER_COLUMNS[:15])
        if key in seen:
            raise VerificationError("bounded ledger contains a duplicate reducer key")
        seen.add(key)
        event_type = str(row.get("event_type") or "")
        if event_type not in _ALLOWED_EVENT_TYPES:
            raise VerificationError(
                f"bounded ledger contains unknown event_type: {event_type}"
            )
        for column in ("leg_type", "accounting_basis"):
            if not str(row.get(column) or "").strip():
                raise VerificationError(f"bounded ledger has blank {column}")
        for column, allowed in (
            ("holder_sector", _ALLOWED_HOLDERS),
            ("route_holder_sector", _ALLOWED_HOLDERS),
            ("instrument_type", _ALLOWED_INSTRUMENTS),
            ("maturity_bucket", _ALLOWED_MATURITY_BUCKETS),
        ):
            if str(row.get(column) or "") not in allowed:
                raise VerificationError(
                    f"bounded ledger contains unknown taxonomy: {column}"
                )
        _bounded_bool(row.get("is_intragovernmental"), label="bounded ledger")
        numeric_values: dict[str, float] = {}
        for column in (
            "face_stock_change_bil",
            "adjusted_principal_change_bil",
            "route_face_stock_change_bil",
            "route_adjusted_principal_change_bil",
            "treasury_cash_change_bil",
            "reserve_change_bil",
            "deposit_change_bil",
        ):
            value = _bounded_float(row, column, label="bounded ledger")
            numeric_values[column] = value
            if column in period_sums[period]:
                period_sums[period][column] += value
        if str(row.get("leg_type") or "") == (
            "nonmarketable_interest_capitalization"
        ):
            period_sums[period][
                "nonmarketable_interest_capitalized_bil"
            ] += numeric_values["face_stock_change_bil"]
        if event_type == "indexation":
            period_sums[period][
                "tips_inflation_accretion_bil"
            ] += numeric_values["adjusted_principal_change_bil"]
        count = _bounded_int(row, "event_count", label="bounded ledger", minimum=1)
        period_counts[period] += count
        period_types[period][event_type] += count
    return {
        "event_count": sum(period_counts.values()),
        "event_type_counts": period_types,
        "period_sums": period_sums,
    }


def _verify_bounded_accounting(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    ledger_sums: Mapping[tuple[str, str], Mapping[str, float]],
    results: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, float]]:
    result_map = _bounded_result_snapshots(results)
    result_dates = list(result_map)
    if list(zip(result_dates, result_dates[1:])) != periods:
        raise VerificationError(
            "bounded results snapshots do not match committed periods"
        )
    expected_periods = set(periods)
    rows: dict[tuple[str, str], dict[str, float]] = {}
    numeric_columns = [
        column
        for column in HANDOFF_TABLE_COLUMNS["tdcsim_accounting_closure"]
        if column not in {"period_start", "period_end", "closure_basis"}
    ]
    for row in _bounded_rows(root, records, "accounting"):
        period = _bounded_period(row, label="bounded accounting closure")
        if period not in expected_periods or period in rows:
            raise VerificationError(
                "bounded accounting closure period set is not exact and unique"
            )
        if row.get("closure_basis") != (
            "independent_opening_and_closing_state_snapshots"
        ):
            raise VerificationError(
                "bounded accounting closure has unexpected closure_basis"
            )
        values = {
            column: _bounded_float(
                row, column, label="bounded accounting closure"
            )
            for column in numeric_columns
        }
        journal = ledger_sums[period]
        for ledger_column, closure_column in (
            ("face_stock_change_bil", "journal_face_stock_change_bil"),
            (
                "adjusted_principal_change_bil",
                "journal_adjusted_principal_change_bil",
            ),
            ("treasury_cash_change_bil", "journal_treasury_cash_change_bil"),
            ("reserve_change_bil", "journal_reserve_change_bil"),
            ("deposit_change_bil", "journal_deposit_change_bil"),
        ):
            _bounded_close(
                values[closure_column],
                float(journal[ledger_column]),
                label=f"bounded accounting journal bridge {closure_column}",
            )
        recomputed_errors = {
            "face_stock_closure_error_bil": (
                values["closing_face_stock_bil"]
                - values["opening_face_stock_bil"]
                - values["journal_face_stock_change_bil"]
            ),
            "adjusted_principal_closure_error_bil": (
                values["closing_adjusted_principal_stock_bil"]
                - values["opening_adjusted_principal_stock_bil"]
                - values["journal_adjusted_principal_change_bil"]
            ),
            "treasury_cash_closure_error_bil": (
                values["closing_treasury_cash_bil"]
                - values["opening_treasury_cash_bil"]
                - values["journal_treasury_cash_change_bil"]
            ),
            "reserve_closure_error_bil": (
                values["reported_reserve_change_bil"]
                - values["journal_reserve_change_bil"]
            ),
            "deposit_closure_error_bil": (
                values["reported_deposit_change_bil"]
                - values["journal_deposit_change_bil"]
            ),
            "holder_total_error_bil": (
                values["holder_debt_total_bil"] - values["aggregate_debt_bil"]
            ),
            "instrument_total_error_bil": (
                values["instrument_debt_total_bil"]
                - values["aggregate_debt_bil"]
            ),
        }
        for column, recomputed in recomputed_errors.items():
            tolerance = (
                CASH_TOLERANCE_BIL
                if column == "treasury_cash_closure_error_bil"
                else STOCK_TOLERANCE_BIL
            )
            _bounded_close(
                values[column],
                recomputed,
                label=f"bounded accounting stale error {column}",
                tolerance=tolerance,
            )
            if abs(recomputed) > tolerance:
                raise VerificationError(
                    f"bounded accounting identity failed: {column}"
                )
        if abs(values["unexplained_residual_bil"]) > STOCK_TOLERANCE_BIL:
            raise VerificationError(
                "bounded accounting closure contains an unexplained residual"
            )
        opening_result = result_map[period[0]]
        closing_result = result_map[period[1]]
        result_checks = {
            "opening_treasury_cash_bil": opening_result["TGA"],
            "closing_treasury_cash_bil": closing_result["TGA"],
            "reported_reserve_change_bil": (
                closing_result["Reserves"] - opening_result["Reserves"]
            ),
            "reported_deposit_change_bil": (
                closing_result["TDC_Level"] - opening_result["TDC_Level"]
            ),
            "aggregate_debt_bil": closing_result["TotalDebt_Agg"],
        }
        for column, expected in result_checks.items():
            _bounded_close(
                values[column],
                expected,
                label=f"bounded accounting result snapshot {column}",
            )
        rows[period] = values
    if set(rows) != expected_periods:
        raise VerificationError(
            "bounded accounting closure does not cover every committed period"
        )
    for previous, current in zip(periods, periods[1:]):
        for opening, closing in (
            ("opening_face_stock_bil", "closing_face_stock_bil"),
            (
                "opening_adjusted_principal_stock_bil",
                "closing_adjusted_principal_stock_bil",
            ),
            ("opening_treasury_cash_bil", "closing_treasury_cash_bil"),
        ):
            _bounded_close(
                rows[current][opening],
                rows[previous][closing],
                label=f"bounded accounting continuity {opening}",
            )
    return rows


def _verify_bounded_stock_closure(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
    totals = {
        period: {
            "holder": {
                "opening_face": 0.0,
                "closing_face": 0.0,
                "opening_adjusted": 0.0,
                "closing_adjusted": 0.0,
                "closing_debt": 0.0,
            },
            "instrument": {
                "opening_face": 0.0,
                "closing_face": 0.0,
                "opening_adjusted": 0.0,
                "closing_adjusted": 0.0,
                "closing_debt": 0.0,
            },
        }
        for period in periods
    }
    row_counts = Counter()
    period_index = 0
    current_period = periods[0]
    previous_closing: dict[
        tuple[str, str, str, str, str, str], tuple[float, float, float]
    ] = {}
    current_opening: dict[
        tuple[str, str, str, str, str, str], tuple[float, float, float]
    ] = {}
    current_closing: dict[
        tuple[str, str, str, str, str, str], tuple[float, float, float]
    ] = {}

    def finish_period() -> None:
        nonlocal previous_closing, current_opening, current_closing
        if row_counts[current_period] == 0:
            raise VerificationError(
                "bounded stock closure does not cover every committed period"
            )
        if period_index:
            for continuity_key in set(previous_closing) | set(current_opening):
                prior = previous_closing.get(
                    continuity_key, (0.0, 0.0, 0.0)
                )
                opening = current_opening.get(
                    continuity_key, (0.0, 0.0, 0.0)
                )
                for value_index in range(3):
                    _bounded_close(
                        opening[value_index],
                        prior[value_index],
                        label="bounded stock key continuity",
                    )
        previous_closing = current_closing
        current_opening = {}
        current_closing = {}

    for row in _bounded_rows(root, records, "stock"):
        period = _bounded_period(row, label="bounded stock closure")
        if period != current_period:
            finish_period()
            period_index += 1
            if period_index >= len(periods) or period != periods[period_index]:
                raise VerificationError(
                    "bounded stock closure periods are missing, duplicated, or out of order"
                )
            current_period = period
        axis = str(row.get("axis") or "")
        if axis not in {"holder", "instrument"}:
            raise VerificationError("bounded stock closure has an unknown axis")
        key = (
            axis,
            str(row.get("holder_sector") or ""),
            str(row.get("holder_subsector") or ""),
            str(row.get("instrument_type") or ""),
            str(row.get("maturity_bucket") or ""),
            str(row.get("debt_scope") or ""),
        )
        if key in current_closing:
            raise VerificationError("bounded stock closure has a duplicate key")
        row_counts[period] += 1
        if key[1] not in _ALLOWED_HOLDERS:
            raise VerificationError("bounded stock closure has unknown holder taxonomy")
        if key[3] not in _ALLOWED_INSTRUMENTS:
            raise VerificationError(
                "bounded stock closure has unknown instrument taxonomy"
            )
        if key[4] not in _ALLOWED_MATURITY_BUCKETS:
            raise VerificationError(
                "bounded stock closure has unknown maturity taxonomy"
            )
        if axis == "instrument" and (key[1] or key[2]):
            raise VerificationError(
                "instrument-axis stock closure must not carry holder keys"
            )
        opening_face = _bounded_float(
            row, "opening_face_stock_bil", label="bounded stock closure"
        )
        event_face = _bounded_float(
            row, "event_face_stock_change_bil", label="bounded stock closure"
        )
        closing_face = _bounded_float(
            row, "closing_face_stock_bil", label="bounded stock closure"
        )
        opening_adjusted = _bounded_float(
            row,
            "opening_adjusted_principal_stock_bil",
            label="bounded stock closure",
        )
        event_adjusted = _bounded_float(
            row,
            "event_adjusted_principal_change_bil",
            label="bounded stock closure",
        )
        closing_adjusted = _bounded_float(
            row,
            "closing_adjusted_principal_stock_bil",
            label="bounded stock closure",
        )
        opening_debt = _bounded_float(
            row, "opening_debt_stock_bil", label="bounded stock closure"
        )
        event_debt = _bounded_float(
            row, "event_debt_stock_change_bil", label="bounded stock closure"
        )
        closing_debt = _bounded_float(
            row, "closing_debt_stock_bil", label="bounded stock closure"
        )
        for error_column, recomputed in (
            (
                "face_stock_closure_error_bil",
                closing_face - opening_face - event_face,
            ),
            (
                "adjusted_principal_closure_error_bil",
                closing_adjusted - opening_adjusted - event_adjusted,
            ),
            (
                "debt_stock_closure_error_bil",
                closing_debt - opening_debt - event_debt,
            ),
        ):
            stored = _bounded_float(
                row, error_column, label="bounded stock closure"
            )
            _bounded_close(
                stored,
                recomputed,
                label=f"bounded stock closure stale error {error_column}",
            )
            if abs(recomputed) > STOCK_TOLERANCE_BIL:
                raise VerificationError(
                    f"bounded stock closure identity failed: {error_column}"
                )
        current_opening[key] = (
            opening_face,
            opening_adjusted,
            opening_debt,
        )
        current_closing[key] = (
            closing_face,
            closing_adjusted,
            closing_debt,
        )
        if key[5] == "all_active_treasury":
            axis_totals = totals[period][axis]
            axis_totals["opening_face"] += opening_face
            axis_totals["closing_face"] += closing_face
            axis_totals["opening_adjusted"] += opening_adjusted
            axis_totals["closing_adjusted"] += closing_adjusted
            axis_totals["closing_debt"] += closing_debt
    finish_period()
    if period_index != len(periods) - 1:
        raise VerificationError(
            "bounded stock closure does not cover every committed period"
        )
    return totals


def _cross_check_bounded_accounting_stock(
    accounting: Mapping[tuple[str, str], Mapping[str, float]],
    stock: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]],
    results: pd.DataFrame,
) -> None:
    result_map = _bounded_result_snapshots(results)
    for period, closure in accounting.items():
        holder = stock[period]["holder"]
        instrument = stock[period]["instrument"]
        checks = {
            "opening_face_stock_bil": holder["opening_face"],
            "closing_face_stock_bil": holder["closing_face"],
            "opening_adjusted_principal_stock_bil": holder[
                "opening_adjusted"
            ],
            "closing_adjusted_principal_stock_bil": holder[
                "closing_adjusted"
            ],
            "holder_debt_total_bil": holder["closing_debt"],
            "instrument_debt_total_bil": instrument["closing_debt"],
            "aggregate_debt_bil": result_map[period[1]]["TotalDebt_Agg"],
        }
        for column, expected in checks.items():
            _bounded_close(
                float(closure[column]),
                expected,
                label=f"bounded stock/accounting bridge {column}",
            )


def _verify_bounded_route_closure(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
) -> None:
    period_index = 0
    current_period = periods[0]
    row_count = 0
    current_seen: set[tuple[str, str, str, str, str]] = set()
    previous_closing: dict[tuple[str, str, str, str, str], float] = {}
    current_opening: dict[tuple[str, str, str, str, str], float] = {}
    current_closing: dict[tuple[str, str, str, str, str], float] = {}

    def finish_period() -> None:
        nonlocal row_count, previous_closing, current_opening, current_closing, current_seen
        if row_count == 0:
            raise VerificationError(
                "bounded route closure does not cover every committed period"
            )
        if period_index:
            for key in set(previous_closing) | set(current_opening):
                _bounded_close(
                    current_opening.get(key, 0.0),
                    previous_closing.get(key, 0.0),
                    label="bounded route-stock continuity",
                )
        previous_closing = current_closing
        current_opening = {}
        current_closing = {}
        current_seen = set()
        row_count = 0

    for row in _bounded_rows(root, records, "route"):
        period = _bounded_period(row, label="bounded route closure")
        if period != current_period:
            finish_period()
            period_index += 1
            if period_index >= len(periods) or period != periods[period_index]:
                raise VerificationError(
                    "bounded route closure periods are missing, duplicated, or out of order"
                )
            current_period = period
        key = (
            str(row.get("route_holder_sector") or ""),
            str(row.get("route_holder_subsector") or ""),
            str(row.get("instrument_type") or ""),
            str(row.get("maturity_bucket") or ""),
            str(row.get("debt_scope") or ""),
        )
        if key in current_seen:
            raise VerificationError("bounded route closure has a duplicate key")
        current_seen.add(key)
        row_count += 1
        if key[0] not in _ALLOWED_HOLDERS:
            raise VerificationError("bounded route closure has unknown holder taxonomy")
        if key[2] not in _ALLOWED_INSTRUMENTS:
            raise VerificationError(
                "bounded route closure has unknown instrument taxonomy"
            )
        if key[3] not in _ALLOWED_MATURITY_BUCKETS:
            raise VerificationError(
                "bounded route closure has unknown maturity taxonomy"
            )
        if row.get("route_stock_basis") != "tdc_principal_settlement_route":
            raise VerificationError(
                "bounded route closure has unexpected route_stock_basis"
            )
        if row.get("residual_basis") != (
            "none_fail_closed_no_unrestricted_residual"
        ):
            raise VerificationError(
                "bounded route closure has unexpected residual_basis"
            )
        opening = _bounded_float(
            row, "opening_route_stock_bil", label="bounded route closure"
        )
        closing = _bounded_float(
            row, "closing_route_stock_bil", label="bounded route closure"
        )
        journal_face = _bounded_float(
            row, "route_journal_face_change_bil", label="bounded route closure"
        )
        journal_adjusted = _bounded_float(
            row,
            "route_journal_adjusted_principal_change_bil",
            label="bounded route closure",
        )
        residual = _bounded_float(
            row,
            "route_stock_residual_or_indexation_bil",
            label="bounded route closure",
        )
        if abs(residual) > STOCK_TOLERANCE_BIL:
            raise VerificationError(
                "bounded route closure contains an unrestricted residual"
            )
        journal_debt = journal_adjusted if key[2] == "TIPS" else journal_face
        recomputed = closing - opening - journal_debt
        stored = _bounded_float(
            row, "closure_identity_error_bil", label="bounded route closure"
        )
        _bounded_close(
            stored,
            recomputed,
            label="bounded route closure stale identity error",
        )
        if abs(recomputed) > STOCK_TOLERANCE_BIL:
            raise VerificationError("bounded route closure identity failed")
        _bounded_float(
            row, "route_face_issued_bil", label="bounded route closure"
        )
        _bounded_float(
            row, "route_face_redeemed_bil", label="bounded route closure"
        )
        current_opening[key] = opening
        current_closing[key] = closing
    finish_period()
    if period_index != len(periods) - 1:
        raise VerificationError(
            "bounded route closure does not cover every committed period"
        )


def _verify_bounded_tdc_components(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    expected_periods = set(periods)
    specs = {
        str(spec["component_key"]): spec for spec in TDC_COMPONENT_SPECS
    }
    totals = {
        period: {"additive": 0.0, "direct": 0.0, "default": 0.0}
        for period in periods
    }
    amounts = {period: {} for period in periods}
    seen: set[tuple[tuple[str, str], str]] = set()
    for row in _bounded_rows(root, records, "tdc_components"):
        period = _bounded_period(row, label="bounded TDC components")
        if period not in expected_periods:
            raise VerificationError("bounded TDC component has an undeclared period")
        key = str(row.get("component_key") or "")
        spec = specs.get(key)
        if spec is None:
            raise VerificationError(
                f"bounded TDC component has unknown component_key: {key}"
            )
        if (period, key) in seen:
            raise VerificationError("bounded TDC component key is duplicated")
        seen.add((period, key))
        expected_id = f"tdc|{period[1]}|{key.replace('|', '_')}"
        if row.get("component_id") != expected_id:
            raise VerificationError("bounded TDC component_id is not canonical")
        for column in (
            "component_family",
            "holder_sector",
            "holder_subsector",
            "instrument_type",
            "payment_type",
            "accounting_basis",
        ):
            if str(row.get(column) or "") != str(spec[column]):
                raise VerificationError(
                    f"bounded TDC component registry drift: {key}.{column}"
                )
        flags = {
            column: _bounded_bool(
                row.get(column), label="bounded TDC components"
            )
            for column in (
                "is_additive_to_tdc_change",
                "enters_direct_interest_support",
                "enters_tdc_deposit_support_default",
            )
        }
        for column, expected in flags.items():
            if expected is not bool(spec[column]):
                raise VerificationError(
                    f"bounded TDC component registry drift: {key}.{column}"
                )
        if flags["enters_direct_interest_support"] and flags[
            "enters_tdc_deposit_support_default"
        ]:
            raise VerificationError(
                "bounded TDC component enters direct and default support"
            )
        if (
            flags["enters_direct_interest_support"]
            and row.get("holder_subsector")
            != "domestic_nonbank_deposit_funded"
        ):
            raise VerificationError(
                "bounded direct-interest component has the wrong holder route"
            )
        if row.get("tdc_amount_basis") != TDC_AMOUNT_BASIS:
            raise VerificationError("bounded TDC amount basis is unsupported")
        if row.get("overlap_policy") != TDC_OVERLAP_POLICY:
            raise VerificationError("bounded TDC overlap policy is unsupported")
        amount = _bounded_float(row, "amount_bil", label="bounded TDC components")
        if abs(amount) <= 1e-12:
            raise VerificationError(
                "bounded TDC component surface must omit zero-valued rows"
            )
        if flags["is_additive_to_tdc_change"]:
            totals[period]["additive"] += amount
        if flags["enters_direct_interest_support"]:
            totals[period]["direct"] += amount
        if flags["enters_tdc_deposit_support_default"]:
            totals[period]["default"] += amount
        amounts[period][key] = amount
    for period in periods:
        result = result_evidence[period]
        for key, spec in specs.items():
            _bounded_close(
                amounts[period].get(key, 0.0),
                float(result[str(spec["column"])]),
                label=f"bounded TDC component/result bridge {key}",
            )
    return totals


def _verify_bounded_tdc_summary(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    components: Mapping[tuple[str, str], Mapping[str, float]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    expected_periods = set(periods)
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    numeric_columns = [
        column
        for column in HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_summary"]
        if column
        not in {
            "period_start",
            "period_end",
            "tdc_amount_basis",
            "holder_allocation_scope",
            "overlap_policy",
        }
    ]
    for row in _bounded_rows(root, records, "tdc_summary"):
        period = _bounded_period(row, label="bounded TDC summary")
        if period not in expected_periods or period in summaries:
            raise VerificationError(
                "bounded TDC summary period set is not exact and unique"
            )
        if row.get("tdc_amount_basis") != TDC_AMOUNT_BASIS:
            raise VerificationError("bounded TDC summary amount basis is unsupported")
        if row.get("holder_allocation_scope") != TDC_HOLDER_SCOPE:
            raise VerificationError(
                "bounded TDC summary holder-allocation scope is unsupported"
            )
        if row.get("overlap_policy") != TDC_OVERLAP_POLICY:
            raise VerificationError("bounded TDC summary overlap policy is unsupported")
        values = {
            column: _bounded_float(row, column, label="bounded TDC summary")
            for column in numeric_columns
        }
        identity = (
            values["tdc_fiscal_flow_bil"]
            + values["tdc_debt_service_bil"]
            + values["tdc_auction_absorption_du_bil"]
            + values["tdc_secondary_trades_bil"]
            + values["tdc_other_bil"]
        )
        _bounded_close(
            values["tdc_change_bil"],
            identity,
            label="bounded TDC summary component identity",
        )
        _bounded_close(
            values["component_sum_bil"],
            identity,
            label="bounded TDC summary component sum",
        )
        _bounded_close(
            values["component_sum_error_bil"],
            values["tdc_change_bil"] - values["component_sum_bil"],
            label="bounded TDC summary stale component error",
        )
        if abs(values["component_sum_error_bil"]) > STOCK_TOLERANCE_BIL:
            raise VerificationError("bounded TDC summary component identity failed")
        _bounded_close(
            values["tdc_change_ex_overlap_bil"],
            values["tdc_change_bil"] - values["overlap_cashflow_bil"],
            label="bounded TDC ex-overlap identity",
        )
        _bounded_close(
            values["net_du_principal_issuance_cashflow_bil"],
            values["gross_principal_cash_paid_to_du_bil"]
            - values["gross_issuance_proceeds_absorbed_by_du_bil"],
            label="bounded TDC principal/issuance cashflow identity",
        )
        _bounded_close(
            components[period]["additive"],
            values["tdc_change_bil"],
            label="bounded TDC additive components",
        )
        _bounded_close(
            components[period]["direct"],
            values["overlap_cashflow_bil"],
            label="bounded TDC direct-interest overlap",
        )
        _bounded_close(
            components[period]["default"],
            values["tdc_change_ex_overlap_bil"],
            label="bounded TDC default-support components",
        )
        result = result_evidence[period]
        for summary_column, result_column in (
            _BOUNDED_TDC_SUMMARY_RESULT_BRIDGES.items()
        ):
            _bounded_close(
                values[summary_column],
                float(result[result_column]),
                label=f"bounded TDC summary/result bridge {summary_column}",
            )
        overlap = sum(
            float(result[column]) for column in TDC_OVERLAP_COLUMNS
        )
        _bounded_close(
            values["overlap_cashflow_bil"],
            overlap,
            label="bounded TDC summary/result bridge overlap_cashflow_bil",
        )
        _bounded_close(
            values["tdc_change_ex_overlap_bil"],
            float(result["TDC_Change"]) - overlap,
            label="bounded TDC summary/result bridge tdc_change_ex_overlap_bil",
        )
        summaries[period] = values
    if set(summaries) != expected_periods:
        raise VerificationError(
            "bounded TDC summary does not cover every committed period"
        )
    return summaries


def _verify_bounded_principal(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    totals = {
        period: {
            "tdc_principal_cash_paid_to_du_bil": 0.0,
            "tdc_principal_redeemed_to_du_bil": 0.0,
            "tdc_principal_cash_paid_to_du_domestic_nonbank_bil": 0.0,
            "tdc_principal_cash_paid_to_du_mmf_bil": 0.0,
            "tdc_principal_cash_paid_to_du_mmf_plumbing_bil": 0.0,
            "face_redeemed_bil": 0.0,
            "cash_paid_bil": 0.0,
            "explicit_retirement_face_bil": 0.0,
            "explicit_retirement_cash_bil": 0.0,
        }
        for period in periods
    }
    tracker = _BoundedPeriodKeyTracker(periods, label="bounded principal")
    key_columns = [
        column
        for column in PRINCIPAL_COLUMNS
        if column
        not in {
            "face_redeemed_bil",
            "principal_redeemed_bil",
            "cash_paid_bil",
            "adjusted_principal_stock_removed_bil",
            "tdc_principal_cash_paid_to_du_bil",
            "tdc_principal_redeemed_to_du_bil",
            "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
            "tdc_principal_redeemed_to_du_domestic_nonbank_bil",
            "tdc_principal_cash_paid_to_du_mmf_bil",
            "tdc_principal_redeemed_to_du_mmf_bil",
            "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
            "tdc_principal_redeemed_to_du_mmf_plumbing_bil",
            "flow_count",
        }
    ]
    numeric_columns = [
        "face_redeemed_bil",
        "principal_redeemed_bil",
        "cash_paid_bil",
        "adjusted_principal_stock_removed_bil",
        "tdc_principal_cash_paid_to_du_bil",
        "tdc_principal_redeemed_to_du_bil",
        "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
        "tdc_principal_redeemed_to_du_domestic_nonbank_bil",
        "tdc_principal_cash_paid_to_du_mmf_bil",
        "tdc_principal_redeemed_to_du_mmf_bil",
        "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
        "tdc_principal_redeemed_to_du_mmf_plumbing_bil",
    ]
    for row in _bounded_rows(root, records, "principal"):
        period = _bounded_period(row, label="bounded principal")
        tracker.add(
            period,
            tuple(str(row.get(column) or "") for column in key_columns[2:]),
        )
        if str(row.get("holder_sector") or "") not in _ALLOWED_HOLDERS:
            raise VerificationError("bounded principal has unknown holder taxonomy")
        if str(row.get("instrument_type") or "") not in _ALLOWED_INSTRUMENTS:
            raise VerificationError("bounded principal has unknown instrument taxonomy")
        if str(row.get("maturity_bucket") or "") not in _ALLOWED_MATURITY_BUCKETS:
            raise VerificationError("bounded principal has unknown maturity taxonomy")
        redemption_type = str(row.get("redemption_type") or "")
        if not redemption_type:
            raise VerificationError("bounded principal has blank redemption_type")
        _bounded_int(row, "flow_count", label="bounded principal", minimum=1)
        values = {
            column: _bounded_float(row, column, label="bounded principal")
            for column in numeric_columns
        }
        _bounded_close(
            values["tdc_principal_cash_paid_to_du_bil"],
            values["tdc_principal_cash_paid_to_du_domestic_nonbank_bil"]
            + values["tdc_principal_cash_paid_to_du_mmf_bil"],
            label="bounded principal cash-to-DU identity",
        )
        _bounded_close(
            values["tdc_principal_redeemed_to_du_bil"],
            values["tdc_principal_redeemed_to_du_domestic_nonbank_bil"]
            + values["tdc_principal_redeemed_to_du_mmf_bil"],
            label="bounded principal redeemed-to-DU identity",
        )
        for column in totals[period]:
            if column.startswith("explicit_retirement_"):
                continue
            totals[period][column] += values[column]
        if redemption_type == "explicit_retirement_at_par":
            totals[period]["explicit_retirement_face_bil"] += values[
                "face_redeemed_bil"
            ]
            totals[period]["explicit_retirement_cash_bil"] += values[
                "cash_paid_bil"
            ]
    for period in periods:
        _bounded_close(
            totals[period]["cash_paid_bil"],
            float(result_evidence[period]["PrincipalPaid_Bonds"]),
            label="bounded principal/results cash-paid bridge",
        )
        _bounded_close(
            totals[period]["explicit_retirement_face_bil"],
            float(result_evidence[period]["CBOBuybackFaceRetired"]),
            label="bounded principal/results buyback-face bridge",
        )
        _bounded_close(
            totals[period]["explicit_retirement_cash_bil"],
            float(result_evidence[period]["CBOBuybackCashPaid"]),
            label="bounded principal/results buyback-cash bridge",
        )
    return totals


def _cross_check_bounded_principal_summary(
    principal: Mapping[tuple[str, str], Mapping[str, float]],
    summaries: Mapping[tuple[str, str], Mapping[str, float]],
) -> None:
    checks = {
        "gross_principal_cash_paid_to_du_bil": "tdc_principal_cash_paid_to_du_bil",
        "tdc_debt_service_principal_to_du_bil": "tdc_principal_redeemed_to_du_bil",
        "gross_principal_cash_paid_to_du_domestic_nonbank_bil": (
            "tdc_principal_cash_paid_to_du_domestic_nonbank_bil"
        ),
        "gross_principal_cash_paid_to_du_mmf_bil": (
            "tdc_principal_cash_paid_to_du_mmf_bil"
        ),
        "gross_principal_cash_paid_to_du_mmf_plumbing_bil": (
            "tdc_principal_cash_paid_to_du_mmf_plumbing_bil"
        ),
    }
    for period, summary in summaries.items():
        for summary_column, principal_column in checks.items():
            _bounded_close(
                summary[summary_column],
                principal[period][principal_column],
                label=f"bounded principal/TDC bridge {summary_column}",
            )


def _verify_bounded_issuance(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[str, Any]:
    annual: dict[int, dict[str, float]] = {}
    period_totals = {
        period: {
            "face_issued_bil": 0.0,
            "cash_proceeds_bil": 0.0,
            "discount_or_premium_bil": 0.0,
            "issue_discount_cost_bil": 0.0,
            "cash_financing_face_issued_bil": 0.0,
            "cash_financing_proceeds_bil": 0.0,
        }
        for period in periods
    }
    tracker = _BoundedPeriodKeyTracker(periods, label="bounded issuance")
    key_columns = (
        "holder_sector",
        "holder_subsector",
        "issuance_leg",
        "instrument_type",
        "maturity_bucket",
    )
    for row in _bounded_rows(root, records, "issuance"):
        period = _bounded_period(row, label="bounded issuance")
        tracker.add(
            period, tuple(str(row.get(column) or "") for column in key_columns)
        )
        holder = str(row.get("holder_sector") or "")
        instrument = str(row.get("instrument_type") or "")
        maturity_bucket = str(row.get("maturity_bucket") or "")
        issuance_leg = str(row.get("issuance_leg") or "ordinary_issuance")
        if issuance_leg not in {
            "ordinary_issuance",
            "cbo_reference_face_issuance",
            "tga_floor_cash_financing",
        }:
            raise VerificationError("bounded issuance has unknown issuance-leg taxonomy")
        if holder not in _ALLOWED_HOLDERS:
            raise VerificationError("bounded issuance has unknown holder taxonomy")
        if instrument not in _ALLOWED_INSTRUMENTS:
            raise VerificationError("bounded issuance has unknown instrument taxonomy")
        if maturity_bucket not in _ALLOWED_MATURITY_BUCKETS:
            raise VerificationError("bounded issuance has unknown maturity taxonomy")
        _bounded_int(row, "flow_count", label="bounded issuance", minimum=1)
        face = _bounded_float(row, "face_issued_bil", label="bounded issuance")
        if face < -STOCK_TOLERANCE_BIL:
            raise VerificationError("bounded issuance has negative face")
        term = _bounded_float(
            row, "weighted_original_term_years", label="bounded issuance"
        )
        if term < 0.0:
            raise VerificationError("bounded issuance has negative original term")
        cash = _bounded_float(
            row, "cash_proceeds_bil", label="bounded issuance"
        )
        discount = _bounded_float(
            row, "discount_or_premium_bil", label="bounded issuance"
        )
        _bounded_close(
            discount,
            face - cash,
            label="bounded issuance discount/proceeds identity",
        )
        period_totals[period]["face_issued_bil"] += face
        period_totals[period]["cash_proceeds_bil"] += cash
        period_totals[period]["discount_or_premium_bil"] += discount
        period_totals[period]["issue_discount_cost_bil"] += max(0.0, discount)
        if issuance_leg == "tga_floor_cash_financing":
            period_totals[period]["cash_financing_face_issued_bil"] += face
            period_totals[period]["cash_financing_proceeds_bil"] += cash
        for column in (
            "coupon_rate_decimal",
            "reference_rate_decimal",
            "spread_bps",
            "issue_yield_decimal",
        ):
            _bounded_optional_float(
                row.get(column), label=f"bounded issuance {column}"
            )
        fiscal_year = _fiscal_year(period[1])
        bucket = annual.setdefault(
            fiscal_year,
            {
                "new_issuance_face_bil": 0.0,
                "new_issuance_original_term_face_years_bil": 0.0,
                "new_issuance_bill_face_bil": 0.0,
                "new_issuance_short_face_bil": 0.0,
            },
        )
        bucket["new_issuance_face_bil"] += face
        bucket["new_issuance_original_term_face_years_bil"] += face * term
        if instrument == "Fixed" and maturity_bucket == "bills":
            bucket["new_issuance_bill_face_bil"] += face
        if term <= 1.0 + 1e-9:
            bucket["new_issuance_short_face_bil"] += face
    for period in periods:
        _bounded_close(
            period_totals[period]["face_issued_bil"],
            float(result_evidence[period]["NewDebtIssued"]),
            label="bounded issuance/results face bridge",
        )
        _bounded_close(
            period_totals[period]["cash_proceeds_bil"],
            float(result_evidence[period]["AuctionProceeds"]),
            label="bounded issuance/results proceeds bridge",
        )
        if "CashFinancingFaceIssued" in result_evidence[period]:
            _bounded_close(
                period_totals[period]["cash_financing_face_issued_bil"],
                float(result_evidence[period]["CashFinancingFaceIssued"]),
                label="bounded cash-financing/results face bridge",
            )
            _bounded_close(
                period_totals[period]["cash_financing_proceeds_bil"],
                float(result_evidence[period]["CashFinancingProceeds"]),
                label="bounded cash-financing/results proceeds bridge",
            )
        if (
            float(result_evidence[period]["IssueDiscountCost_Period"])
            + STOCK_TOLERANCE_BIL
            < period_totals[period]["issue_discount_cost_bil"]
        ):
            raise VerificationError(
                "bounded issuance/results discount cost is below the "
                "recoverable aggregate lower bound"
            )
    return {"annual": annual, "period": period_totals}


def _verify_bounded_payment(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[tuple[str, str], float]:
    tracker = _BoundedPeriodKeyTracker(periods, label="bounded payment")
    additive_cash = {period: 0.0 for period in periods}
    key_columns = tuple(
        column
        for column in PAYMENT_COLUMNS
        if column
        not in {"period_start", "period_end", "amount_bil", "flow_count"}
    )
    for row in _bounded_rows(root, records, "payment"):
        period = _bounded_period(row, label="bounded payment")
        tracker.add(
            period, tuple(str(row.get(column) or "") for column in key_columns)
        )
        if str(row.get("holder_sector") or "") not in _ALLOWED_HOLDERS:
            raise VerificationError("bounded payment has unknown holder taxonomy")
        if str(row.get("instrument_type") or "") not in _ALLOWED_INSTRUMENTS:
            raise VerificationError("bounded payment has unknown instrument taxonomy")
        if str(row.get("maturity_bucket") or "") not in _ALLOWED_MATURITY_BUCKETS:
            raise VerificationError("bounded payment has unknown maturity taxonomy")
        if not str(row.get("payment_type") or "").strip():
            raise VerificationError("bounded payment has blank payment_type")
        if not str(row.get("accounting_basis") or "").strip():
            raise VerificationError("bounded payment has blank accounting_basis")
        additive = _bounded_bool(
            row.get("is_additive_to_cash_total"), label="bounded payment"
        )
        amount = _bounded_float(row, "amount_bil", label="bounded payment")
        if additive:
            additive_cash[period] += amount
        _bounded_int(row, "flow_count", label="bounded payment", minimum=1)
    for period in periods:
        _bounded_close(
            additive_cash[period],
            float(result_evidence[period]["InterestOutlay_Period"]),
            label="bounded payment/results interest-outlay bridge",
        )
    return additive_cash


def _cross_check_bounded_flow_bridges(
    periods: list[tuple[str, str]],
    *,
    ledger: Mapping[tuple[str, str], Mapping[str, float]],
    issuance: Mapping[tuple[str, str], Mapping[str, float]],
    principal: Mapping[tuple[str, str], Mapping[str, float]],
    payment: Mapping[tuple[str, str], float],
    results: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    financing: dict[tuple[str, str], dict[str, float]] = {}
    for period in periods:
        expected_face_change = (
            float(issuance[period]["face_issued_bil"])
            - float(principal[period]["face_redeemed_bil"])
            + float(
                ledger[period][
                    "nonmarketable_interest_capitalized_bil"
                ]
            )
        )
        _bounded_close(
            float(ledger[period]["face_stock_change_bil"]),
            expected_face_change,
            label="bounded ledger/issuance/principal face-stock bridge",
        )
        values = {
            "interest_outlay_bil": float(payment[period]),
            # The compact issuance reducer nets premiums and discounts within
            # a fixed taxonomy key.  It proves a lower bound above, but the
            # retained daily engine value is the exact gross-positive cost.
            "issue_discount_cost_bil": float(
                results[period]["IssueDiscountCost_Period"]
            ),
            "nonmarketable_interest_capitalized_bil": float(
                ledger[period][
                    "nonmarketable_interest_capitalized_bil"
                ]
            ),
            "tips_inflation_accretion_bil": float(
                ledger[period]["tips_inflation_accretion_bil"]
            ),
        }
        values["modeled_financing_cost_bil"] = sum(values.values())
        for finance_column, result_column in (
            _BOUNDED_FINANCE_RESULT_BRIDGES.items()
        ):
            _bounded_close(
                values[finance_column],
                float(results[period][result_column]),
                label=(
                    "bounded independent financing/results bridge "
                    f"{finance_column}"
                ),
            )
        financing[period] = values
    return financing


def _verify_bounded_scenario_metrics(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    result_evidence: Mapping[tuple[str, str], Mapping[str, float]],
) -> dict[int, dict[str, Any]]:
    expected_dates = [end for _start, end in periods]
    period_by_end = {period[1]: period for period in periods}
    observed_dates: list[str] = []
    snapshots: dict[int, dict[str, Any]] = {}
    for row in _bounded_rows(root, records, "scenario_metrics"):
        observed_date = _bounded_date(
            row.get("date"), label="bounded scenario metrics"
        ).isoformat()
        period = period_by_end.get(observed_date)
        if period is None:
            raise VerificationError(
                "bounded scenario metrics contains an undeclared period end"
            )
        observed_dates.append(observed_date)
        cutoff = _bounded_float(
            row,
            "short_maturity_cutoff_years",
            label="bounded scenario metrics",
        )
        if cutoff <= 0.0:
            raise VerificationError(
                "bounded scenario short-maturity cutoff must be positive"
            )
        for column in (
            "new_issuance_wam_years",
            "outstanding_controlled_wam_years",
        ):
            value = _bounded_float(
                row, column, label="bounded scenario metrics"
            )
            if value < 0.0:
                raise VerificationError(
                    f"bounded scenario metric is negative: {column}"
                )
        for column in (
            "new_issuance_bill_share",
            "outstanding_controlled_bill_share",
            "new_issuance_short_maturity_share",
            "outstanding_controlled_short_maturity_share",
        ):
            value = _bounded_float(
                row, column, label="bounded scenario metrics"
            )
            if value < -1e-12 or value > 1.0 + 1e-12:
                raise VerificationError(
                    f"bounded scenario share is outside [0, 1]: {column}"
                )
        for metric_column, result_column in (
            ("outstanding_controlled_wam_years", "OutstandingControlledWAM"),
            (
                "outstanding_controlled_bill_share",
                "OutstandingControlledBillShare",
            ),
            (
                "outstanding_controlled_short_maturity_share",
                "OutstandingControlledShortMaturityShare",
            ),
        ):
            _bounded_close(
                _bounded_float(
                    row, metric_column, label="bounded scenario metrics"
                ),
                float(result_evidence[period][result_column]),
                label=(
                    "bounded scenario-metric/result bridge "
                    f"{metric_column}"
                ),
            )
        parsed = date.fromisoformat(observed_date)
        if parsed.month == 9 and parsed.day == 30:
            snapshots[_fiscal_year(observed_date)] = {
                "snapshot_date": observed_date,
                "outstanding_controlled_wam_years": _bounded_float(
                    row,
                    "outstanding_controlled_wam_years",
                    label="bounded scenario metrics",
                ),
                "outstanding_controlled_bill_share": _bounded_float(
                    row,
                    "outstanding_controlled_bill_share",
                    label="bounded scenario metrics",
                ),
                "outstanding_controlled_short_maturity_share": _bounded_float(
                    row,
                    "outstanding_controlled_short_maturity_share",
                    label="bounded scenario metrics",
                ),
            }
    if observed_dates != expected_dates:
        raise VerificationError(
            "bounded scenario metrics must contain exactly one row per period end"
        )
    return snapshots


def _verify_bounded_debt_bridge(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    *,
    issuance: Mapping[tuple[str, str], Mapping[str, float]],
    principal: Mapping[tuple[str, str], Mapping[str, float]],
    ledger: Mapping[tuple[str, str], Mapping[str, float]],
    results: Mapping[tuple[str, str], Mapping[str, float]],
    expected_funding_modes: set[str],
) -> None:
    expected_dates = [end for _start, end in periods]
    period_by_end = {end: period for period in periods for end in (period[1],)}
    observed_dates: list[str] = []
    numeric_columns = [
        column
        for column in HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"]
        if column.endswith("_bil")
    ]
    for row in _bounded_rows(root, records, "debt_bridge"):
        observed_date = _bounded_date(
            row.get("date"), label="bounded debt bridge"
        ).isoformat()
        observed_dates.append(observed_date)
        period = period_by_end.get(observed_date)
        if period is None:
            raise VerificationError(
                "bounded debt bridge contains an undeclared period end"
            )
        values = {
            column: _bounded_float(row, column, label="bounded debt bridge")
            for column in numeric_columns
        }
        _bounded_close(
            values["target_error_bil"],
            values["controlled_debt_post_issuance_bil"]
            - values["controlled_public_marketable_target_bil"],
            label="bounded debt bridge target identity",
        )
        funding_mode = str(row.get("funding_mode") or "")
        if funding_mode not in expected_funding_modes:
            raise VerificationError(
                "bounded debt bridge funding mode differs from run manifest"
            )
        reference_financing_mode = (
            funding_mode
            == "cbo_debt_reference_plus_tga_floor_financing_v1"
        )
        if (
            not reference_financing_mode
            and abs(values["target_error_bil"]) > _BOUNDED_TARGET_TOLERANCE_BIL
        ):
            raise VerificationError(
                "bounded controlled-debt target error exceeds 1e-6 bil"
            )
        if reference_financing_mode:
            _bounded_close(
                values["cbo_controlled_debt_reference_bil"],
                values["controlled_public_marketable_target_bil"],
                label="bounded CBO debt-reference alias",
            )
            _bounded_close(
                values["scenario_controlled_debt_bil"],
                values["controlled_debt_post_issuance_bil"],
                label="bounded scenario-debt alias",
            )
            _bounded_close(
                values["debt_drift_from_reference_bil"],
                values["scenario_controlled_debt_bil"]
                - values["cbo_controlled_debt_reference_bil"],
                label="bounded debt-drift identity",
            )
            if (
                values["debt_drift_from_reference_bil"]
                < -_BOUNDED_TARGET_TOLERANCE_BIL
            ):
                raise VerificationError(
                    "bounded scenario debt falls below CBO reference"
                )
            _bounded_close(
                values["cash_financing_face_issued_bil"],
                float(issuance[period]["cash_financing_face_issued_bil"]),
                label="bounded debt-bridge/cash-financing face bridge",
            )
            _bounded_close(
                values["cash_financing_proceeds_bil"],
                float(issuance[period]["cash_financing_proceeds_bil"]),
                label="bounded debt-bridge/cash-financing proceeds bridge",
            )
        _bounded_close(
            values["face_issued_bil"],
            float(issuance[period]["face_issued_bil"]),
            label="bounded debt-bridge/issuance face bridge",
        )
        _bounded_close(
            values["face_retired_bil"],
            float(principal[period]["explicit_retirement_face_bil"]),
            label="bounded debt-bridge/principal retirement bridge",
        )
        _bounded_close(
            values["tips_principal_indexation_bil"],
            float(ledger[period]["tips_inflation_accretion_bil"]),
            label="bounded debt-bridge/ledger TIPS indexation bridge",
        )
        _bounded_close(
            values["face_issued_bil"],
            float(results[period]["NewDebtIssued"]),
            label="bounded debt-bridge/results issuance bridge",
        )
        _bounded_close(
            values["face_retired_bil"],
            float(results[period]["CBOBuybackFaceRetired"]),
            label="bounded debt-bridge/results retirement bridge",
        )
        for column in (
            "funding_mode",
            "intragovernmental_treatment",
            "fed_held_treasury_treatment",
            "public_nonmarketable_treatment",
        ):
            if not str(row.get(column) or "").strip():
                raise VerificationError(f"bounded debt bridge has blank {column}")
    if observed_dates != expected_dates:
        raise VerificationError(
            "bounded debt bridge must contain exactly one row per period end"
        )


def _verify_bounded_annual(
    root: Path,
    records: Mapping[str, Mapping[str, Any]],
    periods: list[tuple[str, str]],
    *,
    summaries: Mapping[tuple[str, str], Mapping[str, float]],
    issuance: Mapping[int, Mapping[str, float]],
    metrics: Mapping[int, Mapping[str, Any]],
    financing: Mapping[tuple[str, str], Mapping[str, float]],
) -> None:
    fiscal_buckets: dict[int, dict[str, Any]] = {}
    for period in periods:
        fiscal_year = _fiscal_year(period[1])
        bucket = fiscal_buckets.setdefault(
            fiscal_year,
            {
                "period_start": period[0],
                "period_end": period[1],
                "coverage_days": 0,
                "tdc_change_bil": 0.0,
                "overlap_cashflow_bil": 0.0,
                "tdc_change_ex_overlap_bil": 0.0,
                "interest_outlay_bil": 0.0,
                "issue_discount_cost_bil": 0.0,
                "nonmarketable_interest_capitalized_bil": 0.0,
                "tips_inflation_accretion_bil": 0.0,
                "modeled_financing_cost_bil": 0.0,
            },
        )
        bucket["period_end"] = period[1]
        bucket["coverage_days"] += (
            date.fromisoformat(period[1]) - date.fromisoformat(period[0])
        ).days
        for column in (
            "tdc_change_bil",
            "overlap_cashflow_bil",
            "tdc_change_ex_overlap_bil",
        ):
            bucket[column] += float(summaries[period][column])
        for column in _BOUNDED_FINANCE_RESULT_BRIDGES:
            bucket[column] += float(financing[period][column])

    years = sorted(fiscal_buckets)
    observed_years: list[int] = []
    cumulative = {
        "tdc_change_bil": 0.0,
        "overlap_cashflow_bil": 0.0,
        "tdc_change_ex_overlap_bil": 0.0,
        "interest_outlay_bil": 0.0,
        "issue_discount_cost_bil": 0.0,
        "nonmarketable_interest_capitalized_bil": 0.0,
        "tips_inflation_accretion_bil": 0.0,
        "modeled_financing_cost_bil": 0.0,
    }
    for row in _bounded_rows(root, records, "annual"):
        period = _bounded_period(row, label="bounded annual summary")
        fiscal_year = _fiscal_year(period[1])
        if fiscal_year in observed_years or fiscal_year not in fiscal_buckets:
            raise VerificationError(
                "bounded annual summary fiscal-year set is not exact and unique"
            )
        observed_years.append(fiscal_year)
        bucket = fiscal_buckets[fiscal_year]
        if period != (bucket["period_start"], bucket["period_end"]):
            raise VerificationError(
                "bounded annual summary period bounds disagree with period evidence"
            )
        expected_days = (
            date(fiscal_year, 9, 30) - date(fiscal_year - 1, 9, 30)
        ).days
        coverage_days = int(bucket["coverage_days"])
        partial = coverage_days != expected_days
        if partial and fiscal_year == years[0]:
            expected_label = f"FY{fiscal_year}_PARTIAL_OPENING"
        elif partial and fiscal_year == years[-1]:
            expected_label = f"FY{fiscal_year}_PARTIAL_CLOSING"
        else:
            expected_label = f"FY{fiscal_year}"
        if row.get("period_label") != expected_label:
            raise VerificationError("bounded annual period label is incorrect")
        if _bounded_bool(
            row.get("is_partial_period"), label="bounded annual summary"
        ) is not partial:
            raise VerificationError("bounded annual partial-period flag is stale")
        if _bounded_int(
            row, "coverage_days", label="bounded annual summary", minimum=1
        ) != coverage_days:
            raise VerificationError("bounded annual coverage_days is stale")
        if _bounded_int(
            row,
            "expected_coverage_days",
            label="bounded annual summary",
            minimum=1,
        ) != expected_days:
            raise VerificationError(
                "bounded annual expected_coverage_days is stale"
            )
        if row.get("aggregation_clock_id") != AGGREGATION_CLOCK_ID:
            raise VerificationError("bounded annual aggregation clock is unsupported")
        for column in (
            "tdc_change_bil",
            "overlap_cashflow_bil",
            "tdc_change_ex_overlap_bil",
        ):
            value = _bounded_float(row, column, label="bounded annual summary")
            _bounded_close(
                value,
                float(bucket[column]),
                label=f"bounded annual TDC aggregation {column}",
            )
            cumulative[column] += value
        _bounded_close(
            _bounded_float(
                row, "tdc_change_ex_overlap_bil", label="bounded annual summary"
            ),
            _bounded_float(
                row, "tdc_change_bil", label="bounded annual summary"
            )
            - _bounded_float(
                row, "overlap_cashflow_bil", label="bounded annual summary"
            ),
            label="bounded annual TDC ex-overlap identity",
        )
        for annual_column, cumulative_column in (
            ("tdc_change_bil", "cumulative_tdc_change_bil"),
            ("overlap_cashflow_bil", "cumulative_overlap_cashflow_bil"),
            (
                "tdc_change_ex_overlap_bil",
                "cumulative_tdc_change_ex_overlap_bil",
            ),
        ):
            _bounded_close(
                _bounded_float(
                    row, cumulative_column, label="bounded annual summary"
                ),
                cumulative[annual_column],
                label=f"bounded annual cumulative identity {cumulative_column}",
            )

        finance_columns = (
            "interest_outlay_bil",
            "issue_discount_cost_bil",
            "nonmarketable_interest_capitalized_bil",
            "tips_inflation_accretion_bil",
        )
        finance = {
            column: _bounded_float(
                row, column, label="bounded annual financing"
            )
            for column in finance_columns
        }
        modeled = _bounded_float(
            row, "modeled_financing_cost_bil", label="bounded annual financing"
        )
        _bounded_close(
            modeled,
            sum(finance.values()),
            label="bounded annual modeled financing-cost identity",
        )
        for column, value in (
            *finance.items(),
            ("modeled_financing_cost_bil", modeled),
        ):
            _bounded_close(
                value,
                float(bucket[column]),
                label=f"bounded annual independent financing bridge {column}",
            )
        cumulative["modeled_financing_cost_bil"] += modeled
        for column, cumulative_column in (
            ("interest_outlay_bil", "cumulative_interest_outlay_bil"),
            (
                "issue_discount_cost_bil",
                "cumulative_issue_discount_cost_bil",
            ),
            (
                "nonmarketable_interest_capitalized_bil",
                "cumulative_nonmarketable_interest_capitalized_bil",
            ),
            (
                "tips_inflation_accretion_bil",
                "cumulative_tips_inflation_accretion_bil",
            ),
        ):
            cumulative[column] += finance[column]
            _bounded_close(
                _bounded_float(
                    row, cumulative_column, label="bounded annual financing"
                ),
                cumulative[column],
                label=f"bounded annual cumulative financing {cumulative_column}",
            )
        _bounded_close(
            _bounded_float(
                row,
                "cumulative_modeled_financing_cost_bil",
                label="bounded annual financing",
            ),
            cumulative["modeled_financing_cost_bil"],
            label="bounded annual cumulative modeled financing cost",
        )
        if row.get("modeled_financing_cost_basis") != (
            "nominal_model_cost_incurred_within_simulation_horizon"
        ):
            raise VerificationError(
                "bounded annual modeled financing-cost basis is unsupported"
            )
        if row.get("modeled_financing_cost_units") != (
            "billions_of_nominal_dollars"
        ):
            raise VerificationError(
                "bounded annual modeled financing-cost units are unsupported"
            )
        if row.get("cumulative_basis") != "since_simulation_origin":
            raise VerificationError("bounded annual cumulative basis is unsupported")

        issuance_values = issuance.get(
            fiscal_year,
            {
                "new_issuance_face_bil": 0.0,
                "new_issuance_original_term_face_years_bil": 0.0,
                "new_issuance_bill_face_bil": 0.0,
                "new_issuance_short_face_bil": 0.0,
            },
        )
        for column, expected in issuance_values.items():
            _bounded_close(
                _bounded_float(row, column, label="bounded annual issuance"),
                float(expected),
                label=f"bounded annual issuance aggregation {column}",
            )
        face = float(issuance_values["new_issuance_face_bil"])
        expected_wam = (
            float(
                issuance_values[
                    "new_issuance_original_term_face_years_bil"
                ]
            )
            / face
            if face > 1e-12
            else None
        )
        expected_bill_share = (
            float(issuance_values["new_issuance_bill_face_bil"]) / face
            if face > 1e-12
            else None
        )
        expected_short_share = (
            float(issuance_values["new_issuance_short_face_bil"]) / face
            if face > 1e-12
            else None
        )
        for column, expected in (
            ("new_issuance_wam_years", expected_wam),
            ("new_issuance_bill_share", expected_bill_share),
            ("new_issuance_short_maturity_share", expected_short_share),
        ):
            actual = _bounded_optional_float(
                row.get(column), label=f"bounded annual {column}"
            )
            if expected is None:
                if actual is not None:
                    raise VerificationError(
                        f"bounded annual {column} must be null without issuance"
                    )
            else:
                if actual is None:
                    raise VerificationError(f"bounded annual {column} is missing")
                _bounded_close(
                    actual,
                    expected,
                    label=f"bounded annual issuance ratio {column}",
                )

        snapshot = metrics.get(fiscal_year)
        snapshot_date = str(row.get("snapshot_date") or "")
        if snapshot is None:
            if snapshot_date:
                raise VerificationError(
                    "bounded annual summary has a non-September snapshot"
                )
            for column in (
                "outstanding_controlled_wam_years",
                "outstanding_controlled_bill_share",
                "outstanding_controlled_short_maturity_share",
            ):
                if _bounded_optional_float(
                    row.get(column), label=f"bounded annual {column}"
                ) is not None:
                    raise VerificationError(
                        "bounded annual maturity snapshot must be null without September 30"
                    )
        else:
            if snapshot_date != snapshot["snapshot_date"]:
                raise VerificationError(
                    "bounded annual snapshot date is not the exact September 30"
                )
            for column in (
                "outstanding_controlled_wam_years",
                "outstanding_controlled_bill_share",
                "outstanding_controlled_short_maturity_share",
            ):
                _bounded_close(
                    _bounded_float(row, column, label="bounded annual snapshot"),
                    float(snapshot[column]),
                    label=f"bounded annual snapshot {column}",
                )
    if observed_years != years:
        raise VerificationError(
            "bounded annual summary does not cover each fiscal-year bucket in order"
        )


class _BoundedPeriodKeyTracker:
    """Reject duplicates and disorder while retaining keys for one period only."""

    def __init__(
        self, periods: list[tuple[str, str]], *, label: str
    ) -> None:
        self.positions = {period: index for index, period in enumerate(periods)}
        self.label = label
        self.current_index = -1
        self.current_keys: set[tuple[str, ...]] = set()

    def add(
        self, period: tuple[str, str], key: tuple[str, ...]
    ) -> None:
        position = self.positions.get(period)
        if position is None:
            raise VerificationError(f"{self.label} contains an undeclared period")
        if position < self.current_index:
            raise VerificationError(f"{self.label} periods are out of order")
        if position > self.current_index:
            self.current_index = position
            self.current_keys.clear()
        if key in self.current_keys:
            raise VerificationError(f"{self.label} contains a duplicate reducer key")
        self.current_keys.add(key)


def _bounded_result_snapshots(
    results: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    required = ("Date", "TGA", "Reserves", "TDC_Level", "TotalDebt_Agg")
    _require_columns(results, required, label="bounded results")
    snapshots: dict[str, dict[str, float]] = {}
    for row in results.loc[:, list(required)].to_dict("records"):
        observed_date = _bounded_date(
            row.get("Date"), label="bounded results"
        ).isoformat()
        if observed_date in snapshots:
            raise VerificationError("bounded results contain duplicate dates")
        values: dict[str, float] = {}
        for column in required[1:]:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise VerificationError(
                    f"bounded results have malformed numeric values: {column}"
                ) from exc
            if not math.isfinite(value):
                raise VerificationError(
                    f"bounded results have nonfinite numeric values: {column}"
                )
            values[column] = value
        snapshots[observed_date] = values
    if len(snapshots) < 2:
        raise VerificationError("bounded results require at least two snapshots")
    ordered = sorted(snapshots)
    if list(snapshots) != ordered:
        raise VerificationError("bounded results dates are not ordered")
    return snapshots


def _bounded_period_result_evidence(
    results: pd.DataFrame,
    periods: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Return fixed-width daily-result evidence for each committed period."""

    flow_columns = tuple(
        dict.fromkeys(
            (
                *_BOUNDED_TDC_SUMMARY_RESULT_BRIDGES.values(),
                *TDC_IDENTITY_COLUMNS,
                *TDC_OVERLAP_COLUMNS,
                *(str(spec["column"]) for spec in TDC_COMPONENT_SPECS),
                *_BOUNDED_FINANCE_RESULT_BRIDGES.values(),
                "NewDebtIssued",
                "PrincipalPaid_Bonds",
                "CBOBuybackFaceRetired",
                "CBOBuybackCashPaid",
                "OutstandingControlledWAM",
                "OutstandingControlledBillShare",
                "OutstandingControlledShortMaturityShare",
            )
        )
    )
    required = ("Date", "TDC_Level", *flow_columns)
    _require_columns(results, required, label="bounded results")
    expected_dates = [periods[0][0], *(period[1] for period in periods)]
    observed_dates: list[str] = []
    levels: dict[str, float] = {}
    flows: dict[str, dict[str, float]] = {}
    period_ends = set(expected_dates[1:])
    for values in results.loc[:, list(required)].itertuples(
        index=False, name=None
    ):
        observed_date = _bounded_date(
            values[0], label="bounded results"
        ).isoformat()
        if observed_date in levels:
            raise VerificationError("bounded results contain duplicate dates")
        observed_dates.append(observed_date)
        level = _bounded_result_float(
            values[1], column="TDC_Level"
        )
        levels[observed_date] = level
        if observed_date in period_ends:
            flows[observed_date] = {
                column: _bounded_result_float(value, column=column)
                for column, value in zip(flow_columns, values[2:])
            }
    if observed_dates != expected_dates:
        raise VerificationError(
            "bounded results snapshots do not match committed periods"
        )
    evidence: dict[tuple[str, str], dict[str, float]] = {}
    for period in periods:
        row = flows[period[1]]
        _bounded_close(
            row["TDC_Change"],
            levels[period[1]] - levels[period[0]],
            label="bounded daily-result TDC level bridge",
        )
        _bounded_close(
            row["TDC_Change"],
            sum(row[column] for column in TDC_IDENTITY_COLUMNS),
            label="bounded daily-result TDC component identity",
        )
        _bounded_close(
            row["FinancingCost_Period"],
            sum(
                row[column]
                for name, column in _BOUNDED_FINANCE_RESULT_BRIDGES.items()
                if name != "modeled_financing_cost_bil"
            ),
            label="bounded daily-result financing identity",
        )
        evidence[period] = row
    return evidence


def _bounded_result_float(value: Any, *, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            f"bounded results have malformed numeric values: {column}"
        ) from exc
    if not math.isfinite(parsed):
        raise VerificationError(
            f"bounded results have nonfinite numeric values: {column}"
        )
    return parsed


def _bounded_period(
    row: Mapping[str, Any], *, label: str
) -> tuple[str, str]:
    start = _bounded_date(row.get("period_start"), label=label)
    end = _bounded_date(row.get("period_end"), label=label)
    if end <= start:
        raise VerificationError(f"{label} has a non-increasing period")
    return start.isoformat(), end.isoformat()


def _bounded_date(value: Any, *, label: str) -> date:
    text = str(value or "")
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise VerificationError(f"{label} has a malformed date: {text!r}") from exc
    if parsed.isoformat() != text:
        raise VerificationError(f"{label} date is not canonical ISO format")
    return parsed


def _bounded_float(
    row: Mapping[str, Any], column: str, *, label: str
) -> float:
    value = row.get(column)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(
            f"{label} has malformed numeric values: {column}"
        ) from exc
    if not math.isfinite(number):
        raise VerificationError(
            f"{label} has nonfinite numeric values: {column}"
        )
    return 0.0 if number == 0.0 else number


def _bounded_optional_float(value: Any, *, label: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"{label} is malformed") from exc
    if not math.isfinite(number):
        raise VerificationError(f"{label} is nonfinite")
    return 0.0 if number == 0.0 else number


def _bounded_int(
    row: Mapping[str, Any],
    column: str,
    *,
    label: str,
    minimum: int,
) -> int:
    value = row.get(column)
    if isinstance(value, bool):
        raise VerificationError(f"{label} has malformed integer: {column}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or not stripped.lstrip("-").isdigit():
            raise VerificationError(f"{label} has malformed integer: {column}")
        number = int(stripped)
    else:
        raise VerificationError(f"{label} has malformed integer: {column}")
    if number < minimum:
        raise VerificationError(
            f"{label} integer is below its minimum: {column}"
        )
    return number


def _bounded_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise VerificationError(f"{label} has a malformed boolean")


def _bounded_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise VerificationError(f"{label} is not a lowercase SHA-256")
    return digest


def _bounded_close(
    actual: float,
    expected: float,
    *,
    label: str,
    tolerance: float = STOCK_TOLERANCE_BIL,
) -> None:
    if not math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=tolerance
    ):
        raise VerificationError(
            f"{label} disagrees: observed={actual!r}, expected={expected!r}"
        )


def _fiscal_year(period_end: str) -> int:
    parsed = date.fromisoformat(period_end)
    return parsed.year + 1 if parsed.month >= 10 else parsed.year


def _verify_tdc_handoff_outputs(root: Path, manifest: dict[str, Any]) -> None:
    results = _read_results(_result_artifact_path(root, manifest))
    summary = _read_results(_output_artifact_path(root, manifest, "tdcsim_period_tdc_summary"))
    components = _read_results(_output_artifact_path(root, manifest, "tdcsim_period_tdc_components"))
    issuance = _read_results(_output_artifact_path(root, manifest, "tdcsim_period_issuance_flows"))
    principal = _read_results(_output_artifact_path(root, manifest, "tdcsim_period_principal_flows"))
    holder_stocks = _read_results(_output_artifact_path(root, manifest, "tdcsim_holder_stocks"))
    route_stocks = _read_results(_output_artifact_path(root, manifest, "tdcsim_tdc_principal_route_stocks"))
    route_closure = _read_results(_output_artifact_path(root, manifest, "tdcsim_tdc_principal_route_stock_closure"))
    accounting_journal = _read_results(
        _output_artifact_path(root, manifest, "tdcsim_accounting_journal")
    )
    accounting_closure = _read_results(
        _output_artifact_path(root, manifest, "tdcsim_accounting_closure")
    )
    _require_columns(
        summary,
        (
            "period_start",
            "period_end",
            "tdc_change_bil",
            "tdc_fiscal_flow_bil",
            "tdc_debt_service_bil",
            "tdc_debt_service_principal_to_du_bil",
            "gross_principal_cash_paid_to_du_bil",
            "gross_principal_cash_paid_to_du_domestic_nonbank_bil",
            "gross_principal_cash_paid_to_du_mmf_bil",
            "gross_principal_cash_paid_to_du_mmf_plumbing_bil",
            "tdc_auction_absorption_du_bil",
            "tdc_secondary_trades_bil",
            "tdc_other_bil",
            "overlap_cashflow_bil",
            "tdc_change_ex_overlap_bil",
            "component_sum_bil",
            "component_sum_error_bil",
            "gross_issuance_proceeds_absorbed_by_du_bil",
            "net_du_principal_issuance_cashflow_bil",
            "tdc_amount_basis",
            "holder_allocation_scope",
        ),
        label="tdcsim_period_tdc_summary",
    )
    _require_columns(
        components,
        (
            "period_start",
            "period_end",
            "component_id",
            "component_key",
            "amount_bil",
            "is_additive_to_tdc_change",
            "enters_direct_interest_support",
            "enters_tdc_deposit_support_default",
            "tdc_amount_basis",
        ),
        label="tdcsim_period_tdc_components",
    )
    _require_columns(
        principal,
        (
            "period_start",
            "period_end",
            "tdc_principal_recipient_sector",
            "tdc_principal_recipient_subsector",
            "tdc_principal_cash_paid_to_du_bil",
            "tdc_principal_redeemed_to_du_bil",
            "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
            "tdc_principal_redeemed_to_du_domestic_nonbank_bil",
            "tdc_principal_cash_paid_to_du_mmf_bil",
            "tdc_principal_redeemed_to_du_mmf_bil",
            "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
            "tdc_principal_redeemed_to_du_mmf_plumbing_bil",
            "tdc_principal_recipient_basis",
        ),
        label="tdcsim_period_principal_flows",
    )
    _require_columns(
        route_stocks,
        (
            "date",
            "route_holder_sector",
            "route_holder_subsector",
            "instrument_type",
            "maturity_bucket",
            "route_debt_held_bil",
            "route_face_stock_bil",
            "route_adjusted_principal_stock_bil",
            "debt_scope",
            "route_stock_basis",
        ),
        label="tdcsim_tdc_principal_route_stocks",
    )
    _require_columns(
        route_closure,
        (
            "period_start",
            "period_end",
            "route_holder_sector",
            "route_holder_subsector",
            "instrument_type",
            "maturity_bucket",
            "debt_scope",
            "opening_route_stock_bil",
            "route_face_issued_bil",
            "route_face_redeemed_bil",
            "route_journal_face_change_bil",
            "route_journal_adjusted_principal_change_bil",
            "route_stock_residual_or_indexation_bil",
            "closing_route_stock_bil",
            "closure_identity_error_bil",
            "route_stock_basis",
            "residual_basis",
        ),
        label="tdcsim_tdc_principal_route_stock_closure",
    )
    if summary.empty:
        raise VerificationError("tdcsim_period_tdc_summary must contain period rows")
    if components.empty:
        raise VerificationError("tdcsim_period_tdc_components must contain component rows")
    if route_stocks.empty:
        raise VerificationError("tdcsim_tdc_principal_route_stocks must contain route stock rows")
    if route_closure.empty:
        raise VerificationError("tdcsim_tdc_principal_route_stock_closure must contain period rows")
    route_stocks = _strict_route_stock_rows(route_stocks)
    route_closure = _strict_route_closure_rows(
        route_closure,
        label="tdcsim_tdc_principal_route_stock_closure",
    )
    if set(route_stocks["route_stock_basis"].astype(str).unique()) != {"tdc_principal_settlement_route"}:
        raise VerificationError("route stocks have unexpected route_stock_basis")
    if set(route_closure["route_stock_basis"].astype(str).unique()) != {"tdc_principal_settlement_route"}:
        raise VerificationError("route closure has unexpected route_stock_basis")
    if route_closure["closure_identity_error_bil"].abs().max() > 1e-7:
        raise VerificationError("route stock closure identity failed")
    _verify_accounting_journal_outputs(
        results,
        holder_stocks,
        accounting_journal,
        accounting_closure,
    )
    recomputed_route = pd.DataFrame(
        _route_stock_closure_handoff_tables(
            {
                "tdcsim_accounting_journal": accounting_journal.to_dict("records"),
                "tdcsim_period_issuance_flows": issuance.to_dict("records"),
                "tdcsim_period_principal_flows": principal.to_dict("records"),
                "tdcsim_tdc_principal_route_stocks": route_stocks.to_dict("records"),
            }
        )["tdcsim_tdc_principal_route_stock_closure"]
    )
    if recomputed_route.empty:
        raise VerificationError("route stock closure could not be independently recomputed")
    recomputed_route = _strict_route_closure_rows(
        recomputed_route,
        label="recomputed route stock closure",
    )
    if recomputed_route["closure_identity_error_bil"].abs().max() > 1e-7:
        raise VerificationError(
            "route stock closure fails against independently read journal and snapshots"
        )
    _compare_stored_route_closure(route_closure, recomputed_route)
    identity = (
        _numeric(summary, "tdc_fiscal_flow_bil")
        + _numeric(summary, "tdc_debt_service_bil")
        + _numeric(summary, "tdc_auction_absorption_du_bil")
        + _numeric(summary, "tdc_secondary_trades_bil")
        + _numeric(summary, "tdc_other_bil")
    )
    if (_numeric(summary, "tdc_change_bil") - identity).abs().max() > 1e-7:
        raise VerificationError("TDC summary component identity failed")
    if _numeric(summary, "component_sum_error_bil").abs().max() > 1e-7:
        raise VerificationError("TDC summary component_sum_error_bil exceeds tolerance")
    overlap_identity = (
        _numeric(summary, "tdc_change_bil")
        - _numeric(summary, "overlap_cashflow_bil")
        - _numeric(summary, "tdc_change_ex_overlap_bil")
    )
    if overlap_identity.abs().max() > 1e-7:
        raise VerificationError("TDC summary ex-overlap identity failed")
    net_principal_issuance = (
        _numeric(summary, "gross_principal_cash_paid_to_du_bil")
        - _numeric(summary, "gross_issuance_proceeds_absorbed_by_du_bil")
        - _numeric(summary, "net_du_principal_issuance_cashflow_bil")
    )
    if net_principal_issuance.abs().max() > 1e-7:
        raise VerificationError("TDC summary net principal/issuance cashflow identity failed")
    if principal.empty and _numeric(summary, "gross_principal_cash_paid_to_du_bil").abs().max() > 1e-7:
        raise VerificationError("TDC summary reports DU principal cash but principal flow table is empty")
    direct = _bool_series(components, "enters_direct_interest_support")
    default_tdc = _bool_series(components, "enters_tdc_deposit_support_default")
    if (direct & default_tdc).any():
        raise VerificationError("TDC component cannot enter both direct interest and default TDC support")
    if not set(components.loc[direct, "holder_subsector"].astype(str).unique()) <= {"domestic_nonbank_deposit_funded"}:
        raise VerificationError("direct-interest overlap components must be domestic nonbank only")
    grouped_direct = (
        components.loc[direct]
        .assign(amount_bil=_numeric(components.loc[direct], "amount_bil"))
        .groupby(["period_start", "period_end"], dropna=False)["amount_bil"]
        .sum()
    )
    grouped_tdc = (
        components.loc[default_tdc]
        .assign(amount_bil=_numeric(components.loc[default_tdc], "amount_bil"))
        .groupby(["period_start", "period_end"], dropna=False)["amount_bil"]
        .sum()
    )
    summary_indexed = summary.set_index(["period_start", "period_end"], drop=False)
    direct_delta = grouped_direct.reindex(summary_indexed.index, fill_value=0.0) - _numeric(
        summary_indexed, "overlap_cashflow_bil"
    )
    if direct_delta.abs().max() > 1e-7:
        raise VerificationError("TDC direct-interest overlap components do not match summary overlap")
    tdc_delta = grouped_tdc.reindex(summary_indexed.index, fill_value=0.0) - _numeric(
        summary_indexed, "tdc_change_ex_overlap_bil"
    )
    if tdc_delta.abs().max() > 1e-7:
        raise VerificationError("TDC default-support components do not match summary ex-overlap")
    if not principal.empty:
        principal_cash_identity = (
            _numeric(principal, "tdc_principal_cash_paid_to_du_bil")
            - _numeric(principal, "tdc_principal_cash_paid_to_du_domestic_nonbank_bil")
            - _numeric(principal, "tdc_principal_cash_paid_to_du_mmf_bil")
        )
        if principal_cash_identity.abs().max() > 1e-7:
            raise VerificationError("TDC principal row cash-to-DU identity failed")
        principal_redeemed_identity = (
            _numeric(principal, "tdc_principal_redeemed_to_du_bil")
            - _numeric(principal, "tdc_principal_redeemed_to_du_domestic_nonbank_bil")
            - _numeric(principal, "tdc_principal_redeemed_to_du_mmf_bil")
        )
        if principal_redeemed_identity.abs().max() > 1e-7:
            raise VerificationError("TDC principal row redeemed-to-DU identity failed")
        principal_grouped = (
            principal.assign(
                tdc_principal_cash_paid_to_du_bil=_numeric(principal, "tdc_principal_cash_paid_to_du_bil"),
                tdc_principal_redeemed_to_du_bil=_numeric(principal, "tdc_principal_redeemed_to_du_bil"),
                tdc_principal_cash_paid_to_du_domestic_nonbank_bil=_numeric(
                    principal, "tdc_principal_cash_paid_to_du_domestic_nonbank_bil"
                ),
                tdc_principal_cash_paid_to_du_mmf_bil=_numeric(principal, "tdc_principal_cash_paid_to_du_mmf_bil"),
                tdc_principal_cash_paid_to_du_mmf_plumbing_bil=_numeric(
                    principal, "tdc_principal_cash_paid_to_du_mmf_plumbing_bil"
                ),
            )
            .groupby(["period_start", "period_end"], dropna=False)[
                [
                    "tdc_principal_cash_paid_to_du_bil",
                    "tdc_principal_redeemed_to_du_bil",
                    "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
                    "tdc_principal_cash_paid_to_du_mmf_bil",
                    "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
                ]
            ]
            .sum()
        )
        summary_indexed = summary.set_index(["period_start", "period_end"], drop=False)
        checks = {
            "gross_principal_cash_paid_to_du_bil": "tdc_principal_cash_paid_to_du_bil",
            "tdc_debt_service_principal_to_du_bil": "tdc_principal_redeemed_to_du_bil",
            "gross_principal_cash_paid_to_du_domestic_nonbank_bil": "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
            "gross_principal_cash_paid_to_du_mmf_bil": "tdc_principal_cash_paid_to_du_mmf_bil",
            "gross_principal_cash_paid_to_du_mmf_plumbing_bil": "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
        }
        for summary_col, principal_col in checks.items():
            delta = principal_grouped[principal_col].reindex(summary_indexed.index, fill_value=0.0) - _numeric(
                summary_indexed, summary_col
            )
            if delta.abs().max() > 1e-7:
                raise VerificationError(f"TDC principal bridge does not match summary field: {summary_col}")


def _verify_accounting_journal_outputs(
    results: pd.DataFrame,
    holder_stocks: pd.DataFrame,
    journal: pd.DataFrame,
    closure: pd.DataFrame,
) -> None:
    journal_columns = (
        "period_start",
        "period_end",
        "journal_id",
        "event_type",
        "leg_type",
        "holder_sector",
        "instrument_type",
        "accounting_basis",
        "face_stock_change_bil",
        "adjusted_principal_change_bil",
        "route_face_stock_change_bil",
        "route_adjusted_principal_change_bil",
        "treasury_cash_change_bil",
        "reserve_change_bil",
        "deposit_change_bil",
    )
    closure_columns = (
        "period_start",
        "period_end",
        "opening_face_stock_bil",
        "journal_face_stock_change_bil",
        "closing_face_stock_bil",
        "face_stock_closure_error_bil",
        "opening_adjusted_principal_stock_bil",
        "journal_adjusted_principal_change_bil",
        "closing_adjusted_principal_stock_bil",
        "adjusted_principal_closure_error_bil",
        "opening_treasury_cash_bil",
        "journal_treasury_cash_change_bil",
        "closing_treasury_cash_bil",
        "treasury_cash_closure_error_bil",
        "journal_reserve_change_bil",
        "reported_reserve_change_bil",
        "reserve_closure_error_bil",
        "journal_deposit_change_bil",
        "reported_deposit_change_bil",
        "deposit_closure_error_bil",
        "holder_debt_total_bil",
        "instrument_debt_total_bil",
        "aggregate_debt_bil",
        "holder_total_error_bil",
        "instrument_total_error_bil",
        "closure_basis",
        "unexplained_residual_bil",
    )
    _require_columns(journal, journal_columns, label="tdcsim_accounting_journal")
    _require_columns(closure, closure_columns, label="tdcsim_accounting_closure")
    if journal.empty and closure.empty:
        result_dates = pd.to_datetime(
            results.get("Date", pd.Series(dtype=object)),
            errors="coerce",
        )
        if result_dates.notna().sum() > 1:
            raise VerificationError(
                "multi-period run is missing accounting journal and closure evidence"
            )
        return
    if journal.empty or closure.empty:
        raise VerificationError(
            "accounting journal and independent closure must both contain evidence"
        )
    journal_ids = journal["journal_id"].fillna("").astype(str)
    if journal_ids.eq("").any() or journal_ids.duplicated().any():
        raise VerificationError("accounting journal IDs must be nonempty and unique")
    for column in ("event_type", "leg_type", "accounting_basis"):
        if journal[column].fillna("").astype(str).str.strip().eq("").any():
            raise VerificationError(f"accounting journal has blank {column}")
    amount_columns = [
        "face_stock_change_bil",
        "adjusted_principal_change_bil",
        "route_face_stock_change_bil",
        "route_adjusted_principal_change_bil",
        "treasury_cash_change_bil",
        "reserve_change_bil",
        "deposit_change_bil",
    ]
    numeric_journal = journal.copy()
    for column in amount_columns:
        numeric_journal[column] = _strict_numeric(
            numeric_journal,
            column,
            label="tdcsim_accounting_journal",
        )
    closure_numeric_columns = [
        column
        for column in closure_columns
        if column
        not in {"period_start", "period_end", "closure_basis"}
    ]
    for column in closure_numeric_columns:
        _strict_numeric(
            closure,
            column,
            label="tdcsim_accounting_closure",
        )
    result_dates = pd.to_datetime(
        results.get("Date", pd.Series(dtype=object)),
        errors="coerce",
    )
    if (
        len(result_dates) <= 1
        or result_dates.isna().any()
        or result_dates.duplicated().any()
    ):
        raise VerificationError(
            "results must provide unique finite dates for accounting periods"
        )
    ordered_dates = sorted(pd.Timestamp(value).normalize() for value in result_dates)
    result_periods = {
        (str(start.date()), str(end.date()))
        for start, end in zip(ordered_dates, ordered_dates[1:])
    }

    def period_set(frame: pd.DataFrame, *, label: str) -> set[tuple[str, str]]:
        starts = pd.to_datetime(frame["period_start"], errors="coerce")
        ends = pd.to_datetime(frame["period_end"], errors="coerce")
        if starts.isna().any() or ends.isna().any() or (ends <= starts).any():
            raise VerificationError(
                f"{label} has malformed or non-increasing accounting periods"
            )
        return {
            (
                str(pd.Timestamp(start).normalize().date()),
                str(pd.Timestamp(end).normalize().date()),
            )
            for start, end in zip(starts, ends)
        }

    journal_periods = period_set(
        numeric_journal,
        label="tdcsim_accounting_journal",
    )
    closure_periods = period_set(
        closure,
        label="tdcsim_accounting_closure",
    )
    if (
        journal_periods != result_periods
        or closure_periods != result_periods
        or len(closure) != len(result_periods)
    ):
        raise VerificationError(
            "accounting journal, closure, and results period sets must match exactly"
        )
    if numeric_journal[amount_columns].abs().max(axis=1).le(1e-12).any():
        raise VerificationError("accounting journal contains a zero-value leg")
    if _numeric(closure, "unexplained_residual_bil").abs().max() > 1e-12:
        raise VerificationError(
            "unexplained residual cannot enter the accounting closure"
        )
    if set(closure["closure_basis"].astype(str).unique()) != {
        "independent_opening_and_closing_state_snapshots"
    }:
        raise VerificationError("accounting closure has unexpected closure_basis")
    key_columns = ["period_start", "period_end"]
    journal_grouped = numeric_journal.groupby(
        key_columns, dropna=False
    )[
        [
            "face_stock_change_bil",
            "adjusted_principal_change_bil",
            "treasury_cash_change_bil",
            "reserve_change_bil",
            "deposit_change_bil",
        ]
    ].sum()
    closure_indexed = closure.set_index(key_columns, drop=False)
    journal_checks = {
        "face_stock_change_bil": "journal_face_stock_change_bil",
        "adjusted_principal_change_bil": "journal_adjusted_principal_change_bil",
        "treasury_cash_change_bil": "journal_treasury_cash_change_bil",
        "reserve_change_bil": "journal_reserve_change_bil",
        "deposit_change_bil": "journal_deposit_change_bil",
    }
    for journal_column, closure_column in journal_checks.items():
        actual = journal_grouped[journal_column].reindex(
            closure_indexed.index, fill_value=0.0
        )
        if (actual - _numeric(closure_indexed, closure_column)).abs().max() > 1e-7:
            raise VerificationError(
                f"accounting closure does not match journal: {journal_column}"
            )
    stored_error_checks = {
        "face_stock_closure_error_bil": (
            _numeric(closure, "closing_face_stock_bil")
            - _numeric(closure, "opening_face_stock_bil")
            - _numeric(closure, "journal_face_stock_change_bil")
        ),
        "adjusted_principal_closure_error_bil": (
            _numeric(closure, "closing_adjusted_principal_stock_bil")
            - _numeric(closure, "opening_adjusted_principal_stock_bil")
            - _numeric(closure, "journal_adjusted_principal_change_bil")
        ),
        "treasury_cash_closure_error_bil": (
            _numeric(closure, "closing_treasury_cash_bil")
            - _numeric(closure, "opening_treasury_cash_bil")
            - _numeric(closure, "journal_treasury_cash_change_bil")
        ),
        "reserve_closure_error_bil": (
            _numeric(closure, "reported_reserve_change_bil")
            - _numeric(closure, "journal_reserve_change_bil")
        ),
        "deposit_closure_error_bil": (
            _numeric(closure, "reported_deposit_change_bil")
            - _numeric(closure, "journal_deposit_change_bil")
        ),
        "holder_total_error_bil": (
            _numeric(closure, "holder_debt_total_bil")
            - _numeric(closure, "aggregate_debt_bil")
        ),
        "instrument_total_error_bil": (
            _numeric(closure, "instrument_debt_total_bil")
            - _numeric(closure, "aggregate_debt_bil")
        ),
    }
    for error_column, recomputed in stored_error_checks.items():
        if recomputed.abs().max() > 1e-7:
            raise VerificationError(f"accounting identity failed: {error_column}")
        if (_numeric(closure, error_column) - recomputed).abs().max() > 1e-7:
            raise VerificationError(
                f"accounting closure error is stale: {error_column}"
            )
    _verify_accounting_closure_snapshots(results, holder_stocks, closure)


_ROUTE_KEY_COLUMNS = (
    "period_start",
    "period_end",
    "route_holder_sector",
    "route_holder_subsector",
    "instrument_type",
    "maturity_bucket",
    "debt_scope",
)

_ROUTE_CLOSURE_NUMERIC_COLUMNS = (
    "opening_route_stock_bil",
    "route_face_issued_bil",
    "route_face_redeemed_bil",
    "route_journal_face_change_bil",
    "route_journal_adjusted_principal_change_bil",
    "route_stock_residual_or_indexation_bil",
    "closing_route_stock_bil",
    "closure_identity_error_bil",
)


def _strict_route_stock_rows(frame: pd.DataFrame) -> pd.DataFrame:
    checked = frame.copy()
    dates = pd.to_datetime(checked["date"], errors="coerce")
    if dates.isna().any():
        raise VerificationError(
            "tdcsim_tdc_principal_route_stocks has malformed dates"
        )
    checked["date"] = dates.map(lambda value: str(pd.Timestamp(value).date()))
    for column in (
        "route_debt_held_bil",
        "route_face_stock_bil",
        "route_adjusted_principal_stock_bil",
    ):
        checked[column] = _strict_numeric(
            checked,
            column,
            label="tdcsim_tdc_principal_route_stocks",
        )
    key_columns = (
        "date",
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_scope",
    )
    for column in key_columns[1:]:
        checked[column] = checked[column].fillna("").astype(str)
    if checked.duplicated(list(key_columns)).any():
        raise VerificationError(
            "tdcsim_tdc_principal_route_stocks has duplicate route keys"
        )
    return checked


def _strict_route_closure_rows(
    frame: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    checked = frame.copy()
    starts = pd.to_datetime(checked["period_start"], errors="coerce")
    ends = pd.to_datetime(checked["period_end"], errors="coerce")
    if starts.isna().any() or ends.isna().any() or (ends <= starts).any():
        raise VerificationError(f"{label} has malformed or non-increasing periods")
    checked["period_start"] = starts.map(
        lambda value: str(pd.Timestamp(value).date())
    )
    checked["period_end"] = ends.map(
        lambda value: str(pd.Timestamp(value).date())
    )
    for column in _ROUTE_KEY_COLUMNS[2:]:
        checked[column] = checked[column].fillna("").astype(str)
    for column in _ROUTE_CLOSURE_NUMERIC_COLUMNS:
        checked[column] = _strict_numeric(checked, column, label=label)
    if checked.duplicated(list(_ROUTE_KEY_COLUMNS)).any():
        raise VerificationError(f"{label} has duplicate route-period keys")
    return checked


def _compare_stored_route_closure(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> None:
    stored_indexed = stored.set_index(list(_ROUTE_KEY_COLUMNS)).sort_index()
    recomputed_indexed = recomputed.set_index(
        list(_ROUTE_KEY_COLUMNS)
    ).sort_index()
    if not stored_indexed.index.equals(recomputed_indexed.index):
        raise VerificationError(
            "stored and recomputed route closure key sets do not match"
        )
    for column in _ROUTE_CLOSURE_NUMERIC_COLUMNS:
        if (
            stored_indexed[column] - recomputed_indexed[column]
        ).abs().max() > 1e-7:
            raise VerificationError(
                f"stored route closure disagrees with recomputation: {column}"
            )
    for column in ("route_stock_basis", "residual_basis"):
        stored_values = stored_indexed[column].fillna("").astype(str)
        recomputed_values = recomputed_indexed[column].fillna("").astype(str)
        if not stored_values.equals(recomputed_values):
            raise VerificationError(
                f"stored route closure disagrees with recomputation: {column}"
            )


def _verify_accounting_closure_snapshots(
    results: pd.DataFrame,
    holder_stocks: pd.DataFrame,
    closure: pd.DataFrame,
) -> None:
    _require_columns(
        results,
        ("Date", "TGA", "Reserves", "TDC_Level", "TotalDebt_Agg"),
        label="results",
    )
    _require_columns(
        holder_stocks,
        (
            "date",
            "debt_scope",
            "holder_sector",
            "holder_subsector",
            "instrument_type",
            "maturity_bucket",
            "debt_held_bil",
            "face_stock_bil",
            "adjusted_principal_stock_bil",
        ),
        label="tdcsim_holder_stocks",
    )
    result_rows = results.copy()
    result_rows["_date"] = pd.to_datetime(result_rows["Date"], errors="coerce")
    result_rows = result_rows[result_rows["_date"].notna()].set_index("_date")
    stocks = holder_stocks.copy()
    for column in (
        "debt_held_bil",
        "face_stock_bil",
        "adjusted_principal_stock_bil",
    ):
        stocks[column] = _strict_numeric(
            stocks,
            column,
            label="tdcsim_holder_stocks",
        )
    for column in ("TGA", "Reserves", "TDC_Level", "TotalDebt_Agg"):
        result_rows[column] = _strict_numeric(
            result_rows,
            column,
            label="results",
        )
    stocks["_date"] = pd.to_datetime(stocks["date"], errors="coerce")
    stocks = stocks[
        stocks["_date"].notna()
        & stocks["debt_scope"].astype(str).eq("all_active_treasury")
    ]
    for _, row in closure.iterrows():
        opening_date = pd.Timestamp(row["period_start"])
        closing_date = pd.Timestamp(row["period_end"])
        if opening_date not in result_rows.index or closing_date not in result_rows.index:
            raise VerificationError(
                "accounting closure period is absent from results snapshots"
            )
        opening_result = result_rows.loc[opening_date]
        closing_result = result_rows.loc[closing_date]
        if isinstance(opening_result, pd.DataFrame) or isinstance(
            closing_result, pd.DataFrame
        ):
            raise VerificationError("results contain duplicate accounting snapshot dates")
        opening_stocks = stocks[stocks["_date"].eq(opening_date)]
        closing_stocks = stocks[stocks["_date"].eq(closing_date)]

        def stock_sum(frame: pd.DataFrame, column: str) -> float:
            return float(frame[column].sum())

        holder_total = float(
            closing_stocks.groupby(
                ["holder_sector", "holder_subsector"],
                dropna=False,
            )["debt_held_bil"].sum().sum()
        )
        instrument_total = float(
            closing_stocks.groupby(
                ["instrument_type", "maturity_bucket"],
                dropna=False,
            )["debt_held_bil"].sum().sum()
        )

        expected = {
            "opening_face_stock_bil": stock_sum(
                opening_stocks, "face_stock_bil"
            ),
            "closing_face_stock_bil": stock_sum(
                closing_stocks, "face_stock_bil"
            ),
            "opening_adjusted_principal_stock_bil": stock_sum(
                opening_stocks, "adjusted_principal_stock_bil"
            ),
            "closing_adjusted_principal_stock_bil": stock_sum(
                closing_stocks, "adjusted_principal_stock_bil"
            ),
            "opening_treasury_cash_bil": float(opening_result["TGA"]),
            "closing_treasury_cash_bil": float(closing_result["TGA"]),
            "reported_reserve_change_bil": float(closing_result["Reserves"])
            - float(opening_result["Reserves"]),
            "reported_deposit_change_bil": float(closing_result["TDC_Level"])
            - float(opening_result["TDC_Level"]),
            "holder_debt_total_bil": holder_total,
            "instrument_debt_total_bil": instrument_total,
            "aggregate_debt_bil": float(closing_result["TotalDebt_Agg"]),
        }
        for column, value in expected.items():
            observed = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(observed) or abs(float(observed) - value) > 1e-7:
                raise VerificationError(
                    "accounting closure disagrees with independent snapshot: "
                    f"{column}; closure={observed!r}, snapshot={value!r}"
                )


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


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise VerificationError(f"required numeric column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _strict_numeric(
    frame: pd.DataFrame,
    column: str,
    *,
    label: str,
) -> pd.Series:
    if column not in frame.columns:
        raise VerificationError(f"{label} missing required numeric column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | ~values.map(math.isfinite)
    if invalid.any():
        raise VerificationError(
            f"{label} has malformed or nonfinite numeric values: {column}"
        )
    return values.astype(float)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise VerificationError(f"required boolean column is missing: {column}")
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise VerificationError(f"required boolean column has invalid values: {column}")
    return normalized.eq("true")


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


__all__ = ["VerificationError", "verify_compiled_scenario", "verify_marginal_tdc_pair", "verify_scenario_run"]
