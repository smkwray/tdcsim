"""Fail-closed four-file exporter for the verified OPEN-04 campaign.

The exporter is deliberately downstream of simulation and campaign verification.
It reads only the compact, independently verified annual artifact from each
promoted run, projects the downstream-facing rows, and atomically promotes one
four-file directory.  It does not select scenarios, rebuild daily evidence, or
create a second maturity taxonomy.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from evaluated_nominal_curve import NOMINAL_EVALUATED_SHOCK_FILE
from tdc_shared import MARKETABLE_PREFERENCE_CATEGORIES

from ._json import (
    canonical_json_bytes,
    canonical_json_sha256,
    canonical_json_text,
    read_json,
    sha256_bytes,
    sha256_file,
    write_json,
)
from .bounded_output import ANNUAL_COLUMNS
from .compiler import ISSUANCE_MIX_FILE
from .open04_campaign import (
    OPEN04_AGGREGATION_CLOCK_ID,
    OPEN04_ATI_PDF_SHA256,
    OPEN04_BASELINE_RELEASE_COMMIT_SHA,
    OPEN04_BASELINE_REQUIREMENTS_LOCK_SHA256,
    OPEN04_CAMPAIGN_ROLES,
    OPEN04_CONTROLLER_RECEIPT_KEYS,
    OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    OPEN04_END_DATE,
    OPEN04_HOU_PDF_SHA256,
    OPEN04_ROLE_TO_PROVENANCE,
    OPEN04_START_DATE,
    Open04CampaignError,
    validate_open04_campaign_contract_mapping,
    validate_open04_campaign_post_receipt_mapping,
    validate_open04_scenario_contract,
)
from .output import TDC_AMOUNT_BASIS, TDC_HOLDER_SCOPE, TDC_OVERLAP_POLICY
from .verifier import VerificationError, verify_scenario_run


OPEN04_EXPORT_FILES = (
    "open04_tdc_paths_fy.csv",
    "open04_maturity_metrics_fy.csv",
    "scenario_input_contract.csv",
    "open04_producer_consumer_receipt.json",
)
OPEN04_ROLES = OPEN04_CAMPAIGN_ROLES
OPEN04_ROLE_CLAIM_LABELS = {
    role: str(OPEN04_ROLE_TO_PROVENANCE[role]["label"])
    for role in OPEN04_CAMPAIGN_ROLES
}
OPEN04_ROLE_DESIGN_LABELS = {
    "baseline": (
        "Baseline — common CBO debt reference and mechanical TGA-floor "
        "financing; baseline maturity/rates; fixed holder shares"
    ),
    "candidate_a": (
        "Low-cost / high-total-TDC — Candidate A shorter issuance + "
        "exogenous -25 bp long-end sensitivity + 1.00 pp Private→Banks "
        "shift for new 20-/30-year nominal bonds, solved/chosen to "
        "illustrate the requested sign combination"
    ),
    "candidate_b": (
        "Low-total-TDC / high-cost — Candidate B longer issuance + "
        "exogenous +25 bp long-end sensitivity; fixed holder shares"
    ),
}

EXPORT_RECEIPT_SCHEMA_VERSION = "tdcsim_open04_producer_consumer_receipt_v1"
TDC_PATH_SCHEMA_VERSION = "tdcsim_open04_tdc_paths_fy_v2"
MATURITY_SCHEMA_VERSION = "tdcsim_open04_maturity_metrics_fy_v1"
INPUT_CONTRACT_SCHEMA_VERSION = "tdcsim_open04_scenario_input_contract_v2"

_BASELINE_RELEASE_COMMIT_SHA = OPEN04_BASELINE_RELEASE_COMMIT_SHA
_BASELINE_REQUIREMENTS_LOCK_SHA256 = (
    OPEN04_BASELINE_REQUIREMENTS_LOCK_SHA256
)
_ATI_PRIMARY_PDF_SHA256 = OPEN04_ATI_PDF_SHA256
_HOU_PRIMARY_PDF_SHA256 = OPEN04_HOU_PDF_SHA256
_START_DATE = OPEN04_START_DATE
_END_DATE = OPEN04_END_DATE
_CLOCK_ID = OPEN04_AGGREGATION_CLOCK_ID
_CLAIM_STATUS = "scenario_not_forecast_not_causal_not_optimization"
_CUMULATIVE_BASIS = "since_simulation_origin"
_FINANCING_COST_BASIS = (
    "nominal_model_cost_incurred_within_simulation_horizon"
)
_FINANCING_COST_UNITS = "billions_of_nominal_dollars"
_ANNUAL_SOURCE_FILE = "tdcsim_annual_economic_summary.csv.gz"
_TOLERANCE = 1e-7

TDC_PATH_COLUMNS = (
    "schema_version",
    "clock_id",
    "campaign_id",
    "scenario_role",
    "scenario_id",
    "source_scenario_title",
    "claim_strength_label",
    "claim_status",
    "fiscal_year",
    "period_label",
    "period_status",
    "coverage_start_exclusive",
    "coverage_end_inclusive",
    "coverage_interval",
    "coverage_days",
    "cumulative_origin_date",
    "tdc_change_bil",
    "overlap_cashflow_bil",
    "tdc_change_ex_overlap_bil",
    "cumulative_tdc_change_bil",
    "cumulative_overlap_cashflow_bil",
    "cumulative_tdc_change_ex_overlap_bil",
    "interest_outlay_bil",
    "issue_discount_cost_bil",
    "nonmarketable_interest_capitalized_bil",
    "tips_inflation_accretion_bil",
    "modeled_financing_cost_bil",
    "cumulative_interest_outlay_bil",
    "cumulative_issue_discount_cost_bil",
    "cumulative_nonmarketable_interest_capitalized_bil",
    "cumulative_tips_inflation_accretion_bil",
    "cumulative_modeled_financing_cost_bil",
    "modeled_financing_cost_basis",
    "modeled_financing_cost_units",
    "cumulative_basis",
    "tdc_amount_basis",
    "holder_allocation_scope",
    "overlap_policy",
)

MATURITY_COLUMNS = (
    "schema_version",
    "clock_id",
    "campaign_id",
    "scenario_role",
    "scenario_id",
    "source_scenario_title",
    "claim_strength_label",
    "claim_status",
    "fiscal_year",
    "period_label",
    "period_status",
    "coverage_start_exclusive",
    "coverage_end_inclusive",
    "coverage_interval",
    "coverage_days",
    "issuance_scope",
    "issuance_face_bil",
    "issuance_metric_status",
    "new_issuance_wam_years",
    "new_issuance_bill_share",
    "new_issuance_short_maturity_share",
    "new_issuance_weight_basis",
    "outstanding_scope",
    "outstanding_snapshot_date",
    "snapshot_status",
    "outstanding_controlled_wam_years",
    "outstanding_controlled_bill_share",
    "outstanding_controlled_short_maturity_share",
    "outstanding_weight_basis",
    "short_maturity_cutoff_years",
    "maturity_authority",
)

SCENARIO_INPUT_COLUMNS = (
    "contract_version",
    "campaign_id",
    "campaign_contract_sha256",
    "campaign_verification_receipt_sha256",
    "scenario_role",
    "scenario_filename",
    "scenario_id",
    "source_scenario_title",
    "scenario_design_label",
    "claim_strength_label",
    "claim_status",
    "scenario_source_sha256",
    "scenario_canonical_sha256",
    "run_scenario_copy_sha256",
    "source_run_canonical_match_status",
    "run_id",
    "run_manifest_sha256",
    "controller_completion_receipt_sha256",
    "terminal_summary_sha256",
    "source_output_manifest_sha256",
    "event_schema_version",
    "event_count",
    "event_root_sha256",
    "compiled_inputs_digest",
    "baseline_package_id",
    "baseline_package_sha256",
    "baseline_manifest_sha256",
    "baseline_attestation_sha256",
    "baseline_release_commit_sha",
    "baseline_requirements_lock_sha256",
    "simulation_start_date",
    "simulation_end_date",
    "simulation_frequency",
    "funding_closure_mode",
    "cbo_debt_path_status",
    "financing_closure",
    "cash_closure_target_bil",
    "validation_floor_bil",
    "financing_claim_limits",
    "override_keys_json",
    "coupling_contract_json",
    "issuance_mix_mode",
    "tips_share",
    "frn_share",
    "fixed_remainder_bills_share",
    "fixed_remainder_notes_share",
    "fixed_remainder_bonds_share",
    "maturity_distributions_json",
    "negative_issuance_action",
    "mmf_deposit_pass_through",
    "mmf_deposit_pass_through_status",
    "fiscal_incidence_policy_id",
    "fiscal_incidence_basis",
    "fiscal_incidence_du_share",
    "short_maturity_cutoff_years",
    "implied_new_issuance_wam_years",
    "implied_new_issuance_bill_share",
    "implied_new_issuance_short_maturity_share",
    "nominal_yield_curve_override_status",
    "curve_override_mode",
    "curve_application",
    "curve_interpolation",
    "curve_lower_endpoint",
    "curve_upper_endpoint",
    "curve_time_profile",
    "curve_compounding",
    "curve_key_rates_json",
    "signed_10y_shock_bp",
    "compiled_curve_delta_digest",
    "term_premium_channel_status",
    "calibration_source_label",
    "calibration_status",
    "source_claim_boundary",
    "ati_primary_pdf_sha256",
    "hou_primary_pdf_sha256",
    "calibration_worksheet_sha256",
    "calibration_worksheet_canonical_sha256",
    "calibration_scope",
    "calibration_central_bp",
    "calibration_source_low_bp",
    "calibration_source_high_bp",
    "source_range_status",
    "persistence_status",
    "hou_use_status",
    "holder_override_status",
    "holder_profile_sha256",
    "holder_trading_status",
    "private_mmf_route_sha256",
    "mmf_pass_through_source_sha256",
    "input_freeze_sha256",
    "canonical_issuance_mix_sha256",
    "issuance_mix_file_sha256",
    "economic_changed_paths_json",
    "physical_changed_inputs_json",
    "fixed_input_comparison_status",
    "fixed_input_records_sha256",
    "curve_date_set_sha256",
    "tenor_set_sha256",
    "scenario_provenance_kind",
    "scenario_provenance_as_of_date",
    "provenance_scope_note",
)


class Open04ExportError(ValueError):
    """Raised when the four-file export cannot be proven from verified inputs."""


@dataclass(frozen=True)
class Open04ExportResult:
    """Paths and immutable receipt for one promoted thin export."""

    producer_output_dir: Path
    consumer_output_dir: Path
    tdc_paths: Path
    maturity_metrics: Path
    scenario_input_contract: Path
    producer_consumer_receipt: Path
    consumer_tdc_paths: Path
    consumer_maturity_metrics: Path
    consumer_scenario_input_contract: Path
    consumer_producer_consumer_receipt: Path
    receipt: Mapping[str, Any]

    @property
    def output_dir(self) -> Path:
        """Backward-readable alias for the producer-side directory."""

        return self.producer_output_dir


@dataclass(frozen=True)
class _Document:
    data: dict[str, Any]
    canonical_sha256: str
    source_sha256: str
    source_bytes: int
    source_name: str
    source_kind: str


@dataclass(frozen=True)
class _RunProjection:
    role: str
    root: Path
    manifest: dict[str, Any]
    scenario: dict[str, Any]
    annual_rows: tuple[dict[str, Any], ...]
    issuance_mix: dict[str, Any]
    input_row: dict[str, Any]
    source_records: tuple[dict[str, Any], ...]
    controller_receipt: dict[str, Any]
    overall_wam: float


def export_open04_thin_package(
    run_roots: Mapping[str, str | Path],
    campaign_contract: Mapping[str, Any] | str | Path,
    campaign_verification_receipt: Mapping[str, Any] | str | Path,
    producer_output_dir: str | Path,
    consumer_output_dir: str | Path,
    *,
    consumer_project: str,
    exporter_code_commit_sha: str | None = None,
    verified_run_results: Mapping[str, Mapping[str, Any]] | None = None,
) -> Open04ExportResult:
    """Project one verified campaign into matching four-file destinations.

    ``campaign_contract`` is the pre-output freeze.  The separate verification
    receipt proves that the three sequential runs and all predeclared gates
    passed.  Both may be supplied as JSON paths or already-loaded mappings, but
    their canonical JSON identities are checked identically.

    Both destinations must be absent, distinct, and non-overlapping.  The
    exporter stages and validates byte-identical producer and consumer copies,
    promotes each directory by an atomic same-parent rename, and removes its
    newly created first destination if the second promotion fails.  The JSON
    receipt deliberately does not hash itself; identical receipt bytes are
    verified before either directory is promoted.
    """

    roots, campaign_root = _normalize_run_roots(run_roots)
    contract_doc = _load_document(campaign_contract, label="campaign contract")
    contract = _validate_campaign_contract(
        contract_doc.data,
        roots=roots,
        campaign_root=campaign_root,
        frozen_sha256=contract_doc.source_sha256,
    )
    verification_doc = _load_document(
        campaign_verification_receipt,
        label="campaign verification receipt",
    )
    verification = _validate_campaign_receipt(
        verification_doc.data,
        roots=roots,
        campaign_root=campaign_root,
        contract=contract,
        contract_sha256=contract_doc.source_sha256,
    )
    calibration = _load_calibration_worksheet(
        campaign_root,
        contract=contract,
    )
    consumer = _consumer_expectation(consumer_project)
    actual_exporter_source_sha256 = sha256_file(Path(__file__))
    if exporter_code_commit_sha is None:
        common = _mapping(
            verification["common_identity"],
            label="campaign receipt common identity",
        )
        if (
            actual_exporter_source_sha256
            != common["open04_exporter_source_sha256"]
        ):
            raise Open04ExportError(
                "post-run exporter differs from the simulation producer; "
                "declare its committed adapter identity"
            )
        exporter_identity = {
            "code_commit_sha": common["code_commit_sha"],
            "source_sha256": actual_exporter_source_sha256,
            "identity_mode": "simulation_producer",
        }
    else:
        exporter_identity = {
            "code_commit_sha": _require_git_sha(
                exporter_code_commit_sha,
                label="post-run exporter code commit",
            ),
            "source_sha256": actual_exporter_source_sha256,
            "identity_mode": "committed_post_run_adapter",
        }
    producer_out, consumer_out = _normalize_destinations(
        producer_output_dir,
        consumer_output_dir,
        protected_roots=(campaign_root, *roots.values()),
    )

    cached_verification: dict[str, Mapping[str, Any]] | None = None
    if verified_run_results is not None:
        if set(verified_run_results) != set(OPEN04_ROLES):
            raise Open04ExportError(
                "cached run verification must cover exactly the OPEN-04 roles"
            )
        cached_verification = {}
        for role in OPEN04_ROLES:
            item = _mapping(
                verified_run_results[role],
                label=f"cached run verification {role}",
            )
            if (
                item.get("status") != "pass"
                or item.get("verification_grade") != "bounded_replay_v1"
            ):
                raise Open04ExportError(
                    f"cached run verification {role} did not pass"
                )
            cached_verification[role] = item

    projections = tuple(
        _load_run_projection(
            role,
            roots[role],
            campaign_root=campaign_root,
            contract=contract,
            verification=verification,
            campaign_contract_sha256=contract_doc.source_sha256,
            campaign_verification_receipt_sha256=(
                verification_doc.source_sha256
            ),
            calibration=calibration,
            verified_run_result=(
                None
                if cached_verification is None
                else cached_verification[role]
            ),
        )
        for role in OPEN04_ROLES
    )
    _validate_cross_run_results(
        projections,
        verification=verification,
    )

    campaign_id = str(contract["campaign_id"])
    tdc_rows = [
        _tdc_export_row(projection, row, campaign_id=campaign_id)
        for projection in projections
        for row in projection.annual_rows
    ]
    maturity_rows = [
        _maturity_export_row(projection, row, campaign_id=campaign_id)
        for projection in projections
        for row in projection.annual_rows
    ]
    input_rows = [projection.input_row for projection in projections]

    for destination in (producer_out, consumer_out):
        destination.parent.mkdir(parents=True, exist_ok=True)
    producer_staging = _staging_path(producer_out)
    consumer_staging = _staging_path(consumer_out)
    promoted: list[Path] = []
    try:
        for staging in (producer_staging, consumer_staging):
            if staging.exists():
                raise Open04ExportError(
                    "OPEN-04 export staging directory already exists: "
                    f"{staging}"
                )
            staging.mkdir()
        tdc_path = producer_staging / OPEN04_EXPORT_FILES[0]
        maturity_path = producer_staging / OPEN04_EXPORT_FILES[1]
        input_path = producer_staging / OPEN04_EXPORT_FILES[2]
        receipt_path = producer_staging / OPEN04_EXPORT_FILES[3]
        _write_csv(tdc_path, TDC_PATH_COLUMNS, tdc_rows)
        _write_csv(maturity_path, MATURITY_COLUMNS, maturity_rows)
        _write_csv(input_path, SCENARIO_INPUT_COLUMNS, input_rows)
        for source in (tdc_path, maturity_path, input_path):
            shutil.copyfile(source, consumer_staging / source.name)

        thin_artifacts = []
        for producer_path, row_count in (
            (tdc_path, len(tdc_rows)),
            (maturity_path, len(maturity_rows)),
            (input_path, len(input_rows)),
        ):
            consumer_path = consumer_staging / producer_path.name
            thin_artifacts.append(
                _dual_artifact_record(
                    producer_path,
                    consumer_path,
                    rows=row_count,
                )
            )
        receipt = _build_export_receipt(
            contract_doc=contract_doc,
            verification_doc=verification_doc,
            contract=contract,
            verification=verification,
            projections=projections,
            thin_artifacts=thin_artifacts,
            consumer=consumer,
            exporter_identity=exporter_identity,
        )
        write_json(receipt_path, receipt)
        consumer_receipt_path = (
            consumer_staging / OPEN04_EXPORT_FILES[3]
        )
        write_json(consumer_receipt_path, receipt)
        _require_identical_files(
            receipt_path,
            consumer_receipt_path,
            label="producer/consumer receipt copies",
        )
        _require_exact_output_boundary(producer_staging)
        _require_exact_output_boundary(consumer_staging)
        _fsync_files(producer_staging)
        _fsync_files(consumer_staging)
        consumer_staging.rename(consumer_out)
        promoted.append(consumer_out)
        producer_staging.rename(producer_out)
        promoted.append(producer_out)
        _require_promoted_receipt_authority(
            producer_out,
            consumer_out,
        )
    except Exception:
        _remove_owned_staging(producer_staging, output=producer_out)
        _remove_owned_staging(consumer_staging, output=consumer_out)
        for destination in reversed(promoted):
            _remove_owned_output(destination)
        raise

    return Open04ExportResult(
        producer_output_dir=producer_out,
        consumer_output_dir=consumer_out,
        tdc_paths=producer_out / OPEN04_EXPORT_FILES[0],
        maturity_metrics=producer_out / OPEN04_EXPORT_FILES[1],
        scenario_input_contract=producer_out / OPEN04_EXPORT_FILES[2],
        producer_consumer_receipt=producer_out / OPEN04_EXPORT_FILES[3],
        consumer_tdc_paths=consumer_out / OPEN04_EXPORT_FILES[0],
        consumer_maturity_metrics=consumer_out / OPEN04_EXPORT_FILES[1],
        consumer_scenario_input_contract=(
            consumer_out / OPEN04_EXPORT_FILES[2]
        ),
        consumer_producer_consumer_receipt=(
            consumer_out / OPEN04_EXPORT_FILES[3]
        ),
        receipt=receipt,
    )


def _normalize_run_roots(
    run_roots: Mapping[str, str | Path],
) -> tuple[dict[str, Path], Path]:
    _require_exact_keys(run_roots, set(OPEN04_ROLES), label="run_roots")
    roots = {
        role: Path(run_roots[role]).expanduser().resolve()
        for role in OPEN04_ROLES
    }
    for role, root in roots.items():
        if not root.is_dir():
            raise Open04ExportError(
                f"{role} promoted run root is missing: {root}"
            )
    campaign_parents = {root.parent.parent for root in roots.values()}
    if len(campaign_parents) != 1:
        raise Open04ExportError(
            "OPEN-04 run roots must use isolated parents in one campaign root"
        )
    if len(set(roots.values())) != len(OPEN04_ROLES):
        raise Open04ExportError("OPEN-04 roles must identify three distinct run roots")
    return roots, next(iter(campaign_parents))


def _load_document(
    source: Mapping[str, Any] | str | Path,
    *,
    label: str,
) -> _Document:
    if isinstance(source, Mapping):
        data = dict(source)
        try:
            canonical_json_bytes(data)
            payload = (
                json.dumps(
                    data,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise Open04ExportError(
                f"{label} mapping is not canonical JSON"
            ) from exc
        return _Document(
            data=data,
            canonical_sha256=canonical_json_sha256(data),
            source_sha256=sha256_bytes(payload),
            source_bytes=len(payload),
            source_name=f"{label.replace(' ', '_')}.canonical.json",
            source_kind="canonical_mapping",
        )
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise Open04ExportError(f"{label} JSON file is missing: {path}")
    try:
        data = read_json(path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(f"{label} JSON is unreadable") from exc
    if not isinstance(data, dict):
        raise Open04ExportError(f"{label} must be a JSON object")
    try:
        canonical_sha = canonical_json_sha256(data)
    except (TypeError, ValueError) as exc:
        raise Open04ExportError(
            f"{label} cannot be represented as canonical JSON"
        ) from exc
    return _Document(
        data=data,
        canonical_sha256=canonical_sha,
        source_sha256=sha256_file(path),
        source_bytes=path.stat().st_size,
        source_name=path.name,
        source_kind="json_file",
    )


def _validate_campaign_contract(
    value: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    campaign_root: Path,
    frozen_sha256: str,
) -> dict[str, Any]:
    try:
        contract = validate_open04_campaign_contract_mapping(
            campaign_root,
            value,
            expected_contract_sha256=frozen_sha256,
        )
    except Open04CampaignError as exc:
        raise Open04ExportError(
            "frozen OPEN-04 campaign contract did not validate"
        ) from exc
    declared_roles = _mapping(
        contract["roles"], label="campaign contract roles"
    )
    for role in OPEN04_ROLES:
        declared = _mapping(
            declared_roles[role], label=f"campaign contract role {role}"
        )
        expected_root = (
            campaign_root / str(declared["run_relative_path"])
        ).resolve()
        if expected_root != roots[role]:
            raise Open04ExportError(
                f"{role} run root differs from the frozen campaign contract"
            )
    return contract


def _validate_campaign_receipt(
    value: Mapping[str, Any],
    *,
    roots: Mapping[str, Path],
    campaign_root: Path,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    try:
        receipt = validate_open04_campaign_post_receipt_mapping(
            campaign_root,
            value,
            contract=contract,
            expected_contract_sha256=contract_sha256,
        )
    except Open04CampaignError as exc:
        raise Open04ExportError(
            "OPEN-04 post-run campaign receipt did not validate"
        ) from exc
    declared_roles = _mapping(receipt["roles"], label="receipt roles")
    for role in OPEN04_ROLES:
        declared = _mapping(
            declared_roles[role], label=f"receipt role {role}"
        )
        expected_root = (
            campaign_root / str(declared["run_relative_path"])
        ).resolve()
        if expected_root != roots[role]:
            raise Open04ExportError(
                f"{role} receipt run root differs from the supplied run root"
            )
    return receipt


def _load_calibration_worksheet(
    campaign_root: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    reference = _exact_mapping(
        contract.get("calibration_worksheet"),
        {"relative_path", "sha256", "canonical_sha256"},
        label="campaign calibration worksheet reference",
    )
    relative = _safe_relative(
        reference["relative_path"],
        label="campaign calibration worksheet path",
    )
    if len(relative.parts) != 1:
        raise Open04ExportError(
            "campaign calibration worksheet must be a direct child"
        )
    path = _resolve_under(
        campaign_root,
        relative,
        label="campaign calibration worksheet",
    )
    if not path.is_file():
        raise Open04ExportError(
            "campaign calibration worksheet is missing"
        )
    if sha256_file(path) != _require_sha(
        reference["sha256"],
        label="campaign calibration worksheet SHA-256",
    ):
        raise Open04ExportError(
            "campaign calibration worksheet byte identity mismatch"
        )
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(
            "campaign calibration worksheet is unreadable"
        ) from exc
    worksheet = dict(
        _mapping(raw, label="campaign calibration worksheet")
    )
    if canonical_json_sha256(worksheet) != _require_sha(
        reference["canonical_sha256"],
        label="campaign calibration worksheet canonical SHA-256",
    ):
        raise Open04ExportError(
            "campaign calibration worksheet canonical identity mismatch"
        )
    expected_keys = {
        "schema_version",
        "campaign_id",
        "scenario_contract_id",
        "calibration_object",
        "central_10y_sensitivity_bp",
        "nonprobabilistic_source_range_bp",
        "ati_source",
        "hou_source",
        "application",
        "short_end_rule",
        "long_end_rule",
        "time_profile",
        "no_retuning_label",
        "fallback_label",
        "claim_boundary",
    }
    _require_exact_keys(
        worksheet,
        expected_keys,
        label="campaign calibration worksheet",
    )
    if (
        worksheet["campaign_id"] != contract["campaign_id"]
        or worksheet["scenario_contract_id"] != contract["contract_id"]
        or worksheet["calibration_object"]
        != "assumed_exogenous_10_year_nominal_yield_level_sensitivity"
        or worksheet["application"] != "post_baseline_evaluation"
        or worksheet["short_end_rule"]
        != "unchanged_at_or_below_2_years"
        or worksheet["long_end_rule"]
        != "log_tenor_ramp_to_10_year_then_flat"
        or worksheet["time_profile"] != "constant_across_curve_dates"
        or worksheet["no_retuning_label"] != "predeclared_no_retuning"
        or worksheet["fallback_label"]
        != "failed_gate_no_relabel_no_recalibration"
    ):
        raise Open04ExportError(
            "campaign calibration worksheet contract differs"
        )
    central = _finite(
        worksheet["central_10y_sensitivity_bp"],
        label="calibration central 10-year sensitivity",
    )
    source_range = _exact_mapping(
        worksheet["nonprobabilistic_source_range_bp"],
        {"lower", "upper", "interpretation"},
        label="calibration nonprobabilistic source range",
    )
    lower = _finite(
        source_range["lower"],
        label="calibration source range lower bound",
    )
    upper = _finite(
        source_range["upper"],
        label="calibration source range upper bound",
    )
    if (
        central != 25.0
        or lower != 14.0
        or upper != 40.0
        or source_range["interpretation"]
        != "cross_method_source_range_not_confidence_interval"
    ):
        raise Open04ExportError(
            "campaign calibration magnitudes/range differ"
        )
    ati = _exact_mapping(
        worksheet["ati_source"],
        {"pdf_sha256", "role"},
        label="calibration ATI source",
    )
    hou = _exact_mapping(
        worksheet["hou_source"],
        {"pdf_sha256", "role"},
        label="calibration Hou source",
    )
    if (
        ati["pdf_sha256"] != _ATI_PRIMARY_PDF_SHA256
        or ati["role"] != "paper_facing_calibration_context_only"
        or hou["pdf_sha256"] != _HOU_PRIMARY_PDF_SHA256
        or hou["role"] != "diagnostic_only_not_central_or_fallback"
    ):
        raise Open04ExportError(
            "campaign calibration source identity/role differs"
        )
    return {
        "worksheet": worksheet,
        "relative_path": relative.as_posix(),
        "sha256": reference["sha256"],
        "canonical_sha256": reference["canonical_sha256"],
        "central_bp": central,
        "lower_bp": lower,
        "upper_bp": upper,
    }


def _load_run_projection(
    role: str,
    root: Path,
    *,
    campaign_root: Path,
    contract: Mapping[str, Any],
    verification: Mapping[str, Any],
    campaign_contract_sha256: str,
    campaign_verification_receipt_sha256: str,
    calibration: Mapping[str, Any],
    verified_run_result: Mapping[str, Any] | None = None,
) -> _RunProjection:
    receipt_role = _mapping(
        _mapping(verification["roles"], label="receipt roles")[role],
        label=f"receipt role {role}",
    )
    common = _mapping(
        verification["common_identity"],
        label="receipt common identity",
    )

    manifest_path = root / "tdcsim_cbo_run_manifest.json"
    if not manifest_path.is_file():
        raise Open04ExportError(f"{role} promoted run manifest is missing")
    if sha256_file(manifest_path) != receipt_role["run_manifest_sha256"]:
        raise Open04ExportError(f"{role} run manifest SHA-256 mismatch")
    try:
        manifest_raw = read_json(manifest_path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(f"{role} run manifest is unreadable") from exc
    if not isinstance(manifest_raw, dict):
        raise Open04ExportError(f"{role} run manifest must be an object")
    manifest = manifest_raw
    _validate_run_manifest_identity(
        role,
        root,
        manifest,
        receipt_role=receipt_role,
        common=common,
    )
    controller_receipt, controller_record = (
        _load_controller_completion_receipt(
            role,
            campaign_root=campaign_root,
            contract=contract,
            receipt_role=receipt_role,
            campaign_contract_sha256=campaign_contract_sha256,
        )
    )
    if verified_run_result is None:
        try:
            verified = verify_scenario_run(root)
        except (OSError, ValueError, VerificationError) as exc:
            raise Open04ExportError(
                f"{role} promoted run does not independently verify"
            ) from exc
    else:
        verified = verified_run_result
    if not isinstance(verified, Mapping) or verified.get("status") != "pass":
        raise Open04ExportError(f"{role} promoted run verification did not pass")

    scenario_block = _mapping(
        manifest.get("scenario"),
        label=f"{role} run scenario block",
    )
    scenario_relative = _safe_relative(
        scenario_block.get("relative_path"),
        label=f"{role} run scenario path",
    )
    scenario_path = _resolve_under(
        root,
        scenario_relative,
        label=f"{role} run scenario",
    )
    if not scenario_path.is_file():
        raise Open04ExportError(f"{role} run scenario copy is missing")
    scenario_sha = sha256_file(scenario_path)
    if scenario_sha != receipt_role["run_scenario_copy_sha256"]:
        raise Open04ExportError(f"{role} run scenario copy SHA-256 mismatch")
    if scenario_sha != scenario_block.get("source_file_sha256"):
        raise Open04ExportError(
            f"{role} run manifest does not bind its scenario-copy bytes"
        )
    try:
        scenario_raw = read_json(scenario_path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(f"{role} scenario copy is unreadable") from exc
    if not isinstance(scenario_raw, dict):
        raise Open04ExportError(f"{role} scenario copy must be an object")
    scenario = scenario_raw
    _validate_scenario_perimeter(role, scenario)
    if (
        scenario.get("scenario_id") != receipt_role["scenario_id"]
        or canonical_json_sha256(scenario) != receipt_role["scenario_sha256"]
        or scenario_block.get("canonical_sha256")
        != receipt_role["scenario_sha256"]
    ):
        raise Open04ExportError(f"{role} scenario semantic identity mismatch")

    annual_path, annual_record = _bounded_annual_artifact(
        root,
        manifest,
    )
    annual_rows = tuple(_read_verified_annual_rows(annual_path, role=role))
    issuance_path, issuance_record = _compiled_input_artifact(
        root,
        manifest,
        ISSUANCE_MIX_FILE,
    )
    try:
        issuance_raw = read_json(issuance_path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(
            f"{role} compiled issuance mix is unreadable"
        ) from exc
    if not isinstance(issuance_raw, dict):
        raise Open04ExportError(
            f"{role} compiled issuance mix must be an object"
        )
    issuance_mix = _validate_issuance_mix(
        issuance_raw,
        role=role,
        expected_file_sha=receipt_role["issuance_mix_file_sha256"],
        actual_file_sha=sha256_file(issuance_path),
        expected_canonical_sha=receipt_role[
            "canonical_issuance_mix_sha256"
        ],
    )
    overall_wam = float(issuance_mix["weighted_average_maturity_years"])
    _assert_close(
        overall_wam,
        receipt_role["overall_new_issuance_wam_years"],
        label=f"{role} realized overall new-issuance WAM",
    )

    row_metadata = _exact_row_metadata(manifest, role=role)
    compiled_records = _compiled_input_records(manifest)
    holder_record = _required_compiled_record(
        compiled_records,
        "tdcsim_holder_profile_assumptions.csv",
        role=role,
    )
    runtime_record = _required_compiled_record(
        compiled_records,
        "tdcsim_runtime_assumptions.json",
        role=role,
    )
    private_mmf_route_sha = canonical_json_sha256(
        [
            {
                "logical_name": holder_record["logical_name"],
                "sha256": holder_record["sha256"],
            },
            {
                "logical_name": runtime_record["logical_name"],
                "sha256": runtime_record["sha256"],
            },
        ]
    )
    input_row = _scenario_input_row(
        role,
        manifest=manifest,
        scenario=scenario,
        issuance_mix=issuance_mix,
        row_metadata=row_metadata,
        receipt_role=receipt_role,
        common=common,
        contract=contract,
        campaign_contract_sha256=str(
            verification["campaign_contract_sha256"]
        ),
        campaign_verification_receipt_sha256=(
            campaign_verification_receipt_sha256
        ),
        calibration=calibration,
        holder_profile_sha256=str(holder_record["sha256"]),
        runtime_assumptions_sha256=str(runtime_record["sha256"]),
        private_mmf_route_sha256=private_mmf_route_sha,
    )

    output_records = _manifest_output_artifact_records(
        root,
        manifest,
        role=role,
        receipt_role=receipt_role,
    )
    source_records = [
        _artifact_record(
            manifest_path,
            logical_name="run_manifest",
            relative_path=(
                f"{receipt_role['run_relative_path']}/"
                "tdcsim_cbo_run_manifest.json"
            ),
        ),
        _artifact_record(
            scenario_path,
            logical_name="run_scenario_copy",
            relative_path=(
                f"{receipt_role['run_relative_path']}/"
                f"{scenario_relative.as_posix()}"
            ),
        ),
        _artifact_record(
            issuance_path,
            logical_name="compiled_issuance_mix",
            relative_path=(
                f"{receipt_role['run_relative_path']}/"
                f"{issuance_record['relative_path']}"
            ),
        ),
        controller_record,
        *output_records,
    ]
    if role != "baseline":
        sidecar_path, sidecar_record = _compiled_input_artifact(
            root,
            manifest,
            NOMINAL_EVALUATED_SHOCK_FILE,
        )
        if sha256_file(sidecar_path) != receipt_role["curve_sidecar_sha256"]:
            raise Open04ExportError(f"{role} curve sidecar SHA-256 mismatch")
        source_records.append(
            _artifact_record(
                sidecar_path,
                logical_name="compiled_evaluated_curve_sidecar",
                relative_path=(
                    f"{receipt_role['run_relative_path']}/"
                    f"{sidecar_record['relative_path']}"
                ),
            )
        )

    return _RunProjection(
        role=role,
        root=root,
        manifest=manifest,
        scenario=scenario,
        annual_rows=annual_rows,
        issuance_mix=issuance_mix,
        input_row=input_row,
        source_records=tuple(source_records),
        controller_receipt=controller_receipt,
        overall_wam=overall_wam,
    )


def _validate_run_manifest_identity(
    role: str,
    root: Path,
    manifest: Mapping[str, Any],
    *,
    receipt_role: Mapping[str, Any],
    common: Mapping[str, Any],
) -> None:
    if manifest.get("status") != "complete":
        raise Open04ExportError(f"{role} run manifest is not complete")
    if manifest.get("run_id") != receipt_role["run_id"]:
        raise Open04ExportError(f"{role} run ID differs from campaign receipt")
    if manifest.get("compiled_inputs_digest") != receipt_role[
        "compiled_inputs_digest"
    ]:
        raise Open04ExportError(
            f"{role} compiled-input digest differs from campaign receipt"
        )
    baseline = _mapping(manifest.get("baseline"), label=f"{role} baseline")
    expected_baseline = {
        "package_id": common["package_id"],
        "package_sha256": common["baseline_package_sha256"],
        "manifest_sha256": common["baseline_manifest_sha256"],
        "release_attestation_sha256": common["release_attestation_sha256"],
    }
    for field, expected in expected_baseline.items():
        if baseline.get(field) != expected:
            raise Open04ExportError(
                f"{role} run baseline differs on {field}"
            )
    simulation = _mapping(
        manifest.get("simulation"),
        label=f"{role} simulation",
    )
    if simulation != {
        "start_date": common["start_date"],
        "end_date": common["end_date"],
        "frequency": common["frequency"],
    }:
        raise Open04ExportError(f"{role} simulation horizon is not common")
    clock = _mapping(
        manifest.get("aggregation_clock"),
        label=f"{role} aggregation clock",
    )
    if clock.get("clock_id") != common["aggregation_clock_id"]:
        raise Open04ExportError(f"{role} aggregation clock is not common")

    bounded = _mapping(
        manifest.get("bounded_evidence"),
        label=f"{role} bounded evidence",
    )
    if bounded.get("invariant_status") != "pass":
        raise Open04ExportError(f"{role} bounded invariants did not pass")
    for manifest_field, receipt_field in (
        ("event_schema_version", "event_schema_version"),
        ("event_count", "event_count"),
        ("event_root_sha256", "event_root_sha256"),
    ):
        if bounded.get(manifest_field) != receipt_role[receipt_field]:
            raise Open04ExportError(
                f"{role} bounded evidence differs on {manifest_field}"
            )
    gib = 1024**3
    thresholds = _mapping(
        bounded.get("memory_thresholds"),
        label=f"{role} bounded memory thresholds",
    )
    expected_thresholds = {
        "minimum_available_bytes": 4 * gib,
        "acceptance_peak_rss_bytes": 6 * gib,
        "application_abort_rss_bytes": 8 * gib,
        "parent_graceful_stop_rss_bytes": 10 * gib,
        "parent_kill_rss_bytes": 12 * gib,
    }
    for field, expected in expected_thresholds.items():
        if thresholds.get(field) != expected:
            raise Open04ExportError(
                f"{role} bounded memory threshold differs on {field}"
            )
    worker_peak = _integer(
        bounded.get("peak_rss_bytes"),
        label=f"{role} worker peak RSS",
        minimum=0,
    )
    if worker_peak > expected_thresholds["acceptance_peak_rss_bytes"]:
        raise Open04ExportError(
            f"{role} worker peak RSS exceeds acceptance"
        )
    parent_watchdog = _mapping(
        manifest.get("parent_watchdog"),
        label=f"{role} parent watchdog evidence",
    )
    if (
        parent_watchdog.get("status") != "accepted"
        or parent_watchdog.get("action") != "completed"
        or parent_watchdog.get("child_returncode") != 0
        or parent_watchdog.get("acceptance_peak_rss_bytes")
        != expected_thresholds["acceptance_peak_rss_bytes"]
        or parent_watchdog.get("terminate_rss_bytes")
        != expected_thresholds["parent_graceful_stop_rss_bytes"]
        or parent_watchdog.get("kill_rss_bytes")
        != expected_thresholds["parent_kill_rss_bytes"]
    ):
        raise Open04ExportError(
            f"{role} parent watchdog evidence is not accepted"
        )
    parent_peak = _integer(
        parent_watchdog.get("effective_peak_rss_bytes"),
        label=f"{role} parent effective peak RSS",
        minimum=0,
    )
    if (
        parent_peak > expected_thresholds["acceptance_peak_rss_bytes"]
        or parent_peak
        != max(
            _integer(
                parent_watchdog.get("peak_rss_bytes"),
                label=f"{role} parent peak RSS",
                minimum=0,
            ),
            _integer(
                parent_watchdog.get("worker_peak_rss_bytes"),
                label=f"{role} parent-recorded worker peak RSS",
                minimum=0,
            ),
        )
    ):
        raise Open04ExportError(
            f"{role} parent watchdog RSS evidence differs"
        )
    if canonical_json_sha256(manifest.get("output_manifest")) != receipt_role[
        "source_output_manifest_sha256"
    ]:
        raise Open04ExportError(
            f"{role} source output manifest identity mismatch"
        )
    summary_path = root / "outputs" / "summary.json"
    if not summary_path.is_file():
        raise Open04ExportError(f"{role} terminal summary is missing")
    if sha256_file(summary_path) != receipt_role["terminal_summary_sha256"]:
        raise Open04ExportError(f"{role} terminal summary SHA-256 mismatch")
    _validate_manifest_code_identity(
        role,
        manifest.get("code_environment"),
        common=common,
    )
    execution = _mapping(
        manifest.get("execution_contract"),
        label=f"{role} execution contract",
    )
    marker = _mapping(
        manifest.get("open04_campaign"),
        label=f"{role} OPEN-04 marker",
    )
    if (
        execution.get("writer_claim_scope") != "output_parent"
        or execution.get("writer_claim_scope_id")
        != f"{marker['contract_id']}.{role}"
        or execution.get("one_scenario_per_worker") is not True
        or execution.get("process_pool_enabled") is not False
        or execution.get("parent_watchdog_required") is not True
    ):
        raise Open04ExportError(f"{role} execution contract is not bounded")


def _validate_manifest_code_identity(
    role: str,
    value: Any,
    *,
    common: Mapping[str, Any],
) -> None:
    code = _mapping(value, label=f"{role} code environment")
    direct_fields = (
        "code_commit_sha",
        "dirty_state",
        "requirements_lock_sha256",
        "wheel_sha256",
        "runner_source_sha256",
        "sim_engine_source_sha256",
        "bounded_output_source_sha256",
        "output_source_sha256",
        "verifier_source_sha256",
        "compiler_source_sha256",
        "contract_source_sha256",
        "manifest_source_sha256",
        "run_manifest_schema_sha256",
        "scenario_schema_sha256",
        "scenario_writer_source_sha256",
        "open04_exporter_source_sha256",
        "python_version",
        "package_name",
        "package_version",
        "distribution_file_digest",
        "runtime_identity_source",
    )
    for field in direct_fields:
        if code.get(field) != common[field]:
            raise Open04ExportError(
                f"{role} code environment differs on {field}"
            )
    producer = _mapping(
        code.get("producer_source_identity"),
        label=f"{role} producer source identity",
    )
    source_tree = _mapping(
        producer.get("source_tree"),
        label=f"{role} producer source tree",
    )
    raw_locks = _sequence(
        source_tree.get("dependency_lock_files"),
        label=f"{role} producer dependency locks",
    )
    locks: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_locks):
        item = _exact_mapping(
            raw,
            {"relative_path", "sha256", "bytes"},
            label=f"{role} producer dependency lock {index}",
        )
        relative = _safe_relative(
            item["relative_path"],
            label=f"{role} producer dependency lock {index} path",
        )
        if len(relative.parts) != 1:
            raise Open04ExportError(
                f"{role} producer dependency lock path is not direct"
            )
        locks.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": _require_sha(
                    item["sha256"],
                    label=f"{role} producer dependency lock {index} SHA-256",
                ),
                "bytes": _integer(
                    item["bytes"],
                    label=f"{role} producer dependency lock {index} bytes",
                    minimum=0,
                ),
            }
        )
    if [item["relative_path"] for item in locks] != [
        "uv.lock",
        "requirements.lock.txt",
    ]:
        raise Open04ExportError(
            f"{role} producer dependency lock set/order differs"
        )
    dependency_digest = canonical_json_sha256(locks)
    if (
        locks[0]["sha256"] != common["uv_lock_sha256"]
        or locks[1]["sha256"] != common["requirements_lock_sha256"]
        or dependency_digest != common["dependency_lock_set_sha256"]
        or source_tree.get("dependency_lock_set_sha256")
        != dependency_digest
        or source_tree.get("release_commit_sha")
        != common["code_commit_sha"]
        or producer.get("installed_archive_sha256")
        != common["wheel_sha256"]
    ):
        raise Open04ExportError(
            f"{role} producer source/dependency identity differs"
        )
    wheel_artifact = _mapping(
        code.get("wheel_artifact"),
        label=f"{role} retained wheel artifact",
    )
    if wheel_artifact.get("sha256") != common["wheel_artifact_sha256"]:
        raise Open04ExportError(
            f"{role} retained wheel artifact identity mismatch"
        )


def _load_controller_completion_receipt(
    role: str,
    *,
    campaign_root: Path,
    contract: Mapping[str, Any],
    receipt_role: Mapping[str, Any],
    campaign_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_role = _mapping(
        _mapping(contract["roles"], label="campaign contract roles")[role],
        label=f"campaign contract role {role}",
    )
    relative = _safe_relative(
        contract_role["controller_completion_receipt_relative_path"],
        label=f"{role} controller completion receipt path",
    )
    if len(relative.parts) != 1:
        raise Open04ExportError(
            f"{role} controller receipt must be a campaign-root child"
        )
    path = _resolve_under(
        campaign_root,
        relative,
        label=f"{role} controller completion receipt",
    )
    if not path.is_file():
        raise Open04ExportError(
            f"{role} controller completion receipt is missing"
        )
    if sha256_file(path) != receipt_role[
        "controller_completion_receipt_sha256"
    ]:
        raise Open04ExportError(
            f"{role} controller completion receipt SHA-256 mismatch"
        )
    try:
        raw = read_json(path)
    except (OSError, ValueError) as exc:
        raise Open04ExportError(
            f"{role} controller completion receipt is unreadable"
        ) from exc
    receipt = dict(
        _exact_mapping(
            raw,
            set(OPEN04_CONTROLLER_RECEIPT_KEYS),
            label=f"{role} controller completion receipt",
        )
    )
    controller_run_id = _require_nonblank(
        receipt["controller_run_id"],
        label=f"{role} controller run ID",
    )
    expected = {
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": campaign_contract_sha256,
        "role": role,
        "run_id": receipt_role["run_id"],
        "host": receipt_role["host"],
        "terminal_status": "completed",
        "terminal_summary_sha256": receipt_role[
            "terminal_summary_sha256"
        ],
        "run_manifest_sha256": receipt_role["run_manifest_sha256"],
        "controller_command_exit_code": 0,
        "controller_exit_code": 0,
        "preflight_conflicts": 0,
        "postrun_conflicts": 0,
        "controller_process_tree_drained": True,
    }
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            raise Open04ExportError(
                f"{role} controller receipt differs on {field}"
            )
    if receipt["schema_version"] == OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION:
        controller_expected = {
            "controller": "remote_controller",
            "placement": "auto",
            "controller_telemetry_status": "reported",
            "controller_telemetry_locator": (
                f"controller_run_id:{controller_run_id}"
            ),
        }
    elif (
        receipt["schema_version"]
        == OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION
    ):
        controller_expected = {
            "controller": "host_owned_task",
            "placement": "auto_selected_host",
            "controller_telemetry_status": "bounded_role_watchdogs",
            "controller_telemetry_locator": (
                "controller_summary_sha256:"
                f"{receipt['controller_summary_sha256']}"
            ),
        }
    else:
        raise Open04ExportError(
            f"{role} controller receipt schema differs"
        )
    for field, expected_value in controller_expected.items():
        if receipt[field] != expected_value:
            raise Open04ExportError(
                f"{role} controller receipt differs on {field}"
            )
    for field in (
        "terminal_summary_sha256",
        "run_manifest_sha256",
        "controller_summary_sha256",
        "controller_command_sha256",
    ):
        _require_sha(
            receipt[field],
            label=f"{role} controller receipt {field}",
        )
    for field in (
        "controller_command_exit_code",
        "controller_exit_code",
        "preflight_conflicts",
        "postrun_conflicts",
    ):
        _integer(
            receipt[field],
            label=f"{role} controller receipt {field}",
            minimum=0,
        )
    for field in ("controller_peak_rss_mb", "controller_avg_cpu_pct"):
        if receipt[field] is None and field == "controller_peak_rss_mb":
            raise Open04ExportError(
                f"{role} controller receipt {field} is missing"
            )
        if receipt[field] is None:
            if (
                receipt["schema_version"]
                != OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION
            ):
                raise Open04ExportError(
                    f"{role} controller receipt {field} is missing"
                )
            continue
        value = _finite(
            receipt[field],
            label=f"{role} controller receipt {field}",
        )
        if value < 0:
            raise Open04ExportError(
                f"{role} controller receipt {field} is negative"
            )
    _require_nonblank(
        receipt["controller_memory_guard_status"],
        label=f"{role} controller receipt controller_memory_guard_status",
    )
    started = _utc_datetime(
        receipt["started_at_utc"],
        label=f"{role} controller start",
    )
    completed = _utc_datetime(
        receipt["completed_at_utc"],
        label=f"{role} controller completion",
    )
    exited = _utc_datetime(
        receipt["worker_exit_confirmed_at_utc"],
        label=f"{role} controller worker exit",
    )
    if not started < completed <= exited:
        raise Open04ExportError(
            f"{role} controller receipt timestamps are not ordered"
        )
    return receipt, _artifact_record(
        path,
        logical_name="controller_completion_receipt",
        relative_path=relative.as_posix(),
    )


def _manifest_output_artifact_records(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    role: str,
    receipt_role: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw_records = _sequence(
        manifest.get("outputs"),
        label=f"{role} manifest output artifacts",
    )
    if not raw_records:
        raise Open04ExportError(
            f"{role} manifest output artifact inventory is empty"
        )
    retained: list[dict[str, Any]] = []
    by_relative: dict[str, Mapping[str, Any]] = {}
    run_relative = _safe_relative(
        receipt_role["run_relative_path"],
        label=f"{role} run relative path",
    )
    for index, raw in enumerate(raw_records):
        item = _mapping(
            raw,
            label=f"{role} manifest output artifact {index}",
        )
        logical_name = _require_nonblank(
            item.get("logical_name"),
            label=f"{role} manifest output artifact {index} logical name",
        )
        relative = _safe_relative(
            item.get("relative_path"),
            label=f"{role} manifest output artifact {index} path",
        )
        if (
            len(relative.parts) < 2
            or relative.parts[0] != "outputs"
            or relative.as_posix() in by_relative
        ):
            raise Open04ExportError(
                f"{role} manifest output paths are unsafe or duplicated"
            )
        path = _resolve_under(
            root,
            relative,
            label=f"{role} manifest output {logical_name}",
        )
        if not path.is_file():
            raise Open04ExportError(
                f"{role} manifest output is missing: {relative.as_posix()}"
            )
        expected_sha = _require_sha(
            item.get("sha256"),
            label=f"{role} manifest output {logical_name} SHA-256",
        )
        expected_bytes = _integer(
            item.get("bytes"),
            label=f"{role} manifest output {logical_name} bytes",
            minimum=0,
        )
        if (
            sha256_file(path) != expected_sha
            or path.stat().st_size != expected_bytes
        ):
            raise Open04ExportError(
                f"{role} manifest output identity mismatch: "
                f"{relative.as_posix()}"
            )
        record = _artifact_record(
            path,
            logical_name=f"manifest_output:{logical_name}",
            relative_path=(run_relative / relative).as_posix(),
        )
        record["manifest_logical_name"] = logical_name
        record["manifest_media_type"] = item.get("media_type")
        by_relative[relative.as_posix()] = item
        retained.append(record)

    bounded = _mapping(
        manifest.get("bounded_evidence"),
        label=f"{role} bounded evidence",
    )
    resource = _mapping(
        bounded.get("resource_artifact"),
        label=f"{role} bounded resource artifact",
    )
    resource_name = _require_nonblank(
        resource.get("path"),
        label=f"{role} bounded resource artifact path",
    )
    resource_relative = f"outputs/{resource_name}"
    resource_public = by_relative.get(resource_relative)
    if (
        resource_public is None
        or resource_public.get("sha256") != resource.get("sha256")
        or resource_public.get("bytes") != resource.get("bytes")
    ):
        raise Open04ExportError(
            f"{role} resource evidence is not bound by manifest outputs"
        )
    summary_public = by_relative.get("outputs/summary.json")
    if (
        summary_public is None
        or summary_public.get("sha256")
        != receipt_role["terminal_summary_sha256"]
    ):
        raise Open04ExportError(
            f"{role} terminal summary is not bound by manifest outputs"
        )
    annual_public = by_relative.get(
        f"outputs/{_ANNUAL_SOURCE_FILE}"
    )
    deterministic = _mapping(
        bounded.get("deterministic_artifacts"),
        label=f"{role} deterministic artifacts",
    )
    annual = _mapping(
        deterministic.get("annual"),
        label=f"{role} deterministic annual artifact",
    )
    if (
        annual_public is None
        or annual_public.get("sha256") != annual.get("sha256")
        or annual_public.get("bytes") != annual.get("bytes")
    ):
        raise Open04ExportError(
            f"{role} annual evidence is not bound by manifest outputs"
        )
    return tuple(retained)


def _bounded_annual_artifact(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    bounded = _mapping(
        manifest.get("bounded_evidence"),
        label="run bounded evidence",
    )
    deterministic = _mapping(
        bounded.get("deterministic_artifacts"),
        label="run deterministic artifacts",
    )
    record = dict(
        _mapping(
            deterministic.get("annual"),
            label="bounded annual artifact",
        )
    )
    relative = _safe_relative(
        record.get("path"),
        label="bounded annual artifact path",
    )
    if relative.as_posix() != _ANNUAL_SOURCE_FILE:
        raise Open04ExportError(
            "bounded annual artifact does not use the frozen filename"
        )
    if _integer(
        record.get("row_count"),
        label="bounded annual row count",
        minimum=0,
    ) != 11:
        raise Open04ExportError(
            "bounded annual artifact must contain exactly FY2026-FY2036"
        )
    path = _resolve_under(
        root / "outputs",
        relative,
        label="bounded annual artifact",
    )
    if not path.is_file():
        raise Open04ExportError("bounded annual artifact is missing")
    if sha256_file(path) != _require_sha(
        record.get("sha256"),
        label="bounded annual artifact SHA-256",
    ):
        raise Open04ExportError("bounded annual artifact SHA-256 mismatch")
    if path.stat().st_size != _integer(
        record.get("bytes"),
        label="bounded annual artifact bytes",
        minimum=1,
    ):
        raise Open04ExportError("bounded annual artifact byte count mismatch")
    return path, record


def _read_verified_annual_rows(
    path: Path,
    *,
    role: str,
) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(ANNUAL_COLUMNS):
                raise Open04ExportError(
                    f"{role} bounded annual header differs from producer schema"
                )
            raw_rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Open04ExportError(
            f"{role} bounded annual artifact is unreadable"
        ) from exc
    if len(raw_rows) != 11:
        raise Open04ExportError(
            f"{role} bounded annual artifact must contain 11 rows"
        )

    annual_flow_columns = (
        "tdc_change_bil",
        "overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil",
        "interest_outlay_bil",
        "issue_discount_cost_bil",
        "nonmarketable_interest_capitalized_bil",
        "tips_inflation_accretion_bil",
        "modeled_financing_cost_bil",
    )
    cumulative_columns = {
        "tdc_change_bil": "cumulative_tdc_change_bil",
        "overlap_cashflow_bil": "cumulative_overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil": (
            "cumulative_tdc_change_ex_overlap_bil"
        ),
        "interest_outlay_bil": "cumulative_interest_outlay_bil",
        "issue_discount_cost_bil": (
            "cumulative_issue_discount_cost_bil"
        ),
        "nonmarketable_interest_capitalized_bil": (
            "cumulative_nonmarketable_interest_capitalized_bil"
        ),
        "tips_inflation_accretion_bil": (
            "cumulative_tips_inflation_accretion_bil"
        ),
        "modeled_financing_cost_bil": (
            "cumulative_modeled_financing_cost_bil"
        ),
    }
    cumulative = {column: 0.0 for column in annual_flow_columns}
    normalized: list[dict[str, Any]] = []
    opening = date.fromisoformat(_START_DATE)
    for offset, raw in enumerate(raw_rows):
        fiscal_year = 2026 + offset
        expected_end = date(fiscal_year, 9, 30)
        expected_start = (
            opening if fiscal_year == 2026 else date(fiscal_year - 1, 9, 30)
        )
        full_start = date(fiscal_year - 1, 9, 30)
        coverage_days = (expected_end - expected_start).days
        expected_coverage_days = (expected_end - full_start).days
        partial = fiscal_year == 2026
        expected_label = (
            "FY2026_PARTIAL_OPENING"
            if partial
            else f"FY{fiscal_year}"
        )
        row: dict[str, Any] = {
            "fiscal_year": fiscal_year,
            "period_label": _require_nonblank(
                raw.get("period_label"),
                label=f"{role} FY{fiscal_year} period label",
            ),
            "period_start": _iso_date(
                raw.get("period_start"),
                label=f"{role} FY{fiscal_year} period start",
            ),
            "period_end": _iso_date(
                raw.get("period_end"),
                label=f"{role} FY{fiscal_year} period end",
            ),
            "is_partial_period": _boolean(
                raw.get("is_partial_period"),
                label=f"{role} FY{fiscal_year} partial flag",
            ),
            "coverage_days": _integer(
                raw.get("coverage_days"),
                label=f"{role} FY{fiscal_year} coverage days",
                minimum=1,
            ),
            "expected_coverage_days": _integer(
                raw.get("expected_coverage_days"),
                label=f"{role} FY{fiscal_year} expected coverage days",
                minimum=1,
            ),
            "aggregation_clock_id": _require_nonblank(
                raw.get("aggregation_clock_id"),
                label=f"{role} FY{fiscal_year} aggregation clock",
            ),
        }
        if (
            row["period_label"] != expected_label
            or row["period_start"] != expected_start.isoformat()
            or row["period_end"] != expected_end.isoformat()
            or row["is_partial_period"] is not partial
            or row["coverage_days"] != coverage_days
            or row["expected_coverage_days"] != expected_coverage_days
            or row["aggregation_clock_id"] != _CLOCK_ID
        ):
            raise Open04ExportError(
                f"{role} FY{fiscal_year} annual clock contract is stale"
            )
        if fiscal_year == 2026 and coverage_days != 101:
            raise Open04ExportError(
                f"{role} FY2026 opening interval must contain 101 days"
            )

        for column in annual_flow_columns:
            value = _finite(
                raw.get(column),
                label=f"{role} FY{fiscal_year} {column}",
            )
            row[column] = value
            cumulative[column] += value
            cumulative_column = cumulative_columns[column]
            declared = _finite(
                raw.get(cumulative_column),
                label=f"{role} FY{fiscal_year} {cumulative_column}",
            )
            row[cumulative_column] = declared

        _assert_close(
            row["tdc_change_bil"],
            row["overlap_cashflow_bil"]
            + row["tdc_change_ex_overlap_bil"],
            label=f"{role} FY{fiscal_year} TDC identity",
        )
        finance_components = (
            row["interest_outlay_bil"]
            + row["issue_discount_cost_bil"]
            + row["nonmarketable_interest_capitalized_bil"]
            + row["tips_inflation_accretion_bil"]
        )
        _assert_close(
            row["modeled_financing_cost_bil"],
            finance_components,
            label=f"{role} FY{fiscal_year} financing-cost identity",
        )
        for annual_column, cumulative_column in cumulative_columns.items():
            _assert_close(
                row[cumulative_column],
                cumulative[annual_column],
                label=f"{role} FY{fiscal_year} {cumulative_column}",
            )
        _assert_close(
            row["cumulative_tdc_change_bil"],
            row["cumulative_overlap_cashflow_bil"]
            + row["cumulative_tdc_change_ex_overlap_bil"],
            label=f"{role} FY{fiscal_year} cumulative TDC identity",
        )
        cumulative_finance_components = (
            row["cumulative_interest_outlay_bil"]
            + row["cumulative_issue_discount_cost_bil"]
            + row[
                "cumulative_nonmarketable_interest_capitalized_bil"
            ]
            + row["cumulative_tips_inflation_accretion_bil"]
        )
        _assert_close(
            row["cumulative_modeled_financing_cost_bil"],
            cumulative_finance_components,
            label=(
                f"{role} FY{fiscal_year} cumulative financing-cost identity"
            ),
        )
        for column, expected in (
            ("modeled_financing_cost_basis", _FINANCING_COST_BASIS),
            ("modeled_financing_cost_units", _FINANCING_COST_UNITS),
            ("cumulative_basis", _CUMULATIVE_BASIS),
        ):
            value = _require_nonblank(
                raw.get(column),
                label=f"{role} FY{fiscal_year} {column}",
            )
            if value != expected:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {column} is unsupported"
                )
            row[column] = value

        for column in (
            "new_issuance_face_bil",
            "new_issuance_original_term_face_years_bil",
            "new_issuance_bill_face_bil",
            "new_issuance_short_face_bil",
        ):
            value = _finite(
                raw.get(column),
                label=f"{role} FY{fiscal_year} {column}",
            )
            if value < -_TOLERANCE:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {column} is negative"
                )
            row[column] = 0.0 if abs(value) <= _TOLERANCE else value
        face = row["new_issuance_face_bil"]
        for column in (
            "new_issuance_bill_face_bil",
            "new_issuance_short_face_bil",
        ):
            if row[column] > face + _TOLERANCE:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {column} exceeds issuance"
                )
        ratio_specs = (
            (
                "new_issuance_wam_years",
                "new_issuance_original_term_face_years_bil",
            ),
            ("new_issuance_bill_share", "new_issuance_bill_face_bil"),
            (
                "new_issuance_short_maturity_share",
                "new_issuance_short_face_bil",
            ),
        )
        for ratio_column, numerator_column in ratio_specs:
            actual = _optional_finite(
                raw.get(ratio_column),
                label=f"{role} FY{fiscal_year} {ratio_column}",
            )
            if face <= 1e-12:
                if actual is not None:
                    raise Open04ExportError(
                        f"{role} FY{fiscal_year} {ratio_column} "
                        "must be empty without issuance"
                    )
                _assert_close(
                    row[numerator_column],
                    0.0,
                    label=f"{role} FY{fiscal_year} {numerator_column}",
                )
            else:
                if actual is None:
                    raise Open04ExportError(
                        f"{role} FY{fiscal_year} {ratio_column} is missing"
                    )
                _assert_close(
                    actual,
                    row[numerator_column] / face,
                    label=f"{role} FY{fiscal_year} {ratio_column}",
                )
            row[ratio_column] = actual
        for share_column in (
            "new_issuance_bill_share",
            "new_issuance_short_maturity_share",
        ):
            share = row[share_column]
            if share is not None and not -_TOLERANCE <= share <= 1 + _TOLERANCE:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {share_column} is outside [0,1]"
                )

        snapshot_date = _iso_date(
            raw.get("snapshot_date"),
            label=f"{role} FY{fiscal_year} snapshot date",
        )
        if snapshot_date != expected_end.isoformat():
            raise Open04ExportError(
                f"{role} FY{fiscal_year} snapshot is not exact September 30"
            )
        row["snapshot_date"] = snapshot_date
        for column in (
            "outstanding_controlled_wam_years",
            "outstanding_controlled_bill_share",
            "outstanding_controlled_short_maturity_share",
        ):
            value = _finite(
                raw.get(column),
                label=f"{role} FY{fiscal_year} {column}",
            )
            if value < -_TOLERANCE:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {column} is negative"
                )
            if column.endswith("_share") and value > 1 + _TOLERANCE:
                raise Open04ExportError(
                    f"{role} FY{fiscal_year} {column} exceeds one"
                )
            row[column] = value
        normalized.append(row)
    return normalized


def _compiled_input_records(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    records = _sequence(
        manifest.get("compiled_inputs"),
        label="run compiled inputs",
    )
    normalized: dict[str, dict[str, Any]] = {}
    prefix = PurePosixPath("compile/compiled/forecast_inputs")
    for index, raw in enumerate(records):
        record = dict(
            _mapping(raw, label=f"run compiled input {index}")
        )
        logical_name = _require_nonblank(
            record.get("logical_name"),
            label=f"run compiled input {index} logical name",
        )
        if (
            PurePosixPath(logical_name).name != logical_name
            or logical_name in normalized
        ):
            raise Open04ExportError(
                "run compiled input names are unsafe or duplicated"
            )
        relative = _safe_relative(
            record.get("relative_path"),
            label=f"run compiled input {logical_name} path",
        )
        if relative != prefix / logical_name:
            raise Open04ExportError(
                f"run compiled input path is stale: {logical_name}"
            )
        _require_sha(
            record.get("sha256"),
            label=f"run compiled input {logical_name} SHA-256",
        )
        _integer(
            record.get("bytes"),
            label=f"run compiled input {logical_name} bytes",
            minimum=0,
        )
        normalized[logical_name] = record
    if not normalized:
        raise Open04ExportError("run compiled input inventory is empty")
    return normalized


def _required_compiled_record(
    records: Mapping[str, dict[str, Any]],
    logical_name: str,
    *,
    role: str,
) -> dict[str, Any]:
    record = records.get(logical_name)
    if record is None:
        raise Open04ExportError(
            f"{role} required compiled input is missing: {logical_name}"
        )
    return record


def _compiled_input_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    logical_name: str,
) -> tuple[Path, dict[str, Any]]:
    records = _compiled_input_records(manifest)
    record = _required_compiled_record(
        records,
        logical_name,
        role=str(manifest.get("scenario", {}).get("scenario_id") or "run"),
    )
    path = _resolve_under(
        root,
        _safe_relative(
            record["relative_path"],
            label=f"compiled input {logical_name} path",
        ),
        label=f"compiled input {logical_name}",
    )
    if not path.is_file():
        raise Open04ExportError(
            f"compiled input is missing: {logical_name}"
        )
    if sha256_file(path) != record["sha256"]:
        raise Open04ExportError(
            f"compiled input SHA-256 mismatch: {logical_name}"
        )
    if path.stat().st_size != _integer(
        record["bytes"],
        label=f"compiled input {logical_name} bytes",
        minimum=0,
    ):
        raise Open04ExportError(
            f"compiled input byte count mismatch: {logical_name}"
        )
    return path, record


def _validate_issuance_mix(
    value: Mapping[str, Any],
    *,
    role: str,
    expected_file_sha: str,
    actual_file_sha: str,
    expected_canonical_sha: str,
) -> dict[str, Any]:
    payload = dict(value)
    expected_keys = {
        "schema_version",
        "mode",
        "selection_status",
        "security_shares",
        "maturity_distributions",
        "weighted_average_maturity_years",
        "negative_issuance_action",
        "source_role",
        "runtime_role",
        "claim_boundary",
    }
    _require_exact_keys(
        payload,
        expected_keys,
        label=f"{role} compiled issuance mix",
    )
    expected_metadata = (
        {
            "mode": "default_tdcsim_cbo_runner_profile",
            "selection_status": "configured_default",
            "source_role": "compiler_configured_default",
        }
        if role == "baseline"
        else {
            "mode": "replace_shares",
            "selection_status": "scenario_override",
            "source_role": "scenario_assumption",
        }
    )
    if payload["schema_version"] != "tdcsim_cbo_issuance_mix_assumptions_v1":
        raise Open04ExportError(
            f"{role} compiled issuance mix schema is unsupported"
        )
    for field, expected in expected_metadata.items():
        if payload[field] != expected:
            raise Open04ExportError(
                f"{role} compiled issuance mix differs on {field}"
            )
    if (
        payload["negative_issuance_action"] != "error"
        or payload["runtime_role"] != "hard_target"
        or payload["claim_boundary"]
        != "issuance_mix_is_tdcsim_scenario_assumption_not_cbo_prescription"
    ):
        raise Open04ExportError(
            f"{role} compiled issuance mix boundary is unsupported"
        )
    if actual_file_sha != _require_sha(
        expected_file_sha,
        label=f"{role} issuance mix file SHA-256",
    ):
        raise Open04ExportError(
            f"{role} issuance mix file SHA-256 differs from campaign receipt"
        )
    if canonical_json_sha256(payload) != _require_sha(
        expected_canonical_sha,
        label=f"{role} canonical issuance mix SHA-256",
    ):
        raise Open04ExportError(
            f"{role} canonical issuance mix differs from campaign receipt"
        )

    categories = set(MARKETABLE_PREFERENCE_CATEGORIES)
    shares = _exact_mapping(
        payload["security_shares"],
        categories,
        label=f"{role} issuance security shares",
    )
    distributions = _exact_mapping(
        payload["maturity_distributions"],
        categories,
        label=f"{role} issuance maturity distributions",
    )
    normalized_shares: dict[str, float] = {}
    normalized_distributions: dict[str, list[dict[str, float]]] = {}
    total_share = 0.0
    implied_wam = 0.0
    implied_short_share = 0.0
    for category in sorted(categories):
        security_share = _finite(
            shares[category],
            label=f"{role} issuance share {category}",
        )
        if security_share < 0:
            raise Open04ExportError(
                f"{role} issuance share is negative: {category}"
            )
        normalized_shares[category] = security_share
        total_share += security_share
        raw_distribution = _sequence(
            distributions[category],
            label=f"{role} issuance distribution {category}",
        )
        if not raw_distribution:
            raise Open04ExportError(
                f"{role} issuance distribution is empty: {category}"
            )
        normalized_items: list[dict[str, float]] = []
        within_share = 0.0
        within_wam = 0.0
        within_short = 0.0
        for index, raw_item in enumerate(raw_distribution):
            item = _exact_mapping(
                raw_item,
                {"maturity_years", "share"},
                label=f"{role} {category} maturity row {index}",
            )
            maturity = _finite(
                item["maturity_years"],
                label=f"{role} {category} maturity row {index} tenor",
            )
            weight = _finite(
                item["share"],
                label=f"{role} {category} maturity row {index} share",
            )
            if maturity <= 0 or weight < 0:
                raise Open04ExportError(
                    f"{role} {category} maturity row {index} is invalid"
                )
            within_share += weight
            within_wam += maturity * weight
            if maturity <= 1.0:
                within_short += weight
            normalized_items.append(
                {"maturity_years": maturity, "share": weight}
            )
        _assert_close(
            within_share,
            1.0,
            label=f"{role} {category} maturity shares",
            tolerance=1e-12,
        )
        implied_wam += security_share * within_wam
        implied_short_share += security_share * within_short
        normalized_distributions[category] = normalized_items
    _assert_close(
        total_share,
        1.0,
        label=f"{role} issuance security shares",
        tolerance=1e-12,
    )
    declared_wam = _finite(
        payload["weighted_average_maturity_years"],
        label=f"{role} compiled issuance WAM",
    )
    _assert_close(
        declared_wam,
        implied_wam,
        label=f"{role} compiled issuance WAM",
        tolerance=1e-12,
    )
    payload["security_shares"] = normalized_shares
    payload["maturity_distributions"] = normalized_distributions
    payload["weighted_average_maturity_years"] = declared_wam
    payload["_implied_bill_share"] = normalized_shares["bills"]
    payload["_implied_short_maturity_share"] = implied_short_share
    return payload


def _validate_scenario_perimeter(
    role: str,
    scenario: Mapping[str, Any],
) -> None:
    try:
        from .open04_campaign import validate_open04_scenario_contract

        marker = validate_open04_scenario_contract(scenario)
    except (ImportError, ValueError) as exc:
        raise Open04ExportError(
            f"{role} scenario is outside the canonical OPEN-04 perimeter"
        ) from exc
    if marker.role != role:
        raise Open04ExportError(
            f"{role} scenario campaign role is inconsistent"
        )


def _exact_row_metadata(
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    output_manifest = _mapping(
        manifest.get("output_manifest"),
        label=f"{role} output manifest",
    )
    metadata = dict(
        _mapping(
            output_manifest.get("row_metadata"),
            label=f"{role} output row metadata",
        )
    )
    required = (
        "scenario_id",
        "run_id",
        "package_id",
        "actuals_available_as_of",
        "scenario_config_sha256",
        "compiled_inputs_digest",
        "mmf_deposit_pass_through",
        "mmf_deposit_pass_through_status",
        "fiscal_incidence_policy_id",
        "fiscal_incidence_basis",
        "fiscal_incidence_du_share",
    )
    for field in required:
        if field not in metadata:
            raise Open04ExportError(
                f"{role} output row metadata is missing {field}"
            )
    scenario = _mapping(
        manifest.get("scenario"),
        label=f"{role} run scenario identity",
    )
    baseline = _mapping(
        manifest.get("baseline"),
        label=f"{role} run baseline identity",
    )
    exact = {
        "scenario_id": scenario.get("scenario_id"),
        "run_id": manifest.get("run_id"),
        "package_id": baseline.get("package_id"),
        "scenario_config_sha256": scenario.get("canonical_sha256"),
        "compiled_inputs_digest": manifest.get("compiled_inputs_digest"),
    }
    for field, expected in exact.items():
        if metadata[field] != expected:
            raise Open04ExportError(
                f"{role} output row metadata differs on {field}"
            )
    _iso_date(
        metadata["actuals_available_as_of"],
        label=f"{role} actuals available as of",
    )
    metadata["mmf_deposit_pass_through"] = _finite(
        metadata["mmf_deposit_pass_through"],
        label=f"{role} MMF deposit pass-through",
    )
    if not 0.0 <= metadata["mmf_deposit_pass_through"] <= 1.0:
        raise Open04ExportError(
            f"{role} MMF deposit pass-through is outside [0,1]"
        )
    metadata["fiscal_incidence_du_share"] = _finite(
        metadata["fiscal_incidence_du_share"],
        label=f"{role} fiscal-incidence DU share",
    )
    if not 0.0 <= metadata["fiscal_incidence_du_share"] <= 1.0:
        raise Open04ExportError(
            f"{role} fiscal-incidence DU share is outside [0,1]"
        )
    for field in (
        "mmf_deposit_pass_through_status",
        "fiscal_incidence_policy_id",
        "fiscal_incidence_basis",
    ):
        metadata[field] = _require_nonblank(
            metadata[field],
            label=f"{role} output metadata {field}",
        )
    return metadata


def _scenario_input_row(
    role: str,
    *,
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    issuance_mix: Mapping[str, Any],
    row_metadata: Mapping[str, Any],
    receipt_role: Mapping[str, Any],
    common: Mapping[str, Any],
    contract: Mapping[str, Any],
    campaign_contract_sha256: str,
    campaign_verification_receipt_sha256: str,
    calibration: Mapping[str, Any],
    holder_profile_sha256: str,
    runtime_assumptions_sha256: str,
    private_mmf_route_sha256: str,
) -> dict[str, Any]:
    contract_role = _mapping(
        _mapping(contract["roles"], label="campaign contract roles")[role],
        label=f"campaign contract role {role}",
    )
    provenance = _mapping(
        scenario.get("provenance"),
        label=f"{role} scenario provenance",
    )
    coupling = _mapping(
        scenario.get("coupling"),
        label=f"{role} scenario coupling",
    )
    overrides = _mapping(
        scenario.get("overrides"),
        label=f"{role} scenario overrides",
    )
    shares = _mapping(
        issuance_mix["security_shares"],
        label=f"{role} compiled issuance shares",
    )
    fixed_total = (
        float(shares["bills"])
        + float(shares["notes"])
        + float(shares["bonds"])
    )
    if fixed_total <= 0:
        raise Open04ExportError(
            f"{role} compiled issuance mix has no fixed-rate remainder"
        )

    nominal = overrides.get("nominal_yield_curve")
    if role == "baseline":
        curve = {
            "nominal_yield_curve_override_status": (
                "unchanged_release_bound_baseline"
            ),
            "curve_override_mode": "none",
            "curve_application": "not_applicable",
            "curve_interpolation": "not_applicable",
            "curve_lower_endpoint": "not_applicable",
            "curve_upper_endpoint": "not_applicable",
            "curve_time_profile": "not_applicable",
            "curve_compounding": "not_applicable",
            "curve_key_rates_json": "[]",
            "term_premium_channel_status": "not_applied",
            "calibration_source_label": "not_applicable_baseline",
            "calibration_status": "not_applicable_baseline",
            "source_claim_boundary": "not_applicable_baseline",
            "calibration_scope": "not_applicable_baseline",
            "calibration_central_bp": None,
            "calibration_source_low_bp": None,
            "calibration_source_high_bp": None,
            "source_range_status": "not_applicable_baseline",
            "persistence_status": "not_applicable_baseline",
            "hou_use_status": "not_applicable_baseline",
        }
    else:
        nominal_map = _mapping(
            nominal,
            label=f"{role} nominal curve override",
        )
        shocks = _sequence(
            nominal_map.get("shocks"),
            label=f"{role} nominal curve shocks",
        )
        curve = {
            "nominal_yield_curve_override_status": (
                "assumed_exogenous_post_baseline_evaluation"
            ),
            "curve_override_mode": nominal_map.get("mode"),
            "curve_application": nominal_map.get("application"),
            "curve_interpolation": nominal_map.get("interpolation"),
            "curve_lower_endpoint": nominal_map.get("lower_endpoint"),
            "curve_upper_endpoint": nominal_map.get("upper_endpoint"),
            "curve_time_profile": nominal_map.get("time_profile"),
            "curve_compounding": nominal_map.get("compounding"),
            "curve_key_rates_json": canonical_json_text(shocks),
            "term_premium_channel_status": (
                "assumed_sensitivity_not_behavioral_rate_response"
            ),
            "calibration_source_label": (
                "ATI_informed_assumed_10y_nominal_yield_level_sensitivity"
            ),
            "calibration_status": "conditional_assumption_calibration",
            "source_claim_boundary": (
                "source_informs_sign_and_scenario_magnitude_not_"
                "structural_elasticity"
            ),
            "calibration_scope": (
                "10y_nominal_level_sensitivity_only_no_TDC_holder_or_"
                "behavior_identification"
            ),
            "calibration_central_bp": calibration["central_bp"],
            "calibration_source_low_bp": calibration["lower_bp"],
            "calibration_source_high_bp": calibration["upper_bp"],
            "source_range_status": (
                "heterogeneous_method_range_not_confidence_interval"
            ),
            "persistence_status": (
                "horizon_wide_level_is_additional_TDCSim_scenario_assumption"
            ),
            "hou_use_status": (
                "high_sensitivity_diagnostic_only_not_primary_calibration"
            ),
        }

    row = {
        "contract_version": INPUT_CONTRACT_SCHEMA_VERSION,
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": campaign_contract_sha256,
        "campaign_verification_receipt_sha256": (
            campaign_verification_receipt_sha256
        ),
        "scenario_role": role,
        "scenario_filename": contract_role[
            "scenario_source_relative_path"
        ],
        "scenario_id": receipt_role["scenario_id"],
        "source_scenario_title": OPEN04_ROLE_DESIGN_LABELS[role],
        "scenario_design_label": OPEN04_ROLE_DESIGN_LABELS[role],
        "claim_strength_label": OPEN04_ROLE_CLAIM_LABELS[role],
        "claim_status": _CLAIM_STATUS,
        "scenario_source_sha256": receipt_role["scenario_source_sha256"],
        "scenario_canonical_sha256": receipt_role["scenario_sha256"],
        "run_scenario_copy_sha256": receipt_role[
            "run_scenario_copy_sha256"
        ],
        "source_run_canonical_match_status": receipt_role[
            "source_run_canonical_match_status"
        ],
        "run_id": receipt_role["run_id"],
        "run_manifest_sha256": receipt_role["run_manifest_sha256"],
        "controller_completion_receipt_sha256": receipt_role[
            "controller_completion_receipt_sha256"
        ],
        "terminal_summary_sha256": receipt_role[
            "terminal_summary_sha256"
        ],
        "source_output_manifest_sha256": receipt_role[
            "source_output_manifest_sha256"
        ],
        "event_schema_version": receipt_role["event_schema_version"],
        "event_count": receipt_role["event_count"],
        "event_root_sha256": receipt_role["event_root_sha256"],
        "compiled_inputs_digest": receipt_role["compiled_inputs_digest"],
        "baseline_package_id": common["package_id"],
        "baseline_package_sha256": common["baseline_package_sha256"],
        "baseline_manifest_sha256": common["baseline_manifest_sha256"],
        "baseline_attestation_sha256": common[
            "release_attestation_sha256"
        ],
        "baseline_release_commit_sha": _BASELINE_RELEASE_COMMIT_SHA,
        "baseline_requirements_lock_sha256": (
            _BASELINE_REQUIREMENTS_LOCK_SHA256
        ),
        "simulation_start_date": common["start_date"],
        "simulation_end_date": common["end_date"],
        "simulation_frequency": common["frequency"],
        "funding_closure_mode": scenario["open04_campaign"][
            "funding_closure_mode"
        ],
        "cbo_debt_path_status": (
            "common_reference_not_binding_scenario_target"
        ),
        "financing_closure": (
            "endogenous_additional_treasury_issuance_at_predeclared_tga_floor"
        ),
        "cash_closure_target_bil": 0.0,
        "validation_floor_bil": -0.000001,
        "financing_claim_limits": (
            "not_forecast|not_optimization|not_causal_estimate|"
            "not_estimated_treasury_choice_rule|no_behavioral_holder_response"
        ),
        "override_keys_json": canonical_json_text(sorted(overrides)),
        "coupling_contract_json": canonical_json_text(coupling),
        "issuance_mix_mode": issuance_mix["mode"],
        "tips_share": shares["tips"],
        "frn_share": shares["frn"],
        "fixed_remainder_bills_share": shares["bills"] / fixed_total,
        "fixed_remainder_notes_share": shares["notes"] / fixed_total,
        "fixed_remainder_bonds_share": shares["bonds"] / fixed_total,
        "maturity_distributions_json": canonical_json_text(
            issuance_mix["maturity_distributions"]
        ),
        "negative_issuance_action": issuance_mix[
            "negative_issuance_action"
        ],
        "mmf_deposit_pass_through": row_metadata[
            "mmf_deposit_pass_through"
        ],
        "mmf_deposit_pass_through_status": row_metadata[
            "mmf_deposit_pass_through_status"
        ],
        "fiscal_incidence_policy_id": row_metadata[
            "fiscal_incidence_policy_id"
        ],
        "fiscal_incidence_basis": row_metadata["fiscal_incidence_basis"],
        "fiscal_incidence_du_share": row_metadata[
            "fiscal_incidence_du_share"
        ],
        "short_maturity_cutoff_years": 1.0,
        "implied_new_issuance_wam_years": issuance_mix[
            "weighted_average_maturity_years"
        ],
        "implied_new_issuance_bill_share": issuance_mix[
            "_implied_bill_share"
        ],
        "implied_new_issuance_short_maturity_share": issuance_mix[
            "_implied_short_maturity_share"
        ],
        **curve,
        "signed_10y_shock_bp": receipt_role["signed_10y_shock_bp"],
        "compiled_curve_delta_digest": receipt_role[
            "compiled_curve_delta_digest"
        ],
        "ati_primary_pdf_sha256": _ATI_PRIMARY_PDF_SHA256,
        "hou_primary_pdf_sha256": _HOU_PRIMARY_PDF_SHA256,
        "calibration_worksheet_sha256": common[
            "calibration_worksheet_sha256"
        ],
        "calibration_worksheet_canonical_sha256": calibration[
            "canonical_sha256"
        ],
        "holder_override_status": (
            "hand_specified_outcome_conditioned_1pp_private_to_banks_"
            "new_20y_30y_nominal_bonds"
            if role == "candidate_a"
            else "none_fixed_baseline_holders"
        ),
        "holder_profile_sha256": holder_profile_sha256,
        "holder_trading_status": (
            "disabled_fixed_no_rate_sensitive_demand"
        ),
        "private_mmf_route_sha256": private_mmf_route_sha256,
        "mmf_pass_through_source_sha256": runtime_assumptions_sha256,
        "input_freeze_sha256": campaign_contract_sha256,
        "canonical_issuance_mix_sha256": receipt_role[
            "canonical_issuance_mix_sha256"
        ],
        "issuance_mix_file_sha256": receipt_role[
            "issuance_mix_file_sha256"
        ],
        "economic_changed_paths_json": canonical_json_text(
            receipt_role["economic_changed_paths"]
        ),
        "physical_changed_inputs_json": canonical_json_text(
            receipt_role["physical_changed_inputs"]
        ),
        "fixed_input_comparison_status": receipt_role[
            "fixed_input_comparison_status"
        ],
        "fixed_input_records_sha256": receipt_role[
            "fixed_input_records_sha256"
        ],
        "curve_date_set_sha256": receipt_role["curve_date_set_sha256"],
        "tenor_set_sha256": receipt_role["tenor_set_sha256"],
        "scenario_provenance_kind": provenance["kind"],
        "scenario_provenance_as_of_date": row_metadata[
            "actuals_available_as_of"
        ],
        "provenance_scope_note": (
            "hand_specified_mechanism_sensitivity_not_forecast_causal_"
            "optimization_or_identified_holder_response"
        ),
    }
    if set(row) != set(SCENARIO_INPUT_COLUMNS):
        raise Open04ExportError(
            f"{role} scenario input projection fields are incomplete"
        )
    return row


def _validate_cross_run_results(
    projections: Sequence[_RunProjection],
    *,
    verification: Mapping[str, Any],
) -> None:
    if tuple(item.role for item in projections) != OPEN04_ROLES:
        raise Open04ExportError(
            "OPEN-04 projections are not in canonical role order"
        )
    by_role = {item.role: item for item in projections}
    annual_keys = [
        (
            row["fiscal_year"],
            row["period_label"],
            row["period_start"],
            row["period_end"],
        )
        for row in by_role["baseline"].annual_rows
    ]
    for role in OPEN04_ROLES[1:]:
        observed = [
            (
                row["fiscal_year"],
                row["period_label"],
                row["period_start"],
                row["period_end"],
            )
            for row in by_role[role].annual_rows
        ]
        if observed != annual_keys:
            raise Open04ExportError(
                f"{role} annual fiscal-year clock differs from baseline"
            )

    receipt_roles = _mapping(
        verification["roles"],
        label="campaign receipt roles",
    )
    terminal_cost: dict[str, float] = {}
    terminal_tdc: dict[str, float] = {}
    for role in OPEN04_ROLES:
        last = by_role[role].annual_rows[-1]
        receipt_role = _mapping(
            receipt_roles[role],
            label=f"campaign receipt role {role}",
        )
        terminal_cost[role] = last[
            "cumulative_modeled_financing_cost_bil"
        ]
        terminal_tdc[role] = last["cumulative_tdc_change_bil"]
        _assert_close(
            terminal_cost[role],
            receipt_role[
                "terminal_cumulative_modeled_financing_cost_bil"
            ],
            label=f"{role} terminal cumulative financing cost",
        )
        _assert_close(
            terminal_tdc[role],
            receipt_role["terminal_cumulative_tdc_change_bil"],
            label=f"{role} terminal cumulative TDC",
        )
        _assert_close(
            by_role[role].overall_wam,
            receipt_role["overall_new_issuance_wam_years"],
            label=f"{role} input-implied issuance WAM",
            tolerance=1e-12,
        )

    gates = _mapping(
        verification["pair_gates"],
        label="campaign pair gates",
    )
    deltas = {
        "candidate_a_financing_cost_delta_bil": (
            terminal_cost["candidate_a"] - terminal_cost["baseline"]
        ),
        "candidate_a_tdc_delta_bil": (
            terminal_tdc["candidate_a"] - terminal_tdc["baseline"]
        ),
        "candidate_b_financing_cost_delta_bil": (
            terminal_cost["candidate_b"] - terminal_cost["baseline"]
        ),
        "candidate_b_tdc_delta_bil": (
            terminal_tdc["candidate_b"] - terminal_tdc["baseline"]
        ),
    }
    for field, value in deltas.items():
        _assert_close(
            value,
            gates[field],
            label=f"campaign pair gate {field}",
        )
    threshold = _finite(
        gates["threshold_bil"],
        label="campaign terminal sign threshold",
    )
    if not (
        deltas["candidate_a_financing_cost_delta_bil"] < -threshold
        and deltas["candidate_a_tdc_delta_bil"] > threshold
        and deltas["candidate_b_financing_cost_delta_bil"] > threshold
        and deltas["candidate_b_tdc_delta_bil"] < -threshold
    ):
        raise Open04ExportError(
            "campaign terminal financing-cost/TDC sign gate failed"
        )

    overall = _mapping(
        gates["overall_wam_ordering"],
        label="campaign overall WAM ordering",
    )
    for role in OPEN04_ROLES:
        _assert_close(
            by_role[role].overall_wam,
            overall[role],
            label=f"campaign overall WAM ordering {role}",
            tolerance=1e-12,
        )
    if not (
        by_role["candidate_a"].overall_wam
        < by_role["baseline"].overall_wam
        < by_role["candidate_b"].overall_wam
    ):
        raise Open04ExportError(
            "campaign input-implied WAM ordering failed"
        )

    declared_full_fy = _sequence(
        gates["full_fy_wam_ordering"],
        label="campaign full-FY WAM ordering",
    )
    declared_by_year: dict[int, Mapping[str, Any]] = {}
    for index, raw in enumerate(declared_full_fy):
        record = _mapping(
            raw,
            label=f"campaign full-FY WAM row {index}",
        )
        fiscal_year = _integer(
            record.get("fiscal_year"),
            label=f"campaign full-FY WAM row {index} year",
            minimum=2027,
        )
        if fiscal_year in declared_by_year:
            raise Open04ExportError(
                "campaign full-FY WAM years are duplicated"
            )
        declared_by_year[fiscal_year] = record

    expected_years: list[int] = []
    for row_index in range(1, 11):
        rows = {
            role: by_role[role].annual_rows[row_index]
            for role in OPEN04_ROLES
        }
        fiscal_year = rows["baseline"]["fiscal_year"]
        positive = all(
            row["new_issuance_face_bil"] > 1e-12
            for row in rows.values()
        )
        if not positive:
            if any(
                row["new_issuance_face_bil"] > 1e-12
                for row in rows.values()
            ):
                raise Open04ExportError(
                    f"FY{fiscal_year} issuance support differs across roles"
                )
            continue
        expected_years.append(fiscal_year)
        values = {
            role: rows[role]["new_issuance_wam_years"]
            for role in OPEN04_ROLES
        }
        if any(value is None for value in values.values()):
            raise Open04ExportError(
                f"FY{fiscal_year} positive issuance lacks a WAM"
            )
        if not (
            values["candidate_a"]
            < values["baseline"]
            < values["candidate_b"]
        ):
            raise Open04ExportError(
                f"FY{fiscal_year} realized WAM ordering failed"
            )
        declared = declared_by_year.get(fiscal_year)
        if declared is None or declared.get("status") != "pass":
            raise Open04ExportError(
                f"FY{fiscal_year} WAM gate is absent from campaign receipt"
            )
        for role in OPEN04_ROLES:
            _assert_close(
                values[role],
                declared[role],
                label=f"FY{fiscal_year} realized WAM {role}",
            )
    if sorted(declared_by_year) != expected_years:
        raise Open04ExportError(
            "campaign full-FY WAM receipt does not match positive-issuance years"
        )


def _tdc_export_row(
    projection: _RunProjection,
    annual: Mapping[str, Any],
    *,
    campaign_id: str,
) -> dict[str, Any]:
    partial = bool(annual["is_partial_period"])
    row = {
        "schema_version": TDC_PATH_SCHEMA_VERSION,
        "clock_id": _CLOCK_ID,
        "campaign_id": campaign_id,
        "scenario_role": projection.role,
        "scenario_id": projection.input_row["scenario_id"],
        "source_scenario_title": projection.input_row[
            "source_scenario_title"
        ],
        "claim_strength_label": projection.input_row[
            "claim_strength_label"
        ],
        "claim_status": _CLAIM_STATUS,
        "fiscal_year": annual["fiscal_year"],
        "period_label": annual["period_label"],
        "period_status": (
            "partial_opening"
            if partial
            else "full_fiscal_year"
        ),
        "coverage_start_exclusive": annual["period_start"],
        "coverage_end_inclusive": annual["period_end"],
        "coverage_interval": (
            f"({annual['period_start']},{annual['period_end']}]"
        ),
        "coverage_days": annual["coverage_days"],
        "cumulative_origin_date": _START_DATE,
        "tdc_change_bil": annual["tdc_change_bil"],
        "overlap_cashflow_bil": annual["overlap_cashflow_bil"],
        "tdc_change_ex_overlap_bil": annual[
            "tdc_change_ex_overlap_bil"
        ],
        "cumulative_tdc_change_bil": annual[
            "cumulative_tdc_change_bil"
        ],
        "cumulative_overlap_cashflow_bil": annual[
            "cumulative_overlap_cashflow_bil"
        ],
        "cumulative_tdc_change_ex_overlap_bil": annual[
            "cumulative_tdc_change_ex_overlap_bil"
        ],
        "interest_outlay_bil": annual["interest_outlay_bil"],
        "issue_discount_cost_bil": annual["issue_discount_cost_bil"],
        "nonmarketable_interest_capitalized_bil": annual[
            "nonmarketable_interest_capitalized_bil"
        ],
        "tips_inflation_accretion_bil": annual[
            "tips_inflation_accretion_bil"
        ],
        "modeled_financing_cost_bil": annual[
            "modeled_financing_cost_bil"
        ],
        "cumulative_interest_outlay_bil": annual[
            "cumulative_interest_outlay_bil"
        ],
        "cumulative_issue_discount_cost_bil": annual[
            "cumulative_issue_discount_cost_bil"
        ],
        "cumulative_nonmarketable_interest_capitalized_bil": annual[
            "cumulative_nonmarketable_interest_capitalized_bil"
        ],
        "cumulative_tips_inflation_accretion_bil": annual[
            "cumulative_tips_inflation_accretion_bil"
        ],
        "cumulative_modeled_financing_cost_bil": annual[
            "cumulative_modeled_financing_cost_bil"
        ],
        "modeled_financing_cost_basis": _FINANCING_COST_BASIS,
        "modeled_financing_cost_units": _FINANCING_COST_UNITS,
        "cumulative_basis": _CUMULATIVE_BASIS,
        "tdc_amount_basis": TDC_AMOUNT_BASIS,
        "holder_allocation_scope": TDC_HOLDER_SCOPE,
        "overlap_policy": TDC_OVERLAP_POLICY,
    }
    if set(row) != set(TDC_PATH_COLUMNS):
        raise Open04ExportError("TDC path projection fields are incomplete")
    return row


def _maturity_export_row(
    projection: _RunProjection,
    annual: Mapping[str, Any],
    *,
    campaign_id: str,
) -> dict[str, Any]:
    partial = bool(annual["is_partial_period"])
    issuance_present = annual["new_issuance_face_bil"] > 1e-12
    row = {
        "schema_version": MATURITY_SCHEMA_VERSION,
        "clock_id": _CLOCK_ID,
        "campaign_id": campaign_id,
        "scenario_role": projection.role,
        "scenario_id": projection.input_row["scenario_id"],
        "source_scenario_title": projection.input_row[
            "source_scenario_title"
        ],
        "claim_strength_label": projection.input_row[
            "claim_strength_label"
        ],
        "claim_status": _CLAIM_STATUS,
        "fiscal_year": annual["fiscal_year"],
        "period_label": annual["period_label"],
        "period_status": (
            "partial_opening"
            if partial
            else "full_fiscal_year"
        ),
        "coverage_start_exclusive": annual["period_start"],
        "coverage_end_inclusive": annual["period_end"],
        "coverage_interval": (
            f"({annual['period_start']},{annual['period_end']}]"
        ),
        "coverage_days": annual["coverage_days"],
        "issuance_scope": "controlled_public_marketable",
        "issuance_face_bil": annual["new_issuance_face_bil"],
        "issuance_metric_status": (
            "positive_issuance"
            if issuance_present
            else "no_positive_issuance"
        ),
        "new_issuance_wam_years": annual["new_issuance_wam_years"],
        "new_issuance_bill_share": annual["new_issuance_bill_share"],
        "new_issuance_short_maturity_share": annual[
            "new_issuance_short_maturity_share"
        ],
        "new_issuance_weight_basis": (
            "face_issued_bil_x_original_maturity"
        ),
        "outstanding_scope": "controlled_public_marketable",
        "outstanding_snapshot_date": annual["snapshot_date"],
        "snapshot_status": "exact_period_end",
        "outstanding_controlled_wam_years": annual[
            "outstanding_controlled_wam_years"
        ],
        "outstanding_controlled_bill_share": annual[
            "outstanding_controlled_bill_share"
        ],
        "outstanding_controlled_short_maturity_share": annual[
            "outstanding_controlled_short_maturity_share"
        ],
        "outstanding_weight_basis": (
            "tips_adjusted_principal_else_face_x_remaining_maturity"
        ),
        "short_maturity_cutoff_years": 1.0,
        "maturity_authority": (
            "WAMEST_authority_TDCSim_controlled_scenario_metrics"
        ),
    }
    if set(row) != set(MATURITY_COLUMNS):
        raise Open04ExportError(
            "maturity-metric projection fields are incomplete"
        )
    return row


def _build_export_receipt(
    *,
    contract_doc: _Document,
    verification_doc: _Document,
    contract: Mapping[str, Any],
    verification: Mapping[str, Any],
    projections: Sequence[_RunProjection],
    thin_artifacts: Sequence[Mapping[str, Any]],
    consumer: Mapping[str, Any],
    exporter_identity: Mapping[str, Any],
) -> dict[str, Any]:
    common = _mapping(
        verification["common_identity"],
        label="campaign receipt common identity",
    )
    receipt_roles = _mapping(
        verification["roles"],
        label="campaign receipt roles",
    )
    producer_sources: list[dict[str, Any]] = [
        _document_source_record(
            contract_doc,
            logical_name="frozen_campaign_contract",
        ),
        _document_source_record(
            verification_doc,
            logical_name="campaign_verification_receipt",
        ),
    ]
    for projection in projections:
        producer_sources.extend(dict(item) for item in projection.source_records)

    scenarios: list[dict[str, Any]] = []
    remote_roles: list[dict[str, Any]] = []
    pair_gates = dict(
        _mapping(
            verification["pair_gates"],
            label="campaign receipt pair gates",
        )
    )
    for projection in projections:
        role = projection.role
        role_receipt = _mapping(
            receipt_roles[role],
            label=f"campaign receipt role {role}",
        )
        code_environment = {
            key: common[key]
            for key in (
                "code_commit_sha",
                "dirty_state",
                "requirements_lock_sha256",
                "uv_lock_sha256",
                "dependency_lock_set_sha256",
                "wheel_sha256",
                "wheel_artifact_sha256",
                "runner_source_sha256",
                "sim_engine_source_sha256",
                "bounded_output_source_sha256",
                "output_source_sha256",
                "verifier_source_sha256",
                "compiler_source_sha256",
                "contract_source_sha256",
                "manifest_source_sha256",
                "run_manifest_schema_sha256",
                "scenario_schema_sha256",
                "scenario_writer_source_sha256",
                "open04_exporter_source_sha256",
                "python_version",
                "package_name",
                "package_version",
                "distribution_file_digest",
                "runtime_identity_source",
            )
        }
        bounded = _mapping(
            projection.manifest.get("bounded_evidence"),
            label=f"{role} bounded evidence",
        )
        thresholds = dict(
            _mapping(
                bounded.get("memory_thresholds"),
                label=f"{role} memory thresholds",
            )
        )
        parent_watchdog = dict(
            _mapping(
                projection.manifest.get("parent_watchdog"),
                label=f"{role} parent watchdog evidence",
            )
        )
        controller = dict(projection.controller_receipt)
        scenario_source_outputs = [
            dict(item) for item in projection.source_records
        ]
        if role == "baseline":
            sign_evidence = {
                "candidate_sign_gate_status": "not_applicable_baseline",
                "delta_terminal_modeled_financing_cost_bil": 0.0,
                "delta_terminal_cumulative_tdc_bil": 0.0,
                "sign_epsilon_bil": pair_gates["threshold_bil"],
            }
        else:
            prefix = role
            sign_evidence = {
                "candidate_sign_gate_status": "pass",
                "delta_terminal_modeled_financing_cost_bil": pair_gates[
                    f"{prefix}_financing_cost_delta_bil"
                ],
                "delta_terminal_cumulative_tdc_bil": pair_gates[
                    f"{prefix}_tdc_delta_bil"
                ],
                "sign_epsilon_bil": pair_gates["threshold_bil"],
            }
        scenarios.append(
            {
                "scenario_role": role,
                "scenario_id": role_receipt["scenario_id"],
                "claim_strength_label": OPEN04_ROLE_CLAIM_LABELS[role],
                "claim_status": _CLAIM_STATUS,
                "scenario_source_sha256": role_receipt[
                    "scenario_source_sha256"
                ],
                "scenario_canonical_sha256": role_receipt[
                    "scenario_sha256"
                ],
                "run_scenario_copy_sha256": role_receipt[
                    "run_scenario_copy_sha256"
                ],
                "source_run_canonical_match_status": role_receipt[
                    "source_run_canonical_match_status"
                ],
                "tdcsim_run_id": role_receipt["run_id"],
                "run_manifest_sha256": role_receipt[
                    "run_manifest_sha256"
                ],
                "compiled_inputs_digest": role_receipt[
                    "compiled_inputs_digest"
                ],
                "code_environment": code_environment,
                "verification": {
                    "run_manifest_status": "pass",
                    "independent_run_verification_status": role_receipt[
                        "verification_status"
                    ],
                    "deterministic_replay_status": role_receipt[
                        "deterministic_replay_status"
                    ],
                    "accounting_invariants_status": role_receipt[
                        "accounting_invariants_status"
                    ],
                    "financing_cost_component_identity_status": (
                        role_receipt[
                            "financing_cost_component_identity_status"
                        ]
                    ),
                    "financing_cost_component_closure_status": (
                        role_receipt[
                            "financing_cost_component_identity_status"
                        ]
                    ),
                    "financing_cost_annual_reconciliation_status": "pass",
                    "financing_cost_cumulative_reconciliation_status": (
                        "pass"
                    ),
                    "memory_watchdog_status": role_receipt[
                        "memory_watchdog_status"
                    ],
                    "scenario_contract_status": "pass",
                    "fixed_input_comparison_status": role_receipt[
                        "fixed_input_comparison_status"
                    ],
                },
                "sign_gate": sign_evidence,
                "terminal_cumulative_modeled_financing_cost_bil": (
                    role_receipt[
                        "terminal_cumulative_modeled_financing_cost_bil"
                    ]
                ),
                "terminal_cumulative_tdc_change_bil": role_receipt[
                    "terminal_cumulative_tdc_change_bil"
                ],
                "input_implied_new_issuance_wam_years": role_receipt[
                    "overall_new_issuance_wam_years"
                ],
                "event_schema_version": role_receipt[
                    "event_schema_version"
                ],
                "event_count": role_receipt["event_count"],
                "event_root_sha256": role_receipt["event_root_sha256"],
                "source_output_manifest_sha256": role_receipt[
                    "source_output_manifest_sha256"
                ],
                "source_outputs": scenario_source_outputs,
                "source_output_artifact_count": len(
                    scenario_source_outputs
                ),
                "source_output_artifact_manifest_sha256": (
                    canonical_json_sha256(scenario_source_outputs)
                ),
                "memory_evidence": {
                    "memory_watchdog_status": role_receipt[
                        "memory_watchdog_status"
                    ],
                    "worker_peak_rss_bytes": bounded[
                        "peak_rss_bytes"
                    ],
                    "memory_thresholds": thresholds,
                    "parent_watchdog": parent_watchdog,
                    "resource_artifact": dict(
                        _mapping(
                            bounded.get("resource_artifact"),
                            label=f"{role} resource artifact",
                        )
                    ),
                    "controller_peak_rss_mb": controller[
                        "controller_peak_rss_mb"
                    ],
                    "controller_avg_cpu_pct": controller[
                        "controller_avg_cpu_pct"
                    ],
                    "controller_memory_guard_status": controller[
                        "controller_memory_guard_status"
                    ],
                    "controller_telemetry_status": controller[
                        "controller_telemetry_status"
                    ],
                    "controller_telemetry_locator": controller[
                        "controller_telemetry_locator"
                    ],
                    "controller_process_tree_drained": controller[
                        "controller_process_tree_drained"
                    ],
                },
                "accounting_evidence": {
                    "accounting_invariants_status": role_receipt[
                        "accounting_invariants_status"
                    ],
                    "financing_cost_component_identity_status": (
                        role_receipt[
                            "financing_cost_component_identity_status"
                        ]
                    ),
                    "deterministic_replay_status": role_receipt[
                        "deterministic_replay_status"
                    ],
                    "event_schema_version": role_receipt[
                        "event_schema_version"
                    ],
                    "event_count": role_receipt["event_count"],
                    "event_root_sha256": role_receipt[
                        "event_root_sha256"
                    ],
                },
            }
        )
        remote_roles.append(
            {
                "scenario_role": role,
                "run_id": role_receipt["run_id"],
                "run_relative_path": role_receipt["run_relative_path"],
                "host": role_receipt["host"],
                "controller": controller["controller"],
                "placement": controller["placement"],
                "controller_completion_receipt_sha256": role_receipt[
                    "controller_completion_receipt_sha256"
                ],
                "terminal_summary_sha256": role_receipt[
                    "terminal_summary_sha256"
                ],
                "controller_run_id": controller["controller_run_id"],
                "controller_summary_sha256": controller[
                    "controller_summary_sha256"
                ],
                "controller_command_sha256": controller[
                    "controller_command_sha256"
                ],
                "controller_command_exit_code": controller[
                    "controller_command_exit_code"
                ],
                "controller_exit_code": controller[
                    "controller_exit_code"
                ],
                "preflight_conflicts": controller[
                    "preflight_conflicts"
                ],
                "postrun_conflicts": controller["postrun_conflicts"],
                "started_at_utc": controller["started_at_utc"],
                "completed_at_utc": controller["completed_at_utc"],
                "worker_exit_confirmed_at_utc": controller[
                    "worker_exit_confirmed_at_utc"
                ],
                "controller_peak_rss_mb": controller[
                    "controller_peak_rss_mb"
                ],
                "controller_avg_cpu_pct": controller[
                    "controller_avg_cpu_pct"
                ],
                "controller_memory_guard_status": controller[
                    "controller_memory_guard_status"
                ],
                "controller_telemetry_status": controller[
                    "controller_telemetry_status"
                ],
                "controller_telemetry_locator": controller[
                    "controller_telemetry_locator"
                ],
                "controller_process_tree_drained": controller[
                    "controller_process_tree_drained"
                ],
                "completion_status": "pass",
                "process_terminated_before_next_status": "pass",
            }
        )

    artifacts = [dict(item) for item in thin_artifacts]
    expected_consumer_files = [
        {
            "logical_name": item["logical_name"],
            "relative_path": item["consumer_path"],
            "sha256": item["consumer_sha256"],
            "bytes": item["consumer_bytes"],
            "rows": item["consumer_rows"],
        }
        for item in artifacts
    ]
    expected_consumer_files.append(
        {
            "logical_name": OPEN04_EXPORT_FILES[3],
            "relative_path": OPEN04_EXPORT_FILES[3],
            "self_sha256_status": (
                "excluded_non_self_referential_manifest_boundary"
            ),
            "copy_identity_requirement": (
                "producer_and_consumer_receipt_bytes_identical"
            ),
        }
    )
    consumer_copy_manifest = {
        "scope": "three_non_self_referential_csv_artifacts",
        "receipt_copy_requirement": (
            "producer_and_consumer_receipt_bytes_identical"
        ),
        "artifacts": expected_consumer_files[:-1],
    }
    consumer_copy_manifest_sha256 = canonical_json_sha256(
        consumer_copy_manifest
    )
    thin_output_manifest_sha256 = canonical_json_sha256(artifacts)
    return {
        "schema_version": EXPORT_RECEIPT_SCHEMA_VERSION,
        "open_item_id": "OPEN-04",
        "producer_project": "TDCSim",
        "consumer_project": consumer["consumer_project"],
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "overall_status": "pass",
        "claim_boundary": {
            "claim_status": _CLAIM_STATUS,
            "description": (
                "Signed accounting paths under frozen hand-specified "
                "counterfactual inputs; not forecasts, causal effects, "
                "Treasury optimization, estimated yield responses, or "
                "identified holder behavior."
            ),
            "positive_tdc_interpretation": (
                "larger_modeled_treasury_attributed_deposit_contribution"
            ),
            "negative_tdc_interpretation": (
                "smaller_contribution_or_drain_under_accounting_convention"
            ),
        },
        "clock_contract": {
            "clock_id": _CLOCK_ID,
            "simulation_start_date": _START_DATE,
            "simulation_end_date": _END_DATE,
            "opening_interval": "(2026-06-21,2026-09-30]",
            "opening_period_label": "FY2026_PARTIAL_OPENING",
            "opening_coverage_days": 101,
            "opening_annualization": "none",
            "full_fiscal_years": list(range(2027, 2037)),
            "period_assignment_rule": "federal_fiscal_year_by_period_end",
        },
        "aggregation_contract": {
            "tdc_flows": "sum_within_fiscal_year",
            "financing_cost_components": "sum_within_fiscal_year",
            "cumulative_basis": _CUMULATIVE_BASIS,
            "first_cumulative_equals_first_annual_status": "pass",
            "new_issuance_metrics": (
                "positive_face_weighted_original_term_within_fiscal_year"
            ),
            "outstanding_metrics": "exact_september_30_snapshot_only",
            "cumulative_maturity_metrics": "not_produced_nonadditive",
            "annualization": "none",
        },
        "campaign": {
            "campaign_id": contract["campaign_id"],
            "contract_id": contract["contract_id"],
            "campaign_root_identity": contract[
                "campaign_root_identity"
            ],
            "campaign_contract_sha256": contract_doc.source_sha256,
            "campaign_verification_receipt_sha256": (
                verification_doc.source_sha256
            ),
            "campaign_verification_receipt_canonical_sha256": (
                verification_doc.canonical_sha256
            ),
            "verification_order": list(OPEN04_ROLES),
            "campaign_eligible": True,
            "promotion_status": "eligible",
            "no_retuning_status": verification["no_retuning_status"],
            "pair_gates": pair_gates,
        },
        "baseline": {
            "package_id": common["package_id"],
            "package_sha256": common["baseline_package_sha256"],
            "manifest_sha256": common["baseline_manifest_sha256"],
            "attestation_sha256": common[
                "release_attestation_sha256"
            ],
            "release_commit_sha": _BASELINE_RELEASE_COMMIT_SHA,
            "requirements_lock_sha256": (
                _BASELINE_REQUIREMENTS_LOCK_SHA256
            ),
        },
        "scenarios": scenarios,
        "exporter": {
            "code_commit_sha": exporter_identity["code_commit_sha"],
            "source_sha256": exporter_identity["source_sha256"],
            "identity_mode": exporter_identity["identity_mode"],
            "simulation_producer_code_commit_sha": common[
                "code_commit_sha"
            ],
            "simulation_producer_exporter_source_sha256": common[
                "open04_exporter_source_sha256"
            ],
            "requirements_lock_sha256": common[
                "requirements_lock_sha256"
            ],
            "projection_validation_status": "pass",
            "clock_reconciliation_status": "pass",
            "aggregation_identity_status": "pass",
            "pair_gate_reconciliation_status": "pass",
            "four_file_boundary_status": "pass",
        },
        "remote_execution": {
            "execution_order": list(OPEN04_ROLES),
            "scenario_concurrency_mode": "parallel_distinct_output_parents",
            "controller_output_isolation_status": pair_gates[
                "controller_output_isolation_status"
            ],
            "aggregate_memory_budget_status": pair_gates[
                "aggregate_memory_budget_status"
            ],
            "placement": next(
                iter(
                    {
                        projection.controller_receipt["placement"]
                        for projection in projections
                    }
                )
            ),
            "roles": remote_roles,
        },
        "copy_policy": {
            "baseline_package_copied": False,
            "bulk_run_tree_copied": False,
            "compiled_input_tree_copied": False,
            "source_outputs_copied": False,
            "source_outputs_retained_at_producer": True,
            "consumer_boundary_file_count": 4,
        },
        "producer_sources": producer_sources,
        "producer_source_manifest_sha256": canonical_json_sha256(
            producer_sources
        ),
        "thin_artifacts": artifacts,
        "thin_artifact_manifest_sha256": thin_output_manifest_sha256,
        "thin_output_manifest_sha256": thin_output_manifest_sha256,
        "consumer_copy_manifest": consumer_copy_manifest,
        "consumer_copy_manifest_sha256": (
            consumer_copy_manifest_sha256
        ),
        "consumer_copy_expectations": {
            "consumer_project": consumer["consumer_project"],
            "exact_files": expected_consumer_files,
            "unexpected_file_action": "reject",
        },
        "exact_file_boundary": {
            "expected_filenames": list(OPEN04_EXPORT_FILES),
            "producer_file_count": 4,
            "consumer_file_count": 4,
            "producer_boundary_status": "pass",
            "consumer_boundary_status": "pass",
            "csv_copy_hash_status": "pass",
            "receipt_copy_byte_identity_status": "pass",
            "receipt_self_hash_in_manifest": False,
            "receipt_self_hash_exclusion_reason": (
                "avoid_self_referential_manifest"
            ),
        },
        "receipt_authority": {
            "authority_rule": (
                "valid_only_when_both_promoted_four_file_boundaries_exist_"
                "and_all_same_named_files_are_byte_identical"
            ),
            "producer_copy_required": True,
            "consumer_copy_required": True,
            "consumer_promoted_before_producer": True,
            "receipt_copy_byte_identity_required": True,
            "consumer_copy_manifest_sha256": (
                consumer_copy_manifest_sha256
            ),
        },
    }


def _consumer_expectation(consumer_project: str) -> dict[str, str]:
    return {
        "consumer_project": _require_safe_id(
            consumer_project,
            label="consumer_project",
        )
    }


def _normalize_destinations(
    producer_output_dir: str | Path,
    consumer_output_dir: str | Path,
    *,
    protected_roots: Sequence[Path],
) -> tuple[Path, Path]:
    producer = Path(producer_output_dir).expanduser().resolve()
    consumer = Path(consumer_output_dir).expanduser().resolve()
    if producer == consumer:
        raise Open04ExportError(
            "producer and consumer destinations must be distinct"
        )
    if producer in consumer.parents or consumer in producer.parents:
        raise Open04ExportError(
            "producer and consumer destinations must not overlap"
        )
    for label, destination in (
        ("producer", producer),
        ("consumer", consumer),
    ):
        for protected in protected_roots:
            source = protected.resolve()
            if (
                destination == source
                or source in destination.parents
                or destination in source.parents
            ):
                raise Open04ExportError(
                    f"OPEN-04 {label} destination overlaps a campaign/run "
                    "source root"
                )
        if destination.exists():
            raise Open04ExportError(
                f"OPEN-04 {label} destination must not already exist: "
                f"{destination}"
            )
        if not destination.name or destination.name in {".", ".."}:
            raise Open04ExportError(
                f"OPEN-04 {label} destination is unsafe"
            )
    return producer, consumer


def _staging_path(output: Path) -> Path:
    return output.with_name(
        f".{output.name}.staging-{os.getpid()}-{uuid4().hex[:10]}"
    )


def _dual_artifact_record(
    producer_path: Path,
    consumer_path: Path,
    *,
    rows: int,
) -> dict[str, Any]:
    if producer_path.name != consumer_path.name:
        raise Open04ExportError(
            "producer and consumer artifact names differ"
        )
    expected_rows = _integer(
        rows,
        label=f"{producer_path.name} expected row count",
        minimum=0,
    )
    producer_sha = sha256_file(producer_path)
    consumer_sha = sha256_file(consumer_path)
    producer_bytes = producer_path.stat().st_size
    consumer_bytes = consumer_path.stat().st_size
    producer_rows = _csv_row_count(producer_path)
    consumer_rows = _csv_row_count(consumer_path)
    if (
        producer_sha != consumer_sha
        or producer_bytes != consumer_bytes
        or producer_rows != consumer_rows
        or producer_rows != expected_rows
    ):
        raise Open04ExportError(
            f"producer/consumer CSV copy mismatch: {producer_path.name}"
        )
    return {
        "logical_name": producer_path.name,
        "producer_path": producer_path.name,
        "producer_sha256": producer_sha,
        "producer_bytes": producer_bytes,
        "producer_rows": producer_rows,
        "consumer_path": consumer_path.name,
        "consumer_sha256": consumer_sha,
        "consumer_bytes": consumer_bytes,
        "consumer_rows": consumer_rows,
        "hash_match_status": "pass",
    }


def _document_source_record(
    document: _Document,
    *,
    logical_name: str,
) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "source_kind": document.source_kind,
        "relative_name": document.source_name,
        "sha256": document.source_sha256,
        "bytes": document.source_bytes,
        "canonical_sha256": document.canonical_sha256,
        "manifest_hash_match_status": "pass",
    }


def _artifact_record(
    path: Path,
    *,
    logical_name: str,
    relative_path: str,
    rows: int | None = None,
) -> dict[str, Any]:
    relative = _safe_relative(
        relative_path,
        label=f"{logical_name} retained source path",
    )
    record: dict[str, Any] = {
        "logical_name": _require_nonblank(
            logical_name,
            label="retained source logical name",
        ),
        "relative_path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "manifest_hash_match_status": "pass",
    }
    if rows is not None:
        record["rows"] = _integer(
            rows,
            label=f"{logical_name} rows",
            minimum=0,
        )
    return record


def _write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    expected = tuple(columns)
    if len(set(expected)) != len(expected) or not expected:
        raise Open04ExportError("CSV schema columns are duplicated or empty")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(expected),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            if set(row) != set(expected):
                raise Open04ExportError(
                    f"{path.name} row {index} fields differ from schema"
                )
            writer.writerow(
                {
                    column: _csv_value(row[column])
                    for column in expected
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Open04ExportError("thin CSV contains a nonfinite number")
        return format(0.0 if value == 0.0 else value, ".17g")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return value
    raise Open04ExportError(
        f"thin CSV contains an unsupported value type: {type(value).__name__}"
    )


def _csv_row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise Open04ExportError(f"{path.name} is empty")
            return sum(1 for _ in reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Open04ExportError(
            f"thin CSV is unreadable: {path.name}"
        ) from exc


def _require_identical_files(
    first: Path,
    second: Path,
    *,
    label: str,
) -> None:
    if (
        first.stat().st_size != second.stat().st_size
        or sha256_file(first) != sha256_file(second)
    ):
        raise Open04ExportError(f"{label} are not byte-identical")


def _require_exact_output_boundary(directory: Path) -> None:
    if not directory.is_dir() or directory.is_symlink():
        raise Open04ExportError(
            "OPEN-04 output boundary is not a real directory"
        )
    observed = {
        path.name
        for path in directory.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if observed != set(OPEN04_EXPORT_FILES):
        raise Open04ExportError(
            "OPEN-04 output boundary differs from the exact four files"
        )
    if any(
        not path.is_file() or path.is_symlink()
        for path in directory.iterdir()
    ):
        raise Open04ExportError(
            "OPEN-04 output boundary contains a non-file entry"
        )


def _require_promoted_receipt_authority(
    producer: Path,
    consumer: Path,
) -> None:
    _require_exact_output_boundary(producer)
    _require_exact_output_boundary(consumer)
    for filename in OPEN04_EXPORT_FILES:
        _require_identical_files(
            producer / filename,
            consumer / filename,
            label=f"promoted producer/consumer {filename} copies",
        )
    producer_receipt = read_json(producer / OPEN04_EXPORT_FILES[3])
    consumer_receipt = read_json(consumer / OPEN04_EXPORT_FILES[3])
    if (
        producer_receipt != consumer_receipt
        or not isinstance(producer_receipt, Mapping)
    ):
        raise Open04ExportError(
            "promoted producer/consumer receipt authority differs"
        )
    authority = _mapping(
        producer_receipt.get("receipt_authority"),
        label="promoted receipt authority",
    )
    if (
        authority.get("producer_copy_required") is not True
        or authority.get("consumer_copy_required") is not True
        or authority.get("consumer_promoted_before_producer") is not True
        or authority.get("receipt_copy_byte_identity_required") is not True
        or authority.get("consumer_copy_manifest_sha256")
        != producer_receipt.get("consumer_copy_manifest_sha256")
    ):
        raise Open04ExportError(
            "promoted receipt does not require both authoritative copies"
        )


def _fsync_files(directory: Path) -> None:
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        # Windows requires a writable handle for FlushFileBuffers, which is
        # the operation Python exposes as fsync.
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            # Windows does not support fsync on directory handles.  Every
            # file has already been fsynced above; the same-parent rename is
            # still the atomic visibility boundary on that platform.
            pass
    finally:
        os.close(descriptor)


def _remove_owned_staging(staging: Path, *, output: Path) -> None:
    if not staging.exists():
        return
    expected_prefix = f".{output.name}.staging-"
    if (
        staging.parent != output.parent
        or not staging.name.startswith(expected_prefix)
        or staging == output
        or staging.is_symlink()
    ):
        raise Open04ExportError(
            f"refusing unsafe staging cleanup: {staging}"
        )
    shutil.rmtree(staging)


def _remove_owned_output(output: Path) -> None:
    if not output.exists():
        return
    if not output.is_dir() or output.is_symlink() or not output.name:
        raise Open04ExportError(
            f"refusing unsafe promoted-output cleanup: {output}"
        )
    shutil.rmtree(output)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Open04ExportError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise Open04ExportError(f"{label} field names must be strings")
    return value


def _exact_mapping(
    value: Any,
    expected: set[str] | Sequence[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    mapping = _mapping(value, label=label)
    _require_exact_keys(mapping, set(expected), label=label)
    return mapping


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise Open04ExportError(
            f"{label} fields differ: missing={missing}, extra={extra}"
        )


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise Open04ExportError(f"{label} must be an array")
    return value


def _require_nonblank(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Open04ExportError(
            f"{label} must be a nonblank trimmed string"
        )
    return value


def _require_safe_id(value: Any, *, label: str) -> str:
    text = _require_nonblank(value, label=label)
    if (
        len(text) > 160
        or not text[0].isalnum()
        or any(
            character
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            )
            for character in text
        )
    ):
        raise Open04ExportError(f"{label} is not a safe identifier")
    return text


def _require_sha(value: Any, *, label: str) -> str:
    text = _require_nonblank(value, label=label)
    if (
        len(text) != 64
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise Open04ExportError(f"{label} must be a lowercase SHA-256")
    return text


def _require_git_sha(value: Any, *, label: str) -> str:
    text = _require_nonblank(value, label=label)
    if (
        len(text) != 40
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise Open04ExportError(f"{label} must be a lowercase full Git SHA")
    return text


def _safe_relative(value: Any, *, label: str) -> PurePosixPath:
    text = _require_nonblank(value, label=label)
    if (
        "\\" in text
        or ":" in text
        or text.startswith("/")
        or text.endswith("/")
    ):
        raise Open04ExportError(f"{label} must be a portable relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != text
    ):
        raise Open04ExportError(f"{label} is unsafe")
    return path


def _resolve_under(
    root: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> Path:
    base = root.resolve()
    candidate = base.joinpath(*relative.parts).resolve()
    if candidate == base or base not in candidate.parents:
        raise Open04ExportError(f"{label} escapes its declared root")
    return candidate


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Open04ExportError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Open04ExportError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise Open04ExportError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def _optional_finite(value: Any, *, label: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _finite(value, label=label)


def _integer(value: Any, *, label: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise Open04ExportError(f"{label} must be an integer")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if (
            not text
            or text != value
            or not text.lstrip("-").isdigit()
        ):
            raise Open04ExportError(f"{label} must be an integer")
        number = int(text)
    else:
        raise Open04ExportError(f"{label} must be an integer")
    if number < minimum:
        raise Open04ExportError(f"{label} is below its minimum")
    return number


def _boolean(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise Open04ExportError(f"{label} must be true or false")


def _iso_date(value: Any, *, label: str) -> str:
    text = _require_nonblank(value, label=label)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise Open04ExportError(
            f"{label} must be a canonical ISO-8601 date"
        ) from exc
    if parsed.isoformat() != text:
        raise Open04ExportError(
            f"{label} must be a canonical ISO-8601 date"
        )
    return text


def _utc_datetime(value: Any, *, label: str) -> datetime:
    text = _require_nonblank(value, label=label)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise Open04ExportError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise Open04ExportError(f"{label} must be UTC")
    return parsed


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    label: str,
    tolerance: float = _TOLERANCE,
) -> None:
    left = _finite(actual, label=f"{label} actual")
    right = _finite(expected, label=f"{label} expected")
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise Open04ExportError(
            f"{label} mismatch: actual={left!r}, expected={right!r}"
        )


__all__ = [
    "INPUT_CONTRACT_SCHEMA_VERSION",
    "MATURITY_COLUMNS",
    "MATURITY_SCHEMA_VERSION",
    "OPEN04_EXPORT_FILES",
    "Open04ExportError",
    "Open04ExportResult",
    "SCENARIO_INPUT_COLUMNS",
    "TDC_PATH_COLUMNS",
    "TDC_PATH_SCHEMA_VERSION",
    "export_open04_thin_package",
]
