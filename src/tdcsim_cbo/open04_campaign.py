"""Fail-closed OPEN-04 paired-campaign contracts and verification.

The campaign location and controller identity are deliberately kept outside the
canonical scenario documents.  A scenario carries only a stable typed marker;
the separately frozen campaign contract binds the three source documents,
compiled inputs, run basenames, calibration worksheet, and release identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

from evaluated_nominal_curve import (
    NOMINAL_EVALUATED_SHOCK_FILE,
    OPEN04_FIXED_COUPLING,
    OPEN04_OUTPUT_CONTRACT,
    normalize_open04_override,
)
from tdc_shared import MARKETABLE_PREFERENCE_CATEGORIES

from ._json import (
    canonical_json_sha256,
    read_json,
    sha256_file,
)
from .bounded_output import EVENT_SCHEMA_VERSION


OPEN04_CAMPAIGN_CONTRACT_ID = "open04_paired_tradeoff_v1"
OPEN04_CAMPAIGN_ROLES = ("baseline", "candidate_a", "candidate_b")
OPEN04_HOLDER_PROFILE_FILE = "tdcsim_holder_profile_assumptions.csv"
OPEN04_CANDIDATE_A_HOLDER_SHIFT_PP = 1.0
OPEN04_FUNDING_CLOSURE_MODE = (
    "cbo_debt_reference_plus_tga_floor_financing_v1"
)
OPEN04_ROLE_TO_SCENARIO_ID = MappingProxyType(
    {
        "baseline": "tdcsim_open04_paired_baseline_noop_v1",
        "candidate_a": (
            "tdcsim_open04_candidate_a_shorter_10y_down_"
            "bond_bank_substitution_1pp_v2"
        ),
        "candidate_b": (
            "tdcsim_open04_candidate_b_longer_10y_up_v1"
        ),
    }
)
OPEN04_BASELINE_IDENTITY = MappingProxyType(
    {
        "package_id": "cbo_full_horizon_local_smoke_3m",
        "package_sha256": (
            "be49f5a5d256863649ccf1b139d259679c9a3ce669642751c22e8a7dad9a2d2c"
        ),
        "manifest_sha256": (
            "4cf0d3571abcd0d80c91a28e23f9c4e00ce6075564f5354e3b12e5364727e16c"
        ),
        "release_attestation_sha256": (
            "d30a6f89263cdb80f8f9d81131dc0004b7b9751289e0594ee5005e115641124a"
        ),
    }
)
OPEN04_BASELINE_RELEASE_COMMIT_SHA = (
    "639c033a25b540f7c0b747ff05644e523e82abad"
)
OPEN04_BASELINE_REQUIREMENTS_LOCK_SHA256 = (
    "16e3fc32257a01e1fd2e5a53867cc9e73496b9d35c6a5d4db19c671617325a4e"
)

OPEN04_START_DATE = "2026-06-21"
OPEN04_END_DATE = "2036-09-30"
OPEN04_FREQUENCY = "daily"
OPEN04_AGGREGATION_CLOCK_ID = "federal_fiscal_year_period_end_v1"

OPEN04_CAMPAIGN_SCHEMA_VERSION = "tdcsim_open04_campaign_contract_v1"
OPEN04_PRE_RUN_RECEIPT_SCHEMA_VERSION = (
    "tdcsim_open04_campaign_pre_run_receipt_v1"
)
OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION = (
    "tdcsim_open04_campaign_verification_receipt_v1"
)
OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION = (
    "tdcsim_open04_controller_completion_receipt_v1"
)
OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION = (
    "tdcsim_open04_host_task_completion_receipt_v1"
)
OPEN04_CALIBRATION_WORKSHEET_SCHEMA_VERSION = (
    "tdcsim_open04_calibration_worksheet_v1"
)

OPEN04_NO_SIDECAR_SENTINEL = "none"
OPEN04_NO_CURVE_DELTA_SENTINEL = "none"
OPEN04_SIGN_GATE_THRESHOLD_BIL = 1e-6
OPEN04_ANTISYMMETRY_TOLERANCE_DECIMAL = 2e-12

OPEN04_ATI_PDF_SHA256 = (
    "78e23a700b02b781afa41338e0a8c455c20ec7e890bdd8c12824c80ab17864ee"
)
OPEN04_HOU_PDF_SHA256 = (
    "29a22354f29041c5f06f0d7277024631be466cb6c9e604da245fa60b53b7d5f8"
)

_SCENARIO_SOURCE_NAMES = MappingProxyType(
    {
        "baseline": "open04_baseline_scenario.json",
        "candidate_a": "open04_candidate_a_scenario.json",
        "candidate_b": "open04_candidate_b_scenario.json",
    }
)
_DEFAULT_CONTROLLER_RECEIPT_NAMES = MappingProxyType(
    {
        "baseline": "open04_baseline_controller_completion.json",
        "candidate_a": "open04_candidate_a_controller_completion.json",
        "candidate_b": "open04_candidate_b_controller_completion.json",
    }
)
_ROLE_SHOCK_BP = MappingProxyType(
    {"baseline": 0.0, "candidate_a": -25.0, "candidate_b": 25.0}
)
_ISSUANCE_MIX_FILE = "tdcsim_issuance_mix_assumptions.json"
_NOMINAL_SURFACE_FILE = "tdcsim_yield_curve_surface.csv"
_OPEN04_CAMPAIGN_MUTABLE_INPUTS = frozenset(
    {
        OPEN04_HOLDER_PROFILE_FILE,
        _ISSUANCE_MIX_FILE,
        NOMINAL_EVALUATED_SHOCK_FILE,
    }
)
_COMPILED_MANIFEST_FILE = "tdcsim_cbo_compiled_manifest.json"
_RUN_MANIFEST_FILE = "tdcsim_cbo_run_manifest.json"
_ANNUAL_OUTPUT_FILE = "tdcsim_annual_economic_summary.csv.gz"
_OPENING_INPUT_FILES = (
    "tdcsim_opening_portfolio.csv",
    "tdcsim_opening_runtime_state.json",
)
_CONTROLLER_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "campaign_contract_sha256",
        "role",
        "run_id",
        "host",
        "controller",
        "placement",
        "terminal_status",
        "started_at_utc",
        "completed_at_utc",
        "worker_exit_confirmed_at_utc",
        "terminal_summary_sha256",
        "run_manifest_sha256",
        "controller_run_id",
        "controller_summary_sha256",
        "controller_command_sha256",
        "controller_command_exit_code",
        "controller_exit_code",
        "preflight_conflicts",
        "postrun_conflicts",
        "controller_peak_rss_mb",
        "controller_avg_cpu_pct",
        "controller_memory_guard_status",
        "controller_telemetry_status",
        "controller_telemetry_locator",
        "controller_process_tree_drained",
    }
)

_GIB = 1024**3
_EXECUTION_CONTRACT = MappingProxyType(
    {
        "scenario_concurrency_mode": "parallel_distinct_output_parents",
        "maximum_concurrent_scenarios": 3,
        "aggregate_acceptance_peak_rss_bytes": 10 * _GIB,
        "controller_placement": "auto",
        "writer_claim_scope": "output_parent",
        "parent_watchdog_required": True,
        "process_pool_enabled": False,
        "numerical_thread_count": 1,
        "minimum_available_memory_bytes": 4 * _GIB,
        "acceptance_peak_rss_bytes": 6 * _GIB,
        "application_abort_rss_bytes": 8 * _GIB,
        "parent_graceful_stop_rss_bytes": 10 * _GIB,
        "parent_kill_rss_bytes": 12 * _GIB,
    }
)

_CODE_IDENTITY_KEYS = (
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
_POST_COMMON_IDENTITY_KEYS = (
    "package_id",
    "baseline_package_sha256",
    "baseline_manifest_sha256",
    "release_attestation_sha256",
    *_CODE_IDENTITY_KEYS,
    "start_date",
    "end_date",
    "frequency",
    "aggregation_clock_id",
    "opening_identity_sha256",
    "fixed_input_records_sha256",
    "calibration_worksheet_sha256",
)
_CONTRACT_COMMON_IDENTITY_KEYS = (
    *_POST_COMMON_IDENTITY_KEYS,
    "output_profile",
    "compression",
    "opening_input_records",
    "fixed_input_records",
)
_CONTRACT_ROLE_KEYS = (
    "scenario_id",
    "scenario_source_relative_path",
    "scenario_source_sha256",
    "scenario_sha256",
    "run_relative_path",
    "controller_completion_receipt_relative_path",
    "canonical_issuance_mix_sha256",
    "issuance_mix_file_sha256",
    "compiled_inputs_digest",
    "curve_sidecar_sha256",
    "signed_10y_shock_bp",
    "compiled_curve_delta_digest",
    "economic_changed_paths",
    "physical_changed_inputs",
    "fixed_input_records_sha256",
    "curve_date_set_sha256",
    "tenor_set_sha256",
    "overall_new_issuance_wam_years",
)
OPEN04_POST_COMMON_IDENTITY_KEYS = _POST_COMMON_IDENTITY_KEYS
OPEN04_CONTRACT_COMMON_IDENTITY_KEYS = (
    _CONTRACT_COMMON_IDENTITY_KEYS
)
OPEN04_CONTRACT_ROLE_KEYS = _CONTRACT_ROLE_KEYS
OPEN04_CONTROLLER_RECEIPT_KEYS = tuple(
    sorted(_CONTROLLER_RECEIPT_KEYS)
)
OPEN04_POST_ROLE_KEYS = (
    "scenario_id",
    "scenario_sha256",
    "scenario_source_sha256",
    "run_scenario_copy_sha256",
    "source_run_canonical_match_status",
    "compiled_inputs_digest",
    "canonical_issuance_mix_sha256",
    "issuance_mix_file_sha256",
    "curve_sidecar_sha256",
    "signed_10y_shock_bp",
    "compiled_curve_delta_digest",
    "run_id",
    "run_manifest_sha256",
    "run_relative_path",
    "host",
    "controller_completion_receipt_sha256",
    "terminal_summary_sha256",
    "source_output_manifest_sha256",
    "event_schema_version",
    "event_count",
    "event_root_sha256",
    "verification_status",
    "financing_cost_component_identity_status",
    "accounting_invariants_status",
    "deterministic_replay_status",
    "memory_watchdog_status",
    "terminal_cumulative_modeled_financing_cost_bil",
    "terminal_cumulative_tdc_change_bil",
    "overall_new_issuance_wam_years",
    "economic_changed_paths",
    "physical_changed_inputs",
    "fixed_input_comparison_status",
    "fixed_input_records_sha256",
    "curve_date_set_sha256",
    "tenor_set_sha256",
)
OPEN04_PAIR_GATE_KEYS = (
    "threshold_bil",
    "candidate_a_financing_cost_delta_bil",
    "candidate_a_tdc_delta_bil",
    "candidate_b_financing_cost_delta_bil",
    "candidate_b_tdc_delta_bil",
    "terminal_sign_gate_status",
    "overall_wam_ordering",
    "full_fy_wam_ordering",
    "full_fy_wam_gate_status",
    "controller_output_isolation_status",
    "aggregate_memory_budget_status",
    "evaluated_delta_antisymmetry_sha256",
    "maximum_abs_evaluated_delta_antisymmetry_error",
    "evaluated_delta_antisymmetry_status",
    "curve_date_set_equality_status",
    "tenor_set_equality_status",
    "change_perimeter_status",
    "fixed_input_equality_status",
)
OPEN04_POST_RECEIPT_KEYS = (
    "schema_version",
    "status",
    "campaign_eligible",
    "promotion_status",
    "campaign_id",
    "campaign_contract_sha256",
    "campaign_root_identity",
    "verification_order",
    "common_identity",
    "roles",
    "pair_gates",
    "no_retuning_status",
)
OPEN04_CONTRACT_TOP_LEVEL_KEYS = (
    "schema_version",
    "contract_id",
    "contract_status",
    "signature_reference",
    "campaign_id",
    "campaign_root_identity",
    "execution_order",
    "no_retuning",
    "failed_gate_action",
    "execution_contract",
    "calibration_worksheet",
    "common_identity",
    "roles",
)
OPEN04_CALIBRATION_WORKSHEET_REFERENCE_KEYS = (
    "relative_path",
    "sha256",
    "canonical_sha256",
)
OPEN04_EXECUTION_CONTRACT_KEYS = tuple(_EXECUTION_CONTRACT)

OPEN04_ROLE_TO_PROVENANCE = MappingProxyType(
    {
        "baseline": MappingProxyType(
            {
                "kind": "user_stress_assumption",
                "label": "common_release_bound_baseline_noop",
            }
        ),
        "candidate_a": MappingProxyType(
            {
                "kind": "user_stress_assumption",
                "label": (
                    "hand_specified_outcome_conditioned_private_to_banks_"
                    "new_20y_30y_nominal_bonds_1pp"
                ),
            }
        ),
        "candidate_b": MappingProxyType(
            {
                "kind": "user_stress_assumption",
                "label": (
                    "hand_specified_joint_maturity_curve_sensitivity"
                ),
            }
        ),
    }
)


def _canonical_candidate_a_holder_preferences() -> dict[str, Any]:
    return {
        "mode": "static_shares",
        "rows": [
            {
                "security_type": "bonds",
                "shares": {
                    "Banks": 0.11526315789473685,
                    "CB": 0.0,
                    "FedInternal": 0.0,
                    "Foreign": 0.2631578947368421,
                    "Private": 0.621578947368421,
                    "TrustFunds": 0.0,
                },
            }
        ],
    }


def open04_expected_change_perimeter(
    role: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the independently reconstructable role-specific change perimeter."""

    if role == "baseline":
        return (), ()
    if role == "candidate_a":
        return (
            (
                "issuance_mix",
                "nominal_yield_curve_assumption",
                "holder_preferences",
            ),
            (
                OPEN04_HOLDER_PROFILE_FILE,
                _ISSUANCE_MIX_FILE,
                NOMINAL_EVALUATED_SHOCK_FILE,
            ),
        )
    if role == "candidate_b":
        return (
            ("issuance_mix", "nominal_yield_curve_assumption"),
            (_ISSUANCE_MIX_FILE, NOMINAL_EVALUATED_SHOCK_FILE),
        )
    raise Open04CampaignError(f"unsupported OPEN-04 role: {role!r}")


def _canonical_candidate_issuance_mix(role: str) -> dict[str, Any]:
    if role == "candidate_a":
        return {
            "mode": "replace_shares",
            "tips_share": 0.06,
            "frn_share": 0.04,
            "fixed_remainder_shares": {
                "bills": 0.335030,
                "notes": 0.507485,
                "bonds": 0.157485,
            },
            "maturity_distributions": {
                "bills": [
                    {"maturity_years": 0.25, "share": 0.141717},
                    {"maturity_years": 0.5, "share": 0.858283},
                ],
                "notes": [
                    {"maturity_years": 2.0, "share": 0.198404},
                    {"maturity_years": 5.0, "share": 0.801596},
                ],
                "bonds": [
                    {"maturity_years": 20.0, "share": 0.858283},
                    {"maturity_years": 30.0, "share": 0.141717},
                ],
                "tips": [{"maturity_years": 10.0, "share": 1.0}],
                "frn": [{"maturity_years": 2.0, "share": 1.0}],
            },
            "negative_issuance_action": "error",
        }
    if role == "candidate_b":
        return {
            "mode": "replace_shares",
            "tips_share": 0.06,
            "frn_share": 0.04,
            "fixed_remainder_shares": {
                "bills": 0.228928,
                "notes": 0.535952,
                "bonds": 0.235120,
            },
            "maturity_distributions": {
                "bills": [{"maturity_years": 0.5, "share": 1.0}],
                "notes": [
                    {"maturity_years": 5.0, "share": 0.859522},
                    {"maturity_years": 7.0, "share": 0.063215},
                    {"maturity_years": 10.0, "share": 0.077263},
                ],
                "bonds": [
                    {"maturity_years": 20.0, "share": 0.908689},
                    {"maturity_years": 30.0, "share": 0.091311},
                ],
                "tips": [{"maturity_years": 10.0, "share": 1.0}],
                "frn": [{"maturity_years": 2.0, "share": 1.0}],
            },
            "negative_issuance_action": "error",
        }
    raise Open04CampaignError(
        f"OPEN-04 role has no issuance-mix override: {role!r}"
    )


OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256 = MappingProxyType(
    {
        "baseline": OPEN04_NO_SIDECAR_SENTINEL,
        "candidate_a": canonical_json_sha256(
            _canonical_candidate_issuance_mix("candidate_a")
        ),
        "candidate_b": canonical_json_sha256(
            _canonical_candidate_issuance_mix("candidate_b")
        ),
    }
)


class Open04CampaignError(ValueError):
    """Raised when an OPEN-04 campaign contract fails closed."""


@dataclass(frozen=True)
class Open04CampaignMarker:
    """Stable scenario-level marker; runtime campaign paths are not included."""

    contract_id: str
    role: str
    funding_closure_mode: str


def parse_open04_campaign_marker(
    data: Mapping[str, Any],
) -> Open04CampaignMarker | None:
    """Parse the exact typed marker and bind its role to the scenario ID."""

    scenario = _require_mapping(data, label="scenario")
    raw = scenario.get("open04_campaign")
    if raw is None:
        return None
    marker = _require_mapping(raw, label="scenario.open04_campaign")
    allowed_fields = {"contract_id", "role", "funding_closure_mode"}
    if set(marker) != allowed_fields:
        raise Open04CampaignError(
            "scenario.open04_campaign has missing or unknown fields"
        )
    contract_id = str(marker["contract_id"])
    role = str(marker["role"])
    funding_closure_mode = str(marker["funding_closure_mode"])
    if funding_closure_mode != OPEN04_FUNDING_CLOSURE_MODE:
        raise Open04CampaignError(
            "scenario.open04_campaign.funding_closure_mode must use the "
            "OPEN-04 reference-plus-TGA closure"
        )
    if contract_id != OPEN04_CAMPAIGN_CONTRACT_ID:
        raise Open04CampaignError(
            "scenario.open04_campaign.contract_id is unsupported"
        )
    if role not in OPEN04_CAMPAIGN_ROLES:
        raise Open04CampaignError(
            "scenario.open04_campaign.role is unsupported"
        )
    expected_id = OPEN04_ROLE_TO_SCENARIO_ID[role]
    if scenario.get("scenario_id") != expected_id:
        raise Open04CampaignError(
            "OPEN-04 campaign role does not match its canonical scenario ID"
        )
    return Open04CampaignMarker(
        contract_id=contract_id,
        role=role,
        funding_closure_mode=funding_closure_mode,
    )


def requires_open04_strict_execution(data: Mapping[str, Any]) -> bool:
    """Return whether a scenario requires the strict OPEN-04 runtime contract."""

    scenario = _require_mapping(data, label="scenario")
    marker = parse_open04_campaign_marker(scenario)
    if marker is not None:
        return True
    overrides = scenario.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise Open04CampaignError("scenario.overrides must be an object")
    nominal = overrides.get("nominal_yield_curve")
    return (
        isinstance(nominal, Mapping)
        and nominal.get("mode") == "evaluated_additive_key_rate_bp"
    )


def validate_open04_scenario_contract(
    data: Mapping[str, Any],
) -> Open04CampaignMarker:
    """Validate the exact three-path scenario-level economic perimeter."""

    scenario = _require_mapping(data, label="scenario")
    marker = parse_open04_campaign_marker(scenario)
    if marker is None:
        raise Open04CampaignError("OPEN-04 campaign marker is required")
    _require_exact_fields(
        scenario,
        {
            "schema_version",
            "scenario_id",
            "baseline",
            "provenance",
            "simulation",
            "open04_campaign",
            "coupling",
            "overrides",
            "output",
        },
        label="OPEN-04 scenario",
    )
    if scenario.get("schema_version") != "tdcsim_cbo_scenario_v1":
        raise Open04CampaignError("OPEN-04 scenario schema_version is invalid")
    provenance = _require_mapping(
        scenario.get("provenance"), label="OPEN-04 scenario.provenance"
    )
    if dict(provenance) != dict(
        OPEN04_ROLE_TO_PROVENANCE[marker.role]
    ):
        raise Open04CampaignError(
            "OPEN-04 scenario provenance is not the frozen role-specific "
            "assumption label"
        )

    baseline = _require_mapping(
        scenario.get("baseline"), label="OPEN-04 scenario.baseline"
    )
    _require_exact_fields(
        baseline,
        set(OPEN04_BASELINE_IDENTITY),
        label="OPEN-04 scenario.baseline",
    )
    if dict(baseline) != dict(OPEN04_BASELINE_IDENTITY):
        raise Open04CampaignError(
            "OPEN-04 scenario does not bind the authoritative baseline identity"
        )

    simulation = _require_mapping(
        scenario.get("simulation"), label="OPEN-04 scenario.simulation"
    )
    expected_simulation = {
        "frequency": OPEN04_FREQUENCY,
        "start_date": OPEN04_START_DATE,
        "end_date": OPEN04_END_DATE,
    }
    if dict(simulation) != expected_simulation:
        raise Open04CampaignError(
            "OPEN-04 scenario must use the exact full daily horizon"
        )
    coupling = _require_mapping(
        scenario.get("coupling"), label="OPEN-04 scenario.coupling"
    )
    if dict(coupling) != dict(OPEN04_FIXED_COUPLING):
        raise Open04CampaignError(
            "OPEN-04 scenario coupling is outside the fixed perimeter"
        )
    output = _require_mapping(
        scenario.get("output"), label="OPEN-04 scenario.output"
    )
    if dict(output) != dict(OPEN04_OUTPUT_CONTRACT):
        raise Open04CampaignError(
            "OPEN-04 scenario output must be compact gzip without SQLite"
        )

    overrides = _require_mapping(
        scenario.get("overrides"), label="OPEN-04 scenario.overrides"
    )
    if marker.role == "baseline":
        if overrides:
            raise Open04CampaignError(
                "OPEN-04 baseline must have no scenario overrides"
            )
        return marker

    expected_override_names = {"issuance_mix", "nominal_yield_curve"}
    if marker.role == "candidate_a":
        expected_override_names.add("holder_preferences")
    if set(overrides) != expected_override_names:
        raise Open04CampaignError(
            "OPEN-04 candidate overrides differ from the approved "
            "role-specific perimeter"
        )
    issuance_mix = _require_mapping(
        overrides.get("issuance_mix"),
        label="OPEN-04 scenario.overrides.issuance_mix",
    )
    expected_mix = _canonical_candidate_issuance_mix(marker.role)
    if dict(issuance_mix) != expected_mix:
        raise Open04CampaignError(
            "OPEN-04 candidate issuance mix differs from its frozen "
            "role-specific hand input"
        )
    if (
        canonical_json_sha256(issuance_mix)
        != OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256[marker.role]
    ):
        raise Open04CampaignError(
            "OPEN-04 candidate issuance-mix canonical hash mismatch"
        )
    try:
        nominal = normalize_open04_override(
            _require_mapping(
                overrides.get("nominal_yield_curve"),
                label="OPEN-04 scenario.overrides.nominal_yield_curve",
            )
        )
    except ValueError as exc:
        raise Open04CampaignError(str(exc)) from exc
    shock = float(nominal["shocks"][1]["shock_bp"])
    if shock != _ROLE_SHOCK_BP[marker.role]:
        raise Open04CampaignError(
            "OPEN-04 candidate curve sign does not match its campaign role"
        )
    if marker.role == "candidate_a":
        holder_preferences = _require_mapping(
            overrides.get("holder_preferences"),
            label="OPEN-04 scenario.overrides.holder_preferences",
        )
        if dict(holder_preferences) != _canonical_candidate_a_holder_preferences():
            raise Open04CampaignError(
                "OPEN-04 Candidate A holder condition differs from the "
                "approved one-point Private-to-Banks bond substitution"
            )
    return marker


def build_open04_scenario_mappings(
    *,
    baseline_identity: Mapping[str, Any],
    candidate_a_issuance_mix: Mapping[str, Any],
    candidate_b_issuance_mix: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the three deterministic canonical scenario mappings.

    Campaign ID, campaign root, controller placement, and run paths are
    intentionally absent so environment changes cannot alter scenario hashes.
    """

    declared_baseline = _require_mapping(
        baseline_identity, label="baseline_identity"
    )
    if dict(declared_baseline) != dict(OPEN04_BASELINE_IDENTITY):
        raise Open04CampaignError(
            "OPEN-04 scenarios require the authoritative baseline identity"
        )
    mix_by_role = {
        "candidate_a": deepcopy(
            dict(
                _require_mapping(
                    candidate_a_issuance_mix,
                    label="candidate_a_issuance_mix",
                )
            )
        ),
        "candidate_b": deepcopy(
            dict(
                _require_mapping(
                    candidate_b_issuance_mix,
                    label="candidate_b_issuance_mix",
                )
            )
        ),
    }
    for role, mix in mix_by_role.items():
        expected_mix = _canonical_candidate_issuance_mix(role)
        if mix != expected_mix:
            raise Open04CampaignError(
                f"{role}_issuance_mix differs from the frozen OPEN-04 input"
            )
        if (
            canonical_json_sha256(mix)
            != OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256[role]
        ):
            raise Open04CampaignError(
                f"{role}_issuance_mix canonical hash mismatch"
            )
    scenarios: dict[str, dict[str, Any]] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        if role == "baseline":
            overrides: dict[str, Any] = {}
        else:
            overrides = {
                "issuance_mix": mix_by_role[role],
                "nominal_yield_curve": _evaluated_nominal_override(
                    _ROLE_SHOCK_BP[role]
                ),
            }
            if role == "candidate_a":
                overrides["holder_preferences"] = (
                    _canonical_candidate_a_holder_preferences()
                )
        scenario = {
            "schema_version": "tdcsim_cbo_scenario_v1",
            "scenario_id": OPEN04_ROLE_TO_SCENARIO_ID[role],
            "baseline": dict(declared_baseline),
            "provenance": dict(OPEN04_ROLE_TO_PROVENANCE[role]),
            "simulation": {
                "frequency": OPEN04_FREQUENCY,
                "start_date": OPEN04_START_DATE,
                "end_date": OPEN04_END_DATE,
            },
            "open04_campaign": {
                "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
                "role": role,
                "funding_closure_mode": OPEN04_FUNDING_CLOSURE_MODE,
            },
            "coupling": dict(OPEN04_FIXED_COUPLING),
            "overrides": overrides,
            "output": dict(OPEN04_OUTPUT_CONTRACT),
        }
        validate_open04_scenario_contract(scenario)
        scenarios[role] = scenario
    return scenarios


def campaign_root_identity(
    campaign_id: str,
    role_to_run_relative_path: Mapping[str, str],
) -> str:
    """Hash the logical campaign/run mapping without a host-local root path."""

    normalized = _role_mapping(
        role_to_run_relative_path,
        label="role_to_run_relative_path",
    )
    values = list(normalized.values())
    if len(set(values)) != len(values):
        raise Open04CampaignError(
            "OPEN-04 run_relative_path basenames must be unique"
        )
    for role, relative in normalized.items():
        _role_run_relative_path(relative, label=f"{role}.run_relative_path")
    identifier = _campaign_id(campaign_id)
    return canonical_json_sha256(
        {
            "campaign_id": identifier,
            "role_to_run_relative_path": normalized,
        }
    )


def freeze_open04_campaign_contract(
    campaign_root: str | Path,
    *,
    campaign_id: str,
    signature_reference: str,
    scenarios_by_role: Mapping[str, Mapping[str, Any]],
    compiled_dirs_by_role: Mapping[str, str | Path],
    code_identity: Mapping[str, Any],
    role_to_run_relative_path: Mapping[str, str],
    canonical_issuance_mix_sha256_by_role: Mapping[str, str],
    role_to_controller_completion_receipt_relative_path: (
        Mapping[str, str] | None
    ) = None,
    contract_relative_path: str = "open04_campaign_contract.json",
    calibration_worksheet_relative_path: str = (
        "open04_calibration_worksheet.json"
    ),
) -> dict[str, Any]:
    """Freeze the deterministic pre-output campaign inputs and contract.

    The caller supplies the three canonical issuance-mix hashes; this function
    verifies them against already-compiled input bytes and never derives a
    replacement mix.  Compilation is not performed here.
    """

    root = Path(campaign_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identifier = _campaign_id(campaign_id)
    signature = str(signature_reference).strip()
    if not signature:
        raise Open04CampaignError(
            "OPEN-04 frozen contract requires an owner signature reference"
        )

    scenarios = _role_mapping(
        scenarios_by_role, label="scenarios_by_role"
    )
    normalized_scenarios: dict[str, dict[str, Any]] = {}
    for role, raw in scenarios.items():
        scenario = deepcopy(
            dict(_require_mapping(raw, label=f"scenarios_by_role.{role}"))
        )
        marker = validate_open04_scenario_contract(scenario)
        if marker.role != role:
            raise Open04CampaignError(
                f"scenario mapping role mismatch for {role}"
            )
        normalized_scenarios[role] = scenario

    compiled_dirs = {
        role: Path(value).expanduser().resolve()
        for role, value in _role_mapping(
            compiled_dirs_by_role, label="compiled_dirs_by_role"
        ).items()
    }
    run_paths = {
        role: _role_run_relative_path(
            value, label=f"{role}.run_relative_path"
        )
        for role, value in _role_mapping(
            role_to_run_relative_path,
            label="role_to_run_relative_path",
        ).items()
    }
    if len(set(run_paths.values())) != len(OPEN04_CAMPAIGN_ROLES):
        raise Open04CampaignError(
            "OPEN-04 run_relative_path basenames must be unique"
        )
    for role, relative in run_paths.items():
        if (root / relative).exists():
            raise Open04CampaignError(
                f"cannot freeze OPEN-04 after run output exists for {role}"
            )
    receipt_names_raw = (
        role_to_controller_completion_receipt_relative_path
        if role_to_controller_completion_receipt_relative_path is not None
        else _DEFAULT_CONTROLLER_RECEIPT_NAMES
    )
    receipt_names = {
        role: _direct_child_name(
            value,
            label=f"{role}.controller_completion_receipt_relative_path",
        )
        for role, value in _role_mapping(
            receipt_names_raw,
            label=(
                "role_to_controller_completion_receipt_relative_path"
            ),
        ).items()
    }
    if len(set(receipt_names.values())) != len(OPEN04_CAMPAIGN_ROLES):
        raise Open04CampaignError(
            "OPEN-04 controller receipt basenames must be unique"
        )
    mix_hashes = {
        role: _sha256(
            value, label=f"{role}.canonical_issuance_mix_sha256"
        )
        for role, value in _role_mapping(
            canonical_issuance_mix_sha256_by_role,
            label="canonical_issuance_mix_sha256_by_role",
        ).items()
    }
    code = _validate_code_identity(code_identity)
    compiled = _compiled_campaign_evidence(
        compiled_dirs,
        normalized_scenarios,
        expected_mix_hashes=mix_hashes,
    )

    worksheet_name = _direct_child_name(
        calibration_worksheet_relative_path,
        label="calibration_worksheet_relative_path",
    )
    contract_name = _direct_child_name(
        contract_relative_path, label="contract_relative_path"
    )
    reserved = {
        *run_paths.values(),
        *receipt_names.values(),
        *_SCENARIO_SOURCE_NAMES.values(),
        worksheet_name,
        contract_name,
    }
    expected_reserved_count = (
        len(run_paths)
        + len(receipt_names)
        + len(_SCENARIO_SOURCE_NAMES)
        + 2
    )
    if len(reserved) != expected_reserved_count:
        raise Open04CampaignError(
            "OPEN-04 campaign artifact basenames must be distinct"
        )

    worksheet = _calibration_worksheet(identifier)
    worksheet_path = root / worksheet_name
    _write_new_or_identical_json(worksheet_path, worksheet)
    worksheet_sha = sha256_file(worksheet_path)
    worksheet_canonical_sha = canonical_json_sha256(worksheet)

    source_hashes: dict[str, str] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        source_path = root / _SCENARIO_SOURCE_NAMES[role]
        _write_new_or_identical_json(
            source_path, normalized_scenarios[role]
        )
        source_hashes[role] = sha256_file(source_path)

    common_identity = {
        "package_id": OPEN04_BASELINE_IDENTITY["package_id"],
        "baseline_package_sha256": OPEN04_BASELINE_IDENTITY[
            "package_sha256"
        ],
        "baseline_manifest_sha256": OPEN04_BASELINE_IDENTITY[
            "manifest_sha256"
        ],
        "release_attestation_sha256": OPEN04_BASELINE_IDENTITY[
            "release_attestation_sha256"
        ],
        **code,
        "start_date": OPEN04_START_DATE,
        "end_date": OPEN04_END_DATE,
        "frequency": OPEN04_FREQUENCY,
        "aggregation_clock_id": OPEN04_AGGREGATION_CLOCK_ID,
        "opening_identity_sha256": compiled[
            "opening_identity_sha256"
        ],
        "fixed_input_records_sha256": compiled[
            "fixed_input_records_sha256"
        ],
        "calibration_worksheet_sha256": worksheet_sha,
        "output_profile": OPEN04_OUTPUT_CONTRACT["profile"],
        "compression": OPEN04_OUTPUT_CONTRACT["compression"],
        "opening_input_records": compiled["opening_input_records"],
        "fixed_input_records": compiled["fixed_input_records"],
    }
    _require_exact_fields(
        common_identity,
        set(_CONTRACT_COMMON_IDENTITY_KEYS),
        label="campaign common_identity",
    )

    role_contracts: dict[str, dict[str, Any]] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        evidence = compiled["roles"][role]
        role_contracts[role] = {
            "scenario_id": OPEN04_ROLE_TO_SCENARIO_ID[role],
            "scenario_source_relative_path": _SCENARIO_SOURCE_NAMES[
                role
            ],
            "scenario_source_sha256": source_hashes[role],
            "scenario_sha256": canonical_json_sha256(
                normalized_scenarios[role]
            ),
            "run_relative_path": run_paths[role],
            "controller_completion_receipt_relative_path": (
                receipt_names[role]
            ),
            "canonical_issuance_mix_sha256": mix_hashes[role],
            "issuance_mix_file_sha256": evidence[
                "issuance_mix_file_sha256"
            ],
            "compiled_inputs_digest": evidence[
                "compiled_inputs_digest"
            ],
            "curve_sidecar_sha256": evidence[
                "curve_sidecar_sha256"
            ],
            "signed_10y_shock_bp": evidence["signed_10y_shock_bp"],
            "compiled_curve_delta_digest": evidence[
                "compiled_curve_delta_digest"
            ],
            "economic_changed_paths": evidence[
                "economic_changed_paths"
            ],
            "physical_changed_inputs": evidence[
                "physical_changed_inputs"
            ],
            "fixed_input_records_sha256": evidence[
                "fixed_input_records_sha256"
            ],
            "curve_date_set_sha256": evidence[
                "curve_date_set_sha256"
            ],
            "tenor_set_sha256": evidence["tenor_set_sha256"],
            "overall_new_issuance_wam_years": evidence[
                "overall_new_issuance_wam_years"
            ],
        }

    root_digest = campaign_root_identity(identifier, run_paths)
    contract = {
        "schema_version": OPEN04_CAMPAIGN_SCHEMA_VERSION,
        "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
        "contract_status": "owner_signed_frozen",
        "signature_reference": signature,
        "campaign_id": identifier,
        "campaign_root_identity": root_digest,
        "execution_order": list(OPEN04_CAMPAIGN_ROLES),
        "no_retuning": True,
        "failed_gate_action": "failed_predeclared_paired_sign_gate",
        "execution_contract": dict(_EXECUTION_CONTRACT),
        "calibration_worksheet": {
            "relative_path": worksheet_name,
            "sha256": worksheet_sha,
            "canonical_sha256": worksheet_canonical_sha,
        },
        "common_identity": common_identity,
        "roles": role_contracts,
    }
    _validate_campaign_contract_mapping(root, contract)
    _write_new_or_identical_json(root / contract_name, contract)
    return contract


def verify_open04_campaign_pre_run(
    campaign_root: str | Path,
    *,
    expected_contract_sha256: str,
    compiled_dirs_by_role: Mapping[str, str | Path],
    contract_relative_path: str = "open04_campaign_contract.json",
) -> dict[str, Any]:
    """Verify the frozen contract and compiled inputs without granting eligibility."""

    root, contract, contract_sha = _load_campaign_contract(
        campaign_root,
        expected_contract_sha256=expected_contract_sha256,
        contract_relative_path=contract_relative_path,
    )
    scenarios = _scenario_sources(root, contract)
    compiled_dirs = {
        role: Path(value).expanduser().resolve()
        for role, value in _role_mapping(
            compiled_dirs_by_role, label="compiled_dirs_by_role"
        ).items()
    }
    mix_hashes = {
        role: contract["roles"][role][
            "canonical_issuance_mix_sha256"
        ]
        for role in OPEN04_CAMPAIGN_ROLES
    }
    compiled = _compiled_campaign_evidence(
        compiled_dirs,
        scenarios,
        expected_mix_hashes=mix_hashes,
    )
    _compare_compiled_evidence_to_contract(contract, compiled)
    common = _post_common_identity(contract["common_identity"])
    roles: dict[str, dict[str, Any]] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        declared = contract["roles"][role]
        evidence = compiled["roles"][role]
        roles[role] = {
            "scenario_id": declared["scenario_id"],
            "scenario_sha256": declared["scenario_sha256"],
            "scenario_source_sha256": declared[
                "scenario_source_sha256"
            ],
            "compiled_inputs_digest": evidence[
                "compiled_inputs_digest"
            ],
            "canonical_issuance_mix_sha256": evidence[
                "canonical_issuance_mix_sha256"
            ],
            "issuance_mix_file_sha256": evidence[
                "issuance_mix_file_sha256"
            ],
            "curve_sidecar_sha256": evidence[
                "curve_sidecar_sha256"
            ],
            "signed_10y_shock_bp": evidence["signed_10y_shock_bp"],
            "compiled_curve_delta_digest": evidence[
                "compiled_curve_delta_digest"
            ],
            "economic_changed_paths": evidence[
                "economic_changed_paths"
            ],
            "physical_changed_inputs": evidence[
                "physical_changed_inputs"
            ],
            "fixed_input_comparison_status": "pass",
            "fixed_input_records_sha256": evidence[
                "fixed_input_records_sha256"
            ],
            "curve_date_set_sha256": evidence[
                "curve_date_set_sha256"
            ],
            "tenor_set_sha256": evidence["tenor_set_sha256"],
            "overall_new_issuance_wam_years": evidence[
                "overall_new_issuance_wam_years"
            ],
            "verification_status": "pass",
        }
    return {
        "schema_version": OPEN04_PRE_RUN_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "campaign_eligible": False,
        "promotion_status": "not_eligible_pre_run_only",
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract_sha,
        "campaign_root_identity": contract["campaign_root_identity"],
        "verification_order": list(OPEN04_CAMPAIGN_ROLES),
        "common_identity": common,
        "roles": roles,
        "no_retuning_status": "pass",
        "execution_status": "not_started",
    }


def validate_open04_campaign_contract(
    campaign_root: str | Path,
    *,
    expected_contract_sha256: str,
    contract_relative_path: str = "open04_campaign_contract.json",
) -> dict[str, Any]:
    """Load a frozen contract only through its external SHA-256 anchor."""

    _root, contract, _actual_sha = _load_campaign_contract(
        campaign_root,
        expected_contract_sha256=expected_contract_sha256,
        contract_relative_path=contract_relative_path,
    )
    return contract


def validate_open04_campaign_contract_mapping(
    campaign_root: str | Path,
    contract: Mapping[str, Any],
    *,
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Validate an in-memory contract using its deterministic file-byte hash."""

    root = Path(campaign_root).expanduser().resolve()
    normalized = dict(
        _require_mapping(contract, label="OPEN-04 campaign contract")
    )
    expected = _sha256(
        expected_contract_sha256, label="expected_contract_sha256"
    )
    if _deterministic_json_file_sha256(normalized) != expected:
        raise Open04CampaignError(
            "in-memory OPEN-04 contract differs from its frozen byte hash"
        )
    _validate_campaign_contract_mapping(root, normalized)
    return normalized


def verify_open04_campaign_post_run(
    campaign_root: str | Path,
    *,
    expected_contract_sha256: str,
    baseline_package: str | Path,
    attestation: str | Path,
    contract_relative_path: str = "open04_campaign_contract.json",
) -> dict[str, Any]:
    """Recompute all three verified runs and the predeclared pair gates."""

    # Local import avoids a cycle because verifier imports the marker helpers.
    from .verifier import _verify_code_environment, verify_scenario_run

    root, contract, contract_sha = _load_campaign_contract(
        campaign_root,
        expected_contract_sha256=expected_contract_sha256,
        contract_relative_path=contract_relative_path,
    )
    scenarios = _scenario_sources(root, contract)
    compiled_dirs: dict[str, Path] = {}
    run_evidence: dict[str, dict[str, Any]] = {}
    controller_receipts: dict[str, dict[str, Any]] = {}

    for role in OPEN04_CAMPAIGN_ROLES:
        declared = contract["roles"][role]
        run_root = root / declared["run_relative_path"]
        if not run_root.is_dir() or run_root.parent.parent != root:
            raise Open04CampaignError(
                f"{role} run is not in its declared isolated role parent"
            )
        manifest_path = run_root / _RUN_MANIFEST_FILE
        if not manifest_path.is_file():
            raise Open04CampaignError(f"{role} run manifest is missing")
        manifest = dict(
            _require_mapping(
                read_json(manifest_path), label=f"{role} run manifest"
            )
        )
        verification = verify_scenario_run(
            run_root,
            baseline_package=baseline_package,
            attestation=attestation,
        )
        if (
            not isinstance(verification, Mapping)
            or verification.get("status") != "pass"
            or verification.get("verification_grade")
            != "bounded_replay_v1"
        ):
            raise Open04CampaignError(
                f"{role} did not pass bounded replay verification"
            )
        # This explicit call keeps the no-sidecar baseline on the same strict
        # release-identity gate even if a future generic verifier regresses.
        _verify_code_environment(
            run_root, manifest, require_release_identity=True
        )
        compiled_relative = _safe_relative_file(
            manifest.get("compiled_manifest"),
            label=f"{role} run compiled_manifest",
        )
        compiled_manifest_path = run_root / compiled_relative
        if (
            compiled_manifest_path.name != _COMPILED_MANIFEST_FILE
            or not compiled_manifest_path.is_file()
        ):
            raise Open04CampaignError(
                f"{role} run compiled manifest path differs"
            )
        compiled_dirs[role] = compiled_manifest_path.parent
        run_evidence[role] = _run_role_evidence(
            role,
            root,
            run_root,
            manifest_path,
            manifest,
            scenarios[role],
            contract,
            verification,
        )
        receipt_path = (
            root
            / declared[
                "controller_completion_receipt_relative_path"
            ]
        )
        controller_receipts[role] = _controller_completion_receipt(
            receipt_path,
            role=role,
            campaign_id=contract["campaign_id"],
            campaign_contract_sha256=contract_sha,
            run_id=run_evidence[role]["run_id"],
            run_manifest_sha256=run_evidence[role][
                "run_manifest_sha256"
            ],
            expected_terminal_summary_sha256=run_evidence[role][
                "terminal_summary_sha256"
            ],
        )

    compiled = _compiled_campaign_evidence(
        compiled_dirs,
        scenarios,
        expected_mix_hashes={
            role: contract["roles"][role][
                "canonical_issuance_mix_sha256"
            ]
            for role in OPEN04_CAMPAIGN_ROLES
        },
    )
    _compare_compiled_evidence_to_contract(contract, compiled)
    _verify_controller_resource_contract(controller_receipts, contract)

    common = _post_common_identity(contract["common_identity"])
    runtime_python_versions = {
        str(run_evidence[role]["common_identity"]["python_version"])
        for role in OPEN04_CAMPAIGN_ROLES
    }
    if len(runtime_python_versions) != 1:
        raise Open04CampaignError(
            "OPEN-04 runs do not share one Python runtime version"
        )
    # The frozen contract binds the pre-run/compiler environment.  A
    # platform-independent wheel may execute remotely under a different
    # Python runtime, which is a post-run provenance fact rather than an
    # economic input.  The strict per-run release check above and the common
    # identity comparison below still bind every other producer field.
    common["python_version"] = runtime_python_versions.pop()
    for role in OPEN04_CAMPAIGN_ROLES:
        if run_evidence[role]["common_identity"] != common:
            raise Open04CampaignError(
                f"{role} run common producer/input identity differs"
            )

    pair_gates = _pair_gates(
        run_evidence,
        compiled["roles"],
        controller_receipts,
    )
    roles: dict[str, dict[str, Any]] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        run = run_evidence[role]
        pre = compiled["roles"][role]
        controller = controller_receipts[role]
        roles[role] = {
            "scenario_id": run["scenario_id"],
            "scenario_sha256": run["scenario_sha256"],
            "scenario_source_sha256": run[
                "scenario_source_sha256"
            ],
            "run_scenario_copy_sha256": run[
                "run_scenario_copy_sha256"
            ],
            "source_run_canonical_match_status": "pass",
            "compiled_inputs_digest": pre["compiled_inputs_digest"],
            "canonical_issuance_mix_sha256": pre[
                "canonical_issuance_mix_sha256"
            ],
            "issuance_mix_file_sha256": pre[
                "issuance_mix_file_sha256"
            ],
            "curve_sidecar_sha256": pre["curve_sidecar_sha256"],
            "signed_10y_shock_bp": pre["signed_10y_shock_bp"],
            "compiled_curve_delta_digest": pre[
                "compiled_curve_delta_digest"
            ],
            "run_id": run["run_id"],
            "run_manifest_sha256": run["run_manifest_sha256"],
            "run_relative_path": contract["roles"][role][
                "run_relative_path"
            ],
            "host": controller["host"],
            "controller_completion_receipt_sha256": controller[
                "_receipt_sha256"
            ],
            "terminal_summary_sha256": controller[
                "terminal_summary_sha256"
            ],
            "source_output_manifest_sha256": run[
                "source_output_manifest_sha256"
            ],
            "event_schema_version": run["event_schema_version"],
            "event_count": run["event_count"],
            "event_root_sha256": run["event_root_sha256"],
            "verification_status": "pass",
            "financing_cost_component_identity_status": "pass",
            "accounting_invariants_status": "pass",
            "deterministic_replay_status": "pass",
            "memory_watchdog_status": "pass",
            "terminal_cumulative_modeled_financing_cost_bil": run[
                "terminal_cumulative_modeled_financing_cost_bil"
            ],
            "terminal_cumulative_tdc_change_bil": run[
                "terminal_cumulative_tdc_change_bil"
            ],
            "overall_new_issuance_wam_years": pre[
                "overall_new_issuance_wam_years"
            ],
            "economic_changed_paths": pre[
                "economic_changed_paths"
            ],
            "physical_changed_inputs": pre[
                "physical_changed_inputs"
            ],
            "fixed_input_comparison_status": "pass",
            "fixed_input_records_sha256": pre[
                "fixed_input_records_sha256"
            ],
            "curve_date_set_sha256": pre[
                "curve_date_set_sha256"
            ],
            "tenor_set_sha256": pre["tenor_set_sha256"],
        }
    receipt = {
        "schema_version": OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "campaign_eligible": True,
        "promotion_status": "eligible",
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract_sha,
        "campaign_root_identity": contract["campaign_root_identity"],
        "verification_order": list(OPEN04_CAMPAIGN_ROLES),
        "common_identity": common,
        "roles": roles,
        "pair_gates": pair_gates,
        "no_retuning_status": "pass",
    }
    return validate_open04_campaign_post_receipt_mapping(
        root,
        receipt,
        contract=contract,
        expected_contract_sha256=contract_sha,
    )


def verify_open04_campaign(
    campaign_root: str | Path,
    *,
    expected_contract_sha256: str,
    baseline_package: str | Path,
    attestation: str | Path,
    contract_relative_path: str = "open04_campaign_contract.json",
) -> dict[str, Any]:
    """Public campaign acceptance verifier (alias of the explicit post-run API)."""

    return verify_open04_campaign_post_run(
        campaign_root,
        expected_contract_sha256=expected_contract_sha256,
        baseline_package=baseline_package,
        attestation=attestation,
        contract_relative_path=contract_relative_path,
    )


def validate_open04_campaign_post_receipt_mapping(
    campaign_root: str | Path,
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Validate the exact thin-export receipt without trusting status omissions."""

    root = Path(campaign_root).expanduser().resolve()
    frozen = validate_open04_campaign_contract_mapping(
        root,
        contract,
        expected_contract_sha256=expected_contract_sha256,
    )
    result = dict(
        _require_mapping(receipt, label="OPEN-04 post-run receipt")
    )
    _require_exact_fields(
        result,
        set(OPEN04_POST_RECEIPT_KEYS),
        label="OPEN-04 post-run receipt",
    )
    if (
        result["schema_version"]
        != OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION
        or result["status"] != "pass"
        or result["campaign_eligible"] is not True
        or result["promotion_status"] != "eligible"
        or result["campaign_id"] != frozen["campaign_id"]
        or result["campaign_contract_sha256"]
        != expected_contract_sha256
        or result["campaign_root_identity"]
        != frozen["campaign_root_identity"]
        or result["verification_order"] != list(OPEN04_CAMPAIGN_ROLES)
        or result["no_retuning_status"] != "pass"
    ):
        raise Open04CampaignError(
            "OPEN-04 post-run receipt authority/status differs"
        )
    common = _require_mapping(
        result["common_identity"], label="post receipt common_identity"
    )
    _require_exact_fields(
        common,
        set(OPEN04_POST_COMMON_IDENTITY_KEYS),
        label="post receipt common_identity",
    )
    roles = _role_mapping(result["roles"], label="post receipt roles")
    frozen_common = _post_common_identity(frozen["common_identity"])
    if dict(common) != frozen_common:
        differing = {
            key
            for key in OPEN04_POST_COMMON_IDENTITY_KEYS
            if common[key] != frozen_common[key]
        }
        if differing != {"python_version"}:
            raise Open04CampaignError(
                "post receipt common identity differs from frozen contract"
            )
        runtime_python_version = _post_receipt_runtime_python_version(
            root,
            frozen=frozen,
            roles=roles,
        )
        if common["python_version"] != runtime_python_version:
            raise Open04CampaignError(
                "post receipt Python version differs from run manifests"
            )
    for role in OPEN04_CAMPAIGN_ROLES:
        item = _require_mapping(
            roles[role], label=f"post receipt roles.{role}"
        )
        _require_exact_fields(
            item,
            set(OPEN04_POST_ROLE_KEYS),
            label=f"post receipt roles.{role}",
        )
        declared = frozen["roles"][role]
        for field in (
            "scenario_id",
            "scenario_sha256",
            "scenario_source_sha256",
            "compiled_inputs_digest",
            "canonical_issuance_mix_sha256",
            "issuance_mix_file_sha256",
            "curve_sidecar_sha256",
            "signed_10y_shock_bp",
            "compiled_curve_delta_digest",
            "run_relative_path",
            "overall_new_issuance_wam_years",
            "economic_changed_paths",
            "physical_changed_inputs",
            "fixed_input_records_sha256",
            "curve_date_set_sha256",
            "tenor_set_sha256",
        ):
            if item[field] != declared[field]:
                raise Open04CampaignError(
                    f"post receipt {role}.{field} differs from contract"
                )
        if item["run_relative_path"] != _role_run_relative_path(
            item["run_relative_path"],
            label=f"post receipt {role}.run_relative_path",
        ):
            raise Open04CampaignError(
                f"post receipt {role} run path differs"
            )
        for field in (
            "scenario_sha256",
            "scenario_source_sha256",
            "run_scenario_copy_sha256",
            "compiled_inputs_digest",
            "canonical_issuance_mix_sha256",
            "issuance_mix_file_sha256",
            "run_manifest_sha256",
            "controller_completion_receipt_sha256",
            "terminal_summary_sha256",
            "source_output_manifest_sha256",
            "event_root_sha256",
            "fixed_input_records_sha256",
            "curve_date_set_sha256",
            "tenor_set_sha256",
        ):
            _sha256(item[field], label=f"post receipt {role}.{field}")
        if role != "baseline":
            _sha256(
                item["curve_sidecar_sha256"],
                label=f"post receipt {role}.curve_sidecar_sha256",
            )
            _sha256(
                item["compiled_curve_delta_digest"],
                label=f"post receipt {role}.compiled_curve_delta_digest",
            )
        for status_field in (
            "source_run_canonical_match_status",
            "verification_status",
            "financing_cost_component_identity_status",
            "accounting_invariants_status",
            "deterministic_replay_status",
            "memory_watchdog_status",
            "fixed_input_comparison_status",
        ):
            if item[status_field] != "pass":
                raise Open04CampaignError(
                    f"post receipt {role}.{status_field} did not pass"
                )
        if not str(item["host"]).strip():
            raise Open04CampaignError(
                f"post receipt {role}.host is blank"
            )
        if not str(item["run_id"]).strip():
            raise Open04CampaignError(
                f"post receipt {role}.run_id is blank"
            )
        if (
            item["event_schema_version"]
            != EVENT_SCHEMA_VERSION
        ):
            raise Open04CampaignError(
                f"post receipt {role}.event_schema_version differs"
            )
        _nonnegative_int(
            item["event_count"],
            label=f"post receipt {role}.event_count",
        )
        _finite_float(
            item["terminal_cumulative_modeled_financing_cost_bil"],
            label=f"post receipt {role} terminal financing cost",
        )
        _finite_float(
            item["terminal_cumulative_tdc_change_bil"],
            label=f"post receipt {role} terminal TDC",
        )
    _validate_pair_gate_mapping(result["pair_gates"], roles=roles)
    return result


def _post_receipt_runtime_python_version(
    campaign_root: Path,
    *,
    frozen: Mapping[str, Any],
    roles: Mapping[str, Any],
) -> str:
    """Resolve a contract/runtime Python divergence from hashed run manifests."""

    versions: set[str] = set()
    for role in OPEN04_CAMPAIGN_ROLES:
        declared = _require_mapping(
            frozen["roles"][role], label=f"campaign contract roles.{role}"
        )
        item = _require_mapping(
            roles[role], label=f"post receipt roles.{role}"
        )
        relative = _role_run_relative_path(
            declared["run_relative_path"],
            label=f"campaign contract {role}.run_relative_path",
        )
        if item.get("run_relative_path") != relative:
            raise Open04CampaignError(
                f"post receipt {role} run path differs"
            )
        manifest_path = campaign_root / relative / _RUN_MANIFEST_FILE
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path)
            != _sha256(
                item.get("run_manifest_sha256"),
                label=f"post receipt {role}.run_manifest_sha256",
            )
        ):
            raise Open04CampaignError(
                f"post receipt {role} run manifest identity differs"
            )
        manifest = _require_mapping(
            read_json(manifest_path), label=f"{role} run manifest"
        )
        environment = _require_mapping(
            manifest.get("code_environment"),
            label=f"{role} run code_environment",
        )
        version = str(environment.get("python_version") or "").strip()
        if not version:
            raise Open04CampaignError(
                f"{role} run Python version is blank"
            )
        versions.add(version)
    if len(versions) != 1:
        raise Open04CampaignError(
            "OPEN-04 runs do not share one Python runtime version"
        )
    return versions.pop()


def _run_role_evidence(
    role: str,
    campaign_root: Path,
    run_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    source_scenario: Mapping[str, Any],
    contract: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        manifest.get("schema_version")
        != "tdcsim_cbo_scenario_run_manifest_v2"
        or manifest.get("status") != "complete"
        or manifest.get("verification_grade") != "bounded_replay_v1"
        or manifest.get("evidence_profile")
        != "bounded_period_closure_v1"
    ):
        raise Open04CampaignError(
            f"{role} run is not a complete bounded-replay run"
        )
    expected_run_marker = dict(
        _require_mapping(
            source_scenario.get("open04_campaign"),
            label=f"{role} source campaign marker",
        )
    )
    if manifest.get("open04_campaign") != expected_run_marker:
        raise Open04CampaignError(
            f"{role} run manifest campaign marker differs"
        )
    run_id = str(manifest.get("run_id") or "")
    if not run_id:
        raise Open04CampaignError(f"{role} run_id is blank")
    baseline = _require_mapping(
        manifest.get("baseline"), label=f"{role} run baseline"
    )
    if dict(baseline) != dict(OPEN04_BASELINE_IDENTITY):
        raise Open04CampaignError(
            f"{role} run baseline identity differs"
        )
    simulation = _require_mapping(
        manifest.get("simulation"), label=f"{role} run simulation"
    )
    if dict(simulation) != {
        "start_date": OPEN04_START_DATE,
        "end_date": OPEN04_END_DATE,
        "frequency": OPEN04_FREQUENCY,
    }:
        raise Open04CampaignError(
            f"{role} run horizon/frequency differs"
        )
    aggregation = _require_mapping(
        manifest.get("aggregation_clock"),
        label=f"{role} run aggregation_clock",
    )
    if aggregation.get("clock_id") != OPEN04_AGGREGATION_CLOCK_ID:
        raise Open04CampaignError(
            f"{role} run aggregation clock differs"
        )
    output_manifest = _require_mapping(
        manifest.get("output_manifest"),
        label=f"{role} run output_manifest",
    )
    if (
        output_manifest.get("profile")
        != OPEN04_OUTPUT_CONTRACT["profile"]
        or output_manifest.get("compression")
        != OPEN04_OUTPUT_CONTRACT["compression"]
        or "catalog_sqlite" in output_manifest
    ):
        raise Open04CampaignError(
            f"{role} run output is not compact gzip/no-SQLite"
        )
    summary_record = _require_mapping(
        output_manifest.get("summary"),
        label=f"{role} run output_manifest.summary",
    )
    _require_exact_fields(
        summary_record,
        {"path", "sha256", "bytes"},
        label=f"{role} run output_manifest.summary",
    )
    if summary_record["path"] != "summary.json":
        raise Open04CampaignError(
            f"{role} run terminal summary path differs"
        )
    summary_path = run_root / "outputs" / "summary.json"
    if not summary_path.is_file():
        raise Open04CampaignError(
            f"{role} run terminal summary is missing"
        )
    terminal_summary_sha256 = _sha256(
        summary_record["sha256"],
        label=f"{role} run terminal summary SHA-256",
    )
    if (
        sha256_file(summary_path) != terminal_summary_sha256
        or summary_path.stat().st_size
        != _nonnegative_int(
            summary_record["bytes"],
            label=f"{role} run terminal summary bytes",
        )
    ):
        raise Open04CampaignError(
            f"{role} run terminal summary differs from its output manifest"
        )
    if manifest.get("coupling_decisions") != dict(OPEN04_FIXED_COUPLING):
        raise Open04CampaignError(f"{role} run coupling differs")

    scenario_block = _require_mapping(
        manifest.get("scenario"), label=f"{role} run scenario"
    )
    if (
        scenario_block.get("scenario_id")
        != OPEN04_ROLE_TO_SCENARIO_ID[role]
        or scenario_block.get("canonical_sha256")
        != canonical_json_sha256(source_scenario)
    ):
        raise Open04CampaignError(
            f"{role} run scenario identity differs from source"
        )
    scenario_relative = _safe_relative_file(
        scenario_block.get("relative_path"),
        label=f"{role} run scenario.relative_path",
    )
    run_scenario_path = run_root / scenario_relative
    if not run_scenario_path.is_file():
        raise Open04CampaignError(f"{role} run scenario copy is missing")
    run_scenario_sha = sha256_file(run_scenario_path)
    if run_scenario_sha != scenario_block.get("source_file_sha256"):
        raise Open04CampaignError(
            f"{role} run scenario-copy byte hash differs"
        )
    run_scenario = _require_mapping(
        read_json(run_scenario_path), label=f"{role} run scenario copy"
    )
    run_marker = validate_open04_scenario_contract(run_scenario)
    if (
        run_marker.role != role
        or canonical_json_sha256(run_scenario)
        != canonical_json_sha256(source_scenario)
    ):
        raise Open04CampaignError(
            f"{role} source/run canonical scenarios differ"
        )
    source_path = (
        campaign_root
        / contract["roles"][role]["scenario_source_relative_path"]
    )
    source_sha = sha256_file(source_path)
    if source_sha != contract["roles"][role]["scenario_source_sha256"]:
        raise Open04CampaignError(
            f"{role} source scenario bytes changed after freeze"
        )

    execution = _require_mapping(
        manifest.get("execution_contract"),
        label=f"{role} run execution_contract",
    )
    if (
        execution.get("schema_version")
        != "tdcsim_cbo_execution_contract_v1"
        or execution.get("single_writer_claim") is not True
        or execution.get("writer_claim_file_name")
        != ".tdcsim-cbo-bounded-writer.claim"
        or execution.get("writer_claim_scope")
        != "output_parent"
        or execution.get("writer_claim_scope_id")
        != f"{contract['contract_id']}.{role}"
        or execution.get("one_scenario_per_worker") is not True
        or execution.get("process_pool_enabled") is not False
        or execution.get("parent_watchdog_required") is not True
        or execution.get("scenario_process_mode")
        != "parent_watchdog_worker"
    ):
        raise Open04CampaignError(
            f"{role} run did not use its isolated bounded watchdog contract"
        )
    thread_environment = _require_mapping(
        execution.get("numerical_thread_environment"),
        label=f"{role} run numerical_thread_environment",
    )
    if not thread_environment or any(
        value != "1" for value in thread_environment.values()
    ):
        raise Open04CampaignError(
            f"{role} run numerical threads were not all pinned to one"
        )

    bounded = _require_mapping(
        manifest.get("bounded_evidence"),
        label=f"{role} run bounded_evidence",
    )
    thresholds = _require_mapping(
        bounded.get("memory_thresholds"),
        label=f"{role} run memory_thresholds",
    )
    expected_thresholds = {
        "minimum_available_bytes": _EXECUTION_CONTRACT[
            "minimum_available_memory_bytes"
        ],
        "acceptance_peak_rss_bytes": _EXECUTION_CONTRACT[
            "acceptance_peak_rss_bytes"
        ],
        "application_abort_rss_bytes": _EXECUTION_CONTRACT[
            "application_abort_rss_bytes"
        ],
        "parent_graceful_stop_rss_bytes": _EXECUTION_CONTRACT[
            "parent_graceful_stop_rss_bytes"
        ],
        "parent_kill_rss_bytes": _EXECUTION_CONTRACT[
            "parent_kill_rss_bytes"
        ],
    }
    for field, expected in expected_thresholds.items():
        if thresholds.get(field) != expected:
            raise Open04CampaignError(
                f"{role} run memory threshold differs: {field}"
            )
    peak = _nonnegative_int(
        bounded.get("peak_rss_bytes"),
        label=f"{role} run peak_rss_bytes",
    )
    if peak > _EXECUTION_CONTRACT["acceptance_peak_rss_bytes"]:
        raise Open04CampaignError(
            f"{role} run peak RSS exceeds the acceptance ceiling"
        )
    if bounded.get("invariant_status") != "pass":
        raise Open04CampaignError(
            f"{role} bounded invariant status did not pass"
        )
    event_schema = str(bounded.get("event_schema_version") or "")
    if event_schema != EVENT_SCHEMA_VERSION:
        raise Open04CampaignError(
            f"{role} run event schema differs"
        )
    event_count = _nonnegative_int(
        bounded.get("event_count"), label=f"{role} run event_count"
    )
    event_root = _sha256(
        bounded.get("event_root_sha256"),
        label=f"{role} run event_root_sha256",
    )
    recomputed = _require_mapping(
        verification.get("recomputed"),
        label=f"{role} replay recomputed evidence",
    )
    if (
        recomputed.get("bounded_event_count") != event_count
        or recomputed.get("bounded_peak_rss_bytes") != peak
    ):
        raise Open04CampaignError(
            f"{role} replay evidence differs from bounded manifest"
        )
    parent_watchdog = _require_mapping(
        manifest.get("parent_watchdog"),
        label=f"{role} run parent_watchdog",
    )
    if (
        parent_watchdog.get("status") != "accepted"
        or parent_watchdog.get("action") != "completed"
        or parent_watchdog.get("child_returncode") != 0
    ):
        raise Open04CampaignError(
            f"{role} parent watchdog did not accept a completed worker"
        )

    curve_runtime = manifest.get("evaluated_nominal_curve")
    if role == "baseline":
        if curve_runtime is not None:
            raise Open04CampaignError(
                "OPEN-04 baseline run must not carry curve-sidecar evidence"
            )
        max_curve_error = 0.0
        curve_delta_digest = OPEN04_NO_CURVE_DELTA_SENTINEL
    else:
        curve = _require_mapping(
            curve_runtime, label=f"{role} run evaluated_nominal_curve"
        )
        if (
            curve.get("sidecar_sha256")
            != contract["roles"][role]["curve_sidecar_sha256"]
            or curve.get("evaluated_delta_sha256")
            != contract["roles"][role]["compiled_curve_delta_digest"]
            or curve.get("short_end_bitwise_mismatch_count") != 0
        ):
            raise Open04CampaignError(
                f"{role} run curve receipt differs from pre-run evidence"
            )
        max_curve_error = _finite_float(
            curve.get("max_abs_analytic_delta_error_decimal"),
            label=f"{role} run maximum analytic curve error",
        )
        if max_curve_error > 1e-12:
            raise Open04CampaignError(
                f"{role} run analytic curve error exceeds tolerance"
            )
        curve_delta_digest = _sha256(
            curve.get("evaluated_delta_sha256"),
            label=f"{role} run evaluated delta digest",
        )

    annual = _annual_output_metrics(run_root, role=role)
    common_identity = {
        "package_id": baseline["package_id"],
        "baseline_package_sha256": baseline["package_sha256"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "release_attestation_sha256": baseline[
            "release_attestation_sha256"
        ],
        **_run_code_identity(manifest, role=role),
        "start_date": simulation["start_date"],
        "end_date": simulation["end_date"],
        "frequency": simulation["frequency"],
        "aggregation_clock_id": aggregation["clock_id"],
        "opening_identity_sha256": contract["common_identity"][
            "opening_identity_sha256"
        ],
        "fixed_input_records_sha256": contract["common_identity"][
            "fixed_input_records_sha256"
        ],
        "calibration_worksheet_sha256": contract["common_identity"][
            "calibration_worksheet_sha256"
        ],
    }
    _require_exact_fields(
        common_identity,
        set(OPEN04_POST_COMMON_IDENTITY_KEYS),
        label=f"{role} run common_identity",
    )
    return {
        "scenario_id": OPEN04_ROLE_TO_SCENARIO_ID[role],
        "scenario_sha256": canonical_json_sha256(source_scenario),
        "scenario_source_sha256": source_sha,
        "run_scenario_copy_sha256": run_scenario_sha,
        "run_id": run_id,
        "run_manifest_sha256": sha256_file(manifest_path),
        "terminal_summary_sha256": terminal_summary_sha256,
        "source_output_manifest_sha256": canonical_json_sha256(
            output_manifest
        ),
        "event_schema_version": event_schema,
        "event_count": event_count,
        "event_root_sha256": event_root,
        "terminal_cumulative_modeled_financing_cost_bil": annual[
            "terminal_financing_cost"
        ],
        "terminal_cumulative_tdc_change_bil": annual["terminal_tdc"],
        "full_fy_wam": annual["full_fy_wam"],
        "curve_delta_digest": curve_delta_digest,
        "max_curve_error": max_curve_error,
        "common_identity": common_identity,
    }


def _run_code_identity(
    manifest: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    env = _require_mapping(
        manifest.get("code_environment"),
        label=f"{role} run code_environment",
    )
    source_identity = _require_mapping(
        env.get("producer_source_identity"),
        label=f"{role} run producer_source_identity",
    )
    source_tree = _require_mapping(
        source_identity.get("source_tree"),
        label=f"{role} run producer source_tree",
    )
    locks = _validate_dependency_lock_records(
        source_tree.get("dependency_lock_files"), role=role
    )
    by_name = {item["relative_path"]: item for item in locks}
    if [item["relative_path"] for item in locks] != [
        "uv.lock",
        "requirements.lock.txt",
    ]:
        raise Open04CampaignError(
            f"{role} producer identity must bind requirements.lock.txt and uv.lock"
        )
    dependency_digest = canonical_json_sha256(locks)
    if (
        source_tree.get("dependency_lock_set_sha256")
        != dependency_digest
        or env.get("requirements_lock_sha256")
        != by_name["requirements.lock.txt"]["sha256"]
        or source_tree.get("release_commit_sha")
        != env.get("code_commit_sha")
        or source_identity.get("installed_archive_sha256")
        != env.get("wheel_sha256")
    ):
        raise Open04CampaignError(
            f"{role} producer dependency/source identity differs"
        )
    wheel_artifact = _require_mapping(
        env.get("wheel_artifact"),
        label=f"{role} run wheel_artifact",
    )
    if wheel_artifact.get("sha256") != env.get("wheel_sha256"):
        raise Open04CampaignError(
            f"{role} retained wheel artifact differs"
        )
    identity = {
        "code_commit_sha": env.get("code_commit_sha"),
        "dirty_state": env.get("dirty_state"),
        "requirements_lock_sha256": env.get(
            "requirements_lock_sha256"
        ),
        "uv_lock_sha256": by_name["uv.lock"]["sha256"],
        "dependency_lock_set_sha256": dependency_digest,
        "wheel_sha256": env.get("wheel_sha256"),
        "wheel_artifact_sha256": wheel_artifact.get("sha256"),
        "runner_source_sha256": env.get("runner_source_sha256"),
        "sim_engine_source_sha256": env.get(
            "sim_engine_source_sha256"
        ),
        "bounded_output_source_sha256": env.get(
            "bounded_output_source_sha256"
        ),
        "output_source_sha256": env.get("output_source_sha256"),
        "verifier_source_sha256": env.get("verifier_source_sha256"),
        "compiler_source_sha256": env.get("compiler_source_sha256"),
        "contract_source_sha256": env.get("contract_source_sha256"),
        "manifest_source_sha256": env.get("manifest_source_sha256"),
        "run_manifest_schema_sha256": env.get(
            "run_manifest_schema_sha256"
        ),
        "scenario_schema_sha256": env.get(
            "scenario_schema_sha256"
        ),
        "scenario_writer_source_sha256": env.get(
            "scenario_writer_source_sha256"
        ),
        "open04_exporter_source_sha256": env.get(
            "open04_exporter_source_sha256"
        ),
        "python_version": env.get("python_version"),
        "package_name": env.get("package_name"),
        "package_version": env.get("package_version"),
        "distribution_file_digest": env.get(
            "distribution_file_digest"
        ),
        "runtime_identity_source": env.get(
            "runtime_identity_source"
        ),
    }
    return _validate_code_identity(identity)


def _validate_dependency_lock_records(
    value: Any,
    *,
    role: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise Open04CampaignError(
            f"{role} producer dependency_lock_files is empty"
        )
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _require_mapping(
            raw, label=f"{role} dependency_lock_files[{index}]"
        )
        _require_exact_fields(
            item,
            {"relative_path", "sha256", "bytes"},
            label=f"{role} dependency_lock_files[{index}]",
        )
        relative = _direct_child_name(
            item["relative_path"],
            label=f"{role} dependency_lock_files[{index}].relative_path",
        )
        records.append(
            {
                "relative_path": relative,
                "sha256": _sha256(
                    item["sha256"],
                    label=f"{role} dependency lock {relative}",
                ),
                "bytes": _nonnegative_int(
                    item["bytes"],
                    label=f"{role} dependency lock {relative} bytes",
                ),
            }
        )
    if len({item["relative_path"] for item in records}) != len(records):
        raise Open04CampaignError(
            f"{role} producer dependency locks contain duplicates"
        )
    return records


def _controller_completion_receipt(
    path: Path,
    *,
    role: str,
    campaign_id: str,
    campaign_contract_sha256: str,
    run_id: str,
    run_manifest_sha256: str,
    expected_terminal_summary_sha256: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise Open04CampaignError(
            f"{role} controller completion receipt is missing"
        )
    receipt = dict(
        _require_mapping(
            read_json(path), label=f"{role} controller completion receipt"
        )
    )
    _require_exact_fields(
        receipt,
        set(_CONTROLLER_RECEIPT_KEYS),
        label=f"{role} controller completion receipt",
    )
    controller_run_id = str(receipt["controller_run_id"]).strip()
    if (
        receipt["campaign_id"] != campaign_id
        or receipt["campaign_contract_sha256"]
        != campaign_contract_sha256
        or receipt["role"] != role
        or receipt["run_id"] != run_id
        or receipt["terminal_status"] != "completed"
        or receipt["run_manifest_sha256"] != run_manifest_sha256
        or receipt["terminal_summary_sha256"]
        != expected_terminal_summary_sha256
        or not str(receipt["host"]).strip()
        or not controller_run_id
        or receipt["controller_command_exit_code"] != 0
        or receipt["controller_exit_code"] != 0
        or receipt["preflight_conflicts"] != 0
        or receipt["postrun_conflicts"] != 0
        or not str(receipt["controller_memory_guard_status"]).strip()
        or receipt["controller_process_tree_drained"] is not True
        or receipt["controller_peak_rss_mb"] is None
    ):
        raise Open04CampaignError(
            f"{role} controller completion receipt identity/status differs"
        )
    if receipt["schema_version"] == OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION:
        if (
            receipt["controller"] != "remote_controller"
            or receipt["placement"] != "auto"
            or receipt["controller_telemetry_status"] != "reported"
            or receipt["controller_avg_cpu_pct"] is None
            or receipt["controller_telemetry_locator"]
            != f"controller_run_id:{controller_run_id}"
        ):
            raise Open04CampaignError(
                f"{role} controller completion receipt identity/status differs"
            )
    elif (
        receipt["schema_version"]
        == OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION
    ):
        if (
            receipt["controller"] != "host_owned_task"
            or receipt["placement"] != "auto_selected_host"
            or receipt["controller_telemetry_status"]
            != "bounded_role_watchdogs"
            or receipt["controller_avg_cpu_pct"] is not None
            or receipt["controller_telemetry_locator"]
            != "controller_summary_sha256:"
            f"{receipt['controller_summary_sha256']}"
        ):
            raise Open04CampaignError(
                f"{role} host-task completion receipt identity/status differs"
            )
    else:
        raise Open04CampaignError(
            f"{role} controller completion receipt schema differs"
        )
    for field in (
        "controller_command_exit_code",
        "controller_exit_code",
        "preflight_conflicts",
        "postrun_conflicts",
    ):
        _nonnegative_int(
            receipt[field],
            label=f"{role} controller {field}",
        )
    for field in ("controller_peak_rss_mb", "controller_avg_cpu_pct"):
        if receipt[field] is not None and _finite_float(
            receipt[field],
            label=f"{role} {field}",
        ) < 0.0:
            raise Open04CampaignError(
                f"{role} {field} must be nonnegative when reported"
            )
    _sha256(
        receipt["terminal_summary_sha256"],
        label=f"{role} terminal_summary_sha256",
    )
    _sha256(
        receipt["run_manifest_sha256"],
        label=f"{role} controller run_manifest_sha256",
    )
    _sha256(
        receipt["controller_summary_sha256"],
        label=f"{role} controller_summary_sha256",
    )
    _sha256(
        receipt["controller_command_sha256"],
        label=f"{role} controller_command_sha256",
    )
    started = _utc_datetime(
        receipt["started_at_utc"], label=f"{role} started_at_utc"
    )
    completed = _utc_datetime(
        receipt["completed_at_utc"], label=f"{role} completed_at_utc"
    )
    exited = _utc_datetime(
        receipt["worker_exit_confirmed_at_utc"],
        label=f"{role} worker_exit_confirmed_at_utc",
    )
    if not started < completed <= exited:
        raise Open04CampaignError(
            f"{role} controller timestamps are not ordered"
        )
    receipt["_receipt_sha256"] = sha256_file(path)
    receipt["_started"] = started
    receipt["_completed"] = completed
    receipt["_exited"] = exited
    return receipt


def _verify_controller_resource_contract(
    receipts: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    execution = _require_mapping(
        contract.get("execution_contract"),
        label="campaign execution_contract",
    )
    budget_bytes = _nonnegative_int(
        execution.get("aggregate_acceptance_peak_rss_bytes"),
        label="aggregate_acceptance_peak_rss_bytes",
    )
    if budget_bytes == 0:
        raise Open04CampaignError(
            "aggregate_acceptance_peak_rss_bytes must be positive"
        )
    controller_ids = {
        str(receipt["controller_run_id"]) for receipt in receipts.values()
    }
    controller_modes = {
        (
            str(receipt["schema_version"]),
            str(receipt["controller"]),
            str(receipt["placement"]),
        )
        for receipt in receipts.values()
    }
    if len(controller_ids) != 1 or len(controller_modes) != 1:
        raise Open04CampaignError(
            "OPEN-04 parallel roles must share one batch controller"
        )
    peak_values = {
        float(receipt["controller_peak_rss_mb"]) for receipt in receipts.values()
    }
    if (
        len(peak_values) != 1
        or next(iter(peak_values)) * 1024**2 > budget_bytes
    ):
        raise Open04CampaignError(
            "OPEN-04 batch exceeded the aggregate memory acceptance budget"
        )


def _utc_datetime(value: Any, *, label: str) -> datetime:
    text = str(value)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise Open04CampaignError(
            f"{label} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise Open04CampaignError(f"{label} must be explicitly UTC")
    return parsed


def _annual_output_metrics(
    run_root: Path,
    *,
    role: str,
) -> dict[str, Any]:
    from .bounded_output import ANNUAL_COLUMNS

    path = run_root / "outputs" / _ANNUAL_OUTPUT_FILE
    if not path.is_file():
        raise Open04CampaignError(
            f"{role} annual economic summary is missing"
        )
    rows: list[dict[str, str]] = []
    with gzip.open(
        path, "rt", encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ANNUAL_COLUMNS:
            raise Open04CampaignError(
                f"{role} annual economic summary columns differ"
            )
        for row in reader:
            if None in row:
                raise Open04CampaignError(
                    f"{role} annual economic summary has an over-wide row"
                )
            rows.append(dict(row))
    if len(rows) != 11:
        raise Open04CampaignError(
            f"{role} annual economic summary must have 11 FY buckets"
        )

    running = {
        "tdc_change_bil": 0.0,
        "overlap_cashflow_bil": 0.0,
        "tdc_change_ex_overlap_bil": 0.0,
        "interest_outlay_bil": 0.0,
        "issue_discount_cost_bil": 0.0,
        "nonmarketable_interest_capitalized_bil": 0.0,
        "tips_inflation_accretion_bil": 0.0,
        "modeled_financing_cost_bil": 0.0,
    }
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
    full_fy_wam: dict[int, float] = {}
    previous_end: str | None = None
    for index, row in enumerate(rows):
        start = _iso_date(
            row["period_start"], label=f"{role} annual period_start"
        )
        end = _iso_date(
            row["period_end"], label=f"{role} annual period_end"
        )
        if previous_end is not None and start != previous_end:
            raise Open04CampaignError(
                f"{role} annual periods are not continuous"
            )
        previous_end = end
        is_partial = _strict_bool(
            row["is_partial_period"],
            label=f"{role} annual is_partial_period",
        )
        coverage = _nonnegative_int_text(
            row["coverage_days"],
            label=f"{role} annual coverage_days",
        )
        expected_coverage = _nonnegative_int_text(
            row["expected_coverage_days"],
            label=f"{role} annual expected_coverage_days",
        )
        actual_coverage = (
            date.fromisoformat(end) - date.fromisoformat(start)
        ).days
        if coverage != actual_coverage or expected_coverage < coverage:
            raise Open04CampaignError(
                f"{role} annual coverage contract differs"
            )
        if row["aggregation_clock_id"] != OPEN04_AGGREGATION_CLOCK_ID:
            raise Open04CampaignError(
                f"{role} annual aggregation clock differs"
            )
        if index == 0:
            if (
                start != OPEN04_START_DATE
                or end != "2026-09-30"
                or row["period_label"] != "FY2026_PARTIAL_OPENING"
                or is_partial is not True
                or coverage != 101
            ):
                raise Open04CampaignError(
                    f"{role} opening partial FY bucket differs"
                )
        else:
            fiscal_year = 2026 + index
            if (
                start != f"{fiscal_year - 1}-09-30"
                or end != f"{fiscal_year}-09-30"
                or row["period_label"] != f"FY{fiscal_year}"
                or is_partial is not False
                or coverage != expected_coverage
            ):
                raise Open04CampaignError(
                    f"{role} full FY{fiscal_year} bucket differs"
                )

        values = {
            name: _finite_float(
                row[name], label=f"{role} annual {name}"
            )
            for name in running
        }
        if not _close_enough(
            values["tdc_change_bil"],
            values["overlap_cashflow_bil"]
            + values["tdc_change_ex_overlap_bil"],
        ):
            raise Open04CampaignError(
                f"{role} annual TDC overlap identity failed"
            )
        if not _close_enough(
            values["modeled_financing_cost_bil"],
            values["interest_outlay_bil"]
            + values["issue_discount_cost_bil"]
            + values["nonmarketable_interest_capitalized_bil"]
            + values["tips_inflation_accretion_bil"],
        ):
            raise Open04CampaignError(
                f"{role} annual financing-cost component identity failed"
            )
        for name, value in values.items():
            running[name] += value
            observed = _finite_float(
                row[cumulative_columns[name]],
                label=(
                    f"{role} annual {cumulative_columns[name]}"
                ),
            )
            if not _close_enough(running[name], observed):
                raise Open04CampaignError(
                    f"{role} annual cumulative {name} identity failed"
                )
        if not _close_enough(
            running["tdc_change_bil"],
            running["overlap_cashflow_bil"]
            + running["tdc_change_ex_overlap_bil"],
        ):
            raise Open04CampaignError(
                f"{role} cumulative TDC overlap identity failed"
            )
        if not _close_enough(
            running["modeled_financing_cost_bil"],
            running["interest_outlay_bil"]
            + running["issue_discount_cost_bil"]
            + running["nonmarketable_interest_capitalized_bil"]
            + running["tips_inflation_accretion_bil"],
        ):
            raise Open04CampaignError(
                f"{role} cumulative financing-cost identity failed"
            )

        face = _finite_float(
            row["new_issuance_face_bil"],
            label=f"{role} annual new_issuance_face_bil",
        )
        term_face = _finite_float(
            row["new_issuance_original_term_face_years_bil"],
            label=(
                f"{role} annual "
                "new_issuance_original_term_face_years_bil"
            ),
        )
        if face < -1e-12:
            raise Open04CampaignError(
                f"{role} annual issuance face is negative"
            )
        if not is_partial and face > 1e-12:
            fiscal_year = int(end[:4])
            wam = term_face / face
            declared_wam = _finite_float(
                row["new_issuance_wam_years"],
                label=f"{role} annual new_issuance_wam_years",
            )
            if not math.isclose(
                wam, declared_wam, rel_tol=0.0, abs_tol=1e-12
            ):
                raise Open04CampaignError(
                    f"{role} annual issuance WAM is stale"
                )
            full_fy_wam[fiscal_year] = wam
        elif not is_partial and abs(face) <= 1e-12:
            full_fy_wam[int(end[:4])] = math.nan

    if rows[-1]["period_end"] != OPEN04_END_DATE:
        raise Open04CampaignError(
            f"{role} annual terminal date differs"
        )
    return {
        "terminal_financing_cost": _finite_float(
            rows[-1]["cumulative_modeled_financing_cost_bil"],
            label=f"{role} terminal cumulative financing cost",
        ),
        "terminal_tdc": _finite_float(
            rows[-1]["cumulative_tdc_change_bil"],
            label=f"{role} terminal cumulative TDC",
        ),
        "full_fy_wam": full_fy_wam,
    }


def _pair_gates(
    run_evidence: Mapping[str, Mapping[str, Any]],
    compiled_evidence: Mapping[str, Mapping[str, Any]],
    controller_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    del controller_receipts  # non-overlap is verified independently above.
    baseline = run_evidence["baseline"]
    candidate_a = run_evidence["candidate_a"]
    candidate_b = run_evidence["candidate_b"]
    a_cost = (
        candidate_a["terminal_cumulative_modeled_financing_cost_bil"]
        - baseline["terminal_cumulative_modeled_financing_cost_bil"]
    )
    a_tdc = (
        candidate_a["terminal_cumulative_tdc_change_bil"]
        - baseline["terminal_cumulative_tdc_change_bil"]
    )
    b_cost = (
        candidate_b["terminal_cumulative_modeled_financing_cost_bil"]
        - baseline["terminal_cumulative_modeled_financing_cost_bil"]
    )
    b_tdc = (
        candidate_b["terminal_cumulative_tdc_change_bil"]
        - baseline["terminal_cumulative_tdc_change_bil"]
    )
    threshold = OPEN04_SIGN_GATE_THRESHOLD_BIL
    if not (
        a_cost < -threshold
        and a_tdc > threshold
        and b_cost > threshold
        and b_tdc < -threshold
    ):
        raise Open04CampaignError(
            "failed_predeclared_paired_sign_gate"
        )

    overall = {
        role: _finite_float(
            compiled_evidence[role][
                "overall_new_issuance_wam_years"
            ],
            label=f"{role} compiled-input WAM",
        )
        for role in OPEN04_CAMPAIGN_ROLES
    }
    if not (
        overall["candidate_a"]
        < overall["baseline"]
        < overall["candidate_b"]
    ):
        raise Open04CampaignError(
            "OPEN-04 compiled-input WAM ordering failed"
        )

    fy_sets = {
        role: set(run_evidence[role]["full_fy_wam"])
        for role in OPEN04_CAMPAIGN_ROLES
    }
    expected_years = set(range(2027, 2037))
    if any(values != expected_years for values in fy_sets.values()):
        raise Open04CampaignError(
            "OPEN-04 full-FY WAM coverage differs across roles"
        )
    full_fy: list[dict[str, Any]] = []
    for fiscal_year in sorted(expected_years):
        values = {
            role: _finite_float(
                run_evidence[role]["full_fy_wam"][fiscal_year],
                label=f"{role} FY{fiscal_year} realized WAM",
            )
            for role in OPEN04_CAMPAIGN_ROLES
        }
        if not (
            values["candidate_a"]
            < values["baseline"]
            < values["candidate_b"]
        ):
            raise Open04CampaignError(
                f"OPEN-04 FY{fiscal_year} realized WAM ordering failed"
            )
        full_fy.append(
            {
                "fiscal_year": fiscal_year,
                "candidate_a": values["candidate_a"],
                "baseline": values["baseline"],
                "candidate_b": values["candidate_b"],
                "status": "pass",
            }
        )

    date_sets = {
        compiled_evidence[role]["curve_date_set_sha256"]
        for role in OPEN04_CAMPAIGN_ROLES
    }
    tenor_sets = {
        compiled_evidence[role]["tenor_set_sha256"]
        for role in OPEN04_CAMPAIGN_ROLES
    }
    if len(date_sets) != 1 or len(tenor_sets) != 1:
        raise Open04CampaignError(
            "OPEN-04 curve date/tenor sets differ across roles"
        )
    max_antisymmetry_error = (
        _finite_float(
            candidate_a["max_curve_error"],
            label="candidate_a maximum curve error",
        )
        + _finite_float(
            candidate_b["max_curve_error"],
            label="candidate_b maximum curve error",
        )
    )
    if max_antisymmetry_error > OPEN04_ANTISYMMETRY_TOLERANCE_DECIMAL:
        raise Open04CampaignError(
            "OPEN-04 evaluated-delta antisymmetry tolerance failed"
        )
    antisymmetry_proof = {
        "schema_version": (
            "tdcsim_open04_evaluated_delta_antisymmetry_proof_v1"
        ),
        "candidate_a_evaluated_delta_sha256": candidate_a[
            "curve_delta_digest"
        ],
        "candidate_b_evaluated_delta_sha256": candidate_b[
            "curve_delta_digest"
        ],
        "candidate_a_max_abs_analytic_delta_error_decimal": candidate_a[
            "max_curve_error"
        ],
        "candidate_b_max_abs_analytic_delta_error_decimal": candidate_b[
            "max_curve_error"
        ],
        "triangle_bound_max_abs_antisymmetry_error_decimal": (
            max_antisymmetry_error
        ),
        "analytic_shocks_bp": {
            "candidate_a": -25.0,
            "candidate_b": 25.0,
        },
        "curve_date_set_sha256": next(iter(date_sets)),
        "tenor_set_sha256": next(iter(tenor_sets)),
    }
    gates = {
        "threshold_bil": threshold,
        "candidate_a_financing_cost_delta_bil": a_cost,
        "candidate_a_tdc_delta_bil": a_tdc,
        "candidate_b_financing_cost_delta_bil": b_cost,
        "candidate_b_tdc_delta_bil": b_tdc,
        "terminal_sign_gate_status": "pass",
        "overall_wam_ordering": {
            "candidate_a": overall["candidate_a"],
            "baseline": overall["baseline"],
            "candidate_b": overall["candidate_b"],
            "status": "pass",
        },
        "full_fy_wam_ordering": full_fy,
        "full_fy_wam_gate_status": "pass",
        "controller_output_isolation_status": "pass",
        "aggregate_memory_budget_status": "pass",
        "evaluated_delta_antisymmetry_sha256": canonical_json_sha256(
            antisymmetry_proof
        ),
        "maximum_abs_evaluated_delta_antisymmetry_error": (
            max_antisymmetry_error
        ),
        "evaluated_delta_antisymmetry_status": "pass",
        "curve_date_set_equality_status": "pass",
        "tenor_set_equality_status": "pass",
        "change_perimeter_status": "pass",
        "fixed_input_equality_status": "pass",
    }
    _validate_pair_gate_mapping(gates)
    return gates


def _validate_pair_gate_mapping(
    value: Any,
    *,
    roles: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    gates = _require_mapping(value, label="post receipt pair_gates")
    _require_exact_fields(
        gates,
        set(OPEN04_PAIR_GATE_KEYS),
        label="post receipt pair_gates",
    )
    threshold = _finite_float(
        gates["threshold_bil"], label="pair gate threshold_bil"
    )
    if threshold != OPEN04_SIGN_GATE_THRESHOLD_BIL:
        raise Open04CampaignError("pair gate threshold differs")
    a_cost = _finite_float(
        gates["candidate_a_financing_cost_delta_bil"],
        label="candidate_a financing-cost delta",
    )
    a_tdc = _finite_float(
        gates["candidate_a_tdc_delta_bil"],
        label="candidate_a TDC delta",
    )
    b_cost = _finite_float(
        gates["candidate_b_financing_cost_delta_bil"],
        label="candidate_b financing-cost delta",
    )
    b_tdc = _finite_float(
        gates["candidate_b_tdc_delta_bil"],
        label="candidate_b TDC delta",
    )
    if not (
        a_cost < -threshold
        and a_tdc > threshold
        and b_cost > threshold
        and b_tdc < -threshold
    ):
        raise Open04CampaignError(
            "post receipt predeclared sign gates do not pass"
        )
    if roles is not None:
        baseline = roles["baseline"]
        expected = {
            "candidate_a_financing_cost_delta_bil": (
                roles["candidate_a"][
                    "terminal_cumulative_modeled_financing_cost_bil"
                ]
                - baseline[
                    "terminal_cumulative_modeled_financing_cost_bil"
                ]
            ),
            "candidate_a_tdc_delta_bil": (
                roles["candidate_a"][
                    "terminal_cumulative_tdc_change_bil"
                ]
                - baseline["terminal_cumulative_tdc_change_bil"]
            ),
            "candidate_b_financing_cost_delta_bil": (
                roles["candidate_b"][
                    "terminal_cumulative_modeled_financing_cost_bil"
                ]
                - baseline[
                    "terminal_cumulative_modeled_financing_cost_bil"
                ]
            ),
            "candidate_b_tdc_delta_bil": (
                roles["candidate_b"][
                    "terminal_cumulative_tdc_change_bil"
                ]
                - baseline["terminal_cumulative_tdc_change_bil"]
            ),
        }
        for field, expected_value in expected.items():
            if gates[field] != expected_value:
                raise Open04CampaignError(
                    f"post receipt {field} is stale"
                )
    for status in (
        "terminal_sign_gate_status",
        "full_fy_wam_gate_status",
        "controller_output_isolation_status",
        "aggregate_memory_budget_status",
        "evaluated_delta_antisymmetry_status",
        "curve_date_set_equality_status",
        "tenor_set_equality_status",
        "change_perimeter_status",
        "fixed_input_equality_status",
    ):
        if gates[status] != "pass":
            raise Open04CampaignError(
                f"post receipt pair gate did not pass: {status}"
            )
    overall = _require_mapping(
        gates["overall_wam_ordering"],
        label="pair gate overall_wam_ordering",
    )
    _require_exact_fields(
        overall,
        {"candidate_a", "baseline", "candidate_b", "status"},
        label="pair gate overall_wam_ordering",
    )
    overall_values = {
        role: _finite_float(
            overall[role], label=f"overall WAM {role}"
        )
        for role in OPEN04_CAMPAIGN_ROLES
    }
    if (
        overall["status"] != "pass"
        or not (
            overall_values["candidate_a"]
            < overall_values["baseline"]
            < overall_values["candidate_b"]
        )
    ):
        raise Open04CampaignError(
            "post receipt overall WAM ordering did not pass"
        )
    if roles is not None:
        for role in OPEN04_CAMPAIGN_ROLES:
            if (
                overall_values[role]
                != roles[role]["overall_new_issuance_wam_years"]
            ):
                raise Open04CampaignError(
                    f"post receipt overall WAM is stale for {role}"
                )
    full_fy = gates["full_fy_wam_ordering"]
    if not isinstance(full_fy, list) or len(full_fy) != 10:
        raise Open04CampaignError(
            "post receipt full-FY WAM ordering must cover FY2027-FY2036"
        )
    years: list[int] = []
    for index, raw in enumerate(full_fy):
        item = _require_mapping(
            raw, label=f"full_fy_wam_ordering[{index}]"
        )
        _require_exact_fields(
            item,
            {"fiscal_year", "candidate_a", "baseline", "candidate_b", "status"},
            label=f"full_fy_wam_ordering[{index}]",
        )
        fiscal_year = item["fiscal_year"]
        if isinstance(fiscal_year, bool) or not isinstance(
            fiscal_year, int
        ):
            raise Open04CampaignError(
                "full-FY WAM fiscal_year must be an integer"
            )
        years.append(fiscal_year)
        values = {
            role: _finite_float(
                item[role], label=f"FY{fiscal_year} WAM {role}"
            )
            for role in OPEN04_CAMPAIGN_ROLES
        }
        if (
            item["status"] != "pass"
            or not (
                values["candidate_a"]
                < values["baseline"]
                < values["candidate_b"]
            )
        ):
            raise Open04CampaignError(
                f"post receipt FY{fiscal_year} WAM ordering failed"
            )
    if years != list(range(2027, 2037)):
        raise Open04CampaignError(
            "post receipt full-FY WAM years differ"
        )
    _sha256(
        gates["evaluated_delta_antisymmetry_sha256"],
        label="evaluated_delta_antisymmetry_sha256",
    )
    max_error = _finite_float(
        gates["maximum_abs_evaluated_delta_antisymmetry_error"],
        label="maximum evaluated-delta antisymmetry error",
    )
    if (
        max_error < 0.0
        or max_error > OPEN04_ANTISYMMETRY_TOLERANCE_DECIMAL
    ):
        raise Open04CampaignError(
            "post receipt evaluated-delta antisymmetry error differs"
        )


def _strict_bool(value: Any, *, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise Open04CampaignError(f"{label} must be true or false")


def _nonnegative_int_text(value: Any, *, label: str) -> int:
    text = str(value)
    if not text.isdigit():
        raise Open04CampaignError(
            f"{label} must be a nonnegative integer"
        )
    return int(text)


def _close_enough(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)


def _deterministic_json_file_sha256(value: Any) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evaluated_nominal_override(shock_bp: float) -> dict[str, Any]:
    return normalize_open04_override(
        {
            "mode": "evaluated_additive_key_rate_bp",
            "application": "post_baseline_evaluation",
            "interpolation": "log_tenor_linear",
            "lower_endpoint": "zero_at_or_below_first_key",
            "upper_endpoint": "flat_at_or_above_last_key",
            "time_profile": "constant_across_curve_dates",
            "compounding": "none",
            "shocks": [
                {"tenor_years": 2.0, "shock_bp": 0.0},
                {"tenor_years": 10.0, "shock_bp": float(shock_bp)},
            ],
        }
    )


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Open04CampaignError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise Open04CampaignError(f"{label} field names must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise Open04CampaignError(
            f"{label} fields differ: missing={missing}, extra={extra}"
        )


def _role_mapping(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    mapping = _require_mapping(value, label=label)
    _require_exact_fields(mapping, set(OPEN04_CAMPAIGN_ROLES), label=label)
    return {role: mapping[role] for role in OPEN04_CAMPAIGN_ROLES}


def _campaign_id(value: Any) -> str:
    identifier = str(value)
    if (
        len(identifier) < 3
        or len(identifier) > 160
        or not identifier[0].isalnum()
        or any(
            character
            not in (
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            )
            for character in identifier
        )
    ):
        raise Open04CampaignError("OPEN-04 campaign_id is invalid")
    return identifier


def _direct_child_name(value: Any, *, label: str) -> str:
    text = str(value)
    path = Path(text)
    if (
        not text
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != text
        or text in {".", ".."}
    ):
        raise Open04CampaignError(
            f"{label} must be a direct-child basename"
        )
    return text


def _role_run_relative_path(value: Any, *, label: str) -> str:
    text = _safe_relative_file(value, label=label)
    parts = text.split("/")
    if len(parts) != 2:
        raise Open04CampaignError(
            f"{label} must identify one isolated role directory and run directory"
        )
    return text


def _validate_code_identity(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _require_mapping(value, label="code_identity")
    _require_exact_fields(
        identity, set(_CODE_IDENTITY_KEYS), label="code_identity"
    )
    result = {key: identity[key] for key in _CODE_IDENTITY_KEYS}
    _commit_sha(result["code_commit_sha"], label="code_commit_sha")
    if result["dirty_state"] is not False:
        raise Open04CampaignError("code_identity.dirty_state must be false")
    digest_fields = {
        key
        for key in _CODE_IDENTITY_KEYS
        if key.endswith("_sha256") or key == "distribution_file_digest"
    }
    for key in digest_fields:
        _sha256(result[key], label=f"code_identity.{key}")
        if result[key] == "0" * 64:
            raise Open04CampaignError(
                f"code_identity.{key} must be release-bound"
            )
    if result["wheel_artifact_sha256"] != result["wheel_sha256"]:
        raise Open04CampaignError(
            "code_identity wheel artifact does not match wheel_sha256"
        )
    if result["runtime_identity_source"] != "installed_distribution_files":
        raise Open04CampaignError(
            "OPEN-04 code identity must come from installed distribution files"
        )
    for key in ("python_version", "package_name", "package_version"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise Open04CampaignError(f"code_identity.{key} is blank")
    version_parts = str(result["python_version"]).split(".")
    if len(version_parts) < 3 or not all(
        part.isdigit() for part in version_parts[:3]
    ):
        raise Open04CampaignError(
            "code_identity.python_version must be numeric X.Y.Z"
        )
    return result


def _compiled_campaign_evidence(
    compiled_dirs: Mapping[str, Path],
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    expected_mix_hashes: Mapping[str, str],
) -> dict[str, Any]:
    roles = _role_mapping(compiled_dirs, label="compiled_dirs")
    scenario_map = _role_mapping(scenarios, label="scenarios")
    mix_hashes = _role_mapping(
        expected_mix_hashes, label="expected_mix_hashes"
    )
    evidence: dict[str, dict[str, Any]] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        evidence[role] = _compiled_role_evidence(
            role,
            Path(roles[role]),
            scenario_map[role],
            expected_mix_sha256=_sha256(
                mix_hashes[role],
                label=f"{role}.expected_mix_sha256",
            ),
        )

    reference_fixed = evidence["baseline"]["fixed_input_records"]
    reference_opening = evidence["baseline"]["opening_input_records"]
    reference_baseline = evidence["baseline"]["baseline_identity"]
    reference_date_set = evidence["baseline"]["curve_date_set_sha256"]
    reference_tenor_set = evidence["baseline"]["tenor_set_sha256"]
    for role in OPEN04_CAMPAIGN_ROLES:
        current = evidence[role]
        if current["baseline_identity"] != reference_baseline:
            raise Open04CampaignError(
                f"{role} compiled baseline identity differs"
            )
        if current["fixed_input_records"] != reference_fixed:
            raise Open04CampaignError(
                f"{role} compiled fixed inputs differ from baseline"
            )
        if current["opening_input_records"] != reference_opening:
            raise Open04CampaignError(
                f"{role} compiled opening inputs differ from baseline"
            )
        if current["curve_date_set_sha256"] != reference_date_set:
            raise Open04CampaignError(
                f"{role} nominal curve date set differs from baseline"
            )
        if current["tenor_set_sha256"] != reference_tenor_set:
            raise Open04CampaignError(
                f"{role} nominal curve tenor set differs from baseline"
            )

    fixed_digest = canonical_json_sha256(reference_fixed)
    opening_digest = canonical_json_sha256(reference_opening)
    if any(
        item["fixed_input_records_sha256"] != fixed_digest
        for item in evidence.values()
    ):
        raise Open04CampaignError(
            "compiled role fixed-input digests do not match common records"
        )
    return {
        "roles": evidence,
        "fixed_input_records": reference_fixed,
        "fixed_input_records_sha256": fixed_digest,
        "opening_input_records": reference_opening,
        "opening_identity_sha256": opening_digest,
    }


def _compiled_role_evidence(
    role: str,
    compiled_dir: Path,
    scenario: Mapping[str, Any],
    *,
    expected_mix_sha256: str,
) -> dict[str, Any]:
    # Local imports avoid a module cycle: compiler and verifier import the
    # stable marker helpers from this module.
    from .curve_runtime import build_evaluated_nominal_runtime_binding
    from .verifier import verify_compiled_scenario

    compiled = compiled_dir.expanduser().resolve()
    manifest_path = compiled / _COMPILED_MANIFEST_FILE
    inputs = compiled / "forecast_inputs"
    if not manifest_path.is_file() or not inputs.is_dir():
        raise Open04CampaignError(
            f"{role} compiled directory is incomplete: {compiled}"
        )
    verification = verify_compiled_scenario(compiled)
    if (
        not isinstance(verification, Mapping)
        or verification.get("status") != "pass"
    ):
        raise Open04CampaignError(
            f"{role} compiled verifier did not return pass"
        )
    manifest = _require_mapping(
        read_json(manifest_path), label=f"{role} compiled manifest"
    )
    marker = validate_open04_scenario_contract(scenario)
    if marker.role != role:
        raise Open04CampaignError(f"{role} scenario marker mismatch")
    if manifest.get("scenario_contract") != scenario:
        raise Open04CampaignError(
            f"{role} compiled scenario contract differs from frozen source"
        )
    scenario_sha = canonical_json_sha256(scenario)
    if (
        manifest.get("scenario_id") != OPEN04_ROLE_TO_SCENARIO_ID[role]
        or manifest.get("scenario_sha256") != scenario_sha
    ):
        raise Open04CampaignError(
            f"{role} compiled scenario identity mismatch"
        )
    expected_marker = dict(
        _require_mapping(
            scenario.get("open04_campaign"),
            label=f"{role} scenario campaign marker",
        )
    )
    if manifest.get("open04_campaign") != expected_marker:
        raise Open04CampaignError(
            f"{role} compiled manifest lacks the exact campaign marker"
        )
    baseline_identity = _require_mapping(
        manifest.get("baseline"), label=f"{role} compiled baseline"
    )
    if dict(baseline_identity) != dict(OPEN04_BASELINE_IDENTITY):
        raise Open04CampaignError(
            f"{role} compiled manifest baseline identity differs"
        )
    simulation = _require_mapping(
        manifest.get("open04_simulation_contract"),
        label=f"{role} compiled OPEN-04 simulation contract",
    )
    expected_simulation_fields = {
        "schema_version",
        "frequency",
        "start_date",
        "end_date",
        "runtime_selected_curve_date_count",
        "runtime_selected_curve_date_set_sha256",
        "output_profile",
        "compression",
    }
    _require_exact_fields(
        simulation,
        expected_simulation_fields,
        label=f"{role} compiled OPEN-04 simulation contract",
    )
    if {
        "schema_version": simulation["schema_version"],
        "frequency": simulation["frequency"],
        "start_date": simulation["start_date"],
        "end_date": simulation["end_date"],
        "output_profile": simulation["output_profile"],
        "compression": simulation["compression"],
    } != {
        "schema_version": "tdcsim_open04_simulation_contract_v1",
        "frequency": OPEN04_FREQUENCY,
        "start_date": OPEN04_START_DATE,
        "end_date": OPEN04_END_DATE,
        "output_profile": OPEN04_OUTPUT_CONTRACT["profile"],
        "compression": OPEN04_OUTPUT_CONTRACT["compression"],
    }:
        raise Open04CampaignError(
            f"{role} compiled OPEN-04 clock/output contract differs"
        )

    input_records = _input_records(manifest, inputs, role=role)
    mix_path = inputs / _ISSUANCE_MIX_FILE
    if not mix_path.is_file():
        raise Open04CampaignError(
            f"{role} compiled issuance-mix artifact is missing"
        )
    mix_payload = _require_mapping(
        read_json(mix_path), label=f"{role} compiled issuance mix"
    )
    actual_mix_canonical_sha = canonical_json_sha256(mix_payload)
    if actual_mix_canonical_sha != expected_mix_sha256:
        raise Open04CampaignError(
            f"{role} compiled issuance mix differs from the supplied frozen hash"
        )
    mix_file_sha = sha256_file(mix_path)
    issuance_wam = _issuance_mix_wam(mix_payload, role=role)

    surface_path = inputs / _NOMINAL_SURFACE_FILE
    surface = _surface_identity(surface_path)
    if (
        simulation["runtime_selected_curve_date_count"]
        != surface["selected_curve_date_count"]
        or simulation["runtime_selected_curve_date_set_sha256"]
        != surface["selected_curve_date_set_sha256"]
    ):
        raise Open04CampaignError(
            f"{role} compiled selected curve-date contract differs"
        )

    perimeter = _require_mapping(
        manifest.get("open04_change_perimeter"),
        label=f"{role} compiled change perimeter",
    )
    economic = list(perimeter.get("economic_changed_paths") or [])
    physical = list(perimeter.get("physical_changed_inputs") or [])
    expected_economic_tuple, expected_physical_tuple = (
        open04_expected_change_perimeter(role)
    )
    expected_economic = list(expected_economic_tuple)
    expected_physical = list(expected_physical_tuple)
    if economic != expected_economic or physical != expected_physical:
        raise Open04CampaignError(
            f"{role} compiled change perimeter differs from the approved paths"
        )
    if perimeter.get("fixed_input_comparison_status") != "pass":
        raise Open04CampaignError(
            f"{role} compiled fixed-input comparison did not pass"
        )

    sidecar_path = inputs / NOMINAL_EVALUATED_SHOCK_FILE
    if role == "baseline":
        if (
            sidecar_path.exists()
            or "evaluated_nominal_curve" in manifest
            or verification.get("evaluated_nominal_curve") is not None
        ):
            raise Open04CampaignError(
                "OPEN-04 baseline must not carry a curve sidecar"
            )
        sidecar_sha = OPEN04_NO_SIDECAR_SENTINEL
        curve_delta = OPEN04_NO_CURVE_DELTA_SENTINEL
        signed_shock = 0.0
    else:
        if not sidecar_path.is_file():
            raise Open04CampaignError(
                f"{role} compiled curve sidecar is missing"
            )
        runtime_curve = build_evaluated_nominal_runtime_binding(
            inputs,
            start_date=OPEN04_START_DATE,
            end_date=OPEN04_END_DATE,
        )
        if not isinstance(runtime_curve, Mapping):
            raise Open04CampaignError(
                f"{role} compiled curve runtime binding is missing"
            )
        signed_shock = float(_ROLE_SHOCK_BP[role])
        compiled_curve = _require_mapping(
            manifest.get("evaluated_nominal_curve"),
            label=f"{role} compiled evaluated_nominal_curve",
        )
        if (
            runtime_curve.get("sidecar_sha256") != sha256_file(sidecar_path)
            or compiled_curve.get("signed_10y_shock_bp")
            != signed_shock
            or runtime_curve.get("short_end_bitwise_mismatch_count") != 0
            or float(
                runtime_curve.get(
                    "max_abs_analytic_delta_error_decimal", math.inf
                )
            )
            > 1e-12
        ):
            raise Open04CampaignError(
                f"{role} compiled curve evidence does not pass"
            )
        sidecar_sha = sha256_file(sidecar_path)
        curve_delta = _sha256(
            runtime_curve.get("evaluated_delta_sha256"),
            label=f"{role}.compiled_curve_delta_digest",
        )

    fixed_records = [
        record
        for record in input_records
        if record["path"] not in _OPEN04_CAMPAIGN_MUTABLE_INPUTS
    ]
    opening_records = [
        record
        for record in fixed_records
        if record["path"] in _OPENING_INPUT_FILES
    ]
    if [item["path"] for item in opening_records] != list(
        _OPENING_INPUT_FILES
    ):
        raise Open04CampaignError(
            f"{role} compiled opening-input set is incomplete"
        )
    return {
        "baseline_identity": dict(baseline_identity),
        "compiled_inputs_digest": _sha256(
            manifest.get("compiled_inputs_digest"),
            label=f"{role}.compiled_inputs_digest",
        ),
        "canonical_issuance_mix_sha256": actual_mix_canonical_sha,
        "issuance_mix_file_sha256": mix_file_sha,
        "curve_sidecar_sha256": sidecar_sha,
        "signed_10y_shock_bp": signed_shock,
        "compiled_curve_delta_digest": curve_delta,
        "economic_changed_paths": economic,
        "physical_changed_inputs": physical,
        "fixed_input_comparison_status": "pass",
        "fixed_input_records": fixed_records,
        "fixed_input_records_sha256": canonical_json_sha256(
            fixed_records
        ),
        "opening_input_records": opening_records,
        "curve_date_set_sha256": surface["date_set_sha256"],
        "tenor_set_sha256": surface["tenor_set_sha256"],
        "overall_new_issuance_wam_years": issuance_wam,
    }


def _input_records(
    manifest: Mapping[str, Any],
    inputs: Path,
    *,
    role: str,
) -> list[dict[str, Any]]:
    raw = manifest.get("input_hashes")
    if not isinstance(raw, list) or not raw:
        raise Open04CampaignError(
            f"{role} compiled input_hashes must be nonempty"
        )
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        record = _require_mapping(
            item, label=f"{role}.input_hashes[{index}]"
        )
        _require_exact_fields(
            record,
            {"path", "sha256", "bytes"},
            label=f"{role}.input_hashes[{index}]",
        )
        relative = _safe_relative_file(
            record["path"],
            label=f"{role}.input_hashes[{index}].path",
        )
        path = inputs / relative
        if not path.is_file():
            raise Open04CampaignError(
                f"{role} compiled input is missing: {relative}"
            )
        digest = _sha256(
            record["sha256"],
            label=f"{role}.input_hashes[{index}].sha256",
        )
        size = _nonnegative_int(
            record["bytes"],
            label=f"{role}.input_hashes[{index}].bytes",
        )
        if sha256_file(path) != digest or path.stat().st_size != size:
            raise Open04CampaignError(
                f"{role} compiled input record differs: {relative}"
            )
        records.append(
            {"path": relative, "sha256": digest, "bytes": size}
        )
    if records != sorted(records, key=lambda item: item["path"]):
        raise Open04CampaignError(
            f"{role} compiled input records are not canonically ordered"
        )
    if len({item["path"] for item in records}) != len(records):
        raise Open04CampaignError(
            f"{role} compiled input records contain duplicate paths"
        )
    return records


def _surface_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Open04CampaignError(
            "compiled nominal yield-curve surface is missing"
        )
    rows_by_date: dict[str, list[float]] = {}
    scenarios: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scenario_id",
            "curve_date",
            "tenor_years",
            "nominal_rate_decimal",
        }
        if not required <= set(reader.fieldnames or []):
            raise Open04CampaignError(
                "compiled nominal surface columns are incomplete"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise Open04CampaignError(
                    f"nominal surface row {row_number} is over-wide"
                )
            scenario_id = str(row["scenario_id"]).strip()
            curve_date = _iso_date(
                row["curve_date"],
                label=f"nominal surface row {row_number}.curve_date",
            )
            tenor = _finite_float(
                row["tenor_years"],
                label=f"nominal surface row {row_number}.tenor_years",
            )
            _finite_float(
                row["nominal_rate_decimal"],
                label=(
                    f"nominal surface row {row_number}."
                    "nominal_rate_decimal"
                ),
            )
            if not scenario_id or tenor <= 0.0:
                raise Open04CampaignError(
                    f"nominal surface row {row_number} is invalid"
                )
            scenarios.add(scenario_id)
            rows_by_date.setdefault(curve_date, []).append(tenor)
    if len(scenarios) != 1 or not rows_by_date:
        raise Open04CampaignError(
            "nominal surface must contain one nonempty scenario"
        )
    dates = sorted(rows_by_date)
    reference: list[float] | None = None
    for curve_date in dates:
        tenors = rows_by_date[curve_date]
        if any(
            left >= right for left, right in zip(tenors, tenors[1:])
        ):
            raise Open04CampaignError(
                f"nominal surface tenors are invalid at {curve_date}"
            )
        if reference is None:
            reference = list(tenors)
        elif tenors != reference:
            raise Open04CampaignError(
                "nominal surface tenor sets differ by curve date"
            )
    assert reference is not None
    if 2.0 not in reference or 10.0 not in reference:
        raise Open04CampaignError(
            "nominal surface lacks the exact 2y/10y anchors"
        )
    prior = [item for item in dates if item <= OPEN04_START_DATE]
    if not prior:
        raise Open04CampaignError(
            "nominal surface has no curve date at/before the horizon"
        )
    selected = [prior[-1]]
    selected.extend(
        item
        for item in dates
        if OPEN04_START_DATE < item <= OPEN04_END_DATE
    )
    return {
        "sha256": sha256_file(path),
        "date_set_sha256": canonical_json_sha256(dates),
        "tenor_set_sha256": canonical_json_sha256(reference),
        "selected_curve_date_count": len(selected),
        "selected_curve_date_set_sha256": canonical_json_sha256(
            selected
        ),
    }


def _issuance_mix_wam(
    payload: Mapping[str, Any],
    *,
    role: str,
) -> float:
    shares = _require_mapping(
        payload.get("security_shares"),
        label=f"{role} compiled issuance security_shares",
    )
    distributions = _require_mapping(
        payload.get("maturity_distributions"),
        label=f"{role} compiled issuance maturity_distributions",
    )
    categories = set(MARKETABLE_PREFERENCE_CATEGORIES)
    if set(shares) != categories or set(distributions) != categories:
        raise Open04CampaignError(
            f"{role} compiled issuance categories differ"
        )
    total_share = 0.0
    wam = 0.0
    for category in sorted(categories):
        security_share = _finite_float(
            shares[category],
            label=f"{role} issuance share {category}",
        )
        if security_share < 0.0:
            raise Open04CampaignError(
                f"{role} issuance share is negative for {category}"
            )
        total_share += security_share
        rows = distributions[category]
        if not isinstance(rows, list) or not rows:
            raise Open04CampaignError(
                f"{role} issuance distribution is empty for {category}"
            )
        within_share = 0.0
        within_wam = 0.0
        for index, raw in enumerate(rows):
            item = _require_mapping(
                raw,
                label=(
                    f"{role} issuance {category} distribution[{index}]"
                ),
            )
            _require_exact_fields(
                item,
                {"maturity_years", "share"},
                label=(
                    f"{role} issuance {category} distribution[{index}]"
                ),
            )
            maturity = _finite_float(
                item["maturity_years"],
                label=f"{role} issuance maturity",
            )
            weight = _finite_float(
                item["share"], label=f"{role} issuance maturity share"
            )
            if maturity <= 0.0 or weight < 0.0:
                raise Open04CampaignError(
                    f"{role} issuance maturity distribution is invalid"
                )
            within_share += weight
            within_wam += maturity * weight
        if not math.isclose(
            within_share, 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise Open04CampaignError(
                f"{role} issuance maturity shares do not sum to one"
            )
        wam += security_share * within_wam
    if not math.isclose(total_share, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Open04CampaignError(
            f"{role} issuance security shares do not sum to one"
        )
    declared = _finite_float(
        payload.get("weighted_average_maturity_years"),
        label=f"{role} weighted_average_maturity_years",
    )
    if not math.isclose(wam, declared, rel_tol=0.0, abs_tol=1e-12):
        raise Open04CampaignError(
            f"{role} compiled issuance WAM is stale"
        )
    return wam


def _calibration_worksheet(campaign_id: str) -> dict[str, Any]:
    return {
        "schema_version": OPEN04_CALIBRATION_WORKSHEET_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "scenario_contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
        "calibration_object": (
            "assumed_exogenous_10_year_nominal_yield_level_sensitivity"
        ),
        "central_10y_sensitivity_bp": 25.0,
        "nonprobabilistic_source_range_bp": {
            "lower": 14.0,
            "upper": 40.0,
            "interpretation": (
                "cross_method_source_range_not_confidence_interval"
            ),
        },
        "ati_source": {
            "pdf_sha256": OPEN04_ATI_PDF_SHA256,
            "role": "paper_facing_calibration_context_only",
        },
        "hou_source": {
            "pdf_sha256": OPEN04_HOU_PDF_SHA256,
            "role": "diagnostic_only_not_central_or_fallback",
        },
        "application": "post_baseline_evaluation",
        "short_end_rule": "unchanged_at_or_below_2_years",
        "long_end_rule": "log_tenor_ramp_to_10_year_then_flat",
        "time_profile": "constant_across_curve_dates",
        "no_retuning_label": "predeclared_no_retuning",
        "fallback_label": "failed_gate_no_relabel_no_recalibration",
        "claim_boundary": [
            "scenario_not_forecast_not_causal_not_optimization",
            "no_identified_holder_behavior",
            "no_estimated_deposit_response",
            "no_treasury_choice_rule",
        ],
    }


def _write_new_or_identical_json(path: Path, value: Any) -> None:
    expected = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != expected:
            raise Open04CampaignError(
                f"frozen campaign artifact conflicts: {path.name}"
            )
        return
    path.write_bytes(expected)
    if path.read_bytes() != expected:
        raise Open04CampaignError(
            f"campaign JSON writer was not deterministic: {path.name}"
        )


def _load_campaign_contract(
    campaign_root: str | Path,
    *,
    expected_contract_sha256: str,
    contract_relative_path: str,
) -> tuple[Path, dict[str, Any], str]:
    root = Path(campaign_root).expanduser().resolve()
    contract_name = _direct_child_name(
        contract_relative_path, label="contract_relative_path"
    )
    expected = _sha256(
        expected_contract_sha256, label="expected_contract_sha256"
    )
    path = root / contract_name
    if not path.is_file():
        raise Open04CampaignError("frozen OPEN-04 contract is missing")
    actual = sha256_file(path)
    if actual != expected:
        raise Open04CampaignError(
            "frozen OPEN-04 contract hash differs from the external anchor"
        )
    contract = dict(
        _require_mapping(read_json(path), label="OPEN-04 campaign contract")
    )
    _validate_campaign_contract_mapping(root, contract)
    return root, contract, actual


def _validate_campaign_contract_mapping(
    root: Path,
    contract: Mapping[str, Any],
) -> None:
    _require_exact_fields(
        contract,
        {
            "schema_version",
            "contract_id",
            "contract_status",
            "signature_reference",
            "campaign_id",
            "campaign_root_identity",
            "execution_order",
            "no_retuning",
            "failed_gate_action",
            "execution_contract",
            "calibration_worksheet",
            "common_identity",
            "roles",
        },
        label="OPEN-04 campaign contract",
    )
    if (
        contract.get("schema_version") != OPEN04_CAMPAIGN_SCHEMA_VERSION
        or contract.get("contract_id") != OPEN04_CAMPAIGN_CONTRACT_ID
        or contract.get("contract_status") != "owner_signed_frozen"
        or not str(contract.get("signature_reference") or "").strip()
        or contract.get("execution_order")
        != list(OPEN04_CAMPAIGN_ROLES)
        or contract.get("no_retuning") is not True
        or contract.get("failed_gate_action")
        != "failed_predeclared_paired_sign_gate"
    ):
        raise Open04CampaignError(
            "OPEN-04 frozen campaign authority/order contract differs"
        )
    campaign_id = _campaign_id(contract["campaign_id"])
    execution = _require_mapping(
        contract.get("execution_contract"),
        label="campaign execution_contract",
    )
    if dict(execution) != dict(_EXECUTION_CONTRACT):
        raise Open04CampaignError(
            "OPEN-04 frozen execution/resource contract differs"
        )
    common = _require_mapping(
        contract.get("common_identity"), label="campaign common_identity"
    )
    _require_exact_fields(
        common,
        set(_CONTRACT_COMMON_IDENTITY_KEYS),
        label="campaign common_identity",
    )
    if {
        "package_id": common["package_id"],
        "package_sha256": common["baseline_package_sha256"],
        "manifest_sha256": common["baseline_manifest_sha256"],
        "release_attestation_sha256": common[
            "release_attestation_sha256"
        ],
    } != dict(OPEN04_BASELINE_IDENTITY):
        raise Open04CampaignError(
            "campaign common identity differs from authoritative baseline"
        )
    _validate_code_identity(
        {key: common[key] for key in _CODE_IDENTITY_KEYS}
    )
    if (
        common["start_date"] != OPEN04_START_DATE
        or common["end_date"] != OPEN04_END_DATE
        or common["frequency"] != OPEN04_FREQUENCY
        or common["aggregation_clock_id"]
        != OPEN04_AGGREGATION_CLOCK_ID
        or common["output_profile"]
        != OPEN04_OUTPUT_CONTRACT["profile"]
        or common["compression"]
        != OPEN04_OUTPUT_CONTRACT["compression"]
    ):
        raise Open04CampaignError(
            "campaign common clock/output identity differs"
        )
    opening_records = _validate_frozen_records(
        common["opening_input_records"], label="opening_input_records"
    )
    fixed_records = _validate_frozen_records(
        common["fixed_input_records"], label="fixed_input_records"
    )
    if (
        [item["path"] for item in opening_records]
        != list(_OPENING_INPUT_FILES)
        or canonical_json_sha256(opening_records)
        != common["opening_identity_sha256"]
        or canonical_json_sha256(fixed_records)
        != common["fixed_input_records_sha256"]
        or not all(item in fixed_records for item in opening_records)
    ):
        raise Open04CampaignError(
            "campaign frozen opening/fixed input records differ"
        )

    worksheet_ref = _require_mapping(
        contract.get("calibration_worksheet"),
        label="campaign calibration_worksheet",
    )
    _require_exact_fields(
        worksheet_ref,
        {"relative_path", "sha256", "canonical_sha256"},
        label="campaign calibration_worksheet",
    )
    worksheet_name = _direct_child_name(
        worksheet_ref["relative_path"],
        label="campaign calibration_worksheet.relative_path",
    )
    worksheet_path = root / worksheet_name
    if (
        not worksheet_path.is_file()
        or sha256_file(worksheet_path) != worksheet_ref["sha256"]
        or worksheet_ref["sha256"]
        != common["calibration_worksheet_sha256"]
    ):
        raise Open04CampaignError(
            "campaign calibration worksheet byte binding differs"
        )
    worksheet = _require_mapping(
        read_json(worksheet_path), label="campaign calibration worksheet"
    )
    if (
        dict(worksheet) != _calibration_worksheet(campaign_id)
        or canonical_json_sha256(worksheet)
        != worksheet_ref["canonical_sha256"]
    ):
        raise Open04CampaignError(
            "campaign calibration worksheet contract differs"
        )

    roles = _role_mapping(contract.get("roles"), label="campaign roles")
    run_paths: dict[str, str] = {}
    receipt_paths: set[str] = set()
    source_paths: set[str] = set()
    for role in OPEN04_CAMPAIGN_ROLES:
        item = _require_mapping(
            roles[role], label=f"campaign roles.{role}"
        )
        _require_exact_fields(
            item,
            set(_CONTRACT_ROLE_KEYS),
            label=f"campaign roles.{role}",
        )
        if item["scenario_id"] != OPEN04_ROLE_TO_SCENARIO_ID[role]:
            raise Open04CampaignError(
                f"campaign {role} scenario ID differs"
            )
        source_name = _direct_child_name(
            item["scenario_source_relative_path"],
            label=f"campaign {role}.scenario_source_relative_path",
        )
        if source_name != _SCENARIO_SOURCE_NAMES[role]:
            raise Open04CampaignError(
                f"campaign {role} source basename differs"
            )
        source_path = root / source_name
        if (
            not source_path.is_file()
            or sha256_file(source_path)
            != _sha256(
                item["scenario_source_sha256"],
                label=f"campaign {role}.scenario_source_sha256",
            )
        ):
            raise Open04CampaignError(
                f"campaign {role} scenario source byte hash differs"
            )
        scenario = _require_mapping(
            read_json(source_path),
            label=f"campaign {role} scenario source",
        )
        marker = validate_open04_scenario_contract(scenario)
        if (
            marker.role != role
            or canonical_json_sha256(scenario)
            != _sha256(
                item["scenario_sha256"],
                label=f"campaign {role}.scenario_sha256",
            )
        ):
            raise Open04CampaignError(
                f"campaign {role} scenario canonical identity differs"
            )
        run_paths[role] = _role_run_relative_path(
            item["run_relative_path"],
            label=f"campaign {role}.run_relative_path",
        )
        receipt_paths.add(
            _direct_child_name(
                item["controller_completion_receipt_relative_path"],
                label=(
                    f"campaign {role}."
                    "controller_completion_receipt_relative_path"
                ),
            )
        )
        source_paths.add(source_name)
        _sha256(
            item["canonical_issuance_mix_sha256"],
            label=f"campaign {role}.canonical_issuance_mix_sha256",
        )
        _sha256(
            item["issuance_mix_file_sha256"],
            label=f"campaign {role}.issuance_mix_file_sha256",
        )
        _sha256(
            item["compiled_inputs_digest"],
            label=f"campaign {role}.compiled_inputs_digest",
        )
        _sha256(
            item["fixed_input_records_sha256"],
            label=f"campaign {role}.fixed_input_records_sha256",
        )
        _sha256(
            item["curve_date_set_sha256"],
            label=f"campaign {role}.curve_date_set_sha256",
        )
        _sha256(
            item["tenor_set_sha256"],
            label=f"campaign {role}.tenor_set_sha256",
        )
        if (
            item["fixed_input_records_sha256"]
            != common["fixed_input_records_sha256"]
            or not math.isfinite(
                _finite_float(
                    item["overall_new_issuance_wam_years"],
                    label=f"campaign {role}.overall WAM",
                )
            )
        ):
            raise Open04CampaignError(
                f"campaign {role} fixed identity/WAM differs"
            )
        expected_economic_tuple, expected_physical_tuple = (
            open04_expected_change_perimeter(role)
        )
        expected_economic = list(expected_economic_tuple)
        expected_physical = list(expected_physical_tuple)
        if (
            item["economic_changed_paths"] != expected_economic
            or item["physical_changed_inputs"] != expected_physical
            or float(item["signed_10y_shock_bp"])
            != _ROLE_SHOCK_BP[role]
        ):
            raise Open04CampaignError(
                f"campaign {role} change/sign perimeter differs"
            )
        if role == "baseline":
            if (
                item["curve_sidecar_sha256"]
                != OPEN04_NO_SIDECAR_SENTINEL
                or item["compiled_curve_delta_digest"]
                != OPEN04_NO_CURVE_DELTA_SENTINEL
            ):
                raise Open04CampaignError(
                    "campaign baseline no-sidecar sentinels differ"
                )
        else:
            _sha256(
                item["curve_sidecar_sha256"],
                label=f"campaign {role}.curve_sidecar_sha256",
            )
            _sha256(
                item["compiled_curve_delta_digest"],
                label=f"campaign {role}.compiled_curve_delta_digest",
            )
    if (
        len(set(run_paths.values())) != len(OPEN04_CAMPAIGN_ROLES)
        or len(receipt_paths) != len(OPEN04_CAMPAIGN_ROLES)
        or len(source_paths) != len(OPEN04_CAMPAIGN_ROLES)
        or campaign_root_identity(campaign_id, run_paths)
        != contract["campaign_root_identity"]
    ):
        raise Open04CampaignError(
            "campaign logical root/run mapping differs"
        )


def _scenario_sources(
    root: Path,
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    roles = contract["roles"]
    return {
        role: dict(
            _require_mapping(
                read_json(
                    root / roles[role]["scenario_source_relative_path"]
                ),
                label=f"{role} scenario source",
            )
        )
        for role in OPEN04_CAMPAIGN_ROLES
    }


def _compare_compiled_evidence_to_contract(
    contract: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> None:
    common = contract["common_identity"]
    if (
        compiled["fixed_input_records"] != common["fixed_input_records"]
        or compiled["fixed_input_records_sha256"]
        != common["fixed_input_records_sha256"]
        or compiled["opening_input_records"]
        != common["opening_input_records"]
        or compiled["opening_identity_sha256"]
        != common["opening_identity_sha256"]
    ):
        raise Open04CampaignError(
            "compiled common fixed/opening inputs differ from frozen contract"
        )
    compared_fields = {
        "canonical_issuance_mix_sha256",
        "issuance_mix_file_sha256",
        "compiled_inputs_digest",
        "curve_sidecar_sha256",
        "signed_10y_shock_bp",
        "compiled_curve_delta_digest",
        "economic_changed_paths",
        "physical_changed_inputs",
        "fixed_input_records_sha256",
        "curve_date_set_sha256",
        "tenor_set_sha256",
        "overall_new_issuance_wam_years",
    }
    for role in OPEN04_CAMPAIGN_ROLES:
        declared = contract["roles"][role]
        observed = compiled["roles"][role]
        for field in compared_fields:
            if observed[field] != declared[field]:
                raise Open04CampaignError(
                    f"{role} compiled evidence differs from frozen {field}"
                )


def _post_common_identity(
    contract_common: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: contract_common[key] for key in _POST_COMMON_IDENTITY_KEYS
    }


def _validate_frozen_records(
    value: Any,
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise Open04CampaignError(f"{label} must be an array")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = _require_mapping(raw, label=f"{label}[{index}]")
        _require_exact_fields(
            item, {"path", "sha256", "bytes"}, label=f"{label}[{index}]"
        )
        records.append(
            {
                "path": _safe_relative_file(
                    item["path"], label=f"{label}[{index}].path"
                ),
                "sha256": _sha256(
                    item["sha256"], label=f"{label}[{index}].sha256"
                ),
                "bytes": _nonnegative_int(
                    item["bytes"], label=f"{label}[{index}].bytes"
                ),
            }
        )
    if records != sorted(records, key=lambda item: item["path"]):
        raise Open04CampaignError(f"{label} must be canonically ordered")
    if len({item["path"] for item in records}) != len(records):
        raise Open04CampaignError(f"{label} contains duplicate paths")
    return records


def _safe_relative_file(value: Any, *, label: str) -> str:
    text = str(value).replace("\\", "/")
    path = Path(text)
    if (
        not text
        or text.endswith("/")
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise Open04CampaignError(f"{label} is not a safe relative file")
    return "/".join(path.parts)


def _sha256(value: Any, *, label: str) -> str:
    digest = str(value)
    if (
        len(digest) != 64
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise Open04CampaignError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return digest


def _commit_sha(value: Any, *, label: str) -> str:
    digest = str(value)
    if (
        len(digest) != 40
        or digest == "0" * 40
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise Open04CampaignError(
            f"{label} must be a release-bound lowercase Git SHA"
        )
    return digest


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Open04CampaignError(
            f"{label} must be a nonnegative integer"
        )
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise Open04CampaignError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Open04CampaignError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise Open04CampaignError(f"{label} must be finite")
    return number


def _iso_date(value: Any, *, label: str) -> str:
    text = str(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise Open04CampaignError(
            f"{label} must be an ISO-8601 date"
        ) from exc
    if parsed.isoformat() != text:
        raise Open04CampaignError(
            f"{label} must be a canonical ISO-8601 date"
        )
    return text


__all__ = [
    "OPEN04_AGGREGATION_CLOCK_ID",
    "OPEN04_ANTISYMMETRY_TOLERANCE_DECIMAL",
    "OPEN04_ATI_PDF_SHA256",
    "OPEN04_BASELINE_IDENTITY",
    "OPEN04_BASELINE_RELEASE_COMMIT_SHA",
    "OPEN04_BASELINE_REQUIREMENTS_LOCK_SHA256",
    "OPEN04_CALIBRATION_WORKSHEET_REFERENCE_KEYS",
    "OPEN04_CALIBRATION_WORKSHEET_SCHEMA_VERSION",
    "OPEN04_CAMPAIGN_CONTRACT_ID",
    "OPEN04_CAMPAIGN_ROLES",
    "OPEN04_CAMPAIGN_SCHEMA_VERSION",
    "OPEN04_CANDIDATE_A_HOLDER_SHIFT_PP",
    "OPEN04_CONTRACT_COMMON_IDENTITY_KEYS",
    "OPEN04_CONTRACT_ROLE_KEYS",
    "OPEN04_CONTRACT_TOP_LEVEL_KEYS",
    "OPEN04_CONTROLLER_RECEIPT_KEYS",
    "OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION",
    "OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION",
    "OPEN04_END_DATE",
    "OPEN04_EXECUTION_CONTRACT_KEYS",
    "OPEN04_FREQUENCY",
    "OPEN04_HOU_PDF_SHA256",
    "OPEN04_HOLDER_PROFILE_FILE",
    "OPEN04_NO_CURVE_DELTA_SENTINEL",
    "OPEN04_NO_SIDECAR_SENTINEL",
    "OPEN04_PAIR_GATE_KEYS",
    "OPEN04_POST_COMMON_IDENTITY_KEYS",
    "OPEN04_POST_RECEIPT_KEYS",
    "OPEN04_POST_ROLE_KEYS",
    "OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION",
    "OPEN04_PRE_RUN_RECEIPT_SCHEMA_VERSION",
    "OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256",
    "OPEN04_ROLE_TO_PROVENANCE",
    "OPEN04_ROLE_TO_SCENARIO_ID",
    "OPEN04_SIGN_GATE_THRESHOLD_BIL",
    "OPEN04_START_DATE",
    "Open04CampaignError",
    "Open04CampaignMarker",
    "build_open04_scenario_mappings",
    "campaign_root_identity",
    "freeze_open04_campaign_contract",
    "open04_expected_change_perimeter",
    "parse_open04_campaign_marker",
    "requires_open04_strict_execution",
    "validate_open04_campaign_contract",
    "validate_open04_campaign_contract_mapping",
    "validate_open04_campaign_post_receipt_mapping",
    "validate_open04_scenario_contract",
    "verify_open04_campaign",
    "verify_open04_campaign_post_run",
    "verify_open04_campaign_pre_run",
]
