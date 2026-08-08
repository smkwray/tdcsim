from __future__ import annotations

import csv
import gzip
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tdcsim_cbo._json import sha256_file
from tdcsim_cbo.bounded_output import (
    AGGREGATION_CLOCK_ID,
    ANNUAL_COLUMNS,
    COMMITMENT_COLUMNS,
    EVENT_SCHEMA_VERSION,
    ISSUANCE_COLUMNS,
    LEDGER_COLUMNS,
    PAYMENT_COLUMNS,
    PRINCIPAL_COLUMNS,
    RESOURCE_COLUMNS,
    STOCK_CLOSURE_COLUMNS,
)
from tdcsim_cbo.manifest import RUN_CLAIM_BOUNDARY, RUN_UNSUPPORTED_COMPONENTS
from tdcsim_cbo.output import (
    HANDOFF_TABLE_COLUMNS,
    TDC_AMOUNT_BASIS,
    TDC_COMPONENT_SPECS,
    TDC_HOLDER_SCOPE,
    TDC_OVERLAP_POLICY,
)
from tdcsim_cbo.process_watchdog import THREAD_LIMIT_ENVIRONMENT_VARIABLES
from tdcsim_cbo.verifier import (
    VerificationError,
    _BOUNDED_ARTIFACT_SPECS,
    _BOUNDED_EXECUTION_MILESTONES,
    _BOUNDED_WATCHDOG_ACCEPTED_MILESTONES,
    _validate_run_manifest_schema,
    _verify_bounded_engine_replay,
    _verify_bounded_outputs,
    _verify_bounded_result_limits,
    _verify_cbo_reference_financing_results,
    _verify_open04_output_contract,
    _verify_release_claims,
)


START = "2026-06-21"
END = "2026-06-22"
EVENT_ROOT = "a" * 64
PERIOD_ROOT = "b" * 64
FINAL_STATE = "c" * 64


def _with_columns(columns: list[str], **values: Any) -> dict[str, Any]:
    return {column: values.get(column, "") for column in columns}


def _compact_rows() -> dict[str, list[dict[str, Any]]]:
    ledger = _with_columns(
        LEDGER_COLUMNS,
        period_start=START,
        period_end=END,
        event_type="fiscal",
        leg_type="fiscal_distribution",
        holder_sector="Private",
        holder_subsector="domestic_nonbank_deposit_funded",
        counterparty_sector="Treasury",
        counterparty_subsector="",
        route_holder_sector="Private",
        route_holder_subsector="domestic_nonbank_deposit_funded",
        instrument_type="cash",
        maturity_bucket="not_applicable",
        accounting_basis="cash",
        settlement_scope="treasury_disbursement",
        is_intragovernmental=False,
        face_stock_change_bil=0.0,
        adjusted_principal_change_bil=0.0,
        route_face_stock_change_bil=0.0,
        route_adjusted_principal_change_bil=0.0,
        treasury_cash_change_bil=1.0,
        reserve_change_bil=0.0,
        deposit_change_bil=1.0,
        event_count=1,
    )

    accounting_columns = HANDOFF_TABLE_COLUMNS["tdcsim_accounting_closure"]
    accounting_values = {
        column: 0.0
        for column in accounting_columns
        if column not in {"period_start", "period_end", "closure_basis"}
    }
    accounting_values.update(
        {
            "period_start": START,
            "period_end": END,
            "opening_face_stock_bil": 100.0,
            "closing_face_stock_bil": 100.0,
            "opening_adjusted_principal_stock_bil": 100.0,
            "closing_adjusted_principal_stock_bil": 100.0,
            "opening_treasury_cash_bil": 10.0,
            "journal_treasury_cash_change_bil": 1.0,
            "closing_treasury_cash_bil": 11.0,
            "journal_deposit_change_bil": 1.0,
            "reported_deposit_change_bil": 1.0,
            "holder_debt_total_bil": 100.0,
            "instrument_debt_total_bil": 100.0,
            "aggregate_debt_bil": 100.0,
            "closure_basis": (
                "independent_opening_and_closing_state_snapshots"
            ),
        }
    )
    accounting = _with_columns(accounting_columns, **accounting_values)

    stock_rows = []
    for axis, holder, subsector in (
        ("holder", "Private", "domestic_nonbank_deposit_funded"),
        ("instrument", "", ""),
    ):
        stock_rows.append(
            _with_columns(
                STOCK_CLOSURE_COLUMNS,
                period_start=START,
                period_end=END,
                axis=axis,
                holder_sector=holder,
                holder_subsector=subsector,
                instrument_type="Fixed",
                maturity_bucket="bills",
                debt_scope="all_active_treasury",
                opening_face_stock_bil=100.0,
                event_face_stock_change_bil=0.0,
                closing_face_stock_bil=100.0,
                face_stock_closure_error_bil=0.0,
                opening_adjusted_principal_stock_bil=100.0,
                event_adjusted_principal_change_bil=0.0,
                closing_adjusted_principal_stock_bil=100.0,
                adjusted_principal_closure_error_bil=0.0,
                opening_debt_stock_bil=100.0,
                event_debt_stock_change_bil=0.0,
                closing_debt_stock_bil=100.0,
                debt_stock_closure_error_bil=0.0,
            )
        )

    route = _with_columns(
        HANDOFF_TABLE_COLUMNS["tdcsim_tdc_principal_route_stock_closure"],
        period_start=START,
        period_end=END,
        route_holder_sector="Private",
        route_holder_subsector="domestic_nonbank_deposit_funded",
        instrument_type="Fixed",
        maturity_bucket="bills",
        debt_scope="controlled_public_marketable",
        opening_route_stock_bil=100.0,
        route_face_issued_bil=0.0,
        route_face_redeemed_bil=0.0,
        route_journal_face_change_bil=0.0,
        route_journal_adjusted_principal_change_bil=0.0,
        route_stock_residual_or_indexation_bil=0.0,
        closing_route_stock_bil=100.0,
        closure_identity_error_bil=0.0,
        route_stock_basis="tdc_principal_settlement_route",
        residual_basis="none_fail_closed_no_unrestricted_residual",
    )

    summary_columns = HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_summary"]
    summary_values = {
        column: 0.0
        for column in summary_columns
        if column.endswith("_bil")
    }
    summary_values.update(
        {
            "period_start": START,
            "period_end": END,
            "tdc_change_bil": 1.0,
            "tdc_fiscal_flow_bil": 1.0,
            "tdc_change_ex_overlap_bil": 1.0,
            "component_sum_bil": 1.0,
            "component_sum_error_bil": 0.0,
            "tdc_amount_basis": TDC_AMOUNT_BASIS,
            "holder_allocation_scope": TDC_HOLDER_SCOPE,
            "overlap_policy": TDC_OVERLAP_POLICY,
        }
    )
    tdc_summary = _with_columns(summary_columns, **summary_values)

    spec = TDC_COMPONENT_SPECS[0]
    tdc_component = _with_columns(
        HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_components"],
        period_start=START,
        period_end=END,
        component_id=f"tdc|{END}|{spec['component_key']}",
        component_key=spec["component_key"],
        component_family=spec["component_family"],
        holder_sector=spec["holder_sector"],
        holder_subsector=spec["holder_subsector"],
        instrument_type=spec["instrument_type"],
        payment_type=spec["payment_type"],
        accounting_basis=spec["accounting_basis"],
        amount_bil=1.0,
        is_additive_to_tdc_change=spec["is_additive_to_tdc_change"],
        enters_direct_interest_support=spec[
            "enters_direct_interest_support"
        ],
        enters_tdc_deposit_support_default=spec[
            "enters_tdc_deposit_support_default"
        ],
        tdc_amount_basis=TDC_AMOUNT_BASIS,
        overlap_policy=TDC_OVERLAP_POLICY,
    )

    debt_columns = HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"]
    debt_values = {
        column: 0.0 for column in debt_columns if column.endswith("_bil")
    }
    debt_values.update(
        {
            "date": END,
            "cbo_public_debt_target_bil": 100.0,
            "controlled_public_marketable_target_bil": 100.0,
            "controlled_debt_pre_issuance_bil": 100.0,
            "controlled_debt_post_issuance_bil": 100.0,
            "target_error_bil": 0.0,
            "funding_mode": "cbo_target",
            "intragovernmental_treatment": "excluded",
            "fed_held_treasury_treatment": "included",
            "public_nonmarketable_treatment": "bridge",
        }
    )
    debt_bridge = _with_columns(debt_columns, **debt_values)

    metrics = _with_columns(
        HANDOFF_TABLE_COLUMNS["tdcsim_scenario_metrics"],
        date=END,
        new_issuance_wam_years=0.0,
        outstanding_controlled_wam_years=10.0,
        new_issuance_bill_share=0.0,
        outstanding_controlled_bill_share=0.2,
        new_issuance_short_maturity_share=0.0,
        outstanding_controlled_short_maturity_share=0.3,
        short_maturity_cutoff_years=1.0,
    )

    commitment = _with_columns(
        COMMITMENT_COLUMNS,
        period_start=START,
        period_end=END,
        event_seq_start=1,
        event_seq_end=1,
        event_count=1,
        period_event_root_sha256=PERIOD_ROOT,
        whole_run_root_through_period_sha256=EVENT_ROOT,
        event_type_counts_json='{"fiscal":1}',
    )

    annual = _with_columns(
        ANNUAL_COLUMNS,
        period_label="FY2026_PARTIAL_OPENING",
        period_start=START,
        period_end=END,
        is_partial_period=True,
        coverage_days=1,
        expected_coverage_days=365,
        aggregation_clock_id=AGGREGATION_CLOCK_ID,
        tdc_change_bil=1.0,
        overlap_cashflow_bil=0.0,
        tdc_change_ex_overlap_bil=1.0,
        cumulative_tdc_change_bil=1.0,
        cumulative_overlap_cashflow_bil=0.0,
        cumulative_tdc_change_ex_overlap_bil=1.0,
        interest_outlay_bil=0.0,
        issue_discount_cost_bil=0.0,
        nonmarketable_interest_capitalized_bil=0.0,
        tips_inflation_accretion_bil=0.0,
        modeled_financing_cost_bil=0.0,
        cumulative_interest_outlay_bil=0.0,
        cumulative_issue_discount_cost_bil=0.0,
        cumulative_nonmarketable_interest_capitalized_bil=0.0,
        cumulative_tips_inflation_accretion_bil=0.0,
        cumulative_modeled_financing_cost_bil=0.0,
        modeled_financing_cost_basis=(
            "nominal_model_cost_incurred_within_simulation_horizon"
        ),
        modeled_financing_cost_units="billions_of_nominal_dollars",
        cumulative_basis="since_simulation_origin",
        new_issuance_face_bil=0.0,
        new_issuance_original_term_face_years_bil=0.0,
        new_issuance_bill_face_bil=0.0,
        new_issuance_short_face_bil=0.0,
    )

    resources = [
        _with_columns(
            RESOURCE_COLUMNS,
            sample_kind="admission",
            period_end=START,
            rss_bytes=100,
            peak_rss_bytes=100,
            host_available_bytes=10_000,
            process_cpu_seconds=0.25,
            portfolio_rows=2,
            active_portfolio_rows=2,
            event_count=0,
            current_key_cardinality=2,
            max_key_cardinality=2,
            bytes_written=0,
        ),
        _with_columns(
            RESOURCE_COLUMNS,
            sample_kind="period_close",
            period_end=END,
            rss_bytes=100,
            peak_rss_bytes=100,
            host_available_bytes=10_000,
            process_cpu_seconds=0.5,
            portfolio_rows=2,
            active_portfolio_rows=2,
            event_count=1,
            current_key_cardinality=2,
            max_key_cardinality=2,
            bytes_written=100,
        ),
    ]

    return {
        "ledger": [ledger],
        "issuance": [],
        "principal": [],
        "payment": [],
        "accounting": [accounting],
        "stock": stock_rows,
        "route": [route],
        "tdc_summary": [tdc_summary],
        "tdc_components": [tdc_component],
        "debt_bridge": [debt_bridge],
        "scenario_metrics": [metrics],
        "commitments": [commitment],
        "annual": [annual],
        "resources": resources,
    }


def _write_gzip_csv(
    path: Path, columns: list[str], rows: list[dict[str, Any]]
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _record(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": row_count,
    }


def _package(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    root = tmp_path / "run"
    outputs = root / "outputs"
    outputs.mkdir(parents=True)
    rows = _compact_rows()
    records: dict[str, dict[str, Any]] = {}
    for name, (filename, columns) in _BOUNDED_ARTIFACT_SPECS.items():
        path = outputs / filename
        _write_gzip_csv(path, columns, rows[name])
        records[name] = _record(path, row_count=len(rows[name]))

    bounded = {
        "event_schema_version": EVENT_SCHEMA_VERSION,
        "event_count": 1,
        "event_root_sha256": EVENT_ROOT,
        "final_state_sha256": FINAL_STATE,
        "period_count": 1,
        "portfolio_row_budget": 10,
        "max_portfolio_rows": 2,
        "max_active_portfolio_rows": 2,
        "key_cardinality_budget": 10,
        "max_key_cardinality": 2,
        "peak_rss_bytes": 100,
        "memory_thresholds": {
            "minimum_available_bytes": 0,
            "acceptance_peak_rss_bytes": 1_000,
            "application_abort_rss_bytes": 2_000,
            "parent_graceful_stop_rss_bytes": 3_000,
            "parent_kill_rss_bytes": 4_000,
        },
        "deterministic_artifacts": {
            name: deepcopy(record)
            for name, record in records.items()
            if name != "resources"
        },
        "resource_artifact": deepcopy(records["resources"]),
        "invariant_status": "pass",
    }
    summary = {
        "evidence_profile": "bounded_period_closure_v1",
        "verification_grade": "bounded_replay_v1",
        "event_count": 1,
        "event_root_sha256": EVENT_ROOT,
        "final_state_sha256": FINAL_STATE,
        "peak_rss_bytes": 100,
        "portfolio_row_budget": 10,
        "max_portfolio_rows": 2,
        "max_key_cardinality": 2,
    }
    (outputs / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    output_manifest: dict[str, Any] = {
        "profile": "compact",
        "compression": "gzip",
        "evidence_profile": "bounded_period_closure_v1",
        "verification_grade": "bounded_replay_v1",
        "deterministic_evidence_artifacts": deepcopy(
            bounded["deterministic_artifacts"]
        ),
    }
    output_manifest.update(
        {name: deepcopy(record) for name, record in records.items()}
    )
    public_outputs = [
        {
            "logical_name": record["path"],
            "relative_path": f"outputs/{record['path']}",
            "sha256": record["sha256"],
            "bytes": record["bytes"],
        }
        for record in records.values()
    ]
    manifest: dict[str, Any] = {
        "schema_version": "tdcsim_cbo_scenario_run_manifest_v2",
        "run_id": "test-bounded-run",
        "status": "complete",
        "verification_grade": "bounded_replay_v1",
        "generated_at_utc": "2026-07-29T00:00:00+00:00",
        "baseline": {
            "package_id": "baseline",
            "package_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
            "release_attestation_sha256": "f" * 64,
        },
        "scenario": {
            "scenario_id": "test",
            "canonical_sha256": "1" * 64,
            "source_file_sha256": "2" * 64,
            "relative_path": "scenario.json",
            "referenced_file_hashes": {},
            "referenced_files": [],
        },
        "code_environment": {},
        "claim_boundary": deepcopy(RUN_CLAIM_BOUNDARY),
        "coupling_decisions": {},
        "compiled_manifest": "compile/tdcsim_cbo_compiled_manifest.json",
        "compiled_inputs_digest": "3" * 64,
        "compiled_inputs": [
            {
                "logical_name": "input.csv",
                "relative_path": "compile/forecast_inputs/input.csv",
                "sha256": "4" * 64,
                "bytes": 1,
            }
        ],
        "simulation": {
            "start_date": START,
            "end_date": END,
            "frequency": "daily",
        },
        "fiscal_incidence_policy_id": "test-policy",
        "outputs": public_outputs,
        "output_manifest": output_manifest,
        "output_hashes": [],
        "boundary_checks": {},
        "validation": {
            "status": "pass",
            "gates": [{"id": "test", "status": "pass"}],
            "invariants": [
                {
                    "id": "bounded_period_closure",
                    "status": "pass",
                    "observed": "periods=1",
                },
                {
                    "id": "portfolio_row_budget",
                    "status": "pass",
                    "observed": 2,
                    "limit": 10,
                },
                {
                    "id": "key_cardinality_budget",
                    "status": "pass",
                    "observed": 2,
                    "limit": 10,
                },
            ],
        },
        "unsupported_components": list(RUN_UNSUPPORTED_COMPONENTS),
        "evidence_profile": "bounded_period_closure_v1",
        "aggregation_clock": {
            "clock_id": AGGREGATION_CLOCK_ID,
            "bucket_label_rule": "federal_fiscal_year_by_period_end",
            "opening_partial_policy": "unannualized_separate_partial_bucket",
            "snapshot_date_rule": "exact_september_30",
        },
        "bounded_evidence": bounded,
        "execution_milestones": list(_BOUNDED_EXECUTION_MILESTONES),
        "execution_contract": {
            "schema_version": "tdcsim_cbo_execution_contract_v1",
            "single_writer_claim": True,
            "writer_claim_file_name": ".tdcsim-cbo-bounded-writer.claim",
            "writer_claim_scope": "output_parent",
            "writer_claim_scope_id": "library-output-parent",
            "one_scenario_per_worker": True,
            "process_pool_enabled": False,
            "parent_watchdog_required": False,
            "scenario_process_mode": "library_call",
            "numerical_thread_environment": {
                name: "" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
            },
        },
    }
    result_flow_columns = {
        "NewDebtIssued",
        "AuctionProceeds",
        "PrincipalPaid_Bonds",
        "InterestOutlay_Period",
        "IssueDiscountCost_Period",
        "NonMarketableInterestCapitalized_Period",
        "TIPSInflationAccretion_Period",
        "FinancingCost_Period",
        "TDC_Change",
        "TDC_FiscalFlow",
        "TDC_DebtService",
        "TDC_AuctionAbsorption",
        "TDC_SecondaryTrades",
        "TDC_Other",
        "TDC_PrincipalToDU",
        "TDC_PrincipalCashToDU",
        "TDC_InterestToDU",
        "TDC_PrincipalToDU_DomesticNonbank",
        "TDC_PrincipalToDU_MMF",
        "TDC_PrincipalCashToDU_DomesticNonbank",
        "TDC_PrincipalCashToDU_MMF",
        "TDC_PrincipalCashToDU_MMFPlumbing",
        "TDC_GrossIssuanceProceedsAbsorbedByDU",
        "TDC_NetPrincipalIssuanceCashflowToDU",
        "CBOBuybackFaceRetired",
        "CBOBuybackCashPaid",
        "OutstandingControlledWAM",
        "OutstandingControlledBillShare",
        "OutstandingControlledShortMaturityShare",
        *(str(spec["column"]) for spec in TDC_COMPONENT_SPECS),
    }
    opening_flows = {column: 0.0 for column in result_flow_columns}
    closing_flows = dict(opening_flows)
    closing_flows.update(
        {
            "TDC_Change": 1.0,
            "TDC_FiscalFlow": 1.0,
            "OutstandingControlledWAM": 10.0,
            "OutstandingControlledBillShare": 0.2,
            "OutstandingControlledShortMaturityShare": 0.3,
        }
    )
    results = pd.DataFrame(
        [
            {
                "Date": START,
                "TGA": 10.0,
                "Reserves": 50.0,
                "TDC_Level": 5.0,
                "TotalDebt_Agg": 100.0,
                **opening_flows,
            },
            {
                "Date": END,
                "TGA": 11.0,
                "Reserves": 50.0,
                "TDC_Level": 6.0,
                "TotalDebt_Agg": 100.0,
                **closing_flows,
            },
        ]
    )
    return root, manifest, results, rows


def _refresh_artifact(
    root: Path,
    manifest: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    name: str,
) -> None:
    filename, columns = _BOUNDED_ARTIFACT_SPECS[name]
    path = root / "outputs" / filename
    _write_gzip_csv(path, columns, rows[name])
    record = _record(path, row_count=len(rows[name]))
    bounded = manifest["bounded_evidence"]
    if name == "resources":
        bounded["resource_artifact"] = deepcopy(record)
    else:
        bounded["deterministic_artifacts"][name] = deepcopy(record)
        manifest["output_manifest"]["deterministic_evidence_artifacts"][
            name
        ] = deepcopy(record)
    manifest["output_manifest"][name] = deepcopy(record)
    for item in manifest["outputs"]:
        if item["relative_path"] == f"outputs/{filename}":
            item["sha256"] = record["sha256"]
            item["bytes"] = record["bytes"]
            break


def _add_parent_watchdog_acceptance(manifest: dict[str, Any]) -> None:
    gib = 1024**3
    if "evaluated_nominal_curve" in manifest:
        manifest.setdefault(
            "open04_campaign",
            {"contract_id": "open04_paired_tradeoff_v1", "role": "baseline"},
        )
    thresholds = manifest["bounded_evidence"]["memory_thresholds"]
    thresholds.update(
        {
            "acceptance_peak_rss_bytes": 6 * gib,
            "application_abort_rss_bytes": 8 * gib,
            "parent_graceful_stop_rss_bytes": 10 * gib,
            "parent_kill_rss_bytes": 12 * gib,
        }
    )
    manifest["execution_milestones"] = list(
        _BOUNDED_WATCHDOG_ACCEPTED_MILESTONES
    )
    manifest["execution_contract"].update(
        {
            "parent_watchdog_required": True,
            "scenario_process_mode": "parent_watchdog_worker",
            "writer_claim_scope": "output_parent",
            "writer_claim_scope_id": (
                f"{manifest['open04_campaign']['contract_id']}."
                f"{manifest['open04_campaign']['role']}"
                if "evaluated_nominal_curve" in manifest
                else "library-output-parent"
            ),
            "numerical_thread_environment": {
                name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
            },
        }
    )
    parent_peak = 5 * gib
    worker_peak = manifest["bounded_evidence"]["peak_rss_bytes"]
    effective_peak = max(parent_peak, worker_peak)
    manifest["parent_watchdog"] = {
        "status": "accepted",
        "sampler": "parent_process_rss_poll_v1",
        "child_pid": 1234,
        "child_returncode": 0,
        "action": "completed",
        "peak_rss_bytes": parent_peak,
        "worker_peak_rss_bytes": worker_peak,
        "effective_peak_rss_bytes": effective_peak,
        "acceptance_peak_rss_bytes": 6 * gib,
        "terminate_rss_bytes": 10 * gib,
        "kill_rss_bytes": 12 * gib,
        "poll_interval_seconds": 1.0,
    }
    manifest["validation"]["invariants"].append(
        {
            "id": "parent_watchdog_peak_rss",
            "status": "pass",
            "observed": effective_peak,
            "limit": 6 * gib,
        }
    )


def test_v2_schema_and_compact_verifier_accept_valid_bounded_contract(
    tmp_path: Path,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)

    _validate_run_manifest_schema(manifest)
    verified = _verify_bounded_outputs(root, manifest, results)

    assert verified == {
        "bounded_period_count": 1,
        "bounded_event_count": 1,
        "bounded_peak_rss_bytes": 100,
    }


def test_v2_parent_watchdog_terminal_manifest_is_schema_and_static_valid(
    tmp_path: Path,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)

    _validate_run_manifest_schema(manifest)
    assert _verify_bounded_outputs(root, manifest, results)[
        "bounded_period_count"
    ] == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("single_writer_claim", False, "single_writer_claim"),
        ("one_scenario_per_worker", False, "one_scenario_per_worker"),
        ("process_pool_enabled", True, "process_pool_enabled"),
        ("parent_watchdog_required", False, "parent_watchdog_required"),
        ("scenario_process_mode", "library_call", "scenario_process_mode"),
    ],
)
def test_watchdog_execution_contract_semantic_mutations_fail_closed(
    tmp_path: Path,
    field: str,
    value: Any,
    match: str,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)
    manifest["execution_contract"][field] = value

    with pytest.raises(VerificationError, match=match):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize("mutation", ["missing", "extra", "unpinned"])
def test_watchdog_execution_contract_thread_pin_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)
    environment = manifest["execution_contract"]["numerical_thread_environment"]
    first_name = THREAD_LIMIT_ENVIRONMENT_VARIABLES[0]
    if mutation == "missing":
        environment.pop(first_name)
        match = "thread variables are incomplete"
    elif mutation == "extra":
        environment["UNDECLARED_NUMERICAL_THREADS"] = "1"
        match = "thread variables are incomplete"
    else:
        environment[first_name] = "2"
        match = "pinned to one"

    with pytest.raises(VerificationError, match=match):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("child_returncode", "child_returncode"),
        ("action", "action is not completed"),
        ("worker_peak", "worker peak disagrees"),
        ("effective_peak", "effective peak is stale"),
        ("invariant_observed", "validation invariant is stale"),
    ],
)
def test_parent_watchdog_terminal_record_mutations_fail_closed(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)
    parent = manifest["parent_watchdog"]
    if mutation == "child_returncode":
        parent["child_returncode"] = 7
    elif mutation == "action":
        parent["action"] = "terminate_rss"
    elif mutation == "worker_peak":
        parent["worker_peak_rss_bytes"] += 1
    elif mutation == "effective_peak":
        parent["effective_peak_rss_bytes"] -= 1
    else:
        invariant = next(
            item
            for item in manifest["validation"]["invariants"]
            if item["id"] == "parent_watchdog_peak_rss"
        )
        invariant["observed"] -= 1

    with pytest.raises(VerificationError, match=match):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize(
    ("field", "value"),
    [("profile", "audit"), ("compression", "none")],
)
def test_open04_output_manifest_must_match_compiled_output_contract(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    _root, manifest, _results, _rows = _package(tmp_path)
    manifest["output_manifest"][field] = value
    compiled_contract = {
        "output_profile": "compact",
        "compression": "gzip",
    }

    with pytest.raises(
        VerificationError,
        match="disagrees with the compiled OPEN-04 output contract",
    ):
        _verify_open04_output_contract(manifest, compiled_contract)


def test_open04_runtime_curve_requires_parent_watchdog_acceptance(
    tmp_path: Path,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    manifest["evaluated_nominal_curve"] = {}

    with pytest.raises(
        VerificationError,
        match="requires parent watchdog acceptance",
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_open04_runtime_curve_requires_exact_memory_envelope(
    tmp_path: Path,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    manifest["evaluated_nominal_curve"] = {}
    _add_parent_watchdog_acceptance(manifest)
    manifest["bounded_evidence"]["memory_thresholds"][
        "minimum_available_bytes"
    ] = 19 * 1024**3

    with pytest.raises(
        VerificationError,
        match="exact 4/6/8/10/12 GiB memory envelope",
    ):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize("missing_side", ["parent", "milestone"])
def test_v2_schema_requires_parent_watchdog_exactly_with_terminal_milestone(
    tmp_path: Path,
    missing_side: str,
) -> None:
    _root, manifest, _results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)
    if missing_side == "parent":
        manifest.pop("parent_watchdog")
    else:
        manifest["execution_milestones"] = list(
            _BOUNDED_EXECUTION_MILESTONES
        )

    with pytest.raises(VerificationError, match="schema validation failed"):
        _validate_run_manifest_schema(manifest)


def test_v2_watchdog_terminal_milestone_order_is_exact(
    tmp_path: Path,
) -> None:
    root, manifest, results, _rows = _package(tmp_path)
    _add_parent_watchdog_acceptance(manifest)
    milestones = manifest["execution_milestones"]
    milestones[-2], milestones[-1] = milestones[-1], milestones[-2]

    with pytest.raises(
        VerificationError, match="milestones are incomplete or out of order"
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_manifest_schema_selection_rejects_unknown_version(
    tmp_path: Path,
) -> None:
    _root, manifest, _results, _rows = _package(tmp_path)
    manifest["schema_version"] = "tdcsim_cbo_scenario_run_manifest_v3"

    with pytest.raises(VerificationError, match="unsupported run manifest"):
        _validate_run_manifest_schema(manifest)


def test_v1_release_grade_behavior_is_unchanged() -> None:
    _verify_release_claims(
        {
            "schema_version": "tdcsim_cbo_scenario_run_manifest_v1",
            "status": "complete",
            "verification_grade": "local",
        }
    )
    with pytest.raises(VerificationError, match="release_verified"):
        _verify_release_claims(
            {
                "schema_version": "tdcsim_cbo_scenario_run_manifest_v1",
                "status": "complete",
                "verification_grade": "release_verified",
            }
        )


def test_bounded_commitment_sequence_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["commitments"][0]["event_seq_end"] = 2
    _refresh_artifact(root, manifest, rows, "commitments")

    with pytest.raises(VerificationError, match="sequence is discontinuous"):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_annual_financing_identity_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["annual"][0]["modeled_financing_cost_bil"] = 2.7
    _refresh_artifact(root, manifest, rows, "annual")

    with pytest.raises(
        VerificationError, match="modeled financing-cost identity"
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_coherent_tdc_thin_path_tamper_fails_level_bridge(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    results.loc[results["Date"].eq(END), ["TDC_Change", "TDC_FiscalFlow"]] = 2.0
    rows["tdc_components"][0]["amount_bil"] = 2.0
    rows["tdc_summary"][0].update(
        {
            "tdc_change_bil": 2.0,
            "tdc_fiscal_flow_bil": 2.0,
            "tdc_change_ex_overlap_bil": 2.0,
            "component_sum_bil": 2.0,
        }
    )
    rows["annual"][0].update(
        {
            "tdc_change_bil": 2.0,
            "tdc_change_ex_overlap_bil": 2.0,
            "cumulative_tdc_change_bil": 2.0,
            "cumulative_tdc_change_ex_overlap_bil": 2.0,
        }
    )
    for name in ("tdc_components", "tdc_summary", "annual"):
        _refresh_artifact(root, manifest, rows, name)

    with pytest.raises(
        VerificationError, match="daily-result TDC level bridge"
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_coherent_financing_thin_path_tamper_fails_payment_bridge(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    results.loc[
        results["Date"].eq(END),
        ["InterestOutlay_Period", "FinancingCost_Period"],
    ] = 5.0
    rows["annual"][0].update(
        {
            "interest_outlay_bil": 5.0,
            "modeled_financing_cost_bil": 5.0,
            "cumulative_interest_outlay_bil": 5.0,
            "cumulative_modeled_financing_cost_bil": 5.0,
        }
    )
    _refresh_artifact(root, manifest, rows, "annual")

    with pytest.raises(
        VerificationError, match="payment/results interest-outlay bridge"
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_coherent_outstanding_metric_tamper_fails_daily_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(globals(), "START", "2026-09-29")
    monkeypatch.setitem(globals(), "END", "2026-09-30")
    root, manifest, results, rows = _package(tmp_path)
    rows["annual"][0].update(
        {
            "snapshot_date": END,
            "outstanding_controlled_wam_years": 10.0,
            "outstanding_controlled_bill_share": 0.2,
            "outstanding_controlled_short_maturity_share": 0.3,
        }
    )
    _refresh_artifact(root, manifest, rows, "annual")
    _verify_bounded_outputs(root, manifest, results)

    changed = {
        "outstanding_controlled_wam_years": 11.0,
        "outstanding_controlled_bill_share": 0.25,
        "outstanding_controlled_short_maturity_share": 0.35,
    }
    rows["scenario_metrics"][0].update(changed)
    rows["annual"][0].update(changed)
    for name in ("scenario_metrics", "annual"):
        _refresh_artifact(root, manifest, rows, name)

    with pytest.raises(
        VerificationError,
        match="scenario-metric/result bridge outstanding_controlled_wam_years",
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_coherent_issuance_thin_path_tamper_fails_stock_bridge(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["issuance"].append(
        _with_columns(
            ISSUANCE_COLUMNS,
            period_start=START,
            period_end=END,
            holder_sector="Private",
            holder_subsector="domestic_nonbank_deposit_funded",
            instrument_type="Fixed",
            maturity_bucket="bills",
            weighted_original_term_years=0.5,
            face_issued_bil=10.0,
            cash_proceeds_bil=9.0,
            discount_or_premium_bil=1.0,
            flow_count=1,
        )
    )
    results.loc[
        results["Date"].eq(END),
        [
            "NewDebtIssued",
            "AuctionProceeds",
            "IssueDiscountCost_Period",
            "FinancingCost_Period",
        ],
    ] = [10.0, 9.0, 1.0, 1.0]
    rows["tdc_summary"][0]["gross_issuance_cash_proceeds_bil"] = 9.0
    rows["debt_bridge"][0]["face_issued_bil"] = 10.0
    rows["annual"][0].update(
        {
            "issue_discount_cost_bil": 1.0,
            "modeled_financing_cost_bil": 1.0,
            "cumulative_issue_discount_cost_bil": 1.0,
            "cumulative_modeled_financing_cost_bil": 1.0,
            "new_issuance_face_bil": 10.0,
            "new_issuance_original_term_face_years_bil": 5.0,
            "new_issuance_bill_face_bil": 10.0,
            "new_issuance_short_face_bil": 10.0,
            "new_issuance_wam_years": 0.5,
            "new_issuance_bill_share": 1.0,
            "new_issuance_short_maturity_share": 1.0,
        }
    )
    for name in ("issuance", "tdc_summary", "debt_bridge", "annual"):
        _refresh_artifact(root, manifest, rows, name)

    with pytest.raises(
        VerificationError,
        match="ledger/issuance/principal face-stock bridge",
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_coherent_principal_thin_path_tamper_fails_stock_bridge(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["principal"].append(
        _with_columns(
            PRINCIPAL_COLUMNS,
            period_start=START,
            period_end=END,
            holder_sector="Banks",
            holder_subsector="",
            instrument_type="Fixed",
            maturity_bucket="bills",
            redemption_type="explicit_retirement_at_par",
            face_redeemed_bil=10.0,
            principal_redeemed_bil=10.0,
            cash_paid_bil=10.0,
            adjusted_principal_stock_removed_bil=0.0,
            tdc_principal_recipient_sector="Banks",
            tdc_principal_recipient_subsector="",
            tdc_principal_cash_paid_to_du_bil=0.0,
            tdc_principal_redeemed_to_du_bil=0.0,
            tdc_principal_cash_paid_to_du_domestic_nonbank_bil=0.0,
            tdc_principal_redeemed_to_du_domestic_nonbank_bil=0.0,
            tdc_principal_cash_paid_to_du_mmf_bil=0.0,
            tdc_principal_redeemed_to_du_mmf_bil=0.0,
            tdc_principal_cash_paid_to_du_mmf_plumbing_bil=0.0,
            tdc_principal_redeemed_to_du_mmf_plumbing_bil=0.0,
            tdc_principal_recipient_basis=(
                "current_cb_beneficial_holder_otherwise_recorded_tdc_principal_route"
            ),
            flow_count=1,
        )
    )
    results.loc[
        results["Date"].eq(END),
        [
            "PrincipalPaid_Bonds",
            "CBOBuybackFaceRetired",
            "CBOBuybackCashPaid",
        ],
    ] = 10.0
    rows["debt_bridge"][0]["face_retired_bil"] = 10.0
    for name in ("principal", "debt_bridge"):
        _refresh_artifact(root, manifest, rows, name)

    with pytest.raises(
        VerificationError,
        match="ledger/issuance/principal face-stock bridge",
    ):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("delete", "stock"),
        ("duplicate", "duplicate key"),
    ],
)
def test_bounded_stock_closure_row_set_tamper_fails_after_hash_refresh(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    if mutation == "delete":
        rows["stock"].pop()
    else:
        rows["stock"].append(deepcopy(rows["stock"][0]))
    _refresh_artifact(root, manifest, rows, "stock")

    with pytest.raises(VerificationError, match=match):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_stored_closure_error_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["stock"][0]["face_stock_closure_error_bil"] = 1.0
    _refresh_artifact(root, manifest, rows, "stock")

    with pytest.raises(VerificationError, match="stale error"):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_route_key_or_identity_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["route"][0]["opening_route_stock_bil"] = 99.0
    _refresh_artifact(root, manifest, rows, "route")

    with pytest.raises(VerificationError, match="route closure"):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_ledger_count_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["ledger"][0]["event_count"] = 2
    _refresh_artifact(root, manifest, rows, "ledger")

    with pytest.raises(VerificationError, match="event count"):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_resource_sequence_tamper_fails_after_hash_refresh(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["resources"][0]["sample_kind"] = "period_close"
    _refresh_artifact(root, manifest, rows, "resources")

    with pytest.raises(VerificationError, match="first bounded resource"):
        _verify_bounded_outputs(root, manifest, results)


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        (
            "CBOControlledDebtTargetError",
            1.000001e-6,
            "controlled-debt target error",
        ),
        (
            "CBOFedHoldingsTargetError",
            1.000001e-6,
            "Fed holdings target error",
        ),
    ],
)
def test_bounded_result_target_errors_enforce_exact_one_millionth_limit(
    column: str, value: float, match: str
) -> None:
    passing = pd.DataFrame(
        {
            "CBOControlledDebtTargetError": [1e-6, -1e-6],
            "CBOFedHoldingsTargetError": [1e-6, -1e-6],
        }
    )
    assert _verify_bounded_result_limits(passing) == {
        "max_abs_controlled_debt_target_error": 1e-6,
        "max_abs_fed_holdings_target_error": 1e-6,
    }
    failing = passing.copy()
    failing.loc[0, column] = value

    with pytest.raises(VerificationError, match=match):
        _verify_bounded_result_limits(failing)


def test_bounded_debt_bridge_rejects_target_error_above_one_millionth(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["debt_bridge"][0]["face_issued_bil"] = 1.1e-6
    rows["debt_bridge"][0]["controlled_debt_post_issuance_bil"] = 100.0000011
    rows["debt_bridge"][0]["target_error_bil"] = 1.1e-6
    _refresh_artifact(root, manifest, rows, "debt_bridge")

    with pytest.raises(
        VerificationError, match="controlled-debt target error exceeds"
    ):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_debt_bridge_rejects_funding_mode_different_from_manifest(
    tmp_path: Path,
) -> None:
    root, manifest, results, rows = _package(tmp_path)
    rows["debt_bridge"][0]["funding_mode"] = (
        "cbo_debt_reference_plus_tga_floor_financing_v1"
    )
    _refresh_artifact(root, manifest, rows, "debt_bridge")

    with pytest.raises(VerificationError, match="funding mode differs"):
        _verify_bounded_outputs(root, manifest, results)


def test_bounded_reference_financing_accepts_omitted_applicability_marker() -> None:
    results = pd.DataFrame(
        {
            "CBOControlledDebtReference": [100.0],
            "ScenarioControlledDebt": [101.0],
            "DebtDriftFromReference": [1.0],
            "CashFinancingFaceIssued": [1.0],
            "CashFinancingProceeds": [0.99],
            "IssuePriceCashGap": [0.01],
            "CBOControlledDebtTarget": [100.0],
            "CBOControlledDebtPostIssuance": [101.0],
            "CBOControlledDebtTargetError": [0.0],
            "NewDebtIssued": [1.0],
            "AuctionProceeds": [0.99],
        }
    )

    _verify_cbo_reference_financing_results(
        results,
        require_target_applicable=False,
    )
    with pytest.raises(VerificationError, match="TargetApplicable"):
        _verify_cbo_reference_financing_results(results)


def test_bounded_replay_uses_sink_and_compares_compact_commitments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, _results, _rows = _package(tmp_path)
    manifest["open04_campaign"] = {
        "contract_id": "open04_paired_tradeoff_v1",
        "role": "baseline",
        "funding_closure_mode": (
            "cbo_debt_reference_plus_tga_floor_financing_v1"
        ),
    }
    bounded = manifest["bounded_evidence"]
    replay_summary = {
        **{
            key: deepcopy(bounded[key])
            for key in (
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
                "deterministic_artifacts",
            )
        },
        "peak_rss_bytes": 100,
    }
    calls: dict[str, Any] = {}

    class FakeSink:
        def __init__(self, output_dir: Path, *, limits: Any) -> None:
            calls["sink"] = self
            calls["sink_args"] = (output_dir, limits)

        def abort(self, exc: BaseException) -> None:
            calls["abort"] = exc

    def fake_run(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls["run_kwargs"] = kwargs
        results = pd.DataFrame({"Date": [START, END]})
        results.attrs["bounded_handoff_summary"] = replay_summary
        return results, pd.DataFrame()

    monkeypatch.setattr(
        "tdcsim_cbo.verifier.BoundedScenarioEvidenceSink", FakeSink
    )

    def fake_build_runtime_params(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["runtime_args"] = args
        calls["runtime_kwargs"] = kwargs
        return {}

    monkeypatch.setattr(
        "tdcsim_cbo.verifier.runner_module.build_runtime_params",
        fake_build_runtime_params,
    )
    monkeypatch.setattr(
        "tdcsim_cbo.verifier.runner_module._engine_runtime_params",
        lambda params: params,
    )
    monkeypatch.setattr(
        "tdcsim_cbo.verifier.runner_module._compiled_scenario_id",
        lambda _inputs: "test",
    )
    monkeypatch.setattr("tdcsim_cbo.verifier.run_simulation", fake_run)
    monkeypatch.setattr(
        "tdcsim_cbo.verifier.write_bounded_scenario_outputs",
        lambda *args, **kwargs: {},
    )

    _verify_bounded_engine_replay(root, manifest, tmp_path / "inputs")

    assert calls["run_kwargs"]["require_bounded_handoff"] is True
    assert calls["run_kwargs"]["handoff_sink"] is calls["sink"]
    assert calls["runtime_kwargs"] == {
        "actuals_available_as_of": "",
        "simulation_start_date": START,
        "simulation_end_date": END,
        "scenario_id": "test",
        "funding_closure_mode": (
            "cbo_debt_reference_plus_tga_floor_financing_v1"
        ),
    }

    replay_summary["event_root_sha256"] = "9" * 64
    with pytest.raises(VerificationError, match="event_root_sha256"):
        _verify_bounded_engine_replay(root, manifest, tmp_path / "inputs")
