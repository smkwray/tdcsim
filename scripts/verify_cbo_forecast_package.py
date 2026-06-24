#!/usr/bin/env python3
"""Verify and optionally package a local CBO forecast evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecast_input_builder import (  # noqa: E402
    BUDGET_WORKBOOK_SHA256,
    ECONOMIC_WORKBOOK_SHA256,
    build_cbo_fiscal_baseline_rows,
    build_cbo_macro_forecast_path_rows,
    parse_cbo_budget_source_contract,
    parse_cbo_economic_quarterly_source_contract,
)
from cbo_yield_curve_surface import build_yield_curve_surface_rows  # noqa: E402
from tips_indexation import (  # noqa: E402
    build_monthly_tips_cpi_path_rows,
    build_tips_real_yield_path_rows,
)


REQUIRED_FILES = {
    "requirements.lock.txt",
    "manifest.json",
    "forecast_inputs/source_contract_smoke.json",
    "forecast_inputs/source_fixtures.csv",
    "forecast_inputs/tdcsim_opening_portfolio.csv",
    "forecast_inputs/tdcsim_opening_rollforward_diagnostics.csv",
    "forecast_inputs/tdcsim_opening_tips_indexation_diagnostics.csv",
    "forecast_inputs/tdcsim_opening_frn_indexation_diagnostics.csv",
    "forecast_inputs/tdcsim_cbo_fiscal_baseline.csv",
    "forecast_inputs/tdcsim_current_fy_splice.csv",
    "forecast_inputs/tdcsim_debt_stock_path.csv",
    "forecast_inputs/tdcsim_primary_deficit_path.csv",
    "forecast_inputs/tdcsim_operating_cash_path.csv",
    "forecast_inputs/tdcsim_operating_cash_path_constant_nominal.csv",
    "forecast_inputs/tdcsim_cash_reconciliation_residual.csv",
    "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_real_tga.csv",
    "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_nominal_tga.csv",
    "forecast_inputs/tdcsim_macro_forecast_path.csv",
    "forecast_inputs/tdcsim_yield_curve_surface.csv",
    "forecast_inputs/tdcsim_frn_rate_path.csv",
    "forecast_inputs/tdcsim_tips_cpi_path.csv",
    "forecast_inputs/tdcsim_tips_real_yield_path.csv",
    "forecast_inputs/tdcsim_fed_holdings_path.csv",
    "forecast_inputs/tdcsim_fiscal_incidence_policy.csv",
    "forecast_inputs/tdcsim_fiscal_incidence_sensitivity_results.csv",
    "forecast_inputs/tdcsim_holder_profile_assumptions.csv",
    "forecast_inputs/tdcsim_net_interest_bridge.csv",
    "exports_zero_residual/summary.json",
    "exports_zero_residual/results_compact.csv",
    "exports_zero_residual/final_portfolio_active.csv",
    "exports_zero_residual/maturity_ledger.csv",
    "exports_constant_real_tga/summary.json",
    "exports_constant_real_tga/results_compact.csv",
    "exports_constant_real_tga/final_portfolio_active.csv",
    "exports_constant_real_tga/maturity_ledger.csv",
    "exports_constant_nominal_tga/summary.json",
    "exports_constant_nominal_tga/results_compact.csv",
    "exports_constant_nominal_tga/final_portfolio_active.csv",
    "exports_constant_nominal_tga/maturity_ledger.csv",
    "cash_sensitivity_comparison.csv",
    "tdcsim_partial_interest_financing_diagnostic.csv",
    "sources/source_manifest.json",
    "sources/cbo/51118-2026-02-Budget-Projections.xlsx",
    "sources/cbo/51135-2026-02-Economic-Projections.xlsx",
    "sources/opening/tdcsim_initial_portfolio_cohorts.csv",
    "sources/opening/tdcsim_ratewall_input_manifest.json",
    "sources/fiscaldata/tips_cpi_data_detail.csv",
    "sources/fiscaldata/frn_daily_indexes.csv",
    "sources/fiscaldata/mts_table_1_deficit_2026-05-31.csv",
    "sources/fiscaldata/mts_table_9_net_interest_2026-05-31.csv",
    "sources/fiscaldata/dts_tga_closing_balance_2026-06-16.csv",
    "sources/fiscaldata/mspd_table_1_debt_bridge_2026-05-31.csv",
    "sources/fiscaldata/mspd_table_3_market_2026-05-31.csv",
    "sources/treasury/debt_to_penny_public_debt_2026-05-29.csv",
    "sources/treasury/base_curve_2026-06-16.json",
}

EXPECTED_SOURCE_PACKAGE_FILES = {
    "cbo/51118-2026-02-Budget-Projections.xlsx",
    "cbo/51135-2026-02-Economic-Projections.xlsx",
    "opening/tdcsim_initial_portfolio_cohorts.csv",
    "opening/tdcsim_ratewall_input_manifest.json",
    "fiscaldata/tips_cpi_data_detail.csv",
    "fiscaldata/frn_daily_indexes.csv",
    "fiscaldata/mts_table_1_deficit_2026-05-31.csv",
    "fiscaldata/mts_table_9_net_interest_2026-05-31.csv",
    "fiscaldata/dts_tga_closing_balance_2026-06-16.csv",
    "fiscaldata/mspd_table_1_debt_bridge_2026-05-31.csv",
    "fiscaldata/mspd_table_3_market_2026-05-31.csv",
    "treasury/debt_to_penny_public_debt_2026-05-29.csv",
    "treasury/base_curve_2026-06-16.json",
    "source_manifest.json",
}

APPROVED_OUTPUT_PATHS = {
    "zero_residual_summary": "exports_zero_residual/summary.json",
    "constant_real_tga_summary": "exports_constant_real_tga/summary.json",
    "constant_real_tga_cash_residual_file": "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_real_tga.csv",
    "constant_nominal_tga_summary": "exports_constant_nominal_tga/summary.json",
    "constant_nominal_tga_cash_residual_file": "forecast_inputs/tdcsim_cash_reconciliation_residual_constant_nominal_tga.csv",
    "constant_nominal_tga_operating_cash_file": "forecast_inputs/tdcsim_operating_cash_path_constant_nominal.csv",
    "cash_sensitivity_comparison": "cash_sensitivity_comparison.csv",
    "partial_interest_financing_diagnostic": "tdcsim_partial_interest_financing_diagnostic.csv",
}

APPROVED_ARTIFACT_PATHS = {
    "output_dir": ".",
    "forecast_inputs_dir": "forecast_inputs",
    "manifest": "manifest.json",
}

OPTIONAL_KNOWN_FILES = {
    "forecast_inputs/tdcsim_opening_portfolio_metadata.json",
}

MANIFEST_EQUIVALENT_FILES = {
    "manifest.json",
    "forecast_inputs/source_contract_smoke.json",
}

FROZEN_SOURCE_FILE_METADATA: dict[str, dict[str, Any]] = {
    "cbo/51118-2026-02-Budget-Projections.xlsx": {
        "sha256": BUDGET_WORKBOOK_SHA256,
    },
    "cbo/51135-2026-02-Economic-Projections.xlsx": {
        "sha256": ECONOMIC_WORKBOOK_SHA256,
    },
}

FORBIDDEN_PACKAGE_FILES = {
    "tdcsim_modeled_net_interest_diagnostic.csv",
}

FORBIDDEN_MANIFEST_KEYS = {
    "modeled_net_interest_diagnostic",
    "modeled_net_interest_bil",
    "modeled_net_interest_proxy_bil",
    "bill_discount_amortization_proxy_bil",
}

FORBIDDEN_NET_INTEREST_COLUMNS = {
    "modeled_net_interest_bil",
    "modeled_net_interest_proxy_bil",
    "bill_discount_amortization_proxy_bil",
}

TRUST_ANCHOR_SCHEMA_VERSION = "tdcsim_cbo_package_trust_anchor_v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "tdcsim_cbo_package_claim_boundary_v1"
VERIFIER_VERSION = "tdcsim_cbo_package_verifier_semantic_v2"
PENDING_RELEASE_STATUS = "pending_until_release_commit"
PENDING_ZIP_STATUS = "pending_until_release_commit"

EXPECTED_SOURCE_ROW_SELECTORS = {
    "mts_table_1": "classification_desc=Year-to-Date;line_code_nbr=280;column=current_month_dfct_sur_amt",
    "mts_table_9": "classification_desc=Net Interest;column=current_fytd_rcpt_outly_amt",
    "debt_to_the_penny": "record_date=2026-05-29;column=debt_held_public_amt",
    "dts": "account_type=Treasury General Account (TGA) Closing Balance",
}
EXPECTED_MSPD_SELECTORS = {
    "security_type_desc=Total Nonmarketable",
    "security_type_desc=Total Public Debt Outstanding",
}
EXPECTED_SOURCE_EXTRACT_ROWS = {
    "mts_table_1_deficit_row": {
        "record_date": "2026-05-31",
        "classification_desc": "Year-to-Date",
        "line_code_nbr": "280",
        "current_month_dfct_sur_amt": "1246203266386.93",
    },
    "mts_table_9_net_interest_row": {
        "record_date": "2026-05-31",
        "classification_desc": "Net Interest",
        "current_fytd_rcpt_outly_amt": "722706511243.20",
    },
    "dts_tga_closing_balance_row": {
        "record_date": "2026-06-16",
        "account_type": "Treasury General Account (TGA) Closing Balance",
        "open_today_bal": "981113",
    },
    "debt_to_penny_row": {
        "record_date": "2026-05-29",
        "debt_held_public_amt": "31515369798622.98",
    },
}
EXPECTED_FROZEN_MSPD_ROWS = {
    (
        "2026-05-31",
        "Total Nonmarketable",
        "620821.77409502",
    ),
    (
        "2026-05-31",
        "Total Public Debt Outstanding",
        "31515369.91844520",
    ),
}
EXPECTED_TREASURY_BASE_CURVE_ROWS = {
    (round(1.0 / 12.0, 12), 3.67),
    (0.25, 3.79),
    (0.5, 3.81),
    (1.0, 3.84),
    (2.0, 4.05),
    (3.0, 4.08),
    (5.0, 4.16),
    (7.0, 4.28),
    (10.0, 4.43),
    (20.0, 4.92),
    (30.0, 4.93),
}
CLAIM_BOUNDARY_EXCLUSIONS = {
    "does_not_claim_exact_historical_replay",
    "does_not_claim_full_soma_transaction_replay",
    "does_not_claim_budgetary_net_interest_replication",
    "does_not_claim_generated_evidence_tracked_in_git",
    "does_not_claim_receipts_outlays_decomposition",
    "does_not_use_cbo_net_interest_as_cash_or_issuance_plug",
    "does_not_claim_cbo_issuance_mix",
    "does_not_model_remittances_or_monetary_settlement_effects",
}
APPROVED_OPENING_PORTFOLIO_CLAIM_BOUNDARY = (
    "prebuilt_mspd_table3_market_derived_cohort_book_with_z1_holder_provenance_metadata; "
    "z1_inputs_not_packaged_for_portable_holder_constraint_reproduction; "
    "securities_maturing_between_record_date_and_simulation_start_are_explicit_prestart_rollforward_bills"
)
APPROVED_TOP_LEVEL_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "scenario_id",
    "forecast_name",
    "forecast_publication_date",
    "date_range",
    "source_files",
    "source_package",
    "source_row_selectors",
    "source_values",
    "rate_unit_declaration",
    "opening_portfolio_construction",
    "issuance_mix",
    "negative_issuance_policy",
    "holder_preferences",
    "fed_holdings_path",
    "tips_forward_paths",
    "frn_forward_rate_path",
    "operating_cash_construction",
    "scope_claim_id",
    "claim_boundary",
    "trust_anchor",
    "artifacts",
    "outputs",
    "run_summaries",
    "artifact_hashes",
}
APPROVED_CLAIM_BOUNDARY_KEYS = {
    "schema_version",
    "package_role",
    "net_interest_role",
    "issuance_mix_role",
    "cash_role",
    "excluded_claims",
}
APPROVED_OPENING_PORTFOLIO_KEYS = {
    "claim_boundary",
    "frn_daily_indexes_source",
    "frn_daily_indexes_source_sha256",
    "frn_indexation_diagnostics_file",
    "frn_initialization",
    "frn_opening_date",
    "frn_rows_enriched",
    "frn_source_accrual_end_dates",
    "frn_source_available_as_of",
    "frn_source_boundary_note",
    "frn_source_record_dates",
    "frn_unique_cusips_enriched",
    "holder_totals_bil",
    "mode",
    "opening_cb_holdings_bil",
    "opening_frn_accrued_interest_bil",
    "opening_frn_benchmark_rate_decimal_max",
    "opening_frn_benchmark_rate_decimal_min",
    "opening_frn_face_value_bil",
    "opening_frn_fixed_spread_decimal_max",
    "opening_frn_fixed_spread_decimal_min",
    "opening_rollforward_aligned_debt_base_bil",
    "opening_rollforward_debt_base_delta_bil",
    "opening_rollforward_diagnostics_file",
    "opening_rollforward_policy",
    "opening_rollforward_source_debt_base_bil",
    "opening_rollforward_status",
    "opening_source_common_date_status",
    "opening_source_latest_available_mspd_table3_record_date",
    "opening_tips_adjusted_principal_bil",
    "opening_tips_indexation_accretion_embedded_bil",
    "opening_tips_reconstructed_original_principal_bil",
    "opening_tips_reference_cpi",
    "prestart_rollforward_face_bil",
    "prestart_rollforward_rows",
    "record_date",
    "row_count",
    "security_type_totals_bil",
    "simulation_start_date",
    "source_file",
    "source_file_sha256",
    "source_manifest",
    "source_manifest_original_sha256",
    "source_manifest_portable_sha256",
    "source_manifest_sha256",
    "source_rows_loaded",
    "tips_cpi_detail_index_date",
    "tips_cpi_detail_source",
    "tips_cpi_detail_source_sha256",
    "tips_indexation_diagnostics_file",
    "tips_initialization",
    "tips_rows_enriched",
    "tips_source_ref_cpi_values",
    "tips_unique_cusips_enriched",
}
APPROVED_TIPS_FORWARD_PATH_KEYS = {
    "auction_pricing",
    "claim_boundary",
    "cpi_file",
    "cpi_mode",
    "max_expected_inflation_horizon_date",
    "real_yield_file",
    "real_yield_mode",
    "reference_cpi_lag_months",
    "runtime_effect",
    "source_status",
    "terminal_annualized_cpi_growth_decimal",
    "terminal_cbo_anchor_month",
    "terminal_cpi_rule",
    "terminal_extrapolated_month_rows",
    "terminal_extrapolation_start_month",
    "terminal_growth_source_period_end",
    "terminal_growth_source_period_start",
}
APPROVED_FRN_FORWARD_RATE_PATH_KEYS = {
    "benchmark_tenor_years",
    "claim_boundary",
    "day_count_basis",
    "file",
    "lockout_business_days",
    "mode",
    "runtime_effect",
    "source_status",
}
APPROVED_FED_HOLDINGS_PATH_KEYS = {
    "claim_boundary",
    "mode",
    "runtime_columns",
    "runtime_effect",
}
APPROVED_TRUST_ANCHOR_KEYS = {
    "code_revision",
    "dependency_lock_status",
    "dirty_state",
    "generated_at_utc",
    "package_zip_sha256",
    "package_zip_sha256_status",
    "python_version",
    "release_commit_sha",
    "release_commit_status",
    "requirements_dev_sha256",
    "requirements_lock_sha256",
    "requirements_sha256",
    "schema_version",
    "verifier_runtime",
    "verifier_sha256",
    "verifier_version",
}
APPROVED_SOURCE_PACKAGE_KEYS = {
    "source_file_count",
    "source_files",
    "source_manifest",
    "source_manifest_sha256",
    "sources_dir",
}
APPROVED_SOURCE_MANIFEST_KEYS = {
    "files",
    "inventory_status",
    "schema_version",
    "status",
}
APPROVED_SOURCE_FILE_ENTRY_KEYS = {
    "bytes",
    "origin_repo_path",
    "origin_source_label",
    "original_sha256",
    "portable_reserialized",
    "sha256",
    "source_path",
}
FORBIDDEN_CLAIM_PHRASES = (
    "fully portable source-backed z.1",
    "exact holder reproduction",
    "portable z.1 holder constraints",
    "portable-z.1",
    "cbo-prescribed auction issuance mix",
    "complete cbo budgetary net-interest replication",
)

RUN_DIRS = {
    "zero_residual": "exports_zero_residual",
    "constant_real_tga": "exports_constant_real_tga",
    "constant_nominal_tga": "exports_constant_nominal_tga",
}

RESULT_INVARIANT_COLUMNS = (
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBORequiredFaceIssuance",
    "NewDebtIssued",
    "AuctionProceeds",
    "CBOFedHoldingsTarget",
    "DebtHeld_Banks",
    "DebtHeld_CentralBank",
    "DebtHeld_Foreign",
    "DebtHeld_DomesticNonBanks",
)

LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![\w.:-])/(?:Users|home|Volumes|mnt|var/folders|tmp|private/tmp)/[^\s\"',;)]+"),
    re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:\\|\\\\)[^\s\"']+"),
    re.compile(r"file://[^\s\"']+"),
)

TEXT_SUFFIXES = {".csv", ".json", ".txt", ".md", ".yml", ".yaml"}
NUMERIC_TOLERANCE = 1e-6
STRICT_TOLERANCE = 1e-9


def verify_package(path: str | Path) -> dict[str, Any]:
    """Verify a CBO forecast evidence directory or zip archive."""

    package_path = Path(path)
    if package_path.suffix.lower() == ".zip":
        package_zip_sha256 = _sha256_file(package_path)
        with tempfile.TemporaryDirectory(prefix="tdcsim-cbo-verify-") as tmp:
            with zipfile.ZipFile(package_path) as archive:
                _extract_zip_safely(archive, Path(tmp))
            root = _extracted_root(Path(tmp))
            return _verify_directory(root, package_zip_sha256=package_zip_sha256)
    return _verify_directory(package_path, package_zip_sha256=None)


def create_package_zip(source_dir: str | Path, zip_path: str | Path) -> Path:
    """Create a portable zip from an evidence directory."""

    source = Path(source_dir)
    target = Path(zip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(source.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.resolve() == target.resolve():
                continue
            archive.write(file_path, file_path.relative_to(source))
    return target


def _verify_directory(root: Path, *, package_zip_sha256: str | None) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(f"CBO package directory is missing: {root}")
    manifest_path = root / "manifest.json"
    source_contract_path = root / "forecast_inputs" / "source_contract_smoke.json"
    manifest = _read_json(manifest_path)
    source_contract = _read_json(source_contract_path)
    if manifest != source_contract:
        raise ValueError("manifest.json and forecast_inputs/source_contract_smoke.json differ")

    _reject_local_paths(root)
    _verify_required_files(root)
    _verify_manifest_schema(manifest)
    _reject_overclaiming_manifest_fields(manifest)
    _verify_artifact_hashes(root, manifest)
    _verify_manifest_package_relative_paths(root, manifest)
    _verify_source_package(root, manifest)
    _verify_cbo_source_fixtures_against_workbooks(root)
    _verify_package_file_allowlist(root, manifest)
    _verify_claim_boundary(manifest)
    _verify_manifest_contract(manifest)
    _verify_semantic_contract(root, manifest)
    _verify_trust_anchor(root, manifest, package_zip_sha256=package_zip_sha256)
    return {
        "package_dir": str(root),
        "scenario_id": manifest.get("scenario_id"),
        "artifact_hashes": len(manifest.get("artifact_hashes", {})),
        "required_files": len(REQUIRED_FILES),
        "status": "pass",
    }


def _verify_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("scenario_id") != "cbo_full_horizon_local_smoke_3m":
        raise ValueError("unexpected scenario_id")
    dates = manifest.get("date_range", {})
    if dates.get("simulation_start_date") != "2026-06-21":
        raise ValueError("unexpected simulation_start_date")
    if dates.get("simulation_end_date") != "2036-09-30":
        raise ValueError("unexpected simulation_end_date")

    opening = manifest.get("opening_portfolio_construction", {})
    expected_opening = {
        "row_count": 2290,
        "prestart_rollforward_rows": 35,
        "tips_rows_enriched": 265,
        "tips_unique_cusips_enriched": 53,
        "frn_rows_enriched": 40,
        "frn_unique_cusips_enriched": 8,
    }
    for key, expected in expected_opening.items():
        if int(opening.get(key, -1)) != expected:
            raise ValueError(f"opening_portfolio_construction.{key} expected {expected}, found {opening.get(key)}")
    if abs(float(opening.get("opening_rollforward_debt_base_delta_bil", 1.0))) > 1e-9:
        raise ValueError("opening rollforward debt base delta is not zero")
    if "no_same_day_cusip_level_mspd_table3_book_available" not in str(
        opening.get("opening_source_common_date_status", "")
    ):
        raise ValueError("opening source common-date status is missing expected boundary wording")
    if "contract-compliant" not in str(opening.get("frn_source_boundary_note", "")):
        raise ValueError("FRN source boundary note is missing contract-compliant wording")
    if opening.get("claim_boundary") != APPROVED_OPENING_PORTFOLIO_CLAIM_BOUNDARY:
        raise ValueError("opening_portfolio_construction.claim_boundary changed")

    run_summaries = manifest.get("run_summaries", {})
    for run_id in RUN_DIRS:
        summary = run_summaries.get(run_id)
        if not isinstance(summary, dict):
            raise ValueError(f"missing run summary: {run_id}")
        if int(summary.get("rows", -1)) != 3755:
            raise ValueError(f"{run_id} rows must be 3755")
        if abs(float(summary.get("max_abs_target_error_bil", 1.0))) > 1e-6:
            raise ValueError(f"{run_id} target error exceeds tolerance")
        if abs(float(summary.get("final_cbo_fed_holdings_error_bil", 1.0))) > 1e-6:
            raise ValueError(f"{run_id} final Fed holdings error exceeds tolerance")


def _verify_semantic_contract(root: Path, manifest: Mapping[str, Any]) -> None:
    run_metrics = {}
    manifest_summaries = manifest.get("run_summaries", {})
    if not isinstance(manifest_summaries, Mapping):
        raise ValueError("manifest run_summaries must be populated")

    debt_targets = _verify_debt_stock_path(root, manifest)
    for run_id, run_dir in RUN_DIRS.items():
        results = _read_csv_dicts(root / run_dir / "results_compact.csv")
        summary = _read_json(root / run_dir / "summary.json")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{run_id} summary must be a JSON object")
        metrics = _compute_run_metrics(run_id, results, debt_targets)
        _verify_run_summary(run_id, summary, metrics)
        manifest_summary = manifest_summaries.get(run_id)
        if not isinstance(manifest_summary, Mapping):
            raise ValueError(f"manifest missing run summary: {run_id}")
        _verify_run_summary(f"manifest {run_id}", manifest_summary, metrics)
        run_metrics[run_id] = metrics

    _verify_cash_invariance(run_metrics)
    _verify_cash_sensitivity_comparison(root, run_metrics)
    _verify_partial_interest_financing_diagnostic(root, run_metrics["zero_residual"])
    _verify_opening_portfolio(root, manifest)
    _verify_tips_forward_paths(root, manifest)
    _verify_frn_rate_path(root, manifest)
    _verify_yield_surface_against_sources(root)
    _verify_rollforward(root, manifest)
    _verify_source_selectors_and_frozen_rows(root, manifest)


def _verify_manifest_schema(manifest: Mapping[str, Any]) -> None:
    unknown = sorted(set(str(key) for key in manifest) - APPROVED_TOP_LEVEL_MANIFEST_KEYS)
    if unknown:
        raise ValueError(f"manifest contains unknown top-level fields: {unknown}")
    boundary = manifest.get("claim_boundary")
    if isinstance(boundary, Mapping):
        unknown_boundary = sorted(set(str(key) for key in boundary) - APPROVED_CLAIM_BOUNDARY_KEYS)
        if unknown_boundary:
            raise ValueError(f"claim_boundary contains unknown fields: {unknown_boundary}")
    _verify_mapping_keys(
        manifest,
        "opening_portfolio_construction",
        APPROVED_OPENING_PORTFOLIO_KEYS,
    )
    _verify_mapping_keys(manifest, "tips_forward_paths", APPROVED_TIPS_FORWARD_PATH_KEYS)
    _verify_mapping_keys(manifest, "frn_forward_rate_path", APPROVED_FRN_FORWARD_RATE_PATH_KEYS)
    _verify_mapping_keys(manifest, "fed_holdings_path", APPROVED_FED_HOLDINGS_PATH_KEYS)
    _verify_mapping_keys(manifest, "trust_anchor", APPROVED_TRUST_ANCHOR_KEYS)


def _verify_mapping_keys(manifest: Mapping[str, Any], key: str, approved_keys: set[str]) -> None:
    value = manifest.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest {key} must be populated")
    unknown = sorted(set(str(child_key) for child_key in value) - approved_keys)
    if unknown:
        raise ValueError(f"{key} contains unknown fields: {unknown}")


def _compute_run_metrics(
    run_id: str,
    rows: list[dict[str, str]],
    debt_targets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"{run_id} results_compact.csv has no data rows")
    required = {
        "Date",
        "TotalDebt_Agg",
        "CBOControlledDebtTarget",
        "CBOControlledDebtTargetApplicable",
        "CBOControlledDebtPostIssuance",
        "CBOControlledDebtTargetError",
        "CBORequiredFaceIssuance",
        "NewDebtIssued",
        "AuctionProceeds",
        "IssueDiscountCost_Period",
        "PrimaryDeficit",
        "TGA",
        "CBOOperatingCashTarget",
        "CBOCashResidual",
        "CBOCashReconciliationResidual",
        "CBOFedHoldingsTarget",
        "CBOFedHoldingsTargetApplicable",
        "CBOFedHoldingsTargetError",
        "CBOFedAuctionShare",
        "CBOFedSecondaryPurchaseFace",
        "CBOFedSecondaryPurchaseCash",
        "CBOFedSecondaryPurchaseReserveEffect",
        "CBOFedSecondaryPurchaseDepositEffect",
        "CBOFedBeginStock",
        "CBOFedMaturitiesAndRedemptions",
        "CBOFedTipsPrincipalIndexation",
        "CBOFedAuctionRolloverAddons",
        "CBOFedSyntheticSecondaryPurchases",
        "CBOFedSyntheticSecondarySales",
        "CBOFedEndStock",
        "CBOFedGrossStockFlow",
        "CBOFedNetStockChange",
        "CBOFedStockMode",
        "CBOFedSettlementScope",
        "CBORemittanceCashEffect",
        "CBORemittanceStatus",
        "CB_NetIncome",
        "CB_Remittance",
        "CB_DeferredAsset",
        "DebtHeld_Banks",
        "DebtHeld_CentralBank",
        "DebtHeld_Foreign",
        "DebtHeld_DomesticNonBanks",
        "TIPSInflationAccretion_Cumulative",
        "CBOBuybackFaceRetired",
    }
    _require_columns(f"{run_id} results_compact.csv", rows[0], required)
    dates = [row["Date"] for row in rows]
    if dates != sorted(dates):
        raise ValueError(f"{run_id} results_compact.csv dates are not sorted")
    final = rows[-1]
    cash_residuals = [_float(row, "CBOCashReconciliationResidual") for row in rows]
    target_errors = []
    for row_index, row in enumerate(rows, start=1):
        if _float(row, "CBOControlledDebtTargetApplicable") <= 0.5:
            continue
        target_row = debt_targets.get(row["Date"])
        if target_row is None:
            raise ValueError(f"{run_id} row {row_index} has no debt-stock target for {row['Date']}")
        path_target = target_row["marketable_treasury_public_target_bil"]
        post_issuance = _float(row, "CBOControlledDebtPostIssuance")
        recomputed_error = post_issuance - path_target
        target_errors.append(recomputed_error)
        _assert_close(
            f"{run_id} row {row_index} CBOControlledDebtTarget vs debt stock path",
            _float(row, "CBOControlledDebtTarget"),
            path_target,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} row {row_index} reported debt target error",
            _float(row, "CBOControlledDebtTargetError"),
            recomputed_error,
            NUMERIC_TOLERANCE,
        )
    fed_errors = [
        _float(row, "CBOFedHoldingsTargetError")
        for row in rows
        if _float(row, "CBOFedHoldingsTargetApplicable") > 0.5
    ]
    if not target_errors:
        raise ValueError(f"{run_id} has no applicable controlled-debt target rows")
    if not fed_errors:
        raise ValueError(f"{run_id} has no applicable Fed holdings target rows")
    _verify_fed_stock_bridge(run_id, rows)
    metrics: dict[str, Any] = {
        "rows": len(rows),
        "start_date": dates[0],
        "final_date": dates[-1],
        "max_abs_target_error_bil": max(abs(value) for value in target_errors),
        "max_abs_cbo_fed_holdings_error_bil": max(abs(value) for value in fed_errors),
        "final_cbo_fed_holdings_error_bil": _float(final, "CBOFedHoldingsTargetError"),
        "final_cbo_fed_holdings_target_bil": _float(final, "CBOFedHoldingsTarget"),
        "final_cbo_target_bil": _float(final, "CBOControlledDebtTarget"),
        "final_total_debt_bil": _float(final, "TotalDebt_Agg"),
        "final_cbo_post_issuance_bil": _float(final, "CBOControlledDebtPostIssuance"),
        "final_tga_bil": _float(final, "TGA"),
        "final_operating_cash_target_bil": _float(final, "CBOOperatingCashTarget"),
        "final_cash_residual_bil": _float(final, "CBOCashResidual"),
        "net_cash_reconciliation_residual_bil": sum(cash_residuals),
        "gross_abs_cash_reconciliation_residual_bil": sum(abs(value) for value in cash_residuals),
        "positive_cash_reconciliation_residual_bil": sum(value for value in cash_residuals if value > 0.0),
        "negative_cash_reconciliation_residual_bil": sum(value for value in cash_residuals if value < 0.0),
        "nonzero_cash_reconciliation_residual_days": sum(1 for value in cash_residuals if abs(value) > 1e-12),
        "total_required_face_issuance_bil": _sum(rows, "CBORequiredFaceIssuance"),
        "total_auction_proceeds_bil": _sum(rows, "AuctionProceeds"),
        "total_issue_discount_cost_bil": _sum(rows, "IssueDiscountCost_Period"),
        "total_primary_deficit_bil": _sum(rows, "PrimaryDeficit"),
        "total_cbo_buyback_face_retired_bil": _sum(rows, "CBOBuybackFaceRetired"),
        "tips_inflation_accretion_cumulative_bil": _float(final, "TIPSInflationAccretion_Cumulative"),
        "rows_by_date": rows,
    }
    if metrics["max_abs_target_error_bil"] > NUMERIC_TOLERANCE:
        raise ValueError(f"{run_id} recomputed max debt target error exceeds tolerance")
    if metrics["max_abs_cbo_fed_holdings_error_bil"] > NUMERIC_TOLERANCE:
        raise ValueError(f"{run_id} recomputed max Fed holdings error exceeds tolerance")
    if abs(metrics["final_cbo_fed_holdings_error_bil"]) > NUMERIC_TOLERANCE:
        raise ValueError(f"{run_id} recomputed final Fed holdings error exceeds tolerance")
    _assert_close(
        f"{run_id} final controlled debt target",
        metrics["final_total_debt_bil"],
        metrics["final_cbo_target_bil"],
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        f"{run_id} final post-issuance controlled debt target",
        metrics["final_cbo_post_issuance_bil"],
        metrics["final_cbo_target_bil"],
        NUMERIC_TOLERANCE,
    )
    return metrics


def _verify_fed_stock_bridge(run_id: str, rows: list[dict[str, str]]) -> None:
    for row_index, row in enumerate(rows, start=1):
        target_applicable = _float(row, "CBOFedHoldingsTargetApplicable") > 0.5
        if not target_applicable:
            if _float(row, "CBOFedHoldingsTarget") != 0.0 or _float(row, "CBOFedHoldingsTargetError") != 0.0:
                raise ValueError(f"{run_id} row {row_index} has opening/nonapplicable Fed target sentinels")
            continue
        if row["CBOFedStockMode"] != "synthetic_cb_treasury_stock_target_par_reallocation":
            raise ValueError(f"{run_id} row {row_index} Fed stock mode is not explicit synthetic mode")
        if row["CBOFedSettlementScope"] != "stock_reallocation_only_no_reserve_deposit_or_market_price_claim":
            raise ValueError(f"{run_id} row {row_index} Fed settlement scope overclaims monetary effects")
        begin_stock = _float(row, "CBOFedBeginStock")
        maturities = _float(row, "CBOFedMaturitiesAndRedemptions")
        tips_indexation = _float(row, "CBOFedTipsPrincipalIndexation")
        auction_addons = _float(row, "CBOFedAuctionRolloverAddons")
        purchases = _float(row, "CBOFedSyntheticSecondaryPurchases")
        sales = _float(row, "CBOFedSyntheticSecondarySales")
        end_stock = _float(row, "CBOFedEndStock")
        expected_end = begin_stock - maturities + tips_indexation + auction_addons + purchases - sales
        _assert_close(f"{run_id} row {row_index} Fed stock bridge", end_stock, expected_end, NUMERIC_TOLERANCE)
        _assert_close(
            f"{run_id} row {row_index} Fed end stock vs DebtHeld_CentralBank",
            end_stock,
            _float(row, "DebtHeld_CentralBank"),
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} row {row_index} Fed target error",
            _float(row, "CBOFedHoldingsTargetError"),
            end_stock - _float(row, "CBOFedHoldingsTarget"),
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} row {row_index} Fed secondary purchase face",
            _float(row, "CBOFedSecondaryPurchaseFace"),
            purchases,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} row {row_index} Fed auction share",
            _float(row, "CBOFedAuctionShare"),
            0.0,
            NUMERIC_TOLERANCE,
        )
        expected_gross = sum(abs(value) for value in (maturities, tips_indexation, auction_addons, purchases, sales))
        _assert_close(
            f"{run_id} row {row_index} Fed gross stock flow",
            _float(row, "CBOFedGrossStockFlow"),
            expected_gross,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} row {row_index} Fed net stock change",
            _float(row, "CBOFedNetStockChange"),
            end_stock - begin_stock,
            NUMERIC_TOLERANCE,
        )
        for unsupported_column in (
            "CBOFedSecondaryPurchaseCash",
            "CBOFedSecondaryPurchaseReserveEffect",
            "CBOFedSecondaryPurchaseDepositEffect",
            "CBORemittanceCashEffect",
        ):
            if abs(_float(row, unsupported_column)) > NUMERIC_TOLERANCE:
                raise ValueError(f"{run_id} row {row_index} {unsupported_column} must remain zero in synthetic Fed mode")
        if row["CBORemittanceStatus"] != "not_modeled_cbo_primary_deficit_embeds_baseline_revenues":
            raise ValueError(f"{run_id} row {row_index} remittance status is not explicit non-modeled CBO status")
        for nonmodeled_column in ("CB_NetIncome", "CB_Remittance", "CB_DeferredAsset"):
            if not _is_blank_or_nan(row.get(nonmodeled_column)):
                raise ValueError(f"{run_id} row {row_index} {nonmodeled_column} must remain blank/NaN in CBO mode")


def _verify_run_summary(run_id: str, summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    for field in ("rows", "start_date", "final_date"):
        if summary.get(field) != metrics[field]:
            raise ValueError(f"{run_id} {field} does not match parsed results")
    for field in (
        "max_abs_target_error_bil",
        "max_abs_cbo_fed_holdings_error_bil",
        "final_cbo_fed_holdings_error_bil",
        "final_cbo_fed_holdings_target_bil",
        "final_cbo_target_bil",
        "final_total_debt_bil",
        "final_tga_bil",
        "final_operating_cash_target_bil",
        "final_cash_residual_bil",
        "net_cash_reconciliation_residual_bil",
        "gross_abs_cash_reconciliation_residual_bil",
        "positive_cash_reconciliation_residual_bil",
        "negative_cash_reconciliation_residual_bil",
        "total_required_face_issuance_bil",
        "total_auction_proceeds_bil",
        "total_issue_discount_cost_bil",
        "total_primary_deficit_bil",
        "total_cbo_buyback_face_retired_bil",
        "tips_inflation_accretion_cumulative_bil",
    ):
        if field in summary:
            _assert_close(f"{run_id} {field}", float(summary[field]), float(metrics[field]), NUMERIC_TOLERANCE)
    if "nonzero_cash_reconciliation_residual_days" in summary:
        if int(summary["nonzero_cash_reconciliation_residual_days"]) != int(
            metrics["nonzero_cash_reconciliation_residual_days"]
        ):
            raise ValueError(f"{run_id} nonzero cash residual day count does not match parsed results")


def _verify_cash_invariance(run_metrics: Mapping[str, Mapping[str, Any]]) -> None:
    baseline = run_metrics["zero_residual"]["rows_by_date"]
    for run_id in ("constant_real_tga", "constant_nominal_tga"):
        rows = run_metrics[run_id]["rows_by_date"]
        if len(rows) != len(baseline):
            raise ValueError(f"{run_id} row count differs from zero_residual")
        for index, (base_row, row) in enumerate(zip(baseline, rows)):
            if row["Date"] != base_row["Date"]:
                raise ValueError(f"{run_id} date sequence differs from zero_residual at row {index + 1}")
            for column in RESULT_INVARIANT_COLUMNS:
                _assert_close(
                    f"{run_id} cash-invariant {column} row {index + 1}",
                    _float(row, column),
                    _float(base_row, column),
                    STRICT_TOLERANCE,
                )


def _verify_cash_sensitivity_comparison(root: Path, run_metrics: Mapping[str, Mapping[str, Any]]) -> None:
    rows = _read_csv_dicts(root / "cash_sensitivity_comparison.csv")
    by_run = {row.get("run_id"): row for row in rows}
    for run_id, metrics in run_metrics.items():
        row = by_run.get(run_id)
        if row is None:
            raise ValueError(f"cash_sensitivity_comparison.csv missing row for {run_id}")
        if int(float(row.get("rows", "-1"))) != metrics["rows"]:
            raise ValueError(f"cash sensitivity row count mismatch for {run_id}")
        _assert_close(
            f"{run_id} cash sensitivity final_tga_bil",
            _csv_float(row, "final_tga_bil"),
            float(metrics["final_tga_bil"]),
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} cash sensitivity final_operating_cash_target_bil",
            _csv_float(row, "final_operating_cash_target_bil"),
            float(metrics["final_operating_cash_target_bil"]),
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} cash sensitivity net_cash_reconciliation_residual_bil",
            _csv_float(row, "net_cash_reconciliation_residual_bil"),
            float(metrics["net_cash_reconciliation_residual_bil"]),
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            f"{run_id} cash sensitivity gross_abs_cash_reconciliation_residual_bil",
            _csv_float(row, "gross_abs_cash_reconciliation_residual_bil"),
            float(metrics["gross_abs_cash_reconciliation_residual_bil"]),
            NUMERIC_TOLERANCE,
        )
        for column in RESULT_INVARIANT_COLUMNS:
            comparison_column = f"max_abs_delta_{column}"
            if comparison_column in row:
                expected_delta = _max_abs_delta(
                    run_metrics["zero_residual"]["rows_by_date"],
                    metrics["rows_by_date"],
                    column,
                )
                _assert_close(
                    f"{run_id} cash sensitivity {comparison_column}",
                    _csv_float(row, comparison_column),
                    expected_delta,
                    STRICT_TOLERANCE,
                )


def _verify_partial_interest_financing_diagnostic(root: Path, zero_metrics: Mapping[str, Any]) -> None:
    modeled_path = root / "tdcsim_modeled_net_interest_diagnostic.csv"
    partial_path = root / "tdcsim_partial_interest_financing_diagnostic.csv"
    if modeled_path.exists():
        raise ValueError("deprecated modeled net-interest diagnostic file must not be packaged")
    rows = _read_csv_dicts(partial_path)
    if not rows:
        raise ValueError("tdcsim_partial_interest_financing_diagnostic.csv has no data rows")
    forbidden_columns = sorted(column for column in rows[0] if column in FORBIDDEN_NET_INTEREST_COLUMNS)
    if forbidden_columns:
        raise ValueError(f"partial interest diagnostic contains forbidden overclaim columns: {forbidden_columns}")
    required = {
        "run_id",
        "fiscal_year",
        "cbo_net_interest_bil",
        "residual_bil",
        "scope_status",
        "calibration_mode",
        "threshold_status",
        "claim_status",
        "coupon_cash_interest_bil",
        "issuance_discount_booked_at_issue_proxy_bil",
        "tips_principal_indexation_bil",
        "nonmarketable_interest_capitalized_bil",
        "modeled_interest_related_proxy_bil",
        "runtime_role",
        "scope_completeness_status",
        "numeric_residual_status",
        "claim_boundary",
    }
    _require_columns("tdcsim_partial_interest_financing_diagnostic.csv", rows[0], required)
    fiscal_years = [int(float(row["fiscal_year"])) for row in rows]
    if fiscal_years != sorted(fiscal_years):
        raise ValueError("net-interest diagnostic fiscal years are not sorted")
    if fiscal_years[0] != 2026 or fiscal_years[-1] != 2036:
        raise ValueError("net-interest diagnostic must cover fiscal years 2026 through 2036")
    result_rows = zero_metrics["rows_by_date"]
    grouped: dict[int, list[dict[str, str]]] = {}
    for result_row in result_rows:
        fiscal_year = _fiscal_year(result_row["Date"])
        grouped.setdefault(fiscal_year, []).append(result_row)
    for row in rows:
        fiscal_year = int(float(row["fiscal_year"]))
        source_rows = grouped.get(fiscal_year)
        if not source_rows:
            raise ValueError(f"net-interest diagnostic fiscal year has no zero-residual rows: {fiscal_year}")
        coupon_cash = _sum(source_rows, "InterestPaid_Bonds")
        bill_discount = _sum(source_rows, "IssueDiscountCost_Period")
        tips_indexation = _sum(source_rows, "TIPSInflationAccretion_Period")
        nonmarketable = _sum(source_rows, "NonMarketableInterestCapitalized_Period")
        interest_related_proxy = coupon_cash + bill_discount + tips_indexation + nonmarketable
        _assert_close("net-interest coupon cash", _csv_float(row, "coupon_cash_interest_bil"), coupon_cash, NUMERIC_TOLERANCE)
        _assert_close(
            "net-interest issuance discount booked at issue",
            _csv_float(row, "issuance_discount_booked_at_issue_proxy_bil"),
            bill_discount,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            "net-interest TIPS indexation",
            _csv_float(row, "tips_principal_indexation_bil"),
            tips_indexation,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            "net-interest nonmarketable interest",
            _csv_float(row, "nonmarketable_interest_capitalized_bil"),
            nonmarketable,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            "net-interest modeled interest-related proxy",
            _csv_float(row, "modeled_interest_related_proxy_bil"),
            interest_related_proxy,
            NUMERIC_TOLERANCE,
        )
        cbo_value = _csv_float(row, "cbo_net_interest_bil")
        _assert_close(
            "net-interest residual",
            _csv_float(row, "residual_bil"),
            cbo_value - interest_related_proxy,
            NUMERIC_TOLERANCE,
        )
        if row["scope_status"] != "partial_market_securities_proxy":
            raise ValueError("net-interest diagnostic scope_status must remain partial")
        if row["calibration_mode"] != "diagnostic_only_no_cbo_interest_plug":
            raise ValueError("net-interest diagnostic calibration mode must remain non-plug")
        if row["runtime_role"] != "diagnostic_only":
            raise ValueError("net-interest diagnostic runtime_role must be diagnostic_only")
        if row["scope_completeness_status"] != "partial_interest_related_proxy_not_budgetary_net_interest":
            raise ValueError("net-interest diagnostic scope completeness status must remain partial")
        if row["claim_status"] != "partial_nonbinding_diagnostic_regardless_of_numeric_residual":
            raise ValueError("net-interest diagnostic claim_status must be partial and nonbinding")
        if row["claim_boundary"] != "no_cbo_net_interest_cash_or_issuance_plug":
            raise ValueError("net-interest diagnostic claim boundary changed")


def _verify_debt_stock_path(root: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_debt_stock_path.csv")
    if not rows:
        raise ValueError("tdcsim_debt_stock_path.csv has no data rows")
    row_counts = manifest.get("source_values", {}).get("row_counts", {})
    expected_rows = int(row_counts.get("period_rows", 3755))
    if len(rows) != expected_rows:
        raise ValueError("tdcsim_debt_stock_path.csv row count changed")
    required = {
        "schema_version",
        "scenario_id",
        "period_end",
        "cbo_federal_debt_held_public_target_bil",
        "public_nonmarketable_treasury_bil",
        "non_treasury_and_definition_residual_bil",
        "marketable_treasury_public_target_bil",
        "bridge_source",
        "bridge_method",
        "anchor_type",
        "interpolation_method",
        "source_role",
        "runtime_role",
        "source_fiscal_year",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
    _require_columns("tdcsim_debt_stock_path.csv", rows[0], required)
    fiscal_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_cbo_fiscal_baseline.csv")
    _require_columns("tdcsim_cbo_fiscal_baseline.csv", fiscal_rows[0], {"fiscal_year", "debt_held_public_end_bil"})
    cbo_debt_by_fy = {
        int(float(row["fiscal_year"])): _csv_float(row, "debt_held_public_end_bil")
        for row in fiscal_rows
        if row.get("debt_held_public_end_bil") not in (None, "")
    }
    by_date: dict[str, dict[str, float]] = {}
    dates = [row["period_end"] for row in rows]
    if dates != sorted(dates):
        raise ValueError("tdcsim_debt_stock_path.csv dates are not sorted")
    parsed_dates = [date.fromisoformat(row["period_end"]) for row in rows]
    opening_date = date.fromisoformat(
        str(manifest.get("date_range", {}).get("simulation_start_date", "2026-06-21"))
    )
    expected_start = opening_date if parsed_dates[0] == opening_date else opening_date + timedelta(days=1)
    anchor_points: list[tuple[date, float]] = []
    if parsed_dates[0] > opening_date:
        debt_bridge = manifest.get("source_values", {}).get("debt_bridge", {})
        if not isinstance(debt_bridge, Mapping):
            raise ValueError("manifest source_values.debt_bridge must be populated")
        opening_public_target = _manifest_float(debt_bridge, "treasury_public_debt_bil") + _manifest_float(
            debt_bridge,
            "non_treasury_and_definition_residual_bil",
        )
        anchor_points.append((opening_date, opening_public_target))
    for row, parsed_date in zip(rows, parsed_dates):
        anchor_type = row["anchor_type"]
        if anchor_type in {"actual_as_of", "cbo_fiscal_year_end"}:
            anchor_points.append((parsed_date, _csv_float(row, "cbo_federal_debt_held_public_target_bil")))
    if parsed_dates[0] != expected_start:
        raise ValueError("debt stock path first row date changed")
    if anchor_points != sorted(anchor_points):
        raise ValueError("debt stock path anchors are not sorted")
    anchor_by_date = dict(anchor_points)
    expected_public_by_date: dict[date, float] = {}
    for (start_date, start_value), (end_date, end_value) in zip(anchor_points, anchor_points[1:]):
        span_days = (end_date - start_date).days
        if span_days <= 0:
            raise ValueError("debt stock path anchors must be strictly increasing")
        for parsed_date in parsed_dates:
            if start_date <= parsed_date <= end_date:
                elapsed_days = (parsed_date - start_date).days
                expected_public_by_date[parsed_date] = start_value + (end_value - start_value) * (
                    elapsed_days / span_days
                )
    final_anchor_date = anchor_points[-1][0]
    for parsed_date in parsed_dates:
        if parsed_date > final_anchor_date:
            expected_public_by_date[parsed_date] = anchor_points[-1][1]
    for index, row in enumerate(rows):
        expected_date = expected_start + timedelta(days=index)
        parsed_date = parsed_dates[index]
        if parsed_date != expected_date:
            raise ValueError(f"debt stock path row {index + 1} date sequence changed")
        if row["schema_version"] != "tdcsim_debt_stock_path_v1":
            raise ValueError("debt stock path schema_version changed")
        if row["scenario_id"] != "cbo_full_horizon_local_smoke_3m":
            raise ValueError("debt stock path scenario_id changed")
        if row["runtime_role"] != "hard_target":
            raise ValueError("debt stock path runtime_role must be hard_target")
        if row["interpolation_method"] != "linear_actual_days":
            raise ValueError("debt stock path interpolation_method changed")
        if row["claim_boundary"] != "bridge_components_not_auction_supply":
            raise ValueError("debt stock path claim boundary changed")
        public_target = _csv_float(row, "cbo_federal_debt_held_public_target_bil")
        public_nonmarketable = _csv_float(row, "public_nonmarketable_treasury_bil")
        residual = _csv_float(row, "non_treasury_and_definition_residual_bil")
        marketable_target = _csv_float(row, "marketable_treasury_public_target_bil")
        expected_public_target = expected_public_by_date.get(parsed_date)
        if expected_public_target is None:
            raise ValueError(f"debt stock path row {index + 1} is outside declared interpolation anchors")
        _assert_close(
            f"debt stock path row {index + 1} actual-day interpolated public target",
            public_target,
            expected_public_target,
            NUMERIC_TOLERANCE,
        )
        _assert_close(
            "debt stock path marketable target bridge",
            marketable_target,
            public_target - public_nonmarketable - residual,
            NUMERIC_TOLERANCE,
        )
        anchor_type = row["anchor_type"]
        if anchor_type == "actual_as_of":
            if parsed_date not in anchor_by_date:
                raise ValueError("actual debt stock anchor date missing from interpolation anchors")
            if row["source_role"] != "hard_actual_state":
                raise ValueError("actual debt stock anchor must be hard_actual_state")
        elif anchor_type == "cbo_fiscal_year_end":
            if parsed_date not in anchor_by_date:
                raise ValueError("CBO fiscal-year debt anchor date missing from interpolation anchors")
            fiscal_year = int(float(row["source_fiscal_year"]))
            expected_public = cbo_debt_by_fy.get(fiscal_year)
            if expected_public is None:
                raise ValueError(f"debt stock path has CBO anchor without fiscal baseline year: {fiscal_year}")
            if row["source_role"] != "official_hard_anchor":
                raise ValueError("CBO fiscal-year debt anchor must be official_hard_anchor")
            _assert_close(
                f"debt stock path FY{fiscal_year} CBO public-debt anchor",
                public_target,
                expected_public,
                NUMERIC_TOLERANCE,
            )
        elif anchor_type == "interpolated_assumption":
            if row["source_role"] != "scenario_assumption":
                raise ValueError("interpolated debt stock rows must be scenario_assumption")
        else:
            raise ValueError(f"debt stock path has unexpected anchor_type: {anchor_type}")
        by_date[row["period_end"]] = {
            "cbo_federal_debt_held_public_target_bil": public_target,
            "marketable_treasury_public_target_bil": marketable_target,
        }
    return by_date


def _verify_opening_portfolio(root: Path, manifest: Mapping[str, Any]) -> None:
    opening_manifest = manifest.get("opening_portfolio_construction", {})
    if not isinstance(opening_manifest, Mapping):
        raise ValueError("manifest opening_portfolio_construction must be populated")
    portfolio = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_opening_portfolio.csv")
    tips_diagnostics = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_opening_tips_indexation_diagnostics.csv")
    frn_diagnostics = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_opening_frn_indexation_diagnostics.csv")

    if len(portfolio) != int(opening_manifest.get("row_count", -1)):
        raise ValueError("opening portfolio row count does not match parsed CSV")
    tips_rows = [row for row in portfolio if row.get("SecurityType") == "TIPS"]
    frn_rows = [row for row in portfolio if row.get("SecurityType") == "FRN"]
    if len(tips_rows) != int(opening_manifest.get("tips_rows_enriched", -1)):
        raise ValueError("opening TIPS row count does not match parsed CSV")
    if len(frn_rows) != int(opening_manifest.get("frn_rows_enriched", -1)):
        raise ValueError("opening FRN row count does not match parsed CSV")
    tips_cusips = {row.get("cusip") for row in tips_diagnostics if row.get("cusip")}
    frn_cusips = {row.get("cusip") for row in frn_diagnostics if row.get("cusip")}
    if len(tips_cusips) != int(opening_manifest.get("tips_unique_cusips_enriched", -1)):
        raise ValueError("opening TIPS unique CUSIP count does not match diagnostics CSV")
    if len(frn_cusips) != int(opening_manifest.get("frn_unique_cusips_enriched", -1)):
        raise ValueError("opening FRN unique CUSIP count does not match diagnostics CSV")

    tips_adjusted = sum(_csv_float(row, "AdjustedPrincipal") for row in tips_rows)
    tips_original = sum(_csv_float(row, "OriginalPrincipal") for row in tips_rows)
    frn_face = sum(_csv_float(row, "FaceValue") for row in frn_rows)
    frn_accrued = sum(_csv_float(row, "AccruedInterest_FRN") for row in frn_rows)
    _assert_close(
        "opening TIPS adjusted principal",
        tips_adjusted,
        float(opening_manifest.get("opening_tips_adjusted_principal_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "opening TIPS reconstructed original principal",
        tips_original,
        float(opening_manifest.get("opening_tips_reconstructed_original_principal_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "opening TIPS embedded accretion",
        tips_adjusted - tips_original,
        float(opening_manifest.get("opening_tips_indexation_accretion_embedded_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "opening FRN face value",
        frn_face,
        float(opening_manifest.get("opening_frn_face_value_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "opening FRN accrued interest",
        frn_accrued,
        float(opening_manifest.get("opening_frn_accrued_interest_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )

    _assert_close(
        "TIPS diagnostics adjusted principal",
        sum(_csv_float(row, "adjusted_principal_bil") for row in tips_diagnostics),
        tips_adjusted,
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "TIPS diagnostics reconstructed original principal",
        sum(_csv_float(row, "reconstructed_original_principal_bil") for row in tips_diagnostics),
        tips_original,
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "FRN diagnostics face value",
        sum(_csv_float(row, "face_value_bil") for row in frn_diagnostics),
        frn_face,
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "FRN diagnostics accrued interest",
        sum(_csv_float(row, "accrued_interest_frn_bil") for row in frn_diagnostics),
        frn_accrued,
        NUMERIC_TOLERANCE,
    )
    _verify_opening_totals(opening_manifest, portfolio)


def _verify_opening_totals(opening_manifest: Mapping[str, Any], portfolio: list[dict[str, str]]) -> None:
    security_totals: dict[str, float] = {}
    holder_totals: dict[str, float] = {}
    for row in portfolio:
        debt_base = _opening_debt_base(row)
        security_type = row.get("SecurityType", "")
        holder = row.get("HolderType", "")
        security_totals[security_type] = security_totals.get(security_type, 0.0) + debt_base
        holder_totals[holder] = holder_totals.get(holder, 0.0) + debt_base
    for name, expected in opening_manifest.get("security_type_totals_bil", {}).items():
        _assert_close(f"opening security total {name}", security_totals.get(name, 0.0), float(expected), NUMERIC_TOLERANCE)
    for name, expected in opening_manifest.get("holder_totals_bil", {}).items():
        _assert_close(f"opening holder total {name}", holder_totals.get(name, 0.0), float(expected), NUMERIC_TOLERANCE)


def _verify_tips_forward_paths(root: Path, manifest: Mapping[str, Any]) -> None:
    cpi_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_tips_cpi_path.csv")
    real_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_tips_real_yield_path.csv")
    if not cpi_rows:
        raise ValueError("TIPS CPI path must not be empty")
    if not real_rows:
        raise ValueError("TIPS real-yield path must not be empty")
    row_counts = manifest.get("source_values", {}).get("row_counts", {})
    if int(row_counts.get("tips_cpi_path_rows", -1)) != len(cpi_rows):
        raise ValueError("manifest TIPS CPI path row count does not match CSV")
    if int(row_counts.get("tips_real_yield_path_rows", -1)) != len(real_rows):
        raise ValueError("manifest TIPS real-yield path row count does not match CSV")
    tips_manifest = manifest.get("tips_forward_paths", {})
    if not isinstance(tips_manifest, Mapping):
        raise ValueError("manifest tips_forward_paths must be populated")
    if "not a cash, issuance, or net-interest plug" not in str(tips_manifest.get("runtime_effect", "")):
        raise ValueError("TIPS forward path runtime boundary must remain non-plug")
    if tips_manifest.get("cpi_file") not in (None, "forecast_inputs/tdcsim_tips_cpi_path.csv"):
        raise ValueError("tips_forward_paths.cpi_file changed")
    if tips_manifest.get("real_yield_file") not in (None, "forecast_inputs/tdcsim_tips_real_yield_path.csv"):
        raise ValueError("tips_forward_paths.real_yield_file changed")
    months = [row["month"] for row in cpi_rows]
    if months != sorted(months):
        raise ValueError("TIPS CPI path months must be sorted")
    scale_values = {_csv_float(row, "scale_factor") for row in cpi_rows}
    if len(scale_values) != 1 or next(iter(scale_values)) <= 0.0:
        raise ValueError("TIPS CPI path must use one positive scale_factor")
    for row in cpi_rows:
        if not row["month"].endswith("-01"):
            raise ValueError("TIPS CPI path month values must be first-of-month")
        if row["source_role"] != "scenario_assumption" or row["runtime_role"] != "hard_target":
            raise ValueError("TIPS CPI path role boundary changed")
        if _csv_float(row, "tips_cpi_u_index") <= 0.0 or _csv_float(row, "cbo_cpi_u_index") <= 0.0:
            raise ValueError("TIPS CPI path CPI values must be positive")
    yield_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_yield_curve_surface.csv")
    expected_cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id=str(manifest.get("scenario_id")),
        macro_forecast_rows=_read_csv_dicts(root / "forecast_inputs" / "tdcsim_macro_forecast_path.csv"),
        simulation_start_date=str(manifest.get("date_range", {}).get("simulation_start_date")),
        simulation_end_date=str(manifest.get("date_range", {}).get("simulation_end_date")),
        opening_reference_cpi=float(manifest["opening_portfolio_construction"]["opening_tips_reference_cpi"]),
        reference_lag_months=int(float(tips_manifest.get("reference_cpi_lag_months", cpi_rows[0]["reference_lag_months"]))),
        pricing_horizon_end_date=str(
            tips_manifest.get(
                "max_expected_inflation_horizon_date",
                _max_expected_inflation_horizon_date(yield_rows),
            )
        ),
        observation_date=str(cpi_rows[0]["observation_date"]),
        available_date=str(cpi_rows[0]["available_date"]),
    )
    _verify_csv_rows_match_expected("tdcsim_tips_cpi_path.csv", cpi_rows, expected_cpi_rows)
    nominal_by_key = {
        (row["curve_date"], row["tenor_years"]): _csv_float(row, "nominal_rate_decimal")
        for row in yield_rows
    }
    if len(real_rows) != len(nominal_by_key):
        raise ValueError("TIPS real-yield path must align one-for-one with yield surface rows")
    for row in real_rows:
        key = (row["curve_date"], row["tenor_years"])
        if key not in nominal_by_key:
            raise ValueError(f"TIPS real-yield row missing nominal yield key: {key}")
        if row["source_role"] != "scenario_assumption" or row["runtime_role"] != "hard_target":
            raise ValueError("TIPS real-yield path role boundary changed")
        _assert_close(
            "TIPS real-yield nominal source",
            _csv_float(row, "nominal_rate_decimal"),
            nominal_by_key[key],
            STRICT_TOLERANCE,
        )
        _assert_close(
            "TIPS real-yield identity",
            _csv_float(row, "real_yield_decimal"),
            _csv_float(row, "nominal_rate_decimal") - _csv_float(row, "expected_inflation_decimal"),
            STRICT_TOLERANCE,
        )
        if _csv_float(row, "real_coupon_decimal") < 0.0:
            raise ValueError("TIPS real coupon must be nonnegative")
        if row["pricing_method"] != "real_cashflow_present_value_semiannual":
            raise ValueError("TIPS real-yield pricing method changed")
    expected_real_rows = build_tips_real_yield_path_rows(
        scenario_id=str(manifest.get("scenario_id")),
        yield_surface_rows=yield_rows,
        tips_cpi_path_rows=expected_cpi_rows,
        observation_date=str(real_rows[0]["observation_date"]),
        available_date=str(real_rows[0]["available_date"]),
    )
    _verify_csv_rows_match_expected("tdcsim_tips_real_yield_path.csv", real_rows, expected_real_rows)


def _max_expected_inflation_horizon_date(yield_rows: list[dict[str, str]]) -> str:
    if not yield_rows:
        raise ValueError("TIPS real-yield path requires yield surface rows")
    horizon_dates = []
    for row in yield_rows:
        curve_date = date.fromisoformat(row["curve_date"])
        months = max(1, int(round(_csv_float(row, "tenor_years") * 12.0)))
        year = curve_date.year + ((curve_date.month - 1 + months) // 12)
        month = ((curve_date.month - 1 + months) % 12) + 1
        day = min(curve_date.day, _days_in_month(year, month))
        horizon_dates.append(date(year, month, day))
    return max(horizon_dates).isoformat()


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def _verify_frn_rate_path(root: Path, manifest: Mapping[str, Any]) -> None:
    rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_frn_rate_path.csv")
    row_counts = manifest.get("source_values", {}).get("row_counts", {})
    expected_rows = int(row_counts.get("frn_rate_path_rows", -1))
    if expected_rows != len(rows):
        raise ValueError("manifest FRN rate path row count does not match CSV")
    if expected_rows != int(row_counts.get("period_rows", -2)):
        raise ValueError("FRN rate path row count must equal period_rows")
    frn_manifest = manifest.get("frn_forward_rate_path", {})
    if not isinstance(frn_manifest, Mapping):
        raise ValueError("manifest frn_forward_rate_path must be populated")
    if "not a cash, issuance, or net-interest plug" not in str(frn_manifest.get("runtime_effect", "")):
        raise ValueError("FRN forward path runtime boundary must remain non-plug")
    yield_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_yield_curve_surface.csv")
    three_month = sorted(
        (
            row["curve_date"],
            _csv_float(row, "nominal_rate_decimal"),
        )
        for row in yield_rows
        if abs(_csv_float(row, "tenor_years") - 0.25) <= 1e-9
    )
    if not three_month:
        raise ValueError("FRN rate path requires 3-month yield surface rows")
    prior_end = None
    for row in rows:
        source_candidates = [(date_value, rate) for date_value, rate in three_month if date_value <= row["period_end"]]
        if not source_candidates:
            raise ValueError(f"FRN rate path date has no prior 3-month yield surface row: {row['period_end']}")
        source_date, source_rate = source_candidates[-1]
        if f"latest_curve_date_{source_date}" not in row["source_status"]:
            raise ValueError("FRN rate path source_status must identify latest prior yield curve date")
        if row["source_role"] != "scenario_assumption":
            raise ValueError("FRN rate path source_role must be scenario_assumption")
        if row["runtime_role"] != "hard_target":
            raise ValueError("FRN rate path runtime_role must be hard_target")
        if row["spread_treatment"] != "add_security_fixed_spread_decimal":
            raise ValueError("FRN rate path must preserve security-level fixed spread treatment")
        _assert_close("FRN benchmark vs 3-month yield surface", _csv_float(row, "benchmark_rate_decimal"), source_rate, STRICT_TOLERANCE)
        _assert_close("FRN auction high rate", _csv_float(row, "auction_high_rate_decimal"), _csv_float(row, "benchmark_rate_decimal"), STRICT_TOLERANCE)
        _assert_close("FRN day count basis", _csv_float(row, "day_count_basis"), 360.0, STRICT_TOLERANCE)
        if _csv_float(row, "lockout_business_days") != 2.0:
            raise ValueError("FRN rate path lockout_business_days must be 2")
        period_start = row["period_start"]
        period_end = row["period_end"]
        if period_start >= period_end:
            raise ValueError("FRN rate path period_start must precede period_end")
        if prior_end is not None and period_start != prior_end:
            raise ValueError("FRN rate path must be contiguous by period_start/period_end")
        prior_end = period_end
        prior_end = period_end


def _verify_yield_surface_against_sources(root: Path) -> None:
    actual_rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_yield_curve_surface.csv")
    if not actual_rows:
        raise ValueError("tdcsim_yield_curve_surface.csv has no data rows")
    tenors = sorted({_csv_float(row, "tenor_years") for row in actual_rows})
    expected_rows = build_yield_curve_surface_rows(
        macro_forecast_rows=_read_csv_dicts(root / "forecast_inputs" / "tdcsim_macro_forecast_path.csv"),
        base_curve_rows=_packaged_base_curve_rows(root),
        actuals_available_as_of="2026-06-17",
        output_tenors=tenors,
    )
    _verify_csv_rows_match_expected("tdcsim_yield_curve_surface.csv", actual_rows, expected_rows)


def _packaged_base_curve_rows(root: Path) -> list[dict[str, Any]]:
    base_curve_path = root / "sources" / "treasury" / "base_curve_2026-06-16.json"
    base_curve = _read_json(base_curve_path)
    rows = base_curve.get("rows") if isinstance(base_curve, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("packaged Treasury base curve extract must contain rows")
    rows_without_hash = [
        {
            "tenor_years": float(row["tenor_years"]),
            "nominal_rate": float(row.get("nominal_rate", row.get("nominal_rate_pct", row.get("rate_pct")))),
        }
        for row in rows
    ]
    source_hash = _sha256_file(base_curve_path)
    source_key = str(base_curve.get("source_key", "official_treasury_daily_nominal_par_curve_2026_06_16"))
    return [
        {
            "base_curve_date": str(row.get("base_curve_date", base_curve.get("observation_date", "2026-06-16"))),
            "available_date": str(row.get("available_date", base_curve.get("available_date", "2026-06-16"))),
            "base_curve_source_key": source_key,
            "base_curve_sha256": source_hash,
            "tenor_years": row["tenor_years"],
            "nominal_rate": row["nominal_rate"],
        }
        for row in rows_without_hash
    ]


def _verify_rollforward(root: Path, manifest: Mapping[str, Any]) -> None:
    opening_manifest = manifest.get("opening_portfolio_construction", {})
    rows = _read_csv_dicts(root / "forecast_inputs" / "tdcsim_opening_rollforward_diagnostics.csv")
    summary_rows = [row for row in rows if row.get("row_type") == "summary"]
    replacement_rows = [row for row in rows if row.get("row_type") == "replacement"]
    if len(summary_rows) != 1:
        raise ValueError("roll-forward diagnostics must contain exactly one summary row")
    summary = summary_rows[0]
    expected_count = int(opening_manifest.get("prestart_rollforward_rows", -1))
    if len(replacement_rows) != expected_count:
        raise ValueError("roll-forward replacement count does not match manifest")
    if int(float(summary.get("replacement_rows", "-1"))) != len(replacement_rows):
        raise ValueError("roll-forward summary replacement_rows does not match parsed CSV")
    if int(float(summary.get("source_rows_maturing_on_or_before_start", "-1"))) != len(replacement_rows):
        raise ValueError("roll-forward summary source row count does not match parsed CSV")
    source_face = sum(_csv_float(row, "source_face_bil") for row in replacement_rows)
    replacement_face = sum(_csv_float(row, "replacement_face_bil") for row in replacement_rows)
    _assert_close("roll-forward source face", source_face, _csv_float(summary, "source_face_bil"), NUMERIC_TOLERANCE)
    _assert_close(
        "roll-forward replacement face",
        replacement_face,
        _csv_float(summary, "replacement_face_bil"),
        NUMERIC_TOLERANCE,
    )
    _assert_close(
        "roll-forward manifest face",
        replacement_face,
        float(opening_manifest.get("prestart_rollforward_face_bil", math.nan)),
        NUMERIC_TOLERANCE,
    )
    _assert_close("roll-forward face delta", replacement_face - source_face, _csv_float(summary, "face_delta_bil"), NUMERIC_TOLERANCE)
    _assert_close(
        "roll-forward debt-base delta",
        float(opening_manifest.get("opening_rollforward_debt_base_delta_bil", math.nan)),
        0.0,
        NUMERIC_TOLERANCE,
    )


def _verify_required_files(root: Path) -> None:
    missing = sorted(rel for rel in REQUIRED_FILES if not (root / rel).exists())
    if missing:
        raise FileNotFoundError(f"CBO package is missing required files: {missing}")
    forbidden = sorted(rel for rel in FORBIDDEN_PACKAGE_FILES if (root / rel).exists())
    if forbidden:
        raise ValueError(f"CBO package includes deprecated/overclaim files: {forbidden}")


def _reject_overclaiming_manifest_fields(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if key_text in FORBIDDEN_MANIFEST_KEYS:
                raise ValueError(f"manifest contains forbidden overclaim field: {child_path}")
            _reject_overclaiming_manifest_fields(child, path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_overclaiming_manifest_fields(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lower = value.lower()
        for phrase in FORBIDDEN_CLAIM_PHRASES:
            if phrase in lower:
                raise ValueError(f"manifest contains forbidden claim wording at {path}: {phrase}")


def _verify_package_file_allowlist(root: Path, manifest: Mapping[str, Any]) -> None:
    source_package = manifest.get("source_package")
    source_files = source_package.get("source_files", {}) if isinstance(source_package, Mapping) else {}
    artifact_hashes = manifest.get("artifact_hashes", {})
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("manifest artifact_hashes must be a mapping")

    allowed = set(REQUIRED_FILES) | OPTIONAL_KNOWN_FILES
    if isinstance(source_files, Mapping):
        allowed.update(f"sources/{rel}" for rel in source_files)
    extras = []
    missing_artifact_inventory = []
    missing_source_inventory = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel not in allowed:
            extras.append(rel)
            continue
        if rel not in MANIFEST_EQUIVALENT_FILES and rel not in artifact_hashes:
            missing_artifact_inventory.append(rel)
        if rel.startswith("sources/"):
            source_rel = rel.removeprefix("sources/")
            if not isinstance(source_files, Mapping) or source_rel not in source_files:
                missing_source_inventory.append(rel)
    if extras:
        raise ValueError(f"CBO package includes unknown/unlisted files: {extras}")
    if missing_artifact_inventory:
        raise ValueError(f"CBO package files missing from artifact inventory: {missing_artifact_inventory}")
    if missing_source_inventory:
        raise ValueError(f"CBO source files missing from source inventory: {missing_source_inventory}")


def _verify_trust_anchor(root: Path, manifest: Mapping[str, Any], *, package_zip_sha256: str | None) -> None:
    trust = manifest.get("trust_anchor")
    if not isinstance(trust, Mapping):
        raise ValueError("manifest trust_anchor must be populated")
    if trust.get("schema_version") != TRUST_ANCHOR_SCHEMA_VERSION:
        raise ValueError("trust_anchor schema_version changed")
    release_sha = trust.get("release_commit_sha")
    release_status = trust.get("release_commit_status")
    if release_status == PENDING_RELEASE_STATUS:
        if release_sha not in (None, "", PENDING_RELEASE_STATUS):
            raise ValueError("pending release commit must not carry a concrete release_commit_sha")
    elif not (isinstance(release_sha, str) and re.fullmatch(r"[0-9a-f]{40}", release_sha)):
        raise ValueError("trust_anchor.release_commit_sha must be a 40-character lowercase git SHA or explicit pending")

    dirty_state = trust.get("dirty_state")
    if not isinstance(dirty_state, bool):
        raise ValueError("trust_anchor.dirty_state must be a boolean")
    python_version = trust.get("python_version")
    if not (isinstance(python_version, str) and re.fullmatch(r"\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)?", python_version)):
        raise ValueError("trust_anchor.python_version must look like a Python version")

    zip_status = trust.get("package_zip_sha256_status")
    zip_sha = trust.get("package_zip_sha256")
    if zip_status == PENDING_ZIP_STATUS:
        if zip_sha not in (None, "", PENDING_ZIP_STATUS):
            raise ValueError("pending package zip binding must not carry a concrete package_zip_sha256")
    elif isinstance(zip_sha, str) and re.fullmatch(r"[0-9a-f]{64}", zip_sha):
        if package_zip_sha256 is not None and zip_sha != package_zip_sha256:
            raise ValueError("trust_anchor.package_zip_sha256 does not match verified zip")
    else:
        raise ValueError("trust_anchor.package_zip_sha256 must be a SHA-256 or explicit pending binding")

    if trust.get("verifier_version") != VERIFIER_VERSION:
        raise ValueError("trust_anchor.verifier_version changed")
    verifier_sha = trust.get("verifier_sha256")
    actual_verifier_sha = _sha256_file(Path(__file__))
    if verifier_sha != actual_verifier_sha:
        raise ValueError("trust_anchor.verifier_sha256 does not match verifier code")
    runtime = trust.get("verifier_runtime")
    if runtime not in (None, platform.python_implementation()):
        raise ValueError("trust_anchor.verifier_runtime does not match current Python runtime")
    if trust.get("dependency_lock_status") != "requirements_lock_txt_present":
        raise ValueError("trust_anchor.dependency_lock_status must record requirements.lock.txt")
    lock_sha = trust.get("requirements_lock_sha256")
    if not (isinstance(lock_sha, str) and re.fullmatch(r"[0-9a-f]{64}", lock_sha)):
        raise ValueError("trust_anchor.requirements_lock_sha256 must be a SHA-256")
    actual_lock_sha = _sha256_file(root / "requirements.lock.txt")
    if lock_sha != actual_lock_sha:
        raise ValueError("trust_anchor.requirements_lock_sha256 does not match packaged requirements.lock.txt")


def _verify_claim_boundary(manifest: Mapping[str, Any]) -> None:
    boundary = manifest.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("manifest claim_boundary must be populated")
    expected = {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "package_role": "cbo_baseline_forecast_evidence_package",
        "net_interest_role": "partial_interest_financing_diagnostic_only",
        "issuance_mix_role": "tdcsim_assumption_not_cbo_prescription",
        "cash_role": "operating_cash_sensitivity_only_not_debt_or_issuance_driver",
    }
    for key, expected_value in expected.items():
        if boundary.get(key) != expected_value:
            raise ValueError(f"claim_boundary.{key} expected {expected_value}, found {boundary.get(key)}")
    exclusions = boundary.get("excluded_claims")
    if not isinstance(exclusions, list) or set(exclusions) != CLAIM_BOUNDARY_EXCLUSIONS:
        raise ValueError("claim_boundary.excluded_claims changed")


def _verify_artifact_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    hashes = manifest.get("artifact_hashes", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("manifest artifact_hashes must be populated")
    expected_hashes = (set(REQUIRED_FILES) | OPTIONAL_KNOWN_FILES) - MANIFEST_EQUIVALENT_FILES
    actual_hashes = set(str(rel) for rel in hashes)
    if actual_hashes != expected_hashes:
        missing = sorted(expected_hashes - actual_hashes)
        extra = sorted(actual_hashes - expected_hashes)
        raise ValueError(f"manifest artifact_hashes inventory changed; missing={missing}, extra={extra}")
    for rel, expected in hashes.items():
        path = root / rel
        if not path.exists():
            raise FileNotFoundError(f"hashed artifact is missing: {rel}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"sha256 mismatch for {rel}: {actual_hash} != {expected.get('sha256')}")
        expected_bytes = int(expected.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"byte-size mismatch for {rel}")
        if path.suffix.lower() == ".csv" and "rows_including_header" in expected:
            row_count = _csv_row_count_including_header(path)
            if row_count != int(expected["rows_including_header"]):
                raise ValueError(f"CSV row-count mismatch for {rel}")


def _verify_manifest_package_relative_paths(root: Path, manifest: Mapping[str, Any]) -> None:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise ValueError("manifest.outputs must be populated")
    if dict(outputs) != APPROVED_OUTPUT_PATHS:
        raise ValueError("manifest.outputs must exactly match approved output keys and package paths")
    for key, value in APPROVED_OUTPUT_PATHS.items():
        _verify_package_relative_file(root, value, f"outputs.{key}")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest.artifacts must be populated")
    if dict(artifacts) != APPROVED_ARTIFACT_PATHS:
        raise ValueError("manifest.artifacts must exactly match approved package paths")
    for key in ("forecast_inputs_dir", "manifest"):
        value = APPROVED_ARTIFACT_PATHS[key]
        _verify_safe_package_relative_path(value, f"artifacts.{key}")
        if not (root / value).exists():
            raise FileNotFoundError(f"manifest path does not exist for artifacts.{key}: {value}")
    if artifacts.get("output_dir") != ".":
        raise ValueError("artifacts.output_dir must be package root '.'")
    run_summaries = manifest.get("run_summaries", {})
    if isinstance(run_summaries, Mapping):
        for run_id, summary in run_summaries.items():
            if not isinstance(summary, Mapping):
                continue
            for key in ("maturity_ledger_file", "cash_residual_file", "operating_cash_path_file"):
                value = summary.get(key)
                if isinstance(value, str):
                    _verify_package_relative_file(root, value, f"run_summaries.{run_id}.{key}")


def _verify_package_relative_file(root: Path, rel: str, label: str) -> None:
    _verify_safe_package_relative_path(rel, label)
    if not (root / rel).exists():
        raise FileNotFoundError(f"manifest path does not exist for {label}: {rel}")


def _verify_safe_package_relative_path(rel: str, label: str) -> None:
    path = Path(rel)
    if path.is_absolute() or any(part in ("", "..") for part in path.parts):
        raise ValueError(f"{label} must be package-relative: {rel}")


def _verify_source_package(root: Path, manifest: Mapping[str, Any]) -> None:
    source_package = manifest.get("source_package")
    if not isinstance(source_package, Mapping):
        raise ValueError("manifest source_package must be populated")
    unknown_source_package = sorted(set(str(key) for key in source_package) - APPROVED_SOURCE_PACKAGE_KEYS)
    if unknown_source_package:
        raise ValueError(f"source_package contains unknown fields: {unknown_source_package}")
    if source_package.get("sources_dir") != "sources":
        raise ValueError("source_package.sources_dir must be package-relative 'sources'")
    if source_package.get("source_manifest") != "sources/source_manifest.json":
        raise ValueError("source_package.source_manifest must be package-relative")
    source_files = source_package.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("source_package.source_files must be populated")
    source_file_keys = {str(rel) for rel in source_files}
    if source_file_keys != EXPECTED_SOURCE_PACKAGE_FILES:
        raise ValueError("source_package.source_files must exactly match approved source package files")
    if int(source_package.get("source_file_count", -1)) != len(source_files):
        raise ValueError("source_package.source_file_count does not match source_files")

    sources_dir = root / "sources"
    for rel, expected in source_files.items():
        _verify_safe_relative_source_path(str(rel))
        _verify_source_file_entry_schema(str(rel), expected)
        path = sources_dir / str(rel)
        if not path.exists():
            raise FileNotFoundError(f"packaged source file is missing: sources/{rel}")
        actual_hash = _sha256_file(path)
        if actual_hash != expected.get("sha256"):
            raise ValueError(f"source sha256 mismatch for sources/{rel}: {actual_hash} != {expected.get('sha256')}")
        expected_bytes = int(expected.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"source byte-size mismatch for sources/{rel}")
        frozen_metadata = _frozen_source_file_metadata(str(rel), expected)
        if frozen_metadata is not None:
            frozen_bytes = frozen_metadata.get("bytes")
            if actual_hash != frozen_metadata["sha256"] or (
                frozen_bytes is not None and expected_bytes != int(frozen_bytes)
            ):
                raise ValueError(f"frozen source file identity changed for sources/{rel}")

    source_manifest = _read_json(root / "sources" / "source_manifest.json")
    if source_manifest.get("schema_version") != "tdcsim_cbo_source_package_manifest_v1":
        raise ValueError("sources/source_manifest.json has unexpected schema_version")
    if source_manifest.get("status") != "package_relative_sources_for_cbo_forecast_smoke":
        raise ValueError("sources/source_manifest.json has unexpected status")
    unknown_source_manifest = sorted(set(str(key) for key in source_manifest) - APPROVED_SOURCE_MANIFEST_KEYS)
    if unknown_source_manifest:
        raise ValueError(f"sources/source_manifest.json contains unknown fields: {unknown_source_manifest}")
    if source_manifest.get("inventory_status") != "complete_required_inventory_no_silent_skips":
        raise ValueError("sources/source_manifest.json inventory_status changed")
    manifest_files = source_manifest.get("files")
    if not isinstance(manifest_files, Mapping) or not manifest_files:
        raise ValueError("sources/source_manifest.json files must be populated")
    nested_file_keys = {str(rel) for rel in manifest_files}
    expected_nested_keys = EXPECTED_SOURCE_PACKAGE_FILES - {"source_manifest.json"}
    if nested_file_keys != expected_nested_keys:
        raise ValueError("sources/source_manifest.json files must exactly match source_package.source_files excluding source_manifest.json")
    for rel, expected in manifest_files.items():
        package_entry = source_files.get(rel)
        if not isinstance(package_entry, Mapping):
            raise ValueError(f"sources/source_manifest.json references file missing from source_package: {rel}")
        if package_entry.get("sha256") != expected.get("sha256"):
            raise ValueError(f"source manifest sha256 disagrees with source_package for {rel}")
        if int(package_entry.get("bytes", -1)) != int(expected.get("bytes", -1)):
            raise ValueError(f"source manifest byte-size disagrees with source_package for {rel}")
    source_manifest_entry = source_files.get("source_manifest.json")
    if not isinstance(source_manifest_entry, Mapping):
        raise ValueError("source_package.source_files missing source_manifest.json")
    source_manifest_path = root / "sources" / "source_manifest.json"
    if source_manifest_entry.get("sha256") != _sha256_file(source_manifest_path):
        raise ValueError("source_package.source_files source_manifest.json sha256 mismatch")
    if int(source_manifest_entry.get("bytes", -1)) != source_manifest_path.stat().st_size:
        raise ValueError("source_package.source_files source_manifest.json byte-size mismatch")
    if source_package.get("source_manifest_sha256") != _sha256_file(source_manifest_path):
        raise ValueError("source_package.source_manifest_sha256 mismatch")


def _verify_source_file_entry_schema(rel: str, entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError(f"source_package.source_files.{rel} must be a mapping")
    unknown = sorted(set(str(key) for key in entry) - APPROVED_SOURCE_FILE_ENTRY_KEYS)
    if unknown:
        raise ValueError(f"source_package.source_files.{rel} contains unknown fields: {unknown}")
    if entry.get("source_path") != rel:
        raise ValueError(f"source_package.source_files.{rel}.source_path must equal {rel}")
    if "origin_repo_path" in entry and "origin_source_label" in entry:
        raise ValueError(f"source_package.source_files.{rel} must not mix origin_repo_path and origin_source_label")
    if "sha256" not in entry or not isinstance(entry.get("sha256"), str):
        raise ValueError(f"source_package.source_files.{rel}.sha256 is missing")
    if "bytes" not in entry:
        raise ValueError(f"source_package.source_files.{rel}.bytes is missing")


def _frozen_source_file_metadata(rel: str, entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    official_metadata = FROZEN_SOURCE_FILE_METADATA.get(rel)
    return official_metadata


def _verify_source_selectors_and_frozen_rows(root: Path, manifest: Mapping[str, Any]) -> None:
    selectors = manifest.get("source_row_selectors")
    if not isinstance(selectors, Mapping):
        raise ValueError("manifest source_row_selectors must be populated")
    for key, expected in EXPECTED_SOURCE_ROW_SELECTORS.items():
        if selectors.get(key) != expected:
            raise ValueError(f"source_row_selectors.{key} expected {expected}, found {selectors.get(key)}")
    mspd_selectors = selectors.get("mspd")
    if not isinstance(mspd_selectors, list) or set(mspd_selectors) != EXPECTED_MSPD_SELECTORS:
        raise ValueError("source_row_selectors.mspd changed")

    _verify_single_source_extract(
        root / "sources" / "fiscaldata" / "mts_table_1_deficit_2026-05-31.csv",
        EXPECTED_SOURCE_EXTRACT_ROWS["mts_table_1_deficit_row"],
        money_fields={"current_month_dfct_sur_amt": 1_246_203_266_386.93},
    )
    _verify_single_source_extract(
        root / "sources" / "fiscaldata" / "mts_table_9_net_interest_2026-05-31.csv",
        EXPECTED_SOURCE_EXTRACT_ROWS["mts_table_9_net_interest_row"],
        money_fields={"current_fytd_rcpt_outly_amt": 722_706_511_243.20},
    )
    _verify_single_source_extract(
        root / "sources" / "fiscaldata" / "dts_tga_closing_balance_2026-06-16.csv",
        EXPECTED_SOURCE_EXTRACT_ROWS["dts_tga_closing_balance_row"],
        money_fields={"open_today_bal": 981_113.0},
    )
    _verify_single_source_extract(
        root / "sources" / "treasury" / "debt_to_penny_public_debt_2026-05-29.csv",
        EXPECTED_SOURCE_EXTRACT_ROWS["debt_to_penny_row"],
        money_fields={"debt_held_public_amt": 31_515_369_798_622.98},
    )

    mspd_rows = _read_csv_dicts(root / "sources" / "fiscaldata" / "mspd_table_1_debt_bridge_2026-05-31.csv")
    actual_mspd_rows = {
        (
            str(row.get("record_date")),
            str(row.get("security_type_desc")),
            round(_csv_float(row, "debt_held_public_mil_amt"), 8),
        )
        for row in mspd_rows
    }
    expected_mspd_rows = {
        (date_value, desc, round(float(amount), 8))
        for date_value, desc, amount in EXPECTED_FROZEN_MSPD_ROWS
    }
    if actual_mspd_rows != expected_mspd_rows:
        raise ValueError("packaged MSPD debt bridge rows changed")

    mspd_table_3 = _read_csv_dicts(root / "sources" / "fiscaldata" / "mspd_table_3_market_2026-05-31.csv")
    if len(mspd_table_3) < 800:
        raise ValueError("packaged MSPD Table 3 opening-date extract is unexpectedly small")
    if {row.get("record_date") for row in mspd_table_3} != {"2026-05-31"}:
        raise ValueError("packaged MSPD Table 3 extract must contain only 2026-05-31 rows")

    base_curve = _read_json(root / "sources" / "treasury" / "base_curve_2026-06-16.json")
    rows = base_curve.get("rows") if isinstance(base_curve, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 11:
        raise ValueError("packaged Treasury base curve extract must contain 11 tenor rows")
    if {str(row.get("base_curve_date", row.get("curve_date", ""))) for row in rows} != {"2026-06-16"}:
        raise ValueError("packaged Treasury base curve rows must all be dated 2026-06-16")
    actual_curve_rows = {
        (
            round(float(row.get("tenor_years")), 12),
            round(float(row.get("nominal_rate")), 12),
        )
        for row in rows
    }
    if actual_curve_rows != EXPECTED_TREASURY_BASE_CURVE_ROWS:
        raise ValueError("packaged Treasury base curve rows changed")


def _verify_cbo_source_fixtures_against_workbooks(root: Path) -> None:
    fixture_rows = _read_csv_dicts(root / "forecast_inputs" / "source_fixtures.csv")
    if not fixture_rows:
        raise ValueError("source_fixtures.csv has no data rows")
    budget_contract = parse_cbo_budget_source_contract(
        root / "sources" / "cbo" / "51118-2026-02-Budget-Projections.xlsx",
        verify_hash=True,
    )
    economic_contract = parse_cbo_economic_quarterly_source_contract(
        root / "sources" / "cbo" / "51135-2026-02-Economic-Projections.xlsx",
        verify_hash=True,
    )
    expected_rows = budget_contract.fixture_rows() + economic_contract.fixture_rows()
    _verify_csv_rows_match_expected(
        "tdcsim_cbo_fiscal_baseline.csv",
        _read_csv_dicts(root / "forecast_inputs" / "tdcsim_cbo_fiscal_baseline.csv"),
        build_cbo_fiscal_baseline_rows(budget_contract, scenario_id="cbo_full_horizon_local_smoke_3m"),
    )
    _verify_csv_rows_match_expected(
        "tdcsim_macro_forecast_path.csv",
        _read_csv_dicts(root / "forecast_inputs" / "tdcsim_macro_forecast_path.csv"),
        build_cbo_macro_forecast_path_rows(economic_contract, scenario_id="cbo_full_horizon_local_smoke_3m"),
    )
    expected_by_key = {_source_fixture_key(row): row for row in expected_rows}
    actual_by_key = {_source_fixture_key(row): row for row in fixture_rows}
    if set(actual_by_key) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(actual_by_key))
        extra = sorted(set(actual_by_key) - set(expected_by_key))
        raise ValueError(f"CBO source fixture keys changed; missing={missing[:5]}, extra={extra[:5]}")
    exact_fields = {
        "schema_version",
        "forecast_name",
        "source_family",
        "source_file",
        "source_sha256",
        "source_url",
        "source_sheet",
        "source_row_number",
        "source_row_selector",
        "source_unit_block",
        "source_year_or_period",
        "raw_sign_convention",
        "canonical_transform",
        "observation_date",
        "available_date",
        "source_status",
    }
    for key, expected in expected_by_key.items():
        actual = actual_by_key[key]
        for field in exact_fields:
            if str(actual.get(field)) != str(expected.get(field)):
                raise ValueError(
                    f"CBO source fixture {key} {field} expected {expected.get(field)!r}, found {actual.get(field)!r}"
                )
        _assert_close(
            f"CBO source fixture {key} raw_value",
            _csv_float(actual, "raw_value"),
            float(expected["raw_value"]),
            STRICT_TOLERANCE,
        )
        _assert_close(
            f"CBO source fixture {key} canonical_value",
            _csv_float(actual, "canonical_value"),
            float(expected["canonical_value"]),
            STRICT_TOLERANCE,
        )


def _source_fixture_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_family")),
        str(row.get("source_sheet")),
        str(row.get("source_row_number")),
        str(row.get("source_row_selector")),
        str(row.get("source_year_or_period")),
    )


def _verify_csv_rows_match_expected(
    label: str,
    actual_rows: list[dict[str, str]],
    expected_rows: list[Mapping[str, Any]],
) -> None:
    if len(actual_rows) != len(expected_rows):
        raise ValueError(f"{label} row count changed: expected {len(expected_rows)}, found {len(actual_rows)}")
    for row_index, (actual, expected) in enumerate(zip(actual_rows, expected_rows), start=1):
        actual_keys = set(actual)
        expected_keys = {str(key) for key in expected}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"{label} columns changed at row {row_index}; missing={missing}, extra={extra}")
        for key, expected_value in expected.items():
            actual_value = actual[str(key)]
            if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
                _assert_close(
                    f"{label} row {row_index} {key}",
                    float(actual_value),
                    float(expected_value),
                    STRICT_TOLERANCE,
                )
            elif str(actual_value) != str(expected_value):
                raise ValueError(
                    f"{label} row {row_index} {key} expected {expected_value!r}, found {actual_value!r}"
                )


def _verify_single_source_extract(
    path: Path,
    expected_fields: Mapping[str, str],
    *,
    money_fields: Mapping[str, float],
) -> None:
    rows = _read_csv_dicts(path)
    if len(rows) != 1:
        raise ValueError(f"{path.name} must contain exactly one source row")
    row = rows[0]
    for field, expected_value in expected_fields.items():
        if str(row.get(field)) != expected_value:
            raise ValueError(f"{path.name} {field} expected {expected_value}, found {row.get(field)}")
    for field, expected_value in money_fields.items():
        _assert_close(f"{path.name} {field}", _csv_float(row, field), expected_value, 0.005)


def _verify_safe_relative_source_path(rel: str) -> None:
    path = Path(rel)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe source package relative path: {rel}")


def _reject_local_paths(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in LOCAL_PATH_PATTERNS:
            match = pattern.search(text)
            if match:
                rel = path.relative_to(root)
                raise ValueError(f"machine-local path leaked into {rel}: {match.group(0)}")


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"required CSV file is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader)


def _csv_row_count_including_header(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle))


def _require_columns(label: str, row: Mapping[str, str], columns: set[str]) -> None:
    missing = sorted(column for column in columns if column not in row)
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _float(row: Mapping[str, str], column: str) -> float:
    return _csv_float(row, column)


def _csv_float(row: Mapping[str, Any], column: str) -> float:
    value = row.get(column)
    if value in (None, ""):
        raise ValueError(f"missing numeric value for {column}")
    return float(value)


def _manifest_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"missing manifest numeric value for {key}")
    return float(value)


def _is_blank_or_nan(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    try:
        return math.isnan(float(text))
    except ValueError:
        return False


def _sum(rows: list[dict[str, str]], column: str) -> float:
    return sum(_float(row, column) for row in rows)


def _max_abs_delta(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]], column: str) -> float:
    return max(abs(_float(left, column) - _float(right, column)) for left, right in zip(left_rows, right_rows))


def _opening_debt_base(row: Mapping[str, str]) -> float:
    if row.get("SecurityType") == "TIPS":
        return _csv_float(row, "AdjustedPrincipal")
    return _csv_float(row, "FaceValue")


def _fiscal_year(iso_date: str) -> int:
    year = int(iso_date[:4])
    month = int(iso_date[5:7])
    return year + int(month >= 10)


def _assert_close(label: str, actual: float, expected: float, tolerance: float) -> None:
    if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _extract_zip_safely(archive: zipfile.ZipFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        try:
            destination.relative_to(target_resolved)
        except ValueError as exc:
            raise ValueError(f"zip member escapes extraction root: {member.filename}") from exc
        if member.is_dir():
            continue
        if Path(member.filename).is_absolute():
            raise ValueError(f"zip member escapes extraction root: {member.filename}")
    archive.extractall(target)


def _extracted_root(tmp: Path) -> Path:
    if (tmp / "manifest.json").exists():
        return tmp
    children = [child for child in tmp.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").exists():
        return children[0]
    raise FileNotFoundError("extracted zip does not contain manifest.json at root or a single top-level directory")


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="CBO evidence directory or zip archive to verify.")
    parser.add_argument("--write-zip", help="Create a zip from PACKAGE and verify a fresh extraction of that zip.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = verify_package(args.package)
    if args.write_zip:
        zip_path = create_package_zip(args.package, args.write_zip)
        zip_summary = verify_package(zip_path)
        summary["created_zip"] = str(zip_path)
        summary["fresh_unpacked_zip_status"] = zip_summary["status"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
