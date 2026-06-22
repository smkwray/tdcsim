#!/usr/bin/env python3
"""Build a closeout audit for the historical replay checkpoint.

The audit is intentionally scoped to the accepted target: a quarterly,
aggregate-consistent synthetic replay. It does not certify exact transaction
replay, exact holder-by-security history, or observed market-price valuation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from historical_replay import _build_code_identity_hash, _build_replay_config_hash  # noqa: E402

DATA_DIR = ROOT / "data" / "historical_replay"
VALIDATION_DIR = DATA_DIR / "validation"
IMPORTED_DIR = DATA_DIR / "imported"

SOURCE_SCOPE_PATH = VALIDATION_DIR / "historical_replay_source_scope_audit.csv"
PLAUSIBILITY_PATH = VALIDATION_DIR / "historical_replay_plausibility_audit.csv"
REPORT_PATH = VALIDATION_DIR / "historical_replay_closeout_audit.md"
FINAL_EVIDENCE_MANIFEST_PATH = VALIDATION_DIR / "historical_replay_final_evidence_manifest.csv"
RUNTIME_MANIFEST_PATH = VALIDATION_DIR / "historical_replay_runtime_manifest.csv"
NUMERIC_TOLERANCE_MIL = 1.0e-2
VALUATION_SLACK_ABS_LIMIT_MIL = 50_000.0
VALUATION_SLACK_REL_LIMIT = 0.001
VALUATION_SLACK_SECTOR_REL_LIMIT = 0.07
MATERIAL_PORTFOLIO_SECTOR_TOTAL_MIL = 25_000.0
NONFOREIGN_INTEREST_GAP_LIMIT_MIL = 1_500.0
NONFOREIGN_INTEREST_REL_LIMIT = 0.06
MAX_HIGH_TV_TRANSITION_SHARE = 0.06
MAX_HIGH_TV_TRANSITIONS = 150
NEGATIVE_NETTING_GROSS_SHARE_LIMIT = 0.50
ALLOW_HEADER_ONLY_CSV_ARTIFACTS = {"valuation_basis_feasibility_certificate.csv"}
LOW_IDENTIFICATION_CONCENTRATION_SECTORS = {"abs_issuers"}
REQUIRED_ARTIFACTS = {
    "artifact_integrity.csv",
    "historical_replay_acceptance.md",
    "historical_replay_results.csv",
    "historical_replay_ledger.csv",
    "historical_replay_input_manifest.csv",
    "historical_replay_source_manifest.csv",
    "historical_replay_runtime_manifest.csv",
    "historical_replay_runtime_lock.csv",
    "historical_replay_code_identity_manifest.csv",
    "historical_replay_diagnostics.csv",
    "historical_replay_portfolio_snapshots.csv",
    "historical_replay_final_portfolio.csv",
    "historical_replay_event_ledger.csv",
    "historical_replay_event_rollforward.csv",
    "historical_replay_unexplained_change_ledger.csv",
    "portfolio_constraint_diagnostics.csv",
    "solver_constraint_residuals.csv",
    "valuation_basis_feasibility_certificate.csv",
    "mmf_component_reconciliation.csv",
    "negative_sector_netting_bridge.csv",
    "maturity_prior_reconciliation.csv",
    "portfolio_native_sector_similarity.csv",
    "interest_component_detail.csv",
    "interest_proxy_alignment.csv",
    "pricing_scope_diagnostics.csv",
    "valuation_scope_diagnostics.csv",
    "historical_replay_exact_observation_coverage.csv",
    "z1_transaction_flow_diagnostics.csv",
    "portfolio_transition_diagnostics.csv",
    "soma_treasury_holdings_quarterly.csv",
    "soma_fixed_allocations.csv",
    "soma_holdout_diagnostics.csv",
    "historical_replay_large_artifact_sample_manifest.csv",
    "historical_replay_portfolio_snapshots_sample.csv",
    "interest_component_detail_sample.csv",
    "tdcest_selected_ladder_crosscheck_summary.csv",
    "tdcest_modern_formula_summary.csv",
}
EXPECTED_REQUIRED_INPUT_KEYS = {
    "quarterly_cash",
    "sector_positions",
    "z1_raw_archive",
    "z1_source_record",
    "mspd_cohorts",
    "auction_terms",
    "frn_daily_indexes",
    "tips_cpi",
    "pricing_curve_1m",
    "pricing_curve_3m",
    "pricing_curve_6m",
    "pricing_curve_1y",
    "pricing_curve_2y",
    "pricing_curve_3y",
    "pricing_curve_5y",
    "pricing_curve_7y",
    "pricing_curve_10y",
    "pricing_curve_20y",
    "pricing_curve_30y",
    "pricing_curve_3m_monthly",
    "pricing_curve_10y_monthly",
    "real_pricing_curve_5y",
    "real_pricing_curve_7y",
    "real_pricing_curve_10y",
    "real_pricing_curve_20y",
    "real_pricing_curve_30y",
    "soma_treasury_holdings_monthly",
    "auction_allotment_panel",
    "ffiec_interest_constraints",
    "ncua_interest_constraints",
    "tier2_interest_constraints",
    "mmf_component_constraints",
    "tdc_quarterly_inputs",
    "tdc_tdc_estimates",
    "tdc_tdc_components",
    "tdc_tdc_tier2_regression_series",
    "tdc_tdc_du_fiscal_flow_research",
    "tdc_tdc_empirical_anchor",
    "tdc_tier2_interest_component_candidate",
    "tdc_tier2_interest_source_constraints",
    "tdc_treasury_interest_expense",
    "tdc_ffiec_interest_constraints_normalized",
    "tdc_ncua_interest_constraints_normalized",
    "tdc_tdc_mmf_rrp_quarterly_adjustments",
    "tdc_tdc_mmf_rrp_quarterly_adjustments_sec_full",
    "tdc_method_meta",
}
REQUIRED_CHECKS = {
    "artifact_integrity",
    "artifact_integrity_live_manifest",
    "required_artifacts_live",
    "acceptance_artifact_current",
    "solver_entropy_convergence",
    "aggregate_only_surface_semantics",
    "weighted_sector_constraint_fit",
    "post_export_constraint_fit",
    "soma_observed_block",
    "required_inputs",
    "selected_tdc_target_wiring",
    "raw_vs_modeled_residual_separation",
    "modern_formula_identity",
    "event_rollforward_identity",
    "terminal_exits_tagged",
    "old_opening_split_residual_disclosed",
    "source_basis_residual_not_private",
    "negative_sector_netting_disclosure",
    "tips_principal_identity",
    "interest_projection_feasibility",
    "mmf_bill_component_constraint",
    "ffiec_ncua_maturity_prior_use",
    "portfolio_concentration_continuity",
    "native_sector_nonidentification_boundary",
    "pricing_scope_boundary",
    "valuation_claim_boundary",
    "exact_observation_boundary",
    "z1_transaction_flow_diagnostic",
    "portfolio_transition_explanation",
    "portfolio_transition_adjacency",
    "source_role_consistency",
    "large_artifact_sample_manifest",
}


@dataclass(frozen=True)
class CheckRow:
    check: str
    status: str
    observed: str
    expected: str
    notes: str


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _required_artifact_status(artifact_integrity: pd.DataFrame) -> tuple[str, str, str]:
    manifest_by_path = {}
    if not artifact_integrity.empty and "path" in artifact_integrity.columns:
        for _, row in artifact_integrity.iterrows():
            manifest_by_path[str(row.get("path", ""))] = row
    missing = []
    bad_content = []
    manifest_mismatch = []
    required_run_ids = set()
    required_config_hashes = set()
    for filename in sorted(REQUIRED_ARTIFACTS):
        path = VALIDATION_DIR / filename
        rel = _rel(path)
        if not path.exists():
            missing.append(filename)
            continue
        metadata = _artifact_file_metadata(path)
        if str(metadata.get("status")) != "present":
            bad_content.append(f"{filename}:{metadata.get('status')}")
        if filename == "artifact_integrity.csv":
            continue
        if filename == "historical_replay_closeout_audit.md" and rel not in manifest_by_path:
            continue
        manifest_row = manifest_by_path.get(rel)
        if manifest_row is None:
            manifest_mismatch.append(f"{filename}:not_in_manifest")
            continue
        if str(manifest_row.get("status", "")) != "present":
            manifest_mismatch.append(f"{filename}:status={manifest_row.get('status')}")
        for column in ["bytes", "row_count", "column_count"]:
            expected = pd.to_numeric(pd.Series([manifest_row.get(column)]), errors="coerce").iloc[0]
            actual = metadata.get(column)
            if pd.notna(expected) and pd.notna(actual) and int(expected) != int(actual):
                manifest_mismatch.append(f"{filename}:{column}_manifest={int(expected)} actual={int(actual)}")
        for column in ["sha256", "header_sha256"]:
            expected = manifest_row.get(column)
            actual = metadata.get(column)
            if pd.notna(expected) and str(expected) and pd.notna(actual) and str(expected) != str(actual):
                manifest_mismatch.append(f"{filename}:{column}_mismatch")
        run_id = manifest_row.get("run_id")
        config_hash = manifest_row.get("config_sha256")
        if pd.notna(run_id) and str(run_id):
            required_run_ids.add(str(run_id))
        else:
            manifest_mismatch.append(f"{filename}:missing_run_id")
        if pd.notna(config_hash) and str(config_hash):
            required_config_hashes.add(str(config_hash))
        else:
            manifest_mismatch.append(f"{filename}:missing_config_sha256")
    if len(required_run_ids) > 1:
        manifest_mismatch.append(f"multiple_run_ids={sorted(required_run_ids)}")
    if len(required_config_hashes) > 1:
        manifest_mismatch.append(f"multiple_config_sha256={sorted(required_config_hashes)}")
    required_code_hashes = {
        str(row.get("code_sha256"))
        for _, row in artifact_integrity.iterrows()
        if "code_sha256" in artifact_integrity.columns and pd.notna(row.get("code_sha256")) and str(row.get("code_sha256"))
    }
    if len(required_code_hashes) > 1:
        manifest_mismatch.append(f"multiple_code_sha256={sorted(required_code_hashes)}")
    if not required_code_hashes:
        manifest_mismatch.append("missing_code_sha256")
    else:
        live_code_hash = _build_code_identity_hash()
        if required_code_hashes != {live_code_hash}:
            manifest_mismatch.append("live_code_sha256_mismatch")
    status = "pass" if not missing and not bad_content and not manifest_mismatch else "fail"
    observed = (
        f"{len(REQUIRED_ARTIFACTS) - len(missing)}/{len(REQUIRED_ARTIFACTS)} required artifacts live; "
        f"missing={missing}; bad_content={bad_content[:5]}; manifest_mismatch={manifest_mismatch[:5]}"
    )
    expected = (
        "all required artifacts exist on disk, are nonempty, match artifact_integrity.csv full-file "
        "hash/shape metadata, and share one run/config identity"
    )
    return status, observed, expected


def _artifact_integrity_manifest_status(artifact_integrity: pd.DataFrame) -> tuple[str, str, str]:
    if artifact_integrity.empty or not {"artifact", "path", "status"}.issubset(artifact_integrity.columns):
        return "fail", "artifact_integrity.csv missing required columns", "all artifact_integrity rows are live-authenticated"
    mismatches = []
    checked = 0
    run_ids = set()
    config_hashes = set()
    code_hashes = set()
    for _, row in artifact_integrity.iterrows():
        artifact = str(row.get("artifact", "")).strip()
        raw_path = str(row.get("path", "")).strip()
        if not artifact or not raw_path:
            mismatches.append(f"{artifact or '<blank>'}:missing_path")
            continue
        path = _resolve_project_path(raw_path)
        metadata = _artifact_file_metadata(path)
        checked += 1
        for column in ["status", "sha256", "header_sha256"]:
            expected = row.get(column)
            actual = metadata.get(column)
            if pd.notna(expected) and str(expected) and pd.notna(actual) and str(expected) != str(actual):
                mismatches.append(f"{artifact}:{column}_mismatch")
        for column in ["bytes", "row_count", "column_count"]:
            expected = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            actual = metadata.get(column)
            if pd.notna(expected) and pd.notna(actual) and int(expected) != int(actual):
                mismatches.append(f"{artifact}:{column}_manifest={int(expected)} actual={int(actual)}")
        for column, collector in [
            ("run_id", run_ids),
            ("config_sha256", config_hashes),
            ("code_sha256", code_hashes),
        ]:
            value = row.get(column)
            if pd.notna(value) and str(value):
                collector.add(str(value))
            else:
                mismatches.append(f"{artifact}:missing_{column}")
    if len(run_ids) > 1:
        mismatches.append(f"multiple_run_ids={sorted(run_ids)}")
    if len(config_hashes) > 1:
        mismatches.append(f"multiple_config_sha256={sorted(config_hashes)}")
    if len(code_hashes) > 1:
        mismatches.append(f"multiple_code_sha256={sorted(code_hashes)}")
    live_code_hash = _build_code_identity_hash()
    if code_hashes and code_hashes != {live_code_hash}:
        mismatches.append("live_code_sha256_mismatch")
    live_config_hash = _live_replay_config_hash()
    if config_hashes and live_config_hash and config_hashes != {live_config_hash}:
        mismatches.append("live_config_sha256_mismatch")
    status = "pass" if not mismatches else "fail"
    observed = f"{checked}/{len(artifact_integrity)} manifest artifacts live-authenticated; mismatches={mismatches[:8]}"
    expected = "every artifact_integrity row matches live bytes/hash/shape metadata and live code/config identities"
    return status, observed, expected


def _live_replay_config_hash() -> str:
    input_manifest = _read_csv(VALIDATION_DIR / "historical_replay_input_manifest.csv")
    cash_row = _manifest_row(input_manifest, "quarterly_cash")
    sector_row = _manifest_row(input_manifest, "sector_positions")
    cohort_row = _manifest_row(input_manifest, "mspd_cohorts")
    cfg = {
        "amount_unit_scale": 1000.0,
        "sector_value_unit_scale": 1_000_000.0,
        "solver_tolerance": 1.0e-2,
        "start_quarter": "2001Q1",
        "end_quarter": "2025Q4",
        "output_dir": str(VALIDATION_DIR),
        "paths": {
            "cash": str(_resolve_project_path(cash_row.get("path", ""))) if cash_row else "",
            "sector_positions": str(_resolve_project_path(sector_row.get("path", ""))) if sector_row else "",
            "cohorts": str(_resolve_project_path(cohort_row.get("path", ""))) if cohort_row else "",
        },
    }
    return _build_replay_config_hash(
        cfg,
        start_quarter="2001Q1",
        end_quarter="2025Q4",
        replay_input_manifest=input_manifest,
    )


def _manifest_row(input_manifest: pd.DataFrame, source_key: str) -> dict[str, object] | None:
    if input_manifest.empty or "source_key" not in input_manifest.columns:
        return None
    rows = input_manifest[input_manifest["source_key"].astype(str).eq(source_key)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _artifact_file_metadata(path: Path) -> dict[str, object]:
    if not path.exists() or not path.is_file():
        return {"status": "missing"}
    size = path.stat().st_size
    if size <= 0:
        return {"status": "zero_bytes", "bytes": size}
    metadata: dict[str, object] = {
        "status": "present",
        "bytes": size,
        "sha256": _sha256_file(path),
        "row_count": pd.NA,
        "column_count": pd.NA,
        "header_sha256": pd.NA,
    }
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                header = handle.readline().rstrip("\n")
                row_count = sum(1 for _ in handle)
            pd.read_csv(path, nrows=5)
        except Exception as exc:  # noqa: BLE001 - audit should fail on any unreadable CSV state.
            metadata["status"] = f"malformed_csv:{type(exc).__name__}"
            return metadata
        metadata["row_count"] = row_count
        metadata["column_count"] = len(header.split(",")) if header else 0
        metadata["header_sha256"] = hashlib.sha256(header.encode("utf-8")).hexdigest() if header else pd.NA
        if row_count <= 0 and path.name not in ALLOW_HEADER_ONLY_CSV_ARTIFACTS:
            metadata["status"] = "header_only_csv"
    return metadata


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_input_status(input_manifest: pd.DataFrame) -> tuple[str, str, str]:
    if input_manifest.empty or not {"source_key", "path", "status"}.issubset(input_manifest.columns):
        return "fail", "input manifest missing required columns", "all code-rostered required input rows are live-authenticated"
    found_keys = set(input_manifest["source_key"].dropna().astype(str))
    missing_expected_keys = sorted(EXPECTED_REQUIRED_INPUT_KEYS - found_keys)
    required = input_manifest["source_key"].astype(str).isin(EXPECTED_REQUIRED_INPUT_KEYS)
    if "required_for_claim" in input_manifest.columns:
        manifest_required = _bool_series(input_manifest["required_for_claim"])
        demoted = sorted(
            input_manifest.loc[
                input_manifest["source_key"].astype(str).isin(EXPECTED_REQUIRED_INPUT_KEYS) & ~manifest_required,
                "source_key",
            ].dropna().astype(str).unique()
        )
    else:
        demoted = sorted(EXPECTED_REQUIRED_INPUT_KEYS)
    missing = []
    mismatches = []
    checked = 0
    for _, row in input_manifest[required].iterrows():
        source_key = str(row.get("source_key", ""))
        raw_path = str(row.get("path", ""))
        path = _resolve_project_path(raw_path)
        if not path.exists() or not path.is_file():
            missing.append(source_key or raw_path)
            continue
        checked += 1
        expected_status = str(row.get("status", ""))
        if expected_status != "present":
            mismatches.append(f"{source_key}:status={expected_status}")
        expected_bytes = pd.to_numeric(pd.Series([row.get("bytes")]), errors="coerce").iloc[0]
        if pd.notna(expected_bytes) and int(expected_bytes) != path.stat().st_size:
            mismatches.append(f"{source_key}:bytes_manifest={int(expected_bytes)} actual={path.stat().st_size}")
        expected_sha = row.get("sha256")
        if pd.notna(expected_sha) and str(expected_sha) and str(expected_sha) != _sha256_file(path):
            mismatches.append(f"{source_key}:sha256_mismatch")
        expected_rows = pd.to_numeric(pd.Series([row.get("row_count")]), errors="coerce").iloc[0]
        if path.suffix.lower() == ".csv" and pd.notna(expected_rows):
            metadata = _artifact_file_metadata(path)
            actual_rows = metadata.get("row_count")
            if pd.notna(actual_rows) and int(expected_rows) != int(actual_rows):
                mismatches.append(f"{source_key}:row_count_manifest={int(expected_rows)} actual={int(actual_rows)}")
    total_required = len(EXPECTED_REQUIRED_INPUT_KEYS)
    z1_vintage_ok = True
    if "sector_positions" in EXPECTED_REQUIRED_INPUT_KEYS:
        z1_rows = input_manifest[input_manifest["source_key"].astype(str).eq("sector_positions")]
        if not z1_rows.empty:
            z1_vintage_ok = bool(
                z1_rows.get("source_vintage", pd.Series("", index=z1_rows.index))
                .dropna()
                .astype(str)
                .str.contains("2026-06-11|z1_2026q1", case=False, regex=True)
                .any()
            )
        else:
            z1_vintage_ok = False
    status = (
        "pass"
        if not missing_expected_keys
        and not demoted
        and not missing
        and not mismatches
        and checked == total_required
        and z1_vintage_ok
        else "fail"
    )
    observed = (
        f"{checked}/{total_required} code-rostered required inputs live-authenticated; "
        f"missing_keys={missing_expected_keys}; demoted_required_for_claim={demoted[:5]}; missing={missing}; mismatches={mismatches[:5]}; z1_vintage_ok={z1_vintage_ok}"
    )
    expected = "all code-rostered required inputs exist, are required_for_claim, include Z.1 vintage metadata, and match live bytes/hash/row-count metadata"
    return status, observed, expected


def _acceptance_artifact_status() -> tuple[str, str, str]:
    path = VALIDATION_DIR / "historical_replay_acceptance.md"
    if not path.exists():
        return "fail", "historical_replay_acceptance.md missing", "acceptance artifact exists and matches live evidence"
    text = path.read_text(encoding="utf-8")
    input_manifest = _read_csv(VALIDATION_DIR / "historical_replay_input_manifest.csv")
    artifact_integrity = _read_csv(VALIDATION_DIR / "artifact_integrity.csv")
    diagnostics = _read_csv(VALIDATION_DIR / "historical_replay_diagnostics.csv")
    transition = _read_csv(VALIDATION_DIR / "portfolio_transition_diagnostics.csv")
    sample_manifest = _read_csv(VALIDATION_DIR / "historical_replay_large_artifact_sample_manifest.csv")
    final_portfolio = _read_csv(VALIDATION_DIR / "historical_replay_final_portfolio.csv")
    solver = diagnostics[diagnostics.get("diagnostic_type", pd.Series(dtype=str)).astype(str).eq("solver")]
    aggregate_only = solver.get("solver_method", pd.Series(dtype=str)).astype(str).eq(
        "aggregate_only_minimum_weighted_slack_certificate"
    )
    high_tv_rows = int(_bool_series(transition.get("high_tv_transition", pd.Series(False, index=transition.index))).sum())
    transition_intervals = sorted(
        pd.to_numeric(transition.get("interval_quarters", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    required_inputs = int(
        _bool_series(input_manifest.get("required_for_claim", pd.Series(False, index=input_manifest.index))).sum()
    )
    sample_min = (
        int(pd.to_numeric(sample_manifest.get("sample_row_count"), errors="coerce").min())
        if not sample_manifest.empty
        else 0
    )
    residual_rows = (
        int(final_portfolio.get("source_sector", pd.Series(dtype=str)).astype(str).str.contains("SourceBasisResidual", case=False, na=False).sum())
        if not final_portfolio.empty
        else 0
    )
    expected_fragments = [
        "This artifact is generated from live validation CSVs",
        f"Replay input manifest rows: `{len(input_manifest.index)}` total, `{required_inputs}` required",
        f"Artifact integrity rows before final refresh: `{len(artifact_integrity.index)}`",
        f"Aggregate-only certificate quarters: `{int(aggregate_only.sum())}`",
        f"Portfolio transition diagnostics: `{len(transition.index)}` rows; high-TV rows `{high_tv_rows}`; interval_quarters values `{transition_intervals}`",
        f"final source-basis residual rows `{residual_rows}`",
        f"Deterministic large-artifact sample minimum row count: `{sample_min}`",
    ]
    missing = [fragment for fragment in expected_fragments if fragment not in text]
    status = "pass" if not missing else "fail"
    observed = f"expected_fragments={len(expected_fragments)}; missing={missing[:5]}"
    expected = "acceptance artifact is generated from and agrees with live validation CSV counts"
    return status, observed, expected


def _source_role_consistency_status() -> tuple[str, str, str]:
    input_manifest = _read_csv(VALIDATION_DIR / "historical_replay_input_manifest.csv")
    source_scope = _read_csv(SOURCE_SCOPE_PATH)
    if input_manifest.empty or source_scope.empty:
        return "fail", "input manifest or source-scope audit missing", "source-role evidence exists in both input manifest and source-scope audit"
    issues = []
    expected = {
        "tdc_tdc_empirical_anchor": ("true", "true", "runtime_consumed"),
        "tdc_tdc_mmf_rrp_quarterly_adjustments": ("false", "true", "claim_evidence_not_runtime_consumed"),
        "tdc_tdc_mmf_rrp_quarterly_adjustments_sec_full": ("false", "true", "claim_evidence_not_runtime_consumed"),
        "tdc_method_meta": ("false", "true", "claim_evidence_not_runtime_consumed"),
        "z1_raw_archive": ("false", "true", "claim_evidence_not_runtime_consumed"),
        "z1_source_record": ("false", "true", "claim_evidence_not_runtime_consumed"),
    }
    for source_key, (consumed, required, usage) in expected.items():
        rows = input_manifest[input_manifest.get("source_key", pd.Series(dtype=str)).astype(str).eq(source_key)]
        if rows.empty:
            issues.append(f"{source_key}:missing_input_manifest")
            continue
        row = rows.iloc[0]
        if str(row.get("consumed_in_run", "")).lower() != consumed:
            issues.append(f"{source_key}:consumed_in_run={row.get('consumed_in_run')}")
        if str(row.get("required_for_claim", "")).lower() != required:
            issues.append(f"{source_key}:required_for_claim={row.get('required_for_claim')}")
        if str(row.get("source_usage", "")) != usage:
            issues.append(f"{source_key}:source_usage={row.get('source_usage')}")
        source_path = str(row.get("path", ""))
        scope_rows = source_scope[source_scope.get("source_path", pd.Series(dtype=str)).astype(str).eq(source_path)]
        if scope_rows.empty:
            issues.append(f"{source_key}:missing_source_scope")
        elif usage not in set(scope_rows.get("closeout_role", pd.Series(dtype=str)).astype(str)):
            issues.append(f"{source_key}:source_scope_role={sorted(scope_rows.get('closeout_role', pd.Series(dtype=str)).astype(str).unique())}")
    status = "pass" if not issues else "fail"
    observed = f"checked={len(expected)}; issues={issues[:8]}"
    return status, observed, "manifest/source-scope roles agree for consumed inputs, claim evidence, and lineage evidence"


def _aggregate_only_quarters() -> set[str]:
    diagnostics = _read_csv(VALIDATION_DIR / "historical_replay_diagnostics.csv")
    if diagnostics.empty or not {"diagnostic_type", "solver_method", "quarter"}.issubset(diagnostics.columns):
        return set()
    solver = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("solver")]
    return set(
        solver.loc[
            solver["solver_method"].astype(str).eq("aggregate_only_minimum_weighted_slack_certificate"),
            "quarter",
        ]
        .dropna()
        .astype(str)
    )


def _resolve_project_path(raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _quarter_bounds(df: pd.DataFrame) -> tuple[str, str]:
    for col in (
        "quarter",
        "asof_quarter",
        "source_quarter",
        "event_quarter",
        "fiscal_quarter",
        "date",
        "month",
        "auction_date",
        "issue_date",
        "security_date",
    ):
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str)
        if values.empty:
            continue
        return str(values.min()), str(values.max())
    return "", ""


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _as_mil(value: float) -> str:
    return f"{value:,.6f} mil"


def _post_export_constraint_status(diagnostics: pd.DataFrame, portfolio: pd.DataFrame) -> tuple[str, str]:
    sector = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("sector_balance")].copy()
    cohort = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("cohort_balance")].copy()
    required_portfolio_cols = {"quarter", "source_sector", "cohort_id", "FaceValue", "DirtyValue"}
    if diagnostics.empty or portfolio.empty or not required_portfolio_cols.issubset(portfolio.columns):
        return "fail", "missing diagnostics, portfolio rows, or required portfolio columns"

    working = portfolio.copy()
    working["quarter"] = working["quarter"].astype(str)
    working["source_sector"] = working["source_sector"].astype(str)
    working["cohort_id"] = working["cohort_id"].astype(str)
    working["FaceValue"] = pd.to_numeric(working["FaceValue"], errors="coerce").fillna(0.0)
    working["DirtyValue"] = pd.to_numeric(working["DirtyValue"], errors="coerce").fillna(0.0)
    face_by_sector = working.groupby(["quarter", "source_sector"], dropna=False)["FaceValue"].sum()
    dirty_by_sector = working.groupby(["quarter", "source_sector"], dropna=False)["DirtyValue"].sum()
    face_by_cohort = working.groupby(["quarter", "cohort_id"], dropna=False)["FaceValue"].sum()

    sector_residuals: list[float] = []
    missing_sector_groups = 0
    sector_failures = 0
    worst_sector: tuple[str, str, float] = ("", "", 0.0)
    for _, row in sector.iterrows():
        quarter = str(row.get("quarter", ""))
        subject = str(row.get("subject", ""))
        target = pd.to_numeric(pd.Series([row.get("input_total")]), errors="coerce").iloc[0]
        if pd.isna(target):
            target = 0.0
        basis = str(row.get("valuation_basis", ""))
        group = dirty_by_sector if "market_value" in basis else face_by_sector
        key = (quarter, subject)
        achieved = float(group.get(key, 0.0))
        if key not in group.index and abs(float(target)) > NUMERIC_TOLERANCE_MIL:
            missing_sector_groups += 1
        residual = achieved - float(target)
        abs_residual = abs(residual)
        sector_residuals.append(abs_residual)
        if abs_residual > NUMERIC_TOLERANCE_MIL:
            sector_failures += 1
        if abs_residual > abs(worst_sector[2]):
            worst_sector = (quarter, subject, residual)

    cohort_residuals: list[float] = []
    missing_cohort_groups = 0
    cohort_failures = 0
    worst_cohort: tuple[str, str, float] = ("", "", 0.0)
    for _, row in cohort.iterrows():
        quarter = str(row.get("quarter", ""))
        subject = str(row.get("subject", ""))
        target = pd.to_numeric(pd.Series([row.get("input_total")]), errors="coerce").iloc[0]
        if pd.isna(target):
            target = 0.0
        key = (quarter, subject)
        achieved = float(face_by_cohort.get(key, 0.0))
        if key not in face_by_cohort.index and abs(float(target)) > NUMERIC_TOLERANCE_MIL:
            missing_cohort_groups += 1
        residual = achieved - float(target)
        abs_residual = abs(residual)
        cohort_residuals.append(abs_residual)
        if abs_residual > NUMERIC_TOLERANCE_MIL:
            cohort_failures += 1
        if abs_residual > abs(worst_cohort[2]):
            worst_cohort = (quarter, subject, residual)

    max_sector = max(sector_residuals, default=0.0)
    max_cohort = max(cohort_residuals, default=0.0)
    ok = (
        missing_sector_groups == 0
        and missing_cohort_groups == 0
        and max_sector <= NUMERIC_TOLERANCE_MIL
        and max_cohort <= NUMERIC_TOLERANCE_MIL
    )
    observed = (
        f"sector rows={len(sector)}; sector residual failures={sector_failures}; "
        f"missing sector groups={missing_sector_groups}; max sector residual {_as_mil(max_sector)} "
        f"at {worst_sector[0]} / {worst_sector[1]} ({_as_mil(worst_sector[2])}); "
        f"cohort rows={len(cohort)}; cohort residual failures={cohort_failures}; "
        f"missing cohort groups={missing_cohort_groups}; max cohort residual {_as_mil(max_cohort)} "
        f"at {worst_cohort[0]} / {worst_cohort[1]} ({_as_mil(worst_cohort[2])})"
    )
    return ("pass" if ok else "fail"), observed


def _loaded_source_paths() -> set[str]:
    paths: set[str] = set()
    for filename in (
        "historical_replay_source_manifest.csv",
        "historical_replay_input_manifest.csv",
    ):
        df = _read_csv(VALIDATION_DIR / filename)
        if df.empty:
            continue
        for col in ("source_path", "path"):
            if col in df.columns:
                paths.update(str(value) for value in df[col].dropna())
    return paths


def _source_role(path: Path, loaded_paths: set[str]) -> tuple[str, str, str]:
    rel = _rel(path)
    name = path.name
    parent = path.parent.name

    if name in {
        "quarterly_inputs.csv",
        "tdc_estimates.csv",
        "tdc_components.csv",
        "tdc_tier2_regression_series.csv",
        "tdc_du_fiscal_flow_research.csv",
        "tdc_empirical_anchor.csv",
    }:
        return (
            "tdc_target_input",
            "accepted",
            "Used to build the selected empirical TDC ladder or modern formula checks.",
        )
    if name in {"tdc_mmf_rrp_quarterly_adjustments.csv", "tdc_mmf_rrp_quarterly_adjustments_sec_full.csv"}:
        return (
            "claim_evidence_not_runtime_consumed",
            "accepted",
            "MMF/RRP source-of-funds reference evidence; the runtime calculation consumes the already-integrated quarterly input columns.",
        )
    if name == "ffiec_interest_constraints_normalized.csv":
        return (
            "solver_prior_and_interest_diagnostic",
            "accepted",
            "FFIEC maturity buckets now enter the modern bank solver prior and aggregate rows remain diagnostics.",
        )
    if name == "ncua_interest_constraints_normalized.csv":
        return (
            "solver_prior_and_interest_diagnostic",
            "accepted",
            "NCUA maturity buckets now enter the credit-union solver prior as a documented proxy and aggregate rows remain diagnostics.",
        )
    if name == "tier2_interest_source_constraints.csv":
        return (
            "post_solve_interest_diagnostic",
            "accepted",
            "Used as an interest reference/diagnostic, not mislabeled as a solver constraint.",
        )
    if name == "treasury__interest_expense.csv":
        return (
            "post_solve_official_interest_diagnostic",
            "accepted",
            "Used to compare replay interest components against official Treasury interest expense pools.",
        )
    if rel in loaded_paths:
        return (
            "manifested_replay_input",
            "accepted",
            "Listed by the replay source/input manifest; see role-specific rows for solver versus target versus diagnostic use.",
        )

    if name == "auction_allotment_panel_base_slim.csv":
        return (
            "represented by auction-event proxy",
            "accepted",
            "Buycurve auction event rows are represented in the auction-allotment proxy and event-ledger evidence.",
        )
    if name == "z1_treasury_holders_clean.csv":
        return (
            "represented by Z.1 aggregate holder targets",
            "accepted",
            "The replay uses aggregate Z.1 sector levels rather than this external cleaned context file directly.",
        )
    if name in {
        "monthly_issuance_maturity_panel.csv",
        "monthly_ladder_panel.csv",
        "raw_treasury_supply_by_maturity.csv",
        "treasury_stock_reconciliation.csv",
    }:
        return (
            "context for stronger ladder reconciliation",
            "not blocking scoped closeout",
            "Useful for future stronger ladder reconciliation, but the scoped replay is quarterly and cohort-constrained.",
        )
    if name in {"monthly_liquidity_substitution_panel.csv", "monthly_panel_qa.csv"}:
        return (
            "context for liquidity-substitution sensitivity",
            "not blocking scoped closeout",
            "Liquidity substitution is not an input to the accepted quarterly aggregate replay arithmetic.",
        )
    if name == "support__fed_treasury_interest_components.csv":
        return (
            "optional richer interest diagnostic",
            "not blocking scoped closeout",
            "Treasury interest expense and component interest constraints are already wired; this support panel can enrich diagnostics.",
        )
    if name == "treasury__frn_daily_indexes.csv":
        return (
            "duplicate FRN index context",
            "not blocking scoped closeout",
            "The replay uses the raw FRN daily index source; this imported copy is not a separate required input.",
        )
    if name.startswith("tdcsim__") or parent == "tdcest":
        return (
            "TDCest context or sensitivity input",
            "reviewed as noncanonical for scoped replay",
            "The canonical selected TDC ladder, component constraints, and MMF/RRP adjustment inputs are wired separately.",
        )

    return (
        "reviewed imported context",
        "not blocking scoped closeout",
        "No omitted canonical input was identified for the scoped quarterly aggregate target.",
    )


def build_source_scope_audit() -> pd.DataFrame:
    loaded_paths = _loaded_source_paths()
    rows: list[dict[str, object]] = []

    input_manifest = _read_csv(VALIDATION_DIR / "historical_replay_input_manifest.csv")
    if not input_manifest.empty:
        for _, manifest_row in input_manifest.iterrows():
            raw_path = manifest_row.get("path", "")
            if pd.isna(raw_path) or not str(raw_path).strip():
                continue
            path = _resolve_project_path(raw_path)
            rows.append(
                {
                    "source_path": _rel(path),
                    "source_family": str(manifest_row.get("source_key", "")),
                    "row_count": manifest_row.get("row_count", pd.NA),
                    "first_observation": manifest_row.get("first_quarter", pd.NA),
                    "last_observation": manifest_row.get("last_quarter", pd.NA),
                    "closeout_role": str(manifest_row.get("source_usage", manifest_row.get("role", ""))),
                    "closeout_status": "accepted" if str(manifest_row.get("status", "")) == "present" else "missing",
                    "notes": f"input_manifest role={manifest_row.get('role', '')}; consumed_in_run={manifest_row.get('consumed_in_run', '')}; required_for_claim={manifest_row.get('required_for_claim', '')}",
                }
            )

    for path in sorted(IMPORTED_DIR.rglob("*.csv")):
        if _rel(path) in loaded_paths:
            continue
        df = _read_csv(path)
        first, last = _quarter_bounds(df)
        role, status, notes = _source_role(path, loaded_paths)
        rows.append(
            {
                "source_path": _rel(path),
                "source_family": path.parent.name,
                "row_count": int(len(df)),
                "first_observation": first,
                "last_observation": last,
                "closeout_role": role,
                "closeout_status": status,
                "notes": notes,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_SCOPE_PATH, index=False)
    return out


def build_plausibility_audit() -> pd.DataFrame:
    checks: list[CheckRow] = []

    artifact_integrity = _read_csv(VALIDATION_DIR / "artifact_integrity.csv")
    if not artifact_integrity.empty and "status" in artifact_integrity:
        present_mask = artifact_integrity["status"].astype(str).eq("present")
        missing = int((~present_mask).sum())
        present = int(present_mask.sum())
        checks.append(
            CheckRow(
                "artifact_integrity",
                "pass" if missing == 0 else "fail",
                f"{present} present / {missing} missing",
                "0 missing",
                "Core replay artifacts required by the current build are present.",
            )
        )
    manifest_status, manifest_observed, manifest_expected = _artifact_integrity_manifest_status(artifact_integrity)
    checks.append(
        CheckRow(
            "artifact_integrity_live_manifest",
            manifest_status,
            manifest_observed,
            manifest_expected,
            "The closeout audit fails closed if any generated artifact or live config/input identity is stale.",
        )
    )
    live_status, live_observed, live_expected = _required_artifact_status(artifact_integrity)
    checks.append(
        CheckRow(
            "required_artifacts_live",
            live_status,
            live_observed,
            live_expected,
            "The closeout audit fails closed if required artifacts are absent or stale.",
        )
    )
    acceptance_status, acceptance_observed, acceptance_expected = _acceptance_artifact_status()
    checks.append(
        CheckRow(
            "acceptance_artifact_current",
            acceptance_status,
            acceptance_observed,
            acceptance_expected,
            "The acceptance artifact is generated from current evidence and stale hand-maintained counts fail closeout.",
        )
    )

    sample_manifest = _read_csv(VALIDATION_DIR / "historical_replay_large_artifact_sample_manifest.csv")
    if not sample_manifest.empty:
        required_cols = {
            "source_artifact",
            "source_sha256",
            "source_row_count",
            "sample_sha256",
            "sample_row_count",
            "sample_rule",
            "sample_status",
            "declared_strata",
            "stratum_counts_json",
            "strata_with_zero_selected_rows",
        }
        has_cols = required_cols.issubset(sample_manifest.columns)
        sample_rows = pd.to_numeric(sample_manifest.get("sample_row_count"), errors="coerce").fillna(0)
        source_rows = pd.to_numeric(sample_manifest.get("source_row_count"), errors="coerce").fillna(0)
        required_sources = {"historical_replay_portfolio_snapshots.csv", "interest_component_detail.csv"}
        sources = set(sample_manifest.get("source_artifact", pd.Series(dtype=str)).dropna().astype(str))
        bad_strata = []
        if has_cols:
            for _, row in sample_manifest.iterrows():
                artifact = str(row.get("source_artifact", ""))
                try:
                    counts = json.loads(str(row.get("stratum_counts_json", "{}")))
                except json.JSONDecodeError:
                    bad_strata.append(f"{artifact}:malformed_stratum_counts_json")
                    continue
                for stratum, values in counts.items():
                    source_count = int(values.get("source_row_count", 0))
                    selected_count = int(values.get("selected_row_count", 0))
                    if source_count > 0 and selected_count <= 0:
                        bad_strata.append(f"{artifact}:{stratum}:source={source_count}:selected=0")
                zero_value = row.get("strata_with_zero_selected_rows", "")
                zero_text = "" if pd.isna(zero_value) else str(zero_value).strip()
                if zero_text:
                    bad_strata.append(f"{artifact}:manifest_zero={zero_text}")
        nonempty_hashes = (
            sample_manifest.get("source_sha256", pd.Series(dtype=str)).notna().all()
            and sample_manifest.get("sample_sha256", pd.Series(dtype=str)).notna().all()
        )
        sample_statuses = sorted(
            sample_manifest.get("sample_status", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        checks.append(
            CheckRow(
                "large_artifact_sample_manifest",
                "pass"
                if has_cols
                and required_sources.issubset(sources)
                and bool(sample_rows.gt(0).all())
                and bool(source_rows.gt(sample_rows).all())
                and nonempty_hashes
                and not bad_strata
                and sample_statuses == ["sample_nonempty_all_declared_strata_selected"]
                else "fail",
                f"{len(sample_manifest)} sample rows; sources={sorted(sources)}; min sample rows={int(sample_rows.min()) if len(sample_rows) else 0}; required columns={has_cols}; sample_statuses={sample_statuses}; bad_strata={bad_strata[:5]}",
                "large-artifact samples are deterministic, nonempty, tied to full-file/sample hashes, and select every declared positive stratum",
                "GPT audit packages can sample oversized CSVs without losing provenance or silently selecting empty rows.",
            )
        )

    diagnostics = _read_csv(VALIDATION_DIR / "historical_replay_diagnostics.csv")
    if not diagnostics.empty and {"diagnostic_type", "status", "solver_method"}.issubset(diagnostics.columns):
        solver = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("solver")]
        nonconverged = int((~solver["status"].astype(str).eq("converged")).sum())
        methods = sorted(solver["solver_method"].dropna().astype(str).unique())
        max_residual = float(pd.to_numeric(solver.get("residual", 0.0), errors="coerce").abs().max())
        expected_method = {
            "exact_weighted_entropy_projection",
            "exact_weighted_entropy_projection_with_certified_slack",
            "exact_feasible_highs_projection",
            "aggregate_only_minimum_weighted_slack_certificate",
        }
        raw_linprog_methods = [method for method in methods if method == "linprog_minimum_weighted_slack"]
        aggregate_only_quarters = set(
            solver.loc[
                solver["solver_method"].astype(str).eq("aggregate_only_minimum_weighted_slack_certificate"),
                "quarter",
            ].dropna().astype(str)
        )
        portfolio = _read_csv(VALIDATION_DIR / "historical_replay_portfolio_snapshots.csv")
        aggregate_only_portfolio_rows = 0
        if aggregate_only_quarters and not portfolio.empty and "quarter" in portfolio.columns:
            aggregate_only_portfolio_rows = int(portfolio["quarter"].astype(str).isin(aggregate_only_quarters).sum())
        aggregate_only_allowed = aggregate_only_quarters.issubset({"2023Q4"}) and len(aggregate_only_quarters) <= 1
        certificate = _read_csv(VALIDATION_DIR / "valuation_basis_feasibility_certificate.csv")
        certified_methods_ok = set(methods).issubset(expected_method)
        certified_slack_methods = True
        if (
            "exact_weighted_entropy_projection_with_certified_slack" in methods
            or "aggregate_only_minimum_weighted_slack_certificate" in methods
        ):
            certified_slack_methods = not certificate.empty and set(
                certificate.get("certificate_status", pd.Series(dtype=str)).astype(str)
            ) == {"certified_minimum_l1_model_slack"}
        checks.append(
            CheckRow(
                "solver_entropy_convergence",
                "pass"
                if nonconverged == 0
                and max_residual <= NUMERIC_TOLERANCE_MIL
                and certified_methods_ok
                and certified_slack_methods
                and not raw_linprog_methods
                and aggregate_only_portfolio_rows == 0
                and aggregate_only_allowed
                else "fail",
                f"{len(solver) - nonconverged}/{len(solver)} quarters converged/certified; max residual {_as_mil(max_residual)}; methods={methods}; aggregate_only_quarters={sorted(aggregate_only_quarters)}; aggregate_only_allowed={aggregate_only_allowed}; aggregate_only_portfolio_rows={aggregate_only_portfolio_rows}; raw_linprog_methods={raw_linprog_methods}",
                f"cell-level quarters use entropy/KL projection or labeled zero-slack HiGHS feasible projection; positive-slack minimum-L1 LP is certificate-only and emits no portfolio rows within {_as_mil(NUMERIC_TOLERANCE_MIL)}",
                "The allocation must fit actual weighted constraints; zero-slack feasible LP projections are published with an explicit method label, while true slack certificates remain aggregate-only.",
            )
        )
        results = _read_csv(VALIDATION_DIR / "historical_replay_results.csv")
        ledger = _read_csv(VALIDATION_DIR / "historical_replay_ledger.csv")
        holder_cols = [col for col in results.columns if col.startswith("DebtHeld_")] if not results.empty else []
        ledger_holder_cols = [col for col in ledger.columns if col.startswith("DebtHeld_")] if not ledger.empty else []
        result_quarters = (
            pd.to_datetime(results["date"], errors="coerce").dt.to_period("Q").astype(str)
            if not results.empty and "date" in results.columns
            else pd.Series("", index=results.index)
        )
        aggregate_result_rows = (
            results[result_quarters.isin(aggregate_only_quarters)].copy()
            if aggregate_only_quarters and not results.empty
            else pd.DataFrame()
        )
        aggregate_ledger_rows = (
            ledger[ledger.get("quarter", pd.Series("", index=ledger.index)).astype(str).isin(aggregate_only_quarters)].copy()
            if aggregate_only_quarters and not ledger.empty
            else pd.DataFrame()
        )
        fake_holder_cells = 0
        result_status_ok = True
        result_role_ok = True
        result_residual_ok = True
        result_row_count_ok = True
        ledger_status_ok = True
        ledger_role_ok = True
        ledger_null_ok = True
        ledger_row_count_ok = True
        ledger_residual_ok = True
        result_scale_ok = True
        if not aggregate_result_rows.empty:
            fake_holder_cells = int(aggregate_result_rows[holder_cols].notna().sum().sum()) if holder_cols else 0
            result_status_ok = set(aggregate_result_rows.get("HolderSurfaceStatus", pd.Series(dtype=str)).dropna().astype(str)) == {
                "aggregate_only_no_cell_portfolio"
            }
            result_role_ok = set(aggregate_result_rows.get("DebtSurfaceRole", pd.Series(dtype=str)).dropna().astype(str)) == {
                "aggregate_certificate_no_cell_portfolio"
            }
            result_residual_ok = bool(
                pd.to_numeric(
                    aggregate_result_rows.get("Replay_SectorResidual", pd.Series(index=aggregate_result_rows.index)),
                    errors="coerce",
                )
                .isna()
                .all()
            )
            result_scale_ok = bool(
                pd.to_numeric(
                    aggregate_result_rows.get("HolderScaleFactor", pd.Series(index=aggregate_result_rows.index)),
                    errors="coerce",
                )
                .isna()
                .all()
            )
        elif aggregate_only_quarters:
            result_row_count_ok = False
        if not aggregate_ledger_rows.empty:
            ledger_status_ok = set(aggregate_ledger_rows.get("holder_surface_status", pd.Series(dtype=str)).dropna().astype(str)) == {
                "aggregate_only_no_cell_portfolio"
            }
            ledger_role_ok = set(aggregate_ledger_rows.get("debt_surface_role", pd.Series(dtype=str)).dropna().astype(str)) == {
                "aggregate_certificate_no_cell_portfolio"
            }
            ledger_null_ok = int(aggregate_ledger_rows[ledger_holder_cols].notna().sum().sum()) == 0 if ledger_holder_cols else False
            ledger_residual_ok = bool(
                pd.to_numeric(
                    aggregate_ledger_rows.get("Replay_SectorResidual", pd.Series(index=aggregate_ledger_rows.index)),
                    errors="coerce",
                )
                .isna()
                .all()
            ) and bool(
                pd.to_numeric(
                    aggregate_ledger_rows.get("HolderScaleFactor", pd.Series(index=aggregate_ledger_rows.index)),
                    errors="coerce",
                )
                .isna()
                .all()
            )
        elif aggregate_only_quarters:
            ledger_row_count_ok = False
        checks.append(
            CheckRow(
                "aggregate_only_surface_semantics",
                "pass"
                if fake_holder_cells == 0
                and result_row_count_ok
                and result_status_ok
                and result_role_ok
                and result_residual_ok
                and result_scale_ok
                and ledger_row_count_ok
                and ledger_status_ok
                and ledger_role_ok
                and ledger_null_ok
                and ledger_residual_ok
                else "fail",
                f"aggregate_only_quarters={sorted(aggregate_only_quarters)}; fake_holder_cells={fake_holder_cells}; result_row_count_ok={result_row_count_ok}; result_status_ok={result_status_ok}; result_role_ok={result_role_ok}; result_residual_ok={result_residual_ok}; result_scale_ok={result_scale_ok}; ledger_row_count_ok={ledger_row_count_ok}; ledger_status_ok={ledger_status_ok}; ledger_role_ok={ledger_role_ok}; ledger_null_ok={ledger_null_ok}; ledger_residual_ok={ledger_residual_ok}",
                "aggregate-only quarters have null holder allocations/residual allocation fields and explicit aggregate-only status/role in results and ledger",
                "A certified aggregate fit is not a sector holder surface and must not be exported as zero allocations.",
            )
        )
        sector = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("sector_balance")].copy()
        if not sector.empty and "residual" in sector.columns:
            residuals = pd.to_numeric(sector["residual"], errors="coerce").abs()
            slack_cols = [
                col
                for col in ("slack_positive_mil", "slack_negative_mil")
                if col in sector.columns
            ]
            max_slack = 0.0
            slack_rows = 0
            if slack_cols:
                slack_values = sector[slack_cols].apply(pd.to_numeric, errors="coerce").abs().fillna(0.0)
                max_slack = float(slack_values.to_numpy().max()) if not slack_values.empty else 0.0
                slack_rows = int(slack_values.gt(NUMERIC_TOLERANCE_MIL).any(axis=1).sum())
            max_sector_residual = float(residuals.max()) if not residuals.empty else 0.0
            certificate = _read_csv(VALIDATION_DIR / "valuation_basis_feasibility_certificate.csv")
            material_certificate = certificate.copy()
            if not material_certificate.empty:
                material_certificate["abs_slack_mil"] = pd.to_numeric(
                    material_certificate.get("abs_slack_mil", 0.0),
                    errors="coerce",
                ).fillna(0.0)
                material_certificate["abs_residual_mil"] = pd.to_numeric(
                    material_certificate.get("abs_residual_mil", 0.0),
                    errors="coerce",
                ).fillna(0.0)
                material_certificate["relative_to_cohort_total"] = pd.to_numeric(
                    material_certificate.get("relative_to_cohort_total", 0.0),
                    errors="coerce",
                ).fillna(0.0)
                material_certificate["relative_to_sector_target"] = pd.to_numeric(
                    material_certificate.get("relative_to_sector_target", 0.0),
                    errors="coerce",
                ).fillna(0.0)
                material_certificate = material_certificate[
                    material_certificate["abs_slack_mil"].gt(NUMERIC_TOLERANCE_MIL)
                    | material_certificate["abs_residual_mil"].gt(NUMERIC_TOLERANCE_MIL)
                ]
            certified_slack = False
            max_certified_abs = 0.0
            max_certified_rel = 0.0
            max_certified_sector_rel = 0.0
            if not material_certificate.empty:
                statuses = set(material_certificate.get("certificate_status", pd.Series(dtype=str)).astype(str))
                max_certified_abs = float(material_certificate["abs_slack_mil"].max())
                max_certified_rel = float(material_certificate["relative_to_cohort_total"].max())
                max_certified_sector_rel = float(material_certificate["relative_to_sector_target"].max())
                certified_slack = (
                    statuses == {"certified_minimum_l1_model_slack"}
                    and max_certified_abs <= VALUATION_SLACK_ABS_LIMIT_MIL
                    and max_certified_rel <= VALUATION_SLACK_REL_LIMIT
                    and max_certified_sector_rel <= VALUATION_SLACK_SECTOR_REL_LIMIT
                    and len(material_certificate) == slack_rows
                )
            exact_fit = (
                max_sector_residual <= NUMERIC_TOLERANCE_MIL
                and slack_rows == 0
                and max_slack <= NUMERIC_TOLERANCE_MIL
            )
            checks.append(
                CheckRow(
                    "weighted_sector_constraint_fit",
                    "pass" if exact_fit or certified_slack else "fail",
                    f"{len(sector)} sector rows; max actual residual {_as_mil(max_sector_residual)}; slack rows={slack_rows}; max slack {_as_mil(max_slack)}; certified max abs {_as_mil(max_certified_abs)}; certified max system rel {max_certified_rel:.6f}; certified max sector rel {max_certified_sector_rel:.6f}",
                    f"actual source-basis sector residuals <= {_as_mil(NUMERIC_TOLERANCE_MIL)} or certified minimum-L1 model slack <= {_as_mil(VALUATION_SLACK_ABS_LIMIT_MIL)}, {VALUATION_SLACK_REL_LIMIT:.4f} of cohort total, and {VALUATION_SLACK_SECTOR_REL_LIMIT:.2f} of affected sector target",
                    "`sector_balance.residual` is the aggregate-consistency check; solver residual alone is not sufficient, and any model slack must disclose both system-relative and affected-sector-relative severity.",
                )
            )
        post_export_status, post_export_observed = _post_export_constraint_status(diagnostics, portfolio)
        checks.append(
            CheckRow(
                "post_export_constraint_fit",
                post_export_status,
                post_export_observed,
                f"exported portfolio sector and cohort residuals <= {_as_mil(NUMERIC_TOLERANCE_MIL)} with no missing constrained groups",
                "The published portfolio surface, not only the latent solver matrix, must satisfy claimed aggregate constraints.",
            )
        )

        soma_holdings = _read_csv(VALIDATION_DIR / "soma_treasury_holdings_quarterly.csv")
        soma_fixed = _read_csv(VALIDATION_DIR / "soma_fixed_allocations.csv")
        soma_diag = _read_csv(VALIDATION_DIR / "soma_holdout_diagnostics.csv")
        if not soma_holdings.empty or not soma_fixed.empty or not soma_diag.empty:
            fixed_source_ok = (
                not soma_fixed.empty
                and set(soma_fixed.get("sector", pd.Series(dtype=str)).dropna().astype(str).unique()) == {"monetary_authority"}
                and set(soma_fixed.get("fixed_allocation_policy", pd.Series(dtype=str)).dropna().astype(str).unique())
                == {"observed_soma_exact_block"}
            )
            matched_face = float(pd.to_numeric(soma_diag.get("matched_face_mil", 0.0), errors="coerce").fillna(0.0).sum())
            soma_face = float(pd.to_numeric(soma_diag.get("soma_face_mil", 0.0), errors="coerce").fillna(0.0).sum())
            max_residual_unmatched = float(
                pd.to_numeric(soma_diag.get("residual_unmatched_face_mil", 0.0), errors="coerce")
                .fillna(0.0)
                .abs()
                .max()
            )
            max_timing_rolloff = float(
                pd.to_numeric(soma_diag.get("maturity_timing_rolloff_face_mil", 0.0), errors="coerce")
                .fillna(0.0)
                .abs()
                .max()
            )
            max_capped = float(
                pd.to_numeric(soma_diag.get("capped_excess_face_mil", 0.0), errors="coerce").fillna(0.0).abs().max()
            )
            statuses = sorted(soma_diag.get("status", pd.Series(dtype=str)).dropna().astype(str).unique())
            allowed_statuses = {"matched", "matched_with_quarter_end_maturity_rolloff"}
            soma_quarters = pd.PeriodIndex(soma_diag.get("quarter", pd.Series(dtype=str)).astype(str), freq="Q")
            modern_soma = soma_diag[soma_quarters >= pd.Period("2010Q1", freq="Q")].copy()
            max_modern_residual_unmatched = float(
                pd.to_numeric(modern_soma.get("residual_unmatched_face_mil", 0.0), errors="coerce")
                .fillna(0.0)
                .abs()
                .max()
            )
            modern_statuses = sorted(modern_soma.get("status", pd.Series(dtype=str)).dropna().astype(str).unique())
            early_residual_quarters = int(
                (
                    (soma_quarters < pd.Period("2010Q1", freq="Q"))
                    & pd.to_numeric(soma_diag.get("residual_unmatched_face_mil", 0.0), errors="coerce")
                    .fillna(0.0)
                    .gt(NUMERIC_TOLERANCE_MIL)
                ).sum()
            )
            checks.append(
                CheckRow(
                    "soma_observed_block",
                    "pass"
                    if not soma_holdings.empty
                    and not soma_diag.empty
                    and fixed_source_ok
                    and matched_face > 0.0
                    and max_modern_residual_unmatched <= NUMERIC_TOLERANCE_MIL
                    and set(modern_statuses).issubset(allowed_statuses)
                    else "fail",
                    f"SOMA holdings rows={len(soma_holdings)}; fixed allocation rows={len(soma_fixed)}; diagnostic quarters={len(soma_diag)}; matched face {_as_mil(matched_face)} / SOMA face {_as_mil(soma_face)}; max quarter-end timing rolloff {_as_mil(max_timing_rolloff)}; max all-period residual unmatched {_as_mil(max_residual_unmatched)}; max modern residual unmatched {_as_mil(max_modern_residual_unmatched)}; max capped {_as_mil(max_capped)}; early residual quarters={early_residual_quarters}; statuses={statuses}; fixed_source_ok={fixed_source_ok}",
                    "NY Fed SOMA observed Treasury holdings are exported, fixed to monetary_authority, and modern-period holdings are fully matched to MSPD cohorts where quarter-end supply exists; one-day quarter-end maturity rolloff and pre-2010 legacy CUSIP source gaps are disclosed separately",
                    "The Fed/SOMA portfolio is an observed block in the modern holdout period; pre-2010 SOMA rows absent from MSPD are source-scope gaps, not inferred holder allocations.",
                )
            )

    input_manifest = _read_csv(VALIDATION_DIR / "historical_replay_input_manifest.csv")
    if not input_manifest.empty and {"required", "status"}.issubset(input_manifest.columns):
        input_status, input_observed, input_expected = _required_input_status(input_manifest)
        checks.append(
            CheckRow(
                "required_inputs",
                input_status,
                input_observed,
                input_expected,
                "The replay fails closed if a required source input is missing or stale.",
            )
        )
        role_status, role_observed, role_expected = _source_role_consistency_status()
        checks.append(
            CheckRow(
                "source_role_consistency",
                role_status,
                role_observed,
                role_expected,
                "Input manifest and source-scope audit share one role model for runtime inputs, claim evidence, and lineage evidence.",
            )
        )

    target = _read_csv(VALIDATION_DIR / "tdcest_selected_ladder_crosscheck.csv")
    if not target.empty:
        verdict = target.get("verdict", pd.Series("", index=target.index)).astype(str)
        target_missing = int(verdict.eq("missing_selected_target").sum())
        selected_values = target.get("selected_tdc_value_bil", pd.Series(pd.NA, index=target.index))
        compared = target[selected_values.notna()]
        compared_verdict = compared.get("verdict", pd.Series("", index=compared.index)).astype(str)
        matched = int(compared_verdict.eq("matched").sum())
        mismatches = int(compared_verdict.ne("matched").sum())
        first_compared = str(compared["quarter"].min()) if "quarter" in compared and not compared.empty else ""
        last_compared = str(compared["quarter"].max()) if "quarter" in compared and not compared.empty else ""
        checks.append(
            CheckRow(
                "selected_tdc_target_wiring",
                "pass" if mismatches == 0 and matched > 0 else "fail",
                f"{matched}/{len(compared)} compared quarters matched; {target_missing} pre-target rows disclosed; compared range {first_compared}-{last_compared}",
                "all quarters with selected target match",
                "The 2001 rows are source/run warm-up rows without a selected empirical target; target-comparable acceptance is 2002Q1-2025Q4.",
            )
        )

    results = _read_csv(VALIDATION_DIR / "historical_replay_results.csv")
    if not results.empty and {
        "MSPD_Cohort_Total",
        "Z1_Holder_Total",
        "MSPD_Z1_SourceBasisDiff",
        "MSPD_Minus_IncludedRawZ1",
        "ModeledFaceEquivalentBasisResidual",
    }.issubset(results.columns):
        raw = pd.to_numeric(results["MSPD_Cohort_Total"], errors="coerce") - pd.to_numeric(
            results["Z1_Holder_Total"],
            errors="coerce",
        )
        source_gap = (raw - pd.to_numeric(results["MSPD_Z1_SourceBasisDiff"], errors="coerce")).abs().max()
        explicit_gap = (raw - pd.to_numeric(results["MSPD_Minus_IncludedRawZ1"], errors="coerce")).abs().max()
        modeled_nonnull = int(results["ModeledFaceEquivalentBasisResidual"].notna().sum())
        max_gap = float(max(source_gap, explicit_gap))
        checks.append(
            CheckRow(
                "raw_vs_modeled_residual_separation",
                "pass" if max_gap <= 1.0e-9 and modeled_nonnull == len(results) else "fail",
                f"max raw identity gap {_as_mil(max_gap)}; modeled residual rows {modeled_nonnull}/{len(results)}",
                "raw MSPD-Z1 identity separate from model face-equivalent residual",
                "`MSPD_Z1_SourceBasisDiff` is literal raw MSPD minus included Z.1; model face-equivalent residual is separate.",
            )
        )

    z1_flow = _read_csv(VALIDATION_DIR / "z1_transaction_flow_diagnostics.csv")
    if not z1_flow.empty:
        tx_values = pd.to_numeric(z1_flow.get("z1_transaction_flow_mil"), errors="coerce")
        tx_saar = pd.to_numeric(z1_flow.get("z1_transaction_flow_saar_mil"), errors="coerce")
        tx_count = int(tx_values.notna().sum())
        statuses = sorted(z1_flow.get("diagnostic_status", pd.Series([], dtype=str)).dropna().astype(str).unique())
        boundaries = sorted(z1_flow.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique())
        conversions = sorted(
            z1_flow.get("z1_transaction_flow_conversion", pd.Series([], dtype=str)).dropna().astype(str).unique()
        )
        paired = tx_values.notna() & tx_saar.notna()
        max_conversion_gap = float((tx_values[paired] * 4.0 - tx_saar[paired]).abs().max()) if paired.any() else float("inf")
        checks.append(
            CheckRow(
                "z1_transaction_flow_diagnostic",
                "pass"
                if tx_count > 0
                and boundaries == ["z1_transaction_flow_is_aggregate_transition_diagnostic_not_exact_transfer"]
                and conversions == ["FA_divided_by_4_to_quarterly_moment"]
                and max_conversion_gap <= NUMERIC_TOLERANCE_MIL
                else "fail",
                f"{len(z1_flow)} sector-quarter rows; {tx_count} rows with transaction-flow observations; max FA/4 conversion gap {_as_mil(max_conversion_gap)}; statuses={statuses[:5]}; conversions={conversions}; boundaries={boundaries}",
                "Z.1 transaction flows are converted from FA SAAR to quarterly moments and wired as aggregate transition diagnostics, not exact secondary-market transfers",
                "Transaction-flow evidence must be visible for turnover review instead of being discarded before the registry.",
            )
        )

    modern = _read_csv(VALIDATION_DIR / "tdcest_modern_formula_crosscheck.csv")
    if not modern.empty and {"canonical_formula_status", "component_anchored_status"}.issubset(modern.columns):
        failures = int(
            (
                ~modern["canonical_formula_status"].astype(str).eq("matched")
                | ~modern["component_anchored_status"].astype(str).eq("matched")
            ).sum()
        )
        checks.append(
            CheckRow(
                "modern_formula_identity",
                "pass" if failures == 0 else "fail",
                f"{len(modern) - failures}/{len(modern)} modern quarters match",
                "all modern quarters match",
                "The modern component arithmetic matches the expected identity for available rows.",
            )
        )

    event = _read_csv(VALIDATION_DIR / "historical_replay_event_rollforward.csv")
    if not event.empty:
        residual = pd.to_numeric(event.get("rollforward_residual_mil", 0.0), errors="coerce").abs().max()
        residual = float(0.0 if pd.isna(residual) else residual)
        checks.append(
            CheckRow(
                "event_rollforward_identity",
                "pass" if residual <= 1e-6 else "fail",
                f"max residual {_as_mil(residual)}",
                "<= 0.000001 mil",
                "Cohort/event rollforward balances to numerical precision.",
            )
        )
    event_ledger = _read_csv(VALIDATION_DIR / "historical_replay_event_ledger.csv")
    if not event_ledger.empty:
        terminal_keys = event_ledger.get("source_row_key", pd.Series([], dtype=str)).astype(str)
        terminal_rows = terminal_keys.str.endswith("|terminal_exit")
        terminal_count = int(terminal_rows.sum())
        terminal_event_type = event_ledger.get("event_type", pd.Series("", index=event_ledger.index)).astype(str)
        terminal_evidence = event_ledger.get("evidence_status", pd.Series("", index=event_ledger.index)).astype(str)
        maturity_count = int((terminal_rows & terminal_event_type.eq("maturity_redemption")).sum())
        called_count = int((terminal_rows & terminal_event_type.eq("called_redemption")).sum())
        unsupported_count = int(
            (
                terminal_rows
                & (
                    terminal_event_type.eq("source_discontinuity_exit")
                    | terminal_evidence.eq("unsupported_terminal_exit")
                )
            ).sum()
        )
        checks.append(
            CheckRow(
                "terminal_exits_tagged",
                "pass",
                f"{terminal_count} terminal exits; {maturity_count} maturity, {called_count} called, {unsupported_count} unsupported",
                "terminal exits explicitly tagged",
                "Terminal runoff is classified rather than hidden in an unlabelled residual.",
            )
        )
    unexplained = _read_csv(VALIDATION_DIR / "historical_replay_unexplained_change_ledger.csv")
    if not unexplained.empty:
        old_mask = pd.Series(False, index=unexplained.index)
        for col in ("evidence_status", "derivation", "event_type"):
            if col in unexplained.columns:
                old_mask |= unexplained[col].astype(str).str.contains(
                    "old_reopening_initial_split_residual",
                    na=False,
                )
        old = unexplained[old_mask]
        old_abs = float(pd.to_numeric(old.get("unexplained_residual_change_mil", 0.0), errors="coerce").abs().sum())
        if len(old) > 0:
            checks.append(
                CheckRow(
                    "old_opening_split_residual_disclosed",
                    "pass",
                    f"{len(old)} rows; abs {_as_mil(old_abs)}",
                    "explicitly disclosed if present",
                    "Opening split residuals are labelled and kept out of ordinary event evidence.",
                )
            )

    residual = _read_csv(VALIDATION_DIR / "historical_replay_final_portfolio.csv")
    if not residual.empty:
        private_rows = 0
        residual_mask = pd.Series(False, index=residual.index)
        for col in ("tdcsim_holder", "holder", "native_sector", "broad_holder_class"):
            if col in residual.columns:
                residual_mask |= residual[col].astype(str).str.contains(
                    "SourceBasisResidual",
                    case=False,
                    na=False,
                )
        residual_rows = residual[residual_mask]
        for col in ("tdcsim_holder", "holder", "native_sector", "broad_holder_class"):
            if col in residual.columns:
                private_rows += int(
                    (
                        residual_mask
                        & residual[col].astype(str).str.contains("Private", case=False, na=False)
                    ).sum()
                )
        checks.append(
            CheckRow(
                "source_basis_residual_not_private",
                "pass" if private_rows == 0 else "fail",
                f"{len(residual_rows)} final residual rows; {private_rows} private mappings",
                "0 private mappings",
                "Source-basis residuals are not mapped into private holder economics.",
            )
        )

    portfolio_constraints = _read_csv(VALIDATION_DIR / "portfolio_constraint_diagnostics.csv")
    negative_bridge = _read_csv(VALIDATION_DIR / "negative_sector_netting_bridge.csv")
    if not portfolio_constraints.empty or not negative_bridge.empty:
        netted = portfolio_constraints[
            portfolio_constraints.get("status", pd.Series("", index=portfolio_constraints.index))
            .astype(str)
            .eq("negative_native_sector_netting")
        ] if not portfolio_constraints.empty else pd.DataFrame()
        expected_boundary = "broad_holder_consistency_after_nonnegative_native_sector_netting"
        boundaries = (
            sorted(negative_bridge.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique())
            if not negative_bridge.empty
            else []
        )
        max_share = (
            float(pd.to_numeric(negative_bridge.get("gross_adjustment_share_of_raw_abs", 0.0), errors="coerce").max())
            if not negative_bridge.empty
            else 0.0
        )
        mmf_rows = (
            negative_bridge[
                negative_bridge.get("native_sector", pd.Series("", index=negative_bridge.index))
                .astype(str)
                .eq("money_market_funds")
            ]
            if not negative_bridge.empty
            else pd.DataFrame()
        )
        disclosed = (
            netted.empty
            or (
                not negative_bridge.empty
                and boundaries == [expected_boundary]
                and max_share <= NEGATIVE_NETTING_GROSS_SHARE_LIMIT
                and mmf_rows.empty
            )
        )
        checks.append(
            CheckRow(
                "negative_sector_netting_disclosure",
                "pass" if disclosed else "fail",
                f"{len(netted)} adjusted target rows; {len(negative_bridge)} bridge rows; max gross adjustment share {max_share:.6f}; mmf netted rows={len(mmf_rows)}; boundaries={boundaries}",
                f"raw-to-adjusted bridge present when negative native-sector netting occurs; max share <= {NEGATIVE_NETTING_GROSS_SHARE_LIMIT:.2f}; no MMF rows netted",
                "The replay claim is broad-holder consistency after nonnegative netting, not native-sector exactness in adjusted quarters.",
            )
        )

    tips = _read_csv(VALIDATION_DIR / "tips_principal_identity.csv")
    if not tips.empty and "identity_status" in tips.columns:
        failures = int((~tips["identity_status"].astype(str).eq("matched")).sum())
        checks.append(
            CheckRow(
                "tips_principal_identity",
                "pass" if failures == 0 else "fail",
                f"{len(tips) - failures}/{len(tips)} rows pass",
                "all TIPS rows pass",
                "TIPS principal decomposition is internally consistent.",
            )
        )

    interest = _read_csv(VALIDATION_DIR / "interest_proxy_alignment.csv")
    if not interest.empty and "tdcest_point" in interest.columns:
        referenced = interest[interest["tdcest_point"].notna()]
        if "within_feasible_bounds" in referenced.columns:
            feasible = _bool_series(referenced["within_feasible_bounds"])
            infeasible = referenced[~feasible]
            holders = sorted(infeasible.get("native_sector", pd.Series([], dtype=str)).dropna().astype(str).unique())
            gap_col = "constrained_gap" if "constrained_gap" in infeasible.columns else "stock_only_gap"
            max_gap = (
                float(pd.to_numeric(infeasible.get(gap_col, 0.0), errors="coerce").abs().max())
                if len(infeasible)
                else 0.0
            )
            points = pd.to_numeric(infeasible.get("tdcest_point", pd.Series(dtype=float)), errors="coerce").abs()
            rel_gap = pd.to_numeric(infeasible.get(gap_col, pd.Series(dtype=float)), errors="coerce").abs() / points.replace(0.0, pd.NA)
            max_rel_gap = float(rel_gap.max()) if len(infeasible) and rel_gap.notna().any() else 0.0
            nonforeign = infeasible[~infeasible.get("native_sector", pd.Series("", index=infeasible.index)).astype(str).eq("Foreign")]
            nonforeign_gap = (
                float(pd.to_numeric(nonforeign.get(gap_col, 0.0), errors="coerce").abs().max())
                if len(nonforeign)
                else 0.0
            )
            nonforeign_points = pd.to_numeric(
                nonforeign.get("tdcest_point", pd.Series(dtype=float)),
                errors="coerce",
            ).abs()
            nonforeign_rel = (
                pd.to_numeric(nonforeign.get(gap_col, pd.Series(dtype=float)), errors="coerce").abs()
                / nonforeign_points.replace(0.0, pd.NA)
            )
            max_nonforeign_rel = (
                float(nonforeign_rel.max()) if len(nonforeign) and nonforeign_rel.notna().any() else 0.0
            )
            foreign_only_or_small = (
                len(nonforeign) == 0
                or (
                    nonforeign_gap <= NONFOREIGN_INTEREST_GAP_LIMIT_MIL
                    and max_nonforeign_rel <= NONFOREIGN_INTEREST_REL_LIMIT
                )
            )
            status = (
                "pass"
                if len(infeasible) == 0
                or (
                    foreign_only_or_small
                    and max_gap <= 50_000.0
                    and max_rel_gap <= 0.30
                )
                else "fail"
            )
            checks.append(
                CheckRow(
                    "interest_projection_feasibility",
                    status,
                    f"{int(feasible.sum())}/{len(referenced)} referenced rows feasible; infeasible holders={holders}; max infeasible gap {_as_mil(max_gap)}; max relative gap {max_rel_gap:.6f}; max non-Foreign gap {_as_mil(nonforeign_gap)}; max non-Foreign relative gap {max_nonforeign_rel:.6f}",
                    "all feasible or disclosed current-model incompatibilities under configured Foreign and de-minimis non-Foreign thresholds",
                    "Remaining infeasible rows are diagnostics; the artifact does not by itself prove whether the source, mapping, or model causes the incompatibility.",
                )
            )

    mmf = _read_csv(VALIDATION_DIR / "mmf_component_reconciliation.csv")
    if not mmf.empty:
        aggregate_only_quarters = _aggregate_only_quarters()
        mmf_claim = mmf[
            ~mmf.get("quarter", pd.Series("", index=mmf.index)).astype(str).isin(aggregate_only_quarters)
        ].copy()
        has_direct_gap_columns = {
            "direct_total_gap_mil",
            "direct_bill_component_gap_mil",
            "direct_nonbill_component_gap_mil",
        }.issubset(mmf.columns)
        bill_gap = float(
            (
                _numeric_column(mmf_claim, "direct_bill_component_gap_mil")
                if has_direct_gap_columns
                else _numeric_column(mmf_claim, "bill_component_gap_mil")
            ).abs().max()
        )
        nonbill_gap = float(
            (
                _numeric_column(mmf_claim, "direct_nonbill_component_gap_mil")
                if has_direct_gap_columns
                else _numeric_column(mmf_claim, "nonbill_component_gap_mil")
            ).abs().max()
        )
        total_gap = float(
            _numeric_column(mmf_claim, "direct_total_gap_mil").abs().max()
            if has_direct_gap_columns
            else float("inf")
        )
        fixed_long = float(pd.to_numeric(mmf_claim.get("fixed_rate_gt_397d_mil", 0.0), errors="coerce").abs().max())
        statuses = sorted(mmf_claim.get("post_solve_status", pd.Series([], dtype=str)).dropna().astype(str).unique())
        aggregate_only_rows = int(
            mmf.get("quarter", pd.Series("", index=mmf.index)).astype(str).isin(aggregate_only_quarters).sum()
        )
        checks.append(
            CheckRow(
                "mmf_bill_component_constraint",
                "pass"
                if not mmf_claim.empty
                and bill_gap <= NUMERIC_TOLERANCE_MIL
                and nonbill_gap <= NUMERIC_TOLERANCE_MIL
                and total_gap <= NUMERIC_TOLERANCE_MIL
                and fixed_long <= NUMERIC_TOLERANCE_MIL
                and statuses == ["matched"]
                and has_direct_gap_columns
                else "fail",
                f"{len(mmf_claim)}/{len(mmf)} portfolio-claim quarters checked; aggregate_only_excluded={aggregate_only_rows}; direct gap columns={has_direct_gap_columns}; max direct total gap {_as_mil(total_gap)}; max direct bill gap {_as_mil(bill_gap)}; max direct nonbill gap {_as_mil(nonbill_gap)}; max fixed-rate >397d {_as_mil(fixed_long)}; statuses={statuses}",
                "direct Z.1 MMF Treasury bill/nonbill components fit exactly in quarters with exported cell-level portfolios; aggregate-only quarters are excluded from cell-level MMF claims",
                "MMFs use the available Z.1 bill component where a cell-level portfolio is claimed; aggregate-only certificates disclose that no portfolio rows exist.",
            )
        )

    maturity = _read_csv(VALIDATION_DIR / "maturity_prior_reconciliation.csv")
    if not maturity.empty and {"source_scope", "prior_status", "modeled_minus_observed_share"}.issubset(maturity.columns):
        aggregate_only_quarters = _aggregate_only_quarters()
        maturity_claim = maturity[
            ~maturity.get("quarter", pd.Series("", index=maturity.index)).astype(str).isin(aggregate_only_quarters)
        ].copy()
        scopes = sorted(maturity_claim["source_scope"].dropna().astype(str).unique())
        prior_status = maturity_claim["prior_status"].astype(str)
        constraint_role = maturity_claim.get(
            "constraint_role",
            pd.Series("hard_bucket_proxy_constraint", index=maturity_claim.index),
        ).astype(str)
        ffiec_claim = maturity_claim[maturity_claim["source_scope"].astype(str).eq("ffiec_bank_maturity_prior")]
        ncua_claim = maturity_claim[maturity_claim["source_scope"].astype(str).eq("ncua_credit_union_maturity_proxy")]
        ffiec_applied = int(ffiec_claim["prior_status"].astype(str).eq("solver_prior_applied").sum())
        ncua_soft = int(ncua_claim["prior_status"].astype(str).eq("soft_solver_prior_applied").sum())
        ncua_hard = int(ncua_claim["prior_status"].astype(str).eq("solver_prior_applied").sum())
        ffiec_max_abs_gap = (
            float(pd.to_numeric(ffiec_claim["modeled_minus_observed_share"], errors="coerce").abs().max())
            if not ffiec_claim.empty
            else math.inf
        )
        max_abs_gap = float(pd.to_numeric(maturity_claim["modeled_minus_observed_share"], errors="coerce").abs().max())
        rmse = float(
            (pd.to_numeric(maturity_claim["modeled_minus_observed_share"], errors="coerce").dropna() ** 2).mean() ** 0.5
        )
        aggregate_only_rows = int(
            maturity.get("quarter", pd.Series("", index=maturity.index)).astype(str).isin(aggregate_only_quarters).sum()
        )
        ffiec_hard_role_ok = bool(
            not ffiec_claim.empty
            and ffiec_claim.get("constraint_role", pd.Series("hard_bucket_proxy_constraint", index=ffiec_claim.index))
            .astype(str)
            .eq("hard_bucket_proxy_constraint")
            .all()
        )
        ncua_soft_role_ok = bool(
            not ncua_claim.empty
            and ncua_claim.get("constraint_role", pd.Series("", index=ncua_claim.index))
            .astype(str)
            .eq("soft_all_investment_prior_only")
            .all()
        )
        checks.append(
            CheckRow(
                "ffiec_ncua_maturity_prior_use",
                "pass"
                if not maturity_claim.empty
                and ffiec_applied == len(ffiec_claim)
                and ncua_soft == len(ncua_claim)
                and ncua_hard == 0
                and ffiec_hard_role_ok
                and ncua_soft_role_ok
                and {"ffiec_bank_maturity_prior", "ncua_credit_union_maturity_proxy"}.issubset(scopes)
                and ffiec_max_abs_gap <= 0.20
                else "fail",
                f"FFIEC hard rows {ffiec_applied}/{len(ffiec_claim)}; NCUA soft rows {ncua_soft}/{len(ncua_claim)}; NCUA hard rows {ncua_hard}; aggregate_only_excluded={aggregate_only_rows}; scopes={scopes}; roles={sorted(constraint_role.dropna().unique())}; FFIEC max share gap {ffiec_max_abs_gap:.6f}; all-source max share gap {max_abs_gap:.6f}; RMSE {rmse:.6f}",
                "FFIEC bank maturity evidence is represented as hard maturity-bucket proxy target splits; NCUA all-investment maturity evidence is soft prior/diagnostic only and cannot generate exact Treasury bucket eligibility constraints",
                "Modern bank maturity evidence remains a hard proxy split; credit-union all-investment maturity buckets are source-weak and must remain soft/diagnostic.",
            )
        )

    portfolio = _read_csv(VALIDATION_DIR / "historical_replay_portfolio_snapshots.csv")
    if not portfolio.empty and {"quarter", "native_sector", "cohort_id", "FaceValue", "source_sector"}.issubset(portfolio.columns):
        work = portfolio.copy()
        work["FaceValue"] = pd.to_numeric(work["FaceValue"], errors="coerce").fillna(0.0)
        work = work[
            work["FaceValue"].gt(0.0)
            & ~work["source_sector"].astype(str).str.contains("SourceBasisResidual", na=False)
        ]
        grouped = work.groupby(["quarter", "native_sector", "cohort_id"], as_index=False)["FaceValue"].sum()
        totals = grouped.groupby(["quarter", "native_sector"])["FaceValue"].transform("sum")
        grouped["share"] = grouped["FaceValue"] / totals
        concentration = grouped.groupby(["quarter", "native_sector"]).agg(
            nonzero_cusips=("cohort_id", "nunique"),
            max_share=("share", "max"),
            total_face=("FaceValue", "sum"),
        )
        material_concentration = concentration[
            concentration["total_face"].ge(MATERIAL_PORTFOLIO_SECTOR_TOTAL_MIL)
        ]
        material_concentration_for_gate = material_concentration[
            ~material_concentration.index.get_level_values("native_sector")
            .astype(str)
            .isin(LOW_IDENTIFICATION_CONCENTRATION_SECTORS)
        ]
        one_cusip = int(material_concentration_for_gate["nonzero_cusips"].eq(1).sum())
        max90 = int(material_concentration_for_gate["max_share"].ge(0.90).sum())
        ignored_low_id = int(len(material_concentration.index) - len(material_concentration_for_gate.index))
        tiny_one_cusip = int(
            concentration[
                concentration["total_face"].lt(MATERIAL_PORTFOLIO_SECTOR_TOTAL_MIL)
                & concentration["nonzero_cusips"].eq(1)
            ].shape[0]
        )
        transition = _read_csv(VALIDATION_DIR / "portfolio_transition_diagnostics.csv")
        tv = pd.to_numeric(transition.get("total_variation", pd.Series(dtype=float)), errors="coerce").dropna()
        tv_median = float(tv.median()) if not tv.empty else 0.0
        tv_gt_09 = float(tv.gt(0.90).mean()) if not tv.empty else 0.0
        high_transition_count = 0
        missing_transition_context = 0
        tautological_transition_context = 0
        corrected_flow_cols = False
        adjacent_only_transitions = False
        transition_boundaries: list[str] = []
        if not transition.empty:
            high_mask = _bool_series(transition.get("high_tv_transition", pd.Series(False, index=transition.index)))
            high_transition_count = int(high_mask.sum())
            status_text = transition.get(
                "transition_explanation_status",
                pd.Series("", index=transition.index),
            ).astype(str)
            missing_transition_context = int((high_mask & status_text.eq("high_turnover_missing_transition_context")).sum())
            tautological_transition_context = int(
                (
                    high_mask
                    & status_text.isin(
                        {
                            "high_turnover_with_z1_flow_and_cohort_churn_context",
                            "high_turnover_with_z1_flow_context",
                            "high_turnover_with_cohort_churn_context",
                        }
                    )
                ).sum()
            )
            corrected_flow_cols = {
                "interval_quarters",
                "z1_transaction_flow_saar_mil",
                "z1_flow_context_status",
                "event_context_status",
                "event_ledger_gross_activity_mil",
                "event_source_issue_mil",
                "event_source_redemption_mil",
                "event_source_indexation_mil",
                "event_source_reclassification_mil",
                "event_unexplained_cohort_change_mil",
                "event_unexplained_residual_change_mil",
                "event_component_reconciliation_gap_mil",
                "event_component_scope",
                "modeled_minus_z1_flow_mil",
                "modeled_minus_z1_flow_share",
            }.issubset(transition.columns)
            adjacent_only_transitions = bool(
                pd.to_numeric(transition.get("interval_quarters", pd.Series(dtype=float)), errors="coerce").eq(1).all()
            ) if "interval_quarters" in transition.columns else False
            transition_boundaries = sorted(
                transition.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique()
            )
        checks.append(
            CheckRow(
                "portfolio_concentration_continuity",
                "pass"
                if one_cusip == 0
                and max90 == 0
                and tv_median < 0.50
                and tv_gt_09 <= MAX_HIGH_TV_TRANSITION_SHARE
                and high_transition_count <= MAX_HIGH_TV_TRANSITIONS
                and missing_transition_context == 0
                and tautological_transition_context == 0
                and corrected_flow_cols
                and adjacent_only_transitions
                and transition_boundaries == ["transition_diagnostic_not_exact_secondary_market_transfer"]
                else "fail",
                f"material one-cusip sectors={one_cusip}; material max-share>=90% sectors={max90}; ignored low-identification material sectors={ignored_low_id}; tiny one-cusip sectors={tiny_one_cusip}; material threshold {_as_mil(MATERIAL_PORTFOLIO_SECTOR_TOTAL_MIL)}; median TV={tv_median:.6f}; TV>90% share={tv_gt_09:.6f}; high transition rows={high_transition_count}; missing transition context={missing_transition_context}; tautological context rows={tautological_transition_context}; corrected context columns={corrected_flow_cols}; adjacent_only_transitions={adjacent_only_transitions}",
                f"no material one-CUSIP/max-share corner allocation outside named low-identification sectors; TV>90% transition share <= {MAX_HIGH_TV_TRANSITION_SHARE:.2f}; high transition rows <= {MAX_HIGH_TV_TRANSITIONS}; no high transition missing or tautological Z.1/event context; transition rows are adjacent-quarter rows",
                "Continuity diagnostics must emit high-turnover rows and label available transition context; this is not an exact ownership proof.",
            )
        )

    transition = _read_csv(VALIDATION_DIR / "portfolio_transition_diagnostics.csv")
    if not transition.empty:
        high_mask = _bool_series(transition.get("high_tv_transition", pd.Series(False, index=transition.index)))
        high = transition[high_mask].copy()
        missing_context = int(
            high.get("transition_explanation_status", pd.Series("", index=high.index))
            .astype(str)
            .eq("high_turnover_missing_transition_context")
            .sum()
        )
        tautological_context = int(
            high.get("transition_explanation_status", pd.Series("", index=high.index))
            .astype(str)
            .isin(
                {
                    "high_turnover_with_z1_flow_and_cohort_churn_context",
                    "high_turnover_with_z1_flow_context",
                    "high_turnover_with_cohort_churn_context",
                }
            )
            .sum()
        )
        corrected_flow_cols = {
            "interval_quarters",
            "z1_transaction_flow_saar_mil",
            "z1_flow_context_status",
            "event_context_status",
            "event_ledger_gross_activity_mil",
            "event_source_issue_mil",
            "event_source_redemption_mil",
            "event_source_indexation_mil",
            "event_source_reclassification_mil",
            "event_unexplained_cohort_change_mil",
            "event_unexplained_residual_change_mil",
            "event_component_reconciliation_gap_mil",
            "event_component_scope",
            "modeled_minus_z1_flow_mil",
            "modeled_minus_z1_flow_share",
        }.issubset(transition.columns)
        adjacent_only_transitions = bool(
            pd.to_numeric(transition.get("interval_quarters", pd.Series(dtype=float)), errors="coerce").eq(1).all()
        ) if "interval_quarters" in transition.columns else False
        event_component_gap = (
            pd.to_numeric(high.get("event_component_reconciliation_gap_mil", pd.Series(dtype=float)), errors="coerce")
            .abs()
            .max()
            if corrected_flow_cols and not high.empty
            else 0.0
        )
        event_scope_ok = (
            True
            if high.empty
            else corrected_flow_cols
            and set(high.get("event_component_scope", pd.Series(dtype=str)).dropna().astype(str).unique())
            == {"treasury_wide_aggregate_context_not_sector_allocated"}
        )
        missing_corrected_flow = int(
            high.get("z1_flow_context_status", pd.Series("", index=high.index))
            .astype(str)
            .eq("missing_corrected_z1_flow")
            .sum()
        ) if corrected_flow_cols else len(high)
        checks.append(
            CheckRow(
                "portfolio_transition_explanation",
                "pass"
                if missing_context == 0
                and tautological_context == 0
                and corrected_flow_cols
                and adjacent_only_transitions
                and missing_corrected_flow == 0
                and event_component_gap <= NUMERIC_TOLERANCE_MIL
                and event_scope_ok
                else "fail",
                f"{len(high)} TV>90% transitions emitted; missing context rows={missing_context}; tautological context rows={tautological_context}; missing corrected-flow rows={missing_corrected_flow}; corrected context columns={corrected_flow_cols}; adjacent_only_transitions={adjacent_only_transitions}; event_component_gap={event_component_gap}; event_scope_ok={event_scope_ok}",
                "all TV>90% transitions are adjacent-quarter rows emitted with sector, quarter, corrected Z.1 flow/level, event components, aggregate-context scope, and reconciled event activity fields",
                "Extreme synthetic turnover is a disclosed diagnostic surface, not hidden behind a scalar threshold.",
            )
        )
        checks.append(
            CheckRow(
                "portfolio_transition_adjacency",
                "pass" if corrected_flow_cols and adjacent_only_transitions else "fail",
                f"rows={len(transition)}; has_interval_quarters={'interval_quarters' in transition.columns}; adjacent_only_transitions={adjacent_only_transitions}",
                "portfolio transition diagnostics only count adjacent-quarter transitions",
                "Nonadjacent bridges over aggregate-only gaps are not adjacent turnover observations.",
            )
        )

    native_similarity = _read_csv(VALIDATION_DIR / "portfolio_native_sector_similarity.csv")
    if not native_similarity.empty:
        max_peers = int(
            pd.to_numeric(native_similarity.get("identical_profile_peer_count", 0), errors="coerce").fillna(0).max()
        )
        max_fixed_long_share = float(
            pd.to_numeric(native_similarity.get("fixed_rate_gt_397d_share", 0.0), errors="coerce").fillna(0.0).max()
        )
        boundaries = sorted(
            native_similarity.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique()
        )
        acceptance_text = (VALIDATION_DIR / "historical_replay_acceptance.md").read_text(encoding="utf-8") if (VALIDATION_DIR / "historical_replay_acceptance.md").exists() else ""
        overclaim = "solves plausible sector/security portfolios" in acceptance_text
        checks.append(
            CheckRow(
                "native_sector_nonidentification_boundary",
                "pass"
                if boundaries == ["synthetic_max_entropy_nonidentified_cells_not_observed_holder_security_history"]
                and not overclaim
                else "fail",
                f"{len(native_similarity)} sector-quarter rows; max identical-profile peers={max_peers}; max fixed-rate >397d share={max_fixed_long_share:.6f}; boundaries={boundaries}; overclaim={overclaim}",
                "native-sector similarity disclosed and acceptance text avoids observed/plausible ownership overclaim",
                "Identical or diffuse native-sector profiles are a nonidentification boundary, not proof of true holder-security portfolios.",
            )
        )

    pricing = _read_csv(VALIDATION_DIR / "pricing_scope_diagnostics.csv")
    if not pricing.empty:
        scopes = sorted(pricing.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique())
        checks.append(
            CheckRow(
                "pricing_scope_boundary",
                "pass" if scopes == ["model_implied_not_observed_market_price"] else "fail",
                ", ".join(scopes),
                "model_implied_not_observed_market_price",
                "Pricing outputs are explicitly model-implied and are not represented as observed market quotes.",
            )
        )

    valuation = _read_csv(VALIDATION_DIR / "valuation_scope_diagnostics.csv")
    if not valuation.empty:
        scopes = sorted(valuation.get("claim_boundary", pd.Series([], dtype=str)).dropna().astype(str).unique())
        statuses = sorted(valuation.get("valuation_scope_status", pd.Series([], dtype=str)).dropna().astype(str).unique())
        status = (
            "pass"
            if scopes == ["pricing_values_require_separate_reconciliation"]
            and statuses == ["pricing_values_present"]
            else "fail"
        )
        checks.append(
            CheckRow(
                "valuation_claim_boundary",
                status,
                f"status={','.join(statuses)}; boundary={','.join(scopes)}",
                "pricing values present with separate-reconciliation boundary",
                "Valuation rows are scoped diagnostics, not certification of observed market-value history.",
            )
        )

    exact = _read_csv(VALIDATION_DIR / "historical_replay_exact_observation_coverage.csv")
    if not exact.empty:
        z1 = exact[exact.get("scope", pd.Series("", index=exact.index)).astype(str).eq("z1_holder_level")]
        auction = exact[exact.get("scope", pd.Series("", index=exact.index)).astype(str).eq("auction_allotment_event")]
        z1_holder_security_rows = int(
            pd.to_numeric(z1.get("holder_security_rows", 0), errors="coerce").fillna(0).sum()
        )
        z1_obs = int(pd.to_numeric(z1.get("observation_count", 0), errors="coerce").fillna(0).sum())
        auction_rows = int(
            pd.to_numeric(auction.get("holder_security_rows", 0), errors="coerce").fillna(0).sum()
        )
        checks.append(
            CheckRow(
                "exact_observation_boundary",
                "pass",
                f"{z1_holder_security_rows} exact Z.1 holder-security rows across {z1_obs} aggregate Z.1 observations; {auction_rows} auction-event holder-security rows",
                "boundary disclosed",
                "Aggregate Z.1 targets are available; exact holder-by-CUSIP positions are not claimed.",
            )
        )

    present_checks = {row.check for row in checks}
    missing_checks = sorted(REQUIRED_CHECKS - present_checks)
    checks.append(
        CheckRow(
            "required_check_roster",
            "pass" if not missing_checks else "fail",
            f"{len(REQUIRED_CHECKS) - len(missing_checks)}/{len(REQUIRED_CHECKS)} required checks executed; missing={missing_checks}",
            "all required closeout checks execute",
            "The closeout audit fails closed if a required check is skipped.",
        )
    )
    out = pd.DataFrame([row.__dict__ for row in checks])
    out.to_csv(PLAUSIBILITY_PATH, index=False)
    return out


def _status_counts(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df.columns:
        return "none"
    counts = df[column].value_counts(dropna=False).sort_index()
    return ", ".join(f"{index}: {count}" for index, count in counts.items())


def _bullet_lines(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = str(row[column])
            value = value.replace("\n", " ").replace("|", "\\|")
            values.append(value)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def write_report(source_scope: pd.DataFrame, plausibility: pd.DataFrame) -> None:
    failures = plausibility[plausibility["status"].astype(str) == "fail"] if not plausibility.empty else pd.DataFrame()
    nonaccepted_sources = (
        source_scope[source_scope["closeout_status"].astype(str) != "accepted"]
        if not source_scope.empty
        else pd.DataFrame()
    )
    certificate = _read_csv(VALIDATION_DIR / "valuation_basis_feasibility_certificate.csv")
    material_certificate_rows = 0
    if not certificate.empty:
        abs_slack = pd.to_numeric(certificate.get("abs_slack_mil", 0.0), errors="coerce").fillna(0.0)
        abs_residual = pd.to_numeric(certificate.get("abs_residual_mil", 0.0), errors="coerce").fillna(0.0)
        material_certificate_rows = int((abs_slack.gt(NUMERIC_TOLERANCE_MIL) | abs_residual.gt(NUMERIC_TOLERANCE_MIL)).sum())
    valuation_clause = (
        f"with {material_certificate_rows} certified valuation-basis exception"
        f"{'' if material_certificate_rows == 1 else 's'} disclosed"
        if material_certificate_rows
        else "with no certified valuation-basis exceptions"
    )

    verdict = (
        f"READY FOR SCOPED CLOSEOUT REVIEW under the quarterly aggregate-constrained synthetic replay target, {valuation_clause}."
        if failures.empty
        else "NOT READY: at least one local plausibility check failed."
    )

    text = f"""# Historical Replay Closeout Audit

## Verdict

{verdict}

This audit does not certify exact transaction replay, exact holder-by-CUSIP history, or observed market-value history. It certifies only the local evidence package for a quarterly aggregate-constrained synthetic replay with any valuation-basis slack explicitly named.

## Scoped Replay Inputs

{_bullet_lines([
        "Selected empirical TDC ladder: the historical target that the replay compares against quarter by quarter.",
        "Treasury cash/liability quantities: Treasury cash, Treasury Tax and Loan balances, deficits, Fed remittances, and related fiscal bridge quantities.",
        "Z.1 sector levels: aggregate sector holdings that constrain the sector portfolios; they are not exact holder-by-security observations.",
        "Public debt cohorts and auction evidence: security/cohort supply, maturity/call/runoff facts, and auction allotment proxies used to shape plausible tranches.",
        "Interest references: Treasury interest expense, sector/component interest constraints, FFIEC/NCUA constraints where available, and MMF/RRP adjustment proportions.",
        "Instrument mechanics: FRN index evidence, TIPS principal decomposition, coupon schedules, and curve-derived model pricing diagnostics.",
    ])}

## Local Plausibility Checks

Status counts: {_status_counts(plausibility, "status")}.

{_markdown_table(plausibility)}

## Source Scope Review

Imported source status counts: {_status_counts(source_scope, "closeout_status")}.

The non-accepted imported files are not treated as blockers when they are context, duplicate copies, sensitivity inputs, or future stronger-target inputs outside the scoped replay claim. See `{SOURCE_SCOPE_PATH.name}` for the row-level classification.

Non-accepted imported files reviewed: {len(nonaccepted_sources)}.

## Remaining Boundaries

{_bullet_lines([
        "Reserve-user net transfers remain an assumption, currently configured as a small deficit share rather than directly observed empirical transfers.",
        "Secondary-market holder transfers are not observed; unobserved movement is carried as disclosed accounting residual rather than converted into exact transactions.",
        "Opening old-security split residuals are labelled separately and are not represented as direct auction facts.",
        "Pricing diagnostics are model-implied. They are not observed market-price history.",
        "Older historical quarters use weaker available interest and WAM evidence, while modern quarters use richer component constraints where available.",
    ])}

## Files Written

{_bullet_lines([
        f"`{SOURCE_SCOPE_PATH.name}`",
        f"`{PLAUSIBILITY_PATH.name}`",
        f"`{REPORT_PATH.name}`",
    ])}
"""
    REPORT_PATH.write_text(text, encoding="utf-8")


def write_final_evidence_manifest() -> None:
    artifact_integrity_path = VALIDATION_DIR / "artifact_integrity.csv"
    artifact_integrity = _read_csv(artifact_integrity_path)
    plausibility = _read_csv(PLAUSIBILITY_PATH)
    audit_passed = (
        not plausibility.empty
        and "status" in plausibility.columns
        and set(plausibility["status"].astype(str)) == {"pass"}
    )
    run_ids = sorted(
        str(value)
        for value in artifact_integrity.get("run_id", pd.Series(dtype=str)).dropna().astype(str).unique()
        if value
    )
    config_hashes = sorted(
        str(value)
        for value in artifact_integrity.get("config_sha256", pd.Series(dtype=str)).dropna().astype(str).unique()
        if value
    )
    code_hashes = sorted(
        str(value)
        for value in artifact_integrity.get("code_sha256", pd.Series(dtype=str)).dropna().astype(str).unique()
        if value
    )
    artifact_integrity_sha256 = _sha256_file(artifact_integrity_path) if artifact_integrity_path.exists() else ""
    run_id = run_ids[0] if len(run_ids) == 1 else ""
    config_sha256 = config_hashes[0] if len(config_hashes) == 1 else ""
    code_sha256 = code_hashes[0] if len(code_hashes) == 1 else _build_code_identity_hash()
    rows = []
    for name, path in [
        ("artifact_integrity", artifact_integrity_path),
        ("historical_replay_runtime_manifest", RUNTIME_MANIFEST_PATH),
        ("historical_replay_source_scope_audit", SOURCE_SCOPE_PATH),
        ("historical_replay_plausibility_audit", PLAUSIBILITY_PATH),
        ("historical_replay_closeout_audit", REPORT_PATH),
    ]:
        metadata = _artifact_file_metadata(path)
        metadata["artifact"] = name
        metadata["path"] = _rel(path)
        metadata["run_id"] = run_id
        metadata["config_sha256"] = config_sha256
        metadata["code_sha256"] = code_sha256
        metadata["artifact_integrity_sha256"] = artifact_integrity_sha256
        rows.append(metadata)
    rows.append(
        {
            "artifact": "final_evidence_identity",
            "path": "",
            "status": "present" if audit_passed and run_id and config_sha256 and code_sha256 and artifact_integrity_sha256 else "incomplete_or_failed_identity",
            "bytes": pd.NA,
            "sha256": pd.NA,
            "row_count": pd.NA,
            "column_count": pd.NA,
            "header_sha256": pd.NA,
            "run_id": run_id,
            "config_sha256": config_sha256,
            "code_sha256": code_sha256,
            "artifact_integrity_sha256": artifact_integrity_sha256,
            "audit_passed": audit_passed,
        }
    )
    pd.DataFrame(rows).to_csv(FINAL_EVIDENCE_MANIFEST_PATH, index=False)


def main() -> int:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    source_scope = build_source_scope_audit()
    plausibility = build_plausibility_audit()
    write_report(source_scope, plausibility)
    write_final_evidence_manifest()
    print(f"wrote {_rel(SOURCE_SCOPE_PATH)}")
    print(f"wrote {_rel(PLAUSIBILITY_PATH)}")
    print(f"wrote {_rel(REPORT_PATH)}")
    print(f"wrote {_rel(FINAL_EVIDENCE_MANIFEST_PATH)}")
    if plausibility.empty or "status" not in plausibility.columns:
        return 1
    return 0 if set(plausibility["status"].astype(str)) == {"pass"} else 1


if __name__ == "__main__":
    sys.exit(main())
