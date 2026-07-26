"""Marginal TDC pair assembler for RateWall-facing CBO runs.

The pair lane deliberately wraps two ordinary CBO scenario runs. It does not
change the engine or treat a gross single-run TDC exposure as RateWall support.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ._json import canonical_json_sha256, read_json, sha256_file, write_json
from ._schema import validate_schema


PAIR_SCHEMA_VERSION = "tdcsim_cbo_marginal_tdc_pair_v1"
MANIFEST_SCHEMA_VERSION = "tdcsim_cbo_marginal_tdc_manifest_v1"
OBJECT_ID = "RW_M_PLUS_100BP_YEAR"
SHOCK_PATH_ID = "plus_100bp_year"
FISCAL_INJECTION_OBJECT_ID = "TDC_FISCAL_INJECTION_2028"
FISCAL_INJECTION_SHOCK_PATH_ID = "fiscal_injection_2028_v1"
DENOMINATOR_EQUIVALENCE_KEY = "ratewall_D_conv_plus_100bp_year_v1"
FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY = "tdc_fiscal_injection_2028_no_rate_shock_v1"
CLAIM_BOUNDARY = "tdcsim_marginal_pair_assumption_mode_not_evidence_not_channel_classifier"
SUMMARY_FILE = "tdcsim_ratewall_marginal_tdc_summary.csv"
COMPONENTS_FILE = "tdcsim_ratewall_marginal_tdc_components.csv"
ROUTE_METADATA_FILE = "tdcsim_ratewall_marginal_tdc_route_metadata.csv"
MANIFEST_FILE = "tdcsim_ratewall_marginal_tdc_pair_manifest.json"
STATE_MANIFEST_FILE = "tdcsim_ratewall_scenario_state_manifest.json"
CONTRACT_VERSION = "0.4.0"
TDC_AMOUNT_BASIS = "pre_beta_ex_overlap_delta"
TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION = "tdc_deposit_creation_split_v1"
TDC_INCOME_ADDENDUM_FULL_LEVEL_RATE = 0.035
TDC_INCOME_ADDENDUM_ROUTE_FAMILY = "tdc_income_from_tdcsim_marginal_deposit_stock"
TDC_INCOME_ADDENDUM_ADMISSION_STATUS = "admitted_split_non_interest_bucket"
TDC_INCOME_ADDENDUM_COLLISION_STATUS = "pass_split_collision_excluded"
TDC_SELECTED_SUPPORT_FORMULA = "admissible \u00d7 \u03b2 \u00d7 rate \u00d7 sfc_route_coefficients"
TDC_CHI_SELECTED_STATUS = "retired_not_selected"
INTEREST_PAYMENT_TYPES = {"bill_discount", "fixed_coupon", "frn_interest", "tips_coupon"}
SPLIT_DRIVER_BUCKETS = {
    "interest_driven_direct_interest_collision_excluded",
    "non_interest_auction_absorption_admissible",
    "non_interest_principal_redemption_admissible",
    "non_interest_fiscal_flow_admissible",
    "route_plumbing_memo_excluded",
    "not_in_delta_tdc_ex_overlap_excluded",
}

ALLOWED_RATE_INPUT_NAMES = {
    "tdcsim_yield_curve_surface.csv",
    "tdcsim_frn_rate_path.csv",
    "tdcsim_tips_real_yield_path.csv",
}
FORBIDDEN_NON_RATE_INPUT_NAMES = {
    "tdcsim_primary_deficit_path.csv",
    "tdcsim_debt_stock_path.csv",
    "tdcsim_operating_cash_path.csv",
    "tdcsim_cash_reconciliation_residual.csv",
    "tdcsim_fed_holdings_path.csv",
    "tdcsim_holder_profile_assumptions.csv",
    "tdcsim_holder_preference_events.csv",
    "tdcsim_opening_portfolio.csv",
    "tdcsim_fiscal_incidence_policy.csv",
}
ALLOWED_FISCAL_INJECTION_INPUT_NAMES = {
    "tdcsim_primary_deficit_path.csv",
    "tdcsim_debt_stock_path.csv",
    "tdcsim_fed_holdings_path.csv",
    "tdcsim_issuance_mix_assumptions.json",
}
REQUIRED_SUMMARY_COLUMNS = (
    "period_start",
    "period_end",
    "tdc_change_bil",
    "overlap_cashflow_bil",
    "tdc_change_ex_overlap_bil",
)


class MarginalTdcPairError(ValueError):
    """Raised when a marginal TDC pair cannot be assembled or verified."""


@dataclass(frozen=True)
class MarginalPairResult:
    """Written marginal pair artifact paths."""

    output_dir: Path
    summary_path: Path
    components_path: Path
    route_metadata_path: Path
    state_manifest_path: Path
    manifest_path: Path


def assemble_marginal_tdc_pair(pair_spec: str | Path | Mapping[str, Any], output_dir: str | Path) -> MarginalPairResult:
    """Assemble RateWall marginal TDC files from two ordinary CBO run directories."""

    spec = _load_pair_spec(pair_spec)
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise MarginalTdcPairError(f"marginal pair output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    baseline = _load_run("baseline", spec["baseline_run_dir"])
    shock = _load_run("shock", spec["shock_run_dir"])
    checks = _validate_pair_inputs(spec, baseline, shock)
    cases = _demand_conversion_cases(spec)

    baseline_summary = _load_tdc_summary(baseline)
    shock_summary = _load_tdc_summary(shock)
    components = _assemble_components(spec, baseline, shock, cases)
    summary = _assemble_summary(spec, baseline, shock, baseline_summary, shock_summary, cases, components)
    route_metadata = _assemble_route_metadata(spec, baseline, shock)
    state_manifest = _assemble_state_manifest(spec)

    summary_path = out / SUMMARY_FILE
    components_path = out / COMPONENTS_FILE
    route_metadata_path = out / ROUTE_METADATA_FILE
    state_manifest_path = out / STATE_MANIFEST_FILE
    summary.to_csv(summary_path, index=False)
    components.to_csv(components_path, index=False)
    route_metadata.to_csv(route_metadata_path, index=False)
    write_json(state_manifest_path, state_manifest)

    manifest = _build_pair_manifest(spec, baseline, shock, checks, out)
    manifest_path = out / MANIFEST_FILE
    write_json(manifest_path, manifest)
    verify_marginal_tdc_pair(out)
    return MarginalPairResult(out, summary_path, components_path, route_metadata_path, state_manifest_path, manifest_path)


def verify_marginal_tdc_pair(pair_dir: str | Path) -> dict[str, Any]:
    """Verify a written marginal TDC pair package and fail closed on identity drift."""

    root = Path(pair_dir).expanduser().resolve()
    manifest = _read_pair_manifest(root / MANIFEST_FILE)
    _verify_pair_files(root, manifest)
    spec = manifest.get("pair_spec", {})
    if not isinstance(spec, Mapping):
        raise MarginalTdcPairError("pair manifest pair_spec must be an object")
    baseline = _load_run("baseline", spec["baseline_run_dir"])
    shock = _load_run("shock", spec["shock_run_dir"])
    checks = _validate_pair_inputs(spec, baseline, shock)
    _verify_manifest_checks(manifest, checks)

    summary = pd.read_csv(root / SUMMARY_FILE)
    components = pd.read_csv(root / COMPONENTS_FILE)
    route_metadata = pd.read_csv(root / ROUTE_METADATA_FILE)
    state_manifest = _read_state_manifest(root / STATE_MANIFEST_FILE)
    _verify_state_manifest(state_manifest, spec)
    _verify_summary(summary)
    _verify_components(components, summary)
    _verify_route_metadata(route_metadata, baseline.manifest["run_id"], shock.manifest["run_id"])
    return {"status": "pass", "pair_id": spec["pair_id"], "rows": int(len(summary))}


def _load_pair_spec(pair_spec: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(pair_spec, Mapping):
        data = dict(pair_spec)
    else:
        data = read_json(pair_spec)
    if not isinstance(data, dict):
        raise MarginalTdcPairError("marginal pair spec must be a JSON object")
    with files("tdcsim_cbo").joinpath("schemas/cbo-marginal-tdc-pair-v1.schema.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        validate_schema(data, json.load(handle), label="marginal_pair")
    if data.get("schema_version") != PAIR_SCHEMA_VERSION:
        raise MarginalTdcPairError("unsupported marginal pair schema_version")
    return data


@dataclass(frozen=True)
class _RunBundle:
    role: str
    root: Path
    manifest: dict[str, Any]
    scenario: dict[str, Any]


def _load_run(role: str, run_dir: str | Path) -> _RunBundle:
    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "tdcsim_cbo_run_manifest.json"
    if not manifest_path.exists():
        raise MarginalTdcPairError(f"{role} run manifest is missing")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise MarginalTdcPairError(f"{role} run manifest must be an object")
    scenario_rel = Path(str(manifest.get("scenario", {}).get("relative_path") or "scenario.json"))
    scenario_path = root / scenario_rel
    if not scenario_path.exists():
        raise MarginalTdcPairError(f"{role} scenario copy is missing")
    scenario = read_json(scenario_path)
    if not isinstance(scenario, dict):
        raise MarginalTdcPairError(f"{role} scenario copy must be an object")
    _verify_run_artifacts(role, root, manifest)
    return _RunBundle(role=role, root=root, manifest=manifest, scenario=scenario)


def _validate_pair_inputs(spec: Mapping[str, Any], baseline: _RunBundle, shock: _RunBundle) -> dict[str, str]:
    shock_path_id = str(spec.get("shock_path_id") or "")
    object_id = str(spec.get("object_id") or "")
    if shock_path_id == SHOCK_PATH_ID:
        if object_id != OBJECT_ID:
            raise MarginalTdcPairError("marginal pair object_id is unsupported for +100bp-year shock path")
        if float(spec.get("shock_bps_year")) != 100.0:
            raise MarginalTdcPairError("marginal pair shock_bps_year must equal 100")
        if spec.get("denominator_equivalence_key") != DENOMINATOR_EQUIVALENCE_KEY:
            raise MarginalTdcPairError("marginal pair denominator_equivalence_key is unsupported")
    elif shock_path_id == FISCAL_INJECTION_SHOCK_PATH_ID:
        if object_id != FISCAL_INJECTION_OBJECT_ID:
            raise MarginalTdcPairError("fiscal injection pair object_id is unsupported")
        if float(spec.get("shock_bps_year")) != 0.0:
            raise MarginalTdcPairError("fiscal injection pair shock_bps_year must equal 0")
        if spec.get("denominator_equivalence_key") != FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY:
            raise MarginalTdcPairError("fiscal injection denominator_equivalence_key is unsupported")
    else:
        raise MarginalTdcPairError("unsupported marginal pair shock_path_id")
    _require_state_manifest_spec(spec)
    _require_scenario_id("baseline", baseline, spec.get("baseline_scenario_id"))
    _require_scenario_id("shock", shock, spec.get("shock_scenario_id"))
    _require_same_baseline(baseline.manifest, shock.manifest)
    _require_same_simulation(spec, baseline.manifest, shock.manifest)
    _require_metadata_match(spec, baseline.manifest, shock.manifest)
    if shock_path_id == SHOCK_PATH_ID:
        _require_rate_shock_only(baseline, shock)
        _require_path_area(spec, baseline, shock)
    else:
        _require_fiscal_injection_shock_only(baseline, shock)
    _require_forecast_state_construction(spec, baseline, shock)
    _require_non_rate_digest_match(spec, baseline, shock, allow_drift=shock_path_id == FISCAL_INJECTION_SHOCK_PATH_ID)
    _require_source_grade_labels(spec)
    return {
        "same_state_status": "pass",
        "rate_shock_only_status": "pass" if shock_path_id == SHOCK_PATH_ID else "not_applicable_fiscal_injection_no_rate_shock",
        "shock_path_validation_status": "pass",
        "period_alignment_status": "pass",
        "overlap_identity_status": "pass",
        "component_identity_status": "pass",
        "route_identity_status": "pass",
        "support_identity_status": "pass",
        "state_manifest_status": "pass",
        "contract_ingest_status": "ready_for_ratewall_assumption_mode_ingest",
    }


def _require_state_manifest_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("scenario_state_set_id") in (None, ""):
        raise MarginalTdcPairError("marginal pair requires scenario_state_set_id")
    fingerprint = str(spec.get("state_fingerprint_sha256") or "")
    component_inventory = str(spec.get("state_component_inventory_sha256") or "")
    baseline_fingerprint = str(spec.get("baseline_state_fingerprint_sha256") or fingerprint)
    shock_fingerprint = str(spec.get("shock_state_fingerprint_sha256") or fingerprint)
    for label, value in (
        ("state_fingerprint_sha256", fingerprint),
        ("state_component_inventory_sha256", component_inventory),
        ("baseline_state_fingerprint_sha256", baseline_fingerprint),
        ("shock_state_fingerprint_sha256", shock_fingerprint),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise MarginalTdcPairError(f"{label} must be a SHA256 hex digest")
    if baseline_fingerprint != shock_fingerprint:
        raise MarginalTdcPairError("baseline and shock state fingerprints must match within pair")
    if fingerprint != baseline_fingerprint:
        raise MarginalTdcPairError("pair state_fingerprint_sha256 must match run fingerprints")


def _require_scenario_id(role: str, run: _RunBundle, expected: Any) -> None:
    observed = str(run.manifest.get("scenario", {}).get("scenario_id") or "")
    if expected is not None and observed != str(expected):
        raise MarginalTdcPairError(f"{role} scenario_id does not match pair spec")


def _require_same_baseline(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    if left.get("baseline") != right.get("baseline"):
        raise MarginalTdcPairError("baseline and shock runs do not share the same baseline hashes")


def _require_same_simulation(spec: Mapping[str, Any], baseline: Mapping[str, Any], shock: Mapping[str, Any]) -> None:
    b_sim = baseline.get("simulation", {})
    s_sim = shock.get("simulation", {})
    if b_sim != s_sim:
        raise MarginalTdcPairError("baseline and shock runs do not share the same simulation dates")
    if str(b_sim.get("start_date") or "") != str(spec.get("horizon_start_date") or ""):
        raise MarginalTdcPairError("pair horizon_start_date does not match run simulation start_date")
    if str(b_sim.get("end_date") or "") != str(spec.get("horizon_end_date") or ""):
        raise MarginalTdcPairError("pair horizon_end_date does not match run simulation end_date")
    if str(spec.get("opening_state_date") or "") != str(b_sim.get("start_date") or ""):
        raise MarginalTdcPairError("opening_state_date must match the paired run start_date")


def _require_metadata_match(spec: Mapping[str, Any], baseline: Mapping[str, Any], shock: Mapping[str, Any]) -> None:
    b_meta = _row_metadata(baseline)
    s_meta = _row_metadata(shock)
    for field in ("actuals_available_as_of", "source_vintage"):
        if str(b_meta.get(field) or "") != str(s_meta.get(field) or ""):
            raise MarginalTdcPairError(f"baseline and shock metadata differ: {field}")
        if str(spec.get(field) or "") != str(b_meta.get(field) or ""):
            raise MarginalTdcPairError(f"pair spec {field} does not match run metadata")


def _require_rate_shock_only(baseline: _RunBundle, shock: _RunBundle) -> None:
    b_inputs = _compiled_input_map(baseline.manifest)
    s_inputs = _compiled_input_map(shock.manifest)
    if set(b_inputs) != set(s_inputs):
        raise MarginalTdcPairError("baseline and shock compiled input inventory differs")
    differing: set[str] = set()
    for logical_name, b_artifact in b_inputs.items():
        s_artifact = s_inputs[logical_name]
        if b_artifact.get("sha256") != s_artifact.get("sha256"):
            name = Path(logical_name).name
            differing.add(name)
            if name not in ALLOWED_RATE_INPUT_NAMES:
                raise MarginalTdcPairError(f"non-rate compiled input drift is forbidden: {logical_name}")
            if name in FORBIDDEN_NON_RATE_INPUT_NAMES:
                raise MarginalTdcPairError(f"forbidden non-rate compiled input drift: {logical_name}")
    if "tdcsim_yield_curve_surface.csv" not in differing:
        raise MarginalTdcPairError("shock run must differ from baseline in the nominal yield curve surface")
    b_overrides = baseline.scenario.get("overrides", {})
    s_overrides = shock.scenario.get("overrides", {})
    if isinstance(b_overrides, Mapping) and any(key in b_overrides for key in ("nominal_yield_curve", "frn_benchmark", "tips_real_yield")):
        raise MarginalTdcPairError("baseline scenario must not declare an incremental rate shock")
    if not isinstance(s_overrides, Mapping):
        raise MarginalTdcPairError("shock scenario overrides must be an object")
    b_non_rate = {key: value for key, value in b_overrides.items()} if isinstance(b_overrides, Mapping) else {}
    s_non_rate = {
        key: value
        for key, value in s_overrides.items()
        if key not in {"nominal_yield_curve", "frn_benchmark", "tips_real_yield"}
    }
    if b_non_rate != s_non_rate:
        raise MarginalTdcPairError("shock scenario non-rate overrides must match baseline scenario")
    nominal = s_overrides.get("nominal_yield_curve")
    if not isinstance(nominal, Mapping) or nominal.get("mode") != "full_surface_file":
        raise MarginalTdcPairError("shock scenario must declare nominal_yield_curve full_surface_file")


def _require_fiscal_injection_shock_only(baseline: _RunBundle, shock: _RunBundle) -> None:
    b_inputs = _compiled_input_map(baseline.manifest)
    s_inputs = _compiled_input_map(shock.manifest)
    differing: set[str] = set()
    for logical_name in sorted(set(b_inputs) | set(s_inputs)):
        b_artifact = b_inputs.get(logical_name)
        s_artifact = s_inputs.get(logical_name)
        name = Path(logical_name).name
        if b_artifact is None or s_artifact is None:
            if name != "tdcsim_issuance_mix_assumptions.json":
                raise MarginalTdcPairError(f"fiscal injection compiled input inventory drift is unsupported: {logical_name}")
            differing.add(name)
            continue
        if b_artifact.get("sha256") == s_artifact.get("sha256"):
            continue
        differing.add(name)
        if name in ALLOWED_RATE_INPUT_NAMES:
            raise MarginalTdcPairError(f"fiscal injection shock must not change rate input: {logical_name}")
        if name not in ALLOWED_FISCAL_INJECTION_INPUT_NAMES:
            raise MarginalTdcPairError(f"fiscal injection shock changed unsupported input: {logical_name}")
    missing = sorted(ALLOWED_FISCAL_INJECTION_INPUT_NAMES - differing)
    if missing:
        raise MarginalTdcPairError(f"fiscal injection shock missing required input changes: {missing}")
    b_overrides = baseline.scenario.get("overrides", {})
    s_overrides = shock.scenario.get("overrides", {})
    if isinstance(b_overrides, Mapping) and b_overrides:
        raise MarginalTdcPairError("fiscal injection baseline scenario must not declare overrides")
    if not isinstance(s_overrides, Mapping):
        raise MarginalTdcPairError("fiscal injection shock scenario overrides must be an object")
    expected = {"primary_deficit", "debt_target", "fed_holdings", "issuance_mix"}
    if set(s_overrides) != expected:
        raise MarginalTdcPairError(f"fiscal injection shock overrides must be exactly {sorted(expected)}")


def _require_path_area(spec: Mapping[str, Any], baseline: _RunBundle, shock: _RunBundle) -> None:
    base = _read_compiled_csv(baseline, "tdcsim_yield_curve_surface.csv")
    shocked = _read_compiled_csv(shock, "tdcsim_yield_curve_surface.csv")
    required = {"curve_date", "tenor_years"}
    if not required <= set(base.columns) or not required <= set(shocked.columns):
        raise MarginalTdcPairError("yield curve surface must include curve_date and tenor_years")
    b = base.copy()
    s = shocked.copy()
    b["_key_date"] = pd.to_datetime(b["curve_date"], errors="coerce")
    s["_key_date"] = pd.to_datetime(s["curve_date"], errors="coerce")
    b["_tenor"] = pd.to_numeric(b["tenor_years"], errors="coerce")
    s["_tenor"] = pd.to_numeric(s["tenor_years"], errors="coerce")
    b["_rate"] = _surface_rate_decimal(b)
    s["_rate"] = _surface_rate_decimal(s)
    horizon_end = pd.Timestamp(str(spec["horizon_end_date"]))
    shock_start = pd.Timestamp(str(spec["horizon_start_date"]))
    shock_end = shock_start + pd.DateOffset(years=1)
    shock_year_days = max(1, (min(horizon_end, shock_end) - shock_start).days)
    per_date = _step_function_curve_deltas(b, s, horizon_end)
    dates = list(per_date["_key_date"])
    area = 0.0
    for idx, date_value in enumerate(dates):
        next_date = dates[idx + 1] if idx + 1 < len(dates) else horizon_end
        days = max(0, (min(next_date, horizon_end) - date_value).days)
        delta = float(per_date.loc[idx, "delta_bp"])
        expected = 100.0 if shock_start <= date_value < shock_end else 0.0
        if abs(delta - expected) > 1e-6:
            raise MarginalTdcPairError("shock path is not +100bp during the one-year window and baseline thereafter")
        area += delta * days / shock_year_days
    if abs(area - 100.0) > 1e-6:
        raise MarginalTdcPairError(f"shock path area is not 100 bp-years: {area:.12f}")


def _require_forecast_state_construction(spec: Mapping[str, Any], baseline: _RunBundle, shock: _RunBundle) -> None:
    if spec.get("state_kind") != "forecast_state":
        return
    if spec.get("state_construction_method") != "baseline_rollforward_export_v1":
        raise MarginalTdcPairError("forecast source-grade pair requires baseline_rollforward_export_v1 construction")
    for field in (
        "forecast_state_export_manifest_sha256",
        "derived_state_package_sha256",
        "parent_baseline_package_sha256",
        "parent_baseline_manifest_sha256",
        "parent_attestation_sha256",
        "rollforward_run_manifest_sha256",
    ):
        _require_hex_digest(spec, field)
    derived = str(spec["derived_state_package_sha256"])
    parent = str(spec["parent_baseline_package_sha256"])
    if derived == parent:
        raise MarginalTdcPairError("forecast pair must use a derived state package, not the parent package directly")
    for role, run in (("baseline", baseline), ("shock", shock)):
        package_sha = str(run.manifest.get("baseline", {}).get("package_sha256") or "")
        if package_sha != derived:
            raise MarginalTdcPairError(f"{role} run does not use the derived forecast-state package")


def _require_non_rate_digest_match(
    spec: Mapping[str, Any],
    baseline: _RunBundle,
    shock: _RunBundle,
    *,
    allow_drift: bool = False,
) -> None:
    expected = str(spec.get("compiled_non_rate_inputs_digest") or "")
    expected_is_digest = bool(expected) and len(expected) == 64 and not any(char not in "0123456789abcdef" for char in expected.lower())
    if spec.get("state_kind") == "forecast_state" and not expected_is_digest:
        raise MarginalTdcPairError("forecast compiled_non_rate_inputs_digest must be a SHA256 hex digest")
    b_digest = _non_rate_compiled_inputs_digest(baseline.manifest)
    s_digest = _non_rate_compiled_inputs_digest(shock.manifest)
    if allow_drift:
        if b_digest == s_digest:
            raise MarginalTdcPairError("fiscal injection pair requires non-rate compiled input drift")
        if expected_is_digest and expected != b_digest:
            raise MarginalTdcPairError("pair spec compiled_non_rate_inputs_digest does not match baseline run")
        return
    if b_digest != s_digest:
        raise MarginalTdcPairError("baseline and shock non-rate compiled input digests differ")
    if expected_is_digest and expected != b_digest:
        raise MarginalTdcPairError("pair spec compiled_non_rate_inputs_digest does not match runs")


def _require_source_grade_labels(spec: Mapping[str, Any]) -> None:
    if spec.get("state_kind") != "forecast_state":
        return
    if spec.get("source_grade_status") != "pass_forecast_rollforward_source_grade":
        raise MarginalTdcPairError("forecast source-grade pair requires pass_forecast_rollforward_source_grade")
    if str(spec.get("source_vintage") or "") == "ratewall_assumption_mode_fixture_20260630":
        raise MarginalTdcPairError("forecast source-grade pair cannot use assumption fixture vintage")
    for field in ("pair_id", "baseline_scenario_id", "shock_scenario_id"):
        if "source_grade" not in str(spec.get(field) or "").lower():
            raise MarginalTdcPairError(f"forecast source-grade pair missing source_grade label: {field}")


def _require_hex_digest(spec: Mapping[str, Any], field: str) -> None:
    value = str(spec.get(field) or "")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise MarginalTdcPairError(f"{field} must be a SHA256 hex digest")


def _non_rate_compiled_inputs_digest(manifest: Mapping[str, Any]) -> str:
    records = []
    for item in manifest.get("compiled_inputs", []):
        if not isinstance(item, Mapping):
            continue
        name = Path(str(item.get("logical_name") or item.get("relative_path") or "")).name
        if name in ALLOWED_RATE_INPUT_NAMES:
            continue
        records.append(
            {
                "logical_name": str(item.get("logical_name") or ""),
                "bytes": int(item.get("bytes", 0) or 0),
                "sha256": str(item.get("sha256") or ""),
            }
        )
    return canonical_json_sha256(sorted(records, key=lambda row: row["logical_name"]))


def _step_function_curve_deltas(
    baseline: pd.DataFrame,
    shock: pd.DataFrame,
    horizon_end: pd.Timestamp,
) -> pd.DataFrame:
    tenors = sorted(set(baseline["_tenor"].dropna()) & set(shock["_tenor"].dropna()))
    if not tenors:
        raise MarginalTdcPairError("yield curve surfaces have no shared tenors")
    dates = sorted(
        {
            pd.Timestamp(value)
            for value in baseline["_key_date"].dropna()
        }
        | {
            pd.Timestamp(value)
            for value in shock["_key_date"].dropna()
        }
        | {horizon_end}
    )
    rows: list[dict[str, Any]] = []
    for date_value in dates:
        if date_value >= horizon_end:
            continue
        deltas = []
        for tenor in tenors:
            base_rate = _step_rate_at(baseline, date_value, tenor, label="baseline")
            shock_rate = _step_rate_at(shock, date_value, tenor, label="shock")
            deltas.append((shock_rate - base_rate) * 10000.0)
        if max(deltas) - min(deltas) > 1e-7:
            raise MarginalTdcPairError(
                "shock path must be parallel across the full nominal curve surface"
            )
        rows.append({"_key_date": date_value, "delta_bp": max(deltas)})
    return pd.DataFrame(rows).sort_values("_key_date").reset_index(drop=True)


def _step_rate_at(
    frame: pd.DataFrame,
    date_value: pd.Timestamp,
    tenor: float,
    *,
    label: str,
) -> float:
    candidates = frame[
        (frame["_tenor"] == tenor)
        & (frame["_key_date"] <= date_value)
    ].sort_values("_key_date")
    if candidates.empty:
        raise MarginalTdcPairError(
            f"{label} yield curve surface has no rate for tenor {tenor} at {date_value.date()}"
        )
    return float(candidates.iloc[-1]["_rate"])


def _surface_rate_decimal(frame: pd.DataFrame) -> pd.Series:
    if "nominal_rate_decimal" in frame.columns:
        return pd.to_numeric(frame["nominal_rate_decimal"], errors="coerce")
    if "nominal_rate" in frame.columns:
        return pd.to_numeric(frame["nominal_rate"], errors="coerce") / 100.0
    raise MarginalTdcPairError("yield curve surface must include nominal_rate_decimal or nominal_rate")


def _assemble_summary(
    spec: Mapping[str, Any],
    baseline: _RunBundle,
    shock: _RunBundle,
    baseline_summary: pd.DataFrame,
    shock_summary: pd.DataFrame,
    cases: list[dict[str, Any]],
    components: pd.DataFrame,
) -> pd.DataFrame:
    for label, frame in (("baseline", baseline_summary), ("shock", shock_summary)):
        missing = [column for column in REQUIRED_SUMMARY_COLUMNS if column not in frame.columns]
        if missing:
            raise MarginalTdcPairError(f"{label} TDC summary missing required columns: {missing}")
        _verify_tdc_summary_identities(label, frame)
    baseline_summary = _collapse_summary_to_ratewall_horizon(baseline_summary, spec)
    shock_summary = _collapse_summary_to_ratewall_horizon(shock_summary, spec)
    merged = baseline_summary.merge(
        shock_summary,
        on=["period_start", "period_end"],
        how="inner",
        suffixes=("_baseline", "_shock"),
    )
    if len(merged) != len(baseline_summary) or len(merged) != len(shock_summary):
        raise MarginalTdcPairError("baseline and shock TDC summary periods are not aligned")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        base_tdc = _num(row, "tdc_change_bil_baseline")
        shock_tdc = _num(row, "tdc_change_bil_shock")
        base_overlap = _num(row, "overlap_cashflow_bil_baseline")
        shock_overlap = _num(row, "overlap_cashflow_bil_shock")
        base_ex = _num(row, "tdc_change_ex_overlap_bil_baseline")
        shock_ex = _num(row, "tdc_change_ex_overlap_bil_shock")
        delta_tdc = shock_tdc - base_tdc
        delta_overlap = shock_overlap - base_overlap
        delta_ex = shock_ex - base_ex
        if abs(delta_ex - (delta_tdc - delta_overlap)) > 1e-7:
            raise MarginalTdcPairError("marginal ex-overlap delta identity failed")
        for case in cases:
            beta = float(case["beta"])
            chi = float(case["chi"])
            period = spec.get("ratewall_period", row["period_end"])
            split = _component_split_for_summary(
                components,
                period=period,
                horizon=spec.get("horizon", "annual_h1_100bp_year"),
                demand_conversion_case=case["demand_conversion_case"],
                delta_tdc_ex_overlap_bil=delta_ex,
            )
            admissible = split["non_interest_admissible"]
            interest_excluded = split["interest_driven_excluded"]
            reconciled = split["reconciled"]
            remainder = split["remainder"]
            admissible_stock = admissible * beta
            interest_excluded_stock = interest_excluded * beta
            gross_income = admissible_stock * TDC_INCOME_ADDENDUM_FULL_LEVEL_RATE
            rows.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "tdc_deposit_creation_split_schema_version": TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION,
                    "contract_version": CONTRACT_VERSION,
                    "pair_id": spec["pair_id"],
                    "scenario_state_set_id": spec["scenario_state_set_id"],
                    "object_id": spec["object_id"],
                    "state_id": spec["state_id"],
                    "state_kind": spec["state_kind"],
                    "state_period": spec["state_period"],
                    "scenario_id": spec.get("scenario_id", spec["baseline_scenario_id"]),
                    "opening_state_date": spec["opening_state_date"],
                    "actuals_available_as_of": spec["actuals_available_as_of"],
                    "source_vintage": spec["source_vintage"],
                    "source_grade_status": spec.get("source_grade_status", ""),
                    "state_construction_method": spec.get("state_construction_method", ""),
                    "forecast_state_export_manifest_sha256": spec.get("forecast_state_export_manifest_sha256", ""),
                    "derived_state_package_sha256": spec.get("derived_state_package_sha256", ""),
                    "parent_baseline_package_sha256": spec.get("parent_baseline_package_sha256", ""),
                    "rollforward_run_manifest_sha256": spec.get("rollforward_run_manifest_sha256", ""),
                    "compiled_non_rate_inputs_digest": spec.get("compiled_non_rate_inputs_digest", ""),
                    "state_fingerprint_sha256": spec["state_fingerprint_sha256"],
                    "state_component_inventory_sha256": spec["state_component_inventory_sha256"],
                    "shock_path_id": spec["shock_path_id"],
                    "shock_bps_year": spec["shock_bps_year"],
                    "shock_start_date": spec["horizon_start_date"],
                    "shock_end_date": spec["horizon_end_date"],
                    "shock_horizon_years": 1,
                    "shock_path_area_bp_years": 100,
                    "denominator_equivalence_key": spec["denominator_equivalence_key"],
                    "period": period,
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                    "horizon": spec.get("horizon", "annual_h1_100bp_year"),
                    "horizon_index": spec.get("horizon_index", "h1"),
                    "demand_conversion_case": case["demand_conversion_case"],
                    "baseline_run_id": baseline.manifest["run_id"],
                    "shock_run_id": shock.manifest["run_id"],
                    "tdc_change_baseline_bil": base_tdc,
                    "tdc_change_shock_bil": shock_tdc,
                    "delta_tdc_change_bil": delta_tdc,
                    "overlap_baseline_bil": base_overlap,
                    "overlap_shock_bil": shock_overlap,
                    "delta_overlap_bil": delta_overlap,
                    "tdc_change_ex_overlap_baseline_bil": base_ex,
                    "tdc_change_ex_overlap_shock_bil": shock_ex,
                    "delta_tdc_ex_overlap_bil": delta_ex,
                    "delta_tdc_ex_overlap_interest_driven_excluded_bil": interest_excluded,
                    "delta_tdc_ex_overlap_non_interest_admissible_bil": admissible,
                    "delta_tdc_ex_overlap_split_remainder_bil": remainder,
                    "delta_tdc_ex_overlap_reconciled_bil": reconciled,
                    "tdc_materialized_deposit_stock_admissible_bil": admissible_stock,
                    "tdc_materialized_deposit_stock_interest_excluded_bil": interest_excluded_stock,
                    "tdc_income_addendum_full_level_rate": TDC_INCOME_ADDENDUM_FULL_LEVEL_RATE,
                    "tdc_income_addendum_gross_interest_bil": gross_income,
                    "tdc_income_addendum_route_family": TDC_INCOME_ADDENDUM_ROUTE_FAMILY,
                    "tdc_income_addendum_admission_status": TDC_INCOME_ADDENDUM_ADMISSION_STATUS,
                    "tdc_income_addendum_collision_status": TDC_INCOME_ADDENDUM_COLLISION_STATUS,
                    "selected_support_formula": TDC_SELECTED_SUPPORT_FORMULA,
                    "beta_assumption_id": case["beta_assumption_id"],
                    "beta": beta,
                    "beta_source_status": case["beta_source_status"],
                    "chi_assumption_id": case["chi_assumption_id"],
                    "chi": chi,
                    "chi_source_status": case["chi_source_status"],
                    "beta_times_chi": beta * chi,
                    "tdc_amount_basis": TDC_AMOUNT_BASIS,
                    # The beta-chi construction is retired, not selected. It is exported
                    # only under the legacy_* name so an existing consumer can still find
                    # it deliberately; it is no longer published under a neutral headline
                    # field that reads as the selected object. The selected object is the
                    # admissible stock and its income addendum, per
                    # TDC_SELECTED_SUPPORT_FORMULA.
                    "legacy_support_formula": "delta_tdc_ex_overlap_bil * beta * chi",
                    "legacy_chi_support_diagnostic_bil": delta_ex * beta * chi,
                    "legacy_chi_support_eligible_for_main_ratio": False,
                    "chi_selected_status": TDC_CHI_SELECTED_STATUS,
                    "same_state_status": "pass",
                    "rate_shock_only_status": (
                        "not_applicable_fiscal_injection_no_rate_shock"
                        if spec["shock_path_id"] == FISCAL_INJECTION_SHOCK_PATH_ID
                        else "pass"
                    ),
                    "shock_path_validation_status": "pass",
                    "period_alignment_status": "pass",
                    "overlap_identity_status": "pass",
                    "component_identity_status": "pass",
                    "route_identity_status": "pass",
                    "support_identity_status": "pass",
                    "state_manifest_status": "pass",
                    "contract_ingest_status": "ready_for_ratewall_assumption_mode_ingest",
                    "failure_reason": "",
                    "assumption_mode": True,
                    "evidence_mode_enabled": False,
                    "raw_rate_shock_enabled": False,
                    "named_marginal_shock_path_enabled": True,
                    "tdcsim_channel_classifier_enabled": False,
                    "enters_main_ratio_candidate": True,
                    "canonical_ratio_entry": False,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _component_split_for_summary(
    components: pd.DataFrame,
    *,
    period: Any,
    horizon: Any,
    demand_conversion_case: Any,
    delta_tdc_ex_overlap_bil: float,
) -> dict[str, float]:
    mask = (
        (components["period"].astype(str) == str(period))
        & (components["horizon"].astype(str) == str(horizon))
        & (components["demand_conversion_case"].astype(str) == str(demand_conversion_case))
    )
    frame = components.loc[mask]
    if frame.empty:
        raise MarginalTdcPairError("split summary missing component rows")
    interest_excluded = float(_series(frame, "tdc_split_component_interest_driven_excluded_bil").sum())
    non_interest_admissible = float(_series(frame, "tdc_split_component_non_interest_admissible_bil").sum())
    reconciled = interest_excluded + non_interest_admissible
    remainder = delta_tdc_ex_overlap_bil - reconciled
    if abs(remainder) > 1e-7:
        raise MarginalTdcPairError("split buckets do not reconcile to delta_tdc_ex_overlap_bil")
    return {
        "interest_driven_excluded": interest_excluded,
        "non_interest_admissible": non_interest_admissible,
        "reconciled": reconciled,
        "remainder": 0.0,
    }


def _assemble_components(
    spec: Mapping[str, Any],
    baseline: _RunBundle,
    shock: _RunBundle,
    cases: list[dict[str, Any]],
) -> pd.DataFrame:
    base = _collapse_components_to_ratewall_horizon(
        _load_output_frame(baseline, "tdcsim_period_tdc_components"),
        spec,
    )
    shocked = _collapse_components_to_ratewall_horizon(
        _load_output_frame(shock, "tdcsim_period_tdc_components"),
        spec,
    )
    if base.empty and shocked.empty:
        raise MarginalTdcPairError("TDC component tables must not both be empty")
    keys = ["period_start", "period_end", "component_key"]
    for frame in (base, shocked):
        for column in keys + ["amount_bil"]:
            if column not in frame.columns:
                raise MarginalTdcPairError(f"TDC components missing required column: {column}")
    group_cols = keys + [
        column
        for column in (
            "component_family",
            "holder_sector",
            "holder_subsector",
            "instrument_type",
            "payment_type",
            "accounting_basis",
            "enters_direct_interest_support",
            "enters_tdc_deposit_support_default",
            "tdc_amount_basis",
            "overlap_policy",
        )
        if column in base.columns or column in shocked.columns
    ]
    left = _component_amounts(base, group_cols).rename(columns={"amount_bil": "amount_baseline_bil"})
    right = _component_amounts(shocked, group_cols).rename(columns={"amount_bil": "amount_shock_bil"})
    merged = left.merge(right, on=group_cols, how="outer").fillna({"amount_baseline_bil": 0.0, "amount_shock_bil": 0.0})
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        delta = float(row["amount_shock_bil"]) - float(row["amount_baseline_bil"])
        included_ex_overlap = _truthy(row.get("enters_tdc_deposit_support_default", True))
        support_delta = delta if included_ex_overlap else 0.0
        split = _classify_deposit_creation_split(row, delta, included_ex_overlap)
        for case in cases:
            beta = float(case["beta"])
            chi = float(case["chi"])
            admitted_amount = split["non_interest_admissible_bil"]
            interest_excluded_amount = split["interest_driven_excluded_bil"]
            rows.append(
                {
                    "schema_version": PAIR_SCHEMA_VERSION,
                    "tdc_deposit_creation_split_schema_version": TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION,
                    "pair_id": spec["pair_id"],
                    "scenario_state_set_id": spec["scenario_state_set_id"],
                    "object_id": spec["object_id"],
                    "state_id": spec["state_id"],
                    "state_fingerprint_sha256": spec["state_fingerprint_sha256"],
                    "shock_path_id": spec["shock_path_id"],
                    "period": spec.get("ratewall_period", row["period_end"]),
                    "period_start": row["period_start"],
                    "period_end": row["period_end"],
                    "horizon": spec.get("horizon", "annual_h1_100bp_year"),
                    "demand_conversion_case": case["demand_conversion_case"],
                    "baseline_run_id": baseline.manifest["run_id"],
                    "shock_run_id": shock.manifest["run_id"],
                    **{column: row.get(column, "") for column in group_cols},
                    "amount_baseline_bil": float(row["amount_baseline_bil"]),
                    "amount_shock_bil": float(row["amount_shock_bil"]),
                    "delta_amount_bil": delta,
                    "included_in_delta_tdc_change": True,
                    "included_in_delta_overlap": not included_ex_overlap,
                    "included_in_delta_tdc_ex_overlap": included_ex_overlap,
                    "deposit_creation_driver_bucket": split["deposit_creation_driver_bucket"],
                    "tdc_split_component_admission_status": split["admission_status"],
                    "tdc_split_component_collision_family": split["collision_family"],
                    "tdc_split_component_bucket_reason": split["bucket_reason"],
                    "tdc_split_component_admitted": split["admitted"],
                    "tdc_split_component_excluded": split["excluded"],
                    "tdc_split_component_non_interest_admissible_bil": admitted_amount,
                    "tdc_split_component_interest_driven_excluded_bil": interest_excluded_amount,
                    "tdc_split_component_out_of_scope_excluded_bil": split["out_of_scope_excluded_bil"],
                    "tdc_split_component_split_remainder_bil": 0.0,
                    "tdc_split_component_admissible_materialized_deposit_stock_bil": admitted_amount * beta,
                    "tdc_split_component_interest_excluded_materialized_deposit_stock_bil": interest_excluded_amount * beta,
                    "excluded_from_support_reason": (
                        "" if included_ex_overlap else "direct_interest_or_route_memo_overlap_removed"
                    ),
                    "deposit_pass_through_beta": beta,
                    "deposit_pass_through_basis": TDC_AMOUNT_BASIS,
                    "marginal_deposit_creation_bil": support_delta,
                    "beta": beta,
                    "chi": chi,
                    "marginal_component_support_bil": support_delta * beta * chi,
                    "legacy_chi_support_diagnostic_bil": support_delta * beta * chi,
                    "chi_selected_status": TDC_CHI_SELECTED_STATUS,
                    "tdcsim_direct_interest_overlap_delta_bil": 0.0,
                    "external_support_overlap_delta_bil": 0.0,
                    "overlap_scope": "tdcsim_and_external_support",
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def _component_amounts(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = frame.copy()
    work["amount_bil"] = pd.to_numeric(work["amount_bil"], errors="coerce").fillna(0.0)
    for column in group_cols:
        if column not in work.columns:
            work[column] = ""
    return work.groupby(group_cols, dropna=False, as_index=False)["amount_bil"].sum()


def _classify_deposit_creation_split(row: Mapping[str, Any], delta: float, included_ex_overlap: bool) -> dict[str, Any]:
    component_family = str(row.get("component_family") or "")
    component_key = str(row.get("component_key") or "")
    payment_type = str(row.get("payment_type") or "")
    if not included_ex_overlap:
        if component_family == "route_plumbing_memo" or "plumbing_memo" in component_key:
            bucket = "route_plumbing_memo_excluded"
            reason = "route_plumbing_memo_never_recycled_into_split"
            collision_family = "route_plumbing_memo"
        else:
            bucket = "not_in_delta_tdc_ex_overlap_excluded"
            reason = "component_not_in_delta_tdc_ex_overlap_never_recycled_into_split"
            collision_family = "not_in_delta_tdc_ex_overlap"
        return {
            "deposit_creation_driver_bucket": bucket,
            "admission_status": "excluded_not_in_admissible_split",
            "collision_family": collision_family,
            "bucket_reason": reason,
            "admitted": False,
            "excluded": True,
            "non_interest_admissible_bil": 0.0,
            "interest_driven_excluded_bil": 0.0,
            "out_of_scope_excluded_bil": delta,
        }
    if component_family == "debt_service_interest" or payment_type in INTEREST_PAYMENT_TYPES:
        return {
            "deposit_creation_driver_bucket": "interest_driven_direct_interest_collision_excluded",
            "admission_status": "excluded_interest_driven_direct_interest_collision",
            "collision_family": "debt_service_interest",
            "bucket_reason": "interest_type_payment_to_deposit_user_route_excluded",
            "admitted": False,
            "excluded": True,
            "non_interest_admissible_bil": 0.0,
            "interest_driven_excluded_bil": delta,
            "out_of_scope_excluded_bil": 0.0,
        }
    if component_family == "auction_absorption":
        bucket = "non_interest_auction_absorption_admissible"
        reason = "auction_absorption_non_interest_bucket_admitted"
    elif component_family == "debt_service_principal" or payment_type == "principal":
        bucket = "non_interest_principal_redemption_admissible"
        reason = "principal_redemption_non_interest_bucket_admitted_sign_preserved"
    elif component_family == "fiscal":
        bucket = "non_interest_fiscal_flow_admissible"
        reason = "fiscal_flow_non_interest_bucket_admitted"
    else:
        raise MarginalTdcPairError(f"unsupported included split component: {component_key}")
    return {
        "deposit_creation_driver_bucket": bucket,
        "admission_status": "admitted_split_non_interest_bucket",
        "collision_family": "none",
        "bucket_reason": reason,
        "admitted": True,
        "excluded": False,
        "non_interest_admissible_bil": delta,
        "interest_driven_excluded_bil": 0.0,
        "out_of_scope_excluded_bil": 0.0,
    }


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _collapse_summary_to_ratewall_horizon(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    if not spec.get("ratewall_period") or len(frame) <= 1:
        return frame
    collapsed = _first_row_with_horizon_dates(frame, spec)
    for column in frame.columns:
        if column in {"period_start", "period_end"}:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            collapsed[column] = values.fillna(0.0).sum()
    return pd.DataFrame([collapsed])


def _collapse_components_to_ratewall_horizon(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    if not spec.get("ratewall_period") or len(frame) <= 1:
        return frame
    out = frame.copy()
    out["period_start"] = str(spec["horizon_start_date"])
    out["period_end"] = str(spec["horizon_end_date"])
    return out


def _first_row_with_horizon_dates(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(frame.iloc[0])
    row["period_start"] = str(spec["horizon_start_date"])
    row["period_end"] = str(spec["horizon_end_date"])
    return row


def _assemble_route_metadata(spec: Mapping[str, Any], baseline: _RunBundle, shock: _RunBundle) -> pd.DataFrame:
    frames = []
    for role, run in (("baseline", baseline), ("shock", shock)):
        frame = _load_output_frame(run, "tdcsim_tdc_principal_route_stock_closure")
        if frame.empty:
            raise MarginalTdcPairError(f"{role} route metadata is empty")
        frame = frame.copy()
        _set_front_column(frame, "schema_version", PAIR_SCHEMA_VERSION, 0)
        _set_front_column(frame, "pair_id", spec["pair_id"], 1)
        _set_front_column(frame, "scenario_state_set_id", spec["scenario_state_set_id"], 2)
        _set_front_column(frame, "state_id", spec["state_id"], 3)
        _set_front_column(
            frame,
            "state_fingerprint_sha256",
            spec["state_fingerprint_sha256"],
            4,
        )
        _set_front_column(frame, "object_id", spec["object_id"], 5)
        _set_front_column(frame, "shock_path_id", spec["shock_path_id"], 6)
        _set_front_column(frame, "run_role", role, 7)
        _set_front_column(frame, "run_id", run.manifest["run_id"], 8)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _set_front_column(
    frame: pd.DataFrame,
    column: str,
    value: Any,
    position: int,
) -> None:
    if column in frame.columns:
        frame[column] = value
        moved = frame.pop(column)
        frame.insert(min(position, len(frame.columns)), column, moved)
    else:
        frame.insert(min(position, len(frame.columns)), column, value)


def _assemble_state_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_manifest_schema_version": "tdcsim_ratewall_scenario_state_manifest_v1",
        "scenario_state_set_id": spec["scenario_state_set_id"],
        "states": [
            {
                "state_id": spec["state_id"],
                "state_kind": spec["state_kind"],
                "state_period": spec["state_period"],
                "scenario_id": spec.get("scenario_id", spec["baseline_scenario_id"]),
                "opening_state_date": spec["opening_state_date"],
                "actuals_available_as_of": spec["actuals_available_as_of"],
                "source_vintage": spec["source_vintage"],
                "horizon_start_date": spec["horizon_start_date"],
                "horizon_end_date": spec["horizon_end_date"],
                "state_fingerprint_sha256": spec["state_fingerprint_sha256"],
                "state_component_inventory_sha256": spec["state_component_inventory_sha256"],
                "compiled_non_rate_inputs_digest": spec.get("compiled_non_rate_inputs_digest", ""),
                "source_grade_status": spec.get("source_grade_status", ""),
                "state_construction_method": spec.get("state_construction_method", ""),
                "forecast_state_export_manifest_sha256": spec.get("forecast_state_export_manifest_sha256", ""),
                "derived_state_package_sha256": spec.get("derived_state_package_sha256", ""),
                "parent_baseline_package_sha256": spec.get("parent_baseline_package_sha256", ""),
                "rollforward_run_manifest_sha256": spec.get("rollforward_run_manifest_sha256", ""),
                "opening_portfolio_sha256": spec.get("opening_portfolio_sha256", ""),
                "debt_stock_path_sha256": spec.get("debt_stock_path_sha256", ""),
                "primary_deficit_path_sha256": spec.get("primary_deficit_path_sha256", ""),
                "operating_cash_path_sha256": spec.get("operating_cash_path_sha256", ""),
                "fed_holdings_path_sha256": spec.get("fed_holdings_path_sha256", ""),
                "holder_route_assumptions_sha256": spec.get("holder_route_assumptions_sha256", ""),
                "issuance_mix_sha256": spec.get("issuance_mix_sha256", ""),
                "mmf_split_assumptions_sha256": spec.get("mmf_split_assumptions_sha256", ""),
                "nominal_gdp_bil": spec.get("nominal_gdp_bil", ""),
                "opening_tdc_stock_bil": spec.get("opening_tdc_stock_bil", ""),
                "opening_deposit_liquidity_stock_bil": spec.get("opening_deposit_liquidity_stock_bil", ""),
                "opening_route_stock_total_bil": spec.get("opening_route_stock_total_bil", ""),
                "opening_route_stock_domestic_nonbank_bil": spec.get("opening_route_stock_domestic_nonbank_bil", ""),
                "opening_route_stock_bank_bil": spec.get("opening_route_stock_bank_bil", ""),
                "opening_route_stock_mmf_bil": spec.get("opening_route_stock_mmf_bil", ""),
                "opening_route_stock_foreign_bil": spec.get("opening_route_stock_foreign_bil", ""),
                "opening_route_stock_fed_bil": spec.get("opening_route_stock_fed_bil", ""),
                "baseline_pair_id": spec["baseline_scenario_id"],
                "shock_pair_id": spec["shock_scenario_id"],
                "tdcsim_pair_id": spec["pair_id"],
                "denominator_equivalence_key": spec["denominator_equivalence_key"],
                "shock_path_id": spec["shock_path_id"],
                "claim_boundary": CLAIM_BOUNDARY,
            }
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_pair_manifest(
    spec: Mapping[str, Any],
    baseline: _RunBundle,
    shock: _RunBundle,
    checks: Mapping[str, str],
    out: Path,
) -> dict[str, Any]:
    files_block = {
        name: _artifact(out / name)
        for name in (SUMMARY_FILE, COMPONENTS_FILE, ROUTE_METADATA_FILE, STATE_MANIFEST_FILE)
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pair_id": spec["pair_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_spec": dict(spec),
        "baseline_run": _source_run_block(baseline),
        "shock_run": _source_run_block(shock),
        "validation": {
            "status": "pass",
            **dict(checks),
        },
        "files": files_block,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest["pair_manifest_config_sha256"] = canonical_json_sha256({k: v for k, v in manifest.items() if k != "pair_manifest_config_sha256"})
    with files("tdcsim_cbo").joinpath("schemas/cbo-marginal-tdc-manifest-v1.schema.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        validate_schema(manifest, json.load(handle), label="marginal_pair_manifest")
    return manifest


def _read_pair_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MarginalTdcPairError("marginal TDC pair manifest is missing")
    manifest = read_json(path)
    if not isinstance(manifest, dict):
        raise MarginalTdcPairError("marginal TDC pair manifest must be an object")
    with files("tdcsim_cbo").joinpath("schemas/cbo-marginal-tdc-manifest-v1.schema.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        validate_schema(manifest, json.load(handle), label="marginal_pair_manifest")
    return manifest


def _verify_pair_files(root: Path, manifest: Mapping[str, Any]) -> None:
    files_block = manifest.get("files")
    if not isinstance(files_block, Mapping):
        raise MarginalTdcPairError("pair manifest files must be an object")
    for filename in (SUMMARY_FILE, COMPONENTS_FILE, ROUTE_METADATA_FILE, STATE_MANIFEST_FILE):
        artifact = files_block.get(filename)
        if not isinstance(artifact, Mapping):
            raise MarginalTdcPairError(f"pair manifest missing file artifact: {filename}")
        path = root / filename
        if not path.exists():
            raise MarginalTdcPairError(f"pair artifact is missing: {filename}")
        if sha256_file(path) != artifact.get("sha256"):
            raise MarginalTdcPairError(f"pair artifact SHA mismatch: {filename}")
        if path.stat().st_size != int(artifact.get("bytes", -1)):
            raise MarginalTdcPairError(f"pair artifact byte count mismatch: {filename}")


def _verify_manifest_checks(manifest: Mapping[str, Any], checks: Mapping[str, str]) -> None:
    validation = manifest.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "pass":
        raise MarginalTdcPairError("pair manifest validation.status must be pass")
    for key, value in checks.items():
        if validation.get(key) != value:
            raise MarginalTdcPairError(f"pair manifest validation check mismatch: {key}")


def _verify_summary(summary: pd.DataFrame) -> None:
    required = {
        "schema_version",
        "tdc_deposit_creation_split_schema_version",
        "contract_version",
        "scenario_state_set_id",
        "state_id",
        "state_kind",
        "state_period",
        "scenario_id",
        "state_fingerprint_sha256",
        "state_component_inventory_sha256",
        "shock_path_id",
        "shock_bps_year",
        "horizon",
        "demand_conversion_case",
        "delta_tdc_change_bil",
        "delta_overlap_bil",
        "delta_tdc_ex_overlap_bil",
        "delta_tdc_ex_overlap_interest_driven_excluded_bil",
        "delta_tdc_ex_overlap_non_interest_admissible_bil",
        "delta_tdc_ex_overlap_split_remainder_bil",
        "delta_tdc_ex_overlap_reconciled_bil",
        "tdc_materialized_deposit_stock_admissible_bil",
        "tdc_materialized_deposit_stock_interest_excluded_bil",
        "tdc_income_addendum_full_level_rate",
        "tdc_income_addendum_gross_interest_bil",
        "tdc_income_addendum_route_family",
        "tdc_income_addendum_admission_status",
        "tdc_income_addendum_collision_status",
        "selected_support_formula",
        "beta_assumption_id",
        "beta",
        "chi_assumption_id",
        "chi",
        "beta_times_chi",
        "tdc_amount_basis",
        "legacy_support_formula",
        "legacy_chi_support_diagnostic_bil",
        "legacy_chi_support_eligible_for_main_ratio",
        "chi_selected_status",
        "state_manifest_status",
        "route_identity_status",
        "assumption_mode",
        "evidence_mode_enabled",
        "raw_rate_shock_enabled",
        "named_marginal_shock_path_enabled",
        "tdcsim_channel_classifier_enabled",
        "enters_main_ratio_candidate",
        "canonical_ratio_entry",
        "claim_boundary",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise MarginalTdcPairError(f"marginal summary missing required columns: {missing}")
    if set(summary["schema_version"].astype(str)) != {PAIR_SCHEMA_VERSION}:
        raise MarginalTdcPairError("marginal summary schema_version is invalid")
    if set(summary["tdc_deposit_creation_split_schema_version"].astype(str)) != {TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION}:
        raise MarginalTdcPairError("marginal summary split schema_version is invalid")
    if set(summary["contract_version"].astype(str)) != {CONTRACT_VERSION}:
        raise MarginalTdcPairError("marginal summary contract_version is invalid")
    shock_path_ids = set(summary["shock_path_id"].astype(str))
    shock_bps_values = set(str(int(float(value))) if float(value).is_integer() else str(float(value)) for value in summary["shock_bps_year"])
    if shock_path_ids == {SHOCK_PATH_ID}:
        if shock_bps_values != {"100"}:
            raise MarginalTdcPairError("marginal summary shock_bps_year is invalid")
    elif shock_path_ids == {FISCAL_INJECTION_SHOCK_PATH_ID}:
        if shock_bps_values != {"0"}:
            raise MarginalTdcPairError("fiscal injection summary shock_bps_year is invalid")
    else:
        raise MarginalTdcPairError("marginal summary shock_path_id is invalid")
    if set(summary["tdc_amount_basis"].astype(str)) != {TDC_AMOUNT_BASIS}:
        raise MarginalTdcPairError("marginal summary TDC amount basis is invalid")
    if set(summary["legacy_support_formula"].astype(str)) != {"delta_tdc_ex_overlap_bil * beta * chi"}:
        raise MarginalTdcPairError("marginal summary legacy support formula is invalid")
    identity = _series(summary, "delta_tdc_change_bil") - _series(summary, "delta_overlap_bil")
    if (identity - _series(summary, "delta_tdc_ex_overlap_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary ex-overlap identity failed")
    if (_series(summary, "beta") * _series(summary, "chi") - _series(summary, "beta_times_chi")).abs().max() > 1e-12:
        raise MarginalTdcPairError("marginal summary beta_times_chi identity failed")
    support = _series(summary, "delta_tdc_ex_overlap_bil") * _series(summary, "beta") * _series(summary, "chi")
    if (support - _series(summary, "legacy_chi_support_diagnostic_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary legacy support identity failed")
    split_reconciled = (
        _series(summary, "delta_tdc_ex_overlap_interest_driven_excluded_bil")
        + _series(summary, "delta_tdc_ex_overlap_non_interest_admissible_bil")
    )
    if (split_reconciled - _series(summary, "delta_tdc_ex_overlap_reconciled_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary split reconciled identity failed")
    if (split_reconciled - _series(summary, "delta_tdc_ex_overlap_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary split ex-overlap identity failed")
    if _series(summary, "delta_tdc_ex_overlap_split_remainder_bil").abs().max() > 1e-12:
        raise MarginalTdcPairError("marginal summary split remainder must be zero")
    if (
        _series(summary, "delta_tdc_ex_overlap_non_interest_admissible_bil") * _series(summary, "beta")
        - _series(summary, "tdc_materialized_deposit_stock_admissible_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary admissible stock identity failed")
    if (
        _series(summary, "delta_tdc_ex_overlap_interest_driven_excluded_bil") * _series(summary, "beta")
        - _series(summary, "tdc_materialized_deposit_stock_interest_excluded_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary interest excluded stock identity failed")
    if (
        _series(summary, "tdc_materialized_deposit_stock_admissible_bil")
        * _series(summary, "tdc_income_addendum_full_level_rate")
        - _series(summary, "tdc_income_addendum_gross_interest_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary income addendum identity failed")
    if set(summary["tdc_income_addendum_route_family"].astype(str)) != {TDC_INCOME_ADDENDUM_ROUTE_FAMILY}:
        raise MarginalTdcPairError("marginal summary income addendum route family failed")
    if set(summary["tdc_income_addendum_admission_status"].astype(str)) != {TDC_INCOME_ADDENDUM_ADMISSION_STATUS}:
        raise MarginalTdcPairError("marginal summary income addendum admission status failed")
    if set(summary["tdc_income_addendum_collision_status"].astype(str)) != {TDC_INCOME_ADDENDUM_COLLISION_STATUS}:
        raise MarginalTdcPairError("marginal summary income addendum collision status failed")
    if set(summary["selected_support_formula"].astype(str)) != {TDC_SELECTED_SUPPORT_FORMULA}:
        raise MarginalTdcPairError("marginal summary selected support formula failed")
    if (support - _series(summary, "legacy_chi_support_diagnostic_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal summary legacy chi diagnostic identity failed")
    if set(summary["chi_selected_status"].astype(str)) != {TDC_CHI_SELECTED_STATUS}:
        raise MarginalTdcPairError("marginal summary chi selected status failed")
    _require_bool(summary, "assumption_mode", True)
    _require_bool(summary, "evidence_mode_enabled", False)
    _require_bool(summary, "raw_rate_shock_enabled", False)
    _require_bool(summary, "named_marginal_shock_path_enabled", True)
    _require_bool(summary, "tdcsim_channel_classifier_enabled", False)
    _require_bool(summary, "enters_main_ratio_candidate", True)
    _require_bool(summary, "canonical_ratio_entry", False)
    for status_field in (
        "same_state_status",
        "rate_shock_only_status",
        "shock_path_validation_status",
        "period_alignment_status",
        "overlap_identity_status",
        "component_identity_status",
        "route_identity_status",
        "support_identity_status",
        "state_manifest_status",
    ):
        allowed_statuses = {"pass"}
        if status_field == "rate_shock_only_status" and shock_path_ids == {FISCAL_INJECTION_SHOCK_PATH_ID}:
            allowed_statuses.add("not_applicable_fiscal_injection_no_rate_shock")
        if not set(summary[status_field].astype(str)) <= allowed_statuses:
            raise MarginalTdcPairError(f"marginal summary status failed: {status_field}")
    if set(summary["claim_boundary"].astype(str)) != {CLAIM_BOUNDARY}:
        raise MarginalTdcPairError("marginal summary claim_boundary is invalid")


def _verify_components(components: pd.DataFrame, summary: pd.DataFrame) -> None:
    if components.empty:
        raise MarginalTdcPairError("marginal components must not be empty")
    required = {
        "tdc_deposit_creation_split_schema_version",
        "pair_id",
        "scenario_state_set_id",
        "state_id",
        "state_fingerprint_sha256",
        "shock_path_id",
        "period",
        "horizon",
        "demand_conversion_case",
        "delta_amount_bil",
        "included_in_delta_tdc_change",
        "included_in_delta_overlap",
        "included_in_delta_tdc_ex_overlap",
        "component_family",
        "payment_type",
        "deposit_creation_driver_bucket",
        "tdc_split_component_admission_status",
        "tdc_split_component_collision_family",
        "tdc_split_component_bucket_reason",
        "tdc_split_component_admitted",
        "tdc_split_component_excluded",
        "tdc_split_component_non_interest_admissible_bil",
        "tdc_split_component_interest_driven_excluded_bil",
        "tdc_split_component_out_of_scope_excluded_bil",
        "tdc_split_component_split_remainder_bil",
        "tdc_split_component_admissible_materialized_deposit_stock_bil",
        "tdc_split_component_interest_excluded_materialized_deposit_stock_bil",
        "beta",
        "chi",
        "marginal_component_support_bil",
        "legacy_chi_support_diagnostic_bil",
        "chi_selected_status",
        "claim_boundary",
    }
    missing = sorted(required - set(components.columns))
    if missing:
        raise MarginalTdcPairError(f"marginal components missing required columns: {missing}")
    if set(components["tdc_deposit_creation_split_schema_version"].astype(str)) != {TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION}:
        raise MarginalTdcPairError("marginal components split schema_version is invalid")
    if not set(components["deposit_creation_driver_bucket"].astype(str)) <= SPLIT_DRIVER_BUCKETS:
        raise MarginalTdcPairError("marginal components split driver bucket is invalid")
    included = components["included_in_delta_tdc_ex_overlap"].map(_truthy).astype(float)
    expected = (
        _series(components, "delta_amount_bil")
        * _series(components, "beta")
        * _series(components, "chi")
        * included
    )
    if (expected - _series(components, "marginal_component_support_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal component support identity failed")
    admitted = components["tdc_split_component_admitted"].map(_truthy)
    excluded = components["tdc_split_component_excluded"].map(_truthy)
    if (admitted & excluded).any() or (~admitted & ~excluded).any():
        raise MarginalTdcPairError("marginal component split admission boolean identity failed")
    admissible = _series(components, "tdc_split_component_non_interest_admissible_bil")
    interest_excluded = _series(components, "tdc_split_component_interest_driven_excluded_bil")
    out_of_scope = _series(components, "tdc_split_component_out_of_scope_excluded_bil")
    if _series(components, "tdc_split_component_split_remainder_bil").abs().max() > 1e-12:
        raise MarginalTdcPairError("marginal component split remainder must be zero")
    if (
        admissible * _series(components, "beta")
        - _series(components, "tdc_split_component_admissible_materialized_deposit_stock_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal component admissible stock identity failed")
    if (
        interest_excluded * _series(components, "beta")
        - _series(components, "tdc_split_component_interest_excluded_materialized_deposit_stock_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal component interest excluded stock identity failed")
    admitted_interest = admitted & (
        components["component_family"].astype(str).eq("debt_service_interest")
        | components["payment_type"].astype(str).isin(INTEREST_PAYMENT_TYPES)
    )
    if admitted_interest.any():
        raise MarginalTdcPairError("marginal component split admitted interest row failed")
    admitted_principal = admitted & (
        components["component_family"].astype(str).eq("debt_service_principal")
        | components["payment_type"].astype(str).eq("principal")
    )
    if (
        _series(components.loc[admitted_principal], "delta_amount_bil")
        - _series(components.loc[admitted_principal], "tdc_split_component_non_interest_admissible_bil")
    ).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal component principal sign preservation failed")
    active_split = admissible + interest_excluded
    if (active_split[included.astype(bool)] - _series(components.loc[included.astype(bool)], "delta_amount_bil")).abs().max() > 1e-7:
        raise MarginalTdcPairError("marginal component included split amount identity failed")
    if (active_split[~included.astype(bool)]).abs().max() > 1e-12:
        raise MarginalTdcPairError("marginal component out-of-overlap split amount failed")
    if (out_of_scope[included.astype(bool)]).abs().max() > 1e-12:
        raise MarginalTdcPairError("marginal component included out-of-scope amount failed")
    if set(components["chi_selected_status"].astype(str)) != {TDC_CHI_SELECTED_STATUS}:
        raise MarginalTdcPairError("marginal component chi selected status failed")
    grouped = components.groupby(["period", "horizon", "demand_conversion_case"], dropna=False)
    summary_by_key = {
        (str(row["period"]), str(row["horizon"]), str(row["demand_conversion_case"])): row
        for _, row in summary.iterrows()
    }
    for key, frame in grouped:
        normalized_key = tuple(str(value) for value in key)
        if normalized_key not in summary_by_key:
            raise MarginalTdcPairError("component key missing from summary")
        summary_row = summary_by_key[normalized_key]
        delta_ex = _flagged_sum(frame, "delta_amount_bil", "included_in_delta_tdc_ex_overlap")
        if abs(delta_ex - float(summary_row["delta_tdc_ex_overlap_bil"])) > 1e-7:
            raise MarginalTdcPairError("marginal component ex-overlap sum failed")
        support = float(pd.to_numeric(frame["marginal_component_support_bil"], errors="coerce").fillna(0.0).sum())
        if abs(support - float(summary_row["legacy_chi_support_diagnostic_bil"])) > 1e-7:
            raise MarginalTdcPairError("marginal component support sum failed")
        interest_excluded_sum = float(
            pd.to_numeric(frame["tdc_split_component_interest_driven_excluded_bil"], errors="coerce").fillna(0.0).sum()
        )
        admissible_sum = float(
            pd.to_numeric(frame["tdc_split_component_non_interest_admissible_bil"], errors="coerce").fillna(0.0).sum()
        )
        if abs(interest_excluded_sum - float(summary_row["delta_tdc_ex_overlap_interest_driven_excluded_bil"])) > 1e-7:
            raise MarginalTdcPairError("marginal component interest excluded split sum failed")
        if abs(admissible_sum - float(summary_row["delta_tdc_ex_overlap_non_interest_admissible_bil"])) > 1e-7:
            raise MarginalTdcPairError("marginal component admissible split sum failed")
        if abs(interest_excluded_sum + admissible_sum - float(summary_row["delta_tdc_ex_overlap_bil"])) > 1e-7:
            raise MarginalTdcPairError("marginal component split ex-overlap sum failed")
    if set(components["claim_boundary"].astype(str)) != {CLAIM_BOUNDARY}:
        raise MarginalTdcPairError("marginal components claim_boundary is invalid")


def _verify_route_metadata(route_metadata: pd.DataFrame, baseline_run_id: str, shock_run_id: str) -> None:
    if route_metadata.empty:
        raise MarginalTdcPairError("marginal route metadata must not be empty")
    if set(route_metadata["run_role"].astype(str)) != {"baseline", "shock"}:
        raise MarginalTdcPairError("route metadata must include baseline and shock rows")
    if set(route_metadata["run_id"].astype(str)) != {baseline_run_id, shock_run_id}:
        raise MarginalTdcPairError("route metadata run IDs do not match pair source runs")
    if "opening_route_stock_bil" in route_metadata.columns:
        stock_frame = route_metadata
        if set(route_metadata["shock_path_id"].astype(str)) == {FISCAL_INJECTION_SHOCK_PATH_ID} and "period_start" in route_metadata.columns:
            starts = pd.to_datetime(route_metadata["period_start"], errors="coerce")
            if starts.notna().any():
                stock_frame = route_metadata.loc[starts == starts.min()].copy()
        group_cols = [
            column
            for column in (
                "pair_id",
                "state_id",
                "shock_path_id",
                "period_start",
                "period_end",
                "route_holder_sector",
                "route_holder_subsector",
                "instrument_type",
                "maturity_bucket",
            )
            if column in stock_frame.columns
        ]
        stocks = stock_frame.groupby(group_cols, dropna=False)["opening_route_stock_bil"].nunique()
        if (stocks > 1).any():
            raise MarginalTdcPairError("route opening stocks differ within pair")


def _read_state_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise MarginalTdcPairError("scenario-state manifest is missing")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise MarginalTdcPairError("scenario-state manifest must be a JSON object")
    return payload


def _verify_state_manifest(manifest: Mapping[str, Any], spec: Mapping[str, Any]) -> None:
    if manifest.get("state_manifest_schema_version") != "tdcsim_ratewall_scenario_state_manifest_v1":
        raise MarginalTdcPairError("scenario-state manifest schema version failed")
    if manifest.get("scenario_state_set_id") != spec["scenario_state_set_id"]:
        raise MarginalTdcPairError("scenario-state manifest set id failed")
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        raise MarginalTdcPairError("scenario-state manifest claim boundary failed")
    states = manifest.get("states")
    if not isinstance(states, list) or len(states) != 1:
        raise MarginalTdcPairError("scenario-state manifest must contain one state for pair")
    state = states[0]
    if not isinstance(state, Mapping):
        raise MarginalTdcPairError("scenario-state manifest state must be an object")
    for key in (
        "state_id",
        "state_kind",
        "state_period",
        "state_fingerprint_sha256",
        "state_component_inventory_sha256",
        "shock_path_id",
        "denominator_equivalence_key",
    ):
        if state.get(key) != spec.get(key):
            raise MarginalTdcPairError(f"scenario-state manifest mismatch: {key}")
    if state.get("claim_boundary") != CLAIM_BOUNDARY:
        raise MarginalTdcPairError("scenario-state manifest state boundary failed")


def _verify_tdc_summary_identities(label: str, frame: pd.DataFrame) -> None:
    overlap_identity = _series(frame, "tdc_change_bil") - _series(frame, "overlap_cashflow_bil") - _series(
        frame,
        "tdc_change_ex_overlap_bil",
    )
    if overlap_identity.abs().max() > 1e-7:
        raise MarginalTdcPairError(f"{label} overlap identity failed")
    if "component_sum_error_bil" in frame.columns and _series(frame, "component_sum_error_bil").abs().max() > 1e-7:
        raise MarginalTdcPairError(f"{label} component identity failed")


def _verify_run_artifacts(role: str, root: Path, manifest: Mapping[str, Any]) -> None:
    for item in manifest.get("outputs", []):
        if not isinstance(item, Mapping):
            continue
        rel = Path(str(item.get("relative_path") or ""))
        path = root / rel
        if not path.exists():
            raise MarginalTdcPairError(f"{role} output artifact is missing: {rel}")
        if item.get("sha256") and sha256_file(path) != item.get("sha256"):
            raise MarginalTdcPairError(f"{role} output artifact SHA mismatch: {rel}")
    for item in manifest.get("compiled_inputs", []):
        if not isinstance(item, Mapping):
            continue
        rel = Path(str(item.get("relative_path") or ""))
        path = root / rel
        if not path.exists():
            raise MarginalTdcPairError(f"{role} compiled input artifact is missing: {rel}")
        if item.get("sha256") and sha256_file(path) != item.get("sha256"):
            raise MarginalTdcPairError(f"{role} compiled input artifact SHA mismatch: {rel}")


def _load_tdc_summary(run: _RunBundle) -> pd.DataFrame:
    return _load_output_frame(run, "tdcsim_period_tdc_summary")


def _load_output_frame(run: _RunBundle, logical_name_stem: str) -> pd.DataFrame:
    path = _output_path(run, logical_name_stem)
    return _read_csv(path)


def _output_path(run: _RunBundle, logical_name_stem: str) -> Path:
    for item in run.manifest.get("outputs", []):
        if isinstance(item, Mapping) and str(item.get("logical_name", "")).startswith(f"{logical_name_stem}.csv"):
            return run.root / str(item["relative_path"])
    fallback = run.root / "outputs" / f"{logical_name_stem}.csv"
    if fallback.exists():
        return fallback
    gz = run.root / "outputs" / f"{logical_name_stem}.csv.gz"
    if gz.exists():
        return gz
    raise MarginalTdcPairError(f"{run.role} run is missing required output: {logical_name_stem}")


def _read_compiled_csv(run: _RunBundle, filename: str) -> pd.DataFrame:
    for item in run.manifest.get("compiled_inputs", []):
        if isinstance(item, Mapping) and Path(str(item.get("logical_name") or "")).name == filename:
            return _read_csv(run.root / str(item["relative_path"]))
    fallback = run.root / "compile" / "compiled" / "forecast_inputs" / filename
    if fallback.exists():
        return _read_csv(fallback)
    raise MarginalTdcPairError(f"{run.role} run is missing compiled input: {filename}")


def _read_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def _compiled_input_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for item in manifest.get("compiled_inputs", []):
        if isinstance(item, Mapping):
            out[str(item.get("logical_name") or item.get("relative_path") or "")] = item
    return out


def _row_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    output_manifest = manifest.get("output_manifest", {})
    if isinstance(output_manifest, Mapping):
        row_metadata = output_manifest.get("row_metadata", {})
        if isinstance(row_metadata, Mapping):
            return row_metadata
    return {}


def _demand_conversion_cases(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = spec.get("demand_conversion_cases")
    if not isinstance(raw, list) or not raw:
        raise MarginalTdcPairError("marginal pair spec requires demand_conversion_cases")
    cases: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise MarginalTdcPairError("demand_conversion_cases entries must be objects")
        beta = float(item["beta"])
        chi = float(item["chi"])
        if not pd.notna(beta) or not pd.notna(chi):
            raise MarginalTdcPairError("beta/chi must be finite")
        cases.append(
            {
                "demand_conversion_case": str(item["demand_conversion_case"]),
                "beta": beta,
                "chi": chi,
                "beta_assumption_id": str(item.get("beta_assumption_id", "beta_ratewall_default")),
                "beta_source_status": str(item.get("beta_source_status", "assumption_mode")),
                "chi_assumption_id": str(item.get("chi_assumption_id", "chi_ratewall_default")),
                "chi_source_status": str(item.get("chi_source_status", "assumption_mode")),
            }
        )
    return cases


def _source_run_block(run: _RunBundle) -> dict[str, Any]:
    manifest_path = run.root / "tdcsim_cbo_run_manifest.json"
    return {
        "run_id": run.manifest["run_id"],
        "run_dir": str(run.root),
        "manifest_sha256": sha256_file(manifest_path),
        "compiled_inputs_digest": run.manifest.get("compiled_inputs_digest", ""),
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "logical_name": path.name,
        "relative_path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": "text/csv" if path.suffix == ".csv" else "application/json",
    }


def _num(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return 0.0 if pd.isna(value) else float(value)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise MarginalTdcPairError(f"required numeric column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _require_bool(frame: pd.DataFrame, column: str, expected: bool) -> None:
    values = frame[column]
    if values.dtype == bool:
        observed = set(bool(value) for value in values)
    else:
        observed = set(values.astype(str).str.strip().str.lower().map({"true": True, "false": False}))
    if observed != {expected}:
        raise MarginalTdcPairError(f"marginal summary boolean flag failed: {column}")


def _flagged_sum(frame: pd.DataFrame, amount_column: str, flag_column: str) -> float:
    flags = frame[flag_column]
    if flags.dtype == bool:
        mask = flags
    else:
        mask = flags.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return float(pd.to_numeric(frame.loc[mask, amount_column], errors="coerce").fillna(0.0).sum())


__all__ = [
    "CLAIM_BOUNDARY",
    "DENOMINATOR_EQUIVALENCE_KEY",
    "MANIFEST_FILE",
    "MANIFEST_SCHEMA_VERSION",
    "OBJECT_ID",
    "PAIR_SCHEMA_VERSION",
    "SHOCK_PATH_ID",
    "STATE_MANIFEST_FILE",
    "MarginalPairResult",
    "MarginalTdcPairError",
    "assemble_marginal_tdc_pair",
    "verify_marginal_tdc_pair",
]
