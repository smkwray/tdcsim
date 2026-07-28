"""Deterministic CBO scenario compiler.

The compiler materializes a verified immutable baseline into a work directory,
copies its forecast inputs byte-for-byte, applies a sparse scenario overlay, and
writes a manifest with canonical input digests. It does not invoke the simulator.
"""

from __future__ import annotations

import csv
import math
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tdc_shared import MMF_DEPOSIT_PASS_THROUGH_DEFAULT

from ._json import canonical_json_sha256, read_json, sha256_file, write_json
from .baseline import CboBaselinePackage
from .contract import CboScenarioSpec
from .transforms.fiscal import (
    apply_cash_residual_override,
    apply_debt_target_override,
    apply_fiscal_incidence_override,
    apply_operating_cash_override,
    apply_primary_deficit_override,
)
from .transforms.portfolio import (
    apply_fed_holdings_override,
    compile_holder_preference_events,
    compile_issuance_mix_override,
    validate_holder_preferences,
)
from .transforms.rates import (
    apply_cpi_override,
    apply_frn_override,
    apply_nominal_yield_curve_override,
    apply_tips_real_yield_override,
)


FORECAST_INPUTS = "forecast_inputs"
COMPILED_MANIFEST = "tdcsim_cbo_compiled_manifest.json"
ISSUANCE_MIX_FILE = "tdcsim_issuance_mix_assumptions.json"
HOLDER_PREFERENCE_EVENTS_FILE = "tdcsim_holder_preference_events.json"
RUNTIME_ASSUMPTIONS_FILE = "tdcsim_runtime_assumptions.json"
OPENING_RUNTIME_STATE_FILE = "tdcsim_opening_runtime_state.json"
_SOURCE_BASELINE_OPENING_DEFAULTS = {
    "reserves": 3000.0,
    "tdc_level": 0.0,
}

_DEFAULT_ISSUANCE_SECURITY_SHARES = {
    "bills": 0.225,
    "notes": 0.495,
    "bonds": 0.18,
    "tips": 0.06,
    "frn": 0.04,
}
_DEFAULT_ISSUANCE_MATURITY_DISTRIBUTIONS = {
    "bills": [{"maturity_years": 0.5, "share": 1.0}],
    "notes": [{"maturity_years": 5.0, "share": 1.0}],
    "bonds": [{"maturity_years": 20.0, "share": 1.0}],
    "tips": [{"maturity_years": 10.0, "share": 1.0}],
    "frn": [{"maturity_years": 2.0, "share": 1.0}],
}

INPUT_FILES = {
    "nominal_yield_curve": "tdcsim_yield_curve_surface.csv",
    "frn_benchmark": "tdcsim_frn_rate_path.csv",
    "inflation_cpi": "tdcsim_tips_cpi_path.csv",
    "tips_real_yield": "tdcsim_tips_real_yield_path.csv",
    "operating_cash": "tdcsim_operating_cash_path.csv",
    "cash_reconciliation": "tdcsim_cash_reconciliation_residual.csv",
    "primary_deficit": "tdcsim_primary_deficit_path.csv",
    "debt_target": "tdcsim_debt_stock_path.csv",
    "fed_holdings": "tdcsim_fed_holdings_path.csv",
    "fiscal_incidence": "tdcsim_fiscal_incidence_policy.csv",
}


class CompilerError(ValueError):
    """Raised when a CBO scenario cannot be compiled safely."""


@dataclass(frozen=True)
class CboCompiledScenario:
    """A compiled scenario input package."""

    work_dir: Path
    baseline_dir: Path
    compiled_dir: Path
    forecast_inputs_dir: Path
    manifest_path: Path
    baseline_forecast_inputs_digest: str
    compiled_inputs_digest: str
    scenario_sha256: str
    changed_inputs: tuple[str, ...]
    manifest: Mapping[str, Any]


class CboScenarioCompiler:
    """Compile sparse CBO scenario overlays into TDCSIM forecast inputs."""

    def compile(
        self,
        baseline: CboBaselinePackage,
        spec: CboScenarioSpec,
        work_dir: str | Path,
    ) -> CboCompiledScenario:
        if not baseline.is_zip:
            raise CompilerError("CBO scenario compiler requires a release-bound zip baseline package")
        spec.assert_baseline_matches(baseline)
        scenario = spec.data
        overrides = _overrides(scenario)
        coupling = _coupling(scenario)
        _validate_override_coupling(overrides, coupling)

        root = Path(work_dir).expanduser().resolve()
        baseline_dir = root / "baseline"
        compiled_dir = root / "compiled"
        forecast_inputs_dir = compiled_dir / FORECAST_INPUTS
        manifest_path = compiled_dir / COMPILED_MANIFEST
        _prepare_work_dir(root, baseline_dir, compiled_dir)

        original_package_sha = sha256_file(baseline.package_path) if baseline.is_zip else None
        materialized = baseline.materialize(baseline_dir)
        shutil.copytree(materialized / FORECAST_INPUTS, forecast_inputs_dir)

        baseline_digest = digest_input_tree(materialized / FORECAST_INPUTS)
        changed = _apply_overrides(forecast_inputs_dir, spec, overrides, coupling)
        materialized_defaults, adapter_changes = _materialize_required_adapter_assumptions(
            forecast_inputs_dir,
            baseline=baseline,
            overrides=overrides,
        )
        changed.update(adapter_changes)
        compiled_digest = digest_input_tree(forecast_inputs_dir)
        if original_package_sha is not None and sha256_file(baseline.package_path) != original_package_sha:
            raise CompilerError("baseline package bytes changed during compilation")

        manifest = {
            "schema_version": "tdcsim_cbo_compiled_scenario_manifest_v1",
            "scenario_id": spec.scenario_id,
            "scenario_sha256": spec.canonical_sha256(),
            "baseline": {
                "package_id": baseline.package_id,
                "package_sha256": baseline.package_sha256,
                "manifest_sha256": baseline.manifest_sha256,
                "release_attestation_sha256": baseline.attestation.sha256,
            },
            "baseline_forecast_inputs_digest": baseline_digest,
            "compiled_inputs_digest": compiled_digest,
            "changed_inputs": sorted(changed),
            "materialized_defaults": sorted(materialized_defaults),
            "materialized_default_count": len(materialized_defaults),
            "overrides_applied": sorted(overrides),
            "coupling": dict(coupling),
            "claim_boundary": {
                "compiler_role": "scenario_input_overlay_only",
                "does_not_run_engine": True,
                "does_not_modify_baseline_package": True,
                "net_interest_role": "diagnostic_nonbinding",
                "fed_holdings_role": "holder_allocation_target_not_total_issuance",
                "operating_cash_role": "cash_path_not_issuance_plug",
            },
            "input_hashes": input_tree_hashes(forecast_inputs_dir),
        }
        write_json(manifest_path, manifest)
        return CboCompiledScenario(
            work_dir=root,
            baseline_dir=baseline_dir,
            compiled_dir=compiled_dir,
            forecast_inputs_dir=forecast_inputs_dir,
            manifest_path=manifest_path,
            baseline_forecast_inputs_digest=baseline_digest,
            compiled_inputs_digest=compiled_digest,
            scenario_sha256=spec.canonical_sha256(),
            changed_inputs=tuple(sorted(changed)),
            manifest=manifest,
        )


def digest_input_tree(path: str | Path) -> str:
    """Return a canonical digest of all files under a forecast-input tree."""

    return canonical_json_sha256(input_tree_hashes(path))


def input_tree_hashes(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    records = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": sha256_file(file_path),
            }
        )
    return records


def _prepare_work_dir(root: Path, baseline_dir: Path, compiled_dir: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    blockers = [path for path in (baseline_dir, compiled_dir) if path.exists()]
    if blockers:
        raise CompilerError(f"compile work directory already contains generated paths: {[str(path) for path in blockers]}")


def _overrides(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    overrides = scenario.get("overrides")
    if not isinstance(overrides, Mapping):
        raise CompilerError("scenario overrides block must be a mapping")
    return overrides


def _coupling(scenario: Mapping[str, Any]) -> Mapping[str, Any]:
    coupling = scenario.get("coupling")
    if not isinstance(coupling, Mapping):
        raise CompilerError("scenario coupling block must be a mapping")
    return coupling


def _apply_overrides(
    forecast_inputs_dir: Path,
    spec: CboScenarioSpec,
    overrides: Mapping[str, Any],
    coupling: Mapping[str, Any],
) -> set[str]:
    changed: set[str] = set()

    def apply_csv(name: str, transform: Callable[[list[dict[str, str]], Mapping[str, Any]], list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
        file_name = INPUT_FILES[name]
        rows, header = _read_csv(forecast_inputs_dir / file_name)
        output = [dict(row) for row in transform(rows, _override_mapping(overrides[name]))]
        _write_csv(forecast_inputs_dir / file_name, output, preferred_header=header)
        changed.add(file_name)
        return output

    nominal_rows: list[dict[str, Any]] | None = None
    cpi_rows: list[dict[str, Any]] | None = None
    operating_cash_rows: list[dict[str, Any]] | None = None
    debt_rows: list[dict[str, Any]] | None = None
    fed_active = _fed_stock_target_present(forecast_inputs_dir / INPUT_FILES["fed_holdings"])

    if "nominal_yield_curve" in overrides:
        override = _override_mapping(overrides["nominal_yield_curve"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        nominal_rows = apply_nominal_yield_curve_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"], nominal_rows, preferred_header=header)
        changed.add(INPUT_FILES["nominal_yield_curve"])
    elif _needs_nominal_rows(overrides, coupling):
        nominal_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["nominal_yield_curve"])

    if "inflation_cpi" in overrides:
        override = _override_mapping(overrides["inflation_cpi"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        cpi_rows = apply_cpi_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"], cpi_rows, preferred_header=header)
        changed.add(INPUT_FILES["inflation_cpi"])
    elif coupling.get("operating_cash_inflation") == "scenario_cpi":
        cpi_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["inflation_cpi"])

    if "frn_benchmark" in overrides:
        override = _override_mapping(overrides["frn_benchmark"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["frn_benchmark"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_frn_override(rows, override, nominal_curve_rows=nominal_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["frn_benchmark"], output, preferred_header=header)
        changed.add(INPUT_FILES["frn_benchmark"])

    if "tips_real_yield" in overrides:
        override = _override_mapping(overrides["tips_real_yield"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_tips_real_yield_override(
            rows,
            override,
            nominal_curve_rows=nominal_rows,
            cpi_rows=cpi_rows,
            replacement_rows=replacement,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"], output, preferred_header=header)
        changed.add(INPUT_FILES["tips_real_yield"])
    elif coupling.get("tips_real_yield") == "recompute_from_nominal_and_scenario_inflation" and (
        "inflation_cpi" in overrides or "nominal_yield_curve" in overrides
    ):
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"])
        output = apply_tips_real_yield_override(
            rows,
            {"mode": "linked_recompute"},
            nominal_curve_rows=nominal_rows,
            cpi_rows=cpi_rows,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["tips_real_yield"], output, preferred_header=header)
        changed.add(INPUT_FILES["tips_real_yield"])

    if "operating_cash" in overrides:
        override = _override_mapping(overrides["operating_cash"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["operating_cash"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        operating_cash_rows = apply_operating_cash_override(
            rows,
            override,
            inflation_rows=cpi_rows if coupling.get("operating_cash_inflation") == "scenario_cpi" else None,
            replacement_rows=replacement,
        )
        _write_csv(forecast_inputs_dir / INPUT_FILES["operating_cash"], operating_cash_rows, preferred_header=header)
        changed.add(INPUT_FILES["operating_cash"])

    if "cash_reconciliation" in overrides:
        override = _override_mapping(overrides["cash_reconciliation"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["cash_reconciliation"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_cash_residual_override(rows, override, operating_cash_rows=operating_cash_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["cash_reconciliation"], output, preferred_header=header)
        changed.add(INPUT_FILES["cash_reconciliation"])

    if "primary_deficit" in overrides:
        override = _override_mapping(overrides["primary_deficit"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["primary_deficit"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_primary_deficit_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["primary_deficit"], output, preferred_header=header)
        changed.add(INPUT_FILES["primary_deficit"])

    if "debt_target" in overrides:
        override = _override_mapping(overrides["debt_target"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["debt_target"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        debt_rows = apply_debt_target_override(rows, override, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["debt_target"], debt_rows, preferred_header=header)
        changed.add(INPUT_FILES["debt_target"])
    elif "fed_holdings" in overrides:
        debt_rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["debt_target"])

    if "fed_holdings" in overrides:
        override = _override_mapping(overrides["fed_holdings"])
        rows, header = _read_csv(forecast_inputs_dir / INPUT_FILES["fed_holdings"])
        replacement = _csv_file_override_rows(spec, override, baseline_rows=rows)
        output = apply_fed_holdings_override(rows, override, marketable_debt_rows=debt_rows, replacement_rows=replacement)
        _write_csv(forecast_inputs_dir / INPUT_FILES["fed_holdings"], output, preferred_header=header)
        changed.add(INPUT_FILES["fed_holdings"])

    if "fiscal_incidence" in overrides:
        apply_csv("fiscal_incidence", apply_fiscal_incidence_override)

    if "holder_preferences" in overrides:
        override = _override_mapping(overrides["holder_preferences"])
        if override.get("mode") == "dated_static_shares":
            events = compile_holder_preference_events(override, fed_stock_target_active=fed_active)
            write_json(
                forecast_inputs_dir / HOLDER_PREFERENCE_EVENTS_FILE,
                {
                    "schema_version": "tdcsim_holder_preference_events_v1",
                    "scenario_transform": "dated_static_shares",
                    "source_role": "scenario_assumption",
                    "runtime_role": "runtime_event",
                    "claim_boundary": "holder preference profile not exact holder ownership",
                    "events": events,
                },
            )
            changed.add(HOLDER_PREFERENCE_EVENTS_FILE)
        else:
            rows, header = _read_csv(forecast_inputs_dir / "tdcsim_holder_profile_assumptions.csv")
            output = _compile_holder_preferences(rows, header, override, fed_stock_target_active=fed_active)
            _write_csv(forecast_inputs_dir / "tdcsim_holder_profile_assumptions.csv", output, preferred_header=header)
            changed.add("tdcsim_holder_profile_assumptions.csv")

    if "issuance_mix" in overrides:
        issuance_mix = compile_issuance_mix_override(_override_mapping(overrides["issuance_mix"]))
        _write_issuance_mix(forecast_inputs_dir / ISSUANCE_MIX_FILE, issuance_mix)
        changed.add(ISSUANCE_MIX_FILE)

    if "net_interest_comparator" in overrides:
        net_interest = _override_mapping(overrides["net_interest_comparator"])
        if net_interest.get("role") != "diagnostic_nonbinding":
            raise CompilerError("net_interest_comparator role must remain diagnostic_nonbinding")

    return changed


def _mmf_deposit_pass_through_override(override: Mapping[str, Any]) -> float:
    if override.get("mode") != "fixed_fraction":
        raise CompilerError("mmf_deposit_pass_through mode must be fixed_fraction")
    try:
        value = float(override["value"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CompilerError("mmf_deposit_pass_through value must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise CompilerError("mmf_deposit_pass_through value must be between 0.0 and 1.0")
    return value


def _materialize_required_adapter_assumptions(
    forecast_inputs_dir: Path,
    *,
    baseline: CboBaselinePackage,
    overrides: Mapping[str, Any],
) -> tuple[set[str], set[str]]:
    materialized: set[str] = set()
    changed: set[str] = set()
    issuance_path = forecast_inputs_dir / ISSUANCE_MIX_FILE
    if not issuance_path.exists():
        _write_issuance_mix(issuance_path, None)
        materialized.add(ISSUANCE_MIX_FILE)
        changed.add(ISSUANCE_MIX_FILE)
    runtime_path = forecast_inputs_dir / RUNTIME_ASSUMPTIONS_FILE
    if runtime_path.exists():
        runtime_payload = _read_runtime_assumptions_for_merge(runtime_path)
    else:
        runtime_payload = _runtime_assumptions_payload(
            fiscal_incidence_policy_id=_configured_default_fiscal_incidence_policy_id(
                forecast_inputs_dir
            ),
            mmf_deposit_pass_through=MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
            mmf_deposit_pass_through_status="configured_default",
            fiscal_incidence_policy_status="configured_default",
        )
        materialized.add(RUNTIME_ASSUMPTIONS_FILE)
    runtime_changed = not runtime_path.exists()
    if "mmf_deposit_pass_through" in overrides:
        runtime_payload["mmf_deposit_pass_through"] = _mmf_deposit_pass_through_override(
            _override_mapping(overrides["mmf_deposit_pass_through"])
        )
        runtime_payload["mmf_deposit_pass_through_status"] = "scenario_override"
        runtime_changed = True
    if "fiscal_incidence" in overrides:
        runtime_payload["fiscal_incidence_policy_status"] = "scenario_override"
        runtime_changed = True
    if runtime_changed:
        write_json(runtime_path, runtime_payload)
        changed.add(RUNTIME_ASSUMPTIONS_FILE)
    opening_state_path = forecast_inputs_dir / OPENING_RUNTIME_STATE_FILE
    if not opening_state_path.exists():
        if isinstance(baseline.manifest.get("derived_forecast_state"), Mapping):
            raise CompilerError(
                f"derived forecast-state package requires carried {OPENING_RUNTIME_STATE_FILE}"
            )
        _write_source_baseline_opening_runtime_state(
            opening_state_path,
            baseline=baseline,
            forecast_inputs_dir=forecast_inputs_dir,
        )
        materialized.add(OPENING_RUNTIME_STATE_FILE)
        changed.add(OPENING_RUNTIME_STATE_FILE)
    if _ensure_opening_fed_target_identity(
        forecast_inputs_dir,
        opening_state_date=_source_baseline_opening_state_date(
            baseline,
            forecast_inputs_dir=forecast_inputs_dir,
        ),
        actuals_available_as_of=str(
            baseline.manifest.get("date_range", {}).get(
                "actuals_available_as_of",
                "",
            )
        ),
        baseline_scenario_id=str(
            baseline.manifest.get("scenario_id") or ""
        ),
        fed_override_active="fed_holdings" in overrides,
    ):
        changed.add(INPUT_FILES["fed_holdings"])
    return materialized, changed


def _runtime_assumptions_payload(
    *,
    fiscal_incidence_policy_id: str,
    mmf_deposit_pass_through: float,
    mmf_deposit_pass_through_status: str,
    fiscal_incidence_policy_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": "tdcsim_cbo_runtime_assumptions_v1",
        "fiscal_incidence_policy_id": fiscal_incidence_policy_id,
        "fiscal_incidence_policy_status": fiscal_incidence_policy_status,
        "mmf_deposit_pass_through": mmf_deposit_pass_through,
        "mmf_deposit_pass_through_status": mmf_deposit_pass_through_status,
        "source_role": "compiled_adapter_assumption",
        "runtime_role": "fiscal_selector_and_deposit_channel_parameters",
        "claim_boundary": (
            "fiscal_selector_routes_signed_primary_flow_and_mmf_pass_through_changes_"
            "deposit_plumbing_not_debt_or_issuance"
        ),
    }


def _configured_default_fiscal_incidence_policy_id(
    forecast_inputs_dir: Path,
) -> str:
    rows, _ = _read_csv(
        forecast_inputs_dir / INPUT_FILES["fiscal_incidence"]
    )
    candidates = [
        str(row.get("policy_id") or "").strip()
        for row in rows
        if str(row.get("policy_id") or "").strip() == "central"
        or str(row.get("policy_id") or "").strip().endswith(
            "_central_99du_1ru"
        )
    ]
    if len(candidates) != 1:
        raise CompilerError(
            "fiscal incidence inputs require exactly one configured-default "
            "central policy ID; "
            f"found {candidates}"
        )
    return candidates[0]


def _read_runtime_assumptions_for_merge(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, ValueError) as exc:
        raise CompilerError(f"compiled runtime assumptions are unreadable: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise CompilerError("compiled runtime assumptions must be a JSON object")
    if payload.get("schema_version") != "tdcsim_cbo_runtime_assumptions_v1":
        raise CompilerError("compiled runtime assumptions has an unsupported schema_version")
    required = {
        "fiscal_incidence_policy_id",
        "fiscal_incidence_policy_status",
        "mmf_deposit_pass_through",
        "mmf_deposit_pass_through_status",
        "source_role",
        "runtime_role",
        "claim_boundary",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CompilerError(f"compiled runtime assumptions are missing required keys: {missing}")
    if not isinstance(payload["fiscal_incidence_policy_id"], str) or not str(
        payload["fiscal_incidence_policy_id"]
    ).strip():
        raise CompilerError("compiled runtime assumptions fiscal_incidence_policy_id is invalid")
    for key in ("fiscal_incidence_policy_status", "mmf_deposit_pass_through_status"):
        if payload[key] not in {"configured_default", "scenario_override"}:
            raise CompilerError(
                f"compiled runtime assumptions {key} must be configured_default or scenario_override"
            )
    try:
        mmf_value = float(payload["mmf_deposit_pass_through"])
    except (TypeError, ValueError) as exc:
        raise CompilerError(
            "compiled runtime assumptions mmf_deposit_pass_through must be numeric"
        ) from exc
    if not math.isfinite(mmf_value) or not 0.0 <= mmf_value <= 1.0:
        raise CompilerError(
            "compiled runtime assumptions mmf_deposit_pass_through must be between 0.0 and 1.0"
        )
    return dict(payload)


def _write_source_baseline_opening_runtime_state(
    path: Path,
    *,
    baseline: CboBaselinePackage,
    forecast_inputs_dir: Path,
) -> None:
    opening_state_date = _source_baseline_opening_state_date(
        baseline,
        forecast_inputs_dir=forecast_inputs_dir,
    )
    rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["operating_cash"])
    if not rows or "operating_cash_target_bil" not in rows[0]:
        raise CompilerError(
            f"{INPUT_FILES['operating_cash']} must contain opening operating_cash_target_bil"
        )
    try:
        tga = float(rows[0]["operating_cash_target_bil"])
    except (TypeError, ValueError) as exc:
        raise CompilerError(
            f"{INPUT_FILES['operating_cash']} opening operating_cash_target_bil must be numeric"
        ) from exc
    if not math.isfinite(tga):
        raise CompilerError(
            f"{INPUT_FILES['operating_cash']} opening operating_cash_target_bil must be finite"
        )
    write_json(
        path,
        {
            "schema_version": "tdcsim_cbo_opening_runtime_state_v1",
            "state_kind": "source_baseline_configured_defaults",
            "opening_state_date": opening_state_date,
            "initial_values": {
                **_SOURCE_BASELINE_OPENING_DEFAULTS,
                "tga": tga,
            },
            "initial_value_statuses": {
                "reserves": "configured_default",
                "tdc_level": "configured_default",
                "tga": "source_opening_cash",
            },
            "configured_default_count": len(_SOURCE_BASELINE_OPENING_DEFAULTS),
            "source_role": "compiler_materialized_source_baseline_opening_state",
            "runtime_role": "opening_stock_state",
            "claim_boundary": "source_baseline_defaults_not_carried_transformed_state",
        },
    )


def _source_baseline_opening_state_date(
    baseline: CboBaselinePackage,
    *,
    forecast_inputs_dir: Path,
) -> str:
    date_range = baseline.manifest.get("date_range")
    if isinstance(date_range, Mapping) and date_range.get("opening_state_date"):
        return str(date_range["opening_state_date"])
    metadata_path = forecast_inputs_dir / "tdcsim_opening_portfolio_metadata.json"
    if metadata_path.exists():
        payload = read_json(metadata_path)
        if isinstance(payload, Mapping):
            for key in ("opening_state_date", "simulation_start_date", "opening_date"):
                if payload.get(key):
                    return str(payload[key])
    rows, _ = _read_csv(forecast_inputs_dir / INPUT_FILES["primary_deficit"])
    starts = sorted(str(row["period_start"]) for row in rows if row.get("period_start"))
    if starts:
        return starts[0]
    raise CompilerError(
        "source baseline opening runtime state requires an explicit opening_state_date"
    )


def _ensure_opening_fed_target_identity(
    forecast_inputs_dir: Path,
    *,
    opening_state_date: str,
    actuals_available_as_of: str,
    baseline_scenario_id: str,
    fed_override_active: bool,
) -> bool:
    fed_path = forecast_inputs_dir / INPUT_FILES["fed_holdings"]
    rows, header = _read_csv(fed_path)
    opening_rows = [
        row
        for row in rows
        if str(row.get("period_end") or "") == opening_state_date
        and str(row.get("holder_type") or "") == "CB"
    ]
    if len(opening_rows) > 1:
        raise CompilerError(
            "Fed holdings path has duplicate opening CB target rows"
        )
    portfolio_path = forecast_inputs_dir / "tdcsim_opening_portfolio.csv"
    if not portfolio_path.exists():
        if opening_rows:
            return False
        raise CompilerError(
            "opening Fed target identity requires tdcsim_opening_portfolio.csv"
        )
    opening_cb_stock = _opening_cb_stock(portfolio_path)
    if opening_rows:
        try:
            opening_target = float(
                opening_rows[0]["cbo_fed_holdings_target_bil"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CompilerError(
                "opening Fed target must be numeric"
            ) from exc
        if not math.isfinite(opening_target):
            raise CompilerError("opening Fed target must be finite")
        if abs(opening_target - opening_cb_stock) <= 1e-6:
            return False
        if not fed_override_active:
            raise CompilerError(
                "opening Fed target does not match opening CB Treasury stock"
            )
        opening_row = opening_rows[0]
    else:
        scenario_ids = {
            str(row.get("scenario_id") or "").strip()
            for row in rows
            if str(row.get("scenario_id") or "").strip()
        }
        if not scenario_ids and baseline_scenario_id:
            scenario_ids = {baseline_scenario_id}
        if len(scenario_ids) != 1:
            raise CompilerError(
                "Fed holdings path requires exactly one scenario_id"
            )
        opening_row = {
            "schema_version": "tdcsim_fed_holdings_path_v1",
            "scenario_id": next(iter(scenario_ids)),
            "period_end": opening_state_date,
            "holder_type": "CB",
        }
        rows.append(opening_row)
    metadata_path = (
        forecast_inputs_dir / "tdcsim_opening_portfolio_metadata.json"
    )
    observation_date = opening_state_date
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if isinstance(metadata, Mapping):
            observation_date = str(
                metadata.get("record_date") or opening_state_date
            )
    opening_row.update(
        {
            "cbo_fed_holdings_target_bil": opening_cb_stock,
            "interpolation_method": "opening_state_identity",
            "source_fiscal_year": int(opening_state_date[:4]),
            "source_role": "identity_check",
            "runtime_role": "hard_target",
            "observation_date": observation_date,
            "available_date": (
                actuals_available_as_of or observation_date
            ),
            "source_status": (
                "compiler_materialized_opening_fed_stock_target_identity"
            ),
            "claim_boundary": (
                "opening_portfolio_stock_identity_nonsettling_"
                "not_observed_fed_target_growth"
            ),
            "scenario_transform": (
                "opening_state_identity_not_shocked"
                if fed_override_active
                else "configured_opening_state_identity"
            ),
        }
    )
    rows.sort(key=lambda row: str(row.get("period_end") or ""))
    _write_csv(fed_path, rows, preferred_header=header)
    return True


def _opening_cb_stock(path: Path) -> float:
    rows, _ = _read_csv(path)
    total = 0.0
    for row in rows:
        if str(row.get("Status") or "") != "Active":
            continue
        if str(row.get("HolderType") or "") != "CB":
            continue
        column = (
            "AdjustedPrincipal"
            if str(row.get("SecurityType") or "") == "TIPS"
            else "FaceValue"
        )
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompilerError(
                f"opening CB portfolio requires numeric {column}"
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise CompilerError(
                f"opening CB portfolio {column} must be finite and nonnegative"
            )
        total += value
    return total


def _validate_override_coupling(overrides: Mapping[str, Any], coupling: Mapping[str, Any]) -> None:
    for override_name, override in overrides.items():
        if isinstance(override, Mapping) and "file" in override and not _is_file_mode(str(override.get("mode") or "")):
            raise CompilerError(f"{override_name} file reference is only allowed for file-backed modes")
    frn = overrides.get("frn_benchmark")
    if isinstance(frn, Mapping):
        mode = frn.get("mode")
        if mode == "linked_to_nominal_curve" and coupling.get("frn_benchmark") != "derive_from_scenario_nominal_curve":
            raise CompilerError("linked FRN benchmark requires derive_from_scenario_nominal_curve coupling")
        if mode in {"parallel_bp", "absolute_path_file"} and coupling.get("frn_benchmark") != "independent_explicit_path":
            raise CompilerError("independent FRN benchmark overrides require independent_explicit_path coupling")
    tips = overrides.get("tips_real_yield")
    if isinstance(tips, Mapping):
        mode = tips.get("mode")
        if mode == "linked_recompute" and coupling.get("tips_real_yield") != "recompute_from_nominal_and_scenario_inflation":
            raise CompilerError("linked TIPS real-yield recompute requires recompute_from_nominal_and_scenario_inflation coupling")
        if mode in {"parallel_bp", "key_rate_bp", "absolute_path_file"} and coupling.get("tips_real_yield") != "independent_explicit_path":
            raise CompilerError("independent TIPS real-yield overrides require independent_explicit_path coupling")
    operating_cash = overrides.get("operating_cash")
    if isinstance(operating_cash, Mapping) and operating_cash.get("mode") == "constant_real":
        if coupling.get("operating_cash_inflation") not in {"baseline_cpi", "scenario_cpi", "independent_explicit_path"}:
            raise CompilerError("constant_real operating cash requires an explicit operating_cash_inflation coupling")
    if coupling.get("primary_deficit_to_debt_target") != "independent_no_plug":
        raise CompilerError("primary_deficit_to_debt_target must remain independent_no_plug")


def _is_file_mode(mode: str) -> bool:
    return mode in {
        "full_surface_file",
        "absolute_path_file",
        "monthly_path_file",
        "aggregate_path_file",
        "component_path_file",
        "explicit_path_file",
        "comparison_path_file",
    }


def _needs_nominal_rows(overrides: Mapping[str, Any], coupling: Mapping[str, Any]) -> bool:
    frn = overrides.get("frn_benchmark")
    tips = overrides.get("tips_real_yield")
    return (
        isinstance(frn, Mapping)
        and frn.get("mode") == "linked_to_nominal_curve"
        or isinstance(tips, Mapping)
        and tips.get("mode") == "linked_recompute"
        or coupling.get("frn_benchmark") == "derive_from_scenario_nominal_curve"
        and "frn_benchmark" in overrides
    )


def _override_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompilerError("override value must be a mapping")
    return value


def _csv_file_override_rows(
    spec: CboScenarioSpec,
    override: Mapping[str, Any],
    *,
    baseline_rows: list[dict[str, str]],
) -> list[dict[str, str]] | None:
    file_ref = override.get("file")
    if file_ref is None:
        return None
    if not _is_file_mode(str(override.get("mode") or "")):
        raise CompilerError("file references require a file-backed override mode")
    if not isinstance(file_ref, Mapping):
        raise CompilerError("override file reference must be a mapping")
    if spec.path is None:
        raise CompilerError("file-based overrides require a scenario file path")
    rel = str(file_ref["relative_path"])
    path = (spec.path.parent / rel).resolve()
    if spec.path.parent.resolve() not in path.parents and path != spec.path.parent.resolve():
        raise CompilerError("scenario file reference escapes scenario directory")
    expected_sha = str(file_ref["sha256"])
    if sha256_file(path) != expected_sha:
        raise CompilerError(f"scenario file SHA-256 mismatch: {rel}")
    if file_ref.get("media_type", "text/csv") != "text/csv":
        raise CompilerError("compiler currently supports CSV file overrides only")
    rows, _ = _read_csv(path)
    _assert_replacement_coverage(rows, baseline_rows, mode=str(override.get("mode") or ""))
    return rows


def _assert_replacement_coverage(
    replacement_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    *,
    mode: str,
) -> None:
    if not replacement_rows:
        raise CompilerError(f"{mode} replacement file has no rows")
    if baseline_rows and len(replacement_rows) != len(baseline_rows) and mode != "full_surface_file":
        raise CompilerError(
            f"{mode} replacement row count must match baseline coverage: "
            f"{len(replacement_rows)} != {len(baseline_rows)}"
        )
    key_cols = _replacement_key_columns(baseline_rows, replacement_rows)
    if key_cols:
        baseline_keys = _unique_keys(baseline_rows, key_cols, label="baseline")
        replacement_keys = _unique_keys(replacement_rows, key_cols, label="replacement")
        if mode == "full_surface_file":
            coverage_failed = not baseline_keys <= replacement_keys
        else:
            coverage_failed = baseline_keys != replacement_keys
        if coverage_failed:
            missing = sorted(baseline_keys - replacement_keys)[:5]
            extra = sorted(replacement_keys - baseline_keys)[:5]
            raise CompilerError(
                f"{mode} replacement row count must match baseline coverage; "
                f"key coverage mismatch for {key_cols}: "
                f"missing={missing}, extra={extra}"
            )


def _replacement_key_columns(
    baseline_rows: list[dict[str, str]],
    replacement_rows: list[dict[str, str]],
) -> tuple[str, ...]:
    if not baseline_rows or not replacement_rows:
        return ()
    baseline_cols = set(baseline_rows[0])
    replacement_cols = set(replacement_rows[0])
    for cols in (
        ("curve_date", "tenor_years"),
        ("period_start", "period_end"),
        ("period_end", "holder_type"),
        ("period_end",),
        ("month",),
        ("source_fiscal_year",),
        ("fiscal_year",),
    ):
        if set(cols) <= baseline_cols and set(cols) <= replacement_cols:
            return cols
    return ()


def _unique_keys(rows: list[dict[str, str]], cols: tuple[str, ...], *, label: str) -> set[tuple[str, ...]]:
    keys = [tuple(str(row.get(col) or "") for col in cols) for row in rows]
    if any(any(value == "" for value in key) for key in keys):
        raise CompilerError(f"{label} replacement coverage key has blank values for {cols}")
    out = set(keys)
    if len(out) != len(keys):
        raise CompilerError(f"{label} replacement coverage has duplicate keys for {cols}")
    return out


def _fed_stock_target_present(path: Path) -> bool:
    try:
        rows, _ = _read_csv(path)
    except CompilerError:
        return False
    return any("cbo_fed_holdings_target_bil" in row for row in rows)


def _write_issuance_mix(path: Path, issuance_mix: Any) -> None:
    if issuance_mix is None:
        payload = {
            "schema_version": "tdcsim_cbo_issuance_mix_assumptions_v1",
            "mode": "default_tdcsim_cbo_runner_profile",
            "selection_status": "configured_default",
            "security_shares": dict(_DEFAULT_ISSUANCE_SECURITY_SHARES),
            "maturity_distributions": {
                key: [dict(item) for item in value]
                for key, value in _DEFAULT_ISSUANCE_MATURITY_DISTRIBUTIONS.items()
            },
            "weighted_average_maturity_years": 6.8675,
            "negative_issuance_action": "error",
            "source_role": "compiler_configured_default",
            "runtime_role": "hard_target",
            "claim_boundary": "issuance_mix_is_tdcsim_scenario_assumption_not_cbo_prescription",
        }
    else:
        payload = {
            "schema_version": "tdcsim_cbo_issuance_mix_assumptions_v1",
            "mode": "replace_shares",
            "selection_status": "scenario_override",
            "security_shares": dict(issuance_mix.security_shares),
            "maturity_distributions": {
                key: [dict(item) for item in value]
                for key, value in issuance_mix.maturity_distributions.items()
            },
            "weighted_average_maturity_years": issuance_mix.weighted_average_maturity_years,
            "negative_issuance_action": issuance_mix.negative_issuance_action,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "claim_boundary": "issuance_mix_is_tdcsim_scenario_assumption_not_cbo_prescription",
        }
    write_json(path, payload)


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise CompilerError(f"required forecast input is missing: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CompilerError(f"CSV is missing a header: {path}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], *, preferred_header: list[str]) -> None:
    materialized = [dict(row) for row in rows]
    header = list(preferred_header)
    extra = sorted({key for row in materialized for key in row} - set(header))
    fieldnames = header + extra
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return value
        text = f"{value:.12f}".rstrip("0").rstrip(".")
        return text or "0"
    return value


def _compile_holder_preferences(
    baseline_rows: list[dict[str, str]],
    header: list[str],
    override: Mapping[str, Any],
    *,
    fed_stock_target_active: bool,
) -> list[dict[str, Any]]:
    validated = validate_holder_preferences(override, fed_stock_target_active=fed_stock_target_active)
    share_by_holder: dict[str, dict[str, float]] = {}
    for row in validated:
        security_type = str(row["security_type"])
        if security_type == "nonmarketable":
            continue
        column = f"{security_type}_pct"
        for holder, share in row["shares"].items():
            share_by_holder.setdefault(holder, {})[column] = share
    output: list[dict[str, Any]] = []
    seen_holders = set()
    for row in baseline_rows:
        holder = str(row.get("holder_type") or "")
        new = dict(row)
        if holder in share_by_holder:
            new.update(share_by_holder[holder])
            seen_holders.add(holder)
        new["source_role"] = "scenario_assumption"
        new["runtime_role"] = "memo_only"
        new["claim_boundary"] = "holder preference profile not exact holder ownership"
        new["scenario_transform"] = "static_shares"
        output.append(new)
    for holder in sorted(set(share_by_holder) - seen_holders):
        new = {field: "" for field in header}
        new["holder_type"] = holder
        new.update(share_by_holder[holder])
        new["source_role"] = "scenario_assumption"
        new["runtime_role"] = "memo_only"
        new["claim_boundary"] = "holder preference profile not exact holder ownership"
        new["scenario_transform"] = "static_shares"
        output.append(new)
    return output


__all__ = [
    "CboCompiledScenario",
    "CboScenarioCompiler",
    "CompilerError",
    "digest_input_tree",
    "input_tree_hashes",
]
