"""Forecast opening-state export helpers for RateWall marginal pairs."""

from __future__ import annotations

import csv
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ._json import canonical_json_sha256, read_json, sha256_file, write_json
from .baseline import CboBaselinePackage
from .compiler import digest_input_tree, input_tree_hashes
from .contract import CboScenarioSpec
from .runner import run_cbo_scenario


class ForecastStateExportError(ValueError):
    """Raised when a forecast opening state cannot be exported safely."""


@dataclass(frozen=True)
class ForecastStateWindow:
    state_period: int
    state_id: str
    opening_state_date: str
    horizon_end_date: str
    horizon: str = "annual_h1_100bp_year"


@dataclass(frozen=True)
class ForecastStateExport:
    year: int
    state_id: str
    package_zip: Path
    attestation_path: Path
    export_manifest_path: Path
    rollforward_run_dir: Path
    rollforward_run_manifest_sha256: str
    derived_state_package_sha256: str
    forecast_state_export_manifest_sha256: str
    state_fingerprint_sha256: str
    state_component_inventory_sha256: str
    compiled_non_rate_inputs_digest: str
    compiled_controls_digest: str = ""
    exported_controls_digest: str = ""
    closing_state_digest: str = ""
    construction_kind: str = ""


@dataclass(frozen=True)
class _VerifiedRollforward:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    scenario: CboScenarioSpec
    compiled_inputs_dir: Path
    compiled_controls_digest: str
    results_path: Path
    results_sha256: str
    final_portfolio_path: Path
    final_portfolio_sha256: str
    construction_kind: str
    parent_transforms_digest: str
    transforms_digest: str


ALLOWED_RATE_INPUT_NAMES = {
    "tdcsim_yield_curve_surface.csv",
    "tdcsim_frn_rate_path.csv",
    "tdcsim_tips_real_yield_path.csv",
}

DERIVED_STATE_INPUT_NAMES = {
    "source_contract_smoke.json",
    "tdcsim_opening_portfolio.csv",
    "tdcsim_opening_portfolio_metadata.json",
    "tdcsim_opening_runtime_state.json",
}


def forecast_state_window(year: int) -> ForecastStateWindow:
    """Return the RateWall fiscal-year source-grade state window."""

    if year < 2027 or year > 2036:
        raise ForecastStateExportError("forecast source-grade states are defined only for 2027..2036")
    opening = date(year - 1, 10, 1)
    return ForecastStateWindow(
        state_period=year,
        state_id=f"cbo_baseline_state::{year}",
        opening_state_date=opening.isoformat(),
        horizon_end_date=(opening + timedelta(days=365)).isoformat(),
    )


def make_noop_cbo_scenario(
    baseline: CboBaselinePackage,
    *,
    scenario_id: str,
    start_date: str,
    end_date: str,
    title: str,
    output_profile: str = "compact",
) -> dict[str, Any]:
    """Build a no-override CBO scenario for a verified package."""

    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": scenario_id,
        "title": title,
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "provenance": {
            "kind": "user_stress_assumption",
            "label": title,
        },
        "coupling": {
            "frn_benchmark": "independent_explicit_path",
            "tips_real_yield": "independent_explicit_path",
            "operating_cash_inflation": "baseline_cpi",
            "primary_deficit_to_debt_target": "independent_no_plug",
        },
        "overrides": {},
        "output": {
            "profile": output_profile,
            "compression": "gzip",
            "catalog_sqlite": False,
        },
        "simulation": {
            "frequency": "daily",
            "start_date": start_date,
            "end_date": end_date,
        },
    }


def run_no_shock_rollforward(
    parent_baseline: CboBaselinePackage,
    *,
    state_window: ForecastStateWindow,
    output_dir: Path,
    scenario_dir: Path,
    force: bool = False,
) -> Path:
    """Run the parent baseline forward with no shock to the forecast opening date."""

    output_dir = output_dir.expanduser().resolve()
    scenario_dir = scenario_dir.expanduser().resolve()
    _prepare_empty(output_dir, force=force)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    parent_opening = str(parent_baseline.manifest.get("date_range", {}).get("opening_state_date"))
    scenario = make_noop_cbo_scenario(
        parent_baseline,
        scenario_id=f"ratewall_rollforward_to_{state_window.state_period}_source_grade_noop_v1",
        start_date=parent_opening,
        end_date=state_window.opening_state_date,
        title=f"RateWall no-shock roll-forward to {state_window.state_id}",
        output_profile="compact",
    )
    if scenario.get("overrides"):
        raise ForecastStateExportError("rollforward state construction must have no scenario overrides")
    scenario_path = scenario_dir / f"rollforward_to_{state_window.state_period}_noop.json"
    write_json(scenario_path, scenario)
    spec = CboScenarioSpec.from_file(scenario_path)
    spec.assert_baseline_matches(parent_baseline)
    run_cbo_scenario(parent_baseline, spec, output_dir, output_profile="compact")
    return output_dir


def export_forecast_state_package(
    parent_baseline: CboBaselinePackage,
    *,
    state_window: ForecastStateWindow,
    rollforward_run_dir: Path,
    output_zip: Path,
    output_attestation: Path,
    output_manifest: Path,
    force: bool = False,
) -> ForecastStateExport:
    """Export a derived CBO package whose opening state is a forecast-year state."""

    rollforward_run_dir = rollforward_run_dir.expanduser().resolve()
    output_zip = output_zip.expanduser().resolve()
    output_attestation = output_attestation.expanduser().resolve()
    output_manifest = output_manifest.expanduser().resolve()
    for path in (output_zip, output_attestation, output_manifest):
        if path.exists():
            if not force:
                raise ForecastStateExportError(f"output path exists: {path}")
            path.unlink()
        path.parent.mkdir(parents=True, exist_ok=True)

    source = _verified_rollforward_source(
        parent_baseline,
        state_window=state_window,
        rollforward_run_dir=rollforward_run_dir,
    )
    final_portfolio = pd.read_csv(source.final_portfolio_path)
    if "Status" in final_portfolio.columns:
        final_portfolio = final_portfolio[final_portfolio["Status"].astype(str).eq("Active")].copy()
    _reject_stale_active(final_portfolio, state_window.opening_state_date)
    closing_state = _verified_closing_state(
        source,
        final_portfolio=final_portfolio,
        opening_state_date=state_window.opening_state_date,
    )
    closing_state_digest = canonical_json_sha256(closing_state)
    labels = _construction_labels(source.construction_kind)

    with tempfile.TemporaryDirectory(prefix="tdcsim-forecast-state-") as tmp_name:
        tmp = Path(tmp_name)
        package_dir = tmp / "package"
        parent_baseline.materialize(package_dir)
        inputs = package_dir / "forecast_inputs"
        shutil.rmtree(inputs)
        shutil.copytree(source.compiled_inputs_dir, inputs)
        if digest_input_tree(inputs) != source.compiled_controls_digest:
            raise ForecastStateExportError("copied compiled controls do not match the verified rollforward")
        final_portfolio.to_csv(inputs / "tdcsim_opening_portfolio.csv", index=False)
        _slice_forecast_inputs(inputs, state_window)
        opening_anchor_adjustments = _anchor_opening_controls(
            inputs,
            state_window=state_window,
            closing_state=closing_state,
        )
        opening_anchor_adjustments_digest = canonical_json_sha256(opening_anchor_adjustments)
        exported_controls_digest = digest_exported_controls(inputs)
        _write_opening_metadata(
            parent_baseline,
            inputs,
            state_window,
            source=source,
            closing_state_digest=closing_state_digest,
            exported_controls_digest=exported_controls_digest,
            claim_boundary=labels["metadata_claim_boundary"],
        )
        _write_opening_runtime_state(
            inputs,
            state_window,
            closing_state=closing_state,
            closing_state_digest=closing_state_digest,
        )

        manifest_path = package_dir / "manifest.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ForecastStateExportError("parent manifest must be an object")
        date_range = dict(manifest.get("date_range", {}))
        date_range["opening_state_date"] = state_window.opening_state_date
        date_range["simulation_start_date"] = state_window.opening_state_date
        manifest["date_range"] = date_range
        manifest["derived_forecast_state"] = {
            "method": labels["method"],
            "construction_kind": source.construction_kind,
            "state_id": state_window.state_id,
            "state_period": str(state_window.state_period),
            "opening_state_date": state_window.opening_state_date,
            "horizon_end_date": state_window.horizon_end_date,
            "parent_baseline_package_sha256": parent_baseline.package_sha256,
            "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
            "parent_attestation_sha256": parent_baseline.attestation.sha256,
            "rollforward_run_manifest_sha256": source.manifest_sha256,
            "source_scenario_id": source.scenario.scenario_id,
            "source_scenario_sha256": source.scenario.canonical_sha256(),
            "parent_transforms_digest": source.parent_transforms_digest,
            "transforms_digest": source.transforms_digest,
            "compiled_controls_digest": source.compiled_controls_digest,
            "exported_controls_digest": exported_controls_digest,
            "opening_anchor_adjustments_digest": opening_anchor_adjustments_digest,
            "closing_state_digest": closing_state_digest,
            "closing_results_sha256": source.results_sha256,
            "closing_portfolio_sha256": source.final_portfolio_sha256,
            "forecast_state_export_manifest_sha256": "pending",
        }
        write_json(manifest_path, manifest)
        write_json(inputs / "source_contract_smoke.json", manifest)

        non_rate_digest = digest_non_rate_inputs(inputs)
        component_inventory_payload = _component_inventory_payload(inputs, state_window)
        state_fingerprint_payload = _state_fingerprint_payload(
            parent_baseline,
            state_window,
            source=source,
            derived_state_package_sha256="pending",
            forecast_state_export_manifest_sha256="pending",
            compiled_non_rate_inputs_digest=non_rate_digest,
            exported_controls_digest=exported_controls_digest,
            closing_state_digest=closing_state_digest,
        )
        state_fingerprint_sha = canonical_json_sha256(state_fingerprint_payload)
        state_component_inventory_sha = canonical_json_sha256(component_inventory_payload)
        export_manifest = {
            "schema_version": "tdcsim_cbo_forecast_state_export_manifest_v1",
            "state_id": state_window.state_id,
            "state_period": str(state_window.state_period),
            "opening_state_date": state_window.opening_state_date,
            "horizon_end_date": state_window.horizon_end_date,
            "parent_baseline_package_sha256": parent_baseline.package_sha256,
            "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
            "parent_attestation_sha256": parent_baseline.attestation.sha256,
            "rollforward_run_manifest_sha256": source.manifest_sha256,
            "source_scenario_id": source.scenario.scenario_id,
            "source_scenario_sha256": source.scenario.canonical_sha256(),
            "parent_transforms_digest": source.parent_transforms_digest,
            "transforms_digest": source.transforms_digest,
            "construction_kind": source.construction_kind,
            "construction_method": labels["method"],
            "compiled_controls_digest": source.compiled_controls_digest,
            "exported_controls_digest": exported_controls_digest,
            "opening_anchor_adjustments": opening_anchor_adjustments,
            "opening_anchor_adjustments_digest": opening_anchor_adjustments_digest,
            "closing_state_digest": closing_state_digest,
            "closing_results_sha256": source.results_sha256,
            "closing_portfolio_sha256": source.final_portfolio_sha256,
            "compiled_non_rate_inputs_digest": non_rate_digest,
            "state_fingerprint_sha256": state_fingerprint_sha,
            "state_component_inventory_sha256": state_component_inventory_sha,
            "claim_boundary": labels["export_claim_boundary"],
        }
        forecast_state_export_manifest_sha = canonical_json_sha256(export_manifest)
        export_manifest["forecast_state_export_manifest_sha256"] = forecast_state_export_manifest_sha
        manifest["derived_forecast_state"]["forecast_state_export_manifest_sha256"] = forecast_state_export_manifest_sha
        write_json(manifest_path, manifest)
        write_json(inputs / "source_contract_smoke.json", manifest)
        write_json(output_manifest, export_manifest)
        _zip_dir(package_dir, output_zip)

    derived_sha = sha256_file(output_zip)
    attestation = {
        "schema_version": "tdcsim_cbo_forecast_state_attestation_v1",
        "attestation_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_package_zip": str(output_zip),
        "baseline_package_zip_sha256": derived_sha,
        "baseline_manifest_sha256": _zip_member_sha256(output_zip, "manifest.json"),
        "requirements_lock_sha256": _zip_member_sha256(output_zip, "requirements.lock.txt"),
        "release_commit_sha": str(parent_baseline.attestation.release_commit_sha),
        "dirty_state": False,
        "commands": [
            {
                "name": "forecast_state_export",
                "command": "tdcsim-cbo export-forecast-state",
                "exit_code": 0,
            }
        ],
        "parent_baseline_package_sha256": parent_baseline.package_sha256,
        "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
        "parent_attestation_sha256": parent_baseline.attestation.sha256,
        "forecast_state_export_manifest_sha256": forecast_state_export_manifest_sha,
        "rollforward_run_manifest_sha256": source.manifest_sha256,
        "state_id": state_window.state_id,
        "state_period": str(state_window.state_period),
        "opening_state_date": state_window.opening_state_date,
        "actuals_available_as_of": str(parent_baseline.manifest.get("date_range", {}).get("actuals_available_as_of")),
        "source_vintage": str(parent_baseline.manifest.get("forecast_publication_date")),
        "validation_grade": labels["validation_grade"],
        "claim_boundary": labels["attestation_claim_boundary"],
    }
    write_json(output_attestation, attestation)
    verified = CboBaselinePackage.open(output_zip, attestation_path=output_attestation)
    return ForecastStateExport(
        year=state_window.state_period,
        state_id=state_window.state_id,
        package_zip=output_zip,
        attestation_path=output_attestation,
        export_manifest_path=output_manifest,
        rollforward_run_dir=rollforward_run_dir,
        rollforward_run_manifest_sha256=source.manifest_sha256,
        derived_state_package_sha256=verified.package_sha256,
        forecast_state_export_manifest_sha256=forecast_state_export_manifest_sha,
        state_fingerprint_sha256=state_fingerprint_sha,
        state_component_inventory_sha256=state_component_inventory_sha,
        compiled_non_rate_inputs_digest=non_rate_digest,
        compiled_controls_digest=source.compiled_controls_digest,
        exported_controls_digest=exported_controls_digest,
        closing_state_digest=closing_state_digest,
        construction_kind=source.construction_kind,
    )


def digest_non_rate_inputs(forecast_inputs_dir: Path) -> str:
    return canonical_json_sha256(non_rate_input_hashes(forecast_inputs_dir))


def non_rate_input_hashes(forecast_inputs_dir: Path) -> list[dict[str, Any]]:
    records = []
    for item in input_tree_hashes(forecast_inputs_dir):
        if Path(str(item["path"])).name in ALLOWED_RATE_INPUT_NAMES:
            continue
        records.append(item)
    return records


def digest_exported_controls(forecast_inputs_dir: Path) -> str:
    """Digest the carried controls without derived opening-state surfaces."""

    records = [
        item
        for item in input_tree_hashes(forecast_inputs_dir)
        if Path(str(item["path"])).name not in DERIVED_STATE_INPUT_NAMES
    ]
    return canonical_json_sha256(records)


def _verified_rollforward_source(
    parent_baseline: CboBaselinePackage,
    *,
    state_window: ForecastStateWindow,
    rollforward_run_dir: Path,
) -> _VerifiedRollforward:
    root = rollforward_run_dir.expanduser().resolve()
    manifest_path = root / "tdcsim_cbo_run_manifest.json"
    if not manifest_path.is_file():
        raise ForecastStateExportError("rollforward run manifest is missing")
    manifest_sha_before = sha256_file(manifest_path)
    try:
        from .verifier import verify_scenario_run

        verification = verify_scenario_run(root)
    except Exception as exc:
        raise ForecastStateExportError(f"rollforward run verification failed: {exc}") from exc
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != manifest_sha_before:
        raise ForecastStateExportError("rollforward run manifest changed during verification")
    if verification.get("status") != "pass":
        raise ForecastStateExportError("rollforward run did not pass verification")

    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ForecastStateExportError("rollforward run manifest must be an object")
    _require_parent_identity(parent_baseline, manifest)
    _require_rollforward_dates(parent_baseline, state_window, manifest)

    scenario_block = manifest.get("scenario")
    if not isinstance(scenario_block, Mapping):
        raise ForecastStateExportError("rollforward run scenario block must be an object")
    scenario_path = _safe_run_path(root, str(scenario_block.get("relative_path") or ""), label="scenario")
    scenario = CboScenarioSpec.from_file(scenario_path)
    scenario.assert_baseline_matches(parent_baseline)
    if scenario.scenario_id != str(scenario_block.get("scenario_id") or ""):
        raise ForecastStateExportError("rollforward scenario id does not match the run manifest")
    if scenario.canonical_sha256() != str(scenario_block.get("canonical_sha256") or ""):
        raise ForecastStateExportError("rollforward scenario digest does not match the run manifest")
    scenario_simulation = scenario.data.get("simulation")
    manifest_simulation = manifest.get("simulation")
    if not isinstance(scenario_simulation, Mapping) or not isinstance(manifest_simulation, Mapping):
        raise ForecastStateExportError("rollforward simulation blocks must be objects")
    for key in ("start_date", "end_date", "frequency"):
        if str(scenario_simulation.get(key) or "") != str(manifest_simulation.get(key) or ""):
            raise ForecastStateExportError(f"rollforward scenario {key} does not match the run manifest")

    compiled_manifest_path = _safe_run_path(
        root,
        str(manifest.get("compiled_manifest") or ""),
        label="compiled manifest",
    )
    compiled_inputs_dir = compiled_manifest_path.parent / "forecast_inputs"
    if not compiled_inputs_dir.is_dir():
        raise ForecastStateExportError("verified rollforward compiled controls are missing")
    compiled_controls_digest = str(manifest.get("compiled_inputs_digest") or "")
    if digest_input_tree(compiled_inputs_dir) != compiled_controls_digest:
        raise ForecastStateExportError("rollforward compiled controls changed after verification")

    results_path, results_sha = _run_output_artifact(root, manifest, "results")
    final_portfolio_path, final_portfolio_sha = _run_output_artifact(root, manifest, "final_portfolio")
    overrides = scenario.data.get("overrides")
    if not isinstance(overrides, Mapping):
        raise ForecastStateExportError("rollforward scenario overrides must be an object")
    parent_derived = parent_baseline.manifest.get("derived_forecast_state")
    if parent_derived is not None and not isinstance(parent_derived, Mapping):
        raise ForecastStateExportError("parent derived_forecast_state must be an object")
    parent_kind = ""
    parent_transforms_digest = ""
    if isinstance(parent_derived, Mapping):
        parent_kind = str(parent_derived.get("construction_kind") or "")
        if not parent_kind and parent_derived.get("method") == "baseline_rollforward_export_v1":
            parent_kind = "baseline_noop"
        if parent_kind not in {"baseline_noop", "transformed_scenario"}:
            raise ForecastStateExportError("parent forecast-state construction kind is unsupported")
        parent_transforms_digest = str(parent_derived.get("transforms_digest") or "")
    construction_kind = (
        "transformed_scenario"
        if overrides or parent_kind == "transformed_scenario"
        else "baseline_noop"
    )
    transforms_digest = canonical_json_sha256(
        {
            "parent_transforms_digest": parent_transforms_digest,
            "scenario_overrides": dict(overrides),
        }
    )
    return _VerifiedRollforward(
        root=root,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        scenario=scenario,
        compiled_inputs_dir=compiled_inputs_dir,
        compiled_controls_digest=compiled_controls_digest,
        results_path=results_path,
        results_sha256=results_sha,
        final_portfolio_path=final_portfolio_path,
        final_portfolio_sha256=final_portfolio_sha,
        construction_kind=construction_kind,
        parent_transforms_digest=parent_transforms_digest,
        transforms_digest=transforms_digest,
    )


def _require_parent_identity(parent_baseline: CboBaselinePackage, run_manifest: Mapping[str, Any]) -> None:
    baseline = run_manifest.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ForecastStateExportError("rollforward run baseline block must be an object")
    expected = {
        "package_id": parent_baseline.package_id,
        "package_sha256": parent_baseline.package_sha256,
        "manifest_sha256": parent_baseline.manifest_sha256,
        "release_attestation_sha256": parent_baseline.attestation.sha256,
    }
    for key, value in expected.items():
        if str(baseline.get(key) or "") != str(value):
            raise ForecastStateExportError(f"rollforward run baseline {key} does not match the parent")


def _require_rollforward_dates(
    parent_baseline: CboBaselinePackage,
    state_window: ForecastStateWindow,
    run_manifest: Mapping[str, Any],
) -> None:
    simulation = run_manifest.get("simulation")
    if not isinstance(simulation, Mapping):
        raise ForecastStateExportError("rollforward run simulation block must be an object")
    parent_dates = parent_baseline.manifest.get("date_range")
    if not isinstance(parent_dates, Mapping):
        raise ForecastStateExportError("parent baseline date_range must be an object")
    parent_opening = str(
        parent_dates.get("simulation_start_date")
        or parent_dates.get("opening_state_date")
        or ""
    )
    if str(simulation.get("start_date") or "") != parent_opening:
        raise ForecastStateExportError("rollforward run does not start at the parent opening state")
    if str(simulation.get("end_date") or "") != state_window.opening_state_date:
        raise ForecastStateExportError("rollforward run does not close at the requested opening state")
    if str(simulation.get("frequency") or "") != "daily":
        raise ForecastStateExportError("forecast-state rollforward must use daily frequency")
    opening = pd.Timestamp(state_window.opening_state_date)
    horizon_end = pd.Timestamp(state_window.horizon_end_date)
    if pd.isna(opening) or pd.isna(horizon_end) or horizon_end <= opening:
        raise ForecastStateExportError("forecast-state horizon must end after its opening date")


def _safe_run_path(root: Path, relative: str, *, label: str) -> Path:
    rel = Path(relative)
    if not relative or rel.is_absolute() or ".." in rel.parts:
        raise ForecastStateExportError(f"rollforward {label} path must be package-relative")
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ForecastStateExportError(f"rollforward {label} path escapes the run root") from exc
    if not path.is_file():
        raise ForecastStateExportError(f"rollforward {label} is missing")
    return path


def _run_output_artifact(
    root: Path,
    run_manifest: Mapping[str, Any],
    logical_key: str,
) -> tuple[Path, str]:
    output_manifest = run_manifest.get("output_manifest")
    if not isinstance(output_manifest, Mapping):
        raise ForecastStateExportError("rollforward output_manifest must be an object")
    artifact = output_manifest.get(logical_key)
    if not isinstance(artifact, Mapping):
        raise ForecastStateExportError(f"rollforward output_manifest.{logical_key} must be an object")
    path = _safe_run_path(root, f"outputs/{artifact.get('path') or ''}", label=logical_key)
    sha = str(artifact.get("sha256") or "")
    if sha256_file(path) != sha:
        raise ForecastStateExportError(f"rollforward {logical_key} changed after verification")
    try:
        expected_bytes = int(artifact.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ForecastStateExportError(f"rollforward {logical_key} byte count is invalid") from exc
    if path.stat().st_size != expected_bytes:
        raise ForecastStateExportError(f"rollforward {logical_key} byte count changed after verification")
    return path, sha


def _prepare_empty(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise ForecastStateExportError(f"output path exists: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _reject_stale_active(portfolio: pd.DataFrame, opening_state_date: str) -> None:
    if "MaturityDate" not in portfolio.columns:
        return
    maturity = pd.to_datetime(portfolio["MaturityDate"], errors="coerce")
    stale = portfolio[maturity <= pd.Timestamp(opening_state_date)]
    if not stale.empty:
        raise ForecastStateExportError("rollforward final portfolio contains stale active securities")


def _verified_closing_state(
    source: _VerifiedRollforward,
    *,
    final_portfolio: pd.DataFrame,
    opening_state_date: str,
) -> dict[str, Any]:
    results = pd.read_csv(source.results_path)
    if results.empty or "Date" not in results.columns:
        raise ForecastStateExportError("rollforward results do not contain a dated closing state")
    dates = pd.to_datetime(results["Date"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ForecastStateExportError("rollforward results contain malformed dates")
    opening = pd.Timestamp(opening_state_date).normalize()
    if dates.iloc[-1] != opening:
        raise ForecastStateExportError("rollforward results do not end at the requested opening state")
    matches = results.loc[dates.eq(opening)]
    if len(matches) != 1:
        raise ForecastStateExportError("rollforward results must contain exactly one closing-state row")
    closing = matches.iloc[0]
    required = ("TGA", "Reserves", "TDC_Level", "CBOFedHoldingsTarget", "CBOFedHoldingsTargetError")
    missing = [column for column in required if column not in closing.index]
    if missing:
        raise ForecastStateExportError(f"rollforward results omit required closing stocks: {missing}")
    try:
        values = {column: float(closing[column]) for column in required}
    except (TypeError, ValueError) as exc:
        raise ForecastStateExportError("rollforward closing stocks must be numeric") from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ForecastStateExportError("rollforward closing stocks must be finite")

    fed_holdings = _fed_stock_from_portfolio(final_portfolio)
    fed_target = values["CBOFedHoldingsTarget"]
    fed_error = values["CBOFedHoldingsTargetError"]
    tolerance = 1e-6
    if abs((fed_holdings - fed_target) - fed_error) > tolerance:
        raise ForecastStateExportError("closing Fed stock/target relationship does not match the final portfolio")
    if "DebtHeld_CB" in closing.index:
        try:
            reported_fed_holdings = float(closing["DebtHeld_CB"])
        except (TypeError, ValueError) as exc:
            raise ForecastStateExportError("rollforward closing DebtHeld_CB must be numeric") from exc
        if not math.isfinite(reported_fed_holdings) or abs(reported_fed_holdings - fed_holdings) > tolerance:
            raise ForecastStateExportError("closing DebtHeld_CB does not match the final portfolio")
    if sha256_file(source.results_path) != source.results_sha256:
        raise ForecastStateExportError("rollforward results changed while reading the closing state")
    if sha256_file(source.final_portfolio_path) != source.final_portfolio_sha256:
        raise ForecastStateExportError("rollforward final portfolio changed while reading the closing state")

    return {
        "schema_version": "tdcsim_cbo_verified_closing_state_v1",
        "opening_state_date": opening_state_date,
        "initial_values": {
            "tga": values["TGA"],
            "reserves": values["Reserves"],
            "tdc_level": values["TDC_Level"],
        },
        "fed_state": {
            "holdings_bil": fed_holdings,
            "target_bil": fed_target,
            "target_error_bil": fed_error,
            "target_error_definition": "holdings_bil_minus_target_bil",
            "holdings_basis": "face_except_tips_adjusted_principal",
            "relationship_status": "verified",
        },
        "source_run_manifest_sha256": source.manifest_sha256,
        "source_results_sha256": source.results_sha256,
        "source_final_portfolio_sha256": source.final_portfolio_sha256,
    }


def _fed_stock_from_portfolio(final_portfolio: pd.DataFrame) -> float:
    required = {"SecurityType", "FaceValue", "HolderType"}
    missing = sorted(required - set(final_portfolio.columns))
    if missing:
        raise ForecastStateExportError(f"final portfolio omits Fed stock columns: {missing}")
    face = pd.to_numeric(final_portfolio["FaceValue"], errors="coerce")
    if face.isna().any() or (~face.map(math.isfinite)).any() or (face < 0.0).any():
        raise ForecastStateExportError("final portfolio FaceValue must be finite and nonnegative")
    debt_base = face.copy()
    tips = final_portfolio["SecurityType"].astype(str).eq("TIPS")
    if tips.any():
        if "AdjustedPrincipal" not in final_portfolio.columns:
            raise ForecastStateExportError("final TIPS portfolio omits AdjustedPrincipal")
        adjusted = pd.to_numeric(final_portfolio.loc[tips, "AdjustedPrincipal"], errors="coerce")
        if adjusted.isna().any() or (~adjusted.map(math.isfinite)).any() or (adjusted < 0.0).any():
            raise ForecastStateExportError("final TIPS AdjustedPrincipal must be finite and nonnegative")
        debt_base.loc[tips] = adjusted
    fed = final_portfolio["HolderType"].astype(str).eq("CB")
    return float(debt_base.loc[fed].sum())


def _anchor_opening_controls(
    inputs: Path,
    *,
    state_window: ForecastStateWindow,
    closing_state: Mapping[str, Any],
) -> dict[str, Any]:
    initial_values = closing_state.get("initial_values")
    fed_state = closing_state.get("fed_state")
    if not isinstance(initial_values, Mapping) or not isinstance(fed_state, Mapping):
        raise ForecastStateExportError("verified closing state is incomplete")
    opening = pd.Timestamp(state_window.opening_state_date).normalize()
    cash_adjustment = _anchor_opening_control(
        inputs / "tdcsim_operating_cash_path.csv",
        opening=opening,
        target_column="operating_cash_target_bil",
        target_value=float(initial_values["tga"]),
        extra_values={
            "tga_target_bil": float(initial_values["tga"]),
            "base_balance_bil": float(initial_values["tga"]),
            "base_date": state_window.opening_state_date,
            "runtime_role": "opening_state_anchor",
            "source_status": "verified_rollforward_closing_tga_anchor",
            "claim_boundary": "verified_closing_tga_anchor_not_future_operating_cash_control",
        },
    )
    fed_adjustment = _anchor_opening_control(
        inputs / "tdcsim_fed_holdings_path.csv",
        opening=opening,
        target_column="cbo_fed_holdings_target_bil",
        target_value=float(fed_state["target_bil"]),
        extra_values={
            "source_status": "verified_rollforward_closing_fed_target_anchor",
            "claim_boundary": "verified_closing_fed_target_anchor_and_future_holder_stock_target",
        },
        holder_type="CB",
    )
    return {
        "schema_version": "tdcsim_cbo_opening_control_adjustments_v1",
        "opening_state_date": state_window.opening_state_date,
        "operating_cash": cash_adjustment,
        "fed_holdings": fed_adjustment,
    }


def _anchor_opening_control(
    path: Path,
    *,
    opening: pd.Timestamp,
    target_column: str,
    target_value: float,
    extra_values: Mapping[str, Any],
    holder_type: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise ForecastStateExportError(f"opening control is missing: {path.name}")
    frame = pd.read_csv(path)
    if frame.empty or "period_end" not in frame.columns or target_column not in frame.columns:
        raise ForecastStateExportError(f"{path.name} cannot carry an opening-state anchor")
    dates = pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ForecastStateExportError(f"{path.name} contains malformed period_end values")
    candidate_mask = dates.eq(opening)
    if holder_type is not None:
        if "holder_type" not in frame.columns:
            raise ForecastStateExportError(f"{path.name} omits holder_type")
        candidate_mask &= frame["holder_type"].astype(str).eq(holder_type)
    matches = frame.index[candidate_mask].tolist()
    action = "updated_existing_boundary"
    if len(matches) > 1:
        raise ForecastStateExportError(f"{path.name} has duplicate opening-state controls")
    if not matches:
        future_mask = dates.gt(opening)
        if holder_type is not None:
            future_mask &= frame["holder_type"].astype(str).eq(holder_type)
        future = frame.index[future_mask].tolist()
        if not future:
            raise ForecastStateExportError(f"{path.name} has no future row to anchor")
        anchor = frame.loc[future[0]].copy()
        anchor["period_end"] = opening.date().isoformat()
        frame = pd.concat([pd.DataFrame([anchor]), frame], ignore_index=True)
        row_index = 0
        action = "inserted_boundary"
    else:
        row_index = matches[0]
    original_target = float(pd.to_numeric(pd.Series([frame.loc[row_index, target_column]]), errors="coerce").iloc[0])
    if not math.isfinite(original_target):
        raise ForecastStateExportError(f"{path.name} opening target must be finite")
    frame.loc[row_index, target_column] = target_value
    for column, value in extra_values.items():
        if column in frame.columns:
            frame.loc[row_index, column] = value
    frame["_opening_order"] = (
        pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize().astype("int64")
    )
    sort_columns = ["_opening_order"]
    if holder_type is not None and "holder_type" in frame.columns:
        sort_columns.append("holder_type")
    frame = frame.sort_values(sort_columns, kind="stable").drop(columns="_opening_order")
    frame.to_csv(path, index=False)
    return {
        "file": path.name,
        "action": action,
        "target_column": target_column,
        "source_target_bil": original_target,
        "opening_target_bil": target_value,
    }


def _slice_forecast_inputs(inputs: Path, state_window: ForecastStateWindow) -> None:
    opening = pd.Timestamp(state_window.opening_state_date)
    horizon_end = pd.Timestamp(state_window.horizon_end_date)
    for filename in (
        "tdcsim_primary_deficit_path.csv",
        "tdcsim_cash_reconciliation_residual.csv",
        "tdcsim_debt_stock_path.csv",
        "tdcsim_operating_cash_path.csv",
        "tdcsim_fed_holdings_path.csv",
        "tdcsim_tips_cpi_path.csv",
        "tdcsim_macro_forecast_path.csv",
    ):
        path = inputs / filename
        if path.exists():
            _slice_dated_csv(path, opening=opening)
    for filename in ("tdcsim_yield_curve_surface.csv", "tdcsim_frn_rate_path.csv", "tdcsim_tips_real_yield_path.csv"):
        path = inputs / filename
        if path.exists():
            _slice_rate_like_csv(path, opening=opening, horizon_end=horizon_end)


def _slice_dated_csv(path: Path, *, opening: pd.Timestamp) -> None:
    frame = pd.read_csv(path)
    if frame.empty:
        return
    if {"period_start", "period_end"} <= set(frame.columns):
        end = pd.to_datetime(frame["period_end"], errors="coerce")
        frame = frame[end > opening].copy()
        if frame.empty:
            raise ForecastStateExportError(f"{path.name} has no rows after opening date")
        frame.iloc[0, frame.columns.get_loc("period_start")] = opening.date().isoformat()
    elif "period_end" in frame.columns:
        end = pd.to_datetime(frame["period_end"], errors="coerce")
        frame = frame[end >= opening].copy()
    elif "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[dates >= opening].copy()
    frame.to_csv(path, index=False)


def _slice_rate_like_csv(path: Path, *, opening: pd.Timestamp, horizon_end: pd.Timestamp) -> None:
    frame = pd.read_csv(path)
    if frame.empty:
        return
    date_col = next((col for col in ("curve_date", "date", "period_end") if col in frame.columns), None)
    if date_col is None:
        return
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    keep = frame[(dates >= opening) & (dates <= horizon_end)].copy()
    before = frame[dates <= opening].copy()
    if not before.empty:
        key_cols = [col for col in ("tenor_years", "benchmark", "series_id") if col in frame.columns]
        if key_cols:
            boundary = before.sort_values(date_col).groupby(key_cols, dropna=False).tail(1).copy()
        else:
            boundary = before.sort_values(date_col).tail(1).copy()
        boundary[date_col] = opening.date().isoformat()
        keep = pd.concat([boundary, keep], ignore_index=True)
    if keep.empty:
        raise ForecastStateExportError(f"{path.name} has no coverage for opening window")
    keep = keep.drop_duplicates().sort_values(list(frame.columns[: min(2, len(frame.columns))]))
    keep.to_csv(path, index=False)


def _write_opening_metadata(
    parent_baseline: CboBaselinePackage,
    inputs: Path,
    state_window: ForecastStateWindow,
    *,
    source: _VerifiedRollforward,
    closing_state_digest: str,
    exported_controls_digest: str,
    claim_boundary: str,
) -> None:
    metadata = {
        "schema_version": "tdcsim_opening_portfolio_metadata_v1",
        "opening_state_date": state_window.opening_state_date,
        "simulation_start_date": state_window.opening_state_date,
        "state_id": state_window.state_id,
        "state_period": str(state_window.state_period),
        "source_vintage": str(parent_baseline.manifest.get("forecast_publication_date")),
        "actuals_available_as_of": str(parent_baseline.manifest.get("date_range", {}).get("actuals_available_as_of")),
        "parent_baseline_package_sha256": parent_baseline.package_sha256,
        "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
        "parent_attestation_sha256": parent_baseline.attestation.sha256,
        "rollforward_run_manifest_sha256": source.manifest_sha256,
        "source_scenario_id": source.scenario.scenario_id,
        "source_scenario_sha256": source.scenario.canonical_sha256(),
        "parent_transforms_digest": source.parent_transforms_digest,
        "transforms_digest": source.transforms_digest,
        "construction_kind": source.construction_kind,
        "compiled_controls_digest": source.compiled_controls_digest,
        "exported_controls_digest": exported_controls_digest,
        "closing_state_digest": closing_state_digest,
        "closing_portfolio_sha256": source.final_portfolio_sha256,
        "claim_boundary": claim_boundary,
    }
    write_json(inputs / "tdcsim_opening_portfolio_metadata.json", metadata)


def _write_opening_runtime_state(
    inputs: Path,
    state_window: ForecastStateWindow,
    *,
    closing_state: Mapping[str, Any],
    closing_state_digest: str,
) -> None:
    initial_values = closing_state.get("initial_values")
    fed_state = closing_state.get("fed_state")
    if not isinstance(initial_values, Mapping) or not isinstance(fed_state, Mapping):
        raise ForecastStateExportError("verified closing state is incomplete")
    write_json(
        inputs / "tdcsim_opening_runtime_state.json",
        {
            "schema_version": "tdcsim_cbo_opening_runtime_state_v1",
            "opening_state_date": state_window.opening_state_date,
            "initial_values": dict(initial_values),
            "fed_state": dict(fed_state),
            "closing_state_digest": closing_state_digest,
            "source_run_manifest_sha256": closing_state["source_run_manifest_sha256"],
            "source_results_sha256": closing_state["source_results_sha256"],
            "source_final_portfolio_sha256": closing_state["source_final_portfolio_sha256"],
            "claim_boundary": "verified_rollforward_close_carried_as_next_opening_not_selected_support",
        },
    )


def _component_inventory_payload(inputs: Path, state_window: ForecastStateWindow) -> dict[str, Any]:
    return {
        "schema": "ratewall_state_component_inventory_v2",
        "state_id": state_window.state_id,
        "non_rate_input_hashes": non_rate_input_hashes(inputs),
        "allowed_rate_input_names": sorted(ALLOWED_RATE_INPUT_NAMES),
    }


def _state_fingerprint_payload(
    parent_baseline: CboBaselinePackage,
    state_window: ForecastStateWindow,
    *,
    source: _VerifiedRollforward,
    derived_state_package_sha256: str,
    forecast_state_export_manifest_sha256: str,
    compiled_non_rate_inputs_digest: str,
    exported_controls_digest: str,
    closing_state_digest: str,
) -> dict[str, Any]:
    labels = _construction_labels(source.construction_kind)
    return {
        "schema": "ratewall_state_fingerprint_v2",
        "source_mode": labels["source_mode"],
        "scenario_state_set_id": labels["scenario_state_set_id"],
        "state_id": state_window.state_id,
        "state_kind": "forecast_state",
        "state_period": str(state_window.state_period),
        "scenario_id": labels["fingerprint_scenario_id"] or source.scenario.scenario_id,
        "opening_state_date": state_window.opening_state_date,
        "actuals_available_as_of": str(parent_baseline.manifest.get("date_range", {}).get("actuals_available_as_of")),
        "source_vintage": str(parent_baseline.manifest.get("forecast_publication_date")),
        "parent_baseline_package_sha256": parent_baseline.package_sha256,
        "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
        "parent_attestation_sha256": parent_baseline.attestation.sha256,
        "rollforward_run_manifest_sha256": source.manifest_sha256,
        "source_scenario_sha256": source.scenario.canonical_sha256(),
        "parent_transforms_digest": source.parent_transforms_digest,
        "transforms_digest": source.transforms_digest,
        "construction_kind": source.construction_kind,
        "compiled_controls_digest": source.compiled_controls_digest,
        "exported_controls_digest": exported_controls_digest,
        "closing_state_digest": closing_state_digest,
        "derived_state_package_sha256": derived_state_package_sha256,
        "forecast_state_export_manifest_sha256": forecast_state_export_manifest_sha256,
        "compiled_non_rate_inputs_digest": compiled_non_rate_inputs_digest,
    }


def _construction_labels(construction_kind: str) -> dict[str, str]:
    if construction_kind == "baseline_noop":
        return {
            "method": "baseline_rollforward_export_v1",
            "source_mode": "source_grade_cbo_baseline_rollforward_export",
            "scenario_state_set_id": "ratewall_forecast_cbo_baseline_source_grade_state_set_v1",
            "fingerprint_scenario_id": "cbo_baseline_rollforward_opening_state_v1",
            "metadata_claim_boundary": (
                "source_grade_cbo_baseline_rollforward_opening_state_export_not_selected_support"
            ),
            "export_claim_boundary": (
                "source_grade_cbo_baseline_rollforward_opening_state_export_not_selected_support"
            ),
            "validation_grade": "source_grade_forecast_state_rollforward_export",
            "attestation_claim_boundary": "derived_forecast_state_package_not_selected_support",
        }
    if construction_kind == "transformed_scenario":
        return {
            "method": "transformed_scenario_rollforward_export_v1",
            "source_mode": "transformed_cbo_scenario_rollforward_export",
            "scenario_state_set_id": "transformed_cbo_scenario_continuous_state_set_v1",
            "fingerprint_scenario_id": "",
            "metadata_claim_boundary": (
                "transformed_scenario_rollforward_opening_state_export_not_source_baseline_not_selected_support"
            ),
            "export_claim_boundary": (
                "transformed_scenario_rollforward_opening_state_export_not_source_baseline_not_selected_support"
            ),
            "validation_grade": "transformed_scenario_forecast_state_rollforward_export",
            "attestation_claim_boundary": (
                "derived_transformed_forecast_state_package_not_source_baseline_not_selected_support"
            ),
        }
    raise ForecastStateExportError(f"unsupported forecast-state construction kind: {construction_kind}")


def _zip_dir(root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(root).as_posix())


def _zip_member_sha256(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return __import__("hashlib").sha256(zf.read(member)).hexdigest()
