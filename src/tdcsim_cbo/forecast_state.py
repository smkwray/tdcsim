"""Forecast opening-state export helpers for RateWall marginal pairs."""

from __future__ import annotations

import csv
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


ALLOWED_RATE_INPUT_NAMES = {
    "tdcsim_yield_curve_surface.csv",
    "tdcsim_frn_rate_path.csv",
    "tdcsim_tips_real_yield_path.csv",
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

    final_portfolio_path = rollforward_run_dir / "outputs" / "final_portfolio_compact.csv.gz"
    if not final_portfolio_path.exists():
        raise ForecastStateExportError("rollforward run is missing final_portfolio_compact.csv.gz")
    final_portfolio = pd.read_csv(final_portfolio_path)
    if "Status" in final_portfolio.columns:
        final_portfolio = final_portfolio[final_portfolio["Status"].astype(str).eq("Active")].copy()
    _reject_stale_active(final_portfolio, state_window.opening_state_date)
    rollforward_manifest_path = rollforward_run_dir / "tdcsim_cbo_run_manifest.json"
    rollforward_manifest_sha = sha256_file(rollforward_manifest_path)

    with tempfile.TemporaryDirectory(prefix="tdcsim-forecast-state-") as tmp_name:
        tmp = Path(tmp_name)
        package_dir = tmp / "package"
        parent_baseline.materialize(package_dir)
        inputs = package_dir / "forecast_inputs"
        final_portfolio.to_csv(inputs / "tdcsim_opening_portfolio.csv", index=False)
        _slice_forecast_inputs(inputs, state_window)
        _write_opening_metadata(parent_baseline, inputs, state_window, rollforward_manifest_sha)
        _write_opening_runtime_state(inputs, state_window)

        manifest_path = package_dir / "manifest.json"
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ForecastStateExportError("parent manifest must be an object")
        date_range = dict(manifest.get("date_range", {}))
        date_range["opening_state_date"] = state_window.opening_state_date
        date_range["simulation_start_date"] = state_window.opening_state_date
        manifest["date_range"] = date_range
        manifest["derived_forecast_state"] = {
            "method": "baseline_rollforward_export_v1",
            "state_id": state_window.state_id,
            "state_period": str(state_window.state_period),
            "opening_state_date": state_window.opening_state_date,
            "parent_baseline_package_sha256": parent_baseline.package_sha256,
            "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
            "parent_attestation_sha256": parent_baseline.attestation.sha256,
            "rollforward_run_manifest_sha256": rollforward_manifest_sha,
            "forecast_state_export_manifest_sha256": "pending",
        }
        write_json(manifest_path, manifest)
        write_json(inputs / "source_contract_smoke.json", manifest)

        non_rate_digest = digest_non_rate_inputs(inputs)
        component_inventory_payload = _component_inventory_payload(inputs, state_window)
        state_fingerprint_payload = _state_fingerprint_payload(
            parent_baseline,
            state_window,
            rollforward_manifest_sha=rollforward_manifest_sha,
            derived_state_package_sha256="pending",
            forecast_state_export_manifest_sha256="pending",
            compiled_non_rate_inputs_digest=non_rate_digest,
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
            "rollforward_run_manifest_sha256": rollforward_manifest_sha,
            "compiled_non_rate_inputs_digest": non_rate_digest,
            "state_fingerprint_sha256": state_fingerprint_sha,
            "state_component_inventory_sha256": state_component_inventory_sha,
            "claim_boundary": "source_grade_cbo_baseline_rollforward_opening_state_export_not_selected_support",
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
                "name": "ratewall_forecast_state_export",
                "command": "scripts/write_ratewall_forecast_source_grade_marginal_pairs.py",
                "exit_code": 0,
            }
        ],
        "parent_baseline_package_sha256": parent_baseline.package_sha256,
        "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
        "parent_attestation_sha256": parent_baseline.attestation.sha256,
        "forecast_state_export_manifest_sha256": forecast_state_export_manifest_sha,
        "rollforward_run_manifest_sha256": rollforward_manifest_sha,
        "state_id": state_window.state_id,
        "state_period": str(state_window.state_period),
        "opening_state_date": state_window.opening_state_date,
        "actuals_available_as_of": str(parent_baseline.manifest.get("date_range", {}).get("actuals_available_as_of")),
        "source_vintage": str(parent_baseline.manifest.get("forecast_publication_date")),
        "validation_grade": "source_grade_forecast_state_rollforward_export",
        "claim_boundary": "derived_forecast_state_package_not_selected_support",
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
        rollforward_run_manifest_sha256=rollforward_manifest_sha,
        derived_state_package_sha256=verified.package_sha256,
        forecast_state_export_manifest_sha256=forecast_state_export_manifest_sha,
        state_fingerprint_sha256=state_fingerprint_sha,
        state_component_inventory_sha256=state_component_inventory_sha,
        compiled_non_rate_inputs_digest=non_rate_digest,
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
    rollforward_manifest_sha: str,
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
        "rollforward_run_manifest_sha256": rollforward_manifest_sha,
        "claim_boundary": "source_grade_cbo_baseline_rollforward_opening_state_export_not_selected_support",
    }
    write_json(inputs / "tdcsim_opening_portfolio_metadata.json", metadata)


def _write_opening_runtime_state(inputs: Path, state_window: ForecastStateWindow) -> None:
    cash = pd.read_csv(inputs / "tdcsim_operating_cash_path.csv")
    tga = float(cash.iloc[0].get("operating_cash_target_bil", 0.0))
    write_json(
        inputs / "tdcsim_opening_runtime_state.json",
        {
            "schema_version": "tdcsim_cbo_opening_runtime_state_v1",
            "opening_state_date": state_window.opening_state_date,
            "initial_values": {"tga": tga, "reserves": 3000.0, "tdc_level": 0.0},
            "claim_boundary": "opening_runtime_state_not_selected_support",
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
    rollforward_manifest_sha: str,
    derived_state_package_sha256: str,
    forecast_state_export_manifest_sha256: str,
    compiled_non_rate_inputs_digest: str,
) -> dict[str, Any]:
    return {
        "schema": "ratewall_state_fingerprint_v2",
        "source_mode": "source_grade_cbo_baseline_rollforward_export",
        "scenario_state_set_id": "ratewall_forecast_cbo_baseline_source_grade_state_set_v1",
        "state_id": state_window.state_id,
        "state_kind": "forecast_state",
        "state_period": str(state_window.state_period),
        "scenario_id": "cbo_baseline_rollforward_opening_state_v1",
        "opening_state_date": state_window.opening_state_date,
        "actuals_available_as_of": str(parent_baseline.manifest.get("date_range", {}).get("actuals_available_as_of")),
        "source_vintage": str(parent_baseline.manifest.get("forecast_publication_date")),
        "parent_baseline_package_sha256": parent_baseline.package_sha256,
        "parent_baseline_manifest_sha256": parent_baseline.manifest_sha256,
        "parent_attestation_sha256": parent_baseline.attestation.sha256,
        "rollforward_run_manifest_sha256": rollforward_manifest_sha,
        "derived_state_package_sha256": derived_state_package_sha256,
        "forecast_state_export_manifest_sha256": forecast_state_export_manifest_sha256,
        "compiled_non_rate_inputs_digest": compiled_non_rate_inputs_digest,
    }


def _zip_dir(root: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(path, path.relative_to(root).as_posix())


def _zip_member_sha256(zip_path: Path, member: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        return __import__("hashlib").sha256(zf.read(member)).hexdigest()
