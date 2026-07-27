#!/usr/bin/env python3
"""Write zero-support RateWall assumption-mode marginal TDC pair fixtures."""

from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from tdcsim_cbo._json import canonical_json_sha256, sha256_file, write_json
from tdcsim_cbo.marginal_tdc import (
    DENOMINATOR_EQUIVALENCE_KEY,
    OBJECT_ID,
    SHOCK_PATH_ID,
    assemble_marginal_tdc_pair,
)


SOURCE_VINTAGE = "ratewall_assumption_mode_fixture_20260630"
BETA = 0.34201759129420367
CHI = 0.07
CURRENT_OPENING_TDC_STOCK_BIL = "1038.044030080004"
CURRENT_NOMINAL_GDP_BIL = "31902.006"
BASELINE_RATE_DECIMAL = 0.04
TENORS = ("0.25", "2", "5", "10", "30")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.ratewall_project_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    periods = args.period or ["current:2026"]

    written: list[Path] = []
    for raw_period in periods:
        fixture = _fixture_for(raw_period, project_root)
        _write_source_run(fixture, "baseline")
        _write_source_run(fixture, "shock")
        spec = _pair_spec(fixture)
        spec_dir = project_root / "var" / "preliminary_scenario_results" / "marginal_tdcsim" / "pair_specs"
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"{fixture['pair_id']}.json"
        write_json(spec_path, spec)
        pair_dir = output_root / str(fixture["pair_dir_name"])
        result = assemble_marginal_tdc_pair(spec, pair_dir, require_source_verification=False)
        written.append(result.output_dir)

    for path in written:
        print(path)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratewall-project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--period",
        action="append",
        help="Period to write, as current:2026 or forecast:YYYY. May be repeated.",
    )
    return parser


def _fixture_for(raw_period: str, project_root: Path) -> dict[str, Any]:
    try:
        kind, year_text = raw_period.split(":", 1)
        year = int(year_text)
    except ValueError as exc:
        raise SystemExit(f"invalid --period value: {raw_period!r}") from exc
    if kind not in {"current", "forecast"}:
        raise SystemExit(f"unsupported period kind: {kind!r}")
    if kind == "current" and year != 2026:
        raise SystemExit("current fixtures currently support only current:2026")
    if kind == "forecast" and not (2027 <= year <= 2036):
        raise SystemExit("forecast fixtures support forecast:2027 through forecast:2036")

    if kind == "current":
        horizon_start = date(2026, 6, 21)
        horizon_end = date(2027, 6, 21)
        state_kind = "current_state"
        scenario_state_set_id = "ratewall_current_state_set_v1"
        state_id = "current_state::2026"
        scenario_id = "current_state_assumption_v1"
        pair_id = "ratewall_current_2026_plus100bp_year_assumption_pair_v1"
        baseline_scenario_id = "current_state_no_incremental_shock_v1"
        shock_scenario_id = "current_state_plus100bp_year_v1"
        pair_dir_name = "current_state_2026_plus_100bp_year"
        run_base = project_root / "var" / "preliminary_scenario_results" / "marginal_tdcsim" / "source_runs" / "current_state_2026"
    else:
        horizon_start = date(year, 1, 1)
        horizon_end = horizon_start + timedelta(days=365)
        state_kind = "forecast_state"
        scenario_state_set_id = "ratewall_forecast_cbo_baseline_state_set_v1"
        state_id = f"cbo_baseline_state::{year}"
        scenario_id = "cbo_baseline_state_assumption_v1"
        pair_id = f"ratewall_forecast_cbo_baseline_{year}_plus100bp_year_assumption_pair_v1"
        baseline_scenario_id = f"cbo_baseline_{year}_no_incremental_shock_v1"
        shock_scenario_id = f"cbo_baseline_{year}_plus100bp_year_v1"
        pair_dir_name = f"forecast_cbo_baseline_{year}_plus_100bp_year"
        run_base = project_root / "var" / "preliminary_scenario_results" / "marginal_tdcsim" / "source_runs" / f"forecast_cbo_baseline_{year}"

    state_payload = {
        "schema": "ratewall_state_fingerprint_v1",
        "source_mode": "assumption_mode",
        "scenario_state_set_id": scenario_state_set_id,
        "state_id": state_id,
        "state_kind": state_kind,
        "state_period": str(year),
        "scenario_id": scenario_id,
        "opening_state_date": horizon_start.isoformat(),
        "nominal_gdp_bil": CURRENT_NOMINAL_GDP_BIL if kind == "current" else "",
        "opening_tdc_stock_bil": CURRENT_OPENING_TDC_STOCK_BIL,
        "shock_path_id": SHOCK_PATH_ID,
    }
    inventory_payload = {
        "schema": "ratewall_state_component_inventory_v1",
        "state_id": state_id,
        "compiled_non_rate_inputs": ["tdcsim_primary_deficit_path.csv"],
        "opening_route_stock_total_bil": CURRENT_OPENING_TDC_STOCK_BIL,
        "assumption_mode_zero_tdc_delta": True,
    }
    return {
        "kind": kind,
        "year": year,
        "scenario_state_set_id": scenario_state_set_id,
        "state_id": state_id,
        "state_kind": state_kind,
        "scenario_id": scenario_id,
        "pair_id": pair_id,
        "baseline_scenario_id": baseline_scenario_id,
        "shock_scenario_id": shock_scenario_id,
        "pair_dir_name": pair_dir_name,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "state_fingerprint_sha256": canonical_json_sha256(state_payload),
        "state_component_inventory_sha256": canonical_json_sha256(inventory_payload),
        "compiled_non_rate_inputs_digest": canonical_json_sha256(inventory_payload["compiled_non_rate_inputs"]),
        "baseline_run_dir": run_base / "baseline",
        "shock_run_dir": run_base / "shock",
    }


def _write_source_run(fixture: dict[str, Any], role: str) -> None:
    run_dir = Path(fixture[f"{role}_run_dir"])
    outputs = run_dir / "outputs"
    inputs = run_dir / "compile" / "compiled" / "forecast_inputs"
    outputs.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)

    _write_summary(outputs / "tdcsim_period_tdc_summary.csv", fixture)
    _write_components(outputs / "tdcsim_period_tdc_components.csv", fixture)
    _write_route(outputs / "tdcsim_tdc_principal_route_stock_closure.csv", fixture)
    primary_path = inputs / "tdcsim_primary_deficit_path.csv"
    primary_path.write_text("period,primary_deficit_bil\nassumption_mode,0\n", encoding="utf-8")
    surface_path = inputs / "tdcsim_yield_curve_surface.csv"
    _write_yield_surface(surface_path, fixture, shock=(role == "shock"))

    scenario_id = fixture[f"{role}_scenario_id"]
    scenario = {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": scenario_id,
        "overrides": {},
    }
    if role == "shock":
        scenario["overrides"] = {
            "nominal_yield_curve": {
                "mode": "full_surface_file",
                "file": {
                    "relative_path": "tdcsim_yield_curve_surface.csv",
                    "sha256": sha256_file(surface_path),
                },
            }
        }
    write_json(run_dir / "scenario.json", scenario)

    manifest = {
        "run_id": f"{scenario_id}-{SOURCE_VINTAGE}",
        "baseline": {
            "package_id": "ratewall_assumption_mode_fixture",
            "package_sha256": fixture["state_fingerprint_sha256"],
            "manifest_sha256": fixture["state_component_inventory_sha256"],
            "release_attestation_sha256": fixture["compiled_non_rate_inputs_digest"],
        },
        "scenario": {"scenario_id": scenario_id, "relative_path": "scenario.json"},
        "compiled_inputs_digest": fixture["compiled_non_rate_inputs_digest"],
        "compiled_inputs": [
            _artifact(run_dir, "tdcsim_yield_curve_surface.csv", surface_path),
            _artifact(run_dir, "tdcsim_primary_deficit_path.csv", primary_path),
        ],
        "simulation": {
            "start_date": fixture["horizon_start"].isoformat(),
            "end_date": fixture["horizon_end"].isoformat(),
            "frequency": "daily",
        },
        "output_manifest": {
            "row_metadata": {
                "actuals_available_as_of": fixture["horizon_start"].isoformat(),
                "source_vintage": SOURCE_VINTAGE,
            }
        },
        "outputs": [
            _artifact(run_dir, "tdcsim_period_tdc_summary.csv", outputs / "tdcsim_period_tdc_summary.csv"),
            _artifact(run_dir, "tdcsim_period_tdc_components.csv", outputs / "tdcsim_period_tdc_components.csv"),
            _artifact(
                run_dir,
                "tdcsim_tdc_principal_route_stock_closure.csv",
                outputs / "tdcsim_tdc_principal_route_stock_closure.csv",
            ),
        ],
    }
    write_json(run_dir / "tdcsim_cbo_run_manifest.json", manifest)


def _write_summary(path: Path, fixture: dict[str, Any]) -> None:
    _write_rows(
        path,
        [
            {
                "period_start": fixture["horizon_start"].isoformat(),
                "period_end": fixture["horizon_end"].isoformat(),
                "tdc_change_bil": "0",
                "tdc_fiscal_flow_bil": "0",
                "tdc_debt_service_bil": "0",
                "tdc_auction_absorption_du_bil": "0",
                "tdc_secondary_trades_bil": "0",
                "tdc_other_bil": "0",
                "overlap_cashflow_bil": "0",
                "tdc_change_ex_overlap_bil": "0",
                "component_sum_bil": "0",
                "component_sum_error_bil": "0",
            }
        ],
    )


def _write_components(path: Path, fixture: dict[str, Any]) -> None:
    _write_rows(
        path,
        [
            {
                "period_start": fixture["horizon_start"].isoformat(),
                "period_end": fixture["horizon_end"].isoformat(),
                "component_key": "assumption_zero_tdc_delta",
                "component_family": "assumption_mode",
                "holder_sector": "Private",
                "holder_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "all",
                "payment_type": "assumption_mode_zero_delta",
                "accounting_basis": "pre_beta_ex_overlap_delta",
                "amount_bil": "0",
                "enters_direct_interest_support": "false",
                "enters_tdc_deposit_support_default": "true",
                "tdc_amount_basis": "post_mmf_route_pass_through_pre_ratewall_beta_chi",
                "overlap_policy": "overlap_removed_before_ratewall_beta_chi",
            }
        ],
    )


def _write_route(path: Path, fixture: dict[str, Any]) -> None:
    stock = CURRENT_OPENING_TDC_STOCK_BIL
    _write_rows(
        path,
        [
            {
                "period_start": fixture["horizon_start"].isoformat(),
                "period_end": fixture["horizon_end"].isoformat(),
                "route_holder_sector": "Private",
                "route_holder_subsector": "domestic_nonbank_deposit_funded",
                "instrument_type": "all",
                "maturity_bucket": "all",
                "debt_scope": "controlled_public_marketable",
                "opening_route_stock_bil": stock,
                "route_face_issued_bil": "0",
                "route_face_redeemed_bil": "0",
                "route_stock_residual_or_indexation_bil": "0",
                "closing_route_stock_bil": stock,
                "closure_identity_error_bil": "0",
                "route_stock_basis": "assumption_mode_state_metadata_not_support",
            }
        ],
    )


def _write_yield_surface(path: Path, fixture: dict[str, Any], *, shock: bool) -> None:
    delta_bp = 100.0 if shock else 0.0
    rate = BASELINE_RATE_DECIMAL + delta_bp / 10000.0
    rows = [
        {
            "schema_version": "tdcsim_yield_curve_surface_v1",
            "scenario_id": fixture["shock_scenario_id"] if shock else fixture["baseline_scenario_id"],
            "curve_date": fixture["horizon_start"].isoformat(),
            "tenor_years": tenor,
            "nominal_rate": rate * 100.0,
            "nominal_rate_decimal": rate,
        }
        for tenor in TENORS
    ]
    _write_rows(path, rows)


def _pair_spec(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "pair_id": fixture["pair_id"],
        "scenario_state_set_id": fixture["scenario_state_set_id"],
        "state_id": fixture["state_id"],
        "state_kind": fixture["state_kind"],
        "state_period": str(fixture["year"]),
        "ratewall_period": str(fixture["year"]),
        "scenario_id": fixture["scenario_id"],
        "state_fingerprint_sha256": fixture["state_fingerprint_sha256"],
        "state_component_inventory_sha256": fixture["state_component_inventory_sha256"],
        "baseline_state_fingerprint_sha256": fixture["state_fingerprint_sha256"],
        "shock_state_fingerprint_sha256": fixture["state_fingerprint_sha256"],
        "opening_state_date": fixture["horizon_start"].isoformat(),
        "actuals_available_as_of": fixture["horizon_start"].isoformat(),
        "source_vintage": SOURCE_VINTAGE,
        "horizon_start_date": fixture["horizon_start"].isoformat(),
        "horizon_end_date": fixture["horizon_end"].isoformat(),
        "horizon": "annual_h1_100bp_year",
        "horizon_index": "h1",
        "compiled_non_rate_inputs_digest": fixture["compiled_non_rate_inputs_digest"],
        "nominal_gdp_bil": CURRENT_NOMINAL_GDP_BIL if fixture["kind"] == "current" else "",
        "opening_tdc_stock_bil": CURRENT_OPENING_TDC_STOCK_BIL,
        "opening_deposit_liquidity_stock_bil": "",
        "opening_route_stock_total_bil": CURRENT_OPENING_TDC_STOCK_BIL,
        "opening_route_stock_domestic_nonbank_bil": CURRENT_OPENING_TDC_STOCK_BIL,
        "opening_route_stock_bank_bil": "0",
        "opening_route_stock_mmf_bil": "0",
        "opening_route_stock_foreign_bil": "0",
        "opening_route_stock_fed_bil": "0",
        "baseline_run_dir": str(fixture["baseline_run_dir"]),
        "shock_run_dir": str(fixture["shock_run_dir"]),
        "baseline_scenario_id": fixture["baseline_scenario_id"],
        "shock_scenario_id": fixture["shock_scenario_id"],
        "object_id": OBJECT_ID,
        "shock_path_id": SHOCK_PATH_ID,
        "shock_bps_year": 100,
        "denominator_equivalence_key": DENOMINATOR_EQUIVALENCE_KEY,
        "require_same_baseline_hashes": True,
        "require_same_opening_state": True,
        "require_same_actuals_available_as_of": True,
        "require_same_simulation_dates": True,
        "require_same_period_index": True,
        "require_same_non_rate_compiled_inputs": True,
        "one_named_rate_shock_only": True,
        "demand_conversion_cases": [
            {
                "demand_conversion_case": "central",
                "beta": BETA,
                "beta_assumption_id": "beta_ratewall_default_20260630",
                "beta_source_status": "assumption_mode",
                "chi": CHI,
                "chi_assumption_id": "chi_ratewall_default_20260630",
                "chi_source_status": "assumption_mode",
            }
        ],
    }


def _artifact(run_dir: Path, logical_name: str, path: Path) -> dict[str, Any]:
    return {
        "logical_name": logical_name,
        "relative_path": path.relative_to(run_dir).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": "text/csv",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
