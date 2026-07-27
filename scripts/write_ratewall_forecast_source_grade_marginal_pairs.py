#!/usr/bin/env python3
"""Build RateWall forecast source-grade marginal TDC pairs from CBO states."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo import (  # noqa: E402
    CboBaselinePackage,
    CboScenarioSpec,
    export_forecast_state_package,
    ForecastStateExport,
    forecast_state_window,
    run_cbo_scenario,
    run_no_shock_rollforward,
)
from tdcsim_cbo._json import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402
from tdcsim_cbo.campaign_store import register_source_run  # noqa: E402
from tdcsim_cbo.marginal_tdc import (  # noqa: E402
    DENOMINATOR_EQUIVALENCE_KEY,
    OBJECT_ID,
    SHOCK_PATH_ID,
    assemble_marginal_tdc_pair,
    verify_marginal_tdc_pair,
)


STATE_SET_ID = "ratewall_forecast_cbo_baseline_source_grade_state_set_v1"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    years = _parse_years(args.years)
    ratewall_root = args.ratewall_project_root.expanduser().resolve()
    beta_schedule_path = (
        args.beta_schedule_path.expanduser().resolve()
        if args.beta_schedule_path
        else ratewall_root
        / "var"
        / "preliminary_scenario_results"
        / "marginal_tdcsim"
        / "ratewall_marginal_tdc_beta_schedule.csv"
    )
    output_root = args.output_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    pair_spec_dir = (
        args.pair_spec_root.expanduser().resolve()
        if args.pair_spec_root
        else output_root / "pair_specs"
    )
    if args.phase in {"assemble", "all"}:
        pair_spec_dir.mkdir(parents=True, exist_ok=True)
    parent = CboBaselinePackage.open(args.baseline.expanduser().resolve(), attestation_path=args.attestation.expanduser().resolve())
    failures: list[str] = []

    for year in years:
        try:
            _process_year(
                year=year,
                args=args,
                parent=parent,
                beta_schedule_path=beta_schedule_path,
                output_root=output_root,
                work_root=work_root,
                pair_spec_dir=pair_spec_dir,
            )
        except Exception as exc:  # keep later years moving; report failures loudly.
            failure = f"{year}: {type(exc).__name__}: {exc}"
            failures.append(failure)
            _append_progress(f"forecast_{year}_failed", work_root / "source_runs" / f"forecast_cbo_baseline_{year}_source_grade", failure=failure)
            print(f"FAILED {failure}", file=sys.stderr)
    if failures:
        print("forecast source-grade failures:")
        for failure in failures:
            print(failure)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--ratewall-project-root", required=True, type=Path)
    parser.add_argument("--years", default="2027-2036")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--beta-schedule-path", default=None, type=Path)
    # Regenerating a retained campaign needs the same beta/chi selection the retained pair
    # specs were built with. beta is claim-relevant -- it multiplies into
    # tdc_materialized_deposit_stock_admissible_bil, which the downstream consumer selects --
    # so re-deriving it from a rebuilt schedule risks silently moving a consumed value.
    # Pointing at the retained specs reuses the exact recorded selection instead.
    parser.add_argument("--beta-case-from-spec-root", default=None, type=Path)
    parser.add_argument("--pair-spec-root", default=None, type=Path)
    parser.add_argument("--phase", choices=("compute", "assemble", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    return parser


def _process_year(
    *,
    year: int,
    args: argparse.Namespace,
    parent: CboBaselinePackage,
    beta_schedule_path: Path,
    output_root: Path,
    work_root: Path,
    pair_spec_dir: Path,
) -> None:
    window = forecast_state_window(year)
    package_dir = work_root / "forecast_state_packages" / f"cbo_baseline_state_{year}"
    rollforward_dir = work_root / "rollforward_runs" / f"cbo_baseline_state_{year}"
    scenario_dir = work_root / "scenarios" / f"cbo_baseline_state_{year}"
    pair_run_root = work_root / "source_runs" / f"forecast_cbo_baseline_{year}_source_grade"
    baseline_run_dir = pair_run_root / "baseline"
    shock_run_dir = pair_run_root / "shock"
    pair_dir = output_root / f"forecast_cbo_baseline_{year}_plus_100bp_year"
    if args.phase in {"compute", "all"}:
        for path in (rollforward_dir, baseline_run_dir, shock_run_dir):
            _prepare_path(path, force=args.force, make_dir=False)
        for path in (package_dir, scenario_dir):
            _prepare_path(path, force=args.force, make_dir=True)
    if args.phase in {"assemble", "all"}:
        _prepare_path(pair_dir, force=args.force, make_dir=False)

    if args.phase in {"compute", "all"}:
        rollforward = run_no_shock_rollforward(
            parent,
            state_window=window,
            output_dir=rollforward_dir,
            scenario_dir=scenario_dir,
            force=args.force,
        )
        _append_progress(f"forecast_{year}_rollforward", rollforward_dir)
        export = export_forecast_state_package(
            parent,
            state_window=window,
            rollforward_run_dir=rollforward,
            output_zip=package_dir / f"cbo_baseline_state_{year}_opening_package.zip",
            output_attestation=package_dir / f"cbo_baseline_state_{year}_opening_attestation.json",
            output_manifest=package_dir / f"cbo_baseline_state_{year}_opening_export_manifest.json",
            force=args.force,
        )
        derived = CboBaselinePackage.open(export.package_zip, attestation_path=export.attestation_path)

        baseline_scenario = _baseline_pair_scenario(derived, year, window)
        baseline_path = scenario_dir / f"forecast_{year}_baseline_source_grade.json"
        write_json(baseline_path, baseline_scenario)
        baseline_spec = CboScenarioSpec.from_file(baseline_path)
        baseline_spec.assert_baseline_matches(derived)
        run_cbo_scenario(derived, baseline_spec, baseline_run_dir, output_profile="summary")
        _append_progress(f"forecast_{year}_baseline", baseline_run_dir)

        shock_surface = scenario_dir / f"forecast_{year}_plus100bp_year_surface.csv"
        _write_plus100bp_surface(
            baseline_run_dir / "compile" / "compiled" / "forecast_inputs" / "tdcsim_yield_curve_surface.csv",
            shock_surface,
            shock_start=date.fromisoformat(window.opening_state_date),
            shock_end=date.fromisoformat(window.horizon_end_date),
        )
        shock_scenario = _shock_pair_scenario(baseline_scenario, shock_surface, year)
        shock_path = scenario_dir / f"forecast_{year}_plus100bp_year_source_grade.json"
        write_json(shock_path, shock_scenario)
        shock_spec = CboScenarioSpec.from_file(shock_path)
        shock_spec.assert_baseline_matches(derived)
        run_cbo_scenario(derived, shock_spec, shock_run_dir, output_profile="summary")
        _append_progress(f"forecast_{year}_shock", shock_run_dir)
    else:
        export = _load_forecast_state_export(year, window, package_dir, rollforward_dir)
        baseline_scenario = read_json(scenario_dir / f"forecast_{year}_baseline_source_grade.json")
        shock_scenario = read_json(scenario_dir / f"forecast_{year}_plus100bp_year_source_grade.json")

    if args.phase == "compute":
        print(f"{year}: computed {baseline_run_dir} {shock_run_dir}")
        return

    spec_root = getattr(args, "beta_case_from_spec_root", None)
    if spec_root is None:
        marker = beta_schedule_path.parent / "BETA_SCHEDULE_READY"
        if not marker.exists():
            raise SystemExit(f"RateWall beta schedule readiness marker absent at assembly: {marker}")
    register_source_run(
        output_root,
        baseline_run_dir,
        baseline_package=export.package_zip,
        attestation=export.attestation_path,
    )
    register_source_run(
        output_root,
        shock_run_dir,
        baseline_package=export.package_zip,
        attestation=export.attestation_path,
    )
    spec = _pair_spec(
        year=year,
        window=window,
        export=export,
        baseline_run_dir=baseline_run_dir,
        shock_run_dir=shock_run_dir,
        baseline_scenario_id=baseline_scenario["scenario_id"],
        shock_scenario_id=shock_scenario["scenario_id"],
        beta_case=(
            _beta_case_from_retained_spec(spec_root, year)
            if spec_root is not None
            else _load_beta_case(
                beta_schedule_path=beta_schedule_path,
                period_object="forecast",
                period=str(year),
                state_id=window.state_id,
                state_kind="forecast_state",
                horizon=window.horizon,
                shock_path_id=SHOCK_PATH_ID,
            )
        ),
    )
    spec_path = pair_spec_dir / f"{spec['pair_id']}.json"
    write_json(spec_path, spec)
    result = assemble_marginal_tdc_pair(
        spec_path,
        pair_dir,
        baseline_package=export.package_zip,
        attestation=export.attestation_path,
    )
    verified = verify_marginal_tdc_pair(
        result.output_dir,
        baseline_package=export.package_zip,
        attestation=export.attestation_path,
    )
    print(f"{year}: {verified['status']} {result.output_dir}")


def _load_forecast_state_export(
    year: int,
    window: Any,
    package_dir: Path,
    rollforward_dir: Path,
) -> ForecastStateExport:
    package_zip = package_dir / f"cbo_baseline_state_{year}_opening_package.zip"
    attestation_path = package_dir / f"cbo_baseline_state_{year}_opening_attestation.json"
    export_manifest_path = package_dir / f"cbo_baseline_state_{year}_opening_export_manifest.json"
    export_manifest = read_json(export_manifest_path)
    verified = CboBaselinePackage.open(package_zip, attestation_path=attestation_path)
    return ForecastStateExport(
        year=year,
        state_id=window.state_id,
        package_zip=package_zip,
        attestation_path=attestation_path,
        export_manifest_path=export_manifest_path,
        rollforward_run_dir=rollforward_dir,
        rollforward_run_manifest_sha256=str(export_manifest["rollforward_run_manifest_sha256"]),
        derived_state_package_sha256=verified.package_sha256,
        forecast_state_export_manifest_sha256=str(export_manifest["forecast_state_export_manifest_sha256"]),
        state_fingerprint_sha256=str(export_manifest["state_fingerprint_sha256"]),
        state_component_inventory_sha256=str(export_manifest["state_component_inventory_sha256"]),
        compiled_non_rate_inputs_digest=str(export_manifest["compiled_non_rate_inputs_digest"]),
    )


def _append_progress(stage: str, run_dir: Path, *, failure: str | None = None) -> None:
    package_id = ""
    manifest_path = run_dir / "tdcsim_cbo_run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            package_id = str((manifest.get("baseline") or {}).get("package_id") or "")
        except Exception:
            package_id = ""
    progress_path = PROJECT_ROOT / "output" / "ratewall_pairs_build_progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        stage,
        datetime.now(timezone.utc).isoformat(),
        str(run_dir),
        package_id,
    ]
    if failure:
        fields.append(failure.replace("\n", " "))
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")


def _parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = value.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _prepare_path(path: Path, *, force: bool, make_dir: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"output path exists; pass --force to replace: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    if make_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)


def _baseline_pair_scenario(baseline: CboBaselinePackage, year: int, window: Any) -> dict[str, Any]:
    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": f"ratewall_forecast_{year}_baseline_source_grade_v1",
        "title": f"RateWall forecast {year} source-grade baseline",
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "provenance": {
            "kind": "user_stress_assumption",
            "label": f"Forecast {year} same-state baseline",
        },
        "coupling": {
            "frn_benchmark": "independent_explicit_path",
            "tips_real_yield": "independent_explicit_path",
            "operating_cash_inflation": "baseline_cpi",
            "primary_deficit_to_debt_target": "independent_no_plug",
        },
        "overrides": {},
        "output": {"profile": "summary", "compression": "gzip", "catalog_sqlite": False},
        "simulation": {
            "frequency": "daily",
            "start_date": window.opening_state_date,
            "end_date": window.horizon_end_date,
        },
    }


def _shock_pair_scenario(baseline_scenario: dict[str, Any], shock_surface: Path, year: int) -> dict[str, Any]:
    scenario = dict(baseline_scenario)
    scenario["scenario_id"] = f"ratewall_forecast_{year}_plus100bp_year_source_grade_v1"
    scenario["title"] = f"RateWall forecast {year} +100bp-year source-grade shock"
    scenario["description"] = "Full-surface +100bp nominal curve shock over one RateWall fiscal-year marginal window."
    scenario["overrides"] = {
        "nominal_yield_curve": {
            "mode": "full_surface_file",
            "file": {
                "relative_path": shock_surface.name,
                "sha256": sha256_file(shock_surface),
                "media_type": "text/csv",
            },
        }
    }
    scenario["coupling"] = dict(scenario["coupling"])
    scenario["coupling"]["frn_benchmark"] = "independent_explicit_path"
    scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
    return scenario


def _write_plus100bp_surface(
    baseline_surface_path: Path,
    output_path: Path,
    *,
    shock_start: date,
    shock_end: date,
) -> None:
    with baseline_surface_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("baseline yield curve surface is empty")
    fields = list(rows[0])
    keyed = {(row["curve_date"], row["tenor_years"]): dict(row) for row in rows}
    tenors = sorted({row["tenor_years"] for row in rows}, key=float)
    for boundary in (shock_start, shock_end):
        for tenor in tenors:
            keyed[(boundary.isoformat(), tenor)] = _curve_row_at(rows, boundary, tenor)
    output_rows = sorted(keyed.values(), key=lambda row: (row["curve_date"], float(row["tenor_years"])))
    for row in output_rows:
        curve_date = date.fromisoformat(row["curve_date"])
        if shock_start <= curve_date < shock_end:
            row["nominal_rate_decimal"] = _fmt_float(float(row["nominal_rate_decimal"]) + 0.01)
            for column in ("nominal_rate", "anchor_3m_pct", "anchor_10y_pct"):
                if row.get(column):
                    row[column] = _fmt_float(float(row[column]) + 1.0)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


def _curve_row_at(rows: list[dict[str, str]], target: date, tenor: str) -> dict[str, str]:
    candidates = [row for row in rows if row["tenor_years"] == tenor and date.fromisoformat(row["curve_date"]) <= target]
    if not candidates:
        candidates = [row for row in rows if row["tenor_years"] == tenor]
    candidates.sort(key=lambda row: row["curve_date"])
    out = dict(candidates[-1])
    out["curve_date"] = target.isoformat()
    return out


def _pair_spec(
    *,
    year: int,
    window: Any,
    export: Any,
    baseline_run_dir: Path,
    shock_run_dir: Path,
    baseline_scenario_id: str,
    shock_scenario_id: str,
    beta_case: dict[str, Any],
) -> dict[str, Any]:
    baseline_manifest = read_json(baseline_run_dir / "tdcsim_cbo_run_manifest.json")
    shock_manifest = read_json(shock_run_dir / "tdcsim_cbo_run_manifest.json")
    route = pd.read_csv(baseline_run_dir / "outputs" / "tdcsim_tdc_principal_route_stock_closure.csv.gz")
    stock_total = float(route["opening_route_stock_bil"].sum())
    by_holder = route.groupby("route_holder_sector")["opening_route_stock_bil"].sum().to_dict()
    non_rate_digest = _non_rate_digest(baseline_manifest)
    state_payload = {
        "schema": "ratewall_state_fingerprint_v2",
        "source_mode": "source_grade_cbo_baseline_rollforward_export",
        "scenario_state_set_id": STATE_SET_ID,
        "state_id": window.state_id,
        "state_kind": "forecast_state",
        "state_period": str(year),
        "scenario_id": "cbo_baseline_rollforward_opening_state_v1",
        "opening_state_date": window.opening_state_date,
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "derived_state_package_sha256": export.derived_state_package_sha256,
        "forecast_state_export_manifest_sha256": export.forecast_state_export_manifest_sha256,
        "rollforward_run_manifest_sha256": export.rollforward_run_manifest_sha256,
        "compiled_non_rate_inputs_digest": non_rate_digest,
    }
    inventory_payload = {
        "schema": "ratewall_state_component_inventory_v2",
        "state_id": window.state_id,
        "compiled_non_rate_inputs_digest": non_rate_digest,
        "opening_route_stock_total_bil": stock_total,
        "holders": {key: float(value) for key, value in by_holder.items()},
    }
    state_sha = canonical_json_sha256(state_payload)
    inventory_sha = canonical_json_sha256(inventory_payload)
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "pair_id": f"ratewall_forecast_cbo_baseline_{year}_plus100bp_year_source_grade_pair_v1",
        "scenario_state_set_id": STATE_SET_ID,
        "state_id": window.state_id,
        "state_kind": "forecast_state",
        "state_period": str(year),
        "ratewall_period": str(year),
        "scenario_id": "cbo_baseline_rollforward_opening_state_v1",
        "state_fingerprint_sha256": state_sha,
        "state_component_inventory_sha256": inventory_sha,
        "baseline_state_fingerprint_sha256": state_sha,
        "shock_state_fingerprint_sha256": state_sha,
        "opening_state_date": window.opening_state_date,
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "source_grade_status": "pass_forecast_rollforward_source_grade",
        "state_construction_method": "baseline_rollforward_export_v1",
        "forecast_state_export_manifest_sha256": export.forecast_state_export_manifest_sha256,
        "derived_state_package_sha256": export.derived_state_package_sha256,
        "parent_baseline_package_sha256": read_json(export.export_manifest_path)["parent_baseline_package_sha256"],
        "parent_baseline_manifest_sha256": read_json(export.export_manifest_path)["parent_baseline_manifest_sha256"],
        "parent_attestation_sha256": read_json(export.export_manifest_path)["parent_attestation_sha256"],
        "rollforward_run_manifest_sha256": export.rollforward_run_manifest_sha256,
        "horizon_start_date": window.opening_state_date,
        "horizon_end_date": window.horizon_end_date,
        "horizon": window.horizon,
        "horizon_index": "h1",
        "compiled_non_rate_inputs_digest": non_rate_digest,
        "opening_portfolio_sha256": _compiled_sha(baseline_manifest, "tdcsim_opening_portfolio.csv"),
        "debt_stock_path_sha256": _compiled_sha(baseline_manifest, "tdcsim_debt_stock_path.csv"),
        "primary_deficit_path_sha256": _compiled_sha(baseline_manifest, "tdcsim_primary_deficit_path.csv"),
        "operating_cash_path_sha256": _compiled_sha(baseline_manifest, "tdcsim_operating_cash_path.csv"),
        "fed_holdings_path_sha256": _compiled_sha(baseline_manifest, "tdcsim_fed_holdings_path.csv"),
        "holder_route_assumptions_sha256": _compiled_sha(baseline_manifest, "tdcsim_holder_profile_assumptions.csv"),
        "issuance_mix_sha256": _compiled_sha(baseline_manifest, "tdcsim_issuance_mix_assumptions.json"),
        "mmf_split_assumptions_sha256": _compiled_sha(baseline_manifest, "tdcsim_runtime_assumptions.json"),
        "nominal_gdp_bil": "",
        "opening_tdc_stock_bil": stock_total,
        "opening_deposit_liquidity_stock_bil": "",
        "opening_route_stock_total_bil": stock_total,
        "opening_route_stock_domestic_nonbank_bil": float(by_holder.get("Private", 0)),
        "opening_route_stock_bank_bil": float(by_holder.get("Banks", 0)),
        "opening_route_stock_mmf_bil": "",
        "opening_route_stock_foreign_bil": float(by_holder.get("Foreign", 0)),
        "opening_route_stock_fed_bil": float(by_holder.get("CB", 0)),
        "baseline_run_id": baseline_manifest["run_id"],
        "shock_run_id": shock_manifest["run_id"],
        "baseline_scenario_id": baseline_scenario_id,
        "shock_scenario_id": shock_scenario_id,
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
        "demand_conversion_cases": [beta_case],
    }


BETA_CASE_FIELDS = (
    "demand_conversion_case",
    "beta",
    "beta_assumption_id",
    "beta_source_status",
    "chi",
    "chi_assumption_id",
    "chi_source_status",
)


def _beta_case_from_retained_spec(spec_root: Path, year: int) -> dict[str, Any]:
    """Reuse the beta/chi selection a retained pair spec already recorded.

    Regenerating a retained campaign must reproduce beta exactly. It is not confined to the
    retired diagnostic: ``delta_tdc_ex_overlap_non_interest_admissible_bil * beta`` is
    asserted equal to ``tdc_materialized_deposit_stock_admissible_bil``, which the downstream
    consumer selects. Re-deriving beta from a rebuilt schedule could move a consumed value
    without anything failing, so the recorded selection is the authority.
    """

    root = Path(spec_root).expanduser().resolve()
    matches = sorted(root.glob(f"*forecast_cbo_baseline_{year}_*.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"expected exactly one retained forecast pair spec for {year} under {root}, found {len(matches)}"
        )
    spec = read_json(matches[0])
    if not isinstance(spec, Mapping):
        raise SystemExit(f"retained pair spec must be an object: {matches[0]}")
    cases = [
        case
        for case in spec.get("demand_conversion_cases", [])
        if isinstance(case, Mapping) and case.get("demand_conversion_case") == "central"
    ]
    if len(cases) != 1:
        raise SystemExit(f"retained pair spec must carry exactly one central case: {matches[0]}")
    case = cases[0]
    missing = [field for field in BETA_CASE_FIELDS if field not in case]
    if missing:
        raise SystemExit(f"retained beta case is missing required fields {missing}: {matches[0]}")
    return {field: case[field] for field in BETA_CASE_FIELDS}


def _load_beta_case(
    *,
    beta_schedule_path: Path,
    period_object: str,
    period: str,
    state_id: str,
    state_kind: str,
    horizon: str,
    shock_path_id: str,
) -> dict[str, Any]:
    with beta_schedule_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row
        for row in rows
        if row["period_object"] == period_object
        and row["period"] == period
        and row["state_id"] == state_id
        and row["state_kind"] == state_kind
        and row["horizon"] == horizon
        and row["shock_path_id"] == shock_path_id
        and row["demand_conversion_case"] == "central"
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one RateWall beta schedule row for {period_object} {period} {state_id}")
    row = matches[0]
    return {
        "demand_conversion_case": "central",
        "beta": float(row["beta_selected"]),
        "beta_assumption_id": row["beta_assumption_id"],
        "beta_source_status": row["beta_source_status"],
        "chi": float(row["chi_selected"]),
        "chi_assumption_id": row["chi_assumption_id"],
        "chi_source_status": row["chi_source_status"],
    }


def _compiled_sha(manifest: dict[str, Any], filename: str) -> str:
    return next((item["sha256"] for item in manifest["compiled_inputs"] if item["logical_name"].endswith(filename)), "")


def _non_rate_digest(manifest: dict[str, Any]) -> str:
    records = []
    for item in manifest.get("compiled_inputs", []):
        name = Path(str(item.get("logical_name") or item.get("relative_path") or "")).name
        if name in {"tdcsim_yield_curve_surface.csv", "tdcsim_frn_rate_path.csv", "tdcsim_tips_real_yield_path.csv"}:
            continue
        records.append({"logical_name": str(item.get("logical_name") or ""), "bytes": int(item.get("bytes", 0) or 0), "sha256": str(item.get("sha256") or "")})
    return canonical_json_sha256(sorted(records, key=lambda row: row["logical_name"]))


def _fmt_float(value: float) -> str:
    return format(value, ".15g")


if __name__ == "__main__":
    raise SystemExit(main())
