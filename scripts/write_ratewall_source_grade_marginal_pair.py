#!/usr/bin/env python3
"""Build a source-grade RateWall marginal TDC pair from CBO run outputs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, run_cbo_scenario  # noqa: E402
from tdcsim_cbo._json import canonical_json_sha256, sha256_file, write_json  # noqa: E402
from tdcsim_cbo.campaign_store import register_source_run  # noqa: E402
from tdcsim_cbo.marginal_tdc import (  # noqa: E402
    beta_case_from_retained_spec,
    DENOMINATOR_EQUIVALENCE_KEY,
    OBJECT_ID,
    SHOCK_PATH_ID,
    assemble_marginal_tdc_pair,
    verify_marginal_tdc_pair,
)
from write_cbo_example_scenarios import example_scenarios  # noqa: E402


NOMINAL_GDP_2026_BIL = "31902.006"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    baseline_package = CboBaselinePackage.open(
        args.baseline.expanduser().resolve(),
        attestation_path=args.attestation.expanduser().resolve(),
    )

    pair_dir = output_root / "current_state_2026_plus_100bp_year_source_grade"
    baseline_run_dir = work_root / "source_runs" / "current_state_2026_source_grade" / "baseline"
    shock_run_dir = work_root / "source_runs" / "current_state_2026_source_grade" / "shock"
    scenario_dir = work_root / "source_grade_scenarios" / "current_state_2026"
    if args.phase in {"compute", "all"}:
        for path in (baseline_run_dir, shock_run_dir):
            _prepare_output_path(path, force=args.force)
        _prepare_dir(scenario_dir, force=args.force)
    if args.phase in {"assemble", "all"}:
        _prepare_output_path(pair_dir, force=args.force)
        pair_spec_dir.mkdir(parents=True, exist_ok=True)

    simulation = {
        "frequency": "daily",
        "start_date": "2026-06-21",
        "end_date": "2027-06-21",
    }
    baseline_scenario_path = scenario_dir / "current_2026_baseline_noop.json"
    shock_scenario_path = scenario_dir / "current_2026_plus100bp_year.json"
    if args.phase in {"compute", "all"}:
        baseline_scenario = example_scenarios(baseline_package, simulation=simulation)[
            "00_baseline_noop.json"
        ]
        write_json(baseline_scenario_path, baseline_scenario)
        baseline_spec = CboScenarioSpec.from_file(baseline_scenario_path)
        baseline_spec.assert_baseline_matches(baseline_package)
        run_cbo_scenario(
            baseline_package,
            baseline_spec,
            baseline_run_dir,
            output_profile="summary",
        )
        _append_progress("current_2026_baseline", baseline_run_dir)

        shock_surface_path = scenario_dir / "current_2026_plus100bp_year_surface.csv"
        _write_plus100bp_surface(
            baseline_run_dir / "compile" / "compiled" / "forecast_inputs" / "tdcsim_yield_curve_surface.csv",
            shock_surface_path,
            shock_start=date(2026, 6, 21),
            shock_end=date(2027, 6, 21),
        )
        shock_scenario = dict(baseline_scenario)
        shock_scenario["scenario_id"] = "ratewall_current_2026_plus100bp_year_source_grade_v1"
        shock_scenario["title"] = "RateWall current 2026 +100bp-year source-grade shock"
        shock_scenario["description"] = (
            "Full-surface +100bp nominal curve shock over the current RateWall one-year "
            "marginal window, with baseline rates restored at the shock boundary."
        )
        shock_scenario["overrides"] = {
            "nominal_yield_curve": {
                "mode": "full_surface_file",
                "file": {
                    "relative_path": shock_surface_path.name,
                    "sha256": sha256_file(shock_surface_path),
                    "media_type": "text/csv",
                },
            },
        }
        shock_scenario["coupling"] = dict(shock_scenario["coupling"])
        shock_scenario["coupling"]["frn_benchmark"] = "independent_explicit_path"
        shock_scenario["coupling"]["tips_real_yield"] = "independent_explicit_path"
        write_json(shock_scenario_path, shock_scenario)
        shock_spec = CboScenarioSpec.from_file(shock_scenario_path)
        shock_spec.assert_baseline_matches(baseline_package)
        run_cbo_scenario(
            baseline_package,
            shock_spec,
            shock_run_dir,
            output_profile="summary",
        )
        _append_progress("current_2026_shock", shock_run_dir)

    if args.phase == "compute":
        print(f"computed current 2026 runs: {baseline_run_dir} {shock_run_dir}")
        return 0

    baseline_scenario = _read_json(baseline_scenario_path)
    shock_scenario = _read_json(shock_scenario_path)
    spec_root = getattr(args, "beta_case_from_spec_root", None)
    if spec_root is None:
        marker = beta_schedule_path.parent / "BETA_SCHEDULE_READY"
        if not marker.exists():
            raise SystemExit(f"RateWall beta schedule readiness marker absent at assembly: {marker}")
    register_source_run(
        output_root,
        baseline_run_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    register_source_run(
        output_root,
        shock_run_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    pair_spec = _pair_spec(
        baseline_run_dir=baseline_run_dir,
        shock_run_dir=shock_run_dir,
        baseline_scenario_id=baseline_scenario["scenario_id"],
        shock_scenario_id=shock_scenario["scenario_id"],
        beta_case=(
            beta_case_from_retained_spec(spec_root, "*current_2026*.json")
            if spec_root is not None
            else _load_beta_case(
                beta_schedule_path=beta_schedule_path,
                period_object="current",
                period="2026",
                state_id="current_state::2026",
                state_kind="current_state",
                horizon="annual_h1_100bp_year",
                shock_path_id=SHOCK_PATH_ID,
            )
        ),
    )
    pair_spec_path = pair_spec_dir / f"{pair_spec['pair_id']}.json"
    write_json(pair_spec_path, pair_spec)
    result = assemble_marginal_tdc_pair(
        pair_spec_path,
        pair_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    verified = verify_marginal_tdc_pair(
        result.output_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    print(result.output_dir)
    print(verified["status"])
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratewall-project-root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--beta-schedule-path", default=None, type=Path)
    # Reuse the retained specs' recorded beta/chi rather than a rebuilt schedule; beta is
    # claim-relevant and must stay byte-exact. See marginal_tdc.beta_case_from_retained_spec.
    parser.add_argument("--beta-case-from-spec-root", default=None, type=Path)
    parser.add_argument("--pair-spec-root", default=None, type=Path)
    parser.add_argument("--phase", choices=("compute", "assemble", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    return parser


def _append_progress(stage: str, run_dir: Path, *, failure: str | None = None) -> None:
    package_id = ""
    manifest_path = run_dir / "tdcsim_cbo_run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _read_json(manifest_path)
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


def _prepare_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"output path exists; pass --force to replace: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _prepare_output_path(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"output path exists; pass --force to replace: {path}")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


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
    output_rows = sorted(
        keyed.values(),
        key=lambda row: (row["curve_date"], float(row["tenor_years"])),
    )
    for row in output_rows:
        curve_date = date.fromisoformat(row["curve_date"])
        if shock_start <= curve_date < shock_end:
            row["nominal_rate_decimal"] = _fmt_float(
                float(row["nominal_rate_decimal"]) + 0.01
            )
            for column in ("nominal_rate", "anchor_3m_pct", "anchor_10y_pct"):
                if row.get(column):
                    row[column] = _fmt_float(float(row[column]) + 1.0)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


def _curve_row_at(rows: list[dict[str, str]], target: date, tenor: str) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if row["tenor_years"] == tenor and date.fromisoformat(row["curve_date"]) <= target
    ]
    if not candidates:
        candidates = [row for row in rows if row["tenor_years"] == tenor]
    candidates.sort(key=lambda row: row["curve_date"])
    out = dict(candidates[-1])
    out["curve_date"] = target.isoformat()
    return out


def _pair_spec(
    *,
    baseline_run_dir: Path,
    shock_run_dir: Path,
    baseline_scenario_id: str,
    shock_scenario_id: str,
    beta_case: dict[str, Any],
) -> dict[str, Any]:
    baseline_manifest = _read_json(baseline_run_dir / "tdcsim_cbo_run_manifest.json")
    shock_manifest = _read_json(shock_run_dir / "tdcsim_cbo_run_manifest.json")
    route = pd.read_csv(
        baseline_run_dir / "outputs" / "tdcsim_tdc_principal_route_stock_closure.csv.gz"
    )
    stock_total = float(route["opening_route_stock_bil"].sum())
    by_holder = route.groupby("route_holder_sector")["opening_route_stock_bil"].sum().to_dict()
    non_rate_digest = _non_rate_digest(baseline_manifest)
    state_payload = {
        "schema": "ratewall_state_fingerprint_v1",
        "source_mode": "source_grade_cbo_release_bound_run_pair",
        "scenario_state_set_id": "ratewall_current_opening_state_source_grade_v1",
        "state_id": "current_state::2026",
        "state_kind": "current_state",
        "state_period": "2026",
        "scenario_id": "cbo_release_bound_opening_state_v1",
        "opening_state_date": "2026-06-21",
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "baseline_run_id": baseline_manifest["run_id"],
        "shock_run_id": shock_manifest["run_id"],
    }
    inventory_payload = {
        "schema": "ratewall_state_component_inventory_v1",
        "compiled_non_rate_inputs_digest": non_rate_digest,
        "opening_route_stock_total_bil": stock_total,
        "holders": {key: float(value) for key, value in by_holder.items()},
    }
    state_sha = canonical_json_sha256(state_payload)
    inventory_sha = canonical_json_sha256(inventory_payload)
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "pair_id": "ratewall_current_2026_plus100bp_year_source_grade_pair_v1",
        "scenario_state_set_id": "ratewall_current_opening_state_source_grade_v1",
        "state_id": "current_state::2026",
        "state_kind": "current_state",
        "state_period": "2026",
        "ratewall_period": "2026",
        "scenario_id": "cbo_release_bound_opening_state_v1",
        "state_fingerprint_sha256": state_sha,
        "state_component_inventory_sha256": inventory_sha,
        "baseline_state_fingerprint_sha256": state_sha,
        "shock_state_fingerprint_sha256": state_sha,
        "opening_state_date": "2026-06-21",
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "horizon_start_date": "2026-06-21",
        "horizon_end_date": "2027-06-21",
        "horizon": "annual_h1_100bp_year",
        "horizon_index": "h1",
        "compiled_non_rate_inputs_digest": non_rate_digest,
        "opening_portfolio_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_opening_portfolio.csv"
        ),
        "debt_stock_path_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_debt_stock_path.csv"
        ),
        "primary_deficit_path_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_primary_deficit_path.csv"
        ),
        "operating_cash_path_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_operating_cash_path.csv"
        ),
        "fed_holdings_path_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_fed_holdings_path.csv"
        ),
        "holder_route_assumptions_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_holder_profile_assumptions.csv"
        ),
        "issuance_mix_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_issuance_mix_assumptions.json"
        ),
        "mmf_split_assumptions_sha256": _compiled_sha(
            baseline_manifest, "tdcsim_runtime_assumptions.json"
        ),
        "nominal_gdp_bil": NOMINAL_GDP_2026_BIL,
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
    return next(
        (
            item["sha256"]
            for item in manifest["compiled_inputs"]
            if item["logical_name"].endswith(filename)
        ),
        "",
    )


def _non_rate_digest(manifest: dict[str, Any]) -> str:
    records = []
    for item in manifest.get("compiled_inputs", []):
        name = Path(str(item.get("logical_name") or item.get("relative_path") or "")).name
        if name in {
            "tdcsim_yield_curve_surface.csv",
            "tdcsim_frn_rate_path.csv",
            "tdcsim_tips_real_yield_path.csv",
        }:
            continue
        records.append(
            {
                "logical_name": str(item.get("logical_name") or ""),
                "bytes": int(item.get("bytes", 0) or 0),
                "sha256": str(item.get("sha256") or ""),
            }
        )
    return canonical_json_sha256(sorted(records, key=lambda row: row["logical_name"]))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _fmt_float(value: float) -> str:
    return format(value, ".15g")


if __name__ == "__main__":
    raise SystemExit(main())
