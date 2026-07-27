#!/usr/bin/env python3
"""Build the COVID-like flooded-state TDC pair artifacts."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from datetime import date
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
    ForecastStateExport,
    ForecastStateWindow,
    export_forecast_state_package,
    run_cbo_scenario,
    run_no_shock_rollforward,
)
from tdcsim_cbo._json import canonical_json_sha256, read_json, sha256_file, write_json  # noqa: E402
from tdcsim_cbo.campaign_store import register_source_run  # noqa: E402
from tdcsim_cbo.marginal_tdc import (  # noqa: E402
    DENOMINATOR_EQUIVALENCE_KEY,
    FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY,
    FISCAL_INJECTION_OBJECT_ID,
    FISCAL_INJECTION_SHOCK_PATH_ID,
    OBJECT_ID,
    SHOCK_PATH_ID,
    assemble_marginal_tdc_pair,
    verify_marginal_tdc_pair,
)


STATE_SET_ID = "flooded_2028_v1"
INJECTION_TOTAL_BIL = 3000.0
INJECTION_START = date(2028, 1, 1)
INJECTION_END = date(2029, 1, 1)
FED_ABSORPTION_SHARE = 0.45


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    pair_spec_dir = output_root / "pair_specs"
    if args.force:
        for path in (output_root, work_root):
            if path.exists():
                shutil.rmtree(path)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    pair_spec_dir.mkdir(parents=True, exist_ok=True)

    # No beta-schedule gate here. These pairs are emitted pre-beta on purpose:
    # `_pre_beta_case` sets beta=chi=1.0 with source status
    # `not_applied_in_tdcsim_pair_artifact`, deferring the demand conversion to the consumer
    # side. The schedule's contents were never read -- only its existence was checked -- so the
    # gate blocked the build on a file whose values it would not have used. The schedule's
    # producer has since been retired downstream, which turned a dead gate into a hard stop.
    source_pair_root = args.source_grade_pair_root.expanduser().resolve()
    _verify_prereq_pairs(source_pair_root)

    parent = CboBaselinePackage.open(args.baseline.expanduser().resolve(), attestation_path=args.attestation.expanduser().resolve())
    rows: list[dict[str, Any]] = []

    pre_window = _window(2028, state_id="flooded_state::2028")
    pre_export = _export_baseline_state(parent, pre_window, work_root, force=args.force)
    pre_export_run = _export_from_rollforward(
        parent_baseline=parent,
        state_window=ForecastStateWindow(
            state_period=2028,
            state_id="flooded_state::2028_run_coverage_bridge",
            opening_state_date="2028-01-01",
            horizon_end_date="2032-01-01",
            horizon="bridge_coverage_for_flooded_state_exports",
        ),
        rollforward_run_dir=pre_export.rollforward_run_dir,
        work_root=work_root,
        slug="flooded_state_2028_run_coverage_bridge",
        force=args.force,
    )
    pre_package = CboBaselinePackage.open(pre_export_run.package_zip, attestation_path=pre_export_run.attestation_path)
    rows.append({"stage": "export_flooded_state_2028", "path": str(pre_export.package_zip), "status": "pass"})
    rows.append({"stage": "export_flooded_state_2028_run_bridge", "path": str(pre_export_run.package_zip), "status": "pass"})

    injection_paths = _write_injection_files(pre_package, work_root / "scenarios" / "fiscal_injection_2028_v1")
    baseline_run, injection_run = _run_injection_pair_sources(
        pre_package,
        injection_paths,
        work_root,
        force=args.force,
    )
    register_source_run(
        output_root,
        baseline_run,
        baseline_package=pre_package.package_path,
        attestation=pre_package.attestation.path,
    )
    register_source_run(
        output_root,
        injection_run,
        baseline_package=pre_package.package_path,
        attestation=pre_package.attestation.path,
    )
    injection_pair = _assemble_pair(
        spec=_pair_spec(
            pair_id="fiscal_injection_2028_v1_pair",
            export=pre_export,
            baseline_run_dir=baseline_run,
            shock_run_dir=injection_run,
            baseline_scenario_id="flooded_2028_baseline_no_injection_v1",
            shock_scenario_id="fiscal_injection_2028_v1",
            object_id=FISCAL_INJECTION_OBJECT_ID,
            shock_path_id=FISCAL_INJECTION_SHOCK_PATH_ID,
            shock_bps_year=0,
            denominator_key=FISCAL_INJECTION_DENOMINATOR_EQUIVALENCE_KEY,
            ratewall_period="2028",
            beta_case=_pre_beta_case("fiscal_injection_pre_beta"),
            require_same_non_rate=False,
            one_named_rate_shock_only=False,
        ),
        pair_dir=output_root / "fiscal_injection_2028_v1_pair",
        spec_dir=pair_spec_dir,
        baseline_package=pre_package,
    )
    rows.append({"stage": "assemble_fiscal_injection_pair", "path": str(injection_pair), "status": "pass"})

    flooded_2029 = _export_from_rollforward(
        parent_baseline=pre_package,
        state_window=_window(2029, state_id="flooded_state::2029"),
        rollforward_run_dir=injection_run,
        work_root=work_root,
        slug="flooded_state_2029",
        force=args.force,
    )
    flooded_2029_package = CboBaselinePackage.open(flooded_2029.package_zip, attestation_path=flooded_2029.attestation_path)
    rows.append({"stage": "export_flooded_state_2029", "path": str(flooded_2029.package_zip), "status": "pass"})

    flooded_2029_bridge = _export_from_rollforward(
        parent_baseline=pre_package,
        state_window=ForecastStateWindow(
            state_period=2029,
            state_id="flooded_state::2029_rollforward_bridge",
            opening_state_date="2029-01-01",
            horizon_end_date="2032-01-01",
            horizon="bridge_to_flooded_state_2031",
        ),
        rollforward_run_dir=injection_run,
        work_root=work_root,
        slug="flooded_state_2029_rollforward_bridge",
        force=args.force,
    )
    flooded_2029_bridge_package = CboBaselinePackage.open(
        flooded_2029_bridge.package_zip,
        attestation_path=flooded_2029_bridge.attestation_path,
    )
    rows.append({"stage": "export_flooded_state_2029_bridge", "path": str(flooded_2029_bridge.package_zip), "status": "pass"})

    state_2031_window = _window(2031, state_id="flooded_state::2031")
    roll_2031 = _run_rolloff_bridge(
        flooded_2029_bridge_package,
        state_window=state_2031_window,
        output_dir=work_root / "rollforward_runs" / "flooded_2029_to_2031",
        scenario_dir=work_root / "scenarios" / "flooded_2029_to_2031",
        force=args.force,
    )
    flooded_2031 = _export_from_rollforward(
        parent_baseline=flooded_2029_bridge_package,
        state_window=state_2031_window,
        rollforward_run_dir=roll_2031,
        work_root=work_root,
        slug="flooded_state_2031",
        force=args.force,
    )
    rows.append({"stage": "export_flooded_state_2031", "path": str(flooded_2031.package_zip), "status": "pass"})

    exports = {2028: pre_export, 2029: flooded_2029, 2031: flooded_2031}
    packages = {
        2028: pre_package,
        2029: flooded_2029_package,
        2031: CboBaselinePackage.open(flooded_2031.package_zip, attestation_path=flooded_2031.attestation_path),
    }
    for year in _parse_years(args.years):
        baseline_run_dir, shock_run_dir = _run_plus100_pair_sources(
            year,
            packages[year],
            exports[year],
            work_root,
            force=args.force,
            injection_paths=injection_paths if year == 2028 else None,
        )
        register_source_run(
            output_root,
            baseline_run_dir,
            baseline_package=packages[year].package_path,
            attestation=packages[year].attestation.path,
        )
        register_source_run(
            output_root,
            shock_run_dir,
            baseline_package=packages[year].package_path,
            attestation=packages[year].attestation.path,
        )
        pair_dir = _assemble_pair(
            spec=_pair_spec(
                pair_id=f"flooded_state_{year}_plus100bp_year_pair",
                export=exports[year],
                baseline_run_dir=baseline_run_dir,
                shock_run_dir=shock_run_dir,
                baseline_scenario_id=f"flooded_state_{year}_baseline_v1",
                shock_scenario_id=f"flooded_state_{year}_plus100bp_year_v1",
                object_id=OBJECT_ID,
                shock_path_id=SHOCK_PATH_ID,
                shock_bps_year=100,
                denominator_key=DENOMINATOR_EQUIVALENCE_KEY,
                ratewall_period=str(year),
                beta_case=_pre_beta_case("flooded_state_rw_pre_beta"),
                require_same_non_rate=True,
                one_named_rate_shock_only=True,
            ),
            pair_dir=output_root / f"flooded_state_{year}_plus_100bp_year",
            spec_dir=pair_spec_dir,
            baseline_package=packages[year],
        )
        rows.append({"stage": f"assemble_flooded_state_{year}_plus100", "path": str(pair_dir), "status": "pass"})

    progress = output_root / "flooded_state_build_progress.csv"
    pd.DataFrame(rows).to_csv(progress, index=False)
    print(f"wrote {progress}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--years", default="2028,2029,2031")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument(
        "--source-grade-pair-root",
        default=PROJECT_ROOT / "output" / "ratewall_source_grade_marginal_pairs_20260706",
        type=Path,
    )
    parser.add_argument("--force", action="store_true")
    return parser


def _verify_prereq_pairs(root: Path) -> None:
    expected = [root / "current_state_2026_plus_100bp_year_source_grade"] + [
        root / f"forecast_cbo_baseline_{year}_plus_100bp_year" for year in range(2027, 2037)
    ]
    missing = [str(path) for path in expected if not (path / "tdcsim_ratewall_marginal_tdc_pair_manifest.json").exists()]
    if missing:
        raise SystemExit(f"source-grade pair prereq missing manifests: {missing}")
    for path in expected:
        verify_marginal_tdc_pair(path)


def _window(year: int, *, state_id: str) -> ForecastStateWindow:
    opening = date(year, 1, 1)
    horizon_end = date(year + 1, 1, 1)
    return ForecastStateWindow(
        state_period=year,
        state_id=state_id,
        opening_state_date=opening.isoformat(),
        horizon_end_date=horizon_end.isoformat(),
        horizon="annual_h1_100bp_year",
    )


def _export_baseline_state(
    parent: CboBaselinePackage,
    window: ForecastStateWindow,
    work_root: Path,
    *,
    force: bool,
) -> ForecastStateExport:
    rollforward = run_no_shock_rollforward(
        parent,
        state_window=window,
        output_dir=work_root / "rollforward_runs" / "baseline_to_flooded_2028",
        scenario_dir=work_root / "scenarios" / "baseline_to_flooded_2028",
        force=force,
    )
    return _export_from_rollforward(
        parent_baseline=parent,
        state_window=window,
        rollforward_run_dir=rollforward,
        work_root=work_root,
        slug="flooded_state_2028",
        force=force,
    )


def _export_from_rollforward(
    *,
    parent_baseline: CboBaselinePackage,
    state_window: ForecastStateWindow,
    rollforward_run_dir: Path,
    work_root: Path,
    slug: str,
    force: bool,
) -> ForecastStateExport:
    package_dir = work_root / "state_packages" / slug
    package_dir.mkdir(parents=True, exist_ok=True)
    return export_forecast_state_package(
        parent_baseline,
        state_window=state_window,
        rollforward_run_dir=rollforward_run_dir,
        output_zip=package_dir / f"{slug}_opening_package.zip",
        output_attestation=package_dir / f"{slug}_opening_attestation.json",
        output_manifest=package_dir / f"{slug}_opening_export_manifest.json",
        force=force,
    )


def _write_injection_files(baseline: CboBaselinePackage, scenario_dir: Path) -> dict[str, Path]:
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tdcsim-flooded-inputs-") as tmp_name:
        inputs = baseline.materialize(Path(tmp_name) / "package") / "forecast_inputs"
        primary = pd.read_csv(inputs / "tdcsim_primary_deficit_path.csv")
        debt = pd.read_csv(inputs / "tdcsim_debt_stock_path.csv")
        fed = pd.read_csv(inputs / "tdcsim_fed_holdings_path.csv")

    injections = _daily_injection_by_period(primary)
    primary_out = _primary_with_injection(primary, injections)
    debt_out = _debt_with_injection(debt, injections)
    fed_out = _fed_with_injection(fed, injections)
    paths = {
        "primary_deficit": scenario_dir / "fiscal_injection_primary_deficit_path.csv",
        "debt_target": scenario_dir / "fiscal_injection_debt_target_path.csv",
        "fed_holdings": scenario_dir / "fiscal_injection_fed_holdings_path.csv",
    }
    primary_out.to_csv(paths["primary_deficit"], index=False)
    debt_out.to_csv(paths["debt_target"], index=False)
    fed_out.to_csv(paths["fed_holdings"], index=False)
    return paths


def _daily_injection_by_period(primary: pd.DataFrame) -> pd.Series:
    starts = pd.to_datetime(primary["period_start"], errors="coerce")
    ends = pd.to_datetime(primary["period_end"], errors="coerce")
    out = pd.Series(0.0, index=ends)
    for q_start, q_end in (
        (date(2028, 1, 1), date(2028, 4, 1)),
        (date(2028, 4, 1), date(2028, 7, 1)),
        (date(2028, 7, 1), date(2028, 10, 1)),
        (date(2028, 10, 1), date(2029, 1, 1)),
    ):
        mask = (starts >= pd.Timestamp(q_start)) & (ends <= pd.Timestamp(q_end))
        days = int(mask.sum())
        if days <= 0:
            raise SystemExit(f"no primary-deficit rows found for injection quarter {q_start}:{q_end}")
        out.loc[ends[mask]] = (INJECTION_TOTAL_BIL / 4.0) / days
    return out


def _primary_with_injection(primary: pd.DataFrame, injections: pd.Series) -> pd.DataFrame:
    out = primary.copy()
    out["primary_deficit_bil"] = pd.to_numeric(out["primary_deficit_bil"], errors="raise") + injections.to_numpy()
    fy_totals = out.groupby("source_fiscal_year")["primary_deficit_bil"].transform("sum")
    out["annual_or_remaining_primary_deficit_bil"] = fy_totals
    out["source_role"] = "scenario_assumption"
    out["runtime_role"] = "hard_flow"
    out["source_status"] = "OWNER_APPROVED_STYLIZED_2020_SCALE"
    out["claim_boundary"] = "stylized_2020_scale_primary_deficit_injection_not_fitted_to_2020_data"
    out["scenario_transform"] = "fiscal_injection_2028_v1"
    return out


def _debt_with_injection(debt: pd.DataFrame, injections: pd.Series) -> pd.DataFrame:
    out = debt.copy()
    cumulative = _cumulative_injection_by_date(injections)
    dates = pd.to_datetime(out["period_end"], errors="coerce")
    add = dates.map(lambda value: cumulative.get(value.date().isoformat(), 0.0))
    for column in ("cbo_federal_debt_held_public_target_bil", "marketable_treasury_public_target_bil"):
        out[column] = pd.to_numeric(out[column], errors="raise") + add
    out["source_role"] = "scenario_assumption"
    out["runtime_role"] = "hard_target"
    out["source_status"] = "OWNER_APPROVED_STYLIZED_2020_SCALE_MATCHING_DEBT_TARGET"
    out["claim_boundary"] = "debt_target_shift_explicit_scenario_assumption_not_implicit_primary_deficit_plug"
    out["scenario_transform"] = "fiscal_injection_2028_v1"
    return out


def _fed_with_injection(fed: pd.DataFrame, injections: pd.Series) -> pd.DataFrame:
    out = fed.copy()
    cumulative = _cumulative_injection_by_date(injections)
    dates = pd.to_datetime(out["period_end"], errors="coerce")
    add = dates.map(lambda value: cumulative.get(value.date().isoformat(), 0.0) * FED_ABSORPTION_SHARE)
    out["cbo_fed_holdings_target_bil"] = pd.to_numeric(out["cbo_fed_holdings_target_bil"], errors="raise") + add
    out["source_role"] = "scenario_assumption"
    out["runtime_role"] = "hard_target"
    out["source_status"] = "STYLIZED_2020_21_EPISODE_LESS_AGGRESSIVE_FED_ABSORPTION_ASSUMPTION"
    out["claim_boundary"] = "fed_absorption_routes_new_issuance_not_total_debt_or_cash"
    out["scenario_transform"] = "fiscal_injection_2028_v1"
    out["fed_absorption_share_of_injection"] = FED_ABSORPTION_SHARE
    return out


def _cumulative_injection_by_date(injections: pd.Series) -> dict[str, float]:
    total = 0.0
    out: dict[str, float] = {}
    for row_idx, amount in injections.items():
        total += float(amount)
        out[pd.Timestamp(row_idx).date().isoformat()] = total
    return out


def _run_injection_pair_sources(
    baseline: CboBaselinePackage,
    injection_paths: Mapping[str, Path],
    work_root: Path,
    *,
    force: bool,
) -> tuple[Path, Path]:
    run_root = work_root / "source_runs" / "fiscal_injection_2028_v1"
    baseline_run = run_root / "baseline"
    shock_run = run_root / "shock"
    _prepare_run_dirs((baseline_run, shock_run), force=force)
    scenario_dir = work_root / "scenarios" / "fiscal_injection_2028_v1"
    baseline_scenario = _scenario(
        baseline,
        scenario_id="flooded_2028_baseline_no_injection_v1",
        title="Flooded 2028 same-state baseline without injection",
        start_date=INJECTION_START.isoformat(),
        end_date=INJECTION_END.isoformat(),
        overrides={},
    )
    injection_scenario = _scenario(
        baseline,
        scenario_id="fiscal_injection_2028_v1",
        title="OWNER_APPROVED_STYLIZED_2020_SCALE fiscal injection",
        start_date=INJECTION_START.isoformat(),
        end_date=INJECTION_END.isoformat(),
        overrides=_injection_overrides(injection_paths),
    )
    _run_scenario_file(baseline, baseline_scenario, scenario_dir / "baseline_no_injection.json", baseline_run)
    _run_scenario_file(
        baseline,
        injection_scenario,
        scenario_dir / "fiscal_injection_2028_v1.json",
        shock_run,
        output_profile="compact",
    )
    return baseline_run, shock_run


def _run_plus100_pair_sources(
    year: int,
    baseline: CboBaselinePackage,
    export: ForecastStateExport,
    work_root: Path,
    *,
    force: bool,
    injection_paths: Mapping[str, Path] | None,
) -> tuple[Path, Path]:
    run_root = work_root / "source_runs" / f"flooded_state_{year}_plus100bp_year"
    baseline_run = run_root / "baseline"
    shock_run = run_root / "shock"
    _prepare_run_dirs((baseline_run, shock_run), force=force)
    scenario_dir = work_root / "scenarios" / f"flooded_state_{year}_plus100bp_year"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    start = export_start(export)
    end = export_end(export)
    local_injection_paths = _copy_injection_files(injection_paths, scenario_dir) if injection_paths is not None else None
    if local_injection_paths is not None:
        overrides = _injection_overrides(local_injection_paths)
    elif year in {2029, 2031}:
        overrides = {"issuance_mix": _default_issuance_mix(negative_issuance_action="retire_shortest_public_marketable")}
    else:
        overrides = {}
    baseline_scenario = _scenario(
        baseline,
        scenario_id=f"flooded_state_{year}_baseline_v1",
        title=f"Flooded state {year} baseline path",
        start_date=start,
        end_date=end,
        overrides=overrides,
    )
    _run_scenario_file(baseline, baseline_scenario, scenario_dir / "baseline.json", baseline_run)
    shock_surface = scenario_dir / f"flooded_state_{year}_plus100bp_surface.csv"
    _write_plus100bp_surface(
        baseline_run / "compile" / "compiled" / "forecast_inputs" / "tdcsim_yield_curve_surface.csv",
        shock_surface,
        shock_start=date.fromisoformat(start),
        shock_end=date.fromisoformat(end),
    )
    shock_overrides = dict(overrides)
    shock_overrides["nominal_yield_curve"] = {
        "mode": "full_surface_file",
        "file": _file_ref(shock_surface),
    }
    shock_scenario = _scenario(
        baseline,
        scenario_id=f"flooded_state_{year}_plus100bp_year_v1",
        title=f"Flooded state {year} +100bp-year path",
        start_date=start,
        end_date=end,
        overrides=shock_overrides,
    )
    _run_scenario_file(baseline, shock_scenario, scenario_dir / "plus100bp.json", shock_run)
    return baseline_run, shock_run


def export_start(export: ForecastStateExport) -> str:
    return str(read_json(export.export_manifest_path)["opening_state_date"])


def export_end(export: ForecastStateExport) -> str:
    return str(read_json(export.export_manifest_path)["horizon_end_date"])


def _prepare_run_dirs(paths: Sequence[Path], *, force: bool) -> None:
    for path in paths:
        if path.exists():
            if not force:
                raise SystemExit(f"output path exists; pass --force to replace: {path}")
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)


def _run_rolloff_bridge(
    baseline: CboBaselinePackage,
    *,
    state_window: ForecastStateWindow,
    output_dir: Path,
    scenario_dir: Path,
    force: bool,
) -> Path:
    _prepare_run_dirs((output_dir,), force=force)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    parent_opening = str(baseline.manifest.get("date_range", {}).get("opening_state_date"))
    scenario = _scenario(
        baseline,
        scenario_id=f"flooded_rollforward_to_{state_window.state_period}_explicit_rolloff_v1",
        title=f"Flooded no-shock roll-forward to {state_window.state_id} with explicit shortest rolloff",
        start_date=parent_opening,
        end_date=state_window.opening_state_date,
        overrides={"issuance_mix": _default_issuance_mix(negative_issuance_action="retire_shortest_public_marketable")},
    )
    _run_scenario_file(
        baseline,
        scenario,
        scenario_dir / f"rollforward_to_{state_window.state_period}_explicit_rolloff.json",
        output_dir,
        output_profile="compact",
    )
    return output_dir


def _copy_injection_files(paths: Mapping[str, Path], scenario_dir: Path) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for key, source in paths.items():
        target = scenario_dir / source.name
        shutil.copy2(source, target)
        copied[key] = target
    return copied


def _scenario(
    baseline: CboBaselinePackage,
    *,
    scenario_id: str,
    title: str,
    start_date: str,
    end_date: str,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
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
        "overrides": dict(overrides),
        "output": {"profile": "summary", "compression": "gzip", "catalog_sqlite": False},
        "simulation": {"frequency": "daily", "start_date": start_date, "end_date": end_date},
    }


def _injection_overrides(paths: Mapping[str, Path] | None) -> dict[str, Any]:
    if paths is None:
        return {}
    return {
        "primary_deficit": {"mode": "absolute_path_file", "file": _file_ref(paths["primary_deficit"])},
        "debt_target": {"mode": "absolute_path_file", "file": _file_ref(paths["debt_target"])},
        "fed_holdings": {"mode": "absolute_path_file", "file": _file_ref(paths["fed_holdings"])},
        "issuance_mix": _bills_heavy_issuance_mix(),
    }


def _bills_heavy_issuance_mix() -> dict[str, Any]:
    return {
        "mode": "replace_shares",
        "tips_share": 0.02,
        "frn_share": 0.03,
        "fixed_remainder_shares": {"bills": 0.78, "notes": 0.18, "bonds": 0.04},
        "maturity_distributions": {
            "bills": [{"maturity_years": 0.25, "share": 0.65}, {"maturity_years": 0.5, "share": 0.35}],
            "notes": [{"maturity_years": 2.0, "share": 0.55}, {"maturity_years": 5.0, "share": 0.45}],
            "bonds": [{"maturity_years": 20.0, "share": 1.0}],
            "tips": [{"maturity_years": 10.0, "share": 1.0}],
            "frn": [{"maturity_years": 2.0, "share": 1.0}],
        },
        "negative_issuance_action": "error",
    }


def _default_issuance_mix(*, negative_issuance_action: str) -> dict[str, Any]:
    return {
        "mode": "replace_shares",
        "tips_share": 0.06,
        "frn_share": 0.04,
        "fixed_remainder_shares": {"bills": 0.25, "notes": 0.55, "bonds": 0.20},
        "maturity_distributions": {
            "bills": [{"maturity_years": 0.5, "share": 1.0}],
            "notes": [{"maturity_years": 5.0, "share": 1.0}],
            "bonds": [{"maturity_years": 20.0, "share": 1.0}],
            "tips": [{"maturity_years": 10.0, "share": 1.0}],
            "frn": [{"maturity_years": 2.0, "share": 1.0}],
        },
        "negative_issuance_action": negative_issuance_action,
    }


def _file_ref(path: Path) -> dict[str, str]:
    return {"relative_path": path.name, "sha256": sha256_file(path), "media_type": "text/csv"}


def _run_scenario_file(
    baseline: CboBaselinePackage,
    scenario: Mapping[str, Any],
    path: Path,
    output_dir: Path,
    *,
    output_profile: str = "summary",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, dict(scenario))
    spec = CboScenarioSpec.from_file(path)
    spec.assert_baseline_matches(baseline)
    run_cbo_scenario(baseline, spec, output_dir, output_profile=output_profile)


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
    pair_id: str,
    export: ForecastStateExport,
    baseline_run_dir: Path,
    shock_run_dir: Path,
    baseline_scenario_id: str,
    shock_scenario_id: str,
    object_id: str,
    shock_path_id: str,
    shock_bps_year: float,
    denominator_key: str,
    ratewall_period: str,
    beta_case: dict[str, Any],
    require_same_non_rate: bool,
    one_named_rate_shock_only: bool,
) -> dict[str, Any]:
    baseline_manifest = read_json(baseline_run_dir / "tdcsim_cbo_run_manifest.json")
    shock_manifest = read_json(shock_run_dir / "tdcsim_cbo_run_manifest.json")
    route = pd.read_csv(baseline_run_dir / "outputs" / "tdcsim_tdc_principal_route_stock_closure.csv.gz")
    stock_total = float(route["opening_route_stock_bil"].sum())
    by_holder = route.groupby("route_holder_sector")["opening_route_stock_bil"].sum().to_dict()
    export_manifest = read_json(export.export_manifest_path)
    non_rate_digest = _non_rate_digest(baseline_manifest)
    state_payload = {
        "schema": "ratewall_state_fingerprint_v2",
        "source_mode": "flooded_scenario_state_rollforward_export",
        "scenario_state_set_id": STATE_SET_ID,
        "state_id": export.state_id,
        "state_kind": "scenario_state",
        "state_period": str(export.year),
        "opening_state_date": export_manifest["opening_state_date"],
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "derived_state_package_sha256": export.derived_state_package_sha256,
        "forecast_state_export_manifest_sha256": export.forecast_state_export_manifest_sha256,
        "rollforward_run_manifest_sha256": export.rollforward_run_manifest_sha256,
        "compiled_non_rate_inputs_digest": export.compiled_non_rate_inputs_digest,
        "scenario_design": STATE_SET_ID,
    }
    inventory_payload = {
        "schema": "ratewall_state_component_inventory_v2",
        "state_id": export.state_id,
        "compiled_non_rate_inputs_digest": non_rate_digest,
        "opening_route_stock_total_bil": stock_total,
        "holders": {key: float(value) for key, value in by_holder.items()},
    }
    state_sha = canonical_json_sha256(state_payload)
    inventory_sha = canonical_json_sha256(inventory_payload)
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "pair_id": pair_id,
        "scenario_state_set_id": STATE_SET_ID,
        "state_id": export.state_id,
        "state_kind": "scenario_state",
        "state_period": str(export.year),
        "ratewall_period": ratewall_period,
        "scenario_id": STATE_SET_ID,
        "state_fingerprint_sha256": state_sha,
        "state_component_inventory_sha256": inventory_sha,
        "baseline_state_fingerprint_sha256": state_sha,
        "shock_state_fingerprint_sha256": state_sha,
        "opening_state_date": export_manifest["opening_state_date"],
        "actuals_available_as_of": "2026-06-17",
        "source_vintage": "2026-02-11",
        "source_grade_status": "pass_flooded_scenario_state_export",
        "state_construction_method": "scenario_rollforward_export_v1",
        "forecast_state_export_manifest_sha256": export.forecast_state_export_manifest_sha256,
        "derived_state_package_sha256": export.derived_state_package_sha256,
        "parent_baseline_package_sha256": export_manifest["parent_baseline_package_sha256"],
        "parent_baseline_manifest_sha256": export_manifest["parent_baseline_manifest_sha256"],
        "parent_attestation_sha256": export_manifest["parent_attestation_sha256"],
        "rollforward_run_manifest_sha256": export.rollforward_run_manifest_sha256,
        "horizon_start_date": export_manifest["opening_state_date"],
        "horizon_end_date": export_manifest["horizon_end_date"],
        "horizon": "annual_h1_100bp_year",
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
        "opening_tdc_stock_bil": stock_total,
        "opening_route_stock_total_bil": stock_total,
        "opening_route_stock_domestic_nonbank_bil": float(by_holder.get("Private", 0)),
        "opening_route_stock_bank_bil": float(by_holder.get("Banks", 0)),
        "opening_route_stock_foreign_bil": float(by_holder.get("Foreign", 0)),
        "opening_route_stock_fed_bil": float(by_holder.get("CB", 0)),
        "baseline_run_id": baseline_manifest["run_id"],
        "shock_run_id": shock_manifest["run_id"],
        "baseline_scenario_id": baseline_scenario_id,
        "shock_scenario_id": shock_scenario_id,
        "object_id": object_id,
        "shock_path_id": shock_path_id,
        "shock_bps_year": shock_bps_year,
        "denominator_equivalence_key": denominator_key,
        "require_same_baseline_hashes": True,
        "require_same_opening_state": True,
        "require_same_actuals_available_as_of": True,
        "require_same_simulation_dates": True,
        "require_same_period_index": True,
        "require_same_non_rate_compiled_inputs": require_same_non_rate,
        "one_named_rate_shock_only": one_named_rate_shock_only,
        "demand_conversion_cases": [beta_case],
    }


def _assemble_pair(
    *,
    spec: dict[str, Any],
    pair_dir: Path,
    spec_dir: Path,
    baseline_package: CboBaselinePackage,
) -> Path:
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    spec_path = spec_dir / f"{spec['pair_id']}.json"
    write_json(spec_path, spec)
    result = assemble_marginal_tdc_pair(
        spec_path,
        pair_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    verify_marginal_tdc_pair(
        result.output_dir,
        baseline_package=baseline_package.package_path,
        attestation=baseline_package.attestation.path,
    )
    return result.output_dir


def _pre_beta_case(label: str) -> dict[str, Any]:
    return {
        "demand_conversion_case": "pre_beta_pair",
        "beta": 1.0,
        "beta_assumption_id": f"{label}_ratewall_side_conversion_pending",
        "beta_source_status": "not_applied_in_tdcsim_pair_artifact",
        "chi": 1.0,
        "chi_assumption_id": f"{label}_ratewall_side_conversion_pending",
        "chi_source_status": "not_applied_in_tdcsim_pair_artifact",
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


def _parse_years(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _fmt_float(value: float) -> str:
    return format(value, ".15g")


if __name__ == "__main__":
    raise SystemExit(main())
