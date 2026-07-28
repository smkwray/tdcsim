"""Run compiled CBO scenario inputs through TDCSIM."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_paths import compiled_forecast_input_paths, load_cbo_fiscal_baseline
from sim_engine import run_simulation
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    HOLDER_TYPES,
    MATURITY_CATEGORIES,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKETS,
    SECURITY_TYPES,
)
from tdc_validation import validate_events

from ._json import read_json, sha256_file, write_json
from ._schema import validate_schema
from .baseline import CboBaselinePackage
from .compiler import (
    ISSUANCE_MIX_FILE,
    OPENING_RUNTIME_STATE_FILE,
    CboCompiledScenario,
    CboScenarioCompiler,
    HOLDER_PREFERENCE_EVENTS_FILE,
    RUNTIME_ASSUMPTIONS_FILE,
)
from .contract import CboScenarioSpec
from .manifest import build_run_manifest, validation_from_boundary_checks
from .output import hash_output_tree, write_scenario_outputs
from .runtime_identity import distribution_identity


# Cash-closure checks compare accumulated stock balances, so the tolerance absorbs
# float accumulation over a long horizon without admitting an economically real gap.
CASH_CLOSURE_TOLERANCE = 1e-6
_OPENING_REQUIRED_COLUMNS = (
    "BondID",
    "SecurityType",
    "IssueDate",
    "MaturityDate",
    "OriginalMaturityYears",
    "FaceValue",
    "CouponRate",
    "HolderType",
    "Status",
)
_PORTFOLIO_DATE_COLUMNS = tuple(
    column for column, dtype in PORTFOLIO_DTYPES.items() if dtype == "datetime64[ns]"
)
_PORTFOLIO_NUMERIC_COLUMNS = tuple(
    column
    for column, dtype in PORTFOLIO_DTYPES.items()
    if dtype in {"Int64", "float64"}
)
_BOUNDARY_NUMERIC_COLUMNS = (
    "CBOFedAuctionShare",
    "CBOFedAuctionRolloverAddons",
    "TGA",
    "CBOOperatingCashTarget",
    "CBOCashReconciliationResidual",
)
_CASH_RESIDUAL_BOUNDARY_FLAGS = (
    "affects_primary_deficit",
    "affects_net_interest",
    "affects_total_deficit",
    "affects_debt_target",
    "affects_issuance_size",
    "affects_tdc_fiscal_flow",
)


class RunnerError(ValueError):
    """Raised when a compiled CBO scenario cannot run safely."""


@dataclass(frozen=True)
class CboScenarioRun:
    """Result metadata for a CBO scenario run."""

    output_dir: Path
    compiled: CboCompiledScenario
    results_path: Path
    manifest_path: Path
    run_manifest: Mapping[str, Any]


def run_cbo_scenario(
    baseline: CboBaselinePackage,
    spec: CboScenarioSpec,
    output_dir: str | Path,
    *,
    output_profile: str | None = None,
) -> CboScenarioRun:
    """Compile and run one CBO scenario through the existing simulator."""

    out = Path(output_dir).expanduser().resolve()
    if out.exists():
        raise RunnerError(f"scenario output directory already exists: {out}")
    out.mkdir(parents=True)
    compiled = CboScenarioCompiler().compile(baseline, spec, out / "compile")
    scenario_path = out / "scenario.json"
    write_json(scenario_path, spec.data)
    scenario_referenced_files = _copy_scenario_referenced_files(spec, out)
    inputs = compiled.forecast_inputs_dir
    start, end = _simulation_dates(spec, inputs)
    source_metadata = _source_metadata(baseline, inputs)
    _validate_opening_alignment(start, source_metadata, inputs)
    params = build_runtime_params(inputs, actuals_available_as_of=source_metadata["actuals_available_as_of"])
    assumption_statuses = _adapter_assumption_statuses(inputs)
    engine_params = _engine_runtime_params(params)
    engine_scenario_id = _compiled_scenario_id(inputs)
    results, final_portfolio = run_simulation(
        engine_params,
        start,
        end,
        freq="D",
        scenario_name=engine_scenario_id,
    )
    boundaries = validate_run_boundaries(results, inputs)
    _assert_hard_boundaries_pass(boundaries)
    output_cfg = spec.data.get("output", {})
    if not isinstance(output_cfg, Mapping):
        output_cfg = {}
    profile = output_profile or str(output_cfg.get("profile") or "compact")
    compression = str(output_cfg.get("compression") or "gzip")
    output_metadata = {
        "schema_version": "tdcsim_cbo_handoff_v1",
        "scenario_id": spec.scenario_id,
        "run_id": f"{spec.scenario_id}-{spec.canonical_sha256()[:12]}",
        "package_id": baseline.package_id,
        "source_vintage": source_metadata["source_vintage"],
        "actuals_available_as_of": source_metadata["actuals_available_as_of"],
        "scenario_config_sha256": spec.canonical_sha256(),
        "compiled_inputs_digest": compiled.compiled_inputs_digest,
        "mmf_deposit_pass_through": params["private_mmf_split"]["mmf_deposit_pass_through"],
        "mmf_deposit_pass_through_status": assumption_statuses["mmf_deposit_pass_through_status"],
        "fiscal_incidence_policy_id": params["fiscal_incidence_policy"]["policy_id"],
        "fiscal_incidence_policy_status": assumption_statuses["fiscal_incidence_policy_status"],
        "fiscal_incidence_basis": params["fiscal_incidence_policy"]["incidence_basis"],
        "fiscal_incidence_du_share": params["fiscal_incidence_policy"]["du_share"],
        "fiscal_incidence_ru_share": params["fiscal_incidence_policy"]["ru_share"],
        "fiscal_incidence_foreign_share": params["fiscal_incidence_policy"]["foreign_share"],
        "fiscal_incidence_other_share": params["fiscal_incidence_policy"]["other_share"],
        "issuance_profile_status": assumption_statuses["issuance_profile_status"],
    }
    outputs = write_scenario_outputs(
        results,
        final_portfolio,
        out / "outputs",
        profile=profile,
        compression=compression,
        catalog_sqlite=bool(output_cfg.get("catalog_sqlite", False)),
        metadata=output_metadata,
    )
    run_manifest = build_run_manifest(
        scenario_id=spec.scenario_id,
        scenario_sha256=spec.canonical_sha256(),
        compiled=compiled,
        compiled_manifest_relpath=compiled.manifest_path.relative_to(out).as_posix(),
        baseline=baseline,
        scenario=spec,
        scenario_relpath=scenario_path.relative_to(out).as_posix(),
        scenario_file_sha256=sha256_file(scenario_path),
        scenario_referenced_files=scenario_referenced_files,
        start_date=start,
        end_date=end,
        fiscal_incidence_policy_id=params["fiscal_incidence_policy"]["policy_id"],
        outputs=outputs,
        output_hashes=hash_output_tree(out / "outputs"),
        boundary_checks=boundaries,
        code_environment=_code_environment(baseline, out),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    manifest_path = out / "tdcsim_cbo_run_manifest.json"
    with files("tdcsim_cbo").joinpath("schemas/cbo-run-manifest-v1.schema.json").open("r", encoding="utf-8") as handle:
        run_manifest_schema = json.load(handle)
    validate_schema(
        run_manifest,
        run_manifest_schema,
        label="run_manifest",
    )
    write_json(manifest_path, run_manifest)
    result_path = out / "outputs" / f"results_{profile}{'.csv.gz' if compression == 'gzip' else '.csv'}"
    return CboScenarioRun(
        output_dir=out,
        compiled=compiled,
        results_path=result_path,
        manifest_path=manifest_path,
        run_manifest=run_manifest,
    )


def build_runtime_params(inputs_dir: str | Path, *, actuals_available_as_of: str | None = None) -> dict[str, Any]:
    """Build simulator params from compiled forecast inputs and scenario config."""

    inputs = Path(inputs_dir)
    initial_portfolio = _load_opening_portfolio(inputs / "tdcsim_opening_portfolio.csv")
    base_tga = _opening_operating_cash_target(inputs / "tdcsim_operating_cash_path.csv")
    initial_values = _opening_runtime_initial_values(inputs, base_tga=base_tga)
    holder_preferences = _holder_preferences(inputs / "tdcsim_holder_profile_assumptions.csv")
    holder_events = _holder_preference_events(inputs / HOLDER_PREFERENCE_EVENTS_FILE)
    issuance_profile = _issuance_profile(inputs)
    fiscal_incidence_policy = _fiscal_incidence_policy(inputs)
    if _fed_target_active(inputs / "tdcsim_fed_holdings_path.csv"):
        _assert_no_cb_auction_preferences(holder_preferences)
    return {
        "initial_values": initial_values,
        "tga_params": {"target_balance": base_tga, "floor": -1e15},
        "fiscal_params": {
            "initial_weekly_spending": 0.0,
            "initial_weekly_taxes": 0.0,
            "spending_growth_qtr": 0.0,
            "tax_growth_qtr": 0.0,
        },
        "other_flows": {"reserve_transfer": 0.0, "cb_net_expense": 0.0, "money_minting_transfers": 0.0},
        "treasury_issuance_profile": issuance_profile,
        "yield_curve": {
            "use_static": True,
            "years": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            "rates": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        },
        "yield_curve_surface": {"file": str(inputs / "tdcsim_yield_curve_surface.csv"), "interpolation_method": "pchip", "floor_zero": False},
        "sector_preferences": holder_preferences,
        "private_mmf_split": {
            "bills": 0.25,
            "notes": 0.10,
            "bonds": 0.05,
            "tips": 0.05,
            "frn": 0.20,
            "mmf_deposit_pass_through": _mmf_deposit_pass_through(inputs),
        },
        "tips_params": {
            "cpi_start_level": 100.0,
            "cpi_annual_inflation": 0.0,
            "ref_cpi_lag_months": 3,
            "reference_cpi_start_level": _opening_tips_reference_cpi(initial_portfolio),
            "default_real_coupon_rate": 0.005,
        },
        "frn_params": {"benchmark_maturity_years": 0.25, "default_fixed_spread": 0.0013},
        "financing_cost_options": {"include_tips_inflation_accretion": True},
        "simulation_period": {"enable_preference_trading": False},
        "initial_bonds_df": initial_portfolio,
        "events": holder_events,
        "funding_rule": {
            "mode": "cbo_public_debt_target",
            "target_enforcement": "every_period",
            "negative_required_issuance_action": _negative_issuance_action(inputs),
            "target_tolerance_bil": 0.000001,
            "fed_secondary_sale_buyer_mix": {
                "type": "contemporaneous_non_cb_public_holder_mix",
                "basis": "par_or_adjusted_principal_stock",
            },
        },
        "baseline_input_paths": compiled_forecast_input_paths(inputs),
        "data_vintage": {
            "actuals_available_as_of": actuals_available_as_of or _max_available_date(inputs) or "1900-01-01",
            "allow_lookahead": False,
        },
        "fiscal_incidence_policy": fiscal_incidence_policy,
        "budget_interest": {"cbo_comparison_role": "nonbinding_validation_check"},
    }


def _engine_runtime_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Remove adapter identity metadata that is not part of the generic engine schema."""

    engine_params = dict(params)
    policy = params.get("fiscal_incidence_policy")
    if not isinstance(policy, Mapping) or not policy.get("policy_id"):
        raise RunnerError("runtime fiscal incidence policy must carry a selected policy_id")
    engine_params["fiscal_incidence_policy"] = {
        key: value for key, value in policy.items() if key != "policy_id"
    }
    return engine_params


def validate_run_boundaries(results: pd.DataFrame, inputs_dir: str | Path) -> dict[str, Any]:
    """Compute hard boundary checks for CBO scenario runs."""

    _validate_boundary_result_evidence(results)
    inputs = Path(inputs_dir)
    residual_path = inputs / "tdcsim_cash_reconciliation_residual.csv"
    try:
        residual = pd.read_csv(residual_path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RunnerError(
            f"required cash-reconciliation boundary evidence is missing: {residual_path.name}"
        ) from exc
    if residual.empty:
        raise RunnerError(f"cash-reconciliation boundary evidence is empty: {residual_path.name}")
    residual_flags = {}
    missing_flags = [column for column in _CASH_RESIDUAL_BOUNDARY_FLAGS if column not in residual.columns]
    if missing_flags:
        raise RunnerError(
            f"cash-reconciliation boundary evidence is missing required flags: {missing_flags}"
        )
    for col in _CASH_RESIDUAL_BOUNDARY_FLAGS:
        values = [_strict_bool(value, label=f"{residual_path.name}.{col}") for value in residual[col]]
        if not values:
            raise RunnerError(f"cash-reconciliation boundary flag has no evidence: {col}")
        residual_flags[col] = sorted(set(values))
    fed_auction_share_max = _max_abs(results, "CBOFedAuctionShare")
    fed_auction_face_max = _max_abs(results, "CBOFedAuctionRolloverAddons")
    fed_auction_face_sum = _sum_abs(results, "CBOFedAuctionRolloverAddons")
    fed_boundary_pass = fed_auction_share_max <= 1e-12 and fed_auction_face_max <= 1e-12
    cash_closure = _cash_closure_checks(results)
    omf = omf_reconciliation(results, inputs)
    return {
        "cash_residual_nonfunding_flags": residual_flags,
        "cash_residual_affects_issuance_size": residual_flags.get("affects_issuance_size", []),
        "max_abs_fed_auction_share": fed_auction_share_max,
        "max_abs_fed_auction_face": fed_auction_face_max,
        "sum_abs_fed_auction_face": fed_auction_face_sum,
        "fed_target_holder_allocation_only": fed_boundary_pass,
        "net_interest_role": "diagnostic_nonbinding",
        "remittance_deferred_asset_status": "unsupported_in_cbo_scenario_lane",
        **cash_closure,
        **omf,
    }


def _assert_hard_boundaries_pass(boundary_checks: Mapping[str, Any]) -> None:
    validation = validation_from_boundary_checks(boundary_checks)
    if validation.get("status") == "pass":
        return
    failed = [
        str(item.get("id"))
        for section in ("gates", "invariants")
        for item in validation.get(section, [])
        if isinstance(item, Mapping) and item.get("status") == "fail"
    ]
    raise RunnerError(f"hard run boundary failed: {', '.join(failed) or 'unknown boundary'}")


def _validate_boundary_result_evidence(results: pd.DataFrame) -> None:
    if not isinstance(results, pd.DataFrame) or results.empty:
        raise RunnerError("simulation returned no boundary evidence")
    for column in _BOUNDARY_NUMERIC_COLUMNS:
        _numeric_column(results, column)
    if "CBOCashResidualStatus" not in results.columns:
        raise RunnerError("simulation results are missing boundary evidence column: CBOCashResidualStatus")
    if len(results) > 1:
        statuses = results.iloc[1:]["CBOCashResidualStatus"].astype("string")
        if statuses.isna().any() or statuses.str.strip().eq("").any():
            raise RunnerError("simulation results contain missing CBOCashResidualStatus evidence")
        unexpected = sorted(
            set(statuses.astype(str))
            - {"operating_cash_target_loaded", "operating_cash_path_not_configured"}
        )
        if unexpected:
            raise RunnerError(
                f"simulation results contain unknown CBOCashResidualStatus values: {unexpected}"
            )


def _cash_closure_checks(results: pd.DataFrame) -> dict[str, Any]:
    """Measure whether the modeled Treasury cash chain actually closes.

    Two distinct defects are detected, because the CBO lane can fail either way:

    * a negative TGA is an unmodeled Treasury overdraft — the run financed itself from
      an account that does not exist in the modeled funding chain;
    * a nonzero cash-reconciliation residual is booked to the TGA alone
      (``sim_engine`` adds it to ``tga_change_period`` with no reserve, Fed asset, or
      named financing instrument on the other side), so it creates or destroys cash
      without a counterparty.

    Both are reported as observed magnitudes so a consumer can see the size of the
    breach rather than only that one occurred.
    """

    tga = _numeric_column(results, "TGA")
    min_tga = float(tga.min()) if not tga.empty else 0.0
    negative_tga_periods = int((tga < -CASH_CLOSURE_TOLERANCE).sum()) if not tga.empty else 0

    # An operating-cash gap is only a defect where a target was actually configured;
    # `operating_cash_path_not_configured` periods set the target to the projected
    # balance, so measuring them would compare a value against itself.
    targeted = results
    if "CBOCashResidualStatus" in results.columns:
        targeted = results[results["CBOCashResidualStatus"].astype("string") == "operating_cash_target_loaded"]
    targeted_tga = _numeric_column(targeted, "TGA")
    targeted_target = _numeric_column(targeted, "CBOOperatingCashTarget")
    if targeted_tga.empty or targeted_target.empty:
        max_gap = 0.0
        targeted_periods = 0
    else:
        aligned = targeted_tga.align(targeted_target, join="inner")
        max_gap = float((aligned[0] - aligned[1]).abs().max())
        targeted_periods = int(len(aligned[0]))

    unbooked_residual = _sum_abs(results, "CBOCashReconciliationResidual")
    return {
        "min_tga": min_tga,
        "negative_tga_periods": negative_tga_periods,
        "tga_nonnegative": negative_tga_periods == 0,
        "max_abs_operating_cash_gap": max_gap,
        "operating_cash_targeted_periods": targeted_periods,
        "operating_cash_target_met": max_gap <= CASH_CLOSURE_TOLERANCE,
        "sum_abs_unbooked_cash_residual": unbooked_residual,
        "cash_residual_fully_booked": unbooked_residual <= CASH_CLOSURE_TOLERANCE,
    }


def omf_reconciliation(results: pd.DataFrame, inputs_dir: str | Path) -> dict[str, Any]:
    """Test whether the modeled cash gap is reconcilable to CBO's published OMF control.

    CBO's debt identity is ``debt_begin + total_deficit + other_means_financing = debt_end``,
    and the published "other means of financing" row is a bounded reconciliation category —
    changes in government cash balances, federal-credit cash flows, accrual-versus-paid
    timing, agency debt. It closes CBO's own identity exactly, at a scale of tens of billions
    per year.

    So OMF authorises a *decomposition bridge*, not a plug. If the modeled cash gap exceeds
    what the published control can carry, the honest conclusion is not that OMF absorbs it —
    it is that the operating-cash path cannot be a hard constraint. That path declares itself
    ``source_role=scenario_assumption`` with claim boundary
    ``operating_cash_proxy_not_debt_target_or_issuance_supply``, whereas debt stock, total
    deficit, and OMF are published CBO controls. The assumption yields; the controls do not.

    Returns the comparison so a reader sees the magnitudes rather than only a verdict.
    """

    fiscal = Path(inputs_dir) / "tdcsim_cbo_fiscal_baseline.csv"
    if not fiscal.is_file():
        return {"omf_control_available": False}
    frame = load_cbo_fiscal_baseline(fiscal)
    if "cbo_other_means_financing_bil" not in frame.columns:
        return {"omf_control_available": False}
    raw = frame["cbo_other_means_financing_bil"]
    omf = pd.to_numeric(raw, errors="coerce")
    # The OMF row is a required governed input. Coercing an unparsable value to NaN and
    # dropping it would silently shrink the control this comparison is measured against,
    # making an unreconcilable gap look reconcilable.
    malformed = int((omf.isna() & raw.notna()).sum())
    if malformed:
        raise RunnerError(
            f"CBO fiscal baseline has {malformed} malformed cbo_other_means_financing_bil "
            f"value(s) in {fiscal.name}"
        )
    omf = omf.dropna()
    max_abs_omf = float(omf.abs().max()) if not omf.empty else 0.0
    gap = float(_cash_closure_checks(results).get("max_abs_operating_cash_gap", 0.0))
    return {
        "omf_control_available": True,
        "max_abs_cbo_omf_control_bil": max_abs_omf,
        "max_abs_operating_cash_gap_bil": gap,
        # A gap the published control cannot carry is not an OMF flow. Calling it one would
        # be a project-created plug wearing a CBO label.
        "cash_gap_within_omf_control": gap <= max_abs_omf + CASH_CLOSURE_TOLERANCE,
    }


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise RunnerError(f"simulation results are missing boundary evidence column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise RunnerError(f"simulation results contain missing or malformed {column} evidence")
    if not values.map(math.isfinite).all():
        raise RunnerError(f"simulation results contain nonfinite {column} evidence")
    return values.astype(float)


def _simulation_dates(spec: CboScenarioSpec, inputs_dir: Path) -> tuple[str, str]:
    sim = spec.data.get("simulation", {})
    if isinstance(sim, Mapping) and sim.get("start_date") and sim.get("end_date"):
        return str(sim["start_date"]), str(sim["end_date"])
    primary = pd.read_csv(inputs_dir / "tdcsim_primary_deficit_path.csv")
    if {"period_start", "period_end"} <= set(primary.columns):
        starts = sorted(str(value) for value in primary["period_start"].dropna())
        ends = sorted(str(value) for value in primary["period_end"].dropna())
        if starts and ends:
            return starts[0], ends[-1]
    debt = pd.read_csv(inputs_dir / "tdcsim_debt_stock_path.csv")
    if "period_end" in debt.columns:
        dates = sorted(str(value) for value in debt["period_end"].dropna())
        if dates:
            return dates[0], dates[-1]
    raise RunnerError("could not infer simulation start/end dates from scenario or compiled inputs")


def _source_metadata(baseline: CboBaselinePackage, inputs_dir: Path) -> dict[str, str]:
    date_range = baseline.manifest.get("date_range", {}) if isinstance(baseline.manifest, Mapping) else {}
    if not isinstance(date_range, Mapping):
        date_range = {}
    actuals = str(date_range.get("actuals_available_as_of") or _max_available_date(inputs_dir) or "")
    opening = str(date_range.get("opening_state_date") or _opening_state_date(inputs_dir) or "")
    if not actuals:
        raise RunnerError("could not determine actuals_available_as_of from baseline package or compiled inputs")
    if not opening:
        raise RunnerError("could not determine opening_state_date from baseline package or compiled inputs")
    return {
        "actuals_available_as_of": actuals,
        "opening_state_date": opening,
        "source_vintage": str(baseline.manifest.get("forecast_publication_date") or baseline.package_id),
    }


def _validate_opening_alignment(start_date: str, source_metadata: Mapping[str, str], inputs_dir: Path) -> None:
    opening = str(source_metadata["opening_state_date"])
    if start_date != opening:
        raise RunnerError(
            f"simulation.start_date must equal opening_state_date for CBO runs; "
            f"got start_date={start_date}, opening_state_date={opening}"
        )
    actuals = str(source_metadata["actuals_available_as_of"])
    if actuals > start_date:
        raise RunnerError(
            f"actuals_available_as_of must not be after simulation.start_date; "
            f"got actuals_available_as_of={actuals}, start_date={start_date}"
        )
    portfolio = _load_opening_portfolio(inputs_dir / "tdcsim_opening_portfolio.csv")
    if "Status" not in portfolio.columns or "MaturityDate" not in portfolio.columns:
        return
    active = portfolio[portfolio["Status"].astype(str).eq("Active")].copy()
    if active.empty:
        return
    maturity = pd.to_datetime(active["MaturityDate"], errors="coerce")
    stale = active[maturity <= pd.Timestamp(start_date)]
    if not stale.empty:
        face = pd.to_numeric(stale.get("FaceValue", 0.0), errors="coerce").fillna(0.0).sum()
        raise RunnerError(
            f"opening portfolio has {len(stale)} active securities maturing on/before simulation.start_date "
            f"{start_date}; stale face={float(face):.6f} billion"
        )


def _opening_operating_cash_target(path: Path) -> float:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RunnerError(f"required opening operating-cash input is missing: {path.name}") from exc
    if frame.empty or "operating_cash_target_bil" not in frame.columns:
        raise RunnerError(f"{path.name} must contain operating_cash_target_bil evidence")
    raw = frame.iloc[0]["operating_cash_target_bil"]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"{path.name} opening operating_cash_target_bil must be numeric") from exc
    if not math.isfinite(value):
        raise RunnerError(f"{path.name} opening operating_cash_target_bil must be finite")
    return value


def _opening_runtime_initial_values(inputs: Path, *, base_tga: float) -> dict[str, float]:
    runtime_state_path = inputs / OPENING_RUNTIME_STATE_FILE
    if not runtime_state_path.exists():
        raise RunnerError(f"required opening runtime state is missing: {runtime_state_path.name}")
    try:
        payload = read_json(runtime_state_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"opening runtime state is unreadable: {runtime_state_path.name}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("tdcsim_opening_runtime_state.json must be an object")
    if payload.get("schema_version") != "tdcsim_cbo_opening_runtime_state_v1":
        raise RunnerError("tdcsim_opening_runtime_state.json has an unsupported schema_version")
    opening = str(payload.get("opening_state_date") or "")
    expected_opening = _opening_state_date(inputs)
    if opening != str(expected_opening or ""):
        raise RunnerError("opening runtime state date does not match opening_state_date")
    raw = payload.get("initial_values")
    if not isinstance(raw, Mapping):
        raise RunnerError("opening runtime state initial_values must be an object")
    missing = [key for key in ("reserves", "tdc_level", "tga") if key not in raw]
    if missing:
        raise RunnerError(f"opening runtime state initial_values is missing required keys: {missing}")
    try:
        values = {
            "reserves": float(raw["reserves"]),
            "tdc_level": float(raw["tdc_level"]),
            "tga": float(raw["tga"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("opening runtime state initial_values must be numeric") from exc
    if abs(values["tga"] - base_tga) > 1e-9:
        raise RunnerError("opening runtime state TGA does not match operating cash opening target")
    if not all(math.isfinite(value) for value in values.values()):
        raise RunnerError("opening runtime state initial_values must be finite")
    if payload.get("state_kind") == "source_baseline_configured_defaults":
        expected_statuses = {
            "reserves": "configured_default",
            "tdc_level": "configured_default",
            "tga": "source_opening_cash",
        }
        if payload.get("initial_value_statuses") != expected_statuses:
            raise RunnerError(
                "source baseline opening runtime state has invalid initial_value_statuses"
            )
        if payload.get("configured_default_count") != 2:
            raise RunnerError(
                "source baseline opening runtime state configured_default_count must be 2"
            )
    return values


def _opening_state_date(inputs_dir: Path) -> str | None:
    metadata_path = inputs_dir / "tdcsim_opening_portfolio_metadata.json"
    if metadata_path.exists():
        payload = read_json(metadata_path)
        if isinstance(payload, Mapping):
            for key in ("opening_state_date", "simulation_start_date", "opening_date"):
                if payload.get(key):
                    return str(payload[key])
    primary_path = inputs_dir / "tdcsim_primary_deficit_path.csv"
    if primary_path.exists():
        primary = pd.read_csv(primary_path)
        if "period_start" in primary.columns and not primary.empty:
            dates = sorted(str(value) for value in primary["period_start"].dropna())
            if dates:
                return dates[0]
    return None


def _max_available_date(inputs_dir: Path) -> str | None:
    dates: list[str] = []
    for path in sorted(inputs_dir.glob("*.csv")):
        try:
            frame = pd.read_csv(path, usecols=lambda col: col == "available_date")
        except ValueError:
            continue
        except pd.errors.EmptyDataError:
            continue
        if "available_date" in frame.columns:
            dates.extend(str(value) for value in frame["available_date"].dropna() if str(value).strip())
    return max(dates) if dates else None


def _load_opening_portfolio(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RunnerError(f"required opening portfolio is missing or empty: {path.name}") from exc
    missing_columns = [column for column in _OPENING_REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise RunnerError(f"opening portfolio is missing required columns: {missing_columns}")

    for column in _OPENING_REQUIRED_COLUMNS:
        if not _present_values(frame[column]).all():
            raise RunnerError(f"opening portfolio required column {column} contains missing values")

    for column in _PORTFOLIO_DATE_COLUMNS:
        if column not in frame.columns:
            continue
        present = _present_values(frame[column])
        parsed = pd.to_datetime(frame[column], errors="coerce")
        if parsed[present].isna().any():
            raise RunnerError(f"opening portfolio column {column} contains malformed dates")
        frame[column] = parsed

    for column in _PORTFOLIO_NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        present = _present_values(frame[column])
        parsed = pd.to_numeric(frame[column], errors="coerce")
        if parsed[present].isna().any():
            raise RunnerError(f"opening portfolio column {column} contains malformed numerics")
        if not parsed[present].map(math.isfinite).all():
            raise RunnerError(f"opening portfolio column {column} contains nonfinite numerics")
        frame[column] = parsed

    bond_ids = frame["BondID"]
    if not bond_ids.map(lambda value: float(value).is_integer()).all():
        raise RunnerError("opening portfolio BondID values must be integers")
    if bond_ids.duplicated().any():
        raise RunnerError("opening portfolio BondID values must be unique")
    unknown_security = sorted(set(frame["SecurityType"].astype(str)) - set(SECURITY_TYPES))
    if unknown_security:
        raise RunnerError(f"opening portfolio contains unknown SecurityType values: {unknown_security}")
    unknown_holder = sorted(set(frame["HolderType"].astype(str)) - set(HOLDER_TYPES))
    if unknown_holder:
        raise RunnerError(f"opening portfolio contains unknown HolderType values: {unknown_holder}")
    invalid_status = sorted(set(frame["Status"].astype(str)) - {"Active"})
    if invalid_status:
        raise RunnerError(f"opening portfolio contains unsupported Status values: {invalid_status}")
    if (frame["FaceValue"] < 0.0).any():
        raise RunnerError("opening portfolio FaceValue must be nonnegative")
    if (frame["OriginalMaturityYears"] <= 0.0).any():
        raise RunnerError("opening portfolio OriginalMaturityYears must be positive")
    if (frame["CouponRate"] < 0.0).any():
        raise RunnerError("opening portfolio CouponRate must be nonnegative")
    if (frame["IssueDate"] >= frame["MaturityDate"]).any():
        raise RunnerError("opening portfolio IssueDate must precede MaturityDate")

    fixed = frame["SecurityType"].astype(str).eq("Fixed")
    if fixed.any():
        if "MaturityCategory" not in frame.columns:
            raise RunnerError("opening Fixed securities require MaturityCategory")
        fixed_categories = frame.loc[fixed, "MaturityCategory"]
        if not _present_values(fixed_categories).all():
            raise RunnerError("opening Fixed securities require a named MaturityCategory")
        invalid_categories = sorted(set(fixed_categories.astype(str)) - set(MATURITY_CATEGORIES))
        if invalid_categories:
            raise RunnerError(
                f"opening Fixed securities contain unknown MaturityCategory values: {invalid_categories}"
            )

    tips = frame["SecurityType"].astype(str).eq("TIPS")
    for column in ("OriginalPrincipal", "AdjustedPrincipal", "ReferenceCPI_Issue", "IndexRatio"):
        if tips.any() and (column not in frame.columns or not _present_values(frame.loc[tips, column]).all()):
            raise RunnerError(f"opening TIPS securities require {column}")
    if tips.any():
        if (frame.loc[tips, ["OriginalPrincipal", "AdjustedPrincipal"]] < 0.0).any().any():
            raise RunnerError("opening TIPS principal stocks must be nonnegative")
        if (frame.loc[tips, ["ReferenceCPI_Issue", "IndexRatio"]] <= 0.0).any().any():
            raise RunnerError("opening TIPS reference CPI and index ratio must be positive")

    frn = frame["SecurityType"].astype(str).eq("FRN")
    if frn.any():
        if "FixedSpread" not in frame.columns or not _present_values(
            frame.loc[frn, "FixedSpread"]
        ).all():
            raise RunnerError("opening FRN securities require finite FixedSpread")
        if not frame.loc[frn, "FixedSpread"].map(math.isfinite).all():
            raise RunnerError("opening FRN securities require finite FixedSpread")

    for col in BOND_PORTFOLIO_COLS:
        if col not in frame.columns:
            dtype = PORTFOLIO_DTYPES[col]
            frame[col] = pd.NaT if dtype == "datetime64[ns]" else (float("nan") if dtype == "float64" else pd.NA)
    try:
        return frame[BOND_PORTFOLIO_COLS].astype(PORTFOLIO_DTYPES)
    except (TypeError, ValueError) as exc:
        raise RunnerError("opening portfolio could not be converted to the runtime schema") from exc


def _present_values(values: pd.Series) -> pd.Series:
    return values.notna() & values.astype("string").str.strip().ne("")


def _compiled_scenario_id(inputs: Path) -> str:
    for filename in ("tdcsim_debt_stock_path.csv", "tdcsim_primary_deficit_path.csv", "tdcsim_yield_curve_surface.csv"):
        frame = pd.read_csv(inputs / filename, nrows=1)
        if "scenario_id" in frame.columns and len(frame) > 0:
            return str(frame.iloc[0]["scenario_id"])
    return "default"


def _holder_preferences(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path)
    prefs: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        holder = str(row.get("holder_type") or row.get("HolderType") or "")
        if not holder:
            continue
        holder_subbucket = row.get("holder_subbucket", "")
        if pd.notna(holder_subbucket) and str(holder_subbucket).strip():
            continue
        prefs[holder] = {
            "bills_pct": _finite_share(row.get("bills_pct"), label=f"{holder}.bills_pct"),
            "notes_pct": _finite_share(row.get("notes_pct"), label=f"{holder}.notes_pct"),
            "bonds_pct": _finite_share(row.get("bonds_pct"), label=f"{holder}.bonds_pct"),
            "tips_pct": _finite_share(row.get("tips_pct"), label=f"{holder}.tips_pct"),
            "frn_pct": _finite_share(row.get("frn_pct"), label=f"{holder}.frn_pct"),
        }
    for holder in HOLDER_TYPES:
        prefs.setdefault(holder, {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0, "frn_pct": 0.0})
    _assert_holder_preference_sums(prefs)
    private_subbucket_shares = _private_subbucket_shares(frame)
    if private_subbucket_shares:
        prefs["__private_subbucket_shares__"] = private_subbucket_shares
    return prefs


def _private_subbucket_shares(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    categories = ("bills", "notes", "bonds", "tips", "frn")
    shares: dict[str, dict[str, float]] = {}
    if "holder_subbucket" not in frame.columns:
        return shares
    holder = frame.get("holder_type", frame.get("HolderType", pd.Series("", index=frame.index))).fillna("").astype(str)
    subbucket = frame["holder_subbucket"].fillna("").astype(str)
    private_rows = frame.loc[(holder == "Private") & subbucket.isin(PRIVATE_SUBBUCKETS)].copy()
    if private_rows.empty:
        return shares
    for category in categories:
        col = f"{category}_route_share"
        if col not in private_rows.columns:
            continue
        category_values: dict[str, float] = {}
        has_value = False
        for _, row in private_rows.iterrows():
            route = str(row["holder_subbucket"])
            value = row.get(col)
            if pd.isna(value) or str(value).strip() == "":
                continue
            has_value = True
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise RunnerError(f"private route share {route}.{col} must be numeric") from exc
            if not math.isfinite(number) or number < 0.0:
                raise RunnerError(f"private route share {route}.{col} must be finite and nonnegative")
            category_values[route] = number
        if not has_value:
            continue
        total = sum(category_values.values())
        if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise RunnerError(f"private route shares for {category} must sum to 1.0, got {total}")
        shares[category] = {route: category_values.get(route, 0.0) for route in PRIVATE_SUBBUCKETS}
    return shares


def _finite_share(value: Any, *, label: str) -> float:
    if pd.isna(value):
        raise RunnerError(f"holder preference {label} is missing")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise RunnerError(f"holder preference {label} must be a finite nonnegative number")
    return number


def _strict_bool(value: Any, *, label: str) -> bool:
    if isinstance(value, bool) or pd.api.types.is_bool(value):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise RunnerError(f"{label} must contain only explicit boolean values")


def _assert_holder_preference_sums(prefs: Mapping[str, Mapping[str, float]]) -> None:
    for pref_key in ("bills_pct", "notes_pct", "bonds_pct", "tips_pct", "frn_pct"):
        total = sum(float(prefs.get(holder, {}).get(pref_key, 0.0)) for holder in HOLDER_TYPES)
        if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
            raise RunnerError(f"holder preference {pref_key} must sum to 1.0 across aggregate holders, got {total}")


def _holder_preference_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise RunnerError("compiled holder preference events must be a JSON object")
    events = payload.get("events", [])
    if not isinstance(events, list):
        raise RunnerError("compiled holder preference events must contain an events list")
    errors = validate_events(events, label="compiled holder preference events")
    if errors:
        raise RunnerError("; ".join(errors))
    return events


def _issuance_profile(inputs: Path) -> dict[str, Any]:
    payload = _issuance_assumptions(inputs)
    mode = payload.get("mode")
    if mode not in {"default_tdcsim_cbo_runner_profile", "replace_shares"}:
        raise RunnerError(f"unsupported compiled issuance mix mode: {mode!r}")
    shares = payload.get("security_shares")
    maturity = payload.get("maturity_distributions")
    if not isinstance(shares, Mapping):
        raise RunnerError("compiled issuance mix security_shares must be an object")
    if not isinstance(maturity, Mapping):
        raise RunnerError("compiled issuance mix maturity_distributions must be an object")
    categories = ("bills", "notes", "bonds", "tips", "frn")
    missing_shares = [category for category in categories if category not in shares]
    missing_maturity = [category for category in categories if category not in maturity]
    if missing_shares:
        raise RunnerError(f"compiled issuance mix is missing security shares: {missing_shares}")
    if missing_maturity:
        raise RunnerError(f"compiled issuance mix is missing maturity distributions: {missing_maturity}")
    normalized_shares = {
        category: _finite_nonnegative_number(
            shares[category],
            label=f"issuance security_shares.{category}",
        )
        for category in categories
    }
    share_total = sum(normalized_shares.values())
    if abs(share_total - 1.0) > 1e-9:
        raise RunnerError(f"compiled issuance security shares must sum to 1.0, got {share_total}")
    normalized_maturity = {
        category: _strict_maturity_distribution(maturity[category], category=category)
        for category in categories
    }
    fixed_total = sum(normalized_shares[category] for category in MATURITY_CATEGORIES)
    fixed = {
        category: 0.0 if fixed_total == 0 else normalized_shares[category] / fixed_total
        for category in MATURITY_CATEGORIES
    }
    return {
        "bills": _fixed_profile(fixed["bills"], normalized_maturity["bills"], cutoff=1.0),
        "notes": _fixed_profile(fixed["notes"], normalized_maturity["notes"], cutoff=10.0),
        "bonds": _fixed_profile(fixed["bonds"], normalized_maturity["bonds"], cutoff=999.0),
        "TIPS": _special_profile(normalized_shares["tips"], normalized_maturity["tips"]),
        "FRN": _special_profile(normalized_shares["frn"], normalized_maturity["frn"]),
        "NonMarketable": {"target_percentage": 0.0, "maturities": [30.0], "maturity_distribution": [1.0]},
        "remainder_maturity_years": 1.0,
    }


def _fixed_profile(share: float, rows: list[Mapping[str, Any]], *, cutoff: float) -> dict[str, Any]:
    return {
        "category_cutoff_years": cutoff,
        "target_percentage_of_remainder": share,
        "maturities": [float(row["maturity_years"]) for row in rows],
        "maturity_distribution": [float(row["share"]) for row in rows],
    }


def _special_profile(share: float, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "target_percentage": share,
        "maturities": [float(row["maturity_years"]) for row in rows],
        "maturity_distribution": [float(row["share"]) for row in rows],
    }


def _strict_maturity_distribution(value: Any, *, category: str) -> list[dict[str, float]]:
    if not isinstance(value, list) or not value:
        raise RunnerError(f"issuance maturity distribution {category} must be a nonempty array")
    rows: list[dict[str, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise RunnerError(f"issuance maturity distribution {category}[{index}] must be an object")
        if "maturity_years" not in item or "share" not in item:
            raise RunnerError(
                f"issuance maturity distribution {category}[{index}] requires maturity_years and share"
            )
        maturity_years = _finite_nonnegative_number(
            item["maturity_years"],
            label=f"issuance maturity_distributions.{category}[{index}].maturity_years",
        )
        if maturity_years <= 0.0:
            raise RunnerError(f"issuance maturity {category}[{index}] must be positive")
        share = _finite_nonnegative_number(
            item["share"],
            label=f"issuance maturity_distributions.{category}[{index}].share",
        )
        rows.append({"maturity_years": maturity_years, "share": share})
    total = sum(row["share"] for row in rows)
    if abs(total - 1.0) > 1e-9:
        raise RunnerError(f"issuance maturity distribution {category} must sum to 1.0, got {total}")
    return rows


def _finite_nonnegative_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RunnerError(f"{label} must be finite and nonnegative")
    return number


def _negative_issuance_action(inputs: Path) -> str:
    payload = _issuance_assumptions(inputs)
    action = payload.get("negative_issuance_action")
    if action not in {"error", "retire_shortest_public_marketable"}:
        raise RunnerError(f"unsupported negative_issuance_action: {action!r}")
    return str(action)


def _mmf_deposit_pass_through(inputs: Path) -> float:
    payload = _runtime_assumptions(inputs)
    try:
        value = float(payload["mmf_deposit_pass_through"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("mmf_deposit_pass_through must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RunnerError("mmf_deposit_pass_through must be between 0.0 and 1.0")
    return value


def _runtime_assumptions(inputs: Path) -> Mapping[str, Any]:
    path = inputs / RUNTIME_ASSUMPTIONS_FILE
    if not path.exists():
        raise RunnerError(f"required compiled runtime assumptions are missing: {path.name}")
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"compiled runtime assumptions are unreadable: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("compiled runtime assumptions must be a JSON object")
    if payload.get("schema_version") != "tdcsim_cbo_runtime_assumptions_v1":
        raise RunnerError("compiled runtime assumptions has an unsupported schema_version")
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
        raise RunnerError(f"compiled runtime assumptions are missing required keys: {missing}")
    _assumption_status(
        payload["fiscal_incidence_policy_status"],
        label="fiscal_incidence_policy_status",
    )
    _assumption_status(
        payload["mmf_deposit_pass_through_status"],
        label="mmf_deposit_pass_through_status",
    )
    return payload


def _issuance_assumptions(inputs: Path) -> Mapping[str, Any]:
    path = inputs / ISSUANCE_MIX_FILE
    if not path.exists():
        raise RunnerError(f"required compiled issuance mix assumptions are missing: {path.name}")
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"compiled issuance mix assumptions are unreadable: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise RunnerError("compiled issuance mix assumptions must be a JSON object")
    if payload.get("schema_version") != "tdcsim_cbo_issuance_mix_assumptions_v1":
        raise RunnerError("compiled issuance mix assumptions has an unsupported schema_version")
    mode = payload.get("mode")
    expected_status = {
        "default_tdcsim_cbo_runner_profile": "configured_default",
        "replace_shares": "scenario_override",
    }.get(mode)
    if expected_status is None:
        raise RunnerError(f"unsupported compiled issuance mix mode: {mode!r}")
    status = _assumption_status(payload.get("selection_status"), label="issuance selection_status")
    if status != expected_status:
        raise RunnerError(
            f"compiled issuance mix mode {mode!r} requires selection_status={expected_status!r}"
        )
    return payload


def _assumption_status(value: Any, *, label: str) -> str:
    if value not in {"configured_default", "scenario_override"}:
        raise RunnerError(f"{label} must be configured_default or scenario_override")
    return str(value)


def _adapter_assumption_statuses(inputs: Path) -> dict[str, str]:
    runtime = _runtime_assumptions(inputs)
    issuance = _issuance_assumptions(inputs)
    return {
        "fiscal_incidence_policy_status": _assumption_status(
            runtime["fiscal_incidence_policy_status"],
            label="fiscal_incidence_policy_status",
        ),
        "mmf_deposit_pass_through_status": _assumption_status(
            runtime["mmf_deposit_pass_through_status"],
            label="mmf_deposit_pass_through_status",
        ),
        "issuance_profile_status": _assumption_status(
            issuance["selection_status"],
            label="issuance selection_status",
        ),
    }


def _fiscal_incidence_policy(inputs: Path) -> dict[str, Any]:
    path = inputs / "tdcsim_fiscal_incidence_policy.csv"
    try:
        frame = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise RunnerError(f"required fiscal incidence policy is missing: {path.name}") from exc
    if frame.empty:
        raise RunnerError("compiled fiscal incidence policy is empty")
    required = {
        "policy_id",
        "policy_mode",
        "incidence_basis",
        "du_share",
        "ru_share",
        "foreign_share",
        "other_share",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RunnerError(f"compiled fiscal incidence policy is missing columns: {missing}")
    policy_ids = frame["policy_id"].astype("string")
    if policy_ids.isna().any() or policy_ids.str.strip().eq("").any():
        raise RunnerError("compiled fiscal incidence policy contains a missing policy_id")
    assumptions = _runtime_assumptions(inputs)
    selected_id = assumptions["fiscal_incidence_policy_id"]
    if not isinstance(selected_id, str) or not selected_id.strip():
        raise RunnerError("fiscal_incidence_policy_id must be a nonempty string")
    selected_id = selected_id.strip()
    selected = frame.loc[policy_ids.eq(selected_id)]
    if selected.empty:
        raise RunnerError(f"unknown fiscal_incidence_policy_id: {selected_id}")
    if len(selected) != 1:
        raise RunnerError(f"fiscal_incidence_policy_id must select exactly one row: {selected_id}")
    row = selected.iloc[0]
    mode = str(row["policy_mode"])
    basis = str(row["incidence_basis"])
    if mode != "explicit_scenario_assumption":
        raise RunnerError(f"unsupported fiscal incidence policy_mode: {mode}")
    if basis != "signed_net_primary_proxy":
        raise RunnerError(f"unsupported fiscal incidence basis: {basis}")
    shares = {
        key: _finite_policy_share(row[key], label=f"{selected_id}.{key}")
        for key in ("du_share", "ru_share", "foreign_share", "other_share")
    }
    total = sum(shares.values())
    if abs(total - 1.0) > 1e-9:
        raise RunnerError(f"fiscal incidence policy shares must sum to 1.0, got {total}")
    return {
        "policy_id": selected_id,
        "mode": mode,
        "incidence_basis": basis,
        **shares,
    }


def _finite_policy_share(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RunnerError(f"fiscal incidence policy {label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise RunnerError(f"fiscal incidence policy {label} must be between 0.0 and 1.0")
    return number


def _fed_target_active(path: Path) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    if frame.empty or "cbo_fed_holdings_target_bil" not in frame.columns:
        return False
    return True


def _assert_no_cb_auction_preferences(prefs: Mapping[str, Mapping[str, float]]) -> None:
    cb = prefs.get("CB", {})
    nonzero = {key: value for key, value in cb.items() if abs(float(value)) > 1e-12}
    if nonzero:
        raise RunnerError(f"CB auction preferences must be zero when a CBO Fed stock target path is active: {nonzero}")


def _opening_tips_reference_cpi(portfolio: pd.DataFrame) -> float:
    if "ReferenceCPI_Issue" not in portfolio.columns:
        return 100.0
    values = pd.to_numeric(portfolio["ReferenceCPI_Issue"], errors="coerce")
    positive = values[values > 0.0]
    return float(positive.iloc[0]) if not positive.empty else 100.0


def _max_abs(frame: pd.DataFrame, column: str) -> float:
    values = _numeric_column(frame, column)
    if values.empty:
        raise RunnerError(f"simulation results contain no {column} boundary evidence")
    return float(values.abs().max())


def _sum_abs(frame: pd.DataFrame, column: str) -> float:
    values = _numeric_column(frame, column)
    if values.empty:
        raise RunnerError(f"simulation results contain no {column} boundary evidence")
    return float(values.abs().sum())


def _copy_scenario_referenced_files(spec: CboScenarioSpec, run_root: Path) -> list[dict[str, Any]]:
    refs = _scenario_file_refs(spec.data)
    if not refs:
        return []
    if spec.path is None:
        raise RunnerError("file-backed scenario runs require a scenario file path")
    source_root = spec.path.parent.resolve()
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for rel, expected_sha in sorted(refs.items()):
        if _is_reserved_run_path(rel):
            raise RunnerError(f"scenario referenced file uses a reserved run-package path: {rel}")
        source = (source_root / rel).resolve()
        if source_root not in source.parents and source != source_root:
            raise RunnerError(f"scenario referenced file escapes scenario directory: {rel}")
        if not source.exists():
            raise RunnerError(f"scenario referenced file is missing: {rel}")
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha:
            raise RunnerError(f"scenario referenced file SHA-256 mismatch: {rel}")
        prior = seen.get(rel)
        if prior is not None and prior != expected_sha:
            raise RunnerError(f"scenario referenced file has conflicting hashes: {rel}")
        seen[rel] = expected_sha
        dest = run_root / rel
        if dest == run_root / "scenario.json":
            raise RunnerError("scenario referenced file conflicts with packaged scenario.json")
        if dest.resolve() == spec.path:
            raise RunnerError(f"scenario referenced file conflicts with scenario file: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        records.append(_artifact_record(run_root, dest, logical_name=rel))
    return records


def _is_reserved_run_path(rel: str) -> bool:
    parts = Path(rel).parts
    return rel in {"scenario.json", "tdcsim_cbo_run_manifest.json"} or (bool(parts) and parts[0] in {"compile", "outputs"})


def _scenario_file_refs(value: Any) -> dict[str, str]:
    refs: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if "relative_path" in node and "sha256" in node:
                rel = str(node["relative_path"])
                sha = str(node["sha256"])
                if rel in refs and refs[rel] != sha:
                    raise RunnerError(f"scenario file reference has conflicting hashes: {rel}")
                refs[rel] = sha
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return refs


def _artifact_record(root: Path, path: Path, *, logical_name: str | None = None) -> dict[str, Any]:
    rel = path.relative_to(root).as_posix()
    return {
        "logical_name": logical_name or rel,
        "relative_path": rel,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": _media_type(rel),
    }


def _media_type(path: str) -> str:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        return "text/csv"
    if path.endswith(".sqlite"):
        return "application/vnd.sqlite3"
    if path.endswith(".whl"):
        return "application/zip"
    return "application/octet-stream"


def _code_environment(baseline: CboBaselinePackage, run_root: Path) -> dict[str, Any]:
    dist = distribution_identity()
    wheel_artifact = _copy_release_wheel(run_root)
    wheel_sha256 = str(wheel_artifact["sha256"]) if wheel_artifact else ""
    env_commit = os.environ.get("TDCSIM_CBO_CODE_COMMIT_SHA", "")
    git_commit = _module_git_value(["rev-parse", "HEAD"], default="")
    dirty_state = _dirty_state(wheel_artifact is not None)
    if wheel_artifact is not None and not _is_commit_sha(env_commit):
        raise RunnerError("release wheel runs require TDCSIM_CBO_CODE_COMMIT_SHA")
    if wheel_artifact is not None and dirty_state:
        raise RunnerError("release wheel runs require TDCSIM_CBO_DIRTY_STATE=false")
    return {
        "code_commit_sha": env_commit or git_commit or "0" * 40,
        "dirty_state": dirty_state,
        "requirements_lock_sha256": str(
            os.environ.get("TDCSIM_CBO_REQUIREMENTS_LOCK_SHA256")
            or baseline.attestation.data.get("requirements_lock_sha256")
            or "0" * 64
        ),
        "python_version": _python_version(),
        "runner_version": "tdcsim_cbo_runner_v1",
        "verifier_version": "tdcsim_cbo_verifier_v1",
        "runner_source_sha256": sha256_file(Path(__file__)),
        "sim_engine_source_sha256": sha256_file(Path(run_simulation.__code__.co_filename)),
        "package_name": dist["name"],
        "package_version": dist["version"],
        "distribution_file_digest": dist["file_digest"],
        "wheel_sha256": wheel_sha256,
        "wheel_artifact": wheel_artifact,
        "runtime_identity_source": dist["identity_source"],
    }


def _copy_release_wheel(run_root: Path) -> dict[str, Any] | None:
    wheel_sha = os.environ.get("TDCSIM_CBO_WHEEL_SHA256", "")
    wheel_path_raw = os.environ.get("TDCSIM_CBO_WHEEL_PATH", "")
    if wheel_sha and not wheel_path_raw:
        raise RunnerError("TDCSIM_CBO_WHEEL_SHA256 requires TDCSIM_CBO_WHEEL_PATH so the run package retains wheel bytes")
    if not wheel_path_raw:
        return None
    source = Path(wheel_path_raw).expanduser().resolve()
    if not source.is_file():
        raise RunnerError(f"TDCSIM_CBO_WHEEL_PATH does not point to a file: {source}")
    actual_sha = sha256_file(source)
    if wheel_sha and actual_sha != wheel_sha:
        raise RunnerError("TDCSIM_CBO_WHEEL_SHA256 does not match TDCSIM_CBO_WHEEL_PATH")
    runtime_dir = run_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    dest = runtime_dir / source.name
    shutil.copy2(source, dest)
    return _artifact_record(run_root, dest, logical_name=source.name)


def _python_version() -> str:
    import platform

    return platform.python_version()


def _dirty_state(is_release_wheel_run: bool) -> bool:
    raw = os.environ.get("TDCSIM_CBO_DIRTY_STATE")
    if raw is not None:
        lowered = raw.strip().lower()
        if lowered in {"false", "0", "no"}:
            return False
        if lowered in {"true", "1", "yes"}:
            return True
        raise RunnerError("TDCSIM_CBO_DIRTY_STATE must be true or false")
    if is_release_wheel_run:
        return True
    return bool(_module_git_value(["status", "--short"], default=""))


def _is_commit_sha(value: str) -> bool:
    return len(value) == 40 and all(char in "0123456789abcdef" for char in value)


def _module_git_value(args: list[str], *, default: str) -> str:
    git_root = _module_git_root()
    if git_root is None:
        return default
    try:
        return subprocess.run(
            ["git", *args],
            cwd=git_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return default


def _module_git_root() -> Path | None:
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return None
    return Path(root) if root else None


__all__ = ["CboScenarioRun", "RunnerError", "build_runtime_params", "run_cbo_scenario", "validate_run_boundaries"]
