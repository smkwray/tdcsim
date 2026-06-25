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

from forecast_paths import compiled_forecast_input_paths
from sim_engine import run_simulation
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    HOLDER_TYPES,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKETS,
)
from tdc_validation import validate_events

from ._json import read_json, sha256_file, write_json
from ._schema import validate_schema
from .baseline import CboBaselinePackage
from .compiler import CboCompiledScenario, CboScenarioCompiler, HOLDER_PREFERENCE_EVENTS_FILE, RUNTIME_ASSUMPTIONS_FILE
from .contract import CboScenarioSpec
from .manifest import build_run_manifest
from .output import hash_output_tree, write_scenario_outputs
from .runtime_identity import distribution_identity


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
    engine_scenario_id = _compiled_scenario_id(inputs)
    results, final_portfolio = run_simulation(
        params,
        start,
        end,
        freq="D",
        scenario_name=engine_scenario_id,
    )
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
    boundaries = validate_run_boundaries(results, inputs)
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
    operating_cash = pd.read_csv(inputs / "tdcsim_operating_cash_path.csv")
    base_tga = float(operating_cash.iloc[0].get("operating_cash_target_bil", 0.0))
    holder_preferences = _holder_preferences(inputs / "tdcsim_holder_profile_assumptions.csv")
    holder_events = _holder_preference_events(inputs / HOLDER_PREFERENCE_EVENTS_FILE)
    if _fed_target_active(inputs / "tdcsim_fed_holdings_path.csv"):
        _assert_no_cb_auction_preferences(holder_preferences)
    return {
        "initial_values": {"reserves": 3000.0, "tdc_level": 0.0, "tga": base_tga},
        "tga_params": {"target_balance": base_tga, "floor": -1e15},
        "fiscal_params": {
            "initial_weekly_spending": 0.0,
            "initial_weekly_taxes": 0.0,
            "spending_growth_qtr": 0.0,
            "tax_growth_qtr": 0.0,
        },
        "other_flows": {"reserve_transfer": 0.0, "cb_net_expense": 0.0, "money_minting_transfers": 0.0},
        "treasury_issuance_profile": _issuance_profile(inputs),
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
        },
        "baseline_input_paths": compiled_forecast_input_paths(inputs),
        "data_vintage": {
            "actuals_available_as_of": actuals_available_as_of or _max_available_date(inputs) or "1900-01-01",
            "allow_lookahead": False,
        },
        "fiscal_incidence_policy": _fiscal_incidence_policy(inputs / "tdcsim_fiscal_incidence_policy.csv"),
        "budget_interest": {"cbo_comparison_role": "nonbinding_validation_check"},
    }


def validate_run_boundaries(results: pd.DataFrame, inputs_dir: str | Path) -> dict[str, Any]:
    """Compute hard boundary checks for CBO scenario runs."""

    inputs = Path(inputs_dir)
    residual = pd.read_csv(inputs / "tdcsim_cash_reconciliation_residual.csv")
    residual_flags = {}
    for col in (
        "affects_primary_deficit",
        "affects_net_interest",
        "affects_total_deficit",
        "affects_debt_target",
        "affects_issuance_size",
        "affects_tdc_fiscal_flow",
    ):
        if col in residual.columns:
            residual_flags[col] = sorted(str(value) for value in residual[col].dropna().unique())
    fed_auction_share_max = _max_abs(results, "CBOFedAuctionShare")
    fed_auction_face_max = _max_abs(results, "CBOFedAuctionRolloverAddons")
    fed_auction_face_sum = _sum_abs(results, "CBOFedAuctionRolloverAddons")
    fed_boundary_pass = fed_auction_share_max <= 1e-12 and fed_auction_face_max <= 1e-12
    return {
        "cash_residual_nonfunding_flags": residual_flags,
        "cash_residual_affects_issuance_size": residual_flags.get("affects_issuance_size", []),
        "max_abs_fed_auction_share": fed_auction_share_max,
        "max_abs_fed_auction_face": fed_auction_face_max,
        "sum_abs_fed_auction_face": fed_auction_face_sum,
        "fed_target_holder_allocation_only": fed_boundary_pass,
        "net_interest_role": "diagnostic_nonbinding",
        "remittance_deferred_asset_status": "unsupported_in_cbo_scenario_lane",
    }


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
    frame = pd.read_csv(path)
    for col in BOND_PORTFOLIO_COLS:
        if col not in frame.columns:
            frame[col] = pd.NA
    for col in ("IssueDate", "MaturityDate", "DatedDate", "OriginalDatedDate", "FirstInterestPaymentDate", "LastAccrualDate"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame[BOND_PORTFOLIO_COLS].astype(PORTFOLIO_DTYPES, errors="ignore")


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
    path = inputs / "tdcsim_issuance_mix_assumptions.json"
    if not path.exists():
        return _default_issuance_profile()
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise RunnerError("compiled issuance mix assumptions must be a JSON object")
    if payload.get("mode") != "replace_shares":
        return _default_issuance_profile()
    shares = payload["security_shares"]
    maturity = payload["maturity_distributions"]
    fixed_total = float(shares["bills"]) + float(shares["notes"]) + float(shares["bonds"])
    fixed = {
        "bills": 0.0 if fixed_total == 0 else float(shares["bills"]) / fixed_total,
        "notes": 0.0 if fixed_total == 0 else float(shares["notes"]) / fixed_total,
        "bonds": 0.0 if fixed_total == 0 else float(shares["bonds"]) / fixed_total,
    }
    return {
        "bills": _fixed_profile(float(fixed["bills"]), maturity["bills"], cutoff=1.0),
        "notes": _fixed_profile(float(fixed["notes"]), maturity["notes"], cutoff=10.0),
        "bonds": _fixed_profile(float(fixed["bonds"]), maturity["bonds"], cutoff=999.0),
        "TIPS": _special_profile(float(shares["tips"]), maturity["tips"]),
        "FRN": _special_profile(float(shares["frn"]), maturity["frn"]),
        "NonMarketable": {"target_percentage": 0.0, "maturities": [30.0], "maturity_distribution": [1.0]},
        "remainder_maturity_years": 1.0,
    }


def _default_issuance_profile() -> dict[str, Any]:
    return {
        "bills": _fixed_profile(0.25, [{"maturity_years": 0.5, "share": 1.0}], cutoff=1.0),
        "notes": _fixed_profile(0.55, [{"maturity_years": 5.0, "share": 1.0}], cutoff=10.0),
        "bonds": _fixed_profile(0.20, [{"maturity_years": 20.0, "share": 1.0}], cutoff=999.0),
        "TIPS": _special_profile(0.06, [{"maturity_years": 10.0, "share": 1.0}]),
        "FRN": _special_profile(0.04, [{"maturity_years": 2.0, "share": 1.0}]),
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


def _negative_issuance_action(inputs: Path) -> str:
    path = inputs / "tdcsim_issuance_mix_assumptions.json"
    if path.exists():
        payload = read_json(path)
        if isinstance(payload, Mapping) and payload.get("negative_issuance_action"):
            return str(payload["negative_issuance_action"])
    return "error"


def _mmf_deposit_pass_through(inputs: Path) -> float:
    path = inputs / RUNTIME_ASSUMPTIONS_FILE
    if not path.exists():
        return 0.97
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise RunnerError("compiled runtime assumptions must be a JSON object")
    try:
        value = float(payload.get("mmf_deposit_pass_through", 0.97))
    except (TypeError, ValueError) as exc:
        raise RunnerError("mmf_deposit_pass_through must be numeric") from exc
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise RunnerError("mmf_deposit_pass_through must be between 0.0 and 1.0")
    return value


def _fiscal_incidence_policy(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise RunnerError("compiled fiscal incidence policy is empty")
    row = frame.iloc[0]
    return {
        "mode": "explicit_scenario_assumption",
        "incidence_basis": "signed_net_primary_proxy",
        "du_share": float(row.get("du_share", 0.0) or 0.0),
        "ru_share": float(row.get("ru_share", 0.0) or 0.0),
        "foreign_share": float(row.get("foreign_share", 0.0) or 0.0),
        "other_share": float(row.get("other_share", 0.0) or 0.0),
    }


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
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).abs().max())


def _sum_abs(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).abs().sum())


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
