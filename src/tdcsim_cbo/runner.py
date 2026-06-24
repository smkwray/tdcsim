"""Run compiled CBO scenario inputs through TDCSIM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_paths import compiled_forecast_input_paths
from sim_engine import run_simulation
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES

from ._json import write_json
from .baseline import CboBaselinePackage
from .compiler import CboCompiledScenario, CboScenarioCompiler
from .contract import CboScenarioSpec
from .manifest import build_run_manifest
from .output import hash_output_tree, write_scenario_outputs


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
    inputs = compiled.forecast_inputs_dir
    start, end = _simulation_dates(spec, inputs)
    params = build_runtime_params(inputs, spec)
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
    outputs = write_scenario_outputs(
        results,
        final_portfolio,
        out / "outputs",
        profile=profile,
        compression=compression,
        catalog_sqlite=bool(output_cfg.get("catalog_sqlite", False)),
    )
    boundaries = validate_run_boundaries(results, inputs)
    run_manifest = build_run_manifest(
        scenario_id=spec.scenario_id,
        scenario_sha256=spec.canonical_sha256(),
        compiled=compiled,
        compiled_manifest_relpath=compiled.manifest_path.relative_to(out).as_posix(),
        start_date=start,
        end_date=end,
        outputs=outputs,
        output_hashes=hash_output_tree(out / "outputs"),
        boundary_checks=boundaries,
    )
    manifest_path = out / "tdcsim_cbo_run_manifest.json"
    write_json(manifest_path, run_manifest)
    result_path = out / "outputs" / f"results_{profile}{'.csv.gz' if compression == 'gzip' else '.csv'}"
    return CboScenarioRun(
        output_dir=out,
        compiled=compiled,
        results_path=result_path,
        manifest_path=manifest_path,
        run_manifest=run_manifest,
    )


def build_runtime_params(inputs_dir: str | Path, spec: CboScenarioSpec) -> dict[str, Any]:
    """Build simulator params from compiled forecast inputs and scenario config."""

    inputs = Path(inputs_dir)
    initial_portfolio = _load_opening_portfolio(inputs / "tdcsim_opening_portfolio.csv")
    operating_cash = pd.read_csv(inputs / "tdcsim_operating_cash_path.csv")
    base_tga = float(operating_cash.iloc[0].get("operating_cash_target_bil", 0.0))
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
        "treasury_issuance_profile": _issuance_profile(spec),
        "yield_curve": {
            "use_static": True,
            "years": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            "rates": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        },
        "yield_curve_surface": {"file": str(inputs / "tdcsim_yield_curve_surface.csv"), "interpolation_method": "pchip", "floor_zero": False},
        "sector_preferences": _holder_preferences(inputs / "tdcsim_holder_profile_assumptions.csv"),
        "private_mmf_split": {"bills": 0.25, "notes": 0.10, "bonds": 0.05, "tips": 0.05, "frn": 0.20},
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
        "events": [],
        "funding_rule": {
            "mode": "cbo_public_debt_target",
            "target_enforcement": "every_period",
            "negative_required_issuance_action": _negative_issuance_action(spec),
            "target_tolerance_bil": 0.000001,
        },
        "baseline_input_paths": compiled_forecast_input_paths(inputs),
        "data_vintage": {"actuals_available_as_of": "9999-12-31", "allow_lookahead": False},
        "fiscal_incidence_policy": {
            "mode": "explicit_scenario_assumption",
            "incidence_basis": "signed_net_primary_proxy",
            "du_share": 0.99,
            "ru_share": 0.01,
            "foreign_share": 0.0,
            "other_share": 0.0,
        },
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
    fed_auction_max = _max_abs(results, "CBOFedAuctionShare")
    return {
        "cash_residual_nonfunding_flags": residual_flags,
        "cash_residual_affects_issuance_size": residual_flags.get("affects_issuance_size", []),
        "max_abs_fed_auction_share": fed_auction_max,
        "fed_target_holder_allocation_only": fed_auction_max <= 1e-12,
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
        prefs[holder] = {
            "bills_pct": float(row.get("bills_pct", 0.0) or 0.0),
            "notes_pct": float(row.get("notes_pct", 0.0) or 0.0),
            "bonds_pct": float(row.get("bonds_pct", 0.0) or 0.0),
            "tips_pct": float(row.get("tips_pct", 0.0) or 0.0),
            "frn_pct": float(row.get("frn_pct", 0.0) or 0.0),
        }
    for holder in ("Banks", "CB", "Foreign", "Private", "FedInternal", "TrustFunds"):
        prefs.setdefault(holder, {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0, "frn_pct": 0.0})
    return prefs


def _issuance_profile(spec: CboScenarioSpec) -> dict[str, Any]:
    override = _override(spec, "issuance_mix")
    if not override:
        return _default_issuance_profile()
    tips_share = float(override["tips_share"])
    frn_share = float(override["frn_share"])
    fixed = override["fixed_remainder_shares"]
    maturity = override["maturity_distributions"]
    return {
        "bills": _fixed_profile(float(fixed["bills"]), maturity["bills"], cutoff=1.0),
        "notes": _fixed_profile(float(fixed["notes"]), maturity["notes"], cutoff=10.0),
        "bonds": _fixed_profile(float(fixed["bonds"]), maturity["bonds"], cutoff=999.0),
        "TIPS": _special_profile(tips_share, maturity["tips"]),
        "FRN": _special_profile(frn_share, maturity["frn"]),
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


def _negative_issuance_action(spec: CboScenarioSpec) -> str:
    override = _override(spec, "issuance_mix")
    if override:
        return str(override.get("negative_issuance_action") or "retire_shortest_public_marketable")
    return "retire_shortest_public_marketable"


def _override(spec: CboScenarioSpec, key: str) -> Mapping[str, Any] | None:
    overrides = spec.data.get("overrides", {})
    if isinstance(overrides, Mapping) and isinstance(overrides.get(key), Mapping):
        return overrides[key]
    return None


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


__all__ = ["CboScenarioRun", "RunnerError", "build_runtime_params", "run_cbo_scenario", "validate_run_boundaries"]
