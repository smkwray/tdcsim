#!/usr/bin/env python3
"""Write ready-to-run example CBO scenario overlays for downstream projects."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec
from tdcsim_cbo._json import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-date", help="Optional daily simulation start date, YYYY-MM-DD")
    parser.add_argument("--end-date", help="Optional daily simulation end date, YYYY-MM-DD")
    args = parser.parse_args(argv)

    baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulation = _simulation(args.start_date, args.end_date)
    for filename, scenario in example_scenarios(baseline, simulation=simulation).items():
        spec = CboScenarioSpec.from_mapping(scenario)
        spec.assert_baseline_matches(baseline)
        write_json(args.output_dir / filename, spec.data)
        print(args.output_dir / filename)
    return 0


def example_scenarios(
    baseline: CboBaselinePackage,
    *,
    simulation: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return public example scenarios bound to a concrete baseline package."""

    return {
        "00_baseline_noop.json": _scenario(
            baseline,
            scenario_id="cbo_baseline_noop_v1",
            label="Default CBO baseline passthrough",
            simulation=simulation,
            coupling={
                "frn_benchmark": "independent_explicit_path",
                "tips_real_yield": "independent_explicit_path",
                "operating_cash_inflation": "baseline_cpi",
            },
            overrides={},
        ),
        "01_rates_inflation_frn_tips.json": _scenario(
            baseline,
            scenario_id="cbo_rates_inflation_frn_tips_v1",
            label="Rates, inflation, FRN, and TIPS stress",
            simulation=simulation,
            coupling={
                "frn_benchmark": "derive_from_scenario_nominal_curve",
                "tips_real_yield": "independent_explicit_path",
                "operating_cash_inflation": "scenario_cpi",
            },
            overrides={
                "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 25},
                "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 5},
                "inflation_cpi": {
                    "mode": "annualized_inflation_shift_bp",
                    "shock_bp": 25,
                    "terminal_rule": "carry_last_scenario_growth",
                },
                "tips_real_yield": {"mode": "parallel_bp", "shock_bp": 10},
                "operating_cash": {"mode": "constant_real"},
            },
        ),
        "02_issuance_maturity_mix.json": _scenario(
            baseline,
            scenario_id="cbo_issuance_maturity_mix_v1",
            label="Issuance mix and maturity stress",
            simulation=simulation,
            coupling=_baseline_coupling(),
            overrides={
                "issuance_mix": {
                    "mode": "replace_shares",
                    "tips_share": 0.08,
                    "frn_share": 0.04,
                    "fixed_remainder_shares": {"bills": 0.25, "notes": 0.50, "bonds": 0.25},
                    "maturity_distributions": {
                        "bills": [{"maturity_years": 0.5, "share": 1.0}],
                        "notes": [{"maturity_years": 5.0, "share": 0.65}, {"maturity_years": 10.0, "share": 0.35}],
                        "bonds": [{"maturity_years": 20.0, "share": 0.40}, {"maturity_years": 30.0, "share": 0.60}],
                        "tips": [{"maturity_years": 10.0, "share": 0.70}, {"maturity_years": 30.0, "share": 0.30}],
                        "frn": [{"maturity_years": 2.0, "share": 1.0}],
                    },
                    "negative_issuance_action": "retire_shortest_public_marketable",
                }
            },
        ),
        "03_sector_holders.json": _scenario(
            baseline,
            scenario_id="cbo_sector_holders_v1",
            label="Dated sector holder preference stress",
            simulation=simulation,
            coupling=_baseline_coupling(),
            overrides={
                "holder_preferences": {
                    "mode": "dated_static_shares",
                    "rows": _holder_rows(effective_date=_holder_effective_date(simulation)),
                }
            },
        ),
        "04_fiscal_fed_cash.json": _scenario(
            baseline,
            scenario_id="cbo_fiscal_fed_cash_v1",
            label="Fiscal, Fed stock target, and cash stress",
            simulation=simulation,
            coupling=_baseline_coupling(),
            overrides={
                "primary_deficit": {"mode": "scale_path", "scale": 1.01},
                "fed_holdings": {"mode": "scale_path", "scale": 1.0},
                "operating_cash": {"mode": "constant_nominal"},
                "cash_reconciliation": {"mode": "zero", "funding_effect": "none"},
                "fiscal_incidence": {
                    "mode": "static_shares",
                    "domestic_ultimate_share": 0.50,
                    "rest_of_world_share": 0.25,
                    "foreign_official_share": 0.25,
                    "other_share": 0.0,
                },
            },
        ),
    }


def _scenario(
    baseline: CboBaselinePackage,
    *,
    scenario_id: str,
    label: str,
    simulation: dict[str, str] | None,
    coupling: dict[str, str],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": scenario_id,
        "title": label,
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "provenance": {"kind": "user_stress_assumption", "label": label},
        "coupling": {**coupling, "primary_deficit_to_debt_target": "independent_no_plug"},
        "overrides": overrides,
        "output": {"profile": "compact", "compression": "none", "catalog_sqlite": True},
    }
    if simulation:
        scenario["simulation"] = simulation
    return scenario


def _baseline_coupling() -> dict[str, str]:
    return {
        "frn_benchmark": "independent_explicit_path",
        "tips_real_yield": "independent_explicit_path",
        "operating_cash_inflation": "baseline_cpi",
    }


def _holder_rows(*, effective_date: str | None = None) -> list[dict[str, Any]]:
    shares = {
        "Banks": 0.10,
        "CB": 0.0,
        "Foreign": 0.20,
        "Private": 0.70,
        "TrustFunds": 0.0,
        "FedInternal": 0.0,
    }
    rows = [{"security_type": security_type, "shares": shares} for security_type in ("bills", "notes", "bonds", "tips", "frn")]
    if effective_date:
        for row in rows:
            row["effective_date"] = effective_date
    return rows


def _holder_effective_date(simulation: dict[str, str] | None) -> str:
    if simulation:
        return simulation["start_date"]
    return "2030-01-01"


def _simulation(start_date: str | None, end_date: str | None) -> dict[str, str] | None:
    if bool(start_date) != bool(end_date):
        raise ValueError("--start-date and --end-date must be supplied together")
    if not start_date or not end_date:
        return None
    return {"frequency": "daily", "start_date": start_date, "end_date": end_date}


if __name__ == "__main__":
    raise SystemExit(main())
