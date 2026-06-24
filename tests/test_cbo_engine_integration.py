from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from budget_interest import build_net_interest_diagnostic
from cbo_yield_curve_surface import build_yield_curve_surface_rows
from cbo_policy_bundle import (
    allocate_signed_primary_flow,
    build_fiscal_incidence_policy_rows,
    build_net_interest_bridge_rows,
    validate_fiscal_incidence_policy_rows,
)
from debt_targeting import calculate_controlled_marketable_target
from forecast_bundle_builders import (
    build_cash_reconciliation_residual_rows,
    build_debt_stock_path_rows,
    build_fed_holdings_path_rows,
    build_operating_cash_path_rows,
    build_primary_deficit_path_rows,
)
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import (
    load_cash_reconciliation_residual,
    load_debt_stock_path,
    load_fed_holdings_path,
    load_fiscal_incidence_policy,
    load_frn_rate_path,
    load_net_interest_bridge,
    load_operating_cash_path,
    load_primary_deficit_path,
)
from yield_curve_path import load_yield_curve_surface as load_runtime_yield_curve_surface
from funding_plan import build_funding_plan
from sim_engine import run_simulation
from simulation_calendar import SimulationPeriod, build_simulation_calendar
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES


def _bill_yield_for_price(price_ratio: float) -> float:
    return (1.0 / price_ratio) - 1.0


@dataclass(frozen=True)
class CboContractRun:
    controlled_debt_target: float
    required_face_issuance: float
    post_issuance_controlled_debt: float
    target_error: float
    auction_proceeds: float
    cash_reconciliation_residual: float
    operating_cash_target: float
    holder_allocations: dict[str, float]
    aggregate_signed_primary_flow: float
    net_interest_diagnostic: dict[str, Any]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    write_forecast_rows_csv(path, rows)
    return path


def _single_period(period_start: str = "2026-09-20", period_end: str = "2026-09-30") -> list[SimulationPeriod]:
    start = pd.Timestamp(period_start)
    end = pd.Timestamp(period_end)
    return [
        SimulationPeriod(
            period_start=start.date(),
            period_end=end.date(),
            frequency="weekly",
            day_count=(end - start).days,
            day_count_weight=1.0,
            is_control_date=True,
            is_partial_period=True,
        )
    ]


def _empty_portfolio_df() -> pd.DataFrame:
    return pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)


def _opening_controlled_portfolio(face_value: float = 1_000.0) -> pd.DataFrame:
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update(
        {
            "BondID": 1,
            "SecurityType": "Fixed",
            "IssueDate": pd.Timestamp("2025-09-30"),
            "MaturityDate": pd.Timestamp("2027-09-30"),
            "OriginalMaturityYears": 2.0,
            "FaceValue": face_value,
            "CouponRate": 0.04,
            "HolderType": "Private",
            "HolderSubBucket": "DomesticNonbank",
            "Status": "Active",
            "MaturityCategory": "notes",
            "OriginalPrincipal": face_value,
            "AdjustedPrincipal": face_value,
            "ReferenceCPI_Issue": 0.0,
            "IndexRatio": 1.0,
            "FixedSpread": 0.0,
            "AccruedInterest_FRN": 0.0,
            "BenchmarkRate_FRN": 0.0,
            "LastAccrualDate": pd.NaT,
            "IssuePriceRatio": 1.0,
            "IssueProceeds": face_value,
            "IssueYieldAtIssue": 0.04,
        }
    )
    return pd.DataFrame([row], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def _opening_tips_portfolio(face_value: float = 100.0, reference_cpi: float = 100.0) -> pd.DataFrame:
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update(
        {
            "BondID": 1,
            "SecurityType": "TIPS",
            "IssueDate": pd.Timestamp("2025-09-30"),
            "MaturityDate": pd.Timestamp("2027-09-30"),
            "OriginalMaturityYears": 2.0,
            "FaceValue": face_value,
            "CouponRate": 0.01,
            "HolderType": "Private",
            "HolderSubBucket": "domestic_nonbank_deposit_funded",
            "Status": "Active",
            "MaturityCategory": "tips",
            "OriginalPrincipal": face_value,
            "AdjustedPrincipal": face_value,
            "ReferenceCPI_Issue": reference_cpi,
            "IndexRatio": 1.0,
            "FixedSpread": 0.0,
            "AccruedInterest_FRN": 0.0,
            "BenchmarkRate_FRN": 0.0,
            "LastAccrualDate": pd.NaT,
            "IssuePriceRatio": 1.0,
            "IssueProceeds": face_value,
            "IssueYieldAtIssue": 0.01,
        }
    )
    return pd.DataFrame([row], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def _opening_frn_portfolio(
    *,
    face_value: float = 1_000.0,
    accrued_interest: float = 1.0,
    first_interest_payment_date: str = "2026-09-30",
    maturity_date: str = "2027-09-30",
) -> pd.DataFrame:
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update(
        {
            "BondID": 1,
            "SecurityType": "FRN",
            "IssueDate": pd.Timestamp("2026-06-30"),
            "MaturityDate": pd.Timestamp(maturity_date),
            "DatedDate": pd.Timestamp("2026-06-30"),
            "OriginalDatedDate": pd.Timestamp("2026-06-30"),
            "FirstInterestPaymentDate": pd.Timestamp(first_interest_payment_date),
            "InterestPaymentFrequency": 4.0,
            "OriginalMaturityYears": 2.0,
            "FaceValue": face_value,
            "CouponRate": 0.0,
            "HolderType": "Private",
            "HolderSubBucket": "DomesticNonbank",
            "Status": "Active",
            "MaturityCategory": "frn",
            "OriginalPrincipal": face_value,
            "AdjustedPrincipal": face_value,
            "ReferenceCPI_Issue": 0.0,
            "IndexRatio": 1.0,
            "FixedSpread": 0.001,
            "AccruedInterest_FRN": accrued_interest,
            "BenchmarkRate_FRN": 0.04,
            "LastAccrualDate": pd.Timestamp("2026-09-20"),
            "IssuePriceRatio": 1.0,
            "IssueProceeds": face_value,
            "IssueYieldAtIssue": 0.04,
        }
    )
    return pd.DataFrame([row], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def _base_issuance_profile() -> dict[str, Any]:
    return {
        "bills": {
            "category_cutoff_years": 1.0,
            "target_percentage_of_remainder": 1.0,
            "maturities": [1.0],
            "maturity_distribution": [1.0],
        },
        "notes": {
            "category_cutoff_years": 10.0,
            "target_percentage_of_remainder": 0.0,
            "maturities": [2.0],
            "maturity_distribution": [1.0],
        },
        "bonds": {
            "category_cutoff_years": 999.0,
            "target_percentage_of_remainder": 0.0,
            "maturities": [20.0],
            "maturity_distribution": [1.0],
        },
        "remainder_maturity_years": 1.0,
    }


def _holder_prefs(*, private_bills: float = 1.0, banks_bills: float = 0.0) -> dict[str, dict[str, float]]:
    return {
        "Banks": {"bills_pct": banks_bills, "notes_pct": 0.0, "bonds_pct": 0.0},
        "Private": {"bills_pct": private_bills, "notes_pct": 1.0, "bonds_pct": 1.0},
        "CB": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0},
        "Foreign": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0},
        "FedInternal": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0},
        "TrustFunds": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0},
    }


def _minimal_engine_params(
    paths: dict[str, Path],
    *,
    private_bills: float = 1.0,
    banks_bills: float = 0.0,
    opening_controlled_debt_bil: float = 1_000.0,
) -> dict[str, Any]:
    return {
        "initial_values": {"reserves": 1_000.0, "tdc_level": 0.0, "tga": 800.0},
        "tga_params": {"target_balance": 800.0, "floor": 0.0},
        "fiscal_params": {
            "initial_weekly_spending": 0.0,
            "initial_weekly_taxes": 0.0,
            "spending_growth_qtr": 0.0,
            "tax_growth_qtr": 0.0,
        },
        "other_flows": {"reserve_transfer": 0.0, "cb_net_expense": 0.0, "money_minting_transfers": 0.0},
        "treasury_issuance_profile": _base_issuance_profile(),
        "yield_curve": {
            "use_static": True,
            "years": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            "rates": [0.04, 0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        },
        "sector_preferences": _holder_prefs(private_bills=private_bills, banks_bills=banks_bills),
        "simulation_period": {"enable_preference_trading": False},
        "initial_bonds_df": (
            _empty_portfolio_df()
            if opening_controlled_debt_bil <= 0.0
            else _opening_controlled_portfolio(opening_controlled_debt_bil)
        ),
        "events": [],
        "funding_rule": {
            "mode": "cbo_public_debt_target",
            "target_enforcement": "every_period",
            "negative_required_issuance_action": "error",
            "target_tolerance_bil": 0.000001,
        },
        "baseline_input_paths": {key: str(value) for key, value in paths.items()},
        "data_vintage": {"actuals_available_as_of": "2026-09-30", "allow_lookahead": False},
    }


def _minimal_cash_mode_params(paths: dict[str, Path] | None = None) -> dict[str, Any]:
    params = {
        "initial_values": {"reserves": 1_000.0, "tdc_level": 0.0, "tga": 50.0},
        "tga_params": {"target_balance": 100.0, "floor": 0.0},
        "fiscal_params": {
            "initial_weekly_spending": 0.0,
            "initial_weekly_taxes": 0.0,
            "spending_growth_qtr": 0.0,
            "tax_growth_qtr": 0.0,
        },
        "other_flows": {"reserve_transfer": 0.0, "cb_net_expense": 0.0, "money_minting_transfers": 0.0},
        "treasury_issuance_profile": _base_issuance_profile(),
        "yield_curve": {
            "use_static": True,
            "years": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            "rates": [0.04, 0.045, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        },
        "sector_preferences": _holder_prefs(),
        "simulation_period": {"enable_preference_trading": False},
        "initial_bonds_df": _empty_portfolio_df(),
        "events": [],
    }
    if paths is not None:
        params["funding_rule"] = {"mode": "cash_tga_target"}
        params["baseline_input_paths"] = {key: str(value) for key, value in paths.items()}
        params["data_vintage"] = {"actuals_available_as_of": "2026-09-30", "allow_lookahead": False}
        params["fiscal_incidence_policy"] = {
            "mode": "explicit_scenario_assumption",
            "incidence_basis": "signed_net_primary_proxy",
            "du_share": 0.99,
            "ru_share": 0.01,
            "foreign_share": 0.0,
            "other_share": 0.0,
        }
        params["budget_interest"] = {"cbo_comparison_role": "nonbinding_validation_check"}
    return params


def _build_temp_forecast_inputs(
    tmp_path: Path,
    *,
    scenario_id: str = "baseline",
    periods: list[SimulationPeriod] | None = None,
    period_start: str = "2026-09-20",
    period_end: str = "2026-09-30",
    cbo_public_debt_target_bil: float = 1_250.0,
    public_nonmarketable_bil: float = 100.0,
    definition_residual_bil: float = 25.0,
    pre_issuance_controlled_debt_bil: float = 1_000.0,
    base_cash_balance_bil: float = 800.0,
    cash_residual_bil: float = 0.0,
    signed_primary_flow_bil: float = 40.0,
    cbo_net_interest_bil: float = 120.0,
) -> dict[str, Path]:
    periods = periods or _single_period(period_start, period_end)
    debt_rows = build_debt_stock_path_rows(
        scenario_id=scenario_id,
        periods=periods,
        opening_state_date=period_start,
        opening_cbo_federal_debt_held_public_bil=pre_issuance_controlled_debt_bil
        + public_nonmarketable_bil
        + definition_residual_bil,
        cbo_fy_end_public_debt_targets_bil={2026: cbo_public_debt_target_bil},
        public_nonmarketable_treasury_bil=public_nonmarketable_bil,
        non_treasury_and_definition_residual_bil=definition_residual_bil,
        observation_date="2026-09-20",
        available_date="2026-09-20",
    )
    primary_rows = build_primary_deficit_path_rows(
        scenario_id=scenario_id,
        periods=periods,
        primary_deficit_by_fiscal_year_bil={2026: signed_primary_flow_bil},
    )
    operating_cash_rows = build_operating_cash_path_rows(
        scenario_id=scenario_id,
        periods=periods,
        base_date=period_start,
        base_balance_bil=base_cash_balance_bil,
        inflation_scalar=0.0,
        observation_date=period_start,
        available_date=period_start,
    )
    residual_rows = build_cash_reconciliation_residual_rows(
        scenario_id=scenario_id,
        periods=periods,
        cash_reconciliation_residual_bil=cash_residual_bil,
    )
    incidence_rows = build_fiscal_incidence_policy_rows(
        scenario_id=scenario_id,
        signed_net_primary_flow_bil=signed_primary_flow_bil,
    )
    net_interest_rows = build_net_interest_bridge_rows(
        scenario_id=scenario_id,
        fiscal_year=2026,
        source_vintage="cbo_2026_02_baseline",
        cbo_reported_net_interest_bil=cbo_net_interest_bil,
        components=[
            {"component_key": "fixed_coupon_accrual", "amount_bil": 80.0},
            {"component_key": "bill_discount_amortization", "amount_bil": 5.0},
        ],
    )

    return {
        "debt_stock_path_file": _write_csv(tmp_path / "tdcsim_debt_stock_path.csv", debt_rows),
        "primary_deficit_path_file": _write_csv(tmp_path / "tdcsim_primary_deficit_path.csv", primary_rows),
        "operating_cash_path_file": _write_csv(tmp_path / "tdcsim_operating_cash_path.csv", operating_cash_rows),
        "cash_reconciliation_residual_file": _write_csv(
            tmp_path / "tdcsim_cash_reconciliation_residual.csv",
            residual_rows,
        ),
        "fiscal_incidence_policy_file": _write_csv(
            tmp_path / "tdcsim_fiscal_incidence_policy.csv",
            incidence_rows,
        ),
        "net_interest_bridge_file": _write_csv(tmp_path / "tdcsim_net_interest_bridge.csv", net_interest_rows),
    }


def _frn_rate_path_row(
    *,
    period_start: str,
    period_end: str,
    rate_decimal: float,
    scenario_id: str = "baseline",
) -> dict[str, object]:
    return {
        "schema_version": "tdcsim_frn_rate_path_v1",
        "scenario_id": scenario_id,
        "period_start": period_start,
        "period_end": period_end,
        "rate_effective_start": period_start,
        "rate_effective_end": period_end,
        "benchmark_tenor_years": 0.25,
        "auction_high_rate_decimal": rate_decimal,
        "benchmark_rate_decimal": rate_decimal,
        "money_market_yield_decimal": rate_decimal,
        "spread_treatment": "add_security_fixed_spread_decimal",
        "day_count_basis": 360.0,
        "lockout_business_days": 2.0,
        "rate_source_family": "cbo_yield_surface_3m_tbill_anchor",
        "source_role": "scenario_assumption",
        "runtime_role": "hard_target",
        "observation_date": period_start,
        "available_date": period_start,
        "source_status": "fixture_forward_13week_frn_high_rate_assumption",
        "claim_boundary": "future_frn_resets_are_explicit_forecast_assumptions",
    }


def _write_frn_rate_path(
    tmp_path: Path,
    *,
    rate_decimal: float = 0.06,
    period_start: str = "2026-09-20",
    period_end: str = "2026-09-30",
    rows: list[dict[str, object]] | None = None,
) -> Path:
    return _write_csv(
        tmp_path / "tdcsim_frn_rate_path.csv",
        rows
        or [
            _frn_rate_path_row(
                period_start=period_start,
                period_end=period_end,
                rate_decimal=rate_decimal,
            )
        ],
    )


def _write_tips_forward_paths(tmp_path: Path) -> dict[str, Path]:
    cpi_rows = [
        {
            "schema_version": "tdcsim_tips_cpi_path_v1",
            "scenario_id": "baseline",
            "month": month,
            "cbo_cpi_u_index": cpi,
            "tips_cpi_u_index": cpi,
            "reference_lag_months": 3.0,
            "interpolation_method": "treasury_daily_linear_reference_cpi",
            "anchor_date": "2026-09-20",
            "anchor_reference_cpi": 330.0,
            "scale_factor": 1.0,
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "observation_date": "2026-09-20",
            "available_date": "2026-09-20",
            "source_status": "fixture_tips_cpi_path",
            "claim_boundary": "future_tips_cpi_is_explicit_assumption",
        }
        for month, cpi in [
            ("2026-06-01", 330.0),
            ("2026-07-01", 331.0),
            ("2026-08-01", 332.0),
            ("2026-09-01", 333.0),
            ("2026-10-01", 334.0),
        ]
    ]
    real_yield_rows = [
        {
            "schema_version": "tdcsim_tips_real_yield_path_v1",
            "scenario_id": "baseline",
            "curve_date": "2026-09-30",
            "tenor_years": 10.0,
            "nominal_rate_decimal": 0.015,
            "expected_inflation_decimal": 0.03,
            "real_yield_decimal": -0.015,
            "real_coupon_decimal": 0.00125,
            "pricing_method": "real_cashflow_present_value_semiannual",
            "coupon_rounding": "floor_to_1_8_percent_min_1_8_percent",
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "observation_date": "2026-09-20",
            "available_date": "2026-09-20",
            "source_status": "fixture_tips_real_yield_path",
            "claim_boundary": "future_tips_real_yield_is_explicit_assumption",
        }
    ]
    return {
        "tips_cpi_path_file": _write_csv(tmp_path / "tdcsim_tips_cpi_path.csv", cpi_rows),
        "tips_real_yield_path_file": _write_csv(tmp_path / "tdcsim_tips_real_yield_path.csv", real_yield_rows),
    }


def _write_macro_path(
    tmp_path: Path,
    *,
    scenario_id: str = "baseline",
    cpi_u_index: float = 120.0,
) -> Path:
    return _write_csv(
        tmp_path / "tdcsim_macro_forecast_path.csv",
        [
            {
                "schema_version": "tdcsim_macro_forecast_path_v1",
                "scenario_id": scenario_id,
                "period_start": "2026-07-01",
                "period_end": "2026-09-30",
                "cbo_3m_tbill_rate_pct": 3.70,
                "cbo_10y_treasury_rate_pct": 4.05,
                "cbo_cpi_u_index": cpi_u_index,
                "cbo_cpi_u_inflation_pct": 0.0,
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "source_vintage": "cbo_2026_02_baseline",
                "forecast_publication_date": "2026-02-11",
                "source_table": "1. Quarterly",
                "source_row_selector": "Consumer price index, all urban consumers (CPI-U)",
                "observation_date": "2026-02-11",
                "available_date": "2026-02-11",
                "source_status": "fixture_cbo_quarterly_macro_workbook",
                "claim_boundary": "quarterly_cbo_macro_anchors_not_monthly_reference_cpi",
            }
        ],
    )


def _write_cbo_fiscal_baseline(
    tmp_path: Path,
    *,
    scenario_id: str = "baseline",
    cbo_net_interest_bil: float = 120.0,
    cbo_total_deficit_bil: float = 160.0,
) -> Path:
    return _write_csv(
        tmp_path / "tdcsim_cbo_fiscal_baseline.csv",
        [
            {
                "schema_version": "tdcsim_cbo_fiscal_baseline_v1",
                "scenario_id": scenario_id,
                "fiscal_year": 2026,
                "primary_deficit_bil": 40.0,
                "cbo_net_interest_bil": cbo_net_interest_bil,
                "cbo_total_deficit_bil": cbo_total_deficit_bil,
                "debt_held_public_begin_bil": 1_125.0,
                "debt_held_public_end_bil": 1_250.0,
                "debt_identity_end_bil": 1_250.0,
                "cbo_other_means_financing_bil": 85.0,
                "cbo_financial_assets_end_bil": 0.0,
                "cbo_fed_holdings_end_bil": 0.0,
                "cbo_average_interest_rate_pct": 3.75,
                "source_role": "diagnostic",
                "runtime_role": "check_only",
                "source_vintage": "cbo_2026_02_baseline",
                "forecast_publication_date": "2026-02-11",
                "source_as_of": "2026-02-11",
                "observation_date": "2026-02-11",
                "available_date": "2026-02-11",
                "source_family": "cbo_budget_workbook",
                "source_table": "Table 1-1; Table 1-3",
                "source_row_selector": "Net interest; Total deficit; Debt held by the public",
                "source_status": "fixture_fiscal_baseline_diagnostics_only",
                "claim_boundary": "fiscal_baseline_diagnostics_do_not_size_cash_or_issuance",
            }
        ],
    )


def _dynamic_surface_path(tmp_path: Path) -> Path:
    rows = build_yield_curve_surface_rows(
        macro_forecast_rows=[
            {
                "schema_version": "tdcsim_macro_forecast_path_v1",
                "scenario_id": "baseline",
                "period_start": "2026-09-01",
                "period_end": "2026-09-30",
                "cbo_3m_tbill_rate_pct": 3.70,
                "cbo_10y_treasury_rate_pct": 4.05,
                "cbo_cpi_u_index": 320.0,
                "cbo_cpi_u_inflation_pct": 0.0,
                "available_date": "2026-09-01",
            }
        ],
        base_curve_rows=[
            {
                "base_curve_date": "2026-09-01",
                "available_date": "2026-09-01",
                "base_curve_source_key": "test_treasury_curve",
                "base_curve_sha256": "testsha",
                "tenor_years": tenor,
                "nominal_rate": rate,
            }
            for tenor, rate in [
                (0.25, 4.10),
                (2.0, 4.00),
                (10.0, 4.25),
            ]
        ],
        actuals_available_as_of="2026-09-30",
        output_tenors=(0.25, 2.0, 10.0),
    )
    return _write_csv(tmp_path / "tdcsim_yield_curve_surface.csv", rows)


def _run_cbo_contract_from_temp_inputs(
    paths: dict[str, Path],
    *,
    scenario_id: str = "baseline",
    holder_preferences: dict[str, float] | None = None,
    pre_issuance_controlled_debt_bil: float = 1_000.0,
    funding_rows: list[dict[str, object]] | None = None,
) -> CboContractRun:
    debt_path = load_debt_stock_path(paths["debt_stock_path_file"])
    primary_path = load_primary_deficit_path(paths["primary_deficit_path_file"])
    cash_path = load_operating_cash_path(paths["operating_cash_path_file"])
    residual_path = load_cash_reconciliation_residual(paths["cash_reconciliation_residual_file"])
    incidence_path = load_fiscal_incidence_policy(paths["fiscal_incidence_policy_file"])
    net_interest_path = load_net_interest_bridge(paths["net_interest_bridge_file"])

    debt_row = debt_path[
        (debt_path["scenario_id"] == scenario_id)
        & (debt_path["anchor_type"] == "cbo_fiscal_year_end")
    ].iloc[0]
    decomposition = calculate_controlled_marketable_target(
        debt_row["cbo_federal_debt_held_public_target_bil"],
        debt_row["public_nonmarketable_treasury_bil"],
        debt_row["non_treasury_and_definition_residual_bil"],
    )
    plan = build_funding_plan(
        controlled_debt_target=decomposition.marketable_treasury_public_target_bil,
        pre_issuance_controlled_debt=pre_issuance_controlled_debt_bil,
        funding_rows=funding_rows
        or [
            {
                "security_type": "Fixed",
                "maturity_years": 1.0,
                "coupon_rate": 0.0,
                "yield_at_issuance": _bill_yield_for_price(0.98),
                "face_share": 1.0,
            }
        ],
        holder_preferences=holder_preferences,
        cbo_net_interest_bil=float(
            net_interest_path.loc[
                net_interest_path["component_key"] == "cbo_reported_net_interest_check",
                "amount_bil",
            ].iloc[0]
        ),
    )
    if abs(plan["target_error"]) > 0.000001:
        raise AssertionError(f"CBO controlled-debt target miss exceeds tolerance: {plan['target_error']}")

    policy_row = incidence_path[
        incidence_path["policy_id"].astype(str).str.endswith("central_99du_1ru")
    ].iloc[0].to_dict()
    allocation = allocate_signed_primary_flow(policy_row)
    primary_total = primary_path.loc[primary_path["scenario_id"] == scenario_id, "primary_deficit_bil"].sum()
    if allocation["aggregate_signed_primary_flow_bil"] != pytest.approx(primary_total):
        raise AssertionError("fiscal incidence allocation changed aggregate primary flow")

    modeled_components = net_interest_path[net_interest_path["include_in_budget_interest"]]["amount_bil"].sum()
    cbo_check = net_interest_path[net_interest_path["component_key"] == "cbo_reported_net_interest_check"].iloc[0]
    diagnostic = build_net_interest_diagnostic(
        cbo_net_interest_bil=float(cbo_check["amount_bil"]),
        modeled_net_interest_bil=float(modeled_components),
        fiscal_year=int(cbo_check["fiscal_year"]),
        scope_status="partial",
        calibration_mode="diagnostic_only",
    )
    holder_weights = holder_preferences or {"Private": 1.0}
    holder_allocations = {
        holder: plan["required_face_issuance"] * weight
        for holder, weight in holder_weights.items()
    }

    return CboContractRun(
        controlled_debt_target=decomposition.marketable_treasury_public_target_bil,
        required_face_issuance=plan["required_face_issuance"],
        post_issuance_controlled_debt=plan["post_issuance_controlled_debt"],
        target_error=plan["target_error"],
        auction_proceeds=plan["auction_proceeds"],
        cash_reconciliation_residual=float(residual_path["cash_reconciliation_residual_bil"].sum()),
        operating_cash_target=float(cash_path.iloc[-1]["operating_cash_target_bil"]),
        holder_allocations=holder_allocations,
        aggregate_signed_primary_flow=allocation["aggregate_signed_primary_flow_bil"],
        net_interest_diagnostic=diagnostic,
    )


def test_cbo_branch_hits_controlled_debt_target_at_enforced_period(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)

    results, _ = run_simulation(
        _minimal_engine_params(paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    period = results.iloc[-1]

    assert period["CBOFundingModeActive"] == pytest.approx(1.0)
    assert period["CBOControlledDebtTarget"] == pytest.approx(1_125.0)
    assert period["CBOControlledDebtPreIssuance"] == pytest.approx(1_000.0)
    assert period["NewDebtIssued"] == pytest.approx(125.0)
    assert period["CBOControlledDebtPostIssuance"] == pytest.approx(period["CBOControlledDebtTarget"])
    assert period["CBOControlledDebtTargetError"] == pytest.approx(0.0, abs=0.000001)


def test_cbo_frn_coupon_date_accrues_interval_before_payment(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rate_decimal=0.06)
    assert load_frn_rate_path(paths["frn_rate_path_file"]).iloc[0]["benchmark_rate_decimal"] == pytest.approx(0.06)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(accrued_interest=1.0)

    results, final_portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    period = results.iloc[-1]
    opening_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    expected_period_accrual = 1_000.0 * (0.06 + 0.001) / 360.0 * 10.0
    expected_payment = 1.0 + expected_period_accrual
    assert period["InterestPaid_Bonds"] == pytest.approx(expected_payment)
    assert period["TDC_FRNInterestToDU"] == pytest.approx(expected_payment)
    assert opening_frn["BenchmarkRate_FRN"] == pytest.approx(0.06)
    assert opening_frn["AccruedInterest_FRN"] == pytest.approx(0.0)


def test_cbo_frn_maturity_date_accrues_interval_before_redemption_payment(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rate_decimal=0.05)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(
        accrued_interest=0.25,
        first_interest_payment_date="2026-12-30",
        maturity_date="2026-09-30",
    )

    results, final_portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    expected_period_accrual = 1_000.0 * (0.05 + 0.001) / 360.0 * 10.0
    expected_payment = 0.25 + expected_period_accrual
    assert results.iloc[-1]["InterestPaid_Bonds"] == pytest.approx(expected_payment)
    assert results.iloc[-1]["TDC_FRNInterestToDU"] == pytest.approx(expected_payment)
    redeemed_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    assert redeemed_frn["Status"] == "Matured"
    assert redeemed_frn["AccruedInterest_FRN"] == pytest.approx(0.0)


def test_cbo_frn_coarse_frequency_same_day_coupon_maturity_pays_final_interest_once(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rate_decimal=0.05)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(
        accrued_interest=0.25,
        first_interest_payment_date="2026-09-30",
        maturity_date="2026-09-30",
    )

    results, final_portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    expected_period_accrual = 1_000.0 * (0.05 + 0.001) / 360.0 * 10.0
    expected_final_interest = 0.25 + expected_period_accrual
    assert results.iloc[-1]["InterestPaid_Bonds"] == pytest.approx(expected_final_interest)
    assert results.iloc[-1]["TDC_FRNInterestToDU"] == pytest.approx(expected_final_interest)
    redeemed_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    assert redeemed_frn["Status"] == "Matured"
    assert redeemed_frn["AccruedInterest_FRN"] == pytest.approx(0.0)


def test_cbo_frn_daily_lockout_uses_two_business_days_with_holiday_weekend(tmp_path: Path) -> None:
    periods = build_simulation_calendar("2026-07-01", "2026-07-07", "daily")
    paths = _build_temp_forecast_inputs(tmp_path, periods=periods, period_start="2026-07-01", period_end="2026-07-07")
    rows = []
    rates_by_period_end = {
        "2026-07-02": 0.020,
        "2026-07-03": 0.030,
        "2026-07-04": 0.040,
        "2026-07-05": 0.050,
        "2026-07-06": 0.060,
        "2026-07-07": 0.090,
    }
    for period in periods:
        period_end = period.period_end.isoformat()
        rows.append(
            _frn_rate_path_row(
                period_start=period.period_start.isoformat(),
                period_end=period_end,
                rate_decimal=rates_by_period_end[period_end],
            )
        )
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rows=rows)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(
        accrued_interest=0.0,
        first_interest_payment_date="2026-07-07",
        maturity_date="2027-07-07",
    )

    results, final_portfolio = run_simulation(
        params,
        "2026-07-01",
        "2026-07-07",
        freq="D",
        scenario_name="baseline",
    )

    expected_payment = 1_000.0 * ((0.020 + 0.001) * 6.0 / 360.0)
    paid = float(results["InterestPaid_Bonds"].sum())
    assert paid == pytest.approx(expected_payment)
    final_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    assert final_frn["BenchmarkRate_FRN"] == pytest.approx(0.020)
    assert final_frn["AccruedInterest_FRN"] == pytest.approx(0.0)


def test_cbo_frn_daily_lockout_ignores_rate_change_inside_lockout(tmp_path: Path) -> None:
    periods = build_simulation_calendar("2026-09-20", "2026-09-30", "daily")
    paths = _build_temp_forecast_inputs(tmp_path, periods=periods, period_start="2026-09-20", period_end="2026-09-30")
    rows = []
    for period in periods:
        period_end = period.period_end.isoformat()
        if period_end >= "2026-09-29":
            rate = 0.20
        elif period_end == "2026-09-28":
            rate = 0.04
        else:
            rate = 0.02
        rows.append(
            _frn_rate_path_row(
                period_start=period.period_start.isoformat(),
                period_end=period_end,
                rate_decimal=rate,
            )
        )
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rows=rows)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(accrued_interest=1.0)

    results, final_portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="D",
        scenario_name="baseline",
    )

    expected_payment = 1.0 + 1_000.0 * (((0.02 + 0.001) * 7.0 + (0.04 + 0.001) * 3.0) / 360.0)
    final_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    assert results.iloc[-1]["TDC_FRNInterestToDU"] == pytest.approx(expected_payment)
    assert final_frn["BenchmarkRate_FRN"] == pytest.approx(0.04)
    assert final_frn["AccruedInterest_FRN"] == pytest.approx(0.0)


def test_cbo_frn_coarse_frequency_keeps_post_coupon_accrual_unpaid(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)
    paths["frn_rate_path_file"] = _write_frn_rate_path(tmp_path, rate_decimal=0.06)
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _opening_frn_portfolio(
        accrued_interest=1.0,
        first_interest_payment_date="2026-09-25",
        maturity_date="2027-09-25",
    )

    results, final_portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    paid_through_coupon = 1.0 + 1_000.0 * ((0.06 + 0.001) * 5.0 / 360.0)
    retained_post_coupon = 1_000.0 * ((0.06 + 0.001) * 5.0 / 360.0)
    final_frn = final_portfolio[final_portfolio["BondID"] == 1].iloc[0]
    assert results.iloc[-1]["TDC_FRNInterestToDU"] == pytest.approx(paid_through_coupon)
    assert final_frn["AccruedInterest_FRN"] == pytest.approx(retained_post_coupon)


def test_cbo_dynamic_yield_surface_prices_bills_with_decimal_runtime_rates(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, pre_issuance_controlled_debt_bil=0.0)
    surface_path = _dynamic_surface_path(tmp_path)
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["yield_curve_surface"] = {"file": str(surface_path), "scenario_id": "baseline"}
    params["treasury_issuance_profile"]["bills"]["maturities"] = [0.25]
    params["treasury_issuance_profile"]["bills"]["maturity_distribution"] = [1.0]
    assert load_runtime_yield_curve_surface(surface_path)["nominal_rate_decimal"].iloc[0] == pytest.approx(0.037)

    _, portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    issued = portfolio[portfolio["Status"] == "Active"].iloc[0]

    assert issued["IssueYieldAtIssue"] == pytest.approx(0.037)
    assert issued["IssuePriceRatio"] == pytest.approx(1.0 / ((1.0 + 0.037) ** 0.25))
    assert issued["IssuePriceRatio"] > 0.99


def test_cbo_tips_issuance_uses_real_yield_path_for_premium_pricing(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, pre_issuance_controlled_debt_bil=0.0)
    paths.update(_write_tips_forward_paths(tmp_path))
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["treasury_issuance_profile"]["TIPS"] = {
        "target_percentage": 1.0,
        "maturities": [10.0],
        "maturity_distribution": [1.0],
    }
    params["treasury_issuance_profile"]["FRN"] = {
        "target_percentage": 0.0,
        "maturities": [2.0],
        "maturity_distribution": [1.0],
    }
    for holder, prefs in params["sector_preferences"].items():
        prefs["tips_pct"] = 1.0 if holder == "Private" else 0.0

    results, portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    issued = portfolio[portfolio["SecurityType"].astype(str).eq("TIPS")]
    assert len(issued) == 1
    row = issued.iloc[0]
    assert row["IssueYieldAtIssue"] == pytest.approx(-0.015)
    assert row["CouponRate"] == pytest.approx(0.00125)
    assert row["IssuePriceRatio"] > 1.0
    assert row["IssueProceeds"] > row["FaceValue"]
    assert results.iloc[-1]["AuctionProceeds"] == pytest.approx(row["IssueProceeds"])
    assert row["ReferenceCPI_Issue"] == pytest.approx(results.iloc[-1]["Reference_CPI"])


def test_cbo_rejects_unrelated_scenario_rows(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, scenario_id="other_scenario")

    with pytest.raises(ValueError, match="has no rows for scenario 'baseline'"):
        run_simulation(
            _minimal_engine_params(paths),
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )


def test_engine_uses_baseline_input_yield_surface(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, pre_issuance_controlled_debt_bil=0.0)
    surface_path = _dynamic_surface_path(tmp_path)
    paths["yield_curve_surface_file"] = surface_path
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["yield_curve"] = {
        "use_static": True,
        "years": [0.25, 10.0],
        "rates": [0.25, 0.25],
    }
    params["treasury_issuance_profile"]["bills"]["maturities"] = [0.25]
    params["treasury_issuance_profile"]["bills"]["maturity_distribution"] = [1.0]

    _, portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    issued = portfolio[portfolio["Status"] == "Active"].iloc[0]

    assert issued["IssueYieldAtIssue"] == pytest.approx(0.037)


def test_cbo_macro_cpi_path_drives_tips_reference_cpi_and_adjusted_principal(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=245.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0,
    )
    paths["macro_forecast_path_file"] = _write_macro_path(tmp_path, cpi_u_index=120.0)
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = _opening_tips_portfolio(face_value=100.0, reference_cpi=100.0)
    params["tips_params"] = {
        "cpi_start_level": 100.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.01,
    }
    params["financing_cost_options"] = {"include_tips_inflation_accretion": True}
    params["sector_preferences"] = {
        "Banks": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0},
        "Private": {"bills_pct": 1.0, "notes_pct": 1.0, "bonds_pct": 1.0, "tips_pct": 1.0},
        "CB": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0},
        "Foreign": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0},
        "FedInternal": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0},
        "TrustFunds": {"bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0},
    }

    results, portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    final = results.iloc[-1]
    active_tips = portfolio.loc[portfolio["SecurityType"] == "TIPS"].iloc[0]

    assert final["CPI_Level"] == pytest.approx(120.0)
    assert final["Reference_CPI"] == pytest.approx(120.0)
    assert final["DebtHeldByType_TIPS"] == pytest.approx(120.0)
    assert final["CBORequiredFaceIssuance"] == pytest.approx(0.0)
    assert active_tips["AdjustedPrincipal"] == pytest.approx(120.0)
    assert active_tips["IndexRatio"] == pytest.approx(1.2)


def test_cbo_tips_deflation_accretion_is_signed(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=205.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0,
    )
    paths["macro_forecast_path_file"] = _write_macro_path(tmp_path, cpi_u_index=80.0)
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = _opening_tips_portfolio(face_value=100.0, reference_cpi=100.0)
    params["tips_params"] = {
        "cpi_start_level": 100.0,
        "cpi_annual_inflation": 0.0,
        "ref_cpi_lag_months": 0,
        "default_real_coupon_rate": 0.01,
    }
    params["financing_cost_options"] = {"include_tips_inflation_accretion": True}

    results, portfolio = run_simulation(
        params,
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    final = results.iloc[-1]
    active_tips = portfolio.loc[portfolio["SecurityType"] == "TIPS"].iloc[0]

    assert final["DebtHeldByType_TIPS"] == pytest.approx(80.0)
    assert final["TIPSInflationAccretion_Period"] == pytest.approx(-20.0)
    assert final["TIPSInflationAccretion_Cumulative"] == pytest.approx(-20.0)
    assert final["CBORequiredFaceIssuance"] == pytest.approx(0.0)
    assert active_tips["AdjustedPrincipal"] == pytest.approx(80.0)


def test_cbo_run_metadata_records_funding_mode_and_bridge_rows(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)

    results, _ = run_simulation(
        _minimal_engine_params(paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    metadata = results.attrs["run_metadata"]

    assert metadata["funding_rule_mode"] == "cbo_public_debt_target"
    assert metadata["cbo_funding_mode_active"] is True
    assert metadata["cbo_net_interest_bridge_rows"] == 3
    assert results.iloc[-1]["CBONetInterestBridgeRows"] == pytest.approx(3.0)


def test_holder_preference_changes_do_not_change_face_issuance_or_debt_target(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)

    private_results, _ = run_simulation(
        _minimal_engine_params(paths, private_bills=1.0, banks_bills=0.0),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    banks_results, _ = run_simulation(
        _minimal_engine_params(paths, private_bills=0.0, banks_bills=1.0),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    private_heavy = private_results.iloc[-1]
    banks_heavy = banks_results.iloc[-1]

    assert private_heavy["CBOControlledDebtTarget"] == pytest.approx(banks_heavy["CBOControlledDebtTarget"])
    assert private_heavy["NewDebtIssued"] == pytest.approx(banks_heavy["NewDebtIssued"])
    assert private_heavy["CBOControlledDebtPostIssuance"] == pytest.approx(
        banks_heavy["CBOControlledDebtPostIssuance"]
    )
    assert private_heavy["DebtHeld_DomesticNonBanks"] != pytest.approx(banks_heavy["DebtHeld_DomesticNonBanks"])
    assert private_heavy["DebtHeld_Banks"] != pytest.approx(banks_heavy["DebtHeld_Banks"])


def test_operating_cash_target_and_residual_do_not_resize_required_face_issuance(tmp_path: Path) -> None:
    baseline_paths = _build_temp_forecast_inputs(
        tmp_path / "baseline",
        base_cash_balance_bil=800.0,
        cash_residual_bil=0.0,
    )
    cash_sensitivity_paths = _build_temp_forecast_inputs(
        tmp_path / "cash_sensitivity",
        base_cash_balance_bil=950.0,
        cash_residual_bil=150.0,
    )

    baseline_results, _ = run_simulation(
        _minimal_engine_params(baseline_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    cash_sensitivity_results, _ = run_simulation(
        _minimal_engine_params(cash_sensitivity_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    baseline = baseline_results.iloc[-1]
    cash_sensitivity = cash_sensitivity_results.iloc[-1]

    assert baseline["CBOControlledDebtTarget"] == pytest.approx(cash_sensitivity["CBOControlledDebtTarget"])
    assert baseline["NewDebtIssued"] == pytest.approx(cash_sensitivity["NewDebtIssued"])
    assert baseline["CBORequiredFaceIssuance"] == pytest.approx(cash_sensitivity["CBORequiredFaceIssuance"])
    assert baseline["AuctionProceeds"] == pytest.approx(cash_sensitivity["AuctionProceeds"])
    assert baseline["CBOControlledDebtPostIssuance"] == pytest.approx(
        cash_sensitivity["CBOControlledDebtPostIssuance"]
    )
    assert baseline["CBOOperatingCashTarget"] != pytest.approx(cash_sensitivity["CBOOperatingCashTarget"])
    assert baseline["CBOCashReconciliationResidual"] != pytest.approx(
        cash_sensitivity["CBOCashReconciliationResidual"]
    )


def test_cbo_fed_holdings_path_uses_secondary_purchase_not_auction_share(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, cbo_public_debt_target_bil=1_250.0)
    fed_rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=_single_period(),
        opening_state_date="2026-09-20",
        opening_cb_holdings_bil=0.0,
        cbo_fy_end_fed_holdings_bil={2026: 50.0},
        observation_date="2026-09-20",
        available_date="2026-09-20",
    )
    paths["fed_holdings_path_file"] = _write_csv(tmp_path / "tdcsim_fed_holdings_path.csv", fed_rows)

    loaded_fed_path = load_fed_holdings_path(paths["fed_holdings_path_file"])
    assert loaded_fed_path["cbo_fed_holdings_target_bil"].iloc[-1] == pytest.approx(50.0)

    results, _ = run_simulation(
        _minimal_engine_params(paths, private_bills=1.0, banks_bills=0.0),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )

    period = results.iloc[-1]
    assert period["CBORequiredFaceIssuance"] == pytest.approx(125.0)
    assert period["NewDebtIssued"] == pytest.approx(125.0)
    assert period["AuctionProceeds"] == pytest.approx(119.04761904761904)
    assert period["TGA"] == pytest.approx(859.047619047619)
    assert period["CBOFedHoldingsTarget"] == pytest.approx(50.0)
    assert period["CBOFedHoldingsTargetApplicable"] == pytest.approx(1.0)
    assert period["CBOFedAuctionShare"] == pytest.approx(0.0)
    assert period["CBOFedSecondaryPurchaseFace"] == pytest.approx(50.0)
    assert period["CBOFedSecondaryPurchaseCash"] == pytest.approx(0.0)
    assert period["CBOFedSecondaryPurchaseReserveEffect"] == pytest.approx(0.0)
    assert period["CBOFedSecondaryPurchaseDepositEffect"] == pytest.approx(0.0)
    assert period["CBOFedSyntheticSecondaryPurchases"] == pytest.approx(50.0)
    assert period["CBOFedBeginStock"] == pytest.approx(0.0)
    assert period["CBOFedEndStock"] == pytest.approx(50.0)
    assert period["CBOFedNetStockChange"] == pytest.approx(50.0)
    assert period["CBOFedGrossStockFlow"] == pytest.approx(50.0)
    assert period["CBOFedStockMode"] == "synthetic_cb_treasury_stock_target_par_reallocation"
    assert period["CBOFedSettlementScope"] == "stock_reallocation_only_no_reserve_deposit_or_market_price_claim"
    assert period["CBORemittanceCashEffect"] == pytest.approx(0.0)
    assert period["CBORemittanceStatus"] == "not_modeled_cbo_primary_deficit_embeds_baseline_revenues"
    assert period["DebtHeld_CentralBank"] == pytest.approx(50.0)
    assert period["CBOFedHoldingsTargetError"] == pytest.approx(0.0)
    assert period["DebtHeld_DomesticNonBanks"] == pytest.approx(1_075.0)
    assert period["Reserves"] == pytest.approx(940.952380952381)
    assert period["TDC_Level"] == pytest.approx(-59.44761904761904)


def test_cbo_fed_zero_target_rejects_explicit_cb_auction_preferences(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(tmp_path, cbo_public_debt_target_bil=1_250.0)
    fed_rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=_single_period(),
        opening_state_date="2026-09-20",
        opening_cb_holdings_bil=0.0,
        cbo_fy_end_fed_holdings_bil={2026: 0.0},
        observation_date="2026-09-20",
        available_date="2026-09-20",
    )
    paths["fed_holdings_path_file"] = _write_csv(tmp_path / "tdcsim_fed_holdings_path.csv", fed_rows)
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=1_000.0)
    for holder in params["sector_preferences"]:
        params["sector_preferences"][holder]["bills_pct"] = 0.0
    params["sector_preferences"]["CB"]["bills_pct"] = 1.0
    params["sector_preferences"]["Private"]["bills_pct"] = 0.0

    with pytest.raises(RuntimeError, match="must be met without Fed auction issuance"):
        run_simulation(
            params,
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )


def test_cbo_fiscal_baseline_diagnostics_do_not_resize_issuance_or_cash(tmp_path: Path) -> None:
    no_baseline_paths = _build_temp_forecast_inputs(tmp_path / "no_baseline")
    diagnostic_paths = _build_temp_forecast_inputs(tmp_path / "diagnostic")
    diagnostic_paths["cbo_fiscal_baseline_file"] = _write_cbo_fiscal_baseline(
        tmp_path / "diagnostic",
        cbo_net_interest_bil=999.0,
        cbo_total_deficit_bil=1_234.0,
    )

    no_baseline_results, _ = run_simulation(
        _minimal_engine_params(no_baseline_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    diagnostic_results, _ = run_simulation(
        _minimal_engine_params(diagnostic_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    no_baseline = no_baseline_results.iloc[-1]
    diagnostic = diagnostic_results.iloc[-1]

    assert diagnostic["CBONetInterestDiagnostic"] == pytest.approx(999.0)
    assert diagnostic["CBOTotalDeficitDiagnostic"] == pytest.approx(1_234.0)
    assert diagnostic["NetInterestDiagnosticStatus"] == "cbo_reported_check_only"
    assert no_baseline["NetInterestDiagnosticStatus"] == "not_loaded_check_only"
    assert no_baseline["CBOControlledDebtTarget"] == pytest.approx(diagnostic["CBOControlledDebtTarget"])
    assert no_baseline["NewDebtIssued"] == pytest.approx(diagnostic["NewDebtIssued"])
    assert no_baseline["CBORequiredFaceIssuance"] == pytest.approx(diagnostic["CBORequiredFaceIssuance"])
    assert no_baseline["AuctionProceeds"] == pytest.approx(diagnostic["AuctionProceeds"])
    assert no_baseline["TGA"] == pytest.approx(diagnostic["TGA"])
    assert no_baseline["CBOCashResidual"] == pytest.approx(diagnostic["CBOCashResidual"])


def test_missing_fiscal_incidence_policy_fails_before_tdc_recipient_routing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fiscal incidence policy is required"):
        validate_fiscal_incidence_policy_rows([])

    paths = _build_temp_forecast_inputs(tmp_path)
    paths["fiscal_incidence_policy_file"].unlink()

    with pytest.raises(FileNotFoundError, match="fiscal incidence policy file is missing"):
        run_simulation(
            _minimal_engine_params(paths),
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )


@pytest.mark.parametrize(
    "missing_path_key",
    [
        "debt_stock_path_file",
        "primary_deficit_path_file",
        "operating_cash_path_file",
        "cash_reconciliation_residual_file",
        "fiscal_incidence_policy_file",
    ],
)
def test_cbo_mode_missing_required_baseline_path_fails_loudly(
    tmp_path: Path,
    missing_path_key: str,
) -> None:
    paths = _build_temp_forecast_inputs(tmp_path)
    paths.pop(missing_path_key)

    with pytest.raises(ValueError, match=missing_path_key):
        run_simulation(
            _minimal_engine_params(paths),
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )


def test_net_interest_target_and_bridge_changes_diagnostics_only_not_issuance(tmp_path: Path) -> None:
    baseline_paths = _build_temp_forecast_inputs(tmp_path / "baseline", cbo_net_interest_bil=120.0)
    changed_paths = _build_temp_forecast_inputs(tmp_path / "changed", cbo_net_interest_bil=300.0)

    baseline_results, _ = run_simulation(
        _minimal_engine_params(baseline_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    changed_results, _ = run_simulation(
        _minimal_engine_params(changed_paths),
        "2026-09-20",
        "2026-09-30",
        freq="10D",
        scenario_name="baseline",
    )
    baseline = baseline_results.iloc[-1]
    changed = changed_results.iloc[-1]
    baseline_contract = _run_cbo_contract_from_temp_inputs(baseline_paths)
    changed_contract = _run_cbo_contract_from_temp_inputs(changed_paths)

    assert baseline["CBOControlledDebtTarget"] == pytest.approx(changed["CBOControlledDebtTarget"])
    assert baseline["NewDebtIssued"] == pytest.approx(changed["NewDebtIssued"])
    assert baseline["CBOControlledDebtPostIssuance"] == pytest.approx(changed["CBOControlledDebtPostIssuance"])
    assert baseline["AuctionProceeds"] == pytest.approx(changed["AuctionProceeds"])
    assert baseline_contract.net_interest_diagnostic["residual_bil"] != pytest.approx(
        changed_contract.net_interest_diagnostic["residual_bil"]
    )


def test_negative_required_issuance_fails_closed(tmp_path: Path) -> None:
    paths = _build_temp_forecast_inputs(
        tmp_path,
        cbo_public_debt_target_bil=1_100.0,
        public_nonmarketable_bil=100.0,
        definition_residual_bil=25.0,
        pre_issuance_controlled_debt_bil=1_000.0,
    )

    with pytest.raises(ValueError, match="negative required face issuance"):
        run_simulation(
            _minimal_engine_params(paths),
            "2026-09-20",
            "2026-09-30",
            freq="10D",
            scenario_name="baseline",
        )


def test_current_cash_mode_regression_cbo_blocks_do_not_resize_legacy_cash_issuance(tmp_path: Path) -> None:
    baseline_results, _ = run_simulation(
        _minimal_cash_mode_params(),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="cash_baseline",
    )
    cbo_paths = _build_temp_forecast_inputs(tmp_path / "unused_cbo_inputs")
    with_cbo_metadata_results, _ = run_simulation(
        _minimal_cash_mode_params(cbo_paths),
        "2025-01-01",
        "2025-01-19",
        freq="W",
        scenario_name="cash_with_cbo_metadata",
    )

    baseline_period = baseline_results.iloc[1]
    metadata_period = with_cbo_metadata_results.iloc[1]
    assert metadata_period["CBOFundingModeActive"] == pytest.approx(0.0)
    assert metadata_period["AuctionProceeds"] == pytest.approx(baseline_period["AuctionProceeds"])
    assert metadata_period["NewDebtIssued"] == pytest.approx(baseline_period["NewDebtIssued"])
