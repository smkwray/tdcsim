#!/usr/bin/env python3
"""Repeatable local CBO full-horizon smoke runner.

The runner writes only under ``output/`` by default. It is intentionally a
local smoke harness, not a production forecast bundle: the opening portfolio,
issuance mix, and holder routing are explicit scenario assumptions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from budget_interest import build_average_interest_rate_diagnostic, build_net_interest_diagnostic
from cbo_policy_bundle import (
    build_fiscal_incidence_policy_rows,
    build_fiscal_incidence_sensitivity_results,
    build_net_interest_bridge_rows,
)
from cbo_portfolio_builder import (
    build_holder_profile_rows,
    default_cbo_holder_preferences,
    default_private_subbucket_shares,
    validate_holder_preferences_non_degenerate,
)
from cbo_yield_curve_surface import build_yield_curve_surface_rows
from forecast_bundle_builders import (
    build_cash_reconciliation_residual_rows,
    build_current_fy_splice_row,
    build_current_fy_splice_row_from_fiscaldata,
    build_debt_stock_path_rows,
    build_fed_holdings_path_rows,
    build_operating_cash_path_rows,
    build_primary_deficit_path_rows,
    calculate_public_debt_bridge_components,
    federal_fiscal_year,
    tga_closing_balance_bil_from_dts_row,
)
from forecast_input_builder import (
    BUDGET_WORKBOOK_SHA256,
    ECONOMIC_WORKBOOK_SHA256,
    FORECAST_NAME,
    FORECAST_PUBLICATION_DATE,
    build_cbo_fiscal_baseline_rows,
    build_cbo_macro_forecast_path_rows,
    parse_cbo_budget_source_contract,
    parse_cbo_economic_quarterly_source_contract,
    verify_cbo_workbook_hashes,
    write_forecast_rows_csv,
)
from frn_indexation import build_forward_frn_rate_path_rows, enrich_opening_frn_from_daily_indexes
from sim_engine import run_simulation
from simulation_calendar import SimulationPeriod, build_simulation_calendar
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    TGA_FLOOR_TOLERANCE,
)
from tips_indexation import (
    build_monthly_tips_cpi_path_rows,
    build_tips_real_yield_path_rows,
    enrich_opening_tips_from_cpi_detail,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "cbo_forecast_smoke"
DEFAULT_BUDGET_WORKBOOK = PROJECT_ROOT.parent / "ratewall" / "data" / "raw" / "cbo" / "51118-2026-02-Budget-Projections.xlsx"
DEFAULT_ECONOMIC_WORKBOOK = PROJECT_ROOT.parent / "ratewall" / "data" / "raw" / "cbo" / "51135-2026-02-Economic-Projections.xlsx"
DEFAULT_OPENING_PORTFOLIO = PROJECT_ROOT / "data" / "ratewall_inputs" / "tdcsim_initial_portfolio_cohorts.csv"
DEFAULT_RATEWALL_INPUT_MANIFEST = PROJECT_ROOT / "data" / "ratewall_inputs" / "tdcsim_ratewall_input_manifest.json"
DEFAULT_TIPS_CPI_DETAIL = PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "tips_cpi_data_detail.csv"
DEFAULT_FRN_DAILY_INDEXES = PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "frn_daily_indexes.csv"
DEFAULT_MTS_TABLE_1 = PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "mts_table_1.csv"
DEFAULT_DTS_OPERATING_CASH_BALANCE = (
    PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "dts_operating_cash_balance.csv"
)
DEFAULT_MSPD_TABLE_1 = PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "mspd_table_1.csv"
DEFAULT_MSPD_TABLE_3_MARKET = (
    PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "mspd_table_3_market.csv"
)
DEFAULT_DEBT_OUTSTANDING = PROJECT_ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "debt_outstanding.csv"
FRN_INDEX_SOURCE_AVAILABLE_AS_OF = "2026-06-17"

SOURCE_PACKAGE_INPUTS = {
    "budget_workbook": "cbo/51118-2026-02-Budget-Projections.xlsx",
    "economic_workbook": "cbo/51135-2026-02-Economic-Projections.xlsx",
    "opening_portfolio": "opening/tdcsim_initial_portfolio_cohorts.csv",
    "ratewall_input_manifest": "opening/tdcsim_ratewall_input_manifest.json",
    "tips_cpi_detail": "fiscaldata/tips_cpi_data_detail.csv",
    "frn_daily_indexes": "fiscaldata/frn_daily_indexes.csv",
    "mts_table_1": "fiscaldata/mts_table_1_deficit_2026-05-31.csv",
    "mts_table_9_net_interest": "fiscaldata/mts_table_9_net_interest_2026-05-31.csv",
    "dts_operating_cash_balance": "fiscaldata/dts_tga_closing_balance_2026-06-16.csv",
    "mspd_table_1": "fiscaldata/mspd_table_1_debt_bridge_2026-05-31.csv",
    "mspd_table_3_market": "fiscaldata/mspd_table_3_market_2026-05-31.csv",
    "debt_to_penny_public_debt": "treasury/debt_to_penny_public_debt_2026-05-29.csv",
    "treasury_base_curve": "treasury/base_curve_2026-06-16.json",
}

SCENARIO_ID = "cbo_full_horizon_local_smoke_3m"
SIMULATION_START_DATE = "2026-06-21"
SIMULATION_END_DATE = "2036-09-30"
ACTUALS_AVAILABLE_AS_OF = "2026-06-17"
BASE_CURVE_OBSERVATION_DATE = "2026-06-16"
TGA_OBSERVATION_DATE = "2026-06-16"
FISCAL_ACTUALS_THROUGH = "2026-05-31"
OPENING_STATE_DATE = SIMULATION_START_DATE
OPENING_PORTFOLIO_RECORD_DATE = "2026-05-31"
CLAIM_BOUNDARY = "local_full_horizon_cbo_forecast_input_smoke_prebuilt_mspd_derived_opening_cohort_book"
OPENING_PORTFOLIO_CLAIM_BOUNDARY = (
    "prebuilt_mspd_table3_market_derived_cohort_book_with_z1_holder_provenance_metadata; "
    "z1_inputs_not_packaged_for_portable_holder_constraint_reproduction; "
    "securities_maturing_between_record_date_and_simulation_start_are_explicit_prestart_rollforward_bills"
)

CBO_FULL_FY2026_PRIMARY_DEFICIT_BIL = 813.727
OPENING_TREASURY_PUBLIC_DEBT_HELD_PUBLIC_BIL = 31_515.369918445198
OPENING_CBO_DEFINITION_RESIDUAL_BIL = -105.36444077058
OPENING_CBO_FEDERAL_DEBT_HELD_PUBLIC_BIL = (
    OPENING_TREASURY_PUBLIC_DEBT_HELD_PUBLIC_BIL + OPENING_CBO_DEFINITION_RESIDUAL_BIL
)
OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL = 30_894.5481443502
PUBLIC_NONMARKETABLE_TREASURY_BIL = 620.82177409502
NON_TREASURY_AND_DEFINITION_RESIDUAL_BIL = OPENING_CBO_DEFINITION_RESIDUAL_BIL

FROZEN_MTS_TABLE_1_DEFICIT_ROW = {
    "record_date": FISCAL_ACTUALS_THROUGH,
    "classification_desc": "Year-to-Date",
    "line_code_nbr": "280",
    "current_month_dfct_sur_amt": "1246203266386.93",
}
FROZEN_MTS_TABLE_9_NET_INTEREST_ROW = {
    "record_date": FISCAL_ACTUALS_THROUGH,
    "classification_desc": "Net Interest",
    "current_fytd_rcpt_outly_amt": "722706511243.20",
}
FROZEN_DTS_TGA_CLOSING_BALANCE_ROW = {
    "record_date": TGA_OBSERVATION_DATE,
    "account_type": "Treasury General Account (TGA) Closing Balance",
    "open_today_bal": "981113",
}
FROZEN_DEBT_TO_PENNY_ROW = {
    "record_date": "2026-05-29",
    "debt_held_public_amt": "31515369798622.98",
}
FROZEN_MSPD_ROWS = [
    {
        "record_date": FISCAL_ACTUALS_THROUGH,
        "security_type_desc": "Total Nonmarketable",
        "debt_held_public_mil_amt": f"{PUBLIC_NONMARKETABLE_TREASURY_BIL * 1_000:.8f}",
    },
    {
        "record_date": FISCAL_ACTUALS_THROUGH,
        "security_type_desc": "Total Public Debt Outstanding",
        "debt_held_public_mil_amt": f"{OPENING_TREASURY_PUBLIC_DEBT_HELD_PUBLIC_BIL * 1_000:.8f}",
    },
]

RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR = (
    "classification_desc=Year-to-Date;line_code_nbr=280;column=current_month_dfct_sur_amt"
)
RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR = (
    "classification_desc=Net Interest;column=current_fytd_rcpt_outly_amt"
)
RUNNER_DEBT_TO_PENNY_SELECTOR = "record_date=2026-05-29;column=debt_held_public_amt"

RESULT_COLUMNS = [
    "Date",
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBOControlledDebtTargetApplicable",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
    "NewDebtIssued",
    "AuctionProceeds",
    "IssueDiscountCost_Period",
    "InterestPaid_Bonds",
    "NonMarketableInterestCapitalized_Period",
    "PrimaryDeficit",
    "TGA",
    "CBOOperatingCashTarget",
    "CBOCashResidual",
    "CBOCashReconciliationResidual",
    "CBORemittanceCashEffect",
    "CBOFedHoldingsTarget",
    "CBOFedHoldingsTargetApplicable",
    "CBOFedHoldingsTargetError",
    "CBOFedAuctionShare",
    "CBOFedSecondaryPurchaseFace",
    "CBOFedSecondaryPurchaseCash",
    "CBOFedSecondaryPurchaseReserveEffect",
    "CBOFedSecondaryPurchaseDepositEffect",
    "CBOFedBeginStock",
    "CBOFedMaturitiesAndRedemptions",
    "CBOFedTipsPrincipalIndexation",
    "CBOFedAuctionRolloverAddons",
    "CBOFedSyntheticSecondaryPurchases",
    "CBOFedSyntheticSecondarySales",
    "CBOFedEndStock",
    "CBOFedGrossStockFlow",
    "CBOFedNetStockChange",
    "CBOFedStockMode",
    "CBOFedSettlementScope",
    "CB_InterestIncome",
    "CB_NetIncome",
    "CB_Remittance",
    "CB_DeferredAsset",
    "CBORemittanceStatus",
    "CBONetInterestDiagnostic",
    "CBOTotalDeficitDiagnostic",
    "NetInterestDiagnosticStatus",
    "DebtHeld_Banks",
    "DebtHeld_CentralBank",
    "DebtHeld_Foreign",
    "DebtHeld_DomesticNonBanks",
    "DebtHeldByType_Fixed",
    "DebtHeldByType_TIPS",
    "DebtHeldByType_FRN",
    "TIPSInflationAccretion_Period",
    "TIPSInflationAccretion_Cumulative",
    "CBOBuybackFaceRetired",
    "CBOBuybackCashPaid",
]


@dataclass(frozen=True)
class ForecastBundle:
    output_dir: Path
    inputs_dir: Path
    scenario_id: str
    periods: list[SimulationPeriod]
    manifest: dict[str, Any]
    baseline_input_paths: dict[str, str]
    initial_portfolio: pd.DataFrame
    params: dict[str, Any]


def build_forecast_bundle(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    budget_workbook: str | Path = DEFAULT_BUDGET_WORKBOOK,
    economic_workbook: str | Path = DEFAULT_ECONOMIC_WORKBOOK,
    opening_portfolio: str | Path = DEFAULT_OPENING_PORTFOLIO,
    ratewall_input_manifest: str | Path = DEFAULT_RATEWALL_INPUT_MANIFEST,
    tips_cpi_detail: str | Path = DEFAULT_TIPS_CPI_DETAIL,
    frn_daily_indexes: str | Path = DEFAULT_FRN_DAILY_INDEXES,
    mts_table_1: str | Path = DEFAULT_MTS_TABLE_1,
    dts_operating_cash_balance: str | Path = DEFAULT_DTS_OPERATING_CASH_BALANCE,
    mspd_table_1: str | Path = DEFAULT_MSPD_TABLE_1,
    mspd_table_3_market: str | Path = DEFAULT_MSPD_TABLE_3_MARKET,
    mts_table_9_net_interest: str | Path | None = None,
    debt_to_penny_public_debt: str | Path | None = None,
    treasury_base_curve: str | Path | None = None,
    clean: bool = True,
) -> ForecastBundle:
    """Build source-backed forecast input CSVs and an exact local manifest."""

    out = _resolve_output_dir(output_dir)
    if clean and out.exists():
        shutil.rmtree(out)
    inputs = out / "forecast_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    lock_path = PROJECT_ROOT / "requirements.lock.txt"
    if not lock_path.exists():
        raise FileNotFoundError("requirements.lock.txt is required for CBO package trust anchoring")
    shutil.copy2(lock_path, out / "requirements.lock.txt")

    budget_path = Path(budget_workbook)
    economic_path = Path(economic_workbook)
    opening_portfolio_path = Path(opening_portfolio)
    ratewall_manifest_path = Path(ratewall_input_manifest)
    tips_cpi_detail_path = Path(tips_cpi_detail)
    frn_daily_indexes_path = Path(frn_daily_indexes)
    mts_table_1_path = Path(mts_table_1)
    dts_operating_cash_balance_path = Path(dts_operating_cash_balance)
    mspd_table_1_path = Path(mspd_table_1)
    mspd_table_3_market_path = Path(mspd_table_3_market)
    base_curve_seed_rows = _frozen_base_curve_rows()
    source_package = write_source_package(
        out,
        budget_workbook=budget_path,
        economic_workbook=economic_path,
        opening_portfolio=opening_portfolio_path,
        ratewall_input_manifest=ratewall_manifest_path,
        tips_cpi_detail=tips_cpi_detail_path,
        frn_daily_indexes=frn_daily_indexes_path,
        mts_table_1=mts_table_1_path,
        dts_operating_cash_balance=dts_operating_cash_balance_path,
        mspd_table_1=mspd_table_1_path,
        mspd_table_3_market=mspd_table_3_market_path,
        mts_table_9_net_interest=Path(mts_table_9_net_interest) if mts_table_9_net_interest else None,
        debt_to_penny_public_debt=Path(debt_to_penny_public_debt) if debt_to_penny_public_debt else None,
        treasury_base_curve=Path(treasury_base_curve) if treasury_base_curve else None,
        base_curve_rows=base_curve_seed_rows,
    )
    packaged_sources = {name: out / "sources" / rel for name, rel in SOURCE_PACKAGE_INPUTS.items()}
    budget_path = packaged_sources["budget_workbook"]
    economic_path = packaged_sources["economic_workbook"]
    opening_portfolio_path = packaged_sources["opening_portfolio"]
    ratewall_manifest_path = packaged_sources["ratewall_input_manifest"]
    tips_cpi_detail_path = packaged_sources["tips_cpi_detail"]
    frn_daily_indexes_path = packaged_sources["frn_daily_indexes"]
    workbook_hashes = verify_cbo_workbook_hashes(budget_path, economic_path)
    budget_contract = parse_cbo_budget_source_contract(budget_path)
    economic_contract = parse_cbo_economic_quarterly_source_contract(economic_path)
    mts_table_1_row = parse_mts_table_1_deficit_row(packaged_sources["mts_table_1"])
    mts_table_9_row = parse_mts_table_9_net_interest_row(packaged_sources["mts_table_9_net_interest"])
    dts_tga_closing_balance_row = parse_dts_tga_closing_balance_row(
        packaged_sources["dts_operating_cash_balance"]
    )
    mspd_rows = parse_mspd_debt_bridge_rows(packaged_sources["mspd_table_1"])
    debt_to_penny_row = parse_debt_to_penny_public_debt_row(packaged_sources["debt_to_penny_public_debt"])
    base_curve_rows = parse_base_curve_rows(packaged_sources["treasury_base_curve"])

    fiscal_rows = build_cbo_fiscal_baseline_rows(budget_contract, scenario_id=SCENARIO_ID)
    macro_rows = build_cbo_macro_forecast_path_rows(economic_contract, scenario_id=SCENARIO_ID)
    source_fixture_rows = budget_contract.fixture_rows() + economic_contract.fixture_rows()
    periods = build_simulation_calendar(SIMULATION_START_DATE, SIMULATION_END_DATE, "daily")

    current_fy_splice = build_current_fy_splice_row(
        scenario_id=SCENARIO_ID,
        fiscal_year=2026,
        simulation_start_date=SIMULATION_START_DATE,
        fiscal_actuals_through=FISCAL_ACTUALS_THROUGH,
        actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
        cbo_full_fy_primary_deficit_bil=CBO_FULL_FY2026_PRIMARY_DEFICIT_BIL,
        actual_total_deficit_fytd_bil=_money_value(mts_table_1_row, "current_month_dfct_sur_amt") / 1_000_000_000.0,
        actual_net_interest_fytd_bil=_money_value(mts_table_9_row, "current_fytd_rcpt_outly_amt") / 1_000_000_000.0,
        mts_table_1_selector=RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR,
        mts_table_9_selector=RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR,
        observation_date=FISCAL_ACTUALS_THROUGH,
        available_date=ACTUALS_AVAILABLE_AS_OF,
    )
    current_fy_splice["source_status"] = (
        "mts_table_1_and_table_9_fytd_actuals_spliced_to_cbo_full_year_primary_deficit"
    )

    bridge = calculate_public_debt_bridge_components(
        cbo_actual_federal_debt_held_public_bil=OPENING_CBO_FEDERAL_DEBT_HELD_PUBLIC_BIL,
        debt_to_penny_row=debt_to_penny_row,
        mspd_rows=mspd_rows,
    )

    debt_rows = build_debt_stock_path_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        opening_state_date=OPENING_STATE_DATE,
        opening_cbo_federal_debt_held_public_bil=OPENING_CBO_FEDERAL_DEBT_HELD_PUBLIC_BIL,
        cbo_fy_end_public_debt_targets_bil={
            int(row["fiscal_year"]): float(row["debt_held_public_end_bil"])
            for row in fiscal_rows
            if int(row["fiscal_year"]) >= 2026
        },
        public_nonmarketable_treasury_bil=bridge["public_nonmarketable_treasury_bil"],
        non_treasury_and_definition_residual_bil=bridge["non_treasury_and_definition_residual_bil"],
        observation_date=FISCAL_ACTUALS_THROUGH,
        available_date=ACTUALS_AVAILABLE_AS_OF,
        bridge_method="latest_actual_constant_nominal_by_component",
    )

    primary_deficit_by_fy = {
        int(row["fiscal_year"]): float(row["primary_deficit_bil"])
        for row in fiscal_rows
        if int(row["fiscal_year"]) >= 2026
    }
    primary_deficit_by_fy[2026] = float(current_fy_splice["remaining_cbo_primary_deficit_bil"])
    primary_rows = build_primary_deficit_path_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        primary_deficit_by_fiscal_year_bil=primary_deficit_by_fy,
        source_status="remaining_fy2026_splice_then_cbo_fy2027_2036_primary_deficit",
    )

    base_tga_bil = tga_closing_balance_bil_from_dts_row(dts_tga_closing_balance_row)
    cpi_by_period_end = _cpi_by_period_end(macro_rows, periods)
    base_cpi_level = _macro_level_for_date(macro_rows, date.fromisoformat(TGA_OBSERVATION_DATE))
    operating_cash_rows = build_operating_cash_path_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        base_date=TGA_OBSERVATION_DATE,
        base_balance_bil=base_tga_bil,
        base_inflation_index_level=base_cpi_level,
        inflation_index_by_period_end=cpi_by_period_end,
        inflation_scalar=1.0,
        observation_date=TGA_OBSERVATION_DATE,
        available_date=ACTUALS_AVAILABLE_AS_OF,
        source_status="dts_tga_opening_balance_indexed_by_cbo_cpi_u",
    )
    zero_residual_rows = build_cash_reconciliation_residual_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        cash_reconciliation_residual_bil=0.0,
        source_status="zero_cash_reconciliation_residual_full_horizon_smoke",
    )

    yield_rows = build_yield_curve_surface_rows(
        macro_forecast_rows=macro_rows,
        base_curve_rows=base_curve_rows,
        actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
        output_tenors=(1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    )
    validate_yield_surface_rate_units(yield_rows)
    frn_rate_rows = build_forward_frn_rate_path_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        yield_surface_rows=yield_rows,
        observation_date=FORECAST_PUBLICATION_DATE,
        available_date=FORECAST_PUBLICATION_DATE,
        benchmark_tenor_years=0.25,
    )

    incidence_rows = build_fiscal_incidence_policy_rows(
        scenario_id=SCENARIO_ID,
        signed_net_primary_flow_bil=sum(float(row["primary_deficit_bil"]) for row in primary_rows),
    )
    incidence_result_rows = build_fiscal_incidence_sensitivity_results(incidence_rows)
    net_interest_rows: list[dict[str, Any]] = []
    for row in fiscal_rows:
        net_interest_rows.extend(
            build_net_interest_bridge_rows(
                scenario_id=SCENARIO_ID,
                fiscal_year=int(row["fiscal_year"]),
                components=[],
                source_vintage=FORECAST_NAME,
                cbo_reported_net_interest_bil=float(row["cbo_net_interest_bil"]),
            )
        )

    write_forecast_rows_csv(inputs / "source_fixtures.csv", source_fixture_rows)
    write_forecast_rows_csv(inputs / "tdcsim_cbo_fiscal_baseline.csv", fiscal_rows)
    write_forecast_rows_csv(inputs / "tdcsim_current_fy_splice.csv", [current_fy_splice])
    write_forecast_rows_csv(inputs / "tdcsim_debt_stock_path.csv", debt_rows)
    write_forecast_rows_csv(inputs / "tdcsim_primary_deficit_path.csv", primary_rows)
    write_forecast_rows_csv(inputs / "tdcsim_operating_cash_path.csv", operating_cash_rows)
    write_forecast_rows_csv(inputs / "tdcsim_cash_reconciliation_residual.csv", zero_residual_rows)
    write_forecast_rows_csv(inputs / "tdcsim_macro_forecast_path.csv", macro_rows)
    write_forecast_rows_csv(inputs / "tdcsim_yield_curve_surface.csv", yield_rows)
    write_forecast_rows_csv(inputs / "tdcsim_frn_rate_path.csv", frn_rate_rows)
    write_forecast_rows_csv(inputs / "tdcsim_fiscal_incidence_policy.csv", incidence_rows)
    write_forecast_rows_csv(inputs / "tdcsim_fiscal_incidence_sensitivity_results.csv", incidence_result_rows)
    write_forecast_rows_csv(inputs / "tdcsim_net_interest_bridge.csv", net_interest_rows)

    holder_preferences = default_cbo_holder_preferences(include_private_routes=False)
    validate_holder_preferences_non_degenerate(holder_preferences)
    holder_profile_rows = build_holder_profile_rows(
        scenario_id=SCENARIO_ID,
        holder_preferences=holder_preferences,
        private_subbucket_shares=default_private_subbucket_shares(),
    )
    write_rows_csv_union(inputs / "tdcsim_holder_profile_assumptions.csv", holder_profile_rows)

    initial_portfolio, opening_portfolio_metadata = load_opening_portfolio(
        opening_portfolio_path,
        simulation_start_date=SIMULATION_START_DATE,
        source_manifest=ratewall_manifest_path,
        tips_cpi_detail=tips_cpi_detail_path,
        frn_daily_indexes=frn_daily_indexes_path,
    )
    manifest_entry = source_package["source_files"].get(SOURCE_PACKAGE_INPUTS["ratewall_input_manifest"], {})
    opening_portfolio_metadata["source_manifest_original_sha256"] = manifest_entry.get("original_sha256")
    opening_portfolio_metadata["source_manifest_portable_sha256"] = manifest_entry.get("sha256")
    opening_cb_holdings_bil = _portfolio_total_debt_base(
        initial_portfolio[initial_portfolio["HolderType"].astype(str) == "CB"]
    )
    fed_holdings_rows = build_fed_holdings_path_rows(
        scenario_id=SCENARIO_ID,
        periods=periods,
        opening_state_date=OPENING_STATE_DATE,
        opening_cb_holdings_bil=opening_cb_holdings_bil,
        cbo_fy_end_fed_holdings_bil={
            int(row["fiscal_year"]): float(row["cbo_fed_holdings_end_bil"])
            for row in fiscal_rows
            if int(row["fiscal_year"]) >= 2026
        },
        observation_date=FISCAL_ACTUALS_THROUGH,
        available_date=ACTUALS_AVAILABLE_AS_OF,
        source_status="cbo_table_1_3_fed_holdings_interpolated_to_daily_holder_target",
    )
    initial_portfolio.to_csv(inputs / "tdcsim_opening_portfolio.csv", index=False)
    tips_diagnostics_path = inputs / "tdcsim_opening_tips_indexation_diagnostics.csv"
    write_rows_csv_union(
        tips_diagnostics_path,
        opening_portfolio_metadata.pop("tips_indexation_diagnostic_rows", []),
    )
    opening_portfolio_metadata["tips_indexation_diagnostics_file"] = str(tips_diagnostics_path)
    frn_diagnostics_path = inputs / "tdcsim_opening_frn_indexation_diagnostics.csv"
    write_rows_csv_union(
        frn_diagnostics_path,
        opening_portfolio_metadata.pop("frn_indexation_diagnostic_rows", []),
    )
    opening_portfolio_metadata["frn_indexation_diagnostics_file"] = str(frn_diagnostics_path)
    rollforward_diagnostics_path = inputs / "tdcsim_opening_rollforward_diagnostics.csv"
    write_rows_csv_union(
        rollforward_diagnostics_path,
        opening_portfolio_metadata.pop("opening_rollforward_diagnostic_rows", []),
    )
    opening_portfolio_metadata["opening_rollforward_diagnostics_file"] = str(rollforward_diagnostics_path)
    write_forecast_rows_csv(inputs / "tdcsim_fed_holdings_path.csv", fed_holdings_rows)
    tips_cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id=SCENARIO_ID,
        macro_forecast_rows=macro_rows,
        simulation_start_date=SIMULATION_START_DATE,
        simulation_end_date=SIMULATION_END_DATE,
        opening_reference_cpi=_opening_tips_reference_cpi(initial_portfolio),
        reference_lag_months=3,
        pricing_horizon_end_date=_tips_pricing_horizon_end_date(yield_rows),
        observation_date=FORECAST_PUBLICATION_DATE,
        available_date=FORECAST_PUBLICATION_DATE,
    )
    tips_real_yield_rows = build_tips_real_yield_path_rows(
        scenario_id=SCENARIO_ID,
        yield_surface_rows=yield_rows,
        tips_cpi_path_rows=tips_cpi_rows,
        observation_date=FORECAST_PUBLICATION_DATE,
        available_date=FORECAST_PUBLICATION_DATE,
    )
    write_forecast_rows_csv(inputs / "tdcsim_tips_cpi_path.csv", tips_cpi_rows)
    write_forecast_rows_csv(inputs / "tdcsim_tips_real_yield_path.csv", tips_real_yield_rows)
    baseline_input_paths = _baseline_input_paths(inputs)
    opening_portfolio_metadata = _package_relative_opening_metadata(opening_portfolio_metadata)
    _write_json(inputs / "tdcsim_opening_portfolio_metadata.json", opening_portfolio_metadata)
    manifest = build_manifest(
        output_dir=out,
        inputs_dir=inputs,
        budget_workbook=budget_path,
        economic_workbook=economic_path,
        workbook_hashes=workbook_hashes,
        fiscal_rows=fiscal_rows,
        macro_rows=macro_rows,
        source_fixture_rows=source_fixture_rows,
        current_fy_splice=current_fy_splice,
        bridge=bridge,
        base_curve_rows=base_curve_rows,
        yield_rows=yield_rows,
        frn_rate_rows=frn_rate_rows,
        operating_cash_rows=operating_cash_rows,
        primary_rows=primary_rows,
        incidence_rows=incidence_rows,
        incidence_result_rows=incidence_result_rows,
        net_interest_rows=net_interest_rows,
        holder_profile_rows=holder_profile_rows,
        fed_holdings_rows=fed_holdings_rows,
        tips_cpi_rows=tips_cpi_rows,
        tips_real_yield_rows=tips_real_yield_rows,
        opening_portfolio_metadata=opening_portfolio_metadata,
        source_package=source_package,
    )
    _write_json(out / "manifest.json", manifest)
    _write_json(inputs / "source_contract_smoke.json", manifest)

    params = build_engine_params(inputs, baseline_input_paths, initial_portfolio, base_tga_bil)
    return ForecastBundle(
        output_dir=out,
        inputs_dir=inputs,
        scenario_id=SCENARIO_ID,
        periods=periods,
        manifest=manifest,
        baseline_input_paths=baseline_input_paths,
        initial_portfolio=initial_portfolio,
        params=params,
    )


def run_smoke(bundle: ForecastBundle) -> dict[str, Any]:
    """Run zero-residual and constant-real-TGA full-horizon smoke passes."""

    zero_results, zero_portfolio = run_simulation(
        copy.deepcopy(bundle.params),
        SIMULATION_START_DATE,
        SIMULATION_END_DATE,
        freq="D",
        scenario_name=bundle.scenario_id,
    )
    zero_summary = write_run_evidence(
        zero_results,
        zero_portfolio,
        bundle.output_dir / "exports_zero_residual",
        cash_residual_file=str(bundle.inputs_dir / "tdcsim_cash_reconciliation_residual.csv"),
    )

    constant_residual_rows = derive_constant_real_tga_residual_rows(
        zero_results,
        bundle.inputs_dir / "tdcsim_cash_reconciliation_residual.csv",
    )
    constant_residual_path = bundle.inputs_dir / "tdcsim_cash_reconciliation_residual_constant_real_tga.csv"
    write_forecast_rows_csv(constant_residual_path, constant_residual_rows)

    constant_params = copy.deepcopy(bundle.params)
    constant_params["baseline_input_paths"]["cash_reconciliation_residual_file"] = str(constant_residual_path)
    constant_results, constant_portfolio = run_simulation(
        constant_params,
        SIMULATION_START_DATE,
        SIMULATION_END_DATE,
        freq="D",
        scenario_name=bundle.scenario_id,
    )
    constant_summary = write_run_evidence(
        constant_results,
        constant_portfolio,
        bundle.output_dir / "exports_constant_real_tga",
        cash_residual_file=str(constant_residual_path),
        extra={
            "total_cash_reconciliation_residual_bil": float(
                constant_results["CBOCashReconciliationResidual"].sum()
            ),
            "min_tga_bil": float(constant_results["TGA"].min()),
            "max_tga_bil": float(constant_results["TGA"].max()),
            "residual_derivation": "period_flow_derived_from_zero_residual_tga_path_and_cumulative_prior_residuals",
        },
    )

    nominal_operating_cash_path = bundle.inputs_dir / "tdcsim_operating_cash_path_constant_nominal.csv"
    nominal_operating_cash_rows = derive_constant_nominal_operating_cash_rows(
        bundle.inputs_dir / "tdcsim_operating_cash_path.csv"
    )
    write_forecast_rows_csv(nominal_operating_cash_path, nominal_operating_cash_rows)
    nominal_target_by_period_end = {
        pd.Timestamp(row["period_end"]): float(row["operating_cash_target_bil"])
        for row in nominal_operating_cash_rows
    }
    nominal_residual_rows = derive_target_tga_residual_rows(
        zero_results,
        bundle.inputs_dir / "tdcsim_cash_reconciliation_residual.csv",
        nominal_target_by_period_end=nominal_target_by_period_end,
        source_status="derived_constant_nominal_tga_reconciliation_residual_local_smoke",
    )
    nominal_residual_path = bundle.inputs_dir / "tdcsim_cash_reconciliation_residual_constant_nominal_tga.csv"
    write_forecast_rows_csv(nominal_residual_path, nominal_residual_rows)
    nominal_params = copy.deepcopy(bundle.params)
    nominal_params["baseline_input_paths"]["operating_cash_path_file"] = str(nominal_operating_cash_path)
    nominal_params["baseline_input_paths"]["cash_reconciliation_residual_file"] = str(nominal_residual_path)
    nominal_results, nominal_portfolio = run_simulation(
        nominal_params,
        SIMULATION_START_DATE,
        SIMULATION_END_DATE,
        freq="D",
        scenario_name=bundle.scenario_id,
    )
    nominal_summary = write_run_evidence(
        nominal_results,
        nominal_portfolio,
        bundle.output_dir / "exports_constant_nominal_tga",
        cash_residual_file=str(nominal_residual_path),
        extra={
            "total_cash_reconciliation_residual_bil": float(
                nominal_results["CBOCashReconciliationResidual"].sum()
            ),
            "min_tga_bil": float(nominal_results["TGA"].min()),
            "max_tga_bil": float(nominal_results["TGA"].max()),
            "operating_cash_path_file": _package_relative_path(bundle.output_dir, nominal_operating_cash_path),
            "residual_derivation": "period_flow_derived_from_zero_residual_tga_path_and_constant_nominal_tga_target",
        },
    )

    comparison = build_cash_sensitivity_comparison(
        {
            "zero_residual": zero_results,
            "constant_real_tga": constant_results,
            "constant_nominal_tga": nominal_results,
        }
    )
    write_rows_csv_union(bundle.output_dir / "cash_sensitivity_comparison.csv", comparison)
    net_interest_diagnostic = build_modeled_net_interest_diagnostics(
        zero_results,
        bundle.inputs_dir / "tdcsim_cbo_fiscal_baseline.csv",
        run_id="zero_residual",
    )
    write_rows_csv_union(bundle.output_dir / "tdcsim_partial_interest_financing_diagnostic.csv", net_interest_diagnostic)

    manifest = dict(bundle.manifest)
    manifest["outputs"] = {
        "zero_residual_summary": _package_relative_path(
            bundle.output_dir,
            bundle.output_dir / "exports_zero_residual" / "summary.json",
        ),
        "constant_real_tga_summary": _package_relative_path(
            bundle.output_dir,
            bundle.output_dir / "exports_constant_real_tga" / "summary.json",
        ),
        "constant_real_tga_cash_residual_file": _package_relative_path(bundle.output_dir, constant_residual_path),
        "constant_nominal_tga_summary": _package_relative_path(
            bundle.output_dir,
            bundle.output_dir / "exports_constant_nominal_tga" / "summary.json",
        ),
        "constant_nominal_tga_operating_cash_file": _package_relative_path(bundle.output_dir, nominal_operating_cash_path),
        "constant_nominal_tga_cash_residual_file": _package_relative_path(bundle.output_dir, nominal_residual_path),
        "cash_sensitivity_comparison": _package_relative_path(
            bundle.output_dir,
            bundle.output_dir / "cash_sensitivity_comparison.csv",
        ),
        "partial_interest_financing_diagnostic": _package_relative_path(
            bundle.output_dir,
            bundle.output_dir / "tdcsim_partial_interest_financing_diagnostic.csv"
        ),
    }
    manifest["run_summaries"] = {
        "zero_residual": zero_summary,
        "constant_real_tga": constant_summary,
        "constant_nominal_tga": nominal_summary,
    }
    manifest["artifact_hashes"] = hash_artifacts(bundle.output_dir)
    _write_json(bundle.output_dir / "manifest.json", manifest)
    _write_json(bundle.inputs_dir / "source_contract_smoke.json", manifest)
    return manifest


def build_manifest(
    *,
    output_dir: Path,
    inputs_dir: Path,
    budget_workbook: Path,
    economic_workbook: Path,
    workbook_hashes: Mapping[str, str],
    fiscal_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    source_fixture_rows: Sequence[Mapping[str, Any]],
    current_fy_splice: Mapping[str, Any],
    bridge: Mapping[str, Any],
    base_curve_rows: Sequence[Mapping[str, Any]],
    yield_rows: Sequence[Mapping[str, Any]],
    operating_cash_rows: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    incidence_rows: Sequence[Mapping[str, Any]],
    incidence_result_rows: Sequence[Mapping[str, Any]],
    net_interest_rows: Sequence[Mapping[str, Any]],
    holder_profile_rows: Sequence[Mapping[str, Any]],
    fed_holdings_rows: Sequence[Mapping[str, Any]],
    frn_rate_rows: Sequence[Mapping[str, Any]],
    tips_cpi_rows: Sequence[Mapping[str, Any]],
    tips_real_yield_rows: Sequence[Mapping[str, Any]],
    opening_portfolio_metadata: Mapping[str, Any],
    source_package: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact manifest required by the local smoke contract."""

    fy2026 = _row_by_key(fiscal_rows, "fiscal_year", 2026)
    first_curve = yield_rows[0]
    return {
        "schema_version": "tdcsim_cbo_forecast_smoke_manifest_v1",
        "status": "local_full_horizon_repaired_smoke_not_production_bundle",
        "scenario_id": SCENARIO_ID,
        "forecast_name": FORECAST_NAME,
        "forecast_publication_date": FORECAST_PUBLICATION_DATE,
        "date_range": {
            "simulation_start_date": SIMULATION_START_DATE,
            "simulation_end_date": SIMULATION_END_DATE,
            "opening_state_date": OPENING_STATE_DATE,
            "actuals_available_as_of": ACTUALS_AVAILABLE_AS_OF,
            "fiscal_actuals_through": FISCAL_ACTUALS_THROUGH,
            "lookahead_policy": "no_lookahead_for_treasury_actuals_cbo_forecast_vintage_allowed",
        },
        "source_files": {
            "budget_workbook": f"sources/{SOURCE_PACKAGE_INPUTS['budget_workbook']}",
            "budget_workbook_sha256": workbook_hashes.get("budget", BUDGET_WORKBOOK_SHA256),
            "economic_workbook": f"sources/{SOURCE_PACKAGE_INPUTS['economic_workbook']}",
            "economic_workbook_sha256": workbook_hashes.get("economic", ECONOMIC_WORKBOOK_SHA256),
            "tips_cpi_detail": _package_path_or_portable(output_dir, opening_portfolio_metadata["tips_cpi_detail_source"]),
            "tips_cpi_detail_sha256": opening_portfolio_metadata["tips_cpi_detail_source_sha256"],
            "frn_daily_indexes": _package_path_or_portable(output_dir, opening_portfolio_metadata["frn_daily_indexes_source"]),
            "frn_daily_indexes_sha256": opening_portfolio_metadata["frn_daily_indexes_source_sha256"],
            "base_curve_source_key": first_curve["base_curve_source_key"],
            "base_curve_sha256": first_curve["base_curve_sha256"],
        },
        "source_package": source_package,
        "source_row_selectors": {
            "cbo_budget_required_rows": _selector_map(source_fixture_rows, "cbo_budget_feb_2026_workbook"),
            "cbo_economic_quarterly_rows": _selector_map(
                source_fixture_rows,
                "cbo_economic_feb_2026_workbook",
            ),
            "mts_table_1": RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR,
            "mts_table_9": RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR,
            "debt_to_the_penny": RUNNER_DEBT_TO_PENNY_SELECTOR,
            "mspd": [
                "security_type_desc=Total Nonmarketable",
                "security_type_desc=Total Public Debt Outstanding",
            ],
            "dts": "account_type=Treasury General Account (TGA) Closing Balance",
        },
        "source_values": {
            "fy2026_cbo": {
                "primary_deficit_bil": float(fy2026["primary_deficit_bil"]),
                "net_interest_bil": float(fy2026["cbo_net_interest_bil"]),
                "total_deficit_bil": float(fy2026["cbo_total_deficit_bil"]),
                "debt_held_public_end_bil": float(fy2026["debt_held_public_end_bil"]),
            },
            "current_fy_splice": {
                "actual_total_deficit_fytd_bil": float(current_fy_splice["actual_total_deficit_fytd_bil"]),
                "actual_net_interest_fytd_bil": float(current_fy_splice["actual_net_interest_fytd_bil"]),
                "actual_primary_deficit_fytd_bil": float(
                    current_fy_splice["actual_primary_deficit_fytd_bil"]
                ),
                "remaining_primary_deficit_bil": float(
                    current_fy_splice["remaining_cbo_primary_deficit_bil"]
                ),
            },
            "debt_bridge": {
                "treasury_public_debt_bil": float(bridge["treasury_public_debt_bil"]),
                "debt_to_penny_public_debt_bil": float(bridge["debt_to_penny_public_debt_bil"]),
                "mspd_public_debt_bil": float(bridge["mspd_public_debt_bil"]),
                "mspd_debt_to_penny_residual_bil": float(bridge["mspd_debt_to_penny_residual_bil"]),
                "public_nonmarketable_treasury_bil": float(bridge["public_nonmarketable_treasury_bil"]),
                "non_treasury_and_definition_residual_bil": float(
                    bridge["non_treasury_and_definition_residual_bil"]
                ),
                "opening_marketable_public_portfolio_bil": OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL,
                "record_date": bridge["record_date"],
                "debt_to_penny_record_date": bridge["debt_to_penny_record_date"],
                "mspd_record_date": bridge["mspd_record_date"],
                "date_alignment": bridge["date_alignment"],
            },
            "operating_cash": {
                "dts_tga_opening_balance_bil": float(operating_cash_rows[0]["base_balance_bil"]),
                "base_cpi_u_index": float(operating_cash_rows[0]["inflation_index_level"]),
                "inflation_scalar": float(operating_cash_rows[0]["inflation_scalar"]),
            },
            "row_counts": {
                "period_rows": len(primary_rows),
                "macro_quarter_rows": len(macro_rows),
                "yield_surface_rows": len(yield_rows),
                "incidence_policy_rows": len(incidence_rows),
                "incidence_sensitivity_result_rows": len(incidence_result_rows),
                "net_interest_bridge_rows": len(net_interest_rows),
                "holder_profile_rows": len(holder_profile_rows),
                "fed_holdings_path_rows": len(fed_holdings_rows),
                "frn_rate_path_rows": len(frn_rate_rows),
                "tips_cpi_path_rows": len(tips_cpi_rows),
                "tips_real_yield_path_rows": len(tips_real_yield_rows),
                "opening_portfolio_rows": int(opening_portfolio_metadata["row_count"]),
                "opening_tips_indexation_diagnostic_rows": int(
                    opening_portfolio_metadata["tips_rows_enriched"]
                ),
                "opening_frn_indexation_diagnostic_rows": int(
                    opening_portfolio_metadata["frn_rows_enriched"]
                ),
            },
        },
        "rate_unit_declaration": {
            "cbo_macro_rates": "percent_points",
            "yield_surface_nominal_rate": "percent_points",
            "yield_surface_nominal_rate_decimal": "decimal_runtime_rate",
            "runtime_rate_unit": "decimal",
            "guardrail": "runner_fails_if_nominal_rate_decimal_or_runtime_rate_unit_decimal_is_missing_or_mixed",
        },
        "opening_portfolio_construction": {
            "mode": "prebuilt_mspd_derived_cohort_book_with_explicit_prestart_rollforward",
            "source_file": _package_path_or_portable(output_dir, opening_portfolio_metadata["source_file"]),
            "source_file_sha256": opening_portfolio_metadata["source_file_sha256"],
            "source_manifest": _package_path_or_portable(output_dir, opening_portfolio_metadata["source_manifest"]),
            "source_manifest_sha256": opening_portfolio_metadata.get("source_manifest_sha256"),
            "source_manifest_original_sha256": opening_portfolio_metadata.get("source_manifest_original_sha256"),
            "source_manifest_portable_sha256": opening_portfolio_metadata.get("source_manifest_portable_sha256"),
            "record_date": opening_portfolio_metadata["record_date"],
            "simulation_start_date": SIMULATION_START_DATE,
            "row_count": opening_portfolio_metadata["row_count"],
            "source_rows_loaded": opening_portfolio_metadata["source_rows_loaded"],
            "prestart_rollforward_rows": opening_portfolio_metadata["prestart_rollforward_rows"],
            "prestart_rollforward_face_bil": opening_portfolio_metadata["prestart_rollforward_face_bil"],
            "opening_rollforward_status": opening_portfolio_metadata["opening_rollforward_status"],
            "opening_rollforward_policy": opening_portfolio_metadata["opening_rollforward_policy"],
            "opening_source_common_date_status": opening_portfolio_metadata["opening_source_common_date_status"],
            "opening_source_latest_available_mspd_table3_record_date": opening_portfolio_metadata[
                "opening_source_latest_available_mspd_table3_record_date"
            ],
            "opening_rollforward_source_debt_base_bil": opening_portfolio_metadata[
                "opening_rollforward_source_debt_base_bil"
            ],
            "opening_rollforward_aligned_debt_base_bil": opening_portfolio_metadata[
                "opening_rollforward_aligned_debt_base_bil"
            ],
            "opening_rollforward_debt_base_delta_bil": opening_portfolio_metadata[
                "opening_rollforward_debt_base_delta_bil"
            ],
            "opening_rollforward_diagnostics_file": _package_path_or_portable(
                output_dir,
                opening_portfolio_metadata["opening_rollforward_diagnostics_file"]
            ),
            "opening_tips_reference_cpi": opening_portfolio_metadata["opening_tips_reference_cpi"],
            "tips_cpi_detail_source": _package_path_or_portable(
                output_dir,
                opening_portfolio_metadata["tips_cpi_detail_source"],
            ),
            "tips_cpi_detail_source_sha256": opening_portfolio_metadata["tips_cpi_detail_source_sha256"],
            "tips_cpi_detail_index_date": opening_portfolio_metadata["tips_cpi_detail_index_date"],
            "tips_rows_enriched": opening_portfolio_metadata["tips_rows_enriched"],
            "tips_unique_cusips_enriched": opening_portfolio_metadata["tips_unique_cusips_enriched"],
            "tips_source_ref_cpi_values": opening_portfolio_metadata["tips_source_ref_cpi_values"],
            "opening_tips_adjusted_principal_bil": opening_portfolio_metadata[
                "opening_tips_adjusted_principal_bil"
            ],
            "opening_tips_reconstructed_original_principal_bil": opening_portfolio_metadata[
                "opening_tips_reconstructed_original_principal_bil"
            ],
            "opening_tips_indexation_accretion_embedded_bil": opening_portfolio_metadata[
                "opening_tips_indexation_accretion_embedded_bil"
            ],
            "tips_indexation_diagnostics_file": _package_path_or_portable(
                output_dir,
                opening_portfolio_metadata["tips_indexation_diagnostics_file"]
            ),
            "tips_initialization": opening_portfolio_metadata["tips_initialization"],
            "frn_daily_indexes_source": _package_path_or_portable(
                output_dir,
                opening_portfolio_metadata["frn_daily_indexes_source"],
            ),
            "frn_daily_indexes_source_sha256": opening_portfolio_metadata["frn_daily_indexes_source_sha256"],
            "frn_opening_date": opening_portfolio_metadata["frn_opening_date"],
            "frn_source_available_as_of": opening_portfolio_metadata["frn_source_available_as_of"],
            "frn_rows_enriched": opening_portfolio_metadata["frn_rows_enriched"],
            "frn_unique_cusips_enriched": opening_portfolio_metadata["frn_unique_cusips_enriched"],
            "frn_source_record_dates": opening_portfolio_metadata["frn_source_record_dates"],
            "frn_source_accrual_end_dates": opening_portfolio_metadata["frn_source_accrual_end_dates"],
            "opening_frn_face_value_bil": opening_portfolio_metadata["opening_frn_face_value_bil"],
            "opening_frn_accrued_interest_bil": opening_portfolio_metadata[
                "opening_frn_accrued_interest_bil"
            ],
            "opening_frn_benchmark_rate_decimal_min": opening_portfolio_metadata[
                "opening_frn_benchmark_rate_decimal_min"
            ],
            "opening_frn_benchmark_rate_decimal_max": opening_portfolio_metadata[
                "opening_frn_benchmark_rate_decimal_max"
            ],
            "opening_frn_fixed_spread_decimal_min": opening_portfolio_metadata[
                "opening_frn_fixed_spread_decimal_min"
            ],
            "opening_frn_fixed_spread_decimal_max": opening_portfolio_metadata[
                "opening_frn_fixed_spread_decimal_max"
            ],
            "frn_indexation_diagnostics_file": _package_path_or_portable(
                output_dir,
                opening_portfolio_metadata["frn_indexation_diagnostics_file"]
            ),
            "frn_source_boundary_note": opening_portfolio_metadata["frn_source_boundary_note"],
            "frn_initialization": opening_portfolio_metadata["frn_initialization"],
            "security_type_totals_bil": opening_portfolio_metadata["security_type_totals_bil"],
            "holder_totals_bil": opening_portfolio_metadata["holder_totals_bil"],
            "opening_cb_holdings_bil": float(opening_portfolio_metadata["holder_totals_bil"].get("CB", 0.0)),
            "claim_boundary": OPENING_PORTFOLIO_CLAIM_BOUNDARY,
        },
        "issuance_mix": {
            "mode": "scenario_assumption_mixed_marketable_security_profile",
            "fixed_remainder_shares": {"bills": 0.25, "notes": 0.55, "bonds": 0.20},
            "bills": {"0.5y": 1.0},
            "notes": {"5.0y": 1.0},
            "bonds": {"20.0y": 1.0},
            "tips": {"target_percentage": 0.06, "10.0y": 1.0},
            "frn": {"target_percentage": 0.04, "2.0y": 1.0},
            "nonmarketable": 0.0,
        },
        "negative_issuance_policy": {
            "negative_required_issuance_action": "retire_shortest_public_marketable",
            "cash_treatment": "par_cash_buyback_or_early_retirement_drains_tga",
            "selection_order": "shortest_public_marketable_maturity_first",
            "runtime_columns": ["CBOBuybackFaceRetired", "CBOBuybackCashPaid"],
        },
        "holder_preferences": _holder_preferences(),
        "fed_holdings_path": {
            "mode": "cbo_fed_holdings_endpoints_linear_actual_days_holder_target",
            "runtime_effect": "holder_allocation_only_total_issuance_unchanged",
            "runtime_columns": [
                "CBOFedHoldingsTarget",
                "CBOFedHoldingsTargetError",
                "CBOFedAuctionShare",
            ],
            "claim_boundary": "fed_holdings_path_guides_cb_absorption_not_total_debt_or_cash",
        },
        "tips_forward_paths": {
            "cpi_file": "forecast_inputs/tdcsim_tips_cpi_path.csv",
            "real_yield_file": "forecast_inputs/tdcsim_tips_real_yield_path.csv",
            "cpi_mode": (
                "monthly_cbo_cpi_u_interpolation_through_cbo_horizon_then_tdcsim_terminal_extrapolation"
            ),
            "reference_cpi_lag_months": 3,
            **_tips_terminal_cpi_manifest(tips_cpi_rows, tips_real_yield_rows),
            "real_yield_mode": (
                "nominal_cbo_yield_surface_less_tdcsim_expected_inflation_with_declared_terminal_cpi_rule"
            ),
            "auction_pricing": "real_cashflow_present_value_with_coupon_floor_rounding",
            "runtime_effect": "TIPS indexation and TIPS auction proceeds only; not a cash, issuance, or net-interest plug",
            "claim_boundary": (
                "future_tips_cpi_and_real_yields_are_explicit_forecast_assumptions_not_observed_future_auction_or_cpi_actuals;"
                "cbo_cpi_coverage_ends_at_terminal_cbo_anchor_month_post_horizon_cpi_is_tdcsim_extrapolation"
            ),
        },
        "frn_forward_rate_path": {
            "mode": "contract_mechanics_with_explicit_cbo_3m_forward_rate_assumption",
            "file": "forecast_inputs/tdcsim_frn_rate_path.csv",
            "benchmark_tenor_years": 0.25,
            "day_count_basis": 360.0,
            "lockout_business_days": 2.0,
            "runtime_effect": "FRN benchmark accrual only; not a cash, issuance, or net-interest plug",
            "source_status": "future_13week_high_rates_are_unobserved_and_anchored_to_cbo_3m_tbill_path",
            "claim_boundary": (
                "opening_frn_terms_are_source_actuals_future_resets_are_explicit_forecast_assumptions"
            ),
        },
        "operating_cash_construction": {
            "operating_cash_definition": "tga_only",
            "reserve_settlement_component": "tga",
            "construction_mode": "constant_real_cbo_cpi_u",
            "base_date": TGA_OBSERVATION_DATE,
            "base_balance_source": "DTS Treasury General Account (TGA) Closing Balance",
            "inflation_index": "cbo_cpi_u",
        },
        "scope_claim_id": CLAIM_BOUNDARY,
        "claim_boundary": {
            "schema_version": "tdcsim_cbo_package_claim_boundary_v1",
            "package_role": "cbo_baseline_forecast_evidence_package",
            "net_interest_role": "partial_interest_financing_diagnostic_only",
            "issuance_mix_role": "tdcsim_assumption_not_cbo_prescription",
            "cash_role": "operating_cash_sensitivity_only_not_debt_or_issuance_driver",
            "excluded_claims": sorted(
                [
                    "does_not_claim_exact_historical_replay",
                    "does_not_claim_full_soma_transaction_replay",
                    "does_not_claim_budgetary_net_interest_replication",
                    "does_not_claim_generated_evidence_tracked_in_git",
                    "does_not_claim_receipts_outlays_decomposition",
                    "does_not_use_cbo_net_interest_as_cash_or_issuance_plug",
                    "does_not_claim_cbo_issuance_mix",
                    "does_not_model_remittances_or_monetary_settlement_effects",
                ]
            ),
        },
        "trust_anchor": _local_trust_anchor(),
        "artifacts": {
            "output_dir": ".",
            "forecast_inputs_dir": "forecast_inputs",
            "manifest": "manifest.json",
        },
    }


def _local_trust_anchor() -> dict[str, Any]:
    verifier_path = PROJECT_ROOT / "scripts" / "verify_cbo_forecast_package.py"
    return {
        "schema_version": "tdcsim_cbo_package_trust_anchor_v1",
        "release_commit_status": "pending_until_release_commit",
        "release_commit_sha": "pending_until_release_commit",
        "package_zip_sha256_status": "pending_until_release_commit",
        "package_zip_sha256": "pending_until_release_commit",
        "code_revision": _git_output(["rev-parse", "HEAD"]),
        "dirty_state": bool(_git_output(["status", "--short"])),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "verifier_version": "tdcsim_cbo_package_verifier_semantic_v2",
        "verifier_sha256": _sha256_file(verifier_path) if verifier_path.exists() else "",
        "verifier_runtime": platform.python_implementation(),
        "dependency_lock_status": "requirements_lock_txt_present",
        "requirements_sha256": _sha256_file(PROJECT_ROOT / "requirements.txt")
        if (PROJECT_ROOT / "requirements.txt").exists()
        else "",
        "requirements_dev_sha256": _sha256_file(PROJECT_ROOT / "requirements-dev.txt")
        if (PROJECT_ROOT / "requirements-dev.txt").exists()
        else "",
        "requirements_lock_sha256": _sha256_file(PROJECT_ROOT / "requirements.lock.txt")
        if (PROJECT_ROOT / "requirements.lock.txt").exists()
        else "",
    }


def _git_output(args: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def validate_yield_surface_rate_units(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("yield surface rows must not be empty")
    rate_units = {str(row.get("rate_unit", "")) for row in rows}
    runtime_units = {str(row.get("runtime_rate_unit", "")) for row in rows}
    if rate_units != {"percent_points"}:
        raise ValueError(f"yield surface rate_unit must be percent_points; found {sorted(rate_units)}")
    if runtime_units != {"decimal"}:
        raise ValueError(f"yield surface runtime_rate_unit must be decimal; found {sorted(runtime_units)}")
    for row in rows:
        if "nominal_rate_decimal" not in row:
            raise ValueError("yield surface rows must include nominal_rate_decimal")
        nominal_rate = float(row["nominal_rate"])
        decimal_rate = float(row["nominal_rate_decimal"])
        if abs(decimal_rate - nominal_rate / 100.0) > 1e-12:
            raise ValueError("yield surface nominal_rate_decimal must equal nominal_rate / 100")
        if abs(decimal_rate) > 1.0:
            raise ValueError("yield surface nominal_rate_decimal is outside plausible decimal bounds")


def derive_constant_real_tga_residual_rows(
    zero_results: pd.DataFrame,
    zero_residual_path: str | Path,
) -> list[dict[str, Any]]:
    target_by_period_end = {
        pd.Timestamp(index): float(row["CBOOperatingCashTarget"])
        for index, row in zero_results.iterrows()
    }
    return derive_target_tga_residual_rows(
        zero_results,
        zero_residual_path,
        nominal_target_by_period_end=target_by_period_end,
        source_status="derived_constant_real_tga_reconciliation_residual_local_smoke",
    )


def derive_target_tga_residual_rows(
    zero_results: pd.DataFrame,
    zero_residual_path: str | Path,
    *,
    nominal_target_by_period_end: Mapping[pd.Timestamp, float],
    source_status: str,
) -> list[dict[str, Any]]:
    residual_rows = pd.read_csv(zero_residual_path).to_dict("records")
    cumulative_residual = 0.0
    for row in residual_rows:
        period_end = pd.Timestamp(row["period_end"])
        target = float(nominal_target_by_period_end[period_end])
        zero_tga = float(zero_results.loc[period_end, "TGA"])
        period_residual = target - zero_tga - cumulative_residual
        row["cash_reconciliation_residual_bil"] = period_residual
        row["source_status"] = source_status
        cumulative_residual += period_residual
    return residual_rows


def derive_constant_nominal_operating_cash_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = pd.read_csv(path).to_dict("records")
    if not rows:
        raise ValueError("operating cash path is empty")
    base_balance = float(rows[0]["base_balance_bil"])
    for row in rows:
        row["operating_cash_target_bil"] = base_balance
        row["tga_target_bil"] = base_balance
        row["ttl_target_bil"] = 0.0
        row["other_operating_cash_target_bil"] = 0.0
        row["construction_mode"] = "constant_nominal"
        row["inflation_scalar"] = 0.0
        row["source_status"] = "dts_tga_opening_balance_constant_nominal_sensitivity"
    return rows


def build_cash_sensitivity_comparison(results_by_name: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    baseline = results_by_name["zero_residual"]
    rows = []
    invariant_columns = [
        "CBOControlledDebtTarget",
        "CBORequiredFaceIssuance",
        "NewDebtIssued",
        "AuctionProceeds",
        "DebtHeld_Banks",
        "DebtHeld_CentralBank",
        "DebtHeld_Foreign",
        "DebtHeld_DomesticNonBanks",
    ]
    for name, results in results_by_name.items():
        row: dict[str, Any] = {"run_id": name, "rows": int(len(results))}
        for column in invariant_columns:
            row[f"max_abs_delta_{column}"] = float((results[column] - baseline[column]).abs().max())
        row["final_tga_bil"] = float(results.iloc[-1]["TGA"])
        row["final_operating_cash_target_bil"] = float(results.iloc[-1]["CBOOperatingCashTarget"])
        row["net_cash_reconciliation_residual_bil"] = float(results["CBOCashReconciliationResidual"].sum())
        row["gross_abs_cash_reconciliation_residual_bil"] = float(results["CBOCashReconciliationResidual"].abs().sum())
        rows.append(row)
    return rows


def build_modeled_net_interest_diagnostics(
    results: pd.DataFrame,
    fiscal_baseline_path: str | Path,
    *,
    run_id: str,
) -> list[dict[str, Any]]:
    """Build annual partial interest-financing diagnostics without creating a CBO plug."""

    compact = results.reset_index().rename(columns={"index": "Date"}).copy()
    compact["Date"] = pd.to_datetime(compact["Date"])
    compact["fiscal_year"] = compact["Date"].dt.year + (compact["Date"].dt.month >= 10).astype(int)
    fiscal_rows = pd.read_csv(fiscal_baseline_path).to_dict("records")
    cbo_by_fy = {
        int(row["fiscal_year"]): float(row["cbo_net_interest_bil"])
        for row in fiscal_rows
        if str(row.get("scenario_id", SCENARIO_ID)) == SCENARIO_ID
    }
    output_rows: list[dict[str, Any]] = []
    for fiscal_year, group in compact.groupby("fiscal_year", sort=True):
        fiscal_year_int = int(fiscal_year)
        if fiscal_year_int not in cbo_by_fy:
            continue
        coupon_cash = float(group["InterestPaid_Bonds"].sum())
        bill_discount = float(group["IssueDiscountCost_Period"].sum())
        tips_accretion = float(group["TIPSInflationAccretion_Period"].sum())
        nonmarketable_capitalized = float(group["NonMarketableInterestCapitalized_Period"].sum())
        interest_related_proxy = coupon_cash + bill_discount + tips_accretion + nonmarketable_capitalized
        diagnostic = build_net_interest_diagnostic(
            cbo_net_interest_bil=cbo_by_fy[fiscal_year_int],
            modeled_net_interest_bil=interest_related_proxy,
            fiscal_year=fiscal_year_int,
            scope_status="partial_market_securities_proxy",
            calibration_mode="diagnostic_only_no_cbo_interest_plug",
        )
        diagnostic.pop("modeled_net_interest_bil", None)
        average_debt = float(group["TotalDebt_Agg"].mean())
        diagnostic.update(
            {
                "run_id": run_id,
                "fiscal_year_start": str(group["Date"].min().date()),
                "fiscal_year_end": str(group["Date"].max().date()),
                "rows": int(len(group)),
                "coupon_cash_interest_bil": coupon_cash,
                "issuance_discount_booked_at_issue_proxy_bil": bill_discount,
                "tips_principal_indexation_bil": tips_accretion,
                "nonmarketable_interest_capitalized_bil": nonmarketable_capitalized,
                "modeled_interest_related_proxy_bil": interest_related_proxy,
                "average_total_debt_bil": average_debt,
                "modeled_average_interest_rate_pct": (
                    interest_related_proxy / average_debt * 100.0 if average_debt > 0.0 else 0.0
                ),
                "runtime_role": "diagnostic_only",
                "scope_completeness_status": "partial_interest_related_proxy_not_budgetary_net_interest",
                "numeric_residual_status": diagnostic["threshold_status"],
                "claim_status": "partial_nonbinding_diagnostic_regardless_of_numeric_residual",
                "source_status": (
                    "modeled_tdcsim_interest_flows_compared_to_cbo_net_interest;"
                    "partial_scope_excludes_offsetting_receipts_and_full_budgetary_interest_mapping"
                ),
                "claim_boundary": "no_cbo_net_interest_cash_or_issuance_plug",
            }
        )
        output_rows.append(diagnostic)
    return output_rows


def write_run_evidence(
    results: pd.DataFrame,
    portfolio: pd.DataFrame,
    outdir: str | Path,
    *,
    cash_residual_file: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    compact = results.reset_index().rename(columns={"index": "Date"})
    missing_columns = [column for column in RESULT_COLUMNS if column not in compact.columns]
    if missing_columns:
        raise ValueError(f"compact CBO output missing required columns: {missing_columns}")
    compact[RESULT_COLUMNS].to_csv(out / "results_compact.csv", index=False)

    active = portfolio[portfolio["Status"] == "Active"].copy()
    active.to_csv(out / "final_portfolio_active.csv", index=False)
    matured = portfolio[portfolio["Status"] == "Matured"].copy()
    matured.to_csv(out / "maturity_ledger.csv", index=False)
    if not active.empty:
        active_debt_base = active["FaceValue"].copy()
        tips_mask = active["SecurityType"].astype(str).eq("TIPS")
        active_debt_base.loc[tips_mask] = active.loc[tips_mask, "AdjustedPrincipal"].fillna(
            active.loc[tips_mask, "FaceValue"]
        )
        active = active.assign(DebtBase=active_debt_base)
    summary = {
        "scenario_id": SCENARIO_ID,
        "rows": int(len(results)),
        "start_date": str(results.index.min().date()),
        "final_date": str(results.index.max().date()),
        "final_total_debt_bil": float(results.iloc[-1]["TotalDebt_Agg"]),
        "final_cbo_target_bil": float(results.iloc[-1]["CBOControlledDebtTarget"]),
        "max_abs_target_error_bil": float(results["CBOControlledDebtTargetError"].abs().max()),
        "total_required_face_issuance_bil": float(results["CBORequiredFaceIssuance"].sum()),
        "total_auction_proceeds_bil": float(results["AuctionProceeds"].sum()),
        "weighted_proceeds_to_face": float(
            results["AuctionProceeds"].sum() / results["CBORequiredFaceIssuance"].sum()
        ),
        "total_issue_discount_cost_bil": float(results["IssueDiscountCost_Period"].sum()),
        "total_primary_deficit_bil": float(results["PrimaryDeficit"].sum()),
        "final_tga_bil": float(results.iloc[-1]["TGA"]),
        "final_operating_cash_target_bil": float(results.iloc[-1]["CBOOperatingCashTarget"]),
        "final_cash_residual_bil": float(results.iloc[-1]["CBOCashResidual"]),
        "net_cash_reconciliation_residual_bil": float(results["CBOCashReconciliationResidual"].sum()),
        "gross_abs_cash_reconciliation_residual_bil": float(results["CBOCashReconciliationResidual"].abs().sum()),
        "positive_cash_reconciliation_residual_bil": float(
            results.loc[results["CBOCashReconciliationResidual"] > 0.0, "CBOCashReconciliationResidual"].sum()
        ),
        "negative_cash_reconciliation_residual_bil": float(
            results.loc[results["CBOCashReconciliationResidual"] < 0.0, "CBOCashReconciliationResidual"].sum()
        ),
        "max_abs_daily_cash_reconciliation_residual_bil": float(
            results["CBOCashReconciliationResidual"].abs().max()
        ),
        "nonzero_cash_reconciliation_residual_days": int(
            (results["CBOCashReconciliationResidual"].abs() > 1e-12).sum()
        ),
        "final_cbo_fed_holdings_target_bil": (
            float(results.iloc[-1]["CBOFedHoldingsTarget"])
            if "CBOFedHoldingsTarget" in results.columns
            else 0.0
        ),
        "final_cbo_fed_holdings_error_bil": (
            float(results.iloc[-1]["CBOFedHoldingsTargetError"])
            if "CBOFedHoldingsTargetError" in results.columns
            else 0.0
        ),
        "max_abs_cbo_fed_holdings_error_bil": (
            float(results["CBOFedHoldingsTargetError"].abs().max())
            if "CBOFedHoldingsTargetError" in results.columns
            else 0.0
        ),
        "matured_rows_in_final_portfolio": int((portfolio["Status"] == "Matured").sum()),
        "maturity_ledger_file": _package_relative_path(out.parent, out / "maturity_ledger.csv"),
        "active_rows_in_final_portfolio": int((portfolio["Status"] == "Active").sum()),
        "active_holder_types": sorted(active["HolderType"].dropna().astype(str).unique().tolist()),
        "active_security_types": sorted(active["SecurityType"].dropna().astype(str).unique().tolist()),
        "final_debt_by_holder_bil": (
            {str(k): float(v) for k, v in active.groupby("HolderType")["DebtBase"].sum().to_dict().items()}
            if not active.empty
            else {}
        ),
        "final_debt_by_security_type_bil": (
            {str(k): float(v) for k, v in active.groupby("SecurityType")["DebtBase"].sum().to_dict().items()}
            if not active.empty
            else {}
        ),
        "total_cbo_buyback_face_retired_bil": (
            float(results["CBOBuybackFaceRetired"].sum())
            if "CBOBuybackFaceRetired" in results.columns
            else 0.0
        ),
        "tips_inflation_accretion_cumulative_bil": (
            float(results.iloc[-1]["TIPSInflationAccretion_Cumulative"])
            if "TIPSInflationAccretion_Cumulative" in results.columns
            else 0.0
        ),
        "cash_residual_file": _package_relative_path(out.parent, cash_residual_file),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if extra:
        summary.update(dict(extra))
    _write_json(out / "summary.json", summary)
    return summary


def load_opening_portfolio(
    path: str | Path,
    *,
    simulation_start_date: str,
    source_manifest: str | Path = DEFAULT_RATEWALL_INPUT_MANIFEST,
    tips_cpi_detail: str | Path = DEFAULT_TIPS_CPI_DETAIL,
    frn_daily_indexes: str | Path = DEFAULT_FRN_DAILY_INDEXES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the tracked source-backed cohort book and align it to run start."""

    portfolio_path = Path(path)
    if not portfolio_path.exists():
        raise FileNotFoundError(f"opening portfolio file is missing: {portfolio_path}")
    source = pd.read_csv(portfolio_path)
    source_rows_loaded = int(len(source))
    for column in BOND_PORTFOLIO_COLS:
        if column not in source.columns:
            source[column] = pd.NA
    portfolio = source[BOND_PORTFOLIO_COLS].copy()
    if "cusip_or_synthetic_id" in source.columns:
        source_cusips = source["cusip_or_synthetic_id"]
    elif "CUSIP" in source.columns:
        source_cusips = source["CUSIP"]
    else:
        source_cusips = pd.Series([""] * len(source), index=source.index)
    portfolio["_SourceCUSIP"] = source_cusips.astype(str)
    for date_column in (
        "IssueDate",
        "MaturityDate",
        "DatedDate",
        "OriginalDatedDate",
        "FirstInterestPaymentDate",
        "LastAccrualDate",
    ):
        portfolio[date_column] = pd.to_datetime(portfolio[date_column], errors="coerce")
    for numeric_column in (
        "OriginalMaturityYears",
        "FaceValue",
        "CouponRate",
        "OriginalPrincipal",
        "AdjustedPrincipal",
        "ReferenceCPI_Issue",
        "IndexRatio",
        "FixedSpread",
        "AccruedInterest_FRN",
        "BenchmarkRate_FRN",
        "IssuePriceRatio",
        "IssueProceeds",
        "IssueYieldAtIssue",
    ):
        portfolio[numeric_column] = pd.to_numeric(portfolio[numeric_column], errors="coerce")
    portfolio["Status"] = portfolio["Status"].fillna("Active")
    start = pd.Timestamp(simulation_start_date)
    active = portfolio[portfolio["Status"].astype(str).eq("Active")].copy()
    expired = active[active["MaturityDate"] <= start].copy()
    kept = active[active["MaturityDate"] > start].copy()
    rollforward = _prestart_rollforward_rows(expired, start, next_bond_id=_next_bond_id(portfolio))
    aligned = pd.concat([kept, rollforward], ignore_index=True)
    source_active_debt_base = _portfolio_total_debt_base(active.drop(columns=["_SourceCUSIP"], errors="ignore"))
    aligned, tips_metadata, tips_diagnostics = enrich_opening_tips_from_cpi_detail(
        aligned,
        cpi_detail_path=tips_cpi_detail,
        index_date=simulation_start_date,
    )
    aligned, frn_metadata, frn_diagnostics = enrich_opening_frn_from_daily_indexes(
        aligned,
        frn_daily_indexes_path=frn_daily_indexes,
        opening_date=simulation_start_date,
        source_available_as_of=FRN_INDEX_SOURCE_AVAILABLE_AS_OF,
    )
    aligned = _apply_opening_coupon_terms(aligned)
    aligned = aligned.drop(columns=["_SourceCUSIP"], errors="ignore")
    aligned = aligned[BOND_PORTFOLIO_COLS]
    aligned = aligned.astype(PORTFOLIO_DTYPES, errors="ignore")

    debt_total = _portfolio_total_debt_base(aligned)
    if abs(debt_total - OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL) > 1e-6:
        raise ValueError(
            "opening portfolio does not reconcile to MSPD Table 1 Total Marketable: "
            f"{debt_total:.12f} vs {OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL:.12f}"
        )

    source_manifest_path = Path(source_manifest)
    source_manifest_hash = _sha256_file(source_manifest_path) if source_manifest_path.exists() else None
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    manifest_initial = source_manifest.get("artifacts", {}).get("initial_portfolio", {})
    record_date = str(manifest_initial.get("mspd_record_date", OPENING_PORTFOLIO_RECORD_DATE))
    if record_date != OPENING_PORTFOLIO_RECORD_DATE:
        raise ValueError(f"opening portfolio record date changed unexpectedly: {record_date}")
    rollforward_diagnostics = _opening_rollforward_diagnostic_rows(
        expired,
        rollforward,
        source_record_date=record_date,
        simulation_start_date=simulation_start_date,
    )
    rollforward_debt_delta = float(debt_total - source_active_debt_base)

    metadata = {
        "source_file": str(portfolio_path),
        "source_file_sha256": _sha256_file(portfolio_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "source_url_table_1": manifest_initial.get("source_url_table_1"),
        "source_url_table_3_market": manifest_initial.get("source_url_table_3_market"),
        "record_date": record_date,
        "simulation_start_date": simulation_start_date,
        "source_rows_loaded": source_rows_loaded,
        "row_count": int(len(aligned)),
        "prestart_rollforward_rows": int(len(rollforward)),
        "prestart_rollforward_face_bil": float(expired["FaceValue"].fillna(0.0).sum()),
        "opening_rollforward_status": "mechanically_balanced_source_record_to_simulation_start",
        "opening_rollforward_policy": (
            "active_source_positions_maturing_on_or_before_simulation_start_are_replaced_with_3_month_bills_by_holder"
        ),
        "opening_source_common_date_status": (
            "no_same_day_cusip_level_mspd_table3_book_available_locally_for_simulation_start; "
            "latest_local_mspd_table3_record_date_is_2026-05-31"
        ),
        "opening_source_latest_available_mspd_table3_record_date": OPENING_PORTFOLIO_RECORD_DATE,
        "opening_rollforward_source_debt_base_bil": source_active_debt_base,
        "opening_rollforward_aligned_debt_base_bil": debt_total,
        "opening_rollforward_debt_base_delta_bil": rollforward_debt_delta,
        "opening_rollforward_diagnostic_rows": rollforward_diagnostics,
        **tips_metadata,
        "tips_cpi_detail_source_sha256": _sha256_file(tips_cpi_detail),
        "tips_indexation_diagnostic_rows": tips_diagnostics,
        **frn_metadata,
        "frn_daily_indexes_source_sha256": _sha256_file(frn_daily_indexes),
        "frn_indexation_diagnostic_rows": frn_diagnostics,
        "security_type_totals_bil": _portfolio_security_totals(aligned),
        "holder_totals_bil": _portfolio_holder_totals(aligned),
        "claim_boundary": OPENING_PORTFOLIO_CLAIM_BOUNDARY,
    }
    return aligned, metadata


def _prestart_rollforward_rows(expired: pd.DataFrame, start: pd.Timestamp, *, next_bond_id: int) -> pd.DataFrame:
    if expired.empty:
        return pd.DataFrame(columns=BOND_PORTFOLIO_COLS)
    replacement_rows = []
    maturity_date = start + pd.DateOffset(months=3)
    for offset, (_, row) in enumerate(expired.iterrows()):
        replacement = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
        face_value = float(row.get("FaceValue", 0.0) or 0.0)
        replacement.update(
            {
                "BondID": next_bond_id + offset,
                "SecurityType": "Fixed",
                "IssueDate": start,
                "MaturityDate": maturity_date,
                "DatedDate": start,
                "OriginalDatedDate": start,
                "FirstInterestPaymentDate": pd.NaT,
                "InterestPaymentFrequency": pd.NA,
                "OriginalMaturityYears": 0.25,
                "FaceValue": face_value,
                "CouponRate": 0.0,
                "HolderType": row.get("HolderType", "Private"),
                "HolderSubBucket": row.get("HolderSubBucket", ""),
                "Status": "Active",
                "MaturityCategory": "bills",
                "OriginalPrincipal": 0.0,
                "AdjustedPrincipal": 0.0,
                "ReferenceCPI_Issue": 0.0,
                "IndexRatio": 0.0,
                "FixedSpread": 0.0,
                "AccruedInterest_FRN": 0.0,
                "BenchmarkRate_FRN": 0.0,
                "LastAccrualDate": pd.NaT,
                "IssuePriceRatio": pd.NA,
                "IssueProceeds": pd.NA,
                "IssueYieldAtIssue": pd.NA,
            }
        )
        replacement_rows.append(replacement)
    return pd.DataFrame(replacement_rows, columns=BOND_PORTFOLIO_COLS)


def _opening_rollforward_diagnostic_rows(
    expired: pd.DataFrame,
    rollforward: pd.DataFrame,
    *,
    source_record_date: str,
    simulation_start_date: str,
) -> list[dict[str, Any]]:
    source_total = float(expired["FaceValue"].fillna(0.0).sum()) if not expired.empty else 0.0
    replacement_total = float(rollforward["FaceValue"].fillna(0.0).sum()) if not rollforward.empty else 0.0
    rows: list[dict[str, Any]] = [
        {
            "row_type": "summary",
            "source_record_date": source_record_date,
            "simulation_start_date": simulation_start_date,
            "source_rows_maturing_on_or_before_start": int(len(expired)),
            "replacement_rows": int(len(rollforward)),
            "source_face_bil": source_total,
            "replacement_face_bil": replacement_total,
            "face_delta_bil": replacement_total - source_total,
            "status": "balanced" if abs(replacement_total - source_total) <= 1e-9 else "unbalanced",
            "derivation": (
                "source_MSPD_cohort_rows_maturing_on_or_before_simulation_start_are_replaced_by_3_month_bills"
            ),
            "claim_boundary": "mechanical_rollforward_not_same_day_cusip_level_source_reconstruction",
        }
    ]
    if expired.empty:
        return rows
    replacement_by_offset = rollforward.reset_index(drop=True)
    for offset, (_, source_row) in enumerate(expired.reset_index(drop=True).iterrows()):
        replacement = replacement_by_offset.iloc[offset] if offset < len(replacement_by_offset.index) else {}
        source_face = float(source_row.get("FaceValue", 0.0) or 0.0)
        replacement_face = float(replacement.get("FaceValue", 0.0) or 0.0)
        rows.append(
            {
                "row_type": "replacement",
                "source_record_date": source_record_date,
                "simulation_start_date": simulation_start_date,
                "source_bond_id": source_row.get("BondID", ""),
                "replacement_bond_id": replacement.get("BondID", ""),
                "source_cusip": source_row.get("_SourceCUSIP", ""),
                "source_security_type": source_row.get("SecurityType", ""),
                "replacement_security_type": replacement.get("SecurityType", ""),
                "holder_type": source_row.get("HolderType", ""),
                "holder_subbucket": source_row.get("HolderSubBucket", ""),
                "source_maturity_date": _iso_date_or_empty(source_row.get("MaturityDate")),
                "replacement_issue_date": _iso_date_or_empty(replacement.get("IssueDate")),
                "replacement_maturity_date": _iso_date_or_empty(replacement.get("MaturityDate")),
                "source_face_bil": source_face,
                "replacement_face_bil": replacement_face,
                "face_delta_bil": replacement_face - source_face,
                "status": "balanced" if abs(replacement_face - source_face) <= 1e-9 else "unbalanced",
                "derivation": "holder_preserving_3_month_bill_replacement_for_prestart_maturity",
                "claim_boundary": "mechanical_rollforward_not_same_day_cusip_level_source_reconstruction",
            }
        )
    return rows


def _iso_date_or_empty(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).date().isoformat()


def _next_bond_id(portfolio: pd.DataFrame) -> int:
    bond_ids = pd.to_numeric(portfolio.get("BondID"), errors="coerce")
    if bond_ids.dropna().empty:
        return 1
    return int(bond_ids.max()) + 1


def _portfolio_total_debt_base(portfolio: pd.DataFrame) -> float:
    if portfolio.empty:
        return 0.0
    tips = portfolio["SecurityType"].astype(str).eq("TIPS")
    face = pd.to_numeric(portfolio["FaceValue"], errors="coerce").fillna(0.0)
    adjusted = pd.to_numeric(portfolio["AdjustedPrincipal"], errors="coerce").fillna(face)
    return float(face.mask(tips, adjusted).sum())


def _portfolio_security_totals(portfolio: pd.DataFrame) -> dict[str, float]:
    totals: dict[str, float] = {}
    for security_type, rows in portfolio.groupby(portfolio["SecurityType"].astype(str)):
        totals[security_type] = _portfolio_total_debt_base(rows)
    return totals


def _portfolio_holder_totals(portfolio: pd.DataFrame) -> dict[str, float]:
    totals: dict[str, float] = {}
    for holder, rows in portfolio.groupby(portfolio["HolderType"].astype(str)):
        totals[holder] = _portfolio_total_debt_base(rows)
    return totals


def _apply_opening_coupon_terms(portfolio: pd.DataFrame) -> pd.DataFrame:
    out = portfolio.copy()
    for column in ("DatedDate", "OriginalDatedDate", "FirstInterestPaymentDate"):
        if column not in out.columns:
            out[column] = pd.NaT
        out[column] = pd.to_datetime(out[column], errors="coerce")
    if "InterestPaymentFrequency" not in out.columns:
        out["InterestPaymentFrequency"] = pd.NA
    out["InterestPaymentFrequency"] = pd.to_numeric(out["InterestPaymentFrequency"], errors="coerce")
    issue = pd.to_datetime(out["IssueDate"], errors="coerce")
    maturity = pd.to_datetime(out["MaturityDate"], errors="coerce")
    out["DatedDate"] = out["DatedDate"].fillna(issue)
    out["OriginalDatedDate"] = out["OriginalDatedDate"].fillna(out["DatedDate"])
    security_type = out["SecurityType"].astype(str)
    coupon_rate = pd.to_numeric(out["CouponRate"], errors="coerce").fillna(0.0)
    frn_mask = security_type.eq("FRN")
    fixed_coupon_mask = security_type.isin(["Fixed", "TIPS"]) & (coupon_rate > TGA_FLOOR_TOLERANCE)
    out.loc[frn_mask, "InterestPaymentFrequency"] = out.loc[frn_mask, "InterestPaymentFrequency"].fillna(4.0)
    out.loc[fixed_coupon_mask, "InterestPaymentFrequency"] = out.loc[
        fixed_coupon_mask, "InterestPaymentFrequency"
    ].fillna(2.0)
    needs_first = out["FirstInterestPaymentDate"].isna() & out["InterestPaymentFrequency"].notna()
    out.loc[needs_first, "FirstInterestPaymentDate"] = [
        _first_interest_after_issue(issue_date, maturity_date, frequency)
        for issue_date, maturity_date, frequency in zip(
            issue.loc[needs_first],
            maturity.loc[needs_first],
            out.loc[needs_first, "InterestPaymentFrequency"],
        )
    ]
    return out


def _first_interest_after_issue(issue_date: pd.Timestamp, maturity_date: pd.Timestamp, frequency: float) -> pd.Timestamp:
    if pd.isna(issue_date) or pd.isna(maturity_date) or pd.isna(frequency):
        return pd.NaT
    freq = int(round(float(frequency)))
    if freq <= 0 or 12 % freq != 0:
        return pd.NaT
    months_between = int(round(12 / freq))
    candidate = pd.Timestamp(maturity_date).normalize()
    issue = pd.Timestamp(issue_date).normalize()
    while candidate > issue:
        prior = candidate - pd.DateOffset(months=months_between)
        if prior <= issue:
            return candidate
        candidate = prior
    return pd.NaT


def _opening_tips_reference_cpi(portfolio: pd.DataFrame) -> float:
    tips = portfolio["SecurityType"].astype(str).eq("TIPS")
    if not tips.any():
        return 100.0
    reference_issue = pd.to_numeric(portfolio.loc[tips, "ReferenceCPI_Issue"], errors="coerce")
    index_ratio = pd.to_numeric(portfolio.loc[tips, "IndexRatio"], errors="coerce")
    implied = (reference_issue * index_ratio).dropna()
    if implied.empty:
        return 100.0
    return float(implied.median())


def _tips_pricing_horizon_end_date(yield_rows: Sequence[Mapping[str, Any]]) -> str:
    if not yield_rows:
        return SIMULATION_END_DATE
    horizon_dates: list[pd.Timestamp] = []
    for row in yield_rows:
        curve_date = pd.Timestamp(row["curve_date"]).normalize()
        tenor = float(row["tenor_years"])
        horizon_dates.append(curve_date + pd.DateOffset(months=max(1, int(round(tenor * 12.0)))))
    return max(horizon_dates).date().isoformat()


def _tips_terminal_cpi_manifest(
    tips_cpi_rows: Sequence[Mapping[str, Any]],
    tips_real_yield_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cpi_frame = pd.DataFrame(tips_cpi_rows)
    if cpi_frame.empty or "terminal_cpi_rule" not in cpi_frame.columns:
        return {}
    horizon_roles = cpi_frame.get("cpi_horizon_role", pd.Series("", index=cpi_frame.index)).astype(str)
    terminal_rows = cpi_frame.loc[horizon_roles.eq("tdcsim_terminal")]
    first = cpi_frame.iloc[0]
    terminal_growth = float(first["terminal_annualized_cpi_growth_decimal"])
    horizon_dates = [
        str(row.get("expected_inflation_horizon_date", ""))
        for row in tips_real_yield_rows
        if row.get("expected_inflation_horizon_date")
    ]
    return {
        "terminal_cpi_rule": str(first["terminal_cpi_rule"]),
        "terminal_cbo_anchor_month": str(first["terminal_cbo_anchor_month"]),
        "terminal_extrapolation_start_month": str(first["terminal_extrapolation_start_month"]),
        "terminal_annualized_cpi_growth_decimal": terminal_growth,
        "terminal_growth_source_period_start": str(first["terminal_growth_source_period_start"]),
        "terminal_growth_source_period_end": str(first["terminal_growth_source_period_end"]),
        "terminal_extrapolated_month_rows": int(len(terminal_rows)),
        "max_expected_inflation_horizon_date": max(horizon_dates) if horizon_dates else "",
        "source_status": (
            "cbo_cpi_u_interpolated_through_terminal_cbo_anchor_then_tdcsim_terminal_cpi_extrapolation"
        ),
    }


def build_opening_portfolio() -> pd.DataFrame:
    initial = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    initial.update(
        {
            "BondID": 1,
            "SecurityType": "Fixed",
            "IssueDate": pd.Timestamp("2025-09-30"),
            "MaturityDate": pd.Timestamp("2046-09-30"),
            "OriginalMaturityYears": 21.0,
            "FaceValue": OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL,
            "CouponRate": 0.04,
            "HolderType": "Private",
            "HolderSubBucket": PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
            "Status": "Active",
            "MaturityCategory": "bonds",
            "OriginalPrincipal": OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL,
            "AdjustedPrincipal": OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL,
            "ReferenceCPI_Issue": 0.0,
            "IndexRatio": 1.0,
            "FixedSpread": 0.0,
            "AccruedInterest_FRN": 0.0,
            "BenchmarkRate_FRN": 0.0,
            "LastAccrualDate": pd.NaT,
            "IssuePriceRatio": 1.0,
            "IssueProceeds": OPENING_MARKETABLE_PUBLIC_PORTFOLIO_BIL,
            "IssueYieldAtIssue": 0.04,
        }
    )
    return pd.DataFrame([initial], columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def build_engine_params(
    inputs_dir: Path,
    baseline_input_paths: Mapping[str, str],
    initial_portfolio: pd.DataFrame,
    base_tga_bil: float,
) -> dict[str, Any]:
    return {
        "initial_values": {"reserves": 3000.0, "tdc_level": 0.0, "tga": base_tga_bil},
        "tga_params": {"target_balance": base_tga_bil, "floor": -1e15},
        "fiscal_params": {
            "initial_weekly_spending": 0.0,
            "initial_weekly_taxes": 0.0,
            "spending_growth_qtr": 0.0,
            "tax_growth_qtr": 0.0,
        },
        "other_flows": {
            "reserve_transfer": 0.0,
            "cb_net_expense": 0.0,
            "money_minting_transfers": 0.0,
        },
        "treasury_issuance_profile": {
            "bills": {
                "category_cutoff_years": 1.0,
                "target_percentage_of_remainder": 0.25,
                "maturities": [0.5],
                "maturity_distribution": [1.0],
            },
            "notes": {
                "category_cutoff_years": 10.0,
                "target_percentage_of_remainder": 0.55,
                "maturities": [5.0],
                "maturity_distribution": [1.0],
            },
            "bonds": {
                "category_cutoff_years": 999.0,
                "target_percentage_of_remainder": 0.20,
                "maturities": [20.0],
                "maturity_distribution": [1.0],
            },
            "TIPS": {
                "target_percentage": 0.06,
                "maturities": [10.0],
                "maturity_distribution": [1.0],
            },
            "FRN": {
                "target_percentage": 0.04,
                "maturities": [2.0],
                "maturity_distribution": [1.0],
            },
            "NonMarketable": {
                "target_percentage": 0.0,
                "maturities": [30.0],
                "maturity_distribution": [1.0],
            },
            "remainder_maturity_years": 1.0,
        },
        "yield_curve": {
            "use_static": True,
            "years": [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            "rates": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        },
        "yield_curve_surface": {
            "file": str(inputs_dir / "tdcsim_yield_curve_surface.csv"),
            "scenario_id": SCENARIO_ID,
            "interpolation_method": "pchip",
            "floor_zero": False,
        },
        "sector_preferences": _holder_preferences(),
        "private_mmf_split": {
            "bills": 0.25,
            "notes": 0.10,
            "bonds": 0.05,
            "tips": 0.05,
            "frn": 0.20,
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
        "events": [],
        "funding_rule": {
            "mode": "cbo_public_debt_target",
            "target_enforcement": "every_period",
            "negative_required_issuance_action": "retire_shortest_public_marketable",
            "target_tolerance_bil": 0.000001,
        },
        "baseline_input_paths": dict(baseline_input_paths),
        "data_vintage": {"actuals_available_as_of": ACTUALS_AVAILABLE_AS_OF, "allow_lookahead": False},
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


def _holder_preferences() -> dict[str, dict[str, float]]:
    return default_cbo_holder_preferences(include_private_routes=False)


def _baseline_input_paths(inputs_dir: Path) -> dict[str, str]:
    return {
        "source_contract_file": str(inputs_dir / "source_contract_smoke.json"),
        "source_fixture_file": str(inputs_dir / "source_fixtures.csv"),
        "cbo_fiscal_baseline_file": str(inputs_dir / "tdcsim_cbo_fiscal_baseline.csv"),
        "current_fy_splice_file": str(inputs_dir / "tdcsim_current_fy_splice.csv"),
        "debt_stock_path_file": str(inputs_dir / "tdcsim_debt_stock_path.csv"),
        "primary_deficit_path_file": str(inputs_dir / "tdcsim_primary_deficit_path.csv"),
        "operating_cash_path_file": str(inputs_dir / "tdcsim_operating_cash_path.csv"),
        "cash_reconciliation_residual_file": str(inputs_dir / "tdcsim_cash_reconciliation_residual.csv"),
        "macro_forecast_path_file": str(inputs_dir / "tdcsim_macro_forecast_path.csv"),
        "yield_curve_surface_file": str(inputs_dir / "tdcsim_yield_curve_surface.csv"),
        "frn_rate_path_file": str(inputs_dir / "tdcsim_frn_rate_path.csv"),
        "tips_cpi_path_file": str(inputs_dir / "tdcsim_tips_cpi_path.csv"),
        "tips_real_yield_path_file": str(inputs_dir / "tdcsim_tips_real_yield_path.csv"),
        "fiscal_incidence_policy_file": str(inputs_dir / "tdcsim_fiscal_incidence_policy.csv"),
        "net_interest_bridge_file": str(inputs_dir / "tdcsim_net_interest_bridge.csv"),
        "fed_holdings_path_file": str(inputs_dir / "tdcsim_fed_holdings_path.csv"),
    }


def _cpi_by_period_end(
    macro_rows: Sequence[Mapping[str, Any]],
    periods: Sequence[SimulationPeriod],
) -> dict[str, float]:
    return {
        period.period_end.isoformat(): _macro_level_for_date(macro_rows, period.period_end)
        for period in periods
    }


def _macro_level_for_date(macro_rows: Sequence[Mapping[str, Any]], target: date) -> float:
    rows = sorted(macro_rows, key=lambda row: str(row["period_start"]))
    prior: Mapping[str, Any] | None = None
    for row in rows:
        start = date.fromisoformat(str(row["period_start"]))
        end = date.fromisoformat(str(row["period_end"]))
        if start <= target <= end:
            return float(row["cbo_cpi_u_index"])
        if start <= target:
            prior = row
    if prior is not None:
        return float(prior["cbo_cpi_u_index"])
    raise ValueError(f"no CBO CPI-U macro row covers {target.isoformat()}")


def _frozen_base_curve_rows() -> list[dict[str, Any]]:
    rows_without_hash = [
        {"tenor_years": 1.0 / 12.0, "nominal_rate": 3.67},
        {"tenor_years": 0.25, "nominal_rate": 3.79},
        {"tenor_years": 0.5, "nominal_rate": 3.81},
        {"tenor_years": 1.0, "nominal_rate": 3.84},
        {"tenor_years": 2.0, "nominal_rate": 4.05},
        {"tenor_years": 3.0, "nominal_rate": 4.08},
        {"tenor_years": 5.0, "nominal_rate": 4.16},
        {"tenor_years": 7.0, "nominal_rate": 4.28},
        {"tenor_years": 10.0, "nominal_rate": 4.43},
        {"tenor_years": 20.0, "nominal_rate": 4.92},
        {"tenor_years": 30.0, "nominal_rate": 4.93},
    ]
    source_key = "official_treasury_daily_nominal_par_curve_2026_06_16"
    source_hash = hashlib.sha256(
        json.dumps(rows_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return [
        {
            "base_curve_date": BASE_CURVE_OBSERVATION_DATE,
            "available_date": BASE_CURVE_OBSERVATION_DATE,
            "base_curve_source_key": source_key,
            "base_curve_sha256": source_hash,
            **row,
        }
        for row in rows_without_hash
    ]


def write_source_package(
    output_dir: Path,
    *,
    budget_workbook: Path,
    economic_workbook: Path,
    opening_portfolio: Path,
    ratewall_input_manifest: Path,
    tips_cpi_detail: Path,
    frn_daily_indexes: Path,
    mts_table_1: Path,
    dts_operating_cash_balance: Path,
    mspd_table_1: Path,
    mspd_table_3_market: Path,
    mts_table_9_net_interest: Path | None,
    debt_to_penny_public_debt: Path | None,
    treasury_base_curve: Path | None,
    base_curve_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sources_dir = output_dir / "sources"
    source_entries: dict[str, dict[str, Any]] = {}
    file_map = {
        "cbo/51118-2026-02-Budget-Projections.xlsx": budget_workbook,
        "cbo/51135-2026-02-Economic-Projections.xlsx": economic_workbook,
        "opening/tdcsim_initial_portfolio_cohorts.csv": opening_portfolio,
        "opening/tdcsim_ratewall_input_manifest.json": ratewall_input_manifest,
        "fiscaldata/tips_cpi_data_detail.csv": tips_cpi_detail,
        "fiscaldata/frn_daily_indexes.csv": frn_daily_indexes,
    }
    for rel, source in file_map.items():
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"required CBO source-package input is missing: {rel} <- {source_path}")
        target = sources_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        packaged_entry = _packaged_source_manifest_entry(source_path, rel)
        if packaged_entry is not None:
            if source_path.resolve() != target.resolve():
                shutil.copy2(source_path, target)
            source_entries[rel] = _source_entry_from_packaged_manifest(packaged_entry, target, rel)
            continue
        if source_path == ratewall_input_manifest:
            original_sha256 = _sha256_file(source_path)
            source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            _write_json(target, _portableize_metadata_paths(source_payload))
            extra = {"original_sha256": original_sha256, "portable_reserialized": True}
        else:
            if source_path.resolve() != target.resolve():
                shutil.copy2(source_path, target)
            extra = {}
        source_entries[rel] = {
            "sha256": _sha256_file(target),
            "bytes": target.stat().st_size,
            "source_path": rel,
            "origin_repo_path": _source_origin_label(source_path, rel),
            **extra,
        }
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["mts_table_1"],
        source=None,
        rows=[parse_mts_table_1_deficit_row(mts_table_1)],
        source_entries=source_entries,
    )
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["mts_table_9_net_interest"],
        source=mts_table_9_net_interest,
        rows=[
            {
                **FROZEN_MTS_TABLE_9_NET_INTEREST_ROW,
                "source_family": "monthly_treasury_statement_table_9",
                "source_selector": RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR,
                "source_status": "canonical_exact_source_row_extract",
            }
        ],
        source_entries=source_entries,
    )
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["dts_operating_cash_balance"],
        source=None,
        rows=[parse_dts_tga_closing_balance_row(dts_operating_cash_balance)],
        source_entries=source_entries,
    )
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["mspd_table_1"],
        source=None,
        rows=parse_mspd_debt_bridge_rows(mspd_table_1),
        source_entries=source_entries,
    )
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["mspd_table_3_market"],
        source=None,
        rows=_csv_rows_matching(mspd_table_3_market, record_date=OPENING_PORTFOLIO_RECORD_DATE),
        source_entries=source_entries,
    )
    _copy_or_write_canonical_csv(
        sources_dir=sources_dir,
        rel=SOURCE_PACKAGE_INPUTS["debt_to_penny_public_debt"],
        source=debt_to_penny_public_debt,
        rows=[
            {
                **FROZEN_DEBT_TO_PENNY_ROW,
                "source_family": "debt_to_the_penny",
                "source_selector": RUNNER_DEBT_TO_PENNY_SELECTOR,
                "source_status": "canonical_exact_source_row_extract",
            }
        ],
        source_entries=source_entries,
    )
    base_curve_target = sources_dir / "treasury" / "base_curve_2026-06-16.json"
    if treasury_base_curve is not None:
        source_path = Path(treasury_base_curve)
        if not source_path.exists():
            raise FileNotFoundError(
                f"required CBO source-package input is missing: treasury/base_curve_2026-06-16.json <- {source_path}"
            )
        base_curve_target.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != base_curve_target.resolve():
            shutil.copy2(source_path, base_curve_target)
        rel = "treasury/base_curve_2026-06-16.json"
        packaged_entry = _packaged_source_manifest_entry(source_path, rel)
        if packaged_entry is not None:
            source_entries[rel] = _source_entry_from_packaged_manifest(packaged_entry, base_curve_target, rel)
            extra = None
        else:
            extra = {"origin_repo_path": _source_origin_label(source_path, rel)}
    else:
        rel = "treasury/base_curve_2026-06-16.json"
        _write_json(
            base_curve_target,
            {
                "schema_version": "tdcsim_treasury_base_curve_canonical_extract_v1",
                "source_family": "treasury_daily_nominal_par_yield_curve",
                "source_key": "official_treasury_daily_nominal_par_curve_2026_06_16",
                "observation_date": BASE_CURVE_OBSERVATION_DATE,
                "available_date": BASE_CURVE_OBSERVATION_DATE,
                "source_status": "canonical_source_extract_of_official_daily_treasury_par_curve",
                "rows": [
                    {
                        "base_curve_date": row["base_curve_date"],
                        "available_date": row["available_date"],
                        "tenor_years": row["tenor_years"],
                        "nominal_rate": row["nominal_rate"],
                    }
                    for row in base_curve_rows
                ],
            },
        )
        extra = {"origin_source_label": "runner_canonical_treasury_base_curve_extract"}
    if extra is not None:
        source_entries[rel] = {
            "sha256": _sha256_file(base_curve_target),
            "bytes": base_curve_target.stat().st_size,
            "source_path": rel,
            **extra,
        }
    manifest = {
        "schema_version": "tdcsim_cbo_source_package_manifest_v1",
        "files": source_entries,
        "status": "package_relative_sources_for_cbo_forecast_smoke",
        "inventory_status": "complete_required_inventory_no_silent_skips",
    }
    manifest_path = sources_dir / "source_manifest.json"
    _write_json(manifest_path, manifest)
    source_entries["source_manifest.json"] = {
        "sha256": _sha256_file(manifest_path),
        "bytes": manifest_path.stat().st_size,
        "source_path": "source_manifest.json",
        "origin_source_label": "package_generated_source_manifest",
    }
    return {
        "sources_dir": "sources",
        "source_manifest": "sources/source_manifest.json",
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_file_count": len(source_entries),
        "source_files": source_entries,
    }


def _copy_or_write_canonical_csv(
    *,
    sources_dir: Path,
    rel: str,
    source: Path | None,
    rows: Sequence[Mapping[str, Any]],
    source_entries: dict[str, dict[str, Any]],
) -> None:
    target = sources_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"required CBO source-package input is missing: {rel} <- {source_path}")
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        packaged_entry = _packaged_source_manifest_entry(source_path, rel)
        if packaged_entry is not None:
            source_entries[rel] = _source_entry_from_packaged_manifest(packaged_entry, target, rel)
            return
        extra = {"origin_repo_path": _source_origin_label(source_path, rel)}
    else:
        write_rows_csv_union(target, rows)
        extra = {"origin_source_label": "runner_canonical_exact_source_row_extract"}
    source_entries[rel] = {
        "sha256": _sha256_file(target),
        "bytes": target.stat().st_size,
        "source_path": rel,
        **extra,
    }


def _packaged_source_manifest_entry(source_path: Path, rel: str) -> dict[str, Any] | None:
    resolved = source_path.expanduser().resolve()
    for candidate in resolved.parents:
        manifest_path = candidate / "source_manifest.json"
        if not manifest_path.exists():
            continue
        try:
            if (candidate / rel).resolve() != resolved:
                continue
        except FileNotFoundError:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files", {})
        entry = files.get(rel) if isinstance(files, Mapping) else None
        if isinstance(entry, Mapping):
            return dict(entry)
    return None


def _source_entry_from_packaged_manifest(entry: Mapping[str, Any], target: Path, rel: str) -> dict[str, Any]:
    copied = dict(entry)
    copied["sha256"] = _sha256_file(target)
    copied["bytes"] = target.stat().st_size
    copied["source_path"] = rel
    return copied


def parse_mts_table_1_deficit_row(path: str | Path) -> dict[str, Any]:
    row = _single_csv_row(
        path,
        selector=RUNNER_MTS_TABLE_1_DEFICIT_SELECTOR,
        predicate=lambda item: (
            str(item.get("record_date")) == FISCAL_ACTUALS_THROUGH
            and str(item.get("classification_desc")) == "Year-to-Date"
            and str(item.get("line_code_nbr")) == "280"
        ),
    )
    _require_money(row, "current_month_dfct_sur_amt", expected=1_246_203_266_386.93)
    return row


def parse_mts_table_9_net_interest_row(path: str | Path) -> dict[str, Any]:
    row = _single_csv_row(
        path,
        selector=RUNNER_MTS_TABLE_9_NET_INTEREST_SELECTOR,
        predicate=lambda item: (
            str(item.get("record_date")) == FISCAL_ACTUALS_THROUGH
            and str(item.get("classification_desc")) == "Net Interest"
        ),
    )
    _require_money(row, "current_fytd_rcpt_outly_amt", expected=722_706_511_243.20)
    return row


def parse_dts_tga_closing_balance_row(path: str | Path) -> dict[str, Any]:
    row = _single_csv_row(
        path,
        selector="record_date=2026-06-16;account_type=Treasury General Account (TGA) Closing Balance",
        predicate=lambda item: (
            str(item.get("record_date")) == TGA_OBSERVATION_DATE
            and str(item.get("account_type")) == "Treasury General Account (TGA) Closing Balance"
        ),
    )
    _require_money(row, "open_today_bal", expected=981_113.0)
    return row


def parse_mspd_debt_bridge_rows(path: str | Path) -> list[dict[str, Any]]:
    selectors = {
        "Total Nonmarketable": 620_821.77409502,
        "Total Public Debt Outstanding": 31_515_369.9184452,
    }
    rows = []
    for selector, expected in selectors.items():
        row = _single_csv_row(
            path,
            selector=f"record_date=2026-05-31;security_type_desc={selector}",
            predicate=lambda item, selector=selector: (
                str(item.get("record_date")) == FISCAL_ACTUALS_THROUGH
                and str(item.get("security_type_desc")) == selector
            ),
        )
        _require_money(row, "debt_held_public_mil_amt", expected=expected)
        rows.append(row)
    return rows


def parse_debt_to_penny_public_debt_row(path: str | Path) -> dict[str, Any]:
    row = _single_csv_row(
        path,
        selector=RUNNER_DEBT_TO_PENNY_SELECTOR,
        predicate=lambda item: str(item.get("record_date")) == "2026-05-29",
    )
    _require_money(row, "debt_held_public_amt", expected=31_515_369_798_622.98)
    return row


def parse_base_curve_rows(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Treasury base curve extract must contain rows: {source_path}")
    source_key = str(payload.get("source_key", "official_treasury_daily_nominal_par_curve_2026_06_16"))
    source_hash = _sha256_file(source_path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Treasury base curve rows must be objects")
        curve_date = str(row.get("base_curve_date", row.get("curve_date", payload.get("observation_date", ""))))
        if curve_date != BASE_CURVE_OBSERVATION_DATE:
            raise ValueError(f"Treasury base curve row has wrong date: {curve_date}")
        tenor = float(row["tenor_years"])
        nominal_rate = float(row["nominal_rate"])
        parsed.append(
            {
                "base_curve_date": curve_date,
                "available_date": str(row.get("available_date", payload.get("available_date", curve_date))),
                "base_curve_source_key": str(row.get("base_curve_source_key", source_key)),
                "base_curve_sha256": source_hash,
                "tenor_years": tenor,
                "nominal_rate": nominal_rate,
            }
        )
    tenors = {round(row["tenor_years"], 12) for row in parsed}
    required = {round(value, 12) for value in (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)}
    if tenors != required:
        raise ValueError(f"Treasury base curve tenors changed: {sorted(tenors)}")
    return sorted(parsed, key=lambda row: row["tenor_years"])


def _single_csv_row(
    path: str | Path,
    *,
    selector: str,
    predicate: Any,
) -> dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"source row file is missing for {selector}: {csv_path}")
    matches: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if predicate(row):
                matches.append(dict(row))
    if len(matches) != 1:
        raise ValueError(f"{selector} matched {len(matches)} rows in {csv_path}")
    return matches[0]


def _csv_rows_matching(path: str | Path, **equals: str) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"source extract input is missing: {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if all(str(row.get(key)) == value for key, value in equals.items()):
                rows.append(dict(row))
    if not rows:
        selector = ";".join(f"{key}={value}" for key, value in equals.items())
        raise ValueError(f"{selector} matched 0 rows in {csv_path}")
    return rows


def _csv_rows_matching(path: str | Path, **filters: str) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"source row file is missing: {csv_path}")
    matches: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if all(str(row.get(key)) == str(value) for key, value in filters.items()):
                matches.append(dict(row))
    if not matches:
        raise ValueError(f"{csv_path} matched 0 rows for {filters}")
    return matches


def _require_money(row: Mapping[str, Any], key: str, *, expected: float, tolerance: float = 0.005) -> None:
    actual = _money_value(row, key)
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{key} expected {expected:.2f}, found {actual:.2f}")


def _money_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "" or str(value).lower() == "null":
        raise ValueError(f"required money field missing: {key}")
    return float(str(value).replace(",", ""))


def _selector_map(source_fixture_rows: Sequence[Mapping[str, Any]], source_family: str) -> dict[str, dict[str, Any]]:
    selectors: dict[str, dict[str, Any]] = {}
    for row in source_fixture_rows:
        if row.get("source_family") != source_family:
            continue
        selector = str(row["source_row_selector"])
        selectors.setdefault(
            selector,
            {
                "source_sheet": row["source_sheet"],
                "source_row_number": row["source_row_number"],
                "source_unit_block": row["source_unit_block"],
            },
        )
    return selectors


def _row_by_key(rows: Sequence[Mapping[str, Any]], key: str, value: object) -> Mapping[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    raise ValueError(f"missing row where {key}={value!r}")


def _resolve_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    output_root = (PROJECT_ROOT / "output").resolve()
    if output_root not in (path, *path.parents):
        raise ValueError(f"output_dir must be under ignored output/: {path}")
    return path


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        try:
            return str(resolved.relative_to(PROJECT_ROOT.parent))
        except ValueError:
            return str(resolved)


def _package_relative_path(package_root: str | Path, path: str | Path) -> str:
    root = Path(package_root).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"package path must resolve under package root: {path}") from exc


def _package_path_or_portable(package_root: str | Path, path: str | Path) -> str:
    try:
        return _package_relative_path(package_root, path)
    except ValueError:
        return _portable_path(path)


def _source_origin_label(path: str | Path, rel: str) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
        return _portable_path(resolved)
    except ValueError:
        pass
    try:
        resolved.relative_to(PROJECT_ROOT.parent)
        return _portable_path(resolved)
    except ValueError:
        return f"external_source_package/{rel}"


def _portableize_metadata_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _portableize_metadata_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portableize_metadata_paths(item) for item in value]
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return value
        path = Path(value)
        if path.is_absolute():
            return _portable_path(path)
    return value


def _package_relative_opening_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    portable = _portableize_metadata_paths(metadata)
    canonical_paths = {
        "source_file": f"sources/{SOURCE_PACKAGE_INPUTS['opening_portfolio']}",
        "source_manifest": f"sources/{SOURCE_PACKAGE_INPUTS['ratewall_input_manifest']}",
        "tips_cpi_detail_source": f"sources/{SOURCE_PACKAGE_INPUTS['tips_cpi_detail']}",
        "frn_daily_indexes_source": f"sources/{SOURCE_PACKAGE_INPUTS['frn_daily_indexes']}",
        "opening_rollforward_diagnostics_file": "forecast_inputs/tdcsim_opening_rollforward_diagnostics.csv",
        "tips_indexation_diagnostics_file": "forecast_inputs/tdcsim_opening_tips_indexation_diagnostics.csv",
        "frn_indexation_diagnostics_file": "forecast_inputs/tdcsim_opening_frn_indexation_diagnostics.csv",
    }
    for key, value in canonical_paths.items():
        if key in portable:
            portable[key] = value
    return portable


def resolve_source_package_inputs(source_package: str | Path) -> dict[str, Path]:
    source_root = Path(source_package).expanduser().resolve()
    if (source_root / "source_manifest.json").exists():
        sources_dir = source_root
    elif (source_root / "sources" / "source_manifest.json").exists():
        sources_dir = source_root / "sources"
    else:
        raise FileNotFoundError(
            f"source package must point to a package root or sources/ directory with source_manifest.json: {source_package}"
        )
    manifest = json.loads((sources_dir / "source_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if not isinstance(files, Mapping):
        raise ValueError("source package manifest must contain a files object")
    resolved: dict[str, Path] = {}
    for name, relative_path in SOURCE_PACKAGE_INPUTS.items():
        entry = files.get(relative_path)
        if not isinstance(entry, Mapping):
            raise ValueError(f"source package is missing required source file: {relative_path}")
        path = sources_dir / relative_path
        if not path.exists():
            raise FileNotFoundError(f"source package file is missing on disk: {relative_path}")
        expected_hash = str(entry.get("sha256", ""))
        actual_hash = _sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"source package sha256 mismatch for {relative_path}")
        resolved[name] = path
    return resolved


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows_csv_union(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifacts(output_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(output_dir)
    hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if rel in {"manifest.json", "forecast_inputs/source_contract_smoke.json"}:
            continue
        entry: dict[str, Any] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                entry["rows_including_header"] = sum(1 for _ in handle)
        hashes[rel] = entry
    return hashes


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory under output/.")
    parser.add_argument("--budget-workbook", default=str(DEFAULT_BUDGET_WORKBOOK))
    parser.add_argument("--economic-workbook", default=str(DEFAULT_ECONOMIC_WORKBOOK))
    parser.add_argument("--opening-portfolio", default=str(DEFAULT_OPENING_PORTFOLIO))
    parser.add_argument("--ratewall-input-manifest", default=str(DEFAULT_RATEWALL_INPUT_MANIFEST))
    parser.add_argument("--tips-cpi-detail", default=str(DEFAULT_TIPS_CPI_DETAIL))
    parser.add_argument("--frn-daily-indexes", default=str(DEFAULT_FRN_DAILY_INDEXES))
    parser.add_argument("--mts-table-1", default=str(DEFAULT_MTS_TABLE_1))
    parser.add_argument("--dts-operating-cash-balance", default=str(DEFAULT_DTS_OPERATING_CASH_BALANCE))
    parser.add_argument("--mspd-table-1", default=str(DEFAULT_MSPD_TABLE_1))
    parser.add_argument("--mspd-table-3-market", default=str(DEFAULT_MSPD_TABLE_3_MARKET))
    parser.add_argument("--mts-table-9-net-interest")
    parser.add_argument("--debt-to-penny-public-debt")
    parser.add_argument("--treasury-base-curve")
    parser.add_argument(
        "--source-package",
        help="Resolve required source files from an extracted package root or sources/ directory.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Do not remove the output directory before rebuilding.")
    parser.add_argument("--build-inputs-only", action="store_true", help="Write inputs/manifest without running the engine.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = resolve_source_package_inputs(args.source_package) if args.source_package else {}
    bundle = build_forecast_bundle(
        args.output_dir,
        budget_workbook=source_paths.get("budget_workbook", Path(args.budget_workbook)),
        economic_workbook=source_paths.get("economic_workbook", Path(args.economic_workbook)),
        opening_portfolio=source_paths.get("opening_portfolio", Path(args.opening_portfolio)),
        ratewall_input_manifest=source_paths.get("ratewall_input_manifest", Path(args.ratewall_input_manifest)),
        tips_cpi_detail=source_paths.get("tips_cpi_detail", Path(args.tips_cpi_detail)),
        frn_daily_indexes=source_paths.get("frn_daily_indexes", Path(args.frn_daily_indexes)),
        mts_table_1=source_paths.get("mts_table_1", Path(args.mts_table_1)),
        dts_operating_cash_balance=source_paths.get(
            "dts_operating_cash_balance",
            Path(args.dts_operating_cash_balance),
        ),
        mspd_table_1=source_paths.get("mspd_table_1", Path(args.mspd_table_1)),
        mspd_table_3_market=source_paths.get("mspd_table_3_market", Path(args.mspd_table_3_market)),
        mts_table_9_net_interest=source_paths.get(
            "mts_table_9_net_interest",
            Path(args.mts_table_9_net_interest) if args.mts_table_9_net_interest else None,
        ),
        debt_to_penny_public_debt=source_paths.get(
            "debt_to_penny_public_debt",
            Path(args.debt_to_penny_public_debt) if args.debt_to_penny_public_debt else None,
        ),
        treasury_base_curve=source_paths.get(
            "treasury_base_curve",
            Path(args.treasury_base_curve) if args.treasury_base_curve else None,
        ),
        clean=not args.no_clean,
    )
    manifest = bundle.manifest
    if not args.build_inputs_only:
        manifest = run_smoke(bundle)
    print(json.dumps({"manifest": str(bundle.output_dir / "manifest.json"), "scenario_id": SCENARIO_ID}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
