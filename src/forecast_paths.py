"""Generic forecast-input path resolution and CSV loaders."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ALLOWED_SOURCE_ROLES = frozenset(
    {
        "hard_actual_state",
        "official_hard_anchor",
        "hard_input",
        "identity_check",
        "calibration_check",
        "diagnostic",
        "scenario_assumption",
    }
)

OPERATING_CASH_OPENING_STATE_ROLE = "opening_state_anchor"
OPERATING_CASH_COMPARISON_ROLE = "nonbinding_comparison_target"
FED_REMITTANCE_PATH_FILE = "tdcsim_fed_remittance_path.csv"
FED_REMITTANCE_PATH_SCHEMA_VERSION = "tdcsim_fed_remittance_path_v1"
FED_INCOME_PATH_FILE = "tdcsim_fed_income_path.csv"
FED_INCOME_PATH_SCHEMA_VERSION = "tdcsim_fed_income_path_v3"
POLICY_RATE_PATH_FILE = "tdcsim_policy_rate_path.csv"
POLICY_RATE_PATH_SCHEMA_VERSION = "tdcsim_policy_rate_path_v1"
# The Table 4-3 path stays in the ordinary fiscal flow (CBO counts remittances as receipts);
# the TGA receives the run's own remittance, built from the Fed's income statement.
CBO_FED_REMITTANCE_ROLE = "endogenous_fed_net_income_deferred_asset_fed_to_tga_cash"
# The role of the fixed Table 4-3 input itself: added back to the ordinary fiscal flow only.
CBO_TABLE_4_3_FISCAL_FLOW_ROLE = "cbo_table_4_3_remittance_added_back_to_ordinary_fiscal_flow_not_paid_to_tga"
CBO_UNSUPPORTED_FED_COMPONENTS = (
    "fed_effective_interest_amortization",
    "fed_mbs_payment_clock",
    "fed_valuation_gains_and_losses",
    "mbs_prepayment_rate_response",
    "reserve_bank_level_remittances_under_system_deferred_asset",
    "fed_policy_rate_response_to_scenarios",
)
CBO_FED_REMITTANCE_EXCLUDED_CLAIMS = tuple(
    f"does_not_model_{component}" for component in CBO_UNSUPPORTED_FED_COMPONENTS
)
CBO_FED_REMITTANCE_STATUS = "modeled_fed_accrual_earnings_pooled_weekly_remittance_fed_to_tga_no_reserve_or_deposit_effect"
CBO_FED_REMITTANCE_OPENING_STATUS = "not_applicable_opening_row"

ALLOWED_RUNTIME_ROLES = frozenset(
    {
        "hard_target",
        "hard_flow",
        "check_only",
        "reconciliation_only",
        "memo_only",
        OPERATING_CASH_OPENING_STATE_ROLE,
        OPERATING_CASH_COMPARISON_ROLE,
    }
)


def operating_cash_runtime_role(row_index: int) -> str:
    """Return the runtime role matching the engine's ordered operating-cash use."""

    if row_index < 0:
        raise ValueError("operating-cash row index must be nonnegative")
    if row_index == 0:
        return OPERATING_CASH_OPENING_STATE_ROLE
    return OPERATING_CASH_COMPARISON_ROLE


BASELINE_INPUT_PATH_KEYS = (
    "source_contract_file",
    "source_fixture_file",
    "cbo_fiscal_baseline_file",
    "current_fy_splice_file",
    "debt_stock_path_file",
    "primary_deficit_path_file",
    "fed_remittance_path_file",
    "fed_income_path_file",
    "policy_rate_path_file",
    "operating_cash_path_file",
    "cash_reconciliation_residual_file",
    "macro_forecast_path_file",
    "yield_curve_surface_file",
    "fiscal_incidence_policy_file",
    "net_interest_bridge_file",
    "holder_absorption_path_file",
    "fed_holdings_path_file",
    "frn_rate_path_file",
    "tips_cpi_path_file",
    "tips_real_yield_path_file",
)

COMPILED_FORECAST_INPUT_FILES = {
    "source_contract_file": "source_contract_smoke.json",
    "source_fixture_file": "source_fixtures.csv",
    "cbo_fiscal_baseline_file": "tdcsim_cbo_fiscal_baseline.csv",
    "current_fy_splice_file": "tdcsim_current_fy_splice.csv",
    "debt_stock_path_file": "tdcsim_debt_stock_path.csv",
    "primary_deficit_path_file": "tdcsim_primary_deficit_path.csv",
    "fed_remittance_path_file": FED_REMITTANCE_PATH_FILE,
    "fed_income_path_file": FED_INCOME_PATH_FILE,
    "policy_rate_path_file": POLICY_RATE_PATH_FILE,
    "operating_cash_path_file": "tdcsim_operating_cash_path.csv",
    "cash_reconciliation_residual_file": "tdcsim_cash_reconciliation_residual.csv",
    "macro_forecast_path_file": "tdcsim_macro_forecast_path.csv",
    "yield_curve_surface_file": "tdcsim_yield_curve_surface.csv",
    "fiscal_incidence_policy_file": "tdcsim_fiscal_incidence_policy.csv",
    "net_interest_bridge_file": "tdcsim_net_interest_bridge.csv",
    "holder_absorption_path_file": "tdcsim_holder_profile_assumptions.csv",
    "fed_holdings_path_file": "tdcsim_fed_holdings_path.csv",
    "frn_rate_path_file": "tdcsim_frn_rate_path.csv",
    "tips_cpi_path_file": "tdcsim_tips_cpi_path.csv",
    "tips_real_yield_path_file": "tdcsim_tips_real_yield_path.csv",
}


def compiled_forecast_input_paths(inputs_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Return logical engine path keys for a compiled CBO forecast-input directory."""

    root = Path(inputs_dir)
    paths = {key: str(root / filename) for key, filename in COMPILED_FORECAST_INPUT_FILES.items()}
    missing = [key for key, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"compiled CBO forecast inputs missing required files: {missing}")
    return paths


@dataclass(frozen=True)
class BaselineInputPaths:
    """Resolved generic CBO baseline input paths."""

    source_contract_file: Path | None = None
    source_fixture_file: Path | None = None
    cbo_fiscal_baseline_file: Path | None = None
    current_fy_splice_file: Path | None = None
    debt_stock_path_file: Path | None = None
    primary_deficit_path_file: Path | None = None
    fed_remittance_path_file: Path | None = None
    fed_income_path_file: Path | None = None
    policy_rate_path_file: Path | None = None
    operating_cash_path_file: Path | None = None
    cash_reconciliation_residual_file: Path | None = None
    macro_forecast_path_file: Path | None = None
    yield_curve_surface_file: Path | None = None
    fiscal_incidence_policy_file: Path | None = None
    net_interest_bridge_file: Path | None = None
    holder_absorption_path_file: Path | None = None
    fed_holdings_path_file: Path | None = None
    frn_rate_path_file: Path | None = None
    tips_cpi_path_file: Path | None = None
    tips_real_yield_path_file: Path | None = None

    def as_dict(self) -> dict[str, Path | None]:
        return {key: getattr(self, key) for key in BASELINE_INPUT_PATH_KEYS}


@dataclass(frozen=True)
class ForecastCsvSchema:
    required_columns: frozenset[str]
    unique_key: tuple[str, ...]


DATE_COLUMNS = frozenset(
    {
        "available_date",
        "observation_date",
        "forecast_publication_date",
        "source_as_of",
        "simulation_start_date",
        "fiscal_actuals_through",
        "actuals_available_as_of",
        "period_start",
        "period_end",
        "curve_date",
        "base_curve_date",
        "base_date",
        "rate_effective_start",
        "rate_effective_end",
        "month",
        "anchor_date",
    }
)

BOOLEAN_COLUMNS = frozenset(
    {
        "affects_operating_cash",
        "affects_primary_deficit",
        "affects_net_interest",
        "affects_total_deficit",
        "affects_debt_target",
        "affects_issuance_size",
        "affects_tdc_fiscal_flow",
        "include_in_budget_interest",
    }
)


SOURCE_FIXTURE_COLUMNS = frozenset(
    {
        "schema_version",
        "forecast_name",
        "source_family",
        "source_file",
        "source_sha256",
        "source_url",
        "source_sheet",
        "source_row_number",
        "source_row_selector",
        "source_unit_block",
        "source_year_or_period",
        "raw_value",
        "raw_sign_convention",
        "canonical_transform",
        "canonical_value",
        "observation_date",
        "available_date",
        "source_status",
    }
)


FISCAL_BASELINE_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "fiscal_year",
        "primary_deficit_bil",
        "cbo_net_interest_bil",
        "cbo_total_deficit_bil",
        "debt_held_public_begin_bil",
        "debt_held_public_end_bil",
        "debt_identity_end_bil",
        "cbo_other_means_financing_bil",
        "cbo_financial_assets_end_bil",
        "cbo_fed_holdings_end_bil",
        "cbo_average_interest_rate_pct",
        "source_role",
        "runtime_role",
        "source_vintage",
        "forecast_publication_date",
        "source_as_of",
        "observation_date",
        "available_date",
        "source_family",
        "source_table",
        "source_row_selector",
        "source_status",
        "claim_boundary",
    }
)

CURRENT_FY_SPLICE_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "fiscal_year",
        "simulation_start_date",
        "fiscal_actuals_through",
        "actuals_available_as_of",
        "cbo_full_fy_primary_deficit_bil",
        "actual_total_deficit_fytd_bil",
        "actual_net_interest_fytd_bil",
        "actual_primary_deficit_fytd_bil",
        "remaining_cbo_primary_deficit_bil",
        "mts_table_1_selector",
        "mts_table_9_selector",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

DEBT_STOCK_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_end",
        "cbo_federal_debt_held_public_target_bil",
        "public_nonmarketable_treasury_bil",
        "non_treasury_and_definition_residual_bil",
        "marketable_treasury_public_target_bil",
        "bridge_source",
        "bridge_method",
        "anchor_type",
        "interpolation_method",
        "source_role",
        "runtime_role",
        "source_fiscal_year",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

PRIMARY_DEFICIT_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "primary_deficit_bil",
        "allocation_method",
        "day_count_weight",
        "source_fiscal_year",
        "annual_or_remaining_primary_deficit_bil",
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)

OPERATING_CASH_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_end",
        "operating_cash_target_bil",
        "tga_target_bil",
        "ttl_target_bil",
        "other_operating_cash_target_bil",
        "operating_cash_definition",
        "reserve_settlement_component",
        "construction_mode",
        "base_date",
        "base_balance_bil",
        "inflation_index",
        "inflation_index_level",
        "inflation_scalar",
        "component_coverage",
        "source_role",
        "runtime_role",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

CASH_RECONCILIATION_RESIDUAL_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "cash_reconciliation_residual_bil",
        "component_type",
        "affects_operating_cash",
        "affects_primary_deficit",
        "affects_net_interest",
        "affects_total_deficit",
        "affects_debt_target",
        "affects_issuance_size",
        "affects_tdc_fiscal_flow",
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)

MACRO_FORECAST_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "cbo_3m_tbill_rate_pct",
        "cbo_10y_treasury_rate_pct",
        "cbo_cpi_u_index",
        "cbo_cpi_u_inflation_pct",
        "source_role",
        "runtime_role",
        "source_vintage",
        "forecast_publication_date",
        "source_table",
        "source_row_selector",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

YIELD_CURVE_SURFACE_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "curve_date",
        "tenor_years",
        "nominal_rate",
        "nominal_rate_decimal",
        "rate_unit",
        "runtime_rate_unit",
        "curve_basis",
        "base_curve_date",
        "base_curve_source_key",
        "base_curve_sha256",
        "anchor_3m_pct",
        "anchor_10y_pct",
        "construction_method",
        "temporal_fill_method",
        "interpolation_method",
        "source_role",
        "runtime_role",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

FISCAL_INCIDENCE_POLICY_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "policy_id",
        "policy_mode",
        "incidence_basis",
        "signed_net_primary_flow_bil",
        "primary_outlays_bil",
        "primary_receipts_bil",
        "recipient_share_source",
        "du_share",
        "ru_share",
        "foreign_share",
        "other_share",
        "evidence_status",
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)

NET_INTEREST_BRIDGE_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "fiscal_year",
        "component_key",
        "amount_bil",
        "sign_convention",
        "include_in_budget_interest",
        "component_scope_status",
        "source_family",
        "source_table",
        "source_row_selector",
        "source_vintage",
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)

HOLDER_ABSORPTION_PATH_COLUMNS = frozenset(
    {
        "scenario_id",
        "quarter",
        "holder_type",
        "holder_subbucket",
        "source_status",
        "claim_boundary",
    }
)

FED_INCOME_PATH_CONSTANTS = (
    "h41_as_of_date",
    "h41_reserve_balances_bil",
    "h41_tga_bil",
    "h41_domestic_on_rrp_bil",
    "opening_domestic_on_rrp_bil",
    "h41_foreign_repo_bil",
    "h41_foreign_official_deposits_bil",
    "h41_soma_mbs_bil",
    "h41_currency_bil",
    "h41_other_deposits_bil",
    "h41_other_liabilities_and_capital_bil",
    "h41_earnings_remittances_due_bil",
    "h41_treasury_net_premium_bil",
    "remunerated_other_deposit_share",
    "previous_wednesday_date",
    "previous_wednesday_due_bil",
    "year_end_date",
    "year_end_due_bil",
    "opening_interest_payable_bil",
    "opening_foreign_repo_interest_payable_bil",
)
FED_INCOME_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "soma_mbs_start_bil",
        "soma_mbs_end_bil",
        "soma_mbs_interest_bil",
        "soma_mbs_premium_start_bil",
        "soma_mbs_premium_end_bil",
        "agency_debt_start_bil",
        "agency_debt_end_bil",
        "agency_debt_interest_bil",
        "agency_debt_premium_start_bil",
        "agency_debt_premium_end_bil",
        "currency_start_bil",
        "currency_end_bil",
        "other_deposits_start_bil",
        "other_deposits_end_bil",
        "other_liabilities_and_capital_start_bil",
        "other_liabilities_and_capital_end_bil",
        "other_income_bil",
        "operating_expenses_bil",
        "dividends_bil",
        *FED_INCOME_PATH_CONSTANTS,
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)
POLICY_RATE_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "fed_funds_rate_pct",
        "iorb_spread_pct",
        "on_rrp_spread_pct",
        "rate_source",
        "source_role",
        "runtime_role",
        "source_status",
        "claim_boundary",
    }
)

FED_HOLDINGS_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_end",
        "holder_type",
        "cbo_fed_holdings_target_bil",
        "interpolation_method",
        "source_fiscal_year",
        "source_role",
        "runtime_role",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

FRN_RATE_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "period_start",
        "period_end",
        "rate_effective_start",
        "rate_effective_end",
        "benchmark_tenor_years",
        "auction_high_rate_decimal",
        "benchmark_rate_decimal",
        "money_market_yield_decimal",
        "spread_treatment",
        "day_count_basis",
        "lockout_business_days",
        "rate_source_family",
        "source_role",
        "runtime_role",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

TIPS_CPI_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "month",
        "cbo_cpi_u_index",
        "tips_cpi_u_index",
        "reference_lag_months",
        "interpolation_method",
        "anchor_date",
        "anchor_reference_cpi",
        "scale_factor",
        "source_role",
        "runtime_role",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

TIPS_REAL_YIELD_PATH_COLUMNS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "curve_date",
        "tenor_years",
        "nominal_rate_decimal",
        "expected_inflation_decimal",
        "real_yield_decimal",
        "real_coupon_decimal",
        "pricing_method",
        "coupon_rounding",
        "source_role",
        "runtime_role",
        "observation_date",
        "available_date",
        "source_status",
        "claim_boundary",
    }
)

FORECAST_CSV_SCHEMAS = {
    "source_fixture_file": ForecastCsvSchema(
        SOURCE_FIXTURE_COLUMNS,
        ("forecast_name", "source_file", "source_sheet", "source_row_selector", "source_year_or_period"),
    ),
    "cbo_fiscal_baseline_file": ForecastCsvSchema(
        FISCAL_BASELINE_COLUMNS,
        ("scenario_id", "fiscal_year"),
    ),
    "current_fy_splice_file": ForecastCsvSchema(
        CURRENT_FY_SPLICE_COLUMNS,
        ("scenario_id", "fiscal_year", "simulation_start_date"),
    ),
    "debt_stock_path_file": ForecastCsvSchema(DEBT_STOCK_PATH_COLUMNS, ("scenario_id", "period_end")),
    "primary_deficit_path_file": ForecastCsvSchema(
        PRIMARY_DEFICIT_PATH_COLUMNS,
        ("scenario_id", "period_start", "period_end"),
    ),
    "fed_remittance_path_file": ForecastCsvSchema(
        frozenset({
            "schema_version", "scenario_id", "period_start", "period_end", "source_fiscal_year",
            "remittance_bil", "annual_or_remaining_remittance_bil", "source_role", "runtime_role",
            "source_status", "claim_boundary",
        }),
        ("scenario_id", "period_start", "period_end"),
    ),
    "fed_income_path_file": ForecastCsvSchema(FED_INCOME_PATH_COLUMNS, ("scenario_id", "period_start", "period_end")),
    "policy_rate_path_file": ForecastCsvSchema(POLICY_RATE_PATH_COLUMNS, ("scenario_id", "period_start", "period_end")),
    "operating_cash_path_file": ForecastCsvSchema(
        OPERATING_CASH_PATH_COLUMNS,
        ("scenario_id", "period_end"),
    ),
    "cash_reconciliation_residual_file": ForecastCsvSchema(
        CASH_RECONCILIATION_RESIDUAL_COLUMNS,
        ("scenario_id", "period_start", "period_end"),
    ),
    "macro_forecast_path_file": ForecastCsvSchema(
        MACRO_FORECAST_PATH_COLUMNS,
        ("scenario_id", "period_start", "period_end"),
    ),
    "yield_curve_surface_file": ForecastCsvSchema(
        YIELD_CURVE_SURFACE_COLUMNS,
        ("scenario_id", "curve_date", "tenor_years"),
    ),
    "fiscal_incidence_policy_file": ForecastCsvSchema(
        FISCAL_INCIDENCE_POLICY_COLUMNS,
        ("scenario_id", "policy_id"),
    ),
    "net_interest_bridge_file": ForecastCsvSchema(
        NET_INTEREST_BRIDGE_COLUMNS,
        ("scenario_id", "fiscal_year", "component_key"),
    ),
    "holder_absorption_path_file": ForecastCsvSchema(
        HOLDER_ABSORPTION_PATH_COLUMNS,
        ("scenario_id", "quarter", "holder_type", "holder_subbucket"),
    ),
    "fed_holdings_path_file": ForecastCsvSchema(
        FED_HOLDINGS_PATH_COLUMNS,
        ("scenario_id", "period_end", "holder_type"),
    ),
    "frn_rate_path_file": ForecastCsvSchema(
        FRN_RATE_PATH_COLUMNS,
        ("scenario_id", "period_start", "period_end"),
    ),
    "tips_cpi_path_file": ForecastCsvSchema(
        TIPS_CPI_PATH_COLUMNS,
        ("scenario_id", "month"),
    ),
    "tips_real_yield_path_file": ForecastCsvSchema(
        TIPS_REAL_YIELD_PATH_COLUMNS,
        ("scenario_id", "curve_date", "tenor_years"),
    ),
}


def _resolve_path(
    path_value: str | os.PathLike[str] | None,
    base_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    if base_dir:
        return Path(base_dir) / path
    return path


def resolve_baseline_input_paths(
    config: Mapping[str, object],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    reject_unknown: bool = True,
) -> BaselineInputPaths:
    """Resolve the generic ``baseline_input_paths`` config block."""

    paths_config = config.get("baseline_input_paths", config)
    if not isinstance(paths_config, Mapping):
        raise TypeError("baseline_input_paths must be a mapping")
    if reject_unknown:
        unknown = set(paths_config) - set(BASELINE_INPUT_PATH_KEYS)
        if unknown:
            raise ValueError(f"baseline_input_paths has unknown keys: {sorted(unknown)}")
    values = {
        key: _resolve_path(paths_config.get(key), base_dir)
        for key in BASELINE_INPUT_PATH_KEYS
    }
    return BaselineInputPaths(**values)


def _read_required_csv(path: str | os.PathLike[str], *, dataset_name: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} file is missing: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{dataset_name} file is empty: {csv_path}") from exc
    if frame.empty:
        raise ValueError(f"{dataset_name} file is empty: {csv_path}")
    return frame


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, dataset_name: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_name} missing columns: {sorted(missing)}")


def _validate_role_column(
    frame: pd.DataFrame,
    *,
    column: str,
    allowed: frozenset[str],
    dataset_name: str,
) -> None:
    if column not in frame.columns:
        return
    values = frame[column].dropna().astype(str).str.strip()
    invalid = sorted({value for value in values if value and value not in allowed})
    if invalid:
        raise ValueError(f"{dataset_name} has invalid {column} values: {invalid}")


def _validate_no_duplicate_keys(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...],
    *,
    dataset_name: str,
) -> None:
    present_key_columns = [column for column in key_columns if column in frame.columns]
    if len(present_key_columns) != len(key_columns):
        return
    duplicates = frame.duplicated(subset=present_key_columns, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, present_key_columns].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"{dataset_name} has duplicate key rows for {present_key_columns}: {sample}")


def _validate_available_dates(
    frame: pd.DataFrame,
    *,
    actuals_available_as_of: str | pd.Timestamp | None,
    allow_lookahead: bool,
    dataset_name: str,
) -> None:
    if allow_lookahead or actuals_available_as_of is None or "available_date" not in frame.columns:
        return
    cutoff = pd.to_datetime(actuals_available_as_of)
    available = pd.to_datetime(frame["available_date"], errors="coerce")
    future = available > cutoff
    if future.any():
        sample = sorted({str(value.date()) for value in available.loc[future].dropna().head(5)})
        raise ValueError(
            f"{dataset_name} has available_date after actuals_available_as_of {cutoff.date()}: {sample}"
        )


def _validate_date_columns(frame: pd.DataFrame, *, dataset_name: str) -> None:
    for column in sorted(DATE_COLUMNS & set(frame.columns)):
        values = frame[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        parsed = pd.to_datetime(values, errors="coerce")
        if parsed.isna().any():
            invalid = sorted(set(values.loc[parsed.isna()].head(5)))
            raise ValueError(f"{dataset_name} has invalid {column} dates: {invalid}")


def _validate_numeric_columns(frame: pd.DataFrame, *, dataset_name: str) -> None:
    numeric_columns = [
        column
        for column in frame.columns
        if column.endswith(("_bil", "_pct", "_amt", "_share", "_years", "_level", "_decimal", "_days"))
        or column
        in {
            "fiscal_year",
            "source_row_number",
            "tenor_years",
            "nominal_rate",
            "inflation_scalar",
            "day_count_weight",
            "du_share",
            "ru_share",
            "foreign_share",
            "other_share",
            "reference_lag_months",
            "scale_factor",
            "anchor_reference_cpi",
            "cbo_cpi_u_index",
            "tips_cpi_u_index",
        }
    ]
    for column in numeric_columns:
        values = frame[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            continue
        parsed = pd.to_numeric(values, errors="coerce")
        if parsed.isna().any():
            invalid = sorted(set(values.loc[parsed.isna()].head(5)))
            raise ValueError(f"{dataset_name} has invalid numeric {column} values: {invalid}")


def _validate_boolean_columns(frame: pd.DataFrame, *, dataset_name: str) -> None:
    allowed = {"true", "false", "1", "0", "yes", "no"}
    for column in sorted(BOOLEAN_COLUMNS & set(frame.columns)):
        values = frame[column].dropna().astype(str).str.strip()
        values = values[values != ""]
        invalid = sorted({value for value in values if value.lower() not in allowed})
        if invalid:
            raise ValueError(f"{dataset_name} has invalid boolean {column} values: {invalid[:5]}")


def load_forecast_csv(
    path: str | os.PathLike[str],
    *,
    schema_key: str,
    dataset_name: str | None = None,
    actuals_available_as_of: str | pd.Timestamp | None = None,
    allow_lookahead: bool = False,
) -> pd.DataFrame:
    """Load and validate a generic forecast-input CSV."""

    if schema_key not in FORECAST_CSV_SCHEMAS:
        raise KeyError(f"Unknown forecast CSV schema key: {schema_key}")
    schema = FORECAST_CSV_SCHEMAS[schema_key]
    name = dataset_name or schema_key
    frame = _read_required_csv(path, dataset_name=name)
    _require_columns(frame, schema.required_columns, dataset_name=name)
    _validate_no_duplicate_keys(frame, schema.unique_key, dataset_name=name)
    _validate_role_column(frame, column="source_role", allowed=ALLOWED_SOURCE_ROLES, dataset_name=name)
    _validate_role_column(frame, column="runtime_role", allowed=ALLOWED_RUNTIME_ROLES, dataset_name=name)
    _validate_date_columns(frame, dataset_name=name)
    _validate_numeric_columns(frame, dataset_name=name)
    _validate_boolean_columns(frame, dataset_name=name)
    _validate_available_dates(
        frame,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
        dataset_name=name,
    )
    return frame.copy()


def load_cbo_fiscal_baseline(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="cbo_fiscal_baseline_file",
        dataset_name="cbo fiscal baseline",
        **kwargs,
    )


def load_source_fixture_file(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="source_fixture_file",
        dataset_name="source fixture",
        **kwargs,
    )


def load_current_fy_splice(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="current_fy_splice_file",
        dataset_name="current FY splice",
        **kwargs,
    )


def load_debt_stock_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="debt_stock_path_file",
        dataset_name="debt stock path",
        **kwargs,
    )


def load_primary_deficit_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="primary_deficit_path_file",
        dataset_name="primary deficit path",
        **kwargs,
    )


def load_fed_remittance_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path, schema_key="fed_remittance_path_file", dataset_name="Fed remittance path", **kwargs
    )
    if not frame["schema_version"].eq(FED_REMITTANCE_PATH_SCHEMA_VERSION).all():
        raise ValueError("Fed remittance path has an unsupported schema version")
    for column in ("remittance_bil", "annual_or_remaining_remittance_bil"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not ((values >= 0) & (values < float("inf"))).all():
            raise ValueError(f"Fed remittance path has invalid {column} values")
    if not (pd.to_datetime(frame["period_start"]) < pd.to_datetime(frame["period_end"])).all():
        raise ValueError("Fed remittance path periods must start before they end")
    return frame


def load_fed_income_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(path, schema_key="fed_income_path_file", dataset_name="Fed income path", **kwargs)
    if not frame["schema_version"].eq(FED_INCOME_PATH_SCHEMA_VERSION).all():
        raise ValueError("Fed income path has an unsupported schema version")
    if not (pd.to_datetime(frame["period_start"]) < pd.to_datetime(frame["period_end"])).all():
        raise ValueError("Fed income path periods must start before they end")
    nonnegative = (
        "soma_mbs_start_bil", "soma_mbs_end_bil", "soma_mbs_interest_bil", "currency_start_bil", "currency_end_bil",
        "other_deposits_start_bil", "other_deposits_end_bil", "operating_expenses_bil", "dividends_bil",
        "h41_reserve_balances_bil", "h41_tga_bil", "h41_domestic_on_rrp_bil", "opening_domestic_on_rrp_bil",
        "h41_foreign_repo_bil", "h41_soma_mbs_bil", "h41_currency_bil", "h41_other_deposits_bil",
        "remunerated_other_deposit_share",
    )
    for column in nonnegative:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not ((values >= 0) & (values < float("inf"))).all():
            raise ValueError(f"Fed income path has invalid {column} values")
    if frame[list(FED_INCOME_PATH_CONSTANTS)].nunique(dropna=False).gt(1).any():
        raise ValueError("Fed income path opening balance sheet must be constant")
    return frame


def load_policy_rate_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(path, schema_key="policy_rate_path_file", dataset_name="policy-rate path", **kwargs)
    if not frame["schema_version"].eq(POLICY_RATE_PATH_SCHEMA_VERSION).all():
        raise ValueError("policy-rate path has an unsupported schema version")
    starts, ends = pd.to_datetime(frame["period_start"]), pd.to_datetime(frame["period_end"])
    if not (starts < ends).all() or not (starts.iloc[1:].to_numpy() == ends.iloc[:-1].to_numpy()).all():
        raise ValueError("policy-rate path periods must be contiguous and increasing")
    if frame[["iorb_spread_pct", "on_rrp_spread_pct"]].nunique().gt(1).any():
        raise ValueError("policy-rate path spreads must be constant")
    return frame


def policy_rate_for_day(path: pd.DataFrame, day) -> float:
    """The path's fed funds rate in force on ``day``."""

    day = pd.Timestamp(day).normalize()
    starts, ends = pd.to_datetime(path["period_start"]), pd.to_datetime(path["period_end"])
    match = path.loc[(starts <= day) & (day < ends), "fed_funds_rate_pct"]
    if len(match) != 1:
        raise ValueError(f"the policy-rate path has no single rate for {day.date()}")
    return float(match.iloc[0])


def _period_frame(rows: pd.DataFrame, periods: pd.MultiIndex, dataset: str) -> pd.DataFrame:
    if rows["scenario_id"].nunique(dropna=False) != 1:
        raise ValueError(f"compiled {dataset} must have exactly one scenario")
    frame = rows.set_index([pd.to_datetime(rows["period_start"]), pd.to_datetime(rows["period_end"])])
    frame = frame.reindex(periods)
    if frame["scenario_id"].isna().any():
        raise ValueError(f"compiled {dataset} does not cover every simulated period")
    return frame


def _close(actual, expected, label: str, tolerance: float = 1e-6) -> None:
    actual = pd.Series(actual, dtype=float).to_numpy()
    expected = pd.Series(expected, dtype=float).to_numpy()
    gap = abs(actual - expected)
    scale = 1.0 + abs(expected)
    if not (gap <= tolerance * scale).all():
        worst = int(gap.argmax())
        raise ValueError(f"{label} differs from its rebuild at row {worst}: {actual[worst]!r} != {expected[worst]!r}")


_FED_FLOW_OUTPUTS = (
    "CB_CouponCash", "CB_TIPSCouponCash", "CB_FRNCash", "CB_CouponAccrual", "CB_TIPSCouponAccrual", "CB_FRNAccrual",
    "CB_TIPSIndexation", "CB_TIPSFloorTopup", "CB_BillDiscountAccretion", "CB_CouponPremiumAmortization",
    "CB_RealizedGains", "CB_TreasuryInterestIncome", "CB_AccruedInterestPurchased", "CB_AccruedInterestSold",
    "CB_PremiumAcquired", "CB_PremiumSold", "CB_MBSCouponIncome", "CB_MBSPremiumAmortization", "CB_MBSIncome",
    "CB_AgencyDebtCouponIncome", "CB_AgencyDebtPremiumAmortization", "CB_AgencyDebtIncome",
    "CB_OtherIncome", "CB_IORBExpense", "CB_FMUInterestExpense", "CB_DomesticONRRPExpense", "CB_ForeignRepoExpense",
    "CB_OperatingExpenses", "CB_Dividends", "CB_InterestPosted", "CB_ForeignRepoInterestReinvested",
    "CB_TreasuryReserveEffect", "CBOTable43RemittanceInFiscalFlow", "CB_NetIncome", "CB_Remittance",
    "CBORemittanceCashEffect",
)
_FED_LEVEL_OUTPUTS = (
    "CB_PolicyRatePct", "CB_IORBRatePct", "CB_ONRRPRatePct", "CB_TreasuryUnamortizedPremium",
    "CB_TreasuryAccruedInterest", "CB_MBSHoldings", "CB_MBSUnamortizedPremium", "CB_AgencyDebtHoldings",
    "CB_AgencyDebtUnamortizedPremium", "CB_OtherAssets", "CB_Currency",
    "CB_DomesticONRRP", "CB_ForeignRepo", "CB_ForeignOfficialDeposits", "CB_OtherDeposits", "CB_FMUBalances",
    "CB_OtherLiabilitiesAndCapital", "CB_InterestPayable", "CB_ForeignRepoInterestPayable",
    "CB_EarningsRemittancesDue", "CB_ReserveBalances", "CB_DeferredAsset", "TGA",
)


def _per_period(evidence: pd.DataFrame, dates: pd.Series, value: str, mask) -> np.ndarray:
    rows = evidence.loc[mask]
    totals = pd.to_numeric(rows[value], errors="raise").groupby(pd.to_datetime(rows["period_end"])).sum()
    return totals.reindex(dates.iloc[1:]).fillna(0.0).to_numpy()


def rebuild_fed_lot_income(journal: pd.DataFrame, dates: pd.Series, checkpoints=()) -> dict:
    """Rebuild the Fed's acquisition-basis income from its lot journal, vintage by vintage.

    Every opening lot and every acquisition is its own vintage, amortized straight line to
    its maturity from the day after it was acquired; nothing is ever merged. A merge only
    relabels vintages, and a sale or retirement removes the same fraction of every vintage
    of the lot at its carrying value after that day's amortization. An acquisition's premium
    is its dirty cash less accrued interest less stock. Returns per-period flows, the closing
    premium of each simulated day and the premium of every lot left at the close.
    """

    from fed_income import FED_LOT_ACQUISITIONS, FED_LOT_DISPOSALS, FED_LOT_JOURNAL_COLUMNS

    missing = [column for column in FED_LOT_JOURNAL_COLUMNS if column not in journal.columns]
    if missing:
        raise ValueError(f"Fed lot journal omits {missing}")
    day_numbers = (pd.to_datetime(dates).dt.normalize().to_numpy().astype("datetime64[D]").astype(np.int64))
    position = {int(day): i for i, day in enumerate(day_numbers)}
    events = journal.reset_index(drop=True)
    event_days = pd.to_datetime(events["date"], errors="raise").to_numpy().astype("datetime64[D]").astype(np.int64)
    maturities = pd.to_datetime(events["maturity_date"], errors="raise").to_numpy().astype("datetime64[D]").astype(np.int64)
    kinds = events["event"].astype(str).to_numpy()
    unknown = set(kinds) - {"open", "merge", *FED_LOT_ACQUISITIONS, *FED_LOT_DISPOSALS}
    if unknown:
        raise ValueError(f"Fed lot journal has unknown events {sorted(unknown)}")
    if any(int(day) not in position for day in event_days):
        raise ValueError("Fed lot journal books an event on a day the run did not simulate")
    if ((kinds == "open") != (event_days == day_numbers[0])).any():
        raise ValueError("Fed lot journal opens lots only at the opening and trades only after it")
    numbers = {column: pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float) for column in (
        "fraction", "stock_bil", "cash_bil", "accrued_interest_bil", "premium_bil")}
    bond_ids = pd.to_numeric(events["bond_id"], errors="raise").astype(np.int64).to_numpy()
    bills = pd.to_numeric(events["bill"], errors="raise").astype(bool).to_numpy()
    capacity = int(np.isin(kinds, ("open", *FED_LOT_ACQUISITIONS)).sum())
    premium = np.zeros(capacity)
    through = np.zeros(capacity, dtype=np.int64)
    maturity = np.zeros(capacity, dtype=np.int64)
    bill = np.zeros(capacity, dtype=bool)
    live = np.zeros(capacity, dtype=bool)
    vintages: dict[int, list[int]] = {}
    periods = len(dates) - 1
    flows = {name: np.zeros(periods) for name in (
        "bill_accretion", "coupon_amortization", "premium_acquired", "premium_sold", "realized_gains",
        "accrued_purchased", "accrued_sold")}
    closing = np.zeros(len(dates))
    order = np.argsort(event_days, kind="stable")
    cursor, count = 0, 0
    checks = {int(np.datetime64(pd.Timestamp(day).date(), "D").astype(np.int64)) for day in checkpoints}
    live_at = {}
    for i, day in enumerate(day_numbers):
        if i:
            held = np.flatnonzero(live)
            elapsed = day - through[held]
            remaining = maturity[held] - through[held]
            amount = np.where(remaining <= elapsed, premium[held], premium[held] * elapsed / np.maximum(remaining, 1))
            premium[held] -= amount
            through[held] = day
            flows["bill_accretion"][i - 1] = -amount[bill[held]].sum()
            flows["coupon_amortization"][i - 1] = -amount[~bill[held]].sum()
        while cursor < len(order) and event_days[order[cursor]] == day:
            row = order[cursor]
            cursor += 1
            kind, key = kinds[row], int(bond_ids[row])
            stock_bil, cash, accrued = numbers["stock_bil"][row], numbers["cash_bil"][row], numbers["accrued_interest_bil"][row]
            if kind == "merge":
                into = int(pd.to_numeric(events.at[row, "into_bond_id"], errors="raise"))
                vintages.setdefault(into, []).extend(vintages.pop(key, []))
                continue
            if kind in FED_LOT_DISPOSALS:
                held = [index for index in vintages.get(key, []) if live[index]]
                if not held:
                    raise ValueError(f"Fed lot journal disposes of lot {key}, which the Fed does not hold")
                fraction = numbers["fraction"][row]
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(f"Fed lot journal disposes of an invalid fraction {fraction!r} of lot {key}")
                removed = fraction * premium[held].sum()
                premium[held] *= 1.0 - fraction
                if fraction >= 1.0:
                    live[held] = False
                    vintages.pop(key)
                _close([numbers["premium_bil"][row]], [removed], f"premium removed from Fed lot {key}")
                flows["premium_sold"][i - 1] += removed
                flows["realized_gains"][i - 1] += cash - accrued - stock_bil - removed
                flows["accrued_sold"][i - 1] += accrued
                continue
            acquired = numbers["premium_bil"][row] if kind == "open" else cash - accrued - stock_bil
            if kind != "open":
                _close([numbers["premium_bil"][row]], [acquired], f"premium acquired on Fed lot {key}")
                flows["premium_acquired"][i - 1] += acquired
                flows["accrued_purchased"][i - 1] += accrued
            if not np.isfinite(acquired):
                raise ValueError(f"Fed lot {key} opens with a nonfinite premium")
            premium[count], through[count], maturity[count] = acquired, day, maturities[row]
            bill[count], live[count] = bills[row], True
            vintages.setdefault(key, []).append(count)
            count += 1
        matured = np.flatnonzero(live & (maturity <= day))
        if matured.size:
            _close(premium[matured], np.zeros(matured.size), "premium left on a matured Fed lot")
            live[matured] = False
        closing[i] = premium[live].sum()
        if int(day) in checks:
            live_at[str(np.datetime64(int(day), "D"))] = {
                key for key, held in vintages.items() if any(live[index] for index in held)
            }
    lots = {}
    for key, held in vintages.items():
        held = [index for index in held if live[index]]
        if not held:
            continue
        if len(set(maturity[held])) != 1 or len(set(bill[held])) != 1:
            raise ValueError(f"Fed lot {key} holds vintages of different securities")
        lots[key] = {
            "maturity_date": str(np.datetime64(int(maturity[held[0]]), "D")),
            "bill": bool(bill[held[0]]),
            "premium_bil": float(premium[held].sum()),
        }
    return {**flows, "closing_premium": closing, "lots": lots, "live_at": live_at}


def check_earnings_due_closes(results: pd.DataFrame, opening_due: float) -> None:
    """Closing check 1: opening earnings due plus distributable income less cash remitted."""

    income = pd.to_numeric(results["CB_NetIncome"], errors="raise").to_numpy(dtype=float)[1:].sum()
    remitted = pd.to_numeric(results["CBORemittanceCashEffect"], errors="raise").to_numpy(dtype=float)[1:].sum()
    closing = float(pd.to_numeric(results["CB_EarningsRemittancesDue"], errors="raise").iloc[-1])
    _close([closing], [opening_due + income - remitted], "closing earnings due")


def check_fed_balance_sheet_closes(results: pd.DataFrame) -> None:
    """Closing check 2: reserves are Fed assets less every other liability and capital."""

    out = {column: pd.to_numeric(results[column], errors="raise").to_numpy(dtype=float) for column in (
        "DebtHeld_CentralBank", "CB_TreasuryUnamortizedPremium", "CB_TreasuryAccruedInterest", "CB_MBSHoldings",
        "CB_MBSUnamortizedPremium", "CB_AgencyDebtHoldings", "CB_AgencyDebtUnamortizedPremium", "CB_OtherAssets",
        "CB_Currency", "TGA", "CB_DomesticONRRP", "CB_ForeignRepo",
        "CB_ForeignOfficialDeposits", "CB_OtherDeposits", "CB_OtherLiabilitiesAndCapital", "CB_EarningsRemittancesDue",
        "CB_InterestPayable", "CB_ForeignRepoInterestPayable", "CB_ReserveBalances",
    )}
    other_assets = out["CB_OtherAssets"]
    _close(other_assets, np.full(len(other_assets), other_assets[0]), "constant other assets")
    assets = (out["DebtHeld_CentralBank"] + out["CB_TreasuryUnamortizedPremium"] + out["CB_TreasuryAccruedInterest"]
              + out["CB_MBSHoldings"] + out["CB_MBSUnamortizedPremium"] + out["CB_AgencyDebtHoldings"]
              + out["CB_AgencyDebtUnamortizedPremium"] + other_assets)
    liabilities = (out["CB_Currency"] + out["TGA"] + out["CB_DomesticONRRP"] + out["CB_ForeignRepo"]
                   + out["CB_ForeignOfficialDeposits"] + out["CB_OtherDeposits"] + out["CB_OtherLiabilitiesAndCapital"]
                   + out["CB_EarningsRemittancesDue"] + out["CB_InterestPayable"] + out["CB_ForeignRepoInterestPayable"])
    _close(out["CB_ReserveBalances"], assets - liabilities, "reserves against the Fed balance sheet")
    if out["CB_ReserveBalances"].min() < 0.0:
        raise ValueError("reserve balances fall below zero")


def _check_fed_lot_boundaries(lots: pd.DataFrame, rebuilt: dict, *, opening_portfolio: pd.DataFrame,
                              final_portfolio: pd.DataFrame | None, dates: pd.Series, net_premium: float,
                              carried_lots: Mapping[int, float] | None) -> None:
    """The journal opens exactly the Fed's opening lots, each at the premium a carried state
    closed it with or, at an observed opening, at the class approximation; and the lots it
    leaves open are exactly the Fed's lots in the final portfolio, when the run delivers one."""

    from fed_income import TOLERANCE_BIL, is_bill, opening_premiums, stock

    def fed_rows(portfolio: pd.DataFrame) -> pd.DataFrame:
        held = portfolio["HolderType"].astype(str).eq("CB")
        if "Status" in portfolio.columns:
            held &= portfolio["Status"].astype(str).eq("Active")
        return portfolio.loc[held]

    is_open = lots["event"].astype(str).eq("open")
    opened = lots.loc[is_open].set_index(pd.to_numeric(lots.loc[is_open, "bond_id"], errors="raise").astype(int))
    cb = fed_rows(opening_portfolio)
    cb = cb.set_index(pd.to_numeric(cb["BondID"], errors="raise").astype(int), drop=False)
    keys = sorted(cb.index)
    cb = cb.loc[keys]  # every comparison below is by lot, in one order
    if opened.index.has_duplicates or set(opened.index) != set(keys):
        raise ValueError("Fed lot journal does not open exactly the Fed's opening lots")
    opened = opened.loc[keys]
    _close(opened["stock_bil"], stock(cb).loc[keys], "opening Fed lot stock")
    bills = is_bill(cb).loc[keys]
    if (opened["bill"].astype(int).to_numpy() != bills.astype(int).to_numpy()).any():
        raise ValueError("Fed lot journal misclassifies an opening bill")
    if (pd.to_datetime(opened["maturity_date"]).dt.normalize().to_numpy()
            != pd.to_datetime(cb.loc[keys, "MaturityDate"]).dt.normalize().to_numpy()).any():
        raise ValueError("Fed lot journal opens a lot at a maturity the opening portfolio does not hold")
    if carried_lots is not None:
        if set(carried_lots) != set(keys):
            raise ValueError("the carried Fed lots are not the Fed's lots in the opening portfolio")
        _close(opened["premium_bil"], [carried_lots[key] for key in keys], "carried opening Fed lot premium")
        _close([sum(carried_lots.values())], [net_premium], "carried Fed lots against the opening net premium")
    else:
        # A bill's opening discount rests on its issue proceeds: the portfolio's where it states
        # a discount price, else the engine's inference from the opening curve, below par.
        face = pd.to_numeric(cb.loc[keys, "FaceValue"], errors="raise")
        stated = pd.to_numeric(cb.loc[keys, "IssueProceeds"], errors="coerce")
        priced = bills & stated.gt(TOLERANCE_BIL) & stated.lt(face - 1e-12)
        _close(opened.loc[priced, "cash_bil"], stated[priced], "opening Fed bill issue proceeds")
        used = pd.to_numeric(opened["cash_bil"], errors="raise")
        if not (used[bills].gt(0.0) & used[bills].le(face[bills] + TOLERANCE_BIL)).all():
            raise ValueError("Fed lot journal opens a bill at issue proceeds outside zero to par")
        expected = opening_premiums(cb.assign(IssueProceeds=used.to_numpy()), dates.iloc[0], net_premium)
        _close(opened["premium_bil"], [expected[key][1] for key in keys], "opening Fed lot premium")
    if final_portfolio is None:
        return
    last = dates.iloc[-1]
    final = fed_rows(final_portfolio)
    alive = final.loc[pd.to_datetime(final["MaturityDate"]).dt.normalize() > last, "BondID"]
    if set(pd.to_numeric(alive, errors="raise").astype(int)) != set(rebuilt["lots"]):
        raise ValueError("the Fed lots left open by the journal are not the Fed's lots in the final portfolio")


def _rebuild_fed_receivable(portfolio: pd.DataFrame, day: pd.Timestamp, *, opening_day: pd.Timestamp,
                            inputs: Path) -> float:
    """The Fed's accrued-interest receivable at the close of ``day`` from the security terms of
    the lots it holds: coupons and TIPS coupons accrued from their dated dates, owed coupons not
    yet paid, and FRN interest accrued through the day, including closed days the portfolio
    has not yet caught up at the FRN rate path's rates."""

    from fed_income import accrued_interest
    from simulation_calendar import is_fedwire_business_day
    from sim_engine import _cbo_period_end_lookup, _fed_frn_unposted_accrual

    held = portfolio["HolderType"].astype(str).eq("CB")
    if "Status" in portfolio.columns:
        held &= portfolio["Status"].astype(str).eq("Active")
    fed = portfolio.loc[held].copy()
    for column in ("IssueDate", "MaturityDate", "DatedDate", "LastAccrualDate"):
        if column in fed.columns:
            fed[column] = pd.to_datetime(fed[column], errors="coerce")
    receivable = float(accrued_interest(fed, day).to_numpy().sum())
    frn_path = inputs / "tdcsim_frn_rate_path.csv"
    frn_frame = load_frn_rate_path(frn_path, allow_lookahead=True) if frn_path.is_file() else pd.DataFrame()
    last_open = day
    while day > opening_day and not is_fedwire_business_day(last_open.date()):
        last_open -= pd.Timedelta(days=1)
    scenario_id = str(frn_frame["scenario_id"].iloc[0]) if not frn_frame.empty else ""
    return receivable + _fed_frn_unposted_accrual(
        fed, day, t0=opening_day, scenario_id=scenario_id, last_payment_date=max(last_open, opening_day),
        cbo_inputs={"frn_rate_path": frn_frame, "frn_rate_path_date_lookup": _cbo_period_end_lookup(frn_frame)}
        if not frn_frame.empty else {},
    )


def verify_fed_remittances(
    results: pd.DataFrame,
    inputs_dir: str | os.PathLike[str],
    *,
    lots: pd.DataFrame,
    opening_portfolio: pd.DataFrame,
    final_portfolio: pd.DataFrame | None,
    year_end_lots: pd.DataFrame,
    carried_lots: Mapping[int, float] | None = None,
    trc_results: pd.DataFrame | None = None,
    ledger: pd.DataFrame | None = None,
    payments: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Rebuild the Fed's earnings, balance sheet and remittances from a run's outputs.

    Inputs come from the compiled package: the Table 4-3 path, the Fed income path and the
    policy-rate path. The Fed's amortization, premiums and realized gains are rebuilt from
    its lot journal (``lots``): the opening lots at the premiums a derived state carried
    (``carried_lots``) or, at an observed opening, at the class approximation applied to
    ``opening_portfolio``; then each trade's quantity and price, bound to the run's trade
    cash and to the Fed's lots in ``final_portfolio`` (None when the run delivers no final
    portfolio; the result says whether the lots were bound to it). When the run's settlement
    evidence is given, the Fed's Treasury cash receipts, the TGA, the modeled remittance and
    the Table 4-3 fiscal flow are bound to it. Returns the opening and closing earnings due, the
    cumulative net income and remittances and the largest remittance.
    """

    from simulation_calendar import is_fedwire_business_day, next_fedwire_business_day

    inputs = Path(inputs_dir)
    dates = pd.to_datetime(results["Date"], errors="raise").reset_index(drop=True)
    periods = pd.MultiIndex.from_arrays([dates.iloc[:-1], dates.iloc[1:]])
    out = {
        column: pd.to_numeric(results[column], errors="raise").reset_index(drop=True).to_numpy(dtype=float)
        for column in (*_FED_FLOW_OUTPUTS, *_FED_LEVEL_OUTPUTS, "DebtHeld_CentralBank", "PrimaryDeficit")
    }
    flows = {name: values[1:] for name, values in out.items() if name in _FED_FLOW_OUTPUTS}
    table43 = _period_frame(load_fed_remittance_path(inputs / FED_REMITTANCE_PATH_FILE), periods, "Fed remittance path")
    _close(flows["CBOTable43RemittanceInFiscalFlow"], table43["remittance_bil"], "Table 4-3 remittance in the fiscal flow")
    statuses = [CBO_FED_REMITTANCE_OPENING_STATUS, *[CBO_FED_REMITTANCE_STATUS] * len(periods)]
    if results["CBORemittanceStatus"].tolist() != statuses:
        raise ValueError("Fed remittance cash path has unexpected status")
    path = _period_frame(load_fed_income_path(inputs / FED_INCOME_PATH_FILE), periods, "Fed income path")
    first = path.iloc[0]
    for column, output in (
        ("soma_mbs_interest_bil", "CB_MBSCouponIncome"),
        ("agency_debt_interest_bil", "CB_AgencyDebtCouponIncome"),
        ("other_income_bil", "CB_OtherIncome"),
        ("operating_expenses_bil", "CB_OperatingExpenses"),
        ("dividends_bil", "CB_Dividends"),
    ):
        _close(flows[output], path[column], output)
    for stem, output in (
        ("soma_mbs", "CB_MBSHoldings"), ("soma_mbs_premium", "CB_MBSUnamortizedPremium"), ("currency", "CB_Currency"),
        ("agency_debt", "CB_AgencyDebtHoldings"), ("agency_debt_premium", "CB_AgencyDebtUnamortizedPremium"),
        ("other_deposits", "CB_OtherDeposits"), ("other_liabilities_and_capital", "CB_OtherLiabilitiesAndCapital"),
    ):
        _close(out[output], [first[f"{stem}_start_bil"], *path[f"{stem}_end_bil"]], output)
    _close(flows["CB_MBSPremiumAmortization"], -np.diff(out["CB_MBSUnamortizedPremium"]), "CB_MBSPremiumAmortization")
    _close(flows["CB_MBSIncome"], flows["CB_MBSCouponIncome"] - flows["CB_MBSPremiumAmortization"], "CB_MBSIncome")
    _close(flows["CB_AgencyDebtPremiumAmortization"], -np.diff(out["CB_AgencyDebtUnamortizedPremium"]),
           "CB_AgencyDebtPremiumAmortization")
    _close(flows["CB_AgencyDebtIncome"], flows["CB_AgencyDebtCouponIncome"] - flows["CB_AgencyDebtPremiumAmortization"],
           "CB_AgencyDebtIncome")
    _close(out["CB_FMUBalances"], out["CB_OtherDeposits"] * float(first["remunerated_other_deposit_share"]), "FMU balances")
    # Administered rates come from the compiled policy-rate path, not from any curve.
    policy = load_policy_rate_path(inputs / POLICY_RATE_PATH_FILE)
    iorb_spread, rrp_spread = float(policy["iorb_spread_pct"].iloc[0]), float(policy["on_rrp_spread_pct"].iloc[0])
    policy_rate = np.array([policy_rate_for_day(policy, day) for day in dates])
    rrp_policy = []
    for day in dates:
        open_day = day
        while not is_fedwire_business_day(open_day.date()):
            open_day -= pd.Timedelta(days=1)
        rrp_policy.append(policy_rate_for_day(policy, max(open_day, dates.iloc[0])))
    _close(out["CB_PolicyRatePct"], policy_rate, "policy rate")
    _close(out["CB_IORBRatePct"], policy_rate + iorb_spread, "IORB rate")
    _close(out["CB_ONRRPRatePct"], np.array(rrp_policy) + rrp_spread, "ON RRP rate")
    # Treasury income: accruals roll the receivable, amortization rolls the premium.
    _close(flows["CB_TreasuryInterestIncome"], sum(flows[name] for name in (
        "CB_CouponAccrual", "CB_TIPSCouponAccrual", "CB_FRNAccrual", "CB_TIPSIndexation", "CB_TIPSFloorTopup",
        "CB_BillDiscountAccretion", "CB_CouponPremiumAmortization", "CB_RealizedGains",
    )), "Treasury interest income")
    accruals = flows["CB_CouponAccrual"] + flows["CB_TIPSCouponAccrual"] + flows["CB_FRNAccrual"]
    cash = flows["CB_CouponCash"] + flows["CB_TIPSCouponCash"] + flows["CB_FRNCash"]
    _close(np.diff(out["CB_TreasuryAccruedInterest"]),
           accruals - cash + flows["CB_AccruedInterestPurchased"] - flows["CB_AccruedInterestSold"],
           "Fed accrued interest receivable")
    _close(np.diff(out["CB_TreasuryUnamortizedPremium"]),
           flows["CB_PremiumAcquired"] - flows["CB_PremiumSold"] + flows["CB_BillDiscountAccretion"]
           + flows["CB_CouponPremiumAmortization"], "Fed unamortized premium")
    # The acquisition basis rebuilt lot by lot from the opening and each trade's evidence.
    year_ends = [day for day in dates.iloc[1:] if day.month == 9 and day.day == 30]
    rebuilt = rebuild_fed_lot_income(lots, dates, checkpoints=year_ends)
    for name, output in (
        ("bill_accretion", "CB_BillDiscountAccretion"), ("coupon_amortization", "CB_CouponPremiumAmortization"),
        ("premium_acquired", "CB_PremiumAcquired"), ("premium_sold", "CB_PremiumSold"),
        ("realized_gains", "CB_RealizedGains"), ("accrued_purchased", "CB_AccruedInterestPurchased"),
        ("accrued_sold", "CB_AccruedInterestSold"),
    ):
        _close(flows[output], rebuilt[name], f"{output} against the Fed lot journal")
    _close(out["CB_TreasuryUnamortizedPremium"], rebuilt["closing_premium"], "Fed unamortized premium against its lots")
    # The receivable at both ends, from the security terms of the lots held: with cash bound to
    # the payments and accrued interest bought and sold bound to the lot journal, this pins the
    # interest the Fed earned over the run.
    _close([out["CB_TreasuryAccruedInterest"][0]],
           [_rebuild_fed_receivable(opening_portfolio, dates.iloc[0], opening_day=dates.iloc[0], inputs=inputs)],
           "opening Fed accrued-interest receivable against its lots' terms")
    # And at every fiscal-year end, from the Fed's lots there: exactly the lots the journal leaves
    # open, so income cannot move across a fiscal-year boundary.
    recorded = pd.to_datetime(year_end_lots["date"]).dt.normalize()
    if not set(recorded) <= set(year_ends):
        raise ValueError("the Fed's year-end lots name a day that is not one of the run's fiscal-year ends")
    if pd.DataFrame({"date": recorded, "bond": pd.to_numeric(year_end_lots["BondID"], errors="raise")}).duplicated().any():
        raise ValueError("the Fed's year-end lots record a lot twice on one day")
    from fed_income import stock as lot_stock

    position = {day: i for i, day in enumerate(dates)}
    for day in year_ends:
        held = year_end_lots.loc[recorded.eq(day)]
        alive = held.loc[pd.to_datetime(held["MaturityDate"]).dt.normalize() > day, "BondID"]
        if set(pd.to_numeric(alive, errors="raise").astype(int)) != rebuilt["live_at"][str(day.date())]:
            raise ValueError(f"the Fed's lots at {day.date()} are not the lots its journal leaves open")
        _close([float(lot_stock(held).sum()) if not held.empty else 0.0], [out["DebtHeld_CentralBank"][position[day]]],
               f"the Fed's lots at {day.date()} against its reported Treasury stock")
        _close([out["CB_TreasuryAccruedInterest"][position[day]]],
               [_rebuild_fed_receivable(held, day, opening_day=dates.iloc[0], inputs=inputs)],
               f"Fed accrued-interest receivable at {day.date()} against its lots' terms")
    if final_portfolio is not None:
        _close([out["CB_TreasuryAccruedInterest"][-1]],
               [_rebuild_fed_receivable(final_portfolio, dates.iloc[-1], opening_day=dates.iloc[0], inputs=inputs)],
               "closing Fed accrued-interest receivable against its lots' terms")
    _check_fed_lot_boundaries(lots, rebuilt, opening_portfolio=opening_portfolio, final_portfolio=final_portfolio,
                              dates=dates, net_premium=float(first["h41_treasury_net_premium_bil"]),
                              carried_lots=carried_lots)
    lot_days = pd.to_datetime(lots["date"]).dt.normalize()
    lot_cash = pd.to_numeric(lots["cash_bil"], errors="coerce")
    for event, output in (("purchase", "CBOFedSecondaryPurchaseCash"), ("sale", "CBOFedSecondarySaleCash")):
        traded = lot_cash[lots["event"].eq(event)].groupby(lot_days[lots["event"].eq(event)]).sum()
        reported = pd.to_numeric(results[output], errors="raise").fillna(0.0).to_numpy(dtype=float)[1:]
        _close(reported, traded.reindex(dates.iloc[1:]).fillna(0.0), f"Fed lot journal against {output}")
    retired = lots["event"].eq("retirement")
    _close(lot_cash[retired], pd.to_numeric(lots.loc[retired, "stock_bil"], errors="raise"),
           "a retired Fed lot is paid its stock")
    if payments is not None:
        fed = payments["holder_sector"].astype(str).eq("CB")
        for kind, output in (("fixed_coupon", "CB_CouponCash"), ("tips_coupon", "CB_TIPSCouponCash"),
                             ("frn_interest", "CB_FRNCash"), ("tips_deflation_floor_topup", "CB_TIPSFloorTopup")):
            settled = _per_period(payments, dates, "amount_bil", fed & payments["payment_type"].eq(kind))
            _close(flows[output], settled, f"{output} against the run's payments to the Fed")
    if ledger is not None:
        leg = ledger["leg_type"].astype(str)
        holder = ledger["holder_sector"].astype(str)
        indexed = _per_period(ledger, dates, "adjusted_principal_change_bil",
                              holder.eq("CB") & ledger["event_type"].astype(str).eq("indexation"))
        _close(flows["CB_TIPSIndexation"], indexed, "Fed TIPS indexation against the journal")
        remitted = _per_period(ledger, dates, "treasury_cash_change_bil", leg.eq("central_bank_remittance"))
        _close(flows["CBORemittanceCashEffect"], remitted, "modeled remittance against the TGA journal")
        fiscal = _per_period(ledger, dates, "treasury_cash_change_bil", leg.eq("tax_and_primary_spending_cash"))
        _close(fiscal, -(out["PrimaryDeficit"][1:] + flows["CBOTable43RemittanceInFiscalFlow"]),
               "Table 4-3 remittance against the fiscal journal")
        journal_cash = _per_period(ledger, dates, "treasury_cash_change_bil", np.ones(len(ledger), dtype=bool))
        _close(np.diff(out["TGA"]), journal_cash, "TGA against the Treasury cash journal")
    # Reserves: the Treasury reserve effect is the run's own, bound to TRC or the legacy path;
    # the named cash reconciliation residual reaches the TGA from outside and settles through reserves.
    residual = pd.to_numeric(results["CBOCashReconciliationResidual"], errors="raise").to_numpy(dtype=float)[1:]
    if trc_results is not None:
        trc = trc_results.assign(Date=pd.to_datetime(trc_results["Date"])).set_index("Date").reindex(dates)
        if trc[["TRC_Change", "ONRRP_MMF", "ForeignOfficialFedRRP", "ForeignOfficialFedDeposits"]].isna().any().any():
            raise ValueError("TRC results do not cover every simulated date")
        _close(flows["CB_TreasuryReserveEffect"], trc["TRC_Change"].iloc[1:] - residual, "Treasury reserve effect")
        _close(out["CB_DomesticONRRP"], trc["ONRRP_MMF"], "domestic ON RRP against the routed ledger")
        _close(out["CB_ForeignOfficialDeposits"], trc["ForeignOfficialFedDeposits"], "foreign official deposits")
        foreign_base = trc["ForeignOfficialFedRRP"].to_numpy()
    else:
        legacy = pd.to_numeric(results["Reserves"], errors="raise").to_numpy()
        _close(flows["CB_TreasuryReserveEffect"], np.diff(legacy) - residual, "Treasury reserve effect")
        _close(out["CB_DomesticONRRP"], np.full(len(dates), float(first["opening_domestic_on_rrp_bil"])), "domestic ON RRP")
        _close(out["CB_ForeignOfficialDeposits"], np.full(len(dates), float(first["h41_foreign_official_deposits_bil"])),
               "foreign official deposits")
        foreign_base = np.full(len(dates), float(first["h41_foreign_repo_bil"]))
    _close(out["CB_ForeignRepo"], foreign_base + np.concatenate([[0.0], np.cumsum(flows["CB_ForeignRepoInterestReinvested"])]),
           "foreign repo pool with reinvested interest")
    # Opening bridge from the H.4.1 Wednesday release.
    opening_reserves = (
        float(first["h41_reserve_balances_bil"])
        - (out["TGA"][0] - float(first["h41_tga_bil"]))
        - (out["CB_DomesticONRRP"][0] - float(first["h41_domestic_on_rrp_bil"]))
        - (out["CB_ForeignRepo"][0] - float(first["h41_foreign_repo_bil"]))
        - (out["CB_ForeignOfficialDeposits"][0] - float(first["h41_foreign_official_deposits_bil"]))
        + (out["CB_MBSHoldings"][0] - float(first["h41_soma_mbs_bil"]))
        - (out["CB_Currency"][0] - float(first["h41_currency_bil"]))
        - (out["CB_OtherDeposits"][0] - float(first["h41_other_deposits_bil"]))
        - (out["CB_OtherLiabilitiesAndCapital"][0] - float(first["h41_other_liabilities_and_capital_bil"]))
    )
    _close([out["CB_ReserveBalances"][0]], [opening_reserves], "opening reserve balances")
    if opening_reserves < 0.0:
        raise ValueError("opening reserve balances are negative")
    # Day by day: posting, reserves, expenses, earnings due and the weekly payment.
    days = np.diff(dates).astype("timedelta64[D]").astype(float)
    open_days = np.array([is_fedwire_business_day(day.date()) for day in dates])
    payable, foreign_payable = out["CB_InterestPayable"], out["CB_ForeignRepoInterestPayable"]
    _close([payable[0], foreign_payable[0]], [float(first["opening_interest_payable_bil"]),
           float(first["opening_foreign_repo_interest_payable_bil"])], "opening interest payable")
    _close(flows["CB_InterestPosted"], np.where(open_days[1:], payable[:-1], 0.0), "interest posted to reserves on the next business day")
    _close(flows["CB_ForeignRepoInterestReinvested"], np.where(open_days[1:], foreign_payable[:-1], 0.0),
           "foreign repo pool interest reinvested on the next business day")
    expenses = (flows["CB_IORBExpense"] + flows["CB_FMUInterestExpense"] + flows["CB_DomesticONRRPExpense"]
                + flows["CB_ForeignRepoExpense"])
    _close(np.diff(payable), expenses - flows["CB_ForeignRepoExpense"] - flows["CB_InterestPosted"], "interest payable")
    _close(np.diff(foreign_payable), flows["CB_ForeignRepoExpense"] - flows["CB_ForeignRepoInterestReinvested"],
           "foreign repo pool interest payable")
    accrual = days / 360.0
    for output, rate, balance in (
        ("CB_IORBExpense", out["CB_IORBRatePct"], out["CB_ReserveBalances"]),
        ("CB_FMUInterestExpense", out["CB_IORBRatePct"], out["CB_FMUBalances"]),
        ("CB_DomesticONRRPExpense", out["CB_ONRRPRatePct"], out["CB_DomesticONRRP"]),
        ("CB_ForeignRepoExpense", out["CB_ONRRPRatePct"], out["CB_ForeignRepo"]),
    ):
        _close(flows[output], rate[1:] / 100.0 * balance[1:] * accrual, f"{output} on the day's closing balance")
    _close(np.diff(out["CB_ReserveBalances"]),
           flows["CB_TreasuryReserveEffect"] + flows["CB_InterestPosted"] + np.diff(out["CB_MBSHoldings"])
           - flows["CB_MBSCouponIncome"] + np.diff(out["CB_AgencyDebtHoldings"]) - flows["CB_AgencyDebtCouponIncome"]
           - np.diff(out["CB_Currency"]) - np.diff(out["CB_OtherDeposits"])
           - np.diff(out["CB_OtherLiabilitiesAndCapital"]) - flows["CB_OtherIncome"] + flows["CB_OperatingExpenses"]
           + flows["CB_Dividends"], "reserve balances roll-forward")
    net_income = (flows["CB_TreasuryInterestIncome"] + flows["CB_MBSIncome"] + flows["CB_AgencyDebtIncome"]
                  + flows["CB_OtherIncome"] - expenses
                  - flows["CB_OperatingExpenses"] - flows["CB_Dividends"])
    _close(flows["CB_NetIncome"], net_income, "CB_NetIncome")
    # The pooled account and the Financial Accounting Manual's weekly rule (paragraph 60.55).
    history = {
        pd.Timestamp(first["previous_wednesday_date"]): float(first["previous_wednesday_due_bil"]),
        pd.Timestamp(first["year_end_date"]): float(first["year_end_due_bil"]),
        pd.Timestamp(first["h41_as_of_date"]): float(first["h41_earnings_remittances_due_bil"]),
        dates.iloc[0]: float(first["h41_earnings_remittances_due_bil"]),
    }
    due = float(first["h41_earnings_remittances_due_bil"])
    paid = []
    for i in range(1, len(dates)):
        day = dates.iloc[i]
        wednesday = day - pd.Timedelta(days=(day.weekday() - 2) % 7)
        pay_day = pd.Timestamp(next_fedwire_business_day(wednesday.date())) == day
        amount = 0.0
        if pay_day:
            january_first = pd.Timestamp(year=day.year, month=1, day=1)
            first_wednesday = january_first + pd.Timedelta(days=(2 - january_first.weekday()) % 7)
            if pd.Timestamp(next_fedwire_business_day(first_wednesday.date())) == day:
                basis_day = pd.Timestamp(year=day.year - 1, month=12, day=31)
            else:
                basis_day = wednesday - pd.Timedelta(days=7)
            basis = history[max(when for when in history if when <= basis_day)]
            before = history[max(when for when in history if when <= day - pd.Timedelta(days=1))]
            amount = min(basis, before) if basis > 0.0 and before > 0.0 else 0.0
        paid.append(amount)
        due += net_income[i - 1] - amount
        history[day] = due
    _close(flows["CB_Remittance"], paid, "CB_Remittance")
    _close(flows["CBORemittanceCashEffect"], paid, "Fed-to-TGA remittance cash")
    _close(out["CB_EarningsRemittancesDue"], [history[day] for day in dates], "CB_EarningsRemittancesDue")
    _close(out["CB_DeferredAsset"], np.maximum(0.0, -out["CB_EarningsRemittancesDue"]), "CB_DeferredAsset")
    check_earnings_due_closes(results, float(first["h41_earnings_remittances_due_bil"]))
    check_fed_balance_sheet_closes(results)
    return {
        "opening_earnings_due_bil": float(first["h41_earnings_remittances_due_bil"]),
        "closing_earnings_due_bil": float(out["CB_EarningsRemittancesDue"][-1]),
        "cumulative_net_income_bil": float(net_income.sum()),
        "cumulative_remittances_bil": float(flows["CBORemittanceCashEffect"].sum()),
        "max_remittance_bil": float(max(paid, default=0.0)),
        "fed_lots_bound_to_final_portfolio": final_portfolio is not None,
    }


def load_operating_cash_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="operating_cash_path_file",
        dataset_name="operating cash path",
        **kwargs,
    )


def load_cash_reconciliation_residual(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="cash_reconciliation_residual_file",
        dataset_name="cash reconciliation residual",
        **kwargs,
    )


def load_macro_forecast_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="macro_forecast_path_file",
        dataset_name="macro forecast path",
        **kwargs,
    )


def load_yield_curve_surface(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="yield_curve_surface_file",
        dataset_name="yield curve surface",
        **kwargs,
    )


def load_fiscal_incidence_policy(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="fiscal_incidence_policy_file",
        dataset_name="fiscal incidence policy",
        **kwargs,
    )


def load_net_interest_bridge(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    return load_forecast_csv(
        path,
        schema_key="net_interest_bridge_file",
        dataset_name="net interest bridge",
        **kwargs,
    )


def load_holder_absorption_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path,
        schema_key="holder_absorption_path_file",
        dataset_name="holder absorption path",
        **kwargs,
    )
    pref_cols = [column for column in frame.columns if column.endswith("_pct")]
    if not pref_cols:
        raise ValueError("holder absorption path has no *_pct preference columns")
    return frame


def load_fed_holdings_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path,
        schema_key="fed_holdings_path_file",
        dataset_name="fed holdings path",
        **kwargs,
    )
    unsupported = sorted(set(frame["holder_type"].dropna().astype(str)) - {"CB"})
    if unsupported:
        raise ValueError(f"fed holdings path supports only CB holder_type rows: {unsupported}")
    return frame


def load_frn_rate_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path,
        schema_key="frn_rate_path_file",
        dataset_name="FRN rate path",
        **kwargs,
    )
    for column in ("benchmark_rate_decimal", "auction_high_rate_decimal", "money_market_yield_decimal"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"FRN rate path has nonnumeric {column} values")
        if (values.abs() > 1.0).any():
            raise ValueError(f"FRN rate path {column} values must be decimal rates")
    day_count = pd.to_numeric(frame["day_count_basis"], errors="coerce")
    if day_count.isna().any() or (day_count <= 0).any():
        raise ValueError("FRN rate path day_count_basis must be positive")
    starts = pd.to_datetime(frame["period_start"], errors="coerce").dt.normalize()
    ends = pd.to_datetime(frame["period_end"], errors="coerce").dt.normalize()
    if (starts >= ends).any():
        raise ValueError("FRN rate path rows must have period_start before period_end")
    return frame


def load_tips_cpi_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path,
        schema_key="tips_cpi_path_file",
        dataset_name="TIPS CPI path",
        **kwargs,
    )
    for column in ("cbo_cpi_u_index", "tips_cpi_u_index", "anchor_reference_cpi", "scale_factor"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values <= 0.0).any():
            raise ValueError(f"TIPS CPI path {column} values must be positive")
    lag = pd.to_numeric(frame["reference_lag_months"], errors="coerce")
    if lag.isna().any() or (lag < 0).any():
        raise ValueError("TIPS CPI path reference_lag_months must be nonnegative")
    months = pd.to_datetime(frame["month"], errors="coerce").dt.normalize()
    if months.isna().any():
        raise ValueError("TIPS CPI path has invalid month values")
    if not months.dt.day.eq(1).all():
        raise ValueError("TIPS CPI path month values must be first-of-month dates")
    return frame


def load_tips_real_yield_path(path: str | os.PathLike[str], **kwargs) -> pd.DataFrame:
    frame = load_forecast_csv(
        path,
        schema_key="tips_real_yield_path_file",
        dataset_name="TIPS real-yield path",
        **kwargs,
    )
    for column in (
        "nominal_rate_decimal",
        "expected_inflation_decimal",
        "real_yield_decimal",
        "real_coupon_decimal",
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"TIPS real-yield path has nonnumeric {column} values")
        if (values.abs() > 1.0).any():
            raise ValueError(f"TIPS real-yield path {column} values must be decimal rates")
    if (pd.to_numeric(frame["real_coupon_decimal"], errors="coerce") < 0.0).any():
        raise ValueError("TIPS real-yield path real_coupon_decimal must be nonnegative")
    return frame


def load_baseline_input_paths(
    config: Mapping[str, object],
    *,
    base_dir: str | os.PathLike[str] | None = None,
    actuals_available_as_of: str | pd.Timestamp | None = None,
    allow_lookahead: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load every configured generic baseline CSV path."""

    paths = resolve_baseline_input_paths(config, base_dir=base_dir)
    loaders = {
        "source_fixture_file": load_source_fixture_file,
        "cbo_fiscal_baseline_file": load_cbo_fiscal_baseline,
        "current_fy_splice_file": load_current_fy_splice,
        "debt_stock_path_file": load_debt_stock_path,
        "primary_deficit_path_file": load_primary_deficit_path,
        "fed_remittance_path_file": load_fed_remittance_path,
        "fed_income_path_file": load_fed_income_path,
        "policy_rate_path_file": load_policy_rate_path,
        "operating_cash_path_file": load_operating_cash_path,
        "cash_reconciliation_residual_file": load_cash_reconciliation_residual,
        "macro_forecast_path_file": load_macro_forecast_path,
        "yield_curve_surface_file": load_yield_curve_surface,
        "fiscal_incidence_policy_file": load_fiscal_incidence_policy,
        "net_interest_bridge_file": load_net_interest_bridge,
        "holder_absorption_path_file": load_holder_absorption_path,
        "fed_holdings_path_file": load_fed_holdings_path,
        "frn_rate_path_file": load_frn_rate_path,
        "tips_cpi_path_file": load_tips_cpi_path,
        "tips_real_yield_path_file": load_tips_real_yield_path,
    }
    loaded: dict[str, pd.DataFrame] = {}
    for key, loader in loaders.items():
        path = getattr(paths, key)
        if path is None:
            continue
        loaded[key] = loader(
            path,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    return loaded


__all__ = [
    "ALLOWED_RUNTIME_ROLES",
    "ALLOWED_SOURCE_ROLES",
    "OPERATING_CASH_COMPARISON_ROLE",
    "OPERATING_CASH_OPENING_STATE_ROLE",
    "BASELINE_INPUT_PATH_KEYS",
    "BaselineInputPaths",
    "FORECAST_CSV_SCHEMAS",
    "load_baseline_input_paths",
    "load_cash_reconciliation_residual",
    "load_cbo_fiscal_baseline",
    "load_current_fy_splice",
    "load_debt_stock_path",
    "load_fiscal_incidence_policy",
    "load_forecast_csv",
    "load_holder_absorption_path",
    "load_fed_holdings_path",
    "load_fed_income_path",
    "load_policy_rate_path",
    "load_fed_remittance_path",
    "load_frn_rate_path",
    "load_tips_cpi_path",
    "load_tips_real_yield_path",
    "load_macro_forecast_path",
    "operating_cash_runtime_role",
    "load_net_interest_bridge",
    "load_operating_cash_path",
    "load_primary_deficit_path",
    "load_source_fixture_file",
    "load_yield_curve_surface",
    "resolve_baseline_input_paths",
]
