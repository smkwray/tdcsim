"""Generic forecast-input path resolution and CSV loaders."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Mapping

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

ALLOWED_RUNTIME_ROLES = frozenset(
    {
        "hard_target",
        "hard_flow",
        "check_only",
        "reconciliation_only",
        "memo_only",
    }
)

BASELINE_INPUT_PATH_KEYS = (
    "source_contract_file",
    "source_fixture_file",
    "cbo_fiscal_baseline_file",
    "current_fy_splice_file",
    "debt_stock_path_file",
    "primary_deficit_path_file",
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
    "load_frn_rate_path",
    "load_tips_cpi_path",
    "load_tips_real_yield_path",
    "load_macro_forecast_path",
    "load_net_interest_bridge",
    "load_operating_cash_path",
    "load_primary_deficit_path",
    "load_source_fixture_file",
    "load_yield_curve_surface",
    "resolve_baseline_input_paths",
]
