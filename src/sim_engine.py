
"""Core simulation loop for the Treasury funding-chain simulator."""

import copy
import time
import traceback
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from dateutil.relativedelta import relativedelta

from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    DAYS_PER_YEAR_ACTUAL,
    FRN_DAY_COUNT_BASIS,
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    MATURITY_CATEGORIES,
    MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
    MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
    MMF_DEPOSIT_PASS_THROUGH_STATUS,
    PREFERENCE_CATEGORIES,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
    PRIVATE_SUBBUCKETS,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
)
from sim_helpers import OUTPUT_COLUMN_RENAMES, apply_event_actions, validate_run_params
from sim_pricing import (
    calculate_coupon_rate,
    calculate_face_from_proceeds_target,
    get_coupon_dates_in_period,
    get_maturity_category,
    get_security_category_for_prefs,
    get_yield_for_maturity,
    infer_issue_data_for_loaded_bill,
    quote_issuance_from_face_target,
)
from sim_trading import execute_preference_trades
from tdc_validation import is_coarse_frequency
from tips_indexation import build_projected_cpi_lookup_from_macro, build_tips_daily_cpi_lookups_from_monthly_path
from forecast_paths import (
    load_cbo_fiscal_baseline,
    load_cash_reconciliation_residual,
    load_debt_stock_path,
    load_fiscal_incidence_policy,
    load_fed_holdings_path,
    load_frn_rate_path,
    load_macro_forecast_path,
    load_net_interest_bridge,
    load_operating_cash_path,
    load_primary_deficit_path,
    load_tips_cpi_path,
    load_tips_real_yield_path,
    resolve_baseline_input_paths,
)
from funding_plan import (
    calculate_pre_issuance_controlled_debt,
    calculate_required_face_issuance,
)
from ratewall_paths import (
    holder_preferences_for_period,
    load_holder_absorption_path,
    load_primary_flow_path,
    primary_flow_for_period,
)
from yield_curve_path import curve_for_date, load_yield_curve_surface


CBO_FUNDING_MODE = 'cbo_public_debt_target'
CASH_FUNDING_MODE = 'cash_tga_target'
_US_FEDERAL_HOLIDAYS = USFederalHolidayCalendar()
_BUSINESS_DAY_CACHE = {}
_US_FEDERAL_HOLIDAY_DATES = None


def _is_cbo_funding_mode(params):
    funding_rule = params.get('funding_rule', {})
    if not isinstance(funding_rule, dict):
        return False
    return str(funding_rule.get('mode', '')).strip() == CBO_FUNDING_MODE


def _federal_fiscal_year(date_value):
    ts = pd.to_datetime(date_value)
    return int(ts.year + 1 if ts.month >= 10 else ts.year)


def _scenario_candidates_for_engine(scenario_name):
    return [str(scenario_name), 'all', 'default', '']


def _filter_scenario_rows(frame, scenario_name):
    if frame is None or frame.empty:
        return pd.DataFrame()
    if 'scenario_id' not in frame.columns:
        return frame.copy()
    scenario_values = frame['scenario_id'].fillna('').astype(str)
    for candidate in _scenario_candidates_for_engine(scenario_name):
        rows = frame.loc[scenario_values == candidate]
        if not rows.empty:
            return rows.copy()
    return pd.DataFrame()


def _single_cbo_row(frame, scenario_name, *, dataset_name, date_column=None, date_value=None, fiscal_year=None):
    rows = _filter_scenario_rows(frame, scenario_name)
    if rows.empty:
        raise ValueError(f'No {dataset_name} rows match scenario {scenario_name!r}.')
    if date_column is not None:
        target_date = pd.to_datetime(date_value).normalize()
        row_dates = pd.to_datetime(rows[date_column], errors='coerce').dt.normalize()
        rows = rows.loc[row_dates == target_date]
    if fiscal_year is not None and 'fiscal_year' in rows.columns:
        rows = rows.loc[pd.to_numeric(rows['fiscal_year'], errors='coerce') == int(fiscal_year)]
    if rows.empty:
        suffix = f' at {pd.to_datetime(date_value).date()}' if date_value is not None else ''
        if fiscal_year is not None:
            suffix += f' for fiscal year {int(fiscal_year)}'
        raise ValueError(f'No {dataset_name} row matches scenario {scenario_name!r}{suffix}.')
    if len(rows.index) > 1:
        rows = rows.sort_index()
    return rows.iloc[0]


def _cbo_value(row, column, default=0.0):
    if row is None or column not in row.index or pd.isna(row[column]):
        return default
    return float(row[column])


def _cbo_incidence_from_config(params):
    policy = params.get('fiscal_incidence_policy', {})
    if not isinstance(policy, dict) or not policy:
        return None
    if 'du_share' not in policy:
        return None
    return {
        'du_share': float(policy.get('du_share', 0.0) or 0.0),
        'ru_share': float(policy.get('ru_share', 0.0) or 0.0),
        'status': 'explicit_config_policy',
    }


def _cbo_fiscal_baseline_row(cbo_engine_inputs, scenario_name, current_date):
    baseline = cbo_engine_inputs.get('cbo_fiscal_baseline')
    if baseline is None or baseline.empty:
        return None
    try:
        return _single_cbo_row(
            baseline,
            scenario_name,
            dataset_name='CBO fiscal baseline',
            fiscal_year=_federal_fiscal_year(current_date),
        )
    except ValueError:
        return None


def _cbo_macro_cpi_level(cbo_engine_inputs, scenario_name, current_date, default_value):
    if not isinstance(cbo_engine_inputs, dict):
        return default_value
    macro = cbo_engine_inputs.get('macro_forecast')
    if macro is None or macro.empty or 'cbo_cpi_u_index' not in macro.columns:
        return default_value
    rows = _filter_scenario_rows(macro, scenario_name)
    if rows.empty:
        return default_value
    target_date = pd.to_datetime(current_date).normalize()
    starts = pd.to_datetime(rows.get('period_start'), errors='coerce').dt.normalize()
    ends = pd.to_datetime(rows.get('period_end'), errors='coerce').dt.normalize()
    covering = rows.loc[(starts <= target_date) & (ends >= target_date)]
    if covering.empty:
        prior = rows.loc[starts <= target_date]
        if prior.empty:
            return default_value
        covering = prior.assign(_period_start=starts.loc[prior.index]).sort_values('_period_start').tail(1)
    value = pd.to_numeric(covering.iloc[0]['cbo_cpi_u_index'], errors='coerce')
    return float(value) if not pd.isna(value) else default_value


def _build_cbo_macro_cpi_lookup(cbo_engine_inputs, scenario_name, dates, default_value):
    if not isinstance(cbo_engine_inputs, dict):
        return {}
    macro = cbo_engine_inputs.get('macro_forecast')
    return build_projected_cpi_lookup_from_macro(
        macro,
        dates,
        scenario_id=scenario_name,
        default_value=default_value,
        lag_months=0,
    )


def _get_rate_sensitive_multipliers_config(rate_sensitive_params):
    if not isinstance(rate_sensitive_params, dict):
        return {}
    return rate_sensitive_params


def _load_cbo_engine_inputs(params, scenario_name):
    funding_cfg = params.get('funding_rule', {})
    if not _is_cbo_funding_mode(params):
        return None
    data_vintage = params.get('data_vintage', {})
    if not isinstance(data_vintage, dict):
        data_vintage = {}
    actuals_available_as_of = data_vintage.get('actuals_available_as_of')
    allow_lookahead = bool(data_vintage.get('allow_lookahead', False))
    paths = resolve_baseline_input_paths(params)
    required_paths = {
        'debt_stock_path_file': paths.debt_stock_path_file,
        'primary_deficit_path_file': paths.primary_deficit_path_file,
        'operating_cash_path_file': paths.operating_cash_path_file,
        'cash_reconciliation_residual_file': paths.cash_reconciliation_residual_file,
        'fiscal_incidence_policy_file': paths.fiscal_incidence_policy_file,
    }
    missing = [key for key, path in required_paths.items() if path is None]
    if missing:
        raise ValueError(f"CBO funding mode missing baseline_input_paths: {missing}")
    debt_stock = load_debt_stock_path(
        paths.debt_stock_path_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    primary_deficit = load_primary_deficit_path(
        paths.primary_deficit_path_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    operating_cash = load_operating_cash_path(
        paths.operating_cash_path_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    cash_residual = load_cash_reconciliation_residual(
        paths.cash_reconciliation_residual_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    fiscal_incidence = load_fiscal_incidence_policy(
        paths.fiscal_incidence_policy_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    net_interest_bridge = pd.DataFrame()
    if paths.net_interest_bridge_file is not None:
        net_interest_bridge = load_net_interest_bridge(
            paths.net_interest_bridge_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    cbo_fiscal_baseline = pd.DataFrame()
    if paths.cbo_fiscal_baseline_file is not None:
        cbo_fiscal_baseline = load_cbo_fiscal_baseline(
            paths.cbo_fiscal_baseline_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    fed_holdings = pd.DataFrame()
    if paths.fed_holdings_path_file is not None:
        fed_holdings = load_fed_holdings_path(
            paths.fed_holdings_path_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    macro_forecast = pd.DataFrame()
    if paths.macro_forecast_path_file is not None:
        macro_forecast = load_macro_forecast_path(
            paths.macro_forecast_path_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    yield_curve_surface = pd.DataFrame()
    if paths.yield_curve_surface_file is not None:
        yield_curve_surface = load_yield_curve_surface(paths.yield_curve_surface_file)
    frn_rate_path = pd.DataFrame()
    if paths.frn_rate_path_file is not None:
        frn_rate_path = load_frn_rate_path(
            paths.frn_rate_path_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    frn_rate_path_lookup = _cbo_period_lookup(frn_rate_path)
    frn_rate_path_date_lookup = _cbo_period_end_lookup(frn_rate_path)
    tips_cpi_path = pd.DataFrame()
    if paths.tips_cpi_path_file is not None:
        tips_cpi_path = load_tips_cpi_path(
            paths.tips_cpi_path_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    tips_real_yield_path = pd.DataFrame()
    if paths.tips_real_yield_path_file is not None:
        tips_real_yield_path = load_tips_real_yield_path(
            paths.tips_real_yield_path_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
    return {
        'funding_rule': funding_cfg,
        'scenario_id': funding_cfg.get('scenario_id') or scenario_name,
        'debt_stock': debt_stock,
        'primary_deficit': primary_deficit,
        'operating_cash': operating_cash,
        'cash_residual': cash_residual,
        'fiscal_incidence': fiscal_incidence,
        'net_interest_bridge': net_interest_bridge,
        'cbo_fiscal_baseline': cbo_fiscal_baseline,
        'fed_holdings': fed_holdings,
        'macro_forecast': macro_forecast,
        'yield_curve_surface': yield_curve_surface,
        'frn_rate_path': frn_rate_path,
        'frn_rate_path_lookup': frn_rate_path_lookup,
        'frn_rate_path_date_lookup': frn_rate_path_date_lookup,
        'tips_cpi_path': tips_cpi_path,
        'tips_real_yield_path': tips_real_yield_path,
        'yield_curve_surface_file': str(paths.yield_curve_surface_file) if paths.yield_curve_surface_file else '',
        'frn_rate_path_file': str(paths.frn_rate_path_file) if paths.frn_rate_path_file else '',
        'tips_cpi_path_file': str(paths.tips_cpi_path_file) if paths.tips_cpi_path_file else '',
        'tips_real_yield_path_file': str(paths.tips_real_yield_path_file) if paths.tips_real_yield_path_file else '',
    }


def _cbo_rows_for_scenario(frame, scenario_id):
    if frame is None or frame.empty or 'scenario_id' not in frame.columns:
        return frame
    scenario_values = frame['scenario_id'].fillna('').astype(str)
    for candidate in [str(scenario_id), 'all', 'default', '']:
        scenario_rows = frame[scenario_values == candidate]
        if not scenario_rows.empty:
            return scenario_rows
    return frame.iloc[0:0].copy()


def _cbo_exact_date_row(frame, scenario_id, date_value, date_column, dataset_name):
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        raise ValueError(f"{dataset_name} has no rows for scenario {scenario_id!r}")
    date_key = pd.to_datetime(date_value).normalize()
    dates = pd.to_datetime(rows[date_column], errors='coerce').dt.normalize()
    matches = rows[dates == date_key]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset_name} must have exactly one {date_column} row for {date_key.date()} "
            f"and scenario {scenario_id!r}; found {len(matches)}"
        )
    return matches.iloc[0]


def _cbo_optional_exact_date_row(frame, scenario_id, date_value, date_column, dataset_name):
    if frame is None or frame.empty:
        return None
    return _cbo_exact_date_row(frame, scenario_id, date_value, date_column, dataset_name)


def _apply_cb_auction_share_target(preferences, cb_share):
    adjusted = copy.deepcopy(preferences)
    target_share = min(1.0, max(0.0, float(cb_share)))
    for category in PREFERENCE_CATEGORIES:
        key = f'{category}_pct'
        total_non_cb = 0.0
        for holder in HOLDER_TYPES:
            if holder in {'CB', 'FedInternal', 'TrustFunds'}:
                continue
            total_non_cb += max(0.0, float(adjusted.get(holder, {}).get(key, 0.0) or 0.0))
        for holder in HOLDER_TYPES:
            holder_prefs = adjusted.setdefault(holder, {})
            if holder == 'CB':
                holder_prefs[key] = target_share
            elif holder in {'FedInternal', 'TrustFunds'}:
                holder_prefs[key] = 0.0
            else:
                original = max(0.0, float(holder_prefs.get(key, 0.0) or 0.0))
                holder_prefs[key] = (1.0 - target_share) * original / total_non_cb if total_non_cb > 0.0 else 0.0
    return adjusted


def _cbo_period_row(frame, scenario_id, period_start, period_end, dataset_name):
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        raise ValueError(f"{dataset_name} has no rows for scenario {scenario_id!r}")
    start_key = pd.to_datetime(period_start).normalize()
    end_key = pd.to_datetime(period_end).normalize()
    starts = pd.to_datetime(rows['period_start'], errors='coerce').dt.normalize()
    ends = pd.to_datetime(rows['period_end'], errors='coerce').dt.normalize()
    matches = rows[(starts == start_key) & (ends == end_key)]
    if len(matches) != 1:
        raise ValueError(
            f"{dataset_name} must have exactly one period row for {start_key.date()} to "
            f"{end_key.date()} and scenario {scenario_id!r}; found {len(matches)}"
        )
    return matches.iloc[0]


def _cbo_period_lookup(frame):
    lookup = {}
    if frame is None or frame.empty:
        return lookup
    scenarios = frame['scenario_id'].fillna('').astype(str) if 'scenario_id' in frame.columns else pd.Series('', index=frame.index)
    starts = pd.to_datetime(frame['period_start'], errors='coerce').dt.normalize()
    ends = pd.to_datetime(frame['period_end'], errors='coerce').dt.normalize()
    for idx, row in frame.iterrows():
        if pd.isna(starts.loc[idx]) or pd.isna(ends.loc[idx]):
            continue
        lookup[(scenarios.loc[idx], starts.loc[idx], ends.loc[idx])] = row
    return lookup


def _cbo_period_end_lookup(frame):
    lookup = {}
    if frame is None or frame.empty:
        return lookup
    scenarios = frame['scenario_id'].fillna('').astype(str) if 'scenario_id' in frame.columns else pd.Series('', index=frame.index)
    ends = pd.to_datetime(frame['period_end'], errors='coerce').dt.normalize()
    for idx, row in frame.iterrows():
        if pd.isna(ends.loc[idx]):
            continue
        lookup[(scenarios.loc[idx], ends.loc[idx])] = row
    return lookup


def _lookup_cbo_period_row(lookup, frame, scenario_id, period_start, period_end, dataset_name):
    start_key = pd.to_datetime(period_start).normalize()
    end_key = pd.to_datetime(period_end).normalize()
    for candidate in [str(scenario_id), 'all', 'default', '']:
        row = lookup.get((candidate, start_key, end_key)) if isinstance(lookup, dict) else None
        if row is not None:
            return row
    return _cbo_period_row(frame, scenario_id, period_start, period_end, dataset_name)


def _is_business_day(date_value):
    global _US_FEDERAL_HOLIDAY_DATES
    day = pd.Timestamp(date_value).normalize()
    cache_key = day.date()
    if cache_key in _BUSINESS_DAY_CACHE:
        return _BUSINESS_DAY_CACHE[cache_key]
    if day.weekday() >= 5:
        _BUSINESS_DAY_CACHE[cache_key] = False
        return False
    if _US_FEDERAL_HOLIDAY_DATES is None:
        _US_FEDERAL_HOLIDAY_DATES = {
            pd.Timestamp(value).date()
            for value in _US_FEDERAL_HOLIDAYS.holidays(start="1900-01-01", end="2100-12-31")
        }
    is_business_day = cache_key not in _US_FEDERAL_HOLIDAY_DATES
    _BUSINESS_DAY_CACHE[cache_key] = is_business_day
    return is_business_day


def _subtract_business_days(date_value, business_days):
    day = pd.Timestamp(date_value).normalize()
    remaining = max(0, int(round(float(business_days or 0))))
    while remaining > 0:
        day -= pd.Timedelta(days=1)
        if _is_business_day(day):
            remaining -= 1
    return day


def _add_business_days(date_value, business_days):
    day = pd.Timestamp(date_value).normalize()
    remaining = max(0, int(round(float(business_days or 0))))
    while remaining > 0:
        day += pd.Timedelta(days=1)
        if _is_business_day(day):
            remaining -= 1
    return day


def _lookup_cbo_period_row_ending_on_or_before(frame, scenario_id, period_end, dataset_name):
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        raise ValueError(f"{dataset_name} has no rows for scenario {scenario_id!r}")
    target_end = pd.Timestamp(period_end).normalize()
    ends = pd.to_datetime(rows['period_end'], errors='coerce').dt.normalize()
    candidates = rows.loc[ends <= target_end].copy()
    if candidates.empty:
        raise ValueError(f"{dataset_name} has no row ending on or before {target_end.date()}")
    candidates['_period_end_key'] = pd.to_datetime(candidates['period_end'], errors='coerce').dt.normalize()
    return candidates.sort_values('_period_end_key').iloc[-1].drop(labels=['_period_end_key'], errors='ignore')


def _tips_real_curve_for_date(frame, date_value, scenario_id):
    if frame is None or frame.empty:
        return ([], [], "tips_real_yield_path_not_configured")
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        raise ValueError(f"TIPS real-yield path has no rows for scenario {scenario_id!r}")
    current_date = pd.to_datetime(date_value).normalize()
    curve_dates = pd.to_datetime(rows["curve_date"], errors="coerce").dt.normalize()
    candidates = rows.loc[curve_dates <= current_date].copy()
    if candidates.empty:
        return ([], [], "tips_real_yield_path_no_prior_date")
    candidates["_curve_date"] = pd.to_datetime(candidates["curve_date"], errors="coerce").dt.normalize()
    curve_date = candidates["_curve_date"].max()
    curve = candidates.loc[candidates["_curve_date"] == curve_date].sort_values("tenor_years")
    return (
        [float(value) for value in pd.to_numeric(curve["tenor_years"], errors="raise").tolist()],
        [float(value) for value in pd.to_numeric(curve["real_yield_decimal"], errors="raise").tolist()],
        f"tips_real_yield_path:{curve_date.date()}",
    )


def _cbo_incidence_row(frame, scenario_id):
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        return None
    central = rows[rows['policy_id'].astype(str).str.contains('central_99du_1ru', regex=False)]
    if not central.empty:
        return central.iloc[0]
    return rows.iloc[0]


def _controlled_public_marketable_debt_from_portfolio(bond_portfolio):
    if bond_portfolio is None or bond_portfolio.empty:
        return 0.0
    if isinstance(bond_portfolio, pd.DataFrame):
        active = bond_portfolio['Status'].astype(str).eq('Active')
        public_holder = ~bond_portfolio['HolderType'].astype(str).isin(INTRAGOV_HOLDERS)
        marketable = bond_portfolio['SecurityType'].astype(str).isin(['Fixed', 'TIPS', 'FRN'])
        rows = bond_portfolio.loc[active & public_holder & marketable]
        if rows.empty:
            return 0.0
        face = pd.to_numeric(rows['FaceValue'], errors='coerce').fillna(0.0)
        adjusted = pd.to_numeric(rows['AdjustedPrincipal'], errors='coerce').fillna(face)
        debt_base = face.mask(rows['SecurityType'].astype(str).eq('TIPS'), adjusted)
        return float(debt_base.sum())
    return calculate_pre_issuance_controlled_debt(bond_portfolio.to_dict('records'))


def _bond_debt_base(row):
    face = float(row.get('FaceValue', 0.0) or 0.0)
    if str(row.get('SecurityType', '')) == 'TIPS':
        adjusted = row.get('AdjustedPrincipal', face)
        if pd.isna(adjusted):
            return face
        return float(adjusted)
    return face


def _retire_public_marketable_debt_to_target(bond_portfolio, amount_to_retire, current_date):
    """Retire active public marketable debt at par, shortest maturities first."""
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio, 0.0, {h: 0.0 for h in HOLDER_TYPES}, _zero_private_routes()
    remaining = max(0.0, float(amount_to_retire))
    paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
    paid_by_private_route = _zero_private_routes()
    active = bond_portfolio[
        (bond_portfolio['Status'] == 'Active')
        & (bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS', 'FRN']))
        & (~bond_portfolio['HolderType'].isin(INTRAGOV_HOLDERS))
    ].copy()
    if active.empty:
        return bond_portfolio, 0.0, paid_by_holder, paid_by_private_route
    active['DebtBaseForRetirement'] = np.where(
        active['SecurityType'] == 'TIPS',
        active['AdjustedPrincipal'].fillna(active['FaceValue']),
        active['FaceValue'],
    )
    active['SortMaturity'] = pd.to_datetime(active['MaturityDate'], errors='coerce')
    active = active.sort_values(['SortMaturity', 'SecurityType', 'BondID'])
    retired_total = 0.0
    for idx, row in active.iterrows():
        if remaining <= TGA_FLOOR_TOLERANCE:
            break
        debt_base = float(row['DebtBaseForRetirement'])
        if debt_base <= TGA_FLOOR_TOLERANCE:
            continue
        retire_amount = min(remaining, debt_base)
        holder = row['HolderType']
        paid_by_holder[holder] = paid_by_holder.get(holder, 0.0) + retire_amount
        if holder == 'Private':
            route = _private_subbucket(row)
            paid_by_private_route[route] += retire_amount
        if retire_amount >= debt_base - TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'Status'] = 'Matured'
            bond_portfolio.loc[idx, 'MaturityDate'] = current_date
        else:
            scale = (debt_base - retire_amount) / debt_base
            for col in ['FaceValue', 'IssueProceeds']:
                if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                    bond_portfolio.loc[idx, col] = float(bond_portfolio.loc[idx, col]) * scale
            if row['SecurityType'] == 'TIPS':
                for col in ['OriginalPrincipal', 'AdjustedPrincipal']:
                    if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                        bond_portfolio.loc[idx, col] = float(bond_portfolio.loc[idx, col]) * scale
            if row['SecurityType'] == 'FRN' and pd.notna(bond_portfolio.loc[idx, 'AccruedInterest_FRN']):
                bond_portfolio.loc[idx, 'AccruedInterest_FRN'] = float(bond_portfolio.loc[idx, 'AccruedInterest_FRN']) * scale
        retired_total += retire_amount
        remaining -= retire_amount
    if remaining > TGA_FLOOR_TOLERANCE:
        raise ValueError(
            f"CBO buyback policy could not retire enough debt at {pd.Timestamp(current_date).date()}: "
            f"remaining={remaining:.12f}"
        )
    return bond_portfolio, retired_total, paid_by_holder, paid_by_private_route


def _transfer_public_marketable_debt_to_cb(bond_portfolio, amount_to_purchase):
    """Move active public marketable debt from non-CB holders to CB as a secondary purchase."""
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio, 0.0, {h: 0.0 for h in HOLDER_TYPES}, _zero_private_routes()
    remaining = max(0.0, float(amount_to_purchase))
    sold_by_holder = {h: 0.0 for h in HOLDER_TYPES}
    sold_by_private_route = _zero_private_routes()
    holder_order = ['Private', 'Banks', 'Foreign']
    next_bond_id = None
    purchased_total = 0.0
    new_rows = []
    scalable_columns = [
        'FaceValue',
        'OriginalPrincipal',
        'AdjustedPrincipal',
        'AccruedInterest_FRN',
        'IssueProceeds',
        'AccruedInterest',
        'DirtyValue',
    ]
    while remaining > TGA_FLOOR_TOLERANCE:
        idx = None
        for holder in holder_order:
            holder_mask = (
                (bond_portfolio['Status'] == 'Active')
                & (bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS', 'FRN']))
                & (bond_portfolio['HolderType'] == holder)
            )
            if not holder_mask.any():
                continue
            holder_rows = bond_portfolio.loc[holder_mask, ['SecurityType', 'FaceValue', 'AdjustedPrincipal', 'MaturityDate']]
            holder_debt_base = np.where(
                holder_rows['SecurityType'] == 'TIPS',
                holder_rows['AdjustedPrincipal'].fillna(holder_rows['FaceValue']),
                holder_rows['FaceValue'],
            )
            eligible_indices = holder_rows.index[pd.Series(holder_debt_base, index=holder_rows.index) > TGA_FLOOR_TOLERANCE]
            if len(eligible_indices) == 0:
                continue
            idx = bond_portfolio.loc[eligible_indices, 'MaturityDate'].idxmin()
            break
        if idx is None:
            break
        row = bond_portfolio.loc[idx]
        debt_base = _bond_debt_base(row)
        if debt_base <= TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'FaceValue'] = 0.0
            continue
        purchase_amount = min(remaining, debt_base)
        fraction = purchase_amount / debt_base
        holder = str(row['HolderType'])
        sold_by_holder[holder] = sold_by_holder.get(holder, 0.0) + purchase_amount
        if holder == 'Private':
            route = _private_subbucket(row)
            sold_by_private_route[route] += purchase_amount
        if purchase_amount >= debt_base - TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'HolderType'] = 'CB'
            bond_portfolio.loc[idx, 'HolderSubBucket'] = ''
        else:
            cb_row = bond_portfolio.loc[idx].copy()
            if next_bond_id is None:
                next_bond_id = int(pd.to_numeric(bond_portfolio['BondID'], errors='coerce').max()) + 1
            cb_row['BondID'] = next_bond_id
            next_bond_id += 1
            cb_row['HolderType'] = 'CB'
            cb_row['HolderSubBucket'] = ''
            for col in scalable_columns:
                if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                    original_value = float(bond_portfolio.loc[idx, col])
                    cb_row[col] = original_value * fraction
                    bond_portfolio.loc[idx, col] = original_value * (1.0 - fraction)
            new_rows.append(cb_row)
        purchased_total += purchase_amount
        remaining -= purchase_amount
    if new_rows:
        bond_portfolio = pd.concat([bond_portfolio, pd.DataFrame(new_rows)], ignore_index=True)
    return bond_portfolio, purchased_total, sold_by_holder, sold_by_private_route


def _transfer_cb_marketable_debt_to_public(bond_portfolio, amount_to_sell):
    """Move active marketable CB holdings to Private as a synthetic stock-only sale."""
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio, 0.0
    remaining = max(0.0, float(amount_to_sell))
    sold_total = 0.0
    next_bond_id = None
    new_rows = []
    scalable_columns = [
        'FaceValue',
        'OriginalPrincipal',
        'AdjustedPrincipal',
        'AccruedInterest_FRN',
        'IssueProceeds',
        'AccruedInterest',
        'DirtyValue',
    ]
    while remaining > TGA_FLOOR_TOLERANCE:
        cb_mask = (
            (bond_portfolio['Status'] == 'Active')
            & (bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS', 'FRN']))
            & (bond_portfolio['HolderType'] == 'CB')
        )
        if not cb_mask.any():
            break
        cb_rows = bond_portfolio.loc[cb_mask, ['SecurityType', 'FaceValue', 'AdjustedPrincipal', 'MaturityDate']]
        cb_debt_base = np.where(
            cb_rows['SecurityType'] == 'TIPS',
            cb_rows['AdjustedPrincipal'].fillna(cb_rows['FaceValue']),
            cb_rows['FaceValue'],
        )
        eligible_indices = cb_rows.index[pd.Series(cb_debt_base, index=cb_rows.index) > TGA_FLOOR_TOLERANCE]
        if len(eligible_indices) == 0:
            break
        idx = bond_portfolio.loc[eligible_indices, 'MaturityDate'].idxmin()
        row = bond_portfolio.loc[idx]
        debt_base = _bond_debt_base(row)
        if debt_base <= TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'FaceValue'] = 0.0
            continue
        sale_amount = min(remaining, debt_base)
        fraction = sale_amount / debt_base
        if sale_amount >= debt_base - TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'HolderType'] = 'Private'
            bond_portfolio.loc[idx, 'HolderSubBucket'] = PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
        else:
            public_row = bond_portfolio.loc[idx].copy()
            if next_bond_id is None:
                next_bond_id = int(pd.to_numeric(bond_portfolio['BondID'], errors='coerce').max()) + 1
            public_row['BondID'] = next_bond_id
            next_bond_id += 1
            public_row['HolderType'] = 'Private'
            public_row['HolderSubBucket'] = PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
            for col in scalable_columns:
                if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                    original_value = float(bond_portfolio.loc[idx, col])
                    public_row[col] = original_value * fraction
                    bond_portfolio.loc[idx, col] = original_value * (1.0 - fraction)
            new_rows.append(public_row)
        sold_total += sale_amount
        remaining -= sale_amount
    if new_rows:
        bond_portfolio = pd.concat([bond_portfolio, pd.DataFrame(new_rows)], ignore_index=True)
    return bond_portfolio, sold_total


def _quote_issuance(security_type, maturity_years, coupon_rate, yield_at_issuance, amount, face_target_mode):
    if face_target_mode:
        return quote_issuance_from_face_target(
            security_type,
            maturity_years,
            coupon_rate,
            yield_at_issuance,
            amount,
        )
    return calculate_face_from_proceeds_target(
        security_type,
        maturity_years,
        coupon_rate,
        yield_at_issuance,
        amount,
    )


def _get_weighted_average_maturity(category, issuance_profile, frn_benchmark_mat):
    defaults = {
        'bills': ([0.25, 0.5, 1.0], [0.40, 0.40, 0.20]),
        'notes': ([2.0, 5.0, 10.0], [0.30, 0.40, 0.30]),
        'bonds': ([20.0, 30.0], [0.50, 0.50]),
        'tips': ([5.0, 10.0, 30.0], [0.50, 0.30, 0.20]),
        'frn': ([2.0], [1.0]),
    }
    if category == 'frn':
        return max(0.0, float(frn_benchmark_mat))
    source_key = category if category in MATURITY_CATEGORIES else category.upper()
    cfg = issuance_profile.get(source_key, {}) if isinstance(issuance_profile, dict) else {}
    maturities = cfg.get('maturities', defaults[category][0])
    weights = cfg.get('maturity_distribution', defaults[category][1])
    if not maturities or len(maturities) != len(weights):
        maturities, weights = defaults[category]
    cleaned_weights = np.array([max(0.0, float(w)) for w in weights], dtype=float)
    if cleaned_weights.sum() <= TGA_FLOOR_TOLERANCE:
        cleaned_weights = np.array(defaults[category][1], dtype=float)
    cleaned_weights = cleaned_weights / cleaned_weights.sum()
    cleaned_maturities = np.array([max(0.0, float(m)) for m in maturities], dtype=float)
    return float(np.dot(cleaned_maturities, cleaned_weights))


def _get_category_yield(
    category,
    issuance_profile,
    yield_curve_years,
    yield_curve_rates,
    frn_benchmark_mat,
    *,
    interpolation_method='linear',
    floor_zero=True,
):
    maturity = _get_weighted_average_maturity(category, issuance_profile, frn_benchmark_mat)
    yld = get_yield_for_maturity(
        maturity,
        yield_curve_years,
        yield_curve_rates,
        method=interpolation_method,
        floor_zero=floor_zero,
    )
    return 0.0 if pd.isna(yld) else float(yld)


def _compute_rate_sensitive_multiplier(rate_sensitive_params, section, holder, category, item_yield, anchor_yield, curve_slope):
    if not isinstance(rate_sensitive_params, dict) or not rate_sensitive_params.get('enabled', False):
        return 1.0
    section_cfg = rate_sensitive_params.get(section, {})
    holder_cfg = section_cfg.get(holder, {}) if isinstance(section_cfg, dict) else {}
    category_cfg = holder_cfg.get(category, {}) if isinstance(holder_cfg, dict) else {}
    if not isinstance(category_cfg, dict) or not category_cfg:
        return 1.0
    intercept = float(category_cfg.get('intercept', 0.0) or 0.0)
    yield_beta = float(category_cfg.get('yield_beta', 0.0) or 0.0)
    spread_beta = float(category_cfg.get('spread_beta', 0.0) or 0.0)
    slope_beta = float(category_cfg.get('slope_beta', 0.0) or 0.0)
    score = intercept + (yield_beta * item_yield) + (spread_beta * (item_yield - anchor_yield)) + (slope_beta * curve_slope)
    min_multiplier = max(0.0, float(rate_sensitive_params.get('min_multiplier', 0.0) or 0.0))
    return max(min_multiplier, float(np.exp(np.clip(score, -20.0, 20.0))))


def _get_curve_reference_levels(
    rate_sensitive_params,
    yield_curve_years,
    yield_curve_rates,
    *,
    interpolation_method='linear',
    floor_zero=True,
):
    params = _get_rate_sensitive_multipliers_config(rate_sensitive_params)
    anchor_mat = max(0.0, float(params.get('anchor_maturity_years', 5.0) or 5.0))
    slope_short = max(0.0, float(params.get('slope_short_maturity_years', 2.0) or 2.0))
    slope_long = max(0.0, float(params.get('slope_long_maturity_years', 10.0) or 10.0))
    anchor_yield = get_yield_for_maturity(anchor_mat, yield_curve_years, yield_curve_rates, method=interpolation_method, floor_zero=floor_zero)
    short_yield = get_yield_for_maturity(slope_short, yield_curve_years, yield_curve_rates, method=interpolation_method, floor_zero=floor_zero)
    long_yield = get_yield_for_maturity(slope_long, yield_curve_years, yield_curve_rates, method=interpolation_method, floor_zero=floor_zero)
    anchor_yield = 0.0 if pd.isna(anchor_yield) else float(anchor_yield)
    short_yield = 0.0 if pd.isna(short_yield) else float(short_yield)
    long_yield = 0.0 if pd.isna(long_yield) else float(long_yield)
    return anchor_yield, long_yield - short_yield


def _build_dynamic_secondary_preferences(
    base_secondary_prefs,
    rate_sensitive_params,
    issuance_profile,
    yield_curve_years,
    yield_curve_rates,
    frn_benchmark_mat,
    *,
    interpolation_method='linear',
    floor_zero=True,
):
    if not isinstance(rate_sensitive_params, dict) or not rate_sensitive_params.get('enabled', False):
        return copy.deepcopy(base_secondary_prefs)
    if not isinstance(rate_sensitive_params.get('secondary', {}), dict) or not rate_sensitive_params.get('secondary'):
        return copy.deepcopy(base_secondary_prefs)

    active_categories = ['bills', 'notes', 'bonds']
    if float(issuance_profile.get('TIPS', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('tips')
    if float(issuance_profile.get('FRN', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('frn')

    anchor_yield, curve_slope = _get_curve_reference_levels(
        rate_sensitive_params,
        yield_curve_years,
        yield_curve_rates,
        interpolation_method=interpolation_method,
        floor_zero=floor_zero,
    )
    dynamic_prefs = copy.deepcopy(base_secondary_prefs)
    for holder in HOLDER_TYPES:
        holder_cfg = copy.deepcopy(dynamic_prefs.get(holder, {}))
        adjusted = {}
        for category in active_categories:
            base_share = float(holder_cfg.get(f'{category}_pct', 0.0) or 0.0)
            category_yield = _get_category_yield(
                category,
                issuance_profile,
                yield_curve_years,
                yield_curve_rates,
                frn_benchmark_mat,
                interpolation_method=interpolation_method,
                floor_zero=floor_zero,
            )
            multiplier = _compute_rate_sensitive_multiplier(
                rate_sensitive_params, 'secondary', holder, category, category_yield, anchor_yield, curve_slope
            )
            adjusted[category] = max(0.0, base_share * multiplier)
        total_adjusted = sum(adjusted.values())
        if total_adjusted > TGA_FLOOR_TOLERANCE:
            for category, value in adjusted.items():
                holder_cfg[f'{category}_pct'] = value / total_adjusted
        dynamic_prefs[holder] = holder_cfg
    return dynamic_prefs


def _get_active_preference_categories(issuance_profile):
    active_categories = ['bills', 'notes', 'bonds']
    if float(issuance_profile.get('TIPS', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('tips')
    if float(issuance_profile.get('FRN', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('frn')
    return active_categories


def _compute_preference_shift_summary(base_prefs, effective_prefs, categories):
    diffs = []
    for holder in HOLDER_TYPES:
        base_holder = base_prefs.get(holder, {}) if isinstance(base_prefs, dict) else {}
        effective_holder = effective_prefs.get(holder, {}) if isinstance(effective_prefs, dict) else {}
        for category in categories:
            base_value = float(base_holder.get(f'{category}_pct', 0.0) or 0.0)
            effective_value = float(effective_holder.get(f'{category}_pct', 0.0) or 0.0)
            diffs.append(abs(effective_value - base_value))
    if not diffs:
        return (0.0, 0.0)
    return (float(np.mean(diffs)), float(np.max(diffs)))


def _private_subbucket(row_or_value) -> str:
    value = row_or_value.get('HolderSubBucket', '') if hasattr(row_or_value, 'get') else row_or_value
    value = '' if pd.isna(value) else str(value)
    return value if value in PRIVATE_SUBBUCKETS else PRIVATE_SUBBUCKET_DOMESTIC_NONBANK


def _private_subbucket_split(preferences: dict, category: str) -> dict[str, float]:
    subbucket_map = preferences.get('__private_subbucket_shares__', {}) if isinstance(preferences, dict) else {}
    category_map = subbucket_map.get(category, {}) if isinstance(subbucket_map, dict) else {}
    raw = {
        PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: max(
            0.0, float(category_map.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, 0.0) or 0.0)
        ),
        PRIVATE_SUBBUCKET_MMF: max(0.0, float(category_map.get(PRIVATE_SUBBUCKET_MMF, 0.0) or 0.0)),
    }
    total = sum(raw.values())
    if total <= TGA_FLOOR_TOLERANCE:
        return {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0, PRIVATE_SUBBUCKET_MMF: 0.0}
    return {key: value / total for key, value in raw.items()}


def _effective_private_amount(
    route_totals: dict[str, float],
    mmf_pass_through: float,
    legacy_total: float | None = None,
) -> float:
    if mmf_pass_through >= 1.0 and legacy_total is not None:
        return float(legacy_total)
    return (
        route_totals.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, 0.0)
        + mmf_pass_through * route_totals.get(PRIVATE_SUBBUCKET_MMF, 0.0)
    )


def _zero_private_routes() -> dict[str, float]:
    return {subbucket: 0.0 for subbucket in PRIVATE_SUBBUCKETS}


def _private_route_value(route_totals: dict[str, float], route: str) -> float:
    return float(route_totals.get(route, 0.0) or 0.0)


def _daily_coupon_due_indices(tranches, current_date, frequency):
    if tranches.empty:
        return []
    current = pd.Timestamp(current_date).normalize()
    issue_dates = pd.to_datetime(tranches['IssueDate'], errors='coerce').dt.normalize()
    maturity_dates = pd.to_datetime(tranches['MaturityDate'], errors='coerce').dt.normalize()
    first_interest = (
        pd.to_datetime(tranches['FirstInterestPaymentDate'], errors='coerce').dt.normalize()
        if 'FirstInterestPaymentDate' in tranches
        else pd.Series(pd.NaT, index=tranches.index, dtype='datetime64[ns]')
    )
    payment_frequency = (
        pd.to_numeric(tranches['InterestPaymentFrequency'], errors='coerce')
        if 'InterestPaymentFrequency' in tranches
        else pd.Series(np.nan, index=tranches.index)
    ).fillna(float(frequency)).round().astype(int)
    period_months = (12 / payment_frequency.replace(0, frequency)).round().astype(int)
    anchor_dates = first_interest.where(first_interest.notna(), maturity_dates)
    months_between = (current.year - anchor_dates.dt.year) * 12 + (current.month - anchor_dates.dt.month)
    expected_day = np.minimum(anchor_dates.dt.day.fillna(0).astype(int), current.days_in_month)
    due = (
        issue_dates.notna()
        & maturity_dates.notna()
        & anchor_dates.notna()
        & (current > issue_dates)
        & (current < maturity_dates)
        & (first_interest.isna() | (current >= first_interest))
        & ((months_between % period_months) == 0)
        & (current.day == expected_day)
    )
    return tranches.loc[due].index.tolist()


def _coupon_due_indices_for_period(tranches, prev_date, current_date, frequency):
    if tranches.empty:
        return []
    if (pd.Timestamp(current_date) - pd.Timestamp(prev_date)).days <= 1:
        return _daily_coupon_due_indices(tranches, current_date, frequency=frequency)
    paying_indices = []
    for idx, bond in tranches.iterrows():
        coupon_dates = get_coupon_dates_in_period(
            bond['IssueDate'],
            bond['MaturityDate'],
            prev_date,
            current_date,
            frequency=frequency,
            first_interest_payment_date=bond.get('FirstInterestPaymentDate'),
            interest_payment_frequency=bond.get('InterestPaymentFrequency'),
        )
        if coupon_dates:
            paying_indices.append(idx)
    return paying_indices


def _frn_coupon_dates_for_period(bond, prev_date, current_date):
    return get_coupon_dates_in_period(
        bond['IssueDate'],
        bond['MaturityDate'],
        prev_date,
        current_date,
        frequency=4,
        first_interest_payment_date=bond.get('FirstInterestPaymentDate'),
        interest_payment_frequency=bond.get('InterestPaymentFrequency'),
    )


def _frn_next_coupon_date(bond, current_date):
    maturity_date = pd.Timestamp(bond.get('MaturityDate')).normalize()
    if pd.isna(maturity_date) or current_date > maturity_date:
        return pd.NaT
    coupon_dates = get_coupon_dates_in_period(
        bond['IssueDate'],
        bond['MaturityDate'],
        current_date - pd.Timedelta(days=1),
        maturity_date,
        frequency=4,
        first_interest_payment_date=bond.get('FirstInterestPaymentDate'),
        interest_payment_frequency=bond.get('InterestPaymentFrequency'),
    )
    return min(coupon_dates) if coupon_dates else pd.NaT


def _frn_next_due_date(bond, current_date):
    candidates = []
    coupon_date = _frn_next_coupon_date(bond, current_date)
    if pd.notna(coupon_date):
        candidates.append(pd.Timestamp(coupon_date).normalize())
    maturity_date = pd.Timestamp(bond.get('MaturityDate')).normalize()
    if pd.notna(maturity_date) and maturity_date >= current_date:
        candidates.append(maturity_date)
    return min(candidates) if candidates else pd.NaT


def _frn_lockout_business_days(frn_path_row):
    if frn_path_row is None or 'lockout_business_days' not in frn_path_row.index:
        return 0
    try:
        lockout = int(round(float(frn_path_row.get('lockout_business_days', 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0
    return max(0, lockout)


def _frn_lockout_start_date(due_date, lockout_business_days):
    if pd.isna(due_date) or lockout_business_days <= 0:
        return pd.NaT
    return _subtract_business_days(due_date, lockout_business_days)


def _frn_rate_date_for_accrual_day(bond, accrual_date, lockout_business_days):
    accrual_day = pd.Timestamp(accrual_date).normalize()
    due_date = _frn_next_due_date(bond, accrual_day)
    if pd.isna(due_date):
        return accrual_day
    lockout_start = _frn_lockout_start_date(due_date, lockout_business_days)
    if pd.notna(lockout_start) and lockout_start <= accrual_day <= due_date:
        return lockout_start
    return accrual_day


def _frn_rate_path_row_for_date(frame, scenario_id, date_value, lookup=None):
    target = pd.Timestamp(date_value).normalize()
    if isinstance(lookup, dict):
        for candidate in [str(scenario_id), 'all', 'default', '']:
            row = lookup.get((candidate, target))
            if row is not None:
                return row
    if frame is None or frame.empty:
        return None
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        return None
    starts = pd.to_datetime(rows['period_start'], errors='coerce').dt.normalize()
    ends = pd.to_datetime(rows['period_end'], errors='coerce').dt.normalize()
    covering = rows.loc[(starts < target) & (target <= ends)].copy()
    if covering.empty:
        return None
    durations = (ends.loc[covering.index] - starts.loc[covering.index]).dt.days
    covering = covering.assign(_duration_days=durations, _period_end=ends.loc[covering.index])
    return covering.sort_values(['_duration_days', '_period_end']).iloc[0]


def _first_interest_payment_date(issue_date, maturity_date, frequency):
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return pd.NaT
    issue = pd.Timestamp(issue_date).normalize()
    maturity = pd.Timestamp(maturity_date).normalize()
    try:
        freq = int(round(float(frequency)))
    except (TypeError, ValueError):
        freq = 2
    if freq <= 0 or 12 % freq != 0:
        freq = 2
    months_between = int(round(12 / freq))
    candidate = maturity
    while candidate > issue:
        prior = candidate - relativedelta(months=months_between)
        if prior <= issue:
            return candidate
        candidate = prior
    return pd.NaT


def run_simulation(params, start_date, end_date, freq='W', scenario_name='Default'):
    """
    Runs the core economic simulation including Treasury operations, fiscal flows,
    central bank actions, secondary market trading, AND handles date-based events
    defined in the configuration to dynamically change parameters.
    Tracks key monetary aggregates and debt composition.
    """
    sim_start_time = time.time()
    sim_mode = str(params.get('simulation_period', {}).get('mode', 'forward')).strip().lower()
    if sim_mode == 'historical_replay':
        from historical_replay import run_historical_replay

        return run_historical_replay(params, start_date, end_date, scenario_name=scenario_name)
    validate_run_params(params, scenario_name=scenario_name)
    funding_rule_cfg = params.get('funding_rule', {})
    cbo_funding_mode = str(funding_rule_cfg.get('mode', '')).strip() == CBO_FUNDING_MODE
    yield_surface_cfg = params.get('yield_curve_surface', {}) if isinstance(params.get('yield_curve_surface', {}), dict) else {}
    yield_interpolation_method = str(
        yield_surface_cfg.get('interpolation_method', 'pchip' if cbo_funding_mode else 'linear')
    ).strip().lower()
    yield_floor_zero = bool(yield_surface_cfg.get('floor_zero', not cbo_funding_mode))
    coarse_frequency_warning = None
    try:
        configured_start_date_pd = pd.to_datetime(start_date)
        start_date_pd = configured_start_date_pd
        end_date_pd = pd.to_datetime(end_date)
        coarse_frequency_action = str(params.get('simulation_period', {}).get('coarse_frequency_action', 'warn')).strip().lower()
        if is_coarse_frequency(freq):
            coarse_frequency_warning = (
                f"[{scenario_name}] Frequency '{freq}' is coarse relative to coupon timing. "
                "Weekly remains the recommended mode."
            )
            if coarse_frequency_action == 'warn':
                print(f'WARNING: {coarse_frequency_warning}')
        if freq.upper().startswith('W') and not cbo_funding_mode:
            start_date_pd = start_date_pd - timedelta(days=start_date_pd.weekday())
        dates = pd.date_range(start_date_pd, end_date_pd, freq=freq, name='Date')
        num_periods = len(dates)
        period_counts_by_quarter = (
            pd.Series([f"{date.year}Q{date.quarter}" for date in dates[1:]])
            .value_counts()
            .astype(int)
            .to_dict()
        )
    except Exception as e:
        print(f'ERROR [{scenario_name}]: Failed to create date range: {e}')
        return (pd.DataFrame(), pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES))
    if num_periods <= 1:
        print(f'WARNING [{scenario_name}]: Simulation period results in <= 1 time step. Skipping.')
        results_cols_on_skip = ['TGA', 'Reserves', 'TDC_Level']
        empty_results = pd.DataFrame(index=dates, columns=results_cols_on_skip, dtype=float).fillna(0.0)
        return (empty_results, pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES))
    print(f'--- Starting Simulation: {scenario_name} ---')
    results_cols = ['GovSpending', 'Taxes', 'PrimaryDeficit', 'InterestPaid_Bonds', 'PrincipalPaid_Bonds', 'InterestOutlay_Period', 'InterestOutlay_Cumulative', 'PrincipalRollover_Period', 'PrincipalRollover_Cumulative', 'NewDebtIssued', 'AuctionProceeds', 'IssuanceProceedsTarget', 'IssueDiscountCost_Period', 'IssueDiscountCost_Cumulative', 'FinancingCost_Period', 'FinancingCost_Cumulative', 'NonMarketableInterestCapitalized_Period', 'NonMarketableInterestCapitalized_Cumulative', 'TIPSInflationAccretion_Period', 'TIPSInflationAccretion_Cumulative', 'AuctionDemandShift_AvgAbs', 'AuctionDemandShift_MaxAbs', 'SecondaryDemandShift_AvgAbs', 'SecondaryDemandShift_MaxAbs', 'DebtServiceOutlay_Period', 'DebtServiceOutlay_Cumulative', 'TotalDebt_Agg', 'DebtHeld_Banks', 'DebtHeld_Private', 'DebtHeld_CB', 'DebtHeld_Foreign', 'DebtHeld_FedInternal', 'DebtHeld_TrustFunds', 'TGA', 'Reserves', 'TDC_Level', 'ReserveChange', 'TDC_Change', 'TGAChange', 'TDC_FiscalFlow', 'TDC_DebtService', 'TDC_AuctionAbsorption', 'TDC_SecondaryTrades', 'TDC_Other', 'TDC_PrincipalToDU', 'TDC_InterestToDU', 'TDC_BillDiscountInterestToDU', 'TDC_CouponInterestToDU', 'TDC_FRNInterestToDU', 'TDC_TIPSCouponInterestToDU', 'TDC_TIPSInflationCompensationToDU', 'TDC_GrossIssuanceProceedsAbsorbedByDU', 'TDC_SecondaryDUToRU', 'TDC_SecondaryRUToDU', 'TDC_AuctionAbsorption_DomesticNonbank', 'TDC_AuctionAbsorption_MMF', 'TDC_AuctionAbsorption_MMFPlumbing', 'TDC_PrincipalToDU_DomesticNonbank', 'TDC_PrincipalToDU_MMF', 'TDC_PrincipalToDU_MMFPlumbing', 'TDC_BillDiscountInterestToDU_DomesticNonbank', 'TDC_BillDiscountInterestToDU_MMF', 'TDC_CouponInterestToDU_DomesticNonbank', 'TDC_CouponInterestToDU_MMF', 'TDC_FRNInterestToDU_DomesticNonbank', 'TDC_FRNInterestToDU_MMF', 'TDC_TIPSCouponInterestToDU_DomesticNonbank', 'TDC_TIPSCouponInterestToDU_MMF', 'TDC_TIPSInflationCompensationToDU_DomesticNonbank', 'TDC_TIPSInflationCompensationToDU_MMF', 'TDC_InterestToDU_DomesticNonbank', 'TDC_InterestToDU_MMF', 'TDC_DebtService_MMFPlumbing', 'TDC_GrossIssuanceProceedsAbsorbedByDU_DomesticNonbank', 'TDC_GrossIssuanceProceedsAbsorbedByDU_MMF', 'TDC_SecondaryTrades_DomesticNonbank', 'TDC_SecondaryTrades_MMF', 'TDC_SecondaryTrades_MMFPlumbing', 'CB_InterestIncome', 'CB_NetIncome', 'CB_Remittance', 'CB_DeferredAsset', 'WAM', 'DebtHeldByType_Fixed', 'DebtHeldByType_TIPS', 'DebtHeldByType_FRN', 'DebtHeldByType_NonMarketable', 'CPI_Level', 'Reference_CPI', 'CBOFundingModeActive', 'CBOPrimaryDeficitFlow', 'CBOControlledDebtTarget', 'CBOControlledDebtPreIssuance', 'CBOControlledDebtPostIssuance', 'CBOControlledDebtTargetError', 'CBORequiredFaceIssuance', 'CBOBuybackFaceRetired', 'CBOBuybackCashPaid', 'CBOOperatingCashTarget', 'CBOCashResidual', 'CBOCashReconciliationResidual', 'CBOFiscalIncidencePolicyPresent', 'CBORemittanceCashEffect', 'CBOFedHoldingsTarget', 'CBOFedHoldingsTargetError', 'CBOFedAuctionShare', 'CBOFedSecondaryPurchaseFace', 'CBOFedSecondaryPurchaseCash', 'CBOFedSecondaryPurchaseReserveEffect', 'CBOFedSecondaryPurchaseDepositEffect', 'CBONetInterestDiagnostic', 'CBOTotalDeficitDiagnostic', 'CBONetInterestBridgeRows']
    results = pd.DataFrame(index=dates, columns=results_cols, dtype=float).fillna(0.0)
    for status_col in ['FundingMode', 'FiscalIncidenceStatus', 'NetInterestDiagnosticStatus', 'CBOCashResidualStatus']:
        results[status_col] = ''
    for status_col in ['CBOFedStockMode', 'CBOFedSettlementScope', 'CBORemittanceStatus']:
        results[status_col] = ''
    for numeric_col in [
        'CBOControlledDebtTargetApplicable',
        'CBOFedHoldingsTargetApplicable',
        'CBOFedBeginStock',
        'CBOFedMaturitiesAndRedemptions',
        'CBOFedTipsPrincipalIndexation',
        'CBOFedAuctionRolloverAddons',
        'CBOFedSyntheticSecondaryPurchases',
        'CBOFedSyntheticSecondarySales',
        'CBOFedEndStock',
        'CBOFedGrossStockFlow',
        'CBOFedNetStockChange',
    ]:
        results[numeric_col] = 0.0
    results['FundingMode'] = CBO_FUNDING_MODE if cbo_funding_mode else CASH_FUNDING_MODE
    raw_events = params.get('events', [])
    scheduled_events = defaultdict(list)
    processed_event_ids = set()
    if isinstance(raw_events, list):
        for idx, event_def in enumerate(raw_events):
            if not isinstance(event_def, dict):
                continue
            event_date_str = event_def.get('date')
            actions = event_def.get('actions')
            event_id = event_def.get('id', f'event_{idx}')
            if event_date_str and isinstance(actions, list):
                try:
                    event_date = pd.to_datetime(event_date_str)
                    if configured_start_date_pd <= event_date <= end_date_pd:
                        valid_actions = []
                        for action in actions:
                            if isinstance(action, dict) and 'parameter_path' in action and ('new_value' in action):
                                path_keys = str(action['parameter_path']).split('.')
                                if path_keys:
                                    valid_actions.append({'path_keys': path_keys, 'new_value': action['new_value']})
                        if valid_actions:
                            scheduled_events[event_date].extend(valid_actions)
                            processed_event_ids.add(event_id)
                except (ValueError, TypeError):
                    pass
    try:
        yield_p = copy.deepcopy(params.get('yield_curve', {}))
        current_yield_curve_years = yield_p.get('years', [])
        current_yield_curve_rates = yield_p.get('rates', [])
        cbo_inputs = _load_cbo_engine_inputs(params, scenario_name)
        cbo_scenario_id = (
            cbo_inputs.get('scenario_id', scenario_name)
            if isinstance(cbo_inputs, dict)
            else scenario_name
        )
        yield_curve_surface = pd.DataFrame()
        yield_curve_surface_status = 'static_curve_no_surface'
        surface_cfg = params.get('yield_curve_surface', {})
        cbo_surface_file = cbo_inputs.get('yield_curve_surface_file', '') if isinstance(cbo_inputs, dict) else ''
        explicit_surface_file = str(surface_cfg.get('file') or '') if isinstance(surface_cfg, dict) else ''
        if cbo_surface_file and explicit_surface_file and str(cbo_surface_file) != str(explicit_surface_file):
            raise ValueError(
                "CBO funding mode declares conflicting yield curve surface files: "
                f"baseline_input_paths={cbo_surface_file!r}, yield_curve_surface={explicit_surface_file!r}"
            )
        if cbo_surface_file:
            yield_curve_surface = cbo_inputs.get('yield_curve_surface', pd.DataFrame())
            surface_scenario_id = cbo_scenario_id
        elif isinstance(surface_cfg, dict) and surface_cfg.get('file'):
            yield_curve_surface = load_yield_curve_surface(surface_cfg.get('file'))
            surface_scenario_id = surface_cfg.get('scenario_id')
        else:
            surface_scenario_id = None
        if isinstance(yield_curve_surface, pd.DataFrame) and not yield_curve_surface.empty:
            surface_years, surface_rates, yield_curve_surface_status = curve_for_date(
                yield_curve_surface,
                start_date,
                scenario_id=surface_scenario_id,
            )
            if surface_years and surface_rates:
                current_yield_curve_years = surface_years
                current_yield_curve_rates = surface_rates
        ratewall_input_cfg = params.get('ratewall_input_paths', {})
        primary_flow_lookup = {}
        holder_absorption_lookup = {}
        primary_flow_warning_cache = set()
        holder_absorption_warning_cache = set()
        primary_flow_path_used_count = 0
        if isinstance(ratewall_input_cfg, dict):
            source_base_dir = ratewall_input_cfg.get('base_dir')
            primary_flow_lookup = load_primary_flow_path(ratewall_input_cfg, base_dir=source_base_dir)
            holder_absorption_lookup = load_holder_absorption_path(ratewall_input_cfg, base_dir=source_base_dir)
        fiscal_p = params.get('fiscal_params', {})
        current_fiscal_params = copy.deepcopy(fiscal_p)
        q_start_spending = current_fiscal_params.get('initial_weekly_spending', 0.0)
        q_start_taxes = current_fiscal_params.get('initial_weekly_taxes', 0.0)
        spending_growth_qtr = current_fiscal_params.get('spending_growth_qtr', 0.0)
        tax_growth_qtr = current_fiscal_params.get('tax_growth_qtr', 0.0)
        du_share_spending = current_fiscal_params.get('du_share_spending', 1.0)
        du_share_taxes = current_fiscal_params.get('du_share_taxes', 1.0)
        q_end_target_spending = q_start_spending * (1 + spending_growth_qtr)
        q_end_target_taxes = q_start_taxes * (1 + tax_growth_qtr)
        tga_p = params.get('tga_params', {})
        current_tga_params = copy.deepcopy(tga_p)
        tga_target = current_tga_params.get('target_balance', 0.0)
        tga_floor = current_tga_params.get('floor', 0.0)
        other_p = params.get('other_flows', {})
        current_other_flows = copy.deepcopy(other_p)
        reserve_transfer = current_other_flows.get('reserve_transfer', 0.0)
        cb_net_expense = current_other_flows.get('cb_net_expense', 0.0)
        money_minting_transfers = current_other_flows.get('money_minting_transfers', 0.0)
        issuance_profile = copy.deepcopy(params.get('treasury_issuance_profile', {}))
        current_sector_prefs = copy.deepcopy(params.get('sector_preferences', {}))
        uses_legacy_sector_prefs_for_auction = 'auction_absorption_preferences' not in params
        uses_legacy_sector_prefs_for_secondary = 'secondary_target_preferences' not in params
        current_auction_prefs = copy.deepcopy(params.get('auction_absorption_preferences', current_sector_prefs))
        current_secondary_prefs = copy.deepcopy(params.get('secondary_target_preferences', current_sector_prefs))
        sim_period_p = params.get('simulation_period', {})
        current_enable_trading = sim_period_p.get('enable_preference_trading', False)
        tips_p = copy.deepcopy(params.get('tips_params', {}))
        cpi_start = tips_p.get('cpi_start_level', 100.0)
        cpi_inflation = tips_p.get('cpi_annual_inflation', 0.0)
        ref_cpi_lag = tips_p.get('ref_cpi_lag_months', 3)
        tips_real_coupon = tips_p.get('default_real_coupon_rate', 0.005)
        if cbo_funding_mode:
            cpi_start = _cbo_macro_cpi_level(cbo_inputs, cbo_scenario_id, start_date_pd, cpi_start)
        frn_p = copy.deepcopy(params.get('frn_params', {}))
        frn_benchmark_mat = frn_p.get('benchmark_maturity_years', 0.25)
        frn_spread = frn_p.get('default_fixed_spread', 0.0005)
        nonmkt_p = params.get('nonmarketable_params', {})
        current_nonmkt_params = copy.deepcopy(nonmkt_p)
        nonmkt_rate_mats = current_nonmkt_params.get('interest_rate_basis_maturities', [5.0, 10.0])
        nonmkt_credit_freq = current_nonmkt_params.get('interest_crediting_frequency', 'semi-annual')
        rate_sensitive_p = copy.deepcopy(params.get('rate_sensitive_demand', {}))
        financing_cost_options = copy.deepcopy(params.get('financing_cost_options', {}))
        include_tips_inflation_accretion = bool(financing_cost_options.get('include_tips_inflation_accretion', False))
        private_mmf_split = copy.deepcopy(params.get('private_mmf_split', {}))
        mmf_deposit_pass_through = float(
            private_mmf_split.get(
                'mmf_deposit_pass_through',
                MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
            )
            if isinstance(private_mmf_split, dict)
            else MMF_DEPOSIT_PASS_THROUGH_DEFAULT
        )
        mmf_deposit_pass_through = min(1.0, max(0.0, mmf_deposit_pass_through))
        dynamic_params_state = {'yield_curve': yield_p, 'fiscal_params': current_fiscal_params, 'tga_params': current_tga_params, 'other_flows': current_other_flows, 'sector_preferences': current_sector_prefs, 'auction_absorption_preferences': current_auction_prefs, 'secondary_target_preferences': current_secondary_prefs, 'treasury_issuance_profile': issuance_profile, 'tips_params': tips_p, 'frn_params': frn_p, 'nonmarketable_params': current_nonmkt_params, 'rate_sensitive_demand': rate_sensitive_p, 'financing_cost_options': financing_cost_options, 'simulation_period': {'enable_preference_trading': current_enable_trading}}
    except Exception as e:
        print(f'ERROR [{scenario_name}]: Failed during parameter extraction/initialization: {e}')
        traceback.print_exc()
        raise
    t0 = dates[0]
    pre_t0_event_actions = []
    for event_date in sorted(scheduled_events.keys()):
        if event_date <= t0:
            pre_t0_event_actions.extend(scheduled_events[event_date])
    if pre_t0_event_actions:
        apply_event_actions(pre_t0_event_actions, dynamic_params_state, scenario_name, t0, propagate_legacy_sector_prefs_to_auction=uses_legacy_sector_prefs_for_auction, propagate_legacy_sector_prefs_to_secondary=uses_legacy_sector_prefs_for_secondary)
        current_enable_trading = dynamic_params_state.get('simulation_period', {}).get('enable_preference_trading', False)
        current_tga_params = dynamic_params_state.get('tga_params', current_tga_params)
        current_other_flows = dynamic_params_state.get('other_flows', current_other_flows)
        current_fiscal_params = dynamic_params_state.get('fiscal_params', current_fiscal_params)
        issuance_profile = dynamic_params_state.get('treasury_issuance_profile', issuance_profile)
        current_sector_prefs = dynamic_params_state.get('sector_preferences', current_sector_prefs)
        current_auction_prefs = dynamic_params_state.get('auction_absorption_preferences', current_auction_prefs)
        current_secondary_prefs = dynamic_params_state.get('secondary_target_preferences', current_secondary_prefs)
        yield_p = dynamic_params_state.get('yield_curve', yield_p)
        tips_p = dynamic_params_state.get('tips_params', tips_p)
        frn_p = dynamic_params_state.get('frn_params', frn_p)
        current_nonmkt_params = dynamic_params_state.get('nonmarketable_params', current_nonmkt_params)
        rate_sensitive_p = dynamic_params_state.get('rate_sensitive_demand', rate_sensitive_p)
        financing_cost_options = dynamic_params_state.get('financing_cost_options', financing_cost_options)
        current_yield_curve_years = yield_p.get('years', current_yield_curve_years)
        current_yield_curve_rates = yield_p.get('rates', current_yield_curve_rates)
        q_start_spending = current_fiscal_params.get('initial_weekly_spending', q_start_spending)
        q_start_taxes = current_fiscal_params.get('initial_weekly_taxes', q_start_taxes)
        spending_growth_qtr = current_fiscal_params.get('spending_growth_qtr', spending_growth_qtr)
        tax_growth_qtr = current_fiscal_params.get('tax_growth_qtr', tax_growth_qtr)
        du_share_spending = current_fiscal_params.get('du_share_spending', du_share_spending)
        du_share_taxes = current_fiscal_params.get('du_share_taxes', du_share_taxes)
        q_end_target_spending = q_start_spending * (1 + spending_growth_qtr)
        q_end_target_taxes = q_start_taxes * (1 + tax_growth_qtr)
        tga_target = current_tga_params.get('target_balance', tga_target)
        tga_floor = current_tga_params.get('floor', tga_floor)
        reserve_transfer = current_other_flows.get('reserve_transfer', reserve_transfer)
        cb_net_expense = current_other_flows.get('cb_net_expense', cb_net_expense)
        money_minting_transfers = current_other_flows.get('money_minting_transfers', money_minting_transfers)
        cpi_start = tips_p.get('cpi_start_level', cpi_start)
        cpi_inflation = tips_p.get('cpi_annual_inflation', cpi_inflation)
        ref_cpi_lag = tips_p.get('ref_cpi_lag_months', ref_cpi_lag)
        reference_cpi_start = tips_p.get('reference_cpi_start_level', cpi_start)
        tips_real_coupon = tips_p.get('default_real_coupon_rate', tips_real_coupon)
        frn_benchmark_mat = frn_p.get('benchmark_maturity_years', frn_benchmark_mat)
        frn_spread = frn_p.get('default_fixed_spread', frn_spread)
        nonmkt_rate_mats = current_nonmkt_params.get('interest_rate_basis_maturities', nonmkt_rate_mats)
        nonmkt_credit_freq = current_nonmkt_params.get('interest_crediting_frequency', nonmkt_credit_freq)
        include_tips_inflation_accretion = bool(financing_cost_options.get('include_tips_inflation_accretion', include_tips_inflation_accretion))
    if cbo_funding_mode:
        cpi_start = _cbo_macro_cpi_level(cbo_inputs, cbo_scenario_id, t0, cpi_start)
    reference_cpi_start = tips_p.get('reference_cpi_start_level', cpi_start)
    cbo_cpi_lookup = {}
    cbo_reference_cpi_lookup = {}
    if cbo_funding_mode:
        tips_cpi_path = cbo_inputs.get('tips_cpi_path') if isinstance(cbo_inputs, dict) else pd.DataFrame()
        if isinstance(tips_cpi_path, pd.DataFrame) and not tips_cpi_path.empty:
            cbo_cpi_lookup, cbo_reference_cpi_lookup = build_tips_daily_cpi_lookups_from_monthly_path(
                tips_cpi_path,
                dates,
                scenario_id=cbo_scenario_id,
                default_value=reference_cpi_start,
                reference_lag_months=ref_cpi_lag,
            )
        else:
            cbo_cpi_lookup = _build_cbo_macro_cpi_lookup(cbo_inputs, cbo_scenario_id, dates, cpi_start)
            cbo_reference_cpi_lookup = build_projected_cpi_lookup_from_macro(
                cbo_inputs.get('macro_forecast') if isinstance(cbo_inputs, dict) else None,
                dates,
                scenario_id=cbo_scenario_id,
                default_value=reference_cpi_start,
                lag_months=ref_cpi_lag,
                anchor_date=t0,
                anchor_value=reference_cpi_start,
            )
    if cbo_funding_mode and cbo_cpi_lookup:
        cpi_start = cbo_cpi_lookup.get(pd.to_datetime(t0).normalize(), cpi_start)
    initial_bonds_df_global = params.get('initial_bonds_df')
    bond_portfolio = pd.DataFrame(columns=BOND_PORTFOLIO_COLS)
    if isinstance(initial_bonds_df_global, pd.DataFrame) and (not initial_bonds_df_global.empty):
        bond_portfolio = initial_bonds_df_global.copy(deep=True)
        for col in ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']:
            if col not in bond_portfolio.columns:
                bond_portfolio[col] = np.nan
        for col, dtype in PORTFOLIO_DTYPES.items():
            if col not in bond_portfolio.columns:
                bond_portfolio[col] = pd.NA if dtype == 'Int64' else pd.NaT if dtype == 'datetime64[ns]' else np.nan
        bond_portfolio = bond_portfolio[BOND_PORTFOLIO_COLS]
        try:
            bond_portfolio = bond_portfolio.astype(PORTFOLIO_DTYPES, errors='ignore')
        except Exception:
            bond_portfolio = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
        if not bond_portfolio.empty:
            bond_portfolio['HolderSubBucket'] = (
                bond_portfolio['HolderSubBucket']
                .fillna('')
                .astype(str)
                .replace({'<NA>': '', 'nan': '', 'None': ''})
            )
            private_mask = bond_portfolio['HolderType'] == 'Private'
            valid_private_subbucket = bond_portfolio['HolderSubBucket'].isin(PRIVATE_SUBBUCKETS)
            bond_portfolio.loc[private_mask & ~valid_private_subbucket, 'HolderSubBucket'] = (
                PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
            )
            bond_portfolio.loc[~private_mask, 'HolderSubBucket'] = ''
            issue_price_missing = bond_portfolio['IssuePriceRatio'].isna() | (bond_portfolio['IssuePriceRatio'] <= TGA_FLOOR_TOLERANCE)
            bond_portfolio.loc[issue_price_missing, 'IssuePriceRatio'] = 1.0
            non_bill_mask = ~((bond_portfolio['SecurityType'] == 'Fixed') & (bond_portfolio['CouponRate'].fillna(0.0) <= TGA_FLOOR_TOLERANCE) & (bond_portfolio['OriginalMaturityYears'].fillna(np.inf) <= 1.0 + TGA_FLOOR_TOLERANCE))
            issue_proceeds_missing = bond_portfolio['IssueProceeds'].isna() | (bond_portfolio['IssueProceeds'] <= TGA_FLOOR_TOLERANCE)
            bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'IssueProceeds'] = bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'FaceValue'] * bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'IssuePriceRatio']
            issue_yield_missing = bond_portfolio['IssueYieldAtIssue'].isna()
            bond_portfolio.loc[issue_yield_missing & non_bill_mask, 'IssueYieldAtIssue'] = bond_portfolio.loc[issue_yield_missing & non_bill_mask, 'CouponRate']
            bill_mask = (bond_portfolio['SecurityType'] == 'Fixed') & (bond_portfolio['CouponRate'].fillna(0.0) <= TGA_FLOOR_TOLERANCE) & (bond_portfolio['OriginalMaturityYears'].fillna(np.inf) <= 1.0 + TGA_FLOOR_TOLERANCE)
            bill_needs_inference = bill_mask & (bond_portfolio['IssueYieldAtIssue'].isna() | bond_portfolio['IssueProceeds'].isna() | (bond_portfolio['IssueProceeds'] >= bond_portfolio['FaceValue'] - 1e-12))
            if bill_needs_inference.any():
                inferred_issue_data = bond_portfolio.loc[bill_needs_inference].apply(lambda row: pd.Series(infer_issue_data_for_loaded_bill(row['FaceValue'], row['OriginalMaturityYears'], row.get('IssueYieldAtIssue'), current_yield_curve_years, current_yield_curve_rates), index=['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue']), axis=1)
                bond_portfolio.loc[bill_needs_inference, ['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue']] = inferred_issue_data.values
    else:
        bond_portfolio = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
    t0 = dates[0]
    init_vals = params.get('initial_values', {})
    results.loc[t0, 'TGA'] = init_vals.get('tga', 0.0)
    results.loc[t0, 'Reserves'] = init_vals.get('reserves', 0.0)
    results.loc[t0, 'TDC_Level'] = init_vals.get('tdc_level', init_vals.get('deposits', 0.0))
    results.loc[t0, 'CB_DeferredAsset'] = 0.0
    results.loc[t0, 'CPI_Level'] = cpi_start
    results.loc[t0, 'Reference_CPI'] = cbo_reference_cpi_lookup.get(
        pd.to_datetime(t0).normalize(),
        reference_cpi_start,
    )
    results.loc[t0, 'CBOFundingModeActive'] = 1.0 if cbo_funding_mode else 0.0
    results.loc[t0, ['CBOControlledDebtTarget', 'CBOControlledDebtPreIssuance', 'CBOControlledDebtPostIssuance', 'CBOControlledDebtTargetError', 'CBOOperatingCashTarget', 'CBOCashResidual', 'CBOFiscalIncidencePolicyPresent', 'CBOFedHoldingsTarget', 'CBOFedHoldingsTargetError', 'CBOFedAuctionShare', 'CBONetInterestBridgeRows']] = 0.0
    if cbo_funding_mode:
        results.loc[t0, 'CBOFedStockMode'] = 'synthetic_cb_treasury_stock_target_par_reallocation'
        results.loc[t0, 'CBOFedSettlementScope'] = 'not_applicable_opening_row'
        results.loc[t0, 'CBORemittanceStatus'] = 'not_modeled_cbo_primary_deficit_embeds_baseline_revenues'
        results.loc[t0, ['CB_Remittance', 'CB_DeferredAsset']] = np.nan
    if not bond_portfolio.empty:
        active_bonds_initial = bond_portfolio[bond_portfolio['Status'] == 'Active'].copy()
        active_bonds_initial['DebtBase'] = np.where(active_bonds_initial['SecurityType'] == 'TIPS', active_bonds_initial['AdjustedPrincipal'].fillna(active_bonds_initial['FaceValue']), active_bonds_initial['FaceValue'])
        results.loc[t0, 'TotalDebt_Agg'] = active_bonds_initial['DebtBase'].sum()
        for holder in HOLDER_TYPES:
            results.loc[t0, f'DebtHeld_{holder}'] = active_bonds_initial.loc[active_bonds_initial['HolderType'] == holder, 'DebtBase'].sum()
        for sec_type in SECURITY_TYPES:
            results.loc[t0, f'DebtHeldByType_{sec_type}'] = active_bonds_initial.loc[active_bonds_initial['SecurityType'] == sec_type, 'DebtBase'].sum()
        active_bonds_initial['TimeToMaturity'] = (active_bonds_initial['MaturityDate'] - t0).dt.total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
        active_bonds_initial['TimeToMaturity'] = active_bonds_initial['TimeToMaturity'].clip(lower=0.0)
        wam_num = (active_bonds_initial['DebtBase'] * active_bonds_initial['TimeToMaturity']).sum()
        wam_den = results.loc[t0, 'TotalDebt_Agg']
        results.loc[t0, 'WAM'] = wam_num / wam_den if wam_den > TGA_FLOOR_TOLERANCE else 0.0
    else:
        results.loc[t0, ['TotalDebt_Agg', 'WAM']] = 0.0
        for holder in HOLDER_TYPES:
            results.loc[t0, f'DebtHeld_{holder}'] = 0.0
        for sec_type in SECURITY_TYPES:
            results.loc[t0, f'DebtHeldByType_{sec_type}'] = 0.0
    results.loc[t0, 'GovSpending'] = q_start_spending
    results.loc[t0, 'Taxes'] = q_start_taxes
    results.loc[t0, ['PrimaryDeficit', 'InterestPaid_Bonds', 'PrincipalPaid_Bonds', 'InterestOutlay_Period', 'InterestOutlay_Cumulative', 'PrincipalRollover_Period', 'PrincipalRollover_Cumulative', 'NewDebtIssued', 'AuctionProceeds', 'IssuanceProceedsTarget', 'IssueDiscountCost_Period', 'IssueDiscountCost_Cumulative', 'FinancingCost_Period', 'FinancingCost_Cumulative', 'NonMarketableInterestCapitalized_Period', 'NonMarketableInterestCapitalized_Cumulative', 'TIPSInflationAccretion_Period', 'TIPSInflationAccretion_Cumulative', 'DebtServiceOutlay_Period', 'DebtServiceOutlay_Cumulative']] = 0.0
    results.loc[t0, ['ReserveChange', 'TDC_Change', 'TGAChange']] = 0.0
    results.loc[t0, 'CB_InterestIncome'] = 0.0
    if cbo_funding_mode:
        results.loc[t0, ['CB_NetIncome', 'CB_Remittance', 'CB_DeferredAsset']] = np.nan
    else:
        results.loc[t0, ['CB_NetIncome', 'CB_Remittance']] = 0.0
    max_existing_id = 0
    if not bond_portfolio.empty and bond_portfolio['BondID'].notna().any():
        numeric_bond_ids = pd.to_numeric(bond_portfolio['BondID'], errors='coerce')
        if numeric_bond_ids.notna().any():
            max_existing_id = numeric_bond_ids.max()
    bond_id_counter = int(max_existing_id) + 1
    last_quarter_start_date = t0
    weeks_in_quarter = 0
    for i in range(1, num_periods):
        prev_date = dates[i - 1]
        current_date = dates[i]
        delta_t_days = (current_date - prev_date).days
        delta_t_years = delta_t_days / DAYS_PER_YEAR_ACTUAL
        triggered_events_in_period = False
        for event_date in sorted(scheduled_events.keys()):
            if prev_date < event_date <= current_date:
                if not triggered_events_in_period:
                    triggered_events_in_period = True
                apply_event_actions(scheduled_events[event_date], dynamic_params_state, scenario_name, current_date, propagate_legacy_sector_prefs_to_auction=uses_legacy_sector_prefs_for_auction, propagate_legacy_sector_prefs_to_secondary=uses_legacy_sector_prefs_for_secondary)
        if triggered_events_in_period:
            current_enable_trading = dynamic_params_state.get('simulation_period', {}).get('enable_preference_trading', False)
            current_tga_params = dynamic_params_state.get('tga_params', current_tga_params)
            current_other_flows = dynamic_params_state.get('other_flows', current_other_flows)
            current_fiscal_params = dynamic_params_state.get('fiscal_params', current_fiscal_params)
            issuance_profile = dynamic_params_state.get('treasury_issuance_profile', issuance_profile)
            current_sector_prefs = dynamic_params_state.get('sector_preferences', current_sector_prefs)
            current_auction_prefs = dynamic_params_state.get('auction_absorption_preferences', current_auction_prefs)
            current_secondary_prefs = dynamic_params_state.get('secondary_target_preferences', current_secondary_prefs)
            yield_p = dynamic_params_state.get('yield_curve', yield_p)
            tips_p = dynamic_params_state.get('tips_params', tips_p)
            frn_p = dynamic_params_state.get('frn_params', frn_p)
            current_nonmkt_params = dynamic_params_state.get('nonmarketable_params', current_nonmkt_params)
            rate_sensitive_p = dynamic_params_state.get('rate_sensitive_demand', rate_sensitive_p)
            financing_cost_options = dynamic_params_state.get('financing_cost_options', financing_cost_options)
            current_yield_curve_years = yield_p.get('years', current_yield_curve_years)
            current_yield_curve_rates = yield_p.get('rates', current_yield_curve_rates)
            tga_target = current_tga_params.get('target_balance', tga_target)
            tga_floor = current_tga_params.get('floor', tga_floor)
            q_start_spending = current_fiscal_params.get('initial_weekly_spending', q_start_spending)
            q_start_taxes = current_fiscal_params.get('initial_weekly_taxes', q_start_taxes)
            spending_growth_qtr = current_fiscal_params.get('spending_growth_qtr', spending_growth_qtr)
            tax_growth_qtr = current_fiscal_params.get('tax_growth_qtr', tax_growth_qtr)
            du_share_spending = current_fiscal_params.get('du_share_spending', du_share_spending)
            du_share_taxes = current_fiscal_params.get('du_share_taxes', du_share_taxes)
            q_end_target_spending = q_start_spending * (1 + spending_growth_qtr)
            q_end_target_taxes = q_start_taxes * (1 + tax_growth_qtr)
            last_quarter_start_date = current_date
            weeks_in_quarter = 0
            reserve_transfer = current_other_flows.get('reserve_transfer', reserve_transfer)
            cb_net_expense = current_other_flows.get('cb_net_expense', cb_net_expense)
            money_minting_transfers = current_other_flows.get('money_minting_transfers', money_minting_transfers)
            cpi_inflation = tips_p.get('cpi_annual_inflation', cpi_inflation)
            ref_cpi_lag = tips_p.get('ref_cpi_lag_months', ref_cpi_lag)
            tips_real_coupon = tips_p.get('default_real_coupon_rate', tips_real_coupon)
            frn_benchmark_mat = frn_p.get('benchmark_maturity_years', frn_benchmark_mat)
            frn_spread = frn_p.get('default_fixed_spread', frn_spread)
            nonmkt_rate_mats = current_nonmkt_params.get('interest_rate_basis_maturities', nonmkt_rate_mats)
            nonmkt_credit_freq = current_nonmkt_params.get('interest_crediting_frequency', nonmkt_credit_freq)
            include_tips_inflation_accretion = bool(financing_cost_options.get('include_tips_inflation_accretion', include_tips_inflation_accretion))
        if not yield_curve_surface.empty:
            surface_years, surface_rates, yield_curve_surface_status = curve_for_date(
                yield_curve_surface,
                current_date,
                scenario_id=surface_scenario_id,
            )
            if surface_years and surface_rates:
                current_yield_curve_years = surface_years
                current_yield_curve_rates = surface_rates
        reserve_change_period = 0.0
        deposit_change_period = 0.0
        tga_change_period = 0.0
        tga_inflow_secondary_mkt = 0.0
        tga_drain_secondary_mkt = 0.0
        pref_trade_monetary_impact = {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0}
        nonmkt_interest_capitalized_period = 0.0
        tips_inflation_accretion_period = 0.0
        cb_tips_indexation_period = 0.0
        auction_shift_weighted_sum = 0.0
        auction_shift_weighted_max = 0.0
        auction_shift_weight_total = 0.0
        secondary_shift_avg = 0.0
        secondary_shift_max = 0.0
        prev_cpi = results.loc[prev_date, 'CPI_Level']
        if cbo_funding_mode:
            current_cpi = cbo_cpi_lookup.get(pd.to_datetime(current_date).normalize(), prev_cpi)
        else:
            current_cpi = prev_cpi * (1 + cpi_inflation) ** delta_t_years
        results.loc[current_date, 'CPI_Level'] = current_cpi
        normalized_current_date = pd.to_datetime(current_date).normalize()
        if cbo_funding_mode and cbo_reference_cpi_lookup:
            current_ref_cpi = cbo_reference_cpi_lookup.get(
                normalized_current_date,
                results.loc[prev_date, 'Reference_CPI'],
            )
        else:
            ref_cpi_date = current_date - relativedelta(months=ref_cpi_lag)
            if ref_cpi_date < dates[0]:
                current_ref_cpi = results.loc[dates[0], 'CPI_Level']
            else:
                try:
                    ref_cpi_idx_loc = results.index.get_indexer([ref_cpi_date], method='ffill')[0]
                    current_ref_cpi = results.loc[results.index[ref_cpi_idx_loc], 'CPI_Level']
                except Exception:
                    current_ref_cpi = results.loc[prev_date, 'Reference_CPI']
        results.loc[current_date, 'Reference_CPI'] = current_ref_cpi
        tips_mask = (bond_portfolio['Status'] == 'Active') & (bond_portfolio['SecurityType'] == 'TIPS')
        if tips_mask.any():
            prev_adjusted_principal = bond_portfolio.loc[tips_mask, 'AdjustedPrincipal'].fillna(
                bond_portfolio.loc[tips_mask, 'OriginalPrincipal']
            )
            ref_cpi_issue = bond_portfolio.loc[tips_mask, 'ReferenceCPI_Issue']
            index_ratio = np.where(ref_cpi_issue > TGA_FLOOR_TOLERANCE, current_ref_cpi / ref_cpi_issue, 1.0)
            bond_portfolio.loc[tips_mask, 'IndexRatio'] = index_ratio
            original_principal = bond_portfolio.loc[tips_mask, 'OriginalPrincipal']
            new_adjusted_principal = original_principal * index_ratio
            bond_portfolio.loc[tips_mask, 'AdjustedPrincipal'] = new_adjusted_principal
            if include_tips_inflation_accretion:
                tips_inflation_accretion_period = (new_adjusted_principal - prev_adjusted_principal).sum()
            cb_tips_mask = tips_mask & bond_portfolio['HolderType'].eq('CB')
            if cb_tips_mask.any():
                cb_tips_indices = bond_portfolio.index[cb_tips_mask]
                cb_tips_indexation_period = float(
                    (
                        bond_portfolio.loc[cb_tips_indices, 'AdjustedPrincipal'].fillna(
                            bond_portfolio.loc[cb_tips_indices, 'FaceValue']
                        )
                        - prev_adjusted_principal.loc[cb_tips_indices]
                    ).sum()
                )
        frn_mask = (bond_portfolio['Status'] == 'Active') & (bond_portfolio['SecurityType'] == 'FRN')
        frn_coupon_payment_amounts = {}
        frn_maturity_payment_amounts = {}
        if frn_mask.any():
            frn_path_frame = (
                cbo_inputs.get('frn_rate_path')
                if cbo_funding_mode and isinstance(cbo_inputs, dict)
                else pd.DataFrame()
            )
            frn_path_date_lookup = (
                cbo_inputs.get('frn_rate_path_date_lookup', {})
                if cbo_funding_mode and isinstance(cbo_inputs, dict)
                else {}
            )
            frn_path_row = None
            day_count_basis = FRN_DAY_COUNT_BASIS
            if isinstance(frn_path_frame, pd.DataFrame) and not frn_path_frame.empty:
                frn_path_row = _frn_rate_path_row_for_date(
                    frn_path_frame,
                    cbo_scenario_id,
                    current_date,
                    lookup=frn_path_date_lookup,
                )
                if frn_path_row is None:
                    frn_path_row = _lookup_cbo_period_row(
                        cbo_inputs.get('frn_rate_path_lookup', {}),
                        frn_path_frame,
                        cbo_scenario_id,
                        prev_date,
                        current_date,
                        'CBO FRN rate path',
                    )
                frn_yield = float(frn_path_row['benchmark_rate_decimal'])
                day_count_basis = float(frn_path_row.get('day_count_basis', FRN_DAY_COUNT_BASIS) or FRN_DAY_COUNT_BASIS)
            else:
                frn_yield = get_yield_for_maturity(
                    frn_benchmark_mat,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                )
                frn_yield = 0.0 if pd.isna(frn_yield) else frn_yield
            lockout_business_days = _frn_lockout_business_days(frn_path_row)
            accrual_dates = pd.date_range(
                pd.Timestamp(prev_date).normalize() + pd.Timedelta(days=1),
                pd.Timestamp(current_date).normalize(),
                freq='D',
            )
            if len(accrual_dates) == 1:
                accrual_day = pd.Timestamp(accrual_dates[0]).normalize()
                frn_indices = bond_portfolio.index[frn_mask]
                benchmark_rates = pd.Series(float(frn_yield), index=frn_indices, dtype=float)
                day_count_bases = pd.Series(float(day_count_basis), index=frn_indices, dtype=float)
                if (
                    lockout_business_days > 0
                    and isinstance(frn_path_frame, pd.DataFrame)
                    and not frn_path_frame.empty
                ):
                    lockout_horizon = _add_business_days(accrual_day, lockout_business_days + 1)
                    frn_tranches = bond_portfolio.loc[frn_mask]
                    due_dates = pd.date_range(accrual_day, lockout_horizon + pd.Timedelta(days=3), freq='D')
                    for due_date in due_dates:
                        due_day = pd.Timestamp(due_date).normalize()
                        lockout_start = _frn_lockout_start_date(due_day, lockout_business_days)
                        if pd.isna(lockout_start) or not (lockout_start <= accrual_day <= due_day):
                            continue
                        due_indices = set(_daily_coupon_due_indices(frn_tranches, due_day, frequency=4))
                        maturity_dates = pd.to_datetime(frn_tranches['MaturityDate'], errors='coerce').dt.normalize()
                        due_indices.update(frn_tranches.loc[maturity_dates == due_day].index.tolist())
                        due_indices.intersection_update(frn_indices)
                        if not due_indices:
                            continue
                        lockout_row = _frn_rate_path_row_for_date(
                            frn_path_frame,
                            cbo_scenario_id,
                            lockout_start,
                            lookup=frn_path_date_lookup,
                        )
                        if lockout_row is None:
                            lockout_row = frn_path_row
                        benchmark_rates.loc[list(due_indices)] = float(lockout_row['benchmark_rate_decimal'])
                        day_count_bases.loc[list(due_indices)] = float(
                            lockout_row.get('day_count_basis', FRN_DAY_COUNT_BASIS) or FRN_DAY_COUNT_BASIS
                        )
                fixed_spread = pd.to_numeric(bond_portfolio.loc[frn_indices, 'FixedSpread'], errors='coerce').fillna(0.0)
                face_values = pd.to_numeric(bond_portfolio.loc[frn_indices, 'FaceValue'], errors='coerce').fillna(0.0)
                daily_rates = (benchmark_rates + fixed_spread) / day_count_bases
                if frn_path_row is None:
                    daily_rates = daily_rates.clip(lower=0.0)
                bond_portfolio.loc[frn_indices, 'AccruedInterest_FRN'] = (
                    bond_portfolio.loc[frn_indices, 'AccruedInterest_FRN'].fillna(0.0) + face_values * daily_rates
                )
                bond_portfolio.loc[frn_indices, 'BenchmarkRate_FRN'] = benchmark_rates
                bond_portfolio.loc[frn_indices, 'LastAccrualDate'] = accrual_day
            else:
                for idx, bond in bond_portfolio.loc[frn_mask].copy().iterrows():
                    face_value = pd.to_numeric(bond.get('FaceValue'), errors='coerce')
                    face_value = 0.0 if pd.isna(face_value) else float(face_value)
                    fixed_spread = pd.to_numeric(bond.get('FixedSpread'), errors='coerce')
                    fixed_spread = 0.0 if pd.isna(fixed_spread) else float(fixed_spread)
                    accrued_balance = pd.to_numeric(bond.get('AccruedInterest_FRN'), errors='coerce')
                    accrued_balance = 0.0 if pd.isna(accrued_balance) else float(accrued_balance)
                    coupon_dates = {
                        pd.Timestamp(value).normalize()
                        for value in _frn_coupon_dates_for_period(bond, prev_date, current_date)
                    }
                    maturity_date = pd.Timestamp(bond.get('MaturityDate')).normalize()
                    last_benchmark_rate = float(frn_yield)
                    last_accrual_date = pd.NaT
                    for accrual_day in accrual_dates:
                        if pd.notna(maturity_date) and accrual_day > maturity_date:
                            break
                        rate_date = _frn_rate_date_for_accrual_day(bond, accrual_day, lockout_business_days)
                        rate_row = None
                        benchmark_rate = float(frn_yield)
                        rate_day_count = day_count_basis
                        if isinstance(frn_path_frame, pd.DataFrame) and not frn_path_frame.empty:
                            rate_row = _frn_rate_path_row_for_date(
                                frn_path_frame,
                                cbo_scenario_id,
                                rate_date,
                                lookup=frn_path_date_lookup,
                            )
                            if rate_row is None:
                                rate_row = frn_path_row
                            benchmark_rate = float(rate_row['benchmark_rate_decimal'])
                            rate_day_count = float(rate_row.get('day_count_basis', FRN_DAY_COUNT_BASIS) or FRN_DAY_COUNT_BASIS)
                        daily_rate = (benchmark_rate + fixed_spread) / rate_day_count
                        if rate_row is None:
                            daily_rate = max(0.0, daily_rate)
                        accrued_balance += face_value * daily_rate
                        last_benchmark_rate = benchmark_rate
                        last_accrual_date = accrual_day
                        if pd.notna(maturity_date) and accrual_day == maturity_date:
                            frn_maturity_payment_amounts[idx] = accrued_balance
                            accrued_balance = 0.0
                            break
                        if accrual_day in coupon_dates:
                            frn_coupon_payment_amounts[idx] = frn_coupon_payment_amounts.get(idx, 0.0) + accrued_balance
                            accrued_balance = 0.0
                    bond_portfolio.loc[idx, 'BenchmarkRate_FRN'] = last_benchmark_rate
                    bond_portfolio.loc[idx, 'AccruedInterest_FRN'] = accrued_balance
                    if pd.notna(last_accrual_date):
                        bond_portfolio.loc[idx, 'LastAccrualDate'] = last_accrual_date
        for col in ['TimeToMaturity', 'DiscountYield', 'CleanPrice', 'AccruedInterest', 'DirtyValue', 'DirtyPriceRatio']:
            if col in bond_portfolio.columns:
                bond_portfolio[col] = np.nan
        if current_date.quarter != prev_date.quarter:
            q_start_spending = results.loc[prev_date, 'GovSpending']
            q_start_taxes = results.loc[prev_date, 'Taxes']
            q_end_target_spending = q_start_spending * (1 + spending_growth_qtr)
            q_end_target_taxes = q_start_taxes * (1 + tax_growth_qtr)
            last_quarter_start_date = current_date
            weeks_in_quarter = 0
        weeks_in_quarter += 1
        quarter_end_date = last_quarter_start_date + pd.offsets.QuarterEnd(0)
        days_in_qtr = (quarter_end_date - last_quarter_start_date).days + 1
        approx_weeks_in_qtr = max(1.0, days_in_qtr / 7.0)
        interp_weight = min(1.0, weeks_in_quarter / approx_weeks_in_qtr)
        gov_spending_period = q_start_spending + (q_end_target_spending - q_start_spending) * interp_weight
        taxes_period = q_start_taxes + (q_end_target_taxes - q_start_taxes) * interp_weight
        path_fiscal_flow = primary_flow_for_period(
            primary_flow_lookup,
            scenario_name=scenario_name,
            current_date=current_date,
            period_counts_by_quarter=period_counts_by_quarter,
            warning_cache=primary_flow_warning_cache,
        )
        cbo_period_primary_deficit = None
        cbo_incidence = None
        cbo_fiscal_incidence_status = 'not_applicable'
        if cbo_funding_mode:
            cbo_primary_row = _cbo_period_row(
                cbo_inputs['primary_deficit'],
                cbo_scenario_id,
                prev_date,
                current_date,
                'CBO primary deficit path',
            )
            cbo_period_primary_deficit = float(cbo_primary_row['primary_deficit_bil'])
            cbo_incidence = _cbo_incidence_from_config(params)
            if cbo_incidence is None:
                cbo_incidence = _cbo_incidence_row(cbo_inputs['fiscal_incidence'], cbo_scenario_id)
            cbo_fiscal_incidence_status = (
                cbo_incidence.get('status', 'explicit_policy_present')
                if cbo_incidence is not None
                else 'missing_policy_not_routed_to_tdc'
            )
            gov_spending_period = max(cbo_period_primary_deficit, 0.0)
            taxes_period = max(-cbo_period_primary_deficit, 0.0)
        elif path_fiscal_flow is not None:
            primary_flow_path_used_count += 1
            gov_spending_period = max(float(path_fiscal_flow), 0.0)
            taxes_period = max(-float(path_fiscal_flow), 0.0)
        results.loc[current_date, 'GovSpending'] = gov_spending_period
        results.loc[current_date, 'Taxes'] = taxes_period
        results.loc[current_date, 'PrimaryDeficit'] = gov_spending_period - taxes_period
        fiscal_reserve_change = gov_spending_period - taxes_period
        if cbo_funding_mode:
            signed_flow = cbo_period_primary_deficit
            fiscal_deposit_change = (
                signed_flow * float(cbo_incidence['du_share'])
                if cbo_incidence is not None
                else 0.0
            )
            results.loc[current_date, 'CBOPrimaryDeficitFlow'] = signed_flow
            results.loc[current_date, 'FiscalIncidenceStatus'] = cbo_fiscal_incidence_status
        else:
            fiscal_deposit_change = (
                gov_spending_period * float(du_share_spending)
                - taxes_period * float(du_share_taxes)
            )
        fiscal_tga_change = taxes_period - gov_spending_period
        reserve_change_period += fiscal_reserve_change
        deposit_change_period += fiscal_deposit_change
        tga_change_period += fiscal_tga_change
        principal_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        principal_component_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        bill_discount_interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        coupon_interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        frn_interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        tips_coupon_interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        tips_inflation_compensation_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        principal_paid_by_private_route = _zero_private_routes()
        interest_paid_by_private_route = _zero_private_routes()
        principal_component_paid_by_private_route = _zero_private_routes()
        bill_discount_interest_paid_by_private_route = _zero_private_routes()
        coupon_interest_paid_by_private_route = _zero_private_routes()
        frn_interest_paid_by_private_route = _zero_private_routes()
        tips_coupon_interest_paid_by_private_route = _zero_private_routes()
        tips_inflation_compensation_paid_by_private_route = _zero_private_routes()
        total_principal_paid_period = 0.0
        total_interest_paid_period = 0.0
        maturing_mask = bond_portfolio['MaturityDate'].notna() & (bond_portfolio['MaturityDate'] > prev_date) & (bond_portfolio['MaturityDate'] <= current_date) & (bond_portfolio['Status'] == 'Active')
        maturing_indices = bond_portfolio[maturing_mask].index
        if not maturing_indices.empty:
            maturing_tranches = bond_portfolio.loc[maturing_indices].copy()
            bond_portfolio.loc[maturing_indices, 'Status'] = 'Matured'
            for idx, bond in maturing_tranches.iterrows():
                principal_payment = 0.0
                interest_payment = 0.0
                bill_discount_interest_payment = 0.0
                tips_inflation_compensation_payment = 0.0
                holder = bond['HolderType']
                face_value = bond['FaceValue']
                coupon_rate = bond['CouponRate']
                maturity_date = bond['MaturityDate']
                security_type = bond['SecurityType']
                if security_type == 'TIPS':
                    principal_payment = max(bond.get('OriginalPrincipal', 0.0), bond.get('AdjustedPrincipal', 0.0))
                    original_principal = bond.get('OriginalPrincipal', 0.0)
                    tips_inflation_compensation_payment = max(0.0, principal_payment - original_principal)
                elif security_type in ['Fixed', 'FRN', 'NonMarketable']:
                    principal_payment = face_value
                    original_maturity_years = bond.get('OriginalMaturityYears', np.inf)
                    original_maturity_years = (
                        np.inf
                        if pd.isna(original_maturity_years)
                        else float(original_maturity_years)
                    )
                    is_bill = (
                        security_type == 'Fixed'
                        and coupon_rate <= TGA_FLOOR_TOLERANCE
                        and original_maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE
                    )
                    if is_bill:
                        issue_proceeds = bond.get('IssueProceeds', face_value)
                        issue_proceeds = 0.0 if pd.isna(issue_proceeds) else float(issue_proceeds)
                        bill_discount_interest_payment = max(0.0, face_value - issue_proceeds)
                if security_type == 'FRN':
                    interest_payment = frn_maturity_payment_amounts.get(idx, bond.get('AccruedInterest_FRN', 0.0))
                    bond_portfolio.loc[idx, 'AccruedInterest_FRN'] = 0.0
                    frn_interest_paid_by_holder[holder] += interest_payment
                elif security_type in ['Fixed', 'TIPS'] and coupon_rate > TGA_FLOOR_TOLERANCE:
                    coupon_freq = 2
                    adj_p = bond.get('AdjustedPrincipal') if security_type == 'TIPS' else None
                    principal_base = adj_p if security_type == 'TIPS' and adj_p is not None else face_value
                    interest_payment = principal_base * coupon_rate / coupon_freq
                    if security_type == 'TIPS':
                        tips_coupon_interest_paid_by_holder[holder] += interest_payment
                    else:
                        coupon_interest_paid_by_holder[holder] += interest_payment
                principal_paid_by_holder[holder] += principal_payment
                interest_paid_by_holder[holder] += interest_payment
                principal_component_paid_by_holder[holder] += max(
                    0.0, principal_payment - bill_discount_interest_payment
                )
                bill_discount_interest_paid_by_holder[holder] += bill_discount_interest_payment
                tips_inflation_compensation_paid_by_holder[holder] += (
                    tips_inflation_compensation_payment
                )
                if holder == 'Private':
                    private_route = _private_subbucket(bond)
                    principal_paid_by_private_route[private_route] += principal_payment
                    interest_paid_by_private_route[private_route] += interest_payment
                    principal_component_paid_by_private_route[private_route] += max(
                        0.0, principal_payment - bill_discount_interest_payment
                    )
                    bill_discount_interest_paid_by_private_route[private_route] += bill_discount_interest_payment
                    if security_type == 'FRN':
                        frn_interest_paid_by_private_route[private_route] += interest_payment
                    elif security_type == 'TIPS' and coupon_rate > TGA_FLOOR_TOLERANCE:
                        tips_coupon_interest_paid_by_private_route[private_route] += interest_payment
                    elif security_type == 'Fixed' and coupon_rate > TGA_FLOOR_TOLERANCE:
                        coupon_interest_paid_by_private_route[private_route] += interest_payment
                    tips_inflation_compensation_paid_by_private_route[private_route] += (
                        tips_inflation_compensation_payment
                    )
                total_principal_paid_period += principal_payment
                total_interest_paid_period += interest_payment
        active_mask = bond_portfolio['Status'] == 'Active'
        frn_active_mask = active_mask & (bond_portfolio['SecurityType'] == 'FRN')
        if frn_active_mask.any():
            frn_tranches = bond_portfolio.loc[frn_active_mask]
            frn_paying_indices = _coupon_due_indices_for_period(
                frn_tranches,
                prev_date,
                current_date,
                frequency=4,
            )
            if frn_paying_indices:
                frn_paying_indices = list(set(frn_paying_indices))
                payment_amount = pd.Series(
                    {
                        idx: frn_coupon_payment_amounts.get(
                            idx,
                            bond_portfolio.loc[idx, 'AccruedInterest_FRN'],
                        )
                        for idx in frn_paying_indices
                    },
                    dtype=float,
                ).fillna(0.0)
                pmt_by_holder = payment_amount.groupby(bond_portfolio.loc[frn_paying_indices, 'HolderType']).sum().to_dict()
                for h, pmt in pmt_by_holder.items():
                    interest_paid_by_holder[h] += pmt
                    frn_interest_paid_by_holder[h] += pmt
                private_frn_rows = bond_portfolio.loc[frn_paying_indices].copy()
                private_frn_rows = private_frn_rows[private_frn_rows['HolderType'] == 'Private']
                private_frn_routes = private_frn_rows['HolderSubBucket'].where(
                    private_frn_rows['HolderSubBucket'].isin(PRIVATE_SUBBUCKETS),
                    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                )
                for route in PRIVATE_SUBBUCKETS:
                    route_indices = private_frn_rows[private_frn_routes == route].index
                    route_payment = float(payment_amount.loc[route_indices].sum()) if len(route_indices) else 0.0
                    interest_paid_by_private_route[route] += route_payment
                    frn_interest_paid_by_private_route[route] += route_payment
                fallback_reset_indices = [idx for idx in frn_paying_indices if idx not in frn_coupon_payment_amounts]
                if fallback_reset_indices:
                    bond_portfolio.loc[fallback_reset_indices, 'AccruedInterest_FRN'] = 0.0
                total_interest_paid_period += payment_amount.sum()
        fixed_tips_mask = active_mask & bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS']) & (bond_portfolio['CouponRate'] > TGA_FLOOR_TOLERANCE)
        if fixed_tips_mask.any():
            fixed_tips_tranches = bond_portfolio.loc[fixed_tips_mask]
            if (current_date - prev_date).days <= 1:
                fixed_tips_paying_indices = _daily_coupon_due_indices(fixed_tips_tranches, current_date, frequency=2)
            else:
                fixed_tips_paying_indices = []
                for idx, bond in fixed_tips_tranches.iterrows():
                    issue_dt = bond['IssueDate']
                    maturity_dt = bond['MaturityDate']
                    coupon_dates = get_coupon_dates_in_period(
                        issue_dt,
                        maturity_dt,
                        prev_date,
                        current_date,
                        frequency=2,
                        first_interest_payment_date=bond.get('FirstInterestPaymentDate'),
                        interest_payment_frequency=bond.get('InterestPaymentFrequency'),
                    )
                    if coupon_dates:
                        fixed_tips_paying_indices.append(idx)
            if fixed_tips_paying_indices:
                fixed_tips_paying_indices = list(set(fixed_tips_paying_indices))
                paying_tranches = bond_portfolio.loc[fixed_tips_paying_indices]
                principal_base = np.where(paying_tranches['SecurityType'] == 'TIPS', paying_tranches['AdjustedPrincipal'].fillna(paying_tranches['FaceValue']), paying_tranches['FaceValue'])
                payment_amount = (principal_base * paying_tranches['CouponRate'] / 2.0).clip(lower=0)
                pmt_by_holder = payment_amount.groupby(paying_tranches['HolderType']).sum().to_dict()
                for h, pmt in pmt_by_holder.items():
                    interest_paid_by_holder[h] += pmt
                    fixed_payment = paying_tranches.loc[paying_tranches['HolderType'] == h]
                    if not fixed_payment.empty:
                        fixed_amount = payment_amount.loc[fixed_payment.index]
                        tips_mask_for_holder = fixed_payment['SecurityType'] == 'TIPS'
                        tips_coupon_interest_paid_by_holder[h] += fixed_amount.loc[
                            tips_mask_for_holder
                        ].sum()
                        coupon_interest_paid_by_holder[h] += fixed_amount.loc[
                            ~tips_mask_for_holder
                        ].sum()
                private_fixed_payment = paying_tranches[paying_tranches['HolderType'] == 'Private']
                for route in PRIVATE_SUBBUCKETS:
                    route_rows = private_fixed_payment[
                        private_fixed_payment['HolderSubBucket'].apply(_private_subbucket) == route
                    ]
                    if route_rows.empty:
                        continue
                    route_amount = payment_amount.loc[route_rows.index]
                    route_tips_mask = route_rows['SecurityType'] == 'TIPS'
                    tips_amount = float(route_amount.loc[route_tips_mask].sum())
                    fixed_amount = float(route_amount.loc[~route_tips_mask].sum())
                    interest_paid_by_private_route[route] += tips_amount + fixed_amount
                    tips_coupon_interest_paid_by_private_route[route] += tips_amount
                    coupon_interest_paid_by_private_route[route] += fixed_amount
                total_interest_paid_period += payment_amount.sum()
        nonmkt_mask = active_mask & (bond_portfolio['SecurityType'] == 'NonMarketable')
        if nonmkt_mask.any():
            credit_interest = False
            nonmkt_credit_freq = current_nonmkt_params.get('interest_crediting_frequency', 'semi-annual')
            jun30 = pd.Timestamp(year=current_date.year, month=6, day=30)
            dec31 = pd.Timestamp(year=current_date.year, month=12, day=31)
            if nonmkt_credit_freq == 'semi-annual' and (prev_date < jun30 <= current_date or prev_date < dec31 <= current_date):
                credit_interest = True
            elif nonmkt_credit_freq == 'annual' and prev_date < dec31 <= current_date:
                credit_interest = True
            if credit_interest:
                rate_setting_method = current_nonmkt_params.get('rate_setting_method', 'yield_curve_points')
                calculated_interest_rate = 0.0
                if rate_setting_method == 'avg_outstanding_marketable_yield':
                    min_maturity = current_nonmkt_params.get('marketable_basket_min_remaining_maturity', 4.0)
                    basket_types = current_nonmkt_params.get('marketable_basket_types', ['Fixed', 'TIPS'])
                    weighting = current_nonmkt_params.get('marketable_basket_weighting', 'equal')
                    marketable_bonds_for_rate = bond_portfolio[(bond_portfolio['Status'] == 'Active') & bond_portfolio['SecurityType'].isin(basket_types)].copy()
                    if not marketable_bonds_for_rate.empty:
                        marketable_bonds_for_rate['CurrentTTM'] = (marketable_bonds_for_rate['MaturityDate'] - current_date).dt.total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
                        marketable_bonds_for_rate['CurrentTTM'] = marketable_bonds_for_rate['CurrentTTM'].clip(lower=0.0)
                        eligible_bonds = marketable_bonds_for_rate[marketable_bonds_for_rate['CurrentTTM'] >= min_maturity]
                        if not eligible_bonds.empty:
                            bond_yields = eligible_bonds['CurrentTTM'].apply(
                                lambda ttm: get_yield_for_maturity(
                                    ttm,
                                    current_yield_curve_years,
                                    current_yield_curve_rates,
                                    method=yield_interpolation_method,
                                    floor_zero=yield_floor_zero,
                                )
                            )
                            valid_yields_series = bond_yields.dropna()
                            if not valid_yields_series.empty:
                                eligible_bonds_with_valid_yields = eligible_bonds.loc[valid_yields_series.index]
                                if weighting == 'face_value' and eligible_bonds_with_valid_yields['FaceValue'].sum() > TGA_FLOOR_TOLERANCE:
                                    calculated_interest_rate = np.average(valid_yields_series, weights=eligible_bonds_with_valid_yields['FaceValue'])
                                else:
                                    calculated_interest_rate = valid_yields_series.mean()
                                calculated_interest_rate = max(0.0, calculated_interest_rate)
                else:
                    nonmkt_rate_mats = current_nonmkt_params.get('interest_rate_basis_maturities', [5.0, 10.0])
                    avg_yield = 0.0
                    count = 0
                    for mat_yr in nonmkt_rate_mats:
                        yld = get_yield_for_maturity(
                            mat_yr,
                            current_yield_curve_years,
                            current_yield_curve_rates,
                            method=yield_interpolation_method,
                            floor_zero=yield_floor_zero,
                        )
                        if not pd.isna(yld):
                            avg_yield += yld
                            count += 1
                    calculated_interest_rate = max(0.0, avg_yield / count if count > 0 else 0.0)
                factor = 0.5 if nonmkt_credit_freq == 'semi-annual' else 1.0
                interest_to_credit = (bond_portfolio.loc[nonmkt_mask, 'FaceValue'] * calculated_interest_rate * factor).clip(lower=0)
                bond_portfolio.loc[nonmkt_mask, 'FaceValue'] += interest_to_credit
                nonmkt_interest_capitalized_period = interest_to_credit.sum()
        results.loc[current_date, 'PrincipalPaid_Bonds'] = total_principal_paid_period
        results.loc[current_date, 'InterestPaid_Bonds'] = total_interest_paid_period
        results.loc[current_date, 'PrincipalRollover_Period'] = total_principal_paid_period
        results.loc[current_date, 'InterestOutlay_Period'] = total_interest_paid_period
        results.loc[current_date, 'PrincipalRollover_Cumulative'] = results.loc[prev_date, 'PrincipalRollover_Cumulative'] + total_principal_paid_period
        results.loc[current_date, 'InterestOutlay_Cumulative'] = results.loc[prev_date, 'InterestOutlay_Cumulative'] + total_interest_paid_period
        current_period_outlay = results.loc[current_date, 'PrincipalPaid_Bonds'] + results.loc[current_date, 'InterestPaid_Bonds']
        results.loc[current_date, 'DebtServiceOutlay_Period'] = current_period_outlay
        prev_cumulative_outlay = results.loc[prev_date, 'DebtServiceOutlay_Cumulative']
        results.loc[current_date, 'DebtServiceOutlay_Cumulative'] = prev_cumulative_outlay + current_period_outlay
        results.loc[current_date, 'NonMarketableInterestCapitalized_Period'] = nonmkt_interest_capitalized_period
        results.loc[current_date, 'NonMarketableInterestCapitalized_Cumulative'] = results.loc[prev_date, 'NonMarketableInterestCapitalized_Cumulative'] + nonmkt_interest_capitalized_period
        results.loc[current_date, 'TIPSInflationAccretion_Period'] = tips_inflation_accretion_period
        results.loc[current_date, 'TIPSInflationAccretion_Cumulative'] = results.loc[prev_date, 'TIPSInflationAccretion_Cumulative'] + tips_inflation_accretion_period
        debt_service_tga_change = -(total_principal_paid_period + total_interest_paid_period)
        intragov_interest = sum((interest_paid_by_holder.get(h, 0.0) for h in INTRAGOV_HOLDERS))
        intragov_principal = sum((principal_paid_by_holder.get(h, 0.0) for h in INTRAGOV_HOLDERS))
        if intragov_interest > 0 or intragov_principal > 0:
            debt_service_tga_change += intragov_principal + intragov_interest
        debt_service_reserve_change = 0.0
        for holder in ['Banks', 'Private', 'Foreign']:
            debt_service_reserve_change += principal_paid_by_holder.get(holder, 0.0)
            debt_service_reserve_change += interest_paid_by_holder.get(holder, 0.0)
        debt_service_deposit_change = 0.0
        for holder in ['Private']:
            debt_service_deposit_change += principal_paid_by_holder.get(holder, 0.0) + interest_paid_by_holder.get(holder, 0.0)
        debt_service_deposit_change = _effective_private_amount(
            {
                route: principal_paid_by_private_route[route] + interest_paid_by_private_route[route]
                for route in PRIVATE_SUBBUCKETS
            },
            mmf_deposit_pass_through,
            legacy_total=principal_paid_by_holder.get('Private', 0.0) + interest_paid_by_holder.get('Private', 0.0),
        )
        reserve_change_period += debt_service_reserve_change
        deposit_change_period += debt_service_deposit_change
        tga_change_period += debt_service_tga_change
        cb_interest_income_period = interest_paid_by_holder.get('CB', 0.0)
        if cbo_funding_mode:
            cb_net_income_period = np.nan
            cb_remittance_period = np.nan
            current_deferred_asset = np.nan
            cb_remit_tga_change = 0.0
            results.loc[current_date, 'CBORemittanceStatus'] = (
                'not_modeled_cbo_primary_deficit_embeds_baseline_revenues'
            )
        else:
            prev_deferred_asset = results.loc[prev_date, 'CB_DeferredAsset']
            cb_net_income_period = cb_interest_income_period - cb_net_expense
            cb_remittance_period = 0.0
            current_deferred_asset = prev_deferred_asset
            if current_deferred_asset > TGA_FLOOR_TOLERANCE:
                reduction = min(current_deferred_asset, max(0, cb_net_income_period))
                current_deferred_asset -= reduction
                cb_remittance_period = max(0, cb_net_income_period - reduction)
            elif cb_net_income_period > 0:
                cb_remittance_period = cb_net_income_period
            else:
                current_deferred_asset += abs(cb_net_income_period)
            cb_remit_tga_change = cb_remittance_period
        results.loc[current_date, 'CB_InterestIncome'] = cb_interest_income_period
        results.loc[current_date, 'CB_NetIncome'] = cb_net_income_period
        results.loc[current_date, 'CB_Remittance'] = cb_remittance_period
        results.loc[current_date, 'CB_DeferredAsset'] = current_deferred_asset
        if cbo_funding_mode:
            results.loc[current_date, 'CBORemittanceCashEffect'] = cb_remit_tga_change
        tga_change_period += cb_remit_tga_change
        prev_tga = results.loc[prev_date, 'TGA']
        tga_change_before_issuance = fiscal_tga_change + debt_service_tga_change + (tga_inflow_secondary_mkt - tga_drain_secondary_mkt) + cb_remit_tga_change + reserve_transfer + money_minting_transfers
        projected_tga_pre_issuance = prev_tga + tga_change_before_issuance
        funding_needed = max(0, tga_target - projected_tga_pre_issuance)
        min_issuance_for_floor = max(0, tga_floor - projected_tga_pre_issuance)
        total_issuance_target_period = max(funding_needed, min_issuance_for_floor)
        cbo_face_target_mode = False
        cbo_controlled_debt_target = 0.0
        cbo_pre_issuance_controlled_debt = 0.0
        cbo_operating_cash_target = 0.0
        cbo_cash_residual_period = 0.0
        cbo_cash_reconciliation_residual_input = 0.0
        cbo_cash_residual_status = 'not_applicable'
        cbo_buyback_face_retired = 0.0
        cbo_buyback_cash_paid = 0.0
        cbo_fed_holdings_target = 0.0
        cbo_fed_auction_share = 0.0
        cbo_fed_allocation_override_active = False
        cbo_fed_begin_stock = float(results.loc[prev_date, 'DebtHeld_CB'])
        fed_secondary_purchase_face = 0.0
        fed_secondary_sale_face = 0.0
        if cbo_funding_mode:
            cbo_debt_row = _cbo_exact_date_row(
                cbo_inputs['debt_stock'],
                cbo_scenario_id,
                current_date,
                'period_end',
                'CBO debt stock path',
            )
            cbo_controlled_debt_target = float(cbo_debt_row['marketable_treasury_public_target_bil'])
            cbo_pre_issuance_controlled_debt = _controlled_public_marketable_debt_from_portfolio(bond_portfolio)
            target_tolerance = float(funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001)
            negative_issuance_action = str(
                funding_rule_cfg.get('negative_required_issuance_action', 'error') or 'error'
            )
            raw_required_issuance = cbo_controlled_debt_target - cbo_pre_issuance_controlled_debt
            if raw_required_issuance < -target_tolerance and negative_issuance_action in {
                'retire_shortest_public_marketable',
                'buyback_shortest_public_marketable',
            }:
                amount_to_retire = abs(raw_required_issuance)
                bond_portfolio, retired_amount, retired_by_holder, retired_by_private_route = (
                    _retire_public_marketable_debt_to_target(
                        bond_portfolio,
                        amount_to_retire,
                        current_date,
                    )
                )
                cbo_buyback_face_retired = retired_amount
                cbo_buyback_cash_paid = retired_amount
                for holder, amount in retired_by_holder.items():
                    principal_paid_by_holder[holder] += amount
                    principal_component_paid_by_holder[holder] += amount
                for route, amount in retired_by_private_route.items():
                    principal_paid_by_private_route[route] += amount
                    principal_component_paid_by_private_route[route] += amount
                total_principal_paid_period += retired_amount
                reserve_change_period += sum(retired_by_holder.get(h, 0.0) for h in ['Banks', 'Private', 'Foreign'])
                deposit_change_period += _effective_private_amount(
                    retired_by_private_route,
                    mmf_deposit_pass_through,
                    legacy_total=retired_by_holder.get('Private', 0.0),
                )
                tga_change_period -= retired_amount
                projected_tga_pre_issuance -= retired_amount
                cbo_pre_issuance_controlled_debt = _controlled_public_marketable_debt_from_portfolio(bond_portfolio)
                results.loc[current_date, 'PrincipalPaid_Bonds'] = total_principal_paid_period
                results.loc[current_date, 'PrincipalRollover_Period'] = total_principal_paid_period
                results.loc[current_date, 'DebtServiceOutlay_Period'] = total_principal_paid_period + total_interest_paid_period
                results.loc[current_date, 'PrincipalRollover_Cumulative'] = (
                    results.loc[prev_date, 'PrincipalRollover_Cumulative'] + total_principal_paid_period
                )
                results.loc[current_date, 'DebtServiceOutlay_Cumulative'] = (
                    results.loc[prev_date, 'DebtServiceOutlay_Cumulative']
                    + total_principal_paid_period
                    + total_interest_paid_period
                )
            total_issuance_target_period = calculate_required_face_issuance(
                cbo_controlled_debt_target,
                cbo_pre_issuance_controlled_debt,
                strict=True,
                tolerance=target_tolerance,
            )
            cbo_face_target_mode = True
            results.loc[current_date, 'CBOControlledDebtTargetApplicable'] = 1.0
            if isinstance(cbo_inputs.get('operating_cash'), pd.DataFrame) and not cbo_inputs['operating_cash'].empty:
                cbo_operating_cash_row = _cbo_exact_date_row(
                    cbo_inputs['operating_cash'],
                    cbo_scenario_id,
                    current_date,
                    'period_end',
                    'CBO operating cash path',
                )
                cbo_operating_cash_target = float(cbo_operating_cash_row['operating_cash_target_bil'])
                cbo_cash_residual_status = 'operating_cash_target_loaded'
            else:
                cbo_operating_cash_target = projected_tga_pre_issuance
                cbo_cash_residual_status = 'operating_cash_path_not_configured'
            if isinstance(cbo_inputs.get('cash_residual'), pd.DataFrame) and not cbo_inputs['cash_residual'].empty:
                cbo_cash_residual_row = _cbo_period_row(
                    cbo_inputs['cash_residual'],
                    cbo_scenario_id,
                    prev_date,
                    current_date,
                    'CBO cash reconciliation residual',
                )
                cbo_cash_reconciliation_residual_input = float(cbo_cash_residual_row['cash_reconciliation_residual_bil'])
                cbo_cash_residual_period = cbo_cash_reconciliation_residual_input
            cbo_fed_row = _cbo_optional_exact_date_row(
                cbo_inputs.get('fed_holdings'),
                cbo_scenario_id,
                current_date,
                'period_end',
                'CBO Fed holdings path',
            )
            if cbo_fed_row is not None:
                cbo_fed_allocation_override_active = True
                cbo_fed_holdings_target = float(cbo_fed_row['cbo_fed_holdings_target_bil'])
                results.loc[current_date, 'CBOFedHoldingsTargetApplicable'] = 1.0
                active_for_cb_target = bond_portfolio[bond_portfolio['Status'] == 'Active'].copy()
                if not active_for_cb_target.empty:
                    active_for_cb_target['DebtBase'] = np.where(
                        active_for_cb_target['SecurityType'] == 'TIPS',
                        active_for_cb_target['AdjustedPrincipal'].fillna(active_for_cb_target['FaceValue']),
                        active_for_cb_target['FaceValue'],
                    )
                    current_cb_holdings = active_for_cb_target.loc[
                        active_for_cb_target['HolderType'] == 'CB',
                        'DebtBase',
                    ].sum()
                else:
                    current_cb_holdings = 0.0
            results.loc[current_date, 'CBORequiredFaceIssuance'] = total_issuance_target_period
            results.loc[current_date, 'CBOBuybackFaceRetired'] = cbo_buyback_face_retired
            results.loc[current_date, 'CBOBuybackCashPaid'] = cbo_buyback_cash_paid
            results.loc[current_date, 'CBOFedHoldingsTarget'] = cbo_fed_holdings_target
            results.loc[current_date, 'CBOFedAuctionShare'] = cbo_fed_auction_share
            if cbo_fed_allocation_override_active:
                results.loc[current_date, 'CBOFedStockMode'] = (
                    'synthetic_cb_treasury_stock_target_par_reallocation'
                )
                results.loc[current_date, 'CBOFedSettlementScope'] = (
                    'stock_reallocation_only_no_reserve_deposit_or_market_price_claim'
                )
            results.loc[current_date, 'CBOCashResidualStatus'] = cbo_cash_residual_status
        new_bonds_added_list = []
        total_issued_face_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        total_issued_proceeds_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        total_issued_proceeds_by_private_route = _zero_private_routes()
        actual_issued_amount = 0.0
        actual_auction_proceeds = 0.0
        issue_discount_cost_period = 0.0
        results.loc[current_date, 'IssuanceProceedsTarget'] = 0.0 if cbo_funding_mode else total_issuance_target_period
        if total_issuance_target_period > TGA_FLOOR_TOLERANCE:
            effective_auction_prefs_for_period = holder_preferences_for_period(
                holder_absorption_lookup,
                scenario_name=scenario_name,
                current_date=current_date,
                fallback_preferences=current_auction_prefs,
                warning_cache=holder_absorption_warning_cache,
            )
            tips_pct = issuance_profile.get('TIPS', {}).get('target_percentage', 0.0)
            frn_pct = issuance_profile.get('FRN', {}).get('target_percentage', 0.0)
            nonmkt_pct_profile = issuance_profile.get('NonMarketable', {}).get('target_percentage', 0.0)
            if cbo_funding_mode:
                nonmkt_pct_profile = 0.0
            issued_tips = total_issuance_target_period * tips_pct
            issued_frn = total_issuance_target_period * frn_pct
            issued_nonmkt = total_issuance_target_period * nonmkt_pct_profile
            marketable_fixed_rate_issuance = max(0, total_issuance_target_period - issued_tips - issued_frn - issued_nonmkt)
            issuance_supply_schedule = []
            tips_real_curve_years = []
            tips_real_curve_rates = []
            if cbo_funding_mode and isinstance(cbo_inputs, dict):
                tips_real_curve_years, tips_real_curve_rates, _tips_real_curve_status = _tips_real_curve_for_date(
                    cbo_inputs.get('tips_real_yield_path'),
                    current_date,
                    cbo_scenario_id,
                )
            tips_profile = issuance_profile.get('TIPS', {})
            if issued_tips > TGA_FLOOR_TOLERANCE and tips_profile.get('maturities'):
                dist = tips_profile.get('maturity_distribution', [])
                mats = tips_profile.get('maturities', [])
                dist_sum = sum(dist)
                norm_dist = [d / dist_sum for d in dist] if dist_sum > TGA_FLOOR_TOLERANCE else []
                if len(norm_dist) == len(mats):
                    for maturity_yrs, weight in zip(mats, norm_dist):
                        proceeds_target = issued_tips * weight
                        if proceeds_target > TGA_FLOOR_TOLERANCE:
                            if tips_real_curve_years and tips_real_curve_rates:
                                yld = get_yield_for_maturity(
                                    maturity_yrs,
                                    tips_real_curve_years,
                                    tips_real_curve_rates,
                                    method=yield_interpolation_method,
                                    floor_zero=False,
                                )
                            else:
                                yld = get_yield_for_maturity(
                                    maturity_yrs,
                                    current_yield_curve_years,
                                    current_yield_curve_rates,
                                    method=yield_interpolation_method,
                                    floor_zero=yield_floor_zero,
                                )
                            real_coupon = calculate_coupon_rate('TIPS', maturity_yrs, yld, tips_real_coupon)
                            face_amount, proceeds_amount, issue_price_ratio = _quote_issuance('TIPS', maturity_yrs, real_coupon, yld, proceeds_target, cbo_face_target_mode)
                            issuance_supply_schedule.append({'type': 'TIPS', 'maturity': maturity_yrs, 'face_amount': face_amount, 'proceeds': proceeds_amount, 'coupon': real_coupon, 'issue_price_ratio': issue_price_ratio, 'issue_yield': yld})
            if issued_frn > TGA_FLOOR_TOLERANCE:
                frn_face, frn_proceeds, frn_issue_price_ratio = _quote_issuance('FRN', 2.0, 0.0, np.nan, issued_frn, cbo_face_target_mode)
                issuance_supply_schedule.append({'type': 'FRN', 'maturity': 2.0, 'face_amount': frn_face, 'proceeds': frn_proceeds, 'spread': frn_spread, 'issue_price_ratio': frn_issue_price_ratio, 'issue_yield': np.nan})
            fixed_remainder_alloc = {}
            total_fixed_target_allocation = 0.0
            for cat_name in MATURITY_CATEGORIES:
                cat_profile = issuance_profile.get(cat_name, {})
                cat_pct_remain = cat_profile.get('target_percentage_of_remainder', 0.0)
                cat_amount_target = marketable_fixed_rate_issuance * cat_pct_remain
                maturities = cat_profile.get('maturities', [])
                distribution = cat_profile.get('maturity_distribution', [])
                if cat_amount_target > TGA_FLOOR_TOLERANCE and maturities and distribution:
                    dist_sum = sum(distribution)
                    norm_dist = [d / dist_sum for d in distribution] if dist_sum > TGA_FLOOR_TOLERANCE else []
                    if len(norm_dist) == len(maturities):
                        total_fixed_target_allocation += cat_amount_target
                        for maturity_yrs, weight in zip(maturities, norm_dist):
                            proceeds_target = cat_amount_target * weight
                            if proceeds_target > TGA_FLOOR_TOLERANCE:
                                fixed_remainder_alloc[maturity_yrs] = fixed_remainder_alloc.get(maturity_yrs, 0.0) + proceeds_target
            fixed_unallocated = marketable_fixed_rate_issuance - sum(fixed_remainder_alloc.values())
            if abs(fixed_unallocated) > TGA_FLOOR_TOLERANCE:
                rem_mat_default = issuance_profile.get('remainder_maturity_years', None)
                if rem_mat_default is None:
                    bill_mats = issuance_profile.get('bills', {}).get('maturities', [0.25])
                    rem_mat_default = min(bill_mats) if bill_mats else 0.25
                fixed_remainder_alloc[rem_mat_default] = fixed_remainder_alloc.get(rem_mat_default, 0.0) + fixed_unallocated
            for maturity_yrs, proceeds_target in fixed_remainder_alloc.items():
                if proceeds_target > TGA_FLOOR_TOLERANCE:
                    yld = get_yield_for_maturity(
                        maturity_yrs,
                        current_yield_curve_years,
                        current_yield_curve_rates,
                        method=yield_interpolation_method,
                        floor_zero=yield_floor_zero,
                    )
                    coupon = calculate_coupon_rate('Fixed', maturity_yrs, yld, 0)
                    face_amount, proceeds_amount, issue_price_ratio = _quote_issuance('Fixed', maturity_yrs, coupon, yld, proceeds_target, cbo_face_target_mode)
                    issuance_supply_schedule.append({'type': 'Fixed', 'maturity': maturity_yrs, 'face_amount': face_amount, 'proceeds': proceeds_amount, 'coupon': coupon, 'issue_price_ratio': issue_price_ratio, 'issue_yield': yld})
            if issued_nonmkt > TGA_FLOOR_TOLERANCE:
                nm_issuance_profile_details = issuance_profile.get('NonMarketable', {})
                nm_maturity_years = nm_issuance_profile_details.get('nominal_maturity_years', 30.0)
                nm_face, nm_proceeds, nm_issue_price_ratio = _quote_issuance('NonMarketable', nm_maturity_years, 0.0, np.nan, issued_nonmkt, cbo_face_target_mode)
                issuance_supply_schedule.append({'type': 'NonMarketable', 'maturity': nm_maturity_years, 'face_amount': nm_face, 'proceeds': nm_proceeds, 'coupon': 0.0, 'issue_price_ratio': nm_issue_price_ratio, 'issue_yield': np.nan})
            allocations_by_holder_item_ref = defaultdict(lambda: defaultdict(float))
            item_details_by_item_ref = {}
            next_item_ref_id = 0
            for supply_item in issuance_supply_schedule:
                item_type = supply_item['type']
                item_maturity = supply_item['maturity']
                item_face_amount = supply_item['face_amount']
                if item_face_amount < TGA_FLOOR_TOLERANCE:
                    continue
                current_item_ref_id = next_item_ref_id
                item_details_by_item_ref[current_item_ref_id] = supply_item
                next_item_ref_id += 1
                pref_category_key = get_security_category_for_prefs(item_type, item_maturity, issuance_profile)
                if pref_category_key is None:
                    raise ValueError(f'[{scenario_name}@{current_date.date()}] No preference category key for {item_type} maturity {item_maturity}.')
                total_desired_value = 0.0
                sector_desired_value = {}
                item_yield = supply_item.get('issue_yield')
                if pd.isna(item_yield):
                    item_yield = get_yield_for_maturity(
                        item_maturity,
                        current_yield_curve_years,
                        current_yield_curve_rates,
                        method=yield_interpolation_method,
                        floor_zero=yield_floor_zero,
                    )
                item_yield = 0.0 if pd.isna(item_yield) else float(item_yield)
                anchor_yield, curve_slope = _get_curve_reference_levels(
                    rate_sensitive_p,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    interpolation_method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                )
                holder_share_weights = {}
                for holder in HOLDER_TYPES:
                    holder_prefs = effective_auction_prefs_for_period.get(holder, {})
                    base_pref_pct = float(holder_prefs.get(f'{pref_category_key}_pct', 0.0) or 0.0)
                    multiplier = _compute_rate_sensitive_multiplier(
                        rate_sensitive_p, 'auction', holder, pref_category_key, item_yield, anchor_yield, curve_slope
                    )
                    holder_share_weights[holder] = max(0.0, base_pref_pct * multiplier)
                total_weight = sum(holder_share_weights.values())
                if total_weight > TGA_FLOOR_TOLERANCE:
                    effective_holder_prefs = {
                        holder: {f'{pref_category_key}_pct': adjusted_weight / total_weight}
                        for holder, adjusted_weight in holder_share_weights.items()
                    }
                    base_holder_prefs = {
                        holder: {f'{pref_category_key}_pct': float(current_auction_prefs.get(holder, {}).get(f'{pref_category_key}_pct', 0.0) or 0.0)}
                        for holder in HOLDER_TYPES
                    }
                    item_shift_avg, item_shift_max = _compute_preference_shift_summary(
                        base_holder_prefs, effective_holder_prefs, [pref_category_key]
                    )
                    auction_shift_weighted_sum += item_shift_avg * item_face_amount
                    auction_shift_weighted_max = max(auction_shift_weighted_max, item_shift_max)
                    auction_shift_weight_total += item_face_amount
                    for holder, adjusted_weight in holder_share_weights.items():
                        desired_h = item_face_amount * adjusted_weight / total_weight
                        if desired_h > TGA_FLOOR_TOLERANCE:
                            sector_desired_value[holder] = desired_h
                            total_desired_value += desired_h
                if total_desired_value < TGA_FLOOR_TOLERANCE:
                    allocated_holder_fallback = 'Private'
                    if item_type == 'NonMarketable':
                        fallback_nm_holder = current_nonmkt_params.get('initial_holder', 'Private')
                        allocated_holder_fallback = fallback_nm_holder
                    allocations_by_holder_item_ref[allocated_holder_fallback][current_item_ref_id] += item_face_amount
                else:
                    rationing_factor = min(1.0, item_face_amount / total_desired_value)
                    for holder, desired_amount in sector_desired_value.items():
                        allocated_amount_h = desired_amount * rationing_factor
                        if allocated_amount_h > TGA_FLOOR_TOLERANCE:
                            allocations_by_holder_item_ref[holder][current_item_ref_id] += allocated_amount_h
                    allocated_sum = sum((allocations_by_holder_item_ref[h].get(current_item_ref_id, 0.0) for h in HOLDER_TYPES))
                    if abs(allocated_sum - item_face_amount) > 1e-06:
                        raise ValueError(f'[{scenario_name}@{current_date.date()}] Issuance allocation under- or over-shot face supply for {item_type} {item_maturity}Y. Allocated {allocated_sum:.6f} vs required {item_face_amount:.6f}.')
            next_bond_id_to_assign = bond_id_counter
            for holder, items_allocated_to_holder in allocations_by_holder_item_ref.items():
                for item_ref, face_value_issued_to_holder in items_allocated_to_holder.items():
                    if face_value_issued_to_holder < TGA_FLOOR_TOLERANCE:
                        continue
                    details = item_details_by_item_ref[item_ref]
                    sec_type = details['type']
                    mat_yrs = details['maturity']
                    route_face_values = [('', face_value_issued_to_holder)]
                    if holder == 'Private':
                        pref_category_for_split = get_security_category_for_prefs(
                            sec_type, mat_yrs, issuance_profile
                        )
                        split = _private_subbucket_split(
                            effective_auction_prefs_for_period,
                            pref_category_for_split,
                        )
                        route_face_values = [
                            (route, face_value_issued_to_holder * share)
                            for route, share in split.items()
                        ]
                    for holder_subbucket, route_face_value in route_face_values:
                        if route_face_value < TGA_FLOOR_TOLERANCE:
                            continue
                        maturity_date = current_date + relativedelta(years=int(round(mat_yrs)), months=int(round(mat_yrs % 1 * 12)))
                        coupon_rate_val = details.get('coupon', 0.0)
                        fixed_spread_val = details.get('spread', 0.0) if sec_type == 'FRN' else 0.0
                        issue_price_ratio_val = details.get('issue_price_ratio', 1.0)
                        issue_yield_val = details.get('issue_yield', np.nan)
                        issue_proceeds_val = route_face_value * issue_price_ratio_val
                        interest_payment_frequency_val = 4.0 if sec_type == 'FRN' else (2.0 if sec_type in {'Fixed', 'TIPS'} and coupon_rate_val > TGA_FLOOR_TOLERANCE else np.nan)
                        first_interest_payment_date_val = _first_interest_payment_date(
                            current_date,
                            maturity_date,
                            interest_payment_frequency_val,
                        )
                        original_principal_val = 0.0
                        reference_cpi_issue_val = 0.0
                        index_ratio_init_val = 0.0
                        last_accrual_date_val = pd.NaT
                        if sec_type == 'TIPS':
                            original_principal_val = route_face_value
                            reference_cpi_issue_val = results.loc[current_date, 'Reference_CPI']
                            index_ratio_init_val = 1.0
                        elif sec_type == 'FRN':
                            last_accrual_date_val = current_date
                        maturity_category_val = None
                        if sec_type == 'Fixed':
                            maturity_category_val = get_maturity_category(mat_yrs, issuance_profile)
                        new_tranche = {'BondID': next_bond_id_to_assign, 'FaceValue': route_face_value, 'HolderType': holder, 'HolderSubBucket': holder_subbucket if holder == 'Private' else '', 'SecurityType': sec_type, 'IssueDate': current_date, 'MaturityDate': maturity_date, 'DatedDate': current_date, 'OriginalDatedDate': current_date, 'FirstInterestPaymentDate': first_interest_payment_date_val, 'InterestPaymentFrequency': interest_payment_frequency_val, 'OriginalMaturityYears': mat_yrs, 'CouponRate': coupon_rate_val, 'Status': 'Active', 'MaturityCategory': maturity_category_val, 'OriginalPrincipal': original_principal_val, 'AdjustedPrincipal': original_principal_val, 'ReferenceCPI_Issue': reference_cpi_issue_val, 'IndexRatio': index_ratio_init_val, 'FixedSpread': fixed_spread_val, 'AccruedInterest_FRN': 0.0, 'BenchmarkRate_FRN': 0.0, 'LastAccrualDate': last_accrual_date_val, 'IssuePriceRatio': issue_price_ratio_val, 'IssueProceeds': issue_proceeds_val, 'IssueYieldAtIssue': issue_yield_val, 'TimeToMaturity': np.nan, 'DiscountYield': np.nan, 'CleanPrice': np.nan, 'AccruedInterest': np.nan, 'DirtyValue': np.nan, 'DirtyPriceRatio': np.nan}
                        new_bonds_added_list.append(new_tranche)
                        total_issued_face_by_holder[holder] += route_face_value
                        total_issued_proceeds_by_holder[holder] += issue_proceeds_val
                        if holder == 'Private':
                            total_issued_proceeds_by_private_route[holder_subbucket] += issue_proceeds_val
                        actual_issued_amount += route_face_value
                        actual_auction_proceeds += issue_proceeds_val
                        issue_discount_cost_period += max(0.0, route_face_value - issue_proceeds_val)
                        next_bond_id_to_assign += 1
            bond_id_counter = next_bond_id_to_assign
            if new_bonds_added_list:
                new_bonds_df = pd.DataFrame(new_bonds_added_list)
                for col in BOND_PORTFOLIO_COLS:
                    if col not in new_bonds_df.columns:
                        dtype = PORTFOLIO_DTYPES.get(col, 'object')
                        new_bonds_df[col] = pd.NA if dtype == 'Int64' else pd.NaT if dtype == 'datetime64[ns]' else np.nan
                new_bonds_df = new_bonds_df[BOND_PORTFOLIO_COLS]
                new_bonds_df = new_bonds_df.astype(PORTFOLIO_DTYPES, errors='ignore')
                try:
                    if bond_portfolio.empty:
                        bond_portfolio = new_bonds_df
                    else:
                        bond_portfolio = pd.concat([bond_portfolio, new_bonds_df], ignore_index=True)
                except Exception as concat_err:
                    print(f'ERROR [{scenario_name}@{current_date.date()}]: Concatenation failed: {concat_err}')
        results.loc[current_date, 'NewDebtIssued'] = actual_issued_amount
        results.loc[current_date, 'AuctionProceeds'] = actual_auction_proceeds
        if cbo_funding_mode:
            results.loc[current_date, 'IssuanceProceedsTarget'] = actual_auction_proceeds
        results.loc[current_date, 'IssueDiscountCost_Period'] = issue_discount_cost_period
        results.loc[current_date, 'IssueDiscountCost_Cumulative'] = results.loc[prev_date, 'IssueDiscountCost_Cumulative'] + issue_discount_cost_period
        results.loc[current_date, 'FinancingCost_Period'] = results.loc[current_date, 'InterestOutlay_Period'] + issue_discount_cost_period + nonmkt_interest_capitalized_period + tips_inflation_accretion_period
        results.loc[current_date, 'FinancingCost_Cumulative'] = results.loc[prev_date, 'FinancingCost_Cumulative'] + results.loc[current_date, 'FinancingCost_Period']
        issuance_tga_change = actual_auction_proceeds
        issuance_reserve_change = -(total_issued_proceeds_by_holder.get('Banks', 0.0) + total_issued_proceeds_by_holder.get('Private', 0.0) + total_issued_proceeds_by_holder.get('Foreign', 0.0))
        issuance_deposit_change = -_effective_private_amount(
            total_issued_proceeds_by_private_route,
            mmf_deposit_pass_through,
            legacy_total=total_issued_proceeds_by_holder.get('Private', 0.0),
        )
        reserve_change_period += issuance_reserve_change
        deposit_change_period += issuance_deposit_change
        tga_change_period += issuance_tga_change
        if cbo_funding_mode:
            tga_change_period += cbo_cash_residual_period
        if cbo_funding_mode and cbo_fed_allocation_override_active:
            active_for_fed_purchase = bond_portfolio[bond_portfolio['Status'] == 'Active'].copy()
            if not active_for_fed_purchase.empty:
                active_for_fed_purchase['DebtBase'] = np.where(
                    active_for_fed_purchase['SecurityType'] == 'TIPS',
                    active_for_fed_purchase['AdjustedPrincipal'].fillna(active_for_fed_purchase['FaceValue']),
                    active_for_fed_purchase['FaceValue'],
                )
                current_cb_after_issuance = active_for_fed_purchase.loc[
                    active_for_fed_purchase['HolderType'] == 'CB',
                    'DebtBase',
                ].sum()
            else:
                current_cb_after_issuance = 0.0
            fed_purchase_shortfall = cbo_fed_holdings_target - current_cb_after_issuance
            fed_target_tolerance = float(funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001)
            if fed_purchase_shortfall > TGA_FLOOR_TOLERANCE:
                (
                    bond_portfolio,
                    fed_secondary_purchase_face,
                    fed_secondary_sellers,
                    fed_secondary_private_routes,
                ) = _transfer_public_marketable_debt_to_cb(bond_portfolio, fed_purchase_shortfall)
                if abs(fed_secondary_purchase_face - fed_purchase_shortfall) > float(
                    funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001
                ):
                    raise RuntimeError(
                        f"CBO Fed holdings target cannot be met by secondary purchases at {current_date.date()}: "
                        f"target={cbo_fed_holdings_target:.12f}, current={current_cb_after_issuance:.12f}, "
                        f"purchased={fed_secondary_purchase_face:.12f}"
                    )
                fed_secondary_purchase_cash = 0.0
                fed_secondary_private_deposit = 0.0
                results.loc[current_date, 'CBOFedSecondaryPurchaseReserveEffect'] = 0.0
                results.loc[current_date, 'CBOFedSecondaryPurchaseDepositEffect'] = 0.0
                results.loc[current_date, 'CBOFedSecondaryPurchaseFace'] = fed_secondary_purchase_face
                results.loc[current_date, 'CBOFedSecondaryPurchaseCash'] = fed_secondary_purchase_cash
            elif fed_purchase_shortfall < -fed_target_tolerance:
                bond_portfolio, fed_secondary_sale_face = _transfer_cb_marketable_debt_to_public(
                    bond_portfolio,
                    abs(fed_purchase_shortfall),
                )
                if abs(fed_secondary_sale_face - abs(fed_purchase_shortfall)) > fed_target_tolerance:
                    raise RuntimeError(
                        f"CBO Fed holdings target cannot be met by synthetic sales at {current_date.date()}: "
                        f"target={cbo_fed_holdings_target:.12f}, current={current_cb_after_issuance:.12f}, "
                        f"sold={fed_secondary_sale_face:.12f}"
                    )
        if current_enable_trading and (not cbo_fed_allocation_override_active) and (not bond_portfolio.empty):
            try:
                pref_trade_monetary_impact = {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0}
                bond_portfolio_temp = bond_portfolio.reset_index(drop=True)
                effective_secondary_prefs = _build_dynamic_secondary_preferences(
                    current_secondary_prefs,
                    rate_sensitive_p,
                    issuance_profile,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    frn_benchmark_mat,
                    interpolation_method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                )
                secondary_shift_avg, secondary_shift_max = _compute_preference_shift_summary(
                    current_secondary_prefs,
                    effective_secondary_prefs,
                    _get_active_preference_categories(issuance_profile),
                )
                bond_portfolio_temp, pref_trade_monetary_impact = execute_preference_trades(bond_portfolio_temp, current_date, current_yield_curve_years, current_yield_curve_rates, effective_secondary_prefs, issuance_profile, scenario_name)
                bond_portfolio = bond_portfolio_temp
                secondary_domestic_nonbank_change = pref_trade_monetary_impact.get(
                    'deposit_change_private_deposit_funded',
                    pref_trade_monetary_impact.get('deposit_change', 0.0),
                )
                secondary_mmf_gross_change = pref_trade_monetary_impact.get('deposit_change_private_mmf', 0.0)
                secondary_effective_deposit_change = (
                    pref_trade_monetary_impact.get('deposit_change', 0.0)
                    if mmf_deposit_pass_through >= 1.0
                    else (
                        secondary_domestic_nonbank_change
                        + mmf_deposit_pass_through * secondary_mmf_gross_change
                    )
                )
                pref_trade_monetary_impact['deposit_change_gross_private_legacy'] = pref_trade_monetary_impact.get('deposit_change', 0.0)
                pref_trade_monetary_impact['deposit_change'] = secondary_effective_deposit_change
                pref_trade_monetary_impact['deposit_change_private_mmf_plumbing'] = (
                    (1.0 - mmf_deposit_pass_through) * secondary_mmf_gross_change
                )
                reserve_change_period += pref_trade_monetary_impact.get('reserve_change', 0.0)
                deposit_change_period += pref_trade_monetary_impact.get('deposit_change', 0.0)
                tga_change_period += pref_trade_monetary_impact.get('tga_change', 0.0)
                tga_change_period -= pref_trade_monetary_impact.get('tga_drain', 0.0)
            except Exception as e:
                print(f'ERROR [{scenario_name}@{current_date.date()}]: Preference trading failed: {e}')
                traceback.print_exc()
                pref_trade_monetary_impact = {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0}
        active_bonds_final = bond_portfolio[bond_portfolio['Status'] == 'Active'].copy()
        if not active_bonds_final.empty:
            active_bonds_final['DebtBase'] = np.where(active_bonds_final['SecurityType'] == 'TIPS', active_bonds_final['AdjustedPrincipal'].fillna(active_bonds_final['FaceValue']), active_bonds_final['FaceValue'])
        current_total_debt = active_bonds_final['DebtBase'].sum() if not active_bonds_final.empty else 0.0
        results.loc[current_date, 'TotalDebt_Agg'] = current_total_debt
        if cbo_funding_mode:
            cbo_post_issuance_controlled_debt = _controlled_public_marketable_debt_from_portfolio(active_bonds_final)
            cbo_target_error = cbo_post_issuance_controlled_debt - cbo_controlled_debt_target
            target_tolerance = float(funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001)
            if abs(cbo_target_error) > target_tolerance:
                raise RuntimeError(
                    f"CBO controlled debt target miss at {current_date.date()}: "
                    f"target={cbo_controlled_debt_target:.12f}, "
                    f"post={cbo_post_issuance_controlled_debt:.12f}, "
                    f"error={cbo_target_error:.12f}"
                )
            results.loc[current_date, 'CBOFundingModeActive'] = 1.0
            results.loc[current_date, 'CBOControlledDebtTarget'] = cbo_controlled_debt_target
            results.loc[current_date, 'CBOControlledDebtPreIssuance'] = cbo_pre_issuance_controlled_debt
            results.loc[current_date, 'CBOControlledDebtPostIssuance'] = cbo_post_issuance_controlled_debt
            results.loc[current_date, 'CBOControlledDebtTargetError'] = cbo_target_error
            results.loc[current_date, 'CBOOperatingCashTarget'] = cbo_operating_cash_target
            results.loc[current_date, 'CBOCashReconciliationResidual'] = cbo_cash_reconciliation_residual_input
            results.loc[current_date, 'CBOFiscalIncidencePolicyPresent'] = 1.0 if cbo_incidence is not None else 0.0
            cbo_baseline_row = _cbo_fiscal_baseline_row(cbo_inputs, cbo_scenario_id, current_date)
            if cbo_baseline_row is not None:
                results.loc[current_date, 'CBONetInterestDiagnostic'] = _cbo_value(
                    cbo_baseline_row,
                    'cbo_net_interest_bil',
                )
                results.loc[current_date, 'CBOTotalDeficitDiagnostic'] = _cbo_value(
                    cbo_baseline_row,
                    'cbo_total_deficit_bil',
                )
                results.loc[current_date, 'NetInterestDiagnosticStatus'] = 'cbo_reported_check_only'
            else:
                results.loc[current_date, 'NetInterestDiagnosticStatus'] = 'not_loaded_check_only'
            results.loc[current_date, 'CBONetInterestBridgeRows'] = (
                float(len(cbo_inputs['net_interest_bridge']))
                if isinstance(cbo_inputs.get('net_interest_bridge'), pd.DataFrame)
                else 0.0
            )
        for holder in HOLDER_TYPES:
            results.loc[current_date, f'DebtHeld_{holder}'] = active_bonds_final.loc[active_bonds_final['HolderType'] == holder, 'DebtBase'].sum() if not active_bonds_final.empty else 0.0
        if cbo_funding_mode and cbo_fed_allocation_override_active:
            results.loc[current_date, 'CBOFedHoldingsTargetError'] = (
                results.loc[current_date, 'DebtHeld_CB'] - cbo_fed_holdings_target
            )
            fed_target_tolerance = float(funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001)
            if abs(results.loc[current_date, 'CBOFedHoldingsTargetError']) > fed_target_tolerance:
                raise RuntimeError(
                    f"CBO Fed holdings target miss at {current_date.date()}: "
                    f"target={cbo_fed_holdings_target:.12f}, "
                    f"post={results.loc[current_date, 'DebtHeld_CB']:.12f}, "
                    f"error={results.loc[current_date, 'CBOFedHoldingsTargetError']:.12f}"
                )
            cbo_fed_end_stock = float(results.loc[current_date, 'DebtHeld_CB'])
            cbo_fed_maturities = float(principal_paid_by_holder.get('CB', 0.0))
            cbo_fed_auction_addons = float(total_issued_face_by_holder.get('CB', 0.0))
            results.loc[current_date, 'CBOFedBeginStock'] = cbo_fed_begin_stock
            results.loc[current_date, 'CBOFedMaturitiesAndRedemptions'] = cbo_fed_maturities
            results.loc[current_date, 'CBOFedTipsPrincipalIndexation'] = cb_tips_indexation_period
            results.loc[current_date, 'CBOFedAuctionRolloverAddons'] = cbo_fed_auction_addons
            results.loc[current_date, 'CBOFedSyntheticSecondaryPurchases'] = fed_secondary_purchase_face
            results.loc[current_date, 'CBOFedSyntheticSecondarySales'] = fed_secondary_sale_face
            results.loc[current_date, 'CBOFedEndStock'] = cbo_fed_end_stock
            results.loc[current_date, 'CBOFedNetStockChange'] = cbo_fed_end_stock - cbo_fed_begin_stock
            results.loc[current_date, 'CBOFedGrossStockFlow'] = (
                abs(cbo_fed_maturities)
                + abs(cb_tips_indexation_period)
                + abs(cbo_fed_auction_addons)
                + abs(fed_secondary_purchase_face)
                + abs(fed_secondary_sale_face)
            )
        for sec_type in SECURITY_TYPES:
            results.loc[current_date, f'DebtHeldByType_{sec_type}'] = active_bonds_final.loc[active_bonds_final['SecurityType'] == sec_type, 'DebtBase'].sum() if not active_bonds_final.empty else 0.0
        if not active_bonds_final.empty:
            active_bonds_final['TimeToMaturity_WAM'] = (active_bonds_final['MaturityDate'] - current_date).dt.total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
            active_bonds_final['TimeToMaturity_WAM'] = active_bonds_final['TimeToMaturity_WAM'].clip(lower=0.0)
            wam_numerator = (active_bonds_final['DebtBase'] * active_bonds_final['TimeToMaturity_WAM']).sum()
            wam_denominator = current_total_debt
            current_wam = wam_numerator / wam_denominator if wam_denominator > TGA_FLOOR_TOLERANCE else 0.0
        else:
            current_wam = 0.0
        results.loc[current_date, 'WAM'] = current_wam
        other_reserve_change = -reserve_transfer
        other_deposit_change = 0.0
        other_tga_change = reserve_transfer + money_minting_transfers
        reserve_change_period += other_reserve_change
        deposit_change_period += other_deposit_change
        tga_change_period += other_tga_change
        prev_reserves = results.loc[prev_date, 'Reserves']
        prev_deposits = results.loc[prev_date, 'TDC_Level']
        prev_tga = results.loc[prev_date, 'TGA']
        projected_tga = prev_tga + tga_change_period
        tga_discrepancy = tga_floor - projected_tga
        if (not cbo_funding_mode) and tga_discrepancy > TGA_FLOOR_TOLERANCE:
            raise RuntimeError(
                f"TGA floor would bind without explicit financing at {current_date.date()}: "
                f"shortfall={tga_discrepancy:.12f}"
            )
        current_tga = projected_tga
        if cbo_funding_mode:
            results.loc[current_date, 'CBOCashResidual'] = current_tga - cbo_operating_cash_target
        adjusted_reserve_change_period = reserve_change_period
        adjusted_deposit_change_period = deposit_change_period
        current_reserves = prev_reserves + adjusted_reserve_change_period
        current_deposits = prev_deposits + adjusted_deposit_change_period
        results.loc[current_date, 'Reserves'] = current_reserves
        results.loc[current_date, 'TDC_Level'] = current_deposits
        results.loc[current_date, 'TGA'] = current_tga
        results.loc[current_date, 'ReserveChange'] = adjusted_reserve_change_period
        results.loc[current_date, 'TDC_Change'] = adjusted_deposit_change_period
        results.loc[current_date, 'TGAChange'] = current_tga - prev_tga
        results.loc[current_date, 'TDC_FiscalFlow'] = fiscal_deposit_change
        results.loc[current_date, 'TDC_DebtService'] = debt_service_deposit_change
        results.loc[current_date, 'TDC_AuctionAbsorption'] = issuance_deposit_change
        results.loc[current_date, 'TDC_SecondaryTrades'] = pref_trade_monetary_impact.get('deposit_change', 0.0)
        results.loc[current_date, 'TDC_Other'] = other_deposit_change
        private_secondary_change = pref_trade_monetary_impact.get('deposit_change', 0.0)
        results.loc[current_date, 'TDC_AuctionAbsorption_DomesticNonbank'] = -_private_route_value(
            total_issued_proceeds_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_AuctionAbsorption_MMF'] = -mmf_deposit_pass_through * _private_route_value(
            total_issued_proceeds_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_AuctionAbsorption_MMFPlumbing'] = -(
            (1.0 - mmf_deposit_pass_through)
            * _private_route_value(total_issued_proceeds_by_private_route, PRIVATE_SUBBUCKET_MMF)
        )
        results.loc[current_date, 'TDC_PrincipalToDU_DomesticNonbank'] = _private_route_value(
            principal_component_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_PrincipalToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            principal_component_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_PrincipalToDU_MMFPlumbing'] = (
            (1.0 - mmf_deposit_pass_through)
            * _private_route_value(principal_component_paid_by_private_route, PRIVATE_SUBBUCKET_MMF)
        )
        results.loc[current_date, 'TDC_BillDiscountInterestToDU_DomesticNonbank'] = _private_route_value(
            bill_discount_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_BillDiscountInterestToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            bill_discount_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_CouponInterestToDU_DomesticNonbank'] = _private_route_value(
            coupon_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_CouponInterestToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            coupon_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_FRNInterestToDU_DomesticNonbank'] = _private_route_value(
            frn_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_FRNInterestToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            frn_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_TIPSCouponInterestToDU_DomesticNonbank'] = _private_route_value(
            tips_coupon_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_TIPSCouponInterestToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            tips_coupon_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_TIPSInflationCompensationToDU_DomesticNonbank'] = _private_route_value(
            tips_inflation_compensation_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_TIPSInflationCompensationToDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            tips_inflation_compensation_paid_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_DebtService_MMFPlumbing'] = (
            (1.0 - mmf_deposit_pass_through)
            * _private_route_value(
                {
                    route: principal_paid_by_private_route[route] + interest_paid_by_private_route[route]
                    for route in PRIVATE_SUBBUCKETS
                },
                PRIVATE_SUBBUCKET_MMF,
            )
        )
        results.loc[current_date, 'TDC_PrincipalToDU'] = (
            results.loc[current_date, 'TDC_PrincipalToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_PrincipalToDU_MMF']
        )
        results.loc[current_date, 'TDC_BillDiscountInterestToDU'] = (
            results.loc[current_date, 'TDC_BillDiscountInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_BillDiscountInterestToDU_MMF']
        )
        results.loc[current_date, 'TDC_CouponInterestToDU'] = (
            results.loc[current_date, 'TDC_CouponInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_CouponInterestToDU_MMF']
        )
        results.loc[current_date, 'TDC_FRNInterestToDU'] = (
            results.loc[current_date, 'TDC_FRNInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_FRNInterestToDU_MMF']
        )
        results.loc[current_date, 'TDC_TIPSCouponInterestToDU'] = (
            results.loc[current_date, 'TDC_TIPSCouponInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_TIPSCouponInterestToDU_MMF']
        )
        results.loc[current_date, 'TDC_TIPSInflationCompensationToDU'] = (
            results.loc[current_date, 'TDC_TIPSInflationCompensationToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_TIPSInflationCompensationToDU_MMF']
        )
        results.loc[current_date, 'TDC_InterestToDU_DomesticNonbank'] = (
            results.loc[current_date, 'TDC_BillDiscountInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_CouponInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_FRNInterestToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_TIPSCouponInterestToDU_DomesticNonbank']
        )
        results.loc[current_date, 'TDC_InterestToDU_MMF'] = (
            results.loc[current_date, 'TDC_BillDiscountInterestToDU_MMF']
            + results.loc[current_date, 'TDC_CouponInterestToDU_MMF']
            + results.loc[current_date, 'TDC_FRNInterestToDU_MMF']
            + results.loc[current_date, 'TDC_TIPSCouponInterestToDU_MMF']
        )
        results.loc[current_date, 'TDC_InterestToDU'] = (
            results.loc[current_date, 'TDC_BillDiscountInterestToDU']
            + results.loc[current_date, 'TDC_CouponInterestToDU']
            + results.loc[current_date, 'TDC_FRNInterestToDU']
            + results.loc[current_date, 'TDC_TIPSCouponInterestToDU']
        )
        results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU_DomesticNonbank'] = _private_route_value(
            total_issued_proceeds_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU_MMF'] = mmf_deposit_pass_through * _private_route_value(
            total_issued_proceeds_by_private_route,
            PRIVATE_SUBBUCKET_MMF,
        )
        results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU'] = (
            results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU_MMF']
        )
        results.loc[current_date, 'TDC_SecondaryTrades_DomesticNonbank'] = pref_trade_monetary_impact.get(
            'deposit_change_private_deposit_funded',
            0.0,
        )
        results.loc[current_date, 'TDC_SecondaryTrades_MMF'] = (
            mmf_deposit_pass_through
            * pref_trade_monetary_impact.get('deposit_change_private_mmf', 0.0)
        )
        results.loc[current_date, 'TDC_SecondaryTrades_MMFPlumbing'] = pref_trade_monetary_impact.get(
            'deposit_change_private_mmf_plumbing',
            0.0,
        )
        results.loc[current_date, 'TDC_SecondaryDUToRU'] = max(0.0, private_secondary_change)
        results.loc[current_date, 'TDC_SecondaryRUToDU'] = max(0.0, -private_secondary_change)
        results.loc[current_date, 'AuctionDemandShift_AvgAbs'] = (
            auction_shift_weighted_sum / auction_shift_weight_total if auction_shift_weight_total > TGA_FLOOR_TOLERANCE else 0.0
        )
        results.loc[current_date, 'AuctionDemandShift_MaxAbs'] = auction_shift_weighted_max
        results.loc[current_date, 'SecondaryDemandShift_AvgAbs'] = secondary_shift_avg
        results.loc[current_date, 'SecondaryDemandShift_MaxAbs'] = secondary_shift_max
    sim_duration = time.time() - sim_start_time
    print(f'--- Finished Simulation: {scenario_name} ({sim_duration:.2f} seconds) ---')
    final_portfolio_out = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
    if not bond_portfolio.empty:
        try:
            final_portfolio_out = bond_portfolio[BOND_PORTFOLIO_COLS].astype(PORTFOLIO_DTYPES, errors='ignore')
        except Exception:
            final_portfolio_out = bond_portfolio
    results.attrs['run_metadata'] = {
        'scenario_name': scenario_name,
        'start_date': str(start_date),
        'end_date': str(end_date),
        'frequency': freq,
        'validation_status': 'passed',
        'uses_legacy_sector_preferences_for_auction': uses_legacy_sector_prefs_for_auction,
        'uses_legacy_sector_preferences_for_secondary': uses_legacy_sector_prefs_for_secondary,
        'rate_sensitive_demand_enabled': bool(rate_sensitive_p.get('enabled', False)),
        'includes_tips_inflation_accretion': bool(include_tips_inflation_accretion),
        'yield_curve_surface_status': yield_curve_surface_status,
        'ratewall_primary_flow_status': (
            'aggregate_cash_proxy_from_cbo_total_deficit_less_net_interest'
            if primary_flow_path_used_count > 0
            else 'simulation_fiscal_flow_to_du_proxy'
        ),
        'ratewall_primary_flow_loaded_rows': sum(len(rows) for rows in primary_flow_lookup.values()),
        'ratewall_primary_flow_used_periods': primary_flow_path_used_count,
        'funding_rule_mode': CBO_FUNDING_MODE if cbo_funding_mode else str(funding_rule_cfg.get('mode', 'legacy_cash_tga')),
        'cbo_funding_mode_active': bool(cbo_funding_mode),
        'cbo_net_interest_bridge_rows': (
            int(len(cbo_inputs['net_interest_bridge']))
            if cbo_funding_mode and isinstance(cbo_inputs.get('net_interest_bridge'), pd.DataFrame)
            else 0
        ),
        'mmf_deposit_pass_through': mmf_deposit_pass_through,
        'mmf_deposit_pass_through_status': MMF_DEPOSIT_PASS_THROUGH_STATUS,
        'mmf_ru_plumbing_pass_through': 1.0 - mmf_deposit_pass_through,
        'mmf_deposit_pass_through_sensitivity_grid': list(MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID),
        'coarse_frequency_warning': coarse_frequency_warning,
        'auction_demand_shift_avgabs_mean': float(results['AuctionDemandShift_AvgAbs'].iloc[1:].mean()) if len(results.index) > 1 else 0.0,
        'secondary_demand_shift_avgabs_mean': float(results['SecondaryDemandShift_AvgAbs'].iloc[1:].mean()) if len(results.index) > 1 else 0.0,
    }
    results.rename(columns=OUTPUT_COLUMN_RENAMES, inplace=True)
    return (results, final_portfolio_out)


__all__ = ['run_simulation']
