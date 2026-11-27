
"""Core simulation loop for the Treasury funding-chain simulator."""

import bisect
import copy
import functools
import json
import time
import traceback
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd

from mmf_eligibility import mmf_may_acquire
from dateutil.relativedelta import relativedelta

from bill_quote_basis import CBO_3M_BILL_DAYS, discount_price_ratio
from evaluated_nominal_curve import CurveContractError, EvaluatedNominalShock
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    DAYS_PER_YEAR_ACTUAL,
    FRN_DAY_COUNT_BASIS,
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    JOURNAL_MIN_EVENT_BIL,
    MATURITY_CATEGORIES,
    MMF_ADDED_TDC_COLUMNS,
    MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
    MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
    MMF_DEPOSIT_PASS_THROUGH_STATUS,
    PREFERENCE_CATEGORIES,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
    PRIVATE_SUBBUCKET_MMF_ADDED,
    PRIVATE_SUBBUCKETS,
    PRIVATE_SUBBUCKET_SHARES_KEY,
    PUBLIC_MARKETABLE_SECURITY_TYPES,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
    TRANSFER_SCALABLE_COLUMNS,
)
import holder_targets
import fed_purchase_policy
from issuance_calibration import (
    CASH_MANAGEMENT_BILL_KIND,
    CASH_MANAGEMENT_BILL_WEEKS,
    active_special_categories,
    bill_days,
    bill_issuance_amount,
    bill_maturity,
    bill_mask,
    bill_maturity_years,
    bill_supply_key,
    coupon_supply_key,
    controlled_debt_and_bills,
    controlled_marketable_mask,
    debt_base,
    is_rule_profile,
    rule_target_path,
    dated_date,
    fiscal_year_close_dates,
    maturity_from_dated,
    settlement_events,
    settlement_rules,
    target_for_date,
)
from sim_helpers import OUTPUT_COLUMN_RENAMES, apply_event_actions, validate_run_params
from sim_pricing import (
    calculate_coupon_rate,
    value_treasury_security,
    calculate_face_from_proceeds_target,
    evaluate_nominal_yield,
    get_coupon_dates_in_period,
    get_maturity_category,
    get_security_category_for_prefs,
    get_yield_for_maturity,
    infer_issue_data_for_loaded_bill,
    quote_issuance_from_face_target,
)
from sim_trading import execute_preference_trades
from settlement_routes import MmfAddedPosition, TRC_RESULT_COLUMNS, TrcLedger, parse_trc_routing, trc_metadata
from tdc_validation import is_coarse_frequency
from tips_indexation import build_projected_cpi_lookup_from_macro, build_tips_daily_cpi_lookups_from_monthly_path
from forecast_paths import (
    CBO_FED_REMITTANCE_OPENING_STATUS,
    CBO_FED_REMITTANCE_STATUS,
    load_cbo_fiscal_baseline,
    load_cash_reconciliation_residual,
    load_debt_stock_path,
    load_fiscal_incidence_policy,
    load_fed_holdings_path,
    load_fed_income_path,
    load_fed_remittance_path,
    load_frn_rate_path,
    load_yield_curve_surface as load_forecast_yield_curve_surface,
    load_macro_forecast_path,
    load_net_interest_bridge,
    load_operating_cash_path,
    load_primary_deficit_path,
    load_tips_cpi_path,
    load_tips_real_yield_path,
    resolve_baseline_input_paths,
)
from fed_income import (
    FED_INCOME_RESULT_COLUMNS,
    INTEREST_EXPENSE_DAY_COUNT,
    TREASURY_INCOME_COLUMNS,
    FedIncomeError,
    FedOpeningBalanceSheet,
    non_treasury_reserve_effect,
    remittance_week,
    settle_remittances,
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
    resolve_holder_absorption_scenario,
)
from simulation_calendar import add_fedwire_business_days, is_fedwire_business_day, is_securities_business_day
from yield_curve_path import curve_for_date, load_yield_curve_surface


CBO_FUNDING_MODE = 'cbo_public_debt_target'
CBO_DEBT_REFERENCE_FINANCING_MODE = 'cbo_debt_reference_plus_tga_floor_financing_v1'
CBO_OPERATING_CASH_TARGET_FINANCING_MODE = 'cbo_operating_cash_target_financing_v1'
CBO_FUNDING_MODES = frozenset(
    {CBO_FUNDING_MODE, CBO_DEBT_REFERENCE_FINANCING_MODE, CBO_OPERATING_CASH_TARGET_FINANCING_MODE}
)
# Issuance legs whose proceeds are sized to a cash need rather than a debt path.
OPERATING_CASH_RESULT_COLUMNS = (
    'IssuanceNeedProceeds',
    'PendingIssuanceProceeds',
    'EarlySettlementProceeds',
    'CashManagementBillProceeds',
    'FedRolloverFace',
    'FedRolloverProceeds',
    'FedRolloverReallocatedStock',
    'FedRedeemedFace',
)
# The Fed's add-ons at auction: not operating-cash financing, and paid for by the Fed.
FED_ROLLOVER_ISSUANCE_LEG = 'fed_rollover_add_on'
CASH_FINANCING_ISSUANCE_LEGS = frozenset({'tga_floor_cash_financing', 'operating_cash_target_financing'})
CASH_FUNDING_MODE = 'cash_tga_target'
_CASH_MANAGEMENT_BILL_KEY = (CASH_MANAGEMENT_BILL_KIND, float(CASH_MANAGEMENT_BILL_WEEKS))


def _is_cbo_funding_mode(params):
    funding_rule = params.get('funding_rule', {})
    if not isinstance(funding_rule, dict):
        return False
    return str(funding_rule.get('mode', '')).strip() in CBO_FUNDING_MODES


def _evaluated_nominal_shock_from_config(yield_surface_cfg):
    if not isinstance(yield_surface_cfg, dict):
        return None
    shock = yield_surface_cfg.get('evaluated_nominal_shock')
    if shock is None:
        return None
    if not isinstance(shock, EvaluatedNominalShock):
        raise CurveContractError(
            'yield_curve_surface.evaluated_nominal_shock must be an '
            'EvaluatedNominalShock'
        )
    return shock


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


def _cbo_macro_value(cbo_engine_inputs, scenario_name, current_date, column, default_value):
    if not isinstance(cbo_engine_inputs, dict):
        return default_value
    macro = cbo_engine_inputs.get('macro_forecast')
    if macro is None or macro.empty or column not in macro.columns:
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
    value = pd.to_numeric(covering.iloc[0][column], errors='coerce')
    return float(value) if not pd.isna(value) else default_value


def _cbo_macro_cpi_level(cbo_engine_inputs, scenario_name, current_date, default_value):
    return _cbo_macro_value(
        cbo_engine_inputs,
        scenario_name,
        current_date,
        'cbo_cpi_u_index',
        default_value,
    )


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
        'fed_remittance_path_file': paths.fed_remittance_path_file,
        'fed_income_path_file': paths.fed_income_path_file,
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
    fed_remittance = load_fed_remittance_path(
        paths.fed_remittance_path_file,
        actuals_available_as_of=actuals_available_as_of,
        allow_lookahead=allow_lookahead,
    )
    fed_income = load_fed_income_path(
        paths.fed_income_path_file,
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
        yield_curve_surface = load_forecast_yield_curve_surface(
            paths.yield_curve_surface_file,
            actuals_available_as_of=actuals_available_as_of,
            allow_lookahead=allow_lookahead,
        )
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
        'fed_remittance': fed_remittance,
        'fed_income': fed_income,
        'fed_income_lookup': _cbo_period_lookup(fed_income),
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


def _fed_income_row(cbo_inputs, scenario_id, period_start, period_end):
    return _lookup_cbo_period_row(
        cbo_inputs['fed_income_lookup'], cbo_inputs['fed_income'], scenario_id, period_start, period_end,
        'Fed income path',
    )


def _fed_on_rrp_balance(trc_ledger, opening):
    """ON RRP balances the Fed pays interest on: the run's routed balances, else the H.4.1 level."""

    if trc_ledger is None:
        return opening.on_rrp
    return trc_ledger.balances['ONRRP_MMF'] + trc_ledger.balances['ForeignOfficialFedRRP']


def _subtract_business_days(date_value, business_days):
    """Business days back for an FRN's rate lockout: days the Federal Reserve Banks are
    open, the definition in 31 CFR 356.2 (Fedwire's calendar)."""

    return pd.Timestamp(add_fedwire_business_days(date_value, -max(0, int(round(float(business_days or 0))))))


def _add_business_days(date_value, business_days):
    return pd.Timestamp(add_fedwire_business_days(date_value, max(0, int(round(float(business_days or 0))))))


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


def _generic_tips_real_curve(tips_params):
    curve = tips_params.get('real_yield_curve', {}) if isinstance(tips_params, dict) else {}
    if not isinstance(curve, dict) or not curve:
        return ([], [])
    years = curve.get('years', [])
    rates = curve.get('rates', [])
    if not isinstance(years, list) or not isinstance(rates, list) or len(years) != len(rates):
        raise ValueError('TIPS real yield curve must provide equal-length years and rates lists.')
    if not years:
        return ([], [])
    year_values = [float(value) for value in years]
    rate_values = [float(value) for value in rates]
    if (
        any(not np.isfinite(value) or value <= 0.0 for value in year_values)
        or any(not np.isfinite(value) for value in rate_values)
        or len(set(year_values)) != len(year_values)
    ):
        raise ValueError(
            'TIPS real yield curve must contain finite, unique positive tenors and finite rates.'
        )
    pairs = sorted(zip(year_values, rate_values))
    return ([pair[0] for pair in pairs], [pair[1] for pair in pairs])


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


def _handoff_period(prev_date, current_date):
    return {
        'period_start': str(pd.Timestamp(prev_date).date()),
        'period_end': str(pd.Timestamp(current_date).date()),
    }


def _handoff_holder(row_or_mapping):
    holder = row_or_mapping.get('HolderType', '') if hasattr(row_or_mapping, 'get') else ''
    holder_subbucket = row_or_mapping.get('HolderSubBucket', '') if hasattr(row_or_mapping, 'get') else ''
    return {
        'holder_sector': '' if pd.isna(holder) else str(holder),
        'holder_subsector': '' if pd.isna(holder_subbucket) else str(holder_subbucket),
    }


def _handoff_instrument_type(row_or_type):
    if hasattr(row_or_type, 'get'):
        value = row_or_type.get('SecurityType', '')
    else:
        value = row_or_type
    return '' if pd.isna(value) else str(value)


def _handoff_security_id(row_or_mapping):
    value = row_or_mapping.get('BondID', '') if hasattr(row_or_mapping, 'get') else ''
    if pd.isna(value) or value == '':
        return ''
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _handoff_flow_id(*parts):
    cleaned = []
    for part in parts:
        if pd.isna(part):
            cleaned.append('')
        else:
            cleaned.append(str(part).replace('|', '_'))
    return '|'.join(cleaned)


def _handoff_maturity_bucket(security_type, maturity_years=None, maturity_category=None):
    if str(security_type) == 'Fixed' and not pd.isna(maturity_category):
        category = str(maturity_category).strip()
        if category:
            if category not in PREFERENCE_CATEGORIES:
                raise ValueError(
                    f"Non-canonical maturity category at RateWall handoff: {category!r}"
                )
            return category
    try:
        years = float(maturity_years)
    except (TypeError, ValueError):
        return 'unknown'
    if not np.isfinite(years):
        return 'unknown'
    if years <= 1.0 + TGA_FLOOR_TOLERANCE:
        return 'bills'
    if years <= 10.0 + TGA_FLOOR_TOLERANCE:
        return 'notes'
    return 'bonds'


# Columns _handoff_append_journal reads from a security row.
_JOURNAL_ROW_COLUMNS = (
    'BondID',
    'SecurityType',
    'HolderType',
    'HolderSubBucket',
    'TDCPrincipalHolderType',
    'TDCPrincipalHolderSubBucket',
    'OriginalMaturityYears',
    'MaturityCategory',
)


def _handoff_append_journal(
    handoff_tables,
    prev_date,
    current_date,
    *,
    event_type,
    leg_type,
    accounting_basis,
    row=None,
    security_id='',
    holder_sector='',
    holder_subsector='',
    route_holder_sector=None,
    route_holder_subsector=None,
    counterparty_sector='',
    counterparty_subsector='',
    instrument_type='',
    maturity_bucket='',
    face_stock_change=0.0,
    adjusted_principal_change=0.0,
    route_face_stock_change=None,
    route_adjusted_principal_change=None,
    treasury_cash_change=0.0,
    reserve_change=0.0,
    deposit_change=0.0,
    settlement_scope='',
    is_intragovernmental=False,
    drop_below=TGA_FLOOR_TOLERANCE,
):
    if row is not None:
        security_id = _handoff_security_id(row)
        holder = _handoff_holder(row)
        holder_sector = holder['holder_sector']
        holder_subsector = holder['holder_subsector']
        instrument_type = _handoff_instrument_type(row)
        maturity_bucket = _handoff_maturity_bucket(
            instrument_type,
            row.get('OriginalMaturityYears', np.nan),
            row.get('MaturityCategory', ''),
        )
        route = _tdc_principal_holder(row)
        if route_holder_sector is None:
            route_holder_sector = route['holder_sector']
        if route_holder_subsector is None:
            route_holder_subsector = route['holder_subsector']
    if route_holder_sector is None:
        route_holder_sector = holder_sector
    if route_holder_subsector is None:
        route_holder_subsector = holder_subsector
    if route_face_stock_change is None:
        route_face_stock_change = face_stock_change
    if route_adjusted_principal_change is None:
        route_adjusted_principal_change = adjusted_principal_change
    amounts = (
        face_stock_change,
        adjusted_principal_change,
        route_face_stock_change,
        route_adjusted_principal_change,
        treasury_cash_change,
        reserve_change,
        deposit_change,
    )
    if max(abs(float(value or 0.0)) for value in amounts) <= drop_below:
        return
    journal = handoff_tables.setdefault('tdcsim_accounting_journal', [])
    journal.append(
        {
            **_handoff_period(prev_date, current_date),
            'journal_id': _handoff_flow_id(
                'journal',
                pd.Timestamp(current_date).date(),
                len(journal),
                event_type,
                leg_type,
                security_id,
                holder_sector,
                holder_subsector,
            ),
            'event_type': str(event_type),
            'leg_type': str(leg_type),
            'security_id': str(security_id),
            'holder_sector': str(holder_sector),
            'holder_subsector': str(holder_subsector),
            'counterparty_sector': str(counterparty_sector),
            'counterparty_subsector': str(counterparty_subsector),
            'route_holder_sector': str(route_holder_sector),
            'route_holder_subsector': str(route_holder_subsector),
            'instrument_type': str(instrument_type),
            'maturity_bucket': str(maturity_bucket),
            'accounting_basis': str(accounting_basis),
            'face_stock_change_bil': float(face_stock_change or 0.0),
            'adjusted_principal_change_bil': float(adjusted_principal_change or 0.0),
            'route_face_stock_change_bil': float(route_face_stock_change or 0.0),
            'route_adjusted_principal_change_bil': float(
                route_adjusted_principal_change or 0.0
            ),
            'treasury_cash_change_bil': float(treasury_cash_change or 0.0),
            'reserve_change_bil': float(reserve_change or 0.0),
            'deposit_change_bil': float(deposit_change or 0.0),
            'settlement_scope': str(settlement_scope),
            'is_intragovernmental': bool(is_intragovernmental),
        }
    )


def _handoff_append_payment(
    handoff_tables,
    prev_date,
    current_date,
    row,
    payment_type,
    accounting_basis,
    amount,
    *,
    is_additive_to_cash_total,
):
    amount = float(amount or 0.0)
    if abs(amount) <= TGA_FLOOR_TOLERANCE:
        return
    security_type = _handoff_instrument_type(row)
    security_id = _handoff_security_id(row)
    holder = _handoff_holder(row)
    handoff_tables['tdcsim_period_payment_flows'].append(
        {
            **_handoff_period(prev_date, current_date),
            'flow_id': _handoff_flow_id(
                'payment',
                pd.Timestamp(current_date).date(),
                security_id,
                holder['holder_sector'],
                holder['holder_subsector'],
                payment_type,
                accounting_basis,
            ),
            'security_id': security_id,
            **holder,
            'instrument_type': security_type,
            'maturity_bucket': _handoff_maturity_bucket(
                security_type,
                row.get('OriginalMaturityYears', np.nan) if hasattr(row, 'get') else np.nan,
                row.get('MaturityCategory', '') if hasattr(row, 'get') else '',
            ),
            'payment_type': payment_type,
            'accounting_basis': accounting_basis,
            'amount_bil': amount,
            'is_additive_to_cash_total': bool(is_additive_to_cash_total),
        }
    )
    is_intragovernmental = holder['holder_sector'] in INTRAGOV_HOLDERS
    is_floor_bridge = payment_type == 'tips_deflation_floor_topup'
    _handoff_append_journal(
        handoff_tables,
        prev_date,
        current_date,
        event_type='deflation_floor' if is_floor_bridge else 'coupon_or_payment',
        leg_type=payment_type,
        accounting_basis=accounting_basis,
        row=row,
        treasury_cash_change=(
            -amount
            if (is_additive_to_cash_total or is_floor_bridge) and not is_intragovernmental
            else 0.0
        ),
        settlement_scope=(
            'consolidated_noncash_intragovernmental'
            if is_intragovernmental
            else 'treasury_cash_settlement'
        ),
        is_intragovernmental=is_intragovernmental,
    )


def _handoff_append_principal(
    handoff_tables,
    prev_date,
    current_date,
    row,
    *,
    redemption_type,
    face_redeemed,
    principal_redeemed,
    cash_paid,
    adjusted_principal_stock_removed=None,
    mmf_deposit_pass_through=1.0,
    mmf_added_deposit_share=0.0,
):
    face_redeemed = float(face_redeemed or 0.0)
    principal_redeemed = float(principal_redeemed or 0.0)
    cash_paid = float(cash_paid or 0.0)
    if max(abs(face_redeemed), abs(principal_redeemed), abs(cash_paid)) <= TGA_FLOOR_TOLERANCE:
        return
    security_type = _handoff_instrument_type(row)
    security_id = _handoff_security_id(row)
    tdc_recipient = _tdc_principal_holder(row)
    tdc_cash_to_du = 0.0
    tdc_principal_to_du = 0.0
    tdc_cash_to_du_domestic_nonbank = 0.0
    tdc_principal_to_du_domestic_nonbank = 0.0
    tdc_cash_to_du_mmf = 0.0
    tdc_principal_to_du_mmf = 0.0
    tdc_cash_to_du_mmf_plumbing = 0.0
    tdc_principal_to_du_mmf_plumbing = 0.0
    if tdc_recipient["holder_sector"] == "Private":
        route = tdc_recipient["holder_subsector"]
        if route in (PRIVATE_SUBBUCKET_MMF, PRIVATE_SUBBUCKET_MMF_ADDED):
            share = mmf_added_deposit_share if route == PRIVATE_SUBBUCKET_MMF_ADDED else mmf_deposit_pass_through
            tdc_cash_to_du_mmf = share * cash_paid
            tdc_principal_to_du_mmf = share * principal_redeemed
            tdc_cash_to_du_mmf_plumbing = (1.0 - share) * cash_paid
            tdc_principal_to_du_mmf_plumbing = (1.0 - share) * principal_redeemed
        else:
            tdc_cash_to_du_domestic_nonbank = cash_paid
            tdc_principal_to_du_domestic_nonbank = principal_redeemed
        tdc_cash_to_du = tdc_cash_to_du_domestic_nonbank + tdc_cash_to_du_mmf
        tdc_principal_to_du = tdc_principal_to_du_domestic_nonbank + tdc_principal_to_du_mmf
    handoff_tables['tdcsim_period_principal_flows'].append(
        {
            **_handoff_period(prev_date, current_date),
            'flow_id': _handoff_flow_id(
                'principal',
                pd.Timestamp(current_date).date(),
                security_id,
                redemption_type,
                row.get('HolderType', '') if hasattr(row, 'get') else '',
                row.get('HolderSubBucket', '') if hasattr(row, 'get') else '',
            ),
            'security_id': security_id,
            **_handoff_holder(row),
            'instrument_type': security_type,
            'maturity_bucket': _handoff_maturity_bucket(
                security_type,
                row.get('OriginalMaturityYears', np.nan) if hasattr(row, 'get') else np.nan,
                row.get('MaturityCategory', '') if hasattr(row, 'get') else '',
            ),
            'redemption_type': redemption_type,
            'face_redeemed_bil': face_redeemed,
            'principal_redeemed_bil': principal_redeemed,
            'cash_paid_bil': cash_paid,
            'adjusted_principal_stock_removed_bil': (
                float(adjusted_principal_stock_removed)
                if adjusted_principal_stock_removed is not None
                and not pd.isna(adjusted_principal_stock_removed)
                else np.nan
            ),
            'tdc_principal_recipient_sector': tdc_recipient["holder_sector"],
            'tdc_principal_recipient_subsector': tdc_recipient["holder_subsector"],
            'tdc_principal_cash_paid_to_du_bil': tdc_cash_to_du,
            'tdc_principal_redeemed_to_du_bil': tdc_principal_to_du,
            'tdc_principal_cash_paid_to_du_domestic_nonbank_bil': tdc_cash_to_du_domestic_nonbank,
            'tdc_principal_redeemed_to_du_domestic_nonbank_bil': tdc_principal_to_du_domestic_nonbank,
            'tdc_principal_cash_paid_to_du_mmf_bil': tdc_cash_to_du_mmf,
            'tdc_principal_redeemed_to_du_mmf_bil': tdc_principal_to_du_mmf,
            'tdc_principal_cash_paid_to_du_mmf_plumbing_bil': tdc_cash_to_du_mmf_plumbing,
            'tdc_principal_redeemed_to_du_mmf_plumbing_bil': tdc_principal_to_du_mmf_plumbing,
            'tdc_principal_recipient_basis': 'current_cb_beneficial_holder_otherwise_recorded_tdc_principal_route',
        }
    )
    holder = _handoff_holder(row)
    is_intragovernmental = holder['holder_sector'] in INTRAGOV_HOLDERS
    adjusted_removed = (
        float(adjusted_principal_stock_removed)
        if adjusted_principal_stock_removed is not None
        and not pd.isna(adjusted_principal_stock_removed)
        else 0.0
    )
    stock_funded_cash = (
        min(cash_paid, adjusted_removed)
        if security_type == 'TIPS' and adjusted_removed > TGA_FLOOR_TOLERANCE
        else cash_paid
    )
    _handoff_append_journal(
        handoff_tables,
        prev_date,
        current_date,
        event_type=(
            'buyback'
            if redemption_type == 'explicit_retirement_at_par'
            else 'maturity_redemption'
        ),
        leg_type='principal_stock_removal',
        accounting_basis=(
            'tips_adjusted_principal_stock'
            if security_type == 'TIPS'
            else 'face_stock'
        ),
        row=row,
        face_stock_change=-face_redeemed,
        adjusted_principal_change=(
            -adjusted_removed if security_type == 'TIPS' else 0.0
        ),
        treasury_cash_change=(-stock_funded_cash if not is_intragovernmental else 0.0),
        settlement_scope=(
            'consolidated_noncash_intragovernmental'
            if is_intragovernmental
            else 'treasury_cash_settlement'
        ),
        is_intragovernmental=is_intragovernmental,
    )


def _handoff_append_issuance(
    handoff_tables,
    prev_date,
    current_date,
    *,
    security_type,
    maturity_years,
    face_issued,
    cash_proceeds,
    coupon_rate,
    reference_rate,
    spread_bps,
    issue_yield,
    security_id='',
    holder_sector='',
    holder_subsector='',
    issuance_leg='ordinary_issuance',
    issue_terms=None,
    adjusted_principal_issued=None,
    issue_cost=None,
):
    face_issued = float(face_issued or 0.0)
    cash_proceeds = float(cash_proceeds or 0.0)
    if max(abs(face_issued), abs(cash_proceeds)) <= TGA_FLOOR_TOLERANCE:
        return
    handoff_tables['tdcsim_period_issuance_flows'].append(
        {
            **(issue_terms or {}),
            **_handoff_period(prev_date, current_date),
            'flow_id': _handoff_flow_id('issuance', pd.Timestamp(current_date).date(), security_id),
            'security_id': _handoff_security_id({'BondID': security_id}),
            'holder_sector': '' if pd.isna(holder_sector) else str(holder_sector),
            'holder_subsector': '' if pd.isna(holder_subsector) else str(holder_subsector),
            'issuance_leg': str(issuance_leg),
            'instrument_type': str(security_type),
            'maturity_bucket': _handoff_maturity_bucket(security_type, maturity_years),
            'weighted_original_term_years': float(maturity_years) if not pd.isna(maturity_years) else np.nan,
            'face_issued_bil': face_issued,
            'cash_proceeds_bil': cash_proceeds,
            'discount_or_premium_bil': face_issued - cash_proceeds,
            # What the run books as issue cost: debt added less cash for a calendar issue,
            # the discount (never a premium) otherwise.
            'issue_cost_bil': (
                max(0.0, face_issued - cash_proceeds) if issue_cost is None else float(issue_cost)
            ),
            'coupon_rate_decimal': float(coupon_rate) if not pd.isna(coupon_rate) else np.nan,
            'reference_rate_decimal': float(reference_rate) if not pd.isna(reference_rate) else np.nan,
            'spread_bps': float(spread_bps) if not pd.isna(spread_bps) else np.nan,
            'issue_yield_decimal': float(issue_yield) if not pd.isna(issue_yield) else np.nan,
        }
    )
    is_intragovernmental = str(holder_sector) in INTRAGOV_HOLDERS
    _handoff_append_journal(
        handoff_tables,
        prev_date,
        current_date,
        event_type='intragovernmental_issuance' if is_intragovernmental else 'issuance',
        leg_type=(
            'consolidated_noncash_liability_credit'
            if is_intragovernmental
            else 'auction_liability_and_cash'
        ),
        accounting_basis=(
            'tips_face_and_adjusted_principal'
            if str(security_type) == 'TIPS'
            else 'face_stock'
        ),
        security_id=security_id,
        holder_sector=holder_sector,
        holder_subsector=holder_subsector,
        counterparty_sector=holder_sector,
        counterparty_subsector=holder_subsector,
        instrument_type=security_type,
        maturity_bucket=_handoff_maturity_bucket(security_type, maturity_years),
        face_stock_change=face_issued,
        adjusted_principal_change=(
            (face_issued if adjusted_principal_issued is None else float(adjusted_principal_issued))
            if str(security_type) == 'TIPS'
            else 0.0
        ),
        treasury_cash_change=cash_proceeds,
        settlement_scope=(
            'consolidated_noncash_intragovernmental'
            if is_intragovernmental
            else 'auction_cash_settlement'
        ),
        is_intragovernmental=is_intragovernmental,
    )


def _handoff_weighted_metrics(rows, *, short_cutoff_years=1.0):
    if rows is None or len(rows) == 0:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    if frame.empty or 'FaceValue' not in frame.columns:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    weights = pd.to_numeric(frame['FaceValue'], errors='coerce').fillna(0.0)
    years = pd.to_numeric(frame.get('OriginalMaturityYears', np.nan), errors='coerce').fillna(0.0)
    denom = float(weights.sum())
    if denom <= TGA_FLOOR_TOLERANCE:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    security_type = frame.get('SecurityType', pd.Series('', index=frame.index)).astype(str)
    bills = (security_type.eq('Fixed') & (years <= 1.0 + TGA_FLOOR_TOLERANCE))
    short = years <= short_cutoff_years + TGA_FLOOR_TOLERANCE
    return {
        'wam_years': float((weights * years).sum() / denom),
        'bill_share': float(weights.loc[bills].sum() / denom),
        'short_maturity_share': float(weights.loc[short].sum() / denom),
    }


def _handoff_outstanding_metrics(active_bonds, current_date, *, short_cutoff_years=1.0):
    if active_bonds is None or active_bonds.empty:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    frame = active_bonds.loc[controlled_marketable_mask(active_bonds)].copy()
    if frame.empty:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    weights = debt_base(frame)
    years = (pd.to_datetime(frame['MaturityDate'], errors='coerce') - pd.Timestamp(current_date)).dt.total_seconds() / (
        DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60
    )
    years = years.clip(lower=0.0).fillna(0.0)
    denom = float(weights.sum())
    if denom <= TGA_FLOOR_TOLERANCE:
        return {'wam_years': 0.0, 'bill_share': 0.0, 'short_maturity_share': 0.0}
    bills = bill_mask(frame)
    short = years <= short_cutoff_years + TGA_FLOOR_TOLERANCE
    return {
        'wam_years': float((weights * years).sum() / denom),
        'bill_share': float(weights.loc[bills].sum() / denom),
        'short_maturity_share': float(weights.loc[short].sum() / denom),
    }


def _handoff_append_holder_stocks(handoff_tables, active_bonds, current_date):
    if active_bonds is None or active_bonds.empty:
        return
    frame = active_bonds.copy()
    frame = _ensure_tdc_principal_route_columns(frame)
    effective_routes = frame.apply(_tdc_principal_holder, axis=1)
    frame['EffectiveTDCPrincipalHolderType'] = effective_routes.map(
        lambda route: route['holder_sector']
    )
    frame['EffectiveTDCPrincipalHolderSubBucket'] = effective_routes.map(
        lambda route: route['holder_subsector']
    )
    frame['DebtBase'] = np.where(
        frame['SecurityType'].astype(str).eq('TIPS'),
        pd.to_numeric(frame['AdjustedPrincipal'], errors='coerce').fillna(pd.to_numeric(frame['FaceValue'], errors='coerce')),
        pd.to_numeric(frame['FaceValue'], errors='coerce').fillna(0.0),
    )
    frame['FaceStock'] = pd.to_numeric(
        frame['FaceValue'], errors='coerce'
    ).fillna(0.0)
    frame['AdjustedPrincipalStock'] = np.where(
        frame['SecurityType'].astype(str).eq('TIPS'),
        pd.to_numeric(frame['AdjustedPrincipal'], errors='coerce').fillna(
            frame['FaceStock']
        ),
        0.0,
    )
    security_type = frame['SecurityType'].astype(str)
    maturity_years = pd.to_numeric(frame.get('OriginalMaturityYears', np.nan), errors='coerce')
    maturity_category = frame.get(
        'MaturityCategory', pd.Series(pd.NA, index=frame.index, dtype='string')
    ).astype('string').str.strip()
    fixed_category_provided = (
        security_type.eq('Fixed')
        & maturity_category.notna()
        & maturity_category.ne('')
    )
    for category in maturity_category.loc[fixed_category_provided].unique():
        _handoff_maturity_bucket('Fixed', maturity_category=category)
    frame['maturity_bucket'] = np.select(
        [
            fixed_category_provided,
            maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE,
            maturity_years <= 10.0 + TGA_FLOOR_TOLERANCE,
            maturity_years > 10.0 + TGA_FLOOR_TOLERANCE,
        ],
        [maturity_category.fillna(''), 'bills', 'notes', 'bonds'],
        default='unknown',
    )
    for debt_scope, rows in (
        ('all_active_treasury', frame),
        (
            'controlled_public_marketable',
            frame[
                frame['SecurityType'].astype(str).isin(['Fixed', 'TIPS', 'FRN'])
                & ~frame['HolderType'].astype(str).isin(INTRAGOV_HOLDERS)
            ],
        ),
    ):
        if rows.empty:
            continue
        grouped = rows.groupby(
            ['HolderType', 'HolderSubBucket', 'SecurityType', 'maturity_bucket'],
            dropna=False,
        )[['DebtBase', 'FaceStock', 'AdjustedPrincipalStock']].sum()
        for keys, amounts in grouped.iterrows():
            holder, subbucket, security_type, maturity_bucket = keys
            debt_amount = float(amounts['DebtBase'])
            face_amount = float(amounts['FaceStock'])
            adjusted_amount = float(amounts['AdjustedPrincipalStock'])
            if max(
                abs(debt_amount),
                abs(face_amount),
                abs(adjusted_amount),
            ) <= TGA_FLOOR_TOLERANCE:
                continue
            handoff_tables['tdcsim_holder_stocks'].append(
                {
                    'date': str(pd.Timestamp(current_date).date()),
                    'holder_sector': '' if pd.isna(holder) else str(holder),
                    'holder_subsector': '' if pd.isna(subbucket) else str(subbucket),
                    'instrument_type': '' if pd.isna(security_type) else str(security_type),
                    'maturity_bucket': '' if pd.isna(maturity_bucket) else str(maturity_bucket),
                    'debt_held_bil': debt_amount,
                    'face_stock_bil': face_amount,
                    'adjusted_principal_stock_bil': adjusted_amount,
                    'valuation_basis': 'tips_adjusted_principal' if str(security_type) == 'TIPS' else 'face',
                    'debt_scope': debt_scope,
                    'allocation_method': 'end_of_period_stock_snapshot',
                }
            )
        route_grouped = rows.groupby(
            [
                'EffectiveTDCPrincipalHolderType',
                'EffectiveTDCPrincipalHolderSubBucket',
                'SecurityType',
                'maturity_bucket',
            ],
            dropna=False,
        )[['DebtBase', 'FaceStock', 'AdjustedPrincipalStock']].sum()
        for keys, amounts in route_grouped.iterrows():
            route_holder, route_subbucket, security_type, maturity_bucket = keys
            debt_amount = float(amounts['DebtBase'])
            face_amount = float(amounts['FaceStock'])
            adjusted_amount = float(amounts['AdjustedPrincipalStock'])
            if max(
                abs(debt_amount),
                abs(face_amount),
                abs(adjusted_amount),
            ) <= TGA_FLOOR_TOLERANCE:
                continue
            handoff_tables['tdcsim_tdc_principal_route_stocks'].append(
                {
                    'date': str(pd.Timestamp(current_date).date()),
                    'route_holder_sector': '' if pd.isna(route_holder) else str(route_holder),
                    'route_holder_subsector': '' if pd.isna(route_subbucket) else str(route_subbucket),
                    'instrument_type': '' if pd.isna(security_type) else str(security_type),
                    'maturity_bucket': '' if pd.isna(maturity_bucket) else str(maturity_bucket),
                    'route_debt_held_bil': debt_amount,
                    'route_face_stock_bil': face_amount,
                    'route_adjusted_principal_stock_bil': adjusted_amount,
                    'valuation_basis': 'tips_adjusted_principal' if str(security_type) == 'TIPS' else 'face',
                    'debt_scope': debt_scope,
                    'allocation_method': 'end_of_period_tdc_principal_route_stock_snapshot',
                    'route_stock_basis': 'tdc_principal_settlement_route',
                }
            )


_STOCK_MAP_COLUMNS = (
    'Status',
    'SecurityType',
    'FaceValue',
    'AdjustedPrincipal',
    'OriginalMaturityYears',
    'MaturityCategory',
    'HolderType',
    'HolderSubBucket',
    'TDCPrincipalHolderType',
    'TDCPrincipalHolderSubBucket',
)


def _handoff_portfolio_stock_map(portfolio, *, route=False):
    if portfolio is None or portfolio.empty:
        return {}
    columns = [column for column in _STOCK_MAP_COLUMNS if column in portfolio.columns]
    if 'Status' in portfolio.columns:
        frame = portfolio.loc[portfolio['Status'].astype(str).eq('Active'), columns].copy()
    else:
        frame = portfolio[columns].copy()
    if frame.empty:
        return {}
    frame = _ensure_tdc_principal_route_columns(frame)
    security_type = frame.get(
        'SecurityType', pd.Series('', index=frame.index)
    ).fillna('').astype(str)
    face = pd.to_numeric(
        frame.get('FaceValue', pd.Series(0.0, index=frame.index)),
        errors='coerce',
    ).fillna(0.0)
    adjusted = pd.to_numeric(
        frame.get('AdjustedPrincipal', face),
        errors='coerce',
    ).fillna(face)
    frame['_face_stock'] = face
    frame['_adjusted_principal_stock'] = np.where(
        security_type.eq('TIPS'), adjusted, 0.0
    )
    bucket_cache = {}
    buckets = []
    for key in zip(
        security_type,
        frame.get(
            'OriginalMaturityYears',
            pd.Series(np.nan, index=frame.index),
        ),
        frame.get(
            'MaturityCategory',
            pd.Series('', index=frame.index),
        ),
    ):
        bucket = bucket_cache.get(key)
        if bucket is None:
            bucket = _handoff_maturity_bucket(*key)
            bucket_cache[key] = bucket
        buckets.append(bucket)
    frame['_maturity_bucket'] = buckets
    if route:
        # Vectorized _tdc_principal_holder: the route columns were normalized by
        # _ensure_tdc_principal_route_columns above, so only the CB override remains.
        current_cb = frame['HolderType'].fillna('').astype(str).eq('CB')
        frame['_route_holder'] = frame['TDCPrincipalHolderType'].where(~current_cb, 'CB')
        frame['_route_subbucket'] = frame['TDCPrincipalHolderSubBucket'].where(~current_cb, '')
        holder_columns = ['_route_holder', '_route_subbucket']
    else:
        holder_columns = ['HolderType', 'HolderSubBucket']
    grouped = frame.groupby(
        [*holder_columns, 'SecurityType', '_maturity_bucket'],
        dropna=False,
    )[['_face_stock', '_adjusted_principal_stock']].sum()
    return {
        tuple('' if pd.isna(part) else str(part) for part in key): (
            float(face_total),
            float(adjusted_total),
        )
        for key, face_total, adjusted_total in zip(
            grouped.index,
            grouped['_face_stock'].to_numpy(),
            grouped['_adjusted_principal_stock'].to_numpy(),
        )
    }


def _handoff_append_transfer_changes(
    handoff_tables,
    prev_date,
    current_date,
    before,
    after,
    *,
    event_type,
    settlement_scope,
):
    if isinstance(before, dict) and set(before) == {'holder', 'route'}:
        holder_before = before['holder']
        route_before = before['route']
    else:
        holder_before = _handoff_portfolio_stock_map(before)
        route_before = _handoff_portfolio_stock_map(before, route=True)
    if isinstance(after, dict) and set(after) == {'holder', 'route'}:
        holder_after = after['holder']
        route_after = after['route']
    else:
        holder_after = _handoff_portfolio_stock_map(after)
        route_after = _handoff_portfolio_stock_map(after, route=True)
    for key in sorted(set(holder_before) | set(holder_after)):
        before_face, before_adjusted = holder_before.get(key, (0.0, 0.0))
        after_face, after_adjusted = holder_after.get(key, (0.0, 0.0))
        face_change = after_face - before_face
        adjusted_change = after_adjusted - before_adjusted
        if max(abs(face_change), abs(adjusted_change)) <= TGA_FLOOR_TOLERANCE:
            continue
        holder, subbucket, instrument_type, maturity_bucket = key
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type=event_type,
            leg_type='transfer_in' if max(face_change, adjusted_change) > 0.0 else 'transfer_out',
            accounting_basis='beneficial_holder_stock_transfer',
            holder_sector=holder,
            holder_subsector=subbucket,
            route_holder_sector='',
            route_holder_subsector='',
            counterparty_sector='secondary_market',
            instrument_type=instrument_type,
            maturity_bucket=maturity_bucket,
            face_stock_change=face_change,
            adjusted_principal_change=adjusted_change,
            route_face_stock_change=0.0,
            route_adjusted_principal_change=0.0,
            settlement_scope=settlement_scope,
            is_intragovernmental=holder in INTRAGOV_HOLDERS,
        )
    for key in sorted(set(route_before) | set(route_after)):
        before_face, before_adjusted = route_before.get(key, (0.0, 0.0))
        after_face, after_adjusted = route_after.get(key, (0.0, 0.0))
        face_change = after_face - before_face
        adjusted_change = after_adjusted - before_adjusted
        if max(abs(face_change), abs(adjusted_change)) <= TGA_FLOOR_TOLERANCE:
            continue
        route_holder, route_subbucket, instrument_type, maturity_bucket = key
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type=event_type,
            leg_type=(
                'route_transfer_in'
                if max(face_change, adjusted_change) > 0.0
                else 'route_transfer_out'
            ),
            accounting_basis='tdc_principal_route_stock_transfer',
            holder_sector='',
            holder_subsector='',
            route_holder_sector=route_holder,
            route_holder_subsector=route_subbucket,
            counterparty_sector='secondary_market',
            instrument_type=instrument_type,
            maturity_bucket=maturity_bucket,
            face_stock_change=0.0,
            adjusted_principal_change=0.0,
            route_face_stock_change=face_change,
            route_adjusted_principal_change=adjusted_change,
            settlement_scope=settlement_scope,
            is_intragovernmental=route_holder in INTRAGOV_HOLDERS,
        )


def _retire_public_marketable_debt_to_target(bond_portfolio, amount_to_retire, current_date):
    """Retire active public marketable debt at par, shortest maturities first."""
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio, 0.0, {h: 0.0 for h in HOLDER_TYPES}, _zero_private_routes(), []
    remaining = max(0.0, float(amount_to_retire))
    paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
    paid_by_tdc_private_route = _zero_private_routes()
    retirement_events = []
    active = bond_portfolio[
        (bond_portfolio['Status'] == 'Active')
        & (bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS', 'FRN']))
        & (~bond_portfolio['HolderType'].isin(INTRAGOV_HOLDERS))
    ].copy()
    if active.empty:
        return bond_portfolio, 0.0, paid_by_holder, paid_by_tdc_private_route, retirement_events
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
        tdc_recipient = _tdc_principal_holder(row)
        if tdc_recipient['holder_sector'] == 'Private':
            paid_by_tdc_private_route[tdc_recipient['holder_subsector']] += retire_amount
        retire_fraction = retire_amount / debt_base
        retired_face = float(row['FaceValue']) * retire_fraction
        retired_adjusted = (
            float(row.get('AdjustedPrincipal', debt_base)) * retire_fraction
            if row.get('SecurityType') == 'TIPS'
            else 0.0
        )
        event_row = row.copy()
        event_row['FaceValue'] = retired_face
        event_row['RetiredFaceValue'] = retired_face
        event_row['RetiredCashPaid'] = retire_amount
        if event_row.get('SecurityType') == 'TIPS':
            event_row['AdjustedPrincipal'] = retired_adjusted
            event_row['RetiredAdjustedPrincipal'] = retired_adjusted
        retirement_events.append(event_row)
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
    return bond_portfolio, retired_total, paid_by_holder, paid_by_tdc_private_route, retirement_events


_FED_SALE_MIX_CONTEMPORANEOUS = 'contemporaneous_non_cb_public_holder_mix'
_FED_SALE_MIX_PRIVATE = 'private_domestic_nonbank'
_FED_TRANSFER_SCALABLE_COLUMNS = list(TRANSFER_SCALABLE_COLUMNS)


def _fed_secondary_dirty_value(
    row,
    stock_amount,
    current_date,
    yield_curve_years,
    yield_curve_rates,
    *,
    interpolation_method,
    floor_zero,
    nominal_shock=None,
    tips_real_curve_years=None,
    tips_real_curve_rates=None,
):
    """Return dirty market value for a transferred par/adjusted-principal stock amount.

    TIPS carry real cash flows, so their indexed base is discounted off the real curve when
    one is configured; the deflation floor keeps the nominal curve because it is a nominal
    payoff. Nominal securities use the nominal curve throughout.
    """
    debt_base = _bond_debt_base(row)
    if not np.isfinite(debt_base) or debt_base <= TGA_FLOOR_TOLERANCE:
        raise ValueError('Fed secondary transfer row has no positive stock quantity.')
    maturity_date = pd.to_datetime(row.get('MaturityDate'), errors='coerce')
    settlement_date = pd.to_datetime(current_date, errors='coerce')
    if pd.isna(maturity_date) or pd.isna(settlement_date) or maturity_date <= settlement_date:
        raise ValueError('Fed secondary transfer row has no valid post-settlement maturity date.')
    security_type = str(row.get('SecurityType', ''))
    if security_type not in {'Fixed', 'TIPS', 'FRN'}:
        raise ValueError(f'Fed secondary transfer cannot price security type {security_type!r}.')
    stock_amount = float(stock_amount)
    if not np.isfinite(stock_amount) or stock_amount <= TGA_FLOOR_TOLERANCE:
        raise ValueError('Fed secondary transfer amount must be positive and finite.')
    fraction = stock_amount / debt_base
    if security_type == 'FRN':
        accrued_total = row.get('AccruedInterest_FRN', 0.0)
        accrued_total = 0.0 if pd.isna(accrued_total) else float(accrued_total)
        if not np.isfinite(accrued_total) or accrued_total < -TGA_FLOOR_TOLERANCE:
            raise ValueError('Fed secondary FRN transfer has invalid accrued interest.')
        clean_value = stock_amount
        accrued_interest = max(0.0, accrued_total) * fraction
        dirty_value = clean_value + accrued_interest
        discount_yield = np.nan
        time_to_maturity = (maturity_date - settlement_date).total_seconds() / (
            DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60
        )
    else:
        coupon_rate = row.get('CouponRate')
        if pd.isna(coupon_rate) or not np.isfinite(float(coupon_rate)) or float(coupon_rate) < 0.0:
            raise ValueError('Fed secondary transfer row has an invalid coupon rate.')
        coupon_rate = float(coupon_rate)
        time_to_maturity = (maturity_date - settlement_date).total_seconds() / (
            DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60
        )
        nominal_yield = evaluate_nominal_yield(
            time_to_maturity,
            yield_curve_years,
            yield_curve_rates,
            method=interpolation_method,
            floor_zero=floor_zero,
            shock=nominal_shock,
        )
        if pd.isna(nominal_yield) or not np.isfinite(float(nominal_yield)):
            raise ValueError('Fed secondary transfer row has no finite market yield.')
        # A TIPS holds real cash flows, so its indexed base takes a real yield interpolated at
        # REMAINING maturity — a ten-year TIPS with three years left is a three-year real-rate
        # instrument. Discounting those flows at the nominal curve double-counts inflation
        # relative to cash flows already fixed at today's index ratio.
        discount_yield = nominal_yield
        if security_type == 'TIPS':
            # Fail closed rather than fall back. An absent real curve previously left the
            # nominal yield in place, which prices real cash flows at a nominal rate - a ~26%
            # error on a seasoned TIPS, arrived at silently. A missing curve is a configuration
            # fault, and settling a Fed purchase on the wrong basis is worse than not settling.
            if not tips_real_curve_years or not tips_real_curve_rates:
                raise ValueError(
                    'Fed secondary TIPS transfer requires a real yield curve; the nominal curve '
                    'would misprice real cash flows.'
                )
            real_yield = get_yield_for_maturity(
                time_to_maturity,
                tips_real_curve_years,
                tips_real_curve_rates,
                method=interpolation_method,
                floor_zero=False,
            )
            if pd.isna(real_yield) or not np.isfinite(float(real_yield)):
                raise ValueError('Fed secondary TIPS transfer has no finite real yield.')
            discount_yield = real_yield
        # For this Fed-secondary transfer path, value the security from dated remaining cash
        # flows at the Appendix B semiannual convention, with the TIPS deflation floor as a
        # property of the instrument. The issue-price kernel used here previously rounded
        # remaining life to whole coupon periods, which structurally deleted accrued
        # interest from settlement, and carried no deflation-floor concept at all.
        issue_date = pd.to_datetime(row.get('IssueDate'), errors='coerce')
        if coupon_rate > TGA_FLOOR_TOLERANCE and (
            pd.isna(issue_date) or issue_date > settlement_date
        ):
            raise ValueError('Fed secondary coupon transfer has no valid issue date for accrued interest.')
        valuation = value_treasury_security(
            settlement_date=settlement_date,
            maturity_date=maturity_date,
            coupon_rate=coupon_rate,
            discount_yield=float(discount_yield),
            security_type=security_type,
            face_value=float(row.get('FaceValue', 0.0) or 0.0),
            adjusted_principal=row.get('AdjustedPrincipal'),
            original_principal=row.get('OriginalPrincipal'),
            accrued_frn=row.get('AccruedInterest_FRN'),
            issue_date=issue_date,
            first_interest_payment_date=pd.to_datetime(row.get('FirstInterestPaymentDate'), errors='coerce'),
            nominal_discount_yield=float(nominal_yield),
        )
        clean_value = float(valuation['clean']) * fraction
        accrued_interest = max(0.0, float(valuation['accrued'])) * fraction
        if not np.isfinite(clean_value) or clean_value <= TGA_FLOOR_TOLERANCE:
            raise ValueError('Fed secondary transfer row produced an invalid clean price.')
        if not np.isfinite(accrued_interest) or accrued_interest < -TGA_FLOOR_TOLERANCE:
            raise ValueError('Fed secondary transfer row produced invalid accrued interest.')
        dirty_value = clean_value + accrued_interest
    if not np.isfinite(dirty_value) or dirty_value <= TGA_FLOOR_TOLERANCE:
        raise ValueError('Fed secondary transfer row produced invalid dirty market value.')
    return {
        'clean_value': float(clean_value),
        'accrued_interest': float(accrued_interest),
        'dirty_value': float(dirty_value),
        'dirty_price_ratio': float(dirty_value / stock_amount),
        'discount_yield': discount_yield,
        'time_to_maturity': float(time_to_maturity),
    }


def _cache_fed_transfer_price(bond_portfolio, idx, pricing):
    debt_base = _bond_debt_base(bond_portfolio.loc[idx])
    if debt_base <= TGA_FLOOR_TOLERANCE:
        return
    bond_portfolio.loc[idx, 'TimeToMaturity'] = pricing['time_to_maturity']
    bond_portfolio.loc[idx, 'DiscountYield'] = pricing['discount_yield']
    bond_portfolio.loc[idx, 'CleanPrice'] = pricing['clean_value']
    bond_portfolio.loc[idx, 'AccruedInterest'] = pricing['accrued_interest']
    bond_portfolio.loc[idx, 'DirtyValue'] = pricing['dirty_value']
    bond_portfolio.loc[idx, 'DirtyPriceRatio'] = pricing['dirty_value'] / debt_base


def _next_portfolio_bond_id(bond_portfolio):
    numeric_ids = pd.to_numeric(bond_portfolio['BondID'], errors='coerce')
    return int(numeric_ids.max()) + 1 if numeric_ids.notna().any() else 1


def _resolve_fed_secondary_sale_buyer_mix(bond_portfolio, mix_spec=None):
    """Resolve the declared sale policy to a typed holder/subbucket stock-share vector."""
    if mix_spec is None:
        mix_type = _FED_SALE_MIX_CONTEMPORANEOUS
    elif isinstance(mix_spec, str):
        mix_type = mix_spec.strip()
    elif isinstance(mix_spec, dict):
        mix_type = str(mix_spec.get('type', _FED_SALE_MIX_CONTEMPORANEOUS)).strip()
        mix_basis = str(mix_spec.get('basis', 'par_or_adjusted_principal_stock')).strip()
        if mix_basis != 'par_or_adjusted_principal_stock':
            raise ValueError(f'Unsupported Fed secondary sale buyer mix basis: {mix_basis!r}.')
    else:
        raise ValueError('funding_rule.fed_secondary_sale_buyer_mix must be a string or mapping.')
    if mix_type == _FED_SALE_MIX_PRIVATE:
        return [
            {
                'holder_type': 'Private',
                'holder_subbucket': PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                'share': 1.0,
            }
        ]
    if mix_type != _FED_SALE_MIX_CONTEMPORANEOUS:
        raise ValueError(f'Unsupported Fed secondary sale buyer mix type: {mix_type!r}.')
    active = bond_portfolio[
        (bond_portfolio['Status'] == 'Active')
        & (bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS', 'FRN']))
        & (bond_portfolio['HolderType'].isin(['Banks', 'Private', 'Foreign']))
    ].copy()
    if active.empty:
        raise ValueError('Fed secondary sale buyer mix has no contemporaneous non-CB public holders.')
    active['StockQuantity'] = pd.to_numeric(
        np.where(
            active['SecurityType'] == 'TIPS',
            active['AdjustedPrincipal'].fillna(active['FaceValue']),
            active['FaceValue'],
        ),
        errors='coerce',
    )
    if active['StockQuantity'].isna().any() or (active['StockQuantity'] < 0.0).any():
        raise ValueError('Fed secondary sale buyer mix contains malformed public holder stock.')
    active['BuyerSubBucket'] = ''
    private_mask = active['HolderType'] == 'Private'
    active.loc[private_mask, 'BuyerSubBucket'] = active.loc[private_mask].apply(_private_subbucket, axis=1)
    # Allocate the added position's share to the other domestic nonbank route.
    # Keep the ordinary money funds' weight and the all-holder denominator intact.
    active.loc[active['BuyerSubBucket'].eq(PRIVATE_SUBBUCKET_MMF_ADDED), 'BuyerSubBucket'] = (
        PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    )
    grouped = active.groupby(['HolderType', 'BuyerSubBucket'], dropna=False)['StockQuantity'].sum()
    total = float(grouped.sum())
    if not np.isfinite(total) or total <= TGA_FLOOR_TOLERANCE:
        raise ValueError('Fed secondary sale buyer mix has no positive contemporaneous stock.')
    vector = []
    for (holder, subbucket), amount in grouped.items():
        if float(amount) <= TGA_FLOOR_TOLERANCE:
            continue
        vector.append(
            {
                'holder_type': str(holder),
                'holder_subbucket': str(subbucket) if str(holder) == 'Private' else '',
                'share': float(amount) / total,
            }
        )
    return vector


def _format_fed_holder_mix(vector):
    return '|'.join(
        f"{item['holder_type']}/{item['holder_subbucket']}={item['share']:.12f}"
        for item in vector
    )


def _earliest_lot(bond_portfolio, eligible_indices):
    """The earliest-maturing eligible lot; equal maturities go in a fixed order of route and terms.

    Taking the first such row in frame order would make the Federal Reserve's purchases and
    sales, and everything downstream, depend on how holdings happen to be split into rows.
    """
    maturities = pd.to_datetime(bond_portfolio.loc[eligible_indices, 'MaturityDate'], errors='coerce')
    earliest = maturities.min()
    if pd.isna(earliest):
        raise ValueError('Federal Reserve secondary transfer found no eligible security with a maturity date.')
    tied = maturities.index[maturities == earliest]
    if len(tied) == 1:
        return tied[0]
    return holder_targets.economic_lot_order(bond_portfolio.loc[tied])[0]


def _purchase_issue_allocations(issues, amount):
    """Water-fill pro rata by non-SOMA par, expressing both sides in path stock units."""
    allocations = {}
    remaining = amount
    for sectors in (fed_purchase_policy.BILL_SECTORS, fed_purchase_policy.COUPON_SECTORS):
        initial = remaining
        eligible = {sector: {key: row for key, row in issues.items() if row['sector'] == sector}
                    for sector in sectors}

        def fill(sector, target):
            pool = {
                key: (row['public_stock'], row['capacity'] - allocations.get(key, 0.0))
                for key, row in eligible[sector].items()
                if row['capacity'] - allocations.get(key, 0.0) > TGA_FLOOR_TOLERANCE
            }
            while pool and target > TGA_FLOOR_TOLERANCE:
                scale = target / sum(weight for weight, _ in pool.values())
                capped = [key for key, (weight, cap) in pool.items() if weight * scale >= cap]
                if not capped:
                    for key, (weight, _) in pool.items():
                        allocations[key] = allocations.get(key, 0.0) + weight * scale
                    return 0.0
                for key in capped:
                    _, cap = pool.pop(key)
                    allocations[key] = allocations.get(key, 0.0) + cap
                    target -= cap
            return target

        remaining = sum(fill(sector, initial * sector.weight) for sector in sectors)
        for sector in sectors:
            remaining = fill(sector, remaining)
        if remaining <= TGA_FLOOR_TOLERANCE:
            break
    return allocations, remaining


def _purchase_issues(portfolio, current):
    """Identify the distinct outstanding issues eligible for a reserve-management purchase."""
    marketable = portfolio.loc[
        portfolio['Status'].eq('Active') & portfolio['SecurityType'].isin(PUBLIC_MARKETABLE_SECURITY_TYPES)
    ].sort_values('BondID', kind='stable')
    if marketable.empty:
        return {}
    marketable = marketable.assign(_maturity=pd.to_datetime(marketable['MaturityDate']).dt.normalize())
    if marketable['_maturity'].isna().any():
        raise ValueError('Fed purchase book contains an active marketable security without a maturity')
    bills = bill_mask(marketable)
    ids = fed_purchase_policy.issue_ids(marketable, bills)
    fed_purchase_policy.validate_issue_terms(marketable, ids, bills)
    nominal_coupons = marketable.loc[marketable['SecurityType'].eq('Fixed') & ~bills]
    # The newest issue in each nominal tenor is on the run. Comparing maturity
    # within each tenor keeps reopening settlement dates out of the comparison.
    tenors = nominal_coupons['OriginalMaturityYears'].round()
    on_the_run = nominal_coupons.groupby(tenors)['_maturity'].transform('max').eq(nominal_coupons['_maturity'])

    eligible = marketable.loc[
        ~marketable.index.isin(on_the_run.index[on_the_run])
        & ~marketable.get('IsCashManagementBill', pd.Series(False, index=marketable.index)).fillna(False)
        & (marketable['_maturity'] > current + pd.Timedelta(days=fed_purchase_policy.EXCLUDED_MATURITY_DAYS))
        & (marketable['_maturity'] <= current + pd.DateOffset(months=fed_purchase_policy.COUPON_SECTORS[-1].max_months))
    ].copy()
    eligible['_bill'] = bills.loc[eligible.index]
    eligible['_issue_id'] = ids.loc[eligible.index]
    issues = {}
    for key, issue in eligible.groupby('_issue_id', sort=True):
        is_bill, security_type, maturity = issue[['_bill', 'SecurityType', '_maturity']].iloc[0]
        sector = fed_purchase_policy.purchase_sector(current, maturity, bill=is_bill)
        if sector is None:
            continue
        face = issue['FaceValue']
        if not np.isfinite(face).all() or (face < 0.0).any():
            raise ValueError(f'Fed purchase issue {key} has invalid par outstanding')
        cb = issue['HolderType'].eq('CB')
        sellers = issue.loc[issue['HolderType'].isin(('Banks', 'Private', 'Foreign')) & face.gt(0.0)]
        public_par = float(sellers['FaceValue'].sum())
        cap = min(public_par, fed_purchase_policy.par_capacity(float(face.sum()), float(face[cb].sum()), bill=is_bill))
        if cap <= TGA_FLOOR_TOLERANCE:
            continue
        # Apply the issue limits to par. The SOMA path is in indexed stock for TIPS.
        stock = float(sellers['AdjustedPrincipal'].sum()) if security_type == 'TIPS' else public_par
        if not np.isfinite(stock) or stock <= 0.0:
            raise ValueError(f'Fed purchase issue {key} has invalid public debt stock')
        issues[key] = {
            'sector': sector, 'public_stock': stock, 'capacity': cap * stock / public_par,
            'sellers': sellers.index, 'cb': issue.index[cb], 'bill': is_bill, 'security_type': security_type,
        }
    return issues


def _transfer_public_marketable_debt_to_cb(
    bond_portfolio,
    amount_to_purchase,
    current_date,
    yield_curve_years,
    yield_curve_rates,
    *,
    interpolation_method='linear',
    floor_zero=True,
    nominal_shock=None,
    tips_real_curve_years=None,
    tips_real_curve_rates=None,
    portfolio_row_budget=None,
    operation_summary=None,
):
    """Buy eligible issues and settle the same fraction of every seller lot of an issue."""
    sold_by_holder = dict.fromkeys(HOLDER_TYPES, 0.0)
    sold_by_private_route = _zero_private_routes()
    if amount_to_purchase <= TGA_FLOOR_TOLERANCE:
        return bond_portfolio, 0.0, sold_by_holder, sold_by_private_route, 0.0
    current = pd.Timestamp(current_date).normalize()
    issues = _purchase_issues(bond_portfolio, current) if bond_portfolio is not None else {}
    allocations, shortfall = _purchase_issue_allocations(issues, amount_to_purchase)
    if shortfall > TGA_FLOOR_TOLERANCE:
        raise RuntimeError(f'Fed purchase capacity exhausted on {current.date()}: shortfall={shortfall:.12f} billion')

    row_growth = sum(1 - len(issues[key]['cb']) for key in allocations)
    if portfolio_row_budget is not None and len(bond_portfolio) + row_growth > int(portfolio_row_budget):
        raise RuntimeError('Portfolio-row budget would be exceeded by a Fed secondary purchase split.')

    portfolio = bond_portfolio.copy()
    next_bond_id = _next_portfolio_bond_id(portfolio)
    price = functools.partial(
        _fed_secondary_dirty_value, current_date=current, yield_curve_years=yield_curve_years,
        yield_curve_rates=yield_curve_rates, interpolation_method=interpolation_method,
        floor_zero=floor_zero, nominal_shock=nominal_shock, tips_real_curve_years=tips_real_curve_years,
        tips_real_curve_rates=tips_real_curve_rates,
    )
    scalable = [col for col in TRANSFER_SCALABLE_COLUMNS if col in portfolio]
    sector_totals = {}
    route_totals = defaultdict(lambda: {'par': 0.0, 'cash': 0.0})
    seller_updates = []
    fed_rows = []
    retired_indices = []
    for key, quantity in allocations.items():
        issue = issues[key]
        is_bill, security_type = issue['bill'], issue['security_type']
        fraction = quantity / issue['public_stock']
        indices = issue['sellers'].append(issue['cb'])
        priced = portfolio.loc[indices].copy()
        face = priced['FaceValue']
        # All bills of an issue have one market price. FRN accrued interest and
        # coupon cash on opening lots retain the existing dirty valuation.
        if is_bill:
            quote = price(priced.iloc[0], float(face.iloc[0]))
            for column, value in [('CleanPrice', 'clean_value'), ('AccruedInterest', 'accrued_interest'),
                                  ('DirtyValue', 'dirty_value')]:
                priced[column] = face * (quote[value] / face.iloc[0])
            priced['DiscountYield'] = quote['discount_yield']
            priced['TimeToMaturity'] = quote['time_to_maturity']
        else:
            quotes = [
                price(row, value) if (value := _bond_debt_base(row)) > TGA_FLOOR_TOLERANCE
                else dict.fromkeys(('clean_value', 'accrued_interest', 'dirty_value', 'discount_yield', 'time_to_maturity'), np.nan)
                for _, row in priced.iterrows()
            ]
            for column, value in [('CleanPrice', 'clean_value'), ('AccruedInterest', 'accrued_interest'),
                                  ('DirtyValue', 'dirty_value'), ('DiscountYield', 'discount_yield'),
                                  ('TimeToMaturity', 'time_to_maturity')]:
                priced[column] = [quote[value] for quote in quotes]
        stock = priced['AdjustedPrincipal'] if security_type == 'TIPS' else face
        priced['DirtyPriceRatio'] = priced['DirtyValue'] / stock
        sellers = priced.loc[issue['sellers']].copy()
        cb_lots = priced.loc[issue['cb']]
        proceeds = sellers['DirtyValue'] * fraction
        for idx, row in sellers.iterrows():
            cash = float(proceeds.at[idx])
            holder = str(row['HolderType'])
            route = holder
            sold_by_holder[holder] += cash
            if holder == 'Private':
                route = _private_subbucket(row)
                sold_by_private_route[route] += cash
            route_totals[route]['cash'] += cash
            route_totals[route]['par'] += float(row['FaceValue']) * fraction

        lot = (cb_lots if not cb_lots.empty else sellers).sort_values(['IssueDate', 'BondID']).iloc[0].copy()
        lot[scalable] = (
            pd.concat([cb_lots[scalable], sellers[scalable] * fraction]).sum(min_count=1)
        )
        lot['IssueID'] = key
        lot['IssuePriceRatio'] = lot['IssueProceeds'] / lot['FaceValue']
        lot['DirtyPriceRatio'] = lot['DirtyValue'] / _bond_debt_base(lot)
        if cb_lots.empty:
            lot['BondID'] = next_bond_id
            next_bond_id += 1
        lot[['HolderType', 'HolderSubBucket', 'TDCPrincipalHolderType', 'TDCPrincipalHolderSubBucket']] = ['CB', '', 'CB', '']
        fed_rows.append(lot)
        retired_indices.extend(issue['cb'])

        sector = sector_totals.setdefault(issue['sector'].name, {
            'par': 0.0, 'stock': 0.0, 'cash': 0.0, 'issues': 0, 'capacity_binds': 0,
        })
        sector['par'] += float(sellers['FaceValue'].sum()) * fraction
        sector['stock'] += quantity
        sector['cash'] += float(proceeds.sum())
        sector['issues'] += 1
        sector['capacity_binds'] += int(issue['capacity'] - quantity <= TGA_FLOOR_TOLERANCE)
        sellers[scalable] *= 1.0 - fraction
        seller_updates.append(sellers)

    sellers = pd.concat(seller_updates)
    portfolio.loc[sellers.index, sellers.columns] = sellers
    portfolio = pd.concat([portfolio.drop(index=retired_indices), pd.DataFrame(fed_rows)], ignore_index=True)
    if operation_summary is not None:
        operation_summary.update(source=fed_purchase_policy.SOURCE_URL, sectors=sector_totals, seller_routes=dict(route_totals))
    return portfolio, sum(allocations.values()), sold_by_holder, sold_by_private_route, sum(sold_by_holder.values())

def _transfer_cb_marketable_debt_to_public(
    bond_portfolio,
    amount_to_sell,
    current_date,
    yield_curve_years,
    yield_curve_rates,
    buyer_mix,
    *,
    interpolation_method='linear',
    floor_zero=True,
    nominal_shock=None,
    tips_real_curve_years=None,
    tips_real_curve_rates=None,
    portfolio_row_budget=None,
    record_mmf_acquisition=None,
):
    """Move CB marketable holdings to the declared public buyer vector at dirty value."""
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio, 0.0, {h: 0.0 for h in HOLDER_TYPES}, _zero_private_routes(), 0.0
    remaining = max(0.0, float(amount_to_sell))
    sold_total = 0.0
    sale_cash = 0.0
    bought_by_holder = {h: 0.0 for h in HOLDER_TYPES}
    bought_by_private_route = _zero_private_routes()
    next_bond_id = None
    new_rows = []
    non_mmf_buyers = [item for item in buyer_mix if item['holder_subbucket'] != PRIVATE_SUBBUCKET_MMF]
    non_mmf_share = sum(float(item['share']) for item in non_mmf_buyers)
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
        idx = _earliest_lot(bond_portfolio, eligible_indices)
        row = bond_portfolio.loc[idx].copy()
        debt_base = _bond_debt_base(row)
        if debt_base <= TGA_FLOOR_TOLERANCE:
            bond_portfolio.loc[idx, 'FaceValue'] = 0.0
            continue
        sale_amount = min(remaining, debt_base)
        pricing = _fed_secondary_dirty_value(
            row,
            sale_amount,
            current_date,
            yield_curve_years,
            yield_curve_rates,
            interpolation_method=interpolation_method,
            floor_zero=floor_zero,
            nominal_shock=nominal_shock,
            tips_real_curve_years=tips_real_curve_years,
            tips_real_curve_rates=tips_real_curve_rates,
        )
        full_pricing = {**pricing}
        scale_to_full = debt_base / sale_amount
        for key in ('clean_value', 'accrued_interest', 'dirty_value'):
            full_pricing[key] *= scale_to_full
        _cache_fed_transfer_price(bond_portfolio, idx, full_pricing)
        row = bond_portfolio.loc[idx].copy()
        fraction_sold = sale_amount / debt_base
        lot_buyers = buyer_mix
        if len(non_mmf_buyers) < len(buyer_mix) and not mmf_may_acquire(
            row['SecurityType'], row['MaturityDate'], current_date
        ):
            if non_mmf_share <= 0.0:
                raise ValueError('Fed secondary sale has no eligible buyers for the selected lot.')
            lot_buyers = [
                {**item, 'share': float(item['share']) / non_mmf_share}
                for item in non_mmf_buyers
            ]
        allocations = []
        allocated_face = 0.0
        for mix_index, item in enumerate(lot_buyers):
            allocation = sale_amount - allocated_face if mix_index == len(lot_buyers) - 1 else sale_amount * float(item['share'])
            allocated_face += allocation
            if allocation <= TGA_FLOOR_TOLERANCE:
                continue
            allocations.append((item, allocation, allocation / debt_base))
        if not allocations:
            raise ValueError('Fed secondary sale buyer mix resolved to no positive allocations.')
        if sale_amount < debt_base - TGA_FLOOR_TOLERANCE:
            for col in _FED_TRANSFER_SCALABLE_COLUMNS:
                if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                    bond_portfolio.loc[idx, col] = float(bond_portfolio.loc[idx, col]) * (1.0 - fraction_sold)
            allocation_rows = allocations
        else:
            first_item, first_amount, first_fraction = allocations[0]
            bond_portfolio.loc[idx, 'HolderType'] = first_item['holder_type']
            bond_portfolio.loc[idx, 'HolderSubBucket'] = first_item['holder_subbucket']
            bond_portfolio.loc[idx, 'TDCPrincipalHolderType'] = first_item['holder_type']
            bond_portfolio.loc[idx, 'TDCPrincipalHolderSubBucket'] = first_item['holder_subbucket']
            if len(allocations) > 1:
                for col in _FED_TRANSFER_SCALABLE_COLUMNS:
                    if col in bond_portfolio.columns and pd.notna(bond_portfolio.loc[idx, col]):
                        bond_portfolio.loc[idx, col] = float(bond_portfolio.loc[idx, col]) * first_fraction
            allocation_rows = allocations[1:]
        for item, allocation, allocation_fraction in allocation_rows:
            public_row = row.copy()
            if next_bond_id is None:
                next_bond_id = _next_portfolio_bond_id(bond_portfolio)
            public_row['BondID'] = next_bond_id
            next_bond_id += 1
            public_row['HolderType'] = item['holder_type']
            public_row['HolderSubBucket'] = item['holder_subbucket']
            public_row['TDCPrincipalHolderType'] = item['holder_type']
            public_row['TDCPrincipalHolderSubBucket'] = item['holder_subbucket']
            for col in _FED_TRANSFER_SCALABLE_COLUMNS:
                if col in row.index and pd.notna(row[col]):
                    public_row[col] = float(row[col]) * allocation_fraction
            new_rows.append(public_row)
        for item, allocation, _allocation_fraction in allocations:
            allocation_cash = pricing['dirty_value'] * (allocation / sale_amount)
            holder = item['holder_type']
            bought_by_holder[holder] = bought_by_holder.get(holder, 0.0) + allocation_cash
            if holder == 'Private':
                bought_by_private_route[item['holder_subbucket']] += allocation_cash
                if item['holder_subbucket'] == PRIVATE_SUBBUCKET_MMF and record_mmf_acquisition is not None:
                    record_mmf_acquisition(current_date, row['SecurityType'], row['MaturityDate'])
        sold_total += sale_amount
        sale_cash += pricing['dirty_value']
        remaining -= sale_amount
    if new_rows:
        if (
            portfolio_row_budget is not None
            and len(bond_portfolio) + len(new_rows) > int(portfolio_row_budget)
        ):
            raise RuntimeError(
                'Portfolio-row budget would be exceeded by a Fed secondary sale split.'
            )
        bond_portfolio = pd.concat([bond_portfolio, pd.DataFrame(new_rows)], ignore_index=True)
    return bond_portfolio, sold_total, bought_by_holder, bought_by_private_route, sale_cash


def _quote_issuance(
    security_type,
    maturity_years,
    coupon_rate,
    yield_at_issuance,
    amount,
    face_target_mode,
    *,
    bill_discount_rate=None,
    bill_days_to_maturity=CBO_3M_BILL_DAYS,
):
    if bill_discount_rate is not None and not pd.isna(bill_discount_rate):
        if (
            security_type != 'Fixed'
            or float(maturity_years) > 1.0 + TGA_FLOOR_TOLERANCE
            or float(coupon_rate or 0.0) > TGA_FLOOR_TOLERANCE
        ):
            raise ValueError('Treasury discount quote basis applies only to zero-coupon bills.')
        issue_price_ratio = discount_price_ratio(
            float(bill_discount_rate),
            float(bill_days_to_maturity),
        )
        if face_target_mode:
            face_value = float(amount)
            return face_value, face_value * issue_price_ratio, issue_price_ratio
        face_value = float(amount) / issue_price_ratio
        return face_value, float(amount), issue_price_ratio
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


def _quote_supply_item(
    security_type,
    maturity_years,
    target,
    *,
    face_target_mode,
    issuance_leg,
    current_date,
    bill_term_days,
    pricing,
    bill_days_to_maturity=None,
):
    """Price one issuance item; the caller dates it.

    ``bill_term_days`` is a bill's nominal term, which selects the CBO 3-month quote for
    the 13-week bill; ``bill_days_to_maturity`` is the days it is actually outstanding
    (fewer when a holiday moved its settlement), which that quote is priced over.

    A bill with ``bill_term_days`` equal to the 91-day CBO bill is quoted on the CBO
    3-month discount rate in CBO mode; other bills price off the nominal curve.
    """

    tips_real_curve_years = pricing['tips_real_curve_years']
    tips_real_curve_rates = pricing['tips_real_curve_rates']
    current_yield_curve_years = pricing['current_yield_curve_years']
    current_yield_curve_rates = pricing['current_yield_curve_rates']
    yield_interpolation_method = pricing['yield_interpolation_method']
    yield_floor_zero = pricing['yield_floor_zero']
    if security_type == 'TIPS':
        if not tips_real_curve_years or not tips_real_curve_rates:
            raise ValueError('TIPS issuance requires an explicit real yield curve.')
        issue_yield = get_yield_for_maturity(
            maturity_years,
            tips_real_curve_years,
            tips_real_curve_rates,
            method=yield_interpolation_method,
            floor_zero=False,
        )
        coupon = calculate_coupon_rate('TIPS', maturity_years, issue_yield, pricing['tips_real_coupon'])
        face, proceeds, price = _quote_issuance(
            'TIPS', maturity_years, coupon, issue_yield, target, face_target_mode
        )
        item = {
            'type': 'TIPS', 'maturity': maturity_years, 'face_amount': face,
            'proceeds': proceeds, 'coupon': coupon, 'issue_price_ratio': price,
            'issue_yield': issue_yield, 'issuance_leg': issuance_leg,
        }
    elif security_type == 'FRN':
        face, proceeds, price = _quote_issuance(
            'FRN', maturity_years, 0.0, np.nan, target, face_target_mode
        )
        item = {
            'type': 'FRN', 'maturity': maturity_years, 'face_amount': face,
            'proceeds': proceeds, 'spread': pricing['frn_spread'], 'issue_price_ratio': price,
            'issue_yield': np.nan, 'issuance_leg': issuance_leg,
        }
    elif security_type == 'Fixed':
        issue_yield = evaluate_nominal_yield(
            maturity_years,
            current_yield_curve_years,
            current_yield_curve_rates,
            method=yield_interpolation_method,
            floor_zero=yield_floor_zero,
            shock=pricing['evaluated_nominal_shock'],
        )
        coupon = calculate_coupon_rate('Fixed', maturity_years, issue_yield, 0)
        bill_discount_rate = None
        if (
            pricing['cbo_funding_mode']
            and bill_term_days == CBO_3M_BILL_DAYS
            and coupon <= TGA_FLOOR_TOLERANCE
        ):
            quote_pct = _cbo_macro_value(
                pricing['cbo_inputs'],
                pricing['cbo_scenario_id'],
                current_date,
                'cbo_3m_tbill_rate_pct',
                np.nan,
            )
            if not pd.isna(quote_pct):
                bill_discount_rate = float(quote_pct) / 100.0
            elif not pd.isna(issue_yield):
                bill_discount_rate = 360.0 * float(issue_yield) / (
                    365.0 + float(issue_yield) * CBO_3M_BILL_DAYS
                )
        face, proceeds, price = _quote_issuance(
            'Fixed', maturity_years, coupon, issue_yield, target, face_target_mode,
            bill_discount_rate=bill_discount_rate,
            bill_days_to_maturity=(
                CBO_3M_BILL_DAYS if bill_days_to_maturity is None else bill_days_to_maturity
            ),
        )
        item = {
            'type': 'Fixed', 'maturity': maturity_years, 'face_amount': face,
            'proceeds': proceeds, 'coupon': coupon, 'issue_price_ratio': price,
            'issue_yield': issue_yield, 'issuance_leg': issuance_leg,
        }
    else:
        raise ValueError(f'Unsupported marketable issuance type: {security_type!r}')
    return item


def _issuance_pricing_context(
    *,
    tips_real_curve_years,
    tips_real_curve_rates,
    current_yield_curve_years,
    current_yield_curve_rates,
    yield_interpolation_method,
    yield_floor_zero,
    evaluated_nominal_shock,
    tips_real_coupon,
    frn_spread,
    cbo_funding_mode,
    cbo_inputs,
    cbo_scenario_id,
):
    return {
        'tips_real_curve_years': tips_real_curve_years,
        'tips_real_curve_rates': tips_real_curve_rates,
        'current_yield_curve_years': current_yield_curve_years,
        'current_yield_curve_rates': current_yield_curve_rates,
        'yield_interpolation_method': yield_interpolation_method,
        'yield_floor_zero': yield_floor_zero,
        'evaluated_nominal_shock': evaluated_nominal_shock,
        'tips_real_coupon': tips_real_coupon,
        'frn_spread': frn_spread,
        'cbo_funding_mode': cbo_funding_mode,
        'cbo_inputs': cbo_inputs,
        'cbo_scenario_id': cbo_scenario_id,
    }


def _bill_share_allocations(amount, *, issuance_profile, controlled_debt, controlled_bills, current_date):
    """Split one amount by the bill-share rule into per-security amounts.

    ``controlled_debt`` and ``controlled_bills`` are face (TIPS at adjusted principal)
    after the day's maturities and any issuance already scheduled or pending. When the
    amount is cash proceeds it enters the rule as if it were face: bills issue at a
    discount of about the bill yield times its term (under 5 bp for a four-week bill,
    about 2% for a 52-week bill at 4%), so bill face exceeds the rule's bill amount by
    that discount and the realized share lands slightly above the target. The tests
    bound this approximation.

    Returns ``([(supply_key, security_type, maturity_years, bill_days_or_None, amount)],
    target)``.
    """

    amount = float(amount or 0.0)
    target = target_for_date(rule_target_path(issuance_profile), current_date)
    if amount <= TGA_FLOOR_TOLERANCE:
        return [], target
    bills_amount = bill_issuance_amount(
        amount,
        target=target,
        controlled_debt=controlled_debt,
        bill_face=controlled_bills,
    )
    coupon_amount = amount - bills_amount
    allocations = []
    for term in issuance_profile['bill_terms']:
        weeks = int(term['weeks'])
        allocations.append((
            bill_supply_key(weeks), 'Fixed', bill_maturity_years(weeks), bill_days(weeks),
            bills_amount * float(term['share']),
        ))
    for term in issuance_profile['coupon_terms']:
        security_type = str(term['security_type'])
        maturity_years = float(term['maturity_years'])
        allocations.append((
            coupon_supply_key(security_type, maturity_years), security_type, maturity_years, None,
            coupon_amount * float(term['share']),
        ))
    residual = amount - sum(item[-1] for item in allocations)
    if abs(residual) > max(TGA_FLOOR_TOLERANCE, abs(amount) * 1.0e-12):
        raise ValueError(f'Bill-share issuance allocation does not close: residual={residual}')
    return allocations, target


def _quote_rule_allocations(allocations, *, face_target_mode, issuance_leg, current_date, pricing):
    """Price rule securities issued on ``current_date`` and date their maturities."""

    schedule = []
    for _key, security_type, maturity_years, weekly_days, target_amount in allocations:
        if target_amount <= TGA_FLOOR_TOLERANCE:
            continue
        item = _quote_supply_item(
            security_type,
            maturity_years,
            target_amount,
            face_target_mode=face_target_mode,
            issuance_leg=issuance_leg,
            current_date=current_date,
            bill_term_days=weekly_days,
            pricing=pricing,
        )
        # Bills mature a whole number of weeks after issue (a 4-week bill in 28 days);
        # coupon securities on the same calendar date ``tenor`` years later.
        item['maturity_date'] = (
            pd.Timestamp(current_date) + timedelta(days=weekly_days)
            if weekly_days is not None
            else _maturity_date_from_years(current_date, maturity_years)
        )
        schedule.append(item)
    return schedule


def _live_frn_accrued_per_face(bond_portfolio, template):
    """Accrued FRN interest per unit face on the live rows of a template's security."""

    rows = bond_portfolio.loc[
        bond_portfolio['Status'].eq('Active') & bond_portfolio['SecurityType'].eq('FRN')
    ]
    ids = fed_purchase_policy.issue_ids(rows, pd.Series(False, index=rows.index))
    rows = rows.loc[ids.eq(template['security_label'])]
    face = float(pd.to_numeric(rows['FaceValue'], errors='coerce').sum())
    if face <= TGA_FLOOR_TOLERANCE:
        raise ValueError(f"no outstanding FRN maturing {template['maturity_date'].date()} to reopen")
    return float(pd.to_numeric(rows['AccruedInterest_FRN'], errors='coerce').fillna(0.0).sum()) / face


def _opening_issue_registry(bond_portfolio, calendar_rules, opening_date):
    """The latest issue of each coupon security in the opening book: what a reopening
    before the run's first new issue of that security reopens.

    A security is the book's latest issue date among rows of its type whose original
    term is within half a year of the tenor. A security the book does not hold has no
    entry; reopening it before its first new issue then raises.
    """

    registry = {}
    active = bond_portfolio[bond_portfolio['Status'] == 'Active']
    issue_dates = pd.to_datetime(active['IssueDate']).dt.normalize()
    years = pd.to_numeric(active['OriginalMaturityYears'], errors='coerce')
    for key, rule in calendar_rules.items():
        if rule.is_bill:
            continue
        security_type, tenor = key
        rows = active[
            (active['SecurityType'] == security_type)
            & ((years - tenor).abs() < 0.5)
            & (issue_dates <= pd.Timestamp(opening_date))
        ]
        if rows.empty:
            continue
        latest = rows.loc[pd.to_datetime(rows['IssueDate']).idxmax()]
        issue_id = fed_purchase_policy.issue_ids(rows.loc[[latest.name]], pd.Series(False, index=[latest.name])).iloc[0]
        maturity = pd.Timestamp(latest['MaturityDate']).normalize()
        dated = maturity - relativedelta(years=int(round(tenor)))
        if rule.dated == 'month_end':
            dated = dated + pd.offsets.MonthEnd(0)
        registry[key] = {
            'maturity_date': maturity,
            'dated_date': dated,
            'coupon': float(latest['CouponRate'] or 0.0),
            'spread': float(latest['FixedSpread'] or 0.0) if security_type == 'FRN' else 0.0,
            'reference_cpi_issue': float(latest['ReferenceCPI_Issue']) if security_type == 'TIPS' else np.nan,
            'security_label': issue_id,
        }
    return registry


def _quote_settlements(
    settled,
    *,
    due_events,
    off_calendar_kind,
    issuance_leg,
    current_date,
    pricing,
    calendar_rules,
    issue_registry,
    bond_portfolio,
    reference_cpi_on,
    frn_accrual_per_face,
):
    """Price the securities settling today on Treasury's terms.

    A calendar settlement is a new issue or a reopening as the calendar says; a
    settlement off the calendar for the cash floor reopens the
    security's latest issue, or is a new issue if none is outstanding yet. A new coupon
    issue is dated on its calendar dated date and matures tenor years later; its coupon
    comes from the issue kernel at the tenor's auction yield. One settling after its
    dated date is priced at that yield from its remaining cash flows at settlement, and
    the buyer pays the interest accrued since the dated date (31 CFR 356 Appendix B,
    II.D; TIPS at the index ratio since then). A
    reopening keeps the reopened security's maturity, coupon, spread and reference CPI
    and is sold at its market value with accrued interest, the valuation of Fed
    secondary transfers. ``issue_price_ratio`` is that dirty value per unit face.
    Updates ``issue_registry`` with each new issue.
    """

    current = pd.Timestamp(current_date).normalize()
    schedule = []
    for key, security_type, maturity_years, weekly_days, amount in settled:
        if amount <= TGA_FLOOR_TOLERANCE:
            continue
        if key == _CASH_MANAGEMENT_BILL_KEY:
            maturity = pd.Timestamp(bill_maturity(current.date(), CASH_MANAGEMENT_BILL_WEEKS))
            days_to_maturity = int((maturity - current).days)
            item = _quote_supply_item(
                'Fixed', days_to_maturity / DAYS_PER_YEAR_ACTUAL, amount,
                face_target_mode=False, issuance_leg=issuance_leg, current_date=current,
                bill_term_days=weekly_days, bill_days_to_maturity=days_to_maturity, pricing=pricing,
            )
            item.update(
                supply_key=key, maturity_date=maturity, dated_date=current,
                security_label=f'{CASH_MANAGEMENT_BILL_KIND}-{current.date()}',
                settlement_kind=CASH_MANAGEMENT_BILL_KIND, is_reopening=False,
            )
            schedule.append(item)
            continue
        event = due_events.get(key)
        settlement_kind = 'calendar' if event is not None else off_calendar_kind[key]
        issue_kind = event.issue if event is not None else 'reopening'
        template = issue_registry.get(key)
        if issue_kind == 'reopening' and (template is None or template['maturity_date'] <= current + pd.Timedelta(days=1)):
            if event is not None:
                raise ValueError(f'{key} reopens on {current.date()} but no issue of it is outstanding')
            issue_kind = 'new'
        quote = functools.partial(
            _quote_supply_item,
            face_target_mode=False,
            issuance_leg=issuance_leg,
            current_date=current,
            pricing=pricing,
        )
        extra = {
            'supply_key': key,
            'is_reopening': issue_kind == 'reopening',
            'settlement_kind': settlement_kind,
            'reference_cpi_issue': np.nan,
            'index_ratio': np.nan,
            'accrued_frn_per_face': 0.0,
        }
        if weekly_days is not None:
            # A bill matures on its weekly cycle (issuance_calibration.bill_maturity),
            # whatever day a holiday moved its settlement to, and is the same
            # security as any bill already maturing that day (Treasury reopens it).
            if issue_kind == 'new':
                nominal = pd.Timestamp(event.nominal if event is not None else current.date())
                maturity = pd.Timestamp(bill_maturity(nominal.date(), int(key[1])))
                issue_registry[key] = {'maturity_date': maturity, 'security_label': f'bill-{maturity.date()}'}
            else:
                maturity = template['maturity_date']
            days = int((maturity - current).days)
            item = quote(
                'Fixed', days / DAYS_PER_YEAR_ACTUAL, amount, bill_term_days=weekly_days, bill_days_to_maturity=days
            )
            item['maturity'] = maturity_years
            outstanding = bond_portfolio[
                (bond_portfolio['Status'] == 'Active')
                & (bond_portfolio['SecurityType'] == 'Fixed')
                & (pd.to_numeric(bond_portfolio['CouponRate'], errors='coerce').fillna(0.0) <= TGA_FLOOR_TOLERANCE)
                & (pd.to_datetime(bond_portfolio['MaturityDate']).dt.normalize() == maturity)
                & ~bond_portfolio.get('IsCashManagementBill', pd.Series(False, index=bond_portfolio.index)).eq(True)
            ]
            ids = fed_purchase_policy.issue_ids(outstanding, pd.Series(True, index=outstanding.index)).unique()
            if len(ids) > 1:
                raise ValueError(f'regular bill reopening on {current.date()} matches distinct issues maturing {maturity.date()}')
            label = ids[0] if len(ids) else f'bill-{maturity.date()}'
            issue_registry[key]['security_label'] = label
            extra['is_reopening'] = issue_kind == 'reopening' or not outstanding.empty
            item.update(extra, maturity_date=maturity, dated_date=current, security_label=label)
            schedule.append(item)
            continue
        if issue_kind == 'new':
            rule = calendar_rules[key]
            nominal = event.nominal if event is not None else current.date()
            dated = pd.Timestamp(dated_date(rule, nominal))
            maturity = pd.Timestamp(maturity_from_dated(rule, dated.date()))
            item = quote(security_type, maturity_years, amount, bill_term_days=None)
            ratio = 1.0
            reference_cpi = np.nan
            if security_type == 'TIPS':
                reference_cpi = float(reference_cpi_on(dated))
                ratio = float(reference_cpi_on(current)) / reference_cpi
            if security_type == 'FRN':
                # Interest accrues from the dated date (a month end); a buyer settling
                # after it pays what has accrued and the holding starts with it.
                accrued = frn_accrual_per_face(dated, current) if dated < current else 0.0
                extra['accrued_frn_per_face'] = accrued
                dirty = float(item['issue_price_ratio']) + accrued
            elif dated < current:
                dirty = float(
                    value_treasury_security(
                        maturity_date=maturity,
                        settlement_date=current,
                        coupon_rate=float(item['coupon']),
                        discount_yield=float(item['issue_yield']),
                        security_type=security_type,
                        adjusted_principal=ratio,
                        original_principal=1.0,
                        issue_date=dated,
                        nominal_discount_yield=(
                            evaluate_nominal_yield(
                                maturity_years,
                                pricing['current_yield_curve_years'],
                                pricing['current_yield_curve_rates'],
                                method=pricing['yield_interpolation_method'],
                                floor_zero=pricing['yield_floor_zero'],
                                shock=pricing['evaluated_nominal_shock'],
                            )
                            if security_type == 'TIPS' else None
                        ),
                    )['dirty']
                )
            else:
                dirty = float(item['issue_price_ratio']) * ratio
            issue_registry[key] = {
                'maturity_date': maturity,
                'dated_date': dated,
                'coupon': float(item.get('coupon', 0.0) or 0.0),
                'spread': float(item.get('spread', 0.0) or 0.0),
                'reference_cpi_issue': reference_cpi,
                'security_label': f"{security_type.lower()}-{key[1]:g}y-{maturity.date()}",
            }
            template = issue_registry[key]
            extra.update(reference_cpi_issue=reference_cpi, index_ratio=ratio)
        else:
            maturity = template['maturity_date']
            dated = template['dated_date']
            ratio = 1.0
            unit = {
                'SecurityType': security_type,
                'MaturityDate': maturity,
                'CouponRate': template['coupon'],
                'IssueDate': dated,
                'FirstInterestPaymentDate': pd.NaT,
                'FaceValue': 1.0,
                'OriginalPrincipal': 1.0,
                'AccruedInterest_FRN': 0.0,
            }
            if security_type == 'TIPS':
                ratio = float(reference_cpi_on(current)) / float(template['reference_cpi_issue'])
                unit['AdjustedPrincipal'] = ratio
                extra.update(reference_cpi_issue=float(template['reference_cpi_issue']), index_ratio=ratio)
            elif security_type == 'FRN':
                unit['AccruedInterest_FRN'] = _live_frn_accrued_per_face(bond_portfolio, template)
                extra['accrued_frn_per_face'] = unit['AccruedInterest_FRN']
            valuation = _fed_secondary_dirty_value(
                pd.Series(unit),
                ratio,
                current,
                pricing['current_yield_curve_years'],
                pricing['current_yield_curve_rates'],
                interpolation_method=pricing['yield_interpolation_method'],
                floor_zero=pricing['yield_floor_zero'],
                nominal_shock=pricing['evaluated_nominal_shock'],
                tips_real_curve_years=pricing['tips_real_curve_years'],
                tips_real_curve_rates=pricing['tips_real_curve_rates'],
            )
            dirty = float(valuation['dirty_value'])
            item = {
                'type': security_type,
                'maturity': maturity_years,
                'coupon': template['coupon'],
                'spread': template['spread'],
                'issue_yield': float(valuation['discount_yield']),
                'issuance_leg': issuance_leg,
            }
        face = float(amount) / dirty
        item.update(
            extra,
            face_amount=face,
            proceeds=float(amount),
            issue_price_ratio=dirty,
            maturity_date=maturity,
            dated_date=dated,
            security_label=template['security_label'],
        )
        schedule.append(item)
    return schedule


def _supply_terms(key, issuance_profile):
    """(security type, maturity years, bill days or None) of a rule security's supply key."""

    if key[0] == 'bill':
        weeks = int(key[1])
        return 'Fixed', bill_maturity_years(weeks), bill_days(weeks)
    for term in issuance_profile['coupon_terms']:
        if coupon_supply_key(term['security_type'], term['maturity_years']) == key:
            return str(term['security_type']), float(term['maturity_years']), None
    raise ValueError(f'{key} is not a security of the issuance rule')


def _settlement_unit_prices(keys, *, issuance_profile, due_events, current_date, pricing, calendar_rules,
                            issue_registry, bond_portfolio, reference_cpi_on, frn_accrual_per_face):
    """Per unit face of each security if it settled today: (cash, stock added), leaving the
    register as it is. Stock is adjusted principal for TIPS and face otherwise."""

    items = _quote_settlements(
        [(key, *_supply_terms(key, issuance_profile), 1.0) for key in keys],
        due_events=due_events,
        off_calendar_kind={},
        issuance_leg=FED_ROLLOVER_ISSUANCE_LEG,
        current_date=current_date,
        pricing=pricing,
        calendar_rules=calendar_rules,
        issue_registry=copy.deepcopy(issue_registry),
        bond_portfolio=bond_portfolio,
        reference_cpi_on=reference_cpi_on,
        frn_accrual_per_face=frn_accrual_per_face,
    )
    return {
        item['supply_key']: (
            float(item['issue_price_ratio']),
            float(item['index_ratio']) if item['type'] == 'TIPS' else 1.0,
        )
        for item in items
    }


def _fed_rollover_allocation(stock_by_class, *, offering_face, issuance_profile, prices):
    """Allocate each class's rollover stock in proportion to the public offering face.

    TIPS count at adjusted principal against the class total. If a class has calendar
    auctions but no public issuance today, use the calibrated shares for Fed-only
    add-ons. A class with no calendar auctions today is redeemed.
    """
    allocation = {}
    fed_only_classes = []
    for security_class, stock in stock_by_class.items():
        if stock <= TGA_FLOOR_TOLERANCE:
            continue
        keys = [key for key in sorted(prices) if (key[0] == 'bill') == (security_class == 'bills')]
        if not keys:
            continue
        weights = {key: offering_face.get(key, 0.0) for key in keys}
        if sum(weights.values()) <= TGA_FLOOR_TOLERANCE:
            fed_only_classes.append(security_class)
            if security_class == 'bills':
                shares = {bill_supply_key(int(term['weeks'])): float(term['share']) for term in issuance_profile['bill_terms']}
            else:
                shares = {
                    coupon_supply_key(term['security_type'], term['maturity_years']): float(term['share'])
                    for term in issuance_profile['coupon_terms']
                }
            weights = {key: shares[key] for key in keys}
        scale = stock / sum(weights[key] * prices[key][1] for key in keys)
        for key in keys:
            if weights[key] > 0.0:
                allocation[key] = (scale * weights[key], *prices[key])
    return allocation, fed_only_classes

def _build_bill_share_supply_schedule(
    amount,
    *,
    face_target_mode,
    issuance_leg,
    issuance_profile,
    controlled_debt,
    controlled_bills,
    current_date,
    pricing,
):
    """Split one issuance amount by the bill-share rule and price it the same day."""

    allocations, target = _bill_share_allocations(
        amount,
        issuance_profile=issuance_profile,
        controlled_debt=controlled_debt,
        controlled_bills=controlled_bills,
        current_date=current_date,
    )
    return _quote_rule_allocations(
        allocations,
        face_target_mode=face_target_mode,
        issuance_leg=issuance_leg,
        current_date=current_date,
        pricing=pricing,
    ), target


def _settlement_schedule(issuance_profile, start_date, end_date):
    """Map each settlement business day in ``[start_date, end_date]`` to the securities
    settling then, each with its calendar event (new issue or reopening, nominal date)."""

    by_date = defaultdict(dict)
    for event in settlement_events(
        issuance_profile,
        pd.Timestamp(start_date).date(),
        pd.Timestamp(end_date).date(),
    ):
        by_date[pd.Timestamp(event.settle)][event.key] = event
    return dict(sorted(by_date.items()))


def _build_issuance_supply_schedule(
    amount,
    *,
    face_target_mode,
    issuance_leg,
    issuance_profile,
    current_date,
    pricing,
):
    """Build one priced issuance leg from fixed gross shares, without mutating state."""

    amount = float(amount or 0.0)
    if amount <= TGA_FLOOR_TOLERANCE:
        return []
    tips_pct = issuance_profile.get('TIPS', {}).get('target_percentage', 0.0)
    frn_pct = issuance_profile.get('FRN', {}).get('target_percentage', 0.0)
    nonmkt_pct = issuance_profile.get('NonMarketable', {}).get('target_percentage', 0.0)
    if pricing['cbo_funding_mode']:
        nonmkt_pct = 0.0
    issued_tips = amount * tips_pct
    issued_frn = amount * frn_pct
    issued_nonmkt = amount * nonmkt_pct
    fixed_amount = max(0.0, amount - issued_tips - issued_frn - issued_nonmkt)
    schedule = []

    def _profile_terms(profile, label):
        maturities = profile.get('maturities', [])
        distribution = profile.get('maturity_distribution', [])
        if not maturities or not distribution or len(maturities) != len(distribution):
            raise ValueError(
                f'{label} issuance requires matching non-empty maturities and '
                'maturity_distribution.'
            )
        try:
            maturities = [float(value) for value in maturities]
            distribution = [float(value) for value in distribution]
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label} issuance terms must be numeric.') from exc
        if any(value <= 0.0 for value in maturities):
            raise ValueError(f'{label} issuance maturities must be positive.')
        if any(value < 0.0 for value in distribution):
            raise ValueError(f'{label} issuance weights must be nonnegative.')
        total = sum(distribution)
        if total <= TGA_FLOOR_TOLERANCE:
            raise ValueError(f'{label} issuance weights must sum to a positive value.')
        return maturities, [value / total for value in distribution]

    def _quote(security_type, maturity_years, target, bill_term_days=None):
        item = _quote_supply_item(
            security_type,
            maturity_years,
            target,
            face_target_mode=face_target_mode,
            issuance_leg=issuance_leg,
            current_date=current_date,
            bill_term_days=bill_term_days,
            pricing=pricing,
        )
        item['maturity_date'] = _maturity_date_from_years(current_date, maturity_years)
        return item

    if issued_tips > TGA_FLOOR_TOLERANCE:
        maturities, normalized = _profile_terms(issuance_profile.get('TIPS', {}), 'TIPS')
        for maturity_years, weight in zip(maturities, normalized):
            target = issued_tips * weight
            if target <= TGA_FLOOR_TOLERANCE:
                continue
            schedule.append(_quote('TIPS', maturity_years, target))

    if issued_frn > TGA_FLOOR_TOLERANCE:
        maturities, normalized = _profile_terms(issuance_profile.get('FRN', {}), 'FRN')
        for maturity_years, weight in zip(maturities, normalized):
            target = issued_frn * weight
            if target <= TGA_FLOOR_TOLERANCE:
                continue
            schedule.append(_quote('FRN', maturity_years, target))

    fixed_allocations = {}
    fixed_category_shares = {
        category: float(
            issuance_profile.get(category, {}).get(
                'target_percentage_of_remainder', 0.0
            )
        )
        for category in MATURITY_CATEGORIES
    }
    fixed_share_total = sum(fixed_category_shares.values())
    if fixed_amount > TGA_FLOOR_TOLERANCE:
        if fixed_share_total <= TGA_FLOOR_TOLERANCE:
            raise ValueError('Fixed issuance requires a positive maturity-category allocation.')
        for category in MATURITY_CATEGORIES:
            profile = issuance_profile.get(category, {})
            category_amount = (
                fixed_amount * fixed_category_shares[category] / fixed_share_total
            )
            if category_amount <= TGA_FLOOR_TOLERANCE:
                continue
            maturities, normalized = _profile_terms(profile, category)
            for maturity_years, weight in zip(maturities, normalized):
                target = category_amount * weight
                if target > TGA_FLOOR_TOLERANCE:
                    fixed_allocations[maturity_years] = (
                        fixed_allocations.get(maturity_years, 0.0) + target
                    )
    unallocated = fixed_amount - sum(fixed_allocations.values())
    allocation_tolerance = max(TGA_FLOOR_TOLERANCE, abs(fixed_amount) * 1.0e-12)
    if abs(unallocated) > allocation_tolerance:
        raise ValueError(
            f'Fixed issuance maturity allocation does not close: residual={unallocated}'
        )
    for maturity_years, target in fixed_allocations.items():
        if target <= TGA_FLOOR_TOLERANCE:
            continue
        # Month-tenor profiles date a 0.25-year bill three calendar months out but quote it
        # as the 91-day CBO bill; the weekly-term rule dates bills by days instead.
        quote_days = CBO_3M_BILL_DAYS if abs(float(maturity_years) - 0.25) <= 1e-12 else None
        schedule.append(_quote('Fixed', maturity_years, target, bill_term_days=quote_days))

    if issued_nonmkt > TGA_FLOOR_TOLERANCE:
        maturity_years = issuance_profile.get('NonMarketable', {}).get(
            'nominal_maturity_years', 30.0
        )
        face, proceeds, price = _quote_issuance(
            'NonMarketable', maturity_years, 0.0, np.nan, issued_nonmkt, face_target_mode
        )
        schedule.append({
            'type': 'NonMarketable', 'maturity': maturity_years, 'face_amount': face,
            'proceeds': proceeds, 'coupon': 0.0, 'issue_price_ratio': price,
            'issue_yield': np.nan, 'issuance_leg': issuance_leg,
            'maturity_date': _maturity_date_from_years(current_date, maturity_years),
        })
    return schedule


def _fiscal_year_close_dates(issuance_profile, start_date, end_date):
    """Last business days of the configured close months in the run window."""

    return [
        pd.Timestamp(value)
        for value in fiscal_year_close_dates(
            issuance_profile,
            pd.Timestamp(start_date).date(),
            pd.Timestamp(end_date).date(),
        )
    ]


def _due_settlement_events(settlement_schedule, settlement_dates, prev_date, current_date):
    """Calendar events of the securities whose settlement date falls in ``(prev_date,
    current_date]``, keyed by security; a security due twice keeps its later event."""

    low = bisect.bisect_right(settlement_dates, pd.Timestamp(prev_date))
    high = bisect.bisect_right(settlement_dates, pd.Timestamp(current_date))
    due = {}
    for settle in settlement_dates[low:high]:
        due.update(settlement_schedule[settle])
    return due


def _last_settlements_before_close(settlement_schedule, close_dates):
    """Last calendar settlement of each key in each fiscal-year segment."""
    schedule = iter(settlement_schedule.items())
    next_event = next(schedule, None)
    last_by_close = {}
    for close in close_dates:
        last = {}
        while next_event is not None and next_event[0] <= close:
            day, events = next_event
            last.update(dict.fromkeys(events, day))
            next_event = next(schedule, None)
        last_by_close[close] = last
    return last_by_close


def _redirect_financing_before_close(allocations, issuance_profile, last_settlements, current_date):
    """Redirect a lapsed security's need to regular bills available through the close."""
    available = {key for key, day in last_settlements.items() if day >= current_date}
    eligible = [item for item in allocations if item[0] in available]
    stranded = sum(item[-1] for item in allocations if item[0] not in available)
    if stranded <= TGA_FLOOR_TOLERANCE:
        return eligible
    bills = [
        (term, bill_supply_key(int(term['weeks']))) for term in issuance_profile['bill_terms']
        if bill_supply_key(int(term['weeks'])) in available
    ]
    if not bills:
        weeks = CASH_MANAGEMENT_BILL_WEEKS
        return [*eligible, (_CASH_MANAGEMENT_BILL_KEY, 'Fixed', bill_maturity_years(weeks), bill_days(weeks), stranded)]
    shares = sum(float(term['share']) for term, _ in bills)
    for term, key in bills:
        proceeds = stranded * float(term['share']) / shares
        eligible.append((key, *_supply_terms(key, issuance_profile), proceeds))
    return eligible


def _decide_and_settle_pending(
    pending,
    *,
    need,
    issuance_profile,
    pre_issuance_position,
    due_keys,
    projected_tga_without_settlement,
    current_date,
    last_settlements=None,
    fiscal_close_today=False,
    can_settle=True,
):
    """Add the day's split to pending balances and release the ones due today.

    The split counts pending face as already issued. If settling only what is due would
    leave the TGA negative, every pending bill settles early the same day, and pending
    coupon securities too if the bills are not enough. Mutates ``pending``; returns the
    settling allocations and proceeds settled early. Any cash-management bill
    decided after the last regular bill settlement settles on the fiscal close.
    """

    controlled_debt, controlled_bills = pre_issuance_position
    for security_type, maturity_years, _weekly_days, amount in pending.values():
        controlled_debt += amount
        if security_type == 'Fixed' and maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE:
            controlled_bills += amount
    allocations, _target = _bill_share_allocations(
        need,
        issuance_profile=issuance_profile,
        controlled_debt=controlled_debt,
        controlled_bills=controlled_bills,
        current_date=current_date,
    )
    if last_settlements is not None:
        allocations = _redirect_financing_before_close(
            allocations, issuance_profile, last_settlements, current_date,
        )
    for key, security_type, maturity_years, weekly_days, amount in allocations:
        if amount <= TGA_FLOOR_TOLERANCE:
            continue
        if key in pending:
            pending[key][3] += amount
        else:
            pending[key] = [security_type, maturity_years, weekly_days, amount]
    if not can_settle:
        # The securities market is closed: nothing settles, so the TGA must not need it.
        if due_keys or fiscal_close_today or projected_tga_without_settlement < 0.0:
            raise RuntimeError(
                f'issuance would have to settle on {pd.Timestamp(current_date).date()}, '
                'a day the securities market is closed'
            )
        return [], 0.0
    settling = [key for key in due_keys if key in pending]
    if fiscal_close_today:
        if _CASH_MANAGEMENT_BILL_KEY in pending:
            settling.append(_CASH_MANAGEMENT_BILL_KEY)
        stranded_keys = set(pending).difference(settling)
        if stranded_keys:
            raise RuntimeError(f'Pending financing missed the fiscal close on {current_date.date()}: {sorted(stranded_keys)}')
    settled_cash = sum(pending[key][3] for key in settling)
    early = 0.0
    # Bills first; coupon securities settle early only if the bills cannot keep the TGA
    # from going negative.
    for early_kind in ('bill', 'coupon'):
        if projected_tga_without_settlement + settled_cash + early >= 0.0:
            break
        for key, item in pending.items():
            if (key[0] == 'bill') == (early_kind == 'bill') and key not in settling and key != _CASH_MANAGEMENT_BILL_KEY:
                settling.append(key)
                early += item[3]
    settled = []
    for key in settling:
        security_type, maturity_years, weekly_days, amount = pending.pop(key)
        settled.append((key, security_type, maturity_years, weekly_days, amount))
    return settled, early


def _schedule_issuance_leg(
    amount,
    *,
    face_target_mode,
    issuance_leg,
    issuance_profile,
    current_date,
    pricing,
    pre_issuance_position,
    scheduled,
):
    """Build one issuance leg under the profile's rule.

    For the bill-share rule, items already ``scheduled`` that day count toward debt and
    bills, so a second leg sees the first leg's issuance.
    """

    if not is_rule_profile(issuance_profile):
        return _build_issuance_supply_schedule(
            amount,
            face_target_mode=face_target_mode,
            issuance_leg=issuance_leg,
            issuance_profile=issuance_profile,
            current_date=current_date,
            pricing=pricing,
        )
    controlled_debt, controlled_bills = pre_issuance_position
    for item in scheduled:
        if item['type'] not in {'Fixed', 'TIPS', 'FRN'}:
            continue
        controlled_debt += float(item['face_amount'])
        if item['type'] == 'Fixed' and float(item['maturity']) <= 1.0 + TGA_FLOOR_TOLERANCE:
            controlled_bills += float(item['face_amount'])
    schedule, _target = _build_bill_share_supply_schedule(
        amount,
        face_target_mode=face_target_mode,
        issuance_leg=issuance_leg,
        issuance_profile=issuance_profile,
        controlled_debt=controlled_debt,
        controlled_bills=controlled_bills,
        current_date=current_date,
        pricing=pricing,
    )
    return schedule


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
    if is_rule_profile(issuance_profile):
        raise ValueError('rate-sensitive demand is not supported with the bill-share issuance rule')
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
    nominal_shock=None,
):
    params = _get_rate_sensitive_multipliers_config(rate_sensitive_params)
    anchor_mat = max(0.0, float(params.get('anchor_maturity_years', 5.0) or 5.0))
    slope_short = max(0.0, float(params.get('slope_short_maturity_years', 2.0) or 2.0))
    slope_long = max(0.0, float(params.get('slope_long_maturity_years', 10.0) or 10.0))
    anchor_yield = evaluate_nominal_yield(
        anchor_mat,
        yield_curve_years,
        yield_curve_rates,
        method=interpolation_method,
        floor_zero=floor_zero,
        shock=nominal_shock,
    )
    short_yield = evaluate_nominal_yield(
        slope_short,
        yield_curve_years,
        yield_curve_rates,
        method=interpolation_method,
        floor_zero=floor_zero,
        shock=nominal_shock,
    )
    long_yield = evaluate_nominal_yield(
        slope_long,
        yield_curve_years,
        yield_curve_rates,
        method=interpolation_method,
        floor_zero=floor_zero,
        shock=nominal_shock,
    )
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

    active_categories = _get_active_preference_categories(issuance_profile)

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
    special = active_special_categories(issuance_profile)
    return ['bills', 'notes', 'bonds', *[category for category in ('tips', 'frn') if category in special]]


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


def _normalize_holder_value(value) -> str:
    return '' if pd.isna(value) else str(value)


def _ensure_tdc_principal_route_columns(bond_portfolio: pd.DataFrame) -> pd.DataFrame:
    if bond_portfolio is None or bond_portfolio.empty:
        return bond_portfolio
    if 'TDCPrincipalHolderType' not in bond_portfolio.columns:
        bond_portfolio['TDCPrincipalHolderType'] = bond_portfolio['HolderType']
    if 'TDCPrincipalHolderSubBucket' not in bond_portfolio.columns:
        bond_portfolio['TDCPrincipalHolderSubBucket'] = bond_portfolio['HolderSubBucket']
    route_holder = bond_portfolio['TDCPrincipalHolderType'].fillna('').astype(str).replace({'<NA>': '', 'nan': '', 'None': ''})
    route_subbucket = bond_portfolio['TDCPrincipalHolderSubBucket'].fillna('').astype(str).replace({'<NA>': '', 'nan': '', 'None': ''})
    missing_route = route_holder.eq('')
    route_holder = route_holder.mask(missing_route, bond_portfolio['HolderType'].fillna('').astype(str))
    holder_subbucket = bond_portfolio['HolderSubBucket'].fillna('').astype(str).replace({'<NA>': '', 'nan': '', 'None': ''})
    route_subbucket = route_subbucket.mask(missing_route | route_subbucket.eq(''), holder_subbucket)
    private_route = route_holder.eq('Private')
    valid_route_subbucket = route_subbucket.isin(PRIVATE_SUBBUCKETS)
    route_subbucket = route_subbucket.mask(private_route & ~valid_route_subbucket, PRIVATE_SUBBUCKET_DOMESTIC_NONBANK)
    route_subbucket = route_subbucket.mask(~private_route, '')
    bond_portfolio['TDCPrincipalHolderType'] = route_holder
    bond_portfolio['TDCPrincipalHolderSubBucket'] = route_subbucket
    return bond_portfolio


def _tdc_principal_holder(row_or_mapping) -> dict[str, str]:
    if not hasattr(row_or_mapping, 'get'):
        return {'holder_sector': '', 'holder_subsector': ''}
    current_holder = _normalize_holder_value(row_or_mapping.get('HolderType', ''))
    if current_holder == 'CB':
        return {'holder_sector': 'CB', 'holder_subsector': ''}
    holder = _normalize_holder_value(row_or_mapping.get('TDCPrincipalHolderType', current_holder))
    subbucket = _normalize_holder_value(
        row_or_mapping.get('TDCPrincipalHolderSubBucket', row_or_mapping.get('HolderSubBucket', ''))
    )
    if holder == 'Private':
        subbucket = subbucket if subbucket in PRIVATE_SUBBUCKETS else PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
    else:
        subbucket = ''
    return {'holder_sector': holder, 'holder_subsector': subbucket}


def _private_subbucket_split(preferences: dict, category: str) -> dict[str, float]:
    subbucket_map = preferences.get(PRIVATE_SUBBUCKET_SHARES_KEY, {}) if isinstance(preferences, dict) else {}
    category_map = subbucket_map.get(category, {}) if isinstance(subbucket_map, dict) else {}
    if PRIVATE_SUBBUCKET_MMF_ADDED in category_map:
        raise ValueError('The added money-fund position cannot appear in auction holder preferences.')
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
    *,
    mmf_added_deposit_share: float = 0.0,
) -> float:
    added = route_totals.get(PRIVATE_SUBBUCKET_MMF_ADDED, 0.0)
    return (
        route_totals.get(PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, 0.0)
        + mmf_pass_through * route_totals.get(PRIVATE_SUBBUCKET_MMF, 0.0)
        + mmf_added_deposit_share * added
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
    # Coupon dates follow the maturity date (sim_pricing.coupon_payment_date); the
    # first interest date only sets when payments start.
    months_between = (current.year - maturity_dates.dt.year) * 12 + (current.month - maturity_dates.dt.month)
    expected_day = np.where(
        maturity_dates.dt.is_month_end.fillna(False),
        current.days_in_month,
        np.minimum(maturity_dates.dt.day.fillna(0).astype(int), current.days_in_month),
    )
    due = (
        issue_dates.notna()
        & maturity_dates.notna()
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
    if frame is None or frame.empty:
        return None
    rows = _cbo_rows_for_scenario(frame, scenario_id)
    if rows is None or rows.empty:
        return None
    selected_scenario = str(rows['scenario_id'].fillna('').astype(str).iloc[0])
    if isinstance(lookup, dict):
        row = lookup.get((selected_scenario, target))
        if row is not None:
            return row
    starts = pd.to_datetime(rows['period_start'], errors='coerce').dt.normalize()
    ends = pd.to_datetime(rows['period_end'], errors='coerce').dt.normalize()
    covering = rows.loc[(starts < target) & (target <= ends)].copy()
    if covering.empty:
        return None
    durations = (ends.loc[covering.index] - starts.loc[covering.index]).dt.days
    covering = covering.assign(_duration_days=durations, _period_end=ends.loc[covering.index])
    return covering.sort_values(['_duration_days', '_period_end']).iloc[0]


def _frn_reference_rate_for_date(
    cbo_inputs,
    scenario_id,
    date_value,
    *,
    benchmark_maturity_years,
    yield_curve_years,
    yield_curve_rates,
    interpolation_method,
    floor_zero,
    nominal_shock=None,
):
    frn_path_frame = (
        cbo_inputs.get('frn_rate_path')
        if isinstance(cbo_inputs, dict)
        else pd.DataFrame()
    )
    frn_path_date_lookup = (
        cbo_inputs.get('frn_rate_path_date_lookup', {})
        if isinstance(cbo_inputs, dict)
        else {}
    )
    if isinstance(frn_path_frame, pd.DataFrame) and not frn_path_frame.empty:
        row = _frn_rate_path_row_for_date(
            frn_path_frame,
            scenario_id,
            date_value,
            lookup=frn_path_date_lookup,
        )
        if row is not None:
            return float(row['benchmark_rate_decimal'])
    if nominal_shock is not None:
        raise CurveContractError(
            'OPEN-04 evaluated nominal shock requires complete explicit FRN benchmark coverage'
        )
    rate = get_yield_for_maturity(
        benchmark_maturity_years,
        yield_curve_years,
        yield_curve_rates,
        method=interpolation_method,
        floor_zero=floor_zero,
    )
    return np.nan if pd.isna(rate) else float(rate)


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


def _maturity_date_from_years(issue_date, maturity_years):
    total_months = int(round(float(maturity_years) * 12.0))
    years, months = divmod(max(total_months, 0), 12)
    return pd.Timestamp(issue_date) + relativedelta(years=years, months=months)


def _reference_cpi_at_event(
    event_date,
    *,
    opening_date,
    reference_lag_months,
    annual_inflation,
    results,
    cbo_reference_lookup,
):
    event_day = pd.Timestamp(event_date).normalize()
    if cbo_reference_lookup:
        try:
            return float(cbo_reference_lookup[event_day])
        except KeyError as exc:
            raise ValueError(
                f'TIPS reference CPI path has no exact event-date value for {event_day.date()}.'
            ) from exc
    reference_day = event_day - relativedelta(months=int(reference_lag_months))
    opening_day = pd.Timestamp(opening_date).normalize()
    if reference_day <= opening_day:
        return float(results.loc[opening_day, 'CPI_Level'])
    prior_dates = results.index[results.index <= reference_day]
    if prior_dates.empty:
        return float(results.loc[opening_day, 'CPI_Level'])
    anchor_day = pd.Timestamp(prior_dates[-1]).normalize()
    anchor_cpi = float(results.loc[anchor_day, 'CPI_Level'])
    elapsed_years = (reference_day - anchor_day).days / DAYS_PER_YEAR_ACTUAL
    return anchor_cpi * (1.0 + float(annual_inflation)) ** elapsed_years


def _capitalize_nonmarketable_interest(
    bond_portfolio,
    prev_date,
    current_date,
    nonmarketable_params,
    yield_curve_years,
    yield_curve_rates,
    *,
    interpolation_method,
    floor_zero,
    nominal_shock=None,
    handoff_tables=None,
):
    frequency = str(
        nonmarketable_params.get('interest_crediting_frequency', 'semi-annual')
    ).strip().lower()
    credit_dates = []
    for year in range(pd.Timestamp(prev_date).year, pd.Timestamp(current_date).year + 1):
        for month, day in ((6, 30), (12, 31)):
            if frequency == 'annual' and month != 12:
                continue
            if frequency not in {'annual', 'semi-annual'}:
                continue
            credit_date = pd.Timestamp(year=year, month=month, day=day)
            if pd.Timestamp(prev_date) < credit_date <= pd.Timestamp(current_date):
                credit_dates.append(credit_date)
    total_credit = 0.0
    for credit_date in credit_dates:
        maturity_dates = pd.to_datetime(bond_portfolio['MaturityDate'], errors='coerce')
        eligible = (
            bond_portfolio['Status'].eq('Active')
            & bond_portfolio['SecurityType'].eq('NonMarketable')
            & (maturity_dates.isna() | maturity_dates.ge(credit_date))
        )
        if not eligible.any():
            continue
        method = nonmarketable_params.get('rate_setting_method', 'yield_curve_points')
        rate = 0.0
        if method == 'avg_outstanding_marketable_yield':
            minimum_maturity = float(
                nonmarketable_params.get('marketable_basket_min_remaining_maturity', 4.0)
            )
            basket_types = nonmarketable_params.get(
                'marketable_basket_types',
                ['Fixed', 'TIPS'],
            )
            basket = bond_portfolio[
                bond_portfolio['Status'].eq('Active')
                & bond_portfolio['SecurityType'].isin(basket_types)
            ].copy()
            if not basket.empty:
                basket['CurrentTTM'] = (
                    pd.to_datetime(basket['MaturityDate'], errors='coerce') - credit_date
                ).dt.total_seconds() / (DAYS_PER_YEAR_ACTUAL * 24 * 60 * 60)
                basket = basket[basket['CurrentTTM'] >= minimum_maturity]
                if not basket.empty:
                    yields = basket['CurrentTTM'].apply(
                        lambda maturity: evaluate_nominal_yield(
                            maturity,
                            yield_curve_years,
                            yield_curve_rates,
                            method=interpolation_method,
                            floor_zero=floor_zero,
                            shock=nominal_shock,
                        )
                    ).dropna()
                    if not yields.empty:
                        if (
                            nonmarketable_params.get('marketable_basket_weighting', 'equal')
                            == 'face_value'
                        ):
                            weights = pd.to_numeric(
                                basket.loc[yields.index, 'FaceValue'],
                                errors='coerce',
                            ).fillna(0.0)
                            rate = (
                                float(np.average(yields, weights=weights))
                                if float(weights.sum()) > TGA_FLOOR_TOLERANCE
                                else float(yields.mean())
                            )
                        else:
                            rate = float(yields.mean())
        else:
            maturities = nonmarketable_params.get(
                'interest_rate_basis_maturities',
                [5.0, 10.0],
            )
            yields = [
                evaluate_nominal_yield(
                    maturity,
                    yield_curve_years,
                    yield_curve_rates,
                    method=interpolation_method,
                    floor_zero=floor_zero,
                    shock=nominal_shock,
                )
                for maturity in maturities
            ]
            finite_yields = [float(value) for value in yields if not pd.isna(value)]
            rate = float(np.mean(finite_yields)) if finite_yields else 0.0
        factor = 0.5 if frequency == 'semi-annual' else 1.0
        credit = (
            pd.to_numeric(bond_portfolio.loc[eligible, 'FaceValue'], errors='coerce')
            .fillna(0.0)
            * max(0.0, rate)
            * factor
        )
        if handoff_tables is not None:
            for index, amount in credit.items():
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='intragovernmental',
                    leg_type='nonmarketable_interest_capitalization',
                    accounting_basis='face_stock_noncash_capitalization',
                    row=bond_portfolio.loc[index],
                    face_stock_change=float(amount),
                    settlement_scope='consolidated_noncash_intragovernmental',
                    is_intragovernmental=True,
                )
        bond_portfolio.loc[eligible, 'FaceValue'] += credit
        total_credit += float(credit.sum())
    return total_credit


def run_simulation(
    params,
    start_date,
    end_date,
    freq='W',
    scenario_name='Default',
    *,
    handoff_sink=None,
    require_bounded_handoff=False,
):
    """
    Runs the core economic simulation including Treasury operations, fiscal flows,
    central bank actions, secondary market trading, AND handles date-based events
    defined in the configuration to dynamically change parameters.
    Tracks key monetary aggregates and debt composition.
    """
    sim_start_time = time.time()
    sim_mode = str(params.get('simulation_period', {}).get('mode', 'forward')).strip().lower()
    yield_surface_cfg = params.get('yield_curve_surface', {}) if isinstance(params.get('yield_curve_surface', {}), dict) else {}
    evaluated_nominal_shock = _evaluated_nominal_shock_from_config(yield_surface_cfg)
    if sim_mode == 'historical_replay':
        if evaluated_nominal_shock is not None:
            raise CurveContractError(
                'evaluated nominal shock is supported only by the forward CBO runtime'
            )
        from historical_replay import run_historical_replay

        return run_historical_replay(params, start_date, end_date, scenario_name=scenario_name)
    validate_run_params(params, scenario_name=scenario_name)
    funding_rule_cfg = params.get('funding_rule', {})
    funding_rule_mode = str(funding_rule_cfg.get('mode', '')).strip()
    cbo_funding_mode = funding_rule_mode in CBO_FUNDING_MODES
    cbo_reference_financing_mode = (
        funding_rule_mode == CBO_DEBT_REFERENCE_FINANCING_MODE
    )
    cbo_operating_cash_financing_mode = (
        funding_rule_mode == CBO_OPERATING_CASH_TARGET_FINANCING_MODE
    )
    # Only the hard-target mode issues (or retires) to land on the CBO debt path.
    cbo_debt_target_mode = funding_rule_mode == CBO_FUNDING_MODE
    yield_interpolation_method = str(
        yield_surface_cfg.get('interpolation_method', 'pchip' if cbo_funding_mode else 'linear')
    ).strip().lower()
    yield_floor_zero = bool(yield_surface_cfg.get('floor_zero', not cbo_funding_mode))
    if evaluated_nominal_shock is not None:
        if not cbo_funding_mode:
            raise CurveContractError(
                'evaluated nominal shock requires CBO funding mode'
            )
        if yield_interpolation_method != 'pchip' or yield_floor_zero is not False:
            raise CurveContractError(
                'evaluated nominal shock requires baseline pchip with floor_zero=false'
            )
        rate_sensitive_cfg = params.get('rate_sensitive_demand', {})
        if not isinstance(rate_sensitive_cfg, dict):
            raise CurveContractError(
                'evaluated nominal shock requires a disabled rate_sensitive_demand mapping'
            )
        if (
            'enabled' not in rate_sensitive_cfg
            or rate_sensitive_cfg.get('enabled') is not False
        ):
            raise CurveContractError(
                'evaluated nominal shock requires rate_sensitive_demand.enabled=false'
            )
        simulation_period_cfg = params.get('simulation_period', {})
        if not isinstance(simulation_period_cfg, dict):
            raise CurveContractError(
                'evaluated nominal shock requires a simulation_period mapping'
            )
        if (
            'enable_preference_trading' not in simulation_period_cfg
            or simulation_period_cfg.get('enable_preference_trading') is not False
        ):
            raise CurveContractError(
                'evaluated nominal shock requires simulation_period.enable_preference_trading=false'
            )
        raw_events = params.get('events', [])
        if not isinstance(raw_events, list) or raw_events:
            raise CurveContractError(
                'evaluated nominal shock requires an empty events list'
            )
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
        raise ValueError(f'[{scenario_name}] Failed to create simulation date range: {e}') from e
    if num_periods <= 1:
        raise ValueError(
            f'[{scenario_name}] Simulation period must contain at least one running period '
            f'after the opening date; received {num_periods} time step(s).'
        )
    print(f'--- Starting Simulation: {scenario_name} ---')
    results_cols = ['GovSpending', 'Taxes', 'PrimaryDeficit', 'InterestPaid_Bonds', 'PrincipalPaid_Bonds', 'InterestOutlay_Period', 'InterestOutlay_Cumulative', 'PrincipalRollover_Period', 'PrincipalRollover_Cumulative', 'NewDebtIssued', 'AuctionProceeds', 'IssuanceProceedsTarget', 'IssuePriceCashGap', 'IssueDiscountCost_Period', 'IssueDiscountCost_Cumulative', 'FinancingCost_Period', 'FinancingCost_Cumulative', 'NonMarketableInterestCapitalized_Period', 'NonMarketableInterestCapitalized_Cumulative', 'TIPSInflationAccretion_Period', 'TIPSInflationAccretion_Cumulative', 'AuctionDemandShift_AvgAbs', 'AuctionDemandShift_MaxAbs', 'SecondaryDemandShift_AvgAbs', 'SecondaryDemandShift_MaxAbs', 'DebtServiceOutlay_Period', 'DebtServiceOutlay_Cumulative', 'TotalDebt_Agg', 'DebtHeld_Banks', 'DebtHeld_Private', 'DebtHeld_CB', 'DebtHeld_Foreign', 'DebtHeld_FedInternal', 'DebtHeld_TrustFunds', 'TGA', 'Reserves', 'TDC_Level', 'ReserveChange', 'TDC_Change', 'TGAChange', 'TDC_FiscalFlow', 'TDC_DebtService', 'TDC_AuctionAbsorption', 'TDC_SecondaryTrades', 'TDC_Other', 'TDC_PrincipalToDU', 'TDC_PrincipalCashToDU', 'TDC_InterestToDU', 'TDC_BillDiscountInterestToDU', 'TDC_CouponInterestToDU', 'TDC_FRNInterestToDU', 'TDC_TIPSCouponInterestToDU', 'TDC_TIPSInflationCompensationToDU', 'TDC_GrossIssuanceProceedsAbsorbedByDU', 'TDC_NetPrincipalIssuanceCashflowToDU', 'TDC_SecondaryDUToRU', 'TDC_SecondaryRUToDU', 'TDC_AuctionAbsorption_DomesticNonbank', 'TDC_AuctionAbsorption_MMF', 'TDC_AuctionAbsorption_MMFPlumbing', 'TDC_PrincipalToDU_DomesticNonbank', 'TDC_PrincipalToDU_MMF', 'TDC_PrincipalToDU_MMFPlumbing', 'TDC_PrincipalCashToDU_DomesticNonbank', 'TDC_PrincipalCashToDU_MMF', 'TDC_PrincipalCashToDU_MMFPlumbing', 'TDC_BillDiscountInterestToDU_DomesticNonbank', 'TDC_BillDiscountInterestToDU_MMF', 'TDC_CouponInterestToDU_DomesticNonbank', 'TDC_CouponInterestToDU_MMF', 'TDC_FRNInterestToDU_DomesticNonbank', 'TDC_FRNInterestToDU_MMF', 'TDC_TIPSCouponInterestToDU_DomesticNonbank', 'TDC_TIPSCouponInterestToDU_MMF', 'TDC_TIPSInflationCompensationToDU_DomesticNonbank', 'TDC_TIPSInflationCompensationToDU_MMF', 'TDC_InterestToDU_DomesticNonbank', 'TDC_InterestToDU_MMF', 'TDC_DebtService_MMFPlumbing', 'TDC_GrossIssuanceProceedsAbsorbedByDU_DomesticNonbank', 'TDC_GrossIssuanceProceedsAbsorbedByDU_MMF', 'TDC_SecondaryTrades_DomesticNonbank', 'TDC_SecondaryTrades_MMF', 'TDC_SecondaryTrades_MMFPlumbing', 'CB_TreasuryInterestCashReceived', 'CB_NetIncome', 'CB_Remittance', 'CB_DeferredAsset', 'WAM', 'DebtHeldByType_Fixed', 'DebtHeldByType_TIPS', 'DebtHeldByType_FRN', 'DebtHeldByType_NonMarketable', 'CPI_Level', 'Reference_CPI', 'CBOFundingModeActive', 'CBOPrimaryDeficitFlow', 'CBOControlledDebtTarget', 'CBOControlledDebtReference', 'ScenarioControlledDebt', 'DebtDriftFromReference', 'CashFinancingFaceIssued', 'CashFinancingProceeds', 'CBOControlledDebtPreIssuance', 'CBOControlledDebtPostIssuance', 'CBOControlledDebtTargetError', 'CBORequiredFaceIssuance', 'CBOBuybackFaceRetired', 'CBOBuybackCashPaid', 'CBOOperatingCashTarget', 'CBOCashResidual', 'CBOCashReconciliationResidual', 'CBOFiscalIncidencePolicyPresent', 'CBORemittanceCashEffect', 'CBOFedHoldingsTarget', 'CBOFedHoldingsTargetError', 'CBOFedAuctionShare', 'CBOFedSecondaryPurchaseFace', 'CBOFedSecondaryPurchaseCash', 'CBOFedSecondaryPurchaseReserveEffect', 'CBOFedSecondaryPurchaseDepositEffect', 'CBOFedSecondarySaleCash', 'CBOFedSecondarySaleReserveEffect', 'CBOFedSecondarySaleDepositEffect', 'CBOFedPrivateMaturityTDC', 'CBONetInterestDiagnostic', 'CBOTotalDeficitDiagnostic', 'CBONetInterestBridgeRows']
    results = pd.DataFrame(index=dates, columns=results_cols, dtype=float).fillna(0.0)
    mmf_acquisition_maturities = {}

    def record_mmf_acquisition(day, security_type, maturity_date):
        """Keep the latest maturity acquired by a money fund per date and type."""
        day = pd.Timestamp(day).date()
        maturity = pd.Timestamp(maturity_date).date()
        key = (day, security_type)
        mmf_acquisition_maturities[key] = max(maturity, mmf_acquisition_maturities.get(key, maturity))

    if require_bounded_handoff and handoff_sink is None:
        raise ValueError(
            'Production CBO execution requires a bounded handoff sink; '
            'the legacy list collector is a short-fixture oracle only.'
        )
    bounded_handoff = handoff_sink is not None
    handoff_tables = handoff_sink
    if handoff_tables is None:
        handoff_tables = {
            'tdcsim_accounting_journal': [],
            'tdcsim_period_issuance_flows': [],
            'tdcsim_period_principal_flows': [],
            'tdcsim_period_payment_flows': [],
            'tdcsim_holder_stocks': [],
            'tdcsim_tdc_principal_route_stocks': [],
            'tdcsim_debt_target_bridge': [],
            'tdcsim_scenario_metrics': [],
        }
    for status_col in ['FundingMode', 'FiscalIncidenceStatus', 'NetInterestDiagnosticStatus', 'CBOCashResidualStatus']:
        results[status_col] = ''
    if cbo_operating_cash_financing_mode:
        # Only this mode decides cash needs and settles them on a calendar; the debt-path
        # modes keep their result columns unchanged.
        for numeric_col in (
            *OPERATING_CASH_RESULT_COLUMNS, 'FedRolloverCashSizingDifference', 'FedRolloverCashEstimateRevision',
        ):
            results[numeric_col] = 0.0
    if cbo_funding_mode:
        for numeric_col in FED_INCOME_RESULT_COLUMNS:
            results[numeric_col] = 0.0
    for status_col in [
        'CBOFedStockMode',
        'CBOFedSettlementScope',
        'FedRolloverFedOnlyClasses',
        'CBOFedAcquisitionChannel',
        'CBOFedSecondarySaleBuyerMix',
        'CBORemittanceStatus',
    ]:
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
    results['FundingMode'] = funding_rule_mode if cbo_funding_mode else CASH_FUNDING_MODE
    holder_target_path = holder_targets.holder_target_path(params.get('holder_position_target'))
    stock_columns = [column for group in holder_targets.STOCK_COLUMNS.values() for column in group.values()]
    for numeric_col in [*stock_columns, *MMF_ADDED_TDC_COLUMNS.values()]:
        results[numeric_col] = 0.0
    if holder_target_path is not None:
        results[holder_targets.PHASE_COLUMN] = ''
        for numeric_col in holder_targets.RESULT_COLUMNS:
            results[numeric_col] = 0.0
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
                    if event_date <= end_date_pd:
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
        if evaluated_nominal_shock is not None:
            frn_rate_path = (
                cbo_inputs.get('frn_rate_path')
                if isinstance(cbo_inputs, dict)
                else None
            )
            if not isinstance(frn_rate_path, pd.DataFrame) or frn_rate_path.empty:
                raise CurveContractError(
                    'OPEN-04 evaluated nominal shock requires an explicit FRN benchmark path'
                )
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
        holder_absorption_source_scenario_id = None
        holder_absorption_resolution_status = 'not_configured'
        primary_flow_warning_cache = set()
        holder_absorption_warning_cache = set()
        primary_flow_path_used_count = 0
        if isinstance(ratewall_input_cfg, dict):
            source_base_dir = ratewall_input_cfg.get('base_dir')
            primary_flow_lookup = load_primary_flow_path(ratewall_input_cfg, base_dir=source_base_dir)
            holder_absorption_lookup = load_holder_absorption_path(ratewall_input_cfg, base_dir=source_base_dir)
            if ratewall_input_cfg.get('holder_absorption_path_file'):
                holder_fallbacks = ratewall_input_cfg.get(
                    'holder_absorption_scenario_fallbacks',
                    {},
                )
                if not isinstance(holder_fallbacks, dict):
                    raise ValueError(
                        'holder_absorption_scenario_fallbacks must be a mapping of '
                        'requested scenario IDs to source scenario IDs.'
                    )
                (
                    holder_absorption_source_scenario_id,
                    holder_absorption_resolution_status,
                ) = resolve_holder_absorption_scenario(
                    holder_absorption_lookup,
                    scenario_name=scenario_name,
                    required_quarters=set(period_counts_by_quarter),
                    configured_fallbacks=holder_fallbacks,
                )
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
        # Under operating-cash financing the bill-share rule decides daily but settles each
        # security on Treasury's calendar; decided amounts wait in pending balances.
        opening_issuance_rule = is_rule_profile(issuance_profile)
        batched_settlement = cbo_operating_cash_financing_mode and opening_issuance_rule
        settlement_schedule = (
            _settlement_schedule(issuance_profile, dates[0], dates[-1]) if batched_settlement else {}
        )
        settlement_dates = list(settlement_schedule)
        fiscal_year_close_dates = (
            _fiscal_year_close_dates(issuance_profile, dates[0], dates[-1]) if batched_settlement else []
        )
        close_settlement_plan = _last_settlements_before_close(settlement_schedule, fiscal_year_close_dates)
        settlement_calendar_data = copy.deepcopy(issuance_profile.get('settlement_calendar'))
        pending_issuance = {}
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
        if not np.isfinite(mmf_deposit_pass_through) or not (0.0 <= mmf_deposit_pass_through <= 1.0):
            raise ValueError('private_mmf_split.mmf_deposit_pass_through must be between 0.0 and 1.0')
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
        cpi_evaluation_dates = pd.date_range(t0, end_date_pd, freq='D')
        tips_cpi_path = cbo_inputs.get('tips_cpi_path') if isinstance(cbo_inputs, dict) else pd.DataFrame()
        if isinstance(tips_cpi_path, pd.DataFrame) and not tips_cpi_path.empty:
            cbo_cpi_lookup, cbo_reference_cpi_lookup = build_tips_daily_cpi_lookups_from_monthly_path(
                tips_cpi_path,
                cpi_evaluation_dates,
                scenario_id=cbo_scenario_id,
                default_value=reference_cpi_start,
                reference_lag_months=ref_cpi_lag,
            )
        else:
            cbo_cpi_lookup = _build_cbo_macro_cpi_lookup(
                cbo_inputs,
                cbo_scenario_id,
                cpi_evaluation_dates,
                cpi_start,
            )
            cbo_reference_cpi_lookup = build_projected_cpi_lookup_from_macro(
                cbo_inputs.get('macro_forecast') if isinstance(cbo_inputs, dict) else None,
                cpi_evaluation_dates,
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
            marketable = bond_portfolio['SecurityType'].isin(PUBLIC_MARKETABLE_SECURITY_TYPES)
            marketable_lots = bond_portfolio.loc[marketable]
            opening_bills = (
                bond_portfolio['SecurityType'].eq('Fixed')
                & bond_portfolio['CouponRate'].fillna(0.0).le(TGA_FLOOR_TOLERANCE)
                & bond_portfolio['OriginalMaturityYears'].le(1.0 + TGA_FLOOR_TOLERANCE)
            )
            opening_bills = opening_bills.loc[marketable]
            ids = fed_purchase_policy.issue_ids(marketable_lots, opening_bills)
            fed_purchase_policy.validate_issue_terms(marketable_lots, ids, opening_bills)
            bond_portfolio.loc[marketable, 'IssueID'] = ids
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
            bond_portfolio = _ensure_tdc_principal_route_columns(bond_portfolio)
            issue_price_missing = bond_portfolio['IssuePriceRatio'].isna() | (bond_portfolio['IssuePriceRatio'] <= TGA_FLOOR_TOLERANCE)
            bond_portfolio.loc[issue_price_missing, 'IssuePriceRatio'] = 1.0
            non_bill_mask = ~opening_bills
            issue_proceeds_missing = bond_portfolio['IssueProceeds'].isna() | (bond_portfolio['IssueProceeds'] <= TGA_FLOOR_TOLERANCE)
            bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'IssueProceeds'] = bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'FaceValue'] * bond_portfolio.loc[issue_proceeds_missing & non_bill_mask, 'IssuePriceRatio']
            issue_yield_missing = bond_portfolio['IssueYieldAtIssue'].isna()
            bond_portfolio.loc[issue_yield_missing & non_bill_mask, 'IssueYieldAtIssue'] = bond_portfolio.loc[issue_yield_missing & non_bill_mask, 'CouponRate']
            bill_needs_inference = opening_bills & (bond_portfolio['IssueYieldAtIssue'].isna() | bond_portfolio['IssueProceeds'].isna() | (bond_portfolio['IssueProceeds'] >= bond_portfolio['FaceValue'] - 1e-12))
            if bill_needs_inference.any():
                inferred_issue_data = bond_portfolio.loc[bill_needs_inference].apply(lambda row: pd.Series(infer_issue_data_for_loaded_bill(row['FaceValue'], row['OriginalMaturityYears'], row.get('IssueYieldAtIssue'), current_yield_curve_years, current_yield_curve_rates), index=['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue']), axis=1)
                bond_portfolio.loc[bill_needs_inference, ['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue']] = inferred_issue_data.values
    else:
        bond_portfolio = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
    t0 = dates[0]
    init_vals = params.get('initial_values', {})
    added_position = (
        MmfAddedPosition.from_mapping(params[PRIVATE_SUBBUCKET_MMF_ADDED])
        if PRIVATE_SUBBUCKET_MMF_ADDED in params else None
    )
    if (
        not bond_portfolio.empty
        and bond_portfolio[['HolderSubBucket', 'TDCPrincipalHolderSubBucket']].eq(PRIVATE_SUBBUCKET_MMF_ADDED).any().any()
    ):
        raise ValueError('The added money-fund position must open with no securities.')
    target_buyer = (params.get('holder_position_target') or {}).get('buyer_private_route')
    if target_buyer == PRIVATE_SUBBUCKET_MMF_ADDED and added_position is None:
        raise ValueError('The added money-fund buyer needs a declared settlement route.')
    if added_position is not None and params.get('trc_routing') is None:
        raise ValueError('The added money-fund position requires route-aware reserve accounting.')
    mmf_added_deposit_share = added_position.deposit_share if added_position is not None else 0.0
    opening_reserves = float(init_vals.get('reserves', 0.0))
    if added_position is not None:
        opening_reserves -= added_position.opening_on_rrp_bil
    results.loc[t0, 'TGA'] = init_vals.get('tga', 0.0)
    results.loc[t0, 'Reserves'] = opening_reserves
    trc_ledger = None
    trc_rows = []
    if params.get('trc_routing') is not None:
        trc_ledger = TrcLedger(
            parse_trc_routing(
                params['trc_routing'],
                mmf_deposit_pass_through=mmf_deposit_pass_through,
                mmf_added_position=added_position,
            ),
            opening_reserves=opening_reserves,
        )
        trc_rows.append(trc_ledger.opening_row(t0))
    elif mmf_deposit_pass_through != 1.0:
        raise ValueError('money-fund covered-deposit and ON RRP shares must sum to one; declare TRC routing for ON RRP')
    # TDC starts from its Treasury contribution anchor. The opening deposit
    # exchange for money-fund shares is recorded in the routing policy.
    results.loc[t0, 'TDC_Level'] = init_vals.get('tdc_level', init_vals.get('deposits', 0.0))
    results.loc[t0, 'CB_DeferredAsset'] = 0.0
    results.loc[t0, 'CPI_Level'] = cpi_start
    results.loc[t0, 'Reference_CPI'] = cbo_reference_cpi_lookup.get(
        pd.to_datetime(t0).normalize(),
        reference_cpi_start,
    )
    results.loc[t0, 'CBOFundingModeActive'] = 1.0 if cbo_funding_mode else 0.0
    opening_stocks = holder_targets.closing_stocks(bond_portfolio)
    for column, value in holder_targets.stock_result_values(opening_stocks).items():
        results.loc[t0, column] = value
    if holder_target_path is not None and holder_target_path.bill_change_weights is not None:
        opening_bills = opening_stocks[holder_target_path.stock_column_stem]
        _, opening_targets = holder_target_path.for_date(t0, opening_bills)
        if max(abs(opening_bills[p] - opening_targets[p]) for p in holder_targets.PARTICIPANTS) > holder_target_path.tolerance_bil:
            raise holder_targets.HolderTargetError('Opening bill holdings differ from the matched baseline.')
    results.loc[t0, ['CBOControlledDebtTarget', 'CBOControlledDebtPreIssuance', 'CBOControlledDebtPostIssuance', 'CBOControlledDebtTargetError', 'CBOOperatingCashTarget', 'CBOCashResidual', 'CBOFiscalIncidencePolicyPresent', 'CBOFedHoldingsTarget', 'CBOFedHoldingsTargetError', 'CBOFedAuctionShare', 'CBONetInterestBridgeRows']] = 0.0
    if bounded_handoff:
        handoff_tables.capture_portfolio_snapshot(
            bond_portfolio,
            snapshot_date=t0,
        )
    else:
        _handoff_append_holder_stocks(
            handoff_tables,
            bond_portfolio[bond_portfolio['Status'] == 'Active'].copy() if not bond_portfolio.empty else bond_portfolio,
            t0,
        )
    if cbo_funding_mode:
        results.loc[t0, 'CBOFedStockMode'] = 'synthetic_cb_treasury_stock_target_beneficial_holder'
        results.loc[t0, 'CBOFedSettlementScope'] = 'not_applicable_opening_row'
        results.loc[t0, 'CBORemittanceStatus'] = CBO_FED_REMITTANCE_OPENING_STATUS
        results.loc[t0, 'CBORemittanceCashEffect'] = 0.0
        fed_first_row = _fed_income_row(cbo_inputs, cbo_scenario_id, t0, dates[1])
        fed_opening = FedOpeningBalanceSheet.from_row(fed_first_row)
        fed_on_rrp = _fed_on_rrp_balance(trc_ledger, fed_opening)
        fed_reserves = fed_opening.rolled_reserves(
            tga=float(results.loc[t0, 'TGA']),
            on_rrp=fed_on_rrp,
            soma_mbs=float(fed_first_row['soma_mbs_start_bil']),
            currency=float(fed_first_row['currency_start_bil']),
        )
        fed_due = fed_opening.earnings_remittances_due
        fed_last_remittance_week = remittance_week(t0)
        fed_short_tenor = float(fed_first_row['short_rate_tenor_years'])
        fed_iorb_spread = float(fed_first_row['iorb_spread_pct'])
        fed_on_rrp_spread = float(fed_first_row['on_rrp_spread_pct'])
        fed_opening_short = 100.0 * float(evaluate_nominal_yield(
            fed_short_tenor, current_yield_curve_years, current_yield_curve_rates,
            method=yield_interpolation_method, floor_zero=yield_floor_zero, shock=evaluated_nominal_shock,
        ))
        results.loc[t0, 'CB_ShortRatePct'] = fed_opening_short
        results.loc[t0, 'CB_IORBRatePct'] = fed_opening_short + fed_iorb_spread
        results.loc[t0, 'CB_ONRRPRatePct'] = fed_opening_short + fed_on_rrp_spread
        results.loc[t0, 'CB_ReserveBalances'] = fed_reserves
        results.loc[t0, 'CB_ONRRPBalances'] = fed_on_rrp
        results.loc[t0, 'CB_Currency'] = float(fed_first_row['currency_start_bil'])
        results.loc[t0, 'CB_MBSHoldings'] = float(fed_first_row['soma_mbs_start_bil'])
        results.loc[t0, 'CB_EarningsRemittancesDue'] = fed_due
        results.loc[t0, 'CB_OtherLiabilitiesAndCapital'] = fed_opening.other_liabilities_and_capital
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
    if cbo_funding_mode:
        fed_rows = _cbo_rows_for_scenario(cbo_inputs.get('fed_holdings'), cbo_scenario_id)
        if fed_rows is not None and not fed_rows.empty:
            fed_rows = fed_rows.copy()
            fed_rows['_period_end'] = pd.to_datetime(
                fed_rows['period_end'],
                errors='coerce',
            ).dt.normalize()
            opening_rows = fed_rows[
                fed_rows['_period_end'].eq(pd.Timestamp(t0).normalize())
            ]
            if len(opening_rows) != 1:
                raise ValueError(
                    'Opening Fed target path must contain exactly one row at '
                    f'{pd.Timestamp(t0).date()}; found {len(opening_rows)}.'
                )
            opening_fed_target = float(
                opening_rows.iloc[0]['cbo_fed_holdings_target_bil']
            )
            opening_cb_stock = float(results.loc[t0, 'DebtHeld_CB'])
            opening_tolerance = float(
                funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001
            )
            if abs(opening_cb_stock - opening_fed_target) > opening_tolerance:
                raise ValueError(
                    'Opening Fed Treasury stock does not match the opening Fed target: '
                    f'stock={opening_cb_stock:.12f}, target={opening_fed_target:.12f}. '
                    'Provide a pre-start nonsettling restatement instead of an in-period trade.'
                )
            results.loc[t0, 'CBOFedHoldingsTarget'] = opening_fed_target
            results.loc[t0, 'CBOFedHoldingsTargetApplicable'] = 1.0
            results.loc[t0, 'CBOFedHoldingsTargetError'] = opening_cb_stock - opening_fed_target
            results.loc[t0, 'CBOFedSettlementScope'] = 'opening_stock_target_match_nonsettling'
    results.loc[t0, 'GovSpending'] = q_start_spending
    results.loc[t0, 'Taxes'] = q_start_taxes
    results.loc[t0, ['PrimaryDeficit', 'InterestPaid_Bonds', 'PrincipalPaid_Bonds', 'InterestOutlay_Period', 'InterestOutlay_Cumulative', 'PrincipalRollover_Period', 'PrincipalRollover_Cumulative', 'NewDebtIssued', 'AuctionProceeds', 'IssuanceProceedsTarget', 'IssueDiscountCost_Period', 'IssueDiscountCost_Cumulative', 'FinancingCost_Period', 'FinancingCost_Cumulative', 'NonMarketableInterestCapitalized_Period', 'NonMarketableInterestCapitalized_Cumulative', 'TIPSInflationAccretion_Period', 'TIPSInflationAccretion_Cumulative', 'DebtServiceOutlay_Period', 'DebtServiceOutlay_Cumulative']] = 0.0
    results.loc[t0, ['ReserveChange', 'TDC_Change', 'TGAChange']] = 0.0
    results.loc[t0, 'CB_TreasuryInterestCashReceived'] = 0.0
    results.loc[t0, ['CB_NetIncome', 'CB_Remittance']] = 0.0
    if cbo_funding_mode:
        results.loc[t0, 'CB_DeferredAsset'] = max(0.0, -fed_due)
    max_existing_id = 0
    if not bond_portfolio.empty and bond_portfolio['BondID'].notna().any():
        numeric_bond_ids = pd.to_numeric(bond_portfolio['BondID'], errors='coerce')
        if numeric_bond_ids.notna().any():
            max_existing_id = numeric_bond_ids.max()
    bond_id_counter = int(max_existing_id) + 1
    last_quarter_start_date = t0
    weeks_in_quarter = 0
    if bounded_handoff:
        handoff_tables.begin_run(
            opening_date=t0,
            opening_result=results.loc[t0].to_dict(),
        )
    if batched_settlement:
        calendar_rules = settlement_rules(issuance_profile)
        issue_registry = _opening_issue_registry(bond_portfolio, calendar_rules, dates[0])

    def frn_accrual_per_face(dated, settle):
        """Interest per unit face on a new FRN over (dated, settle], at the FRN rate path's
        benchmark plus the new-issue spread, on the engine's daily accrual convention."""

        frame = cbo_inputs.get('frn_rate_path') if isinstance(cbo_inputs, dict) else None
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError('a new FRN settling after its dated date needs the FRN rate path')
        total = 0.0
        for day in pd.date_range(pd.Timestamp(dated) + pd.Timedelta(days=1), pd.Timestamp(settle), freq='D'):
            row = _frn_rate_path_row_for_date(
                frame, cbo_scenario_id, day, lookup=cbo_inputs.get('frn_rate_path_date_lookup', {})
            )
            if row is None:
                raise ValueError(f'the FRN rate path has no rate for {day.date()}')
            basis = float(row.get('day_count_basis', FRN_DAY_COUNT_BASIS) or FRN_DAY_COUNT_BASIS)
            total += (float(row['benchmark_rate_decimal']) + float(frn_spread)) / basis
        return total
    last_payment_date = dates[0]
    last_market_date = dates[0]
    for i in range(1, num_periods):
        prev_date = dates[i - 1]
        current_date = dates[i]
        delta_t_days = (current_date - prev_date).days
        # On Treasury's calendar cash moves only when Fedwire is open: payments due on a
        # closed day are made on the next business day, whose window reaches back to the
        # last one. Securities settle and trade only on securities business days, which
        # also exclude Good Friday and the Friday before a Saturday holiday.
        payments_open = (not batched_settlement) or is_fedwire_business_day(current_date)
        market_open = (not batched_settlement) or is_securities_business_day(current_date)
        window_start = last_payment_date if batched_settlement else prev_date
        settle_window_start = last_market_date if batched_settlement else prev_date
        if payments_open:
            last_payment_date = current_date
        if market_open:
            last_market_date = current_date
        due_events = (
            _due_settlement_events(settlement_schedule, settlement_dates, settle_window_start, current_date)
            if batched_settlement and market_open
            else {}
        )
        fed_rollover = {}
        fed_rollover_cash = 0.0
        fed_redeemed_face = 0.0
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
            indexation_changes = new_adjusted_principal - prev_adjusted_principal
            tips_journal_rows = bond_portfolio.loc[
                indexation_changes.index,
                [column for column in _JOURNAL_ROW_COLUMNS if column in bond_portfolio.columns],
            ].to_dict('index')
            for index, amount in indexation_changes.items():
                tips_row = tips_journal_rows[index]
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='indexation',
                    leg_type='tips_adjusted_principal_indexation',
                    accounting_basis='adjusted_principal_stock',
                    row=tips_row,
                    adjusted_principal_change=float(amount),
                    settlement_scope='noncash_stock_remeasurement',
                    is_intragovernmental=(
                        str(tips_row['HolderType']) in INTRAGOV_HOLDERS
                    ),
                    # The results count every lot's indexation, so the journal must too.
                    drop_below=JOURNAL_MIN_EVENT_BIL,
                )
            if include_tips_inflation_accretion:
                tips_inflation_accretion_period = indexation_changes.sum()
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
        if frn_mask.any() and payments_open:
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
                        window_start,
                        current_date,
                        'CBO FRN rate path',
                    )
                frn_yield = float(frn_path_row['benchmark_rate_decimal'])
                day_count_basis = float(frn_path_row.get('day_count_basis', FRN_DAY_COUNT_BASIS) or FRN_DAY_COUNT_BASIS)
            else:
                if evaluated_nominal_shock is not None:
                    raise CurveContractError(
                        'OPEN-04 evaluated nominal shock cannot use the nominal-curve FRN fallback'
                    )
                frn_yield = get_yield_for_maturity(
                    frn_benchmark_mat,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                )
                frn_yield = 0.0 if pd.isna(frn_yield) else frn_yield
            lockout_business_days = _frn_lockout_business_days(frn_path_row)
            frn_indices = bond_portfolio.index[frn_mask]
            last_accrued = pd.to_datetime(bond_portfolio.loc[frn_indices, 'LastAccrualDate']).dt.normalize()
            last_accrued = last_accrued.fillna(pd.Timestamp(window_start).normalize())
            accrual_dates = pd.date_range(
                last_accrued.min() + pd.Timedelta(days=1), pd.Timestamp(current_date).normalize(), freq='D',
            )
            if len(accrual_dates) == 1 and last_accrued.lt(current_date).all():
                accrual_day = pd.Timestamp(accrual_dates[0]).normalize()
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
                        for value in _frn_coupon_dates_for_period(bond, last_accrued.at[idx], current_date)
                    }
                    maturity_date = pd.Timestamp(bond.get('MaturityDate')).normalize()
                    last_benchmark_rate = bond.get("BenchmarkRate_FRN")
                    last_accrual_date = pd.NaT
                    for accrual_day in accrual_dates:
                        if accrual_day <= last_accrued.at[idx]:
                            continue
                        if pd.notna(maturity_date) and accrual_day > maturity_date:
                            break
                        rate_date = _frn_rate_date_for_accrual_day(bond, accrual_day, lockout_business_days)
                        rate_row = None
                        benchmark_rate = float(frn_yield)
                        rate_day_count = day_count_basis
                        if accrual_day <= t0 and pd.notna(bond.get("BenchmarkRate_FRN")):
                            # The observed opening reset carries across the
                            # pre-simulation days before the forecast rates apply.
                            benchmark_rate = float(bond["BenchmarkRate_FRN"])
                            rate_day_count = FRN_DAY_COUNT_BASIS
                        elif isinstance(frn_path_frame, pd.DataFrame) and not frn_path_frame.empty:
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
            cbo_remittance_row = _cbo_period_row(
                cbo_inputs['fed_remittance'],
                cbo_scenario_id,
                prev_date,
                current_date,
                'Fed remittance path',
            )
            cbo_period_remittance = float(cbo_remittance_row['remittance_bil'])
            if not payments_open and cbo_period_remittance > 0.0:
                raise ValueError(f'Fed remittance on {current_date.date()}, a day Fedwire is closed')
            if not payments_open and abs(cbo_period_primary_deficit) > TGA_FLOOR_TOLERANCE:
                raise ValueError(
                    f'the primary deficit path books {cbo_period_primary_deficit} billion on '
                    f'{current_date.date()}, a day Fedwire is closed; fiscal flows must fall on business days'
                )
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
            # CBO counts Table 4-3 remittances as receipts; the ordinary fiscal flow excludes them.
            fiscal_reserve_change += cbo_period_remittance
            results.loc[current_date, 'CBOTable43RemittanceInFiscalFlow'] = cbo_period_remittance
            fiscal_deposit_change = (
                fiscal_reserve_change * float(cbo_incidence['du_share'])
                if cbo_incidence is not None
                else 0.0
            )
            results.loc[current_date, 'CBOPrimaryDeficitFlow'] = cbo_period_primary_deficit
            results.loc[current_date, 'FiscalIncidenceStatus'] = cbo_fiscal_incidence_status
        else:
            fiscal_deposit_change = (
                gov_spending_period * float(du_share_spending)
                - taxes_period * float(du_share_taxes)
            )
        fiscal_tga_change = -fiscal_reserve_change
        # Under CBO funding the TGA pays P + R_CBO here and receives the run's own Fed
        # remittance below, so a remittance different from Table 4-3 changes financing.
        net_fiscal_tga_change = fiscal_tga_change
        reserve_change_period += fiscal_reserve_change
        if trc_ledger is not None:
            trc_ledger.record_public_cash('FiscalFlow', fiscal_reserve_change, 'covered_deposit_settled')
        deposit_change_period += fiscal_deposit_change
        tga_change_period += net_fiscal_tga_change
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type='fiscal',
            leg_type='tax_and_primary_spending_cash',
            accounting_basis='treasury_cash_reserve_deposit',
            holder_sector='Treasury',
            counterparty_sector='aggregate_fiscal_counterparties',
            instrument_type='fiscal',
            maturity_bucket='not_applicable',
            treasury_cash_change=fiscal_tga_change,
            reserve_change=fiscal_reserve_change,
            deposit_change=fiscal_deposit_change,
            settlement_scope='fiscal_cash_settlement',
        )
        nonmkt_interest_capitalized_period = _capitalize_nonmarketable_interest(
            bond_portfolio,
            prev_date,
            current_date,
            current_nonmkt_params,
            current_yield_curve_years,
            current_yield_curve_rates,
            interpolation_method=yield_interpolation_method,
            floor_zero=yield_floor_zero,
            nominal_shock=evaluated_nominal_shock,
            handoff_tables=handoff_tables,
        )
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
        cb_private_maturity_tdc_period = 0.0
        # The Fed's stock removed at maturity (adjusted principal for TIPS; a deflation
        # floor's top-up is cash, not stock).
        cb_stock_matured = 0.0
        cb_matured_bills = 0.0
        total_principal_paid_period = 0.0
        total_interest_paid_period = 0.0
        maturing_mask = bond_portfolio['MaturityDate'].notna() & (bond_portfolio['MaturityDate'] > window_start) & (bond_portfolio['MaturityDate'] <= current_date) & (bond_portfolio['Status'] == 'Active') & payments_open
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
                tdc_principal_recipient = _tdc_principal_holder(bond)
                face_value = bond['FaceValue']
                coupon_rate = bond['CouponRate']
                maturity_date = bond['MaturityDate']
                security_type = bond['SecurityType']
                if security_type == 'TIPS':
                    original_principal = float(bond.get('OriginalPrincipal', 0.0) or 0.0)
                    reference_cpi_issue = float(bond.get('ReferenceCPI_Issue', 0.0) or 0.0)
                    event_reference_cpi = _reference_cpi_at_event(
                        maturity_date,
                        opening_date=t0,
                        reference_lag_months=ref_cpi_lag,
                        annual_inflation=cpi_inflation,
                        results=results,
                        cbo_reference_lookup=cbo_reference_cpi_lookup,
                    )
                    event_index_ratio = (
                        event_reference_cpi / reference_cpi_issue
                        if reference_cpi_issue > TGA_FLOOR_TOLERANCE
                        else 1.0
                    )
                    adjusted_principal_stock_removed = original_principal * event_index_ratio
                    period_end_adjusted_principal = float(
                        bond.get('AdjustedPrincipal', original_principal) or original_principal
                    )
                    bond_portfolio.loc[idx, 'IndexRatio'] = event_index_ratio
                    bond_portfolio.loc[idx, 'AdjustedPrincipal'] = adjusted_principal_stock_removed
                    bond['IndexRatio'] = event_index_ratio
                    bond['AdjustedPrincipal'] = adjusted_principal_stock_removed
                    event_date_correction = (
                        adjusted_principal_stock_removed - period_end_adjusted_principal
                    )
                    _handoff_append_journal(
                        handoff_tables,
                        prev_date,
                        current_date,
                        event_type='indexation',
                        leg_type='tips_exact_event_date_indexation_correction',
                        accounting_basis='adjusted_principal_stock',
                        row=bond,
                        adjusted_principal_change=event_date_correction,
                        settlement_scope='noncash_stock_remeasurement',
                        is_intragovernmental=holder in INTRAGOV_HOLDERS,
                    )
                    if include_tips_inflation_accretion:
                        tips_inflation_accretion_period += event_date_correction
                    if holder == 'CB':
                        cb_tips_indexation_period += event_date_correction
                    principal_payment = max(
                        original_principal,
                        adjusted_principal_stock_removed,
                    )
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
                principal_component_payment = max(0.0, principal_payment - bill_discount_interest_payment)
                _handoff_append_principal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    bond,
                    redemption_type='scheduled_maturity',
                    face_redeemed=face_value,
                    principal_redeemed=principal_component_payment,
                    cash_paid=principal_payment,
                    adjusted_principal_stock_removed=(
                        adjusted_principal_stock_removed
                        if security_type == 'TIPS'
                        else None
                    ),
                    mmf_deposit_pass_through=mmf_deposit_pass_through,
                    mmf_added_deposit_share=mmf_added_deposit_share,
                )
                if bill_discount_interest_payment > TGA_FLOOR_TOLERANCE:
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        bond,
                        'bill_discount',
                        'budget_accrual',
                        bill_discount_interest_payment,
                        is_additive_to_cash_total=False,
                    )
                if tips_inflation_compensation_payment > TGA_FLOOR_TOLERANCE:
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        bond,
                        'tips_indexation',
                        'memo_decomposition',
                        tips_inflation_compensation_payment,
                        is_additive_to_cash_total=False,
                    )
                tips_deflation_floor_topup = (
                    max(0.0, principal_payment - adjusted_principal_stock_removed)
                    if security_type == 'TIPS'
                    else 0.0
                )
                if tips_deflation_floor_topup > TGA_FLOOR_TOLERANCE:
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        bond,
                        'tips_deflation_floor_topup',
                        'stock_to_cash_bridge',
                        tips_deflation_floor_topup,
                        is_additive_to_cash_total=False,
                    )
                if interest_payment > TGA_FLOOR_TOLERANCE:
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        bond,
                        'frn_interest'
                        if security_type == 'FRN'
                        else ('tips_coupon' if security_type == 'TIPS' else 'fixed_coupon'),
                        'cash',
                        interest_payment,
                        is_additive_to_cash_total=True,
                    )
                principal_paid_by_holder[holder] += principal_payment
                if holder == 'CB':
                    cb_stock_matured += (
                        adjusted_principal_stock_removed if security_type == 'TIPS' else principal_payment
                    )
                if holder == 'CB' and security_type == 'Fixed' and coupon_rate <= TGA_FLOOR_TOLERANCE:
                    cb_matured_bills += principal_payment
                interest_paid_by_holder[holder] += interest_payment
                principal_component_paid_by_holder[holder] += max(
                    0.0, principal_payment - bill_discount_interest_payment
                )
                bill_discount_interest_paid_by_holder[holder] += bill_discount_interest_payment
                tips_inflation_compensation_paid_by_holder[holder] += (
                    tips_inflation_compensation_payment
                )
                if tdc_principal_recipient['holder_sector'] == 'Private':
                    if holder == 'CB':
                        cb_private_maturity_tdc_period += principal_payment
                    principal_route = tdc_principal_recipient['holder_subsector']
                    principal_paid_by_private_route[principal_route] += principal_payment
                    principal_component_paid_by_private_route[principal_route] += max(
                        0.0, principal_payment - bill_discount_interest_payment
                    )
                    bill_discount_interest_paid_by_private_route[principal_route] += bill_discount_interest_payment
                if holder == 'Private':
                    private_route = _private_subbucket(bond)
                    interest_paid_by_private_route[private_route] += interest_payment
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
        if frn_active_mask.any() and payments_open:
            frn_tranches = bond_portfolio.loc[frn_active_mask]
            frn_paying_indices = _coupon_due_indices_for_period(
                frn_tranches,
                window_start,
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
                for idx, amount in payment_amount.items():
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        bond_portfolio.loc[idx],
                        'frn_interest',
                        'cash',
                        amount,
                        is_additive_to_cash_total=True,
                    )
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
        if fixed_tips_mask.any() and payments_open:
            fixed_tips_tranches = bond_portfolio.loc[fixed_tips_mask]
            coupon_due_dates = {}
            if (current_date - window_start).days <= 1:
                fixed_tips_paying_indices = _daily_coupon_due_indices(fixed_tips_tranches, current_date, frequency=2)
            else:
                fixed_tips_paying_indices = []
                for idx, bond in fixed_tips_tranches.iterrows():
                    issue_dt = bond['IssueDate']
                    maturity_dt = bond['MaturityDate']
                    coupon_dates = get_coupon_dates_in_period(
                        issue_dt,
                        maturity_dt,
                        window_start,
                        current_date,
                        frequency=2,
                        first_interest_payment_date=bond.get('FirstInterestPaymentDate'),
                        interest_payment_frequency=bond.get('InterestPaymentFrequency'),
                    )
                    if coupon_dates:
                        fixed_tips_paying_indices.append(idx)
                        coupon_due_dates[idx] = pd.Timestamp(max(coupon_dates)).normalize()
            if fixed_tips_paying_indices:
                fixed_tips_paying_indices = list(set(fixed_tips_paying_indices))
                paying_tranches = bond_portfolio.loc[fixed_tips_paying_indices]
                principal_base = np.where(paying_tranches['SecurityType'] == 'TIPS', paying_tranches['AdjustedPrincipal'].fillna(paying_tranches['FaceValue']), paying_tranches['FaceValue'])
                # A TIPS coupon paid after a closed due date uses the due date's index
                # ratio: the payment moves, the amount does not.
                for position, idx in enumerate(paying_tranches.index):
                    due = coupon_due_dates.get(idx)
                    if due is None or due >= pd.Timestamp(current_date).normalize():
                        continue
                    if paying_tranches.at[idx, 'SecurityType'] == 'TIPS':
                        principal_base[position] = float(paying_tranches.at[idx, 'OriginalPrincipal']) * _reference_cpi_at_event(
                            due,
                            opening_date=t0,
                            reference_lag_months=ref_cpi_lag,
                            annual_inflation=cpi_inflation,
                            results=results,
                            cbo_reference_lookup=cbo_reference_cpi_lookup,
                        ) / float(paying_tranches.at[idx, 'ReferenceCPI_Issue'])
                payment_amount = (principal_base * paying_tranches['CouponRate'] / 2.0).clip(lower=0)
                for idx, amount in payment_amount.items():
                    row = paying_tranches.loc[idx]
                    _handoff_append_payment(
                        handoff_tables,
                        prev_date,
                        current_date,
                        row,
                        'tips_coupon' if row['SecurityType'] == 'TIPS' else 'fixed_coupon',
                        'cash',
                        amount,
                        is_additive_to_cash_total=True,
                    )
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
        results.loc[current_date, 'PrincipalPaid_Bonds'] = total_principal_paid_period
        results.loc[current_date, 'CBOFedPrivateMaturityTDC'] = cb_private_maturity_tdc_period
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
        debt_service_deposit_change = _effective_private_amount(
            {
                route: principal_paid_by_private_route[route] + interest_paid_by_private_route[route]
                for route in PRIVATE_SUBBUCKETS
            },
            mmf_deposit_pass_through,
            mmf_added_deposit_share=mmf_added_deposit_share,
        )
        reserve_change_period += debt_service_reserve_change
        if trc_ledger is not None:
            trc_ledger.record_holder_cash(
                'DebtService',
                {
                    holder: principal_paid_by_holder.get(holder, 0.0) + interest_paid_by_holder.get(holder, 0.0)
                    for holder in ('Banks', 'Private', 'Foreign')
                },
                {
                    route: principal_paid_by_private_route[route] + interest_paid_by_private_route[route]
                    for route in PRIVATE_SUBBUCKETS
                },
            )
        deposit_change_period += debt_service_deposit_change
        tga_change_period += debt_service_tga_change
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type='settlement',
            leg_type='debt_service_counterparty_settlement',
            accounting_basis='reserve_and_deposit_settlement',
            holder_sector='aggregate_public_holders',
            counterparty_sector='Treasury',
            instrument_type='all_treasury',
            maturity_bucket='all',
            reserve_change=debt_service_reserve_change,
            deposit_change=debt_service_deposit_change,
            settlement_scope='aggregate_holder_cash_settlement',
        )
        cb_treasury_interest_cash_received_period = interest_paid_by_holder.get('CB', 0.0)
        if cbo_funding_mode:
            fed_row = _fed_income_row(cbo_inputs, cbo_scenario_id, prev_date, current_date)
            fed_treasury_income = {
                'CB_BillDiscountIncome': bill_discount_interest_paid_by_holder.get('CB', 0.0),
                'CB_CouponIncome': coupon_interest_paid_by_holder.get('CB', 0.0),
                'CB_FRNIncome': frn_interest_paid_by_holder.get('CB', 0.0),
                'CB_TIPSCouponIncome': tips_coupon_interest_paid_by_holder.get('CB', 0.0),
                'CB_TIPSInflationCompensationIncome': tips_inflation_compensation_paid_by_holder.get('CB', 0.0),
            }
            fed_short = 100.0 * float(evaluate_nominal_yield(
                fed_short_tenor, current_yield_curve_years, current_yield_curve_rates,
                method=yield_interpolation_method, floor_zero=yield_floor_zero, shock=evaluated_nominal_shock,
            ))
            fed_iorb_rate = fed_short + fed_iorb_spread
            fed_on_rrp_rate = fed_short + fed_on_rrp_spread
            fed_accrual = (current_date - prev_date).days / INTEREST_EXPENSE_DAY_COUNT
            fed_iorb_expense = fed_iorb_rate / 100.0 * fed_reserves * fed_accrual
            fed_on_rrp_expense = fed_on_rrp_rate / 100.0 * fed_on_rrp * fed_accrual
            fed_lines = {
                'CB_MBSIncome': float(fed_row['soma_mbs_interest_bil']),
                'CB_OtherIncome': float(fed_row['other_income_bil']),
                'CB_IORBExpense': fed_iorb_expense,
                'CB_ONRRPExpense': fed_on_rrp_expense,
                'CB_OperatingExpenses': float(fed_row['operating_expenses_bil']),
                'CB_Dividends': float(fed_row['dividends_bil']),
            }
            fed_treasury_total = sum(fed_treasury_income[column] for column in TREASURY_INCOME_COLUMNS)
            cb_net_income_period = (
                fed_treasury_total + fed_lines['CB_MBSIncome'] + fed_lines['CB_OtherIncome']
                - fed_iorb_expense - fed_on_rrp_expense
                - fed_lines['CB_OperatingExpenses'] - fed_lines['CB_Dividends']
            )
            fed_week = remittance_week(current_date)
            fed_pay_day = is_fedwire_business_day(current_date) and fed_week > fed_last_remittance_week
            if fed_pay_day:
                fed_last_remittance_week = fed_week
            fed_due, cb_remittance_period = settle_remittances(fed_due, cb_net_income_period, fed_pay_day)
            current_deferred_asset = max(0.0, -fed_due)
            cb_remit_tga_change = cb_remittance_period
            for column, value in (*fed_treasury_income.items(), *fed_lines.items()):
                results.loc[current_date, column] = value
            results.loc[current_date, 'CB_TreasuryInterestIncome'] = fed_treasury_total
            results.loc[current_date, 'CB_ShortRatePct'] = fed_short
            results.loc[current_date, 'CB_IORBRatePct'] = fed_iorb_rate
            results.loc[current_date, 'CB_ONRRPRatePct'] = fed_on_rrp_rate
            results.loc[current_date, 'CB_EarningsRemittancesDue'] = fed_due
            results.loc[current_date, 'CB_OtherLiabilitiesAndCapital'] = (
                fed_opening.other_liabilities_and_capital + fed_due - fed_opening.earnings_remittances_due
            )
            results.loc[current_date, 'CBORemittanceStatus'] = CBO_FED_REMITTANCE_STATUS
        else:
            prev_deferred_asset = results.loc[prev_date, 'CB_DeferredAsset']
            cb_net_income_period = cb_treasury_interest_cash_received_period - cb_net_expense
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
        results.loc[current_date, 'CB_TreasuryInterestCashReceived'] = cb_treasury_interest_cash_received_period
        results.loc[current_date, 'CB_NetIncome'] = cb_net_income_period
        results.loc[current_date, 'CB_Remittance'] = cb_remittance_period
        results.loc[current_date, 'CB_DeferredAsset'] = current_deferred_asset
        if cbo_funding_mode:
            results.loc[current_date, 'CBORemittanceCashEffect'] = cb_remit_tga_change
        tga_change_period += cb_remit_tga_change
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type='transfer',
            leg_type='central_bank_remittance',
            accounting_basis='treasury_cash',
            holder_sector='Treasury',
            counterparty_sector='CB',
            instrument_type='cash',
            maturity_bucket='not_applicable',
            treasury_cash_change=cb_remit_tga_change,
            settlement_scope='central_bank_remittance_settlement',
        )
        prev_tga = results.loc[prev_date, 'TGA']
        tga_change_before_issuance = net_fiscal_tga_change + debt_service_tga_change + (tga_inflow_secondary_mkt - tga_drain_secondary_mkt)
        tga_change_before_issuance += cb_remit_tga_change
        tga_change_before_issuance = tga_change_before_issuance + reserve_transfer + money_minting_transfers
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
        cash_financing_face_issued = 0.0
        cash_financing_proceeds = 0.0
        issue_price_cash_gap = 0.0
        cbo_fed_holdings_target = 0.0
        cbo_fed_auction_share = 0.0
        cbo_fed_allocation_override_active = False
        cbo_fed_begin_stock = float(results.loc[prev_date, 'DebtHeld_CB'])
        fed_secondary_purchase_face = 0.0
        fed_secondary_purchase_cash = 0.0
        fed_secondary_sale_face = 0.0
        fed_secondary_sale_cash = 0.0
        fed_secondary_reserve_effect = 0.0
        fed_secondary_deposit_effect = 0.0
        fed_secondary_private_route_effects = _zero_private_routes()
        holder_target_deposit_effect = 0.0
        holder_target_private_route_effects = _zero_private_routes()
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
            if (
                cbo_debt_target_mode
                and raw_required_issuance < -target_tolerance
                and negative_issuance_action in {
                'retire_shortest_public_marketable',
                'buyback_shortest_public_marketable',
                }
            ):
                amount_to_retire = abs(raw_required_issuance)
                bond_portfolio, retired_amount, retired_by_holder, retired_by_private_route, retirement_events = (
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
                for event_row in retirement_events:
                    retire_face = float(
                        event_row.get(
                            'RetiredFaceValue',
                            event_row.get('FaceValue', 0.0),
                        )
                        or 0.0
                    )
                    retire_cash = float(
                        event_row.get('RetiredCashPaid', retire_face) or 0.0
                    )
                    if max(retire_face, retire_cash) <= TGA_FLOOR_TOLERANCE:
                        continue
                    _handoff_append_principal(
                        handoff_tables,
                        prev_date,
                        current_date,
                        event_row,
                        redemption_type='explicit_retirement_at_par',
                        face_redeemed=retire_face,
                        principal_redeemed=retire_cash,
                        cash_paid=retire_cash,
                        adjusted_principal_stock_removed=(
                            float(
                                event_row.get(
                                    'RetiredAdjustedPrincipal',
                                    event_row.get('AdjustedPrincipal', retire_cash),
                                )
                            )
                            if str(event_row.get('SecurityType', '')) == 'TIPS'
                            else None
                        ),
                        mmf_deposit_pass_through=mmf_deposit_pass_through,
                        mmf_added_deposit_share=mmf_added_deposit_share,
                    )
                total_principal_paid_period += retired_amount
                buyback_reserve_change = sum(
                    retired_by_holder.get(h, 0.0)
                    for h in ['Banks', 'Private', 'Foreign']
                )
                reserve_change_period += buyback_reserve_change
                if trc_ledger is not None:
                    trc_ledger.record_holder_cash(
                        'Buyback',
                        {holder: retired_by_holder.get(holder, 0.0) for holder in ('Banks', 'Private', 'Foreign')},
                        retired_by_private_route,
                    )
                buyback_deposit_change = _effective_private_amount(
                    retired_by_private_route,
                    mmf_deposit_pass_through,
                    mmf_added_deposit_share=mmf_added_deposit_share,
                )
                debt_service_deposit_change += buyback_deposit_change
                deposit_change_period += buyback_deposit_change
                tga_change_period -= retired_amount
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='settlement',
                    leg_type='buyback_counterparty_settlement',
                    accounting_basis='reserve_and_deposit_settlement',
                    holder_sector='aggregate_buyback_holders',
                    counterparty_sector='Treasury',
                    instrument_type='all_treasury',
                    maturity_bucket='all',
                    reserve_change=buyback_reserve_change,
                    deposit_change=buyback_deposit_change,
                    settlement_scope='explicit_retirement_at_par',
                )
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
            if cbo_reference_financing_mode:
                total_issuance_target_period = max(0.0, raw_required_issuance)
            elif cbo_debt_target_mode:
                total_issuance_target_period = calculate_required_face_issuance(
                    cbo_controlled_debt_target,
                    cbo_pre_issuance_controlled_debt,
                    strict=True,
                    tolerance=target_tolerance,
                )
            else:
                # Sized below, once the operating-cash target and the day's cash
                # reconciliation residual are known; it is a proceeds amount.
                total_issuance_target_period = 0.0
            cbo_face_target_mode = not cbo_operating_cash_financing_mode
            results.loc[current_date, 'CBOControlledDebtTargetApplicable'] = (
                1.0 if cbo_debt_target_mode else 0.0
            )
            if isinstance(cbo_inputs.get('operating_cash'), pd.DataFrame) and not cbo_inputs['operating_cash'].empty:
                cbo_operating_cash_row = _cbo_exact_date_row(
                    cbo_inputs['operating_cash'],
                    cbo_scenario_id,
                    current_date,
                    'period_end',
                    'CBO operating cash path',
                )
                cbo_operating_cash_target = (
                    float(cbo_operating_cash_row['operating_cash_target_bil'])
                    if payments_open
                    else float(results.loc[prev_date, 'CBOOperatingCashTarget'])
                )
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
                if not payments_open and abs(cbo_cash_residual_period) > TGA_FLOOR_TOLERANCE:
                    raise ValueError(
                        f'the cash reconciliation residual moves cash on {current_date.date()}, '
                        'a day Fedwire is closed'
                    )
            if cbo_operating_cash_financing_mode:
                if cbo_cash_residual_status != 'operating_cash_target_loaded':
                    raise ValueError(
                        f'{CBO_OPERATING_CASH_TARGET_FINANCING_MODE} requires an operating cash path '
                        f'covering {current_date.date()}.'
                    )
                # The TGA lands on its target whenever it would otherwise fall short;
                # above target nothing is issued and the surplus stays in cash. Amounts
                # already decided but not yet settled count as issued.
                total_issuance_target_period = max(
                    0.0,
                    cbo_operating_cash_target
                    - (projected_tga_pre_issuance + cbo_cash_residual_period)
                    - sum(item[3] for item in pending_issuance.values()),
                )
            cbo_fed_row = _cbo_optional_exact_date_row(
                cbo_inputs.get('fed_holdings'),
                cbo_scenario_id,
                current_date,
                'period_end',
                'CBO Fed holdings path',
            )
            if cbo_fed_row is not None:
                cbo_fed_allocation_override_active = True
                # The Fed does not operate while the securities market is closed: its
                # target is then what it holds (TIPS accretion and maturities paid on
                # Fedwire days move that), and it returns to its path on the next
                # market day. Securities that matured on a closed day are not yet paid,
                # so no trade may touch them.
                cbo_fed_holdings_target = (
                    float(cbo_fed_row['cbo_fed_holdings_target_bil'])
                    if market_open
                    else float(
                        bond_portfolio.loc[
                            (bond_portfolio['Status'] == 'Active') & (bond_portfolio['HolderType'] == 'CB')
                        ].apply(_bond_debt_base, axis=1).sum()
                    )
                )
                results.loc[current_date, 'CBOFedHoldingsTargetApplicable'] = 1.0
            # The operating-cash mode sizes proceeds, not face; its need is CashFinancingProceeds.
            results.loc[current_date, 'CBORequiredFaceIssuance'] = (
                total_issuance_target_period if cbo_face_target_mode else 0.0
            )
            results.loc[current_date, 'CBOBuybackFaceRetired'] = cbo_buyback_face_retired
            results.loc[current_date, 'CBOBuybackCashPaid'] = cbo_buyback_cash_paid
            results.loc[current_date, 'CBOFedHoldingsTarget'] = cbo_fed_holdings_target
            results.loc[current_date, 'CBOFedAuctionShare'] = cbo_fed_auction_share
            if cbo_fed_allocation_override_active:
                results.loc[current_date, 'CBOFedStockMode'] = (
                    'synthetic_cb_treasury_stock_target_beneficial_holder'
                )
                results.loc[current_date, 'CBOFedSettlementScope'] = (
                    'beneficial_holder_with_model_dirty_value_settlement'
                )
                results.loc[current_date, 'CBOFedAcquisitionChannel'] = 'none_required_to_hit_cbo_stock_path'
            results.loc[current_date, 'CBOCashResidualStatus'] = cbo_cash_residual_status
        new_bonds_added_list = []
        total_issued_face_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        total_issued_stock_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        total_issued_proceeds_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        fed_rollover_face_issued = 0.0
        fed_rollover_proceeds = 0.0
        total_issued_proceeds_by_private_route = _zero_private_routes()
        actual_issued_amount = 0.0
        actual_auction_proceeds = 0.0
        issue_discount_cost_period = 0.0
        results.loc[current_date, 'IssuanceProceedsTarget'] = 0.0 if cbo_funding_mode else total_issuance_target_period
        # Period scope: a period can transfer Fed stock without issuing, and the TIPS
        # real curve must still be resolvable for that settlement.
        tips_real_curve_years = []
        tips_real_curve_rates = []
        if cbo_funding_mode and isinstance(cbo_inputs, dict):
            tips_real_curve_years, tips_real_curve_rates, _tips_real_curve_status = _tips_real_curve_for_date(
                cbo_inputs.get('tips_real_yield_path'),
                current_date,
                cbo_scenario_id,
            )
        else:
            tips_real_curve_years, tips_real_curve_rates = _generic_tips_real_curve(tips_p)
        issuance_pricing = _issuance_pricing_context(
            tips_real_curve_years=tips_real_curve_years,
            tips_real_curve_rates=tips_real_curve_rates,
            current_yield_curve_years=current_yield_curve_years,
            current_yield_curve_rates=current_yield_curve_rates,
            yield_interpolation_method=yield_interpolation_method,
            yield_floor_zero=yield_floor_zero,
            evaluated_nominal_shock=evaluated_nominal_shock,
            tips_real_coupon=tips_real_coupon,
            frn_spread=frn_spread,
            cbo_funding_mode=cbo_funding_mode,
            cbo_inputs=cbo_inputs,
            cbo_scenario_id=cbo_scenario_id,
        )
        if is_rule_profile(issuance_profile) != opening_issuance_rule:
            # Calendar settlement, pending balances and the row budget are fixed from the
            # opening rule; switching rules mid-run would strand or double-count issuance.
            raise ValueError(
                f'the issuance rule changed during the run at {current_date.date()}; '
                'events may not switch between the bill-share rule and fixed shares'
            )
        # The bill-share rule measures debt after the day's maturities and before issuance.
        pre_issuance_position = (
            controlled_debt_and_bills(bond_portfolio)
            if is_rule_profile(issuance_profile)
            else None
        )
        if cbo_operating_cash_financing_mode:
            primary_issuance_leg = 'operating_cash_target_financing'
        elif cbo_reference_financing_mode:
            primary_issuance_leg = 'cbo_reference_face_issuance'
        else:
            primary_issuance_leg = 'ordinary_issuance'
        early_settlement_proceeds = 0.0
        cash_management_bill_proceeds = 0.0
        close_books_today = False
        rollover_prices = {}
        rollover_stock_by_class = {}
        estimated_rollover = {}
        fed_rollover_cash_estimate = 0.0
        initial_rollover_cash_estimate = 0.0
        fed_rollover_reallocated_stock = 0.0
        fed_only_rollover_classes = []
        if batched_settlement and cbo_fed_allocation_override_active:
            # The Fed rolls what matured today over at the day's auctions, bills into
            # bills and coupon securities into coupon securities, as far as its path needs
            # it; on a falling path the rest runs off before any sale.
            cb_matured = cb_stock_matured
            cb_stock = float(
                bond_portfolio.loc[
                    (bond_portfolio['Status'] == 'Active') & (bond_portfolio['HolderType'] == 'CB')
                ].apply(_bond_debt_base, axis=1).sum()
            )
            rollover_stock = min(cb_matured, max(0.0, cbo_fed_holdings_target - cb_stock))
            rolled_share = rollover_stock / cb_matured if cb_matured > TGA_FLOOR_TOLERANCE else 0.0
            rollover_stock_by_class = {
                'bills': cb_matured_bills * rolled_share,
                'coupons': (cb_matured - cb_matured_bills) * rolled_share,
            }
            rollover_keys = sorted(
                key for key in due_events
                if rollover_stock_by_class['bills' if key[0] == 'bill' else 'coupons'] > TGA_FLOOR_TOLERANCE
            )
            if rollover_keys:
                rollover_prices = _settlement_unit_prices(
                    rollover_keys,
                    issuance_profile=issuance_profile,
                    due_events=due_events,
                    current_date=current_date,
                    pricing=issuance_pricing,
                    calendar_rules=calendar_rules,
                    issue_registry=issue_registry,
                    bond_portfolio=bond_portfolio,
                    reference_cpi_on=lambda day: float(results.loc[pd.Timestamp(day).normalize(), 'Reference_CPI']),
                    frn_accrual_per_face=frn_accrual_per_face,
                )
            # Settlement needs Fed cash before the public offer sizes exist. Pending face
            # supplies a sizing estimate. After the public decision, its realized face
            # determines the split. Same-day TGA cash gets the realized proceeds, including
            # the difference from pricing a fixed amount of stock across different issues.
            estimated_rollover, _ = _fed_rollover_allocation(
                rollover_stock_by_class,
                offering_face={key: pending_issuance[key][3] / rollover_prices[key][0]
                               for key in rollover_keys if key in pending_issuance},
                issuance_profile=issuance_profile,
                prices=rollover_prices,
            )
            initial_rollover_cash_estimate = sum(face * price for face, price, _ in estimated_rollover.values())
            fed_rollover_cash_estimate = initial_rollover_cash_estimate
            fed_rollover_cash = fed_rollover_cash_estimate
            fed_rollover_stock = sum(face * stock for face, _price, stock in estimated_rollover.values())
            fed_redeemed_face = cb_matured - fed_rollover_stock
            # The add-ons count as issued before the public decision, in debt and in bills.
            pre_issuance_position = (
                pre_issuance_position[0] + fed_rollover_stock,
                pre_issuance_position[1]
                + sum(face * stock for key, (face, _price, stock) in estimated_rollover.items() if key[0] == 'bill'),
            )
            if cbo_operating_cash_financing_mode:
                total_issuance_target_period = max(
                    0.0,
                    cbo_operating_cash_target
                    - (projected_tga_pre_issuance + cbo_cash_residual_period + fed_rollover_cash)
                    - sum(item[3] for item in pending_issuance.values()),
                )
        if batched_settlement:
            if issuance_profile.get('settlement_calendar') != settlement_calendar_data:
                raise ValueError('the settlement calendar cannot change during a run')
            close_index = bisect.bisect_left(fiscal_year_close_dates, current_date)
            close_day = fiscal_year_close_dates[close_index] if close_index < len(fiscal_year_close_dates) else None
            close_books_today = current_date == close_day
            if not payments_open:
                if total_issuance_target_period > TGA_FLOOR_TOLERANCE:
                    raise RuntimeError(
                        f'operating cash needs {total_issuance_target_period} billion on {current_date.date()}, '
                        'a day Fedwire is closed'
                    )
                total_issuance_target_period = 0.0
            quote_settlements = functools.partial(
                _quote_settlements,
                due_events=due_events,
                issuance_leg=primary_issuance_leg,
                current_date=current_date,
                pricing=issuance_pricing,
                calendar_rules=calendar_rules,
                issue_registry=issue_registry,
                bond_portfolio=bond_portfolio,
                reference_cpi_on=lambda day: float(results.loc[pd.Timestamp(day).normalize(), 'Reference_CPI']),
                frn_accrual_per_face=frn_accrual_per_face,
            )
            net_cash_before_rollover = projected_tga_pre_issuance + cbo_cash_residual_period
            # Re-size if the price difference would leave a cash gap over a
            # Fedwire closure, at fiscal close, or below the TGA floor.
            retry_possible = bool(rollover_prices)
            next_day_closed = retry_possible and not is_fedwire_business_day(current_date + pd.Timedelta(days=1))
            if retry_possible:
                previous_pending = {key: item.copy() for key, item in pending_issuance.items()}
                previous_registry = issue_registry.copy()
            for attempt in range(16):
                if attempt:
                    pending_issuance.clear()
                    pending_issuance.update({key: item.copy() for key, item in previous_pending.items()})
                    issue_registry.clear()
                    issue_registry.update(previous_registry)
                    fed_rollover_cash_estimate = fed_rollover_cash
                    total_issuance_target_period = max(
                        0.0, cbo_operating_cash_target - net_cash_before_rollover - fed_rollover_cash_estimate
                        - sum(item[3] for item in pending_issuance.values())
                    )
                settlement_allocations, early_settlement_proceeds = _decide_and_settle_pending(
                    pending_issuance,
                    need=total_issuance_target_period,
                    issuance_profile=issuance_profile,
                    pre_issuance_position=pre_issuance_position,
                    due_keys=list(due_events),
                    projected_tga_without_settlement=net_cash_before_rollover + fed_rollover_cash_estimate,
                    current_date=current_date,
                    last_settlements=close_settlement_plan.get(close_day),
                    fiscal_close_today=close_books_today,
                    can_settle=market_open,
                )
                operating_cash_settlement_proceeds = sum(item[-1] for item in settlement_allocations)
                issuance_supply_schedule = quote_settlements(
                    settlement_allocations,
                    off_calendar_kind={item[0]: 'early_settlement' for item in settlement_allocations},
                )
                fed_rollover, fed_only_rollover_classes = _fed_rollover_allocation(
                    rollover_stock_by_class,
                    offering_face={item['supply_key']: item['face_amount'] for item in issuance_supply_schedule},
                    issuance_profile=issuance_profile,
                    prices=rollover_prices,
                )
                fed_rollover_cash = sum(face * price for face, price, _ in fed_rollover.values())
                expected_tga = net_cash_before_rollover + fed_rollover_cash + operating_cash_settlement_proceeds
                if close_books_today:
                    on_cash_path = abs(expected_tga - cbo_operating_cash_target) <= TGA_FLOOR_TOLERANCE
                elif next_day_closed:
                    pending_cash = sum(item[3] for item in pending_issuance.values())
                    on_cash_path = expected_tga + pending_cash >= cbo_operating_cash_target - TGA_FLOOR_TOLERANCE
                else:
                    on_cash_path = True
                if on_cash_path and expected_tga >= -TGA_FLOOR_TOLERANCE:
                    break
                if not retry_possible or abs(fed_rollover_cash - fed_rollover_cash_estimate) <= TGA_FLOOR_TOLERANCE:
                    break  # The cash checks below report a gap issuance could not close.
            cash_management_bill_proceeds = sum(
                item['proceeds'] for item in issuance_supply_schedule
                if item['settlement_kind'] == CASH_MANAGEMENT_BILL_KIND
            )
            fed_rollover_reallocated_stock = 0.5 * sum(
                abs(fed_rollover.get(key, (0.0,))[0] - estimated_rollover.get(key, (0.0,))[0]) * stock_per_face
                for key, (_, stock_per_face) in rollover_prices.items()
            )
            for key, (fed_face, fed_price, _fed_stock) in fed_rollover.items():
                item = next((item for item in issuance_supply_schedule if item['supply_key'] == key), None)
                if item is None:
                    # Nothing settles to the public in this security today: the Fed's
                    # add-on is the whole issue.
                    item = _quote_settlements(
                        [(key, *_supply_terms(key, issuance_profile), fed_face * fed_price)],
                        due_events=due_events,
                        off_calendar_kind={},
                        issuance_leg=FED_ROLLOVER_ISSUANCE_LEG,
                        current_date=current_date,
                        pricing=issuance_pricing,
                        calendar_rules=calendar_rules,
                        issue_registry=issue_registry,
                        bond_portfolio=bond_portfolio,
                        reference_cpi_on=lambda day: float(results.loc[pd.Timestamp(day).normalize(), 'Reference_CPI']),
                        frn_accrual_per_face=frn_accrual_per_face,
                    )[0]
                    item.update(face_amount=0.0, proceeds=0.0)
                    issuance_supply_schedule.append(item)
                if abs(float(item['issue_price_ratio']) - fed_price) > 1e-12:
                    raise RuntimeError(f'the Fed add-on in {key} on {current_date.date()} is priced off its issue')
                item['fed_face'] = fed_face
        else:
            operating_cash_settlement_proceeds = total_issuance_target_period
            issuance_supply_schedule = _schedule_issuance_leg(
                total_issuance_target_period,
                face_target_mode=cbo_face_target_mode,
                issuance_leg=primary_issuance_leg,
                issuance_profile=issuance_profile,
                current_date=current_date,
                pricing=issuance_pricing,
                pre_issuance_position=pre_issuance_position,
                scheduled=(),
            )
        if cbo_reference_financing_mode:
            reference_cash_proceeds = sum(
                float(item['proceeds']) for item in issuance_supply_schedule
            )
            cash_closure_target = float(
                funding_rule_cfg.get('cash_closure_target_bil', 0.0) or 0.0
            )
            projected_after_reference = (
                projected_tga_pre_issuance
                + cbo_cash_residual_period
                + reference_cash_proceeds
            )
            cash_financing_proceeds_target = max(
                0.0, cash_closure_target - projected_after_reference
            )
            financing_schedule = _schedule_issuance_leg(
                cash_financing_proceeds_target,
                face_target_mode=False,
                issuance_leg='tga_floor_cash_financing',
                issuance_profile=issuance_profile,
                current_date=current_date,
                pricing=issuance_pricing,
                pre_issuance_position=pre_issuance_position,
                scheduled=issuance_supply_schedule,
            )
            expected_cash_financing_face = sum(
                float(item['face_amount']) for item in financing_schedule
            )
            expected_cash_financing_proceeds = sum(
                float(item['proceeds']) for item in financing_schedule
            )
            if abs(expected_cash_financing_proceeds - cash_financing_proceeds_target) > target_tolerance:
                raise RuntimeError(
                    f"TGA-floor financing proceeds miss at {current_date.date()}: "
                    f"target={cash_financing_proceeds_target:.12f}, "
                    f"actual={expected_cash_financing_proceeds:.12f}"
                )
            issuance_supply_schedule.extend(financing_schedule)
        if issuance_supply_schedule:
            effective_auction_prefs_for_period = holder_preferences_for_period(
                holder_absorption_lookup,
                scenario_name=scenario_name,
                resolved_scenario_id=holder_absorption_source_scenario_id,
                current_date=current_date,
                fallback_preferences=current_auction_prefs,
                warning_cache=holder_absorption_warning_cache,
            )
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
                rate_sensitive_auction_enabled = (
                    isinstance(rate_sensitive_p, dict)
                    and bool(rate_sensitive_p.get('enabled', False))
                )
                if rate_sensitive_auction_enabled:
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
                        nominal_shock=evaluated_nominal_shock,
                    )
                else:
                    item_yield = 0.0
                    anchor_yield = 0.0
                    curve_slope = 0.0
                holder_share_weights = {}
                for holder in HOLDER_TYPES:
                    holder_prefs = effective_auction_prefs_for_period.get(holder, {})
                    base_pref_pct = float(holder_prefs.get(f'{pref_category_key}_pct', 0.0) or 0.0)
                    multiplier = _compute_rate_sensitive_multiplier(
                        rate_sensitive_p, 'auction', holder, pref_category_key, item_yield, anchor_yield, curve_slope
                    )
                    holder_share_weights[holder] = max(0.0, base_pref_pct * multiplier)
                total_weight = sum(holder_share_weights.values())
                split = _private_subbucket_split(effective_auction_prefs_for_period, pref_category_key)
                if split[PRIVATE_SUBBUCKET_MMF] > 0.0 and (
                    item_type == 'NonMarketable'
                    or not mmf_may_acquire(item_type, supply_item['maturity_date'], current_date)
                ):
                    holder_share_weights['Private'] *= split[PRIVATE_SUBBUCKET_DOMESTIC_NONBANK]
                    split = {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK: 1.0}
                    eligible_weight = sum(holder_share_weights.values())
                    if total_weight > TGA_FLOOR_TOLERANCE and eligible_weight <= TGA_FLOOR_TOLERANCE:
                        raise ValueError(f'No eligible auction buyers for {item_type} maturing {supply_item["maturity_date"]}')
                    total_weight = eligible_weight
                supply_item['private_route_split'] = split
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
            for supply_item in issuance_supply_schedule:
                fed_face = float(supply_item.get('fed_face', 0.0))
                if fed_face <= TGA_FLOOR_TOLERANCE:
                    continue
                item_ref = next(
                    (ref for ref, details in item_details_by_item_ref.items() if details is supply_item),
                    None,
                )
                if item_ref is None:
                    item_ref = next_item_ref_id
                    item_details_by_item_ref[item_ref] = supply_item
                    next_item_ref_id += 1
                allocations_by_holder_item_ref['CB'][item_ref] += fed_face
            # Secondary transfers can split a holding into a newly identified
            # tranche after the prior period's auction. Reconcile the auction
            # counter to the live portfolio before assigning the next IDs.
            bond_id_counter = max(
                bond_id_counter,
                _next_portfolio_bond_id(bond_portfolio),
            )
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
                        route_face_values = [
                            (route, face_value_issued_to_holder * share)
                            for route, share in details['private_route_split'].items()
                        ]
                    for holder_subbucket, route_face_value in route_face_values:
                        if route_face_value < TGA_FLOOR_TOLERANCE:
                            continue
                        maturity_date = details['maturity_date']
                        if holder_subbucket == PRIVATE_SUBBUCKET_MMF:
                            record_mmf_acquisition(current_date, sec_type, maturity_date)
                        coupon_rate_val = details.get('coupon', 0.0)
                        fixed_spread_val = details.get('spread', 0.0) if sec_type == 'FRN' else 0.0
                        issue_price_ratio_val = details.get('issue_price_ratio', 1.0)
                        issue_yield_val = details.get('issue_yield', np.nan)
                        issuance_leg_val = (
                            FED_ROLLOVER_ISSUANCE_LEG
                            if holder == 'CB' and details.get('fed_face')
                            else details.get('issuance_leg', 'ordinary_issuance')
                        )
                        is_noncash_intragovernmental_issuance = holder in INTRAGOV_HOLDERS
                        issue_proceeds_val = (
                            0.0
                            if is_noncash_intragovernmental_issuance
                            else route_face_value * issue_price_ratio_val
                        )
                        frn_reference_rate_val = (
                            _frn_reference_rate_for_date(
                                cbo_inputs,
                                cbo_scenario_id,
                                current_date,
                                benchmark_maturity_years=frn_benchmark_mat,
                                yield_curve_years=current_yield_curve_years,
                                yield_curve_rates=current_yield_curve_rates,
                                interpolation_method=yield_interpolation_method,
                                floor_zero=yield_floor_zero,
                                nominal_shock=evaluated_nominal_shock,
                            )
                            if sec_type == 'FRN'
                            else np.nan
                        )
                        interest_payment_frequency_val = 4.0 if sec_type == 'FRN' else (2.0 if sec_type in {'Fixed', 'TIPS'} and coupon_rate_val > TGA_FLOOR_TOLERANCE else np.nan)
                        first_interest_payment_date_val = _first_interest_payment_date(
                            current_date,
                            maturity_date,
                            interest_payment_frequency_val,
                        )
                        # Treasury terms from the settlement calendar: a tranche is issued
                        # on its settlement day; its dated date is the security's (a
                        # reopening's is the reopened security's), the buyer paid what had
                        # accrued since the last coupon, and the next coupon is paid whole.
                        dated_date_val = pd.Timestamp(details.get('dated_date', current_date))
                        original_principal_val = 0.0
                        adjusted_principal_val = 0.0
                        reference_cpi_issue_val = 0.0
                        index_ratio_init_val = 0.0
                        accrued_frn_val = 0.0
                        last_accrual_date_val = pd.NaT
                        if sec_type == 'TIPS':
                            original_principal_val = route_face_value
                            reference_cpi_issue_val = details.get('reference_cpi_issue', np.nan)
                            index_ratio_init_val = details.get('index_ratio', np.nan)
                            if pd.isna(reference_cpi_issue_val):
                                reference_cpi_issue_val = results.loc[current_date, 'Reference_CPI']
                                index_ratio_init_val = 1.0
                            adjusted_principal_val = route_face_value * float(index_ratio_init_val)
                        elif sec_type == 'FRN':
                            last_accrual_date_val = current_date
                            accrued_frn_val = route_face_value * float(details.get('accrued_frn_per_face', 0.0))
                        maturity_category_val = None
                        if sec_type == 'Fixed':
                            maturity_category_val = get_maturity_category(mat_yrs, issuance_profile)
                        new_tranche = {'BondID': next_bond_id_to_assign, 'FaceValue': route_face_value, 'HolderType': holder, 'HolderSubBucket': holder_subbucket if holder == 'Private' else '', 'TDCPrincipalHolderType': holder, 'TDCPrincipalHolderSubBucket': holder_subbucket if holder == 'Private' else '', 'SecurityType': sec_type, 'IssueDate': current_date, 'MaturityDate': maturity_date, 'DatedDate': dated_date_val, 'OriginalDatedDate': dated_date_val, 'FirstInterestPaymentDate': first_interest_payment_date_val, 'InterestPaymentFrequency': interest_payment_frequency_val, 'OriginalMaturityYears': mat_yrs, 'CouponRate': coupon_rate_val, 'Status': 'Active', 'MaturityCategory': maturity_category_val, 'OriginalPrincipal': original_principal_val, 'AdjustedPrincipal': adjusted_principal_val, 'ReferenceCPI_Issue': reference_cpi_issue_val, 'IndexRatio': index_ratio_init_val, 'FixedSpread': fixed_spread_val, 'AccruedInterest_FRN': accrued_frn_val, 'BenchmarkRate_FRN': 0.0, 'LastAccrualDate': last_accrual_date_val, 'IssuePriceRatio': issue_price_ratio_val, 'IssueProceeds': issue_proceeds_val, 'IssueYieldAtIssue': issue_yield_val, 'TimeToMaturity': np.nan, 'DiscountYield': np.nan, 'CleanPrice': np.nan, 'AccruedInterest': np.nan, 'DirtyValue': np.nan, 'DirtyPriceRatio': np.nan}
                        new_tranche['IssueID'] = details.get('security_label')
                        if details.get('settlement_kind') == CASH_MANAGEMENT_BILL_KIND:
                            new_tranche['IsCashManagementBill'] = True
                        new_bonds_added_list.append(new_tranche)
                        tranche_stock = adjusted_principal_val if sec_type == 'TIPS' else route_face_value
                        if is_noncash_intragovernmental_issuance:
                            tranche_issue_cost = 0.0
                        elif 'settlement_kind' in details:
                            # Debt added less cash received, signed: a reopening can sell
                            # at a premium, and accrued interest received offsets the
                            # full first coupon the holder will be paid.
                            tranche_issue_cost = tranche_stock - issue_proceeds_val
                        else:
                            tranche_issue_cost = max(0.0, route_face_value - issue_proceeds_val)
                        issue_discount_cost_period += tranche_issue_cost
                        total_issued_face_by_holder[holder] += route_face_value
                        total_issued_stock_by_holder[holder] += tranche_stock
                        total_issued_proceeds_by_holder[holder] += issue_proceeds_val
                        if issuance_leg_val == FED_ROLLOVER_ISSUANCE_LEG:
                            fed_rollover_face_issued += route_face_value
                            fed_rollover_proceeds += issue_proceeds_val
                        if holder == 'Private':
                            total_issued_proceeds_by_private_route[holder_subbucket] += issue_proceeds_val
                        actual_issued_amount += route_face_value
                        actual_auction_proceeds += issue_proceeds_val
                        issue_price_cash_gap += route_face_value - issue_proceeds_val
                        if issuance_leg_val in CASH_FINANCING_ISSUANCE_LEGS:
                            cash_financing_face_issued += route_face_value
                            cash_financing_proceeds += issue_proceeds_val
                        _handoff_append_issuance(
                            handoff_tables,
                            prev_date,
                            current_date,
                            security_type=sec_type,
                            maturity_years=mat_yrs,
                            face_issued=route_face_value,
                            cash_proceeds=issue_proceeds_val,
                            coupon_rate=coupon_rate_val,
                            reference_rate=frn_reference_rate_val,
                            spread_bps=fixed_spread_val * 10000.0 if sec_type == 'FRN' else np.nan,
                            issue_yield=issue_yield_val,
                            security_id=next_bond_id_to_assign,
                            holder_sector=holder,
                            holder_subsector=holder_subbucket if holder == 'Private' else '',
                            issuance_leg=issuance_leg_val,
                            issue_cost=tranche_issue_cost,
                            adjusted_principal_issued=adjusted_principal_val if sec_type == 'TIPS' else None,
                            issue_terms=(
                                {
                                    'security_label': details['security_label'],
                                    'maturity_date': str(pd.Timestamp(maturity_date).date()),
                                    'dated_date': str(dated_date_val.date()),
                                    'is_reopening': bool(details['is_reopening']),
                                    'settlement_kind': details['settlement_kind'],
                                }
                                if 'settlement_kind' in details
                                else None
                            ),
                        )
                        next_bond_id_to_assign += 1
            bond_id_counter = next_bond_id_to_assign
            if new_bonds_added_list:
                new_bonds_df = pd.DataFrame(new_bonds_added_list)
                for col in BOND_PORTFOLIO_COLS:
                    if col not in new_bonds_df.columns:
                        dtype = PORTFOLIO_DTYPES.get(col, 'object')
                        new_bonds_df[col] = pd.NA if dtype == 'Int64' else pd.NaT if dtype == 'datetime64[ns]' else np.nan
                runtime_columns = list(BOND_PORTFOLIO_COLS)
                if 'IsCashManagementBill' in new_bonds_df:
                    new_bonds_df['IsCashManagementBill'] = new_bonds_df['IsCashManagementBill'].eq(True)
                    runtime_columns.append('IsCashManagementBill')
                new_bonds_df = new_bonds_df[runtime_columns]
                new_bonds_df = new_bonds_df.astype(PORTFOLIO_DTYPES, errors='ignore')
                marketable = new_bonds_df['SecurityType'].isin(PUBLIC_MARKETABLE_SECURITY_TYPES)
                marketable_lots = new_bonds_df.loc[marketable]
                new_bonds_df.loc[marketable, 'IssueID'] = fed_purchase_policy.issue_ids(marketable_lots, bill_mask(marketable_lots))
                if bounded_handoff:
                    handoff_tables.check_portfolio_rows(
                        len(bond_portfolio),
                        pending=len(new_bonds_df),
                    )
                try:
                    if bond_portfolio.empty:
                        bond_portfolio = new_bonds_df
                    else:
                        bond_portfolio = pd.concat([bond_portfolio, new_bonds_df], ignore_index=True)
                except Exception:
                    raise
        results.loc[current_date, 'NewDebtIssued'] = actual_issued_amount
        results.loc[current_date, 'AuctionProceeds'] = actual_auction_proceeds
        results.loc[current_date, 'IssuePriceCashGap'] = issue_price_cash_gap
        if cbo_reference_financing_mode:
            if abs(cash_financing_proceeds - cash_financing_proceeds_target) > target_tolerance:
                raise RuntimeError(
                    f"TGA-floor financing cash settlement miss at {current_date.date()}: "
                    f"target={cash_financing_proceeds_target:.12f}, "
                    f"actual={cash_financing_proceeds:.12f}"
                )
            if abs(cash_financing_face_issued - expected_cash_financing_face) > target_tolerance:
                raise RuntimeError(
                    f"TGA-floor financing face allocation miss at {current_date.date()}"
                )
            results.loc[current_date, 'CashFinancingFaceIssued'] = cash_financing_face_issued
            results.loc[current_date, 'CashFinancingProceeds'] = cash_financing_proceeds
        if cbo_operating_cash_financing_mode:
            if abs(cash_financing_proceeds - operating_cash_settlement_proceeds) > target_tolerance:
                raise RuntimeError(
                    f"Operating-cash financing cash settlement miss at {current_date.date()}: "
                    f"target={operating_cash_settlement_proceeds:.12f}, "
                    f"actual={cash_financing_proceeds:.12f}"
                )
            expected_cash_financing_face = sum(
                float(item['face_amount']) for item in issuance_supply_schedule
            )
            if abs(cash_financing_face_issued - expected_cash_financing_face) > target_tolerance:
                raise RuntimeError(
                    f"Operating-cash financing face allocation miss at {current_date.date()}"
                )
            results.loc[current_date, 'CashFinancingFaceIssued'] = cash_financing_face_issued
            results.loc[current_date, 'CashFinancingProceeds'] = cash_financing_proceeds
            results.loc[current_date, 'IssuanceNeedProceeds'] = total_issuance_target_period
            results.loc[current_date, 'PendingIssuanceProceeds'] = sum(
                item[3] for item in pending_issuance.values()
            )
            results.loc[current_date, 'EarlySettlementProceeds'] = early_settlement_proceeds
            results.loc[current_date, 'CashManagementBillProceeds'] = cash_management_bill_proceeds
            results.loc[current_date, 'FedRolloverFace'] = fed_rollover_face_issued
            results.loc[current_date, 'FedRolloverProceeds'] = fed_rollover_proceeds
            results.loc[current_date, 'FedRolloverCashSizingDifference'] = fed_rollover_proceeds - fed_rollover_cash_estimate
            results.loc[current_date, 'FedRolloverCashEstimateRevision'] = (
                fed_rollover_cash_estimate - initial_rollover_cash_estimate
            )
            results.loc[current_date, 'FedRolloverReallocatedStock'] = fed_rollover_reallocated_stock
            if fed_only_rollover_classes:
                results.loc[current_date, 'FedRolloverFedOnlyClasses'] = '|'.join(fed_only_rollover_classes)
            results.loc[current_date, 'FedRedeemedFace'] = fed_redeemed_face
            if abs(fed_rollover_proceeds - fed_rollover_cash) > target_tolerance:
                raise RuntimeError(f'the Fed add-on cash at {current_date.date()} differs from what the allocation priced')
        if cbo_funding_mode:
            results.loc[current_date, 'IssuanceProceedsTarget'] = actual_auction_proceeds
            cbo_fed_auction_addons_current = float(total_issued_face_by_holder.get('CB', 0.0))
            cbo_fed_auction_share = (
                cbo_fed_auction_addons_current / actual_issued_amount
                if actual_issued_amount > TGA_FLOOR_TOLERANCE
                else 0.0
            )
            results.loc[current_date, 'CBOFedAuctionShare'] = cbo_fed_auction_share
            if (
                cbo_fed_allocation_override_active
                and not batched_settlement
                and abs(cbo_fed_auction_addons_current) > TGA_FLOOR_TOLERANCE
            ):
                # Off Treasury's settlement calendar the Fed path is met in the secondary
                # market only; on it the Fed rolls maturing holdings at auction.
                raise RuntimeError(
                    f"CBO Fed holdings target must be met without Fed auction issuance at {current_date.date()}: "
                    f"cb_auction_face={cbo_fed_auction_addons_current:.12f}, "
                    f"total_issued_face={actual_issued_amount:.12f}"
                )
        results.loc[current_date, 'IssueDiscountCost_Period'] = issue_discount_cost_period
        results.loc[current_date, 'IssueDiscountCost_Cumulative'] = results.loc[prev_date, 'IssueDiscountCost_Cumulative'] + issue_discount_cost_period
        results.loc[current_date, 'FinancingCost_Period'] = results.loc[current_date, 'InterestOutlay_Period'] + issue_discount_cost_period + nonmkt_interest_capitalized_period + tips_inflation_accretion_period
        results.loc[current_date, 'FinancingCost_Cumulative'] = results.loc[prev_date, 'FinancingCost_Cumulative'] + results.loc[current_date, 'FinancingCost_Period']
        issuance_tga_change = actual_auction_proceeds
        issuance_reserve_change = -(total_issued_proceeds_by_holder.get('Banks', 0.0) + total_issued_proceeds_by_holder.get('Private', 0.0) + total_issued_proceeds_by_holder.get('Foreign', 0.0))
        issuance_deposit_change = -_effective_private_amount(
            total_issued_proceeds_by_private_route,
            mmf_deposit_pass_through,
        )
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type='settlement',
            leg_type='auction_counterparty_settlement',
            accounting_basis='reserve_and_deposit_settlement',
            holder_sector='aggregate_auction_holders',
            counterparty_sector='Treasury',
            instrument_type='all_treasury',
            maturity_bucket='all',
            reserve_change=issuance_reserve_change,
            deposit_change=issuance_deposit_change,
            settlement_scope='auction_cash_settlement',
        )
        reserve_change_period += issuance_reserve_change
        if trc_ledger is not None:
            trc_ledger.record_holder_cash(
                'AuctionSettlement',
                {
                    holder: -total_issued_proceeds_by_holder.get(holder, 0.0)
                    for holder in ('Banks', 'Private', 'Foreign')
                },
                {route: -value for route, value in total_issued_proceeds_by_private_route.items()},
            )
        deposit_change_period += issuance_deposit_change
        tga_change_period += issuance_tga_change
        if cbo_funding_mode:
            tga_change_period += cbo_cash_residual_period
            _handoff_append_journal(
                handoff_tables,
                prev_date,
                current_date,
                event_type='cash_reconciliation',
                leg_type='explicit_cbo_cash_reconciliation_residual',
                accounting_basis='treasury_cash_named_nonfunding_adjustment',
                holder_sector='Treasury',
                counterparty_sector='cash_reconciliation_source',
                instrument_type='cash',
                maturity_bucket='not_applicable',
                treasury_cash_change=cbo_cash_residual_period,
                settlement_scope='named_reconciliation_leg_requires_hard_boundary_validation',
            )
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
                fed_transfer_before = {
                    'holder': _handoff_portfolio_stock_map(bond_portfolio),
                    'route': _handoff_portfolio_stock_map(bond_portfolio, route=True),
                }
                purchase_operation = {}
                (
                    bond_portfolio,
                    fed_secondary_purchase_face,
                    fed_secondary_sellers,
                    fed_secondary_private_routes,
                    fed_secondary_purchase_cash,
                ) = _transfer_public_marketable_debt_to_cb(
                    bond_portfolio,
                    fed_purchase_shortfall,
                    current_date,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    interpolation_method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                    nominal_shock=evaluated_nominal_shock,
                    tips_real_curve_years=tips_real_curve_years,
                    tips_real_curve_rates=tips_real_curve_rates,
                    operation_summary=purchase_operation,
                    portfolio_row_budget=(
                        handoff_tables.portfolio_row_budget
                        if bounded_handoff
                        else None
                    ),
                )
                _handoff_append_transfer_changes(
                    handoff_tables,
                    prev_date,
                    current_date,
                    fed_transfer_before,
                    bond_portfolio,
                    event_type='fed_secondary_transfer',
                    settlement_scope='fed_secondary_purchase',
                )
                if abs(fed_secondary_purchase_face - fed_purchase_shortfall) > float(
                    funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001
                ):
                    raise RuntimeError(
                        f"CBO Fed holdings target cannot be met by secondary purchases at {current_date.date()}: "
                        f"target={cbo_fed_holdings_target:.12f}, current={current_cb_after_issuance:.12f}, "
                        f"purchased={fed_secondary_purchase_face:.12f}"
                    )
                fed_secondary_reserve_effect = sum(
                    fed_secondary_sellers.get(holder, 0.0)
                    for holder in ('Banks', 'Private', 'Foreign')
                )
                fed_secondary_deposit_effect = _effective_private_amount(
                    fed_secondary_private_routes,
                    mmf_deposit_pass_through,
                    mmf_added_deposit_share=mmf_added_deposit_share,
                )
                for route in PRIVATE_SUBBUCKETS:
                    fed_secondary_private_route_effects[route] += fed_secondary_private_routes[route]
                reserve_change_period += fed_secondary_reserve_effect
                if trc_ledger is not None:
                    trc_ledger.record_holder_cash(
                        'FedSecondary',
                        {holder: fed_secondary_sellers.get(holder, 0.0) for holder in ('Banks', 'Private', 'Foreign')},
                        fed_secondary_private_routes,
                    )
                deposit_change_period += fed_secondary_deposit_effect
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='settlement',
                    leg_type='fed_secondary_purchase_settlement',
                    accounting_basis='reserve_and_deposit_settlement',
                    holder_sector='aggregate_secondary_sellers',
                    counterparty_sector='CB',
                    instrument_type='all_treasury',
                    maturity_bucket='all',
                    reserve_change=fed_secondary_reserve_effect,
                    deposit_change=fed_secondary_deposit_effect,
                    settlement_scope='fed_secondary_purchase',
                )
                results.loc[current_date, 'CBOFedSecondaryPurchaseReserveEffect'] = fed_secondary_reserve_effect
                results.loc[current_date, 'CBOFedSecondaryPurchaseDepositEffect'] = fed_secondary_deposit_effect
                results.loc[current_date, 'CBOFedSecondaryPurchaseFace'] = fed_secondary_purchase_face
                results.loc[current_date, 'CBOFedSecondaryPurchaseCash'] = fed_secondary_purchase_cash
                results.loc[current_date, 'CBOFedPurchaseOperation'] = json.dumps(purchase_operation, sort_keys=True)
                results.loc[current_date, 'CBOFedAcquisitionChannel'] = (
                    'synthetic_secondary_purchase_to_hit_cbo_stock_path'
                )
            elif fed_purchase_shortfall < -fed_target_tolerance:
                fed_secondary_sale_buyer_mix = _resolve_fed_secondary_sale_buyer_mix(
                    bond_portfolio,
                    funding_rule_cfg.get('fed_secondary_sale_buyer_mix'),
                )
                fed_transfer_before = {
                    'holder': _handoff_portfolio_stock_map(bond_portfolio),
                    'route': _handoff_portfolio_stock_map(bond_portfolio, route=True),
                }
                (
                    bond_portfolio,
                    fed_secondary_sale_face,
                    fed_secondary_buyers,
                    fed_secondary_buyer_private_routes,
                    fed_secondary_sale_cash,
                ) = _transfer_cb_marketable_debt_to_public(
                    bond_portfolio,
                    abs(fed_purchase_shortfall),
                    current_date,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    fed_secondary_sale_buyer_mix,
                    record_mmf_acquisition=record_mmf_acquisition,
                    interpolation_method=yield_interpolation_method,
                    floor_zero=yield_floor_zero,
                    nominal_shock=evaluated_nominal_shock,
                    tips_real_curve_years=tips_real_curve_years,
                    tips_real_curve_rates=tips_real_curve_rates,
                    portfolio_row_budget=(
                        handoff_tables.portfolio_row_budget
                        if bounded_handoff
                        else None
                    ),
                )
                _handoff_append_transfer_changes(
                    handoff_tables,
                    prev_date,
                    current_date,
                    fed_transfer_before,
                    bond_portfolio,
                    event_type='fed_secondary_transfer',
                    settlement_scope='fed_secondary_sale',
                )
                if abs(fed_secondary_sale_face - abs(fed_purchase_shortfall)) > fed_target_tolerance:
                    raise RuntimeError(
                        f"CBO Fed holdings target cannot be met by synthetic sales at {current_date.date()}: "
                        f"target={cbo_fed_holdings_target:.12f}, current={current_cb_after_issuance:.12f}, "
                        f"sold={fed_secondary_sale_face:.12f}"
                    )
                fed_secondary_reserve_effect = -sum(
                    fed_secondary_buyers.get(holder, 0.0)
                    for holder in ('Banks', 'Private', 'Foreign')
                )
                fed_secondary_deposit_effect = -_effective_private_amount(
                    fed_secondary_buyer_private_routes,
                    mmf_deposit_pass_through,
                    mmf_added_deposit_share=mmf_added_deposit_share,
                )
                for route in PRIVATE_SUBBUCKETS:
                    fed_secondary_private_route_effects[route] -= fed_secondary_buyer_private_routes[route]
                reserve_change_period += fed_secondary_reserve_effect
                if trc_ledger is not None:
                    trc_ledger.record_holder_cash(
                        'FedSecondary',
                        {holder: -fed_secondary_buyers.get(holder, 0.0) for holder in ('Banks', 'Private', 'Foreign')},
                        {route: -value for route, value in fed_secondary_buyer_private_routes.items()},
                    )
                deposit_change_period += fed_secondary_deposit_effect
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='settlement',
                    leg_type='fed_secondary_sale_settlement',
                    accounting_basis='reserve_and_deposit_settlement',
                    holder_sector='aggregate_secondary_buyers',
                    counterparty_sector='CB',
                    instrument_type='all_treasury',
                    maturity_bucket='all',
                    reserve_change=fed_secondary_reserve_effect,
                    deposit_change=fed_secondary_deposit_effect,
                    settlement_scope='fed_secondary_sale',
                )
                results.loc[current_date, 'CBOFedSecondarySaleCash'] = fed_secondary_sale_cash
                results.loc[current_date, 'CBOFedSecondarySaleReserveEffect'] = fed_secondary_reserve_effect
                results.loc[current_date, 'CBOFedSecondarySaleDepositEffect'] = fed_secondary_deposit_effect
                results.loc[current_date, 'CBOFedSecondarySaleBuyerMix'] = _format_fed_holder_mix(
                    fed_secondary_sale_buyer_mix
                )
                results.loc[current_date, 'CBOFedAcquisitionChannel'] = (
                    'synthetic_secondary_sale_to_hit_cbo_stock_path'
                )
        closing_holder_stocks = None
        if holder_target_path is not None:
            if current_enable_trading:
                raise RuntimeError(
                    'Holder position targets close after the Federal Reserve operation and '
                    'cannot run with preference trading, which would move positions afterward.'
                )
            closing_holder_stocks = holder_targets.closing_stocks(bond_portfolio)
            stock_stem = holder_target_path.stock_column_stem
            holder_target_stocks = closing_holder_stocks[stock_stem]
            holder_target_phase, holder_target_vector = holder_target_path.for_date(current_date, holder_target_stocks)
            holder_target_tolerance = holder_target_path.tolerance_bil
            holder_target_gaps = holder_targets.target_gaps(
                holder_target_stocks,
                holder_target_vector,
                tolerance=holder_target_tolerance,
                current_date=current_date,
            )
            holder_target_pre_gap = max(abs(value) for value in holder_target_gaps.values())
            holder_target_report = holder_targets.RebalanceReport()
            for route in holder_target_path.frozen_routes:
                if abs(holder_target_gaps[route]) > holder_target_tolerance:
                    raise holder_targets.HolderTargetError(
                        f"Untraded route {route} differs from the baseline at the {current_date.date()} close."
                    )
                holder_target_gaps[route] = 0.0
            if holder_target_pre_gap > holder_target_tolerance:
                if not market_open:
                    raise holder_targets.HolderTargetError(
                        f"holder position targets move by {holder_target_pre_gap:.9f} billion on "
                        f"{current_date.date()}, a day the securities market is closed"
                    )
                if holder_target_phase == 'pre':
                    raise holder_targets.HolderTargetError(
                        f"holder positions differ from the matched baseline by {holder_target_pre_gap:.9f} "
                        f"billion on {current_date.date()}, before the intervention starts"
                    )
                holder_target_scope = holder_targets.SETTLEMENT_SCOPES[holder_target_phase]
                holder_target_before = {
                    'holder': _handoff_portfolio_stock_map(bond_portfolio),
                    'route': _handoff_portfolio_stock_map(bond_portfolio, route=True),
                }
                bond_portfolio, holder_target_report = holder_targets.rebalance_holder_positions(
                    bond_portfolio,
                    holder_target_gaps,
                    current_date=current_date,
                    price=lambda row, amount: _fed_secondary_dirty_value(
                        row,
                        amount,
                        current_date,
                        current_yield_curve_years,
                        current_yield_curve_rates,
                        interpolation_method=yield_interpolation_method,
                        floor_zero=yield_floor_zero,
                        nominal_shock=evaluated_nominal_shock,
                        tips_real_curve_years=tips_real_curve_years,
                        tips_real_curve_rates=tips_real_curve_rates,
                    ),
                    row_budget=handoff_tables.portfolio_row_budget if bounded_handoff else None,
                    record_mmf_acquisition=record_mmf_acquisition,
                )
                _handoff_append_transfer_changes(
                    handoff_tables,
                    prev_date,
                    current_date,
                    holder_target_before,
                    bond_portfolio,
                    event_type=holder_targets.TRANSFER_EVENT_TYPE,
                    settlement_scope=holder_target_scope,
                )
                for (seller, buyer), (_face, cash) in holder_target_report.pairs.items():
                    if seller[0] == 'Private':
                        holder_target_private_route_effects[seller[1]] += cash
                    if buyer[0] == 'Private':
                        holder_target_private_route_effects[buyer[1]] -= cash
                    if trc_ledger is not None:
                        # Holder-to-holder settlement moves reserves between banks; the
                        # ledger books each side on its own settlement route.
                        trc_ledger.record_trade(
                            'SecondaryTrades',
                            buyer=buyer[0],
                            buyer_private_route=buyer[1] or None,
                            seller=seller[0],
                            seller_private_route=seller[1] or None,
                            amount=cash,
                        )
                holder_target_deposit_effect = _effective_private_amount(
                    holder_target_private_route_effects,
                    mmf_deposit_pass_through,
                    mmf_added_deposit_share=mmf_added_deposit_share,
                )
                deposit_change_period += holder_target_deposit_effect
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='settlement',
                    leg_type=holder_targets.SETTLEMENT_LEG,
                    accounting_basis='reserve_and_deposit_settlement',
                    holder_sector='aggregate_secondary_counterparties',
                    counterparty_sector='secondary_market',
                    instrument_type='all_treasury',
                    maturity_bucket='all',
                    reserve_change=0.0,
                    deposit_change=holder_target_deposit_effect,
                    settlement_scope=holder_target_scope,
                )
                closing_holder_stocks = holder_targets.closing_stocks(bond_portfolio)
            holder_target_values = holder_targets.result_values(
                phase=holder_target_phase,
                targets=holder_target_vector,
                stocks=closing_holder_stocks[stock_stem],
                pre_trade_gap=holder_target_pre_gap,
                report=holder_target_report,
                deposit_effect=holder_target_deposit_effect,
            )
            if holder_target_values['HolderTargetMaxError'] > holder_target_tolerance:
                raise holder_targets.HolderTargetError(
                    f"holder positions miss their target by {holder_target_values['HolderTargetMaxError']:.9f} "
                    f"billion after trading on {current_date.date()}"
                )
            results.loc[current_date, holder_targets.PHASE_COLUMN] = holder_target_phase
            for column, value in holder_target_values.items():
                results.loc[current_date, column] = value
        if current_enable_trading and market_open and (not cbo_fed_allocation_override_active) and (not bond_portfolio.empty):
            try:
                pref_trade_monetary_impact = {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0}
                bond_portfolio_temp = bond_portfolio.reset_index(drop=True)
                preference_transfer_before = {
                    'holder': _handoff_portfolio_stock_map(bond_portfolio_temp),
                    'route': _handoff_portfolio_stock_map(bond_portfolio_temp, route=True),
                }
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
                bond_portfolio_temp, pref_trade_monetary_impact = execute_preference_trades(
                    bond_portfolio_temp,
                    current_date,
                    current_yield_curve_years,
                    current_yield_curve_rates,
                    effective_secondary_prefs,
                    issuance_profile,
                    scenario_name,
                    tips_real_curve_years=tips_real_curve_years,
                    tips_real_curve_rates=tips_real_curve_rates,
                    trc_ledger=trc_ledger,
                    record_mmf_acquisition=record_mmf_acquisition,
                )
                bond_portfolio = bond_portfolio_temp
                _handoff_append_transfer_changes(
                    handoff_tables,
                    prev_date,
                    current_date,
                    preference_transfer_before,
                    bond_portfolio,
                    event_type='preference_secondary_transfer',
                    settlement_scope='generic_secondary_trading',
                )
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
                _handoff_append_journal(
                    handoff_tables,
                    prev_date,
                    current_date,
                    event_type='settlement',
                    leg_type='secondary_trade_settlement',
                    accounting_basis='treasury_cash_reserve_deposit',
                    holder_sector='aggregate_secondary_counterparties',
                    counterparty_sector='secondary_market',
                    instrument_type='all_treasury',
                    maturity_bucket='all',
                    treasury_cash_change=(
                        pref_trade_monetary_impact.get('tga_change', 0.0)
                        - pref_trade_monetary_impact.get('tga_drain', 0.0)
                    ),
                    reserve_change=pref_trade_monetary_impact.get('reserve_change', 0.0),
                    deposit_change=pref_trade_monetary_impact.get('deposit_change', 0.0),
                    settlement_scope='generic_secondary_trading',
                )
            except Exception:
                raise
        active_bonds_final = bond_portfolio[bond_portfolio['Status'] == 'Active'].copy()
        if not active_bonds_final.empty:
            active_bonds_final['DebtBase'] = np.where(active_bonds_final['SecurityType'] == 'TIPS', active_bonds_final['AdjustedPrincipal'].fillna(active_bonds_final['FaceValue']), active_bonds_final['FaceValue'])
        current_total_debt = active_bonds_final['DebtBase'].sum() if not active_bonds_final.empty else 0.0
        results.loc[current_date, 'TotalDebt_Agg'] = current_total_debt
        if cbo_funding_mode:
            cbo_post_issuance_controlled_debt = _controlled_public_marketable_debt_from_portfolio(active_bonds_final)
            cbo_target_error = cbo_post_issuance_controlled_debt - cbo_controlled_debt_target
            target_tolerance = float(funding_rule_cfg.get('target_tolerance_bil', 0.000001) or 0.000001)
            if cbo_debt_target_mode and abs(cbo_target_error) > target_tolerance:
                raise RuntimeError(
                    f"CBO controlled debt target miss at {current_date.date()}: "
                    f"target={cbo_controlled_debt_target:.12f}, "
                    f"post={cbo_post_issuance_controlled_debt:.12f}, "
                    f"error={cbo_target_error:.12f}"
                )
            results.loc[current_date, 'CBOFundingModeActive'] = 1.0
            results.loc[current_date, 'CBOControlledDebtTarget'] = cbo_controlled_debt_target
            results.loc[current_date, 'CBOControlledDebtReference'] = cbo_controlled_debt_target
            results.loc[current_date, 'ScenarioControlledDebt'] = cbo_post_issuance_controlled_debt
            results.loc[current_date, 'DebtDriftFromReference'] = cbo_target_error
            results.loc[current_date, 'CBOControlledDebtPreIssuance'] = cbo_pre_issuance_controlled_debt
            results.loc[current_date, 'CBOControlledDebtPostIssuance'] = cbo_post_issuance_controlled_debt
            results.loc[current_date, 'CBOControlledDebtTargetError'] = (
                cbo_target_error if cbo_debt_target_mode else 0.0
            )
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
            intragov_excluded = 0.0
            fed_included = 0.0
            if not active_bonds_final.empty:
                intragov_excluded = float(
                    active_bonds_final.loc[
                        active_bonds_final['HolderType'].astype(str).isin(INTRAGOV_HOLDERS),
                        'DebtBase',
                    ].sum()
                )
                fed_included = float(
                    active_bonds_final.loc[
                        active_bonds_final['HolderType'].astype(str).eq('CB'),
                        'DebtBase',
                    ].sum()
                )
            handoff_tables['tdcsim_debt_target_bridge'].append(
                {
                    'date': str(pd.Timestamp(current_date).date()),
                    'cbo_public_debt_target_bil': _cbo_value(
                        cbo_debt_row,
                        'cbo_federal_debt_held_public_target_bil',
                    ),
                    'public_nonmarketable_bridge_bil': _cbo_value(
                        cbo_debt_row,
                        'public_nonmarketable_treasury_bil',
                    ),
                    'non_treasury_and_definition_bridge_bil': _cbo_value(
                        cbo_debt_row,
                        'non_treasury_and_definition_residual_bil',
                    ),
                    'controlled_public_marketable_target_bil': cbo_controlled_debt_target,
                    'cbo_controlled_debt_reference_bil': cbo_controlled_debt_target,
                    'scenario_controlled_debt_bil': cbo_post_issuance_controlled_debt,
                    'debt_drift_from_reference_bil': cbo_target_error,
                    'cash_financing_face_issued_bil': cash_financing_face_issued,
                    'cash_financing_proceeds_bil': cash_financing_proceeds,
                    'controlled_debt_pre_issuance_bil': cbo_pre_issuance_controlled_debt,
                    'face_issued_bil': actual_issued_amount,
                    'face_retired_bil': cbo_buyback_face_retired,
                    'tips_principal_indexation_bil': tips_inflation_accretion_period,
                    'controlled_debt_post_issuance_bil': cbo_post_issuance_controlled_debt,
                    'target_error_bil': cbo_target_error,
                    'intragovernmental_excluded_bil': intragov_excluded,
                    'fed_included_bil': fed_included,
                    'funding_mode': funding_rule_mode,
                    'intragovernmental_treatment': 'excluded_from_public_debt',
                    'fed_held_treasury_treatment': 'included_in_public_debt',
                    'public_nonmarketable_treatment': 'explicit_bridge_not_marketable_issuance',
                }
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
            cbo_fed_maturities = cb_stock_matured
            # Stock added at auction: adjusted principal for TIPS, as in the Fed bridge.
            cbo_fed_auction_addons = float(total_issued_stock_by_holder.get('CB', 0.0))
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
        new_issuance_metrics = _handoff_weighted_metrics(new_bonds_added_list)
        outstanding_metrics = _handoff_outstanding_metrics(active_bonds_final, current_date)
        if closing_holder_stocks is None:
            closing_holder_stocks = holder_targets.closing_stocks(active_bonds_final)
        for column, value in holder_targets.stock_result_values(closing_holder_stocks).items():
            results.loc[current_date, column] = value
        results.loc[current_date, 'NewIssuanceWAM'] = new_issuance_metrics['wam_years']
        results.loc[current_date, 'NewIssuanceBillShare'] = new_issuance_metrics['bill_share']
        results.loc[current_date, 'NewIssuanceShortMaturityShare'] = new_issuance_metrics['short_maturity_share']
        results.loc[current_date, 'OutstandingControlledWAM'] = outstanding_metrics['wam_years']
        results.loc[current_date, 'OutstandingControlledBillShare'] = outstanding_metrics['bill_share']
        results.loc[current_date, 'OutstandingControlledShortMaturityShare'] = outstanding_metrics['short_maturity_share']
        if bounded_handoff:
            handoff_tables.capture_portfolio_snapshot(
                bond_portfolio,
                snapshot_date=current_date,
            )
        else:
            _handoff_append_holder_stocks(
                handoff_tables,
                active_bonds_final,
                current_date,
            )
        handoff_tables['tdcsim_scenario_metrics'].append(
            {
                'date': str(pd.Timestamp(current_date).date()),
                'new_issuance_wam_years': new_issuance_metrics['wam_years'],
                'outstanding_controlled_wam_years': outstanding_metrics['wam_years'],
                'new_issuance_bill_share': new_issuance_metrics['bill_share'],
                'outstanding_controlled_bill_share': outstanding_metrics['bill_share'],
                'new_issuance_short_maturity_share': new_issuance_metrics['short_maturity_share'],
                'outstanding_controlled_short_maturity_share': outstanding_metrics['short_maturity_share'],
                'short_maturity_cutoff_years': 1.0,
            }
        )
        other_reserve_change = -reserve_transfer
        other_deposit_change = 0.0
        other_tga_change = reserve_transfer + money_minting_transfers
        _handoff_append_journal(
            handoff_tables,
            prev_date,
            current_date,
            event_type='transfer',
            leg_type='reserve_transfer_and_money_minting',
            accounting_basis='treasury_cash_reserve_deposit',
            holder_sector='Treasury',
            counterparty_sector='other_named_transfer_source',
            instrument_type='cash',
            maturity_bucket='not_applicable',
            treasury_cash_change=other_tga_change,
            reserve_change=other_reserve_change,
            deposit_change=other_deposit_change,
            settlement_scope='configured_other_transfer',
        )
        reserve_change_period += other_reserve_change
        if trc_ledger is not None:
            trc_ledger.record_public_cash('OtherTreasuryFed', other_reserve_change, 'bank_reserve_settled')
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
        if cbo_reference_financing_mode or cbo_operating_cash_financing_mode:
            validation_floor = float(
                funding_rule_cfg.get('validation_floor_bil', -0.000001)
            )
            if current_tga < validation_floor:
                raise RuntimeError(
                    f"TGA-floor financing failed at {current_date.date()}: "
                    f"TGA={current_tga:.12f}, floor={validation_floor:.12f}"
                )
        if cbo_operating_cash_financing_mode:
            if close_books_today and abs(current_tga - cbo_operating_cash_target) > target_tolerance:
                raise RuntimeError(
                    f'Fiscal close did not meet the operating-cash target at {current_date.date()}: '
                    f'TGA={current_tga:.12f}, target={cbo_operating_cash_target:.12f}'
                )
            # The TGA plus what is decided but not yet settled sits on the target on every
            # day with a need, and never below it; a cash flow booked after issuance would
            # break this.
            shortfall = cbo_operating_cash_target - (
                current_tga + results.loc[current_date, 'PendingIssuanceProceeds']
                - results.loc[current_date, 'FedRolloverCashSizingDifference']
            )
            off_target = abs(shortfall) if total_issuance_target_period > target_tolerance else max(0.0, shortfall)
            if off_target > target_tolerance:
                raise RuntimeError(
                    f"Operating-cash financing left the TGA off target at {current_date.date()}: "
                    f"TGA={current_tga:.12f}, target={cbo_operating_cash_target:.12f}"
                )
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
        treasury_reserve_effect = adjusted_reserve_change_period
        if trc_ledger is not None:
            trc_rows.append(
                trc_ledger.close_period(
                    current_date,
                    legacy_reserve_change=adjusted_reserve_change_period,
                    legacy_reserves=current_reserves,
                )
            )
            treasury_reserve_effect = trc_rows[-1]['TRC_Change']
        if cbo_funding_mode:
            fed_reserves += treasury_reserve_effect + non_treasury_reserve_effect(
                mbs_start=float(fed_row['soma_mbs_start_bil']),
                mbs_end=float(fed_row['soma_mbs_end_bil']),
                currency_start=float(fed_row['currency_start_bil']),
                currency_end=float(fed_row['currency_end_bil']),
                mbs_income=fed_lines['CB_MBSIncome'],
                other_income=fed_lines['CB_OtherIncome'],
                iorb_expense=fed_iorb_expense,
                on_rrp_expense=fed_on_rrp_expense,
                operating_expenses=fed_lines['CB_OperatingExpenses'],
                dividends=fed_lines['CB_Dividends'],
            )
            if fed_reserves < 0.0:
                raise FedIncomeError(
                    f'reserve balances fell below zero at {current_date.date()}: {fed_reserves!r}'
                )
            fed_on_rrp = _fed_on_rrp_balance(trc_ledger, fed_opening)
            results.loc[current_date, 'CB_TreasuryReserveEffect'] = treasury_reserve_effect
            results.loc[current_date, 'CB_ReserveBalances'] = fed_reserves
            results.loc[current_date, 'CB_ONRRPBalances'] = fed_on_rrp
            results.loc[current_date, 'CB_Currency'] = float(fed_row['currency_end_bil'])
            results.loc[current_date, 'CB_MBSHoldings'] = float(fed_row['soma_mbs_end_bil'])
        results.loc[current_date, 'TDC_Change'] = adjusted_deposit_change_period
        results.loc[current_date, 'TGAChange'] = current_tga - prev_tga
        results.loc[current_date, 'TDC_FiscalFlow'] = fiscal_deposit_change
        results.loc[current_date, 'TDC_DebtService'] = debt_service_deposit_change
        results.loc[current_date, 'TDC_AuctionAbsorption'] = issuance_deposit_change
        results.loc[current_date, 'TDC_SecondaryTrades'] = (
            pref_trade_monetary_impact.get('deposit_change', 0.0)
            + fed_secondary_deposit_effect
            + holder_target_deposit_effect
        )
        results.loc[current_date, 'TDC_Other'] = other_deposit_change
        private_secondary_change = results.loc[current_date, 'TDC_SecondaryTrades']
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
        added_principal = principal_component_paid_by_private_route[PRIVATE_SUBBUCKET_MMF_ADDED]
        added_principal_tdc = mmf_added_deposit_share * added_principal
        results.loc[current_date, MMF_ADDED_TDC_COLUMNS['TDC_PrincipalToDU']] = added_principal_tdc
        results.loc[current_date, 'TDC_PrincipalToDU_MMF'] = (
            added_principal_tdc
            + mmf_deposit_pass_through * _private_route_value(
                principal_component_paid_by_private_route, PRIVATE_SUBBUCKET_MMF,
            )
        )
        results.loc[current_date, 'TDC_PrincipalToDU_MMFPlumbing'] = (
            (1.0 - mmf_deposit_pass_through)
            * _private_route_value(principal_component_paid_by_private_route, PRIVATE_SUBBUCKET_MMF)
            + (1.0 - mmf_added_deposit_share) * added_principal
        )
        results.loc[current_date, 'TDC_BillDiscountInterestToDU_DomesticNonbank'] = _private_route_value(
            bill_discount_interest_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        added_discount = bill_discount_interest_paid_by_private_route[PRIVATE_SUBBUCKET_MMF_ADDED]
        added_discount_tdc = mmf_added_deposit_share * added_discount
        results.loc[current_date, MMF_ADDED_TDC_COLUMNS['TDC_BillDiscountInterestToDU']] = added_discount_tdc
        results.loc[current_date, 'TDC_BillDiscountInterestToDU_MMF'] = (
            added_discount_tdc
            + mmf_deposit_pass_through * _private_route_value(bill_discount_interest_paid_by_private_route, PRIVATE_SUBBUCKET_MMF)
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
            + (1.0 - mmf_added_deposit_share) * (
                principal_paid_by_private_route[PRIVATE_SUBBUCKET_MMF_ADDED]
                + interest_paid_by_private_route[PRIVATE_SUBBUCKET_MMF_ADDED]
            )
        )
        results.loc[current_date, 'TDC_PrincipalToDU'] = (
            results.loc[current_date, 'TDC_PrincipalToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_PrincipalToDU_MMF']
        )
        results.loc[current_date, 'TDC_PrincipalCashToDU_DomesticNonbank'] = _private_route_value(
            principal_paid_by_private_route,
            PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
        )
        added_principal_cash = principal_paid_by_private_route[PRIVATE_SUBBUCKET_MMF_ADDED]
        results.loc[current_date, 'TDC_PrincipalCashToDU_MMF'] = (
            mmf_added_deposit_share * added_principal_cash
            + mmf_deposit_pass_through * _private_route_value(principal_paid_by_private_route, PRIVATE_SUBBUCKET_MMF)
        )
        results.loc[current_date, 'TDC_PrincipalCashToDU_MMFPlumbing'] = (
            (1.0 - mmf_deposit_pass_through)
            * _private_route_value(principal_paid_by_private_route, PRIVATE_SUBBUCKET_MMF)
            + (1.0 - mmf_added_deposit_share) * added_principal_cash
        )
        results.loc[current_date, 'TDC_PrincipalCashToDU'] = (
            results.loc[current_date, 'TDC_PrincipalCashToDU_DomesticNonbank']
            + results.loc[current_date, 'TDC_PrincipalCashToDU_MMF']
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
        results.loc[current_date, 'TDC_NetPrincipalIssuanceCashflowToDU'] = (
            results.loc[current_date, 'TDC_PrincipalCashToDU']
            - results.loc[current_date, 'TDC_GrossIssuanceProceedsAbsorbedByDU']
        )
        results.loc[current_date, 'TDC_SecondaryTrades_DomesticNonbank'] = (
            pref_trade_monetary_impact.get('deposit_change_private_deposit_funded', 0.0)
            + fed_secondary_private_route_effects[PRIVATE_SUBBUCKET_DOMESTIC_NONBANK]
            + holder_target_private_route_effects[PRIVATE_SUBBUCKET_DOMESTIC_NONBANK]
        )
        fed_secondary_mmf_gross = (
            fed_secondary_private_route_effects[PRIVATE_SUBBUCKET_MMF]
            + holder_target_private_route_effects[PRIVATE_SUBBUCKET_MMF]
        )
        added_secondary_cash = (
            fed_secondary_private_route_effects[PRIVATE_SUBBUCKET_MMF_ADDED]
            + holder_target_private_route_effects[PRIVATE_SUBBUCKET_MMF_ADDED]
        )
        added_secondary_tdc = mmf_added_deposit_share * added_secondary_cash
        results.loc[current_date, MMF_ADDED_TDC_COLUMNS['TDC_SecondaryTrades']] = added_secondary_tdc
        results.loc[current_date, 'TDC_SecondaryTrades_MMF'] = (
            added_secondary_tdc + mmf_deposit_pass_through
            * (
                pref_trade_monetary_impact.get('deposit_change_private_mmf', 0.0)
                + fed_secondary_mmf_gross
            )
        )
        results.loc[current_date, 'TDC_SecondaryTrades_MMFPlumbing'] = (
            pref_trade_monetary_impact.get('deposit_change_private_mmf_plumbing', 0.0)
            + (1.0 - mmf_deposit_pass_through) * fed_secondary_mmf_gross
            + (1.0 - mmf_added_deposit_share) * added_secondary_cash
        )
        results.loc[current_date, 'TDC_SecondaryDUToRU'] = max(0.0, private_secondary_change)
        results.loc[current_date, 'TDC_SecondaryRUToDU'] = max(0.0, -private_secondary_change)
        results.loc[current_date, 'AuctionDemandShift_AvgAbs'] = (
            auction_shift_weighted_sum / auction_shift_weight_total if auction_shift_weight_total > TGA_FLOOR_TOLERANCE else 0.0
        )
        results.loc[current_date, 'AuctionDemandShift_MaxAbs'] = auction_shift_weighted_max
        results.loc[current_date, 'SecondaryDemandShift_AvgAbs'] = secondary_shift_avg
        results.loc[current_date, 'SecondaryDemandShift_MaxAbs'] = secondary_shift_max
        if bounded_handoff:
            handoff_tables.close_period(
                prev_date=prev_date,
                current_date=current_date,
                closing_result=results.loc[current_date].to_dict(),
            )
    sim_duration = time.time() - sim_start_time
    print(f'--- Finished Simulation: {scenario_name} ({sim_duration:.2f} seconds) ---')
    final_portfolio_out = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
    if not bond_portfolio.empty:
        try:
            final_portfolio_out = bond_portfolio[BOND_PORTFOLIO_COLS].astype(PORTFOLIO_DTYPES, errors='ignore')
        except Exception:
            final_portfolio_out = bond_portfolio
    if trc_ledger is not None:
        trc_frame = pd.DataFrame(trc_rows, columns=list(TRC_RESULT_COLUMNS))
        trc_frame['Date'] = pd.to_datetime(trc_frame['Date']).dt.strftime('%Y-%m-%d')
        results.attrs['trc_results'] = trc_frame
        results.attrs['trc_metadata'] = trc_metadata(trc_ledger.policy, trc_ledger)
    if bounded_handoff:
        results.attrs['bounded_handoff_summary'] = handoff_tables.finalize()
    else:
        results.attrs['handoff_tables'] = handoff_tables
    results.attrs['mmf_acquisition_maturities'] = mmf_acquisition_maturities
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
        'ratewall_holder_absorption_resolution_status': holder_absorption_resolution_status,
        'ratewall_holder_absorption_source_scenario_id': holder_absorption_source_scenario_id,
        'ratewall_holder_absorption_loaded_quarters': (
            len(holder_absorption_lookup.get(holder_absorption_source_scenario_id, {}))
            if holder_absorption_source_scenario_id
            else 0
        ),
        'funding_rule_mode': funding_rule_mode if cbo_funding_mode else str(funding_rule_cfg.get('mode', 'legacy_cash_tga')),
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
