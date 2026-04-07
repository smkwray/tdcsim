
"""Core simulation loop for the Treasury funding-chain simulator."""

import copy
import time
import traceback
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    DAYS_PER_YEAR_ACTUAL,
    FRN_DAY_COUNT_BASIS,
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    MATURITY_CATEGORIES,
    PORTFOLIO_DTYPES,
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
)
from sim_trading import execute_preference_trades
from tdc_validation import is_coarse_frequency


def _get_rate_sensitive_multipliers_config(rate_sensitive_params):
    if not isinstance(rate_sensitive_params, dict):
        return {}
    return rate_sensitive_params


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


def _get_category_yield(category, issuance_profile, yield_curve_years, yield_curve_rates, frn_benchmark_mat):
    maturity = _get_weighted_average_maturity(category, issuance_profile, frn_benchmark_mat)
    yld = get_yield_for_maturity(maturity, yield_curve_years, yield_curve_rates)
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


def _get_curve_reference_levels(rate_sensitive_params, yield_curve_years, yield_curve_rates):
    params = _get_rate_sensitive_multipliers_config(rate_sensitive_params)
    anchor_mat = max(0.0, float(params.get('anchor_maturity_years', 5.0) or 5.0))
    slope_short = max(0.0, float(params.get('slope_short_maturity_years', 2.0) or 2.0))
    slope_long = max(0.0, float(params.get('slope_long_maturity_years', 10.0) or 10.0))
    anchor_yield = get_yield_for_maturity(anchor_mat, yield_curve_years, yield_curve_rates)
    short_yield = get_yield_for_maturity(slope_short, yield_curve_years, yield_curve_rates)
    long_yield = get_yield_for_maturity(slope_long, yield_curve_years, yield_curve_rates)
    anchor_yield = 0.0 if pd.isna(anchor_yield) else float(anchor_yield)
    short_yield = 0.0 if pd.isna(short_yield) else float(short_yield)
    long_yield = 0.0 if pd.isna(long_yield) else float(long_yield)
    return anchor_yield, long_yield - short_yield


def _build_dynamic_secondary_preferences(base_secondary_prefs, rate_sensitive_params, issuance_profile, yield_curve_years, yield_curve_rates, frn_benchmark_mat):
    if not isinstance(rate_sensitive_params, dict) or not rate_sensitive_params.get('enabled', False):
        return copy.deepcopy(base_secondary_prefs)
    if not isinstance(rate_sensitive_params.get('secondary', {}), dict) or not rate_sensitive_params.get('secondary'):
        return copy.deepcopy(base_secondary_prefs)

    active_categories = ['bills', 'notes', 'bonds']
    if float(issuance_profile.get('TIPS', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('tips')
    if float(issuance_profile.get('FRN', {}).get('target_percentage', 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
        active_categories.append('frn')

    anchor_yield, curve_slope = _get_curve_reference_levels(rate_sensitive_params, yield_curve_years, yield_curve_rates)
    dynamic_prefs = copy.deepcopy(base_secondary_prefs)
    for holder in HOLDER_TYPES:
        holder_cfg = copy.deepcopy(dynamic_prefs.get(holder, {}))
        adjusted = {}
        for category in active_categories:
            base_share = float(holder_cfg.get(f'{category}_pct', 0.0) or 0.0)
            category_yield = _get_category_yield(category, issuance_profile, yield_curve_years, yield_curve_rates, frn_benchmark_mat)
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

def run_simulation(params, start_date, end_date, freq='W', scenario_name='Default'):
    """
    Runs the core economic simulation including Treasury operations, fiscal flows,
    central bank actions, secondary market trading, AND handles date-based events
    defined in the configuration to dynamically change parameters.
    Tracks key monetary aggregates and debt composition.
    """
    sim_start_time = time.time()
    validate_run_params(params, scenario_name=scenario_name)
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
        if freq.upper().startswith('W'):
            start_date_pd = start_date_pd - timedelta(days=start_date_pd.weekday())
        dates = pd.date_range(start_date_pd, end_date_pd, freq=freq, name='Date')
        num_periods = len(dates)
    except Exception as e:
        print(f'ERROR [{scenario_name}]: Failed to create date range: {e}')
        return (pd.DataFrame(), pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES))
    if num_periods <= 1:
        print(f'WARNING [{scenario_name}]: Simulation period results in <= 1 time step. Skipping.')
        results_cols_on_skip = ['TGA', 'Reserves', 'TDC_Level']
        empty_results = pd.DataFrame(index=dates, columns=results_cols_on_skip, dtype=float).fillna(0.0)
        return (empty_results, pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES))
    print(f'--- Starting Simulation: {scenario_name} ---')
    results_cols = ['GovSpending', 'Taxes', 'PrimaryDeficit', 'InterestPaid_Bonds', 'PrincipalPaid_Bonds', 'InterestOutlay_Period', 'InterestOutlay_Cumulative', 'PrincipalRollover_Period', 'PrincipalRollover_Cumulative', 'NewDebtIssued', 'AuctionProceeds', 'IssuanceProceedsTarget', 'IssueDiscountCost_Period', 'IssueDiscountCost_Cumulative', 'FinancingCost_Period', 'FinancingCost_Cumulative', 'NonMarketableInterestCapitalized_Period', 'NonMarketableInterestCapitalized_Cumulative', 'TIPSInflationAccretion_Period', 'TIPSInflationAccretion_Cumulative', 'AuctionDemandShift_AvgAbs', 'AuctionDemandShift_MaxAbs', 'SecondaryDemandShift_AvgAbs', 'SecondaryDemandShift_MaxAbs', 'DebtServiceOutlay_Period', 'DebtServiceOutlay_Cumulative', 'TotalDebt_Agg', 'DebtHeld_Banks', 'DebtHeld_Private', 'DebtHeld_CB', 'DebtHeld_Foreign', 'DebtHeld_FedInternal', 'DebtHeld_TrustFunds', 'TGA', 'Reserves', 'TDC_Level', 'ReserveChange', 'TDC_Change', 'TGAChange', 'TDC_FiscalFlow', 'TDC_DebtService', 'TDC_AuctionAbsorption', 'TDC_SecondaryTrades', 'TDC_Other', 'CB_InterestIncome', 'CB_NetIncome', 'CB_Remittance', 'CB_DeferredAsset', 'WAM', 'DebtHeldByType_Fixed', 'DebtHeldByType_TIPS', 'DebtHeldByType_FRN', 'DebtHeldByType_NonMarketable', 'CPI_Level', 'Reference_CPI']
    results = pd.DataFrame(index=dates, columns=results_cols, dtype=float).fillna(0.0)
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
        fiscal_p = params.get('fiscal_params', {})
        current_fiscal_params = copy.deepcopy(fiscal_p)
        q_start_spending = current_fiscal_params.get('initial_weekly_spending', 0.0)
        q_start_taxes = current_fiscal_params.get('initial_weekly_taxes', 0.0)
        spending_growth_qtr = current_fiscal_params.get('spending_growth_qtr', 0.0)
        tax_growth_qtr = current_fiscal_params.get('tax_growth_qtr', 0.0)
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
        dynamic_params_state = {'yield_curve': yield_p, 'fiscal_params': current_fiscal_params, 'tga_params': current_tga_params, 'other_flows': current_other_flows, 'sector_preferences': current_sector_prefs, 'auction_absorption_preferences': current_auction_prefs, 'secondary_target_preferences': current_secondary_prefs, 'treasury_issuance_profile': issuance_profile, 'tips_params': tips_p, 'frn_params': frn_p, 'nonmarketable_params': current_nonmkt_params, 'rate_sensitive_demand': rate_sensitive_p, 'financing_cost_options': financing_cost_options, 'simulation_period': {'enable_preference_trading': current_enable_trading}}
    except Exception as e:
        print(f'ERROR [{scenario_name}]: Failed during parameter extraction/initialization: {e}')
        traceback.print_exc()
        return (pd.DataFrame(), pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES))
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
        tips_real_coupon = tips_p.get('default_real_coupon_rate', tips_real_coupon)
        frn_benchmark_mat = frn_p.get('benchmark_maturity_years', frn_benchmark_mat)
        frn_spread = frn_p.get('default_fixed_spread', frn_spread)
        nonmkt_rate_mats = current_nonmkt_params.get('interest_rate_basis_maturities', nonmkt_rate_mats)
        nonmkt_credit_freq = current_nonmkt_params.get('interest_crediting_frequency', nonmkt_credit_freq)
        include_tips_inflation_accretion = bool(financing_cost_options.get('include_tips_inflation_accretion', include_tips_inflation_accretion))
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
        except Exception as e:
            bond_portfolio = pd.DataFrame(columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES)
        if not bond_portfolio.empty:
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
    results.loc[t0, 'Reference_CPI'] = cpi_start
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
    results.loc[t0, ['CB_InterestIncome', 'CB_NetIncome', 'CB_Remittance']] = 0.0
    max_existing_id = 0
    if not bond_portfolio.empty and bond_portfolio['BondID'].notna().any():
        numeric_bond_ids = pd.to_numeric(bond_portfolio['BondID'], errors='coerce')
        if numeric_bond_ids.notna().any():
            max_existing_id = numeric_bond_ids.max()
    bond_id_counter = int(max_existing_id) + 1
    last_quarter_start_date = t0
    weeks_in_quarter = 0
    last_cpi_update_date = t0
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
        reserve_change_period = 0.0
        deposit_change_period = 0.0
        tga_change_period = 0.0
        reserve_change_secondary_mkt = 0.0
        deposit_change_secondary_mkt = 0.0
        tga_inflow_secondary_mkt = 0.0
        tga_drain_secondary_mkt = 0.0
        pref_trade_monetary_impact = {'reserve_change': 0.0, 'deposit_change': 0.0, 'tga_change': 0.0, 'tga_drain': 0.0}
        nonmkt_interest_capitalized_period = 0.0
        tips_inflation_accretion_period = 0.0
        auction_shift_weighted_sum = 0.0
        auction_shift_weighted_max = 0.0
        auction_shift_weight_total = 0.0
        secondary_shift_avg = 0.0
        secondary_shift_max = 0.0
        prev_cpi = results.loc[prev_date, 'CPI_Level']
        current_cpi = prev_cpi * (1 + cpi_inflation) ** delta_t_years
        results.loc[current_date, 'CPI_Level'] = current_cpi
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
            index_ratio = np.maximum(index_ratio, 1.0)
            bond_portfolio.loc[tips_mask, 'IndexRatio'] = index_ratio
            original_principal = bond_portfolio.loc[tips_mask, 'OriginalPrincipal']
            new_adjusted_principal = original_principal * index_ratio
            bond_portfolio.loc[tips_mask, 'AdjustedPrincipal'] = new_adjusted_principal
            if include_tips_inflation_accretion:
                tips_inflation_accretion_period = (new_adjusted_principal - prev_adjusted_principal).clip(lower=0.0).sum()
        frn_mask = (bond_portfolio['Status'] == 'Active') & (bond_portfolio['SecurityType'] == 'FRN')
        if frn_mask.any():
            frn_yield = get_yield_for_maturity(frn_benchmark_mat, current_yield_curve_years, current_yield_curve_rates)
            frn_yield = 0.0 if pd.isna(frn_yield) else frn_yield
            bond_portfolio.loc[frn_mask, 'BenchmarkRate_FRN'] = frn_yield
            daily_rate = (bond_portfolio.loc[frn_mask, 'BenchmarkRate_FRN'] + bond_portfolio.loc[frn_mask, 'FixedSpread']) / FRN_DAY_COUNT_BASIS
            daily_accrual = bond_portfolio.loc[frn_mask, 'FaceValue'] * daily_rate.clip(lower=0)
            period_accrual = daily_accrual * delta_t_days
            bond_portfolio.loc[frn_mask, 'AccruedInterest_FRN'] = bond_portfolio.loc[frn_mask, 'AccruedInterest_FRN'].fillna(0.0) + period_accrual
            bond_portfolio.loc[frn_mask, 'LastAccrualDate'] = current_date
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
        results.loc[current_date, 'GovSpending'] = gov_spending_period
        results.loc[current_date, 'Taxes'] = taxes_period
        results.loc[current_date, 'PrimaryDeficit'] = gov_spending_period - taxes_period
        fiscal_reserve_change = gov_spending_period - taxes_period
        fiscal_deposit_change = gov_spending_period - taxes_period
        fiscal_tga_change = taxes_period - gov_spending_period
        reserve_change_period += fiscal_reserve_change
        deposit_change_period += fiscal_deposit_change
        tga_change_period += fiscal_tga_change
        principal_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        interest_paid_by_holder = {h: 0.0 for h in HOLDER_TYPES}
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
                holder = bond['HolderType']
                face_value = bond['FaceValue']
                coupon_rate = bond['CouponRate']
                issue_date = bond['IssueDate']
                maturity_date = bond['MaturityDate']
                security_type = bond['SecurityType']
                if security_type == 'TIPS':
                    principal_payment = max(bond.get('OriginalPrincipal', 0.0), bond.get('AdjustedPrincipal', 0.0))
                elif security_type in ['Fixed', 'FRN', 'NonMarketable']:
                    principal_payment = face_value
                if security_type == 'FRN':
                    interest_payment = bond.get('AccruedInterest_FRN', 0.0)
                    bond_portfolio.loc[idx, 'AccruedInterest_FRN'] = 0.0
                elif security_type in ['Fixed', 'TIPS'] and coupon_rate > TGA_FLOOR_TOLERANCE:
                    coupon_freq = 2
                    adj_p = bond.get('AdjustedPrincipal') if security_type == 'TIPS' else None
                    principal_base = adj_p if security_type == 'TIPS' and adj_p is not None else face_value
                    interest_payment = principal_base * coupon_rate / coupon_freq
                principal_paid_by_holder[holder] += principal_payment
                interest_paid_by_holder[holder] += interest_payment
                total_principal_paid_period += principal_payment
                total_interest_paid_period += interest_payment
        active_mask = bond_portfolio['Status'] == 'Active'
        frn_active_mask = active_mask & (bond_portfolio['SecurityType'] == 'FRN')
        if frn_active_mask.any():
            frn_paying_indices = []
            frn_tranches = bond_portfolio.loc[frn_active_mask]
            for idx, bond in frn_tranches.iterrows():
                issue_dt = bond['IssueDate']
                maturity_dt = bond['MaturityDate']
                coupon_dates = get_coupon_dates_in_period(issue_dt, maturity_dt, prev_date, current_date, frequency=4)
                if coupon_dates:
                    frn_paying_indices.append(idx)
            if frn_paying_indices:
                frn_paying_indices = list(set(frn_paying_indices))
                payment_amount = bond_portfolio.loc[frn_paying_indices, 'AccruedInterest_FRN'].fillna(0.0)
                pmt_by_holder = payment_amount.groupby(bond_portfolio.loc[frn_paying_indices, 'HolderType']).sum().to_dict()
                for h, pmt in pmt_by_holder.items():
                    interest_paid_by_holder[h] += pmt
                bond_portfolio.loc[frn_paying_indices, 'AccruedInterest_FRN'] = 0.0
                total_interest_paid_period += payment_amount.sum()
        fixed_tips_mask = active_mask & bond_portfolio['SecurityType'].isin(['Fixed', 'TIPS']) & (bond_portfolio['CouponRate'] > TGA_FLOOR_TOLERANCE)
        if fixed_tips_mask.any():
            fixed_tips_paying_indices = []
            fixed_tips_tranches = bond_portfolio.loc[fixed_tips_mask]
            for idx, bond in fixed_tips_tranches.iterrows():
                issue_dt = bond['IssueDate']
                maturity_dt = bond['MaturityDate']
                coupon_dates = get_coupon_dates_in_period(issue_dt, maturity_dt, prev_date, current_date, frequency=2)
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
                            bond_yields = eligible_bonds['CurrentTTM'].apply(lambda ttm: get_yield_for_maturity(ttm, current_yield_curve_years, current_yield_curve_rates))
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
                        yld = get_yield_for_maturity(mat_yr, current_yield_curve_years, current_yield_curve_rates)
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
        reserve_change_period += debt_service_reserve_change
        deposit_change_period += debt_service_deposit_change
        tga_change_period += debt_service_tga_change
        prev_deferred_asset = results.loc[prev_date, 'CB_DeferredAsset']
        cb_interest_income_period = interest_paid_by_holder.get('CB', 0.0)
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
        results.loc[current_date, 'CB_InterestIncome'] = cb_interest_income_period
        results.loc[current_date, 'CB_NetIncome'] = cb_net_income_period
        results.loc[current_date, 'CB_Remittance'] = cb_remittance_period
        results.loc[current_date, 'CB_DeferredAsset'] = current_deferred_asset
        cb_remit_tga_change = cb_remittance_period
        tga_change_period += cb_remit_tga_change
        prev_tga = results.loc[prev_date, 'TGA']
        tga_change_before_issuance = fiscal_tga_change + debt_service_tga_change + (tga_inflow_secondary_mkt - tga_drain_secondary_mkt) + cb_remit_tga_change + reserve_transfer + money_minting_transfers
        projected_tga_pre_issuance = prev_tga + tga_change_before_issuance
        funding_needed = max(0, tga_target - projected_tga_pre_issuance)
        min_issuance_for_floor = max(0, tga_floor - projected_tga_pre_issuance)
        total_issuance_target_period = max(funding_needed, min_issuance_for_floor)
        new_bonds_added_list = []
        total_issued_face_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        total_issued_proceeds_by_holder = {h: 0.0 for h in HOLDER_TYPES}
        actual_issued_amount = 0.0
        actual_auction_proceeds = 0.0
        issue_discount_cost_period = 0.0
        results.loc[current_date, 'IssuanceProceedsTarget'] = total_issuance_target_period
        if total_issuance_target_period > TGA_FLOOR_TOLERANCE:
            tips_pct = issuance_profile.get('TIPS', {}).get('target_percentage', 0.0)
            frn_pct = issuance_profile.get('FRN', {}).get('target_percentage', 0.0)
            nonmkt_pct_profile = issuance_profile.get('NonMarketable', {}).get('target_percentage', 0.0)
            issued_tips = total_issuance_target_period * tips_pct
            issued_frn = total_issuance_target_period * frn_pct
            issued_nonmkt = total_issuance_target_period * nonmkt_pct_profile
            marketable_fixed_rate_issuance = max(0, total_issuance_target_period - issued_tips - issued_frn - issued_nonmkt)
            issuance_supply_schedule = []
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
                            yld = get_yield_for_maturity(maturity_yrs, current_yield_curve_years, current_yield_curve_rates)
                            real_coupon = calculate_coupon_rate('TIPS', maturity_yrs, yld, tips_real_coupon)
                            face_amount, proceeds_amount, issue_price_ratio = calculate_face_from_proceeds_target('TIPS', maturity_yrs, real_coupon, yld, proceeds_target)
                            issuance_supply_schedule.append({'type': 'TIPS', 'maturity': maturity_yrs, 'face_amount': face_amount, 'proceeds': proceeds_amount, 'coupon': real_coupon, 'issue_price_ratio': issue_price_ratio, 'issue_yield': yld})
            if issued_frn > TGA_FLOOR_TOLERANCE:
                frn_face, frn_proceeds, frn_issue_price_ratio = calculate_face_from_proceeds_target('FRN', 2.0, 0.0, np.nan, issued_frn)
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
                    yld = get_yield_for_maturity(maturity_yrs, current_yield_curve_years, current_yield_curve_rates)
                    coupon = calculate_coupon_rate('Fixed', maturity_yrs, yld, 0)
                    face_amount, proceeds_amount, issue_price_ratio = calculate_face_from_proceeds_target('Fixed', maturity_yrs, coupon, yld, proceeds_target)
                    issuance_supply_schedule.append({'type': 'Fixed', 'maturity': maturity_yrs, 'face_amount': face_amount, 'proceeds': proceeds_amount, 'coupon': coupon, 'issue_price_ratio': issue_price_ratio, 'issue_yield': yld})
            if issued_nonmkt > TGA_FLOOR_TOLERANCE:
                nm_issuance_profile_details = issuance_profile.get('NonMarketable', {})
                nm_maturity_years = nm_issuance_profile_details.get('nominal_maturity_years', 30.0)
                nm_face, nm_proceeds, nm_issue_price_ratio = calculate_face_from_proceeds_target('NonMarketable', nm_maturity_years, 0.0, np.nan, issued_nonmkt)
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
                    item_yield = get_yield_for_maturity(item_maturity, current_yield_curve_years, current_yield_curve_rates)
                item_yield = 0.0 if pd.isna(item_yield) else float(item_yield)
                anchor_yield, curve_slope = _get_curve_reference_levels(rate_sensitive_p, current_yield_curve_years, current_yield_curve_rates)
                holder_share_weights = {}
                for holder in HOLDER_TYPES:
                    holder_prefs = current_auction_prefs.get(holder, {})
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
                    maturity_date = current_date + relativedelta(years=int(round(mat_yrs)), months=int(round(mat_yrs % 1 * 12)))
                    coupon_rate_val = details.get('coupon', 0.0)
                    fixed_spread_val = details.get('spread', 0.0) if sec_type == 'FRN' else 0.0
                    issue_price_ratio_val = details.get('issue_price_ratio', 1.0)
                    issue_yield_val = details.get('issue_yield', np.nan)
                    issue_proceeds_val = face_value_issued_to_holder * issue_price_ratio_val
                    original_principal_val = 0.0
                    reference_cpi_issue_val = 0.0
                    index_ratio_init_val = 0.0
                    last_accrual_date_val = pd.NaT
                    if sec_type == 'TIPS':
                        original_principal_val = face_value_issued_to_holder
                        reference_cpi_issue_val = results.loc[current_date, 'Reference_CPI']
                        index_ratio_init_val = 1.0
                    elif sec_type == 'FRN':
                        last_accrual_date_val = current_date
                    maturity_category_val = None
                    if sec_type == 'Fixed':
                        maturity_category_val = get_maturity_category(mat_yrs, issuance_profile)
                    new_tranche = {'BondID': next_bond_id_to_assign, 'FaceValue': face_value_issued_to_holder, 'HolderType': holder, 'SecurityType': sec_type, 'IssueDate': current_date, 'MaturityDate': maturity_date, 'OriginalMaturityYears': mat_yrs, 'CouponRate': coupon_rate_val, 'Status': 'Active', 'MaturityCategory': maturity_category_val, 'OriginalPrincipal': original_principal_val, 'AdjustedPrincipal': original_principal_val, 'ReferenceCPI_Issue': reference_cpi_issue_val, 'IndexRatio': index_ratio_init_val, 'FixedSpread': fixed_spread_val, 'AccruedInterest_FRN': 0.0, 'BenchmarkRate_FRN': 0.0, 'LastAccrualDate': last_accrual_date_val, 'IssuePriceRatio': issue_price_ratio_val, 'IssueProceeds': issue_proceeds_val, 'IssueYieldAtIssue': issue_yield_val, 'TimeToMaturity': np.nan, 'DiscountYield': np.nan, 'CleanPrice': np.nan, 'AccruedInterest': np.nan, 'DirtyValue': np.nan, 'DirtyPriceRatio': np.nan}
                    new_bonds_added_list.append(new_tranche)
                    total_issued_face_by_holder[holder] += face_value_issued_to_holder
                    total_issued_proceeds_by_holder[holder] += issue_proceeds_val
                    actual_issued_amount += face_value_issued_to_holder
                    actual_auction_proceeds += issue_proceeds_val
                    issue_discount_cost_period += max(0.0, face_value_issued_to_holder - issue_proceeds_val)
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
                        bond_portfolio = pd.concat([bond_portfolio.astype(PORTFOLIO_DTYPES, errors='ignore'), new_bonds_df], ignore_index=True)
                except Exception as concat_err:
                    print(f'ERROR [{scenario_name}@{current_date.date()}]: Concatenation failed: {concat_err}')
        results.loc[current_date, 'NewDebtIssued'] = actual_issued_amount
        results.loc[current_date, 'AuctionProceeds'] = actual_auction_proceeds
        results.loc[current_date, 'IssueDiscountCost_Period'] = issue_discount_cost_period
        results.loc[current_date, 'IssueDiscountCost_Cumulative'] = results.loc[prev_date, 'IssueDiscountCost_Cumulative'] + issue_discount_cost_period
        results.loc[current_date, 'FinancingCost_Period'] = results.loc[current_date, 'InterestOutlay_Period'] + issue_discount_cost_period + nonmkt_interest_capitalized_period + tips_inflation_accretion_period
        results.loc[current_date, 'FinancingCost_Cumulative'] = results.loc[prev_date, 'FinancingCost_Cumulative'] + results.loc[current_date, 'FinancingCost_Period']
        issuance_tga_change = actual_auction_proceeds
        issuance_reserve_change = -(total_issued_proceeds_by_holder.get('Banks', 0.0) + total_issued_proceeds_by_holder.get('Private', 0.0) + total_issued_proceeds_by_holder.get('Foreign', 0.0))
        issuance_deposit_change = -total_issued_proceeds_by_holder.get('Private', 0.0)
        reserve_change_period += issuance_reserve_change
        deposit_change_period += issuance_deposit_change
        tga_change_period += issuance_tga_change
        if current_enable_trading and (not bond_portfolio.empty):
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
                )
                secondary_shift_avg, secondary_shift_max = _compute_preference_shift_summary(
                    current_secondary_prefs,
                    effective_secondary_prefs,
                    _get_active_preference_categories(issuance_profile),
                )
                bond_portfolio_temp, pref_trade_monetary_impact = execute_preference_trades(bond_portfolio_temp, current_date, current_yield_curve_years, current_yield_curve_rates, effective_secondary_prefs, issuance_profile, scenario_name)
                bond_portfolio = bond_portfolio_temp
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
        for holder in HOLDER_TYPES:
            results.loc[current_date, f'DebtHeld_{holder}'] = active_bonds_final.loc[active_bonds_final['HolderType'] == holder, 'DebtBase'].sum() if not active_bonds_final.empty else 0.0
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
        current_tga = max(tga_floor, projected_tga)
        tga_discrepancy = current_tga - projected_tga
        adjusted_reserve_change_period = reserve_change_period + tga_discrepancy
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
        except Exception as e:
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
        'coarse_frequency_warning': coarse_frequency_warning,
        'auction_demand_shift_avgabs_mean': float(results['AuctionDemandShift_AvgAbs'].iloc[1:].mean()) if len(results.index) > 1 else 0.0,
        'secondary_demand_shift_avgabs_mean': float(results['SecondaryDemandShift_AvgAbs'].iloc[1:].mean()) if len(results.index) > 1 else 0.0,
    }
    results.rename(columns=OUTPUT_COLUMN_RENAMES, inplace=True)
    return (results, final_portfolio_out)


__all__ = ['run_simulation']
