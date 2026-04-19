"""Public API and CLI entry point for the Treasury funding-chain simulator."""

import concurrent.futures
import copy
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml

from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    HOLDER_TYPES,
    PORTFOLIO_DTYPES,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
)
from tdc_validation import validate_config

from sim_engine import run_simulation
from sim_groups import process_scenario_group
from sim_helpers import (
    OUTPUT_COLUMN_RENAMES,
    VALID_OVERRIDE_KEYS,
    _set_nested_value,
    apply_event_actions,
    update_dict_recursive,
    validate_run_params,
)
from sim_plotting import plot_multi_results
from sim_pricing import (
    _is_bill_like_fixed,
    calculate_accrued_interest,
    calculate_bond_market_price,
    calculate_coupon_rate,
    calculate_face_from_proceeds_target,
    calculate_issue_price_ratio,
    find_last_coupon_date,
    get_coupon_dates_in_period,
    get_maturity_category,
    get_payment_date,
    get_security_category_for_prefs,
    get_yield_for_maturity,
    infer_issue_data_for_loaded_bill,
)
from sim_trading import (
    calculate_portfolio_value_and_composition,
    execute_preference_trades,
)


def _load_base_config(config_file):
    try:
        with open(config_file, 'r') as f:
            base_config = yaml.safe_load(f)
        if base_config is None:
            raise ValueError('Config file is empty or invalid.')
        print('Base configuration loaded successfully.')
        return base_config
    except FileNotFoundError:
        print(f"FATAL ERROR: Config file '{config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL ERROR: Error parsing YAML config '{config_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: Unexpected error loading config '{config_file}': {e}")
        traceback.print_exc()
        sys.exit(1)


def _process_loaded_initial_portfolio(initial_bonds_df_global, base_config):
    if initial_bonds_df_global.empty:
        print('Initial bonds file was loaded but contained no data.')
        return initial_bonds_df_global

    print(f'Loaded initial portfolio ({len(initial_bonds_df_global)} rows). Processing...')

    for col in BOND_PORTFOLIO_COLS:
        if col not in initial_bonds_df_global.columns:
            if col in [
                'FaceValue',
                'CouponRate',
                'OriginalMaturityYears',
                'OriginalPrincipal',
                'AdjustedPrincipal',
                'ReferenceCPI_Issue',
                'IndexRatio',
                'FixedSpread',
                'AccruedInterest_FRN',
                'BenchmarkRate_FRN',
                'IssuePriceRatio',
                'IssueProceeds',
                'IssueYieldAtIssue',
                'TimeToMaturity',
                'DiscountYield',
                'CleanPrice',
                'AccruedInterest',
                'DirtyValue',
                'DirtyPriceRatio',
            ]:
                initial_bonds_df_global[col] = 0.0
            elif col in ['IssueDate', 'MaturityDate', 'LastAccrualDate']:
                initial_bonds_df_global[col] = pd.NaT
            elif col == 'BondID':
                initial_bonds_df_global[col] = pd.NA
            else:
                initial_bonds_df_global[col] = None

    for col in ['IssueDate', 'MaturityDate', 'LastAccrualDate']:
        initial_bonds_df_global[col] = pd.to_datetime(initial_bonds_df_global[col], errors='coerce')

    num_cols = [
        'FaceValue',
        'CouponRate',
        'OriginalMaturityYears',
        'OriginalPrincipal',
        'AdjustedPrincipal',
        'ReferenceCPI_Issue',
        'IndexRatio',
        'FixedSpread',
        'AccruedInterest_FRN',
        'BenchmarkRate_FRN',
        'IssuePriceRatio',
        'IssueProceeds',
        'IssueYieldAtIssue',
    ]
    for col in num_cols:
        initial_bonds_df_global[col] = pd.to_numeric(initial_bonds_df_global[col], errors='coerce').fillna(0.0)

    initial_bonds_df_global['BondID'] = pd.to_numeric(initial_bonds_df_global['BondID'], errors='coerce').astype('Int64')

    initial_bonds_df_global['SecurityType'] = initial_bonds_df_global['SecurityType'].astype(str).apply(
        lambda x: x if x in SECURITY_TYPES else 'Fixed'
    )
    default_nm_holder = base_config.get('nonmarketable_params', {}).get('initial_holder', 'Private')
    initial_bonds_df_global['HolderType'] = initial_bonds_df_global['HolderType'].astype(str).apply(
        lambda x: x if x in HOLDER_TYPES else default_nm_holder
    )
    initial_bonds_df_global['Status'] = initial_bonds_df_global['Status'].fillna('Active').astype(str)

    tips_init_mask = (initial_bonds_df_global['SecurityType'] == 'TIPS') & (
        initial_bonds_df_global['OriginalPrincipal'] < TGA_FLOOR_TOLERANCE
    )
    initial_bonds_df_global.loc[tips_init_mask, 'OriginalPrincipal'] = initial_bonds_df_global.loc[
        tips_init_mask, 'FaceValue'
    ]
    tips_update_mask = initial_bonds_df_global['SecurityType'] == 'TIPS'
    tips_adj_missing = tips_update_mask & (
        initial_bonds_df_global['AdjustedPrincipal'].isna()
        | (initial_bonds_df_global['AdjustedPrincipal'] < TGA_FLOOR_TOLERANCE)
    )
    initial_bonds_df_global.loc[tips_adj_missing, 'AdjustedPrincipal'] = initial_bonds_df_global.loc[
        tips_adj_missing, 'OriginalPrincipal'
    ]
    tips_ir_missing = tips_update_mask & (
        initial_bonds_df_global['IndexRatio'].isna()
        | (initial_bonds_df_global['IndexRatio'] < TGA_FLOOR_TOLERANCE)
    )
    initial_bonds_df_global.loc[tips_ir_missing, 'IndexRatio'] = 1.0

    issue_price_missing = initial_bonds_df_global['IssuePriceRatio'].isna() | (
        initial_bonds_df_global['IssuePriceRatio'] <= TGA_FLOOR_TOLERANCE
    )
    initial_bonds_df_global.loc[issue_price_missing, 'IssuePriceRatio'] = 1.0
    non_bill_mask = ~(
        (initial_bonds_df_global['SecurityType'] == 'Fixed')
        & (initial_bonds_df_global['CouponRate'].fillna(0.0) <= TGA_FLOOR_TOLERANCE)
        & (initial_bonds_df_global['OriginalMaturityYears'].fillna(np.inf) <= 1.0 + TGA_FLOOR_TOLERANCE)
    )
    issue_proceeds_missing = initial_bonds_df_global['IssueProceeds'].isna() | (
        initial_bonds_df_global['IssueProceeds'] <= TGA_FLOOR_TOLERANCE
    )
    initial_bonds_df_global.loc[issue_proceeds_missing & non_bill_mask, 'IssueProceeds'] = (
        initial_bonds_df_global.loc[issue_proceeds_missing & non_bill_mask, 'FaceValue']
        * initial_bonds_df_global.loc[issue_proceeds_missing & non_bill_mask, 'IssuePriceRatio']
    )
    issue_yield_missing = initial_bonds_df_global['IssueYieldAtIssue'].isna()
    initial_bonds_df_global.loc[issue_yield_missing & non_bill_mask, 'IssueYieldAtIssue'] = initial_bonds_df_global.loc[
        issue_yield_missing & non_bill_mask, 'CouponRate'
    ]

    base_yield_curve = base_config.get('yield_curve', {})
    base_yield_curve_years = base_yield_curve.get('years', [])
    base_yield_curve_rates = base_yield_curve.get('rates', [])
    bill_mask = (
        (initial_bonds_df_global['SecurityType'] == 'Fixed')
        & (initial_bonds_df_global['CouponRate'].fillna(0.0) <= TGA_FLOOR_TOLERANCE)
        & (initial_bonds_df_global['OriginalMaturityYears'].fillna(np.inf) <= 1.0 + TGA_FLOOR_TOLERANCE)
    )
    bill_needs_inference = bill_mask & (
        initial_bonds_df_global['IssueYieldAtIssue'].isna()
        | initial_bonds_df_global['IssueProceeds'].isna()
        | (initial_bonds_df_global['IssueProceeds'] >= initial_bonds_df_global['FaceValue'] - 1.0e-12)
    )
    if bill_needs_inference.any():
        inferred_issue_data = initial_bonds_df_global.loc[bill_needs_inference].apply(
            lambda row: pd.Series(
                infer_issue_data_for_loaded_bill(
                    row['FaceValue'],
                    row['OriginalMaturityYears'],
                    row.get('IssueYieldAtIssue'),
                    base_yield_curve_years,
                    base_yield_curve_rates,
                ),
                index=['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue'],
            ),
            axis=1,
        )
        initial_bonds_df_global.loc[
            bill_needs_inference,
            ['IssuePriceRatio', 'IssueProceeds', 'IssueYieldAtIssue'],
        ] = inferred_issue_data.values

    temp_issuance_profile = base_config.get('treasury_issuance_profile', {})
    fixed_mask = initial_bonds_df_global['SecurityType'] == 'Fixed'
    initial_bonds_df_global.loc[fixed_mask, 'MaturityCategory'] = initial_bonds_df_global.loc[fixed_mask].apply(
        lambda row: get_maturity_category(row['OriginalMaturityYears'], temp_issuance_profile),
        axis=1,
    )

    initial_bonds_df_global = initial_bonds_df_global[BOND_PORTFOLIO_COLS].reset_index(drop=True)
    print('Initial bonds dataframe processed successfully.')
    return initial_bonds_df_global


def _load_initial_portfolio(base_config, sim_start_date, script_dir):
    initial_bonds_df_global = pd.DataFrame(columns=BOND_PORTFOLIO_COLS)

    portfolio_config = base_config.get('initial_portfolio', {})
    portfolio_mode = portfolio_config.get('mode', None)

    if portfolio_mode is None:
        initial_values_config = base_config.get('initial_values', {})
        initial_bonds_path = initial_values_config.get('initial_bonds_file') or base_config.get('initial_bonds_file')
        if initial_bonds_path:
            portfolio_mode = 'file'
        else:
            portfolio_mode = 'empty'
    else:
        initial_bonds_path = portfolio_config.get('file')

    if portfolio_mode in {'generated', 'config_derived'}:
        print(f'Portfolio mode: {portfolio_mode} — running portfolio generator...')
        from csv_gen import generate_initial_portfolio, save_portfolio_csv

        gen_config = portfolio_config.get('generation', base_config.get('initial_portfolio_generation', {}))
        gen_config = copy.deepcopy(gen_config)
        if portfolio_mode == 'config_derived':
            gen_config.setdefault('generation_method', 'config_derived')
        initial_bonds_df_global = generate_initial_portfolio(gen_config, sim_start_date, base_config=base_config)
        output_filename = gen_config.get('output_filename')
        if output_filename:
            save_path = os.path.join(script_dir, output_filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_portfolio_csv(initial_bonds_df_global, save_path)
        print(f'Generated portfolio: {len(initial_bonds_df_global)} bonds.')
        return initial_bonds_df_global

    if portfolio_mode == 'file' and initial_bonds_path:
        if not os.path.isabs(initial_bonds_path):
            initial_bonds_path = os.path.join(script_dir, initial_bonds_path)
        print(f'Attempting to load initial bonds from: {initial_bonds_path}')
        try:
            if not os.path.exists(initial_bonds_path):
                raise FileNotFoundError(f'Initial bonds file not found at {initial_bonds_path}')

            if initial_bonds_path.lower().endswith('.csv'):
                initial_bonds_df_global = pd.read_csv(initial_bonds_path)
            elif initial_bonds_path.lower().endswith(('.xls', '.xlsx')):
                initial_bonds_df_global = pd.read_excel(initial_bonds_path)
            else:
                print(
                    f'Warning: Unsupported initial bonds file format: {initial_bonds_path}. '
                    'Only CSV and Excel supported.'
                )
                initial_bonds_df_global = pd.DataFrame(columns=BOND_PORTFOLIO_COLS)

            return _process_loaded_initial_portfolio(initial_bonds_df_global, base_config)
        except FileNotFoundError as e:
            print(f'FATAL ERROR: {e}.')
            sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR loading/processing initial bonds file '{initial_bonds_path}': {e}.")
            traceback.print_exc()
            sys.exit(1)

    print(f'Portfolio mode: {portfolio_mode} — starting with empty portfolio.')
    return initial_bonds_df_global


def main(config_file=None):
    overall_start_time = time.time()
    try:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
    except NameError:
        project_root = os.getcwd()
    if config_file is None:
        config_file = os.path.join(project_root, 'tdc_config.yaml')
    elif not os.path.isabs(config_file):
        config_file = os.path.join(project_root, config_file)

    print('--- Treasury Deposit Contribution (TDC) Simulator ---')
    print(f'Using configuration file: {config_file}')

    base_config = _load_base_config(config_file)

    base_config_errors = validate_config(base_config)
    if base_config_errors:
        print('FATAL ERROR: Configuration validation failed:')
        for err in base_config_errors:
            print(f' - {err}')
        sys.exit(1)

    sim_period = base_config.get('simulation_period', {})
    sim_start_date = sim_period.get('start_date', '2023-01-01')
    sim_end_date = sim_period.get('end_date', '2025-01-01')
    sim_freq = sim_period.get('frequency', 'W')

    initial_bonds_df_global = _load_initial_portfolio(base_config, sim_start_date, project_root)

    scenario_groups = base_config.get('scenario_groups', [])
    if isinstance(scenario_groups, list) and scenario_groups:
        num_groups = len(scenario_groups)
        print(f'\nFound {num_groups} scenario group(s) defined in the configuration.')
        base_config_subset_for_groups = {k: v for k, v in base_config.items() if k != 'scenario_groups'}

        if num_groups == 1:
            print('Running single scenario group...')
            process_scenario_group(
                scenario_groups[0],
                base_config_subset_for_groups,
                initial_bonds_df_global,
                sim_start_date,
                sim_end_date,
                sim_freq,
                group_index=0,
                total_groups=1,
            )
        else:
            try:
                max_workers_outer = min(
                    num_groups,
                    max(1, os.cpu_count() - 1 if os.cpu_count() else 1),
                    4,
                )
            except NotImplementedError:
                max_workers_outer = min(num_groups, 2)

            print(
                f'Running {num_groups} scenario groups in parallel using up to '
                f'{max_workers_outer} outer workers...'
            )
            outer_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_outer)
            group_futures = []
            group_results_summary = []

            try:
                for group_index, group_def in enumerate(scenario_groups):
                    future = outer_executor.submit(
                        process_scenario_group,
                        group_def,
                        copy.deepcopy(base_config_subset_for_groups),
                        initial_bonds_df_global.copy(deep=True),
                        sim_start_date,
                        sim_end_date,
                        sim_freq,
                        group_index,
                        num_groups,
                    )
                    group_futures.append(future)

                for future in concurrent.futures.as_completed(group_futures):
                    try:
                        result = future.result()
                        group_results_summary.append(result)
                    except Exception as exc:
                        print(f'\n!!! FATAL ERROR processing a scenario group result: {exc} !!!')
                        traceback.print_exc()
                        group_results_summary.append(
                            {
                                'group_name': 'Error Processing Group',
                                'status': 'group_result_error',
                                'message': str(exc),
                            }
                        )
            finally:
                outer_executor.shutdown(wait=True)

            print('\n--- Group Processing Summary ---')
            for summary in sorted(group_results_summary, key=lambda x: x.get('group_name', '')):
                time_str = (
                    f"{summary.get('execution_time', 0):.2f}s"
                    if 'execution_time' in summary
                    else 'N/A'
                )
                print(
                    f" - Group: {summary.get('group_name', 'N/A'):<30} | "
                    f"Status: {summary.get('status', 'N/A'):<25} | Time: {time_str}"
                )
    else:
        print("\nNo 'scenario_groups' list found or list is empty in the config file.")
        print("To run simulations, define at least one group under 'scenario_groups'.")

    overall_execution_time = time.time() - overall_start_time
    print(f'\n--- Overall execution finished in {overall_execution_time:.2f} seconds. ---')


__all__ = [
    'VALID_OVERRIDE_KEYS',
    'OUTPUT_COLUMN_RENAMES',
    '_set_nested_value',
    'update_dict_recursive',
    'apply_event_actions',
    'validate_run_params',
    '_is_bill_like_fixed',
    'calculate_issue_price_ratio',
    'calculate_face_from_proceeds_target',
    'infer_issue_data_for_loaded_bill',
    'get_maturity_category',
    'get_security_category_for_prefs',
    'get_yield_for_maturity',
    'calculate_coupon_rate',
    'get_payment_date',
    'get_coupon_dates_in_period',
    'find_last_coupon_date',
    'calculate_accrued_interest',
    'calculate_bond_market_price',
    'calculate_portfolio_value_and_composition',
    'execute_preference_trades',
    'run_simulation',
    'plot_multi_results',
    'process_scenario_group',
    'main',
]


if __name__ == '__main__':
    main()
