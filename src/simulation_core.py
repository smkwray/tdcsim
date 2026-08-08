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
    PRIVATE_SUBBUCKETS,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
)
from tdc_validation import validate_config

from sim_engine import run_simulation
from sim_groups import process_scenario_group, resolve_worker_count
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
        raise ValueError('Initial portfolio file contains no data rows.')

    print(f'Loaded initial portfolio ({len(initial_bonds_df_global)} rows). Processing...')

    required_columns = (
        'BondID',
        'SecurityType',
        'IssueDate',
        'MaturityDate',
        'OriginalMaturityYears',
        'FaceValue',
        'CouponRate',
        'HolderType',
        'Status',
    )
    missing_columns = [
        column for column in required_columns if column not in initial_bonds_df_global
    ]
    if missing_columns:
        raise ValueError(
            f'Initial portfolio is missing required columns: {missing_columns}'
        )
    for column in required_columns:
        present = (
            initial_bonds_df_global[column].notna()
            & initial_bonds_df_global[column].astype('string').str.strip().ne('')
        )
        if not present.all():
            raise ValueError(
                f'Initial portfolio required column {column} contains missing values.'
            )

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
                'InterestPaymentFrequency',
            ]:
                initial_bonds_df_global[col] = np.nan
            elif col in ['IssueDate', 'MaturityDate', 'DatedDate', 'OriginalDatedDate', 'FirstInterestPaymentDate', 'LastAccrualDate']:
                initial_bonds_df_global[col] = pd.NaT
            elif col == 'BondID':
                initial_bonds_df_global[col] = pd.NA
            else:
                initial_bonds_df_global[col] = None

    for col in ['IssueDate', 'MaturityDate', 'DatedDate', 'OriginalDatedDate', 'FirstInterestPaymentDate', 'LastAccrualDate']:
        present = (
            initial_bonds_df_global[col].notna()
            & initial_bonds_df_global[col].astype('string').str.strip().ne('')
        )
        parsed = pd.to_datetime(initial_bonds_df_global[col], errors='coerce')
        if parsed[present].isna().any():
            raise ValueError(f'Initial portfolio column {col} contains malformed dates.')
        initial_bonds_df_global[col] = parsed

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
        'InterestPaymentFrequency',
    ]
    for col in num_cols:
        present = (
            initial_bonds_df_global[col].notna()
            & initial_bonds_df_global[col].astype('string').str.strip().ne('')
        )
        parsed = pd.to_numeric(initial_bonds_df_global[col], errors='coerce')
        if parsed[present].isna().any() or not parsed[present].map(np.isfinite).all():
            raise ValueError(f'Initial portfolio column {col} contains malformed numerics.')
        initial_bonds_df_global[col] = parsed

    bond_ids = pd.to_numeric(initial_bonds_df_global['BondID'], errors='coerce')
    if bond_ids.isna().any() or not bond_ids.map(np.isfinite).all():
        raise ValueError('Initial portfolio BondID contains malformed numerics.')
    if not bond_ids.map(lambda value: float(value).is_integer()).all():
        raise ValueError('Initial portfolio BondID values must be integers.')
    if bond_ids.duplicated().any():
        raise ValueError('Initial portfolio BondID values must be unique.')
    initial_bonds_df_global['BondID'] = bond_ids.astype('Int64')

    initial_bonds_df_global['SecurityType'] = initial_bonds_df_global['SecurityType'].astype(str)
    unknown_security_types = sorted(
        set(initial_bonds_df_global['SecurityType']) - set(SECURITY_TYPES)
    )
    if unknown_security_types:
        raise ValueError(
            f'Initial portfolio contains unknown SecurityType values: {unknown_security_types}'
        )
    initial_bonds_df_global['HolderType'] = initial_bonds_df_global['HolderType'].astype(str)
    unknown_holder_types = sorted(
        set(initial_bonds_df_global['HolderType']) - set(HOLDER_TYPES)
    )
    if unknown_holder_types:
        raise ValueError(
            f'Initial portfolio contains unknown HolderType values: {unknown_holder_types}'
        )
    initial_bonds_df_global['HolderSubBucket'] = (
        initial_bonds_df_global['HolderSubBucket']
        .fillna("")
        .astype(str)
        .replace({"<NA>": "", "nan": "", "None": ""})
    )
    private_mask = initial_bonds_df_global['HolderType'] == 'Private'
    valid_private_subbucket = initial_bonds_df_global['HolderSubBucket'].isin(PRIVATE_SUBBUCKETS)
    if (private_mask & ~valid_private_subbucket).any():
        raise ValueError(
            'Initial portfolio Private rows require a canonical HolderSubBucket.'
        )
    if (~private_mask & initial_bonds_df_global['HolderSubBucket'].ne('')).any():
        raise ValueError(
            'Initial portfolio non-Private rows must not declare HolderSubBucket.'
        )
    initial_bonds_df_global['Status'] = initial_bonds_df_global['Status'].astype(str)
    invalid_statuses = sorted(set(initial_bonds_df_global['Status']) - {'Active'})
    if invalid_statuses:
        raise ValueError(
            f'Initial portfolio contains unsupported Status values: {invalid_statuses}'
        )
    if (initial_bonds_df_global['FaceValue'] < 0.0).any():
        raise ValueError('Initial portfolio FaceValue must be nonnegative.')
    if (initial_bonds_df_global['OriginalMaturityYears'] <= 0.0).any():
        raise ValueError('Initial portfolio OriginalMaturityYears must be positive.')
    if (initial_bonds_df_global['CouponRate'] < 0.0).any():
        raise ValueError('Initial portfolio CouponRate must be nonnegative.')
    if (initial_bonds_df_global['IssueDate'] >= initial_bonds_df_global['MaturityDate']).any():
        raise ValueError('Initial portfolio IssueDate must precede MaturityDate.')

    tips_rows = initial_bonds_df_global['SecurityType'].eq('TIPS')
    for column in ('OriginalPrincipal', 'AdjustedPrincipal', 'ReferenceCPI_Issue', 'IndexRatio'):
        if tips_rows.any() and initial_bonds_df_global.loc[tips_rows, column].isna().any():
            raise ValueError(f'Initial portfolio TIPS rows require {column}.')
    frn_rows = initial_bonds_df_global['SecurityType'].eq('FRN')
    if frn_rows.any() and initial_bonds_df_global.loc[frn_rows, 'FixedSpread'].isna().any():
        raise ValueError('Initial portfolio FRN rows require FixedSpread.')

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

    allowed_portfolio_modes = {'empty', 'file', 'generated', 'config_derived'}
    if portfolio_mode not in allowed_portfolio_modes:
        raise ValueError(
            f'Unsupported initial portfolio mode {portfolio_mode!r}; '
            f'expected one of {sorted(allowed_portfolio_modes)}.'
        )
    if portfolio_mode == 'file' and not initial_bonds_path:
        raise ValueError('Initial portfolio mode file requires initial_portfolio.file.')

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
                raise ValueError(
                    f'Unsupported initial portfolio file format: {initial_bonds_path}. '
                    'Only CSV and Excel are supported.'
                )

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
            group_summary = process_scenario_group(
                scenario_groups[0],
                base_config_subset_for_groups,
                initial_bonds_df_global,
                sim_start_date,
                sim_end_date,
                sim_freq,
                group_index=0,
                total_groups=1,
            )
            if group_summary.get('status') != 'completed':
                print(
                    'FATAL ERROR: Scenario group did not complete: '
                    f"{group_summary.get('message', group_summary.get('status'))}"
                )
                raise SystemExit(1)
        else:
            try:
                cpu_count = os.cpu_count() or 1
            except NotImplementedError:
                cpu_count = 1
            parallel_cfg = base_config.get('parallel_execution', {})
            configured_group_workers = (
                parallel_cfg.get('group_workers')
                if isinstance(parallel_cfg, dict)
                else None
            )
            default_group_workers = min(num_groups, max(1, cpu_count - 1), 4)
            max_workers_outer = resolve_worker_count(
                configured=configured_group_workers,
                env_var='TDCSIM_GROUP_WORKERS',
                default=default_group_workers,
                upper_bound=num_groups,
            )

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
            failed_groups = [
                summary
                for summary in group_results_summary
                if summary.get('status') != 'completed'
            ]
            if failed_groups:
                print('FATAL ERROR: One or more scenario groups did not complete.')
                raise SystemExit(1)
    else:
        print("\nFATAL ERROR: Define at least one scenario group in the configuration.")
        raise SystemExit(1)

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
    main(sys.argv[1] if len(sys.argv) > 1 else None)
