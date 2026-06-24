
"""Scenario-group orchestration for simulation batches and plotting."""

import concurrent.futures
import copy
import json
import os
import re
import time
import traceback
from collections.abc import Mapping, Sequence

import pandas as pd

from tdc_shared import BOND_PORTFOLIO_COLS
from sim_engine import run_simulation
from sim_helpers import VALID_OVERRIDE_KEYS, update_dict_recursive, validate_cbo_config_blocks
from sim_plotting import plot_multi_results
from ratewall_contract import export_ratewall_bundle

def _clean_filename(value):
    """Return a filesystem-safe stem for generated output files."""
    return re.sub('[^\\w\\-]+', '_', str(value)).strip('_') or 'results'


def _positive_int(value):
    """Return a positive integer or None for unset/auto values."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {'', 'auto', 'default'}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_worker_count(*, configured=None, env_var=None, default, upper_bound):
    """Resolve a bounded worker count from config, env override, and default."""
    env_value = _positive_int(os.environ.get(env_var)) if env_var else None
    configured_value = _positive_int(configured)
    requested = env_value or configured_value or default
    return max(1, min(int(requested), int(upper_bound)))


def _resolve_output_base(group_output_settings, group_name, filename_key):
    configured_filename = group_output_settings.get(filename_key)
    if not configured_filename:
        configured_filename = group_output_settings.get('save_plot_filename') or group_name
    base_name, _ = os.path.splitext(configured_filename)
    clean_base_name = _clean_filename(os.path.basename(base_name))
    save_dir = os.path.dirname(configured_filename) or '.'
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, clean_base_name)


def _metadata_value_for_csv(value):
    """Return a stable CSV-friendly metadata value, or None for nested objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        simple_values = []
        for item in value:
            if item is None or isinstance(item, (str, int, float, bool)):
                simple_values.append(item)
            else:
                return None
        return json.dumps(simple_values, sort_keys=True)
    return None


def _metadata_row_for_scenario(scenario_name, results_df):
    run_metadata = getattr(results_df, 'attrs', {}).get('run_metadata')
    if not isinstance(run_metadata, Mapping) or not run_metadata:
        return None
    row = {'Scenario': scenario_name}
    for key in sorted(run_metadata):
        value = _metadata_value_for_csv(run_metadata[key])
        if value is not None:
            row[str(key)] = value
    return row


def save_group_results_csv(successful_scenario_results, scenario_order, group_output_settings, group_name):
    """Write scenario time-series and final-state CSVs for downstream charting."""
    output_base = _resolve_output_base(group_output_settings, group_name, 'save_results_filename')
    long_frames = []
    metadata_rows = []

    for scenario_name in scenario_order:
        results_df = successful_scenario_results.get(scenario_name)
        if results_df is None or results_df.empty:
            continue
        metadata_rows.append(_metadata_row_for_scenario(scenario_name, results_df) or {'Scenario': scenario_name})
        scenario_frame = results_df.copy()
        scenario_frame.insert(0, 'Scenario', scenario_name)
        scenario_frame.insert(1, 'Date', scenario_frame.index)
        long_frames.append(scenario_frame.reset_index(drop=True))

        scenario_stem = _clean_filename(scenario_name)
        scenario_path = f'{output_base}_{scenario_stem}.csv'
        scenario_frame.to_csv(scenario_path, index=False)

    if not long_frames:
        print(f"Warning [Group {group_name}]: No successful scenario data available for CSV export.")
        return

    combined = pd.concat(long_frames, ignore_index=True)
    combined_path = f'{output_base}_results.csv'
    combined.to_csv(combined_path, index=False)

    final_state = combined.sort_values('Date').groupby('Scenario', sort=False).tail(1)
    final_state_path = f'{output_base}_final_state.csv'
    final_state.to_csv(final_state_path, index=False)

    if any(len(row) > 1 for row in metadata_rows):
        metadata_path = f'{output_base}_metadata.csv'
        metadata = pd.DataFrame(metadata_rows)
        metadata.to_csv(metadata_path, index=False)
        print(f'[Group {group_name}] Metadata CSV saved: {metadata_path}')

    print(f'[Group {group_name}] Results CSV saved: {combined_path}')
    print(f'[Group {group_name}] Final-state CSV saved: {final_state_path}')


def process_scenario_group(group_def, base_config_subset, initial_bonds_df_global, sim_start_date, sim_end_date, sim_freq, group_index, total_groups):
    """Processes scenarios within a group, runs simulations, and handles plotting."""
    group_start_time = time.time()
    group_name = group_def.get('group_name', f'Group_{group_index + 1}')
    group_scenarios = group_def.get('scenarios', [])
    group_output_settings = group_def.get('group_output_settings', {})
    output_mode = group_output_settings.get('output_mode', 'single_file').lower()
    save_results_csv = bool(group_output_settings.get('save_results_csv', False))
    show_plots_interactively = total_groups == 1
    save_plots_to_file = total_groups > 1 or (total_groups == 1 and group_output_settings.get('save_plot_filename'))
    actual_save_filename_base = None
    if save_plots_to_file:
        configured_filename = group_output_settings.get('save_plot_filename')
        if configured_filename:
            base_name, _ = os.path.splitext(configured_filename)
            clean_base_name = re.sub('[^\\w\\-]+', '_', os.path.basename(base_name))
            save_dir = os.path.dirname(configured_filename)
            if not save_dir:
                save_dir = '.'
            if not os.path.exists(save_dir) and save_dir != '.':
                try:
                    os.makedirs(save_dir)
                    print(f'Created directory for saving plots: {save_dir}')
                except OSError as e:
                    print(f"Warning: Could not create directory '{save_dir}': {e}. Saving to script directory.")
                    save_dir = '.'
            actual_save_filename_base = os.path.join(save_dir, clean_base_name)
        else:
            print(f"Warning [Group {group_name}]: Saving enabled but no 'save_plot_filename' provided. Plots will not be saved.")
            save_plots_to_file = False
    print(f'\n--- Processing Group {group_index + 1}/{total_groups}: {group_name} ({len(group_scenarios)} scenarios) ---')
    if not group_scenarios or not isinstance(group_scenarios, list):
        print(f"Warning [Group {group_name}]: No valid 'scenarios' list found. Skipping group.")
        return {'group_name': group_name, 'status': 'skipped', 'message': 'No scenarios found'}
    scenario_configs_to_run_group = []
    scenario_names_ordered_group = []
    plot_config_group = copy.deepcopy(base_config_subset)
    plot_config_group['group_name'] = group_name
    if isinstance(group_output_settings, dict):
        plot_config_group = update_dict_recursive(plot_config_group, group_output_settings)
    group_overrides = group_def.get('overrides', {})
    if group_overrides and not isinstance(group_overrides, dict):
        raise ValueError(f"[{group_name}] Group overrides must be a mapping.")
    if isinstance(group_overrides, dict) and group_overrides:
        unknown_group_override_keys = set(group_overrides.keys()) - VALID_OVERRIDE_KEYS
        if unknown_group_override_keys:
            raise ValueError(
                f"[{group_name}] Unknown group override keys (possible typos): {sorted(unknown_group_override_keys)}. "
                f"Valid keys: {sorted(VALID_OVERRIDE_KEYS)}"
            )
        group_cbo_errors = validate_cbo_config_blocks(group_overrides, label=f"group_overrides[{group_name}]")
        if group_cbo_errors:
            raise ValueError(f"[{group_name}] Group CBO override validation failed: {'; '.join(group_cbo_errors)}")
    for scen_index, scenario_def in enumerate(group_scenarios):
        if not isinstance(scenario_def, dict):
            print(f'Warning [Group {group_name}]: Scenario definition {scen_index} is not a valid dictionary. Skipping.')
            continue
        scenario_name = scenario_def.get('name', f'{group_name}_Scen_{scen_index + 1}')
        overrides = scenario_def.get('overrides', {})
        if not isinstance(overrides, dict):
            print(f"Warning [Group {group_name}]: 'overrides' for scenario '{scenario_name}' is not a valid dictionary. Skipping scenario.")
            continue
        unknown_override_keys = set(overrides.keys()) - VALID_OVERRIDE_KEYS
        if unknown_override_keys:
            raise ValueError(
                f"[{scenario_name}] Unknown override keys (possible typos): {sorted(unknown_override_keys)}. "
                f"Valid keys: {sorted(VALID_OVERRIDE_KEYS)}"
            )
        cbo_override_errors = validate_cbo_config_blocks(overrides, label=f"scenario_overrides[{scenario_name}]")
        if cbo_override_errors:
            raise ValueError(f"[{scenario_name}] CBO override validation failed: {'; '.join(cbo_override_errors)}")
        scenario_params = copy.deepcopy(base_config_subset)
        scenario_params = update_dict_recursive(scenario_params, overrides)
        if group_overrides:
            group_adjusted_params = update_dict_recursive(copy.deepcopy(base_config_subset), group_overrides)
            scenario_params = update_dict_recursive(group_adjusted_params, overrides)
        scenario_params['initial_bonds_df'] = initial_bonds_df_global.copy(deep=True) if initial_bonds_df_global is not None else pd.DataFrame(columns=BOND_PORTFOLIO_COLS)
        scenario_configs_to_run_group.append({'name': scenario_name, 'params': scenario_params})
        scenario_names_ordered_group.append(scenario_name)
    group_results = {}
    num_scenarios_in_group = len(scenario_configs_to_run_group)
    if num_scenarios_in_group == 0:
        print(f'[Group {group_name}] No valid scenarios prepared for execution. Skipping group.')
        return {'group_name': group_name, 'status': 'skipped', 'message': 'No valid scenarios'}
    try:
        cpu_count = os.cpu_count() or 1
    except NotImplementedError:
        cpu_count = 1
    default_inner_workers = min(
        num_scenarios_in_group,
        max(1, cpu_count // 2),
        8,
    )
    configured_inner_workers = (
        group_def.get('parallel_workers')
        or group_output_settings.get('parallel_workers')
        or base_config_subset.get('scenario_parallel_workers')
    )
    max_workers_inner = resolve_worker_count(
        configured=configured_inner_workers,
        env_var='TDCSIM_SCENARIO_WORKERS',
        default=default_inner_workers,
        upper_bound=num_scenarios_in_group,
    )
    print(f'[Group {group_name}] Running {num_scenarios_in_group} scenarios using up to {max_workers_inner} parallel processes...')
    inner_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers_inner)
    group_sim_success = True
    future_to_scenario = {}
    try:
        for s_info in scenario_configs_to_run_group:
            future = inner_executor.submit(run_simulation, s_info['params'], sim_start_date, sim_end_date, sim_freq, s_info['name'])
            future_to_scenario[future] = s_info['name']
        for future in concurrent.futures.as_completed(future_to_scenario):
            scenario_name = future_to_scenario[future]
            try:
                results_df, _ = future.result()
                group_results[scenario_name] = results_df
                print(f'   [Group {group_name}] Completed Scenario: {scenario_name}')
            except Exception as exc:
                print(f"\n   !!! [Group {group_name}] Scenario '{scenario_name}' FAILED during execution !!!")
                print(f'   Exception Type: {type(exc).__name__}: {exc}')
                group_results[scenario_name] = pd.DataFrame()
                group_sim_success = False
    finally:
        inner_executor.shutdown(wait=True)
    group_execution_time = time.time() - group_start_time
    print(f'[Group {group_name}] Scenario executions finished in {group_execution_time:.2f} seconds.')
    successful_scenario_results = {name: df for name, df in group_results.items() if isinstance(df, pd.DataFrame) and (not df.empty)}
    successful_scenario_names_group = [name for name in scenario_names_ordered_group if name in successful_scenario_results]
    if not successful_scenario_names_group:
        print(f'ERROR [Group {group_name}]: No scenarios completed successfully. Skipping results summary and plots.')
        return {'group_name': group_name, 'status': 'failed', 'message': 'No successful simulations'}
    else:
        if len(successful_scenario_names_group) < len(scenario_names_ordered_group):
            failed_names = sorted([name for name in scenario_names_ordered_group if name not in successful_scenario_names_group])
            print(f'Warning [Group {group_name}]: The following scenarios failed or produced empty results: {failed_names}')
        print(f'\n--- [Group {group_name}] Final States (Sample) ---')
        try:
            pd.set_option('display.float_format', '{:,.1f}'.format)
            sample_cols = ['TGA', 'TDC_Level', 'Reserves', 'TotalDebt_Agg', 'DebtHeld_Banks', 'DebtHeld_CentralBank', 'DebtHeld_DomesticNonBanks', 'DebtHeld_FedInternal', 'DebtHeld_TrustFunds', 'WAM', 'AuctionProceeds', 'InterestOutlay_Cumulative', 'IssueDiscountCost_Cumulative', 'FinancingCost_Cumulative', 'DebtServiceOutlay_Cumulative']
            for scenario_name in successful_scenario_names_group[:min(5, len(successful_scenario_names_group))]:
                results_df = successful_scenario_results[scenario_name]
                sample_cols_exist = [col for col in sample_cols if col in results_df.columns]
                if not results_df.empty and sample_cols_exist:
                    print(f'\n   Scenario: {scenario_name}')
                    print(results_df.iloc[-1][sample_cols_exist].to_string())
                elif not results_df.empty:
                    print(f'   {scenario_name}: Missing some sample columns for final state display.')
                else:
                    print(f'   {scenario_name}: Results DataFrame is empty.')
        except Exception as e:
            print(f'   Error generating final state summary: {e}')
        finally:
            pd.reset_option('display.float_format')
        if save_results_csv:
            try:
                save_group_results_csv(
                    successful_scenario_results,
                    successful_scenario_names_group,
                    group_output_settings,
                    group_name,
                )
            except Exception as e:
                print(f'\nERROR [Group {group_name}]: Results CSV export failed: {e}')
                traceback.print_exc()
                return {'group_name': group_name, 'status': 'csv_export_failed', 'message': str(e)}
        ratewall_contract_cfg = base_config_subset.get('ratewall_contract', {})
        if isinstance(ratewall_contract_cfg, dict) and ratewall_contract_cfg.get('enabled', False):
            try:
                output_dir = ratewall_contract_cfg.get(
                    'output_dir',
                    os.path.join('output', 'ratewall_contract'),
                )
                export_ratewall_bundle(
                    successful_scenario_results,
                    output_dir,
                    config=ratewall_contract_cfg,
                )
                print(f'[Group {group_name}] RateWall contract bundle saved: {output_dir}')
            except Exception as e:
                print(f'\nERROR [Group {group_name}]: RateWall contract export failed: {e}')
                traceback.print_exc()
                return {'group_name': group_name, 'status': 'ratewall_contract_export_failed', 'message': str(e)}
        try:
            plot_multi_results(successful_scenario_results, plot_config_group, scenario_order=successful_scenario_names_group, base_save_filename=actual_save_filename_base if save_plots_to_file else None, show_plot=show_plots_interactively, output_mode=output_mode)
        except ImportError:
            print('\nPlotting requires matplotlib. Please install it: pip install matplotlib')
        except Exception as e:
            print(f'\nERROR [Group {group_name}]: Plotting failed: {e}')
            traceback.print_exc()
            return {'group_name': group_name, 'status': 'plotting_failed', 'message': str(e)}
    status = 'completed_with_failures' if not group_sim_success else 'completed'
    return {'group_name': group_name, 'status': status, 'execution_time': group_execution_time}


__all__ = ['process_scenario_group', 'resolve_worker_count', 'save_group_results_csv']
