
"""Plotting helpers for simulation result groups."""

import os
import re

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

def plot_multi_results(all_results_data, config_for_plots, scenario_order=None, base_save_filename=None, show_plot=True, output_mode='single_file'):
    """Generates comparative plots (updated for new aggregate types)."""
    group_name_plot = config_for_plots.get('group_name', 'Unknown Group')
    plot_settings_source = config_for_plots
    default_graphs_list = ['TDC_Level', 'Reserves', 'TGA', 'TotalDebt_Agg', 'WAM', 'AuctionProceeds', 'FinancingCost_Cumulative']
    graphs_to_show_raw = plot_settings_source.get('graphs_to_show', default_graphs_list)
    if not isinstance(graphs_to_show_raw, list):
        graphs_to_show_list = []
    else:
        graphs_to_show_list = [g for g in graphs_to_show_raw if isinstance(g, str)]
    plot_zero_thresh_val = plot_settings_source.get('plot_zero_threshold', 1e-09)
    graph_start_date_str = plot_settings_source.get('graph_start_date')
    graph_end_date_str = plot_settings_source.get('graph_end_date')
    output_mode_str = plot_settings_source.get('output_mode', 'single_file').lower()
    group_name = group_name_plot
    graphs_to_show = graphs_to_show_list
    plot_zero_thresh = plot_zero_thresh_val
    graph_start = graph_start_date_str
    graph_end = graph_end_date_str
    output_mode = output_mode_str
    should_save = bool(base_save_filename)
    if not graphs_to_show:
        print(f"WARNING [plot_multi_results for '{group_name_plot}']: No valid graph names specified in 'graphs_to_show' list. No plots will be generated for this group.")
        return
    if not all_results_data:
        print('No simulation results found to plot.')
        return
    if not graphs_to_show:
        print("No graphs specified in 'graphs_to_show'.")
        return
    scenario_names = scenario_order if scenario_order else sorted(all_results_data.keys())
    expanded_graphs_to_show = []
    first_valid_df = next((df for df in all_results_data.values() if isinstance(df, pd.DataFrame) and (not df.empty)), None)
    for graph_key_raw in graphs_to_show:
        if graph_key_raw.lower() == 'debtheldcat_*' and first_valid_df is not None:
            expanded_graphs_to_show.extend(sorted([col for col in first_valid_df.columns if col.startswith('DebtHeldCat_')]))
        elif graph_key_raw.lower() == 'debtheldbytype_*' and first_valid_df is not None:
            expanded_graphs_to_show.extend(sorted([col for col in first_valid_df.columns if col.startswith('DebtHeldByType_')]))
        elif graph_key_raw.lower() == 'debtheld_*' and first_valid_df is not None:
            expanded_graphs_to_show.extend(sorted([col for col in first_valid_df.columns if col.startswith('DebtHeld_')]))
        else:
            expanded_graphs_to_show.append(graph_key_raw)
    final_graphs_to_plot = []
    processed_keys_single = set()
    cb_balance_requested = 'CB_Balances' in expanded_graphs_to_show
    cb_related_keys = ['CB_InterestIncome', 'CB_NetIncome', 'CB_Remittance', 'CB_DeferredAsset']
    for key in expanded_graphs_to_show:
        is_cb_related = key in cb_related_keys
        if is_cb_related:
            if cb_balance_requested and 'CB_Balances' not in processed_keys_single:
                final_graphs_to_plot.append('CB_Balances')
                processed_keys_single.add('CB_Balances')
            elif not cb_balance_requested and key not in processed_keys_single:
                final_graphs_to_plot.append(key)
                processed_keys_single.add(key)
        elif key not in processed_keys_single:
            final_graphs_to_plot.append(key)
            processed_keys_single.add(key)
    num_plots = len(final_graphs_to_plot)
    if num_plots == 0:
        print('No valid graphs to plot after expansion/filtering.')
        return
    if output_mode == 'multiple_files' and should_save:
        files_saved_count = 0
        warned_keys = set()
        for graph_key in final_graphs_to_plot:
            fig, ax = plt.subplots(1, 1, figsize=(14, 5.5))
            plot_title = graph_key.replace('_', ' ')
            y_label = 'Billions USD'
            use_currency_format = True
            plotted_something_on_axis = False
            for scenario_name in scenario_names:
                results_df = all_results_data.get(scenario_name)
                if results_df is None or results_df.empty:
                    continue
                plot_data = results_df.copy()
                if graph_start:
                    plot_data = plot_data[plot_data.index >= pd.to_datetime(graph_start)]
                if graph_end:
                    plot_data = plot_data[plot_data.index <= pd.to_datetime(graph_end)]
                if plot_data.empty:
                    continue
                data_to_plot = None
                if graph_key == 'CB_Balances':
                    plot_title = 'Central Bank Balances & Flows'
                    for col in cb_related_keys:
                        if col in plot_data.columns:
                            col_data = plot_data[col].copy()
                            col_data[col_data.abs() < plot_zero_thresh] = 0.0
                            if not col_data.isnull().all():
                                ax.plot(col_data.index, col_data.values, label=f"{scenario_name} - {col.replace('CB_', '')}", linewidth=1.5)
                                plotted_something_on_axis = True
                elif graph_key == 'WAM':
                    plot_title = 'Weighted Average Maturity (WAM)'
                    y_label = 'Years'
                    use_currency_format = False
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key.startswith('DebtHeldByType_'):
                    plot_title = f"Debt Held By Type: {graph_key.split('_')[-1]}"
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key.startswith('DebtHeld_'):
                    plot_title = f"Debt Held By {graph_key.split('_')[-1]}"
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key in plot_data.columns:
                    plot_title = graph_key.replace('_', ' ')
                    data_to_plot = plot_data[graph_key]
                else:
                    if graph_key not in warned_keys:
                        print(f"Warning: Graph key '{graph_key}' not found in results for scenario '{scenario_name}'. Skipping for this scenario.")
                        warned_keys.add(graph_key)
                    continue
                if data_to_plot is not None and graph_key != 'CB_Balances':
                    data_copy = data_to_plot.copy()
                    data_copy[data_copy.abs() < plot_zero_thresh] = 0.0
                    if not data_copy.isnull().all():
                        ax.plot(data_copy.index, data_copy.values, label=f'{scenario_name}', linewidth=1.5)
                        plotted_something_on_axis = True
            if plotted_something_on_axis:
                ax.set_title(f'{group_name}: {plot_title}')
                ax.set_ylabel(y_label)
                if use_currency_format:
                    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
                else:
                    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    if len(labels) > 6:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    else:
                        ax.legend(loc='best', fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.7)
                plt.xlabel('Date')
                fig.tight_layout()
                clean_graph_key = re.sub('[^\\w\\-]+', '_', graph_key)
                save_path_single = f'{base_save_filename}_{clean_graph_key}.png'
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path_single)), exist_ok=True)
                    fig.savefig(save_path_single, bbox_inches='tight', dpi=300)
                    files_saved_count += 1
                except Exception as e:
                    print(f'   Error saving plot {save_path_single}: {e}')
            plt.close(fig)
    else:
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, num_plots * 4.5), sharex=True, squeeze=False)
        axes = axes.flatten()
        valid_plots_generated = False
        warned_keys = set()
        for i, graph_key in enumerate(final_graphs_to_plot):
            if i >= len(axes):
                break
            ax = axes[i]
            plot_title = graph_key.replace('_', ' ')
            y_label = 'Billions USD'
            use_currency_format = True
            plotted_something_on_axis = False
            for scenario_name in scenario_names:
                results_df = all_results_data.get(scenario_name)
                if results_df is None or results_df.empty:
                    continue
                plot_data = results_df.copy()
                if graph_start:
                    plot_data = plot_data[plot_data.index >= pd.to_datetime(graph_start)]
                if graph_end:
                    plot_data = plot_data[plot_data.index <= pd.to_datetime(graph_end)]
                if plot_data.empty:
                    continue
                data_to_plot = None
                if graph_key == 'CB_Balances':
                    plot_title = 'Central Bank Balances & Flows'
                    for col in cb_related_keys:
                        if col in plot_data.columns:
                            col_data = plot_data[col].copy()
                            col_data[col_data.abs() < plot_zero_thresh] = 0.0
                            if not col_data.isnull().all():
                                ax.plot(col_data.index, col_data.values, label=f"{scenario_name} - {col.replace('CB_', '')}", linewidth=1.5)
                                plotted_something_on_axis = True
                elif graph_key == 'WAM':
                    plot_title = 'Weighted Average Maturity (WAM)'
                    y_label = 'Years'
                    use_currency_format = False
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key.startswith('DebtHeldByType_'):
                    plot_title = f"Debt Held By Type: {graph_key.split('_')[-1]}"
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key.startswith('DebtHeld_'):
                    plot_title = f"Debt Held By {graph_key.split('_')[-1]}"
                    if graph_key in plot_data.columns:
                        data_to_plot = plot_data[graph_key]
                elif graph_key in plot_data.columns:
                    plot_title = graph_key.replace('_', ' ')
                    data_to_plot = plot_data[graph_key]
                else:
                    if graph_key not in warned_keys:
                        print(f"Warning: Graph key '{graph_key}' not found. Skipping subplot.")
                        warned_keys.add(graph_key)
                    ax.set_visible(False)
                    continue
                if data_to_plot is not None and graph_key != 'CB_Balances':
                    data_copy = data_to_plot.copy()
                    data_copy[data_copy.abs() < plot_zero_thresh] = 0.0
                    if not data_copy.isnull().all():
                        ax.plot(data_copy.index, data_copy.values, label=f'{scenario_name}', linewidth=1.5)
                        plotted_something_on_axis = True
            if plotted_something_on_axis:
                valid_plots_generated = True
                ax.set_title(f'{plot_title}')
                ax.set_ylabel(y_label)
                if use_currency_format:
                    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('${x:,.0f}'))
                else:
                    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.1f}'))
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    if len(labels) > 6:
                        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                    else:
                        ax.legend(loc='best', fontsize='small')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_visible(True)
            elif ax.get_visible():
                ax.set_visible(False)
        if valid_plots_generated:
            plt.xlabel('Date')
            sim_period_plot = config_for_plots.get('simulation_period', {})
            sim_start_plot = sim_period_plot.get('start_date', 'N/A')
            sim_end_plot = sim_period_plot.get('end_date', 'N/A')
            fig.suptitle(f'{group_name}: Comparative Results ({sim_start_plot} to {sim_end_plot})', fontsize=16, y=0.98)
            try:
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            except Exception as layout_err:
                print(f'Warning: Plot layout adjustment failed: {layout_err}')
            if should_save:
                save_path_combined = f'{base_save_filename}.png'
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(save_path_combined)), exist_ok=True)
                    print(f'Attempting to save combined plot: {save_path_combined}')
                    fig.savefig(save_path_combined, bbox_inches='tight', dpi=300)
                    print(f'Combined plot saved successfully.')
                except Exception as e:
                    print(f'Error saving combined plot to {save_path_combined}: {e}')
            if show_plot:
                print('Plot saved to file (non-interactive backend).')
        plt.close(fig)


__all__ = ['plot_multi_results']
