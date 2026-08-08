"""Scenario-group output contract tests."""

from __future__ import annotations

from pathlib import Path
import concurrent.futures

import pandas as pd
import pytest

import sim_groups
import simulation_core
from sim_groups import process_scenario_group, save_group_results_csv


def _scenario_frame(start_value: int) -> pd.DataFrame:
    """Return a tiny time-series frame matching chart-export inputs."""
    return pd.DataFrame(
        {
            'TDC_Level': [start_value, start_value + 1, start_value + 2],
            'Reserves': [100 + start_value, 101 + start_value, 102 + start_value],
        },
        index=pd.to_datetime(['2025-01-01', '2025-01-08', '2025-01-15']),
    )


class _ImmediateExecutor:
    """Execute submitted callables synchronously while returning real futures."""

    def __init__(self, *args, **kwargs):
        pass

    def submit(self, function, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True):
        pass


def test_process_scenario_group_does_not_publish_a_successful_subset(
    monkeypatch,
) -> None:
    exported = []

    def fake_run(_params, _start, _end, _frequency, scenario_name):
        if scenario_name == 'failed':
            raise RuntimeError('deliberate failure')
        return _scenario_frame(1), {}

    monkeypatch.setattr(
        sim_groups.concurrent.futures,
        'ProcessPoolExecutor',
        _ImmediateExecutor,
    )
    monkeypatch.setattr(sim_groups, 'run_simulation', fake_run)
    monkeypatch.setattr(
        sim_groups,
        'save_group_results_csv',
        lambda *args, **kwargs: exported.append('csv'),
    )
    monkeypatch.setattr(
        sim_groups,
        'export_ratewall_bundle',
        lambda *args, **kwargs: exported.append('ratewall'),
    )
    monkeypatch.setattr(
        sim_groups,
        'plot_multi_results',
        lambda *args, **kwargs: exported.append('plot'),
    )

    result = process_scenario_group(
        {
            'group_name': 'atomic-group',
            'scenarios': [
                {'name': 'successful', 'overrides': {}},
                {'name': 'failed', 'overrides': {}},
            ],
            'group_output_settings': {'save_results_csv': True},
        },
        {'ratewall_contract': {'enabled': True}},
        pd.DataFrame(),
        '2025-01-01',
        '2025-01-15',
        'W',
        0,
        1,
    )

    assert result['status'] == 'failed'
    assert result['failed_scenarios'] == ['failed']
    assert exported == []


def test_simulation_main_exits_nonzero_when_single_group_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        simulation_core,
        '_load_base_config',
        lambda _path: {
            'simulation_period': {
                'start_date': '2025-01-01',
                'end_date': '2025-01-15',
                'frequency': 'W',
            },
            'scenario_groups': [{'group_name': 'atomic-group', 'scenarios': [{}]}],
        },
    )
    monkeypatch.setattr(simulation_core, 'validate_config', lambda _config: [])
    monkeypatch.setattr(
        simulation_core,
        '_load_initial_portfolio',
        lambda *_args: pd.DataFrame(),
    )
    monkeypatch.setattr(
        simulation_core,
        'process_scenario_group',
        lambda *_args, **_kwargs: {
            'group_name': 'atomic-group',
            'status': 'failed',
            'message': 'scenario failed',
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        simulation_core.main('unused.yaml')

    assert exc_info.value.code == 1


def _cbo_export_frame(start_value: int) -> pd.DataFrame:
    """Return a tiny frame with CBO diagnostics and renamed holder columns."""
    frame = pd.DataFrame(
        {
            'DebtHeld_DomesticNonBanks': [200 + start_value, 210 + start_value],
            'DebtHeld_CentralBank': [300 + start_value, 320 + start_value],
            'CBOFundingModeActive': [1.0, 1.0],
            'FundingMode': ['cbo_public_debt_target', 'cbo_public_debt_target'],
            'CBOPrimaryDeficitFlow': [25.0, 27.5],
            'CBOControlledDebtTarget': [1_100.0, 1_125.0],
            'CBOControlledDebtPreIssuance': [1_000.0, 1_010.0],
            'CBOControlledDebtPostIssuance': [1_100.0, 1_125.0],
            'CBOControlledDebtTargetError': [0.0, 0.0],
            'CBORequiredFaceIssuance': [100.0, 115.0],
            'CBOOperatingCashTarget': [800.0, 805.0],
            'CBOCashResidual': [3.0, 4.5],
            'CBOCashReconciliationResidual': [2.0, 2.5],
            'CBOFiscalIncidencePolicyPresent': [1.0, 1.0],
            'FiscalIncidenceStatus': ['explicit_policy_present', 'explicit_policy_present'],
            'CBORemittanceCashEffect': [0.0, 0.0],
            'CBONetInterestDiagnostic': [120.0, 125.0],
            'CBOTotalDeficitDiagnostic': [200.0, 207.5],
            'NetInterestDiagnosticStatus': ['cbo_reported_check_only', 'cbo_reported_check_only'],
            'CBONetInterestBridgeRows': [3.0, 3.0],
            'CBOCashResidualStatus': ['operating_cash_target_loaded', 'operating_cash_target_loaded'],
        },
        index=pd.to_datetime(['2026-09-20', '2026-09-30']),
    )
    frame.attrs['run_metadata'] = {
        'scenario_name': f'cbo_{start_value}',
        'funding_rule_mode': 'cbo_public_debt_target',
        'cbo_funding_mode_active': True,
        'cbo_net_interest_bridge_rows': 3,
        'mmf_deposit_pass_through_sensitivity_grid': [0.9, 0.97, 1.0],
        'nested_debug_payload': {'not': 'csv-friendly'},
    }
    return frame


def test_save_group_results_csv_writes_long_form_and_final_state(tmp_path):
    """Downstream charting expects per-scenario and combined CSVs keyed by Scenario/Date."""
    output_file = tmp_path / 'group export.csv'
    results = {
        'baseline': _scenario_frame(10),
        'stress/case': _scenario_frame(20),
    }

    save_group_results_csv(
        results,
        ['baseline', 'stress/case'],
        {'save_results_filename': str(output_file)},
        'Chart Group',
    )

    expected_paths = {
        'baseline': tmp_path / 'group_export_baseline.csv',
        'stress/case': tmp_path / 'group_export_stress_case.csv',
        'combined': tmp_path / 'group_export_results.csv',
        'final_state': tmp_path / 'group_export_final_state.csv',
    }
    for path in expected_paths.values():
        assert path.exists()
    assert not (tmp_path / 'group_export_metadata.csv').exists()

    baseline = pd.read_csv(expected_paths['baseline'])
    combined = pd.read_csv(expected_paths['combined'])
    final_state = pd.read_csv(expected_paths['final_state'])

    assert list(baseline.columns[:2]) == ['Scenario', 'Date']
    assert list(combined.columns[:2]) == ['Scenario', 'Date']
    assert list(final_state.columns[:2]) == ['Scenario', 'Date']
    assert len(combined) == 6
    assert set(combined['Scenario']) == {'baseline', 'stress/case'}

    expected_last_rows = (
        combined.sort_values('Date')
        .groupby('Scenario', sort=False)
        .tail(1)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        final_state.reset_index(drop=True),
        expected_last_rows,
        check_dtype=False,
    )


def test_save_group_results_csv_preserves_cbo_diagnostics_and_renamed_holder_columns(tmp_path):
    """CBO engine output columns should pass through every group CSV surface unchanged."""
    output_file = tmp_path / 'cbo export.csv'
    results = {
        'baseline': _cbo_export_frame(0),
        'policy-shift': _cbo_export_frame(50),
    }

    save_group_results_csv(
        results,
        ['baseline', 'policy-shift'],
        {'save_results_filename': str(output_file)},
        'CBO Group',
    )

    per_scenario = pd.read_csv(tmp_path / 'cbo_export_baseline.csv')
    combined = pd.read_csv(tmp_path / 'cbo_export_results.csv')
    final_state = pd.read_csv(tmp_path / 'cbo_export_final_state.csv')
    metadata = pd.read_csv(tmp_path / 'cbo_export_metadata.csv')

    expected_columns = [
        'DebtHeld_DomesticNonBanks',
        'DebtHeld_CentralBank',
        'CBOFundingModeActive',
        'FundingMode',
        'CBOPrimaryDeficitFlow',
        'CBOControlledDebtTarget',
        'CBOControlledDebtPreIssuance',
        'CBOControlledDebtPostIssuance',
        'CBOControlledDebtTargetError',
        'CBORequiredFaceIssuance',
        'CBOOperatingCashTarget',
        'CBOCashResidual',
        'CBOCashReconciliationResidual',
        'CBOFiscalIncidencePolicyPresent',
        'FiscalIncidenceStatus',
        'CBORemittanceCashEffect',
        'CBONetInterestDiagnostic',
        'CBOTotalDeficitDiagnostic',
        'NetInterestDiagnosticStatus',
        'CBONetInterestBridgeRows',
        'CBOCashResidualStatus',
    ]
    for frame in (per_scenario, combined, final_state):
        assert list(frame.columns[:2]) == ['Scenario', 'Date']
        for column in expected_columns:
            assert column in frame.columns
        assert 'DebtHeld_Private' not in frame.columns
        assert 'DebtHeld_CB' not in frame.columns

    assert len(combined) == 4
    assert len(final_state) == 2
    assert set(final_state['Scenario']) == {'baseline', 'policy-shift'}
    baseline_final = final_state[final_state['Scenario'] == 'baseline'].iloc[0]
    assert baseline_final['Date'] == '2026-09-30'
    assert baseline_final['DebtHeld_DomesticNonBanks'] == 210
    assert baseline_final['DebtHeld_CentralBank'] == 320
    assert baseline_final['CBOControlledDebtTarget'] == 1_125.0
    assert baseline_final['CBOCashResidualStatus'] == 'operating_cash_target_loaded'
    assert baseline_final['NetInterestDiagnosticStatus'] == 'cbo_reported_check_only'

    assert list(metadata['Scenario']) == ['baseline', 'policy-shift']
    assert metadata.loc[0, 'funding_rule_mode'] == 'cbo_public_debt_target'
    assert bool(metadata.loc[0, 'cbo_funding_mode_active']) is True
    assert metadata.loc[0, 'cbo_net_interest_bridge_rows'] == 3
    assert metadata.loc[0, 'mmf_deposit_pass_through_sensitivity_grid'] == '[0.9, 0.97, 1.0]'
    assert 'nested_debug_payload' not in metadata.columns


def test_save_group_results_csv_uses_plot_filename_then_group_name(tmp_path, monkeypatch):
    """The fallback stem keeps chart CSV names aligned when no result filename is configured."""
    results = {'baseline': _scenario_frame(1)}

    save_group_results_csv(
        results,
        ['baseline'],
        {'save_plot_filename': str(tmp_path / 'plot output.png')},
        'Fallback Group',
    )
    assert (tmp_path / 'plot_output_results.csv').exists()
    assert (tmp_path / 'plot_output_final_state.csv').exists()

    monkeypatch.chdir(tmp_path)
    save_group_results_csv(
        results,
        ['baseline'],
        {},
        'Fallback Group',
    )
    assert Path('Fallback_Group_results.csv').exists()
    assert Path('Fallback_Group_final_state.csv').exists()


def test_save_group_results_csv_empty_data_warns_without_summary_files(tmp_path, capsys):
    """No-data groups should fail soft without creating empty downstream chart inputs."""
    save_group_results_csv(
        {
            'missing': None,
            'empty': pd.DataFrame(),
        },
        ['missing', 'empty'],
        {'save_results_filename': str(tmp_path / 'empty export.csv')},
        'Empty Group',
    )

    captured = capsys.readouterr()
    assert 'No successful scenario data available for CSV export' in captured.out
    assert not (tmp_path / 'empty_export_results.csv').exists()
    assert not (tmp_path / 'empty_export_final_state.csv').exists()


def test_save_group_results_csv_cleans_scenario_names_through_public_export(tmp_path, monkeypatch):
    """Sanitized scenario stems keep chart-export file paths portable across platforms."""
    output_file = tmp_path / 'edge export.csv'
    results = {
        ' special / chars! ': _scenario_frame(1),
        '!!!': _scenario_frame(2),
    }
    written_names = []
    original_to_csv = pd.DataFrame.to_csv

    def recording_to_csv(self, path, *args, **kwargs):
        written_names.append(Path(path).name)
        return original_to_csv(self, path, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'to_csv', recording_to_csv)

    save_group_results_csv(
        results,
        [' special / chars! ', '!!!'],
        {'save_results_filename': str(output_file)},
        'Edge Group',
    )

    assert (tmp_path / 'edge_export_special_chars.csv').exists()
    assert (tmp_path / 'edge_export_results.csv').exists()
    assert (tmp_path / 'edge_export_final_state.csv').exists()
    assert written_names.count('edge_export_results.csv') == 2

    combined = pd.read_csv(tmp_path / 'edge_export_results.csv')
    assert set(combined['Scenario']) == {' special / chars! ', '!!!'}
