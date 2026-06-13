"""Scenario-group output contract tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sim_groups import save_group_results_csv


def _scenario_frame(start_value: int) -> pd.DataFrame:
    """Return a tiny time-series frame matching chart-export inputs."""
    return pd.DataFrame(
        {
            'TDC_Level': [start_value, start_value + 1, start_value + 2],
            'Reserves': [100 + start_value, 101 + start_value, 102 + start_value],
        },
        index=pd.to_datetime(['2025-01-01', '2025-01-08', '2025-01-15']),
    )


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
