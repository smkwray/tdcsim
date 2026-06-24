from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecast_paths import (
    BASELINE_INPUT_PATH_KEYS,
    FORECAST_CSV_SCHEMAS,
    load_cbo_fiscal_baseline,
    load_forecast_csv,
    load_holder_absorption_path,
    load_macro_forecast_path,
    resolve_baseline_input_paths,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _row_for(schema_key: str, **overrides: object) -> dict[str, object]:
    row = {column: "" for column in FORECAST_CSV_SCHEMAS[schema_key].required_columns}
    row.update(
        {
            "schema_version": "1",
            "scenario_id": "baseline",
            "source_role": "scenario_assumption",
            "runtime_role": "hard_target",
            "source_status": "fixture",
            "claim_boundary": "loader_only",
            "available_date": "2026-04-30",
            "observation_date": "2026-04-30",
        }
    )
    row.update(overrides)
    return row


def test_resolve_baseline_input_paths_uses_generic_keys_only(tmp_path: Path) -> None:
    paths = {
        key: f"data/forecast_inputs/{key}.csv"
        for key in BASELINE_INPUT_PATH_KEYS
    }

    resolved = resolve_baseline_input_paths({"baseline_input_paths": paths}, base_dir=tmp_path)

    assert set(resolved.as_dict()) == set(BASELINE_INPUT_PATH_KEYS)
    assert resolved.cbo_fiscal_baseline_file == tmp_path / "data/forecast_inputs/cbo_fiscal_baseline_file.csv"
    assert "ratewall_input_paths" not in resolved.as_dict()


def test_resolve_baseline_input_paths_rejects_unknown_key(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_baseline_input_paths(
            {"baseline_input_paths": {"ratewall_input_paths": "not-generic.csv"}},
            base_dir=tmp_path,
        )


def test_fiscal_baseline_loader_accepts_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "tdcsim_cbo_fiscal_baseline.csv"
    _write_csv(
        path,
        [
            _row_for(
                "cbo_fiscal_baseline_file",
                fiscal_year=2026,
                primary_deficit_bil=813.727,
                cbo_net_interest_bil=1038.976,
                cbo_total_deficit_bil=1852.703,
                source_role="hard_input",
                runtime_role="hard_flow",
            )
        ],
    )

    frame = load_cbo_fiscal_baseline(path, actuals_available_as_of="2026-04-30")

    assert frame.loc[0, "primary_deficit_bil"] == pytest.approx(813.727)


def test_loader_fails_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="cbo fiscal baseline file is missing"):
        load_cbo_fiscal_baseline(tmp_path / "missing.csv")


def test_loader_fails_on_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="file is empty"):
        load_cbo_fiscal_baseline(path)


def test_loader_fails_on_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, [{"scenario_id": "baseline", "fiscal_year": 2026}])

    with pytest.raises(ValueError, match="missing columns"):
        load_cbo_fiscal_baseline(path)


def test_fiscal_baseline_duplicate_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "dupes.csv"
    row = _row_for("cbo_fiscal_baseline_file", fiscal_year=2026)
    _write_csv(path, [row, dict(row)])

    with pytest.raises(ValueError, match="duplicate key rows"):
        load_cbo_fiscal_baseline(path)


def test_macro_forecast_duplicate_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "macro.csv"
    row = _row_for(
        "macro_forecast_path_file",
        period_start="2026-04-01",
        period_end="2026-06-30",
        cbo_3m_tbill_rate_pct=3.5,
        cbo_10y_treasury_rate_pct=4.2,
        cbo_cpi_u_index=330.0,
        cbo_cpi_u_inflation_pct=2.0,
    )
    _write_csv(path, [row, dict(row)])

    with pytest.raises(ValueError, match="duplicate key rows"):
        load_macro_forecast_path(path)


def test_source_and_runtime_roles_are_validated(tmp_path: Path) -> None:
    path = tmp_path / "bad_roles.csv"
    _write_csv(
        path,
        [
            _row_for(
                "cbo_fiscal_baseline_file",
                fiscal_year=2026,
                source_role="not_a_source_role",
                runtime_role="hard_flow",
            )
        ],
    )

    with pytest.raises(ValueError, match="invalid source_role"):
        load_cbo_fiscal_baseline(path)

    _write_csv(
        path,
        [
            _row_for(
                "cbo_fiscal_baseline_file",
                fiscal_year=2026,
                source_role="hard_input",
                runtime_role="not_a_runtime_role",
            )
        ],
    )

    with pytest.raises(ValueError, match="invalid runtime_role"):
        load_cbo_fiscal_baseline(path)


def test_available_date_after_cutoff_fails_without_lookahead(tmp_path: Path) -> None:
    path = tmp_path / "future.csv"
    _write_csv(
        path,
        [
            _row_for(
                "macro_forecast_path_file",
                period_start="2026-04-01",
                period_end="2026-06-30",
                available_date="2026-05-01",
            )
        ],
    )

    with pytest.raises(ValueError, match="available_date after actuals_available_as_of"):
        load_macro_forecast_path(path, actuals_available_as_of="2026-04-30", allow_lookahead=False)

    frame = load_macro_forecast_path(path, actuals_available_as_of="2026-04-30", allow_lookahead=True)
    assert len(frame) == 1


def test_loader_fails_on_invalid_date_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad_date.csv"
    _write_csv(
        path,
        [
            _row_for(
                "macro_forecast_path_file",
                period_start="not-a-date",
                period_end="2026-06-30",
            )
        ],
    )

    with pytest.raises(ValueError, match="invalid period_start dates"):
        load_macro_forecast_path(path)


def test_loader_fails_on_invalid_numeric_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad_numeric.csv"
    _write_csv(path, [_row_for("yield_curve_surface_file", curve_date="2026-04-01", tenor_years="ten")])

    with pytest.raises(ValueError, match="invalid numeric tenor_years values"):
        load_forecast_csv(path, schema_key="yield_curve_surface_file")


def test_loader_fails_on_invalid_boolean_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad_boolean.csv"
    _write_csv(
        path,
        [
            _row_for(
                "cash_reconciliation_residual_file",
                period_start="2026-04-01",
                period_end="2026-04-30",
                affects_operating_cash="maybe",
            )
        ],
    )

    with pytest.raises(ValueError, match="invalid boolean affects_operating_cash values"):
        load_forecast_csv(path, schema_key="cash_reconciliation_residual_file")


@pytest.mark.parametrize(
    ("schema_key", "overrides"),
    [
        (
            "source_fixture_file",
            {
                "forecast_name": "cbo_2026_02_baseline",
                "source_file": "51118-2026-02-Budget-Projections.xlsx",
                "source_sheet": "Table 1-1",
                "source_row_selector": "Primary deficit (-)",
                "source_year_or_period": "2026",
            },
        ),
        ("current_fy_splice_file", {"fiscal_year": 2026, "simulation_start_date": "2026-04-30"}),
        ("debt_stock_path_file", {"period_end": "2026-04-30"}),
        ("primary_deficit_path_file", {"period_start": "2026-04-30", "period_end": "2026-05-07"}),
        ("operating_cash_path_file", {"period_end": "2026-04-30"}),
        ("cash_reconciliation_residual_file", {"period_start": "2026-04-30", "period_end": "2026-05-07"}),
        ("macro_forecast_path_file", {"period_start": "2026-04-01", "period_end": "2026-06-30"}),
        ("yield_curve_surface_file", {"curve_date": "2026-04-01", "tenor_years": 10}),
        ("tips_cpi_path_file", {"month": "2026-04-01"}),
        ("tips_real_yield_path_file", {"curve_date": "2026-04-01", "tenor_years": 10}),
        ("fiscal_incidence_policy_file", {"policy_id": "central"}),
        ("net_interest_bridge_file", {"fiscal_year": 2026, "component_key": "coupon_interest"}),
    ],
)
def test_phase1_loaders_validate_all_documented_schema_columns(
    tmp_path: Path,
    schema_key: str,
    overrides: dict[str, object],
) -> None:
    path = tmp_path / f"{schema_key}.csv"
    _write_csv(path, [_row_for(schema_key, **overrides)])

    frame = load_forecast_csv(path, schema_key=schema_key)

    assert len(frame) == 1


def test_holder_absorption_loader_uses_generic_path_contract(tmp_path: Path) -> None:
    path = tmp_path / "tdcsim_holder_absorption_path.csv"
    _write_csv(
        path,
        [
            {
                "scenario_id": "baseline",
                "quarter": "2026Q2",
                "holder_type": "Private",
                "holder_subbucket": "domestic_nonbank",
                "bills_pct": 0.5,
                "source_status": "fixture",
                "claim_boundary": "loader_only",
            }
        ],
    )

    frame = load_holder_absorption_path(path)

    assert frame.loc[0, "bills_pct"] == pytest.approx(0.5)
