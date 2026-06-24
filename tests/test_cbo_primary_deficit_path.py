import pytest

from forecast_bundle_builders import build_primary_deficit_path_rows
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_primary_deficit_path
from simulation_calendar import build_simulation_calendar


def test_primary_deficit_path_allocates_remaining_current_fy_by_actual_days(tmp_path):
    periods = build_simulation_calendar("2026-06-30", "2026-09-30", "monthly")
    rows = build_primary_deficit_path_rows(
        scenario_id="baseline",
        periods=periods,
        primary_deficit_by_fiscal_year_bil={2026: 623.727},
    )

    assert sum(row["day_count_weight"] for row in rows) == pytest.approx(1.0)
    assert sum(row["primary_deficit_bil"] for row in rows) == pytest.approx(623.727, abs=0.000001)
    assert {row["allocation_method"] for row in rows} == {"actual_day_weighted_equal_by_day"}
    assert {row["source_fiscal_year"] for row in rows} == {2026}

    path = tmp_path / "tdcsim_primary_deficit_path.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_primary_deficit_path(path)
    assert loaded["primary_deficit_bil"].sum() == pytest.approx(623.727, abs=0.000001)


def test_primary_deficit_path_allocates_multiple_fiscal_years_independently():
    periods = build_simulation_calendar("2026-09-01", "2027-10-31", "monthly")
    rows = build_primary_deficit_path_rows(
        scenario_id="baseline",
        periods=periods,
        primary_deficit_by_fiscal_year_bil={2026: 100.0, 2027: 1200.0, 2028: 200.0},
    )

    totals: dict[int, float] = {}
    for row in rows:
        totals[row["source_fiscal_year"]] = totals.get(row["source_fiscal_year"], 0.0) + row["primary_deficit_bil"]

    assert totals[2026] == pytest.approx(100.0, abs=0.000001)
    assert totals[2027] == pytest.approx(1200.0, abs=0.000001)
    assert totals[2028] == pytest.approx(200.0, abs=0.000001)
