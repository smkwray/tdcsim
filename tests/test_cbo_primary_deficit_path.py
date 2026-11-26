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
    assert {row["allocation_method"] for row in rows} == {"equal_by_fedwire_business_day"}
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


def test_primary_deficit_moves_only_on_fedwire_business_days():
    """Treasury cash moves only when Fedwire is open: a weekend or holiday books nothing,
    every business day of a fiscal year books the same amount, and the year's total is
    unchanged. September 30, 2028 is a Saturday, so FY2028's flows end on Friday the
    29th and the weekend after it books nothing in either year."""

    from simulation_calendar import is_fedwire_business_day

    periods = build_simulation_calendar("2028-09-01", "2028-10-31", "daily")
    rows = build_primary_deficit_path_rows(
        scenario_id="baseline",
        periods=periods,
        primary_deficit_by_fiscal_year_bil={2028: 210.0, 2029: 2300.0},
    )
    by_day = {row["period_end"]: row["primary_deficit_bil"] for row in rows}
    closed = [day for day in by_day if not is_fedwire_business_day(day)]
    assert "2028-09-30" in closed and "2028-10-01" in closed and "2028-10-09" in closed
    assert all(by_day[day] == 0.0 for day in closed)
    fy2028 = [value for day, value in by_day.items() if day <= "2028-09-30" and is_fedwire_business_day(day)]
    assert len(fy2028) == 19 and max(fy2028) == pytest.approx(min(fy2028))  # Labor Day closed
    assert sum(fy2028) == pytest.approx(210.0, abs=1e-9)
    assert by_day["2028-09-29"] == pytest.approx(210.0 / 19)
