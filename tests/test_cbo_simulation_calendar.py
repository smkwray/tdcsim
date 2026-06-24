from __future__ import annotations

from datetime import date

import pytest

from simulation_calendar import build_simulation_calendar


def test_weekly_calendar_never_moves_configured_start_backward() -> None:
    periods = build_simulation_calendar("2026-02-18", "2026-03-11", "weekly")

    assert periods[0].period_start == date(2026, 2, 18)
    assert periods[0].period_end == date(2026, 2, 25)
    assert all(period.period_start >= date(2026, 2, 18) for period in periods)


def test_september_30_control_date_is_unioned_into_weekly_grid() -> None:
    periods = build_simulation_calendar("2026-09-20", "2026-10-11", "weekly")

    control_periods = [period for period in periods if period.is_control_date]
    assert [period.period_end for period in control_periods] == [date(2026, 9, 30)]
    assert date(2026, 9, 27) in {period.period_end for period in periods}
    assert date(2026, 10, 4) in {period.period_end for period in periods}


def test_monthly_calendar_handles_partial_first_and_final_period_day_weights() -> None:
    periods = build_simulation_calendar("2026-02-15", "2026-04-10", "monthly")

    assert [(period.period_start, period.period_end, period.day_count) for period in periods] == [
        (date(2026, 2, 15), date(2026, 2, 28), 13),
        (date(2026, 2, 28), date(2026, 3, 31), 31),
        (date(2026, 3, 31), date(2026, 4, 10), 10),
    ]
    assert sum(period.day_count for period in periods) == 54
    assert sum(period.day_count_weight for period in periods) == pytest.approx(1.0)
    assert periods[0].day_count_weight == pytest.approx(13 / 54)
    assert periods[-1].day_count_weight == pytest.approx(10 / 54)
    assert [period.is_partial_period for period in periods] == [True, False, True]


def test_quarterly_calendar_includes_fiscal_year_control_inside_regular_quarter() -> None:
    periods = build_simulation_calendar("2026-08-15", "2026-12-31", "quarterly")

    assert date(2026, 9, 30) in {period.period_end for period in periods}
    assert [period.period_end for period in periods if period.is_control_date] == [date(2026, 9, 30)]
