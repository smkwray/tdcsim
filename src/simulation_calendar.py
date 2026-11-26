"""Pure simulation calendar helpers for forecast-driven runs."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Literal

CalendarFrequency = Literal["daily", "weekly", "monthly", "quarterly"]


@dataclass(frozen=True)
class SimulationPeriod:
    """A half-open simulation interval ending on a target/control boundary."""

    period_start: date
    period_end: date
    frequency: CalendarFrequency
    day_count: int
    day_count_weight: float
    is_control_date: bool
    is_partial_period: bool


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"expected date-like value, got {type(value).__name__}")


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _last_day_of_month(year, month))
    return date(year, month, day)


def _month_end(value: date) -> date:
    return date(value.year, value.month, _last_day_of_month(value.year, value.month))


def _quarter_end(value: date) -> date:
    quarter_end_month = ((value.month - 1) // 3 + 1) * 3
    return date(value.year, quarter_end_month, _last_day_of_month(value.year, quarter_end_month))


def _regular_boundaries(start: date, end: date, frequency: CalendarFrequency) -> list[date]:
    if frequency == "daily":
        step = timedelta(days=1)
        boundaries: list[date] = []
        current = start + step
        while current < end:
            boundaries.append(current)
            current += step
        return boundaries

    if frequency == "weekly":
        boundaries = []
        current = start + timedelta(days=7)
        while current < end:
            boundaries.append(current)
            current += timedelta(days=7)
        return boundaries

    if frequency == "monthly":
        boundaries = []
        current = _month_end(start)
        while current < end:
            if current > start:
                boundaries.append(current)
            current = _month_end(_add_months(current, 1))
        return boundaries

    if frequency == "quarterly":
        boundaries = []
        current = _quarter_end(start)
        while current < end:
            if current > start:
                boundaries.append(current)
            current = _quarter_end(_add_months(current, 3))
        return boundaries

    raise ValueError(f"unsupported calendar frequency: {frequency!r}")


def _is_regular_interval(period_start: date, period_end: date, frequency: CalendarFrequency) -> bool:
    if frequency == "daily":
        return (period_end - period_start).days == 1
    if frequency == "weekly":
        return (period_end - period_start).days == 7
    if frequency == "monthly":
        return period_end == _month_end(period_end) and period_start == _month_end(_add_months(period_end, -1))
    if frequency == "quarterly":
        return period_end == _quarter_end(period_end) and period_start == _quarter_end(_add_months(period_end, -3))
    raise ValueError(f"unsupported calendar frequency: {frequency!r}")


_FEDWIRE_CLOSED_DATES: frozenset[date] | None = None


def _fedwire_closed_dates() -> frozenset[date]:
    """Weekday federal holidays on which Fedwire is closed, 1900-2100.

    The Federal Reserve observes a fixed-date holiday that falls on a Sunday on the
    Monday, but stays open on the Friday before a Saturday holiday (it settled Treasury
    issues on 2021-12-31, the Friday before New Year's Day 2022).
    """

    global _FEDWIRE_CLOSED_DATES
    if _FEDWIRE_CLOSED_DATES is None:
        from pandas import Timestamp
        from pandas.tseries.holiday import USFederalHolidayCalendar

        closed: set[date] = set()
        for rule in USFederalHolidayCalendar().rules:
            if rule.offset is None and rule.day is not None:
                for year in range(1900, 2101):
                    actual = date(year, rule.month, rule.day)
                    if rule.start_date is not None and Timestamp(actual) < rule.start_date:
                        continue
                    if actual.weekday() == 6:
                        closed.add(actual + timedelta(days=1))
                    elif actual.weekday() < 5:
                        closed.add(actual)
            else:
                closed.update(value.date() for value in rule.dates("1900-01-01", "2100-12-31"))
        _FEDWIRE_CLOSED_DATES = frozenset(closed)
    return _FEDWIRE_CLOSED_DATES


def is_fedwire_business_day(value: date | datetime | str) -> bool:
    """True on days Fedwire settles: weekdays that are not federal holidays it observes."""

    day = _as_date(value)
    return day.weekday() < 5 and day not in _fedwire_closed_dates()


def next_fedwire_business_day(value: date | datetime | str) -> date:
    """The day itself if Fedwire settles on it, else the next day it does."""

    day = _as_date(value)
    while not is_fedwire_business_day(day):
        day += timedelta(days=1)
    return day


def previous_fedwire_business_day(value: date | datetime | str) -> date:
    """The day itself if Fedwire settles on it, else the last day before it that does."""

    day = _as_date(value)
    while not is_fedwire_business_day(day):
        day -= timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Gregorian Easter (anonymous Gregorian algorithm)."""

    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    weekday_offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * weekday_offset) // 451
    month, day = divmod(h + weekday_offset - 7 * m + 114, 31)
    return date(year, month, day + 1)


def is_securities_business_day(value: date | datetime | str) -> bool:
    """True on days Treasury securities trade and new issues settle.

    The securities market closes on every day Fedwire does, and also on Good Friday
    (except when it is the first Friday of its month, the employment report's day, when
    the market only closes early) and on the Friday before a Saturday federal holiday
    other than New Year's Day. Treasury settled the March 2018 and March 2024 TIPS on
    the Thursday before Good Friday and the March 2018 FRN on the Monday after it, while
    it settled on Friday 2021-12-31, the day before New Year's Day 2022.
    """

    day = _as_date(value)
    if not is_fedwire_business_day(day):
        return False
    if day == _easter_sunday(day.year) - timedelta(days=2) and day.day > 7:
        return False
    saturday = day + timedelta(days=1)
    return not (
        day.weekday() == 4
        and saturday in _saturday_federal_holidays()
        and (saturday.month, saturday.day) != (1, 1)
    )


_SATURDAY_FEDERAL_HOLIDAYS: frozenset[date] | None = None


def _saturday_federal_holidays() -> frozenset[date]:
    global _SATURDAY_FEDERAL_HOLIDAYS
    if _SATURDAY_FEDERAL_HOLIDAYS is None:
        from pandas import Timestamp
        from pandas.tseries.holiday import USFederalHolidayCalendar

        days: set[date] = set()
        for rule in USFederalHolidayCalendar().rules:
            if rule.offset is None and rule.day is not None:
                for year in range(1900, 2101):
                    actual = date(year, rule.month, rule.day)
                    if rule.start_date is not None and Timestamp(actual) < rule.start_date:
                        continue
                    if actual.weekday() == 5:
                        days.add(actual)
        _SATURDAY_FEDERAL_HOLIDAYS = frozenset(days)
    return _SATURDAY_FEDERAL_HOLIDAYS


def next_securities_business_day(value: date | datetime | str) -> date:
    """The day itself if Treasury securities settle on it, else the next day they do."""

    day = _as_date(value)
    while not is_securities_business_day(day):
        day += timedelta(days=1)
    return day


def previous_securities_business_day(value: date | datetime | str) -> date:
    """The day itself if Treasury securities settle on it, else the last day before that does."""

    day = _as_date(value)
    while not is_securities_business_day(day):
        day -= timedelta(days=1)
    return day


def add_fedwire_business_days(value: date | datetime | str, count: int) -> date:
    """Move ``count`` Fedwire business days forward (negative: backward) from ``value``."""

    day = _as_date(value)
    step = timedelta(days=1 if count >= 0 else -1)
    remaining = abs(int(count))
    while remaining > 0:
        day += step
        if is_fedwire_business_day(day):
            remaining -= 1
    return day


def september_30_control_dates(start_date: date | datetime | str, end_date: date | datetime | str) -> list[date]:
    """Return September 30 control boundaries inside the configured date span."""

    start = _as_date(start_date)
    end = _as_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    return [
        date(year, 9, 30)
        for year in range(start.year, end.year + 1)
        if start <= date(year, 9, 30) <= end
    ]


def build_simulation_calendar(
    start_date: date | datetime | str,
    end_date: date | datetime | str,
    frequency: CalendarFrequency,
    *,
    include_fiscal_year_controls: bool = True,
    extra_control_dates: Iterable[date | datetime | str] = (),
) -> list[SimulationPeriod]:
    """Build half-open periods whose first start is exactly ``start_date``.

    Regular boundaries are unioned with September 30 and caller-supplied control
    dates. The helper never shifts the configured start backward to align to a
    week, month, or quarter boundary.
    """

    start = _as_date(start_date)
    end = _as_date(end_date)
    if end <= start:
        raise ValueError("end_date must be after start_date")

    control_dates = {_as_date(value) for value in extra_control_dates}
    if include_fiscal_year_controls:
        control_dates.update(september_30_control_dates(start, end))
    out_of_span = sorted(value for value in control_dates if value < start or value > end)
    if out_of_span:
        raise ValueError(f"control dates outside simulation span: {out_of_span}")

    regular = set(_regular_boundaries(start, end, frequency))
    boundaries = [start, *sorted(regular | control_dates | {end})]
    if boundaries[0] != start:
        raise AssertionError("internal calendar error: start boundary shifted")

    total_days = (end - start).days
    periods: list[SimulationPeriod] = []
    for period_start, period_end in zip(boundaries, boundaries[1:]):
        day_count = (period_end - period_start).days
        if day_count <= 0:
            continue
        periods.append(
            SimulationPeriod(
                period_start=period_start,
                period_end=period_end,
                frequency=frequency,
                day_count=day_count,
                day_count_weight=day_count / total_days,
                is_control_date=period_end in control_dates,
                is_partial_period=not _is_regular_interval(period_start, period_end, frequency),
            )
        )
    return periods
