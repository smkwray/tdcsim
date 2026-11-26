"""Treasury issuance terms, auction calibration, and the bill-share issuance rule.

One registry defines the securities the CBO lane issues:

* bills by term in weeks; a bill matures exactly ``7 * weeks`` days after issue and its
  original term in years is ``days / 365.25``, the convention of the opening book;
* notes, bonds, TIPS and FRNs by original tenor in whole years.

The calibration reads Treasury's auction results (fiscaldata "Treasury Securities
Auctions Data") for the twelve months before the simulation start and measures how the
public, non-Fed part of gross auction volume splits across bill terms and across coupon
tenors. The issuance rule then sizes each day's bill issuance to move the outstanding
bill share toward a target and splits the rest across coupon securities by the measured
shares.
"""

from __future__ import annotations

import bisect
import calendar as calendar_module
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation_calendar import next_securities_business_day, previous_securities_business_day
from tdc_shared import DAYS_PER_YEAR_ACTUAL, INTRAGOV_HOLDERS, TGA_FLOOR_TOLERANCE

ISSUANCE_RULE_MODE = "bill_share_target_v1"
CALIBRATION_SCHEMA_VERSION = "tdcsim_issuance_calibration_v2"
CALIBRATION_FILE = "tdcsim_issuance_calibration.json"

BILL_TERM_WEEKS = (4, 6, 8, 13, 17, 26, 52)
# Coupon kinds map to the engine security type; tenors are original tenors in years.
COUPON_KIND_SECURITY_TYPE = {"nominal": "Fixed", "tips": "TIPS", "frn": "FRN"}
COUPON_TENORS = {
    "nominal": (2, 3, 5, 7, 10, 20, 30),
    "tips": (5, 10, 30),
    "frn": (2,),
}
# A reopening is sold with a few months less than its original tenor (for example
# "29-Year 10-Month" for the 30-year bond); anything further from a registered tenor
# is not a reopening this registry knows and raises.
REOPENING_MAX_SHORTFALL_YEARS = 1.0
CONTROLLED_SECURITY_TYPES = ("Fixed", "TIPS", "FRN")
# Treasury's settlement calendar for the bill-share rule's securities, on the Fedwire
# securities business-day calendar (simulation_calendar.is_securities_business_day).
# Checked against
# the auction file's twelve months before the start by check_settlement_calendar().
#
# Bills settle on a weekday every ``every_weeks`` weeks counted from ``anchor_date``
# (every week when absent), on the next business day when that weekday is closed.
# A coupon security settles in the listed months, each settlement either a new issue
# or a reopening of the latest new issue; an entry may hold only ``from`` or ``through``
# a settlement month (``YYYY-MM``) when Treasury changed the convention. Its day rule is
# a day of the month or ``month_end`` (the next business day when closed),
# ``last_business_day`` (the business day on or before the month's last day),
# ``last_friday`` or ``friday_before_two_year_auction_week`` (each the next business
# day when closed; see two_year_auction_week_monday). A new issue is dated on ``dated`` of the month it belongs to (the 15th or the
# month end, whatever day it settles) and matures that many years later, on the month
# end for month-end dates; a buyer pays the interest accrued since the dated date or the
# last coupon. The books close on the last business day of each federal fiscal year:
# everything pending settles then, as reopenings, so fiscal-year-end stocks carry no
# pending cash. Packaged in the calibration and read back through settlement_rules()
# and fiscal_year_close_rules().
_ALL_MONTHS = list(range(1, 13))
_QUARTERLY_REFUNDING_MONTHS = [2, 5, 8, 11]
_REFUNDING_REOPENING_MONTHS = [1, 3, 4, 6, 7, 9, 10, 12]
SETTLEMENT_CALENDAR = {
    "fiscal_year_close": [{"month": 9, "day_of_month": "last_business_day"}],
    "bills": [
        {"weeks": 4, "weekday": "Tuesday"},
        {"weeks": 8, "weekday": "Tuesday"},
        {"weeks": 17, "weekday": "Tuesday"},
        {"weeks": 6, "weekday": "Thursday"},
        {"weeks": 13, "weekday": "Thursday"},
        {"weeks": 26, "weekday": "Thursday"},
        {"weeks": 52, "weekday": "Thursday", "every_weeks": 4, "anchor_date": "2026-06-11"},
    ],
    "coupons": [
        {
            "kind": "nominal", "tenor_years": 2, "dated": "month_end",
            "schedule": [{"issue": "new", "months": _ALL_MONTHS, "day": "month_end"}],
        },
        {
            "kind": "nominal", "tenor_years": 3, "dated": 15,
            "schedule": [{"issue": "new", "months": _ALL_MONTHS, "day": 15}],
        },
        {
            "kind": "nominal", "tenor_years": 5, "dated": "month_end",
            "schedule": [{"issue": "new", "months": _ALL_MONTHS, "day": "month_end"}],
        },
        {
            # Treasury's November 4, 2026 refunding statement makes the 7-year a quarterly
            # new issue reopened in the next two months, from the March 2027 new issue.
            "kind": "nominal", "tenor_years": 7, "dated": "month_end",
            "schedule": [
                {"issue": "new", "months": _ALL_MONTHS, "day": "month_end", "through": "2027-02"},
                {"issue": "new", "months": [3, 6, 9, 12], "day": "month_end", "from": "2027-03"},
                {"issue": "reopening", "months": [1, 2, 4, 5, 7, 8, 10, 11], "day": "month_end", "from": "2027-03"},
            ],
        },
        {
            "kind": "nominal", "tenor_years": 10, "dated": 15,
            "schedule": [
                {"issue": "new", "months": _QUARTERLY_REFUNDING_MONTHS, "day": 15},
                {"issue": "reopening", "months": _REFUNDING_REOPENING_MONTHS, "day": 15},
            ],
        },
        {
            # Treasury's May 6, 2026 refunding statement moved 20-year reopenings, from
            # the June 16, 2026 auction, to the Friday of the auction week (the next
            # business day when it is a holiday); new issues still settle at month end.
            "kind": "nominal", "tenor_years": 20, "dated": 15,
            "schedule": [
                {"issue": "new", "months": _QUARTERLY_REFUNDING_MONTHS, "day": "month_end"},
                {"issue": "reopening", "months": _REFUNDING_REOPENING_MONTHS, "day": "month_end", "through": "2026-05"},
                {
                    "issue": "reopening", "months": _REFUNDING_REOPENING_MONTHS,
                    "day": "friday_before_two_year_auction_week", "from": "2026-06",
                },
            ],
        },
        {
            "kind": "nominal", "tenor_years": 30, "dated": 15,
            "schedule": [
                {"issue": "new", "months": _QUARTERLY_REFUNDING_MONTHS, "day": 15},
                {"issue": "reopening", "months": _REFUNDING_REOPENING_MONTHS, "day": 15},
            ],
        },
        {
            "kind": "frn", "tenor_years": 2, "dated": "month_end",
            "schedule": [
                {"issue": "new", "months": [1, 4, 7, 10], "day": "month_end"},
                {"issue": "reopening", "months": [2, 3, 5, 6, 8, 9, 11, 12], "day": "last_friday"},
            ],
        },
        {
            "kind": "tips", "tenor_years": 5, "dated": 15,
            "schedule": [
                {"issue": "new", "months": [4, 10], "day": "last_business_day"},
                {"issue": "reopening", "months": [6, 12], "day": "last_business_day"},
            ],
        },
        {
            "kind": "tips", "tenor_years": 10, "dated": 15,
            "schedule": [
                {"issue": "new", "months": [1, 7], "day": "last_business_day"},
                {"issue": "reopening", "months": [3, 5, 9, 11], "day": "last_business_day"},
            ],
        },
        {
            "kind": "tips", "tenor_years": 30, "dated": 15,
            "schedule": [
                {"issue": "new", "months": [2], "day": "last_business_day"},
                {"issue": "reopening", "months": [8], "day": "last_business_day"},
            ],
        },
    ],
}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
BILL_CUTOFF_YEARS = 1.0

AUCTION_REQUIRED_COLUMNS = (
    "security_type",
    "security_term",
    "issue_date",
    "maturity_date",
    "total_accepted",
    "soma_accepted",
    "cash_management_bill_cmb",
    "floating_rate",
    "inflation_index_security",
)
# A weekly bill calendar means the source must reach within this many days of the
# window end; a staler file would silently calibrate on a partial window.
AUCTION_SOURCE_MAX_END_GAP_DAYS = 7

_TERM_PATTERN = re.compile(r"^(?:(\d+)-Year)?\s*(?:(\d+)-Month)?$")
_WEEK_PATTERN = re.compile(r"^(\d+)-Week$")


class IssuanceCalibrationError(ValueError):
    """Raised when auction data or an issuance rule cannot be used as declared."""


def bill_days(weeks: int) -> int:
    return 7 * int(weeks)


def bill_maturity(nominal: date, weeks: int) -> date:
    """A bill's maturity: its weekly cycle date (the nominal settlement day plus its term),
    or the next securities business day when that is a holiday, as Treasury dates them."""

    return next_securities_business_day(nominal + timedelta(days=bill_days(weeks)))


def bill_maturity_years(weeks: int) -> float:
    return bill_days(weeks) / DAYS_PER_YEAR_ACTUAL


# --- Auction calibration -------------------------------------------------------------


def _flag(row: Mapping[str, str], column: str, *, line: int) -> bool:
    value = str(row.get(column, "")).strip()
    if value == "Yes":
        return True
    if value == "No":
        return False
    raise IssuanceCalibrationError(
        f"auction row {line}: {column} must be Yes or No, got {value!r}"
    )


def _dollars(row: Mapping[str, str], column: str, *, line: int) -> float:
    text = str(row.get(column, "")).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise IssuanceCalibrationError(
            f"auction row {line}: {column} is not a number ({text!r})"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise IssuanceCalibrationError(
            f"auction row {line}: {column} must be finite and nonnegative"
        )
    return value


def _date(row: Mapping[str, str], column: str, *, line: int) -> date:
    text = str(row.get(column, "")).strip()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise IssuanceCalibrationError(
            f"auction row {line}: {column} is not an ISO date ({text!r})"
        ) from exc


def _bill_weeks(term: str, *, line: int) -> int:
    match = _WEEK_PATTERN.match(term)
    if match is None or int(match.group(1)) not in BILL_TERM_WEEKS:
        raise IssuanceCalibrationError(f"auction row {line}: unmapped bill term {term!r}")
    return int(match.group(1))


def _term_years(term: str, *, line: int) -> float:
    match = _TERM_PATTERN.match(term.strip())
    if match is None or not any(match.groups()):
        raise IssuanceCalibrationError(f"auction row {line}: unparsed coupon term {term!r}")
    return int(match.group(1) or 0) + int(match.group(2) or 0) / 12.0


def coupon_tenor(kind: str, term: str, *, line: int = 0) -> int:
    """Map an auction term (original or reopening) to its registered original tenor."""

    years = _term_years(term, line=line)
    for tenor in COUPON_TENORS[kind]:
        if years <= tenor + 1e-9:
            if tenor - years < REOPENING_MAX_SHORTFALL_YEARS - 1e-9:
                return tenor
            break
    raise IssuanceCalibrationError(
        f"auction row {line}: unmapped {kind} term {term!r}"
    )


def build_auction_calibration(
    auctions_csv: str | Path,
    *,
    simulation_start_date: str,
) -> dict[str, Any]:
    """Measure public gross auction shares over the twelve months before the start.

    Public amount is ``total_accepted - soma_accepted``: the CBO lane models the Fed's
    add-ons separately, as rollovers of its own maturing holdings, so they are not part of
    the public supply these shares split.
    Cash management bills are excluded. Raises on missing columns, malformed values,
    unmapped terms, an empty window, or a source that does not cover the window.
    """

    path = Path(auctions_csv)
    raw = path.read_bytes()
    start = date.fromisoformat(simulation_start_date)
    window_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).date()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in AUCTION_REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise IssuanceCalibrationError(f"auction source is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise IssuanceCalibrationError("auction source has no rows")

    bill_amounts = {weeks: 0.0 for weeks in BILL_TERM_WEEKS}
    bill_counts = {weeks: 0 for weeks in BILL_TERM_WEEKS}
    bill_days_matching = {weeks: 0 for weeks in BILL_TERM_WEEKS}
    coupon_amounts = {(kind, tenor): 0.0 for kind, tenors in COUPON_TENORS.items() for tenor in tenors}
    coupon_counts = {key: 0 for key in coupon_amounts}
    excluded_cmb = 0
    reopenings_mapped = 0
    soma_total = 0.0
    total_accepted = 0.0
    earliest_issue: date | None = None
    latest_in_window: date | None = None
    for line, row in enumerate(rows, start=2):
        issue = _date(row, "issue_date", line=line)
        earliest_issue = issue if earliest_issue is None else min(earliest_issue, issue)
        if not (window_start <= issue < start):
            continue
        security_type = str(row.get("security_type", "")).strip()
        if security_type not in {"Bill", "Note", "Bond"}:
            raise IssuanceCalibrationError(
                f"auction row {line}: unknown security_type {security_type!r}"
            )
        if _flag(row, "cash_management_bill_cmb", line=line):
            if security_type != "Bill":
                raise IssuanceCalibrationError(
                    f"auction row {line}: cash management flag on a {security_type}"
                )
            excluded_cmb += 1
            continue
        maturity = _date(row, "maturity_date", line=line)
        accepted = _dollars(row, "total_accepted", line=line)
        soma = _dollars(row, "soma_accepted", line=line)
        if soma > accepted:
            raise IssuanceCalibrationError(
                f"auction row {line}: soma_accepted exceeds total_accepted"
            )
        public_bil = (accepted - soma) / 1.0e9
        total_accepted += accepted / 1.0e9
        soma_total += soma / 1.0e9
        latest_in_window = issue if latest_in_window is None else max(latest_in_window, issue)
        term = str(row.get("security_term", "")).strip()
        floating = _flag(row, "floating_rate", line=line)
        inflation = _flag(row, "inflation_index_security", line=line)
        if security_type == "Bill":
            if floating or inflation:
                raise IssuanceCalibrationError(f"auction row {line}: bill flagged FRN or TIPS")
            weeks = _bill_weeks(term, line=line)
            bill_amounts[weeks] += public_bil
            bill_counts[weeks] += 1
            if (maturity - issue).days == bill_days(weeks):
                bill_days_matching[weeks] += 1
            continue
        if floating and inflation:
            raise IssuanceCalibrationError(f"auction row {line}: flagged both FRN and TIPS")
        kind = "frn" if floating else "tips" if inflation else "nominal"
        tenor = coupon_tenor(kind, term, line=line)
        if abs(_term_years(term, line=line) - tenor) > 1e-9:
            reopenings_mapped += 1
        coupon_amounts[(kind, tenor)] += public_bil
        coupon_counts[(kind, tenor)] += 1

    bill_total = sum(bill_amounts.values())
    coupon_total = sum(coupon_amounts.values())
    if latest_in_window is None or earliest_issue is None:
        raise IssuanceCalibrationError(
            f"auction source has no auctions issued in [{window_start}, {start})"
        )
    if earliest_issue > window_start:
        raise IssuanceCalibrationError(
            f"auction source starts {earliest_issue}, after the window start {window_start}"
        )
    if (start - latest_in_window).days > AUCTION_SOURCE_MAX_END_GAP_DAYS:
        raise IssuanceCalibrationError(
            f"auction source ends {latest_in_window}, more than "
            f"{AUCTION_SOURCE_MAX_END_GAP_DAYS} days before the window end {start}"
        )
    if bill_total <= TGA_FLOOR_TOLERANCE or coupon_total <= TGA_FLOOR_TOLERANCE:
        raise IssuanceCalibrationError("auction window has no public bill or coupon volume")
    for weeks in BILL_TERM_WEEKS:
        # Holiday shifts move a few auctions by a day; the modal term must still be the
        # registered day count, or the term label is being read wrongly.
        if bill_counts[weeks] and bill_days_matching[weeks] * 2 <= bill_counts[weeks]:
            raise IssuanceCalibrationError(
                f"{weeks}-week bills mostly do not mature {bill_days(weeks)} days after issue"
            )

    return {
        "window": {
            "issue_date_start_inclusive": window_start.isoformat(),
            "issue_date_end_exclusive": start.isoformat(),
            "latest_issue_date": latest_in_window.isoformat(),
        },
        "public_amount_definition": "total_accepted_minus_soma_accepted",
        "excluded": {"cash_management_bills": excluded_cmb},
        "amounts_bil": {
            "total_accepted": total_accepted,
            "soma_accepted": soma_total,
            "public_bills": bill_total,
            "public_coupons": coupon_total,
        },
        "public_bill_share_of_gross_volume": bill_total / (bill_total + coupon_total),
        "reopenings_mapped_to_original_tenor": reopenings_mapped,
        "bill_terms": [
            {
                "weeks": weeks,
                "days": bill_days(weeks),
                "share": bill_amounts[weeks] / bill_total,
                "public_amount_bil": bill_amounts[weeks],
                "auction_count": bill_counts[weeks],
                "auctions_maturing_on_registered_day": bill_days_matching[weeks],
            }
            for weeks in BILL_TERM_WEEKS
        ],
        "coupon_terms": [
            {
                "kind": kind,
                "tenor_years": tenor,
                "share": coupon_amounts[(kind, tenor)] / coupon_total,
                "public_amount_bil": coupon_amounts[(kind, tenor)],
                "auction_count": coupon_counts[(kind, tenor)],
            }
            for kind, tenors in COUPON_TENORS.items()
            for tenor in tenors
        ],
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
    }


def _full_calendar_profile(settlement_calendar: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "settlement_calendar": settlement_calendar,
        "bill_terms": [{"weeks": weeks} for weeks in BILL_TERM_WEEKS],
        "coupon_terms": [
            {"security_type": COUPON_KIND_SECURITY_TYPE[kind], "maturity_years": float(tenor)}
            for kind, tenors in COUPON_TENORS.items()
            for tenor in tenors
        ],
    }


def _cross_tenor_reopening(row: Mapping[str, str], *, line: int) -> bool:
    original = str(row.get("original_security_term", "")).strip()
    term = str(row.get("security_term", "")).strip()
    if original in {"", "null"}:
        raise IssuanceCalibrationError(f"auction row {line}: a reopening has no original_security_term")
    return _term_years(original, line=line) > _term_years(term, line=line) + REOPENING_MAX_SHORTFALL_YEARS


def check_settlement_calendar(
    auctions_csv: str | Path,
    *,
    simulation_start_date: str,
    settlement_calendar: Mapping[str, Any] = SETTLEMENT_CALENDAR,
) -> dict[str, Any]:
    """Check the calendar against Treasury's auctions in the twelve months before the start.

    Every non-cash-management auction issued in the window must fall on a calendar
    settlement of its security, as a new issue exactly when the calendar says new (a
    reopening carries an earlier original issue date), and every new coupon issue must
    mature on the calendar's dated date plus its tenor. A bill must mature on its weekly
    cycle (``bill_maturity``) and is a reopening exactly when an
    earlier bill matures the same day. Every calendar settlement in the window must have
    its auction. A security whose calendar uses the auction week at any time
    (``friday_before_two_year_auction_week``) must be auctioned in that week, whichever
    convention its settlements followed then, since the scenario years extrapolate it.
    Raises with the mismatches; returns the counts.
    """

    start = date.fromisoformat(simulation_start_date)
    window_start = (pd.Timestamp(start) - pd.DateOffset(years=1)).date()
    window_end = start - timedelta(days=1)
    profile = _full_calendar_profile(settlement_calendar)
    rules = settlement_rules(profile)
    expected = {(event.settle, event.key): event for event in settlement_events(profile, window_start, window_end)}
    with Path(auctions_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # A bill reopens any earlier bill maturing the same day; the file's whole history
    # tells which bills were already outstanding.
    first_bill_issue: dict[date, date] = {}
    for line, row in enumerate(rows, start=2):
        if str(row.get("security_type", "")).strip() == "Bill" and not _flag(row, "cash_management_bill_cmb", line=line):
            maturity = _date(row, "maturity_date", line=line)
            issue = _date(row, "issue_date", line=line)
            first_bill_issue[maturity] = min(issue, first_bill_issue.get(maturity, issue))
    auction_week_keys = {
        key for key, rule in rules.items()
        if any(entry.day == "friday_before_two_year_auction_week" for entry in rule.entries)
    }
    seen: set[tuple[date, tuple[str, float]]] = set()
    problems: list[str] = []
    counts = {"auctions": 0, "new_issues": 0, "reopenings": 0, "cross_tenor_reopenings": 0, "auction_weeks": 0}
    for line, row in enumerate(rows, start=2):
        issue = _date(row, "issue_date", line=line)
        if not (window_start <= issue < start) or _flag(row, "cash_management_bill_cmb", line=line):
            continue
        security_type = str(row.get("security_type", "")).strip()
        term = str(row.get("security_term", "")).strip()
        if security_type == "Bill":
            key = bill_supply_key(_bill_weeks(term, line=line))
        else:
            kind = (
                "frn" if _flag(row, "floating_rate", line=line)
                else "tips" if _flag(row, "inflation_index_security", line=line)
                else "nominal"
            )
            key = coupon_supply_key(COUPON_KIND_SECURITY_TYPE[kind], coupon_tenor(kind, term, line=line))
        if key in auction_week_keys:
            auction = _date(row, "auction_date", line=line)
            monday = two_year_auction_week_monday(auction.year, auction.month) - timedelta(days=7)
            if monday <= auction <= monday + timedelta(days=4):
                counts["auction_weeks"] += 1
            else:
                problems.append(f"{key} auctioned {auction}, not in the week of {monday} the calendar assumes")
        event = expected.get((issue, key))
        if event is None:
            problems.append(f"{key} settled {issue} off the calendar")
            continue
        if (issue, key) in seen:
            problems.append(f"{key} settled twice on {issue}")
            continue
        seen.add((issue, key))
        counts["auctions"] += 1
        original_text = str(row.get("original_issue_date", "")).strip()
        reopened = original_text not in {"", "null"} and _date(row, "original_issue_date", line=line) < issue
        if key[0] == "bill":
            maturity = _date(row, "maturity_date", line=line)
            calendar_maturity = bill_maturity(event.nominal, int(key[1]))
            if maturity != calendar_maturity:
                problems.append(f"{key} issued {issue} matures {maturity}, calendar says {calendar_maturity}")
                continue
            if reopened != (first_bill_issue[maturity] < issue):
                problems.append(f"{key} on {issue}: auction is {'a reopening' if reopened else 'new'}, the bills outstanding say otherwise")
                continue
            counts["reopenings" if reopened else "new_issues"] += 1
            continue
        if reopened and event.issue == "new" and _cross_tenor_reopening(row, line=line):
            # Treasury occasionally sells a new issue's slot as a reopening of an older,
            # longer note with the same maturity date (the January 2026 two-year reopened
            # a January 2023 five-year); the security sold has the calendar's terms.
            counts["cross_tenor_reopenings"] += 1
            continue
        if reopened != (event.issue == "reopening"):
            problems.append(f"{key} on {issue}: auction is {'a reopening' if reopened else 'new'}, calendar says {event.issue}")
            continue
        counts["reopenings" if reopened else "new_issues"] += 1
        if not reopened:
            maturity = _date(row, "maturity_date", line=line)
            calendar_maturity = maturity_from_dated(rules[key], dated_date(rules[key], event.nominal))
            if maturity != calendar_maturity:
                problems.append(f"{key} issued {issue} matures {maturity}, calendar says {calendar_maturity}")
    missing = sorted(set(expected) - seen)
    problems.extend(f"{key} has no auction settling {settle}" for settle, key in missing)
    if problems:
        raise IssuanceCalibrationError(
            f"settlement calendar disagrees with Treasury's auctions in [{window_start}, {start}): "
            + "; ".join(problems[:20])
            + (f"; and {len(problems) - 20} more" if len(problems) > 20 else "")
        )
    return {
        "window": {"issue_date_start_inclusive": window_start.isoformat(), "issue_date_end_exclusive": start.isoformat()},
        **counts,
        "calendar_settlements": len(expected),
        "status": "pass",
    }


# --- Controlled debt and the bill share ------------------------------------------------


def controlled_marketable_mask(frame: pd.DataFrame) -> pd.Series:
    """Active Fixed, TIPS and FRN held outside the intragovernmental holders."""

    return (
        frame["Status"].astype(str).eq("Active")
        & frame["SecurityType"].astype(str).isin(CONTROLLED_SECURITY_TYPES)
        & ~frame["HolderType"].astype(str).isin(INTRAGOV_HOLDERS)
    )


def bill_mask(frame: pd.DataFrame) -> pd.Series:
    """Bills are Fixed securities with an original term of at most one year."""

    original_years = pd.to_numeric(frame["OriginalMaturityYears"], errors="coerce")
    fixed = frame["SecurityType"].astype(str).eq("Fixed")
    unknown = fixed & original_years.isna()
    if unknown.any():
        raise IssuanceCalibrationError(
            f"{int(unknown.sum())} Fixed rows have no original maturity; cannot tell bills from coupons"
        )
    return fixed & (original_years <= BILL_CUTOFF_YEARS + TGA_FLOOR_TOLERANCE)


def debt_base(frame: pd.DataFrame) -> pd.Series:
    """Face value, with TIPS at adjusted principal."""

    face = pd.to_numeric(frame["FaceValue"], errors="coerce")
    adjusted = pd.to_numeric(frame["AdjustedPrincipal"], errors="coerce")
    tips = frame["SecurityType"].astype(str).eq("TIPS")
    missing = tips & adjusted.isna()
    if missing.any():
        raise IssuanceCalibrationError(
            f"{int(missing.sum())} TIPS rows have no adjusted principal; face would understate them"
        )
    values = np.where(tips, adjusted, face.fillna(0.0))
    return pd.Series(values, index=frame.index).fillna(0.0)


def controlled_debt_and_bills(frame: pd.DataFrame | None) -> tuple[float, float]:
    """Return controlled marketable debt and the bills within it."""

    if frame is None or frame.empty:
        return 0.0, 0.0
    rows = frame.loc[controlled_marketable_mask(frame)]
    if rows.empty:
        return 0.0, 0.0
    weights = debt_base(rows)
    return float(weights.sum()), float(weights.loc[bill_mask(rows)].sum())


# --- The rule --------------------------------------------------------------------------


def bill_issuance_amount(
    amount: float,
    *,
    target: float,
    controlled_debt: float,
    bill_face: float,
) -> float:
    """Bills needed to move the bill share toward ``target`` after issuing ``amount``.

    ``clamp(target * (D + A) - B, 0, A)``: all of ``A`` goes to bills while the share is
    below target by more than ``A`` can close, none while it is above target.
    """

    amount = float(amount)
    if amount <= 0.0:
        return 0.0
    gap = float(target) * (float(controlled_debt) + amount) - float(bill_face)
    return min(amount, max(0.0, gap))


def target_for_date(path: Sequence[tuple[pd.Timestamp, float]], when: Any) -> float:
    """Linear in calendar days between dated points, flat outside them."""

    if not path:
        raise IssuanceCalibrationError("bill-share target path is empty")
    moment = pd.Timestamp(when).normalize()
    dates = [point[0] for point in path]
    index = bisect.bisect_right(dates, moment)
    if index == 0:
        return float(path[0][1])
    if index == len(path):
        return float(path[-1][1])
    (left_date, left), (right_date, right) = path[index - 1], path[index]
    weight = (moment - left_date).days / (right_date - left_date).days
    return float(left) + weight * (float(right) - float(left))


def normalize_target_path(value: Any, *, label: str) -> list[dict[str, Any]]:
    """Return ``[{date, bill_share_target}, ...]`` from a constant or a dated path."""

    if isinstance(value, bool):
        raise IssuanceCalibrationError(f"{label} must be a number or a dated path")
    if isinstance(value, (int, float)):
        return [{"date": None, "bill_share_target": _share(value, label=label)}]
    if not isinstance(value, list) or not value:
        raise IssuanceCalibrationError(f"{label} must be a number or a nonempty dated path")
    points = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {"date", "bill_share_target"}:
            raise IssuanceCalibrationError(
                f"{label}[{index}] must have exactly date and bill_share_target"
            )
        try:
            when = date.fromisoformat(str(item["date"]))
        except ValueError as exc:
            raise IssuanceCalibrationError(f"{label}[{index}].date must be an ISO date") from exc
        points.append(
            {
                "date": when.isoformat(),
                "bill_share_target": _share(
                    item["bill_share_target"], label=f"{label}[{index}].bill_share_target"
                ),
            }
        )
    dates = [point["date"] for point in points]
    if dates != sorted(set(dates)):
        raise IssuanceCalibrationError(f"{label} dates must be strictly increasing")
    return points


def _share(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IssuanceCalibrationError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise IssuanceCalibrationError(f"{label} must be between 0 and 1")
    return number


def engine_rule_profile(
    *,
    target_path: Iterable[Mapping[str, Any]],
    bill_terms: Iterable[Mapping[str, Any]],
    coupon_terms: Iterable[Mapping[str, Any]],
    settlement_calendar: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the engine issuance profile for the bill-share rule.

    ``bills``/``notes`` cutoffs keep holder-preference categories (bills <= 1y, notes
    <= 10y, bonds > 10y) on the shared classifier.
    """

    path = []
    for point in target_path:
        when = point.get("date")
        path.append(
            {
                "date": None if when is None else str(when),
                "bill_share_target": _share(point["bill_share_target"], label="bill_share_target"),
            }
        )
    if not path:
        raise IssuanceCalibrationError("bill-share target path is empty")
    bills = [
        {"weeks": int(term["weeks"]), "share": float(term["share"])}
        for term in bill_terms
    ]
    coupons = [
        {
            "security_type": COUPON_KIND_SECURITY_TYPE[str(term["kind"])],
            "maturity_years": float(term["tenor_years"]),
            "share": float(term["share"]),
        }
        for term in coupon_terms
    ]
    profile = {
        "mode": ISSUANCE_RULE_MODE,
        "bill_share_target_path": path,
        "bill_terms": bills,
        "coupon_terms": coupons,
        "settlement_calendar": json.loads(json.dumps(settlement_calendar)),
        "bills": {"category_cutoff_years": BILL_CUTOFF_YEARS},
        "notes": {"category_cutoff_years": 10.0},
        "bonds": {"category_cutoff_years": 999.0},
        "NonMarketable": {"target_percentage": 0.0},
    }
    errors = validate_rule_profile(profile)
    if errors:
        raise IssuanceCalibrationError("; ".join(errors))
    return profile


def validate_rule_profile(profile: Mapping[str, Any], label: str = "treasury_issuance_profile") -> list[str]:
    errors: list[str] = []
    path = profile.get("bill_share_target_path")
    if not isinstance(path, list) or not path:
        errors.append(f"{label}.bill_share_target_path must be a nonempty list")
    else:
        dated = [point.get("date") for point in path]
        if len(path) > 1 and (None in dated or dated != sorted(set(dated))):
            errors.append(f"{label}.bill_share_target_path dates must be strictly increasing")
    for key in ("bill_terms", "coupon_terms"):
        terms = profile.get(key)
        if not isinstance(terms, list) or not terms:
            errors.append(f"{label}.{key} must be a nonempty list")
            continue
        total = 0.0
        for index, term in enumerate(terms):
            share = term.get("share") if isinstance(term, Mapping) else None
            if not isinstance(share, (int, float)) or not math.isfinite(share) or share < 0.0:
                errors.append(f"{label}.{key}[{index}].share must be finite and nonnegative")
                continue
            total += float(share)
        if abs(total - 1.0) > 1.0e-9:
            errors.append(f"{label}.{key} shares must sum to 1.0, got {total}")
    for index, term in enumerate(profile.get("bill_terms") or []):
        if isinstance(term, Mapping) and int(term.get("weeks", 0)) not in BILL_TERM_WEEKS:
            errors.append(f"{label}.bill_terms[{index}].weeks is not a registered bill term")
    allowed_coupons = {
        (COUPON_KIND_SECURITY_TYPE[kind], float(tenor))
        for kind, tenors in COUPON_TENORS.items()
        for tenor in tenors
    }
    for index, term in enumerate(profile.get("coupon_terms") or []):
        if not isinstance(term, Mapping):
            continue
        key = (str(term.get("security_type")), float(term.get("maturity_years", 0.0)))
        if key not in allowed_coupons:
            errors.append(f"{label}.coupon_terms[{index}] is not a registered coupon security")
    if not errors:
        try:
            settlement_rules(profile)
        except IssuanceCalibrationError as exc:
            errors.append(f"{label}.settlement_calendar: {exc}")
    return errors


COUPON_DAY_RULES = ("month_end", "last_business_day", "last_friday", "friday_before_two_year_auction_week")
ISSUE_KINDS = ("new", "reopening")


@dataclass(frozen=True)
class SettlementEntry:
    """Months in which a coupon security settles, the day rule, and new or reopening,
    held from ``first`` through ``last`` (year, month) when the convention changed."""

    months: frozenset[int]
    day: int | str
    issue: str
    first: tuple[int, int] | None = None
    last: tuple[int, int] | None = None

    def holds(self, year: int, month: int) -> bool:
        return (
            month in self.months
            and (self.first is None or (year, month) >= self.first)
            and (self.last is None or (year, month) <= self.last)
        )


@dataclass(frozen=True)
class SecurityCalendar:
    """One security's settlement calendar: a bill's weekday cycle or a coupon's months."""

    key: tuple[str, float]
    weekday: int | None = None
    every_weeks: int = 1
    anchor: date | None = None
    dated: int | str | None = None
    entries: tuple[SettlementEntry, ...] = ()

    @property
    def is_bill(self) -> bool:
        return self.weekday is not None


@dataclass(frozen=True)
class SettlementEvent:
    """A calendar settlement: ``nominal`` is the unadjusted date, ``settle`` the business day."""

    settle: date
    key: tuple[str, float]
    issue: str
    nominal: date


def fiscal_year_close_rules(profile: Mapping[str, Any]) -> list[int]:
    """Months whose last business day closes the books: everything pending settles."""

    entries = profile["settlement_calendar"].get("fiscal_year_close")
    if not isinstance(entries, list) or not entries:
        raise IssuanceCalibrationError("settlement_calendar fiscal_year_close must be a nonempty list")
    months = []
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"month", "day_of_month"}
            or entry["day_of_month"] != "last_business_day"
        ):
            raise IssuanceCalibrationError(f"fiscal_year_close entry is malformed: {entry!r}")
        month = entry["month"]
        if not isinstance(month, int) or isinstance(month, bool) or not 1 <= month <= 12:
            raise IssuanceCalibrationError(f"fiscal_year_close month is malformed: {month!r}")
        months.append(month)
    return months


def bill_supply_key(weeks: int) -> tuple[str, float]:
    return ("bill", float(weeks))


def coupon_supply_key(security_type: str, tenor_years: float) -> tuple[str, float]:
    return (str(security_type), float(tenor_years))


def _months(value: Any, *, label: str) -> frozenset[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(m, int) or isinstance(m, bool) or not 1 <= m <= 12 for m in value)
        or len(set(value)) != len(value)
    ):
        raise IssuanceCalibrationError(f"{label} months are malformed: {value!r}")
    return frozenset(value)


def _settlement_month(value: Any, *, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{2})", str(value))
    if match is None or not 1 <= int(match.group(2)) <= 12:
        raise IssuanceCalibrationError(f"{label} from/through must be YYYY-MM: {value!r}")
    return int(match.group(1)), int(match.group(2))


def _coupon_day(value: Any, *, label: str) -> int | str:
    if value in COUPON_DAY_RULES:
        return value
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 28:
        return value
    raise IssuanceCalibrationError(f"{label} day is malformed: {value!r}")


def settlement_rules(profile: Mapping[str, Any]) -> dict[tuple[str, float], SecurityCalendar]:
    """Return each rule security's settlement calendar, keyed like the supply items.

    Raises unless every bill term and coupon security in the profile has exactly one
    calendar, every calendar names a profile security, and each coupon security's
    months are covered at most once and include at least one new-issue month.
    """

    calendar = profile.get("settlement_calendar")
    if not isinstance(calendar, Mapping) or set(calendar) != {"bills", "coupons", "fiscal_year_close"}:
        raise IssuanceCalibrationError(
            "settlement_calendar must have exactly bills, coupons and fiscal_year_close"
        )
    fiscal_year_close_rules(profile)
    rules: dict[tuple[str, float], SecurityCalendar] = {}

    def add(rule: SecurityCalendar) -> None:
        if rule.key in rules:
            raise IssuanceCalibrationError(f"settlement_calendar lists {rule.key} twice")
        rules[rule.key] = rule

    for entry in calendar["bills"]:
        if not isinstance(entry, Mapping) or not {"weeks", "weekday"} <= set(entry) <= {
            "weeks", "weekday", "every_weeks", "anchor_date",
        }:
            raise IssuanceCalibrationError(f"settlement_calendar bill entry is malformed: {entry!r}")
        weekday = entry["weekday"]
        if weekday not in _WEEKDAYS:
            raise IssuanceCalibrationError(f"settlement_calendar bill weekday is malformed: {entry!r}")
        every = entry.get("every_weeks", 1)
        if not isinstance(every, int) or isinstance(every, bool) or every < 1:
            raise IssuanceCalibrationError(f"settlement_calendar bill every_weeks is malformed: {entry!r}")
        anchor = None
        if every > 1:
            try:
                anchor = date.fromisoformat(str(entry.get("anchor_date")))
            except ValueError as exc:
                raise IssuanceCalibrationError(
                    f"settlement_calendar bill cycle needs an ISO anchor_date: {entry!r}"
                ) from exc
            if anchor.weekday() != _WEEKDAYS.index(weekday):
                raise IssuanceCalibrationError(f"settlement_calendar bill anchor is not a {weekday}: {entry!r}")
        elif "anchor_date" in entry:
            raise IssuanceCalibrationError(f"settlement_calendar weekly bill has an anchor_date: {entry!r}")
        add(
            SecurityCalendar(
                key=bill_supply_key(int(entry["weeks"])),
                weekday=_WEEKDAYS.index(weekday),
                every_weeks=every,
                anchor=anchor,
            )
        )
    for entry in calendar["coupons"]:
        if not isinstance(entry, Mapping) or set(entry) != {"kind", "tenor_years", "dated", "schedule"}:
            raise IssuanceCalibrationError(f"settlement_calendar coupon entry is malformed: {entry!r}")
        kind = entry["kind"]
        if kind not in COUPON_KIND_SECURITY_TYPE:
            raise IssuanceCalibrationError(f"settlement_calendar has unknown coupon kind {kind!r}")
        label = f"settlement_calendar {kind} {entry['tenor_years']}-year"
        if entry["dated"] not in (15, "month_end"):
            raise IssuanceCalibrationError(f"{label} dated must be 15 or month_end")
        schedule = entry["schedule"]
        if not isinstance(schedule, list) or not schedule:
            raise IssuanceCalibrationError(f"{label} schedule must be a nonempty list")
        entries: list[SettlementEntry] = []
        for item in schedule:
            if not isinstance(item, Mapping) or not {"issue", "months", "day"} <= set(item) <= {
                "issue", "months", "day", "from", "through",
            }:
                raise IssuanceCalibrationError(f"{label} schedule entry is malformed: {item!r}")
            if item["issue"] not in ISSUE_KINDS:
                raise IssuanceCalibrationError(f"{label} issue must be new or reopening: {item!r}")
            parsed = SettlementEntry(
                _months(item["months"], label=label),
                _coupon_day(item["day"], label=label),
                item["issue"],
                _settlement_month(item.get("from"), label=label),
                _settlement_month(item.get("through"), label=label),
            )
            if parsed.first is not None and parsed.last is not None and parsed.first > parsed.last:
                raise IssuanceCalibrationError(f"{label} schedule entry ends before it starts: {item!r}")
            for other in entries:
                if other.months & parsed.months and (
                    (parsed.last is None or other.first is None or other.first <= parsed.last)
                    and (other.last is None or parsed.first is None or parsed.first <= other.last)
                ):
                    raise IssuanceCalibrationError(f"{label} lists a month twice")
            entries.append(parsed)
        if not any(item.issue == "new" for item in entries):
            raise IssuanceCalibrationError(f"{label} has no new-issue month")
        add(
            SecurityCalendar(
                key=coupon_supply_key(COUPON_KIND_SECURITY_TYPE[kind], entry["tenor_years"]),
                dated=entry["dated"],
                entries=tuple(entries),
            )
        )
    securities = {bill_supply_key(term["weeks"]) for term in profile["bill_terms"]} | {
        coupon_supply_key(term["security_type"], term["maturity_years"]) for term in profile["coupon_terms"]
    }
    if set(rules) != securities:
        missing = sorted(securities - set(rules))
        extra = sorted(set(rules) - securities)
        raise IssuanceCalibrationError(
            f"settlement_calendar must cover exactly the profile's securities: missing={missing}, extra={extra}"
        )
    return rules


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar_module.monthrange(year, month)[1])


def two_year_auction_week_monday(year: int, month: int) -> date:
    """Monday of the week Treasury auctions the month's 2-year note.

    The note settles at month end (the next business day when closed), a few business
    days after its auction: in the same week when that settlement falls on a Thursday or
    Friday, otherwise in the week before. The 20-year bond is auctioned the week before
    that one, so ``friday_before_two_year_auction_week`` is its auction week's Friday.
    check_settlement_calendar() checks this week against the auction file's dates.
    """

    settle = next_securities_business_day(_month_end(year, month))
    monday = settle - timedelta(days=settle.weekday())
    return monday if settle.weekday() >= 3 else monday - timedelta(days=7)


def _coupon_settlement(rule: SettlementEntry, year: int, month: int) -> tuple[date, date]:
    """Return (nominal, settle) for a coupon settlement in ``year``-``month``."""

    if rule.day == "month_end":
        nominal = _month_end(year, month)
        return nominal, next_securities_business_day(nominal)
    if rule.day == "last_business_day":
        nominal = _month_end(year, month)
        return nominal, previous_securities_business_day(nominal)
    if rule.day == "last_friday":
        nominal = _month_end(year, month)
        nominal -= timedelta(days=(nominal.weekday() - 4) % 7)
        return nominal, next_securities_business_day(nominal)
    if rule.day == "friday_before_two_year_auction_week":
        nominal = two_year_auction_week_monday(year, month) - timedelta(days=3)
        return nominal, next_securities_business_day(nominal)
    nominal = date(year, month, int(rule.day))
    return nominal, next_securities_business_day(nominal)


def settlement_events(profile: Mapping[str, Any], start: date, end: date) -> list[SettlementEvent]:
    """Calendar settlements whose business day falls in ``[start, end]``, in date order."""

    first = start - timedelta(days=40)
    events: list[SettlementEvent] = []
    for key, rule in settlement_rules(profile).items():
        if rule.is_bill:
            if rule.anchor is None:
                day = first + timedelta(days=(rule.weekday - first.weekday()) % 7)
            else:
                cycle = 7 * rule.every_weeks
                day = rule.anchor + timedelta(days=((first - rule.anchor).days // cycle) * cycle)
            while day <= end:
                settle = next_securities_business_day(day)
                if start <= settle <= end:
                    events.append(SettlementEvent(settle, key, "new", day))
                day += timedelta(days=7 * rule.every_weeks)
            continue
        year, month = first.year, first.month
        while date(year, month, 1) <= end:
            for entry in rule.entries:
                if entry.holds(year, month):
                    nominal, settle = _coupon_settlement(entry, year, month)
                    if start <= settle <= end:
                        events.append(SettlementEvent(settle, key, entry.issue, nominal))
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return sorted(events, key=lambda event: (event.settle, event.key))


def fiscal_year_close_dates(profile: Mapping[str, Any], start: date, end: date) -> list[date]:
    """Last business days of the close months whose business day falls in ``[start, end]``."""

    close_months = set(fiscal_year_close_rules(profile))
    out = []
    for year in range(start.year, end.year + 1):
        for month in sorted(close_months):
            close = previous_securities_business_day(_month_end(year, month))
            if start <= close <= end:
                out.append(close)
    return out


def dated_date(rule: SecurityCalendar, nominal: date) -> date:
    """A new coupon issue's dated date: the 15th or the end of its nominal month."""

    if rule.dated == "month_end":
        return _month_end(nominal.year, nominal.month)
    if rule.dated == 15:
        return date(nominal.year, nominal.month, 15)
    raise IssuanceCalibrationError(f"{rule.key} has no dated-date rule")


def maturity_from_dated(rule: SecurityCalendar, dated: date) -> date:
    """Tenor years after the dated date; month-end dated issues mature on a month end."""

    tenor = int(round(rule.key[1]))
    if abs(rule.key[1] - tenor) > 1e-9:
        raise IssuanceCalibrationError(f"{rule.key} tenor is not a whole number of years")
    if rule.dated == "month_end":
        return _month_end(dated.year + tenor, dated.month)
    return date(dated.year + tenor, dated.month, dated.day)


def is_rule_profile(profile: Any) -> bool:
    return isinstance(profile, Mapping) and profile.get("mode") == ISSUANCE_RULE_MODE


def rule_target_path(profile: Mapping[str, Any]) -> list[tuple[pd.Timestamp, float]]:
    path = profile["bill_share_target_path"]
    if len(path) == 1 and path[0].get("date") is None:
        return [(pd.Timestamp("1900-01-01"), float(path[0]["bill_share_target"]))]
    return [
        (pd.Timestamp(point["date"]).normalize(), float(point["bill_share_target"]))
        for point in path
    ]


def active_special_categories(profile: Mapping[str, Any]) -> set[str]:
    """Preference categories beyond bills/notes/bonds that a profile can issue."""

    if is_rule_profile(profile):
        types = {
            str(term["security_type"])
            for term in profile.get("coupon_terms", [])
            if float(term.get("share", 0.0)) > TGA_FLOOR_TOLERANCE
        }
        return {category for category, kind in (("tips", "TIPS"), ("frn", "FRN")) if kind in types}
    active = set()
    for key, category in (("TIPS", "tips"), ("FRN", "frn")):
        cfg = profile.get(key, {}) if isinstance(profile, Mapping) else {}
        if float((cfg or {}).get("target_percentage", 0.0) or 0.0) > TGA_FLOOR_TOLERANCE:
            active.add(category)
    return active


# --- Package calibration file ----------------------------------------------------------


def build_issuance_calibration(
    auctions_csv: str | Path,
    *,
    simulation_start_date: str,
    source_relative_path: str,
    opening_portfolio: pd.DataFrame,
    check_calendar: bool = True,
) -> dict[str, Any]:
    """Auction shares, the opening bill share of controlled marketable debt, and the
    settlement calendar with its check against the auction source.

    ``check_calendar=False`` is for synthetic auction fixtures that do not carry a full
    year of Treasury's calendar; the calibration then records that it was not checked.
    """

    auctions = build_auction_calibration(auctions_csv, simulation_start_date=simulation_start_date)
    controlled_debt, bills = controlled_debt_and_bills(opening_portfolio)
    if controlled_debt <= TGA_FLOOR_TOLERANCE:
        raise IssuanceCalibrationError("opening portfolio has no controlled marketable debt")
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_file": source_relative_path,
        **auctions,
        "bill_share_target": bills / controlled_debt,
        "settlement_calendar": json.loads(json.dumps(SETTLEMENT_CALENDAR)),
        "settlement_calendar_check": (
            check_settlement_calendar(auctions_csv, simulation_start_date=simulation_start_date)
            if check_calendar
            else {"status": "not_checked_synthetic_auction_source"}
        ),
        "bill_share_target_basis": {
            "as_of": simulation_start_date,
            "portfolio": "opening_portfolio_after_prestart_rollforward",
            "controlled_marketable_debt_bil": controlled_debt,
            "bill_face_bil": bills,
            "definition": (
                "active Fixed, TIPS and FRN held outside FedInternal and TrustFunds, TIPS at "
                "adjusted principal; bills are Fixed with original maturity of at most one year"
            ),
        },
        "claim_boundary": "issuance_shares_measured_from_public_auction_volume_not_cbo_prescription",
    }


def load_calibration(path: str | Path) -> dict[str, Any]:
    """Read and check a package issuance calibration; raises on any malformation."""

    file_path = Path(path)
    if not file_path.is_file():
        raise IssuanceCalibrationError(f"issuance calibration is missing: {file_path.name}")
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IssuanceCalibrationError(f"issuance calibration is unreadable: {file_path.name}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise IssuanceCalibrationError("issuance calibration has an unsupported schema_version")
    for key in ("bill_terms", "coupon_terms", "bill_share_target", "settlement_calendar"):
        if key not in payload:
            raise IssuanceCalibrationError(f"issuance calibration is missing {key}")
    _share(payload["bill_share_target"], label="calibration bill_share_target")
    engine_rule_profile(
        target_path=[{"date": None, "bill_share_target": payload["bill_share_target"]}],
        bill_terms=payload["bill_terms"],
        coupon_terms=payload["coupon_terms"],
        settlement_calendar=payload["settlement_calendar"],
    )
    return payload


# --- Pre-start rollforward of the opening book ------------------------------------------

PRESTART_ROLLFORWARD_RULE = (
    "each_security_maturing_between_the_record_date_and_the_start_is_reinvested_holder_by_holder_"
    "in_the_treasury_issues_settling_on_its_payment_date_pro_rata_to_their_accepted_amounts"
)
_AUCTION_SECURITY_CATEGORY = {"Bill": "bills", "Note": "notes", "Bond": "bonds"}
# Row fields that describe the holder and the amount, not the security.
_HOLDER_FIELDS = (
    "BondID",
    "FaceValue",
    "HolderType",
    "HolderSubBucket",
    "TDCPrincipalHolderType",
    "TDCPrincipalHolderSubBucket",
    "IssueProceeds",
    "Status",
)


def _percent(row: Mapping[str, str], column: str, *, line: int) -> float:
    text = str(row.get(column, "")).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise IssuanceCalibrationError(
            f"auction row {line}: {column} is not a number ({text!r})"
        ) from exc
    if not math.isfinite(value):
        raise IssuanceCalibrationError(f"auction row {line}: {column} is not finite")
    return value / 100.0


def prestart_settlements(
    auctions_csv: str | Path,
    *,
    record_date: str,
    simulation_start_date: str,
) -> pd.DataFrame:
    """Treasury issues settling after the opening book's record date, up to the start.

    One row per security and settlement date, with the CUSIP's original issue date (a
    reopening carries it), maturity, coupon rate, the auction's accepted amount including
    Fed add-ons, and its price and yield. Raises on an indexed (TIPS or FRN) issue, which
    the rollforward does not reconstruct, on malformed rows, and on an empty window.
    """

    record = date.fromisoformat(record_date)
    start = date.fromisoformat(simulation_start_date)
    path = Path(auctions_csv)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for line, row in enumerate(rows, start=2):
        issue = _date(row, "issue_date", line=line)
        if not (record < issue <= start):
            continue
        security_type = str(row.get("security_type", "")).strip()
        if security_type not in _AUCTION_SECURITY_CATEGORY:
            raise IssuanceCalibrationError(f"auction row {line}: unknown security_type {security_type!r}")
        if _flag(row, "floating_rate", line=line) or _flag(row, "inflation_index_security", line=line):
            raise IssuanceCalibrationError(
                f"auction row {line}: an indexed security settles before the start; "
                "the pre-start rollforward reconstructs only bills, notes and bonds"
            )
        original_text = str(row.get("original_issue_date", "")).strip()
        bill = security_type == "Bill"
        out.append(
            {
                "cusip": str(row.get("cusip", "")).strip(),
                "security_type": security_type,
                "settlement_date": pd.Timestamp(issue),
                "original_issue_date": pd.Timestamp(
                    issue if original_text in {"", "null"} else _date(row, "original_issue_date", line=line)
                ),
                "maturity_date": pd.Timestamp(_date(row, "maturity_date", line=line)),
                "accepted_bil": _dollars(row, "total_accepted", line=line) / 1.0e9,
                "coupon_rate": 0.0 if bill else _percent(row, "int_rate", line=line),
                "issue_price_ratio": _percent(row, "price_per100", line=line),
                "issue_yield": _percent(row, "high_investment_rate" if bill else "high_yield", line=line),
            }
        )
    if not out:
        raise IssuanceCalibrationError(f"no Treasury issue settles in ({record}, {start}]")
    frame = pd.DataFrame(out)
    if frame.duplicated(["cusip", "settlement_date"]).any():
        raise IssuanceCalibrationError("an auction file lists one CUSIP twice on one settlement date")
    if (frame["accepted_bil"] <= TGA_FLOOR_TOLERANCE).any():
        raise IssuanceCalibrationError("a pre-start issue has no accepted amount")
    return frame.sort_values(["settlement_date", "maturity_date", "cusip"]).reset_index(drop=True)


def prestart_rollforward(
    book: pd.DataFrame,
    settlements: pd.DataFrame,
    *,
    simulation_start_date: str,
    next_bond_id: int,
    cusip_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an active opening book at the start and roll its matured rows forward.

    A row maturing on or before the start is paid on the first settlement date on or after
    its maturity, and its face is reinvested, for the same holder, in every Treasury issue
    settling that day in proportion to the issue's accepted amount. A replacement in a
    CUSIP the book already holds takes that CUSIP's terms from the book. Returns the
    surviving rows and the replacement rows; the replacements carry ``_SourceBondID`` and
    ``_PaymentDate`` for diagnostics. Raises unless face and holder totals are unchanged,
    every replacement matures after the start, and no maturity date receives more
    replacement face than the largest amount that matured on any one date before the start.
    """

    start = pd.Timestamp(simulation_start_date)
    matured = book[book["MaturityDate"] <= start]
    kept = book[book["MaturityDate"] > start]
    if matured.empty:
        return kept, pd.DataFrame(columns=[*book.columns, "_SourceBondID", "_PaymentDate"])
    templates = kept.drop_duplicates(cusip_column).set_index(cusip_column)
    security_fields = [c for c in book.columns if c not in _HOLDER_FIELDS and c != cusip_column]
    terms: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
    for issue in settlements.itertuples(index=False):
        if issue.cusip in templates.index:
            template = templates.loc[issue.cusip]
            if (
                pd.Timestamp(template["MaturityDate"]) != issue.maturity_date
                or pd.Timestamp(template["IssueDate"]) != issue.original_issue_date
                or abs(float(template["CouponRate"]) - issue.coupon_rate) > 1e-9
            ):
                raise IssuanceCalibrationError(
                    f"the opening book and the auction file disagree on the terms of {issue.cusip}"
                )
            fields = {column: template[column] for column in security_fields}
        elif issue.original_issue_date < issue.settlement_date:
            raise IssuanceCalibrationError(
                f"{issue.cusip} is reopened on {issue.settlement_date.date()} but is not in the opening book"
            )
        else:
            fields = {column: pd.NA for column in security_fields}
            fields.update(
                {
                    "SecurityType": "Fixed",
                    "IssueDate": issue.original_issue_date,
                    "MaturityDate": issue.maturity_date,
                    "OriginalMaturityYears": (issue.maturity_date - issue.original_issue_date).days
                    / DAYS_PER_YEAR_ACTUAL,
                    "CouponRate": issue.coupon_rate,
                    "MaturityCategory": _AUCTION_SECURITY_CATEGORY[issue.security_type],
                    "OriginalPrincipal": 0.0,
                    "AdjustedPrincipal": 0.0,
                    "ReferenceCPI_Issue": 0.0,
                    "IndexRatio": 0.0,
                    "FixedSpread": 0.0,
                    "AccruedInterest_FRN": 0.0,
                    "BenchmarkRate_FRN": 0.0,
                    "IssuePriceRatio": issue.issue_price_ratio,
                    "IssueYieldAtIssue": issue.issue_yield,
                }
            )
        if issue.maturity_date <= start:
            raise IssuanceCalibrationError(f"{issue.cusip} issued before the start also matures before it")
        terms[(issue.cusip, issue.settlement_date)] = fields

    dates = settlements["settlement_date"].drop_duplicates().sort_values().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    bond_id = int(next_bond_id)
    for _, source in matured.iterrows():
        later = dates[dates >= pd.Timestamp(source["MaturityDate"])]
        if later.empty:
            raise IssuanceCalibrationError(
                f"no Treasury issue settles between {pd.Timestamp(source['MaturityDate']).date()} and the start"
            )
        payment = later.iloc[0]
        issues = settlements[settlements["settlement_date"] == payment]
        weights = issues["accepted_bil"] / issues["accepted_bil"].sum()
        face = float(source["FaceValue"])
        for issue, weight in zip(issues.itertuples(index=False), weights):
            row = {column: pd.NA for column in book.columns}
            row.update(terms[(issue.cusip, payment)])
            share = face * float(weight)
            price = row.get("IssuePriceRatio")
            row.update(
                {
                    "BondID": bond_id,
                    "FaceValue": share,
                    "HolderType": source["HolderType"],
                    "HolderSubBucket": source["HolderSubBucket"],
                    "Status": "Active",
                    "IssueProceeds": share * float(price) if pd.notna(price) else pd.NA,
                    cusip_column: issue.cusip,
                    "_SourceBondID": source["BondID"],
                    "_PaymentDate": payment,
                }
            )
            rows.append(row)
            bond_id += 1
    replacements = pd.DataFrame(rows)
    _check_prestart_rollforward(matured, replacements, start)
    return kept, replacements


def _check_prestart_rollforward(matured: pd.DataFrame, replacements: pd.DataFrame, start: pd.Timestamp) -> None:
    def holder_totals(frame: pd.DataFrame) -> pd.Series:
        keys = [frame["HolderType"].astype(str), frame["HolderSubBucket"].fillna("").astype(str)]
        return frame["FaceValue"].astype(float).groupby(keys).sum()

    gap = holder_totals(replacements).subtract(holder_totals(matured), fill_value=0.0).abs()
    if gap.max() > 1e-9:
        raise IssuanceCalibrationError(f"the pre-start rollforward changes a holder's face by {gap.max()}")
    if (pd.to_datetime(replacements["MaturityDate"]) <= start).any():
        raise IssuanceCalibrationError("a pre-start replacement matures on or before the start")
    largest_paid = matured["FaceValue"].astype(float).groupby(pd.to_datetime(matured["MaturityDate"])).sum().max()
    by_maturity = replacements["FaceValue"].astype(float).groupby(pd.to_datetime(replacements["MaturityDate"])).sum()
    if by_maturity.max() > largest_paid + 1e-9:
        raise IssuanceCalibrationError(
            f"the pre-start rollforward puts {by_maturity.max():.1f} billion on {by_maturity.idxmax().date()}, "
            f"more than the {largest_paid:.1f} billion that matured on any one day before the start"
        )
