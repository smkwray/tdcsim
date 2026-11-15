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
import csv
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdc_shared import DAYS_PER_YEAR_ACTUAL, INTRAGOV_HOLDERS, TGA_FLOOR_TOLERANCE

ISSUANCE_RULE_MODE = "bill_share_target_v1"
CALIBRATION_SCHEMA_VERSION = "tdcsim_issuance_calibration_v1"
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
# Treasury's settlement calendar for the bill-share rule's securities. Weekday and
# day-of-month dates move to the next business day; month-end issues settle on the
# month's last business day. Packaged in the calibration and read back through
# settlement_rules(), the one loader.
SETTLEMENT_CALENDAR = {
    "bills": [
        {"weeks": 4, "weekday": "Tuesday"},
        {"weeks": 8, "weekday": "Tuesday"},
        {"weeks": 17, "weekday": "Tuesday"},
        {"weeks": 6, "weekday": "Thursday"},
        {"weeks": 13, "weekday": "Thursday"},
        {"weeks": 26, "weekday": "Thursday"},
        {"weeks": 52, "weekday": "Thursday"},
    ],
    "coupons": [
        {"kind": "nominal", "tenor_years": 3, "day_of_month": 15},
        {"kind": "nominal", "tenor_years": 10, "day_of_month": 15},
        {"kind": "nominal", "tenor_years": 30, "day_of_month": 15},
        {"kind": "nominal", "tenor_years": 2, "day_of_month": "last_business_day"},
        {"kind": "nominal", "tenor_years": 5, "day_of_month": "last_business_day"},
        {"kind": "nominal", "tenor_years": 7, "day_of_month": "last_business_day"},
        {"kind": "nominal", "tenor_years": 20, "day_of_month": "last_business_day"},
        {"kind": "frn", "tenor_years": 2, "day_of_month": "last_business_day"},
        {"kind": "tips", "tenor_years": 5, "day_of_month": "last_business_day"},
        {"kind": "tips", "tenor_years": 10, "day_of_month": "last_business_day"},
        {"kind": "tips", "tenor_years": 30, "day_of_month": "last_business_day"},
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

    Public amount is ``total_accepted - soma_accepted``: the Fed takes no auction
    allocation in the CBO lane, so its add-ons are not part of the modeled supply.
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

    original_years = pd.to_numeric(frame["OriginalMaturityYears"], errors="coerce").fillna(0.0)
    return frame["SecurityType"].astype(str).eq("Fixed") & (
        original_years <= BILL_CUTOFF_YEARS + TGA_FLOOR_TOLERANCE
    )


def debt_base(frame: pd.DataFrame) -> pd.Series:
    """Face value, with TIPS at adjusted principal."""

    face = pd.to_numeric(frame["FaceValue"], errors="coerce")
    adjusted = pd.to_numeric(frame["AdjustedPrincipal"], errors="coerce").fillna(face)
    values = np.where(frame["SecurityType"].astype(str).eq("TIPS"), adjusted, face.fillna(0.0))
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


def bill_supply_key(weeks: int) -> tuple[str, float]:
    return ("bill", float(weeks))


def coupon_supply_key(security_type: str, tenor_years: float) -> tuple[str, float]:
    return (str(security_type), float(tenor_years))


def settlement_rules(profile: Mapping[str, Any]) -> dict[tuple[str, float], tuple[str, int | None]]:
    """Return each rule security's settlement rule, keyed like the supply items.

    A rule is ``("weekday", 0-4)``, ``("day_of_month", 1-28)`` or
    ``("last_business_day", None)``. Raises unless every bill term and coupon security in
    the profile has exactly one rule and every rule names a profile security.
    """

    calendar = profile.get("settlement_calendar")
    if not isinstance(calendar, Mapping) or set(calendar) != {"bills", "coupons"}:
        raise IssuanceCalibrationError("settlement_calendar must have exactly bills and coupons")
    rules: dict[tuple[str, float], tuple[str, int | None]] = {}

    def add(key: tuple[str, float], rule: tuple[str, int | None]) -> None:
        if key in rules:
            raise IssuanceCalibrationError(f"settlement_calendar lists {key} twice")
        rules[key] = rule

    for entry in calendar["bills"]:
        weekday = entry.get("weekday") if isinstance(entry, Mapping) else None
        if weekday not in _WEEKDAYS or set(entry) != {"weeks", "weekday"}:
            raise IssuanceCalibrationError(f"settlement_calendar bill entry is malformed: {entry!r}")
        add(bill_supply_key(int(entry["weeks"])), ("weekday", _WEEKDAYS.index(weekday)))
    for entry in calendar["coupons"]:
        if not isinstance(entry, Mapping) or set(entry) != {"kind", "tenor_years", "day_of_month"}:
            raise IssuanceCalibrationError(f"settlement_calendar coupon entry is malformed: {entry!r}")
        kind = entry["kind"]
        if kind not in COUPON_KIND_SECURITY_TYPE:
            raise IssuanceCalibrationError(f"settlement_calendar has unknown coupon kind {kind!r}")
        day = entry["day_of_month"]
        if day == "last_business_day":
            rule: tuple[str, int | None] = ("last_business_day", None)
        elif isinstance(day, int) and not isinstance(day, bool) and 1 <= day <= 28:
            rule = ("day_of_month", day)
        else:
            raise IssuanceCalibrationError(f"settlement_calendar day_of_month is malformed: {day!r}")
        add(coupon_supply_key(COUPON_KIND_SECURITY_TYPE[kind], entry["tenor_years"]), rule)
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
) -> dict[str, Any]:
    """Auction shares plus the opening bill share of controlled marketable debt."""

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
