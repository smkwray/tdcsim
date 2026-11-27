#!/usr/bin/env python3
"""Write a scenario family's CBO scenario specs from a committed family spec.

Each scenario may scale one holder's auction-absorption share by a multiplier
(one number, or one per marketable security type) and reassign the released
share to named absorbers in fixed proportions. A scenario may move the
bill-share target by dated changes from the package's opening bill share, and
may name the holders whose bill holdings absorb that change (the others keep
their baseline bill holdings in dollars). A scenario may give one buyer a dollar
amount of added bill holdings, built evenly over a window and held after it, or
add a dollar ramp to the Federal Reserve's holdings path. A scenario may instead hold a
buyer's realized Treasury holdings a stated dollar amount above the family baseline's,
built evenly over a window and held after it, with named donors giving up the position in
fixed proportions to their bill holdings at the window's start; the targets are read from
the completed baseline run, so these scenarios are written in a later phase. A family may
declare its simulation interval. A scenario may also impose a
nominal curve shift: either a constant key-rate shock, or a dated rate path
derived from a completed quantity-only run of the family, in which the 10-year
shift at each curve date is proportional to the shifted holder's shortfall as a
share of marketable debt at that date. Baseline shares and curves are read from
the release-bound baseline package or the family's own runs, never restated here.

``--phase quantity`` writes the scenarios that need no completed run; ``--phase rates``
writes the rate-path scenarios, and ``--phase positions`` the holder-target scenarios,
into the same directory once the runs they are derived from have completed.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import holder_targets as ht  # noqa: E402
from simulation_calendar import previous_securities_business_day  # noqa: E402
from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec  # noqa: E402
from tdcsim_cbo._json import canonical_json_sha256, sha256_file, write_json  # noqa: E402
from tdc_shared import (  # noqa: E402
    HOLDER_TYPES, MARKETABLE_PREFERENCE_CATEGORIES, PRIVATE_AUCTION_SUBBUCKETS, PRIVATE_SUBBUCKET_MMF_ADDED,
)
from issuance_calibration import CALIBRATION_FILE, ISSUANCE_RULE_MODE, load_calibration, target_for_date  # noqa: E402
from write_cbo_example_scenarios import _baseline_coupling, _scenario  # noqa: E402

FAMILY_SPEC_SCHEMA = "tdcsim_scenario_family_spec_v1"
HOLDER_PROFILE_FILE = "tdcsim_holder_profile_assumptions.csv"
FED_HOLDINGS_FILE = "tdcsim_fed_holdings_path.csv"
DEBT_STOCK_FILE = "tdcsim_debt_stock_path.csv"
ITEM_KEYS = frozenset({
    "role", "scenario_id", "label", "notes", "holder_shift", "bill_share_change", "bill_holder_shift",
    "bill_buyer_shift", "fed_holdings_change", "key_rate_shocks_bp", "rate_path", "mmf_deposit_pass_through",
    "holder_position_target", "mmf_added_position_settlement",
})
SUBKEYS = {
    "bill_holder_shift": frozenset({"absorbers", "stock_control"}),
    "bill_buyer_shift": frozenset({"holder", "size_bil", "build", "private_route"}),
    "fed_holdings_change": frozenset({"size_bil", "build"}),
    "holder_position_target": frozenset({
        "buyer_holder", "buyer_private_route", "size_bil", "origin", "build_end", "donors", "unchanged_routes",
    }),
}
PHASES = ("quantity", "rates", "positions")
# How a bill holder shift is delivered. "achieved_bill_stocks" adds a daily position control
# that holds every other route at its baseline bill stock and gives the absorbers the realized
# difference; "auction_shares_only" changes the auction shares and lets the stocks follow.
BILL_STOCK_CONTROLS = ("achieved_bill_stocks", "auction_shares_only")
STOCK_CLOSURE = Path("outputs/tdcsim_period_stock_closure.csv.gz")
RUN_MANIFEST = "tdcsim_cbo_run_manifest.json"
HOLDER_TARGET_TOLERANCE_BIL = 0.000001
SHARE_TOLERANCE = 1.0e-12
CURVE_SURFACE = Path("compile/compiled/forecast_inputs/tdcsim_yield_curve_surface.csv")
RESULTS = Path("outputs/results_compact.csv.gz")
HOLDER_STOCK_COLUMNS = {
    "Banks": "DebtHeld_Banks",
    "CB": "DebtHeld_CentralBank",
    "Foreign": "DebtHeld_Foreign",
    "Private": "DebtHeld_DomesticNonBanks",
}


def baseline_holder_shares(baseline: CboBaselinePackage) -> dict[str, dict[str, float]]:
    """Return {security_type: {holder: share}} from the baseline holder profile."""

    with tempfile.TemporaryDirectory(prefix="tdcsim-family-baseline-") as tmp:
        root = baseline.materialize(Path(tmp) / "baseline")
        matches = list(root.rglob(HOLDER_PROFILE_FILE))
        if len(matches) != 1:
            raise ValueError(f"expected one baseline {HOLDER_PROFILE_FILE}, found {len(matches)}")
        with matches[0].open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if not str(row.get("holder_subbucket") or "").strip()]
    shares: dict[str, dict[str, float]] = {}
    for security_type in MARKETABLE_PREFERENCE_CATEGORIES:
        column = f"{security_type}_pct"
        observed = {row["holder_type"]: float(row[column]) for row in rows}
        unknown = set(observed) - set(HOLDER_TYPES)
        if unknown:
            raise ValueError(f"baseline holder profile names unknown holders {sorted(unknown)}")
        # A holder without a baseline profile row has no auction preference; the
        # override states that as an explicit zero, as the validator requires.
        shares[security_type] = {holder: observed.get(holder, 0.0) for holder in HOLDER_TYPES}
        if abs(sum(shares[security_type].values()) - 1.0) > 1.0e-9:
            raise ValueError(f"baseline {security_type} shares do not sum to one")
    return shares


def shifted_holder_rows(
    base: dict[str, dict[str, float]],
    *,
    holder: str,
    multiplier: float | dict[str, float],
    absorbers: dict[str, float],
) -> list[dict[str, Any]]:
    if isinstance(multiplier, dict):
        if set(multiplier) != set(base):
            raise ValueError(f"per-type multipliers must name exactly {sorted(base)}")
        multipliers = {key: float(value) for key, value in multiplier.items()}
    else:
        multipliers = {key: float(multiplier) for key in base}
    if not all(0.0 <= value <= 1.0 for value in multipliers.values()):
        raise ValueError("holder multiplier must lie in [0, 1]")
    if holder in absorbers or not absorbers or abs(sum(absorbers.values()) - 1.0) > SHARE_TOLERANCE:
        raise ValueError("absorbers must exclude the shifted holder and sum to one")
    unknown = (set(absorbers) | {holder}) - set(HOLDER_TYPES)
    if unknown:
        raise ValueError(f"unknown holders {sorted(unknown)}")
    rows = []
    for security_type, shares in base.items():
        released = shares[holder] * (1.0 - multipliers[security_type])
        new = dict(shares)
        new[holder] = shares[holder] * multipliers[security_type]
        for absorber, weight in absorbers.items():
            new[absorber] = shares[absorber] + released * weight
        if abs(sum(new.values()) - 1.0) > 1.0e-9:
            raise ValueError(f"{security_type} shares do not sum to one after the shift")
        rows.append({"security_type": security_type, "shares": {h: new[h] for h in HOLDER_TYPES}})
    return rows


def baseline_bill_share(baseline: CboBaselinePackage) -> float:
    """Return the package's opening bill share, the bill-share rule's baseline target."""

    with tempfile.TemporaryDirectory(prefix="tdcsim-family-calibration-") as tmp:
        root = baseline.materialize(Path(tmp) / "baseline")
        matches = list(root.rglob(CALIBRATION_FILE))
        if len(matches) != 1:
            raise ValueError(f"expected one baseline {CALIBRATION_FILE}, found {len(matches)}")
        return float(load_calibration(matches[0])["bill_share_target"])


def bill_share_target_path(base_target: float, changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dated bill-share targets: the package's opening share plus each dated change."""

    if not changes:
        raise ValueError("bill_share_change needs at least one dated point")
    path = []
    for point in changes:
        if set(point) != {"date", "change"}:
            raise ValueError("bill_share_change points must have exactly date and change")
        target = base_target + float(point["change"])
        if not 0.0 < target < 1.0:
            raise ValueError(f"bill-share target {target} on {point['date']} is outside (0, 1)")
        path.append({"date": str(point["date"]), "bill_share_target": target})
    return path


def bill_holder_rows(
    bill_shares: dict[str, float],
    *,
    base_target: float,
    target_path: list[dict[str, Any]],
    absorbers: list[str],
) -> list[dict[str, Any]]:
    """First-pass auction shares for a bill-stock change assigned to ``absorbers``.

    Auction shares apply to all bills sold, and bills roll within a year, so a holder's
    share of bill auctions is close to its share of the bill stock. With baseline shares
    s, opening bill share b and target t, every other holder keeps its baseline bill
    holdings in dollars (share s * b / t) under that approximation, and the absorbers split
    the change t - b in proportion to their baseline shares. The position controller
    reconciles actual closing stocks. Rows are written monthly across the target path.
    """

    unknown = set(absorbers) - set(HOLDER_TYPES)
    if not absorbers or unknown:
        raise ValueError(f"bill absorbers must name known holders, got {absorbers}")
    absorber_share = sum(bill_shares[holder] for holder in absorbers)
    if absorber_share <= 0.0:
        raise ValueError("bill absorbers hold no bills in the baseline")
    points = [(pd.Timestamp(point["date"]).normalize(), float(point["bill_share_target"])) for point in target_path]
    dates = sorted({point[0] for point in points} | set(pd.date_range(points[0][0], points[-1][0], freq="ME")))
    rows = []
    for day in dates:
        target = target_for_date(points, day)
        change = target - base_target
        shares = {
            holder: (
                bill_shares[holder] * (base_target + change / absorber_share) / target
                if holder in absorbers
                else bill_shares[holder] * base_target / target
            )
            for holder in HOLDER_TYPES
        }
        if min(shares.values()) < -SHARE_TOLERANCE:
            raise ValueError(
                f"bill absorbers {absorbers} hold too few bills to give up {-change:.4f} of debt on {day.date()}"
            )
        shares = {holder: max(0.0, value) for holder, value in shares.items()}
        if abs(sum(shares.values()) - 1.0) > 1.0e-9:
            raise ValueError(f"bill shares do not sum to one on {day.date()}")
        rows.append({"effective_date": day.date().isoformat(), "security_type": "bills", "shares": shares})
    return rows


def baseline_package_rows(baseline: CboBaselinePackage, filename: str) -> list[dict[str, str]]:
    """Return the rows of one packaged forecast input, read from the baseline package."""

    with tempfile.TemporaryDirectory(prefix="tdcsim-family-input-") as tmp:
        root = baseline.materialize(Path(tmp) / "baseline")
        matches = list(root.rglob(filename))
        if len(matches) != 1:
            raise ValueError(f"expected one baseline {filename}, found {len(matches)}")
        with matches[0].open(newline="") as handle:
            return list(csv.DictReader(handle))


def build_fraction(day: pd.Timestamp, build: list[str]) -> float:
    """Share of a dollar shift in place on ``day``: 0 before the window, linear in it, 1 after."""

    if len(build) != 2:
        raise ValueError("build must be [start date, end date]")
    start, end = (pd.Timestamp(value).normalize() for value in build)
    if end <= start:
        raise ValueError("build window must end after it starts")
    # Shifts build on securities business days only: a closed day keeps the last business
    # day's share, and a window ending on a closed day is complete on the business day before.
    day, start, end = (pd.Timestamp(previous_securities_business_day(value)) for value in (day, start, end))
    if end <= start:
        raise ValueError("build window must span a securities business day")
    if day <= start:
        return 0.0
    if day >= end:
        return 1.0
    return (day - start).days / (end - start).days


def bill_buyer_rows(
    bill_shares: dict[str, float],
    *,
    holder: str,
    size_bil: float,
    build: list[str],
    bill_stock_by_date: list[tuple[pd.Timestamp, float]],
    private_route: str | None = None,
    baseline_route_share: float | None = None,
) -> list[dict[str, Any]]:
    """Dated bill shares that give ``holder`` ``size_bil`` of added bill holdings.

    With ``private_route`` (Private only), the added holdings go to that private route:
    the other route keeps its baseline bill dollars, so its fraction of Private bills
    falls from 1 - m to (1 - m) * s / s', where m is the route's baseline bill fraction.

    Bills roll within a year, so a holder's share of bill auctions is close to its share
    of the bill stock. With baseline share s and expected bill stock B on a date, the
    holder's share rises to s + E/B, where E is the dollar shift in place on that date,
    and every other holder's share falls in proportion to its baseline share. Rows are
    written at the start and end of the build window and at each month end from the
    start to the end of the horizon, so the dollar amount is held after the window.
    """

    if holder not in HOLDER_TYPES or holder in {"CB", "FedInternal", "TrustFunds"}:
        raise ValueError(f"bill buyer must be a public auction holder, got {holder!r}")
    if size_bil <= 0.0:
        raise ValueError("bill buyer shift size must be positive")
    others = 1.0 - bill_shares[holder]
    if others <= 0.0:
        raise ValueError(f"{holder} already takes every bill")
    if private_route is not None:
        if holder != "Private" or private_route not in PRIVATE_AUCTION_SUBBUCKETS:
            raise ValueError(f"a private route needs holder Private and one of {list(PRIVATE_AUCTION_SUBBUCKETS)}")
        if baseline_route_share is None or not 0.0 <= baseline_route_share <= 1.0:
            raise ValueError("a private route needs its baseline bill route share in [0, 1]")
        (other_route,) = [route for route in PRIVATE_AUCTION_SUBBUCKETS if route != private_route]
    stocks = sorted(bill_stock_by_date)
    start = pd.Timestamp(build[0]).normalize()
    dates = sorted({start, pd.Timestamp(build[1]).normalize()} | set(pd.date_range(start, stocks[-1][0], freq="ME")))
    rows = []
    for day in dates:
        stock = next((value for when, value in reversed(stocks) if when <= day), None)
        if stock is None or stock <= 0.0:
            raise ValueError(f"no expected bill stock on or before {day.date()}")
        delta = size_bil * build_fraction(day, build) / stock
        target = bill_shares[holder] + delta
        if target > 1.0:
            raise ValueError(f"{holder} would need more than every bill on {day.date()}")
        shares = {
            name: target if name == holder else value * (1.0 - target) / others
            for name, value in bill_shares.items()
        }
        if abs(sum(shares.values()) - 1.0) > 1.0e-9:
            raise ValueError(f"bill shares do not sum to one on {day.date()}")
        row: dict[str, Any] = {"effective_date": day.date().isoformat(), "security_type": "bills", "shares": shares}
        if private_route is not None:
            other = (1.0 - baseline_route_share) * bill_shares["Private"] / target
            row["private_routes"] = {other_route: other, private_route: 1.0 - other}
        rows.append(row)
    return rows


def baseline_route_share(baseline: CboBaselinePackage, route: str, security_type: str = "bills") -> float:
    """The package's baseline share of Private purchases of one security type that goes to ``route``."""

    rows = [
        row
        for row in baseline_package_rows(baseline, HOLDER_PROFILE_FILE)
        if row["holder_type"] == "Private" and row["holder_subbucket"] == route
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one baseline Private {route} profile row, found {len(rows)}")
    return float(rows[0][f"{security_type}_route_share"])


def expected_bill_stock(debt_rows: list[dict[str, str]], bill_share: float) -> list[tuple[pd.Timestamp, float]]:
    """Expected bill stock: the bill-share target times the packaged marketable debt reference."""

    return [
        (pd.Timestamp(row["period_end"]).normalize(), bill_share * float(row["marketable_treasury_public_target_bil"]))
        for row in debt_rows
    ]


def fed_holdings_rows(rows: list[dict[str, str]], *, size_bil: float, build: list[str]) -> list[dict[str, str]]:
    """The packaged Fed holdings path plus a dollar ramp built over the window and held after it."""

    if pd.Timestamp(build[0]).normalize() <= min(pd.Timestamp(row["period_end"]).normalize() for row in rows):
        raise ValueError("a Fed holdings ramp must start after the opening")
    out = []
    for row in rows:
        new = dict(row)
        shift = size_bil * build_fraction(pd.Timestamp(row["period_end"]).normalize(), build)
        value = float(row["cbo_fed_holdings_target_bil"]) + shift
        if value < 0.0:
            raise ValueError(f"Fed holdings would turn negative on {row['period_end']}")
        new["cbo_fed_holdings_target_bil"] = repr(value)
        if shift != 0.0:
            new["source_status"] = "family_writer_fed_holdings_ramp_over_baseline_path"
        out.append(new)
    return out


def _write_fed_holdings_path(item: dict[str, Any], baseline: CboBaselinePackage, output_dir: Path) -> dict[str, Any]:
    spec = item["fed_holdings_change"]
    rows = fed_holdings_rows(
        baseline_package_rows(baseline, FED_HOLDINGS_FILE),
        size_bil=float(spec["size_bil"]),
        build=[str(value) for value in spec["build"]],
    )
    name = f"{item['role']}_fed_holdings_path.csv"
    with (output_dir / name).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "mode": "absolute_path_file",
        "file": {"relative_path": name, "sha256": sha256_file(output_dir / name), "media_type": "text/csv"},
    }


def nominal_curve_override(shocks: list[dict[str, float]]) -> dict[str, Any]:
    """Key-rate shock applied to the baseline curve, as in the term-premium examples."""

    return {
        "mode": "key_rate_bp",
        "interpolation": "log_tenor_linear",
        "shocks": [{"tenor_years": float(s["tenor_years"]), "shock_bp": float(s["shock_bp"])} for s in shocks],
    }


def _weight(tenor: float, weights: list[dict[str, float]]) -> float:
    """Key-rate weight: zero at or below the first key, flat beyond the last, log-tenor linear between."""

    keys = sorted((float(w["tenor_years"]), float(w["weight"])) for w in weights)
    if tenor <= keys[0][0]:
        return 0.0
    if tenor >= keys[-1][0]:
        return keys[-1][1]
    for (t0, w0), (t1, w1) in zip(keys, keys[1:]):
        if t0 <= tenor <= t1:
            return w0 + (w1 - w0) * (math.log(tenor) - math.log(t0)) / (math.log(t1) - math.log(t0))
    raise ValueError(f"tenor {tenor} is outside the key grid")


def _holder_stocks(run_dir: Path) -> list[tuple[str, dict[str, float]]]:
    with gzip.open(run_dir / RESULTS, "rt", encoding="utf-8", newline="") as handle:
        return [
            (row["Date"][:10], {holder: float(row[column]) for holder, column in HOLDER_STOCK_COLUMNS.items()})
            for row in csv.DictReader(handle)
        ]


def _as_of(stocks: list[tuple[str, dict[str, float]]], day: str) -> dict[str, float] | None:
    current = None
    for date, values in stocks:
        if date > day:
            break
        current = values
    return current


def rate_path_surface(
    surface: list[dict[str, str]],
    *,
    baseline_stocks: list[tuple[str, dict[str, float]]],
    source_stocks: list[tuple[str, dict[str, float]]],
    holder: str,
    bp_per_share_point: float,
    key_rate_weights: list[dict[str, float]],
) -> tuple[list[dict[str, str]], list[dict[str, float]]]:
    """Shift each curve date by the holder's shortfall share of marketable debt at that date.

    The shortfall is read as of the curve date (no look-ahead); dates before the
    first simulated observation are unchanged. Returns the shifted surface and the
    10-year shift path.
    """

    shifted, path = [], {}
    for row in surface:
        day = row["curve_date"]
        base, source = _as_of(baseline_stocks, day), _as_of(source_stocks, day)
        share = 0.0
        if base is not None and source is not None:
            share = (base[holder] - source[holder]) / sum(base.values())
        shift_10y_bp = bp_per_share_point * 100.0 * share
        path[day] = {"curve_date": day, "shortfall_share": share, "shift_10y_bp": shift_10y_bp}
        bp = shift_10y_bp * _weight(float(row["tenor_years"]), key_rate_weights)
        new = dict(row)
        new["nominal_rate_decimal"] = repr(float(row["nominal_rate_decimal"]) + bp / 10000.0)
        new["nominal_rate"] = repr(float(row["nominal_rate"]) + bp / 100.0)
        shifted.append(new)
    return shifted, list(path.values())


def _write_rate_path(item: dict[str, Any], runs_root: Path, output_dir: Path) -> dict[str, Any]:
    spec = item["rate_path"]
    source = runs_root / str(spec["from_role"]) / "run"
    with (runs_root / "baseline" / "run" / CURVE_SURFACE).open(newline="") as handle:
        reader = csv.DictReader(handle)
        header, surface = reader.fieldnames, list(reader)
    rows, path = rate_path_surface(
        surface,
        baseline_stocks=_holder_stocks(runs_root / "baseline" / "run"),
        source_stocks=_holder_stocks(source),
        holder=str(item["holder_shift"]["holder"]),
        bp_per_share_point=float(spec["bp_per_share_point"]),
        key_rate_weights=list(spec["key_rate_weights"]),
    )
    name = f"{item['role']}_nominal_surface.csv"
    with (output_dir / name).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / f"{item['role']}_rate_path.json", {"rate_path": spec, "shift_path": path})
    return {
        "mode": "full_surface_file",
        "file": {"relative_path": name, "sha256": sha256_file(output_dir / name), "media_type": "text/csv"},
    }


def _route(value: Any) -> tuple[str, str]:
    holder, route = value
    return str(holder), str(route or "")


def _item_phase(item: dict[str, Any]) -> str:
    if item.get("holder_position_target") or item.get("bill_holder_shift"):
        return "positions"
    return "rates" if item.get("rate_path") else "quantity"


def _write_bill_target(
    item: dict[str, Any], bill_shares: dict[str, float], runs_root: Path, output_dir: Path,
) -> dict[str, Any]:
    run = runs_root / "baseline" / "run"
    results = pd.read_csv(run / RESULTS, float_precision="round_trip")
    absorbers = item["bill_holder_shift"]["absorbers"]
    routes = [(holder, "") for holder in absorbers]
    total = sum(bill_shares[holder] for holder in absorbers)
    weights = [
        {"holder_type": holder, "holder_subbucket": subbucket, "weight": bill_shares[holder] / total}
        for holder, subbucket in routes
    ]
    ht.bill_change_weights(weights)
    columns = ht.STOCK_COLUMNS["HolderBillStock"]
    rows = [
        {"date": pd.Timestamp(row["Date"]).date().isoformat(), "holder_type": holder, "holder_subbucket": subbucket,
         "baseline_stock_bil": row[column]}
        for row in results[["Date", *columns.values()]].to_dict("records")
        for (holder, subbucket), column in columns.items()
    ]
    name = f"{item['role']}_holder_position_target.csv"
    with (output_dir / name).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ht.BILL_TARGET_FILE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    baseline_scenario = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    return {
        "mode": ht.BILL_TARGET_MODE,
        "bill_change_weights": weights,
        "baseline_reference": {
            "scenario_id": str(baseline_scenario["scenario_id"]),
            "scenario_sha256": canonical_json_sha256(baseline_scenario),
            "run_manifest_sha256": sha256_file(run / RUN_MANIFEST),
            "results_sha256": sha256_file(run / RESULTS),
        },
        "target_tolerance_bil": HOLDER_TARGET_TOLERANCE_BIL,
        "file": {"relative_path": name, "sha256": sha256_file(output_dir / name), "media_type": "text/csv"},
    }


def _write_holder_target(
    item: dict[str, Any], simulation: dict[str, str], runs_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Target file and override for a buyer held above the family baseline's realized holdings."""

    spec = item["holder_position_target"]
    run = runs_root / "baseline" / "run"
    closure_path = run / STOCK_CLOSURE
    closure = pd.read_csv(closure_path, keep_default_na=False, float_precision="round_trip")
    origin = str(spec["origin"])
    bills = ht.closing_holder_stocks(closure, bills_only=True)
    bills = bills[bills["period_end"] == origin]
    if bills.empty:
        raise ValueError(f"{item['role']}: the baseline has no close on {origin}")
    origin_bills = {(r.holder_sector, r.holder_subsector): float(r.closing_stock_bil) for r in bills.itertuples()}
    buyer = (str(spec["buyer_holder"]), str(spec.get("buyer_private_route") or ""))
    donors = [_route(value) for value in spec["donors"]]
    unchanged = [_route(value) for value in spec["unchanged_routes"]]
    weights = ht.displacement_weights(origin_bills, buyer=buyer, donors=donors)
    target_path = [
        {"date": origin, "increment_bil": 0.0},
        {"date": str(spec["build_end"]), "increment_bil": float(spec["size_bil"])},
        {"date": str(simulation["end_date"]), "increment_bil": float(spec["size_bil"])},
    ]
    rows = ht.target_rows(
        ht.closing_holder_stocks(closure), path=ht.BuildAndHoldPath.from_points(target_path), weights=weights
    )
    name = f"{item['role']}_holder_position_target.csv"
    with (output_dir / name).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ht.TARGET_FILE_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: repr(value) if isinstance(value, float) else value for key, value in row.items()} for row in rows
        )
    baseline_scenario = json.loads((run / "scenario.json").read_text(encoding="utf-8"))
    return {
        "mode": ht.HOLDER_TARGET_MODE,
        "buyer_holder": buyer[0],
        "buyer_private_route": buyer[1],
        "target_path": target_path,
        "interpolation": "linear_build_then_constant",
        "target_scope": "controlled_public_marketable",
        "quantity_basis": "par_or_adjusted_principal",
        "acquisition_market": "secondary",
        "eligible_instruments": "nominal_bills",
        "rebalance_timing": "after_fed_target_operation_before_period_close",
        "baseline_reference": {
            "scenario_id": str(baseline_scenario["scenario_id"]),
            "scenario_sha256": canonical_json_sha256(baseline_scenario),
            "run_manifest_sha256": sha256_file(run / RUN_MANIFEST),
            "stock_closure_sha256": sha256_file(closure_path),
        },
        "displacement_rule": "fixed_bill_stock_proportions_at_origin",
        "displacement": [
            {"holder_type": d[0], "holder_subbucket": d[1], "origin_bill_stock_bil": origin_bills[d], "weight": weights[d]}
            for d in donors
        ],
        "unchanged_routes": [{"holder_type": u[0], "holder_subbucket": u[1]} for u in unchanged],
        "allow_maintenance_sales": True,
        "target_tolerance_bil": HOLDER_TARGET_TOLERANCE_BIL,
        "on_infeasible": "error",
        "file": {"relative_path": name, "sha256": sha256_file(output_dir / name), "media_type": "text/csv"},
    }


def family_scenarios(
    baseline: CboBaselinePackage,
    family: dict[str, Any],
    *,
    phase: str = "quantity",
    runs_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if family.get("schema_version") != FAMILY_SPEC_SCHEMA:
        raise ValueError(f"family spec must declare {FAMILY_SPEC_SCHEMA}")
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {list(PHASES)}")
    simulation = family.get("simulation")
    if simulation is not None and set(simulation) != {"start_date", "end_date"}:
        raise ValueError("a family simulation interval names exactly start_date and end_date")
    for item in family["scenarios"]:
        unknown = set(item) - ITEM_KEYS
        if unknown:
            raise ValueError(f"{item.get('role')}: unknown family scenario keys {sorted(unknown)}")
        for key, allowed in SUBKEYS.items():
            if key in item and set(item[key]) - allowed:
                raise ValueError(f"{item['role']}: unknown {key} keys {sorted(set(item[key]) - allowed)}")
    base = baseline_holder_shares(baseline)
    base_bill_share = None
    scenarios: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(family["scenarios"]):
        if _item_phase(item) != phase:
            continue
        overrides: dict[str, Any] = {}
        shift = item.get("holder_shift")
        if item.get("bill_holder_shift") and not item.get("bill_share_change"):
            raise ValueError(f"{item['role']}: a bill holder shift needs a bill_share_change")
        if item.get("bill_share_change"):
            if base_bill_share is None:
                base_bill_share = baseline_bill_share(baseline)
            target_path = bill_share_target_path(base_bill_share, list(item["bill_share_change"]))
            overrides["issuance_mix"] = {"mode": ISSUANCE_RULE_MODE, "bill_share_target": target_path}
            if item.get("bill_holder_shift"):
                if shift:
                    raise ValueError(f"{item['role']}: declare a holder shift or a bill holder shift, not both")
                if runs_root is None or output_dir is None:
                    raise ValueError(f"{item['role']}: a bill holder shift needs the baseline run and an output directory")
                overrides["holder_preferences"] = {
                    "mode": "dated_static_shares",
                    "rows": bill_holder_rows(
                        base["bills"],
                        base_target=base_bill_share,
                        target_path=target_path,
                        absorbers=[str(h) for h in item["bill_holder_shift"]["absorbers"]],
                    ),
                }
                control = item["bill_holder_shift"].get("stock_control")
                if control not in BILL_STOCK_CONTROLS:
                    raise ValueError(f"{item['role']}: bill_holder_shift.stock_control must be one of {BILL_STOCK_CONTROLS}")
                if control == "achieved_bill_stocks":
                    overrides["holder_position_target"] = _write_bill_target(item, base["bills"], runs_root, output_dir)
        if item.get("bill_buyer_shift"):
            if item.get("bill_share_change"):
                raise ValueError(f"{item['role']}: a bill-buyer shift is sized on the opening bill share and cannot be combined with a bill_share_change")
            if shift or "holder_preferences" in overrides:
                raise ValueError(f"{item['role']}: declare one holder-share change per scenario")
            if base_bill_share is None:
                base_bill_share = baseline_bill_share(baseline)
            buyer = item["bill_buyer_shift"]
            overrides["holder_preferences"] = {
                "mode": "dated_static_shares",
                "rows": bill_buyer_rows(
                    base["bills"],
                    holder=str(buyer["holder"]),
                    size_bil=float(buyer["size_bil"]),
                    build=[str(value) for value in buyer["build"]],
                    bill_stock_by_date=expected_bill_stock(
                        baseline_package_rows(baseline, DEBT_STOCK_FILE), base_bill_share
                    ),
                    private_route=buyer.get("private_route"),
                    baseline_route_share=(
                        baseline_route_share(baseline, str(buyer["private_route"]))
                        if buyer.get("private_route")
                        else None
                    ),
                ),
            }
        if "mmf_deposit_pass_through" in item:
            overrides["mmf_deposit_pass_through"] = {
                "mode": "fixed_fraction",
                "value": float(item["mmf_deposit_pass_through"]),
            }
        if "opening_mmf_added_on_rrp_bil" in family:
            overrides[PRIVATE_SUBBUCKET_MMF_ADDED] = {
                "settlement_route": item.get("mmf_added_position_settlement", "covered_deposit_settled"),
                "opening_on_rrp_bil": float(family["opening_mmf_added_on_rrp_bil"]),
            }
        elif "mmf_added_position_settlement" in item:
            raise ValueError(f"{item['role']}: added money-fund settlement needs a common opening ON RRP balance")
        if item.get("holder_position_target"):
            if shift or "holder_preferences" in overrides or item.get("fed_holdings_change"):
                raise ValueError(f"{item['role']}: a holder position target is the scenario's only holdings change")
            if runs_root is None or output_dir is None or simulation is None:
                raise ValueError(
                    f"{item['role']}: a holder position target needs the baseline run, an output directory "
                    "and the family's simulation interval"
                )
            overrides["holder_position_target"] = _write_holder_target(item, simulation, runs_root, output_dir)
        if item.get("fed_holdings_change"):
            if output_dir is None:
                raise ValueError(f"{item['role']}: a Fed holdings change needs an output directory")
            overrides["fed_holdings"] = _write_fed_holdings_path(item, baseline, output_dir)
        if shift:
            multiplier = shift["multiplier"]
            overrides["holder_preferences"] = {
                "mode": "static_shares",
                "rows": shifted_holder_rows(
                    base,
                    holder=str(shift["holder"]),
                    multiplier=dict(multiplier) if isinstance(multiplier, dict) else float(multiplier),
                    absorbers={str(k): float(v) for k, v in shift["absorbers"].items()},
                ),
            }
        if item.get("key_rate_shocks_bp") and item.get("rate_path"):
            raise ValueError(f"{item['role']}: declare a key-rate shock or a rate path, not both")
        if item.get("key_rate_shocks_bp"):
            overrides["nominal_yield_curve"] = nominal_curve_override(item["key_rate_shocks_bp"])
        if item.get("rate_path"):
            if runs_root is None or output_dir is None or not shift:
                raise ValueError(f"{item['role']}: a rate path needs a holder shift, source runs and an output directory")
            overrides["nominal_yield_curve"] = _write_rate_path(item, runs_root, output_dir)
        scenario = _scenario(
            baseline,
            scenario_id=str(item["scenario_id"]),
            label=str(item["label"]),
            simulation={"frequency": "daily", **simulation} if simulation else None,
            coupling=_baseline_coupling(),
            overrides=overrides,
        )
        imposed_rates = item.get("key_rate_shocks_bp") or item.get("rate_path")
        scenario["provenance"] = {
            "kind": "external_source_assumption" if imposed_rates else "user_stress_assumption",
            "label": str(item["label"]),
            "prepared_by": "TDCSim scenario family writer",
            "as_of_date": str(family["as_of_date"]),
            "notes": str(item.get("notes") or family.get("description") or ""),
        }
        if family.get("funding_closure_mode"):
            scenario["funding_closure"] = {"mode": str(family["funding_closure_mode"])}
        spec = CboScenarioSpec.from_mapping(scenario)
        spec.assert_baseline_matches(baseline)
        scenarios[f"{index:02d}_{item['role']}.json"] = spec.data
    return scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-spec", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=PHASES, default="quantity")
    parser.add_argument("--runs-root", type=Path, help="completed family runs (rates and positions phases)")
    args = parser.parse_args(argv)
    family = json.loads(args.family_spec.read_text(encoding="utf-8"))
    baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
    if args.phase == "quantity":
        if args.output_dir.exists():
            raise SystemExit(f"output directory already exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    elif not args.output_dir.is_dir() or args.runs_root is None:
        raise SystemExit("the rates and positions phases write into the quantity directory and need --runs-root")
    written = family_scenarios(
        baseline, family, phase=args.phase, runs_root=args.runs_root, output_dir=args.output_dir
    )
    for filename, data in written.items():
        if (args.output_dir / filename).exists():
            raise SystemExit(f"scenario already exists: {args.output_dir / filename}")
        write_json(args.output_dir / filename, data)
        print(f"{args.output_dir / filename} {canonical_json_sha256(data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
