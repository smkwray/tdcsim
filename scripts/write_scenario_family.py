#!/usr/bin/env python3
"""Write a scenario family's CBO scenario specs from a committed family spec.

Each scenario may scale one holder's auction-absorption share by a multiplier
(one number, or one per marketable security type) and reassign the released
share to named absorbers in fixed proportions. A scenario may move the
bill-share target by dated changes from the package's opening bill share, and
may name the holders whose bill holdings absorb that change (the others keep
their baseline bill holdings in dollars). A scenario may also impose a
nominal curve shift: either a constant key-rate shock, or a dated rate path
derived from a completed quantity-only run of the family, in which the 10-year
shift at each curve date is proportional to the shifted holder's shortfall as a
share of marketable debt at that date. Baseline shares and curves are read from
the release-bound baseline package or the family's own runs, never restated here.

``--phase quantity`` writes the scenarios without a rate path; ``--phase rates``
writes the rate-path scenarios into the same directory once their source runs
have completed.
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

from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec  # noqa: E402
from tdcsim_cbo._json import canonical_json_sha256, sha256_file, write_json  # noqa: E402
from tdc_shared import HOLDER_TYPES, MARKETABLE_PREFERENCE_CATEGORIES  # noqa: E402
from issuance_calibration import CALIBRATION_FILE, ISSUANCE_RULE_MODE, load_calibration, target_for_date  # noqa: E402
from write_cbo_example_scenarios import _baseline_coupling, _scenario  # noqa: E402

FAMILY_SPEC_SCHEMA = "tdcsim_scenario_family_spec_v1"
HOLDER_PROFILE_FILE = "tdcsim_holder_profile_assumptions.csv"
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
    """Dated bill shares under which only ``absorbers`` take up the change in the bill stock.

    Auction shares apply to all bills sold, and bills roll within a year, so a holder's
    share of bill auctions is close to its share of the bill stock. With baseline shares
    s, opening bill share b and target t, every other holder keeps its baseline bill
    holdings in dollars (share s * b / t), and the absorbers split the change t - b in
    proportion to their baseline shares. Rows are written monthly from the first to the
    last target date so the shares follow the target path.
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
    if phase not in {"quantity", "rates"}:
        raise ValueError("phase must be quantity or rates")
    base = baseline_holder_shares(baseline)
    base_bill_share = None
    scenarios: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(family["scenarios"]):
        if bool(item.get("rate_path")) != (phase == "rates"):
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
                overrides["holder_preferences"] = {
                    "mode": "dated_static_shares",
                    "rows": bill_holder_rows(
                        base["bills"],
                        base_target=base_bill_share,
                        target_path=target_path,
                        absorbers=[str(h) for h in item["bill_holder_shift"]["absorbers"]],
                    ),
                }
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
            simulation=None,
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
    parser.add_argument("--phase", choices=("quantity", "rates"), default="quantity")
    parser.add_argument("--runs-root", type=Path, help="completed family runs (rates phase)")
    args = parser.parse_args(argv)
    family = json.loads(args.family_spec.read_text(encoding="utf-8"))
    baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
    if args.phase == "quantity":
        if args.output_dir.exists():
            raise SystemExit(f"output directory already exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True)
    elif not args.output_dir.is_dir() or args.runs_root is None:
        raise SystemExit("the rates phase writes into the quantity directory and needs --runs-root")
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
