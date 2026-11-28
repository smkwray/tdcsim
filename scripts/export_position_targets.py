#!/usr/bin/env python3
"""Export the consumer tables of holder-position-target scenario families.

Each family named on the command line has a baseline and scenarios that hold one buyer's
Treasury holdings a stated amount above the baseline's. For every such scenario the export
compares the run with its own family baseline, at the end of the build and of the hold,
measured from the origin close, and writes (USD billions):

- position_targets.csv: each holder route's displacement weight and amount, target
  increment, share and amount of the debt difference D it holds, and realized holdings
  increment at both dates, and its largest target miss over every close;
- dated_effects.csv: cumulative TDC and TRC since the origin close in the scenario and the
  baseline, and their difference, and the change in the public's controlled marketable stock,
  whose difference is D;
- cash_components.csv: the TDC difference split into net other security settlements
  (principal to deposit users, auction absorption and secondary settlements, the last split
  into the target trades and the Federal Reserve's background trades), Treasury interest
  credited to deposit users, fiscal and other flows and a reconciliation residual, and the
  TRC difference split into the holder-routed reserve path and the route-held balances;
- controller_turnover.csv: the target trades by phase and holder route, gross and net of
  the build, with the difference in the Federal Reserve's purchase cash and the corrections
  that return the untraded ordinary money funds to their target when D reaches them at
  auction (net and gross face, net cash, and days), and the largest gap left at a close when the
  securities market is shut;
- receipt.json: producer, inputs, baseline references, clock, stock basis, funding states
  and the checks below.

D is the scenario public stock less the baseline's at a close. A position that changes the
Federal Reserve's income changes its remittances and so Treasury's borrowing; the donors hold
that difference in their donor weights, and every other route keeps its stated target.

The export fails unless every run passed bounded replay verification, each scenario names
its family baseline's run, the paired runs are identical through the origin close, the
recorded D is the public stock less the baseline's at every close, Treasury's side matches
the baseline in a run whose D stays zero, every close meets its holder target with D held by
the donors, and the cash components and TRC reconcile. Run it with the installed release the runs were produced with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_open04_reproduction import _assert_installed_distribution_import  # noqa: E402

SCHEMA_VERSION = "tdcsim_position_target_export_v1"
RESULTS = Path("outputs/results_compact.csv.gz")
TRC = Path("outputs/tdcsim_trc_results.csv.gz")
STOCK_CLOSURE = Path("outputs/tdcsim_period_stock_closure.csv.gz")
RUN_MANIFEST = Path("tdcsim_cbo_run_manifest.json")
TARGET_FILE = Path("compile/compiled/forecast_inputs/tdcsim_holder_position_target.csv")
ROUTE_STATES = ("ONRRP_MMF", "ForeignOfficialFedDeposits", "ForeignOfficialFedRRP")
# Treasury's side of every paired run: debt, issuance, payments, financing cost, cash and
# the Federal Reserve's stock must match the baseline.
ISSUER_COLUMNS = (
    "TotalDebt_Agg", "NewDebtIssued", "AuctionProceeds", "InterestOutlay_Period", "PrincipalPaid_Bonds",
    "FinancingCost_Period", "TGA", "DebtHeld_CentralBank",
)
INTEREST_PARTS = {
    "bill_discount": "TDC_BillDiscountInterestToDU",
    "coupon": "TDC_CouponInterestToDU",
    "frn": "TDC_FRNInterestToDU",
    "tips_coupon": "TDC_TIPSCouponInterestToDU",
}
FED_DEPOSIT = ("CBOFedSecondaryPurchaseDepositEffect", "CBOFedSecondarySaleDepositEffect")
TRC_FAMILIES = ("FiscalFlow", "DebtService", "Buyback", "AuctionSettlement", "FedSecondary", "SecondaryTrades",
                "TTandL", "OtherTreasuryFed")
TOLERANCE_BIL = 1e-6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(run_dir: Path) -> dict[str, pd.DataFrame]:
    results = pd.read_csv(run_dir / RESULTS, keep_default_na=False, na_values=[""], float_precision="round_trip")
    results["Date"] = results["Date"].str[:10]
    trc = pd.read_csv(run_dir / TRC, float_precision="round_trip")
    trc["Date"] = trc["Date"].str[:10]
    return {
        "results": results.set_index("Date"),
        "trc": trc.set_index("Date"),
        "closure": pd.read_csv(run_dir / STOCK_CLOSURE, keep_default_na=False, float_precision="round_trip"),
    }


def _since(frame: pd.DataFrame, column: str, origin: str, day: str) -> float:
    window = frame.index[(frame.index > origin) & (frame.index <= day)]
    return float(frame.loc[window, column].sum())


def _verified(root: Path, role: str) -> dict[str, Any]:
    record = json.loads((root / role / "verification.json").read_text(encoding="utf-8"))
    if record.get("status") != "pass":
        raise SystemExit(f"{role}: bounded replay verification did not pass")
    return record


def _check_hierarchy(role: str, day: str, rows: list[tuple[str, str, str, str, str, float]]) -> None:
    """Fail unless each decomposition's additive rows, and each parent's "of which" rows, sum to it."""

    values = {(obj, decomposition, name): value for obj, decomposition, _, _, name, value in rows}
    sums: dict[tuple[str, str, str], float] = {}
    for obj, decomposition, kind, parent, _, value in rows:
        if kind != "total":
            sums[(obj, decomposition, parent)] = sums.get((obj, decomposition, parent), 0.0) + value
    for key, total in sums.items():
        if abs(values[key] - total) > TOLERANCE_BIL:
            raise SystemExit(f"{role}: {key[2]} ({key[1]}) does not equal the sum of its components on {day}")


def _case(family: dict[str, Any], root: Path, role: str, base: dict[str, Any], ht: Any) -> dict[str, Any]:
    from settlement_routes import MmfAddedPosition
    from tdc_shared import MMF_ADDED_TDC_COLUMNS, PRIVATE_SUBBUCKET_MMF_ADDED

    run_dir = root / role / "run"
    scenario = json.loads((run_dir / "scenario.json").read_text(encoding="utf-8"))
    target = scenario["overrides"]["holder_position_target"]
    added_buyer = target["buyer_private_route"] == PRIVATE_SUBBUCKET_MMF_ADDED
    added_position = MmfAddedPosition.from_mapping(scenario["overrides"][PRIVATE_SUBBUCKET_MMF_ADDED]) if added_buyer else None
    reference = target["baseline_reference"]
    base_dir = root / "baseline" / "run"
    if (reference["run_manifest_sha256"] != sha256(base_dir / RUN_MANIFEST)
            or reference["stock_closure_sha256"] != sha256(base_dir / STOCK_CLOSURE)):
        raise SystemExit(f"{role}: the scenario's baseline reference is not this family's baseline run")
    if sha256(run_dir / TARGET_FILE) != target["file"]["sha256"]:
        raise SystemExit(f"{role}: the compiled target file is not the scenario's")
    path = ht.BuildAndHoldPath.from_points(target["target_path"])
    origin, build_end, end = (day.date().isoformat() for day in (path.origin, path.build_end, path.hold_through))
    data = _load(run_dir)
    results, base_results = data["results"], base["results"]
    if list(results.index) != list(base_results.index):
        raise SystemExit(f"{role}: the scenario and baseline close on different dates")

    # Identical through the origin close; Treasury's side identical throughout.
    shared = [c for c in base_results.columns if pd.api.types.is_numeric_dtype(base_results[c])]
    pre = results.index <= origin
    pre_gap = float((results.loc[pre, shared] - base_results.loc[pre, shared]).abs().max().max())
    pre_trc_gap = float((data["trc"].loc[pre] - base["trc"].loc[pre]).abs().max().max())
    base_pre = base["closure"][base["closure"]["period_end"] <= origin].reset_index(drop=True)
    case_pre = data["closure"][data["closure"]["period_end"] <= origin].reset_index(drop=True)
    numeric = [c for c in base_pre.columns if c.endswith("_bil")]
    keys = [c for c in base_pre.columns if c not in numeric]
    same_closure = (
        base_pre.shape == case_pre.shape
        and base_pre[keys].astype(str).equals(case_pre[keys].astype(str))
        and float((base_pre[numeric].astype(float) - case_pre[numeric].astype(float)).abs().max().max()) == 0.0
    )
    if pre_gap != 0.0 or pre_trc_gap != 0.0 or not same_closure:
        raise SystemExit(f"{role}: the paired runs differ before the origin close ({pre_gap}, {pre_trc_gap})")
    # Treasury's side differs from the baseline only through the debt difference D.
    def public(frame: pd.DataFrame) -> pd.Series:
        return sum(frame[ht.STOCK_COLUMNS["HolderStock"][p]].astype(float) for p in ht.PARTICIPANTS)

    difference = results[ht.DEBT_DIFFERENCE_COLUMN].astype(float)
    identity_gap = float(((public(results) - public(base_results)) - difference).abs().max())
    if identity_gap > TOLERANCE_BIL:
        raise SystemExit(f"{role}: the recorded debt difference is not the public stock less the baseline's ({identity_gap})")
    max_difference = float(difference.abs().max())
    issuer = {c: float((results[c] - base_results[c]).abs().max()) for c in ISSUER_COLUMNS}
    if max_difference <= TOLERANCE_BIL and max(issuer.values()) > TOLERANCE_BIL:
        raise SystemExit(f"{role}: Treasury's side differs from the baseline with no debt difference: {issuer}")

    # Holder targets at every close, from the run's own stock closure.
    rows = pd.read_csv(run_dir / TARGET_FILE, keep_default_na=False, float_precision="round_trip")
    realized = ht.closing_holder_stocks(data["closure"]).rename(
        columns={"period_end": "date", "holder_sector": "holder_type", "holder_subsector": "holder_subbucket"})
    checked = rows.merge(realized, on=["date", "holder_type", "holder_subbucket"], how="left", validate="one_to_one")
    if checked["closing_stock_bil"].isna().any() or len(checked) != len(rows):
        raise SystemExit(f"{role}: a target close has no realized holdings")
    checked["debt_difference_bil"] = checked["date"].map(difference)
    if checked["debt_difference_bil"].isna().any():
        raise SystemExit(f"{role}: a target close has no recorded debt difference")
    weight = checked["displacement_weight"].astype(float)
    checked["debt_difference_share"] = (-weight).where(weight < 0.0, 0.0)
    checked["debt_difference_held"] = checked["debt_difference_bil"] * checked["debt_difference_share"]
    checked["error"] = checked["closing_stock_bil"] - (checked["target_stock_bil"] + checked["debt_difference_held"])
    # Exact at market-day closes; with the securities market shut, TIPS indexation may leave a
    # gap that the next market-day close trades back.
    market_day = checked["date"].map(results[ht.MARKET_OPEN_COLUMN]) == 1.0
    checked["error"] = checked["error"].where(market_day, 0.0)
    if checked["error"].abs().max() > float(target["target_tolerance_bil"]):
        raise SystemExit(f"{role}: holdings miss their target by {checked['error'].abs().max()}")
    checked["realized_increment"] = checked["closing_stock_bil"] - checked["baseline_stock_bil"]
    base_stocks = ht.closing_holder_stocks(base["closure"]).rename(
        columns={"period_end": "date", "holder_sector": "holder_type", "holder_subsector": "holder_subbucket",
                 "closing_stock_bil": "baseline_realized"})
    checked = checked.merge(base_stocks, on=["date", "holder_type", "holder_subbucket"], how="left")
    if (checked["baseline_realized"] - checked["baseline_stock_bil"]).abs().max() > 1e-9:
        raise SystemExit(f"{role}: the target file's baseline stocks are not the baseline run's")

    donors = {(d["holder_type"], d["holder_subbucket"]): d for d in target["displacement"]}
    positions = []
    for (holder, route), group in checked.groupby(["holder_type", "holder_subbucket"], sort=False):
        at = group.set_index("date")
        donor = donors.get((holder, route))
        positions.append({
            "family_id": family["family_id"], "role": role, "holder_type": holder, "holder_subbucket": route,
            "position_role": "buyer" if at["displacement_weight"].iloc[0] == 1.0 else ("donor" if donor else "unchanged"),
            "origin_bill_stock_bil": float(donor["origin_bill_stock_bil"]) if donor else float("nan"),
            "displacement_weight": float(at["displacement_weight"].iloc[0]),
            "displacement_amount_bil": float(at["displacement_weight"].iloc[0]) * path.size_bil,
            "debt_difference_share": float(at["debt_difference_share"].iloc[0]),
            "target_increment_build_end_bil": float(at.loc[build_end, "target_increment_bil"]),
            "debt_difference_held_build_end_bil": float(at.loc[build_end, "debt_difference_held"]),
            "realized_increment_build_end_bil": float(at.loc[build_end, "realized_increment"]),
            "target_increment_end_bil": float(at.loc[end, "target_increment_bil"]),
            "debt_difference_held_end_bil": float(at.loc[end, "debt_difference_held"]),
            "realized_increment_end_bil": float(at.loc[end, "realized_increment"]),
            "max_abs_target_error_bil": float(at["error"].abs().max()),
        })

    effects, components = [], []
    trc, base_trc = data["trc"], base["trc"]
    interest_parts = {
        name: results[f"{column}_DomesticNonbank"] + results[f"{column}_MMF"] for name, column in INTEREST_PARTS.items()
    }
    base_interest_parts = {
        name: base_results[f"{column}_DomesticNonbank"] + base_results[f"{column}_MMF"]
        for name, column in INTEREST_PARTS.items()
    }
    if (sum(interest_parts.values()) - results["TDC_InterestToDU"]).abs().max() > TOLERANCE_BIL:
        raise SystemExit(f"{role}: the interest components do not sum to Treasury interest to deposit users")
    public_stock, base_public_stock = public(results), public(base_results)
    for day in (build_end, end):
        def diff(frame, other, column):
            return _since(frame, column, origin, day) - _since(other, column, origin, day)

        stock_change = float(public_stock[day] - public_stock[origin])
        base_stock_change = float(base_public_stock[day] - base_public_stock[origin])
        effects.append({
            "family_id": family["family_id"], "role": role, "date": day, "measure": "public_marketable_stock",
            "scenario_since_origin_bil": stock_change, "baseline_since_origin_bil": base_stock_change,
            "scenario_minus_baseline_bil": stock_change - base_stock_change,
            "scenario_from_opening_bil": float(public_stock[day] - public_stock.iloc[0]),
            "baseline_from_opening_bil": float(base_public_stock[day] - base_public_stock.iloc[0]),
        })

        for measure, frame, other, column in (("cumulative_tdc", results, base_results, "TDC_Change"),
                                              ("cumulative_trc", trc, base_trc, "TRC_Change")):
            scenario_value = _since(frame, column, origin, day)
            baseline_value = _since(other, column, origin, day)
            effects.append({
                "family_id": family["family_id"], "role": role, "date": day, "measure": measure,
                "scenario_since_origin_bil": scenario_value, "baseline_since_origin_bil": baseline_value,
                "scenario_minus_baseline_bil": scenario_value - baseline_value,
                "scenario_from_opening_bil": float(frame.loc[frame.index <= day, column].sum()),
                "baseline_from_opening_bil": float(other.loc[other.index <= day, column].sum()),
            })
        tdc = diff(results, base_results, "TDC_Change")
        principal = diff(results, base_results, "TDC_PrincipalToDU")
        auction = diff(results, base_results, "TDC_AuctionAbsorption")
        secondary = diff(results, base_results, "TDC_SecondaryTrades")
        target_trades = _since(results, "HolderTargetDepositEffect", origin, day)
        fed = sum(diff(results, base_results, column) for column in FED_DEPOSIT)
        interest = diff(results, base_results, "TDC_InterestToDU")
        fiscal_other = diff(results, base_results, "TDC_FiscalFlow") + diff(results, base_results, "TDC_Other")
        window = (results.index > origin) & (results.index <= day)
        parts = {name: float(interest_parts[name][window].sum() - base_interest_parts[name][window].sum())
                 for name in INTEREST_PARTS}
        net_other = principal + auction + secondary
        residual = tdc - (net_other + interest + fiscal_other)
        trc_change = diff(trc, base_trc, "TRC_Change")
        legacy = diff(trc, base_trc, "LegacyReserveChange_HolderRouted")
        states = sum(float(trc.loc[day, s] - trc.loc[origin, s] - base_trc.loc[day, s] + base_trc.loc[origin, s])
                     for s in ROUTE_STATES)
        trc_residual = trc_change - (legacy - states)
        if abs(residual) > TOLERANCE_BIL or abs(secondary - target_trades - fed) > TOLERANCE_BIL \
                or abs(trc_residual) > TOLERANCE_BIL:
            raise SystemExit(f"{role}: the cash components or TRC do not reconcile on {day}")
        # (object, decomposition, row kind, parent, component, value). Additive rows sum to
        # their decomposition's total; "of which" rows sum to their parent.
        rows_out = [
            ("tdc", "settlement_type", "total", "", "tdc_change", tdc),
            ("tdc", "settlement_type", "additive", "tdc_change", "net_other_security_settlements", net_other),
            ("tdc", "settlement_type", "of_which", "net_other_security_settlements", "principal_to_deposit_users", principal),
            ("tdc", "settlement_type", "of_which", "net_other_security_settlements", "auction_absorption", auction),
            ("tdc", "settlement_type", "of_which", "net_other_security_settlements", "secondary_settlements", secondary),
            ("tdc", "settlement_type", "of_which", "secondary_settlements", "secondary_settlements_target_trades", target_trades),
            ("tdc", "settlement_type", "of_which", "secondary_settlements", "secondary_settlements_federal_reserve", fed),
            ("tdc", "settlement_type", "additive", "tdc_change", "treasury_interest_to_deposit_users", interest),
            *(("tdc", "settlement_type", "of_which", "treasury_interest_to_deposit_users",
               f"treasury_interest_to_deposit_users_{name}", value) for name, value in parts.items()),
            ("tdc", "settlement_type", "additive", "tdc_change", "fiscal_and_other", fiscal_other),
            ("tdc", "settlement_type", "additive", "tdc_change", "reconciliation_residual", residual),
            ("trc", "transaction_family", "total", "", "trc_change", trc_change),
            *(("trc", "transaction_family", "additive", "trc_change", f"trc_{name}", diff(trc, base_trc, f"TRC_{name}"))
              for name in TRC_FAMILIES),
            ("trc", "settlement_account", "total", "", "trc_change", trc_change),
            ("trc", "settlement_account", "additive", "trc_change", "holder_routed_reserve_change", legacy),
            ("trc", "settlement_account", "additive", "trc_change", "minus_route_held_balance_change", -states),
            ("trc", "settlement_account", "additive", "trc_change", "reconciliation_residual", trc_residual),
        ]
        if added_position is not None:
            label = ht.PARTICIPANT_LABELS[("Private", PRIVATE_SUBBUCKET_MMF_ADDED)]
            added_trade = added_position.deposit_share * (
                _since(results, f"HolderTargetSoldCash_{label}", origin, day)
                - _since(results, f"HolderTargetBoughtCash_{label}", origin, day)
            )
            added_secondary = diff(results, base_results, MMF_ADDED_TDC_COLUMNS["TDC_SecondaryTrades"])
            for name, total, added in (
                ("principal_to_deposit_users", principal, diff(results, base_results, MMF_ADDED_TDC_COLUMNS["TDC_PrincipalToDU"])),
                ("treasury_interest_to_deposit_users_bill_discount", parts["bill_discount"],
                 diff(results, base_results, MMF_ADDED_TDC_COLUMNS["TDC_BillDiscountInterestToDU"])),
                ("secondary_settlements_target_trades", target_trades, added_trade),
                ("secondary_settlements_federal_reserve", fed, added_secondary - added_trade),
            ):
                rows_out.extend([
                    ("tdc", "settlement_type", "of_which", name, f"{name}_added_mmf_position", added),
                    ("tdc", "settlement_type", "of_which", name, f"{name}_other_routes", total - added),
                ])
        _check_hierarchy(role, day, rows_out)
        components.extend({"family_id": family["family_id"], "role": role, "date": day, "object": obj,
                           "decomposition": decomposition, "row_kind": kind, "parent_component": parent,
                           "component": name, "scenario_minus_baseline_bil": value}
                          for obj, decomposition, kind, parent, name, value in rows_out)

    turnover = []
    for phase in ("build", "hold"):
        rows_phase = results[results["HolderTargetPhase"] == phase]
        base_phase = base_results.loc[rows_phase.index]
        net = path.size_bil if phase == "build" else 0.0
        common = {"family_id": family["family_id"], "role": role, "phase": phase,
                  "first_date": rows_phase.index.min(), "last_date": rows_phase.index.max()}
        for participant in ht.PARTICIPANTS:
            label = ht.PARTICIPANT_LABELS[participant]
            turnover.append({**common, "holder_type": participant[0], "holder_subbucket": participant[1],
                             **{f"{kind}_bil": float(rows_phase[f"HolderTarget{stem}_{label}"].sum())
                                for kind, stem in (("bought_face", "BoughtFace"), ("sold_face", "SoldFace"),
                                                   ("bought_cash", "BoughtCash"), ("sold_cash", "SoldCash"))}})
        gross = float(rows_phase["HolderTargetTradeFace"].sum())
        turnover.append({
            **common, "holder_type": "all", "holder_subbucket": "",
            "gross_face_bil": gross, "gross_cash_bil": float(rows_phase["HolderTargetTradeCash"].sum()),
            "net_target_change_bil": net, "turnover_above_net_target_change_bil": gross - net,
            "trade_days": int((rows_phase["HolderTargetTradeFace"] > 0.0).sum()),
            "lot_splits": int(rows_phase["HolderTargetLotSplits"].sum()),
            "correction_net_face_bil": float(rows_phase[ht.CORRECTION_FACE_COLUMN].sum()),
            "correction_gross_face_bil": float(rows_phase[ht.CORRECTION_FACE_COLUMN].abs().sum()),
            "correction_net_cash_bil": float(rows_phase[ht.CORRECTION_CASH_COLUMN].sum()),
            "correction_days": int((rows_phase[ht.CORRECTION_FACE_COLUMN] != 0.0).sum()),
            "closed_day_max_gap_bil": float(rows_phase[ht.CLOSED_DAY_GAP_COLUMN].max()),
            "fed_purchase_face_difference_bil": float(
                (rows_phase["CBOFedSecondaryPurchaseFace"] - base_phase["CBOFedSecondaryPurchaseFace"]).sum()),
            "fed_purchase_cash_difference_bil": float(
                (rows_phase["CBOFedSecondaryPurchaseCash"] - base_phase["CBOFedSecondaryPurchaseCash"]).sum()),
        })
    return {
        "positions": positions, "effects": effects, "components": components, "turnover": turnover,
        "checks": {
            "pre_origin_max_abs_difference": max(pre_gap, pre_trc_gap),
            "issuer_max_abs_difference_bil": issuer,
            "max_abs_debt_difference_bil": max_difference,
            "max_abs_daily_correction_face_bil": float(results[ht.CORRECTION_FACE_COLUMN].abs().max()),
            "max_closed_day_gap_bil": float(results[ht.CLOSED_DAY_GAP_COLUMN].max()),
            "debt_difference_identity_max_abs_bil": identity_gap,
            "max_abs_target_error_bil": float(checked["error"].abs().max()),
            "closes_checked": int(checked["date"].nunique()),
        },
        "inputs": {"scenario_sha256": sha256(run_dir / "scenario.json"), "run_manifest_sha256": sha256(run_dir / RUN_MANIFEST),
                   "target_file_sha256": target["file"]["sha256"], "verification": _verified(root, role).get("status")},
        "declaration": {k: v for k, v in target.items() if k not in {"file", "baseline_reference"}},
        "baseline_reference": reference,
        "dates": {"origin": origin, "build_end": build_end, "hold_end": end},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", nargs=3, action="append", required=True, metavar=("SPEC", "RUNS_ROOT", "ROUTING"))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    _assert_installed_distribution_import()
    import holder_targets as ht
    from tdcsim_cbo.runtime_identity import distribution_identity, installed_archive_sha256

    if subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True,
                      text=True).stdout.strip():
        raise SystemExit("export requires a clean committed producer checkout")
    if args.out_dir.exists():
        raise SystemExit(f"output directory already exists: {args.out_dir}")
    tables: dict[str, list[dict[str, Any]]] = {"positions": [], "effects": [], "components": [], "turnover": []}
    families = []
    for spec_path, runs_root, routing in args.family:
        family = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        root = Path(runs_root)
        base = _load(root / "baseline" / "run")
        base_scenario = json.loads((root / "baseline" / "run" / "scenario.json").read_text(encoding="utf-8"))
        cases = {}
        for item in family["scenarios"]:
            if not item.get("holder_position_target"):
                continue
            result = _case(family, root, item["role"], base, ht)
            for name in tables:
                tables[name].extend(result[name])
            cases[item["role"]] = {key: result[key] for key in ("checks", "inputs", "declaration", "baseline_reference", "dates")}
        families.append({
            "family_id": family["family_id"],
            "family_spec_sha256": sha256(Path(spec_path)),
            "trc_routing_sha256": sha256(Path(routing)),
            "trc_routing": json.loads(Path(routing).read_text(encoding="utf-8")),
            "mmf_deposit_pass_through": base_scenario["overrides"]["mmf_deposit_pass_through"]["value"],
            "simulation": base_scenario["simulation"],
            "baseline": {"scenario_id": base_scenario["scenario_id"], "verification": _verified(root, "baseline").get("status"),
                         "run_manifest_sha256": sha256(root / "baseline" / "run" / RUN_MANIFEST),
                         **({"routing": json.loads((root / "baseline" / "run" / "outputs" / "tdcsim_trc_metadata.json")
                                                    .read_text(encoding="utf-8"))["routing"]}
                            if "mmf_added_position" in base_scenario["overrides"] else {})},
            "cases": cases,
        })
    args.out_dir.mkdir(parents=True)
    names = {"positions": "position_targets.csv", "effects": "dated_effects.csv",
             "components": "cash_components.csv", "turnover": "controller_turnover.csv"}
    for key, name in names.items():
        pd.DataFrame(tables[key]).to_csv(args.out_dir / name, index=False, float_format="%.12f", lineterminator="\n")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True,
                            text=True).stdout.strip()
    dist = distribution_identity()
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "producer": {"repository_commit": commit, "installed_archive_sha256": installed_archive_sha256(),
                     "distribution_file_digest": dist["file_digest"], "package_version": dist["version"],
                     "python_version": platform.python_version()},
        "clock": "daily model period close; holder targets are met after the Federal Reserve's operation and before the close",
        "stock_basis": "controlled public marketable Treasury holdings at par, adjusted principal for TIPS",
        "readings": "scenario less its own family baseline, cumulative since the origin close, at the end of the build and of the hold",
        "interest_note": "Treasury interest to deposit users is Treasury's payment to covered deposits, including bill discount at redemption; secondary-price effects stay in security settlements",
        "settlement_note": (
            "The difference between dated TDC readings is the subsequent net Treasury-attributed cash settlement effect of "
            "maintaining the complete holdings vector under the stated controller and the synthetic Federal Reserve "
            "allocation rule. It includes maintenance transactions, changes in Federal Reserve settlements and explicit "
            "Treasury interest. It is not a measure of earned carry on the additional position or of its terminal market "
            "value. Unchanged total holdings of a route do not imply unchanged security composition or cash flows."
        ),
        "component_equations": {
            "tdc settlement_type": "tdc_change = net_other_security_settlements + treasury_interest_to_deposit_users + fiscal_and_other + reconciliation_residual",
            "net_other_security_settlements": "principal_to_deposit_users + auction_absorption + secondary_settlements",
            "secondary_settlements": "secondary_settlements_target_trades + secondary_settlements_federal_reserve",
            "treasury_interest_to_deposit_users": "sum of treasury_interest_to_deposit_users_<type>",
            "trc transaction_family": "trc_change = sum of trc_<family>",
            "trc settlement_account": "trc_change = holder_routed_reserve_change + minus_route_held_balance_change + reconciliation_residual",
            "rule": "row_kind additive rows sum to their decomposition's total; of_which rows sum to parent_component; the two TRC decompositions are alternatives and are not summed together",
        },
        "turnover_note": (
            "Gross face is the controller's transfers summed once per transfer. turnover_above_net_target_change_bil is "
            "gross face less the change in the buyer's target over the phase: during the build it is turnover above the "
            "net build, not an identification of maintenance intent; during the hold the target is constant, so it is all "
            "maintenance."
        ),
        "allocation_conventions": (
            "Each seller's sale is spread across remaining-maturity buckets (0-30, 31-91, 92-182 and over 182 days) in "
            "proportion to its eligible bills, whole lots first in order of maturity, holder route and terms; a seller's "
            "selected lots are then divided among buyers in the fixed order Banks, Foreign, other domestic nonbanks, ordinary "
            "money funds, and the added money-fund position."
        ),
        "families": families,
        "outputs": {name: sha256(args.out_dir / name) for name in names.values()},
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), **receipt["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
