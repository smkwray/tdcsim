"""Hold public holders' Treasury positions on a dated target by secondary trades in bills.

A scenario may require each public holder's controlled public marketable Treasury stock
(par, adjusted principal for TIPS) to equal a dated target at every period close: the
holder's realized closing stock in a matched baseline run plus a stated increment. The
increments sum to zero, so the targets sum to the baseline public's stock. Where the
scenario's own debt differs from the baseline's (a position that changes the Federal
Reserve's income changes its remittances, and so Treasury's borrowing), the day's
difference D, the scenario public stock less the baseline's, is held by the holders who give
up the position, in their declared donor weights; the buyer and every other route keep
their targets exactly. Before the intervention starts D must be zero. A route the controller
leaves untraded (ordinary money funds when the added position is the buyer) must match the
baseline by itself; when D is above tolerance, the extra or reduced issuance reaching it at
auction is traded back with the donors, in their donor weights, as a correction.

After the day's Federal Reserve operation, holders above target sell nominal bills to
holders below target, each seller's excess split across the buyers in proportion to their
shortfalls. A seller's sale is spread across remaining-maturity buckets in proportion to
its eligible bills; within a bucket whole lots go first, in maturity order, and only the
last lot needed is split. A transferred lot changes beneficial holder and principal
settlement route together, keeps its contractual terms and its original issue proceeds
(scaled when a lot is split), and settles at the model dirty price. The caller books the
settlement cash.

A target the controller cannot meet raises: a holder without enough eligible bills, a
target vector that does not sum to the stock the public holds, positions that differ from
the baseline before the intervention starts, or a miss after trading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from issuance_calibration import bill_mask
from mmf_eligibility import mmf_may_acquire
from simulation_calendar import previous_securities_business_day
from tdc_shared import (
    INTRAGOV_HOLDERS,
    MMF_SUBBUCKETS,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
    PRIVATE_SUBBUCKET_MMF_ADDED,
    PRIVATE_SUBBUCKETS,
    PUBLIC_MARKETABLE_SECURITY_TYPES,
    TRANSFER_SCALABLE_COLUMNS,
)

HOLDER_TARGET_MODE = "baseline_relative_stock_v1"
BILL_TARGET_MODE = "baseline_relative_bill_stock_v1"
# Every public holder of controlled marketable debt other than the Federal Reserve, whose
# stock follows its own path. Order fixes the buyer order when a seller's lots are divided.
PARTICIPANTS = (
    ("Banks", ""),
    ("Foreign", ""),
    ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK),
    ("Private", PRIVATE_SUBBUCKET_MMF),
    ("Private", PRIVATE_SUBBUCKET_MMF_ADDED),
)
PARTICIPANT_LABELS = {
    ("Banks", ""): "Banks",
    ("Foreign", ""): "Foreign",
    ("Private", PRIVATE_SUBBUCKET_DOMESTIC_NONBANK): "PrivateDomesticNonbank",
    ("Private", PRIVATE_SUBBUCKET_MMF): "PrivateMMF",
    ("Private", PRIVATE_SUBBUCKET_MMF_ADDED): "PrivateMMFAdded",
}
PHASES = ("pre", "build", "hold")
TRANSFER_EVENT_TYPE = "holder_target_secondary_transfer"
SETTLEMENT_LEG = "holder_target_secondary_settlement"
SETTLEMENT_SCOPES = {
    "build": "holder_position_target_build",
    "hold": "holder_position_target_hold",
}
# Remaining-maturity buckets for the representative bill mix, in days: (0, 30], (30, 91],
# (91, 182] and longer.
BUCKET_EDGES_DAYS = (30, 91, 182)
# Gaps and lot remainders this small are floating-point dust, not positions to trade.
DUST_BIL = 1.0e-9

PHASE_COLUMN = "HolderTargetPhase"
# The scenario public stock less the baseline's at the close; recorded in stock-target runs.
DEBT_DIFFERENCE_COLUMN = "HolderTargetDebtDifference"
# Net face and cash the untraded routes bought from the donors at the close (negative: sold).
CORRECTION_FACE_COLUMN = "HolderTargetCorrectionFace"
CORRECTION_CASH_COLUMN = "HolderTargetCorrectionCash"
STOCK_TARGET_COLUMNS = (DEBT_DIFFERENCE_COLUMN, CORRECTION_FACE_COLUMN, CORRECTION_CASH_COLUMN)
SUMMARY_COLUMNS = (
    "HolderTargetPreTradeMaxGap",
    "HolderTargetMaxError",
    "HolderTargetTradeFace",
    "HolderTargetTradeCash",
    "HolderTargetDepositEffect",
    "HolderTargetLotSplits",
)
PARTICIPANT_COLUMN_STEMS = (
    "HolderTarget",
    "HolderTargetBoughtFace",
    "HolderTargetSoldFace",
    "HolderTargetBoughtCash",
    "HolderTargetSoldCash",
)
RESULT_COLUMNS = (
    *SUMMARY_COLUMNS,
    *(f"{stem}_{PARTICIPANT_LABELS[p]}" for stem in PARTICIPANT_COLUMN_STEMS for p in PARTICIPANTS),
)
STOCK_COLUMNS = {
    stem: {p: f"{stem}_{PARTICIPANT_LABELS[p]}" for p in PARTICIPANTS}
    for stem in ("HolderStock", "HolderBillStock")
}
BILL_TARGET_FILE_COLUMNS = ("date", "holder_type", "holder_subbucket", "baseline_stock_bil")


class HolderTargetError(RuntimeError):
    """A holder position target cannot be met as declared."""


@dataclass(frozen=True)
class HolderTargetPath:
    """Dated phase and target stock of every participant, keyed by period close."""

    tolerance_bil: float
    rows: Mapping[pd.Timestamp, tuple[str, dict[tuple[str, str], float]]]
    bill_change_weights: Mapping[tuple[str, str], float] | None = None
    frozen_routes: tuple[tuple[str, str], ...] = ()
    # Stock targets: each donor's share of the position it gives up, which it also holds of D.
    donor_shares: Mapping[tuple[str, str], float] = field(default_factory=dict)

    @property
    def stock_column_stem(self) -> str:
        return "HolderBillStock" if self.bill_change_weights is not None else "HolderStock"

    def for_date(
        self, day: Any, stocks: Mapping[tuple[str, str], float]
    ) -> tuple[str, dict[tuple[str, str], float]]:
        key = pd.Timestamp(day).normalize()
        try:
            phase, reference = self.rows[key]
        except KeyError:
            raise HolderTargetError(f"no holder position target for the {key.date()} close") from None
        if self.bill_change_weights is None:
            return phase, self.stock_targets(key, phase, reference, stocks)[0]
        delta = math.fsum(stocks.values()) - math.fsum(reference.values())
        targets = {p: reference[p] + delta * self.bill_change_weights.get(p, 0.0) for p in PARTICIPANTS}
        for p, target in targets.items():
            if target < 0.0:
                raise HolderTargetError(f"{PARTICIPANT_LABELS[p]} bill target on {key.date()} is negative: {target}")
        return phase, targets


    def stock_targets(
        self, day: pd.Timestamp, phase: str, reference: Mapping[tuple[str, str], float],
        stocks: Mapping[tuple[str, str], float],
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Stock targets with the day's debt difference D held by the donors, and D."""

        difference = math.fsum(stocks.values()) - math.fsum(reference.values())
        if phase == "pre":
            if abs(difference) > self.tolerance_bil:
                raise HolderTargetError(
                    f"the public holds {difference:.9f} billion more than the matched baseline on {day.date()}, "
                    "before the intervention starts"
                )
            return dict(reference), difference
        if not self.donor_shares:
            if abs(difference) > self.tolerance_bil:
                raise HolderTargetError(
                    f"the public holds {difference:.9f} billion more than the matched baseline on {day.date()} "
                    "and the target names no donors to hold it"
                )
            return dict(reference), difference
        targets = {p: reference[p] + difference * self.donor_shares.get(p, 0.0) for p in PARTICIPANTS}
        for p, target in targets.items():
            if target < 0.0:
                raise HolderTargetError(f"{PARTICIPANT_LABELS[p]} target on {day.date()} is negative: {target}")
        return targets, difference

    def debt_difference(self, day: Any, stocks: Mapping[tuple[str, str], float]) -> float:
        _, reference = self.rows[pd.Timestamp(day).normalize()]
        return math.fsum(stocks.values()) - math.fsum(reference.values())

    def correction_flows(
        self, gaps: Mapping[tuple[str, str], float], difference: float, *, current_date: Any,
    ) -> list[tuple[tuple[str, str], tuple[str, str], float]]:
        """Seller-to-buyer face that returns each untraded route to its target with the donors.

        An untraded route may differ from its target only through D: with D above tolerance it
        trades its whole gap with the donors in their donor weights; otherwise a gap above
        tolerance raises and a smaller one is left alone."""

        flows = []
        for route in self.frozen_routes:
            gap = float(gaps[route])
            if abs(difference) <= self.tolerance_bil or not self.donor_shares:
                if abs(gap) > self.tolerance_bil:
                    raise HolderTargetError(
                        f"Untraded route {route} differs from the baseline at the "
                        f"{pd.Timestamp(current_date).date()} close."
                    )
                continue
            if abs(gap) <= DUST_BIL:
                continue
            for donor in PARTICIPANTS:
                if donor in self.donor_shares:
                    amount = abs(gap) * self.donor_shares[donor]
                    flows.append((route, donor, amount) if gap < 0.0 else (donor, route, amount))
        return flows


def bill_change_weights(rows: list[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
    """Positive shares of the change in non-Fed bills. Unnamed routes keep their baseline bill stock."""
    weights = {}
    for row in rows:
        route = (str(row["holder_type"]), str(row["holder_subbucket"]))
        weight = float(row["weight"])
        if route not in PARTICIPANTS or route in weights or not math.isfinite(weight) or weight <= 0.0:
            raise HolderTargetError(f"invalid or repeated bill change weight for {route}")
        weights[route] = weight
    if not math.isclose(math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise HolderTargetError("bill change weights must sum to one")
    return weights


def holder_target_path(config: Mapping[str, Any] | None) -> HolderTargetPath | None:
    """Validate the engine's ``holder_position_target`` block; None when absent."""

    if config is None:
        return None
    mode = config.get("mode")
    if mode not in (HOLDER_TARGET_MODE, BILL_TARGET_MODE):
        raise HolderTargetError(f"unknown holder position target mode {mode!r}")
    bills = mode == BILL_TARGET_MODE
    weights = bill_change_weights(config["bill_change_weights"]) if bills else None
    tolerance = float(config.get("tolerance_bil", float("nan")))
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise HolderTargetError("holder position target tolerance must be positive")
    dated: dict[pd.Timestamp, tuple[str, dict[tuple[str, str], float]]] = {}
    displacement: dict[tuple[str, str], float] = {}
    for row in config.get("rows") or ():
        day = pd.Timestamp(row["date"]).normalize()
        phase = "hold" if bills else str(row["phase"])
        holder = (str(row["holder_type"]), str(row.get("holder_subbucket") or ""))
        if phase not in PHASES:
            raise HolderTargetError(f"unknown holder target phase {phase!r} on {day.date()}")
        if holder not in PARTICIPANTS:
            raise HolderTargetError(f"holder target names {holder}, which is not a public holder route")
        target = float(row["baseline_stock_bil" if bills else "target_stock_bil"])
        if not math.isfinite(target) or target < 0.0:
            raise HolderTargetError(f"holder target for {holder} on {day.date()} must be finite and nonnegative")
        existing_phase, targets = dated.setdefault(day, (phase, {}))
        if existing_phase != phase:
            raise HolderTargetError(f"holder targets on {day.date()} name more than one phase")
        if holder in targets:
            raise HolderTargetError(f"holder target for {holder} on {day.date()} is repeated")
        targets[holder] = target
        if not bills and row.get("displacement_weight") not in (None, ""):
            weight = float(row["displacement_weight"])
            if displacement.setdefault(holder, weight) != weight:
                raise HolderTargetError(f"displacement weight for {holder} changes across dates")
    if not dated:
        raise HolderTargetError("holder position target has no dated rows")
    for day, (_, targets) in dated.items():
        if set(targets) != set(PARTICIPANTS):
            raise HolderTargetError(f"holder targets on {day.date()} must name every public holder route")
    frozen_routes = ()
    if config.get("buyer_private_route") == PRIVATE_SUBBUCKET_MMF_ADDED:
        frozen_routes = (("Private", PRIVATE_SUBBUCKET_MMF),)
    # The donors give up the position in proportion to their weights; they hold D the same way.
    donor_shares = {route: -weight for route, weight in displacement.items() if weight < 0.0}
    if donor_shares and not math.isclose(math.fsum(donor_shares.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise HolderTargetError("donor displacement weights must sum to minus one")
    return HolderTargetPath(tolerance_bil=tolerance, rows=dated, bill_change_weights=weights,
                            frozen_routes=frozen_routes, donor_shares=donor_shares)


def check_stock_target_run(
    results: pd.DataFrame,
    stock_closure: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    target_path: "BuildAndHoldPath",
    weights: Mapping[tuple[str, str], float],
) -> float:
    """Rebuild a stock-target run's every close and fail unless the run follows it.

    Targets come from the dated baseline stocks, the declared path and weights, and D from the
    post-trade stock closure: the public stock less the baseline's. The closes after the
    opening must be exactly the dated target sequence. Each must have finite inputs, finite
    nonnegative trade legs, reported stocks equal to the closure, a recorded D and reported
    targets within the declared tolerance of the rebuilt ones (donors hold D from the build
    on, as the controller sets them; before it every target is the baseline's), stocks within
    tolerance of their targets, and corrections only of untraded routes, only with D above
    tolerance, matching those routes' trades. Returns the largest absolute D."""

    path = holder_target_path(config)
    if path is None or path.bill_change_weights is not None:
        return 0.0
    rows = list(config.get("rows") or ())
    check_target_rows(rows, path=target_path, weights=weights)
    tolerance = path.tolerance_bil
    shares = {route: -weight for route, weight in weights.items() if weight < 0.0}
    baseline: dict[pd.Timestamp, dict[tuple[str, str], float]] = {}
    for row in rows:
        baseline.setdefault(pd.Timestamp(row["date"]).normalize(), {})[
            (str(row["holder_type"]), str(row.get("holder_subbucket") or ""))
        ] = float(row["baseline_stock_bil"])
    closing = closing_holder_stocks(stock_closure)
    stocks_by_day: dict[pd.Timestamp, dict[tuple[str, str], float]] = {}
    for record in closing.itertuples(index=False):
        stocks_by_day.setdefault(pd.Timestamp(record.period_end).normalize(), {})[
            (record.holder_sector, record.holder_subsector)
        ] = float(record.closing_stock_bil)
    closes = [pd.Timestamp(day).normalize() for day in results.index[1:]]  # the controller acts after the opening
    dated = sorted(baseline)
    if closes != dated:
        missing, extra = sorted(set(dated) - set(closes)), sorted(set(closes) - set(dated))
        if missing:
            raise HolderTargetError(f"the run has no close on {missing[0].date()}, a dated target")
        raise HolderTargetError(
            "the run's closes after the opening are not the dated target sequence"
            + (f": {extra[0].date()} has no target" if extra else "")
        )
    legs = ("BoughtFace", "SoldFace", "BoughtCash", "SoldCash")
    largest = 0.0
    for day, (_, row) in zip(closes, results.iloc[1:].iterrows(), strict=True):
        if day not in baseline or set(baseline[day]) != set(PARTICIPANTS):
            raise HolderTargetError(f"no complete holder target for the {day.date()} close")
        if day not in stocks_by_day:
            raise HolderTargetError(f"the stock closure has no {day.date()} close")
        reference = {p: baseline[day][p] + target_path.increment(day) * weights[p] for p in PARTICIPANTS}
        stocks = stocks_by_day[day]
        values = [*reference.values(), *stocks.values(), float(row[DEBT_DIFFERENCE_COLUMN]),
                  float(row[CORRECTION_FACE_COLUMN]), float(row[CORRECTION_CASH_COLUMN])]
        if not all(math.isfinite(value) for value in values):
            raise HolderTargetError(f"the {day.date()} close has a nonfinite target input")
        for p in PARTICIPANTS:
            label = PARTICIPANT_LABELS[p]
            reported_stock = float(row[STOCK_COLUMNS["HolderStock"][p]])
            if not math.isfinite(reported_stock) or abs(reported_stock - stocks[p]) > tolerance:
                raise HolderTargetError(f"{label} reported stock on {day.date()} is not its stock closure")
            for stem in legs:
                amount = float(row[f"HolderTarget{stem}_{label}"])
                if not math.isfinite(amount) or amount < 0.0:
                    raise HolderTargetError(f"{label} {stem} on {day.date()} is not a finite nonnegative amount")
        difference = math.fsum(stocks.values()) - math.fsum(reference.values())
        if abs(float(row[DEBT_DIFFERENCE_COLUMN]) - difference) > tolerance:
            raise HolderTargetError(f"the recorded debt difference on {day.date()} is not the public stock less the baseline's")
        pre = target_path.phase(day) == "pre"
        if pre and abs(difference) > tolerance:
            raise HolderTargetError(f"the public stock differs from the baseline on {day.date()}, before the intervention starts")
        for p in PARTICIPANTS:
            label = PARTICIPANT_LABELS[p]
            target = reference[p] + (0.0 if pre else difference * shares.get(p, 0.0))
            reported = float(row[f"HolderTarget_{label}"])
            if not math.isfinite(reported) or abs(reported - target) > tolerance:
                raise HolderTargetError(f"{label} target on {day.date()} does not follow the debt-difference rule")
            if abs(stocks[p] - target) > tolerance:
                raise HolderTargetError(f"{label} misses its target on {day.date()}")
        face, cash = float(row[CORRECTION_FACE_COLUMN]), float(row[CORRECTION_CASH_COLUMN])
        if path.frozen_routes:
            traded = [
                tuple(float(row[f"HolderTarget{stem}_{PARTICIPANT_LABELS[route]}"]) for route in path.frozen_routes)
                for stem in legs
            ]
            bought, sold, bought_cash, sold_cash = (math.fsum(values) for values in traded)
            if min(bought, sold) > DUST_BIL or abs(face - (bought - sold)) > tolerance \
                    or abs(cash - (bought_cash - sold_cash)) > tolerance:
                raise HolderTargetError(f"the {day.date()} correction is not the untraded routes' trades")
        if (abs(face) > DUST_BIL or abs(cash) > DUST_BIL) and (not path.frozen_routes or abs(difference) <= tolerance):
            raise HolderTargetError(f"a correction trades on {day.date()} without a debt difference to correct")
        largest = max(largest, abs(difference))
    return largest


def closing_stocks(portfolio: pd.DataFrame) -> dict[str, dict[tuple[str, str], float]]:
    """Controlled public marketable stock and nominal bill face by participant."""

    stocks = {participant: 0.0 for participant in PARTICIPANTS}
    bills = stocks.copy()
    result = {"HolderStock": stocks, "HolderBillStock": bills}
    if portfolio is None or portfolio.empty:
        return result
    holder = portfolio["HolderType"].fillna("").astype(str)
    security = portfolio["SecurityType"].fillna("").astype(str)
    mask = (
        portfolio["Status"].astype(str).eq("Active")
        & security.isin(PUBLIC_MARKETABLE_SECURITY_TYPES)
        & ~holder.isin(INTRAGOV_HOLDERS | {"CB"})
    )
    if not mask.any():
        return result
    rows = portfolio.loc[mask]
    face = pd.to_numeric(rows["FaceValue"], errors="coerce")
    adjusted = pd.to_numeric(rows["AdjustedPrincipal"], errors="coerce").fillna(face)
    base = face.mask(security[mask].eq("TIPS"), adjusted)
    if base.isna().any():
        raise HolderTargetError("a public holding has no par or adjusted principal")
    subbucket = rows["HolderSubBucket"].fillna("").astype(str).replace({"<NA>": "", "nan": "", "None": ""})
    private = holder[mask].eq("Private")
    if (private & ~subbucket.isin(PRIVATE_SUBBUCKETS)).any():
        raise HolderTargetError("a Private holding has no money-fund or deposit-funded route")
    if (~private & subbucket.ne("")).any():
        raise HolderTargetError("a non-Private public holding names a private route")
    bill = bill_mask(rows)
    grouped = pd.DataFrame({"stock": base, "bill": face.where(bill, 0.0)}).groupby([holder[mask], subbucket]).sum()
    for (name, route), (stock, bill_face) in grouped.iterrows():
        key = (str(name), str(route))
        if key not in stocks:
            raise HolderTargetError(f"public holder {key} is not a holder target route")
        stocks[key] = float(stock)
        bills[key] = float(bill_face)
    return result


def holder_stocks(portfolio: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Controlled public marketable stock: par, adjusted principal for TIPS."""
    return closing_stocks(portfolio)["HolderStock"]


def stock_result_values(stocks: Mapping[str, Mapping[tuple[str, str], float]]) -> dict[str, float]:
    return {column: stocks[stem][p] for stem, columns in STOCK_COLUMNS.items() for p, column in columns.items()}


def target_gaps(
    stocks: Mapping[tuple[str, str], float],
    targets: Mapping[tuple[str, str], float],
    *,
    tolerance: float,
    current_date: Any,
) -> dict[tuple[str, str], float]:
    """Target less current stock for each participant; the gaps must sum to zero."""

    gaps = {participant: float(targets[participant]) - float(stocks[participant]) for participant in PARTICIPANTS}
    imbalance = sum(gaps.values())
    if abs(imbalance) > tolerance:
        raise HolderTargetError(
            f"holder targets on {pd.Timestamp(current_date).date()} sum to {sum(targets.values()):.9f} billion "
            f"but the public holds {sum(stocks.values()):.9f} billion; the controller does not absorb a "
            "debt discrepancy"
        )
    return gaps


def plan_transfers(gaps: Mapping[tuple[str, str], float]) -> list[tuple[tuple[str, str], tuple[str, str], float]]:
    """Seller-to-buyer face amounts: each seller's excess, divided in proportion to the shortfalls."""

    buyers = [(p, gaps[p]) for p in PARTICIPANTS if gaps[p] > DUST_BIL]
    sellers = [(p, -gaps[p]) for p in PARTICIPANTS if gaps[p] < -DUST_BIL]
    if not buyers or not sellers:
        raise HolderTargetError("a holder target gap has no counterparty")
    total_need = sum(need for _, need in buyers)
    return [(seller, buyer, excess * need / total_need) for seller, excess in sellers for buyer, need in buyers]


# Terms that make two lots economically different, in the order they rank lots. Per-face
# ratios stand in for the quantities a split scales, so the pieces of one lot rank equal and
# stay adjacent; BondID only breaks ties among lots whose terms all agree.
_LOT_TERM_COLUMNS = (
    "MaturityDate", "_route", "TDCPrincipalHolderType", "TDCPrincipalHolderSubBucket", "SecurityType",
    "CouponRate", "FixedSpread", "IssueDate", "DatedDate", "FirstInterestPaymentDate", "InterestPaymentFrequency",
    "OriginalMaturityYears", "MaturityCategory", "ReferenceCPI_Issue",
)
_LOT_RATIO_COLUMNS = ("IssueProceeds", "OriginalPrincipal", "AdjustedPrincipal", "AccruedInterest_FRN")


def economic_lot_order(lots: pd.DataFrame) -> list[Any]:
    """Row labels of ``lots`` ordered by maturity, holder route and contractual terms.

    The order does not depend on row position or on how a holding is split into lots, so a
    rule that takes lots in this order treats economically identical books identically.
    """

    route_rank = {"": 0, **{route: position + 1 for position, route in enumerate(PRIVATE_SUBBUCKETS)}}
    keys = pd.DataFrame(index=lots.index)
    keys["MaturityDate"] = pd.to_datetime(lots["MaturityDate"], errors="coerce")
    holder = lots["HolderType"].fillna("").astype(str)
    subbucket = lots["HolderSubBucket"].fillna("").astype(str).replace({"<NA>": "", "nan": "", "None": ""})
    keys["_route"] = [f"{name}|{route_rank.get(route, len(route_rank))}|{route}" for name, route in zip(holder, subbucket)]
    for column in _LOT_TERM_COLUMNS[2:]:
        if column not in lots.columns:
            continue
        values = lots[column]
        if column in {"IssueDate", "DatedDate", "FirstInterestPaymentDate"}:
            keys[column] = pd.to_datetime(values, errors="coerce")
        elif column in {"CouponRate", "FixedSpread", "InterestPaymentFrequency", "OriginalMaturityYears", "ReferenceCPI_Issue"}:
            keys[column] = pd.to_numeric(values, errors="coerce")
        else:
            keys[column] = values.fillna("").astype(str).replace({"<NA>": "", "nan": "", "None": ""})
    face = pd.to_numeric(lots["FaceValue"], errors="coerce")
    for column in _LOT_RATIO_COLUMNS:
        if column in lots.columns:
            keys[f"_{column}_per_face"] = (pd.to_numeric(lots[column], errors="coerce") / face).round(12)
    keys["_bond"] = pd.to_numeric(lots["BondID"], errors="coerce")
    return list(keys.sort_values(list(keys.columns), kind="mergesort", na_position="last").index)


def _bill_lots(portfolio: pd.DataFrame, seller: tuple[str, str], current_date: pd.Timestamp) -> pd.DataFrame:
    holder, route = seller
    owned = portfolio["Status"].astype(str).eq("Active") & portfolio["HolderType"].fillna("").astype(str).eq(holder)
    if holder == "Private":
        owned &= portfolio["HolderSubBucket"].fillna("").astype(str).eq(route)
    face = pd.to_numeric(portfolio["FaceValue"], errors="coerce")
    maturity = pd.to_datetime(portfolio["MaturityDate"], errors="coerce")
    eligible = owned & bill_mask(portfolio) & (maturity > current_date) & (face > DUST_BIL)
    # The economic order starts with maturity, so it also orders the buckets.
    lots = portfolio.loc[economic_lot_order(portfolio.loc[eligible]), ["MaturityDate", "FaceValue"]].copy()
    lots["FaceValue"] = face[lots.index]
    days = (pd.to_datetime(lots["MaturityDate"]) - current_date).dt.days.to_numpy()
    lots["_bucket"] = np.searchsorted(np.asarray(BUCKET_EDGES_DAYS), days, side="left")
    return lots


def _select_lots(
    portfolio: pd.DataFrame,
    seller: tuple[str, str],
    amount: float,
    current_date: pd.Timestamp,
) -> list[list[Any]]:
    """[index, face] pieces summing to ``amount``, spread over remaining-maturity buckets."""

    lots = _bill_lots(portfolio, seller, current_date)
    available = float(lots["FaceValue"].sum())
    if amount > available + DUST_BIL:
        raise HolderTargetError(
            f"{PARTICIPANT_LABELS[seller]} holds {available:.6f} billion of eligible bills on "
            f"{current_date.date()}, short of the {amount:.6f} billion its sale requires "
            f"(shortfall {amount - available:.6f})"
        )
    by_bucket = lots.groupby("_bucket", sort=True)["FaceValue"].sum()
    buckets = [int(b) for b, stock in by_bucket.items() if stock > DUST_BIL]
    pieces: list[list[Any]] = []
    assigned = 0.0
    for position, bucket in enumerate(buckets):
        share = amount - assigned if position == len(buckets) - 1 else amount * float(by_bucket[bucket]) / available
        assigned += share
        remaining = share
        for index, face in lots.loc[lots["_bucket"] == bucket, "FaceValue"].items():
            if remaining <= DUST_BIL:
                break
            take = float(face) if float(face) <= remaining + DUST_BIL else remaining
            pieces.append([index, take])
            remaining -= take
        if remaining > DUST_BIL:
            raise HolderTargetError(
                f"{PARTICIPANT_LABELS[seller]} bill bucket {bucket} cannot supply its share on {current_date.date()}"
            )
    return pieces


@dataclass
class RebalanceReport:
    trade_face: float = 0.0
    trade_cash: float = 0.0
    lot_splits: int = 0
    pairs: dict[tuple[tuple[str, str], tuple[str, str]], list[float]] = field(default_factory=dict)
    bought_face: dict[tuple[str, str], float] = field(default_factory=lambda: {p: 0.0 for p in PARTICIPANTS})
    sold_face: dict[tuple[str, str], float] = field(default_factory=lambda: {p: 0.0 for p in PARTICIPANTS})
    bought_cash: dict[tuple[str, str], float] = field(default_factory=lambda: {p: 0.0 for p in PARTICIPANTS})
    sold_cash: dict[tuple[str, str], float] = field(default_factory=lambda: {p: 0.0 for p in PARTICIPANTS})

    def add(self, other: "RebalanceReport") -> None:
        """Fold another trade step of the same close into this report."""

        self.trade_face += other.trade_face
        self.trade_cash += other.trade_cash
        self.lot_splits += other.lot_splits
        for pair, (face, cash) in other.pairs.items():
            totals = self.pairs.setdefault(pair, [0.0, 0.0])
            totals[0] += face
            totals[1] += cash
        for name in ("bought_face", "sold_face", "bought_cash", "sold_cash"):
            for participant, value in getattr(other, name).items():
                getattr(self, name)[participant] += value


def rebalance_holder_positions(
    portfolio: pd.DataFrame,
    gaps: Mapping[tuple[str, str], float],
    *,
    current_date: Any,
    price: Callable[[pd.Series, float], Mapping[str, float]],
    row_budget: int | None = None,
    record_mmf_acquisition: Callable[[Any, str, Any], None] | None = None,
    flows: list[tuple[tuple[str, str], tuple[str, str], float]] | None = None,
) -> tuple[pd.DataFrame, RebalanceReport]:
    """Close ``gaps`` by moving nominal bills from holders above target to holders below it,
    or move the given seller-to-buyer ``flows`` instead.

    ``price(row, face)`` returns the dirty settlement value of ``face`` of the lot in ``row``.
    """

    day = pd.Timestamp(current_date).normalize()
    flows = plan_transfers(gaps) if flows is None else flows
    report = RebalanceReport()
    pieces: list[tuple[Any, float, tuple[str, str], tuple[str, str]]] = []
    for seller in PARTICIPANTS:
        quotas = [(buyer, amount) for source, buyer, amount in flows if source == seller]
        if not quotas:
            continue
        queue = _select_lots(portfolio, seller, sum(amount for _, amount in quotas), day)
        for position, (buyer, quota) in enumerate(quotas):
            last = position == len(quotas) - 1
            while queue and (last or quota > DUST_BIL):
                index, available = queue[0]
                take = available if last or available <= quota + DUST_BIL else quota
                if buyer[0] == "Private" and buyer[1] in MMF_SUBBUCKETS:
                    lot = portfolio.loc[index]
                    if not mmf_may_acquire(lot["SecurityType"], lot["MaturityDate"], day):
                        raise HolderTargetError(
                            f"{PARTICIPANT_LABELS[buyer]} transfer includes an ineligible bill on {day.date()}"
                        )
                pieces.append((index, take, seller, buyer))
                quota -= take
                if take >= available - DUST_BIL:
                    queue.pop(0)
                else:
                    queue[0][1] = available - take
    numeric_ids = pd.to_numeric(portfolio["BondID"], errors="coerce")
    next_bond_id = int(numeric_ids.max()) + 1 if numeric_ids.notna().any() else 1
    new_rows = []
    holder_columns = ["HolderType", "HolderSubBucket", "TDCPrincipalHolderType", "TDCPrincipalHolderSubBucket"]
    for index, take, seller, buyer in pieces:
        row = portfolio.loc[index].copy()
        if buyer[0] == "Private" and buyer[1] in MMF_SUBBUCKETS and record_mmf_acquisition is not None:
            record_mmf_acquisition(day, row["SecurityType"], row["MaturityDate"])
        face = float(row["FaceValue"])
        pricing = price(row, take)
        cash = float(pricing["dirty_value"])
        whole = take >= face - DUST_BIL
        scale_to_full = face / take
        portfolio.loc[index, "TimeToMaturity"] = pricing["time_to_maturity"]
        portfolio.loc[index, "DiscountYield"] = pricing["discount_yield"]
        portfolio.loc[index, "CleanPrice"] = pricing["clean_value"] * scale_to_full
        portfolio.loc[index, "AccruedInterest"] = pricing["accrued_interest"] * scale_to_full
        portfolio.loc[index, "DirtyValue"] = cash * scale_to_full
        portfolio.loc[index, "DirtyPriceRatio"] = cash / take
        if whole:
            portfolio.loc[index, holder_columns] = [buyer[0], buyer[1], buyer[0], buyer[1]]
        else:
            fraction = take / face
            moved = portfolio.loc[index].copy()
            moved["BondID"] = next_bond_id
            next_bond_id += 1
            moved[holder_columns] = [buyer[0], buyer[1], buyer[0], buyer[1]]
            for column in TRANSFER_SCALABLE_COLUMNS:
                if column in portfolio.columns and pd.notna(portfolio.loc[index, column]):
                    original = float(portfolio.loc[index, column])
                    moved[column] = original * fraction
                    portfolio.loc[index, column] = original * (1.0 - fraction)
            new_rows.append(moved)
            report.lot_splits += 1
        pair = report.pairs.setdefault((seller, buyer), [0.0, 0.0])
        pair[0] += take
        pair[1] += cash
        report.trade_face += take
        report.trade_cash += cash
        report.sold_face[seller] += take
        report.sold_cash[seller] += cash
        report.bought_face[buyer] += take
        report.bought_cash[buyer] += cash
    if new_rows:
        if row_budget is not None and len(portfolio) + len(new_rows) > int(row_budget):
            raise RuntimeError("Portfolio-row budget would be exceeded by a holder-target transfer split.")
        portfolio = pd.concat([portfolio, pd.DataFrame(new_rows)], ignore_index=True)
    return portfolio, report


TARGET_FILE_COLUMNS = (
    "date",
    "phase",
    "holder_type",
    "holder_subbucket",
    "baseline_stock_bil",
    "displacement_weight",
    "target_increment_bil",
    "target_stock_bil",
)


@dataclass(frozen=True)
class BuildAndHoldPath:
    """Added position: zero at the origin close, linear to ``size_bil`` at ``build_end``, then held."""

    origin: pd.Timestamp
    build_end: pd.Timestamp
    hold_through: pd.Timestamp
    size_bil: float

    @classmethod
    def from_points(cls, points: list[Mapping[str, Any]]) -> "BuildAndHoldPath":
        if len(points) != 3 or any(set(point) != {"date", "increment_bil"} for point in points):
            raise HolderTargetError("target_path must be three points: origin, end of build, end of hold")
        dates = [pd.Timestamp(point["date"]).normalize() for point in points]
        values = [float(point["increment_bil"]) for point in points]
        if not dates[0] < dates[1] <= dates[2]:
            raise HolderTargetError("target_path dates must be origin < end of build <= end of hold")
        if values[0] != 0.0 or values[1] != values[2] or not math.isfinite(values[1]) or values[1] <= 0.0:
            raise HolderTargetError("target_path must start at zero and hold one positive size after the build")
        return cls(origin=dates[0], build_end=dates[1], hold_through=dates[2], size_bil=values[1])

    def increment(self, day: Any) -> float:
        day = pd.Timestamp(day).normalize()
        if day > self.hold_through:
            raise HolderTargetError(f"no holder target is declared after {self.hold_through.date()}")
        # The position builds on securities business days only: a close when the market is
        # shut keeps the last business day's increment, when the baseline does not move
        # either, and a build ending on a closed day is complete the business day before.
        day, origin, build_end = (
            pd.Timestamp(previous_securities_business_day(value))
            for value in (day, self.origin, self.build_end)
        )
        if day <= origin:
            return 0.0
        if day >= build_end:
            return self.size_bil
        return self.size_bil * (day - origin).days / (build_end - origin).days

    def phase(self, day: Any) -> str:
        day = pd.Timestamp(day).normalize()
        if day <= self.origin:
            return "pre"
        return "build" if day <= self.build_end else "hold"


def closing_holder_stocks(stock_closure: pd.DataFrame, *, bills_only: bool = False) -> pd.DataFrame:
    """Participants' closing controlled public marketable stock by date from a run's stock closure.

    ``stock_closure`` is a run's ``tdcsim_period_stock_closure`` table; the result has one row
    per closing date and participant (zero where a participant holds nothing).
    """

    frame = stock_closure.copy()
    for column in ("holder_sector", "holder_subsector", "instrument_type", "maturity_bucket"):
        frame[column] = frame[column].fillna("").astype(str)
    frame = frame[(frame["axis"] == "holder") & (frame["debt_scope"] == "controlled_public_marketable")]
    if bills_only:
        frame = frame[(frame["instrument_type"] == "Fixed") & (frame["maturity_bucket"] == "bills")]
    public = frame[frame["holder_sector"].isin({holder for holder, _ in PARTICIPANTS})]
    unknown = set(zip(public["holder_sector"], public["holder_subsector"])) - set(PARTICIPANTS)
    if unknown:
        raise HolderTargetError(f"stock closure names holder routes outside the target set: {sorted(unknown)}")
    grouped = public.groupby(["period_end", "holder_sector", "holder_subsector"])["closing_debt_stock_bil"].sum()
    dates = sorted(frame["period_end"].unique())
    index = pd.MultiIndex.from_tuples(
        [(day, holder, route) for day in dates for holder, route in PARTICIPANTS],
        names=["period_end", "holder_sector", "holder_subsector"],
    )
    return grouped.reindex(index, fill_value=0.0).rename("closing_stock_bil").reset_index()


def displacement_weights(
    origin_bill_stocks: Mapping[tuple[str, str], float],
    *,
    buyer: tuple[str, str],
    donors: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """Each participant's share of the added position: +1 for the buyer, minus each donor's
    share of the donors' bill stock at the origin, zero for every other route."""

    if buyer not in PARTICIPANTS or not donors or buyer in donors or len(set(donors)) != len(donors):
        raise HolderTargetError("a holder target needs one buyer route and distinct donor routes")
    if set(donors) - set(PARTICIPANTS):
        raise HolderTargetError(f"unknown donor routes {sorted(set(donors) - set(PARTICIPANTS))}")
    total = sum(float(origin_bill_stocks[donor]) for donor in donors)
    if not total > 0.0:
        raise HolderTargetError("the donors hold no bills at the origin")
    weights = {participant: 0.0 for participant in PARTICIPANTS}
    weights[buyer] = 1.0
    for donor in donors:
        weights[donor] = -float(origin_bill_stocks[donor]) / total
    return weights


def declared_weights(override: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    """Weights stated by a scenario's holder target, checked against their origin bill stocks.

    The buyer, the donors and the unchanged routes must together name every participant
    once, and each donor's weight must be minus its share of the donors' origin bill stock.
    """

    buyer = (str(override["buyer_holder"]), str(override["buyer_private_route"]))
    donors = {
        (str(row["holder_type"]), str(row["holder_subbucket"])): (float(row["origin_bill_stock_bil"]), float(row["weight"]))
        for row in override["displacement"]
    }
    unchanged = [(str(row["holder_type"]), str(row["holder_subbucket"])) for row in override["unchanged_routes"]]
    named = [buyer, *donors, *unchanged]
    if len(named) != len(set(named)) or set(named) != set(PARTICIPANTS) or len(donors) != len(override["displacement"]):
        raise HolderTargetError("the buyer, donors and unchanged routes must name every public holder route once")
    if buyer[1] == PRIVATE_SUBBUCKET_MMF_ADDED and ("Private", PRIVATE_SUBBUCKET_MMF) not in unchanged:
        raise HolderTargetError("ordinary money funds must be unchanged when the added position is the buyer")
    weights = displacement_weights({route: stock for route, (stock, _) in donors.items()} | {
        route: 0.0 for route in PARTICIPANTS if route not in donors
    }, buyer=buyer, donors=list(donors))
    for route, (_, weight) in donors.items():
        if abs(weight - weights[route]) > 1.0e-12:
            raise HolderTargetError(f"displacement weight for {route} is not its share of the donors' origin bills")
    return weights


def target_rows(
    baseline_stocks: pd.DataFrame,
    *,
    path: BuildAndHoldPath,
    weights: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Dated target rows: each participant's baseline closing stock plus its increment."""

    rows = []
    for record in baseline_stocks.sort_values("period_end", kind="mergesort").itertuples(index=False):
        day = pd.Timestamp(record.period_end).normalize()
        if day > path.hold_through:
            continue
        participant = (str(record.holder_sector), str(record.holder_subsector))
        increment = path.increment(day) * weights[participant]
        target = float(record.closing_stock_bil) + increment
        if target < 0.0:
            raise HolderTargetError(
                f"{PARTICIPANT_LABELS[participant]} would need {target:.6f} billion on {day.date()}; "
                "its baseline holdings cannot supply its share of the shift"
            )
        rows.append(
            {
                "date": day.date().isoformat(),
                "phase": path.phase(day),
                "holder_type": participant[0],
                "holder_subbucket": participant[1],
                "baseline_stock_bil": float(record.closing_stock_bil),
                "displacement_weight": float(weights[participant]),
                "target_increment_bil": increment,
                "target_stock_bil": target,
            }
        )
    order = {participant: position for position, participant in enumerate(PARTICIPANTS)}
    rows.sort(key=lambda row: (row["date"], order[(row["holder_type"], row["holder_subbucket"])]))
    if not rows or pd.Timestamp(rows[-1]["date"]) != path.hold_through:
        raise HolderTargetError(f"the baseline does not close on {path.hold_through.date()}, the end of the hold")
    return rows


def check_target_rows(rows: list[Mapping[str, Any]], *, path: BuildAndHoldPath, weights: Mapping[tuple[str, str], float]) -> None:
    """Fail unless every row's phase, increment and target follow the declared path and weights."""

    for row in rows:
        participant = (str(row["holder_type"]), str(row.get("holder_subbucket") or ""))
        if participant not in weights:
            raise HolderTargetError(f"target row names unknown route {participant}")
        day = pd.Timestamp(row["date"])
        expected = path.increment(day) * weights[participant]
        if str(row["phase"]) != path.phase(day):
            raise HolderTargetError(f"target row phase on {day.date()} disagrees with the declared path")
        if abs(float(row["displacement_weight"]) - weights[participant]) > 1.0e-12:
            raise HolderTargetError(f"target row weight for {participant} disagrees with the declared displacement")
        if abs(float(row["target_increment_bil"]) - expected) > 1.0e-9:
            raise HolderTargetError(f"target row increment for {participant} on {day.date()} disagrees with the path")
        if abs(float(row["baseline_stock_bil"]) + float(row["target_increment_bil"]) - float(row["target_stock_bil"])) > 1.0e-9:
            raise HolderTargetError(f"target row for {participant} on {day.date()} is not baseline plus increment")


def result_values(
    *,
    phase: str,
    targets: Mapping[tuple[str, str], float],
    stocks: Mapping[tuple[str, str], float],
    pre_trade_gap: float,
    report: RebalanceReport,
    deposit_effect: float,
) -> dict[str, float]:
    """Engine result columns for one period close (the phase goes in ``PHASE_COLUMN``)."""

    values = {
        "HolderTargetPreTradeMaxGap": pre_trade_gap,
        "HolderTargetMaxError": max(abs(stocks[p] - targets[p]) for p in PARTICIPANTS),
        "HolderTargetTradeFace": report.trade_face,
        "HolderTargetTradeCash": report.trade_cash,
        "HolderTargetDepositEffect": deposit_effect,
        "HolderTargetLotSplits": float(report.lot_splits),
    }
    for participant in PARTICIPANTS:
        label = PARTICIPANT_LABELS[participant]
        values[f"HolderTarget_{label}"] = float(targets[participant])
        values[f"HolderTargetBoughtFace_{label}"] = report.bought_face[participant]
        values[f"HolderTargetSoldFace_{label}"] = report.sold_face[participant]
        values[f"HolderTargetBoughtCash_{label}"] = report.bought_cash[participant]
        values[f"HolderTargetSoldCash_{label}"] = report.sold_cash[participant]
    return values


__all__ = [
    "CORRECTION_CASH_COLUMN",
    "CORRECTION_FACE_COLUMN",
    "DEBT_DIFFERENCE_COLUMN",
    "HOLDER_TARGET_MODE",
    "PARTICIPANTS",
    "STOCK_TARGET_COLUMNS",
    "check_stock_target_run",
    "PARTICIPANT_LABELS",
    "PHASES",
    "PHASE_COLUMN",
    "RESULT_COLUMNS",
    "SETTLEMENT_LEG",
    "SETTLEMENT_SCOPES",
    "TRANSFER_EVENT_TYPE",
    "HolderTargetError",
    "HolderTargetPath",
    "RebalanceReport",
    "holder_stocks",
    "holder_target_path",
    "plan_transfers",
    "rebalance_holder_positions",
    "result_values",
    "target_gaps",
]
