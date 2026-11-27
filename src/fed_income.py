"""Federal Reserve earnings, balance sheet and weekly remittances for CBO runs.

Earnings are accrual basis on the Fed's own acquisition basis:

- Treasury coupons, TIPS coupons and FRN interest accrue daily. Earned interest is the cash
  the Fed receives plus the change in its accrued-interest receivable, so the receivable at
  the opening is paid down by the first coupon and purchased accrued interest is never
  income.
- Every Fed lot carries its own unamortized premium or discount: the clean acquisition
  price (the dirty price less accrued interest) less the stock acquired. It is amortized
  straight line to maturity, so a bill bought at 99 and redeemed at 100 earns 1 whatever
  Treasury received at issue. TIPS indexation counts only from acquisition onward.
- Inherited lots open with one approximation per class: bills keep their remaining issue
  discount, coupon securities share the rest of the H.4.1 net premium in proportion to
  stock.
- Agency MBS income is a smoothed proxy: coupon on a daily runoff path less premium
  amortization in proportion to runoff. Operating expenses, dividends and other income are
  smoothed daily proxies too; none of these has a payment clock.

Expenses accrue on each day's closing balances: IORB on reserves and on designated
financial market utilities' balances, the ON RRP rate on domestic ON RRP and on the foreign
official repo pool. Interest is posted on the next Fedwire business day, a simplified clock:
the Reserve Banks pay interest on reserve balances once per 14-day maintenance period, after
it ends. IORB and domestic ON RRP interest credit reserves. FMU interest credits FMU accounts,
whose balance follows the exogenous other-deposit path, so the FMUs are assumed to pass it on
to banks and it adds to reserves. Foreign repo pool interest is reinvested in the pool and
grows that liability, not reserves.

The pooled H.4.1 line "Earnings remittances due to the U.S. Treasury" is one signed balance:
net income after dividends adds to it, a debit is the deferred asset. The Reserve Banks remit
weekly (Financial Accounting Manual, paragraph 60.55): each Wednesday, or the next business
day after a Wednesday holiday, they pay the lower of the credit balances at the previous
Wednesday and on the day before payment, and nothing if either is zero or a debit. The first
payment of a calendar year uses the December 31 balance in place of the previous Wednesday's.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from simulation_calendar import is_fedwire_business_day, next_fedwire_business_day, previous_fedwire_business_day


REMITTANCE_WEEKDAY = 2  # Wednesday
INTEREST_EXPENSE_DAY_COUNT = 360.0  # IORB and ON RRP accrue actual/360.
BILL_MAX_YEARS = 1.0
TOLERANCE_BIL = 1e-9

# Results columns the CBO engine writes for the Fed. Flows are per period; balances close the day.
FED_INCOME_FLOW_COLUMNS = (
    "CB_CouponCash",
    "CB_TIPSCouponCash",
    "CB_FRNCash",
    "CB_CouponAccrual",
    "CB_TIPSCouponAccrual",
    "CB_FRNAccrual",
    "CB_TIPSIndexation",
    "CB_TIPSFloorTopup",
    "CB_BillDiscountAccretion",
    "CB_CouponPremiumAmortization",
    "CB_RealizedGains",
    "CB_TreasuryInterestIncome",
    "CB_AccruedInterestPurchased",
    "CB_AccruedInterestSold",
    "CB_PremiumAcquired",
    "CB_PremiumSold",
    "CB_MBSCouponIncome",
    "CB_MBSPremiumAmortization",
    "CB_MBSIncome",
    "CB_OtherIncome",
    "CB_IORBExpense",
    "CB_FMUInterestExpense",
    "CB_DomesticONRRPExpense",
    "CB_ForeignRepoExpense",
    "CB_OperatingExpenses",
    "CB_Dividends",
    "CB_InterestPosted",
    "CB_ForeignRepoInterestReinvested",
    "CB_TreasuryReserveEffect",
    "CBOTable43RemittanceInFiscalFlow",
)
FED_BALANCE_COLUMNS = (
    "CB_PolicyRatePct",
    "CB_IORBRatePct",
    "CB_ONRRPRatePct",
    "CB_TreasuryUnamortizedPremium",
    "CB_TreasuryAccruedInterest",
    "CB_MBSHoldings",
    "CB_MBSUnamortizedPremium",
    "CB_OtherAssets",
    "CB_Currency",
    "CB_DomesticONRRP",
    "CB_ForeignRepo",
    "CB_ForeignOfficialDeposits",
    "CB_OtherDeposits",
    "CB_FMUBalances",
    "CB_OtherLiabilitiesAndCapital",
    "CB_InterestPayable",
    "CB_ForeignRepoInterestPayable",
    "CB_EarningsRemittancesDue",
    "CB_ReserveBalances",
)
FED_INCOME_RESULT_COLUMNS = (*FED_INCOME_FLOW_COLUMNS, *FED_BALANCE_COLUMNS)

# The parts of net income, signed as they enter it.
NET_INCOME_TERMS = {
    "CB_TreasuryInterestIncome": 1.0,
    "CB_MBSIncome": 1.0,
    "CB_OtherIncome": 1.0,
    "CB_IORBExpense": -1.0,
    "CB_FMUInterestExpense": -1.0,
    "CB_DomesticONRRPExpense": -1.0,
    "CB_ForeignRepoExpense": -1.0,
    "CB_OperatingExpenses": -1.0,
    "CB_Dividends": -1.0,
}
TREASURY_INCOME_TERMS = (
    "CB_CouponAccrual",
    "CB_TIPSCouponAccrual",
    "CB_FRNAccrual",
    "CB_TIPSIndexation",
    "CB_TIPSFloorTopup",
    "CB_BillDiscountAccretion",
    "CB_CouponPremiumAmortization",
    "CB_RealizedGains",
)
ACCRUAL_TYPES = {"coupon": "CB_Coupon", "tips": "CB_TIPSCoupon", "frn": "CB_FRN"}

# Export names, shared by the visual ledger and the scenario-family fiscal-year table.
FED_INCOME_EXPORT_FLOWS = {
    "CB_TreasuryInterestIncome": "fed_treasury_income_accrued_bil",
    "CB_CouponAccrual": "fed_coupon_accrual_bil",
    "CB_TIPSCouponAccrual": "fed_tips_coupon_accrual_bil",
    "CB_FRNAccrual": "fed_frn_accrual_bil",
    "CB_TIPSIndexation": "fed_tips_indexation_bil",
    "CB_TIPSFloorTopup": "fed_tips_floor_topup_bil",
    "CB_BillDiscountAccretion": "fed_bill_discount_accretion_bil",
    "CB_CouponPremiumAmortization": "fed_coupon_premium_amortization_bil",
    "CB_RealizedGains": "fed_realized_gains_bil",
    "CB_MBSIncome": "fed_mbs_income_bil",
    "CB_MBSPremiumAmortization": "fed_mbs_premium_amortization_bil",
    "CB_OtherIncome": "fed_other_income_bil",
    "CB_IORBExpense": "fed_iorb_on_reserves_expense_bil",
    "CB_FMUInterestExpense": "fed_iorb_on_fmu_balances_expense_bil",
    "CB_DomesticONRRPExpense": "fed_domestic_on_rrp_expense_bil",
    "CB_ForeignRepoExpense": "fed_foreign_repo_pool_expense_bil",
    "CB_OperatingExpenses": "fed_operating_expenses_bil",
    "CB_Dividends": "fed_dividends_bil",
    "CB_NetIncome": "fed_net_income_bil",
}
FED_INCOME_EXPORT_LEVELS = {
    "CB_DeferredAsset": "fed_deferred_asset_bil",
    "CB_EarningsRemittancesDue": "fed_earnings_remittances_due_bil",
    "CB_ReserveBalances": "fed_reserve_balances_bil",
    "CB_DomesticONRRP": "fed_domestic_on_rrp_bil",
    "CB_ForeignRepo": "fed_foreign_repo_pool_bil",
    "CB_OtherDeposits": "fed_other_deposits_bil",
    "CB_FMUBalances": "fed_fmu_balances_bil",
    "CB_Currency": "fed_currency_bil",
    "CB_MBSHoldings": "fed_mbs_holdings_bil",
    "CB_OtherLiabilitiesAndCapital": "fed_other_liabilities_and_capital_bil",
    "CB_TreasuryUnamortizedPremium": "fed_treasury_unamortized_premium_bil",
    "CB_TreasuryAccruedInterest": "fed_treasury_accrued_interest_bil",
    "CB_PolicyRatePct": "policy_rate_pct",
    "CB_IORBRatePct": "iorb_rate_pct",
    "CB_ONRRPRatePct": "on_rrp_rate_pct",
}


class FedIncomeError(ValueError):
    """Raised when the Fed balance sheet or earnings cannot be booked consistently."""


# --- Weekly remittance rule -------------------------------------------------------------


def remittance_week(day) -> pd.Timestamp:
    """The Wednesday that starts the remittance week containing ``day``."""

    day = pd.Timestamp(day).normalize()
    return day - pd.Timedelta(days=(day.weekday() - REMITTANCE_WEEKDAY) % 7)


def is_remittance_day(day) -> bool:
    """Wednesday, or the next Fedwire business day when Wednesday is closed."""

    day = pd.Timestamp(day).normalize()
    return pd.Timestamp(next_fedwire_business_day(remittance_week(day).date())) == day


def first_remittance_day_of_year(year: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=1, day=1)
    wednesday = first + pd.Timedelta(days=(REMITTANCE_WEEKDAY - first.weekday()) % 7)
    return pd.Timestamp(next_fedwire_business_day(wednesday.date()))


def weekly_remittance(basis_balance: float, day_before_balance: float) -> float:
    """The lower of two credit balances, and nothing if either is zero or a debit."""

    if basis_balance <= 0.0 or day_before_balance <= 0.0:
        return 0.0
    return min(float(basis_balance), float(day_before_balance))


def remittance_basis_date(day) -> pd.Timestamp:
    """Whose balance sets the cap: December 31 for a year's first payment, else last Wednesday."""

    day = pd.Timestamp(day).normalize()
    if day == first_remittance_day_of_year(day.year):
        return pd.Timestamp(year=day.year - 1, month=12, day=31)
    return remittance_week(day) - pd.Timedelta(days=7)


# --- Accrued interest receivable --------------------------------------------------------


def _month_index_to_date(month_index: np.ndarray, day_of_month: np.ndarray, month_end: np.ndarray) -> np.ndarray:
    years = month_index // 12
    months = month_index % 12 + 1
    first = pd.to_datetime({"year": years, "month": months, "day": np.ones_like(years)})
    days_in_month = first.dt.days_in_month.to_numpy()
    days = np.where(month_end, days_in_month, np.minimum(day_of_month, days_in_month))
    return (first + pd.to_timedelta(days - 1, unit="D")).to_numpy()


def accrued_interest(rows: pd.DataFrame, day) -> pd.DataFrame:
    """Accrued interest per row at the close of ``day``, split by type.

    Fixed and TIPS coupons accrue on the engine's schedule: semiannual on the maturity
    day of month (month end when maturity is a month end), paid on the next Fedwire
    business day when that date is closed. A coupon still awaiting payment stays in the
    receivable. FRN interest is the engine's own accrued balance. Bills accrue nothing
    here; their discount is carried as a negative premium.
    """

    day = pd.Timestamp(day).normalize()
    out = pd.DataFrame(0.0, index=rows.index, columns=list(ACCRUAL_TYPES))
    if rows.empty:
        return out
    security = rows["SecurityType"].astype(str)
    coupon = pd.to_numeric(rows["CouponRate"], errors="coerce").fillna(0.0).to_numpy()
    frn = security.eq("FRN").to_numpy()
    out.loc[frn, "frn"] = pd.to_numeric(rows.loc[frn, "AccruedInterest_FRN"], errors="coerce").fillna(0.0)
    bearing = security.isin(["Fixed", "TIPS"]).to_numpy() & (coupon > TOLERANCE_BIL)
    if not bearing.any():
        return out
    sub = rows.loc[bearing]
    maturity = pd.to_datetime(sub["MaturityDate"]).dt.normalize()
    m_year = maturity.dt.year.to_numpy()
    m_month = maturity.dt.month.to_numpy()
    m_day = maturity.dt.day.to_numpy()
    month_end = maturity.dt.is_month_end.to_numpy()
    current = day.year * 12 + (day.month - 1)
    lag = (current - (m_year * 12 + (m_month - 1))) % 6
    prev_index = current - lag
    prev = _month_index_to_date(prev_index, m_day, month_end)
    later = prev > day.to_datetime64()
    prev_index = np.where(later, prev_index - 6, prev_index)
    prev = _month_index_to_date(prev_index, m_day, month_end)
    nxt = _month_index_to_date(prev_index + 6, m_day, month_end)
    dated = pd.to_datetime(sub["DatedDate"], errors="coerce").fillna(pd.to_datetime(sub["IssueDate"])).dt.normalize()
    start = np.maximum(prev, dated.to_numpy())
    span = (nxt - prev).astype("timedelta64[D]").astype(float)
    elapsed = np.clip((day.to_datetime64() - start).astype("timedelta64[D]").astype(float), 0.0, None)
    matured = prev >= maturity.to_numpy()
    fraction = np.where(matured, 0.0, np.minimum(elapsed / span, 1.0))
    tips = sub["SecurityType"].astype(str).eq("TIPS").to_numpy()
    base = np.where(
        tips,
        pd.to_numeric(sub["AdjustedPrincipal"], errors="coerce").fillna(sub["FaceValue"]).to_numpy(dtype=float),
        pd.to_numeric(sub["FaceValue"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
    )
    half = base * coupon[bearing] / 2.0
    # A coupon dated on a closed day is paid on the next business day: until then it is owed.
    issued = pd.to_datetime(sub["IssueDate"]).dt.normalize().to_numpy()
    unique_prev = np.unique(prev)
    settles = {value: np.datetime64(next_fedwire_business_day(pd.Timestamp(value).date()), "ns") for value in unique_prev}
    paid_by = np.array([settles[value] for value in prev], dtype="datetime64[ns]")
    pending = (paid_by > day.to_datetime64()) & (prev > issued) & (prev >= dated.to_numpy())
    accrued = half * fraction + np.where(pending, half, 0.0)
    out.loc[sub.index[~tips], "coupon"] = accrued[~tips]
    out.loc[sub.index[tips], "tips"] = accrued[tips]
    return out


def is_bill(rows: pd.DataFrame) -> pd.Series:
    years = pd.to_numeric(rows["OriginalMaturityYears"], errors="coerce").fillna(np.inf)
    coupon = pd.to_numeric(rows["CouponRate"], errors="coerce").fillna(0.0)
    return rows["SecurityType"].astype(str).eq("Fixed") & (coupon <= TOLERANCE_BIL) & (years <= BILL_MAX_YEARS + 1e-9)


def stock(rows: pd.DataFrame) -> pd.Series:
    face = pd.to_numeric(rows["FaceValue"], errors="coerce").fillna(0.0)
    adjusted = pd.to_numeric(rows["AdjustedPrincipal"], errors="coerce").fillna(face)
    return face.where(~rows["SecurityType"].astype(str).eq("TIPS"), adjusted)


# --- The Fed's lots on its own acquisition basis --------------------------------------


@dataclass
class FedLot:
    maturity: pd.Timestamp
    premium: float  # unamortized premium (positive) or discount (negative)
    bill: bool
    through: pd.Timestamp  # amortization is booked through the close of this day


# One row per change to a Fed lot, in the order the engine books them: the evidence a verifier
# needs to rebuild the Fed's acquisition-basis income without the engine's lot book.
FED_LOT_JOURNAL_COLUMNS = (
    "date",
    "event",  # open, auction, purchase, merge, sale or retirement
    "bond_id",
    "into_bond_id",  # merge: the lot this one joins
    "maturity_date",
    "bill",
    "fraction",  # sale or retirement: the share of the lot disposed of
    "stock_bil",  # stock acquired or disposed of (indexed principal for TIPS)
    "cash_bil",  # dirty cash paid or received; a retirement receives the stock retired
    "accrued_interest_bil",  # accrued interest bought, sold or forfeited
    "premium_bil",  # premium or discount opened, acquired or removed at carrying value
)
FED_LOT_ACQUISITIONS = ("auction", "purchase")
FED_LOT_DISPOSALS = ("sale", "retirement")


class FedLotBook:
    """Unamortized premium or discount of each Fed lot, keyed by the portfolio's BondID.

    Each lot amortizes straight line to maturity on its own clock, from the day after it was
    acquired. A trade first books the existing lots through the trade day, so a top-up or a
    sale never moves the day's amortization of what the Fed already held, and a purchase
    starts earning the next day. Lots of one security share a maturity, so merging them
    after that keeps each vintage's straight line exactly. Every change is written to
    ``journal`` with its trade evidence.
    """

    def __init__(self) -> None:
        self.lots: dict[int, FedLot] = {}
        self.earned = {"bill": 0.0, "coupon": 0.0}  # booked since the last close
        self.journal: list[tuple] = []

    def add(self, bond_id, maturity, premium: float, bill: bool, acquired) -> None:
        key = int(bond_id)
        if key in self.lots:
            raise FedIncomeError(f"Fed lot {key} is already on the book")
        if not math.isfinite(premium):
            raise FedIncomeError(f"Fed lot {key} has a nonfinite acquisition premium")
        self.lots[key] = FedLot(pd.Timestamp(maturity).normalize(), float(premium), bool(bill),
                                pd.Timestamp(acquired).normalize())

    def _record(self, day, event, key, *, into="", fraction=np.nan, stock=np.nan, cash=np.nan, accrued=np.nan,
                premium=np.nan, lot: FedLot | None = None) -> None:
        lot = lot or self.lots[key]
        self.journal.append((
            pd.Timestamp(day).date().isoformat(), event, key, into, lot.maturity.date().isoformat(), int(lot.bill),
            float(fraction), float(stock), float(cash), float(accrued), float(premium),
        ))

    def open(self, bond_id, maturity, premium: float, bill: bool, day, *, stock: float, proceeds: float) -> None:
        """An inherited lot at the opening, carrying ``premium`` through ``day``; ``proceeds``
        are the issue proceeds the opening approximation read."""

        self.add(bond_id, maturity, premium, bill, day)
        self._record(day, "open", int(bond_id), stock=stock, cash=proceeds, premium=premium)

    def acquire(self, event, bond_id, maturity, bill: bool, day, *, stock: float, cash: float, accrued: float,
                merge_ids=()) -> float:
        """Buy ``stock`` for dirty ``cash`` on ``day``; merge the Fed's lots ``merge_ids`` of the
        same security into ``bond_id``. Returns the premium acquired: the clean cost less the stock."""

        if event not in FED_LOT_ACQUISITIONS:
            raise FedIncomeError(f"unknown Fed lot acquisition: {event!r}")
        key = int(bond_id)
        premium = float(cash) - float(accrued) - float(stock)
        carried = 0.0
        for source in (int(value) for value in merge_ids):
            if source not in self.lots:
                continue
            if source != key:
                self._record(day, "merge", source, into=key)
            carried += self._remove(source, 1.0, day)
        self.add(key, maturity, carried + premium, bill, day)
        self._record(day, event, key, stock=stock, cash=cash, accrued=accrued, premium=premium)
        return premium

    def dispose(self, event, bond_id, fraction: float, day, *, stock: float, cash: float, accrued: float) -> float:
        """Sell or retire a fraction of a lot on ``day`` at its carrying value; return the premium removed."""

        if event not in FED_LOT_DISPOSALS:
            raise FedIncomeError(f"unknown Fed lot disposal: {event!r}")
        key = int(bond_id)
        lot = self.lots.get(key)
        if lot is None:
            raise FedIncomeError(f"Fed lot {key} is not on the book on {pd.Timestamp(day).date()}")
        fraction = min(float(fraction), 1.0)
        removed = self._remove(key, fraction, day)
        self._record(day, event, key, fraction=fraction, stock=stock, cash=cash, accrued=accrued, premium=removed,
                     lot=lot)
        return removed

    def _remove(self, key: int, fraction: float, day) -> float:
        self.accrue(day, [key])
        lot = self.lots.get(key)
        if lot is None:
            return 0.0
        if fraction >= 1.0:
            del self.lots[key]
            return lot.premium
        removed = lot.premium * fraction
        lot.premium -= removed
        return removed

    def total(self) -> float:
        return float(sum(lot.premium for lot in self.lots.values()))

    def accrue(self, day, keys=None) -> None:
        """Book straight-line amortization through ``day`` on the given lots, or on all of them."""

        day = pd.Timestamp(day).normalize()
        for key in list(self.lots) if keys is None else [key for key in keys if key in self.lots]:
            lot = self.lots[key]
            days = (day - lot.through).days
            if days <= 0:
                continue
            remaining = (lot.maturity - lot.through).days
            amount = lot.premium if remaining <= days else lot.premium * days / remaining
            lot.premium -= amount
            lot.through = day
            self.earned["bill" if lot.bill else "coupon"] -= amount
            if lot.maturity <= day:
                if abs(lot.premium) > 1e-9:
                    raise FedIncomeError(f"Fed lot {key} matured with premium {lot.premium!r} unamortized")
                del self.lots[key]

    def close(self, day) -> tuple[float, float]:
        """Book every lot through ``day``; return the period's (bill accretion, coupon amortization) as income."""

        self.accrue(day)
        earned = (self.earned["bill"], self.earned["coupon"])
        self.earned = {"bill": 0.0, "coupon": 0.0}
        return earned


def opening_premiums(cb_rows: pd.DataFrame, day, treasury_net_premium: float) -> dict[int, tuple]:
    """One opening approximation per class for the inherited Fed lots.

    Returns each lot's (maturity, premium, bill, stock, issue proceeds).

    Bills keep their remaining issue discount, straight line from issue to maturity. Coupon
    securities share the rest of the H.4.1 Treasury net premium in proportion to stock times
    remaining life, so every coupon lot amortizes at the same rate per unit of stock and the
    class amortizes at the premium divided by its stock-weighted remaining life.
    """

    day = pd.Timestamp(day).normalize()
    bills = is_bill(cb_rows)
    issue = pd.to_datetime(cb_rows["IssueDate"]).dt.normalize()
    maturity = pd.to_datetime(cb_rows["MaturityDate"]).dt.normalize()
    face = pd.to_numeric(cb_rows["FaceValue"], errors="coerce").fillna(0.0)
    proceeds = pd.to_numeric(cb_rows["IssueProceeds"], errors="coerce").fillna(face)
    life = (maturity - issue).dt.days.clip(lower=1)
    remaining = (maturity - day).dt.days.clip(lower=0)
    bill_discount = -(face - proceeds) * remaining / life
    coupon_weight = stock(cb_rows).where(~bills, 0.0) * remaining / 365.25
    coupon_premium = treasury_net_premium - float(bill_discount[bills].sum())
    # With no coupon securities to carry it, the residual stays in other assets.
    ratio = coupon_premium / float(coupon_weight.sum()) if coupon_weight.sum() > TOLERANCE_BIL else 0.0
    premium = bill_discount.where(bills, coupon_weight * ratio)
    lot_stock = stock(cb_rows)
    return {
        int(row_id): (maturity.at[idx], float(premium.at[idx]), bool(bills.at[idx]), float(lot_stock.at[idx]),
                      float(proceeds.at[idx]))
        for idx, row_id in cb_rows["BondID"].items()
    }


def carried_premiums(cb_rows: pd.DataFrame, carried) -> dict[int, tuple]:
    """The Fed's lots at a carried opening: each keeps the premium it closed with."""

    ids = pd.to_numeric(cb_rows["BondID"], errors="raise").astype(int)
    if set(ids) != set(carried):
        raise FedIncomeError("the carried Fed lots are not the Fed's lots in the opening portfolio")
    bills = is_bill(cb_rows)
    face = pd.to_numeric(cb_rows["FaceValue"], errors="coerce").fillna(0.0)
    proceeds = pd.to_numeric(cb_rows["IssueProceeds"], errors="coerce").fillna(face)
    lot_stock = stock(cb_rows)
    maturity = pd.to_datetime(cb_rows["MaturityDate"]).dt.normalize()
    return {
        int(ids.at[idx]): (maturity.at[idx], float(carried[int(ids.at[idx])]), bool(bills.at[idx]),
                           float(lot_stock.at[idx]), float(proceeds.at[idx]))
        for idx in cb_rows.index
    }


# --- Daily close ----------------------------------------------------------------------


@dataclass(frozen=True)
class FedOpening:
    """Opening balances (USD billions) from the H.4.1 Wednesday release and its bridge."""

    as_of_date: pd.Timestamp
    reserve_balances: float
    tga: float
    domestic_on_rrp: float
    observed_domestic_on_rrp: float
    foreign_repo: float
    foreign_official_deposits: float
    soma_mbs: float
    currency: float
    other_deposits: float
    other_liabilities_and_capital: float
    earnings_remittances_due: float
    treasury_net_premium: float
    remunerated_other_deposit_share: float
    previous_wednesday: pd.Timestamp
    previous_wednesday_due: float
    year_end: pd.Timestamp
    year_end_due: float
    interest_payable: float
    foreign_repo_interest_payable: float

    @classmethod
    def from_row(cls, row) -> FedOpening:
        return cls(
            as_of_date=pd.Timestamp(row["h41_as_of_date"]).normalize(),
            reserve_balances=float(row["h41_reserve_balances_bil"]),
            tga=float(row["h41_tga_bil"]),
            domestic_on_rrp=float(row["h41_domestic_on_rrp_bil"]),
            observed_domestic_on_rrp=float(row["opening_domestic_on_rrp_bil"]),
            foreign_repo=float(row["h41_foreign_repo_bil"]),
            foreign_official_deposits=float(row["h41_foreign_official_deposits_bil"]),
            soma_mbs=float(row["h41_soma_mbs_bil"]),
            currency=float(row["h41_currency_bil"]),
            other_deposits=float(row["h41_other_deposits_bil"]),
            other_liabilities_and_capital=float(row["h41_other_liabilities_and_capital_bil"]),
            earnings_remittances_due=float(row["h41_earnings_remittances_due_bil"]),
            treasury_net_premium=float(row["h41_treasury_net_premium_bil"]),
            remunerated_other_deposit_share=float(row["remunerated_other_deposit_share"]),
            previous_wednesday=pd.Timestamp(row["previous_wednesday_date"]).normalize(),
            previous_wednesday_due=float(row["previous_wednesday_due_bil"]),
            year_end=pd.Timestamp(row["year_end_date"]).normalize(),
            year_end_due=float(row["year_end_due_bil"]),
            interest_payable=float(row["opening_interest_payable_bil"]),
            foreign_repo_interest_payable=float(row["opening_foreign_repo_interest_payable_bil"]),
        )


class FedBook:
    """The Fed's balance sheet and earnings, closed once a day."""

    def __init__(self, opening: FedOpening, *, opening_day, tga: float, treasury_book: dict, path_row,
                 route_balances: dict | None) -> None:
        self.opening = opening
        day = pd.Timestamp(opening_day).normalize()
        self.lots = FedLotBook()
        for bond_id, (maturity, premium, bill, lot_stock, proceeds) in treasury_book["premiums"].items():
            self.lots.open(bond_id, maturity, premium, bill, day, stock=lot_stock, proceeds=proceeds)
        self.accrued = dict(treasury_book["accrued"])
        self.treasury_face = float(treasury_book["stock"])
        # ON RRP and foreign repo balances: the routed ledger when the run has one, else the
        # observed opening. The routed opening's difference from the observed opening is an
        # opening transfer out of reserves.
        routed = route_balances or {}
        self.domestic_rrp = float(routed.get("ONRRP_MMF", opening.observed_domestic_on_rrp))
        self.foreign_repo_base = float(routed.get("ForeignOfficialFedRRP", opening.foreign_repo))
        self.foreign_deposits = float(routed.get("ForeignOfficialFedDeposits", opening.foreign_official_deposits))
        self.foreign_repo_reinvested = 0.0
        self.mbs = float(path_row["soma_mbs_start_bil"])
        self.mbs_premium = float(path_row["soma_mbs_premium_start_bil"])
        self.currency = float(path_row["currency_start_bil"])
        self.other_deposits = float(path_row["other_deposits_start_bil"])
        self.other_liabilities = float(path_row["other_liabilities_and_capital_start_bil"])
        self.payable_reserves = opening.interest_payable
        self.payable_foreign = opening.foreign_repo_interest_payable
        self.due = opening.earnings_remittances_due
        self.reserves = (
            opening.reserve_balances
            - (tga - opening.tga)
            - (self.domestic_rrp - opening.domestic_on_rrp)
            - (self.foreign_repo_base - opening.foreign_repo)
            - (self.foreign_deposits - opening.foreign_official_deposits)
            + (self.mbs - opening.soma_mbs)
            - (self.currency - opening.currency)
            - (self.other_deposits - opening.other_deposits)
            - (self.other_liabilities - opening.other_liabilities_and_capital)
        )
        if not math.isfinite(self.reserves) or self.reserves < 0.0:
            raise FedIncomeError(f"opening reserve balances are not a finite nonnegative level: {self.reserves!r}")
        self.other_assets = self.reserves + self._liabilities(tga) - self._assets_ex_other()
        self.history = {
            opening.previous_wednesday: opening.previous_wednesday_due,
            opening.year_end: opening.year_end_due,
            opening.as_of_date: opening.earnings_remittances_due,
            day: self.due,
        }

    @property
    def foreign_repo(self) -> float:
        return self.foreign_repo_base + self.foreign_repo_reinvested

    def _assets_ex_other(self) -> float:
        return self.treasury_face + self.lots.total() + sum(self.accrued.values()) + self.mbs + self.mbs_premium

    def _liabilities(self, tga: float) -> float:
        return (
            self.currency + tga + self.domestic_rrp + self.foreign_repo + self.foreign_deposits + self.other_deposits
            + self.other_liabilities + self.due + self.payable_reserves + self.payable_foreign
        )

    def remittance(self, day) -> float:
        """Cash the Banks pay the Treasury today, from balances closed before today."""

        day = pd.Timestamp(day).normalize()
        if not is_remittance_day(day):
            return 0.0
        return weekly_remittance(
            self.balance_at(remittance_basis_date(day)), self.balance_at(day - pd.Timedelta(days=1))
        )

    def balance_at(self, day) -> float:
        """The pooled balance at the close of ``day``: the last close on or before it."""

        day = pd.Timestamp(day).normalize()
        closes = [when for when in self.history if when <= day]
        if not closes:
            raise FedIncomeError(f"no earnings-due balance on or before {day.date()}")
        return self.history[max(closes)]

    def balances(self) -> dict[str, float]:
        return {
            "CB_TreasuryUnamortizedPremium": self.lots.total(),
            "CB_TreasuryAccruedInterest": sum(self.accrued.values()),
            "CB_MBSHoldings": self.mbs,
            "CB_MBSUnamortizedPremium": self.mbs_premium,
            "CB_OtherAssets": self.other_assets,
            "CB_Currency": self.currency,
            "CB_DomesticONRRP": self.domestic_rrp,
            "CB_ForeignRepo": self.foreign_repo,
            "CB_ForeignOfficialDeposits": self.foreign_deposits,
            "CB_OtherDeposits": self.other_deposits,
            "CB_FMUBalances": self.other_deposits * self.opening.remunerated_other_deposit_share,
            "CB_OtherLiabilitiesAndCapital": self.other_liabilities,
            "CB_InterestPayable": self.payable_reserves,
            "CB_ForeignRepoInterestPayable": self.payable_foreign,
            "CB_EarningsRemittancesDue": self.due,
            "CB_ReserveBalances": self.reserves,
        }

    def close_day(
        self,
        prev_day,
        day,
        *,
        tga: float,
        remitted: float,
        treasury_reserve_effect: float,
        treasury: dict,
        path_row,
        rates: dict,
        route_balances: dict | None,
    ) -> dict[str, float]:
        """Book one day; return the day's flows and closing balances."""

        prev_day, day = pd.Timestamp(prev_day).normalize(), pd.Timestamp(day).normalize()
        days = (day - prev_day).days
        open_today = is_fedwire_business_day(day.date())
        flows = {}
        # Treasury income on the Fed's basis.
        for kind, prefix in ACCRUAL_TYPES.items():
            cash = float(treasury["cash"][kind])
            flows[f"{prefix}Cash"] = cash
            flows[f"{prefix}Accrual"] = (
                cash + treasury["accrued_end"][kind] - self.accrued[kind]
                - treasury["accrued_purchased"][kind] + treasury["accrued_sold"][kind]
            )
        bill_income, coupon_income = self.lots.close(day)
        flows.update({
            "CB_TIPSIndexation": float(treasury["tips_indexation"]),
            "CB_TIPSFloorTopup": float(treasury["floor_topup"]),
            "CB_BillDiscountAccretion": bill_income,
            "CB_CouponPremiumAmortization": coupon_income,
            "CB_RealizedGains": float(treasury["realized_gains"]),
            "CB_AccruedInterestPurchased": sum(treasury["accrued_purchased"].values()),
            "CB_AccruedInterestSold": sum(treasury["accrued_sold"].values()),
            "CB_PremiumAcquired": float(treasury["premium_acquired"]),
            "CB_PremiumSold": float(treasury["premium_sold"]),
        })
        flows["CB_TreasuryInterestIncome"] = sum(flows[name] for name in TREASURY_INCOME_TERMS)
        self.accrued = dict(treasury["accrued_end"])
        self.treasury_face = float(treasury["stock"])
        # Smoothed proxies on their daily paths.
        mbs_end = float(path_row["soma_mbs_end_bil"])
        mbs_premium_end = float(path_row["soma_mbs_premium_end_bil"])
        flows["CB_MBSCouponIncome"] = float(path_row["soma_mbs_interest_bil"])
        flows["CB_MBSPremiumAmortization"] = self.mbs_premium - mbs_premium_end
        flows["CB_MBSIncome"] = flows["CB_MBSCouponIncome"] - flows["CB_MBSPremiumAmortization"]
        flows["CB_OtherIncome"] = float(path_row["other_income_bil"])
        flows["CB_OperatingExpenses"] = float(path_row["operating_expenses_bil"])
        flows["CB_Dividends"] = float(path_row["dividends_bil"])
        # Post yesterday's accrued interest on a business day.
        posted = self.payable_reserves if open_today else 0.0
        reinvested = self.payable_foreign if open_today else 0.0
        self.payable_reserves -= posted
        self.payable_foreign -= reinvested
        self.foreign_repo_reinvested += reinvested
        flows["CB_InterestPosted"] = posted
        flows["CB_ForeignRepoInterestReinvested"] = reinvested
        # Closing balances before today's interest expense.
        other_deposits_end = float(path_row["other_deposits_end_bil"])
        other_liabilities_end = float(path_row["other_liabilities_and_capital_end_bil"])
        currency_end = float(path_row["currency_end_bil"])
        routed = route_balances or {}
        domestic_rrp_end = float(routed.get("ONRRP_MMF", self.domestic_rrp))
        foreign_base_end = float(routed.get("ForeignOfficialFedRRP", self.foreign_repo_base))
        foreign_deposits_end = float(routed.get("ForeignOfficialFedDeposits", self.foreign_deposits))
        self.reserves += (
            treasury_reserve_effect
            + posted
            + (mbs_end - self.mbs)
            - flows["CB_MBSCouponIncome"]
            - (currency_end - self.currency)
            - (other_deposits_end - self.other_deposits)
            - (other_liabilities_end - self.other_liabilities)
            - flows["CB_OtherIncome"]
            + flows["CB_OperatingExpenses"]
            + flows["CB_Dividends"]
        )
        self.mbs, self.mbs_premium, self.currency = mbs_end, mbs_premium_end, currency_end
        self.other_deposits, self.other_liabilities = other_deposits_end, other_liabilities_end
        self.domestic_rrp, self.foreign_repo_base, self.foreign_deposits = domestic_rrp_end, foreign_base_end, foreign_deposits_end
        if not math.isfinite(self.reserves) or self.reserves < 0.0:
            raise FedIncomeError(f"reserve balances fell below zero at {day.date()}: {self.reserves!r}")
        # Today's interest on today's closing balances, posted on the next business day.
        accrual = days / INTEREST_EXPENSE_DAY_COUNT
        fmu = self.other_deposits * self.opening.remunerated_other_deposit_share
        flows["CB_IORBExpense"] = rates["iorb"] / 100.0 * self.reserves * accrual
        flows["CB_FMUInterestExpense"] = rates["iorb"] / 100.0 * fmu * accrual
        flows["CB_DomesticONRRPExpense"] = rates["on_rrp"] / 100.0 * self.domestic_rrp * accrual
        flows["CB_ForeignRepoExpense"] = rates["on_rrp"] / 100.0 * self.foreign_repo * accrual
        self.payable_reserves += flows["CB_IORBExpense"] + flows["CB_FMUInterestExpense"] + flows["CB_DomesticONRRPExpense"]
        self.payable_foreign += flows["CB_ForeignRepoExpense"]
        net_income = sum(sign * flows[name] for name, sign in NET_INCOME_TERMS.items())
        flows["CB_NetIncome"] = net_income
        flows["CB_Remittance"] = remitted
        flows["CB_TreasuryReserveEffect"] = treasury_reserve_effect
        self.due += net_income - remitted
        self.history[day] = self.due
        balances = self.balances()
        gap = balances["CB_ReserveBalances"] - (
            self._assets_ex_other() + self.other_assets - (self._liabilities(tga) - 0.0)
        ) + 0.0
        if abs(gap) > 1e-6 * max(1.0, abs(self.reserves)):
            raise FedIncomeError(f"the Fed balance sheet does not close at {day.date()}: gap {gap!r}")
        flows["CB_DeferredAsset"] = max(0.0, -self.due)
        return {**flows, **balances, "CB_PolicyRatePct": rates["policy"], "CB_IORBRatePct": rates["iorb"],
                "CB_ONRRPRatePct": rates["on_rrp"]}

    def opening_row(self) -> dict[str, float]:
        return {**self.balances(), "CB_DeferredAsset": max(0.0, -self.due)}


def last_business_day(day) -> pd.Timestamp:
    return pd.Timestamp(previous_fedwire_business_day(pd.Timestamp(day).date()))
