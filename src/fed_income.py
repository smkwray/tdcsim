"""Federal Reserve net income, deferred asset and remittances for CBO runs.

Net income is cash basis, as the run pays it: SOMA Treasury interest received from the
run's own debt service, agency MBS interest on an exogenous runoff path, and other
income, less interest on reserve balances, interest on reverse repos, operating
expenses and dividends. Reserve balances are a balance-sheet level: the H.4.1 opening,
rolled to the run's opening and then moved by the run's Treasury contribution to
reserves and by the non-Treasury factors (currency, MBS principal runoff and the
non-Treasury income lines).

The deferred asset is the negative side of one signed state, the H.4.1 line "Earnings
remittances due to the U.S. Treasury". Net income adds to it every period; a loss
lowers it. The Reserve Banks pay a positive balance to the Treasury on the first
Fedwire business day of each remittance week (weeks start on Wednesday). While the
state is negative it is the deferred asset, and nothing is paid until net income has
repaid it.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


REMITTANCE_WEEK_START_WEEKDAY = 2  # Wednesday: the Reserve Banks remit weekly.
INTEREST_EXPENSE_DAY_COUNT = 360.0  # IORB and ON RRP accrue actual/360.

FED_INCOME_RESULT_COLUMNS = (
    "CB_BillDiscountIncome",
    "CB_CouponIncome",
    "CB_FRNIncome",
    "CB_TIPSCouponIncome",
    "CB_TIPSInflationCompensationIncome",
    "CB_TreasuryInterestIncome",
    "CB_MBSIncome",
    "CB_OtherIncome",
    "CB_IORBExpense",
    "CB_ONRRPExpense",
    "CB_OperatingExpenses",
    "CB_Dividends",
    "CB_EarningsRemittancesDue",
    "CB_ShortRatePct",
    "CB_IORBRatePct",
    "CB_ONRRPRatePct",
    "CB_TreasuryReserveEffect",
    "CB_ReserveBalances",
    "CB_ONRRPBalances",
    "CB_Currency",
    "CB_MBSHoldings",
    "CB_OtherLiabilitiesAndCapital",
    "CBOTable43RemittanceInFiscalFlow",
)

TREASURY_INCOME_COLUMNS = (
    "CB_BillDiscountIncome",
    "CB_CouponIncome",
    "CB_FRNIncome",
    "CB_TIPSCouponIncome",
    "CB_TIPSInflationCompensationIncome",
)


class FedIncomeError(ValueError):
    """Raised when the Fed balance sheet or income inputs are inconsistent."""


def remittance_week(day) -> pd.Timestamp:
    """Return the Wednesday that starts the remittance week containing ``day``."""

    day = pd.Timestamp(day).normalize()
    return day - pd.Timedelta(days=(day.weekday() - REMITTANCE_WEEK_START_WEEKDAY) % 7)


def settle_remittances(earnings_remittances_due: float, net_income: float, pay_today: bool) -> tuple[float, float]:
    """Book net income into the signed due balance and pay a positive balance on a pay day.

    A negative balance is the deferred asset: income repays it before anything is paid,
    and a loss adds to it. Returns ``(new_balance, paid)``.
    """

    balance = float(earnings_remittances_due) + float(net_income)
    if not math.isfinite(balance):
        raise FedIncomeError("Fed earnings remittances due became nonfinite")
    paid = balance if pay_today and balance > 0.0 else 0.0
    return balance - paid, paid


@dataclass(frozen=True)
class FedOpeningBalanceSheet:
    """H.4.1 Wednesday levels, USD billions, that anchor the run's Fed balance sheet."""

    as_of_date: str
    reserve_balances: float
    tga: float
    on_rrp: float
    soma_mbs: float
    currency: float
    earnings_remittances_due: float
    other_liabilities_and_capital: float

    @classmethod
    def from_row(cls, row) -> FedOpeningBalanceSheet:
        return cls(
            as_of_date=str(row["h41_as_of_date"]),
            reserve_balances=float(row["h41_reserve_balances_bil"]),
            tga=float(row["h41_tga_bil"]),
            on_rrp=float(row["h41_on_rrp_bil"]),
            soma_mbs=float(row["h41_soma_mbs_bil"]),
            currency=float(row["h41_currency_bil"]),
            earnings_remittances_due=float(row["h41_earnings_remittances_due_bil"]),
            other_liabilities_and_capital=float(row["h41_other_liabilities_and_capital_bil"]),
        )

    def rolled_reserves(self, *, tga: float, on_rrp: float, soma_mbs: float, currency: float) -> float:
        """Roll the Wednesday reserve level to the run's opening by the factors that moved."""

        return (
            self.reserve_balances
            - (tga - self.tga)
            - (on_rrp - self.on_rrp)
            + (soma_mbs - self.soma_mbs)
            - (currency - self.currency)
        )


def non_treasury_reserve_effect(
    *,
    mbs_start: float,
    mbs_end: float,
    currency_start: float,
    currency_end: float,
    mbs_income: float,
    other_income: float,
    iorb_expense: float,
    on_rrp_expense: float,
    operating_expenses: float,
    dividends: float,
) -> float:
    """Reserve change from the Fed's non-Treasury factors in one period.

    MBS principal runoff and currency growth drain reserves; interest and fees the Fed
    receives from the private sector drain them; interest, expenses and dividends it pays
    add to them. Treasury interest and remittances move only the TGA.
    """

    return (
        (mbs_end - mbs_start)
        - (currency_end - currency_start)
        - (mbs_income + other_income)
        + (iorb_expense + on_rrp_expense + operating_expenses + dividends)
    )


# Export names, shared by the visual ledger and the scenario-family fiscal-year table.
# Flows are per period (summed over a fiscal year); levels are end-of-period.
FED_INCOME_EXPORT_FLOWS = {
    "CB_TreasuryInterestIncome": "fed_treasury_interest_income_bil",
    "CB_BillDiscountIncome": "fed_bill_discount_income_bil",
    "CB_CouponIncome": "fed_coupon_income_bil",
    "CB_FRNIncome": "fed_frn_income_bil",
    "CB_TIPSCouponIncome": "fed_tips_coupon_income_bil",
    "CB_TIPSInflationCompensationIncome": "fed_tips_inflation_compensation_income_bil",
    "CB_MBSIncome": "fed_mbs_income_bil",
    "CB_OtherIncome": "fed_other_income_bil",
    "CB_IORBExpense": "fed_iorb_expense_bil",
    "CB_ONRRPExpense": "fed_on_rrp_expense_bil",
    "CB_OperatingExpenses": "fed_operating_expenses_bil",
    "CB_Dividends": "fed_dividends_bil",
    "CB_NetIncome": "fed_net_income_bil",
}
FED_INCOME_EXPORT_LEVELS = {
    "CB_DeferredAsset": "fed_deferred_asset_bil",
    "CB_EarningsRemittancesDue": "fed_earnings_remittances_due_bil",
    "CB_ReserveBalances": "fed_reserve_balances_bil",
    "CB_ONRRPBalances": "fed_on_rrp_balances_bil",
    "CB_Currency": "fed_currency_bil",
    "CB_MBSHoldings": "fed_mbs_holdings_bil",
    "CB_OtherLiabilitiesAndCapital": "fed_other_liabilities_and_capital_bil",
    "CB_IORBRatePct": "iorb_rate_pct",
    "CB_ONRRPRatePct": "on_rrp_rate_pct",
}
