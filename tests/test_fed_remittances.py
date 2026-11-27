"""Federal Reserve earnings, balance sheet and weekly remittances in CBO runs.

The remittance must follow the Fed's own books: income accrued on what the Fed paid for its
securities, interest on the balances it actually owes, the pooled deferred asset, and the
Reserve Banks' weekly payment rule. These tests run the production engine on small fixtures
and check each channel, and that the verifier rejects books that do not close.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from fed_income import (
    FED_LOT_JOURNAL_COLUMNS, FedLotBook, first_remittance_day_of_year, remittance_basis_date, remittance_week,
    weekly_remittance,
)
from forecast_bundle_builders import build_fed_income_path_rows, fed_income_source_fixture, fed_net_premiums_bil
from forecast_paths import (
    check_earnings_due_closes, check_fed_balance_sheet_closes, rebuild_fed_lot_income, verify_fed_remittances,
)
from sim_engine import run_simulation
from simulation_calendar import build_simulation_calendar
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES
from test_cbo_engine_integration import _build_temp_forecast_inputs, _minimal_engine_params


START, END = "2026-09-20", "2026-10-10"
# Opening reserves land at 2,200 after the H.4.1 roll (the run's TGA opens at 800); a
# steady other-income line keeps net income positive so remittances can flow.
SMALL_FED = {"h41_reserve_balances_bil": 3_000.0, "h41_tga_bil": 0.0, "other_income_bil": 0.5}


def _fed_note(face: float, coupon: float) -> dict:
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update({
        "BondID": 2, "SecurityType": "Fixed", "IssueDate": pd.Timestamp("2025-09-30"),
        "MaturityDate": pd.Timestamp("2027-09-30"), "DatedDate": pd.Timestamp("2025-09-30"),
        "OriginalMaturityYears": 2.0, "FaceValue": face,
        "CouponRate": coupon, "HolderType": "CB", "HolderSubBucket": "", "Status": "Active",
        "MaturityCategory": "notes", "OriginalPrincipal": face, "AdjustedPrincipal": face,
        "ReferenceCPI_Issue": 0.0, "IndexRatio": 1.0, "FixedSpread": 0.0, "AccruedInterest_FRN": 0.0,
        "BenchmarkRate_FRN": 0.0, "LastAccrualDate": pd.NaT, "IssuePriceRatio": 1.0,
        "IssueProceeds": face, "IssueYieldAtIssue": coupon,
    })
    return row


def _run(tmp_path, *, fed_coupon: float = 0.04, remittance_bil: float = 0.0, rate_shift: float = 0.0,
         **fed_income):
    periods = build_simulation_calendar(START, END, "daily")
    paths = _build_temp_forecast_inputs(
        tmp_path, periods=periods, period_start=START, period_end=END,
        cbo_public_debt_target_bil=1_700.0, pre_issuance_controlled_debt_bil=1_500.0,
        remittance_bil=remittance_bil, fed_income={**SMALL_FED, **fed_income},
        later_fiscal_years={2027: (1_750.0, 0.0)},
    )
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=1_000.0)
    params["yield_curve"]["rates"] = [rate + rate_shift for rate in params["yield_curve"]["rates"]]
    portfolio = params["initial_bonds_df"]
    params["initial_bonds_df"] = pd.concat(
        [portfolio, pd.DataFrame([_fed_note(500.0, fed_coupon)], columns=BOND_PORTFOLIO_COLS)],
        ignore_index=True,
    ).astype(PORTFOLIO_DTYPES, errors="ignore")
    return _simulate(params, START, END), paths


def _simulate(params, start: str, end: str):
    """Run the engine and keep the portfolios the verifier binds the Fed's lot journal to."""

    opening = params["initial_bonds_df"].copy()
    results, final = run_simulation(params, start, end, freq="D", scenario_name="baseline")
    results.attrs.update(opening_portfolio=opening, final_portfolio=final)
    return results


def _frame(results):
    return results.reset_index().rename(columns={"DebtHeld_CB": "DebtHeld_CentralBank"})


def _verify(results, paths) -> dict:
    return verify_fed_remittances(
        _frame(results), paths["fed_income_path_file"].parent, lots=results.attrs["fed_lot_journal"],
        opening_portfolio=results.attrs["opening_portfolio"], final_portfolio=results.attrs["final_portfolio"],
        year_end_lots=results.attrs["fed_year_end_lots"],
    )


def test_weekly_payment_is_the_lower_of_two_credit_balances_and_nothing_on_a_debit():
    assert weekly_remittance(5.0, 7.0) == 5.0
    assert weekly_remittance(7.0, 5.0) == 5.0
    assert weekly_remittance(5.0, -0.1) == 0.0  # a debit the day before blocks the payment
    assert weekly_remittance(-0.1, 5.0) == 0.0  # so does a debit last Wednesday
    assert weekly_remittance(0.0, 5.0) == 0.0


def test_remittance_days_and_the_first_payment_of_a_year():
    assert remittance_week("2026-10-06") == pd.Timestamp("2026-09-30")
    # A regular week caps the payment at last Wednesday's balance ...
    assert remittance_basis_date("2026-10-07") == pd.Timestamp("2026-09-30")
    # ... Veterans Day closes Wednesday 2026-11-11, so Thursday pays against 2026-11-04 ...
    assert remittance_basis_date("2026-11-12") == pd.Timestamp("2026-11-04")
    # ... and the year's first payment uses the December 31 balance.
    assert first_remittance_day_of_year(2027) == pd.Timestamp("2027-01-06")
    assert remittance_basis_date("2027-01-06") == pd.Timestamp("2026-12-31")


def test_a_wednesday_coupon_does_not_fund_that_wednesdays_remittance(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=0.0)
    due = results["CB_EarningsRemittancesDue"]
    for wednesday in ("2026-09-23", "2026-09-30", "2026-10-07"):
        day = pd.Timestamp(wednesday)
        cap = min(due.loc[day - pd.Timedelta(days=7)] if day - pd.Timedelta(days=7) in due.index
                  else 0.0, due.loc[day - pd.Timedelta(days=1)])
        assert results.loc[day, "CB_Remittance"] == pytest.approx(max(cap, 0.0))
    # The opening Wednesday balance (zero) caps the first payment at nothing.
    assert results.loc["2026-09-23", "CB_Remittance"] == 0.0
    assert results["CB_Remittance"].sum() > 0.0
    _verify(results, paths)


def test_the_first_coupon_pays_down_the_opening_receivable(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=-100.0)
    receivable = results["CB_TreasuryAccruedInterest"]
    # Interest from 2026-03-31 (a month-end maturity pays on month ends) to the opening is a
    # receivable, not income in the run.
    assert receivable.iloc[0] == pytest.approx(500 * 0.04 / 2 * 173 / 183, rel=1e-6)
    coupon_day = pd.Timestamp("2026-09-30")
    assert results.loc[coupon_day, "CB_CouponCash"] == pytest.approx(10.0)
    assert receivable.loc[coupon_day] == pytest.approx(0.0, abs=1e-12)
    # The coupon earns only the days the Fed held it during the run.
    earned = results.loc[START:"2026-09-30", "CB_CouponAccrual"].iloc[1:].sum()
    assert earned == pytest.approx(10.0 - receivable.iloc[0])
    assert results.loc[coupon_day, "CB_CouponAccrual"] < 0.1
    _verify(results, paths)


def test_higher_fed_coupon_income_raises_remittances_once_the_deferred_asset_is_clear(tmp_path):
    low, paths = _run(tmp_path / "low", fed_coupon=0.04, h41_earnings_remittances_due_bil=0.0)
    high, _ = _run(tmp_path / "high", fed_coupon=0.06, h41_earnings_remittances_due_bil=0.0)
    extra = (high["CB_CouponAccrual"] - low["CB_CouponAccrual"]).sum()
    assert extra == pytest.approx(500 * 0.02 / 365 * 20, rel=0.05)
    assert high["CB_Remittance"].sum() > low["CB_Remittance"].sum()
    # Every extra dollar earned is either remitted or still owed to the Treasury.
    passed_on = (high["CB_Remittance"].sum() + high["CB_EarningsRemittancesDue"].iloc[-1]) - (
        low["CB_Remittance"].sum() + low["CB_EarningsRemittancesDue"].iloc[-1]
    )
    assert passed_on == pytest.approx(extra, rel=1e-3)
    _verify(low, paths)


def test_higher_fed_coupon_income_repays_more_deferred_asset_and_remits_nothing(tmp_path):
    low, _ = _run(tmp_path / "low", fed_coupon=0.04, h41_earnings_remittances_due_bil=-100.0)
    high, _ = _run(tmp_path / "high", fed_coupon=0.06, h41_earnings_remittances_due_bil=-100.0)
    assert low["CB_Remittance"].sum() == high["CB_Remittance"].sum() == 0.0
    extra = (high["CB_CouponAccrual"] - low["CB_CouponAccrual"]).sum()
    assert low["CB_DeferredAsset"].iloc[-1] - high["CB_DeferredAsset"].iloc[-1] == pytest.approx(extra, rel=1e-3)
    # While the deferred asset is positive the Treasury bears the whole extra coupon.
    assert low["TGA"].iloc[-1] - high["TGA"].iloc[-1] == pytest.approx(5.0, abs=1e-6)


def test_higher_reserves_raise_iorb_expense_and_lower_remittances(tmp_path):
    lean, _ = _run(tmp_path / "lean", h41_earnings_remittances_due_bil=0.0)
    flush, paths = _run(tmp_path / "flush", h41_earnings_remittances_due_bil=0.0, h41_reserve_balances_bil=5_000.0)
    gap = flush["CB_ReserveBalances"] - lean["CB_ReserveBalances"]
    assert gap.iloc[0] == pytest.approx(2_000.0)
    # Each day's IORB is that day's rate on that day's closing reserves, actual/360.
    expected = (flush["CB_IORBRatePct"] / 100.0 * gap / 360.0).iloc[1:]
    assert (flush["CB_IORBExpense"] - lean["CB_IORBExpense"]).iloc[1:].to_numpy() == pytest.approx(expected.to_numpy())
    assert flush["CB_Remittance"].sum() < lean["CB_Remittance"].sum()
    assert flush["TGA"].iloc[-1] < lean["TGA"].iloc[-1]
    _verify(flush, paths)


def test_fmu_balances_earn_iorb_and_interest_posts_on_the_next_business_day(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=0.0, other_deposits_bil=200.0)
    fmu = results["CB_FMUBalances"]
    assert fmu.iloc[0] == pytest.approx(100.0)  # the fixture's remunerated share is one half
    assert results["CB_FMUInterestExpense"].iloc[1:].to_numpy() == pytest.approx(
        (results["CB_IORBRatePct"] / 100.0 * fmu / 360.0).iloc[1:].to_numpy()
    )
    # Saturday's and Sunday's interest posts on Monday, not on the weekend.
    assert results.loc["2026-09-26", "CB_InterestPosted"] == 0.0
    assert results.loc["2026-09-27", "CB_InterestPosted"] == 0.0
    assert results.loc["2026-09-28", "CB_InterestPosted"] == pytest.approx(
        results.loc["2026-09-27", "CB_InterestPayable"]
    )
    _verify(results, paths)


def test_a_long_yield_shock_leaves_iorb_and_on_rrp_unchanged(tmp_path):
    base, _ = _run(tmp_path / "base", h41_earnings_remittances_due_bil=0.0)
    shocked, _ = _run(tmp_path / "shocked", h41_earnings_remittances_due_bil=0.0, rate_shift=0.01)
    for column in ("CB_PolicyRatePct", "CB_IORBRatePct", "CB_ONRRPRatePct"):
        pd.testing.assert_series_equal(base[column], shocked[column], check_exact=True)
    assert base["CB_IORBRatePct"].iloc[1] == pytest.approx(4.02)
    assert base["CB_ONRRPRatePct"].iloc[1] == pytest.approx(3.87)


FRN_DAILY = 500.0 * (0.05 + 0.001) / 360.0  # the Fed's FRN: 500 at a 5% benchmark plus 0.1%


def _fed_frn_run(tmp_path, start: str, end: str, *, coupon: str, due: float, other_income: float):
    """The operating-cash lane, where Treasury pays only when Fedwire is open, with the Fed
    holding a 500 FRN whose quarterly coupons fall on the first of the month."""

    import csv

    from test_cbo_engine_integration import (
        _frn_rate_path_row, _opening_frn_portfolio, _write_fixture_fed_income_path, _write_frn_rate_path,
    )
    from test_issuance_rule import _engine_params, _portfolio, _portfolio_row
    from test_treasury_calendar import _profile

    opening = pd.Timestamp(start)
    params = _engine_params(tmp_path, profile=_profile(nominal_only=True), mode="cbo_operating_cash_target_financing_v1",
                            start=start, end=end, later_fiscal_years={opening.year + 1 + (opening.month >= 10): (1_100.0, 0.0)})
    params["data_vintage"]["actuals_available_as_of"] = start
    paths = params["baseline_input_paths"]
    paths["frn_rate_path_file"] = str(_write_frn_rate_path(tmp_path, rows=[
        {**_frn_rate_path_row(period_start=period.period_start.isoformat(), period_end=period.period_end.isoformat(),
                              rate_decimal=0.05), "observation_date": start, "available_date": start}
        for period in build_simulation_calendar(start, end, "daily")
    ]))
    with open(paths["primary_deficit_path_file"], newline="") as handle:
        primary = list(csv.DictReader(handle))
    paths["fed_income_path_file"] = str(_write_fixture_fed_income_path(
        tmp_path, primary, **{**SMALL_FED, "other_income_bil": other_income, "h41_earnings_remittances_due_bil": due},
    ))
    frn = _opening_frn_portfolio(face_value=500.0, accrued_interest=0.0, first_interest_payment_date=coupon,
                                 maturity_date="2029-10-01", last_accrued=start).to_dict("records")[0]
    frn.update(BondID=50, HolderType="CB", HolderSubBucket="")
    params["initial_bonds_df"] = _portfolio([
        _portfolio_row(1, security_type="Fixed", years=28 / 365.25, maturity=str((opening + pd.Timedelta(days=27)).date()),
                       face=100.0),
        _portfolio_row(4, security_type="Fixed", years=2.0, maturity="2029-09-30", face=400.0),
        frn,
    ])
    results = _simulate(params, start, end)
    _verify(results, {"fed_income_path_file": Path(paths["fed_income_path_file"])})
    return results


def test_fed_frn_interest_accrues_on_a_closed_tuesday_and_sizes_wednesdays_payment(tmp_path):
    """Treasury pays the FRN coupon dated Saturday 2028-07-01 on Monday and catches up
    Independence Day (a Tuesday) on Wednesday, but the Fed earns the interest every
    calendar day. Wednesday's payment is capped by Tuesday's balance, so a Tuesday that
    earned nothing would shrink it."""

    results = _fed_frn_run(tmp_path, "2028-06-28", "2028-07-06", coupon="2028-07-01", due=50.0, other_income=0.0)
    accrual, receivable = results["CB_FRNAccrual"].iloc[1:], results["CB_TreasuryAccruedInterest"]
    assert accrual.to_numpy() == pytest.approx([FRN_DAILY] * len(accrual))
    # The coupon due on Saturday stays in the receivable until Monday pays it.
    assert receivable.loc["2028-07-02"] == pytest.approx(4 * FRN_DAILY)
    assert results.loc["2028-07-03", "CB_FRNCash"] == pytest.approx(3 * FRN_DAILY)
    assert receivable.loc["2028-07-03"] == pytest.approx(2 * FRN_DAILY)
    # Net income is negative, so the closed Tuesday's balance sets Wednesday's cap, and
    # that balance includes Tuesday's FRN interest.
    due = results["CB_EarningsRemittancesDue"]
    assert due.loc["2028-07-04"] < due.loc["2028-06-28"]
    assert results.loc["2028-07-05", "CB_Remittance"] == pytest.approx(due.loc["2028-07-04"])
    assert due.loc["2028-07-04"] - due.loc["2028-07-03"] == pytest.approx(results.loc["2028-07-04", "CB_NetIncome"])


def test_fed_frn_interest_earned_on_a_closed_fiscal_year_end_stays_in_that_year(tmp_path):
    """2028-09-30 is a Saturday: the FRN interest the Fed earns that day belongs to fiscal
    2028 even though Treasury next pays on Monday, in fiscal 2029."""

    results = _fed_frn_run(tmp_path, "2028-09-27", "2028-10-03", coupon="2028-10-01", due=0.0, other_income=0.5)
    fiscal_2028 = results.loc["2028-09-28":"2028-09-30", "CB_FRNAccrual"]
    assert fiscal_2028.sum() == pytest.approx(3 * FRN_DAILY)
    assert results.loc["2028-09-30", "CB_TreasuryAccruedInterest"] == pytest.approx(3 * FRN_DAILY)
    assert results.loc["2028-10-02", "CB_FRNCash"] == pytest.approx(4 * FRN_DAILY)
    assert results.loc["2028-10-02", "CB_FRNAccrual"] == pytest.approx(FRN_DAILY)


def test_the_fed_meets_cbos_published_stock_on_a_closed_fiscal_year_end(tmp_path):
    """CBO publishes the Fed's Treasury stock at each fiscal-year end. 2028-09-30 is a Saturday:
    the Fed cannot trade then, so it must buy on Friday what takes it to that published stock,
    not stop at Friday's point on the path between year ends."""

    from forecast_bundle_builders import build_fed_holdings_path_rows
    from test_cbo_engine_integration import _write_csv
    from test_issuance_rule import _engine_params, _portfolio, _portfolio_row
    from test_treasury_calendar import _profile

    start, end = "2028-09-20", "2028-10-03"
    params = _engine_params(tmp_path, profile=_profile(nominal_only=True), mode="cbo_operating_cash_target_financing_v1",
                            start=start, end=end, later_fiscal_years={2029: (1_100.0, 0.0)})
    params["data_vintage"]["actuals_available_as_of"] = start
    params["initial_bonds_df"] = _portfolio([
        _portfolio_row(1, security_type="Fixed", years=91 / 365.25, maturity="2028-12-21", face=300.0),
        _portfolio_row(4, security_type="Fixed", years=2.0, maturity="2029-09-30", face=200.0),
        _portfolio_row(9, security_type="Fixed", years=2.0, maturity="2030-03-31", face=100.0, holder="CB"),
        # The August 2028 refunding issues that September and October reopen.
        *[_portfolio_row(900 + years, security_type="Fixed", years=float(years), maturity=f"{2028 + years}-08-15",
                         face=20.0) for years in (10, 20, 30)],
    ])
    params["baseline_input_paths"]["fed_holdings_path_file"] = str(_write_csv(
        tmp_path / "fed.csv",
        build_fed_holdings_path_rows(
            scenario_id="baseline", periods=build_simulation_calendar(start, end, "daily"), opening_state_date=start,
            opening_cb_holdings_bil=100.0, cbo_fy_end_fed_holdings_bil={2028: 150.0, 2029: 200.0},
            observation_date=start, available_date=start,
        ),
    ))
    results = _simulate(params, start, end)
    path = pd.read_csv(tmp_path / "fed.csv").set_index("period_end")["cbo_fed_holdings_target_bil"]
    assert results.loc["2028-09-30", "DebtHeld_CentralBank"] == pytest.approx(150.0, abs=1e-6)
    assert path["2028-09-29"] < 150.0 - 0.01  # Friday's point on the path falls short of the year end
    assert results.loc["2028-09-29", "DebtHeld_CentralBank"] == pytest.approx(150.0, abs=1e-6)
    # A run that stops on Friday positions the same way, so a continuation from its close (which
    # cannot trade on the weekend) still opens the year end at CBO's stock.
    import copy

    friday = _simulate(copy.deepcopy(params), start, "2028-09-29")
    assert friday.loc["2028-09-29", "DebtHeld_CentralBank"] == pytest.approx(150.0, abs=1e-6)


def test_the_year_end_trade_allows_for_the_tips_it_buys_or_sells():
    """The audit's case: on positioning Friday 2028-09-29 the only security the Fed can buy is a
    TIPS, and the reference CPI moves from 340 to 340.02 by Sunday's fiscal-year end. Sized on
    the book before the trade, buying 0.1 would leave the year end 5.9e-6 above CBO's stock; the
    trade must allow for its own TIPS so the year-end stop fires only on a real miss."""

    from sim_engine import _fed_stock_at, _fed_year_end_trade, _purchase_issue_allocations, _purchase_issues
    from test_issuance_rule import _portfolio, _portfolio_row

    friday, year_end = pd.Timestamp("2028-09-29"), pd.Timestamp("2028-09-30")

    def tips(bond_id, holder, face):
        row = _portfolio_row(bond_id, security_type="TIPS", years=5.0, maturity="2029-04-15", face=face, holder=holder)
        row.update(IssueID="TIPS-2029-04", ReferenceCPI_Issue=300.0, OriginalPrincipal=face,
                   AdjustedPrincipal=face * 340.0 / 300.0, IndexRatio=340.0 / 300.0, CouponRate=0.00125)
        return row

    book = _portfolio([tips(1, "Private", 50.0), tips(2, "CB", 10.0)])
    base = _fed_stock_at(book, year_end, 340.02)
    for wanted in (base + 0.1, base - 0.1):  # a purchase and a sale of TIPS
        trade = _fed_year_end_trade(book, friday, year_end, wanted, 340.02, 340.0)
        naive = wanted - base
        assert abs(naive * 340.02 / 340.0 - naive) == pytest.approx(0.1 * 0.02 / 340.0, rel=1e-6)  # the audit's 5.9e-6
        if trade > 0:
            allocations, _ = _purchase_issue_allocations(_purchase_issues(book, friday), trade)
            assert sum(allocations.values()) == pytest.approx(trade)
        assert base + trade * 340.02 / 340.0 == pytest.approx(wanted, abs=1e-10)




def test_the_opening_lot_check_reads_each_lot_by_its_id_not_its_row():
    """The opening portfolio lists the Fed's lots in no particular order; the verifier must
    rebuild each lot's opening premium from that lot's own terms and issue proceeds."""

    from forecast_paths import _check_fed_lot_boundaries
    from fed_income import FED_LOT_JOURNAL_COLUMNS, opening_premiums

    bill = {**_fed_note(100.0, 0.0), "BondID": 9, "IssueDate": pd.Timestamp("2026-07-01"),
            "MaturityDate": pd.Timestamp("2026-12-31"), "DatedDate": pd.Timestamp("2026-07-01"),
            "OriginalMaturityYears": 0.5, "MaturityCategory": "bills", "IssueProceeds": 98.0}
    note = _fed_note(500.0, 0.04)  # BondID 2
    portfolio = pd.DataFrame([bill, note], columns=BOND_PORTFOLIO_COLS)  # the bill's row first
    day = pd.Timestamp("2026-09-20")
    book = FedLotBook()
    for key, (maturity, premium, is_bill, lot_stock, proceeds) in sorted(opening_premiums(portfolio, day, 5.0).items()):
        book.open(key, maturity, premium, is_bill, day, stock=lot_stock, proceeds=proceeds)
    journal = pd.DataFrame(book.journal, columns=FED_LOT_JOURNAL_COLUMNS)
    assert journal["bond_id"].tolist() == [2, 9]
    _check_fed_lot_boundaries(journal, {"lots": {}}, opening_portfolio=portfolio, final_portfolio=None,
                              dates=pd.Series([day]), net_premium=5.0, carried_lots=None)
    swapped = journal.copy()  # each lot recorded with the other's premium
    swapped["premium_bil"] = swapped["premium_bil"].to_numpy()[::-1]
    with pytest.raises(ValueError, match="opening Fed lot premium"):
        _check_fed_lot_boundaries(swapped, {"lots": {}}, opening_portfolio=portfolio, final_portfolio=None,
                                  dates=pd.Series([day]), net_premium=5.0, carried_lots=None)


def test_a_bill_bought_at_99_earns_1_whatever_it_was_issued_at():
    """Sol's case: issued at 97, bought by the Fed at 99, redeemed at 100."""

    book = FedLotBook()
    book.add(7, "2026-10-20", premium=99.0 - 100.0, bill=True, acquired="2026-09-30")
    earned = sum(sum(book.close(day)) for day in pd.date_range("2026-10-01", "2026-10-20"))
    assert earned == pytest.approx(1.0)
    assert 7 not in book.lots


def test_a_premium_purchase_amortizes_against_income_to_maturity():
    book = FedLotBook()
    book.add(8, "2026-10-11", premium=2.0, bill=False, acquired="2026-10-01")
    daily = [sum(book.close(day)) for day in pd.date_range("2026-10-02", "2026-10-11")]
    assert sum(daily) == pytest.approx(-2.0)
    assert daily[0] == pytest.approx(-0.2)  # straight line over the ten days held


def _ten_day_bill_probe(trade):
    """The audit's probe: a ten-day bill bought at a discount of 10, a second purchase of the
    same bill at a discount of 0.8 on day 2 (booked by ``trade``), half of everything sold on
    day 6 and the rest held to maturity. Returns each day's accretion and the discount sold."""

    book = FedLotBook()
    book.open(1, "2026-10-10", premium=-10.0, bill=True, day="2026-09-30", stock=100.0, proceeds=90.0)
    daily, sold = [], 0.0
    for day in pd.date_range("2026-10-01", "2026-10-10"):
        if day == pd.Timestamp("2026-10-02"):
            trade(book, day)
        if day == pd.Timestamp("2026-10-06"):
            sold += sum(book.dispose("sale", key, 0.5, day, stock=50.0, cash=49.8, accrued=0.0) for key in list(book.lots))
        daily.append(book.close(day)[0])
    assert not book.lots
    # The verifier rebuilds the same days from the journal alone, never merging vintages.
    journal = pd.DataFrame(book.journal, columns=FED_LOT_JOURNAL_COLUMNS)
    rebuilt = rebuild_fed_lot_income(journal, pd.Series(pd.date_range("2026-09-30", "2026-10-10")))
    assert rebuilt["bill_accretion"] == pytest.approx(daily)
    assert rebuilt["premium_sold"].sum() == pytest.approx(sold)
    return daily, sold


def test_a_top_up_leaves_the_existing_holdings_amortization_unchanged():
    """Buying more of a bill the Fed already owns must not stop the day's accretion on what it
    held. Each day's accretion reaches the earnings balance the weekly payment reads, so the
    merged holding must earn exactly what the two vintages earn held separately, every day
    through a partial sale and maturity."""

    def buy(bond_id, merge_ids):
        return lambda book, day: book.acquire("purchase", bond_id, "2026-10-10", True, day, stock=100.0, cash=99.2,
                                              accrued=0.0, merge_ids=merge_ids)

    merged, merged_sold = _ten_day_bill_probe(buy(1, [1]))
    separate, separate_sold = _ten_day_bill_probe(buy(2, []))
    assert merged == pytest.approx(separate)
    assert merged_sold == pytest.approx(separate_sold)
    # Day 2 still earns the first vintage's 1; the purchase earns its 0.1 a day from day 3.
    assert merged == pytest.approx([1.0, 1.0, 1.1, 1.1, 1.1, 1.1, 0.55, 0.55, 0.55, 0.55])
    # The day-6 sale goes at the carrying value after that day's accretion: half of 4.4.
    assert merged_sold == pytest.approx(-2.2)
    assert sum(merged) - merged_sold == pytest.approx(10.8)


def test_a_secondary_purchase_is_carried_at_its_clean_cost(tmp_path):
    from forecast_bundle_builders import build_fed_holdings_path_rows
    from test_cbo_engine_integration import _single_period, _write_csv

    # The purchase settles on 2026-09-25, between coupon dates, so the Fed buys accrued interest.
    window = _single_period("2026-09-20", "2026-09-25")
    paths = _build_temp_forecast_inputs(tmp_path, periods=window, period_end="2026-09-25",
                                        cbo_public_debt_target_bil=1_250.0)
    paths["fed_holdings_path_file"] = _write_csv(tmp_path / "tdcsim_fed_holdings_path.csv", build_fed_holdings_path_rows(
        scenario_id="baseline", periods=window, opening_state_date="2026-09-20", opening_cb_holdings_bil=0.0,
        cbo_fy_end_fed_holdings_bil={2026: 20.0}, observation_date="2026-09-20", available_date="2026-09-20",
    ))
    results, _ = run_simulation(_minimal_engine_params(paths), "2026-09-20", "2026-09-25", freq="5D",
                                scenario_name="baseline")
    period = results.iloc[-1]
    face, cash = period["CBOFedSecondaryPurchaseFace"], period["CBOFedSecondaryPurchaseCash"]
    assert face > 0.0
    # The Fed buys a bill: its lot carries the discount it paid (cash less face), which it then
    # accretes, not the discount Treasury sold the bill at.
    assert period["CB_AccruedInterestPurchased"] == 0.0
    assert period["CB_PremiumAcquired"] == pytest.approx(cash - face)
    assert period["CB_PremiumAcquired"] < 0.0
    assert period["CB_TreasuryUnamortizedPremium"] == pytest.approx(cash - face)
    # Nothing is earned on the day of purchase.
    assert period["CB_BillDiscountAccretion"] == pytest.approx(0.0, abs=1e-12)


def test_both_closing_checks_reject_tampering(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=0.0)
    frame = _frame(results)
    check_earnings_due_closes(frame, 0.0)
    check_fed_balance_sheet_closes(frame)
    understated = frame.copy()
    understated.loc[understated.index[-1], "CB_EarningsRemittancesDue"] -= 0.5
    with pytest.raises(ValueError, match="closing earnings due"):
        check_earnings_due_closes(understated, 0.0)
    unbooked = frame.copy()
    unbooked.loc[5:, "CB_OtherDeposits"] += 1.0  # a liability that never drained reserves
    with pytest.raises(ValueError, match="reserves against the Fed balance sheet"):
        check_fed_balance_sheet_closes(unbooked)


def test_verifier_rejects_a_tampered_remittance_and_income(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=0.0)
    assert _verify(results, paths)["max_remittance_bil"] > 0.0
    paid = results.index[results["CB_Remittance"] > 0.0][0]
    for column in ("CBORemittanceCashEffect", "CB_Remittance"):
        tampered = results.copy()
        tampered.loc[paid, column] += 1.0
        with pytest.raises(ValueError, match="differs from its rebuild"):
            _verify(tampered, paths)
    tampered = results.copy()
    tampered.loc[paid, "CB_IORBExpense"] -= 1.0
    with pytest.raises(ValueError, match="CB_IORBExpense|interest payable"):
        _verify(tampered, paths)
    tampered = results.copy()
    tampered["CBOTable43RemittanceInFiscalFlow"] += 0.01
    with pytest.raises(ValueError, match="Table 4-3"):
        _verify(tampered, paths)


def _tampered_runner_run(tmp_path, tamper, *, refresh_output_manifest: bool = True):
    """A production run whose results are rewritten after the fact, with every hash of the
    results file in the run manifest refreshed to match (or, without
    ``refresh_output_manifest``, every hash but the output manifest's own record)."""

    from functools import partial

    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo._json import read_json, sha256_file, write_json
    from tdcsim_cbo.verifier import verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _refresh_result_hashes, _runner_baseline_and_scenarios

    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path, remittance_bil=1.0)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    verified = verify_scenario_run(run.output_dir)
    assert verified["status"] == "pass" and verified["recomputed"]["fed_lots_bound_to_final_portfolio"] is True
    results = pd.read_csv(run.results_path)
    tamper(results)
    results.to_csv(run.results_path, index=False)
    manifest = read_json(run.manifest_path)
    if refresh_output_manifest:
        _refresh_result_hashes(run.output_dir, manifest)
    else:  # the audit's refresh: the output inventories, not the output manifest's own record
        record = {"sha256": sha256_file(run.results_path), "bytes": run.results_path.stat().st_size}
        for item in manifest["outputs"]:
            if item["relative_path"] == run.results_path.relative_to(run.output_dir).as_posix():
                item.update(record)
        for item in manifest["output_hashes"]:
            if item["path"] == run.results_path.name:
                item.update(record)
    write_json(run.manifest_path, manifest)
    replay = partial(verify_scenario_run, baseline_package=baseline.package_path, attestation=baseline.attestation.path)
    return run.output_dir, verify_scenario_run, replay


def _fabricate_closing_bill_accretion(results):
    """The audit's probe: 1 billion of bill accretion on the last day, carried consistently
    through the premium, Treasury income, net income, earnings due and deferred asset."""

    last = results.index[-1]
    for column in ("CB_BillDiscountAccretion", "CB_TreasuryUnamortizedPremium", "CB_TreasuryInterestIncome",
                   "CB_NetIncome", "CB_EarningsRemittancesDue"):
        results.loc[last, column] += 1.0
    results.loc[last, "CB_DeferredAsset"] = max(0.0, -results.loc[last, "CB_EarningsRemittancesDue"])


def test_verifier_rebuilds_fed_amortization_from_the_lots_not_the_reported_totals(tmp_path):
    """Every hash refreshed and every reported total consistent: only rebuilding the Fed's
    amortization from its opening lots and trades exposes the extra billion, locally and
    in the attested replay."""

    output_dir, verify, replay = _tampered_runner_run(tmp_path, _fabricate_closing_bill_accretion)
    with pytest.raises(ValueError, match="CB_BillDiscountAccretion against the Fed lot journal"):
        verify(output_dir)
    # The replay recomputes the results, which now differ from the delivered ones.
    with pytest.raises(ValueError, match="bounded engine replay output mismatch: results.sha256"):
        replay(output_dir)


def test_verifier_rebuilds_the_fed_receivable_from_its_lots_terms(tmp_path):
    """The audit's second probe: a billion of coupon income on the last day, carried through
    the receivable, Treasury income, net income, earnings due and deferred asset, with every
    hash refreshed. The receivable the delivered lots' terms imply exposes it."""

    def fabricate(results):
        last = results.index[-1]
        for column in ("CB_CouponAccrual", "CB_TreasuryAccruedInterest", "CB_TreasuryInterestIncome", "CB_NetIncome",
                       "CB_EarningsRemittancesDue"):
            results.loc[last, column] += 1.0
        results.loc[last, "CB_DeferredAsset"] = max(0.0, -results.loc[last, "CB_EarningsRemittancesDue"])

    output_dir, verify, _ = _tampered_runner_run(tmp_path, fabricate)
    # The run closes on a fiscal-year end, where the year-end rebuild catches it first.
    with pytest.raises(ValueError, match="receivable (at 2026-09-30 )?against its lots' terms"):
        verify(output_dir)


def test_verifier_rebuilds_the_fed_receivable_at_every_fiscal_year_end(tmp_path):
    """The audit's boundary probe: 0.1 of coupon income earned on 2026-10-01 booked a day early,
    with that day's receivable, income, earnings due and deferred asset moved to match. Annual
    income moves across the fiscal-year boundary; only the receivable the Fed's lots imply at
    September 30 exposes it."""

    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=-100.0)
    _verify(results, paths)
    moved = results.copy()
    for column in ("CB_CouponAccrual", "CB_TreasuryInterestIncome", "CB_NetIncome"):
        moved.loc["2026-09-30", column] += 0.1
        moved.loc["2026-10-01", column] -= 0.1
    for column in ("CB_TreasuryAccruedInterest", "CB_EarningsRemittancesDue"):
        moved.loc["2026-09-30", column] += 0.1
    moved.loc["2026-09-30", "CB_DeferredAsset"] = max(0.0, -moved.loc["2026-09-30", "CB_EarningsRemittancesDue"])
    with pytest.raises(ValueError, match="receivable at 2026-09-30 against its lots' terms"):
        _verify(moved, paths)


def test_the_fed_year_end_lots_are_each_recorded_once_and_sum_to_its_stock(tmp_path):
    """The audit's probe: a second copy of a lot, its face sized to accrue the 0.1 moved across
    the fiscal-year boundary, hid the shift because only the set of lot IDs was compared. Each
    lot may appear once on a day, and the lots recorded must add up to the Fed's reported stock."""

    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=-100.0)
    lots = results.attrs["fed_year_end_lots"]
    shifted = results.copy()
    for column in ("CB_CouponAccrual", "CB_TreasuryInterestIncome", "CB_NetIncome"):
        shifted.loc["2026-09-30", column] += 0.1
        shifted.loc["2026-10-01", column] -= 0.1
    for column in ("CB_TreasuryAccruedInterest", "CB_EarningsRemittancesDue"):
        shifted.loc["2026-09-30", column] += 0.1
    shifted.loc["2026-09-30", "CB_DeferredAsset"] = max(0.0, -shifted.loc["2026-09-30", "CB_EarningsRemittancesDue"])
    copy_lot = lots.iloc[[0]].copy()
    copy_lot["FaceValue"] = copy_lot["AdjustedPrincipal"] = 0.1 / (0.04 / 2 * 1 / 183)  # one day's coupon is 0.1
    copy_lot["CouponRate"] = 0.04
    copy_lot["MaturityDate"] = pd.Timestamp("2027-03-31")  # its last coupon was the day before
    shifted.attrs["fed_year_end_lots"] = pd.concat([lots, copy_lot], ignore_index=True)
    with pytest.raises(ValueError, match="record a lot twice on one day"):
        _verify(shifted, paths)
    inflated = results.copy()  # one copy, but a face the Fed's reported stock does not hold
    inflated.attrs["fed_year_end_lots"] = lots.assign(FaceValue=lots["FaceValue"] * 1.01,
                                                      AdjustedPrincipal=lots["AdjustedPrincipal"] * 1.01)
    with pytest.raises(ValueError, match="against its reported Treasury stock"):
        _verify(inflated, paths)


def test_replay_is_bound_to_the_results_file_delivered(tmp_path):
    """The audit's refresh left the output manifest's own record of the results stale, so the
    replay matched the original bytes while the consumer received others."""

    output_dir, verify, replay = _tampered_runner_run(
        tmp_path, _fabricate_closing_bill_accretion, refresh_output_manifest=False,
    )
    for check in (verify, replay):
        with pytest.raises(ValueError, match="output manifest record differs from the delivered file: results"):
            check(output_dir)


def test_verifier_binds_fed_treasury_income_to_settlement_evidence(tmp_path):
    """Sol's first probe: move income between Treasury lines on a payment day."""

    def move(results):
        day = results.index[results["CB_CouponCash"].abs() > 0.0]
        row = day[0] if len(day) else 2
        results.loc[row, "CB_CouponAccrual"] -= 1.0
        results.loc[row, "CB_BillDiscountAccretion"] += 1.0

    output_dir, verify, _ = _tampered_runner_run(tmp_path, move)
    with pytest.raises(ValueError, match="Fed remittance verification failed"):
        verify(output_dir)


def test_verifier_binds_the_tga_to_the_treasury_cash_journal(tmp_path):
    """Sol's second probe: cut one day's closing TGA by 1 billion."""

    def cut(results):
        results.loc[3, "TGA"] -= 1.0

    output_dir, verify, _ = _tampered_runner_run(tmp_path, cut)
    with pytest.raises(ValueError):
        verify(output_dir)


def test_table_4_3_remittance_stays_in_the_fiscal_flow_while_the_tga_gets_the_modeled_one(tmp_path):
    neutral, _ = _run(tmp_path / "zero", h41_earnings_remittances_due_bil=-1_000.0)
    cbo, paths = _run(tmp_path / "cbo", remittance_bil=2.338, h41_earnings_remittances_due_bil=-1_000.0)
    table_4_3 = cbo["CBOTable43RemittanceInFiscalFlow"].sum()
    assert table_4_3 == pytest.approx(2 * 2.338)  # 2.338 in each of FY2026 and FY2027
    assert cbo["CB_Remittance"].sum() == 0.0
    assert cbo["TDC_Level"].iloc[-1] - neutral["TDC_Level"].iloc[-1] == pytest.approx(0.99 * table_4_3)
    assert neutral["TGA"].iloc[-1] - cbo["TGA"].iloc[-1] == pytest.approx(table_4_3, abs=1e-6)
    _verify(cbo, paths)


def test_the_table_4_3_input_declares_itself_a_fiscal_flow_add_back_not_a_tga_payment():
    """A consumer following the input's own label must not credit the TGA with CBO's path:
    the TGA receives the run's modeled remittance, so the label says add-back only."""

    from forecast_bundle_builders import build_fed_remittance_path_rows, build_primary_deficit_path_rows
    from forecast_paths import CBO_FED_REMITTANCE_ROLE, CBO_TABLE_4_3_FISCAL_FLOW_ROLE

    primary = build_primary_deficit_path_rows(
        scenario_id="baseline", periods=build_simulation_calendar(START, END, "daily"),
        primary_deficit_by_fiscal_year_bil={2026: 1.0, 2027: 1.0},
    )
    rows = build_fed_remittance_path_rows(primary, {"fiscal_year": 2026}, {"transaction_fytd_amt": "0"})
    assert {row["claim_boundary"] for row in rows} == {CBO_TABLE_4_3_FISCAL_FLOW_ROLE}
    assert "fiscal_flow" in CBO_TABLE_4_3_FISCAL_FLOW_ROLE and "not_paid_to_tga" in CBO_TABLE_4_3_FISCAL_FLOW_ROLE
    assert CBO_TABLE_4_3_FISCAL_FLOW_ROLE != CBO_FED_REMITTANCE_ROLE


def test_the_fed_earns_a_tips_deflation_floor_and_the_exports_show_it(tmp_path):
    """The audit's case: a held TIPS redeemed at par 100 with indexed principal 80. Treasury
    pays the Fed 20 above the principal it held; that is Fed income, bound to the payment
    evidence and shown among the exported lines that sum to Treasury income."""

    from fed_income import FED_INCOME_EXPORT_FLOWS, TREASURY_INCOME_TERMS
    from test_cbo_bounded_equivalence_cases import _run_equivalence
    from test_cbo_engine_integration import _minimal_engine_params, _opening_tips_portfolio, _write_macro_path

    paths = _build_temp_forecast_inputs(
        tmp_path / "inputs", cbo_public_debt_target_bil=205.0, pre_issuance_controlled_debt_bil=100.0,
        signed_primary_flow_bil=0.0, fed_income={**SMALL_FED, "h41_earnings_remittances_due_bil": 0.0},
    )
    paths["macro_forecast_path_file"] = _write_macro_path(tmp_path / "inputs", cpi_u_index=80.0)
    opening = _opening_tips_portfolio(face_value=100.0, reference_cpi=100.0)
    opening.loc[:, ["MaturityDate", "AdjustedPrincipal", "IndexRatio", "HolderType", "HolderSubBucket"]] = [
        pd.Timestamp("2026-09-25"), 80.0, 0.8, "CB", "",
    ]
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=0.0)
    params["initial_bonds_df"] = opening
    params["tips_params"] = {"cpi_start_level": 80.0, "cpi_annual_inflation": 0.0, "ref_cpi_lag_months": 0,
                             "default_real_coupon_rate": 0.0}
    results, _, sink = _run_equivalence(tmp_path, params)
    assert results["CB_TIPSFloorTopup"].sum() == pytest.approx(20.0)
    payments = pd.read_csv(sink.output_dir / "tdcsim_period_payment_aggregates.csv.gz")

    def verify(frame):
        return verify_fed_remittances(
            _frame(frame), paths["fed_income_path_file"].parent, lots=results.attrs["fed_lot_journal"],
            opening_portfolio=opening, final_portfolio=None, year_end_lots=results.attrs["fed_year_end_lots"],
            payments=payments,
        )

    verify(results)
    moved = results.copy()  # the same income booked as indexation instead of the floor
    moved["CB_TIPSIndexation"] += moved["CB_TIPSFloorTopup"]
    moved["CB_TIPSFloorTopup"] = 0.0
    with pytest.raises(ValueError, match="CB_TIPSFloorTopup against the run's payments"):
        verify(moved)
    # Every Treasury income line is exported, so the exported lines sum to the exported total.
    assert set(TREASURY_INCOME_TERMS) <= set(FED_INCOME_EXPORT_FLOWS)
    assert FED_INCOME_EXPORT_FLOWS["CB_TIPSFloorTopup"] == "fed_tips_floor_topup_bil"


def test_gse_debt_earns_each_lots_coupon_until_it_matures_and_no_2025_realized_loss_is_grown():
    """GSE debt earns its coupon only while a lot is outstanding (the last matures 2032-07-15),
    and realized Treasury gains and losses come only from the run's own sales; neither may be
    grown from the 2025 statements forever."""

    from datetime import date as day

    source = fed_income_source_fixture()
    lots = source["soma_agency_debt"]["lots"]
    rows = pd.DataFrame(build_fed_income_path_rows(
        scenario_id="baseline", periods=build_simulation_calendar("2026-06-21", "2032-12-31", "daily"),
    )).set_index("period_end")
    par = sum(float(lot["par"]) for lot in lots) / 1e9
    daily_coupon = sum(float(lot["par"]) / 1e9 * float(lot["coupon_pct"]) / 100 for lot in lots) / 365.25
    assert rows.loc["2026-06-22", "agency_debt_start_bil"] == pytest.approx(par)
    assert rows.loc["2026-06-22", "agency_debt_interest_bil"] == pytest.approx(daily_coupon)
    # The class premium at the June 17 holdings date, shared by par times remaining life, is four
    # days into its straight line at the June 21 opening.
    as_of, opening = day(2026, 6, 17), day(2026, 6, 21)
    life = {lot["cusip"]: (day.fromisoformat(lot["maturity_date"]) - as_of).days for lot in lots}
    weight = {lot["cusip"]: float(lot["par"]) * life[lot["cusip"]] for lot in lots}
    expected = sum(fed_net_premiums_bil()["gse"] * weight[key] / sum(weight.values()) * (life[key] - (opening - as_of).days)
                   / life[key] for key in life)
    assert rows.loc["2026-06-22", "agency_debt_premium_start_bil"] == pytest.approx(expected)
    # The first lot (486 million) is repaid on 2029-05-15 and stops earning that day.
    assert rows.loc["2029-05-15", "agency_debt_start_bil"] - rows.loc["2029-05-15", "agency_debt_end_bil"] == pytest.approx(0.486)
    # Freddie Mac 3134A4AA2 matures on Saturday 2031-03-15: interest stops then, and its 316 million
    # is repaid on Monday, the next Fedwire business day, so weekend reserves still hold it.
    assert rows.loc["2031-03-16", "agency_debt_interest_bil"] < rows.loc["2031-03-15", "agency_debt_interest_bil"]
    for weekend in ("2031-03-15", "2031-03-16"):
        assert rows.loc[weekend, "agency_debt_start_bil"] == pytest.approx(rows.loc[weekend, "agency_debt_end_bil"])
    assert rows.loc["2031-03-17", "agency_debt_start_bil"] - rows.loc["2031-03-17", "agency_debt_end_bil"] == pytest.approx(0.316)
    after = rows.loc["2032-07-16":]
    assert (after[["agency_debt_end_bil", "agency_debt_interest_bil", "agency_debt_premium_end_bil"]] == 0.0).all().all()
    banks = source["reserve_banks_2025_mil"]["other_income"]
    kept = sum(float(value) for key, value in banks.items()
               if key not in {"treasury_securities_losses_net", "gse_debt_securities"}) / 1000
    first = rows.iloc[0]
    year_fraction = (pd.Timestamp(first.name) - pd.Timestamp(first["period_start"])).days / 365.25
    growth = (1.04 ** ((day(2026, 6, 22) - day(2025, 7, 2)).days / 365.25))
    assert first["other_income_bil"] == pytest.approx(kept * growth * year_fraction, rel=1e-3)


def test_mbs_income_follows_the_nyfed_runoff_path_from_the_h41_opening():
    source = fed_income_source_fixture()
    periods = build_simulation_calendar("2026-06-21", "2036-09-30", "daily")
    rows = build_fed_income_path_rows(scenario_id="s", periods=periods)
    first, last = rows[0], rows[-1]
    wac = float(source["soma_agency_mbs"]["weighted_average_coupon_pct"]) / 100.0
    assert first["soma_mbs_interest_bil"] == pytest.approx(wac * first["soma_mbs_start_bil"] / 365.25)
    assert first["soma_mbs_start_bil"] == pytest.approx(1_964.798, rel=2e-3)
    dec_2035 = next(row for row in rows if row["period_end"] == "2035-12-31")
    assert dec_2035["soma_mbs_end_bil"] / 1_964.798 == pytest.approx(677.0 / 1_950.6, rel=5e-3)
    assert last["soma_mbs_end_bil"] < dec_2035["soma_mbs_end_bil"]
    # The MBS premium runs down with the holdings it belongs to.
    premiums = fed_net_premiums_bil()
    assert first["soma_mbs_premium_start_bil"] / first["soma_mbs_start_bil"] == pytest.approx(
        premiums["mbs"] / 1_964.798
    )
    # Other liabilities and capital exclude the earnings due carried separately: +55.509.
    assert first["h41_other_liabilities_and_capital_bil"] == pytest.approx(55.509)
    assert first["remunerated_other_deposit_share"] == pytest.approx(220.5 / 259.503)


def test_net_premiums_split_by_class_and_reconcile_to_the_june_h41():
    premiums = fed_net_premiums_bil()
    assert sum(premiums.values()) == pytest.approx(215.113 - 25.650)
    assert premiums["treasury"] == pytest.approx(156.283 * 189.463 / 193.870, rel=1e-6)


def test_source_transcription_names_every_packaged_document():
    source = fed_income_source_fixture()
    assert {doc["file"] for doc in source["documents"].values()} >= {
        "fed/h41_2026-06-18.htm",
        "fed/fomc_implementation_note_2026-06-17.htm",
        "fed/nyfed_effr_2026-06-15_2026-06-19.json",
        "fed/z1_2026-06-11_s125s5.htm",
        "fed/reserve_banks_quarterly_report_2026q1.pdf",
        "fed/fam_chapter6_reporting_requirements.htm",
    }
    assert date.fromisoformat(source["h41_wednesday_levels_mil"]["as_of_date"]) == date(2026, 6, 17)


def _document_text(path) -> str:
    import html
    import re
    import shutil
    import subprocess

    if path.suffix == ".pdf":
        pdftotext = shutil.which("pdftotext")
        assert pdftotext, "pdftotext is required to verify the Fed income transcription"
        text = subprocess.run([pdftotext, "-layout", str(path), "-"], check=True, capture_output=True, text=True).stdout
    else:
        text = html.unescape(re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="replace")))
    return re.sub(r"\s+", " ", text)


@pytest.mark.integration
def test_fed_income_transcription_matches_the_source_documents():
    """Every transcribed figure must appear in the document whose hash the package pins."""

    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "data" / "fed_income_sources"
    source = fed_income_source_fixture()
    texts = {}
    for key, document in source["documents"].items():
        path = root / Path(document["file"]).name
        assert path.is_file(), f"Fed income source document is missing: {path.name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == document["sha256"], key
        if path.suffix in {".pdf", ".htm"}:
            texts[key] = _document_text(path)

    def published(value: str) -> str:
        return f"{float(value):,.0f}"

    for key, value in source["h41_wednesday_levels_mil"].items():
        if key in {"as_of_date", "tables"}:
            continue
        assert published(value) in texts["h41"], key
    assert "Wednesday Jun 17, 2026" in texts["h41"]
    note = texts["fomc_implementation_note"]
    rates = source["administered_rates_pct"]
    assert f"interest rate paid on reserve balances at {rates['interest_on_reserve_balances']} percent" in note
    assert f"offering rate of {rates['overnight_reverse_repo_offering_rate']} percent" in note
    assert "Effective June 18, 2026" in note
    report = texts["nyfed_soma_annual_report"]
    assert "(4.0 percent)" in report and "early 2030" in report
    statements = texts["reserve_banks_financial_statements"]
    banks = source["reserve_banks_2025_mil"]
    for value in (banks["operating_expenses"], banks["dividends_on_capital_stock"],
                  banks["remittances_transferred_while_deferred_asset_grew"], banks["deferred_asset_year_end"],
                  *banks["excluded_valuation_items"].values()):
        assert published(value).lstrip("-") in statements, value
    for value in banks["other_income"].values():
        if value not in {"216", "-14"}:  # sums of two published lines
            assert published(value).lstrip("-") in statements, value
    import json

    effr = json.loads((root / Path(source["documents"]["nyfed_effr"]["file"]).name).read_text())
    opening_effr = [row for row in effr["refRates"] if row["effectiveDate"] == "2026-06-18"]
    assert [str(row["percentRate"]) for row in opening_effr] == [rates["effective_federal_funds_rate_2026_06_18"]]
    holdings = json.loads((root / Path(source["documents"]["soma_agency_holdings"]["file"]).name).read_text())
    debts = sorted(
        (row["cusip"], row["maturityDate"], row["coupon"], row["parValue"])
        for row in (holdings["soma"]["holdings"] if "soma" in holdings else holdings)
        if row["securityType"] == "Agency Debts"
    )
    assert debts == sorted((lot["cusip"], lot["maturity_date"], lot["coupon_pct"], lot["par"])
                           for lot in source["soma_agency_debt"]["lots"])
    operations = json.loads((root / Path(source["documents"]["nyfed_reverse_repo_operations"]["file"]).name).read_text())
    june_18 = [op for op in operations["repo"]["operations"] if op["operationDate"] == "2026-06-18"]
    assert [op["totalAmtAccepted"] / 1e9 for op in june_18] == [float(source["opening_domestic_on_rrp"]["total_accepted_bil"])]
    accounts = texts["financial_accounts_ccp_table"]
    ccp = source["financial_accounts_ccp_federal_reserve_deposits_bil"]
    assert ccp["series"] in accounts and "Federal Reserve deposits" in accounts and ccp["2026Q1"] in accounts
    quarterly = texts["reserve_banks_quarterly_report"]
    for key, value in source["reserve_banks_2026q1_mil"].items():
        if key.endswith("_pct"):
            assert f"{value}%" in quarterly, key
        elif key == "amortization_notes_and_bonds_q1":  # notes (1,262) plus bonds (2,686)
            assert "(1,262)" in quarterly and "(2,686)" in quarterly and value == "-3948"
        elif key != "tables":
            assert published(value).lstrip("-") in quarterly, key
    manual = texts["financial_accounting_manual_chapter_6"]
    assert "lower of the liability (credit) balance" in manual and "Special procedures exist for the first remittance" in manual
