"""Treasury's calendar in the CBO lane: business days, auction cadence and reopenings.

The thesis deck shows a month of settlements and every security's dates, so calendar
artifacts are visible: a payment or settlement on a weekend, a 52-week bill every week,
TIPS every month, or a 10-year note maturing on a date Treasury never uses. These tests
pin the calendar to Treasury's practice and the engine to it.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from issuance_calibration import (
    BILL_TERM_WEEKS,
    bill_maturity,
    COUPON_TENORS,
    SETTLEMENT_CALENDAR,
    IssuanceCalibrationError,
    check_settlement_calendar,
    dated_date,
    engine_rule_profile,
    maturity_from_dated,
    settlement_events,
    settlement_rules,
)
from sim_engine import run_simulation
from simulation_calendar import is_fedwire_business_day, is_securities_business_day
from test_issuance_rule import ENGINE_END, ENGINE_START, _engine_params, _portfolio, _portfolio_row


def _profile(calendar=SETTLEMENT_CALENDAR, *, nominal_only: bool = False) -> dict:
    kinds = ("nominal",) if nominal_only else tuple(COUPON_TENORS)
    coupons = [(kind, tenor) for kind in kinds for tenor in COUPON_TENORS[kind]]
    return engine_rule_profile(
        target_path=[{"date": None, "bill_share_target": 0.2}],
        bill_terms=[{"weeks": weeks, "share": 1.0 / len(BILL_TERM_WEEKS)} for weeks in BILL_TERM_WEEKS],
        coupon_terms=[{"kind": kind, "tenor_years": tenor, "share": 1.0 / len(coupons)} for kind, tenor in coupons],
        settlement_calendar={
            **calendar,
            "coupons": [entry for entry in calendar["coupons"] if entry["kind"] in kinds],
        },
    )


# --- The calendar ------------------------------------------------------------------------


def test_each_security_settles_as_often_as_treasury_auctions_it() -> None:
    """Treasury sells the 52-week bill every four weeks and each TIPS a few times a year;
    issuing them every week or month would overstate how often they come to market."""

    events = settlement_events(_profile(), date(2027, 1, 1), date(2027, 12, 31))
    counts = Counter(event.key for event in events)
    for weeks in (4, 6, 8, 13, 17, 26):
        assert counts[("bill", float(weeks))] == 52
    assert counts[("bill", 52.0)] == 13
    for tenor in COUPON_TENORS["nominal"]:
        assert counts[("Fixed", float(tenor))] == 12
    assert counts[("FRN", 2.0)] == 12
    assert (counts[("TIPS", 5.0)], counts[("TIPS", 10.0)], counts[("TIPS", 30.0)]) == (4, 6, 2)

    def months(key, issue):
        return sorted(event.nominal.month for event in events if event.key == key and event.issue == issue)

    assert months(("TIPS", 5.0), "new") == [4, 10] and months(("TIPS", 5.0), "reopening") == [6, 12]
    assert months(("TIPS", 10.0), "new") == [1, 7]
    assert months(("TIPS", 10.0), "reopening") == [3, 5, 9, 11]
    assert months(("TIPS", 30.0), "new") == [2] and months(("TIPS", 30.0), "reopening") == [8]
    for tenor in (10.0, 20.0, 30.0):
        assert months(("Fixed", tenor), "new") == [2, 5, 8, 11]
    assert months(("FRN", 2.0), "new") == [1, 4, 7, 10]
    for tenor in (2.0, 3.0, 5.0, 7.0):
        assert months(("Fixed", tenor), "reopening") == []
    assert all(is_securities_business_day(event.settle) for event in events)


def test_settlement_days_follow_treasurys_rolls() -> None:
    """Month-end issues roll forward to the next business day, TIPS settle on the month's
    last business day, FRN reopenings on its last Friday, and nothing settles on Good
    Friday although Fedwire is open (Treasury settled the March 2018 TIPS on the Thursday
    and the March 2018 FRN reopening on the Monday)."""

    events = settlement_events(_profile(), date(2026, 6, 22), date(2033, 12, 31))
    settles = {(event.key, event.nominal): event.settle for event in events}
    assert settles[(("Fixed", 2.0), date(2026, 10, 31))] == date(2026, 11, 2)  # Saturday month end
    assert settles[(("FRN", 2.0), date(2026, 12, 25))] == date(2026, 12, 28)  # last Friday is Christmas
    assert settles[(("TIPS", 10.0), date(2029, 3, 31))] == date(2029, 3, 29)  # Good Friday 2029-03-30
    assert settles[(("FRN", 2.0), date(2027, 3, 26))] == date(2027, 3, 29)  # Good Friday 2027-03-26
    assert settles[(("Fixed", 10.0), date(2033, 4, 15))] == date(2033, 4, 18)  # Good Friday 2033-04-15
    assert settles[(("bill", 13.0), date(2026, 11, 26))] == date(2026, 11, 27)  # Thanksgiving
    fifty_two = sorted(event.settle for event in events if event.key == ("bill", 52.0))[:4]
    assert fifty_two == [date(2026, 7, 9), date(2026, 8, 6), date(2026, 9, 3), date(2026, 10, 1)]
    # Payments follow Fedwire, which is open on Good Friday; the market is not.
    assert is_fedwire_business_day("2033-04-15") and not is_securities_business_day("2033-04-15")
    assert is_securities_business_day("2026-04-03")  # Good Friday on a jobs-report Friday
    assert not is_securities_business_day("2026-07-03")  # Independence Day observed


def test_new_issues_are_dated_and_mature_on_treasurys_dates() -> None:
    """A 20-year bond or a TIPS is dated on the 15th of its month and settles at month end;
    a month-end note matures on a month end even when it settled on the 1st or 2nd."""

    rules = settlement_rules(_profile())
    assert maturity_from_dated(rules[("Fixed", 20.0)], dated_date(rules[("Fixed", 20.0)], date(2026, 8, 31))) == date(2046, 8, 15)
    assert maturity_from_dated(rules[("TIPS", 30.0)], dated_date(rules[("TIPS", 30.0)], date(2027, 2, 28))) == date(2057, 2, 15)
    assert maturity_from_dated(rules[("Fixed", 2.0)], dated_date(rules[("Fixed", 2.0)], date(2026, 2, 28))) == date(2028, 2, 29)
    assert maturity_from_dated(rules[("FRN", 2.0)], dated_date(rules[("FRN", 2.0)], date(2026, 1, 31))) == date(2028, 1, 31)


def _auction_rows_from_calendar(start: date, end: date) -> list[dict[str, str]]:
    """A synthetic auction file that is exactly the calendar's settlements."""

    profile = _profile()
    rules = settlement_rules(profile)
    terms = {("Fixed", float(t)): ("Note" if t <= 10 else "Bond", "nominal") for t in COUPON_TENORS["nominal"]}
    rows = []
    originals: dict[tuple[str, float], str] = {}
    bills_maturing: set[date] = set()
    for event in settlement_events(profile, start - timedelta(days=400), end):
        key = event.key
        row = {
            "issue_date": event.settle.isoformat(),
            "cash_management_bill_cmb": "No",
            "floating_rate": "Yes" if key[0] == "FRN" else "No",
            "inflation_index_security": "Yes" if key[0] == "TIPS" else "No",
            "original_issue_date": "null",
            "original_security_term": "null",
        }
        if key[0] == "bill":
            maturity = bill_maturity(event.nominal, int(key[1]))
            row.update(security_type="Bill", security_term=f"{int(key[1])}-Week", maturity_date=maturity.isoformat())
            if maturity in bills_maturing:
                row["original_issue_date"] = (event.settle - timedelta(days=7)).isoformat()
            bills_maturing.add(maturity)
        else:
            security_type = terms.get(key, ("Note", ""))[0] if key[0] == "Fixed" else ("Note" if key[1] <= 10 else "Bond")
            row.update(security_type=security_type, security_term=f"{int(key[1])}-Year")
            if event.issue == "new":
                maturity = maturity_from_dated(rules[key], dated_date(rules[key], event.nominal))
                row["maturity_date"] = maturity.isoformat()
                originals[key] = event.settle.isoformat()
            else:
                row["maturity_date"] = "2099-01-01"
                row["original_issue_date"] = originals.get(key, (event.settle - timedelta(days=40)).isoformat())
                row["original_security_term"] = f"{int(key[1])}-Year"
        rows.append(row)
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_calendar_check_rejects_a_year_that_departs_from_treasurys_calendar(tmp_path: Path) -> None:
    """The package build checks the calendar against the auction file, so a change in
    Treasury's calendar stops the build instead of silently mis-timing issuance."""

    rows = _auction_rows_from_calendar(date(2025, 6, 21), date(2026, 6, 20))
    report = check_settlement_calendar(_write(tmp_path / "ok.csv", rows), simulation_start_date="2026-06-21")
    assert report["status"] == "pass" and report["reopenings"] > 0 and report["new_issues"] > 0

    moved = [dict(row) for row in rows]
    target = next(row for row in moved if row["security_term"] == "10-Year" and row["issue_date"] >= "2025-07-01")
    target["issue_date"] = "2025-07-16" if target["issue_date"] == "2025-07-15" else "2025-07-15"
    with pytest.raises(IssuanceCalibrationError, match="off the calendar|has no auction"):
        check_settlement_calendar(_write(tmp_path / "moved.csv", moved), simulation_start_date="2026-06-21")

    relabeled = [dict(row) for row in rows]
    reopening = next(
        row for row in relabeled
        if row["inflation_index_security"] == "Yes" and row["original_issue_date"] != "null" and row["issue_date"] >= "2025-07-01"
    )
    reopening["original_issue_date"] = "null"
    with pytest.raises(IssuanceCalibrationError, match="auction is new, calendar says reopening"):
        check_settlement_calendar(_write(tmp_path / "relabeled.csv", relabeled), simulation_start_date="2026-06-21")


# --- The engine on Treasury's calendar -----------------------------------------------------


def _refunding_securities(ten_year_coupon: float = 0.04) -> list[dict]:
    """The May 2026 10-, 20- and 30-year issues that June and July reopen."""

    rows = [
        _portfolio_row(903, security_type="Fixed", years=10.0, maturity="2036-05-15", face=60.0),
        _portfolio_row(904, security_type="Fixed", years=20.0, maturity="2046-05-15", face=20.0),
        _portfolio_row(905, security_type="Fixed", years=30.0, maturity="2056-05-15", face=20.0),
    ]
    rows[0]["CouponRate"] = ten_year_coupon
    return rows


def _params(tmp_path: Path, extra_rows: list[dict]) -> dict:
    params = _engine_params(tmp_path, profile=_profile(nominal_only=True), mode="cbo_operating_cash_target_financing_v1")
    params["initial_bonds_df"] = _portfolio(
        [*params["initial_bonds_df"].to_dict("records"), *_refunding_securities(), *extra_rows]
    )
    return params


def _run(tmp_path: Path, extra_rows: list[dict], *, ten_year_coupon: float = 0.04) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = _engine_params(tmp_path, profile=_profile(nominal_only=True), mode="cbo_operating_cash_target_financing_v1")
    params["initial_bonds_df"] = _portfolio(
        [*params["initial_bonds_df"].to_dict("records"), *_refunding_securities(ten_year_coupon), *extra_rows]
    )
    return run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")


def test_payments_due_on_a_closed_day_are_paid_the_next_business_day_in_full_and_no_more(tmp_path: Path) -> None:
    """2026-08-15 is a Saturday: a note maturing then and a coupon due then are paid on
    Monday the 17th, the coupon at its contractual amount, and over the weekend no cash
    moves, nothing is decided or settled, and every new tranche is issued on a
    securities business day."""

    maturing = _portfolio_row(901, security_type="Fixed", years=2.0, maturity="2026-08-15", face=50.0)
    paying = _portfolio_row(902, security_type="Fixed", years=3.0, maturity="2028-08-15", face=40.0)
    results, portfolio = _run(tmp_path, [maturing, paying])

    for weekend in ("2026-08-15", "2026-08-16"):
        day = results.loc[weekend]
        assert day["PrincipalPaid_Bonds"] == 0.0 and day["InterestPaid_Bonds"] == 0.0
        assert day["NewDebtIssued"] == 0.0 and day["IssuanceNeedProceeds"] == 0.0
        assert day["TGA"] == results.loc["2026-08-14", "TGA"]
    monday = results.loc["2026-08-17"]
    assert monday["PrincipalPaid_Bonds"] == pytest.approx(50.0)
    assert monday["InterestPaid_Bonds"] == pytest.approx((50.0 + 40.0) * 0.04 / 2.0)
    closed = [day for day in results.index[1:] if not is_fedwire_business_day(day)]
    assert closed and (results.loc[closed, ["PrincipalPaid_Bonds", "InterestPaid_Bonds", "NewDebtIssued"]] == 0.0).all().all()
    issued = pd.to_datetime(portfolio["IssueDate"])
    assert all(is_securities_business_day(day) for day in issued[issued > pd.Timestamp(ENGINE_START)])


def test_a_reopening_keeps_the_reopened_notes_terms_and_sells_at_market(tmp_path: Path) -> None:
    """Treasury sells a 10-year note new in May and reopens it in June and July. The July
    tranche matures with the May note and pays its coupon, and buyers pay its market value
    plus the interest accrued since May 15, so face differs from cash. August is a new
    issue maturing on August 15 ten years later."""

    results, portfolio = _run(tmp_path, [], ten_year_coupon=0.03)
    tranches = portfolio[(portfolio["OriginalMaturityYears"] == 10.0) & (portfolio["BondID"] != 903)]
    july = tranches[pd.to_datetime(tranches["IssueDate"]) == pd.Timestamp("2026-07-15")]
    august = tranches[pd.to_datetime(tranches["IssueDate"]) == pd.Timestamp("2026-08-17")]
    assert not july.empty and not august.empty
    assert set(pd.to_datetime(july["MaturityDate"])) == {pd.Timestamp("2036-05-15")}
    assert set(july["CouponRate"]) == {0.03}
    # Two months of accrued 3% coupon plus a discount to par at the higher market yield.
    assert (july["IssuePriceRatio"] < 1.0).all()
    assert set(pd.to_datetime(august["MaturityDate"])) == {pd.Timestamp("2036-08-15")}
    assert set(pd.to_datetime(august["DatedDate"])) == {pd.Timestamp("2026-08-15")}
    assert not (august["CouponRate"] == 0.03).all()


def test_a_closed_day_with_fiscal_cash_is_rejected(tmp_path: Path) -> None:
    """A package that books fiscal flows on a weekend would move the TGA while Fedwire is
    shut; the engine refuses it rather than settle or pay around it."""

    params = _params(tmp_path, [])
    path = Path(params["baseline_input_paths"]["primary_deficit_path_file"])
    frame = pd.read_csv(path)
    frame.loc[frame["period_end"] == "2026-08-15", "primary_deficit_bil"] = 1.0
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="a day Fedwire is closed"):
        run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")


def test_the_fed_does_not_trade_while_the_securities_market_is_closed(tmp_path: Path) -> None:
    """Friday 2026-07-03 observes Independence Day: Fedwire pays the Fed's maturing bill,
    but securities do not trade, so the Fed buys nothing until Monday and no trade can
    touch a security that matured on a closed day before it is paid."""

    from forecast_bundle_builders import build_fed_holdings_path_rows
    from simulation_calendar import build_simulation_calendar
    from test_cbo_engine_integration import _write_csv

    params = _params(
        tmp_path,
        [_portfolio_row(912, security_type="Fixed", years=28 / 365.25, maturity="2026-07-03", face=10.0, holder="CB")],
    )
    rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=build_simulation_calendar(ENGINE_START, ENGINE_END, "daily"),
        opening_state_date=ENGINE_START,
        opening_cb_holdings_bil=10.0,
        cbo_fy_end_fed_holdings_bil={2026: 10.0},
        observation_date=ENGINE_START,
        available_date=ENGINE_START,
    )
    params["baseline_input_paths"]["fed_holdings_path_file"] = str(_write_csv(tmp_path / "fed.csv", rows))
    results, _ = run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    closed = [day for day in results.index[1:] if not is_securities_business_day(day)]
    trades = results.loc[closed, ["CBOFedSecondaryPurchaseFace", "CBOFedSecondarySaleCash"]]
    assert pd.Timestamp("2026-07-03") in closed and (trades.abs() < 1e-12).all().all()
    assert results.loc["2026-07-03", "PrincipalPaid_Bonds"] >= 10.0
    assert results.loc["2026-07-03", "DebtHeld_CentralBank"] == pytest.approx(0.0, abs=1e-9)
    assert results.loc["2026-07-06", "DebtHeld_CentralBank"] == pytest.approx(10.0, abs=1e-6)


def test_the_pricer_counts_each_contractual_coupon_once() -> None:
    """Reopenings, Fed transfers and the opening book are priced on contractual coupon
    dates: a month-end note keeps its month-end coupons (no extra August 28), and a note
    the day before a coupon still owes that coupon, so a par note is worth par plus
    almost half a year's interest."""

    from sim_pricing import _coupon_dates_around_settlement, value_treasury_security

    dates, last = _coupon_dates_around_settlement(
        settlement_date=pd.Timestamp("2026-09-30"), maturity_date=pd.Timestamp("2028-08-31"),
        first_interest_payment_date=None, frequency=2,
    )
    assert dates == [pd.Timestamp(day) for day in ("2027-02-28", "2027-08-31", "2028-02-29", "2028-08-31")]
    assert last == pd.Timestamp("2026-08-31")
    dates, last = _coupon_dates_around_settlement(
        settlement_date=pd.Timestamp("2026-10-14"), maturity_date=pd.Timestamp("2036-10-15"),
        first_interest_payment_date=None, frequency=2,
    )
    assert dates[0] == pd.Timestamp("2026-10-15") and last == pd.Timestamp("2026-04-15") and len(dates) == 21
    value = value_treasury_security(
        settlement_date=pd.Timestamp("2026-10-14"), maturity_date=pd.Timestamp("2036-10-15"),
        coupon_rate=0.04, discount_yield=0.04, face_value=1.0,
    )
    assert value["clean"] == pytest.approx(1.0, abs=1e-5)
    assert value["accrued"] == pytest.approx(0.02 * 182 / 183)


def test_bills_mature_on_their_weekly_cycle_moved_off_holidays() -> None:
    """The 13-week bill that settled the Friday after Thanksgiving 2025 matured on the
    Thursday cycle (February 26), and a cycle date on a holiday moves to the next business
    day (the 26-week bill due Christmas 2025 matured December 26)."""

    from issuance_calibration import bill_maturity

    assert bill_maturity(date(2025, 11, 27), 13) == date(2026, 2, 26)
    assert bill_maturity(date(2025, 6, 26), 26) == date(2025, 12, 26)


def test_the_verifier_books_a_premium_as_the_run_does(tmp_path: Path) -> None:
    """A reopening sold above par costs less than nothing; the bounded verifier must sum
    the signed cost the run booked, not floor each row at zero."""

    import gzip

    from tdcsim_cbo import verifier

    (tmp_path / "outputs").mkdir()
    row = {
        "period_start": "2026-07-14", "period_end": "2026-07-15", "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route", "issuance_leg": "operating_cash_target_financing",
        "instrument_type": "Fixed", "maturity_bucket": "notes", "face_issued_bil": "100.0",
        "cash_proceeds_bil": "101.0", "discount_or_premium_bil": "-1.0", "issue_cost_bil": "-1.0",
        "weighted_original_term_years": "10.0", "coupon_rate_decimal": "0.05", "reference_rate_decimal": "",
        "spread_bps": "", "issue_yield_decimal": "0.04", "flow_count": "1",
    }
    with gzip.open(tmp_path / "outputs" / "tdcsim_period_issuance_aggregates.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=verifier.ISSUANCE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    period = ("2026-07-14", "2026-07-15")
    evidence = {period: {"NewDebtIssued": 100.0, "AuctionProceeds": 101.0, "IssueDiscountCost_Period": -1.0}}
    totals = verifier._verify_bounded_issuance(tmp_path, {"issuance": {"row_count": 1}}, [period], evidence)
    assert totals["period"][period]["issue_discount_cost_bil"] == pytest.approx(-1.0)
    evidence[period]["IssueDiscountCost_Period"] = 0.0
    with pytest.raises(verifier.VerificationError, match="issue cost bridge"):
        verifier._verify_bounded_issuance(tmp_path, {"issuance": {"row_count": 1}}, [period], evidence)


def test_a_holiday_shortened_13_week_bill_keeps_the_cbo_bill_quote(monkeypatch) -> None:
    """The 13-week bill settling the Friday after Thanksgiving runs 90 days. It is still
    the bill CBO's 3-month rate quotes, priced over the 90 days it is outstanding, not a
    bill priced off the nominal curve because its days are not 91."""

    import sim_engine
    from bill_quote_basis import discount_price_ratio

    monkeypatch.setattr(sim_engine, "_cbo_macro_value", lambda *args, **kwargs: 3.0)
    pricing = sim_engine._issuance_pricing_context(
        tips_real_curve_years=[], tips_real_curve_rates=[],
        current_yield_curve_years=[0.08, 0.25, 1.0, 10.0], current_yield_curve_rates=[0.06, 0.06, 0.06, 0.06],
        yield_interpolation_method="linear", yield_floor_zero=True, evaluated_nominal_shock=None,
        tips_real_coupon=0.0, frn_spread=0.001, cbo_funding_mode=True, cbo_inputs={}, cbo_scenario_id="test",
    )
    item = sim_engine._quote_supply_item(
        "Fixed", 90 / 365.25, 10.0, face_target_mode=False, issuance_leg="operating_cash_target_financing",
        current_date=pd.Timestamp("2026-11-27"), bill_term_days=91, pricing=pricing, bill_days_to_maturity=90,
    )
    assert item["issue_price_ratio"] == pytest.approx(discount_price_ratio(0.03, 90))


# --- The Fed rolls its maturing holdings at auction ----------------------------------------


def _fed_run(tmp_path: Path, fy_end_target: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The Fed holds a 4-week bill maturing Tuesday 2026-07-07 (20) and a 13-week bill
    maturing Thursday 2026-07-09 (30), and follows a path from 50 to ``fy_end_target``
    by 2026-09-30."""

    from forecast_bundle_builders import build_fed_holdings_path_rows
    from simulation_calendar import build_simulation_calendar
    from test_cbo_engine_integration import _write_csv

    params = _params(
        tmp_path,
        [
            _portfolio_row(910, security_type="Fixed", years=28 / 365.25, maturity="2026-07-07", face=20.0, holder="CB"),
            _portfolio_row(911, security_type="Fixed", years=91 / 365.25, maturity="2026-07-09", face=30.0, holder="CB"),
        ],
    )
    rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=build_simulation_calendar(ENGINE_START, ENGINE_END, "daily"),
        opening_state_date=ENGINE_START,
        opening_cb_holdings_bil=50.0,
        cbo_fy_end_fed_holdings_bil={2026: fy_end_target},
        observation_date=ENGINE_START,
        available_date=ENGINE_START,
    )
    params["baseline_input_paths"]["fed_holdings_path_file"] = str(_write_csv(tmp_path / "fed.csv", rows))
    return run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")


def _assert_cash_rule_closes(results: pd.DataFrame) -> None:
    running = results.iloc[1:]
    excess = running["TGA"] + running["PendingIssuanceProceeds"] - running["CBOOperatingCashTarget"]
    decided = running["IssuanceNeedProceeds"] > 1e-9
    assert excess.min() > -1e-6 and excess[decided].abs().max() < 1e-6


def test_the_fed_rolls_maturing_holdings_into_the_days_auctions_not_the_secondary_market(tmp_path: Path) -> None:
    """On a flat path the Fed replaces each maturing bill with add-ons at the auctions
    settling that day, pro rata across them, paying for them itself: no secondary
    purchase pulls securities, reserves or deposits from private holders, and the
    Treasury's cash rule still closes because the add-on cash is counted the same day."""

    results, portfolio = _fed_run(tmp_path, 50.0)
    assert results.loc["2026-07-07", "FedRolloverFace"] == pytest.approx(20.0)
    assert results.loc["2026-07-09", "FedRolloverFace"] == pytest.approx(30.0)
    # The add-on bills mature inside the window and roll again: every maturity is rolled.
    assert (results["FedRolloverFace"] - results["CBOFedMaturitiesAndRedemptions"]).abs().max() < 1e-9
    assert results["FedRolloverFace"].sum() > 50.0
    assert results["FedRedeemedFace"].abs().max() < 1e-9
    assert results["CBOFedSecondaryPurchaseFace"].abs().max() < 1e-9
    assert results["TDC_SecondaryTrades"].abs().max() < 1e-9
    assert (results["DebtHeld_CentralBank"].iloc[1:] - 50.0).abs().max() < 1e-6
    add_ons = portfolio[(portfolio["HolderType"] == "CB") & (pd.to_datetime(portfolio["IssueDate"]) > pd.Timestamp(ENGINE_START))]
    tuesday = add_ons[pd.to_datetime(add_ons["IssueDate"]) == pd.Timestamp("2026-07-07")]
    weeks = sorted({round(float(years) * 365.25 / 7) for years in tuesday["OriginalMaturityYears"]})
    assert weeks == [4, 8, 17]  # the bills that settle on Tuesdays
    assert (results["CBOFedAuctionRolloverAddons"] <= results["CBOFedMaturitiesAndRedemptions"] + 1e-9).all()
    _assert_cash_rule_closes(results)


def test_the_fed_buys_in_the_secondary_market_only_for_growth(tmp_path: Path) -> None:
    """On a rising path the Fed rolls all of its maturing holdings and buys only the
    path's growth in the secondary market: 30 over the window, not 30 plus the 50 that
    matured, which is what replacing maturities by purchases would churn."""

    results, _ = _fed_run(tmp_path, 80.0)
    assert (results["FedRolloverFace"] - results["CBOFedMaturitiesAndRedemptions"]).abs().max() < 1e-9
    assert results["CBOFedSecondaryPurchaseFace"].sum() == pytest.approx(30.0, abs=1e-6)
    assert results.loc["2026-09-30", "DebtHeld_CentralBank"] == pytest.approx(80.0, abs=1e-6)
    _assert_cash_rule_closes(results)


def test_on_a_falling_path_maturities_run_off_before_the_fed_sells(tmp_path: Path) -> None:
    """When the path falls the Fed rolls only what keeps it on the path and lets the rest
    of a maturity run off; it sells in the secondary market only on days when nothing it
    holds matures."""

    results, _ = _fed_run(tmp_path, 30.0)
    maturity_days = results.index[results["CBOFedMaturitiesAndRedemptions"] > 1e-9]
    assert pd.Timestamp("2026-07-07") in maturity_days and pd.Timestamp("2026-07-09") in maturity_days
    for day in maturity_days:
        row = results.loc[day]
        assert row["FedRedeemedFace"] > 0.0 and row["CBOFedSecondarySaleCash"] == 0.0
        assert row["DebtHeld_CentralBank"] == pytest.approx(row["CBOFedHoldingsTarget"], abs=1e-6)
    assert results["FedRedeemedFace"].sum() > 0.0
    assert results.loc["2026-09-30", "DebtHeld_CentralBank"] == pytest.approx(30.0, abs=1e-6)
    _assert_cash_rule_closes(results)


def test_the_fed_rolls_bills_into_bills_and_coupons_into_coupons(tmp_path: Path) -> None:
    """2026-09-15 is a Tuesday: the 3-, 10- and 30-year notes settle with the Tuesday
    bills. A maturing Fed bill rolls only into the bills, as the Desk does, however much
    coupon issuance is pending."""

    from forecast_bundle_builders import build_fed_holdings_path_rows
    from simulation_calendar import build_simulation_calendar
    from test_cbo_engine_integration import _write_csv

    params = _params(
        tmp_path,
        [_portfolio_row(913, security_type="Fixed", years=91 / 365.25, maturity="2026-09-15", face=20.0, holder="CB")],
    )
    rows = build_fed_holdings_path_rows(
        scenario_id="baseline",
        periods=build_simulation_calendar(ENGINE_START, ENGINE_END, "daily"),
        opening_state_date=ENGINE_START,
        opening_cb_holdings_bil=20.0,
        cbo_fy_end_fed_holdings_bil={2026: 20.0},
        observation_date=ENGINE_START,
        available_date=ENGINE_START,
    )
    params["baseline_input_paths"]["fed_holdings_path_file"] = str(_write_csv(tmp_path / "fed.csv", rows))
    results, portfolio = run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    assert results.loc["2026-09-15", "FedRolloverFace"] == pytest.approx(20.0)
    add_ons = portfolio[(portfolio["HolderType"] == "CB") & (pd.to_datetime(portfolio["IssueDate"]) == pd.Timestamp("2026-09-15"))]
    assert not add_ons.empty and (add_ons["OriginalMaturityYears"] <= 1.0).all()
    coupons = portfolio[(pd.to_datetime(portfolio["IssueDate"]) == pd.Timestamp("2026-09-15")) & (portfolio["OriginalMaturityYears"] > 1.0)]
    assert not coupons.empty and (coupons["HolderType"] != "CB").all()


def test_a_tips_add_on_is_sized_in_adjusted_principal() -> None:
    """A TIPS reopened at an index ratio of 1.1 adds 1.1 of stock per unit face. The Fed's
    face across securities stays proportional to what is offered, scaled so the stock it
    adds is the stock it rolls, and it lands on its path instead of overshooting it."""

    from sim_engine import _fed_rollover_allocation

    profile = _profile()
    due = {("TIPS", 10.0): object(), ("Fixed", 2.0): object()}
    # Pending cash 55 at 1.1 per face and 50 at par: 50 face pending in each.
    pending = {("TIPS", 10.0): ["TIPS", 10.0, None, 55.0], ("Fixed", 2.0): ["Fixed", 2.0, None, 50.0]}
    allocation = _fed_rollover_allocation(
        {"bills": 0.0, "coupons": 22.0},
        due_events=due,
        pending=pending,
        issuance_profile=profile,
        unit_prices=lambda keys: {("TIPS", 10.0): (1.1, 1.1), ("Fixed", 2.0): (1.0, 1.0)},
    )
    stock = sum(face * per_face for face, _price, per_face in allocation.values())
    assert stock == pytest.approx(22.0)
    # Equal face offered, so equal face taken, as the Desk's tenders are in face.
    assert allocation[("TIPS", 10.0)][0] == pytest.approx(allocation[("Fixed", 2.0)][0])
