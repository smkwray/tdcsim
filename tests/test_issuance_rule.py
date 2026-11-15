"""Bill-share issuance rule, its auction calibration, and operating-cash financing.

Gross Treasury issuance is mostly bills because bills roll over several times a year;
outstanding debt is only about a fifth bills. A rule that splits gross issuance by fixed
shares therefore terms the debt out as bills mature. These tests pin the replacement:
bills fill toward an outstanding bill-share target measured from the opening book, the
rest follows coupon shares measured from public auction volume, and cash financing keeps
the TGA on its operating-cash path so scenario cost differences land in debt.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from issuance_calibration import (
    AUCTION_REQUIRED_COLUMNS,
    CALIBRATION_FILE,
    CALIBRATION_SCHEMA_VERSION,
    SETTLEMENT_CALENDAR,
    IssuanceCalibrationError,
    bill_issuance_amount,
    build_auction_calibration,
    build_issuance_calibration,
    controlled_debt_and_bills,
    engine_rule_profile,
    normalize_target_path,
    target_for_date,
)
from sim_engine import (
    _build_bill_share_supply_schedule,
    _issuance_pricing_context,
    run_simulation,
)
from sim_pricing import get_security_category_for_prefs
from simulation_calendar import build_simulation_calendar
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES
from tdcsim_cbo._json import write_json
from test_cbo_engine_integration import _build_temp_forecast_inputs, _minimal_engine_params


START = "2026-06-21"


# --- Fixture helpers shared with the package-level tests --------------------------------


FIXTURE_BILL_TERMS = [
    {"weeks": 4, "share": 0.5},
    {"weeks": 13, "share": 0.3},
    {"weeks": 26, "share": 0.2},
]
FIXTURE_COUPON_TERMS = [
    {"kind": "nominal", "tenor_years": 2, "share": 0.5},
    {"kind": "nominal", "tenor_years": 10, "share": 0.3},
    {"kind": "nominal", "tenor_years": 30, "share": 0.2},
]
# The registry calendar restricted to the fixture's securities.
FIXTURE_SETTLEMENT_CALENDAR = {
    "bills": [entry for entry in SETTLEMENT_CALENDAR["bills"] if entry["weeks"] in {4, 13, 26}],
    "coupons": [
        entry
        for entry in SETTLEMENT_CALENDAR["coupons"]
        if entry["kind"] == "nominal" and entry["tenor_years"] in {2, 10, 30}
    ],
}


def write_fixture_issuance_calibration(inputs_dir: Path, *, bill_share_target: float = 0.2) -> dict:
    """Write a package calibration with nominal coupons only (no TIPS or FRN curves needed)."""

    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_file": "sources/fiscaldata/auctions_query.csv",
        "bill_terms": FIXTURE_BILL_TERMS,
        "coupon_terms": FIXTURE_COUPON_TERMS,
        "settlement_calendar": FIXTURE_SETTLEMENT_CALENDAR,
        "bill_share_target": bill_share_target,
    }
    write_json(Path(inputs_dir) / CALIBRATION_FILE, payload)
    return payload


def _auction_row(
    issue: str,
    security_type: str,
    term: str,
    total: float,
    soma: float = 0.0,
    *,
    cmb: str = "No",
    frn: str = "No",
    tips: str = "No",
    days: int | None = None,
) -> dict[str, str]:
    issue_date = date.fromisoformat(issue)
    maturity = issue_date + timedelta(days=days if days is not None else 365)
    return {
        "record_date": issue,
        "security_type": security_type,
        "security_term": term,
        "issue_date": issue,
        "maturity_date": maturity.isoformat(),
        "total_accepted": f"{total * 1e9:.0f}",
        "soma_accepted": f"{soma * 1e9:.0f}",
        "cash_management_bill_cmb": cmb,
        "floating_rate": frn,
        "inflation_index_security": tips,
    }


def _fixture_auctions() -> list[dict[str, str]]:
    return [
        # Before the window: proves the source covers the window start, never counted.
        _auction_row("2025-05-01", "Bill", "4-Week", 999.0, days=28),
        # Bills: public = total - SOMA. 4-week 90, 13-week 30 => 75% / 25%.
        _auction_row("2025-07-01", "Bill", "4-Week", 100.0, 10.0, days=28),
        _auction_row("2026-06-16", "Bill", "13-Week", 30.0, days=91),
        # A cash management bill is excluded however large.
        _auction_row("2026-01-05", "Bill", "27-Day", 500.0, cmb="Yes", days=27),
        # Coupons: a 30-year reopening, a TIPS reopening, an FRN reopening, a 2-year note.
        _auction_row("2025-11-15", "Bond", "29-Year 10-Month", 20.0, 5.0),
        _auction_row("2026-01-31", "Note", "9-Year 8-Month", 10.0, tips="Yes"),
        _auction_row("2026-02-27", "Note", "1-Year 10-Month", 5.0, frn="Yes"),
        _auction_row("2026-03-31", "Note", "2-Year", 30.0),
        # After the window: issued on or after the start, never counted.
        _auction_row("2026-06-25", "Bill", "26-Week", 999.0, days=182),
    ]


def _write_auctions(path: Path, rows: list[dict[str, str]], columns=None) -> Path:
    columns = list(columns or ["record_date", *AUCTION_REQUIRED_COLUMNS])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _portfolio_row(bond_id: int, *, security_type: str, years: float, maturity: str, face: float, holder="Private"):
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    issue = pd.Timestamp(maturity) - pd.Timedelta(days=round(years * 365.25))
    row.update(
        {
            "BondID": bond_id,
            "SecurityType": security_type,
            "IssueDate": issue,
            "MaturityDate": pd.Timestamp(maturity),
            "OriginalMaturityYears": years,
            "FaceValue": face,
            "CouponRate": 0.0 if years <= 1.0 else 0.04,
            "HolderType": holder,
            "HolderSubBucket": "",
            "Status": "Active",
            "MaturityCategory": "bills" if years <= 1.0 else "notes",
            "OriginalPrincipal": face,
            "AdjustedPrincipal": face,
            "ReferenceCPI_Issue": 0.0,
            "IndexRatio": 1.0,
            "FixedSpread": 0.0,
            "AccruedInterest_FRN": 0.0,
            "BenchmarkRate_FRN": 0.0,
            "LastAccrualDate": pd.NaT,
            "IssuePriceRatio": 0.99 if years <= 1.0 else 1.0,
            "IssueProceeds": face * (0.99 if years <= 1.0 else 1.0),
            "IssueYieldAtIssue": 0.04,
        }
    )
    return row


def _portfolio(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def _rule_profile(target: float) -> dict:
    return engine_rule_profile(
        target_path=[{"date": None, "bill_share_target": target}],
        bill_terms=FIXTURE_BILL_TERMS,
        coupon_terms=FIXTURE_COUPON_TERMS,
        settlement_calendar=FIXTURE_SETTLEMENT_CALENDAR,
    )


def _pricing() -> dict:
    return _issuance_pricing_context(
        tips_real_curve_years=[],
        tips_real_curve_rates=[],
        current_yield_curve_years=[0.08, 0.25, 0.5, 1.0, 2.0, 10.0, 30.0],
        current_yield_curve_rates=[0.04, 0.04, 0.04, 0.04, 0.04, 0.045, 0.05],
        yield_interpolation_method="linear",
        yield_floor_zero=True,
        evaluated_nominal_shock=None,
        tips_real_coupon=0.0,
        frn_spread=0.001,
        cbo_funding_mode=False,
        cbo_inputs={},
        cbo_scenario_id="test",
    )


# --- Auction calibration ----------------------------------------------------------------


def test_calibration_measures_public_auction_shares(tmp_path: Path) -> None:
    """Shares must reflect what the public bought: the Fed's add-ons and cash management
    bills are not modeled supply, and a reopening is the security it reopens."""

    path = _write_auctions(tmp_path / "auctions.csv", _fixture_auctions())
    calibration = build_auction_calibration(path, simulation_start_date=START)

    bills = {term["weeks"]: term for term in calibration["bill_terms"]}
    assert bills[4]["public_amount_bil"] == pytest.approx(90.0)
    assert bills[4]["share"] == pytest.approx(0.75)
    assert bills[13]["share"] == pytest.approx(0.25)
    assert sum(term["share"] for term in calibration["bill_terms"]) == pytest.approx(1.0)
    assert calibration["excluded"]["cash_management_bills"] == 1
    assert calibration["amounts_bil"]["soma_accepted"] == pytest.approx(15.0)

    coupons = {(term["kind"], term["tenor_years"]): term for term in calibration["coupon_terms"]}
    # Public coupon volume 15 + 10 + 5 + 30 = 60; TIPS and FRN are separate kinds.
    assert coupons[("nominal", 30)]["share"] == pytest.approx(15.0 / 60.0)
    assert coupons[("tips", 10)]["share"] == pytest.approx(10.0 / 60.0)
    assert coupons[("frn", 2)]["share"] == pytest.approx(5.0 / 60.0)
    assert coupons[("nominal", 2)]["share"] == pytest.approx(30.0 / 60.0)
    assert coupons[("nominal", 10)]["share"] == 0.0
    assert calibration["reopenings_mapped_to_original_tenor"] == 3
    assert calibration["public_bill_share_of_gross_volume"] == pytest.approx(120.0 / 180.0)
    assert calibration["window"] == {
        "issue_date_start_inclusive": "2025-06-21",
        "issue_date_end_exclusive": START,
        "latest_issue_date": "2026-06-16",
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.append(_auction_row("2026-03-01", "Bill", "10-Week", 5.0, days=70)), "unmapped bill term"),
        (lambda rows: rows.append(_auction_row("2026-03-01", "Bond", "15-Year", 5.0)), "unmapped nominal term"),
        (lambda rows: rows[1].update(total_accepted="null"), "total_accepted is not a number"),
        (lambda rows: rows[1].update(soma_accepted=""), "soma_accepted is not a number"),
        (lambda rows: rows[1].update(floating_rate="maybe"), "floating_rate must be Yes or No"),
        (lambda rows: rows[2].update(maturity_date="2026-09-16"), None),
    ],
)
def test_calibration_raises_on_malformed_or_unmapped_rows(tmp_path: Path, mutate, message) -> None:
    """A row the registry cannot place would silently shift every share; it must raise."""

    rows = _fixture_auctions()
    mutate(rows)
    path = _write_auctions(tmp_path / "auctions.csv", rows)
    if message is None:
        # A 13-week bill dated 92 days out on its only auction: the modal term is wrong.
        with pytest.raises(IssuanceCalibrationError, match="13-week bills mostly do not mature"):
            build_auction_calibration(path, simulation_start_date=START)
        return
    with pytest.raises(IssuanceCalibrationError, match=message):
        build_auction_calibration(path, simulation_start_date=START)


def test_calibration_raises_on_missing_columns_empty_or_uncovered_window(tmp_path: Path) -> None:
    """A source that cannot cover the twelve months must not calibrate on part of them."""

    rows = _fixture_auctions()
    missing = _write_auctions(
        tmp_path / "missing.csv", rows, columns=[c for c in AUCTION_REQUIRED_COLUMNS if c != "soma_accepted"]
    )
    with pytest.raises(IssuanceCalibrationError, match="missing columns"):
        build_auction_calibration(missing, simulation_start_date=START)

    empty = _write_auctions(tmp_path / "empty.csv", [rows[0]])
    with pytest.raises(IssuanceCalibrationError, match="no auctions issued"):
        build_auction_calibration(empty, simulation_start_date=START)

    stale = _write_auctions(tmp_path / "stale.csv", [row for row in rows if row["issue_date"] < "2026-06-01"])
    with pytest.raises(IssuanceCalibrationError, match="more than 7 days before"):
        build_auction_calibration(stale, simulation_start_date=START)

    late_start = _write_auctions(tmp_path / "late.csv", rows[1:])
    with pytest.raises(IssuanceCalibrationError, match="after the window start"):
        build_auction_calibration(late_start, simulation_start_date=START)


def test_package_calibration_records_opening_bill_share(tmp_path: Path) -> None:
    """The default target is the opening book's own bill share, measured on the same basis
    as the run's outstanding metrics, so a baseline starts on target rather than jumping."""

    path = _write_auctions(tmp_path / "auctions.csv", _fixture_auctions())
    portfolio = _portfolio(
        [
            _portfolio_row(1, security_type="Fixed", years=28 / 365.25, maturity="2026-07-09", face=200.0),
            _portfolio_row(2, security_type="Fixed", years=2.0, maturity="2027-09-30", face=700.0),
            # Intragovernmental and matured holdings are outside controlled debt.
            _portfolio_row(3, security_type="Fixed", years=0.5, maturity="2026-09-30", face=999.0, holder="TrustFunds"),
            _portfolio_row(4, security_type="TIPS", years=10.0, maturity="2030-01-15", face=80.0),
        ]
    )
    portfolio.loc[3, "AdjustedPrincipal"] = 100.0
    calibration = build_issuance_calibration(
        path,
        simulation_start_date=START,
        source_relative_path="sources/fiscaldata/auctions_query.csv",
        opening_portfolio=portfolio,
    )
    assert controlled_debt_and_bills(portfolio) == pytest.approx((1000.0, 200.0))
    assert calibration["bill_share_target"] == pytest.approx(0.2)
    assert calibration["source_sha256"]


# --- The rule ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bill_face, expected_bills",
    [
        (150.0, 70.0),  # below target: fill the gap 0.2 * 1100 - 150
        (0.0, 100.0),  # gap larger than the day's issuance: all bills
        (300.0, 0.0),  # above target: no bills, the share drifts down through coupons
    ],
)
def test_bill_amount_fills_toward_target(bill_face: float, expected_bills: float) -> None:
    """Bills are sized on the post-issuance stock, so rollovers keep the share on target."""

    assert bill_issuance_amount(100.0, target=0.2, controlled_debt=1000.0, bill_face=bill_face) == pytest.approx(
        expected_bills
    )


def test_schedule_splits_bills_by_term_and_coupons_by_shares() -> None:
    """The split must close to the day's amount and date each bill in whole weeks, or
    bills would roll on the wrong days and coupons would land in the wrong holder bucket."""

    profile = _rule_profile(0.2)
    schedule, target = _build_bill_share_supply_schedule(
        100.0,
        face_target_mode=True,
        issuance_leg="ordinary_issuance",
        issuance_profile=profile,
        controlled_debt=1000.0,
        controlled_bills=150.0,
        current_date=pd.Timestamp("2026-07-07"),
        pricing=_pricing(),
    )
    assert target == pytest.approx(0.2)
    bills = [item for item in schedule if item["type"] == "Fixed" and item["maturity"] <= 1.0]
    coupons = [item for item in schedule if item not in bills]
    assert sum(item["face_amount"] for item in bills) == pytest.approx(70.0)
    assert sum(item["face_amount"] for item in coupons) == pytest.approx(30.0)
    by_days = {(item["maturity_date"] - pd.Timestamp("2026-07-07")).days: item for item in bills}
    assert sorted(by_days) == [28, 91, 182]
    assert by_days[28]["face_amount"] == pytest.approx(35.0)
    assert by_days[91]["face_amount"] == pytest.approx(21.0)
    for item in bills:
        assert get_security_category_for_prefs("Fixed", item["maturity"], profile) == "bills"
    categories = {
        item["maturity"]: get_security_category_for_prefs(item["type"], item["maturity"], profile)
        for item in coupons
    }
    assert categories == {2.0: "notes", 10.0: "notes", 30.0: "bonds"}
    assert get_security_category_for_prefs("TIPS", 10.0, profile) == "tips"
    assert get_security_category_for_prefs("FRN", 2.0, profile) == "frn"


def test_proceeds_amount_overshoots_target_by_at_most_the_bill_discount() -> None:
    """Cash financing sizes proceeds, and the rule treats them as face. Bills issue at a
    discount, so bill face exceeds the rule's amount; the realized share stays within the
    discount of the target rather than drifting."""

    profile = _rule_profile(0.2)
    schedule, _ = _build_bill_share_supply_schedule(
        100.0,
        face_target_mode=False,
        issuance_leg="operating_cash_target_financing",
        issuance_profile=profile,
        controlled_debt=1000.0,
        controlled_bills=150.0,
        current_date=pd.Timestamp("2026-07-07"),
        pricing=_pricing(),
    )
    assert sum(item["proceeds"] for item in schedule) == pytest.approx(100.0)
    bill_face = sum(item["face_amount"] for item in schedule if item["type"] == "Fixed" and item["maturity"] <= 1.0)
    total_face = sum(item["face_amount"] for item in schedule)
    realized = (150.0 + bill_face) / (1000.0 + total_face)
    assert bill_face > 70.0
    # 26-week bills at 4% discount about 2%; a fifth of 70 bn of bills is in them.
    assert realized - 0.2 == pytest.approx(0.0, abs=0.001)


def test_target_path_is_linear_between_dates_and_flat_outside() -> None:
    """Scenarios phase a target in over time; the path must not extrapolate past its ends."""

    path = normalize_target_path(
        [{"date": "2027-01-01", "bill_share_target": 0.2}, {"date": "2027-01-11", "bill_share_target": 0.3}],
        label="path",
    )
    points = [(pd.Timestamp(p["date"]), p["bill_share_target"]) for p in path]
    assert target_for_date(points, "2026-06-01") == pytest.approx(0.2)
    assert target_for_date(points, "2027-01-06") == pytest.approx(0.25)
    assert target_for_date(points, "2028-01-01") == pytest.approx(0.3)
    with pytest.raises(IssuanceCalibrationError, match="strictly increasing"):
        normalize_target_path(
            [{"date": "2027-01-11", "bill_share_target": 0.2}, {"date": "2027-01-01", "bill_share_target": 0.3}],
            label="path",
        )
    with pytest.raises(IssuanceCalibrationError, match="between 0 and 1"):
        normalize_target_path(1.2, label="path")


# --- Engine -----------------------------------------------------------------------------

ENGINE_START = "2026-06-30"
ENGINE_END = "2026-09-30"


def _engine_params(tmp_path: Path, *, profile, mode: str, rate_shift: float = 0.0) -> dict:
    periods = build_simulation_calendar(ENGINE_START, ENGINE_END, "daily")
    paths = _build_temp_forecast_inputs(
        tmp_path,
        periods=periods,
        period_start=ENGINE_START,
        period_end=ENGINE_END,
        cbo_public_debt_target_bil=1_080.0,
        pre_issuance_controlled_debt_bil=1_000.0,
        base_cash_balance_bil=800.0,
        signed_primary_flow_bil=60.0,
    )
    params = _minimal_engine_params(paths)
    params["initial_bonds_df"] = _portfolio(
        [
            _portfolio_row(1, security_type="Fixed", years=28 / 365.25, maturity="2026-07-14", face=100.0),
            _portfolio_row(2, security_type="Fixed", years=56 / 365.25, maturity="2026-08-11", face=100.0),
            _portfolio_row(3, security_type="Fixed", years=91 / 365.25, maturity="2026-09-08", face=100.0),
            _portfolio_row(4, security_type="Fixed", years=2.0, maturity="2027-09-30", face=700.0),
        ]
    )
    params["treasury_issuance_profile"] = profile
    params["yield_curve"]["rates"] = [rate + rate_shift for rate in params["yield_curve"]["rates"]]
    params["funding_rule"].update(
        {"mode": mode, "cash_closure_target_bil": 0.0, "validation_floor_bil": -0.000001}
    )
    return params


def _bill_share_by_date(results) -> pd.Series:
    rows = results.attrs["handoff_tables"]["tdcsim_scenario_metrics"]
    return pd.Series({row["date"]: row["outstanding_controlled_bill_share"] for row in rows})


def test_engine_keeps_bill_share_on_target_through_rollovers(tmp_path: Path) -> None:
    """Every opening bill matures inside the window. Fixed gross shares would refinance
    them mostly into coupons; the rule rolls them into bills and holds the share.

    Each opening bill is a tenth of the book and is replaced partly by 13- and 26-week
    bills that settle on the next Thursday, so the share dips between a Tuesday maturity
    and that Thursday. Once the week's bills have settled it is back on target, within
    the bill discount and the coupons still pending until their monthly dates."""

    rule = _engine_params(tmp_path / "rule", profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    results, _ = run_simulation(rule, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    share = _bill_share_by_date(results)
    assert len(share) == len(results) - 1
    thursdays = [day for day in share.index if pd.Timestamp(day).weekday() == 3]
    assert len(thursdays) == 13
    assert (share[thursdays] - 0.3).abs().max() < 0.005
    assert share["2026-09-10"] == pytest.approx(0.3, abs=0.005)
    assert (share - 0.3).abs().mean() < 0.01

    fixed = _engine_params(tmp_path / "fixed", profile=_fixed_share_profile(), mode="cbo_operating_cash_target_financing_v1")
    fixed_results, _ = run_simulation(fixed, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    assert _bill_share_by_date(fixed_results).iloc[-1] < 0.15


def _fixed_share_profile() -> dict:
    return {
        "bills": {"category_cutoff_years": 1.0, "target_percentage_of_remainder": 0.225, "maturities": [0.5], "maturity_distribution": [1.0]},
        "notes": {"category_cutoff_years": 10.0, "target_percentage_of_remainder": 0.55, "maturities": [5.0], "maturity_distribution": [1.0]},
        "bonds": {"category_cutoff_years": 999.0, "target_percentage_of_remainder": 0.225, "maturities": [20.0], "maturity_distribution": [1.0]},
        "remainder_maturity_years": 1.0,
    }


def test_operating_cash_mode_holds_tga_and_puts_rate_costs_in_debt(tmp_path: Path) -> None:
    """Cash financing that waits for the TGA to hit zero makes scenario comparisons depend
    on timing. Holding the TGA on its path makes a costlier scenario show up as more debt,
    with the same cash, every day."""

    mode = "cbo_operating_cash_target_financing_v1"
    base, _ = run_simulation(
        _engine_params(tmp_path / "base", profile=_rule_profile(0.3), mode=mode),
        ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline",
    )
    high, _ = run_simulation(
        _engine_params(tmp_path / "high", profile=_rule_profile(0.3), mode=mode, rate_shift=0.01),
        ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline",
    )
    for results in (base, high):
        running = results.iloc[1:]
        decided = running["IssuanceNeedProceeds"] > 1e-9
        assert decided.any()
        # Cash decided but not yet settled closes the gap exactly: no day is financed twice.
        excess = running["TGA"] + running["PendingIssuanceProceeds"] - running["CBOOperatingCashTarget"]
        assert (excess >= -1e-6).all()
        assert excess[decided].abs().max() < 1e-6
        assert (running["TGA"] >= 0.0).all()
        assert running["CashFinancingProceeds"].sum() == pytest.approx(running["AuctionProceeds"].sum())
        assert running["CBOControlledDebtTargetError"].abs().max() == 0.0
        assert running["DebtDriftFromReference"].iloc[-1] == pytest.approx(
            running["ScenarioControlledDebt"].iloc[-1] - running["CBOControlledDebtReference"].iloc[-1]
        )
    def cash_with_pending(results):
        return results["TGA"].iloc[-1] + results["PendingIssuanceProceeds"].iloc[-1]

    assert cash_with_pending(high) == pytest.approx(cash_with_pending(base), abs=1e-6)
    assert high["ScenarioControlledDebt"].iloc[-1] > base["ScenarioControlledDebt"].iloc[-1] + 0.1


def test_operating_cash_mode_issues_nothing_while_tga_is_above_target(tmp_path: Path) -> None:
    """Surplus cash stays in the TGA until spending draws it back to the target; issuing
    into a surplus would build debt the scenario does not need."""

    params = _engine_params(tmp_path, profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    params["initial_values"]["tga"] = 900.0
    results, _ = run_simulation(params, ENGINE_START, "2026-07-10", freq="D", scenario_name="baseline")
    running = results.iloc[1:]
    assert (running["TGA"] > running["CBOOperatingCashTarget"] + 1.0).all()
    assert running["AuctionProceeds"].abs().max() == 0.0


def test_operating_cash_mode_fails_on_negative_tga(tmp_path: Path) -> None:
    """A target below zero cannot be met without an overdraft; the run must fail."""

    params = _engine_params(tmp_path, profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    params["initial_values"]["tga"] = 0.0
    operating_cash = Path(params["baseline_input_paths"]["operating_cash_path_file"])
    frame = pd.read_csv(operating_cash)
    frame["operating_cash_target_bil"] = -5.0
    frame.to_csv(operating_cash, index=False)
    with pytest.raises(RuntimeError, match="TGA-floor financing failed"):
        run_simulation(params, ENGINE_START, "2026-07-10", freq="D", scenario_name="baseline")


def test_runner_row_budget_covers_every_rule_supply_item() -> None:
    """The bounded runner caps portfolio rows up front; a cap sized for five fixed-share
    categories aborts a long rule run partway (it did, 360 days into a 15-month run)."""

    from tdcsim_cbo.runner import _portfolio_row_budget
    from tdc_shared import HOLDER_TYPES

    profile = _rule_profile(0.2)
    params = {"initial_bonds_df": _portfolio([]), "treasury_issuance_profile": profile}
    budget = _portfolio_row_budget(params, start="2026-06-21", end="2027-09-30")
    days = len(pd.date_range("2026-06-21", "2027-09-30", freq="D")) - 1
    items = len(profile["bill_terms"]) + len(profile["coupon_terms"])
    assert budget >= days * items * (len(HOLDER_TYPES) + 1)


# --- Settlement calendar ----------------------------------------------------------------


def _new_issues(portfolio) -> pd.DataFrame:
    issued = portfolio.loc[pd.to_datetime(portfolio["IssueDate"]) > pd.Timestamp(ENGINE_START)].copy()
    issued["IssueDate"] = pd.to_datetime(issued["IssueDate"]).dt.normalize()
    return issued


def _supply_key(row) -> tuple[str, float]:
    years = float(row["OriginalMaturityYears"])
    if row["SecurityType"] == "Fixed" and years <= 1.0:
        return ("bill", round(years * 365.25 / 7.0))
    return (str(row["SecurityType"]), years)


def test_securities_settle_only_on_their_calendar_days(tmp_path: Path) -> None:
    """Treasury settles each security on its own calendar; issuing every security every day
    multiplies portfolio rows and engine time without adding realism."""

    from sim_engine import _settlement_schedule

    params = _engine_params(tmp_path, profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    results, portfolio = run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    schedule = _settlement_schedule(params["treasury_issuance_profile"], ENGINE_START, ENGINE_END)
    allowed = {(day, key) for day, keys in schedule.items() for key in keys}
    issued = _new_issues(portfolio)
    assert not issued.empty
    assert results["EarlySettlementProceeds"].abs().max() == 0.0
    for _, row in issued.iterrows():
        assert (row["IssueDate"], _supply_key(row)) in allowed
    # 4-week bills on Tuesdays, 13- and 26-week on Thursdays, 10/30-year on the 15th
    # (2026-08-15 is a Saturday, so the 17th), 2-year on the month's last business day.
    days_by_key = issued.groupby(issued.apply(_supply_key, axis=1))["IssueDate"].apply(set)
    assert {day.weekday() for day in days_by_key[("bill", 4)]} == {1}
    assert {day.weekday() for day in days_by_key[("bill", 13)]} == {3}
    assert pd.Timestamp("2026-08-17") in days_by_key[("Fixed", 10.0)]
    assert days_by_key[("Fixed", 2.0)] == {pd.Timestamp("2026-07-31"), pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-30")}


def test_row_growth_is_bounded_by_the_calendar(tmp_path: Path) -> None:
    """Rows per year must scale with settlement events, not with days times securities."""

    from sim_engine import _settlement_schedule
    from tdc_shared import HOLDER_TYPES

    params = _engine_params(tmp_path, profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    _, portfolio = run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    schedule = _settlement_schedule(params["treasury_issuance_profile"], ENGINE_START, ENGINE_END)
    events = sum(len(keys) for day, keys in schedule.items() if day > pd.Timestamp(ENGINE_START))
    issued = _new_issues(portfolio)
    assert issued.groupby(["IssueDate", "OriginalMaturityYears"]).ngroups <= events
    assert len(issued) <= events * (len(HOLDER_TYPES) + 1)
    days = len(pd.date_range(ENGINE_START, ENGINE_END)) - 1
    securities = len(FIXTURE_BILL_TERMS) + len(FIXTURE_COUPON_TERMS)
    # Daily issuance would create one item per security per day; the calendar cuts that
    # several-fold (about thirteen-fold for the full eighteen-security mix).
    assert events * 3 < days * securities


def test_pending_amounts_count_as_issued_for_cash_and_bill_share() -> None:
    """A security decided on Monday and settling Thursday must not be decided again on
    Tuesday: pending face counts toward debt and bills, and settlement releases exactly
    what was pending."""

    from sim_engine import _decide_and_settle_pending

    profile = _rule_profile(0.2)
    pending = {("bill", 4.0): ["Fixed", 28 / 365.25, 28, 50.0]}
    settled, early = _decide_and_settle_pending(
        pending,
        need=100.0,
        issuance_profile=profile,
        pre_issuance_position=(1000.0, 150.0),
        due_keys=[],
        projected_tga_without_settlement=500.0,
        current_date=pd.Timestamp("2026-07-06"),
    )
    assert settled == [] and early == 0.0
    # 0.2 x (1000 + 50 pending + 100) - (150 + 50 pending) = 30 of bills, not 70.
    bills = sum(item[3] for key, item in pending.items() if key[0] == "bill")
    assert bills == pytest.approx(50.0 + 30.0)
    assert sum(item[3] for item in pending.values()) == pytest.approx(150.0)

    settled, early = _decide_and_settle_pending(
        pending,
        need=0.0,
        issuance_profile=profile,
        pre_issuance_position=(1000.0, 150.0),
        due_keys=[("bill", 4.0)],
        projected_tga_without_settlement=500.0,
        current_date=pd.Timestamp("2026-07-07"),
    )
    assert early == 0.0
    assert [item[0] for item in settled] == [("bill", 4.0)]
    assert settled[0][-1] == pytest.approx(50.0 + 30.0 * 0.5)


def test_pending_bills_settle_early_rather_than_overdraw_the_tga() -> None:
    """Between settlements the TGA runs below target by what is pending; it must never be
    pushed negative while bills that could settle are waiting."""

    from sim_engine import _decide_and_settle_pending

    pending = {
        ("bill", 13.0): ["Fixed", 91 / 365.25, 91, 40.0],
        ("Fixed", 10.0): ["Fixed", 10.0, None, 25.0],
    }
    settled, early = _decide_and_settle_pending(
        pending,
        need=0.0,
        issuance_profile=_rule_profile(0.2),
        pre_issuance_position=(1000.0, 150.0),
        due_keys=[],
        projected_tga_without_settlement=-10.0,
        current_date=pd.Timestamp("2026-07-06"),
    )
    assert early == pytest.approx(40.0)
    assert [item[0] for item in settled] == [("bill", 13.0)]
    assert list(pending) == [("Fixed", 10.0)]

    # Only when pending bills cannot cover the shortfall do coupons settle early too.
    pending = {
        ("bill", 13.0): ["Fixed", 91 / 365.25, 91, 4.0],
        ("Fixed", 10.0): ["Fixed", 10.0, None, 25.0],
    }
    settled, early = _decide_and_settle_pending(
        pending,
        need=0.0,
        issuance_profile=_rule_profile(0.2),
        pre_issuance_position=(1000.0, 150.0),
        due_keys=[],
        projected_tga_without_settlement=-10.0,
        current_date=pd.Timestamp("2026-07-06"),
    )
    assert early == pytest.approx(29.0)
    assert pending == {}


def test_tga_returns_to_target_when_everything_pending_settles(tmp_path: Path) -> None:
    """The TGA dips below target only by cash that is decided and waiting; once a quiet
    spell lets every balance settle, it is back on target, and it never goes negative."""

    params = _engine_params(tmp_path, profile=_rule_profile(0.3), mode="cbo_operating_cash_target_financing_v1")
    results, _ = run_simulation(params, ENGINE_START, ENGINE_END, freq="D", scenario_name="baseline")
    running = results.iloc[1:]
    assert (running["TGA"] >= 0.0).all()
    gap = running["CBOOperatingCashTarget"] - running["TGA"]
    assert gap.max() > 1.0
    assert (gap - running["PendingIssuanceProceeds"]).abs().max() < 1e-6
    # With an all-bill target and the deficit stopped after July, the only later need is
    # Tuesday 2026-08-04's rollover, whose bills all settle by Thursday 08-06; by Friday
    # nothing is pending and the TGA is back on target.
    quiet = _engine_params(tmp_path / "quiet", profile=_rule_profile(1.0), mode="cbo_operating_cash_target_financing_v1")
    primary = Path(quiet["baseline_input_paths"]["primary_deficit_path_file"])
    frame = pd.read_csv(primary)
    frame.loc[pd.to_datetime(frame["period_end"]) > pd.Timestamp("2026-07-31"), "primary_deficit_bil"] = 0.0
    frame.to_csv(primary, index=False)
    quiet_results, _ = run_simulation(quiet, ENGINE_START, "2026-08-07", freq="D", scenario_name="baseline")
    assert quiet_results.loc["2026-08-04", "IssuanceNeedProceeds"] > 1.0
    last = quiet_results.iloc[-1]
    assert last["PendingIssuanceProceeds"] == pytest.approx(0.0, abs=1e-9)
    assert last["TGA"] == pytest.approx(last["CBOOperatingCashTarget"], abs=1e-6)
