"""Federal Reserve earnings, balance sheet and weekly remittances in CBO runs.

The remittance must follow the Fed's own books: income accrued on what the Fed paid for its
securities, interest on the balances it actually owes, the pooled deferred asset, and the
Reserve Banks' weekly payment rule. These tests run the production engine on small fixtures
and check each channel, and that the verifier rejects books that do not close.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fed_income import FedLotBook, first_remittance_day_of_year, remittance_basis_date, remittance_week, weekly_remittance
from forecast_bundle_builders import build_fed_income_path_rows, fed_income_source_fixture, fed_net_premiums_bil
from forecast_paths import check_earnings_due_closes, check_fed_balance_sheet_closes, verify_fed_remittances
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
    results, _ = run_simulation(params, START, END, freq="D", scenario_name="baseline")
    return results, paths


def _frame(results):
    return results.reset_index().rename(columns={"DebtHeld_CB": "DebtHeld_CentralBank"})


def _verify(results, paths) -> dict:
    return verify_fed_remittances(_frame(results), paths["fed_income_path_file"].parent)


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


def test_a_bill_bought_at_99_earns_1_whatever_it_was_issued_at():
    """Sol's case: issued at 97, bought by the Fed at 99, redeemed at 100."""

    book = FedLotBook()
    book.add(7, "2026-10-20", premium=99.0 - 100.0, bill=True, acquired="2026-09-30")
    earned = 0.0
    for day in pd.date_range("2026-10-01", "2026-10-20"):
        bill, coupon = book.amortize(day - pd.Timedelta(days=1), day)
        earned += bill + coupon
    assert earned == pytest.approx(1.0)
    assert 7 not in book.lots


def test_a_premium_purchase_amortizes_against_income_to_maturity():
    book = FedLotBook()
    book.add(8, "2026-10-11", premium=2.0, bill=False, acquired="2026-10-01")
    daily = [sum(book.amortize(day - pd.Timedelta(days=1), day)) for day in pd.date_range("2026-10-02", "2026-10-11")]
    assert sum(daily) == pytest.approx(-2.0)
    assert daily[0] == pytest.approx(-0.2)  # straight line over the ten days held


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


def _tampered_runner_run(tmp_path, tamper):
    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo._json import read_json, write_json
    from tdcsim_cbo.verifier import verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _refresh_result_hashes, _runner_baseline_and_scenarios

    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path, remittance_bil=1.0)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    assert verify_scenario_run(run.output_dir)["status"] == "pass"
    results = pd.read_csv(run.results_path)
    tamper(results)
    results.to_csv(run.results_path, index=False)
    manifest = read_json(run.manifest_path)
    _refresh_result_hashes(run.output_dir, manifest)
    write_json(run.manifest_path, manifest)
    return run.output_dir, verify_scenario_run


def test_verifier_binds_fed_treasury_income_to_settlement_evidence(tmp_path):
    """Sol's first probe: move income between Treasury lines on a payment day."""

    def move(results):
        day = results.index[results["CB_CouponCash"].abs() > 0.0]
        row = day[0] if len(day) else 2
        results.loc[row, "CB_CouponAccrual"] -= 1.0
        results.loc[row, "CB_BillDiscountAccretion"] += 1.0

    output_dir, verify = _tampered_runner_run(tmp_path, move)
    with pytest.raises(ValueError, match="Fed remittance verification failed"):
        verify(output_dir)


def test_verifier_binds_the_tga_to_the_treasury_cash_journal(tmp_path):
    """Sol's second probe: cut one day's closing TGA by 1 billion."""

    def cut(results):
        results.loc[3, "TGA"] -= 1.0

    output_dir, verify = _tampered_runner_run(tmp_path, cut)
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
