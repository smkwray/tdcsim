"""Federal Reserve net income, deferred asset and remittances in CBO runs.

The remittance must respond to what the run does: the Fed's Treasury income, the reserve
balances it pays interest on, and the deferred asset it must repay first. These tests run
the production engine on small fixtures and check each channel moves remittances, and
hence Treasury financing, the way the Fed's income statement says it should.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fed_income import non_treasury_reserve_effect, remittance_week, settle_remittances
from forecast_bundle_builders import build_fed_income_path_rows, fed_income_source_fixture
from forecast_paths import verify_fed_remittances
from sim_engine import run_simulation
from simulation_calendar import build_simulation_calendar
from tdc_shared import BOND_PORTFOLIO_COLS, PORTFOLIO_DTYPES
from test_cbo_engine_integration import _build_temp_forecast_inputs, _minimal_engine_params


START, END = "2026-09-20", "2026-10-10"
# Opening reserves land at 2,200 after the H.4.1 roll: the run's TGA opens at 800.
SMALL_FED = {"h41_reserve_balances_bil": 3_000.0, "h41_tga_bil": 0.0, "iorb_spread_pct": 0.1}


def _fed_note(face: float, coupon: float) -> dict:
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    row.update({
        "BondID": 2, "SecurityType": "Fixed", "IssueDate": pd.Timestamp("2025-09-30"),
        "MaturityDate": pd.Timestamp("2027-09-30"), "OriginalMaturityYears": 2.0, "FaceValue": face,
        "CouponRate": coupon, "HolderType": "CB", "HolderSubBucket": "", "Status": "Active",
        "MaturityCategory": "notes", "OriginalPrincipal": face, "AdjustedPrincipal": face,
        "ReferenceCPI_Issue": 0.0, "IndexRatio": 1.0, "FixedSpread": 0.0, "AccruedInterest_FRN": 0.0,
        "BenchmarkRate_FRN": 0.0, "LastAccrualDate": pd.NaT, "IssuePriceRatio": 1.0,
        "IssueProceeds": face, "IssueYieldAtIssue": coupon,
    })
    return row


def _run(tmp_path, *, fed_coupon: float = 0.04, remittance_bil: float = 0.0, **fed_income):
    periods = build_simulation_calendar(START, END, "daily")
    paths = _build_temp_forecast_inputs(
        tmp_path, periods=periods, period_start=START, period_end=END,
        cbo_public_debt_target_bil=1_700.0, pre_issuance_controlled_debt_bil=1_500.0,
        remittance_bil=remittance_bil, fed_income={**SMALL_FED, **fed_income},
        later_fiscal_years={2027: (1_750.0, 0.0)},
    )
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=1_000.0)
    portfolio = params["initial_bonds_df"]
    params["initial_bonds_df"] = pd.concat(
        [portfolio, pd.DataFrame([_fed_note(500.0, fed_coupon)], columns=BOND_PORTFOLIO_COLS)],
        ignore_index=True,
    ).astype(PORTFOLIO_DTYPES, errors="ignore")
    results, _ = run_simulation(params, START, END, freq="D", scenario_name="baseline")
    return results, paths


def _verify(results, paths) -> float:
    return verify_fed_remittances(
        results.reset_index(), paths["fed_remittance_path_file"], paths["fed_income_path_file"],
    )


def test_deferred_asset_blocks_remittances_until_net_income_has_repaid_it():
    due = -5.0
    due, paid = settle_remittances(due, -1.0, pay_today=True)
    assert (due, paid) == (-6.0, 0.0)  # a loss adds to the deferred asset
    due, paid = settle_remittances(due, 4.0, pay_today=True)
    assert (due, paid) == (-2.0, 0.0)  # income repays it; nothing is remitted meanwhile
    due, paid = settle_remittances(due, 3.0, pay_today=False)
    assert (due, paid) == (1.0, 0.0)  # cleared, but the Banks remit only on a pay day
    due, paid = settle_remittances(due, 2.0, pay_today=True)
    assert (due, paid) == (0.0, 3.0)  # then all net income since the deferred asset cleared
    due, paid = settle_remittances(due, -0.5, pay_today=True)
    assert (due, paid) == (-0.5, 0.0)  # a new loss starts a new deferred asset


def test_remittances_are_weekly_and_a_closed_wednesday_pays_on_thursday():
    assert remittance_week("2026-09-30") == pd.Timestamp("2026-09-30")
    assert remittance_week("2026-10-06") == pd.Timestamp("2026-09-30")
    # Veterans Day 2026 closes Fedwire on a Wednesday; Thursday is still in that week.
    assert remittance_week("2026-11-12") == pd.Timestamp("2026-11-11")


def test_higher_fed_coupon_income_raises_remittances_once_the_deferred_asset_is_clear(tmp_path):
    low, paths = _run(tmp_path / "low", fed_coupon=0.04, h41_earnings_remittances_due_bil=0.0)
    high, _ = _run(tmp_path / "high", fed_coupon=0.06, h41_earnings_remittances_due_bil=0.0)
    # The note pays 10 (4%) or 15 (6%) on Wednesday 2026-09-30, a remittance day.
    assert high["CB_CouponIncome"].sum() - low["CB_CouponIncome"].sum() == pytest.approx(5.0)
    assert high["CB_Remittance"].sum() - low["CB_Remittance"].sum() == pytest.approx(5.0, abs=1e-6)
    assert low.loc["2026-09-30", "CB_Remittance"] > 0.0
    # With no deferred asset the extra coupon comes straight back: the Treasury's cash is unchanged.
    assert high["TGA"].to_numpy() == pytest.approx(low["TGA"].to_numpy())
    _verify(low, paths)


def test_higher_fed_coupon_income_repays_more_deferred_asset_and_remits_nothing(tmp_path):
    low, _ = _run(tmp_path / "low", fed_coupon=0.04, h41_earnings_remittances_due_bil=-100.0)
    high, _ = _run(tmp_path / "high", fed_coupon=0.06, h41_earnings_remittances_due_bil=-100.0)
    assert low["CB_Remittance"].sum() == high["CB_Remittance"].sum() == 0.0
    assert low["CB_DeferredAsset"].iloc[-1] - high["CB_DeferredAsset"].iloc[-1] == pytest.approx(5.0, abs=1e-6)
    # While the deferred asset is positive the Treasury bears the extra coupon in full.
    assert low["TGA"].iloc[-1] - high["TGA"].iloc[-1] == pytest.approx(5.0, abs=1e-6)


def test_higher_reserves_raise_iorb_expense_and_lower_remittances(tmp_path):
    lean, _ = _run(tmp_path / "lean", h41_earnings_remittances_due_bil=0.0)
    flush, paths = _run(tmp_path / "flush", h41_earnings_remittances_due_bil=0.0, h41_reserve_balances_bil=5_000.0)
    gap = flush["CB_ReserveBalances"] - lean["CB_ReserveBalances"]
    assert gap.iloc[0] == pytest.approx(2_000.0)
    # Each day's IORB is the rate times the reserves held overnight, actual/360.
    expected = (flush["CB_IORBRatePct"] / 100.0 * gap.shift(1) / 360.0).iloc[1:]
    assert (flush["CB_IORBExpense"] - lean["CB_IORBExpense"]).iloc[1:].to_numpy() == pytest.approx(expected.to_numpy())
    # About ten days of IORB on 2,000 more reserves (the extra IORB itself adds reserves),
    # all paid before the 2026-09-30 remittance.
    shortfall = 2_000.0 * flush["CB_IORBRatePct"].iloc[1] / 100.0 * 10 / 360.0
    assert lean["CB_Remittance"].sum() - flush["CB_Remittance"].sum() == pytest.approx(shortfall, rel=1e-3)
    assert flush["TGA"].iloc[-1] < lean["TGA"].iloc[-1]
    _verify(flush, paths)


def test_reserve_level_is_the_h41_balance_sheet_moved_by_the_runs_own_factors(tmp_path):
    results, paths = _run(
        tmp_path, h41_earnings_remittances_due_bil=0.0, h41_tga_bil=956.502, h41_reserve_balances_bil=2_936.355,
        currency_growth_bil=0.3, soma_mbs_bil=1_964.8, soma_mbs_interest_bil=0.13, operating_expenses_bil=0.03,
    )
    # Rolling the Wednesday H.4.1 to the opening: the TGA fell from 956.502 to 800.
    assert results["CB_ReserveBalances"].iloc[0] == pytest.approx(2_936.355 + 156.502)
    step = results["CB_ReserveBalances"].diff().iloc[1:]
    explained = (
        results["CB_TreasuryReserveEffect"] - results["CB_Currency"].diff()
        + results["CB_MBSHoldings"].diff() - results["CB_MBSIncome"] - results["CB_OtherIncome"]
        + results["CB_IORBExpense"] + results["CB_ONRRPExpense"] + results["CB_OperatingExpenses"]
        + results["CB_Dividends"]
    ).iloc[1:]
    assert step.to_numpy() == pytest.approx(explained.to_numpy())
    # The Treasury part is the run's own reserve path; Treasury interest and remittances are not in it.
    assert results["CB_TreasuryReserveEffect"].iloc[1:].to_numpy() == pytest.approx(
        results["Reserves"].diff().iloc[1:].to_numpy()
    )
    assert results["CB_Currency"].iloc[-1] - results["CB_Currency"].iloc[0] == pytest.approx(0.3 * 20)
    _verify(results, paths)


def test_table_4_3_remittance_stays_in_the_fiscal_flow_while_the_tga_gets_the_modeled_one(tmp_path):
    neutral, _ = _run(tmp_path / "zero", h41_earnings_remittances_due_bil=-1_000.0)
    cbo, paths = _run(tmp_path / "cbo", remittance_bil=2.338, h41_earnings_remittances_due_bil=-1_000.0)
    # The fixture books 2.338 in each of FY2026 and FY2027 inside the window.
    table_4_3 = cbo["CBOTable43RemittanceInFiscalFlow"].sum()
    assert table_4_3 == pytest.approx(2 * 2.338)
    assert cbo["CB_Remittance"].sum() == 0.0  # the deferred asset blocks the modeled remittance
    # The public receives P + R_CBO, 99 percent of it in covered deposits ...
    assert cbo["TDC_Level"].iloc[-1] - neutral["TDC_Level"].iloc[-1] == pytest.approx(0.99 * table_4_3)
    # ... and the Treasury, short of the remittance CBO counted on, finances the gap.
    assert neutral["TGA"].iloc[-1] - cbo["TGA"].iloc[-1] == pytest.approx(table_4_3, abs=1e-6)
    _verify(cbo, paths)


def test_verifier_rejects_a_tampered_remittance_path(tmp_path):
    results, paths = _run(tmp_path, h41_earnings_remittances_due_bil=0.0)
    assert _verify(results, paths) > 0.0
    paid = results.index[results["CB_Remittance"] > 0.0][0]
    for column in ("CBORemittanceCashEffect", "CB_Remittance"):
        tampered = results.copy()
        tampered.loc[paid, column] += 1.0
        with pytest.raises(ValueError, match="differs from its rebuild"):
            _verify(tampered, paths)
    tampered = results.copy()
    tampered.loc[paid, "CB_IORBExpense"] -= 1.0
    with pytest.raises(ValueError, match="CB_IORBExpense"):
        _verify(tampered, paths)
    tampered = results.copy()
    tampered["CBOTable43RemittanceInFiscalFlow"] += 0.01
    with pytest.raises(ValueError, match="Table 4-3"):
        _verify(tampered, paths)


def test_mbs_income_follows_the_nyfed_runoff_path_from_the_h41_opening():
    source = fed_income_source_fixture()
    periods = build_simulation_calendar("2026-06-21", "2036-09-30", "daily")
    curve = [{"scenario_id": "s", "curve_date": "2026-04-01", "tenor_years": 0.25, "nominal_rate": 3.35}]
    rows = build_fed_income_path_rows(scenario_id="s", periods=periods, yield_rows=curve)
    first, last = rows[0], rows[-1]
    wac = float(source["soma_agency_mbs"]["weighted_average_coupon_pct"]) / 100.0
    assert first["soma_mbs_interest_bil"] == pytest.approx(wac * first["soma_mbs_start_bil"] / 365.25)
    # The path is the H.4.1 level scaled by the NY Fed projection, so it opens near 1,964.8 ...
    assert first["soma_mbs_start_bil"] == pytest.approx(1_964.798, rel=2e-3)
    # ... runs off to the projection's 2035 share of its mid-2026 level ...
    dec_2035 = next(row for row in rows if row["period_end"] == "2035-12-31")
    assert dec_2035["soma_mbs_end_bil"] / 1_964.798 == pytest.approx(677.0 / 1_950.6, rel=5e-3)
    # ... and keeps running off after the projection ends, so income keeps falling.
    assert last["soma_mbs_end_bil"] < dec_2035["soma_mbs_end_bil"]
    assert last["soma_mbs_interest_bil"] < first["soma_mbs_interest_bil"] / 3.0
    # IORB opens at the observed 3.65 percent on the package curve's 3.35 percent bill rate.
    assert first["iorb_spread_pct"] == pytest.approx(0.30)
    assert first["on_rrp_spread_pct"] == pytest.approx(0.15)


def test_non_treasury_reserve_effect_signs():
    base = dict(mbs_start=10.0, mbs_end=10.0, currency_start=5.0, currency_end=5.0, mbs_income=0.0,
                other_income=0.0, iorb_expense=0.0, on_rrp_expense=0.0, operating_expenses=0.0, dividends=0.0)
    assert non_treasury_reserve_effect(**{**base, "currency_end": 6.0}) == -1.0
    assert non_treasury_reserve_effect(**{**base, "mbs_end": 9.0}) == -1.0
    assert non_treasury_reserve_effect(**{**base, "iorb_expense": 1.0}) == 1.0
    assert non_treasury_reserve_effect(**{**base, "mbs_income": 1.0}) == -1.0


def test_source_transcription_names_every_packaged_document():
    source = fed_income_source_fixture()
    assert {doc["file"] for doc in source["documents"].values()} == {
        "fed/h41_2026-06-18.htm",
        "fed/fomc_implementation_note_2026-06-17.htm",
        "fed/nyfed_omo2025_projection_data.xlsx",
        "fed/nyfed_omo2025_report.pdf",
        "fed/nyfed_soma_agency_holdings_2026-06-17.json",
        "fed/reserve_banks_combined_financial_statements_2025.pdf",
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
