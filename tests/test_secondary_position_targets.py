"""Holding buyers' Treasury positions on a dated target through secondary trades in bills.

A buyer-shift experiment compares scenarios that give each buyer exactly the same added
Treasury position. That comparison is only meaningful if every buyer's realized holdings,
and every other holder's, meet the declared target at each close, whatever Treasury
redeems and the Federal Reserve buys that day; if the trades that hold the target are
priced, settled and booked like any other security transfer; and if Treasury's own side
is unchanged, so that every difference from the matched baseline comes from who holds
the securities and how they pay.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pandas as pd
import pytest

import holder_targets as ht
from forecast_bundle_builders import build_fed_holdings_path_rows
from settlement_routes import TRC_ROUTE_STATES, TRC_ROUTING_SCHEMA_VERSION, TrcRoutingError
from sim_engine import run_simulation
from simulation_calendar import build_simulation_calendar
from tdc_shared import (
    BOND_PORTFOLIO_COLS, MMF_ADDED_TDC_COLUMNS, PORTFOLIO_DTYPES, PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF, PRIVATE_SUBBUCKET_MMF_ADDED, PRIVATE_SUBBUCKET_SHARES_KEY,
)
from tdcsim_cbo.bounded_output import BoundedResourceLimits, BoundedScenarioEvidenceSink
from test_cbo_engine_integration import _build_temp_forecast_inputs, _minimal_engine_params, _write_csv
from test_treasury_calendar import _profile

START, END = "2026-09-20", "2026-10-12"
ORIGIN, BUILD_END = "2026-09-24", "2026-09-28"
DNB, MMF = PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF
BANKS, FOREIGN = ("Banks", ""), ("Foreign", "")
PRIVATE_DNB, PRIVATE_MMF = ("Private", DNB), ("Private", MMF)
PRIVATE_MMF_ADDED = ("Private", PRIVATE_SUBBUCKET_MMF_ADDED)
TOLERANCE = 1.0e-6


def _security(bond_id, holder, route, kind, maturity, face, *, coupon=0.0, price=0.99, issue="2026-03-01"):
    row = {column: pd.NA for column in BOND_PORTFOLIO_COLS}
    bill = kind == "bills"
    row.update({
        "BondID": bond_id, "SecurityType": "Fixed", "IssueDate": pd.Timestamp(issue),
        "MaturityDate": pd.Timestamp(maturity), "OriginalMaturityYears": 0.5 if bill else 5.0,
        "FaceValue": face, "CouponRate": coupon, "HolderType": holder, "HolderSubBucket": route,
        "TDCPrincipalHolderType": holder, "TDCPrincipalHolderSubBucket": route, "Status": "Active",
        "MaturityCategory": kind, "OriginalPrincipal": face, "AdjustedPrincipal": face,
        "ReferenceCPI_Issue": 0.0, "IndexRatio": 1.0, "FixedSpread": 0.0, "AccruedInterest_FRN": 0.0,
        "BenchmarkRate_FRN": 0.0, "LastAccrualDate": pd.NaT, "IssuePriceRatio": price,
        "IssueProceeds": face * price, "IssueYieldAtIssue": 0.04,
        "InterestPaymentFrequency": pd.NA if bill else 2,
    })
    return row


# Bills spread over every remaining-maturity bucket, several maturing during the run, plus
# notes, so a buyer's purchased bills redeem while the target must still be held.
OPENING = [
    _security(1, "Banks", "", "bills", "2026-10-02", 60.0),
    _security(2, "Banks", "", "bills", "2026-12-15", 40.0),
    _security(3, "Banks", "", "notes", "2029-09-15", 100.0, coupon=0.04, price=1.0, issue="2024-09-15"),
    _security(4, "Foreign", "", "bills", "2026-10-05", 80.0),
    _security(5, "Foreign", "", "bills", "2027-03-01", 70.0),
    _security(6, "Foreign", "", "notes", "2030-03-15", 200.0, coupon=0.04, price=1.0, issue="2025-03-15"),
    _security(7, "Private", DNB, "bills", "2026-09-29", 50.0),
    _security(8, "Private", DNB, "bills", "2026-11-20", 90.0),
    _security(9, "Private", DNB, "bills", "2027-06-15", 60.0),
    _security(10, "Private", DNB, "notes", "2028-09-15", 150.0, coupon=0.04, price=1.0, issue="2024-09-15"),
    _security(11, "Private", MMF, "bills", "2026-10-01", 100.0),
    _security(12, "Private", MMF, "bills", "2026-12-31", 80.0),
]
OPENING_DEBT = sum(row["FaceValue"] for row in OPENING)


def _routing(on_rrp_opening: float = 5_000.0, on_rrp_share: float = 0.0) -> dict:
    return {
        "schema_version": TRC_ROUTING_SCHEMA_VERSION,
        "foreign_settlement_mix": {"foreign_bank_settled": 0.5, "foreign_official_deposit": 0.5, "foreign_official_rrp": 0.0},
        "mmf_cash_fund_route": {"on_rrp_share": on_rrp_share, "source": "test fixture"},
        "opening_fed_nonreserve_balances_bil": {
            "ONRRP_MMF": on_rrp_opening, "ForeignOfficialFedDeposits": 5_000.0, "ForeignOfficialFedRRP": 5_000.0,
        },
        "opening_balance_sources": {state: "test fixture" for state in TRC_ROUTE_STATES},
    }


def _params(
    tmp_path: Path,
    *,
    pass_through: float = 1.0,
    routing: dict | None = None,
    fed: bool = True,
    book: list[dict] = OPENING,
    fed_targets: dict[int, float] | None = None,
    debt_step: tuple[str, float] | None = None,
    fed_income: dict[str, float] | None = None,
    start: str = START,
    end: str = END,
) -> dict:
    """Engine params for a daily run; ``debt_step`` lowers the debt path by an amount from a date on."""

    periods = build_simulation_calendar(start, end, "daily")
    debt = sum(row["FaceValue"] for row in book)
    paths = _build_temp_forecast_inputs(
        tmp_path,
        periods=periods,
        period_start=start,
        pre_issuance_controlled_debt_bil=debt,
        cbo_public_debt_target_bil=debt + 125.0,
        later_fiscal_years={2027: (debt + 125.0, 0.0)},
        fed_income=fed_income,
    )
    if debt_step is not None:
        path = paths["debt_stock_path_file"]
        frame = pd.read_csv(path, keep_default_na=False)
        later = frame["period_end"] >= debt_step[0]
        for column in ("cbo_federal_debt_held_public_target_bil", "marketable_treasury_public_target_bil"):
            frame.loc[later, column] = frame.loc[later, column].astype(float) - debt_step[1]
        frame.to_csv(path, index=False)
    if fed:
        paths["fed_holdings_path_file"] = _write_csv(
            tmp_path / "tdcsim_fed_holdings_path.csv",
            build_fed_holdings_path_rows(
                scenario_id="baseline", periods=periods, opening_state_date=start, opening_cb_holdings_bil=0.0,
                cbo_fy_end_fed_holdings_bil=fed_targets or {2026: 20.0, 2027: 300.0}, observation_date=start,
                available_date=start,
            ),
        )
    params = _minimal_engine_params(paths, opening_controlled_debt_bil=debt)
    params["initial_bonds_df"] = pd.DataFrame(book, columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")
    params["private_mmf_split"] = {"mmf_deposit_pass_through": pass_through}
    params["trc_routing"] = copy.deepcopy(routing or _routing())
    return params


def _run(params: dict, out: Path, *, start: str = START, end: str = END):
    sink = BoundedScenarioEvidenceSink(
        out,
        limits=BoundedResourceLimits(
            minimum_available_bytes=0, application_abort_rss_bytes=0, parent_graceful_stop_rss_bytes=0,
            parent_kill_rss_bytes=0, acceptance_peak_rss_bytes=0, portfolio_row_budget=10_000,
            key_cardinality_budget=8_192,
        ),
        rss_reader=lambda: 1,
        available_reader=lambda: 1,
    )
    results, portfolio = run_simulation(
        params, start, end, freq="D", scenario_name="baseline", handoff_sink=sink, require_bounded_handoff=True
    )
    closure = pd.read_csv(out / "tdcsim_period_stock_closure.csv.gz", keep_default_na=False)
    return results, portfolio, closure


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp("baseline")
    params = _params(root / "inputs")
    results, portfolio, closure = _run(params, root / "run")
    return {**params, "_final_portfolio": portfolio}, results, closure


def _targets(closure: pd.DataFrame, *, buyer, donors, size: float, origin_day: str = ORIGIN,
             build_end: str = BUILD_END, end: str = END) -> tuple[list[dict], dict]:
    origin = ht.closing_holder_stocks(closure, bills_only=True)
    origin = origin[origin["period_end"] == origin_day]
    origin_bills = {(r.holder_sector, r.holder_subsector): r.closing_stock_bil for r in origin.itertuples()}
    weights = ht.displacement_weights(origin_bills, buyer=buyer, donors=donors)
    path = ht.BuildAndHoldPath.from_points(
        [{"date": origin_day, "increment_bil": 0.0}, {"date": build_end, "increment_bil": size},
         {"date": end, "increment_bil": size}]
    )
    rows = ht.target_rows(ht.closing_holder_stocks(closure), path=path, weights=weights)
    ht.check_target_rows(rows, path=path, weights=weights)
    return rows, weights


def _scenario(tmp_path: Path, baseline, rows: list[dict], **kwargs):
    params = copy.deepcopy({k: v for k, v in baseline[0].items() if k != "_final_portfolio"}) if not kwargs \
        else _params(tmp_path / "inputs", **kwargs)
    params["holder_position_target"] = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows}
    return _run(params, tmp_path / "run")


def _realized(closure: pd.DataFrame) -> dict[tuple[str, tuple[str, str]], float]:
    stocks = ht.closing_holder_stocks(closure)
    return {(r.period_end, (r.holder_sector, r.holder_subsector)): r.closing_stock_bil for r in stocks.itertuples()}


BANK_DONORS = [FOREIGN, PRIVATE_DNB, PRIVATE_MMF]
MMF_DONORS = [BANKS, FOREIGN, PRIVATE_DNB]


@pytest.fixture(scope="module")
def bank_case(tmp_path_factory, baseline):
    rows, weights = _targets(baseline[2], buyer=BANKS, donors=BANK_DONORS, size=60.0)
    results, portfolio, closure = _scenario(tmp_path_factory.mktemp("banks"), baseline, rows)
    return rows, weights, results, portfolio, closure


# --- Zero-target parity and the issuer side ---


def test_a_zero_target_trades_nothing_and_leaves_every_result_unchanged(tmp_path, baseline) -> None:
    stocks = ht.closing_holder_stocks(baseline[2])
    rows = [
        {"date": r.period_end, "phase": "hold", "holder_type": r.holder_sector, "holder_subbucket": r.holder_subsector,
         "target_stock_bil": r.closing_stock_bil}
        for r in stocks.itertuples()
    ]
    results, _, closure = _scenario(tmp_path, baseline, rows)
    assert results["HolderTargetTradeFace"].sum() == 0.0
    shared = [column for column in baseline[1].columns]
    pd.testing.assert_frame_equal(results[shared], baseline[1], check_exact=True)
    pd.testing.assert_frame_equal(closure, baseline[2], check_exact=True)


def test_positions_must_match_the_baseline_before_the_intervention(tmp_path, baseline) -> None:
    rows, _ = _targets(baseline[2], buyer=BANKS, donors=BANK_DONORS, size=60.0)
    moved = [dict(row) for row in rows]
    for row in moved:
        if row["date"] == "2026-09-22" and row["holder_type"] == "Banks":
            row["target_stock_bil"] += 1.0
        if row["date"] == "2026-09-22" and row["holder_type"] == "Foreign":
            row["target_stock_bil"] -= 1.0
    with pytest.raises(ht.HolderTargetError, match="before the intervention starts"):
        _scenario(tmp_path, baseline, moved)


def test_treasury_side_is_identical_to_the_matched_baseline(bank_case, baseline) -> None:
    results, base = bank_case[2], baseline[1]
    for column in ("TotalDebt_Agg", "NewDebtIssued", "AuctionProceeds", "InterestOutlay_Period",
                   "PrincipalPaid_Bonds", "FinancingCost_Period", "TGA", "DebtHeld_CentralBank"):
        assert results[column].to_numpy() == pytest.approx(base[column].to_numpy(), rel=1e-12, abs=1e-9), column


# --- Realized positions ---


def test_every_close_meets_the_complete_route_vector(bank_case) -> None:
    rows, _, results, _, closure = bank_case
    realized = _realized(closure)
    errors = [abs(realized[(r["date"], (r["holder_type"], r["holder_subbucket"]))] - r["target_stock_bil"]) for r in rows]
    assert max(errors) <= TOLERANCE
    assert results["HolderTargetMaxError"].max() <= TOLERANCE
    # The added bank position is the declared path at the end of the build and after it.
    base = {(r["date"], (r["holder_type"], r["holder_subbucket"])): r["baseline_stock_bil"] for r in rows}
    for day in (BUILD_END, END):
        assert realized[(day, BANKS)] - base[(day, BANKS)] == pytest.approx(60.0, abs=TOLERANCE)


def test_donors_give_up_the_position_in_fixed_origin_bill_proportions(bank_case, baseline) -> None:
    rows, weights, _, _, closure = bank_case
    origin = ht.closing_holder_stocks(baseline[2], bills_only=True)
    bills = origin[origin["period_end"] == ORIGIN].set_index(["holder_sector", "holder_subsector"])["closing_stock_bil"]
    donor_bills = sum(bills[donor] for donor in BANK_DONORS)
    realized = _realized(closure)
    base = {(r["date"], (r["holder_type"], r["holder_subbucket"])): r["baseline_stock_bil"] for r in rows}
    for donor in BANK_DONORS:
        assert weights[donor] == pytest.approx(-bills[donor] / donor_bills)
        assert realized[(END, donor)] - base[(END, donor)] == pytest.approx(60.0 * weights[donor], abs=TOLERANCE)


def test_money_funds_buy_from_other_domestic_nonbanks_too_who_are_paid_into_deposits(tmp_path, baseline) -> None:
    # The money funds' increment is theirs alone, while other domestic nonbanks sell their
    # share like any other holder, so Private as a whole gains less than the money funds.
    rows, weights = _targets(baseline[2], buyer=PRIVATE_MMF, donors=MMF_DONORS, size=40.0)
    results, _, closure = _scenario(tmp_path, baseline, rows)
    realized = _realized(closure)
    base = {(r["date"], (r["holder_type"], r["holder_subbucket"])): r["baseline_stock_bil"] for r in rows}
    assert weights[PRIVATE_DNB] < 0.0
    for day in sorted({r["date"] for r in rows if r["date"] >= BUILD_END}):
        assert realized[(day, PRIVATE_MMF)] - base[(day, PRIVATE_MMF)] == pytest.approx(40.0, abs=TOLERANCE)
        assert realized[(day, PRIVATE_DNB)] - base[(day, PRIVATE_DNB)] == pytest.approx(
            40.0 * weights[PRIVATE_DNB], abs=TOLERANCE)
    # Both Private routes settle in covered deposits here (pass-through 1.0): what the funds
    # pay the other domestic nonbanks comes back to deposits, so only their purchases from
    # banks and foreign holders, net, leave them.
    net = {label: results[f"HolderTargetSoldCash_{label}"] - results[f"HolderTargetBoughtCash_{label}"]
           for label in ("PrivateDomesticNonbank", "PrivateMMF")}
    assert net["PrivateDomesticNonbank"].sum() > 0.0
    assert results["HolderTargetDepositEffect"].to_numpy() == pytest.approx(
        (net["PrivateDomesticNonbank"] + net["PrivateMMF"]).to_numpy(), abs=1e-9)


def test_a_route_declared_unchanged_stays_on_its_baseline_holdings(tmp_path, baseline) -> None:
    # A route that is neither buyer nor donor keeps exactly its baseline holdings at every
    # close, even while the routes around it trade.
    rows, weights = _targets(baseline[2], buyer=PRIVATE_MMF, donors=[BANKS, FOREIGN], size=40.0)
    _, _, closure = _scenario(tmp_path, baseline, rows)
    realized = _realized(closure)
    base = {(r["date"], (r["holder_type"], r["holder_subbucket"])): r["baseline_stock_bil"] for r in rows}
    assert weights[PRIVATE_DNB] == 0.0
    for day in sorted({r["date"] for r in rows if r["date"] >= BUILD_END}):
        assert realized[(day, PRIVATE_MMF)] - base[(day, PRIVATE_MMF)] == pytest.approx(40.0, abs=TOLERANCE)
        assert realized[(day, PRIVATE_DNB)] - base[(day, PRIVATE_DNB)] == pytest.approx(0.0, abs=TOLERANCE)


def test_a_redemption_with_no_auction_is_restored_by_a_maintenance_purchase(tmp_path) -> None:
    # Foreign's 80 of bills maturing 2026-10-05 are the only securities redeemed that day,
    # and the debt path steps down by them, so Treasury issues nothing that day. Part of
    # those bills were sold to banks during the build, so the banks' holdings fall below
    # target and only a secondary purchase can restore them.
    day = "2026-10-05"
    base_params = _params(tmp_path / "base_inputs", debt_step=(day, 80.0))
    base_results, _, closure = _run(base_params, tmp_path / "base")
    rows, _ = _targets(closure, buyer=BANKS, donors=BANK_DONORS, size=60.0)
    params = _params(tmp_path / "inputs", debt_step=(day, 80.0))
    params["holder_position_target"] = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows}
    results, portfolio, closure = _run(params, tmp_path / "run")
    assert base_results.loc[day, "NewDebtIssued"] == 0.0 and results.loc[day, "NewDebtIssued"] == 0.0
    redeemed_by_banks = float(portfolio.loc[
        (portfolio["HolderType"] == "Banks") & (portfolio["MaturityDate"] == pd.Timestamp(day)), "FaceValue"
    ].sum())
    assert redeemed_by_banks > 1.0
    assert results.loc[day, "HolderTargetPhase"] == "hold"
    # The Fed buys from banks that day. Only the change in bank sales relative
    # to the matched baseline increases the amount the controller must restore.
    def fed_bought_from_banks(run):
        seller_routes = json.loads(run.loc[day, "CBOFedPurchaseOperation"])["seller_routes"]
        return seller_routes.get("Banks", {}).get("par", 0.0)

    expected_gap = redeemed_by_banks + fed_bought_from_banks(results) - fed_bought_from_banks(base_results)
    assert results.loc[day, "HolderTargetPreTradeMaxGap"] == pytest.approx(expected_gap, abs=TOLERANCE)
    assert results.loc[day, "HolderTargetBoughtFace_Banks"] == pytest.approx(expected_gap, abs=TOLERANCE)
    realized = _realized(closure)
    for row in rows:
        if row["date"] == day:
            participant = (row["holder_type"], row["holder_subbucket"])
            assert realized[(day, participant)] == pytest.approx(row["target_stock_bil"], abs=TOLERANCE)


def test_federal_reserve_purchases_cannot_leave_a_target_missed(tmp_path, baseline) -> None:
    rows, _ = _targets(baseline[2], buyer=PRIVATE_MMF, donors=MMF_DONORS, size=40.0)
    results, _, closure = _scenario(tmp_path, baseline, rows)
    assert results["CBOFedSecondaryPurchaseFace"].sum() > 0.0
    realized = _realized(closure)
    assert max(abs(realized[(r["date"], (r["holder_type"], r["holder_subbucket"]))] - r["target_stock_bil"])
               for r in rows) <= TOLERANCE


# --- Settlement and accounting ---


def test_trades_settle_at_the_dated_bill_price(tmp_path) -> None:
    # With a flat 4% curve a bill settles at face / (1 + 0.02 * days to maturity / days in
    # the half-year ending at maturity): Appendix B's simple interest for a part-period, which
    # for a bill is Treasury's investment-rate convention. The Fed's trades use it too.
    from dateutil.relativedelta import relativedelta

    from sim_engine import _fed_secondary_dirty_value

    years = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    day = pd.Timestamp("2026-09-25")

    def price(row, amount):
        return _fed_secondary_dirty_value(row, amount, day, years, [0.04] * len(years),
                                          interpolation_method="linear", floor_zero=False)

    gaps = {BANKS: -10.0, FOREIGN: 10.0, PRIVATE_DNB: 0.0, PRIVATE_MMF: 0.0, PRIVATE_MMF_ADDED: 0.0}
    _, report = ht.rebalance_holder_positions(_book(), gaps, current_date=day, price=price)

    def closed_form(maturity: str, face: float) -> float:
        end = pd.Timestamp(maturity)
        start = end - relativedelta(months=6)
        return face / (1.0 + 0.02 * (end - day).days / (end - start).days)

    # The banks hold 60 maturing within a month and 40 in under three months: the sale of
    # 10 takes 6 and 4.
    assert report.trade_face == pytest.approx(10.0)
    assert report.trade_cash == pytest.approx(closed_form("2026-10-02", 6.0) + closed_form("2026-12-15", 4.0), rel=1e-12)


def test_trades_settle_at_dirty_prices_while_the_stock_target_is_exact(bank_case) -> None:
    results = bank_case[2]
    traded = results[results["HolderTargetTradeFace"] > 0.0]
    assert not traded.empty
    assert (traded["HolderTargetTradeCash"] < traded["HolderTargetTradeFace"]).all()
    assert traded["HolderTargetMaxError"].max() <= TOLERANCE


def test_tdc_components_sum_and_the_trades_settle_once_in_secondary_trades(bank_case) -> None:
    results = bank_case[2]
    identity = results[["TDC_FiscalFlow", "TDC_DebtService", "TDC_AuctionAbsorption", "TDC_SecondaryTrades", "TDC_Other"]]
    assert identity.sum(axis=1).to_numpy() == pytest.approx(results["TDC_Change"].to_numpy(), abs=1e-9)
    assert results["TDC_SecondaryTrades"].to_numpy() == pytest.approx(
        (results["CBOFedSecondaryPurchaseDepositEffect"] + results["HolderTargetDepositEffect"]).to_numpy(), abs=1e-9
    )
    # Banks pay with reserves: deposit-funded sellers' deposits rise by what they receive.
    sold_to_banks = results["HolderTargetSoldCash_PrivateDomesticNonbank"] + results["HolderTargetSoldCash_PrivateMMF"]
    bought = results["HolderTargetBoughtCash_PrivateDomesticNonbank"] + results["HolderTargetBoughtCash_PrivateMMF"]
    assert results["HolderTargetDepositEffect"].to_numpy() == pytest.approx((sold_to_banks - bought).to_numpy(), abs=1e-9)


def test_trc_closes_with_route_states_when_money_funds_pay_from_on_rrp(tmp_path) -> None:
    routing = _routing(on_rrp_opening=500.0, on_rrp_share=1.0)
    base_params = _params(tmp_path / "base_inputs", pass_through=0.0, routing=routing)
    base_results, _, closure = _run(base_params, tmp_path / "base")
    rows, _ = _targets(closure, buyer=PRIVATE_MMF, donors=MMF_DONORS, size=40.0)
    params = _params(tmp_path / "inputs", pass_through=0.0, routing=routing)
    params["holder_position_target"] = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows}
    results, _, _ = _run(params, tmp_path / "run")
    trc = results.attrs["trc_results"].set_index("Date")
    base_trc = base_results.attrs["trc_results"].set_index("Date")
    assert trc["TRC_LegacyClosureGap"].abs().max() <= 1e-9
    # The money funds pay sellers from ON RRP: their claim falls and the sellers' banks
    # gain reserves, by the cash paid net of what the funds received on redemptions.
    net_paid = (results["HolderTargetBoughtCash_PrivateMMF"] - results["HolderTargetSoldCash_PrivateMMF"]).sum()
    assert net_paid > 30.0
    assert base_trc["ONRRP_MMF"].iloc[-1] - trc["ONRRP_MMF"].iloc[-1] > 0.0
    # The TRC difference is the difference in holder-routed reserves (which moves only
    # because the Federal Reserve buys a different mix at different prices) less the
    # change in route-held balances: ON RRP and the foreign official accounts.
    states = list(TRC_ROUTE_STATES)
    legacy = results["Reserves"].iloc[-1] - base_results["Reserves"].iloc[-1]
    assert abs(legacy) < 1e-3
    assert trc["TRC_Cumulative"].iloc[-1] - base_trc["TRC_Cumulative"].iloc[-1] == pytest.approx(
        legacy - (trc[states].iloc[-1] - base_trc[states].iloc[-1]).sum(), abs=1e-9
    )
    # With no money-fund pass-through, only the deposit-funded route's trades reach
    # deposits: the other domestic nonbanks are paid into deposits for the bills they sell.
    dnb_net = results["HolderTargetSoldCash_PrivateDomesticNonbank"] - results["HolderTargetBoughtCash_PrivateDomesticNonbank"]
    assert dnb_net.sum() > 0.0
    assert results["HolderTargetDepositEffect"].to_numpy() == pytest.approx(dnb_net.to_numpy(), abs=1e-12)


def test_too_little_on_rrp_raises_with_the_route_and_date(tmp_path) -> None:
    routing = _routing(on_rrp_opening=5.0, on_rrp_share=1.0)
    base_params = _params(tmp_path / "base_inputs", pass_through=0.0, routing=routing)
    _, _, closure = _run(base_params, tmp_path / "base")
    rows, _ = _targets(closure, buyer=PRIVATE_MMF, donors=MMF_DONORS, size=40.0)
    params = _params(tmp_path / "inputs", pass_through=0.0, routing=routing)
    params["holder_position_target"] = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows}
    with pytest.raises(TrcRoutingError, match=r"ONRRP_MMF is negative at the 2026-09-2\d"):
        _run(params, tmp_path / "run")


def test_added_mmf_position_routes_purchases_redemptions_and_reinvestment(tmp_path) -> None:
    params = _params(tmp_path / "inputs", fed=False)
    observed, _, _ = _run(params, tmp_path / "observed")
    position = {"settlement_route": "covered_deposit_settled", "opening_on_rrp_bil": 60.0}
    params[PRIVATE_SUBBUCKET_MMF_ADDED] = position
    base, _, base_closure = _run(params, tmp_path / "base")
    assert base["TDC_Change"].equals(observed["TDC_Change"])
    assert base.attrs["trc_results"]["TRC_Change"].equals(observed.attrs["trc_results"]["TRC_Change"])
    assert base["Reserves"].iloc[0] == observed["Reserves"].iloc[0] - 60.0
    assert base.attrs["trc_results"]["ONRRP_MMF"].iloc[0] == observed.attrs["trc_results"]["ONRRP_MMF"].iloc[0] + 60.0
    rows, _ = _targets(base_closure, buyer=PRIVATE_MMF_ADDED, donors=MMF_DONORS, size=60.0)
    params["holder_position_target"] = {
        "mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows,
        "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED,
    }
    deposits, dep_portfolio, dep_closure = _run(params, tmp_path / "deposits")
    position["settlement_route"] = "on_rrp_mmf"
    onrrp, rrp_portfolio, rrp_closure = _run(params, tmp_path / "onrrp")
    pd.testing.assert_frame_equal(dep_portfolio, rrp_portfolio)
    pd.testing.assert_frame_equal(ht.closing_holder_stocks(dep_closure), ht.closing_holder_stocks(rrp_closure))
    trade_columns = [column for column in ht.RESULT_COLUMNS if column.startswith(("HolderTargetBought", "HolderTargetSold"))]
    assert deposits[trade_columns].equals(onrrp[trade_columns])
    for result in (deposits, onrrp):
        assert result["HolderStock_PrivateMMF"].to_numpy() == pytest.approx(base["HolderStock_PrivateMMF"].to_numpy())
        assert result["HolderStock_PrivateMMFAdded"].iloc[0] == 0.0
        assert result["HolderStock_PrivateMMFAdded"].iloc[-1] == pytest.approx(60.0)
        assert result["HolderTargetSoldFace_PrivateMMF"].sum() == 0.0
        assert result["HolderTargetBoughtFace_PrivateMMF"].sum() == 0.0
    additions = deposits[list(MMF_ADDED_TDC_COLUMNS.values())]
    assert additions[MMF_ADDED_TDC_COLUMNS["TDC_PrincipalToDU"]].sum() > 0
    payments = pd.read_csv(tmp_path / "deposits" / "tdcsim_period_payment_aggregates.csv.gz")
    discounts = payments[payments["holder_subsector"].eq(PRIVATE_SUBBUCKET_MMF_ADDED)
                         & payments["payment_type"].eq("bill_discount")]
    paid_discount = float(discounts["amount_bil"].sum())
    assert paid_discount > 0.0
    assert additions[MMF_ADDED_TDC_COLUMNS["TDC_BillDiscountInterestToDU"]].sum() == pytest.approx(paid_discount)
    cash = additions.sum(axis=1).to_numpy()
    assert (deposits["TDC_Change"] - onrrp["TDC_Change"]).to_numpy() == pytest.approx(cash, abs=1e-10)
    dep_trc, rrp_trc = (r.attrs["trc_results"] for r in (deposits, onrrp))
    assert (dep_trc["TRC_Change"] - rrp_trc["TRC_Change"]).to_numpy() == pytest.approx(cash, abs=1e-10)
    assert (rrp_trc["ONRRP_MMF"] - observed.attrs["trc_results"]["ONRRP_MMF"]).min() >= 0.0
    position["opening_on_rrp_bil"] = 0.1
    with pytest.raises(TrcRoutingError, match="ONRRP_MMF is negative"):
        _run(params, tmp_path / "overdrawn")


def test_added_mmf_funding_preserves_ordinary_holdings_when_the_fed_buys(tmp_path) -> None:
    # Exercise the production Fed purchase rule and controller in the same run.
    size = 60.0
    params = _params(tmp_path / "inputs", fed=True)
    position = {"settlement_route": "covered_deposit_settled", "opening_on_rrp_bil": size}
    params[PRIVATE_SUBBUCKET_MMF_ADDED] = position
    base, _, closure = _run(params, tmp_path / "baseline")
    assert base["CBOFedSecondaryPurchaseFace"].sum() > 0.0
    rows, _ = _targets(closure, buyer=PRIVATE_MMF_ADDED, donors=MMF_DONORS, size=size)
    target = pd.Series({
        pd.Timestamp(row["date"]): row["target_stock_bil"]
        for row in rows if (row["holder_type"], row["holder_subbucket"]) == PRIVATE_MMF_ADDED
    })
    params["holder_position_target"] = {
        "mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows,
        "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED,
    }
    deposits, deposit_book, _ = _run(params, tmp_path / "deposits")
    params[PRIVATE_SUBBUCKET_MMF_ADDED] = {**position, "settlement_route": "on_rrp_mmf"}
    onrrp, onrrp_book, _ = _run(params, tmp_path / "onrrp")

    for results in (deposits, onrrp):
        purchases = results.loc[results["CBOFedSecondaryPurchaseFace"] > 0, "CBOFedPurchaseOperation"]
        sellers = [json.loads(operation)["seller_routes"] for operation in purchases]
        assert any(sale.get(MMF, {}).get("par", 0.0) > 0.0 for sale in sellers)
        assert any(sale.get(PRIVATE_SUBBUCKET_MMF_ADDED, {}).get("par", 0.0) > 0.0 for sale in sellers)
        pd.testing.assert_series_equal(results["HolderStock_PrivateMMF"], base["HolderStock_PrivateMMF"],
                                       check_exact=False, rtol=0.0, atol=1e-10)
        assert (results[["HolderTargetBoughtFace_PrivateMMF", "HolderTargetSoldFace_PrivateMMF"]] == 0.0).all().all()
        pd.testing.assert_series_equal(results["TGA"], base["TGA"], check_exact=False, rtol=0.0, atol=1e-9)
        added = results["HolderStock_PrivateMMFAdded"]
        assert added.iloc[0] == 0.0
        pd.testing.assert_series_equal(added.iloc[1:], target, check_names=False, check_freq=False,
                                       check_exact=False, rtol=0.0, atol=TOLERANCE)
        assert added.iloc[-1] == pytest.approx(size, abs=TOLERANCE)

    donors = [ht.PARTICIPANT_LABELS[route] for route in MMF_DONORS]
    sales = [f"HolderTargetSold{basis}_{donor}" for basis in ("Face", "Cash") for donor in donors]
    assert (deposits[sales].sum() > 0.0).all()
    pd.testing.assert_frame_equal(deposits[sales], onrrp[sales], check_exact=True)
    pd.testing.assert_frame_equal(deposit_book, onrrp_book, check_exact=True)


REMIT_ORIGIN, REMIT_BUILD_END = "2026-09-21", "2026-09-22"


def _remitting_pair(tmp_path: Path, *, routing: dict | None = None, mmf_bill_share: float = 0.0,
                    book: list[dict] = OPENING, start: str = START, end: str = END, origin: str = REMIT_ORIGIN,
                    build_end: str = REMIT_BUILD_END):
    """A baseline and an ON RRP-funded money-fund scenario whose Fed remits, so a change in
    its income changes Treasury's borrowing. The build ends early enough that the changed
    income is remitted on 2026-09-30, a day Treasury sizes its auction to land cash on target."""

    # A Fed that earns more than it pays: 3,000 of reserves once the fixture's 15,860 of
    # ON RRP and foreign balances are drawn from the H.4.1 level.
    small_fed = {"h41_reserve_balances_bil": 18_860.0, "h41_tga_bil": 0.0, "other_income_bil": 3.0,
                 "h41_earnings_remittances_due_bil": 0.0}
    params = _params(tmp_path / "inputs", fed=True, fed_income=small_fed, routing=routing, book=book, start=start,
                     end=end)
    # Issuance follows Treasury's cash, as in the release runs, so a remittance moves borrowing.
    params["treasury_issuance_profile"] = _profile(nominal_only=True)
    params["funding_rule"].update({"mode": "cbo_operating_cash_target_financing_v1", "cash_closure_target_bil": 0.0,
                                   "validation_floor_bil": -0.000001})
    if mmf_bill_share:
        params["sector_preferences"][PRIVATE_SUBBUCKET_SHARES_KEY] = {
            "bills": {DNB: 1.0 - mmf_bill_share, MMF: mmf_bill_share}
        }
    position = {"settlement_route": "covered_deposit_settled", "opening_on_rrp_bil": 60.0}
    params[PRIVATE_SUBBUCKET_MMF_ADDED] = position
    base, _, closure = _run(params, tmp_path / "baseline", start=start, end=end)
    assert ht.DEBT_DIFFERENCE_COLUMN not in base.columns  # the baseline has no target
    rows, weights = _targets(closure, buyer=PRIVATE_MMF_ADDED, donors=MMF_DONORS, size=60.0,
                             origin_day=origin, build_end=build_end, end=end)
    config = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows,
              "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED}
    params["holder_position_target"] = config
    params[PRIVATE_SUBBUCKET_MMF_ADDED] = {**position, "settlement_route": "on_rrp_mmf"}
    onrrp, _, onrrp_closure = _run(params, tmp_path / "onrrp", start=start, end=end)
    target_path = ht.BuildAndHoldPath.from_points([
        {"date": origin, "increment_bil": 0.0}, {"date": build_end, "increment_bil": 60.0},
        {"date": end, "increment_bil": 60.0},
    ])
    principal = pd.read_csv(tmp_path / "onrrp" / "tdcsim_period_principal_aggregates.csv.gz", keep_default_na=False)
    return base, closure, onrrp, onrrp_closure, rows, weights, config, target_path, principal


def _assert_debt_difference_held_by_donors(onrrp, onrrp_closure, rows, weights) -> None:
    """Every close: the buyer and the untraded routes on their targets, the donors on theirs
    plus their share of the day's D."""

    difference = onrrp[ht.DEBT_DIFFERENCE_COLUMN]
    realized = _realized(onrrp_closure)
    for row in rows:
        day, route = row["date"], (row["holder_type"], row["holder_subbucket"])
        held = difference.loc[pd.Timestamp(day)] * (-weights[route] if weights[route] < 0.0 else 0.0)
        if row["phase"] == "pre":
            assert difference.loc[pd.Timestamp(day)] == pytest.approx(0.0, abs=TOLERANCE)
            held = 0.0
        assert realized[(day, route)] == pytest.approx(row["target_stock_bil"] + held, abs=1e-9), (day, route)


def test_a_position_that_changes_fed_income_leaves_the_debt_difference_with_its_donors(tmp_path) -> None:
    """The trades change what the Fed pays on its liabilities: the added fund's ON RRP cash
    becomes reserves earning IORB, and foreign sellers leave part of their proceeds in
    unremunerated Fed deposits. Here the second effect is larger, so the Fed remits more and
    Treasury borrows less than in the baseline. That debt difference D is held by the holders
    who give up the position, in their donor weights; the added position and the ordinary
    money funds stay exactly on their targets."""

    base, _, onrrp, onrrp_closure, rows, weights, config, target_path, principal = _remitting_pair(tmp_path)
    difference = onrrp[ht.DEBT_DIFFERENCE_COLUMN]
    remitted = onrrp.loc[:"2026-09-30", "CB_Remittance"].sum() - base.loc[:"2026-09-30", "CB_Remittance"].sum()
    assert abs(remitted) > TOLERANCE  # the trades moved what the Fed remitted
    assert onrrp.loc[:"2026-09-29", ht.DEBT_DIFFERENCE_COLUMN].abs().max() < TOLERANCE
    # and Treasury borrows that much less (or more) at the auction that lands its cash on target
    assert difference.iloc[-1] * remitted < 0.0 and abs(difference.iloc[-1]) > TOLERANCE
    _assert_debt_difference_held_by_donors(onrrp, onrrp_closure, rows, weights)
    assert onrrp[ht.CORRECTION_FACE_COLUMN].abs().max() == 0.0  # ordinary funds take no auction bills here

    def check(results, closure=onrrp_closure, rows=rows):
        return ht.check_stock_target_run(results, closure, {**config, "rows": rows}, target_path=target_path,
                                         weights=weights, batched_settlement=True, principal=principal)

    assert check(onrrp) == pytest.approx(difference.abs().max())
    last = onrrp.index[-1]
    tampered = {
        "does not follow the debt-difference rule": [  # D booked with the ordinary money funds
            ("HolderTarget_PrivateMMF", difference.iloc[-1]), ("HolderTarget_Banks", difference.iloc[-1] * weights[BANKS])],
        "public stock less the baseline's": [(ht.DEBT_DIFFERENCE_COLUMN, -difference.iloc[-1])],
        "nonfinite": [(ht.DEBT_DIFFERENCE_COLUMN, float("nan"))],
        # A reported target 9e-6 off is far below a relative 1e-9 of 10,000, but above the tolerance.
        "HolderTarget_|Banks target": [("HolderTarget_Banks", 9.0 * TOLERANCE)],
    }
    for message, changes in tampered.items():
        results = onrrp.copy()
        for column, change in changes:
            results.loc[last, column] = results.loc[last, column] + change
        with pytest.raises(ht.HolderTargetError, match=message):
            check(results)
    # Every close after the opening must be there, the first and last included, against the
    # dated target sequence rather than the surviving rows' own endpoints.
    for dropped, message in (("2026-10-01", "2026-10-01"), ("2026-09-21", "2026-09-21"),
                             ("2026-10-12", "2026-10-12"), ("2026-09-20", "2026-09-21")):
        with pytest.raises(ht.HolderTargetError, match=f"has no close on {message}"):
            check(onrrp.drop(pd.Timestamp(dropped)))
    # Reported stocks are what the fiscal-year holder table publishes: offsetting errors that
    # leave the public total unchanged must still fail against the stock closure.
    offset = onrrp.copy()
    offset.loc[last, ht.STOCK_COLUMNS["HolderStock"][BANKS]] += 50.0
    offset.loc[last, ht.STOCK_COLUMNS["HolderStock"][FOREIGN]] -= 50.0
    with pytest.raises(ht.HolderTargetError, match="Banks reported stock on 2026-10-12 is not its stock closure"):
        check(offset)

    # Before the build every target is the baseline's, as the controller sets it: a public
    # stock 0.9e-6 above the baseline, inside the 1e-6 tolerance, is not given to the donors.
    def deviate(day):
        results, closure = onrrp.copy(), onrrp_closure.copy()
        for route, change in ((BANKS, -0.9e-6), (FOREIGN, 0.9e-6), (PRIVATE_DNB, 0.9e-6)):
            match = ((closure["period_end"] == day) & (closure["axis"] == "holder")
                     & (closure["debt_scope"] == "controlled_public_marketable")
                     & (closure["holder_sector"] == route[0])
                     & (closure["holder_subsector"].fillna("").astype(str) == route[1]))
            closure.loc[closure.index[match][0], "closing_debt_stock_bil"] += change
            results.loc[pd.Timestamp(day), ht.STOCK_COLUMNS["HolderStock"][route]] += change
        results.loc[pd.Timestamp(day), ht.DEBT_DIFFERENCE_COLUMN] += 0.9e-6
        return results, closure

    results, closure = deviate(REMIT_ORIGIN)
    assert check(results, closure=closure) == pytest.approx(difference.abs().max())
    results, closure = deviate("2026-09-23")  # in the hold the donors hold D, so banks now miss
    with pytest.raises(ht.HolderTargetError, match="Banks misses its target on 2026-09-23"):
        check(results, closure=closure)
    with pytest.raises(ht.HolderTargetError, match="not the dated target sequence: 2026-10-01 has no target"):
        check(onrrp, rows=[row for row in rows if row["date"] != "2026-10-01"])
    with pytest.raises(ht.HolderTargetError, match="stock closure has no 2026-10-01 close"):
        check(onrrp, closure=onrrp_closure[onrrp_closure["period_end"] != "2026-10-01"])


def test_ordinary_money_funds_trade_back_what_the_debt_difference_brings_them_at_auction(tmp_path) -> None:
    """The release's auction split: ordinary money funds take 35 percent of private bill
    allocations, and foreign holders settle through banks. The ON RRP-funded position then
    only raises the Fed's interest expense, since reserves earn IORB above the ON RRP rate, so
    the Fed remits less and Treasury issues more. Part of the added bills reaches the untraded
    ordinary money funds at auction; they sell it to the donors in donor weights, ending on
    their baseline target, and that correction is recorded on its own."""

    routing = _routing()
    routing["foreign_settlement_mix"] = {"foreign_bank_settled": 1.0, "foreign_official_deposit": 0.0,
                                         "foreign_official_rrp": 0.0}
    base, base_closure, onrrp, onrrp_closure, rows, weights, config, target_path, principal = _remitting_pair(
        tmp_path, routing=routing, mmf_bill_share=0.35)
    difference = onrrp[ht.DEBT_DIFFERENCE_COLUMN]
    remitted = onrrp.loc[:"2026-09-30", "CB_Remittance"].sum() - base.loc[:"2026-09-30", "CB_Remittance"].sum()
    assert remitted < -TOLERANCE and difference.iloc[-1] > TOLERANCE  # the Fed remits less; Treasury owes more
    correction = onrrp[ht.CORRECTION_FACE_COLUMN]
    assert correction.min() < -TOLERANCE  # the ordinary funds sold their auction share of D back
    days = correction.index[correction != 0.0]
    assert (difference.loc[days].abs() > TOLERANCE).all()
    assert onrrp.loc[days, ht.CORRECTION_CASH_COLUMN].lt(0.0).all()  # they were paid for it
    label = ht.PARTICIPANT_LABELS[PRIVATE_MMF]
    assert onrrp[f"HolderTargetSoldFace_{label}"].to_numpy() == pytest.approx(-correction.to_numpy(), abs=1e-12)
    assert onrrp[f"HolderTargetBoughtFace_{label}"].abs().max() == 0.0
    # The ordinary funds end every close exactly on their baseline stock, as do the buyer and
    # the donors on theirs plus their share of D.
    baseline_mmf = {(r.period_end, (r.holder_sector, r.holder_subsector)): r.closing_stock_bil
                    for r in ht.closing_holder_stocks(base_closure).itertuples()}
    realized = _realized(onrrp_closure)
    for key, value in realized.items():
        if key[1] == PRIVATE_MMF:
            assert value == pytest.approx(baseline_mmf[key], abs=1e-9), key
    _assert_debt_difference_held_by_donors(onrrp, onrrp_closure, rows, weights)

    def check(results, closure=onrrp_closure):
        return ht.check_stock_target_run(results, closure, config, target_path=target_path, weights=weights,
                                         batched_settlement=True, principal=principal)

    assert check(onrrp) == pytest.approx(difference.abs().max())
    day = days[0]
    for column, change in ((ht.CORRECTION_FACE_COLUMN, 0.001), (ht.CORRECTION_CASH_COLUMN, -0.001),
                           (f"HolderTargetBoughtFace_{label}", 0.001)):
        tampered = onrrp.copy()
        tampered.loc[day, column] += change
        with pytest.raises(ht.HolderTargetError, match="correction is not the untraded routes' trades"):
            check(tampered)
    for value in (float("nan"), -0.001):  # every trade leg must be a finite nonnegative amount
        tampered = onrrp.copy()
        tampered.loc[day, f"HolderTargetSoldFace_{label}"] = value
        with pytest.raises(ht.HolderTargetError, match=f"{label} SoldFace on .* is not a finite nonnegative amount"):
            check(tampered)
    # Weekends are closed-market closes: nothing trades and, with no TIPS here, nothing moves.
    assert onrrp.loc["2026-10-03", ht.MARKET_OPEN_COLUMN] == 0.0 and onrrp.loc["2026-10-05", ht.MARKET_OPEN_COLUMN] == 1.0
    assert onrrp[ht.CLOSED_DAY_GAP_COLUMN].max() <= TOLERANCE
    # Banks hold 4.0 of TIPS the baseline lacks, in place of 4.0 of bills, from the 2026-10-02
    # market close; the reference CPI rises 1e-6 by the 2026-10-03 and 10-04 weekend closes. That
    # indexation leaves a gap the next market day trades back, and the check accepts it.
    friday, tips_day, face, step = "2026-10-02", pd.Timestamp("2026-10-03"), 4.0, 1.0e-6
    tips_change = face * step
    share = -weights[BANKS]
    indexed, indexed_closure = onrrp.copy(), onrrp_closure.copy()

    def holder_row(closure, day, holder):
        return closure.index[(closure["period_end"] == day) & (closure["axis"] == "holder")
                             & (closure["debt_scope"] == "controlled_public_marketable")
                             & (closure["holder_sector"] == holder) & (closure["instrument_type"] == "Fixed")][0]

    stock_columns = ["closing_face_stock_bil", "closing_adjusted_principal_stock_bil", "closing_debt_stock_bil"]
    added = []
    weekend = ("2026-10-03", "2026-10-04")
    for day, adjusted in ((friday, face), *((closed, face * (1.0 + step)) for closed in weekend)):
        bill = holder_row(indexed_closure, day, "Banks")
        indexed_closure.loc[bill, stock_columns] -= face
        added.append(indexed_closure.loc[[bill]].assign(
            instrument_type="TIPS", closing_face_stock_bil=face, closing_adjusted_principal_stock_bil=adjusted,
            closing_debt_stock_bil=adjusted))
        added.append(added[-1].assign(axis="instrument", holder_sector="", holder_subsector=""))
    indexed_closure = pd.concat([indexed_closure, *added], ignore_index=True)
    banks_tips = indexed_closure.index[-4]  # the Banks TIPS row at the 2026-10-03 close
    for closed in map(pd.Timestamp, weekend):
        indexed.loc[closed, ht.STOCK_COLUMNS["HolderStock"][BANKS]] += tips_change
        indexed.loc[closed, ht.DEBT_DIFFERENCE_COLUMN] += tips_change
        for route in MMF_DONORS:
            indexed.loc[closed, f"HolderTarget_{ht.PARTICIPANT_LABELS[route]}"] += tips_change * -weights[route]
        indexed.loc[closed, ht.CLOSED_DAY_GAP_COLUMN] = tips_change * (1.0 - share)
    assert check(indexed, closure=indexed_closure) == pytest.approx(indexed[ht.DEBT_DIFFERENCE_COLUMN].abs().max())
    probes = {
        "trade on .* a day the securities market is closed": [(tips_day, "HolderTargetBoughtFace_Banks", 0.001)],
        "recorded closed-day gap": [(tips_day, ht.CLOSED_DAY_GAP_COLUMN, 0.001)],
        "closed-day gap on 2026-10-03 is not a finite nonnegative": [(tips_day, ht.CLOSED_DAY_GAP_COLUMN, float("nan"))],
        # The market flag follows the calendar both ways: a weekday marked shut, and a Saturday
        # marked open with a fabricated purchase.
        "2026-10-05 close's market flag disagrees": [(pd.Timestamp("2026-10-05"), ht.MARKET_OPEN_COLUMN, -1.0)],
        "2026-10-03 close's market flag disagrees": [(tips_day, ht.MARKET_OPEN_COLUMN, 1.0),
                                                     (tips_day, "HolderTargetBoughtFace_Banks", 0.001)],
    }
    for message, changes in probes.items():
        tampered = indexed.copy()
        for probe_day, column, change in changes:
            tampered.loc[probe_day, column] += change
        with pytest.raises(ht.HolderTargetError, match=message):
            check(tampered, closure=indexed_closure)
    # Tampering probes. Moving 1.0 of the weekend TIPS from banks to foreign holders keeps D and
    # the total indexation, but not each holder's TIPS principal.
    moved, moved_results = indexed_closure.copy(), indexed.copy()
    moved.loc[banks_tips, stock_columns] -= 1.0
    foreign = moved.loc[[banks_tips]].assign(holder_sector="Foreign", closing_face_stock_bil=1.0,
                                             closing_adjusted_principal_stock_bil=1.0, closing_debt_stock_bil=1.0)
    moved = pd.concat([moved, foreign], ignore_index=True)
    moved_results.loc[tips_day, ht.STOCK_COLUMNS["HolderStock"][BANKS]] -= 1.0
    moved_results.loc[tips_day, ht.STOCK_COLUMNS["HolderStock"][FOREIGN]] += 1.0
    moved_results.loc[tips_day, ht.CLOSED_DAY_GAP_COLUMN] = max(
        abs(moved_results.loc[tips_day, f"HolderTarget_{label}"] - moved_results.loc[tips_day, ht.STOCK_COLUMNS["HolderStock"][p]])
        for p, label in ht.PARTICIPANT_LABELS.items())
    with pytest.raises(ht.HolderTargetError, match="Banks TIPS principal moves on 2026-10-03"):
        check(moved_results, closure=moved)
    # 1.0 more bank bills offset by 1.0 less bank TIPS leaves every gap as it was, inside the
    # tolerance; holdings other than TIPS must still not move.
    offset = indexed_closure.copy()
    offset.loc[holder_row(offset, tips_day.date().isoformat(), "Banks"), "closing_debt_stock_bil"] += 1.0
    offset.loc[banks_tips, ["closing_adjusted_principal_stock_bil", "closing_debt_stock_bil"]] -= 1.0
    with pytest.raises(ht.HolderTargetError, match="Banks holdings other than TIPS move on 2026-10-03"):
        check(indexed, closure=offset)
    quiet = onrrp.copy()  # a correction on a close with no debt difference to correct
    quiet.loc[onrrp.index[2], [ht.CORRECTION_FACE_COLUMN, f"HolderTargetBoughtFace_{label}"]] = 0.001
    with pytest.raises(ht.HolderTargetError, match="without a debt difference"):
        check(quiet)


def test_fedwire_redeems_a_scheduled_maturity_while_the_securities_market_is_shut(tmp_path) -> None:
    """Friday 2026-07-03 is a Fedwire business day with the securities market closed. A bank
    bill due that day is paid, in the baseline as in the scenario, before the intervention:
    the guard and the run check accept the booked redemption and still reject the bank stock
    moving by anything else."""

    bill = _security(13, "Banks", "", "bills", "2026-07-03", 10.0)
    friday = pd.Timestamp("2026-07-03")
    _, _, onrrp, closure, rows, weights, config, target_path, principal = _remitting_pair(
        tmp_path, book=[*OPENING, bill], start="2026-06-29", end="2026-07-14", origin="2026-07-08",
        build_end="2026-07-09")
    assert onrrp.loc[friday, ht.MARKET_OPEN_COLUMN] == 0.0
    paid = principal[(principal["period_end"] == "2026-07-03") & (principal["holder_sector"] == "Banks")]
    assert paid["face_redeemed_bil"].sum() == pytest.approx(10.0)
    assert set(paid["redemption_type"]) == {"scheduled_maturity"}
    realized = _realized(closure)
    assert realized[("2026-07-02", BANKS)] - realized[("2026-07-03", BANKS)] == pytest.approx(10.0)
    assert abs(onrrp.loc[friday, ht.DEBT_DIFFERENCE_COLUMN]) <= TOLERANCE
    assert onrrp.loc[friday, ht.CLOSED_DAY_GAP_COLUMN] <= TOLERANCE

    def check(results=onrrp, closure=closure, principal=principal):
        return ht.check_stock_target_run(results, closure, config, target_path=target_path, weights=weights,
                                         batched_settlement=True, principal=principal)

    check()
    unbooked = principal.drop(paid.index)
    with pytest.raises(ht.HolderTargetError, match="Banks holdings other than TIPS move on 2026-07-03, .* other than by booked redemptions"):
        check(principal=unbooked)
    bought_back = principal.copy()
    bought_back.loc[paid.index, "redemption_type"] = "buyback"
    with pytest.raises(ht.HolderTargetError, match="principal is redeemed on 2026-07-03, a closed-market day, other than at maturity"):
        check(principal=bought_back)
    # A transfer of 5.0 from foreign holders to banks that day is still refused.
    moved, moved_closure = onrrp.copy(), closure.copy()
    for holder, route, change in (("Banks", BANKS, 5.0), ("Foreign", FOREIGN, -5.0)):
        row = moved_closure.index[(moved_closure["period_end"] == "2026-07-03") & (moved_closure["axis"] == "holder")
                                  & (moved_closure["debt_scope"] == "controlled_public_marketable")
                                  & (moved_closure["holder_sector"] == holder)
                                  & (moved_closure["instrument_type"] == "Fixed")][0]
        moved_closure.loc[row, "closing_debt_stock_bil"] += change
        moved.loc[friday, ht.STOCK_COLUMNS["HolderStock"][route]] += change
    moved.loc[friday, ht.CLOSED_DAY_GAP_COLUMN] = 5.0
    with pytest.raises(ht.HolderTargetError, match="Banks holdings other than TIPS move on 2026-07-03"):
        check(results=moved, closure=moved_closure)


def test_a_closed_market_gap_may_move_only_by_tips_indexation_relative_to_the_baseline() -> None:
    """With the securities market shut nothing trades. Every lot stays with its holder at its
    principal, and only TIPS adjusted principal moves, by the change in the reference CPI.
    Indexation of TIPS a participant holds beyond the baseline's moves its gap, and D moves
    with it, which the donors hold in their shares. Any other move raises, whatever the gap."""

    day = pd.Timestamp("2030-10-14")  # Columbus Day, after a weekend
    weights = {BANKS: -0.5, FOREIGN: -0.5, PRIVATE_MMF_ADDED: 1.0}
    path = ht.holder_target_path({
        "mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED,
        "rows": [{"date": day.date().isoformat(), "phase": "hold", "holder_type": holder, "holder_subbucket": route,
                  "target_stock_bil": 100.0 + (0.000046 if (holder, route) == FOREIGN else 0.0),
                  "displacement_weight": weights.get((holder, route), 0.0)}
                 for holder, route in ht.PARTICIPANTS],
    })
    lots = {(BANKS, 1): (False, 100.0, 100.0), (FOREIGN, 2): (True, 50.0, 50.0), (FOREIGN, 3): (False, 50.0, 50.0)}
    last = {route: 100.0 for route in ht.PARTICIPANTS}
    close = ht.MarketClose(lots=lots, reference_cpi=300.0, reference=last, gaps={route: 0.0 for route in ht.PARTICIPANTS})
    # The CPI rises 1e-6: foreign's 50 of TIPS index by 0.00005, the baseline's 46 by 0.000046.
    today = {**lots, (FOREIGN, 2): (True, 50.0, 50.0 * (1.0 + 1e-6))}
    stocks = {**last, FOREIGN: 100.0 + 50.0 * 1e-6}
    reference = {**last, FOREIGN: 100.000046}
    difference = math.fsum(stocks.values()) - math.fsum(reference.values())
    gaps = {route: reference[route] + difference * max(0.0, -weights.get(route, 0.0)) - stocks[route]
            for route in ht.PARTICIPANTS}

    def gap(**changes):
        return path.closed_day_gap(day, **{"gaps": gaps, "lots": today, "matured": {},
                                           "reference_cpi": 300.0 * (1.0 + 1e-6), "market_close": close, **changes})

    assert gap() == pytest.approx(2e-6)
    # Tampering probes: TIPS changing hands, and bills up by what TIPS lose, both with every gap
    # inside the tolerance.
    quiet = {route: 0.0 for route in ht.PARTICIPANTS}
    with pytest.raises(ht.HolderTargetError, match="Banks acquires holdings on 2030-10-14"):
        gap(gaps=quiet, lots={**{k: v for k, v in today.items() if k != (FOREIGN, 2)}, (BANKS, 2): today[(FOREIGN, 2)]})
    with pytest.raises(ht.HolderTargetError, match="Foreign holdings move on 2030-10-14, .* other than by TIPS indexation"):
        gap(gaps=quiet, lots={**today, (FOREIGN, 3): (False, 51.0, 51.0), (FOREIGN, 2): (True, 50.0, 49.00005)})
    with pytest.raises(ht.HolderTargetError, match="other than by TIPS indexation"):
        gap(reference_cpi=300.0)  # the TIPS moved but the CPI did not
    with pytest.raises(ht.HolderTargetError, match="Banks's gap is not explained by TIPS indexation"):
        gap(gaps={**gaps, BANKS: gaps[BANKS] + 3e-6})
    # Every participant's gap must be finite, whichever comes first.
    for route in (BANKS, FOREIGN):
        with pytest.raises(ht.HolderTargetError, match=f"{ht.PARTICIPANT_LABELS[route]} gap on 2030-10-14 is not finite"):
            gap(gaps={**gaps, route: float("nan")})

    # Fedwire may redeem a lot at its scheduled maturity with the securities market shut: the
    # bank bill booked as matured that day leaves, and its holder's gap moves by what the
    # baseline does not also lose.
    redeemed = {k: v for k, v in today.items() if k != (BANKS, 1)}
    matured = {(BANKS, 1): (100.0, day)}
    paid = {**stocks, BANKS: stocks[BANKS] - 100.0}  # the baseline kept that bill, so D falls by 100
    paid_difference = math.fsum(paid.values()) - math.fsum(reference.values())
    paid_gaps = {route: reference[route] + paid_difference * max(0.0, -weights.get(route, 0.0)) - paid[route]
                 for route in ht.PARTICIPANTS}
    assert gap(lots=redeemed, matured=matured, gaps=paid_gaps) == pytest.approx(50.0, abs=1e-5)
    with pytest.raises(ht.HolderTargetError, match="Banks holdings leave on 2030-10-14, .* other than by a scheduled redemption"):
        gap(lots=redeemed, gaps=paid_gaps)  # nothing booked it as matured
    with pytest.raises(ht.HolderTargetError, match="other than by a scheduled redemption"):
        gap(lots=redeemed, matured={(BANKS, 1): (100.0, day + pd.Timedelta(days=1))}, gaps=paid_gaps)  # not yet due
    with pytest.raises(ht.HolderTargetError, match="other than by a scheduled redemption"):
        gap(lots=redeemed, matured={(BANKS, 1): (60.0, day)}, gaps=paid_gaps)  # less than the lot's principal
    with pytest.raises(ht.HolderTargetError, match="acquires holdings"):  # new issuance is still refused
        gap(lots={**today, (BANKS, 9): (False, 5.0, 5.0)}, gaps=quiet)

    # Before the first market-day close the opening holdings are the snapshot: the same checks
    # apply, and the intervention must not have started, with every gap within tolerance.
    opening = ht.MarketClose(lots=lots, reference_cpi=300.0, reference=None, gaps=None)
    small = {route: 0.5 * TOLERANCE for route in ht.PARTICIPANTS}
    with pytest.raises(ht.HolderTargetError, match="before any market-day close"):
        gap(gaps=small, market_close=opening)  # the hold has started
    pre_path = ht.holder_target_path({
        "mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED,
        "rows": [{"date": day.date().isoformat(), "phase": "pre", "holder_type": holder, "holder_subbucket": route,
                  "target_stock_bil": 100.0, "displacement_weight": weights.get((holder, route), 0.0)}
                 for holder, route in ht.PARTICIPANTS],
    })

    def pre_gap(**changes):
        return pre_path.closed_day_gap(day, **{"gaps": small, "lots": today, "matured": {},
                                               "reference_cpi": 300.0 * (1.0 + 1e-6), "market_close": opening,
                                               **changes})

    assert pre_gap() == pytest.approx(0.5 * TOLERANCE)
    with pytest.raises(ht.HolderTargetError, match="before any market-day close"):
        pre_gap(gaps=gaps)
    with pytest.raises(ht.HolderTargetError, match="acquires holdings"):
        pre_gap(lots={**today, (BANKS, 9): (False, 5.0, 5.0)})


def test_an_untraded_route_trades_only_what_a_debt_difference_brings_it() -> None:
    day = pd.Timestamp("2026-09-22")
    weights = {BANKS: -0.5, FOREIGN: -0.5, PRIVATE_MMF_ADDED: 1.0}
    path = ht.holder_target_path({
        "mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "buyer_private_route": PRIVATE_SUBBUCKET_MMF_ADDED,
        "rows": [{"date": day.date().isoformat(), "phase": "build", "holder_type": holder, "holder_subbucket": route,
                  "target_stock_bil": 100.0, "displacement_weight": weights.get((holder, route), 0.0)}
                 for holder, route in ht.PARTICIPANTS],
    })
    gaps = {route: 0.0 for route in ht.PARTICIPANTS}
    gaps[PRIVATE_MMF] = -0.2  # the ordinary funds hold 0.2 above their baseline
    with pytest.raises(ht.HolderTargetError, match="Untraded route"):
        path.correction_flows(gaps, TOLERANCE / 2.0, current_date=day)
    assert path.correction_flows({**gaps, PRIVATE_MMF: -TOLERANCE / 2.0}, 0.0, current_date=day) == []
    assert path.correction_flows(gaps, 0.6, current_date=day) == [
        (PRIVATE_MMF, BANKS, pytest.approx(0.1)), (PRIVATE_MMF, FOREIGN, pytest.approx(0.1))]
    gaps[PRIVATE_MMF] = 0.2  # short of it: it buys from the donors
    assert path.correction_flows(gaps, -0.6, current_date=day) == [
        (BANKS, PRIVATE_MMF, pytest.approx(0.1)), (FOREIGN, PRIVATE_MMF, pytest.approx(0.1))]


def test_a_debt_difference_before_the_build_or_with_no_donors_raises() -> None:
    day = pd.Timestamp("2026-09-22")
    donors = {BANKS: -0.5, FOREIGN: -0.5, PRIVATE_MMF_ADDED: 1.0}

    def path(phase, weights):
        return ht.holder_target_path({"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": [
            {"date": day.date().isoformat(), "phase": phase, "holder_type": holder, "holder_subbucket": route,
             "target_stock_bil": 100.0, "displacement_weight": weights.get((holder, route), 0.0)}
            for holder, route in ht.PARTICIPANTS
        ]})

    stocks = {route: 100.0 for route in ht.PARTICIPANTS}
    stocks[PRIVATE_DNB] += 0.5  # the public holds 0.5 more than the baseline
    with pytest.raises(ht.HolderTargetError, match="before the intervention starts"):
        path("pre", donors).for_date(day, stocks)
    with pytest.raises(ht.HolderTargetError, match="names no donors"):
        path("build", {}).for_date(day, stocks)
    _, targets = path("build", donors).for_date(day, stocks)
    assert targets[BANKS] == targets[FOREIGN] == pytest.approx(100.25)
    assert targets[PRIVATE_MMF_ADDED] == targets[PRIVATE_MMF] == targets[PRIVATE_DNB] == 100.0


# --- The controller on a hand-built book ---


def _book(rows=OPENING) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=BOND_PORTFOLIO_COLS).astype(PORTFOLIO_DTYPES, errors="ignore")


def _price(row, amount):
    face = float(row["FaceValue"])
    dirty = amount * 0.98
    return {"dirty_value": dirty, "clean_value": dirty, "accrued_interest": 0.0, "discount_yield": 0.04,
            "time_to_maturity": 0.25, "dirty_price_ratio": dirty / amount, "face": face}


def test_a_holder_above_target_sells_its_excess_rather_than_being_overwritten() -> None:
    book = _book()
    before = ht.holder_stocks(book)
    total_face = float(book["FaceValue"].sum())
    gaps = {BANKS: -10.0, FOREIGN: 4.0, PRIVATE_DNB: 6.0, PRIVATE_MMF: 0.0, PRIVATE_MMF_ADDED: 0.0}
    after_book, report = ht.rebalance_holder_positions(book, gaps, current_date="2026-09-25", price=_price)
    after = ht.holder_stocks(after_book)
    for participant, gap in gaps.items():
        assert after[participant] - before[participant] == pytest.approx(gap)
    assert report.sold_face[BANKS] == pytest.approx(10.0)
    assert report.pairs[(BANKS, FOREIGN)][0] == pytest.approx(4.0)
    assert report.pairs[(BANKS, PRIVATE_DNB)][0] == pytest.approx(6.0)
    # Only bills move; total face is conserved row by row.
    assert after_book["FaceValue"].sum() == pytest.approx(total_face)
    moved = after_book[after_book["BondID"] > 12]
    assert set(moved["MaturityCategory"]) == {"bills"}


def test_a_transferred_bill_redeems_to_its_new_holder_with_its_issue_discount(bank_case, baseline) -> None:
    # The money funds' bill maturing 2026-10-01 was partly sold to banks during the build.
    # At redemption the banks receive that face in reserves, and the money funds' deposits
    # receive that much less principal and original-issue discount than in the baseline.
    day, ratio = "2026-10-01", 0.99
    results, portfolio = bank_case[2], bank_case[3]
    base_results, base_portfolio = baseline[1], baseline[0]["_final_portfolio"]

    def matured(book, holder, route=""):
        rows = book[(book["MaturityDate"] == pd.Timestamp(day)) & (book["HolderType"] == holder)]
        if holder == "Private":
            rows = rows[rows["HolderSubBucket"] == route]
        return float(rows["FaceValue"].sum())

    moved = matured(portfolio, "Banks")
    assert moved > 1.0 and matured(base_portfolio, "Banks") == 0.0
    assert matured(portfolio, "CB") == pytest.approx(matured(base_portfolio, "CB"), abs=1e-12)
    assert matured(base_portfolio, "Private", MMF) - matured(portfolio, "Private", MMF) == pytest.approx(moved, rel=1e-12)
    bought = portfolio[(portfolio["MaturityDate"] == pd.Timestamp(day)) & (portfolio["HolderType"] == "Banks")]
    assert set(bought["TDCPrincipalHolderType"]) == {"Banks"}
    assert (bought["IssueProceeds"] / bought["FaceValue"]).to_numpy() == pytest.approx(ratio)
    principal = results.loc[day, "TDC_PrincipalToDU_MMF"] - base_results.loc[day, "TDC_PrincipalToDU_MMF"]
    discount = results.loc[day, "TDC_BillDiscountInterestToDU_MMF"] - base_results.loc[day, "TDC_BillDiscountInterestToDU_MMF"]
    assert principal == pytest.approx(-moved * ratio, rel=1e-9)
    assert discount == pytest.approx(-moved * (1.0 - ratio), rel=1e-9)
    assert results.loc[day, "PrincipalPaid_Bonds"] == pytest.approx(base_results.loc[day, "PrincipalPaid_Bonds"], rel=1e-12)


def test_redemption_follows_the_new_holder_and_keeps_its_original_issue_discount() -> None:
    book = _book()
    gaps = {BANKS: 30.0, FOREIGN: 0.0, PRIVATE_DNB: -30.0, PRIVATE_MMF: 0.0, PRIVATE_MMF_ADDED: 0.0}
    after, _ = ht.rebalance_holder_positions(book, gaps, current_date="2026-09-25", price=_price)
    bought = after[(after["HolderType"] == "Banks") & (after["MaturityCategory"] == "bills") & ~after["BondID"].isin([1, 2])]
    assert not bought.empty
    assert set(bought["TDCPrincipalHolderType"]) == {"Banks"}
    assert set(bought["TDCPrincipalHolderSubBucket"].fillna("")) == {""}
    assert (bought["IssueProceeds"] / bought["FaceValue"]).to_numpy() == pytest.approx(0.99)
    remaining = after[after["BondID"].isin([7, 8, 9])]
    assert (remaining["IssueProceeds"] / remaining["FaceValue"]).to_numpy() == pytest.approx(0.99)


def test_a_seller_without_enough_bills_raises_with_route_date_and_shortfall() -> None:
    gaps = {BANKS: 0.0, FOREIGN: 0.0, PRIVATE_DNB: 300.0, PRIVATE_MMF: -300.0, PRIVATE_MMF_ADDED: 0.0}
    with pytest.raises(ht.HolderTargetError, match=r"PrivateMMF holds 180\.000000 billion of eligible bills on "
                       r"2026-09-25, short of the 300\.000000 billion its sale requires \(shortfall 120\.000000\)"):
        ht.rebalance_holder_positions(_book(), gaps, current_date="2026-09-25", price=_price)


def test_bill_stock_control_assigns_the_realized_change_and_settles_at_dirty_prices(tmp_path) -> None:
    params = _params(tmp_path / "inputs", fed=False)
    treatment = copy.deepcopy(params)
    params["treasury_issuance_profile"]["bills"]["target_percentage_of_remainder"] = 0.60
    params["treasury_issuance_profile"]["notes"]["target_percentage_of_remainder"] = 0.40
    params["treasury_issuance_profile"]["remainder_maturity_years"] = 2.0
    base, _, _ = _run(params, tmp_path / "base")
    bills = ht.STOCK_COLUMNS["HolderBillStock"]
    weights = {BANKS: 0.30, FOREIGN: 0.70}
    treatment["holder_position_target"] = {
        "mode": ht.BILL_TARGET_MODE,
        "tolerance_bil": TOLERANCE,
        "bill_change_weights": [
            {"holder_type": holder, "holder_subbucket": route, "weight": weight}
            for (holder, route), weight in weights.items()
        ],
        "rows": [
            {"date": day, "holder_type": holder, "holder_subbucket": route, "baseline_stock_bil": row[column]}
            for day, row in base.iterrows()
            for (holder, route), column in bills.items()
        ],
    }
    result, _, _ = _run(treatment, tmp_path / "treatment")
    # Bill-stock targets assign the realized change by their own weights; the stock-target
    # debt-difference rule and its column do not apply.
    assert not set(ht.STOCK_TARGET_COLUMNS) & set(result.columns)
    difference = result[list(bills.values())] - base[list(bills.values())]
    total = difference.sum(axis=1)
    assert total.abs().max() >= 10.0
    for participant, column in bills.items():
        assert difference[column].to_numpy() == pytest.approx(total.to_numpy() * weights.get(participant, 0), abs=TOLERANCE)
    traded = result[result["HolderTargetTradeFace"] > 0]
    assert not traded.empty
    assert (traded["HolderTargetTradeCash"] < traded["HolderTargetTradeFace"]).all()
    deposit_cash = sum(
        traded[f"HolderTargetSoldCash_{label}"] - traded[f"HolderTargetBoughtCash_{label}"]
        for label in ("PrivateDomesticNonbank", "PrivateMMF")
    )
    assert traded["HolderTargetDepositEffect"].to_numpy() == pytest.approx(deposit_cash.to_numpy())


def test_targets_that_do_not_sum_to_the_public_stock_raise() -> None:
    stocks = ht.holder_stocks(_book())
    targets = {participant: value for participant, value in stocks.items()}
    targets[BANKS] += 1.0
    with pytest.raises(ht.HolderTargetError, match="debt discrepancy"):
        ht.target_gaps(stocks, targets, tolerance=TOLERANCE, current_date="2026-09-25")


def test_sales_spread_over_remaining_maturity_buckets_in_proportion_to_holdings() -> None:
    # Foreign holds 80 maturing within a month and 70 beyond six months: a sale of 15
    # takes 8 and 7, not the shortest bills first.
    gaps = {BANKS: 15.0, FOREIGN: -15.0, PRIVATE_DNB: 0.0, PRIVATE_MMF: 0.0, PRIVATE_MMF_ADDED: 0.0}
    after, _ = ht.rebalance_holder_positions(_book(), gaps, current_date="2026-09-25", price=_price)
    bought = after[(after["HolderType"] == "Banks") & ~after["BondID"].isin([1, 2, 3])]
    by_maturity = bought.groupby("MaturityDate")["FaceValue"].sum()
    assert by_maturity[pd.Timestamp("2026-10-05")] == pytest.approx(8.0)
    assert by_maturity[pd.Timestamp("2027-03-01")] == pytest.approx(7.0)


def test_splitting_or_reordering_identical_lots_leaves_the_outcome_unchanged() -> None:
    gaps = {BANKS: 25.0, FOREIGN: -5.0, PRIVATE_DNB: -12.0, PRIVATE_MMF: -8.0, PRIVATE_MMF_ADDED: 0.0}
    one, one_report = ht.rebalance_holder_positions(_book(), gaps, current_date="2026-09-25", price=_price)
    split = [dict(row) for row in OPENING]
    halves = []
    for row in split:
        if row["BondID"] in {8, 11}:
            half = dict(row)
            for column in ("FaceValue", "OriginalPrincipal", "AdjustedPrincipal", "IssueProceeds"):
                row[column] = row[column] / 2.0
                half[column] = row[column]
            half["BondID"] = 100 + row["BondID"]
            halves.append(half)
    shuffled = list(reversed(split + halves))
    two, two_report = ht.rebalance_holder_positions(_book(shuffled), gaps, current_date="2026-09-25", price=_price)

    def holdings(book):
        active = book[book["Status"] == "Active"]
        return active.groupby(["HolderType", "HolderSubBucket", "MaturityDate"], dropna=False)[
            ["FaceValue", "IssueProceeds"]
        ].sum()

    pd.testing.assert_frame_equal(holdings(one), holdings(two), check_exact=False, rtol=1e-12)
    assert one_report.trade_cash == pytest.approx(two_report.trade_cash, rel=1e-12)


# --- Through the family writer, runner, bounded verifier and export ---


def test_a_written_holder_target_family_runs_verifies_and_exports(tmp_path, monkeypatch) -> None:
    import sys

    import tdcsim_cbo.runner as runner_module
    from tdcsim_cbo import CboScenarioSpec, run_cbo_scenario
    from tdcsim_cbo.verifier import verify_scenario_run
    from test_tdcsim_cbo_closeout_interface import _write_runner_package

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from export_position_targets import _case, _load
    from write_scenario_family import FAMILY_SPEC_SCHEMA, family_scenarios

    from tdcsim_cbo import CboBaselinePackage

    monkeypatch.setattr(runner_module, "host_available_memory_bytes", lambda: 1 << 50)
    opening = _book([row if row["BondID"] != 6 else {**row, "FaceValue": 120.0, "OriginalPrincipal": 120.0,
                                                      "AdjustedPrincipal": 120.0, "IssueProceeds": 120.0}
                     for row in OPENING])
    assert float(opening["FaceValue"].sum()) == 1_000.0
    package, attestation = _write_runner_package(tmp_path / "package", opening_portfolio=opening)
    baseline = CboBaselinePackage.open(package, attestation_path=attestation)
    family = {
        "schema_version": FAMILY_SPEC_SCHEMA, "family_id": "positions_fixture", "as_of_date": "2026-11-21",
        "opening_mmf_added_on_rrp_bil": 20.0,
        "simulation": {"start_date": "2026-09-20", "end_date": "2026-09-30"},
        "scenarios": [
            {"role": "baseline", "scenario_id": "positions_fixture_baseline", "label": "Baseline",
             "mmf_deposit_pass_through": 1.0},
            {"role": "banks", "scenario_id": "positions_fixture_banks", "label": "Banks", "mmf_deposit_pass_through": 1.0,
             "holder_position_target": {
                 "buyer_holder": "Banks", "buyer_private_route": "", "size_bil": 20.0, "origin": "2026-09-23",
                 "build_end": "2026-09-26", "donors": [["Foreign", ""], ["Private", DNB], ["Private", MMF]],
                 "unchanged_routes": [list(PRIVATE_MMF_ADDED)]}},
        ],
    }
    mmf = copy.deepcopy(family["scenarios"][1])
    mmf.update(role="mmf_onrrp", scenario_id="positions_fixture_mmf_onrrp", label="Added money-fund position",
               mmf_added_position_settlement="on_rrp_mmf")
    mmf["holder_position_target"].update(
        buyer_holder="Private", buyer_private_route=PRIVATE_SUBBUCKET_MMF_ADDED, donors=list(map(list, MMF_DONORS)),
        unchanged_routes=[list(PRIVATE_MMF)],
    )
    family["scenarios"].append(mmf)
    scenario_dir = tmp_path / "scenarios"
    scenario_dir.mkdir()
    routing = tmp_path / "routing.json"
    routing.write_text(json.dumps(_routing()))
    root = tmp_path / "runs"

    def run(items: dict) -> None:
        for name, data in items.items():
            role = name.split("_", 1)[1].removesuffix(".json")
            path = scenario_dir / name
            path.write_text(json.dumps(data))
            result = run_cbo_scenario(baseline, CboScenarioSpec.from_file(path), root / role / "run",
                                      trc_routing_path=routing)
            verification = verify_scenario_run(result.output_dir, baseline_package=package, attestation=attestation)
            assert verification["status"] == "pass", role
            (root / role / "verification.json").write_text(json.dumps(verification, default=str))

    quantity = family_scenarios(baseline, family)
    assert list(quantity) == ["00_baseline.json"]
    assert quantity["00_baseline.json"]["simulation"] == {"frequency": "daily", **family["simulation"]}
    run(quantity)
    positions = family_scenarios(baseline, family, phase="positions", runs_root=root, output_dir=scenario_dir)
    override = positions["01_banks.json"]["overrides"]["holder_position_target"]
    assert override["baseline_reference"]["scenario_id"] == "positions_fixture_baseline"
    assert sum(d["weight"] for d in override["displacement"]) == pytest.approx(-1.0)
    run(positions)

    banks = _load(root / "banks" / "run")["results"]
    assert banks["HolderTargetMaxError"].max() <= TOLERANCE
    assert ht.DEBT_DIFFERENCE_COLUMN in banks.columns
    assert banks.loc["2026-09-26":, "HolderTarget_Banks"].iloc[0] > 0.0
    assert "HolderTargetPhase" not in _load(root / "baseline" / "run")["results"].columns
    case = _case(family, root, "banks", _load(root / "baseline" / "run"), ht)
    increments = {(p["holder_type"], p["holder_subbucket"]): p for p in case["positions"]}
    assert increments[BANKS]["realized_increment_end_bil"] == pytest.approx(20.0, abs=TOLERANCE)
    assert sum(p["realized_increment_end_bil"] for p in case["positions"]) == pytest.approx(0.0, abs=4 * TOLERANCE)
    components = {(c["date"], c["component"]): c["scenario_minus_baseline_bil"] for c in case["components"]
                  if c["object"] == "tdc"}
    assert abs(components[("2026-09-30", "reconciliation_residual")]) <= TOLERANCE
    # Banks pay with reserves, so deposit-funded donors' deposits rise by what they receive.
    assert components[("2026-09-30", "secondary_settlements_target_trades")] > 0.0
    effects = {(e["date"], e["measure"]): e for e in case["effects"]}
    assert effects[("2026-09-30", "public_marketable_stock")]["scenario_minus_baseline_bil"] == pytest.approx(
        banks.loc["2026-09-30", ht.DEBT_DIFFERENCE_COLUMN], abs=TOLERANCE)
    for position in case["positions"]:
        assert position["realized_increment_end_bil"] == pytest.approx(
            position["target_increment_end_bil"] + position["debt_difference_held_end_bil"], abs=TOLERANCE)
    assert sum(p["debt_difference_share"] for p in case["positions"]) == pytest.approx(1.0)
    assert case["checks"]["max_abs_debt_difference_bil"] == pytest.approx(banks[ht.DEBT_DIFFERENCE_COLUMN].abs().max())
    mmf_case = _case(family, root, "mmf_onrrp", _load(root / "baseline" / "run"), ht)
    assert any(c["component"] == "secondary_settlements_target_trades_added_mmf_position"
               and c["scenario_minus_baseline_bil"] == 0.0 for c in mmf_case["components"])
    issuance = pd.read_csv(root / "mmf_onrrp/run/outputs/tdcsim_period_issuance_aggregates.csv.gz")
    assert PRIVATE_SUBBUCKET_MMF_ADDED not in set(issuance["holder_subsector"])

    # The family export carries each stock-target role's D; the baseline's columns stay empty.
    from tdcsim_cbo.scenario_family_export import export_scenario_family

    runs = {item["role"]: {"run_dir": root / item["role"] / "run", "scenario_id": item["scenario_id"],
                           "scenario_sha256": "0" * 64,
                           "verification": json.loads((root / item["role"] / "verification.json").read_text())}
            for item in family["scenarios"]}
    export_scenario_family(family_spec=family, family_spec_sha256="1" * 64, runs=runs, output_dir=tmp_path / "export",
                           producer={"repository_commit": "a" * 40})
    exported = pd.read_csv(tmp_path / "export" / "scenario_family_fy.csv", keep_default_na=False)
    for row in exported.itertuples():
        if row.scenario_role == "baseline":
            assert row.holder_target_debt_difference_end_bil == row.holder_target_debt_difference_max_abs_bil == ""
            continue
        recorded = _load(root / row.scenario_role / "run")["results"][ht.DEBT_DIFFERENCE_COLUMN]
        window = recorded[(recorded.index > row.coverage_start_exclusive) & (recorded.index <= row.coverage_end_inclusive)]
        # The export parses results with pandas' default float reader, so the last digit may differ.
        assert float(row.holder_target_debt_difference_end_bil) == pytest.approx(
            recorded[row.coverage_end_inclusive[:10]], rel=1e-12, abs=1e-15)
        assert float(row.holder_target_debt_difference_max_abs_bil) == pytest.approx(
            window.abs().max(), rel=1e-12, abs=1e-15)
    with pytest.raises(SystemExit, match="baseline reference"):
        (root / "baseline" / "run" / "tdcsim_cbo_run_manifest.json").write_text("{}")
        _case(family, root, "banks", _load(root / "baseline" / "run"), ht)


def test_shipped_buyer_family_has_one_baseline_and_identical_mmf_positions() -> None:
    spec = json.loads((Path(__file__).resolve().parents[1] / "scenarios" / "buyer_shift_v2.json").read_text())
    cases = {item["role"]: item for item in spec["scenarios"]}
    assert list(cases) == ["baseline", "banks", "foreign", "mmf_deposits", "mmf_onrrp"]
    deposits = cases["mmf_deposits"]["holder_position_target"]
    assert deposits == cases["mmf_onrrp"]["holder_position_target"]
    assert deposits["size_bil"] == spec["opening_mmf_added_on_rrp_bil"] == 500.0
    assert tuple(deposits["donors"]) == tuple(map(list, MMF_DONORS))
    assert deposits["unchanged_routes"] == [list(PRIVATE_MMF)]
    assert all(case["mmf_deposit_pass_through"] == 1.0 for case in cases.values())


# --- Declarations the contract and compiler refuse ---


def _declaration(**changes) -> dict:
    override = {
        "buyer_holder": "Banks", "buyer_private_route": "",
        "target_path": [{"date": ORIGIN, "increment_bil": 0.0}, {"date": BUILD_END, "increment_bil": 60.0},
                        {"date": END, "increment_bil": 60.0}],
        "displacement": [
            {"holder_type": "Foreign", "holder_subbucket": "", "origin_bill_stock_bil": 150.0, "weight": -0.3},
            {"holder_type": "Private", "holder_subbucket": DNB, "origin_bill_stock_bil": 200.0, "weight": -0.4},
            {"holder_type": "Private", "holder_subbucket": MMF, "origin_bill_stock_bil": 150.0, "weight": -0.3},
        ],
        "unchanged_routes": [dict(zip(("holder_type", "holder_subbucket"), PRIVATE_MMF_ADDED))],
    }
    override.update(changes)
    return override


def test_declared_weights_must_be_each_donors_share_of_origin_bills() -> None:
    weights = ht.declared_weights(_declaration())
    assert weights == {BANKS: 1.0, FOREIGN: -0.3, PRIVATE_DNB: -0.4, PRIVATE_MMF: -0.3, PRIVATE_MMF_ADDED: 0.0}
    skewed = _declaration()
    skewed["displacement"][0]["weight"] = -0.35
    with pytest.raises(ht.HolderTargetError, match="share of the donors' origin bills"):
        ht.declared_weights(skewed)
    with pytest.raises(ht.HolderTargetError, match="every public holder route once"):
        ht.declared_weights(_declaration(unchanged_routes=[{"holder_type": "Private", "holder_subbucket": DNB}]))


def test_the_contract_requires_a_simulation_that_ends_with_the_hold() -> None:
    from tdcsim_cbo.contract import _validate_holder_position_target

    simulation = {"frequency": "daily", "start_date": START, "end_date": END}
    _validate_holder_position_target({"simulation": simulation}, _declaration())
    with pytest.raises(ValueError, match="ends on the last hold date"):
        _validate_holder_position_target({"simulation": {**simulation, "end_date": "2026-10-11"}}, _declaration())
    with pytest.raises(ValueError, match="start at zero"):
        _validate_holder_position_target(
            {"simulation": simulation},
            _declaration(target_path=[{"date": ORIGIN, "increment_bil": 5.0}, {"date": BUILD_END, "increment_bil": 60.0},
                                      {"date": END, "increment_bil": 60.0}]),
        )


def test_a_target_file_that_departs_from_its_declaration_is_refused(baseline) -> None:
    rows, weights = _targets(baseline[2], buyer=BANKS, donors=BANK_DONORS, size=60.0)
    path = ht.BuildAndHoldPath.from_points(_declaration()["target_path"])
    ht.check_target_rows(rows, path=path, weights=weights)
    tampered = [dict(row) for row in rows]
    last = tampered[-4]
    last["target_increment_bil"] += 0.5
    last["target_stock_bil"] += 0.5
    with pytest.raises(ht.HolderTargetError, match="increment"):
        ht.check_target_rows(tampered, path=path, weights=weights)


# --- Representation: the Federal Reserve's purchases, then the targets ---

# Both Private routes hold bills of the same two maturities, so the Fed's earliest-maturity
# purchases from Private meet equal maturities in two routes, and the controller then
# restores every route's holdings at maturity-dependent prices.
TIED = [
    _security(1, "Private", DNB, "bills", "2026-10-05", 10.0),
    _security(2, "Private", DNB, "bills", "2027-06-01", 10.0),
    _security(3, "Private", MMF, "bills", "2026-10-05", 10.0),
    _security(4, "Private", MMF, "bills", "2027-06-01", 10.0),
    _security(5, "Banks", "", "bills", "2027-06-01", 20.0),
    _security(6, "Foreign", "", "bills", "2027-06-01", 20.0),
]


def _split_first_lot(book: list[dict]) -> list[dict]:
    first, *rest = [dict(row) for row in book]
    half = dict(first)
    for column in ("FaceValue", "OriginalPrincipal", "AdjustedPrincipal", "IssueProceeds"):
        first[column] = half[column] = first[column] / 2.0
    half["BondID"] = 7
    return [first, *rest, half]


@pytest.mark.parametrize("representation", ["split", "reversed"])
def test_fed_and_holder_target_are_representation_invariant(tmp_path, representation) -> None:
    def paired(name: str, book: list[dict]):
        root = tmp_path / name
        options = {"book": book, "fed_targets": {2026: 7.5, 2027: 40.0}}
        base_results, _, closure = _run(_params(root / "base_inputs", **options), root / "base")
        rows, _ = _targets(closure, buyer=BANKS, donors=BANK_DONORS, size=5.0)
        params = _params(root / "inputs", **options)
        params["holder_position_target"] = {"mode": ht.HOLDER_TARGET_MODE, "tolerance_bil": TOLERANCE, "rows": rows}
        results, portfolio, _ = _run(params, root / "run")
        trc, base_trc = results.attrs["trc_results"], base_results.attrs["trc_results"]
        active = portfolio[portfolio["Status"] == "Active"]
        holdings = active.groupby(["HolderType", "HolderSubBucket", "SecurityType", "MaturityDate"], dropna=False)[
            ["FaceValue", "IssueProceeds"]].sum()
        return {
            "tdc": float(results["TDC_Change"].sum() - base_results["TDC_Change"].sum()),
            "trc": float(trc["TRC_Change"].sum() - base_trc["TRC_Change"].sum()),
            "fed_cash": float(results["CBOFedSecondaryPurchaseCash"].sum() - base_results["CBOFedSecondaryPurchaseCash"].sum()),
            "target_cash": float(results["HolderTargetTradeCash"].sum()),
            "routes": results[[f"TDC_SecondaryTrades_{r}" for r in ("DomesticNonbank", "MMF")]].sum().to_numpy(),
            "holdings": holdings,
        }

    original = paired("original", TIED)
    other = paired(representation, _split_first_lot(TIED) if representation == "split" else list(reversed(TIED)))
    assert original["fed_cash"] != 0.0 or original["target_cash"] > 0.0
    for key in ("tdc", "trc", "fed_cash", "target_cash"):
        assert other[key] == pytest.approx(original[key], abs=1e-12), key
    assert other["routes"] == pytest.approx(original["routes"], abs=1e-12)
    pd.testing.assert_frame_equal(other["holdings"], original["holdings"], check_exact=False, rtol=1e-12, atol=1e-12)
