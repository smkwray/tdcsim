"""Release opening dates and Treasury cash-table selectors for the CBO package."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from forecast_bundle_builders import (
    MTS_TABLE_1_DEFICIT_SELECTOR,
    MTS_TABLE_9_NET_INTEREST_SELECTOR,
    build_current_fy_splice_row,
)

from issuance_calibration import prestart_rollforward, prestart_settlements
from ratewall_input_builder import load_soma_cusip_holdings
from tips_indexation import enrich_opening_tips_from_cpi_detail, load_tips_cpi_detail, opening_tips_to_par

SIMULATION_START_DATE = "2026-06-21"
OPENING_CLOSE_DATE = "2026-06-18"
OPENING_PORTFOLIO_RECORD_DATE = "2026-05-31"
ACTUALS_AVAILABLE_AS_OF = "2026-06-22"
SOMA_SNAPSHOT_DATE = "2026-06-17"
MTS_NET_INTEREST_SELECTOR = f"{MTS_TABLE_9_NET_INTEREST_SELECTOR};column=current_fytd_rcpt_outly_amt"

DTS_MONTH_OPEN = OPENING_CLOSE_DATE[:8] + "01"
DTS_CASH_SOURCE = f"fiscaldata/dts_operating_cash_balance_{DTS_MONTH_OPEN}_{OPENING_CLOSE_DATE}.csv"
DTS_FLOWS_SOURCE = f"fiscaldata/dts_deposits_withdrawals_operating_cash_{OPENING_CLOSE_DATE}.csv"
SOMA_SOURCE = f"opening/nyfed_soma_treasury_holdings_{SOMA_SNAPSHOT_DATE}.csv"
DTS_TGA_OPENING_ACCOUNT = "Treasury General Account (TGA) Opening Balance"
DTS_TGA_CLOSING_ACCOUNT = "Treasury General Account (TGA) Closing Balance"
DTS_TGA_ACCOUNT = "Treasury General Account (TGA)"
DTS_FISCAL_LINES = {
    "deposits": ("Deposits", "Treasury General Account Total Deposits", "null"),
    "withdrawals": ("Withdrawals", "Treasury General Account Total Withdrawals", "null"),
    "debt_cash_issues": ("Deposits", DTS_TGA_ACCOUNT, "Public Debt Cash Issues (Table IIIB)"),
    "debt_cash_redemptions": ("Withdrawals", DTS_TGA_ACCOUNT, "Public Debt Cash Redemp. (Table IIIB)"),
    "treasury_interest_paid": ("Withdrawals", DTS_TGA_ACCOUNT, "Interest on Treasury Securities"),
}
# Two stock observations and two MTD flow totals independently round to $1 million.
TGA_IDENTITY_TOLERANCE_MILLIONS = 2
OPENING_PORTFOLIO_CLAIM_BOUNDARY = (
    "prebuilt_mspd_table3_market_derived_cohort_book; "
    "cb_from_nyfed_soma_by_cusip_noncb_split_by_wamest_sector_bill_nonbill_holdings; "
    "june_close_from_actual_auction_accepted_face_less_paid_maturities; "
    "fed_positions_from_june_soma_cusips_and_subsequent_auction_flows; "
    "nonfed_security_allocations_are_modeled"
)
OPENING_ROLLFORWARD_STATUS = "actual_settlement_face_reconciled_through_opening_close"


def dts_fiscal_selectors() -> dict[str, str]:
    return {
        f"{key}_bil": f"transaction_type={kind};account_type={account};transaction_catg={category};column=transaction_mtd_amt"
        for key, (kind, account, category) in DTS_FISCAL_LINES.items()
    }


def _read_dts_opening_cash(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = {}
    for day, account in ((DTS_MONTH_OPEN, DTS_TGA_OPENING_ACCOUNT), (OPENING_CLOSE_DATE, DTS_TGA_CLOSING_ACCOUNT)):
        found = [row for row in rows if row["record_date"] == day and row["account_type"] == account]
        if len(found) != 1:
            raise ValueError(f"DTS table I needs exactly one {day} {account} row, found {len(found)}")
        matches[day] = found[0]
    return matches[DTS_MONTH_OPEN], matches[OPENING_CLOSE_DATE]


def _read_dts_opening_flows(path: Path) -> dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["record_date"] == OPENING_CLOSE_DATE]
    flows = {}
    for key, selector in DTS_FISCAL_LINES.items():
        matched = [
            row for row in rows
            if (row["transaction_type"], row["account_type"], row["transaction_catg"] or "null") == selector
        ]
        if len(matched) != 1:
            raise ValueError(f"DTS table II needs exactly one {OPENING_CLOSE_DATE} {selector} row")
        flows[key] = int(matched[0]["transaction_mtd_amt"])
    return flows


def opening_cash_reconciliation(cash: Path, flows: Path) -> dict[str, float | str]:
    """Use Table II month-to-date totals at the common close, in USD billions."""
    opening, closing = _read_dts_opening_cash(cash)
    amounts = _read_dts_opening_flows(flows)
    tga_open = int(opening["open_today_bal"])
    tga_close = int(closing["open_today_bal"])
    net_debt = amounts["debt_cash_issues"] - amounts["debt_cash_redemptions"]
    deficit = amounts["withdrawals"] - amounts["deposits"] + net_debt
    primary = deficit - amounts["treasury_interest_paid"]
    residual = tga_close - tga_open - net_debt + primary + amounts["treasury_interest_paid"]
    if abs(residual) > TGA_IDENTITY_TOLERANCE_MILLIONS:
        raise ValueError(f"DTS June TGA identity residual is {residual} million, exceeding "
                         f"{TGA_IDENTITY_TOLERANCE_MILLIONS} million rounding tolerance")
    return {
        **{f"{key}_bil": amount / 1000.0 for key, amount in amounts.items()},
        "opening_close_date": OPENING_CLOSE_DATE,
        "tga_june_open_bil": tga_open / 1000.0,
        "tga_close_bil": tga_close / 1000.0,
        "total_deficit_bil": deficit / 1000.0,
        "primary_deficit_bil": primary / 1000.0,
        "tga_identity_residual_bil": residual / 1000.0,
        "tga_identity_tolerance_bil": TGA_IDENTITY_TOLERANCE_MILLIONS / 1000.0,
    }


def build_opening_fy_splice(
    *, scenario_id: str, cbo_primary_deficit_bil: float, mts_deficit_bil: float, mts_interest_bil: float,
    dts: Mapping[str, float | str],
) -> dict[str, Any]:
    if dts["opening_close_date"] != OPENING_CLOSE_DATE:
        raise ValueError("the DTS splice cutoff differs from the opening book close")
    return build_current_fy_splice_row(
        scenario_id=scenario_id,
        fiscal_year=2026,
        simulation_start_date=SIMULATION_START_DATE,
        fiscal_actuals_through=OPENING_CLOSE_DATE,
        actuals_available_as_of=ACTUALS_AVAILABLE_AS_OF,
        cbo_full_fy_primary_deficit_bil=cbo_primary_deficit_bil,
        actual_total_deficit_fytd_bil=mts_deficit_bil + float(dts["total_deficit_bil"]),
        actual_net_interest_fytd_bil=mts_interest_bil + float(dts["treasury_interest_paid_bil"]),
        mts_table_1_selector=MTS_TABLE_1_DEFICIT_SELECTOR,
        mts_table_9_selector=MTS_NET_INTEREST_SELECTOR,
        observation_date=OPENING_CLOSE_DATE,
        available_date=ACTUALS_AVAILABLE_AS_OF,
        source_status="may_mts_fytd_plus_june_dts_cash_proxy_spliced_to_cbo_full_year_primary_deficit",
        claim_boundary="june_stub_is_a_dts_cash_proxy_tga_residual_tests_only_the_cash_identity",
    )

@dataclass(frozen=True)
class OpeningMarketables:
    book: pd.DataFrame
    summary: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    tips_metadata: dict[str, Any]
    tips_diagnostics: list[dict[str, Any]]


def roll_opening_marketables(
    book: pd.DataFrame, *, cusip_column: str, noncb_weights: Mapping[str, Mapping[str, float]],
    auctions_query: Path, soma_holdings: Path, tips_cpi_detail: Path,
) -> OpeningMarketables:
    """Reconcile contractual par first, then value the rolled book on the June close."""
    has_tips = book["SecurityType"].eq("TIPS").any()
    detail = load_tips_cpi_detail(tips_cpi_detail) if has_tips else None
    par_book, ratio_sources = opening_tips_to_par(
        book, detail=detail, record_date=OPENING_PORTFOLIO_RECORD_DATE,
        auctions_path=auctions_query, cusip_column=cusip_column,
    )
    settlements = prestart_settlements(
        auctions_query, record_date=OPENING_PORTFOLIO_RECORD_DATE, opening_close_date=OPENING_CLOSE_DATE,
    )
    soma_date, soma = load_soma_cusip_holdings(soma_holdings)
    if soma_date != SOMA_SNAPSHOT_DATE:
        raise ValueError(f"opening SOMA snapshot must be dated {SOMA_SNAPSHOT_DATE}, found {soma_date}")
    rolled, summary, diagnostics = prestart_rollforward(
        par_book, settlements, opening_close_date=OPENING_CLOSE_DATE, record_date=OPENING_PORTFOLIO_RECORD_DATE,
        cusip_column=cusip_column, noncb_weights=noncb_weights,
        soma=soma.assign(amount_mil=soma["par_mil"]), soma_as_of_date=soma_date,
    )
    valued, tips_metadata, tips_diagnostics = enrich_opening_tips_from_cpi_detail(
        rolled, cpi_detail_path=tips_cpi_detail, index_date=OPENING_CLOSE_DATE, detail=detail, cusip_column=cusip_column,
    )
    source_tips = par_book.loc[par_book["SecurityType"].eq("TIPS")]
    close_tips = valued.loc[valued["SecurityType"].eq("TIPS")]
    # No indexed issues settle in the declared window. Preserve TIPS par by
    # CUSIP and show valuation changes separately from net nominal borrowing.
    source_live = source_tips.loc[source_tips["MaturityDate"].gt(pd.Timestamp(OPENING_CLOSE_DATE))]
    source_live_par = source_live.groupby(cusip_column)["FaceValue"].sum()
    close_tips_par = close_tips.groupby(cusip_column)["FaceValue"].sum()
    if not source_live_par.index.equals(close_tips_par.index) or (source_live_par - close_tips_par).abs().max() > 1e-8:
        raise ValueError("opening roll changed outstanding TIPS par by CUSIP")
    source_indexation = float((source_tips["AdjustedPrincipal"] - source_tips["FaceValue"]).sum())
    close_indexation = float((close_tips["AdjustedPrincipal"] - close_tips["FaceValue"]).sum())
    fed_tips = close_tips.loc[close_tips["HolderType"].eq("CB")]
    soma_indexation = float((fed_tips["AdjustedPrincipal"] - fed_tips["FaceValue"]).sum())
    summary.update(
        source_tips_indexation_bil=source_indexation,
        opening_tips_indexation_bil=close_indexation,
        source_debt_base_bil=summary["source_face_bil"] + source_indexation,
        opening_debt_base_bil=summary["opening_face_bil"] + close_indexation,
        soma_snapshot_indexation_bil=float((soma["amount_mil"] - soma["par_mil"]).sum()) / 1000,
        soma_close_indexation_bil=soma_indexation,
        soma_close_debt_base_bil=summary["soma_close_bil"] + soma_indexation,
    )
    for row in tips_diagnostics:
        row["par_source_date"] = OPENING_PORTFOLIO_RECORD_DATE
        row["par_source_method"] = ratio_sources[row["cusip"]]
    return OpeningMarketables(valued, summary, diagnostics, tips_metadata, tips_diagnostics)
