"""Output writers for compiled CBO scenario runs."""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ._json import sha256_file, write_json


SUMMARY_COLUMNS = [
    "Date",
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBOControlledDebtPreIssuance",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
    "NewDebtIssued",
    "AuctionProceeds",
    "PrimaryDeficit",
    "TGA",
    "CBOOperatingCashTarget",
    "CBOCashReconciliationResidual",
    "CBOCashResidualStatus",
    "CBOFedHoldingsTarget",
    "CBOFedHoldingsTargetError",
    "CBOFedAuctionShare",
    "CBOFedAuctionRolloverAddons",
    "CBOFedSecondaryPurchaseFace",
    "CBOFedSecondaryPurchaseCash",
    "CBOFedSecondaryPurchaseReserveEffect",
    "CBOFedSecondaryPurchaseDepositEffect",
    "CBOFedSyntheticSecondaryPurchases",
    "CBOFedSyntheticSecondarySales",
    "CBORemittanceCashEffect",
    "CBORemittanceStatus",
    "CB_Remittance",
    "CB_DeferredAsset",
    "CBONetInterestDiagnostic",
    "CBOTotalDeficitDiagnostic",
    "CBONetInterestBridgeRows",
    "NetInterestDiagnosticStatus",
    "DebtHeld_Banks",
    "DebtHeld_CB",
    "DebtHeld_Foreign",
    "DebtHeld_Private",
    "DebtHeldByType_Fixed",
    "DebtHeldByType_TIPS",
    "DebtHeldByType_FRN",
    "NewIssuanceWAM",
    "NewIssuanceBillShare",
    "NewIssuanceShortMaturityShare",
    "OutstandingControlledWAM",
    "OutstandingControlledBillShare",
    "OutstandingControlledShortMaturityShare",
]

COMMON_METADATA_COLUMNS = [
    "schema_version",
    "scenario_id",
    "run_id",
    "package_id",
    "source_vintage",
    "actuals_available_as_of",
    "scenario_config_sha256",
    "compiled_inputs_digest",
    "mmf_deposit_pass_through",
    "mmf_deposit_pass_through_status",
    "fiscal_incidence_policy_id",
    "fiscal_incidence_basis",
    "fiscal_incidence_du_share",
    "fiscal_incidence_ru_share",
    "fiscal_incidence_foreign_share",
    "fiscal_incidence_other_share",
]

HANDOFF_TABLE_COLUMNS = {
    "tdcsim_period_issuance_flows": [
        "period_start",
        "period_end",
        "flow_id",
        "security_id",
        "holder_sector",
        "holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "weighted_original_term_years",
        "face_issued_bil",
        "cash_proceeds_bil",
        "discount_or_premium_bil",
        "coupon_rate_decimal",
        "reference_rate_decimal",
        "spread_bps",
        "issue_yield_decimal",
    ],
    "tdcsim_period_principal_flows": [
        "period_start",
        "period_end",
        "flow_id",
        "security_id",
        "holder_sector",
        "holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "redemption_type",
        "face_redeemed_bil",
        "principal_redeemed_bil",
        "cash_paid_bil",
    ],
    "tdcsim_period_payment_flows": [
        "period_start",
        "period_end",
        "flow_id",
        "security_id",
        "holder_sector",
        "holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "payment_type",
        "accounting_basis",
        "amount_bil",
        "is_additive_to_cash_total",
    ],
    "tdcsim_holder_stocks": [
        "date",
        "holder_sector",
        "holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_held_bil",
        "valuation_basis",
        "debt_scope",
        "allocation_method",
    ],
    "tdcsim_debt_target_bridge": [
        "date",
        "cbo_public_debt_target_bil",
        "public_nonmarketable_bridge_bil",
        "non_treasury_and_definition_bridge_bil",
        "controlled_public_marketable_target_bil",
        "controlled_debt_pre_issuance_bil",
        "face_issued_bil",
        "face_retired_bil",
        "tips_principal_indexation_bil",
        "controlled_debt_post_issuance_bil",
        "target_error_bil",
        "intragovernmental_excluded_bil",
        "fed_included_bil",
        "funding_mode",
        "intragovernmental_treatment",
        "fed_held_treasury_treatment",
        "public_nonmarketable_treatment",
    ],
    "tdcsim_scenario_metrics": [
        "date",
        "new_issuance_wam_years",
        "outstanding_controlled_wam_years",
        "new_issuance_bill_share",
        "outstanding_controlled_bill_share",
        "new_issuance_short_maturity_share",
        "outstanding_controlled_short_maturity_share",
        "short_maturity_cutoff_years",
    ],
    "tdcsim_period_tdc_summary": [
        "period_start",
        "period_end",
        "tdc_change_bil",
        "tdc_fiscal_flow_bil",
        "tdc_debt_service_bil",
        "tdc_debt_service_principal_to_du_bil",
        "tdc_debt_service_interest_to_du_bil",
        "tdc_auction_absorption_du_bil",
        "tdc_secondary_trades_bil",
        "tdc_other_bil",
        "overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil",
        "component_sum_bil",
        "component_sum_error_bil",
        "gross_issuance_cash_proceeds_bil",
        "gross_issuance_proceeds_absorbed_by_du_bil",
        "tdc_amount_basis",
        "holder_allocation_scope",
        "overlap_policy",
    ],
    "tdcsim_period_tdc_components": [
        "period_start",
        "period_end",
        "component_id",
        "component_key",
        "component_family",
        "holder_sector",
        "holder_subsector",
        "instrument_type",
        "payment_type",
        "accounting_basis",
        "amount_bil",
        "is_additive_to_tdc_change",
        "enters_direct_interest_support",
        "enters_tdc_deposit_support_default",
        "tdc_amount_basis",
        "overlap_policy",
    ],
}


TDC_AMOUNT_BASIS = "post_mmf_route_pass_through_pre_ratewall_beta_chi"
TDC_HOLDER_SCOPE = "new_issuance_only_for_forecast_holder_preference_scenarios"
TDC_OVERLAP_POLICY = "domestic_nonbank_nominal_interest_components_enter_direct_support_not_default_tdc_support"
TDC_IDENTITY_COLUMNS = (
    "TDC_FiscalFlow",
    "TDC_DebtService",
    "TDC_AuctionAbsorption",
    "TDC_SecondaryTrades",
    "TDC_Other",
)
TDC_OVERLAP_COLUMNS = (
    "TDC_BillDiscountInterestToDU_DomesticNonbank",
    "TDC_CouponInterestToDU_DomesticNonbank",
    "TDC_FRNInterestToDU_DomesticNonbank",
    "TDC_TIPSCouponInterestToDU_DomesticNonbank",
)
TDC_COMPONENT_SPECS = [
    {
        "column": "TDC_FiscalFlow",
        "component_key": "fiscal_flow",
        "component_family": "fiscal",
        "holder_sector": "Private",
        "holder_subsector": "domestic_ultimate_net_primary_proxy",
        "instrument_type": "",
        "payment_type": "primary_deficit_or_surplus",
        "accounting_basis": "signed_net_primary_proxy",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_PrincipalToDU_DomesticNonbank",
        "component_key": "principal_to_du_domestic_nonbank",
        "component_family": "debt_service_principal",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "all",
        "payment_type": "principal",
        "accounting_basis": "cash_or_principal_component",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_PrincipalToDU_MMF",
        "component_key": "principal_to_du_mmf",
        "component_family": "debt_service_principal",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "all",
        "payment_type": "principal",
        "accounting_basis": "post_mmf_route_cash_or_principal_component",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_BillDiscountInterestToDU_DomesticNonbank",
        "component_key": "bill_discount_interest_to_du_domestic_nonbank",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "Fixed",
        "payment_type": "bill_discount",
        "accounting_basis": "budget_accrual",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": True,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_BillDiscountInterestToDU_MMF",
        "component_key": "bill_discount_interest_to_du_mmf",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "Fixed",
        "payment_type": "bill_discount",
        "accounting_basis": "post_mmf_route_budget_accrual",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_CouponInterestToDU_DomesticNonbank",
        "component_key": "fixed_coupon_interest_to_du_domestic_nonbank",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "Fixed",
        "payment_type": "fixed_coupon",
        "accounting_basis": "cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": True,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_CouponInterestToDU_MMF",
        "component_key": "fixed_coupon_interest_to_du_mmf",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "Fixed",
        "payment_type": "fixed_coupon",
        "accounting_basis": "post_mmf_route_cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_FRNInterestToDU_DomesticNonbank",
        "component_key": "frn_interest_to_du_domestic_nonbank",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "FRN",
        "payment_type": "frn_interest",
        "accounting_basis": "cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": True,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_FRNInterestToDU_MMF",
        "component_key": "frn_interest_to_du_mmf",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "FRN",
        "payment_type": "frn_interest",
        "accounting_basis": "post_mmf_route_cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_TIPSCouponInterestToDU_DomesticNonbank",
        "component_key": "tips_coupon_interest_to_du_domestic_nonbank",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "TIPS",
        "payment_type": "tips_coupon",
        "accounting_basis": "cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": True,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_TIPSCouponInterestToDU_MMF",
        "component_key": "tips_coupon_interest_to_du_mmf",
        "component_family": "debt_service_interest",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "TIPS",
        "payment_type": "tips_coupon",
        "accounting_basis": "post_mmf_route_cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_TIPSInflationCompensationToDU_DomesticNonbank",
        "component_key": "tips_indexation_memo_domestic_nonbank",
        "component_family": "debt_service_memo",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "TIPS",
        "payment_type": "tips_indexation",
        "accounting_basis": "memo_decomposition_embedded_in_principal",
        "is_additive_to_tdc_change": False,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_TIPSInflationCompensationToDU_MMF",
        "component_key": "tips_indexation_memo_mmf",
        "component_family": "debt_service_memo",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "TIPS",
        "payment_type": "tips_indexation",
        "accounting_basis": "post_mmf_route_memo_decomposition_embedded_in_principal",
        "is_additive_to_tdc_change": False,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_AuctionAbsorption_DomesticNonbank",
        "component_key": "auction_absorption_domestic_nonbank",
        "component_family": "auction_absorption",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "all",
        "payment_type": "issuance_proceeds",
        "accounting_basis": "cash_proceeds",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_AuctionAbsorption_MMF",
        "component_key": "auction_absorption_mmf",
        "component_family": "auction_absorption",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "all",
        "payment_type": "issuance_proceeds",
        "accounting_basis": "post_mmf_route_cash_proceeds",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_AuctionAbsorption_MMFPlumbing",
        "component_key": "auction_absorption_mmf_ru_plumbing_memo",
        "component_family": "route_plumbing_memo",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "all",
        "payment_type": "issuance_proceeds",
        "accounting_basis": "memo_non_deposit_route",
        "is_additive_to_tdc_change": False,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_SecondaryTrades_DomesticNonbank",
        "component_key": "secondary_trades_domestic_nonbank",
        "component_family": "secondary_trades",
        "holder_sector": "Private",
        "holder_subsector": "domestic_nonbank_deposit_funded",
        "instrument_type": "all",
        "payment_type": "secondary_trade_cash",
        "accounting_basis": "cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_SecondaryTrades_MMF",
        "component_key": "secondary_trades_mmf",
        "component_family": "secondary_trades",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "all",
        "payment_type": "secondary_trade_cash",
        "accounting_basis": "post_mmf_route_cash",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
    {
        "column": "TDC_SecondaryTrades_MMFPlumbing",
        "component_key": "secondary_trades_mmf_ru_plumbing_memo",
        "component_family": "route_plumbing_memo",
        "holder_sector": "Private",
        "holder_subsector": "mmf_cash_fund_route",
        "instrument_type": "all",
        "payment_type": "secondary_trade_cash",
        "accounting_basis": "memo_non_deposit_route",
        "is_additive_to_tdc_change": False,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": False,
    },
    {
        "column": "TDC_Other",
        "component_key": "other_tdc",
        "component_family": "other",
        "holder_sector": "",
        "holder_subsector": "",
        "instrument_type": "",
        "payment_type": "other",
        "accounting_basis": "engine_residual_component",
        "is_additive_to_tdc_change": True,
        "enters_direct_interest_support": False,
        "enters_tdc_deposit_support_default": True,
    },
]


def write_scenario_outputs(
    results: pd.DataFrame,
    final_portfolio: pd.DataFrame,
    output_dir: str | Path,
    *,
    profile: str = "compact",
    compression: str = "gzip",
    catalog_sqlite: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write run outputs and return a hash-listed output manifest."""

    if profile not in {"summary", "compact", "audit"}:
        raise ValueError(f"unsupported output profile: {profile}")
    if compression not in {"gzip", "none"}:
        raise ValueError(f"unsupported compression: {compression}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"

    results_out = _ensure_date_column(results)
    result_cols = [col for col in SUMMARY_COLUMNS if col in results_out.columns]
    if profile == "audit":
        result_cols = list(results_out.columns)
    result_path = out / f"results_{profile}{suffix}"
    _write_frame(results_out[result_cols] if result_cols else results_out, result_path, compression=compression)

    outputs: dict[str, Any] = {
        "profile": profile,
        "compression": compression,
        "results": _artifact_record(out, result_path),
    }
    if profile in {"compact", "audit"}:
        portfolio = final_portfolio
        if profile == "compact" and "Status" in portfolio.columns:
            portfolio = portfolio[portfolio["Status"].astype(str).eq("Active")]
        portfolio_path = out / f"final_portfolio_{profile}{suffix}"
        _write_frame(portfolio, portfolio_path, compression=compression)
        outputs["final_portfolio"] = _artifact_record(out, portfolio_path)

    summary = _summary(results, final_portfolio)
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    outputs["summary"] = _artifact_record(out, summary_path)
    outputs["summary_values"] = summary
    if metadata:
        outputs["row_metadata"] = dict(metadata)

    handoff_tables = _handoff_tables(results, metadata or {})
    for logical_name, frame in handoff_tables.items():
        path = out / f"{logical_name}{suffix}"
        _write_frame(frame, path, compression=compression)
        outputs[logical_name] = _artifact_record(out, path)
        outputs[logical_name]["row_count"] = int(len(frame))

    if catalog_sqlite:
        catalog_path = out / "catalog.sqlite"
        _write_catalog(catalog_path, outputs)
        outputs["catalog_sqlite"] = _artifact_record(out, catalog_path)
    return outputs


def _handoff_tables(results: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    raw = results.attrs.get("handoff_tables", {})
    if not isinstance(raw, Mapping):
        raw = {}
    derived = _tdc_handoff_tables(results)
    tables: dict[str, pd.DataFrame] = {}
    for name, columns in HANDOFF_TABLE_COLUMNS.items():
        rows = derived.get(name, raw.get(name, []))
        if rows is None:
            continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[columns + [col for col in frame.columns if col not in columns]]
        tables[name] = _with_metadata(frame, metadata)
    return tables


def _tdc_handoff_tables(results: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    frame = _ensure_date_column(results)
    required = {"Date", "TDC_Change", *TDC_IDENTITY_COLUMNS}
    if frame.empty or not required <= set(frame.columns):
        return {}
    rows = frame.copy()
    rows["Date"] = pd.to_datetime(rows["Date"], errors="coerce")
    rows = rows[rows["Date"].notna()].sort_values("Date").reset_index(drop=True)
    if len(rows) <= 1:
        return {
            "tdcsim_period_tdc_summary": [],
            "tdcsim_period_tdc_components": [],
        }
    summary_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for idx in range(1, len(rows)):
        prev_date = rows.loc[idx - 1, "Date"]
        current_date = rows.loc[idx, "Date"]
        period = {
            "period_start": str(pd.Timestamp(prev_date).date()),
            "period_end": str(pd.Timestamp(current_date).date()),
        }
        row = rows.loc[idx]
        overlap = sum(_number(row, column) for column in TDC_OVERLAP_COLUMNS)
        component_sum = sum(_number(row, column) for column in TDC_IDENTITY_COLUMNS)
        tdc_change = _number(row, "TDC_Change")
        summary_rows.append(
            {
                **period,
                "tdc_change_bil": tdc_change,
                "tdc_fiscal_flow_bil": _number(row, "TDC_FiscalFlow"),
                "tdc_debt_service_bil": _number(row, "TDC_DebtService"),
                "tdc_debt_service_principal_to_du_bil": _number(row, "TDC_PrincipalToDU"),
                "tdc_debt_service_interest_to_du_bil": _number(row, "TDC_InterestToDU"),
                "tdc_auction_absorption_du_bil": _number(row, "TDC_AuctionAbsorption"),
                "tdc_secondary_trades_bil": _number(row, "TDC_SecondaryTrades"),
                "tdc_other_bil": _number(row, "TDC_Other"),
                "overlap_cashflow_bil": overlap,
                "tdc_change_ex_overlap_bil": tdc_change - overlap,
                "component_sum_bil": component_sum,
                "component_sum_error_bil": tdc_change - component_sum,
                "gross_issuance_cash_proceeds_bil": _number(row, "AuctionProceeds"),
                "gross_issuance_proceeds_absorbed_by_du_bil": _number(
                    row, "TDC_GrossIssuanceProceedsAbsorbedByDU"
                ),
                "tdc_amount_basis": TDC_AMOUNT_BASIS,
                "holder_allocation_scope": TDC_HOLDER_SCOPE,
                "overlap_policy": TDC_OVERLAP_POLICY,
            }
        )
        for spec in TDC_COMPONENT_SPECS:
            amount = _number(row, spec["column"])
            if abs(amount) <= 1e-12:
                continue
            component_key = str(spec["component_key"])
            component_rows.append(
                {
                    **period,
                    "component_id": _component_id(current_date, component_key),
                    "component_key": component_key,
                    "component_family": spec["component_family"],
                    "holder_sector": spec["holder_sector"],
                    "holder_subsector": spec["holder_subsector"],
                    "instrument_type": spec["instrument_type"],
                    "payment_type": spec["payment_type"],
                    "accounting_basis": spec["accounting_basis"],
                    "amount_bil": amount,
                    "is_additive_to_tdc_change": bool(spec["is_additive_to_tdc_change"]),
                    "enters_direct_interest_support": bool(spec["enters_direct_interest_support"]),
                    "enters_tdc_deposit_support_default": bool(spec["enters_tdc_deposit_support_default"]),
                    "tdc_amount_basis": TDC_AMOUNT_BASIS,
                    "overlap_policy": TDC_OVERLAP_POLICY,
                }
            )
    return {
        "tdcsim_period_tdc_summary": summary_rows,
        "tdcsim_period_tdc_components": component_rows,
    }


def _number(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return 0.0
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value):
        return 0.0
    return float(value)


def _component_id(date: pd.Timestamp, component_key: str) -> str:
    clean_key = component_key.replace("|", "_")
    return f"tdc|{pd.Timestamp(date).date()}|{clean_key}"


def _with_metadata(frame: pd.DataFrame, metadata: Mapping[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for key, value in metadata.items():
        if key not in out.columns:
            out[key] = value
    metadata_cols = [key for key in COMMON_METADATA_COLUMNS if key in out.columns]
    metadata_cols.extend(key for key in metadata if key in out.columns and key not in metadata_cols)
    other_cols = [col for col in out.columns if col not in metadata_cols]
    return out[metadata_cols + other_cols]


def hash_output_tree(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    records = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(_artifact_record(root, file_path))
    return records


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str) -> None:
    if compression == "gzip":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                    frame.to_csv(handle, index=False)
    else:
        frame.to_csv(path, index=False)


def _ensure_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "Date" in frame.columns:
        return frame
    if frame.index.name == "Date":
        return frame.reset_index()
    return frame


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _summary(results: pd.DataFrame, final_portfolio: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(results)),
        "final_portfolio_rows": int(len(final_portfolio)),
    }
    for col in (
        "CBORequiredFaceIssuance",
        "CBOControlledDebtTargetError",
        "CBOFedHoldingsTargetError",
        "CBOFedAuctionShare",
        "CBOFedAuctionRolloverAddons",
        "CBOCashReconciliationResidual",
    ):
        if col in results.columns and len(results[col]) > 0:
            numeric = pd.to_numeric(results[col], errors="coerce").fillna(0.0)
            summary[f"{col}_sum"] = float(numeric.sum())
            summary[f"{col}_max_abs"] = float(numeric.abs().max())
    for col in ("CBORemittanceStatus", "NetInterestDiagnosticStatus", "CBOFedStockMode"):
        if col in results.columns:
            summary[f"{col}_values"] = sorted(str(value) for value in results[col].dropna().unique())
    return summary


def _write_catalog(path: Path, outputs: Mapping[str, Any]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE artifacts (key TEXT PRIMARY KEY, path TEXT, bytes INTEGER, sha256 TEXT)")
        for key, value in outputs.items():
            if isinstance(value, Mapping) and {"path", "bytes", "sha256"} <= set(value):
                conn.execute(
                    "INSERT INTO artifacts (key, path, bytes, sha256) VALUES (?, ?, ?, ?)",
                    (key, value["path"], int(value["bytes"]), value["sha256"]),
                )
        conn.execute("CREATE TABLE summary (payload TEXT NOT NULL)")
        conn.execute("INSERT INTO summary (payload) VALUES (?)", (json.dumps(outputs.get("summary_values", {}), sort_keys=True),))


__all__ = ["SUMMARY_COLUMNS", "hash_output_tree", "write_scenario_outputs"]
