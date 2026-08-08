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
    "CBOControlledDebtReference",
    "ScenarioControlledDebt",
    "DebtDriftFromReference",
    "CashFinancingFaceIssued",
    "CashFinancingProceeds",
    "CBOControlledDebtPreIssuance",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
    "NewDebtIssued",
    "AuctionProceeds",
    "IssuePriceCashGap",
    "PrimaryDeficit",
    "TGA",
    "Reserves",
    "TDC_Level",
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
    "CBOFedSecondarySaleCash",
    "CBOFedSecondarySaleReserveEffect",
    "CBOFedSecondarySaleDepositEffect",
    "CBOFedPrivateMaturityTDC",
    "CBOFedSyntheticSecondaryPurchases",
    "CBOFedSyntheticSecondarySales",
    "CBOFedStockMode",
    "CBOFedSettlementScope",
    "CBOFedAcquisitionChannel",
    "CBOFedSecondarySaleBuyerMix",
    "CBORemittanceCashEffect",
    "CBORemittanceStatus",
    "CB_Remittance",
    "CB_DeferredAsset",
    "CBONetInterestDiagnostic",
    "CBOTotalDeficitDiagnostic",
    "CBONetInterestBridgeRows",
    "NetInterestDiagnosticStatus",
    "DebtHeld_Banks",
    "DebtHeld_CentralBank",
    "DebtHeld_Foreign",
    "DebtHeld_DomesticNonBanks",
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

BOUNDED_SUMMARY_COLUMNS = [
    *SUMMARY_COLUMNS,
    "PrincipalPaid_Bonds",
    "InterestOutlay_Period",
    "IssueDiscountCost_Period",
    "NonMarketableInterestCapitalized_Period",
    "TIPSInflationAccretion_Period",
    "FinancingCost_Period",
    "TDC_Change",
    "TDC_FiscalFlow",
    "TDC_DebtService",
    "TDC_AuctionAbsorption",
    "TDC_SecondaryTrades",
    "TDC_Other",
    "TDC_PrincipalToDU",
    "TDC_PrincipalCashToDU",
    "TDC_InterestToDU",
    "TDC_PrincipalToDU_DomesticNonbank",
    "TDC_PrincipalToDU_MMF",
    "TDC_PrincipalCashToDU_DomesticNonbank",
    "TDC_PrincipalCashToDU_MMF",
    "TDC_PrincipalCashToDU_MMFPlumbing",
    "TDC_BillDiscountInterestToDU_DomesticNonbank",
    "TDC_BillDiscountInterestToDU_MMF",
    "TDC_CouponInterestToDU_DomesticNonbank",
    "TDC_CouponInterestToDU_MMF",
    "TDC_FRNInterestToDU_DomesticNonbank",
    "TDC_FRNInterestToDU_MMF",
    "TDC_TIPSCouponInterestToDU_DomesticNonbank",
    "TDC_TIPSCouponInterestToDU_MMF",
    "TDC_TIPSInflationCompensationToDU_DomesticNonbank",
    "TDC_TIPSInflationCompensationToDU_MMF",
    "TDC_GrossIssuanceProceedsAbsorbedByDU",
    "TDC_NetPrincipalIssuanceCashflowToDU",
    "TDC_AuctionAbsorption_DomesticNonbank",
    "TDC_AuctionAbsorption_MMF",
    "TDC_AuctionAbsorption_MMFPlumbing",
    "TDC_SecondaryTrades_DomesticNonbank",
    "TDC_SecondaryTrades_MMF",
    "TDC_SecondaryTrades_MMFPlumbing",
    "CBOBuybackFaceRetired",
    "CBOBuybackCashPaid",
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
    "tdcsim_accounting_journal": [
        "period_start",
        "period_end",
        "journal_id",
        "event_type",
        "leg_type",
        "security_id",
        "holder_sector",
        "holder_subsector",
        "counterparty_sector",
        "counterparty_subsector",
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "accounting_basis",
        "face_stock_change_bil",
        "adjusted_principal_change_bil",
        "route_face_stock_change_bil",
        "route_adjusted_principal_change_bil",
        "treasury_cash_change_bil",
        "reserve_change_bil",
        "deposit_change_bil",
        "settlement_scope",
        "is_intragovernmental",
    ],
    "tdcsim_accounting_closure": [
        "period_start",
        "period_end",
        "opening_face_stock_bil",
        "journal_face_stock_change_bil",
        "closing_face_stock_bil",
        "face_stock_closure_error_bil",
        "opening_adjusted_principal_stock_bil",
        "journal_adjusted_principal_change_bil",
        "closing_adjusted_principal_stock_bil",
        "adjusted_principal_closure_error_bil",
        "opening_treasury_cash_bil",
        "journal_treasury_cash_change_bil",
        "closing_treasury_cash_bil",
        "treasury_cash_closure_error_bil",
        "journal_reserve_change_bil",
        "reported_reserve_change_bil",
        "reserve_closure_error_bil",
        "journal_deposit_change_bil",
        "reported_deposit_change_bil",
        "deposit_closure_error_bil",
        "holder_debt_total_bil",
        "instrument_debt_total_bil",
        "aggregate_debt_bil",
        "holder_total_error_bil",
        "instrument_total_error_bil",
        "closure_basis",
        "unexplained_residual_bil",
    ],
    "tdcsim_period_issuance_flows": [
        "period_start",
        "period_end",
        "flow_id",
        "security_id",
        "holder_sector",
        "holder_subsector",
        "issuance_leg",
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
        "adjusted_principal_stock_removed_bil",
        "tdc_principal_recipient_sector",
        "tdc_principal_recipient_subsector",
        "tdc_principal_cash_paid_to_du_bil",
        "tdc_principal_redeemed_to_du_bil",
        "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
        "tdc_principal_redeemed_to_du_domestic_nonbank_bil",
        "tdc_principal_cash_paid_to_du_mmf_bil",
        "tdc_principal_redeemed_to_du_mmf_bil",
        "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
        "tdc_principal_redeemed_to_du_mmf_plumbing_bil",
        "tdc_principal_recipient_basis",
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
        "face_stock_bil",
        "adjusted_principal_stock_bil",
        "valuation_basis",
        "debt_scope",
        "allocation_method",
    ],
    "tdcsim_tdc_principal_route_stocks": [
        "date",
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "route_debt_held_bil",
        "route_face_stock_bil",
        "route_adjusted_principal_stock_bil",
        "valuation_basis",
        "debt_scope",
        "allocation_method",
        "route_stock_basis",
    ],
    "tdcsim_tdc_principal_route_stock_closure": [
        "period_start",
        "period_end",
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_scope",
        "opening_route_stock_bil",
        "route_face_issued_bil",
        "route_face_redeemed_bil",
        "route_journal_face_change_bil",
        "route_journal_adjusted_principal_change_bil",
        "route_stock_residual_or_indexation_bil",
        "closing_route_stock_bil",
        "closure_identity_error_bil",
        "route_stock_basis",
        "residual_basis",
    ],
    "tdcsim_debt_target_bridge": [
        "date",
        "cbo_public_debt_target_bil",
        "public_nonmarketable_bridge_bil",
        "non_treasury_and_definition_bridge_bil",
        "controlled_public_marketable_target_bil",
        "cbo_controlled_debt_reference_bil",
        "scenario_controlled_debt_bil",
        "debt_drift_from_reference_bil",
        "cash_financing_face_issued_bil",
        "cash_financing_proceeds_bil",
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
        "gross_principal_cash_paid_to_du_bil",
        "principal_redeemed_to_du_domestic_nonbank_bil",
        "principal_redeemed_to_du_mmf_bil",
        "gross_principal_cash_paid_to_du_domestic_nonbank_bil",
        "gross_principal_cash_paid_to_du_mmf_bil",
        "gross_principal_cash_paid_to_du_mmf_plumbing_bil",
        "tdc_auction_absorption_du_bil",
        "tdc_secondary_trades_bil",
        "tdc_other_bil",
        "overlap_cashflow_bil",
        "tdc_change_ex_overlap_bil",
        "component_sum_bil",
        "component_sum_error_bil",
        "gross_issuance_cash_proceeds_bil",
        "gross_issuance_proceeds_absorbed_by_du_bil",
        "net_du_principal_issuance_cashflow_bil",
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


def write_bounded_scenario_outputs(
    results: pd.DataFrame,
    final_portfolio: pd.DataFrame,
    output_dir: str | Path,
    *,
    bounded_summary: Mapping[str, Any],
    profile: str = "compact",
    compression: str = "gzip",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write fixed-size run state around already-streamed bounded evidence.

    The bounded sink owns every compact accounting/economic evidence file.  This
    finalizer intentionally does not call ``_handoff_tables`` or materialize any
    whole-history handoff DataFrame.
    """

    if profile not in {"summary", "compact", "audit"}:
        raise ValueError(f"unsupported output profile: {profile}")
    if compression not in {"gzip", "none"}:
        raise ValueError(f"unsupported compression: {compression}")
    if bounded_summary.get("evidence_profile") != "bounded_period_closure_v1":
        raise ValueError("bounded output summary has an unsupported evidence profile")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"

    results_out = _ensure_date_column(results)
    result_cols = [
        col for col in BOUNDED_SUMMARY_COLUMNS if col in results_out.columns
    ]
    if profile == "audit":
        result_cols = list(results_out.columns)
    result_path = out / f"results_{profile}{suffix}"
    _write_frame(
        results_out[result_cols] if result_cols else results_out,
        result_path,
        compression=compression,
    )

    outputs: dict[str, Any] = {
        "profile": profile,
        "compression": compression,
        "results": _artifact_record(out, result_path),
        "evidence_profile": bounded_summary["evidence_profile"],
        "verification_grade": bounded_summary["verification_grade"],
    }
    if profile in {"compact", "audit"}:
        portfolio = final_portfolio
        if profile == "compact" and "Status" in portfolio.columns:
            portfolio = portfolio[portfolio["Status"].astype(str).eq("Active")]
        portfolio_path = out / f"final_portfolio_{profile}{suffix}"
        _write_frame(portfolio, portfolio_path, compression=compression)
        outputs["final_portfolio"] = _artifact_record(out, portfolio_path)

    summary = _summary(results, final_portfolio)
    summary.update(
        {
            "evidence_profile": bounded_summary["evidence_profile"],
            "verification_grade": bounded_summary["verification_grade"],
            "event_count": int(bounded_summary["event_count"]),
            "event_root_sha256": str(bounded_summary["event_root_sha256"]),
            "final_state_sha256": str(bounded_summary["final_state_sha256"]),
            "peak_rss_bytes": int(bounded_summary["peak_rss_bytes"]),
            "portfolio_row_budget": int(bounded_summary["portfolio_row_budget"]),
            "max_portfolio_rows": int(bounded_summary["max_portfolio_rows"]),
            "max_key_cardinality": int(bounded_summary["max_key_cardinality"]),
        }
    )
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    outputs["summary"] = _artifact_record(out, summary_path)
    outputs["summary_values"] = summary
    if metadata:
        outputs["row_metadata"] = dict(metadata)

    artifacts = bounded_summary.get("artifacts")
    deterministic = bounded_summary.get("deterministic_artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(deterministic, Mapping):
        raise ValueError("bounded output summary is missing artifact manifests")
    for logical_name, expected in sorted(artifacts.items()):
        if not isinstance(expected, Mapping):
            raise ValueError(f"bounded artifact record is malformed: {logical_name}")
        path = out / str(expected.get("path") or "")
        actual = _artifact_record(out, path)
        if (
            actual["sha256"] != expected.get("sha256")
            or actual["bytes"] != expected.get("bytes")
        ):
            raise ValueError(f"bounded artifact changed after sink finalization: {logical_name}")
        actual["row_count"] = int(expected.get("row_count", 0))
        outputs[logical_name] = actual
    outputs["deterministic_evidence_artifacts"] = dict(deterministic)
    return outputs


def _handoff_tables(results: pd.DataFrame, metadata: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    raw = results.attrs.get("handoff_tables", {})
    if not isinstance(raw, Mapping):
        raw = {}
    derived = _tdc_handoff_tables(results)
    derived.update(_route_stock_closure_handoff_tables(raw))
    derived.update(_accounting_closure_handoff_tables(results, raw))
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
    frame = _ensure_date_column(results).copy()
    frame.attrs = {}
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
                "gross_principal_cash_paid_to_du_bil": _number(row, "TDC_PrincipalCashToDU"),
                "principal_redeemed_to_du_domestic_nonbank_bil": _number(
                    row, "TDC_PrincipalToDU_DomesticNonbank"
                ),
                "principal_redeemed_to_du_mmf_bil": _number(row, "TDC_PrincipalToDU_MMF"),
                "gross_principal_cash_paid_to_du_domestic_nonbank_bil": _number(
                    row, "TDC_PrincipalCashToDU_DomesticNonbank"
                ),
                "gross_principal_cash_paid_to_du_mmf_bil": _number(row, "TDC_PrincipalCashToDU_MMF"),
                "gross_principal_cash_paid_to_du_mmf_plumbing_bil": _number(
                    row, "TDC_PrincipalCashToDU_MMFPlumbing"
                ),
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
                "net_du_principal_issuance_cashflow_bil": _number(
                    row, "TDC_NetPrincipalIssuanceCashflowToDU"
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


def _route_stock_closure_handoff_tables(raw: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    stocks = pd.DataFrame(raw.get("tdcsim_tdc_principal_route_stocks", []))
    if stocks.empty:
        return {"tdcsim_tdc_principal_route_stock_closure": []}
    issuance = pd.DataFrame(raw.get("tdcsim_period_issuance_flows", []))
    principal = pd.DataFrame(raw.get("tdcsim_period_principal_flows", []))
    journal = pd.DataFrame(raw.get("tdcsim_accounting_journal", []))
    rows: list[dict[str, Any]] = []
    stocks = stocks.copy()
    stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
    stocks = stocks[stocks["date"].notna()]
    if stocks.empty:
        return {"tdcsim_tdc_principal_route_stock_closure": []}
    date_values = set(stocks["date"].dropna().tolist())
    for flow_frame in (journal, issuance, principal):
        if flow_frame.empty:
            continue
        for column in ("period_start", "period_end"):
            if column not in flow_frame.columns:
                continue
            parsed = pd.to_datetime(flow_frame[column], errors="coerce")
            date_values.update(parsed[parsed.notna()].tolist())
    dates = sorted(date_values)
    for start, end in zip(dates, dates[1:]):
        period_start = str(pd.Timestamp(start).date())
        period_end = str(pd.Timestamp(end).date())
        opening = _route_stock_map(stocks[stocks["date"].eq(start)])
        closing = _route_stock_map(stocks[stocks["date"].eq(end)])
        issued = _route_issuance_map(issuance, period_start=period_start, period_end=period_end)
        redeemed = _route_redemption_map(principal, period_start=period_start, period_end=period_end)
        journal_face, journal_adjusted, journal_debt = _route_journal_change_maps(
            journal,
            period_start=period_start,
            period_end=period_end,
        )
        journal_mode = bool(journal_face or journal_adjusted or journal_debt)
        keys = (
            set(opening)
            | set(closing)
            | set(issued)
            | set(redeemed)
            | set(journal_face)
            | set(journal_adjusted)
            | set(journal_debt)
        )
        for key in sorted(keys):
            open_value = opening.get(key, 0.0)
            issued_value = issued.get(key, 0.0)
            redeemed_value = redeemed.get(key, 0.0)
            close_value = closing.get(key, 0.0)
            if journal_mode:
                identity_error = (
                    close_value - open_value - journal_debt.get(key, 0.0)
                )
            else:
                identity_error = (
                    close_value - open_value - issued_value + redeemed_value
                )
            route_holder, route_subbucket, instrument_type, maturity_bucket, debt_scope = key
            rows.append(
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "route_holder_sector": route_holder,
                    "route_holder_subsector": route_subbucket,
                    "instrument_type": instrument_type,
                    "maturity_bucket": maturity_bucket,
                    "debt_scope": debt_scope,
                    "opening_route_stock_bil": open_value,
                    "route_face_issued_bil": issued_value,
                    "route_face_redeemed_bil": redeemed_value,
                    "route_journal_face_change_bil": journal_face.get(key, 0.0),
                    "route_journal_adjusted_principal_change_bil": journal_adjusted.get(
                        key, 0.0
                    ),
                    "route_stock_residual_or_indexation_bil": 0.0,
                    "closing_route_stock_bil": close_value,
                    "closure_identity_error_bil": identity_error,
                    "route_stock_basis": "tdc_principal_settlement_route",
                    "residual_basis": "none_fail_closed_no_unrestricted_residual",
                }
            )
    return {"tdcsim_tdc_principal_route_stock_closure": rows}


def _route_journal_change_maps(
    frame: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
) -> tuple[
    dict[tuple[str, str, str, str, str], float],
    dict[tuple[str, str, str, str, str], float],
    dict[tuple[str, str, str, str, str], float],
]:
    if frame.empty:
        return {}, {}, {}
    work = frame[
        frame.get("period_start", pd.Series("", index=frame.index)).astype(str).eq(period_start)
        & frame.get("period_end", pd.Series("", index=frame.index)).astype(str).eq(period_end)
    ].copy()
    if work.empty:
        return {}, {}, {}
    for column in (
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
    ):
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].fillna("").astype(str)
    work["_face"] = pd.to_numeric(
        work.get(
            "route_face_stock_change_bil",
            pd.Series(0.0, index=work.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    work["_adjusted"] = pd.to_numeric(
        work.get(
            "route_adjusted_principal_change_bil",
            pd.Series(0.0, index=work.index),
        ),
        errors="coerce",
    ).fillna(0.0)
    work["_debt"] = work["_face"].where(
        ~work["instrument_type"].eq("TIPS"), work["_adjusted"]
    )
    work["_intragov"] = work.get(
        "is_intragovernmental", pd.Series(False, index=work.index)
    ).fillna(False).astype(str).str.lower().isin({"true", "1"})
    scopes: list[pd.DataFrame] = []
    all_active = work.copy()
    all_active["debt_scope"] = "all_active_treasury"
    scopes.append(all_active)
    controlled = work[
        work["instrument_type"].isin(["Fixed", "TIPS", "FRN"])
        & ~work["_intragov"]
    ].copy()
    controlled["debt_scope"] = "controlled_public_marketable"
    scopes.append(controlled)
    expanded = pd.concat(scopes, ignore_index=True)
    keys = [
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_scope",
    ]

    def grouped(column: str) -> dict[tuple[str, str, str, str, str], float]:
        values = expanded.groupby(keys, dropna=False)[column].sum()
        return {
            tuple(key): float(value)
            for key, value in values.items()
            if abs(float(value)) > 1e-12
        }

    return grouped("_face"), grouped("_adjusted"), grouped("_debt")


def _accounting_closure_handoff_tables(
    results: pd.DataFrame,
    raw: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    journal = pd.DataFrame(raw.get("tdcsim_accounting_journal", []))
    if journal.empty:
        return {"tdcsim_accounting_closure": []}
    result_rows = _ensure_date_column(results).copy()
    result_rows["Date"] = pd.to_datetime(result_rows["Date"], errors="coerce")
    result_rows = (
        result_rows[result_rows["Date"].notna()]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if len(result_rows) <= 1:
        return {"tdcsim_accounting_closure": []}
    stocks = pd.DataFrame(raw.get("tdcsim_holder_stocks", []))
    if not stocks.empty:
        stocks = stocks.copy()
        stocks["date"] = pd.to_datetime(stocks["date"], errors="coerce")
        stocks = stocks[
            stocks["date"].notna()
            & stocks.get(
                "debt_scope",
                pd.Series("", index=stocks.index),
            ).astype(str).eq("all_active_treasury")
        ].copy()
    journal = journal.copy()

    def strict_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
        if column not in frame.columns:
            raise ValueError(f"accounting source is missing numeric column: {column}")
        values = pd.to_numeric(frame[column], errors="raise")
        if values.isna().any() or not values.map(
            lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf")
        ).all():
            raise ValueError(
                f"accounting source has malformed or nonfinite values: {column}"
            )
        return values.astype(float)

    for column in (
        "face_stock_change_bil",
        "adjusted_principal_change_bil",
        "treasury_cash_change_bil",
        "reserve_change_bil",
        "deposit_change_bil",
    ):
        journal[column] = strict_numeric(journal, column)
    rows: list[dict[str, Any]] = []
    for index in range(1, len(result_rows)):
        opening = result_rows.iloc[index - 1]
        closing = result_rows.iloc[index]
        period_start = str(pd.Timestamp(opening["Date"]).date())
        period_end = str(pd.Timestamp(closing["Date"]).date())
        period_journal = journal[
            journal.get(
                "period_start", pd.Series("", index=journal.index)
            ).astype(str).eq(period_start)
            & journal.get(
                "period_end", pd.Series("", index=journal.index)
            ).astype(str).eq(period_end)
        ]
        opening_stocks = (
            stocks[stocks["date"].eq(opening["Date"])]
            if not stocks.empty
            else stocks
        )
        closing_stocks = (
            stocks[stocks["date"].eq(closing["Date"])]
            if not stocks.empty
            else stocks
        )

        def stock_total(frame: pd.DataFrame, column: str) -> float:
            if frame.empty:
                return 0.0
            return float(strict_numeric(frame, column).sum())

        opening_face = stock_total(opening_stocks, "face_stock_bil")
        closing_face = stock_total(closing_stocks, "face_stock_bil")
        opening_adjusted = stock_total(
            opening_stocks, "adjusted_principal_stock_bil"
        )
        closing_adjusted = stock_total(
            closing_stocks, "adjusted_principal_stock_bil"
        )
        journal_face = float(period_journal["face_stock_change_bil"].sum())
        journal_adjusted = float(
            period_journal["adjusted_principal_change_bil"].sum()
        )
        journal_cash = float(period_journal["treasury_cash_change_bil"].sum())
        journal_reserve = float(period_journal["reserve_change_bil"].sum())
        journal_deposit = float(period_journal["deposit_change_bil"].sum())
        opening_cash = _number(opening, "TGA")
        closing_cash = _number(closing, "TGA")
        reported_reserve = _number(closing, "Reserves") - _number(
            opening, "Reserves"
        )
        reported_deposit = _number(closing, "TDC_Level") - _number(
            opening, "TDC_Level"
        )
        if closing_stocks.empty:
            holder_total = 0.0
            instrument_total = 0.0
        else:
            closing_stocks = closing_stocks.copy()
            closing_stocks["_debt_held"] = strict_numeric(
                closing_stocks, "debt_held_bil"
            )
            holder_total = float(
                closing_stocks.groupby(
                    ["holder_sector", "holder_subsector"],
                    dropna=False,
                )["_debt_held"].sum().sum()
            )
            instrument_total = float(
                closing_stocks.groupby(
                    ["instrument_type", "maturity_bucket"],
                    dropna=False,
                )["_debt_held"].sum().sum()
            )
        aggregate_debt = _number(closing, "TotalDebt_Agg")
        rows.append(
            {
                "period_start": period_start,
                "period_end": period_end,
                "opening_face_stock_bil": opening_face,
                "journal_face_stock_change_bil": journal_face,
                "closing_face_stock_bil": closing_face,
                "face_stock_closure_error_bil": closing_face
                - opening_face
                - journal_face,
                "opening_adjusted_principal_stock_bil": opening_adjusted,
                "journal_adjusted_principal_change_bil": journal_adjusted,
                "closing_adjusted_principal_stock_bil": closing_adjusted,
                "adjusted_principal_closure_error_bil": closing_adjusted
                - opening_adjusted
                - journal_adjusted,
                "opening_treasury_cash_bil": opening_cash,
                "journal_treasury_cash_change_bil": journal_cash,
                "closing_treasury_cash_bil": closing_cash,
                "treasury_cash_closure_error_bil": closing_cash
                - opening_cash
                - journal_cash,
                "journal_reserve_change_bil": journal_reserve,
                "reported_reserve_change_bil": reported_reserve,
                "reserve_closure_error_bil": reported_reserve - journal_reserve,
                "journal_deposit_change_bil": journal_deposit,
                "reported_deposit_change_bil": reported_deposit,
                "deposit_closure_error_bil": reported_deposit - journal_deposit,
                "holder_debt_total_bil": holder_total,
                "instrument_debt_total_bil": instrument_total,
                "aggregate_debt_bil": aggregate_debt,
                "holder_total_error_bil": holder_total - aggregate_debt,
                "instrument_total_error_bil": instrument_total - aggregate_debt,
                "closure_basis": "independent_opening_and_closing_state_snapshots",
                "unexplained_residual_bil": 0.0,
            }
        )
    return {"tdcsim_accounting_closure": rows}


def _route_stock_map(frame: pd.DataFrame) -> dict[tuple[str, str, str, str, str], float]:
    if frame.empty:
        return {}
    work = frame.copy()
    for column in (
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_scope",
    ):
        if column not in work.columns:
            work[column] = ""
        work[column] = work[column].fillna("").astype(str)
    work["amount"] = pd.to_numeric(work.get("route_debt_held_bil", 0.0), errors="coerce").fillna(0.0)
    grouped = work.groupby(
        [
            "route_holder_sector",
            "route_holder_subsector",
            "instrument_type",
            "maturity_bucket",
            "debt_scope",
        ],
        dropna=False,
    )["amount"].sum()
    return {tuple(key): float(value) for key, value in grouped.items()}


def _route_issuance_map(
    frame: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
) -> dict[tuple[str, str, str, str, str], float]:
    if frame.empty:
        return {}
    work = frame[
        frame.get("period_start", pd.Series("", index=frame.index)).astype(str).eq(period_start)
        & frame.get("period_end", pd.Series("", index=frame.index)).astype(str).eq(period_end)
    ].copy()
    if work.empty:
        return {}
    rename = {
        "holder_sector": "route_holder_sector",
        "holder_subsector": "route_holder_subsector",
    }
    work = work.rename(columns=rename)
    work["debt_scope"] = "controlled_public_marketable"
    return _flow_amount_map(work, "face_issued_bil")


def _route_redemption_map(
    frame: pd.DataFrame,
    *,
    period_start: str,
    period_end: str,
) -> dict[tuple[str, str, str, str, str], float]:
    if frame.empty:
        return {}
    work = frame[
        frame.get("period_start", pd.Series("", index=frame.index)).astype(str).eq(period_start)
        & frame.get("period_end", pd.Series("", index=frame.index)).astype(str).eq(period_end)
    ].copy()
    if work.empty:
        return {}
    rename = {
        "tdc_principal_recipient_sector": "route_holder_sector",
        "tdc_principal_recipient_subsector": "route_holder_subsector",
    }
    work = work.rename(columns=rename)
    work["debt_scope"] = "controlled_public_marketable"
    return _flow_amount_map(work, "face_redeemed_bil")


def _flow_amount_map(frame: pd.DataFrame, amount_column: str) -> dict[tuple[str, str, str, str, str], float]:
    for column in (
        "route_holder_sector",
        "route_holder_subsector",
        "instrument_type",
        "maturity_bucket",
        "debt_scope",
    ):
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    if amount_column in frame.columns:
        amount_values = frame[amount_column]
    else:
        amount_values = pd.Series(0.0, index=frame.index)
    frame["amount"] = pd.to_numeric(amount_values, errors="coerce").fillna(0.0)
    grouped = frame.groupby(
        [
            "route_holder_sector",
            "route_holder_subsector",
            "instrument_type",
            "maturity_bucket",
            "debt_scope",
        ],
        dropna=False,
    )["amount"].sum()
    return {tuple(key): float(value) for key, value in grouped.items()}


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


__all__ = [
    "SUMMARY_COLUMNS",
    "hash_output_tree",
    "write_bounded_scenario_outputs",
    "write_scenario_outputs",
]
