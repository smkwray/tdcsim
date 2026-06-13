"""RateWall contract export for tdcsim.

This module turns tdcsim period results into a versioned, quarterly contract
bundle. It does not change the simulation itself; it exposes the funding-chain
components RateWall needs to ingest and audit TDC assumptions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from contract_validation import validate_ratewall_contract
from holder_mapping import mapping_rows
from quarterly_aggregation import add_quarter_column
from tdc_shared import (
    MMF_DEPOSIT_PASS_THROUGH_DEFAULT,
    MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID,
    MMF_DEPOSIT_PASS_THROUGH_STATUS,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)


CONTRACT_VERSION = "0.2.0"
CLAIM_BOUNDARY = "tdcsim_forward_scenario_not_forecast_or_evidence_mode"
PRIVATE_ROUTE_SENSITIVITY_KEY = "tdcsim_private_route_sensitivity_contract_v1"
PRIVATE_ROUTE_SENSITIVITY_BLOCKER = (
    "requires_source_backed_split_from_current_private_holder_bucket"
)


def _series(results: pd.DataFrame, column: str) -> pd.Series:
    if column in results.columns:
        return pd.to_numeric(results[column], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(results.index), index=results.index)


def _primary_flow_status(config: dict | None, results: pd.DataFrame | None = None) -> str:
    if isinstance(results, pd.DataFrame):
        metadata = results.attrs.get("run_metadata", {})
        if isinstance(metadata, dict) and metadata.get("ratewall_primary_flow_status"):
            return str(metadata["ratewall_primary_flow_status"])
    ratewall_inputs = config.get("ratewall_input_paths", {}) if isinstance(config, dict) else {}
    if isinstance(ratewall_inputs, dict) and ratewall_inputs.get("primary_flow_to_du_file"):
        return "aggregate_cash_proxy_from_cbo_total_deficit_less_net_interest"
    input_manifest_path = config.get("input_manifest") if isinstance(config, dict) else None
    if input_manifest_path:
        try:
            manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        primary_source = str(manifest.get("source_hierarchy", {}).get("primary_flow", "")).lower()
        if "cbo" in primary_source and ("primary" in primary_source or "deficit" in primary_source):
            return "aggregate_cash_proxy_from_cbo_total_deficit_less_net_interest"
    return "simulation_fiscal_flow_to_du_proxy"


def _quarterly_summary_for_scenario(
    results: pd.DataFrame,
    scenario_id: str,
    *,
    primary_flow_status: str,
) -> pd.DataFrame:
    frame = results.copy()
    frame.insert(0, "Date", pd.to_datetime(frame.index))
    frame = add_quarter_column(frame, "Date")
    grouped = frame.groupby("quarter", sort=True)
    rows = []
    for quarter, group in grouped:
        if len(group) <= 1 and abs(float(_series(group, "TDC_Change").sum())) < 1e-12:
            continue
        fiscal = _series(group, "TDC_FiscalFlow").sum()
        principal = _series(group, "TDC_PrincipalToDU").sum()
        interest = _series(group, "TDC_InterestToDU").sum()
        auction = _series(group, "TDC_AuctionAbsorption").sum()
        secondary = _series(group, "TDC_SecondaryTrades").sum()
        other = _series(group, "TDC_Other").sum()
        principal_dn = _series(group, "TDC_PrincipalToDU_DomesticNonbank").sum()
        principal_mmf = _series(group, "TDC_PrincipalToDU_MMF").sum()
        bill_dn = _series(group, "TDC_BillDiscountInterestToDU_DomesticNonbank").sum()
        bill_mmf = _series(group, "TDC_BillDiscountInterestToDU_MMF").sum()
        coupon_dn = _series(group, "TDC_CouponInterestToDU_DomesticNonbank").sum()
        coupon_mmf = _series(group, "TDC_CouponInterestToDU_MMF").sum()
        frn_dn = _series(group, "TDC_FRNInterestToDU_DomesticNonbank").sum()
        frn_mmf = _series(group, "TDC_FRNInterestToDU_MMF").sum()
        tips_coupon_dn = _series(group, "TDC_TIPSCouponInterestToDU_DomesticNonbank").sum()
        tips_coupon_mmf = _series(group, "TDC_TIPSCouponInterestToDU_MMF").sum()
        tips_infl_dn = _series(group, "TDC_TIPSInflationCompensationToDU_DomesticNonbank").sum()
        tips_infl_mmf = _series(group, "TDC_TIPSInflationCompensationToDU_MMF").sum()
        auction_dn = _series(group, "TDC_AuctionAbsorption_DomesticNonbank").sum()
        auction_mmf = _series(group, "TDC_AuctionAbsorption_MMF").sum()
        secondary_dn = _series(group, "TDC_SecondaryTrades_DomesticNonbank").sum()
        secondary_mmf = _series(group, "TDC_SecondaryTrades_MMF").sum()
        mmf_plumbing = (
            _series(group, "TDC_PrincipalToDU_MMFPlumbing").sum()
            + _series(group, "TDC_DebtService_MMFPlumbing").sum()
            + _series(group, "TDC_AuctionAbsorption_MMFPlumbing").sum()
            + _series(group, "TDC_SecondaryTrades_MMFPlumbing").sum()
        )
        tdc_change = _series(group, "TDC_Change").sum()
        cb_remittance = _series(group, "CB_Remittance").sum()
        cb_deferred_asset_end = _series(group, "CB_DeferredAsset").iloc[-1]
        overlap = interest
        component_sum = fiscal + principal + interest + auction + secondary + other
        rows.append(
            {
                "schema_version": CONTRACT_VERSION,
                "simulation_version": "tdcsim_ratewall_contract_0_1",
                "scenario_id": scenario_id,
                "quarter": quarter,
                "tdc_change_bil": f"{tdc_change:.12f}",
                "tdc_fiscal_flow_bil": f"{fiscal:.12f}",
                "tdc_debt_service_principal_to_du_bil": f"{principal:.12f}",
                "tdc_debt_service_interest_to_du_bil": f"{interest:.12f}",
                "tdc_auction_absorption_du_bil": f"{auction:.12f}",
                "tdc_secondary_trades_bil": f"{secondary:.12f}",
                "tdc_other_bil": f"{other:.12f}",
                "principal_to_du_domestic_nonbank_bil": f"{principal_dn:.12f}",
                "principal_to_du_mmf_cash_fund_route_bil": f"{principal_mmf:.12f}",
                "bill_discount_interest_to_du_domestic_nonbank_bil": f"{bill_dn:.12f}",
                "bill_discount_interest_to_du_mmf_cash_fund_route_bil": f"{bill_mmf:.12f}",
                "coupon_interest_to_du_domestic_nonbank_bil": f"{coupon_dn:.12f}",
                "coupon_interest_to_du_mmf_cash_fund_route_bil": f"{coupon_mmf:.12f}",
                "frn_interest_to_du_domestic_nonbank_bil": f"{frn_dn:.12f}",
                "frn_interest_to_du_mmf_cash_fund_route_bil": f"{frn_mmf:.12f}",
                "tips_coupon_interest_to_du_domestic_nonbank_bil": f"{tips_coupon_dn:.12f}",
                "tips_coupon_interest_to_du_mmf_cash_fund_route_bil": f"{tips_coupon_mmf:.12f}",
                "tips_inflation_compensation_to_du_domestic_nonbank_bil": f"{tips_infl_dn:.12f}",
                "tips_inflation_compensation_to_du_mmf_cash_fund_route_bil": f"{tips_infl_mmf:.12f}",
                "auction_absorption_domestic_nonbank_bil": f"{auction_dn:.12f}",
                "auction_absorption_mmf_cash_fund_route_bil": f"{auction_mmf:.12f}",
                "secondary_trades_domestic_nonbank_bil": f"{secondary_dn:.12f}",
                "secondary_trades_mmf_cash_fund_route_bil": f"{secondary_mmf:.12f}",
                "mmf_ru_plumbing_bil": f"{mmf_plumbing:.12f}",
                "mmf_deposit_pass_through": f"{MMF_DEPOSIT_PASS_THROUGH_DEFAULT:.12f}",
                "mmf_deposit_pass_through_status": MMF_DEPOSIT_PASS_THROUGH_STATUS,
                "overlap_cashflow_bil": f"{overlap:.12f}",
                "tdc_change_ex_overlap_bil": f"{tdc_change - overlap:.12f}",
                "gross_issuance_cash_proceeds_bil": f"{_series(group, 'AuctionProceeds').sum():.12f}",
                "gross_issuance_proceeds_absorbed_by_du_bil": f"{_series(group, 'TDC_GrossIssuanceProceedsAbsorbedByDU').sum():.12f}",
                "principal_redeemed_total_bil": f"{_series(group, 'PrincipalPaid_Bonds').sum():.12f}",
                "bill_discount_interest_to_du_bil": f"{_series(group, 'TDC_BillDiscountInterestToDU').sum():.12f}",
                "coupon_interest_to_du_bil": f"{_series(group, 'TDC_CouponInterestToDU').sum():.12f}",
                "frn_interest_to_du_bil": f"{_series(group, 'TDC_FRNInterestToDU').sum():.12f}",
                "tips_coupon_interest_to_du_bil": f"{_series(group, 'TDC_TIPSCouponInterestToDU').sum():.12f}",
                "tips_inflation_compensation_to_du_bil": f"{_series(group, 'TDC_TIPSInflationCompensationToDU').sum():.12f}",
                "secondary_du_to_ru_bil": f"{_series(group, 'TDC_SecondaryDUToRU').sum():.12f}",
                "secondary_ru_to_du_bil": f"{_series(group, 'TDC_SecondaryRUToDU').sum():.12f}",
                "cb_remittance_to_tga_bil": f"{cb_remittance:.12f}",
                "cb_deferred_asset_end_bil": f"{cb_deferred_asset_end:.12f}",
                "component_sum_bil": f"{component_sum:.12f}",
                "component_sum_error_bil": f"{component_sum - tdc_change:.12f}",
                "primary_flow_status": primary_flow_status,
                "secondary_trade_status": "simulated" if abs(float(secondary)) > 1e-12 else "absent_not_imputed",
                "other_status": "explicit_zero" if abs(float(other)) <= 1e-12 else "explicit_configured",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )
    return pd.DataFrame(rows)


def _components_for_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    component_specs = [
        ("primary_fiscal_flow_to_du", "tdc_fiscal_flow_bil", "DU", "", "aggregate", "fiscal_flow", "false", "true"),
        ("principal_to_du", "tdc_debt_service_principal_to_du_bil", "Private", "", "multiple", "principal", "false", "false"),
        ("interest_to_du", "tdc_debt_service_interest_to_du_bil", "Private", "", "multiple", "interest", "true", "false"),
        ("auction_absorption_by_du", "tdc_auction_absorption_du_bil", "Private", "", "multiple", "issuance_cash_proceeds", "false", "false"),
        ("secondary_trades", "tdc_secondary_trades_bil", "Private", "", "multiple", "secondary_settlement", "false", "false"),
        ("other", "tdc_other_bil", "DU", "", "aggregate", "other_explicit_or_zero", "false", "true"),
        ("bill_discount_interest_to_du", "bill_discount_interest_to_du_bil", "Private", "", "Fixed", "bill_discount_interest", "true", "false"),
        ("coupon_interest_to_du", "coupon_interest_to_du_bil", "Private", "", "Fixed", "coupon_interest", "true", "false"),
        ("frn_interest_to_du", "frn_interest_to_du_bil", "Private", "", "FRN", "frn_interest", "true", "false"),
        ("tips_coupon_interest_to_du", "tips_coupon_interest_to_du_bil", "Private", "", "TIPS", "tips_coupon_interest", "true", "false"),
        ("tips_inflation_compensation_to_du", "tips_inflation_compensation_to_du_bil", "Private", "", "TIPS", "tips_inflation_compensation", "false", "false"),
        ("central_bank_remittance_to_tga", "cb_remittance_to_tga_bil", "CB", "", "FedInternal", "central_bank_remittance_to_tga", "false", "false"),
    ]
    split_specs = [
        ("principal_to_du", "principal", "principal_to_du_domestic_nonbank_bil", "principal_to_du_mmf_cash_fund_route_bil", "false", "true"),
        ("auction_absorption_by_du", "issuance_cash_proceeds", "auction_absorption_domestic_nonbank_bil", "auction_absorption_mmf_cash_fund_route_bil", "false", "true"),
        ("secondary_trades", "secondary_settlement", "secondary_trades_domestic_nonbank_bil", "secondary_trades_mmf_cash_fund_route_bil", "false", "true"),
        ("bill_discount_interest_to_du", "bill_discount_interest", "bill_discount_interest_to_du_domestic_nonbank_bil", "bill_discount_interest_to_du_mmf_cash_fund_route_bil", "true", "false"),
        ("coupon_interest_to_du", "coupon_interest", "coupon_interest_to_du_domestic_nonbank_bil", "coupon_interest_to_du_mmf_cash_fund_route_bil", "true", "false"),
        ("frn_interest_to_du", "frn_interest", "frn_interest_to_du_domestic_nonbank_bil", "frn_interest_to_du_mmf_cash_fund_route_bil", "true", "false"),
        ("tips_coupon_interest_to_du", "tips_coupon_interest", "tips_coupon_interest_to_du_domestic_nonbank_bil", "tips_coupon_interest_to_du_mmf_cash_fund_route_bil", "true", "false"),
        ("tips_inflation_compensation_to_du", "tips_inflation_compensation", "tips_inflation_compensation_to_du_domestic_nonbank_bil", "tips_inflation_compensation_to_du_mmf_cash_fund_route_bil", "false", "false"),
    ]

    def add_row(summary_row, *, component_key, amount, holder_bucket, holder_subbucket, security_type, cash_component, direct, tdc_default, source_key, route_class=""):
        rows.append(
            {
                "schema_version": CONTRACT_VERSION,
                "scenario_id": summary_row["scenario_id"],
                "quarter": summary_row["quarter"],
                "component_key": component_key,
                "holder_bucket": holder_bucket,
                "holder_subbucket": holder_subbucket,
                "source_holder_bucket": (
                    "domestic_nonbank"
                    if holder_subbucket == PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
                    else "mmf"
                    if holder_subbucket == PRIVATE_SUBBUCKET_MMF
                    else holder_bucket
                ),
                "ratewall_perimeter": "DU" if holder_bucket in {"DU", "Private"} else "bridge",
                "route_class": route_class,
                "security_type": security_type,
                "cash_component_key": cash_component,
                "amount_bil": f"{float(amount):.12f}",
                "deposit_pass_through": (
                    "1.000000000000"
                    if holder_subbucket == PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
                    else f"{MMF_DEPOSIT_PASS_THROUGH_DEFAULT:.12f}"
                    if holder_subbucket == PRIVATE_SUBBUCKET_MMF
                    else ""
                ),
                "ru_plumbing_pass_through": (
                    "0.000000000000"
                    if holder_subbucket == PRIVATE_SUBBUCKET_DOMESTIC_NONBANK
                    else f"{1.0 - MMF_DEPOSIT_PASS_THROUGH_DEFAULT:.12f}"
                    if holder_subbucket == PRIVATE_SUBBUCKET_MMF
                    else ""
                ),
                "mmf_split_status": MMF_DEPOSIT_PASS_THROUGH_STATUS if holder_subbucket else "",
                "sign_convention": "positive_support_or_cash_to_du;auction_absorption_negative",
                "enters_direct_interest_support": direct,
                "enters_tdc_deposit_support_default": tdc_default,
                "source_family": "tdcsim",
                "source_key": source_key,
                "observability_tier": "simulation_contract",
                "assumption_status": "assumption_mode_scenario",
                "claim_boundary": CLAIM_BOUNDARY,
            }
        )

    for _, summary_row in summary.iterrows():
        for component_key, field, holder_bucket, holder_subbucket, security_type, cash_component, direct, tdc_default in component_specs:
            amount = float(summary_row.get(field, 0.0) or 0.0)
            if abs(amount) <= 1e-12 and component_key not in {"other", "secondary_trades"}:
                continue
            add_row(
                summary_row,
                component_key=component_key,
                amount=amount,
                holder_bucket=holder_bucket,
                holder_subbucket=holder_subbucket,
                security_type=security_type,
                cash_component=cash_component,
                direct=direct,
                tdc_default=tdc_default,
                source_key="simulated_component_aggregate_memo"
                if holder_bucket == "Private"
                else "simulated_component",
            )
        for component_key, cash_component, domestic_field, mmf_field, direct, tdc_default in split_specs:
            domestic_amount = float(summary_row.get(domestic_field, 0.0) or 0.0)
            mmf_amount = float(summary_row.get(mmf_field, 0.0) or 0.0)
            if abs(domestic_amount) > 1e-12 or tdc_default == "true":
                add_row(
                    summary_row,
                    component_key=f"{component_key}__{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}",
                    amount=domestic_amount,
                    holder_bucket="Private",
                    holder_subbucket=PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                    security_type="multiple",
                    cash_component=cash_component,
                    direct=direct,
                    tdc_default=tdc_default,
                    source_key="simulated_component_private_subbucket",
                    route_class="du_deposit_funded",
                )
            if abs(mmf_amount) > 1e-12 or tdc_default == "true":
                add_row(
                    summary_row,
                    component_key=f"{component_key}__{PRIVATE_SUBBUCKET_MMF}",
                    amount=mmf_amount,
                    holder_bucket="Private",
                    holder_subbucket=PRIVATE_SUBBUCKET_MMF,
                    security_type="multiple",
                    cash_component=cash_component,
                    direct=direct,
                    tdc_default=tdc_default,
                    source_key="simulated_component_private_subbucket",
                    route_class="mmf_cash_fund_route",
                )
        plumbing_amount = float(summary_row.get("mmf_ru_plumbing_bil", 0.0) or 0.0)
        if abs(plumbing_amount) > 1e-12:
            add_row(
                summary_row,
                component_key="mmf_residual_ru_plumbing",
                amount=plumbing_amount,
                holder_bucket="Private",
                holder_subbucket=PRIVATE_SUBBUCKET_MMF,
                security_type="multiple",
                cash_component="mmf_residual_plumbing",
                direct="false",
                tdc_default="false",
                source_key="simulated_component_private_subbucket_residual_plumbing",
                route_class="mmf_cash_fund_plumbing",
            )
    return pd.DataFrame(rows)


def _source_registry_rows(config: dict | None = None) -> list[dict[str, str]]:
    config = config or {}
    rows = [
        {
            "source_family": "tdcsim",
            "source_key": "ratewall_contract_export",
            "source_status": "versioned_simulation_contract",
            "ratewall_role": "forward_tdc_mechanics",
            "central_default_eligible": "true",
            "sensitivity_only": "false",
            "binding_blocker": "",
        },
        {
            "source_family": "wamest",
            "source_key": "weak_revaluation_wam_rows",
            "source_status": "blocked_from_central_default",
            "ratewall_role": "maturity_cross_check_only",
            "central_default_eligible": "false",
            "sensitivity_only": "true",
            "binding_blocker": "weak_sector_revaluation_or_peer_fallback_wam_not_default",
        },
        {
            "source_family": "tdcmix",
            "source_key": "holder_absorption_prior_contract",
            "source_status": "prior_or_regularization_only",
            "ratewall_role": "holder_scenario_prior_not_allocation_claim",
            "central_default_eligible": "true",
            "sensitivity_only": "false",
            "binding_blocker": "",
        },
        {
            "source_family": "tdcsim_holder_bucket_limitation",
            "source_key": "mmf_collapsed_into_du_current_private_bucket",
            "source_status": "superseded_by_private_route_split",
            "ratewall_role": "legacy_audit_memo",
            "central_default_eligible": "false",
            "sensitivity_only": "true",
            "binding_blocker": "superseded_by_private_subbucket_split_contract_0_2_0",
        },
        {
            "source_family": "tdcsim_private_subbucket",
            "source_key": PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
            "source_status": "source_grounded_private_subbucket_split",
            "ratewall_role": "du_deposit_funded_private_route",
            "central_default_eligible": "true",
            "sensitivity_only": "false",
            "binding_blocker": "",
        },
        {
            "source_family": "tdcsim_private_subbucket",
            "source_key": PRIVATE_SUBBUCKET_MMF,
            "source_status": MMF_DEPOSIT_PASS_THROUGH_STATUS,
            "ratewall_role": "mmf_cash_fund_route_pass_through_0_15_residual_plumbing",
            "central_default_eligible": "true",
            "sensitivity_only": "false",
            "binding_blocker": "",
        },
    ]
    input_manifest_path = config.get("input_manifest") if isinstance(config, dict) else None
    if input_manifest_path:
        try:
            manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
            for name, meta in manifest.get("files", {}).items():
                rows.append(
                    {
                        "source_family": "tdcsim_ratewall_input_bundle",
                        "source_key": name,
                        "source_status": "source_backed_input_contract",
                        "ratewall_role": "forward_tdc_input",
                        "central_default_eligible": "true",
                        "sensitivity_only": "false",
                        "binding_blocker": f"sha256={meta.get('sha256', '')}",
                    }
                )
        except Exception:
            rows.append(
                {
                    "source_family": "tdcsim_ratewall_input_bundle",
                    "source_key": str(input_manifest_path),
                    "source_status": "input_manifest_unreadable",
                    "ratewall_role": "forward_tdc_input",
                    "central_default_eligible": "false",
                    "sensitivity_only": "true",
                    "binding_blocker": "input_manifest_unreadable",
                }
            )
    rows.extend(mapping_rows(config.get("holder_perimeter_map") if isinstance(config, dict) else None))
    if isinstance(config, dict) and config.get("private_route_sensitivity_file"):
        rows.append(
            {
                "source_family": "tdcest",
                "source_key": "tdc_tdcsim_private_route_allocation_sensitivity_v1",
                "source_status": (
                    "bounded_noncanonical_proxy_not_source_backed_private_bucket_split"
                ),
                "ratewall_role": "tdcsim_private_route_sensitivity_sidecar",
                "central_default_eligible": "false",
                "sensitivity_only": "true",
                "binding_blocker": PRIVATE_ROUTE_SENSITIVITY_BLOCKER,
            }
        )
    return rows


def _private_route_sensitivity_path(config: dict | None) -> Path | None:
    if not isinstance(config, dict):
        return None
    raw_path = config.get("private_route_sensitivity_file")
    if not raw_path:
        return None
    return Path(raw_path)


def _load_private_route_sensitivity(config: dict | None) -> pd.DataFrame:
    path = _private_route_sensitivity_path(config)
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise ValueError(f"Private route sensitivity file is missing: {path}")
    frame = pd.read_csv(path)
    required = {
        "contract_version",
        "ref_quarter",
        "object_family",
        "route_class",
        "share_lambda_0",
        "share_lambda_0_5",
        "share_lambda_1",
        "current_demand_eligible",
        "canonical_tdc_math_change",
        "source_backed_private_bucket_split_status",
        "evidence_mode_enabled",
        "holder_allocation_enabled",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Private route sensitivity file missing columns: " + ", ".join(sorted(missing))
        )

    fail_closed_columns = {
        "current_demand_eligible",
        "canonical_tdc_math_change",
        "evidence_mode_enabled",
        "holder_allocation_enabled",
    }
    for column in fail_closed_columns:
        if not frame[column].astype(str).str.lower().eq("false").all():
            raise ValueError(f"Private route sensitivity column must be false: {column}")
    if not frame["source_backed_private_bucket_split_status"].astype(str).eq(
        "not_source_backed_private_bucket_split"
    ).all():
        raise ValueError(
            "Private route sensitivity must remain not_source_backed_private_bucket_split"
        )
    if "evidence_tier" in frame.columns and not frame["evidence_tier"].astype(str).eq(
        "bounded_proxy"
    ).all():
        raise ValueError("Private route sensitivity evidence_tier must be bounded_proxy")
    if "assumption_status" in frame.columns and not frame["assumption_status"].astype(
        str
    ).isin({"bounded_assumption", "mechanical_lambda_assumption"}).all():
        raise ValueError(
            "Private route sensitivity assumption_status must remain bounded"
        )
    if "mapping_burden" in frame.columns and frame["mapping_burden"].astype(str).eq(
        "none"
    ).any():
        raise ValueError(
            "Private route sensitivity must declare a nonzero mapping burden"
        )

    out = frame.copy()
    out.insert(0, "tdcsim_contract_key", PRIVATE_ROUTE_SENSITIVITY_KEY)
    out.insert(1, "schema_version", CONTRACT_VERSION)
    out["central_default_eligible"] = "false"
    out["sensitivity_only"] = "true"
    out["does_not_modify_default_holder_perimeter"] = "true"
    out["private_remains_du_under_central_contract"] = "true"
    out["binding_blocker"] = PRIVATE_ROUTE_SENSITIVITY_BLOCKER
    out["tdcsim_allowed_use"] = "ratewall_sensitivity_metadata_export_only"
    out["tdcsim_blocked_use"] = (
        "central_holder_math;default_holder_perimeter;canonical_tdc_math;"
        "source_backed_private_bucket_split"
    )
    return out


def _mmf_pass_through_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    central_p = MMF_DEPOSIT_PASS_THROUGH_DEFAULT
    for _, row in summary.iterrows():
        effective_auction = float(row.get("auction_absorption_mmf_cash_fund_route_bil", 0.0) or 0.0)
        effective_principal = float(row.get("principal_to_du_mmf_cash_fund_route_bil", 0.0) or 0.0)
        effective_secondary = float(row.get("secondary_trades_mmf_cash_fund_route_bil", 0.0) or 0.0)
        gross_auction = effective_auction / central_p if central_p else 0.0
        gross_principal = effective_principal / central_p if central_p else 0.0
        gross_secondary = effective_secondary / central_p if central_p else 0.0
        gross_total = gross_auction + gross_principal + gross_secondary
        for rung in MMF_DEPOSIT_PASS_THROUGH_SENSITIVITY_GRID:
            delta_auction = (float(rung) - 1.0) * gross_auction
            delta_principal = (float(rung) - 1.0) * gross_principal
            delta_secondary = (float(rung) - 1.0) * gross_secondary
            rows.append(
                {
                    "schema_version": CONTRACT_VERSION,
                    "scenario_id": row["scenario_id"],
                    "quarter": row["quarter"],
                    "mmf_deposit_pass_through": f"{float(rung):.12f}",
                    "legacy_reference_pass_through": "1.000000000000",
                    "central_default_pass_through": f"{central_p:.12f}",
                    "gross_mmf_auction_absorption_bil": f"{gross_auction:.12f}",
                    "gross_mmf_principal_to_du_bil": f"{gross_principal:.12f}",
                    "gross_mmf_secondary_trades_bil": f"{gross_secondary:.12f}",
                    "gross_mmf_total_signed_bil": f"{gross_total:.12f}",
                    "tdc_delta_vs_legacy_auction_bil": f"{delta_auction:.12f}",
                    "tdc_delta_vs_legacy_principal_bil": f"{delta_principal:.12f}",
                    "tdc_delta_vs_legacy_secondary_bil": f"{delta_secondary:.12f}",
                    "tdc_delta_vs_legacy_bil": f"{delta_auction + delta_principal + delta_secondary:.12f}",
                    "mmf_residual_plumbing_bil": f"{(1.0 - float(rung)) * gross_total:.12f}",
                    "source_status": MMF_DEPOSIT_PASS_THROUGH_STATUS,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            )
    return pd.DataFrame(rows)


def export_ratewall_bundle(
    scenario_results: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    config: dict | None = None,
) -> dict[str, Path]:
    """Export the RateWall contract bundle for completed scenarios."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_frames = [
        _quarterly_summary_for_scenario(
            results,
            scenario_id,
            primary_flow_status=_primary_flow_status(config, results),
        )
        for scenario_id, results in scenario_results.items()
        if isinstance(results, pd.DataFrame) and not results.empty
    ]
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    components = _components_for_summary(summary) if not summary.empty else pd.DataFrame()
    source_registry = pd.DataFrame(_source_registry_rows(config))
    private_route_sensitivity = _load_private_route_sensitivity(config)
    mmf_pass_through_sensitivity = _mmf_pass_through_sensitivity(summary) if not summary.empty else pd.DataFrame()
    validation = validate_ratewall_contract(summary, components)
    input_artifacts = {}
    source_hierarchy = {}
    input_manifest_path = config.get("input_manifest") if isinstance(config, dict) else None
    if input_manifest_path:
        try:
            input_manifest = json.loads(Path(input_manifest_path).read_text(encoding="utf-8"))
            input_artifacts = input_manifest.get("files", {})
            source_hierarchy = input_manifest.get("source_hierarchy", {})
        except Exception:
            input_artifacts = {"input_manifest": {"status": "unreadable", "path": str(input_manifest_path)}}

    summary_path = output_path / "tdcsim_ratewall_quarterly_summary.csv"
    components_path = output_path / "tdcsim_ratewall_quarterly_components.csv"
    source_registry_path = output_path / "tdcsim_ratewall_source_registry.csv"
    private_route_sensitivity_path = (
        output_path / "tdcsim_private_route_sensitivity_contract.csv"
    )
    mmf_pass_through_sensitivity_path = output_path / "tdcsim_mmf_pass_through_sensitivity.csv"
    manifest_path = output_path / "tdcsim_ratewall_manifest.json"

    summary.to_csv(summary_path, index=False)
    components.to_csv(components_path, index=False)
    source_registry.to_csv(source_registry_path, index=False)
    if not private_route_sensitivity.empty:
        private_route_sensitivity.to_csv(private_route_sensitivity_path, index=False)
    if not mmf_pass_through_sensitivity.empty:
        mmf_pass_through_sensitivity.to_csv(mmf_pass_through_sensitivity_path, index=False)

    files = {
        "summary": summary_path.name,
        "components": components_path.name,
        "source_registry": source_registry_path.name,
    }
    if not private_route_sensitivity.empty:
        files["private_route_sensitivity"] = private_route_sensitivity_path.name
    if not mmf_pass_through_sensitivity.empty:
        files["mmf_pass_through_sensitivity"] = mmf_pass_through_sensitivity_path.name
    manifest_payload = {
        "contract_version": CONTRACT_VERSION,
        "schema": "tdcsim_ratewall_bundle",
        "scenario_ids": sorted(str(k) for k in scenario_results),
        "claim_boundary": CLAIM_BOUNDARY,
        "validation": validation,
        "input_status": {
            "yield_curve_source_status": "dynamic_surface_or_static_curve_config",
            "holder_prior_review_status": "tdcmix_priors_only_not_allocation_claim",
            "maturity_source_status": "weak_wamest_rows_sensitivity_only",
        },
        "input_artifacts": input_artifacts,
        "source_hierarchy": source_hierarchy,
        "files": files,
    }
    payload_no_hash = json.dumps(manifest_payload, sort_keys=True)
    manifest_payload["config_hash"] = hashlib.sha256(payload_no_hash.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "manifest": manifest_path,
        "summary": summary_path,
        "components": components_path,
        "source_registry": source_registry_path,
    }
    if not private_route_sensitivity.empty:
        result["private_route_sensitivity"] = private_route_sensitivity_path
    if not mmf_pass_through_sensitivity.empty:
        result["mmf_pass_through_sensitivity"] = mmf_pass_through_sensitivity_path
    return result
