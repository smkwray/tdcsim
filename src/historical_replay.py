"""Quarterly historical replay runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from historical_replay_contract import normalize_quarter_value
from historical_replay_auction import (
    build_auction_absorption_reconciliation,
    build_auction_holder_prior_panel,
    load_auction_allotment_proxy,
)
from historical_replay_event_ledger import (
    build_event_rollforward,
    build_historical_replay_event_ledger,
)
from historical_replay_interest import (
    aggregate_stock_only_interest,
    build_quarterly_interest_detail,
    build_treasury_interest_expense_diagnostic,
)
from historical_replay_loader import (
    enrich_mspd_cohorts_with_security_sources,
    load_mspd_cohorts,
    load_quarterly_cash,
    load_sector_positions,
)
from historical_replay_materializer import materialize_portfolio
from historical_replay_observations import (
    build_exact_observation_coverage,
    build_historical_replay_observations,
    build_holder_basis_bridge,
)
from historical_replay_pricing import (
    apply_model_pricing,
    build_pricing_scope_diagnostics,
    load_nominal_yield_curve,
    load_real_yield_curve,
    price_period_end_portfolios,
)
from historical_replay_solver import solve_sector_cohort_allocations
from historical_replay_soma import build_soma_fixed_allocations, load_soma_treasury_holdings
from historical_replay_tdc import (
    apply_auction_absorption_to_tdc_panel,
    build_historical_replay_tdc_panel,
    write_tdc_validation_artifacts,
)
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    PORTFOLIO_DTYPES,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_SECTOR_PREFIXES = ("total", "total_", "all_sector", "all_sector_")
NUMERIC_TOLERANCE_MIL = 1.0e-2
REQUIRED_TDCEST_REPLAY_SOURCE_KEYS = {
    "tdc_empirical_anchor",
    "tdc_tier2_regression_series",
    "tdc_du_fiscal_flow_research",
    "tier2_interest_component_candidate",
    "tier2_interest_source_constraints",
    "treasury_interest_expense",
    "ffiec_interest_constraints_normalized",
    "ncua_interest_constraints_normalized",
    "tdc_mmf_rrp_quarterly_adjustments",
    "tdc_mmf_rrp_quarterly_adjustments_sec_full",
    "method_meta",
}
RUNTIME_CONSUMED_TDCEST_SOURCE_KEYS = {
    "quarterly_inputs",
    "tdc_estimates",
    "tdc_components",
    "tdc_tier2_regression_series",
    "tdc_du_fiscal_flow_research",
    "tdc_empirical_anchor",
    "tier2_interest_component_candidate",
    "tier2_interest_source_constraints",
    "treasury_interest_expense",
    "ffiec_interest_constraints_normalized",
    "ncua_interest_constraints_normalized",
}

REPLAY_RESULT_COLUMNS = [
    "TOC_Level",
    "TOC_Change",
    "TGA",
    "TTL",
    "TGAChange",
    "TotalDebt_Agg",
    "DebtHeld_Banks",
    "DebtHeld_Private",
    "DebtHeld_CB",
    "DebtHeld_Foreign",
    "DebtHeld_FedInternal",
    "DebtHeld_TrustFunds",
    "TDC_Level",
    "TDC_Change",
    "TDC_FiscalFlow",
    "TDC_DebtService",
    "TDC_AuctionAbsorption",
    "TDC_SecondaryTrades",
    "TDC_Other",
    "TDC_Residual",
    "Z1_Holder_Total",
    "MSPD_Cohort_Total",
    "MSPD_Z1_SourceBasisDiff",
    "MSPD_Minus_IncludedRawZ1",
    "ModeledFaceEquivalentBasisResidual",
    "HolderScaleFactor",
    "Replay_CohortResidual",
    "Replay_SectorResidual",
    "HolderSurfaceStatus",
    "DebtSurfaceRole",
]

TDC_RESULT_COLUMNS = [
    "TDC_Level",
    "TDC_Change",
    "TDC_FiscalFlow",
    "TDC_DebtService",
    "TDC_AuctionAbsorption",
    "TDC_SecondaryTrades",
    "TDC_Other",
    "TDC_Residual",
]


def run_historical_replay(params: dict, start_date, end_date, scenario_name: str = "HistoricalReplay"):
    cfg = params.get("historical_replay", {})
    paths = cfg.get("paths", {})
    amount_unit_scale = float(cfg.get("amount_unit_scale", 1000.0) or 1000.0)
    sector_value_unit_scale = float(cfg.get("sector_value_unit_scale", 1.0) or 1.0)
    start_quarter = cfg.get("start_quarter") or normalize_quarter_value(start_date)
    end_quarter = cfg.get("end_quarter") or normalize_quarter_value(end_date)

    cash = load_quarterly_cash(paths["cash"], start_quarter, end_quarter)
    sectors = load_sector_positions(paths["sector_positions"], start_quarter, end_quarter)
    cohorts = load_mspd_cohorts(paths["cohorts"], start_quarter, end_quarter)
    cohorts = enrich_mspd_cohorts_with_security_sources(
        cohorts,
        auction_path=paths.get("auctions", "data/historical_replay/raw/fiscaldata/auctions_query.csv"),
        frn_daily_indexes_path=paths.get("frn_daily_indexes", "data/historical_replay/raw/fiscaldata/frn_daily_indexes.csv"),
        tips_cpi_path=paths.get("tips_cpi", "data/historical_replay/raw/fiscaldata/tips_cpi_data_detail.csv"),
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    tdc_panel, tdc_formula_crosscheck, tdc_source_manifest = _load_tdc_panel(
        cfg,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    auction_allotment_proxy, auction_absorption_reconciliation = _load_auction_absorption(
        cfg,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    cohorts, auction_use_reconciliation = _attach_auction_holder_priors(
        cohorts,
        auction_allotment_proxy,
    )
    if tdc_panel is not None and not auction_absorption_reconciliation.empty:
        tdc_panel = apply_auction_absorption_to_tdc_panel(
            tdc_panel,
            auction_absorption_reconciliation,
        )
    pricing_curve = load_nominal_yield_curve()
    real_pricing_curve = load_real_yield_curve()
    cohorts = _attach_cohort_market_value_ratios(cohorts, pricing_curve, real_pricing_curve)
    soma_holdings = load_soma_treasury_holdings(
        cfg.get(
            "soma_treasury_holdings",
            "data/historical_replay/raw/nyfed/soma_treasury_holdings_monthly.csv",
        ),
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    soma_fixed_allocations, soma_holdout_diagnostics = build_soma_fixed_allocations(
        cohorts,
        soma_holdings,
    )

    sector_observations = sectors.copy()
    if sector_value_unit_scale != 1.0:
        sector_observations["value"] = sector_observations["value"] / sector_value_unit_scale
    if "broad_holder_class" in sector_observations.columns:
        sector_observations["tdcsim_holder"] = pd.NA
        nonaggregate_sector_observations = ~_aggregate_sector_mask(sector_observations)
        sector_observations.loc[nonaggregate_sector_observations, "tdcsim_holder"] = sector_observations.loc[
            nonaggregate_sector_observations,
            "broad_holder_class",
        ].apply(_holder_for_sector)
    sector_levels = sector_observations[
        sector_observations["measure"].astype(str).str.lower().isin({"level", "stock"})
    ].copy()
    raw_sector_levels = sector_levels.copy()
    observation_registry = build_historical_replay_observations(
        sector_observations,
        auction_allotment_proxy,
        ffiec_path=cfg.get(
            "ffiec_interest_constraints",
            "data/historical_replay/imported/tdcest/ffiec_interest_constraints_normalized.csv",
        ),
        ncua_path=cfg.get(
            "ncua_interest_constraints",
            "data/historical_replay/imported/tdcest/ncua_interest_constraints_normalized.csv",
        ),
        tier2_constraints_path=cfg.get(
            "tier2_interest_constraints",
            "data/historical_replay/imported/tdcest/tier2_interest_source_constraints.csv",
        ),
        quarterly_inputs_path=cfg.get(
            "mmf_component_constraints",
            "data/historical_replay/imported/tdcest/quarterly_inputs.csv",
        ),
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    exact_observation_coverage = build_exact_observation_coverage(observation_registry)
    holder_basis_bridge = build_holder_basis_bridge(observation_registry)
    z1_transaction_flow_diagnostics = _build_z1_transaction_flow_diagnostics(
        sector_observations,
        raw_sector_levels,
    )
    source_event_ledger = build_historical_replay_event_ledger(
        cohorts,
        auction_allotment_proxy,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    event_ledger, event_rollforward, unexplained_change_ledger = build_event_rollforward(
        cohorts,
        source_event_ledger,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    sector_levels = _prepare_native_sector_levels(sector_levels)
    mmf_component_targets = _load_mmf_component_targets(
        cfg,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    sector_levels, cohorts, mmf_component_input = _apply_mmf_component_constraints(
        sector_levels,
        cohorts,
        mmf_component_targets,
    )
    sector_levels = _net_negative_native_sectors(sector_levels)
    negative_sector_netting_bridge = _build_negative_sector_netting_bridge(sector_levels)
    cohorts, maturity_prior_input = _attach_maturity_bucket_priors(cohorts, observation_registry)
    sector_levels, cohorts = _apply_maturity_bucket_proxy_constraints(
        sector_levels,
        cohorts,
        maturity_prior_input,
    )
    solver_tolerance = float(cfg.get("solver_tolerance", 1.0e-6) or 1.0e-6)
    allocations, diagnostics = _solve_by_quarter(
        cohorts,
        sector_levels,
        tolerance=solver_tolerance,
        fixed_allocations=soma_fixed_allocations,
    )
    mmf_component_reconciliation = _build_mmf_component_reconciliation(
        mmf_component_input,
        allocations,
        cohorts,
    )
    maturity_prior_reconciliation = _build_maturity_prior_reconciliation(
        maturity_prior_input,
        allocations,
        cohorts,
    )
    z1_scope_reconciliation = _build_z1_scope_reconciliation(
        raw_sector_levels,
        sector_levels,
        cohorts,
        diagnostics,
    )
    portfolio_constraint_diagnostics = _build_portfolio_constraint_diagnostics(
        sector_levels,
        diagnostics,
    )
    period_end_portfolios = _materialize_period_end_portfolios(allocations, cohorts)
    period_end_portfolios, pricing_scope_diagnostics = price_period_end_portfolios(
        period_end_portfolios,
        yield_curve=pricing_curve,
        real_yield_curve=real_pricing_curve,
    )
    portfolio_transition_diagnostics = _build_portfolio_transition_diagnostics(
        period_end_portfolios,
        z1_transaction_flow_diagnostics,
        event_rollforward,
    )
    holder_mix_differentiation = _build_holder_mix_differentiation(period_end_portfolios)
    interest_component_detail = build_quarterly_interest_detail(period_end_portfolios)
    interest_proxy_alignment = _build_interest_proxy_alignment(
        interest_component_detail,
        tdc_panel,
    )
    treasury_interest_expense_diagnostic = _build_treasury_interest_expense_diagnostic_from_manifest(
        tdc_source_manifest,
        interest_component_detail,
    )
    final_quarter = _last_quarter(period_end_portfolios.keys()) or end_quarter
    final_portfolio = period_end_portfolios.get(final_quarter)
    if final_portfolio is None:
        final_allocations = allocations[allocations["quarter"].astype(str) == str(final_quarter)].copy()
        final_cohorts = cohorts[cohorts["quarter"].astype(str) == str(final_quarter)].copy()
        final_portfolio = materialize_portfolio(final_allocations, final_cohorts)
        final_portfolio = price_period_end_portfolios(
            {str(final_quarter): final_portfolio},
            yield_curve=pricing_curve,
            real_yield_curve=real_pricing_curve,
        )[0].get(str(final_quarter), final_portfolio)
    pricing_scope_diagnostics = pd.concat(
        [
            pricing_scope_diagnostics,
            build_pricing_scope_diagnostics({}, final_portfolio),
        ],
        ignore_index=True,
    )

    results = _build_results(
        cash,
        allocations,
        diagnostics,
        tdc_panel=tdc_panel,
        amount_unit_scale=amount_unit_scale,
    )
    replay_ledger = _build_replay_ledger(cash, results, tdc_panel=tdc_panel)
    results.attrs["period_end_portfolios"] = period_end_portfolios
    results.attrs["replay_ledger"] = replay_ledger
    results.attrs["replay_diagnostics"] = diagnostics.copy()
    results.attrs["portfolio_constraint_diagnostics"] = portfolio_constraint_diagnostics.copy()
    results.attrs["z1_scope_reconciliation"] = z1_scope_reconciliation.copy()
    results.attrs["observation_registry"] = observation_registry.copy()
    results.attrs["exact_observation_coverage"] = exact_observation_coverage.copy()
    results.attrs["holder_basis_bridge"] = holder_basis_bridge.copy()
    results.attrs["z1_transaction_flow_diagnostics"] = z1_transaction_flow_diagnostics.copy()
    results.attrs["portfolio_transition_diagnostics"] = portfolio_transition_diagnostics.copy()
    results.attrs["maturity_prior_reconciliation"] = maturity_prior_reconciliation.copy()
    results.attrs["mmf_component_reconciliation"] = mmf_component_reconciliation.copy()
    results.attrs["negative_sector_netting_bridge"] = negative_sector_netting_bridge.copy()
    results.attrs["historical_event_ledger"] = event_ledger.copy()
    results.attrs["historical_event_rollforward"] = event_rollforward.copy()
    results.attrs["historical_unexplained_change_ledger"] = unexplained_change_ledger.copy()
    results.attrs["pricing_scope_diagnostics"] = pricing_scope_diagnostics.copy()
    results.attrs["auction_use_reconciliation"] = auction_use_reconciliation.copy()
    results.attrs["holder_mix_differentiation"] = holder_mix_differentiation.copy()
    results.attrs["interest_component_detail"] = interest_component_detail.copy()
    results.attrs["interest_proxy_alignment"] = interest_proxy_alignment.copy()
    results.attrs["treasury_interest_expense_diagnostic"] = treasury_interest_expense_diagnostic.copy()
    results.attrs["auction_allotment_proxy"] = auction_allotment_proxy.copy()
    results.attrs["auction_absorption_reconciliation"] = auction_absorption_reconciliation.copy()
    results.attrs["soma_holdings"] = soma_holdings.copy()
    results.attrs["soma_fixed_allocations"] = soma_fixed_allocations.copy()
    results.attrs["soma_holdout_diagnostics"] = soma_holdout_diagnostics.copy()
    strips_scope_diagnostics = _build_strips_scope_diagnostics(
        cohorts=cohorts,
        period_end_portfolios=period_end_portfolios,
        final_portfolio=final_portfolio,
    )
    valuation_scope_diagnostics = _build_valuation_scope_diagnostics(
        period_end_portfolios=period_end_portfolios,
        final_portfolio=final_portfolio,
    )
    results.attrs["strips_scope_diagnostics"] = strips_scope_diagnostics.copy()
    results.attrs["valuation_scope_diagnostics"] = valuation_scope_diagnostics.copy()
    if tdc_panel is not None:
        results.attrs["tdc_panel"] = tdc_panel.copy()
        results.attrs["tdc_formula_crosscheck"] = tdc_formula_crosscheck.copy()
        results.attrs["tdc_source_manifest"] = tdc_source_manifest.copy()
    replay_input_manifest = _build_replay_input_manifest(cfg, tdc_source_manifest)
    results.attrs["replay_input_manifest"] = replay_input_manifest.copy()

    export_paths = _export_replay_outputs(
        cfg,
        results=results,
        ledger=replay_ledger,
        diagnostics=diagnostics,
        portfolio_constraint_diagnostics=portfolio_constraint_diagnostics,
        z1_scope_reconciliation=z1_scope_reconciliation,
        observation_registry=observation_registry,
        exact_observation_coverage=exact_observation_coverage,
        holder_basis_bridge=holder_basis_bridge,
        z1_transaction_flow_diagnostics=z1_transaction_flow_diagnostics,
        portfolio_transition_diagnostics=portfolio_transition_diagnostics,
        maturity_prior_reconciliation=maturity_prior_reconciliation,
        mmf_component_reconciliation=mmf_component_reconciliation,
        negative_sector_netting_bridge=negative_sector_netting_bridge,
        event_ledger=event_ledger,
        event_rollforward=event_rollforward,
        unexplained_change_ledger=unexplained_change_ledger,
        pricing_scope_diagnostics=pricing_scope_diagnostics,
        auction_use_reconciliation=auction_use_reconciliation,
        holder_mix_differentiation=holder_mix_differentiation,
        interest_component_detail=interest_component_detail,
        interest_proxy_alignment=interest_proxy_alignment,
        treasury_interest_expense_diagnostic=treasury_interest_expense_diagnostic,
        auction_allotment_proxy=auction_allotment_proxy,
        auction_absorption_reconciliation=auction_absorption_reconciliation,
        soma_holdings=soma_holdings,
        soma_fixed_allocations=soma_fixed_allocations,
        soma_holdout_diagnostics=soma_holdout_diagnostics,
        strips_scope_diagnostics=strips_scope_diagnostics,
        valuation_scope_diagnostics=valuation_scope_diagnostics,
        period_end_portfolios=period_end_portfolios,
        final_portfolio=final_portfolio,
        tdc_panel=tdc_panel,
        tdc_formula_crosscheck=tdc_formula_crosscheck,
        tdc_source_manifest=tdc_source_manifest,
        replay_input_manifest=replay_input_manifest,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    run_metadata = {
        "scenario_name": scenario_name,
        "mode": "historical_replay",
        "start_quarter": start_quarter,
        "end_quarter": end_quarter,
        "replay_quarters": int(len(results.index)),
        "amount_unit_scale": amount_unit_scale,
        "sector_value_unit_scale": sector_value_unit_scale,
        "historical_replay_status": "quarterly_observed_aggregate_replay",
    }
    if export_paths:
        run_metadata["historical_replay_output_paths"] = export_paths
    results.attrs["run_metadata"] = run_metadata
    return results, final_portfolio


def _load_tdc_panel(
    cfg: dict,
    *,
    start_quarter: str,
    end_quarter: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame, pd.DataFrame]:
    tdc_cfg = cfg.get("tdc")
    if tdc_cfg is False:
        return None, pd.DataFrame(), pd.DataFrame()
    source_overrides = {}
    if isinstance(tdc_cfg, dict):
        source_overrides.update(tdc_cfg.get("sources", {}) or {})
    try:
        transfer_share = 0.01
        if isinstance(tdc_cfg, dict):
            transfer_share = float(
                tdc_cfg.get(
                    "treasury_to_ru_transfer_share_of_deficit",
                    cfg.get("treasury_to_ru_transfer_share_of_deficit", 0.01),
                )
            )
        else:
            transfer_share = float(cfg.get("treasury_to_ru_transfer_share_of_deficit", 0.01))
        return build_historical_replay_tdc_panel(
            source_overrides=source_overrides or None,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
            treasury_to_ru_transfer_share_of_deficit=transfer_share,
        )
    except FileNotFoundError:
        if isinstance(tdc_cfg, dict) and tdc_cfg.get("required", False):
            raise
        return None, pd.DataFrame(), pd.DataFrame()


def _load_auction_absorption(
    cfg: dict,
    *,
    start_quarter: str,
    end_quarter: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    auction_cfg = cfg.get("auction_allotment_proxy", {})
    auction_paths = auction_cfg if isinstance(auction_cfg, dict) else {}
    try:
        proxy = load_auction_allotment_proxy(
            allotment_panel_path=auction_paths.get(
                "allotment_panel",
                "data/historical_replay/imported/buycurve/auction_allotment_panel_base_slim.csv",
            ),
            auction_terms_path=auction_paths.get(
                "auction_terms",
                "data/historical_replay/raw/fiscaldata/auctions_query.csv",
            ),
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        )
    except FileNotFoundError:
        if cfg.get("auction_allotment_proxy_required", False):
            raise
        return pd.DataFrame(), pd.DataFrame()
    return proxy, build_auction_absorption_reconciliation(proxy)


def _attach_auction_holder_priors(
    cohorts: pd.DataFrame,
    auction_allotment_proxy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cohorts.empty or auction_allotment_proxy.empty:
        return cohorts.copy(), _build_auction_use_reconciliation(cohorts, pd.DataFrame())
    prior_panel = build_auction_holder_prior_panel(auction_allotment_proxy)
    if prior_panel.empty:
        return cohorts.copy(), _build_auction_use_reconciliation(cohorts, pd.DataFrame())
    required = {"cusip", "issue_date", "maturity_date"}
    if not required.issubset(cohorts.columns):
        return cohorts.copy(), _build_auction_use_reconciliation(cohorts, pd.DataFrame())
    working = cohorts.copy()
    for column in required:
        working[column] = working[column].where(working[column].notna(), "").astype(str).str.strip()
    working["cusip"] = working["cusip"].str.upper()
    merged = working.merge(
        prior_panel,
        on=["cusip", "issue_date", "maturity_date"],
        how="left",
        validate="many_to_one",
    )
    prior_cols = [f"prior_holder_{holder}" for holder in ["Banks", "Private", "CB", "Foreign"]]
    for column in prior_cols:
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(0.0)
    merged["auction_prior_total_amount"] = pd.to_numeric(
        merged.get("auction_prior_total_amount"),
        errors="coerce",
    ).fillna(0.0)
    merged["auction_prior_holder_count"] = pd.to_numeric(
        merged.get("auction_prior_holder_count"),
        errors="coerce",
    ).fillna(0).astype(int)
    merged["auction_prior_status"] = merged.get(
        "auction_prior_status",
        pd.Series(index=merged.index, dtype=object),
    ).fillna("no_source_backed_auction_holder_prior")
    return merged, _build_auction_use_reconciliation(merged, prior_panel)


def _build_auction_use_reconciliation(
    cohorts: pd.DataFrame,
    prior_panel: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "cohort_rows",
        "cohort_rows_with_auction_prior",
        "cohort_outstanding",
        "cohort_outstanding_with_auction_prior",
        "auction_prior_coverage_share",
        "source_prior_security_keys",
        "matched_prior_security_keys",
        "unmatched_source_prior_security_keys",
        "status",
    ]
    if cohorts.empty:
        return pd.DataFrame(columns=columns)
    working = cohorts.copy()
    working["outstanding"] = pd.to_numeric(working.get("outstanding"), errors="coerce").fillna(0.0)
    if "auction_prior_total_amount" not in working.columns:
        working["auction_prior_total_amount"] = 0.0
    working["_has_auction_prior"] = pd.to_numeric(
        working["auction_prior_total_amount"],
        errors="coerce",
    ).fillna(0.0) > 0.0
    source_keys = int(len(prior_panel.index)) if not prior_panel.empty else 0
    matched_keys = 0
    if source_keys and {"cusip", "issue_date", "maturity_date"}.issubset(working.columns):
        matched_keys = int(
            working.loc[working["_has_auction_prior"], ["cusip", "issue_date", "maturity_date"]]
            .drop_duplicates()
            .shape[0]
        )
    rows = []
    for quarter, group in working.groupby("quarter", sort=True, dropna=False):
        outstanding = float(group["outstanding"].sum())
        with_prior = float(group.loc[group["_has_auction_prior"], "outstanding"].sum())
        rows.append(
            {
                "quarter": str(quarter),
                "cohort_rows": int(len(group.index)),
                "cohort_rows_with_auction_prior": int(group["_has_auction_prior"].sum()),
                "cohort_outstanding": outstanding,
                "cohort_outstanding_with_auction_prior": with_prior,
                "auction_prior_coverage_share": with_prior / outstanding if outstanding else 0.0,
                "source_prior_security_keys": source_keys,
                "matched_prior_security_keys": matched_keys,
                "unmatched_source_prior_security_keys": max(source_keys - matched_keys, 0),
                "status": "auction_priors_applied" if with_prior > 0.0 else "no_matching_auction_prior",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _solve_by_quarter(
    cohorts: pd.DataFrame,
    sector_levels: pd.DataFrame,
    *,
    tolerance: float = 1.0e-6,
    fixed_allocations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocation_frames = []
    diagnostic_frames = []
    quarters = sorted(set(cohorts["quarter"]).intersection(set(sector_levels["quarter"])))
    prior_allocations: pd.DataFrame | None = None
    for idx, quarter in enumerate(quarters, start=1):
        print(f"solving historical replay quarter {quarter} ({idx}/{len(quarters)})", flush=True)
        q_cohorts = cohorts[cohorts["quarter"].astype(str).eq(str(quarter))].copy()
        q_sector_levels = sector_levels[sector_levels["quarter"].astype(str).eq(str(quarter))].copy()
        q_fixed_allocations = (
            fixed_allocations[fixed_allocations["quarter"].astype(str).eq(str(quarter))].copy()
            if fixed_allocations is not None and not fixed_allocations.empty and "quarter" in fixed_allocations.columns
            else None
        )
        q_allocations, q_diagnostics = solve_sector_cohort_allocations(
            _solver_cohort_view(q_cohorts),
            q_sector_levels,
            quarter=quarter,
            tolerance=tolerance,
            prior_allocations=prior_allocations,
            fixed_allocations=q_fixed_allocations,
        )
        allocation_frames.append(q_allocations)
        diagnostic_frames.append(q_diagnostics)
        if not q_allocations.empty:
            prior_allocations = q_allocations
        else:
            prior_allocations = None
    allocations = (
        pd.concat(allocation_frames, ignore_index=True)
        if allocation_frames
        else pd.DataFrame(columns=["quarter", "sector", "cohort_id", "allocated_outstanding"])
    )
    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame(columns=["quarter", "diagnostic_type", "subject", "residual"])
    )
    return allocations, diagnostics


def _solver_cohort_view(cohorts: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["quarter", "cohort_id", "outstanding", "market_value_ratio"]
    prior_columns = [
        column
        for column in cohorts.columns
        if str(column).startswith(("prior_", "prior_holder_", "soft_target_", "softtarget"))
        or str(column).startswith(("eligible_", "eligibility_"))
        or str(column) in {"prior_weight", "soft_target_weight", "weight"}
    ]
    columns = [column for column in [*base_columns, *prior_columns] if column in cohorts.columns]
    return cohorts[columns].copy()


def _attach_maturity_bucket_priors(
    cohorts: pd.DataFrame,
    observation_registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "quarter",
        "source_scope",
        "target_sector",
        "maturity_bucket",
        "observed_value_mil",
        "observed_share",
        "cohort_supply_mil",
        "cohort_supply_share",
        "prior_weight",
        "prior_column",
        "prior_status",
        "constraint_role",
        "coverage_note",
    ]
    if cohorts.empty or observation_registry.empty:
        return cohorts.copy(), pd.DataFrame(columns=columns)
    obs = observation_registry[
        observation_registry["scope"].astype(str).isin(
            {"ffiec_bank_maturity_prior", "ncua_credit_union_maturity_proxy"}
        )
        & observation_registry["maturity_bucket"].notna()
    ].copy()
    if obs.empty:
        return cohorts.copy(), pd.DataFrame(columns=columns)

    working = cohorts.copy()
    working["outstanding"] = pd.to_numeric(working.get("outstanding"), errors="coerce").fillna(0.0)
    rows = []
    for (quarter, scope), group in obs.groupby(["quarter", "scope"], sort=True, dropna=False):
        quarter = str(quarter)
        scope = str(scope)
        target_sector = (
            "us_chartered_depository_institutions"
            if scope == "ffiec_bank_maturity_prior"
            else "credit_unions"
        )
        prior_column = f"prior_{target_sector}"
        if prior_column not in working.columns:
            working[prior_column] = 1.0
        q_mask = working["quarter"].astype(str).eq(quarter)
        q_cohorts = working.loc[q_mask].copy()
        if q_cohorts.empty:
            continue
        q_cohorts["_maturity_bucket"] = _maturity_bucket_for_scope(q_cohorts, quarter=quarter, scope=scope)
        source = (
            group.assign(reported_value_mil=pd.to_numeric(group["reported_value_mil"], errors="coerce").fillna(0.0))
            .groupby("maturity_bucket", sort=False)["reported_value_mil"]
            .sum()
        )
        source_total = float(source.sum())
        supply = q_cohorts.groupby("_maturity_bucket", sort=False)["outstanding"].sum()
        supply_total = float(supply.sum())
        if source_total <= 0.0 or supply_total <= 0.0:
            continue
        bucket_weights: dict[str, float] = {}
        for bucket, observed_value in source.items():
            bucket = str(bucket)
            observed_share = float(observed_value) / source_total
            cohort_supply = float(supply.get(bucket, 0.0))
            cohort_supply_share = cohort_supply / supply_total if supply_total else 0.0
            prior_ratio = observed_share / cohort_supply_share if cohort_supply_share > 0.0 else 0.0
            prior_power = 1.0 if scope == "ffiec_bank_maturity_prior" else 0.25
            prior_weight = prior_ratio**prior_power
            prior_weight = max(prior_weight, 1.0e-9)
            bucket_weights[bucket] = prior_weight
            is_hard_bucket_constraint = scope == "ffiec_bank_maturity_prior"
            rows.append(
                {
                    "quarter": quarter,
                    "source_scope": scope,
                    "target_sector": target_sector,
                    "maturity_bucket": bucket,
                    "observed_value_mil": float(observed_value),
                    "observed_share": observed_share,
                    "cohort_supply_mil": cohort_supply,
                    "cohort_supply_share": cohort_supply_share,
                    "prior_weight": prior_weight,
                    "prior_column": prior_column,
                    "prior_status": (
                        "solver_prior_applied"
                        if is_hard_bucket_constraint and cohort_supply > 0.0
                        else "soft_solver_prior_applied"
                        if cohort_supply > 0.0
                        else "no_current_cohort_supply"
                    ),
                    "constraint_role": (
                        "hard_bucket_proxy_constraint"
                        if is_hard_bucket_constraint
                        else "soft_all_investment_prior_only"
                    ),
                    "coverage_note": (
                        "FFIEC bank maturity ladder prior; population/basis differs from Z1 sector stock."
                        if scope == "ffiec_bank_maturity_prior"
                        else "NCUA credit-union investment maturity proxy; fallback split, not exact Treasury holdings."
                    ),
                }
            )
        for idx, bucket in q_cohorts["_maturity_bucket"].items():
            working.loc[idx, prior_column] = bucket_weights.get(str(bucket), 1.0e-9)
    return working, pd.DataFrame(rows, columns=columns)


def _apply_maturity_bucket_proxy_constraints(
    sector_levels: pd.DataFrame,
    cohorts: pd.DataFrame,
    prior_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sector_levels.empty or cohorts.empty or prior_rows.empty:
        return sector_levels.copy(), cohorts.copy()
    required = {"quarter", "source_scope", "target_sector", "maturity_bucket", "observed_share", "prior_status"}
    if not required.issubset(prior_rows.columns):
        return sector_levels.copy(), cohorts.copy()

    constraint_role = prior_rows.get(
        "constraint_role",
        pd.Series("hard_bucket_proxy_constraint", index=prior_rows.index),
    ).astype(str)
    active = prior_rows[
        prior_rows["prior_status"].astype(str).eq("solver_prior_applied")
        & constraint_role.eq("hard_bucket_proxy_constraint")
    ].copy()
    if active.empty:
        return sector_levels.copy(), cohorts.copy()
    active["observed_share"] = pd.to_numeric(active["observed_share"], errors="coerce").fillna(0.0)

    out_sectors = sector_levels.copy()
    out_cohorts = cohorts.copy()
    sector_stock_col = next(
        (
            column
            for column in ("sector_stock", "sector_outstanding", "stock", "outstanding", "amount", "value", "holding")
            if column in out_sectors.columns
        ),
        None,
    )
    if sector_stock_col is None:
        return sector_levels.copy(), cohorts.copy()

    replacement_rows = []
    drop_index: list[int] = []
    for (quarter, scope, target_sector), group in active.groupby(
        ["quarter", "source_scope", "target_sector"],
        sort=False,
        dropna=False,
    ):
        quarter = str(quarter)
        scope = str(scope)
        target_sector = str(target_sector)
        sector_mask = (
            out_sectors["quarter"].astype(str).eq(quarter)
            & out_sectors["sector"].astype(str).eq(target_sector)
        )
        if not sector_mask.any():
            continue
        base_rows = out_sectors.loc[sector_mask]
        drop_index.extend(base_rows.index.tolist())

        q_mask = out_cohorts["quarter"].astype(str).eq(quarter)
        q_buckets = _maturity_bucket_for_scope(out_cohorts.loc[q_mask], quarter=quarter, scope=scope)
        for bucket in group["maturity_bucket"].dropna().astype(str).unique():
            eligibility_col = f"eligible_{target_sector}__{bucket}"
            if eligibility_col not in out_cohorts.columns:
                out_cohorts[eligibility_col] = False
            out_cohorts.loc[q_mask, eligibility_col] = q_buckets.astype(str).eq(bucket).to_numpy(dtype=bool)

        total_share = float(group["observed_share"].sum())
        if total_share <= 0.0:
            continue
        for _, base_row in base_rows.iterrows():
            for _, prior_row in group.iterrows():
                share = float(prior_row["observed_share"]) / total_share
                bucket = str(prior_row["maturity_bucket"])
                new_row = base_row.copy()
                new_row["sector"] = f"{target_sector}__{bucket}"
                amount_columns = {
                    sector_stock_col,
                    "sector_stock",
                    "sector_outstanding",
                    "raw_z1_level",
                    "sector_target_before_scale",
                }
                for amount_col in amount_columns:
                    if amount_col in new_row.index:
                        value = pd.to_numeric(pd.Series([new_row[amount_col]]), errors="coerce").fillna(0.0).iloc[0]
                        new_row[amount_col] = float(value) * share
                new_row["native_sector"] = target_sector
                new_row["source_status"] = "maturity_bucket_proxy_constraint"
                new_row["evidence_label"] = str(prior_row["source_scope"])
                new_row["sector_adjustment_status"] = "maturity_bucket_proxy_split"
                replacement_rows.append(new_row.to_dict())

    if drop_index:
        out_sectors = out_sectors.drop(index=drop_index)
    if replacement_rows:
        out_sectors = pd.concat([out_sectors, pd.DataFrame(replacement_rows)], ignore_index=True, sort=False)
    return out_sectors, out_cohorts


def _build_maturity_prior_reconciliation(
    prior_rows: pd.DataFrame,
    allocations: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(prior_rows.columns) + [
        "modeled_allocated_mil",
        "modeled_share",
        "modeled_minus_observed_share",
        "post_solve_status",
    ]
    if prior_rows.empty:
        return pd.DataFrame(columns=columns)
    if allocations.empty or cohorts.empty:
        out = prior_rows.copy()
        out["modeled_allocated_mil"] = 0.0
        out["modeled_share"] = 0.0
        out["modeled_minus_observed_share"] = -pd.to_numeric(out["observed_share"], errors="coerce").fillna(0.0)
        out["post_solve_status"] = "no_allocations"
        return out[columns]
    cohort_meta = cohorts[["quarter", "cohort_id"]].drop_duplicates().copy()
    modeled_frames = []
    for (quarter, scope, target_sector), group in prior_rows.groupby(
        ["quarter", "source_scope", "target_sector"],
        sort=False,
        dropna=False,
    ):
        q_cohorts = cohorts[cohorts["quarter"].astype(str).eq(str(quarter))].copy()
        if q_cohorts.empty:
            continue
        q_cohorts = q_cohorts[["quarter", "cohort_id"]].drop_duplicates().copy()
        q_cohorts["maturity_bucket"] = _maturity_bucket_for_scope(q_cohorts.merge(
            cohorts[cohorts["quarter"].astype(str).eq(str(quarter))].drop_duplicates("cohort_id"),
            on=["quarter", "cohort_id"],
            how="left",
        ), quarter=str(quarter), scope=str(scope)).to_numpy()
        q_alloc = allocations[allocations["quarter"].astype(str).eq(str(quarter))].copy()
        if "native_sector" in q_alloc.columns:
            q_alloc = q_alloc[q_alloc["native_sector"].astype(str).eq(str(target_sector))].copy()
        else:
            q_alloc = q_alloc[q_alloc["sector"].astype(str).eq(str(target_sector))].copy()
        if q_alloc.empty:
            continue
        q_alloc = q_alloc.merge(q_cohorts, on=["quarter", "cohort_id"], how="left")
        q_alloc["allocated_outstanding"] = pd.to_numeric(
            q_alloc["allocated_outstanding"],
            errors="coerce",
        ).fillna(0.0)
        modeled = q_alloc.groupby("maturity_bucket", dropna=False, sort=False)["allocated_outstanding"].sum()
        modeled_total = float(modeled.sum())
        modeled_frame = modeled.reset_index().rename(columns={"allocated_outstanding": "modeled_allocated_mil"})
        modeled_frame["quarter"] = str(quarter)
        modeled_frame["source_scope"] = str(scope)
        modeled_frame["target_sector"] = str(target_sector)
        modeled_frame["modeled_share"] = modeled_frame["modeled_allocated_mil"] / modeled_total if modeled_total else 0.0
        modeled_frames.append(modeled_frame)
    modeled_all = pd.concat(modeled_frames, ignore_index=True) if modeled_frames else pd.DataFrame()
    out = prior_rows.copy()
    if not modeled_all.empty:
        out = out.merge(
            modeled_all,
            on=["quarter", "source_scope", "target_sector", "maturity_bucket"],
            how="left",
        )
    else:
        out["modeled_allocated_mil"] = 0.0
        out["modeled_share"] = 0.0
    out["modeled_allocated_mil"] = pd.to_numeric(out.get("modeled_allocated_mil"), errors="coerce").fillna(0.0)
    out["modeled_share"] = pd.to_numeric(out.get("modeled_share"), errors="coerce").fillna(0.0)
    out["modeled_minus_observed_share"] = out["modeled_share"] - pd.to_numeric(
        out["observed_share"],
        errors="coerce",
    ).fillna(0.0)
    out["post_solve_status"] = out["prior_status"].where(
        out["modeled_allocated_mil"].gt(0.0),
        "prior_bucket_unallocated_or_no_supply",
    )
    return out[columns]


def _maturity_bucket_for_scope(cohorts: pd.DataFrame, *, quarter: str, scope: str) -> pd.Series:
    maturity_col = "maturity_date" if "maturity_date" in cohorts.columns else "MaturityDate"
    maturity = pd.to_datetime(cohorts.get(maturity_col), errors="coerce")
    q_end = pd.Period(str(quarter), freq="Q").end_time.normalize()
    years = ((maturity - q_end).dt.days / 365.25).clip(lower=0.0)
    if scope == "ffiec_bank_maturity_prior":
        return years.map(_ffiec_bucket_from_years)
    return years.map(_ncua_bucket_from_years)


def _ffiec_bucket_from_years(years: float) -> str:
    if pd.isna(years):
        return "unknown"
    if years <= 0.25:
        return "le_3m"
    if years <= 1.0:
        return "3m_1y"
    if years <= 3.0:
        return "1_3y"
    if years <= 5.0:
        return "3_5y"
    if years <= 15.0:
        return "5_15y"
    return "over_15y"


def _ncua_bucket_from_years(years: float) -> str:
    if pd.isna(years):
        return "unknown"
    if years <= 1.0:
        return "le_1y"
    if years <= 3.0:
        return "1_3y"
    if years <= 5.0:
        return "3_5y"
    if years <= 10.0:
        return "5_10y"
    return "over_10y"


def _attach_cohort_market_value_ratios(
    cohorts: pd.DataFrame,
    pricing_curve: pd.DataFrame,
    real_pricing_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cohorts.empty:
        out = cohorts.copy()
        out["market_value_ratio"] = pd.Series(dtype=float)
        return out
    frames = []
    for quarter, group in cohorts.groupby("quarter", sort=False, dropna=False):
        working = group.copy()
        outstanding = pd.to_numeric(working.get("outstanding"), errors="coerce").fillna(0.0)
        probe_allocations = pd.DataFrame(
            {
                "quarter": working.get("quarter", pd.Series(str(quarter), index=working.index)).astype(str).to_numpy(),
                "sector": "PricingProbe",
                "cohort_id": working["cohort_id"].to_numpy(),
                "allocated_outstanding": outstanding.to_numpy(dtype=float),
            }
        )
        probe_portfolio = materialize_portfolio(probe_allocations, working)
        priced = apply_model_pricing(
            probe_portfolio,
            quarter=str(quarter),
            yield_curve=pricing_curve,
            real_yield_curve=real_pricing_curve,
        )
        priced["FaceValue"] = pd.to_numeric(priced.get("FaceValue"), errors="coerce").fillna(0.0)
        priced["DirtyValue"] = pd.to_numeric(priced.get("DirtyValue"), errors="coerce").fillna(0.0)
        ratio_rows = []
        for cohort_id, cohort_frame in priced.groupby("cohort_id", sort=False, dropna=False):
            face_total = float(cohort_frame["FaceValue"].sum())
            dirty_total = float(cohort_frame["DirtyValue"].sum())
            ratio_rows.append(
                {
                    "cohort_id": cohort_id,
                    "market_value_ratio": dirty_total / face_total if face_total > 0.0 else 1.0,
                }
            )
        ratios = pd.DataFrame(ratio_rows)
        working = working.merge(ratios, on="cohort_id", how="left")
        working["market_value_ratio"] = (
            pd.to_numeric(working["market_value_ratio"], errors="coerce").fillna(1.0).clip(lower=1e-9)
        )
        frames.append(working)
    return pd.concat(frames, ignore_index=True) if frames else cohorts.copy()


def _materialize_period_end_portfolios(allocations: pd.DataFrame, cohorts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    snapshots: dict[str, pd.DataFrame] = {}
    if allocations.empty:
        return snapshots
    for quarter in _sort_quarters(allocations["quarter"].dropna().astype(str).unique()):
        q_allocations = allocations[allocations["quarter"].astype(str) == quarter].copy()
        q_cohorts = cohorts[cohorts["quarter"].astype(str) == quarter].copy()
        snapshots[quarter] = materialize_portfolio(q_allocations, q_cohorts)
    return snapshots


def _prepare_native_sector_levels(sector_levels: pd.DataFrame) -> pd.DataFrame:
    if sector_levels.empty:
        return sector_levels.copy()
    frame = sector_levels.copy()
    if "native_sector" not in frame.columns:
        frame["native_sector"] = frame["sector"]
    if "broad_holder_class" not in frame.columns:
        frame["broad_holder_class"] = frame["sector"]
    aggregate_mask = _aggregate_sector_mask(frame)
    frame = frame.loc[~aggregate_mask].copy()
    frame["tdcsim_holder"] = frame["broad_holder_class"].apply(_holder_for_sector)
    frame["tdcsim_holder_subbucket"] = frame.apply(_private_subbucket_for_sector, axis=1)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").fillna(0.0)
    frame["raw_z1_level"] = frame["value"]
    frame["sector_target_before_scale"] = frame["value"]
    if "valuation_basis" not in frame.columns:
        frame["valuation_basis"] = frame.apply(_z1_valuation_basis, axis=1)
    frame["sector_adjustment_status"] = "observed_nonnegative"
    return frame


def _z1_valuation_basis(row: pd.Series) -> str:
    z1_series = str(row.get("z1_series", row.get("z1_code", ""))).strip().upper()
    if z1_series.startswith("LM"):
        return "z1_market_value_level"
    if z1_series.startswith("FL"):
        return "z1_book_or_par_level"
    if z1_series.startswith("FA") or str(row.get("measure", "")).lower() in {"flow", "transaction"}:
        return "z1_transaction_flow"
    return "published_aggregate_level"


def _aggregate_sector_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for column in ("sector", "native_sector", "broad_holder_class"):
        if column not in frame.columns:
            continue
        text = frame[column].where(frame[column].notna(), "").astype(str).str.strip().str.lower()
        mask |= text.eq("total") | text.str.startswith("total_") | text.str.startswith("all_sector_")
    return mask


def _build_z1_scope_reconciliation(
    raw_sector_levels: pd.DataFrame,
    cleaned_sector_levels: pd.DataFrame,
    cohorts: pd.DataFrame,
    solver_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "raw_z1_level_total",
        "included_granular_z1_total",
        "excluded_aggregate_z1_total",
        "all_sector_treasury_assets_control",
        "mspd_cohort_total",
        "mspd_minus_included_z1_total",
        "holder_scale_factor",
        "aggregate_rows_entered_solver",
        "status",
    ]
    if raw_sector_levels.empty:
        return pd.DataFrame(columns=columns)

    raw = raw_sector_levels.copy()
    if "native_sector" not in raw.columns:
        raw["native_sector"] = raw.get("sector", "")
    if "broad_holder_class" not in raw.columns:
        raw["broad_holder_class"] = raw.get("sector", "")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce").fillna(0.0)
    raw["_is_aggregate_control"] = _aggregate_sector_mask(raw)

    cleaned = cleaned_sector_levels.copy()
    if not cleaned.empty:
        cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce").fillna(0.0)
    cohort_totals = (
        cohorts.assign(outstanding=pd.to_numeric(cohorts.get("outstanding"), errors="coerce").fillna(0.0))
        .groupby("quarter", dropna=False, sort=False)["outstanding"]
        .sum()
        if not cohorts.empty and "quarter" in cohorts.columns
        else pd.Series(dtype=float)
    )
    total_diag = _total_balance_by_quarter(solver_diagnostics)

    quarters = _sort_quarters(set(raw["quarter"].dropna().astype(str)) | set(cleaned.get("quarter", pd.Series(dtype=object)).dropna().astype(str)))
    rows = []
    for quarter in quarters:
        q_raw = raw[raw["quarter"].astype(str) == quarter]
        q_clean = cleaned[cleaned["quarter"].astype(str) == quarter] if not cleaned.empty else pd.DataFrame()
        q_aggregate = q_raw[q_raw["_is_aggregate_control"]]
        all_sector_control = q_raw[
            q_raw["native_sector"].where(q_raw["native_sector"].notna(), "").astype(str).str.strip().str.lower().eq("all_sector_treasury_assets")
        ]["value"].sum(min_count=1)
        included_total = q_clean["value"].sum(min_count=1) if not q_clean.empty else 0.0
        mspd_total = float(cohort_totals.get(quarter, 0.0))
        diag = total_diag.get(quarter, {})
        aggregate_entered_solver = int(_aggregate_sector_mask(q_clean).sum()) if not q_clean.empty else 0
        rows.append(
            {
                "quarter": quarter,
                "raw_z1_level_total": q_raw["value"].sum(min_count=1),
                "included_granular_z1_total": included_total,
                "excluded_aggregate_z1_total": q_aggregate["value"].sum(min_count=1),
                "all_sector_treasury_assets_control": all_sector_control,
                "mspd_cohort_total": mspd_total,
                "mspd_minus_included_z1_total": mspd_total - float(included_total),
                "holder_scale_factor": diag.get("scale_factor", pd.NA),
                "aggregate_rows_entered_solver": aggregate_entered_solver,
                "status": "clean_granular_scope" if aggregate_entered_solver == 0 else "aggregate_rows_in_solver",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_z1_transaction_flow_diagnostics(
    sector_observations: pd.DataFrame,
    sector_levels: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "native_sector",
        "broad_holder_class",
        "tdcsim_holder",
        "z1_level_mil",
        "prior_z1_level_mil",
        "z1_level_change_mil",
        "z1_transaction_flow_saar_mil",
        "z1_transaction_flow_mil",
        "z1_transaction_flow_frequency",
        "z1_transaction_flow_conversion",
        "implied_valuation_or_other_volume_mil",
        "abs_flow_gap_mil",
        "flow_gap_share_of_prior_level",
        "diagnostic_status",
        "claim_boundary",
    ]
    if sector_observations.empty:
        return pd.DataFrame(columns=columns)
    obs = sector_observations.copy()
    if "native_sector" not in obs.columns:
        obs["native_sector"] = obs.get("sector", "")
    if "broad_holder_class" not in obs.columns:
        obs["broad_holder_class"] = obs.get("native_sector", "")
    if "tdcsim_holder" not in obs.columns:
        obs["tdcsim_holder"] = obs["broad_holder_class"].apply(_holder_for_sector)
    obs["value"] = pd.to_numeric(obs.get("value"), errors="coerce").fillna(0.0)
    obs["_valuation_basis"] = obs.apply(_z1_valuation_basis, axis=1)
    obs["_is_aggregate"] = _aggregate_sector_mask(obs)
    tx = obs[
        ~obs["_is_aggregate"]
        & (
            obs["_valuation_basis"].astype(str).eq("z1_transaction_flow")
            | obs["measure"].astype(str).str.lower().isin({"flow", "transaction"})
        )
    ].copy()
    levels = sector_levels.copy()
    if levels.empty:
        levels = obs[
            ~obs["_is_aggregate"]
            & obs["measure"].astype(str).str.lower().isin({"level", "stock"})
        ].copy()
    if levels.empty and tx.empty:
        return pd.DataFrame(columns=columns)
    if "native_sector" not in levels.columns:
        levels["native_sector"] = levels.get("sector", "")
    if "broad_holder_class" not in levels.columns:
        levels["broad_holder_class"] = levels.get("native_sector", "")
    if "tdcsim_holder" not in levels.columns:
        levels["tdcsim_holder"] = levels["broad_holder_class"].apply(_holder_for_sector)
    levels["value"] = pd.to_numeric(levels.get("value"), errors="coerce").fillna(0.0)
    key_cols = ["quarter", "native_sector", "broad_holder_class", "tdcsim_holder"]
    level_grouped = (
        levels.groupby(key_cols, dropna=False, sort=False)["value"].sum().reset_index(name="z1_level_mil")
    )
    tx_grouped = (
        tx.groupby(key_cols, dropna=False, sort=False)["value"].sum().reset_index(name="z1_transaction_flow_saar_mil")
        if not tx.empty
        else pd.DataFrame(columns=[*key_cols, "z1_transaction_flow_saar_mil"])
    )
    base = level_grouped.merge(tx_grouped, on=key_cols, how="outer")
    base["z1_level_mil"] = pd.to_numeric(base["z1_level_mil"], errors="coerce")
    base["z1_transaction_flow_saar_mil"] = pd.to_numeric(base["z1_transaction_flow_saar_mil"], errors="coerce")
    # Federal Reserve Z.1 F.210 FA series are annual-rate flows. The quarterly
    # transition moment used against L.210 levels is therefore FA / 4.
    base["z1_transaction_flow_mil"] = base["z1_transaction_flow_saar_mil"] / 4.0
    rows = []
    for (native, broad, holder), group in base.groupby(
        ["native_sector", "broad_holder_class", "tdcsim_holder"],
        dropna=False,
        sort=False,
    ):
        group = group.copy()
        group["_quarter_order"] = pd.PeriodIndex(group["quarter"].astype(str), freq="Q")
        group = group.sort_values("_quarter_order")
        group["prior_z1_level_mil"] = group["z1_level_mil"].shift(1)
        group["z1_level_change_mil"] = group["z1_level_mil"] - group["prior_z1_level_mil"]
        group["implied_valuation_or_other_volume_mil"] = (
            group["z1_level_change_mil"] - group["z1_transaction_flow_mil"]
        )
        for _, row in group.iterrows():
            level_change = _num(row.get("z1_level_change_mil"))
            flow = _num(row.get("z1_transaction_flow_mil"))
            prior_level = abs(_num(row.get("prior_z1_level_mil")))
            gap = level_change - flow if pd.notna(row.get("z1_transaction_flow_mil")) else pd.NA
            if pd.isna(row.get("prior_z1_level_mil")):
                status = "first_level_observation_no_prior_change"
            elif pd.isna(row.get("z1_transaction_flow_mil")):
                status = "missing_z1_transaction_flow_reference"
            else:
                status = "z1_transaction_flow_reconciled_as_transition_diagnostic"
            rows.append(
                {
                    "quarter": row.get("quarter"),
                    "native_sector": native,
                    "broad_holder_class": broad,
                    "tdcsim_holder": holder,
                    "z1_level_mil": row.get("z1_level_mil"),
                    "prior_z1_level_mil": row.get("prior_z1_level_mil"),
                    "z1_level_change_mil": row.get("z1_level_change_mil"),
                    "z1_transaction_flow_saar_mil": row.get("z1_transaction_flow_saar_mil"),
                    "z1_transaction_flow_mil": row.get("z1_transaction_flow_mil"),
                    "z1_transaction_flow_frequency": "SAAR",
                    "z1_transaction_flow_conversion": "FA_divided_by_4_to_quarterly_moment",
                    "implied_valuation_or_other_volume_mil": row.get("implied_valuation_or_other_volume_mil"),
                    "abs_flow_gap_mil": abs(float(gap)) if pd.notna(gap) else pd.NA,
                    "flow_gap_share_of_prior_level": abs(float(gap)) / prior_level if pd.notna(gap) and prior_level else 0.0,
                    "diagnostic_status": status,
                    "claim_boundary": "z1_transaction_flow_is_aggregate_transition_diagnostic_not_exact_transfer",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _build_portfolio_transition_diagnostics(
    period_end_portfolios: dict[str, pd.DataFrame],
    z1_transaction_flow_diagnostics: pd.DataFrame,
    event_rollforward: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "from_quarter",
        "to_quarter",
        "interval_quarters",
        "native_sector",
        "prior_face_mil",
        "current_face_mil",
        "modeled_face_change_mil",
        "total_variation",
        "high_tv_transition",
        "entering_or_increased_face_mil",
        "exiting_or_decreased_face_mil",
        "z1_level_change_mil",
        "z1_transaction_flow_saar_mil",
        "z1_transaction_flow_mil",
        "implied_valuation_or_other_volume_mil",
        "event_source_issue_mil",
        "event_source_redemption_mil",
        "event_source_indexation_mil",
        "event_source_reclassification_mil",
        "event_unexplained_cohort_change_mil",
        "event_unexplained_residual_change_mil",
        "event_ledger_net_change_mil",
        "event_ledger_gross_activity_mil",
        "event_component_reconciliation_gap_mil",
        "event_component_scope",
        "z1_flow_context_status",
        "event_context_status",
        "modeled_minus_z1_flow_mil",
        "modeled_minus_z1_flow_share",
        "transition_explanation_status",
        "claim_boundary",
    ]
    portfolio = _concat_period_end_portfolios(period_end_portfolios)
    if portfolio.empty or not {"quarter", "native_sector", "cohort_id", "FaceValue"}.issubset(portfolio.columns):
        return pd.DataFrame(columns=columns)
    work = portfolio.copy()
    work["FaceValue"] = pd.to_numeric(work["FaceValue"], errors="coerce").fillna(0.0)
    if "source_sector" in work.columns:
        work = work[
            ~work["source_sector"].astype(str).str.contains("SourceBasisResidual", case=False, na=False)
        ].copy()
    grouped = (
        work[work["FaceValue"].gt(0.0)]
        .groupby(["quarter", "native_sector", "cohort_id"], as_index=False, dropna=False)["FaceValue"]
        .sum()
    )
    if grouped.empty:
        return pd.DataFrame(columns=columns)
    flow_lookup = {}
    if not z1_transaction_flow_diagnostics.empty:
        flow_cols = [
            "quarter",
            "native_sector",
            "z1_level_change_mil",
            "z1_transaction_flow_saar_mil",
            "z1_transaction_flow_mil",
            "implied_valuation_or_other_volume_mil",
        ]
        available = [col for col in flow_cols if col in z1_transaction_flow_diagnostics.columns]
        flow_grouped = (
            z1_transaction_flow_diagnostics[available]
            .groupby(["quarter", "native_sector"], dropna=False, sort=False)
            .sum(numeric_only=True)
            .reset_index()
        )
        flow_lookup = {
            (str(row["quarter"]), str(row["native_sector"])): row
            for _, row in flow_grouped.iterrows()
        }
    event_lookup = {}
    event_cols = [
        "source_issue_mil",
        "source_redemption_mil",
        "source_indexation_mil",
        "source_reclassification_mil",
        "unexplained_cohort_change_mil",
        "unexplained_residual_change_mil",
    ]
    if not event_rollforward.empty and {"quarter", *event_cols}.issubset(event_rollforward.columns):
        event_frame = event_rollforward[["quarter", *event_cols]].copy()
        for column in event_cols:
            event_frame[column] = pd.to_numeric(event_frame[column], errors="coerce").fillna(0.0)
        event_frame["event_net_change_mil"] = event_frame[event_cols].sum(axis=1)
        event_frame["event_gross_activity_mil"] = event_frame[event_cols].abs().sum(axis=1)
        event_grouped = event_frame.groupby("quarter", dropna=False, sort=False)[
            [*event_cols, "event_net_change_mil", "event_gross_activity_mil"]
        ].sum()
        event_lookup = {
            str(quarter): {column: float(row[column]) for column in [*event_cols, "event_net_change_mil", "event_gross_activity_mil"]}
            for quarter, row in event_grouped.iterrows()
        }
    quarters = _sort_quarters(grouped["quarter"].dropna().astype(str).unique())
    rows = []
    for sector in sorted(grouped["native_sector"].dropna().astype(str).unique()):
        sector_group = grouped[grouped["native_sector"].astype(str).eq(sector)]
        panel = sector_group.pivot_table(index="cohort_id", columns="quarter", values="FaceValue", fill_value=0.0)
        totals = sector_group.groupby("quarter")["FaceValue"].sum()
        shares = panel.divide(totals, axis=1).fillna(0.0)
        for left, right in zip(quarters, quarters[1:], strict=False):
            interval_quarters = pd.Period(str(right), freq="Q").ordinal - pd.Period(str(left), freq="Q").ordinal
            if interval_quarters != 1:
                continue
            if left not in panel.columns and right not in panel.columns:
                continue
            left_face = panel[left] if left in panel.columns else pd.Series(0.0, index=panel.index)
            right_face = panel[right] if right in panel.columns else pd.Series(0.0, index=panel.index)
            left_share = shares[left] if left in shares.columns else pd.Series(0.0, index=panel.index)
            right_share = shares[right] if right in shares.columns else pd.Series(0.0, index=panel.index)
            tv = float(0.5 * (left_share - right_share).abs().sum())
            prior_total = float(left_face.sum())
            current_total = float(right_face.sum())
            delta = right_face - left_face
            entering = float(delta.clip(lower=0.0).sum())
            exiting = float((-delta.clip(upper=0.0)).sum())
            flow_row = flow_lookup.get((right, sector))
            z1_level_change = _num(flow_row.get("z1_level_change_mil")) if flow_row is not None else pd.NA
            z1_flow_saar = _num(flow_row.get("z1_transaction_flow_saar_mil")) if flow_row is not None else pd.NA
            z1_flow = _num(flow_row.get("z1_transaction_flow_mil")) if flow_row is not None else pd.NA
            z1_other = _num(flow_row.get("implied_valuation_or_other_volume_mil")) if flow_row is not None else pd.NA
            high_tv = tv > 0.90
            event_row = event_lookup.get(right, {})
            event_source_issue = event_row.get("source_issue_mil", pd.NA)
            event_source_redemption = event_row.get("source_redemption_mil", pd.NA)
            event_source_indexation = event_row.get("source_indexation_mil", pd.NA)
            event_source_reclassification = event_row.get("source_reclassification_mil", pd.NA)
            event_unexplained_cohort_change = event_row.get("unexplained_cohort_change_mil", pd.NA)
            event_unexplained_residual_change = event_row.get("unexplained_residual_change_mil", pd.NA)
            event_net_change = event_row.get("event_net_change_mil", pd.NA)
            event_gross_activity = event_row.get("event_gross_activity_mil", pd.NA)
            if pd.notna(event_net_change):
                component_sum = sum(
                    float(value)
                    for value in [
                        event_source_issue,
                        event_source_redemption,
                        event_source_indexation,
                        event_source_reclassification,
                        event_unexplained_cohort_change,
                        event_unexplained_residual_change,
                    ]
                    if pd.notna(value)
                )
                event_component_gap = component_sum - float(event_net_change)
            else:
                event_component_gap = pd.NA
            modeled_change = current_total - prior_total
            modeled_minus_flow = modeled_change - z1_flow if pd.notna(z1_flow) else pd.NA
            flow_context, modeled_minus_flow_share = _classify_z1_flow_context(modeled_change, z1_flow)
            event_context = _classify_event_context(modeled_change, event_net_change, event_gross_activity)
            material_base = max(abs(prior_total), abs(current_total), 1.0)
            cohort_churn_share = max(entering, exiting) / material_base
            if not high_tv:
                explanation = "not_high_turnover"
            elif flow_context != "missing_corrected_z1_flow" and event_context != "missing_event_context":
                explanation = f"high_turnover_disclosed_{flow_context}_and_{event_context}"
            elif flow_context != "missing_corrected_z1_flow":
                explanation = f"high_turnover_disclosed_{flow_context}"
            elif event_context != "missing_event_context":
                explanation = f"high_turnover_disclosed_{event_context}"
            elif cohort_churn_share >= 0.25:
                explanation = "high_turnover_disclosed_model_churn_only_no_external_context"
            else:
                explanation = "high_turnover_missing_transition_context"
            rows.append(
                {
                    "from_quarter": left,
                    "to_quarter": right,
                    "interval_quarters": interval_quarters,
                    "native_sector": sector,
                    "prior_face_mil": prior_total,
                    "current_face_mil": current_total,
                    "modeled_face_change_mil": current_total - prior_total,
                    "total_variation": tv,
                    "high_tv_transition": bool(high_tv),
                    "entering_or_increased_face_mil": entering,
                    "exiting_or_decreased_face_mil": exiting,
                    "z1_level_change_mil": z1_level_change,
                    "z1_transaction_flow_saar_mil": z1_flow_saar,
                    "z1_transaction_flow_mil": z1_flow,
                    "implied_valuation_or_other_volume_mil": z1_other,
                    "event_source_issue_mil": event_source_issue,
                    "event_source_redemption_mil": event_source_redemption,
                    "event_source_indexation_mil": event_source_indexation,
                    "event_source_reclassification_mil": event_source_reclassification,
                    "event_unexplained_cohort_change_mil": event_unexplained_cohort_change,
                    "event_unexplained_residual_change_mil": event_unexplained_residual_change,
                    "event_ledger_net_change_mil": event_net_change,
                    "event_ledger_gross_activity_mil": event_gross_activity,
                    "event_component_reconciliation_gap_mil": event_component_gap,
                    "event_component_scope": "treasury_wide_aggregate_context_not_sector_allocated",
                    "z1_flow_context_status": flow_context,
                    "event_context_status": event_context,
                    "modeled_minus_z1_flow_mil": modeled_minus_flow,
                    "modeled_minus_z1_flow_share": modeled_minus_flow_share,
                    "transition_explanation_status": explanation,
                    "claim_boundary": "transition_diagnostic_not_exact_secondary_market_transfer",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _classify_z1_flow_context(modeled_change: float, z1_flow: float | pd._libs.missing.NAType) -> tuple[str, float | object]:
    if pd.isna(z1_flow):
        return "missing_corrected_z1_flow", pd.NA
    modeled = float(modeled_change)
    flow = float(z1_flow)
    gap = modeled - flow
    base = max(abs(modeled), abs(flow), 1.0)
    share = abs(gap) / base
    if abs(modeled) <= NUMERIC_TOLERANCE_MIL and abs(flow) <= NUMERIC_TOLERANCE_MIL:
        return "corrected_z1_flow_and_model_near_zero", share
    if abs(flow) <= NUMERIC_TOLERANCE_MIL:
        return "corrected_z1_flow_near_zero_model_changes", share
    if abs(modeled) <= NUMERIC_TOLERANCE_MIL:
        return "model_near_zero_corrected_z1_flow_changes", share
    if np.sign(modeled) != np.sign(flow):
        return "corrected_z1_flow_opposes_modeled_change", share
    if share <= 0.25:
        return "corrected_z1_flow_direction_and_magnitude_match", share
    return "corrected_z1_flow_direction_matches_large_magnitude_gap", share


def _classify_event_context(
    modeled_change: float,
    event_net_change: float | pd._libs.missing.NAType,
    event_gross_activity: float | pd._libs.missing.NAType,
) -> str:
    if pd.isna(event_net_change) and pd.isna(event_gross_activity):
        return "missing_event_context"
    modeled = float(modeled_change)
    net = 0.0 if pd.isna(event_net_change) else float(event_net_change)
    gross = 0.0 if pd.isna(event_gross_activity) else float(event_gross_activity)
    if gross <= NUMERIC_TOLERANCE_MIL:
        return "event_context_no_source_activity"
    if abs(modeled) <= NUMERIC_TOLERANCE_MIL:
        return "event_activity_present_model_near_zero"
    if abs(net) <= NUMERIC_TOLERANCE_MIL:
        return "event_gross_activity_present_net_near_zero"
    if np.sign(modeled) == np.sign(net):
        return "event_net_direction_matches_modeled_change"
    return "event_net_direction_opposes_modeled_change"


def _net_negative_native_sectors(sector_levels: pd.DataFrame) -> pd.DataFrame:
    if sector_levels.empty:
        return sector_levels.copy()
    frames = []
    group_cols = ["quarter", "measure", "tdcsim_holder"]
    for _, group in sector_levels.groupby(group_cols, sort=False, dropna=False):
        adjusted = group.copy()
        values = pd.to_numeric(adjusted["value"], errors="coerce").fillna(0.0)
        if (values < 0.0).any():
            protected = adjusted.apply(_is_mmf_sector_row, axis=1)
            positive = values.clip(lower=0.0)
            negative_abs = abs(float(values.clip(upper=0.0).sum()))
            protected_positive = float(positive[protected].sum())
            unprotected_positive = positive.where(~protected, 0.0)
            unprotected_positive_sum = float(unprotected_positive.sum())
            adjusted_values = values.clip(lower=0.0)
            if unprotected_positive_sum > 0.0:
                unprotected_target = max(unprotected_positive_sum - negative_abs, 0.0)
                factor = unprotected_target / unprotected_positive_sum
                adjusted_values.loc[~protected] = unprotected_positive.loc[~protected] * factor
            else:
                adjusted_values.loc[~protected] = 0.0
            adjusted_values.loc[protected] = positive.loc[protected]
            adjusted["value"] = adjusted_values
            if "negative_netting_protection_status" not in adjusted.columns:
                adjusted["negative_netting_protection_status"] = pd.NA
            adjusted.loc[protected & positive.gt(0.0), "negative_netting_protection_status"] = (
                "protected_from_negative_native_sector_netting"
            )
            adjusted.loc[~protected, "negative_netting_protection_status"] = (
                "eligible_for_negative_native_sector_netting"
            )
            changed = (adjusted_values - values).abs().gt(1.0e-9)
            adjusted.loc[changed & ~protected, "sector_adjustment_status"] = "negative_native_sector_netting"
            adjusted.loc[protected, "sector_adjustment_status"] = adjusted.loc[
                protected,
                "sector_adjustment_status",
            ].where(
                adjusted.loc[protected, "sector_adjustment_status"].notna(),
                "observed_nonnegative",
            )
            shortfall = max(negative_abs - unprotected_positive_sum, 0.0)
            if shortfall > 1.0e-9:
                adjusted["negative_netting_unprotected_shortfall_mil"] = shortfall
            else:
                adjusted["negative_netting_unprotected_shortfall_mil"] = 0.0
        frames.append(adjusted)
    if not frames:
        return sector_levels.copy()
    return pd.concat(frames, ignore_index=True)


def _is_mmf_sector_row(row: pd.Series) -> bool:
    native = str(row.get("native_sector", row.get("sector", "")))
    broad = str(row.get("broad_holder_class", ""))
    sector = str(row.get("sector", ""))
    source_status = str(row.get("source_status", ""))
    return (
        _normalize_simple(native) == "moneymarketfunds"
        or _normalize_simple(broad) == "moneymarketcash"
        or _normalize_simple(sector).startswith("moneymarketfunds")
        or source_status.startswith("z1_mmf_")
    )


def _load_mmf_component_targets(
    cfg: dict,
    *,
    start_quarter: str,
    end_quarter: str,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "mmf_tsy_total_source_mil",
        "mmf_tsy_bills_source_mil",
        "source_file",
        "source_status",
    ]
    path = Path(
        cfg.get(
            "mmf_component_constraints",
            "data/historical_replay/imported/tdcest/quarterly_inputs.csv",
        )
    )
    if not path.exists():
        path = Path("data/historical_replay/imported/tdcest/quarterly_inputs.csv")
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, low_memory=False)
    if not {"date", "mmf_tsy_level", "mmf_tsy_bills_level"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = frame[["date", "mmf_tsy_level", "mmf_tsy_bills_level"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out[out["date"].notna()].copy()
    out["quarter"] = out["date"].dt.to_period("Q").astype(str)
    out = out[(out["quarter"] >= str(start_quarter)) & (out["quarter"] <= str(end_quarter))].copy()
    out["mmf_tsy_total_source_mil"] = pd.to_numeric(out["mmf_tsy_level"], errors="coerce").fillna(0.0)
    out["mmf_tsy_bills_source_mil"] = pd.to_numeric(out["mmf_tsy_bills_level"], errors="coerce").fillna(0.0)
    out["source_file"] = _display_path(path)
    out["source_status"] = "z1_mmf_total_and_bill_component"
    out = out[out["mmf_tsy_total_source_mil"].gt(0.0)].copy()
    return out.loc[:, columns]


def _apply_mmf_component_constraints(
    sector_levels: pd.DataFrame,
    cohorts: pd.DataFrame,
    mmf_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "quarter",
        "native_sector",
        "raw_mmf_sector_target_mil",
        "mmf_tsy_total_source_mil",
        "mmf_tsy_bills_source_mil",
        "bill_share_source",
        "bill_component_target_mil",
        "nonbill_component_target_mil",
        "source_file",
        "status",
    ]
    if sector_levels.empty or cohorts.empty or mmf_targets.empty:
        return sector_levels.copy(), cohorts.copy(), pd.DataFrame(columns=columns)
    target_by_quarter = {
        str(row["quarter"]): row
        for _, row in mmf_targets.iterrows()
    }
    frames = []
    rows = []
    for _, row in sector_levels.iterrows():
        native = str(row.get("native_sector", row.get("sector", "")))
        broad = str(row.get("broad_holder_class", ""))
        is_mmf = _normalize_simple(native) == "moneymarketfunds" or _normalize_simple(broad) == "moneymarketcash"
        quarter = str(row.get("quarter"))
        target = target_by_quarter.get(quarter)
        if not is_mmf or target is None:
            frames.append(pd.DataFrame([row]))
            continue
        raw_value = _num(row.get("value"))
        source_total = _num(target.get("mmf_tsy_total_source_mil"))
        source_bills = _num(target.get("mmf_tsy_bills_source_mil"))
        if raw_value <= 0.0 or source_total <= 0.0 or source_bills < 0.0:
            frames.append(pd.DataFrame([row]))
            continue
        bill_share = min(max(source_bills / source_total, 0.0), 1.0)
        bill_target = min(source_total, max(0.0, source_bills))
        nonbill_target = max(source_total - bill_target, 0.0)
        bill_row = row.copy()
        bill_row["sector"] = "money_market_funds__treasury_bills"
        bill_row["value"] = bill_target
        bill_row["raw_z1_level"] = bill_target
        bill_row["sector_target_before_scale"] = bill_target
        bill_row["source_status"] = "z1_mmf_treasury_bill_component_constraint"
        bill_row["sector_adjustment_status"] = "mmf_bill_component_constraint"
        bill_row["negative_netting_protection_status"] = "protected_from_negative_native_sector_netting"
        nonbill_row = row.copy()
        nonbill_row["sector"] = "money_market_funds__nonbill_treasuries"
        nonbill_row["value"] = nonbill_target
        nonbill_row["raw_z1_level"] = nonbill_target
        nonbill_row["sector_target_before_scale"] = nonbill_target
        nonbill_row["source_status"] = "z1_mmf_nonbill_component_constraint"
        nonbill_row["sector_adjustment_status"] = "mmf_nonbill_component_constraint"
        nonbill_row["negative_netting_protection_status"] = "protected_from_negative_native_sector_netting"
        frames.append(pd.DataFrame([bill_row, nonbill_row]))
        rows.append(
            {
                "quarter": quarter,
                "native_sector": native,
                "raw_mmf_sector_target_mil": raw_value,
                "mmf_tsy_total_source_mil": source_total,
                "mmf_tsy_bills_source_mil": source_bills,
                "bill_share_source": bill_share,
                "bill_component_target_mil": bill_target,
                "nonbill_component_target_mil": nonbill_target,
                "source_file": target.get("source_file"),
                "status": (
                    "mmf_direct_dollar_component_constraint_applied"
                    if abs(raw_value - source_total) <= 1.0e-2
                    else "mmf_direct_dollar_component_constraint_overrides_sector_level"
                ),
            }
        )
    out_sectors = pd.concat(frames, ignore_index=True) if frames else sector_levels.copy()
    out_cohorts = cohorts.copy()
    out_cohorts["eligible_money_market_funds__treasury_bills"] = _is_bill_cohort(out_cohorts)
    out_cohorts["eligible_money_market_funds__nonbill_treasuries"] = _is_mmf_nonbill_eligible(out_cohorts)
    return out_sectors, out_cohorts, pd.DataFrame(rows, columns=columns)


def _build_mmf_component_reconciliation(
    input_rows: pd.DataFrame,
    allocations: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> pd.DataFrame:
    columns = list(input_rows.columns) + [
        "modeled_bill_component_mil",
        "modeled_nonbill_component_mil",
        "modeled_total_mmf_treasury_mil",
        "direct_total_gap_mil",
        "direct_bill_component_gap_mil",
        "direct_nonbill_component_gap_mil",
        "modeled_bill_share",
        "bill_component_gap_mil",
        "nonbill_component_gap_mil",
        "fixed_rate_gt_397d_mil",
        "fixed_rate_gt_397d_share",
        "post_solve_status",
    ]
    if input_rows.empty:
        return pd.DataFrame(columns=columns)
    if allocations.empty or cohorts.empty:
        out = input_rows.copy()
        for column in columns:
            if column not in out.columns:
                out[column] = 0.0 if column.endswith(("_mil", "_share")) else pd.NA
        out["post_solve_status"] = "no_allocations"
        return out.loc[:, columns]
    cohort_meta = cohorts[["quarter", "cohort_id", "security_type", "maturity_date"]].drop_duplicates().copy()
    alloc = allocations.merge(cohort_meta, on=["quarter", "cohort_id"], how="left")
    alloc["allocated_outstanding"] = pd.to_numeric(alloc["allocated_outstanding"], errors="coerce").fillna(0.0)
    alloc["_is_mmf_bill_component"] = alloc["sector"].astype(str).eq("money_market_funds__treasury_bills")
    alloc["_is_mmf_nonbill_component"] = alloc["sector"].astype(str).eq("money_market_funds__nonbill_treasuries")
    q_end = pd.PeriodIndex(alloc["quarter"].astype(str), freq="Q").to_timestamp(how="end").normalize()
    maturity = pd.to_datetime(alloc["maturity_date"], errors="coerce")
    remaining_days = (maturity - q_end).dt.days
    security_key = alloc["security_type"].astype(str).map(_normalize_simple)
    alloc["_fixed_gt_397"] = (
        alloc["_is_mmf_nonbill_component"]
        & ~security_key.isin({"bill", "bills", "cashmanagementbill", "cmb", "frn", "floatingratenote"})
        & remaining_days.gt(397)
    )
    grouped = alloc.groupby("quarter", sort=False, dropna=False).agg(
        modeled_bill_component_mil=(
            "allocated_outstanding",
            lambda s: float(s[alloc.loc[s.index, "_is_mmf_bill_component"]].sum()),
        ),
        modeled_nonbill_component_mil=(
            "allocated_outstanding",
            lambda s: float(s[alloc.loc[s.index, "_is_mmf_nonbill_component"]].sum()),
        ),
        fixed_rate_gt_397d_mil=(
            "allocated_outstanding",
            lambda s: float(s[alloc.loc[s.index, "_fixed_gt_397"]].sum()),
        ),
    ).reset_index()
    out = input_rows.merge(grouped, on="quarter", how="left")
    for column in ["modeled_bill_component_mil", "modeled_nonbill_component_mil", "fixed_rate_gt_397d_mil"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    total_modeled = out["modeled_bill_component_mil"] + out["modeled_nonbill_component_mil"]
    out["modeled_total_mmf_treasury_mil"] = total_modeled
    source_total = pd.to_numeric(out["mmf_tsy_total_source_mil"], errors="coerce").fillna(0.0)
    source_bills = pd.to_numeric(out["mmf_tsy_bills_source_mil"], errors="coerce").fillna(0.0)
    source_nonbills = (source_total - source_bills).clip(lower=0.0)
    out["direct_total_gap_mil"] = total_modeled - source_total
    out["direct_bill_component_gap_mil"] = out["modeled_bill_component_mil"] - source_bills
    out["direct_nonbill_component_gap_mil"] = out["modeled_nonbill_component_mil"] - source_nonbills
    out["modeled_bill_share"] = out["modeled_bill_component_mil"] / total_modeled.replace(0.0, pd.NA)
    out["modeled_bill_share"] = out["modeled_bill_share"].fillna(0.0)
    out["bill_component_gap_mil"] = out["modeled_bill_component_mil"] - pd.to_numeric(
        out["bill_component_target_mil"],
        errors="coerce",
    ).fillna(0.0)
    out["nonbill_component_gap_mil"] = out["modeled_nonbill_component_mil"] - pd.to_numeric(
        out["nonbill_component_target_mil"],
        errors="coerce",
    ).fillna(0.0)
    out["fixed_rate_gt_397d_share"] = out["fixed_rate_gt_397d_mil"] / total_modeled.replace(0.0, pd.NA)
    out["fixed_rate_gt_397d_share"] = out["fixed_rate_gt_397d_share"].fillna(0.0)
    out["post_solve_status"] = "matched"
    component_tolerance = 1.0e-2
    bad = (
        out["bill_component_gap_mil"].abs().gt(component_tolerance)
        | out["nonbill_component_gap_mil"].abs().gt(component_tolerance)
        | out["direct_total_gap_mil"].abs().gt(component_tolerance)
        | out["direct_bill_component_gap_mil"].abs().gt(component_tolerance)
        | out["direct_nonbill_component_gap_mil"].abs().gt(component_tolerance)
        | out["fixed_rate_gt_397d_mil"].abs().gt(component_tolerance)
    )
    out.loc[bad, "post_solve_status"] = "component_or_eligibility_gap"
    return out.loc[:, columns]


def _build_negative_sector_netting_bridge(sector_levels: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "measure",
        "tdcsim_holder",
        "native_sector",
        "raw_sector_level_mil",
        "adjusted_sector_level_mil",
        "sector_adjustment_mil",
        "offsetting_positive_sectors",
        "group_raw_positive_mil",
        "group_protected_positive_mil",
        "group_raw_negative_mil",
        "group_adjusted_total_mil",
        "gross_adjustment_mil",
        "gross_adjustment_share_of_raw_abs",
        "claim_boundary",
    ]
    if sector_levels.empty or "sector_adjustment_status" not in sector_levels.columns:
        return pd.DataFrame(columns=columns)
    working = sector_levels.copy()
    working["raw_z1_level"] = pd.to_numeric(working.get("raw_z1_level"), errors="coerce").fillna(0.0)
    working["value"] = pd.to_numeric(working.get("value"), errors="coerce").fillna(0.0)
    rows = []
    group_cols = ["quarter", "measure", "tdcsim_holder"]
    for keys, group in working.groupby(group_cols, sort=False, dropna=False):
        changed = group[group["sector_adjustment_status"].astype(str).eq("negative_native_sector_netting")]
        if changed.empty:
            continue
        protected = group.apply(_is_mmf_sector_row, axis=1)
        positive = group["raw_z1_level"].clip(lower=0.0)
        protected_positive = float(positive[protected].sum())
        adjustable_positive = positive.where(~protected, 0.0)
        raw_positive = float(adjustable_positive.sum())
        raw_negative = float(group["raw_z1_level"].clip(upper=0.0).sum())
        adjusted_total = float(group["value"].sum())
        gross_adjustment = float((group["value"] - group["raw_z1_level"]).abs().sum())
        raw_abs = raw_positive + abs(raw_negative)
        offsetting = ";".join(
            group.loc[
                group["raw_z1_level"].gt(0.0) & ~protected,
                "native_sector",
            ].dropna().astype(str).unique().tolist()
        )
        for _, row in changed.iterrows():
            rows.append(
                {
                    "quarter": keys[0],
                    "measure": keys[1],
                    "tdcsim_holder": keys[2],
                    "native_sector": row.get("native_sector"),
                    "raw_sector_level_mil": float(row["raw_z1_level"]),
                    "adjusted_sector_level_mil": float(row["value"]),
                    "sector_adjustment_mil": float(row["value"] - row["raw_z1_level"]),
                    "offsetting_positive_sectors": offsetting,
                    "group_raw_positive_mil": raw_positive,
                    "group_protected_positive_mil": protected_positive,
                    "group_raw_negative_mil": raw_negative,
                    "group_adjusted_total_mil": adjusted_total,
                    "gross_adjustment_mil": gross_adjustment,
                    "gross_adjustment_share_of_raw_abs": gross_adjustment / raw_abs if raw_abs else 0.0,
                    "claim_boundary": "broad_holder_consistency_after_nonnegative_native_sector_netting",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _build_portfolio_constraint_diagnostics(
    sector_levels: pd.DataFrame,
    solver_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    if not sector_levels.empty:
        for _, row in sector_levels.iterrows():
            raw = _num(row.get("raw_z1_level"))
            scaled = _num(row.get("value"))
            rows.append(
                {
                    "quarter": row.get("quarter"),
                    "diagnostic_type": "native_sector_target",
                    "native_sector": row.get("native_sector", row.get("sector")),
                    "broad_holder_class": row.get("broad_holder_class", pd.NA),
                    "tdcsim_holder": row.get("tdcsim_holder", pd.NA),
                    "cohort_id": pd.NA,
                    "raw_z1_level": raw,
                    "scope_adjustment": scaled - raw,
                    "scaled_target": scaled,
                    "mspd_total": pd.NA,
                    "achieved_total": pd.NA,
                    "residual": pd.NA,
                    "scale_factor": pd.NA,
                    "status": row.get("sector_adjustment_status", "observed_nonnegative"),
                    "source_status": row.get("source_status", pd.NA),
                    "evidence_label": row.get("evidence_label", pd.NA),
                }
            )
    if not solver_diagnostics.empty:
        for _, row in solver_diagnostics.iterrows():
            if row.get("diagnostic_type") != "sector_balance":
                continue
            raw = row.get("raw_z1_level", row.get("input_total", pd.NA))
            scaled = row.get("target_total", pd.NA)
            rows.append(
                {
                    "quarter": row.get("quarter"),
                    "diagnostic_type": "sector_balance",
                    "native_sector": row.get("native_sector", row.get("subject")),
                    "broad_holder_class": row.get("broad_holder_class", pd.NA),
                    "tdcsim_holder": row.get("tdcsim_holder", pd.NA),
                    "cohort_id": pd.NA,
                    "raw_z1_level": raw,
                    "scope_adjustment": _maybe_diff(scaled, raw),
                    "scaled_target": scaled,
                    "mspd_total": pd.NA,
                    "achieved_total": row.get("achieved_total", pd.NA),
                    "residual": row.get("residual", pd.NA),
                    "scale_factor": row.get("scale_factor", pd.NA),
                    "status": row.get("status", pd.NA),
                    "source_status": row.get("source_status", pd.NA),
                    "evidence_label": row.get("evidence_label", pd.NA),
                }
            )
    return pd.DataFrame(rows)


def _build_interest_proxy_alignment(
    interest_detail: pd.DataFrame,
    tdc_panel: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "native_sector",
        "component",
        "stock_only_proxy",
        "interest_constrained_proxy",
        "tdcest_point",
        "tdcest_low",
        "tdcest_high",
        "feasible_min",
        "feasible_max",
        "stock_only_gap",
        "constrained_gap",
        "adjustment_from_stock_only",
        "within_feasible_bounds",
        "stock_only_within_tolerance",
        "constrained_within_tolerance",
        "within_tolerance",
        "method_tier",
    ]
    if interest_detail.empty:
        return pd.DataFrame(columns=columns)
    aggregate = aggregate_stock_only_interest(interest_detail, holder_column="tdcsim_holder")
    if aggregate.empty:
        return pd.DataFrame(columns=columns)
    references = _tdc_interest_reference_by_quarter(tdc_panel)
    bounds = _interest_bounds_by_quarter_holder(interest_detail)
    rows = []
    for _, row in aggregate.iterrows():
        quarter = str(row["quarter"])
        holder = _holder_for_sector(row.get("tdcsim_holder"))
        reference_row = references.get(quarter, {})
        reference = reference_row.get(holder, {})
        point = reference.get("point", pd.NA)
        low = reference.get("low", point)
        high = reference.get("high", point)
        stock_only = float(row.get("stock_only_interest_proxy", 0.0))
        gap = pd.NA if pd.isna(point) else stock_only - float(point)
        bound = bounds.get(quarter, {}).get(holder, {})
        feasible_min = bound.get("feasible_min", pd.NA)
        feasible_max = bound.get("feasible_max", pd.NA)
        constrained = _clip_to_bounds(point, feasible_min, feasible_max)
        constrained_gap = (
            constrained - float(point)
            if pd.notna(constrained) and pd.notna(point)
            else pd.NA
        )
        within_feasible = _target_intersects_bounds(low, high, feasible_min, feasible_max)
        stock_only_within_tolerance = (
            pd.NA if pd.isna(gap) else abs(float(gap)) <= 1.0e-9
        )
        constrained_within_tolerance = (
            pd.NA
            if pd.isna(constrained_gap)
            else abs(float(constrained_gap)) <= 1.0e-9
        )
        rows.append(
            {
                "quarter": quarter,
                "native_sector": holder,
                "component": "stock_only_interest_proxy",
                "stock_only_proxy": stock_only,
                "interest_constrained_proxy": constrained,
                "tdcest_point": point,
                "tdcest_low": low,
                "tdcest_high": high,
                "feasible_min": feasible_min,
                "feasible_max": feasible_max,
                "stock_only_gap": gap,
                "constrained_gap": constrained_gap,
                "adjustment_from_stock_only": (
                    constrained - stock_only if pd.notna(constrained) else pd.NA
                ),
                "within_feasible_bounds": within_feasible,
                "stock_only_within_tolerance": stock_only_within_tolerance,
                "constrained_within_tolerance": constrained_within_tolerance,
                "within_tolerance": constrained_within_tolerance,
                "method_tier": (
                    "single_holder_stock_bound_projection"
                    if pd.notna(constrained)
                    else "stock_only_unconstrained"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_holder_mix_differentiation(
    period_end_portfolios: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = [
        "quarter",
        "mix_dimension",
        "dimension_value",
        "holder_count",
        "min_holder_share",
        "max_holder_share",
        "max_minus_min_holder_share",
        "status",
    ]
    snapshots = _concat_period_end_portfolios(period_end_portfolios)
    if snapshots.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "HolderType", "FaceValue", "SecurityType", "MaturityCategory"}
    if not required.issubset(snapshots.columns):
        return pd.DataFrame(columns=columns)
    working = snapshots.copy()
    working["FaceValue"] = pd.to_numeric(working["FaceValue"], errors="coerce").fillna(0.0)
    holder_totals = (
        working.groupby(["quarter", "HolderType"], dropna=False, sort=False)["FaceValue"]
        .sum()
        .rename("holder_total")
        .reset_index()
    )
    rows = []
    for dimension, column in [
        ("security_type", "SecurityType"),
        ("maturity_category", "MaturityCategory"),
    ]:
        grouped = (
            working.groupby(["quarter", "HolderType", column], dropna=False, sort=False)["FaceValue"]
            .sum()
            .reset_index()
            .merge(holder_totals, on=["quarter", "HolderType"], how="left", validate="many_to_one")
        )
        grouped["share"] = grouped["FaceValue"] / grouped["holder_total"].replace(0.0, pd.NA)
        for (quarter, dimension_value), group in grouped.groupby(["quarter", column], sort=True, dropna=False):
            shares = pd.to_numeric(group["share"], errors="coerce").dropna()
            if shares.empty:
                continue
            spread = float(shares.max() - shares.min())
            rows.append(
                {
                    "quarter": str(quarter),
                    "mix_dimension": dimension,
                    "dimension_value": dimension_value,
                    "holder_count": int(shares.size),
                    "min_holder_share": float(shares.min()),
                    "max_holder_share": float(shares.max()),
                    "max_minus_min_holder_share": spread,
                    "status": "differentiated" if spread > 1.0e-9 else "identical_mix",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _tdc_interest_reference_by_quarter(tdc_panel: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if tdc_panel is None or tdc_panel.empty or "quarter" not in tdc_panel.columns:
        return {}
    out: dict[str, dict[str, dict[str, object]]] = {}
    for _, row in tdc_panel.iterrows():
        quarter = str(row.get("quarter"))
        banks = _first_available(
            row,
            "banks_plus_credit_union_tier2_candidate_interest_mil",
            fallback=_sum_optional(
                row,
                "bank_tier2_component_interest_proxy",
                "credit_union_tier2_component_interest_proxy",
            ),
        )
        banks_low = _first_available(row, "banks_plus_credit_union_tier2_candidate_interest_low_mil", fallback=banks)
        banks_high = _first_available(row, "banks_plus_credit_union_tier2_candidate_interest_high_mil", fallback=banks)
        foreign = _first_available(row, "row_tier2_candidate_interest_mil", fallback=row.get("row_tier2_component_interest_proxy", pd.NA))
        foreign_low = _first_available(row, "row_tier2_candidate_interest_low_mil", fallback=foreign)
        foreign_high = _first_available(row, "row_tier2_candidate_interest_high_mil", fallback=foreign)
        cb = row.get("fed_tsy_coupon_interest_proxy", pd.NA)
        out[quarter] = {
            "CB": {"point": cb, "low": cb, "high": cb, "source_status": "tdcest_fed_coupon_proxy"},
            "Banks": {"point": banks, "low": banks_low, "high": banks_high, "source_status": "tdcest_tier2_candidate_or_component_proxy"},
            "Foreign": {"point": foreign, "low": foreign_low, "high": foreign_high, "source_status": "tdcest_tier2_candidate_or_component_proxy"},
        }
    return out


def _interest_bounds_by_quarter_holder(interest_detail: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    if interest_detail.empty:
        return {}
    required = {"quarter", "cusip/cohort_id", "tdcsim_holder", "interest_amount", "exposure_base"}
    if not required.issubset(interest_detail.columns):
        return {}
    working = interest_detail.copy()
    if "excluded_from_default_canonical" in working.columns:
        working = working.loc[~working["excluded_from_default_canonical"].fillna(False)].copy()
    working["interest_amount"] = pd.to_numeric(working["interest_amount"], errors="coerce").fillna(0.0)
    working["exposure_base"] = pd.to_numeric(working["exposure_base"], errors="coerce").fillna(0.0)
    interest_by_cohort = (
        working.groupby(["quarter", "cusip/cohort_id"], dropna=False, sort=False)["interest_amount"]
        .sum()
        .rename("interest_amount")
        .reset_index()
    )
    base_positions = working.drop_duplicates(
        subset=["quarter", "cusip/cohort_id", "native_sector", "tdcsim_holder"],
        keep="first",
    ).copy()
    capacity_by_cohort = (
        base_positions.groupby(["quarter", "cusip/cohort_id"], dropna=False, sort=False)["exposure_base"]
        .sum()
        .rename("cohort_capacity")
        .reset_index()
    )
    coefficients = interest_by_cohort.merge(
        capacity_by_cohort,
        on=["quarter", "cusip/cohort_id"],
        how="inner",
        validate="one_to_one",
    )
    coefficients["interest_rate_proxy"] = coefficients["interest_amount"] / coefficients["cohort_capacity"].replace(0.0, pd.NA)
    base_positions["holder"] = base_positions["tdcsim_holder"].map(_holder_for_sector)
    holder_capacity = (
        base_positions.groupby(["quarter", "holder"], dropna=False, sort=False)["exposure_base"]
        .sum()
        .rename("holder_capacity")
        .reset_index()
    )
    out: dict[str, dict[str, dict[str, float]]] = {}
    for quarter, q_coeff in coefficients.groupby("quarter", sort=False):
        q_coeff = q_coeff.dropna(subset=["interest_rate_proxy", "cohort_capacity"])
        capacities = q_coeff[["interest_rate_proxy", "cohort_capacity"]].copy()
        for _, holder_row in holder_capacity.loc[holder_capacity["quarter"].astype(str) == str(quarter)].iterrows():
            holder = str(holder_row["holder"])
            total = float(holder_row["holder_capacity"])
            out.setdefault(str(quarter), {})[holder] = {
                "feasible_min": _greedy_interest_bound(capacities, total, maximize=False),
                "feasible_max": _greedy_interest_bound(capacities, total, maximize=True),
            }
    return out


def _greedy_interest_bound(capacities: pd.DataFrame, holder_capacity: float, *, maximize: bool) -> float:
    if holder_capacity <= 0.0 or capacities.empty:
        return 0.0
    ordered = capacities.sort_values("interest_rate_proxy", ascending=not maximize)
    remaining = holder_capacity
    total_interest = 0.0
    for _, row in ordered.iterrows():
        if remaining <= 0.0:
            break
        capacity = max(float(row["cohort_capacity"]), 0.0)
        take = min(remaining, capacity)
        total_interest += take * float(row["interest_rate_proxy"])
        remaining -= take
    return total_interest


def _clip_to_bounds(point, feasible_min, feasible_max):
    if pd.isna(point) or pd.isna(feasible_min) or pd.isna(feasible_max):
        return pd.NA
    return min(max(float(point), float(feasible_min)), float(feasible_max))


def _target_intersects_bounds(low, high, feasible_min, feasible_max):
    if pd.isna(low) or pd.isna(high) or pd.isna(feasible_min) or pd.isna(feasible_max):
        return pd.NA
    return float(high) >= float(feasible_min) and float(low) <= float(feasible_max)


def _sum_optional(row: pd.Series, *columns: str):
    values = []
    for column in columns:
        value = row.get(column, pd.NA)
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return pd.NA
    return sum(values)


def _first_available(row: pd.Series, column: str, *, fallback=pd.NA):
    value = row.get(column, pd.NA)
    return value if pd.notna(value) else fallback


def _build_results(
    cash: pd.DataFrame,
    allocations: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    tdc_panel: pd.DataFrame | None = None,
    amount_unit_scale: float,
) -> pd.DataFrame:
    rows = []
    alloc_by_quarter_holder = _allocation_by_quarter_holder(allocations)
    alloc_total = allocations.groupby("quarter")["allocated_outstanding"].sum() if not allocations.empty else pd.Series(dtype=float)
    total_diag_by_quarter = _total_balance_by_quarter(diagnostics)
    diag_target_total = {
        quarter: float(row.get("target_total", 0.0))
        for quarter, row in total_diag_by_quarter.items()
    }
    tdc_by_quarter = _tdc_by_quarter(tdc_panel)

    for _, cash_row in cash.sort_values("quarter").iterrows():
        quarter = cash_row["quarter"]
        row = {column: 0.0 for column in REPLAY_RESULT_COLUMNS}
        for column in TDC_RESULT_COLUMNS:
            row[column] = pd.NA
        row["HolderSurfaceStatus"] = "cell_portfolio_not_evaluated"
        row["DebtSurfaceRole"] = "synthetic_portfolio_from_mspd_cohorts_and_z1_holders"
        row["Date"] = pd.Period(str(quarter), freq="Q").end_time.normalize()
        selected_level = _num(cash_row.get("toc")) / amount_unit_scale
        selected_tx = _num(cash_row.get("toc_transaction", cash_row.get("toc_tx", 0.0))) / amount_unit_scale
        tga = _num(cash_row.get("tga"))
        ttl = _num(cash_row.get("ttl"))
        row["TOC_Level"] = selected_level
        row["TOC_Change"] = selected_tx
        row["TGA"] = (tga / amount_unit_scale) if pd.notna(tga) else selected_level
        row["TTL"] = (ttl / amount_unit_scale) if pd.notna(ttl) else 0.0
        row["TGAChange"] = selected_tx
        row["TotalDebt_Agg"] = float(alloc_total.get(quarter, diag_target_total.get(str(quarter), 0.0))) / amount_unit_scale
        total_diag = total_diag_by_quarter.get(str(quarter), {})
        solver_method = str(total_diag.get("solver_method", "")) if total_diag else ""
        cell_portfolio_status = str(total_diag.get("cell_portfolio_status", "")) if total_diag else ""
        aggregate_only = solver_method == "aggregate_only_minimum_weighted_slack_certificate" or cell_portfolio_status == "aggregate_only_no_cell_portfolio"
        row["HolderSurfaceStatus"] = (
            "aggregate_only_no_cell_portfolio"
            if aggregate_only
            else "cell_portfolio_entropy_projected"
        )
        row["DebtSurfaceRole"] = (
            "aggregate_certificate_no_cell_portfolio"
            if aggregate_only
            else "synthetic_portfolio_from_mspd_cohorts_and_z1_holders"
        )
        holder_totals = alloc_by_quarter_holder.get(quarter, {})
        for holder in ["Banks", "Private", "CB", "Foreign", "FedInternal", "TrustFunds"]:
            row[f"DebtHeld_{holder}"] = (
                pd.NA
                if aggregate_only
                else float(holder_totals.get(holder, 0.0)) / amount_unit_scale
            )
        if total_diag:
            z1_holder_total = float(total_diag.get("input_total", 0.0))
            mspd_cohort_total = float(total_diag.get("target_total", 0.0))
            raw_source_basis_diff = float(
                total_diag.get("raw_mspd_minus_z1_total", total_diag.get("residual", mspd_cohort_total - z1_holder_total))
            )
            modeled_face_equivalent_residual = float(
                total_diag.get("modeled_face_equivalent_basis_residual", total_diag.get("basis_residual", raw_source_basis_diff))
            )
            row["Z1_Holder_Total"] = z1_holder_total / amount_unit_scale
            row["MSPD_Cohort_Total"] = mspd_cohort_total / amount_unit_scale
            row["MSPD_Z1_SourceBasisDiff"] = raw_source_basis_diff / amount_unit_scale
            row["MSPD_Minus_IncludedRawZ1"] = raw_source_basis_diff / amount_unit_scale
            row["ModeledFaceEquivalentBasisResidual"] = modeled_face_equivalent_residual / amount_unit_scale
            row["HolderScaleFactor"] = pd.NA if aggregate_only else float(total_diag.get("scale_factor", 0.0))
            row["Replay_CohortResidual"] = row["MSPD_Z1_SourceBasisDiff"]
            row["Replay_SectorResidual"] = (
                pd.NA
                if aggregate_only
                else max(
                    0.0,
                    row["TotalDebt_Agg"]
                    - sum(row[f"DebtHeld_{holder}"] for holder in ["Banks", "Private", "CB", "Foreign", "FedInternal", "TrustFunds"]),
                )
            )
        tdc_row = tdc_by_quarter.get(str(quarter))
        if tdc_row is not None:
            row["TDC_Level"] = _panel_num(tdc_row, "closing_tdc_level_mil", amount_unit_scale)
            row["TDC_Change"] = _panel_num(tdc_row, "selected_tdc_value_mil", amount_unit_scale)
            row["TDC_FiscalFlow"] = _panel_num(tdc_row, "tdc_fiscal_flow_mil", amount_unit_scale)
            row["TDC_DebtService"] = _panel_num(tdc_row, "tdc_debt_service_mil", amount_unit_scale)
            row["TDC_AuctionAbsorption"] = _panel_num(tdc_row, "tdc_auction_absorption_mil", amount_unit_scale)
            row["TDC_SecondaryTrades"] = _panel_num(tdc_row, "tdc_secondary_trades_mil", amount_unit_scale)
            row["TDC_Other"] = _panel_num(tdc_row, "tdc_other_mil", amount_unit_scale)
            row["TDC_Residual"] = _panel_num(tdc_row, "tdc_residual_mil", amount_unit_scale)
        rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        return pd.DataFrame(columns=REPLAY_RESULT_COLUMNS)
    results = results.set_index("Date")
    for column in REPLAY_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = pd.NA if column in TDC_RESULT_COLUMNS else 0.0
    return results[REPLAY_RESULT_COLUMNS]


def _build_replay_ledger(
    cash: pd.DataFrame,
    results: pd.DataFrame,
    *,
    tdc_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    cash_by_quarter = cash.set_index("quarter", drop=False) if "quarter" in cash.columns else pd.DataFrame()
    ledger = results.reset_index().rename(columns={"Date": "date"})
    ledger.insert(0, "quarter", ledger["date"].map(lambda value: str(pd.Period(value, freq="Q"))))
    source_columns = [
        "toc_source_status",
        "tga_source_status",
        "ttl_source_status",
        "toc_transaction_source_status",
        "evidence_label",
    ]
    for source_column in source_columns:
        if source_column not in cash.columns:
            continue
        ledger[source_column] = ledger["quarter"].map(
            lambda quarter: cash_by_quarter.at[quarter, source_column]
            if quarter in cash_by_quarter.index
            else pd.NA
        )
    ledger["debt_surface_role"] = (
        ledger["DebtSurfaceRole"]
        if "DebtSurfaceRole" in ledger.columns
        else "synthetic_portfolio_from_mspd_cohorts_and_z1_holders"
    )
    ledger["holder_surface_status"] = (
        ledger["HolderSurfaceStatus"]
        if "HolderSurfaceStatus" in ledger.columns
        else "cell_portfolio_not_evaluated"
    )
    ledger["source_basis_diff_role"] = "literal_mspd_cohort_total_minus_included_raw_z1_holder_total"
    ledger["modeled_basis_residual_role"] = "model_price_face_equivalent_residual_used_for_source_basis_residual_portfolio_row"
    if tdc_panel is not None and not tdc_panel.empty:
        tdc_cols = [
            "quarter",
            "selected_tdc_series_key",
            "replay_tdc_method_label",
            "replay_tdc_method_tier",
            "tdc_measurement_status",
            "tdc_units",
            "secondary_trades_measurement_status",
            "tdc_residual_status",
            "treasury_to_ru_transfer_share_of_deficit",
            "assumed_treasury_to_ru_transfer_mil",
            "tdc_unobserved_ru_transfer_contribution_mil",
            "assumed_transfer_status",
            "tdc_level_status",
        ]
        available = [col for col in tdc_cols if col in tdc_panel.columns]
        ledger = ledger.merge(
            tdc_panel[available].drop_duplicates(subset=["quarter"]),
            on="quarter",
            how="left",
            validate="one_to_one",
        )
    return ledger


def _export_replay_outputs(
    cfg: dict,
    *,
    results: pd.DataFrame,
    ledger: pd.DataFrame,
    diagnostics: pd.DataFrame,
    portfolio_constraint_diagnostics: pd.DataFrame,
    z1_scope_reconciliation: pd.DataFrame,
    observation_registry: pd.DataFrame,
    exact_observation_coverage: pd.DataFrame,
    holder_basis_bridge: pd.DataFrame,
    z1_transaction_flow_diagnostics: pd.DataFrame,
    portfolio_transition_diagnostics: pd.DataFrame,
    maturity_prior_reconciliation: pd.DataFrame,
    mmf_component_reconciliation: pd.DataFrame,
    negative_sector_netting_bridge: pd.DataFrame,
    event_ledger: pd.DataFrame,
    event_rollforward: pd.DataFrame,
    unexplained_change_ledger: pd.DataFrame,
    pricing_scope_diagnostics: pd.DataFrame,
    auction_use_reconciliation: pd.DataFrame,
    holder_mix_differentiation: pd.DataFrame,
    interest_component_detail: pd.DataFrame,
    interest_proxy_alignment: pd.DataFrame,
    treasury_interest_expense_diagnostic: pd.DataFrame,
    auction_allotment_proxy: pd.DataFrame,
    auction_absorption_reconciliation: pd.DataFrame,
    soma_holdings: pd.DataFrame,
    soma_fixed_allocations: pd.DataFrame,
    soma_holdout_diagnostics: pd.DataFrame,
    strips_scope_diagnostics: pd.DataFrame,
    valuation_scope_diagnostics: pd.DataFrame,
    period_end_portfolios: dict[str, pd.DataFrame],
    final_portfolio: pd.DataFrame,
    tdc_panel: pd.DataFrame | None = None,
    tdc_formula_crosscheck: pd.DataFrame | None = None,
    tdc_source_manifest: pd.DataFrame | None = None,
    replay_input_manifest: pd.DataFrame | None = None,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> dict[str, str]:
    output_dir = _resolve_output_dir(cfg)
    if output_dir is None:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_duplicate = output_dir / "interest_constraint_feasibility.csv"
    if stale_duplicate.exists():
        stale_duplicate.unlink()
    paths = {
        "results": output_dir / "historical_replay_results.csv",
        "ledger": output_dir / "historical_replay_ledger.csv",
        "diagnostics": output_dir / "historical_replay_diagnostics.csv",
        "portfolio_constraint_diagnostics": output_dir / "portfolio_constraint_diagnostics.csv",
        "solver_constraint_residuals": output_dir / "solver_constraint_residuals.csv",
        "z1_scope_reconciliation": output_dir / "z1_scope_reconciliation.csv",
        "observation_registry": output_dir / "historical_replay_observation_registry.csv",
        "exact_observation_coverage": output_dir / "historical_replay_exact_observation_coverage.csv",
        "holder_basis_bridge": output_dir / "historical_replay_holder_basis_bridge.csv",
        "z1_transaction_flow_diagnostics": output_dir / "z1_transaction_flow_diagnostics.csv",
        "portfolio_transition_diagnostics": output_dir / "portfolio_transition_diagnostics.csv",
        "maturity_prior_reconciliation": output_dir / "maturity_prior_reconciliation.csv",
        "mmf_component_reconciliation": output_dir / "mmf_component_reconciliation.csv",
        "negative_sector_netting_bridge": output_dir / "negative_sector_netting_bridge.csv",
        "event_ledger": output_dir / "historical_replay_event_ledger.csv",
        "event_rollforward": output_dir / "historical_replay_event_rollforward.csv",
        "unexplained_change_ledger": output_dir / "historical_replay_unexplained_change_ledger.csv",
        "pricing_scope_diagnostics": output_dir / "pricing_scope_diagnostics.csv",
        "auction_use_reconciliation": output_dir / "auction_use_reconciliation.csv",
        "holder_mix_differentiation": output_dir / "holder_mix_differentiation.csv",
        "security_term_invariants": output_dir / "security_term_invariants.csv",
        "tips_principal_identity": output_dir / "tips_principal_identity.csv",
        "interest_component_detail": output_dir / "interest_component_detail.csv",
        "interest_proxy_alignment": output_dir / "interest_proxy_alignment.csv",
        "interest_proxy_alignment_summary": output_dir / "interest_proxy_alignment_summary.md",
        "treasury_interest_expense_diagnostic": output_dir / "treasury_interest_expense_diagnostic.csv",
        "auction_allotment_proxy": output_dir / "auction_allotment_proxy.csv",
        "auction_absorption_reconciliation": output_dir / "auction_absorption_quarter_holder_reconciliation.csv",
        "soma_holdings": output_dir / "soma_treasury_holdings_quarterly.csv",
        "soma_fixed_allocations": output_dir / "soma_fixed_allocations.csv",
        "soma_holdout_diagnostics": output_dir / "soma_holdout_diagnostics.csv",
        "strips_scope_diagnostics": output_dir / "strips_scope_diagnostics.csv",
        "valuation_scope_diagnostics": output_dir / "valuation_scope_diagnostics.csv",
        "valuation_basis_feasibility_certificate": output_dir / "valuation_basis_feasibility_certificate.csv",
        "portfolio_native_sector_similarity": output_dir / "portfolio_native_sector_similarity.csv",
        "portfolio_snapshots": output_dir / "historical_replay_portfolio_snapshots.csv",
        "final_portfolio": output_dir / "historical_replay_final_portfolio.csv",
        "replay_input_manifest": output_dir / "historical_replay_input_manifest.csv",
        "large_artifact_sample_manifest": output_dir / "historical_replay_large_artifact_sample_manifest.csv",
        "portfolio_snapshots_sample": output_dir / "historical_replay_portfolio_snapshots_sample.csv",
        "interest_component_detail_sample": output_dir / "interest_component_detail_sample.csv",
        "runtime_manifest": output_dir / "historical_replay_runtime_manifest.csv",
        "runtime_lock": output_dir / "historical_replay_runtime_lock.csv",
        "code_identity_manifest": output_dir / "historical_replay_code_identity_manifest.csv",
    }
    results.reset_index().rename(columns={"Date": "date"}).to_csv(paths["results"], index=False)
    ledger.to_csv(paths["ledger"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    diagnostics.to_csv(paths["solver_constraint_residuals"], index=False)
    portfolio_constraint_diagnostics.to_csv(paths["portfolio_constraint_diagnostics"], index=False)
    z1_scope_reconciliation.to_csv(paths["z1_scope_reconciliation"], index=False)
    observation_registry.to_csv(paths["observation_registry"], index=False)
    exact_observation_coverage.to_csv(paths["exact_observation_coverage"], index=False)
    holder_basis_bridge.to_csv(paths["holder_basis_bridge"], index=False)
    z1_transaction_flow_diagnostics.to_csv(paths["z1_transaction_flow_diagnostics"], index=False)
    portfolio_transition_diagnostics.to_csv(paths["portfolio_transition_diagnostics"], index=False)
    maturity_prior_reconciliation.to_csv(paths["maturity_prior_reconciliation"], index=False)
    mmf_component_reconciliation.to_csv(paths["mmf_component_reconciliation"], index=False)
    negative_sector_netting_bridge.to_csv(paths["negative_sector_netting_bridge"], index=False)
    event_ledger.to_csv(paths["event_ledger"], index=False)
    event_rollforward.to_csv(paths["event_rollforward"], index=False)
    unexplained_change_ledger.to_csv(paths["unexplained_change_ledger"], index=False)
    pricing_scope_diagnostics.to_csv(paths["pricing_scope_diagnostics"], index=False)
    auction_use_reconciliation.to_csv(paths["auction_use_reconciliation"], index=False)
    holder_mix_differentiation.to_csv(paths["holder_mix_differentiation"], index=False)
    interest_component_detail.to_csv(paths["interest_component_detail"], index=False)
    interest_proxy_alignment.to_csv(paths["interest_proxy_alignment"], index=False)
    treasury_interest_expense_diagnostic.to_csv(paths["treasury_interest_expense_diagnostic"], index=False)
    paths["interest_proxy_alignment_summary"].write_text(
        _build_interest_proxy_alignment_summary(interest_proxy_alignment),
        encoding="utf-8",
    )
    auction_allotment_proxy.to_csv(paths["auction_allotment_proxy"], index=False)
    auction_absorption_reconciliation.to_csv(paths["auction_absorption_reconciliation"], index=False)
    soma_holdings.to_csv(paths["soma_holdings"], index=False)
    soma_fixed_allocations.to_csv(paths["soma_fixed_allocations"], index=False)
    soma_holdout_diagnostics.to_csv(paths["soma_holdout_diagnostics"], index=False)
    strips_scope_diagnostics.to_csv(paths["strips_scope_diagnostics"], index=False)
    valuation_scope_diagnostics.to_csv(paths["valuation_scope_diagnostics"], index=False)
    _build_valuation_basis_feasibility_certificate(diagnostics).to_csv(
        paths["valuation_basis_feasibility_certificate"],
        index=False,
    )
    portfolio_snapshots = _concat_period_end_portfolios(period_end_portfolios)
    _build_native_sector_similarity_diagnostics(portfolio_snapshots).to_csv(
        paths["portfolio_native_sector_similarity"],
        index=False,
    )
    _build_security_term_invariants(portfolio_snapshots).to_csv(paths["security_term_invariants"], index=False)
    _build_tips_principal_identity(portfolio_snapshots).to_csv(paths["tips_principal_identity"], index=False)
    portfolio_snapshots.to_csv(paths["portfolio_snapshots"], index=False)
    final_portfolio.to_csv(paths["final_portfolio"], index=False)
    _write_large_artifact_samples(
        paths,
        portfolio_snapshots=portfolio_snapshots,
        interest_component_detail=interest_component_detail,
    )
    _build_runtime_manifest().to_csv(paths["runtime_manifest"], index=False)
    _build_runtime_lock().to_csv(paths["runtime_lock"], index=False)
    _build_code_identity_manifest().to_csv(paths["code_identity_manifest"], index=False)
    (
        replay_input_manifest
        if replay_input_manifest is not None
        else pd.DataFrame()
    ).to_csv(paths["replay_input_manifest"], index=False)
    out = {name: str(path) for name, path in paths.items()}
    if tdc_panel is not None and tdc_formula_crosscheck is not None and tdc_source_manifest is not None:
        out.update(
            write_tdc_validation_artifacts(
                output_dir,
                panel=tdc_panel,
                formula_crosscheck=tdc_formula_crosscheck,
                source_manifest=tdc_source_manifest,
            )
        )
    acceptance_path = output_dir / "historical_replay_acceptance.md"
    if acceptance_path.exists():
        out["historical_replay_acceptance"] = str(acceptance_path)
    artifact_integrity_path = output_dir / "artifact_integrity.csv"
    config_sha256 = _build_replay_config_hash(
        cfg,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
        replay_input_manifest=replay_input_manifest,
    )
    code_sha256 = _build_code_identity_hash()
    run_id = _build_run_id(config_sha256=config_sha256, code_sha256=code_sha256)
    _build_artifact_integrity(
        out,
        run_id=run_id,
        config_sha256=config_sha256,
        code_sha256=code_sha256,
    ).to_csv(artifact_integrity_path, index=False)
    out["artifact_integrity"] = str(artifact_integrity_path)
    return out


def _write_large_artifact_samples(
    paths: dict[str, Path],
    *,
    portfolio_snapshots: pd.DataFrame,
    interest_component_detail: pd.DataFrame,
) -> None:
    manifest_rows = []
    sample_specs = [
        (
            "historical_replay_portfolio_snapshots.csv",
            paths["portfolio_snapshots"],
            paths["portfolio_snapshots_sample"],
            portfolio_snapshots,
            _large_portfolio_sample_strata,
            "quarters in {2023Q4,2024Q1,2025Q4} OR source/native sector contains money_market, holding_companies, or SourceBasisResidual, plus deterministic head/tail rows",
        ),
        (
            "interest_component_detail.csv",
            paths["interest_component_detail"],
            paths["interest_component_detail_sample"],
            interest_component_detail,
            _large_interest_sample_strata,
            "quarters in {2023Q4,2025Q4} OR native sector contains Foreign, money_market, or Banks, plus deterministic head/tail rows",
        ),
    ]
    for source_name, source_path, sample_path, frame, strata_func, sample_rule in sample_specs:
        sample, stratum_counts = _deterministic_large_artifact_sample(frame, strata_func)
        sample.to_csv(sample_path, index=False)
        zero_selected = [
            name
            for name, counts in stratum_counts.items()
            if int(counts.get("source_row_count", 0)) > 0 and int(counts.get("selected_row_count", 0)) <= 0
        ]
        manifest_rows.append(
            {
                "source_artifact": source_name,
                "source_path": _display_path(source_path),
                "source_sha256": _sha256_file(source_path),
                "source_bytes": source_path.stat().st_size,
                "source_row_count": int(len(frame.index)),
                "sample_path": _display_path(sample_path),
                "sample_sha256": _sha256_file(sample_path),
                "sample_bytes": sample_path.stat().st_size,
                "sample_row_count": int(len(sample.index)),
                "sample_rule": sample_rule,
                "declared_strata": "|".join(stratum_counts.keys()),
                "stratum_counts_json": json.dumps(stratum_counts, sort_keys=True),
                "strata_with_zero_selected_rows": "|".join(zero_selected),
                "sample_status": (
                    "sample_nonempty_all_declared_strata_selected"
                    if len(sample.index) and not zero_selected
                    else "sample_rule_failed"
                ),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(paths["large_artifact_sample_manifest"], index=False)


def _build_runtime_manifest() -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "PyYAML", "python-dateutil", "matplotlib"]
    rows: list[dict[str, object]] = [
        {"component": "python", "version": sys.version.split()[0], "detail": sys.executable},
        {"component": "platform", "version": platform.platform(), "detail": platform.machine()},
    ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            status = "present"
        except importlib.metadata.PackageNotFoundError:
            version = ""
            status = "missing"
        rows.append({"component": package, "version": version, "detail": status})
    rows.extend(
        [
            {
                "component": "rebuild_command",
                "version": "",
                "detail": "uv run --with pandas --with numpy --with scipy --with pyyaml --with python-dateutil --with matplotlib python scripts/build_historical_replay_tdcest_crosscheck.py",
            },
            {
                "component": "audit_command",
                "version": "",
                "detail": "uv run --with pandas python scripts/audit_historical_replay_closeout.py",
            },
        ]
    )
    return pd.DataFrame(rows)


def _build_runtime_lock() -> pd.DataFrame:
    packages = ["numpy", "pandas", "scipy", "PyYAML", "python-dateutil", "matplotlib"]
    rows = []
    uv_path = shutil.which("uv") or ""
    uv_version = ""
    if uv_path:
        try:
            uv_version = subprocess.run(
                [uv_path, "--version"],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            uv_version = "unavailable"
    rows.append({"package": "uv", "version": uv_version, "source": uv_path, "lock_status": "resolved_runtime_tool"})
    rows.append(
        {
            "package": "python",
            "version": sys.version.split()[0],
            "source": sys.executable,
            "lock_status": "resolved_runtime_interpreter",
        }
    )
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            status = "resolved_runtime_distribution"
        except importlib.metadata.PackageNotFoundError:
            version = ""
            status = "missing_runtime_distribution"
        rows.append({"package": package, "version": version, "source": package, "lock_status": status})
    return pd.DataFrame(rows)


def _deterministic_large_artifact_sample(frame: pd.DataFrame, strata_func) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    if frame.empty:
        return frame.copy(), {}
    stratum_frames = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for name, mask, limit in strata_func(frame):
        selected = frame.loc[mask].copy()
        source_count = int(len(selected.index))
        if limit is not None:
            selected = selected.head(limit)
        stratum_frames.append(selected)
        stratum_counts[name] = {
            "source_row_count": source_count,
            "selected_row_count": int(len(selected.index)),
        }
    sample = pd.concat(stratum_frames, ignore_index=False) if stratum_frames else frame.iloc[0:0].copy()
    sample = sample.drop_duplicates().reset_index(drop=True)
    return sample, stratum_counts


def _large_portfolio_sample_strata(frame: pd.DataFrame) -> list[tuple[str, pd.Series, int | None]]:
    quarter = frame.get("quarter", pd.Series("", index=frame.index)).astype(str)
    source_sector = frame.get("source_sector", pd.Series("", index=frame.index)).astype(str)
    native_sector = frame.get("native_sector", pd.Series("", index=frame.index)).astype(str)
    text = source_sector.str.cat(native_sector, sep="|")
    return [
        ("quarter_2023Q4", quarter.eq("2023Q4"), None),
        ("quarter_2024Q1", quarter.eq("2024Q1"), None),
        ("quarter_2025Q4", quarter.eq("2025Q4"), None),
        ("sector_money_market", text.str.contains("money_market", case=False, na=False), 5_000),
        ("sector_holding_companies", text.str.contains("holding_companies", case=False, na=False), 5_000),
        ("sector_source_basis_residual", text.str.contains("SourceBasisResidual", case=False, na=False), 5_000),
        ("deterministic_head", pd.Series(False, index=frame.index).mask(frame.index.isin(frame.head(2_500).index), True), None),
        ("deterministic_tail", pd.Series(False, index=frame.index).mask(frame.index.isin(frame.tail(2_500).index), True), None),
    ]


def _large_interest_sample_strata(frame: pd.DataFrame) -> list[tuple[str, pd.Series, int | None]]:
    quarter = frame.get("quarter", pd.Series("", index=frame.index)).astype(str)
    native_sector = frame.get("native_sector", pd.Series("", index=frame.index)).astype(str)
    return [
        ("quarter_2023Q4", quarter.eq("2023Q4"), None),
        ("quarter_2025Q4", quarter.eq("2025Q4"), None),
        ("sector_foreign", native_sector.str.contains("Foreign", case=False, na=False), 5_000),
        ("sector_money_market", native_sector.str.contains("money_market", case=False, na=False), 5_000),
        ("sector_banks", native_sector.str.contains("Banks", case=False, na=False), 5_000),
        ("deterministic_head", pd.Series(False, index=frame.index).mask(frame.index.isin(frame.head(2_500).index), True), None),
        ("deterministic_tail", pd.Series(False, index=frame.index).mask(frame.index.isin(frame.tail(2_500).index), True), None),
    ]


def _resolve_output_dir(cfg: dict) -> Path | None:
    output_dir = cfg.get("output_dir")
    outputs = cfg.get("outputs")
    if output_dir is None and isinstance(outputs, dict):
        output_dir = outputs.get("output_dir")
    if not output_dir:
        return None
    return Path(output_dir)


def _concat_period_end_portfolios(period_end_portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for quarter in _sort_quarters(period_end_portfolios.keys()):
        frame = period_end_portfolios[quarter].copy()
        if "quarter" in frame.columns:
            frame["quarter"] = quarter
        else:
            frame.insert(0, "quarter", quarter)
        frames.append(frame)
    if not frames:
        return _empty_portfolio_frame()
    return pd.concat(frames, ignore_index=True)


def _empty_portfolio_frame() -> pd.DataFrame:
    metadata_columns = [
        "quarter",
        "cohort_id",
        "cusip",
        "series_cd",
        "native_sector",
        "broad_holder_class",
        "tdcsim_holder",
        "tdcsim_holder_subbucket",
        "source_status",
        "security_enrichment_status",
        "source_sector",
        "allocated_outstanding_source",
        "interest_pay_date_1",
        "interest_pay_date_2",
        "interest_pay_date_3",
        "interest_pay_date_4",
    ]
    frame = pd.DataFrame(columns=BOND_PORTFOLIO_COLS + metadata_columns)
    return frame.astype(PORTFOLIO_DTYPES, errors="ignore")


def _build_security_term_invariants(portfolio: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "cohort_id",
        "cusip",
        "SecurityType",
        "MaturityCategory",
        "IssueDate",
        "MaturityDate",
        "OriginalMaturityYears",
        "date_implied_maturity_years",
        "term_error_years",
        "CouponRate",
        "IssueYieldAtIssue",
        "term_status",
    ]
    if portfolio.empty:
        return pd.DataFrame(columns=columns)
    required = ["quarter", "cohort_id", "cusip", "IssueDate", "MaturityDate", "OriginalMaturityYears"]
    available = [column for column in required if column in portfolio.columns]
    if len(available) < len(required):
        return pd.DataFrame(columns=columns)
    frame = portfolio.drop_duplicates(
        subset=["quarter", "cohort_id", "cusip"],
        keep="first",
    ).copy()
    issue = pd.to_datetime(frame["IssueDate"], errors="coerce")
    maturity = pd.to_datetime(frame["MaturityDate"], errors="coerce")
    original = pd.to_numeric(frame["OriginalMaturityYears"], errors="coerce")
    frame["date_implied_maturity_years"] = (maturity - issue).dt.days / 365.25
    frame["term_error_years"] = (original - frame["date_implied_maturity_years"]).abs()
    frame["term_status"] = frame["term_error_years"].map(
        lambda value: "matched" if pd.notna(value) and float(value) <= 0.02 else "term_mismatch"
    )
    return frame[[column for column in columns if column in frame.columns]]


def _build_tips_principal_identity(portfolio: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "cohort_id",
        "cusip",
        "IndexRatio",
        "OriginalPrincipal",
        "AdjustedPrincipal",
        "identity_gap",
        "identity_status",
    ]
    if portfolio.empty or "SecurityType" not in portfolio.columns:
        return pd.DataFrame(columns=columns)
    frame = portfolio.loc[portfolio["SecurityType"].astype(str) == "TIPS"].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        frame.assign(
            OriginalPrincipal=pd.to_numeric(frame["OriginalPrincipal"], errors="coerce").fillna(0.0),
            AdjustedPrincipal=pd.to_numeric(frame["AdjustedPrincipal"], errors="coerce").fillna(0.0),
            IndexRatio=pd.to_numeric(frame["IndexRatio"], errors="coerce"),
        )
        .groupby(["quarter", "cohort_id", "cusip", "IndexRatio"], dropna=False, sort=True)
        .agg({"OriginalPrincipal": "sum", "AdjustedPrincipal": "sum"})
        .reset_index()
    )
    grouped["identity_gap"] = grouped["AdjustedPrincipal"] - grouped["OriginalPrincipal"] * grouped["IndexRatio"]
    grouped["identity_status"] = grouped["identity_gap"].map(
        lambda value: "matched" if pd.notna(value) and abs(float(value)) <= 1.0e-6 else "identity_mismatch"
    )
    return grouped[columns]


def _build_native_sector_similarity_diagnostics(portfolio_snapshots: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "native_sector",
        "face_total_mil",
        "nonzero_cohorts",
        "hhi",
        "effective_cohort_count",
        "bill_share",
        "frn_share",
        "fixed_rate_gt_397d_share",
        "wam_years",
        "identical_profile_peer_count",
        "claim_boundary",
    ]
    if portfolio_snapshots.empty or not {
        "quarter",
        "native_sector",
        "cohort_id",
        "FaceValue",
    }.issubset(portfolio_snapshots.columns):
        return pd.DataFrame(columns=columns)

    frame = portfolio_snapshots.copy()
    source_text = frame.get("source_sector", pd.Series("", index=frame.index)).astype(str)
    frame = frame[~source_text.str.contains("SourceBasisResidual", na=False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["FaceValue"] = pd.to_numeric(frame["FaceValue"], errors="coerce").fillna(0.0)
    frame = frame[frame["FaceValue"].gt(0.0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    security_key = _cohort_text(frame, "SecurityType", "security_type").map(_normalize_simple)
    maturity_category = _cohort_text(frame, "MaturityCategory", "maturity_category").map(_normalize_simple)
    is_bill = security_key.isin({"bill", "bills", "cashmanagementbill", "cmb"}) | maturity_category.eq("bills")
    is_frn = security_key.isin({"frn", "floating", "floatingratenote"})
    quarter_end = pd.PeriodIndex(frame["quarter"].astype(str), freq="Q").to_timestamp(how="end").normalize()
    maturity = pd.to_datetime(frame.get("MaturityDate", frame.get("maturity_date")), errors="coerce")
    remaining_days = (maturity - quarter_end).dt.days
    remaining_years = remaining_days.clip(lower=0).fillna(0.0) / 365.25
    frame["_is_bill"] = is_bill.to_numpy(dtype=bool)
    frame["_is_frn"] = is_frn.to_numpy(dtype=bool)
    frame["_fixed_gt_397"] = (~is_bill & ~is_frn & remaining_days.gt(397).fillna(False)).to_numpy(dtype=bool)
    frame["_remaining_years"] = remaining_years.to_numpy(dtype=float)

    rows = []
    grouped = (
        frame.groupby(["quarter", "native_sector", "cohort_id"], as_index=False, sort=False)
        .agg(
            FaceValue=("FaceValue", "sum"),
            is_bill=("_is_bill", "max"),
            is_frn=("_is_frn", "max"),
            fixed_gt_397=("_fixed_gt_397", "max"),
            remaining_years=("_remaining_years", "max"),
        )
    )
    totals = grouped.groupby(["quarter", "native_sector"])["FaceValue"].transform("sum")
    grouped["share"] = grouped["FaceValue"] / totals.replace(0.0, pd.NA)
    grouped["share"] = grouped["share"].fillna(0.0)

    signature_counts: dict[tuple[str, tuple[tuple[str, float], ...]], int] = {}
    for (quarter, sector), sector_group in grouped.groupby(["quarter", "native_sector"], sort=False):
        signature = tuple(
            sorted(
                (
                    str(row["cohort_id"]),
                    round(float(row["share"]), 12),
                )
                for _, row in sector_group.iterrows()
                if float(row["share"]) > 1.0e-12
            )
        )
        signature_counts[(str(quarter), signature)] = signature_counts.get((str(quarter), signature), 0) + 1

    for (quarter, sector), sector_group in grouped.groupby(["quarter", "native_sector"], sort=False):
        total = float(sector_group["FaceValue"].sum())
        shares = pd.to_numeric(sector_group["share"], errors="coerce").fillna(0.0)
        hhi = float((shares**2).sum())
        signature = tuple(
            sorted(
                (
                    str(row["cohort_id"]),
                    round(float(row["share"]), 12),
                )
                for _, row in sector_group.iterrows()
                if float(row["share"]) > 1.0e-12
            )
        )
        rows.append(
            {
                "quarter": quarter,
                "native_sector": sector,
                "face_total_mil": total,
                "nonzero_cohorts": int((shares > 1.0e-12).sum()),
                "hhi": hhi,
                "effective_cohort_count": (1.0 / hhi) if hhi > 0.0 else 0.0,
                "bill_share": float(sector_group.loc[sector_group["is_bill"].astype(bool), "FaceValue"].sum() / total)
                if total
                else 0.0,
                "frn_share": float(sector_group.loc[sector_group["is_frn"].astype(bool), "FaceValue"].sum() / total)
                if total
                else 0.0,
                "fixed_rate_gt_397d_share": float(
                    sector_group.loc[sector_group["fixed_gt_397"].astype(bool), "FaceValue"].sum() / total
                )
                if total
                else 0.0,
                "wam_years": float((sector_group["FaceValue"] * sector_group["remaining_years"]).sum() / total)
                if total
                else 0.0,
                "identical_profile_peer_count": max(0, signature_counts.get((str(quarter), signature), 1) - 1),
                "claim_boundary": "synthetic_max_entropy_nonidentified_cells_not_observed_holder_security_history",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_valuation_basis_feasibility_certificate(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "quarter",
        "subject",
        "valuation_basis",
        "input_total_mil",
        "achieved_total_mil",
        "residual_mil",
        "slack_positive_mil",
        "slack_negative_mil",
        "abs_slack_mil",
        "abs_residual_mil",
        "cohort_total_mil",
        "relative_to_cohort_total",
        "relative_to_sector_target",
        "solver_method",
        "solver_status",
        "certificate_status",
        "claim_boundary",
    ]
    if diagnostics.empty or "diagnostic_type" not in diagnostics.columns:
        return pd.DataFrame(columns=columns)

    sector = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("sector_balance")].copy()
    totals = diagnostics[diagnostics["diagnostic_type"].astype(str).eq("total_balance")].copy()
    if sector.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "input_total",
        "achieved_total",
        "residual",
        "slack_positive_mil",
        "slack_negative_mil",
    ]:
        if column not in sector.columns:
            sector[column] = 0.0
        sector[column] = pd.to_numeric(sector[column], errors="coerce").fillna(0.0)
    slack_abs = sector[["slack_positive_mil", "slack_negative_mil"]].abs().max(axis=1)
    residual_abs = sector["residual"].abs()
    material = sector[slack_abs.gt(1.0e-2) | residual_abs.gt(1.0e-2)].copy()
    if material.empty:
        return pd.DataFrame(columns=columns)

    cohort_totals = pd.Series(dtype=float)
    if not totals.empty and {"quarter", "target_total"}.issubset(totals.columns):
        cohort_totals = pd.to_numeric(totals["target_total"], errors="coerce")
        cohort_totals.index = totals["quarter"].astype(str)
    material["abs_slack_mil"] = material[["slack_positive_mil", "slack_negative_mil"]].abs().max(axis=1)
    material["abs_residual_mil"] = material["residual"].abs()
    material["cohort_total_mil"] = material["quarter"].astype(str).map(cohort_totals).fillna(0.0)
    material["relative_to_cohort_total"] = material["abs_slack_mil"] / material["cohort_total_mil"].replace(0.0, pd.NA)
    material["relative_to_cohort_total"] = material["relative_to_cohort_total"].fillna(0.0)
    material["relative_to_sector_target"] = material["abs_slack_mil"] / material["input_total"].abs().replace(0.0, pd.NA)
    material["relative_to_sector_target"] = material["relative_to_sector_target"].fillna(0.0)
    valuation_basis = material.get("valuation_basis", pd.Series("", index=material.index)).astype(str)
    solver_method = material.get("solver_method", pd.Series("", index=material.index)).astype(str)
    certified_methods = {
        "exact_weighted_entropy_projection_with_certified_slack",
        "aggregate_only_minimum_weighted_slack_certificate",
    }
    material["certificate_status"] = np.where(
        valuation_basis.str.contains("market_value|book_or_par", case=False, na=False)
        & solver_method.isin(certified_methods),
        "certified_minimum_l1_model_slack",
        "uncertified_constraint_residual",
    )
    material["claim_boundary"] = (
        "source_valuation_basis_constraints_checked_with_minimum_l1_model_slack_not_silently_rescaled"
    )
    out = material.rename(
        columns={
            "input_total": "input_total_mil",
            "achieved_total": "achieved_total_mil",
            "residual": "residual_mil",
            "status": "solver_status",
        }
    )
    return out[[column for column in columns if column in out.columns]].copy()


def _build_strips_scope_diagnostics(
    *,
    cohorts: pd.DataFrame,
    period_end_portfolios: dict[str, pd.DataFrame],
    final_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "surface",
        "row_count",
        "strips_like_rows",
        "strips_like_amount",
        "amount_column",
        "status",
        "evidence_label",
    ]
    portfolio_snapshots = _concat_period_end_portfolios(period_end_portfolios)
    surfaces = [
        ("loaded_mspd_cohorts", cohorts, "outstanding"),
        ("portfolio_snapshots", portfolio_snapshots, "FaceValue"),
        ("final_portfolio", final_portfolio, "FaceValue"),
    ]
    rows = []
    for surface, frame, amount_column in surfaces:
        if frame is None or frame.empty:
            rows.append(
                {
                    "surface": surface,
                    "row_count": 0,
                    "strips_like_rows": 0,
                    "strips_like_amount": 0.0,
                    "amount_column": amount_column,
                    "status": "empty_surface",
                    "evidence_label": "not_applicable",
                }
            )
            continue
        mask = _strips_like_mask(frame)
        amounts = (
            pd.to_numeric(frame[amount_column], errors="coerce").fillna(0.0)
            if amount_column in frame.columns
            else pd.Series([0.0] * len(frame.index), index=frame.index)
        )
        strips_rows = int(mask.sum())
        rows.append(
            {
                "surface": surface,
                "row_count": int(len(frame.index)),
                "strips_like_rows": strips_rows,
                "strips_like_amount": float(amounts.loc[mask].sum()) if strips_rows else 0.0,
                "amount_column": amount_column if amount_column in frame.columns else pd.NA,
                "status": "strips_like_rows_present" if strips_rows else "no_strips_detected",
                "evidence_label": "loaded_security_text_scan",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _strips_like_mask(frame: pd.DataFrame) -> pd.Series:
    candidate_columns = [
        "SecurityType",
        "security_type",
        "MaturityCategory",
        "maturity_category",
        "security_class1_desc",
        "security_class2_desc",
        "security_type_desc",
        "series_cd",
    ]
    mask = pd.Series(False, index=frame.index)
    for column in candidate_columns:
        if column not in frame.columns:
            continue
        text = frame[column].where(frame[column].notna(), "").astype(str).str.strip().str.lower()
        mask = mask | text.str.contains("strip|zero coupon|corpus|principal strip|interest strip", regex=True)
    return mask


def _build_valuation_scope_diagnostics(
    *,
    period_end_portfolios: dict[str, pd.DataFrame],
    final_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    price_columns = [
        "TimeToMaturity",
        "DiscountYield",
        "CleanPrice",
        "AccruedInterest",
        "DirtyValue",
        "DirtyPriceRatio",
    ]
    columns = [
        "surface",
        "row_count",
        "pricing_columns_present",
        "rows_with_any_pricing_value",
        "clean_price_non_null_rows",
        "dirty_value_non_null_rows",
        "valuation_scope_status",
        "claim_boundary",
    ]
    portfolio_snapshots = _concat_period_end_portfolios(period_end_portfolios)
    rows = []
    for surface, frame in [
        ("portfolio_snapshots", portfolio_snapshots),
        ("final_portfolio", final_portfolio),
    ]:
        if frame is None or frame.empty:
            rows.append(
                {
                    "surface": surface,
                    "row_count": 0,
                    "pricing_columns_present": "",
                    "rows_with_any_pricing_value": 0,
                    "clean_price_non_null_rows": 0,
                    "dirty_value_non_null_rows": 0,
                    "valuation_scope_status": "empty_surface",
                    "claim_boundary": "not_applicable",
                }
            )
            continue
        present = [column for column in price_columns if column in frame.columns]
        if present:
            priced_mask = frame[present].notna().any(axis=1)
        else:
            priced_mask = pd.Series(False, index=frame.index)
        clean_non_null = int(frame["CleanPrice"].notna().sum()) if "CleanPrice" in frame.columns else 0
        dirty_non_null = int(frame["DirtyValue"].notna().sum()) if "DirtyValue" in frame.columns else 0
        priced_rows = int(priced_mask.sum())
        rows.append(
            {
                "surface": surface,
                "row_count": int(len(frame.index)),
                "pricing_columns_present": ";".join(present),
                "rows_with_any_pricing_value": priced_rows,
                "clean_price_non_null_rows": clean_non_null,
                "dirty_value_non_null_rows": dirty_non_null,
                "valuation_scope_status": (
                    "pricing_values_present" if priced_rows else "par_current_principal_only_no_pricing_bridge"
                ),
                "claim_boundary": (
                    "quarter_end_market_value_claims_not_supported"
                    if not priced_rows
                    else "pricing_values_require_separate_reconciliation"
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_treasury_interest_expense_diagnostic_from_manifest(
    tdc_source_manifest: pd.DataFrame | None,
    interest_component_detail: pd.DataFrame,
) -> pd.DataFrame:
    if tdc_source_manifest is None or tdc_source_manifest.empty:
        return build_treasury_interest_expense_diagnostic(pd.DataFrame(), interest_component_detail)
    treasury_interest = _read_manifest_csv(tdc_source_manifest, "treasury_interest_expense")
    tier2_candidate = _read_manifest_csv(tdc_source_manifest, "tier2_interest_component_candidate")
    return build_treasury_interest_expense_diagnostic(
        treasury_interest,
        interest_component_detail,
        tier2_interest_candidate=tier2_candidate,
    )


def _read_manifest_csv(manifest: pd.DataFrame, source_key: str) -> pd.DataFrame:
    if "source_key" not in manifest.columns or "path" not in manifest.columns:
        return pd.DataFrame()
    matches = manifest.loc[manifest["source_key"].astype(str).eq(source_key)]
    if matches.empty:
        return pd.DataFrame()
    raw_path = matches.iloc[0].get("path")
    if pd.isna(raw_path) or not str(raw_path).strip():
        return pd.DataFrame()
    path = _resolve_project_path(raw_path)
    if not path.exists() or path.suffix.lower() != ".csv":
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _build_replay_input_manifest(cfg: dict, tdc_source_manifest: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "source_key",
        "path",
        "role",
        "required",
        "configured",
        "present",
        "consumed_in_run",
        "required_for_claim",
        "source_usage",
        "source_vintage",
        "status",
        "bytes",
        "sha256",
        "row_count",
        "first_quarter",
        "last_quarter",
    ]
    rows = []
    paths = cfg.get("paths", {}) or {}
    auction_cfg = cfg.get("auction_allotment_proxy", {})
    auction_paths = auction_cfg if isinstance(auction_cfg, dict) else {}
    base_inputs = [
        ("quarterly_cash", paths.get("cash"), "treasury_operating_cash_and_tga_ttl", True),
        ("sector_positions", paths.get("sector_positions"), "z1_holder_sector_positions", True),
        (
            "z1_raw_archive",
            paths.get("z1_raw_archive", "data/historical_replay/raw/z1/z1_csv_files.zip"),
            "z1_l210_f210_raw_archive_lineage",
            True,
        ),
        (
            "z1_source_record",
            paths.get("z1_source_record", "data/historical_replay/raw/z1/z1_source_record.json"),
            "z1_l210_f210_archive_hash_and_transform_record",
            True,
        ),
        ("mspd_cohorts", paths.get("cohorts"), "mspd_security_cohort_panel", True),
        (
            "auction_terms",
            paths.get("auctions", "data/historical_replay/raw/fiscaldata/auctions_query.csv"),
            "treasury_auction_security_terms",
            True,
        ),
        (
            "frn_daily_indexes",
            paths.get("frn_daily_indexes", "data/historical_replay/raw/fiscaldata/frn_daily_indexes.csv"),
            "frn_index_and_accrual_reference",
            True,
        ),
        (
            "tips_cpi",
            paths.get("tips_cpi", "data/historical_replay/raw/fiscaldata/tips_cpi_data_detail.csv"),
            "tips_cpi_index_reference",
            True,
        ),
        (
            "soma_treasury_holdings_monthly",
            cfg.get("soma_treasury_holdings", "data/historical_replay/raw/nyfed/soma_treasury_holdings_monthly.csv"),
            "nyfed_soma_observed_fed_portfolio_block",
            True,
        ),
        (
            "pricing_curve_1m",
            paths.get("pricing_curve_1m", "data/historical_replay/raw/fred/treasury_1m_daily__DGS1MO.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_3m",
            paths.get("pricing_curve_3m", "data/historical_replay/raw/fred/treasury_3m_daily__DGS3MO.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_6m",
            paths.get("pricing_curve_6m", "data/historical_replay/raw/fred/treasury_6m_daily__DGS6MO.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_1y",
            paths.get("pricing_curve_1y", "data/historical_replay/raw/fred/treasury_1y_daily__DGS1.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_2y",
            paths.get("pricing_curve_2y", "data/historical_replay/raw/fred/treasury_2y_daily__DGS2.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_3y",
            paths.get("pricing_curve_3y", "data/historical_replay/raw/fred/treasury_3y_daily__DGS3.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_5y",
            paths.get("pricing_curve_5y", "data/historical_replay/raw/fred/treasury_5y_daily__DGS5.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_7y",
            paths.get("pricing_curve_7y", "data/historical_replay/raw/fred/treasury_7y_daily__DGS7.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_10y",
            paths.get("pricing_curve_10y", "data/historical_replay/raw/fred/treasury_10y_daily__DGS10.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_20y",
            paths.get("pricing_curve_20y", "data/historical_replay/raw/fred/treasury_20y_daily__DGS20.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_30y",
            paths.get("pricing_curve_30y", "data/historical_replay/raw/fred/treasury_30y_daily__DGS30.csv"),
            "model_pricing_nominal_yield_curve_input",
            True,
        ),
        (
            "pricing_curve_3m_monthly",
            paths.get("pricing_curve_3m_monthly", "data/historical_replay/raw/fred/tbill_3m_monthly__TB3MS.csv"),
            "model_pricing_nominal_yield_curve_fallback_input",
            True,
        ),
        (
            "pricing_curve_10y_monthly",
            paths.get("pricing_curve_10y_monthly", "data/historical_replay/raw/fred/treasury_10y_monthly__GS10.csv"),
            "model_pricing_nominal_yield_curve_fallback_input",
            True,
        ),
        (
            "real_pricing_curve_5y",
            paths.get("real_pricing_curve_5y", "data/historical_replay/raw/fred/tips_5y_daily__DFII5.csv"),
            "model_pricing_tips_real_yield_curve_input",
            True,
        ),
        (
            "real_pricing_curve_7y",
            paths.get("real_pricing_curve_7y", "data/historical_replay/raw/fred/tips_7y_daily__DFII7.csv"),
            "model_pricing_tips_real_yield_curve_input",
            True,
        ),
        (
            "real_pricing_curve_10y",
            paths.get("real_pricing_curve_10y", "data/historical_replay/raw/fred/tips_10y_daily__DFII10.csv"),
            "model_pricing_tips_real_yield_curve_input",
            True,
        ),
        (
            "real_pricing_curve_20y",
            paths.get("real_pricing_curve_20y", "data/historical_replay/raw/fred/tips_20y_daily__DFII20.csv"),
            "model_pricing_tips_real_yield_curve_input",
            True,
        ),
        (
            "real_pricing_curve_30y",
            paths.get("real_pricing_curve_30y", "data/historical_replay/raw/fred/tips_30y_daily__DFII30.csv"),
            "model_pricing_tips_real_yield_curve_input",
            True,
        ),
        (
            "auction_allotment_panel",
            auction_paths.get("allotment_panel", "data/historical_replay/imported/buycurve/auction_allotment_panel_base_slim.csv"),
            "auction_allotment_holder_proxy",
            True,
        ),
        (
            "ffiec_interest_constraints",
            cfg.get("ffiec_interest_constraints", "data/historical_replay/imported/tdcest/ffiec_interest_constraints_normalized.csv"),
            "bank_interest_constraint_reference",
            True,
        ),
        (
            "ncua_interest_constraints",
            cfg.get("ncua_interest_constraints", "data/historical_replay/imported/tdcest/ncua_interest_constraints_normalized.csv"),
            "credit_union_interest_constraint_reference",
            True,
        ),
        (
            "tier2_interest_constraints",
            cfg.get("tier2_interest_constraints", "data/historical_replay/imported/tdcest/tier2_interest_source_constraints.csv"),
            "tier2_interest_source_constraint_reference",
            True,
        ),
        (
            "mmf_component_constraints",
            cfg.get("mmf_component_constraints", "data/historical_replay/imported/tdcest/quarterly_inputs.csv"),
            "mmf_total_and_treasury_bill_component_constraint",
            True,
        ),
    ]
    for source_key, path, role, required in base_inputs:
        consumed = False if source_key in {"z1_raw_archive", "z1_source_record"} else None
        rows.append(_input_manifest_row(source_key, path, role=role, required=required, consumed_in_run=consumed))
    if tdc_source_manifest is not None and not tdc_source_manifest.empty:
        for _, row in tdc_source_manifest.iterrows():
            raw_source_key = str(row.get("source_key"))
            source_key = f"tdc_{raw_source_key}"
            required = bool(row.get("required", True)) or raw_source_key in REQUIRED_TDCEST_REPLAY_SOURCE_KEYS
            consumed_in_run = raw_source_key in RUNTIME_CONSUMED_TDCEST_SOURCE_KEYS
            manifest_row = _input_manifest_row(
                source_key,
                row.get("path"),
                role=row.get("role", "tdc_source_input"),
                required=required,
                consumed_in_run=consumed_in_run,
                required_for_claim=required,
            )
            for column in ["sha256", "row_count", "first_quarter", "last_quarter"]:
                if column in row and pd.notna(row[column]):
                    manifest_row[column] = row[column]
            rows.append(manifest_row)
    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["source_key", "path"], keep="first")


def _input_manifest_row(
    source_key: str,
    raw_path,
    *,
    role: object,
    required: bool,
    consumed_in_run: bool | None = None,
    required_for_claim: bool | None = None,
) -> dict[str, object]:
    configured = raw_path is not None and pd.notna(raw_path) and bool(str(raw_path).strip())
    consumed = bool(required) if consumed_in_run is None else bool(consumed_in_run)
    claim_required = bool(required) if required_for_claim is None else bool(required_for_claim)
    row = {
        "source_key": source_key,
        "path": _display_path(raw_path) if pd.notna(raw_path) else "",
        "role": role,
        "required": bool(required),
        "configured": configured,
        "present": False,
        "consumed_in_run": consumed,
        "required_for_claim": claim_required,
        "source_usage": "runtime_consumed" if consumed else "claim_evidence_not_runtime_consumed",
        "source_vintage": _source_vintage_label(source_key),
        "status": "missing",
        "bytes": pd.NA,
        "sha256": pd.NA,
        "row_count": pd.NA,
        "first_quarter": pd.NA,
        "last_quarter": pd.NA,
    }
    if not configured:
        row["status"] = "not_configured"
        return row
    path = _resolve_project_path(raw_path)
    if not path.exists() or not path.is_file():
        return row
    row["path"] = _display_path(path)
    row["status"] = "present"
    row["present"] = True
    row["bytes"] = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    row["sha256"] = digest.hexdigest()
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                next(handle, None)
                row["row_count"] = sum(1 for _ in handle)
        except OSError:
            row["row_count"] = pd.NA
    return row


def _source_vintage_label(source_key: str) -> str:
    if source_key in {"sector_positions", "z1_raw_archive", "z1_source_record"}:
        return "local_z1_l210_download_current_as_of_2026-06-11_z1_2026q1_release"
    return ""


def _build_replay_config_hash(
    cfg: dict,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
    replay_input_manifest: pd.DataFrame | None,
) -> str:
    manifest_sha = ""
    if replay_input_manifest is not None and not replay_input_manifest.empty:
        manifest_sha = hashlib.sha256(
            replay_input_manifest.sort_index(axis=1).to_csv(index=False).encode("utf-8")
        ).hexdigest()
    payload = {
        "cfg": _canonicalize_config_for_hash(cfg),
        "start_quarter": start_quarter,
        "end_quarter": end_quarter,
        "input_manifest_sha256": manifest_sha,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_config_for_hash(value):
    if isinstance(value, dict):
        return {str(key): _canonicalize_config_for_hash(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonicalize_config_for_hash(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_config_for_hash(item) for item in value)
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, str):
        try:
            path = Path(value)
        except TypeError:
            return value
        if path.is_absolute() or "/" in value or "\\" in value:
            return _display_path(path)
    return value


def _code_identity_rows() -> list[dict[str, object]]:
    source_paths = (
        sorted((_PROJECT_ROOT / "src").glob("*.py"))
        + sorted((_PROJECT_ROOT / "scripts").glob("*.py"))
        + sorted(_PROJECT_ROOT.glob("tdc_config*.yaml"))
    )
    rows = []
    for path in source_paths:
        if not path.is_file():
            continue
        rows.append(
            {
                "path": _display_path(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _build_code_identity_manifest() -> pd.DataFrame:
    return pd.DataFrame(_code_identity_rows(), columns=["path", "bytes", "sha256"])


def _build_code_identity_hash() -> str:
    digest = hashlib.sha256()
    for row in _code_identity_rows():
        path = _resolve_project_path(row["path"])
        digest.update(_display_path(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_run_id(*, config_sha256: str, code_sha256: str) -> str:
    payload = json.dumps(
        {
            "config_sha256": config_sha256,
            "code_sha256": code_sha256,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _build_artifact_integrity(
    paths: dict[str, str],
    *,
    run_id: str,
    config_sha256: str,
    code_sha256: str,
) -> pd.DataFrame:
    rows = []
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path)
        if not path.exists() or not path.is_file():
            rows.append(
                {
                    "artifact": name,
                    "path": _display_path(path),
                    "status": "missing",
                    "run_id": run_id,
                    "config_sha256": config_sha256,
                    "code_sha256": code_sha256,
                }
            )
            continue
        header = ""
        row_count = pd.NA
        column_count = pd.NA
        header_sha256 = pd.NA
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                header = handle.readline().rstrip("\n")
                row_count = sum(1 for _ in handle)
            column_count = len(header.split(",")) if header else 0
            header_sha256 = hashlib.sha256(header.encode("utf-8")).hexdigest() if header else pd.NA
        rows.append(
            {
                "artifact": name,
                "path": _display_path(path),
                "status": "present",
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "row_count": row_count,
                "column_count": column_count,
                "header_sha256": header_sha256,
                "run_id": run_id,
                "config_sha256": config_sha256,
                "code_sha256": code_sha256,
            }
        )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _display_path(raw_path: object) -> str:
    path = Path(str(raw_path))
    try:
        if path.is_absolute():
            return str(path.relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)
    return str(path)


def _build_interest_proxy_alignment_summary(alignment: pd.DataFrame) -> str:
    if alignment.empty:
        return "# Interest Proxy Alignment Summary\n\nNo interest alignment rows were emitted.\n"
    rows = len(alignment.index)
    referenced = int(pd.to_numeric(alignment["tdcest_point"], errors="coerce").notna().sum())
    feasible = alignment["within_feasible_bounds"].fillna(False).astype(bool)
    stock_tolerance = alignment["stock_only_within_tolerance"].fillna(False).astype(bool)
    constrained_tolerance = alignment["constrained_within_tolerance"].fillna(False).astype(bool)
    constrained_gap = pd.to_numeric(alignment["constrained_gap"], errors="coerce").abs()
    stock_gap = pd.to_numeric(alignment["stock_only_gap"], errors="coerce").abs()
    method_counts = alignment["method_tier"].value_counts(dropna=False)
    lines = [
        "# Interest Proxy Alignment Summary",
        "",
        f"- Rows: {rows}",
        f"- Rows with TDC-EST reference point: {referenced}",
        f"- Rows within single-holder stock bounds: {int(feasible.sum())}",
        f"- Stock-only rows within exact tolerance: {int(stock_tolerance.sum())}",
        f"- Constrained/projection rows within exact tolerance: {int(constrained_tolerance.sum())}",
        f"- Max absolute stock-only gap (mil): {stock_gap.max() if not stock_gap.empty else 'NA'}",
        f"- Max absolute constrained/projection gap (mil): {constrained_gap.max() if not constrained_gap.empty else 'NA'}",
        "",
        "Method counts:",
    ]
    for method, count in method_counts.items():
        lines.append(f"- {method}: {int(count)}")
    lines.extend(
        [
            "",
            "The constrained proxy is a single-holder stock-bound projection. It is a feasibility diagnostic, not a final event-ledger interest solution.",
            "",
        ]
    )
    return "\n".join(lines)


def _total_balance_by_quarter(diagnostics: pd.DataFrame) -> dict[str, dict[str, object]]:
    if diagnostics.empty:
        return {}
    required = {"quarter", "diagnostic_type"}
    if not required.issubset(diagnostics.columns):
        return {}
    total_rows = diagnostics[diagnostics["diagnostic_type"] == "total_balance"].copy()
    out: dict[str, dict[str, object]] = {}
    for _, row in total_rows.iterrows():
        out[str(row["quarter"])] = row.to_dict()
    return out


def _allocation_by_quarter_holder(allocations: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if allocations.empty:
        return out
    frame = allocations.copy()
    if "tdcsim_holder" in frame.columns:
        frame["holder"] = frame["tdcsim_holder"].apply(_holder_for_sector)
    elif "broad_holder_class" in frame.columns:
        frame["holder"] = frame["broad_holder_class"].apply(_holder_for_sector)
    else:
        frame["holder"] = frame["sector"].apply(_holder_for_sector)
    grouped = frame.groupby(["quarter", "holder"])["allocated_outstanding"].sum()
    for (quarter, holder), value in grouped.items():
        out.setdefault(quarter, {})[holder] = float(value)
    return out


def _tdc_by_quarter(tdc_panel: pd.DataFrame | None) -> dict[str, pd.Series]:
    if tdc_panel is None or tdc_panel.empty or "quarter" not in tdc_panel.columns:
        return {}
    return {str(row["quarter"]): row for _, row in tdc_panel.iterrows()}


def _panel_num(row: pd.Series, column: str, amount_unit_scale: float):
    value = row.get(column, pd.NA)
    if pd.isna(value):
        return pd.NA
    return float(value) / amount_unit_scale


def _sort_quarters(values) -> list[str]:
    return sorted((str(value) for value in values), key=lambda value: pd.Period(value, freq="Q"))


def _last_quarter(values) -> str | None:
    quarters = _sort_quarters(values)
    return quarters[-1] if quarters else None


def _normalize_simple(value: object) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _is_bill_cohort(cohorts: pd.DataFrame) -> pd.Series:
    if cohorts.empty:
        return pd.Series(dtype=bool)
    security = _cohort_text(cohorts, "security_type", "SecurityType")
    maturity_category = _cohort_text(cohorts, "maturity_category", "MaturityCategory")
    security_key = security.map(_normalize_simple)
    category_key = maturity_category.map(_normalize_simple)
    return security_key.isin({"bill", "bills", "cashmanagementbill", "cmb"}) | category_key.eq("bills")


def _is_mmf_nonbill_eligible(cohorts: pd.DataFrame) -> pd.Series:
    if cohorts.empty:
        return pd.Series(dtype=bool)
    bill = _is_bill_cohort(cohorts)
    security_key = _cohort_text(cohorts, "security_type", "SecurityType").map(_normalize_simple)
    frn = security_key.isin({"frn", "floating", "floatingratenote"})
    quarter_end = pd.PeriodIndex(cohorts["quarter"].astype(str), freq="Q").to_timestamp(how="end").normalize()
    maturity_col = "maturity_date" if "maturity_date" in cohorts.columns else "MaturityDate"
    maturity = pd.to_datetime(cohorts.get(maturity_col), errors="coerce")
    remaining_days = (maturity - quarter_end).dt.days
    short_fixed = remaining_days.le(397).fillna(False)
    return (~bill) & (frn | short_fixed)


def _cohort_text(frame: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return frame[column].where(frame[column].notna(), "").astype(str)
    return pd.Series("", index=frame.index, dtype=object)


def _join_unique_values(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna().tolist() if str(value).strip()})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return ";".join(unique)


def _holder_for_sector(value) -> str:
    text = str(value).strip().lower()
    if text.startswith("private:"):
        return "Private"
    if text in {
        "banks",
        "bank",
        "depository",
        "depository_institutions",
        "credit_unions",
        "us_chartered_depository_institutions",
        "foreign_banking_offices_in_us",
        "banks_in_us_affiliated_areas",
    }:
        return "Banks"
    if text in {"cb", "fed", "federal_reserve", "central_bank", "monetary_authority", "federal_reserve"}:
        return "CB"
    if text in {"foreign", "row", "rest_of_world", "foreign_official", "foreign_international"}:
        return "Foreign"
    if text in {"fedinternal", "fed_internal"}:
        return "FedInternal"
    if text in {"trustfunds", "trust_funds"}:
        return "TrustFunds"
    if text in {"sourcebasisresidual", "source_basis_residual", "mspd_z1_sourcebasisresidual"}:
        return "source_basis_residual"
    if text in {
        "abs_issuers",
        "dealers",
        "domestic_nonbank_other",
        "government_sponsored_enterprises",
        "individuals",
        "investment_funds",
        "money_market_cash",
        "money_market_funds",
        "other_financial",
        "otherfinancial",
        "pensions_insurers",
        "state_local_government",
        "private",
        "brokers_and_dealers",
        "closed_end_funds",
        "exchange_traded_funds",
        "federal_government_pension_funds",
        "government_sponsored_enterprises",
        "holding_companies",
        "household_sector",
        "life_insurance",
        "mutual_funds",
        "nonfinancial_corporate_business",
        "nonfinancial_noncorporate_business",
        "other_financial_business_central_clearing_counterparties",
        "private_pension_funds",
        "property_casualty_insurance",
        "state_local_government_pension_funds",
        "state_local_governments",
    }:
        return "Private"
    raise ValueError(f"unmapped historical replay holder class: {value!r}")


def _private_subbucket_for_sector(row: pd.Series) -> str:
    holder = _holder_for_sector(row.get("broad_holder_class", row.get("sector")))
    if holder != "Private":
        return ""
    key = str(row.get("broad_holder_class", row.get("sector"))).strip().lower()
    if key in {"money_market_cash", "money_market_funds", "mmf"}:
        return PRIVATE_SUBBUCKET_MMF
    return PRIVATE_SUBBUCKET_DOMESTIC_NONBANK


def _maybe_diff(left, right):
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return float(left) - float(right)


def _num(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


__all__ = ["REPLAY_RESULT_COLUMNS", "run_historical_replay"]
