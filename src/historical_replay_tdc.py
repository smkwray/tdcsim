"""TDC target construction for quarterly historical replay."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from historical_replay_tdc_inputs import load_tdcest_replay_sources, normalize_date_and_quarter


CANONICAL_TIER2_ROW = "tdc_tier2_canonical_depository_institution_mmf_rrp_prop_ru_flow"
COMPONENT_ANCHORED_CANONICAL_ROW = "tdc_tier2_component_anchored_mmf_rrp_prop_depository_institution_np_cu_ru_flow"
REGRESSION_DI_ROW = "tdc_tier2_regression_mmf_rrp_prop_depository_institution_np_cu_ru_flow"
REGRESSION_BANK_ONLY_ROW = "tdc_tier2_regression_mmf_rrp_prop_bank_only_ru_flow"

MODERN_CANONICAL_REQUIRED_COLUMNS = (
    "fed_tsy_tx",
    "us_chartered_tsy_tx",
    "foreign_offices_tsy_tx",
    "affiliated_areas_tsy_tx",
    "np_credit_unions_tsy_tx",
    "row_tsy_tx",
    "treasury_operating_cash_tx",
    "fed_remit_or_deferred",
    "fed_tsy_coupon_interest_proxy",
    "bank_tier2_component_interest_proxy",
    "row_tier2_component_interest_proxy",
    "credit_union_tier2_component_interest_proxy",
    "mmf_rrp_adjustment_prop",
)

TDC_PANEL_NUMERIC_COLUMNS = (
    "selected_tdc_value_mil",
    "selected_tdc_value_bil",
    "opening_tdc_level_mil",
    "closing_tdc_level_mil",
    "tdc_fiscal_flow_mil",
    "tdc_debt_service_mil",
    "tdc_auction_absorption_mil",
    "tdc_secondary_trades_mil",
    "tdc_other_mil",
    "tdc_residual_mil",
    "auction_identified_primary_allotment_mil",
    "auction_identified_du_absorption_gross_mil",
    "auction_non_du_identified_allotment_mil",
    "auction_bridge_or_unresolved_mil",
    "auction_accepted_amount_mil",
    "auction_allotment_reconciliation_gap_mil",
    "treasury_to_ru_transfer_share_of_deficit",
    "assumed_treasury_to_ru_transfer_mil",
    "tdc_unobserved_ru_transfer_contribution_mil",
    "deficit_mil",
)


def build_historical_replay_tdc_panel(
    source_overrides: dict | None = None,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    treasury_to_ru_transfer_share_of_deficit: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the selected TDC replay panel and source manifest.

    Returns `(tdc_panel, modern_formula_crosscheck, source_manifest)`.
    """

    sources, manifest = load_tdcest_replay_sources(source_overrides)
    panel = build_selected_tdc_panel(
        sources,
        treasury_to_ru_transfer_share_of_deficit=treasury_to_ru_transfer_share_of_deficit,
    )
    if "tier2_interest_component_candidate" in sources:
        panel = _merge_interest_component_candidate(
            panel,
            sources["tier2_interest_component_candidate"],
        )
    formula = build_modern_canonical_formula_crosscheck(
        sources["quarterly_inputs"],
        sources["tdc_estimates"],
    )
    panel = _filter_quarters(panel, start_quarter=start_quarter, end_quarter=end_quarter)
    formula = _filter_quarters(formula, start_quarter=start_quarter, end_quarter=end_quarter)
    return panel, formula, manifest


def build_modern_canonical_formula_crosscheck(
    quarterly_inputs: pd.DataFrame,
    tdc_estimates: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute the modern canonical Tier 2 row from source columns."""

    quarterly = normalize_date_and_quarter(quarterly_inputs, dataset_name="quarterly_inputs")
    estimates = normalize_date_and_quarter(tdc_estimates, dataset_name="tdc_estimates")
    _require_columns(quarterly, MODERN_CANONICAL_REQUIRED_COLUMNS, dataset_name="quarterly_inputs")
    _require_columns(
        estimates,
        [CANONICAL_TIER2_ROW, COMPONENT_ANCHORED_CANONICAL_ROW],
        dataset_name="tdc_estimates",
    )

    out = quarterly[["date", "quarter", *MODERN_CANONICAL_REQUIRED_COLUMNS]].copy()
    for column in MODERN_CANONICAL_REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["bank_depository_tsy_tx"] = out[
        ["us_chartered_tsy_tx", "foreign_offices_tsy_tx", "affiliated_areas_tsy_tx"]
    ].sum(axis=1, min_count=1)
    out["broad_depository_np_cu_tsy_tx"] = (
        out["bank_depository_tsy_tx"] + out["np_credit_unions_tsy_tx"]
    )
    out["base_di_ru_flow_mil"] = (
        out["fed_tsy_tx"]
        + out["broad_depository_np_cu_tsy_tx"]
        + out["row_tsy_tx"]
        - out["treasury_operating_cash_tx"]
        + out["fed_remit_or_deferred"]
    )
    out["canonical_tier2_recomputed_mil"] = (
        out["base_di_ru_flow_mil"]
        - out["fed_tsy_coupon_interest_proxy"]
        - out["bank_tier2_component_interest_proxy"]
        - out["row_tier2_component_interest_proxy"]
        - out["credit_union_tier2_component_interest_proxy"]
        + out["mmf_rrp_adjustment_prop"].fillna(0.0)
    )

    published = estimates[["date", CANONICAL_TIER2_ROW, COMPONENT_ANCHORED_CANONICAL_ROW]].copy()
    published[CANONICAL_TIER2_ROW] = pd.to_numeric(published[CANONICAL_TIER2_ROW], errors="coerce")
    published[COMPONENT_ANCHORED_CANONICAL_ROW] = pd.to_numeric(
        published[COMPONENT_ANCHORED_CANONICAL_ROW], errors="coerce"
    )
    out = out.merge(published, on="date", how="left", validate="one_to_one")
    out = out.loc[out[CANONICAL_TIER2_ROW].notna()].copy()
    out["published_canonical_tier2_mil"] = out[CANONICAL_TIER2_ROW]
    out["published_component_anchored_canonical_mil"] = out[COMPONENT_ANCHORED_CANONICAL_ROW]
    out["canonical_formula_difference_mil"] = (
        out["canonical_tier2_recomputed_mil"] - out["published_canonical_tier2_mil"]
    )
    out["component_anchored_difference_mil"] = (
        out["published_canonical_tier2_mil"] - out["published_component_anchored_canonical_mil"]
    )
    out["canonical_formula_status"] = out["canonical_formula_difference_mil"].map(
        lambda value: "matched" if abs(float(value)) <= 1e-6 else "mismatch"
    )
    out["component_anchored_status"] = out["component_anchored_difference_mil"].map(
        lambda value: "matched" if abs(float(value)) <= 1e-6 else "mismatch"
    )
    return out.drop(columns=[CANONICAL_TIER2_ROW, COMPONENT_ANCHORED_CANONICAL_ROW])


def build_selected_tdc_panel(
    sources: dict[str, pd.DataFrame],
    *,
    treasury_to_ru_transfer_share_of_deficit: float = 0.01,
) -> pd.DataFrame:
    """Build one selected TDC target row per supported replay quarter."""

    estimates = normalize_date_and_quarter(sources["tdc_estimates"], dataset_name="tdc_estimates")
    regression = normalize_date_and_quarter(
        sources["tdc_tier2_regression_series"],
        dataset_name="tdc_tier2_regression_series",
    )
    components = normalize_date_and_quarter(sources["tdc_components"], dataset_name="tdc_components")
    fiscal = normalize_date_and_quarter(
        sources["tdc_du_fiscal_flow_research"],
        dataset_name="tdc_du_fiscal_flow_research",
    )
    anchor = sources.get("tdc_empirical_anchor")
    if anchor is not None:
        anchor = normalize_date_and_quarter(anchor, dataset_name="tdc_empirical_anchor")

    _require_columns(estimates, [CANONICAL_TIER2_ROW], dataset_name="tdc_estimates")
    _require_columns(
        regression,
        [REGRESSION_DI_ROW, REGRESSION_BANK_ONLY_ROW, "tier2_regression_di_method_tier"],
        dataset_name="tdc_tier2_regression_series",
    )

    base = regression[["date", "quarter", REGRESSION_DI_ROW, REGRESSION_BANK_ONLY_ROW, "tier2_regression_di_method_tier"]].copy()
    base = base.merge(
        estimates[["date", CANONICAL_TIER2_ROW]].copy(),
        on="date",
        how="left",
        validate="one_to_one",
    )
    base = _merge_optional_components(base, components)
    base = _merge_optional_fiscal(base, fiscal)
    if anchor is not None:
        base = _merge_optional_anchor(base, anchor)

    rows = []
    for _, row in base.sort_values("date").iterrows():
        quarter = str(row["quarter"])
        period = pd.Period(quarter, freq="Q")
        if period >= pd.Period("2022Q1", freq="Q") and pd.notna(row.get(CANONICAL_TIER2_ROW)):
            selected_key = CANONICAL_TIER2_ROW
            selected_value = row[CANONICAL_TIER2_ROW]
            method_label = "canonical_tier2_component_anchored_di_mmf_rrp_prop"
            method_tier = "constrained_component"
            status = "source_backed_canonical_modern"
        elif period >= pd.Period("2013Q3", freq="Q"):
            selected_key = REGRESSION_DI_ROW
            selected_value = row[REGRESSION_DI_ROW]
            method_label = "tier2_regression_di_component_pool_wamest_bucket_backcast_mmf_rrp_prop"
            method_tier = str(row.get("tier2_regression_di_method_tier", "component_pool_wamest_bucket_backcast"))
            status = "source_backed_regression_companion"
        elif period >= pd.Period("2010Q2", freq="Q"):
            selected_key = REGRESSION_DI_ROW
            selected_value = row[REGRESSION_DI_ROW]
            method_label = "tier2_regression_di_component_pool_wamest_bucket_backcast_mmf_rrp_structural_zero"
            method_tier = str(row.get("tier2_regression_di_method_tier", "component_pool_wamest_bucket_backcast"))
            status = "source_backed_regression_companion"
        else:
            selected_key = REGRESSION_DI_ROW
            selected_value = row[REGRESSION_DI_ROW]
            method_label = "tier2_regression_di_pre_component_h15_scaled_backcast_mmf_rrp_structural_zero"
            method_tier = str(row.get("tier2_regression_di_method_tier", "pre_component_h15_scaled_backcast"))
            status = "source_backed_regression_companion"

        out = row.to_dict()
        out.update(
            {
                "selected_tdc_series_key": selected_key,
                "selected_tdc_value_mil": float(selected_value) if pd.notna(selected_value) else pd.NA,
                "replay_tdc_method_label": method_label,
                "replay_tdc_method_tier": method_tier,
                "tdc_measurement_status": status,
                "tdc_units": "millions_of_dollars",
            }
        )
        rows.append(out)

    panel = pd.DataFrame(rows)
    panel["selected_tdc_value_mil"] = pd.to_numeric(panel["selected_tdc_value_mil"], errors="coerce")
    panel["selected_tdc_value_bil"] = panel["selected_tdc_value_mil"] / 1000.0
    panel["alternative_bank_only_regression_mil"] = pd.to_numeric(panel[REGRESSION_BANK_ONLY_ROW], errors="coerce")
    panel["di_minus_bank_regression_wedge_mil"] = (
        pd.to_numeric(panel[REGRESSION_DI_ROW], errors="coerce") - panel["alternative_bank_only_regression_mil"]
    )
    panel = _add_transfer_bridge(
        panel,
        treasury_to_ru_transfer_share_of_deficit=treasury_to_ru_transfer_share_of_deficit,
    )
    panel = _add_decomposition(panel)
    panel = _add_tdc_level_index(panel)
    return panel


def write_tdc_validation_artifacts(
    output_dir: str | Path,
    *,
    panel: pd.DataFrame,
    formula_crosscheck: pd.DataFrame,
    source_manifest: pd.DataFrame,
) -> dict[str, str]:
    """Write first-pass TDC validation artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_manifest": out / "historical_replay_source_manifest.csv",
        "tdc_input_coverage": out / "historical_replay_tdc_input_coverage.csv",
        "tdc_components": out / "historical_replay_tdc_components.csv",
        "modern_formula_crosscheck": out / "tdcest_modern_formula_crosscheck.csv",
        "modern_formula_summary": out / "tdcest_modern_formula_summary.csv",
        "selected_ladder_coverage": out / "tdcest_selected_ladder_coverage.csv",
        "fiscal_transfer_bridge": out / "fiscal_transfer_bridge.csv",
        "fiscal_transfer_bridge_sensitivity": out / "fiscal_transfer_bridge_sensitivity.csv",
        "tdc_mechanism_identity": out / "tdc_mechanism_identity.csv",
        "mechanism_component_validation": out / "mechanism_component_validation.csv",
    }
    source_manifest.to_csv(paths["source_manifest"], index=False)
    _build_input_coverage(panel).to_csv(paths["tdc_input_coverage"], index=False)
    panel.to_csv(paths["tdc_components"], index=False)
    formula_crosscheck.to_csv(paths["modern_formula_crosscheck"], index=False)
    _modern_formula_summary(formula_crosscheck).to_csv(paths["modern_formula_summary"], index=False)
    _selected_ladder_coverage(panel).to_csv(paths["selected_ladder_coverage"], index=False)
    _fiscal_transfer_bridge(panel).to_csv(paths["fiscal_transfer_bridge"], index=False)
    _fiscal_transfer_bridge_sensitivity(panel).to_csv(paths["fiscal_transfer_bridge_sensitivity"], index=False)
    _tdc_mechanism_identity(panel).to_csv(paths["tdc_mechanism_identity"], index=False)
    build_mechanism_component_validation(panel).to_csv(
        paths["mechanism_component_validation"],
        index=False,
    )
    return {key: str(path) for key, path in paths.items()}


def apply_auction_absorption_to_tdc_panel(
    panel: pd.DataFrame,
    auction_reconciliation: pd.DataFrame,
) -> pd.DataFrame:
    """Wire source-backed auction allotment totals into the TDC decomposition."""

    if panel.empty or auction_reconciliation.empty:
        return panel.copy()
    required = [
        "quarter",
        "is_bridge_class",
        "included_in_identified_primary_allotment",
        "included_in_tdc_auction_absorption",
        "allotment_amount_mil",
        "signed_tdc_auction_absorption_mil",
        "unique_auction_accepted_amount_mil",
        "quarter_allotment_reconciliation_gap_mil",
    ]
    _require_columns(auction_reconciliation, required, dataset_name="auction_absorption_reconciliation")
    auction = auction_reconciliation.copy()
    auction["allotment_amount_mil"] = pd.to_numeric(auction["allotment_amount_mil"], errors="coerce").fillna(0.0)
    auction["signed_tdc_auction_absorption_mil"] = pd.to_numeric(
        auction["signed_tdc_auction_absorption_mil"],
        errors="coerce",
    ).fillna(0.0)
    auction["included_in_identified_primary_allotment"] = auction[
        "included_in_identified_primary_allotment"
    ].astype(bool)
    auction["included_in_tdc_auction_absorption"] = auction[
        "included_in_tdc_auction_absorption"
    ].astype(bool)
    auction["is_bridge_class"] = auction["is_bridge_class"].astype(bool)

    rows = []
    for quarter, group in auction.groupby("quarter", sort=False, dropna=False):
        identified = group.loc[group["included_in_identified_primary_allotment"], "allotment_amount_mil"].sum()
        tdc_signed = group["signed_tdc_auction_absorption_mil"].sum()
        du_gross = group.loc[group["included_in_tdc_auction_absorption"], "allotment_amount_mil"].sum()
        non_du_identified = group.loc[
            group["included_in_identified_primary_allotment"]
            & ~group["included_in_tdc_auction_absorption"],
            "allotment_amount_mil",
        ].sum()
        bridge_or_unresolved = group.loc[
            ~group["included_in_identified_primary_allotment"],
            "allotment_amount_mil",
        ].sum()
        accepted = pd.to_numeric(group["unique_auction_accepted_amount_mil"], errors="coerce").dropna()
        gap = pd.to_numeric(group["quarter_allotment_reconciliation_gap_mil"], errors="coerce").dropna()
        rows.append(
            {
                "quarter": str(quarter),
                "tdc_auction_absorption_mil": float(tdc_signed),
                "auction_identified_primary_allotment_mil": float(identified),
                "auction_identified_du_absorption_gross_mil": float(du_gross),
                "auction_non_du_identified_allotment_mil": float(non_du_identified),
                "auction_bridge_or_unresolved_mil": float(bridge_or_unresolved),
                "auction_accepted_amount_mil": float(accepted.iloc[0]) if not accepted.empty else pd.NA,
                "auction_allotment_reconciliation_gap_mil": float(gap.iloc[0]) if not gap.empty else pd.NA,
                "auction_absorption_measurement_status": (
                    "source_backed_private_class_primary_allotment_signed_negative;"
                    "dealer_and_bridge_amounts_explicit_not_final_holders"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    out = panel.copy().merge(summary, on="quarter", how="left", suffixes=("", "__auction"))
    replacement = pd.to_numeric(out.get("tdc_auction_absorption_mil__auction"), errors="coerce")
    if "tdc_auction_absorption_mil__auction" in out.columns:
        out["tdc_auction_absorption_mil"] = replacement.where(
            replacement.notna(),
            pd.to_numeric(out.get("tdc_auction_absorption_mil"), errors="coerce"),
        )
        out = out.drop(columns=["tdc_auction_absorption_mil__auction"])
    out["auction_absorption_measurement_status"] = out.get(
        "auction_absorption_measurement_status",
        pd.Series(index=out.index, dtype=object),
    ).fillna("not_available")
    return _recompute_decomposition_residual(out)


def build_mechanism_component_validation(panel: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form source/status table for TDC mechanism components."""

    columns = [
        "quarter",
        "component",
        "value_mil",
        "measurement_status",
        "source_role",
        "selected_tdc_value_mil",
        "component_share_of_selected_tdc",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    specs = [
        ("fiscal_flow", "tdc_fiscal_flow_mil", "fiscal_flow_measurement_status", "tdcest_fiscal_proxy"),
        ("debt_service", "tdc_debt_service_mil", "debt_service_measurement_status", "tdcest_allocated_coupon_proxy"),
        ("auction_absorption", "tdc_auction_absorption_mil", "auction_absorption_measurement_status", "auction_allotment_panel"),
        ("secondary_trades", "tdc_secondary_trades_mil", "secondary_trades_measurement_status", "unobserved"),
        ("other_named", "tdc_other_mil", "other_measurement_status", "toc_remittance_named_cash_effects"),
        ("residual", "tdc_residual_mil", "tdc_residual_status", "identity_residual"),
    ]
    rows = []
    for _, row in panel.iterrows():
        selected = pd.to_numeric(row.get("selected_tdc_value_mil"), errors="coerce")
        for component, value_column, status_column, source_role in specs:
            value = pd.to_numeric(row.get(value_column), errors="coerce")
            rows.append(
                {
                    "quarter": row.get("quarter"),
                    "component": component,
                    "value_mil": value if pd.notna(value) else pd.NA,
                    "measurement_status": row.get(status_column, "not_available"),
                    "source_role": source_role,
                    "selected_tdc_value_mil": selected if pd.notna(selected) else pd.NA,
                    "component_share_of_selected_tdc": (
                        float(value) / float(selected)
                        if pd.notna(value) and pd.notna(selected) and abs(float(selected)) > 1.0e-12
                        else pd.NA
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _merge_optional_components(base: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "fed_tsy_tx",
        "bank_depository_tsy_tx",
        "np_credit_unions_tsy_tx",
        "broad_depository_np_cu_tsy_tx",
        "row_tsy_tx",
        "minus_treasury_operating_cash_tx",
        "fed_remit_positive",
        "fed_tsy_coupon_interest_proxy",
        "bank_tier2_component_interest_proxy",
        "row_tier2_component_interest_proxy",
        "credit_union_tier2_component_interest_proxy",
        "mmf_rrp_adjustment_lb",
        "mmf_rrp_adjustment_prop",
        "mmf_rrp_adjustment_ub",
        "du_noninterest_outlay_proxy",
        "du_receipt_proxy",
        "du_coupon_proxy_selected_narrow",
    ]
    available = [col for col in cols if col in components.columns]
    return base.merge(components[available], on="date", how="left", validate="one_to_one")


def _tdc_mechanism_identity(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "quarter",
        "selected_tdc_series_key",
        "selected_tdc_value_mil",
        "tdc_fiscal_flow_mil",
        "tdc_debt_service_mil",
        "tdc_auction_absorption_mil",
        "auction_identified_primary_allotment_mil",
        "auction_identified_du_absorption_gross_mil",
        "auction_non_du_identified_allotment_mil",
        "auction_bridge_or_unresolved_mil",
        "auction_accepted_amount_mil",
        "auction_allotment_reconciliation_gap_mil",
        "tdc_secondary_trades_mil",
        "tdc_other_mil",
        "tdc_residual_mil",
        "recomputed_tdc_identity_mil",
        "identity_gap_mil",
        "secondary_trades_measurement_status",
        "auction_absorption_measurement_status",
        "tdc_residual_status",
        "identity_status",
    ]
    if panel.empty:
        return pd.DataFrame(columns=columns)
    out = panel.copy()
    components = [
        "tdc_fiscal_flow_mil",
        "tdc_debt_service_mil",
        "tdc_auction_absorption_mil",
        "tdc_secondary_trades_mil",
        "tdc_other_mil",
        "tdc_residual_mil",
    ]
    recomputed = pd.Series(0.0, index=out.index)
    for column in components:
        recomputed = recomputed + pd.to_numeric(out.get(column), errors="coerce").fillna(0.0)
    out["recomputed_tdc_identity_mil"] = recomputed
    out["identity_gap_mil"] = (
        pd.to_numeric(out["selected_tdc_value_mil"], errors="coerce")
        - out["recomputed_tdc_identity_mil"]
    )
    out["identity_status"] = out["identity_gap_mil"].map(
        lambda value: "identity_closed_with_explicit_residual"
        if pd.notna(value) and abs(float(value)) <= 1.0e-6
        else "identity_gap_requires_review"
    )
    return out[[column for column in columns if column in out.columns]]


def _merge_optional_fiscal(base: pd.DataFrame, fiscal: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "treasury_total_outlays_proxy",
        "treasury_total_receipts_proxy",
        "treasury_interest_gross_proxy",
    ]
    available = [col for col in cols if col in fiscal.columns]
    return base.merge(fiscal[available], on="date", how="left", validate="one_to_one")


def _merge_optional_anchor(base: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "opening_tdc_level": "anchor_opening_tdc_level_mil",
        "tdc_change": "anchor_tdc_change_mil",
        "closing_tdc_level": "anchor_closing_tdc_level_mil",
        "tdc_fiscal_flow": "anchor_tdc_fiscal_flow_mil",
        "tdc_debt_service": "anchor_tdc_debt_service_mil",
        "tdc_auction_absorption_primary_proxy": "anchor_tdc_auction_absorption_mil",
        "tdc_secondary_and_reconciliation_residual": "anchor_tdc_residual_mil",
        "tdc_other_named": "anchor_tdc_other_mil",
        "secondary_trades_measurement_status": "anchor_secondary_trades_measurement_status",
    }
    available = ["date", *[col for col in rename if col in anchor.columns]]
    return base.merge(anchor[available].rename(columns=rename), on="date", how="left", validate="one_to_one")


def _merge_interest_component_candidate(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "sector_group",
        "component_anchored_interest_mil",
        "component_anchored_interest_low_mil",
        "component_anchored_interest_high_mil",
    }
    if candidate.empty or not required.issubset(candidate.columns):
        return base
    work = candidate.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
    for column in [
        "component_anchored_interest_mil",
        "component_anchored_interest_low_mil",
        "component_anchored_interest_high_mil",
    ]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = (
        work.groupby(["date", "sector_group"], dropna=False, sort=False)[
            [
                "component_anchored_interest_mil",
                "component_anchored_interest_low_mil",
                "component_anchored_interest_high_mil",
            ]
        ]
        .sum(min_count=1)
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index="date",
        columns="sector_group",
        values=[
            "component_anchored_interest_mil",
            "component_anchored_interest_low_mil",
            "component_anchored_interest_high_mil",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{sector}_tier2_candidate_{measure.replace('component_anchored_', '')}" for measure, sector in pivot.columns]
    pivot = pivot.reset_index()
    rename = {
        "bank_tier2_candidate_interest_mil": "bank_tier2_candidate_interest_mil",
        "bank_tier2_candidate_interest_low_mil": "bank_tier2_candidate_interest_low_mil",
        "bank_tier2_candidate_interest_high_mil": "bank_tier2_candidate_interest_high_mil",
        "credit_union_tier2_candidate_interest_mil": "credit_union_tier2_candidate_interest_mil",
        "credit_union_tier2_candidate_interest_low_mil": "credit_union_tier2_candidate_interest_low_mil",
        "credit_union_tier2_candidate_interest_high_mil": "credit_union_tier2_candidate_interest_high_mil",
        "row_tier2_candidate_interest_mil": "row_tier2_candidate_interest_mil",
        "row_tier2_candidate_interest_low_mil": "row_tier2_candidate_interest_low_mil",
        "row_tier2_candidate_interest_high_mil": "row_tier2_candidate_interest_high_mil",
    }
    pivot = pivot.rename(columns=rename)
    merged = base.merge(pivot, on="date", how="left", validate="one_to_one")
    bank = pd.to_numeric(merged.get("bank_tier2_candidate_interest_mil"), errors="coerce")
    cu = pd.to_numeric(merged.get("credit_union_tier2_candidate_interest_mil"), errors="coerce")
    merged["banks_plus_credit_union_tier2_candidate_interest_mil"] = bank + cu
    for suffix in ["low_mil", "high_mil"]:
        bank_col = f"bank_tier2_candidate_interest_{suffix}"
        cu_col = f"credit_union_tier2_candidate_interest_{suffix}"
        merged[f"banks_plus_credit_union_tier2_candidate_interest_{suffix}"] = (
            pd.to_numeric(merged.get(bank_col), errors="coerce")
            + pd.to_numeric(merged.get(cu_col), errors="coerce")
        )
    merged["tier2_candidate_interest_reference_status"] = merged[
        "banks_plus_credit_union_tier2_candidate_interest_mil"
    ].map(lambda value: "present" if pd.notna(value) else "not_available")
    return merged


def _add_transfer_bridge(
    panel: pd.DataFrame,
    *,
    treasury_to_ru_transfer_share_of_deficit: float,
) -> pd.DataFrame:
    out = panel.copy()
    share = float(treasury_to_ru_transfer_share_of_deficit)
    out["deficit_mil"] = (
        pd.to_numeric(out.get("treasury_total_outlays_proxy"), errors="coerce")
        - pd.to_numeric(out.get("treasury_total_receipts_proxy"), errors="coerce")
    )
    out["treasury_to_ru_transfer_share_of_deficit"] = share
    out["assumed_treasury_to_ru_transfer_mil"] = share * out["deficit_mil"]
    out["tdc_unobserved_ru_transfer_contribution_mil"] = -out["assumed_treasury_to_ru_transfer_mil"]
    out["assumed_transfer_status"] = out["deficit_mil"].map(
        lambda value: f"assumption_{share:g}_share_of_deficit" if pd.notna(value) else "not_available"
    )
    return out


def _add_decomposition(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    computed_fiscal = (
        pd.to_numeric(out.get("du_noninterest_outlay_proxy"), errors="coerce")
        - pd.to_numeric(out.get("du_receipt_proxy"), errors="coerce")
    )
    computed_debt = pd.to_numeric(out.get("du_coupon_proxy_selected_narrow"), errors="coerce")
    computed_other = (
        pd.to_numeric(out.get("minus_treasury_operating_cash_tx"), errors="coerce")
        + pd.to_numeric(out.get("fed_remit_positive"), errors="coerce")
    )

    out["tdc_fiscal_flow_mil"] = _prefer_anchor(out, "anchor_tdc_fiscal_flow_mil", computed_fiscal)
    out["tdc_debt_service_mil"] = _prefer_anchor(out, "anchor_tdc_debt_service_mil", computed_debt)
    out["tdc_auction_absorption_mil"] = _prefer_anchor(out, "anchor_tdc_auction_absorption_mil", 0.0)
    out["tdc_other_mil"] = _prefer_anchor(out, "anchor_tdc_other_mil", computed_other)
    out["tdc_secondary_trades_mil"] = pd.NA
    computed_residual = (
        out["selected_tdc_value_mil"]
        - pd.to_numeric(out["tdc_fiscal_flow_mil"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["tdc_debt_service_mil"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["tdc_auction_absorption_mil"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["tdc_other_mil"], errors="coerce").fillna(0.0)
    )
    out["tdc_residual_mil"] = _prefer_anchor(out, "anchor_tdc_residual_mil", computed_residual)
    out["secondary_trades_measurement_status"] = out.get(
        "anchor_secondary_trades_measurement_status",
        pd.Series(index=out.index, dtype=object),
    ).fillna("unobserved_not_zero_evidence")
    out["tdc_residual_status"] = "identity_residual_not_observed_channel"
    return out


def _recompute_decomposition_residual(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for column in [
        "tdc_fiscal_flow_mil",
        "tdc_debt_service_mil",
        "tdc_auction_absorption_mil",
        "tdc_other_mil",
    ]:
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
    out["tdc_residual_mil"] = (
        pd.to_numeric(out["selected_tdc_value_mil"], errors="coerce")
        - out["tdc_fiscal_flow_mil"].fillna(0.0)
        - out["tdc_debt_service_mil"].fillna(0.0)
        - out["tdc_auction_absorption_mil"].fillna(0.0)
        - out["tdc_other_mil"].fillna(0.0)
    )
    out["tdc_residual_status"] = "identity_residual_not_observed_channel"
    return out


def _add_tdc_level_index(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values("date").copy()
    cumulative = out["selected_tdc_value_mil"].cumsum()
    seam_rows = out.loc[out["quarter"] == "2021Q4"]
    offset = float(cumulative.loc[seam_rows.index[0]]) if not seam_rows.empty else 0.0
    out["closing_tdc_level_mil"] = cumulative - offset
    out["opening_tdc_level_mil"] = out["closing_tdc_level_mil"].shift(1).fillna(0.0 - offset)
    has_anchor = out["anchor_closing_tdc_level_mil"].notna() if "anchor_closing_tdc_level_mil" in out.columns else False
    if isinstance(has_anchor, pd.Series):
        out.loc[has_anchor, "opening_tdc_level_mil"] = out.loc[has_anchor, "anchor_opening_tdc_level_mil"]
        out.loc[has_anchor, "closing_tdc_level_mil"] = out.loc[has_anchor, "anchor_closing_tdc_level_mil"]
    out["tdc_level_status"] = "cumulative_accounting_index_not_observed_stock"
    return out


def _prefer_anchor(frame: pd.DataFrame, anchor_column: str, fallback):
    if anchor_column not in frame.columns:
        return fallback
    anchor = pd.to_numeric(frame[anchor_column], errors="coerce")
    fallback_series = fallback if isinstance(fallback, pd.Series) else pd.Series(fallback, index=frame.index)
    return anchor.where(anchor.notna(), fallback_series)


def _build_input_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = [
        "selected_tdc_value_mil",
        "fed_tsy_tx",
        "bank_depository_tsy_tx",
        "np_credit_unions_tsy_tx",
        "row_tsy_tx",
        "minus_treasury_operating_cash_tx",
        "fed_remit_positive",
        "fed_tsy_coupon_interest_proxy",
        "bank_tier2_component_interest_proxy",
        "row_tier2_component_interest_proxy",
        "credit_union_tier2_component_interest_proxy",
        "mmf_rrp_adjustment_prop",
    ]
    for _, row in panel.iterrows():
        for field in fields:
            rows.append(
                {
                    "quarter": row["quarter"],
                    "date": row["date"],
                    "input_key": field,
                    "value": row.get(field, pd.NA),
                    "source_status": "present" if pd.notna(row.get(field, pd.NA)) else "missing",
                    "unit": "millions_of_dollars",
                }
            )
    return pd.DataFrame(rows)


def _modern_formula_summary(formula: pd.DataFrame) -> pd.DataFrame:
    if formula.empty:
        return pd.DataFrame(
            [
                {
                    "compared_rows": 0,
                    "missing_input_rows": 0,
                    "canonical_formula_mismatch_rows": 0,
                    "component_anchored_mismatch_rows": 0,
                    "max_abs_formula_difference_mil": pd.NA,
                }
            ]
        )
    return pd.DataFrame(
        [
            {
                "compared_rows": int(len(formula.index)),
                "missing_input_rows": int(formula[list(MODERN_CANONICAL_REQUIRED_COLUMNS)].isna().any(axis=1).sum()),
                "canonical_formula_mismatch_rows": int((formula["canonical_formula_status"] != "matched").sum()),
                "component_anchored_mismatch_rows": int((formula["component_anchored_status"] != "matched").sum()),
                "max_abs_formula_difference_mil": formula["canonical_formula_difference_mil"].abs().max(),
            }
        ]
    )


def _selected_ladder_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, label, tier), group in panel.groupby(
        ["selected_tdc_series_key", "replay_tdc_method_label", "replay_tdc_method_tier"],
        dropna=False,
        sort=False,
    ):
        rows.append(
            {
                "selected_tdc_series_key": key,
                "replay_tdc_method_label": label,
                "replay_tdc_method_tier": tier,
                "quarter_count": int(len(group.index)),
                "first_quarter": group["quarter"].min(),
                "last_quarter": group["quarter"].max(),
            }
        )
    return pd.DataFrame(rows)


def _fiscal_transfer_bridge(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "quarter",
        "treasury_total_outlays_proxy",
        "treasury_total_receipts_proxy",
        "deficit_mil",
        "treasury_to_ru_transfer_share_of_deficit",
        "assumed_treasury_to_ru_transfer_mil",
        "tdc_unobserved_ru_transfer_contribution_mil",
        "assumed_transfer_status",
    ]
    return panel[[col for col in cols if col in panel.columns]].copy()


def _fiscal_transfer_bridge_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "quarter",
                "transfer_share_of_deficit",
                "deficit_mil",
                "assumed_treasury_to_ru_transfer_mil",
                "tdc_unobserved_ru_transfer_contribution_mil",
                "scenario_status",
            ]
        )
    rows = []
    configured = pd.to_numeric(
        panel.get("treasury_to_ru_transfer_share_of_deficit", pd.Series(0.01, index=panel.index)),
        errors="coerce",
    ).dropna()
    shares = [0.0]
    if not configured.empty:
        configured_share = float(configured.iloc[0])
        if configured_share not in shares:
            shares.append(configured_share)
    for share in shares:
        for _, row in panel.iterrows():
            deficit = pd.to_numeric(row.get("deficit_mil"), errors="coerce")
            assumed = share * float(deficit) if pd.notna(deficit) else pd.NA
            rows.append(
                {
                    "date": row.get("date"),
                    "quarter": row.get("quarter"),
                    "transfer_share_of_deficit": share,
                    "deficit_mil": deficit,
                    "assumed_treasury_to_ru_transfer_mil": assumed,
                    "tdc_unobserved_ru_transfer_contribution_mil": -assumed if pd.notna(assumed) else pd.NA,
                    "scenario_status": f"assumption_{share:g}_share_of_deficit"
                    if pd.notna(deficit)
                    else "not_available",
                }
            )
    return pd.DataFrame(rows)


def _filter_quarters(frame: pd.DataFrame, *, start_quarter: str | None, end_quarter: str | None) -> pd.DataFrame:
    if frame.empty or (start_quarter is None and end_quarter is None):
        return frame.copy()
    periods = pd.PeriodIndex(frame["quarter"], freq="Q")
    mask = pd.Series(True, index=frame.index)
    if start_quarter is not None:
        mask &= periods >= pd.Period(start_quarter, freq="Q")
    if end_quarter is not None:
        mask &= periods <= pd.Period(end_quarter, freq="Q")
    return frame.loc[mask].reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, columns, *, dataset_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


__all__ = [
    "CANONICAL_TIER2_ROW",
    "COMPONENT_ANCHORED_CANONICAL_ROW",
    "REGRESSION_BANK_ONLY_ROW",
    "REGRESSION_DI_ROW",
    "TDC_PANEL_NUMERIC_COLUMNS",
    "apply_auction_absorption_to_tdc_panel",
    "build_mechanism_component_validation",
    "build_historical_replay_tdc_panel",
    "build_modern_canonical_formula_crosscheck",
    "build_selected_tdc_panel",
    "write_tdc_validation_artifacts",
]
