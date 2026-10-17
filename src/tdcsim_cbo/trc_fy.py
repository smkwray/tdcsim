"""Fiscal-year projection of a run's opt-in TRC table onto its verified annual clock."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from settlement_routes import TRC_FAMILIES, TRC_FAMILY_SLUGS, TRC_ROUTE_STATES

from .open04_campaign import OPEN04_AGGREGATION_CLOCK_ID

CLOSURE_TOLERANCE_BIL = 1.0e-6


class TrcFiscalYearError(RuntimeError):
    """A run's TRC table cannot be projected onto its annual clock."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_output_csv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return pd.read_csv(handle)
    return pd.read_csv(path)


def trc_fiscal_year_rows(outputs: Path, *, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check TRC closure against the legacy alias, then sum TRC within each fiscal year."""

    trc = read_output_csv(outputs / "tdcsim_trc_results.csv.gz")
    results = read_output_csv(outputs / "results_compact.csv.gz")
    annual = read_output_csv(outputs / "tdcsim_annual_economic_summary.csv.gz")
    metadata = json.loads((outputs / "tdcsim_trc_metadata.json").read_text(encoding="utf-8"))
    if set(annual["aggregation_clock_id"]) != {OPEN04_AGGREGATION_CLOCK_ID}:
        raise TrcFiscalYearError(f"{label}: annual summary uses an unexpected clock")
    merged = trc.merge(results[["Date", "Reserves"]], on="Date", how="left", validate="one_to_one")
    if merged["Reserves"].isna().any() or len(merged) != len(results):
        raise TrcFiscalYearError(f"{label}: TRC and legacy result dates differ")
    opening = merged.iloc[0]
    route_change = sum(merged[state] - float(opening[state]) for state in TRC_ROUTE_STATES)
    alias_gap = (merged["Reserves"] - merged["SyntheticReserveBalance_ExNonTreasuryFactors"] - route_change).abs()
    components = merged[[f"TRC_{family}" for family in TRC_FAMILIES]].sum(axis=1)
    component_gap = (components - merged["TRC_Change"]).abs()
    worst = {"legacy_alias_gap_bil": float(alias_gap.max()), "component_sum_gap_bil": float(component_gap.max())}
    if max(worst.values()) > CLOSURE_TOLERANCE_BIL:
        raise TrcFiscalYearError(f"{label}: TRC closure fails: {worst}")
    for state in TRC_ROUTE_STATES:
        if float(merged[state].min()) < -1.0e-9:
            raise TrcFiscalYearError(f"{label}: {state} becomes negative")

    dates = pd.to_datetime(merged["Date"])
    rows = []
    for period in annual.itertuples(index=False):
        start, end = pd.Timestamp(period.period_start), pd.Timestamp(period.period_end)
        window = merged[(dates > start) & (dates <= end)]
        at_end = merged[dates == end]
        if window.empty or at_end.empty:
            raise TrcFiscalYearError(f"{label}: no TRC rows for {period.period_label}")
        end_row = at_end.iloc[0]
        row = {
            "period_label": period.period_label,
            "period_status": "partial_opening" if bool(period.is_partial_period) else "full_fiscal_year",
            "coverage_start_exclusive": str(start.date()),
            "coverage_end_inclusive": str(end.date()),
            "cumulative_origin_date": str(pd.Timestamp(opening["Date"]).date()),
            "trc_change_bil": float(window["TRC_Change"].sum()),
        }
        for family in TRC_FAMILIES:
            row[f"trc_{TRC_FAMILY_SLUGS[family]}_bil"] = float(window[f"TRC_{family}"].sum())
        row.update(
            {
                "route_state_change_bil": float(window["TRC_RouteStateChange"].sum()),
                "legacy_holder_routed_reserve_change_bil": float(window["LegacyReserveChange_HolderRouted"].sum()),
                "cumulative_trc_bil": float(end_row["TRC_Cumulative"]),
                "onrrp_mmf_end_bil": float(end_row["ONRRP_MMF"]),
                "foreign_official_fed_deposits_end_bil": float(end_row["ForeignOfficialFedDeposits"]),
                "foreign_official_fed_rrp_end_bil": float(end_row["ForeignOfficialFedRRP"]),
                "synthetic_reserve_balance_ex_nontreasury_end_bil": float(
                    end_row["SyntheticReserveBalance_ExNonTreasuryFactors"]
                ),
                "legacy_reserves_alias_end_bil": float(end_row["Reserves"]),
                "units": "billions_of_nominal_dollars",
                "cumulative_basis": "since_simulation_origin",
                "perimeter": metadata["perimeter"],
                "foreign_gross_settlement_cash_bil": float(window["TRC_ForeignGrossSettlementCash"].sum()),
            }
        )
        rows.append(row)
    evidence = {
        "trc_results_sha256": sha256_path(outputs / "tdcsim_trc_results.csv.gz"),
        "trc_metadata_sha256": sha256_path(outputs / "tdcsim_trc_metadata.json"),
        "routing_source": metadata.get("routing_source"),
        "routing": metadata["routing"],
        "max_closure_gaps": worst,
        "trc_rows": int(len(trc)),
    }
    return rows, evidence
