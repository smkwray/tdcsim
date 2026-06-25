#!/usr/bin/env python3
"""Build long-history sector interest allocations from replay validation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from historical_replay_interest import (  # noqa: E402
    build_sector_interest_allocation,
    build_sector_interest_totals,
)


DEFAULT_VALIDATION_DIR = ROOT / "data/historical_replay/validation"


PORTFOLIO_USECOLS = [
    "quarter",
    "SecurityType",
    "IssueDate",
    "MaturityDate",
    "MaturityCategory",
    "FaceValue",
    "AdjustedPrincipal",
    "IndexRatio",
    "cusip",
    "HolderType",
    "HolderSubBucket",
    "broad_holder_class",
    "tdcsim_holder",
    "tdcsim_holder_subbucket",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default=str(DEFAULT_VALIDATION_DIR))
    parser.add_argument("--interest-detail", default="interest_component_detail.csv")
    parser.add_argument("--portfolio-snapshots", default="historical_replay_portfolio_snapshots.csv")
    parser.add_argument("--component-certification", default="historical_replay_interest_component_certification.csv")
    parser.add_argument("--bill-interest-flow", default="historical_replay_interest_flow_detail.csv")
    parser.add_argument("--frn-interest-flow", default="historical_replay_frn_interest_daily_detail.csv")
    parser.add_argument("--allocation-out", default="historical_replay_sector_interest_allocation.csv")
    parser.add_argument("--totals-out", default="historical_replay_sector_interest_totals.csv")
    parser.add_argument("--summary-out", default="historical_replay_sector_interest_allocation_summary.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_dir = Path(args.validation_dir)
    interest_detail = pd.read_csv(validation_dir / args.interest_detail, low_memory=False)
    portfolio = pd.read_csv(
        validation_dir / args.portfolio_snapshots,
        usecols=PORTFOLIO_USECOLS,
        low_memory=False,
    )
    component_certification = pd.read_csv(validation_dir / args.component_certification, low_memory=False)
    bill_interest_flow = pd.read_csv(validation_dir / args.bill_interest_flow, low_memory=False)
    frn_interest_flow = pd.read_csv(validation_dir / args.frn_interest_flow, low_memory=False)

    allocation = build_sector_interest_allocation(
        interest_detail,
        portfolio,
        component_certification,
        bill_interest_flow_detail=bill_interest_flow,
        frn_interest_flow_detail=frn_interest_flow,
    )
    totals = build_sector_interest_totals(allocation)

    allocation_path = validation_dir / args.allocation_out
    totals_path = validation_dir / args.totals_out
    summary_path = validation_dir / args.summary_out
    allocation.to_csv(allocation_path, index=False)
    totals.to_csv(totals_path, index=False)
    summary_path.write_text(_render_summary(allocation, totals), encoding="utf-8")

    print(f"wrote {allocation_path} ({len(allocation):,} rows)")
    print(f"wrote {totals_path} ({len(totals):,} rows)")
    print(f"wrote {summary_path}")
    return 0


def _render_summary(allocation: pd.DataFrame, totals: pd.DataFrame) -> str:
    if allocation.empty:
        return "# Historical Replay Sector Interest Allocation\n\nNo allocation rows were produced.\n"

    official_check = _component_conservation(allocation, "official")
    model_check = _component_conservation(allocation, "model")
    latest_quarter = str(allocation["quarter"].max())
    latest_totals = totals.loc[totals["quarter"].astype(str).eq(latest_quarter)].copy()
    latest_totals = latest_totals.sort_values(["scope_id", "selected_allocated_interest_mil"], ascending=[True, False])

    lines = [
        "# Historical Replay Sector Interest Allocation",
        "",
        "This artifact allocates each quarter/component aggregate interest control across TDCSIM holders using replay-derived component weights. Sector rows are joint allocations: they sum back to one aggregate control per quarter/component.",
        "",
        "## Coverage",
        "",
        f"- quarters: {allocation['quarter'].min()} to {allocation['quarter'].max()} ({allocation['quarter'].nunique()} quarters)",
        f"- rows: {len(allocation):,}",
        f"- sectors: {', '.join(sorted(allocation['tdc_sector'].dropna().astype(str).unique()))}",
        f"- detailed holders: {', '.join(sorted(allocation['tdcsim_holder'].dropna().astype(str).unique()))}",
        f"- components: {', '.join(sorted(allocation['component_id'].dropna().astype(str).unique()))}",
        f"- total scopes: {', '.join(sorted(totals['scope_id'].dropna().astype(str).unique()))}",
        "",
        "## Allocation Status Counts",
        "",
        _markdown_table(allocation["allocation_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="rows")),
        "",
        "## Conservation Checks",
        "",
        f"- official-control component max absolute residual: {_max_abs_residual(official_check):.12g} million",
        f"- model/proxy-control component max absolute residual: {_max_abs_residual(model_check):.12g} million",
        "",
        f"## Latest Quarter Sector Totals By Scope ({latest_quarter})",
        "",
        _markdown_table(
            latest_totals[
                [
                    "tdc_sector",
                    "scope_id",
                    "selected_allocated_interest_mil",
                    "allocated_official_interest_mil",
                    "allocated_model_interest_mil",
                    "component_count",
                    "tdcsim_holder_count",
                    "min_attributed_weight_coverage_pct",
                ]
            ]
        ),
        "",
    ]
    return "\n".join(lines)


def _component_conservation(allocation: pd.DataFrame, kind: str) -> pd.DataFrame:
    control_col = f"{kind}_interest_mil"
    alloc_col = f"allocated_{kind}_interest_mil"
    frame = allocation.loc[pd.to_numeric(allocation[control_col], errors="coerce").notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=["quarter", "component_id", "control_mil", "allocated_mil", "residual_mil"])
    grouped = (
        frame.groupby(["quarter", "component_id"], dropna=False, sort=True)
        .agg(
            control_mil=(control_col, "first"),
            allocated_mil=(alloc_col, "sum"),
        )
        .reset_index()
    )
    grouped["residual_mil"] = grouped["allocated_mil"] - grouped["control_mil"]
    return grouped


def _max_abs_residual(check: pd.DataFrame) -> float:
    if check.empty:
        return 0.0
    residual = pd.to_numeric(check["residual_mil"], errors="coerce").abs()
    return float(residual.max()) if residual.notna().any() else 0.0


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in frame.columns:
            value = row[column]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
