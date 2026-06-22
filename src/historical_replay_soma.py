"""SOMA observed Treasury holdings for historical replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOMA_FIXED_SECTOR = "monetary_authority"


def load_soma_treasury_holdings(
    path: str | Path,
    *,
    start_quarter: str,
    end_quarter: str,
) -> pd.DataFrame:
    """Load NY Fed SOMA Treasury holdings and select one observation per quarter."""

    path_obj = Path(path)
    if not path_obj.exists():
        return _empty_soma_frame()
    raw = pd.read_csv(path_obj, low_memory=False)
    if raw.empty:
        return _empty_soma_frame()
    by_normalized = {_normalize_column(col): col for col in raw.columns}

    def col(name: str) -> str | None:
        return by_normalized.get(_normalize_column(name))

    asof_col = col("As Of Date")
    cusip_col = col("CUSIP")
    maturity_col = col("Maturity Date")
    security_col = col("Security Type")
    current_face_col = col("Current Face Value")
    par_col = col("Par Value")
    inflation_col = col("Inflation Compensation")
    if asof_col is None or cusip_col is None or maturity_col is None:
        return _empty_soma_frame()
    working = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(raw[asof_col], errors="coerce"),
            "cusip": raw[cusip_col].map(_normalize_cusip),
            "maturity_date": pd.to_datetime(raw[maturity_col], errors="coerce").dt.date.astype(str),
            "security_type": raw[security_col].astype(str) if security_col else "",
            "current_face_value": _numeric(raw[current_face_col]) if current_face_col else np.nan,
            "par_value": _numeric(raw[par_col]) if par_col else np.nan,
            "inflation_compensation": _numeric(raw[inflation_col]) if inflation_col else np.nan,
        }
    )
    working = working.dropna(subset=["as_of_date"])
    working = working.loc[working["cusip"].astype(str).str.len().gt(0)]
    if working.empty:
        return _empty_soma_frame()
    working["quarter"] = working["as_of_date"].dt.to_period("Q").astype(str)
    start = pd.Period(str(start_quarter), freq="Q")
    end = pd.Period(str(end_quarter), freq="Q")
    working = working.loc[
        working["as_of_date"].dt.to_period("Q").between(start, end)
    ].copy()
    if working.empty:
        return _empty_soma_frame()
    par_plus_inflation = pd.to_numeric(working["par_value"], errors="coerce").fillna(0.0) + pd.to_numeric(
        working["inflation_compensation"],
        errors="coerce",
    ).fillna(0.0)
    working["observed_face_value_mil"] = working["current_face_value"].where(
        working["current_face_value"].notna() & working["current_face_value"].gt(0.0),
        par_plus_inflation,
    )
    working["observed_face_value_mil"] = pd.to_numeric(
        working["observed_face_value_mil"],
        errors="coerce",
    ).fillna(0.0) / 1_000_000.0
    working["observed_par_value_mil"] = pd.to_numeric(
        working["par_value"],
        errors="coerce",
    ).fillna(0.0) / 1_000_000.0
    working["inflation_compensation_mil"] = pd.to_numeric(
        working["inflation_compensation"],
        errors="coerce",
    ).fillna(0.0) / 1_000_000.0
    working = working.loc[working["observed_face_value_mil"].gt(0.0)].copy()
    if working.empty:
        return _empty_soma_frame()

    latest_asof = working.groupby("quarter", sort=True)["as_of_date"].transform("max")
    selected = working.loc[working["as_of_date"].eq(latest_asof)].copy()
    return selected[
        [
            "quarter",
            "as_of_date",
            "cusip",
            "maturity_date",
            "security_type",
            "observed_face_value_mil",
            "observed_par_value_mil",
            "inflation_compensation_mil",
        ]
    ].reset_index(drop=True)


def build_soma_fixed_allocations(
    cohorts: pd.DataFrame,
    soma_holdings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map SOMA CUSIP holdings onto MSPD cohorts as fixed Fed allocations."""

    allocation_columns = [
        "quarter",
        "sector",
        "cohort_id",
        "allocated_outstanding",
        "fixed_allocation_source",
        "fixed_allocation_policy",
    ]
    diagnostic_columns = [
        "quarter",
        "soma_as_of_date",
        "soma_rows",
        "matched_soma_rows",
        "unmatched_soma_rows",
        "soma_face_mil",
        "matched_face_mil",
        "unmatched_face_mil",
        "maturity_timing_rolloff_face_mil",
        "residual_unmatched_face_mil",
        "capped_excess_face_mil",
        "cohort_supply_mil",
        "status",
    ]
    if cohorts.empty or soma_holdings.empty:
        return pd.DataFrame(columns=allocation_columns), pd.DataFrame(columns=diagnostic_columns)

    required = {"quarter", "cohort_id", "cusip", "maturity_date", "outstanding"}
    if not required.issubset(cohorts.columns):
        return pd.DataFrame(columns=allocation_columns), pd.DataFrame(columns=diagnostic_columns)

    cohort = cohorts.copy()
    cohort["quarter"] = cohort["quarter"].astype(str)
    cohort["cusip"] = cohort["cusip"].map(_normalize_cusip)
    cohort["maturity_date"] = pd.to_datetime(cohort["maturity_date"], errors="coerce").dt.date.astype(str)
    cohort["outstanding"] = pd.to_numeric(cohort["outstanding"], errors="coerce").fillna(0.0)
    cohort = cohort.loc[cohort["outstanding"].gt(0.0)].copy()

    soma = soma_holdings.copy()
    soma["quarter"] = soma["quarter"].astype(str)
    soma["cusip"] = soma["cusip"].map(_normalize_cusip)
    soma["maturity_date"] = pd.to_datetime(soma["maturity_date"], errors="coerce").dt.date.astype(str)
    soma["observed_face_value_mil"] = pd.to_numeric(
        soma["observed_face_value_mil"],
        errors="coerce",
    ).fillna(0.0)
    soma = soma.loc[soma["observed_face_value_mil"].gt(0.0)].copy()
    if cohort.empty or soma.empty:
        return pd.DataFrame(columns=allocation_columns), pd.DataFrame(columns=diagnostic_columns)

    supply = (
        cohort.groupby(["quarter", "cusip", "maturity_date"], sort=False)["outstanding"]
        .sum()
        .rename("cohort_key_supply_mil")
        .reset_index()
    )
    soma_key = (
        soma.groupby(["quarter", "cusip", "maturity_date"], sort=False)
        .agg(
            observed_face_value_mil=("observed_face_value_mil", "sum"),
            observed_par_value_mil=("observed_par_value_mil", "sum"),
            soma_rows=("observed_face_value_mil", "size"),
            soma_as_of_date=("as_of_date", "max"),
        )
        .reset_index()
    )
    matched = soma_key.merge(
        supply,
        on=["quarter", "cusip", "maturity_date"],
        how="left",
        validate="one_to_one",
    )
    matched["cohort_key_supply_mil"] = pd.to_numeric(
        matched["cohort_key_supply_mil"],
        errors="coerce",
    ).fillna(0.0)
    matched["matched_face_mil"] = np.minimum(
        matched["observed_face_value_mil"],
        matched["cohort_key_supply_mil"],
    )
    matched["unmatched_face_mil"] = (matched["observed_face_value_mil"] - matched["matched_face_mil"]).clip(lower=0.0)
    matched["_quarter_end"] = matched["quarter"].map(lambda value: pd.Period(str(value), freq="Q").end_time.date())
    matched["_maturity_date_obj"] = pd.to_datetime(matched["maturity_date"], errors="coerce").dt.date
    matched["_as_of_date_obj"] = pd.to_datetime(matched["soma_as_of_date"], errors="coerce").dt.date
    timing_rolloff = (
        matched["cohort_key_supply_mil"].le(0.0)
        & matched["_maturity_date_obj"].notna()
        & matched["_as_of_date_obj"].notna()
        & (matched["_as_of_date_obj"] < matched["_maturity_date_obj"])
        & (matched["_maturity_date_obj"] <= matched["_quarter_end"])
    )
    matched["maturity_timing_rolloff_face_mil"] = matched["unmatched_face_mil"].where(timing_rolloff, 0.0)
    matched["residual_unmatched_face_mil"] = (
        matched["unmatched_face_mil"] - matched["maturity_timing_rolloff_face_mil"]
    ).clip(lower=0.0)
    matched["capped_excess_face_mil"] = matched["residual_unmatched_face_mil"]

    cohort_alloc = cohort.merge(
        matched[
            [
                "quarter",
                "cusip",
                "maturity_date",
                "matched_face_mil",
                "cohort_key_supply_mil",
            ]
        ],
        on=["quarter", "cusip", "maturity_date"],
        how="inner",
        validate="many_to_one",
    )
    cohort_alloc = cohort_alloc.loc[cohort_alloc["matched_face_mil"].gt(0.0)].copy()
    if not cohort_alloc.empty:
        cohort_alloc["allocated_outstanding"] = (
            cohort_alloc["matched_face_mil"]
            * cohort_alloc["outstanding"]
            / cohort_alloc["cohort_key_supply_mil"].replace(0.0, np.nan)
        ).fillna(0.0)
        allocations = pd.DataFrame(
            {
                "quarter": cohort_alloc["quarter"],
                "sector": SOMA_FIXED_SECTOR,
                "cohort_id": cohort_alloc["cohort_id"],
                "allocated_outstanding": cohort_alloc["allocated_outstanding"],
                "fixed_allocation_source": "nyfed_soma_treasury_holdings_monthly",
                "fixed_allocation_policy": "observed_soma_exact_block",
            }
        )
        allocations = allocations.loc[allocations["allocated_outstanding"].gt(0.0)].reset_index(drop=True)
    else:
        allocations = pd.DataFrame(columns=allocation_columns)

    diag_rows = []
    for quarter, group in matched.groupby("quarter", sort=True, dropna=False):
        soma_face = float(group["observed_face_value_mil"].sum())
        matched_face = float(group["matched_face_mil"].sum())
        timing_rolloff_face = float(group["maturity_timing_rolloff_face_mil"].sum())
        residual_unmatched_face = float(group["residual_unmatched_face_mil"].sum())
        status = (
            "matched"
            if abs(soma_face - matched_face) <= 1.0e-6
            else "matched_with_quarter_end_maturity_rolloff"
            if residual_unmatched_face <= 1.0e-6 and timing_rolloff_face > 0.0
            else "partial_match"
        )
        diag_rows.append(
            {
                "quarter": str(quarter),
                "soma_as_of_date": str(group["soma_as_of_date"].dropna().max()) if group["soma_as_of_date"].notna().any() else "",
                "soma_rows": int(group["soma_rows"].sum()),
                "matched_soma_rows": int(group["cohort_key_supply_mil"].gt(0.0).sum()),
                "unmatched_soma_rows": int(group["cohort_key_supply_mil"].le(0.0).sum()),
                "soma_face_mil": soma_face,
                "matched_face_mil": matched_face,
                "unmatched_face_mil": max(soma_face - matched_face, 0.0),
                "maturity_timing_rolloff_face_mil": timing_rolloff_face,
                "residual_unmatched_face_mil": residual_unmatched_face,
                "capped_excess_face_mil": float(group["capped_excess_face_mil"].sum()),
                "cohort_supply_mil": float(group["cohort_key_supply_mil"].sum()),
                "status": status,
            }
        )
    diagnostics = pd.DataFrame(diag_rows, columns=diagnostic_columns)
    return allocations.reindex(columns=allocation_columns), diagnostics


def _empty_soma_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "quarter",
            "as_of_date",
            "cusip",
            "maturity_date",
            "security_type",
            "observed_face_value_mil",
            "observed_par_value_mil",
            "inflation_compensation_mil",
        ]
    )


def _normalize_column(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _normalize_cusip(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip("'").strip('"').upper()
    return "".join(ch for ch in text if ch.isalnum())


def _numeric(series: pd.Series | object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
    return pd.Series(dtype=float)


__all__ = [
    "SOMA_FIXED_SECTOR",
    "build_soma_fixed_allocations",
    "load_soma_treasury_holdings",
]
