"""Coarse sector-by-cohort allocation for historical replay."""

from __future__ import annotations

import math
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd
try:
    from scipy.optimize import least_squares, linprog, minimize
except ImportError:  # pragma: no cover - exercised only in stripped environments.
    least_squares = None
    linprog = None
    minimize = None


_COHORT_ID_COL = "cohort_id"
_SECTOR_COL = "sector"
_QUARTER_COL = "quarter"
_ALLOCATION_COL = "allocated_outstanding"

_COHORT_VALUE_CANDIDATES = (
    "outstanding",
    "cohort_outstanding",
    "outstanding_amount",
    "amount",
    "face_value",
    "face_value_bil",
    "adjusted_principal",
    "adjusted_principal_bil",
)
_SECTOR_VALUE_CANDIDATES = (
    "sector_outstanding",
    "sector_stock",
    "stock",
    "outstanding",
    "amount",
    "value",
    "holding",
)
_GENERIC_WEIGHT_CANDIDATES = (
    "soft_target_weight",
    "prior_weight",
    "weight",
)
_WEIGHTED_BREGMAN_TIME_BUDGET_SECONDS = 120.0
_EXPORT_ALLOCATION_TOLERANCE = 1.0e-8


def _normalize_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _pick_column(frame: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    by_normalized = {_normalize_label(col): col for col in frame.columns}
    for candidate in candidates:
        direct = by_normalized.get(_normalize_label(candidate))
        if direct is not None:
            return direct
    raise ValueError(f"Could not find a {label} column in {list(frame.columns)}")


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def _select_quarter(frame: pd.DataFrame, quarter: object | None) -> tuple[pd.DataFrame, object | None]:
    if _QUARTER_COL not in frame.columns:
        return frame.copy(), quarter
    if quarter is not None:
        mask = frame[_QUARTER_COL].astype(str) == str(quarter)
        return frame.loc[mask].copy(), quarter
    unique = frame[_QUARTER_COL].dropna().astype(str).unique().tolist()
    if len(unique) <= 1:
        resolved = unique[0] if unique else quarter
        return frame.copy(), resolved
    return frame.copy(), quarter


def _prepare_cohorts(cohorts: pd.DataFrame, quarter: object | None) -> tuple[pd.DataFrame, object | None, str]:
    if _COHORT_ID_COL not in cohorts.columns:
        raise ValueError(f"`cohorts` must include `{_COHORT_ID_COL}`")
    filtered, resolved_quarter = _select_quarter(cohorts, quarter)
    value_col = _pick_column(filtered, _COHORT_VALUE_CANDIDATES, "cohort outstanding")
    working = filtered.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    if (working[value_col] < -1e-12).any():
        raise ValueError("Cohort outstanding inputs must be nonnegative")
    agg_map = {
        col: ("sum" if col == value_col else _first_non_null)
        for col in working.columns
        if col != _COHORT_ID_COL
    }
    prepared = (
        working.groupby(_COHORT_ID_COL, sort=False, dropna=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={value_col: "cohort_outstanding"})
    )
    return prepared, resolved_quarter, value_col


def _prepare_sector_stocks(sector_stocks: pd.DataFrame, quarter: object | None) -> tuple[pd.DataFrame, object | None, str]:
    if _SECTOR_COL not in sector_stocks.columns:
        raise ValueError(f"`sector_stocks` must include `{_SECTOR_COL}`")
    filtered, resolved_quarter = _select_quarter(sector_stocks, quarter)
    value_col = _pick_column(filtered, _SECTOR_VALUE_CANDIDATES, "sector stock")
    working = filtered.copy()
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    if (working[value_col] < -1e-12).any():
        raise ValueError("Sector stock inputs must be nonnegative")
    metadata_columns = [
        column
        for column in [
            "native_sector",
            "broad_holder_class",
            "tdcsim_holder",
            "tdcsim_holder_subbucket",
            "raw_z1_level",
            "sector_target_before_scale",
            "valuation_basis",
            "source_status",
            "evidence_label",
            "sector_adjustment_status",
        ]
        if column in working.columns
    ]
    agg_map = {value_col: "sum"}
    for column in metadata_columns:
        agg_map[column] = _first_non_null
    prepared = (
        working.groupby(_SECTOR_COL, sort=False, dropna=False)
        .agg(agg_map)
        .reset_index()
        .rename(columns={value_col: "sector_stock"})
    )
    return prepared, resolved_quarter, value_col


def _resolve_output_quarter(
    requested_quarter: object | None,
    cohort_quarter: object | None,
    sector_quarter: object | None,
) -> object | None:
    if requested_quarter is not None:
        return requested_quarter
    if cohort_quarter is not None:
        return cohort_quarter
    if sector_quarter is not None:
        return sector_quarter
    return None


def _extract_seed_matrix(
    cohorts: pd.DataFrame,
    sectors: pd.DataFrame,
    *,
    tolerance: float,
) -> np.ndarray:
    generic_weight_col = None
    for candidate in _GENERIC_WEIGHT_CANDIDATES:
        if candidate in cohorts.columns:
            generic_weight_col = candidate
            break
    base_weights = None
    if generic_weight_col is not None:
        base_weights = pd.to_numeric(cohorts[generic_weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        base_weights = np.ones(len(cohorts), dtype=float)
    base_weights = np.clip(base_weights, a_min=tolerance, a_max=None)

    matrix = np.tile(base_weights, (len(sectors.index), 1))
    normalized_columns = {_normalize_label(col): col for col in cohorts.columns}
    for row_idx, (_, sector_row) in enumerate(sectors.iterrows()):
        specific_col = None
        for key in _sector_prior_keys(sector_row):
            specific_col = normalized_columns.get(f"softtarget{_normalize_label(key)}")
            if specific_col is None:
                specific_col = normalized_columns.get(f"prior{_normalize_label(key)}")
            if specific_col is None:
                specific_col = normalized_columns.get(f"priorholder{_normalize_label(key)}")
            if specific_col is not None:
                break
        if specific_col is None:
            continue
        specific = pd.to_numeric(cohorts[specific_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        specific = np.clip(specific, a_min=tolerance, a_max=None)
        specific = _temper_prior_vector(specific, tolerance=tolerance)
        matrix[row_idx, :] = specific
    return matrix


def _temper_prior_vector(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    """Keep soft prior columns informative without letting them become hard corners."""

    out = np.asarray(values, dtype=float).copy()
    positive = out[out > tolerance]
    if positive.size == 0:
        return np.clip(out, a_min=tolerance, a_max=None)
    median = float(np.median(positive))
    if median <= tolerance:
        median = float(np.mean(positive))
    if median <= tolerance:
        return np.clip(out, a_min=tolerance, a_max=None)
    cap = median * 25.0
    floor = median / 25.0
    out = np.clip(out, a_min=floor, a_max=cap)
    return np.clip(out, a_min=tolerance, a_max=None)


def _extract_eligibility_matrix(cohorts: pd.DataFrame, sectors: pd.DataFrame) -> np.ndarray:
    """Return structural eligibility mask for sector/cohort cells."""

    matrix = np.ones((len(sectors.index), len(cohorts.index)), dtype=bool)
    if matrix.size == 0:
        return matrix
    normalized_columns = {_normalize_label(col): col for col in cohorts.columns}
    for row_idx, (_, sector_row) in enumerate(sectors.iterrows()):
        eligibility_col = None
        for key in _sector_prior_keys(sector_row):
            for prefix in ("eligible", "eligibility"):
                eligibility_col = normalized_columns.get(f"{prefix}{_normalize_label(key)}")
                if eligibility_col is not None:
                    break
            if eligibility_col is not None:
                break
        if eligibility_col is None:
            continue
        values = cohorts[eligibility_col]
        if values.dtype == bool:
            eligible = values.fillna(False).to_numpy(dtype=bool)
        else:
            text = values.where(values.notna(), "").astype(str).str.strip().str.lower()
            eligible = text.isin({"1", "true", "t", "yes", "y"}).to_numpy(dtype=bool)
            numeric = pd.to_numeric(values, errors="coerce")
            eligible |= numeric.fillna(0.0).gt(0.0).to_numpy(dtype=bool)
        matrix[row_idx, :] = eligible
    return matrix


def _sector_prior_keys(sector_row: pd.Series) -> list[object]:
    keys = []
    for column in ["sector", "native_sector", "broad_holder_class", "tdcsim_holder"]:
        if column not in sector_row.index:
            continue
        value = sector_row.get(column)
        if pd.notna(value) and str(value).strip():
            keys.append(value)
    return list(dict.fromkeys(keys))


def _apply_continuity_prior(
    seed: np.ndarray,
    cohorts: pd.DataFrame,
    sectors: pd.DataFrame,
    prior_allocations: pd.DataFrame | None,
    *,
    tolerance: float,
    continuity_weight: float,
    eligibility: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    if prior_allocations is None or prior_allocations.empty or seed.size == 0:
        return seed, {
            "continuity_prior_overlap_face": 0.0,
            "continuity_prior_rows": 0.0,
            "continuity_prior_weight": continuity_weight,
        }
    required = {_SECTOR_COL, _COHORT_ID_COL, _ALLOCATION_COL}
    if not required.issubset(prior_allocations.columns):
        return seed, {
            "continuity_prior_overlap_face": 0.0,
            "continuity_prior_rows": 0.0,
            "continuity_prior_weight": continuity_weight,
        }

    cohort_ids = cohorts[_COHORT_ID_COL].astype(str).tolist()
    sector_ids = sectors[_SECTOR_COL].astype(str).tolist()
    cohort_index = {cohort_id: idx for idx, cohort_id in enumerate(cohort_ids)}
    sector_index = {sector_id: idx for idx, sector_id in enumerate(sector_ids)}
    prior = prior_allocations.copy()
    prior[_ALLOCATION_COL] = pd.to_numeric(prior[_ALLOCATION_COL], errors="coerce").fillna(0.0)
    prior_matrix = np.zeros_like(seed, dtype=float)
    matched_rows = 0
    for _, row in prior.iterrows():
        row_idx = sector_index.get(str(row.get(_SECTOR_COL)))
        col_idx = cohort_index.get(str(row.get(_COHORT_ID_COL)))
        if row_idx is None or col_idx is None:
            continue
        value = float(row.get(_ALLOCATION_COL, 0.0))
        if value <= 0.0:
            continue
        prior_matrix[row_idx, col_idx] += value
        matched_rows += 1

    overlap_face = float(prior_matrix.sum())
    if overlap_face <= tolerance:
        return seed, {
            "continuity_prior_overlap_face": 0.0,
            "continuity_prior_rows": float(matched_rows),
            "continuity_prior_weight": continuity_weight,
        }

    out = seed.astype(float, copy=True)
    allowed = np.ones_like(out, dtype=bool) if eligibility is None else eligibility.astype(bool, copy=False)
    for row_idx in range(out.shape[0]):
        row_prior = prior_matrix[row_idx, :]
        row_prior = np.where(allowed[row_idx, :], row_prior, 0.0)
        prior_sum = float(row_prior.sum())
        if prior_sum <= tolerance:
            continue
        source_row = out[row_idx, :]
        source_row = np.where(allowed[row_idx, :], source_row, 0.0)
        source_sum = float(source_row.sum())
        if source_sum <= tolerance:
            source_sum = float(allowed[row_idx, :].sum())
            source_row = np.where(allowed[row_idx, :], 1.0, 0.0)
        continuity_row = (row_prior / prior_sum) * source_sum
        out[row_idx, :] = source_row + (continuity_weight * continuity_row)
    out = np.where(allowed, np.clip(out, a_min=tolerance, a_max=None), 0.0)
    return out, {
        "continuity_prior_overlap_face": overlap_face,
        "continuity_prior_rows": float(matched_rows),
        "continuity_prior_weight": continuity_weight,
    }


def _row_value_coefficients(sectors: pd.DataFrame, cohorts: pd.DataFrame) -> np.ndarray:
    row_count = len(sectors.index)
    col_count = len(cohorts.index)
    coefficients = np.ones((row_count, col_count), dtype=float)
    if row_count == 0 or col_count == 0:
        return coefficients
    price_ratio = pd.to_numeric(
        cohorts.get("market_value_ratio", pd.Series(1.0, index=cohorts.index)),
        errors="coerce",
    ).fillna(1.0)
    price_ratio = price_ratio.clip(lower=1e-9).to_numpy(dtype=float)
    valuation_basis = sectors.get("valuation_basis", pd.Series("", index=sectors.index))
    for row_idx, basis in enumerate(valuation_basis):
        if "market_value" in str(basis).lower():
            coefficients[row_idx, :] = price_ratio
    return coefficients


def _approximate_sector_face_equivalent_total(sectors: pd.DataFrame, cohorts: pd.DataFrame) -> float:
    if sectors.empty:
        return 0.0
    values = pd.to_numeric(sectors.get("sector_stock"), errors="coerce").fillna(0.0)
    price_ratio = pd.to_numeric(
        cohorts.get("market_value_ratio", pd.Series(1.0, index=cohorts.index)),
        errors="coerce",
    ).fillna(1.0)
    cohort_outstanding = pd.to_numeric(
        cohorts.get("cohort_outstanding", pd.Series(0.0, index=cohorts.index)),
        errors="coerce",
    ).fillna(0.0)
    if float(cohort_outstanding.sum()) > 0.0:
        weighted_price = float((price_ratio * cohort_outstanding).sum() / cohort_outstanding.sum())
    else:
        weighted_price = float(price_ratio.mean()) if not price_ratio.empty else 1.0
    weighted_price = max(weighted_price, 1e-9)
    valuation_basis = sectors.get("valuation_basis", pd.Series("", index=sectors.index)).astype(str).str.lower()
    face_equivalent = values.copy()
    market_mask = valuation_basis.str.contains("market_value", na=False)
    face_equivalent.loc[market_mask] = values.loc[market_mask] / weighted_price
    return float(face_equivalent.sum())


def _face_equivalent_solver_targets(
    sector_totals: np.ndarray,
    coefficients: np.ndarray,
    seed: np.ndarray,
    sectors: pd.DataFrame,
    total_cohort: float,
    *,
    tolerance: float,
) -> np.ndarray:
    if sector_totals.size == 0:
        return sector_totals.copy()
    targets = np.asarray(sector_totals, dtype=float).copy()
    if coefficients.size == 0 or np.allclose(coefficients, 1.0, atol=1e-12, rtol=0.0):
        return targets
    for row_idx in range(targets.size):
        row_seed = seed[row_idx, :] if row_idx < seed.shape[0] else np.array([], dtype=float)
        row_coeff = coefficients[row_idx, :] if row_idx < coefficients.shape[0] else np.array([], dtype=float)
        seed_sum = float(row_seed.sum())
        if seed_sum <= tolerance or row_coeff.size == 0:
            avg_coeff = float(row_coeff.mean()) if row_coeff.size else 1.0
        else:
            avg_coeff = float((row_seed * row_coeff).sum() / seed_sum)
        avg_coeff = max(avg_coeff, tolerance)
        targets[row_idx] = targets[row_idx] / avg_coeff

    sector_names = sectors[_SECTOR_COL].astype(str).tolist() if _SECTOR_COL in sectors.columns else []
    residual_indices = [idx for idx, sector in enumerate(sector_names) if sector == "MSPD_Z1_SourceBasisResidual"]
    if residual_indices:
        residual_idx = residual_indices[0]
        non_residual_total = float(targets.sum() - targets[residual_idx])
        if non_residual_total > total_cohort and non_residual_total > tolerance:
            scale = total_cohort / non_residual_total
            for idx in range(targets.size):
                if idx != residual_idx:
                    targets[idx] *= scale
            targets[residual_idx] = 0.0
        else:
            targets[residual_idx] = max(total_cohort - non_residual_total, 0.0)
    target_sum = float(targets.sum())
    if target_sum > tolerance and abs(target_sum - total_cohort) > tolerance:
        targets *= total_cohort / target_sum
    return np.clip(targets, a_min=0.0, a_max=None)


def _run_ipf(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, int, bool, float]:
    matrix = seed.astype(float, copy=True)
    if matrix.shape != (len(row_targets), len(col_targets)):
        raise ValueError("Seed matrix shape does not match targets")
    if matrix.size == 0:
        return matrix, 0, True, 0.0

    converged = False
    max_error = math.inf
    for iteration in range(1, max_iterations + 1):
        row_sums = matrix.sum(axis=1)
        row_factors = np.divide(
            row_targets,
            row_sums,
            out=np.ones_like(row_targets, dtype=float),
            where=row_sums > 0.0,
        )
        matrix *= row_factors[:, None]

        col_sums = matrix.sum(axis=0)
        col_factors = np.divide(
            col_targets,
            col_sums,
            out=np.ones_like(col_targets, dtype=float),
            where=col_sums > 0.0,
        )
        matrix *= col_factors[None, :]

        row_error = np.max(np.abs(matrix.sum(axis=1) - row_targets)) if row_targets.size else 0.0
        col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets)) if col_targets.size else 0.0
        max_error = max(float(row_error), float(col_error))
        if max_error <= tolerance:
            converged = True
            return matrix, iteration, converged, max_error
    return matrix, max_iterations, converged, max_error


def _run_coefficient_ipf(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    coefficients: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int = 600,
) -> tuple[np.ndarray, int, bool, float]:
    matrix = seed.astype(float, copy=True)
    coefficients = np.asarray(coefficients, dtype=float)
    if matrix.shape != (len(row_targets), len(col_targets)):
        raise ValueError("Seed matrix shape does not match targets")
    if coefficients.shape != matrix.shape:
        raise ValueError("Coefficient matrix shape does not match seed")
    if matrix.size == 0:
        return matrix, 0, True, 0.0

    converged = False
    max_error = math.inf
    effective_tolerance = max(tolerance, 1.0e-6)
    for iteration in range(1, max_iterations + 1):
        row_values = (matrix * coefficients).sum(axis=1)
        row_factors = np.divide(
            row_targets,
            row_values,
            out=np.ones_like(row_targets, dtype=float),
            where=row_values > 0.0,
        )
        matrix *= row_factors[:, None]

        col_sums = matrix.sum(axis=0)
        col_factors = np.divide(
            col_targets,
            col_sums,
            out=np.ones_like(col_targets, dtype=float),
            where=col_sums > 0.0,
        )
        matrix *= col_factors[None, :]

        row_error = np.max(np.abs((matrix * coefficients).sum(axis=1) - row_targets)) if row_targets.size else 0.0
        col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets)) if col_targets.size else 0.0
        max_error = max(float(row_error), float(col_error))
        if max_error <= effective_tolerance:
            converged = True
            return matrix, iteration, converged, max_error
    return matrix, max_iterations, converged, max_error


def _run_exact_weighted_entropy_solver(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    coefficients: np.ndarray,
    eligibility: np.ndarray,
    row_constrained: np.ndarray | None = None,
    *,
    tolerance: float,
    max_iterations: int = 2_000,
) -> tuple[np.ndarray, int, bool, float]:
    """Solve exact weighted row and cohort constraints via KL projection."""

    if minimize is None:
        return _run_weighted_bregman_projection(
            seed,
            row_targets,
            col_targets,
            coefficients,
            eligibility,
            row_constrained=row_constrained,
            tolerance=tolerance,
        )
    matrix_shape = (len(row_targets), len(col_targets))
    if seed.shape != matrix_shape:
        raise ValueError("Seed matrix shape does not match targets")
    coefficients = np.asarray(coefficients, dtype=float)
    eligibility = eligibility.astype(bool, copy=False)
    if coefficients.shape != seed.shape or eligibility.shape != seed.shape:
        raise ValueError("Coefficient or eligibility matrix shape does not match seed")
    if row_constrained is None:
        row_constrained = np.ones(len(row_targets), dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    if row_constrained.shape != (len(row_targets),):
        raise ValueError("row_constrained shape does not match row targets")
    if seed.size == 0:
        return seed.astype(float, copy=True), 0, True, 0.0

    row_targets = np.asarray(row_targets, dtype=float)
    col_targets = np.asarray(col_targets, dtype=float)
    effective_tolerance = max(float(tolerance), 1.0e-8)

    if not _targets_have_support(
        row_targets,
        col_targets,
        eligibility,
        row_constrained=row_constrained,
        tolerance=effective_tolerance,
    ):
        return seed.astype(float, copy=True), 0, False, math.inf

    scale = max(float(np.abs(row_targets).sum()), float(np.abs(col_targets).sum()), 1.0)
    constrained_rows = np.flatnonzero(row_constrained)
    row_scaled = row_targets[constrained_rows] / scale
    col_scaled = col_targets / scale
    prior = np.where(eligibility, seed, 0.0).astype(float, copy=False) / scale
    prior = np.where(eligibility & (prior <= 0.0), effective_tolerance / scale, prior)

    full_row_count, col_count = seed.shape
    row_count = len(constrained_rows)

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        row_dual = params[:row_count]
        col_dual = params[row_count:]
        full_row_dual = np.zeros(full_row_count, dtype=float)
        full_row_dual[constrained_rows] = row_dual
        exponent = -((full_row_dual[:, None] * coefficients) + col_dual[None, :])
        exponent = np.clip(exponent, -700.0, 500.0)
        values = prior * np.exp(exponent)
        values = np.where(eligibility, values, 0.0)
        row_values = (values * coefficients).sum(axis=1)[constrained_rows]
        col_values = values.sum(axis=0)
        objective = float(values.sum() + np.dot(row_dual, row_scaled) + np.dot(col_dual, col_scaled))
        gradient = np.concatenate([row_scaled - row_values, col_scaled - col_values])
        return objective, gradient

    result = minimize(
        objective_and_gradient,
        np.zeros(row_count + col_count, dtype=float),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(max_iterations),
            "ftol": 1.0e-14,
            "gtol": max(effective_tolerance / scale, 1.0e-12),
            "maxls": 50,
        },
    )
    objective_and_gradient(np.asarray(result.x, dtype=float))
    row_dual = np.asarray(result.x[:row_count], dtype=float)
    col_dual = np.asarray(result.x[row_count:], dtype=float)
    full_row_dual = np.zeros(full_row_count, dtype=float)
    full_row_dual[constrained_rows] = row_dual
    exponent = -((full_row_dual[:, None] * coefficients) + col_dual[None, :])
    exponent = np.clip(exponent, -700.0, 500.0)
    matrix = np.where(eligibility, prior * np.exp(exponent), 0.0) * scale
    max_error = _weighted_constraint_error(
        matrix,
        row_targets,
        col_targets,
        coefficients,
        row_constrained=row_constrained,
    )
    if max_error <= effective_tolerance:
        return matrix, int(getattr(result, "nit", max_iterations)), True, max_error

    parameter_count = row_count + col_count
    if least_squares is not None and parameter_count <= 600:
        def residual_vector(params: np.ndarray) -> np.ndarray:
            return objective_and_gradient(params)[1]

        polished = least_squares(
            residual_vector,
            np.asarray(result.x, dtype=float),
            method="trf",
            max_nfev=600,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=max(effective_tolerance / scale, 1.0e-10),
        )
        row_dual = np.asarray(polished.x[:row_count], dtype=float)
        col_dual = np.asarray(polished.x[row_count:], dtype=float)
        full_row_dual = np.zeros(full_row_count, dtype=float)
        full_row_dual[constrained_rows] = row_dual
        exponent = -((full_row_dual[:, None] * coefficients) + col_dual[None, :])
        exponent = np.clip(exponent, -700.0, 500.0)
        polished_matrix = np.where(eligibility, prior * np.exp(exponent), 0.0) * scale
        polished_error = _weighted_constraint_error(
            polished_matrix,
            row_targets,
            col_targets,
            coefficients,
            row_constrained=row_constrained,
        )
        if polished_error < max_error:
            matrix = polished_matrix
            max_error = polished_error
        if polished_error <= effective_tolerance:
            return polished_matrix, int(getattr(result, "nit", max_iterations)) + int(polished.nfev), True, polished_error

    fallback_matrix, iterations, converged, fallback_error = _run_weighted_bregman_projection(
        seed,
        row_targets,
        col_targets,
        coefficients,
        eligibility,
        row_constrained=row_constrained,
        tolerance=tolerance,
        max_seconds=_WEIGHTED_BREGMAN_TIME_BUDGET_SECONDS,
    )
    if fallback_error < max_error:
        return fallback_matrix, iterations, converged, fallback_error
    return matrix, int(getattr(result, "nit", max_iterations)), False, max_error


def _targets_have_support(
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    eligibility: np.ndarray,
    *,
    row_constrained: np.ndarray | None = None,
    tolerance: float,
) -> bool:
    if row_constrained is None:
        row_constrained = np.ones(len(row_targets), dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    positive_rows = (np.asarray(row_targets, dtype=float) > tolerance) & row_constrained
    positive_cols = np.asarray(col_targets, dtype=float) > tolerance
    if positive_rows.any() and (~eligibility[positive_rows, :].any(axis=1)).any():
        return False
    if positive_cols.any() and (~eligibility[:, positive_cols].any(axis=0)).any():
        return False
    return True


def _weighted_constraint_error(
    matrix: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    coefficients: np.ndarray,
    *,
    row_constrained: np.ndarray | None = None,
) -> float:
    if row_constrained is None:
        row_constrained = np.ones(len(row_targets), dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    row_residual = (matrix * coefficients).sum(axis=1) - row_targets
    row_error = np.max(np.abs(row_residual[row_constrained])) if row_targets.size and row_constrained.any() else 0.0
    col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets)) if col_targets.size else 0.0
    return float(max(row_error, col_error))


def _run_prior_preserving_qp_projection(
    seed: np.ndarray,
    feasible_matrix: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    coefficients: np.ndarray,
    eligibility: np.ndarray,
    row_constrained: np.ndarray | None = None,
    *,
    tolerance: float,
    max_active_set_iterations: int = 25,
) -> tuple[np.ndarray, int, bool, float]:
    """Project the prior to exact constraints with a bounded strictly convex QP.

    The minimum-slack LP is useful as a feasibility certificate, but its linear
    objective can choose arbitrary polytope corners. This projection instead
    minimizes a weighted squared distance to the continuity-tempered prior under
    the same hard row/cohort equalities and nonnegative eligibility bounds.
    """

    matrix_shape = seed.shape
    if feasible_matrix.shape != matrix_shape:
        raise ValueError("Feasible matrix shape does not match seed")
    if coefficients.shape != matrix_shape or eligibility.shape != matrix_shape:
        raise ValueError("Coefficient or eligibility matrix shape does not match seed")
    row_count, col_count = matrix_shape
    if row_constrained is None:
        row_constrained = np.ones(row_count, dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    if row_constrained.shape != (row_count,):
        raise ValueError("row_constrained shape does not match row targets")
    if seed.size == 0:
        return seed.astype(float, copy=True), 0, True, 0.0

    effective_tolerance = max(float(tolerance), 1.0e-10)
    feasible_error = _weighted_constraint_error(
        feasible_matrix,
        row_targets,
        col_targets,
        coefficients,
        row_constrained=row_constrained,
    )
    if feasible_error > effective_tolerance:
        return feasible_matrix.astype(float, copy=True), 0, False, feasible_error
    if not _targets_have_support(
        row_targets,
        col_targets,
        eligibility,
        row_constrained=row_constrained,
        tolerance=effective_tolerance,
    ):
        return feasible_matrix.astype(float, copy=True), 0, False, math.inf

    prior = np.where(eligibility, seed, 0.0).astype(float, copy=True)
    feasible = np.where(eligibility, feasible_matrix, 0.0).astype(float, copy=True)
    total = max(float(np.abs(col_targets).sum()), float(np.abs(row_targets[row_constrained]).sum()), 1.0)
    floor = max(effective_tolerance, total * 1.0e-12)
    prior = np.where(eligibility & (prior <= 0.0), floor, prior)
    reference = 0.995 * prior + 0.005 * np.maximum(feasible, 0.0)
    reference = np.where(eligibility, np.clip(reference, a_min=floor, a_max=None), 0.0)

    constrained_rows = np.flatnonzero(row_constrained)
    target = np.concatenate([np.asarray(row_targets, dtype=float)[constrained_rows], np.asarray(col_targets, dtype=float)])
    row_position = np.full(row_count, -1, dtype=int)
    row_position[constrained_rows] = np.arange(len(constrained_rows), dtype=int)
    solution = np.zeros(matrix_shape, dtype=float)
    free = eligibility.astype(bool, copy=True)

    for iteration in range(1, max_active_set_iterations + 1):
        row_idx, col_idx = np.nonzero(free)
        free_count = len(row_idx)
        if free_count == 0:
            break

        constraint_count = len(constrained_rows) + col_count
        constraint_matrix = np.zeros((constraint_count, free_count), dtype=float)
        variable_positions = np.arange(free_count)
        constrained_position = row_position[row_idx]
        constrained_mask = constrained_position >= 0
        if constrained_mask.any():
            constraint_matrix[
                constrained_position[constrained_mask],
                variable_positions[constrained_mask],
            ] = coefficients[row_idx[constrained_mask], col_idx[constrained_mask]]
        constraint_matrix[len(constrained_rows) + col_idx, variable_positions] = 1.0

        reference_vector = reference[row_idx, col_idx]
        inverse_weights = np.clip(reference_vector, a_min=floor, a_max=None)
        weighted_constraints = constraint_matrix * inverse_weights[None, :]
        normal_matrix = weighted_constraints @ constraint_matrix.T
        rhs = constraint_matrix @ reference_vector - target
        try:
            dual = np.linalg.lstsq(normal_matrix, rhs, rcond=1.0e-12)[0]
        except np.linalg.LinAlgError:
            return feasible, iteration, False, feasible_error
        candidate_vector = reference_vector - inverse_weights * (constraint_matrix.T @ dual)

        negative = candidate_vector < -effective_tolerance
        if negative.any():
            negative_rows = row_idx[negative]
            negative_cols = col_idx[negative]
            free[negative_rows, negative_cols] = False
            solution[negative_rows, negative_cols] = 0.0
            continue

        candidate = np.zeros(matrix_shape, dtype=float)
        candidate[row_idx, col_idx] = np.clip(candidate_vector, a_min=0.0, a_max=None)
        candidate = np.where(eligibility, candidate, 0.0)
        max_error = _weighted_constraint_error(
            candidate,
            row_targets,
            col_targets,
            coefficients,
            row_constrained=row_constrained,
        )
        if max_error <= effective_tolerance:
            return candidate, iteration, True, max_error
        solution = candidate
        break

    max_error = _weighted_constraint_error(
        solution,
        row_targets,
        col_targets,
        coefficients,
        row_constrained=row_constrained,
    )
    if max_error <= effective_tolerance:
        return solution, max_active_set_iterations, True, max_error
    return feasible, max_active_set_iterations, False, feasible_error


def _run_weighted_bregman_projection(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    coefficients: np.ndarray,
    eligibility: np.ndarray,
    row_constrained: np.ndarray | None = None,
    *,
    tolerance: float,
    max_iterations: int = 4_000,
    max_seconds: float | None = None,
) -> tuple[np.ndarray, int, bool, float]:
    """Dependency-light alternating KL projections for weighted row constraints."""

    matrix = np.where(eligibility, seed, 0.0).astype(float, copy=True)
    started_at = time.monotonic()
    coefficients = np.asarray(coefficients, dtype=float)
    effective_tolerance = max(float(tolerance), 1.0e-8)
    if row_constrained is None:
        row_constrained = np.ones(len(row_targets), dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    if matrix.size == 0:
        return matrix, 0, True, 0.0
    if not _targets_have_support(
        row_targets,
        col_targets,
        eligibility,
        row_constrained=row_constrained,
        tolerance=effective_tolerance,
    ):
        return matrix, 0, False, math.inf

    for iteration in range(1, max_iterations + 1):
        if max_seconds is not None and time.monotonic() - started_at > max_seconds:
            return matrix, iteration - 1, False, _weighted_constraint_error(
                matrix,
                row_targets,
                col_targets,
                coefficients,
                row_constrained=row_constrained,
            )
        for row_idx in range(matrix.shape[0]):
            if not row_constrained[row_idx]:
                continue
            allowed = eligibility[row_idx, :]
            target = float(row_targets[row_idx])
            if target <= effective_tolerance:
                matrix[row_idx, :] = 0.0
                continue
            if not allowed.any():
                continue
            current = matrix[row_idx, allowed]
            coeff = coefficients[row_idx, allowed]
            if current.sum() <= 0.0:
                current = np.ones_like(current) * effective_tolerance
            matrix[row_idx, allowed] = _weighted_row_projection(current, coeff, target)
            matrix[row_idx, ~allowed] = 0.0

        col_sums = matrix.sum(axis=0)
        col_factors = np.divide(
            col_targets,
            col_sums,
            out=np.ones_like(col_targets, dtype=float),
            where=col_sums > 0.0,
        )
        matrix *= col_factors[None, :]
        matrix = np.where(eligibility, matrix, 0.0)

        max_error = _weighted_constraint_error(
            matrix,
            row_targets,
            col_targets,
            coefficients,
            row_constrained=row_constrained,
        )
        if max_error <= effective_tolerance:
            return matrix, iteration, True, max_error
    return matrix, max_iterations, False, _weighted_constraint_error(
        matrix,
        row_targets,
        col_targets,
        coefficients,
        row_constrained=row_constrained,
    )


def _weighted_row_projection(current: np.ndarray, coefficients: np.ndarray, target: float) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if current.size == 0:
        return current
    current = np.clip(current, a_min=0.0, a_max=None)
    coefficients = np.clip(coefficients, a_min=1.0e-12, a_max=None)
    base = float((current * coefficients).sum())
    if base <= 0.0:
        current = np.ones_like(current)
        base = float((current * coefficients).sum())
    if abs(base - target) <= 1.0e-12:
        return current
    if float(np.ptp(coefficients)) <= 1.0e-12:
        return current * (target / base)
    lo = -1.0
    hi = 1.0

    def value(theta: float) -> float:
        return float((coefficients * current * np.exp(np.clip(theta * coefficients, -700.0, 700.0))).sum())

    while value(lo) > target and lo > -1.0e8:
        lo *= 2.0
    while value(hi) < target and hi < 1.0e8:
        hi *= 2.0
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if value(mid) < target:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2.0
    return current * np.exp(np.clip(theta * coefficients, -700.0, 700.0))


def solve_sector_cohort_allocations(
    cohorts: pd.DataFrame,
    sector_stocks: pd.DataFrame,
    *,
    quarter: object | None = None,
    tolerance: float = 1e-9,
    prior_allocations: pd.DataFrame | None = None,
    continuity_weight: float = 8.0,
    fixed_allocations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate cohort outstanding across sectors with explicit diagnostics.

    Cohort supply is hard. Aggregate holder observations remain source facts:
    when MSPD supply exceeds observed sector totals, the difference is allocated
    to an explicit basis-residual sector; when observed sector totals exceed
    MSPD supply, the solver records holder-level slack rather than silently
    rescaling the source observations.
    """

    prepared_cohorts, cohort_quarter, _ = _prepare_cohorts(cohorts, quarter)
    prepared_sectors, sector_quarter, _ = _prepare_sector_stocks(sector_stocks, quarter)
    resolved_quarter = _resolve_output_quarter(quarter, cohort_quarter, sector_quarter)

    cohort_totals = prepared_cohorts["cohort_outstanding"].to_numpy(dtype=float)
    sector_totals = prepared_sectors["sector_stock"].to_numpy(dtype=float)
    total_cohort = float(cohort_totals.sum())
    total_sector = float(sector_totals.sum())
    raw_mspd_minus_z1_total = total_cohort - total_sector

    if total_sector <= tolerance and total_cohort <= tolerance:
        allocations = pd.DataFrame(
            columns=[_QUARTER_COL, _SECTOR_COL, _COHORT_ID_COL, _ALLOCATION_COL]
        )
        diagnostics = pd.DataFrame(
            [
                {
                    _QUARTER_COL: resolved_quarter,
                    "diagnostic_type": "total_balance",
                    "subject": "all",
                    "input_total": 0.0,
                    "target_total": 0.0,
                    "achieved_total": 0.0,
                    "residual": 0.0,
                    "scale_factor": 1.0,
                    "status": "empty",
                }
            ]
        )
        return allocations, diagnostics

    solver_sectors = prepared_sectors.copy()
    solver_status = "balanced"
    solver_method = "exact_weighted_entropy_projection"
    cell_portfolio_status = "cell_portfolio_entropy_projected"
    sector_face_equivalent_total = _approximate_sector_face_equivalent_total(
        prepared_sectors,
        prepared_cohorts,
    )
    approximate_basis_residual = total_cohort - sector_face_equivalent_total
    needs_basis_residual = max(raw_mspd_minus_z1_total, approximate_basis_residual) > tolerance
    if needs_basis_residual:
        solver_status = "explicit_basis_residual"
        residual_row = {column: pd.NA for column in solver_sectors.columns}
        residual_row.update(
            {
                _SECTOR_COL: "MSPD_Z1_SourceBasisResidual",
                "sector_stock": 0.0,
                "native_sector": "MSPD_Z1_SourceBasisResidual",
                "broad_holder_class": "source_basis_residual",
                "tdcsim_holder": "source_basis_residual",
                "tdcsim_holder_subbucket": "",
                "raw_z1_level": 0.0,
                "sector_target_before_scale": 0.0,
                "valuation_basis": "source_basis_residual_current_principal",
                "source_status": "explicit_source_or_valuation_basis_residual",
                "evidence_label": "residual",
                "sector_adjustment_status": "explicit_basis_residual",
            }
        )
        solver_sectors = pd.concat(
            [solver_sectors, pd.DataFrame([residual_row])],
            ignore_index=True,
        )
    elif approximate_basis_residual < -tolerance:
        solver_status = "negative_source_basis_requires_slack"

    solver_sector_totals = solver_sectors["sector_stock"].to_numpy(dtype=float)
    residual_rows = solver_sectors[_SECTOR_COL].astype(str).eq("MSPD_Z1_SourceBasisResidual").to_numpy()
    row_constrained = ~residual_rows
    seed = _extract_seed_matrix(prepared_cohorts, solver_sectors, tolerance=tolerance)
    eligibility = _extract_eligibility_matrix(prepared_cohorts, solver_sectors)
    seed = np.where(eligibility, seed, 0.0)
    seed, continuity_diagnostics = _apply_continuity_prior(
        seed,
        prepared_cohorts,
        solver_sectors,
        prior_allocations,
        tolerance=tolerance,
        continuity_weight=continuity_weight,
        eligibility=eligibility,
    )
    row_coefficients = _row_value_coefficients(solver_sectors, prepared_cohorts)
    fixed_matrix, fixed_diagnostics = _fixed_allocation_matrix(
        fixed_allocations,
        prepared_cohorts,
        solver_sectors,
        row_coefficients,
        quarter=resolved_quarter,
        tolerance=tolerance,
    )
    fixed_face_by_cohort = fixed_matrix.sum(axis=0) if fixed_matrix.size else np.zeros_like(cohort_totals)
    fixed_value_by_sector = (
        (fixed_matrix * row_coefficients).sum(axis=1)
        if fixed_matrix.size
        else np.zeros_like(solver_sector_totals)
    )
    solve_cohort_totals = np.clip(cohort_totals - fixed_face_by_cohort, a_min=0.0, a_max=None)
    solve_sector_totals = np.clip(solver_sector_totals - fixed_value_by_sector, a_min=0.0, a_max=None)
    fixed_exact_rows = fixed_diagnostics.get("fixed_exact_row", np.zeros_like(solver_sector_totals, dtype=bool))
    if len(fixed_exact_rows):
        solve_sector_totals[fixed_exact_rows] = 0.0
        solver_sector_totals = solver_sector_totals.copy()
        solver_sector_totals[fixed_exact_rows] = fixed_value_by_sector[fixed_exact_rows]
    modeled_face_equivalent_basis_residual = approximate_basis_residual
    matrix, iterations, converged, max_error = _run_exact_weighted_entropy_solver(
        seed,
        solve_sector_totals,
        solve_cohort_totals,
        row_coefficients,
        eligibility,
        row_constrained=row_constrained,
        tolerance=tolerance,
    )
    sector_slack_plus = np.zeros(len(solver_sectors.index), dtype=float)
    sector_slack_minus = np.zeros(len(solver_sectors.index), dtype=float)
    if converged:
        if solver_status == "balanced" and raw_mspd_minus_z1_total > tolerance:
            solver_status = "explicit_basis_residual"
    else:
        solver_status = "weighted_basis_slack"
        solver_method = "linprog_minimum_weighted_slack"
        slack_matrix, sector_slack_plus, sector_slack_minus, converged, max_error = _run_min_slack_solver(
            seed,
            solve_sector_totals,
            solve_cohort_totals,
            tolerance=tolerance,
            row_coefficients=row_coefficients,
            eligibility=eligibility,
            row_constrained=row_constrained,
        )
        slack_certificate_error = max_error
        matrix = slack_matrix
        iterations = 1
        if converged:
            adjusted_sector_totals = solve_sector_totals.copy()
            adjusted_sector_totals[row_constrained] = (
                adjusted_sector_totals[row_constrained]
                + sector_slack_plus[row_constrained]
                - sector_slack_minus[row_constrained]
            )
            entropy_matrix, entropy_iterations, entropy_converged, entropy_error = _run_exact_weighted_entropy_solver(
                seed,
                adjusted_sector_totals,
                solve_cohort_totals,
                row_coefficients,
                eligibility,
                row_constrained=row_constrained,
                tolerance=tolerance,
            )
            if entropy_converged:
                matrix = entropy_matrix
                iterations = entropy_iterations
                max_error = entropy_error
                solver_method = "exact_weighted_entropy_projection_with_certified_slack"
            else:
                zero_slack_certificate = (
                    slack_abs_sum(sector_slack_plus, sector_slack_minus, row_constrained)
                    <= max(tolerance, 1.0e-8)
                )
                if zero_slack_certificate:
                    qp_matrix, qp_iterations, qp_converged, qp_error = _run_prior_preserving_qp_projection(
                        seed,
                        slack_matrix,
                        solve_sector_totals,
                        solve_cohort_totals,
                        row_coefficients,
                        eligibility,
                        row_constrained=row_constrained,
                        tolerance=tolerance,
                    )
                    if qp_converged:
                        matrix = qp_matrix
                        iterations = 1 + entropy_iterations + qp_iterations
                        max_error = qp_error
                        solver_status = "prior_preserving_exact_fallback"
                        solver_method = "exact_prior_preserving_qp_projection_from_highs_feasible"
                        cell_portfolio_status = "cell_portfolio_prior_preserving_qp_projected"
                    else:
                        matrix = slack_matrix
                        iterations = 1 + entropy_iterations + qp_iterations
                        max_error = slack_certificate_error
                        solver_status = "zero_slack_highs_feasible_projection"
                        solver_method = "exact_feasible_highs_projection"
                        cell_portfolio_status = "cell_portfolio_zero_slack_highs_feasible_projected"
                        converged = True
                else:
                    matrix = slack_matrix
                    iterations = 1 + entropy_iterations
                    max_error = slack_certificate_error
                    solver_method = "aggregate_only_minimum_weighted_slack_certificate"
                    cell_portfolio_status = "aggregate_only_no_cell_portfolio"
                    converged = True
        if (
            converged
            and cell_portfolio_status == "aggregate_only_no_cell_portfolio"
            and slack_abs_sum(sector_slack_plus, sector_slack_minus, row_constrained) <= max(tolerance, 1.0e-8)
        ):
            max_error = slack_certificate_error

    if fixed_matrix.size:
        matrix = matrix + fixed_matrix

    diagnostic_tolerance = max(float(tolerance), 1.0e-8)
    allocation_rows: list[dict[str, object]] = []
    sectors = solver_sectors[_SECTOR_COL].tolist()
    cohort_ids = prepared_cohorts[_COHORT_ID_COL].tolist()
    if cell_portfolio_status != "aggregate_only_no_cell_portfolio":
        for sector_idx, sector in enumerate(sectors):
            sector_meta = solver_sectors.iloc[sector_idx].to_dict()
            for cohort_idx, cohort_id in enumerate(cohort_ids):
                row = {
                    _QUARTER_COL: resolved_quarter,
                    _SECTOR_COL: sector,
                    _COHORT_ID_COL: cohort_id,
                    _ALLOCATION_COL: float(matrix[sector_idx, cohort_idx]),
                }
                for column in [
                    "native_sector",
                    "broad_holder_class",
                    "tdcsim_holder",
                    "tdcsim_holder_subbucket",
                    "source_status",
                    "evidence_label",
                    "sector_adjustment_status",
                    "valuation_basis",
                ]:
                    if column in sector_meta:
                        row[column] = sector_meta[column]
                if fixed_matrix.size and fixed_matrix[sector_idx, cohort_idx] > diagnostic_tolerance:
                    row["fixed_allocation_source"] = str(
                        fixed_diagnostics.get("fixed_source_by_cell", np.full_like(fixed_matrix, "", dtype=object))[sector_idx, cohort_idx]
                    )
                    row["fixed_allocation_policy"] = str(
                        fixed_diagnostics.get("fixed_policy_by_cell", np.full_like(fixed_matrix, "", dtype=object))[sector_idx, cohort_idx]
                    )
                allocation_rows.append(row)
    allocation_columns = [
        _QUARTER_COL,
        _SECTOR_COL,
        _COHORT_ID_COL,
        _ALLOCATION_COL,
        "native_sector",
        "broad_holder_class",
        "tdcsim_holder",
        "tdcsim_holder_subbucket",
        "source_status",
        "evidence_label",
        "sector_adjustment_status",
        "valuation_basis",
        "fixed_allocation_source",
        "fixed_allocation_policy",
    ]
    allocations = pd.DataFrame(allocation_rows, columns=allocation_columns)
    achieved_sector_totals = (matrix * row_coefficients).sum(axis=1)
    achieved_sector_face_totals = matrix.sum(axis=1)
    if residual_rows.any():
        modeled_face_equivalent_basis_residual = float(achieved_sector_face_totals[residual_rows].sum())
        solver_sector_totals = solver_sector_totals.copy()
        solver_sector_totals[residual_rows] = achieved_sector_totals[residual_rows]
    if abs(modeled_face_equivalent_basis_residual) <= diagnostic_tolerance:
        modeled_face_equivalent_basis_residual = 0.0
    slack_abs_max = 0.0
    if len(sector_slack_plus):
        slack_abs_max = max(slack_abs_max, float(np.max(np.abs(sector_slack_plus))))
    if len(sector_slack_minus):
        slack_abs_max = max(slack_abs_max, float(np.max(np.abs(sector_slack_minus))))
    actual_constraint_error = _weighted_constraint_error(
        matrix,
        solver_sector_totals,
        cohort_totals,
        row_coefficients,
        row_constrained=row_constrained,
    )
    exact_balance_scale = (
        1.0
        if abs(raw_mspd_minus_z1_total) <= max(tolerance, 1.0e-6)
        and actual_constraint_error <= max(tolerance, 1.0e-6)
        else np.nan
    )

    diagnostics_rows = [
        {
            _QUARTER_COL: resolved_quarter,
            "diagnostic_type": "total_balance",
            "subject": "all",
            "input_total": total_sector,
            "target_total": total_cohort,
            "achieved_total": float(matrix.sum()),
            "residual": raw_mspd_minus_z1_total,
            "scale_factor": exact_balance_scale,
            "status": solver_status,
            "solver_method": solver_method,
            "cell_portfolio_status": cell_portfolio_status,
            "basis_residual": modeled_face_equivalent_basis_residual,
            "raw_mspd_minus_z1_total": raw_mspd_minus_z1_total,
            "modeled_face_equivalent_basis_residual": modeled_face_equivalent_basis_residual,
            "sector_face_equivalent_total": sector_face_equivalent_total,
            "actual_weighted_constraint_error": actual_constraint_error,
            "slack_abs_max": slack_abs_max,
            **continuity_diagnostics,
            "fixed_observed_face_mil": float(fixed_face_by_cohort.sum()),
            "fixed_observed_value_mil": float(fixed_value_by_sector.sum()),
        },
        {
            _QUARTER_COL: resolved_quarter,
            "diagnostic_type": "solver",
            "subject": solver_method,
            "input_total": float(iterations),
            "target_total": np.nan,
            "achieved_total": float(max_error),
            "residual": float(max_error),
            "scale_factor": np.nan,
            "status": "converged" if converged else "max_iterations_reached",
            "solver_method": solver_method,
            "cell_portfolio_status": cell_portfolio_status,
            "basis_residual": modeled_face_equivalent_basis_residual,
            "raw_mspd_minus_z1_total": raw_mspd_minus_z1_total,
            "modeled_face_equivalent_basis_residual": modeled_face_equivalent_basis_residual,
            "sector_face_equivalent_total": sector_face_equivalent_total,
            **continuity_diagnostics,
            "fixed_observed_face_mil": float(fixed_face_by_cohort.sum()),
            "fixed_observed_value_mil": float(fixed_value_by_sector.sum()),
        },
    ]

    for sector_idx, (sector, input_total, target_total, achieved_total) in enumerate(zip(
        sectors,
        solver_sector_totals,
        solver_sector_totals,
        achieved_sector_totals,
        strict=False,
    )):
        sector_meta = solver_sectors.iloc[sector_idx].to_dict()
        slack_plus = float(sector_slack_plus[sector_idx]) if sector_idx < len(sector_slack_plus) else 0.0
        slack_minus = float(sector_slack_minus[sector_idx]) if sector_idx < len(sector_slack_minus) else 0.0
        sector_status = "matched" if abs(achieved_total - input_total) <= diagnostic_tolerance else "basis_slack"
        if sector == "MSPD_Z1_SourceBasisResidual":
            sector_status = "explicit_basis_residual"
        row = {
            _QUARTER_COL: resolved_quarter,
            "diagnostic_type": "sector_balance",
            "subject": sector,
            "input_total": float(input_total),
            "target_total": float(target_total),
            "achieved_total": float(achieved_total),
            "achieved_face_total": float(achieved_sector_face_totals[sector_idx]),
            "residual": float(achieved_total - input_total),
            "scale_factor": np.nan,
            "status": sector_status,
            "solver_method": solver_method,
            "cell_portfolio_status": cell_portfolio_status,
            "basis_residual": modeled_face_equivalent_basis_residual,
            "raw_mspd_minus_z1_total": raw_mspd_minus_z1_total,
            "modeled_face_equivalent_basis_residual": modeled_face_equivalent_basis_residual,
            "sector_face_equivalent_total": sector_face_equivalent_total,
            "solver_face_target_total": pd.NA,
            "slack_positive_mil": slack_plus,
            "slack_negative_mil": slack_minus,
            **continuity_diagnostics,
            "fixed_observed_face_mil": float(fixed_matrix[sector_idx, :].sum()) if fixed_matrix.size else 0.0,
            "fixed_observed_value_mil": float(fixed_value_by_sector[sector_idx]) if sector_idx < len(fixed_value_by_sector) else 0.0,
        }
        for column in [
            "native_sector",
            "broad_holder_class",
            "tdcsim_holder",
            "tdcsim_holder_subbucket",
            "raw_z1_level",
            "sector_target_before_scale",
            "valuation_basis",
            "source_status",
            "evidence_label",
            "sector_adjustment_status",
        ]:
            if column in sector_meta:
                row[column] = sector_meta[column]
        diagnostics_rows.append(row)

    achieved_cohort_totals = matrix.sum(axis=0)
    for cohort_idx, (cohort_id, input_total, achieved_total) in enumerate(zip(
        cohort_ids,
        cohort_totals,
        achieved_cohort_totals,
        strict=False,
    )):
        diagnostics_rows.append(
            {
                _QUARTER_COL: resolved_quarter,
                "diagnostic_type": "cohort_balance",
                "subject": cohort_id,
                "input_total": float(input_total),
                "target_total": float(input_total),
                "achieved_total": float(achieved_total),
                "residual": float(achieved_total - input_total),
                "scale_factor": 1.0,
                "status": "matched" if abs(achieved_total - input_total) <= tolerance else "mismatch",
                "solver_method": solver_method,
                "cell_portfolio_status": cell_portfolio_status,
                "basis_residual": modeled_face_equivalent_basis_residual,
                "raw_mspd_minus_z1_total": raw_mspd_minus_z1_total,
                "modeled_face_equivalent_basis_residual": modeled_face_equivalent_basis_residual,
                "sector_face_equivalent_total": sector_face_equivalent_total,
                **continuity_diagnostics,
                "fixed_observed_face_mil": float(fixed_face_by_cohort[cohort_idx]) if fixed_matrix.size else 0.0,
                "fixed_observed_value_mil": pd.NA,
            }
        )

    diagnostics = pd.DataFrame(diagnostics_rows)
    allocations[_ALLOCATION_COL] = pd.to_numeric(allocations[_ALLOCATION_COL], errors="coerce").fillna(0.0).clip(lower=0.0)
    allocations.loc[allocations[_ALLOCATION_COL].abs() <= _EXPORT_ALLOCATION_TOLERANCE, _ALLOCATION_COL] = 0.0
    allocations = allocations[allocations[_ALLOCATION_COL].gt(0.0)].reset_index(drop=True)
    return allocations, diagnostics


def slack_abs_sum(
    slack_plus: np.ndarray,
    slack_minus: np.ndarray,
    row_constrained: np.ndarray | None = None,
) -> float:
    if slack_plus.size == 0 and slack_minus.size == 0:
        return 0.0
    if row_constrained is None:
        mask = np.ones(max(len(slack_plus), len(slack_minus)), dtype=bool)
    else:
        mask = np.asarray(row_constrained, dtype=bool)
    plus = np.asarray(slack_plus, dtype=float)
    minus = np.asarray(slack_minus, dtype=float)
    total = 0.0
    if plus.size:
        total += float(np.abs(plus[mask[: plus.size]]).sum())
    if minus.size:
        total += float(np.abs(minus[mask[: minus.size]]).sum())
    return total


def _fixed_allocation_matrix(
    fixed_allocations: pd.DataFrame | None,
    cohorts: pd.DataFrame,
    sectors: pd.DataFrame,
    row_coefficients: np.ndarray,
    *,
    quarter: object | None,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    row_count = len(sectors.index)
    col_count = len(cohorts.index)
    matrix = np.zeros((row_count, col_count), dtype=float)
    source_by_cell = np.full((row_count, col_count), "", dtype=object)
    policy_by_cell = np.full((row_count, col_count), "", dtype=object)
    fixed_exact_row = np.zeros(row_count, dtype=bool)
    if fixed_allocations is None or fixed_allocations.empty or row_count == 0 or col_count == 0:
        return matrix, {
            "fixed_exact_row": fixed_exact_row,
            "fixed_source_by_cell": source_by_cell,
            "fixed_policy_by_cell": policy_by_cell,
        }
    required = {_SECTOR_COL, _COHORT_ID_COL, _ALLOCATION_COL}
    if not required.issubset(fixed_allocations.columns):
        return matrix, {
            "fixed_exact_row": fixed_exact_row,
            "fixed_source_by_cell": source_by_cell,
            "fixed_policy_by_cell": policy_by_cell,
        }
    fixed = fixed_allocations.copy()
    if _QUARTER_COL in fixed.columns and quarter is not None:
        fixed = fixed.loc[fixed[_QUARTER_COL].astype(str).eq(str(quarter))].copy()
    if fixed.empty:
        return matrix, {
            "fixed_exact_row": fixed_exact_row,
            "fixed_source_by_cell": source_by_cell,
            "fixed_policy_by_cell": policy_by_cell,
        }
    fixed[_ALLOCATION_COL] = pd.to_numeric(fixed[_ALLOCATION_COL], errors="coerce").fillna(0.0)
    fixed = fixed.loc[fixed[_ALLOCATION_COL].gt(tolerance)].copy()
    if fixed.empty:
        return matrix, {
            "fixed_exact_row": fixed_exact_row,
            "fixed_source_by_cell": source_by_cell,
            "fixed_policy_by_cell": policy_by_cell,
        }
    cohort_index = {str(value): idx for idx, value in enumerate(cohorts[_COHORT_ID_COL].astype(str))}
    sector_index = {str(value): idx for idx, value in enumerate(sectors[_SECTOR_COL].astype(str))}
    for _, row in fixed.iterrows():
        row_idx = sector_index.get(str(row.get(_SECTOR_COL)))
        col_idx = cohort_index.get(str(row.get(_COHORT_ID_COL)))
        if row_idx is None or col_idx is None:
            continue
        value = float(row.get(_ALLOCATION_COL, 0.0))
        if value <= tolerance:
            continue
        matrix[row_idx, col_idx] += value
        source = str(row.get("fixed_allocation_source", "observed_fixed_allocation"))
        policy = str(row.get("fixed_allocation_policy", "observed_exact_block"))
        source_by_cell[row_idx, col_idx] = source
        policy_by_cell[row_idx, col_idx] = policy
        if "exact" in policy:
            fixed_exact_row[row_idx] = True
    return matrix, {
        "fixed_exact_row": fixed_exact_row,
        "fixed_source_by_cell": source_by_cell,
        "fixed_policy_by_cell": policy_by_cell,
    }


def _run_min_slack_solver(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    tolerance: float,
    row_coefficients: np.ndarray | None = None,
    eligibility: np.ndarray | None = None,
    row_constrained: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Solve hard cohort totals with minimum row-observation slack."""

    if linprog is None:
        return _run_min_slack_fallback(
            seed,
            row_targets,
            col_targets,
            tolerance=tolerance,
            row_constrained=row_constrained,
        )
    row_count, col_count = seed.shape
    coefficients = (
        np.ones_like(seed, dtype=float)
        if row_coefficients is None
        else np.asarray(row_coefficients, dtype=float)
    )
    allowed = np.ones_like(seed, dtype=bool) if eligibility is None else eligibility.astype(bool, copy=False)
    if coefficients.shape != seed.shape:
        raise ValueError("row_coefficients shape does not match seed")
    if allowed.shape != seed.shape:
        raise ValueError("eligibility shape does not match seed")
    if row_constrained is None:
        row_constrained = np.ones(row_count, dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    if row_constrained.shape != (row_count,):
        raise ValueError("row_constrained shape does not match row targets")
    x_count = row_count * col_count
    variable_count = x_count + (2 * row_count)
    c = np.zeros(variable_count, dtype=float)
    normalized_seed = seed / max(float(seed.max()), tolerance)
    c[:x_count] = -1e-9 * normalized_seed.reshape(-1)
    c[x_count:] = 1.0

    a_eq = []
    b_eq = []
    for col_idx in range(col_count):
        row = np.zeros(variable_count, dtype=float)
        for row_idx in range(row_count):
            row[(row_idx * col_count) + col_idx] = 1.0
        a_eq.append(row)
        b_eq.append(float(col_targets[col_idx]))
    for row_idx in range(row_count):
        if not row_constrained[row_idx]:
            continue
        row = np.zeros(variable_count, dtype=float)
        start = row_idx * col_count
        row[start:start + col_count] = coefficients[row_idx, :]
        row[x_count + row_idx] = -1.0
        row[x_count + row_count + row_idx] = 1.0
        a_eq.append(row)
        b_eq.append(float(row_targets[row_idx]))

    bounds = [
        (0.0, None) if allowed.reshape(-1)[idx] else (0.0, 0.0)
        for idx in range(x_count)
    ]
    bounds.extend([(0.0, None)] * (2 * row_count))
    result = linprog(
        c,
        A_eq=np.asarray(a_eq),
        b_eq=np.asarray(b_eq),
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return _run_min_slack_fallback(
            seed,
            row_targets,
            col_targets,
            tolerance=tolerance,
            row_constrained=row_constrained,
        )
    values = np.asarray(result.x, dtype=float)
    matrix = values[:x_count].reshape(row_count, col_count)
    slack_plus = values[x_count:x_count + row_count]
    slack_minus = values[x_count + row_count:]
    slack_plus[~row_constrained] = 0.0
    slack_minus[~row_constrained] = 0.0
    achieved_rows = (matrix * coefficients).sum(axis=1)
    row_residual = (achieved_rows - slack_plus + slack_minus) - row_targets
    row_error = np.max(np.abs(row_residual[row_constrained])) if row_count and row_constrained.any() else 0.0
    col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets)) if col_count else 0.0
    return matrix, slack_plus, slack_minus, True, float(max(row_error, col_error))


def _run_min_slack_fallback(
    seed: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    *,
    tolerance: float,
    row_constrained: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """Dependency-free minimum-slack fallback with explicit diagnostics."""

    if row_constrained is None:
        row_constrained = np.ones(len(row_targets), dtype=bool)
    else:
        row_constrained = np.asarray(row_constrained, dtype=bool)
    total_rows = float(row_targets[row_constrained].sum())
    total_cols = float(col_targets.sum())
    if total_rows <= tolerance:
        matrix = np.zeros_like(seed, dtype=float)
    else:
        adjusted_rows = row_targets.copy()
        adjusted_rows[row_constrained] = row_targets[row_constrained] * (total_cols / total_rows)
        adjusted_rows[~row_constrained] = 0.0
        matrix, _, _, _ = _run_ipf(seed, adjusted_rows, col_targets, tolerance=tolerance)
    achieved = matrix.sum(axis=1)
    slack_plus = np.clip(achieved - row_targets, a_min=0.0, a_max=None)
    slack_minus = np.clip(row_targets - achieved, a_min=0.0, a_max=None)
    slack_plus[~row_constrained] = 0.0
    slack_minus[~row_constrained] = 0.0
    row_residual = (achieved - slack_plus + slack_minus) - row_targets
    row_error = np.max(np.abs(row_residual[row_constrained])) if row_targets.size and row_constrained.any() else 0.0
    col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets)) if col_targets.size else 0.0
    return matrix, slack_plus, slack_minus, False, float(max(row_error, col_error))
