"""Model-implied pricing bridge for historical replay portfolios."""

from __future__ import annotations

import math
import calendar
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CURVE_SOURCES = {
    "treasury_1m_daily": ("data/historical_replay/raw/fred/treasury_1m_daily__DGS1MO.csv", "DGS1MO", 1.0 / 12.0),
    "treasury_3m_daily": ("data/historical_replay/raw/fred/treasury_3m_daily__DGS3MO.csv", "DGS3MO", 0.25),
    "treasury_6m_daily": ("data/historical_replay/raw/fred/treasury_6m_daily__DGS6MO.csv", "DGS6MO", 0.5),
    "treasury_1y_daily": ("data/historical_replay/raw/fred/treasury_1y_daily__DGS1.csv", "DGS1", 1.0),
    "treasury_2y_daily": ("data/historical_replay/raw/fred/treasury_2y_daily__DGS2.csv", "DGS2", 2.0),
    "treasury_3y_daily": ("data/historical_replay/raw/fred/treasury_3y_daily__DGS3.csv", "DGS3", 3.0),
    "treasury_5y_daily": ("data/historical_replay/raw/fred/treasury_5y_daily__DGS5.csv", "DGS5", 5.0),
    "treasury_7y_daily": ("data/historical_replay/raw/fred/treasury_7y_daily__DGS7.csv", "DGS7", 7.0),
    "treasury_10y_daily": ("data/historical_replay/raw/fred/treasury_10y_daily__DGS10.csv", "DGS10", 10.0),
    "treasury_20y_daily": ("data/historical_replay/raw/fred/treasury_20y_daily__DGS20.csv", "DGS20", 20.0),
    "treasury_30y_daily": ("data/historical_replay/raw/fred/treasury_30y_daily__DGS30.csv", "DGS30", 30.0),
    "tbill_3m_monthly": ("data/historical_replay/raw/fred/tbill_3m_monthly__TB3MS.csv", "TB3MS", 0.25),
    "treasury_10y_monthly": ("data/historical_replay/raw/fred/treasury_10y_monthly__GS10.csv", "GS10", 10.0),
}

DEFAULT_REAL_CURVE_SOURCES = {
    "tips_5y_daily": ("data/historical_replay/raw/fred/tips_5y_daily__DFII5.csv", "DFII5", 5.0),
    "tips_7y_daily": ("data/historical_replay/raw/fred/tips_7y_daily__DFII7.csv", "DFII7", 7.0),
    "tips_10y_daily": ("data/historical_replay/raw/fred/tips_10y_daily__DFII10.csv", "DFII10", 10.0),
    "tips_20y_daily": ("data/historical_replay/raw/fred/tips_20y_daily__DFII20.csv", "DFII20", 20.0),
    "tips_30y_daily": ("data/historical_replay/raw/fred/tips_30y_daily__DFII30.csv", "DFII30", 30.0),
}

PRICING_DIAGNOSTIC_COLUMNS = [
    "quarter",
    "surface",
    "row_count",
    "priced_rows",
    "model_priced_rows",
    "unpriced_rows",
    "dirty_value_mil",
    "face_value_mil",
    "pricing_status",
    "claim_boundary",
]

PAYMENT_DATE_COLUMNS = [
    "interest_pay_date_1",
    "interest_pay_date_2",
    "interest_pay_date_3",
    "interest_pay_date_4",
]


def price_period_end_portfolios(
    period_end_portfolios: dict[str, pd.DataFrame],
    *,
    yield_curve: pd.DataFrame | None = None,
    real_yield_curve: pd.DataFrame | None = None,
    curve_sources: dict[str, object] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fill portfolio pricing columns with deterministic model-implied values."""

    curve = yield_curve if yield_curve is not None else load_nominal_yield_curve(curve_sources=curve_sources)
    real_curve = real_yield_curve if real_yield_curve is not None else load_real_yield_curve()
    priced: dict[str, pd.DataFrame] = {}
    diagnostic_rows = []
    for quarter, frame in period_end_portfolios.items():
        priced_frame = apply_model_pricing(
            frame,
            quarter=quarter,
            yield_curve=curve,
            real_yield_curve=real_curve,
        )
        priced[quarter] = priced_frame
        diagnostic_rows.append(_pricing_diagnostic_row(quarter, "period_end_portfolio", priced_frame))
    return priced, pd.DataFrame(diagnostic_rows, columns=PRICING_DIAGNOSTIC_COLUMNS)


def apply_model_pricing(
    portfolio: pd.DataFrame,
    *,
    quarter: str | None = None,
    yield_curve: pd.DataFrame | None = None,
    real_yield_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply an internally consistent model-price approximation to replay rows."""

    if portfolio is None or portfolio.empty:
        return portfolio.copy() if portfolio is not None else pd.DataFrame()
    frame = portfolio.copy()
    valuation_date = _valuation_date(quarter, frame)
    issue_date = pd.to_datetime(frame.get("IssueDate"), errors="coerce")
    maturity_date = pd.to_datetime(frame.get("MaturityDate"), errors="coerce")
    days_to_maturity = (maturity_date - valuation_date).dt.days
    time_to_maturity = (days_to_maturity.clip(lower=0) / 365.25).astype(float)

    face = _numeric_series(frame, "FaceValue", default=0.0)
    adjusted = _numeric_series(frame, "AdjustedPrincipal", default=np.nan)
    adjusted = adjusted.where(adjusted.notna() & (adjusted > 0.0), face)
    coupon = _numeric_series(frame, "CouponRate", default=0.0)
    fixed_spread = _numeric_series(frame, "FixedSpread", default=np.nan)
    frn_index = _numeric_series(frame, "BenchmarkRate_FRN", default=np.nan)
    issue_yield = _numeric_series(frame, "IssueYieldAtIssue", default=np.nan)
    issue_price = _numeric_series(frame, "IssuePriceRatio", default=np.nan)
    security_type = frame.get("SecurityType", pd.Series("", index=frame.index)).astype(str).str.lower()

    discount_yield = issue_yield.where(issue_yield.notna() & (issue_yield >= 0.0), coupon)
    discount_yield = discount_yield.fillna(0.0).clip(lower=0.0)
    discount_yield = discount_yield.where(discount_yield <= 1.0, discount_yield / 100.0)
    curve_points = _curve_points_for_valuation(yield_curve, valuation_date)
    real_curve_points = _curve_points_for_valuation(
        real_yield_curve,
        valuation_date,
        allow_negative_yields=True,
    )

    dirty_price = []
    accrued_per100 = []
    applied_yields = []
    methods = []
    statuses = []
    curve_statuses = []
    for idx in frame.index:
        stype = security_type.at[idx]
        ttm = float(time_to_maturity.at[idx]) if pd.notna(time_to_maturity.at[idx]) else 0.0
        fallback_yld = float(discount_yield.at[idx]) if pd.notna(discount_yield.at[idx]) else 0.0
        curve_yld, curve_status = _curve_yield_from_points(curve_points, ttm)
        real_curve_yld, real_curve_status = _curve_yield_from_points(real_curve_points, ttm)
        yld = curve_yld if curve_yld is not None else fallback_yld
        cpn = float(coupon.at[idx]) if pd.notna(coupon.at[idx]) else 0.0
        maturity = maturity_date.at[idx]
        payment_month_days = _payment_month_days(frame, idx)
        if "frn" in stype:
            frn_rate = _frn_index_plus_spread_rate(frn_index.at[idx], fixed_spread.at[idx])
            loaded_accrued = _loaded_frn_accrued_per100(frame, idx, face.at[idx])
            if frn_rate is not None:
                yld = frn_rate
                curve_status = "frn_index_plus_spread_par_approximation"
                method = "frn_index_plus_spread_par_approximation"
            else:
                curve_status = "frn_par_immaterial_approximation_missing_index_or_spread"
                method = "frn_par_immaterial_approximation_missing_index_or_spread"
            price = 100.0 + loaded_accrued
            accrued = loaded_accrued
        elif "bill" in stype or cpn <= 0.0:
            price = _zero_coupon_price(yld, ttm)
            accrued = 0.0
            method = "zero_coupon_current_curve_discount" if curve_yld is not None else "zero_coupon_issue_yield_discount"
        else:
            if "tips" in stype and real_curve_yld is not None:
                yld = real_curve_yld
                curve_status = real_curve_status.replace("current_curve", "real_current_curve")
            cashflow_dates = _remaining_coupon_dates(payment_month_days, valuation_date, maturity)
            price = _coupon_bond_dirty_price(
                cpn,
                yld,
                ttm,
                cashflow_dates=cashflow_dates,
                valuation_date=valuation_date,
                maturity_date=maturity,
                payment_frequency=_payment_frequency(payment_month_days, default=2),
            )
            accrued = _coupon_accrued_per100(
                issue_date.at[idx],
                valuation_date,
                cpn,
                payment_month_days=payment_month_days,
                maturity_date=maturity,
            )
            if "tips" in stype and real_curve_yld is not None:
                method = "tips_coupon_pv_from_real_curve"
            elif "tips" in stype and curve_yld is not None:
                method = "tips_coupon_pv_from_nominal_curve_proxy"
            else:
                method = "coupon_pv_from_current_curve" if curve_yld is not None else "coupon_pv_from_issue_yield"
            if cashflow_dates:
                curve_status = f"{curve_status}_scheduled_payment_dates"
        if pd.notna(issue_price.at[idx]) and (price <= 0.0 or not math.isfinite(price)):
            price = float(issue_price.at[idx]) * 100.0
            method = "issue_price_fallback"
            curve_status = "issue_price_fallback"
        price = _bounded_price(price)
        accrued = float(max(accrued, 0.0)) if math.isfinite(float(accrued)) else 0.0
        dirty_price.append(price)
        accrued_per100.append(accrued)
        applied_yields.append(yld)
        methods.append(method)
        statuses.append("model_implied_price")
        curve_statuses.append(curve_status)

    frame["TimeToMaturity"] = time_to_maturity
    frame["DiscountYield"] = pd.Series(applied_yields, index=frame.index)
    frame["DirtyPriceRatio"] = pd.Series(dirty_price, index=frame.index) / 100.0
    frame["AccruedInterest"] = pd.Series(accrued_per100, index=frame.index)
    frame["CleanPrice"] = (pd.Series(dirty_price, index=frame.index) - frame["AccruedInterest"]).clip(lower=0.0)
    principal_base = adjusted.where(security_type.str.contains("tips", na=False), face)
    frame["DirtyValue"] = principal_base * frame["DirtyPriceRatio"]
    frame["pricing_method"] = methods
    frame["pricing_source_status"] = statuses
    frame["pricing_curve_status"] = curve_statuses
    frame["pricing_claim_boundary"] = "model_implied_not_observed_market_price"
    return frame


def load_nominal_yield_curve(*, curve_sources: dict[str, object] | None = None) -> pd.DataFrame:
    """Load local Treasury yield observations into a long nominal curve table."""

    return _load_yield_curve(
        curve_sources or DEFAULT_CURVE_SOURCES,
        curve_family="nominal",
        allow_negative_yields=False,
    )


def load_real_yield_curve(*, curve_sources: dict[str, object] | None = None) -> pd.DataFrame:
    """Load local TIPS real-yield observations into a long curve table."""

    return _load_yield_curve(
        curve_sources or DEFAULT_REAL_CURVE_SOURCES,
        curve_family="real",
        allow_negative_yields=True,
    )


def _load_yield_curve(
    sources: dict[str, object],
    *,
    curve_family: str,
    allow_negative_yields: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, spec in sources.items():
        if isinstance(spec, dict):
            path = spec.get("path")
            value_col = spec.get("value_col")
            tenor_years = spec.get("tenor_years")
        else:
            path, value_col, tenor_years = spec
        if path is None or value_col is None or tenor_years is None:
            continue
        path_obj = Path(path)
        if not path_obj.exists():
            continue
        try:
            raw = pd.read_csv(path_obj)
        except Exception:
            continue
        if "observation_date" not in raw.columns or str(value_col) not in raw.columns:
            continue
        frame = pd.DataFrame(
            {
                "observation_date": pd.to_datetime(raw["observation_date"], errors="coerce"),
                "tenor_years": float(tenor_years),
                "yield_decimal": pd.to_numeric(raw[str(value_col)], errors="coerce") / 100.0,
                "source_key": str(key),
                "curve_family": curve_family,
            }
        )
        frame = frame.dropna(subset=["observation_date", "yield_decimal"])
        if not allow_negative_yields:
            frame = frame.loc[frame["yield_decimal"] >= 0.0]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["observation_date", "tenor_years", "yield_decimal", "source_key", "curve_family"])
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["observation_date", "tenor_years", "source_key"], kind="stable")
        .reset_index(drop=True)
    )


def build_pricing_scope_diagnostics(
    period_end_portfolios: dict[str, pd.DataFrame],
    final_portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Build pricing diagnostics for exported replay surfaces."""

    rows = []
    for quarter, frame in period_end_portfolios.items():
        rows.append(_pricing_diagnostic_row(quarter, "period_end_portfolio", frame))
    rows.append(_pricing_diagnostic_row(pd.NA, "final_portfolio", final_portfolio))
    return pd.DataFrame(rows, columns=PRICING_DIAGNOSTIC_COLUMNS)


def _pricing_diagnostic_row(quarter: object, surface: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame is None or frame.empty:
        return {
            "quarter": quarter,
            "surface": surface,
            "row_count": 0,
            "priced_rows": 0,
            "model_priced_rows": 0,
            "unpriced_rows": 0,
            "dirty_value_mil": 0.0,
            "face_value_mil": 0.0,
            "pricing_status": "empty_surface",
            "claim_boundary": "not_applicable",
        }
    price_cols = [col for col in ["CleanPrice", "DirtyValue", "DirtyPriceRatio"] if col in frame.columns]
    priced = frame[price_cols].notna().any(axis=1) if price_cols else pd.Series(False, index=frame.index)
    source_status = frame.get("pricing_source_status", pd.Series("", index=frame.index)).astype(str)
    model_priced = priced & source_status.eq("model_implied_price")
    dirty_value = pd.to_numeric(frame.get("DirtyValue"), errors="coerce").fillna(0.0).sum()
    face_value = pd.to_numeric(frame.get("FaceValue"), errors="coerce").fillna(0.0).sum()
    return {
        "quarter": quarter,
        "surface": surface,
        "row_count": int(len(frame.index)),
        "priced_rows": int(priced.sum()),
        "model_priced_rows": int(model_priced.sum()),
        "unpriced_rows": int((~priced).sum()),
        "dirty_value_mil": float(dirty_value),
        "face_value_mil": float(face_value),
        "pricing_status": "model_priced" if bool(priced.all()) else "partially_priced",
        "claim_boundary": "model_implied_not_observed_market_price",
    }


def _valuation_date(quarter: str | None, frame: pd.DataFrame) -> pd.Timestamp:
    if quarter is not None:
        return pd.Period(str(quarter), freq="Q").end_time.normalize()
    if "quarter" in frame.columns and frame["quarter"].notna().any():
        return pd.Period(str(frame["quarter"].dropna().iloc[0]), freq="Q").end_time.normalize()
    return pd.Timestamp.today().normalize()


def _curve_yield_for_maturity(
    yield_curve: pd.DataFrame | None,
    valuation_date: pd.Timestamp,
    maturity_years: float,
) -> tuple[float | None, str]:
    return _curve_yield_from_points(_curve_points_for_valuation(yield_curve, valuation_date), maturity_years)


def _curve_points_for_valuation(
    yield_curve: pd.DataFrame | None,
    valuation_date: pd.Timestamp,
    *,
    allow_negative_yields: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    if yield_curve is None or yield_curve.empty:
        return np.array([], dtype=float), np.array([], dtype=float), "no_curve_available_issue_yield_fallback"
    required = {"observation_date", "tenor_years", "yield_decimal"}
    if not required.issubset(yield_curve.columns):
        return np.array([], dtype=float), np.array([], dtype=float), "invalid_curve_issue_yield_fallback"
    curve = yield_curve.copy()
    curve["observation_date"] = pd.to_datetime(curve["observation_date"], errors="coerce")
    curve["tenor_years"] = pd.to_numeric(curve["tenor_years"], errors="coerce")
    curve["yield_decimal"] = pd.to_numeric(curve["yield_decimal"], errors="coerce")
    curve = curve.dropna(subset=["observation_date", "tenor_years", "yield_decimal"])
    curve = curve.loc[curve["observation_date"] <= valuation_date]
    if not allow_negative_yields:
        curve = curve.loc[curve["yield_decimal"] >= 0.0]
    if curve.empty:
        return np.array([], dtype=float), np.array([], dtype=float), "no_curve_observation_before_quarter_issue_yield_fallback"
    latest_by_tenor = (
        curve.sort_values(["tenor_years", "observation_date"], kind="stable")
        .groupby("tenor_years", sort=True, as_index=False)
        .tail(1)
        .sort_values("tenor_years", kind="stable")
    )
    if latest_by_tenor.empty:
        return np.array([], dtype=float), np.array([], dtype=float), "no_curve_tenor_issue_yield_fallback"
    return (
        latest_by_tenor["tenor_years"].to_numpy(dtype=float),
        latest_by_tenor["yield_decimal"].to_numpy(dtype=float),
        "current_curve",
    )


def _curve_yield_from_points(
    curve_points: tuple[np.ndarray, np.ndarray, str],
    maturity_years: float,
) -> tuple[float | None, str]:
    tenors, yields, empty_status = curve_points
    if maturity_years < 0.0:
        return None, "negative_maturity_issue_yield_fallback"
    if len(tenors) == 0:
        return None, empty_status
    if len(yields) == 0:
        return None, "no_curve_tenor_issue_yield_fallback"
    if len(tenors) == 1:
        return float(yields[0]), "single_tenor_current_curve"
    target = max(float(maturity_years), 0.0)
    if target <= float(tenors.min()):
        return float(yields[0]), "short_end_current_curve"
    if target >= float(tenors.max()):
        return float(yields[-1]), "long_end_current_curve"
    return float(np.interp(target, tenors, yields)), "interpolated_current_curve"


def _loaded_frn_accrued_per100(frame: pd.DataFrame, idx: object, face_value: object) -> float:
    face = float(face_value) if pd.notna(face_value) else 0.0
    accrued_total = _numeric_series(frame, "AccruedInterest_FRN", default=np.nan)
    loaded = accrued_total.at[idx]
    if pd.notna(loaded) and face > 0.0 and float(loaded) >= 0.0:
        return float(loaded) / face * 100.0
    return 0.0


def _frn_index_plus_spread_rate(index_rate: object, fixed_spread: object) -> float | None:
    index_value = _normalize_rate(index_rate)
    spread_value = _normalize_rate(fixed_spread)
    if index_value is None or spread_value is None:
        return None
    return index_value + spread_value


def _normalize_rate(value: object) -> float | None:
    if pd.isna(value):
        return None
    rate = float(value)
    if not math.isfinite(rate):
        return None
    if abs(rate) > 1.0:
        rate = rate / 100.0
    return rate


def _numeric_series(frame: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _zero_coupon_price(yield_rate: float, ttm: float) -> float:
    if ttm <= 0.0:
        return 100.0
    return 100.0 / (1.0 + max(yield_rate, 0.0) * ttm)


def _coupon_bond_dirty_price(
    coupon_rate: float,
    yield_rate: float,
    ttm: float,
    *,
    cashflow_dates: list[pd.Timestamp] | None = None,
    valuation_date: pd.Timestamp | None = None,
    maturity_date: object | None = None,
    payment_frequency: int = 2,
) -> float:
    if ttm <= 0.0:
        return 100.0
    frequency = max(int(payment_frequency), 1)
    if cashflow_dates and valuation_date is not None and maturity_date is not None:
        maturity = pd.to_datetime(maturity_date, errors="coerce")
        if pd.notna(maturity):
            period_coupon = coupon_rate * 100.0 / frequency
            coupon_pv = sum(
                period_coupon / _periodic_discount_base(yield_rate, frequency) ** _discount_periods(valuation_date, date, frequency)
                for date in cashflow_dates
                if date > valuation_date
            )
            principal_periods = _discount_periods(valuation_date, maturity, frequency)
            principal_pv = 100.0 / _periodic_discount_base(yield_rate, frequency) ** principal_periods
            return coupon_pv + principal_pv
    periods = max(int(math.ceil(ttm * frequency)), 1)
    period_coupon = coupon_rate * 100.0 / frequency
    period_yield = max(float(yield_rate) / frequency, -0.999999)
    if abs(period_yield) <= 1e-12:
        return 100.0 + period_coupon * periods
    coupons = sum(period_coupon / ((1.0 + period_yield) ** n) for n in range(1, periods + 1))
    principal = 100.0 / ((1.0 + period_yield) ** periods)
    return coupons + principal


def _periodic_discount_base(yield_rate: float, frequency: int) -> float:
    return max(1.0 + float(yield_rate) / max(int(frequency), 1), 1e-9)


def _discount_periods(
    valuation_date: pd.Timestamp,
    cashflow_date: pd.Timestamp,
    frequency: int,
) -> float:
    years = max(float((cashflow_date - valuation_date).days) / 365.25, 0.0)
    return years * max(int(frequency), 1)


def _coupon_accrued_per100(
    issue_date: object,
    valuation_date: pd.Timestamp,
    coupon_rate: float,
    *,
    payment_month_days: list[tuple[int, int]] | None = None,
    maturity_date: object | None = None,
) -> float:
    issued = pd.to_datetime(issue_date, errors="coerce")
    if pd.isna(issued) or coupon_rate <= 0.0:
        return 0.0
    if payment_month_days:
        maturity = pd.to_datetime(maturity_date, errors="coerce")
        schedule_end = valuation_date + pd.DateOffset(years=1)
        if pd.notna(maturity) and maturity > valuation_date:
            schedule_end = max(schedule_end, maturity)
        schedule = _scheduled_payment_dates(payment_month_days, issued, schedule_end)
        previous_dates = [date for date in schedule if date <= valuation_date]
        next_dates = [date for date in schedule if date > valuation_date]
        previous_date = max(previous_dates) if previous_dates else issued
        if issued > previous_date:
            previous_date = issued
        if previous_date == valuation_date:
            return 0.0
        if next_dates:
            next_date = min(next_dates)
            period_days = max(int((next_date - previous_date).days), 1)
            elapsed_days = max(int((valuation_date - previous_date).days), 0)
            frequency = _payment_frequency(payment_month_days, default=2)
            return coupon_rate * 100.0 / frequency * min(elapsed_days / period_days, 1.0)
    days_since_issue = max(int((valuation_date - issued).days), 0)
    days_in_coupon_period = 182.625
    days_elapsed = days_since_issue % int(round(days_in_coupon_period))
    return coupon_rate * 100.0 * (days_elapsed / 365.25)


def _payment_month_days(frame: pd.DataFrame, idx: object) -> list[tuple[int, int]]:
    month_days: list[tuple[int, int]] = []
    for column in PAYMENT_DATE_COLUMNS:
        if column not in frame.columns:
            continue
        parsed = _parse_payment_month_day(frame.at[idx, column])
        if parsed is not None and parsed not in month_days:
            month_days.append(parsed)
    return sorted(month_days)


def _parse_payment_month_day(value: object) -> tuple[int, int] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "nan", "na", "n/a"}:
        return None
    if "/" in text:
        parts = text.split("/")
        if len(parts) >= 2:
            try:
                month = int(parts[0])
                day = int(parts[1])
                if 1 <= month <= 12 and 1 <= day <= 31:
                    return month, day
            except ValueError:
                return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.month), int(parsed.day)


def _payment_frequency(month_days: list[tuple[int, int]] | None, *, default: int) -> int:
    if not month_days:
        return default
    return max(len(set(month_days)), 1)


def _remaining_coupon_dates(
    month_days: list[tuple[int, int]],
    valuation_date: pd.Timestamp,
    maturity_date: object,
) -> list[pd.Timestamp]:
    maturity = pd.to_datetime(maturity_date, errors="coerce")
    if not month_days or pd.isna(maturity) or maturity <= valuation_date:
        return []
    schedule = _scheduled_payment_dates(month_days, valuation_date, maturity)
    return [date for date in schedule if valuation_date < date <= maturity]


def _scheduled_payment_dates(
    month_days: list[tuple[int, int]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[pd.Timestamp]:
    if not month_days or pd.isna(start_date) or pd.isna(end_date):
        return []
    dates: set[pd.Timestamp] = set()
    for year in range(int(start_date.year) - 1, int(end_date.year) + 2):
        for month, day in month_days:
            last_day = calendar.monthrange(year, month)[1]
            dates.add(pd.Timestamp(year=year, month=month, day=min(day, last_day)))
    return sorted(date for date in dates if start_date <= date <= end_date)


def _bounded_price(price: float) -> float:
    if not math.isfinite(float(price)):
        return 100.0
    return float(np.clip(price, 1.0, 250.0))


__all__ = [
    "DEFAULT_CURVE_SOURCES",
    "DEFAULT_REAL_CURVE_SOURCES",
    "PRICING_DIAGNOSTIC_COLUMNS",
    "apply_model_pricing",
    "build_pricing_scope_diagnostics",
    "load_nominal_yield_curve",
    "load_real_yield_curve",
    "price_period_end_portfolios",
]
