"""Which marketable Treasury securities a money market fund may acquire.

SEC Rule 2a-7 bars a money market fund from acquiring a security with a remaining
maturity of more than 397 calendar days (17 CFR 270.2a-7(d)(1)(i)). For that limit a
government floating-rate security is deemed to mature in one day (17 CFR
270.2a-7(i)(1)), so Treasury FRNs are eligible at any remaining term. Bills and coupon
securities, nominal or inflation-protected, are eligible only within 397 days of
maturity. The fund-wide weighted-average maturity and life limits apply to a whole
fund's portfolio, which the model does not observe, and are not enforced here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from tdc_shared import PUBLIC_MARKETABLE_SECURITY_TYPES

MMF_MAX_REMAINING_DAYS = 397
# Security types with no maturity limit for a money fund: a government floating-rate
# security is deemed to have a remaining maturity of one day.
MMF_UNLIMITED_TERM_TYPES = frozenset({'FRN'})


class MmfEligibilityError(ValueError):
    """A security's eligibility cannot be decided from the terms given."""


def mmf_may_acquire(security_type: str, maturity_date: Any, on_date: Any) -> bool:
    """True when a money fund may acquire the security on ``on_date``."""

    security_type = str(security_type)
    if security_type not in PUBLIC_MARKETABLE_SECURITY_TYPES:
        raise MmfEligibilityError(f'money-fund eligibility is defined for marketable securities, not {security_type!r}')
    if security_type in MMF_UNLIMITED_TERM_TYPES:
        return True
    maturity = pd.Timestamp(maturity_date) if maturity_date is not None else pd.NaT
    if pd.isna(maturity):
        raise MmfEligibilityError(f'a {security_type} security needs a maturity date to decide money-fund eligibility')
    remaining_days = (maturity.normalize() - pd.Timestamp(on_date).normalize()).days
    return remaining_days <= MMF_MAX_REMAINING_DAYS


def mmf_eligible_mask(frame: pd.DataFrame, on_date: Any) -> pd.Series:
    """Row-wise eligibility of a portfolio frame with ``SecurityType`` and ``MaturityDate``."""

    if frame.empty:
        return pd.Series(dtype=bool, index=frame.index)
    types = frame['SecurityType'].astype(str)
    unknown = sorted(set(types) - PUBLIC_MARKETABLE_SECURITY_TYPES)
    if unknown:
        raise MmfEligibilityError(f'money-fund eligibility is defined for marketable securities, not {unknown}')
    unlimited = types.isin(MMF_UNLIMITED_TERM_TYPES).to_numpy()
    maturities = pd.to_datetime(frame['MaturityDate'], errors='coerce')
    missing = maturities.isna().to_numpy() & ~unlimited
    if missing.any():
        raise MmfEligibilityError(
            f'{int(missing.sum())} non-FRN rows lack a maturity date to decide money-fund eligibility'
        )
    remaining = (maturities.dt.normalize() - pd.Timestamp(on_date).normalize()).dt.days.to_numpy()
    eligible = unlimited | (np.nan_to_num(remaining, nan=np.inf) <= MMF_MAX_REMAINING_DAYS)
    return pd.Series(eligible, index=frame.index)
