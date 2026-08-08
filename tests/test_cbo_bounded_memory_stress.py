from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import pytest

from tdc_shared import HOLDER_TYPES
from tdcsim_cbo.bounded_output import (
    BOUNDED_KEY_CARDINALITY_CEILING,
    BoundedResourceLimits,
    BoundedScenarioEvidenceSink,
    current_rss_bytes,
)
from tdcsim_cbo.compiler import (
    _DEFAULT_ISSUANCE_MATURITY_DISTRIBUTIONS,
    _DEFAULT_ISSUANCE_SECURITY_SHARES,
)
from tdcsim_cbo.runner import _portfolio_row_budget


GIB = 1024**3
MIB = 1024**2
OPEN04_BASELINE_START = "2026-06-21"
OPEN04_BASELINE_END = "2036-09-30"
OPEN04_BASELINE_OPENING_ROWS = 2_290


def _unconstrained_limits(
    *,
    portfolio_row_budget: int = 750_000,
) -> BoundedResourceLimits:
    return BoundedResourceLimits(
        minimum_available_bytes=0,
        application_abort_rss_bytes=0,
        parent_graceful_stop_rss_bytes=0,
        parent_kill_rss_bytes=0,
        acceptance_peak_rss_bytes=0,
        portfolio_row_budget=portfolio_row_budget,
        key_cardinality_budget=BOUNDED_KEY_CARDINALITY_CEILING,
    )


def _event() -> dict[str, object]:
    return {
        "period_start": "2026-09-20",
        "period_end": "2026-09-30",
        "event_type": "fiscal",
        "leg_type": "tax_and_primary_spending_cash",
        "holder_sector": "Treasury",
        "holder_subsector": "",
        "counterparty_sector": "aggregate_fiscal_counterparties",
        "counterparty_subsector": "",
        "route_holder_sector": "Treasury",
        "route_holder_subsector": "",
        "instrument_type": "fiscal",
        "maturity_bucket": "not_applicable",
        "accounting_basis": "treasury_cash_reserve_deposit",
        "face_stock_change_bil": 0.0,
        "adjusted_principal_change_bil": 0.0,
        "route_face_stock_change_bil": 0.0,
        "route_adjusted_principal_change_bil": 0.0,
        "treasury_cash_change_bil": 1.0,
        "reserve_change_bil": 0.0,
        "deposit_change_bil": 0.0,
        "settlement_scope": "fiscal_cash_settlement",
        "is_intragovernmental": False,
    }


def _open04_compiled_cap_params() -> dict[str, object]:
    def maturities(category: str) -> list[float]:
        return [
            float(row["maturity_years"])
            for row in _DEFAULT_ISSUANCE_MATURITY_DISTRIBUTIONS[category]
        ]

    return {
        "initial_bonds_df": pd.DataFrame(
            index=pd.RangeIndex(OPEN04_BASELINE_OPENING_ROWS)
        ),
        "treasury_issuance_profile": {
            "bills": {"maturities": maturities("bills")},
            "notes": {"maturities": maturities("notes")},
            "bonds": {"maturities": maturities("bonds")},
            "TIPS": {
                "target_percentage": _DEFAULT_ISSUANCE_SECURITY_SHARES["tips"],
                "maturities": maturities("tips"),
            },
            "FRN": {
                "target_percentage": _DEFAULT_ISSUANCE_SECURITY_SHARES["frn"],
                "maturities": maturities("frn"),
            },
            "NonMarketable": {
                "target_percentage": 0.0,
                "maturities": [30.0],
            },
        },
    }


def _open04_compiled_portfolio_cap() -> int:
    return _portfolio_row_budget(
        _open04_compiled_cap_params(),
        start=OPEN04_BASELINE_START,
        end=OPEN04_BASELINE_END,
    )


def test_open04_compiled_portfolio_cap_contract_uses_declared_routes() -> None:
    periods = (
        len(
            pd.date_range(
                OPEN04_BASELINE_START,
                OPEN04_BASELINE_END,
                freq="D",
            )
        )
        - 1
    )
    enabled_supply_rows = sum(
        len(_DEFAULT_ISSUANCE_MATURITY_DISTRIBUTIONS[category])
        for category in ("bills", "notes", "bonds", "tips", "frn")
        if _DEFAULT_ISSUANCE_SECURITY_SHARES[category] > 0.0
    )
    declared_issuance_routes = enabled_supply_rows * (len(HOLDER_TYPES) + 1)
    expected = (
        OPEN04_BASELINE_OPENING_ROWS
        + periods * (declared_issuance_routes + 32)
        + 64
    )

    assert _open04_compiled_portfolio_cap() == expected == 253_872
    assert expected < BoundedResourceLimits().portfolio_row_budget


@pytest.mark.memory_stress
def test_million_event_sink_rss_plateaus_after_warmup(tmp_path: Path) -> None:
    sink = BoundedScenarioEvidenceSink(
        tmp_path / "events",
        limits=_unconstrained_limits(),
        available_reader=lambda: 64 * GIB,
    )
    event = _event()
    milestones = (1_000, 10_000, 100_000, 1_000_000)
    rss: dict[int, int] = {}
    shapes: dict[int, dict[str, int]] = {}
    emitted = 0
    try:
        for milestone in milestones:
            for _ in range(milestone - emitted):
                sink.on_accounting_event(event)
            emitted = milestone
            gc.collect()
            rss[milestone] = current_rss_bytes()
            shapes[milestone] = sink.retained_shape()

        assert sink.event_count == 1_000_000
        assert shapes[1_000] == shapes[10_000] == shapes[100_000] == shapes[1_000_000]
        assert shapes[1_000_000]["current_groups"] == 1
        assert shapes[1_000_000]["failure_ring"] == 64
        assert rss[10_000] > 0
        assert rss[1_000_000] - rss[10_000] < 32 * MIB
        assert not hasattr(sink, "events")
        assert not any(
            isinstance(value, pd.DataFrame) for value in vars(sink).values()
        )
        print(
            "TDCSIM_MEMORY_RECEIPT "
            + json.dumps(
                {
                    "test": "million_event_sink_rss_plateau",
                    "event_count": sink.event_count,
                    "rss_milestones_bytes": {
                        str(milestone): rss[milestone]
                        for milestone in milestones
                    },
                    "rss_growth_10k_to_1m_bytes": (
                        rss[1_000_000] - rss[10_000]
                    ),
                    "current_groups": shapes[1_000_000]["current_groups"],
                    "failure_ring": shapes[1_000_000]["failure_ring"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    finally:
        sink.abort(RuntimeError("memory stress cleanup"))


@pytest.mark.memory_stress
def test_compiled_cap_portfolio_snapshot_stays_below_six_gib(
    tmp_path: Path,
) -> None:
    cap = _open04_compiled_portfolio_cap()
    sink = BoundedScenarioEvidenceSink(
        tmp_path / "portfolio",
        limits=_unconstrained_limits(portfolio_row_budget=cap),
        available_reader=lambda: 64 * GIB,
    )
    portfolio = pd.DataFrame(
        {
            "SecurityType": pd.Series(["Fixed"] * cap, dtype="string"),
            "FaceValue": pd.Series([1.0] * cap, dtype="float64"),
            "AdjustedPrincipal": pd.Series([1.0] * cap, dtype="float64"),
            "HolderType": pd.Series(["Private"] * cap, dtype="string"),
            "HolderSubBucket": pd.Series(
                ["domestic_nonbank_deposit_funded"] * cap,
                dtype="string",
            ),
            "TDCPrincipalHolderType": pd.Series(
                ["Private"] * cap,
                dtype="string",
            ),
            "TDCPrincipalHolderSubBucket": pd.Series(
                ["domestic_nonbank_deposit_funded"] * cap,
                dtype="string",
            ),
            "Status": pd.Series(["Active"] * cap, dtype="string"),
            "OriginalMaturityYears": pd.Series([5.0] * cap, dtype="float64"),
            "MaturityCategory": pd.Series(["notes"] * cap, dtype="string"),
        }
    )
    try:
        sink.capture_portfolio_snapshot(
            portfolio,
            snapshot_date="2026-09-20",
        )
        gc.collect()
        observed_rss = current_rss_bytes()
        assert sink.max_portfolio_rows == cap
        assert sink.retained_shape()["current_groups"] == 4
        assert 0 < observed_rss <= 6 * GIB
        print(
            "TDCSIM_MEMORY_RECEIPT "
            + json.dumps(
                {
                    "test": "compiled_cap_portfolio_snapshot",
                    "portfolio_cap_rows": cap,
                    "max_portfolio_rows": sink.max_portfolio_rows,
                    "observed_rss_bytes": observed_rss,
                    "current_groups": sink.retained_shape()["current_groups"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    finally:
        sink.abort(RuntimeError("portfolio stress cleanup"))
