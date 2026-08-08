from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import tdcsim_cbo.bounded_output as bounded_output_module
from sim_engine import _handoff_append_holder_stocks
from tdcsim_cbo.bounded_output import (
    BOUNDED_KEY_CARDINALITY_CEILING,
    BoundedEvidenceError,
    BoundedResourceLimits,
    BoundedScenarioEvidenceSink,
    ResourceLimitError,
    _financing_row,
    current_rss_bytes,
)


GIB = 1024**3


def _limits(**overrides: int) -> BoundedResourceLimits:
    values = {
        "minimum_available_bytes": 0,
        "application_abort_rss_bytes": 0,
        "parent_graceful_stop_rss_bytes": 0,
        "parent_kill_rss_bytes": 0,
        "acceptance_peak_rss_bytes": 0,
        "portfolio_row_budget": 10_000,
        "key_cardinality_budget": BOUNDED_KEY_CARDINALITY_CEILING,
    }
    values.update(overrides)
    return BoundedResourceLimits(**values)


def _sink(
    root: Path,
    *,
    limits: BoundedResourceLimits | None = None,
    rss_bytes: int = 1,
    available_bytes: int = 1,
    progress_callback=None,
) -> BoundedScenarioEvidenceSink:
    return BoundedScenarioEvidenceSink(
        root,
        limits=limits or _limits(),
        rss_reader=lambda: rss_bytes,
        available_reader=lambda: available_bytes,
        cpu_reader=lambda: 0.25,
        progress_callback=progress_callback,
    )


def test_default_key_cardinality_budget_is_source_derived_ceiling() -> None:
    assert BOUNDED_KEY_CARDINALITY_CEILING == 4_991
    assert (
        BoundedResourceLimits().key_cardinality_budget
        == BOUNDED_KEY_CARDINALITY_CEILING
    )


def test_darwin_available_memory_matches_vm_stat_page_accounting() -> None:
    assert bounded_output_module._darwin_available_bytes_from_counts(
        page_size=16_384,
        free_count=455_867,
        inactive_count=1_355_685,
        speculative_count=426_532,
    ) == 36_668_768_256


def test_host_available_memory_dispatches_to_darwin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bounded_output_module.platform,
        "system",
        lambda: "Darwin",
    )
    monkeypatch.setattr(
        bounded_output_module,
        "_darwin_available_memory_bytes",
        lambda: 24 * GIB,
    )

    assert bounded_output_module.host_available_memory_bytes() == 24 * GIB


@pytest.mark.parametrize(
    "values",
    [
        {
            "page_size": -1,
            "free_count": 1,
            "inactive_count": 1,
            "speculative_count": 1,
        },
        {
            "page_size": 1,
            "free_count": -1,
            "inactive_count": 1,
            "speculative_count": 1,
        },
        {
            "page_size": True,
            "free_count": 1,
            "inactive_count": 1,
            "speculative_count": 1,
        },
    ],
)
def test_darwin_available_memory_rejects_invalid_counts(
    values: dict[str, int],
) -> None:
    assert (
        bounded_output_module._darwin_available_bytes_from_counts(**values)
        == 0
    )


def _opening_snapshots(sink: BoundedScenarioEvidenceSink) -> None:
    sink.capture_portfolio_snapshot(
        _opening_portfolio(),
        snapshot_date="2026-09-20",
    )


def _opening_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SecurityType": "Fixed",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 100.0,
                "HolderType": "Private",
                "HolderSubBucket": "domestic_nonbank_deposit_funded",
                "TDCPrincipalHolderType": "Private",
                "TDCPrincipalHolderSubBucket": (
                    "domestic_nonbank_deposit_funded"
                ),
                "Status": "Active",
                "OriginalMaturityYears": 5.0,
                "MaturityCategory": "notes",
            }
        ]
    )


def test_progress_callback_records_bounded_admission(tmp_path: Path) -> None:
    progress: list[dict[str, object]] = []
    sink = _sink(
        tmp_path / "progress-admission",
        progress_callback=lambda value: progress.append(dict(value)),
    )
    _opening_snapshots(sink)

    sink.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )

    assert progress == [
        {
            "progress_state": "admission",
            "admission_date": "2026-09-20",
            "last_completed_period": None,
            "period_count": 0,
            "event_count": 0,
            "event_root_sha256": hashlib.sha256().hexdigest(),
            "peak_rss_bytes": 1,
            "failure_invariant": None,
            "failure_key": None,
            "last_events": [],
        }
    ]
    sink.abort(RuntimeError("test cleanup"))


def _journal_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
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
    event.update(overrides)
    return event


def test_retained_object_shape_does_not_scale_with_event_count(tmp_path: Path) -> None:
    sink = _sink(tmp_path / "shape")
    event = _journal_event()
    try:
        for _ in range(10_000):
            sink.on_accounting_event(event)
        shape_at_10k = sink.retained_shape()
        for _ in range(40_000):
            sink.on_accounting_event(event)
        shape_at_50k = sink.retained_shape()

        assert sink.event_count == 50_000
        assert shape_at_10k == shape_at_50k
        assert shape_at_50k["current_groups"] == 1
        assert shape_at_50k["failure_ring"] == 64
        assert not hasattr(sink, "events")
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_unknown_or_zero_accounting_event_fails_before_admission(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "taxonomy")
    try:
        with pytest.raises(BoundedEvidenceError, match="unknown accounting taxonomy"):
            sink.on_accounting_event(_journal_event(instrument_type="unknown"))
        with pytest.raises(BoundedEvidenceError, match="all-zero"):
            sink.on_accounting_event(
                _journal_event(treasury_cash_change_bil=0.0)
            )
        assert sink.event_count == 0
    finally:
        sink.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"leg_type": "unknown_leg"},
        {"accounting_basis": "unknown_basis"},
        {"settlement_scope": "unknown_scope"},
        {"holder_sector": "unknown_holder"},
        {"holder_subsector": "unknown_subsector"},
        {"counterparty_sector": "unknown_counterparty"},
        {"route_holder_sector": "unknown_route"},
    ],
)
def test_unknown_accounting_key_fails_before_reduction(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    sink = _sink(tmp_path / next(iter(overrides)))
    try:
        with pytest.raises(BoundedEvidenceError, match="unknown"):
            sink.on_accounting_event(_journal_event(**overrides))
        assert sink.event_count == 0
        assert sink.retained_shape()["current_groups"] == 0
    finally:
        sink.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize(
    "column",
    ["event_type", "leg_type", "accounting_basis"],
)
def test_blank_required_accounting_labels_fail_before_reduction(
    tmp_path: Path,
    column: str,
) -> None:
    sink = _sink(tmp_path / column)
    try:
        with pytest.raises(BoundedEvidenceError, match=f"blank {column}"):
            sink.on_accounting_event(_journal_event(**{column: "  "}))
        assert sink.event_count == 0
    finally:
        sink.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "malformed"])
def test_malformed_accounting_numeric_fails_before_reduction(
    tmp_path: Path,
    value: object,
) -> None:
    sink = _sink(tmp_path / "numeric")
    try:
        with pytest.raises(BoundedEvidenceError, match="treasury_cash_change_bil"):
            sink.on_accounting_event(
                _journal_event(treasury_cash_change_bil=value)
            )
        assert sink.event_count == 0
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_missing_accounting_numeric_fails_before_reduction(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "missing-numeric")
    event = _journal_event()
    del event["reserve_change_bil"]
    try:
        with pytest.raises(
            BoundedEvidenceError,
            match="missing reserve_change_bil",
        ):
            sink.on_accounting_event(event)
        assert sink.event_count == 0
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_duplicate_accounting_sequence_fails_before_second_reduction(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "sequence")
    try:
        sink.on_accounting_event(_journal_event(), event_seq=1)
        with pytest.raises(BoundedEvidenceError, match="sequence discontinuity"):
            sink.on_accounting_event(_journal_event(), event_seq=1)
        assert sink.event_count == 1
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_key_cardinality_budget_fails_on_the_first_excess_key(
    tmp_path: Path,
) -> None:
    sink = _sink(
        tmp_path / "keys",
        limits=_limits(key_cardinality_budget=1),
    )
    try:
        sink.on_accounting_event(_journal_event())
        with pytest.raises(ResourceLimitError, match="key-cardinality"):
            sink.on_accounting_event(
                _journal_event(
                    event_type="transfer",
                    leg_type="reserve_transfer_and_money_minting",
                    counterparty_sector="other_named_transfer_source",
                    instrument_type="cash",
                    accounting_basis="treasury_cash_reserve_deposit",
                    settlement_scope="configured_other_transfer",
                )
            )
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_nonmarketable_issuance_rejects_undeclared_holder_before_reduction(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "nonmarketable-holder")
    row = {
        "period_start": "2026-09-20",
        "period_end": "2026-09-30",
        "holder_sector": "UnregisteredTrust",
        "holder_subsector": "",
        "instrument_type": "NonMarketable",
        "maturity_bucket": "bonds",
        "weighted_original_term_years": 30.0,
        "face_issued_bil": 1.0,
        "cash_proceeds_bil": 0.0,
        "discount_or_premium_bil": 1.0,
        "coupon_rate_decimal": 0.0,
        "issuance_leg": "ordinary_issuance",
        "reference_rate_decimal": None,
        "spread_bps": None,
        "issue_yield_decimal": None,
    }
    try:
        with pytest.raises(BoundedEvidenceError, match="holder pair"):
            sink["tdcsim_period_issuance_flows"].append(row)
        assert sink.retained_shape()["current_groups"] == 0

        row["holder_sector"] = "TrustFunds"
        sink["tdcsim_period_issuance_flows"].append(row)
        assert sink.retained_shape()["current_groups"] == 1
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_portfolio_budget_accepts_exact_cap_and_rejects_next_row(
    tmp_path: Path,
) -> None:
    sink = _sink(
        tmp_path / "portfolio",
        limits=_limits(portfolio_row_budget=100),
    )
    try:
        sink.check_portfolio_rows(90, pending=10)
        with pytest.raises(ResourceLimitError, match="portfolio-row"):
            sink.check_portfolio_rows(100, pending=1)
    finally:
        sink.abort(RuntimeError("test cleanup"))


def test_current_process_rss_probe_returns_positive_bytes() -> None:
    assert current_rss_bytes() > 0


def test_live_portfolio_snapshot_streams_active_holder_and_route_stocks(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "live-snapshot")
    portfolio = pd.DataFrame(
        [
            {
                "SecurityType": "TIPS",
                "FaceValue": 100.0,
                "AdjustedPrincipal": 112.5,
                "HolderType": "Private",
                "HolderSubBucket": "domestic_nonbank_deposit_funded",
                "TDCPrincipalHolderType": "Banks",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Active",
                "OriginalMaturityYears": 20.0,
                "MaturityCategory": "tips",
            },
            {
                "SecurityType": "Fixed",
                "FaceValue": 900.0,
                "AdjustedPrincipal": 900.0,
                "HolderType": "Foreign",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "Foreign",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Matured",
                "OriginalMaturityYears": 5.0,
                "MaturityCategory": "notes",
            },
        ]
    )

    sink.capture_portfolio_snapshot(
        portfolio,
        snapshot_date="2026-09-20",
    )
    holder_rows = sink._rows("tdcsim_holder_stocks")
    route_rows = sink._rows("tdcsim_tdc_principal_route_stocks")

    assert len(holder_rows) == 2
    assert {row["debt_scope"] for row in holder_rows} == {
        "all_active_treasury",
        "controlled_public_marketable",
    }
    assert {row["holder_sector"] for row in holder_rows} == {"Private"}
    assert {row["debt_held_bil"] for row in holder_rows} == {112.5}
    assert {row["face_stock_bil"] for row in holder_rows} == {100.0}
    assert {row["adjusted_principal_stock_bil"] for row in holder_rows} == {
        112.5
    }
    assert {row["maturity_bucket"] for row in holder_rows} == {"bonds"}
    assert {row["route_holder_sector"] for row in route_rows} == {"Banks"}
    assert {row["route_holder_subsector"] for row in route_rows} == {""}

    sink.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )
    assert sink.max_portfolio_rows == 2
    sink.abort(RuntimeError("test cleanup"))


def test_live_portfolio_snapshot_rejects_duplicate_date_and_unknown_taxonomy(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "snapshot-rejections")
    _opening_snapshots(sink)
    with pytest.raises(BoundedEvidenceError, match="not consumed"):
        _opening_snapshots(sink)
    sink.abort(RuntimeError("test cleanup"))

    unknown = _sink(tmp_path / "snapshot-unknown")
    portfolio = pd.DataFrame(
        [
            {
                "SecurityType": "UnknownDebt",
                "FaceValue": 1.0,
                "HolderType": "Private",
                "HolderSubBucket": "",
                "Status": "Active",
            }
        ]
    )
    with pytest.raises(BoundedEvidenceError, match="unknown security type"):
        unknown.capture_portfolio_snapshot(
            portfolio,
            snapshot_date="2026-09-20",
        )
    unknown.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (
            "HolderSubBucket",
            "undeclared_private_bucket",
            "invalid Private holder subbucket",
        ),
        (
            "TDCPrincipalHolderSubBucket",
            "undeclared_private_route",
            "invalid Private TDC principal route subbucket",
        ),
    ],
)
def test_live_portfolio_snapshot_rejects_undeclared_private_subbuckets(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    sink = _sink(tmp_path / column)
    portfolio = _opening_portfolio()
    portfolio.loc[0, column] = value
    with pytest.raises(BoundedEvidenceError, match=message):
        sink.capture_portfolio_snapshot(
            portfolio,
            snapshot_date="2026-09-20",
        )
    assert sink.retained_shape()["current_groups"] == 0
    sink.abort(RuntimeError("test cleanup"))


def test_live_portfolio_snapshot_rejects_free_subbucket_on_nonprivate_route(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "nonprivate-route")
    portfolio = _opening_portfolio()
    portfolio.loc[0, "TDCPrincipalHolderType"] = "Banks"
    portfolio.loc[0, "TDCPrincipalHolderSubBucket"] = "free_form_axis"
    with pytest.raises(BoundedEvidenceError, match="nonblank TDC principal route"):
        sink.capture_portfolio_snapshot(
            portfolio,
            snapshot_date="2026-09-20",
        )
    assert sink.retained_shape()["current_groups"] == 0
    sink.abort(RuntimeError("test cleanup"))


def test_live_portfolio_snapshot_matches_legacy_oracle_matrix(
    tmp_path: Path,
) -> None:
    portfolio = pd.DataFrame(
        [
            {
                "SecurityType": "Fixed",
                "FaceValue": 11.0,
                "AdjustedPrincipal": 11.0,
                "HolderType": "Banks",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "Banks",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Active",
                "OriginalMaturityYears": 0.5,
                "MaturityCategory": "bills",
            },
            {
                "SecurityType": "TIPS",
                "FaceValue": 13.0,
                "AdjustedPrincipal": pd.NA,
                "HolderType": "Private",
                "HolderSubBucket": "domestic_nonbank_deposit_funded",
                "TDCPrincipalHolderType": "Private",
                "TDCPrincipalHolderSubBucket": "mmf_cash_fund_route",
                "Status": "Active",
                "OriginalMaturityYears": 20.0,
                "MaturityCategory": "tips",
            },
            {
                "SecurityType": "FRN",
                "FaceValue": 17.0,
                "AdjustedPrincipal": 17.0,
                "HolderType": "CB",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "Private",
                "TDCPrincipalHolderSubBucket": "mmf_cash_fund_route",
                "Status": "Active",
                "OriginalMaturityYears": 2.0,
                "MaturityCategory": "frn",
            },
            {
                "SecurityType": "NonMarketable",
                "FaceValue": 19.0,
                "AdjustedPrincipal": 19.0,
                "HolderType": "TrustFunds",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "TrustFunds",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Active",
                "OriginalMaturityYears": 30.0,
                "MaturityCategory": "nonmarketable",
            },
            {
                "SecurityType": "Fixed",
                "FaceValue": 23.0,
                "AdjustedPrincipal": 23.0,
                "HolderType": "Private",
                "HolderSubBucket": "domestic_nonbank_deposit_funded",
                "TDCPrincipalHolderType": "",
                "TDCPrincipalHolderSubBucket": "mmf_cash_fund_route",
                "Status": "Active",
                "OriginalMaturityYears": 5.0,
                "MaturityCategory": "notes",
            },
            {
                "SecurityType": "Fixed",
                "FaceValue": 29.0,
                "AdjustedPrincipal": 29.0,
                "HolderType": "Foreign",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "Foreign",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Matured",
                "OriginalMaturityYears": 5.0,
                "MaturityCategory": "notes",
            },
            {
                "SecurityType": "Fixed",
                "FaceValue": 0.0,
                "AdjustedPrincipal": 0.0,
                "HolderType": "Foreign",
                "HolderSubBucket": "",
                "TDCPrincipalHolderType": "Foreign",
                "TDCPrincipalHolderSubBucket": "",
                "Status": "Active",
                "OriginalMaturityYears": 5.0,
                "MaturityCategory": "notes",
            },
        ]
    )
    legacy = {
        "tdcsim_holder_stocks": [],
        "tdcsim_tdc_principal_route_stocks": [],
    }
    _handoff_append_holder_stocks(
        legacy,
        portfolio.loc[portfolio["Status"].eq("Active")].copy(),
        "2026-09-20",
    )
    sink = _sink(tmp_path / "oracle")
    sink.capture_portfolio_snapshot(
        portfolio,
        snapshot_date="2026-09-20",
    )

    def ordered(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda row: tuple(
                (key, str(value)) for key, value in sorted(row.items())
            ),
        )

    assert ordered(sink._rows("tdcsim_holder_stocks")) == ordered(
        legacy["tdcsim_holder_stocks"]
    )
    assert ordered(
        sink._rows("tdcsim_tdc_principal_route_stocks")
    ) == ordered(legacy["tdcsim_tdc_principal_route_stocks"])
    sink.abort(RuntimeError("test cleanup"))


@pytest.mark.parametrize("literal", ["nan", "<NA>", "None"])
def test_live_portfolio_snapshot_rejects_literal_missing_category_sentinels(
    tmp_path: Path,
    literal: str,
) -> None:
    sink = _sink(tmp_path / "sentinel")
    portfolio = _opening_portfolio()
    portfolio.loc[0, "MaturityCategory"] = literal
    with pytest.raises(BoundedEvidenceError, match="noncanonical maturity"):
        sink.capture_portfolio_snapshot(
            portfolio,
            snapshot_date="2026-09-20",
        )
    sink.abort(RuntimeError("test cleanup"))


def test_live_snapshot_key_cap_includes_retained_opening_groups(
    tmp_path: Path,
) -> None:
    rejected = _sink(
        tmp_path / "rejected-live-keys",
        limits=_limits(key_cardinality_budget=7),
    )
    _opening_snapshots(rejected)
    rejected.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )
    with pytest.raises(ResourceLimitError, match="key-cardinality"):
        rejected.capture_portfolio_snapshot(
            _opening_portfolio(),
            snapshot_date="2026-09-30",
        )
    assert rejected._max_key_cardinality == 8
    rejected.abort(RuntimeError("test cleanup"))

    admitted = _sink(
        tmp_path / "admitted-live-keys",
        limits=_limits(key_cardinality_budget=8),
    )
    _opening_snapshots(admitted)
    admitted.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )
    admitted.capture_portfolio_snapshot(
        _opening_portfolio(),
        snapshot_date="2026-09-30",
    )
    assert admitted._max_key_cardinality == 8
    assert admitted.retained_shape()["current_groups"] == 8
    admitted.abort(RuntimeError("test cleanup"))


def test_admission_requires_at_least_twenty_gib(tmp_path: Path) -> None:
    rejected = _sink(
        tmp_path / "rejected",
        limits=_limits(minimum_available_bytes=20 * GIB),
        available_bytes=20 * GIB - 1,
    )
    _opening_snapshots(rejected)
    with pytest.raises(ResourceLimitError, match="admission"):
        rejected.begin_run(
            opening_date="2026-09-20",
            opening_result={},
        )
    rejected.abort(RuntimeError("test cleanup"))

    admitted = _sink(
        tmp_path / "admitted",
        limits=_limits(minimum_available_bytes=20 * GIB),
        available_bytes=20 * GIB,
    )
    _opening_snapshots(admitted)
    admitted.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )
    admitted.abort(RuntimeError("test cleanup"))


def test_application_abort_triggers_at_exactly_eight_gib(tmp_path: Path) -> None:
    sink = _sink(
        tmp_path / "abort",
        limits=_limits(
            application_abort_rss_bytes=8 * GIB,
            parent_graceful_stop_rss_bytes=10 * GIB,
            parent_kill_rss_bytes=12 * GIB,
        ),
        rss_bytes=8 * GIB,
    )
    _opening_snapshots(sink)
    with pytest.raises(ResourceLimitError, match="RSS hard stop"):
        sink.begin_run(
            opening_date="2026-09-20",
            opening_result={},
        )
    sink.abort(RuntimeError("test cleanup"))


def test_acceptance_fails_above_six_gib_even_below_abort(tmp_path: Path) -> None:
    sink = _sink(
        tmp_path / "acceptance",
        limits=_limits(
            acceptance_peak_rss_bytes=6 * GIB,
            application_abort_rss_bytes=8 * GIB,
            parent_graceful_stop_rss_bytes=10 * GIB,
            parent_kill_rss_bytes=12 * GIB,
        ),
        rss_bytes=6 * GIB + 1,
    )
    _opening_snapshots(sink)
    sink.begin_run(
        opening_date="2026-09-20",
        opening_result={},
    )
    with pytest.raises(ResourceLimitError, match="acceptance RSS"):
        sink.finalize()
    sink.abort(RuntimeError("test cleanup"))


def test_financing_reducer_derives_total_and_rejects_missing_scalar() -> None:
    closing = {
        "InterestOutlay_Period": 1.0,
        "IssueDiscountCost_Period": 2.0,
        "NonMarketableInterestCapitalized_Period": 3.0,
        "TIPSInflationAccretion_Period": 4.0,
        "FinancingCost_Period": 10.0,
    }
    row = _financing_row(closing, "2026-09-20", "2026-09-30")

    assert row["modeled_financing_cost_bil"] == pytest.approx(10.0)
    assert row["component_identity_error_bil"] == pytest.approx(0.0)
    assert _financing_row(
        {**closing, "FinancingCost_Period": 11.0},
        "2026-09-20",
        "2026-09-30",
    )["component_identity_error_bil"] == pytest.approx(1.0)
    with pytest.raises(BoundedEvidenceError, match="InterestOutlay_Period"):
        _financing_row(
            {key: value for key, value in closing.items() if key != "InterestOutlay_Period"},
            "2026-09-20",
            "2026-09-30",
        )


def test_annual_reducer_carries_across_full_and_partial_fiscal_years(
    tmp_path: Path,
) -> None:
    sink = _sink(tmp_path / "multi-fy")

    def add_period(
        period_start: str,
        period_end: str,
        *,
        tdc: float,
        overlap: float,
        financing_cost: float,
    ) -> None:
        metric_rows = (
            [
                {
                    "outstanding_controlled_wam_years": 6.0,
                    "outstanding_controlled_bill_share": 0.2,
                    "outstanding_controlled_short_maturity_share": 0.3,
                }
            ]
            if period_end.endswith("-09-30")
            else []
        )
        sink._update_annual(
            period_start=period_start,
            period_end=period_end,
            closing_result={},
            issuance_rows=[],
            scenario_metric_rows=metric_rows,
            tdc_summary=[
                {
                    "tdc_change_bil": tdc,
                    "overlap_cashflow_bil": overlap,
                    "tdc_change_ex_overlap_bil": tdc - overlap,
                }
            ],
            financing={
                "interest_outlay_bil": financing_cost,
                "issue_discount_cost_bil": 0.0,
                "nonmarketable_interest_capitalized_bil": 0.0,
                "tips_inflation_accretion_bil": 0.0,
                "modeled_financing_cost_bil": financing_cost,
            },
        )

    try:
        add_period(
            "2026-09-20",
            "2026-09-30",
            tdc=1.0,
            overlap=0.25,
            financing_cost=2.0,
        )
        add_period(
            "2026-09-30",
            "2027-09-30",
            tdc=2.0,
            overlap=0.5,
            financing_cost=3.0,
        )
        add_period(
            "2027-09-30",
            "2028-03-31",
            tdc=-0.5,
            overlap=0.1,
            financing_cost=4.0,
        )

        rows = sink._annual_rows()
        assert [row["period_label"] for row in rows] == [
            "FY2026_PARTIAL_OPENING",
            "FY2027",
            "FY2028_PARTIAL_CLOSING",
        ]
        assert [row["is_partial_period"] for row in rows] == [
            True,
            False,
            True,
        ]
        assert [row["coverage_days"] for row in rows] == [10, 365, 183]
        assert [row["expected_coverage_days"] for row in rows] == [
            365,
            365,
            366,
        ]
        assert [row["cumulative_tdc_change_bil"] for row in rows] == pytest.approx(
            [1.0, 3.0, 2.5]
        )
        assert [
            row["cumulative_overlap_cashflow_bil"] for row in rows
        ] == pytest.approx([0.25, 0.75, 0.85])
        assert [
            row["cumulative_tdc_change_ex_overlap_bil"] for row in rows
        ] == pytest.approx([0.75, 2.25, 1.65])
        assert [
            row["cumulative_modeled_financing_cost_bil"] for row in rows
        ] == pytest.approx([2.0, 5.0, 9.0])
    finally:
        sink.abort(RuntimeError("test cleanup"))
