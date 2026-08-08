"""Bounded-memory evidence sink for production CBO scenario runs.

The simulator's handoff helpers deliberately accept a mapping of appendable tables.  This
module implements that small interface without retaining security-event rows: accounting
events are validated, committed to deterministic hashes, and reduced by a fixed taxonomy;
other flow tables are reduced within the current period; independent opening/closing stock
snapshots are closed before the next period can begin.

The legacy dictionary-of-lists remains useful as a short-fixture oracle.  It is not a
production fallback.
"""

from __future__ import annotations

import csv
import ctypes
import gzip
import hashlib
import io
import json
import math
import os
import platform
import struct
import time
from collections import Counter, deque
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from tdc_shared import (
    HOLDER_TYPES,
    INTRAGOV_HOLDERS,
    PREFERENCE_CATEGORIES,
    PRIVATE_SUBBUCKETS,
    PUBLIC_MARKETABLE_SECURITY_TYPES,
    SECURITY_TYPES,
    TGA_FLOOR_TOLERANCE,
)

from ._json import sha256_file
from .output import (
    HANDOFF_TABLE_COLUMNS,
    TDC_COMPONENT_SPECS,
    TDC_IDENTITY_COLUMNS,
    TDC_OVERLAP_COLUMNS,
    _accounting_closure_handoff_tables,
    _route_stock_closure_handoff_tables,
    _tdc_handoff_tables,
)


EVENT_SCHEMA_VERSION = "tdcsim_transient_accounting_event_v1"
EVIDENCE_PROFILE = "bounded_period_closure_v1"
VERIFICATION_GRADE = "bounded_replay_v1"
AGGREGATION_CLOCK_ID = "federal_fiscal_year_period_end_v1"

STOCK_TOLERANCE_BIL = 1e-7
CASH_TOLERANCE_BIL = 1e-6
TGA_MINIMUM_BIL = -1e-6

_GIB = 1024**3
BOUNDED_KEY_CARDINALITY_CEILING = 4_991

_PHYSICAL_HOLDER_PAIRS = frozenset(
    (holder, subbucket)
    for holder in HOLDER_TYPES
    for subbucket in (
        tuple(PRIVATE_SUBBUCKETS) if holder == "Private" else ("",)
    )
)
_AGGREGATE_HOLDERS = frozenset(
    {
        "aggregate_auction_holders",
        "aggregate_buyback_holders",
        "aggregate_public_holders",
        "aggregate_secondary_buyers",
        "aggregate_secondary_counterparties",
        "aggregate_secondary_sellers",
    }
)
_JOURNAL_HOLDER_PAIRS = frozenset(
    {
        *_PHYSICAL_HOLDER_PAIRS,
        ("", ""),
        ("Treasury", ""),
        *((holder, "") for holder in _AGGREGATE_HOLDERS),
    }
)
_COUNTERPARTY_PAIRS = frozenset(
    {
        *_PHYSICAL_HOLDER_PAIRS,
        ("", ""),
        ("Treasury", ""),
        ("secondary_market", ""),
        ("aggregate_fiscal_counterparties", ""),
        ("cash_reconciliation_source", ""),
        ("other_named_transfer_source", ""),
    }
)
_PHYSICAL_INSTRUMENT_MATURITY_PAIRS = frozenset(
    (instrument, maturity)
    for instrument in SECURITY_TYPES
    for maturity in ("bills", "notes", "bonds")
)
_JOURNAL_INSTRUMENT_MATURITY_PAIRS = frozenset(
    {
        *_PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        ("cash", "not_applicable"),
        ("fiscal", "not_applicable"),
        ("all_treasury", "all"),
    }
)
_PAYMENT_ARCHETYPES = frozenset(
    {
        ("bill_discount", "budget_accrual", False),
        ("tips_indexation", "memo_decomposition", False),
        ("tips_deflation_floor_topup", "stock_to_cash_bridge", False),
        ("frn_interest", "cash", True),
        ("tips_coupon", "cash", True),
        ("fixed_coupon", "cash", True),
    }
)
_JOURNAL_CORE_ARCHETYPES = frozenset(
    {
        *(
            (
                "deflation_floor"
                if leg == "tips_deflation_floor_topup"
                else "coupon_or_payment",
                leg,
                basis,
                scope,
            )
            for leg, basis, _ in _PAYMENT_ARCHETYPES
            for scope in (
                "treasury_cash_settlement",
                "consolidated_noncash_intragovernmental",
            )
        ),
        *(
            (event, "principal_stock_removal", basis, scope)
            for event in ("buyback", "maturity_redemption")
            for basis in ("face_stock", "tips_adjusted_principal_stock")
            for scope in (
                "treasury_cash_settlement",
                "consolidated_noncash_intragovernmental",
            )
        ),
        *(
            (event, leg, basis, scope)
            for event, leg, scope in (
                (
                    "issuance",
                    "auction_liability_and_cash",
                    "auction_cash_settlement",
                ),
                (
                    "intragovernmental_issuance",
                    "consolidated_noncash_liability_credit",
                    "consolidated_noncash_intragovernmental",
                ),
            )
            for basis in ("face_stock", "tips_face_and_adjusted_principal")
        ),
        *(
            (event, leg, basis, scope)
            for event, scope in (
                ("fed_secondary_transfer", "fed_secondary_purchase"),
                ("fed_secondary_transfer", "fed_secondary_sale"),
                ("preference_secondary_transfer", "generic_secondary_trading"),
            )
            for leg, basis in (
                ("transfer_in", "beneficial_holder_stock_transfer"),
                ("transfer_out", "beneficial_holder_stock_transfer"),
                ("route_transfer_in", "tdc_principal_route_stock_transfer"),
                ("route_transfer_out", "tdc_principal_route_stock_transfer"),
            )
        ),
        (
            "intragovernmental",
            "nonmarketable_interest_capitalization",
            "face_stock_noncash_capitalization",
            "consolidated_noncash_intragovernmental",
        ),
        (
            "indexation",
            "tips_adjusted_principal_indexation",
            "adjusted_principal_stock",
            "noncash_stock_remeasurement",
        ),
        (
            "indexation",
            "tips_exact_event_date_indexation_correction",
            "adjusted_principal_stock",
            "noncash_stock_remeasurement",
        ),
        (
            "fiscal",
            "tax_and_primary_spending_cash",
            "treasury_cash_reserve_deposit",
            "fiscal_cash_settlement",
        ),
        (
            "settlement",
            "debt_service_counterparty_settlement",
            "reserve_and_deposit_settlement",
            "aggregate_holder_cash_settlement",
        ),
        (
            "transfer",
            "central_bank_remittance",
            "treasury_cash",
            "central_bank_remittance_settlement",
        ),
        (
            "settlement",
            "buyback_counterparty_settlement",
            "reserve_and_deposit_settlement",
            "explicit_retirement_at_par",
        ),
        (
            "settlement",
            "auction_counterparty_settlement",
            "reserve_and_deposit_settlement",
            "auction_cash_settlement",
        ),
        (
            "cash_reconciliation",
            "explicit_cbo_cash_reconciliation_residual",
            "treasury_cash_named_nonfunding_adjustment",
            "named_reconciliation_leg_requires_hard_boundary_validation",
        ),
        (
            "settlement",
            "fed_secondary_purchase_settlement",
            "reserve_and_deposit_settlement",
            "fed_secondary_purchase",
        ),
        (
            "settlement",
            "fed_secondary_sale_settlement",
            "reserve_and_deposit_settlement",
            "fed_secondary_sale",
        ),
        (
            "settlement",
            "secondary_trade_settlement",
            "treasury_cash_reserve_deposit",
            "generic_secondary_trading",
        ),
        (
            "transfer",
            "reserve_transfer_and_money_minting",
            "treasury_cash_reserve_deposit",
            "configured_other_transfer",
        ),
    }
)

# One source of truth for every categorical key retained by the bounded reducers.
BOUNDED_HANDOFF_TAXONOMY = {
    "tdcsim_accounting_journal": {
        "core_archetypes": _JOURNAL_CORE_ARCHETYPES,
        "holder_pairs": _JOURNAL_HOLDER_PAIRS,
        "counterparty_pairs": _COUNTERPARTY_PAIRS,
        "route_holder_pairs": _JOURNAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _JOURNAL_INSTRUMENT_MATURITY_PAIRS,
    },
    "tdcsim_period_issuance_flows": {
        "holder_pairs": _PHYSICAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        "issuance_legs": frozenset(
            {
                "ordinary_issuance",
                "cbo_reference_face_issuance",
                "tga_floor_cash_financing",
            }
        ),
    },
    "tdcsim_period_principal_flows": {
        "holder_pairs": _PHYSICAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        "redemption_types": frozenset(
            {"scheduled_maturity", "explicit_retirement_at_par"}
        ),
        "recipient_pairs": _PHYSICAL_HOLDER_PAIRS,
        "recipient_bases": frozenset(
            {
                "current_cb_beneficial_holder_otherwise_recorded_tdc_principal_route"
            }
        ),
    },
    "tdcsim_period_payment_flows": {
        "holder_pairs": _PHYSICAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        "payment_archetypes": _PAYMENT_ARCHETYPES,
    },
    "tdcsim_holder_stocks": {
        "holder_pairs": _PHYSICAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        "valuation_bases": frozenset({"face", "tips_adjusted_principal"}),
        "debt_scopes": frozenset(
            {"all_active_treasury", "controlled_public_marketable"}
        ),
        "allocation_methods": frozenset({"end_of_period_stock_snapshot"}),
    },
    "tdcsim_tdc_principal_route_stocks": {
        "holder_pairs": _PHYSICAL_HOLDER_PAIRS,
        "instrument_maturity_pairs": _PHYSICAL_INSTRUMENT_MATURITY_PAIRS,
        "valuation_bases": frozenset({"face", "tips_adjusted_principal"}),
        "debt_scopes": frozenset(
            {"all_active_treasury", "controlled_public_marketable"}
        ),
        "allocation_methods": frozenset(
            {"end_of_period_tdc_principal_route_stock_snapshot"}
        ),
        "route_stock_bases": frozenset({"tdc_principal_settlement_route"}),
    },
    "tdcsim_debt_target_bridge": {
        "categorical_tuples": frozenset(
            {
                (
                    "cbo_public_debt_target",
                    "excluded_from_public_debt",
                    "included_in_public_debt",
                    "explicit_bridge_not_marketable_issuance",
                ),
                (
                    "cbo_debt_reference_plus_tga_floor_financing_v1",
                    "excluded_from_public_debt",
                    "included_in_public_debt",
                    "explicit_bridge_not_marketable_issuance",
                ),
            }
        ),
    },
    "tdcsim_scenario_metrics": {"short_maturity_cutoff_years": 1.0},
}

_ALLOWED_EVENT_TYPES = frozenset(
    archetype[0] for archetype in _JOURNAL_CORE_ARCHETYPES
)
_ALLOWED_HOLDERS = frozenset(pair[0] for pair in _JOURNAL_HOLDER_PAIRS)
_ALLOWED_INSTRUMENTS = frozenset(
    pair[0] for pair in _JOURNAL_INSTRUMENT_MATURITY_PAIRS
)
_ALLOWED_MATURITY_BUCKETS = frozenset(
    pair[1] for pair in _JOURNAL_INSTRUMENT_MATURITY_PAIRS
)

_JOURNAL_NUMERIC = (
    "face_stock_change_bil",
    "adjusted_principal_change_bil",
    "route_face_stock_change_bil",
    "route_adjusted_principal_change_bil",
    "treasury_cash_change_bil",
    "reserve_change_bil",
    "deposit_change_bil",
)
_JOURNAL_KEYS = (
    "period_start",
    "period_end",
    "event_type",
    "leg_type",
    "holder_sector",
    "holder_subsector",
    "counterparty_sector",
    "counterparty_subsector",
    "route_holder_sector",
    "route_holder_subsector",
    "instrument_type",
    "maturity_bucket",
    "accounting_basis",
    "settlement_scope",
    "is_intragovernmental",
)
_ISSUANCE_KEYS = (
    "period_start",
    "period_end",
    "holder_sector",
    "holder_subsector",
    "issuance_leg",
    "instrument_type",
    "maturity_bucket",
)
_ISSUANCE_SUMS = (
    "face_issued_bil",
    "cash_proceeds_bil",
    "discount_or_premium_bil",
)
_ISSUANCE_WEIGHTED = (
    "weighted_original_term_years",
    "coupon_rate_decimal",
    "reference_rate_decimal",
    "spread_bps",
    "issue_yield_decimal",
)
_PRINCIPAL_KEYS = (
    "period_start",
    "period_end",
    "holder_sector",
    "holder_subsector",
    "instrument_type",
    "maturity_bucket",
    "redemption_type",
    "tdc_principal_recipient_sector",
    "tdc_principal_recipient_subsector",
    "tdc_principal_recipient_basis",
)
_PRINCIPAL_SUMS = (
    "face_redeemed_bil",
    "principal_redeemed_bil",
    "cash_paid_bil",
    "adjusted_principal_stock_removed_bil",
    "tdc_principal_cash_paid_to_du_bil",
    "tdc_principal_redeemed_to_du_bil",
    "tdc_principal_cash_paid_to_du_domestic_nonbank_bil",
    "tdc_principal_redeemed_to_du_domestic_nonbank_bil",
    "tdc_principal_cash_paid_to_du_mmf_bil",
    "tdc_principal_redeemed_to_du_mmf_bil",
    "tdc_principal_cash_paid_to_du_mmf_plumbing_bil",
    "tdc_principal_redeemed_to_du_mmf_plumbing_bil",
)
_PAYMENT_KEYS = (
    "period_start",
    "period_end",
    "holder_sector",
    "holder_subsector",
    "instrument_type",
    "maturity_bucket",
    "payment_type",
    "accounting_basis",
    "is_additive_to_cash_total",
)
_HOLDER_STOCK_KEYS = tuple(
    column
    for column in HANDOFF_TABLE_COLUMNS["tdcsim_holder_stocks"]
    if column
    not in {"debt_held_bil", "face_stock_bil", "adjusted_principal_stock_bil"}
)
_ROUTE_STOCK_KEYS = tuple(
    column
    for column in HANDOFF_TABLE_COLUMNS["tdcsim_tdc_principal_route_stocks"]
    if column
    not in {
        "route_debt_held_bil",
        "route_face_stock_bil",
        "route_adjusted_principal_stock_bil",
    }
)
_DEBT_BRIDGE_KEYS = tuple(
    column
    for column in HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"]
    if not column.endswith("_bil")
)
_DEBT_BRIDGE_SUMS = tuple(
    column
    for column in HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"]
    if column.endswith("_bil")
)
_SCENARIO_METRIC_KEYS = ("date", "short_maturity_cutoff_years")
_SCENARIO_METRIC_VALUES = tuple(
    column
    for column in HANDOFF_TABLE_COLUMNS["tdcsim_scenario_metrics"]
    if column not in _SCENARIO_METRIC_KEYS
)

LEDGER_COLUMNS = [
    *_JOURNAL_KEYS,
    *_JOURNAL_NUMERIC,
    "event_count",
]
ISSUANCE_COLUMNS = [
    *_ISSUANCE_KEYS,
    *_ISSUANCE_SUMS,
    *_ISSUANCE_WEIGHTED,
    "flow_count",
]
PRINCIPAL_COLUMNS = [
    *_PRINCIPAL_KEYS,
    *_PRINCIPAL_SUMS,
    "flow_count",
]
PAYMENT_COLUMNS = [
    *_PAYMENT_KEYS,
    "amount_bil",
    "flow_count",
]
STOCK_CLOSURE_COLUMNS = [
    "period_start",
    "period_end",
    "axis",
    "holder_sector",
    "holder_subsector",
    "instrument_type",
    "maturity_bucket",
    "debt_scope",
    "opening_face_stock_bil",
    "event_face_stock_change_bil",
    "closing_face_stock_bil",
    "face_stock_closure_error_bil",
    "opening_adjusted_principal_stock_bil",
    "event_adjusted_principal_change_bil",
    "closing_adjusted_principal_stock_bil",
    "adjusted_principal_closure_error_bil",
    "opening_debt_stock_bil",
    "event_debt_stock_change_bil",
    "closing_debt_stock_bil",
    "debt_stock_closure_error_bil",
]
COMMITMENT_COLUMNS = [
    "period_start",
    "period_end",
    "event_seq_start",
    "event_seq_end",
    "event_count",
    "period_event_root_sha256",
    "whole_run_root_through_period_sha256",
    "event_type_counts_json",
]
RESOURCE_COLUMNS = [
    "sample_kind",
    "period_end",
    "rss_bytes",
    "peak_rss_bytes",
    "host_available_bytes",
    "process_cpu_seconds",
    "portfolio_rows",
    "active_portfolio_rows",
    "event_count",
    "current_key_cardinality",
    "max_key_cardinality",
    "bytes_written",
]
ANNUAL_COLUMNS = [
    "period_label",
    "period_start",
    "period_end",
    "is_partial_period",
    "coverage_days",
    "expected_coverage_days",
    "aggregation_clock_id",
    "tdc_change_bil",
    "overlap_cashflow_bil",
    "tdc_change_ex_overlap_bil",
    "cumulative_tdc_change_bil",
    "cumulative_overlap_cashflow_bil",
    "cumulative_tdc_change_ex_overlap_bil",
    "interest_outlay_bil",
    "issue_discount_cost_bil",
    "nonmarketable_interest_capitalized_bil",
    "tips_inflation_accretion_bil",
    "modeled_financing_cost_bil",
    "cumulative_interest_outlay_bil",
    "cumulative_issue_discount_cost_bil",
    "cumulative_nonmarketable_interest_capitalized_bil",
    "cumulative_tips_inflation_accretion_bil",
    "cumulative_modeled_financing_cost_bil",
    "modeled_financing_cost_basis",
    "modeled_financing_cost_units",
    "cumulative_basis",
    "new_issuance_face_bil",
    "new_issuance_original_term_face_years_bil",
    "new_issuance_bill_face_bil",
    "new_issuance_short_face_bil",
    "new_issuance_wam_years",
    "new_issuance_bill_share",
    "new_issuance_short_maturity_share",
    "snapshot_date",
    "outstanding_controlled_wam_years",
    "outstanding_controlled_bill_share",
    "outstanding_controlled_short_maturity_share",
]
class BoundedEvidenceError(RuntimeError):
    """Raised when bounded evidence cannot be admitted or closed."""


class ResourceLimitError(BoundedEvidenceError):
    """Raised when a declared memory or cardinality limit is crossed."""


@dataclass(frozen=True)
class BoundedResourceLimits:
    """Production resource envelope, with test-overridable explicit values."""

    minimum_available_bytes: int = 4 * _GIB
    application_abort_rss_bytes: int = 8 * _GIB
    parent_graceful_stop_rss_bytes: int = 10 * _GIB
    parent_kill_rss_bytes: int = 12 * _GIB
    acceptance_peak_rss_bytes: int = 6 * _GIB
    portfolio_row_budget: int = 750_000
    key_cardinality_budget: int = BOUNDED_KEY_CARDINALITY_CEILING

    def __post_init__(self) -> None:
        values = (
            self.minimum_available_bytes,
            self.application_abort_rss_bytes,
            self.parent_graceful_stop_rss_bytes,
            self.parent_kill_rss_bytes,
            self.acceptance_peak_rss_bytes,
            self.portfolio_row_budget,
            self.key_cardinality_budget,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("bounded resource limits must be nonnegative")
        if not (
            self.application_abort_rss_bytes
            <= self.parent_graceful_stop_rss_bytes
            <= self.parent_kill_rss_bytes
        ):
            raise ValueError("RSS stop thresholds must be monotonic")


class _Kahan:
    __slots__ = ("total", "compensation")

    def __init__(self) -> None:
        self.total = 0.0
        self.compensation = 0.0

    def add(self, value: float) -> None:
        adjusted = value - self.compensation
        updated = self.total + adjusted
        self.compensation = (updated - self.total) - adjusted
        self.total = updated


class _Aggregate:
    __slots__ = ("base", "sums", "weighted", "weight", "count")

    def __init__(self, base: Mapping[str, Any], numeric: tuple[str, ...]) -> None:
        self.base = dict(base)
        self.sums = {column: _Kahan() for column in numeric}
        self.weighted: dict[str, _Kahan] = {}
        self.weight: dict[str, _Kahan] = {}
        self.count = 0


class _BoundedChannel:
    """Append-compatible current-period reducer."""

    __slots__ = ("owner", "name", "total_count", "_groups")

    def __init__(self, owner: "BoundedScenarioEvidenceSink", name: str) -> None:
        self.owner = owner
        self.name = name
        self.total_count = 0
        self._groups: dict[tuple[Any, ...], _Aggregate] = {}

    def append(self, row: Mapping[str, Any]) -> None:
        self.owner._append(self.name, row)

    def __len__(self) -> int:
        return self.total_count

    def clear(self) -> None:
        self._groups.clear()


class _DeterministicCsvGzip:
    """One deterministic, append-only gzip CSV with a bounded writer buffer."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self.row_count = 0
        self._raw = path.open("xb")
        self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, mtime=0)
        self._text = io.TextIOWrapper(self._gzip, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._text,
            fieldnames=self.columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._writer.writeheader()
        self._closed = False

    def write_rows(self, rows: list[Mapping[str, Any]]) -> None:
        for row in rows:
            self._writer.writerow(
                {column: _csv_value(row.get(column)) for column in self.columns}
            )
            self.row_count += 1

    def flush(self) -> None:
        if self._closed:
            return
        self._text.flush()
        self._gzip.flush()
        self._raw.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._text.flush()
        self._text.detach()
        self._gzip.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._closed = True


class BoundedScenarioEvidenceSink(MutableMapping[str, _BoundedChannel]):
    """Mapping-compatible sink used by the production CBO engine path."""

    _CHANNEL_NAMES = (
        "tdcsim_accounting_journal",
        "tdcsim_period_issuance_flows",
        "tdcsim_period_principal_flows",
        "tdcsim_period_payment_flows",
        "tdcsim_holder_stocks",
        "tdcsim_tdc_principal_route_stocks",
        "tdcsim_debt_target_bridge",
        "tdcsim_scenario_metrics",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        limits: BoundedResourceLimits,
        rss_reader: Callable[[], int] | None = None,
        available_reader: Callable[[], int] | None = None,
        cpu_reader: Callable[[], float] | None = None,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self._rss_reader = rss_reader or current_rss_bytes
        self._available_reader = available_reader or host_available_memory_bytes
        self._cpu_reader = cpu_reader or time.process_time
        self._progress_callback = progress_callback
        self._channels = {
            name: _BoundedChannel(self, name) for name in self._CHANNEL_NAMES
        }
        self._writers = self._open_writers()
        self._begun = False
        self._finalized = False
        self._aborted = False
        self._opening_result: dict[str, Any] | None = None
        self._last_period_end = ""
        self._event_seq = 0
        self._period_seq_start = 1
        self._period_event_count = 0
        self._period_event_types: Counter[str] = Counter()
        self._period_hash = hashlib.sha256()
        self._whole_hash = hashlib.sha256()
        self._failure_ring: deque[dict[str, Any]] = deque(maxlen=64)
        self._period_count = 0
        self._max_key_cardinality = 0
        self._max_portfolio_rows = 0
        self._max_active_portfolio_rows = 0
        self._peak_rss_bytes = 0
        self._annual: dict[int, dict[str, Any]] = {}
        self._final_state_sha256 = ""
        self._pending_portfolio_snapshot: tuple[str, int, int] | None = None

    def __getitem__(self, key: str) -> _BoundedChannel:
        return self._channels[key]

    def __setitem__(self, key: str, value: _BoundedChannel) -> None:
        raise TypeError("bounded sink channels are fixed")

    def __delitem__(self, key: str) -> None:
        raise TypeError("bounded sink channels are fixed")

    def __iter__(self) -> Iterator[str]:
        return iter(self._channels)

    def __len__(self) -> int:
        return len(self._channels)

    def setdefault(self, key: str, default: Any = None) -> _BoundedChannel:
        if key not in self._channels:
            raise BoundedEvidenceError(f"undeclared bounded handoff table: {key}")
        return self._channels[key]

    @property
    def event_count(self) -> int:
        return self._event_seq

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    @property
    def max_portfolio_rows(self) -> int:
        return self._max_portfolio_rows

    @property
    def max_key_cardinality(self) -> int:
        return self._max_key_cardinality

    @property
    def portfolio_row_budget(self) -> int:
        return self.limits.portfolio_row_budget

    def check_portfolio_rows(self, observed: int, *, pending: int = 0) -> None:
        """Fail before a portfolio append that would cross the declared cap."""

        projected = int(observed) + int(pending)
        self._max_portfolio_rows = max(self._max_portfolio_rows, projected)
        if projected > self.limits.portfolio_row_budget:
            raise ResourceLimitError(
                "portfolio-row budget exceeded before append: "
                f"projected={projected}, budget={self.limits.portfolio_row_budget}"
            )

    def capture_portfolio_snapshot(
        self,
        portfolio: pd.DataFrame,
        *,
        snapshot_date: Any,
    ) -> None:
        """Reduce one live portfolio snapshot without retaining security rows.

        This is deliberately separate from the legacy handoff snapshot helper.  The
        bounded production path streams the live portfolio once, retains only capped
        holder/route aggregates, and then uses those aggregates for period closure.
        """

        if self._pending_portfolio_snapshot is not None:
            raise BoundedEvidenceError(
                "previous live portfolio snapshot was not consumed"
            )
        if not isinstance(portfolio, pd.DataFrame):
            raise BoundedEvidenceError("live portfolio snapshot must be a DataFrame")
        date = str(pd.Timestamp(snapshot_date).date())
        if any(
            str(row.get("date")) == date
            for name in (
                "tdcsim_holder_stocks",
                "tdcsim_tdc_principal_route_stocks",
            )
            for row in self._rows(name)
        ):
            raise BoundedEvidenceError(
                f"duplicate live portfolio snapshot date: {date}"
            )

        portfolio_rows = len(portfolio)
        self.check_portfolio_rows(portfolio_rows)
        required = {
            "SecurityType",
            "FaceValue",
            "HolderType",
            "HolderSubBucket",
        }
        missing = sorted(required - set(map(str, portfolio.columns)))
        if portfolio_rows and missing:
            raise BoundedEvidenceError(
                "live portfolio snapshot is missing columns: " + ", ".join(missing)
            )

        column_index = {
            str(column): position
            for position, column in enumerate(portfolio.columns)
        }
        holder_totals: dict[
            tuple[str, str, str, str, str], list[_Kahan]
        ] = {}
        route_totals: dict[
            tuple[str, str, str, str, str], list[_Kahan]
        ] = {}
        retained_group_count = sum(
            len(channel._groups) for channel in self._channels.values()
        )
        active_rows = 0

        def value(row: tuple[Any, ...], column: str, default: Any = None) -> Any:
            position = column_index.get(column)
            return default if position is None else row[position]

        def add(
            totals: dict[tuple[str, str, str, str, str], list[_Kahan]],
            key: tuple[str, str, str, str, str],
            debt: float,
            face: float,
            adjusted: float,
        ) -> None:
            sums = totals.get(key)
            if sums is None:
                projected = (
                    retained_group_count
                    + len(holder_totals)
                    + len(route_totals)
                    + 1
                )
                self._max_key_cardinality = max(
                    self._max_key_cardinality,
                    projected,
                )
                if projected > self.limits.key_cardinality_budget:
                    raise ResourceLimitError(
                        "live portfolio snapshot exceeds key-cardinality budget"
                    )
                sums = [_Kahan(), _Kahan(), _Kahan()]
                totals[key] = sums
            sums[0].add(debt)
            sums[1].add(face)
            sums[2].add(adjusted)

        for row in portfolio.itertuples(index=False, name=None):
            if "Status" in column_index and _snapshot_text(value(row, "Status")) != "Active":
                continue
            active_rows += 1
            security = _snapshot_text(value(row, "SecurityType"))
            holder = _snapshot_text(value(row, "HolderType"))
            holder_subbucket = _snapshot_text(value(row, "HolderSubBucket"))
            if security not in SECURITY_TYPES:
                raise BoundedEvidenceError(
                    f"live portfolio snapshot has unknown security type: {security!r}"
                )
            if holder not in HOLDER_TYPES:
                raise BoundedEvidenceError(
                    f"live portfolio snapshot has unknown holder type: {holder!r}"
                )
            _validate_snapshot_subbucket(
                holder,
                holder_subbucket,
                label="holder",
            )
            face = _snapshot_required_number(value(row, "FaceValue"), "FaceValue")
            adjusted = (
                _snapshot_optional_number(
                    value(row, "AdjustedPrincipal"),
                    default=face,
                    column="AdjustedPrincipal",
                )
                if security == "TIPS"
                else 0.0
            )
            debt = adjusted if security == "TIPS" else face
            maturity = _snapshot_maturity_bucket(
                security,
                value(row, "OriginalMaturityYears"),
                value(row, "MaturityCategory"),
            )
            route_holder, route_subbucket = _snapshot_route(
                holder=holder,
                holder_subbucket=holder_subbucket,
                route_holder=value(row, "TDCPrincipalHolderType"),
                route_subbucket=value(row, "TDCPrincipalHolderSubBucket"),
            )
            if route_holder not in HOLDER_TYPES:
                raise BoundedEvidenceError(
                    "live portfolio snapshot has unknown TDC principal route: "
                    f"{route_holder!r}"
                )
            scopes = ["all_active_treasury"]
            if (
                security in PUBLIC_MARKETABLE_SECURITY_TYPES
                and holder not in INTRAGOV_HOLDERS
            ):
                scopes.append("controlled_public_marketable")
            for scope in scopes:
                add(
                    holder_totals,
                    (holder, holder_subbucket, security, maturity, scope),
                    debt,
                    face,
                    adjusted,
                )
                add(
                    route_totals,
                    (route_holder, route_subbucket, security, maturity, scope),
                    debt,
                    face,
                    adjusted,
                )

        while holder_totals:
            key, reducers = holder_totals.popitem()
            holder, subbucket, security, maturity, scope = key
            debt, face, adjusted = (reducer.total for reducer in reducers)
            if max(abs(debt), abs(face), abs(adjusted)) <= TGA_FLOOR_TOLERANCE:
                continue
            self["tdcsim_holder_stocks"].append(
                {
                    "date": date,
                    "holder_sector": holder,
                    "holder_subsector": subbucket,
                    "instrument_type": security,
                    "maturity_bucket": maturity,
                    "debt_held_bil": debt,
                    "face_stock_bil": face,
                    "adjusted_principal_stock_bil": adjusted,
                    "valuation_basis": (
                        "tips_adjusted_principal" if security == "TIPS" else "face"
                    ),
                    "debt_scope": scope,
                    "allocation_method": "end_of_period_stock_snapshot",
                }
            )
        while route_totals:
            key, reducers = route_totals.popitem()
            holder, subbucket, security, maturity, scope = key
            debt, face, adjusted = (reducer.total for reducer in reducers)
            if max(abs(debt), abs(face), abs(adjusted)) <= TGA_FLOOR_TOLERANCE:
                continue
            self["tdcsim_tdc_principal_route_stocks"].append(
                {
                    "date": date,
                    "route_holder_sector": holder,
                    "route_holder_subsector": subbucket,
                    "instrument_type": security,
                    "maturity_bucket": maturity,
                    "route_debt_held_bil": debt,
                    "route_face_stock_bil": face,
                    "route_adjusted_principal_stock_bil": adjusted,
                    "valuation_basis": (
                        "tips_adjusted_principal" if security == "TIPS" else "face"
                    ),
                    "debt_scope": scope,
                    "allocation_method": (
                        "end_of_period_tdc_principal_route_stock_snapshot"
                    ),
                    "route_stock_basis": "tdc_principal_settlement_route",
                }
            )

        self._pending_portfolio_snapshot = (date, portfolio_rows, active_rows)

    def begin_run(
        self,
        *,
        opening_date: Any,
        opening_result: Mapping[str, Any],
    ) -> None:
        if self._begun:
            raise BoundedEvidenceError("bounded sink begin_run called twice")
        opening = str(pd.Timestamp(opening_date).date())
        portfolio_rows, active_portfolio_rows = self._consume_portfolio_snapshot(
            opening
        )
        holder_rows = self._rows("tdcsim_holder_stocks")
        route_rows = self._rows("tdcsim_tdc_principal_route_stocks")
        if not holder_rows or not route_rows:
            raise BoundedEvidenceError("bounded sink requires opening holder and route snapshots")
        if {str(row.get("date")) for row in holder_rows} != {opening}:
            raise BoundedEvidenceError("opening holder snapshot date is not unique")
        if {str(row.get("date")) for row in route_rows} != {opening}:
            raise BoundedEvidenceError("opening route snapshot date is not unique")
        available = int(self._available_reader())
        if (
            self.limits.minimum_available_bytes
            and available < self.limits.minimum_available_bytes
        ):
            raise ResourceLimitError(
                "bounded run admission failed: "
                f"available={available}, required={self.limits.minimum_available_bytes}"
            )
        self._opening_result = dict(opening_result)
        self._last_period_end = opening
        self._begun = True
        self._observe_resources(
            sample_kind="admission",
            period_end=opening,
            portfolio_rows=portfolio_rows,
            active_portfolio_rows=active_portfolio_rows,
            enforce_abort=True,
        )
        self._notify_progress(
            progress_state="admission",
            admission_date=opening,
            last_completed_period=None,
        )

    def close_period(
        self,
        *,
        prev_date: Any,
        current_date: Any,
        closing_result: Mapping[str, Any],
    ) -> None:
        if not self._begun or self._finalized or self._aborted:
            raise BoundedEvidenceError("bounded sink is not open for period close")
        period_start = str(pd.Timestamp(prev_date).date())
        period_end = str(pd.Timestamp(current_date).date())
        portfolio_rows, active_portfolio_rows = self._consume_portfolio_snapshot(
            period_end
        )
        if period_start != self._last_period_end:
            raise BoundedEvidenceError(
                f"period continuity failed: expected {self._last_period_end}, got {period_start}"
            )
        journal_rows = self._rows("tdcsim_accounting_journal")
        if journal_rows and {
            (str(row.get("period_start")), str(row.get("period_end")))
            for row in journal_rows
        } != {(period_start, period_end)}:
            raise BoundedEvidenceError("ledger reducer contains rows from another period")

        holder_rows = self._rows("tdcsim_holder_stocks")
        route_rows = self._rows("tdcsim_tdc_principal_route_stocks")
        dates = {str(row.get("date")) for row in holder_rows}
        route_dates = {str(row.get("date")) for row in route_rows}
        if dates != {period_start, period_end}:
            raise BoundedEvidenceError(
                f"holder snapshots do not bracket period: {sorted(dates)}"
            )
        if route_dates != {period_start, period_end}:
            raise BoundedEvidenceError(
                f"route snapshots do not bracket period: {sorted(route_dates)}"
            )

        results = pd.DataFrame(
            [
                {"Date": period_start, **dict(self._opening_result or {})},
                {"Date": period_end, **dict(closing_result)},
            ]
        )
        raw = {
            "tdcsim_accounting_journal": journal_rows,
            "tdcsim_period_issuance_flows": self._rows(
                "tdcsim_period_issuance_flows"
            ),
            "tdcsim_period_principal_flows": self._rows(
                "tdcsim_period_principal_flows"
            ),
            "tdcsim_period_payment_flows": self._rows(
                "tdcsim_period_payment_flows"
            ),
            "tdcsim_holder_stocks": holder_rows,
            "tdcsim_tdc_principal_route_stocks": route_rows,
            "tdcsim_debt_target_bridge": self._rows("tdcsim_debt_target_bridge"),
            "tdcsim_scenario_metrics": self._rows("tdcsim_scenario_metrics"),
        }
        accounting = _accounting_closure_handoff_tables(results, raw)[
            "tdcsim_accounting_closure"
        ]
        route_closure = _route_stock_closure_handoff_tables(raw)[
            "tdcsim_tdc_principal_route_stock_closure"
        ]
        tdc = _tdc_handoff_tables(results)
        stock_closure = _stock_closure_rows(raw, period_start, period_end)
        financing = _financing_row(
            closing_result, period_start, period_end
        )
        self._validate_period(
            accounting=accounting,
            route_closure=route_closure,
            stock_closure=stock_closure,
            tdc=tdc,
            financing=financing,
            closing_result=closing_result,
        )

        commitment = {
            "period_start": period_start,
            "period_end": period_end,
            "event_seq_start": (
                self._period_seq_start if self._period_event_count else 0
            ),
            "event_seq_end": self._event_seq if self._period_event_count else 0,
            "event_count": self._period_event_count,
            "period_event_root_sha256": self._period_hash.hexdigest(),
            "whole_run_root_through_period_sha256": self._whole_hash.hexdigest(),
            "event_type_counts_json": json.dumps(
                dict(sorted(self._period_event_types.items())),
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

        self._writers["ledger"].write_rows(journal_rows)
        self._writers["issuance"].write_rows(raw["tdcsim_period_issuance_flows"])
        self._writers["principal"].write_rows(raw["tdcsim_period_principal_flows"])
        self._writers["payment"].write_rows(raw["tdcsim_period_payment_flows"])
        self._writers["accounting"].write_rows(accounting)
        self._writers["stock"].write_rows(stock_closure)
        self._writers["route"].write_rows(route_closure)
        self._writers["tdc_summary"].write_rows(
            tdc.get("tdcsim_period_tdc_summary", [])
        )
        self._writers["tdc_components"].write_rows(
            tdc.get("tdcsim_period_tdc_components", [])
        )
        self._writers["debt_bridge"].write_rows(raw["tdcsim_debt_target_bridge"])
        self._writers["scenario_metrics"].write_rows(
            raw["tdcsim_scenario_metrics"]
        )
        self._writers["commitments"].write_rows([commitment])
        self._update_annual(
            period_start=period_start,
            period_end=period_end,
            closing_result=closing_result,
            issuance_rows=raw["tdcsim_period_issuance_flows"],
            scenario_metric_rows=raw["tdcsim_scenario_metrics"],
            tdc_summary=tdc.get("tdcsim_period_tdc_summary", []),
            financing=financing,
        )

        closing_holder = [
            row for row in holder_rows if str(row.get("date")) == period_end
        ]
        closing_route = [
            row for row in route_rows if str(row.get("date")) == period_end
        ]
        self._clear_period_channels()
        self._restore_snapshot_rows(
            "tdcsim_holder_stocks", closing_holder
        )
        self._restore_snapshot_rows(
            "tdcsim_tdc_principal_route_stocks", closing_route
        )
        self._opening_result = dict(closing_result)
        self._last_period_end = period_end
        self._period_count += 1
        self._period_seq_start = self._event_seq + 1
        self._period_event_count = 0
        self._period_event_types.clear()
        self._period_hash = hashlib.sha256()
        self._observe_resources(
            sample_kind="period_close",
            period_end=period_end,
            portfolio_rows=portfolio_rows,
            active_portfolio_rows=active_portfolio_rows,
            enforce_abort=True,
        )
        for writer in self._writers.values():
            writer.flush()
        self._notify_progress(
            progress_state="period_complete",
            admission_date=None,
            last_completed_period=period_end,
        )

    def _consume_portfolio_snapshot(self, expected_date: str) -> tuple[int, int]:
        pending = self._pending_portfolio_snapshot
        if pending is None:
            raise BoundedEvidenceError(
                f"missing live portfolio snapshot for {expected_date}"
            )
        date, portfolio_rows, active_rows = pending
        if date != expected_date:
            raise BoundedEvidenceError(
                "live portfolio snapshot date mismatch: "
                f"expected {expected_date}, got {date}"
            )
        self._pending_portfolio_snapshot = None
        return portfolio_rows, active_rows

    def finalize(self) -> dict[str, Any]:
        if not self._begun or self._aborted:
            raise BoundedEvidenceError("bounded sink cannot finalize")
        if self._finalized:
            raise BoundedEvidenceError("bounded sink finalized twice")
        if self._period_event_count:
            raise BoundedEvidenceError("unclosed accounting events remain at finalize")
        if self._pending_portfolio_snapshot is not None:
            raise BoundedEvidenceError(
                "unclosed live portfolio snapshot remains at finalize"
            )
        if (
            self.limits.acceptance_peak_rss_bytes
            and self._peak_rss_bytes > self.limits.acceptance_peak_rss_bytes
        ):
            raise ResourceLimitError(
                "bounded run exceeded the acceptance RSS ceiling: "
                f"peak={self._peak_rss_bytes}, "
                f"limit={self.limits.acceptance_peak_rss_bytes}"
            )
        self._writers["annual"].write_rows(self._annual_rows())
        self._final_state_sha256 = _final_state_digest(
            self._opening_result or {},
            self._rows("tdcsim_holder_stocks"),
            self._rows("tdcsim_tdc_principal_route_stocks"),
        )
        for writer in self._writers.values():
            writer.close()
        self._finalized = True
        artifacts = {
            logical_name: _artifact(self.output_dir, writer.path, writer.row_count)
            for logical_name, writer in sorted(self._writers.items())
        }
        deterministic = {
            key: value
            for key, value in artifacts.items()
            if key != "resources"
        }
        return {
            "evidence_profile": EVIDENCE_PROFILE,
            "verification_grade": VERIFICATION_GRADE,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_count": self._event_seq,
            "event_root_sha256": self._whole_hash.hexdigest(),
            "final_state_sha256": self._final_state_sha256,
            "period_count": self._period_count,
            "peak_rss_bytes": self._peak_rss_bytes,
            "portfolio_row_budget": self.limits.portfolio_row_budget,
            "max_portfolio_rows": self._max_portfolio_rows,
            "max_active_portfolio_rows": self._max_active_portfolio_rows,
            "key_cardinality_budget": self.limits.key_cardinality_budget,
            "max_key_cardinality": self._max_key_cardinality,
            "memory_thresholds": {
                "minimum_available_bytes": self.limits.minimum_available_bytes,
                "acceptance_peak_rss_bytes": self.limits.acceptance_peak_rss_bytes,
                "application_abort_rss_bytes": self.limits.application_abort_rss_bytes,
                "parent_graceful_stop_rss_bytes": self.limits.parent_graceful_stop_rss_bytes,
                "parent_kill_rss_bytes": self.limits.parent_kill_rss_bytes,
            },
            "artifacts": artifacts,
            "deterministic_artifacts": deterministic,
        }

    def abort(self, exc: BaseException) -> dict[str, Any]:
        if not self._finalized:
            for writer in self._writers.values():
                try:
                    writer.close()
                except Exception:
                    pass
        self._aborted = True
        return {
            "status": "failed",
            "exception_class": type(exc).__name__,
            "exception_message": str(exc),
            "last_completed_period": self._last_period_end,
            "event_count": self._event_seq,
            "event_root_sha256": self._whole_hash.hexdigest(),
            "peak_rss_bytes": self._peak_rss_bytes,
            "application_abort_rss_bytes": self.limits.application_abort_rss_bytes,
            "portfolio_row_budget": self.limits.portfolio_row_budget,
            "max_portfolio_rows": self._max_portfolio_rows,
            "last_events": list(self._failure_ring),
        }

    def on_accounting_event(
        self, row: Mapping[str, Any], *, event_seq: int | None = None
    ) -> None:
        self._append_journal(row, event_seq=event_seq)

    def retained_shape(self) -> dict[str, int]:
        """Return retained object cardinalities for memory-shape tests."""

        return {
            "current_groups": sum(
                len(channel._groups) for channel in self._channels.values()
            ),
            "failure_ring": len(self._failure_ring),
            "annual_buckets": len(self._annual),
            "writer_count": len(self._writers),
        }

    def _open_writers(self) -> dict[str, _DeterministicCsvGzip]:
        specs = {
            "ledger": ("tdcsim_period_ledger_totals.csv.gz", LEDGER_COLUMNS),
            "issuance": (
                "tdcsim_period_issuance_aggregates.csv.gz",
                ISSUANCE_COLUMNS,
            ),
            "principal": (
                "tdcsim_period_principal_aggregates.csv.gz",
                PRINCIPAL_COLUMNS,
            ),
            "payment": (
                "tdcsim_period_payment_aggregates.csv.gz",
                PAYMENT_COLUMNS,
            ),
            "accounting": (
                "tdcsim_period_accounting_closure.csv.gz",
                HANDOFF_TABLE_COLUMNS["tdcsim_accounting_closure"],
            ),
            "stock": ("tdcsim_period_stock_closure.csv.gz", STOCK_CLOSURE_COLUMNS),
            "route": (
                "tdcsim_period_route_stock_closure.csv.gz",
                HANDOFF_TABLE_COLUMNS[
                    "tdcsim_tdc_principal_route_stock_closure"
                ],
            ),
            "tdc_summary": (
                "tdcsim_period_tdc_summary.csv.gz",
                HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_summary"],
            ),
            "tdc_components": (
                "tdcsim_period_tdc_components.csv.gz",
                HANDOFF_TABLE_COLUMNS["tdcsim_period_tdc_components"],
            ),
            "debt_bridge": (
                "tdcsim_debt_target_bridge.csv.gz",
                HANDOFF_TABLE_COLUMNS["tdcsim_debt_target_bridge"],
            ),
            "scenario_metrics": (
                "tdcsim_scenario_metrics.csv.gz",
                HANDOFF_TABLE_COLUMNS["tdcsim_scenario_metrics"],
            ),
            "commitments": (
                "tdcsim_event_commitments.csv.gz",
                COMMITMENT_COLUMNS,
            ),
            "resources": ("tdcsim_resource_samples.csv.gz", RESOURCE_COLUMNS),
            "annual": (
                "tdcsim_annual_economic_summary.csv.gz",
                ANNUAL_COLUMNS,
            ),
        }
        return {
            logical_name: _DeterministicCsvGzip(
                self.output_dir / filename, columns
            )
            for logical_name, (filename, columns) in specs.items()
        }

    def _append(self, name: str, row: Mapping[str, Any]) -> None:
        if self._finalized or self._aborted:
            raise BoundedEvidenceError("cannot append to closed bounded sink")
        if name == "tdcsim_accounting_journal":
            self._append_journal(row)
            return
        if name == "tdcsim_period_issuance_flows":
            self._aggregate(
                name,
                row,
                keys=_ISSUANCE_KEYS,
                numeric=_ISSUANCE_SUMS,
                weighted=_ISSUANCE_WEIGHTED,
                weight_column="face_issued_bil",
            )
        elif name == "tdcsim_period_principal_flows":
            self._aggregate(name, row, keys=_PRINCIPAL_KEYS, numeric=_PRINCIPAL_SUMS)
        elif name == "tdcsim_period_payment_flows":
            self._aggregate(name, row, keys=_PAYMENT_KEYS, numeric=("amount_bil",))
        elif name == "tdcsim_holder_stocks":
            self._aggregate(
                name,
                row,
                keys=_HOLDER_STOCK_KEYS,
                numeric=(
                    "debt_held_bil",
                    "face_stock_bil",
                    "adjusted_principal_stock_bil",
                ),
            )
        elif name == "tdcsim_tdc_principal_route_stocks":
            self._aggregate(
                name,
                row,
                keys=_ROUTE_STOCK_KEYS,
                numeric=(
                    "route_debt_held_bil",
                    "route_face_stock_bil",
                    "route_adjusted_principal_stock_bil",
                ),
            )
        elif name == "tdcsim_debt_target_bridge":
            self._aggregate(
                name, row, keys=_DEBT_BRIDGE_KEYS, numeric=_DEBT_BRIDGE_SUMS
            )
        elif name == "tdcsim_scenario_metrics":
            self._aggregate(
                name,
                row,
                keys=_SCENARIO_METRIC_KEYS,
                numeric=_SCENARIO_METRIC_VALUES,
            )
        else:
            raise BoundedEvidenceError(f"undeclared bounded handoff table: {name}")
        self._check_key_budget()

    def _append_journal(
        self, row: Mapping[str, Any], *, event_seq: int | None = None
    ) -> None:
        normalized = _validated_journal_event(row)
        _validate_bounded_reducer_key("tdcsim_accounting_journal", normalized)
        expected = self._event_seq + 1
        sequence = expected if event_seq is None else int(event_seq)
        if sequence != expected:
            raise BoundedEvidenceError(
                f"accounting event sequence discontinuity: expected {expected}, got {sequence}"
            )
        encoded = _encode_event(sequence, normalized)
        self._event_seq = sequence
        self._period_event_count += 1
        self._period_event_types[str(normalized["event_type"])] += 1
        self._period_hash.update(encoded)
        self._whole_hash.update(encoded)
        diagnostic = {
            "event_seq": sequence,
            **{
                key: normalized.get(key)
                for key in (
                    "period_start",
                    "period_end",
                    "event_type",
                    "leg_type",
                    "accounting_basis",
                    "holder_sector",
                    "route_holder_sector",
                    "instrument_type",
                )
            },
        }
        self._failure_ring.append(diagnostic)
        self._aggregate(
            "tdcsim_accounting_journal",
            normalized,
            keys=_JOURNAL_KEYS,
            numeric=_JOURNAL_NUMERIC,
        )
        self._check_key_budget()

    def _aggregate(
        self,
        name: str,
        row: Mapping[str, Any],
        *,
        keys: tuple[str, ...],
        numeric: tuple[str, ...],
        weighted: tuple[str, ...] = (),
        weight_column: str | None = None,
    ) -> None:
        channel = self._channels[name]
        base = {key: _normalized_key(row.get(key)) for key in keys}
        _validate_bounded_reducer_key(name, base)
        key = tuple(base[column] for column in keys)
        aggregate = channel._groups.get(key)
        if aggregate is None:
            aggregate = _Aggregate(base, numeric)
            channel._groups[key] = aggregate
        for column in numeric:
            value = _finite_or_optional(row.get(column), column=column)
            if value is not None:
                aggregate.sums[column].add(value)
        if weighted:
            weight = _finite_or_optional(row.get(weight_column or ""), column=weight_column or "")
            weight = 0.0 if weight is None else weight
            for column in weighted:
                value = _finite_or_optional(row.get(column), column=column)
                if value is None:
                    continue
                aggregate.weighted.setdefault(column, _Kahan()).add(value * weight)
                aggregate.weight.setdefault(column, _Kahan()).add(weight)
        aggregate.count += 1
        channel.total_count += 1

    def _rows(self, name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        channel = self._channels[name]
        for key in sorted(channel._groups, key=lambda value: tuple(map(str, value))):
            aggregate = channel._groups[key]
            row = dict(aggregate.base)
            row.update(
                {column: value.total for column, value in aggregate.sums.items()}
            )
            for column, value in aggregate.weighted.items():
                denom = aggregate.weight[column].total
                row[column] = value.total / denom if abs(denom) > 1e-18 else math.nan
            if name == "tdcsim_accounting_journal":
                row["event_count"] = aggregate.count
            elif name in {
                "tdcsim_period_issuance_flows",
                "tdcsim_period_principal_flows",
                "tdcsim_period_payment_flows",
            }:
                row["flow_count"] = aggregate.count
            rows.append(row)
        return rows

    def _validate_period(
        self,
        *,
        accounting: list[dict[str, Any]],
        route_closure: list[dict[str, Any]],
        stock_closure: list[dict[str, Any]],
        tdc: Mapping[str, list[dict[str, Any]]],
        financing: Mapping[str, Any],
        closing_result: Mapping[str, Any],
    ) -> None:
        if len(accounting) != 1:
            raise BoundedEvidenceError("period accounting closure must contain one row")
        accounting_row = accounting[0]
        for column in (
            "face_stock_closure_error_bil",
            "adjusted_principal_closure_error_bil",
            "treasury_cash_closure_error_bil",
            "reserve_closure_error_bil",
            "deposit_closure_error_bil",
            "holder_total_error_bil",
            "instrument_total_error_bil",
            "unexplained_residual_bil",
        ):
            tolerance = CASH_TOLERANCE_BIL if column == "treasury_cash_closure_error_bil" else STOCK_TOLERANCE_BIL
            if abs(float(accounting_row.get(column, 0.0) or 0.0)) > tolerance:
                raise BoundedEvidenceError(
                    f"period accounting closure failed on {column}: {accounting_row.get(column)}"
                )
        for rows, column, label in (
            (route_closure, "closure_identity_error_bil", "route"),
            (stock_closure, "debt_stock_closure_error_bil", "stock debt"),
            (stock_closure, "face_stock_closure_error_bil", "stock face"),
            (
                stock_closure,
                "adjusted_principal_closure_error_bil",
                "stock adjusted principal",
            ),
        ):
            if rows and max(abs(float(row.get(column, 0.0) or 0.0)) for row in rows) > STOCK_TOLERANCE_BIL:
                raise BoundedEvidenceError(f"period {label} closure failed")
        summaries = tdc.get("tdcsim_period_tdc_summary", [])
        if len(summaries) != 1:
            raise BoundedEvidenceError("period TDC summary must contain one row")
        summary = summaries[0]
        if abs(float(summary.get("component_sum_error_bil", 0.0) or 0.0)) > STOCK_TOLERANCE_BIL:
            raise BoundedEvidenceError("period TDC component identity failed")
        ex_overlap_error = (
            float(summary.get("tdc_change_bil", 0.0) or 0.0)
            - float(summary.get("overlap_cashflow_bil", 0.0) or 0.0)
            - float(summary.get("tdc_change_ex_overlap_bil", 0.0) or 0.0)
        )
        if abs(ex_overlap_error) > STOCK_TOLERANCE_BIL:
            raise BoundedEvidenceError("period ex-overlap identity failed")
        if abs(float(financing["component_identity_error_bil"])) > STOCK_TOLERANCE_BIL:
            raise BoundedEvidenceError("period financing-cost component identity failed")
        tga = float(closing_result.get("TGA", 0.0) or 0.0)
        if tga < TGA_MINIMUM_BIL:
            raise BoundedEvidenceError(f"negative TGA at period close: {tga}")

    def _update_annual(
        self,
        *,
        period_start: str,
        period_end: str,
        closing_result: Mapping[str, Any],
        issuance_rows: list[Mapping[str, Any]],
        scenario_metric_rows: list[Mapping[str, Any]],
        tdc_summary: list[Mapping[str, Any]],
        financing: Mapping[str, Any],
    ) -> None:
        end = pd.Timestamp(period_end)
        fiscal_year = int(end.year + 1 if end.month >= 10 else end.year)
        bucket = self._annual.get(fiscal_year)
        if bucket is None:
            bucket = {
                "period_start": period_start,
                "period_end": period_end,
                "coverage_days": 0,
                "sums": {},
                "snapshot": {},
            }
            self._annual[fiscal_year] = bucket
        bucket["period_end"] = period_end
        bucket["coverage_days"] += int(
            (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days
        )

        def add(name: str, value: Any) -> None:
            number = _finite_or_optional(value, column=name)
            if number is None:
                raise BoundedEvidenceError(
                    f"annual reducer is missing required value {name}"
                )
            bucket["sums"].setdefault(name, _Kahan()).add(number)

        summary = tdc_summary[0]
        add("tdc_change_bil", summary.get("tdc_change_bil"))
        add("overlap_cashflow_bil", summary.get("overlap_cashflow_bil"))
        add(
            "tdc_change_ex_overlap_bil",
            summary.get("tdc_change_ex_overlap_bil"),
        )
        for name in (
            "interest_outlay_bil",
            "issue_discount_cost_bil",
            "nonmarketable_interest_capitalized_bil",
            "tips_inflation_accretion_bil",
            "modeled_financing_cost_bil",
        ):
            add(name, financing.get(name))
        for row in issuance_rows:
            face = _required_finite(row, "face_issued_bil")
            maturity = _required_finite(row, "weighted_original_term_years")
            add("new_issuance_face_bil", face)
            add("new_issuance_original_term_face_years_bil", face * maturity)
            if (
                str(row.get("instrument_type")) == "Fixed"
                and str(row.get("maturity_bucket")) == "bills"
            ):
                add("new_issuance_bill_face_bil", face)
            if maturity <= 1.0 + 1e-9:
                add("new_issuance_short_face_bil", face)
        if end.month == 9 and end.day == 30:
            if not scenario_metric_rows:
                raise BoundedEvidenceError(
                    f"missing September 30 maturity snapshot for {period_end}"
                )
            metric = scenario_metric_rows[-1]
            bucket["snapshot"] = {
                "snapshot_date": period_end,
                "outstanding_controlled_wam_years": _required_finite(
                    metric, "outstanding_controlled_wam_years"
                ),
                "outstanding_controlled_bill_share": _required_finite(
                    metric, "outstanding_controlled_bill_share"
                ),
                "outstanding_controlled_short_maturity_share": _required_finite(
                    metric, "outstanding_controlled_short_maturity_share"
                ),
            }

    def _annual_rows(self) -> list[dict[str, Any]]:
        cumulative = {
            name: 0.0
            for name in (
                "tdc_change_bil",
                "overlap_cashflow_bil",
                "tdc_change_ex_overlap_bil",
                "interest_outlay_bil",
                "issue_discount_cost_bil",
                "nonmarketable_interest_capitalized_bil",
                "tips_inflation_accretion_bil",
                "modeled_financing_cost_bil",
            )
        }
        rows: list[dict[str, Any]] = []
        first_year = min(self._annual) if self._annual else None
        last_year = max(self._annual) if self._annual else None
        for fiscal_year in sorted(self._annual):
            bucket = self._annual[fiscal_year]
            fy_start = pd.Timestamp(f"{fiscal_year - 1}-09-30")
            fy_end = pd.Timestamp(f"{fiscal_year}-09-30")
            expected = int((fy_end - fy_start).days)
            coverage = int(bucket["coverage_days"])
            is_partial = coverage != expected
            if is_partial and fiscal_year == first_year:
                label = f"FY{fiscal_year}_PARTIAL_OPENING"
            elif is_partial and fiscal_year == last_year:
                label = f"FY{fiscal_year}_PARTIAL_CLOSING"
            else:
                label = f"FY{fiscal_year}"
            sums = {
                name: value.total for name, value in bucket["sums"].items()
            }
            for name in (
                "tdc_change_bil",
                "overlap_cashflow_bil",
                "tdc_change_ex_overlap_bil",
                "interest_outlay_bil",
                "issue_discount_cost_bil",
                "nonmarketable_interest_capitalized_bil",
                "tips_inflation_accretion_bil",
                "modeled_financing_cost_bil",
                "new_issuance_face_bil",
                "new_issuance_original_term_face_years_bil",
                "new_issuance_bill_face_bil",
                "new_issuance_short_face_bil",
            ):
                sums.setdefault(name, 0.0)
            _assert_close(
                sums["tdc_change_bil"],
                sums["overlap_cashflow_bil"]
                + sums["tdc_change_ex_overlap_bil"],
                label=f"FY{fiscal_year} TDC ex-overlap identity",
            )
            _assert_close(
                sums["modeled_financing_cost_bil"],
                sums["interest_outlay_bil"]
                + sums["issue_discount_cost_bil"]
                + sums["nonmarketable_interest_capitalized_bil"]
                + sums["tips_inflation_accretion_bil"],
                label=f"FY{fiscal_year} financing-cost component identity",
            )
            for name in cumulative:
                cumulative[name] += float(sums.get(name, 0.0))
            _assert_close(
                cumulative["tdc_change_bil"],
                cumulative["overlap_cashflow_bil"]
                + cumulative["tdc_change_ex_overlap_bil"],
                label=f"FY{fiscal_year} cumulative TDC ex-overlap identity",
            )
            _assert_close(
                cumulative["modeled_financing_cost_bil"],
                cumulative["interest_outlay_bil"]
                + cumulative["issue_discount_cost_bil"]
                + cumulative["nonmarketable_interest_capitalized_bil"]
                + cumulative["tips_inflation_accretion_bil"],
                label=f"FY{fiscal_year} cumulative financing-cost component identity",
            )
            face = float(sums.get("new_issuance_face_bil", 0.0))
            row = {
                "period_label": label,
                "period_start": bucket["period_start"],
                "period_end": bucket["period_end"],
                "is_partial_period": is_partial,
                "coverage_days": coverage,
                "expected_coverage_days": expected,
                "aggregation_clock_id": AGGREGATION_CLOCK_ID,
                **sums,
                "cumulative_tdc_change_bil": cumulative["tdc_change_bil"],
                "cumulative_overlap_cashflow_bil": cumulative[
                    "overlap_cashflow_bil"
                ],
                "cumulative_tdc_change_ex_overlap_bil": cumulative[
                    "tdc_change_ex_overlap_bil"
                ],
                "cumulative_interest_outlay_bil": cumulative[
                    "interest_outlay_bil"
                ],
                "cumulative_issue_discount_cost_bil": cumulative[
                    "issue_discount_cost_bil"
                ],
                "cumulative_nonmarketable_interest_capitalized_bil": cumulative[
                    "nonmarketable_interest_capitalized_bil"
                ],
                "cumulative_tips_inflation_accretion_bil": cumulative[
                    "tips_inflation_accretion_bil"
                ],
                "cumulative_modeled_financing_cost_bil": cumulative[
                    "modeled_financing_cost_bil"
                ],
                "modeled_financing_cost_basis": (
                    "nominal_model_cost_incurred_within_simulation_horizon"
                ),
                "modeled_financing_cost_units": "billions_of_nominal_dollars",
                "cumulative_basis": "since_simulation_origin",
                "new_issuance_wam_years": (
                    float(sums.get("new_issuance_original_term_face_years_bil", 0.0))
                    / face
                    if face > 1e-12
                    else math.nan
                ),
                "new_issuance_bill_share": (
                    float(sums.get("new_issuance_bill_face_bil", 0.0)) / face
                    if face > 1e-12
                    else math.nan
                ),
                "new_issuance_short_maturity_share": (
                    float(sums.get("new_issuance_short_face_bil", 0.0)) / face
                    if face > 1e-12
                    else math.nan
                ),
                **bucket["snapshot"],
            }
            rows.append(row)
        return rows

    def _check_key_budget(self) -> None:
        cardinality = sum(
            len(channel._groups) for channel in self._channels.values()
        )
        self._max_key_cardinality = max(self._max_key_cardinality, cardinality)
        if cardinality > self.limits.key_cardinality_budget:
            raise ResourceLimitError(
                "bounded key-cardinality budget exceeded: "
                f"observed={cardinality}, budget={self.limits.key_cardinality_budget}"
            )

    def _observe_resources(
        self,
        *,
        sample_kind: str,
        period_end: str,
        portfolio_rows: int,
        active_portfolio_rows: int,
        enforce_abort: bool,
    ) -> None:
        portfolio_rows = int(portfolio_rows)
        active_portfolio_rows = int(active_portfolio_rows)
        self._max_portfolio_rows = max(self._max_portfolio_rows, portfolio_rows)
        self._max_active_portfolio_rows = max(
            self._max_active_portfolio_rows, active_portfolio_rows
        )
        if portfolio_rows > self.limits.portfolio_row_budget:
            raise ResourceLimitError(
                "portfolio-row budget exceeded: "
                f"observed={portfolio_rows}, budget={self.limits.portfolio_row_budget}"
            )
        rss = int(self._rss_reader())
        if rss <= 0:
            raise ResourceLimitError(
                f"process RSS sampler returned an invalid value: {rss!r}"
            )
        available = int(self._available_reader())
        if available < 0:
            raise ResourceLimitError(
                f"host-memory sampler returned an invalid value: {available!r}"
            )
        cpu_seconds = float(self._cpu_reader())
        if not math.isfinite(cpu_seconds) or cpu_seconds < 0.0:
            raise ResourceLimitError(
                f"process CPU sampler returned an invalid value: {cpu_seconds!r}"
            )
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        if (
            enforce_abort
            and self.limits.application_abort_rss_bytes
            and rss >= self.limits.application_abort_rss_bytes
        ):
            raise ResourceLimitError(
                "application RSS hard stop crossed: "
                f"rss={rss}, limit={self.limits.application_abort_rss_bytes}"
            )
        self._writers["resources"].write_rows(
            [
                {
                    "sample_kind": sample_kind,
                    "period_end": period_end,
                    "rss_bytes": rss,
                    "peak_rss_bytes": self._peak_rss_bytes,
                    "host_available_bytes": available,
                    "process_cpu_seconds": cpu_seconds,
                    "portfolio_rows": portfolio_rows,
                    "active_portfolio_rows": active_portfolio_rows,
                    "event_count": self._event_seq,
                    "current_key_cardinality": sum(
                        len(channel._groups)
                        for channel in self._channels.values()
                    ),
                    "max_key_cardinality": self._max_key_cardinality,
                    "bytes_written": sum(
                        writer.path.stat().st_size
                        for writer in self._writers.values()
                        if writer.path.exists()
                    ),
                }
            ]
        )

    def _notify_progress(
        self,
        *,
        progress_state: str,
        admission_date: str | None,
        last_completed_period: str | None,
    ) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            {
                "progress_state": progress_state,
                "admission_date": admission_date,
                "last_completed_period": last_completed_period,
                "period_count": self._period_count,
                "event_count": self._event_seq,
                "event_root_sha256": self._whole_hash.hexdigest(),
                "peak_rss_bytes": self._peak_rss_bytes,
                "failure_invariant": None,
                "failure_key": None,
                "last_events": list(self._failure_ring),
            }
        )

    def _clear_period_channels(self) -> None:
        for channel in self._channels.values():
            channel.clear()

    def _restore_snapshot_rows(
        self, name: str, rows: list[Mapping[str, Any]]
    ) -> None:
        for row in rows:
            self._append(name, row)


def _validate_bounded_reducer_key(name: str, row: Mapping[str, Any]) -> None:
    """Fail before reduction when a categorical key is outside the live contract."""

    taxonomy = BOUNDED_HANDOFF_TAXONOMY.get(name)
    if taxonomy is None:
        raise BoundedEvidenceError(f"undeclared bounded handoff table: {name}")

    date_columns = (
        ("period_start", "period_end")
        if name
        in {
            "tdcsim_accounting_journal",
            "tdcsim_period_issuance_flows",
            "tdcsim_period_principal_flows",
            "tdcsim_period_payment_flows",
        }
        else ("date",)
    )
    for column in date_columns:
        value = str(row.get(column, "")).strip()
        if not value:
            raise BoundedEvidenceError(f"bounded reducer key has blank {column}")
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise BoundedEvidenceError(
                f"bounded reducer key has malformed {column}: {value!r}"
            ) from exc
        if pd.isna(parsed):
            raise BoundedEvidenceError(
                f"bounded reducer key has malformed {column}: {value!r}"
            )

    def require_pair(
        first: str,
        second: str,
        domain_name: str,
        *,
        label: str,
    ) -> tuple[str, str]:
        pair = (str(row.get(first, "")), str(row.get(second, "")))
        if pair not in taxonomy[domain_name]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy {label}: {pair!r}"
            )
        return pair

    if name == "tdcsim_accounting_journal":
        archetype = tuple(
            str(row.get(column, ""))
            for column in (
                "event_type",
                "leg_type",
                "accounting_basis",
                "settlement_scope",
            )
        )
        if archetype not in taxonomy["core_archetypes"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy accounting archetype: {archetype!r}"
            )
        require_pair(
            "holder_sector",
            "holder_subsector",
            "holder_pairs",
            label="journal holder pair",
        )
        require_pair(
            "counterparty_sector",
            "counterparty_subsector",
            "counterparty_pairs",
            label="journal counterparty pair",
        )
        require_pair(
            "route_holder_sector",
            "route_holder_subsector",
            "route_holder_pairs",
            label="journal route-holder pair",
        )
        require_pair(
            "instrument_type",
            "maturity_bucket",
            "instrument_maturity_pairs",
            label="journal instrument/maturity pair",
        )
        if not isinstance(row.get("is_intragovernmental"), bool):
            raise BoundedEvidenceError(
                "unknown bounded taxonomy is_intragovernmental: "
                f"{row.get('is_intragovernmental')!r}"
            )
        return

    if name in {
        "tdcsim_period_issuance_flows",
        "tdcsim_period_principal_flows",
        "tdcsim_period_payment_flows",
        "tdcsim_holder_stocks",
    }:
        require_pair(
            "holder_sector",
            "holder_subsector",
            "holder_pairs",
            label=f"{name} holder pair",
        )
    elif name == "tdcsim_tdc_principal_route_stocks":
        require_pair(
            "route_holder_sector",
            "route_holder_subsector",
            "holder_pairs",
            label=f"{name} route-holder pair",
        )

    if name not in {"tdcsim_debt_target_bridge", "tdcsim_scenario_metrics"}:
        instrument, maturity = require_pair(
            "instrument_type",
            "maturity_bucket",
            "instrument_maturity_pairs",
            label=f"{name} instrument/maturity pair",
        )
    else:
        instrument = maturity = ""

    if name == "tdcsim_period_principal_flows":
        redemption_type = str(row.get("redemption_type", ""))
        if redemption_type not in taxonomy["redemption_types"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy redemption_type: {redemption_type!r}"
            )
        require_pair(
            "tdc_principal_recipient_sector",
            "tdc_principal_recipient_subsector",
            "recipient_pairs",
            label="principal recipient pair",
        )
        recipient_basis = str(row.get("tdc_principal_recipient_basis", ""))
        if recipient_basis not in taxonomy["recipient_bases"]:
            raise BoundedEvidenceError(
                "unknown bounded taxonomy tdc_principal_recipient_basis: "
                f"{recipient_basis!r}"
            )
    elif name == "tdcsim_period_payment_flows":
        additive = row.get("is_additive_to_cash_total")
        if not isinstance(additive, bool):
            raise BoundedEvidenceError(
                "unknown bounded taxonomy is_additive_to_cash_total: "
                f"{additive!r}"
            )
        payment_archetype = (
            str(row.get("payment_type", "")),
            str(row.get("accounting_basis", "")),
            additive,
        )
        if payment_archetype not in taxonomy["payment_archetypes"]:
            raise BoundedEvidenceError(
                "unknown bounded taxonomy payment archetype: "
                f"{payment_archetype!r}"
            )
    elif name in {
        "tdcsim_holder_stocks",
        "tdcsim_tdc_principal_route_stocks",
    }:
        expected_valuation = (
            "tips_adjusted_principal" if instrument == "TIPS" else "face"
        )
        valuation = str(row.get("valuation_basis", ""))
        if (
            valuation not in taxonomy["valuation_bases"]
            or valuation != expected_valuation
        ):
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy valuation_basis: {valuation!r}"
            )
        scope = str(row.get("debt_scope", ""))
        if scope not in taxonomy["debt_scopes"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy debt_scope: {scope!r}"
            )
        if (
            scope == "controlled_public_marketable"
            and instrument not in PUBLIC_MARKETABLE_SECURITY_TYPES
        ):
            raise BoundedEvidenceError(
                "controlled public marketable stock has nonmarketable instrument"
            )
        if (
            name == "tdcsim_holder_stocks"
            and scope == "controlled_public_marketable"
            and str(row.get("holder_sector", "")) in INTRAGOV_HOLDERS
        ):
            raise BoundedEvidenceError(
                "controlled public marketable stock has intragovernmental holder"
            )
        allocation = str(row.get("allocation_method", ""))
        if allocation not in taxonomy["allocation_methods"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy allocation_method: {allocation!r}"
            )
        if name == "tdcsim_tdc_principal_route_stocks":
            route_basis = str(row.get("route_stock_basis", ""))
            if route_basis not in taxonomy["route_stock_bases"]:
                raise BoundedEvidenceError(
                    f"unknown bounded taxonomy route_stock_basis: {route_basis!r}"
                )
    elif name == "tdcsim_period_issuance_flows":
        issuance_leg = str(row.get("issuance_leg", ""))
        if issuance_leg not in taxonomy["issuance_legs"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy issuance_leg: {issuance_leg!r}"
            )
    elif name == "tdcsim_debt_target_bridge":
        categorical = tuple(
            str(row.get(column, ""))
            for column in (
                "funding_mode",
                "intragovernmental_treatment",
                "fed_held_treasury_treatment",
                "public_nonmarketable_treatment",
            )
        )
        if categorical not in taxonomy["categorical_tuples"]:
            raise BoundedEvidenceError(
                f"unknown bounded taxonomy debt-target bridge: {categorical!r}"
            )
    elif name == "tdcsim_scenario_metrics":
        cutoff = _finite_or_optional(
            row.get("short_maturity_cutoff_years"),
            column="short_maturity_cutoff_years",
        )
        if cutoff != taxonomy["short_maturity_cutoff_years"]:
            raise BoundedEvidenceError(
                "unknown bounded taxonomy short_maturity_cutoff_years: "
                f"{cutoff!r}"
            )


def _validated_journal_event(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for column in ("event_type", "leg_type", "accounting_basis"):
        value = str(normalized.get(column, "")).strip()
        if not value:
            raise BoundedEvidenceError(f"accounting event has blank {column}")
        normalized[column] = value
    if normalized["event_type"] not in _ALLOWED_EVENT_TYPES:
        raise BoundedEvidenceError(
            f"unknown accounting event_type: {normalized['event_type']}"
        )
    for column, allowed in (
        ("holder_sector", _ALLOWED_HOLDERS),
        ("route_holder_sector", _ALLOWED_HOLDERS),
        ("instrument_type", _ALLOWED_INSTRUMENTS),
        ("maturity_bucket", _ALLOWED_MATURITY_BUCKETS),
    ):
        value = str(normalized.get(column, ""))
        if value not in allowed:
            raise BoundedEvidenceError(f"unknown accounting taxonomy {column}: {value}")
        normalized[column] = value
    magnitude = 0.0
    for column in _JOURNAL_NUMERIC:
        value = _finite_or_optional(normalized.get(column), column=column)
        if value is None:
            raise BoundedEvidenceError(f"accounting event is missing {column}")
        normalized[column] = value
        magnitude = max(magnitude, abs(value))
    if magnitude <= 1e-12:
        raise BoundedEvidenceError("all-zero accounting event is not admissible")
    for column in (
        "period_start",
        "period_end",
        "holder_subsector",
        "counterparty_sector",
        "counterparty_subsector",
        "route_holder_subsector",
        "settlement_scope",
    ):
        normalized[column] = str(normalized.get(column, ""))
    intragovernmental = normalized.get("is_intragovernmental")
    if not isinstance(intragovernmental, bool):
        raise BoundedEvidenceError(
            "accounting event has malformed is_intragovernmental"
        )
    normalized["is_intragovernmental"] = intragovernmental
    return normalized


def _encode_event(sequence: int, row: Mapping[str, Any]) -> bytes:
    parts = [EVENT_SCHEMA_VERSION.encode("utf-8"), struct.pack(">Q", sequence)]
    for column in _JOURNAL_KEYS:
        value = row.get(column)
        if isinstance(value, bool):
            parts.append(b"\x01" if value else b"\x00")
        else:
            encoded = str(value).encode("utf-8")
            parts.append(struct.pack(">I", len(encoded)))
            parts.append(encoded)
    for column in _JOURNAL_NUMERIC:
        value = float(row[column])
        if value == 0.0:
            value = 0.0
        parts.append(struct.pack(">d", value))
    return b"".join(parts)


def _stock_closure_rows(
    raw: Mapping[str, Any], period_start: str, period_end: str
) -> list[dict[str, Any]]:
    stocks = list(raw.get("tdcsim_holder_stocks", []))
    journal = list(raw.get("tdcsim_accounting_journal", []))
    opening_rows = [row for row in stocks if str(row.get("date")) == period_start]
    closing_rows = [row for row in stocks if str(row.get("date")) == period_end]
    rows: list[dict[str, Any]] = []
    for axis in ("holder", "instrument"):
        opening = _stock_map(opening_rows, axis=axis)
        closing = _stock_map(closing_rows, axis=axis)
        changes = _journal_stock_map(journal, axis=axis)
        for key in sorted(set(opening) | set(closing) | set(changes)):
            open_values = opening.get(key, (0.0, 0.0, 0.0))
            close_values = closing.get(key, (0.0, 0.0, 0.0))
            change_values = changes.get(key, (0.0, 0.0, 0.0))
            holder, subbucket, instrument, maturity, scope = key
            rows.append(
                {
                    "period_start": period_start,
                    "period_end": period_end,
                    "axis": axis,
                    "holder_sector": holder,
                    "holder_subsector": subbucket,
                    "instrument_type": instrument,
                    "maturity_bucket": maturity,
                    "debt_scope": scope,
                    "opening_face_stock_bil": open_values[0],
                    "event_face_stock_change_bil": change_values[0],
                    "closing_face_stock_bil": close_values[0],
                    "face_stock_closure_error_bil": close_values[0]
                    - open_values[0]
                    - change_values[0],
                    "opening_adjusted_principal_stock_bil": open_values[1],
                    "event_adjusted_principal_change_bil": change_values[1],
                    "closing_adjusted_principal_stock_bil": close_values[1],
                    "adjusted_principal_closure_error_bil": close_values[1]
                    - open_values[1]
                    - change_values[1],
                    "opening_debt_stock_bil": open_values[2],
                    "event_debt_stock_change_bil": change_values[2],
                    "closing_debt_stock_bil": close_values[2],
                    "debt_stock_closure_error_bil": close_values[2]
                    - open_values[2]
                    - change_values[2],
                }
            )
    return rows


def _stock_map(
    rows: list[Mapping[str, Any]], *, axis: str
) -> dict[tuple[str, str, str, str, str], tuple[float, float, float]]:
    result: dict[tuple[str, str, str, str, str], list[_Kahan]] = {}
    for row in rows:
        holder = str(row.get("holder_sector", "")) if axis == "holder" else ""
        subbucket = (
            str(row.get("holder_subsector", "")) if axis == "holder" else ""
        )
        key = (
            holder,
            subbucket,
            str(row.get("instrument_type", "")),
            str(row.get("maturity_bucket", "")),
            str(row.get("debt_scope", "")),
        )
        sums = result.setdefault(key, [_Kahan(), _Kahan(), _Kahan()])
        sums[0].add(float(row.get("face_stock_bil", 0.0) or 0.0))
        sums[1].add(
            float(row.get("adjusted_principal_stock_bil", 0.0) or 0.0)
        )
        sums[2].add(float(row.get("debt_held_bil", 0.0) or 0.0))
    return {key: tuple(item.total for item in sums) for key, sums in result.items()}


def _journal_stock_map(
    rows: list[Mapping[str, Any]], *, axis: str
) -> dict[tuple[str, str, str, str, str], tuple[float, float, float]]:
    result: dict[tuple[str, str, str, str, str], list[_Kahan]] = {}
    for row in rows:
        instrument = str(row.get("instrument_type", ""))
        intragov = bool(row.get("is_intragovernmental", False))
        scopes = ["all_active_treasury"]
        if instrument in {"Fixed", "TIPS", "FRN"} and not intragov:
            scopes.append("controlled_public_marketable")
        for scope in scopes:
            holder = str(row.get("holder_sector", "")) if axis == "holder" else ""
            subbucket = (
                str(row.get("holder_subsector", "")) if axis == "holder" else ""
            )
            key = (
                holder,
                subbucket,
                instrument,
                str(row.get("maturity_bucket", "")),
                scope,
            )
            sums = result.setdefault(key, [_Kahan(), _Kahan(), _Kahan()])
            face = float(row.get("face_stock_change_bil", 0.0) or 0.0)
            adjusted = float(
                row.get("adjusted_principal_change_bil", 0.0) or 0.0
            )
            debt = adjusted if instrument == "TIPS" else face
            sums[0].add(face)
            sums[1].add(adjusted)
            sums[2].add(debt)
    return {key: tuple(item.total for item in sums) for key, sums in result.items()}


def _financing_row(
    closing: Mapping[str, Any],
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    interest = _required_finite(closing, "InterestOutlay_Period")
    discount = _required_finite(closing, "IssueDiscountCost_Period")
    nonmarketable = _required_finite(
        closing, "NonMarketableInterestCapitalized_Period"
    )
    tips = _required_finite(closing, "TIPSInflationAccretion_Period")
    engine_total = _required_finite(closing, "FinancingCost_Period")
    total_reducer = _Kahan()
    for value in (interest, discount, nonmarketable, tips):
        total_reducer.add(value)
    derived_total = total_reducer.total
    return {
        "period_start": period_start,
        "period_end": period_end,
        "modeled_financing_cost_bil": derived_total,
        "interest_outlay_bil": interest,
        "issue_discount_cost_bil": discount,
        "nonmarketable_interest_capitalized_bil": nonmarketable,
        "tips_inflation_accretion_bil": tips,
        "component_identity_error_bil": engine_total - derived_total,
    }


def _final_state_digest(
    result: Mapping[str, Any],
    holder_rows: list[Mapping[str, Any]],
    route_rows: list[Mapping[str, Any]],
) -> str:
    payload = {
        "result": {
            str(key): _json_scalar(value)
            for key, value in sorted(result.items(), key=lambda item: str(item[0]))
        },
        "holder_stocks": [
            {
                str(key): _json_scalar(value)
                for key, value in sorted(row.items())
            }
            for row in sorted(
                holder_rows,
                key=lambda row: json.dumps(
                    {str(k): _json_scalar(v) for k, v in row.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ],
        "route_stocks": [
            {
                str(key): _json_scalar(value)
                for key, value in sorted(row.items())
            }
            for row in sorted(
                route_rows,
                key=lambda row: json.dumps(
                    {str(k): _json_scalar(v) for k, v in row.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def current_rss_bytes() -> int:
    """Return current-process resident bytes without adding a runtime dependency."""

    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        ok = get_process_memory_info(
            get_current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.WorkingSetSize)
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return usage if platform.system() == "Darwin" else usage * 1024
    except (ImportError, OSError):
        return 0


def host_available_memory_bytes() -> int:
    """Return host-available memory, or zero when the platform cannot report it."""

    if platform.system() == "Darwin":
        return _darwin_available_memory_bytes()
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullAvailPhys)
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return 0


def _darwin_available_memory_bytes() -> int:
    """Read macOS available pages without spawning ``vm_stat`` per period."""

    natural_t = ctypes.c_uint
    integer_t = ctypes.c_int
    mach_port_t = ctypes.c_uint
    mach_msg_type_number_t = ctypes.c_uint

    class VmStatistics64(ctypes.Structure):
        _fields_ = [
            ("free_count", natural_t),
            ("active_count", natural_t),
            ("inactive_count", natural_t),
            ("wire_count", natural_t),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", natural_t),
            ("speculative_count", natural_t),
            ("decompressions", ctypes.c_uint64),
            ("compressions", ctypes.c_uint64),
            ("swapins", ctypes.c_uint64),
            ("swapouts", ctypes.c_uint64),
            ("compressor_page_count", natural_t),
            ("throttled_count", natural_t),
            ("external_page_count", natural_t),
            ("internal_page_count", natural_t),
            ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
        ]

    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        mach_host_self = libsystem.mach_host_self
        mach_host_self.argtypes = []
        mach_host_self.restype = mach_port_t
        host_page_size = libsystem.host_page_size
        host_page_size.argtypes = [
            mach_port_t,
            ctypes.POINTER(natural_t),
        ]
        host_page_size.restype = integer_t
        host_statistics64 = libsystem.host_statistics64
        host_statistics64.argtypes = [
            mach_port_t,
            integer_t,
            ctypes.POINTER(integer_t),
            ctypes.POINTER(mach_msg_type_number_t),
        ]
        host_statistics64.restype = integer_t

        host = mach_host_self()
        page_size = natural_t()
        if host_page_size(host, ctypes.byref(page_size)) != 0:
            return 0
        statistics = VmStatistics64()
        count = mach_msg_type_number_t(
            ctypes.sizeof(statistics) // ctypes.sizeof(integer_t)
        )
        values = ctypes.cast(
            ctypes.byref(statistics),
            ctypes.POINTER(integer_t),
        )
        if host_statistics64(host, 4, values, ctypes.byref(count)) != 0:
            return 0
        return _darwin_available_bytes_from_counts(
            page_size=int(page_size.value),
            free_count=int(statistics.free_count),
            inactive_count=int(statistics.inactive_count),
            speculative_count=int(statistics.speculative_count),
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return 0


def _darwin_available_bytes_from_counts(
    *,
    page_size: int,
    free_count: int,
    inactive_count: int,
    speculative_count: int,
) -> int:
    """Match macOS ``vm_stat`` available-memory page accounting."""

    values = (page_size, free_count, inactive_count, speculative_count)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        return 0
    return page_size * (free_count + inactive_count + speculative_count)


def _finite_or_optional(value: Any, *, column: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundedEvidenceError(f"malformed numeric {column}: {value!r}") from exc
    if not math.isfinite(number):
        raise BoundedEvidenceError(f"nonfinite numeric {column}: {value!r}")
    return 0.0 if number == 0.0 else number


def _required_finite(values: Mapping[str, Any], column: str) -> float:
    number = _finite_or_optional(values.get(column), column=column)
    if number is None:
        raise BoundedEvidenceError(f"missing required numeric {column}")
    return number


def _snapshot_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text in {"<NA>", "nan", "None"} else text


def _snapshot_required_number(value: Any, column: str) -> float:
    if value is None or pd.isna(value):
        raise BoundedEvidenceError(
            f"live portfolio snapshot is missing numeric {column}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BoundedEvidenceError(
            f"live portfolio snapshot has malformed {column}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise BoundedEvidenceError(
            f"live portfolio snapshot has nonfinite {column}: {value!r}"
        )
    return 0.0 if number == 0.0 else number


def _snapshot_optional_number(
    value: Any,
    *,
    default: float,
    column: str,
) -> float:
    if value is None or pd.isna(value):
        return default
    return _snapshot_required_number(value, column)


def _snapshot_maturity_bucket(
    security: str,
    original_maturity_years: Any,
    maturity_category: Any,
) -> str:
    if security == "Fixed" and not (
        maturity_category is None or pd.isna(maturity_category)
    ):
        category = str(maturity_category).strip()
        if category:
            if category not in PREFERENCE_CATEGORIES:
                raise BoundedEvidenceError(
                    "live portfolio snapshot has noncanonical maturity category: "
                    f"{category!r}"
                )
            return category
    try:
        years = float(original_maturity_years)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(years):
        return "unknown"
    if years <= 1.0 + TGA_FLOOR_TOLERANCE:
        return "bills"
    if years <= 10.0 + TGA_FLOOR_TOLERANCE:
        return "notes"
    return "bonds"


def _snapshot_route(
    *,
    holder: str,
    holder_subbucket: str,
    route_holder: Any,
    route_subbucket: Any,
) -> tuple[str, str]:
    if holder == "CB":
        return "CB", ""
    routed_holder = _snapshot_text(route_holder)
    if routed_holder:
        explicit_subbucket = _snapshot_text(route_subbucket)
        routed_subbucket = explicit_subbucket or (
            holder_subbucket if routed_holder == holder else ""
        )
    else:
        routed_holder = holder
        routed_subbucket = holder_subbucket
    _validate_snapshot_subbucket(
        routed_holder,
        routed_subbucket,
        label="TDC principal route",
    )
    return routed_holder, routed_subbucket


def _validate_snapshot_subbucket(
    holder: str,
    subbucket: str,
    *,
    label: str,
) -> None:
    """Reject undeclared categorical values before they become aggregate keys."""

    if holder == "Private":
        if subbucket not in PRIVATE_SUBBUCKETS:
            raise BoundedEvidenceError(
                f"live portfolio snapshot has invalid Private {label} subbucket: "
                f"{subbucket!r}"
            )
        return
    if subbucket:
        raise BoundedEvidenceError(
            f"live portfolio snapshot has nonblank {label} subbucket for "
            f"{holder!r}: {subbucket!r}"
        )


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if abs(float(actual) - float(expected)) > STOCK_TOLERANCE_BIL:
        raise BoundedEvidenceError(
            f"{label} failed: actual={actual!r}, expected={expected!r}"
        )


def _normalized_key(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _csv_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(0.0 if value == 0.0 else value, ".17g")
    return value


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return 0.0 if value == 0.0 else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        number = float(value)
        if math.isfinite(number):
            return 0.0 if number == 0.0 else number
    except (TypeError, ValueError):
        pass
    return str(value)


def _artifact(root: Path, path: Path, row_count: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_count": int(row_count),
    }


__all__ = [
    "AGGREGATION_CLOCK_ID",
    "BoundedEvidenceError",
    "BoundedResourceLimits",
    "BoundedScenarioEvidenceSink",
    "EVIDENCE_PROFILE",
    "EVENT_SCHEMA_VERSION",
    "ResourceLimitError",
    "VERIFICATION_GRADE",
    "current_rss_bytes",
    "host_available_memory_bytes",
]
