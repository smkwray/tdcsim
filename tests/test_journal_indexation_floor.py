"""A small TIPS lot's daily indexation must reach the accounting journal.

The results count every lot's indexation as TIPS accretion. If the journal dropped the
small ones, a year of a few slivers' indexation would separate the annual accretion in the
results from the journal the verifier rebuilds it from.
"""

import pandas as pd

import sim_engine
from tdc_shared import JOURNAL_MIN_EVENT_BIL, TGA_FLOOR_TOLERANCE


def _append(tables, amount, **extra):
    sim_engine._handoff_append_journal(
        tables, pd.Timestamp("2026-06-21"), pd.Timestamp("2026-06-22"),
        event_type="indexation", leg_type="tips_adjusted_principal_indexation",
        accounting_basis="adjusted_principal_stock", holder_sector="CB", route_holder_sector="CB",
        instrument_type="TIPS", maturity_bucket="notes", adjusted_principal_change=amount,
        settlement_scope="noncash_stock_remeasurement", **extra,
    )


def test_indexation_below_the_cash_floor_is_journaled_down_to_the_ledger_floor():
    tiny = TGA_FLOOR_TOLERANCE / 2
    assert tiny > JOURNAL_MIN_EVENT_BIL
    tables = {}
    _append(tables, tiny, drop_below=JOURNAL_MIN_EVENT_BIL)
    assert [row["adjusted_principal_change_bil"] for row in tables["tdcsim_accounting_journal"]] == [tiny]
    # The default floor still drops cash-sized dust from other events.
    other = {}
    _append(other, tiny)
    assert "tdcsim_accounting_journal" not in other
