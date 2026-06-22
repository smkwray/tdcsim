"""Quarterly event ledger for historical replay."""

from __future__ import annotations

import pandas as pd

from historical_replay_contract import filter_quarter_range, normalize_quarter_value


EVENT_LEDGER_COLUMNS = [
    "event_id",
    "event_date",
    "quarter",
    "security_id",
    "cohort_id",
    "cusip",
    "lot_id",
    "event_type",
    "holder",
    "tdcsim_holder",
    "par_delta_mil",
    "current_principal_delta_mil",
    "cash_amount_mil",
    "price_per100",
    "accrued_per100",
    "coupon_rate_decimal",
    "frn_index_rate_decimal",
    "frn_fixed_spread_decimal",
    "tips_index_ratio",
    "auction_class",
    "source_file",
    "source_row_key",
    "evidence_status",
    "derivation",
    "reconciliation_group",
]

ROLLFORWARD_COLUMNS = [
    "quarter",
    "cohort_id",
    "cusip",
    "opening_balance_mil",
    "source_issue_mil",
    "source_redemption_mil",
    "source_indexation_mil",
    "source_reclassification_mil",
    "unexplained_cohort_change_mil",
    "unexplained_residual_change_mil",
    "ending_balance_mil",
    "rollforward_residual_mil",
    "status",
]

UNEXPLAINED_CHANGE_COLUMNS = [
    "quarter",
    "cohort_id",
    "cusip",
    "event_type",
    "unexplained_cohort_change_mil",
    "unexplained_residual_change_mil",
    "evidence_status",
    "derivation",
]

_AMOUNT_TO_MILLIONS = 1_000_000.0


def build_historical_replay_event_ledger(
    cohorts: pd.DataFrame,
    auction_allotment_proxy: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build a source and modeled event ledger for replay reconciliation."""

    frames = [
        _auction_issue_events(
            auction_allotment_proxy,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _cohort_issuance_events(
            cohorts,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _cohort_redemption_events(
            cohorts,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _cohort_indexation_events(
            cohorts,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _cohort_terminal_exit_events(
            cohorts,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
        _cohort_reclassification_events(
            cohorts,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
        ),
    ]
    ledger = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if ledger.empty:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    ledger["event_id"] = [f"evt_{idx:09d}" for idx in range(1, len(ledger.index) + 1)]
    return ledger.loc[:, EVENT_LEDGER_COLUMNS].sort_values(
        ["quarter", "event_date", "event_type", "cohort_id", "source_row_key"],
        na_position="last",
    ).reset_index(drop=True)


def build_event_rollforward(
    cohorts: pd.DataFrame,
    event_ledger: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconcile quarter-end cohorts to source and modeled events."""

    if cohorts.empty:
        empty_roll = pd.DataFrame(columns=ROLLFORWARD_COLUMNS)
        empty_unexplained = pd.DataFrame(columns=UNEXPLAINED_CHANGE_COLUMNS)
        return event_ledger.copy(), empty_roll, empty_unexplained
    cohort_balances = _cohort_balance_panel(
        cohorts,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    source_events = _event_sums(event_ledger)
    old_opening_split_keys = _old_opening_split_residual_keys(event_ledger)
    rows = []
    unexplained_rows = []
    residual_event_rows = []
    for _, row in cohort_balances.iterrows():
        quarter = row["quarter"]
        cohort_id = row["cohort_id"]
        key = (quarter, cohort_id)
        opening = float(row["opening_balance_mil"])
        ending = float(row["ending_balance_mil"])
        events = source_events.get(
            key,
            {
                "source_issue_mil": 0.0,
                "source_redemption_mil": 0.0,
                "source_indexation_mil": 0.0,
                "source_reclassification_mil": 0.0,
            },
        )
        unexplained_change = (
            ending
            - opening
            - float(events["source_issue_mil"])
            - float(events["source_redemption_mil"])
            - float(events["source_indexation_mil"])
            - float(events["source_reclassification_mil"])
        )
        roll_residual = (
            opening
            + float(events["source_issue_mil"])
                + float(events["source_redemption_mil"])
                + float(events["source_indexation_mil"])
                + float(events["source_reclassification_mil"])
                + unexplained_change
            - ending
        )
        old_opening_split_residual = _is_old_opening_split_residual(
            row,
            quarter=quarter,
            old_opening_split_keys=old_opening_split_keys,
        )
        if abs(unexplained_change) > 1e-9 and old_opening_split_residual:
            rollforward_status = "closed_with_old_reopening_initial_split_residual"
            unexplained_evidence_status = "unexplained_old_reopening_initial_split_residual"
            unexplained_derivation = "old_reopening_initial_split_counterpart_left_unexplained"
            residual_event_derivation = "old_reopening_initial_split_counterpart_needed_to_close_rollforward"
        elif abs(unexplained_change) > 1e-9:
            rollforward_status = "closed_with_unexplained_change"
            unexplained_evidence_status = "unexplained_residual"
            unexplained_derivation = "ending_balance_minus_opening_and_source_events"
            residual_event_derivation = "unexplained_change_needed_to_close_quarter_end_rollforward"
        else:
            rollforward_status = "closed_by_source_events"
            unexplained_evidence_status = "unexplained_residual"
            unexplained_derivation = "ending_balance_minus_opening_and_source_events"
            residual_event_derivation = "unexplained_change_needed_to_close_quarter_end_rollforward"
        rows.append(
            {
                "quarter": quarter,
                "cohort_id": cohort_id,
                "cusip": row.get("cusip", pd.NA),
                "opening_balance_mil": opening,
                "source_issue_mil": float(events["source_issue_mil"]),
                "source_redemption_mil": float(events["source_redemption_mil"]),
                "source_indexation_mil": float(events["source_indexation_mil"]),
                "source_reclassification_mil": float(events["source_reclassification_mil"]),
                "unexplained_cohort_change_mil": unexplained_change,
                "unexplained_residual_change_mil": unexplained_change,
                "ending_balance_mil": ending,
                "rollforward_residual_mil": roll_residual,
                "status": rollforward_status,
            }
        )
        if abs(unexplained_change) > 1e-9:
            unexplained_rows.append(
                {
                    "quarter": quarter,
                    "cohort_id": cohort_id,
                    "cusip": row.get("cusip", pd.NA),
                    "event_type": "unexplained_cohort_change",
                    "unexplained_cohort_change_mil": unexplained_change,
                    "unexplained_residual_change_mil": unexplained_change,
                    "evidence_status": unexplained_evidence_status,
                    "derivation": unexplained_derivation,
                }
            )
            residual_event_rows.append(
                _event_row(
                    event_date=_quarter_end_date(quarter),
                    quarter=quarter,
                    security_id=row.get("cusip", pd.NA),
                    cohort_id=cohort_id,
                    cusip=row.get("cusip", pd.NA),
                    lot_id=f"{cohort_id}|{quarter}|unexplained_cohort_change",
                    event_type="unexplained_cohort_change",
                    holder="residual_unallocated",
                    tdcsim_holder="residual_unallocated",
                    par_delta_mil=unexplained_change,
                    current_principal_delta_mil=unexplained_change,
                    cash_amount_mil=0.0,
                    source_file="mspd_quarter_end_cohort_rollforward",
                    source_row_key=f"{quarter}|{cohort_id}|unexplained_cohort_change",
                    evidence_status=unexplained_evidence_status,
                    derivation=residual_event_derivation,
                    reconciliation_group="cohort_rollforward",
                )
            )
    rollforward = pd.DataFrame(rows, columns=ROLLFORWARD_COLUMNS)
    complete_ledger = event_ledger.copy()
    if residual_event_rows:
        complete_ledger = pd.concat(
            [complete_ledger, pd.DataFrame(residual_event_rows, columns=EVENT_LEDGER_COLUMNS)],
            ignore_index=True,
        )
        complete_ledger["event_id"] = [f"evt_{idx:09d}" for idx in range(1, len(complete_ledger.index) + 1)]
    unexplained = pd.DataFrame(unexplained_rows, columns=UNEXPLAINED_CHANGE_COLUMNS)
    return complete_ledger.loc[:, EVENT_LEDGER_COLUMNS], rollforward, unexplained


def _auction_issue_events(
    auction_allotment_proxy: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if auction_allotment_proxy.empty:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    required = {"quarter", "cusip", "issue_date", "maturity_date", "allotment_amount"}
    if not required.issubset(auction_allotment_proxy.columns):
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = auction_allotment_proxy.copy()
    frame = filter_quarter_range(frame, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["allotment_amount"] = pd.to_numeric(frame["allotment_amount"], errors="coerce").fillna(0.0)
    rows = []
    for idx, row in frame.iterrows():
        amount_mil = float(row["allotment_amount"]) / _AMOUNT_TO_MILLIONS
        cohort_id = _cohort_id(row)
        holder = _auction_holder(row.get("broad_investor_class"))
        rows.append(
            _event_row(
                event_date=row.get("issue_date"),
                quarter=row.get("quarter"),
                security_id=row.get("cusip"),
                cohort_id=cohort_id,
                cusip=row.get("cusip"),
                lot_id=f"{cohort_id}|{row.get('raw_investor_class', idx)}|issue",
                event_type="auction_issue",
                holder=row.get("broad_investor_class", holder),
                tdcsim_holder=holder,
                par_delta_mil=amount_mil,
                current_principal_delta_mil=amount_mil,
                cash_amount_mil=-amount_mil,
                coupon_rate_decimal=_percent_or_decimal(row.get("coupon_rate")),
                auction_class=row.get("raw_investor_class", pd.NA),
                source_file="buycurve_auction_allotment_panel",
                source_row_key="|".join(
                    str(row.get(col, ""))
                    for col in ["cusip", "auction_date", "issue_date", "raw_investor_class"]
                ),
                evidence_status="source_observed",
                derivation="auction_allotment_amount_dollars_to_millions",
                reconciliation_group="auction_allotment",
            )
        )
        maturity_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        issue_date = pd.to_datetime(row.get("issue_date"), errors="coerce")
        if pd.notna(maturity_date) and pd.notna(issue_date):
            issue_quarter = normalize_quarter_value(issue_date)
            maturity_quarter = normalize_quarter_value(maturity_date)
            if issue_quarter == maturity_quarter:
                rows.append(
                    _event_row(
                        event_date=maturity_date.strftime("%Y-%m-%d"),
                        quarter=maturity_quarter,
                        security_id=row.get("cusip"),
                        cohort_id=cohort_id,
                        cusip=row.get("cusip"),
                        lot_id=f"{cohort_id}|{row.get('raw_investor_class', idx)}|same_quarter_maturity",
                        event_type="maturity_redemption",
                        holder=row.get("broad_investor_class", holder),
                        tdcsim_holder=holder,
                        par_delta_mil=-amount_mil,
                        current_principal_delta_mil=-amount_mil,
                        cash_amount_mil=amount_mil,
                        coupon_rate_decimal=_percent_or_decimal(row.get("coupon_rate")),
                        auction_class=row.get("raw_investor_class", pd.NA),
                        source_file="buycurve_auction_allotment_panel",
                        source_row_key="|".join(
                            str(row.get(col, ""))
                            for col in ["cusip", "auction_date", "maturity_date", "raw_investor_class"]
                        ),
                        evidence_status="source_inferred_from_auction_maturity_date",
                        derivation="same_quarter_issue_maturity_not_visible_in_period_end_cohorts",
                        reconciliation_group="transient_within_quarter_security",
                    )
                )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_issuance_events(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if cohorts.empty or "issued_amt" not in cohorts.columns:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = _cohort_event_frame(cohorts, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["issued_delta"] = _first_difference_stock_field(
        frame,
        "issued_amt",
        start_quarter=start_quarter,
        first_row_issue_only=True,
    )
    rows = []
    for _, row in frame.loc[frame["issued_delta"].abs() > 1e-12].iterrows():
        amount = max(float(row["issued_delta"]), 0.0)
        if amount <= 1e-12:
            continue
        rows.append(
            _event_row(
                event_date=row.get("issue_date", _quarter_end_date(row.get("quarter"))),
                quarter=row.get("quarter"),
                security_id=row.get("cusip", row.get("cohort_id")),
                cohort_id=row.get("cohort_id"),
                cusip=row.get("cusip", pd.NA),
                lot_id=f"{row.get('cohort_id')}|{row.get('quarter')}|mspd_issued_delta",
                event_type="mspd_issuance",
                holder="all_holders",
                tdcsim_holder="all_holders",
                par_delta_mil=amount,
                current_principal_delta_mil=amount,
                cash_amount_mil=-amount,
                coupon_rate_decimal=row.get("coupon_rate_decimal", pd.NA),
                source_file="mspd_table_3_market",
                source_row_key=f"{row.get('quarter')}|{row.get('cohort_id')}|issued_amt_delta",
                evidence_status="source_derived",
                derivation="first_difference_of_mspd_issued_amt",
                reconciliation_group="cohort_rollforward",
            )
        )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_redemption_events(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if cohorts.empty or "redeemed_amt" not in cohorts.columns:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = _cohort_event_frame(cohorts, start_quarter=start_quarter, end_quarter=end_quarter)
    frame["redeemed_delta"] = _first_difference_stock_field(
        frame,
        "redeemed_amt",
        start_quarter=start_quarter,
        first_row_issue_only=False,
    )
    rows = []
    for _, row in frame.loc[frame["redeemed_delta"].abs() > 1e-12].iterrows():
        amount = min(float(row["redeemed_delta"]), 0.0)
        if amount >= -1e-12:
            continue
        rows.append(
            _event_row(
                event_date=_quarter_end_date(row.get("quarter")),
                quarter=row.get("quarter"),
                security_id=row.get("cusip", row.get("cohort_id")),
                cohort_id=row.get("cohort_id"),
                cusip=row.get("cusip", pd.NA),
                lot_id=f"{row.get('cohort_id')}|{row.get('quarter')}|mspd_redeemed_delta",
                event_type="principal_redemption",
                holder="all_holders",
                tdcsim_holder="all_holders",
                par_delta_mil=amount,
                current_principal_delta_mil=amount,
                cash_amount_mil=-amount,
                coupon_rate_decimal=row.get("coupon_rate_decimal", pd.NA),
                source_file="mspd_table_3_market",
                source_row_key=f"{row.get('quarter')}|{row.get('cohort_id')}|redeemed_amt_delta",
                evidence_status="source_derived",
                derivation="first_difference_of_mspd_redeemed_amt",
                reconciliation_group="cohort_rollforward",
            )
        )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_indexation_events(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if cohorts.empty or "inflation_adj_amt" not in cohorts.columns:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = _cohort_event_frame(cohorts, start_quarter=start_quarter, end_quarter=end_quarter)
    security_text = frame.get("security_type", pd.Series("", index=frame.index)).astype(str).str.lower()
    frame = frame.loc[security_text.str.contains("tips|inflation", regex=True, na=False)].copy()
    if frame.empty:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame["inflation_delta"] = _first_difference_stock_field(
        frame,
        "inflation_adj_amt",
        start_quarter=start_quarter,
        first_row_issue_only=False,
    )
    rows = []
    for _, row in frame.loc[frame["inflation_delta"].abs() > 1e-12].iterrows():
        amount = float(row["inflation_delta"])
        rows.append(
            _event_row(
                event_date=_quarter_end_date(row.get("quarter")),
                quarter=row.get("quarter"),
                security_id=row.get("cusip", row.get("cohort_id")),
                cohort_id=row.get("cohort_id"),
                cusip=row.get("cusip", pd.NA),
                lot_id=f"{row.get('cohort_id')}|{row.get('quarter')}|tips_indexation",
                event_type="tips_principal_indexation",
                holder="all_holders",
                tdcsim_holder="all_holders",
                par_delta_mil=0.0,
                current_principal_delta_mil=amount,
                cash_amount_mil=0.0,
                coupon_rate_decimal=row.get("coupon_rate_decimal", pd.NA),
                tips_index_ratio=row.get("index_ratio", row.get("tips_index_ratio", pd.NA)),
                source_file="mspd_table_3_market",
                source_row_key=f"{row.get('quarter')}|{row.get('cohort_id')}|inflation_adj_amt_delta",
                evidence_status="source_derived",
                derivation="first_difference_of_mspd_inflation_adj_amt",
                reconciliation_group="cohort_rollforward",
            )
        )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_reclassification_events(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if cohorts.empty or "outstanding" not in cohorts.columns:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = _cohort_event_frame(cohorts, start_quarter=start_quarter, end_quarter=end_quarter)
    source_status = frame.get("source_status", pd.Series("", index=frame.index)).astype(str).str.lower()
    decomposed = source_status.str.contains("mspd_reopening_outstanding_decomposed", na=False)
    if not decomposed.any():
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    stock_delta, old_reopening_first_seen = _outstanding_delta_for_residual(
        frame,
        start_quarter=start_quarter,
    )
    issue_flow = _positive_first_difference_flow(
        frame,
        "issued_amt",
        start_quarter=start_quarter,
        first_row_issue_only=True,
    )
    redemption_flow = _negative_first_difference_flow(
        frame,
        "redeemed_amt",
        start_quarter=start_quarter,
    )
    indexation_flow = _tips_indexation_flow(
        frame,
        start_quarter=start_quarter,
    )
    candidate = stock_delta - issue_flow - redemption_flow - indexation_flow
    candidate = candidate.where(decomposed, 0.0).fillna(0.0)
    group_key = [
        frame["quarter"].astype(str),
        frame.get("cusip", pd.Series("", index=frame.index)).astype(str),
        frame.get("maturity_date", pd.Series("", index=frame.index)).astype(str),
    ]
    group_sum = candidate.groupby(group_key, sort=False).transform("sum")
    zero_sum_reclassification = group_sum.abs() <= 1.0e-6
    source_reclassification = decomposed & (zero_sum_reclassification | old_reopening_first_seen)
    frame["reclassification_delta"] = candidate.where(source_reclassification, 0.0).fillna(0.0)
    frame["reclassification_derivation"] = "mspd_reopening_residual_stock_shift_zero_sum"
    frame.loc[
        old_reopening_first_seen & frame["reclassification_delta"].abs().gt(1.0e-12),
        "reclassification_derivation",
    ] = "old_reopening_initial_split_residual_stock_shift"
    rows = []
    for _, row in frame.loc[frame["reclassification_delta"].abs() > 1e-12].iterrows():
        amount = float(row["reclassification_delta"])
        rows.append(
            _event_row(
                event_date=_quarter_end_date(row.get("quarter")),
                quarter=row.get("quarter"),
                security_id=row.get("cusip", row.get("cohort_id")),
                cohort_id=row.get("cohort_id"),
                cusip=row.get("cusip", pd.NA),
                lot_id=f"{row.get('cohort_id')}|{row.get('quarter')}|mspd_source_reclassification",
                event_type="mspd_source_reclassification",
                holder="all_holders",
                tdcsim_holder="all_holders",
                par_delta_mil=amount,
                current_principal_delta_mil=amount,
                cash_amount_mil=0.0,
                coupon_rate_decimal=row.get("coupon_rate_decimal", pd.NA),
                source_file="mspd_table_3_market",
                source_row_key=f"{row.get('quarter')}|{row.get('cohort_id')}|reopening_decomposition",
                evidence_status="source_derived",
                derivation=row.get("reclassification_derivation", "mspd_reopening_residual_stock_shift"),
                reconciliation_group="cohort_rollforward",
            )
        )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_terminal_exit_events(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if cohorts.empty or "outstanding" not in cohorts.columns:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame = filter_quarter_range(cohorts.copy(), start_quarter=start_quarter, end_quarter=end_quarter)
    if frame.empty:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)
    frame["_quarter_period"] = pd.PeriodIndex(frame["quarter"].astype(str), freq="Q")
    end_period = pd.Period(str(end_quarter), freq="Q") if end_quarter else frame["_quarter_period"].max()
    rows = []
    for cohort_id, group in frame.sort_values(["cohort_id", "_quarter_period"]).groupby(
        "cohort_id",
        sort=False,
        dropna=False,
    ):
        last = group.iloc[-1]
        terminal_period = last["_quarter_period"] + 1
        if terminal_period > end_period:
            continue
        current_principal = _terminal_current_principal(last)
        if current_principal <= 1.0e-12:
            continue
        par_principal = _terminal_par_principal(last, current_principal)
        event_type, event_date, evidence_status, derivation = _terminal_exit_classification(
            last,
            terminal_period,
        )
        rows.append(
            _event_row(
                event_date=event_date,
                quarter=str(terminal_period),
                security_id=last.get("cusip", cohort_id),
                cohort_id=cohort_id,
                cusip=last.get("cusip", pd.NA),
                lot_id=f"{cohort_id}|{terminal_period}|terminal_exit",
                event_type=event_type,
                holder="all_holders",
                tdcsim_holder="all_holders",
                par_delta_mil=-par_principal,
                current_principal_delta_mil=-current_principal,
                cash_amount_mil=current_principal,
                coupon_rate_decimal=last.get("coupon_rate_decimal", pd.NA),
                tips_index_ratio=last.get("IndexRatio", last.get("index_ratio", pd.NA)),
                source_file="mspd_table_3_market;auction_terms",
                source_row_key=f"{terminal_period}|{cohort_id}|terminal_exit",
                evidence_status=evidence_status,
                derivation=derivation,
                reconciliation_group="cohort_rollforward",
            )
        )
    return pd.DataFrame(rows, columns=EVENT_LEDGER_COLUMNS)


def _cohort_balance_panel(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    frame = filter_quarter_range(cohorts.copy(), start_quarter=start_quarter, end_quarter=end_quarter)
    frame = _append_terminal_zero_balance_rows(frame, end_quarter=end_quarter)
    frame["outstanding"] = pd.to_numeric(frame["outstanding"], errors="coerce").fillna(0.0)
    grouped = (
        frame.groupby(["quarter", "cohort_id"], sort=False, dropna=False)
        .agg(
            ending_balance_mil=("outstanding", "sum"),
            cusip=("cusip", _first_non_null),
            maturity_date=("maturity_date", _first_non_null),
        )
        .reset_index()
    )
    grouped["_quarter_period"] = pd.PeriodIndex(grouped["quarter"].astype(str), freq="Q")
    grouped = grouped.sort_values(["cohort_id", "_quarter_period"])
    grouped["opening_balance_mil"] = (
        grouped.groupby("cohort_id", sort=False)["ending_balance_mil"].shift(1).fillna(0.0)
    )
    initial_quarter = str(start_quarter) if start_quarter is not None else str(grouped["quarter"].min())
    first_for_cohort = ~grouped.duplicated(subset=["cohort_id"], keep="first")
    initial_mask = first_for_cohort & grouped["quarter"].astype(str).eq(initial_quarter)
    grouped.loc[initial_mask, "opening_balance_mil"] = grouped.loc[initial_mask, "ending_balance_mil"]
    return grouped.drop(columns=["_quarter_period"])


def _old_opening_split_residual_keys(event_ledger: pd.DataFrame) -> set[tuple[str, str, str]]:
    if event_ledger.empty or "derivation" not in event_ledger.columns:
        return set()
    frame = event_ledger.loc[
        event_ledger["event_type"].astype(str).eq("mspd_source_reclassification")
        & event_ledger["derivation"].astype(str).eq("old_reopening_initial_split_residual_stock_shift")
    ].copy()
    if frame.empty:
        return set()
    keys: set[tuple[str, str, str]] = set()
    for _, row in frame.iterrows():
        keys.add(
            (
                str(row.get("quarter", "")).strip(),
                str(row.get("cusip", "")).strip(),
                _cohort_maturity_from_id(row.get("cohort_id")),
            )
        )
    return keys


def _is_old_opening_split_residual(
    row: pd.Series,
    *,
    quarter: object,
    old_opening_split_keys: set[tuple[str, str, str]],
) -> bool:
    if not old_opening_split_keys:
        return False
    key = (
        str(quarter).strip(),
        str(row.get("cusip", "")).strip(),
        _normalize_date_text(row.get("maturity_date", _cohort_maturity_from_id(row.get("cohort_id")))),
    )
    return key in old_opening_split_keys


def _event_sums(event_ledger: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    if event_ledger.empty:
        return {}
    frame = event_ledger.copy()
    frame["par_delta_mil"] = pd.to_numeric(frame["par_delta_mil"], errors="coerce").fillna(0.0)
    frame["current_principal_delta_mil"] = pd.to_numeric(
        frame["current_principal_delta_mil"], errors="coerce"
    ).fillna(frame["par_delta_mil"])
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for (quarter, cohort_id), group in frame.groupby(["quarter", "cohort_id"], sort=False, dropna=False):
        issue = float(group.loc[group["event_type"].eq("mspd_issuance"), "par_delta_mil"].sum())
        redemption = float(
            group.loc[
                group["event_type"].isin(
                    [
                        "maturity_redemption",
                        "principal_redemption",
                        "called_redemption",
                        "source_discontinuity_exit",
                    ]
                ),
                "current_principal_delta_mil",
            ].sum()
        )
        indexation = float(
            group.loc[group["event_type"].eq("tips_principal_indexation"), "current_principal_delta_mil"].sum()
        )
        reclassification = float(
            group.loc[group["event_type"].eq("mspd_source_reclassification"), "current_principal_delta_mil"].sum()
        )
        rows[(str(quarter), str(cohort_id))] = {
            "source_issue_mil": issue,
            "source_redemption_mil": redemption,
            "source_indexation_mil": indexation,
            "source_reclassification_mil": reclassification,
        }
    return rows


def _cohort_event_frame(
    cohorts: pd.DataFrame,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    frame = filter_quarter_range(cohorts.copy(), start_quarter=start_quarter, end_quarter=end_quarter)
    for column in ["issued_amt", "redeemed_amt", "inflation_adj_amt"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["_quarter_period"] = pd.PeriodIndex(frame["quarter"].astype(str), freq="Q")
    return frame.sort_values(["cohort_id", "_quarter_period"]).reset_index(drop=True)


def _first_difference_stock_field(
    frame: pd.DataFrame,
    column: str,
    *,
    start_quarter: str | None,
    first_row_issue_only: bool,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    previous = values.groupby(frame["cohort_id"], sort=False).shift(1)
    delta = values - previous
    first_for_cohort = previous.isna()
    delta = delta.where(~first_for_cohort, 0.0)
    if first_row_issue_only and "issue_date" in frame.columns:
        issue_quarter = pd.to_datetime(frame["issue_date"], errors="coerce").dt.to_period("Q").astype(str)
        current_quarter = frame["quarter"].astype(str)
        start_text = str(start_quarter) if start_quarter is not None else current_quarter.min()
        first_issue_in_range = first_for_cohort & current_quarter.ne(start_text) & issue_quarter.eq(current_quarter)
        delta = delta.where(~first_issue_in_range, values)
    return delta.fillna(0.0)


def _positive_first_difference_flow(
    frame: pd.DataFrame,
    column: str,
    *,
    start_quarter: str | None,
    first_row_issue_only: bool,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return _first_difference_stock_field(
        frame,
        column,
        start_quarter=start_quarter,
        first_row_issue_only=first_row_issue_only,
    ).clip(lower=0.0)


def _negative_first_difference_flow(
    frame: pd.DataFrame,
    column: str,
    *,
    start_quarter: str | None,
) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return _first_difference_stock_field(
        frame,
        column,
        start_quarter=start_quarter,
        first_row_issue_only=False,
    ).clip(upper=0.0)


def _tips_indexation_flow(frame: pd.DataFrame, *, start_quarter: str | None) -> pd.Series:
    if "inflation_adj_amt" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    security_text = frame.get("security_type", pd.Series("", index=frame.index)).astype(str).str.lower()
    tips = security_text.str.contains("tips|inflation", regex=True, na=False)
    flow = _first_difference_stock_field(
        frame,
        "inflation_adj_amt",
        start_quarter=start_quarter,
        first_row_issue_only=False,
    )
    return flow.where(tips, 0.0).fillna(0.0)


def _outstanding_delta_for_residual(
    frame: pd.DataFrame,
    *,
    start_quarter: str | None,
) -> tuple[pd.Series, pd.Series]:
    outstanding = pd.to_numeric(frame["outstanding"], errors="coerce").fillna(0.0)
    previous = outstanding.groupby(frame["cohort_id"], sort=False).shift(1)
    delta = outstanding - previous
    first_for_cohort = previous.isna()
    issue_quarter = (
        pd.to_datetime(frame.get("issue_date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
        .dt.to_period("Q")
        .astype(str)
    )
    current_quarter = frame["quarter"].astype(str)
    start_text = str(start_quarter) if start_quarter is not None else current_quarter.min()
    first_issue_in_range = first_for_cohort & current_quarter.ne(start_text) & issue_quarter.eq(current_quarter)
    old_reopening_first_seen = first_for_cohort & current_quarter.ne(start_text) & issue_quarter.ne(current_quarter)
    delta = delta.where(~first_for_cohort, 0.0)
    delta = delta.where(~(first_issue_in_range | old_reopening_first_seen), outstanding)
    return delta.fillna(0.0), old_reopening_first_seen.fillna(False)


def _append_terminal_zero_balance_rows(
    frame: pd.DataFrame,
    *,
    end_quarter: str | None,
) -> pd.DataFrame:
    if frame.empty or "quarter" not in frame.columns or "cohort_id" not in frame.columns:
        return frame
    working = frame.copy()
    working["_quarter_period"] = pd.PeriodIndex(working["quarter"].astype(str), freq="Q")
    end_period = pd.Period(str(end_quarter), freq="Q") if end_quarter else working["_quarter_period"].max()
    terminal_rows = []
    for _, group in working.sort_values(["cohort_id", "_quarter_period"]).groupby(
        "cohort_id",
        sort=False,
        dropna=False,
    ):
        last = group.iloc[-1].copy()
        terminal_period = last["_quarter_period"] + 1
        if terminal_period > end_period:
            continue
        terminal = last.copy()
        terminal["quarter"] = str(terminal_period)
        terminal["outstanding"] = 0.0
        if "source_status" in terminal.index:
            terminal["source_status"] = f"{terminal.get('source_status', '')};terminal_zero_balance_synthetic"
        terminal_rows.append(terminal)
    if not terminal_rows:
        return working.drop(columns=["_quarter_period"])
    out = pd.concat([working, pd.DataFrame(terminal_rows)], ignore_index=True)
    return out.drop(columns=["_quarter_period"])


def _terminal_current_principal(row: pd.Series) -> float:
    adjusted = pd.to_numeric(pd.Series([row.get("AdjustedPrincipal", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(adjusted) and float(adjusted) > 0.0:
        return float(adjusted)
    outstanding = pd.to_numeric(pd.Series([row.get("outstanding", 0.0)]), errors="coerce").iloc[0]
    return float(outstanding) if pd.notna(outstanding) else 0.0


def _terminal_par_principal(row: pd.Series, current_principal: float) -> float:
    security_text = str(row.get("security_type", "")).lower()
    if "tips" not in security_text and "inflation" not in security_text:
        return current_principal
    original = pd.to_numeric(pd.Series([row.get("OriginalPrincipal", pd.NA)]), errors="coerce").iloc[0]
    if pd.notna(original) and float(original) > 0.0:
        return float(original)
    index_ratio = pd.to_numeric(
        pd.Series([row.get("IndexRatio", row.get("index_ratio", pd.NA))]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(index_ratio) and float(index_ratio) > 0.0:
        return current_principal / float(index_ratio)
    return current_principal


def _terminal_exit_classification(
    row: pd.Series,
    terminal_period: pd.Period,
) -> tuple[str, str, str, str]:
    maturity_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
    if pd.notna(maturity_date) and pd.Period(maturity_date, freq="Q") == terminal_period:
        return (
            "maturity_redemption",
            maturity_date.strftime("%Y-%m-%d"),
            "source_inferred_from_contractual_maturity_date",
            "first_absent_quarter_matches_contractual_maturity_quarter",
        )
    called_date = pd.to_datetime(row.get("called_date"), errors="coerce")
    if pd.notna(called_date) and pd.Period(called_date, freq="Q") == terminal_period:
        return (
            "called_redemption",
            called_date.strftime("%Y-%m-%d"),
            "source_observed_called_date",
            "first_absent_quarter_matches_called_date",
        )
    call_date = pd.to_datetime(row.get("call_date"), errors="coerce")
    if pd.notna(call_date) and pd.Period(call_date, freq="Q") == terminal_period:
        return (
            "called_redemption",
            call_date.strftime("%Y-%m-%d"),
            "source_inferred_from_call_date_fallback",
            "first_absent_quarter_matches_call_date_fallback",
        )
    return (
        "source_discontinuity_exit",
        terminal_period.end_time.strftime("%Y-%m-%d"),
        "unsupported_terminal_exit",
        "first_absent_quarter_without_maturity_or_call_evidence",
    )


def _event_row(**kwargs) -> dict[str, object]:
    row = {column: pd.NA for column in EVENT_LEDGER_COLUMNS}
    row.update(kwargs)
    return row


def _cohort_id(row: pd.Series) -> str:
    return "|".join(
        str(row.get(col, "")).strip()
        for col in ["cusip", "issue_date", "maturity_date"]
    )


def _cohort_maturity_from_id(cohort_id: object) -> str:
    parts = str(cohort_id).split("|")
    if len(parts) < 3:
        return ""
    return _normalize_date_text(parts[2])


def _normalize_date_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.notna(timestamp):
        return timestamp.date().isoformat()
    return str(value).strip()


def _auction_holder(value: object) -> str:
    text = str(value).strip().lower()
    if text == "banks":
        return "Banks"
    if text == "federal_reserve":
        return "CB"
    if text == "foreign_international":
        return "Foreign"
    return "Private"


def _quarter_end_date(quarter: object) -> str:
    return pd.Period(normalize_quarter_value(quarter), freq="Q").end_time.strftime("%Y-%m-%d")


def _percent_or_decimal(value: object) -> object:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return pd.NA
    numeric = float(numeric)
    return numeric / 100.0 if numeric > 1.0 else numeric


def _first_non_null(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


__all__ = [
    "EVENT_LEDGER_COLUMNS",
    "ROLLFORWARD_COLUMNS",
    "UNEXPLAINED_CHANGE_COLUMNS",
    "build_event_rollforward",
    "build_historical_replay_event_ledger",
]
