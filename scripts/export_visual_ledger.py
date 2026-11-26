#!/usr/bin/env python3
"""Export a thin security ledger of one completed CBO scenario run for visualization.

Writes opening_book.csv, issues.csv, daily.csv, fiscal.csv, README.md and receipt.json
(USD billions).

The opening book and the daily series come from the run tree. The bounded run outputs keep
issuance only as daily aggregates by holder and maturity bucket, so the per-security issues
are recorded by replaying the run through the bounded verifier and copying each issuance
event as the engine emits it. The export fails unless that replay passes verification
(byte-identical outputs) and the recorded issuance sums back to the run's registered
issuance aggregates. Run it with the installed release the run was produced with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from run_open04_reproduction import _assert_installed_distribution_import  # noqa: E402
from simulation_calendar import is_fedwire_business_day, next_fedwire_business_day  # noqa: E402

SCHEMA_VERSION = "tdcsim_visual_ledger_v2"
HOLDER_COLUMNS = ("fed", "banks", "foreign", "mmf", "other_domestic")
OPENING_PORTFOLIO = Path("compile/compiled/forecast_inputs/tdcsim_opening_portfolio.csv")
OPENING_SOURCE = Path("compile/baseline/sources/opening/tdcsim_initial_portfolio_cohorts.csv")
OPENING_METADATA = Path("compile/compiled/forecast_inputs/tdcsim_opening_portfolio_metadata.json")
ROLLFORWARD = Path("compile/compiled/forecast_inputs/tdcsim_opening_rollforward_diagnostics.csv")
RESULTS = Path("outputs/results_compact.csv.gz")
TRC = Path("outputs/tdcsim_trc_results.csv.gz")
ISSUANCE_AGGREGATES = Path("outputs/tdcsim_period_issuance_aggregates.csv.gz")
FINAL_PORTFOLIO = Path("outputs/final_portfolio_compact.csv.gz")
RUN_MANIFEST = Path("tdcsim_cbo_run_manifest.json")
FISCAL_POLICY = Path("compile/compiled/forecast_inputs/tdcsim_fiscal_incidence_policy.csv")
ANNUAL_SUMMARY = Path("outputs/tdcsim_annual_economic_summary.csv.gz")
# Fiscal-year cumulative financing cost in the run's annual summary -> the engine's daily flow.
FINANCING_COST_COLUMNS = {
    "cumulative_interest_outlay_bil": "InterestOutlay_Period",
    "cumulative_issue_discount_cost_bil": "IssueDiscountCost_Period",
    "cumulative_nonmarketable_interest_capitalized_bil": "NonMarketableInterestCapitalized_Period",
    "cumulative_tips_inflation_accretion_bil": "TIPSInflationAccretion_Period",
    "cumulative_modeled_financing_cost_bil": "FinancingCost_Period",
}
TOLERANCE_BIL = 1e-6
DAYS_PER_YEAR = 365.25


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holder_column(holder: str, subbucket: Any) -> str:
    """Map the model's holder and private route to the export's five holder columns."""

    sub = "" if subbucket is None or (isinstance(subbucket, float) and math.isnan(subbucket)) else str(subbucket)
    if holder == "CB":
        return "fed"
    if holder == "Banks":
        return "banks"
    if holder == "Foreign":
        return "foreign"
    if holder == "Private" and sub == "mmf_cash_fund_route":
        return "mmf"
    if holder == "Private" and sub == "domestic_nonbank_deposit_funded":
        return "other_domestic"
    raise ValueError(f"unmapped holder {holder!r}/{sub!r}")


def security_type(model_type: str, original_years: float) -> str:
    if model_type == "TIPS":
        return "TIPS"
    if model_type == "FRN":
        return "FRN"
    if model_type != "Fixed" or not math.isfinite(original_years):
        raise ValueError(f"unclassifiable security {model_type!r} with term {original_years!r}")
    if original_years <= 1.0 + 1e-9:
        return "bill"
    return "note" if original_years <= 10.0 + 1e-9 else "bond"


def payment_date(due: Any) -> pd.Timestamp:
    """When a payment due on ``due`` is made: that day if Fedwire is open, else the next day it is."""

    return pd.Timestamp(next_fedwire_business_day(pd.Timestamp(due)))


def replay_issuance(run_dir: Path, baseline: Path, attestation: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import tdcsim_cbo.verifier as verifier

    rows: list[dict[str, Any]] = []
    base_sink = verifier.BoundedScenarioEvidenceSink

    class RecordingSink(base_sink):  # type: ignore[misc, valid-type]
        def _append(self, name: str, row: Any) -> None:
            if name == "tdcsim_period_issuance_flows":
                rows.append(dict(row))
            super()._append(name, row)

    verifier.BoundedScenarioEvidenceSink = RecordingSink
    try:
        result = verifier.verify_scenario_run(run_dir, baseline_package=baseline, attestation=attestation)
    finally:
        verifier.BoundedScenarioEvidenceSink = base_sink
    if result.get("status") != "pass":
        raise SystemExit(f"replay verification did not pass: {result.get('status')}")
    if not rows:
        raise SystemExit("replay recorded no issuance events")
    return result, rows


ISSUE_TERM_FIELDS = ("security_label", "maturity_date", "dated_date", "is_reopening", "settlement_kind")


def build_issues(flows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per security and settlement date, on the terms the engine issued it.

    The engine names each security (a reopening carries the reopened security's name),
    dates it and says whether the settlement was on Treasury's calendar, early, or at the
    fiscal-year close; a reopening therefore appears on several settlement dates under one
    security id.
    """

    frame = pd.DataFrame(flows)
    missing = [field for field in ISSUE_TERM_FIELDS if field not in frame.columns or frame[field].isna().any()]
    if missing:
        raise SystemExit(f"issuance events lack the engine's issue terms: {missing}")
    frame["settlement_date"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    frame["term"] = frame["weighted_original_term_years"].astype(float)
    frame["kind"] = [security_type(t, y) for t, y in zip(frame["instrument_type"], frame["term"])]
    frame["holder"] = [holder_column(h, s) for h, s in zip(frame["holder_sector"], frame["holder_subsector"])]
    frame["security_id"] = frame["security_label"].astype(str)
    keys = ["security_id", "settlement_date", "kind", "term", "coupon_rate_decimal", "maturity_date", "dated_date",
            "is_reopening", "settlement_kind"]
    if frame[keys].isna().any().any():
        raise SystemExit("an issuance event lacks a date, term or coupon")
    terms = frame.groupby("security_id")[["maturity_date", "coupon_rate_decimal"]].nunique()
    if (terms > 1).any().any():
        raise SystemExit("a security id carries more than one maturity or coupon")
    grouped = frame.groupby(keys)
    out = frame.pivot_table(index=keys, columns="holder", values="face_issued_bil", aggfunc="sum", fill_value=0.0)
    for column in HOLDER_COLUMNS:
        if column not in out:
            out[column] = 0.0
    out["face_total"] = out[list(HOLDER_COLUMNS)].sum(axis=1)
    out = out.join(grouped["cash_proceeds_bil"].sum().rename("cash_proceeds_total"))
    out = out.join(grouped["security_id_tranche"].agg(lambda s: ";".join(sorted(s, key=int))).rename("model_tranche_ids"))
    out = out.reset_index()
    out["maturity_date"] = pd.to_datetime(out["maturity_date"])
    out["maturity_payment_date"] = [payment_date(day) for day in out["maturity_date"]]
    out = out.rename(columns={
        "kind": "security_type", "term": "original_term_years", "coupon_rate_decimal": "coupon_rate",
        "settlement_kind": "settlement_day_kind", **{c: f"face_{c}" for c in HOLDER_COLUMNS},
    })
    columns = [
        "security_id", "settlement_date", "security_type", "original_term_years", "dated_date", "maturity_date",
        "maturity_payment_date", "coupon_rate", "face_total", *(f"face_{c}" for c in HOLDER_COLUMNS),
        "cash_proceeds_total", "is_reopening", "settlement_day_kind", "model_tranche_ids",
    ]
    return out[columns].sort_values(["settlement_date", "security_type", "original_term_years"]).reset_index(drop=True)


def name_reopened_opening_securities(issues: pd.DataFrame, opening: pd.DataFrame) -> pd.DataFrame:
    """Give a reopening of an opening-book security that security's CUSIP."""

    lookup = {
        (kind, pd.Timestamp(maturity), round(float(coupon), 9)): security_id
        for security_id, kind, maturity, coupon in zip(
            opening["security_id"], opening["security_type"], opening["maturity_date"], opening["coupon_rate"]
        )
    }
    out = issues.copy()
    for index, row in out.iterrows():
        key = (row["security_type"], pd.Timestamp(row["maturity_date"]), round(float(row["coupon_rate"]), 9))
        if row["is_reopening"] and key in lookup:
            out.loc[index, "security_id"] = lookup[key]
    return out


def build_opening(run_dir: Path) -> pd.DataFrame:
    book = pd.read_csv(run_dir / OPENING_PORTFOLIO, low_memory=False)
    source = pd.read_csv(run_dir / OPENING_SOURCE, low_memory=False)[["BondID", "cusip_or_synthetic_id", "MaturityDate"]]
    book = book.merge(source, on="BondID", how="left", suffixes=("", "_source"))
    joined = book["cusip_or_synthetic_id"].notna()
    if (book.loc[joined, "MaturityDate"] != book.loc[joined, "MaturityDate_source"]).any():
        raise SystemExit("opening book and its source disagree on maturity dates")
    book["kind"] = [security_type(t, float(y)) for t, y in zip(book["SecurityType"], book["OriginalMaturityYears"])]
    # Rows not in the source book are the pre-start rollforward's reinvestment in Treasury
    # issues settling before the opening; the run's rollforward diagnostics name their CUSIPs.
    rollforward = pd.read_csv(run_dir / ROLLFORWARD, low_memory=False)
    added = rollforward[rollforward["row_type"] == "replacement"].set_index("replacement_bond_id")["replacement_cusip"]
    book["security_id"] = book["cusip_or_synthetic_id"].fillna(book["BondID"].map(added))
    if book["security_id"].isna().any():
        raise SystemExit("an opening row is neither in the source book nor a pre-start rollforward replacement")
    book["holder"] = [holder_column(h, s) for h, s in zip(book["HolderType"], book["HolderSubBucket"])]
    book["debt_basis"] = book["AdjustedPrincipal"].where(book["SecurityType"].eq("TIPS"), book["FaceValue"])
    keys = ["security_id", "kind", "IssueDate", "MaturityDate", "CouponRate"]
    if (book.groupby("security_id")[["IssueDate", "MaturityDate", "CouponRate"]].nunique() > 1).any().any():
        raise SystemExit("an opening security id carries more than one issue date, maturity or coupon")
    face = book.pivot_table(index=keys, columns="holder", values="FaceValue", aggfunc="sum", fill_value=0.0).reset_index()
    for column in HOLDER_COLUMNS:
        if column not in face:
            face[column] = 0.0
    face["face_total"] = face[list(HOLDER_COLUMNS)].sum(axis=1)
    debt = book.groupby("security_id")["debt_basis"].sum()
    face["debt_basis_total"] = face["security_id"].map(debt)
    face = face.rename(columns={
        "kind": "security_type", "IssueDate": "issue_date", "MaturityDate": "maturity_date", "CouponRate": "coupon_rate",
        **{c: f"face_{c}" for c in HOLDER_COLUMNS},
    })
    face["maturity_payment_date"] = [payment_date(day) for day in pd.to_datetime(face["maturity_date"])]
    columns = ["security_id", "security_type", "issue_date", "maturity_date", "maturity_payment_date", "coupon_rate",
               "face_total", *(f"face_{c}" for c in HOLDER_COLUMNS), "debt_basis_total"]
    return face[columns].sort_values(["maturity_date", "security_id"]).reset_index(drop=True)


def build_daily(run_dir: Path, results: pd.DataFrame, opening: pd.DataFrame, issues: pd.DataFrame,
                financing_cost: pd.Series) -> pd.DataFrame:
    trc = pd.read_csv(run_dir / TRC, parse_dates=["Date"]).set_index("Date")
    daily = pd.DataFrame(index=results.index)
    daily["fedwire_open"] = [is_fedwire_business_day(day) for day in results.index]
    daily["cumulative_tdc"] = results["TDC_Level"]
    daily["cumulative_trc"] = trc["TRC_Cumulative"].reindex(results.index)
    daily["tga"] = results["TGA"]
    daily["marketable_debt"] = results["TotalDebt_Agg"]
    daily["pending_issuance"] = results["PendingIssuanceProceeds"]
    # A security counts once, from its first settlement (the opening for the opening
    # book, which the run may reopen) until its principal is paid.
    new = issues[~issues["security_id"].isin(opening["security_id"])]
    first = new.groupby("security_id").agg(start=("settlement_date", "min"), end=("maturity_payment_date", "first"))
    maturities = pd.concat([
        pd.to_datetime(opening["maturity_payment_date"]),
        pd.to_datetime(first["end"]),
    ])
    starts = pd.concat([
        pd.Series(results.index[0], index=range(len(opening))),
        pd.to_datetime(first["start"]).reset_index(drop=True),
    ])
    counts = []
    for day in results.index:
        counts.append(int(((starts <= day).to_numpy() & (maturities > day).to_numpy()).sum()))
    daily["securities_outstanding_count"] = counts
    daily["principal_paid"] = results["PrincipalPaid_Bonds"]
    daily["coupon_interest_paid"] = results["InterestOutlay_Period"]
    issued = issues.groupby(pd.to_datetime(issues["settlement_date"]))["face_total"].sum()
    daily["gross_face_issued"] = issued.reindex(results.index, fill_value=0.0)
    daily["cumulative_financing_cost"] = financing_cost
    if (daily.isna().any()).any():
        raise SystemExit("daily series has gaps")
    return daily.reset_index().rename(columns={"Date": "date"})


def build_fiscal(run_dir: Path, results: pd.DataFrame, policy_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Daily net fiscal flow and its deposit, non-deposit and reserve incidence.

    The run models the CBO primary deficit as one signed net flow (no separate gross outlays
    and receipts); the active incidence policy splits it into a covered-deposit share and
    shares settling outside covered deposits.
    """

    policies = pd.read_csv(run_dir / FISCAL_POLICY)
    policy = policies[policies["policy_id"] == policy_id]
    if len(policy) != 1:
        raise SystemExit(f"fiscal incidence policy {policy_id!r} is not in the run's compiled inputs")
    policy = policy.iloc[0]
    if str(policy["incidence_basis"]) != "signed_net_primary_proxy":
        raise SystemExit(f"unexpected fiscal incidence basis {policy['incidence_basis']!r}")
    outside_share = float(policy["ru_share"]) + float(policy["foreign_share"]) + float(policy["other_share"])
    trc = pd.read_csv(run_dir / TRC, parse_dates=["Date"]).set_index("Date")
    fiscal = pd.DataFrame(index=results.index)
    fiscal["primary_deficit"] = results["PrimaryDeficit"]
    fiscal["fiscal_to_du"] = results["TDC_FiscalFlow"]
    fiscal["fiscal_to_rs"] = outside_share * results["PrimaryDeficit"]
    fiscal["trc_fiscal_flow"] = trc["TRC_FiscalFlow"].reindex(results.index)
    if fiscal.isna().any().any():
        raise SystemExit("fiscal series has gaps")
    identity_gap = float((fiscal["fiscal_to_du"] + fiscal["fiscal_to_rs"] - fiscal["primary_deficit"]).abs().max())
    if identity_gap > TOLERANCE_BIL:
        raise SystemExit(f"fiscal incidence does not add up to the primary deficit (gap {identity_gap})")
    checks = {
        "fiscal_incidence_policy_id": policy_id,
        "du_share": float(policy["du_share"]),
        "outside_covered_deposits_share": outside_share,
        "gross_outlays_and_receipts_modeled": bool(
            pd.notna(policy["primary_outlays_bil"]) or pd.notna(policy["primary_receipts_bil"])
        ),
        "max_gap_du_plus_rs_minus_primary_deficit_bil": identity_gap,
        "net_receipt_days": int((fiscal["primary_deficit"] < 0).sum()),
    }
    return fiscal.reset_index().rename(columns={"Date": "date"}), checks


def build_financing_cost(run_dir: Path, results: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Cumulative modeled financing cost since the opening, on the basis of the run's annual summary.

    Sums the engine's daily cost flows and requires the running totals, and each of the four
    components, to equal the annual summary (the source of the scenario-family export) at every
    fiscal-year end.
    """

    cumulative = results[list(FINANCING_COST_COLUMNS.values())].cumsum()
    annual = pd.read_csv(run_dir / ANNUAL_SUMMARY, parse_dates=["period_end"]).set_index("period_end")
    missing = annual.index.difference(cumulative.index)
    if len(missing):
        raise SystemExit(f"annual summary period ends missing from the daily results: {list(missing)}")
    expected = annual[list(FINANCING_COST_COLUMNS)].rename(columns=FINANCING_COST_COLUMNS)
    gap = float((cumulative.loc[annual.index] - expected).abs().max().max())
    if not gap <= TOLERANCE_BIL:
        raise SystemExit(f"daily cumulative financing cost differs from the run's annual summary by {gap}")
    final = annual.index.max()
    checks = {
        "fiscal_year_ends_checked": len(annual),
        "max_gap_vs_run_annual_summary_bil": gap,
        "components_through": final.date().isoformat(),
        "components_bil": {
            name.removeprefix("cumulative_").removesuffix("_bil"): float(cumulative.loc[final, column])
            for name, column in FINANCING_COST_COLUMNS.items()
        },
    }
    return cumulative["FinancingCost_Period"], checks


def check_against_aggregates(run_dir: Path, flows: list[dict[str, Any]]) -> float:
    """Largest gap between the recorded issuance and the run's registered issuance aggregates."""

    registered = pd.read_csv(run_dir / ISSUANCE_AGGREGATES)
    keys = ["period_end", "holder_sector", "holder_subsector", "instrument_type", "maturity_bucket"]
    registered[keys] = registered[keys].fillna("").astype(str)
    recorded = pd.DataFrame(flows)
    recorded[keys] = recorded[keys].fillna("").astype(str)
    left = registered.groupby(keys)["face_issued_bil"].sum()
    right = recorded.groupby(keys)["face_issued_bil"].sum()
    gap = left.subtract(right, fill_value=0.0).abs().max()
    if gap > TOLERANCE_BIL:
        raise SystemExit(f"recorded issuance differs from the registered aggregates by {gap}")
    return float(gap)


def check_maturities(run_dir: Path, issues: pd.DataFrame) -> int:
    """Match derived maturity dates against tranches still outstanding at the end of the run."""

    final = pd.read_csv(run_dir / FINAL_PORTFOLIO, low_memory=False, usecols=["BondID", "MaturityDate"])
    final_maturity = dict(zip(final["BondID"].astype(str), final["MaturityDate"].astype(str)))
    checked = 0
    for ids, maturity in zip(issues["model_tranche_ids"], issues["maturity_date"]):
        for tranche in ids.split(";"):
            if tranche in final_maturity:
                if final_maturity[tranche][:10] != pd.Timestamp(maturity).date().isoformat():
                    raise SystemExit(f"tranche {tranche} matures {final_maturity[tranche]}, derived {maturity}")
                checked += 1
    if checked == 0:
        raise SystemExit("no issued tranche survives to the end of the run to check maturities against")
    return checked


README = """# Visual ledger: {run_label}

Thin export of one completed TDCSim CBO scenario run, for illustration. USD billions.
Produced by `scripts/export_visual_ledger.py`; provenance and hashes are in `receipt.json`.

## Files
- `opening_book.csv`: every security outstanding at the {start} opening, by CUSIP. The source
  book is Treasury's month-end statement for {record_date}; what matured between then and the
  opening is reinvested, holder by holder, in the Treasury issues that settled on its payment
  date, in proportion to their accepted auction amounts, so those issues appear under their own
  CUSIPs. Face is par; TIPS par excludes inflation compensation, which `debt_basis_total`
  includes. `maturity_payment_date` is when the principal is paid: the maturity date, or the
  next day Fedwire is open when the maturity date is a weekend or holiday.
- `issues.csv`: every security the run issued, one row per security and settlement date, on
  Treasury's auction calendar: bills weekly (the 52-week bill every four weeks), 2-, 3- and
  5-year notes monthly as new issues, the 7-year monthly until February 2027 and then new
  quarterly from March 2027 and reopened in the two months after, 10-, 20- and 30-year
  securities new in February, May, August and November and reopened in the two months after
  (20-year reopenings settle on the Friday of their auction week from June 2026), the
  floating-rate note new quarterly and reopened monthly, and TIPS on Treasury's schedule. The
  dates match Treasury's published auction schedules through April 2027. A reopening
  (`is_reopening`) sells more of an existing security on its terms and appears under that
  security's id; a reopening of an opening-book security uses its CUSIP. `dated_date` is
  when interest starts (buyers pay what has accrued since it or the last coupon),
  `maturity_date` the contractual maturity and `maturity_payment_date` the day principal is
  paid. `settlement_day_kind` is `calendar`, `early_settlement` (pending issuance settled
  early to keep the TGA from going negative) or `fiscal_year_close` (everything pending
  settles on the fiscal year's last business day); off-calendar settlements reopen each
  security's latest issue. `model_tranche_ids` lists the model's per-holder rows.
- `fiscal.csv`: one row per day. `primary_deficit` is the net fiscal flow out of the TGA
  ({fiscal_note}). `fiscal_to_du` is the part landing in covered deposits (TDC_FiscalFlow);
  `fiscal_to_rs` is the part settling outside covered deposits, here the reserve-settled share
  of the run's fiscal incidence policy ({rs_share:.0%}; foreign and other shares are zero), so
  the two sum to `primary_deficit` every day. `trc_fiscal_flow` is the reserve effect
  (TRC_FiscalFlow). Gross outlays and receipts are not modeled separately.
- `daily.csv`: one row per calendar day. `fedwire_open` marks the days cash can move: on the
  others nothing is paid, issued or settled, and fiscal flows are zero; a payment due then is
  made on the next open day at its contractual amount. Securities do not settle on Good
  Friday or the Friday before a Saturday holiday either, although Fedwire is open.
  `cumulative_tdc` and `cumulative_trc` are net totals since the
  opening (no component legs). `tga`, `marketable_debt` (TIPS at par plus inflation
  compensation) and `pending_issuance` (issuance decided but awaiting its settlement date) are
  end-of-day levels. `securities_outstanding_count` counts securities from the two files above
  that are outstanding at the end of the day. `principal_paid`, `coupon_interest_paid` (cash
  coupons and floating-rate interest; bill interest is the discount, paid within principal at
  maturity) and `gross_face_issued` are daily flows. `cumulative_financing_cost` is the run's
  modeled financing cost since the opening: cash interest paid, issue discount (face minus cash
  proceeds, booked at settlement, which is how bill interest enters), capitalized nonmarketable
  interest and TIPS inflation accretion; `receipt.json` gives the split at the last day.

## Holder split
Holders: `fed` (Federal Reserve), `banks`, `foreign`, `mmf` (money-market funds) and
`other_domestic` (all other domestic holders, which pay from bank deposits).
- Opening book: holders as of the opening. The Federal Reserve's holdings are its SOMA holdings
  by CUSIP. The remainder is split uniformly within bills and within longer securities: banks,
  foreign holders and money funds keep their measured sector holdings converted to par, and
  other domestic holders take the rest.
- Issues: holders at issuance. The Federal Reserve rolls its maturing holdings over at auction
  as add-ons (`face_fed`), in the securities settling the day its holdings are paid, in
  proportion to the amounts sold to the public. Beyond that it follows its holdings path by
  buying from or selling to other holders in the secondary market (domestic private holders
  first, shortest maturity first), so holdings change after issuance. Label this split
  "at issuance".
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    _assert_installed_distribution_import()
    status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=PROJECT_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if status:
        raise SystemExit("export requires a clean committed producer checkout")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    run_dir = args.run_dir.resolve()
    if args.out_dir.exists():
        raise SystemExit(f"output directory already exists: {args.out_dir}")

    results = pd.read_csv(run_dir / RESULTS, parse_dates=["Date"]).set_index("Date")
    verification, flows = replay_issuance(run_dir, args.baseline, args.attestation)
    for row in flows:
        row["security_id_tranche"] = str(row["security_id"])
    aggregate_gap = check_against_aggregates(run_dir, flows)
    issues = build_issues(flows)
    maturities_checked = check_maturities(run_dir, issues)
    if issues.duplicated(["security_id", "settlement_date"]).any():
        raise SystemExit("a security settles twice on one day")
    for column in ("settlement_date", "maturity_payment_date"):
        closed = [day for day in pd.to_datetime(issues[column]) if not is_fedwire_business_day(day)]
        if closed:
            raise SystemExit(f"{column} falls on a day Fedwire is closed: {closed[:3]}")
    opening = build_opening(run_dir)
    issues = name_reopened_opening_securities(issues, opening)
    manifest = json.loads((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))
    scenario_id = (manifest.get("scenario") or {}).get("scenario_id")
    if not scenario_id:
        raise SystemExit("run manifest records no scenario.scenario_id")
    fiscal, fiscal_checks = build_fiscal(run_dir, results, str(manifest.get("fiscal_incidence_policy_id") or ""))
    financing_cost, financing_cost_checks = build_financing_cost(run_dir, results)
    daily = build_daily(run_dir, results, opening, issues, financing_cost)
    opening_gap = abs(float(opening["debt_basis_total"].sum()) - float(results["TotalDebt_Agg"].iloc[0]))
    if opening_gap > 1e-6:
        raise SystemExit(f"opening book does not sum to the opening marketable debt (gap {opening_gap})")
    issue_gap = float((daily.set_index("date")["gross_face_issued"] - results["NewDebtIssued"]).abs().max())
    if issue_gap > TOLERANCE_BIL:
        raise SystemExit(f"issued face differs from the run's NewDebtIssued by {issue_gap}")

    args.out_dir.mkdir(parents=True)
    outputs = {
        "opening_book.csv": opening,
        "issues.csv": issues,
        "daily.csv": daily,
        "fiscal.csv": fiscal,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False, lineterminator="\n", date_format="%Y-%m-%d", float_format="%.9f")
    (args.out_dir / "README.md").write_text(
        README.format(
            run_label=scenario_id,
            start=results.index[0].date(),
            record_date=json.loads((run_dir / OPENING_METADATA).read_text(encoding="utf-8"))["record_date"],
            rs_share=fiscal_checks["outside_covered_deposits_share"],
            fiscal_note=(
                "CBO's primary deficit for each fiscal year spread evenly over its days, so it is"
                " constant within a fiscal year and has no tax-date or payment-date seasonality"
                if fiscal_checks["net_receipt_days"] == 0
                else "negative on net-receipt days"
            ),
        ),
        encoding="utf-8",
    )
    from tdcsim_cbo.runtime_identity import distribution_identity, installed_archive_sha256

    inputs = {str(path): sha256(run_dir / path) for path in (
        RUN_MANIFEST, OPENING_PORTFOLIO, OPENING_SOURCE, ROLLFORWARD, OPENING_METADATA, RESULTS, TRC, ISSUANCE_AGGREGATES, FINAL_PORTFOLIO,
        FISCAL_POLICY, ANNUAL_SUMMARY)}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "producer": {
            "repository_commit": commit,
            "script": "scripts/export_visual_ledger.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "installed_archive_sha256": installed_archive_sha256(),
            "distribution_file_digest": distribution_identity()["file_digest"],
            "python_version": platform.python_version(),
        },
        "run": {
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir.name,
            "scenario_id": scenario_id,
            "baseline_package_sha256": sha256(args.baseline),
            "attestation_sha256": sha256(args.attestation),
        },
        "inputs_sha256": inputs,
        "replay": {
            "method": "bounded verifier replay of the run, recording each issuance event",
            "verification_status": verification.get("status"),
            "verification_grade": verification.get("verification_grade"),
            "issuance_events_recorded": len(flows),
            "max_gap_vs_registered_issuance_aggregates_bil": aggregate_gap,
        },
        "checks": {
            "opening_book_minus_opening_marketable_debt_bil": opening_gap,
            "max_daily_gap_issued_face_vs_new_debt_issued_bil": issue_gap,
            "issued_tranches_matched_to_final_portfolio_maturities": maturities_checked,
            "fiscal": fiscal_checks,
            "financing_cost": financing_cost_checks,
        },
        "outputs": {
            name: {"sha256": sha256(args.out_dir / name), "bytes": (args.out_dir / name).stat().st_size,
                   **({"rows": len(outputs[name])} if name in outputs else {})}
            for name in ("opening_book.csv", "issues.csv", "daily.csv", "fiscal.csv", "README.md")
        },
        "units": "USD billions",
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "receipt_sha256": sha256(args.out_dir / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
