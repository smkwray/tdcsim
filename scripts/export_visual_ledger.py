#!/usr/bin/env python3
"""Export a thin security ledger of one completed CBO scenario run for visualization.

Writes opening_book.csv, issues.csv, daily.csv, README.md and receipt.json (USD billions).

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

SCHEMA_VERSION = "tdcsim_visual_ledger_v1"
HOLDER_COLUMNS = ("fed", "banks", "foreign", "mmf", "other_domestic")
OPENING_PORTFOLIO = Path("compile/compiled/forecast_inputs/tdcsim_opening_portfolio.csv")
OPENING_SOURCE = Path("compile/baseline/sources/opening/tdcsim_initial_portfolio_cohorts.csv")
RESULTS = Path("outputs/results_compact.csv.gz")
TRC = Path("outputs/tdcsim_trc_results.csv.gz")
ISSUANCE_AGGREGATES = Path("outputs/tdcsim_period_issuance_aggregates.csv.gz")
FINAL_PORTFOLIO = Path("outputs/final_portfolio_compact.csv.gz")
RUN_MANIFEST = Path("tdcsim_cbo_run_manifest.json")
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
    if holder == "Private" and sub in {"", "domestic_nonbank_deposit_funded"}:
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


def term_label(kind: str, original_years: float) -> str:
    if kind == "bill":
        return f"{round(original_years * DAYS_PER_YEAR / 7)}w"
    return f"{round(original_years)}y"


def issue_maturity(settlement: pd.Timestamp, kind: str, original_years: float) -> pd.Timestamp:
    """Maturity as the engine dates it: bills a whole number of weeks, others whole months."""

    if kind == "bill":
        return settlement + pd.Timedelta(days=7 * round(original_years * DAYS_PER_YEAR / 7))
    months = int(round(original_years * 12.0))
    return settlement + pd.DateOffset(years=months // 12, months=months % 12)


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


def build_issues(flows: list[dict[str, Any]], results: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(flows)
    frame["settlement_date"] = pd.to_datetime(frame["period_end"]).dt.normalize()
    frame["term"] = frame["weighted_original_term_years"].astype(float)
    frame["kind"] = [security_type(t, y) for t, y in zip(frame["instrument_type"], frame["term"])]
    frame["holder"] = [holder_column(h, s) for h, s in zip(frame["holder_sector"], frame["holder_subsector"])]
    frame["label"] = [term_label(k, y) for k, y in zip(frame["kind"], frame["term"])]
    frame["security_id"] = (
        frame["kind"] + "-" + frame["label"] + "-" + frame["settlement_date"].dt.strftime("%Y-%m-%d")
    )
    keys = ["security_id", "settlement_date", "kind", "term", "label", "coupon_rate_decimal"]
    if frame[keys].isna().any().any():
        raise SystemExit("an issuance event lacks a date, term or coupon")
    coupon_groups = frame.groupby("security_id")["coupon_rate_decimal"].nunique()
    if (coupon_groups > 1).any():
        raise SystemExit("a security id carries more than one coupon")
    grouped = frame.groupby(keys)
    out = frame.pivot_table(index=keys, columns="holder", values="face_issued_bil", aggfunc="sum", fill_value=0.0)
    for column in HOLDER_COLUMNS:
        if column not in out:
            out[column] = 0.0
    out["face_total"] = out[list(HOLDER_COLUMNS)].sum(axis=1)
    out = out.join(grouped["cash_proceeds_bil"].sum().rename("cash_proceeds_total"))
    out = out.join(grouped["security_id_tranche"].agg(lambda s: ";".join(sorted(s, key=int))).rename("model_tranche_ids"))
    out = out.reset_index()
    out["maturity_date"] = [issue_maturity(d, k, y) for d, k, y in zip(out["settlement_date"], out["kind"], out["term"])]
    day_kind = {}
    for day, row in results.iterrows():
        if float(row.get("FiscalYearCloseSettlementProceeds", 0.0) or 0.0) > TOLERANCE_BIL:
            day_kind[day] = "fiscal_year_close"
        elif float(row.get("EarlySettlementProceeds", 0.0) or 0.0) > TOLERANCE_BIL:
            day_kind[day] = "early_settlement"
    out["settlement_day_kind"] = [day_kind.get(day, "calendar") for day in out["settlement_date"]]
    out["is_reopening"] = False
    out = out.rename(columns={"kind": "security_type", "term": "original_term_years", "coupon_rate_decimal": "coupon_rate"})
    columns = [
        "security_id", "settlement_date", "security_type", "original_term_years", "maturity_date", "coupon_rate",
        "face_total", *(f"face_{c}" for c in HOLDER_COLUMNS), "cash_proceeds_total", "is_reopening",
        "settlement_day_kind", "model_tranche_ids",
    ]
    out = out.rename(columns={c: f"face_{c}" for c in HOLDER_COLUMNS})
    return out[columns].sort_values(["settlement_date", "security_type", "original_term_years"]).reset_index(drop=True)


def build_opening(run_dir: Path) -> pd.DataFrame:
    book = pd.read_csv(run_dir / OPENING_PORTFOLIO, low_memory=False)
    source = pd.read_csv(run_dir / OPENING_SOURCE, low_memory=False)[["BondID", "cusip_or_synthetic_id", "MaturityDate"]]
    book = book.merge(source, on="BondID", how="left", suffixes=("", "_source"))
    joined = book["cusip_or_synthetic_id"].notna()
    if (book.loc[joined, "MaturityDate"] != book.loc[joined, "MaturityDate_source"]).any():
        raise SystemExit("opening book and its source disagree on maturity dates")
    book["kind"] = [security_type(t, float(y)) for t, y in zip(book["SecurityType"], book["OriginalMaturityYears"])]
    # Securities issued by the model on the opening date to replace issues that matured before
    # it have no CUSIP; they are identified by type, term and maturity.
    book["security_id"] = book["cusip_or_synthetic_id"]
    model = ~joined
    book.loc[model, "security_id"] = (
        "prestart-" + book.loc[model, "kind"] + "-" + book.loc[model, "MaturityDate"].astype(str)
    )
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
    columns = ["security_id", "security_type", "issue_date", "maturity_date", "coupon_rate", "face_total",
               *(f"face_{c}" for c in HOLDER_COLUMNS), "debt_basis_total"]
    return face[columns].sort_values(["maturity_date", "security_id"]).reset_index(drop=True)


def build_daily(run_dir: Path, results: pd.DataFrame, opening: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    trc = pd.read_csv(run_dir / TRC, parse_dates=["Date"]).set_index("Date")
    daily = pd.DataFrame(index=results.index)
    daily["cumulative_tdc"] = results["TDC_Level"]
    daily["cumulative_trc"] = trc["TRC_Cumulative"].reindex(results.index)
    daily["tga"] = results["TGA"]
    daily["marketable_debt"] = results["TotalDebt_Agg"]
    daily["pending_issuance"] = results["PendingIssuanceProceeds"]
    maturities = pd.concat([
        pd.to_datetime(opening["maturity_date"]),
        pd.to_datetime(issues["maturity_date"]),
    ])
    starts = pd.concat([
        pd.Series(results.index[0], index=range(len(opening))),
        pd.to_datetime(issues["settlement_date"]).reset_index(drop=True),
    ])
    counts = []
    for day in results.index:
        counts.append(int(((starts <= day).to_numpy() & (maturities > day).to_numpy()).sum()))
    daily["securities_outstanding_count"] = counts
    daily["principal_paid"] = results["PrincipalPaid_Bonds"]
    daily["coupon_interest_paid"] = results["InterestOutlay_Period"]
    issued = issues.groupby(pd.to_datetime(issues["settlement_date"]))["face_total"].sum()
    daily["gross_face_issued"] = issued.reindex(results.index, fill_value=0.0)
    if (daily.isna().any()).any():
        raise SystemExit("daily series has gaps")
    return daily.reset_index().rename(columns={"Date": "date"})


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
- `opening_book.csv`: every security outstanding at the {start} opening, by CUSIP. Securities
  the model issued on the opening date to replace issues that matured on the weekend before it
  have no CUSIP and are identified as `prestart-bill-<maturity>`. Face is par; TIPS par excludes
  inflation compensation, which `debt_basis_total` includes.
- `issues.csv`: every security the run issued, one row per security and settlement date. The
  model issues a new security at each settlement; it never reopens an existing one
  (`is_reopening` is always false). `settlement_day_kind` marks days when pending issuance was
  settled early to keep the TGA from going negative, or at the fiscal-year close.
  `model_tranche_ids` lists the model's per-holder rows.
- `daily.csv`: one row per day. `cumulative_tdc` and `cumulative_trc` are net totals since the
  opening (no component legs). `tga`, `marketable_debt` (TIPS at par plus inflation
  compensation) and `pending_issuance` (issuance decided but awaiting its settlement date) are
  end-of-day levels. `securities_outstanding_count` counts securities from the two files above
  that are outstanding at the end of the day. `principal_paid`, `coupon_interest_paid` (cash
  coupons and floating-rate interest; bill interest is the discount, paid within principal at
  maturity) and `gross_face_issued` are daily flows.

## Holder split
Holders: `fed` (Federal Reserve), `banks`, `foreign`, `mmf` (money-market funds) and
`other_domestic` (all other domestic holders, which pay from bank deposits).
- Opening book: holders as of the opening. The Federal Reserve's holdings are its SOMA holdings
  by CUSIP; the remainder is split by measured sector shares of bills and of longer securities.
- Issues: holders at issuance. The Federal Reserve takes no auction allocation in this model,
  so `face_fed` is zero for every issue. After issuance the Federal Reserve keeps its holdings
  on its path by buying securities from other holders in the secondary market (domestic private
  holders first, shortest maturity first), so holdings change after issuance. Label this split
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
    issues = build_issues(flows, results)
    maturities_checked = check_maturities(run_dir, issues)
    opening = build_opening(run_dir)
    daily = build_daily(run_dir, results, opening, issues)
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
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False, lineterminator="\n", date_format="%Y-%m-%d", float_format="%.9f")
    manifest = json.loads((run_dir / RUN_MANIFEST).read_text(encoding="utf-8"))
    (args.out_dir / "README.md").write_text(
        README.format(run_label=manifest.get("scenario_id", run_dir.name), start=results.index[0].date()),
        encoding="utf-8",
    )
    from tdcsim_cbo.runtime_identity import distribution_identity, installed_archive_sha256

    inputs = {str(path): sha256(run_dir / path) for path in (
        RUN_MANIFEST, OPENING_PORTFOLIO, OPENING_SOURCE, RESULTS, TRC, ISSUANCE_AGGREGATES, FINAL_PORTFOLIO)}
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
            "scenario_id": manifest.get("scenario_id"),
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
        },
        "outputs": {
            name: {"sha256": sha256(args.out_dir / name), "bytes": (args.out_dir / name).stat().st_size,
                   **({"rows": len(outputs[name])} if name in outputs else {})}
            for name in ("opening_book.csv", "issues.csv", "daily.csv", "README.md")
        },
        "units": "USD billions",
    }
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n",
                                               encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "receipt_sha256": sha256(args.out_dir / "receipt.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
