"""Extract money-fund Treasury holdings by CUSIP from SEC Form N-MFP data sets.

Reads the SEC DERA Form N-MFP quarterly data-set zips and FiscalData MSPD Table 3, and
writes one row per report month-end and Treasury CUSIP with the money-fund industry's
holdings, plus a provenance JSON naming every source file with its sha256.

Rules:
- Filings: N-MFP submissions (amendments included) whose report date falls in a month
  of the requested range and that were filed by the information date (--filed-through);
  later filings are dropped before ranking. Per series and month one filing is kept: a
  complete schedule first, then the latest filing date, then the latest zip, then the
  accession number. The provenance lists every selected accession and filing date.
- Feeder funds (FEEDERFUNDFLAG=Y) are dropped, so a master's holdings count once. The
  Treasury value of dropped feeders is recorded.
- A portfolio row is a Treasury when its CUSIP is a marketable Treasury in MSPD Table 3
  (any record date in the file), typed by MSPD class. Otherwise a row filed as
  "U.S. Treasury Debt" is typed by its title, then by CUSIP prefix (912796/912797 a
  bill; 912828, 91282C or 912810 a nominal coupon). Repurchase agreements are not
  Treasury holdings. Treasury-category rows no rule types are kept as "unresolved".
- Value: the form's security value excluding sponsor support (including it when the
  excluding field is blank). N-MFP reports value, not par; its principal field is a
  legacy field that N-MFP3 filings leave blank.
- Maturity: the MSPD maturity date, else the filing's final legal maturity.
- Coverage: every selected filing must carry a complete schedule. Each floating-rate note
  outstanding in MSPD at a report month-end that no fund holds gets an explicit zero row
  (class source "mspd_outstanding_not_held"), so a missing row means missing coverage,
  never zero holdings.

Usage:
  python -I scripts/extract_sec_nmfp_mmf_holdings.py \
      --mspd-table-3-market data/historical_replay/raw/fiscaldata/mspd_table_3_market.csv \
      --first-month 2025-05 --last-month 2026-05 --filed-through 2026-06-22 \
      --output data/ratewall_inputs/sec_nmfp_mmf_treasury_holdings.csv \
      <N-MFP zip> [<N-MFP zip> ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

MSPD_CLASSES = {
    "Bills Maturity Value": "bill",
    "Notes": "fixed_coupon",
    "Bonds": "fixed_coupon",
    "Inflation-Protected Securities": "tips",
    "Inflation-Indexed Notes": "tips",
    "Inflation-Indexed Bonds": "tips",
    "Floating Rate Notes": "frn",
}
TITLE_CLASSES = (
    ("frn", re.compile(r"floating|\bfrns?\b", re.I)),
    ("tips", re.compile(r"inflation|\btips\b", re.I)),
    ("bill", re.compile(r"\bbills?\b|t-bill|cash management", re.I)),
    ("fixed_coupon", re.compile(r"\bnotes?\b|\bbonds?\b", re.I)),
)
TREASURY_CATEGORY = "U.S. Treasury Debt"
SUBMISSION_COLUMNS = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "REPORTDATE", "SERIESID"]
SERIES_COLUMNS = ["ACCESSION_NUMBER", "FEEDERFUNDFLAG", "NETASSETOFSERIES"]
SECURITY_COLUMNS = [
    "ACCESSION_NUMBER",
    "NAMEOFISSUER",
    "TITLEOFISSUER",
    "CUSIP_NUMBER",
    "INVESTMENTCATEGORY",
    "FINALLEGALINVESTMENTMATURITYDA",
    "INCLUDINGVALUEOFANYSPONSORSUPP",
    "EXCLUDINGVALUEOFANYSPONSORSUPP",
]
OUTPUT_COLUMNS = ["report_date", "cusip", "security_class", "maturity_date", "value_bil", "class_source", "series_count"]
VALUE_BASIS = (
    "N-MFP security value excluding sponsor support, in billions; not par. "
    "The form's principal field is legacy and blank in N-MFP3 filings."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tsv(archive: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(archive.open(name), sep="\t", dtype=str, usecols=columns, low_memory=False)


def _mspd_cusips(path: Path, month_ends: list[pd.Timestamp]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return MSPD class and maturity by CUSIP, and the FRNs outstanding at each month-end."""

    mspd = pd.read_csv(path, dtype=str)
    record = pd.to_datetime(mspd["record_date"])
    missing = [day.date().isoformat() for day in month_ends if day not in set(record)]
    if missing:
        raise ValueError(f"{path.name} has no record date for report month-ends {missing}")
    # The month before the range types securities that matured on the first report date.
    first = month_ends[0] - pd.offsets.MonthEnd(1)
    mspd = mspd.loc[
        record.between(first, month_ends[-1])
        & mspd["security_class1_desc"].isin(MSPD_CLASSES)
        & mspd["security_class2_desc"].fillna("").str.fullmatch(r"[0-9A-Z]{9}")
    ]
    cusips = pd.DataFrame(
        {
            "cusip": mspd["security_class2_desc"],
            "security_class": mspd["security_class1_desc"].map(MSPD_CLASSES),
            "maturity_date": pd.to_datetime(mspd["maturity_date"]),
        }
    ).drop_duplicates()
    if cusips["cusip"].duplicated().any():
        raise ValueError(f"{path.name} gives a CUSIP two classes or maturities")
    outstanding = mspd.loc[
        mspd["security_class1_desc"].eq("Floating Rate Notes") & pd.to_numeric(mspd["outstanding_amt"], errors="coerce").gt(0)
    ]
    frns = {
        day.date().isoformat(): sorted(set(outstanding.loc[pd.to_datetime(outstanding["record_date"]).eq(day), "security_class2_desc"]))
        for day in month_ends
    }
    return cusips.set_index("cusip"), frns


def _read_filings(zips: list[Path], months: set[str], filed_through: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    submissions, series, securities = [], [], []
    for order, path in enumerate(zips):
        with zipfile.ZipFile(path) as archive:
            sub = _tsv(archive, "NMFP_SUBMISSION.tsv", SUBMISSION_COLUMNS)
            sub["month"] = pd.to_datetime(sub["REPORTDATE"], format="%d-%b-%Y").dt.strftime("%Y-%m")
            sub["filed"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y")
            # A filing made after the information date was not known then, so it never competes.
            sub = sub.loc[sub["SUBMISSIONTYPE"].str.startswith("N-MFP") & sub["month"].isin(months) & sub["filed"].le(filed_through)]
            if sub.empty:
                continue
            accessions = set(sub["ACCESSION_NUMBER"])
            level = _tsv(archive, "NMFP_SERIESLEVELINFO.tsv", SERIES_COLUMNS)
            schedule = _tsv(archive, "NMFP_SCHPORTFOLIOSECURITIES.tsv", SECURITY_COLUMNS)
        submissions.append(sub.assign(zip=path.name, zip_order=order))
        series.append(level.loc[level["ACCESSION_NUMBER"].isin(accessions)].assign(zip=path.name))
        securities.append(schedule.loc[schedule["ACCESSION_NUMBER"].isin(accessions)].assign(zip=path.name))
    if not submissions:
        raise ValueError("no N-MFP filing in the zips reports in the requested months")
    sub = pd.concat(submissions, ignore_index=True)
    level = pd.concat(series, ignore_index=True).drop_duplicates(["ACCESSION_NUMBER", "zip"], keep="last")
    schedule = pd.concat(securities, ignore_index=True)
    rows = schedule.groupby(["ACCESSION_NUMBER", "zip"]).size().rename("schedule_rows")
    sub = sub.merge(level, on=["ACCESSION_NUMBER", "zip"], how="left").join(rows, on=["ACCESSION_NUMBER", "zip"])
    sub["schedule_rows"] = sub["schedule_rows"].fillna(0).astype(int)
    sub["complete"] = sub["schedule_rows"].gt(0) & pd.to_numeric(sub["NETASSETOFSERIES"], errors="coerce").notna()
    sub["fund_id"] = sub["SERIESID"].fillna(sub["ACCESSION_NUMBER"])
    sub = sub.sort_values(["fund_id", "month", "complete", "filed", "zip_order", "ACCESSION_NUMBER"])
    sub["selected"] = ~sub.duplicated(["fund_id", "month"], keep="last")
    sub["feeder"] = sub["FEEDERFUNDFLAG"].fillna("").str.upper().eq("Y")
    return sub, schedule


def _classify(schedule: pd.DataFrame, cusips: pd.DataFrame) -> pd.DataFrame:
    rows = schedule.copy()
    rows["cusip"] = rows["CUSIP_NUMBER"].fillna("").str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
    category = rows["INVESTMENTCATEGORY"].fillna("")
    repo = category.str.contains("Repurchase Agreement", case=False)
    rows["security_class"] = rows["cusip"].map(cusips["security_class"]).where(~repo)
    rows["class_source"] = rows["security_class"].notna().map({True: "mspd_cusip", False: ""})
    # Untyped rows filed as Treasury debt with a Treasury (912...) or blank CUSIP; any other
    # CUSIP in that category is another issuer's paper and stays unresolved.
    treasury_filed = rows["security_class"].isna() & category.eq(TREASURY_CATEGORY)
    open_rows = treasury_filed & (rows["cusip"].eq("") | rows["cusip"].str.startswith("912"))
    text = rows["NAMEOFISSUER"].fillna("") + " " + rows["TITLEOFISSUER"].fillna("")
    for name, pattern in TITLE_CLASSES:
        hit = open_rows & rows["security_class"].isna() & text.str.contains(pattern)
        rows.loc[hit, ["security_class", "class_source"]] = [name, "title"]
    prefix_bill = rows["cusip"].str.match(r"91279[67]")
    prefix_coupon = rows["cusip"].str.match(r"91282[8C]|912810") & ~text.str.contains("strip", case=False)
    rows.loc[open_rows & rows["security_class"].isna() & prefix_bill, ["security_class", "class_source"]] = ["bill", "cusip_prefix"]
    rows.loc[open_rows & rows["security_class"].isna() & prefix_coupon, ["security_class", "class_source"]] = [
        "fixed_coupon",
        "cusip_prefix",
    ]
    rows.loc[treasury_filed & rows["security_class"].isna(), ["security_class", "class_source"]] = ["unresolved", "none"]
    rows = rows.loc[rows["security_class"].notna()].copy()
    excluding = pd.to_numeric(rows["EXCLUDINGVALUEOFANYSPONSORSUPP"], errors="coerce")
    including = pd.to_numeric(rows["INCLUDINGVALUEOFANYSPONSORSUPP"], errors="coerce")
    rows["value_bil"] = excluding.fillna(including) / 1e9
    if rows["value_bil"].isna().any():
        raise ValueError(f"{int(rows['value_bil'].isna().sum())} Treasury schedule rows have no value")
    legal = pd.to_datetime(rows["FINALLEGALINVESTMENTMATURITYDA"], format="%d-%b-%Y", errors="coerce")
    rows["maturity_date"] = rows["cusip"].map(cusips["maturity_date"]).fillna(legal)
    return rows


def extract(
    zips: list[Path], *, mspd_table_3_market: Path, first_month: str, last_month: str, filed_through: str
) -> tuple[pd.DataFrame, dict]:
    month_index = pd.period_range(first_month, last_month, freq="M")
    if month_index.empty:
        raise ValueError(f"empty month range {first_month}..{last_month}")
    months = {period.strftime("%Y-%m") for period in month_index}
    month_ends = [period.to_timestamp(how="end").normalize() for period in month_index]
    cusips, outstanding_frns = _mspd_cusips(mspd_table_3_market, month_ends)
    sub, schedule = _read_filings(sorted(zips, key=lambda path: path.name), months, pd.Timestamp(filed_through))
    absent = sorted(months - set(sub["month"]))
    if absent:
        raise ValueError(f"the zips hold no N-MFP filing for report months {absent}")
    chosen = sub.loc[sub["selected"], ["ACCESSION_NUMBER", "zip", "fund_id", "month", "feeder", "complete"]]
    incomplete = chosen.loc[~chosen["complete"]]
    if not incomplete.empty:
        raise ValueError(f"selected filings without a complete schedule: {sorted(incomplete['ACCESSION_NUMBER'])}")
    held = schedule.merge(chosen, on=["ACCESSION_NUMBER", "zip"], how="inner")
    treasury = _classify(held, cusips)
    feeders = treasury.loc[treasury["feeder"]]
    direct = treasury.loc[~treasury["feeder"]].copy()
    direct["report_date"] = pd.PeriodIndex(direct["month"], freq="M").to_timestamp(how="end").normalize()
    grouped = direct.groupby(["report_date", "cusip", "security_class"], dropna=False)
    holdings = grouped.agg(
        maturity_date=("maturity_date", "min"),
        value_bil=("value_bil", "sum"),
        class_source=("class_source", lambda sources: ";".join(sorted(set(sources)))),
        series_count=("fund_id", "nunique"),
    ).reset_index()
    if holdings.duplicated(["report_date", "cusip"]).any():
        raise ValueError("one CUSIP is typed as two security classes in one report month")
    held_pairs = set(zip(holdings["report_date"].dt.date.astype(str), holdings["cusip"]))
    zeros = pd.DataFrame(
        [
            {
                "report_date": pd.Timestamp(day),
                "cusip": cusip,
                "security_class": "frn",
                "maturity_date": cusips.loc[cusip, "maturity_date"],
                "value_bil": 0.0,
                "class_source": "mspd_outstanding_not_held",
                "series_count": 0,
            }
            for day, frns in outstanding_frns.items()
            for cusip in frns
            if (day, cusip) not in held_pairs
        ],
        columns=list(holdings.columns),
    )
    holdings = pd.concat([holdings, zeros], ignore_index=True).sort_values(["report_date", "security_class", "cusip"]).reset_index(drop=True)
    holdings = holdings.astype({"value_bil": float, "series_count": int})
    holdings["report_date"] = pd.to_datetime(holdings["report_date"]).dt.date.astype(str)
    holdings["maturity_date"] = pd.to_datetime(holdings["maturity_date"]).dt.date.astype("string").fillna("")
    by_month = direct.groupby(["month", "class_source"])["value_bil"].sum().unstack(fill_value=0.0)
    provenance = {
        "producer": "scripts/extract_sec_nmfp_mmf_holdings.py",
        "report_months": [first_month, last_month],
        "filed_through": filed_through,
        "value_basis": VALUE_BASIS,
        "zips": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in sorted(zips)},
        "mspd_table_3_market": {"file": mspd_table_3_market.name, "sha256": _sha256(mspd_table_3_market)},
        "filings_by_month": {
            month: {
                "candidates": int(len(group)),
                "selected": int(group["selected"].sum()),
                "selected_feeders": int((group["selected"] & group["feeder"]).sum()),
                "selected_without_complete_schedule": int((group["selected"] & ~group["complete"]).sum()),
            }
            for month, group in sub.groupby("month")
        },
        "feeder_treasury_value_bil_dropped": feeders.groupby("month")["value_bil"].sum().round(6).to_dict(),
        "treasury_value_bil_by_class_source": {
            month: {source: round(float(value), 6) for source, value in row.items()} for month, row in by_month.iterrows()
        },
        "selected_filings": {
            month: [
                [row.fund_id, row.ACCESSION_NUMBER, row.filed.date().isoformat(), row.zip]
                for row in group.sort_values(["fund_id", "ACCESSION_NUMBER"]).itertuples(index=False)
            ]
            for month, group in sub.loc[sub["selected"]].groupby("month")
        },
        "frn_zero_rows_by_month": {day: int(zeros["report_date"].eq(pd.Timestamp(day)).sum()) for day in outstanding_frns},
        "rows": int(len(holdings)),
    }
    return holdings[OUTPUT_COLUMNS], provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("zips", nargs="+", type=Path, help="SEC DERA Form N-MFP data-set zips")
    parser.add_argument("--mspd-table-3-market", type=Path, required=True)
    parser.add_argument("--first-month", required=True, help="first report month, YYYY-MM")
    parser.add_argument("--last-month", required=True, help="last report month, YYYY-MM")
    parser.add_argument("--filed-through", required=True, help="information date, YYYY-MM-DD: later filings are ignored")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    holdings, provenance = extract(
        args.zips,
        mspd_table_3_market=args.mspd_table_3_market,
        first_month=args.first_month,
        last_month=args.last_month,
        filed_through=args.filed_through,
    )
    holdings.to_csv(args.output, index=False, float_format="%.9f")
    provenance["output"] = {"file": args.output.name, "sha256": _sha256(args.output), "bytes": args.output.stat().st_size}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: provenance[key] for key in ("output", "rows", "feeder_treasury_value_bil_dropped")}, indent=2))


if __name__ == "__main__":
    main()
