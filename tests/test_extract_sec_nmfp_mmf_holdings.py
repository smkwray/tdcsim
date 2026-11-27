"""The N-MFP extract that fixes the opening money-fund FRN book and the FRN auction share."""

from __future__ import annotations

import zipfile

import pandas as pd
import pytest

from scripts.extract_sec_nmfp_mmf_holdings import extract

MSPD_ROWS = [
    # record date, class, CUSIP, maturity, outstanding (millions; blank on reopening detail rows)
    ("2026-03-31", "Floating Rate Notes", "912828ZZ1", "2027-04-30", "60000"),
    ("2026-04-30", "Floating Rate Notes", "912828ZZ1", "2027-04-30", "60000"),
    ("2026-04-30", "Floating Rate Notes", "912828ZZ1", "2027-04-30", ""),
    # Outstanding but held by no fund: a verified zero, not missing coverage.
    ("2026-04-30", "Floating Rate Notes", "912828YY2", "2027-07-31", "30000"),
    ("2026-04-30", "Bills Maturity Value", "912797AA1", "2026-07-30", "50000"),
    ("2026-04-30", "Notes", "91282CAA1", "2026-05-15", "20000"),
]


def _mspd(tmp_path):
    path = tmp_path / "mspd_table_3_market.csv"
    pd.DataFrame(
        [
            {
                "record_date": day,
                "security_class1_desc": kind,
                "security_class2_desc": cusip,
                "maturity_date": maturity,
                "outstanding_amt": outstanding,
            }
            for day, kind, cusip, maturity, outstanding in MSPD_ROWS
        ]
    ).to_csv(path, index=False)
    return path


def _zip(path, filings):
    """Write an N-MFP data-set zip. ``filings`` rows: (accession, series, filed, feeder, holdings)."""

    submissions, series, holdings = [], [], []
    for accession, series_id, filed, feeder, rows in filings:
        submissions.append(
            {
                "ACCESSION_NUMBER": accession,
                "FILING_DATE": filed,
                "SUBMISSIONTYPE": "N-MFP3",
                "REPORTDATE": "30-APR-2026",
                "SERIESID": series_id,
            }
        )
        series.append({"ACCESSION_NUMBER": accession, "FEEDERFUNDFLAG": feeder, "NETASSETOFSERIES": "1000"})
        for cusip, category, title, value in rows:
            holdings.append(
                {
                    "ACCESSION_NUMBER": accession,
                    "NAMEOFISSUER": "UNITED STATES TREASURY",
                    "TITLEOFISSUER": title,
                    "CUSIP_NUMBER": cusip,
                    "INVESTMENTCATEGORY": category,
                    "FINALLEGALINVESTMENTMATURITYDA": "30-JUL-2026",
                    "INCLUDINGVALUEOFANYSPONSORSUPP": str(value),
                    "EXCLUDINGVALUEOFANYSPONSORSUPP": str(value),
                }
            )
    with zipfile.ZipFile(path, "w") as archive:
        for name, frame in (
            ("NMFP_SUBMISSION.tsv", submissions),
            ("NMFP_SERIESLEVELINFO.tsv", series),
            ("NMFP_SCHPORTFOLIOSECURITIES.tsv", holdings),
        ):
            archive.writestr(name, pd.DataFrame(frame).to_csv(sep="\t", index=False))
    return path


TSY = "U.S. Treasury Debt"


def test_extract_counts_each_fund_once_from_its_latest_complete_filing(tmp_path):
    first = _zip(
        tmp_path / "20260508-20260605_nmfp.zip",
        [
            ("A-1", "S1", "05-MAY-2026", "N", [("912828ZZ1", TSY, "FRN", 5e9)]),
            ("B-1", "S2", "05-MAY-2026", "N", [
                ("912828ZZ1", TSY, "FRN", 2e9),
                # A when-issued bill MSPD does not list yet is typed by its title.
                ("912797ZZ9", TSY, "TREASURY BILL", 1e9),
                ("91282CAA1", TSY, "NOTE", 0.5e9),
                # Another issuer's note filed as Treasury debt is kept, unresolved, not typed as Treasury.
                ("3130AAAA1", TSY, "BONDS", 0.25e9),
                ("912828ZZ1", "U.S. Treasury Repurchase Agreement", "REPO", 9e9),
            ]),
            # A feeder's holdings belong to its master and must not count twice.
            ("C-1", "S3", "05-MAY-2026", "Y", [("912828ZZ1", TSY, "FRN", 7e9)]),
        ],
    )
    # A later amendment for the same fund and month replaces the original filing.
    amended = _zip(
        tmp_path / "20260608-20260708_nmfp.zip",
        [("A-2", "S1", "10-JUN-2026", "N", [("912828ZZ1", TSY, "FRN", 6e9)])],
    )
    # An amendment filed after the information date was not known on it and must not compete.
    late = _zip(
        tmp_path / "20260608-20260708_late_nmfp.zip",
        [("A-3", "S1", "23-JUN-2026", "N", [("912828ZZ1", TSY, "FRN", 99e9)])],
    )
    holdings, provenance = extract(
        [late, amended, first],
        mspd_table_3_market=_mspd(tmp_path),
        first_month="2026-04",
        last_month="2026-04",
        filed_through="2026-06-22",
    )
    rows = holdings.set_index("cusip")

    assert rows.loc["912828ZZ1", "value_bil"] == pytest.approx(8.0)
    assert rows.loc["912828ZZ1", "security_class"] == "frn"
    assert rows.loc["912828ZZ1", "series_count"] == 2
    assert rows.loc["912797ZZ9", ["security_class", "class_source"]].tolist() == ["bill", "title"]
    assert rows.loc["91282CAA1", ["security_class", "maturity_date"]].tolist() == ["fixed_coupon", "2026-05-15"]
    assert rows.loc["3130AAAA1", "security_class"] == "unresolved"
    assert rows.loc["912828YY2", ["security_class", "value_bil", "class_source", "series_count"]].tolist() == [
        "frn", 0.0, "mspd_outstanding_not_held", 0,
    ]
    assert provenance["frn_zero_rows_by_month"] == {"2026-04-30": 1}
    assert set(holdings["report_date"]) == {"2026-04-30"}
    assert provenance["filings_by_month"]["2026-04"] == {
        "candidates": 4,
        "selected": 3,
        "selected_feeders": 1,
        "selected_without_complete_schedule": 0,
    }
    assert provenance["feeder_treasury_value_bil_dropped"] == {"2026-04": pytest.approx(7.0)}
    assert provenance["filed_through"] == "2026-06-22"
    assert provenance["selected_filings"]["2026-04"] == [
        ["S1", "A-2", "2026-06-10", amended.name],
        ["S2", "B-1", "2026-05-05", first.name],
        ["S3", "C-1", "2026-05-05", first.name],
    ]


def test_extract_fails_on_a_month_without_filings_or_mspd(tmp_path):
    archive = _zip(tmp_path / "20260508-20260605_nmfp.zip", [("A-1", "S1", "05-MAY-2026", "N", [("912828ZZ1", TSY, "FRN", 5e9)])])
    with pytest.raises(ValueError, match=r"no N-MFP filing for report months \['2026-03'\]"):
        extract([archive], mspd_table_3_market=_mspd(tmp_path), first_month="2026-03", last_month="2026-04", filed_through="2026-06-22")
    # A selected filing without a schedule would turn its fund's holdings into silent zeros.
    empty = _zip(
        tmp_path / "20260608-20260708_nmfp.zip",
        [("D-1", "S4", "05-MAY-2026", "N", []), ("E-1", "S5", "05-MAY-2026", "N", [("912828ZZ1", TSY, "FRN", 1e9)])],
    )
    with pytest.raises(ValueError, match="without a complete schedule: \\['D-1'\\]"):
        extract([archive, empty], mspd_table_3_market=_mspd(tmp_path), first_month="2026-04", last_month="2026-04", filed_through="2026-06-22")
    with pytest.raises(ValueError, match="no record date"):
        extract([archive], mspd_table_3_market=_mspd(tmp_path), first_month="2026-04", last_month="2026-05", filed_through="2026-06-22")
