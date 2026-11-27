"""Federal Reserve remittance source and allocation tests."""

from decimal import Decimal
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import pytest

from forecast_bundle_builders import (
    build_fed_remittance_path_rows,
    build_primary_deficit_path_rows,
    cbo_remittance_source_fixture,
    read_dts_fed_earnings,
)
from simulation_calendar import build_simulation_calendar


def test_table_4_3_annual_values_reconcile_to_printed_totals_at_reporting_precision():
    source = cbo_remittance_source_fixture()
    for period, printed in source["published_totals_bil"].items():
        start, end = map(int, period.split("-"))
        total = sum(Decimal(source["levels_bil"][str(year)]) for year in range(start, end + 1))
        # Both the annual entries and their separately printed sum round to tenths.
        assert abs(total - Decimal(printed)) <= Decimal("0.05") * (end - start + 2)
    assert sum(Decimal(source["levels_bil"][str(year)]) for year in range(2027, 2037)) == Decimal("928.4")


@pytest.mark.integration
def test_table_4_3_transcription_matches_the_pdf():
    pdf_setting = os.environ.get("CBO_REMITTANCE_PDF")
    assert pdf_setting, "set CBO_REMITTANCE_PDF to the official CBO outlook PDF"
    pdf = Path(pdf_setting)
    assert pdf.is_file(), f"CBO outlook PDF is missing: {pdf}"
    pdftotext = shutil.which("pdftotext")
    assert pdftotext, "pdftotext is required to verify the CBO remittance transcription"
    source = cbo_remittance_source_fixture()
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == source["source_sha256"]
    page = str(source["pdf_page"])
    extracted = subprocess.run(
        [pdftotext, "-f", page, "-l", page, "-layout", str(pdf), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "Table 4-3" in extracted
    rows = [line.strip() for line in extracted.splitlines() if line.strip().startswith(source["row"])]
    assert len(rows) == 1
    values = rows[0].removeprefix(source["row"]).split()
    assert values[1:] == [*source["levels_bil"].values(), *source["published_totals_bil"].values()]


def test_remittance_remainder_uses_splice_and_primary_business_day_weights(tmp_path):
    source = tmp_path / "dts.csv"
    pd.DataFrame([
        {
            "record_date": day, "account_type": "Treasury General Account (TGA)", "table_nbr": "II",
            "transaction_catg": "Federal Reserve Earnings", "transaction_type": "Deposits",
            "transaction_fytd_amt": value,
        }
        for day, value in (("2026-05-29", "5562"), ("2026-06-18", "6559"), ("2026-06-22", "7901"))
    ]).to_csv(source, index=False)
    primary = build_primary_deficit_path_rows(
        scenario_id="baseline", periods=build_simulation_calendar("2026-06-21", "2027-09-30", "daily"),
        primary_deficit_by_fiscal_year_bil={2026: 290.0, 2027: 500.0},
    )
    for cutoff, expected_post_splice in (("2026-05-31", 2.338), ("2026-06-18", 1.341)):
        splice = {"fiscal_actuals_through": cutoff, "fiscal_year": 2026}
        remittances = build_fed_remittance_path_rows(primary, splice, read_dts_fed_earnings(source, splice))
        for year, expected in ((2026, expected_post_splice), (2027, 8.6)):
            assert sum(
                row["remittance_bil"] for row in remittances if row["source_fiscal_year"] == year
            ) == pytest.approx(expected)
        assert all(
            row["remittance_bil"] == 0
            for row, primary_row in zip(remittances, primary, strict=True)
            if primary_row["day_count_weight"] == 0
        )
    splice["fiscal_actuals_through"] = "2026-06-22"
    with pytest.raises(ValueError, match="remaining FY2026 Fed remittances are negative"):
        build_fed_remittance_path_rows(primary, splice, read_dts_fed_earnings(source, splice))
    splice["fiscal_actuals_through"] = "2026-06-23"
    with pytest.raises(ValueError, match="expected one DTS Federal Reserve Earnings row"):
        read_dts_fed_earnings(source, splice)
