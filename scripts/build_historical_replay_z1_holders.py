"""Build a full Z.1 L.210 Treasury holder panel for historical replay."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_replay" / "processed" / "z1_treasury_holders_l210_full.csv"
RAW_ZIP_CANDIDATES = [
    ROOT / "data" / "historical_replay" / "raw" / "z1" / "z1_csv_files.zip",
    ROOT.parent / "buycurve" / "data" / "raw" / "validation" / "z1" / "z1_csv_files.zip",
]


# Top-level L.210 lines only. Detail lines such as "Treasury bills" under a
# holder class are excluded so broad holder totals do not double count.
L210_CODES = {
    "313161105": ("total_treasury_liabilities", "total"),
    "313161110": ("total_treasury_bills", "total"),
    "313161275": ("total_treasury_notes_bonds_tips", "total"),
    "893061105": ("all_sector_treasury_assets", "total"),
    "153061105": ("household_sector", "individuals"),
    "103061103": ("nonfinancial_corporate_business", "domestic_nonbank_other"),
    "113061003": ("nonfinancial_noncorporate_business", "domestic_nonbank_other"),
    "213061103": ("state_local_governments", "state_local_government"),
    "713061103": ("monetary_authority", "federal_reserve"),
    "763061100": ("us_chartered_depository_institutions", "banks"),
    "753061103": ("foreign_banking_offices_in_us", "banks"),
    "743061103": ("banks_in_us_affiliated_areas", "banks"),
    "473061105": ("credit_unions", "banks"),
    "513061105": ("property_casualty_insurance", "pensions_insurers"),
    "543061105": ("life_insurance", "pensions_insurers"),
    "573061105": ("private_pension_funds", "pensions_insurers"),
    "343061105": ("federal_government_pension_funds", "pensions_insurers"),
    "223061143": ("state_local_government_pension_funds", "pensions_insurers"),
    "633061105": ("money_market_funds", "money_market_cash"),
    "653061105": ("mutual_funds", "investment_funds"),
    "553061103": ("closed_end_funds", "investment_funds"),
    "563061103": ("exchange_traded_funds", "investment_funds"),
    "403061105": ("government_sponsored_enterprises", "government_sponsored_enterprises"),
    "673061103": ("abs_issuers", "abs_issuers"),
    "663061105": ("brokers_and_dealers", "dealers"),
    "733061103": ("holding_companies", "other_financial"),
    "503061123": ("other_financial_business_central_clearing_counterparties", "other_financial"),
    "263061105": ("rest_of_world", "foreign_international"),
}


def _raw_zip_path() -> Path:
    for path in RAW_ZIP_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in RAW_ZIP_CANDIDATES)
    raise FileNotFoundError(f"Could not find Z.1 raw zip. Checked:\n{candidates}")


def _z1_code(column: str) -> str:
    return column.split(".")[0][2:]


def _quarter_to_timestamp(value: str) -> pd.Timestamp:
    year, quarter = str(value).split(":Q")
    return pd.Timestamp(year=int(year), month=int(quarter) * 3, day=1)


def _normalize_member(raw_zip: Path, member: str, measure: str) -> pd.DataFrame:
    with zipfile.ZipFile(raw_zip) as archive:
        with archive.open(member) as handle:
            frame = pd.read_csv(handle, low_memory=False)

    rows: list[pd.DataFrame] = []
    work = frame.copy()
    work["quarter"] = work["date"].map(_quarter_to_timestamp)
    for column in [col for col in work.columns if col not in {"date", "quarter"}]:
        code = _z1_code(column)
        if code not in L210_CODES:
            continue
        sector, broad_class = L210_CODES[code]
        values = pd.to_numeric(work[column].replace("ND", pd.NA), errors="coerce") * 1_000_000
        rows.append(
            pd.DataFrame(
                {
                    "quarter": work["quarter"],
                    "z1_series": column,
                    "z1_code": code,
                    "z1_sector": sector,
                    "broad_investor_class": broad_class,
                    "measure": measure,
                    "value": values,
                    "source_file": f"{raw_zip.name}::{member}",
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def main() -> None:
    raw_zip = _raw_zip_path()
    panels = [
        _normalize_member(raw_zip, "csv/l210.csv", "level"),
        _normalize_member(raw_zip, "csv/f210.csv", "transaction"),
    ]
    out = pd.concat(panels, ignore_index=True)
    out = out.sort_values(["quarter", "measure", "z1_sector"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
