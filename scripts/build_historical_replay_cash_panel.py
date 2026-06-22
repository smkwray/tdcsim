"""Build the daily operating-cash panel for historical replay."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "historical_replay" / "raw" / "fiscaldata" / "dts_operating_cash_balance.csv"
OUT = ROOT / "data" / "historical_replay" / "processed" / "treasury_operating_cash_daily.csv"


def amount(row: dict) -> float:
    for key in ("close_today_bal", "open_today_bal"):
        value = str(row.get(key, "")).strip()
        if value and value.lower() != "null":
            return float(value)
    return 0.0


def classify(account_type: str) -> str | None:
    label = account_type.lower()
    if "closing balance" in label and "tga" in label:
        return "tga_mil"
    if label in {"federal reserve account", "treasury general account (tga)"}:
        return "tga_mil"
    if "tax and loan" in label or "financial institution accoun" in label:
        return "ttl_mil"
    if "short-term cash investments" in label:
        return "short_term_cash_investments_mil"
    if "supplementary financing" in label:
        return "supplementary_financing_program_mil"
    return None


def main() -> int:
    by_date: dict[str, dict[str, float | str]] = defaultdict(
        lambda: {
            "tga_mil": 0.0,
            "ttl_mil": 0.0,
            "short_term_cash_investments_mil": 0.0,
            "supplementary_financing_program_mil": 0.0,
            "source_rows": 0,
        }
    )
    with RAW.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record_date = row["record_date"]
            bucket = by_date[record_date]
            bucket["record_date"] = record_date
            bucket["record_fiscal_year"] = row.get("record_fiscal_year", "")
            bucket["record_calendar_year"] = row.get("record_calendar_year", "")
            bucket["record_calendar_quarter"] = row.get("record_calendar_quarter", "")
            target = classify(row.get("account_type", ""))
            if target:
                bucket[target] = float(bucket[target]) + amount(row)
            bucket["source_rows"] = int(bucket["source_rows"]) + 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "record_date",
        "record_fiscal_year",
        "record_calendar_year",
        "record_calendar_quarter",
        "tga_mil",
        "ttl_mil",
        "toc_tga_plus_ttl_mil",
        "short_term_cash_investments_mil",
        "supplementary_financing_program_mil",
        "source_rows",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record_date in sorted(by_date):
            row = by_date[record_date]
            row["toc_tga_plus_ttl_mil"] = float(row["tga_mil"]) + float(row["ttl_mil"])
            writer.writerow(row)
    print(f"wrote {OUT.relative_to(ROOT)} rows={len(by_date)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
