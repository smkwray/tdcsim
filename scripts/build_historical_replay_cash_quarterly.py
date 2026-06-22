"""Build the quarterly operating-cash panel for historical replay."""

from __future__ import annotations

import csv
from calendar import monthrange
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "historical_replay"
DAILY_CASH = BASE / "processed" / "treasury_operating_cash_daily.csv"
Z1_LEVEL = BASE / "raw" / "fred" / "treasury_operating_cash_level__BOGZ1FL313024000Q.csv"
Z1_TX = BASE / "raw" / "fred" / "treasury_operating_cash_tx__BOGZ1FU313024000Q.csv"
H41_TGA_LEVEL = BASE / "raw" / "fred" / "tga_wednesday_level__WDTGAL.csv"
H41_TGA_AVG = BASE / "raw" / "fred" / "tga_week_average__WTREGEN.csv"
OUT = BASE / "processed" / "treasury_operating_cash_quarterly.csv"


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def quarter_for(day: date) -> str:
    return f"{day.year}Q{((day.month - 1) // 3) + 1}"


def quarter_end_for(day: date) -> date:
    q_month = ((day.month - 1) // 3 + 1) * 3
    return date(day.year, q_month, monthrange(day.year, q_month)[1])


def clean_number(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "."}:
        return None
    return float(text)


def load_fred_quarterly(path: Path, value_col: str, out_col: str) -> dict[str, dict[str, str | float]]:
    rows: dict[str, dict[str, str | float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            obs_date = parse_date(row["observation_date"])
            value = clean_number(row.get(value_col))
            if value is None:
                continue
            q = quarter_for(obs_date)
            rows[q] = {
                "quarter": q,
                "quarter_end": quarter_end_for(obs_date).isoformat(),
                out_col: value,
            }
    return rows


def load_last_observation_by_quarter(path: Path, value_col: str, out_col: str) -> dict[str, dict[str, str | float]]:
    rows: dict[str, dict[str, str | float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            obs_date = parse_date(row["observation_date"])
            value = clean_number(row.get(value_col))
            if value is None:
                continue
            q = quarter_for(obs_date)
            existing_date = parse_date(rows[q]["source_date"]) if q in rows else None
            if existing_date is None or obs_date >= existing_date:
                rows[q] = {
                    "quarter": q,
                    "quarter_end": quarter_end_for(obs_date).isoformat(),
                    "source_date": obs_date.isoformat(),
                    out_col: value,
                }
    return rows


def load_dts_quarter_end() -> dict[str, dict[str, str | float]]:
    rows: dict[str, dict[str, str | float]] = {}
    with DAILY_CASH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            record_date = parse_date(row["record_date"])
            q = quarter_for(record_date)
            existing_date = parse_date(rows[q]["dts_source_date"]) if q in rows else None
            if existing_date is not None and record_date < existing_date:
                continue
            rows[q] = {
                "quarter": q,
                "quarter_end": quarter_end_for(record_date).isoformat(),
                "dts_source_date": record_date.isoformat(),
                "dts_tga_mil": clean_number(row.get("tga_mil")) or 0.0,
                "dts_ttl_mil": clean_number(row.get("ttl_mil")) or 0.0,
                "dts_toc_tga_plus_ttl_mil": clean_number(row.get("toc_tga_plus_ttl_mil")) or 0.0,
                "dts_short_term_cash_investments_mil": clean_number(row.get("short_term_cash_investments_mil")) or 0.0,
                "dts_supplementary_financing_program_mil": clean_number(
                    row.get("supplementary_financing_program_mil")
                )
                or 0.0,
            }
    return rows


def main() -> int:
    dts = load_dts_quarter_end()
    z1_level = load_fred_quarterly(Z1_LEVEL, "BOGZ1FL313024000Q", "z1_treasury_operating_cash_level_mil")
    z1_tx = load_fred_quarterly(Z1_TX, "BOGZ1FU313024000Q", "z1_treasury_operating_cash_tx_mil")
    h41_level = load_last_observation_by_quarter(H41_TGA_LEVEL, "WDTGAL", "h41_tga_wednesday_level_mil")
    h41_avg = load_last_observation_by_quarter(H41_TGA_AVG, "WTREGEN", "h41_tga_week_average_mil")

    quarters = sorted(set(dts) | set(z1_level) | set(z1_tx) | set(h41_level) | set(h41_avg))
    columns = [
        "quarter",
        "quarter_end",
        "selected_operating_cash_level_mil",
        "selected_operating_cash_level_source",
        "selected_operating_cash_tx_mil",
        "selected_operating_cash_tx_source",
        "dts_source_date",
        "dts_tga_mil",
        "dts_ttl_mil",
        "dts_toc_tga_plus_ttl_mil",
        "dts_short_term_cash_investments_mil",
        "dts_supplementary_financing_program_mil",
        "z1_treasury_operating_cash_level_mil",
        "z1_treasury_operating_cash_tx_mil",
        "h41_tga_source_date",
        "h41_tga_wednesday_level_mil",
        "h41_tga_week_average_mil",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for q in quarters:
            row: dict[str, str | float] = {
                "quarter": q,
                "quarter_end": (
                    dts.get(q, {}).get("quarter_end")
                    or z1_level.get(q, {}).get("quarter_end")
                    or z1_tx.get(q, {}).get("quarter_end")
                    or h41_level.get(q, {}).get("quarter_end")
                    or h41_avg.get(q, {}).get("quarter_end")
                    or ""
                ),
            }
            row.update(z1_level.get(q, {}))
            row.update(z1_tx.get(q, {}))
            row.update(dts.get(q, {}))
            if q in h41_level:
                row["h41_tga_source_date"] = h41_level[q]["source_date"]
                row["h41_tga_wednesday_level_mil"] = h41_level[q]["h41_tga_wednesday_level_mil"]
            if q in h41_avg:
                row["h41_tga_source_date"] = row.get("h41_tga_source_date", h41_avg[q]["source_date"])
                row["h41_tga_week_average_mil"] = h41_avg[q]["h41_tga_week_average_mil"]

            if row.get("dts_toc_tga_plus_ttl_mil") not in (None, ""):
                row["selected_operating_cash_level_mil"] = row["dts_toc_tga_plus_ttl_mil"]
                row["selected_operating_cash_level_source"] = "dts_quarter_end_tga_plus_ttl"
            elif row.get("z1_treasury_operating_cash_level_mil") not in (None, ""):
                row["selected_operating_cash_level_mil"] = row["z1_treasury_operating_cash_level_mil"]
                row["selected_operating_cash_level_source"] = "z1_quarterly_treasury_operating_cash_level"
            elif row.get("h41_tga_wednesday_level_mil") not in (None, ""):
                row["selected_operating_cash_level_mil"] = row["h41_tga_wednesday_level_mil"]
                row["selected_operating_cash_level_source"] = "h41_quarter_end_tga_wednesday_level"

            if row.get("z1_treasury_operating_cash_tx_mil") not in (None, ""):
                row["selected_operating_cash_tx_mil"] = row["z1_treasury_operating_cash_tx_mil"]
                row["selected_operating_cash_tx_source"] = "z1_quarterly_treasury_operating_cash_transaction"

            writer.writerow({column: row.get(column, "") for column in columns})

    print(f"wrote {OUT.relative_to(ROOT)} rows={len(quarters)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
