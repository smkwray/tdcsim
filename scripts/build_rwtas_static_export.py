#!/usr/bin/env python3
"""Build static TDCSim calibration exports for RateWall RWTAS."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


COUPON_CLASSES = ("Notes", "Bonds", "Inflation-Protected Securities")
MSPD_PATH = Path("data/historical_replay/raw/fiscaldata/mspd_table_3_market.csv")
CONFIG_PATH = Path("tdc_config_ratewall_source_backed.yaml")
OUTPUT_DIR = Path("output/rwtas_export")
MANIFEST_PATH = Path("data/historical_replay/manifest.json")


@dataclass(frozen=True)
class ExportSummary:
    as_of_date: str
    source_vintage: str
    total_coupon_stock_bil: float
    first_12m_runoff_bil: float
    first_12m_runoff_share: float
    bill_share: float
    issuance_mix_basis: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mspd", type=Path, default=MSPD_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    schedule, as_of_date, source_vintage, total_stock_bil, first_12m_bil = build_coupon_roll_schedule(args.mspd)
    mix, bill_share = build_issuance_tenor_mix(args.config)

    schedule_path = args.output_dir / "coupon_roll_schedule.csv"
    mix_path = args.output_dir / "issuance_tenor_mix.csv"
    readme_path = args.output_dir / "README.md"

    schedule.to_csv(schedule_path, index=False)
    mix.to_csv(mix_path, index=False)

    summary = ExportSummary(
        as_of_date=as_of_date,
        source_vintage=source_vintage,
        total_coupon_stock_bil=total_stock_bil,
        first_12m_runoff_bil=first_12m_bil,
        first_12m_runoff_share=first_12m_bil / total_stock_bil if total_stock_bil else 0.0,
        bill_share=bill_share,
        issuance_mix_basis="policy_assumption",
    )
    write_readme(readme_path, args.mspd, args.config, summary)

    print(f"wrote {schedule_path}")
    print(f"wrote {mix_path}")
    print(f"wrote {readme_path}")
    print(f"as_of_date={summary.as_of_date}")
    print(f"total_coupon_stock_bil={summary.total_coupon_stock_bil:.6f}")
    print(f"first_12m_runoff_share={summary.first_12m_runoff_share:.6%}")
    print(f"issuance_bill_share={summary.bill_share:.6%}")


def build_coupon_roll_schedule(mspd_path: Path) -> tuple[pd.DataFrame, str, str, float, float]:
    raw = pd.read_csv(mspd_path, low_memory=False)
    raw["record_date"] = pd.to_datetime(raw["record_date"], errors="raise")
    as_of = raw["record_date"].max()
    current = raw.loc[raw["record_date"].eq(as_of)].copy()

    is_coupon = current["security_class1_desc"].isin(COUPON_CLASSES)
    is_cusip = current["security_class2_desc"].astype(str).str.fullmatch(r"[0-9A-Z]{9}")
    current = current.loc[is_coupon & is_cusip].copy()
    current["outstanding_amt"] = pd.to_numeric(current["outstanding_amt"], errors="coerce")
    current["maturity_date"] = pd.to_datetime(current["maturity_date"], errors="coerce")
    current = current.loc[current["outstanding_amt"].notna() & current["maturity_date"].notna()].copy()

    total_stock_mil = float(current["outstanding_amt"].sum())
    if total_stock_mil <= 0.0:
        raise ValueError("current coupon stock is empty after MSPD filters")

    start_month = (as_of + pd.offsets.MonthBegin(1)).to_period("M")
    months = pd.period_range(start=start_month, periods=120, freq="M")
    current["month"] = current["maturity_date"].dt.to_period("M")
    grouped = current.groupby("month", sort=True)["outstanding_amt"].sum()

    rows: list[dict[str, object]] = []
    cumulative_mil = 0.0
    source_vintage = _source_vintage(mspd_path, as_of)
    for month in months:
        maturing_mil = float(grouped.get(month, 0.0))
        cumulative_mil += maturing_mil
        rows.append(
            {
                "month": str(month),
                "maturing_principal_bil": round(maturing_mil / 1000.0, 6),
                "cumulative_share_of_current_stock": round(cumulative_mil / total_stock_mil, 10),
                "source_vintage": source_vintage,
            }
        )

    schedule = pd.DataFrame(rows)
    first_12m_bil = float(schedule.head(12)["maturing_principal_bil"].sum())
    return schedule, as_of.date().isoformat(), source_vintage, total_stock_mil / 1000.0, first_12m_bil


def build_issuance_tenor_mix(config_path: Path) -> tuple[pd.DataFrame, float]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    profile = config.get("treasury_issuance_profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{config_path} has no treasury_issuance_profile mapping")

    absolute_share = 0.0
    for category, spec in profile.items():
        if category == "remainder_maturity_years" or not isinstance(spec, dict):
            continue
        absolute_share += float(spec.get("target_percentage", 0.0))
    remainder = max(0.0, 1.0 - absolute_share)

    rows: list[dict[str, object]] = []
    bill_share = 0.0
    source_vintage = f"{config_path.name} treasury_issuance_profile"
    for category, spec in profile.items():
        if category == "remainder_maturity_years" or not isinstance(spec, dict):
            continue
        if "target_percentage" in spec:
            category_share = float(spec["target_percentage"])
        else:
            category_share = remainder * float(spec.get("target_percentage_of_remainder", 0.0))

        maturities = [float(value) for value in spec.get("maturities", [])]
        distribution = [float(value) for value in spec.get("maturity_distribution", [])]
        if len(maturities) != len(distribution) or not maturities:
            raise ValueError(f"{config_path} category {category!r} has invalid maturity_distribution")

        for maturity_years, maturity_share in zip(maturities, distribution, strict=True):
            share = category_share * maturity_share
            if category == "bills":
                bill_share += share
            rows.append(
                {
                    "tenor_bucket": _tenor_bucket(category, maturity_years),
                    "share_of_gross_issuance": round(share, 10),
                    "basis": "policy_assumption",
                    "source_vintage": source_vintage,
                }
            )

    frame = pd.DataFrame(rows)
    total = float(frame["share_of_gross_issuance"].sum())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"issuance mix shares sum to {total:.12f}, expected 1.0")
    return frame, bill_share


def _tenor_bucket(category: str, maturity_years: float) -> str:
    if maturity_years < 1.0:
        months = int(round(maturity_years * 12.0))
        suffix = f"{months}m"
    else:
        years = int(round(maturity_years))
        suffix = f"{years}y"
    return f"{category.lower()}_{suffix}"


def _source_vintage(mspd_path: Path, as_of: pd.Timestamp) -> str:
    parts = [f"FiscalData MSPD Table 3 Market record_date={as_of.date().isoformat()}"]
    manifest_path = MANIFEST_PATH
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return "; ".join(parts)

    target = mspd_path.as_posix()
    for record in manifest.get("records", []):
        if record.get("path") == target:
            retrieved_at = record.get("retrieved_at_utc")
            if retrieved_at:
                parts.append(f"retrieved_at_utc={retrieved_at}")
            break
    return "; ".join(parts)


def write_readme(path: Path, mspd_path: Path, config_path: Path, summary: ExportSummary) -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    text = f"""# RWTAS Static Export

This directory contains TDCSim static calibration inputs for RateWall RWTAS.

## Files

- `coupon_roll_schedule.csv`: monthly scheduled maturity runoff for the current outstanding coupon stock over the next 120 months. The stock includes MSPD CUSIP rows for Notes, Bonds, and Inflation-Protected Securities, and excludes bills, FRNs, Federal Financing Bank rows, and MSPD subtotal rows. Principal is face/outstanding value from MSPD, reported in billions of dollars.
- `issuance_tenor_mix.csv`: TDCSim's current RateWall source-backed gross issuance policy by native tenor bucket. Shares are policy assumptions, not a CBO or Treasury prescription.

## Provenance

- Coupon runoff source: `{mspd_path}`.
- Coupon runoff vintage: `{summary.source_vintage}`.
- Coupon stock as-of date: `{summary.as_of_date}`.
- Issuance policy source: `{config_path}`.
- Export generated at: `{generated_at}`.

## Summary

- Total coupon stock: `{summary.total_coupon_stock_bil:.6f}` billion.
- First 12 months scheduled runoff: `{summary.first_12m_runoff_bil:.6f}` billion.
- First 12 months scheduled runoff share: `{summary.first_12m_runoff_share:.6%}`.
- RWTAS blended approximation reference: about `10%` per year.
- Issuance bill share: `{summary.bill_share:.6%}`.

## Regeneration

```bash
uv run python scripts/build_rwtas_static_export.py
```

Consumer note: read by RateWall as a static calibration input; regenerate on demand, no live coupling.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
