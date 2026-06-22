"""Download the raw data bundle needed for tdcsim historical replay."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "historical_replay"
FISCALDATA_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
RETRIEVED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


FISCALDATA_TABLES = [
    ("dts_operating_cash_balance", "v1/accounting/dts/operating_cash_balance", {}),
    ("dts_deposits_withdrawals_operating_cash", "v1/accounting/dts/deposits_withdrawals_operating_cash", {}),
    ("dts_public_debt_transactions", "v1/accounting/dts/public_debt_transactions", {}),
    ("auctions_query", "v1/accounting/od/auctions_query", {}),
    ("mspd_table_1", "v1/debt/mspd/mspd_table_1", {}),
    ("mspd_table_3_market", "v1/debt/mspd/mspd_table_3_market", {}),
    ("avg_interest_rates", "v2/accounting/od/avg_interest_rates", {}),
    ("debt_outstanding", "v2/accounting/od/debt_outstanding", {}),
    ("interest_expense", "v2/accounting/od/interest_expense", {}),
    ("frn_daily_indexes", "v1/accounting/od/frn_daily_indexes", {}),
    ("tips_cpi_data_detail", "v1/accounting/od/tips_cpi_data_detail", {}),
    ("mts_table_1", "v1/accounting/mts/mts_table_1", {}),
    ("mts_table_4_receipts", "v1/accounting/mts/mts_table_4", {}),
    ("mts_table_5_outlays", "v1/accounting/mts/mts_table_5", {}),
]


FRED_SERIES = [
    ("tga_wednesday_level", "WDTGAL"),
    ("tga_week_average", "WTREGEN"),
    ("reserve_balances_wednesday", "WRBWFRBL"),
    ("reserve_balances_week_average", "WRESBAL"),
    ("reverse_repos_total", "WLRRAL"),
    ("reverse_repos_others", "WLRRAOL"),
    ("on_rrp_daily_total", "RRPONTSYD"),
    ("reverse_repo_treasury_daily", "RRPTSYD"),
    ("fed_remit_or_deferred", "RESPPLLOPNWW"),
    ("soma_treasury_holdings", "WSHOMCB"),
    ("commercial_bank_deposits_weekly_sa", "DPSACBW027SBOG"),
    ("commercial_bank_deposits_monthly_nsa", "DPSACBM027NBOG"),
    ("bank_treasury_agency_securities_weekly_sa", "TASACBW027SBOG"),
    ("commercial_bank_cash_assets_weekly_sa", "CASACBW027SBOG"),
    ("commercial_bank_cash_assets_monthly_sa", "CASACBM027SBOG"),
    ("bank_credit", "TOTBKCR"),
    ("loans_and_leases_bank_credit", "LOANS"),
    ("securities_in_bank_credit", "INVEST"),
    ("treasury_agency_non_mbs_bank_securities", "TNMACBM027SBOG"),
    ("foreign_related_treasury_agency_non_mbs", "TNMFRIM027SBOG"),
    ("m2", "M2SL"),
    ("currency", "CURRCIR"),
    ("retail_money_market_funds", "RMFSL"),
    ("small_time_deposits", "STDSL"),
    ("large_time_deposits_all_commercial_banks", "LTDACBM027SBOG"),
    ("other_deposits_all_commercial_banks", "ODSACBM027SBOG"),
    ("term_deposits_at_fed", "TERMT"),
    ("other_deposits_at_fed", "WLODLL"),
    ("foreign_official_custody_treasuries", "WMTSEC1"),
    ("fed_funds_effective_rate", "FEDFUNDS"),
    ("iorb", "IORB"),
    ("sofr", "SOFR"),
    ("tbill_3m_monthly", "TB3MS"),
    ("treasury_10y_monthly", "GS10"),
    ("treasury_1m_daily", "DGS1MO"),
    ("treasury_3m_daily", "DGS3MO"),
    ("treasury_6m_daily", "DGS6MO"),
    ("treasury_1y_daily", "DGS1"),
    ("treasury_2y_daily", "DGS2"),
    ("treasury_3y_daily", "DGS3"),
    ("treasury_5y_daily", "DGS5"),
    ("treasury_7y_daily", "DGS7"),
    ("treasury_10y_daily", "DGS10"),
    ("treasury_20y_daily", "DGS20"),
    ("treasury_30y_daily", "DGS30"),
    ("tips_5y_daily", "DFII5"),
    ("tips_7y_daily", "DFII7"),
    ("tips_10y_daily", "DFII10"),
    ("tips_20y_daily", "DFII20"),
    ("tips_30y_daily", "DFII30"),
    ("cpi_all_urban", "CPIAUCSL"),
    ("fed_tsy_tx", "BOGZ1FU713061103Q"),
    ("fed_tsy_level", "BOGZ1FL713061103Q"),
    ("us_chartered_tsy_tx", "BOGZ1FU763061100Q"),
    ("us_chartered_tsy_level", "BOGZ1FL763061100Q"),
    ("foreign_offices_tsy_tx", "BOGZ1FU753061103Q"),
    ("foreign_offices_tsy_level", "BOGZ1FL753061103Q"),
    ("affiliated_areas_tsy_tx", "BOGZ1FU743061103Q"),
    ("affiliated_areas_tsy_level", "BOGZ1FL743061103Q"),
    ("np_credit_unions_tsy_tx", "BOGZ1FU473061103Q"),
    ("np_credit_unions_tsy_level", "BOGZ1FL473061103Q"),
    ("corp_credit_unions_tsy_tx", "BOGZ1FU473061153Q"),
    ("corp_credit_unions_tsy_level", "BOGZ1FL473061153Q"),
    ("credit_unions_total_tsy_tx", "BOGZ1FU473061105Q"),
    ("credit_unions_total_tsy_level", "BOGZ1FL473061105Q"),
    ("row_tsy_tx", "BOGZ1FU263061105Q"),
    ("row_tsy_level", "BOGZ1LM263061105Q"),
    ("treasury_operating_cash_tx", "BOGZ1FU313024000Q"),
    ("treasury_operating_cash_level", "BOGZ1FL313024000Q"),
    ("mmf_tsy_tx", "BOGZ1FU633061105Q"),
    ("mmf_tsy_level", "BOGZ1FL633061105Q"),
    ("mmf_tsy_bills_level", "BOGZ1FL633061110Q"),
    ("gse_tsy_tx", "BOGZ1FA403061105Q"),
    ("gse_tsy_level", "BOGZ1FL403061105Q"),
    ("bea_row_fed_interest_paid_saar", "B093RC1Q027SBEA"),
    ("bea_row_taxes_received_saar", "W008RC1Q027SBEA"),
    ("bea_row_social_insurance_received_saar", "W781RC1Q027SBEA"),
    ("bea_row_current_transfer_receipts_received_saar", "LA0000281Q027SBEA"),
]


SIBLING_IMPORTS = [
    ("tdcest", "../tdcest/data/processed/tdc_estimates.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_components.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_tier2_regression_series.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_mmf_rrp_source_comparison.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_mmf_route_split_context.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_tdcsim_private_route_support_contract.csv"),
    ("tdcest", "../tdcest/data/processed/tdc_tdcsim_private_route_allocation_sensitivity.csv"),
    ("tdcest", "../tdcest/data/processed/ratewall_du_ru_methodology_panel.csv"),
    ("tdcest", "../tdcest/data/raw/support__fed_treasury_interest_components.csv"),
    ("tdcest", "../tdcest/data/raw/treasury__interest_expense.csv"),
    ("tdcest", "../tdcest/data/raw/treasury__frn_daily_indexes.csv"),
    ("buycurve", "../buycurve/data/clean/monthly_issuance_maturity_panel.csv"),
    ("buycurve", "../buycurve/data/clean/auction_allotment_panel_base_slim.csv"),
    ("buycurve", "../buycurve/data/interim/z1_treasury_holders_clean.csv"),
    ("tdcladder", "../tdcladder/data/interim/raw_treasury_supply_by_maturity.csv"),
    ("tdcladder", "../tdcladder/data/clean/monthly_ladder_panel.csv"),
    ("tdcladder", "../tdcladder/output/qa/treasury_stock_reconciliation.csv"),
    ("liqsub", "../liqsub/data/clean/monthly_liquidity_substitution_panel.csv"),
    ("liqsub", "../liqsub/data/clean/monthly_panel_qa.csv"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_url_bytes(url: str) -> bytes:
    result = subprocess.run(
        [
            "curl",
            "--http1.1",
            "-g",
            "-L",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--retry-all-errors",
            "--max-time",
            "180",
            "-sS",
            url,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"curl failed with exit {result.returncode}: {stderr}")
    return result.stdout


def read_json_url(url: str) -> dict:
    return json.loads(read_url_bytes(url).decode("utf-8"))


def write_csv(path: Path, rows: list[dict]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return columns


def csv_summary(path: Path) -> dict:
    row_count = 0
    first_date = ""
    last_date = ""
    columns: list[str] = []
    date_keys = ("record_date", "observation_date", "date", "auction_date", "issue_date", "index_date")
    date_values: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        date_key = next((key for key in date_keys if key in columns), None)
        for row in reader:
            row_count += 1
            if date_key:
                value = row.get(date_key, "").strip()
                if value:
                    date_values.append(value)
    if date_values:
        first_date = min(date_values)
        last_date = max(date_values)
    return {
        "rows": row_count,
        "columns": columns,
        "first_date": first_date,
        "last_date": last_date,
    }


def record_for(path: Path, **metadata) -> dict:
    summary = csv_summary(path) if path.suffix.lower() == ".csv" else {"rows": None, "columns": [], "first_date": "", "last_date": ""}
    return {
        **metadata,
        "path": str(path.relative_to(ROOT)),
        "rows": summary["rows"],
        "first_date": summary["first_date"],
        "last_date": summary["last_date"],
        "columns": summary["columns"],
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "retrieved_at_utc": RETRIEVED_AT,
    }


def download_fiscaldata_table(key: str, endpoint: str, extra_params: dict[str, str]) -> dict:
    params = {"page[size]": "10000", "format": "json", **extra_params}
    first_params = {**params, "page[number]": "1"}
    first_url = FISCALDATA_BASE + endpoint + "?" + urlencode(first_params)
    path = OUT / "raw" / "fiscaldata" / f"{key}.csv"
    if path.exists():
        return record_for(
            path,
            kind="fiscaldata",
            key=key,
            endpoint=endpoint,
            source_url=first_url,
            pages="existing",
            source_note="Existing completed file reused by resumable downloader.",
        )
    first = read_json_url(first_url)
    rows = list(first.get("data", []))
    total_pages = int(first.get("meta", {}).get("total-pages", 1))
    for page_number in range(2, total_pages + 1):
        page_params = {**params, "page[number]": str(page_number)}
        page_url = FISCALDATA_BASE + endpoint + "?" + urlencode(page_params)
        payload = read_json_url(page_url)
        rows.extend(payload.get("data", []))
    columns = write_csv(path, rows)
    return record_for(
        path,
        kind="fiscaldata",
        key=key,
        endpoint=endpoint,
        source_url=first_url,
        pages=total_pages,
        source_note="Downloaded from Treasury FiscalData API. Python SSL verification was disabled because this local environment rejects the Treasury certificate chain.",
        columns=columns,
    )


def download_fred_series(key: str, series_id: str) -> dict:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    path = OUT / "raw" / "fred" / f"{key}__{series_id}.csv"
    if path.exists():
        return record_for(path, kind="fred", key=key, series_id=series_id, source_url=url)
    data = read_url_bytes(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return record_for(path, kind="fred", key=key, series_id=series_id, source_url=url)


def download_nyfed_soma_treasury_holdings() -> dict:
    url = "https://markets.newyorkfed.org/api/soma/tsy/get/monthly.csv"
    path = OUT / "raw" / "nyfed" / "soma_treasury_holdings_monthly.csv"
    if path.exists():
        return record_for(path, kind="nyfed_soma", key="soma_treasury_holdings_monthly", source_url=url)
    data = read_url_bytes(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return record_for(path, kind="nyfed_soma", key="soma_treasury_holdings_monthly", source_url=url)


def copy_sibling_file(project: str, relative_source: str) -> dict:
    source = (ROOT / relative_source).resolve()
    path = OUT / "imported" / project / source.name
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, path)
    return record_for(
        path,
        kind="sibling_import",
        key=source.stem,
        source_project=project,
        source_path=str(source),
    )


def write_manifest(records: list[dict], errors: list[dict]) -> None:
    manifest = {
        "kind": "tdcsim_historical_replay_data_bundle",
        "retrieved_at_utc": RETRIEVED_AT,
        "root": str(ROOT),
        "data_dir": str(OUT.relative_to(ROOT)),
        "record_count": len(records),
        "error_count": len(errors),
        "records": records,
        "errors": errors,
    }
    path = OUT / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_path = OUT / "manifest.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "kind",
            "key",
            "series_id",
            "endpoint",
            "source_project",
            "path",
            "rows",
            "first_date",
            "last_date",
            "sha256",
            "bytes",
            "source_url",
            "source_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    records: list[dict] = []
    errors: list[dict] = []

    for key, endpoint, params in FISCALDATA_TABLES:
        try:
            record = download_fiscaldata_table(key, endpoint, params)
            records.append(record)
            print(f"fiscaldata {key}: {record['rows']} rows", flush=True)
        except Exception as exc:
            errors.append({"kind": "fiscaldata", "key": key, "endpoint": endpoint, "error": repr(exc)})
            print(f"ERROR fiscaldata {key}: {exc}", flush=True)

    for key, series_id in FRED_SERIES:
        try:
            record = download_fred_series(key, series_id)
            records.append(record)
            print(f"fred {key} {series_id}: {record['rows']} rows", flush=True)
        except Exception as exc:
            errors.append({"kind": "fred", "key": key, "series_id": series_id, "error": repr(exc)})
            print(f"ERROR fred {key} {series_id}: {exc}", flush=True)

    try:
        record = download_nyfed_soma_treasury_holdings()
        records.append(record)
        print(f"nyfed soma_treasury_holdings_monthly: {record['rows']} rows", flush=True)
    except Exception as exc:
        errors.append({"kind": "nyfed_soma", "key": "soma_treasury_holdings_monthly", "error": repr(exc)})
        print(f"ERROR nyfed soma_treasury_holdings_monthly: {exc}", flush=True)

    for project, source in SIBLING_IMPORTS:
        try:
            record = copy_sibling_file(project, source)
            records.append(record)
            print(f"import {project} {Path(source).name}: {record['rows']} rows", flush=True)
        except Exception as exc:
            errors.append({"kind": "sibling_import", "project": project, "source": source, "error": repr(exc)})
            print(f"ERROR import {project} {source}: {exc}", flush=True)

    write_manifest(records, errors)
    print(f"wrote {OUT / 'manifest.json'}", flush=True)
    print(f"records={len(records)} errors={len(errors)}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
