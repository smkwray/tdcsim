"""Output writers for compiled CBO scenario runs."""

from __future__ import annotations

import gzip
import io
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ._json import sha256_file, write_json


SUMMARY_COLUMNS = [
    "Date",
    "TotalDebt_Agg",
    "CBOControlledDebtTarget",
    "CBOControlledDebtPreIssuance",
    "CBOControlledDebtPostIssuance",
    "CBOControlledDebtTargetError",
    "CBORequiredFaceIssuance",
    "NewDebtIssued",
    "AuctionProceeds",
    "PrimaryDeficit",
    "TGA",
    "CBOOperatingCashTarget",
    "CBOCashReconciliationResidual",
    "CBOCashResidualStatus",
    "CBOFedHoldingsTarget",
    "CBOFedHoldingsTargetError",
    "CBOFedAuctionShare",
    "CBOFedSecondaryPurchaseFace",
    "CBOFedSecondaryPurchaseCash",
    "CBOFedSecondaryPurchaseReserveEffect",
    "CBOFedSecondaryPurchaseDepositEffect",
    "CBOFedSyntheticSecondaryPurchases",
    "CBOFedSyntheticSecondarySales",
    "CBORemittanceCashEffect",
    "CBORemittanceStatus",
    "CB_Remittance",
    "CB_DeferredAsset",
    "CBONetInterestDiagnostic",
    "CBOTotalDeficitDiagnostic",
    "CBONetInterestBridgeRows",
    "NetInterestDiagnosticStatus",
    "DebtHeld_Banks",
    "DebtHeld_CentralBank",
    "DebtHeld_Foreign",
    "DebtHeld_DomesticNonBanks",
    "DebtHeldByType_Fixed",
    "DebtHeldByType_TIPS",
    "DebtHeldByType_FRN",
]


def write_scenario_outputs(
    results: pd.DataFrame,
    final_portfolio: pd.DataFrame,
    output_dir: str | Path,
    *,
    profile: str = "compact",
    compression: str = "gzip",
    catalog_sqlite: bool = False,
) -> dict[str, Any]:
    """Write run outputs and return a hash-listed output manifest."""

    if profile not in {"summary", "compact", "audit"}:
        raise ValueError(f"unsupported output profile: {profile}")
    if compression not in {"gzip", "none"}:
        raise ValueError(f"unsupported compression: {compression}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"

    results_out = _ensure_date_column(results)
    result_cols = [col for col in SUMMARY_COLUMNS if col in results_out.columns]
    if profile == "audit":
        result_cols = list(results_out.columns)
    result_path = out / f"results_{profile}{suffix}"
    _write_frame(results_out[result_cols] if result_cols else results_out, result_path, compression=compression)

    outputs: dict[str, Any] = {
        "profile": profile,
        "compression": compression,
        "results": _artifact_record(out, result_path),
    }
    if profile in {"compact", "audit"}:
        portfolio = final_portfolio
        if profile == "compact" and "Status" in portfolio.columns:
            portfolio = portfolio[portfolio["Status"].astype(str).eq("Active")]
        portfolio_path = out / f"final_portfolio_{profile}{suffix}"
        _write_frame(portfolio, portfolio_path, compression=compression)
        outputs["final_portfolio"] = _artifact_record(out, portfolio_path)

    summary = _summary(results, final_portfolio)
    summary_path = out / "summary.json"
    write_json(summary_path, summary)
    outputs["summary"] = _artifact_record(out, summary_path)
    outputs["summary_values"] = summary

    if catalog_sqlite:
        catalog_path = out / "catalog.sqlite"
        _write_catalog(catalog_path, outputs)
        outputs["catalog_sqlite"] = _artifact_record(out, catalog_path)
    return outputs


def hash_output_tree(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    records = []
    for file_path in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append(_artifact_record(root, file_path))
    return records


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str) -> None:
    if compression == "gzip":
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                    frame.to_csv(handle, index=False)
    else:
        frame.to_csv(path, index=False)


def _ensure_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "Date" in frame.columns:
        return frame
    if frame.index.name == "Date":
        return frame.reset_index()
    return frame


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _summary(results: pd.DataFrame, final_portfolio: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(results)),
        "final_portfolio_rows": int(len(final_portfolio)),
    }
    for col in (
        "CBORequiredFaceIssuance",
        "CBOControlledDebtTargetError",
        "CBOFedHoldingsTargetError",
        "CBOFedAuctionShare",
        "CBOCashReconciliationResidual",
    ):
        if col in results.columns and len(results[col]) > 0:
            numeric = pd.to_numeric(results[col], errors="coerce").fillna(0.0)
            summary[f"{col}_sum"] = float(numeric.sum())
            summary[f"{col}_max_abs"] = float(numeric.abs().max())
    for col in ("CBORemittanceStatus", "NetInterestDiagnosticStatus", "CBOFedStockMode"):
        if col in results.columns:
            summary[f"{col}_values"] = sorted(str(value) for value in results[col].dropna().unique())
    return summary


def _write_catalog(path: Path, outputs: Mapping[str, Any]) -> None:
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE artifacts (key TEXT PRIMARY KEY, path TEXT, bytes INTEGER, sha256 TEXT)")
        for key, value in outputs.items():
            if isinstance(value, Mapping) and {"path", "bytes", "sha256"} <= set(value):
                conn.execute(
                    "INSERT INTO artifacts (key, path, bytes, sha256) VALUES (?, ?, ?, ?)",
                    (key, value["path"], int(value["bytes"]), value["sha256"]),
                )
        conn.execute("CREATE TABLE summary (payload TEXT NOT NULL)")
        conn.execute("INSERT INTO summary (payload) VALUES (?)", (json.dumps(outputs.get("summary_values", {}), sort_keys=True),))


__all__ = ["SUMMARY_COLUMNS", "hash_output_tree", "write_scenario_outputs"]
