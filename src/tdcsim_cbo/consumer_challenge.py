"""Producer-side ingest challenge for the RateWall export boundary.

TDCSim cannot prove what a downstream consumer did with its bytes; only the consumer can.
What TDCSim *can* do is state precisely what it published and what it expects to be selected,
in a form the consumer's own receipt can be compared against field by field. That is this
artifact's job: it discharges the producer half of the export-boundary claim and turns the
remainder into an explicit external dependency rather than an assumption.

The value digest is computed over the **original decimal strings** as written to CSV, never
over parsed floats. Round-tripping through binary floating point would let two files that
differ in their last decimal place produce the same digest, which is exactly the drift this is
meant to catch.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from ._json import sha256_file

# The one field RateWall is permitted to take as the marginal TDC amount. Verified against the
# consumer's own source: it reads this column and never reads the retired beta-chi diagnostic.
SELECTED_FIELD = "delta_tdc_ex_overlap_non_interest_admissible_bil"

# Fields the consumer reads alongside the selected amount, recorded so a receipt can confirm it
# consumed the same supporting values rather than recomputing them.
SUPPORTING_FIELDS = (
    "delta_tdc_ex_overlap_interest_driven_excluded_bil",
    "tdc_materialized_deposit_stock_admissible_bil",
    "tdc_income_addendum_full_level_rate",
)

# Explicitly not for consumption. Named here so a receipt can assert it was not read, rather
# than leaving that silence to be inferred.
EXCLUDED_FIELDS = (
    "legacy_chi_support_diagnostic_bil",
    "delta_tdc_ex_overlap_bil",
)

CANONICAL_ROW_KEY = "period"
CHALLENGE_SCHEMA_VERSION = "tdcsim_ratewall_ingest_challenge_v1"


def build_ingest_challenge(
    summary_csv: str | Path,
    *,
    pair_id: str,
    pair_manifest_path: str | Path | None = None,
    runtime_release_sha: str = "",
) -> dict[str, Any]:
    """Describe exactly what was published and what the consumer is expected to select."""

    path = Path(summary_csv)
    if not path.is_file():
        raise FileNotFoundError(f"summary CSV is missing: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"summary CSV has no rows: {path}")

    missing = [field for field in (CANONICAL_ROW_KEY, SELECTED_FIELD) if field not in rows[0]]
    if missing:
        raise ValueError(f"summary CSV is missing required fields: {missing}")

    digest = _value_digest(rows)
    challenge: dict[str, Any] = {
        "schema_version": CHALLENGE_SCHEMA_VERSION,
        "pair_id": pair_id,
        "runtime_release_sha": runtime_release_sha,
        "source_file_name": path.name,
        "source_file_sha256": sha256_file(path),
        "source_file_bytes": path.stat().st_size,
        "canonical_row_key": CANONICAL_ROW_KEY,
        "row_count": len(rows),
        "selected_field": SELECTED_FIELD,
        "supporting_fields": list(SUPPORTING_FIELDS),
        "excluded_fields": list(EXCLUDED_FIELDS),
        "value_digest_sha256": digest,
        "value_digest_basis": "sha256_over_original_decimal_strings_not_parsed_floats",
        "claim": (
            "producer published these bytes and expects the consumer to select only "
            f"{SELECTED_FIELD}; a consumer receipt must reproduce value_digest_sha256"
        ),
    }
    if pair_manifest_path is not None:
        manifest = Path(pair_manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"pair manifest is missing: {manifest}")
        challenge["pair_manifest_sha256"] = sha256_file(manifest)
    return challenge


def _value_digest(rows: list[dict[str, str]]) -> str:
    """Digest canonical row keys against the selected value, as literal strings."""

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item.get(CANONICAL_ROW_KEY, ""))):
        key = str(row.get(CANONICAL_ROW_KEY, ""))
        value = str(row.get(SELECTED_FIELD, ""))
        digest.update(f"{key}\x1f{value}\x1e".encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "CANONICAL_ROW_KEY",
    "CHALLENGE_SCHEMA_VERSION",
    "EXCLUDED_FIELDS",
    "SELECTED_FIELD",
    "SUPPORTING_FIELDS",
    "build_ingest_challenge",
]
