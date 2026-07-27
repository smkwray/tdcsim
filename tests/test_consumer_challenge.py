"""Producer-side ingest challenge for the RateWall export boundary."""

import json

import pytest

from tdcsim_cbo.consumer_challenge import (
    EXCLUDED_FIELDS,
    SELECTED_FIELD,
    build_ingest_challenge,
)


def _write_summary(path, rows):
    header = [
        "period",
        SELECTED_FIELD,
        "delta_tdc_ex_overlap_bil",
        "legacy_chi_support_diagnostic_bil",
    ]
    lines = [",".join(header)]
    lines.extend(",".join(str(cell) for cell in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_challenge_binds_bytes_rows_and_the_selected_field(tmp_path):
    summary = _write_summary(
        tmp_path / "summary.csv",
        [("2027", "3.802346", "5.344736", "0.141267"), ("2028", "3.511209", "4.998112", "0.130004")],
    )

    challenge = build_ingest_challenge(summary, pair_id="pair-1", runtime_release_sha="abc123")

    assert challenge["pair_id"] == "pair-1"
    assert challenge["runtime_release_sha"] == "abc123"
    assert challenge["row_count"] == 2
    assert challenge["source_file_bytes"] == summary.stat().st_size
    assert len(challenge["source_file_sha256"]) == 64
    assert challenge["selected_field"] == SELECTED_FIELD
    assert challenge["canonical_row_key"] == "period"
    assert len(challenge["value_digest_sha256"]) == 64


def test_challenge_names_the_retired_field_as_excluded(tmp_path):
    """Silence about the legacy diagnostic is not evidence it was not consumed.

    Naming it lets a consumer receipt assert it was not read, rather than leaving that to
    inference.
    """

    summary = _write_summary(tmp_path / "summary.csv", [("2027", "3.802346", "5.344736", "0.141267")])

    challenge = build_ingest_challenge(summary, pair_id="pair-1")

    assert "legacy_chi_support_diagnostic_bil" in challenge["excluded_fields"]
    assert "delta_tdc_ex_overlap_bil" in challenge["excluded_fields"]
    assert SELECTED_FIELD not in EXCLUDED_FIELDS


def test_digest_is_over_decimal_strings_not_parsed_floats(tmp_path):
    """Two values equal as floats but differing as text must produce different digests.

    Round-tripping through binary floating point would let genuine last-place drift pass
    unnoticed, which is exactly what this digest exists to catch.
    """

    a = _write_summary(tmp_path / "a.csv", [("2027", "3.8023460000000001", "5.3", "0.1")])
    b = _write_summary(tmp_path / "b.csv", [("2027", "3.8023460000000002", "5.3", "0.1")])

    assert float("3.8023460000000001") == float("3.8023460000000002")

    digest_a = build_ingest_challenge(a, pair_id="p")["value_digest_sha256"]
    digest_b = build_ingest_challenge(b, pair_id="p")["value_digest_sha256"]

    assert digest_a != digest_b


def test_digest_is_stable_under_row_reordering(tmp_path):
    """Row order in the file must not change the digest; the canonical key defines identity."""

    forward = _write_summary(
        tmp_path / "forward.csv",
        [("2027", "3.802346", "5.3", "0.1"), ("2028", "3.511209", "4.9", "0.1")],
    )
    reversed_rows = _write_summary(
        tmp_path / "reversed.csv",
        [("2028", "3.511209", "4.9", "0.1"), ("2027", "3.802346", "5.3", "0.1")],
    )

    assert (
        build_ingest_challenge(forward, pair_id="p")["value_digest_sha256"]
        == build_ingest_challenge(reversed_rows, pair_id="p")["value_digest_sha256"]
    )


def test_digest_changes_when_a_selected_value_changes(tmp_path):
    base = _write_summary(tmp_path / "base.csv", [("2027", "3.802346", "5.3", "0.1")])
    moved = _write_summary(tmp_path / "moved.csv", [("2027", "3.802347", "5.3", "0.1")])

    assert (
        build_ingest_challenge(base, pair_id="p")["value_digest_sha256"]
        != build_ingest_challenge(moved, pair_id="p")["value_digest_sha256"]
    )


def test_challenge_fails_closed_on_a_missing_selected_field(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("period,something_else\n2027,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        build_ingest_challenge(bad, pair_id="p")


def test_challenge_fails_closed_on_an_empty_or_absent_file(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("period,%s\n" % SELECTED_FIELD, encoding="utf-8")

    with pytest.raises(ValueError, match="no rows"):
        build_ingest_challenge(empty, pair_id="p")

    with pytest.raises(FileNotFoundError):
        build_ingest_challenge(tmp_path / "nope.csv", pair_id="p")


def test_challenge_is_json_serialisable(tmp_path):
    summary = _write_summary(tmp_path / "summary.csv", [("2027", "3.802346", "5.3", "0.1")])

    challenge = build_ingest_challenge(summary, pair_id="p")

    assert json.loads(json.dumps(challenge))["selected_field"] == SELECTED_FIELD
