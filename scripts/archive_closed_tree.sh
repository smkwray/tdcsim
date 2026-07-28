#!/usr/bin/env bash
set -euo pipefail

# archive_closed_tree.sh — byte-exact archive of a closed run/evidence tree.
#
# Project retention policy: a closed retained tree is stored once — in a verified
# campaign container when it meets that contract, otherwise in a byte-exact compressed
# archive with a recorded retention reason and verification status.
#
# This is the archive path. It never deletes the source: it writes the archive, extracts
# it into a scratch directory, and compares a path/size/SHA-256 inventory of the extract
# against the source. Only a matching inventory produces a receipt. Pruning the source is
# a separate, deliberate act by the caller, after reading the receipt.
#
# Usage:
#   scripts/archive_closed_tree.sh <tree> <archive-dir> "<retention reason>"
#
# Produces, in <archive-dir>:
#   <name>.tar.zst          the archive
#   <name>.receipt.json     inventory digest, archive hash, extraction verdict, counts
#
# Exit codes: 0 verified · 1 usage/precondition · 2 extraction mismatch (archive kept
# for inspection; never trust it).

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <tree> <archive-dir> \"<retention reason>\"" >&2
  exit 1
fi

TREE="$1"; ARCHIVE_DIR="$2"; REASON="$3"

[[ -d "$TREE" ]] || { echo "archive: not a directory: $TREE" >&2; exit 1; }
command -v zstd >/dev/null || { echo "archive: zstd not found" >&2; exit 1; }

TREE="$(cd "$TREE" && pwd)"
NAME="$(basename "$TREE")"
mkdir -p "$ARCHIVE_DIR"
ARCHIVE_DIR="$(cd "$ARCHIVE_DIR" && pwd)"
ARCHIVE="$ARCHIVE_DIR/$NAME.tar.zst"
RECEIPT="$ARCHIVE_DIR/$NAME.receipt.json"

[[ -e "$ARCHIVE" ]] && { echo "archive: refusing to overwrite $ARCHIVE" >&2; exit 1; }

# Inventory helper: sorted "sha256  size  relpath" over regular files, relative to $1.
inventory() {
  ( cd "$1" && find . -type f -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 -n1 sh -c 'printf "%s  %s  %s\n" "$(shasum -a 256 "$1" | cut -d" " -f1)" "$(wc -c < "$1" | tr -d " ")" "$1"' _ )
}

echo "archive: inventorying $NAME ..." >&2
SRC_INV="$(mktemp)"; EXT_INV="$(mktemp)"
STAGE="$(mktemp -d)"
cleanup() { rm -rf "$STAGE" "$SRC_INV" "$EXT_INV"; }
trap cleanup EXIT

inventory "$TREE" > "$SRC_INV"
SRC_FILES="$(wc -l < "$SRC_INV" | tr -d ' ')"
SRC_BYTES="$(awk '{s+=$2} END{print s+0}' "$SRC_INV")"
SRC_DIGEST="$(shasum -a 256 "$SRC_INV" | cut -d' ' -f1)"

echo "archive: writing $ARCHIVE ($SRC_FILES files, $SRC_BYTES bytes) ..." >&2
tar -C "$(dirname "$TREE")" -cf - "$NAME" | zstd -3 -q -o "$ARCHIVE"
ARCHIVE_SHA="$(shasum -a 256 "$ARCHIVE" | cut -d' ' -f1)"
ARCHIVE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"

echo "archive: verifying by extraction ..." >&2
zstd -dc "$ARCHIVE" | tar -xf - -C "$STAGE"
inventory "$STAGE/$NAME" > "$EXT_INV"
EXT_DIGEST="$(shasum -a 256 "$EXT_INV" | cut -d' ' -f1)"

if [[ "$SRC_DIGEST" != "$EXT_DIGEST" ]]; then
  echo "archive: EXTRACTION MISMATCH — archive does not reproduce the source tree" >&2
  diff "$SRC_INV" "$EXT_INV" | head -20 >&2
  exit 2
fi

RATIO="$(awk -v a="$ARCHIVE_BYTES" -v s="$SRC_BYTES" 'BEGIN{ if (s>0) printf "%.6f", a/s; else print "0" }')"
cat > "$RECEIPT" <<JSON
{
  "schema_version": "tdcsim_archive_receipt_v1",
  "tree_name": "$NAME",
  "retention_reason": "$REASON",
  "verification_status": "extraction_inventory_match",
  "source_file_count": $SRC_FILES,
  "source_logical_bytes": $SRC_BYTES,
  "source_inventory_sha256": "$SRC_DIGEST",
  "extracted_inventory_sha256": "$EXT_DIGEST",
  "archive_relative_path": "$NAME.tar.zst",
  "archive_sha256": "$ARCHIVE_SHA",
  "archive_bytes": $ARCHIVE_BYTES,
  "stored_fraction_of_source": $RATIO,
  "restore_command": "zstd -dc $NAME.tar.zst | tar -xf - -C <empty-dir>"
}
JSON

echo "archive: VERIFIED $NAME" >&2
echo "  source   : $SRC_FILES files, $SRC_BYTES bytes" >&2
echo "  archive  : $ARCHIVE_BYTES bytes (${RATIO} of source)" >&2
echo "  receipt  : $RECEIPT" >&2
echo "  source tree NOT removed; prune deliberately after reading the receipt." >&2
