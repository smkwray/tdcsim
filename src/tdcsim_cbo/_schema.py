"""Small strict JSON-schema subset used by the CBO interface contracts.

The project intentionally avoids adding a new runtime dependency for Phase 1.
This validator implements the Draft 2020-12 features used by the shipped
scenario and attestation schemas: objects, arrays, primitive types, required
fields, additionalProperties=false, const, enum, pattern, numeric bounds,
min/max length, min items, and local $defs references.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from ._json import read_json


class SchemaValidationError(ValueError):
    """Raised when a CBO contract does not satisfy its schema."""


def load_schema(path: str | Path) -> dict[str, Any]:
    schema = read_json(path)
    if not isinstance(schema, dict):
        raise SchemaValidationError("schema root must be an object")
    return schema


def validate_schema(value: Any, schema: Mapping[str, Any], *, label: str = "document") -> None:
    _validate(value, schema, root=schema, path=label)


def _validate(value: Any, schema: Mapping[str, Any], *, root: Mapping[str, Any], path: str) -> None:
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if not ref.startswith("#/$defs/"):
            raise SchemaValidationError(f"{path}: unsupported schema reference {ref!r}")
        name = ref.removeprefix("#/$defs/")
        defs = root.get("$defs", {})
        if not isinstance(defs, Mapping) or name not in defs:
            raise SchemaValidationError(f"{path}: unresolved schema reference {ref!r}")
        _validate(value, defs[name], root=root, path=path)
        return

    if "const" in schema and value != schema["const"]:
        raise SchemaValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise SchemaValidationError(f"{path}: expected one of {schema['enum']!r}")

    schema_type = schema.get("type")
    if schema_type is not None:
        _validate_type(value, schema_type, path)

    if isinstance(value, Mapping):
        _validate_object(value, schema, root=root, path=path)
    elif isinstance(value, list):
        _validate_array(value, schema, root=root, path=path)
    elif isinstance(value, str):
        _validate_string(value, schema, path=path)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _validate_number(float(value), schema, path=path)

    if "format" in schema and isinstance(value, str):
        _validate_format(value, str(schema["format"]), path=path)


def _validate_type(value: Any, schema_type: Any, path: str) -> None:
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    if any(_is_type(value, str(item)) for item in types):
        return
    raise SchemaValidationError(f"{path}: expected type {types!r}")


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _validate_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    path: str,
) -> None:
    required = schema.get("required", [])
    for key in required:
        if key not in value:
            raise SchemaValidationError(f"{path}: missing required field {key!r}")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    if schema.get("additionalProperties") is False:
        unknown = sorted(str(key) for key in value if key not in properties)
        if unknown:
            raise SchemaValidationError(f"{path}: unknown fields {unknown}")

    for key, item in value.items():
        if key in properties:
            _validate(item, properties[key], root=root, path=f"{path}.{key}")


def _validate_array(value: list[Any], schema: Mapping[str, Any], *, root: Mapping[str, Any], path: str) -> None:
    min_items = schema.get("minItems")
    if min_items is not None and len(value) < int(min_items):
        raise SchemaValidationError(f"{path}: expected at least {min_items} items")
    max_items = schema.get("maxItems")
    if max_items is not None and len(value) > int(max_items):
        raise SchemaValidationError(f"{path}: expected at most {max_items} items")
    if schema.get("uniqueItems") is True:
        seen = set()
        for item in value:
            marker = repr(item)
            if marker in seen:
                raise SchemaValidationError(f"{path}: expected unique items")
            seen.add(marker)
    items_schema = schema.get("items")
    if isinstance(items_schema, Mapping):
        for index, item in enumerate(value):
            _validate(item, items_schema, root=root, path=f"{path}[{index}]")


def _validate_string(value: str, schema: Mapping[str, Any], *, path: str) -> None:
    min_length = schema.get("minLength")
    if min_length is not None and len(value) < int(min_length):
        raise SchemaValidationError(f"{path}: string shorter than {min_length}")
    max_length = schema.get("maxLength")
    if max_length is not None and len(value) > int(max_length):
        raise SchemaValidationError(f"{path}: string longer than {max_length}")
    pattern = schema.get("pattern")
    if pattern is not None and re.search(str(pattern), value) is None:
        raise SchemaValidationError(f"{path}: string does not match required pattern")


def _validate_number(value: float, schema: Mapping[str, Any], *, path: str) -> None:
    minimum = schema.get("minimum")
    if minimum is not None and value < float(minimum):
        raise SchemaValidationError(f"{path}: value below minimum {minimum}")
    maximum = schema.get("maximum")
    if maximum is not None and value > float(maximum):
        raise SchemaValidationError(f"{path}: value above maximum {maximum}")
    exclusive_minimum = schema.get("exclusiveMinimum")
    if exclusive_minimum is not None and value <= float(exclusive_minimum):
        raise SchemaValidationError(f"{path}: value not greater than {exclusive_minimum}")


def _validate_format(value: str, fmt: str, *, path: str) -> None:
    if fmt == "date":
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise SchemaValidationError(f"{path}: invalid date") from exc
    elif fmt == "date-time":
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaValidationError(f"{path}: invalid date-time") from exc
