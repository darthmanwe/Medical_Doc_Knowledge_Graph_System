"""Shared utilities for retrieval (e.g. Neo4j → JSON-safe property sanitization)."""

from __future__ import annotations

import datetime
from typing import Any


def sanitize_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Make Neo4j node/relationship properties JSON-serializable.

    - Drops embedding and binary values.
    - Converts Neo4j DateTime/Date/Time and Python datetime to ISO strings.
    """
    out: dict[str, Any] = {}
    for k, v in props.items():
        if k == "embedding" or isinstance(v, (bytes, bytearray)):
            continue
        out[k] = _to_json_safe(v)
    return out


def _to_json_safe(value: Any) -> Any:
    """Recursively convert a value to a JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    # Neo4j driver time types
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(x) for x in value]
    return value
