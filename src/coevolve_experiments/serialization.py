"""Canonical serialization shared by plans, locks, and audit sidecars."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path, PurePath
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    """Convert frozen experiment records into stable JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if isinstance(value, PurePath):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): to_primitive(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize experiment value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        to_primitive(value),
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    destination.write_text(payload + "\n", encoding="utf-8")


def strict_json_loads(value: str | bytes | bytearray) -> Any:
    """Decode unambiguous, standards-compliant JSON.

    The default Python decoder silently accepts duplicate object members and
    non-standard ``NaN``/``Infinity`` values.  Hash-bound experiment inputs
    reject both so every consumer assigns one meaning to the same bytes.
    """

    return json.loads(
        value,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_nonfinite_json_constant,
    )


def read_json(path: str | Path) -> Any:
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = child
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


__all__ = [
    "canonical_json",
    "read_json",
    "sha256_json",
    "strict_json_loads",
    "to_primitive",
    "write_json",
]
