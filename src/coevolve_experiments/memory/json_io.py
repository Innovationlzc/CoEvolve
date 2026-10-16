"""Small fail-closed JSON and atomic-replacement helpers for memory files."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


class MemoryFileError(ValueError):
    """A run-local memory file is missing, malformed, or unsafe to replace."""


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MemoryFileError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise MemoryFileError(f"{label} root must be an object")
    return dict(value)


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one object while rejecting ambiguous duplicate JSON members."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    """Reject Python's non-standard NaN/Infinity JSON extensions."""

    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def atomic_write_json(path: Path, value: object) -> None:
    """fsync a same-directory temporary file, then atomically replace ``path``."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise MemoryFileError(f"cannot serialize {path.name}: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise MemoryFileError(f"cannot atomically replace {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def resolve_local_artifact(root: Path, value: object, filename: str) -> Path:
    """Resolve an audit path relative to ``root`` and reject path substitution."""

    if not isinstance(value, str) or not value.strip():
        raise MemoryFileError(f"memory audit path for {filename} is missing")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    expected = (root / filename).resolve()
    if candidate != expected or candidate.parent != root.resolve():
        raise MemoryFileError(f"memory artifact path does not resolve to {filename}")
    return candidate


__all__ = [
    "MemoryFileError",
    "atomic_write_json",
    "read_json_object",
    "resolve_local_artifact",
]
