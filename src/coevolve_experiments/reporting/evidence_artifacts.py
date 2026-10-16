"""Small, shared primitives for hash-bound reporting evidence.

Reporting code must never follow a receipt-provided absolute path or trust a
hash written next to an artifact.  This module resolves one strict relative
reference below a run directory and verifies its exact bytes before JSON is
interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_REF_KEYS = frozenset({"relative_path", "sha256", "size_bytes"})


class BoundArtifactError(ValueError):
    """Raised when a reporting artifact is missing, unsafe, or not hash-bound."""


@dataclass(frozen=True)
class BoundJsonArtifact:
    """One verified JSON artifact below a run root."""

    relative_path: str
    path: Path
    sha256: str
    size_bytes: int
    payload: Mapping[str, Any]

    def public_ref(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def load_bound_json_artifact(
    run_root: str | Path,
    value: object,
    *,
    label: str,
) -> BoundJsonArtifact:
    """Load a strict ``relative_path/sha256/size_bytes`` JSON reference."""

    if not isinstance(value, Mapping):
        raise BoundArtifactError(f"{label} must be an artifact reference object")
    if set(value) != _ARTIFACT_REF_KEYS:
        raise BoundArtifactError(
            f"{label} must contain exactly {sorted(_ARTIFACT_REF_KEYS)}"
        )
    relative_path = value.get("relative_path")
    expected_sha256 = value.get("sha256")
    expected_size = value.get("size_bytes")
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise BoundArtifactError(f"{label}.relative_path must be a non-empty string")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
        expected_sha256
    ):
        raise BoundArtifactError(f"{label}.sha256 must be lowercase SHA-256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
    ):
        raise BoundArtifactError(f"{label}.size_bytes must be a non-negative integer")

    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or relative_path != pure.as_posix()
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise BoundArtifactError(f"{label}.relative_path must be normalized and relative")

    root = Path(run_root)
    if root.is_symlink():
        raise BoundArtifactError(f"{label}: run root must not be a symlink")
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise BoundArtifactError(f"{label}: symlinked artifact path is forbidden")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise BoundArtifactError(f"{label}: artifact is missing or escapes run root") from exc
    if not resolved.is_file():
        raise BoundArtifactError(f"{label}: artifact is not a regular file")
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise BoundArtifactError(f"{label}: cannot read artifact") from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected_size:
        raise BoundArtifactError(
            f"{label}: size mismatch ({len(raw)} != {expected_size})"
        )
    if actual_sha256 != expected_sha256:
        raise BoundArtifactError(f"{label}: SHA-256 mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoundArtifactError(f"{label}: artifact is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise BoundArtifactError(f"{label}: JSON payload must be an object")
    return BoundJsonArtifact(
        relative_path=pure.as_posix(),
        path=resolved,
        sha256=actual_sha256,
        size_bytes=len(raw),
        payload=dict(payload),
    )


def sha256_file(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ""


__all__ = [
    "BoundArtifactError",
    "BoundJsonArtifact",
    "load_bound_json_artifact",
    "sha256_file",
]
