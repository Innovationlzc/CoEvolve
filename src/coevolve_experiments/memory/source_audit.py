"""Resolve immutable Table 4 source identity without audit relabelling.

The run manifest is authoritative whenever it carries a value.  Explicit
audit metadata exists only to fill fields missing from older manifests; it
must never rewrite a model, scene, protocol, seed, or evidence-binding hash
already recorded by the source run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import semantic_sha256


@dataclass(frozen=True)
class ResolvedSourceIdentity:
    model: str
    scene: str
    protocol: str
    seed: str
    method_commit: str
    condition_fingerprint: str
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedEvidenceBindingPin:
    """Manifest-first hash pin for the independent source-evidence binding."""

    sha256: str
    source: str
    conflicts: tuple[str, ...] = ()


def resolve_source_identity(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> ResolvedSourceIdentity:
    """Resolve manifest-first identity and record every contradictory override."""

    manifest_model = _text(
        _nested(
            manifest,
            ("model", "resolved"),
            ("model", "requested"),
            ("effective_arguments", "planner_model"),
            ("source_model",),
        )
    )
    audit_model = _text(
        _nested(
            audit,
            ("source_model",),
            ("model",),
            ("backbone",),
            ("resolved_model",),
        )
    )

    manifest_scene = _text(
        _nested(
            manifest,
            ("scene", "id"),
            ("scene", "name"),
            ("scene", "variant"),
            ("scene_id",),
            ("scene_variant",),
            ("effective_arguments", "scene_id"),
        )
    )
    if not manifest_scene and str(manifest.get("benchmark", "")).lower() == "backbone":
        manifest_scene = "base"
    audit_scene = _text(
        _nested(audit, ("scene",), ("scene_id",), ("source_scene",))
    )

    manifest_protocol = _text(
        _nested(
            manifest,
            ("eval_protocol",),
            ("protocol", "id"),
            ("effective_arguments", "eval_protocol"),
        )
    )
    audit_protocol = _text(
        _nested(audit, ("protocol",), ("protocol_id",), ("eval_protocol",))
    )

    manifest_seed = _seed_text(
        _nested(
            manifest,
            ("seed",),
            ("run", "seed"),
            ("effective_arguments", "seed"),
        )
    )
    audit_seed = _seed_text(
        _nested(audit, ("seed",), ("source_seed",), ("run_seed",))
    )

    manifest_method_commit = _text(
        _nested(
            manifest,
            ("codebase", "git_head"),
            ("tag_commit",),
            ("method_commit",),
            ("experiment", "base_method_commit"),
        )
    ).lower()
    audit_method_commit = _text(
        _nested(audit, ("method_commit",), ("base_method_commit",), ("commit",))
    ).lower()

    manifest_condition_fingerprint = _text(
        _nested(
            manifest,
            ("condition_fingerprint",),
            ("experiment", "condition_fingerprint"),
        )
    ).lower()
    audit_condition_fingerprint = _text(
        _nested(
            audit,
            ("condition_fingerprint",),
            ("source_condition_fingerprint",),
        )
    ).lower()

    conflicts: list[str] = []
    _append_conflict(conflicts, "model", manifest_model, audit_model)
    _append_conflict(
        conflicts,
        "scene",
        manifest_scene,
        audit_scene,
        normalize=normalize_source_scene,
    )
    _append_conflict(
        conflicts, "protocol", manifest_protocol, audit_protocol
    )
    _append_conflict(conflicts, "seed", manifest_seed, audit_seed)
    _append_conflict(
        conflicts,
        "method commit",
        manifest_method_commit,
        audit_method_commit,
    )
    _append_conflict(
        conflicts,
        "condition fingerprint",
        manifest_condition_fingerprint,
        audit_condition_fingerprint,
    )
    return ResolvedSourceIdentity(
        model=manifest_model or audit_model,
        scene=manifest_scene or audit_scene,
        protocol=manifest_protocol or audit_protocol,
        seed=manifest_seed or audit_seed,
        method_commit=manifest_method_commit or audit_method_commit,
        condition_fingerprint=(
            manifest_condition_fingerprint or audit_condition_fingerprint
        ),
        conflicts=tuple(conflicts),
    )


def normalize_source_scene(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "base_scene": "base",
        "authored_base": "base",
        "manual_layout": "base",
        "base_manual_layout": "base",
    }
    return aliases.get(normalized, normalized)


def resolve_evidence_binding_pin(
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> ResolvedEvidenceBindingPin:
    """Resolve the binding hash without permitting audit relabelling.

    New manifests pin the evidence document at the single frozen field
    ``result_artifact_hashes.strategy_export_evidence_sha256``.  A frozen
    source audit may fill that value only when the manifest predates the
    field.  Supplying both values with different hashes is always a conflict.
    """

    manifest_hash = _text(
        _nested(
            manifest,
            ("result_artifact_hashes", "strategy_export_evidence_sha256"),
        )
    ).lower()
    audit_hash = _text(
        _nested(audit, ("strategy_export_evidence_sha256",))
    ).lower()
    conflicts: list[str] = []
    _append_conflict(
        conflicts,
        "strategy-export evidence binding SHA-256",
        manifest_hash,
        audit_hash,
    )
    return ResolvedEvidenceBindingPin(
        sha256=manifest_hash or audit_hash,
        source="manifest" if manifest_hash else "audit" if audit_hash else "",
        conflicts=tuple(conflicts),
    )


def stable_manifest_sha256(value: object) -> str:
    """Identity hash insensitive to whitespace/key order in a copied manifest."""

    return semantic_sha256(value)


def require_unique_source_manifests(
    manifests: Sequence[object],
    *,
    required: int,
) -> None:
    identities = [stable_manifest_sha256(value) for value in manifests]
    if len(identities) != required or len(set(identities)) != required:
        raise ValueError(
            f"source cohort must contain {required} unique manifest documents; "
            f"got {len(set(identities))} unique identities"
        )


def _append_conflict(
    conflicts: list[str],
    label: str,
    manifest_value: str,
    audit_value: str,
    *,
    normalize=lambda value: value,
) -> None:
    if (
        manifest_value
        and audit_value
        and normalize(manifest_value) != normalize(audit_value)
    ):
        conflicts.append(
            f"audit {label} conflicts with source manifest: "
            f"{audit_value!r} != {manifest_value!r}"
        )


def _nested(payload: Mapping[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            if current not in (None, ""):
                return current
    return None


def _text(value: object) -> str:
    if value is None or isinstance(value, (Mapping, list, tuple, bool)):
        return ""
    return str(value).strip()


def _seed_text(value: object) -> str:
    text = _text(value)
    if not text:
        return ""
    try:
        parsed = int(text)
    except ValueError:
        return text
    return str(parsed)


__all__ = [
    "ResolvedEvidenceBindingPin",
    "ResolvedSourceIdentity",
    "normalize_source_scene",
    "require_unique_source_manifests",
    "resolve_evidence_binding_pin",
    "resolve_source_identity",
    "stable_manifest_sha256",
]
