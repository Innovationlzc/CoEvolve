"""Fail-closed collector audit for Table-2/3 from-scratch memory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..memory.from_scratch import (
    FROM_SCRATCH_BOOTSTRAP_FILENAME,
    FROM_SCRATCH_BOOTSTRAP_KIND,
    FROM_SCRATCH_INITIAL_FILES,
    FROM_SCRATCH_RUNTIME_AUDIT_FILENAME,
    FROM_SCRATCH_RUNTIME_AUDIT_KIND,
    FROM_SCRATCH_SCHEMA_VERSION,
    empty_accepted_episode_manifest,
    empty_failure_cache,
)
from ..memory.online_memory import (
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
    validate_action_exemplar_memory,
    validate_online_strategy_overlay,
)
from ..memory.schema import semantic_sha256


@dataclass(frozen=True)
class FromScratchMemoryAuditValidation:
    invalid_reasons: tuple[str, ...]
    artifact_hashes: Mapping[str, str]


def validate_from_scratch_memory_audit(
    run_dir: str | Path,
    *,
    expected_run_binding: Mapping[str, Any],
    expected_memory_condition: str,
    expected_memory_policy: str,
) -> FromScratchMemoryAuditValidation:
    """Validate initial emptiness, operation gates, and final store lineage."""

    root = Path(run_dir).absolute()
    memory = root / "memory"
    bootstrap_path = memory / FROM_SCRATCH_BOOTSTRAP_FILENAME
    runtime_path = memory / FROM_SCRATCH_RUNTIME_AUDIT_FILENAME
    overlay_path = memory / "online_strategy_overlay.json"
    action_path = memory / "action_exemplars.json"
    accepted_path = memory / "accepted_episode_manifest.json"
    failure_path = memory / "failure_cache.json"
    reasons: list[str] = []

    registered = (
        bootstrap_path,
        runtime_path,
        overlay_path,
        action_path,
        accepted_path,
        failure_path,
    )
    if not _paths_are_isolated(root, memory, registered):
        _add(reasons, "FROM_SCRATCH_MEMORY_PATHS_NOT_ISOLATED")

    bootstrap = _read_object(bootstrap_path)
    runtime = _read_object(runtime_path)
    expected_gates = _operation_gates(expected_memory_policy)
    expected_initial = {
        "physical_state": "s0",
        "checkpoint": "c0",
        "strategy_records": 0,
        "action_exemplars": 0,
        "accepted_manifest_steps": 0,
        "failure_cache_records": 0,
    }

    if set(bootstrap) != {
        "kind",
        "schema_version",
        "run_binding",
        "memory_condition",
        "memory_policy",
        "cross_run_retrieval",
        "operation_gates",
        "initial_state",
        "initial_files",
    }:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_FIELDS_MISMATCH")
    if bootstrap.get("kind") != FROM_SCRATCH_BOOTSTRAP_KIND:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_KIND_MISMATCH")
    if bootstrap.get("schema_version") != FROM_SCRATCH_SCHEMA_VERSION:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_SCHEMA_MISMATCH")
    if bootstrap.get("run_binding") != dict(expected_run_binding):
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_RUN_BINDING_MISMATCH")
    if bootstrap.get("memory_condition") != expected_memory_condition:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_CONDITION_MISMATCH")
    if bootstrap.get("memory_policy") != expected_memory_policy:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_POLICY_MISMATCH")
    if bootstrap.get("cross_run_retrieval") is not False:
        _add(reasons, "FROM_SCRATCH_CROSS_RUN_RETRIEVAL_NOT_DISABLED")
    if bootstrap.get("operation_gates") != expected_gates:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_GATES_MISMATCH")
    if bootstrap.get("initial_state") != expected_initial:
        _add(reasons, "FROM_SCRATCH_INITIAL_STATE_UNPROVEN")
    try:
        if bootstrap_path.stat().st_mode & 0o222:
            _add(reasons, "FROM_SCRATCH_BOOTSTRAP_WRITABLE")
    except OSError:
        _add(reasons, "FROM_SCRATCH_BOOTSTRAP_MISSING")
    _validate_initial_file_registry(bootstrap.get("initial_files"), reasons)

    expected_runtime_fields = {
        "kind",
        "schema_version",
        "lifecycle_state",
        "run_binding",
        "memory_condition",
        "memory_policy",
        "cross_run_retrieval",
        "operation_gates",
        "bootstrap_path",
        "bootstrap_sha256",
        "initial_state_verified",
        "online_overlay",
        "action_exemplars",
        "operations",
        "violations",
        "finalized",
    }
    if set(runtime) != expected_runtime_fields:
        _add(reasons, "FROM_SCRATCH_RUNTIME_FIELDS_MISMATCH")
    if runtime.get("kind") != FROM_SCRATCH_RUNTIME_AUDIT_KIND:
        _add(reasons, "FROM_SCRATCH_RUNTIME_KIND_MISMATCH")
    if runtime.get("schema_version") != FROM_SCRATCH_SCHEMA_VERSION:
        _add(reasons, "FROM_SCRATCH_RUNTIME_SCHEMA_MISMATCH")
    if runtime.get("lifecycle_state") != "finalized" or runtime.get("finalized") is not True:
        _add(reasons, "FROM_SCRATCH_MEMORY_NOT_FINALIZED")
    if runtime.get("run_binding") != dict(expected_run_binding):
        _add(reasons, "FROM_SCRATCH_RUNTIME_RUN_BINDING_MISMATCH")
    if (
        runtime.get("memory_condition") != expected_memory_condition
        or runtime.get("memory_policy") != expected_memory_policy
    ):
        _add(reasons, "FROM_SCRATCH_RUNTIME_POLICY_MISMATCH")
    if (
        runtime.get("cross_run_retrieval") is not False
        or runtime.get("operation_gates") != expected_gates
    ):
        _add(reasons, "FROM_SCRATCH_RUNTIME_GATES_MISMATCH")
    if runtime.get("bootstrap_path") != FROM_SCRATCH_BOOTSTRAP_FILENAME:
        _add(reasons, "FROM_SCRATCH_RUNTIME_BOOTSTRAP_PATH_MISMATCH")
    if runtime.get("bootstrap_sha256") != _file_sha256(bootstrap_path):
        _add(reasons, "FROM_SCRATCH_RUNTIME_BOOTSTRAP_HASH_MISMATCH")
    if runtime.get("initial_state_verified") is not True:
        _add(reasons, "FROM_SCRATCH_RUNTIME_INITIAL_STATE_UNPROVEN")
    violations = runtime.get("violations")
    if not isinstance(violations, list) or violations:
        _add(reasons, "FROM_SCRATCH_MEMORY_LIFECYCLE_VIOLATIONS")

    overlay = _validated_store(
        overlay_path,
        validate_online_strategy_overlay,
        reasons,
        "FROM_SCRATCH_OVERLAY_INVALID",
    )
    actions = _validated_store(
        action_path,
        validate_action_exemplar_memory,
        reasons,
        "FROM_SCRATCH_ACTION_MEMORY_INVALID",
    )
    _validate_runtime_store(
        runtime.get("online_overlay"),
        payload=overlay,
        collection="entries",
        expected_path=overlay_path.name,
        expected_initial_sha=semantic_sha256(empty_online_strategy_overlay()),
        reason_prefix="FROM_SCRATCH_OVERLAY",
        reasons=reasons,
    )
    _validate_runtime_store(
        runtime.get("action_exemplars"),
        payload=actions,
        collection="exemplars",
        expected_path=action_path.name,
        expected_initial_sha=semantic_sha256(empty_action_exemplar_memory()),
        reason_prefix="FROM_SCRATCH_ACTION_MEMORY",
        reasons=reasons,
    )
    operations = _validate_operations(runtime.get("operations"), reasons)
    if expected_memory_policy == "disabled":
        if any(operations.values()):
            _add(reasons, "FROM_SCRATCH_DISABLED_MEMORY_OPERATION_DETECTED")
        if overlay.get("entries") or actions.get("exemplars"):
            _add(reasons, "FROM_SCRATCH_DISABLED_MEMORY_NOT_EMPTY")
    elif expected_memory_policy == "online_from_empty":
        if operations.get("strategy_promotions") != len(overlay.get("entries", [])):
            _add(reasons, "FROM_SCRATCH_STRATEGY_PROMOTION_COUNT_MISMATCH")
        if operations.get("exemplar_promotions") != len(actions.get("exemplars", [])):
            _add(reasons, "FROM_SCRATCH_EXEMPLAR_PROMOTION_COUNT_MISMATCH")
    else:
        _add(reasons, "FROM_SCRATCH_MEMORY_POLICY_UNSUPPORTED")

    return FromScratchMemoryAuditValidation(
        invalid_reasons=tuple(reasons),
        artifact_hashes={
            "from_scratch_memory_bootstrap": _file_sha256(bootstrap_path),
            "from_scratch_memory_runtime_audit": _file_sha256(runtime_path),
        },
    )


def _validate_initial_file_registry(value: object, reasons: list[str]) -> None:
    if not isinstance(value, list):
        _add(reasons, "FROM_SCRATCH_INITIAL_FILE_REGISTRY_INVALID")
        return
    expected_documents = {
        "online_strategy_overlay.json": empty_online_strategy_overlay(),
        "action_exemplars.json": empty_action_exemplar_memory(),
        "accepted_episode_manifest.json": empty_accepted_episode_manifest(),
        "failure_cache.json": empty_failure_cache(),
    }
    names = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "filename",
            "sha256",
            "mutable_after_dispatch",
        }:
            _add(reasons, "FROM_SCRATCH_INITIAL_FILE_REGISTRY_INVALID")
            return
        name = item.get("filename")
        names.append(name)
        if (
            name not in expected_documents
            or item.get("mutable_after_dispatch") is not True
            or item.get("sha256") != _rendered_json_sha256(expected_documents[str(name)])
        ):
            _add(reasons, "FROM_SCRATCH_INITIAL_FILE_HASH_MISMATCH")
    if names != list(FROM_SCRATCH_INITIAL_FILES):
        _add(reasons, "FROM_SCRATCH_INITIAL_FILE_REGISTRY_INVALID")


def _validate_runtime_store(
    value: object,
    *,
    payload: Mapping[str, Any],
    collection: str,
    expected_path: str,
    expected_initial_sha: str,
    reason_prefix: str,
    reasons: list[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "initial_sha256",
        "current_sha256",
        "final_sha256",
        "initial_count",
        "current_count",
        "final_count",
    }:
        _add(reasons, f"{reason_prefix}_AUDIT_INVALID")
        return
    if value.get("path") != expected_path:
        _add(reasons, f"{reason_prefix}_PATH_MISMATCH")
    if value.get("initial_sha256") != expected_initial_sha or value.get("initial_count") != 0:
        _add(reasons, f"{reason_prefix}_INITIAL_STATE_MISMATCH")
    digest = semantic_sha256(payload) if payload else ""
    for field in ("current_sha256", "final_sha256"):
        if value.get(field) != digest:
            _add(reasons, f"{reason_prefix}_{field.upper()}_MISMATCH")
    records = payload.get(collection)
    count = len(records) if isinstance(records, list) else -1
    for field in ("current_count", "final_count"):
        if value.get(field) != count:
            _add(reasons, f"{reason_prefix}_{field.upper()}_MISMATCH")


def _validate_operations(value: object, reasons: list[str]) -> dict[str, int]:
    expected = {
        "strategy_retrievals",
        "exemplar_retrievals",
        "strategy_promotions",
        "exemplar_promotions",
        "promotion_noops",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        _add(reasons, "FROM_SCRATCH_MEMORY_OPERATIONS_INVALID")
        return {name: -1 for name in expected}
    result: dict[str, int] = {}
    for name in expected:
        count = value.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _add(reasons, "FROM_SCRATCH_MEMORY_OPERATIONS_INVALID")
            return {item: -1 for item in expected}
        result[name] = count
    return result


def _validated_store(path: Path, validator: Any, reasons: list[str], reason: str) -> dict[str, Any]:
    try:
        return validator(_read_object(path))
    except (TypeError, ValueError):
        _add(reasons, reason)
        return {}


def _operation_gates(memory_policy: str) -> dict[str, bool]:
    enabled = memory_policy == "online_from_empty"
    return {
        "strategy_retrieval": enabled,
        "strategy_promotion": enabled,
        "action_exemplar_retrieval": enabled,
        "action_exemplar_promotion": enabled,
    }


def _paths_are_isolated(root: Path, memory: Path, paths: tuple[Path, ...]) -> bool:
    try:
        if memory.parent != root or _path_or_ancestor_is_symlink(memory):
            return False
        resolved = memory.resolve(strict=True)
        return all(
            not path.is_symlink()
            and path.resolve(strict=True).parent == resolved
            for path in paths
        )
    except (OSError, RuntimeError):
        return False


def _path_or_ancestor_is_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _rendered_json_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _add(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


__all__ = [
    "FromScratchMemoryAuditValidation",
    "validate_from_scratch_memory_audit",
]
