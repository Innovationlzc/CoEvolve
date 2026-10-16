"""Fail-closed validation of finalized Table-4 target-memory evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from ..contracts import StrategyBankPin
from ..memory.bootstrap import (
    TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME,
    MemoryBootstrapError,
    validate_transfer_bootstrap_snapshot,
)
from ..memory.online_memory import (
    empty_action_exemplar_memory,
    empty_online_strategy_overlay,
    validate_action_exemplar_memory,
    validate_online_strategy_overlay,
)
from ..memory.provenance import (
    StrategyBankProvenanceError,
    validate_strategy_bank_provenance,
)
from ..memory.schema import semantic_sha256
from ..memory.validation import validate_strategy_bank


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_OVERLAY_SHA256 = semantic_sha256(empty_online_strategy_overlay())
_EMPTY_ACTION_EXEMPLARS_SHA256 = semantic_sha256(
    empty_action_exemplar_memory()
)


@dataclass(frozen=True)
class Table4MemoryAuditValidation:
    invalid_reasons: tuple[str, ...]
    loaded_seed_sha256: str = ""
    artifact_hashes: Mapping[str, str] | None = None


def validate_table4_memory_audit(
    run_dir: str | Path,
    *,
    expected_strategy_bank_sha256: str,
    expected_condition: str,
    expected_target_model: str,
    expected_source_model: str,
    expected_strategy_bank_pin: StrategyBankPin,
    expected_formal_binding: Mapping[str, str],
) -> Table4MemoryAuditValidation:
    """Validate bootstrap, finalized lifecycle, and current memory bytes.

    The expected seed digest must come from the external suite plan.  Audit
    documents can describe what happened, but they cannot choose that digest.
    """

    root = Path(run_dir)
    memory_root = root / "memory"
    bootstrap_path = memory_root / "memory_bootstrap_audit.json"
    snapshot_path = memory_root / TRANSFER_BOOTSTRAP_SNAPSHOT_FILENAME
    runtime_path = memory_root / "memory_runtime_audit.json"
    seed_path = memory_root / "seed_strategy_bank.json"
    overlay_path = memory_root / "online_strategy_overlay.json"
    action_path = memory_root / "action_exemplars.json"
    reasons: list[str] = []

    registered_paths = (
        bootstrap_path,
        snapshot_path,
        runtime_path,
        seed_path,
        overlay_path,
        action_path,
    )
    if not _memory_paths_are_isolated(root, memory_root, registered_paths):
        _add(reasons, "TABLE4_MEMORY_PATHS_NOT_ISOLATED")

    bootstrap = _read_object(
        bootstrap_path,
        reasons,
        "TABLE4_BOOTSTRAP_AUDIT_MISSING_OR_INVALID",
    )
    runtime = _read_object(
        runtime_path,
        reasons,
        "TABLE4_RUNTIME_AUDIT_MISSING_OR_INVALID",
    )
    bootstrap_sha256 = _semantic_file_sha256(bootstrap_path)
    bootstrap_file_sha256 = _file_sha256(bootstrap_path)
    snapshot_file_sha256 = _file_sha256(snapshot_path)
    runtime_sha256 = _file_sha256(runtime_path)
    expected_condition_value = _normalize_condition(expected_condition)

    try:
        validate_transfer_bootstrap_snapshot(
            memory_root,
            require_initial_state=False,
            expected_formal_binding=expected_formal_binding,
            expected_condition=expected_condition,
            expected_target_model=expected_target_model,
            expected_seed_bank_sha256=expected_strategy_bank_sha256,
        )
    except (MemoryBootstrapError, OSError, TypeError, ValueError):
        _add(reasons, "TABLE4_BOOTSTRAP_SNAPSHOT_INVALID")

    if bootstrap:
        _expect_exact_fields(
            reasons,
            bootstrap,
            {
                "kind",
                "schema_version",
                "condition",
                "source_model",
                "source_model_family",
                "target_model",
                "target_model_family",
                "seed",
                "online_overlay",
                "fresh_target_state",
                "formal_binding",
            },
            "TABLE4_BOOTSTRAP_AUDIT_FIELDS_MISMATCH",
        )
        _expect_equal(
            reasons,
            bootstrap.get("kind"),
            "table4_memory_bootstrap_audit",
            "TABLE4_BOOTSTRAP_AUDIT_KIND_MISMATCH",
        )
        _expect_schema_v1(
            reasons,
            bootstrap.get("schema_version"),
            "TABLE4_BOOTSTRAP_AUDIT_SCHEMA_MISMATCH",
        )
        _expect_equal(
            reasons,
            _normalize_condition(bootstrap.get("condition")),
            expected_condition_value,
            "TABLE4_BOOTSTRAP_CONDITION_MISMATCH",
        )
        _expect_equal(
            reasons,
            bootstrap.get("target_model"),
            expected_target_model,
            "TABLE4_BOOTSTRAP_TARGET_MODEL_MISMATCH",
        )
        if expected_source_model:
            _expect_equal(
                reasons,
                bootstrap.get("source_model"),
                expected_source_model,
                "TABLE4_BOOTSTRAP_SOURCE_MODEL_MISMATCH",
            )
        _validate_bootstrap_seed(
            bootstrap.get("seed"),
            root,
            memory_root,
            seed_path,
            expected_strategy_bank_sha256,
            expected_strategy_bank_pin,
            reasons,
        )
        _validate_bootstrap_overlay(
            bootstrap.get("online_overlay"),
            root,
            memory_root,
            overlay_path,
            reasons,
        )
        fresh_state = bootstrap.get("fresh_target_state")
        if (
            not isinstance(fresh_state, dict)
            or set(fresh_state)
            != {
                "physical_state",
                "checkpoint",
                "action_exemplars_empty",
                "accepted_manifest_steps_empty",
                "failure_cache_empty",
            }
            or fresh_state.get("physical_state") != "s0"
            or fresh_state.get("checkpoint") != "c0"
            or any(
                fresh_state.get(field) is not True
                for field in (
                    "action_exemplars_empty",
                    "accepted_manifest_steps_empty",
                    "failure_cache_empty",
                )
            )
        ):
            _add(reasons, "TABLE4_BOOTSTRAP_FRESH_STATE_UNPROVEN")
        _validate_formal_binding(
            bootstrap.get("formal_binding"),
            expected_formal_binding,
            reasons,
        )

    if runtime:
        _expect_exact_fields(
            reasons,
            runtime,
            {
                "kind",
                "schema_version",
                "lifecycle_state",
                "condition",
                "target_model",
                "run_id",
                "protocol_id",
                "condition_fingerprint",
                "fairness_fingerprints",
                "bootstrap_audit_path",
                "bootstrap_audit_sha256",
                "plan_path",
                "plan_sha256",
                "seed",
                "online_overlay",
                "action_exemplars",
                "operations",
                "violations",
                "finalized",
            },
            "TABLE4_RUNTIME_AUDIT_FIELDS_MISMATCH",
        )
        _expect_equal(
            reasons,
            runtime.get("kind"),
            "table4_memory_runtime_audit",
            "TABLE4_RUNTIME_AUDIT_KIND_MISMATCH",
        )
        _expect_schema_v1(
            reasons,
            runtime.get("schema_version"),
            "TABLE4_RUNTIME_AUDIT_SCHEMA_MISMATCH",
        )
        _expect_equal(
            reasons,
            runtime.get("lifecycle_state"),
            "finalized",
            "TABLE4_MEMORY_LIFECYCLE_NOT_FINALIZED",
        )
        if runtime.get("finalized") is not True:
            _add(reasons, "TABLE4_MEMORY_FINALIZED_FLAG_MISSING")
        violations = runtime.get("violations")
        if not isinstance(violations, list) or violations:
            _add(reasons, "TABLE4_MEMORY_LIFECYCLE_VIOLATIONS")
        _expect_equal(
            reasons,
            _normalize_condition(runtime.get("condition")),
            expected_condition_value,
            "TABLE4_RUNTIME_CONDITION_MISMATCH",
        )
        _expect_equal(
            reasons,
            runtime.get("target_model"),
            expected_target_model,
            "TABLE4_RUNTIME_TARGET_MODEL_MISMATCH",
        )
        for field in ("run_id", "protocol_id", "condition_fingerprint"):
            _expect_equal(
                reasons,
                runtime.get(field),
                expected_formal_binding.get(field),
                f"TABLE4_RUNTIME_{field.upper()}_MISMATCH",
            )
        fairness = runtime.get("fairness_fingerprints")
        expected_fairness = {
            field: expected_formal_binding.get(field)
            for field in (
                "observation_contract_fingerprint",
                "task_instruction_fingerprint",
                "shared_prompt_contract_fingerprint",
                "decision_prompt_fingerprint",
                "provider_route_fingerprint",
                "skill_api_fingerprint",
                "full_method_compatibility_fingerprint",
            )
        }
        if not isinstance(fairness, dict) or fairness != expected_fairness:
            _add(reasons, "TABLE4_RUNTIME_FAIRNESS_FINGERPRINTS_MISMATCH")
        _expect_external_path(
            reasons,
            runtime.get("plan_path"),
            expected_formal_binding.get("plan_path"),
            "TABLE4_RUNTIME_PLAN_PATH_MISMATCH",
        )
        _expect_equal(
            reasons,
            runtime.get("plan_sha256"),
            expected_formal_binding.get("plan_sha256"),
            "TABLE4_RUNTIME_PLAN_HASH_MISMATCH",
        )
        _expect_path(
            reasons,
            runtime.get("bootstrap_audit_path"),
            root,
            memory_root,
            bootstrap_path,
            "TABLE4_RUNTIME_BOOTSTRAP_PATH_MISMATCH",
        )
        _expect_equal(
            reasons,
            runtime.get("bootstrap_audit_sha256"),
            bootstrap_sha256,
            "TABLE4_RUNTIME_BOOTSTRAP_HASH_MISMATCH",
        )
        _validate_runtime_seed(
            runtime.get("seed"),
            root,
            memory_root,
            seed_path,
            expected_strategy_bank_sha256,
            reasons,
        )
        _validate_runtime_mutable_store(
            runtime.get("online_overlay"),
            root=root,
            memory_root=memory_root,
            expected_path=overlay_path,
            initial_sha256=_EMPTY_OVERLAY_SHA256,
            initial_count_field="initial_record_count",
            final_count_field="final_record_count",
            collection_field="entries",
            validator=validate_online_strategy_overlay,
            reason_prefix="TABLE4_OVERLAY",
            reasons=reasons,
        )
        _validate_runtime_mutable_store(
            runtime.get("action_exemplars"),
            root=root,
            memory_root=memory_root,
            expected_path=action_path,
            initial_sha256=_EMPTY_ACTION_EXEMPLARS_SHA256,
            initial_count_field="initial_count",
            final_count_field="final_count",
            collection_field="exemplars",
            validator=validate_action_exemplar_memory,
            reason_prefix="TABLE4_ACTION_EXEMPLARS",
            reasons=reasons,
        )
        _validate_operations(runtime.get("operations"), reasons)
        _validate_operation_store_counts(runtime, reasons)

    seed_payload = _read_json(seed_path)
    try:
        _require_document_schema_v1(seed_payload)
        validate_strategy_bank(seed_payload)
    except (TypeError, ValueError):
        _add(reasons, "TABLE4_SEED_SCHEMA_INVALID")
        loaded_seed = ""
    else:
        loaded_seed = _semantic_payload_sha256(seed_payload)
    if not loaded_seed:
        _add(reasons, "TABLE4_SEED_FILE_MISSING_OR_INVALID")
    elif loaded_seed != expected_strategy_bank_sha256:
        _add(reasons, "TABLE4_CURRENT_SEED_HASH_MISMATCH")
    else:
        try:
            if seed_path.stat().st_mode & 0o222:
                _add(reasons, "TABLE4_SEED_FILE_WRITABLE")
        except OSError:
            _add(reasons, "TABLE4_SEED_FILE_MISSING_OR_INVALID")

    return Table4MemoryAuditValidation(
        invalid_reasons=tuple(reasons),
        loaded_seed_sha256=loaded_seed,
        artifact_hashes={
            "memory_bootstrap_audit": bootstrap_file_sha256,
            "memory_bootstrap_snapshot": snapshot_file_sha256,
            "memory_runtime_audit": runtime_sha256,
        },
    )


def _validate_bootstrap_seed(
    value: object,
    root: Path,
    memory_root: Path,
    expected_path: Path,
    expected_sha256: str,
    expected_strategy_bank_pin: StrategyBankPin,
    reasons: list[str],
) -> None:
    if not isinstance(value, dict):
        _add(reasons, "TABLE4_BOOTSTRAP_SEED_EVIDENCE_MISSING")
        return
    _expect_exact_fields(
        reasons,
        value,
        {
            "path",
            "sha256",
            "record_count",
            "immutable",
            "mode",
            "provenance_path",
            "provenance_sha256",
        },
        "TABLE4_BOOTSTRAP_SEED_FIELDS_MISMATCH",
    )
    _expect_path(
        reasons,
        value.get("path"),
        root,
        memory_root,
        expected_path,
        "TABLE4_BOOTSTRAP_SEED_PATH_MISMATCH",
    )
    _expect_equal(
        reasons,
        value.get("sha256"),
        expected_sha256,
        "TABLE4_BOOTSTRAP_SEED_HASH_MISMATCH",
    )
    if value.get("immutable") is not True:
        _add(reasons, "TABLE4_BOOTSTRAP_SEED_IMMUTABILITY_UNPROVEN")
    count = value.get("record_count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        _add(reasons, "TABLE4_BOOTSTRAP_SEED_RECORD_COUNT_INVALID")
    provenance_sha256 = value.get("provenance_sha256")
    if not isinstance(provenance_sha256, str) or not _SHA256_RE.fullmatch(
        provenance_sha256
    ):
        _add(reasons, "TABLE4_BOOTSTRAP_PROVENANCE_HASH_INVALID")
    provenance_path = value.get("provenance_path")
    if not isinstance(provenance_path, str) or not provenance_path.strip():
        _add(reasons, "TABLE4_BOOTSTRAP_PROVENANCE_PATH_INVALID")
    else:
        candidate = Path(provenance_path)
        if not candidate.is_absolute():
            _add(reasons, "TABLE4_BOOTSTRAP_PROVENANCE_PATH_INVALID")
        elif (
            isinstance(provenance_sha256, str)
            and _SHA256_RE.fullmatch(provenance_sha256)
            and _semantic_file_sha256(candidate) != provenance_sha256
        ):
            _add(reasons, "TABLE4_BOOTSTRAP_PROVENANCE_HASH_MISMATCH")
        else:
            try:
                validate_strategy_bank_provenance(
                    candidate,
                    strategy_bank_pin=expected_strategy_bank_pin,
                    expected_bank_sha256=expected_sha256,
                )
            except StrategyBankProvenanceError:
                _add(reasons, "TABLE4_BOOTSTRAP_PROVENANCE_IDENTITY_MISMATCH")


def _validate_formal_binding(
    value: object,
    expected: Mapping[str, str],
    reasons: list[str],
) -> None:
    fields = (
        "run_id",
        "protocol_id",
        "condition_fingerprint",
        "strategy_bank_sha256",
        "source_model",
        "source_provider_route_id",
        "source_provider_route_fingerprint",
        "target_model",
        "provider_route_id",
        "provider_route_fingerprint",
        "plan_path",
        "plan_sha256",
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    )
    if not isinstance(value, dict):
        _add(reasons, "TABLE4_FORMAL_BINDING_MISSING")
        return
    _expect_exact_fields(
        reasons,
        value,
        set(fields),
        "TABLE4_FORMAL_BINDING_FIELDS_MISMATCH",
    )
    if any(not str(expected.get(field) or "").strip() for field in fields):
        _add(reasons, "TABLE4_FORMAL_BINDING_EXPECTATION_MISSING")
        return
    for field in fields:
        if field == "plan_path":
            _expect_external_path(
                reasons,
                value.get(field),
                expected.get(field),
                "TABLE4_FORMAL_BINDING_PLAN_PATH_MISMATCH",
            )
        else:
            _expect_equal(
                reasons,
                value.get(field),
                expected.get(field),
                f"TABLE4_FORMAL_BINDING_{field.upper()}_MISMATCH",
            )
    plan_path = Path(expected["plan_path"])
    if (
        not plan_path.is_absolute()
        or _file_sha256(plan_path) != expected["plan_sha256"]
    ):
        _add(reasons, "TABLE4_EXTERNAL_PLAN_HASH_MISMATCH")


def _validate_bootstrap_overlay(
    value: object,
    root: Path,
    memory_root: Path,
    expected_path: Path,
    reasons: list[str],
) -> None:
    if not isinstance(value, dict):
        _add(reasons, "TABLE4_BOOTSTRAP_OVERLAY_EVIDENCE_MISSING")
        return
    _expect_exact_fields(
        reasons,
        value,
        {
            "path",
            "sha256_at_bootstrap",
            "record_count_at_bootstrap",
            "separate_from_seed",
            "target_run_local_updates_allowed",
        },
        "TABLE4_BOOTSTRAP_OVERLAY_FIELDS_MISMATCH",
    )
    _expect_path(
        reasons,
        value.get("path"),
        root,
        memory_root,
        expected_path,
        "TABLE4_BOOTSTRAP_OVERLAY_PATH_MISMATCH",
    )
    _expect_equal(
        reasons,
        value.get("sha256_at_bootstrap"),
        _EMPTY_OVERLAY_SHA256,
        "TABLE4_BOOTSTRAP_OVERLAY_NOT_EMPTY",
    )
    if value.get("record_count_at_bootstrap") != 0 or isinstance(
        value.get("record_count_at_bootstrap"), bool
    ):
        _add(reasons, "TABLE4_BOOTSTRAP_OVERLAY_NOT_EMPTY")
    if value.get("separate_from_seed") is not True:
        _add(reasons, "TABLE4_BOOTSTRAP_OVERLAY_NOT_ISOLATED")


def _validate_runtime_seed(
    value: object,
    root: Path,
    memory_root: Path,
    expected_path: Path,
    expected_sha256: str,
    reasons: list[str],
) -> None:
    if not isinstance(value, dict):
        _add(reasons, "TABLE4_RUNTIME_SEED_EVIDENCE_MISSING")
        return
    _expect_exact_fields(
        reasons,
        value,
        {
            "path",
            "expected_sha256",
            "preflight_sha256",
            "current_sha256",
            "final_sha256",
            "immutable_preflight",
            "immutable_final",
        },
        "TABLE4_RUNTIME_SEED_FIELDS_MISMATCH",
    )
    _expect_path(
        reasons,
        value.get("path"),
        root,
        memory_root,
        expected_path,
        "TABLE4_RUNTIME_SEED_PATH_MISMATCH",
    )
    for field in (
        "expected_sha256",
        "preflight_sha256",
        "current_sha256",
        "final_sha256",
    ):
        _expect_equal(
            reasons,
            value.get(field),
            expected_sha256,
            f"TABLE4_RUNTIME_SEED_{field.upper()}_MISMATCH",
        )
    if value.get("immutable_preflight") is not True:
        _add(reasons, "TABLE4_RUNTIME_SEED_PREFLIGHT_IMMUTABILITY_UNPROVEN")
    if value.get("immutable_final") is not True:
        _add(reasons, "TABLE4_RUNTIME_SEED_FINAL_IMMUTABILITY_UNPROVEN")


def _validate_runtime_mutable_store(
    value: object,
    *,
    root: Path,
    memory_root: Path,
    expected_path: Path,
    initial_sha256: str,
    initial_count_field: str,
    final_count_field: str,
    collection_field: str,
    validator: Any,
    reason_prefix: str,
    reasons: list[str],
) -> None:
    if not isinstance(value, dict):
        _add(reasons, f"{reason_prefix}_EVIDENCE_MISSING")
        return
    _expect_exact_fields(
        reasons,
        value,
        {
            "path",
            "initial_sha256",
            "current_sha256",
            "final_sha256",
            initial_count_field,
            final_count_field,
            "current_record_count"
            if collection_field == "entries"
            else "current_count",
        },
        f"{reason_prefix}_FIELDS_MISMATCH",
    )
    _expect_path(
        reasons,
        value.get("path"),
        root,
        memory_root,
        expected_path,
        f"{reason_prefix}_PATH_MISMATCH",
    )
    _expect_equal(
        reasons,
        value.get("initial_sha256"),
        initial_sha256,
        f"{reason_prefix}_INITIAL_HASH_MISMATCH",
    )
    initial_count = value.get(initial_count_field)
    if isinstance(initial_count, bool) or initial_count != 0:
        _add(reasons, f"{reason_prefix}_INITIAL_COUNT_NOT_ZERO")

    payload = _read_json(expected_path)
    try:
        # Mutable strategy/action stores own independent schema versions.  In
        # particular, the online overlay advanced to v2 when route-abstraction
        # became mandatory.  The collection validator is the sole schema
        # authority here; the frozen seed remains explicitly schema v1.
        payload = validator(payload)
    except (TypeError, ValueError):
        _add(reasons, f"{reason_prefix}_COLLECTION_INVALID")
        return
    current_sha256 = _semantic_payload_sha256(payload)
    if not current_sha256:
        _add(reasons, f"{reason_prefix}_FILE_MISSING_OR_INVALID")
        return
    for field in ("current_sha256", "final_sha256"):
        _expect_equal(
            reasons,
            value.get(field),
            current_sha256,
            f"{reason_prefix}_{field.upper()}_MISMATCH",
        )
    collection = payload.get(collection_field) if isinstance(payload, dict) else None
    if not isinstance(collection, list):
        _add(reasons, f"{reason_prefix}_COLLECTION_INVALID")
        return
    final_count = value.get(final_count_field)
    if isinstance(final_count, bool) or final_count != len(collection):
        _add(reasons, f"{reason_prefix}_FINAL_COUNT_MISMATCH")
    current_count_field = (
        "current_record_count" if collection_field == "entries" else "current_count"
    )
    current_count = value.get(current_count_field)
    if isinstance(current_count, bool) or current_count != len(collection):
        _add(reasons, f"{reason_prefix}_CURRENT_COUNT_MISMATCH")


def _validate_operations(value: object, reasons: list[str]) -> None:
    if not isinstance(value, dict):
        _add(reasons, "TABLE4_MEMORY_OPERATIONS_MISSING")
        return
    for field in (
        "strategy_retrievals",
        "exemplar_retrievals",
        "strategy_promotions",
        "exemplar_promotions",
        "promotion_noops",
    ):
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _add(reasons, "TABLE4_MEMORY_OPERATIONS_INVALID")
            return


def _validate_operation_store_counts(
    runtime: Mapping[str, Any],
    reasons: list[str],
) -> None:
    operations = runtime.get("operations")
    overlay = runtime.get("online_overlay")
    actions = runtime.get("action_exemplars")
    if not all(isinstance(value, Mapping) for value in (operations, overlay, actions)):
        return
    pairs = (
        (
            operations.get("strategy_promotions"),
            overlay.get("final_record_count"),
            "TABLE4_STRATEGY_PROMOTION_COUNT_MISMATCH",
        ),
        (
            operations.get("exemplar_promotions"),
            actions.get("final_count"),
            "TABLE4_EXEMPLAR_PROMOTION_COUNT_MISMATCH",
        ),
    )
    for promotions, final_count, reason in pairs:
        if (
            isinstance(promotions, bool)
            or not isinstance(promotions, int)
            or isinstance(final_count, bool)
            or not isinstance(final_count, int)
            or promotions != final_count
        ):
            _add(reasons, reason)


def _memory_paths_are_isolated(
    root: Path,
    memory_root: Path,
    registered_paths: tuple[Path, ...],
) -> bool:
    try:
        root_absolute = root.absolute()
        memory_absolute = memory_root.absolute()
        if (
            _path_or_ancestor_is_symlink(root_absolute)
            or _path_or_ancestor_is_symlink(memory_absolute)
            or memory_absolute.parent != root_absolute
        ):
            return False
        resolved_memory_root = memory_root.resolve(strict=True)
        resolved_paths = [path.resolve() for path in registered_paths]
        return (
            len(set(resolved_paths)) == len(resolved_paths)
            and all(
                path.absolute().parent == memory_absolute
                and not path.is_symlink()
                and resolved.is_relative_to(resolved_memory_root)
                for path, resolved in zip(registered_paths, resolved_paths)
            )
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _path_or_ancestor_is_symlink(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def _expect_exact_fields(
    reasons: list[str],
    value: Mapping[str, Any],
    expected: set[str],
    reason: str,
) -> None:
    if set(value) != expected:
        _add(reasons, reason)


def _expect_path(
    reasons: list[str],
    value: object,
    root: Path,
    memory_root: Path,
    expected: Path,
    reason: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _add(reasons, reason)
        return
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (
            root / candidate
            if candidate.parts[:1] == ("memory",)
            else memory_root / candidate
        )
    try:
        if (
            candidate.absolute() != expected.absolute()
            or candidate.resolve() != expected.resolve()
            or expected.is_symlink()
        ):
            _add(reasons, reason)
    except (OSError, RuntimeError):
        _add(reasons, reason)


def _expect_schema_v1(reasons: list[str], value: object, reason: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        _add(reasons, reason)


def _require_document_schema_v1(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("document root must be an object")
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError("document schema_version must be integer 1")


def _expect_external_path(
    reasons: list[str],
    value: object,
    expected: object,
    reason: str,
) -> None:
    if not isinstance(value, str) or not isinstance(expected, str):
        _add(reasons, reason)
        return
    try:
        if not Path(value).is_absolute() or Path(value).resolve() != Path(expected).resolve():
            _add(reasons, reason)
    except (OSError, RuntimeError):
        _add(reasons, reason)


def _expect_equal(
    reasons: list[str], value: object, expected: object, reason: str
) -> None:
    if value != expected:
        _add(reasons, reason)


def _normalize_condition(value: object) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "")
    normalized = normalized.replace("->", "→").replace("_to_", "→").replace("-to-", "→")
    return {
        "gpt→gpt": "gpt_to_gpt",
        "gpttogpt": "gpt_to_gpt",
        "gemini→gpt": "gemini_to_gpt",
        "geminitogpt": "gemini_to_gpt",
        "empty": "empty",
    }.get(normalized, normalized)


def _read_object(path: Path, reasons: list[str], reason: str) -> dict[str, Any]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        _add(reasons, reason)
        return {}
    return payload


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _semantic_file_sha256(path: Path) -> str:
    payload = _read_json(path)
    return _semantic_payload_sha256(payload)


def _semantic_payload_sha256(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    try:
        return semantic_sha256(payload)
    except (TypeError, ValueError):
        return ""


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _add(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


__all__ = ["Table4MemoryAuditValidation", "validate_table4_memory_audit"]
