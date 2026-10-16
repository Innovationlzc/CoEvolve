"""Fail-closed validation of finalized V83-native memory evidence.

This collector-side module understands the audit emitted by
``V83NativeMemoryBridge`` without importing the frozen V83 runner.  Expected
run identity, policy, terminal, and transfer-bank identity are supplied by the
caller; the audit is evidence, never its own source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from ..memory.schema import semantic_sha256
from ..memory.v83_native import (
    V83_NATIVE_ACTION_LIMIT,
    V83_NATIVE_AUDIT_FILENAME,
    V83_NATIVE_AUDIT_KIND,
    V83_NATIVE_AUDIT_SCHEMA_VERSION,
    V83_NATIVE_STRATEGY_LIMIT,
)
from ..memory.validation import validate_strategy_bank
from ..serialization import strict_json_loads, to_primitive


V83_NATIVE_MEMORY_RELATIVE_ROOT = Path("memory/v83_native")
V83_NATIVE_STRATEGY_FILENAME = "strategy_specs.json"
V83_NATIVE_ACTION_FILENAME = "action_exemplars.json"

_VALID_MODES = frozenset({"from_empty", "disabled", "preloaded_seed"})
_VALID_TASK_TERMINALS = frozenset(
    {
        "SUCCESS",
        "SYSTEM_STALL",
        "PHYSICAL_BUDGET_EXHAUSTED",
        "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
    }
)
_GATE_FIELDS = (
    "strategy_retrieval",
    "strategy_promotion",
    "action_exemplar_retrieval",
    "action_exemplar_promotion",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_FIELDS = {
    "kind",
    "schema_version",
    "lifecycle_state",
    "finalized",
    "mode",
    "run_binding",
    "run_binding_sha256",
    "cross_run_retrieval",
    "operation_gates",
    "initial_state",
    "operations",
    "suppressed_boundary_calls",
    "operation_errors",
    "violations",
    "final_state",
    "terminal_reason",
}
_STORE_STATE_FIELDS = {"path", "exists", "sha256", "record_count"}
_SEED_STATE_FIELDS = {
    "path",
    "file_sha256",
    "semantic_sha256",
    "record_count",
    "schema_projection",
    "native_mode_invented",
    "immutable_verified",
    "verification_point",
}
_OPERATION_FIELDS: Mapping[str, tuple[str, ...]] = {
    "strategy_retrieval": (
        "invocations",
        "returned_records",
        "native_records_returned",
        "transfer_seed_records_returned",
    ),
    "strategy_promotion": ("attempts", "promotions", "noops", "errors"),
    "action_exemplar_retrieval": ("invocations", "returned_exemplars"),
    "action_exemplar_promotion": (
        "attempts",
        "promotions",
        "noops",
        "errors",
    ),
    "memory_writer": ("calls", "failures"),
}


@dataclass(frozen=True, slots=True)
class V83NativeMemoryAuditValidation:
    invalid_reasons: tuple[str, ...]
    loaded_seed_sha256: str
    artifact_hashes: Mapping[str, str]

    @property
    def valid(self) -> bool:
        return not self.invalid_reasons


@dataclass(frozen=True, slots=True)
class _CurrentStore:
    exists: bool
    sha256: str
    record_count: int
    valid: bool


@dataclass(frozen=True, slots=True)
class _ExpectedSeed:
    path: Path
    file_sha256: str
    semantic_sha256: str
    record_count: int
    valid: bool


def validate_v83_native_memory_audit(
    run_dir: str | Path,
    *,
    expected_run_binding: Mapping[str, Any],
    expected_mode: str,
    expected_operation_gates: Mapping[str, bool],
    expected_terminal_reason: str,
    expected_transfer_bank: str | Path | None = None,
    expected_strategy_bank_sha256: str = "",
) -> V83NativeMemoryAuditValidation:
    """Validate one finalized native-memory audit against external identity.

    Native strategy/action files have one fixed location beneath ``run_dir``.
    They may remain absent when the final record count is zero.  If present,
    their strict JSON, byte digest, and record count must match the final audit
    state.  Initial state is necessarily historical, so it is required to be
    a well-formed zero-record state and must equal the current/final state when
    the audit reports no successful promotion.
    """

    reasons: list[str] = []
    root = Path(run_dir).absolute()
    native_root = root / V83_NATIVE_MEMORY_RELATIVE_ROOT
    audit_path = native_root / V83_NATIVE_AUDIT_FILENAME
    strategy_path = native_root / V83_NATIVE_STRATEGY_FILENAME
    action_path = native_root / V83_NATIVE_ACTION_FILENAME

    if not _paths_are_fixed_and_safe(
        root,
        native_root,
        (audit_path, strategy_path, action_path),
    ):
        _add(reasons, "V83_NATIVE_MEMORY_PATHS_NOT_ISOLATED")

    audit_hash = _file_sha256(audit_path)
    strategy = _load_native_store(
        strategy_path,
        expected_kind="deployment_decision_memory",
        expected_schema=2,
        collection="specs",
        reason_prefix="V83_NATIVE_STRATEGY_STORE",
        reasons=reasons,
    )
    action = _load_native_store(
        action_path,
        expected_kind="verified_action_exemplar_memory",
        expected_schema=1,
        collection="exemplars",
        reason_prefix="V83_NATIVE_ACTION_STORE",
        reasons=reasons,
    )
    expected_seed = _load_expected_seed(
        expected_transfer_bank,
        expected_mode=expected_mode,
        expected_strategy_bank_sha256=expected_strategy_bank_sha256,
        registered_paths=(audit_path, strategy_path, action_path),
        reasons=reasons,
    )
    artifacts = {
        str(V83_NATIVE_MEMORY_RELATIVE_ROOT / V83_NATIVE_AUDIT_FILENAME): audit_hash,
        str(V83_NATIVE_MEMORY_RELATIVE_ROOT / V83_NATIVE_STRATEGY_FILENAME): (
            strategy.sha256
        ),
        str(V83_NATIVE_MEMORY_RELATIVE_ROOT / V83_NATIVE_ACTION_FILENAME): (
            action.sha256
        ),
        "v83_native_transfer_seed": (
            expected_seed.file_sha256 if expected_seed is not None else ""
        ),
    }

    audit = _read_strict_object(
        audit_path,
        "V83_NATIVE_AUDIT_MISSING_OR_INVALID",
        reasons,
    )
    if audit is None:
        return _result(reasons, expected_seed, artifacts)

    if set(audit) != _AUDIT_FIELDS:
        _add(reasons, "V83_NATIVE_AUDIT_FIELDS_MISMATCH")
    if audit.get("kind") != V83_NATIVE_AUDIT_KIND:
        _add(reasons, "V83_NATIVE_AUDIT_KIND_MISMATCH")
    audit_schema = audit.get("schema_version")
    if (
        isinstance(audit_schema, bool)
        or not isinstance(audit_schema, int)
        or audit_schema != V83_NATIVE_AUDIT_SCHEMA_VERSION
    ):
        _add(reasons, "V83_NATIVE_AUDIT_SCHEMA_MISMATCH")
    if audit.get("lifecycle_state") != "finalized" or audit.get("finalized") is not True:
        _add(reasons, "V83_NATIVE_AUDIT_NOT_FINALIZED")

    terminal = audit.get("terminal_reason")
    if (
        not isinstance(expected_terminal_reason, str)
        or expected_terminal_reason not in _VALID_TASK_TERMINALS
    ):
        _add(reasons, "V83_NATIVE_EXPECTED_TERMINAL_NOT_TASK_TERMINAL")
    if not isinstance(terminal, str) or terminal not in _VALID_TASK_TERMINALS:
        _add(reasons, "V83_NATIVE_AUDIT_TERMINAL_NOT_TASK_TERMINAL")
    if terminal != expected_terminal_reason:
        _add(reasons, "V83_NATIVE_AUDIT_TERMINAL_MISMATCH")

    expected_binding = _normalized_expected_binding(expected_run_binding, reasons)
    if audit.get("run_binding") != expected_binding:
        _add(reasons, "V83_NATIVE_RUN_BINDING_MISMATCH")
    try:
        binding_sha256 = semantic_sha256(expected_binding)
    except (TypeError, ValueError):
        binding_sha256 = ""
        _add(reasons, "V83_NATIVE_EXPECTED_RUN_BINDING_INVALID")
    if not binding_sha256 or audit.get("run_binding_sha256") != binding_sha256:
        _add(reasons, "V83_NATIVE_RUN_BINDING_HASH_MISMATCH")

    if not isinstance(expected_mode, str) or expected_mode not in _VALID_MODES:
        _add(reasons, "V83_NATIVE_EXPECTED_MODE_INVALID")
    audit_mode = audit.get("mode")
    if (
        audit_mode != expected_mode
        or not isinstance(audit_mode, str)
        or audit_mode not in _VALID_MODES
    ):
        _add(reasons, "V83_NATIVE_MODE_MISMATCH")
    expected_gates = _normalize_expected_gates(expected_operation_gates, reasons)
    gates = audit.get("operation_gates")
    if gates != expected_gates or not _valid_gate_document(gates):
        _add(reasons, "V83_NATIVE_OPERATION_GATES_MISMATCH")
    if expected_mode == "disabled" and (
        not expected_gates or any(expected_gates.values())
    ):
        _add(reasons, "V83_NATIVE_DISABLED_GATES_NOT_CLOSED")
    if isinstance(expected_mode, str) and expected_mode in {
        "from_empty",
        "preloaded_seed",
    } and (
        expected_gates and not any(expected_gates.values())
    ):
        _add(reasons, "V83_NATIVE_ACTIVE_MODE_GATES_ALL_CLOSED")
    if audit.get("cross_run_retrieval") is not (expected_mode == "preloaded_seed"):
        _add(reasons, "V83_NATIVE_CROSS_RUN_RETRIEVAL_MISMATCH")

    violations = audit.get("violations")
    if not isinstance(violations, list) or violations:
        _add(reasons, "V83_NATIVE_AUDIT_VIOLATIONS")
    operation_errors = audit.get("operation_errors")
    if not isinstance(operation_errors, list) or not all(
        isinstance(item, Mapping) for item in operation_errors
    ):
        _add(reasons, "V83_NATIVE_OPERATION_ERRORS_INVALID")

    operations = _validate_operations(audit.get("operations"), reasons)
    _validate_suppressed_calls(audit.get("suppressed_boundary_calls"), reasons)
    _validate_operations_against_gates(
        operations,
        expected_gates,
        expected_mode,
        operation_errors,
        reasons,
    )

    initial = audit.get("initial_state")
    final = audit.get("final_state")
    if not isinstance(initial, Mapping) or set(initial) != {
        "strategy_store",
        "action_exemplar_store",
        "transfer_seed",
    }:
        _add(reasons, "V83_NATIVE_INITIAL_STATE_INVALID")
        initial = {}
    if not isinstance(final, Mapping) or set(final) != {
        "strategy_store",
        "action_exemplar_store",
        "transfer_seed",
    }:
        _add(reasons, "V83_NATIVE_FINAL_STATE_INVALID")
        final = {}

    strategy_promotions = _operation_value(
        operations, "strategy_promotion", "promotions"
    )
    action_promotions = _operation_value(
        operations, "action_exemplar_promotion", "promotions"
    )
    _validate_store_evidence(
        initial.get("strategy_store"),
        final.get("strategy_store"),
        expected_path=strategy_path,
        current=strategy,
        successful_promotions=strategy_promotions,
        prefix="V83_NATIVE_STRATEGY_STORE",
        reasons=reasons,
    )
    _validate_store_evidence(
        initial.get("action_exemplar_store"),
        final.get("action_exemplar_store"),
        expected_path=action_path,
        current=action,
        successful_promotions=action_promotions,
        prefix="V83_NATIVE_ACTION_STORE",
        reasons=reasons,
    )
    expected_seed = _validate_seed_evidence(
        initial.get("transfer_seed"),
        final.get("transfer_seed"),
        expected_mode=expected_mode,
        expected=expected_seed,
        reasons=reasons,
    )

    if expected_mode == "disabled":
        if strategy.record_count != 0 or action.record_count != 0:
            _add(reasons, "V83_NATIVE_DISABLED_MEMORY_RECORDS_NONZERO")
        if isinstance(operation_errors, list) and operation_errors:
            _add(reasons, "V83_NATIVE_DISABLED_OPERATION_ERRORS_NONEMPTY")

    return _result(reasons, expected_seed, artifacts)


def _load_native_store(
    path: Path,
    *,
    expected_kind: str,
    expected_schema: int,
    collection: str,
    reason_prefix: str,
    reasons: list[str],
) -> _CurrentStore:
    if not path.exists():
        return _CurrentStore(False, "", 0, True)
    digest = _file_sha256(path)
    payload = _read_strict_object(path, f"{reason_prefix}_INVALID", reasons)
    if payload is None:
        return _CurrentStore(True, digest, -1, False)
    records = payload.get(collection)
    valid = (
        payload.get("kind") == expected_kind
        and not isinstance(payload.get("schema_version"), bool)
        and isinstance(payload.get("schema_version"), int)
        and payload.get("schema_version") == expected_schema
        and isinstance(records, list)
        and all(isinstance(item, Mapping) for item in records)
    )
    if not valid:
        _add(reasons, f"{reason_prefix}_SCHEMA_INVALID")
        return _CurrentStore(True, digest, -1, False)
    return _CurrentStore(True, digest, len(records), True)


def _load_expected_seed(
    value: str | Path | None,
    *,
    expected_mode: str,
    expected_strategy_bank_sha256: str,
    registered_paths: tuple[Path, ...],
    reasons: list[str],
) -> _ExpectedSeed | None:
    if expected_mode != "preloaded_seed":
        if value is not None:
            _add(reasons, "V83_NATIVE_UNEXPECTED_TRANSFER_BANK")
        if expected_strategy_bank_sha256:
            _add(reasons, "V83_NATIVE_UNEXPECTED_STRATEGY_BANK_EXPECTATION")
        return None
    if value is None:
        _add(reasons, "V83_NATIVE_EXPECTED_TRANSFER_BANK_MISSING")
        return None
    if not isinstance(expected_strategy_bank_sha256, str) or not _SHA256_RE.fullmatch(
        expected_strategy_bank_sha256
    ):
        _add(reasons, "V83_NATIVE_TRANSFER_SEED_EXPECTATION_INVALID")
    if not isinstance(value, (str, Path)):
        _add(reasons, "V83_NATIVE_EXPECTED_TRANSFER_BANK_INVALID")
        return _ExpectedSeed(Path("."), "", "", -1, False)
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        _add(reasons, "V83_NATIVE_EXPECTED_TRANSFER_BANK_INVALID")
        return _ExpectedSeed(path.absolute(), "", "", -1, False)
    path = path.resolve(strict=True)
    if path in {item.resolve(strict=False) for item in registered_paths}:
        _add(reasons, "V83_NATIVE_TRANSFER_BANK_NOT_SEPARATE")
    file_sha256 = _file_sha256(path)
    payload = _read_strict_object(
        path,
        "V83_NATIVE_EXPECTED_TRANSFER_BANK_INVALID",
        reasons,
    )
    if payload is None:
        return _ExpectedSeed(path, file_sha256, "", -1, False)
    validation = validate_strategy_bank(payload, raise_on_error=False)
    if not validation.valid:
        _add(reasons, "V83_NATIVE_EXPECTED_TRANSFER_BANK_SCHEMA_INVALID")
        return _ExpectedSeed(path, file_sha256, "", validation.record_count, False)
    if validation.bank_sha256 != expected_strategy_bank_sha256:
        _add(reasons, "V83_NATIVE_TRANSFER_SEED_SEMANTIC_HASH_MISMATCH")
    return _ExpectedSeed(
        path,
        file_sha256,
        validation.bank_sha256,
        validation.record_count,
        True,
    )


def _validate_store_evidence(
    initial: object,
    final: object,
    *,
    expected_path: Path,
    current: _CurrentStore,
    successful_promotions: int,
    prefix: str,
    reasons: list[str],
) -> None:
    if not isinstance(initial, Mapping) or set(initial) != _STORE_STATE_FIELDS:
        _add(reasons, f"{prefix}_INITIAL_STATE_INVALID")
        initial = {}
    if not isinstance(final, Mapping) or set(final) != _STORE_STATE_FIELDS:
        _add(reasons, f"{prefix}_FINAL_STATE_INVALID")
        final = {}
    expected_path_text = str(expected_path.resolve(strict=False))
    if initial.get("path") != expected_path_text:
        _add(reasons, f"{prefix}_INITIAL_PATH_MISMATCH")
    if final.get("path") != expected_path_text:
        _add(reasons, f"{prefix}_FINAL_PATH_MISMATCH")
    if initial.get("record_count") != 0:
        _add(reasons, f"{prefix}_INITIAL_COUNT_NONZERO")
    _validate_declared_file_state(initial, f"{prefix}_INITIAL", reasons)

    expected_final = {
        "path": expected_path_text,
        "exists": current.exists,
        "sha256": current.sha256,
        "record_count": current.record_count,
    }
    if dict(final) != expected_final:
        _add(reasons, f"{prefix}_FINAL_CURRENT_STATE_MISMATCH")

    if successful_promotions < 0:
        return
    if successful_promotions == 0:
        if dict(initial) != expected_final:
            _add(reasons, f"{prefix}_UNEXPLAINED_STATE_CHANGE")
    elif (
        not current.valid
        or not current.exists
        or current.record_count < 1
        or initial.get("sha256") == current.sha256
    ):
        _add(reasons, f"{prefix}_PROMOTION_STATE_MISMATCH")


def _validate_declared_file_state(
    value: Mapping[str, Any],
    prefix: str,
    reasons: list[str],
) -> None:
    exists = value.get("exists")
    sha256 = value.get("sha256")
    count = value.get("record_count")
    if not isinstance(exists, bool):
        _add(reasons, f"{prefix}_EXISTS_INVALID")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        _add(reasons, f"{prefix}_COUNT_INVALID")
    if exists is False and sha256 != "":
        _add(reasons, f"{prefix}_HASH_INVALID")
    if exists is True and (
        not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256)
    ):
        _add(reasons, f"{prefix}_HASH_INVALID")


def _validate_seed_evidence(
    initial: object,
    final: object,
    *,
    expected_mode: str,
    expected: _ExpectedSeed | None,
    reasons: list[str],
) -> _ExpectedSeed | None:
    if expected_mode != "preloaded_seed":
        if initial is not None or final is not None:
            _add(reasons, "V83_NATIVE_UNEXPECTED_TRANSFER_SEED_STATE")
        return None
    if expected is None or not expected.valid:
        _add(reasons, "V83_NATIVE_TRANSFER_SEED_EXPECTATION_INVALID")
        return expected
    expected_common = {
        "path": str(expected.path),
        "file_sha256": expected.file_sha256,
        "semantic_sha256": expected.semantic_sha256,
        "record_count": expected.record_count,
        "schema_projection": "immutable_public_six_field_strategy_bank",
        "native_mode_invented": False,
        "immutable_verified": True,
    }
    _validate_one_seed_state(initial, expected_common, "initial", reasons)
    _validate_one_seed_state(final, expected_common, "final", reasons)
    if isinstance(initial, Mapping) and isinstance(final, Mapping):
        left = {key: value for key, value in initial.items() if key != "verification_point"}
        right = {key: value for key, value in final.items() if key != "verification_point"}
        if left != right:
            _add(reasons, "V83_NATIVE_TRANSFER_SEED_CHANGED_DURING_RUN")
    return expected


def _validate_one_seed_state(
    value: object,
    expected_common: Mapping[str, Any],
    point: str,
    reasons: list[str],
) -> None:
    prefix = f"V83_NATIVE_TRANSFER_SEED_{point.upper()}"
    if not isinstance(value, Mapping) or set(value) != _SEED_STATE_FIELDS:
        _add(reasons, f"{prefix}_STATE_INVALID")
        return
    expected = {**expected_common, "verification_point": point}
    if dict(value) != expected:
        _add(reasons, f"{prefix}_STATE_MISMATCH")


def _validate_operations(value: object, reasons: list[str]) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != set(_OPERATION_FIELDS):
        _add(reasons, "V83_NATIVE_OPERATIONS_INVALID")
        return {}
    result: dict[str, dict[str, int]] = {}
    for operation, fields in _OPERATION_FIELDS.items():
        child = value.get(operation)
        if not isinstance(child, Mapping) or set(child) != set(fields):
            _add(reasons, "V83_NATIVE_OPERATIONS_INVALID")
            return {}
        counts: dict[str, int] = {}
        for field in fields:
            count = child.get(field)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                _add(reasons, "V83_NATIVE_OPERATIONS_INVALID")
                return {}
            counts[field] = count
        result[operation] = counts

    strategy_retrieval = result["strategy_retrieval"]
    if strategy_retrieval["returned_records"] != (
        strategy_retrieval["native_records_returned"]
        + strategy_retrieval["transfer_seed_records_returned"]
    ) or strategy_retrieval["returned_records"] > (
        strategy_retrieval["invocations"] * V83_NATIVE_STRATEGY_LIMIT
    ):
        _add(reasons, "V83_NATIVE_STRATEGY_RETRIEVAL_ACCOUNTING_MISMATCH")
    action_retrieval = result["action_exemplar_retrieval"]
    if action_retrieval["returned_exemplars"] > (
        action_retrieval["invocations"] * V83_NATIVE_ACTION_LIMIT
    ):
        _add(reasons, "V83_NATIVE_ACTION_RETRIEVAL_ACCOUNTING_MISMATCH")
    for operation in ("strategy_promotion", "action_exemplar_promotion"):
        counts = result[operation]
        if counts["attempts"] != (
            counts["promotions"] + counts["noops"] + counts["errors"]
        ):
            _add(reasons, "V83_NATIVE_PROMOTION_ACCOUNTING_MISMATCH")
    writer = result["memory_writer"]
    if (
        writer["failures"] > writer["calls"]
        or writer["calls"] > result["strategy_promotion"]["attempts"]
    ):
        _add(reasons, "V83_NATIVE_MEMORY_WRITER_ACCOUNTING_MISMATCH")
    return result


def _validate_suppressed_calls(value: object, reasons: list[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_GATE_FIELDS):
        _add(reasons, "V83_NATIVE_SUPPRESSED_CALLS_INVALID")
        return
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value.values()
    ):
        _add(reasons, "V83_NATIVE_SUPPRESSED_CALLS_INVALID")


def _validate_operations_against_gates(
    operations: Mapping[str, Mapping[str, int]],
    gates: Mapping[str, bool],
    mode: str,
    operation_errors: object,
    reasons: list[str],
) -> None:
    if not operations:
        return
    gated_operations = {
        "strategy_retrieval": "strategy_retrieval",
        "strategy_promotion": "strategy_promotion",
        "action_exemplar_retrieval": "action_exemplar_retrieval",
        "action_exemplar_promotion": "action_exemplar_promotion",
    }
    for gate, operation in gated_operations.items():
        if gates.get(gate) is False and any(operations[operation].values()):
            _add(reasons, "V83_NATIVE_CLOSED_GATE_OPERATION_DETECTED")
    if gates.get("strategy_promotion") is False and any(
        operations["memory_writer"].values()
    ):
        _add(reasons, "V83_NATIVE_CLOSED_GATE_WRITER_CALL_DETECTED")
    if mode != "preloaded_seed" and operations["strategy_retrieval"].get(
        "transfer_seed_records_returned", 0
    ):
        _add(reasons, "V83_NATIVE_UNEXPECTED_TRANSFER_RECORD_RETRIEVAL")
    if mode == "disabled":
        if any(
            count
            for child in operations.values()
            for count in child.values()
        ):
            _add(reasons, "V83_NATIVE_DISABLED_OPERATION_DETECTED")
        if isinstance(operation_errors, list) and operation_errors:
            _add(reasons, "V83_NATIVE_DISABLED_OPERATION_ERRORS_NONEMPTY")


def _normalize_expected_gates(
    value: Mapping[str, bool], reasons: list[str]
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(_GATE_FIELDS) or not all(
        isinstance(item, bool) for item in value.values()
    ):
        _add(reasons, "V83_NATIVE_EXPECTED_OPERATION_GATES_INVALID")
        return {}
    return {field: value[field] for field in _GATE_FIELDS}


def _valid_gate_document(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(_GATE_FIELDS)
        and all(isinstance(item, bool) for item in value.values())
    )


def _normalized_expected_binding(
    value: Mapping[str, Any], reasons: list[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _add(reasons, "V83_NATIVE_EXPECTED_RUN_BINDING_INVALID")
        return {}
    try:
        primitive = to_primitive(dict(value))
    except (TypeError, ValueError):
        _add(reasons, "V83_NATIVE_EXPECTED_RUN_BINDING_INVALID")
        return {}
    if not isinstance(primitive, dict):  # guaranteed by dict(value)
        _add(reasons, "V83_NATIVE_EXPECTED_RUN_BINDING_INVALID")
        return {}
    return primitive


def _operation_value(
    operations: Mapping[str, Mapping[str, int]], operation: str, field: str
) -> int:
    child = operations.get(operation)
    return child.get(field, -1) if isinstance(child, Mapping) else -1


def _read_strict_object(
    path: Path,
    reason: str,
    reasons: list[str],
) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("path is missing, a symlink, or not a regular file")
        value = strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        _add(reasons, reason)
        return None
    if not isinstance(value, Mapping):
        _add(reasons, reason)
        return None
    return dict(value)


def _paths_are_fixed_and_safe(
    root: Path,
    native_root: Path,
    files: tuple[Path, ...],
) -> bool:
    try:
        if root.is_symlink() or (root / "memory").is_symlink() or native_root.is_symlink():
            return False
        resolved_root = root.resolve(strict=True)
        resolved_native = native_root.resolve(strict=True)
        if resolved_native != resolved_root / V83_NATIVE_MEMORY_RELATIVE_ROOT:
            return False
        return all(
            not path.is_symlink()
            and (
                not path.exists()
                or (path.is_file() and path.resolve(strict=True).parent == resolved_native)
            )
            for path in files
        )
    except (OSError, RuntimeError):
        return False


def _file_sha256(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _result(
    reasons: list[str],
    seed: _ExpectedSeed | None,
    artifacts: Mapping[str, str],
) -> V83NativeMemoryAuditValidation:
    return V83NativeMemoryAuditValidation(
        invalid_reasons=tuple(reasons),
        loaded_seed_sha256=(
            seed.semantic_sha256 if seed is not None and seed.valid else ""
        ),
        artifact_hashes=MappingProxyType(dict(artifacts)),
    )


def _add(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


__all__ = [
    "V83NativeMemoryAuditValidation",
    "V83_NATIVE_ACTION_FILENAME",
    "V83_NATIVE_MEMORY_RELATIVE_ROOT",
    "V83_NATIVE_STRATEGY_FILENAME",
    "validate_v83_native_memory_audit",
]
