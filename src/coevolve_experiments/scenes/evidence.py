"""Host-validated simulator evidence for formal scene execution.

The JSON receipts handled here are indexes over raw simulator artifacts, not
success certificates.  In particular, receipts do not contain a ``passed``
field: this module derives acceptance from an allowlisted producer identity,
the current protocol/scene pins, the bytes of every referenced artifact, and
frozen check-specific thresholds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

from ..contracts import ProtocolLock
from .evidence_measurements import measurement_errors, mujoco_model_inventory_errors
from .schema import EvidencePin, SceneSpec


EVIDENCE_RECEIPT_KIND = "coevolve_scene_execution_evidence"
EVIDENCE_RECEIPT_SCHEMA_VERSION = 2
EVIDENCE_CHECKS = (
    "simulator_integrity",
    "initial_collision",
    "scripted_solvability",
)

# Only repository-owned evidence producers may issue formal receipts.  Their
# implementation commit is also recorded and fingerprinted, while the stable
# producer ID/version pair is the explicit allowlist boundary.
ALLOWED_EVIDENCE_PRODUCERS: Mapping[str, Tuple[str, str]] = {
    "simulator_integrity": ("coevolve.scene.mujoco-model-integrity", "2.0.0"),
    "initial_collision": ("coevolve.scene.mujoco-initial-contact-support", "2.0.0"),
    "scripted_solvability": ("coevolve.scene.mujoco-scripted-solvability", "2.0.0"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ROOT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "check",
        "scene_hash",
        "execution_sha256",
        "execution_composite_sha256",
        "protocol_id",
        "invariant_fingerprint",
        "terminal_fingerprint",
        "skill_api_fingerprint",
        "producer_id",
        "producer_version",
        "producer_commit",
        "command_sha256",
        "config_sha256",
        "measurements",
        "raw_artifacts",
    }
)
_RAW_ARTIFACT_FIELDS = frozenset(
    {"artifact_type", "relative_path", "sha256", "size_bytes"}
)
_RAW_ARTIFACT_TYPES: Mapping[str, frozenset[str]] = {
    "simulator_integrity": frozenset(
        {"command", "config", "compiled_model", "model_inventory"}
    ),
    "initial_collision": frozenset(
        {"command", "config", "contact_trace", "support_trace"}
    ),
    "scripted_solvability": frozenset(
        {
            "command",
            "config",
            "script",
            "controller",
            "trajectory_trace",
            "verifier_trace",
        }
    ),
}


@dataclass(frozen=True)
class FrozenControllerPin:
    """Reviewed scripted controller bytes compatible with one frozen API."""

    controller_id: str
    controller_version: str
    skill_api_fingerprint: str
    controller_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.controller_id, str)
            or not self.controller_id.strip()
            or not isinstance(self.controller_version, str)
            or not self.controller_version.strip()
        ):
            raise ValueError("frozen controller identity must be non-empty")
        if not _is_sha256(self.skill_api_fingerprint):
            raise ValueError("frozen controller skill API must be a SHA-256 digest")
        if not _is_sha256(self.controller_sha256):
            raise ValueError("frozen controller bytes must be a SHA-256 digest")


# This is deliberately empty until the actual deterministic scene-solver
# bundle has been reviewed and its exact bytes have been pinned.  Evidence
# therefore fails closed; a receipt cannot turn an arbitrary privileged
# controller into a proof that the benchmark scene is solvable.  Adding a pin
# is a source-reviewed protocol change, not a receipt-controlled operation.
ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS: Mapping[
    tuple[str, str], FrozenControllerPin
] = MappingProxyType({})


@dataclass(frozen=True)
class RawArtifactBinding:
    """One raw artifact whose current bytes were checked by the host."""

    artifact_type: str
    relative_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class EvidenceReceiptBinding:
    """The auditable identity retained in an execution preflight report."""

    check: str
    receipt_path: str
    receipt_sha256: str
    normalized_receipt_sha256: str
    producer_id: str
    producer_version: str
    producer_commit: str
    command_sha256: str
    config_sha256: str
    measurements_sha256: str
    raw_artifacts: Tuple[RawArtifactBinding, ...]


@dataclass(frozen=True)
class EvidenceValidation:
    errors: Tuple[str, ...]
    binding: EvidenceReceiptBinding | None = None


def validate_evidence_receipt(
    root: Path,
    pin: EvidencePin,
    evidence_name: str,
    scene: SceneSpec,
    protocol_lock: ProtocolLock | None,
) -> EvidenceValidation:
    """Validate one receipt and independently derive its check outcome."""

    errors: list[str] = []
    if evidence_name not in EVIDENCE_CHECKS:
        return EvidenceValidation((f"unsupported simulator evidence check: {evidence_name}",))
    if protocol_lock is None:
        return EvidenceValidation(("protocol lock is required for simulator evidence",))
    receipt_path = root / pin.path
    errors.extend(
        _pinned_file_errors(
            root,
            pin.path,
            pin.sha256,
            None,
            f"{evidence_name} evidence",
        )
    )
    if errors:
        return EvidenceValidation(tuple(errors))
    try:
        raw_receipt = receipt_path.read_bytes()
        payload = json.loads(
            raw_receipt.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return EvidenceValidation(
            (f"{evidence_name} evidence is not readable JSON: {exc}",)
        )
    if not isinstance(payload, dict):
        return EvidenceValidation(
            (f"{evidence_name} evidence must be a JSON object",)
        )

    actual_fields = set(payload)
    missing_fields = sorted(_ROOT_FIELDS - actual_fields)
    unknown_fields = sorted(actual_fields - _ROOT_FIELDS)
    if "passed" in payload:
        errors.append(
            f"{evidence_name} evidence must not contain caller-supplied passed; "
            "the host validator derives the outcome"
        )
        unknown_fields = [field for field in unknown_fields if field != "passed"]
    if missing_fields or unknown_fields:
        errors.append(
            f"{evidence_name} evidence fields differ: "
            f"missing={missing_fields}, unknown={unknown_fields}"
        )

    expected_scalars: Mapping[str, Any] = {
        "kind": EVIDENCE_RECEIPT_KIND,
        "schema_version": EVIDENCE_RECEIPT_SCHEMA_VERSION,
        "check": evidence_name,
        "scene_hash": scene.scene_hash,
        "execution_sha256": scene.asset.execution_sha256,
        "execution_composite_sha256": scene.asset.execution_composite_sha256,
        "protocol_id": protocol_lock.protocol_id,
        "invariant_fingerprint": protocol_lock.invariant_fingerprint,
        "terminal_fingerprint": protocol_lock.terminal_fingerprint,
        "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
    }
    for field, expected in expected_scalars.items():
        if field not in payload:
            continue
        actual = payload[field]
        if field == "schema_version" and (
            isinstance(actual, bool) or not isinstance(actual, int)
        ):
            errors.append(f"{evidence_name} evidence has non-integer schema_version")
        elif actual != expected:
            errors.append(f"{evidence_name} evidence has wrong {field}")

    producer = ALLOWED_EVIDENCE_PRODUCERS[evidence_name]
    if payload.get("producer_id") != producer[0] or payload.get("producer_version") != producer[1]:
        errors.append(
            f"{evidence_name} evidence producer is not allowlisted for this check"
        )
    producer_commit = payload.get("producer_commit")
    if not _is_commit(producer_commit):
        errors.append(f"{evidence_name} evidence producer_commit must be lowercase 40-hex")
    elif producer_commit != protocol_lock.base_method_commit:
        errors.append(
            f"{evidence_name} evidence producer_commit does not match the protocol lock"
        )
    for field in ("command_sha256", "config_sha256"):
        if not _is_sha256(payload.get(field)):
            errors.append(f"{evidence_name} evidence {field} must be lowercase SHA-256")

    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        errors.append(f"{evidence_name} evidence measurements must be a JSON object")
    else:
        errors.extend(measurement_errors(evidence_name, measurements, scene))

    artifacts, artifact_errors = _validate_raw_artifacts(
        root,
        receipt_relative_path=pin.path,
        value=payload.get("raw_artifacts"),
        evidence_name=evidence_name,
    )
    errors.extend(artifact_errors)
    artifacts_by_type = {artifact.artifact_type: artifact for artifact in artifacts}
    if _is_sha256(payload.get("command_sha256")) and "command" in artifacts_by_type:
        if artifacts_by_type["command"].sha256 != payload["command_sha256"]:
            errors.append(f"{evidence_name} evidence command_sha256 does not match raw command")
    if _is_sha256(payload.get("config_sha256")) and "config" in artifacts_by_type:
        if artifacts_by_type["config"].sha256 != payload["config_sha256"]:
            errors.append(f"{evidence_name} evidence config_sha256 does not match raw config")
    if evidence_name == "scripted_solvability" and isinstance(measurements, dict):
        for measurement_field, artifact_type in (
            ("script_sha256", "script"),
            ("controller_sha256", "controller"),
        ):
            digest = measurements.get(measurement_field)
            artifact = artifacts_by_type.get(artifact_type)
            if _is_sha256(digest) and artifact is not None and artifact.sha256 != digest:
                errors.append(
                    f"{evidence_name} evidence {measurement_field} does not match raw {artifact_type}"
                )
        if measurements.get("skill_api_fingerprint") != protocol_lock.skill_api_fingerprint:
            errors.append(
                f"{evidence_name} evidence skill_api_fingerprint does not match "
                "the protocol lock"
            )
        errors.extend(
            _scripted_controller_errors(
                measurements,
                artifacts_by_type.get("controller"),
                protocol_lock,
                evidence_name,
            )
        )
    if evidence_name == "simulator_integrity" and isinstance(measurements, dict):
        for measurement_field, artifact_type in (
            ("compiled_model_sha256", "compiled_model"),
            ("model_inventory_sha256", "model_inventory"),
        ):
            digest = measurements.get(measurement_field)
            artifact = artifacts_by_type.get(artifact_type)
            if _is_sha256(digest) and artifact is not None and artifact.sha256 != digest:
                errors.append(
                    f"{evidence_name} evidence {measurement_field} does not match "
                    f"raw {artifact_type}"
                )
        inventory_artifact = artifacts_by_type.get("model_inventory")
        if inventory_artifact is not None:
            inventory, inventory_errors = _load_strict_json_artifact(
                root,
                inventory_artifact,
                f"{evidence_name} raw model_inventory",
            )
            errors.extend(inventory_errors)
            if not inventory_errors:
                errors.extend(
                    mujoco_model_inventory_errors(
                        inventory,
                        measurements,
                        evidence_name,
                    )
                )

    if errors:
        return EvidenceValidation(tuple(errors))
    # All fields have been type-checked above, so these casts are safe and keep
    # the immutable report compact while retaining every raw file identity.
    normalized_receipt_sha256 = _canonical_sha256(payload)
    measurements_sha256 = _canonical_sha256(measurements)
    binding = EvidenceReceiptBinding(
        check=evidence_name,
        receipt_path=pin.path,
        receipt_sha256=pin.sha256,
        normalized_receipt_sha256=normalized_receipt_sha256,
        producer_id=str(payload["producer_id"]),
        producer_version=str(payload["producer_version"]),
        producer_commit=str(payload["producer_commit"]),
        command_sha256=str(payload["command_sha256"]),
        config_sha256=str(payload["config_sha256"]),
        measurements_sha256=measurements_sha256,
        raw_artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_type)),
    )
    return EvidenceValidation((), binding)


def execution_evidence_fingerprint(
    scene: SceneSpec,
    protocol_lock: ProtocolLock,
    bindings: Sequence[EvidenceReceiptBinding],
) -> str:
    """Bind realized asset, lock, normalized receipts, and every raw hash."""

    ordered = tuple(sorted(bindings, key=lambda item: item.check))
    if tuple(item.check for item in ordered) != tuple(sorted(EVIDENCE_CHECKS)):
        raise ValueError("execution evidence fingerprint requires exactly all three checks")
    payload = {
        "kind": "coevolve_scene_execution_evidence_fingerprint",
        "schema_version": 1,
        "scene_hash": scene.scene_hash,
        "execution_sha256": scene.asset.execution_sha256,
        "execution_composite_sha256": scene.asset.execution_composite_sha256,
        "protocol_id": protocol_lock.protocol_id,
        "invariant_fingerprint": protocol_lock.invariant_fingerprint,
        "terminal_fingerprint": protocol_lock.terminal_fingerprint,
        "skill_api_fingerprint": protocol_lock.skill_api_fingerprint,
        "receipts": [asdict(binding) for binding in ordered],
    }
    return _canonical_sha256(payload)


def _scripted_controller_errors(
    measurements: Mapping[str, Any],
    artifact: RawArtifactBinding | None,
    protocol_lock: ProtocolLock,
    evidence_name: str,
) -> list[str]:
    """Require reviewed controller bytes, not a receipt-selected controller."""

    controller_id = measurements.get("controller_id")
    controller_version = measurements.get("controller_version")
    if not isinstance(controller_id, str) or not isinstance(controller_version, str):
        return []  # measurement schema already reports malformed identities
    key = (controller_id, controller_version)
    pin = ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS.get(key)
    if pin is None:
        return [
            f"{evidence_name} evidence controller {controller_id!r} "
            f"version {controller_version!r} is not in the reviewed controller allowlist"
        ]
    errors: list[str] = []
    if pin.controller_id != controller_id or pin.controller_version != controller_version:
        errors.append(f"{evidence_name} evidence controller allowlist identity is invalid")
    if pin.skill_api_fingerprint != protocol_lock.skill_api_fingerprint:
        errors.append(
            f"{evidence_name} evidence controller is not approved for the locked skill API"
        )
    if measurements.get("skill_api_fingerprint") != pin.skill_api_fingerprint:
        errors.append(
            f"{evidence_name} evidence controller skill API does not match its allowlist pin"
        )
    if measurements.get("controller_sha256") != pin.controller_sha256:
        errors.append(
            f"{evidence_name} evidence controller_sha256 does not match its allowlist pin"
        )
    if artifact is not None and artifact.sha256 != pin.controller_sha256:
        errors.append(
            f"{evidence_name} evidence raw controller bytes are not allowlisted"
        )
    return errors


def _load_strict_json_artifact(
    root: Path,
    artifact: RawArtifactBinding,
    label: str,
) -> tuple[Any, list[str]]:
    """Rehash and parse one raw JSON artifact without JSON ambiguities."""

    errors = _pinned_file_errors(
        root,
        artifact.relative_path,
        artifact.sha256,
        artifact.size_bytes,
        label,
    )
    if errors:
        return None, errors
    try:
        value = json.loads(
            (root / artifact.relative_path).read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, [f"{label} is not strict JSON: {exc}"]
    return value, []


def _validate_raw_artifacts(
    root: Path,
    *,
    receipt_relative_path: str,
    value: Any,
    evidence_name: str,
) -> tuple[list[RawArtifactBinding], list[str]]:
    errors: list[str] = []
    bindings: list[RawArtifactBinding] = []
    if not isinstance(value, list):
        return [], [f"{evidence_name} evidence raw_artifacts must be an array"]
    seen_types: set[str] = set()
    seen_paths: set[str] = set()
    for index, item in enumerate(value):
        label = f"{evidence_name} evidence raw_artifacts[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        missing = sorted(_RAW_ARTIFACT_FIELDS - set(item))
        unknown = sorted(set(item) - _RAW_ARTIFACT_FIELDS)
        if missing or unknown:
            errors.append(f"{label} fields differ: missing={missing}, unknown={unknown}")
            continue
        artifact_type = item.get("artifact_type")
        relative_path = item.get("relative_path")
        digest = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if not isinstance(artifact_type, str) or not artifact_type:
            errors.append(f"{label} artifact_type must be a non-empty string")
            continue
        if artifact_type in seen_types:
            errors.append(f"{label} duplicates artifact_type {artifact_type}")
        seen_types.add(artifact_type)
        if not _is_relative_path(relative_path):
            errors.append(f"{label} relative_path must be a normalized POSIX relative path")
            continue
        if relative_path == receipt_relative_path:
            errors.append(f"{label} cannot reuse the receipt as a raw artifact")
        if relative_path in seen_paths:
            errors.append(f"{label} duplicates raw artifact path {relative_path}")
        seen_paths.add(relative_path)
        if not _is_sha256(digest):
            errors.append(f"{label} sha256 must be a lowercase SHA-256 digest")
            continue
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
            errors.append(f"{label} size_bytes must be a positive integer")
            continue
        file_errors = _pinned_file_errors(
            root,
            relative_path,
            digest,
            size_bytes,
            f"{evidence_name} raw artifact {artifact_type}",
        )
        errors.extend(file_errors)
        if not file_errors:
            bindings.append(
                RawArtifactBinding(
                    artifact_type=artifact_type,
                    relative_path=relative_path,
                    sha256=digest,
                    size_bytes=size_bytes,
                )
            )
    expected_types = _RAW_ARTIFACT_TYPES[evidence_name]
    if seen_types != expected_types:
        errors.append(
            f"{evidence_name} evidence raw artifact types differ: "
            f"missing={sorted(expected_types - seen_types)}, "
            f"unknown={sorted(seen_types - expected_types)}"
        )
    return bindings, errors


def _pinned_file_errors(
    root: Path,
    relative_path: str,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> list[str]:
    root_path = Path(root)
    path = root_path / relative_path
    errors: list[str] = []
    try:
        resolved_root = root_path.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
    except (FileNotFoundError, OSError):
        return [f"{label} is missing: {path}"]
    except ValueError:
        return [f"{label} escapes evidence root: {relative_path}"]
    current = root_path
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            errors.append(f"{label} must not traverse a symlink: {current}")
            return errors
    if not resolved_path.is_file():
        return [f"{label} is missing: {path}"]
    try:
        content = resolved_path.read_bytes()
    except OSError as exc:
        return [f"{label} is unreadable: {exc}"]
    if expected_size is not None and len(content) != expected_size:
        errors.append(
            f"{label} size mismatch: expected {expected_size}, got {len(content)}"
        )
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_sha256:
        errors.append(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return errors


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None


def _is_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.as_posix() == value


__all__ = [
    "ALLOWED_EVIDENCE_PRODUCERS",
    "ALLOWED_SCRIPTED_SOLVABILITY_CONTROLLERS",
    "EVIDENCE_CHECKS",
    "EVIDENCE_RECEIPT_KIND",
    "EVIDENCE_RECEIPT_SCHEMA_VERSION",
    "EvidenceReceiptBinding",
    "EvidenceValidation",
    "FrozenControllerPin",
    "RawArtifactBinding",
    "execution_evidence_fingerprint",
    "validate_evidence_receipt",
]
