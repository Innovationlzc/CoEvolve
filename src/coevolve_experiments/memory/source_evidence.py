"""Manifest-pinned verifier evidence required before Table-4 export."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any

from .schema import semantic_sha256
from .source_audit import resolve_evidence_binding_pin


SOURCE_EVIDENCE_BINDING_KIND = "strategy_source_evidence_binding"
SOURCE_EVIDENCE_BINDING_SCHEMA_VERSION = 2

_BINDING_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "strategy_specs_sha256",
        "metrics_sha256",
        "records",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "source_record_sha256",
        "verified_outcome",
        "evidence_count",
        "evidence_receipts",
    }
)
_RECEIPT_REFERENCE_FIELDS = frozenset(
    {"relpath", "sha256", "host_evidence_sha256"}
)
_RECEIPT_DOCUMENT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "source_record_sha256",
        "verified_outcome",
        "rollout_ref",
        "rollout_key",
        "host_evidence",
    }
)
_ROLLOUT_REF_FIELDS = frozenset(
    {"script_id", "attempt", "checkpoint_before", "checkpoint_after", "outcome"}
)
_HOST_EVIDENCE_REF_FIELDS = frozenset(
    {"relative_path", "sha256", "size_bytes"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECEIPT_KIND = "strategy_source_verifier_receipt"
_RECEIPT_SCHEMA_VERSION = 2
_RECEIPT_DIRECTORY = PurePosixPath("memory/strategy_export_receipts")


@dataclass(frozen=True)
class BoundSourceEvidence:
    """Verifier-owned outcome and support count for one exact source spec."""

    outcome: str
    count: int


@dataclass(frozen=True)
class LoadedEvidenceReceipt:
    """One receipt document and the SHA-256 of its exact serialized bytes."""

    value: object
    sha256: str
    host_evidence_value: object | None = None
    host_evidence_sha256: str = ""
    host_evidence_size_bytes: int = -1
    host_evidence_relative_path: str = ""


@dataclass(frozen=True)
class SourceEvidenceValidation:
    """Validated binding records plus any fail-closed preflight errors."""

    pin_source: str
    records: Mapping[str, BoundSourceEvidence]
    errors: tuple[str, ...]


def validate_source_evidence_binding(
    *,
    source_index: int,
    binding: object | None,
    binding_sha256: str,
    strategy_specs: object,
    strategy_specs_sha256: str,
    metrics: object,
    metrics_sha256: str,
    manifest: object,
    audit: Mapping[str, Any],
    evidence_receipts: Mapping[str, LoadedEvidenceReceipt],
) -> SourceEvidenceValidation:
    """Validate one binding without consulting spec-embedded evidence.

    The return value carries errors instead of raising so the cohort validator
    can disclose every invalid source run in one preflight result.
    """

    prefix = f"source run {source_index}"
    required = "SOURCE_EVIDENCE_BINDING_REQUIRED"
    invalid = "SOURCE_EVIDENCE_BINDING_INVALID"
    if binding is None:
        return SourceEvidenceValidation(
            pin_source="",
            records={},
            errors=(
                f"{prefix}: {required}: missing "
                "memory/strategy_export_evidence.json",
            ),
        )
    if not isinstance(manifest, Mapping):
        return SourceEvidenceValidation(
            pin_source="",
            records={},
            errors=(f"{prefix}: {invalid}: source manifest must be an object",),
        )

    pin = resolve_evidence_binding_pin(manifest, audit)
    errors = [f"{prefix}: {invalid}: {message}" for message in pin.conflicts]
    if not pin.sha256:
        errors.append(
            f"{prefix}: {required}: binding SHA-256 must be pinned by "
            "run_manifest.result_artifact_hashes."
            "strategy_export_evidence_sha256 or filled by the frozen audit"
        )
    elif not _SHA256_RE.fullmatch(pin.sha256):
        errors.append(f"{prefix}: {invalid}: binding pin is not a SHA-256")
    elif binding_sha256 != pin.sha256:
        errors.append(
            f"{prefix}: {invalid}: evidence binding SHA-256 does not match "
            f"the {pin.source} pin"
        )

    if not isinstance(binding, Mapping):
        errors.append(f"{prefix}: {invalid}: binding root must be an object")
        return SourceEvidenceValidation(pin.source, {}, tuple(errors))
    if set(binding) != _BINDING_FIELDS:
        errors.append(
            f"{prefix}: {invalid}: binding fields must be exactly "
            f"{sorted(_BINDING_FIELDS)}"
        )
    if binding.get("kind") != SOURCE_EVIDENCE_BINDING_KIND:
        errors.append(
            f"{prefix}: {invalid}: kind must be "
            f"{SOURCE_EVIDENCE_BINDING_KIND!r}"
        )
    schema_version = binding.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SOURCE_EVIDENCE_BINDING_SCHEMA_VERSION
    ):
        errors.append(
            f"{prefix}: {invalid}: schema_version must be "
            f"{SOURCE_EVIDENCE_BINDING_SCHEMA_VERSION}"
        )
    if binding.get("strategy_specs_sha256") != strategy_specs_sha256:
        errors.append(
            f"{prefix}: {invalid}: strategy_specs_sha256 does not match the "
            "loaded strategy document"
        )
    if binding.get("metrics_sha256") != metrics_sha256:
        errors.append(
            f"{prefix}: {invalid}: metrics_sha256 does not match the loaded "
            "metrics document"
        )

    records = binding.get("records")
    if not isinstance(records, list):
        errors.append(f"{prefix}: {invalid}: records must be an array")
        return SourceEvidenceValidation(pin.source, {}, tuple(errors))
    specs = _extract_specs(strategy_specs)
    if specs is None:
        errors.append(
            f"{prefix}: {invalid}: strategy_specs must be a list or an object "
            "containing object specs"
        )
        return SourceEvidenceValidation(pin.source, {}, tuple(errors))
    spec_hashes = [semantic_sha256(spec) for spec in specs]
    if len(spec_hashes) != len(set(spec_hashes)):
        errors.append(
            f"{prefix}: {invalid}: source strategy records must have unique "
            "semantic SHA-256 identities"
        )
    spec_by_hash = dict(zip(spec_hashes, specs))
    known_specs = set(spec_by_hash)
    bound: dict[str, BoundSourceEvidence] = {}
    declared_counts = {"accepted": 0, "rejected": 0}
    seen_source_digests: set[str] = set()
    seen_receipt_paths: set[str] = set()
    seen_host_evidence: set[str] = set()
    for record_index, record in enumerate(records):
        record_prefix = f"{prefix}: {invalid}: records[{record_index}]"
        error_count_before = len(errors)
        if not isinstance(record, Mapping):
            errors.append(f"{record_prefix} must be an object")
            continue
        if set(record) != _RECORD_FIELDS:
            errors.append(
                f"{record_prefix} fields must be exactly "
                f"{sorted(_RECORD_FIELDS)}"
            )
            continue
        digest = record.get("source_record_sha256")
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            errors.append(f"{record_prefix}.source_record_sha256 is invalid")
            continue
        if digest not in known_specs:
            errors.append(
                f"{record_prefix}.source_record_sha256 does not identify a "
                "record in strategy_specs"
            )
        if digest in seen_source_digests:
            errors.append(
                f"{record_prefix}.source_record_sha256 duplicates an earlier binding"
            )
        seen_source_digests.add(digest)
        outcome = record.get("verified_outcome")
        if outcome not in {"accepted", "rejected"}:
            errors.append(
                f"{record_prefix}.verified_outcome must be accepted or rejected"
            )
            continue
        count = record.get("evidence_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            errors.append(
                f"{record_prefix}.evidence_count must be a positive integer"
            )
            continue
        declared_counts[outcome] += count
        receipt_refs = record.get("evidence_receipts")
        if not isinstance(receipt_refs, list):
            errors.append(f"{record_prefix}.evidence_receipts must be an array")
            continue
        if len(receipt_refs) != count:
            errors.append(
                f"{record_prefix}.evidence_count={count} must equal the number "
                f"of evidence_receipts={len(receipt_refs)}"
            )
        for receipt_index, receipt_ref in enumerate(receipt_refs):
            receipt_prefix = (
                f"{record_prefix}.evidence_receipts[{receipt_index}]"
            )
            if not isinstance(receipt_ref, Mapping):
                errors.append(f"{receipt_prefix} must be an object")
                continue
            if set(receipt_ref) != _RECEIPT_REFERENCE_FIELDS:
                errors.append(
                    f"{receipt_prefix} fields must be exactly "
                    f"{sorted(_RECEIPT_REFERENCE_FIELDS)}"
                )
                continue
            relpath = receipt_ref.get("relpath")
            receipt_sha256 = receipt_ref.get("sha256")
            host_sha256 = receipt_ref.get("host_evidence_sha256")
            if not _valid_receipt_relpath(relpath):
                errors.append(
                    f"{receipt_prefix}.relpath must name a file directly under "
                    f"{_RECEIPT_DIRECTORY.as_posix()}/"
                )
                continue
            assert isinstance(relpath, str)
            if relpath in seen_receipt_paths:
                errors.append(f"{receipt_prefix}.relpath is reused")
            seen_receipt_paths.add(relpath)
            if not isinstance(receipt_sha256, str) or not _SHA256_RE.fullmatch(
                receipt_sha256
            ):
                errors.append(f"{receipt_prefix}.sha256 is invalid")
                continue
            if not isinstance(host_sha256, str) or not _SHA256_RE.fullmatch(
                host_sha256
            ):
                errors.append(
                    f"{receipt_prefix}.host_evidence_sha256 is invalid"
                )
                continue
            if host_sha256 in seen_host_evidence:
                errors.append(
                    f"{receipt_prefix}.host_evidence_sha256 is reused"
                )
            seen_host_evidence.add(host_sha256)
            loaded = evidence_receipts.get(relpath)
            if loaded is None:
                errors.append(
                    f"{receipt_prefix}: SOURCE_EVIDENCE_RECEIPT_REQUIRED: "
                    "referenced receipt was not loaded"
                )
                continue
            if loaded.sha256 != receipt_sha256:
                errors.append(
                    f"{receipt_prefix}.sha256 does not match receipt bytes"
                )
            errors.extend(
                _receipt_errors(
                    loaded.value,
                    prefix=receipt_prefix,
                    source_record_sha256=digest,
                    outcome=outcome,
                    host_evidence_sha256=host_sha256,
                    source_spec=spec_by_hash.get(digest),
                    loaded=loaded,
                )
            )
        if len(errors) == error_count_before:
            bound[digest] = BoundSourceEvidence(outcome=outcome, count=count)
    limits, limit_errors = _metrics_evidence_limits(metrics)
    errors.extend(f"{prefix}: {invalid}: {message}" for message in limit_errors)
    for outcome in ("accepted", "rejected"):
        bound_count = declared_counts[outcome]
        limit = limits.get(outcome)
        if limit is not None and bound_count > limit:
            errors.append(
                f"{prefix}: SOURCE_EVIDENCE_COUNT_EXCEEDS_METRICS: "
                f"bound {outcome} evidence_count={bound_count} exceeds "
                f"hashed metrics {outcome}_physical={limit}"
            )
    return SourceEvidenceValidation(pin.source, bound, tuple(errors))


def _valid_receipt_relpath(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.parent == _RECEIPT_DIRECTORY
        and path.suffix == ".json"
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _receipt_errors(
    value: object,
    *,
    prefix: str,
    source_record_sha256: str,
    outcome: str,
    host_evidence_sha256: str,
    source_spec: Mapping[str, Any] | None,
    loaded: LoadedEvidenceReceipt,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}: receipt root must be an object"]
    errors: list[str] = []
    if set(value) != _RECEIPT_DOCUMENT_FIELDS:
        errors.append(
            f"{prefix}: receipt fields must be exactly "
            f"{sorted(_RECEIPT_DOCUMENT_FIELDS)}"
        )
    if value.get("kind") != _RECEIPT_KIND:
        errors.append(f"{prefix}: receipt kind must be {_RECEIPT_KIND!r}")
    version = value.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != _RECEIPT_SCHEMA_VERSION
    ):
        errors.append(
            f"{prefix}: receipt schema_version must be "
            f"{_RECEIPT_SCHEMA_VERSION}"
        )
    if value.get("source_record_sha256") != source_record_sha256:
        errors.append(f"{prefix}: receipt source_record_sha256 does not match")
    if value.get("verified_outcome") != outcome:
        errors.append(f"{prefix}: receipt verified_outcome does not match")
    rollout_ref = value.get("rollout_ref")
    if not isinstance(rollout_ref, Mapping) or set(rollout_ref) != _ROLLOUT_REF_FIELDS:
        errors.append(f"{prefix}: receipt rollout_ref is invalid")
        rollout_ref = {}
    rollout_key = value.get("rollout_key")
    if not isinstance(rollout_key, str) or not _SHA256_RE.fullmatch(rollout_key):
        errors.append(f"{prefix}: receipt rollout_key is invalid")
        rollout_key = ""
    host_ref = value.get("host_evidence")
    if not isinstance(host_ref, Mapping) or set(host_ref) != _HOST_EVIDENCE_REF_FIELDS:
        errors.append(f"{prefix}: receipt host_evidence reference is invalid")
        host_ref = {}
    if host_ref.get("sha256") != host_evidence_sha256:
        errors.append(f"{prefix}: receipt host-evidence SHA-256 differs from binding")
    if (
        host_ref.get("relative_path") != loaded.host_evidence_relative_path
        or host_ref.get("sha256") != loaded.host_evidence_sha256
        or host_ref.get("size_bytes") != loaded.host_evidence_size_bytes
    ):
        errors.append(f"{prefix}: receipt host-evidence reference differs from loaded bytes")
    errors.extend(
        _raw_rollout_errors(
            loaded.host_evidence_value,
            prefix=prefix,
            rollout_ref=rollout_ref,
            rollout_key=rollout_key,
            outcome=outcome,
            source_spec=source_spec,
        )
    )
    return errors


def _raw_rollout_errors(
    value: object,
    *,
    prefix: str,
    rollout_ref: Mapping[str, Any],
    rollout_key: str,
    outcome: str,
    source_spec: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"{prefix}: raw Host evaluation is missing or not an object"]
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append(f"{prefix}: raw Host evaluation lacks verifier metrics")
        metrics = {}
    success = value.get("success")
    failure_type = str(value.get("failure_type") or "").strip()
    expected_outcome = "accepted" if outcome == "accepted" else failure_type
    if outcome == "accepted":
        if success is not True or failure_type not in {"", "none"}:
            errors.append(f"{prefix}: accepted receipt contradicts raw Host evaluation")
    elif success is not False or not failure_type or failure_type == "none":
        errors.append(f"{prefix}: rejected receipt contradicts raw Host evaluation")
    if rollout_ref.get("outcome") != expected_outcome:
        errors.append(f"{prefix}: rollout_ref outcome contradicts raw Host evaluation")
    attempt = rollout_ref.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        errors.append(f"{prefix}: rollout_ref attempt must be non-negative integer")
    for field in ("script_id", "checkpoint_before", "checkpoint_after"):
        if not isinstance(rollout_ref.get(field), str):
            errors.append(f"{prefix}: rollout_ref {field} must be a string")
    recomputed = _rollout_evidence_key(
        script_id=str(rollout_ref.get("script_id") or ""),
        attempt=attempt,
        checkpoint_before=str(rollout_ref.get("checkpoint_before") or ""),
        checkpoint_after=str(rollout_ref.get("checkpoint_after") or ""),
        outcome=expected_outcome,
        metrics=metrics,
    )
    if rollout_key != recomputed:
        errors.append(f"{prefix}: rollout_key does not match raw verifier metrics")
    source_evidence = source_spec.get("evidence") if isinstance(source_spec, Mapping) else None
    if not isinstance(source_evidence, Mapping):
        errors.append(f"{prefix}: source strategy has no rollout evidence block")
        return errors
    refs = source_evidence.get("rollout_refs")
    keys = source_evidence.get("rollout_keys")
    if not isinstance(refs, list) or dict(rollout_ref) not in refs:
        errors.append(f"{prefix}: receipt rollout_ref is absent from source strategy")
    if not isinstance(keys, list) or rollout_key not in keys:
        errors.append(f"{prefix}: receipt rollout_key is absent from source strategy")
    return errors


def _rollout_evidence_key(
    *,
    script_id: str,
    attempt: object,
    checkpoint_before: str,
    checkpoint_after: str,
    outcome: str,
    metrics: Mapping[str, Any],
) -> str:
    fingerprint = {
        "script_id": script_id,
        "attempt": attempt,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "outcome": outcome,
        "metrics": dict(metrics),
    }
    return hashlib.sha256(
        json.dumps(
            fingerprint,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _metrics_evidence_limits(
    value: object,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Read strict verifier totals from the already hash-bound metrics object."""

    if not isinstance(value, Mapping):
        return {}, ("metrics root must be an object",)
    physical = value.get("physical_accounting")
    if not isinstance(physical, Mapping):
        return {}, ("metrics.physical_accounting must be an object",)
    limits: dict[str, int] = {}
    errors: list[str] = []
    for outcome in ("accepted", "rejected"):
        physical_value = physical.get(f"{outcome}_physical")
        if physical_value is None:
            errors.append(
                f"hashed metrics do not expose "
                f"physical_accounting.{outcome}_physical"
            )
            continue
        if (
            isinstance(physical_value, bool)
            or not isinstance(physical_value, int)
            or physical_value < 0
        ):
            errors.append(
                f"hashed metrics physical_accounting.{outcome}_physical is "
                "not a non-negative integer"
            )
            continue
        step_value = value.get(f"{outcome}_step_count")
        if step_value is not None:
            if (
                isinstance(step_value, bool)
                or not isinstance(step_value, int)
                or step_value < 0
            ):
                errors.append(
                    f"hashed metrics {outcome}_step_count is not a "
                    "non-negative integer"
                )
                continue
            if step_value != physical_value:
                errors.append(f"hashed metrics {outcome} counters disagree")
                continue
        limits[outcome] = physical_value
    return limits, tuple(errors)


def _extract_specs(value: object) -> list[dict[str, Any]] | None:
    raw = (
        value.get("specs", value.get("records"))
        if isinstance(value, Mapping)
        else value
    )
    if not isinstance(raw, list) or not all(
        isinstance(item, Mapping) for item in raw
    ):
        return None
    return [dict(item) for item in raw]


__all__ = [
    "BoundSourceEvidence",
    "LoadedEvidenceReceipt",
    "SOURCE_EVIDENCE_BINDING_KIND",
    "SOURCE_EVIDENCE_BINDING_SCHEMA_VERSION",
    "SourceEvidenceValidation",
    "validate_source_evidence_binding",
]
