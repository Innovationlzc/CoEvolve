"""Immutable result/evidence publication for the production Praxis runner.

The frozen V83 loop is responsible for producing physical observations and
per-call backend telemetry.  This module owns the reporting boundary around
those observations:

* launch and result identities are copied only from the preflight injection
  bundle, never from a runner-authored manifest;
* FM totals are derived from the individual successful response records;
* terminal evidence is validated against its raw MuJoCo sample trace before
  any result file is published;
* every file is committed atomically without replacing an existing path, and
  ``run_manifest.json`` is committed last as the publication marker; and
* the existing terminal-evidence and usage-evidence validators are run again
  against the durable files immediately after publication.

The writer deliberately does not execute MuJoCo, call a model, or modify any
V83 behavior-bearing file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..live.bridge import LIVE_RESULT_BINDING_KIND
from ..live.capability import RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
from ..reporting.evidence_artifacts import (
    BoundArtifactError,
    BoundJsonArtifact,
    load_bound_json_artifact,
)
from ..reporting.terminal_evidence import (
    PAYLOAD_DROP_LIMIT,
    SYSTEM_STALL_GENERATION_CYCLES,
    TERMINAL_EVIDENCE_FILENAME,
    TERMINAL_EVIDENCE_KIND,
    TERMINAL_EVIDENCE_SCHEMA_VERSION,
    TerminalEvidenceValidation,
    validate_terminal_evidence,
)
from ..reporting.terminal_aux_evidence import (
    AuxiliaryEvidenceDerivation,
    capability_bound_auxiliary_producer_allowlist,
    recompute_payload_drop_ledger,
    recompute_system_stall_ledger,
)
from ..reporting.transaction_accounting import (
    ACCEPTED_MANIFEST_PIN,
    ACCEPTED_MANIFEST_RELPATH,
    FROZEN_PHASE_PLAN_KIND,
    FROZEN_PHASE_PLAN_PIN,
    FROZEN_PHASE_PLAN_RELPATH,
    FROZEN_PHASE_PLAN_SCHEMA_VERSION,
    TRANSACTION_ACCOUNTING_LEDGER_PIN,
    TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
    TransactionAccountingDerivation,
    TransactionAccountingEvidenceValidation,
    recompute_transaction_accounting_ledger,
    validate_transaction_accounting_evidence,
)
from ..reporting.usage_evidence import (
    FM_USAGE_LEDGER_FILENAME,
    FM_USAGE_LEDGER_KIND,
    FM_USAGE_LEDGER_SCHEMA_VERSION,
    UsageEvidenceValidation,
    validate_fm_usage_evidence,
)
from ..serialization import sha256_json, strict_json_loads, to_primitive
from .contracts import (
    ProductionChannelBinding,
    ProductionInjectionBundle,
    REQUIRED_PRODUCTION_CHANNELS,
)


RUN_PLAN_FILENAME = "run_plan.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
METRICS_FILENAME = "metrics.json"
TERMINAL_SAMPLE_TRACE_RELPATH = "evidence/terminal_sample_trace.json"
TERMINAL_STATE_RELPATH = "evidence/terminal_state.json"
SYSTEM_STALL_LEDGER_RELPATH = "evidence/system_stall_ledger.json"
PAYLOAD_DROP_LEDGER_RELPATH = "evidence/payload_drop_ledger.json"
V83_NATIVE_MEMORY_AUDIT_RELPATH = (
    "memory/v83_native/v83_native_memory_runtime_audit.json"
)
V83_NATIVE_MEMORY_AUDIT_PIN = "v83_native_memory_runtime_audit_sha256"

_FROM_SCRATCH_BOOTSTRAP_RELPATH = "memory/from_scratch_memory_bootstrap.json"
_FROM_SCRATCH_RUNTIME_RELPATH = "memory/from_scratch_memory_runtime_audit.json"
_FROM_SCRATCH_BOOTSTRAP_PIN = "from_scratch_memory_bootstrap"
_FROM_SCRATCH_RUNTIME_PIN = "from_scratch_memory_runtime_audit"
_TERMINAL_PIN = "terminal_verifier_evidence_sha256"
_USAGE_PIN = "fm_usage_ledger_sha256"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VALID_TERMINALS = frozenset(
    {
        "SUCCESS",
        "SYSTEM_STALL",
        "PHYSICAL_BUDGET_EXHAUSTED",
        "CONSECUTIVE_PAYLOAD_DROP_LIMIT",
    }
)
_RESULT_BINDING_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "run_id",
        "run_spec_sha256",
        "launch_fingerprint",
        "config_fingerprint",
        "launch_config_sha256",
        "plan_sha256",
        "runner_capability_sha256",
        "base_method_commit",
        "runner_integration_commit",
        "entrypoint_sha256",
        "full_method_compatibility_fingerprint",
    }
)
_RUN_SPEC_FIELDS = frozenset(
    {
        "run_id",
        "table_id",
        "seed",
        "method_variant",
        "scene_id",
        "memory_condition",
        "memory_policy",
        "target_backbone",
        "provider_route_id",
        "provider_route_fingerprint",
        "physical_budget",
        "condition_fingerprint",
        "invariant_fingerprint",
        "base_method_commit",
        "protocol_id",
        "output_relpath",
        "scene_fingerprint",
        "terminal_fingerprint",
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
        "strategy_bank",
    }
)
_FM_CALL_INPUT_REQUIRED_FIELDS = frozenset(
    {
        "request_id",
        "role",
        "modality",
        "requested_model",
        "resolved_model",
        "response_model",
        "http_status",
        "input_tokens",
        "output_tokens",
    }
)
_FM_CALL_INPUT_OPTIONAL_FIELDS = frozenset(
    {"billing_amount_usd", "outcome", "transport_attempts"}
)
_FM_CALL_INPUT_FIELDS = (
    _FM_CALL_INPUT_REQUIRED_FIELDS | _FM_CALL_INPUT_OPTIONAL_FIELDS
)
_FM_ATTEMPT_INPUT_FIELDS = frozenset(
    {"outcome", "http_status", "error_type"}
)
_RESERVED_PINS = frozenset(
    {_TERMINAL_PIN, _USAGE_PIN, TRANSACTION_ACCOUNTING_LEDGER_PIN}
)


class ProductionEvidenceWriteError(ValueError):
    """The proposed evidence is invalid or cannot be published safely."""


@dataclass(frozen=True, slots=True)
class EvidenceProducer:
    """Hash-pinned implementation identity for one evidence producer."""

    id: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class ProductionArtifactRef:
    """An existing immutable raw artifact consumed by terminal finalization."""

    relative_path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class FoundationModelTransportAttempt:
    """One HTTP response or lower-level transport failure for a logical call."""

    outcome: str
    http_status: int | None = None
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class FoundationModelCall:
    """One logical FM request; ordinals and aggregate totals are writer-owned.

    ``transport_attempts`` may contain multiple provider retries. For backward
    compatibility, an empty tuple is expanded from ``http_status`` into one
    HTTP attempt. ``billing_amount_usd`` is the total provider-reported cost of
    this logical request across its attempts; ``None`` means unavailable, not
    zero.
    """

    request_id: str
    role: str
    modality: str
    requested_model: str
    resolved_model: str | None
    response_model: str | None
    http_status: int | None
    input_tokens: int | None
    output_tokens: int | None
    billing_amount_usd: float | None = None
    outcome: str | None = None
    transport_attempts: tuple[FoundationModelTransportAttempt, ...] = ()


@dataclass(frozen=True, slots=True)
class ProductionRunAccounting:
    """Redundant Term4 claim plus the independently measured wall-clock time.

    Physical counts in this record are never authoritative.  The writer sends
    them to the transaction-ledger derivation only as a consistency check and
    publishes counts reconstructed from immutable physical receipts.
    """

    physical_transactions: int
    accepted_transactions: int
    rejected_transactions: int
    accepted_setup_transactions: int
    accepted_progress_transactions: int
    elapsed_s: float
    phase_completed: int
    phase_total: int
    failure_counts: Mapping[str, int] = field(default_factory=dict)
    mutation_counts: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransactionAccountingEvidence:
    """Hash-bound raw inputs needed to reconstruct all paper accounting."""

    transaction_receipts: tuple[ProductionArtifactRef, ...]
    frozen_phase_plan: ProductionArtifactRef
    accepted_episode_manifest: ProductionArtifactRef
    mutation_commit_receipts: tuple[ProductionArtifactRef, ...] = ()


@dataclass(frozen=True, slots=True)
class SystemStallCycleEvidence:
    """References to one already-published, no-dispatch recovery cycle."""

    model_output: ProductionArtifactRef
    static_validation: ProductionArtifactRef
    host_preflight: ProductionArtifactRef


@dataclass(frozen=True, slots=True)
class SystemStallEvidence:
    """References only; the writer recomputes all SYSTEM_STALL facts."""

    checkpoint_validity: ProductionArtifactRef
    host_action_space: ProductionArtifactRef
    generation_cycles: tuple[SystemStallCycleEvidence, ...]


@dataclass(frozen=True, slots=True)
class PayloadDropEvidence:
    """Five existing transaction receipts; no caller-authored drop facts."""

    transaction_receipts: tuple[ProductionArtifactRef, ...]


@dataclass(frozen=True, slots=True)
class ManifestArtifactPin:
    """A manifest pin whose digest is always computed by the writer."""

    name: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class ProductionEvidenceWriteResult:
    """Durable result paths and post-publication validations."""

    run_root: Path
    run_manifest_path: Path
    metrics_path: Path
    terminal_evidence_path: Path
    fm_usage_ledger_path: Path
    transaction_accounting_ledger_path: Path
    result_artifact_hashes: Mapping[str, str]
    usage_totals: Mapping[str, Any]
    terminal_validation: TerminalEvidenceValidation
    usage_validation: UsageEvidenceValidation
    accounting_validation: TransactionAccountingEvidenceValidation


def write_immutable_raw_evidence_artifact(
    run_root: str | Path,
    relative_path: str,
    payload: Mapping[str, Any],
) -> ProductionArtifactRef:
    """Exclusively publish one runtime-produced raw evidence artifact.

    This helper is intentionally separate from terminal finalization.  A
    producer calls it at the transaction/checkpoint stage; the terminal writer
    later accepts only the returned path/hash/size reference and refuses to
    manufacture missing facts.
    """

    root = _validated_output_root(Path(run_root))
    if not isinstance(relative_path, str) or not (
        relative_path.startswith("evidence/transactions/")
        or relative_path.startswith("evidence/system_stall/")
        or relative_path.startswith("evidence/accounting/")
        or relative_path.startswith("evidence/mutations/")
    ):
        raise ProductionEvidenceWriteError(
            "raw evidence must live below a reviewed evidence namespace"
        )
    if relative_path in {
        SYSTEM_STALL_LEDGER_RELPATH,
        PAYLOAD_DROP_LEDGER_RELPATH,
        TERMINAL_SAMPLE_TRACE_RELPATH,
        TERMINAL_STATE_RELPATH,
        TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
    }:
        raise ProductionEvidenceWriteError(
            "raw evidence path is reserved for terminal finalization"
        )
    raw = _json_bytes(payload, "raw production evidence")
    path = _fixed_output_path(root, relative_path)
    if path.exists() or path.is_symlink():
        raise ProductionEvidenceWriteError(
            f"refusing to overwrite existing raw evidence artifact: {path}"
        )
    _atomic_create_exclusive(path, raw)
    reference = _artifact_ref(relative_path, raw)
    try:
        loaded = load_bound_json_artifact(root, reference, label="raw production evidence")
    except BoundArtifactError as exc:
        raise ProductionEvidenceWriteError(
            "immutable raw evidence failed durable revalidation"
        ) from exc
    return ProductionArtifactRef(
        relative_path=loaded.relative_path,
        sha256=loaded.sha256,
        size_bytes=loaded.size_bytes,
    )


def write_frozen_phase_plan_artifact(
    run_root: str | Path,
    *,
    run_id: str,
    phase_ids: Sequence[str],
) -> ProductionArtifactRef:
    """Freeze the Host-validated initial persistent plan before dispatch.

    A production Term4 worker must call this exactly once after the initial
    Task Orchestrator plan passes schema/Host validation and before the first
    physical transaction.  Exclusive creation prevents a terminal collector
    from replacing the plan after observing the outcome.
    """

    if not isinstance(run_id, str) or not run_id.strip():
        raise ProductionEvidenceWriteError("phase-plan run_id must be non-empty")
    if isinstance(phase_ids, (str, bytes)):
        raise ProductionEvidenceWriteError("phase_ids must be a sequence")
    try:
        frozen = tuple(phase_ids)
    except TypeError as exc:
        raise ProductionEvidenceWriteError("phase_ids must be a sequence") from exc
    if (
        not frozen
        or any(not isinstance(item, str) or not item.strip() for item in frozen)
        or len(set(frozen)) != len(frozen)
    ):
        raise ProductionEvidenceWriteError(
            "phase_ids must be unique non-empty strings"
        )
    return write_immutable_raw_evidence_artifact(
        run_root,
        FROZEN_PHASE_PLAN_RELPATH,
        {
            "kind": FROZEN_PHASE_PLAN_KIND,
            "schema_version": FROZEN_PHASE_PLAN_SCHEMA_VERSION,
            "run_id": run_id,
            "phase_ids": list(frozen),
        },
    )


def write_production_result_evidence(
    bundle: ProductionInjectionBundle,
    *,
    terminal_status: str,
    terminal_sample_trace: Mapping[str, Any],
    terminal_state: Mapping[str, Any],
    accounting: ProductionRunAccounting,
    transaction_accounting_evidence: TransactionAccountingEvidence,
    fm_calls: Sequence[FoundationModelCall | Mapping[str, Any]],
    terminal_producer: EvidenceProducer,
    fm_producer: EvidenceProducer,
    system_stall: SystemStallEvidence | None = None,
    payload_drop: PayloadDropEvidence | None = None,
    additional_artifact_pins: Sequence[ManifestArtifactPin] = (),
) -> ProductionEvidenceWriteResult:
    """Validate and immutably publish one complete production result.

    ``run_manifest.json`` is the final file committed.  A failed preflight or
    schema validation therefore publishes no apparent result.  The output
    directory may already contain the immutable memory bootstrap and finalized
    memory audit, but none of this function's destination files may exist.
    """

    channels = _validated_channels(bundle)
    identity = _extract_identity(bundle, channels)
    if isinstance(additional_artifact_pins, (str, bytes)):
        raise ProductionEvidenceWriteError("additional artifact pins must be a sequence")
    try:
        frozen_additional_pins = tuple(additional_artifact_pins)
    except TypeError as exc:
        raise ProductionEvidenceWriteError(
            "additional artifact pins must be a sequence"
        ) from exc
    if isinstance(fm_calls, (str, bytes)):
        raise ProductionEvidenceWriteError("FM calls must be a sequence")
    try:
        frozen_fm_calls = tuple(fm_calls)
    except TypeError as exc:
        raise ProductionEvidenceWriteError("FM calls must be a sequence") from exc
    _validate_producer(terminal_producer, "terminal producer")
    _validate_producer(fm_producer, "FM producer")
    if terminal_status not in _VALID_TERMINALS:
        raise ProductionEvidenceWriteError(
            f"unsupported terminal_status: {terminal_status!r}"
        )

    output_root = _validated_output_root(identity.output_dir)
    claimed_accounting = _claimed_accounting(accounting)
    accounting_derivation = _derive_transaction_accounting(
        output_root,
        identity=identity,
        evidence=transaction_accounting_evidence,
        claimed_accounting=claimed_accounting,
    )
    if not accounting_derivation.valid or accounting_derivation.ledger is None:
        raise ProductionEvidenceWriteError(
            "transaction accounting evidence is invalid: "
            + ", ".join(accounting_derivation.invalid_reasons)
        )
    accounting = _authoritative_accounting(
        accounting_derivation.ledger,
        elapsed_s=accounting.elapsed_s,
    )
    _validate_accounting(accounting, terminal_status, identity.physical_budget)
    artifact_pins = _hash_existing_manifest_artifacts(
        output_root,
        table_id=identity.run["table_id"],
        required_pins=identity.required_result_pins,
        additional=frozen_additional_pins,
    )
    call_documents, totals = _build_fm_calls(frozen_fm_calls, identity.target_model)

    documents: dict[str, bytes] = {
        TRANSACTION_ACCOUNTING_LEDGER_RELPATH: _json_bytes(
            accounting_derivation.ledger,
            "transaction accounting ledger",
        )
    }
    documents[TERMINAL_SAMPLE_TRACE_RELPATH] = _json_bytes(
        terminal_sample_trace,
        "terminal sample trace",
    )
    documents[TERMINAL_STATE_RELPATH] = _json_bytes(
        terminal_state,
        "terminal state",
    )
    terminal_state_ref = _artifact_ref(
        TERMINAL_STATE_RELPATH,
        documents[TERMINAL_STATE_RELPATH],
    )
    terminal_trace_ref = _artifact_ref(
        TERMINAL_SAMPLE_TRACE_RELPATH,
        documents[TERMINAL_SAMPLE_TRACE_RELPATH],
    )

    stall_ref: Mapping[str, Any] | None = None
    drop_ref: Mapping[str, Any] | None = None
    raw_source_artifacts: list[BoundJsonArtifact] = list(
        accounting_derivation.source_artifacts
    )
    if terminal_status == "SYSTEM_STALL":
        if system_stall is None or payload_drop is not None:
            raise ProductionEvidenceWriteError(
                "SYSTEM_STALL requires only system_stall evidence"
            )
        stall_derivation = _build_system_stall_document(
            output_root,
            run_id=identity.run_id,
            evidence=system_stall,
            calls=call_documents,
            terminal_state=terminal_state,
            terminal_state_sha256=str(terminal_state_ref["sha256"]),
            terminal_sample_trace=terminal_sample_trace,
            terminal_sample_trace_sha256=str(terminal_trace_ref["sha256"]),
            capability_entrypoint_sha256=str(
                identity.result_binding["entrypoint_sha256"]
            ),
        )
        assert stall_derivation.ledger is not None
        stall_document = stall_derivation.ledger
        raw_source_artifacts.extend(stall_derivation.source_artifacts)
        documents[SYSTEM_STALL_LEDGER_RELPATH] = _json_bytes(
            stall_document,
            "system stall ledger",
        )
        stall_ref = _artifact_ref(
            SYSTEM_STALL_LEDGER_RELPATH,
            documents[SYSTEM_STALL_LEDGER_RELPATH],
        )
    elif system_stall is not None:
        raise ProductionEvidenceWriteError(
            "system_stall evidence is allowed only for SYSTEM_STALL"
        )

    if terminal_status == "CONSECUTIVE_PAYLOAD_DROP_LIMIT":
        if payload_drop is None or system_stall is not None:
            raise ProductionEvidenceWriteError(
                "CONSECUTIVE_PAYLOAD_DROP_LIMIT requires only payload_drop evidence"
            )
        if (
            accounting.rejected_transactions < PAYLOAD_DROP_LIMIT
            or accounting.failure_counts.get("payload_drop", 0) < PAYLOAD_DROP_LIMIT
        ):
            raise ProductionEvidenceWriteError(
                "payload-drop terminal does not reconcile with rejected/failure accounting"
            )
        drop_derivation = _build_payload_drop_document(
            output_root,
            run_id=identity.run_id,
            physical_transactions=accounting.physical_transactions,
            terminal_sample_trace=terminal_sample_trace,
            evidence=payload_drop,
            capability_entrypoint_sha256=str(
                identity.result_binding["entrypoint_sha256"]
            ),
            expected_compiled_model_sha256=identity.compiled_model_sha256,
            expected_scene_realization_sha256=identity.scene_realization_sha256,
        )
        assert drop_derivation.ledger is not None
        drop_document = drop_derivation.ledger
        raw_source_artifacts.extend(drop_derivation.source_artifacts)
        documents[PAYLOAD_DROP_LEDGER_RELPATH] = _json_bytes(
            drop_document,
            "payload drop ledger",
        )
        drop_ref = _artifact_ref(
            PAYLOAD_DROP_LEDGER_RELPATH,
            documents[PAYLOAD_DROP_LEDGER_RELPATH],
        )
    elif payload_drop is not None:
        raise ProductionEvidenceWriteError(
            "payload_drop evidence is allowed only for the drop-limit terminal"
        )

    terminal_receipt = {
        "kind": TERMINAL_EVIDENCE_KIND,
        "schema_version": TERMINAL_EVIDENCE_SCHEMA_VERSION,
        "producer": _producer_document(terminal_producer),
        "run_binding": {
            "run_id": identity.run_id,
            "seed": identity.seed,
            "condition_fingerprint": identity.condition_fingerprint,
            "terminal_fingerprint": identity.terminal_fingerprint,
            "scene_fingerprint": identity.scene_fingerprint,
            "scene_execution_evidence_fingerprint": (
                identity.scene_execution_evidence_fingerprint
            ),
        },
        "terminal_status": terminal_status,
        "physical_transactions": accounting.physical_transactions,
        "terminal_state": terminal_state_ref,
        "terminal_sample_trace": terminal_trace_ref,
        "system_stall": stall_ref,
        "payload_drop": drop_ref,
    }
    documents[TERMINAL_EVIDENCE_FILENAME] = _json_bytes(
        terminal_receipt,
        "terminal evidence receipt",
    )

    usage_ledger = {
        "kind": FM_USAGE_LEDGER_KIND,
        "schema_version": FM_USAGE_LEDGER_SCHEMA_VERSION,
        "producer": _producer_document(fm_producer),
        "run_binding": {
            "run_id": identity.run_id,
            "seed": identity.seed,
            "condition_fingerprint": identity.condition_fingerprint,
            "provider_route_id": identity.provider_route_id,
            "provider_route_fingerprint": identity.provider_route_fingerprint,
            "target_model": identity.target_model,
        },
        "calls": call_documents,
        "totals": totals,
    }
    documents[FM_USAGE_LEDGER_FILENAME] = _json_bytes(
        usage_ledger,
        "FM usage ledger",
    )
    artifact_pins[_TERMINAL_PIN] = hashlib.sha256(
        documents[TERMINAL_EVIDENCE_FILENAME]
    ).hexdigest()
    artifact_pins[_USAGE_PIN] = hashlib.sha256(
        documents[FM_USAGE_LEDGER_FILENAME]
    ).hexdigest()
    artifact_pins[TRANSACTION_ACCOUNTING_LEDGER_PIN] = hashlib.sha256(
        documents[TRANSACTION_ACCOUNTING_LEDGER_RELPATH]
    ).hexdigest()

    metrics = _build_metrics(
        identity,
        terminal_status=terminal_status,
        accounting=accounting,
        totals=totals,
    )
    manifest = _build_manifest(
        identity,
        terminal_status=terminal_status,
        result_artifact_hashes=artifact_pins,
    )
    documents[RUN_PLAN_FILENAME] = _json_bytes(identity.run, "run plan")
    documents[METRICS_FILENAME] = _json_bytes(metrics, "metrics")
    documents[RUN_MANIFEST_FILENAME] = _json_bytes(manifest, "run manifest")

    raw_source_documents = _snapshot_source_artifacts(raw_source_artifacts)
    _validate_staged_documents(
        documents,
        raw_source_documents=raw_source_documents,
        manifest=manifest,
        identity=identity,
        terminal_status=terminal_status,
        accounting=accounting,
        totals=totals,
    )
    _require_all_destinations_absent(output_root, documents)
    _recheck_source_artifacts(output_root, raw_source_artifacts)
    _recheck_manifest_artifacts(output_root, artifact_pins, frozen_additional_pins)

    # A missing manifest cannot be mistaken for a complete result.  Commit all
    # dependencies first and the manifest last.
    order = [name for name in documents if name != RUN_MANIFEST_FILENAME]
    order.append(RUN_MANIFEST_FILENAME)
    for relative_path in order:
        _atomic_create_exclusive(
            _fixed_output_path(output_root, relative_path),
            documents[relative_path],
        )

    terminal_validation, usage_validation = _run_existing_validators(
        output_root,
        manifest=manifest,
        identity=identity,
        terminal_status=terminal_status,
        accounting=accounting,
        totals=totals,
    )
    if not terminal_validation.valid or not usage_validation.valid:
        details = (*terminal_validation.invalid_reasons, *usage_validation.invalid_reasons)
        raise ProductionEvidenceWriteError(
            "durable evidence failed post-publication validation: "
            + ", ".join(details)
        )
    accounting_validation = _run_accounting_validator(
        output_root,
        identity=identity,
        accounting=accounting,
    )
    if not accounting_validation.valid:
        raise ProductionEvidenceWriteError(
            "durable accounting evidence failed post-publication validation: "
            + ", ".join(accounting_validation.invalid_reasons)
        )
    _require_durable_json_matches(
        output_root / RUN_MANIFEST_FILENAME,
        manifest,
        "run manifest",
    )
    _require_durable_json_matches(
        output_root / METRICS_FILENAME,
        metrics,
        "metrics",
    )
    _require_durable_json_matches(
        output_root / TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
        accounting_derivation.ledger,
        "transaction accounting ledger",
    )
    _recheck_manifest_artifacts(output_root, artifact_pins, frozen_additional_pins)
    _recheck_source_artifacts(output_root, raw_source_artifacts)
    return ProductionEvidenceWriteResult(
        run_root=output_root,
        run_manifest_path=output_root / RUN_MANIFEST_FILENAME,
        metrics_path=output_root / METRICS_FILENAME,
        terminal_evidence_path=output_root / TERMINAL_EVIDENCE_FILENAME,
        fm_usage_ledger_path=output_root / FM_USAGE_LEDGER_FILENAME,
        transaction_accounting_ledger_path=(
            output_root / TRANSACTION_ACCOUNTING_LEDGER_RELPATH
        ),
        result_artifact_hashes=MappingProxyType(dict(artifact_pins)),
        usage_totals=MappingProxyType(dict(totals)),
        terminal_validation=terminal_validation,
        usage_validation=usage_validation,
        accounting_validation=accounting_validation,
    )


@dataclass(frozen=True, slots=True)
class _Identity:
    output_dir: Path
    run: Mapping[str, Any]
    result_binding: Mapping[str, Any]
    run_id: str
    seed: int
    physical_budget: int
    condition_fingerprint: str
    terminal_fingerprint: str
    scene_fingerprint: str
    scene_realization_sha256: str
    compiled_model_sha256: str
    scene_execution_evidence_fingerprint: str
    target_center_xyz_m: tuple[float, float, float]
    provider: str
    provider_route_id: str
    provider_route_fingerprint: str
    target_model: str
    required_result_pins: frozenset[str]


def _validated_channels(
    bundle: ProductionInjectionBundle,
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(bundle, ProductionInjectionBundle):
        raise ProductionEvidenceWriteError(
            "bundle must be a ProductionInjectionBundle from production preflight"
        )
    if tuple(item.name for item in bundle.channels) != REQUIRED_PRODUCTION_CHANNELS:
        raise ProductionEvidenceWriteError("production injection channel set differs")
    for item in bundle.channels:
        rebuilt = ProductionChannelBinding.create(item.name, item.payload)
        if rebuilt.fingerprint != item.fingerprint:
            raise ProductionEvidenceWriteError(
                f"production channel fingerprint differs: {item.name}"
            )
    rebuilt_bundle = ProductionInjectionBundle.create(
        run_id=bundle.run_id,
        launch_config_path=bundle.launch_config_path,
        launch_config_sha256=bundle.launch_config_sha256,
        launch_fingerprint=bundle.launch_fingerprint,
        channels=bundle.channels,
    )
    if rebuilt_bundle.fingerprint != bundle.fingerprint:
        raise ProductionEvidenceWriteError("production injection bundle fingerprint differs")
    return MappingProxyType(
        {item.name: dict(item.payload) for item in bundle.channels}
    )


def _extract_identity(
    bundle: ProductionInjectionBundle,
    channels: Mapping[str, Mapping[str, Any]],
) -> _Identity:
    formal_manifest = _mapping(channels["formal_run_manifest"], "formal manifest")
    run = _mapping(formal_manifest.get("run_spec"), "formal run_spec")
    binding = _mapping(formal_manifest.get("result_binding"), "result binding")
    output = _mapping(channels["formal_output"], "formal output")
    provider = _mapping(channels["provider_route"], "provider route")
    simulator = _mapping(channels["term1_simulator_asset"], "simulator asset")

    if set(run) != _RUN_SPEC_FIELDS:
        raise ProductionEvidenceWriteError("formal RunSpec fields differ")
    if set(binding) != _RESULT_BINDING_FIELDS:
        raise ProductionEvidenceWriteError("result binding does not use schema v4 fields")
    if (
        binding.get("kind") != LIVE_RESULT_BINDING_KIND
        or binding.get("schema_version") != RUNNER_RESULT_CONTRACT_SCHEMA_VERSION
    ):
        raise ProductionEvidenceWriteError("result binding identity is invalid")
    if binding.get("run_spec_sha256") != sha256_json(run):
        raise ProductionEvidenceWriteError("result binding run_spec hash differs")
    if binding.get("run_id") != bundle.run_id or run.get("run_id") != bundle.run_id:
        raise ProductionEvidenceWriteError("bundle/run/result run_id differs")
    if binding.get("launch_fingerprint") != bundle.launch_fingerprint:
        raise ProductionEvidenceWriteError("result binding launch fingerprint differs")
    _sha(bundle.launch_config_sha256, "bundle launch-config hash")
    _sha(bundle.launch_fingerprint, "bundle launch fingerprint")
    if formal_manifest.get("launch_config_sha256") != bundle.launch_config_sha256:
        raise ProductionEvidenceWriteError("formal manifest launch-config hash differs")
    if binding.get("launch_config_sha256") != bundle.launch_config_sha256:
        raise ProductionEvidenceWriteError("result binding launch-config hash differs")
    if binding.get("plan_sha256") != formal_manifest.get("plan_sha256"):
        raise ProductionEvidenceWriteError("result binding plan hash differs")
    if binding.get("base_method_commit") != run.get("base_method_commit"):
        raise ProductionEvidenceWriteError("result binding base commit differs")
    if binding.get("full_method_compatibility_fingerprint") != run.get(
        "full_method_compatibility_fingerprint"
    ):
        raise ProductionEvidenceWriteError(
            "result binding Full-method fingerprint differs"
        )

    run_id = _nonempty(run.get("run_id"), "run_id")
    seed = _integer(run.get("seed"), "seed", minimum=0)
    physical_budget = _integer(run.get("physical_budget"), "physical_budget", minimum=1)
    condition = _sha(run.get("condition_fingerprint"), "condition fingerprint")
    terminal = _sha(run.get("terminal_fingerprint"), "terminal fingerprint")
    scene = _sha(run.get("scene_fingerprint"), "scene fingerprint")
    for field_name in (
        "invariant_fingerprint",
        "observation_contract_fingerprint",
        "task_instruction_fingerprint",
        "shared_prompt_contract_fingerprint",
        "decision_prompt_fingerprint",
        "skill_api_fingerprint",
        "full_method_compatibility_fingerprint",
    ):
        _sha(run.get(field_name), f"RunSpec {field_name}")
    for field_name in (
        "table_id",
        "method_variant",
        "scene_id",
        "memory_condition",
        "memory_policy",
        "protocol_id",
        "output_relpath",
    ):
        _nonempty(run.get(field_name), f"RunSpec {field_name}")
    base_commit = run.get("base_method_commit")
    if not isinstance(base_commit, str) or _GIT_COMMIT_RE.fullmatch(base_commit) is None:
        raise ProductionEvidenceWriteError("RunSpec base commit must be lowercase Git SHA-1")
    route_id = _nonempty(run.get("provider_route_id"), "provider route ID")
    route_fp = _sha(run.get("provider_route_fingerprint"), "provider route fingerprint")
    model = _nonempty(run.get("target_backbone"), "target model")
    if (
        provider.get("route_id") != route_id
        or provider.get("route_fingerprint") != route_fp
        or provider.get("target_model") != model
    ):
        raise ProductionEvidenceWriteError("provider channel differs from RunSpec")
    provider_name = _nonempty(provider.get("provider"), "provider")
    if output.get("seed") != seed or output.get("physical_budget") != physical_budget:
        raise ProductionEvidenceWriteError("formal output seed/budget differs from RunSpec")
    if output.get("result_contract_schema_version") != RUNNER_RESULT_CONTRACT_SCHEMA_VERSION:
        raise ProductionEvidenceWriteError("formal output result schema differs")
    if output.get("terminal_evidence_schema_version") != TERMINAL_EVIDENCE_SCHEMA_VERSION:
        raise ProductionEvidenceWriteError("formal output terminal schema differs")
    if output.get("fm_usage_ledger_schema_version") != FM_USAGE_LEDGER_SCHEMA_VERSION:
        raise ProductionEvidenceWriteError("formal output usage schema differs")
    if output.get("transaction_accounting_ledger_schema_version") != (
        TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION
    ):
        raise ProductionEvidenceWriteError(
            "formal output transaction accounting schema differs"
        )
    output_value = output.get("output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise ProductionEvidenceWriteError("formal output directory is missing")
    results_root_value = output.get("results_root")
    output_relpath_value = output.get("output_relpath")
    if not isinstance(results_root_value, str) or not results_root_value:
        raise ProductionEvidenceWriteError("formal results root is missing")
    if output_relpath_value != run.get("output_relpath"):
        raise ProductionEvidenceWriteError("formal output relpath differs from RunSpec")
    results_root = Path(results_root_value)
    output_path = Path(output_value)
    if not results_root.is_absolute() or not output_path.is_absolute():
        raise ProductionEvidenceWriteError("formal result paths must be absolute")
    relative = PurePosixPath(str(output_relpath_value))
    if (
        relative.is_absolute()
        or relative.as_posix() != output_relpath_value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ProductionEvidenceWriteError("RunSpec output relpath is unsafe")
    expected_output = results_root.joinpath(*relative.parts)
    if output_path != expected_output:
        raise ProductionEvidenceWriteError(
            "formal output directory is not results_root/output_relpath"
        )

    if simulator.get("scene_id") != run.get("scene_id"):
        raise ProductionEvidenceWriteError("simulator scene ID differs from RunSpec")
    scene_realization = _sha(
        simulator.get("execution_composite_sha256"),
        "scene realization hash",
    )
    compiled = _sha(simulator.get("compiled_model_sha256"), "compiled model hash")
    execution_evidence = _sha(
        simulator.get("execution_evidence_fingerprint"),
        "scene execution evidence fingerprint",
    )
    target = _mapping(simulator.get("target"), "target geometry")
    center = _mapping(target.get("center"), "target center")
    target_xyz = tuple(
        _number(center.get(axis), f"target center {axis}") for axis in ("x", "y", "z")
    )
    target_radius = _number(target.get("radius_m"), "target radius")
    if not math.isclose(target_radius, 0.30, rel_tol=0.0, abs_tol=1e-12):
        raise ProductionEvidenceWriteError("target radius differs from the frozen predicate")
    required = _mapping(
        output.get("required_result_artifact_pins"),
        "required result artifact pins",
    )
    required_values = frozenset(str(value) for value in required.values())
    if not {
        _TERMINAL_PIN,
        _USAGE_PIN,
        V83_NATIVE_MEMORY_AUDIT_PIN,
        TRANSACTION_ACCOUNTING_LEDGER_PIN,
        ACCEPTED_MANIFEST_PIN,
        FROZEN_PHASE_PLAN_PIN,
    }.issubset(required_values):
        raise ProductionEvidenceWriteError("formal output required pin set is incomplete")
    return _Identity(
        output_dir=output_path,
        run=MappingProxyType(dict(run)),
        result_binding=MappingProxyType(dict(binding)),
        run_id=run_id,
        seed=seed,
        physical_budget=physical_budget,
        condition_fingerprint=condition,
        terminal_fingerprint=terminal,
        scene_fingerprint=scene,
        scene_realization_sha256=scene_realization,
        compiled_model_sha256=compiled,
        scene_execution_evidence_fingerprint=execution_evidence,
        target_center_xyz_m=(float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])),
        provider=provider_name,
        provider_route_id=route_id,
        provider_route_fingerprint=route_fp,
        target_model=model,
        required_result_pins=required_values,
    )


def _build_fm_calls(
    calls: Sequence[FoundationModelCall | Mapping[str, Any]],
    target_model: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(calls, (str, bytes)) or not isinstance(calls, Sequence) or not calls:
        raise ProductionEvidenceWriteError("at least one FM call record is required")
    documents: list[dict[str, Any]] = []
    request_ids: set[str] = set()
    text_calls = vision_calls = input_tokens = output_tokens = 0
    successful_responses = failed_requests = transport_attempt_count = 0
    costs: list[float] = []
    for ordinal, item in enumerate(calls, 1):
        if isinstance(item, FoundationModelCall):
            value = {name: getattr(item, name) for name in _FM_CALL_INPUT_FIELDS}
        elif isinstance(item, Mapping):
            value = dict(item)
            supplied_ordinal = value.pop("ordinal", ordinal)
            if supplied_ordinal != ordinal:
                raise ProductionEvidenceWriteError("FM call ordinal differs from sequence")
            fields = set(value)
            if (
                not _FM_CALL_INPUT_REQUIRED_FIELDS.issubset(fields)
                or not fields.issubset(_FM_CALL_INPUT_FIELDS)
            ):
                raise ProductionEvidenceWriteError(
                    "FM call fields differ; aggregate totals are not accepted"
                )
            for name in _FM_CALL_INPUT_OPTIONAL_FIELDS:
                value.setdefault(name, None if name != "transport_attempts" else ())
        else:
            raise ProductionEvidenceWriteError("FM call must be a record or object")
        request_id = _nonempty(value.get("request_id"), "FM request_id")
        if request_id in request_ids:
            raise ProductionEvidenceWriteError("FM request IDs must be unique")
        request_ids.add(request_id)
        _nonempty(value.get("role"), "FM role")
        modality = value.get("modality")
        if modality not in {"text", "vision"}:
            raise ProductionEvidenceWriteError("FM modality must be text or vision")
        if value.get("requested_model") != target_model:
            raise ProductionEvidenceWriteError(
                "FM requested_model differs from the launch target model"
            )
        resolved_model = value.get("resolved_model")
        if resolved_model is not None and resolved_model != target_model:
            raise ProductionEvidenceWriteError(
                "FM resolved_model differs from the launch target model"
            )
        attempts, inferred_outcome, final_http_status = _build_transport_attempts(
            value.get("transport_attempts"),
            fallback_http_status=value.get("http_status"),
        )
        supplied_outcome = value.get("outcome")
        if supplied_outcome is not None and supplied_outcome != inferred_outcome:
            raise ProductionEvidenceWriteError(
                "FM logical outcome differs from its final transport attempt"
            )
        outcome = inferred_outcome
        if value.get("http_status") != final_http_status:
            raise ProductionEvidenceWriteError(
                "FM final HTTP status differs from its transport attempts"
            )
        response_model = value.get("response_model")
        input_value = value.get("input_tokens")
        output_value = value.get("output_tokens")
        if outcome == "success":
            if resolved_model != target_model or response_model != target_model:
                raise ProductionEvidenceWriteError(
                    "successful FM response model differs from the launch target"
                )
            input_count = _integer(input_value, "FM input tokens", minimum=0)
            output_count = _integer(output_value, "FM output tokens", minimum=0)
            successful_responses += 1
            input_tokens += input_count
            output_tokens += output_count
        else:
            if response_model is not None or input_value is not None or output_value is not None:
                raise ProductionEvidenceWriteError(
                    "failed FM request cannot contain model-response or token values"
                )
            failed_requests += 1
        cost_value = value.get("billing_amount_usd")
        if cost_value is not None:
            cost = _number(cost_value, "FM billing amount")
            if cost < 0:
                raise ProductionEvidenceWriteError(
                    "FM billing amount cannot be negative"
                )
            costs.append(cost)
        text_calls += modality == "text"
        vision_calls += modality == "vision"
        transport_attempt_count += len(attempts)
        documents.append(
            {
                "ordinal": ordinal,
                "request_id": request_id,
                "role": value["role"],
                "modality": modality,
                "requested_model": target_model,
                "resolved_model": resolved_model,
                "outcome": outcome,
                "transport_attempts": attempts,
                "response_model": response_model,
                "input_tokens": input_value,
                "output_tokens": output_value,
                "billing_amount_usd": cost_value,
            }
        )
    billing_records = len(costs)
    if billing_records == len(calls):
        billing_completeness = "complete"
        billing_amount_usd: float | None = math.fsum(costs)
    elif billing_records:
        billing_completeness = "partial"
        billing_amount_usd = None
    else:
        billing_completeness = "unavailable"
        billing_amount_usd = None
    return documents, {
        "llm_calls": text_calls,
        "vlm_calls": vision_calls,
        "successful_responses": successful_responses,
        "failed_requests": failed_requests,
        "transport_attempts": transport_attempt_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "billing_records": billing_records,
        "billing_amount_usd": billing_amount_usd,
        "billing_completeness": billing_completeness,
    }


def _build_transport_attempts(
    supplied: object,
    *,
    fallback_http_status: object,
) -> tuple[list[dict[str, Any]], str, int | None]:
    if supplied in (None, ()):
        status = _integer(fallback_http_status, "FM HTTP status", minimum=100)
        if status > 599:
            raise ProductionEvidenceWriteError("FM HTTP status must be <= 599")
        attempts: Sequence[FoundationModelTransportAttempt | Mapping[str, Any]] = (
            FoundationModelTransportAttempt(
                outcome="http_response",
                http_status=status,
            ),
        )
    else:
        if isinstance(supplied, (str, bytes)) or not isinstance(supplied, Sequence):
            raise ProductionEvidenceWriteError(
                "FM transport attempts must be a non-empty sequence"
            )
        attempts = supplied
        if not attempts:
            raise ProductionEvidenceWriteError(
                "FM transport attempts must be a non-empty sequence"
            )
    documents: list[dict[str, Any]] = []
    final_success = False
    final_http_status: int | None = None
    for ordinal, item in enumerate(attempts, 1):
        if isinstance(item, FoundationModelTransportAttempt):
            value = {name: getattr(item, name) for name in _FM_ATTEMPT_INPUT_FIELDS}
        elif isinstance(item, Mapping):
            value = dict(item)
            supplied_ordinal = value.pop("ordinal", ordinal)
            if supplied_ordinal != ordinal:
                raise ProductionEvidenceWriteError(
                    "FM transport attempt ordinal differs from sequence"
                )
            if set(value) != _FM_ATTEMPT_INPUT_FIELDS:
                raise ProductionEvidenceWriteError(
                    "FM transport attempt fields differ"
                )
        else:
            raise ProductionEvidenceWriteError(
                "FM transport attempt must be a record or object"
            )
        attempt_outcome = value.get("outcome")
        http_status = value.get("http_status")
        error_type = value.get("error_type")
        if attempt_outcome == "http_response":
            status = _integer(http_status, "FM attempt HTTP status", minimum=100)
            if status > 599 or error_type is not None:
                raise ProductionEvidenceWriteError(
                    "FM HTTP attempt must have status 100..599 and no error_type"
                )
            succeeded = 200 <= status < 300
            final_status: int | None = status
        elif attempt_outcome == "transport_error":
            if http_status is not None:
                raise ProductionEvidenceWriteError(
                    "FM transport error cannot have an HTTP status"
                )
            _nonempty(error_type, "FM transport error_type")
            succeeded = False
            final_status = None
        else:
            raise ProductionEvidenceWriteError(
                "FM attempt outcome must be http_response or transport_error"
            )
        if succeeded and ordinal != len(attempts):
            raise ProductionEvidenceWriteError(
                "FM transport attempts cannot continue after success"
            )
        if ordinal == len(attempts):
            final_success = succeeded
            final_http_status = final_status
        documents.append(
            {
                "ordinal": ordinal,
                "outcome": attempt_outcome,
                "http_status": final_status,
                "error_type": error_type,
            }
        )
    return (
        documents,
        "success" if final_success else "failure",
        final_http_status,
    )


def _build_system_stall_document(
    run_root: Path,
    *,
    run_id: str,
    evidence: SystemStallEvidence,
    calls: Sequence[Mapping[str, Any]],
    terminal_state: Mapping[str, Any],
    terminal_state_sha256: str,
    terminal_sample_trace: Mapping[str, Any],
    terminal_sample_trace_sha256: str,
    capability_entrypoint_sha256: str,
) -> AuxiliaryEvidenceDerivation:
    if not isinstance(evidence, SystemStallEvidence):
        raise ProductionEvidenceWriteError("system_stall evidence has the wrong type")
    if len(evidence.generation_cycles) != SYSTEM_STALL_GENERATION_CYCLES:
        raise ProductionEvidenceWriteError("SYSTEM_STALL requires exactly three cycles")
    cycles = []
    for cycle in evidence.generation_cycles:
        if not isinstance(cycle, SystemStallCycleEvidence):
            raise ProductionEvidenceWriteError("stall cycle has the wrong type")
        cycles.append(
            {
                "model_output": _raw_artifact_ref(cycle.model_output, "stall model output"),
                "static_validation": _raw_artifact_ref(
                    cycle.static_validation, "stall static validation"
                ),
                "host_preflight": _raw_artifact_ref(
                    cycle.host_preflight, "stall Host preflight"
                ),
            }
        )
    checkpoint_id, checkpoint_event_id = _terminal_checkpoint_binding(
        terminal_sample_trace,
        terminal_state,
    )
    derived_checkpoint = _stall_checkpoint_summary(terminal_state)
    result = recompute_system_stall_ledger(
        run_root,
        source_index={
            "checkpoint_validity": _raw_artifact_ref(
                evidence.checkpoint_validity, "stall checkpoint validity"
            ),
            "host_action_space": _raw_artifact_ref(
                evidence.host_action_space, "stall Host action space"
            ),
            "generation_cycles": cycles,
        },
        expected_run_id=run_id,
        expected_checkpoint_id=checkpoint_id,
        expected_checkpoint_event_id=checkpoint_event_id,
        expected_terminal_state_sha256=terminal_state_sha256,
        expected_terminal_sample_trace_sha256=terminal_sample_trace_sha256,
        derived_checkpoint=derived_checkpoint,
        expected_producers=capability_bound_auxiliary_producer_allowlist(
            capability_entrypoint_sha256
        ),
        fm_calls=calls,
    )
    _require_auxiliary_derivation(result, "SYSTEM_STALL")
    return result


def _build_payload_drop_document(
    run_root: Path,
    *,
    run_id: str,
    physical_transactions: int,
    terminal_sample_trace: Mapping[str, Any],
    evidence: PayloadDropEvidence,
    capability_entrypoint_sha256: str,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
) -> AuxiliaryEvidenceDerivation:
    if not isinstance(evidence, PayloadDropEvidence):
        raise ProductionEvidenceWriteError("payload_drop evidence has the wrong type")
    refs = tuple(evidence.transaction_receipts)
    if len(refs) != PAYLOAD_DROP_LIMIT:
        raise ProductionEvidenceWriteError(
            "payload-drop evidence requires five immutable transaction receipts"
        )
    checkpoint_id = _nonempty(
        terminal_sample_trace.get("checkpoint_id"),
        "terminal trace checkpoint",
    )
    checkpoint_event_id = _nonempty(
        terminal_sample_trace.get("checkpoint_event_id"),
        "terminal trace checkpoint event",
    )
    result = recompute_payload_drop_ledger(
        run_root,
        source_index={
            "source_transaction_receipts": [
                _raw_artifact_ref(item, "payload-drop transaction receipt")
                for item in refs
            ]
        },
        expected_run_id=run_id,
        expected_terminal_checkpoint_id=checkpoint_id,
        expected_terminal_checkpoint_event_id=checkpoint_event_id,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_scene_realization_sha256=expected_scene_realization_sha256,
        expected_producers=capability_bound_auxiliary_producer_allowlist(
            capability_entrypoint_sha256
        ),
        physical_transactions=physical_transactions,
    )
    _require_auxiliary_derivation(result, "payload-drop")
    return result


def _raw_artifact_ref(value: ProductionArtifactRef, label: str) -> dict[str, Any]:
    if not isinstance(value, ProductionArtifactRef):
        raise ProductionEvidenceWriteError(f"{label} must be a ProductionArtifactRef")
    relative = value.relative_path
    pure = PurePosixPath(relative) if isinstance(relative, str) else None
    if (
        pure is None
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ProductionEvidenceWriteError(f"{label} path must be normalized and relative")
    _sha(value.sha256, f"{label} SHA-256")
    _integer(value.size_bytes, f"{label} size", minimum=0)
    return value.as_dict()


def _terminal_checkpoint_binding(
    terminal_sample_trace: Mapping[str, Any],
    terminal_state: Mapping[str, Any],
) -> tuple[str, str]:
    trace_checkpoint = _nonempty(
        terminal_sample_trace.get("checkpoint_id"), "terminal trace checkpoint"
    )
    state_checkpoint = _nonempty(
        terminal_state.get("checkpoint_id"), "terminal state checkpoint"
    )
    trace_event = _nonempty(
        terminal_sample_trace.get("checkpoint_event_id"),
        "terminal trace checkpoint event",
    )
    state_event = _nonempty(
        terminal_state.get("checkpoint_event_id"),
        "terminal state checkpoint event",
    )
    if (trace_checkpoint, trace_event) != (state_checkpoint, state_event):
        raise ProductionEvidenceWriteError(
            "terminal state and trace checkpoint bindings differ"
        )
    return trace_checkpoint, trace_event


def _stall_checkpoint_summary(terminal_state: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(terminal_state.get("payload"), "terminal state payload")
    robots = _mapping(terminal_state.get("robots"), "terminal state robots")
    summary_robots: dict[str, Any] = {}
    for robot_id in ("front", "back"):
        robot = _mapping(robots.get(robot_id), f"terminal state {robot_id} robot")
        summary_robots[robot_id] = {
            "upright": robot.get("upright"),
            "pelvis_height_m": robot.get("pelvis_height_m"),
        }
    return {
        "sample_window_s": terminal_state.get("sample_window_s"),
        "sample_count": terminal_state.get("sample_count"),
        "payload": {
            "floor_contact_samples": payload.get("floor_contact_samples"),
            "wall_contact_samples": payload.get("wall_contact_samples"),
        },
        "robots": summary_robots,
    }


def _require_auxiliary_derivation(
    result: AuxiliaryEvidenceDerivation,
    label: str,
) -> None:
    if not result.valid:
        raise ProductionEvidenceWriteError(
            f"{label} raw evidence is invalid: "
            + ", ".join(result.invalid_reasons or ("DERIVATION_FAILED",))
        )


def _snapshot_source_artifacts(
    artifacts: Sequence[BoundJsonArtifact],
) -> dict[str, bytes]:
    snapshots: dict[str, bytes] = {}
    for artifact in artifacts:
        if artifact.relative_path in snapshots:
            continue
        try:
            raw = artifact.path.read_bytes()
        except OSError as exc:
            raise ProductionEvidenceWriteError(
                f"cannot snapshot raw evidence: {artifact.relative_path}"
            ) from exc
        if (
            len(raw) != artifact.size_bytes
            or hashlib.sha256(raw).hexdigest() != artifact.sha256
        ):
            raise ProductionEvidenceWriteError(
                f"raw evidence changed while being inspected: {artifact.relative_path}"
            )
        snapshots[artifact.relative_path] = raw
    return snapshots


def _recheck_source_artifacts(
    root: Path,
    artifacts: Sequence[BoundJsonArtifact],
) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.relative_path in seen:
            continue
        seen.add(artifact.relative_path)
        try:
            current = load_bound_json_artifact(
                root,
                artifact.public_ref(),
                label=f"raw evidence {artifact.relative_path}",
            )
        except BoundArtifactError as exc:
            raise ProductionEvidenceWriteError(
                f"raw evidence changed or became unsafe: {artifact.relative_path}"
            ) from exc
        if current.sha256 != artifact.sha256:
            raise ProductionEvidenceWriteError(
                f"raw evidence changed while result was being prepared: {artifact.relative_path}"
            )


def _claimed_accounting(value: ProductionRunAccounting) -> dict[str, Any]:
    """Validate a redundant collector claim without treating it as evidence."""

    if not isinstance(value, ProductionRunAccounting):
        raise ProductionEvidenceWriteError("accounting claim has the wrong type")
    names = (
        "physical_transactions",
        "accepted_transactions",
        "rejected_transactions",
        "accepted_setup_transactions",
        "accepted_progress_transactions",
        "phase_completed",
        "phase_total",
    )
    result = {
        name: _integer(
            getattr(value, name),
            f"accounting claim {name}",
            minimum=0,
        )
        for name in names
    }
    result["failure_counts"] = _count_mapping(
        value.failure_counts,
        "accounting claim failure_counts",
    )
    result["mutation_counts"] = _count_mapping(
        value.mutation_counts,
        "accounting claim mutation_counts",
    )
    elapsed = _number(value.elapsed_s, "accounting claim elapsed_s")
    if elapsed < 0:
        raise ProductionEvidenceWriteError("elapsed_s cannot be negative")
    return result


def _derive_transaction_accounting(
    root: Path,
    *,
    identity: _Identity,
    evidence: TransactionAccountingEvidence,
    claimed_accounting: Mapping[str, Any],
) -> TransactionAccountingDerivation:
    if not isinstance(evidence, TransactionAccountingEvidence):
        raise ProductionEvidenceWriteError(
            "transaction_accounting_evidence has the wrong type"
        )
    receipts = _artifact_ref_sequence(
        evidence.transaction_receipts,
        "physical transaction receipts",
    )
    mutations = _artifact_ref_sequence(
        evidence.mutation_commit_receipts,
        "mutation commit receipts",
    )
    _require_artifact_path(
        evidence.frozen_phase_plan,
        FROZEN_PHASE_PLAN_RELPATH,
        "frozen phase plan",
    )
    _require_artifact_path(
        evidence.accepted_episode_manifest,
        ACCEPTED_MANIFEST_RELPATH,
        "accepted episode manifest",
    )
    for ordinal, reference in enumerate(receipts, 1):
        _require_artifact_path(
            reference,
            f"evidence/transactions/transaction_{ordinal:03d}.json",
            f"physical transaction receipt {ordinal}",
        )
    for ordinal, reference in enumerate(mutations, 1):
        _require_artifact_path(
            reference,
            f"evidence/mutations/mutation_commit_{ordinal:03d}.json",
            f"mutation commit receipt {ordinal}",
        )
    _require_complete_canonical_artifact_set(
        root / "evidence" / "transactions",
        prefix="transaction_",
        suffix=".json",
        expected_paths={item.relative_path for item in receipts},
        label="physical transaction receipt",
    )
    _require_complete_canonical_artifact_set(
        root / "evidence" / "mutations",
        prefix="mutation_commit_",
        suffix=".json",
        expected_paths={item.relative_path for item in mutations},
        label="mutation commit receipt",
    )
    producers = capability_bound_auxiliary_producer_allowlist(
        str(identity.result_binding["entrypoint_sha256"])
    )
    return recompute_transaction_accounting_ledger(
        root,
        transaction_receipt_refs=tuple(item.as_dict() for item in receipts),
        phase_plan_ref=evidence.frozen_phase_plan.as_dict(),
        accepted_manifest_ref=evidence.accepted_episode_manifest.as_dict(),
        mutation_commit_receipt_refs=tuple(item.as_dict() for item in mutations),
        expected_run_id=identity.run_id,
        # Coverage comes from the complete canonical ref set, never from the
        # caller-authored aggregate claim.
        expected_physical_transactions=len(receipts),
        physical_budget=identity.physical_budget,
        expected_compiled_model_sha256=identity.compiled_model_sha256,
        expected_scene_realization_sha256=identity.scene_realization_sha256,
        expected_producers=producers,
        claimed_accounting=claimed_accounting,
    )


def _artifact_ref_sequence(
    value: object,
    label: str,
) -> tuple[ProductionArtifactRef, ...]:
    if isinstance(value, (str, bytes)):
        raise ProductionEvidenceWriteError(f"{label} must be a sequence")
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ProductionEvidenceWriteError(f"{label} must be a sequence") from exc
    if not all(isinstance(item, ProductionArtifactRef) for item in result):
        raise ProductionEvidenceWriteError(
            f"{label} must contain ProductionArtifactRef values"
        )
    return result


def _require_artifact_path(
    value: ProductionArtifactRef,
    expected: str,
    label: str,
) -> None:
    if not isinstance(value, ProductionArtifactRef):
        raise ProductionEvidenceWriteError(f"{label} has the wrong type")
    if value.relative_path != expected:
        raise ProductionEvidenceWriteError(f"{label} path is not canonical")
    _sha(value.sha256, f"{label} hash")
    _integer(value.size_bytes, f"{label} size", minimum=1)


def _require_complete_canonical_artifact_set(
    directory: Path,
    *,
    prefix: str,
    suffix: str,
    expected_paths: set[str],
    label: str,
) -> None:
    if not directory.exists():
        if expected_paths:
            raise ProductionEvidenceWriteError(f"{label} directory is missing")
        return
    if directory.is_symlink() or not directory.is_dir():
        raise ProductionEvidenceWriteError(f"{label} directory is unsafe")
    discovered: set[str] = set()
    for path in directory.iterdir():
        name = path.name
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        ordinal = name[len(prefix) : -len(suffix)]
        if len(ordinal) != 3 or not ordinal.isdigit() or int(ordinal) < 1:
            continue
        if path.is_symlink() or not path.is_file():
            raise ProductionEvidenceWriteError(f"{label} path is unsafe")
        discovered.add(path.relative_to(directory.parents[1]).as_posix())
    if discovered != expected_paths:
        raise ProductionEvidenceWriteError(
            f"{label} references do not cover the complete canonical set"
        )


def _authoritative_accounting(
    ledger: Mapping[str, Any],
    *,
    elapsed_s: float,
) -> ProductionRunAccounting:
    return ProductionRunAccounting(
        physical_transactions=_integer(
            ledger.get("physical_transactions"),
            "derived physical transactions",
            minimum=0,
        ),
        accepted_transactions=_integer(
            ledger.get("accepted_transactions"),
            "derived accepted transactions",
            minimum=0,
        ),
        rejected_transactions=_integer(
            ledger.get("rejected_transactions"),
            "derived rejected transactions",
            minimum=0,
        ),
        accepted_setup_transactions=_integer(
            ledger.get("accepted_setup_transactions"),
            "derived accepted setup transactions",
            minimum=0,
        ),
        accepted_progress_transactions=_integer(
            ledger.get("accepted_progress_transactions"),
            "derived accepted progress transactions",
            minimum=0,
        ),
        elapsed_s=_number(elapsed_s, "elapsed_s"),
        phase_completed=_integer(
            ledger.get("phase_completed"),
            "derived completed phases",
            minimum=0,
        ),
        phase_total=_integer(
            ledger.get("phase_total"),
            "derived total phases",
            minimum=1,
        ),
        failure_counts=_count_mapping(
            _mapping(ledger.get("failure_counts"), "derived failure counts"),
            "derived failure counts",
        ),
        mutation_counts=_count_mapping(
            _mapping(ledger.get("mutation_counts"), "derived mutation counts"),
            "derived mutation counts",
        ),
    )


def _run_accounting_validator(
    root: Path,
    *,
    identity: _Identity,
    accounting: ProductionRunAccounting,
) -> TransactionAccountingEvidenceValidation:
    return validate_transaction_accounting_evidence(
        root,
        expected_run_id=identity.run_id,
        physical_budget=identity.physical_budget,
        expected_compiled_model_sha256=identity.compiled_model_sha256,
        expected_scene_realization_sha256=identity.scene_realization_sha256,
        expected_producers=capability_bound_auxiliary_producer_allowlist(
            str(identity.result_binding["entrypoint_sha256"])
        ),
        claimed_accounting=_claimed_accounting(accounting),
    )


def _build_metrics(
    identity: _Identity,
    *,
    terminal_status: str,
    accounting: ProductionRunAccounting,
    totals: Mapping[str, Any],
) -> dict[str, Any]:
    model_totals = dict(totals)
    billing_status = str(model_totals.pop("billing_completeness"))
    return {
        "task_success": terminal_status == "SUCCESS",
        "physical_iterations": accounting.physical_transactions,
        "accepted_step_count": accounting.accepted_transactions,
        "rejected_step_count": accounting.rejected_transactions,
        "accepted_setup_count": accounting.accepted_setup_transactions,
        "accepted_progress_count": accounting.accepted_progress_transactions,
        "elapsed_s": float(accounting.elapsed_s),
        "phase_completion": {
            "completed": accounting.phase_completed,
            "total": accounting.phase_total,
            "fraction": accounting.phase_completed / accounting.phase_total,
        },
        "failure_counts": dict(accounting.failure_counts),
        "mutation_counts": dict(accounting.mutation_counts),
        "model": {
            **model_totals,
            "billing_reconciliation_status": billing_status,
        },
        "terminal": {"status": terminal_status},
        "live_launch": dict(identity.result_binding),
    }


def _build_manifest(
    identity: _Identity,
    *,
    terminal_status: str,
    result_artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    run = identity.run
    output_parts = PurePosixPath(str(run["output_relpath"])).parts
    if len(output_parts) < 2:
        raise ProductionEvidenceWriteError("RunSpec output path has no condition component")
    experiment = {
        "run_id": identity.run_id,
        "table": run["table_id"],
        "condition": output_parts[1],
        "method": run["method_variant"],
        "scene": run["scene_id"],
        "memory_condition": run["memory_condition"],
        "memory_policy": run["memory_policy"],
        "target_model": identity.target_model,
        "provider": identity.provider,
        "provider_route_id": identity.provider_route_id,
        "provider_route_fingerprint": identity.provider_route_fingerprint,
        "seed": identity.seed,
        "protocol_id": run["protocol_id"],
        "base_method_commit": run["base_method_commit"],
        "output_relpath": run["output_relpath"],
        "condition_fingerprint": identity.condition_fingerprint,
        "invariant_fingerprint": run["invariant_fingerprint"],
        "scene_fingerprint": identity.scene_fingerprint,
        "terminal_fingerprint": identity.terminal_fingerprint,
        "observation_contract_fingerprint": run[
            "observation_contract_fingerprint"
        ],
        "task_instruction_fingerprint": run["task_instruction_fingerprint"],
        "shared_prompt_contract_fingerprint": run[
            "shared_prompt_contract_fingerprint"
        ],
        "decision_prompt_fingerprint": run["decision_prompt_fingerprint"],
        "skill_api_fingerprint": run["skill_api_fingerprint"],
        "full_method_compatibility_fingerprint": run[
            "full_method_compatibility_fingerprint"
        ],
    }
    strategy_bank = run.get("strategy_bank")
    if isinstance(strategy_bank, Mapping):
        experiment.update(
            {
                "source_model": strategy_bank.get("source_backbone", ""),
                "source_provider_route_id": strategy_bank.get(
                    "source_provider_route_id", ""
                ),
                "source_provider_route_fingerprint": strategy_bank.get(
                    "source_provider_route_fingerprint", ""
                ),
                "strategy_bank_sha256": strategy_bank.get("sha256", ""),
            }
        )
    return {
        "status": terminal_status,
        "experiment": experiment,
        "protocol": {"protocol_id": run["protocol_id"]},
        "codebase": {"git_head": run["base_method_commit"]},
        "model": {
            "provider": identity.provider,
            "provider_route_id": identity.provider_route_id,
            "provider_route_fingerprint": identity.provider_route_fingerprint,
            "requested": identity.target_model,
            "resolved": identity.target_model,
        },
        "live_launch": dict(identity.result_binding),
        "result_artifact_hashes": dict(result_artifact_hashes),
    }


def _validate_staged_documents(
    documents: Mapping[str, bytes],
    *,
    raw_source_documents: Mapping[str, bytes],
    manifest: Mapping[str, Any],
    identity: _Identity,
    terminal_status: str,
    accounting: ProductionRunAccounting,
    totals: Mapping[str, Any],
) -> None:
    with tempfile.TemporaryDirectory(prefix="coevolve-evidence-validation-") as value:
        root = Path(value)
        overlap = set(documents) & set(raw_source_documents)
        if overlap:
            raise ProductionEvidenceWriteError(
                "raw evidence collides with final writer destinations: "
                + ", ".join(sorted(overlap))
            )
        for relative_path, payload in {**raw_source_documents, **documents}.items():
            path = root.joinpath(*PurePosixPath(relative_path).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        terminal, usage = _run_existing_validators(
            root,
            manifest=manifest,
            identity=identity,
            terminal_status=terminal_status,
            accounting=accounting,
            totals=totals,
        )
        accounting_validation = _run_accounting_validator(
            root,
            identity=identity,
            accounting=accounting,
        )
    if not terminal.valid or not usage.valid or not accounting_validation.valid:
        details = (
            *terminal.invalid_reasons,
            *usage.invalid_reasons,
            *accounting_validation.invalid_reasons,
        )
        raise ProductionEvidenceWriteError(
            "proposed evidence fails existing validators: " + ", ".join(details)
        )


def _run_existing_validators(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    identity: _Identity,
    terminal_status: str,
    accounting: ProductionRunAccounting,
    totals: Mapping[str, Any],
) -> tuple[TerminalEvidenceValidation, UsageEvidenceValidation]:
    terminal = validate_terminal_evidence(
        root,
        manifest=manifest,
        expected_run_id=identity.run_id,
        expected_seed=identity.seed,
        expected_condition_fingerprint=identity.condition_fingerprint,
        expected_terminal_fingerprint=identity.terminal_fingerprint,
        expected_terminal_status=terminal_status,
        expected_scene_fingerprint=identity.scene_fingerprint,
        expected_scene_realization_sha256=identity.scene_realization_sha256,
        expected_compiled_model_sha256=identity.compiled_model_sha256,
        expected_scene_execution_evidence_fingerprint=(
            identity.scene_execution_evidence_fingerprint
        ),
        expected_target_center_xyz_m=identity.target_center_xyz_m,
        physical_transactions=accounting.physical_transactions,
        physical_budget=identity.physical_budget,
    )
    usage = validate_fm_usage_evidence(
        root,
        manifest=manifest,
        expected_run_id=identity.run_id,
        expected_seed=identity.seed,
        expected_condition_fingerprint=identity.condition_fingerprint,
        expected_provider_route_id=identity.provider_route_id,
        expected_provider_route_fingerprint=identity.provider_route_fingerprint,
        expected_target_model=identity.target_model,
        expected_llm_calls=int(totals["llm_calls"]),
        expected_vlm_calls=int(totals["vlm_calls"]),
        expected_input_tokens=int(totals["input_tokens"]),
        expected_output_tokens=int(totals["output_tokens"]),
        expected_billing_amount_usd=(
            float(totals["billing_amount_usd"])
            if totals.get("billing_amount_usd") is not None
            else None
        ),
        expected_billing_reconciliation_status=str(
            totals["billing_completeness"]
        ),
        expected_schema_version=FM_USAGE_LEDGER_SCHEMA_VERSION,
        required_request_ids=terminal.stall_model_request_ids,
    )
    return terminal, usage


def _hash_existing_manifest_artifacts(
    root: Path,
    *,
    table_id: object,
    required_pins: frozenset[str],
    additional: Sequence[ManifestArtifactPin],
) -> dict[str, str]:
    sources: dict[str, str] = {
        V83_NATIVE_MEMORY_AUDIT_PIN: V83_NATIVE_MEMORY_AUDIT_RELPATH,
        ACCEPTED_MANIFEST_PIN: ACCEPTED_MANIFEST_RELPATH,
        FROZEN_PHASE_PLAN_PIN: FROZEN_PHASE_PLAN_RELPATH,
    }
    if table_id in {"table2_method_comparison", "table3_scene_generalization"}:
        sources.update(
            {
                _FROM_SCRATCH_BOOTSTRAP_PIN: _FROM_SCRATCH_BOOTSTRAP_RELPATH,
                _FROM_SCRATCH_RUNTIME_PIN: _FROM_SCRATCH_RUNTIME_RELPATH,
            }
        )
    for item in additional:
        if not isinstance(item, ManifestArtifactPin):
            raise ProductionEvidenceWriteError("additional artifact pin has the wrong type")
        name = _nonempty(item.name, "artifact pin name")
        if name in _RESERVED_PINS or name in sources:
            raise ProductionEvidenceWriteError(f"duplicate or reserved artifact pin: {name}")
        sources[name] = item.relative_path
    missing = required_pins - _RESERVED_PINS - set(sources)
    if missing:
        raise ProductionEvidenceWriteError(
            "required manifest artifact pins have no fixed source: "
            + ", ".join(sorted(missing))
        )
    return {
        name: _hash_safe_relative_file(root, relative_path, f"artifact pin {name}")
        for name, relative_path in sources.items()
    }


def _recheck_manifest_artifacts(
    root: Path,
    artifact_pins: Mapping[str, str],
    additional: Sequence[ManifestArtifactPin],
) -> None:
    sources = {
        V83_NATIVE_MEMORY_AUDIT_PIN: V83_NATIVE_MEMORY_AUDIT_RELPATH,
        ACCEPTED_MANIFEST_PIN: ACCEPTED_MANIFEST_RELPATH,
        FROZEN_PHASE_PLAN_PIN: FROZEN_PHASE_PLAN_RELPATH,
        **{
            item.name: item.relative_path
            for item in additional
        },
    }
    if _FROM_SCRATCH_BOOTSTRAP_PIN in artifact_pins:
        sources[_FROM_SCRATCH_BOOTSTRAP_PIN] = _FROM_SCRATCH_BOOTSTRAP_RELPATH
    if _FROM_SCRATCH_RUNTIME_PIN in artifact_pins:
        sources[_FROM_SCRATCH_RUNTIME_PIN] = _FROM_SCRATCH_RUNTIME_RELPATH
    for name, relative_path in sources.items():
        actual = _hash_safe_relative_file(root, relative_path, f"artifact pin {name}")
        if artifact_pins.get(name) != actual:
            raise ProductionEvidenceWriteError(
                f"artifact changed while result was being prepared: {name}"
            )


def _validate_accounting(
    value: ProductionRunAccounting,
    terminal_status: str,
    physical_budget: int,
) -> None:
    if not isinstance(value, ProductionRunAccounting):
        raise ProductionEvidenceWriteError("accounting has the wrong type")
    fields = (
        "physical_transactions",
        "accepted_transactions",
        "rejected_transactions",
        "accepted_setup_transactions",
        "accepted_progress_transactions",
        "phase_completed",
        "phase_total",
    )
    parsed = {
        name: _integer(getattr(value, name), f"accounting {name}", minimum=0)
        for name in fields
    }
    if parsed["phase_total"] < 1:
        raise ProductionEvidenceWriteError("phase_total must be positive")
    if parsed["physical_transactions"] != (
        parsed["accepted_transactions"] + parsed["rejected_transactions"]
    ):
        raise ProductionEvidenceWriteError("physical accounting does not reconcile")
    if parsed["accepted_transactions"] != (
        parsed["accepted_setup_transactions"] + parsed["accepted_progress_transactions"]
    ):
        raise ProductionEvidenceWriteError("accepted-kind accounting does not reconcile")
    if parsed["physical_transactions"] > physical_budget:
        raise ProductionEvidenceWriteError("physical budget was exceeded")
    if parsed["phase_completed"] > parsed["phase_total"]:
        raise ProductionEvidenceWriteError("phase completion exceeds phase total")
    elapsed = _number(value.elapsed_s, "elapsed_s")
    if elapsed < 0:
        raise ProductionEvidenceWriteError("elapsed_s cannot be negative")
    if terminal_status == "SUCCESS" and parsed["phase_completed"] != parsed["phase_total"]:
        raise ProductionEvidenceWriteError("SUCCESS requires all phases complete")
    if terminal_status == "PHYSICAL_BUDGET_EXHAUSTED" and (
        parsed["physical_transactions"] != physical_budget
    ):
        raise ProductionEvidenceWriteError(
            "PHYSICAL_BUDGET_EXHAUSTED requires the exact budget"
        )
    _count_mapping(value.failure_counts, "failure_counts")
    _count_mapping(value.mutation_counts, "mutation_counts")


def _validated_output_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ProductionEvidenceWriteError("formal output directory must be absolute")
    if path != Path(os.path.normpath(str(path))):
        raise ProductionEvidenceWriteError("formal output directory must be normalized")
    # Existing path components are checked explicitly; resolve(strict=False)
    # alone is insufficient when a dangling symlink is present.
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ProductionEvidenceWriteError("formal output path contains a symlink")
        if cursor.exists() and not cursor.is_dir():
            raise ProductionEvidenceWriteError(
                "formal output path contains a non-directory component"
            )
    return path


def _hash_safe_relative_file(root: Path, relative: str, label: str) -> str:
    path = _safe_relative_path(root, relative, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionEvidenceWriteError(f"cannot read {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ProductionEvidenceWriteError(f"{label} is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _safe_relative_path(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ProductionEvidenceWriteError(f"{label} path must be non-empty")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ProductionEvidenceWriteError(f"{label} path must be normalized and relative")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ProductionEvidenceWriteError(f"{label} path contains a symlink")
    try:
        cursor.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ProductionEvidenceWriteError(f"{label} is missing or escapes run root") from exc
    return cursor


def _require_all_destinations_absent(
    root: Path,
    documents: Mapping[str, bytes],
) -> None:
    for relative_path in documents:
        path = _fixed_output_path(root, relative_path)
        if path.exists() or path.is_symlink():
            raise ProductionEvidenceWriteError(
                f"refusing to overwrite existing result artifact: {path}"
            )


def _fixed_output_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ProductionEvidenceWriteError("internal result path is unsafe")
    return root.joinpath(*pure.parts)


def _atomic_create_exclusive(path: Path, payload: bytes) -> None:
    _ensure_safe_directory(path.parent)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    except OSError as exc:
        raise ProductionEvidenceWriteError(
            f"cannot stage result artifact: {path}"
        ) from exc
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ProductionEvidenceWriteError(
                f"refusing to overwrite existing result artifact: {path}"
            ) from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except ProductionEvidenceWriteError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProductionEvidenceWriteError(
            f"cannot atomically publish result artifact: {path}"
        ) from exc


def _ensure_safe_directory(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        cursor = cursor.parent
    if cursor.is_symlink() or not cursor.is_dir():
        raise ProductionEvidenceWriteError("result parent is missing, unsafe, or a symlink")
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ProductionEvidenceWriteError(
                f"cannot create result directory: {directory}"
            ) from exc
        if directory.is_symlink() or not directory.is_dir():
            raise ProductionEvidenceWriteError("result directory creation was unsafe")
    try:
        if path.resolve(strict=True) != path:
            raise ProductionEvidenceWriteError(
                "result directory contains a symlinked or non-normalized component"
            )
    except OSError as exc:
        raise ProductionEvidenceWriteError("cannot verify result directory") from exc


def _require_durable_json_matches(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ProductionEvidenceWriteError(f"durable {label} is not a regular file")
    try:
        actual = strict_json_loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ProductionEvidenceWriteError(f"cannot re-read durable {label}") from exc
    if actual != to_primitive(expected):
        raise ProductionEvidenceWriteError(f"durable {label} differs after publication")


def _json_bytes(value: Any, label: str) -> bytes:
    try:
        primitive = to_primitive(value)
        if not isinstance(primitive, Mapping):
            raise TypeError("root is not an object")
        return (
            json.dumps(
                primitive,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionEvidenceWriteError(f"cannot serialize {label}: {exc}") from exc


def _artifact_ref(relative_path: str, payload: bytes) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _producer_document(value: EvidenceProducer) -> dict[str, str]:
    return {"id": value.id, "source_sha256": value.source_sha256}


def _validate_producer(value: EvidenceProducer, label: str) -> None:
    if not isinstance(value, EvidenceProducer):
        raise ProductionEvidenceWriteError(f"{label} has the wrong type")
    _nonempty(value.id, f"{label} id")
    _sha(value.source_sha256, f"{label} source hash")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionEvidenceWriteError(f"{label} must be an object")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionEvidenceWriteError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProductionEvidenceWriteError(f"{label} must be lowercase SHA-256")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionEvidenceWriteError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionEvidenceWriteError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise ProductionEvidenceWriteError(f"{label} must be finite numeric data")
    return result


def _count_mapping(value: Mapping[str, int], label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ProductionEvidenceWriteError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        name = _nonempty(key, f"{label} key")
        result[name] = _integer(count, f"{label}.{name}", minimum=0)
    return result


__all__ = [
    "EvidenceProducer",
    "FoundationModelCall",
    "FoundationModelTransportAttempt",
    "ManifestArtifactPin",
    "METRICS_FILENAME",
    "PAYLOAD_DROP_LEDGER_RELPATH",
    "PayloadDropEvidence",
    "ProductionArtifactRef",
    "ProductionEvidenceWriteError",
    "ProductionEvidenceWriteResult",
    "ProductionRunAccounting",
    "RUN_MANIFEST_FILENAME",
    "RUN_PLAN_FILENAME",
    "SYSTEM_STALL_LEDGER_RELPATH",
    "SystemStallCycleEvidence",
    "SystemStallEvidence",
    "TERMINAL_SAMPLE_TRACE_RELPATH",
    "TERMINAL_STATE_RELPATH",
    "TransactionAccountingEvidence",
    "V83_NATIVE_MEMORY_AUDIT_PIN",
    "V83_NATIVE_MEMORY_AUDIT_RELPATH",
    "write_production_result_evidence",
    "write_frozen_phase_plan_artifact",
    "write_immutable_raw_evidence_artifact",
]
