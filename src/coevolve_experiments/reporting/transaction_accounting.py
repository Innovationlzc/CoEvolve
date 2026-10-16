"""Deterministic accounting reconstructed from immutable physical receipts.

This module is deliberately a pure reporting boundary.  It neither executes a
policy nor writes an artifact.  Instead it reloads the complete physical-
transaction chain, delegates each transaction's raw MuJoCo validation to the
shared terminal-evidence validator, and derives every count that appears in
the paper tables.

Caller-authored metrics are never evidence.  They may be supplied only as a
redundant consistency claim.  The authoritative result is reconstructed from
the hash-bound receipts, frozen phase plan, finalized accepted-checkpoint
manifest, and capability-bound mutation-commit receipts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..serialization import canonical_json, strict_json_loads
from .evidence_artifacts import (
    BoundArtifactError,
    BoundJsonArtifact,
    load_bound_json_artifact,
)
from .terminal_aux_evidence import (
    PAYLOAD_DROP_EVALUATION_KIND,
    capability_bound_auxiliary_producer_allowlist,
    validate_physical_transaction_receipt,
)


TRANSACTION_ACCOUNTING_LEDGER_KIND = "coevolve_transaction_accounting_ledger"
TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION = 1
TRANSACTION_ACCOUNTING_LEDGER_RELPATH = "evidence/transaction_accounting_ledger.json"
TRANSACTION_ACCOUNTING_LEDGER_PIN = "transaction_accounting_ledger_sha256"
ACCEPTED_MANIFEST_RELPATH = "memory/v83_native/accepted_episode_manifest.json"
ACCEPTED_MANIFEST_PIN = "accepted_episode_manifest_sha256"
FROZEN_PHASE_PLAN_RELPATH = "evidence/accounting/frozen_phase_plan.json"
FROZEN_PHASE_PLAN_PIN = "frozen_phase_plan_sha256"
FROZEN_PHASE_PLAN_KIND = "coevolve_frozen_verifier_phase_plan"
FROZEN_PHASE_PLAN_SCHEMA_VERSION = 1
MUTATION_COMMIT_RECEIPT_KIND = "coevolve_mutation_commit_receipt"
MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION = 1
MUTATION_COMMIT_PRODUCER_ID = "praxis_coop.production.mutation.commit"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_RECEIPT_FILENAME_RE = re.compile(r"^transaction_(\d+)\.json$")
_CANONICAL_MUTATION_FILENAME_RE = re.compile(r"^mutation_commit_(\d+)\.json$")
_ARTIFACT_REF_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})
_PHASE_PLAN_FIELDS = frozenset(
    {"kind", "schema_version", "run_id", "phase_ids"}
)
_ACCEPTED_MANIFEST_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "initial_checkpoint",
        "latest_checkpoint",
        "initial_state",
        "steps",
    }
)
_ACCEPTED_STEP_REQUIRED_FIELDS = frozenset(
    {"index", "checkpoint_before", "checkpoint_after", "transition_outcome"}
)
_MUTATION_COMMIT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "mutation_ordinal",
        "source_rejected_physical_transaction",
        "checkpoint_id",
        "mutation_level",
        "committed_mutation_sha256",
        "previous_commit_receipt_sha256",
    }
)
_MUTATION_LEVELS = frozenset(
    {"parameter_tuning", "coordination_mutation", "strategy_mutation"}
)
_ACCEPTED_OUTCOMES = frozenset(
    {"ACCEPT_SETUP", "ACCEPT_PROGRESS", "TASK_COMPLETE"}
)
_CLAIMED_ACCOUNTING_FIELDS = frozenset(
    {
        "physical_transactions",
        "accepted_transactions",
        "rejected_transactions",
        "accepted_setup_transactions",
        "accepted_progress_transactions",
        "phase_completed",
        "phase_total",
        "failure_counts",
        "mutation_counts",
    }
)


@dataclass(frozen=True, slots=True)
class TransactionAccountingDerivation:
    """Immutable derived ledger plus all transitively consumed artifacts."""

    ledger: Mapping[str, Any] | None = None
    artifact_hashes: Mapping[str, str] | None = None
    source_artifacts: tuple[BoundJsonArtifact, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.ledger is not None and not self.invalid_reasons


@dataclass(frozen=True, slots=True)
class TransactionAccountingEvidenceValidation:
    """Collector-side validation of one manifest-published accounting ledger."""

    published_ledger: Mapping[str, Any] | None = None
    recomputed_ledger: Mapping[str, Any] | None = None
    artifact_hashes: Mapping[str, str] | None = None
    source_artifacts: tuple[BoundJsonArtifact, ...] = ()
    mismatched_fields: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return (
            self.published_ledger is not None
            and self.recomputed_ledger is not None
            and not self.mismatched_fields
            and not self.invalid_reasons
        )


def recompute_transaction_accounting_ledger(
    run_root: str | Path,
    *,
    transaction_receipt_refs: Sequence[Mapping[str, Any]],
    phase_plan_ref: Mapping[str, Any],
    accepted_manifest_ref: Mapping[str, Any],
    mutation_commit_receipt_refs: Sequence[Mapping[str, Any]] = (),
    expected_run_id: str,
    expected_physical_transactions: int,
    physical_budget: int,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    expected_producers: Mapping[str, Mapping[str, str]],
    claimed_accounting: Mapping[str, Any] | None = None,
) -> TransactionAccountingDerivation:
    """Reconstruct paper accounting without trusting aggregate counters.

    Receipt references must cover physical transactions ``1..N`` in order.
    ``N=0`` is supported for a genuine pre-dispatch SYSTEM_STALL, but its
    accepted manifest must consequently contain no steps or mutations.
    """

    errors: list[str] = []
    sources: list[BoundJsonArtifact] = []
    hashes: dict[str, str] = {}

    receipt_refs = _sequence(transaction_receipt_refs)
    mutation_refs = _sequence(mutation_commit_receipt_refs)
    if receipt_refs is None:
        errors.append("TRANSACTION_ACCOUNTING_RECEIPT_REFS_INVALID")
        receipt_refs = ()
    if mutation_refs is None:
        errors.append("TRANSACTION_ACCOUNTING_MUTATION_REFS_INVALID")
        mutation_refs = ()
    expected_count = _nonnegative_integer(expected_physical_transactions)
    budget = _positive_integer(physical_budget)
    if expected_count is None:
        errors.append("TRANSACTION_ACCOUNTING_EXPECTED_PHYSICAL_COUNT_INVALID")
        expected_count = len(receipt_refs)
    if budget is None:
        errors.append("TRANSACTION_ACCOUNTING_PHYSICAL_BUDGET_INVALID")
    elif expected_count > budget:
        errors.append("TRANSACTION_ACCOUNTING_PHYSICAL_BUDGET_EXCEEDED")
    if len(receipt_refs) != expected_count:
        errors.append("TRANSACTION_ACCOUNTING_RECEIPT_COVERAGE_INCOMPLETE")
    if not _nonempty(expected_run_id):
        errors.append("TRANSACTION_ACCOUNTING_RUN_ID_INVALID")
    if not _sha(expected_compiled_model_sha256):
        errors.append("TRANSACTION_ACCOUNTING_COMPILED_MODEL_SHA_INVALID")
    if not _sha(expected_scene_realization_sha256):
        errors.append("TRANSACTION_ACCOUNTING_SCENE_REALIZATION_SHA_INVALID")

    capability_sha = _validate_capability_allowlist(expected_producers, errors)
    phase_artifact = _load_source(
        run_root,
        phase_plan_ref,
        "TRANSACTION_ACCOUNTING_PHASE_PLAN",
        errors,
        hashes,
        sources,
    )
    manifest_artifact = _load_source(
        run_root,
        accepted_manifest_ref,
        "TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST",
        errors,
        hashes,
        sources,
    )
    phase_ids = _validate_phase_plan(
        phase_artifact.payload if phase_artifact is not None else None,
        expected_run_id=expected_run_id,
        errors=errors,
    )
    manifest_steps, initial_checkpoint, latest_checkpoint = _validate_manifest(
        manifest_artifact.payload if manifest_artifact is not None else None,
        errors=errors,
    )

    transactions: list[dict[str, Any]] = []
    accepted_for_manifest: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    accepted_count = 0
    rejected_count = 0
    setup_count = 0
    progress_count = 0
    phase_cursor: tuple[str, ...] = ()
    checkpoint_cursor = ""
    checkpoint_initial = ""
    checkpoint_event_cursor = ""
    previous_receipt_sha: str | None = None
    drop_counter_cursor = 0
    maximum_drop_streak = 0
    receipt_artifacts: list[BoundJsonArtifact] = []
    transaction_by_index: dict[int, dict[str, Any]] = {}

    for transaction, ref in enumerate(receipt_refs, start=1):
        validation = validate_physical_transaction_receipt(
            run_root,
            receipt_ref=ref,
            expected_run_id=expected_run_id,
            expected_transaction=transaction,
            expected_compiled_model_sha256=expected_compiled_model_sha256,
            expected_scene_realization_sha256=expected_scene_realization_sha256,
            expected_producers=expected_producers,
        )
        errors.extend(validation.invalid_reasons)
        for artifact in validation.source_artifacts:
            _record_source(artifact, hashes, sources, errors)
        receipt_artifact = _artifact_at_path(
            validation.source_artifacts,
            f"evidence/transactions/transaction_{transaction:03d}.json",
        )
        if receipt_artifact is None:
            errors.append(
                f"TRANSACTION_ACCOUNTING_TRANSACTION_{transaction}_RECEIPT_MISSING"
            )
            continue
        receipt_artifacts.append(receipt_artifact)
        receipt = validation.receipt
        evaluation = _transaction_evaluation(
            validation.source_artifacts,
            transaction,
        )
        if receipt is None or evaluation is None:
            errors.append(
                f"TRANSACTION_ACCOUNTING_TRANSACTION_{transaction}_NOT_DERIVABLE"
            )
            previous_receipt_sha = receipt_artifact.sha256
            continue

        label = f"TRANSACTION_ACCOUNTING_TRANSACTION_{transaction}"
        if receipt.get("previous_receipt_sha256") != previous_receipt_sha:
            errors.append(f"{label}_RECEIPT_HASH_CHAIN_MISMATCH")
        previous_receipt_sha = receipt_artifact.sha256

        before_checkpoint = _text(receipt.get("checkpoint_id_before"))
        after_checkpoint = _text(receipt.get("checkpoint_id_after"))
        before_event = _text(receipt.get("checkpoint_event_id_before"))
        after_event = _text(receipt.get("checkpoint_event_id_after"))
        if not checkpoint_initial:
            checkpoint_initial = before_checkpoint
            checkpoint_cursor = before_checkpoint
            checkpoint_event_cursor = before_event
        if before_checkpoint != checkpoint_cursor:
            errors.append(f"{label}_CHECKPOINT_CHAIN_MISMATCH")
        if before_event != checkpoint_event_cursor:
            errors.append(f"{label}_CHECKPOINT_EVENT_CHAIN_MISMATCH")

        outcome = _text(evaluation.get("transition_outcome"))
        accepted = outcome in _ACCEPTED_OUTCOMES
        verifier = receipt.get("verifier")
        verifier_accepted = (
            verifier.get("accepted") if isinstance(verifier, Mapping) else None
        )
        failure_type = _text(
            verifier.get("failure_type") if isinstance(verifier, Mapping) else None
        )
        if verifier_accepted is not accepted:
            errors.append(f"{label}_OUTCOME_ACCEPTANCE_MISMATCH")
        if accepted:
            accepted_count += 1
            if before_checkpoint == after_checkpoint:
                errors.append(f"{label}_ACCEPTED_CHECKPOINT_NOT_ADVANCED")
            if receipt.get("rollback_performed") is not False:
                errors.append(f"{label}_ACCEPTED_ROLLBACK_INVALID")
            if outcome == "ACCEPT_SETUP":
                setup_count += 1
            else:
                progress_count += 1
            accepted_for_manifest.append(
                {
                    "physical_transaction": transaction,
                    "checkpoint_before": before_checkpoint,
                    "checkpoint_after": after_checkpoint,
                    "transition_outcome": outcome,
                }
            )
        else:
            rejected_count += 1
            if before_checkpoint != after_checkpoint:
                errors.append(f"{label}_REJECTED_CHECKPOINT_CHANGED")
            if receipt.get("rollback_performed") is not True:
                errors.append(f"{label}_REJECTED_ROLLBACK_MISSING")
            if failure_type:
                failure_counts[failure_type] += 1

        before_phases = _ordered_strings(evaluation.get("completed_phase_ids_before"))
        after_phases = _ordered_strings(evaluation.get("completed_phase_ids_after"))
        if before_phases is None or after_phases is None:
            errors.append(f"{label}_COMPLETED_PHASE_IDS_INVALID")
            before_phases = phase_cursor
            after_phases = phase_cursor
        if before_phases != phase_cursor:
            errors.append(f"{label}_PHASE_HISTORY_CHAIN_MISMATCH")
        if not _is_prefix(before_phases, phase_ids):
            errors.append(f"{label}_PHASE_BEFORE_NOT_PLAN_PREFIX")
        if not _is_prefix(after_phases, phase_ids):
            errors.append(f"{label}_PHASE_AFTER_NOT_PLAN_PREFIX")
        if not _is_prefix(before_phases, after_phases):
            errors.append(f"{label}_PHASE_HISTORY_REGRESSION")
        if outcome in {"REJECT", "ACCEPT_SETUP"} and after_phases != before_phases:
            errors.append(f"{label}_NONPROGRESS_PHASE_CHANGED")
        if outcome == "TASK_COMPLETE":
            if after_phases != phase_ids:
                errors.append(f"{label}_TASK_COMPLETE_PHASES_INCOMPLETE")
            if transaction != len(receipt_refs):
                errors.append(f"{label}_TASK_COMPLETE_NOT_FINAL")

        raw_drop = bool(
            validation.physical_derivation is not None
            and validation.physical_derivation.payload_drop
        )
        counter_before = _nonnegative_integer(receipt.get("drop_counter_before"))
        counter_after = _nonnegative_integer(receipt.get("drop_counter_after"))
        derived_counter_before = drop_counter_cursor
        expected_counter_after = derived_counter_before + 1 if raw_drop else 0
        if counter_before != derived_counter_before:
            errors.append(f"{label}_DROP_COUNTER_CHAIN_MISMATCH")
        if counter_after != expected_counter_after:
            errors.append(f"{label}_DROP_COUNTER_NOT_RAW_DERIVED")
        drop_counter_cursor = expected_counter_after
        maximum_drop_streak = max(maximum_drop_streak, drop_counter_cursor)
        if raw_drop and failure_type != "payload_drop":
            errors.append(f"{label}_RAW_DROP_FAILURE_CLASS_MISMATCH")

        record = {
            "physical_transaction": transaction,
            "source_receipt_sha256": receipt_artifact.sha256,
            "checkpoint_id_before": before_checkpoint,
            "checkpoint_event_id_before": before_event,
            "checkpoint_id_after": after_checkpoint,
            "checkpoint_event_id_after": after_event,
            "accepted": accepted,
            "transition_outcome": outcome,
            "failure_type": failure_type,
            "completed_phase_ids_before": list(before_phases),
            "completed_phase_ids_after": list(after_phases),
            "raw_payload_drop": raw_drop,
            "drop_counter_before": derived_counter_before,
            "drop_counter_after": expected_counter_after,
        }
        transactions.append(record)
        transaction_by_index[transaction] = record
        checkpoint_cursor = after_checkpoint
        checkpoint_event_cursor = after_event
        phase_cursor = after_phases

    if len(receipt_artifacts) != len(receipt_refs):
        errors.append("TRANSACTION_ACCOUNTING_RECEIPT_ARTIFACT_SET_INCOMPLETE")
    _crosscheck_accepted_manifest(
        manifest_steps,
        accepted_for_manifest,
        manifest_initial=initial_checkpoint,
        manifest_latest=latest_checkpoint,
        receipt_initial=checkpoint_initial,
        receipt_latest=checkpoint_cursor,
        expected_physical_transactions=expected_count,
        errors=errors,
    )

    mutation_documents, mutation_counts, mutation_artifacts = _derive_mutations(
        run_root,
        mutation_refs,
        expected_run_id=expected_run_id,
        capability_sha=capability_sha,
        transaction_by_index=transaction_by_index,
        errors=errors,
        hashes=hashes,
        sources=sources,
    )
    if expected_count == 0 and mutation_documents:
        errors.append("TRANSACTION_ACCOUNTING_ZERO_TRANSACTION_MUTATION_INVALID")

    derived_claim = {
        "physical_transactions": len(receipt_refs),
        "accepted_transactions": accepted_count,
        "rejected_transactions": rejected_count,
        "accepted_setup_transactions": setup_count,
        "accepted_progress_transactions": progress_count,
        "phase_completed": len(phase_cursor),
        "phase_total": len(phase_ids),
        "failure_counts": dict(sorted(failure_counts.items())),
        "mutation_counts": dict(sorted(mutation_counts.items())),
    }
    _crosscheck_claimed_accounting(claimed_accounting, derived_claim, errors)

    ledger_document = {
        "kind": TRANSACTION_ACCOUNTING_LEDGER_KIND,
        "schema_version": TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION,
        "run_id": expected_run_id,
        "physical_budget": physical_budget,
        **derived_claim,
        "phase_completion": {
            "completed_phase_ids": list(phase_cursor),
            "completed": len(phase_cursor),
            "total": len(phase_ids),
            "fraction": (
                len(phase_cursor) / len(phase_ids) if phase_ids else 0.0
            ),
        },
        "checkpoint_lineage": {
            "initial_checkpoint_id": initial_checkpoint,
            "latest_checkpoint_id": latest_checkpoint,
        },
        "drop_accounting": {
            "final_consecutive_drop_count": drop_counter_cursor,
            "maximum_consecutive_drop_count": maximum_drop_streak,
        },
        "receipt_chain": {
            "coverage_start_transaction": 1 if receipt_artifacts else None,
            "coverage_end_transaction": len(receipt_refs) if receipt_artifacts else 0,
            "head_sha256": receipt_artifacts[0].sha256 if receipt_artifacts else None,
            "tail_sha256": receipt_artifacts[-1].sha256 if receipt_artifacts else None,
        },
        "source_phase_plan": (
            phase_artifact.public_ref() if phase_artifact is not None else None
        ),
        "source_accepted_manifest": (
            manifest_artifact.public_ref() if manifest_artifact is not None else None
        ),
        "source_transaction_receipts": [
            artifact.public_ref() for artifact in receipt_artifacts
        ],
        "source_mutation_commit_receipts": [
            artifact.public_ref() for artifact in mutation_artifacts
        ],
        "transactions": transactions,
        "mutations": mutation_documents,
    }
    ledger = _deep_freeze(ledger_document)
    return TransactionAccountingDerivation(
        ledger=ledger,
        artifact_hashes=MappingProxyType(dict(sorted(hashes.items()))),
        source_artifacts=tuple(sources),
        invalid_reasons=tuple(dict.fromkeys(errors)),
    )


def validate_transaction_accounting_evidence(
    run_root: str | Path,
    *,
    expected_run_id: str,
    physical_budget: int,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    expected_producers: Mapping[str, Mapping[str, str]],
    claimed_accounting: Mapping[str, Any] | None = None,
) -> TransactionAccountingEvidenceValidation:
    """Reload a published ledger and independently reconstruct every field.

    The publication marker is the canonical ``run_manifest.json``.  Its three
    required result pins bind the ledger, finalized native accepted manifest,
    and frozen verifier phase plan.  Receipt and mutation references are then
    taken only from the pinned ledger and passed back through
    :func:`recompute_transaction_accounting_ledger`.
    """

    root = Path(run_root)
    errors: list[str] = []
    mismatches: list[str] = []
    published: Mapping[str, Any] | None = None
    recomputed: Mapping[str, Any] | None = None
    source_artifacts: tuple[BoundJsonArtifact, ...] = ()
    artifact_hashes: Mapping[str, str] = MappingProxyType({})

    if root.is_symlink() or not root.is_dir():
        errors.append("TRANSACTION_ACCOUNTING_RUN_ROOT_INVALID")
    manifest = _read_canonical_json(
        root / "run_manifest.json",
        "TRANSACTION_ACCOUNTING_RUN_MANIFEST",
        errors,
        root=root,
    )
    result_hashes = (
        manifest.get("result_artifact_hashes")
        if isinstance(manifest, Mapping)
        else None
    )
    if not isinstance(result_hashes, Mapping):
        errors.append("TRANSACTION_ACCOUNTING_RUN_MANIFEST_PINS_INVALID")
        result_hashes = {}
    experiment = manifest.get("experiment") if isinstance(manifest, Mapping) else None
    manifest_run_id = (
        experiment.get("run_id") if isinstance(experiment, Mapping) else None
    )
    if manifest_run_id != expected_run_id:
        errors.append("TRANSACTION_ACCOUNTING_RUN_MANIFEST_RUN_MISMATCH")

    ledger_pin = _required_pin(
        result_hashes,
        TRANSACTION_ACCOUNTING_LEDGER_PIN,
        errors,
    )
    accepted_pin = _required_pin(
        result_hashes,
        ACCEPTED_MANIFEST_PIN,
        errors,
    )
    phase_pin = _required_pin(
        result_hashes,
        FROZEN_PHASE_PLAN_PIN,
        errors,
    )
    published = _read_pinned_canonical_json(
        root / TRANSACTION_ACCOUNTING_LEDGER_RELPATH,
        expected_sha256=ledger_pin,
        label="TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER",
        errors=errors,
        root=root,
    )
    if published is None:
        return TransactionAccountingEvidenceValidation(
            invalid_reasons=tuple(dict.fromkeys(errors))
        )
    if (
        published.get("kind") != TRANSACTION_ACCOUNTING_LEDGER_KIND
        or published.get("schema_version")
        != TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION
        or published.get("run_id") != expected_run_id
    ):
        errors.append("TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_IDENTITY_INVALID")

    phase_ref = _source_ref(
        published.get("source_phase_plan"),
        expected_relative_path=FROZEN_PHASE_PLAN_RELPATH,
        expected_sha256=phase_pin,
        label="TRANSACTION_ACCOUNTING_PUBLISHED_PHASE_PLAN_REF",
        errors=errors,
    )
    accepted_ref = _source_ref(
        published.get("source_accepted_manifest"),
        expected_relative_path=ACCEPTED_MANIFEST_RELPATH,
        expected_sha256=accepted_pin,
        label="TRANSACTION_ACCOUNTING_PUBLISHED_ACCEPTED_MANIFEST_REF",
        errors=errors,
    )
    transaction_refs = _source_ref_sequence(
        published.get("source_transaction_receipts"),
        "TRANSACTION_ACCOUNTING_PUBLISHED_TRANSACTION_REFS",
        errors,
    )
    mutation_refs = _source_ref_sequence(
        published.get("source_mutation_commit_receipts"),
        "TRANSACTION_ACCOUNTING_PUBLISHED_MUTATION_REFS",
        errors,
    )
    _validate_canonical_receipt_directory(
        root,
        transaction_refs,
        errors,
    )
    _validate_canonical_mutation_directory(
        root,
        mutation_refs,
        errors,
    )

    if phase_ref is not None and accepted_ref is not None:
        derivation = recompute_transaction_accounting_ledger(
            root,
            transaction_receipt_refs=transaction_refs,
            phase_plan_ref=phase_ref,
            accepted_manifest_ref=accepted_ref,
            mutation_commit_receipt_refs=mutation_refs,
            expected_run_id=expected_run_id,
            expected_physical_transactions=len(transaction_refs),
            physical_budget=physical_budget,
            expected_compiled_model_sha256=expected_compiled_model_sha256,
            expected_scene_realization_sha256=expected_scene_realization_sha256,
            expected_producers=expected_producers,
            claimed_accounting=claimed_accounting,
        )
        errors.extend(derivation.invalid_reasons)
        recomputed = derivation.ledger
        source_artifacts = derivation.source_artifacts
        artifact_hashes = derivation.artifact_hashes or MappingProxyType({})
        if recomputed is not None:
            mismatches.extend(_mismatched_document_fields(published, recomputed))
            if canonical_json(published) != canonical_json(recomputed):
                errors.append(
                    "TRANSACTION_ACCOUNTING_PUBLISHED_LEDGER_NOT_RECOMPUTED"
                )
            elif mismatches:
                # Defensive: recursive comparison and canonical equivalence
                # should agree, including strict distinctions between bool/int.
                errors.append(
                    "TRANSACTION_ACCOUNTING_LEDGER_COMPARISON_INCONSISTENT"
                )

    return TransactionAccountingEvidenceValidation(
        published_ledger=_deep_freeze(published),
        recomputed_ledger=recomputed,
        artifact_hashes=artifact_hashes,
        source_artifacts=source_artifacts,
        mismatched_fields=tuple(dict.fromkeys(mismatches)),
        invalid_reasons=tuple(dict.fromkeys(errors)),
    )


def _read_canonical_json(
    path: Path,
    label: str,
    errors: list[str],
    *,
    root: Path,
) -> Mapping[str, Any] | None:
    if _unsafe_fixed_path(path, root):
        errors.append(f"{label}_MISSING_OR_UNSAFE")
        return None
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        errors.append(f"{label}_NOT_STRICT_JSON")
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{label}_ROOT_INVALID")
        return None
    return value


def _read_pinned_canonical_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    errors: list[str],
    root: Path,
) -> Mapping[str, Any] | None:
    if not _sha(expected_sha256):
        errors.append(f"{label}_PIN_INVALID")
        return None
    if _unsafe_fixed_path(path, root):
        errors.append(f"{label}_MISSING_OR_UNSAFE")
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        errors.append(f"{label}_MISSING_OR_UNSAFE")
        return None
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        errors.append(f"{label}_PIN_MISMATCH")
        return None
    try:
        value = strict_json_loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        errors.append(f"{label}_NOT_STRICT_JSON")
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{label}_ROOT_INVALID")
        return None
    return value


def _unsafe_fixed_path(path: Path, root: Path) -> bool:
    if root.is_symlink() or not root.is_dir():
        return True
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return not path.is_file()


def _required_pin(
    result_hashes: Mapping[str, Any],
    name: str,
    errors: list[str],
) -> str:
    value = result_hashes.get(name)
    if not _sha(value):
        errors.append(f"TRANSACTION_ACCOUNTING_RUN_MANIFEST_{name.upper()}_INVALID")
        return ""
    return str(value)


def _source_ref(
    value: object,
    *,
    expected_relative_path: str,
    expected_sha256: str,
    label: str,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_REF_FIELDS:
        errors.append(f"{label}_INVALID")
        return None
    if value.get("relative_path") != expected_relative_path:
        errors.append(f"{label}_PATH_MISMATCH")
    if value.get("sha256") != expected_sha256:
        errors.append(f"{label}_PIN_MISMATCH")
    size = value.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        errors.append(f"{label}_SIZE_INVALID")
    return value


def _source_ref_sequence(
    value: object,
    label: str,
    errors: list[str],
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        errors.append(f"{label}_INVALID")
        return ()
    result: list[Mapping[str, Any]] = []
    for ordinal, ref in enumerate(value, start=1):
        if not isinstance(ref, Mapping) or set(ref) != _ARTIFACT_REF_FIELDS:
            errors.append(f"{label}_{ordinal}_INVALID")
            continue
        result.append(ref)
    if len(result) != len(value):
        errors.append(f"{label}_INCOMPLETE")
    return tuple(result)


def _validate_canonical_receipt_directory(
    root: Path,
    refs: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    directory = root / "evidence" / "transactions"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        errors.append("TRANSACTION_ACCOUNTING_RECEIPT_DIRECTORY_INVALID")
        return
    if not directory.exists():
        if refs:
            errors.append("TRANSACTION_ACCOUNTING_RECEIPT_DIRECTORY_MISSING")
        return
    discovered_paths: set[str] = set()
    discovered_indices: set[int] = set()
    for path in directory.iterdir():
        match = _CANONICAL_RECEIPT_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            errors.append("TRANSACTION_ACCOUNTING_CANONICAL_RECEIPT_UNSAFE")
            continue
        digits = match.group(1)
        if len(digits) != 3:
            errors.append("TRANSACTION_ACCOUNTING_RECEIPT_FILENAME_NONCANONICAL")
            continue
        transaction = int(digits)
        if transaction < 1:
            errors.append("TRANSACTION_ACCOUNTING_RECEIPT_FILENAME_NONCANONICAL")
            continue
        discovered_indices.add(transaction)
        discovered_paths.add(path.relative_to(root).as_posix())
    expected_indices = set(range(1, len(refs) + 1))
    referenced_paths = {
        str(ref.get("relative_path"))
        for ref in refs
        if isinstance(ref.get("relative_path"), str)
    }
    if discovered_indices - expected_indices:
        errors.append("TRANSACTION_ACCOUNTING_UNREFERENCED_RECEIPT_TAIL")
    if expected_indices - discovered_indices:
        errors.append("TRANSACTION_ACCOUNTING_CANONICAL_RECEIPT_GAP")
    if discovered_paths - referenced_paths:
        errors.append("TRANSACTION_ACCOUNTING_UNREFERENCED_CANONICAL_RECEIPT")
    if referenced_paths - discovered_paths:
        errors.append("TRANSACTION_ACCOUNTING_REFERENCED_RECEIPT_NOT_CANONICAL")


def _validate_canonical_mutation_directory(
    root: Path,
    refs: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> None:
    directory = root / "evidence" / "mutations"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        errors.append("TRANSACTION_ACCOUNTING_MUTATION_DIRECTORY_INVALID")
        return
    if not directory.exists():
        if refs:
            errors.append("TRANSACTION_ACCOUNTING_MUTATION_DIRECTORY_MISSING")
        return
    discovered_paths: set[str] = set()
    discovered_indices: set[int] = set()
    for path in directory.iterdir():
        match = _CANONICAL_MUTATION_FILENAME_RE.fullmatch(path.name)
        if match is None:
            continue
        if path.is_symlink() or not path.is_file():
            errors.append("TRANSACTION_ACCOUNTING_CANONICAL_MUTATION_UNSAFE")
            continue
        digits = match.group(1)
        if len(digits) != 3 or int(digits) < 1:
            errors.append("TRANSACTION_ACCOUNTING_MUTATION_FILENAME_NONCANONICAL")
            continue
        discovered_indices.add(int(digits))
        discovered_paths.add(path.relative_to(root).as_posix())
    expected_indices = set(range(1, len(refs) + 1))
    referenced_paths = {
        str(ref.get("relative_path"))
        for ref in refs
        if isinstance(ref.get("relative_path"), str)
    }
    if discovered_indices - expected_indices:
        errors.append("TRANSACTION_ACCOUNTING_UNREFERENCED_MUTATION_TAIL")
    if expected_indices - discovered_indices:
        errors.append("TRANSACTION_ACCOUNTING_CANONICAL_MUTATION_GAP")
    if discovered_paths - referenced_paths:
        errors.append("TRANSACTION_ACCOUNTING_UNREFERENCED_CANONICAL_MUTATION")
    if referenced_paths - discovered_paths:
        errors.append("TRANSACTION_ACCOUNTING_REFERENCED_MUTATION_NOT_CANONICAL")


def _mismatched_document_fields(left: Any, right: Any, path: str = "$") -> tuple[str, ...]:
    differences: list[str] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right), key=str)
        for key in keys:
            child_path = f"{path}.{key}"
            if key not in left or key not in right:
                differences.append(child_path)
            else:
                differences.extend(
                    _mismatched_document_fields(left[key], right[key], child_path)
                )
        return tuple(differences)
    left_sequence = isinstance(left, (list, tuple))
    right_sequence = isinstance(right, (list, tuple))
    if left_sequence and right_sequence:
        if len(left) != len(right):
            differences.append(f"{path}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(
                _mismatched_document_fields(
                    left_item,
                    right_item,
                    f"{path}[{index}]",
                )
            )
        return tuple(differences)
    if type(left) is not type(right) or left != right:
        differences.append(path)
    return tuple(differences)


def _validate_capability_allowlist(
    value: Mapping[str, Mapping[str, str]],
    errors: list[str],
) -> str:
    if not isinstance(value, Mapping):
        errors.append("TRANSACTION_ACCOUNTING_PRODUCER_ALLOWLIST_INVALID")
        return ""
    anchor = value.get("payload_drop.transaction_receipt")
    capability_sha = (
        anchor.get("source_sha256") if isinstance(anchor, Mapping) else None
    )
    if not _sha(capability_sha):
        errors.append("TRANSACTION_ACCOUNTING_CAPABILITY_SHA_INVALID")
        return ""
    try:
        expected = capability_bound_auxiliary_producer_allowlist(str(capability_sha))
    except ValueError:
        errors.append("TRANSACTION_ACCOUNTING_CAPABILITY_SHA_INVALID")
        return ""
    normalized = {
        str(family): dict(producer) if isinstance(producer, Mapping) else producer
        for family, producer in value.items()
    }
    if normalized != {family: dict(producer) for family, producer in expected.items()}:
        errors.append("TRANSACTION_ACCOUNTING_PRODUCER_ALLOWLIST_NOT_CAPABILITY_BOUND")
    return str(capability_sha)


def _validate_phase_plan(
    value: Mapping[str, Any] | None,
    *,
    expected_run_id: str,
    errors: list[str],
) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        errors.append("TRANSACTION_ACCOUNTING_PHASE_PLAN_INVALID")
        return ()
    if set(value) != _PHASE_PLAN_FIELDS:
        errors.append("TRANSACTION_ACCOUNTING_PHASE_PLAN_SCHEMA_INVALID")
    if (
        value.get("kind") != FROZEN_PHASE_PLAN_KIND
        or value.get("schema_version") != FROZEN_PHASE_PLAN_SCHEMA_VERSION
    ):
        errors.append("TRANSACTION_ACCOUNTING_PHASE_PLAN_IDENTITY_INVALID")
    if value.get("run_id") != expected_run_id:
        errors.append("TRANSACTION_ACCOUNTING_PHASE_PLAN_RUN_MISMATCH")
    phases = _ordered_strings(value.get("phase_ids"))
    if phases is None or not phases:
        errors.append("TRANSACTION_ACCOUNTING_PHASE_PLAN_PHASES_INVALID")
        return ()
    return phases


def _validate_manifest(
    value: Mapping[str, Any] | None,
    *,
    errors: list[str],
) -> tuple[tuple[Mapping[str, Any], ...], str, str]:
    if not isinstance(value, Mapping):
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_INVALID")
        return (), "", ""
    if set(value) != _ACCEPTED_MANIFEST_FIELDS:
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_SCHEMA_INVALID")
    if value.get("kind") != "accepted_checkpoint_episode" or value.get(
        "schema_version"
    ) != 1:
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_IDENTITY_INVALID")
    initial = _text(value.get("initial_checkpoint"))
    latest = _text(value.get("latest_checkpoint"))
    if not initial or not latest or not isinstance(value.get("initial_state"), Mapping):
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_HEADER_INVALID")
    raw_steps = value.get("steps")
    if not isinstance(raw_steps, list):
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_STEPS_INVALID")
        return (), initial, latest
    steps: list[Mapping[str, Any]] = []
    cursor = initial
    for ordinal, step in enumerate(raw_steps, start=1):
        label = f"TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_STEP_{ordinal}"
        if not isinstance(step, Mapping):
            errors.append(f"{label}_INVALID")
            continue
        if not _ACCEPTED_STEP_REQUIRED_FIELDS.issubset(step):
            errors.append(f"{label}_FIELDS_MISSING")
        if step.get("index") != ordinal:
            errors.append(f"{label}_INDEX_INVALID")
        before = _text(step.get("checkpoint_before"))
        after = _text(step.get("checkpoint_after"))
        if before != cursor or not after or after == before:
            errors.append(f"{label}_CHECKPOINT_LINEAGE_INVALID")
        outcome = _text(step.get("transition_outcome"))
        if outcome not in _ACCEPTED_OUTCOMES:
            errors.append(f"{label}_OUTCOME_INVALID")
        cursor = after
        steps.append(step)
    if cursor != latest:
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_LATEST_MISMATCH")
    return tuple(steps), initial, latest


def _crosscheck_accepted_manifest(
    manifest_steps: Sequence[Mapping[str, Any]],
    accepted: Sequence[Mapping[str, Any]],
    *,
    manifest_initial: str,
    manifest_latest: str,
    receipt_initial: str,
    receipt_latest: str,
    expected_physical_transactions: int,
    errors: list[str],
) -> None:
    if manifest_initial != receipt_initial and not (
        expected_physical_transactions == 0 and not receipt_initial
    ):
        errors.append("TRANSACTION_ACCOUNTING_MANIFEST_INITIAL_CHECKPOINT_MISMATCH")
    if expected_physical_transactions == 0:
        if manifest_steps:
            errors.append("TRANSACTION_ACCOUNTING_ZERO_TRANSACTION_MANIFEST_NOT_EMPTY")
        if manifest_latest != manifest_initial:
            errors.append("TRANSACTION_ACCOUNTING_ZERO_TRANSACTION_CHECKPOINT_MISMATCH")
        return
    if manifest_latest != receipt_latest:
        errors.append("TRANSACTION_ACCOUNTING_MANIFEST_LATEST_CHECKPOINT_MISMATCH")
    if len(manifest_steps) != len(accepted):
        errors.append("TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_COUNT_MISMATCH")
    for ordinal, (step, transaction) in enumerate(
        zip(manifest_steps, accepted),
        start=1,
    ):
        if any(
            step.get(field) != transaction.get(field)
            for field in (
                "checkpoint_before",
                "checkpoint_after",
                "transition_outcome",
            )
        ):
            errors.append(
                f"TRANSACTION_ACCOUNTING_ACCEPTED_MANIFEST_STEP_{ordinal}_MISMATCH"
            )


def _derive_mutations(
    run_root: str | Path,
    refs: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str,
    capability_sha: str,
    transaction_by_index: Mapping[int, Mapping[str, Any]],
    errors: list[str],
    hashes: dict[str, str],
    sources: list[BoundJsonArtifact],
) -> tuple[list[dict[str, Any]], Counter[str], list[BoundJsonArtifact]]:
    documents: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    artifacts: list[BoundJsonArtifact] = []
    previous_sha: str | None = None
    for ordinal, ref in enumerate(refs, start=1):
        label = f"TRANSACTION_ACCOUNTING_MUTATION_COMMIT_{ordinal}"
        artifact = _load_source(
            run_root,
            ref,
            label,
            errors,
            hashes,
            sources,
        )
        if artifact is None:
            continue
        artifacts.append(artifact)
        expected_path = f"evidence/mutations/mutation_commit_{ordinal:03d}.json"
        if artifact.relative_path != expected_path:
            errors.append(f"{label}_PATH_INVALID")
        value = artifact.payload
        if set(value) != _MUTATION_COMMIT_FIELDS:
            errors.append(f"{label}_SCHEMA_INVALID")
        if (
            value.get("kind") != MUTATION_COMMIT_RECEIPT_KIND
            or value.get("schema_version") != MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION
        ):
            errors.append(f"{label}_IDENTITY_INVALID")
        producer = value.get("producer")
        if not isinstance(producer, Mapping) or set(producer) != {
            "id",
            "source_sha256",
        } or producer.get("id") != MUTATION_COMMIT_PRODUCER_ID or producer.get(
            "source_sha256"
        ) != capability_sha:
            errors.append(f"{label}_PRODUCER_NOT_CAPABILITY_BOUND")
        if value.get("run_id") != expected_run_id:
            errors.append(f"{label}_RUN_MISMATCH")
        if value.get("mutation_ordinal") != ordinal:
            errors.append(f"{label}_ORDINAL_INVALID")
        if value.get("previous_commit_receipt_sha256") != previous_sha:
            errors.append(f"{label}_HASH_CHAIN_MISMATCH")
        previous_sha = artifact.sha256
        level = _text(value.get("mutation_level"))
        if level not in _MUTATION_LEVELS:
            errors.append(f"{label}_LEVEL_INVALID")
        committed_sha = value.get("committed_mutation_sha256")
        if not _sha(committed_sha):
            errors.append(f"{label}_COMMIT_SHA_INVALID")
        source_transaction = _positive_integer(
            value.get("source_rejected_physical_transaction")
        )
        transaction = (
            transaction_by_index.get(source_transaction)
            if source_transaction is not None
            else None
        )
        if transaction is None or transaction.get("accepted") is not False:
            errors.append(f"{label}_SOURCE_NOT_REJECTED_TRANSACTION")
        elif value.get("checkpoint_id") != transaction.get("checkpoint_id_after"):
            errors.append(f"{label}_CHECKPOINT_MISMATCH")
        if level in _MUTATION_LEVELS:
            counts[level] += 1
        documents.append(
            {
                "mutation_ordinal": ordinal,
                "source_receipt_sha256": artifact.sha256,
                "source_rejected_physical_transaction": source_transaction,
                "checkpoint_id": _text(value.get("checkpoint_id")),
                "mutation_level": level,
                "committed_mutation_sha256": committed_sha,
            }
        )
    if len(artifacts) != len(refs):
        errors.append("TRANSACTION_ACCOUNTING_MUTATION_RECEIPT_SET_INCOMPLETE")
    return documents, counts, artifacts


def _crosscheck_claimed_accounting(
    claimed: Mapping[str, Any] | None,
    derived: Mapping[str, Any],
    errors: list[str],
) -> None:
    if claimed is None:
        return
    if not isinstance(claimed, Mapping):
        errors.append("TRANSACTION_ACCOUNTING_CLAIM_INVALID")
        return
    missing = _CLAIMED_ACCOUNTING_FIELDS - set(claimed)
    if missing:
        errors.append("TRANSACTION_ACCOUNTING_CLAIM_FIELDS_MISSING")
    for field in sorted(_CLAIMED_ACCOUNTING_FIELDS):
        if field not in claimed:
            continue
        actual = claimed.get(field)
        expected = derived[field]
        if field in {"failure_counts", "mutation_counts"}:
            actual = _count_mapping(actual)
        if actual != expected:
            errors.append(
                f"TRANSACTION_ACCOUNTING_CLAIM_{field.upper()}_MISMATCH"
            )


def _count_mapping(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, int] = {}
    for key, count in value.items():
        if not _nonempty(key) or _nonnegative_integer(count) is None:
            return None
        if int(count) > 0:
            result[str(key)] = int(count)
    return dict(sorted(result.items()))


def _transaction_evaluation(
    artifacts: Sequence[BoundJsonArtifact],
    transaction: int,
) -> Mapping[str, Any] | None:
    matches = [
        artifact.payload
        for artifact in artifacts
        if artifact.payload.get("kind") == PAYLOAD_DROP_EVALUATION_KIND
        and artifact.payload.get("physical_transaction") == transaction
    ]
    return matches[0] if len(matches) == 1 else None


def _artifact_at_path(
    artifacts: Sequence[BoundJsonArtifact],
    relative_path: str,
) -> BoundJsonArtifact | None:
    matches = [item for item in artifacts if item.relative_path == relative_path]
    return matches[0] if len(matches) == 1 else None


def _load_source(
    root: str | Path,
    ref: object,
    label: str,
    errors: list[str],
    hashes: dict[str, str],
    sources: list[BoundJsonArtifact],
) -> BoundJsonArtifact | None:
    try:
        artifact = load_bound_json_artifact(root, ref, label=label)
    except BoundArtifactError:
        errors.append(f"{label}_MISSING_OR_INVALID")
        return None
    _record_source(artifact, hashes, sources, errors)
    return artifact


def _record_source(
    artifact: BoundJsonArtifact,
    hashes: dict[str, str],
    sources: list[BoundJsonArtifact],
    errors: list[str],
) -> None:
    previous = hashes.get(artifact.relative_path)
    if previous is not None:
        if previous != artifact.sha256:
            errors.append("TRANSACTION_ACCOUNTING_SOURCE_PATH_HASH_COLLISION")
        else:
            errors.append("TRANSACTION_ACCOUNTING_SOURCE_ARTIFACT_REUSED")
        return
    hashes[artifact.relative_path] = artifact.sha256
    sources.append(artifact)


def _sequence(value: object) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        result.append(item)
    return tuple(result)


def _ordered_strings(value: object) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or not all(_nonempty(item) for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(str(item) for item in value)


def _is_prefix(prefix: Sequence[str], value: Sequence[str]) -> bool:
    return tuple(value[: len(prefix)]) == tuple(prefix)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _text(value: object) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _nonnegative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _positive_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


__all__ = [
    "ACCEPTED_MANIFEST_PIN",
    "ACCEPTED_MANIFEST_RELPATH",
    "FROZEN_PHASE_PLAN_KIND",
    "FROZEN_PHASE_PLAN_PIN",
    "FROZEN_PHASE_PLAN_RELPATH",
    "FROZEN_PHASE_PLAN_SCHEMA_VERSION",
    "MUTATION_COMMIT_PRODUCER_ID",
    "MUTATION_COMMIT_RECEIPT_KIND",
    "MUTATION_COMMIT_RECEIPT_SCHEMA_VERSION",
    "TRANSACTION_ACCOUNTING_LEDGER_KIND",
    "TRANSACTION_ACCOUNTING_LEDGER_PIN",
    "TRANSACTION_ACCOUNTING_LEDGER_RELPATH",
    "TRANSACTION_ACCOUNTING_LEDGER_SCHEMA_VERSION",
    "TransactionAccountingDerivation",
    "TransactionAccountingEvidenceValidation",
    "recompute_transaction_accounting_ledger",
    "validate_transaction_accounting_evidence",
]
