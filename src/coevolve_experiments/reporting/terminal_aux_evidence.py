"""Hash-bound source evidence for non-success production terminals.

The final result writer is intentionally not an evidence producer.  Runtime
integration code must publish the artifacts described here *before* terminal
finalization.  This module reloads every referenced byte, validates the
producer/source identity and cross-stage bindings, and deterministically
reconstructs the small SYSTEM_STALL or payload-drop ledger that the terminal
receipt indexes.

No function in this module writes a file or trusts a caller-authored terminal
boolean.  Artifact references are always resolved below one run root by the
shared canonical, non-symlink, hash-and-size-bound loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from praxis_coop.cap_policy.contracts import RobotPolicyScript
from praxis_coop.cap_policy.coordination_specs import (
    MODEL_COORDINATION_PRIMITIVE_METHODS,
    coordination_modes_for_method,
)
from praxis_coop.cap_policy.policy_executor import (
    validate_step_policy_code,
    validate_team_policy_scripts,
)
from praxis_coop.cap_policy.static_specs import MODEL_ACTION_SKILL_METHODS

from .evidence_artifacts import (
    BoundArtifactError,
    BoundJsonArtifact,
    load_bound_json_artifact,
)


SYSTEM_STALL_LEDGER_KIND = "coevolve_system_stall_ledger"
SYSTEM_STALL_LEDGER_SCHEMA_VERSION = 4
PAYLOAD_DROP_LEDGER_KIND = "coevolve_payload_drop_ledger"
PAYLOAD_DROP_LEDGER_SCHEMA_VERSION = 4
SYSTEM_STALL_GENERATION_CYCLES = 3
PAYLOAD_DROP_LIMIT = 5
TRUSTED_CHECKPOINT_MIN_PELVIS_HEIGHT_M = 0.35
FROZEN_TASK_MIN_PAYLOAD_Z_M = 0.65
PAYLOAD_DROP_HEIGHT_MARGIN_M = 0.05
PAYLOAD_DROP_Z_THRESHOLD_M = (
    FROZEN_TASK_MIN_PAYLOAD_Z_M - PAYLOAD_DROP_HEIGHT_MARGIN_M
)
PHYSICAL_CONTACT_DISTANCE_TOLERANCE_M = 1.0e-6
MAX_PHYSICAL_CONTACT_FORCE_N = 1.0e6
FROZEN_PAYLOAD_DROP_SAFETY_PROFILE = "frozen_v83_payload_drop_v1"

SYSTEM_STALL_CHECKPOINT_KIND = "coevolve_stall_checkpoint_validity"
SYSTEM_STALL_ACTION_SPACE_KIND = "coevolve_stall_host_action_space"
SYSTEM_STALL_MODEL_OUTPUT_KIND = "coevolve_stall_model_output"
SYSTEM_STALL_STATIC_VALIDATION_KIND = "coevolve_stall_static_validation"
SYSTEM_STALL_HOST_PREFLIGHT_KIND = "coevolve_stall_host_preflight"

# Every physical transaction uses the same receipt envelope.  Payload-drop is
# one independently derived outcome of that envelope, not a special producer
# assertion.  Keep the historical Python name as an alias so callers can
# migrate without silently accepting the old wire identity.
PHYSICAL_TRANSACTION_RECEIPT_KIND = (
    "coevolve_physical_transaction_verifier_receipt"
)
PAYLOAD_DROP_TRANSACTION_RECEIPT_KIND = PHYSICAL_TRANSACTION_RECEIPT_KIND
PAYLOAD_DROP_POLICY_KIND = "coevolve_physical_transaction_policy"
PAYLOAD_DROP_EVALUATION_KIND = "coevolve_physical_transaction_verifier_evaluation"
PAYLOAD_DROP_EXECUTION_TRACE_KIND = "coevolve_physical_transaction_execution_trace"
PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND = "coevolve_physical_transaction_mujoco_samples"
RAW_AUX_EVIDENCE_SCHEMA_VERSION = 2

SYSTEM_STALL_CHECKPOINT_RELPATH = "evidence/system_stall/checkpoint_validity.json"
SYSTEM_STALL_ACTION_SPACE_RELPATH = "evidence/system_stall/host_action_space.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_FIELDS = frozenset({"relative_path", "sha256", "size_bytes"})
_PRODUCER_FIELDS = frozenset({"id", "source_sha256"})
_SIMULATOR_FIELDS = frozenset(
    {"name", "version", "compiled_model_sha256", "scene_realization_sha256"}
)
_POSE_FIELDS = frozenset({"position_xyz_m", "quaternion_wxyz"})
_PHYSICAL_SAMPLE_FIELDS = frozenset(
    {"sample_index", "time_s", "payload_active", "payload_pose", "contacts", "support"}
)
_PHYSICAL_CONTACT_FIELDS = frozenset(
    {
        "geom1_id",
        "geom2_id",
        "geom1_role",
        "geom2_role",
        "geom1_name",
        "geom2_name",
        "body1_name",
        "body2_name",
        "distance_m",
        "normal_force_n",
    }
)
_PHYSICAL_SUPPORT_FIELDS = frozenset(
    {
        "available",
        "supporting_robots",
        "normal_force_n_by_robot",
        "total_normal_force_n",
    }
)
_PHYSICAL_DROP_PREDICATE_FIELDS = frozenset(
    {
        "safety_profile",
        "minimum_payload_z_m",
        "height_margin_m",
        "drop_z_threshold_m",
        "contact_distance_tolerance_m",
    }
)
_PHYSICAL_SAMPLES_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "physical_transaction",
        "checkpoint_id",
        "checkpoint_event_id",
        "simulator",
        "drop_predicate",
        "samples",
    }
)
_CONTACT_ROLES = frozenset(
    {"payload", "target", "floor", "wall", "front_robot", "back_robot"}
)
_ROBOT_IDS = ("front", "back")
_UPPER_LIMB_SUPPORT_MARKERS = (
    "_elbow_link",
    "_wrist_roll_link",
    "_wrist_pitch_link",
    "_wrist_yaw_link",
    "_hand_",
)
_MODEL_ACTION_PRIMITIVES = frozenset(
    {
        *MODEL_ACTION_SKILL_METHODS,
        *MODEL_COORDINATION_PRIMITIVE_METHODS,
        "hold_position",
        "release_payload",
        "exit_forward",
        "exit_backward",
        "exit_lateral_left",
        "exit_lateral_right",
    }
)
_LIFECYCLE_STATES = frozenset({"carry", "place", "release", "retreat", "complete"})

# These logical producer names are closed-world.  The source digest is supplied
# by the externally attested runner capability, never learned from an artifact.
AUXILIARY_PRODUCER_IDS = MappingProxyType(
    {
        "stall.checkpoint": "praxis_coop.production.stall.checkpoint",
        "stall.host_action_space": "praxis_coop.production.stall.host_action_space",
        "stall.model_output": "praxis_coop.production.stall.model_output",
        "stall.static_validation": "praxis_coop.production.stall.static_validation",
        "stall.host_preflight": "praxis_coop.production.stall.host_preflight",
        "payload_drop.transaction_receipt": (
            "praxis_coop.production.payload_drop.transaction_receipt"
        ),
        "payload_drop.policy": "praxis_coop.production.payload_drop.policy",
        "payload_drop.evaluation": "praxis_coop.production.payload_drop.evaluation",
        "payload_drop.execution_trace": (
            "praxis_coop.production.payload_drop.execution_trace"
        ),
        "payload_drop.physical_samples": (
            "praxis_coop.production.payload_drop.physical_samples"
        ),
    }
)
_CHECKPOINT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "checkpoint_id",
        "checkpoint_event_id",
        "terminal_state_sha256",
        "terminal_sample_trace_sha256",
        "sample_window_s",
        "sample_count",
        "payload",
        "robots",
    }
)
_CHECKPOINT_PAYLOAD_FIELDS = frozenset(
    {"floor_contact_samples", "wall_contact_samples"}
)
_CHECKPOINT_ROBOT_FIELDS = frozenset({"upright", "pelvis_height_m"})
_ACTION_SPACE_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "checkpoint_id",
        "checkpoint_event_id",
        "checkpoint_validity_sha256",
        "lifecycle_state",
        "admissible_action_families",
        "allowed_primitives",
        "allowed_coordination_modes",
    }
)
_STALL_MODEL_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "checkpoint_id",
        "checkpoint_event_id",
        "cycle",
        "checkpoint_validity_sha256",
        "host_action_space_sha256",
        "previous_cycle_preflight_sha256",
        "request_id",
        "usage_call_ordinal",
        "role",
        "model",
        "finish_reason",
        "content",
    }
)
_STALL_STATIC_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "checkpoint_id",
        "checkpoint_event_id",
        "cycle",
        "model_output_sha256",
        "parser_passed",
        "script_present",
        "static_validation_passed",
        "failure_codes",
    }
)
_STALL_PREFLIGHT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "checkpoint_id",
        "checkpoint_event_id",
        "cycle",
        "static_validation_sha256",
        "host_action_space_sha256",
        "evaluated",
        "passed",
        "skipped_reason",
        "failure_codes",
        "executable_child_produced",
        "physical_dispatch",
    }
)
_TRANSACTION_RECEIPT_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "physical_transaction",
        "checkpoint_id_before",
        "checkpoint_event_id_before",
        "checkpoint_id_after",
        "checkpoint_event_id_after",
        "physically_dispatched",
        "verifier",
        "rollback_performed",
        "drop_counter_before",
        "drop_counter_after",
        "policy",
        "evaluation",
        "execution_trace",
        "physical_samples",
        "previous_receipt_sha256",
    }
)
_VERIFIER_FIELDS = frozenset({"accepted", "failure_type"})
_POLICY_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "physical_transaction",
        "checkpoint_id",
        "checkpoint_event_id",
        "action_script",
        "action_script_sha256",
    }
)
_EVALUATION_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "physical_transaction",
        "checkpoint_id",
        "checkpoint_event_id",
        "accepted",
        "failure_type",
        "transition_outcome",
        "completed_phase_ids_before",
        "completed_phase_ids_after",
    }
)
_TRANSACTION_OUTCOMES = frozenset(
    {"ACCEPT_SETUP", "ACCEPT_PROGRESS", "TASK_COMPLETE", "REJECT"}
)
_EXECUTION_FIELDS = frozenset(
    {
        "kind",
        "schema_version",
        "producer",
        "run_id",
        "physical_transaction",
        "checkpoint_id_before",
        "checkpoint_event_id_before",
        "checkpoint_id_after",
        "checkpoint_event_id_after",
        "physically_dispatched",
        "rollback_performed",
        "physics_step_count",
        "dispatch_started_s",
        "dispatch_finished_s",
    }
)


@dataclass(frozen=True)
class _StaticCandidateDerivation:
    parser_passed: bool
    script_present: bool
    static_validation_passed: bool
    failure_codes: tuple[str, ...]
    complete_team: bool = False
    action_family: str = ""


@dataclass(frozen=True)
class PayloadDropPhysicalDerivation:
    """Payload-drop facts recomputed only from raw MuJoCo samples."""

    payload_drop: bool
    minimum_payload_z_m: float
    floor_contact_sample_indices: tuple[int, ...]
    below_height_sample_indices: tuple[int, ...]
    final_supporting_robots: tuple[str, ...]
    final_support_force_n_by_robot: Mapping[str, float]


@dataclass(frozen=True)
class PhysicalTransactionEvidenceValidation:
    """Validation result for one reusable physical-transaction receipt.

    The base validator intentionally does not require a rejected payload-drop
    outcome.  Accounting can therefore consume the same receipt primitive for
    accepted, rejected, setup, and progress transactions.  Terminal-specific
    streak semantics are applied by :func:`recompute_payload_drop_ledger`.
    """

    receipt: Mapping[str, Any] | None = None
    physical_derivation: PayloadDropPhysicalDerivation | None = None
    artifact_hashes: Mapping[str, str] | None = None
    source_artifacts: tuple[BoundJsonArtifact, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.receipt is not None and not self.invalid_reasons


@dataclass(frozen=True)
class AuxiliaryEvidenceDerivation:
    """A deterministic ledger plus every source artifact it transitively pins."""

    ledger: Mapping[str, Any] | None = None
    artifact_hashes: Mapping[str, str] | None = None
    source_artifacts: tuple[BoundJsonArtifact, ...] = ()
    request_ids: tuple[str, ...] = ()
    invalid_reasons: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.ledger is not None and not self.invalid_reasons


def capability_bound_auxiliary_producer_allowlist(
    capability_entrypoint_sha256: str,
) -> Mapping[str, Mapping[str, str]]:
    """Return the sole accepted raw-producer identities for one capability.

    ``producer.source_sha256`` is not a self-attested provenance string.  All
    raw terminal sidecars are emitted through the one byte-pinned production
    entrypoint recorded by the runner capability, so every logical producer is
    bound to that externally supplied digest.  The logical IDs remain distinct
    so an artifact cannot be substituted across evidence stages.
    """

    if not _sha(capability_entrypoint_sha256):
        raise ValueError("capability entrypoint source SHA-256 is invalid")
    return MappingProxyType(
        {
            family: MappingProxyType(
                {"id": producer_id, "source_sha256": capability_entrypoint_sha256}
            )
            for family, producer_id in AUXILIARY_PRODUCER_IDS.items()
        }
    )


def derive_payload_drop_from_physical_samples(
    value: Mapping[str, Any],
    *,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
) -> tuple[PayloadDropPhysicalDerivation, tuple[str, ...]]:
    """Validate raw MuJoCo samples and recompute the frozen drop predicate.

    This public pure-function boundary is shared by terminal and accounting
    reconstruction.  A caller-authored ``failure_type`` is deliberately not
    an input.
    """

    errors: list[str] = []
    derivation = _derive_physical_payload_drop(
        value,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_scene_realization_sha256=expected_scene_realization_sha256,
        label="PHYSICAL_TRANSACTION_SAMPLES",
        errors=errors,
    )
    return derivation, tuple(dict.fromkeys(errors))


def validate_physical_transaction_receipt(
    run_root: str | Path,
    *,
    receipt_ref: Mapping[str, Any],
    expected_run_id: str,
    expected_transaction: int,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    expected_producers: Mapping[str, Mapping[str, str]],
) -> PhysicalTransactionEvidenceValidation:
    """Reload and validate one hash-bound physical transaction and its leaves.

    The receipt/evaluation outcome remains descriptive here.  The returned
    ``physical_derivation`` is authoritative for payload-drop classification;
    callers separately impose accepted/rejected, rollback, checkpoint-chain,
    and counter semantics appropriate to their ledger.
    """

    errors: list[str] = []
    hashes: dict[str, str] = {}
    loaded: list[BoundJsonArtifact] = []
    allowlist = _validated_producer_allowlist(
        expected_producers,
        required_families={
            "payload_drop.transaction_receipt",
            "payload_drop.policy",
            "payload_drop.evaluation",
            "payload_drop.execution_trace",
            "payload_drop.physical_samples",
        },
        errors=errors,
    )
    receipt = _load(
        run_root,
        receipt_ref,
        f"PHYSICAL_TRANSACTION_{expected_transaction}_RECEIPT",
        errors,
        hashes,
        loaded,
        expected_path=f"evidence/transactions/transaction_{expected_transaction:03d}.json",
    )
    parsed: Mapping[str, Any] | None = None
    derivation: PayloadDropPhysicalDerivation | None = None
    if receipt is not None:
        parsed, derivation = _validate_physical_transaction_receipt_base(
            run_root,
            receipt,
            expected_run_id=expected_run_id,
            expected_transaction=expected_transaction,
            errors=errors,
            hashes=hashes,
            loaded=loaded,
            producer_allowlist=allowlist,
            expected_compiled_model_sha256=expected_compiled_model_sha256,
            expected_scene_realization_sha256=expected_scene_realization_sha256,
        )
    return PhysicalTransactionEvidenceValidation(
        receipt=MappingProxyType(dict(parsed)) if parsed is not None else None,
        physical_derivation=derivation,
        artifact_hashes=MappingProxyType(dict(hashes)),
        source_artifacts=tuple(loaded),
        invalid_reasons=tuple(dict.fromkeys(errors)),
    )


def recompute_system_stall_ledger(
    run_root: str | Path,
    *,
    source_index: Mapping[str, Any],
    expected_run_id: str,
    expected_checkpoint_id: str,
    expected_checkpoint_event_id: str,
    expected_terminal_state_sha256: str,
    expected_terminal_sample_trace_sha256: str,
    derived_checkpoint: Mapping[str, Any],
    expected_producers: Mapping[str, Mapping[str, str]],
    fm_calls: Sequence[Mapping[str, Any]] | None = None,
) -> AuxiliaryEvidenceDerivation:
    """Reload raw stall artifacts and reconstruct the complete schema-v3 ledger."""

    errors: list[str] = []
    hashes: dict[str, str] = {}
    loaded: list[BoundJsonArtifact] = []
    producer_allowlist = _validated_producer_allowlist(
        expected_producers,
        required_families={
            "stall.checkpoint",
            "stall.host_action_space",
            "stall.model_output",
            "stall.static_validation",
            "stall.host_preflight",
        },
        errors=errors,
    )
    if not isinstance(source_index, Mapping):
        return AuxiliaryEvidenceDerivation(
            invalid_reasons=("SYSTEM_STALL_SOURCE_INDEX_INVALID",)
        )
    checkpoint_ref = source_index.get("checkpoint_validity")
    action_ref = source_index.get("host_action_space")
    cycle_refs = source_index.get("generation_cycles")
    checkpoint = _load(
        run_root,
        checkpoint_ref,
        "SYSTEM_STALL_CHECKPOINT_VALIDITY",
        errors,
        hashes,
        loaded,
        expected_path=SYSTEM_STALL_CHECKPOINT_RELPATH,
    )
    action = _load(
        run_root,
        action_ref,
        "SYSTEM_STALL_HOST_ACTION_SPACE",
        errors,
        hashes,
        loaded,
        expected_path=SYSTEM_STALL_ACTION_SPACE_RELPATH,
    )
    checkpoint_valid = False
    if checkpoint is not None:
        _require_expected_producer(
            checkpoint, "stall.checkpoint", producer_allowlist, errors
        )
        checkpoint_valid = _validate_stall_checkpoint(
            checkpoint,
            expected_run_id=expected_run_id,
            expected_checkpoint_id=expected_checkpoint_id,
            expected_checkpoint_event_id=expected_checkpoint_event_id,
            expected_terminal_state_sha256=expected_terminal_state_sha256,
            expected_terminal_sample_trace_sha256=(
                expected_terminal_sample_trace_sha256
            ),
            derived_checkpoint=derived_checkpoint,
            errors=errors,
        )
    families: tuple[str, ...] = ()
    if action is not None:
        _require_expected_producer(
            action, "stall.host_action_space", producer_allowlist, errors
        )
        families = _validate_stall_action_space(
            action,
            expected_run_id=expected_run_id,
            expected_checkpoint_id=expected_checkpoint_id,
            expected_checkpoint_event_id=expected_checkpoint_event_id,
            checkpoint_sha256=checkpoint.sha256 if checkpoint else "",
            errors=errors,
        )
    if not isinstance(cycle_refs, list) or len(cycle_refs) != SYSTEM_STALL_GENERATION_CYCLES:
        errors.append("SYSTEM_STALL_RECOVERY_BUDGET_NOT_EXHAUSTED")
        cycle_refs = []

    ledger_cycles: list[dict[str, Any]] = []
    request_ids: list[str] = []
    previous_preflight_sha: str | None = None
    for cycle_number, refs in enumerate(cycle_refs, 1):
        required_cycle_refs = {
            "model_output",
            "static_validation",
            "host_preflight",
        }
        if not isinstance(refs, Mapping) or not required_cycle_refs.issubset(refs):
            errors.append("SYSTEM_STALL_CYCLE_SOURCE_INDEX_INVALID")
            continue
        artifacts: dict[str, BoundJsonArtifact] = {}
        for name in ("model_output", "static_validation", "host_preflight"):
            artifact = _load(
                run_root,
                refs.get(name),
                f"SYSTEM_STALL_CYCLE_{cycle_number}_{name.upper()}",
                errors,
                hashes,
                loaded,
                expected_path=(
                    f"evidence/system_stall/cycle_{cycle_number:03d}_{name}.json"
                ),
            )
            if artifact is not None:
                artifacts[name] = artifact
        if len(artifacts) != 3 or checkpoint is None or action is None:
            errors.append("SYSTEM_STALL_CYCLE_ARTIFACT_SET_INCOMPLETE")
            continue
        for name, artifact in artifacts.items():
            _require_expected_producer(
                artifact, f"stall.{name}", producer_allowlist, errors
            )
        request_id = _validate_stall_cycle(
            artifacts,
            expected_run_id=expected_run_id,
            checkpoint_id=expected_checkpoint_id,
            checkpoint_event_id=expected_checkpoint_event_id,
            cycle=cycle_number,
            checkpoint_sha256=checkpoint.sha256,
            action_space_sha256=action.sha256,
            action_space_families=families,
            previous_preflight_sha256=previous_preflight_sha,
            fm_calls=fm_calls,
            errors=errors,
        )
        previous_preflight_sha = artifacts["host_preflight"].sha256
        if request_id:
            if request_id in request_ids:
                errors.append("SYSTEM_STALL_MODEL_REQUEST_ID_REUSED")
            request_ids.append(request_id)
        ledger_cycles.append(
            {
                "cycle": cycle_number,
                "checkpoint_id": expected_checkpoint_id,
                "checkpoint_event_id": expected_checkpoint_event_id,
                "outcome": "NO_EXECUTABLE_CHILD_PRODUCED",
                "executable_child_produced": False,
                "physical_dispatch": False,
                "model_output": artifacts["model_output"].public_ref(),
                "static_validation": artifacts["static_validation"].public_ref(),
                "host_preflight": artifacts["host_preflight"].public_ref(),
            }
        )

    ledger = {
        "kind": SYSTEM_STALL_LEDGER_KIND,
        "schema_version": SYSTEM_STALL_LEDGER_SCHEMA_VERSION,
        "run_id": expected_run_id,
        "checkpoint_id": expected_checkpoint_id,
        "checkpoint_event_id": expected_checkpoint_event_id,
        "checkpoint_validity": checkpoint.public_ref() if checkpoint else checkpoint_ref,
        "host_action_space": action.public_ref() if action else action_ref,
        "checkpoint_physically_valid": checkpoint_valid,
        "host_contract_consistent": bool(families),
        "admissible_action_families": list(families),
        "maximum_generation_cycles": SYSTEM_STALL_GENERATION_CYCLES,
        "generation_cycles": ledger_cycles,
    }
    return AuxiliaryEvidenceDerivation(
        ledger=MappingProxyType(ledger),
        artifact_hashes=MappingProxyType(dict(hashes)),
        source_artifacts=tuple(loaded),
        request_ids=tuple(request_ids),
        invalid_reasons=tuple(dict.fromkeys(errors)),
    )


def recompute_payload_drop_ledger(
    run_root: str | Path,
    *,
    source_index: Mapping[str, Any],
    expected_run_id: str,
    expected_terminal_checkpoint_id: str,
    expected_terminal_checkpoint_event_id: str,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    expected_producers: Mapping[str, Mapping[str, str]],
    physical_transactions: int,
) -> AuxiliaryEvidenceDerivation:
    """Reload five transaction receipts/leaves and derive the 0-to-5 streak."""

    errors: list[str] = []
    hashes: dict[str, str] = {}
    loaded: list[BoundJsonArtifact] = []
    producer_allowlist = _validated_producer_allowlist(
        expected_producers,
        required_families={
            "payload_drop.transaction_receipt",
            "payload_drop.policy",
            "payload_drop.evaluation",
            "payload_drop.execution_trace",
            "payload_drop.physical_samples",
        },
        errors=errors,
    )
    if not _sha(expected_compiled_model_sha256):
        errors.append("PAYLOAD_DROP_COMPILED_MODEL_EXPECTATION_INVALID")
    if not _sha(expected_scene_realization_sha256):
        errors.append("PAYLOAD_DROP_SCENE_REALIZATION_EXPECTATION_INVALID")
    refs = source_index.get("source_transaction_receipts") if isinstance(
        source_index, Mapping
    ) else None
    if not isinstance(refs, list) or len(refs) != PAYLOAD_DROP_LIMIT:
        return AuxiliaryEvidenceDerivation(
            invalid_reasons=("PAYLOAD_DROP_SOURCE_RECEIPT_SET_INVALID",)
        )
    start = physical_transactions - PAYLOAD_DROP_LIMIT + 1
    if start < 1:
        errors.append("PAYLOAD_DROP_TRANSACTION_COVERAGE_INVALID")
    receipts: list[BoundJsonArtifact] = []
    for offset, ref in enumerate(refs):
        transaction = start + offset
        receipt = _load(
            run_root,
            ref,
            f"PAYLOAD_DROP_TRANSACTION_{transaction}_RECEIPT",
            errors,
            hashes,
            loaded,
            expected_path=f"evidence/transactions/transaction_{transaction:03d}.json",
        )
        if receipt is not None:
            receipts.append(receipt)
    transactions: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    previous_receipt_sha: str | None = None
    previous_event_after: str | None = None
    checkpoint_id = ""
    for offset, receipt in enumerate(receipts):
        transaction = start + offset
        parsed = _validate_drop_transaction_receipt(
            run_root,
            receipt,
            expected_run_id=expected_run_id,
            expected_transaction=transaction,
            expected_counter_before=offset,
            expected_previous_receipt_sha256=previous_receipt_sha,
            allow_predecessor_outside_source_set=(offset == 0 and start > 1),
            expected_event_before=previous_event_after,
            errors=errors,
            hashes=hashes,
            loaded=loaded,
            producer_allowlist=producer_allowlist,
            expected_compiled_model_sha256=expected_compiled_model_sha256,
            expected_scene_realization_sha256=expected_scene_realization_sha256,
        )
        previous_receipt_sha = receipt.sha256
        if parsed is None:
            continue
        if not checkpoint_id:
            checkpoint_id = parsed["checkpoint_id_before"]
        if (
            parsed["checkpoint_id_before"] != checkpoint_id
            or parsed["checkpoint_id_after"] != checkpoint_id
        ):
            errors.append("PAYLOAD_DROP_CHECKPOINT_BINDING_MISMATCH")
        previous_event_after = parsed["checkpoint_event_id_after"]
        transactions.append(
            {
                "physical_transaction": transaction,
                "checkpoint_id": checkpoint_id,
                "checkpoint_event_id_before": parsed["checkpoint_event_id_before"],
                "checkpoint_event_id_after": parsed["checkpoint_event_id_after"],
                "event_type": "payload_drop",
                "verified_failure_type": "payload_drop",
                "counter_before": offset,
                "counter_after": offset + 1,
                "streak_effect": "increment_drop",
                "source_receipt_sha256": receipt.sha256,
                "raw_physical_derivation": parsed["physical_derivation"],
            }
        )
        drops.append(
            {
                "ordinal": offset + 1,
                "checkpoint_id": checkpoint_id,
                "checkpoint_event_id_after": parsed["checkpoint_event_id_after"],
                "physical_transaction": transaction,
                "verified_failure_type": "payload_drop",
                "counted": True,
                "source_receipt_sha256": receipt.sha256,
                "raw_physical_derivation": parsed["physical_derivation"],
            }
        )
    if checkpoint_id != expected_terminal_checkpoint_id:
        errors.append("PAYLOAD_DROP_CHECKPOINT_BINDING_MISMATCH")
    if previous_event_after != expected_terminal_checkpoint_event_id:
        errors.append("PAYLOAD_DROP_TERMINAL_CHECKPOINT_EVENT_MISMATCH")
    if len(receipts) != PAYLOAD_DROP_LIMIT or len(transactions) != PAYLOAD_DROP_LIMIT:
        errors.append("PAYLOAD_DROP_TRANSACTION_RECEIPT_SET_INCOMPLETE")
    ledger = {
        "kind": PAYLOAD_DROP_LEDGER_KIND,
        "schema_version": PAYLOAD_DROP_LEDGER_SCHEMA_VERSION,
        "run_id": expected_run_id,
        "checkpoint_id": checkpoint_id,
        "terminal_checkpoint_event_id": expected_terminal_checkpoint_event_id,
        "maximum_drops": PAYLOAD_DROP_LIMIT,
        "coverage_start_transaction": start,
        "coverage_end_transaction": physical_transactions,
        "receipt_chain_head_sha256": receipts[0].sha256 if receipts else "",
        "receipt_chain_tail_sha256": receipts[-1].sha256 if receipts else "",
        "source_transaction_receipts": [item.public_ref() for item in receipts],
        "drops": drops,
        "transactions": transactions,
    }
    return AuxiliaryEvidenceDerivation(
        ledger=MappingProxyType(ledger),
        artifact_hashes=MappingProxyType(dict(hashes)),
        source_artifacts=tuple(loaded),
        invalid_reasons=tuple(dict.fromkeys(errors)),
    )


def _validate_stall_checkpoint(
    artifact: BoundJsonArtifact,
    *,
    expected_run_id: str,
    expected_checkpoint_id: str,
    expected_checkpoint_event_id: str,
    expected_terminal_state_sha256: str,
    expected_terminal_sample_trace_sha256: str,
    derived_checkpoint: Mapping[str, Any],
    errors: list[str],
) -> bool:
    value = artifact.payload
    if not _envelope(
        value,
        fields=_CHECKPOINT_FIELDS,
        kind=SYSTEM_STALL_CHECKPOINT_KIND,
        label="SYSTEM_STALL_CHECKPOINT_VALIDITY",
        errors=errors,
    ):
        return False
    _binding(
        value,
        expected_run_id,
        expected_checkpoint_id,
        expected_checkpoint_event_id,
        "SYSTEM_STALL_CHECKPOINT_VALIDITY",
        errors,
    )
    if value.get("terminal_state_sha256") != expected_terminal_state_sha256:
        errors.append("SYSTEM_STALL_CHECKPOINT_TERMINAL_STATE_MISMATCH")
    if value.get("terminal_sample_trace_sha256") != expected_terminal_sample_trace_sha256:
        errors.append("SYSTEM_STALL_CHECKPOINT_TERMINAL_TRACE_MISMATCH")
    if not _close(value.get("sample_window_s"), derived_checkpoint.get("sample_window_s")):
        errors.append("SYSTEM_STALL_CHECKPOINT_SAMPLE_WINDOW_MISMATCH")
    if value.get("sample_count") != derived_checkpoint.get("sample_count"):
        errors.append("SYSTEM_STALL_CHECKPOINT_SAMPLE_COUNT_MISMATCH")
    payload = value.get("payload")
    expected_payload = derived_checkpoint.get("payload")
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _CHECKPOINT_PAYLOAD_FIELDS
        or not isinstance(expected_payload, Mapping)
        or any(payload.get(key) != expected_payload.get(key) for key in _CHECKPOINT_PAYLOAD_FIELDS)
    ):
        errors.append("SYSTEM_STALL_CHECKPOINT_PAYLOAD_MISMATCH")
    robots = value.get("robots")
    expected_robots = derived_checkpoint.get("robots")
    valid_robots = isinstance(robots, Mapping) and set(robots) == {"front", "back"}
    for robot_id in ("front", "back"):
        robot = robots.get(robot_id) if isinstance(robots, Mapping) else None
        expected_robot = (
            expected_robots.get(robot_id) if isinstance(expected_robots, Mapping) else None
        )
        if (
            not isinstance(robot, Mapping)
            or set(robot) != _CHECKPOINT_ROBOT_FIELDS
            or not isinstance(expected_robot, Mapping)
            or robot.get("upright") is not expected_robot.get("upright")
            or not _close(robot.get("pelvis_height_m"), expected_robot.get("pelvis_height_m"))
        ):
            errors.append(f"SYSTEM_STALL_CHECKPOINT_ROBOT_{robot_id.upper()}_MISMATCH")
            valid_robots = False
    physical_valid = (
        isinstance(payload, Mapping)
        and payload.get("floor_contact_samples") == 0
        and payload.get("wall_contact_samples") == 0
        and valid_robots
        and all(
            isinstance(robots.get(robot_id), Mapping)
            and robots[robot_id].get("upright") is True
            and _number(robots[robot_id].get("pelvis_height_m")) is not None
            and float(robots[robot_id]["pelvis_height_m"])
            >= TRUSTED_CHECKPOINT_MIN_PELVIS_HEIGHT_M
            for robot_id in ("front", "back")
        )
    )
    if not physical_valid:
        errors.append("SYSTEM_STALL_CHECKPOINT_INVALID")
    return physical_valid


def _validate_stall_action_space(
    artifact: BoundJsonArtifact,
    *,
    expected_run_id: str,
    expected_checkpoint_id: str,
    expected_checkpoint_event_id: str,
    checkpoint_sha256: str,
    errors: list[str],
) -> tuple[str, ...]:
    value = artifact.payload
    if not _envelope(
        value,
        fields=_ACTION_SPACE_FIELDS,
        kind=SYSTEM_STALL_ACTION_SPACE_KIND,
        label="SYSTEM_STALL_HOST_ACTION_SPACE",
        errors=errors,
    ):
        return ()
    _binding(
        value,
        expected_run_id,
        expected_checkpoint_id,
        expected_checkpoint_event_id,
        "SYSTEM_STALL_HOST_ACTION_SPACE",
        errors,
    )
    if value.get("checkpoint_validity_sha256") != checkpoint_sha256:
        errors.append("SYSTEM_STALL_ACTION_SPACE_CHECKPOINT_MISMATCH")
    lifecycle_state = value.get("lifecycle_state")
    if lifecycle_state not in _LIFECYCLE_STATES:
        errors.append("SYSTEM_STALL_ACTION_SPACE_LIFECYCLE_INVALID")
    primitives = _strings(value.get("allowed_primitives"))
    modes = _strings(value.get("allowed_coordination_modes"))
    declared_families = _strings(value.get("admissible_action_families"))
    if not primitives or any(item not in _MODEL_ACTION_PRIMITIVES for item in primitives):
        errors.append("SYSTEM_STALL_ACTION_SPACE_ALLOWED_PRIMITIVES_INVALID")
        primitives = ()
    if not modes or any(item not in {"sync", "async"} for item in modes):
        errors.append("SYSTEM_STALL_ACTION_SPACE_ALLOWED_COORDINATION_MODES_INVALID")
        modes = ()
    families = _derive_action_families(primitives, modes)
    if not families:
        errors.append("SYSTEM_STALL_NO_ADMISSIBLE_ACTION_FAMILY")
    if declared_families != families:
        errors.append("SYSTEM_STALL_ACTION_SPACE_DERIVATION_MISMATCH")
    return families


def _validate_stall_cycle(
    artifacts: Mapping[str, BoundJsonArtifact],
    *,
    expected_run_id: str,
    checkpoint_id: str,
    checkpoint_event_id: str,
    cycle: int,
    checkpoint_sha256: str,
    action_space_sha256: str,
    action_space_families: Sequence[str],
    previous_preflight_sha256: str | None,
    fm_calls: Sequence[Mapping[str, Any]] | None,
    errors: list[str],
) -> str:
    model, static, preflight = (
        artifacts["model_output"],
        artifacts["static_validation"],
        artifacts["host_preflight"],
    )
    valid_model = _envelope(
        model.payload,
        fields=_STALL_MODEL_FIELDS,
        kind=SYSTEM_STALL_MODEL_OUTPUT_KIND,
        label="SYSTEM_STALL_MODEL_OUTPUT",
        errors=errors,
    )
    valid_static = _envelope(
        static.payload,
        fields=_STALL_STATIC_FIELDS,
        kind=SYSTEM_STALL_STATIC_VALIDATION_KIND,
        label="SYSTEM_STALL_STATIC_VALIDATION",
        errors=errors,
    )
    valid_preflight = _envelope(
        preflight.payload,
        fields=_STALL_PREFLIGHT_FIELDS,
        kind=SYSTEM_STALL_HOST_PREFLIGHT_KIND,
        label="SYSTEM_STALL_HOST_PREFLIGHT",
        errors=errors,
    )
    for name, artifact, valid in (
        ("MODEL_OUTPUT", model, valid_model),
        ("STATIC_VALIDATION", static, valid_static),
        ("HOST_PREFLIGHT", preflight, valid_preflight),
    ):
        if not valid:
            continue
        _binding(
            artifact.payload,
            expected_run_id,
            checkpoint_id,
            checkpoint_event_id,
            f"SYSTEM_STALL_{name}",
            errors,
        )
        if artifact.payload.get("cycle") != cycle:
            errors.append(f"SYSTEM_STALL_{name}_CYCLE_MISMATCH")
    request_id = ""
    if valid_model:
        value = model.payload
        if value.get("checkpoint_validity_sha256") != checkpoint_sha256:
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_CHECKPOINT_MISMATCH")
        if value.get("host_action_space_sha256") != action_space_sha256:
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_ACTION_SPACE_MISMATCH")
        if value.get("previous_cycle_preflight_sha256") != previous_preflight_sha256:
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_CYCLE_CHAIN_MISMATCH")
        request_id = str(value.get("request_id") or "").strip()
        if not request_id:
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_REQUEST_ID_INVALID")
        ordinal = _integer(value.get("usage_call_ordinal"))
        if ordinal is None or ordinal < 1:
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_USAGE_ORDINAL_INVALID")
        for key in ("role", "model", "finish_reason"):
            if not _nonempty(value.get(key)):
                errors.append(f"SYSTEM_STALL_MODEL_OUTPUT_{key.upper()}_INVALID")
        if not isinstance(value.get("content"), str):
            errors.append("SYSTEM_STALL_MODEL_OUTPUT_CONTENT_INVALID")
        if fm_calls is not None and ordinal is not None:
            if ordinal > len(fm_calls):
                errors.append("SYSTEM_STALL_MODEL_OUTPUT_USAGE_ORDINAL_OUT_OF_RANGE")
            else:
                call = fm_calls[ordinal - 1]
                if (
                    value.get("request_id") != call.get("request_id")
                    or value.get("role") != call.get("role")
                    or value.get("model") != call.get("response_model")
                ):
                    errors.append("SYSTEM_STALL_MODEL_OUTPUT_USAGE_CALL_MISMATCH")
    static_derivation = _StaticCandidateDerivation(
        parser_passed=False,
        script_present=False,
        static_validation_passed=False,
        failure_codes=("MODEL_OUTPUT_SCHEMA_INVALID",),
    )
    if valid_model:
        static_derivation = _derive_static_candidate(str(model.payload.get("content", "")))
    if valid_static:
        value = static.payload
        if value.get("model_output_sha256") != model.sha256:
            errors.append("SYSTEM_STALL_STATIC_VALIDATION_MODEL_OUTPUT_MISMATCH")
        expected_static = {
            "parser_passed": static_derivation.parser_passed,
            "script_present": static_derivation.script_present,
            "static_validation_passed": static_derivation.static_validation_passed,
            "failure_codes": list(static_derivation.failure_codes),
        }
        if any(value.get(key) != expected for key, expected in expected_static.items()):
            errors.append("SYSTEM_STALL_STATIC_VALIDATION_DERIVATION_MISMATCH")
    if valid_preflight:
        value = preflight.payload
        if value.get("static_validation_sha256") != static.sha256:
            errors.append("SYSTEM_STALL_HOST_PREFLIGHT_STATIC_MISMATCH")
        if value.get("host_action_space_sha256") != action_space_sha256:
            errors.append("SYSTEM_STALL_HOST_PREFLIGHT_ACTION_SPACE_MISMATCH")
        # ``host_preflight`` binds the action-space artifact by hash.  Reloaded
        # action-space families are passed explicitly here; a hidden producer
        # boolean is never used as the expected outcome.
        expected_preflight = _derive_host_preflight(
            static_derivation,
            action_space_families=tuple(action_space_families),
        )
        if any(
            value.get(key) != expected
            for key, expected in expected_preflight.items()
        ):
            errors.append("SYSTEM_STALL_HOST_PREFLIGHT_DERIVATION_MISMATCH")
        if value.get("physical_dispatch") is not False:
            errors.append("SYSTEM_STALL_CYCLE_DISPATCHED_PHYSICS")
        if value.get("executable_child_produced") is not False:
            errors.append("SYSTEM_STALL_HOST_PREFLIGHT_EXECUTABLE_CHILD_EXISTS")
    return request_id


def _validate_drop_transaction_receipt(
    root: str | Path,
    receipt: BoundJsonArtifact,
    *,
    expected_run_id: str,
    expected_transaction: int,
    expected_counter_before: int,
    expected_previous_receipt_sha256: str | None,
    allow_predecessor_outside_source_set: bool,
    expected_event_before: str | None,
    errors: list[str],
    hashes: dict[str, str],
    loaded: list[BoundJsonArtifact],
    producer_allowlist: Mapping[str, tuple[str, str]],
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
) -> Mapping[str, Any] | None:
    label = f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}"
    value, physical_derivation = _validate_physical_transaction_receipt_base(
        root,
        receipt,
        expected_run_id=expected_run_id,
        expected_transaction=expected_transaction,
        errors=errors,
        hashes=hashes,
        loaded=loaded,
        producer_allowlist=producer_allowlist,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_scene_realization_sha256=expected_scene_realization_sha256,
        label=label,
    )
    if value is None:
        return None
    checkpoint_before = value.get("checkpoint_id_before")
    checkpoint_after = value.get("checkpoint_id_after")
    event_before = value.get("checkpoint_event_id_before")
    event_after = value.get("checkpoint_event_id_after")
    if checkpoint_before != checkpoint_after:
        errors.append(f"{label}_ROLLBACK_CHECKPOINT_MISMATCH")
    if expected_event_before is not None and event_before != expected_event_before:
        errors.append(f"{label}_CHECKPOINT_EVENT_CHAIN_MISMATCH")
    if value.get("physically_dispatched") is not True:
        errors.append(f"{label}_NOT_PHYSICALLY_DISPATCHED")
    verifier = value.get("verifier")
    if (
        not isinstance(verifier, Mapping)
        or set(verifier) != _VERIFIER_FIELDS
        or verifier.get("accepted") is not False
        or verifier.get("failure_type") != "payload_drop"
    ):
        errors.append(f"{label}_NOT_VERIFIED_PAYLOAD_DROP")
    if value.get("rollback_performed") is not True:
        errors.append(f"{label}_ROLLBACK_NOT_PERFORMED")
    if (
        value.get("drop_counter_before") != expected_counter_before
        or value.get("drop_counter_after") != expected_counter_before + 1
    ):
        errors.append(f"{label}_DROP_COUNTER_INVALID")
    previous_receipt_sha256 = value.get("previous_receipt_sha256")
    if allow_predecessor_outside_source_set:
        # The terminal drop ledger deliberately carries only the five-drop
        # suffix.  When that suffix starts after transaction 1, its first
        # receipt must point to the preceding full-accounting transaction, but
        # that predecessor is outside this validator's source set.  Require a
        # well-formed external link here; the transaction-accounting ledger
        # independently binds it to the exact preceding receipt.  Transactions
        # two through five of the suffix remain strictly hash-chained below.
        if not _sha(previous_receipt_sha256):
            errors.append(f"{label}_RECEIPT_HASH_CHAIN_MISMATCH")
    elif previous_receipt_sha256 != expected_previous_receipt_sha256:
        errors.append(f"{label}_RECEIPT_HASH_CHAIN_MISMATCH")
    if physical_derivation is None or not physical_derivation.payload_drop:
        errors.append(f"{label}_RAW_PHYSICS_NOT_PAYLOAD_DROP")
        physical_derivation = _invalid_physical_drop_derivation()
    return {**value, "physical_derivation": _physical_drop_document(physical_derivation)}


def _validate_physical_transaction_receipt_base(
    root: str | Path,
    receipt: BoundJsonArtifact,
    *,
    expected_run_id: str,
    expected_transaction: int,
    errors: list[str],
    hashes: dict[str, str],
    loaded: list[BoundJsonArtifact],
    producer_allowlist: Mapping[str, tuple[str, str]],
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    label: str | None = None,
) -> tuple[Mapping[str, Any] | None, PayloadDropPhysicalDerivation | None]:
    label = label or f"PHYSICAL_TRANSACTION_{expected_transaction}"
    value = receipt.payload
    if not _envelope(
        value,
        fields=_TRANSACTION_RECEIPT_FIELDS,
        kind=PHYSICAL_TRANSACTION_RECEIPT_KIND,
        label=label,
        errors=errors,
    ):
        return None, None
    _require_expected_producer(
        receipt,
        "payload_drop.transaction_receipt",
        producer_allowlist,
        errors,
    )
    if value.get("run_id") != expected_run_id:
        errors.append(f"{label}_RUN_BINDING_MISMATCH")
    if value.get("physical_transaction") != expected_transaction:
        errors.append(f"{label}_SEQUENCE_INVALID")
    for suffix in (
        "checkpoint_id_before",
        "checkpoint_event_id_before",
        "checkpoint_id_after",
        "checkpoint_event_id_after",
    ):
        if not _nonempty(value.get(suffix)):
            errors.append(f"{label}_{suffix.upper()}_INVALID")
    if value.get("checkpoint_event_id_before") == value.get(
        "checkpoint_event_id_after"
    ):
        errors.append(f"{label}_CHECKPOINT_EVENT_NOT_ADVANCED")
    if value.get("physically_dispatched") is not True:
        errors.append(f"{label}_NOT_PHYSICALLY_DISPATCHED")
    if not isinstance(value.get("rollback_performed"), bool):
        errors.append(f"{label}_ROLLBACK_FLAG_INVALID")
    verifier = value.get("verifier")
    if (
        not isinstance(verifier, Mapping)
        or set(verifier) != _VERIFIER_FIELDS
        or not isinstance(verifier.get("accepted"), bool)
        or not _nonempty(verifier.get("failure_type"))
    ):
        errors.append(f"{label}_VERIFIER_INVALID")
    elif (
        (verifier["accepted"] is True and verifier["failure_type"] != "none")
        or (verifier["accepted"] is False and verifier["failure_type"] == "none")
    ):
        errors.append(f"{label}_VERIFIER_OUTCOME_INCONSISTENT")
    for counter_name in ("drop_counter_before", "drop_counter_after"):
        counter = _integer(value.get(counter_name))
        if counter is None or counter < 0:
            errors.append(f"{label}_{counter_name.upper()}_INVALID")
    previous = value.get("previous_receipt_sha256")
    if previous is not None and not _sha(previous):
        errors.append(f"{label}_PREVIOUS_RECEIPT_SHA256_INVALID")

    leaves: dict[str, BoundJsonArtifact] = {}
    for name in ("policy", "evaluation", "execution_trace", "physical_samples"):
        artifact = _load(
            root,
            value.get(name),
            f"{label}_{name.upper()}",
            errors,
            hashes,
            loaded,
            expected_path=(
                f"evidence/transactions/transaction_{expected_transaction:03d}_{name}.json"
            ),
        )
        if artifact is not None:
            leaves[name] = artifact
    if len(leaves) != 4:
        errors.append(f"{label}_RAW_ARTIFACT_SET_INCOMPLETE")
        return value, None
    derivation = _validate_transaction_leaves(
        leaves,
        receipt=value,
        expected_run_id=expected_run_id,
        expected_transaction=expected_transaction,
        errors=errors,
        producer_allowlist=producer_allowlist,
        expected_compiled_model_sha256=expected_compiled_model_sha256,
        expected_scene_realization_sha256=expected_scene_realization_sha256,
        label=label,
    )
    accepted = verifier.get("accepted") if isinstance(verifier, Mapping) else None
    if isinstance(accepted, bool):
        if value.get("rollback_performed") is accepted:
            errors.append(f"{label}_ROLLBACK_ACCEPTANCE_MISMATCH")
        if accepted is False and value.get("checkpoint_id_before") != value.get(
            "checkpoint_id_after"
        ):
            errors.append(f"{label}_ROLLBACK_CHECKPOINT_MISMATCH")
    counter_before = _integer(value.get("drop_counter_before"))
    counter_after = _integer(value.get("drop_counter_after"))
    if counter_before is not None and counter_after is not None:
        expected_after = counter_before + 1 if derivation.payload_drop else 0
        if counter_after != expected_after:
            errors.append(f"{label}_RAW_DERIVED_DROP_COUNTER_MISMATCH")
    return value, derivation


def _validate_transaction_leaves(
    leaves: Mapping[str, BoundJsonArtifact],
    *,
    receipt: Mapping[str, Any],
    expected_run_id: str,
    expected_transaction: int,
    errors: list[str],
    producer_allowlist: Mapping[str, tuple[str, str]],
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    label: str,
) -> PayloadDropPhysicalDerivation:
    policy, evaluation, execution, physical = (
        leaves["policy"],
        leaves["evaluation"],
        leaves["execution_trace"],
        leaves["physical_samples"],
    )
    specs = (
        (policy, _POLICY_FIELDS, PAYLOAD_DROP_POLICY_KIND, "POLICY"),
        (evaluation, _EVALUATION_FIELDS, PAYLOAD_DROP_EVALUATION_KIND, "EVALUATION"),
        (execution, _EXECUTION_FIELDS, PAYLOAD_DROP_EXECUTION_TRACE_KIND, "EXECUTION_TRACE"),
        (
            physical,
            _PHYSICAL_SAMPLES_FIELDS,
            PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND,
            "PHYSICAL_SAMPLES",
        ),
    )
    validity: dict[str, bool] = {}
    for artifact, fields, kind, name in specs:
        leaf_label = f"{label}_{name}"
        validity[name] = _envelope(
            artifact.payload,
            fields=fields,
            kind=kind,
            label=leaf_label,
            errors=errors,
        )
        _require_expected_producer(
            artifact,
            f"payload_drop.{name.lower()}",
            producer_allowlist,
            errors,
        )
        if artifact.payload.get("run_id") != expected_run_id:
            errors.append(f"{leaf_label}_RUN_BINDING_MISMATCH")
        if artifact.payload.get("physical_transaction") != expected_transaction:
            errors.append(f"{leaf_label}_TRANSACTION_MISMATCH")
    if validity.get("POLICY"):
        value = policy.payload
        if (
            value.get("checkpoint_id") != receipt.get("checkpoint_id_before")
            or value.get("checkpoint_event_id")
            != receipt.get("checkpoint_event_id_before")
        ):
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_POLICY_CHECKPOINT_MISMATCH"
            )
        script = value.get("action_script")
        if not isinstance(script, str) or not script.strip():
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_POLICY_SCRIPT_INVALID"
            )
        elif value.get("action_script_sha256") != hashlib.sha256(
            script.encode("utf-8")
        ).hexdigest():
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_POLICY_SCRIPT_HASH_MISMATCH"
            )
    if validity.get("EVALUATION"):
        value = evaluation.payload
        if (
            value.get("checkpoint_id") != receipt.get("checkpoint_id_before")
            or value.get("checkpoint_event_id")
            != receipt.get("checkpoint_event_id_before")
        ):
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_EVALUATION_CHECKPOINT_MISMATCH"
            )
        verifier = receipt.get("verifier")
        if not isinstance(verifier, Mapping) or (
            value.get("accepted") != verifier.get("accepted")
            or value.get("failure_type") != verifier.get("failure_type")
        ):
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_VERIFIER_RESULT_MISMATCH"
            )
        _validate_transaction_outcome(
            value,
            label=f"{label}_EVALUATION",
            errors=errors,
        )
    if validity.get("EXECUTION_TRACE"):
        value = execution.payload
        comparisons = {
            "checkpoint_id_before": "checkpoint_id_before",
            "checkpoint_event_id_before": "checkpoint_event_id_before",
            "checkpoint_id_after": "checkpoint_id_after",
            "checkpoint_event_id_after": "checkpoint_event_id_after",
            "physically_dispatched": "physically_dispatched",
            "rollback_performed": "rollback_performed",
        }
        if any(value.get(source) != receipt.get(target) for source, target in comparisons.items()):
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_EXECUTION_RESULT_MISMATCH"
            )
        steps = _integer(value.get("physics_step_count"))
        start = _number(value.get("dispatch_started_s"))
        finish = _number(value.get("dispatch_finished_s"))
        if steps is None or steps < 1 or start is None or finish is None or finish <= start:
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_EXECUTION_TRACE_INVALID"
            )
    physical_derivation = _invalid_physical_drop_derivation()
    if validity.get("PHYSICAL_SAMPLES"):
        value = physical.payload
        if (
            value.get("checkpoint_id") != receipt.get("checkpoint_id_before")
            or value.get("checkpoint_event_id")
            != receipt.get("checkpoint_event_id_before")
        ):
            errors.append(
                f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_PHYSICAL_CHECKPOINT_MISMATCH"
            )
        physical_derivation = _derive_physical_payload_drop(
            value,
            expected_compiled_model_sha256=expected_compiled_model_sha256,
            expected_scene_realization_sha256=expected_scene_realization_sha256,
            label=f"PAYLOAD_DROP_TRANSACTION_{expected_transaction}_PHYSICAL_SAMPLES",
            errors=errors,
        )
        if physical_derivation.payload_drop and (
            evaluation.payload.get("accepted") is not False
            or evaluation.payload.get("failure_type") != "payload_drop"
            or evaluation.payload.get("transition_outcome") != "REJECT"
        ):
            errors.append(
                f"{label}_RAW_DROP_CLASSIFICATION_MISMATCH"
            )
        elif (
            not physical_derivation.payload_drop
            and evaluation.payload.get("failure_type") == "payload_drop"
        ):
            errors.append(f"{label}_FORGED_PAYLOAD_DROP_CLASSIFICATION")
    return physical_derivation


def _validated_producer_allowlist(
    value: Mapping[str, Mapping[str, str]],
    *,
    required_families: set[str],
    errors: list[str],
) -> Mapping[str, tuple[str, str]]:
    """Validate externally supplied closed-world producer identities."""

    if not isinstance(value, Mapping):
        errors.append("TERMINAL_AUXILIARY_PRODUCER_ALLOWLIST_INVALID")
        return MappingProxyType({})
    supplied = set(value)
    unknown = supplied - set(AUXILIARY_PRODUCER_IDS)
    missing = required_families - supplied
    if unknown:
        errors.append("TERMINAL_AUXILIARY_PRODUCER_ALLOWLIST_UNKNOWN_FAMILY")
    if missing:
        errors.append("TERMINAL_AUXILIARY_PRODUCER_ALLOWLIST_INCOMPLETE")
    result: dict[str, tuple[str, str]] = {}
    for family in sorted(supplied & set(AUXILIARY_PRODUCER_IDS)):
        producer = value.get(family)
        if (
            not isinstance(producer, Mapping)
            or set(producer) != _PRODUCER_FIELDS
            or producer.get("id") != AUXILIARY_PRODUCER_IDS[family]
            or not _sha(producer.get("source_sha256"))
        ):
            errors.append(
                f"{family.upper().replace('.', '_')}_EXPECTED_PRODUCER_INVALID"
            )
            continue
        result[family] = (
            str(producer["id"]),
            str(producer["source_sha256"]),
        )
    required_sources = {
        result[family][1] for family in required_families if family in result
    }
    if len(required_sources) > 1:
        errors.append("TERMINAL_AUXILIARY_PRODUCER_CAPABILITY_SHA_MISMATCH")
    return MappingProxyType(result)


def _require_expected_producer(
    artifact: BoundJsonArtifact,
    producer_family: str,
    allowlist: Mapping[str, tuple[str, str]],
    errors: list[str],
) -> None:
    expected = allowlist.get(producer_family)
    producer = artifact.payload.get("producer")
    actual = None
    if isinstance(producer, Mapping):
        producer_id = producer.get("id")
        source_sha256 = producer.get("source_sha256")
        if _nonempty(producer_id) and _sha(source_sha256):
            actual = (str(producer_id), str(source_sha256))
    label = producer_family.upper().replace(".", "_")
    if expected is None:
        errors.append(f"{label}_EXPECTED_PRODUCER_MISSING")
    elif actual != expected:
        errors.append(f"{label}_PRODUCER_NOT_CAPABILITY_BOUND")


def _derive_action_families(
    primitives: Sequence[str],
    modes: Sequence[str],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{mode}:{primitive}"
            for primitive in primitives
            for mode in modes
            if mode in coordination_modes_for_method(primitive)
        )
    )


def _strict_model_json(content: str) -> Mapping[str, Any]:
    """Mirror V83's JSON-object extraction while rejecting ambiguous JSON."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    decoder = json.JSONDecoder(
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    candidates = (0,) if text.startswith("{") else tuple(
        match.start() for match in re.finditer(r"\{", text)
    )
    last_error: Exception | None = None
    for start in candidates:
        try:
            value, end = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            continue
        trailing = text[start + end :].strip()
        # V83 accepts a first JSON object embedded in prose.  Retain that
        # behavior, but reject non-finite values hidden in valid numeric syntax.
        if _contains_nonfinite_json_number(value):
            last_error = ValueError("non-finite JSON number")
            continue
        if isinstance(value, Mapping):
            return value
        last_error = ValueError("model response is not a JSON object")
        if not trailing:
            break
    raise ValueError("model response has no unambiguous JSON object") from last_error


def _contains_nonfinite_json_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite_json_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite_json_number(item) for item in value)
    return False


def _derive_static_candidate(content: str) -> _StaticCandidateDerivation:
    try:
        payload = _strict_model_json(content)
    except (TypeError, ValueError):
        return _StaticCandidateDerivation(
            parser_passed=False,
            script_present=False,
            static_validation_passed=False,
            failure_codes=("MODEL_OUTPUT_JSON_INVALID",),
        )

    raw_scripts = payload.get("robot_scripts")
    entries: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_scripts, Mapping):
        for robot_id, item in raw_scripts.items():
            if isinstance(item, str):
                entries.append((str(robot_id), {"code": item}))
            elif isinstance(item, Mapping):
                entries.append((str(robot_id), item))
            else:
                return _invalid_static_candidate(parser_passed=True, script_present=True)
    elif isinstance(raw_scripts, list):
        for item in raw_scripts:
            if not isinstance(item, Mapping):
                return _invalid_static_candidate(parser_passed=True, script_present=True)
            entries.append((str(item.get("robot_id", "")), item))
    elif raw_scripts is not None:
        return _invalid_static_candidate(parser_passed=True, script_present=True)

    if not entries:
        code = payload.get("code")
        if not isinstance(code, str) or not code.strip():
            return _StaticCandidateDerivation(
                parser_passed=True,
                script_present=False,
                static_validation_passed=False,
                failure_codes=("NO_SCRIPT",),
            )
        try:
            method, _ = validate_step_policy_code(code.strip())
        except Exception:  # noqa: BLE001 - pure fail-closed validator boundary
            return _invalid_static_candidate(parser_passed=True, script_present=True)
        return _StaticCandidateDerivation(
            parser_passed=True,
            script_present=True,
            static_validation_passed=True,
            failure_codes=(),
            complete_team=False,
            action_family=f"sync:{method}",
        )

    scripts: list[RobotPolicyScript] = []
    methods: list[str] = []
    modes: list[str] = []
    seen_robots: set[str] = set()
    for robot_id, item in entries:
        code = item.get("code")
        group = item.get("synchronization_group")
        mode = item.get("coordination_mode", "sync")
        if (
            not _nonempty(robot_id)
            or robot_id in seen_robots
            or not isinstance(code, str)
            or not code.strip()
            or not _nonempty(group)
            or mode not in {"sync", "async"}
        ):
            return _invalid_static_candidate(parser_passed=True, script_present=True)
        try:
            method, _ = validate_step_policy_code(code.strip())
        except Exception:  # noqa: BLE001 - model output is untrusted
            return _invalid_static_candidate(parser_passed=True, script_present=True)
        seen_robots.add(robot_id)
        methods.append(method)
        modes.append(str(mode))
        scripts.append(
            RobotPolicyScript(
                robot_id=robot_id,
                script_id=str(item.get("script_id") or f"{robot_id}_policy"),
                code=code.strip(),
                synchronization_group=str(group).strip(),
                phase_id=str(item.get("phase_id") or group).strip(),
                contact_intent=str(item.get("contact_intent") or "unspecified")
                .strip()
                .lower(),
                coordination_mode=str(mode),  # type: ignore[arg-type]
            )
        )

    complete_team = len(scripts) == 2 and seen_robots == {"front", "back"}
    if complete_team:
        try:
            validate_team_policy_scripts(tuple(scripts))
        except Exception:  # noqa: BLE001 - model output is untrusted
            return _invalid_static_candidate(parser_passed=True, script_present=True)
    family = ""
    if len(set(methods)) == 1 and len(set(modes)) == 1:
        family = f"{modes[0]}:{methods[0]}"
    if not family:
        return _invalid_static_candidate(parser_passed=True, script_present=True)
    return _StaticCandidateDerivation(
        parser_passed=True,
        script_present=True,
        static_validation_passed=True,
        failure_codes=(),
        complete_team=complete_team,
        action_family=family,
    )


def _invalid_static_candidate(
    *, parser_passed: bool, script_present: bool
) -> _StaticCandidateDerivation:
    return _StaticCandidateDerivation(
        parser_passed=parser_passed,
        script_present=script_present,
        static_validation_passed=False,
        failure_codes=("STATIC_POLICY_INVALID",),
    )


def _derive_host_preflight(
    candidate: _StaticCandidateDerivation,
    *,
    action_space_families: Sequence[str],
) -> Mapping[str, Any]:
    if not candidate.static_validation_passed:
        return {
            "evaluated": False,
            "passed": False,
            "skipped_reason": "STATIC_VALIDATION_FAILED",
            "failure_codes": ["STATIC_VALIDATION_FAILED"],
            "executable_child_produced": False,
            "physical_dispatch": False,
        }
    if not candidate.complete_team:
        return {
            "evaluated": True,
            "passed": False,
            "skipped_reason": None,
            "failure_codes": ["INCOMPLETE_TEAM_POLICY"],
            "executable_child_produced": False,
            "physical_dispatch": False,
        }
    if candidate.action_family not in set(action_space_families):
        return {
            "evaluated": True,
            "passed": False,
            "skipped_reason": None,
            "failure_codes": ["ACTION_FAMILY_NOT_ADMISSIBLE"],
            "executable_child_produced": False,
            "physical_dispatch": False,
        }
    # Geometry-dependent Host checks are not replayable from this compact
    # artifact set.  Therefore a statically valid admissible team candidate is
    # conservatively treated as executable, making a SYSTEM_STALL claim fail
    # closed rather than trusting an opaque dynamic-preflight rejection.
    return {
        "evaluated": True,
        "passed": True,
        "skipped_reason": None,
        "failure_codes": [],
        "executable_child_produced": True,
        "physical_dispatch": False,
    }


def _validate_transaction_outcome(
    value: Mapping[str, Any],
    *,
    label: str,
    errors: list[str],
) -> None:
    outcome = value.get("transition_outcome")
    before = _ordered_strings(value.get("completed_phase_ids_before"))
    after = _ordered_strings(value.get("completed_phase_ids_after"))
    if outcome not in _TRANSACTION_OUTCOMES:
        errors.append(f"{label}_TRANSITION_OUTCOME_INVALID")
    if before is None or after is None:
        errors.append(f"{label}_COMPLETED_PHASE_IDS_INVALID")
        return
    if after[: len(before)] != before:
        errors.append(f"{label}_COMPLETED_PHASE_HISTORY_REGRESSED")
    accepted = value.get("accepted")
    expected_accepted = outcome in (
        "ACCEPT_SETUP",
        "ACCEPT_PROGRESS",
        "TASK_COMPLETE",
    )
    if not isinstance(accepted, bool) or accepted is not expected_accepted:
        errors.append(f"{label}_OUTCOME_ACCEPTANCE_MISMATCH")
    if outcome in {"REJECT", "ACCEPT_SETUP"} and before != after:
        errors.append(f"{label}_OUTCOME_PHASE_DELTA_MISMATCH")
    failure_type = value.get("failure_type")
    if (
        (expected_accepted and failure_type != "none")
        or (not expected_accepted and (not _nonempty(failure_type) or failure_type == "none"))
    ):
        errors.append(f"{label}_OUTCOME_FAILURE_TYPE_MISMATCH")


def _ordered_strings(value: object) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or not all(_nonempty(item) for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(str(item) for item in value)


def _invalid_physical_drop_derivation() -> PayloadDropPhysicalDerivation:
    return PayloadDropPhysicalDerivation(
        payload_drop=False,
        minimum_payload_z_m=0.0,
        floor_contact_sample_indices=(),
        below_height_sample_indices=(),
        final_supporting_robots=(),
        final_support_force_n_by_robot=MappingProxyType(
            {robot_id: 0.0 for robot_id in _ROBOT_IDS}
        ),
    )


def _derive_physical_payload_drop(
    value: Mapping[str, Any],
    *,
    expected_compiled_model_sha256: str,
    expected_scene_realization_sha256: str,
    label: str,
    errors: list[str],
) -> PayloadDropPhysicalDerivation:
    simulator = value.get("simulator")
    if not isinstance(simulator, Mapping) or set(simulator) != _SIMULATOR_FIELDS:
        errors.append(f"{label}_SIMULATOR_INVALID")
    else:
        if str(simulator.get("name", "")).strip().lower() != "mujoco":
            errors.append(f"{label}_NOT_MUJOCO")
        if not _nonempty(simulator.get("version")):
            errors.append(f"{label}_MUJOCO_VERSION_INVALID")
        for key, expected in (
            ("compiled_model_sha256", expected_compiled_model_sha256),
            ("scene_realization_sha256", expected_scene_realization_sha256),
        ):
            if not _sha(simulator.get(key)):
                errors.append(f"{label}_{key.upper()}_INVALID")
            elif simulator.get(key) != expected:
                errors.append(f"{label}_{key.upper()}_MISMATCH")
    predicate = value.get("drop_predicate")
    expected_predicate: Mapping[str, object] = {
        "safety_profile": FROZEN_PAYLOAD_DROP_SAFETY_PROFILE,
        "minimum_payload_z_m": FROZEN_TASK_MIN_PAYLOAD_Z_M,
        "height_margin_m": PAYLOAD_DROP_HEIGHT_MARGIN_M,
        "drop_z_threshold_m": PAYLOAD_DROP_Z_THRESHOLD_M,
        "contact_distance_tolerance_m": PHYSICAL_CONTACT_DISTANCE_TOLERANCE_M,
    }
    if not isinstance(predicate, Mapping) or set(predicate) != _PHYSICAL_DROP_PREDICATE_FIELDS:
        errors.append(f"{label}_DROP_PREDICATE_INVALID")
    else:
        for key, expected in expected_predicate.items():
            actual = predicate.get(key)
            matches = actual == expected if isinstance(expected, str) else _close(actual, expected)
            if not matches:
                errors.append(f"{label}_DROP_PREDICATE_{key.upper()}_MISMATCH")

    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append(f"{label}_SAMPLES_INVALID")
        return _invalid_physical_drop_derivation()
    times: list[float] = []
    active_heights: list[float] = []
    floor_indices: list[int] = []
    below_indices: list[int] = []
    final_supporters: tuple[str, ...] = ()
    final_forces: Mapping[str, float] = MappingProxyType(
        {robot_id: 0.0 for robot_id in _ROBOT_IDS}
    )
    valid_samples = 0
    for index, sample in enumerate(samples):
        prefix = f"{label}_SAMPLE_{index}"
        if not isinstance(sample, Mapping) or set(sample) != _PHYSICAL_SAMPLE_FIELDS:
            errors.append(f"{prefix}_SCHEMA_INVALID")
            continue
        if sample.get("sample_index") != index:
            errors.append(f"{prefix}_INDEX_INVALID")
        time_s = _number(sample.get("time_s"))
        if time_s is None or (times and time_s <= times[-1]):
            errors.append(f"{prefix}_TIME_INVALID")
        else:
            times.append(time_s)
        active = sample.get("payload_active")
        if not isinstance(active, bool):
            errors.append(f"{prefix}_PAYLOAD_ACTIVE_INVALID")
            active = False
        pose = _physical_payload_pose(sample.get("payload_pose"), prefix, errors)
        contacts = _physical_contacts(sample.get("contacts"), prefix, errors)
        support = _validate_physical_support(
            sample.get("support"),
            contacts=contacts,
            label=prefix,
            errors=errors,
        )
        if pose is None or contacts is None or support is None or time_s is None:
            continue
        valid_samples += 1
        height = pose[2]
        if active:
            active_heights.append(height)
            if any(item["active"] and item["other_role"] == "floor" for item in contacts):
                floor_indices.append(index)
            if height <= PAYLOAD_DROP_Z_THRESHOLD_M:
                below_indices.append(index)
        final_supporters, final_forces = support
    if valid_samples != len(samples) or len(times) != len(samples):
        errors.append(f"{label}_INCOMPLETE_VALID_SAMPLE_SET")
    minimum_height = min(active_heights) if active_heights else min(
        (
            float(sample["payload_pose"]["position_xyz_m"][2])
            for sample in samples
            if isinstance(sample, Mapping)
            and isinstance(sample.get("payload_pose"), Mapping)
            and _vector(sample["payload_pose"].get("position_xyz_m"), 3) is not None
        ),
        default=0.0,
    )
    return PayloadDropPhysicalDerivation(
        payload_drop=bool(floor_indices or below_indices),
        minimum_payload_z_m=minimum_height,
        floor_contact_sample_indices=tuple(floor_indices),
        below_height_sample_indices=tuple(below_indices),
        final_supporting_robots=final_supporters,
        final_support_force_n_by_robot=MappingProxyType(dict(final_forces)),
    )


def _physical_payload_pose(
    value: object,
    label: str,
    errors: list[str],
) -> tuple[float, float, float] | None:
    if not isinstance(value, Mapping) or set(value) != _POSE_FIELDS:
        errors.append(f"{label}_PAYLOAD_POSE_INVALID")
        return None
    position = _vector(value.get("position_xyz_m"), 3)
    quaternion = _vector(value.get("quaternion_wxyz"), 4)
    if position is None:
        errors.append(f"{label}_PAYLOAD_POSITION_INVALID")
    if quaternion is None or not math.isclose(
        math.sqrt(sum(item * item for item in quaternion or ())),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-6,
    ):
        errors.append(f"{label}_PAYLOAD_QUATERNION_INVALID")
    if position is None or quaternion is None:
        return None
    return (position[0], position[1], position[2])


def _physical_contacts(
    value: object,
    label: str,
    errors: list[str],
) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        errors.append(f"{label}_CONTACTS_INVALID")
        return None
    result: list[dict[str, Any]] = []
    for index, contact in enumerate(value):
        prefix = f"{label}_CONTACT_{index}"
        if not isinstance(contact, Mapping) or set(contact) != _PHYSICAL_CONTACT_FIELDS:
            errors.append(f"{prefix}_SCHEMA_INVALID")
            continue
        geom1, geom2 = _integer(contact.get("geom1_id")), _integer(contact.get("geom2_id"))
        role1, role2 = contact.get("geom1_role"), contact.get("geom2_role")
        names = {
            key: contact.get(key)
            for key in ("geom1_name", "geom2_name", "body1_name", "body2_name")
        }
        distance = _number(contact.get("distance_m"))
        raw_force = contact.get("normal_force_n")
        force = None if raw_force is None else _number(raw_force)
        valid = True
        if geom1 is None or geom2 is None or min(geom1, geom2) < 0 or geom1 == geom2:
            errors.append(f"{prefix}_GEOM_ID_INVALID")
            valid = False
        if (
            role1 not in _CONTACT_ROLES
            or role2 not in _CONTACT_ROLES
            or (role1 == "payload") == (role2 == "payload")
        ):
            errors.append(f"{prefix}_ROLE_INVALID")
            valid = False
        if any(not _nonempty(item) for item in names.values()):
            errors.append(f"{prefix}_NAME_INVALID")
            valid = False
        if distance is None:
            errors.append(f"{prefix}_DISTANCE_INVALID")
            valid = False
        if raw_force is not None and (
            force is None or force < 0.0 or force > MAX_PHYSICAL_CONTACT_FORCE_N
        ):
            errors.append(f"{prefix}_NORMAL_FORCE_INVALID")
            valid = False
        if not valid:
            continue
        assert distance is not None
        payload_first = role1 == "payload"
        other_role = str(role2 if payload_first else role1)
        other_geom = str(names["geom2_name" if payload_first else "geom1_name"])
        other_body = str(names["body2_name" if payload_first else "body1_name"])
        if not _contact_role_matches_names(other_role, other_geom, other_body):
            errors.append(f"{prefix}_ROLE_NAME_MISMATCH")
            continue
        active = distance <= PHYSICAL_CONTACT_DISTANCE_TOLERANCE_M
        robot_id = _support_robot_from_names(other_geom, other_body)
        if other_role in {"front_robot", "back_robot"}:
            expected_robot = other_role.removesuffix("_robot")
            if robot_id and robot_id != expected_robot:
                errors.append(f"{prefix}_ROBOT_ROLE_NAME_MISMATCH")
                continue
        result.append(
            {
                "active": active,
                "other_role": other_role,
                "support_robot": robot_id,
                "normal_force_n": force,
            }
        )
    if len(result) != len(value):
        return None
    return result


def _contact_role_matches_names(role: str, geom_name: str, body_name: str) -> bool:
    names = (geom_name.lower(), body_name.lower())
    if role == "floor":
        return any(name in {"floor", "manual_floor"} or name.startswith("floor_") for name in names)
    if role == "wall":
        return any(name.startswith("wall_") for name in names)
    if role == "target":
        return any("target" in name for name in names)
    if role in {"front_robot", "back_robot"}:
        prefix = role.removesuffix("robot")
        return any(name.startswith(prefix) for name in names)
    return False


def _support_robot_from_names(geom_name: str, body_name: str) -> str:
    for robot_id in _ROBOT_IDS:
        prefix = f"{robot_id}_"
        if any(
            name.startswith(prefix) and any(marker in name for marker in _UPPER_LIMB_SUPPORT_MARKERS)
            for name in (geom_name, body_name)
        ):
            return robot_id
    return ""


def _validate_physical_support(
    value: object,
    *,
    contacts: list[dict[str, Any]] | None,
    label: str,
    errors: list[str],
) -> tuple[tuple[str, ...], Mapping[str, float]] | None:
    if not isinstance(value, Mapping) or set(value) != _PHYSICAL_SUPPORT_FIELDS:
        errors.append(f"{label}_SUPPORT_SCHEMA_INVALID")
        return None
    if contacts is None:
        return None
    supporting = tuple(
        sorted(
            {
                str(item["support_robot"])
                for item in contacts
                if item["active"] and item["support_robot"]
            }
        )
    )
    forces = {robot_id: 0.0 for robot_id in _ROBOT_IDS}
    force_available = True
    for item in contacts:
        robot_id = str(item["support_robot"])
        if not item["active"] or not robot_id:
            continue
        force = item["normal_force_n"]
        if force is None:
            force_available = False
        else:
            forces[robot_id] += float(force)
    declared_supporting = _ordered_strings(value.get("supporting_robots"))
    declared_forces = value.get("normal_force_n_by_robot")
    if value.get("available") is not force_available:
        errors.append(f"{label}_SUPPORT_AVAILABILITY_MISMATCH")
    if declared_supporting is None or declared_supporting != supporting:
        errors.append(f"{label}_SUPPORTING_ROBOTS_MISMATCH")
    if not isinstance(declared_forces, Mapping) or set(declared_forces) != set(_ROBOT_IDS):
        errors.append(f"{label}_SUPPORT_FORCE_MAP_INVALID")
    else:
        for robot_id in _ROBOT_IDS:
            if not _close(declared_forces.get(robot_id), forces[robot_id]):
                errors.append(f"{label}_{robot_id.upper()}_SUPPORT_FORCE_MISMATCH")
    if not _close(value.get("total_normal_force_n"), math.fsum(forces.values())):
        errors.append(f"{label}_TOTAL_SUPPORT_FORCE_MISMATCH")
    return supporting, MappingProxyType(forces)


def _physical_drop_document(value: PayloadDropPhysicalDerivation) -> dict[str, Any]:
    return {
        "payload_drop": value.payload_drop,
        "minimum_payload_z_m": value.minimum_payload_z_m,
        "floor_contact_sample_indices": list(value.floor_contact_sample_indices),
        "below_height_sample_indices": list(value.below_height_sample_indices),
        "final_supporting_robots": list(value.final_supporting_robots),
        "final_support_force_n_by_robot": dict(value.final_support_force_n_by_robot),
    }


def _vector(value: object, size: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        return None
    numbers = tuple(_number(item) for item in value)
    return None if any(item is None for item in numbers) else tuple(
        float(item) for item in numbers if item is not None
    )


def _load(
    root: str | Path,
    ref: object,
    label: str,
    errors: list[str],
    hashes: dict[str, str],
    loaded: list[BoundJsonArtifact],
    *,
    expected_path: str,
) -> BoundJsonArtifact | None:
    try:
        artifact = load_bound_json_artifact(root, ref, label=label)
    except BoundArtifactError:
        errors.append(f"{label}_MISSING_OR_INVALID")
        return None
    if artifact.relative_path != expected_path:
        errors.append(f"{label}_PATH_INVALID")
    if artifact.relative_path in hashes:
        errors.append("TERMINAL_AUXILIARY_ARTIFACT_REUSED")
    hashes[artifact.relative_path] = artifact.sha256
    loaded.append(artifact)
    return artifact


def _envelope(
    value: Mapping[str, Any],
    *,
    fields: frozenset[str],
    kind: str,
    label: str,
    errors: list[str],
) -> bool:
    valid = True
    if set(value) != fields:
        errors.append(f"{label}_SCHEMA_INVALID")
        valid = False
    if value.get("kind") != kind or value.get("schema_version") != RAW_AUX_EVIDENCE_SCHEMA_VERSION:
        errors.append(f"{label}_IDENTITY_INVALID")
        valid = False
    producer = value.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != _PRODUCER_FIELDS
        or not _nonempty(producer.get("id"))
        or not _sha(producer.get("source_sha256"))
    ):
        errors.append(f"{label}_PRODUCER_INVALID")
        valid = False
    return valid


def _binding(
    value: Mapping[str, Any],
    run_id: str,
    checkpoint_id: str,
    checkpoint_event_id: str,
    label: str,
    errors: list[str],
) -> None:
    if value.get("run_id") != run_id:
        errors.append(f"{label}_RUN_BINDING_MISMATCH")
    if value.get("checkpoint_id") != checkpoint_id:
        errors.append(f"{label}_CHECKPOINT_BINDING_MISMATCH")
    if value.get("checkpoint_event_id") != checkpoint_event_id:
        errors.append(f"{label}_CHECKPOINT_EVENT_BINDING_MISMATCH")


def _strings(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(_nonempty(item) for item in value)
        or len(value) != len(set(value))
    ):
        return ()
    return tuple(str(item) for item in value)


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _close(value: object, expected: object) -> bool:
    left, right = _number(value), _number(expected)
    return (
        left is not None
        and right is not None
        and math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-6)
    )


__all__ = [
    "AUXILIARY_PRODUCER_IDS",
    "AuxiliaryEvidenceDerivation",
    "FROZEN_PAYLOAD_DROP_SAFETY_PROFILE",
    "PAYLOAD_DROP_EVALUATION_KIND",
    "PAYLOAD_DROP_EXECUTION_TRACE_KIND",
    "PAYLOAD_DROP_LEDGER_KIND",
    "PAYLOAD_DROP_LEDGER_SCHEMA_VERSION",
    "PAYLOAD_DROP_LIMIT",
    "PAYLOAD_DROP_POLICY_KIND",
    "PAYLOAD_DROP_PHYSICAL_SAMPLES_KIND",
    "PAYLOAD_DROP_TRANSACTION_RECEIPT_KIND",
    "PHYSICAL_TRANSACTION_RECEIPT_KIND",
    "PayloadDropPhysicalDerivation",
    "PhysicalTransactionEvidenceValidation",
    "RAW_AUX_EVIDENCE_SCHEMA_VERSION",
    "SYSTEM_STALL_ACTION_SPACE_KIND",
    "SYSTEM_STALL_ACTION_SPACE_RELPATH",
    "SYSTEM_STALL_CHECKPOINT_KIND",
    "SYSTEM_STALL_CHECKPOINT_RELPATH",
    "SYSTEM_STALL_GENERATION_CYCLES",
    "SYSTEM_STALL_HOST_PREFLIGHT_KIND",
    "SYSTEM_STALL_LEDGER_KIND",
    "SYSTEM_STALL_LEDGER_SCHEMA_VERSION",
    "SYSTEM_STALL_MODEL_OUTPUT_KIND",
    "SYSTEM_STALL_STATIC_VALIDATION_KIND",
    "capability_bound_auxiliary_producer_allowlist",
    "derive_payload_drop_from_physical_samples",
    "recompute_payload_drop_ledger",
    "recompute_system_stall_ledger",
    "validate_physical_transaction_receipt",
]
